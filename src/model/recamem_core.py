"""Canonical RecaMem core runtime factory and safetensors persistence bundle.

Loads `configs/recamem_selected_core.json` produced by the 4-stage tournament
(`scripts/run_core_tournament.py`) and instantiates the verified winning stack:
  - Sequence backbone (`CandidateLM` or `SplitSpecialistSystem` if `D4` won)
  - Lossless UTF-8 `TokenizerAdapter` (`bytes` V=256 or `bpe` V=4096)
  - Calibrated System-1 `DecisionReadout` (`D1`..`D4` with `T_cal`)
  - Safety-guarded 5-way `ActionRouter` (`R1`..`R4` with `tau_abstain` + schema guard)
  - Non-destructive `MemoryArm` (`M0`..`M5`)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file

from src.model.cores.contracts import CandidateLM, DecisionResult, RoutingResult, TokenizerAdapter
from src.model.cores.decision_readouts import SplitSpecialistSystem, build_decision_readout
from src.model.cores.memory_arms import build_memory_arm
from src.model.cores.param_matcher import solve_core_config
from src.model.cores.registry import build_candidate_lm
from src.model.cores.routers import build_router
from src.model.cores.tokenizer import CompactBPETokenizer, UTF8ByteTokenizer, build_tokenizer

DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "configs" / "recamem_selected_core.json"
REQUIRED_TOURNAMENT_STAGES = [0, 1, 2, 3, 4]


def validate_selected_core_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate `recamem_selected_core.json` schema and tournament integrity invariants."""
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported manifest schema_version {manifest.get('schema_version')!r}; expected 1")

    status = str(manifest.get("selection_status", ""))
    if status not in {"verified_winner", "provisional", "provisional_core", "blocked"}:
        raise ValueError(f"Invalid selection_status {status!r}")

    for req_section in ("selected_architecture", "hyperparameters", "calibration", "tournament_provenance"):
        if req_section not in manifest or not isinstance(manifest[req_section], Mapping):
            raise ValueError(f"Manifest missing required mapping section {req_section!r}")

    prov = manifest["tournament_provenance"]
    if status == "verified_winner":
        stages = list(prov.get("stages_completed", []))
        if stages != REQUIRED_TOURNAMENT_STAGES:
            raise ValueError(
                f"Cannot mark selection_status='verified_winner' unless stages_completed == {REQUIRED_TOURNAMENT_STAGES}; "
                f"got {stages}"
            )
        if not bool(prov.get("all_stage_gates_evaluated", False)):
            raise ValueError(
                "Cannot mark selection_status='verified_winner' when all_stage_gates_evaluated is False"
            )
        if int(prov.get("unmeasured_metrics_count", 0)) != 0:
            raise ValueError(
                f"Cannot mark selection_status='verified_winner' with unmeasured_metrics_count={prov.get('unmeasured_metrics_count')}"
            )
        if not prov.get("evidence_ledger_path"):
            raise ValueError("verified_winner manifest must record non-empty evidence_ledger_path")
        if prov.get("target_17m_trained") is False:
            raise ValueError(
                "Cannot mark selection_status='verified_winner' when target_17m_trained is False; "
                "use selection_status='provisional_core' until 17M training completes."
            )

    arch = manifest["selected_architecture"]
    for slot in ("core_id", "tokenizer_id", "objective_id", "gradient_mode", "decision_readout_id", "router_id", "memory_arm_id", "architecture_mode"):
        if not arch.get(slot):
            raise ValueError(f"selected_architecture missing required slot {slot!r}")

    return dict(manifest)


class RecaMemCoreBundle(nn.Module):
    """Unified RecaMem runtime bundle instantiated from `recamem_selected_core.json`."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        *,
        tokenizer: TokenizerAdapter,
        lm_model: CandidateLM,
        decision_readout: nn.Module,
        router: nn.Module,
        memory_arm: nn.Module,
        specialist_system: Optional[SplitSpecialistSystem] = None,
    ) -> None:
        super().__init__()
        self.manifest = validate_selected_core_manifest(manifest)
        self.tokenizer = tokenizer
        self.lm_model = lm_model
        self.decision_readout = decision_readout
        self.router = router
        self.memory_arm = memory_arm
        self.specialist_system = specialist_system

        cal = self.manifest["calibration"]
        self.decision_temperature = float(cal.get("decision_temperature", 1.0))
        self.router_temperature = float(cal.get("router_temperature", 1.0))
        self.router_abstain_threshold = float(cal.get("router_abstain_threshold", 0.45))
        if hasattr(self.router, "set_calibration"):
            self.router.set_calibration(self.router_temperature, self.router_abstain_threshold)

    @property
    def architecture_mode(self) -> str:
        return str(self.manifest["selected_architecture"]["architecture_mode"])

    @torch.no_grad()
    def _encode_tokens(self, text: str, add_eos: bool = False) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_eos=add_eos)
        if not ids:
            ids = [self.tokenizer.sep_id]
        device = self.lm_model.embed.weight.device
        return torch.tensor([ids], dtype=torch.long, device=device)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        use_memory: bool = True,
    ) -> str:
        """Autoregressively decode up to `max_new_tokens` using the core's `O(1)`/`O(W)` step path."""
        self.lm_model.eval()
        effective_prompt = prompt
        if use_memory and hasattr(self.memory_arm, "query"):
            mem_out = self.memory_arm.query(prompt, top_k=2)  # type: ignore[operator]
            if "augmented_prompt" in mem_out:
                effective_prompt = str(mem_out["augmented_prompt"])
            elif mem_out.get("hits"):
                top_fact = mem_out["hits"][0]["text"]
                effective_prompt = f"[Context] {top_fact}\n{prompt}"

        t_ids = self._encode_tokens(effective_prompt, add_eos=False)
        details = self.lm_model(t_ids, return_details=True)
        state = details["state"]
        logits = details["logits"][0, -1, :]

        out_tokens: List[int] = []
        device = t_ids.device
        for _ in range(max(1, int(max_new_tokens))):
            # Mask out reserved control IDs except EOS during free text generation
            masked = logits.clone()
            for ctrl_id in (
                self.tokenizer.pad_id,
                self.tokenizer.opt_id,
                self.tokenizer.sep_id,
                *self.tokenizer.route_token_ids.values(),
            ):
                if 0 <= ctrl_id < masked.numel():
                    masked[ctrl_id] = float("-inf")
            nxt = int(torch.argmax(masked).item())
            if nxt == self.tokenizer.eos_id:
                break
            out_tokens.append(nxt)
            step_res = self.lm_model.step(torch.tensor([nxt], device=device), state=state)
            state = step_res["state"]
            logits = step_res["logits"][0]
        return self.tokenizer.decode(out_tokens)

    @torch.no_grad()
    def chat(self, prompt: str, max_new_tokens: int = 32, use_memory: bool = True) -> Dict[str, Any]:
        route_res = self.route(prompt)
        if route_res.action == "abstain":
            return {
                "reply": "",
                "routed_action": route_res.action,
                "abstained": True,
                "routing": route_res.to_dict(),
            }
        reply = self.generate(prompt, max_new_tokens=max_new_tokens, use_memory=use_memory)
        return {
            "reply": reply,
            "routed_action": route_res.action,
            "abstained": False,
            "routing": route_res.to_dict(),
        }

    @torch.no_grad()
    def decide(
        self,
        query: str,
        option_descriptions: Sequence[str],
        option_names: Optional[Sequence[str]] = None,
        include_null: bool = True,
    ) -> DecisionResult:
        self.eval()
        names = list(option_names) if option_names is not None else [f"option_{i}" for i in range(len(option_descriptions))]
        q_ids = self._encode_tokens(query, add_eos=True)
        opt_ids = [self._encode_tokens(desc, add_eos=True) for desc in option_descriptions]

        if self.specialist_system is not None and self.architecture_mode == "split_specialist":
            return self.specialist_system.score_token_options(
                q_ids,
                opt_ids,
                option_names=names,
                include_null=include_null,
                temperature=self.decision_temperature,
            )

        q_h = self.lm_model(q_ids, return_details=True)["hidden_states"]
        opt_h = [self.lm_model(oid, return_details=True)["hidden_states"] for oid in opt_ids]
        return self.decision_readout.score_options(  # type: ignore[operator]
            q_h,
            opt_h,
            option_names=names,
            include_null=include_null,
            temperature=self.decision_temperature,
        )

    @torch.no_grad()
    def route(
        self,
        query: str,
        retrieval_similarity: Optional[float] = None,
        tool_schema: Optional[Mapping[str, Any]] = None,
        proposed_tool_call: Optional[Mapping[str, Any]] = None,
    ) -> RoutingResult:
        self.eval()
        q_ids = self._encode_tokens(query, add_eos=True)
        encoder = (
            self.specialist_system.specialist_model
            if (self.specialist_system is not None and self.architecture_mode == "split_specialist")
            else self.lm_model
        )
        details = encoder(q_ids, return_details=True)
        return self.router.route(  # type: ignore[operator]
            query,
            hidden_states=details["hidden_states"],
            lm_logits_last=details["logits"][:, -1, :],
            retrieval_similarity=retrieval_similarity,
            tool_schema=tool_schema,
            proposed_tool_call=proposed_tool_call,
        )

    def teach(self, fact_text: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return self.memory_arm.ingest(fact_text, metadata=metadata)  # type: ignore[operator]

    def recall(self, query_text: str, top_k: int = 4) -> Dict[str, Any]:
        return self.memory_arm.query(query_text, top_k=top_k)  # type: ignore[operator]

    def supersede(self, entity_key: str, new_fact_text: Optional[str] = None) -> Dict[str, Any]:
        return self.memory_arm.erase_or_supersede(entity_key, new_text=new_fact_text)  # type: ignore[operator]

    def reset(self) -> None:
        self.lm_model.reset_state()
        if self.specialist_system is not None:
            self.specialist_system.lm_model.reset_state()
            self.specialist_system.specialist_model.reset_state()
        self.memory_arm.reset()  # type: ignore[operator]

    def save_pretrained(self, directory: str | Path) -> Path:
        """Persist `manifest.json`, `tokenizer.json` (if BPE), and deduplicated `model.safetensors`."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Deduplicate shared/tied tensor storages (e.g. tied lm_head.weight == embed.weight)
        raw_sd = self.state_dict()
        clean_sd: Dict[str, torch.Tensor] = {}
        seen_ptrs: Dict[int, str] = {}
        tied_aliases: Dict[str, str] = {}
        for key, tensor in raw_sd.items():
            ptr = tensor.data_ptr()
            if ptr in seen_ptrs:
                tied_aliases[key] = seen_ptrs[ptr]
                continue
            seen_ptrs[ptr] = key
            clean_sd[key] = tensor.detach().cpu().contiguous()

        manifest_copy = dict(self.manifest)
        manifest_copy["_safetensors_tied_aliases"] = tied_aliases

        if isinstance(self.tokenizer, CompactBPETokenizer):
            self.tokenizer.save(out_dir / "tokenizer.json")
            manifest_copy["_saved_tokenizer_file"] = "tokenizer.json"

        fd, tmp_st = tempfile.mkstemp(dir=str(out_dir), prefix=".weights_", suffix=".safetensors")
        os.close(fd)
        try:
            safetensors_save_file(clean_sd, tmp_st)
            os.replace(tmp_st, out_dir / "model.safetensors")
        finally:
            if os.path.exists(tmp_st):
                os.unlink(tmp_st)

        (out_dir / "manifest.json").write_text(json.dumps(manifest_copy, indent=2) + "\n", encoding="utf-8")
        return out_dir

    @classmethod
    def load_pretrained(cls, directory: str | Path) -> "RecaMemCoreBundle":
        in_dir = Path(directory)
        manifest = json.loads((in_dir / "manifest.json").read_text(encoding="utf-8"))
        tok: Optional[TokenizerAdapter] = None
        if manifest.get("_saved_tokenizer_file") and (in_dir / manifest["_saved_tokenizer_file"]).is_file():
            tok = CompactBPETokenizer.load(in_dir / manifest["_saved_tokenizer_file"])

        bundle = build_recamem_core(manifest, tokenizer_override=tok)
        loaded = safetensors_load_file(str(in_dir / "model.safetensors"))
        aliases = manifest.get("_safetensors_tied_aliases", {})
        for alias_key, src_key in aliases.items():
            if src_key in loaded:
                loaded[alias_key] = loaded[src_key]
        bundle.load_state_dict(loaded, strict=False)
        # Re-establish weight tying invariants explicitly
        if bundle.lm_model.tied_embeddings:
            bundle.lm_model.lm_head.weight = bundle.lm_model.embed.weight
        if bundle.specialist_system is not None:
            if bundle.specialist_system.lm_model.tied_embeddings:
                bundle.specialist_system.lm_model.lm_head.weight = bundle.specialist_system.lm_model.embed.weight
            if bundle.specialist_system.specialist_model.tied_embeddings:
                bundle.specialist_system.specialist_model.lm_head.weight = (
                    bundle.specialist_system.specialist_model.embed.weight
                )
        return bundle


def build_recamem_core(
    manifest_or_path: Mapping[str, Any] | str | Path = DEFAULT_MANIFEST_PATH,
    *,
    use_tier: str = "active",  # "active" uses hyperparameters; "target_17m" uses target_17m_hyperparameters if present
    train_texts: Optional[Sequence[str]] = None,
    tokenizer_override: Optional[TokenizerAdapter] = None,
) -> RecaMemCoreBundle:
    """Instantiate a `RecaMemCoreBundle` from `configs/recamem_selected_core.json` or dict."""
    if isinstance(manifest_or_path, (str, Path)):
        mpath = Path(manifest_or_path)
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    else:
        mpath = DEFAULT_MANIFEST_PATH
        manifest = dict(manifest_or_path)

    validated = validate_selected_core_manifest(manifest)
    arch = validated["selected_architecture"]
    hp = (
        validated.get("target_17m_hyperparameters", validated["hyperparameters"])
        if use_tier == "target_17m"
        else validated["hyperparameters"]
    )

    vocab_size = int(hp["vocab_size"])
    tok_id = str(arch["tokenizer_id"])
    if tokenizer_override is not None:
        tok = tokenizer_override
    elif tok_id.lower() in {"bytes", "tok_bytes_256", "utf8_bytes"} or vocab_size == 256:
        tok = UTF8ByteTokenizer()
    else:
        saved_bpe = validated.get("tokenizer_artifact_path")
        if saved_bpe and Path(saved_bpe).is_file():
            tok = CompactBPETokenizer.load(saved_bpe)
        elif mpath.parent and (mpath.parent / "recamem_selected_bpe.json").is_file():
            tok = CompactBPETokenizer.load(mpath.parent / "recamem_selected_bpe.json")
        else:
            tok = build_tokenizer("bpe", vocab_size=vocab_size, train_texts=train_texts or ["RecaMem core default vocabulary."])

    core_id = str(arch["core_id"])
    arch_mode = str(arch["architecture_mode"])
    chunk_size = int(hp.get("chunk_size", 64))

    specialist_sys: Optional[SplitSpecialistSystem] = None
    if arch_mode == "split_specialist":
        total_budget = int(hp.get("target_params", 2_000_000))
        specialist_sys = SplitSpecialistSystem(
            core_id=core_id,
            target_total_params=total_budget,
            vocab_size=tok.vocab_size,
            chunk_size=chunk_size,
        )
        lm_model = specialist_sys.lm_model
        decision_readout: nn.Module = specialist_sys.readout
        router_d = specialist_sys.specialist_model.d_model
    else:
        cfg_dict = dict(hp)
        cfg_dict["core_id"] = core_id
        cfg_dict["vocab_size"] = tok.vocab_size
        lm_model = build_candidate_lm(cfg_dict, chunk_size=chunk_size)
        dec_id = str(arch["decision_readout_id"])
        if dec_id.upper() == "D4":
            dec_id = "D3"
        decision_readout = build_decision_readout(dec_id, d_model=lm_model.d_model)
        router_d = lm_model.d_model

    router = build_router(
        str(arch["router_id"]),
        d_model=router_d,
        route_token_ids=tok.route_token_ids,
    )

    mem_kwargs: Dict[str, Any] = {}
    mem_id = str(arch["memory_arm_id"]).upper()
    if mem_id == "M1":
        mem_kwargs["hash_buckets"] = int(hp.get("engram_hash_buckets", 512))
    elif mem_id == "M3":
        mem_kwargs["injection_mode"] = str(hp.get("memory_injection_mode", "prompt_stuffing"))

    memory_arm = build_memory_arm(mem_id, lm_model, tok, **mem_kwargs)

    return RecaMemCoreBundle(
        validated,
        tokenizer=tok,
        lm_model=lm_model,
        decision_readout=decision_readout,
        router=router,
        memory_arm=memory_arm,
        specialist_system=specialist_sys,
    )
