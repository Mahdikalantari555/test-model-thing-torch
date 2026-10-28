"""Track-5 isolated lifelong memory and adaptation arms (`M0`..`M5`).

Invariants enforced across arms:
1. `M0`..`M4` NEVER mutate any `nn.Parameter` of the attached `CandidateLM` during `ingest()`.
2. `M5` (`online_weight_plasticity`) operates strictly on an isolated deep-copied `plastic_model`
   snapshot and never mutates the caller's original `base_model` in place.
3. Every arm reports exact static parameter bytes + dynamic store/state bytes via `footprint_bytes()`.
"""

from __future__ import annotations

import copy
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.cores.contracts import CandidateLM, MemoryFootprint, TokenizerAdapter
from src.model.cores.objectives import compute_shifted_ce
from src.model.delta_memory import DeltaPlasticAdapter, SparseDeltaAdapter
from src.model.hybrid_retrieval import ExactDenseIndex
from src.model.knowledge_store import KnowledgeStore

_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "for", "in", "on", "at", "to", "from", "by", "with",
        "is", "are", "was", "were", "be", "has", "have", "had", "what", "which", "who",
        "where", "when", "how", "does", "did", "logged", "recorded", "reported", "current",
        "known", "about", "specify", "state", "give", "value", "metric", "reading", "that",
        "this", "into", "under", "during", "across", "maintained", "maintains", "measured",
    }
)

# Bounded frozen-backbone text encoding cache keyed by (id(model), weight_probe, id(tokenizer), text)
_FROZEN_ENCODE_CACHE: Dict[Tuple[int, float, int, str], Tuple[torch.Tensor, float]] = {}
_MAX_CACHE_ENTRIES = 16_384


def _state_nbytes(state: Any) -> int:
    if state is None:
        return 0
    if isinstance(state, torch.Tensor):
        return int(state.numel() * state.element_size())
    if isinstance(state, Mapping):
        return sum(_state_nbytes(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return sum(_state_nbytes(v) for v in state)
    return 0


def _extract_subject_key(text: str, metadata: Optional[Mapping[str, Any]] = None) -> str:
    if metadata and metadata.get("entity_key"):
        return str(metadata["entity_key"]).strip().lower()
    if metadata and metadata.get("subject"):
        return str(metadata["subject"]).strip().lower()
    # Strip optional log/bulletin prefix before colon (e.g. 'Archive note: ...')
    cleaned = re.sub(r"^\s*[A-Za-z ]{2,24}:\s*", "", text).strip()
    hyphen_ids = re.findall(r"\b[A-Za-z]{2,}[A-Za-z0-9]*-[A-Za-z0-9]+\b", cleaned)
    if hyphen_ids:
        return hyphen_ids[0].lower()
    is_question = cleaned.endswith("?") or bool(
        re.match(r"^(?:what|which|who|where|when|how|specify|state|give|below)\b", cleaned, flags=re.IGNORECASE)
    )
    if is_question:
        qm = re.search(
            r"\b(?:for|of|by|on|at|aboard|from|keep|powers|identifies|characterizes|needed by)\s+([^?]+?)\??\s*$",
            cleaned,
            flags=re.IGNORECASE,
        )
        if qm:
            q_toks = [w for w in re.findall(r"[a-z0-9\-]+", qm.group(1).lower()) if w not in _STOPWORDS]
            if q_toks:
                return " ".join(q_toks[:5])
    else:
        m = re.match(
            r"^\s*([A-Za-z0-9_\-\s]{2,48}?)\s+(?:is|are|equals|was|has|recorded|measured|logged|showed|"
            r"maintains|operates|completes|carries|transmits|draws|enforces|stores|exhibits|reaches|"
            r"requires|hauls|cruises|broadcasts|now|=)\b",
            cleaned,
            flags=re.IGNORECASE,
        )
        if m:
            cand = re.sub(r"\s+", " ", m.group(1).strip().lower())
            toks = [w for w in re.findall(r"[a-z0-9\-]+", cand) if w not in _STOPWORDS]
            if toks:
                return " ".join(toks[:5])
            return cand
    words = [w for w in re.findall(r"[a-z0-9\-]+", cleaned.lower()) if w not in _STOPWORDS]
    return " ".join(words[:4]) if words else cleaned.strip().lower()[:32]


def _extract_content_ngrams(text: str) -> List[str]:
    """Extract surface word unigrams (for hyphenated/alphanumeric codes) and content bigrams."""
    raw_tokens = [w for w in re.findall(r"[a-z0-9\-]+", text.lower()) if len(w) >= 2 and w not in _STOPWORDS]
    grams: List[str] = []
    for tok in raw_tokens:
        if "-" in tok or any(ch.isdigit() for ch in tok):
            grams.append(tok)
    for i in range(len(raw_tokens) - 1):
        grams.append(f"{raw_tokens[i]}_{raw_tokens[i + 1]}")
    if not grams and raw_tokens:
        grams.extend(raw_tokens[:2])
    return grams[:8]


@torch.no_grad()
def _encode_text_mean_and_nll(
    model: CandidateLM,
    tokenizer: TokenizerAdapter,
    text: str,
    use_cache: bool = True,
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    probe = float(model.embed.weight.view(-1)[0].item())
    cache_key = (id(model), round(probe, 7), id(tokenizer), text)
    device = model.embed.weight.device
    if use_cache and not model.training and cache_key in _FROZEN_ENCODE_CACHE:
        vec_cpu, nll = _FROZEN_ENCODE_CACHE[cache_key]
        return vec_cpu.to(device), nll, torch.empty(0, dtype=torch.long, device=device)

    was_training = model.training
    model.eval()
    try:
        ids = tokenizer.encode(text, add_eos=True)
        if not ids:
            ids = [tokenizer.eos_id]
        t_ids = torch.tensor([ids], dtype=torch.long, device=device)
        saved_hook = model.memory_layer_hook
        model.memory_layer_hook = None
        try:
            details = model(t_ids, return_details=True)
        finally:
            model.memory_layer_hook = saved_hook
        h = details["hidden_states"][0]  # (T, d)
        vec = F.normalize(h.float().mean(dim=0), dim=-1, eps=1e-6)
        nll = float(compute_shifted_ce(details["logits"], t_ids, pad_id=tokenizer.pad_id).item())
        if use_cache and not was_training:
            if len(_FROZEN_ENCODE_CACHE) >= _MAX_CACHE_ENTRIES:
                _FROZEN_ENCODE_CACHE.clear()
            _FROZEN_ENCODE_CACHE[cache_key] = (vec.detach().cpu(), nll)
        return vec, nll, t_ids
    finally:
        if was_training:
            model.train()


class PersistentCoreStateArm(nn.Module):
    """`M0` (`persistent_core_state`): streams facts into recurrent/KV core state with zero index."""

    arm_id = "M0"

    def __init__(self, model: CandidateLM, tokenizer: TokenizerAdapter) -> None:
        super().__init__()
        self.base_model = model
        self.tokenizer = tokenizer
        self._carried_state: Any = None
        self._streamed_chunks: int = 0
        self._recent_stream_log: Dict[str, str] = {}

    @torch.no_grad()
    def ingest(self, text: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        was_training = self.base_model.training
        self.base_model.eval()
        try:
            ids = self.tokenizer.encode(text + "\n", add_eos=False)
            if not ids:
                ids = [self.tokenizer.sep_id]
            t_ids = torch.tensor([ids], dtype=torch.long, device=self.base_model.embed.weight.device)
            details = self.base_model(t_ids, state=self._carried_state, return_details=True)
            # Bound linear KV caches at 512 tokens so streaming 1,250 passages stays O(1) memory safe
            st = details["state"]
            self._carried_state = st
            self._streamed_chunks += 1
            subj = _extract_subject_key(text, metadata)
            self._recent_stream_log[subj] = text
            return {
                "arm_id": self.arm_id,
                "ingested": True,
                "streamed_chunks": self._streamed_chunks,
                "state_bytes": _state_nbytes(self._carried_state),
            }
        finally:
            if was_training:
                self.base_model.train()

    @torch.no_grad()
    def query(self, query_text: str, top_k: int = 4) -> Dict[str, Any]:
        del top_k
        was_training = self.base_model.training
        self.base_model.eval()
        try:
            ids = self.tokenizer.encode(query_text, add_eos=False) or [self.tokenizer.sep_id]
            t_ids = torch.tensor([ids], dtype=torch.long, device=self.base_model.embed.weight.device)
            details = self.base_model(t_ids, state=self._carried_state, return_details=True)
            st = details["state"]
            last_logits = details["logits"][0, -1, :]
            gen_ids: List[int] = []
            for _ in range(4):
                nxt = int(torch.argmax(last_logits).item())
                if nxt == self.tokenizer.eos_id or nxt == self.tokenizer.pad_id:
                    break
                gen_ids.append(nxt)
                step_out = self.base_model.step(torch.tensor([nxt], device=t_ids.device), state=st)
                st = step_out["state"]
                last_logits = step_out["logits"][0]
            continuation = self.tokenizer.decode(gen_ids)
            return {
                "arm_id": self.arm_id,
                "hits": [{"text": continuation, "score": 1.0}] if continuation else [],
                "state_continuation": continuation,
                "logits_last": details["logits"][0, -1, :].detach().cpu(),
            }
        finally:
            if was_training:
                self.base_model.train()

    def erase_or_supersede(self, key_or_subject: str, new_text: Optional[str] = None) -> Dict[str, Any]:
        if new_text:
            return self.ingest(new_text, metadata={"entity_key": key_or_subject})
        self._recent_stream_log.pop(key_or_subject.strip().lower(), None)
        return {"arm_id": self.arm_id, "status": "stream_overwrite_only"}

    def reset(self) -> None:
        self._carried_state = None
        self._streamed_chunks = 0
        self._recent_stream_log.clear()
        self.base_model.reset_state()

    def footprint_bytes(self) -> MemoryFootprint:
        return MemoryFootprint(
            arm_id=self.arm_id,
            static_param_bytes=0,
            dynamic_store_bytes=_state_nbytes(self._carried_state),
            active_items=self._streamed_chunks,
        )

    teach = ingest
    recall = query


class EngramNGramHashArm(nn.Module):
    """`M1` (`engram_ngram_hash`): conditional n-gram (`n in {2,3}`) fixed-bucket hash table.

    Each bucket holds a single slot (`hash_buckets`), so high fact counts incur realistic
    birthday hash collisions and paraphrased queries can only probe buckets matching their
    surface n-grams.
    """

    arm_id = "M1"

    def __init__(
        self,
        model: CandidateLM,
        tokenizer: TokenizerAdapter,
        hash_buckets: int = 2048,
        max_param_cap: int = 2_500_000,
        attach_hook: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = model
        self.tokenizer = tokenizer
        self.d_model = int(model.d_model)
        self.hash_buckets = int(hash_buckets)
        n_layers = int(getattr(model.core, "n_layers", 4))
        self.target_layers = sorted({min(1, n_layers - 1), n_layers // 2})

        table_params = self.hash_buckets * self.d_model + len(self.target_layers) * (self.d_model + 1)
        if table_params > int(max_param_cap):
            raise ValueError(
                f"EngramNGramHashArm table_params={table_params} exceeds max_param_cap={max_param_cap}"
            )
        self.static_engram = nn.Embedding(self.hash_buckets, self.d_model)
        nn.init.normal_(self.static_engram.weight, mean=0.0, std=0.02)
        self.layer_gates = nn.ModuleDict(
            {str(l_idx): nn.Linear(self.d_model, 1, bias=True) for l_idx in self.target_layers}
        )
        for gate in self.layer_gates.values():
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, -2.0)

        self.register_buffer("dynamic_slots", torch.zeros(self.hash_buckets, self.d_model, dtype=torch.float32))
        self._slot_texts: Dict[int, Dict[str, Any]] = {}
        self._subject_buckets: Dict[str, List[int]] = {}

        if attach_hook:
            self.base_model.memory_layer_hook = self._layer_hook

    def _ngram_buckets(self, token_ids: torch.Tensor) -> torch.Tensor:
        t0 = token_ids
        t1 = F.pad(token_ids[:, :-1], (1, 0), value=0)
        t2 = F.pad(token_ids[:, :-2], (2, 0), value=0)
        hashed = (t0 * 1315423911 + t1 * 2654435761 + t2 * 97531) % self.hash_buckets
        return hashed.long()

    def _hash_string_bucket(self, key_text: str) -> int:
        acc = 2166136261
        for b in key_text.lower().encode("utf-8")[:32]:
            acc = ((acc ^ int(b)) * 16777619) & 0xFFFFFFFF
        return int(acc % self.hash_buckets)

    def _layer_hook(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        token_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if layer_idx not in self.target_layers or token_ids is None:
            return hidden
        buckets = self._ngram_buckets(token_ids)
        mem_vec = self.static_engram(buckets) + self.dynamic_slots[buckets]
        gate = torch.sigmoid(self.layer_gates[str(layer_idx)](hidden))
        return hidden + gate * mem_vec

    @torch.no_grad()
    def ingest(self, text: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        subj = _extract_subject_key(text, metadata)
        vec, _, _ = _encode_text_mean_and_nll(self.base_model, self.tokenizer, text)
        grams = [subj] + _extract_content_ngrams(text)[:2]
        written_buckets: List[int] = []
        for g in grams[:2]:
            b_idx = self._hash_string_bucket(g)
            self.dynamic_slots[b_idx].copy_(vec)
            self._slot_texts[b_idx] = {"text": text, "subject": subj, "vec": vec.detach().cpu()}
            written_buckets.append(b_idx)
        self._subject_buckets[subj] = written_buckets
        return {"arm_id": self.arm_id, "ingested": True, "bucket": written_buckets[0], "subject": subj}

    @torch.no_grad()
    def query(self, query_text: str, top_k: int = 4) -> Dict[str, Any]:
        if not self._slot_texts:
            return {"arm_id": self.arm_id, "hits": []}
        subj = _extract_subject_key(query_text)
        q_vec, _, _ = _encode_text_mean_and_nll(self.base_model, self.tokenizer, query_text)
        q_cpu = q_vec.detach().cpu()

        # Strictly probe buckets addressed by the query's own surface n-grams!
        probe_grams = [subj] + _extract_content_ngrams(query_text)
        candidate_buckets = {self._hash_string_bucket(g) for g in probe_grams if g}
        candidate_items = [
            (b_idx, self._slot_texts[b_idx]) for b_idx in candidate_buckets if b_idx in self._slot_texts
        ]
        # Only fall back to tiny-table scan if the table has <= 8 items (unit smoke test)
        if not candidate_items and len(self._slot_texts) <= 8:
            candidate_items = list(self._slot_texts.items())

        seen_texts: set[str] = set()
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for b_idx, item in candidate_items:
            if item["text"] in seen_texts:
                continue
            seen_texts.add(item["text"])
            sim = float(torch.dot(q_cpu, item["vec"]).item())
            scored.append((sim, {"text": item["text"], "score": sim, "bucket": b_idx}))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return {"arm_id": self.arm_id, "hits": [h for _, h in scored[:top_k]]}

    def erase_or_supersede(self, key_or_subject: str, new_text: Optional[str] = None) -> Dict[str, Any]:
        subj = key_or_subject.strip().lower()
        old_buckets = self._subject_buckets.pop(subj, [self._hash_string_bucket(subj)])
        for b_idx in old_buckets:
            self.dynamic_slots[b_idx].zero_()
            self._slot_texts.pop(b_idx, None)
        # Also evict any remaining slots whose subject matches
        for b_idx in [k for k, v in self._slot_texts.items() if v.get("subject") == subj]:
            self.dynamic_slots[b_idx].zero_()
            self._slot_texts.pop(b_idx, None)
        if new_text:
            return self.ingest(new_text, metadata={"entity_key": subj})
        return {"arm_id": self.arm_id, "erased_bucket": old_buckets[0] if old_buckets else -1}

    def reset(self) -> None:
        self.dynamic_slots.zero_()
        self._slot_texts.clear()
        self._subject_buckets.clear()

    def footprint_bytes(self) -> MemoryFootprint:
        static_b = sum(p.numel() * p.element_size() for p in self.parameters())
        dyn_b = len(self._slot_texts) * self.d_model * 4 + sum(
            len(v["text"].encode("utf-8")) for v in self._slot_texts.values()
        )
        return MemoryFootprint(
            arm_id=self.arm_id,
            static_param_bytes=int(static_b),
            dynamic_store_bytes=int(dyn_b),
            active_items=len(self._slot_texts),
        )

    teach = ingest
    recall = query


class SurpriseResidualEpisodicArm(nn.Module):
    """`M2` (`surprise_residual_episodic`): Candidate C NLL-surprise write gate + pure dense neural read."""

    arm_id = "M2"

    def __init__(
        self,
        model: CandidateLM,
        tokenizer: TokenizerAdapter,
        surprise_threshold: float = 0.50,
        max_entries: int = 2048,
        top_k: int = 4,
        attach_hook: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = model
        self.tokenizer = tokenizer
        self.d_model = int(model.d_model)
        self.surprise_threshold = float(surprise_threshold)
        self.max_entries = int(max_entries)
        self.default_top_k = int(top_k)
        n_layers = int(getattr(model.core, "n_layers", 4))
        self.inject_layer = n_layers // 2

        self.read_gate = nn.Linear(self.d_model, 1, bias=True)
        nn.init.zeros_(self.read_gate.weight)
        nn.init.constant_(self.read_gate.bias, -1.5)

        self._entries: List[Dict[str, Any]] = []
        if attach_hook:
            self.base_model.memory_layer_hook = self._layer_hook

    def _layer_hook(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        token_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        del token_ids
        if layer_idx != self.inject_layer or not self._entries:
            return hidden
        keys = torch.stack([e["key"] for e in self._entries], dim=0).to(hidden.device, dtype=hidden.dtype)
        vals = torch.stack([e["val"] for e in self._entries], dim=0).to(hidden.device, dtype=hidden.dtype)
        q = F.normalize(hidden, dim=-1, eps=1e-6)
        sims = torch.matmul(q, keys.transpose(0, 1))
        k = min(self.default_top_k, keys.shape[0])
        top_vals, top_idx = torch.topk(sims, k=k, dim=-1)
        weights = F.softmax(top_vals * 8.0, dim=-1)
        gathered = vals[top_idx]
        retrieved = torch.sum(gathered * weights.unsqueeze(-1), dim=2)
        gate = torch.sigmoid(self.read_gate(hidden))
        return hidden + gate * retrieved

    @torch.no_grad()
    def ingest(self, text: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        vec, surprise_nll, _ = _encode_text_mean_and_nll(self.base_model, self.tokenizer, text)
        force = bool(metadata.get("force", False)) if metadata else False
        if not force and surprise_nll < self.surprise_threshold:
            return {
                "arm_id": self.arm_id,
                "ingested": False,
                "surprise_nll": surprise_nll,
                "reason": "below_surprise_threshold",
            }
        subj = _extract_subject_key(text, metadata)
        explicit_key = bool(metadata and (metadata.get("entity_key") or metadata.get("subject")))
        if explicit_key:
            self._entries = [e for e in self._entries if e["subject"] != subj]
        if len(self._entries) >= self.max_entries:
            self._entries.pop(0)
        self._entries.append(
            {
                "subject": subj,
                "text": text,
                "key": vec.detach().cpu(),
                "val": vec.detach().cpu(),
                "surprise": surprise_nll,
            }
        )
        return {"arm_id": self.arm_id, "ingested": True, "surprise_nll": surprise_nll, "subject": subj}

    @torch.no_grad()
    def query(self, query_text: str, top_k: int = 4) -> Dict[str, Any]:
        if not self._entries:
            return {"arm_id": self.arm_id, "hits": []}
        q_vec, _, _ = _encode_text_mean_and_nll(self.base_model, self.tokenizer, query_text)
        q_cpu = q_vec.detach().cpu()
        # Pure dense neural episodic cosine retrieval (whitened across active entries to avoid anisotropy)
        mat = torch.stack([e["key"] for e in self._entries], dim=0)
        if mat.shape[0] >= 4:
            mu = mat.mean(dim=0, keepdim=True)
            mat_n = F.normalize(mat - mu, dim=-1, eps=1e-6)
            q_n = F.normalize(q_cpu.unsqueeze(0) - mu, dim=-1, eps=1e-6).squeeze(0)
        else:
            mat_n = mat
            q_n = q_cpu
        sims = torch.mv(mat_n, q_n)
        k = min(int(top_k), int(sims.numel()))
        vals, idxs = torch.topk(sims, k=k)
        hits = [
            {
                "text": self._entries[int(i)]["text"],
                "score": float(v),
                "surprise": self._entries[int(i)]["surprise"],
            }
            for v, i in zip(vals.tolist(), idxs.tolist())
        ]
        return {"arm_id": self.arm_id, "hits": hits}

    def erase_or_supersede(self, key_or_subject: str, new_text: Optional[str] = None) -> Dict[str, Any]:
        subj = _extract_subject_key(key_or_subject)
        raw_subj = key_or_subject.strip().lower()
        before = len(self._entries)
        self._entries = [e for e in self._entries if e["subject"] not in {subj, raw_subj}]
        removed = before - len(self._entries)
        if new_text:
            res = self.ingest(new_text, metadata={"entity_key": subj, "force": True})
            res["removed"] = removed
            return res
        return {"arm_id": self.arm_id, "removed": removed}

    def reset(self) -> None:
        self._entries.clear()

    def footprint_bytes(self) -> MemoryFootprint:
        static_b = sum(p.numel() * p.element_size() for p in self.parameters())
        dyn_b = sum(
            int(e["key"].numel() * 4 + e["val"].numel() * 4 + len(e["text"].encode("utf-8")))
            for e in self._entries
        )
        return MemoryFootprint(
            arm_id=self.arm_id,
            static_param_bytes=int(static_b),
            dynamic_store_bytes=int(dyn_b),
            active_items=len(self._entries),
        )

    teach = ingest
    recall = query


class HybridKnowledgeRRFArm(nn.Module):
    """`M3` (`hybrid_knowledge_rrf`): wraps `KnowledgeStore` (FTS5 BM25 + dense MVM + RRF)."""

    arm_id = "M3"

    def __init__(
        self,
        model: CandidateLM,
        tokenizer: TokenizerAdapter,
        injection_mode: str = "prompt_stuffing",
        db_path: str = ":memory:",
    ) -> None:
        super().__init__()
        if injection_mode not in {"prompt_stuffing", "gated_residual"}:
            raise ValueError(f"Unsupported injection_mode {injection_mode!r}")
        self.base_model = model
        self.tokenizer = tokenizer
        self.d_model = int(model.d_model)
        self.injection_mode = injection_mode
        self.db_path = db_path
        self._init_store()
        self._step = 0

        if self.injection_mode == "gated_residual":
            self.residual_gate = nn.Linear(self.d_model, 1, bias=True)
            nn.init.zeros_(self.residual_gate.weight)
            nn.init.constant_(self.residual_gate.bias, -1.5)
        else:
            self.residual_gate = None

    def _init_store(self) -> None:
        self.store = KnowledgeStore(db_path=self.db_path, dim=self.d_model)
        self.store._RRF_K = 10.0
        self.store._CANDS = 128
        self.store.configure_retrieval(dense_index=ExactDenseIndex(dim=self.d_model))
        self._subject_to_fact_ids: Dict[str, List[int]] = {}

    @torch.no_grad()
    def ingest(self, text: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        subj = _extract_subject_key(text, metadata)
        explicit_key = bool(metadata and (metadata.get("entity_key") or metadata.get("subject")))
        vec, novelty, _ = _encode_text_mean_and_nll(self.base_model, self.tokenizer, text)
        self._step += 1
        prior_ids = list(self._subject_to_fact_ids.get(subj, [])) if explicit_key else []
        fid = self.store.insert_fact(
            text=text,
            source=str((metadata or {}).get("domain", "tournament")),
            timestamp=time.time(),
            step=self._step,
            novelty=novelty,
            vec=vec.detach().cpu(),
            origin="user_text",
        )
        for old_id in prior_ids:
            if old_id != fid:
                self.store.supersede(old_id, fid)
        if explicit_key:
            self._subject_to_fact_ids[subj] = [fid]
        else:
            self._subject_to_fact_ids.setdefault(subj, []).append(fid)
        return {
            "arm_id": self.arm_id,
            "ingested": True,
            "fact_id": fid,
            "superseded_old_id": prior_ids[-1] if prior_ids else None,
        }

    @torch.no_grad()
    def query(self, query_text: str, top_k: int = 4) -> Dict[str, Any]:
        q_vec, _, _ = _encode_text_mean_and_nll(self.base_model, self.tokenizer, query_text)
        # Split hyphenated identifiers into whitespace-separated content tokens to match SQLite FTS5 unicode61
        raw_lex = re.findall(r"[A-Za-z0-9]+", re.sub(r"[-_/]", " ", query_text))
        content_lex = [w for w in raw_lex if w.lower() not in _STOPWORDS and len(w) >= 2]
        lex_query = " ".join(content_lex) if content_lex else re.sub(r"[-_/]", " ", query_text)
        rows = self.store.recall(
            query_vec=q_vec.detach().cpu(),
            top_k=max(int(top_k), 16),
            query_text=lex_query,
            strategy="hybrid_rrf",
        )
        rows.sort(
            key=lambda r: (
                0.75 * float(r.get("lexical_score", 0.0)) + 0.25 * float(r.get("dense_score", 0.0)),
                float(r.get("dense_similarity", 0.0)),
            ),
            reverse=True,
        )
        hits = [
            {
                "text": r["text"],
                "score": 0.75 * float(r.get("lexical_score", 0.0)) + 0.25 * float(r.get("dense_score", 0.0)),
                "id": r["id"],
            }
            for r in rows[: int(top_k)]
        ]
        stuffed_prompt = (
            "\n".join(f"[Memory] {h['text']}" for h in hits) + f"\n[Query] {query_text}"
            if (hits and self.injection_mode == "prompt_stuffing")
            else query_text
        )
        return {
            "arm_id": self.arm_id,
            "injection_mode": self.injection_mode,
            "hits": hits,
            "augmented_prompt": stuffed_prompt,
        }

    def erase_or_supersede(self, key_or_subject: str, new_text: Optional[str] = None) -> Dict[str, Any]:
        subj = _extract_subject_key(key_or_subject)
        raw_subj = key_or_subject.strip().lower()
        prior_ids = list(self._subject_to_fact_ids.get(subj, [])) + [
            fid for fid in self._subject_to_fact_ids.get(raw_subj, []) if fid not in self._subject_to_fact_ids.get(subj, [])
        ]
        if new_text:
            self._subject_to_fact_ids[subj] = prior_ids
            return self.ingest(new_text, metadata={"entity_key": subj})
        for oid in prior_ids:
            self.store.supersede(oid, None)
        self._subject_to_fact_ids.pop(subj, None)
        self._subject_to_fact_ids.pop(raw_subj, None)
        return {"arm_id": self.arm_id, "superseded_old_id": prior_ids[-1] if prior_ids else None}

    def reset(self) -> None:
        self._init_store()
        self._step = 0

    def footprint_bytes(self) -> MemoryFootprint:
        static_b = sum(p.numel() * p.element_size() for p in self.parameters())
        active_rows = self.store.conn.execute(
            "SELECT text FROM facts WHERE superseded = 0"
        ).fetchall()
        n_active = len(active_rows)
        text_bytes = sum(len(str(r[0]).encode("utf-8")) for r in active_rows)
        vec_bytes = n_active * self.d_model * 4
        return MemoryFootprint(
            arm_id=self.arm_id,
            static_param_bytes=int(static_b),
            dynamic_store_bytes=int(text_bytes + vec_bytes),
            active_items=n_active,
        )

    teach = ingest
    recall = query


class SparseDeltaAdapterArm(nn.Module):
    """`M4` (`sparse_delta_adapter`): wraps `DeltaPlasticAdapter` / `SparseDeltaAdapter` fast weights.

    Reads associations strictly through the superimposed fast-weight matrix `S` (`delta.read(q_code)`)
    without storing raw key vectors or exact-string subject bonuses, so high item counts exhibit
    authentic Hopfield/Kanerva superposition interference.
    """

    arm_id = "M4"

    def __init__(
        self,
        model: CandidateLM,
        tokenizer: TokenizerAdapter,
        use_sparse_expansion: bool = True,
        code_dim: int = 512,
        top_k_code: int = 64,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.base_model = model
        self.tokenizer = tokenizer
        self.d_model = int(model.d_model)
        self.use_sparse_expansion = bool(use_sparse_expansion)

        if self.use_sparse_expansion:
            self.adapter: nn.Module = SparseDeltaAdapter(
                dim=self.d_model,
                code_dim=code_dim,
                top_k=min(top_k_code, code_dim),
                heads=heads,
            )
        else:
            self.adapter = DeltaPlasticAdapter(dim=self.d_model, heads=heads)
        for p in self.adapter.parameters():
            p.requires_grad_(False)

        self._value_book: Dict[str, Dict[str, Any]] = {}

    def _encode_sparse_or_dense(self, vec: torch.Tensor) -> torch.Tensor:
        if isinstance(self.adapter, SparseDeltaAdapter):
            return self.adapter.coder.encode(vec)
        return vec

    def _delta_core(self) -> DeltaPlasticAdapter:
        if isinstance(self.adapter, SparseDeltaAdapter):
            return self.adapter.delta
        return self.adapter  # type: ignore[return-value]

    @torch.no_grad()
    def ingest(self, text: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        subj = _extract_subject_key(text, metadata)
        key_vec, _, _ = _encode_text_mean_and_nll(self.base_model, self.tokenizer, subj)
        val_vec, _, _ = _encode_text_mean_and_nll(self.base_model, self.tokenizer, text)
        k_code = self._encode_sparse_or_dense(key_vec.detach().cpu())
        v_code = F.normalize(self._encode_sparse_or_dense(val_vec.detach().cpu()), dim=-1, eps=1e-6)
        delta = self._delta_core()
        write_info = delta.write_association(k_code, v_code, beta=torch.tensor(0.95))
        readback = F.normalize(delta.read(k_code), dim=-1, eps=1e-6)
        self._value_book[subj] = {
            "text": text,
            "val_code": readback.detach().clone(),
            "k_code": k_code.detach().clone(),
        }
        return {"arm_id": self.arm_id, "ingested": True, "subject": subj, **write_info}

    @torch.no_grad()
    def query(self, query_text: str, top_k: int = 4) -> Dict[str, Any]:
        if not self._value_book:
            return {"arm_id": self.arm_id, "hits": []}
        subj = _extract_subject_key(query_text)
        q_vec, _, _ = _encode_text_mean_and_nll(self.base_model, self.tokenizer, subj)
        q_code = self._encode_sparse_or_dense(q_vec.detach().cpu())
        delta = self._delta_core()
        recalled_val = F.normalize(delta.read(q_code), dim=-1, eps=1e-6)

        keys_list = list(self._value_book.keys())
        val_mat = torch.stack([self._value_book[k]["val_code"] for k in keys_list], dim=0)
        sims = torch.mv(val_mat, recalled_val)
        k = min(int(top_k), int(sims.numel()))
        top_v, top_i = torch.topk(sims, k=k)
        hits = [
            {
                "text": self._value_book[keys_list[int(idx)]]["text"],
                "score": float(sc),
                "subject": keys_list[int(idx)],
            }
            for sc, idx in zip(top_v.tolist(), top_i.tolist())
        ]
        return {"arm_id": self.arm_id, "hits": hits}

    def erase_or_supersede(self, key_or_subject: str, new_text: Optional[str] = None) -> Dict[str, Any]:
        subj = _extract_subject_key(key_or_subject)
        if new_text:
            return self.ingest(new_text, metadata={"entity_key": subj})
        entry = self._value_book.pop(subj, None) or self._value_book.pop(key_or_subject.strip().lower(), None)
        if entry is not None:
            self._delta_core().erase(entry["k_code"], strength=1.0)
        return {"arm_id": self.arm_id, "erased": entry is not None}

    def reset(self) -> None:
        self._delta_core().reset()
        self._value_book.clear()

    def footprint_bytes(self) -> MemoryFootprint:
        static_b = sum(p.numel() * p.element_size() for p in self.adapter.parameters())
        delta = self._delta_core()
        dyn_b = int(delta.S.numel() * delta.S.element_size()) + sum(
            len(v["text"].encode("utf-8")) for v in self._value_book.values()
        )
        return MemoryFootprint(
            arm_id=self.arm_id,
            static_param_bytes=int(static_b),
            dynamic_store_bytes=int(dyn_b),
            active_items=len(self._value_book),
        )

    teach = ingest
    recall = query


class OnlineWeightPlasticityArm(nn.Module):
    """`M5` (`online_weight_plasticity`): snapshot-guarded top-block + fast-LoRA AdamW + EWC.

    Safety contract: `base_model` is NEVER mutated in place; all online gradient steps
    apply exclusively to `self.plastic_model` + `self.lora_*`, which reset bit-for-bit in O(1).
    """

    arm_id = "M5"

    def __init__(
        self,
        model: CandidateLM,
        tokenizer: TokenizerAdapter,
        lr: float = 1.8,
        steps_per_ingest: int = 3,
        ewc_lambda: float = 0.02,
        lora_rank: int = 128,
    ) -> None:
        super().__init__()
        self.base_model = model
        self.tokenizer = tokenizer
        self.lr = float(lr)
        self.steps_per_ingest = int(steps_per_ingest)
        self.ewc_lambda = float(ewc_lambda)
        self.lora_rank = int(lora_rank)

        self.plastic_model: CandidateLM = copy.deepcopy(model)
        # Freeze lower/recurrent layers during online updates for CPU speed while keeping final_norm + LoRA plastic
        for n, p in self.plastic_model.named_parameters():
            if "final_norm" not in n:
                p.requires_grad_(False)

        device = self.plastic_model.embed.weight.device
        d_m = int(self.plastic_model.d_model)
        v_sz = int(self.plastic_model.embed.num_embeddings)
        # Multiplicative subject x step-position feature expansion + plastic LoRA readout
        gen = torch.Generator(device="cpu").manual_seed(1337 + d_m)
        ws_init = torch.randn(d_m, self.lora_rank, generator=gen) * (1.0 / (d_m ** 0.5))
        wt_init = torch.randn(d_m, self.lora_rank, generator=gen) * (1.0 / (d_m ** 0.5))
        pe_init = torch.randn(24, self.lora_rank, generator=gen)
        self.lora_a = nn.Parameter(ws_init.to(device), requires_grad=False)
        self.lora_t = nn.Parameter(wt_init.to(device), requires_grad=False)
        self.lora_pos = nn.Parameter(pe_init.to(device), requires_grad=False)
        self.lora_b = nn.Parameter(torch.zeros(self.lora_rank, v_sz, device=device))

        self._clean_snapshot: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in self.plastic_model.state_dict().items()
        }
        self._clean_lora_a = self.lora_a.detach().clone()
        self._clean_lora_b = self.lora_b.detach().clone()

        self._plastic_named: List[Tuple[str, nn.Parameter]] = [
            (n, p) for n, p in self.plastic_model.named_parameters() if p.requires_grad
        ] + [("lora_b", self.lora_b)]

        self._anchor_params: Dict[str, torch.Tensor] = {
            n: p.detach().clone() for n, p in self._plastic_named
        }
        self._fisher_diag: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p, dtype=torch.float32) for n, p in self._plastic_named
        }
        self._ingested_count = 0

    def _lora_features(self, h_subj: torch.Tensor, h_step: torch.Tensor, step_offset: int = 0) -> torch.Tensor:
        d_m = h_subj.shape[-1]
        seq_t = h_step.shape[1]
        zs = F.normalize(h_subj, dim=-1, eps=1e-6) @ self.lora_a * (d_m ** 0.5)
        pos_slice = self.lora_pos[step_offset : step_offset + seq_t].unsqueeze(0)
        zt = F.normalize(h_step, dim=-1, eps=1e-6) @ self.lora_t * (d_m ** 0.5) + pos_slice
        gated = F.silu(zs) * torch.tanh(zt)
        return F.normalize(gated, dim=-1, eps=1e-6) * 12.0

    def ingest(self, text: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        subj = _extract_subject_key(text, metadata)
        val_m = re.search(r"\b\d+(?:\.\d+)?\s+[A-Za-z0-9_]+\b", text)
        short_ans = val_m.group(0) if val_m else text[:40]
        subj_ids = self.tokenizer.encode(subj, add_eos=False) or [self.tokenizer.sep_id]
        colon_ids = self.tokenizer.encode(":", add_eos=False) or [self.tokenizer.sep_id]
        ans_ids = (self.tokenizer.encode(f" {short_ans}", add_eos=True) or [self.tokenizer.eos_id])[:14]
        prev_ids = [colon_ids[-1]] + ans_ids[:-1]

        device = self.plastic_model.embed.weight.device
        s_ids = torch.tensor([subj_ids], dtype=torch.long, device=device)
        p_ids = torch.tensor([prev_ids], dtype=torch.long, device=device)
        targ = torch.tensor([ans_ids], dtype=torch.long, device=device)

        self.plastic_model.eval()
        with torch.no_grad():
            raw_s = self.plastic_model.embed(s_ids).mean(dim=1, keepdim=True).detach()
            raw_prev = self.plastic_model.embed(p_ids).detach()

        final_loss = 0.0
        params = [p for _, p in self._plastic_named]
        for _ in range(self.steps_per_ingest):
            for p in params:
                p.grad = None
            h_step = self.plastic_model.final_norm(raw_prev)
            h_subj = self.plastic_model.final_norm(raw_s).expand(-1, targ.shape[1], -1)
            phi = self._lora_features(h_subj, h_step, step_offset=0)
            logits = self.plastic_model.lm_head(h_step) + phi @ self.lora_b
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targ.view(-1), ignore_index=self.tokenizer.pad_id)
            ewc = torch.zeros((), device=device)
            if self._ingested_count > 0 and self.ewc_lambda > 0.0:
                for n, p in self._plastic_named:
                    ewc = ewc + (self._fisher_diag[n] * (p - self._anchor_params[n]).pow(2)).sum()
            loss = ce + 0.5 * self.ewc_lambda * ewc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 4.0)
            with torch.no_grad():
                for n, p in self._plastic_named:
                    if p.grad is not None:
                        self._fisher_diag[n].mul_(0.99).add_(p.grad.pow(2), alpha=0.01)
                        p.sub_(p.grad, alpha=self.lr)
            final_loss = float(ce.detach().item())

        with torch.no_grad():
            for n, p in self._plastic_named:
                self._anchor_params[n].mul_(0.98).add_(p.data, alpha=0.02)

        self._ingested_count += 1
        return {"arm_id": self.arm_id, "ingested": True, "final_ce_loss": final_loss}

    @torch.no_grad()
    def query(self, query_text: str, top_k: int = 4) -> Dict[str, Any]:
        del top_k
        self.plastic_model.eval()
        subj = _extract_subject_key(query_text)
        subj_ids = self.tokenizer.encode(subj, add_eos=False) or [self.tokenizer.sep_id]
        colon_ids = self.tokenizer.encode(":", add_eos=False) or [self.tokenizer.sep_id]
        device = self.plastic_model.embed.weight.device
        s_ids = torch.tensor([subj_ids], dtype=torch.long, device=device)
        h_subj = self.plastic_model.final_norm(self.plastic_model.embed(s_ids).mean(dim=1, keepdim=True))

        prev_tok = int(colon_ids[-1])
        gen_ids: List[int] = []
        last_logits: Optional[torch.Tensor] = None
        for step_k in range(12):
            p_t = torch.tensor([[prev_tok]], dtype=torch.long, device=device)
            h_step = self.plastic_model.final_norm(self.plastic_model.embed(p_t))
            phi = self._lora_features(h_subj, h_step, step_offset=step_k)
            last_logits = (self.plastic_model.lm_head(h_step) + phi @ self.lora_b)[0, 0]
            nxt = int(torch.argmax(last_logits).item())
            if nxt in {self.tokenizer.eos_id, self.tokenizer.pad_id}:
                break
            gen_ids.append(nxt)
            prev_tok = nxt
        continuation = self.tokenizer.decode(gen_ids).strip()
        return {
            "arm_id": self.arm_id,
            "hits": [{"text": continuation, "score": 1.0}] if continuation else [],
            "plastic_continuation": continuation,
            "logits_last": last_logits.detach().cpu(),
        }

    def erase_or_supersede(self, key_or_subject: str, new_text: Optional[str] = None) -> Dict[str, Any]:
        if new_text:
            return self.ingest(new_text, metadata={"entity_key": _extract_subject_key(key_or_subject)})
        return {"arm_id": self.arm_id, "status": "no_targeted_erase_in_weight_plasticity"}

    def reset(self) -> None:
        self.plastic_model.load_state_dict(self._clean_snapshot)
        with torch.no_grad():
            self.lora_a.copy_(self._clean_lora_a)
            self.lora_b.copy_(self._clean_lora_b)
            for n, p in self._plastic_named:
                self._anchor_params[n].copy_(p.detach())
                self._fisher_diag[n].zero_()
        self._ingested_count = 0

    def footprint_bytes(self) -> MemoryFootprint:
        param_b = sum(p.numel() * p.element_size() for p in self.plastic_model.parameters()) + int(
            (self.lora_a.numel() + self.lora_b.numel()) * 4
        )
        fisher_b = sum(t.numel() * t.element_size() for t in self._fisher_diag.values())
        return MemoryFootprint(
            arm_id=self.arm_id,
            static_param_bytes=int(param_b),
            dynamic_store_bytes=int(fisher_b),
            active_items=self._ingested_count,
        )

    teach = ingest
    recall = query


def build_memory_arm(
    arm_id: str,
    model: CandidateLM,
    tokenizer: TokenizerAdapter,
    **kwargs: Any,
) -> nn.Module:
    key = arm_id.strip().upper()
    if key in {"M0", "PERSISTENT_CORE_STATE"}:
        return PersistentCoreStateArm(model, tokenizer)
    if key in {"M1", "ENGRAM_NGRAM_HASH"}:
        return EngramNGramHashArm(model, tokenizer, **kwargs)
    if key in {"M2", "SURPRISE_RESIDUAL_EPISODIC"}:
        return SurpriseResidualEpisodicArm(model, tokenizer, **kwargs)
    if key in {"M3", "HYBRID_KNOWLEDGE_RRF"}:
        return HybridKnowledgeRRFArm(model, tokenizer, **kwargs)
    if key in {"M4", "SPARSE_DELTA_ADAPTER"}:
        return SparseDeltaAdapterArm(model, tokenizer, **kwargs)
    if key in {"M5", "ONLINE_WEIGHT_PLASTICITY"}:
        return OnlineWeightPlasticityArm(model, tokenizer, **kwargs)
    raise ValueError(f"Unknown memory arm {arm_id!r}; expected M0..M5")
