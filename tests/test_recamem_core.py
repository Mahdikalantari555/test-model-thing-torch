"""Tests for `src.model.recamem_core` (`RecaMemCoreBundle`, `build_recamem_core`, safetensors round-trip)."""

from __future__ import annotations

import pytest
import torch

import src.model.cores.param_matcher as param_matcher
import src.model.recamem_core as recamem_core


def _sample_manifest(
    *,
    selection_status: str = "verified_winner",
    stages_completed: list[int] | None = None,
    all_gates: bool = True,
    unmeasured: int = 0,
    architecture_mode: str = "unified",
) -> dict:
    cfg = param_matcher.solve_core_config("C4", tier_or_budget="smoke", vocab_size=512)
    return {
        "schema_version": 1,
        "selection_status": selection_status,
        "selected_architecture": {
            "core_id": "C4",
            "core_slug": "gated_delta_matrix",
            "complexity_class": "constant_matrix",
            "tokenizer_id": "Tok_bpe_4096",
            "objective_id": "O1",
            "gradient_mode": "chunked_bptt",
            "decision_readout_id": "D4" if architecture_mode == "split_specialist" else "D3",
            "router_id": "R4",
            "memory_arm_id": "M3",
            "architecture_mode": architecture_mode,
        },
        "hyperparameters": {
            **cfg.to_dict(),
            "chunk_size": 16,
            "memory_injection_mode": "prompt_stuffing",
        },
        "calibration": {
            "decision_temperature": 1.25,
            "router_temperature": 1.10,
            "router_abstain_threshold": 0.15,
        },
        "tournament_provenance": {
            "stages_completed": [0, 1, 2, 3, 4] if stages_completed is None else stages_completed,
            "all_stage_gates_evaluated": all_gates,
            "unmeasured_metrics_count": unmeasured,
            "dataset_source": "fixture",
            "evidence_ledger_path": "reports/evidence/recamem-core-tournament.json",
        },
    }


def test_recamem_core_rejects_incomplete_or_unmeasured_verified_winner() -> None:
    with pytest.raises(ValueError, match="stages_completed"):
        recamem_core.validate_selected_core_manifest(_sample_manifest(stages_completed=[0, 1, 2]))

    with pytest.raises(ValueError, match="all_stage_gates_evaluated"):
        recamem_core.validate_selected_core_manifest(_sample_manifest(all_gates=False))

    with pytest.raises(ValueError, match="unmeasured_metrics_count"):
        recamem_core.validate_selected_core_manifest(_sample_manifest(unmeasured=2))

    untrained_17m = _sample_manifest(selection_status="verified_winner")
    untrained_17m["tournament_provenance"]["target_17m_trained"] = False
    with pytest.raises(ValueError, match="provisional_core"):
        recamem_core.validate_selected_core_manifest(untrained_17m)


@pytest.mark.parametrize("arch_mode", ["unified", "split_specialist"])
def test_build_recamem_core_end_to_end_and_safetensors_roundtrip(tmp_path, arch_mode: str) -> None:
    torch.manual_seed(41)
    manifest = _sample_manifest(architecture_mode=arch_mode)
    bundle = recamem_core.build_recamem_core(
        manifest,
        train_texts=["Orbital satellite radiometry and gated delta memory." * 10],
    )

    # 1. teach + recall + supersede
    bundle.teach("Sentinel revisit is 10 days", metadata={"entity_key": "sentinel revisit"})
    bundle.supersede("sentinel revisit", "Sentinel revisit is 5 days")
    rec = bundle.recall("sentinel revisit", top_k=1)
    assert "5 days" in rec["hits"][0]["text"]

    # 2. System-1 decide
    dec = bundle.decide(
        "Compute orbital maneuver thruster burn",
        ["Execute delta-v burn for perigee adjustment", "Mask cloud pixels in optical tile"],
        option_names=["maneuver", "cloud_mask"],
    )
    assert dec.selected_option in {"maneuver", "cloud_mask", "null"}

    # 3. 5-way route + safety guard
    rt = bundle.route(
        "Run unpermitted root_wipe tool",
        tool_schema={"allowed_tools": ["calculator"], "tools": {}},
        proposed_tool_call={"name": "root_wipe", "arguments": {}},
    )
    assert rt.action in {"generate", "classify", "retrieve", "tool", "abstain"}
    assert rt.action != "tool"  # unpermitted tool can never pass safety guard!

    # 4. generate + chat
    gen_text = bundle.generate("Satellite:", max_new_tokens=6)
    assert isinstance(gen_text, str)
    chat_res = bundle.chat("Explain radiometry:", max_new_tokens=6)
    assert "reply" in chat_res and "routed_action" in chat_res

    # 5. safetensors save_pretrained / load_pretrained bit-exact weight round-trip
    save_dir = tmp_path / f"saved_{arch_mode}"
    bundle.save_pretrained(save_dir)
    assert (save_dir / "model.safetensors").is_file()
    reloaded = recamem_core.RecaMemCoreBundle.load_pretrained(save_dir)

    for (n1, p1), (n2, p2) in zip(bundle.lm_model.named_parameters(), reloaded.lm_model.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1.detach(), p2.detach())

    bundle.reset()
    assert bundle.memory_arm.footprint_bytes().active_items == 0


def test_canonical_selected_core_manifest_loads_active_and_17m_tiers() -> None:
    from pathlib import Path

    manifest_path = Path(__file__).resolve().parents[1] / "configs" / "recamem_selected_core.json"
    assert manifest_path.is_file(), "configs/recamem_selected_core.json must exist after tournament execution"

    bundle_active = recamem_core.build_recamem_core(manifest_path, use_tier="active")
    prov = bundle_active.manifest["tournament_provenance"]
    if prov.get("target_17m_trained") is False:
        assert bundle_active.manifest["selection_status"] == "provisional_core"
    else:
        assert bundle_active.manifest["selection_status"] == "verified_winner"
    assert prov["stages_completed"] == [0, 1, 2, 3, 4]
    assert prov["unmeasured_metrics_count"] == 0

    bundle_active.teach("Perigee altitude is 510 km", metadata={"entity_key": "perigee altitude"})
    bundle_active.supersede("perigee altitude", "Perigee altitude is 545 km")
    rec = bundle_active.recall("perigee altitude", top_k=1)
    assert "545 km" in rec["hits"][0]["text"]
    assert bundle_active.decide("Orbital adjustment", ["Raise perigee", "Mask cloud"]).selected_option in {
        "option_0",
        "option_1",
        "null",
    }
    assert bundle_active.route("Compute delta-v").action in {"generate", "classify", "retrieve", "tool", "abstain"}

    # Verify target ~17M production parameter budget instantiates within ±2.0% of 17,000,000
    bundle_17m = recamem_core.build_recamem_core(manifest_path, use_tier="target_17m")
    lm_params_17m = sum(p.numel() for p in bundle_17m.lm_model.parameters() if p.requires_grad)
    assert abs(lm_params_17m - 17_000_000) / 17_000_000 <= 0.02

