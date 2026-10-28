"""Tests for Track-5 isolated lifelong memory and adaptation arms (M0..M5)."""

from __future__ import annotations

import pytest
import torch

import src.model.cores.contracts as contracts
import src.model.cores.memory_arms as memory_arms
import src.model.cores.param_matcher as param_matcher
import src.model.cores.registry as registry_mod
import src.model.cores.tokenizer as tokenizer_mod


def _snapshot_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _assert_bit_identical(before: dict[str, torch.Tensor], model: torch.nn.Module) -> None:
    for name, p in model.named_parameters():
        assert torch.equal(before[name], p.detach()), f"Base backbone parameter {name!r} mutated!"


def test_memory_arms_m0_to_m4_keep_base_weights_bit_identical_and_handle_contradictions() -> None:
    torch.manual_seed(31)
    tok = tokenizer_mod.build_tokenizer("bytes", vocab_size=256)
    cfg = param_matcher.solve_core_config("C4", tier_or_budget=250_000, vocab_size=256)
    lm = registry_mod.build_candidate_lm(cfg, chunk_size=8)
    snap = _snapshot_params(lm)

    arms = {
        "M0": memory_arms.build_memory_arm("M0", lm, tok),
        "M1": memory_arms.build_memory_arm("M1", lm, tok, hash_buckets=256),
        "M2": memory_arms.build_memory_arm("M2", lm, tok, surprise_threshold=0.05),
        "M3": memory_arms.build_memory_arm("M3", lm, tok, injection_mode="prompt_stuffing"),
        "M4": memory_arms.build_memory_arm("M4", lm, tok, code_dim=256, top_k_code=32, heads=4),
    }

    for aid, arm in arms.items():
        assert isinstance(arm, contracts.MemoryArm)
        arm.ingest("Sentinel revisit is 10 days", metadata={"entity_key": "sentinel revisit"})
        arm.ingest("Landsat swath is 185 km", metadata={"entity_key": "landsat swath"})
        # Contradiction update for 'sentinel revisit'
        arm.erase_or_supersede("sentinel revisit", "Sentinel revisit is 5 days")

        fp = arm.footprint_bytes()
        assert fp.arm_id == aid
        assert fp.total_bytes >= 0
        assert fp.active_items >= 1

        q_res = arm.query("sentinel revisit", top_k=2)
        if aid in {"M1", "M2", "M3", "M4"}:
            assert q_res["hits"], f"{aid} returned empty hits"
            assert "5 days" in q_res["hits"][0]["text"], f"{aid} failed contradiction update: {q_res['hits']}"

        _assert_bit_identical(snap, lm)
        arm.reset()
        assert arm.footprint_bytes().active_items == 0

    # Verify M1 parameter-cap rejection
    with pytest.raises(ValueError, match="max_param_cap"):
        memory_arms.build_memory_arm("M1", lm, tok, hash_buckets=8192, max_param_cap=1024)


def test_memory_arm_m5_snapshot_guarded_plasticity_never_mutates_base_model() -> None:
    torch.manual_seed(37)
    tok = tokenizer_mod.build_tokenizer("bytes", vocab_size=256)
    cfg = param_matcher.solve_core_config("C1", tier_or_budget=250_000, vocab_size=256)
    base_lm = registry_mod.build_candidate_lm(cfg, chunk_size=8)
    base_snap = _snapshot_params(base_lm)

    m5 = memory_arms.build_memory_arm("M5", base_lm, tok, lr=1e-2, steps_per_ingest=2)
    plastic_before = _snapshot_params(m5.plastic_model)

    m5.ingest("Alpha telemetry code is ZETA-904.")
    # 1. Caller's original inference model MUST remain 100% bit-identical!
    _assert_bit_identical(base_snap, base_lm)
    # 2. Isolated plastic copy MUST have updated weights
    changed = any(
        not torch.equal(plastic_before[n], p.detach())
        for n, p in m5.plastic_model.named_parameters()
    )
    assert changed, "Expected M5 plastic_model parameters to update on ingest()"
    assert m5.footprint_bytes().static_param_bytes > 0

    # 3. reset() restores plastic_model bit-for-bit to clean snapshot
    m5.reset()
    _assert_bit_identical(plastic_before, m5.plastic_model)
