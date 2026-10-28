"""Tests for Track-1 sequence-mixing core protocols, parameter matching, and C0..C6 cores."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import src.model.cores.attention_cores as attention_cores
import src.model.cores.contracts as contracts
import src.model.cores.diagonal_cores as diagonal_cores
import src.model.cores.matrix_cores as matrix_cores
import src.model.cores.param_matcher as param_matcher
import src.model.cores.registry as registry
import src.model.cores.ssm_core as ssm_core


class _DummyCore(nn.Module):
    core_id = "dummy"
    supports_online_rtrl = False

    def __init__(self, d_model: int = 32, n_layers: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def state_spec(self, batch_size: int = 1, seq_len: int = 1) -> contracts.StateSpec:
        return contracts.StateSpec(
            complexity_class="constant_vector",
            bytes_per_sequence=self.n_layers * self.d_model * 4,
            shape_per_layer=(self.d_model,),
            num_layers=self.n_layers,
        )

    def init_state(self, batch_size: int = 1, device=None, dtype=torch.float32):
        return torch.zeros(self.n_layers, batch_size, self.d_model, device=device, dtype=dtype)

    def reset_state(self) -> None:
        return None

    def forward_sequence(self, x: torch.Tensor, state=None, token_ids=None, layer_hook=None):
        if state is None:
            state = self.init_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        h = self.proj(x)
        return contracts.CoreOutput(
            hidden_states=h,
            state=state,
            state_spec=self.state_spec(batch_size=x.shape[0], seq_len=x.shape[1]),
        )

    def step(self, x_t: torch.Tensor, state=None, token_id=None, layer_hook=None):
        return self.forward_sequence(x_t.unsqueeze(1), state=state)


def test_protocols_and_state_spec_validation() -> None:
    core = _DummyCore(d_model=32, n_layers=2)
    assert isinstance(core, contracts.SequenceCore)
    lm = contracts.CandidateLM(core=core, vocab_size=64, tied_embeddings=True)
    ids = torch.randint(0, 64, (2, 10))
    details = lm(ids, return_details=True)
    assert details["logits"].shape == (2, 10, 64)
    assert details["hidden_states"].shape == (2, 10, 32)
    assert details["state_spec"].complexity_class == "constant_vector"

    with pytest.raises(ValueError, match="complexity_class"):
        contracts.StateSpec(
            complexity_class="quadratic_magic",
            bytes_per_sequence=16,
            shape_per_layer=(4,),
            num_layers=1,
        )

    with pytest.raises(ValueError, match="routed action"):
        contracts.RoutingResult(
            action="invented_action",
            raw_action="generate",
            probabilities={"generate": 1.0},
            confidence=0.9,
            guard_triggered=False,
            guard_reason=None,
            latency_ms=0.2,
        )


@pytest.mark.parametrize("tier", ["micro", "small", "target"])
@pytest.mark.parametrize("vocab_size", [256, 4096])
def test_param_matcher_matches_all_seven_cores_within_two_percent(tier: str, vocab_size: int) -> None:
    solved = param_matcher.solve_all_cores_for_tier(tier, vocab_size=vocab_size, tolerance=0.02)
    assert set(solved.keys()) == set(param_matcher.ALL_CORE_IDS)
    for cid, cfg in solved.items():
        assert cfg.relative_error <= 0.02, f"{cid} exceeded ±2% at {tier}: {cfg.relative_error:.4f}"
        assert cfg.d_model % cfg.n_heads == 0


def test_param_matcher_exact_constructed_count_and_unsatisfiable_error() -> None:
    for cid, cfg in param_matcher.solve_all_cores_for_tier("smoke", vocab_size=256).items():
        lm = registry.build_candidate_lm(cfg)
        actual = sum(p.numel() for p in lm.parameters() if p.requires_grad)
        assert actual == cfg.estimated_params, f"{cid}: actual={actual} != estimated={cfg.estimated_params}"
        assert abs(actual - cfg.target_params) / cfg.target_params <= 0.02

    with pytest.raises(ValueError, match="unsatisfiable"):
        param_matcher.solve_core_config("C4", tier_or_budget=500, vocab_size=4096)


@pytest.mark.parametrize("cid", ["C0", "C1"])
def test_c0_and_c1_diagonal_cores_chunk_and_step_equivalence(cid: str) -> None:
    torch.manual_seed(0)
    core = (
        diagonal_cores.UnstackedRTUCore(d_model=48, n_layers=3, d_ff=64, chunk_size=8)
        if cid == "C0"
        else diagonal_cores.StackedGatedRTUCore(d_model=48, n_layers=3, d_ff=64, chunk_size=8)
    )
    core.eval()
    x = torch.randn(2, 20, 48)
    full = core.forward_sequence(x)

    # Two-chunk continuation
    part1 = core.forward_sequence(x[:, :9, :])
    part2 = core.forward_sequence(x[:, 9:, :], state=part1.state)
    cat_h = torch.cat([part1.hidden_states, part2.hidden_states], dim=1)
    assert torch.allclose(full.hidden_states, cat_h, atol=1e-4)

    # Token-by-token step equivalence
    st = core.init_state(batch_size=2)
    steps = []
    for t in range(x.shape[1]):
        out_t = core.step(x[:, t, :], state=st)
        steps.append(out_t.hidden_states)
        st = out_t.state
    step_h = torch.cat(steps, dim=1)
    assert torch.allclose(full.hidden_states, step_h, atol=1e-4)
    core.reset_state()
    assert core._persistent_state is None


def test_c2_selective_ssm_chunk_continuation_and_causality() -> None:
    torch.manual_seed(1)
    core = ssm_core.SelectiveSSMCore(d_model=48, n_layers=2, d_ff=96, state_dim=8, conv_kernel=4, chunk_size=8)
    core.eval()
    x = torch.randn(2, 18, 48)
    full = core.forward_sequence(x)

    p1 = core.forward_sequence(x[:, :7, :])
    p2 = core.forward_sequence(x[:, 7:, :], state=p1.state)
    assert torch.allclose(full.hidden_states, torch.cat([p1.hidden_states, p2.hidden_states], dim=1), atol=1e-4)

    # Causality check: perturbing future positions [10:] must not change [:10]
    x_mod = x.clone()
    x_mod[:, 10:, :] += 5.0
    full_mod = core.forward_sequence(x_mod)
    assert torch.allclose(full.hidden_states[:, :10, :], full_mod.hidden_states[:, :10, :], atol=1e-5)


@pytest.mark.parametrize("cid", ["C3", "C4"])
def test_c3_and_c4_matrix_cores_wy_chunk_vs_step_and_conv_ablation(cid: str) -> None:
    torch.manual_seed(2)
    core = (
        matrix_cores.RetNetDecayMatrixCore(d_model=48, n_layers=2, n_heads=4, d_ff=96, chunk_size=8)
        if cid == "C3"
        else matrix_cores.GatedDeltaMatrixCore(d_model=48, n_layers=2, n_heads=4, d_ff=96, use_conv=True, chunk_size=8)
    )
    core.eval()
    x = torch.randn(2, 21, 48)
    full = core.forward_sequence(x)
    assert full.state_spec.complexity_class == "constant_matrix"
    assert core.state_spec(seq_len=16).bytes_per_sequence == core.state_spec(seq_len=4096).bytes_per_sequence

    # Two-chunk continuation
    p1 = core.forward_sequence(x[:, :11, :])
    p2 = core.forward_sequence(x[:, 11:, :], state=p1.state)
    assert torch.allclose(full.hidden_states, torch.cat([p1.hidden_states, p2.hidden_states], dim=1), atol=1e-4)

    # Step-by-step O(1) decode equivalence against WY chunkwise scan
    st = core.init_state(batch_size=2)
    steps = []
    for t in range(x.shape[1]):
        out_t = core.step(x[:, t, :], state=st)
        steps.append(out_t.hidden_states)
        st = out_t.state
    assert torch.allclose(full.hidden_states, torch.cat(steps, dim=1), atol=1e-4)

    if cid == "C4":
        c4_no_conv = matrix_cores.GatedDeltaMatrixCore(
            d_model=48, n_layers=2, n_heads=4, d_ff=96, use_conv=False, chunk_size=8
        )
        assert c4_no_conv.layers[0].conv_q is None
        out_nc = c4_no_conv.forward_sequence(x)
        assert out_nc.hidden_states.shape == (2, 21, 48)


def test_c5_needle_san_and_c6_transformer_and_registry_swap() -> None:
    torch.manual_seed(3)
    c5 = attention_cores.NeedleSANCore(d_model=48, n_layers=2, n_heads=4, d_ff=96, window_size=8, sink_tokens=2)
    c6 = attention_cores.SoftmaxTransformerCore(d_model=48, n_layers=2, n_heads=4, d_ff=96)
    x = torch.randn(2, 20, 48)

    o5 = c5.forward_sequence(x)
    o6 = c6.forward_sequence(x)
    assert o5.state_spec.complexity_class == "sliding_window_kv"
    assert o5.state[0]["k"].shape[2] == 10  # bounded strictly to sink_tokens (2) + window_size (8)
    assert o6.state_spec.complexity_class == "linear_kv"
    assert o6.state[0]["k"].shape[2] == 20

    # Verify all 7 registered cores swap cleanly inside CandidateLM with identical output shapes
    ids = torch.randint(0, 128, (2, 16))
    expected_classes = {
        "C0": "constant_vector",
        "C1": "constant_vector",
        "C2": "constant_vector",
        "C3": "constant_matrix",
        "C4": "constant_matrix",
        "C5": "sliding_window_kv",
        "C6": "linear_kv",
    }
    for cid in registry.list_cores():
        cfg = param_matcher.solve_core_config(cid, tier_or_budget=250_000, vocab_size=128)
        lm = registry.build_candidate_lm(cfg, chunk_size=8)
        res = lm(ids, return_details=True)
        assert res["logits"].shape == (2, 16, 128)
        assert res["state_spec"].complexity_class == expected_classes[cid]
