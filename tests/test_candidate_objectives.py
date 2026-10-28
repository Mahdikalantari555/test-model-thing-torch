"""Tests for Track-2 tokenizers, auxiliary objectives (O1..O3), O4 RTRL, and trainer."""

from __future__ import annotations

import copy

import pytest
import torch

import src.model.cores.objectives as objectives_mod
import src.model.cores.param_matcher as param_matcher
import src.model.cores.registry as registry_mod
import src.model.cores.tokenizer as tokenizer_mod
import src.model.tournament.trainer as trainer_mod


def test_tokenizer_lossless_utf8_byte_accounting_and_distinct_controls(tmp_path) -> None:
    byte_tok = tokenizer_mod.build_tokenizer("bytes", vocab_size=256)
    train_docs = [
        "Remote sensing acquires geospatial data without physical contact. " * 20,
        "Gated delta rule updates fast weights with normalized keys and values. " * 20,
    ]
    bpe_tok = tokenizer_mod.build_tokenizer("bpe", vocab_size=512, train_texts=train_docs)

    for tok in (byte_tok, bpe_tok):
        controls = [
            tok.pad_id,
            tok.eos_id,
            tok.opt_id,
            tok.sep_id,
            *tok.route_token_ids.values(),
        ]
        assert len(set(controls)) == 9
        assert all(0 <= c < tok.vocab_size for c in controls)

    sample = "Remote sensing uses multispectral indices (NDVI = 0.82) — café & Δt!"
    expected_bytes = len(sample.encode("utf-8"))

    b_span = byte_tok.encode_span(sample, add_eos=True)
    p_span = bpe_tok.encode_span(sample, add_eos=True)

    assert b_span.utf8_byte_count == expected_bytes
    assert p_span.utf8_byte_count == expected_bytes
    assert b_span.control_token_count == 1
    assert p_span.control_token_count == 1
    assert len(p_span.token_ids) < len(b_span.token_ids)

    assert byte_tok.decode(b_span.token_ids) == sample
    assert bpe_tok.decode(p_span.token_ids) == sample

    held_out = "Unseen palabra Xyzzy-98765 αβγ!"
    assert bpe_tok.decode(bpe_tok.encode(held_out, add_eos=True)) == held_out

    saved = tmp_path / "bpe.json"
    bpe_tok.save(saved)
    reloaded = tokenizer_mod.CompactBPETokenizer.load(saved)
    assert reloaded.encode(sample, add_eos=True) == p_span.token_ids

    with pytest.raises(ValueError, match="vocab_size=256"):
        tokenizer_mod.build_tokenizer("bytes", vocab_size=4096)


def test_objectives_o1_o2_o3_zero_eval_overhead_and_ema_update() -> None:
    torch.manual_seed(4)
    cfg = param_matcher.solve_core_config("C4", tier_or_budget=250_000, vocab_size=128)
    lm = registry_mod.build_candidate_lm(cfg, chunk_size=8)
    base_param_count = sum(p.numel() for p in lm.parameters())

    ids = torch.randint(4, 128, (3, 18))
    ids[:, -1] = 1  # EOS

    o1 = objectives_mod.build_objective("O1", d_model=cfg.d_model)
    o2 = objectives_mod.build_objective("O2", d_model=cfg.d_model)
    o3 = objectives_mod.build_objective("O3", d_model=cfg.d_model)

    # Confirm O1/O2/O3 attach zero parameters onto CandidateLM itself
    assert sum(p.numel() for p in lm.parameters()) == base_param_count
    assert sum(p.numel() for p in o1.parameters()) == 0

    details = lm(ids, return_details=True)
    res1 = o1.compute_loss(
        model=lm,
        input_ids=ids,
        logits=details["logits"],
        hidden_states=details["hidden_states"],
        raw_embeds=details["raw_embeds"],
        pad_id=0,
        eos_id=1,
    )
    assert float(res1["aux_loss"].item()) == 0.0
    assert torch.allclose(res1["ce_loss"], res1["total_loss"])

    res2 = o2.compute_loss(
        model=lm,
        input_ids=ids,
        logits=details["logits"],
        hidden_states=details["hidden_states"],
        raw_embeds=details["raw_embeds"],
        pad_id=0,
        eos_id=1,
    )
    assert float(res2["aux_loss"].item()) > 0.0

    ema_before = o3.ema_target_proj.weight.detach().clone()
    res3 = o3.compute_loss(
        model=lm,
        input_ids=ids,
        logits=details["logits"],
        hidden_states=details["hidden_states"],
        raw_embeds=details["raw_embeds"],
        pad_id=0,
        eos_id=1,
    )
    assert float(res3["aux_loss"].item()) > 0.0
    res3["total_loss"].backward()
    with torch.no_grad():
        o3.target_proj.weight.add_(0.5)
    o3.on_step_end(lm)
    assert not torch.allclose(ema_before, o3.ema_target_proj.weight)

    # At eval(), O2 and O3 drop all auxiliary losses cleanly
    o2.eval()
    o3.eval()
    assert float(
        o2.compute_loss(
            model=lm,
            input_ids=ids,
            logits=details["logits"],
            hidden_states=details["hidden_states"],
            raw_embeds=details["raw_embeds"],
            pad_id=0,
            eos_id=1,
        )["aux_loss"].item()
    ) == 0.0
    assert float(
        o3.compute_loss(
            model=lm,
            input_ids=ids,
            logits=details["logits"],
            hidden_states=details["hidden_states"],
            raw_embeds=details["raw_embeds"],
            pad_id=0,
            eos_id=1,
        )["aux_loss"].item()
    ) == 0.0


def test_trainer_microbatch_gradient_equivalence_rtrl_and_interleaved_sampling(tmp_path) -> None:
    torch.manual_seed(7)
    cfg = param_matcher.solve_core_config("C4", tier_or_budget=250_000, vocab_size=128)
    lm_full = registry_mod.build_candidate_lm(cfg, chunk_size=8)
    lm_micro = copy.deepcopy(lm_full)

    batch_ids = torch.randint(4, 128, (8, 16))
    batch_ids[0, 12:] = 0  # uneven padding to test exact valid-token weighting!

    tr_full = trainer_mod.TournamentTrainer(
        lm_full,
        objective=objectives_mod.PureCEObjective(),
        config=trainer_mod.TrainerConfig(micro_batch_size=8),
        pad_id=0,
        eos_id=1,
    )
    tr_micro = trainer_mod.TournamentTrainer(
        lm_micro,
        objective=objectives_mod.PureCEObjective(),
        config=trainer_mod.TrainerConfig(micro_batch_size=2),
        pad_id=0,
        eos_id=1,
    )

    tr_full.accumulate_lm_batch(batch_ids)
    tr_micro.accumulate_lm_batch(batch_ids)

    max_diff = 0.0
    for (n1, p1), (n2, p2) in zip(lm_full.named_parameters(), lm_micro.named_parameters()):
        assert n1 == n2
        assert p1.grad is not None and p2.grad is not None
        diff = float((p1.grad - p2.grad).abs().max().item())
        max_diff = max(max_diff, diff)
    assert max_diff < 1e-5, f"Micro-batch gradient drift {max_diff} >= 1e-5"
    assert tr_micro.peak_rss_bytes < trainer_mod.MAX_CPU_RSS_BYTES

    # O4 online_rtrl: must reject matrix/attention cores (C3..C6) and run cleanly on C0/C1
    for invalid_cid in ("C3", "C4", "C5", "C6"):
        m_inv = registry_mod.build_candidate_lm(
            param_matcher.solve_core_config(invalid_cid, tier_or_budget=250_000, vocab_size=128)
        )
        with pytest.raises(ValueError, match="online_rtrl"):
            trainer_mod.TournamentTrainer(
                m_inv,
                config=trainer_mod.TrainerConfig(gradient_mode="online_rtrl"),
            )

    for valid_cid in ("C0", "C1"):
        m_rtrl = registry_mod.build_candidate_lm(
            param_matcher.solve_core_config(valid_cid, tier_or_budget=250_000, vocab_size=128)
        )
        tr_rtrl = trainer_mod.TournamentTrainer(
            m_rtrl,
            config=trainer_mod.TrainerConfig(gradient_mode="online_rtrl", rtrl_chunk_len=6),
        )
        res = tr_rtrl.train_step(lm_batch=batch_ids[:2])
        assert res["grad_norm"] > 0.0

    # Seeded interleaved multi-task sampler ratio check (0.80 / 0.15 / 0.05) + checkpoint resume
    sampler = trainer_mod.MultiTaskBatchSampler({"lm": 0.80, "decision": 0.15, "routing": 0.05}, seed=42)
    for _ in range(1000):
        sampler.sample_task()
    assert 750 <= sampler.counts["lm"] <= 850
    assert 115 <= sampler.counts["decision"] <= 185
    assert 25 <= sampler.counts["routing"] <= 75

    ckpt = tr_micro.save_checkpoint(tmp_path / "trainer_state.pt")
    tr_micro.load_checkpoint(ckpt)
    assert ckpt.exists()
