#!/usr/bin/env python3
"""Execute the 4-stage RecaMem Core Architecture Tournament (`recamem-core-architecture-tournament`).

Evaluates all 24+ isolated candidate arms across 5 orthogonal tracks:
  - Track 1 (Cores):     C0, C1, C2, C3, C4, C5, C6 (+ optional ~3.0M scale-up on C4, C3, C6)
  - Track 2 (Obj & Tok): O1, O2, O3, O4_online_rtrl, Tok_bytes_256, Tok_bpe_4096 (equal non-emb params)
  - Track 3 (Decisions): D1, D2, D3, D4 (Equal-Budget Split-Specialist)
  - Track 4 (Routers):   R1, R2, R3, R4
  - Track 5 (Memory):    M0, M1, M2, M3, M4, M5

Outputs:
  - `configs/recamem_selected_core.json`
  - `reports/evidence/recamem-core-tournament.json`
  - `reports/recamem-core-selection.md`
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import gc
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.benchmarks import (
    build_equal_budget_tables,
    build_report,
    measure_held_out_bpb,
    measure_int8_decode_speed,
    measure_prefill_ttft,
    measurement,
    write_report,
)
from src.model.continual_bench import (
    evaluate_contradiction_updates,
    evaluate_domain_retention_matrix,
)
from src.model.cores.calibration import (
    TemperatureCalibrator,
    compute_calibration_metrics,
    select_abstention_threshold,
)
from src.model.cores.contracts import ACTION_NAMES, CandidateLM, TokenizerAdapter
from src.model.cores.decision_readouts import (
    ClosedFormRidgeReadout,
    OptionConditionedReadout,
    SplitSpecialistSystem,
    ZeroShotPrototypeReadout,
)
from src.model.cores.memory_arms import build_memory_arm
from src.model.cores.objectives import build_objective
from src.model.cores.param_matcher import (
    ALL_CORE_IDS,
    TIER_BUDGETS,
    MatchedCoreConfig,
    solve_all_cores_for_tier,
    solve_core_config,
    solve_core_config_non_embedding,
)
from src.model.cores.registry import CORE_METADATA, build_candidate_lm
from src.model.cores.routers import build_router, compute_routing_utility
from src.model.cores.tokenizer import CompactBPETokenizer, UTF8ByteTokenizer
from src.model.recamem_core import validate_selected_core_manifest
from src.model.tournament.funnel import (
    evaluate_stage1_gates,
    evaluate_stage2_tracks,
    evaluate_stage3_interference,
    evaluate_stage4_loo_pruning,
)
from src.model.tournament.stats import DEFAULT_LR_GRID, select_best_lr
from src.model.tournament.trainer import TrainerConfig, TournamentTrainer
from src.model.tournament.workloads import (
    WorkloadBundle,
    compute_corpus_unigram_bpb,
    copy_example_to_lm_sequence,
    ensure_tinystories_real_data_dir,
    generate_exact_copy_examples,
    generate_mqar_examples,
    load_tournament_workloads,
    mqar_example_to_lm_sequence,
)

# Per-tokenizer corpus token cache so repeated LM arms do not re-tokenize 4+ MB of prose
_CORPUS_TOKEN_CACHE: Dict[Tuple[int, int, str], Any] = {}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    os.close(fd)
    try:
        Path(tmp_name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".md")
    os.close(fd)
    try:
        Path(tmp_name).write_text(text, encoding="utf-8")
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _macro_f1(preds: Sequence[int], golds: Sequence[int], labels: Sequence[int]) -> float:
    f1s: List[float] = []
    for c in labels:
        tp = sum(1 for p, g in zip(preds, golds) if p == c and g == c)
        fp = sum(1 for p, g in zip(preds, golds) if p == c and g != c)
        fn = sum(1 for p, g in zip(preds, golds) if p != c and g == c)
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 1.0)
    return float(statistics.mean(f1s)) if f1s else 0.0


def _encode_corpus_docs_cached(
    docs_text: Mapping[str, str],
    tokenizer: TokenizerAdapter,
    split_tag: str,
) -> Dict[str, List[int]]:
    key = (id(tokenizer), len(docs_text), split_tag)
    if key in _CORPUS_TOKEN_CACHE:
        return _CORPUS_TOKEN_CACHE[key]
    out = {k: tokenizer.encode(txt, add_eos=True) for k, txt in sorted(docs_text.items())}
    _CORPUS_TOKEN_CACHE[key] = out
    return out


def _compute_token_unigram_bpb(
    train_texts: Mapping[str, str],
    eval_texts: Mapping[str, str],
    tokenizer: TokenizerAdapter,
) -> float:
    """Empirical smoothed token-unigram cross-entropy on `eval_texts` in bits per UTF-8 byte."""
    tr_docs = _encode_corpus_docs_cached(train_texts, tokenizer, "train")
    ev_docs = _encode_corpus_docs_cached(eval_texts, tokenizer, "test_eval")
    counts: Counter[int] = Counter()
    total_tr = 0
    for ids in tr_docs.values():
        counts.update(ids)
        total_tr += len(ids)
    v_sz = max(int(tokenizer.vocab_size), max(counts.keys(), default=0) + 1)
    n_tr = float(max(1, total_tr))
    unif = 1.0 / float(max(1, v_sz))
    total_bits = 0.0
    for ids in ev_docs.values():
        if len(ids) < 2:
            continue
        for tid in ids[1:]:
            p = 0.98 * (counts.get(tid, 0) / n_tr) + 0.02 * unif
            total_bits -= math.log2(max(1e-12, p))
    eval_bytes = max(1, sum(len(t.encode("utf-8")) for t in eval_texts.values()))
    return float(total_bits / eval_bytes)


def _pack_token_batches(
    texts: Sequence[str],
    tokenizer: TokenizerAdapter,
    seq_len: int = 128,
    batch_size: int = 8,
    cache_tag: str = "default",
) -> Tuple[List[torch.Tensor], float]:
    cache_key = (id(tokenizer), len(texts), f"batches_{cache_tag}_{seq_len}_{batch_size}")
    if cache_key in _CORPUS_TOKEN_CACHE:
        return _CORPUS_TOKEN_CACHE[cache_key]

    stream: List[int] = []
    total_bytes = 0
    for txt in texts:
        total_bytes += len(txt.encode("utf-8"))
        stream.extend(tokenizer.encode(txt, add_eos=True))
    avg_bytes_per_tok = total_bytes / float(max(1, len(stream)))
    if len(stream) < seq_len + 2:
        stream = stream * ((seq_len + 2) // max(1, len(stream)) + 2)

    windows: List[List[int]] = []
    stride = max(32, seq_len - 16)
    for start in range(0, max(1, len(stream) - seq_len), stride):
        chunk = stream[start : start + seq_len]
        if len(chunk) < seq_len:
            chunk = chunk + [tokenizer.pad_id] * (seq_len - len(chunk))
        windows.append(chunk)

    rng = random.Random(1337 + seq_len + batch_size)
    rng.shuffle(windows)
    batches: List[torch.Tensor] = []
    for i in range(0, len(windows), batch_size):
        sub = windows[i : i + batch_size]
        batches.append(torch.tensor(sub, dtype=torch.long))
    res = (batches or [torch.tensor([stream[:seq_len]], dtype=torch.long)], avg_bytes_per_tok)
    _CORPUS_TOKEN_CACHE[cache_key] = res
    return res


@torch.no_grad()
def _score_corpus_nll_nats_and_per_doc(
    model: CandidateLM,
    tokenizer: TokenizerAdapter,
    docs_bytes: Mapping[str, bytes],
    max_seq_len: int = 128,
    eval_batch_size: int = 16,
) -> Tuple[float, int, List[float]]:
    """Score corpus NLL (nats) + per-doc negative BPB using batched evaluation windows."""
    model.eval()
    doc_items = sorted(docs_bytes.items())
    doc_nlls = [0.0 for _ in doc_items]
    doc_bytes = [max(1, len(raw_b)) for _, raw_b in doc_items]
    total_toks = 0

    all_windows: List[Tuple[int, List[int]]] = []
    enc_cache_key = (id(tokenizer), len(doc_items), sum(doc_bytes), "bytes_enc")
    if enc_cache_key in _CORPUS_TOKEN_CACHE:
        encoded_list = _CORPUS_TOKEN_CACHE[enc_cache_key]
    else:
        encoded_list = [
            tokenizer.encode(raw_b.decode("utf-8", errors="replace"), add_eos=True)
            for _, raw_b in doc_items
        ]
        _CORPUS_TOKEN_CACHE[enc_cache_key] = encoded_list

    for d_idx, ids in enumerate(encoded_list):
        if len(ids) < 2:
            continue
        for start in range(0, len(ids) - 1, max_seq_len - 1):
            w = ids[start : start + max_seq_len]
            if len(w) >= 2:
                all_windows.append((d_idx, w))

    for b_start in range(0, len(all_windows), eval_batch_size):
        sub = all_windows[b_start : b_start + eval_batch_size]
        max_l = max(len(w) for _, w in sub)
        t_mat = torch.full((len(sub), max_l), tokenizer.pad_id, dtype=torch.long)
        lengths: List[int] = []
        for bi, (_, w) in enumerate(sub):
            t_mat[bi, : len(w)] = torch.tensor(w, dtype=torch.long)
            lengths.append(len(w))
        logits = model(t_mat)
        per_tok = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            t_mat[:, 1:].reshape(-1),
            ignore_index=tokenizer.pad_id,
            reduction="none",
        ).view(len(sub), max_l - 1)
        for bi, (d_idx, _) in enumerate(sub):
            valid_n = lengths[bi] - 1
            nll_row = float(per_tok[bi, :valid_n].sum().item())
            doc_nlls[d_idx] += nll_row
            total_toks += valid_n

    total_nll = float(sum(doc_nlls))
    per_doc_neg_bpb = [
        -(nll / (math.log(2.0) * b_len)) for nll, b_len in zip(doc_nlls, doc_bytes)
    ]
    return total_nll, total_toks, per_doc_neg_bpb


@torch.no_grad()
def _score_corpus_nll_nats(
    model: CandidateLM,
    tokenizer: TokenizerAdapter,
    docs_bytes: Mapping[str, bytes],
    max_seq_len: int = 128,
) -> Tuple[float, int]:
    total_nll, total_toks, _ = _score_corpus_nll_nats_and_per_doc(
        model, tokenizer, docs_bytes, max_seq_len=max_seq_len
    )
    return total_nll, total_toks


@torch.no_grad()
def _evaluate_grammar_probes(
    model: CandidateLM,
    tokenizer: TokenizerAdapter,
    probes: Sequence[Mapping[str, Any]],
    batch_size: int = 64,
) -> float:
    if not probes:
        return 0.0
    model.eval()
    flat_strings: List[str] = []
    probe_spans: List[Tuple[int, List[int]]] = []
    for probe in probes:
        ctx = str(probe["context"])
        g_idx = len(flat_strings)
        flat_strings.append(ctx + str(probe["correct"]))
        b_idxs: List[int] = []
        for inc in probe["incorrect"]:
            b_idxs.append(len(flat_strings))
            flat_strings.append(ctx + str(inc))
        probe_spans.append((g_idx, b_idxs))

    scores = [99.0 for _ in flat_strings]
    for start in range(0, len(flat_strings), batch_size):
        sub = flat_strings[start : start + batch_size]
        enc = [tokenizer.encode(s, add_eos=True) for s in sub]
        max_l = max((len(x) for x in enc), default=2)
        if max_l < 2:
            continue
        t_mat = torch.full((len(sub), max_l), tokenizer.pad_id, dtype=torch.long)
        for bi, ids in enumerate(enc):
            if ids:
                t_mat[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        logits = model(t_mat)
        per_tok = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            t_mat[:, 1:].reshape(-1),
            ignore_index=tokenizer.pad_id,
            reduction="none",
        ).view(len(sub), max_l - 1)
        for bi, (s_txt, ids) in enumerate(zip(sub, enc)):
            if len(ids) >= 2:
                nll = float(per_tok[bi, : len(ids) - 1].sum().item())
                scores[start + bi] = nll / max(1, len(s_txt.encode("utf-8")))

    correct_count = sum(
        1 for g_i, b_is in probe_spans if all(scores[g_i] <= scores[b_i] for b_i in b_is)
    )
    return correct_count / float(len(probes))


@torch.no_grad()
def _eval_mqar_split_batched(
    model: CandidateLM,
    eval_exs: Sequence[Any],
    batch_size: int = 64,
) -> List[float]:
    model.eval()
    eval_pairs = [mqar_example_to_lm_sequence(ex, sep_token=3, eos_token=1) for ex in eval_exs]
    item_scores: List[float] = []
    for start in range(0, len(eval_pairs), batch_size):
        sub = eval_pairs[start : start + batch_size]
        max_l = max(len(s) for s, _ in sub)
        tok_mat = torch.zeros(len(sub), max_l, dtype=torch.long)
        for bi, (seq, _) in enumerate(sub):
            tok_mat[bi, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        logits_sub = model(tok_mat)
        for bi, (seq, ans_pos) in enumerate(sub):
            hits = sum(1 for p in ans_pos if int(logits_sub[bi, p - 1].argmax().item()) == seq[p])
            item_scores.append(hits / float(max(1, len(ans_pos))))
    return item_scores


def _run_mqar_training(
    cfg: MatchedCoreConfig,
    overrides: Mapping[str, Any],
    *,
    mqar_steps: int,
    lr: float,
    seed: int,
    synth_vocab: int = 128,
) -> Tuple[Optional[CandidateLM], bool]:
    _set_seed(seed)
    model = build_candidate_lm(cfg, chunk_size=64, **overrides)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    model.train()
    batch_sz = 16 if (cfg.core_id == "C2" or cfg.estimated_params >= 1_500_000) else 32
    for step in range(mqar_steps):
        if step < int(0.25 * mqar_steps):
            kv_step = 4
        elif step < int(0.60 * mqar_steps):
            kv_step = 8
        else:
            kv_step = 16
        online_exs = generate_mqar_examples(
            batch_sz,
            num_kv_pairs=kv_step,
            allow_overwrite=(step >= int(0.65 * mqar_steps)),
            seed=int(seed) * 131 + 10_000 + step,
        )
        batch_samples = [mqar_example_to_lm_sequence(ex, sep_token=3, eos_token=1) for ex in online_exs]
        max_l = max(len(s) for s, _ in batch_samples)
        tok_mat = torch.zeros(batch_sz, max_l, dtype=torch.long)
        ans_mask = torch.zeros(batch_sz, max_l - 1, dtype=torch.float32)
        for bi, (seq, ans_pos) in enumerate(batch_samples):
            tok_mat[bi, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            for p in ans_pos:
                if 0 < p < max_l:
                    ans_mask[bi, p - 1] = 1.0

        opt.zero_grad(set_to_none=True)
        logits = model(tok_mat)
        per_tok = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, synth_vocab),
            tok_mat[:, 1:].reshape(-1),
            ignore_index=0,
            reduction="none",
        ).view(batch_sz, max_l - 1)
        loss = (per_tok * ans_mask).sum() / ans_mask.sum().clamp(min=1.0)
        if not torch.isfinite(loss):
            return None, True
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model, False


def _train_and_eval_mqar_and_copy(
    core_id: str,
    tier_budget: int,
    workloads: WorkloadBundle,
    *,
    mqar_steps: int,
    copy_steps: int,
    lr: float,
    seed: int,
    window_size_override: Optional[int] = None,
    use_conv: bool = True,
    sweep_mqar_lr: bool = False,
) -> Dict[str, Any]:
    """Train core cleanly on mixed MQAR (4/8/16) + Delayed Copy and evaluate MQAR (4/16/32/64) & Copy."""
    synth_vocab = 128
    cfg = solve_core_config(core_id, tier_or_budget=tier_budget, vocab_size=synth_vocab, use_conv=use_conv)
    overrides: Dict[str, Any] = {}
    if window_size_override is not None and core_id == "C5":
        overrides["window_size"] = window_size_override

    chosen_lr = float(lr)
    if sweep_mqar_lr and mqar_steps >= 60:
        pilot_steps = min(220, max(40, mqar_steps // 2))
        best_val = -1.0
        val_exs = (workloads.mqar_splits[16].get("val") or workloads.mqar_splits[16].get("val_cal") or generate_mqar_examples(64, num_kv_pairs=16, seed=seed + 7777))[:64]
        for cand_lr in (1e-3, 3e-3):
            m_p, div_p = _run_mqar_training(
                cfg, overrides, mqar_steps=pilot_steps, lr=cand_lr, seed=seed, synth_vocab=synth_vocab
            )
            if m_p is not None and not div_p:
                sc_p = float(statistics.mean(_eval_mqar_split_batched(m_p, val_exs, batch_size=64)))
                if sc_p > best_val:
                    best_val = sc_p
                    chosen_lr = cand_lr

    model, diverged = _run_mqar_training(
        cfg, overrides, mqar_steps=mqar_steps, lr=chosen_lr, seed=seed, synth_vocab=synth_vocab
    )
    if model is None or diverged:
        return {
            "diverged": True,
            "mqar_lr": chosen_lr,
            "mqar_4_acc": 0.0,
            "mqar_16_acc": 0.0,
            "mqar_32_acc": 0.0,
            "mqar_64_acc": 0.0,
            "copy_acc": 0.0,
            "per_item_mqar_scores": [0.0] * len(workloads.mqar_splits[16]["eval"]),
        }

    mqar_accs: Dict[int, float] = {}
    per_item_16: List[float] = []
    for kv in (4, 16, 32, 64):
        item_scores = _eval_mqar_split_batched(model, workloads.mqar_splits[kv]["eval"], batch_size=64)
        mqar_accs[kv] = float(statistics.mean(item_scores))
        if kv == 16:
            per_item_16 = item_scores

    # Delayed Exact String Copy benchmark
    _set_seed(seed + 101)
    copy_model = build_candidate_lm(cfg, chunk_size=64, **overrides)
    c_opt = torch.optim.AdamW(copy_model.parameters(), lr=chosen_lr, betas=(0.9, 0.98), weight_decay=0.01)
    copy_model.train()
    batch_sz = 32
    for c_step in range(copy_steps):
        c_exs = generate_exact_copy_examples(batch_sz, seed=int(seed) * 97 + 20_000 + c_step)
        batch_samples = [copy_example_to_lm_sequence(ex) for ex in c_exs]
        max_l = max(len(s) for s, _ in batch_samples)
        tok_mat = torch.zeros(batch_sz, max_l, dtype=torch.long)
        ans_mask = torch.zeros(batch_sz, max_l - 1, dtype=torch.float32)
        for bi, (seq, ans_pos) in enumerate(batch_samples):
            tok_mat[bi, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            for p in ans_pos:
                ans_mask[bi, p - 1] = 1.0
        c_opt.zero_grad(set_to_none=True)
        logits = copy_model(tok_mat)
        per_tok = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, synth_vocab),
            tok_mat[:, 1:].reshape(-1),
            ignore_index=0,
            reduction="none",
        ).view(batch_sz, max_l - 1)
        loss = (per_tok * ans_mask).sum() / ans_mask.sum().clamp(min=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(copy_model.parameters(), 1.0)
        c_opt.step()

    copy_model.eval()
    copy_scores: List[float] = []
    with torch.no_grad():
        eval_c_pairs = [copy_example_to_lm_sequence(ex) for ex in workloads.copy_splits["eval"]]
        for start in range(0, len(eval_c_pairs), 64):
            sub = eval_c_pairs[start : start + 64]
            max_l = max(len(s) for s, _ in sub)
            tok_mat = torch.zeros(len(sub), max_l, dtype=torch.long)
            for bi, (seq, _) in enumerate(sub):
                tok_mat[bi, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            logits_c = copy_model(tok_mat)
            for bi, (seq, ans_pos) in enumerate(sub):
                hits = sum(1 for p in ans_pos if int(logits_c[bi, p - 1].argmax().item()) == seq[p])
                copy_scores.append(hits / float(max(1, len(ans_pos))))

    return {
        "diverged": False,
        "mqar_lr": chosen_lr,
        "mqar_4_acc": mqar_accs[4],
        "mqar_16_acc": mqar_accs[16],
        "mqar_32_acc": mqar_accs[32],
        "mqar_64_acc": mqar_accs[64],
        "copy_acc": float(statistics.mean(copy_scores)),
        "per_item_mqar_scores": per_item_16,
    }


def _train_and_eval_lm_arm(
    core_id: str,
    tier_budget: int,
    tokenizer: TokenizerAdapter,
    tokenizer_id: str,
    objective_id: str,
    workloads: WorkloadBundle,
    *,
    lm_steps: int,
    lr: float,
    seed: int,
    gradient_mode: str = "chunked_bptt",
    use_conv: bool = True,
    measure_speed: bool = False,
    custom_cfg: Optional[MatchedCoreConfig] = None,
    seq_len: int = 128,
    batch_size: int = 8,
    max_utf8_bytes_budget: Optional[int] = None,
    max_wall_clock_s: Optional[float] = None,
    probe_aux_grad: bool = False,
) -> Tuple[CandidateLM, Dict[str, Any]]:
    _set_seed(seed)
    cfg = custom_cfg or solve_core_config(
        core_id, tier_or_budget=tier_budget, vocab_size=tokenizer.vocab_size, use_conv=use_conv
    )
    # Keep per-step batch memory modest on constant-matrix C2 or >1M parameter models
    eff_bs = min(batch_size, 4) if (core_id == "C2" or cfg.estimated_params >= 1_500_000) else batch_size
    model = build_candidate_lm(cfg, chunk_size=32)
    objective = build_objective(objective_id, d_model=cfg.d_model)
    max_steps_cap = lm_steps if max_wall_clock_s is None else max(lm_steps * 6, 4000)
    trainer = TournamentTrainer(
        model,
        objective=objective,
        config=TrainerConfig(
            lr=lr,
            total_steps=max(1, lm_steps),
            micro_batch_size=eff_bs,
            gradient_mode=gradient_mode,
            rtrl_chunk_len=16,
            seed=seed,
        ),
        pad_id=tokenizer.pad_id,
        eos_id=tokenizer.eos_id,
    )

    batches, avg_bytes_per_tok = _pack_token_batches(
        list(workloads.lm_train.values()),
        tokenizer,
        seq_len=seq_len,
        batch_size=eff_bs,
        cache_tag=tokenizer_id,
    )

    aux_grad_norm = 0.0
    if probe_aux_grad and objective_id in {"O2", "O3"}:
        model.train()
        objective.train()
        b0 = batches[0]
        for p in model.parameters():
            p.grad = None
        details0 = model(b0, return_details=True)
        aux_out0 = trainer._call_objective(b0, details0)
        if aux_out0["aux_loss"].requires_grad:
            aux_out0["aux_loss"].backward()
            sq = sum(
                float(p.grad.detach().pow(2).sum().item())
                for n, p in model.named_parameters()
                if p.grad is not None and "embed" not in n
            )
            aux_grad_norm = math.sqrt(sq)
        for p in model.parameters():
            p.grad = None
        for p in objective.parameters():
            p.grad = None

    t_start = time.perf_counter()
    tokens_trained = 0
    utf8_bytes_seen = 0
    steps_completed = 0
    aux_losses: List[float] = []

    for step in range(max_steps_cap):
        if max_wall_clock_s is not None and step > 0 and (time.perf_counter() - t_start) >= max_wall_clock_s:
            break
        if max_utf8_bytes_budget is not None and utf8_bytes_seen >= max_utf8_bytes_budget:
            break
        batch = batches[step % len(batches)]
        step_rec = trainer.train_step(lm_batch=batch)
        aux_losses.append(float(step_rec.get("aux_loss", 0.0)))
        valid_toks = int((batch != tokenizer.pad_id).sum().item())
        tokens_trained += valid_toks
        utf8_bytes_seen += int(round(valid_toks * avg_bytes_per_tok))
        steps_completed += 1
        if max_wall_clock_s is None and max_utf8_bytes_budget is None and steps_completed >= lm_steps:
            break
    elapsed_s = max(1e-6, time.perf_counter() - t_start)

    test_holder: Dict[str, List[float]] = {}

    def _test_scorer(docs_b: Mapping[str, bytes]) -> Tuple[float, int]:
        nll, toks, per_doc = _score_corpus_nll_nats_and_per_doc(model, tokenizer, docs_b, max_seq_len=128)
        test_holder["per_doc_neg_bpb"] = per_doc
        return nll, toks

    bpb_rec = measure_held_out_bpb(
        workloads.lm_train,
        workloads.lm_test_eval,
        _test_scorer,
        vocab_size=tokenizer.vocab_size,
        tokenizer_id=tokenizer_id,
    )
    val_bpb_rec = measure_held_out_bpb(
        workloads.lm_train,
        workloads.lm_val_cal,
        lambda docs_b: _score_corpus_nll_nats(model, tokenizer, docs_b, max_seq_len=128),
        vocab_size=tokenizer.vocab_size,
        tokenizer_id=tokenizer_id,
    )
    held_out_bpb = float(bpb_rec["value"])
    val_bpb = float(val_bpb_rec["value"])
    per_doc_neg_bpb = test_holder.get("per_doc_neg_bpb", [-held_out_bpb])

    ext_256, _ = _score_corpus_nll_nats(
        model, tokenizer, {"e": workloads.extrapolation_docs[512][0][:256].encode("utf-8")}, max_seq_len=256
    )
    bpb_256 = ext_256 / (math.log(2.0) * max(1, len(workloads.extrapolation_docs[512][0][:256].encode("utf-8"))))
    long_doc = workloads.extrapolation_docs[1024][0][:1200].encode("utf-8")
    ext_long, _ = _score_corpus_nll_nats(model, tokenizer, {"l": long_doc}, max_seq_len=64)
    bpb_long = ext_long / (math.log(2.0) * max(1, len(long_doc)))
    extrap_ratio = bpb_long / max(1e-4, bpb_256)

    grammar_acc = _evaluate_grammar_probes(model, tokenizer, workloads.grammar_probes)
    state_spec_4k = model.core.state_spec(batch_size=1, seq_len=4096)

    speed_info: Dict[str, Any] = {}
    if measure_speed:
        def _build_decode(qm: Any):
            def _run() -> None:
                st = qm.init_state(batch_size=1)
                tok_t = torch.tensor([12], dtype=torch.long)
                with torch.no_grad():
                    for _ in range(16):
                        res = qm.step(tok_t, state=st)
                        st = res["state"]
            return _run

        int8_res = measure_int8_decode_speed(
            model,
            _build_decode,
            tokens_per_run=16,
            bytes_per_token=avg_bytes_per_tok,
            runs=3,
            warmup=1,
            variance_threshold=0.50,
            load_threshold=5.0,
        )
        if int8_res.get("status") == "passed":
            speed_info.update(int8_res["value"])
        else:
            t0 = time.perf_counter()
            _build_decode(model)()
            dt = max(1e-6, time.perf_counter() - t0)
            speed_info["int8_tokens_per_second"] = 16.0 / dt
            speed_info["int8_bytes_per_second"] = (16.0 / dt) * avg_bytes_per_tok
            speed_info["int8_note"] = int8_res.get("reason", "fallback_fp32_step")

        ttft_res = measure_prefill_ttft(
            {
                length: (
                    lambda l=length: model(torch.randint(4, min(128, tokenizer.vocab_size), (1, l)))
                )
                for length in (128, 512)
            },
            runs=3,
            warmup=1,
            variance_threshold=0.50,
            load_threshold=5.0,
        )
        speed_info["ttft_ms"] = ttft_res

    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_emb_p = cfg.non_embedding_params
    emb_p = cfg.embedding_params
    tok_per_s = tokens_trained / elapsed_s
    effective_gflops_total = (6.0 * trainable_p * tok_per_s) / 1e9
    effective_gflops_non_emb = (6.0 * non_emb_p * tok_per_s) / 1e9

    record = {
        "arm_id": core_id,
        "core_id": core_id,
        "complexity_class": state_spec_4k.complexity_class,
        "state_bytes_at_4k": state_spec_4k.bytes_per_sequence,
        "trainable_params": trainable_p,
        "non_embedding_params": non_emb_p,
        "embedding_params": emb_p,
        "tokenizer_id": tokenizer_id,
        "vocab_size": tokenizer.vocab_size,
        "objective_id": objective_id,
        "gradient_mode": gradient_mode,
        "val_bpb": val_bpb,
        "held_out_bpb": held_out_bpb,
        "per_doc_neg_bpb": per_doc_neg_bpb,
        "grammar_acc": grammar_acc,
        "extrapolation_bpb_ratio_4k_vs_256": extrap_ratio,
        "steps_completed": steps_completed,
        "tokens_trained": tokens_trained,
        "train_wall_clock_s": elapsed_s,
        "train_tokens_per_sec": tok_per_s,
        "effective_gflops_total": effective_gflops_total,
        "effective_gflops_non_emb": effective_gflops_non_emb,
        "utf8_bytes_seen": utf8_bytes_seen,
        "mean_aux_loss": float(statistics.mean(aux_losses)) if aux_losses else 0.0,
        "aux_grad_norm": aux_grad_norm,
        "peak_rss_bytes": trainer.peak_rss_bytes,
        **speed_info,
    }
    return model, record


@torch.no_grad()
def _batch_encode_decision_texts(
    model_enc: CandidateLM,
    tokenizer: TokenizerAdapter,
    decisions: Mapping[str, Sequence[Mapping[str, Any]]],
    batch_size: int = 64,
) -> Dict[str, torch.Tensor]:
    model_enc.eval()
    uniq_strs: Dict[str, Optional[torch.Tensor]] = {}
    for rows in decisions.values():
        for r in rows:
            uniq_strs[str(r["query"])] = None
            for d in r["option_descriptions"]:
                uniq_strs[str(d)] = None
    keys = list(uniq_strs.keys())
    out: Dict[str, torch.Tensor] = {}
    for start in range(0, len(keys), batch_size):
        sub = keys[start : start + batch_size]
        enc = [tokenizer.encode(s, add_eos=True) or [tokenizer.eos_id] for s in sub]
        max_l = max(len(x) for x in enc)
        pad = torch.full((len(sub), max_l), tokenizer.pad_id, dtype=torch.long)
        lengths: List[int] = []
        for bi, ids in enumerate(enc):
            pad[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            lengths.append(len(ids))
        hs = model_enc(pad, return_details=True)["hidden_states"]
        for bi, s in enumerate(sub):
            # Pool once to (1, 1, d) since _pool_hidden averages over token dim=1
            pooled = hs[bi, : lengths[bi], :].mean(dim=0, keepdim=True).unsqueeze(0).contiguous()
            out[s] = pooled
    return out


def _evaluate_decision_arm(
    readout_id: str,
    lm_model: CandidateLM,
    tokenizer: TokenizerAdapter,
    workloads: WorkloadBundle,
    *,
    tier_budget: int,
    core_id: str,
    train_steps: int = 160,
    seed: int = 1337,
    precomputed_enc: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[nn.Module, TemperatureCalibrator, Dict[str, Any]]:
    """Train/fit decision arm (`D1`..`D4`), calibrate thresholds & `T_cal` on `val_cal`, evaluate on `test_eval`."""
    _set_seed(seed)

    train_rows = workloads.decisions["train"]
    val_rows = workloads.decisions["val_cal"]
    test_in_rows = workloads.decisions["test_in_domain"]
    test_sh_rows = workloads.decisions["test_shifted_unseen"]

    if readout_id in {"D1", "D2", "D3"}:
        enc_map = precomputed_enc or _batch_encode_decision_texts(lm_model, tokenizer, workloads.decisions)
        enc_model = lm_model
    else:
        enc_map = {}
        enc_model = lm_model

    if readout_id == "D1":
        readout: nn.Module = ZeroShotPrototypeReadout(d_model=lm_model.d_model)
        corp_vecs: List[torch.Tensor] = [enc_map[str(r["query"])][0, 0] for r in train_rows]
        for r in train_rows[: min(120, len(train_rows))]:
            for d in r["option_descriptions"]:
                corp_vecs.append(enc_map[str(d)][0, 0])
        readout.fit_corpus_stats(torch.stack(corp_vecs, dim=0))
        val_qs = [enc_map[str(r["query"])] for r in val_rows]
        val_os = [[enc_map[str(d)] for d in r["option_descriptions"]] for r in val_rows]
        val_null = [int(r["gold_index"]) < 0 for r in val_rows]
        readout.calibrate_null_threshold(val_qs, val_os, val_null)
    elif readout_id == "D2":
        readout = ClosedFormRidgeReadout(d_model=lm_model.d_model, reg_lambda=2.0)
        tr_qs = [enc_map[str(r["query"])] for r in train_rows]
        tr_os = [[enc_map[str(d)] for d in r["option_descriptions"]] for r in train_rows]
        tr_gs = [None if int(r["gold_index"]) < 0 else int(r["gold_index"]) for r in train_rows]
        readout.fit_pairwise_ridge(tr_qs, tr_os, tr_gs)
        val_qs = [enc_map[str(r["query"])] for r in val_rows]
        val_os = [[enc_map[str(d)] for d in r["option_descriptions"]] for r in val_rows]
        val_ns = [r["option_names"] for r in val_rows]
        val_null = [int(r["gold_index"]) < 0 for r in val_rows]
        readout.calibrate_null_threshold(val_qs, val_os, val_ns, val_null)
    elif readout_id == "D3":
        readout = OptionConditionedReadout(d_model=lm_model.d_model)
        opt = torch.optim.AdamW(readout.parameters(), lr=3e-3, weight_decay=0.01)
        rng = random.Random(seed)
        readout.train()
        mb_sz = min(16, max(1, len(train_rows)))
        for _ in range(train_steps):
            opt.zero_grad(set_to_none=True)
            sub_rows = rng.sample(train_rows, mb_sz)
            losses: List[torch.Tensor] = []
            for r in sub_rows:
                qh = enc_map[str(r["query"])]
                ohs = [enc_map[str(d)] for d in r["option_descriptions"]]
                logits = readout.forward_logits(qh, ohs, include_null=True)
                target_idx = len(ohs) if int(r["gold_index"]) < 0 else int(r["gold_index"])
                losses.append(F.cross_entropy(logits, torch.tensor([target_idx], dtype=torch.long)))
            torch.stack(losses).mean().backward()
            opt.step()
        readout.eval()
    elif readout_id == "D4":
        split_sys = SplitSpecialistSystem(
            core_id=core_id,
            target_total_params=tier_budget,
            vocab_size=tokenizer.vocab_size,
            chunk_size=32,
        )
        opt = torch.optim.AdamW(
            list(split_sys.specialist_model.parameters()) + list(split_sys.readout.parameters()),
            lr=2.5e-3,
            weight_decay=0.01,
        )
        rng = random.Random(seed)
        split_sys.train()
        mb_sz = min(4, max(1, len(train_rows)))
        d4_steps = max(40, min(train_steps, 120))
        for _ in range(d4_steps):
            opt.zero_grad(set_to_none=True)
            sub_rows = rng.sample(train_rows, mb_sz)
            losses = []
            for r in sub_rows:
                q_ids = torch.tensor([tokenizer.encode(str(r["query"]), add_eos=True) or [tokenizer.eos_id]], dtype=torch.long)
                qh = split_sys.specialist_model(q_ids, return_details=True)["hidden_states"]
                ohs = []
                for d in r["option_descriptions"]:
                    d_ids = torch.tensor([tokenizer.encode(str(d), add_eos=True) or [tokenizer.eos_id]], dtype=torch.long)
                    ohs.append(split_sys.specialist_model(d_ids, return_details=True)["hidden_states"])
                logits = split_sys.readout.forward_logits(qh, ohs, include_null=True)
                target_idx = len(ohs) if int(r["gold_index"]) < 0 else int(r["gold_index"])
                losses.append(F.cross_entropy(logits, torch.tensor([target_idx], dtype=torch.long)))
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(split_sys.parameters(), 1.0)
            opt.step()
        split_sys.eval()
        readout = split_sys
        enc_model = split_sys.specialist_model
        enc_map = _batch_encode_decision_texts(enc_model, tokenizer, workloads.decisions)
    else:
        raise ValueError(f"Unknown readout_id {readout_id!r}")

    def _collect_padded_logits(
        rows: Sequence[Mapping[str, Any]], max_cols: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        logits_rows: List[torch.Tensor] = []
        gold_rows: List[int] = []
        ids_out: List[str] = []
        with torch.no_grad():
            for r in rows:
                qh = enc_map[str(r["query"])]
                ohs = [enc_map[str(d)] for d in r["option_descriptions"]]
                res = readout.score_options(qh, ohs, r["option_names"], include_null=True, temperature=1.0)  # type: ignore[operator]
                raw_l = torch.tensor(res.logits, dtype=torch.float32)
                padded = torch.full((max_cols,), -1e9, dtype=torch.float32)
                padded[: raw_l.numel()] = raw_l
                logits_rows.append(padded)
                g_idx = len(ohs) if int(r["gold_index"]) < 0 else int(r["gold_index"])
                gold_rows.append(g_idx)
                ids_out.append(str(r["id"]))
        return torch.stack(logits_rows, dim=0), torch.tensor(gold_rows, dtype=torch.long), ids_out

    val_logits, val_golds, val_ids = _collect_padded_logits(val_rows)
    cal = TemperatureCalibrator(init_temperature=1.0)
    t_cal = cal.fit(val_logits, val_golds, val_ids=val_ids)

    test_in_logits, test_in_golds, test_in_ids = _collect_padded_logits(test_in_rows)
    test_sh_logits, test_sh_golds, test_sh_ids = _collect_padded_logits(test_sh_rows)
    cal_in = cal.evaluate(test_in_logits, test_in_golds, test_ids=test_in_ids)
    cal_sh = cal.evaluate(test_sh_logits, test_sh_golds, test_ids=test_sh_ids)

    def _slice_f1_and_oos(
        rows: Sequence[Mapping[str, Any]], logits_m: torch.Tensor, golds_t: torch.Tensor
    ) -> Tuple[float, float, float, List[float]]:
        preds = logits_m.argmax(dim=-1).tolist()
        golds = golds_t.tolist()
        labels = sorted(set(golds))
        mf1 = _macro_f1(preds, golds, labels)
        oos_preds = [1 if p == len(r["option_names"]) else 0 for p, r in zip(preds, rows)]
        oos_golds = [1 if int(r["gold_index"]) < 0 else 0 for r in rows]
        oos_f1 = _macro_f1(oos_preds, oos_golds, [0, 1])
        pred_null_rate = sum(oos_preds) / float(max(1, len(oos_preds)))
        correct = [1.0 if p == g else 0.0 for p, g in zip(preds, golds)]
        return mf1, oos_f1, pred_null_rate, correct

    in_f1, in_oos, in_null_rate, in_corr = _slice_f1_and_oos(test_in_rows, test_in_logits, test_in_golds)
    sh_f1, sh_oos, sh_null_rate, sh_corr = _slice_f1_and_oos(test_sh_rows, test_sh_logits, test_sh_golds)

    r0 = test_in_rows[0]
    with torch.no_grad():
        qh0 = enc_map[str(r0["query"])]
        oh0 = [enc_map[str(d)] for d in r0["option_descriptions"]]
        res_a = readout.score_options(qh0, oh0, r0["option_names"], include_null=True, temperature=t_cal)  # type: ignore[operator]
        res_b = readout.score_options(
            qh0, list(reversed(oh0)), list(reversed(r0["option_names"])), include_null=True, temperature=t_cal
        )  # type: ignore[operator]
    perm_invariant = abs(res_a.probabilities[0] - res_b.probabilities[-2]) < 1e-4

    return readout, cal, {
        "arm_id": readout_id,
        "temperature": t_cal,
        "null_threshold": float(getattr(readout, "null_threshold", 0.0)),
        "in_domain_macro_f1": in_f1,
        "shifted_macro_f1": sh_f1,
        "oos_f1": 0.5 * (in_oos + sh_oos),
        "shifted_oos_f1": sh_oos,
        "pred_null_rate_in_domain": in_null_rate,
        "pred_null_rate_shifted": sh_null_rate,
        "nll": 0.5 * (cal_in.nll + cal_sh.nll),
        "brier": 0.5 * (cal_in.brier + cal_sh.brier),
        "adaptive_ece": 0.5 * (cal_in.adaptive_ece + cal_sh.adaptive_ece),
        "aurc": 0.5 * (cal_in.aurc + cal_sh.aurc),
        "shifted_adaptive_ece": cal_sh.adaptive_ece,
        "shifted_aurc": cal_sh.aurc,
        "n_test_in_domain": len(test_in_rows),
        "n_test_shifted_unseen": len(test_sh_rows),
        "permutation_invariant": perm_invariant,
        "per_item_correct": in_corr + sh_corr,
    }


@torch.no_grad()
def _batch_encode_routing_queries(
    lm_model: CandidateLM,
    tokenizer: TokenizerAdapter,
    routing_splits: Mapping[str, Sequence[Mapping[str, Any]]],
    batch_size: int = 64,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    lm_model.eval()
    uniq_q: Dict[str, None] = {}
    for rows in routing_splits.values():
        for r in rows:
            uniq_q[str(r["query"])] = None
    keys = list(uniq_q.keys())
    out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for start in range(0, len(keys), batch_size):
        sub = keys[start : start + batch_size]
        enc = [tokenizer.encode(s, add_eos=True) or [tokenizer.eos_id] for s in sub]
        max_l = max(len(x) for x in enc)
        pad = torch.full((len(sub), max_l), tokenizer.pad_id, dtype=torch.long)
        lengths: List[int] = []
        for bi, ids in enumerate(enc):
            pad[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            lengths.append(len(ids))
        det = lm_model(pad, return_details=True)
        hs = det["hidden_states"]
        lg = det["logits"]
        for bi, s in enumerate(sub):
            ln = lengths[bi]
            # Store valid hidden slice (1, L, d) and final-token logits (1, V)
            out[s] = (hs[bi : bi + 1, :ln, :].contiguous(), lg[bi : bi + 1, ln - 1, :].contiguous())
    return out


def _evaluate_router_arm(
    router_id: str,
    lm_model: CandidateLM,
    tokenizer: TokenizerAdapter,
    workloads: WorkloadBundle,
    *,
    train_steps: int = 160,
    seed: int = 1337,
    precomputed_rt: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    _set_seed(seed)
    router = build_router(
        router_id,
        d_model=lm_model.d_model,
        route_token_ids=tokenizer.route_token_ids,
        abstain_threshold=0.35,
    )

    train_rows = workloads.routing["train"]
    val_rows = workloads.routing["val_cal"]
    test_rows = workloads.routing["test_eval"]

    rt_cache = precomputed_rt or _batch_encode_routing_queries(lm_model, tokenizer, workloads.routing)

    if router_id == "R1":
        with torch.no_grad():
            feats = [rt_cache[str(r["query"])][0][0].mean(dim=0) for r in train_rows]
            labs = [ACTION_NAMES.index(r["gold_action"]) for r in train_rows]
            router.fit_prototypes(torch.stack(feats), torch.tensor(labs, dtype=torch.long))  # type: ignore[operator]
    elif router_id in {"R2", "R4"}:
        opt = torch.optim.AdamW(router.parameters(), lr=3e-3, weight_decay=0.01)
        rng = random.Random(seed)
        router.train()
        mb_sz = min(24, max(1, len(train_rows)))
        for _ in range(train_steps):
            opt.zero_grad(set_to_none=True)
            sub_rows = rng.sample(train_rows, mb_sz)
            losses = []
            for r in sub_rows:
                h = rt_cache[str(r["query"])][0]
                logits = (
                    router.linear(h[0].mean(dim=0, keepdim=True))  # type: ignore[operator]
                    if router_id == "R2"
                    else router.forward_logits(h)  # type: ignore[operator]
                )
                target = torch.tensor([ACTION_NAMES.index(r["gold_action"])], dtype=torch.long)
                losses.append(F.cross_entropy(logits, target))
            torch.stack(losses).mean().backward()
            opt.step()
        router.eval()
    elif router_id == "R3":
        lm_copy = copy.deepcopy(lm_model)
        opt = torch.optim.AdamW([lm_copy.embed.weight], lr=2.5e-3)
        rng = random.Random(seed)
        r3_steps = max(40, min(train_steps // 2, 90))
        mb_sz = min(8, max(1, len(train_rows)))
        for _ in range(r3_steps):
            opt.zero_grad(set_to_none=True)
            sub = rng.sample(train_rows, mb_sz)
            enc = [tokenizer.encode(str(r["query"]), add_eos=True) or [tokenizer.eos_id] for r in sub]
            ml = max(len(x) for x in enc)
            pad = torch.full((len(sub), ml), tokenizer.pad_id, dtype=torch.long)
            lns = [len(x) for x in enc]
            for bi, ids in enumerate(enc):
                pad[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            lg_all = lm_copy(pad)
            last_l = torch.stack([lg_all[bi, lns[bi] - 1, :] for bi in range(len(sub))], dim=0)
            r_logits = router.forward_route_logits(last_l)  # type: ignore[operator]
            targets = torch.tensor([ACTION_NAMES.index(r["gold_action"]) for r in sub], dtype=torch.long)
            F.cross_entropy(r_logits, targets).backward()
            opt.step()
        rt_cache = _batch_encode_routing_queries(lm_copy, tokenizer, {"val": val_rows, "test": test_rows})

    if router_id == "R4":
        with torch.no_grad():
            v_logits = torch.cat([router.forward_logits(rt_cache[str(r["query"])][0]) for r in val_rows], dim=0)  # type: ignore[operator]
            v_golds = torch.tensor([ACTION_NAMES.index(r["gold_action"]) for r in val_rows], dtype=torch.long)
        cal = TemperatureCalibrator(init_temperature=1.0)
        t_cal = cal.fit(v_logits, v_golds, val_ids=[r["id"] for r in val_rows])
        v_probs = cal.calibrate_probs(v_logits)
        tau_abs = select_abstention_threshold(v_probs, v_golds, target_max_risk=0.20, fallback_threshold=0.40)
        router.set_calibration(t_cal, min(0.55, tau_abs))  # type: ignore[operator]
    else:
        t_cal = 1.0

    preds: List[str] = []
    golds: List[str] = []
    latencies: List[float] = []
    probs_mat: List[List[float]] = []
    unblocked_invalid = 0
    invalid_cases = 0

    with torch.no_grad():
        for r in test_rows:
            h_cached, lg_cached = rt_cache[str(r["query"])]
            out = router.route(  # type: ignore[operator]
                r["query"],
                hidden_states=h_cached,
                lm_logits_last=lg_cached,
                retrieval_similarity=r.get("retrieval_similarity"),
                tool_schema=r.get("tool_schema"),
                proposed_tool_call=r.get("proposed_tool_call"),
            )
            preds.append(out.action)
            golds.append(str(r["gold_action"]))
            latencies.append(out.latency_ms)
            probs_mat.append([out.probabilities[a] for a in ACTION_NAMES])
            if r.get("proposed_tool_call") is not None and r["gold_action"] == "abstain":
                invalid_cases += 1
                if out.action == "tool":
                    unblocked_invalid += 1

    util_res = compute_routing_utility(preds, golds)
    per_item_u = [
        compute_routing_utility([p], [g])["normalized_utility"] for p, g in zip(preds, golds)
    ]
    probs_t = torch.tensor(probs_mat, dtype=torch.float32).clamp(min=1e-6)
    golds_t = torch.tensor([ACTION_NAMES.index(g) for g in golds], dtype=torch.long)
    cal_rep = compute_calibration_metrics(torch.log(probs_t), golds_t, temperature=1.0)

    return router, {
        "arm_id": router_id,
        "temperature": t_cal,
        "normalized_utility": util_res["normalized_utility"],
        "accuracy": util_res["accuracy"],
        "false_tool_rate": util_res["false_tool_rate"],
        "unblocked_invalid_tool_rate": unblocked_invalid / float(max(1, invalid_cases)),
        "adaptive_ece": cal_rep.adaptive_ece,
        "aurc": cal_rep.aurc,
        "n_test_eval": len(test_rows),
        "p50_latency_ms": float(statistics.median(latencies)),
        "per_item_utility": per_item_u,
    }


def _render_selection_markdown(
    *,
    manifest: Mapping[str, Any],
    stage0: Mapping[str, Any],
    core_records: Sequence[Mapping[str, Any]],
    scaleup_records: Sequence[Mapping[str, Any]],
    obj_records: Sequence[Mapping[str, Any]],
    tok_records: Sequence[Mapping[str, Any]],
    dec_records: Sequence[Mapping[str, Any]],
    rt_records: Sequence[Mapping[str, Any]],
    mem_records: Sequence[Mapping[str, Any]],
    stage1: Mapping[str, Any],
    stage2: Mapping[str, Any],
    stage3: Mapping[str, Any],
    stage4: Mapping[str, Any],
    equal_tables: Mapping[str, Any],
) -> str:
    del equal_tables
    arch = manifest["selected_architecture"]
    hp = manifest["hyperparameters"]
    hp17 = manifest["target_17m_hyperparameters"]
    cal = manifest["calibration"]
    prov = manifest["tournament_provenance"]
    counts = prov.get("evaluation_counts", {})
    baselines = stage0.get("unigram_baselines", {})

    t17_trained = bool(prov.get("target_17m_trained", False))
    t17_note = (
        f"trained & verified at `{hp17['estimated_params']:,}` params"
        if t17_trained
        else f"architecturally solved ONLY (`{hp17['estimated_params']:,}` params, **untrained** on this 2-vCPU / 3.8 GiB host — carried forward as `{manifest['selection_status']}`)"
    )

    lines: List[str] = [
        "# RecaMem Core Architecture Selection Report",
        "",
        f"- **Selection Status**: `{manifest['selection_status']}` (`target_17m_trained={t17_trained}`)",
        f"- **Winning Stack**: `{arch['core_id']}` + `{arch['tokenizer_id']}` + `{arch['objective_id']}` (`{arch['gradient_mode']}`) + `{arch['decision_readout_id']}` + `{arch['router_id']}` + `{arch['memory_arm_id']}`",
        f"- **Backbone Sharing Mode**: `{arch['architecture_mode']}` (`{stage3['selected_batching_mode']}` multi-task schedule)",
        f"- **Dataset Provenance**: `{prov['dataset_source']}` — {prov.get('corpus_description', 'verified zero-contamination split')}",
        f"- **Corpus Baselines (`test_eval`)**: Empirical UTF-8 Byte Unigram = `{baselines.get('byte_unigram_bpb', 0.0):.4f} BPB` | Empirical BPE-4096 Token Unigram = `{baselines.get('bpe_token_unigram_bpb', 0.0):.4f} BPB`",
        f"- **Trained Tournament Tiers**: Primary 25-arm tournament trained at `{hp['tier']}` (`{hp['estimated_params']:,}` total params; `{solve_core_config(arch['core_id'], hp['estimated_params'], hp['vocab_size']).non_embedding_params:,}` non-embedding params)"
        + (
            f" + Scale-Up verification trained at `~3.0M` (`{scaleup_records[0]['trainable_params']:,}` params across `{[r['arm_id'] for r in scaleup_records]}`); "
            if scaleup_records
            else "; "
        )
        + f"target `~17.0M` shape {t17_note}.",
        f"- **Held-Out Evaluation Item Counts (`N`)**:",
        f"  - LM Corpus: `train={counts.get('lm_train_utf8_bytes', 0):,} B` (`{counts.get('lm_train_docs', 0)}` shards), `val_cal` (`{counts.get('lm_val_cal_docs', 0)}` shards), `test_eval={counts.get('lm_test_eval_utf8_bytes', 0):,} B` (`{counts.get('lm_test_eval_docs', 0)}` shards, `0` 64-byte contamination overlaps), `grammar_probes={counts.get('grammar_probes', 0)}`",
        f"  - Mechanistic Recall & Copy: `MQAR={counts.get('mqar_eval_sequences_per_kv', 0)}` seqs/kv (`{counts.get('mqar_eval_queries_per_kv', 0):,}` queries per `kv` in `{{4,16,32,64}}`, `{counts.get('mqar_eval_queries_total', 0):,}` total queries), `Delayed Copy={counts.get('copy_eval_sequences', 0)}` seqs (`{counts.get('copy_eval_tokens', 0):,}` target tokens)",
        f"  - Track 3 Decisions: `train={counts.get('decision_train', 0)}`, `val_cal={counts.get('decision_val_cal', 0)}`, `test_in_domain={counts.get('decision_test_in_domain', 0)}`, `test_shifted_unseen={counts.get('decision_test_shifted_unseen', 0)}` (`{counts.get('decision_test_total', 0):,}` held-out test items)",
        f"  - Track 4 Routers: `train={counts.get('routing_train', 0)}`, `val_cal={counts.get('routing_val_cal', 0)}`, `test_eval={counts.get('routing_test_eval', 0):,}` held-out items",
        f"  - Track 5 Continual Memory: `{counts.get('continual_facts_total', 0):,}` propositions (`5` domains) + `{counts.get('continual_distractors_total', 0)}` interleaved distractors + `{counts.get('contradiction_pairs', 0)}` contradiction updates",
        "",
        "## 1. Track 1 — Sequence-Mixing Core Comparison (`C0`–`C6` at `0.35M`)",
        "",
        f"*Reference unigram baselines on `test_eval`: Byte-Unigram `{baselines.get('byte_unigram_bpb', 0.0):.4f} BPB`, BPE-4096 Token-Unigram `{baselines.get('bpe_token_unigram_bpb', 0.0):.4f} BPB`.*",
        "",
        "| Core ID | Complexity Class | Total Params | Non-Emb Params | MQAR-4 | MQAR-16 | MQAR-32 | MQAR-64 | Copy Acc | Held-Out BPB | Grammar Acc | State @ 4K (KB) | Stage-1 Gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    verdict_map = {v["arm_id"]: v for v in stage1["verdicts"]}
    for r in core_records:
        cid = r["arm_id"]
        v = verdict_map[cid]
        gate_str = "REFERENCE" if v["is_reference_only"] else ("PASS" if v["passed"] else f"REJECT ({v['rejection_reason']})")
        lines.append(
            f"| `{cid}` | `{r['complexity_class']}` | {r['trainable_params']:,} | {r.get('non_embedding_params', 0):,} | "
            f"{r['mqar_4_acc']:.3f} | {r['mqar_16_acc']:.3f} | {r['mqar_32_acc']:.3f} | {r['mqar_64_acc']:.3f} | "
            f"{r['copy_acc']:.3f} | {r['held_out_bpb']:.4f} | {r['grammar_acc']:.3f} | "
            f"{r['state_bytes_at_4k'] / 1024.0:.1f} | {gate_str} |"
        )

    ci_core = stage2["track1_core"].get("bootstrap_vs_runner_up")
    if ci_core:
        lines.extend(
            [
                "",
                f"**Track-1 Winner**: `{stage2['track1_core']['winner_id']}` (Pareto frontier: `{stage2['track1_core']['pareto_frontier']}`). "
                f"10,000-resample paired bootstrap vs `{ci_core['arm_b']}` (`N={counts.get('mqar_eval_sequences_per_kv', 0)}` eval sequences, `{counts.get('mqar_eval_queries_per_kv', 0):,}` queries) on MQAR-16: "
                f"`Δ = {ci_core['mean_diff']:+.3f}`, 95% CI `[{ci_core['ci_low']:+.3f}, {ci_core['ci_high']:+.3f}]`, `p = {ci_core['p_value_two_sided']:.4f}`.",
            ]
        )

    if scaleup_records:
        lines.extend(
            [
                "",
                "### 1.1 Scale-Up Verification (`~3.0M` Trainable Parameters on Real TinyStories)",
                "",
                "| Core ID | Total Params | Non-Emb Params | `0.35M` BPB | `3.0M` Held-Out BPB | Δ vs BPE Unigram | MQAR-16 (`3.0M`) | MQAR-32 (`3.0M`) | Train Tok/s | Effective GFLOP/s (`6·N·tok/s`) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        micro_bpb_map = {r["arm_id"]: r["held_out_bpb"] for r in core_records}
        bpe_uni = float(baselines.get("bpe_token_unigram_bpb", 0.0))
        for sr in scaleup_records:
            cid = sr["arm_id"]
            m_bpb = micro_bpb_map.get(cid, float("nan"))
            lines.append(
                f"| `{cid}` | {sr['trainable_params']:,} | {sr['non_embedding_params']:,} | "
                f"{m_bpb:.4f} | **{sr['held_out_bpb']:.4f}** | {sr['held_out_bpb'] - bpe_uni:+.4f} | "
                f"{sr['mqar_16_acc']:.3f} | {sr['mqar_32_acc']:.3f} | "
                f"{sr['train_tokens_per_sec']:.0f} | {sr['effective_gflops_total']:.1f} |"
            )

    lines.extend(
        [
            "",
            "## 2. Track 2 — Auxiliary Objectives (`O1`–`O4`) & Equal-Non-Embedding Tokenizer Comparison",
            "",
            "### 2.1 Objectives & Online RTRL (`O1`–`O4`)",
            "",
            "| Arm ID | Held-Out BPB | Mean Aux Loss | Aux Grad Norm | Shifted Probe Macro-F1 | In-Domain Probe F1 | Train Tok/s | Stage-1 Gate Verdict |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for r in obj_records:
        oid = r["arm_id"]
        v = verdict_map[oid]
        gate_str = "PASS" if v["passed"] else f"REJECT ({v['rejection_reason']})"
        lines.append(
            f"| `{oid}` | {r['held_out_bpb']:.4f} | {r.get('mean_aux_loss', 0.0):.4f} | "
            f"{r.get('aux_grad_norm', 0.0):.4f} | {r.get('probe_macro_f1', 0.0):.3f} | "
            f"{r.get('probe_in_domain_macro_f1', 0.0):.3f} | {r['train_tokens_per_sec']:.1f} | {gate_str} |"
        )

    lines.extend(
        [
            "",
            "### 2.2 Paired Tokenizer Tables (`V=256` UTF-8 Bytes vs `V=4096` Compact BPE at Equal Non-Embedding Params)",
            "",
            "*Four independent training runs with matched core body (`non_embedding_params` identical across `Tok_bytes_256` and `Tok_bpe_4096`): two independent Equal-UTF8-Byte runs and two independent Equal-Wall-Clock runs.*",
            "",
            "| Arm ID | Vocab | Non-Emb Params | Total Params | Equal-Byte BPB (Bytes Seen) | Equal-Clock BPB (Wall s) | Eager Step Bytes/s | Lossless UTF-8 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for r in tok_records:
        lines.append(
            f"| `{r['arm_id']}` | {r['vocab_size']} | {r['non_embedding_params']:,} | {r['trainable_params']:,} | "
            f"{r['equal_byte_bpb']:.4f} (`{r['equal_byte_utf8_bytes_seen']:,} B`) | "
            f"{r['equal_clock_bpb']:.4f} (`{r['equal_clock_wall_s']:.1f}s`) | "
            f"{r.get('int8_bytes_per_second', 0.0):.1f} | `{r['lossless_roundtrip']}` |"
        )

    lines.extend(
        [
            "",
            f"## 3. Track 3 — System-1 Decision Readouts & Calibration (`D1`–`D4`, `N={counts.get('decision_test_total', 0):,}` Held-Out Test Items)",
            "",
            "*Ranked by held-out unseen `shifted_macro_f1`, `oos_f1`, `aurc` (lower is better), and `adaptive_ece` (lower is better) — never in-domain F1.*",
            "",
            "| Readout ID | $T_{\\text{cal}}$ | Shifted/Unseen Macro-F1 | OOS Null F1 | Shifted Null Rate | In-Domain Macro-F1 | Adaptive ECE | AURC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in dec_records:
        lines.append(
            f"| `{r['arm_id']}` | {r['temperature']:.3f} | **{r['shifted_macro_f1']:.3f}** | "
            f"{r['oos_f1']:.3f} | {r.get('pred_null_rate_shifted', 0.0):.3f} | "
            f"{r['in_domain_macro_f1']:.3f} | {r['adaptive_ece']:.4f} | {r['aurc']:.4f} |"
        )

    lines.extend(
        [
            "",
            f"## 4. Track 4 — 5-Way Action Routers & Safety Guards (`R1`–`R4`, `N={counts.get('routing_test_eval', 0):,}` Held-Out Items)",
            "",
            "| Router ID | Normalized Utility | Accuracy | False Tool Rate | Unblocked Invalid Tool | Adaptive ECE | AURC | p50 Latency (ms) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in rt_records:
        lines.append(
            f"| `{r['arm_id']}` | {r['normalized_utility']:.3f} | {r['accuracy']:.3f} | "
            f"{r['false_tool_rate']:.3f} | {r['unblocked_invalid_tool_rate']:.3f} | "
            f"{r['adaptive_ece']:.4f} | {r.get('aurc', 0.0):.4f} | {r['p50_latency_ms']:.3f} |"
        )

    lines.extend(
        [
            "",
            f"## 5. Track 5 — Paraphrased Lifelong Memory & Contradiction Adaptation (`M0`–`M5`, `{counts.get('continual_facts_total', 0):,}` Facts + `{counts.get('continual_distractors_total', 0)}` Distractors + `{counts.get('contradiction_pairs', 0)}` Contradictions)",
            "",
            "| Memory Arm | 5-Domain Top-1 Retention | Top-4 Recall | Backward Forgetting | Contradiction Update Acc | Stale Answer Rate | Active Items | Static + Dynamic Bytes | Stage-1 Gate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for r in mem_records:
        v = verdict_map[r["arm_id"]]
        gate_str = "PASS" if v["passed"] else f"REJECT ({v['rejection_reason']})"
        lines.append(
            f"| `{r['arm_id']}` | {r['average_final_accuracy']:.3f} | {r.get('average_final_recall_at_4', 0.0):.3f} | "
            f"{r['backward_forgetting']:.3f} | {r['contradiction_update_accuracy']:.3f} | "
            f"{r['stale_answer_rate']:.3f} | {r.get('active_items', 0):,} | "
            f"{r['total_footprint_bytes']:,} | {gate_str} |"
        )

    comp3 = stage3["comparison"]
    lines.extend(
        [
            "",
            "## 6. Stage 3 — Shared-Backbone vs Equal-Budget Split-Specialist (`D4`) Interference Verdict",
            "",
            f"- **Selected Mode**: `{stage3['selected_architecture_mode']}` (`{stage3['selected_batching_mode']}`)",
            f"- **Verdict**: {stage3['verdict_reason']}",
            f"- **Parameter Parity**: relative difference `{stage3['param_relative_diff']:.2%}` (`±2.0%` budget rule enforced)",
            "",
            "| Regime | Held-Out BPB | Shifted/Avg Decision Macro-F1 |",
            "| --- | ---: | ---: |",
            f"| Standalone LM / Standalone Decision | {comp3['standalone_lm_bpb']:.4f} | {comp3['standalone_decision_macro_f1']:.3f} |",
            f"| Unified Shared (`interleaved 0.80/0.15/0.05`) | {comp3['unified_interleaved_bpb']:.4f} | {comp3['unified_interleaved_macro_f1']:.3f} |",
            f"| Unified Shared (`summed` loss ablation) | {comp3['unified_summed_bpb']:.4f} | {comp3['unified_summed_macro_f1']:.3f} |",
            f"| Equal-Budget Split-Specialist (`D4`) | {comp3['split_d4_bpb']:.4f} | {comp3['split_d4_macro_f1']:.3f} |",
            "",
            "## 7. Stage 4 — Leave-One-Out (`LOO`) Complexity Pruning Waterfall",
            "",
            "| Complexity Component | Simpler Fallback | Utility w/ Component | Utility w/o Component | Marginal Gain | Decision |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for w in stage4["waterfall"]:
        lines.append(
            f"| `{w['component']}` | `{w['simpler_fallback']}` | {w['utility_with_component']:.4f} | "
            f"{w['utility_without_component']:.4f} | {w['marginal_gain']:+.4f} | **{w['decision'].upper()}** |"
        )

    lines.extend(
        [
            "",
            "## 8. Provisional RecaMem Core Specification & Hardware Reality Check",
            "",
            f"- **Manifest Path**: `configs/recamem_selected_core.json` (`selection_status=\"{manifest['selection_status']}\"`, `target_17m_trained={t17_trained}`)",
            f"- **Evidence Ledger**: `reports/evidence/recamem-core-tournament.json`",
            f"- **Runtime Loader**: `from src.model.recamem_core import build_recamem_core`",
            f"- **Solved (Untrained) ~17M Production Shape**: `d_model={hp17['d_model']}`, `n_layers={hp17['n_layers']}`, `n_heads={hp17['n_heads']}`, `d_ff={hp17['d_ff']}`, `vocab_size={hp17['vocab_size']}` (`{hp17['estimated_params']:,}` trainable parameters — requires multi-core/GPU run before promoting from `provisional_core` to `verified_winner`)",
            f"- **CPU Training & Decode Reality Check**: At `0.35M` params (`~2.4–11K tok/s` depending on batch/chunk width), effective CPU compute is `~5–25 GFLOP/s` (`~50–70 GFLOP/s` at `3.0M` dense matmuls). At `10–50 GFLOP/s`, training a `17M`-parameter model on `30M` tokens (`~3.06 × 10^15` FLOPs) requires **`17 hours` (`50 GFLOP/s`) to `3.5 days` (`10 GFLOP/s`)** on 2 CPU vCPUs. Likewise, `batch=1` Python step-by-step decoding is Python-dispatch overhead-bound (`~400–850 bytes/s`), not compute-bound.",
            f"- **Calibrated Thresholds**: `decision_temperature={cal['decision_temperature']:.3f}`, `router_temperature={cal['router_temperature']:.3f}`, `router_abstain_threshold={cal['router_abstain_threshold']:.3f}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_tournament(args: argparse.Namespace) -> int:
    torch.set_num_threads(max(2, os.cpu_count() or 2))
    real_dir: Optional[Path] = None
    if getattr(args, "use_tinystories", False):
        real_dir = ensure_tinystories_real_data_dir()
    elif args.real_data_dir:
        real_dir = Path(args.real_data_dir)

    scale_mode = str(getattr(args, "scale", "auto"))
    if scale_mode == "auto":
        scale_mode = "large" if real_dir is not None else "compact"

    workloads = load_tournament_workloads(
        real_data_dir=real_dir,
        require_real_data=bool(args.require_real_data),
        seed=int(args.seed),
        scale=scale_mode,
    )

    out_manifest = Path(args.output_manifest)
    out_evidence = Path(args.output_evidence)
    out_report = Path(args.output_report)

    if workloads.status == "blocked":
        blocked_doc = build_report(
            command=" ".join(sys.argv),
            config=vars(args),
            measurements={},
            blocked=True,
            skip_reason=workloads.reason,
        )
        write_report(blocked_doc, out_evidence)
        print(f"[BLOCKED] {workloads.reason}", file=sys.stderr)
        return 2

    tier = str(args.tier)
    tier_budget = int(args.budget) if args.budget is not None else TIER_BUDGETS[tier]
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    primary_seed = seeds[0]

    if args.dry_run:
        solved = solve_all_cores_for_tier(tier_budget, vocab_size=4096)
        print(
            json.dumps(
                {
                    "status": "dry_run_ok",
                    "tier": tier,
                    "tier_budget": tier_budget,
                    "scale": scale_mode,
                    "dataset_source": workloads.dataset_source,
                    "evaluation_counts": workloads.item_counts,
                    "solved_cores": {k: v.to_dict() for k, v in solved.items()},
                },
                indent=2,
            )
        )
        return 0

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _CORPUS_TOKEN_CACHE.clear()

    # ------------------------------------------------------------------ #
    # Stage 0: Milestone M0 — Parameter Parity, Tokenizers & Unigram Baselines
    # ------------------------------------------------------------------ #
    byte_tok = UTF8ByteTokenizer()
    bpe_train_bytes = 600_000 if scale_mode == "large" else 200_000
    bpe_path = out_manifest.parent / "recamem_selected_bpe.json"
    if getattr(args, "resume", False) and bpe_path.is_file():
        print(f"[Stage 0] Resuming cached CompactBPETokenizer from {bpe_path} ...", flush=True)
        bpe_tok = CompactBPETokenizer.load(bpe_path)
    else:
        print(f"[Stage 0] Training CompactBPETokenizer(vocab_size=4096, max_bytes={bpe_train_bytes:,}) ...", flush=True)
        bpe_tok = CompactBPETokenizer.train_from_texts(
            workloads.lm_train.values(), vocab_size=4096, max_training_bytes=bpe_train_bytes
        )
        bpe_tok.save(bpe_path)

    byte_unigram_bpb = float(
        compute_corpus_unigram_bpb(workloads.lm_train, workloads.lm_test_eval)["byte_unigram_bpb"]
    )
    bpe_token_unigram_bpb = _compute_token_unigram_bpb(workloads.lm_train, workloads.lm_test_eval, bpe_tok)
    print(
        f"[Stage 0] Unigram baselines on test_eval: byte={byte_unigram_bpb:.4f} BPB, "
        f"bpe_token={bpe_token_unigram_bpb:.4f} BPB",
        flush=True,
    )

    solved_active = solve_all_cores_for_tier(tier_budget, vocab_size=4096, tolerance=0.02)
    solved_target_17m = solve_all_cores_for_tier("target", vocab_size=4096, tolerance=0.02)
    sample_probe = "Orbital radiometry Δt = 0.05s — lossless UTF-8 check!"
    stage0 = {
        "tier": tier,
        "tier_budget": tier_budget,
        "scale": scale_mode,
        "dataset_source": workloads.dataset_source,
        "corpus_metadata": workloads.corpus_metadata,
        "evaluation_counts": workloads.item_counts,
        "unigram_baselines": {
            "byte_unigram_bpb": byte_unigram_bpb,
            "bpe_token_unigram_bpb": bpe_token_unigram_bpb,
        },
        "param_matching_active_tier": {k: v.to_dict() for k, v in solved_active.items()},
        "param_matching_target_17m": {k: v.to_dict() for k, v in solved_target_17m.items()},
        "byte_tok_lossless": byte_tok.decode(byte_tok.encode(sample_probe, add_eos=True)) == sample_probe,
        "bpe_tok_lossless": bpe_tok.decode(bpe_tok.encode(sample_probe, add_eos=True)) == sample_probe,
    }
    _atomic_write_json(ckpt_dir / "stage0.json", stage0)

    # ------------------------------------------------------------------ #
    # Stage 1 & 2: Track-by-Track Controlled Evaluations Across All 24+ Arms
    # ------------------------------------------------------------------ #
    mqar_steps = int(args.mqar_steps)
    copy_steps = int(args.copy_steps)
    lm_steps = int(args.lm_steps)
    lm_seq_len = 128 if scale_mode == "large" else 80
    lm_batch_size = 16 if (scale_mode == "large" and tier_budget <= 600_000) else 4

    # Track 1: Evaluate all 7 cores (C0..C6) with 3-point LR micro-sweep on val_cal
    core_records: List[Dict[str, Any]] = []
    core_models: Dict[str, CandidateLM] = {}
    for cid in ALL_CORE_IDS:
        core_ckpt_path = ckpt_dir / f"core_{cid}.json"
        if getattr(args, "resume", False) and core_ckpt_path.is_file():
            saved_c = json.loads(core_ckpt_path.read_text(encoding="utf-8"))
            if (
                saved_c.get("_tier_budget") == tier_budget
                and saved_c.get("_scale") == scale_mode
                and saved_c.get("seeds_evaluated") is not None
            ):
                if cid == "C4":
                    lm_m, _ = _train_and_eval_lm_arm(
                        cid,
                        tier_budget,
                        bpe_tok,
                        "Tok_bpe_4096",
                        "O1",
                        workloads,
                        lm_steps=lm_steps,
                        lr=float(saved_c["selected_lr"]),
                        seed=primary_seed,
                        seq_len=lm_seq_len,
                        batch_size=lm_batch_size,
                        measure_speed=False,
                    )
                    core_models[cid] = lm_m
                saved_clean = {k: v for k, v in saved_c.items() if not k.startswith("_")}
                print(
                    f"[Track 1] Resumed cached core {cid}: mqar4={saved_clean['mqar_4_acc']:.3f} "
                    f"mqar16={saved_clean['mqar_16_acc']:.3f} bpb={saved_clean['held_out_bpb']:.4f}",
                    flush=True,
                )
                core_records.append(saved_clean)
                continue

        def _eval_lr(cand_lr: float, _cid: str = cid) -> Dict[str, Any]:
            _, rec = _train_and_eval_lm_arm(
                _cid,
                tier_budget,
                bpe_tok,
                "Tok_bpe_4096",
                "O1",
                workloads,
                lm_steps=max(12, lm_steps // 3),
                lr=cand_lr,
                seed=primary_seed,
                seq_len=lm_seq_len,
                batch_size=lm_batch_size,
            )
            return {"val_loss": rec["val_bpb"]}

        lr_sweep = select_best_lr(_eval_lr, lr_grid=DEFAULT_LR_GRID, metric_key="val_loss")
        best_lr = float(lr_sweep["selected_lr"])

        seed_mqar4, seed_mqar16, seed_mqar32, seed_mqar64, seed_copy, seed_bpb = [], [], [], [], [], []
        last_lm_model: Optional[CandidateLM] = None
        last_lm_rec: Dict[str, Any] = {}
        last_mech_rec: Dict[str, Any] = {}

        c5_win = 16 if cid == "C5" else None
        run_seeds = seeds if cid in {"C3", "C4", "C5", "C6"} else [primary_seed]
        print(f"[Track 1] Evaluating core {cid} (best_lm_lr={best_lr}, seeds={run_seeds}) ...", flush=True)
        chosen_mqar_lr: Optional[float] = None
        for sd_idx, sd in enumerate(run_seeds):
            mech = _train_and_eval_mqar_and_copy(
                cid,
                tier_budget,
                workloads,
                mqar_steps=mqar_steps,
                copy_steps=copy_steps,
                lr=chosen_mqar_lr if chosen_mqar_lr is not None else 3e-3,
                seed=sd,
                window_size_override=c5_win,
                sweep_mqar_lr=(sd_idx == 0),
            )
            chosen_mqar_lr = float(mech.get("mqar_lr", 3e-3))
            lm_m, lm_r = _train_and_eval_lm_arm(
                cid,
                tier_budget,
                bpe_tok,
                "Tok_bpe_4096",
                "O1",
                workloads,
                lm_steps=lm_steps,
                lr=best_lr,
                seed=sd,
                seq_len=lm_seq_len,
                batch_size=lm_batch_size,
                measure_speed=(sd == run_seeds[-1]),
            )
            seed_mqar4.append(mech["mqar_4_acc"])
            seed_mqar16.append(mech["mqar_16_acc"])
            seed_mqar32.append(mech["mqar_32_acc"])
            seed_mqar64.append(mech["mqar_64_acc"])
            seed_copy.append(mech["copy_acc"])
            seed_bpb.append(lm_r["held_out_bpb"])
            last_lm_model = lm_m
            last_lm_rec = lm_r
            last_mech_rec = mech

        assert last_lm_model is not None
        if cid == "C4":
            core_models[cid] = last_lm_model
        del last_lm_model
        gc.collect()
        merged_core = {
            **last_lm_rec,
            **last_mech_rec,
            "arm_id": cid,
            "selected_lr": best_lr,
            "mqar_selected_lr": chosen_mqar_lr,
            "mqar_4_acc": float(statistics.mean(seed_mqar4)),
            "mqar_16_acc": float(statistics.mean(seed_mqar16)),
            "mqar_32_acc": float(statistics.mean(seed_mqar32)),
            "mqar_64_acc": float(statistics.mean(seed_mqar64)),
            "copy_acc": float(statistics.mean(seed_copy)),
            "held_out_bpb": float(statistics.mean(seed_bpb)),
            "seeds_evaluated": run_seeds,
        }
        core_records.append(merged_core)
        _atomic_write_json(core_ckpt_path, {**merged_core, "_tier_budget": tier_budget, "_scale": scale_mode})
        print(
            f"[Track 1] Done {cid}: mqar4={merged_core['mqar_4_acc']:.3f} "
            f"mqar16={merged_core['mqar_16_acc']:.3f} bpb={merged_core['held_out_bpb']:.4f}",
            flush=True,
        )

    anchor_core_id = "C4"
    anchor_lr = float(next(r["selected_lr"] for r in core_records if r["arm_id"] == anchor_core_id))

    # Track 1b: Scale-Up Verification (`~3.0M` trainable parameters across C4, C3, C6)
    scaleup_records: List[Dict[str, Any]] = []
    if getattr(args, "include_scaleup", False):
        scaleup_budget = int(getattr(args, "scaleup_budget", 3_000_000))
        scaleup_ckpt = ckpt_dir / f"scaleup_{scaleup_budget}.json"
        if getattr(args, "resume", False) and scaleup_ckpt.is_file():
            scaleup_records = json.loads(scaleup_ckpt.read_text(encoding="utf-8")).get("records", [])
            print(f"[Track 1b] Resumed {len(scaleup_records)} scale-up records @ {scaleup_budget:,} params.", flush=True)
        else:
            scaleup_lm_steps = max(lm_steps, int(getattr(args, "scaleup_lm_steps", 200)))
            scaleup_mqar_steps = max(120, min(mqar_steps, 420))
            for sc_cid in ("C4", "C3", "C6"):
                print(f"[Track 1b] Scale-up {sc_cid} @ ~{scaleup_budget:,} params (steps={scaleup_lm_steps}) ...", flush=True)
                sc_lr = 1.5e-3
                _, sc_lm_rec = _train_and_eval_lm_arm(
                    sc_cid,
                    scaleup_budget,
                    bpe_tok,
                    "Tok_bpe_4096",
                    "O1",
                    workloads,
                    lm_steps=scaleup_lm_steps,
                    lr=sc_lr,
                    seed=primary_seed,
                    seq_len=128,
                    batch_size=8,
                )
                sc_mech = _train_and_eval_mqar_and_copy(
                    sc_cid,
                    scaleup_budget,
                    workloads,
                    mqar_steps=scaleup_mqar_steps,
                    copy_steps=max(20, copy_steps // 2),
                    lr=1e-3 if sc_cid == "C6" else 2e-3,
                    seed=primary_seed,
                    sweep_mqar_lr=False,
                )
                sc_merged = {
                    **sc_lm_rec,
                    "mqar_4_acc": sc_mech["mqar_4_acc"],
                    "mqar_16_acc": sc_mech["mqar_16_acc"],
                    "mqar_32_acc": sc_mech["mqar_32_acc"],
                    "mqar_64_acc": sc_mech["mqar_64_acc"],
                    "copy_acc": sc_mech["copy_acc"],
                    "arm_id": sc_cid,
                    "scale_budget": scaleup_budget,
                }
                scaleup_records.append(sc_merged)
                gc.collect()
                print(
                    f"[Track 1b] {sc_cid} @ {sc_merged['trainable_params']:,} params: "
                    f"bpb={sc_merged['held_out_bpb']:.4f} mqar16={sc_merged['mqar_16_acc']:.3f} "
                    f"gflops={sc_merged['effective_gflops_total']:.1f}",
                    flush=True,
                )
            _atomic_write_json(scaleup_ckpt, {"budget": scaleup_budget, "records": scaleup_records})

    # Track 2a: Objectives O1, O2, O3 on C4 + O4_online_rtrl on C1
    print("[Track 2] Evaluating Objectives O1..O4 and Equal-Non-Embedding Tokenizers ...", flush=True)
    obj_records: List[Dict[str, Any]] = []
    for oid in ("O1", "O2", "O3"):
        m_obj, r_obj = _train_and_eval_lm_arm(
            anchor_core_id,
            tier_budget,
            bpe_tok,
            "Tok_bpe_4096",
            oid,
            workloads,
            lm_steps=lm_steps,
            lr=anchor_lr,
            seed=primary_seed,
            seq_len=lm_seq_len,
            batch_size=lm_batch_size,
            probe_aux_grad=True,
        )
        _, _, d_probe = _evaluate_decision_arm(
            "D2", m_obj, bpe_tok, workloads, tier_budget=tier_budget, core_id=anchor_core_id, seed=primary_seed
        )
        r_obj["arm_id"] = oid
        r_obj["probe_macro_f1"] = d_probe["shifted_macro_f1"]
        r_obj["probe_in_domain_macro_f1"] = d_probe["in_domain_macro_f1"]
        obj_records.append(r_obj)

    _, c1_bptt = _train_and_eval_lm_arm(
        "C1",
        tier_budget,
        bpe_tok,
        "Tok_bpe_4096",
        "O1",
        workloads,
        lm_steps=lm_steps,
        lr=1e-3,
        seed=primary_seed,
        seq_len=lm_seq_len,
        batch_size=4,
    )
    m_c1_rtrl, c1_rtrl = _train_and_eval_lm_arm(
        "C1",
        tier_budget,
        bpe_tok,
        "Tok_bpe_4096",
        "O1",
        workloads,
        lm_steps=lm_steps,
        lr=1e-3,
        seed=primary_seed,
        gradient_mode="online_rtrl",
        seq_len=80,
        batch_size=4,
    )
    _, _, d_probe_o4 = _evaluate_decision_arm(
        "D2", m_c1_rtrl, bpe_tok, workloads, tier_budget=tier_budget, core_id="C1", seed=primary_seed
    )
    c1_rtrl["arm_id"] = "O4_online_rtrl"
    c1_rtrl["probe_macro_f1"] = d_probe_o4["shifted_macro_f1"]
    c1_rtrl["probe_in_domain_macro_f1"] = d_probe_o4["in_domain_macro_f1"]
    c1_rtrl["paired_bptt_bpb"] = c1_bptt["held_out_bpb"]
    c1_rtrl["paired_bptt_tps"] = c1_bptt["train_tokens_per_sec"]
    c1_rtrl["supports_matrix_cores"] = False
    obj_records.append(c1_rtrl)

    # Track 2b: Tokenizers (`Tok_bytes_256` vs `Tok_bpe_4096`) at EQUAL NON-EMBEDDING PARAMETERS across 4 independent runs!
    non_emb_target = max(24_000, int(round(tier_budget * 0.70)))
    bpe_non_emb_cfg = solve_core_config_non_embedding(
        anchor_core_id, target_non_embedding_params=non_emb_target, vocab_size=bpe_tok.vocab_size
    )
    byte_non_emb_cfg = bpe_non_emb_cfg.with_vocab_size(byte_tok.vocab_size)
    assert bpe_non_emb_cfg.non_embedding_params == byte_non_emb_cfg.non_embedding_params, (
        "Non-embedding parameter parity violated between Tok_bpe_4096 and Tok_bytes_256"
    )

    # Run 1 (BPE Equal-Byte) & Run 2 (Bytes Equal-Byte): identical UTF-8 byte budget
    _, bpe_probe_batches_res = _pack_token_batches(
        list(workloads.lm_train.values()),
        bpe_tok,
        seq_len=lm_seq_len,
        batch_size=lm_batch_size,
        cache_tag="Tok_bpe_4096",
    )
    target_utf8_bytes = int(round(lm_steps * lm_seq_len * lm_batch_size * bpe_probe_batches_res))
    _, bpe_eqbyte_rec = _train_and_eval_lm_arm(
        anchor_core_id,
        tier_budget,
        bpe_tok,
        "Tok_bpe_4096",
        "O1",
        workloads,
        lm_steps=lm_steps * 2,
        lr=anchor_lr,
        seed=primary_seed + 11,
        custom_cfg=bpe_non_emb_cfg,
        seq_len=lm_seq_len,
        batch_size=lm_batch_size,
        max_utf8_bytes_budget=target_utf8_bytes,
        measure_speed=True,
    )
    _, byte_eqbyte_rec = _train_and_eval_lm_arm(
        anchor_core_id,
        tier_budget,
        byte_tok,
        "Tok_bytes_256",
        "O1",
        workloads,
        lm_steps=lm_steps * 5,
        lr=anchor_lr,
        seed=primary_seed + 11,
        custom_cfg=byte_non_emb_cfg,
        seq_len=lm_seq_len,
        batch_size=lm_batch_size,
        max_utf8_bytes_budget=target_utf8_bytes,
        measure_speed=True,
    )

    # Run 3 (BPE Equal-Wall-Clock) & Run 4 (Bytes Equal-Wall-Clock): independent runs with identical wall-clock cap
    wall_budget_s = max(0.8, min(15.0, float(bpe_eqbyte_rec["train_wall_clock_s"])))
    _, bpe_eqclock_rec = _train_and_eval_lm_arm(
        anchor_core_id,
        tier_budget,
        bpe_tok,
        "Tok_bpe_4096",
        "O1",
        workloads,
        lm_steps=lm_steps,
        lr=anchor_lr,
        seed=primary_seed + 29,
        custom_cfg=bpe_non_emb_cfg,
        seq_len=lm_seq_len,
        batch_size=lm_batch_size,
        max_wall_clock_s=wall_budget_s,
    )
    _, byte_eqclock_rec = _train_and_eval_lm_arm(
        anchor_core_id,
        tier_budget,
        byte_tok,
        "Tok_bytes_256",
        "O1",
        workloads,
        lm_steps=lm_steps,
        lr=anchor_lr,
        seed=primary_seed + 29,
        custom_cfg=byte_non_emb_cfg,
        seq_len=lm_seq_len,
        batch_size=lm_batch_size,
        max_wall_clock_s=wall_budget_s,
    )

    tok_records = [
        {
            **byte_eqbyte_rec,
            "arm_id": "Tok_bytes_256",
            "equal_byte_bpb": byte_eqbyte_rec["held_out_bpb"],
            "equal_byte_utf8_bytes_seen": byte_eqbyte_rec["utf8_bytes_seen"],
            "equal_byte_wall_clock_s": byte_eqbyte_rec["train_wall_clock_s"],
            "equal_clock_bpb": byte_eqclock_rec["held_out_bpb"],
            "equal_clock_wall_s": byte_eqclock_rec["train_wall_clock_s"],
            "equal_clock_utf8_bytes_seen": byte_eqclock_rec["utf8_bytes_seen"],
            "wall_clock_budget_s": wall_budget_s,
            "utf8_byte_budget": target_utf8_bytes,
            "independent_runs_executed": 2,
            "lossless_roundtrip": stage0["byte_tok_lossless"],
        },
        {
            **bpe_eqbyte_rec,
            "arm_id": "Tok_bpe_4096",
            "equal_byte_bpb": bpe_eqbyte_rec["held_out_bpb"],
            "equal_byte_utf8_bytes_seen": bpe_eqbyte_rec["utf8_bytes_seen"],
            "equal_byte_wall_clock_s": bpe_eqbyte_rec["train_wall_clock_s"],
            "equal_clock_bpb": bpe_eqclock_rec["held_out_bpb"],
            "equal_clock_wall_s": bpe_eqclock_rec["train_wall_clock_s"],
            "equal_clock_utf8_bytes_seen": bpe_eqclock_rec["utf8_bytes_seen"],
            "wall_clock_budget_s": wall_budget_s,
            "utf8_byte_budget": target_utf8_bytes,
            "independent_runs_executed": 2,
            "lossless_roundtrip": stage0["bpe_tok_lossless"],
        },
    ]
    equal_tables = build_equal_budget_tables(tok_records)

    # Track 3: Decision Readouts D1, D2, D3, D4
    print("[Track 3-5] Evaluating Decision Readouts D1..D4, Routers R1..R4, Memory Arms M0..M5 ...", flush=True)
    anchor_lm = core_models[anchor_core_id]
    shared_dec_enc = _batch_encode_decision_texts(anchor_lm, bpe_tok, workloads.decisions)
    dec_train_steps = 180 if scale_mode == "large" else max(30, lm_steps)
    dec_records: List[Dict[str, Any]] = []
    dec_calibrators: Dict[str, TemperatureCalibrator] = {}
    for did in ("D1", "D2", "D3", "D4"):
        _, cal_d, rec_d = _evaluate_decision_arm(
            did,
            anchor_lm,
            bpe_tok,
            workloads,
            tier_budget=tier_budget,
            core_id=anchor_core_id,
            train_steps=dec_train_steps,
            seed=primary_seed,
            precomputed_enc=shared_dec_enc if did != "D4" else None,
        )
        dec_calibrators[did] = cal_d
        dec_records.append(rec_d)

    # Track 4: Routers R1, R2, R3, R4
    shared_rt_enc = _batch_encode_routing_queries(anchor_lm, bpe_tok, workloads.routing)
    rt_train_steps = 180 if scale_mode == "large" else max(30, lm_steps)
    rt_records: List[Dict[str, Any]] = []
    rt_modules: Dict[str, nn.Module] = {}
    for rid in ("R1", "R2", "R3", "R4"):
        rt_mod, rec_r = _evaluate_router_arm(
            rid,
            anchor_lm,
            bpe_tok,
            workloads,
            train_steps=rt_train_steps,
            seed=primary_seed,
            precomputed_rt=shared_rt_enc,
        )
        rt_modules[rid] = rt_mod
        rt_records.append(rec_r)

    # Track 5: Continual Memory & Adaptation Arms M0..M5
    mem_records: List[Dict[str, Any]] = []
    eval_dom_limit = 50 if scale_mode == "large" else None
    for mid in ("M0", "M1", "M2", "M3", "M4", "M5"):
        lm_mem = copy.deepcopy(anchor_lm)
        arm_kwargs: Dict[str, Any] = {}
        if mid == "M1":
            arm_kwargs = {"hash_buckets": 512 if scale_mode == "large" else 128}
        elif mid == "M2":
            arm_kwargs = {"surprise_threshold": 4.45 if scale_mode == "large" else 1.5}
        elif mid == "M4":
            arm_kwargs = {"code_dim": 512 if scale_mode == "large" else 256, "top_k_code": 32, "heads": 4}
        m_arm = build_memory_arm(mid, lm_mem, bpe_tok, **arm_kwargs)
        ret = evaluate_domain_retention_matrix(
            m_arm, workloads.continual_domains, eval_per_domain_limit=eval_dom_limit
        )
        con = evaluate_contradiction_updates(m_arm, workloads.contradiction_pairs)
        fp = con["footprint"]
        mem_rec = {
            "arm_id": mid,
            "average_final_accuracy": ret["average_final_accuracy"],
            "average_final_recall_at_4": ret.get("average_final_recall_at_4", 0.0),
            "backward_forgetting": ret["backward_forgetting"],
            "contradiction_update_accuracy": con["contradiction_update_accuracy"],
            "stale_answer_rate": con["stale_answer_rate"],
            "total_footprint_bytes": fp.get("total_bytes", 0),
            "active_items": ret["footprint"].get("active_items", 0),
            "retention_matrix": ret["retention_matrix"],
        }
        mem_records.append(mem_rec)
        print(
            f"[Track 5] {mid}: avg_top1={mem_rec['average_final_accuracy']:.3f} "
            f"top4={mem_rec['average_final_recall_at_4']:.3f} forget={mem_rec['backward_forgetting']:.3f} "
            f"contra={mem_rec['contradiction_update_accuracy']:.3f}",
            flush=True,
        )

    stage1 = evaluate_stage1_gates(
        core_records=core_records,
        objective_records=obj_records,
        tokenizer_records=tok_records,
        decision_records=dec_records,
        router_records=rt_records,
        memory_records=mem_records,
    )
    _atomic_write_json(ckpt_dir / "stage1.json", stage1)

    stage2 = evaluate_stage2_tracks(
        stage1_result=stage1,
        core_records=core_records,
        objective_records=obj_records,
        tokenizer_records=tok_records,
        decision_records=dec_records,
        router_records=rt_records,
        memory_records=mem_records,
        seed=primary_seed,
    )
    _atomic_write_json(ckpt_dir / "stage2.json", stage2)

    win_core = stage2["track1_core"]["winner_id"]
    win_obj = stage2["track2_objective"]["winner_id"]
    win_tok_id = stage2["track2_tokenizer"]["winner_id"]
    win_tok: TokenizerAdapter = bpe_tok if win_tok_id == "Tok_bpe_4096" else byte_tok
    win_dec = stage2["track3_decision"]["winner_id"]
    win_rt = stage2["track4_router"]["winner_id"]
    win_mem = stage2["track5_memory"]["winner_id"]

    # ------------------------------------------------------------------ #
    # Stage 3: Shared-Backbone vs Split-Specialist (`D4`) Interference Gate
    # ------------------------------------------------------------------ #
    print("[Stage 3-4] Running Shared-vs-Split Interference Gate & Leave-One-Out Pruning ...", flush=True)
    standalone_lm_bpb = next(r["held_out_bpb"] for r in core_records if r["arm_id"] == win_core)
    standalone_dec_rec = next(r for r in dec_records if r["arm_id"] == win_dec)
    standalone_f1 = float(standalone_dec_rec["shifted_macro_f1"])

    def _run_joint_multitask(batching_mode: str) -> Tuple[float, float, int]:
        _set_seed(primary_seed + 500)
        cfg_u = solve_core_config(win_core, tier_or_budget=tier_budget, vocab_size=win_tok.vocab_size)
        lm_u = build_candidate_lm(cfg_u, chunk_size=32)
        dec_u = OptionConditionedReadout(d_model=cfg_u.d_model)
        rt_u = build_router("R4", d_model=cfg_u.d_model, route_token_ids=win_tok.route_token_ids)
        obj_u = build_objective(win_obj, d_model=cfg_u.d_model)
        tr_u = TournamentTrainer(
            lm_u,
            objective=obj_u,
            extra_modules=[dec_u, rt_u],
            config=TrainerConfig(
                lr=anchor_lr,
                total_steps=lm_steps,
                micro_batch_size=lm_batch_size,
                batching_mode=batching_mode,
                seed=primary_seed,
            ),
            pad_id=win_tok.pad_id,
            eos_id=win_tok.eos_id,
        )
        lm_batches, _ = _pack_token_batches(
            list(workloads.lm_train.values()),
            win_tok,
            seq_len=lm_seq_len,
            batch_size=lm_batch_size,
            cache_tag=win_tok_id,
        )
        rng = random.Random(primary_seed)
        dec_train = workloads.decisions["train"]
        rt_train = workloads.routing["train"]

        def _dec_loss() -> torch.Tensor:
            row = rng.choice(dec_train)
            q_ids = torch.tensor([win_tok.encode(row["query"], add_eos=True) or [win_tok.eos_id]], dtype=torch.long)
            o_ids = [
                torch.tensor([win_tok.encode(d, add_eos=True) or [win_tok.eos_id]], dtype=torch.long)
                for d in row["option_descriptions"]
            ]
            qh = lm_u(q_ids, return_details=True)["hidden_states"]
            ohs = [lm_u(oid, return_details=True)["hidden_states"] for oid in o_ids]
            logits = dec_u.forward_logits(qh, ohs, include_null=True)
            targ = len(ohs) if int(row["gold_index"]) < 0 else int(row["gold_index"])
            return F.cross_entropy(logits, torch.tensor([targ], dtype=torch.long))

        def _rt_loss() -> torch.Tensor:
            row = rng.choice(rt_train)
            q_ids = torch.tensor([win_tok.encode(row["query"], add_eos=True) or [win_tok.eos_id]], dtype=torch.long)
            qh = lm_u(q_ids, return_details=True)["hidden_states"]
            logits = rt_u.forward_logits(qh)  # type: ignore[operator]
            targ = ACTION_NAMES.index(row["gold_action"])
            return F.cross_entropy(logits, torch.tensor([targ], dtype=torch.long))

        for step in range(lm_steps):
            tr_u.train_step(
                lm_batch=lm_batches[step % len(lm_batches)],
                decision_loss_fn=_dec_loss,
                routing_loss_fn=_rt_loss,
            )

        bpb_u = float(
            measure_held_out_bpb(
                workloads.lm_train,
                workloads.lm_test_eval,
                lambda db: _score_corpus_nll_nats(lm_u, win_tok, db, max_seq_len=128),
                vocab_size=win_tok.vocab_size,
                tokenizer_id=win_tok_id,
            )["value"]
        )
        _, _, d_eval = _evaluate_decision_arm(
            "D3",
            lm_u,
            win_tok,
            workloads,
            tier_budget=tier_budget,
            core_id=win_core,
            train_steps=min(120, dec_train_steps),
            seed=primary_seed,
        )
        f1_u = float(d_eval["shifted_macro_f1"])
        tot_p = sum(p.numel() for m in (lm_u, dec_u, rt_u) for p in m.parameters() if p.requires_grad)
        return bpb_u, f1_u, tot_p

    inter_bpb, inter_f1, unified_params = _run_joint_multitask("interleaved")
    summed_bpb, summed_f1, _ = _run_joint_multitask("summed")

    split_d4_sys = SplitSpecialistSystem(
        core_id=win_core,
        target_total_params=unified_params,
        vocab_size=win_tok.vocab_size,
        tolerance=0.02,
        chunk_size=32,
    )
    tr_d4_lm = TournamentTrainer(
        split_d4_sys.lm_model,
        objective=build_objective(win_obj, d_model=split_d4_sys.lm_model.d_model),
        config=TrainerConfig(lr=anchor_lr, total_steps=lm_steps, micro_batch_size=lm_batch_size, seed=primary_seed),
        pad_id=win_tok.pad_id,
        eos_id=win_tok.eos_id,
    )
    d4_batches, _ = _pack_token_batches(
        list(workloads.lm_train.values()),
        win_tok,
        seq_len=lm_seq_len,
        batch_size=lm_batch_size,
        cache_tag=win_tok_id,
    )
    for step in range(lm_steps):
        tr_d4_lm.train_step(lm_batch=d4_batches[step % len(d4_batches)])
    split_d4_bpb = float(
        measure_held_out_bpb(
            workloads.lm_train,
            workloads.lm_test_eval,
            lambda db: _score_corpus_nll_nats(split_d4_sys.lm_model, win_tok, db, max_seq_len=128),
            vocab_size=win_tok.vocab_size,
            tokenizer_id=win_tok_id,
        )["value"]
    )
    d4_dec_rec = next(r for r in dec_records if r["arm_id"] == "D4")
    split_d4_f1 = float(d4_dec_rec["shifted_macro_f1"])

    stage3 = evaluate_stage3_interference(
        standalone_lm_bpb=standalone_lm_bpb,
        standalone_decision_macro_f1=standalone_f1,
        unified_interleaved_bpb=inter_bpb,
        unified_interleaved_macro_f1=inter_f1,
        unified_summed_bpb=summed_bpb,
        unified_summed_macro_f1=summed_f1,
        split_d4_bpb=split_d4_bpb,
        split_d4_macro_f1=split_d4_f1,
        unified_total_params=unified_params,
        split_d4_total_params=split_d4_sys.total_trainable_params(),
    )
    _atomic_write_json(ckpt_dir / "stage3.json", stage3)

    # ------------------------------------------------------------------ #
    # Stage 4: Leave-One-Out (LOO) Complexity Pruning Waterfall
    # ------------------------------------------------------------------ #
    ablations: List[Dict[str, Any]] = []
    win_core_mqar16 = float(next(r["mqar_16_acc"] for r in core_records if r["arm_id"] == win_core))
    active_bpb = inter_bpb if stage3["selected_architecture_mode"] == "unified" else split_d4_bpb
    active_f1 = inter_f1 if stage3["selected_architecture_mode"] == "unified" else split_d4_f1

    def _loo_composite(mqar16_val: float, bpb_val: float, f1_val: float) -> float:
        return float(0.35 * mqar16_val + 0.40 * (2.0 / (1.0 + bpb_val)) + 0.25 * f1_val)

    base_utility = _loo_composite(win_core_mqar16, active_bpb, active_f1)

    use_conv_final = True
    if win_core == "C4":
        _, rec_no_conv = _train_and_eval_lm_arm(
            "C4",
            tier_budget,
            win_tok,
            win_tok_id,
            win_obj,
            workloads,
            lm_steps=lm_steps,
            lr=anchor_lr,
            seed=primary_seed,
            use_conv=False,
            seq_len=lm_seq_len,
            batch_size=lm_batch_size,
        )
        mech_no_conv = _train_and_eval_mqar_and_copy(
            "C4",
            tier_budget,
            workloads,
            mqar_steps=mqar_steps,
            copy_steps=max(20, copy_steps // 2),
            lr=3e-3,
            seed=primary_seed,
            use_conv=False,
        )
        no_conv_u = _loo_composite(mech_no_conv["mqar_16_acc"], rec_no_conv["held_out_bpb"], active_f1)
        ablations.append(
            {
                "component": "short_conv4",
                "simpler_fallback": "conv_off",
                "ablated_utility": no_conv_u,
            }
        )

    o1_bpb = next(r["held_out_bpb"] for r in obj_records if r["arm_id"] == "O1")
    o3_bpb = next(r["held_out_bpb"] for r in obj_records if r["arm_id"] == "O3")
    u_with_o3 = _loo_composite(win_core_mqar16, o3_bpb, active_f1)
    u_with_o1 = _loo_composite(win_core_mqar16, o1_bpb, active_f1)
    ablations.append(
        {
            "component": "ema_jepa_vicreg_aux_loss",
            "simpler_fallback": "O1_ce_eos",
            "ablated_utility": u_with_o1,
        }
    )

    u_summed = _loo_composite(win_core_mqar16, summed_bpb, summed_f1)
    ablations.append(
        {
            "component": "interleaved_multitask_schedule",
            "simpler_fallback": "summed_multitask_loss",
            "ablated_utility": u_summed,
        }
    )

    stage4 = evaluate_stage4_loo_pruning(
        full_stack_utility=max(base_utility, u_with_o3),
        ablations=ablations,
        seed=primary_seed,
    )
    _atomic_write_json(ckpt_dir / "stage4.json", stage4)

    if "short_conv4" in stage4["pruned_components"]:
        use_conv_final = False
    if "ema_jepa_vicreg_aux_loss" in stage4["pruned_components"]:
        win_obj = "O1"

    # ------------------------------------------------------------------ #
    # Export Canonical Manifest, Evidence Ledger, and Markdown Synthesis
    # ------------------------------------------------------------------ #
    active_cfg = solve_core_config(
        win_core, tier_or_budget=tier_budget, vocab_size=win_tok.vocab_size, use_conv=use_conv_final
    )
    target_17m_cfg = solve_core_config(
        win_core, tier_or_budget="target", vocab_size=win_tok.vocab_size, use_conv=use_conv_final
    )

    win_dec_cal = dec_calibrators[win_dec].temperature
    win_rt_mod = rt_modules[win_rt]
    win_rt_temp = float(getattr(win_rt_mod, "temperature", 1.0))
    win_rt_abs = float(getattr(win_rt_mod, "abstain_threshold", 0.45))

    target_17m_trained = bool(tier_budget >= 16_000_000)
    selection_status = "verified_winner" if target_17m_trained else "provisional_core"
    trained_tiers = [int(active_cfg.estimated_params)] + [
        int(sr["trainable_params"]) for sr in scaleup_records if sr["arm_id"] == win_core
    ]
    corpus_meta = workloads.corpus_metadata
    corpus_desc = (
        f"{corpus_meta.get('corpus_name', workloads.dataset_source)} "
        f"(sha256[:16]={str(corpus_meta.get('source_sha256', 'n/a'))[:16]}, "
        f"64B contamination overlaps={corpus_meta.get('contamination_64b_offenders', 0)})"
    )

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "selection_status": selection_status,
        "selected_architecture": {
            "core_id": win_core,
            "core_slug": CORE_METADATA[win_core]["slug"],
            "complexity_class": CORE_METADATA[win_core]["complexity_class"],
            "tokenizer_id": win_tok_id,
            "objective_id": win_obj,
            "gradient_mode": stage2["selected_gradient_mode"],
            "decision_readout_id": "D4" if stage3["selected_architecture_mode"] == "split_specialist" else win_dec,
            "router_id": win_rt,
            "memory_arm_id": win_mem,
            "architecture_mode": stage3["selected_architecture_mode"],
            "multitask_batching_mode": stage3["selected_batching_mode"],
        },
        "hyperparameters": {
            **active_cfg.to_dict(),
            "chunk_size": 64,
            "memory_injection_mode": "prompt_stuffing",
        },
        "target_17m_hyperparameters": {
            **target_17m_cfg.to_dict(),
            "chunk_size": 64,
            "memory_injection_mode": "prompt_stuffing",
            "trained": target_17m_trained,
        },
        "calibration": {
            "decision_temperature": win_dec_cal,
            "router_temperature": win_rt_temp,
            "router_abstain_threshold": win_rt_abs,
        },
        "tokenizer_artifact_path": str(bpe_path.relative_to(REPO_ROOT)) if bpe_path.is_relative_to(REPO_ROOT) else str(bpe_path),
        "tournament_provenance": {
            "stages_completed": [0, 1, 2, 3, 4],
            "all_stage_gates_evaluated": True,
            "unmeasured_metrics_count": 0,
            "target_17m_trained": target_17m_trained,
            "trained_parameter_tiers": trained_tiers,
            "dataset_source": workloads.dataset_source,
            "corpus_description": corpus_desc,
            "corpus_metadata": corpus_meta,
            "evaluation_counts": workloads.item_counts,
            "unigram_baselines": stage0["unigram_baselines"],
            "seeds": seeds,
            "evidence_ledger_path": str(out_evidence.relative_to(REPO_ROOT)) if out_evidence.is_relative_to(REPO_ROOT) else str(out_evidence),
            "selection_report_path": str(out_report.relative_to(REPO_ROOT)) if out_report.is_relative_to(REPO_ROOT) else str(out_report),
            "kept_optional_components": stage4["kept_components"],
            "pruned_optional_components": stage4["pruned_components"],
        },
    }
    validate_selected_core_manifest(manifest)
    _atomic_write_json(out_manifest, manifest)

    measurements = {
        "stage0_verification": measurement(stage0),
        "track1_cores": measurement(core_records),
        "track1b_scaleup": measurement(scaleup_records),
        "track2_objectives": measurement(obj_records),
        "track2_tokenizers": measurement(tok_records, equal_budget_tables=equal_tables),
        "track3_decisions": measurement(dec_records),
        "track4_routers": measurement(rt_records),
        "track5_memory_arms": measurement(mem_records),
        "stage1_gates": measurement(stage1),
        "stage2_rankings": measurement(stage2),
        "stage3_interference": measurement(stage3),
        "stage4_loo_pruning": measurement(stage4),
        "selected_manifest": measurement(manifest),
    }
    evidence_doc = build_report(
        command=" ".join(sys.argv),
        config={
            "tier": tier,
            "tier_budget": tier_budget,
            "scale": scale_mode,
            "seeds": seeds,
            "mqar_steps": mqar_steps,
            "copy_steps": copy_steps,
            "lm_steps": lm_steps,
            "include_scaleup": bool(getattr(args, "include_scaleup", False)),
            "scaleup_budget": int(getattr(args, "scaleup_budget", 3_000_000)),
            "dataset_source": workloads.dataset_source,
            "evaluation_counts": workloads.item_counts,
            "unigram_baselines": stage0["unigram_baselines"],
        },
        measurements=measurements,
        artifacts={
            "manifest": str(out_manifest),
            "selection_report": str(out_report),
            "tokenizer": str(bpe_path),
        },
    )
    write_report(evidence_doc, out_evidence)

    md_text = _render_selection_markdown(
        manifest=manifest,
        stage0=stage0,
        core_records=core_records,
        scaleup_records=scaleup_records,
        obj_records=obj_records,
        tok_records=tok_records,
        dec_records=dec_records,
        rt_records=rt_records,
        mem_records=mem_records,
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        stage4=stage4,
        equal_tables=equal_tables,
    )
    _atomic_write_text(out_report, md_text)
    print(
        f"[OK] Selected RecaMem Core ({manifest['selection_status']}): {manifest['selected_architecture']} -> "
        f"{out_manifest}, {out_evidence}, {out_report}",
        flush=True,
    )
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 4-stage RecaMem Core Architecture Tournament.")
    parser.add_argument("--tier", choices=sorted(TIER_BUDGETS.keys()), default="micro", help="Active tournament parameter tier.")
    parser.add_argument("--budget", type=int, default=None, help="Optional exact integer trainable-parameter budget override.")
    parser.add_argument("--scale", choices=["auto", "compact", "large"], default="auto", help="Benchmark split scale (`large` enforces >=1,000 items).")
    parser.add_argument("--seeds", type=str, default="1337,2026,4242", help="Comma-separated random seeds for multi-seed evaluation.")
    parser.add_argument("--seed", type=int, default=1337, help="Base workload generation seed.")
    parser.add_argument("--mqar-steps", type=int, default=560, help="Training steps per core on MQAR recall.")
    parser.add_argument("--copy-steps", type=int, default=120, help="Training steps per core on exact delayed copy.")
    parser.add_argument("--lm-steps", type=int, default=160, help="Training steps per arm on language modeling.")
    parser.add_argument("--include-scaleup", action="store_true", help="Also run Track 1b ~3.0M scale-up across C4, C3, and C6.")
    parser.add_argument("--scaleup-budget", type=int, default=3_000_000, help="Parameter budget for Track 1b scale-up comparison.")
    parser.add_argument("--scaleup-lm-steps", type=int, default=220, help="LM training steps for Track 1b scale-up models.")
    parser.add_argument("--output-manifest", type=str, default=str(REPO_ROOT / "configs" / "recamem_selected_core.json"))
    parser.add_argument("--output-evidence", type=str, default=str(REPO_ROOT / "reports" / "evidence" / "recamem-core-tournament.json"))
    parser.add_argument("--output-report", type=str, default=str(REPO_ROOT / "reports" / "recamem-core-selection.md"))
    parser.add_argument("--checkpoint-dir", type=str, default=str(REPO_ROOT / "reports" / "evidence" / "tournament_stages"))
    parser.add_argument("--resume", action="store_true", help="Resume from completed per-core/per-stage checkpoints.")
    parser.add_argument("--real-data-dir", type=str, default=None, help="Optional directory with real train/val_cal/test_eval JSONL corpora.")
    parser.add_argument("--use-tinystories", action="store_true", help="Provision/use verified zero-contamination 5 MiB TinyStories splits.")
    parser.add_argument("--require-real-data", action="store_true", help="Exit blocked instead of using offline fixtures if real data is missing.")
    parser.add_argument("--dry-run", action="store_true", help="Verify parameter budgets and workload splits without running training.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run_tournament(parse_args()))
