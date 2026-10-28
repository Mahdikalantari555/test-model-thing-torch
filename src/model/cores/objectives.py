"""Track-2 training auxiliary objectives: O1 (`ce_eos`), O2 (`repo_four_loss`), O3 (`ema_latent_vicreg`).

All auxiliary heads and EMA buffers live strictly on the objective instance rather than
on `CandidateLM`, guaranteeing zero extra parameters and zero extra FLOPs during `eval()`.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.cores.contracts import CandidateLM


def compute_shifted_ce(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    pad_id: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """Standard causal next-token cross-entropy (`logits[:, :-1]` -> `input_ids[:, 1:]`)."""
    if input_ids.shape[1] < 2:
        return logits.sum() * 0.0
    pred = logits[:, :-1, :].contiguous()
    targ = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        pred.view(-1, pred.shape[-1]),
        targ.view(-1),
        ignore_index=int(pad_id),
        reduction=reduction,
    )


class PureCEObjective(nn.Module):
    """`O1` (`ce_eos`): pure next-token cross-entropy + explicit EOS prediction."""

    objective_id = "O1"

    def __init__(self) -> None:
        super().__init__()

    def compute_loss(
        self,
        *,
        model: CandidateLM,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        raw_embeds: torch.Tensor,
        pad_id: int,
        eos_id: int,
        raw_hidden_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del model, hidden_states, raw_embeds, eos_id, raw_hidden_states
        ce = compute_shifted_ce(logits, input_ids, pad_id=pad_id)
        zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
        return {
            "ce_loss": ce,
            "aux_loss": zero,
            "total_loss": ce,
        }

    @torch.no_grad()
    def on_step_end(self, model: CandidateLM) -> None:
        del model


class RepoFourLossObjective(nn.Module):
    """`O2` (`repo_four_loss`): CE + feature variance hinge + stop-grad next-embed MSE + stop-head MSE."""

    objective_id = "O2"

    def __init__(
        self,
        d_model: int,
        lambda_var: float = 0.08,
        lambda_pred: float = 0.08,
        lambda_stop: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.lambda_var = float(lambda_var)
        self.lambda_pred = float(lambda_pred)
        self.lambda_stop = float(lambda_stop)
        self.stop_probe = nn.Linear(self.d_model, 1, bias=True)

    def compute_loss(
        self,
        *,
        model: CandidateLM,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        raw_embeds: torch.Tensor,
        pad_id: int,
        eos_id: int,
        raw_hidden_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        ce = compute_shifted_ce(logits, input_ids, pad_id=pad_id)
        if not self.training or input_ids.shape[1] < 2:
            zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
            return {"ce_loss": ce, "aux_loss": zero, "total_loss": ce}

        z_src = raw_hidden_states if raw_hidden_states is not None else hidden_states
        valid_mask = (input_ids != int(pad_id)).float()
        # 1. Per-token feature std hinge on pre-norm residual stream (non-degenerate before RMSNorm!)
        token_std = torch.sqrt(z_src.var(dim=-1, unbiased=False) + 1e-4)
        var_loss = (F.relu(1.0 - token_std) * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

        # 2. Next raw-embedding MSE with stop-gradient target
        pred_loss = F.mse_loss(hidden_states[:, :-1, :], raw_embeds[:, 1:, :].detach())

        # 3. Stop probability MSE on EOS token positions
        stop_logits = (
            model.stop_head(hidden_states[:, :-1, :])
            if model.stop_head is not None
            else self.stop_probe(hidden_states[:, :-1, :])
        )
        stop_prob = torch.sigmoid(stop_logits).squeeze(-1)
        is_eos = (input_ids[:, 1:] == int(eos_id)).to(stop_prob.dtype)
        stop_loss = F.mse_loss(stop_prob, is_eos)

        aux = (
            self.lambda_var * var_loss
            + self.lambda_pred * pred_loss
            + self.lambda_stop * stop_loss
        )
        return {
            "ce_loss": ce,
            "var_loss": var_loss.detach(),
            "pred_loss": pred_loss.detach(),
            "stop_loss": stop_loss.detach(),
            "aux_loss": aux,
            "total_loss": ce + aux,
        }

    @torch.no_grad()
    def on_step_end(self, model: CandidateLM) -> None:
        del model


class EMALatentVICRegObjective(nn.Module):
    """`O3` (`ema_latent_vicreg`): Candidate C EMA next-chunk predictor + VICReg regularizer.

    Applies BYOL/JEPA-style online projector (`target_proj`), EMA teacher projector
    (`ema_target_proj`), and future-chunk predictor (`predictor`) directly to the
    un-normalized backbone residual stream (`raw_hidden_states`), while regularizing
    per-dimension variance (`std >= 1.0`) and off-diagonal covariance on `online_z`
    so representation variance/decorrelation gradients flow cleanly into the core.
    """

    objective_id = "O3"

    def __init__(
        self,
        d_model: int,
        chunk_horizon: int = 8,
        ema_decay: float = 0.99,
        lambda_jepa: float = 0.25,
        lambda_var: float = 0.15,
        lambda_cov: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.chunk_horizon = int(chunk_horizon)
        self.ema_decay = float(ema_decay)
        self.lambda_jepa = float(lambda_jepa)
        self.lambda_var = float(lambda_var)
        self.lambda_cov = float(lambda_cov)

        self.target_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.ema_target_proj = copy.deepcopy(self.target_proj)
        for p in self.ema_target_proj.parameters():
            p.requires_grad_(False)

        self.predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=False),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model, bias=False),
        )

    def compute_loss(
        self,
        *,
        model: CandidateLM,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        raw_embeds: torch.Tensor,
        pad_id: int,
        eos_id: int,
        raw_hidden_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del model, raw_embeds, eos_id
        ce = compute_shifted_ce(logits, input_ids, pad_id=pad_id)
        if not self.training or input_ids.shape[1] < 4:
            zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
            return {"ce_loss": ce, "aux_loss": zero, "total_loss": ce}

        z_src = raw_hidden_states if raw_hidden_states is not None else hidden_states
        _, seq_len, d = z_src.shape
        online_all = self.target_proj(z_src)  # (B, T, d)
        valid = input_ids != int(pad_id)
        flat_z = online_all[valid]  # (N_valid, d)

        # VICReg variance + off-diagonal covariance across batch/time projected latents
        if flat_z.shape[0] > 4:
            centered = flat_z - flat_z.mean(dim=0, keepdim=True)
            std_per_dim = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
            var_loss = F.relu(1.0 - std_per_dim).mean()
            cov = (centered.transpose(0, 1) @ centered) / float(flat_z.shape[0] - 1)
            off_diag = cov - torch.diag(torch.diag(cov))
            cov_loss = off_diag.pow(2).sum() / float(max(1, d))
        else:
            var_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
            cov_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

        # Next-chunk latent cosine prediction: predictor(online_z[t]) -> stop_grad(mean_{k=1..W} ema_z[t+k])
        w = min(self.chunk_horizon, max(1, seq_len // 2))
        if seq_len > w:
            pred_future = self.predictor(online_all[:, :-w, :])  # (B, T - W, d)
            with torch.no_grad():
                ema_all = self.ema_target_proj(z_src.detach())  # (B, T, d)
                ema_t = ema_all[:, 1:, :].transpose(1, 2)  # (B, d, T-1)
                future_pool = F.avg_pool1d(ema_t, kernel_size=w, stride=1).transpose(1, 2)  # (B, T-W, d)
            cos_pred = F.cosine_similarity(pred_future, future_pool.detach(), dim=-1)
            jepa_loss = (1.0 - cos_pred).mean()
        else:
            jepa_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

        aux = (
            self.lambda_jepa * jepa_loss
            + self.lambda_var * var_loss
            + self.lambda_cov * cov_loss
        )
        return {
            "ce_loss": ce,
            "jepa_loss": jepa_loss.detach(),
            "var_loss": var_loss.detach(),
            "cov_loss": cov_loss.detach(),
            "aux_loss": aux,
            "total_loss": ce + aux,
        }

    @torch.no_grad()
    def on_step_end(self, model: CandidateLM) -> None:
        del model
        for p_ema, p_online in zip(self.ema_target_proj.parameters(), self.target_proj.parameters()):
            p_ema.mul_(self.ema_decay).add_(p_online.data, alpha=1.0 - self.ema_decay)


def build_objective(objective_id: str, d_model: int) -> nn.Module:
    key = objective_id.strip().upper()
    aliases = {
        "O1": "O1",
        "CE_EOS": "O1",
        "O2": "O2",
        "REPO_FOUR_LOSS": "O2",
        "O3": "O3",
        "EMA_LATENT_VICREG": "O3",
    }
    oid = aliases.get(key)
    if oid == "O1":
        return PureCEObjective()
    if oid == "O2":
        return RepoFourLossObjective(d_model=d_model)
    if oid == "O3":
        return EMALatentVICRegObjective(d_model=d_model)
    raise ValueError(f"Unknown objective_id {objective_id!r}; expected O1, O2, or O3")
