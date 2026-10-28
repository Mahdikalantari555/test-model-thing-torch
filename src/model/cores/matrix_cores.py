"""Track-1 matrix fast-weight cores: C3 (RetNet decay matrix) and C4 (Gated-Delta matrix).

Both implement exact chunkwise parallel scans (`chunk_size = 64`) backed by BLAS/LAPACK
primitives with zero intra-chunk Python token loops, paired with exact `O(1)` step
decode paths verified to match within `1e-4` in fp32.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.cores.contracts import CoreOutput, LayerMemoryHook, StateSpec
from src.model.cores.diagonal_cores import SwiGLUBlock


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps)


class _StatefulCausalConv(nn.Module):
    """Bias-free causal depthwise Conv1d (`d_model * conv_kernel` params) with exact cache."""

    def __init__(self, d_model: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(
            self.d_model,
            self.d_model,
            kernel_size=self.kernel_size,
            groups=self.d_model,
            bias=False,
        )
        with torch.no_grad():
            nn.init.normal_(self.conv.weight, mean=0.0, std=0.05)
            # Near-Dirac causal init on current & previous token so local key-value shift forms cleanly
            self.conv.weight[:, 0, -1] += 0.85
            if self.kernel_size >= 2:
                self.conv.weight[:, 0, -2] += 0.35

    def forward_with_cache(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, d), cache: (B, d, K-1)
        xt = x.transpose(1, 2)
        if self.kernel_size > 1:
            padded = torch.cat([cache, xt], dim=-1)
            next_cache = padded[:, :, -(self.kernel_size - 1):].contiguous()
        else:
            padded = xt
            next_cache = cache
        out = self.conv(padded).transpose(1, 2)
        return out, next_cache


class _RetNetDecayMatrixLayer(nn.Module):
    """`C3` pure-decay multi-head matrix recurrence without delta-rule overwrite."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads

        self.pre_norm = nn.RMSNorm(self.d_model, eps=1e-6)
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_g = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_norm = nn.RMSNorm(self.d_model, eps=1e-6)

        gammas = 1.0 - torch.logspace(math.log10(0.15), math.log10(0.005), self.n_heads, dtype=torch.float32)
        self.head_decay_logit = nn.Parameter(torch.log(gammas) - torch.log1p(-gammas))
        self.ffn = SwiGLUBlock(self.d_model, int(d_ff))

        for mod in (self.w_q, self.w_k, self.w_v, self.w_g, self.w_o):
            nn.init.normal_(mod.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))

    def forward_chunk(
        self,
        x: torch.Tensor,
        s_state: torch.Tensor,
        chunk_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # s_state: (B, H, d_v, d_k)
        batch, seq_len, _ = x.shape
        u = self.pre_norm(x)
        q = _l2norm(self.w_q(u).view(batch, seq_len, self.n_heads, self.head_dim)).transpose(1, 2)  # (B, H, T, d_k)
        k = _l2norm(self.w_k(u).view(batch, seq_len, self.n_heads, self.head_dim)).transpose(1, 2)
        v = (self.w_v(u).view(batch, seq_len, self.n_heads, self.head_dim) / math.sqrt(self.head_dim)).transpose(1, 2)

        gamma = torch.sigmoid(self.head_decay_logit)  # (H,)
        log_g = torch.log(gamma.clamp(min=1e-6))  # (H,)

        s = s_state
        outputs: List[torch.Tensor] = []
        for start in range(0, seq_len, chunk_size):
            stop = min(start + chunk_size, seq_len)
            c_len = stop - start
            qc = q[:, :, start:stop, :]  # (B, H, C, d_k)
            kc = k[:, :, start:stop, :]
            vc = v[:, :, start:stop, :]

            idx = torch.arange(c_len, device=x.device, dtype=x.dtype)
            diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # (C, C): i - j
            causal = diff >= 0
            # D[h, i, j] = gamma_h ** (i - j) for i >= j else 0
            decay_mat = torch.exp(log_g.view(self.n_heads, 1, 1) * diff.clamp(min=0.0).unsqueeze(0))
            decay_mat = torch.where(causal.unsqueeze(0), decay_mat, torch.zeros_like(decay_mat))

            # Cross-chunk contribution: gamma_h ** (i + 1) * (S @ q_i)
            cross_scale = torch.exp(log_g.view(1, self.n_heads, 1, 1) * (idx + 1.0).view(1, 1, c_len, 1))
            s_q = torch.matmul(qc, s.transpose(-1, -2))  # (B, H, C, d_v) since S is (d_v, d_k)
            cross_out = cross_scale * s_q

            # Intra-chunk contribution: ((Q K^T) * D) V
            qk = torch.matmul(qc, kc.transpose(-1, -2)) * decay_mat.unsqueeze(0)
            intra_out = torch.matmul(qk, vc)
            outputs.append(cross_out + intra_out)

            # Update S: gamma^C * S + sum_j gamma^(C - 1 - j) v_j k_j^T
            tail_scale = torch.exp(log_g.view(1, self.n_heads, 1, 1) * (c_len - 1.0 - idx).view(1, 1, c_len, 1))
            vc_scaled = vc * tail_scale
            s_total_decay = torch.exp(log_g.view(1, self.n_heads, 1, 1) * float(c_len))
            s = s_total_decay * s + torch.matmul(vc_scaled.transpose(-1, -2), kc)

        o = torch.cat(outputs, dim=2).transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        h = x + self.w_o(self.o_norm(o) * F.silu(self.w_g(u)))
        h = self.ffn(h)
        return h, s

    def step_one(self, x_t: torch.Tensor, s_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = x_t.shape[0]
        u_t = self.pre_norm(x_t)
        q_t = _l2norm(self.w_q(u_t).view(batch, self.n_heads, self.head_dim))
        k_t = _l2norm(self.w_k(u_t).view(batch, self.n_heads, self.head_dim))
        v_t = self.w_v(u_t).view(batch, self.n_heads, self.head_dim) / math.sqrt(self.head_dim)
        gamma = torch.sigmoid(self.head_decay_logit).view(1, self.n_heads, 1, 1)

        s_t = gamma * s_prev + torch.matmul(v_t.unsqueeze(-1), k_t.unsqueeze(-2))
        o_t = torch.matmul(s_t, q_t.unsqueeze(-1)).squeeze(-1).reshape(batch, self.d_model)
        h_t = x_t + self.w_o(self.o_norm(o_t) * F.silu(self.w_g(u_t)))
        h_t = self.ffn(h_t.unsqueeze(1)).squeeze(1)
        return h_t, s_t


class RetNetDecayMatrixCore(nn.Module):
    """`C3` (`retnet_decay_matrix`): multi-head decay fast-weight matrix recurrence."""

    core_id = "C3"
    supports_online_rtrl = False

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        d_ff: int = 512,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.d_ff = int(d_ff)
        self.chunk_size = int(chunk_size)

        self.layers = nn.ModuleList(
            [_RetNetDecayMatrixLayer(self.d_model, self.n_heads, self.d_ff) for _ in range(self.n_layers)]
        )
        self._persistent_state: Optional[torch.Tensor] = None

    def state_spec(self, batch_size: int = 1, seq_len: int = 1) -> StateSpec:
        del batch_size, seq_len
        floats = self.n_layers * self.n_heads * self.head_dim * self.head_dim
        return StateSpec(
            complexity_class="constant_matrix",
            bytes_per_sequence=floats * 4,
            shape_per_layer=(self.n_heads, self.head_dim, self.head_dim),
            num_layers=self.n_layers,
        )

    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        dev = device if device is not None else self.layers[0].w_q.weight.device
        return torch.zeros(
            self.n_layers,
            batch_size,
            self.n_heads,
            self.head_dim,
            self.head_dim,
            device=dev,
            dtype=dtype,
        )

    def reset_state(self) -> None:
        self._persistent_state = None

    def forward_sequence(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
        layer_hook: Optional[LayerMemoryHook] = None,
    ) -> CoreOutput:
        batch, seq_len, _ = x.shape
        if state is None:
            state = self.init_state(batch_size=batch, device=x.device, dtype=x.dtype)
        h = x
        next_s: List[torch.Tensor] = []
        for idx, layer in enumerate(self.layers):
            h, s_last = layer.forward_chunk(h, state[idx], chunk_size=self.chunk_size)
            if layer_hook is not None:
                h = layer_hook(idx, h, token_ids)
            next_s.append(s_last)
        out_state = torch.stack(next_s, dim=0)
        self._persistent_state = out_state.detach()
        return CoreOutput(
            hidden_states=h,
            state=out_state,
            state_spec=self.state_spec(batch_size=batch, seq_len=seq_len),
        )

    def step(
        self,
        x_t: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        token_id: Optional[torch.Tensor] = None,
        layer_hook: Optional[LayerMemoryHook] = None,
    ) -> CoreOutput:
        batch = x_t.shape[0]
        if state is None:
            state = self.init_state(batch_size=batch, device=x_t.device, dtype=x_t.dtype)
        h = x_t
        next_s: List[torch.Tensor] = []
        tok_seq = token_id.unsqueeze(1) if token_id is not None else None
        for idx, layer in enumerate(self.layers):
            h, s_t = layer.step_one(h, state[idx])
            if layer_hook is not None:
                h = layer_hook(idx, h.unsqueeze(1), tok_seq).squeeze(1)
            next_s.append(s_t)
        out_state = torch.stack(next_s, dim=0)
        self._persistent_state = out_state.detach()
        return CoreOutput(
            hidden_states=h.unsqueeze(1),
            state=out_state,
            state_spec=self.state_spec(batch_size=batch, seq_len=1),
        )


class _GatedDeltaMatrixLayer(nn.Module):
    """Candidate A Gated-DeltaNet layer with exact unitriangular WY chunkwise scan."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        use_conv: bool = True,
        conv_kernel: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.use_conv = bool(use_conv)
        self.conv_kernel = int(conv_kernel)

        self.pre_norm = nn.RMSNorm(self.d_model, eps=1e-6)
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_g = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

        if self.use_conv:
            self.conv_q = _StatefulCausalConv(self.d_model, self.conv_kernel)
            self.conv_k = _StatefulCausalConv(self.d_model, self.conv_kernel)
            self.conv_v = _StatefulCausalConv(self.d_model, self.conv_kernel)
        else:
            self.conv_q = None
            self.conv_k = None
            self.conv_v = None

        self.w_alpha = nn.Linear(self.d_model, self.n_heads, bias=True)
        self.w_beta = nn.Linear(self.d_model, self.n_heads, bias=True)
        self.o_norm = nn.RMSNorm(self.d_model, eps=1e-6)
        self.ffn = SwiGLUBlock(self.d_model, int(d_ff))

        for mod in (self.w_q, self.w_k, self.w_v, self.w_g, self.w_o):
            nn.init.normal_(mod.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        nn.init.normal_(self.w_alpha.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_beta.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            # High initial retention (~0.92..0.99) + moderate write gate (~0.5)
            alphas = torch.linspace(2.4, 4.2, self.n_heads)
            self.w_alpha.bias.copy_(alphas)
            self.w_beta.bias.fill_(0.0)

    def _project_qkv(
        self,
        u: torch.Tensor,
        conv_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = u.shape
        q_raw = self.w_q(u)
        k_raw = self.w_k(u)
        v_raw = self.w_v(u)
        if self.use_conv and self.conv_q is not None and self.conv_k is not None and self.conv_v is not None:
            q_raw, cq = self.conv_q.forward_with_cache(q_raw, conv_cache[:, 0])
            k_raw, ck = self.conv_k.forward_with_cache(k_raw, conv_cache[:, 1])
            v_raw, cv = self.conv_v.forward_with_cache(v_raw, conv_cache[:, 2])
            next_cache = torch.stack([cq, ck, cv], dim=1)
        else:
            next_cache = conv_cache

        q = _l2norm(q_raw.view(batch, seq_len, self.n_heads, self.head_dim)).transpose(1, 2)
        k = _l2norm(k_raw.view(batch, seq_len, self.n_heads, self.head_dim)).transpose(1, 2)
        v = F.silu(v_raw).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        return q, k, v, next_cache

    def forward_chunk(
        self,
        x: torch.Tensor,
        s_state: torch.Tensor,
        conv_cache: torch.Tensor,
        chunk_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact WY/UT chunkwise parallel gated-delta recurrence (`chunk_size = 64`)."""
        batch, seq_len, _ = x.shape
        u = self.pre_norm(x)
        q, k, v, next_cache = self._project_qkv(u, conv_cache)
        alpha = torch.sigmoid(self.w_alpha(u)).transpose(1, 2)  # (B, H, T)
        beta = torch.sigmoid(self.w_beta(u)).transpose(1, 2)    # (B, H, T)

        log_alpha = torch.log(alpha.clamp(min=1e-6))
        s = s_state  # (B, H, d_v, d_k)
        outputs: List[torch.Tensor] = []

        for start in range(0, seq_len, chunk_size):
            stop = min(start + chunk_size, seq_len)
            c_len = stop - start
            qc = q[:, :, start:stop, :]          # (B, H, C, d_k)
            kc = k[:, :, start:stop, :]          # (B, H, C, d_k)
            vc = v[:, :, start:stop, :]          # (B, H, C, d_v)
            lac = log_alpha[:, :, start:stop]    # (B, H, C)
            bc = beta[:, :, start:stop]          # (B, H, C)

            # Prefix log-decays: cum[t] = sum_{s=0..t} log(alpha_s)
            cum_la = torch.cumsum(lac, dim=-1)  # (B, H, C)
            gamma_prefix = torch.exp(cum_la).unsqueeze(-1)  # (B, H, C, 1)

            # Pairwise decay Gamma_{t:j} = prod_{s=j+1..t} alpha_s = exp(cum_la[t] - cum_la[j]) for t >= j
            idx = torch.arange(c_len, device=x.device)
            strict_lower = idx.unsqueeze(1) > idx.unsqueeze(0)   # (C, C): t > j
            lower_incl = idx.unsqueeze(1) >= idx.unsqueeze(0)    # (C, C): t >= j

            pair_log = cum_la.unsqueeze(-1) - cum_la.unsqueeze(-2)  # (B, H, C, C)
            # Mask upper triangle BEFORE exp() so large positive differences for j > t never overflow fp32!
            pair_log_safe = torch.where(lower_incl.view(1, 1, c_len, c_len), pair_log, torch.zeros_like(pair_log))
            gamma_pair = torch.exp(pair_log_safe)

            # Strictly lower-triangular WY coupling matrix:
            # L[t, j] = beta_t * Gamma_{t:j} * (k_j^T k_t) for t > j
            kk = torch.matmul(kc, kc.transpose(-1, -2))  # (B, H, C, C)
            l_mat = bc.unsqueeze(-1) * gamma_pair * kk
            l_mat = torch.where(strict_lower.view(1, 1, c_len, c_len), l_mat, torch.zeros_like(l_mat))
            eye = torch.eye(c_len, device=x.device, dtype=x.dtype).view(1, 1, c_len, c_len)

            # RHS: B_rhs[t] = beta_t * (v_t - Gamma_t * (S_0 k_t))
            s0_k = torch.matmul(kc, s.transpose(-1, -2))  # (B, H, C, d_v)
            rhs = bc.unsqueeze(-1) * (vc - gamma_prefix * s0_k)

            # Solve (I + L) R_tilde = rhs cleanly via unitriangular forward substitution
            r_tilde = torch.linalg.solve_triangular(eye + l_mat, rhs, upper=False, unitriangular=True)

            # Readout o_t = Gamma_t (S_0 q_t) + sum_{j <= t} Gamma_{t:j} (q_t^T k_j) r_tilde_j
            s0_q = torch.matmul(qc, s.transpose(-1, -2))  # (B, H, C, d_v)
            qk = torch.matmul(qc, kc.transpose(-1, -2)) * torch.where(
                lower_incl.view(1, 1, c_len, c_len), gamma_pair, torch.zeros_like(gamma_pair)
            )
            oc = gamma_prefix * s0_q + torch.matmul(qk, r_tilde)
            outputs.append(oc)

            # End-of-chunk state S_{C-1} = Gamma_{C-1} S_0 + sum_{j=0..C-1} Gamma_{C-1:j} r_tilde_j k_j^T
            tail_gamma = gamma_pair[:, :, -1, :].unsqueeze(-1)  # (B, H, C, 1)
            total_gamma = gamma_prefix[:, :, -1:, :]            # (B, H, 1, 1)
            s = total_gamma * s + torch.matmul((r_tilde * tail_gamma).transpose(-1, -2), kc)

        o = torch.cat(outputs, dim=2).transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        h = x + self.w_o(self.o_norm(o) * F.silu(self.w_g(u)))
        h = self.ffn(h)
        return h, s, next_cache

    def step_one(
        self,
        x_t: torch.Tensor,
        s_prev: torch.Tensor,
        conv_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact single-token O(1) recurrent step: S_t = alpha_t S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T."""
        batch = x_t.shape[0]
        u_seq = self.pre_norm(x_t).unsqueeze(1)
        q, k, v, next_cache = self._project_qkv(u_seq, conv_cache)
        q_t = q[:, :, 0, :]  # (B, H, d_k)
        k_t = k[:, :, 0, :]
        v_t = v[:, :, 0, :]
        u_t = u_seq[:, 0, :]

        alpha_t = torch.sigmoid(self.w_alpha(u_t)).view(batch, self.n_heads, 1, 1)
        beta_t = torch.sigmoid(self.w_beta(u_t)).view(batch, self.n_heads, 1)

        s_decayed = alpha_t * s_prev
        s_k = torch.matmul(s_decayed, k_t.unsqueeze(-1)).squeeze(-1)  # (B, H, d_v)
        r_t = beta_t * (v_t - s_k)                                    # (B, H, d_v)
        s_t = s_decayed + torch.matmul(r_t.unsqueeze(-1), k_t.unsqueeze(-2))
        o_t = torch.matmul(s_t, q_t.unsqueeze(-1)).squeeze(-1).reshape(batch, self.d_model)

        h_t = x_t + self.w_o(self.o_norm(o_t) * F.silu(self.w_g(u_t)))
        h_t = self.ffn(h_t.unsqueeze(1)).squeeze(1)
        return h_t, s_t, next_cache


class GatedDeltaMatrixCore(nn.Module):
    """`C4` (`gated_delta_matrix`): Candidate A multi-head gated-delta recurrent core."""

    core_id = "C4"
    supports_online_rtrl = False

    def __init__(
        self,
        d_model: int = 384,
        n_layers: int = 8,
        n_heads: int = 6,
        d_ff: int = 1024,
        use_conv: bool = True,
        conv_kernel: int = 4,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.d_ff = int(d_ff)
        self.use_conv = bool(use_conv)
        self.conv_kernel = int(conv_kernel)
        self.chunk_size = int(chunk_size)

        self.layers = nn.ModuleList(
            [
                _GatedDeltaMatrixLayer(
                    self.d_model,
                    self.n_heads,
                    self.d_ff,
                    use_conv=self.use_conv,
                    conv_kernel=self.conv_kernel,
                )
                for _ in range(self.n_layers)
            ]
        )
        self._persistent_state: Optional[Dict[str, torch.Tensor]] = None

    def state_spec(self, batch_size: int = 1, seq_len: int = 1) -> StateSpec:
        del batch_size, seq_len
        mat_floats = self.n_heads * self.head_dim * self.head_dim
        conv_floats = 3 * self.d_model * max(0, self.conv_kernel - 1) if self.use_conv else 0
        return StateSpec(
            complexity_class="constant_matrix",
            bytes_per_sequence=self.n_layers * (mat_floats + conv_floats) * 4,
            shape_per_layer=(self.n_heads, self.head_dim, self.head_dim),
            num_layers=self.n_layers,
        )

    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, torch.Tensor]:
        dev = device if device is not None else self.layers[0].w_q.weight.device
        conv_len = max(0, self.conv_kernel - 1) if self.use_conv else 0
        return {
            "matrix": torch.zeros(
                self.n_layers,
                batch_size,
                self.n_heads,
                self.head_dim,
                self.head_dim,
                device=dev,
                dtype=dtype,
            ),
            "conv": torch.zeros(
                self.n_layers,
                batch_size,
                3,
                self.d_model,
                conv_len,
                device=dev,
                dtype=dtype,
            ),
        }

    def reset_state(self) -> None:
        self._persistent_state = None

    def forward_sequence(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        token_ids: Optional[torch.Tensor] = None,
        layer_hook: Optional[LayerMemoryHook] = None,
    ) -> CoreOutput:
        batch, seq_len, _ = x.shape
        if state is None:
            state = self.init_state(batch_size=batch, device=x.device, dtype=x.dtype)
        h = x
        next_m: List[torch.Tensor] = []
        next_c: List[torch.Tensor] = []
        for idx, layer in enumerate(self.layers):
            h, s_last, c_last = layer.forward_chunk(
                h,
                state["matrix"][idx],
                state["conv"][idx],
                chunk_size=self.chunk_size,
            )
            if layer_hook is not None:
                h = layer_hook(idx, h, token_ids)
            next_m.append(s_last)
            next_c.append(c_last)
        out_state = {
            "matrix": torch.stack(next_m, dim=0),
            "conv": torch.stack(next_c, dim=0),
        }
        self._persistent_state = {k: v.detach() for k, v in out_state.items()}
        return CoreOutput(
            hidden_states=h,
            state=out_state,
            state_spec=self.state_spec(batch_size=batch, seq_len=seq_len),
        )

    def step(
        self,
        x_t: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        token_id: Optional[torch.Tensor] = None,
        layer_hook: Optional[LayerMemoryHook] = None,
    ) -> CoreOutput:
        batch = x_t.shape[0]
        if state is None:
            state = self.init_state(batch_size=batch, device=x_t.device, dtype=x_t.dtype)
        h = x_t
        next_m: List[torch.Tensor] = []
        next_c: List[torch.Tensor] = []
        tok_seq = token_id.unsqueeze(1) if token_id is not None else None
        for idx, layer in enumerate(self.layers):
            h, s_t, c_t = layer.step_one(h, state["matrix"][idx], state["conv"][idx])
            if layer_hook is not None:
                h = layer_hook(idx, h.unsqueeze(1), tok_seq).squeeze(1)
            next_m.append(s_t)
            next_c.append(c_t)
        out_state = {
            "matrix": torch.stack(next_m, dim=0),
            "conv": torch.stack(next_c, dim=0),
        }
        self._persistent_state = {k: v.detach() for k, v in out_state.items()}
        return CoreOutput(
            hidden_states=h.unsqueeze(1),
            state=out_state,
            state_spec=self.state_spec(batch_size=batch, seq_len=1),
        )
