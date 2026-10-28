"""Track-1 attention baselines: C5 (Needle SAN sliding-window GQA) and C6 (Softmax Transformer)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.cores.contracts import CoreOutput, LayerMemoryHook, StateSpec
from src.model.cores.diagonal_cores import SwiGLUBlock


def _apply_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to `x` of shape `(B, H, T, D)`."""
    head_dim = x.shape[-1]
    half = head_dim // 2
    if half == 0:
        return x
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / float(half))
    )
    angles = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)  # (T, half)
    cos = torch.cos(angles).view(1, 1, -1, half).to(x.dtype)
    sin = torch.sin(angles).view(1, 1, -1, half).to(x.dtype)
    x1 = x[..., :half]
    x2 = x[..., half: 2 * half]
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    if 2 * half < head_dim:
        rotated = torch.cat([rotated, x[..., 2 * half:]], dim=-1)
    return rotated


class _HadamardChannelMixer(nn.Module):
    """Needle-style Hadamard elementwise product MLP (`d_model + 3 * d_model * d_ff` params)."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=1e-6)
        self.w_1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_2 = nn.Linear(d_model, d_ff, bias=False)
        self.w_out = nn.Linear(d_ff, d_model, bias=False)
        for mod in (self.w_1, self.w_2, self.w_out):
            nn.init.normal_(mod.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.norm(x)
        # Pure Hadamard bilinear gating (no activation distortion on branch 1, light SiLU stability)
        return x + self.w_out((self.w_1(u) / math.sqrt(self.w_1.out_features)) * F.silu(self.w_2(u)))


class _NeedleSANLayer(nn.Module):
    """Sliding-window GQA + QK-RMSNorm + pinned sink prefix KV + Hadamard mixer (`C5`)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        window_size: int = 256,
        sink_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.n_kv_heads = max(1, self.n_heads // 2)
        # Ensure n_heads divisible by n_kv_heads
        while self.n_heads % self.n_kv_heads != 0 and self.n_kv_heads > 1:
            self.n_kv_heads -= 1
        self.group_size = self.n_heads // self.n_kv_heads
        self.d_kv = self.n_kv_heads * self.head_dim
        self.window_size = int(window_size)
        self.sink_tokens = int(sink_tokens)

        self.pre_norm = nn.RMSNorm(self.d_model, eps=1e-6)
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_kv, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_kv, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.mixer = _HadamardChannelMixer(self.d_model, int(d_ff))

        for mod in (self.w_q, self.w_k, self.w_v, self.w_o):
            nn.init.normal_(mod.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))

    def forward_with_cache(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch, seq_len, _ = x.shape
        u = self.pre_norm(x)
        past_len = int(kv_cache["pos"]) if kv_cache is not None else 0
        q_pos = torch.arange(past_len, past_len + seq_len, device=x.device)

        q = self.q_norm(self.w_q(u).view(batch, seq_len, self.n_heads, self.head_dim)).transpose(1, 2)
        k_new = self.k_norm(self.w_k(u).view(batch, seq_len, self.n_kv_heads, self.head_dim)).transpose(1, 2)
        v_new = self.w_v(u).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, q_pos)
        k_new = _apply_rope(k_new, q_pos)

        if kv_cache is not None and kv_cache["k"].shape[2] > 0:
            k_all = torch.cat([kv_cache["k"], k_new], dim=2)
            v_all = torch.cat([kv_cache["v"], v_new], dim=2)
            pos_all = torch.cat([kv_cache["positions"], q_pos], dim=0)
        else:
            k_all = k_new
            v_all = v_new
            pos_all = q_pos

        if self.group_size > 1:
            k_exp = k_all.repeat_interleave(self.group_size, dim=1)
            v_exp = v_all.repeat_interleave(self.group_size, dim=1)
        else:
            k_exp, v_exp = k_all, v_all

        # Mask: causal (pos_k <= pos_q) AND (is_sink(pos_k < sink_tokens) OR pos_q - pos_k < window_size)
        pq = q_pos.unsqueeze(1)    # (T, 1)
        pk = pos_all.unsqueeze(0)  # (1, K_len)
        causal = pk <= pq
        in_window = (pq - pk) < self.window_size
        is_sink = pk < self.sink_tokens
        allowed = causal & (in_window | is_sink)

        scores = torch.matmul(q, k_exp.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allowed.view(1, 1, seq_len, -1), float("-inf"))
        probs = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v_exp).transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

        # Compact cache to bounded O(sink_tokens + window_size)
        max_keep = self.sink_tokens + self.window_size
        if k_all.shape[2] > max_keep:
            sink_k = k_all[:, :, :self.sink_tokens, :]
            sink_v = v_all[:, :, :self.sink_tokens, :]
            sink_p = pos_all[:self.sink_tokens]
            tail_len = self.window_size
            k_kept = torch.cat([sink_k, k_all[:, :, -tail_len:, :]], dim=2)
            v_kept = torch.cat([sink_v, v_all[:, :, -tail_len:, :]], dim=2)
            p_kept = torch.cat([sink_p, pos_all[-tail_len:]], dim=0)
        else:
            k_kept, v_kept, p_kept = k_all, v_all, pos_all

        next_cache = {
            "k": k_kept,
            "v": v_kept,
            "positions": p_kept,
            "pos": past_len + seq_len,
        }
        h = x + self.w_o(attn_out)
        h = self.mixer(h)
        return h, next_cache


class NeedleSANCore(nn.Module):
    """`C5` (`needle_san`): Needle-style sliding-window QK-norm attention with pinned KV sinks."""

    core_id = "C5"
    supports_online_rtrl = False

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        d_ff: int = 512,
        window_size: int = 256,
        sink_tokens: int = 4,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.d_ff = int(d_ff)
        self.window_size = int(window_size)
        self.sink_tokens = int(sink_tokens)

        self.layers = nn.ModuleList(
            [
                _NeedleSANLayer(
                    self.d_model,
                    self.n_heads,
                    self.d_ff,
                    window_size=self.window_size,
                    sink_tokens=self.sink_tokens,
                )
                for _ in range(self.n_layers)
            ]
        )
        self._persistent_state: Optional[List[Dict[str, Any]]] = None

    def state_spec(self, batch_size: int = 1, seq_len: int = 1) -> StateSpec:
        del batch_size
        bounded = min(max(1, seq_len), self.window_size + self.sink_tokens)
        d_kv = self.layers[0].d_kv
        bytes_per_seq = self.n_layers * 2 * bounded * d_kv * 4
        return StateSpec(
            complexity_class="sliding_window_kv",
            bytes_per_sequence=bytes_per_seq,
            shape_per_layer=(2, self.window_size + self.sink_tokens, d_kv),
            num_layers=self.n_layers,
            window_size=self.window_size,
        )

    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> List[Dict[str, Any]]:
        dev = device if device is not None else self.layers[0].w_q.weight.device
        n_kv = self.layers[0].n_kv_heads
        return [
            {
                "k": torch.zeros(batch_size, n_kv, 0, self.head_dim, device=dev, dtype=dtype),
                "v": torch.zeros(batch_size, n_kv, 0, self.head_dim, device=dev, dtype=dtype),
                "positions": torch.zeros(0, device=dev, dtype=torch.long),
                "pos": 0,
            }
            for _ in range(self.n_layers)
        ]

    def reset_state(self) -> None:
        self._persistent_state = None

    def forward_sequence(
        self,
        x: torch.Tensor,
        state: Optional[List[Dict[str, Any]]] = None,
        token_ids: Optional[torch.Tensor] = None,
        layer_hook: Optional[LayerMemoryHook] = None,
    ) -> CoreOutput:
        batch, seq_len, _ = x.shape
        if state is None:
            state = self.init_state(batch_size=batch, device=x.device, dtype=x.dtype)
        h = x
        next_state: List[Dict[str, Any]] = []
        for idx, layer in enumerate(self.layers):
            h, c_next = layer.forward_with_cache(h, state[idx])
            if layer_hook is not None:
                h = layer_hook(idx, h, token_ids)
            next_state.append(c_next)
        total_cached = next_state[0]["k"].shape[2]
        return CoreOutput(
            hidden_states=h,
            state=next_state,
            state_spec=self.state_spec(batch_size=batch, seq_len=total_cached),
        )

    def step(
        self,
        x_t: torch.Tensor,
        state: Optional[List[Dict[str, Any]]] = None,
        token_id: Optional[torch.Tensor] = None,
        layer_hook: Optional[LayerMemoryHook] = None,
    ) -> CoreOutput:
        tok_seq = token_id.unsqueeze(1) if token_id is not None else None
        return self.forward_sequence(x_t.unsqueeze(1), state=state, token_ids=tok_seq, layer_hook=layer_hook)


class _SoftmaxTransformerLayer(nn.Module):
    """Standard causal MHA + RoPE + QK-RMSNorm + SwiGLU block (`C6`)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads

        self.pre_norm = nn.RMSNorm(self.d_model, eps=1e-6)
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.ffn = SwiGLUBlock(self.d_model, int(d_ff))

        for mod in (self.w_q, self.w_k, self.w_v, self.w_o):
            nn.init.normal_(mod.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))

    def forward_with_cache(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch, seq_len, _ = x.shape
        u = self.pre_norm(x)
        past_len = int(kv_cache["k"].shape[2]) if kv_cache is not None else 0
        q_pos = torch.arange(past_len, past_len + seq_len, device=x.device)

        q = self.q_norm(self.w_q(u).view(batch, seq_len, self.n_heads, self.head_dim)).transpose(1, 2)
        k_new = self.k_norm(self.w_k(u).view(batch, seq_len, self.n_heads, self.head_dim)).transpose(1, 2)
        v_new = self.w_v(u).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, q_pos)
        k_new = _apply_rope(k_new, q_pos)

        if kv_cache is not None and past_len > 0:
            k_all = torch.cat([kv_cache["k"], k_new], dim=2)
            v_all = torch.cat([kv_cache["v"], v_new], dim=2)
        else:
            k_all, v_all = k_new, v_new

        attn_out = F.scaled_dot_product_attention(
            q,
            k_all,
            v_all,
            is_causal=(past_len == 0 and seq_len > 1),
            attn_mask=(
                None
                if (past_len == 0 or seq_len == 1)
                else (
                    torch.arange(past_len + seq_len, device=x.device).unsqueeze(0)
                    <= q_pos.unsqueeze(1)
                )
            ),
        )
        merged = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        h = x + self.w_o(merged)
        h = self.ffn(h)
        return h, {"k": k_all, "v": v_all}


class SoftmaxTransformerCore(nn.Module):
    """`C6` (`softmax_transformer`): parameter-matched standard causal Softmax Transformer."""

    core_id = "C6"
    supports_online_rtrl = False

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        d_ff: int = 512,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.d_ff = int(d_ff)

        self.layers = nn.ModuleList(
            [_SoftmaxTransformerLayer(self.d_model, self.n_heads, self.d_ff) for _ in range(self.n_layers)]
        )
        self._persistent_state: Optional[List[Dict[str, Any]]] = None

    def state_spec(self, batch_size: int = 1, seq_len: int = 1) -> StateSpec:
        del batch_size
        t_eff = max(1, int(seq_len))
        bytes_per_seq = self.n_layers * 2 * t_eff * self.d_model * 4
        return StateSpec(
            complexity_class="linear_kv",
            bytes_per_sequence=bytes_per_seq,
            shape_per_layer=(2, t_eff, self.d_model),
            num_layers=self.n_layers,
        )

    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> List[Dict[str, Any]]:
        dev = device if device is not None else self.layers[0].w_q.weight.device
        return [
            {
                "k": torch.zeros(batch_size, self.n_heads, 0, self.head_dim, device=dev, dtype=dtype),
                "v": torch.zeros(batch_size, self.n_heads, 0, self.head_dim, device=dev, dtype=dtype),
            }
            for _ in range(self.n_layers)
        ]

    def reset_state(self) -> None:
        self._persistent_state = None

    def forward_sequence(
        self,
        x: torch.Tensor,
        state: Optional[List[Dict[str, Any]]] = None,
        token_ids: Optional[torch.Tensor] = None,
        layer_hook: Optional[LayerMemoryHook] = None,
    ) -> CoreOutput:
        batch, seq_len, _ = x.shape
        if state is None:
            state = self.init_state(batch_size=batch, device=x.device, dtype=x.dtype)
        h = x
        next_state: List[Dict[str, Any]] = []
        for idx, layer in enumerate(self.layers):
            h, c_next = layer.forward_with_cache(h, state[idx])
            if layer_hook is not None:
                h = layer_hook(idx, h, token_ids)
            next_state.append(c_next)
        total_len = next_state[0]["k"].shape[2]
        return CoreOutput(
            hidden_states=h,
            state=next_state,
            state_spec=self.state_spec(batch_size=batch, seq_len=total_len),
        )

    def step(
        self,
        x_t: torch.Tensor,
        state: Optional[List[Dict[str, Any]]] = None,
        token_id: Optional[torch.Tensor] = None,
        layer_hook: Optional[LayerMemoryHook] = None,
    ) -> CoreOutput:
        tok_seq = token_id.unsqueeze(1) if token_id is not None else None
        return self.forward_sequence(x_t.unsqueeze(1), state=state, token_ids=tok_seq, layer_hook=layer_hook)
