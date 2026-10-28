"""Track-1 diagonal vector-state cores: C0 (unstacked RTU) and C1 (stacked gated RTU)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.cores.contracts import CoreOutput, LayerMemoryHook, StateSpec
from src.model.parallel_scan import associative_scan


class SwiGLUBlock(nn.Module):
    """Pre-RMSNorm SwiGLU feed-forward block (`d_model + 3 * d_model * d_ff` params)."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=1e-6)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        for mod in (self.w_gate, self.w_up, self.w_down):
            nn.init.normal_(mod.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.norm(x)
        return x + self.w_down(F.silu(self.w_gate(u)) * self.w_up(u))


class _UnstackedRTULayer(nn.Module):
    """Exact diagonal EMA layer matching `src/model/rtu.py` Layer math."""

    def __init__(self, d_model: int, spread: int = 32) -> None:
        super().__init__()
        halflives = torch.exp(torch.linspace(0.0, math.log(float(max(2, spread))), d_model, dtype=torch.float32))
        retention = torch.exp(-math.log(2.0) / halflives)
        decay_init = torch.log(retention) - torch.log1p(-retention)
        self.decay = nn.Parameter(decay_init)
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.weights = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.weights.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

    def forward_chunk(
        self,
        e: torch.Tensor,
        s0: torch.Tensor,
        chunk_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        decay = torch.sigmoid(self.decay)  # (d_model,)
        states, s_last = associative_scan(decay, e, h0=s0, chunk=chunk_size)
        delta_x = F.silu(self.weights(self.norm(states)))
        return delta_x, s_last

    def step_one(self, e_t: torch.Tensor, s_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        decay = torch.sigmoid(self.decay).unsqueeze(0)
        s_t = decay * s_prev + e_t
        delta_x = F.silu(self.weights(self.norm(s_t)))
        return delta_x, s_t


class UnstackedRTUCore(nn.Module):
    """`C0` (`unstacked_rtu`): parallel input-independent diagonal EMA layers reading `e_t` only."""

    core_id = "C0"
    supports_online_rtrl = True

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        d_ff: int = 0,
        spread: int = 32,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.d_ff = int(d_ff)
        self.spread = int(spread)
        self.chunk_size = int(chunk_size)
        self.layers = nn.ModuleList([_UnstackedRTULayer(self.d_model, spread=self.spread) for _ in range(self.n_layers)])
        self.ffns = nn.ModuleList(
            [SwiGLUBlock(self.d_model, self.d_ff) for _ in range(self.n_layers)]
        ) if self.d_ff > 0 else None
        self._persistent_state: Optional[torch.Tensor] = None

    def state_spec(self, batch_size: int = 1, seq_len: int = 1) -> StateSpec:
        del batch_size, seq_len
        return StateSpec(
            complexity_class="constant_vector",
            bytes_per_sequence=self.n_layers * self.d_model * 4,
            shape_per_layer=(self.d_model,),
            num_layers=self.n_layers,
        )

    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        dev = device if device is not None else self.layers[0].decay.device
        return torch.zeros(self.n_layers, batch_size, self.d_model, device=dev, dtype=dtype)

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
        # Critical C0 property: every layer reads raw embedding `e_t = x`, NOT `x^{l-1}`
        e = x
        acc = x
        next_states: List[torch.Tensor] = []
        for idx, layer in enumerate(self.layers):
            dx, s_last = layer.forward_chunk(e, state[idx], chunk_size=self.chunk_size)
            acc = acc + dx
            if self.ffns is not None:
                acc = self.ffns[idx](acc)
            if layer_hook is not None:
                acc = layer_hook(idx, acc, token_ids)
            next_states.append(s_last)
        stacked_state = torch.stack(next_states, dim=0)
        self._persistent_state = stacked_state.detach()
        return CoreOutput(
            hidden_states=acc,
            state=stacked_state,
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
        e_t = x_t
        acc = x_t
        next_states: List[torch.Tensor] = []
        tok_seq = token_id.unsqueeze(1) if token_id is not None else None
        for idx, layer in enumerate(self.layers):
            dx, s_t = layer.step_one(e_t, state[idx])
            acc = acc + dx
            if self.ffns is not None:
                acc = self.ffns[idx](acc.unsqueeze(1)).squeeze(1)
            if layer_hook is not None:
                acc = layer_hook(idx, acc.unsqueeze(1), tok_seq).squeeze(1)
            next_states.append(s_t)
        stacked_state = torch.stack(next_states, dim=0)
        self._persistent_state = stacked_state.detach()
        return CoreOutput(
            hidden_states=acc.unsqueeze(1),
            state=stacked_state,
            state_spec=self.state_spec(batch_size=batch, seq_len=1),
        )


class _StackedGatedRTULayer(nn.Module):
    """Candidate B stacked input-gated diagonal recurrence layer."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.pre_norm = nn.RMSNorm(d_model, eps=1e-6)
        self.w_g = nn.Linear(d_model, d_model, bias=True)
        self.w_in = nn.Linear(d_model, d_model, bias=False)
        self.s_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.w_out = nn.Linear(d_model, d_model, bias=False)

        for mod in (self.w_g, self.w_in, self.w_out):
            nn.init.normal_(mod.weight, mean=0.0, std=1.0 / math.sqrt(d_model))
        # Log-spaced timescales spanning decay 0.5 .. 0.9999
        decays = 1.0 - torch.logspace(math.log10(0.5), math.log10(1e-4), d_model, dtype=torch.float32)
        decays = decays.clamp(1e-5, 1.0 - 1e-5)
        with torch.no_grad():
            self.w_g.bias.copy_(torch.log(decays) - torch.log1p(-decays))

    def forward_chunk(
        self,
        x: torch.Tensor,
        s0: torch.Tensor,
        chunk_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        u = self.pre_norm(x)
        g = torch.sigmoid(self.w_g(u))  # (B, T, d)
        drive = (1.0 - g) * self.w_in(u)
        states, s_last = associative_scan(g, drive, h0=s0, chunk=chunk_size)
        out = x + self.w_out(F.silu(self.s_norm(states)))
        return out, s_last

    def step_one(self, x_t: torch.Tensor, s_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u_t = self.pre_norm(x_t)
        g_t = torch.sigmoid(self.w_g(u_t))
        s_t = g_t * s_prev + (1.0 - g_t) * self.w_in(u_t)
        out_t = x_t + self.w_out(F.silu(self.s_norm(s_t)))
        return out_t, s_t


class StackedGatedRTUCore(nn.Module):
    """`C1` (`stacked_gated_rtu`): Candidate B repaired stacked input-gated RTU core."""

    core_id = "C1"
    supports_online_rtrl = True

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        d_ff: int = 0,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.d_ff = int(d_ff)
        self.chunk_size = int(chunk_size)
        self.layers = nn.ModuleList([_StackedGatedRTULayer(self.d_model) for _ in range(self.n_layers)])
        self.ffns = nn.ModuleList(
            [SwiGLUBlock(self.d_model, self.d_ff) for _ in range(self.n_layers)]
        ) if self.d_ff > 0 else None
        self._persistent_state: Optional[torch.Tensor] = None

    def state_spec(self, batch_size: int = 1, seq_len: int = 1) -> StateSpec:
        del batch_size, seq_len
        return StateSpec(
            complexity_class="constant_vector",
            bytes_per_sequence=self.n_layers * self.d_model * 4,
            shape_per_layer=(self.d_model,),
            num_layers=self.n_layers,
        )

    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        dev = device if device is not None else self.layers[0].w_g.weight.device
        return torch.zeros(self.n_layers, batch_size, self.d_model, device=dev, dtype=dtype)

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
        next_states: List[torch.Tensor] = []
        for idx, layer in enumerate(self.layers):
            h, s_last = layer.forward_chunk(h, state[idx], chunk_size=self.chunk_size)
            if self.ffns is not None:
                h = self.ffns[idx](h)
            if layer_hook is not None:
                h = layer_hook(idx, h, token_ids)
            next_states.append(s_last)
        stacked_state = torch.stack(next_states, dim=0)
        self._persistent_state = stacked_state.detach()
        return CoreOutput(
            hidden_states=h,
            state=stacked_state,
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
        next_states: List[torch.Tensor] = []
        tok_seq = token_id.unsqueeze(1) if token_id is not None else None
        for idx, layer in enumerate(self.layers):
            h, s_t = layer.step_one(h, state[idx])
            if self.ffns is not None:
                h = self.ffns[idx](h.unsqueeze(1)).squeeze(1)
            if layer_hook is not None:
                h = layer_hook(idx, h.unsqueeze(1), tok_seq).squeeze(1)
            next_states.append(s_t)
        stacked_state = torch.stack(next_states, dim=0)
        self._persistent_state = stacked_state.detach()
        return CoreOutput(
            hidden_states=h.unsqueeze(1),
            state=stacked_state,
            state_spec=self.state_spec(batch_size=batch, seq_len=1),
        )
