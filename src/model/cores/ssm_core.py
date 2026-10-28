"""Track-1 selective diagonal SSM core: C2 (Mamba-style input-selective SSM)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.cores.contracts import CoreOutput, LayerMemoryHook, StateSpec
from src.model.cores.diagonal_cores import SwiGLUBlock
from src.model.parallel_scan import associative_scan


class _SelectiveSSMLayer(nn.Module):
    """Input-selective diagonal state-space block with stateful causal Conv1d."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        state_dim: int = 16,
        conv_kernel: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.state_dim = int(state_dim)
        self.conv_kernel = int(conv_kernel)

        self.pre_norm = nn.RMSNorm(self.d_model, eps=1e-6)
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_model, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_size=self.conv_kernel,
            groups=self.d_model,
            bias=True,
        )
        self.x_dbc = nn.Linear(self.d_model, self.d_model + 2 * self.state_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(self.d_model))
        a_init = torch.arange(1, self.state_dim + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_model, -1)
        self.A_log = nn.Parameter(torch.log(a_init.clone()))
        self.D_skip = nn.Parameter(torch.ones(self.d_model, dtype=torch.float32))
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.ffn = SwiGLUBlock(self.d_model, int(d_ff))

        for mod in (self.in_proj, self.x_dbc, self.out_proj):
            nn.init.normal_(mod.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        with torch.no_grad():
            dt_init = torch.exp(torch.linspace(math.log(0.001), math.log(0.1), self.d_model))
            self.dt_bias.copy_(dt_init + torch.log(-torch.expm1(-dt_init)))

    def forward_chunk(
        self,
        x: torch.Tensor,
        ssm_state: torch.Tensor,
        conv_cache: torch.Tensor,
        chunk_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, d = x.shape
        u = self.pre_norm(x)
        xz = self.in_proj(u)
        x_branch, z_gate = xz.chunk(2, dim=-1)

        # Stateful causal depthwise conv1d
        x_t = x_branch.transpose(1, 2)  # (B, d, T)
        if self.conv_kernel > 1:
            padded = torch.cat([conv_cache, x_t], dim=-1)
            next_conv_cache = padded[:, :, -(self.conv_kernel - 1):].contiguous()
        else:
            padded = x_t
            next_conv_cache = conv_cache
        x_conv = F.silu(self.conv1d(padded)).transpose(1, 2)  # (B, T, d)

        dbc = self.x_dbc(x_conv)
        dt_raw = dbc[:, :, :d]
        b_t = dbc[:, :, d: d + self.state_dim]
        c_t = dbc[:, :, d + self.state_dim:]

        dt = F.softplus(dt_raw + self.dt_bias)  # (B, T, d)
        a_neg = -torch.exp(self.A_log)  # (d, N)
        decay = torch.exp(dt.unsqueeze(-1) * a_neg.unsqueeze(0).unsqueeze(0))  # (B, T, d, N)
        drive = (dt * x_conv).unsqueeze(-1) * b_t.unsqueeze(2)  # (B, T, d, N)

        flat_dim = d * self.state_dim
        states_flat, s_last_flat = associative_scan(
            decay.reshape(batch, seq_len, flat_dim),
            drive.reshape(batch, seq_len, flat_dim),
            h0=ssm_state.reshape(batch, flat_dim),
            chunk=chunk_size,
        )
        states = states_flat.reshape(batch, seq_len, d, self.state_dim)
        s_last = s_last_flat.reshape(batch, d, self.state_dim)

        y = (states * c_t.unsqueeze(2)).sum(dim=-1) + x_conv * self.D_skip
        y = y * F.silu(z_gate)
        h = x + self.out_proj(y)
        h = self.ffn(h)
        return h, s_last, next_conv_cache


class SelectiveSSMCore(nn.Module):
    """`C2` (`selective_ssm`): Mamba-style input-dependent diagonal SSM core."""

    core_id = "C2"
    supports_online_rtrl = False

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        d_ff: int = 512,
        state_dim: int = 16,
        conv_kernel: int = 4,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.d_ff = int(d_ff)
        self.state_dim = int(state_dim)
        self.conv_kernel = int(conv_kernel)
        self.chunk_size = int(chunk_size)

        self.layers = nn.ModuleList(
            [
                _SelectiveSSMLayer(
                    self.d_model,
                    self.d_ff,
                    state_dim=self.state_dim,
                    conv_kernel=self.conv_kernel,
                )
                for _ in range(self.n_layers)
            ]
        )
        self._persistent_state: Optional[Dict[str, torch.Tensor]] = None

    def state_spec(self, batch_size: int = 1, seq_len: int = 1) -> StateSpec:
        del batch_size, seq_len
        conv_len = max(0, self.conv_kernel - 1)
        floats_per_layer = self.d_model * self.state_dim + self.d_model * conv_len
        return StateSpec(
            complexity_class="constant_vector",
            bytes_per_sequence=self.n_layers * floats_per_layer * 4,
            shape_per_layer=(self.d_model, self.state_dim),
            num_layers=self.n_layers,
        )

    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Dict[str, torch.Tensor]:
        dev = device if device is not None else self.layers[0].in_proj.weight.device
        conv_len = max(0, self.conv_kernel - 1)
        return {
            "ssm": torch.zeros(self.n_layers, batch_size, self.d_model, self.state_dim, device=dev, dtype=dtype),
            "conv": torch.zeros(self.n_layers, batch_size, self.d_model, conv_len, device=dev, dtype=dtype),
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
        next_ssm: List[torch.Tensor] = []
        next_conv: List[torch.Tensor] = []
        for idx, layer in enumerate(self.layers):
            h, s_last, c_last = layer.forward_chunk(
                h,
                state["ssm"][idx],
                state["conv"][idx],
                chunk_size=self.chunk_size,
            )
            if layer_hook is not None:
                h = layer_hook(idx, h, token_ids)
            next_ssm.append(s_last)
            next_conv.append(c_last)
        out_state = {
            "ssm": torch.stack(next_ssm, dim=0),
            "conv": torch.stack(next_conv, dim=0),
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
        tok_seq = token_id.unsqueeze(1) if token_id is not None else None
        return self.forward_sequence(x_t.unsqueeze(1), state=state, token_ids=tok_seq, layer_hook=layer_hook)
