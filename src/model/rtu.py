
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from typing import Tuple, Dict, Any, Optional
# Avoid circular import: define a minimal PlasticAdapter base if needed
# We will try to import the real one, but if it fails (circular), use a dummy that will be replaced later
try:
    # Check if already loaded (circular case)
    import sys
    if 'src.model.plastic_adapter' in sys.modules and hasattr(sys.modules['src.model.plastic_adapter'], 'PlasticAdapter'):
        PlasticAdapter = sys.modules['src.model.plastic_adapter'].PlasticAdapter
    else:
        raise ImportError("Not yet loaded")
except:
    try:
        from src.model.plastic_adapter import PlasticAdapter
    except:
        try:
            from .plastic_adapter import PlasticAdapter
        except:
            from abc import ABC
            class PlasticAdapter(ABC):
                pass

# ponytail: minimal RTU implementation matching MLX 0.32.2 semantics. Upgrade to batched CUDA kernel if scaling beyond 16 layers.

class CustomAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, bias_correction=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, bias_correction=bias_correction)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            bias_correction = group['bias_correction']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if bias_correction:
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = lr / bias_correction1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                else:
                    step_size = lr
                    denom = exp_avg_sq.sqrt().add_(eps)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class Encoder(nn.Module):
    def __init__(self, dim: int, vocab_size: int = 256):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0 / math.sqrt(dim))

    def forward(self, x: torch.Tensor | int) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.long, device=self.embed.weight.device)
        elif x.dtype != torch.long:
            x = x.to(torch.long)
        return self.embed(x)


class Decoder(nn.Module):
    def __init__(self, dim: int, vocab_size: int = 256):
        super().__init__()
        self.decode = nn.Linear(dim, vocab_size)
        self.stop = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decode(x), torch.sigmoid(self.stop(x))


class Layer(nn.Module):
    def __init__(self, dim: int, spread: int = 32, vocab_size: int = 256):
        super().__init__()
        self.dim = dim
        self.spread = spread
        self.vocab_size = vocab_size

        halflives = torch.exp(torch.linspace(0.0, math.log(float(spread)), dim, dtype=torch.float32))
        retention = torch.exp(-math.log(2.0) / halflives)
        decay_init = torch.log(retention) - torch.log1p(-retention)
        self.decay = nn.Parameter(decay_init)

        self.register_buffer('states', torch.zeros(dim, dtype=torch.float32))
        self.register_buffer('decaytrace', torch.zeros(dim, dtype=torch.float32))
        self.register_buffer('embedtrace', torch.zeros(vocab_size, dim, dtype=torch.float32))

        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.weights = nn.Linear(dim, dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, enc: torch.Tensor, x: torch.Tensor, dummy: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if dummy is None:
            dummy = torch.zeros(self.dim, dtype=torch.float32, device=enc.device)
        decay = torch.sigmoid(self.decay)
        state = (decay * self.states) + enc + dummy
        x_out = x + self.silu(self.weights(self.norm(state)))
        return x_out, state, decay

    def reset(self):
        self.states.zero_()
        self.decaytrace.zero_()
        self.embedtrace.zero_()


def loss_variance(x: torch.Tensor) -> torch.Tensor:
    var = torch.var(x, unbiased=False)
    return torch.clamp(1.0 - torch.sqrt(var + 1e-4), min=0.0)


def loss_pred_mse(x: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return torch.mean((x - tgt) ** 2)


def loss_crossentropy(output: torch.Tensor, nextb: int | torch.Tensor) -> torch.Tensor:
    if isinstance(nextb, torch.Tensor):
        if nextb.dim() > 0:
            nextb = nextb.squeeze()
        idx = nextb.item() if nextb.numel() == 1 else nextb
    else:
        idx = nextb
    return -output[idx] + torch.logsumexp(output, dim=-1)


def loss_stop_mse(stop: torch.Tensor, end: bool) -> torch.Tensor:
    target = torch.tensor([1.0 if end else 0.0], dtype=stop.dtype, device=stop.device)
    return torch.mean((stop - target) ** 2)


def compute_losses(x: torch.Tensor, output: torch.Tensor, stop: torch.Tensor,
                   tgt: torch.Tensor | None = None,
                   nextb: int | torch.Tensor | None = None,
                   end: bool = False,
                   ce_only: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if ce_only:
        # Crossentropy-only training mode: variance / prediction / stop losses are skipped.
        if nextb is not None:
            total = loss_crossentropy(output, nextb)
            losses: Dict[str, torch.Tensor] = {'ce': total}
        else:
            total = torch.zeros((), dtype=output.dtype, device=output.device)
            losses = {}
        losses['total'] = total
        return total, losses

    l_var = loss_variance(x)
    losses = {'var': l_var}
    total = l_var

    if nextb is not None:
        if tgt is not None:
            l_pred = loss_pred_mse(x, tgt)
            losses['pred'] = l_pred
            total = total + l_pred
        l_ce = loss_crossentropy(output, nextb)
        l_stop = loss_stop_mse(stop, end)
        losses['ce'] = l_ce
        losses['stop'] = l_stop
        total = total + l_ce + l_stop

    losses['total'] = total
    return total, losses


class Model(nn.Module):
    def __init__(self, dim: int = 512, layers: int = 16, spread: int = 32, temp: float = 0.75,
                 rate: float = 5e-4, bound: Tuple[int, int] = (40000, 120000),
                 vocab_size: int = 256, encoder: nn.Module | None = None, decoder: nn.Module | None = None):
        super().__init__()
        self.dim = dim
        self.layers = layers
        self.spread = spread
        self.temp = temp
        self.rate = rate
        self.bound = bound
        self.vocab_size = vocab_size

        self.encoder = encoder if encoder is not None else Encoder(dim, vocab_size=vocab_size)
        self.decoder = decoder if decoder is not None else Decoder(dim, vocab_size=vocab_size)
        self.blocks = nn.ModuleList([Layer(dim, spread, vocab_size=vocab_size) for _ in range(layers)])

        self.step_count = 0
        self.last_loss = 0.0
        self.optimizer = CustomAdamW(self.parameters(), lr=rate, weight_decay=0.01, bias_correction=False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def sample(self, output: torch.Tensor) -> int:
        probs = torch.softmax(output, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)) / math.log(float(self.vocab_size))
        temp = max(0.1, float(self.temp * (1.0 - self.temp * entropy)))
        dist = torch.distributions.Categorical(logits=output / temp)
        return int(dist.sample().item())

    def step(self, c: torch.Tensor | int, dummies: list[torch.Tensor] | None = None, frozen: bool = False):
        if dummies is None:
            dummies = [torch.zeros(self.dim, dtype=torch.float32, device=self.device) for _ in range(self.layers)]

        enc = self.encoder(c)
        x = enc
        states, decays = [], []

        for i, layer in enumerate(self.blocks):
            x, state, decay = layer(enc, x, dummies[i])
            if frozen:
                layer.states.copy_(state.detach())
            states.append(state)
            decays.append(decay)

        logits, stop = self.decoder(x)
        return (x, states, decays), (logits, stop)

    def lr_at_step(self, step: int) -> float:
        b0, b1 = float(self.bound[0]), float(self.bound[1])
        progress = max(0.0, min(1.0, (float(step) + 1.0 - b0) / (b1 - b0)))
        return self.rate * (1.0 - 0.9 * progress)

    def forward(self, currb: int | torch.Tensor, nextb: int | torch.Tensor | None = None,
                end: bool = False, frozen: bool = False,
                ce_only: bool = False) -> Tuple[int, float]:
        if frozen:
            with torch.no_grad():
                c = torch.tensor(currb, dtype=torch.long, device=self.device) if not isinstance(currb, torch.Tensor) else currb
                (_, states, _), (output, stop) = self.step(c, frozen=True)
                return self.sample(output), float(stop.item())

        c = torch.tensor(currb, dtype=torch.long, device=self.device) if not isinstance(currb, torch.Tensor) else currb
        dummies = [torch.zeros(self.dim, dtype=torch.float32, device=self.device, requires_grad=True) for _ in range(self.layers)]

        self.zero_grad(set_to_none=True)

        tgt = None
        if nextb is not None:
            n = torch.tensor(nextb, dtype=torch.long, device=self.device) if not isinstance(nextb, torch.Tensor) else nextb
            with torch.no_grad():
                tgt = self.encoder(n).detach()

        (x, states, decays), (output, stop) = self.step(c, dummies=dummies, frozen=False)
        total_loss, _ = compute_losses(x, output, stop, tgt=tgt, nextb=nextb, end=end, ce_only=ce_only)
        self.last_loss = float(total_loss.item())
        if total_loss.requires_grad:
            total_loss.backward()

        c_val = int(currb) if not isinstance(currb, torch.Tensor) else int(currb.item())
        c_range = (torch.arange(self.vocab_size, device=self.device) == c_val).float().unsqueeze(1)

        for i, layer in enumerate(self.blocks):
            dlds = dummies[i].grad
            if dlds is None:
                dlds = torch.zeros(self.dim, device=self.device)

            decay_embedtrace = layer.embedtrace * decays[i].detach()
            embedtrace = decay_embedtrace + c_range

            if hasattr(self.encoder, 'embed') and self.encoder.embed.weight.grad is not None:
                self.encoder.embed.weight.grad.add_(dlds * decay_embedtrace)

            dec_val = decays[i].detach()
            decaytrace = (dec_val * layer.decaytrace) + (dec_val * (1.0 - dec_val) * layer.states)
            layer.decay.grad = (dlds * decaytrace).clone()

            layer.states.copy_(states[i].detach())
            layer.decaytrace.copy_(decaytrace.detach())
            layer.embedtrace.copy_(embedtrace.detach())

        lr = self.lr_at_step(self.step_count)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.optimizer.step()
        self.step_count += 1

        return self.sample(output.detach()), float(stop.detach().item())

    def __call__(self, currb: int | torch.Tensor, nextb: int | torch.Tensor | None = None,
                 end: bool = False, frozen: bool = False,
                 ce_only: bool = False) -> Tuple[int, float]:
        return self.forward(currb, nextb=nextb, end=end, frozen=frozen, ce_only=ce_only)

    def reset(self):
        for layer in self.blocks:
            layer.reset()

    def count(self) -> int:
        per_layer = self.dim * self.dim + 3 * self.dim
        return self.vocab_size * self.dim + self.layers * per_layer + self.vocab_size * self.dim + self.vocab_size + self.dim + 1

    def save(self, path: str):
        data = {}
        for k, v in self.named_parameters():
            data[f"m.{k}"] = v.detach().cpu().contiguous()

        data["step_count"] = torch.tensor([self.step_count], dtype=torch.long)

        for i, layer in enumerate(self.blocks):
            data[f"state.{i}"] = layer.states.detach().cpu().contiguous()
            data[f"decaytrace.{i}"] = layer.decaytrace.detach().cpu().contiguous()
            data[f"embedtrace.{i}"] = layer.embedtrace.detach().cpu().contiguous()

        for i, p in enumerate(self.parameters()):
            p_state = self.optimizer.state.get(p)
            if p_state:
                if 'exp_avg' in p_state:
                    data[f"o.{i}.exp_avg"] = p_state['exp_avg'].detach().cpu().contiguous()
                if 'exp_avg_sq' in p_state:
                    data[f"o.{i}.exp_avg_sq"] = p_state['exp_avg_sq'].detach().cpu().contiguous()
                if 'step' in p_state:
                    data[f"o.{i}.step"] = torch.tensor([p_state['step']], dtype=torch.long)

        tmp = 'temporary-' + os.path.basename(path)
        tmp_dir = os.path.dirname(path)
        tmp_path = os.path.join(tmp_dir, tmp) if tmp_dir else tmp
        save_file(data, tmp_path)
        os.replace(tmp_path, path)

    def load(self, path: str):
        if not os.path.exists(path):
            return

        data = load_file(path)
        model_dict = {}

        for k, v in data.items():
            if k.startswith("m."):
                model_dict[k[2:]] = v
            elif k.startswith("state."):
                idx = int(k.split('.')[1])
                self.blocks[idx].states.copy_(v.to(self.device))
            elif k.startswith("decaytrace."):
                idx = int(k.split('.')[1])
                self.blocks[idx].decaytrace.copy_(v.to(self.device))
            elif k.startswith("embedtrace."):
                idx = int(k.split('.')[1])
                self.blocks[idx].embedtrace.copy_(v.to(self.device))
            elif k == "step_count":
                self.step_count = int(v.item())

        if model_dict:
            self.load_state_dict(model_dict, strict=False)

        params = list(self.parameters())
        for i, p in enumerate(params):
            if f"o.{i}.exp_avg" in data and f"o.{i}.exp_avg_sq" in data:
                self.optimizer.state[p] = {
                    'step': int(data[f"o.{i}.step"].item()) if f"o.{i}.step" in data else 0,
                    'exp_avg': data[f"o.{i}.exp_avg"].to(self.device),
                    'exp_avg_sq': data[f"o.{i}.exp_avg_sq"].to(self.device),
                }


# ---------------------------------------------------------------------------
# Legacy RTUMemoryBlock kept for backward compat - deprecated, use PlasticAssociativeRTU
# ---------------------------------------------------------------------------
class RTUMemoryBlock(nn.Module):
    """Plastic Recurrent Trace Unit for semantic sentence/chunk vectors - LEGACY, deprecated."""
    def __init__(self, dim: int = 384, spread: int = 32):
        super().__init__()
        self.dim = dim
        self.spread = spread
        halflives = torch.exp(torch.linspace(0.0, math.log(float(spread)), dim, dtype=torch.float32))
        retention = torch.exp(-math.log(2.0) / halflives)
        decay_init = torch.log(retention) - torch.log1p(-retention)
        self.decay = nn.Parameter(decay_init)
        self.register_buffer('states', torch.zeros(dim, dtype=torch.float32))
        self.register_buffer('decaytrace', torch.zeros(dim, dtype=torch.float32))
        self.register_buffer('embedtrace', torch.zeros(dim, dtype=torch.float32))
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        decay = torch.sigmoid(self.decay)
        if x.dim() == 1:
            new_state = (decay * self.states) + x
            delta = self.silu(self.proj(self.norm(new_state)))
            return x + delta, new_state
        else:
            outputs = []
            curr_state = self.states.clone()
            for t in range(x.shape[0]):
                curr_state = (decay * curr_state) + x[t]
                delta = self.silu(self.proj(self.norm(curr_state)))
                outputs.append(x[t] + delta)
            return torch.stack(outputs, dim=0), curr_state

    def reset(self):
        self.states.zero_()
        self.decaytrace.zero_()
        self.embedtrace.zero_()


# ---------------------------------------------------------------------------
# New: PlasticAssociativeRTU - Titans + RWKV-7 hybrid
# ---------------------------------------------------------------------------
class PlasticAssociativeRTU(PlasticAdapter, nn.Module):
    """
    Multi-head associative plastic memory (Titans + RWKV-7 hybrid) replacing 1D EMA RTU
    with predictive surprise gating, momentum-augmented surprise updates,
    and Householder-like selective erasure for bounded lifelong learning.
    
    Maintains:
    - S: multi-head matrix-valued associative state R^{H x Dh x Dh}
    - momentum: momentum buffer same shape
    - h_trace: 1D vector trace summary
    - decay: logit decay params
    - Projections: W_pred, W_K, W_V, W_Q, W_gate, W_alpha
    """
    def __init__(self, dim: int = 384, heads: int = 4, spread: int = 32, 
                 momentum_decay: float = 0.85, d_model: int = None, **kwargs):
        if d_model is not None:
            dim = d_model
        # Handle adapter_kwargs that might contain d_model
        if "d_model" in kwargs:
            dim = kwargs.pop("d_model")
        super().__init__()
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.spread = spread
        self.momentum_decay = momentum_decay

        # Associative state S in R^{H x Dh x Dh}
        self.register_buffer('S', torch.zeros(heads, self.head_dim, self.head_dim, dtype=torch.float32))
        self.register_buffer('momentum', torch.zeros(heads, self.head_dim, self.head_dim, dtype=torch.float32))
        self.register_buffer('h_trace', torch.zeros(dim, dtype=torch.float32))
        
        # Decay params (like RTUMemoryBlock)
        halflives = torch.exp(torch.linspace(0.0, math.log(float(spread)), dim, dtype=torch.float32))
        retention = torch.exp(-math.log(2.0) / halflives)
        decay_init = torch.log(retention) - torch.log1p(-retention)
        self.decay = nn.Parameter(decay_init)

        # Projections
        self.W_pred = nn.Linear(dim, dim, bias=False)
        self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False)
        self.W_Q = nn.Linear(dim, dim, bias=False)
        self.W_gate = nn.Linear(dim, heads, bias=False)  # retention weight per head
        self.W_alpha = nn.Linear(dim, heads, bias=False)  # erasure rate per head
        
        self.norm_pred = nn.LayerNorm(dim, eps=1e-5)
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.proj = nn.Linear(dim, dim, bias=False)  # for compatibility with old RTU
        self.silu = nn.SiLU()

        # For backward compat, alias states to h_trace
        # self.states will be property

        # Initialize projections with small std
        for m in [self.W_pred, self.W_K, self.W_V, self.W_Q, self.W_gate, self.W_alpha, self.proj]:
            nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(dim))

    @property
    def states(self):
        """Compatibility: states alias to h_trace for old code expecting RTUMemoryBlock.states"""
        return self.h_trace

    @states.setter
    def states(self, value):
        with torch.no_grad():
            self.h_trace.copy_(value)

    def predict_next(self) -> torch.Tensor:
        """
        Generate top-down expectation hat{e}_t = LayerNorm(W_pred h_{t-1}) from current trace.
        Returns normalized prediction vector of dimension dim.
        """
        pred = self.W_pred(self.h_trace)
        pred = self.norm_pred(pred)
        # Normalize
        pred = pred / (torch.linalg.vector_norm(pred).clamp(min=1e-8))
        return pred

    def compute_surprise(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Compute prediction surprise as s_t = 1 - cos(e_t, hat{e}_t) clamped to [0,2].
        Returns (error_vector, surprise_scalar)
        """
        e_t = x.detach().float().reshape(-1)
        if e_t.shape[0] != self.dim:
            # Handle dim mismatch
            if e_t.shape[0] > self.dim:
                e_t = e_t[:self.dim]
            else:
                e_t = F.pad(e_t, (0, self.dim - e_t.shape[0]))
        
        # Normalize e_t for cosine
        e_norm = e_t / (torch.linalg.vector_norm(e_t).clamp(min=1e-8))
        
        hat_e = self.predict_next()  # already normalized
        
        cos_sim = torch.dot(e_norm, hat_e).item()
        # Clamp cos_sim to [-1,1] for numerical stability
        cos_sim = max(-1.0, min(1.0, cos_sim))
        surprise = 1.0 - cos_sim
        # Clamp to [0,2] per spec
        surprise = max(0.0, min(2.0, surprise))
        
        error_vec = e_t - hat_e * torch.linalg.vector_norm(e_t).clamp(min=1e-8)
        # Or simple: e_t - hat_e scaled?
        # Use e_t - pred (unnormalized?) For simplicity: e_t - hat_e * ||e_t||
        # Actually error vector should be e_t - hat_e_t (both normalized? spec says error_vector)
        # We'll return e_t - hat_e (with e_t normalized equivalent)
        error_vec = e_norm - hat_e
        
        return error_vec, surprise

    def update_associative_memory(self, x: torch.Tensor, surprise_factor: float = 1.0) -> torch.Tensor:
        """
        Update associative state using momentum-augmented surprise:
        M_t = eta M_{t-1} + grad * s_t, then S_t = S_{t-1} G_t + M_t
        where G_t is Householder-like erasure operator.
        
        Also updates 1D trace: h_t = 0.9 h_{t-1} + 0.1 (1+0.5 s_t) e_t
        Returns adapted output (for PlasticAdapter interface)
        """
        e_t = x.detach().float().reshape(-1)
        if e_t.shape[0] != self.dim:
            if e_t.shape[0] > self.dim:
                e_t = e_t[:self.dim]
            else:
                e_t = F.pad(e_t, (0, self.dim - e_t.shape[0]))
        
        # Compute projections
        k = self.W_K(e_t)  # dim
        v = self.W_V(e_t)  # dim
        # q not needed for update, but for completeness
        # q = self.W_Q(e_t)
        
        # Reshape to heads
        k_heads = k.view(self.heads, self.head_dim)  # H x Dh
        v_heads = v.view(self.heads, self.head_dim)  # H x Dh
        
        # Normalize keys per head for erasure operator
        k_norms = torch.linalg.vector_norm(k_heads, dim=1, keepdim=True).clamp(min=1e-8)
        k_hat = k_heads / k_norms  # H x Dh, normalized
        
        # Compute alpha_h (erasure rate) and w_h (retention) per head
        alpha_logits = self.W_alpha(e_t)  # heads
        w_logits = self.W_gate(e_t)  # heads
        alpha_h = torch.sigmoid(alpha_logits)  # [0,1]
        w_h = torch.sigmoid(w_logits)  # [0,1], retention weight
        
        # Momentum update: M_t = eta * M_{t-1} + outer(k_h, v_h) * s_t
        # outer product per head: k_h outer v_h -> Dh x Dh
        with torch.no_grad():
            # Decay momentum
            self.momentum.mul_(self.momentum_decay)
            
            # Add new gradient scaled by surprise
            for h in range(self.heads):
                outer = torch.outer(k_heads[h], v_heads[h])  # Dh x Dh
                self.momentum[h].add_(outer * surprise_factor)
            
            # Householder-like erasure: G_h = (I - alpha_h * k_hat k_hat^T) * diag(w_h)
            # For simplicity, w_h is scalar per head controlling overall retention
            # So G_h = (I - alpha_h * k_hat k_hat^T) * w_h
            # Then S_t = S_{t-1} @ G_t + M_t  (or S_{t-1} * G_t + M_t)
            # We'll implement S = S @ G + M
            
            new_S = torch.zeros_like(self.S)
            for h in range(self.heads):
                I = torch.eye(self.head_dim, device=self.S.device, dtype=self.S.dtype)
                kh = k_hat[h]  # Dh
                # G = (I - alpha * kh kh^T) * w
                G = I - alpha_h[h] * torch.outer(kh, kh)
                G = G * w_h[h]  # retention scaling
                # S_t = S_{t-1} G + M_t
                # Note: S is Dh x Dh, G is Dh x Dh
                new_S[h] = self.S[h] @ G + self.momentum[h]
            
            self.S.copy_(new_S)
            
            # Update 1D trace: h_t = 0.9 h_{t-1} + 0.1 (1+0.5 s_t) e_t
            trace_coeff = 0.1 * (1.0 + 0.5 * surprise_factor)
            self.h_trace.mul_(0.9).add_(e_t * trace_coeff)

        # Return adapted output: e_t + SiLU(proj(norm(h_trace))) like old RTU
        with torch.no_grad():
            normed = self.norm(self.h_trace)
            delta = self.silu(self.proj(normed))
            output = e_t + delta
        
        return output

    def retrieve(self, q_vec: torch.Tensor) -> torch.Tensor:
        """
        Associative recall from fast weights.
        q_vec: query embedding dim
        Returns retrieved representation dim
        """
        q = q_vec.detach().float().reshape(-1)
        if q.shape[0] != self.dim:
            if q.shape[0] > self.dim:
                q = q[:self.dim]
            else:
                q = F.pad(q, (0, self.dim - q.shape[0]))
        
        q_proj = self.W_Q(q)  # dim
        q_heads = q_proj.view(self.heads, self.head_dim)  # H x Dh
        
        # For each head, retrieve: out_h = S_h^T @ q_h or S_h @ q_h
        # Using S^T @ q for associative recall
        out_heads = []
        for h in range(self.heads):
            # S[h] is Dh x Dh, q_heads[h] is Dh
            # Retrieve: S[h].T @ q_heads[h] or S[h] @ q_heads[h]
            # We'll use S @ q
            out_h = self.S[h] @ q_heads[h]  # Dh
            out_heads.append(out_h)
        
        out = torch.cat(out_heads, dim=0)  # dim
        # Optionally add residual from h_trace alignment?
        # For simplicity, return out + small h_trace contribution
        out = out + 0.1 * self.h_trace
        return out

    def update(self, x: torch.Tensor) -> torch.Tensor:
        """PlasticAdapter interface: update with input embedding, return adapted output."""
        return self.update_associative_memory(x, surprise_factor=1.0)

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        # Override to include buffers
        # Use nn.Module's state_dict for params, plus our buffers
        base = super().state_dict(*args, **kwargs)
        # Add buffers that are not in base (S, momentum, h_trace are buffers)
        # Actually buffers are included in state_dict by default if persistent=True
        # But we also want to ensure custom keys
        # Return combined
        sd = {
            "S": self.S.detach().cpu().contiguous(),
            "momentum": self.momentum.detach().cpu().contiguous(),
            "h_trace": self.h_trace.detach().cpu().contiguous(),
            "decay": self.decay.detach().cpu().contiguous(),
            "W_pred_weight": self.W_pred.weight.detach().cpu().contiguous(),
            "W_K_weight": self.W_K.weight.detach().cpu().contiguous(),
            "W_V_weight": self.W_V.weight.detach().cpu().contiguous(),
            "W_Q_weight": self.W_Q.weight.detach().cpu().contiguous(),
            "W_gate_weight": self.W_gate.weight.detach().cpu().contiguous(),
            "W_alpha_weight": self.W_alpha.weight.detach().cpu().contiguous(),
            "proj_weight": self.proj.weight.detach().cpu().contiguous(),
            "norm_weight": self.norm.weight.detach().cpu().contiguous(),
            "norm_bias": self.norm.bias.detach().cpu().contiguous(),
            "norm_pred_weight": self.norm_pred.weight.detach().cpu().contiguous(),
            "norm_pred_bias": self.norm_pred.bias.detach().cpu().contiguous(),
        }
        # Merge with base for compatibility, but our custom keys take precedence
        # Include base params too
        for k, v in base.items():
            if k not in sd:
                sd[k] = v.detach().cpu().contiguous() if isinstance(v, torch.Tensor) else v
        sd["_adapter_type"] = "associative_rtu"
        sd["_adapter_version"] = 1
        sd["dim"] = self.dim
        sd["heads"] = self.heads
        return sd

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True):
        # Handle both our custom format and standard nn.Module format
        with torch.no_grad():
            if "S" in state:
                self.S.copy_(state["S"].to(self.S.device))
            if "momentum" in state:
                self.momentum.copy_(state["momentum"].to(self.momentum.device))
            if "h_trace" in state:
                self.h_trace.copy_(state["h_trace"].to(self.h_trace.device))
            if "states" in state:
                # Legacy compatibility: states -> h_trace
                self.h_trace.copy_(state["states"].to(self.h_trace.device))
            if "decay" in state:
                self.decay.copy_(state["decay"].to(self.decay.device))
            
            # Load projection weights if present
            if "W_pred_weight" in state:
                self.W_pred.weight.copy_(state["W_pred_weight"].to(self.W_pred.weight.device))
            if "W_K_weight" in state:
                self.W_K.weight.copy_(state["W_K_weight"].to(self.W_K.weight.device))
            if "W_V_weight" in state:
                self.W_V.weight.copy_(state["W_V_weight"].to(self.W_V.weight.device))
            if "W_Q_weight" in state:
                self.W_Q.weight.copy_(state["W_Q_weight"].to(self.W_Q.weight.device))
            if "W_gate_weight" in state:
                self.W_gate.weight.copy_(state["W_gate_weight"].to(self.W_gate.weight.device))
            if "W_alpha_weight" in state:
                self.W_alpha.weight.copy_(state["W_alpha_weight"].to(self.W_alpha.weight.device))
            if "proj_weight" in state:
                self.proj.weight.copy_(state["proj_weight"].to(self.proj.weight.device))
            if "norm_weight" in state:
                self.norm.weight.copy_(state["norm_weight"].to(self.norm.weight.device))
            if "norm_bias" in state:
                self.norm.bias.copy_(state["norm_bias"].to(self.norm.bias.device))
            if "norm_pred_weight" in state:
                self.norm_pred.weight.copy_(state["norm_pred_weight"].to(self.norm_pred.weight.device))
            if "norm_pred_bias" in state:
                self.norm_pred.bias.copy_(state["norm_pred_bias"].to(self.norm_pred.bias.device))
            
            # Also try to load via parent for any remaining keys
            try:
                # Filter to only keys that are in parent's state_dict
                parent_sd = super().state_dict()
                filtered = {k: v for k, v in state.items() if k in parent_sd}
                if filtered:
                    super().load_state_dict(filtered, strict=False)
            except:
                pass

    def reset(self):
        self.S.zero_()
        self.momentum.zero_()
        self.h_trace.zero_()

    def get_memory_health(self) -> Dict[str, float]:
        """Basic health metrics for this module alone."""
        with torch.no_grad():
            assoc_norm = torch.linalg.vector_norm(self.S).item()
            trace_norm = torch.linalg.vector_norm(self.h_trace).item()
            return {
                "associative_norm": assoc_norm,
                "trace_norm": trace_norm,
                "saturation": min(1.0, assoc_norm / 100.0),  # heuristic
            }
