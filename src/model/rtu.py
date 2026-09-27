import math
import os
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def forward(self, enc: torch.Tensor, x: torch.Tensor, dummy: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                   end: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
                 rate: float = 5e-4, bound: tuple[int, int] = (40000, 120000),
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
                end: bool = False, frozen: bool = False) -> tuple[int, float]:
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
        total_loss, _ = compute_losses(x, output, stop, tgt=tgt, nextb=nextb, end=end)
        self.last_loss = float(total_loss.item())
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
                 end: bool = False, frozen: bool = False) -> tuple[int, float]:
        return self.forward(currb, nextb=nextb, end=end, frozen=frozen)

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

        # optimizer states
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

        # restore optimizer state if present
        params = list(self.parameters())
        for i, p in enumerate(params):
            if f"o.{i}.exp_avg" in data and f"o.{i}.exp_avg_sq" in data:
                self.optimizer.state[p] = {
                    'step': int(data[f"o.{i}.step"].item()) if f"o.{i}.step" in data else 0,
                    'exp_avg': data[f"o.{i}.exp_avg"].to(self.device),
                    'exp_avg_sq': data[f"o.{i}.exp_avg_sq"].to(self.device),
                }
