# 04 Translation Rules (MLX -> PyTorch)

Rule: translate one component at a time. No redesign, no optimization,
no aggressive refactoring. Preserve behavior first.

## Mapping table

| MLX | PyTorch |
|-----|---------|
| mx.array | torch.tensor |
| nn.Embedding(V, dim) | nn.Embedding(V, dim) |
| nn.Linear(dim, dim, bias=False) | nn.Linear(dim, dim, bias=False) |
| nn.LayerNorm(dim) | nn.LayerNorm(dim, eps=?) |
| nn.SiLU() | nn.SiLU() |
| nn.Linear(dim, 256) | nn.Linear(dim, 256) |
| nn.Linear(dim, 1) | nn.Linear(dim, 1) |
| mx.sigmoid | torch.sigmoid |
| mx.softmax | torch.softmax |
| mx.logsumexp | torch.logsumexp |
| mx.var | torch.var(unbiased=False)  # MLX default ddof=0; PyTorch defaults to ddof=1 |
| mx.mean | torch.mean |
| mx.square | torch.square |
| mx.maximum(0, ...) | torch.clamp(min=0) or torch.relu |
| mx.stop_gradient | .detach() |
| mx.random.categorical | torch.multinomial |
| mx.arange(256) == c | torch.arange(256) == c |
| mx.zeros | torch.zeros |
| mx.eval(...) | no-op (PyTorch eager) |
| mx.value_and_grad(f, argnums=(0,1)) | torch.autograd.grad |
| self.update(params) | load_state_dict |
| opt.AdamW(lr) | torch.optim.AdamW |
| util.tree_flatten / tree_unflatten | state_dict |
| mx.save_safetensors | safetensors.save_file |
| mx.load | safetensors.load_file |

## Determinism

Use deterministic seeds. Set:
  torch.manual_seed(seed)
  torch.use_deterministic_algorithms(True, warn_only=True)
  os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" (CUDA only)

## Dtypes

Pin to float32 explicitly. MLX is float32 by default; PyTorch defaults
differ per op. Do not rely on promotion.

## Component order

A. Embedding      -> tests/test_embedding.py
B. Encoder blocks -> tests/test_encoder.py
C. RTU memory    -> src/model/rtu.py, docs/rtu.md, tests/test_rtu.py
D. Latent predictor -> tests/test_predictor.py
E. Loss functions -> tests/test_losses.py
F. Sampling       -> tests/test_sampling.py
G. Custom grad hooks -> tests/test_grad_hooks.py
H. Save/load       -> tests/test_checkpoint.py

## Do NOT do

- Do not replace the custom grad hooks with a cleaner autodiff formulation.
- Do not batch the recurrent core.
- Do not swap AdamW for a different optimizer.
- Do not change the loss terms.