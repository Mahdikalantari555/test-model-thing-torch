
import time
import math
import json
from typing import List, Dict, Optional, Tuple, Any
import torch
import torch.nn.functional as F

# ponytail: System-1 decision head - hyperspherical cosine argmax routing <3ms CPU, no autoregression

class DecisionHead:
    """
    Non-autoregressive System-1 decision head with hyperspherical cosine argmax routing.
    Holds prototype embeddings P in R^{K x D} and routes queries via cosine similarity.
    """
    def __init__(self, prototypes: Optional[Dict[str, List[float]]] = None, 
                 dim: int = 384, 
                 prototype_names: Optional[List[str]] = None):
        self.dim = dim
        # Default prototypes: chat, recall, teach, tool, fast_recall, slow_synthesis
        if prototype_names is None:
            prototype_names = ["chat", "recall", "teach", "tool", "fast_recall", "slow_synthesis"]
        self.prototype_names = prototype_names
        self.K = len(prototype_names)
        
        # Initialize prototypes as normalized random vectors if not provided
        # In real usage, these would be embeddings of prototype descriptions
        if prototypes is not None:
            # prototypes dict name->embedding list
            self.prototype_names = list(prototypes.keys())
            self.K = len(self.prototype_names)
            embs = []
            for name in self.prototype_names:
                v = torch.tensor(prototypes[name], dtype=torch.float32)
                v = v / (torch.linalg.vector_norm(v).clamp(min=1e-8))
                embs.append(v)
            self.P = torch.stack(embs, dim=0)  # K x D
        else:
            # Create deterministic prototypes based on name hashing for reproducibility
            # Use simple orthogonal-ish initialization
            torch.manual_seed(42)
            P = torch.randn(self.K, dim)
            P = F.normalize(P, dim=1)
            self.P = P
        
        # For Ridge probe training
        self.W_probe: Optional[torch.Tensor] = None  # D x K or D x num_tools
        self.probe_labels: Optional[List[str]] = None
        self.lambda_reg = 1.0

    def _cosine_sim(self, query_emb: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between query and prototypes."""
        q = query_emb.detach().float().reshape(-1)
        if q.shape[0] != self.dim:
            # If dim mismatch, project or truncate - for safety, use first dim
            if q.shape[0] > self.dim:
                q = q[:self.dim]
            else:
                # pad
                pad = torch.zeros(self.dim - q.shape[0])
                q = torch.cat([q, pad])
        q = q / (torch.linalg.vector_norm(q).clamp(min=1e-8))
        # P is K x D, normalized rows
        # cosine = P @ q
        sims = torch.mv(self.P, q)  # K
        return sims

    def decide(self, query_emb: torch.Tensor, threshold: float = 0.35) -> Dict[str, Any]:
        """
        Zero-shot prototype routing via cosine similarity.
        Returns structured decision object.
        """
        t0 = time.perf_counter()
        sims = self._cosine_sim(query_emb)  # K
        
        # Softmax over scaled similarities (temperature 0.1 for sharp distribution)
        # Use scaled dot-product: softmax(sims / temp)
        temp = 0.1
        logits = sims / temp
        probs = F.softmax(logits, dim=0)
        
        # Normalized Shannon entropy confidence: 1 - H_norm
        # H_norm = -sum(p_k ln p_k) / ln K
        # Avoid log(0) with epsilon
        eps = 1e-12
        p_safe = probs.clamp(min=eps)
        entropy = -torch.sum(p_safe * torch.log(p_safe)).item()
        lnK = math.log(self.K) if self.K > 1 else 1.0
        h_norm = entropy / lnK if lnK > 0 else 0.0
        confidence = 1.0 - h_norm
        confidence = max(0.0, min(1.0, confidence))
        
        max_sim = float(torch.max(sims).item())
        argmax_idx = int(torch.argmax(sims).item())
        action = self.prototype_names[argmax_idx]
        
        # Out-of-domain rejection if max cosine < threshold
        if max_sim < threshold:
            action = "ambiguous"
            # For ambiguous, confidence should be low
            confidence = min(confidence, max_sim / threshold * 0.5)
        
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        result = {
            "action": action,
            "probabilities": probs.tolist(),
            "confidence": confidence,
            "latency_ms": latency_ms,
            "max_similarity": max_sim,
            "prototype_names": self.prototype_names,
            "similarities": sims.tolist(),
        }
        return result

    def train_probe(self, X: torch.Tensor, Y: torch.Tensor, lambda_reg: float = 1.0) -> torch.Tensor:
        """
        Closed-form Ridge regression: W = (X^T X + lambda I)^{-1} X^T Y
        X: N x D, Y: N x K (one-hot or prob)
        Returns W: D x K
        Completes in <5ms on CPU for small N.
        """
        t0 = time.perf_counter()
        if X.dim() != 2:
            raise ValueError(f"X must be 2D N x D, got {X.shape}")
        if Y.dim() != 2:
            raise ValueError(f"Y must be 2D N x K, got {Y.shape}")
        N, D = X.shape
        N2, K = Y.shape
        if N != N2:
            raise ValueError(f"N mismatch: X {N} vs Y {N2}")
        if N < 5:
            raise ValueError(f"Need at least 5 examples, got {N}")
        
        self.lambda_reg = lambda_reg
        # Ridge: (X^T X + lambda I) W = X^T Y
        # Solve via torch.linalg.solve for efficiency
        XtX = X.T @ X  # D x D
        # Add regularization
        XtX_reg = XtX + lambda_reg * torch.eye(D, dtype=X.dtype, device=X.device)
        XtY = X.T @ Y  # D x K
        
        # Use solve: W = inv(XtX_reg) @ XtY
        try:
            W = torch.linalg.solve(XtX_reg, XtY)
        except:
            # Fallback to pinv if singular
            W = torch.linalg.pinv(XtX_reg) @ XtY
        
        self.W_probe = W
        latency_ms = (time.perf_counter() - t0) * 1000.0
        # For benchmarking, we could store latency, but return W
        return W

    def probe_predict(self, query_emb: torch.Tensor) -> Dict[str, Any]:
        """Predict using trained Ridge probe if available, else fallback to prototype routing."""
        if self.W_probe is None or self.probe_labels is None:
            return self.decide(query_emb)
        
        t0 = time.perf_counter()
        q = query_emb.detach().float().reshape(-1)
        q = q / (torch.linalg.vector_norm(q).clamp(min=1e-8))
        logits = q @ self.W_probe  # K
        probs = F.softmax(logits, dim=0)
        
        eps = 1e-12
        p_safe = probs.clamp(min=eps)
        entropy = -torch.sum(p_safe * torch.log(p_safe)).item()
        lnK = math.log(len(self.probe_labels)) if len(self.probe_labels) > 1 else 1.0
        h_norm = entropy / lnK if lnK > 0 else 0.0
        confidence = max(0.0, min(1.0, 1.0 - h_norm))
        
        argmax_idx = int(torch.argmax(probs).item())
        action = self.probe_labels[argmax_idx]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        return {
            "action": action,
            "probabilities": probs.tolist(),
            "confidence": confidence,
            "latency_ms": latency_ms,
            "prototype_names": self.probe_labels,
        }

    def state_dict(self) -> Dict[str, Any]:
        d = {
            "P": self.P,
            "prototype_names": self.prototype_names,
            "dim": self.dim,
        }
        if self.W_probe is not None:
            d["W_probe"] = self.W_probe
        if self.probe_labels is not None:
            d["probe_labels"] = self.probe_labels
        return d

    def load_state_dict(self, state: Dict[str, Any]):
        if "P" in state:
            self.P = state["P"]
            self.K = self.P.shape[0]
        if "prototype_names" in state:
            self.prototype_names = state["prototype_names"]
            self.K = len(self.prototype_names)
        if "dim" in state:
            self.dim = state["dim"]
        if "W_probe" in state:
            self.W_probe = state["W_probe"]
        if "probe_labels" in state:
            self.probe_labels = state["probe_labels"]

    def to_json(self) -> str:
        """Serialize decision output - but this is for the class state, not decision."""
        # For decision output serialization, decide() already returns JSON-serializable dict
        return json.dumps({
            "prototype_names": self.prototype_names,
            "dim": self.dim,
            "K": self.K,
        })
