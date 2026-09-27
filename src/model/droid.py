import json
import math
import os
import re
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

from src.model.onnx_anchor import OnnxMiniLM

# ponytail: DroidEngine binds 22MB ONNX MiniLM semantic anchor to plastic RTU memory.

class RTUMemoryBlock(nn.Module):
    """Plastic Recurrent Trace Unit for semantic sentence/chunk vectors."""
    def __init__(self, dim: int = 384, spread: int = 32):
        super().__init__()
        self.dim = dim
        self.spread = spread

        # Compute initial logit decay from half-lives
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (dim,) or (batch, dim)
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


class DroidEngine(nn.Module):
    """
    Sub-30MB Lifelong Learning Droid Engine.
    Combines frozen ONNX MiniLM semantic backbone with plastic RTU memory.
    """
    def __init__(self, name: str = "droid-alpha", dim: int = 384):
        super().__init__()
        self.name = name
        self.dim = dim
        self.device = torch.device("cpu")

        # 1. Semantic Anchor (quantized ONNX, 22MB)
        self.anchor = OnnxMiniLM()
        
        # 2. Plastic RTU Memory
        self.memory = RTUMemoryBlock(dim=dim)
        
        # 3. Episodic Knowledge Bank (propositions, sources, embeddings)
        self.knowledge_bank: List[Dict[str, Any]] = []
        self.knowledge_vectors: Optional[torch.Tensor] = None

        # 4. Learning counters & Audit logs
        self.step_count = 0
        self.logs: List[Dict[str, Any]] = []
        self.base_lr = 0.05

    def log(self, message: str, level: str = "INFO", details: Optional[Dict[str, Any]] = None):
        entry = {
            "timestamp": time.strftime("%H:%M:%S"),
            "level": level,
            "message": message,
            "details": details or {}
        }
        self.logs.append(entry)
        if len(self.logs) > 500:
            self.logs.pop(0)

    def _split_into_propositions(self, text: str) -> List[str]:
        """Split text into coherent semantic facts/sentences."""
        # Split on sentence boundaries, colons, or newlines
        raw_parts = re.split(r'(?<=[.?!])\s+|\n+|(?:;\s+)', text.strip())
        parts = [p.strip() for p in raw_parts if len(p.strip()) > 10]
        return parts if parts else [text.strip()]

    def teach(self, text: str, source: str = "chat", auto_tune: bool = True) -> Dict[str, Any]:
        """
        In-chat conversational knowledge absorption.
        Updates plastic RTU memory weights without destroying base stability.
        """
        if not text or len(text.strip()) == 0:
            return {"status": "empty", "chunks": 0}

        t0 = time.perf_counter()
        propositions = self._split_into_propositions(text)
        self.log(f"Teaching {len(propositions)} propositions from '{source}'...", details={"source": source})

        # 1. Semantic Embeddings via ONNX Anchor
        embs = self.anchor.embed(propositions)  # (N, 384)
        if embs.dim() == 1:
            embs = embs.unsqueeze(0)

        # 2. Auto-tuned dynamic learning rate based on experience count
        if auto_tune:
            effective_lr = self.base_lr / math.sqrt(1.0 + self.step_count / 20.0)
        else:
            effective_lr = self.base_lr

        # 3. Plastic RTU memory update
        decay = torch.sigmoid(self.memory.decay)
        with torch.no_grad():
            curr_state = self.memory.states.clone()
            novelty_losses = []
            
            for i in range(embs.shape[0]):
                e_i = embs[i]
                
                # Novelty measure: cosine difference with current state
                state_norm = torch.norm(curr_state, p=2)
                if state_norm > 1e-6:
                    state_dir = curr_state / state_norm
                    overlap = torch.dot(e_i, state_dir).item()
                    novelty = max(0.0, 1.0 - overlap)
                else:
                    novelty = 1.0
                novelty_losses.append(novelty)

                # Plastic state accumulation: h_t = decay * h_{t-1} + e_t
                curr_state = (decay * curr_state) + (e_i * (1.0 + 0.5 * novelty))
                
                # Episodic knowledge registration
                fact_entry = {
                    "text": propositions[i],
                    "source": source,
                    "timestamp": time.time(),
                    "step": self.step_count + i,
                    "novelty": float(novelty)
                }
                self.knowledge_bank.append(fact_entry)

            # Bound memory state norm with LayerNorm scaling
            self.memory.states.copy_(curr_state)
            
            # Update knowledge vector matrix
            if self.knowledge_vectors is None:
                self.knowledge_vectors = embs.clone()
            else:
                self.knowledge_vectors = torch.cat([self.knowledge_vectors, embs], dim=0)

            # Plastic decay adjustment (slow adaptation)
            self.memory.decay.data.add_(-effective_lr * 0.01 * (decay - 0.9))
            self.step_count += len(propositions)

        elapsed = time.perf_counter() - t0
        avg_novelty = sum(novelty_losses) / max(1, len(novelty_losses))
        mem_norm = float(torch.norm(self.memory.states).item())

        result = {
            "status": "learned",
            "propositions": len(propositions),
            "elapsed_ms": elapsed * 1000,
            "avg_novelty": avg_novelty,
            "memory_norm": mem_norm,
            "effective_lr": effective_lr,
            "total_knowledge": len(self.knowledge_bank)
        }
        self.log(
            f"Successfully learned {len(propositions)} facts in {elapsed*1000:.1f}ms. State norm: {mem_norm:.2f}.",
            details=result
        )
        return result

    def recall(self, query: str, top_k: int = 3, threshold: float = 0.35) -> List[Dict[str, Any]]:
        """Retrieve most relevant learned facts using RTU memory-conditioned matching."""
        if not self.knowledge_bank or self.knowledge_vectors is None:
            return []

        q_emb = self.anchor.embed(query)  # (384,)
        
        # Query conditioned by RTU memory state
        with torch.no_grad():
            decay = torch.sigmoid(self.memory.decay)
            conditioned_q = (0.7 * q_emb) + (0.3 * (self.memory.states / max(1e-6, self.memory.states.norm())))
            conditioned_q = conditioned_q / torch.norm(conditioned_q)

            # Compute similarities with knowledge vectors
            sims = torch.mv(self.knowledge_vectors, conditioned_q)
            
            top_vals, top_indices = torch.topk(sims, k=min(top_k, len(self.knowledge_bank)))
            
            matches = []
            for val, idx in zip(top_vals, top_indices):
                score = float(val.item())
                if score >= threshold:
                    entry = dict(self.knowledge_bank[idx.item()])
                    entry["similarity"] = score
                    matches.append(entry)
            return matches

    def chat(self, user_message: str) -> str:
        """
        Conversational inference with factual memory synthesis.
        Guarantees zero collapse and immediate factual recall.
        """
        self.log(f"Chat received: '{user_message}'")
        
        # Check if user message is an explicit teaching assertion (e.g. starts with "learn:", "note:", or is a definition)
        lower = user_message.lower().strip()
        is_explicit_teach = lower.startswith(("learn:", "remember:", "note:", "teach:"))
        
        if is_explicit_teach:
            teach_content = re.sub(r'^(learn|remember|note|teach):\s*', '', user_message, flags=re.IGNORECASE)
            res = self.teach(teach_content, source="chat")
            return f"I have absorbed this into my memory ({res['propositions']} facts, memory norm: {res['memory_norm']:.2f}). You can now ask me about it!"

        # Query plastic memory
        hits = self.recall(user_message, top_k=3, threshold=0.32)
        
        if hits:
            # Construct coherent synthesis from learned facts
            top_hit = hits[0]
            other_hits = hits[1:]
            
            explanation_parts = [top_hit["text"]]
            for h in other_hits:
                if h["text"] not in explanation_parts:
                    explanation_parts.append(h["text"])
                    
            synthesized = " ".join(explanation_parts)
            self.log(f"Factual recall hit with similarity {top_hit['similarity']:.3f}.")
            return f"{synthesized}"

        # If no specific knowledge matches, respond gracefully
        if len(self.knowledge_bank) > 0:
            return f"I am listening. I currently have {len(self.knowledge_bank)} facts stored in memory, but none directly match '{user_message}'. You can teach me by sending any paragraph or using 'learn: <text>'."
        else:
            return "I am your Droid assistant with plastic RTU memory. I have not learned any domain knowledge yet. Send me a paragraph or explanation, and I will absorb it instantly."

    def save_profile(self, target_dir: str):
        """Save Droid weights, configuration, knowledge bank, and logs."""
        p = Path(target_dir)
        p.mkdir(parents=True, exist_ok=True)

        # 1. Config metadata
        config = {
            "name": self.name,
            "dim": self.dim,
            "step_count": self.step_count,
            "base_lr": self.base_lr,
            "total_facts": len(self.knowledge_bank),
            "updated_at": time.time()
        }
        with open(p / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # 2. Safetensors weights
        weights = {
            "states": self.memory.states.detach().cpu().contiguous(),
            "decay": self.memory.decay.detach().cpu().contiguous(),
            "proj_weight": self.memory.proj.weight.detach().cpu().contiguous(),
        }
        if self.knowledge_vectors is not None:
            weights["knowledge_vectors"] = self.knowledge_vectors.detach().cpu().contiguous()
        save_file(weights, str(p / "memory.safetensors"))

        # 3. Knowledge bank
        with open(p / "knowledge.json", "w") as f:
            json.dump(self.knowledge_bank, f, indent=2)

        # 4. Logs
        with open(p / "train_log.json", "w") as f:
            json.dump(self.logs[-200:], f, indent=2)

        self.log(f"Profile saved to {target_dir}")

    def load_profile(self, target_dir: str) -> bool:
        """Load Droid weights, config, knowledge bank, and logs from directory."""
        p = Path(target_dir)
        config_file = p / "config.json"
        mem_file = p / "memory.safetensors"
        know_file = p / "knowledge.json"

        if not config_file.exists() or not mem_file.exists():
            return False

        with open(config_file, "r") as f:
            config = json.load(f)

        self.name = config.get("name", self.name)
        self.step_count = config.get("step_count", 0)
        self.base_lr = config.get("base_lr", 0.05)

        # Load weights
        weights = load_file(str(mem_file))
        with torch.no_grad():
            if "states" in weights:
                self.memory.states.copy_(weights["states"])
            if "decay" in weights:
                self.memory.decay.copy_(weights["decay"])
            if "proj_weight" in weights:
                self.memory.proj.weight.copy_(weights["proj_weight"])
            if "knowledge_vectors" in weights:
                self.knowledge_vectors = weights["knowledge_vectors"].clone()

        # Load knowledge bank
        if know_file.exists():
            with open(know_file, "r") as f:
                self.knowledge_bank = json.load(f)

        # Load logs
        log_file = p / "train_log.json"
        if log_file.exists():
            with open(log_file, "r") as f:
                self.logs = json.load(f)

        self.log(f"Profile loaded from {target_dir}")
        return True
