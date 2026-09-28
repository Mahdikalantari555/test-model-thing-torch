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
from src.model.knowledge_store import KnowledgeStore

# ponytail: DroidEngine binds 22MB ONNX MiniLM semantic anchor to plastic RTU memory.

# ---------------------------------------------------------------------------
# Zero-dependency discourse / TMS helpers (Decision 3 & 4 in design.md)
# ---------------------------------------------------------------------------
_ANAPHORA_RE = re.compile(r"^(It|They|This|These|Those)\b")

# Function-predicate verb set for SVO slot extraction (kept deliberately coarse).
_VERB_SET = {
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "will", "would", "can", "could", "shall", "should",
    "may", "might", "must", "use", "uses", "using", "refer", "refers",
    "mean", "means", "involve", "involves", "include", "includes",
    "provide", "provides", "enable", "enables", "allow", "allows",
    "help", "helps", "measure", "measures", "detect", "detects",
    "capture", "captures", "record", "records", "observe", "observes",
    "monitor", "monitors", "support", "supports", "contain", "contains",
    "consist", "consists", "apply", "applies", "work", "works",
    "operate", "operates", "give", "gives", "make", "makes", "take",
    "takes", "need", "needs", "require", "requires", "cover", "covers",
    "produce", "produces", "create", "creates", "build", "builds",
    "develop", "develops", "design", "designs", "test", "tests",
    "check", "checks", "compare", "compares", "analyze", "analyzes",
    "evaluate", "evaluates", "assess", "assesses", "estimate",
    "estimates", "calculate", "calculates", "compute", "computes",
    "derive", "derives", "extract", "extracts", "process", "processes",
    "transform", "transforms", "convert", "converts", "transmit",
    "transmits", "receive", "receives", "send", "sends", "store",
    "stores", "load", "loads", "save", "saves", "scan", "scans", "map",
    "maps", "model", "models", "simulate", "simulates", "predict",
    "predicts", "forecast", "forecasts", "classify", "classifies",
    "cluster", "clusters", "segment", "segments", "recognize",
    "recognizes", "identify", "identifies", "locate", "locates", "track",
    "tracks", "quantify", "quantifies", "validate", "validates",
    "verify", "verifies", "calibrate", "calibrates", "correct",
    "corrects", "adjust", "adjusts", "optimize", "optimizes", "improve",
    "improves", "enhance", "enhances", "reduce", "reduces", "increase",
    "increases", "decrease", "decreases", "minimize", "minimizes",
    "maximize", "maximizes", "balance", "balances", "control",
    "controls", "manage", "manages", "handle", "handles", "deal",
    "deals", "cope", "copes", "execute", "executes", "implement",
    "implements", "utilize", "utilizes", "employ", "employs", "leverage",
    "leverages", "adopt", "adopts", "adapt", "adapts",
}

_AUXILIARIES = {
    "do", "does", "did", "is", "are", "was", "were", "be", "been",
    "being", "has", "have", "had", "can", "could", "will", "would",
    "shall", "should", "may", "might", "must",
}

_NEGATIONS = {
    "not", "no", "never", "none", "nor", "neither", "without", "n't",
    "hardly", "barely",
}


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9°'’-]+", text.lower())


def _subject_and_predicate(text: str) -> tuple[str, Optional[str]]:
    """Rough SVO slot extraction: subject words before the head verb."""
    words = re.findall(r"[A-Za-z0-9°.\u2019'-]+", text)
    if not words:
        return "", None
    lowers = [w.lower() for w in words]
    i = next((k for k, w in enumerate(lowers) if w in _VERB_SET), None)
    if i is None:
        return " ".join(words[:2]), None
    subj = " ".join(words[:i]).strip() if i > 0 else " ".join(words[:2])
    j = i
    # walk past auxiliaries + negations to reach the head verb
    while j + 1 < len(lowers) and (lowers[j] in _AUXILIARIES or lowers[j] in _NEGATIONS):
        j += 1
    head = lowers[j]
    if head not in _VERB_SET:
        head = lowers[i]  # no real head verb ahead: stay on the copula
    elif head not in _AUXILIARIES and head.endswith("s"):
        head = head.rstrip("s")  # crude lemma: "requires" -> "require"
    return subj, head


def _extract_subject(sentence: str) -> Optional[str]:
    subj, _ = _subject_and_predicate(sentence)
    return subj or None


def _has_negation(text: str) -> bool:
    return any(t in _NEGATIONS for t in _tokens(text))


def _content_overlap(a: str, b: str) -> float:
    """Shared content-token fraction over the shorter sentence."""
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


# "The <noun> of <subject> is <object>" -> functional predicate, one output per
# input. Covers the spec's capital-of examples without a full NLP parse.
_FUNC_PRED_RE = re.compile(
    r"^\s*the\s+(\w+)\s+of\s+(.+?)\s+(?:is|are|was|were)\s+(.+)$",
    re.IGNORECASE,
)

_COPIULA = _AUXILIARIES


def _svo_slots(text: str) -> tuple[str, Optional[str]]:
    """(subject, functional predicate) for contradiction gating."""
    m = _FUNC_PRED_RE.match(text)
    if m:
        return m.group(2).strip().lower(), m.group(1).lower() + " of"
    subj, pred = _subject_and_predicate(text)
    return subj.lower(), pred


def _is_contradiction(new_text: str, old_text: str) -> bool:
    """Deterministic tier-2 check: functional-predicate slot clash or
    polarity inversion. Caller pre-gates on dense sim >= 0.65."""
    s_new, p_new = _svo_slots(new_text)
    s_old, p_old = _svo_slots(old_text)
    if not p_new or not p_old:
        # No functional predicate to compare: fall back to pure polarity
        # inversion on overlapping content.
        overlap = set(_tokens(new_text)) & set(_tokens(old_text))
        shared = {t for t in overlap if t not in _NEGATIONS}
        return _has_negation(new_text) != _has_negation(old_text) and len(shared) >= 3
    if s_new != s_old or p_new != p_old:
        return False
    if p_new in _COPIULA:
        # "X is A" vs "X is B" is not a contradiction (non-functional copula);
        # only a negation flip on the *same statement* is.
        return (
            _has_negation(new_text) != _has_negation(old_text)
            and _content_overlap(new_text, old_text) >= 0.5
        )
    obj_new = set(_tokens(new_text)) - set(_tokens(s_new)) - {p_new}
    obj_old = set(_tokens(old_text)) - set(_tokens(s_old)) - {p_old}
    slot_clash = bool(obj_new != obj_old and (obj_new | obj_old))
    polarity = (
        _has_negation(new_text) != _has_negation(old_text)
        and _content_overlap(new_text, old_text) >= 0.5
    )
    return slot_clash or polarity

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
    def __init__(self, name: str = "droid-alpha", dim: int = 384,
                 db_path: Optional[str] = None):
        super().__init__()
        self.name = name
        self.dim = dim
        self.device = torch.device("cpu")

        # 1. Semantic Anchor (quantized ONNX, 22MB)
        self.anchor = OnnxMiniLM()

        # 2. Plastic RTU Memory
        self.memory = RTUMemoryBlock(dim=dim)

        # 3. Episodic Knowledge Store (SQLite WAL + FTS5 + dense MVM + TMS)
        self.knowledge = KnowledgeStore(db_path or ":memory:", dim=dim)
        # ponytail: kept as inert attribute for app.py reset-button compat;
        # vectors now live in the SQLite store.
        self.knowledge_vectors: Optional[torch.Tensor] = None

        # Discourse state for rolling anaphora resolution (design Decision 4)
        self._discourse_subject: Optional[str] = None

        # 4. Learning counters & Audit logs
        self.step_count = 0
        self.logs: List[Dict[str, Any]] = []
        self.base_lr = 0.05
        self.last_query: Optional[str] = None

    @property
    def knowledge_bank(self) -> List[Dict[str, Any]]:
        """Back-compat view: active facts as plain dict list."""
        return self.knowledge.get_active_facts()

    def clear_knowledge(self):
        """Wipe episodic knowledge (RTU memory stays)."""
        self.knowledge.clear_all()
        self._discourse_subject = None

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
        """Split Markdown/text into coherent standalone semantic facts without truncating on abbreviations."""
        # Strip markdown syntax: [anchor](url) -> anchor, remove bold/italic/code markers
        cleaned_md = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text.strip())
        cleaned_md = re.sub(r'[*_`]', '', cleaned_md)

        # Strip line prefixes (headers #, blockquotes >, list bullets -, *)
        lines = []
        for line in cleaned_md.splitlines():
            clean_line = re.sub(r'^[#*>\-\d.]+\s+', '', line).strip()
            if clean_line:
                lines.append(clean_line)
        cleaned_text = "\n".join(lines) if lines else text.strip()

        # Protect common abbreviations and decimal numbers from premature splitting
        protected = re.sub(r'\b(e\.g|i\.e|etc|vs|al|dr|mr|mrs|prof)\.', r'\1<DOT>', cleaned_text, flags=re.IGNORECASE)
        protected = re.sub(r'(\d+)\.(\d+)', r'\1<DOT>\2', protected)

        raw_parts = re.split(r'(?<=[.?!])\s+|\n+|(?:;\s+)', protected)
        cleaned = [p.replace('<DOT>', '.').strip() for p in raw_parts if p.strip()]

        # Re-merge fragments if open parentheses or brackets exist
        merged: List[str] = []
        buffer = ""
        for part in cleaned:
            if buffer:
                buffer += " " + part
            else:
                buffer = part

            open_parens = buffer.count("(") - buffer.count(")")
            open_brackets = buffer.count("[") - buffer.count("]")
            if open_parens <= 0 and open_brackets <= 0 and len(buffer) > 10:
                merged.append(buffer)
                buffer = ""

        if buffer:
            if merged:
                merged[-1] += " " + buffer
            else:
                merged.append(buffer)

        return merged if merged else [text.strip()]

    def _resolve_anaphora(self, propositions: List[str]) -> List[str]:
        """Rolling discourse anaphora resolution (design Decision 4).

        Third-person pronouns at a proposition's start are anchored to the
        last active subject, carried across paragraph/teach() boundaries.
        Only replaced when a single clear subject exists; otherwise untouched.
        """
        resolved: List[str] = []
        last_subject = self._discourse_subject
        for p in propositions:
            s = p.strip()
            m = _ANAPHORA_RE.match(s)
            if m and last_subject:
                s = _ANAPHORA_RE.sub(last_subject, s, count=1)
            resolved.append(s)
            subj = _extract_subject(s)
            if subj:
                last_subject = subj
        self._discourse_subject = last_subject
        return resolved

    def teach(self, text: str, source: str = "chat", auto_tune: bool = True) -> Dict[str, Any]:
        """
        In-chat conversational knowledge absorption.
        Updates plastic RTU memory weights without destroying base stability.
        Propositions go through rolling anaphora resolution and two-tier
        contradiction gating before insertion into the KnowledgeStore.
        """
        if not text or len(text.strip()) == 0:
            return {"status": "empty", "chunks": 0}

        t0 = time.perf_counter()
        propositions = self._split_into_propositions(text)
        propositions = self._resolve_anaphora(propositions)

        # Deduplicate against active stored knowledge to prevent repeated facts
        existing_facts = self.knowledge.get_active_texts()
        unique_propositions = []
        for p in propositions:
            clean_p = p.lower().strip()
            if clean_p and clean_p not in existing_facts:
                unique_propositions.append(p)
                existing_facts.add(clean_p)

        if not unique_propositions:
            self.log("Knowledge already absorbed in memory, skipping duplicate storage.")
            return {
                "status": "already_known",
                "propositions": 0,
                "elapsed_ms": 0.0,
                "avg_novelty": 0.0,
                "memory_norm": float(torch.norm(self.memory.states).item()),
                "effective_lr": self.base_lr,
                "total_knowledge": self.knowledge.active_count()
            }

        propositions = unique_propositions
        self.log(f"Teaching {len(propositions)} new propositions from '{source}'...", details={"source": source})

        # 1. Semantic Embeddings via ONNX Anchor, pre-normalized on insert
        embs = self.anchor.embed(propositions)  # (N, 384)
        if embs.dim() == 1:
            embs = embs.unsqueeze(0)
        embs = torch.nn.functional.normalize(embs, p=2, dim=-1)

        # 2. Auto-tuned dynamic learning rate based on experience count
        if auto_tune:
            effective_lr = self.base_lr / math.sqrt(1.0 + self.step_count / 20.0)
        else:
            effective_lr = self.base_lr

        # 3. Plastic RTU memory update + episodic registration with TMS
        decay = torch.sigmoid(self.memory.decay)
        with torch.no_grad():
            curr_state = self.memory.states.clone()
            novelty_losses = []
            superseded_total = 0

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

                # Two-tier contradiction gating (design Decision 3):
                # tier-1 dense sim >= 0.65, tier-2 deterministic SVO clash.
                conflicts = [
                    c["id"] for c in self.knowledge.find_similar(e_i, 0.65)
                    if _is_contradiction(propositions[i], c["text"])
                ]
                new_id = self.knowledge.insert_fact(
                    propositions[i], source, time.time(),
                    self.step_count + i, float(novelty), e_i,
                )
                for old_id in conflicts:
                    self.knowledge.supersede(old_id, new_id)
                superseded_total += len(conflicts)

            # Bound memory state norm with LayerNorm scaling
            self.memory.states.copy_(curr_state)

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
            "total_knowledge": self.knowledge.active_count(),
            "superseded": superseded_total
        }
        self.log(
            f"Successfully learned {len(propositions)} facts in {elapsed*1000:.1f}ms. State norm: {mem_norm:.2f}.",
            details=result
        )
        return result

    def recall(self, query: str, top_k: int = 3, threshold: float = 0.50) -> List[Dict[str, Any]]:
        """Retrieve most relevant active facts via hybrid lexical+dense RRF.

        Superseded (contradicted) facts are excluded; `similarity` reports
        the dense cosine component of the fused result.
        """
        if self.knowledge.total_count() == 0:
            return []

        q_emb = self.anchor.embed(query)  # (384,)
        if q_emb.dim() > 1:
            q_emb = q_emb.squeeze(0)
        q_emb = q_emb / torch.linalg.vector_norm(q_emb).clamp(min=1e-6)

        matches = self.knowledge.recall(q_emb, top_k=top_k, query_text=query)
        hits: List[Dict[str, Any]] = []
        for m in matches:
            sim = m.get("dense_similarity") or 0.0
            m["similarity"] = sim
            if sim >= threshold:
                hits.append(m)
        return hits

    def _is_duplicate_or_subsumed(self, s1: str, s2: str) -> bool:
        """Check if two sentences are near-duplicates or one is subsumed by the other."""
        s1_c = re.sub(r'[^\w\s]', '', s1.lower()).strip()
        s2_c = re.sub(r'[^\w\s]', '', s2.lower()).strip()
        if s1_c == s2_c or s1_c in s2_c or s2_c in s1_c:
            return True
        w1 = set(s1_c.split())
        w2 = set(s2_c.split())
        if not w1 or not w2:
            return False
        overlap = len(w1 & w2) / max(len(w1), len(w2))
        return overlap > 0.70

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
            return f"I have absorbed this into my memory ({res['propositions']} new facts, memory norm: {res['memory_norm']:.2f}). You can now ask me about it!"

        # Query plastic memory with strict threshold
        hits = self.recall(user_message, top_k=4, threshold=0.45)

        # Contextual follow-up fallback: e.g. "explain complete", "what about RS", "tell me more"
        if not hits and self.last_query:
            contextual_query = f"{self.last_query} {user_message}"
            hits = self.recall(contextual_query, top_k=4, threshold=0.40)
            if hits:
                self.log(f"Resolved follow-up query using previous context: '{self.last_query}'.")

        if hits:
            # Update conversational focus
            self.last_query = user_message
            
            # Construct coherent synthesis without repeating duplicated/subsumed facts
            explanation_parts: List[str] = []
            for h in hits:
                t = h["text"].strip()
                if not any(self._is_duplicate_or_subsumed(t, existing) for existing in explanation_parts):
                    explanation_parts.append(t)
                    
            synthesized = " ".join(explanation_parts)
            self.log(f"Factual recall hit with similarity {hits[0]['similarity']:.3f}.")
            return f"{synthesized}"

        self.log("No relevant facts found above threshold.")
        return "I do not have information about this topic in my memory yet. Teach me by saying: 'learn: <facts>'."

        # If no specific knowledge matches, respond gracefully
        if len(self.knowledge_bank) > 0:
            return f"I am listening. I currently have {len(self.knowledge_bank)} facts stored in memory, but none directly match '{user_message}'. You can teach me by sending any paragraph or using 'learn: <text>'."
        else:
            return "I am your Droid assistant with plastic RTU memory. I have not learned any domain knowledge yet. Send me a paragraph or explanation, and I will absorb it instantly."

    def save_profile(self, target_dir: str):
        """Save Droid weights, configuration, knowledge store DB, and logs."""
        p = Path(target_dir)
        p.mkdir(parents=True, exist_ok=True)

        # 1. Config metadata
        config = {
            "name": self.name,
            "dim": self.dim,
            "step_count": self.step_count,
            "base_lr": self.base_lr,
            "total_facts": self.knowledge.active_count(),
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
        save_file(weights, str(p / "memory.safetensors"))

        # 3. Knowledge store: checkpoint WAL, then export a consistent DB
        self.knowledge.checkpoint()
        self.knowledge.export_to(str(p / "knowledge.db"))

        # 4. Logs
        with open(p / "train_log.json", "w") as f:
            json.dump(self.logs[-200:], f, indent=2)

        self.log(f"Profile saved to {target_dir}")

    def load_profile(self, target_dir: str) -> bool:
        """Load Droid weights, config, knowledge store DB, and logs from dir.

        Migrate legacy profiles that still store knowledge in knowledge.json.
        """
        p = Path(target_dir)
        config_file = p / "config.json"
        mem_file = p / "memory.safetensors"

        if not config_file.exists() or not mem_file.exists():
            return False

        with open(config_file, "r") as f:
            config = json.load(f)

        self.name = config.get("name", self.name)
        self.step_count = config.get("step_count", 0)
        self.base_lr = config.get("base_lr", self.base_lr)

        # Load weights
        weights = load_file(str(mem_file))
        with torch.no_grad():
            if "states" in weights:
                self.memory.states.copy_(weights["states"])
            if "decay" in weights:
                self.memory.decay.copy_(weights["decay"])
            if "proj_weight" in weights:
                self.memory.proj.weight.copy_(weights["proj_weight"])

        # Load knowledge store (SQLite DB, or legacy knowledge.json migration)
        db_file = p / "knowledge.db"
        know_file = p / "knowledge.json"
        self.knowledge.close()
        if db_file.exists():
            self.knowledge = KnowledgeStore(str(db_file), dim=self.dim)
        else:
            self.knowledge = KnowledgeStore(":memory:", dim=self.dim)
            if know_file.exists():
                with open(know_file, "r") as f:
                    legacy_facts = json.load(f)
                # Legacy profiles stored raw vectors in memory.safetensors when
                # available; re-embed only what is missing.
                legacy_vecs = weights.get("knowledge_vectors")
                for i, fact in enumerate(legacy_facts):
                    vec = None
                    if legacy_vecs is not None and i < len(legacy_vecs):
                        vec = legacy_vecs[i].float()
                    if vec is None:
                        vec = self.anchor.embed(fact["text"])
                        if vec.dim() == 1:
                            vec = vec
                        else:
                            vec = vec.squeeze(0)
                    self.knowledge.insert_fact(
                        fact["text"],
                        fact.get("source", "legacy"),
                        fact.get("timestamp", time.time()),
                        fact.get("step", i),
                        fact.get("novelty", 0.0),
                        vec,
                    )

        # Load logs
        log_file = p / "train_log.json"
        if log_file.exists():
            with open(log_file, "r") as f:
                self.logs = json.load(f)

        self.log(f"Profile loaded from {target_dir}")
        return True
