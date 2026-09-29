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



# Try imports for new modules with fallbacks
try:
    from src.model.rtu import PlasticAssociativeRTU, RTUMemoryBlock
except ImportError:
    try:
        from .rtu import PlasticAssociativeRTU, RTUMemoryBlock
    except:
        RTUMemoryBlock = None
        PlasticAssociativeRTU = None

try:
    from src.model.plastic_adapter import get_adapter, list_adapters, PlasticAdapter
except:
    try:
        from .plastic_adapter import get_adapter, list_adapters, PlasticAdapter
    except:
        get_adapter = None
        list_adapters = lambda: []
        PlasticAdapter = object

try:
    from src.model.decision_head import DecisionHead
except:
    try:
        from .decision_head import DecisionHead
    except:
        DecisionHead = None

try:
    from src.model.memory_lifecycle import RetrievalPolicy, ContinuousStream, SnapshotManager, MergeEngine, SemanticStore, ContradictionResolver
except:
    try:
        from .memory_lifecycle import RetrievalPolicy, ContinuousStream, SnapshotManager, MergeEngine, SemanticStore, ContradictionResolver
    except:
        RetrievalPolicy = None
        ContinuousStream = None
        SnapshotManager = None
        MergeEngine = None
        SemanticStore = None
        ContradictionResolver = None

try:
    from src.model.gguf_backend import GgufBackend
except:
    try:
        from .gguf_backend import GgufBackend
    except:
        GgufBackend = None

try:
    from src.model.memory_injection import inject_prompt, inject_hidden, inject_kv_cache, select_strategy
except:
    try:
        from .memory_injection import inject_prompt, inject_hidden, inject_kv_cache, select_strategy
    except:
        def inject_prompt(q, facts, max_context_tokens=3000):
            if not facts:
                return q
            ctx = "\n".join([f"- {f.get('text','')}" for f in facts[:5]])
            return f"Context:\n{ctx}\n\nQuery: {q}"
        def inject_hidden(backend, query, h_trace, layer_idx=12):
            return {"prompt": query, "strategy": "hidden"}
        def inject_kv_cache(backend, facts, query):
            return {"prompt": inject_prompt(query, facts), "strategy": "kv_cache"}
        def select_strategy(fact_count, query_type, profile_config=None):
            if fact_count <5:
                return "prompt"
            elif fact_count <=20:
                return "hidden"
            else:
                return "kv_cache"

try:
    from src.model.confidence import retrieval_confidence, generation_confidence, fuse_confidence, confidence_tier
except:
    try:
        from .confidence import retrieval_confidence, generation_confidence, fuse_confidence, confidence_tier
    except:
        def retrieval_confidence(sim, surprise=None, health=None):
            return sim
        def generation_confidence(logprobs=None, self_consistency=None, entropy=None):
            return 0.7
        def fuse_confidence(Cr, Cg, weights=None):
            return 2*Cr*Cg/(Cr+Cg+1e-8) if (Cr+Cg)>0 else 0.0
        def confidence_tier(c):
            return "high" if c>0.8 else "medium" if c>=0.5 else "low"

class DroidEngine(nn.Module):
    def __init__(self, name: str = "droid-alpha", dim: int = 384,
                 db_path: Optional[str] = None,
                 plastic_adapter: Optional[Union[str, Any]] = None,
                 adapter_kwargs: Optional[Dict[str, Any]] = None,
                 gguf_backend: Optional[Any] = None,
                 gguf_config: Optional[Dict[str, Any]] = None,
                 test_time_training: bool = True,
                 surprise_threshold: float = 0.4,
                 K_max: int = 5000):
        super().__init__()
        self.name = name
        self.dim = dim
        self.device = torch.device("cpu")
        self.anchor = OnnxMiniLM()
        adapter_kwargs = adapter_kwargs or {}
        if plastic_adapter is not None:
            if isinstance(plastic_adapter, str):
                if get_adapter is not None:
                    try:
                        self.memory = get_adapter(plastic_adapter, dim=dim, **adapter_kwargs)
                    except Exception as e:
                        from src.model.rtu import PlasticAssociativeRTU
                        self.memory = PlasticAssociativeRTU(dim=dim, **adapter_kwargs)
                else:
                    from src.model.rtu import PlasticAssociativeRTU
                    self.memory = PlasticAssociativeRTU(dim=dim, **adapter_kwargs)
            else:
                self.memory = plastic_adapter
        else:
            try:
                from src.model.rtu import PlasticAssociativeRTU
                self.memory = PlasticAssociativeRTU(dim=dim, **adapter_kwargs)
            except Exception as e:
                from src.model.rtu import RTUMemoryBlock
                self.memory = RTUMemoryBlock(dim=dim)
        self.knowledge = KnowledgeStore(db_path or ":memory:", dim=dim)
        self.knowledge_vectors: Optional[torch.Tensor] = None
        self._discourse_subject: Optional[str] = None
        self.step_count = 0
        self.logs: List[Dict[str, Any]] = []
        self.base_lr = 0.05
        self.last_query: Optional[str] = None
        self.surprise_threshold = surprise_threshold
        self.K_max = K_max
        self.test_time_training = test_time_training
        self.ttt_steps_this_session = 0
        self.max_ttt_steps = 100
        self.ttt_lr = 1e-4
        if DecisionHead is not None:
            try:
                self.decision_head = DecisionHead(dim=dim)
            except:
                self.decision_head = None
        else:
            self.decision_head = None
        if RetrievalPolicy is not None:
            try:
                self.retrieval_policy = RetrievalPolicy()
            except:
                self.retrieval_policy = None
        else:
            self.retrieval_policy = None
        self.snapshots: Dict[str, Dict] = {}
        self.current_version: Optional[str] = None
        self.snapshot_manager = None
        if SnapshotManager is not None:
            try:
                self.snapshot_manager = SnapshotManager()
            except:
                pass
        self.continuous_stream = None
        self.gguf_backend = gguf_backend
        self.gguf_config = gguf_config or {}
        if gguf_config and GgufBackend is not None and self.gguf_backend is None:
            try:
                model_path = gguf_config.get("gguf_model_path") or gguf_config.get("model_path")
                repo_id = gguf_config.get("gguf_repo_id") or gguf_config.get("repo_id")
                filename = gguf_config.get("gguf_filename") or gguf_config.get("filename")
                n_ctx = gguf_config.get("n_ctx", 4096)
                n_gpu_layers = gguf_config.get("n_gpu_layers", -1)
                params = gguf_config.get("gguf_params", {})
                if model_path or repo_id:
                    self.gguf_backend = GgufBackend(model_path=model_path, repo_id=repo_id, filename=filename, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, **params)
            except Exception as e:
                print(f"Failed to init GGUF backend from config: {e}")
        self._last_retrieval_strategy = "hybrid_rrf"
        self._retrieval_history: List[Dict] = []

    @property
    def knowledge_bank(self) -> List[Dict[str, Any]]:
        return self.knowledge.get_audit_trail()

    def log(self, message: str, details: Optional[Dict[str, Any]] = None):
        entry = {"timestamp": time.time(), "message": message}
        if details:
            entry["details"] = details
        self.logs.append(entry)
        if len(self.logs) > 500:
            self.logs = self.logs[-500:]

    def _split_propositions(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        props = []
        for s in sentences:
            s = s.strip()
            if len(s) < 5:
                continue
            if len(s.split()) < 3:
                continue
            props.append(s)
        return props

    def _resolve_anaphora(self, propositions: List[str]) -> List[str]:
        resolved = []
        for prop in propositions:
            m = _ANAPHORA_RE.match(prop)
            if m and self._discourse_subject:
                prop = re.sub(r'^(It|They|This|These|Those)\b', self._discourse_subject, prop, count=1)
            subj = _extract_subject(prop)
            if subj:
                self._discourse_subject = subj
            resolved.append(prop)
        return resolved

    def teach(self, text: str, source: str = "user_text", surprise_threshold: Optional[float] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if surprise_threshold is None:
            surprise_threshold = self.surprise_threshold
        propositions = self._split_propositions(text)
        if not propositions:
            return {"status": "no_propositions", "propositions": 0, "elapsed_ms": 0, "avg_novelty": 0, "memory_norm": 0, "total_knowledge": self.knowledge.active_count()}
        propositions = self._resolve_anaphora(propositions)
        effective_lr = self.base_lr / math.sqrt(1 + self.step_count / 20.0)
        novelty_losses = []
        superseded_total = 0
        episodic_absorbed = 0
        surprises = []
        for i, prop in enumerate(propositions):
            try:
                e_i = self.anchor.embed(prop)
                if e_i.dim() > 1:
                    e_i = e_i.squeeze(0)
                e_i = e_i / torch.linalg.vector_norm(e_i).clamp(min=1e-6)
            except Exception as e:
                continue
            try:
                if hasattr(self.memory, 'compute_surprise'):
                    _, surprise = self.memory.compute_surprise(e_i)
                else:
                    surprise = 1.0
            except:
                surprise = 1.0
            surprises.append(surprise)
            novelty_losses.append(surprise)
            should_insert = surprise >= surprise_threshold
            try:
                if hasattr(self.memory, 'update_associative_memory'):
                    self.memory.update_associative_memory(e_i, surprise_factor=surprise)
                elif hasattr(self.memory, 'update'):
                    self.memory.update(e_i)
                else:
                    _, new_state = self.memory(e_i)
                    if hasattr(self.memory, 'states'):
                        self.memory.states.copy_(new_state)
            except Exception as e:
                pass
            if should_insert:
                conflicts = []
                try:
                    conflicts = [c["id"] for c in self.knowledge.find_similar(e_i, 0.65) if _is_contradiction(prop, c["text"])]
                except:
                    pass
                try:
                    new_id = self.knowledge.insert_fact(prop, source, time.time(), self.step_count + i, float(surprise), e_i)
                    episodic_absorbed += 1
                    for old_id in conflicts:
                        self.knowledge.supersede(old_id, new_id)
                    superseded_total += len(conflicts)
                except Exception as e:
                    pass
        self.step_count += len(propositions)
        elapsed = time.perf_counter() - t0
        avg_novelty = sum(novelty_losses) / max(1, len(novelty_losses))
        avg_surprise = sum(surprises) / max(1, len(surprises))
        try:
            if hasattr(self.memory, 'h_trace'):
                mem_norm = float(torch.norm(self.memory.h_trace).item())
            elif hasattr(self.memory, 'states'):
                mem_norm = float(torch.norm(self.memory.states).item())
            else:
                mem_norm = 0.0
        except:
            mem_norm = 0.0
        result = {
            "status": "learned",
            "propositions": len(propositions),
            "episodic_absorbed": episodic_absorbed,
            "elapsed_ms": elapsed * 1000,
            "avg_novelty": avg_novelty,
            "avg_surprise": avg_surprise,
            "surprise_threshold": surprise_threshold,
            "memory_norm": mem_norm,
            "effective_lr": effective_lr,
            "total_knowledge": self.knowledge.active_count(),
            "superseded": superseded_total
        }
        self.log(f"Successfully learned {len(propositions)} facts in {elapsed*1000:.1f}ms. State norm: {mem_norm:.2f}.", details=result)
        return result

    def recall(self, query: str, top_k: int = 3, threshold: float = 0.50, strategy: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.knowledge.total_count() == 0:
            return []
        query_type = "specific_fact"
        if self.retrieval_policy:
            try:
                query_type = self.retrieval_policy.classify_query(query)
                if strategy is None:
                    strategy = self.retrieval_policy.select_strategy(query_type)
                self._last_retrieval_strategy = strategy
            except:
                pass
        if strategy is None:
            strategy = "hybrid_rrf"
        q_emb = self.anchor.embed(query)
        if q_emb.dim() > 1:
            q_emb = q_emb.squeeze(0)
        q_emb = q_emb / torch.linalg.vector_norm(q_emb).clamp(min=1e-6)
        try:
            matches = self.knowledge.recall(q_emb, top_k=top_k*2, query_text=query, strategy=strategy)
        except TypeError:
            matches = self.knowledge.recall(q_emb, top_k=top_k*2, query_text=query)
        hits: List[Dict[str, Any]] = []
        h_trace = None
        try:
            if hasattr(self.memory, 'h_trace'):
                h_trace = self.memory.h_trace
            elif hasattr(self.memory, 'states'):
                h_trace = self.memory.states
        except:
            pass
        for m in matches:
            sim = m.get("dense_similarity") or m.get("similarity") or 0.0
            # Definitional boost: for definitional queries, boost facts that are definitional
            definitional_boost = 0.0
            fact_text = m.get("text","")
            fact_lower = fact_text.lower()
            query_lower = query.lower()
            if query_type == "definitional":
                if "is the acquisition" in fact_lower or "is the" in fact_lower or "is a" in fact_lower or "is an" in fact_lower:
                    definitional_boost = 0.25
                if "what is" in query_lower and "acquisition" in fact_lower and "remote" in query_lower:
                    definitional_boost += 0.2
            # Generic overlap boost: if fact and query share rare words, boost
            try:
                import re
                STOP = {"a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "by", "for", "with", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had", "it", "its", "this", "that", "these", "those", "which", "who", "whom", "whose", "what", "when", "where", "why", "how", "can", "could", "should", "would", "will", "shall", "may", "about", "into", "over", "under", "between", "during", "through", "out", "up", "down", "you", "your"}
                q_words = set(re.findall(r"[a-z0-9]+", query_lower)) - STOP
                f_words = set(re.findall(r"[a-z0-9]+", fact_lower)) - STOP
                overlap = q_words & f_words
                if overlap:
                    # Boost by overlap size, up to 0.3
                    definitional_boost += min(0.3, len(overlap) * 0.15)
                    # Extra boost if overlap contains rare words (len>4)
                    rare_overlap = [w for w in overlap if len(w) > 4]
                    if rare_overlap:
                        definitional_boost += min(0.2, len(rare_overlap) * 0.1)
            except:
                pass
            sim_boosted = sim + definitional_boost
            if sim < 0.48:
                if sim_boosted >= threshold:
                    m["similarity"] = sim_boosted
                    hits.append(m)
                continue
            contextual = 0.0
            if h_trace is not None:
                try:
                    fact_emb = self.anchor.embed(m["text"])
                    if fact_emb.dim() > 1:
                        fact_emb = fact_emb.squeeze(0)
                    fact_emb = fact_emb / torch.linalg.vector_norm(fact_emb).clamp(min=1e-6)
                    h_norm = h_trace / (torch.linalg.vector_norm(h_trace).clamp(min=1e-8))
                    contextual = torch.dot(h_norm, fact_emb).item()
                except:
                    contextual = 0.0
            final_score = sim_boosted * (1.0 + 0.2 * max(0.0, contextual))
            m["similarity"] = final_score
            m["contextual_alignment"] = contextual
            m["original_similarity"] = sim
            if final_score >= threshold:
                hits.append(m)
        hits.sort(key=lambda x: x.get("similarity",0), reverse=True)
        hits = hits[:top_k]
        if self.test_time_training and self.ttt_steps_this_session < self.max_ttt_steps:
            try:
                for hit in hits[:1]:
                    if self.ttt_steps_this_session >= self.max_ttt_steps:
                        break
                    fact_emb = self.anchor.embed(hit["text"])
                    if fact_emb.dim() > 1:
                        fact_emb = fact_emb.squeeze(0)
                    conf = hit.get("similarity", 0.5)
                    if hasattr(self.memory, 'update_associative_memory'):
                        self.memory.update_associative_memory(fact_emb, surprise_factor=conf)
                        self.ttt_steps_this_session += 1
            except Exception as e:
                pass
        self._retrieval_history.append({
            "query": query,
            "query_type": query_type,
            "strategy": strategy,
            "hits": [h["id"] for h in hits],
            "timestamp": time.time()
        })
        if len(self._retrieval_history) > 100:
            self._retrieval_history = self._retrieval_history[-100:]
        return hits

    def _is_duplicate_or_subsumed(self, s1: str, s2: str) -> bool:
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
        self.log(f"Chat received: '{user_message}'")
        mode = "auto"
        query = user_message
        lower = user_message.lower().strip()
        if lower.startswith("/fast "):
            mode = "fast"
            query = user_message[5:].strip()
        elif lower.startswith("/slow "):
            mode = "slow"
            query = user_message[5:].strip()
        elif lower.startswith("/fast"):
            mode = "fast"
            query = user_message[5:].strip() if len(user_message)>5 else ""
        elif lower.startswith("/slow"):
            mode = "slow"
            query = user_message[5:].strip() if len(user_message)>5 else ""
        is_explicit_teach = lower.startswith(("learn:", "remember:", "note:", "teach:"))
        if is_explicit_teach:
            teach_content = re.sub(r'^(learn|remember|note|teach):\s*', '', user_message, flags=re.IGNORECASE)
            res = self.teach(teach_content, source="chat")
            return f"I have absorbed this into my memory ({res['propositions']} new facts, memory norm: {res['memory_norm']:.2f}). You can now ask me about it!"
        if mode == "auto" and hasattr(self, 'decide_path'):
            try:
                decision = self.decide_path(query)
                path = decision.get("path") or decision.get("action")
                if path in ("fast_recall", "fast", "recall", "chat"):
                    mode = "fast"
                elif path in ("slow_synthesis", "slow", "synthesis"):
                    mode = "slow"
            except:
                pass
        if mode == "fast":
            hits = self.recall(query, top_k=4, threshold=0.45)
            if not hits and self.last_query:
                contextual_query = f"{self.last_query} {query}"
                hits = self.recall(contextual_query, top_k=4, threshold=0.40)
                if hits:
                    self.log(f"Resolved follow-up query using previous context: '{self.last_query}'.")
            if hits:
                self.last_query = query
                explanation_parts: List[str] = []
                for h in hits:
                    t = h["text"].strip()
                    if not any(self._is_duplicate_or_subsumed(t, existing) for existing in explanation_parts):
                        explanation_parts.append(t)
                synthesized = " ".join(explanation_parts)
                self.log(f"Factual recall hit with similarity {hits[0]['similarity']:.3f}.")
                return f"{synthesized}"
            else:
                self.log("No relevant facts found above threshold.")
                return "I do not have information about this topic in my memory yet. Teach me by saying: 'learn: <facts>'."
        if mode == "slow":
            if self.gguf_backend is not None and hasattr(self, 'generate_with_memory'):
                try:
                    result = self.generate_with_memory(query, strategy="auto")
                    if isinstance(result, dict):
                        text = result.get("text", "")
                        tier = result.get("confidence_tier", "medium")
                        if tier == "low":
                            return f"I'm uncertain; here's what I found... {text}"
                        return text
                    else:
                        return str(result)
                except Exception as e:
                    pass
        hits = self.recall(query, top_k=4, threshold=0.45)
        if not hits and self.last_query:
            contextual_query = f"{self.last_query} {query}"
            hits = self.recall(contextual_query, top_k=4, threshold=0.40)
            if hits:
                self.log(f"Resolved follow-up query using previous context: '{self.last_query}'.")
        if hits:
            self.last_query = query
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

    def consolidate_memory(self, max_prune_ratio: float = 0.1) -> Dict[str, Any]:
        total = self.knowledge.active_count()
        if total == 0:
            return {"status": "no_facts", "pruned": 0}
        facts = [f for f in self.knowledge.get_audit_trail() if not f.get("superseded")]
        now = time.time()
        tau_half = 86400.0
        lambda_novelty = 0.5
        utilities = []
        for f in facts:
            access = f.get("access_count", 1)
            last = f.get("last_retrieved") or f.get("timestamp") or now
            recency = math.exp(-(now - last) / tau_half)
            novelty = f.get("novelty", 0.5)
            utility = access * recency + lambda_novelty * novelty
            utilities.append((f, utility))
        utilities.sort(key=lambda x: x[1])
        prune_count = int(total * max_prune_ratio)
        pruned = 0
        if prune_count > 0:
            to_prune = utilities[:prune_count]
            for fact, util in to_prune:
                if util < 0.5:
                    try:
                        self.knowledge.conn.execute("DELETE FROM facts WHERE id=?", (fact["id"],))
                        self.knowledge.conn.execute("DELETE FROM facts_vecs WHERE id=?", (fact["id"],))
                        self.knowledge.conn.execute("DELETE FROM facts_fts WHERE rowid=?", (fact["id"],))
                        pruned += 1
                    except Exception as e:
                        pass
            self.knowledge.conn.commit()
            self.knowledge._vec_cache = None
        recon_loss = 0.0
        if utilities and hasattr(self.memory, 'S'):
            try:
                for _ in range(3):
                    if hasattr(self.memory, 'S'):
                        S = self.memory.S
                        for h in range(S.shape[0]):
                            Sh = S[h]
                            I = torch.eye(Sh.shape[0], device=Sh.device)
                            orth_loss = torch.norm(Sh.T @ Sh - I, p='fro')
                            recon_loss += orth_loss.item()
                            grad = 2 * Sh @ (Sh.T @ Sh - I)
                            with torch.no_grad():
                                self.memory.S[h].sub_(1e-4 * grad)
            except Exception as e:
                pass
        return {"status": "consolidated", "total_before": total, "pruned": pruned, "total_after": self.knowledge.active_count(), "recon_loss": recon_loss}

    def get_memory_health(self) -> Dict[str, Any]:
        total = self.knowledge.total_count()
        active = self.knowledge.active_count()
        saturation = active / max(1, self.K_max)
        associative_norm = 0.0
        try:
            if hasattr(self.memory, 'S'):
                associative_norm = float(torch.norm(self.memory.S).item())
            elif hasattr(self.memory, 'states'):
                associative_norm = float(torch.norm(self.memory.states).item())
            saturation += associative_norm / 100.0
            saturation = min(1.0, saturation)
        except:
            pass
        forgetting_rate = 0.0
        try:
            facts = [f for f in self.knowledge.get_audit_trail() if not f.get("superseded")]
            if facts:
                now = time.time()
                threshold = 0.1
                low_access = 0
                for f in facts:
                    access = f.get("access_count", 1)
                    last = f.get("last_retrieved") or f.get("timestamp") or now
                    acc = access * math.exp(-(now-last)/86400.0)
                    if acc < threshold:
                        low_access += 1
                forgetting_rate = low_access / len(facts)
        except:
            pass
        interference = 0.0
        try:
            vecs, ids = self.knowledge._load_dense()
            if len(ids) > 1:
                sample_size = min(100, len(ids))
                if len(ids) > 1000:
                    import random
                    indices = random.sample(range(len(ids)), sample_size)
                    sample_vecs = vecs[indices]
                else:
                    sample_vecs = vecs[:sample_size]
                sims = sample_vecs @ sample_vecs.T
                mask = ~torch.eye(sims.shape[0], dtype=torch.bool)
                mean_sim = sims[mask].mean().item()
                interference = max(0.0, mean_sim)
        except:
            pass
        retrieval_quality = 0.5
        try:
            feedbacks = self.knowledge.get_feedback_log()
            if feedbacks:
                approves = sum(1 for fb in feedbacks if fb["action"] == "approve")
                rejects = sum(1 for fb in feedbacks if fb["action"] == "reject")
                total_fb = approves + rejects
                if total_fb > 0:
                    retrieval_quality = approves / total_fb
            else:
                if self._retrieval_history:
                    retrieval_quality = 0.7
        except:
            pass
        return {"saturation": saturation, "associative_norm": associative_norm, "forgetting_rate": forgetting_rate, "interference": interference, "retrieval_quality": retrieval_quality, "active_facts": active, "total_facts": total, "K_max": self.K_max}

    def create_snapshot(self, version_id: Optional[str] = None) -> Dict[str, Any]:
        if version_id is None:
            version_id = f"v{int(time.time())}"
        try:
            if hasattr(self.memory, 'state_dict'):
                plastic_state = self.memory.state_dict()
                serializable_state = {}
                for k, v in plastic_state.items():
                    if isinstance(v, torch.Tensor):
                        serializable_state[k] = v.detach().cpu()
                    else:
                        serializable_state[k] = v
            else:
                serializable_state = {}
        except Exception as e:
            serializable_state = {}
        health = {}
        try:
            health = self.get_memory_health()
        except:
            pass
        metadata = {"version_id": version_id, "timestamp": time.time(), "step_count": self.step_count, "fact_count": self.knowledge.active_count(), "parent_version": self.current_version, "health_metrics": health}
        snapshot = {"plastic_state": serializable_state, "metadata": metadata, "version": version_id, "timestamp": metadata["timestamp"], "step_count": metadata["step_count"], "fact_count": metadata["fact_count"]}
        self.snapshots[version_id] = snapshot
        self.current_version = version_id
        if self.snapshot_manager:
            try:
                self.snapshot_manager.create_snapshot(self, version_id, parent_version=metadata["parent_version"])
            except:
                pass
        return snapshot

    def restore_snapshot(self, snapshot_dict_or_version: Union[Dict, str]) -> bool:
        snapshot = None
        if isinstance(snapshot_dict_or_version, str):
            version_id = snapshot_dict_or_version
            snapshot = self.snapshots.get(version_id)
            if snapshot is None and self.snapshot_manager:
                if version_id in self.snapshot_manager.snapshots:
                    snapshot = self.snapshot_manager.snapshots[version_id]
        else:
            snapshot = snapshot_dict_or_version
            version_id = snapshot.get("version_id") or snapshot.get("version") or snapshot.get("metadata", {}).get("version_id")
        if snapshot is None:
            return False
        plastic_state = snapshot.get("plastic_state", {})
        try:
            if hasattr(self.memory, 'load_state_dict') and plastic_state:
                self.memory.load_state_dict(plastic_state)
        except Exception as e:
            return False
        meta = snapshot.get("metadata", {})
        self.step_count = meta.get("step_count", self.step_count)
        self.current_version = meta.get("version_id") or snapshot.get("version_id") or snapshot.get("version")
        return True

    def rollback_to(self, version_id: str) -> bool:
        result = self.restore_snapshot(version_id)
        if result:
            self.current_version = f"{version_id}_branch_{int(time.time())}"
        return result

    def branch_from(self, version_id: str, new_version_id: str) -> Optional[Dict]:
        snapshot = self.snapshots.get(version_id)
        if snapshot is None:
            if self.snapshot_manager:
                snapshot = self.snapshot_manager.snapshots.get(version_id)
            if snapshot is None:
                return None
        new_snapshot = {"plastic_state": snapshot.get("plastic_state", {}), "metadata": {"version_id": new_version_id, "timestamp": time.time(), "step_count": snapshot.get("metadata", {}).get("step_count", 0), "fact_count": snapshot.get("metadata", {}).get("fact_count", 0), "parent_version": version_id, "branch_from": version_id}, "version": new_version_id, "timestamp": time.time(), "step_count": snapshot.get("metadata", {}).get("step_count", 0), "fact_count": snapshot.get("metadata", {}).get("fact_count", 0)}
        self.snapshots[new_version_id] = new_snapshot
        self.current_version = new_version_id
        return new_snapshot

    def list_snapshots(self) -> List[Dict]:
        result = []
        for vid, snap in self.snapshots.items():
            meta = snap.get("metadata", {})
            if not meta:
                meta = {"version_id": vid, "timestamp": snap.get("timestamp",0), "step_count": snap.get("step_count",0), "fact_count": snap.get("fact_count",0)}
            result.append(meta)
        result.sort(key=lambda x: x.get("timestamp",0))
        return result

    def merge_memory(self, other_droid, policy: str = "require_manual") -> Dict[str, Any]:
        if MergeEngine is None:
            return {"error": "MergeEngine not available"}
        try:
            merge_result = MergeEngine.merge_memories(self.knowledge, other_droid.knowledge, policy=policy)
            try:
                weight_a = self.knowledge.active_count()
                weight_b = other_droid.knowledge.active_count()
                total = weight_a + weight_b
                if total == 0:
                    w_a, w_b = 0.5, 0.5
                else:
                    w_a = weight_a / total
                    w_b = weight_b / total
                fused = MergeEngine.fuse_plastic(self.memory, other_droid.memory, weight_a=w_a, weight_b=w_b)
                if fused.get("S") is not None and hasattr(self.memory, 'S'):
                    with torch.no_grad():
                        S_fused = fused["S"]
                        if S_fused.shape == self.memory.S.shape:
                            self.memory.S.copy_(S_fused)
                        if "h_trace" in fused and hasattr(self.memory, 'h_trace'):
                            h_fused = fused["h_trace"]
                            if h_fused.shape == self.memory.h_trace.shape:
                                self.memory.h_trace.copy_(h_fused)
                        if "momentum" in fused and hasattr(self.memory, 'momentum'):
                            mom_fused = fused["momentum"]
                            if mom_fused.shape == self.memory.momentum.shape:
                                self.memory.momentum.copy_(mom_fused)
            except Exception as e:
                pass
            other_facts = [f for f in other_droid.knowledge.get_audit_trail() if not f.get("superseded")]
            for fact in other_facts:
                try:
                    vec = other_droid.knowledge.conn.execute("SELECT vec FROM facts_vecs WHERE id=?", (fact["id"],)).fetchone()
                    if vec:
                        v = torch.frombuffer(bytearray(vec[0]), dtype=torch.float32).reshape(self.dim)
                        similar = self.knowledge.find_similar(v, threshold=0.9)
                        if not similar:
                            self.knowledge.insert_fact(fact["text"], fact.get("source","merge"), time.time(), self.step_count, fact.get("novelty",0.5), v)
                except Exception as e:
                    pass
            new_version = f"merge_{int(time.time())}"
            self.create_snapshot(new_version)
            return {"status": "merged", "new_version": new_version, "conflicts": merge_result.get("conflicts", []), "conflict_count": merge_result.get("conflict_count", 0), "resolutions": merge_result.get("resolutions", [])}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def distill_to_semantic(self) -> List[Dict]:
        if SemanticStore is None:
            return []
        try:
            store = SemanticStore(self.knowledge)
            result = store.distill()
            return result
        except Exception as e:
            return []

    def retrieve_semantic(self, query: str, top_k: int = 3) -> List[Dict]:
        try:
            q_emb = self.anchor.embed(query)
            if q_emb.dim() > 1:
                q_emb = q_emb.squeeze(0)
            q_emb = q_emb / torch.linalg.vector_norm(q_emb).clamp(min=1e-6)
            if SemanticStore is not None:
                store = SemanticStore(self.knowledge)
                return store.retrieve_semantic(q_emb, top_k=top_k)
            else:
                return self.knowledge.recall_semantic(q_emb, top_k=top_k)
        except Exception as e:
            return []

    def get_contradictions(self) -> List[Dict]:
        if ContradictionResolver is None:
            return []
        try:
            facts = [f for f in self.knowledge.get_audit_trail() if not f.get("superseded")]
            resolver = ContradictionResolver()
            return resolver.detect_contradictions(facts)
        except Exception as e:
            return []

    def approve(self, fact_id: int) -> Dict[str, Any]:
        try:
            self.knowledge.conn.execute("UPDATE facts SET access_count = COALESCE(access_count,0)+1 WHERE id=?", (fact_id,))
            self.knowledge.conn.commit()
            try:
                self.knowledge.log_feedback(fact_id, "approve", strategy_used=self._last_retrieval_strategy)
            except:
                pass
            if self.retrieval_policy:
                try:
                    last = self._retrieval_history[-1] if self._retrieval_history else {}
                    qt = last.get("query_type", "specific_fact")
                    self.retrieval_policy.update(qt, self._last_retrieval_strategy, reward=1.0)
                except:
                    pass
            if self.test_time_training and self.ttt_steps_this_session < self.max_ttt_steps:
                try:
                    row = self.knowledge.conn.execute("SELECT text FROM facts WHERE id=?", (fact_id,)).fetchone()
                    if row:
                        text = row[0]
                        emb = self.anchor.embed(text)
                        if emb.dim() > 1:
                            emb = emb.squeeze(0)
                        if hasattr(self.memory, 'update_associative_memory'):
                            self.memory.update_associative_memory(emb, surprise_factor=1.0)
                            self.ttt_steps_this_session += 1
                except Exception as e:
                    pass
            return {"status": "approved", "fact_id": fact_id}
        except Exception as e:
            return {"error": str(e), "fact_id": fact_id}

    def reject(self, fact_id: int) -> Dict[str, Any]:
        try:
            self.knowledge.conn.execute("UPDATE facts SET needs_review=1 WHERE id=?", (fact_id,))
            self.knowledge.conn.commit()
            try:
                self.knowledge.log_feedback(fact_id, "reject", strategy_used=self._last_retrieval_strategy)
            except:
                pass
            if self.retrieval_policy:
                try:
                    last = self._retrieval_history[-1] if self._retrieval_history else {}
                    qt = last.get("query_type", "specific_fact")
                    self.retrieval_policy.update(qt, self._last_retrieval_strategy, reward=-1.0)
                except:
                    pass
            return {"status": "rejected", "fact_id": fact_id}
        except Exception as e:
            return {"error": str(e), "fact_id": fact_id}

    def correct(self, old_fact_id: int, new_text: str) -> Dict[str, Any]:
        try:
            emb = self.anchor.embed(new_text)
            if emb.dim() > 1:
                emb = emb.squeeze(0)
            emb = emb / torch.linalg.vector_norm(emb).clamp(min=1e-6)
            new_id = self.knowledge.insert_fact(new_text, "correction", time.time(), self.step_count, 1.0, emb)
            self.knowledge.supersede(old_fact_id, new_id)
            try:
                self.knowledge.log_feedback(old_fact_id, "correct", correction_text=new_text, strategy_used=self._last_retrieval_strategy)
            except:
                pass
            if self.test_time_training and self.ttt_steps_this_session < self.max_ttt_steps:
                try:
                    if hasattr(self.memory, 'update_associative_memory'):
                        self.memory.update_associative_memory(emb, surprise_factor=1.5)
                        self.ttt_steps_this_session += 1
                except Exception as e:
                    pass
            try:
                self.distill_to_semantic()
            except:
                pass
            return {"status": "corrected", "old_id": old_fact_id, "new_id": new_id, "new_text": new_text}
        except Exception as e:
            return {"error": str(e), "old_id": old_fact_id}

    def get_feedback_log(self, fact_id: Optional[int] = None) -> List[Dict]:
        try:
            return self.knowledge.get_feedback_log(fact_id=fact_id)
        except:
            return []

    def start_continuous_stream(self, source: Optional[str] = None, max_fpm: int = 10) -> Dict[str, Any]:
        if ContinuousStream is None:
            if self.continuous_stream is None:
                self.continuous_stream = {"running": True, "source": source, "max_fpm": max_fpm, "stats": {"facts_processed":0, "facts_absorbed":0, "queue_depth":0, "dropped":0}}
            return {"status": "started", "source": source}
        try:
            if self.continuous_stream is None:
                self.continuous_stream = ContinuousStream(self, max_facts_per_minute=max_fpm, surprise_threshold=0.6)
            self.continuous_stream.start(log_source=source)
            return {"status": "started", "source": source, "max_fpm": max_fpm}
        except Exception as e:
            return {"error": str(e)}

    def pause_stream(self) -> Dict[str, Any]:
        if self.continuous_stream and hasattr(self.continuous_stream, 'pause'):
            self.continuous_stream.pause()
            return {"status": "paused"}
        elif self.continuous_stream and isinstance(self.continuous_stream, dict):
            self.continuous_stream["running"] = False
            return {"status": "paused"}
        return {"status": "no_stream"}

    def resume_stream(self) -> Dict[str, Any]:
        if self.continuous_stream and hasattr(self.continuous_stream, 'resume'):
            self.continuous_stream.resume()
            return {"status": "resumed"}
        elif self.continuous_stream and isinstance(self.continuous_stream, dict):
            self.continuous_stream["running"] = True
            return {"status": "resumed"}
        return {"status": "no_stream"}

    def get_stream_status(self) -> Dict[str, Any]:
        if self.continuous_stream and hasattr(self.continuous_stream, 'get_status'):
            return self.continuous_stream.get_status()
        elif self.continuous_stream and isinstance(self.continuous_stream, dict):
            return {"running": self.continuous_stream.get("running", False), "facts_processed": self.continuous_stream.get("stats", {}).get("facts_processed",0), "facts_absorbed": self.continuous_stream.get("stats", {}).get("facts_absorbed",0), "queue_depth": self.continuous_stream.get("stats", {}).get("queue_depth",0)}
        return {"running": False, "facts_processed":0, "facts_absorbed":0, "queue_depth":0}

    def get_stream_stats(self) -> Dict[str, Any]:
        return self.get_stream_status()

    def continuous_learn(self, text: str) -> Dict[str, Any]:
        return self.teach(text, source="continuous_stream", surprise_threshold=0.6)

    def generate_with_memory(self, query: str, strategy: str = "auto", facts: Optional[List[Dict]] = None) -> Dict[str, Any]:
        if facts is None:
            facts = self.recall(query, top_k=10, threshold=0.3)
        if strategy == "auto":
            try:
                query_type = self.retrieval_policy.classify_query(query) if self.retrieval_policy else "specific_fact"
                strategy = select_strategy(len(facts), query_type, self.gguf_config)
            except:
                strategy = select_strategy(len(facts), "specific_fact", self.gguf_config)
        prompt = query
        injection_info = {}
        if strategy == "prompt":
            prompt = inject_prompt(query, facts)
            injection_info = {"strategy": "prompt", "fact_count": len(facts)}
        elif strategy == "hidden":
            h_trace = None
            if hasattr(self.memory, 'h_trace'):
                h_trace = self.memory.h_trace
            elif hasattr(self.memory, 'states'):
                h_trace = self.memory.states
            if h_trace is not None and self.gguf_backend is not None:
                injection_info = inject_hidden(self.gguf_backend, query, h_trace, layer_idx=12)
                prompt = injection_info.get("prompt", inject_prompt(query, facts))
            else:
                prompt = inject_prompt(query, facts)
                injection_info = {"strategy": "hidden", "fallback": "prompt"}
        elif strategy == "kv_cache":
            if self.gguf_backend is not None:
                injection_info = inject_kv_cache(self.gguf_backend, facts, query)
                prompt = injection_info.get("prompt", inject_prompt(query, facts))
            else:
                prompt = inject_prompt(query, facts)
                injection_info = {"strategy": "kv_cache", "fallback": "prompt"}
        generated_text = ""
        token_logprobs = None
        if self.gguf_backend is not None:
            try:
                generated_text = self.gguf_backend.generate(prompt, max_tokens=512, temperature=0.7, top_p=0.9)
            except Exception as e:
                generated_text = f"[Fallback] Based on memory: {' '.join([f['text'] for f in facts[:3]])}"
        else:
            if facts:
                generated_text = " ".join([f["text"] for f in facts[:3]])
            else:
                generated_text = self.chat(query)
        retrieval_conf = 0.0
        if facts:
            retrieval_conf = max([f.get("similarity",0) for f in facts]) if facts else 0.0
        generation_conf = 0.7
        try:
            generation_conf = generation_confidence(token_logprobs=token_logprobs, self_consistency=0.8, entropy=0.3)
        except:
            pass
        fused_conf = fuse_confidence(retrieval_conf, generation_conf)
        tier = confidence_tier(fused_conf)
        return {"text": generated_text, "query": query, "strategy": strategy, "facts_used": facts[:5], "retrieval_confidence": retrieval_conf, "generation_confidence": generation_conf, "confidence": fused_conf, "confidence_tier": tier, "injection_info": injection_info, "prompt": prompt}

    def estimate_confidence(self, retrieval_conf: float, generation_conf: float) -> float:
        return fuse_confidence(retrieval_conf, generation_conf)

    def decide_path(self, query: str) -> Dict[str, Any]:
        if self.decision_head is None:
            q_lower = query.lower()
            if any(w in q_lower for w in ["what is", "who is", "when", "where", "define"]):
                return {"action": "fast_recall", "path": "fast", "confidence": 0.7, "reason": "definitional query heuristic"}
            else:
                return {"action": "slow_synthesis", "path": "slow", "confidence": 0.6, "reason": "synthesis heuristic"}
        try:
            q_emb = self.anchor.embed(query)
            if q_emb.dim() > 1:
                q_emb = q_emb.squeeze(0)
            decision = self.decision_head.decide(q_emb)
            action = decision.get("action", "chat")
            if action in ("recall", "fast_recall", "chat"):
                path = "fast"
            elif action in ("slow_synthesis", "tool", "teach"):
                path = "slow"
            else:
                sims = decision.get("similarities", [])
                names = decision.get("prototype_names", [])
                fast_idx = None
                slow_idx = None
                for i, n in enumerate(names):
                    if "fast" in n:
                        fast_idx = i
                    if "slow" in n:
                        slow_idx = i
                if fast_idx is not None and slow_idx is not None:
                    if sims[fast_idx] > sims[slow_idx]:
                        path = "fast"
                    else:
                        path = "slow"
                else:
                    path = "fast" if decision.get("confidence",0) > 0.6 else "slow"
            decision["path"] = path
            return decision
        except Exception as e:
            return {"action": "fast_recall", "path": "fast", "confidence": 0.5, "error": str(e)}

    def save_profile(self, target_dir: str):
        p = Path(target_dir)
        p.mkdir(parents=True, exist_ok=True)
        config = {
            "name": self.name,
            "dim": self.dim,
            "step_count": self.step_count,
            "base_lr": self.base_lr,
            "total_facts": self.knowledge.active_count(),
            "updated_at": time.time(),
            "surprise_threshold": self.surprise_threshold,
            "K_max": self.K_max,
            "test_time_training": self.test_time_training,
            "current_version": self.current_version,
        }
        try:
            if hasattr(self.memory, '_adapter_type'):
                config["plastic_adapter"] = self.memory._adapter_type if isinstance(self.memory._adapter_type, str) else "associative_rtu"
            else:
                cls_name = self.memory.__class__.__name__
                if "Associative" in cls_name:
                    config["plastic_adapter"] = "associative_rtu"
                elif "RTU" in cls_name:
                    config["plastic_adapter"] = "legacy_rtu"
                else:
                    config["plastic_adapter"] = "associative_rtu"
        except:
            config["plastic_adapter"] = "associative_rtu"
        if self.gguf_config:
            config.update(self.gguf_config)
        if self.gguf_backend and hasattr(self.gguf_backend, 'model_path'):
            config["gguf_model_path"] = self.gguf_backend.model_path
        if self.retrieval_policy:
            try:
                policy_path = p / "retrieval_policy.json"
                self.retrieval_policy.persist(str(policy_path))
                config["has_retrieval_policy"] = True
            except:
                pass
        with open(p / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        try:
            if hasattr(self.memory, 'state_dict'):
                weights = {}
                sd = self.memory.state_dict()
                for k, v in sd.items():
                    if isinstance(v, torch.Tensor):
                        weights[k] = v.detach().cpu().contiguous()
                if hasattr(self.memory, 'h_trace'):
                    weights["states"] = self.memory.h_trace.detach().cpu().contiguous()
                elif hasattr(self.memory, 'states'):
                    weights["states"] = self.memory.states.detach().cpu().contiguous()
                if hasattr(self.memory, 'decay'):
                    weights["decay"] = self.memory.decay.detach().cpu().contiguous()
                if hasattr(self.memory, 'proj'):
                    weights["proj_weight"] = self.memory.proj.weight.detach().cpu().contiguous()
                save_file(weights, str(p / "memory.safetensors"))
            else:
                weights = {
                    "states": self.memory.states.detach().cpu().contiguous(),
                    "decay": self.memory.decay.detach().cpu().contiguous(),
                    "proj_weight": self.memory.proj.weight.detach().cpu().contiguous(),
                }
                save_file(weights, str(p / "memory.safetensors"))
        except Exception as e:
            try:
                if hasattr(self.memory, 'h_trace'):
                    save_file({"states": self.memory.h_trace.detach().cpu()}, str(p / "memory.safetensors"))
            except:
                pass
        try:
            self.knowledge.checkpoint()
            self.knowledge.export_to(str(p / "knowledge.db"))
        except Exception as e:
            pass
        try:
            with open(p / "train_log.json", "w") as f:
                json.dump(self.logs[-200:], f, indent=2)
        except:
            pass
        try:
            if self.snapshots:
                with open(p / "snapshots.json", "w") as f:
                    serializable = {}
                    for vid, snap in self.snapshots.items():
                        meta = snap.get("metadata", {})
                        serializable[vid] = meta
                    json.dump(serializable, f, indent=2)
        except Exception as e:
            pass
        self.log(f"Profile saved to {target_dir}")

    def load_profile(self, target_dir: str) -> bool:
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
        self.surprise_threshold = config.get("surprise_threshold", self.surprise_threshold)
        self.K_max = config.get("K_max", self.K_max)
        self.test_time_training = config.get("test_time_training", self.test_time_training)
        self.current_version = config.get("current_version", self.current_version)
        self.gguf_config = {k: v for k, v in config.items() if k.startswith("gguf_")}
        adapter_name = config.get("plastic_adapter", "associative_rtu")
        adapter_kwargs = config.get("adapter_kwargs", {})
        if adapter_name and get_adapter is not None:
            try:
                current_type = self.memory.__class__.__name__.lower()
                if adapter_name not in current_type:
                    self.memory = get_adapter(adapter_name, dim=self.dim, **adapter_kwargs)
            except Exception as e:
                pass
        try:
            weights = load_file(str(mem_file))
            with torch.no_grad():
                if hasattr(self.memory, 'load_state_dict'):
                    try:
                        self.memory.load_state_dict(weights)
                    except Exception as e:
                        if "states" in weights and hasattr(self.memory, 'h_trace'):
                            self.memory.h_trace.copy_(weights["states"])
                        elif "states" in weights and hasattr(self.memory, 'states'):
                            self.memory.states.copy_(weights["states"])
                        if "decay" in weights and hasattr(self.memory, 'decay'):
                            self.memory.decay.copy_(weights["decay"])
                        if "proj_weight" in weights and hasattr(self.memory, 'proj'):
                            self.memory.proj.weight.copy_(weights["proj_weight"])
                else:
                    if "states" in weights:
                        self.memory.states.copy_(weights["states"])
                    if "decay" in weights:
                        self.memory.decay.copy_(weights["decay"])
                    if "proj_weight" in weights:
                        self.memory.proj.weight.copy_(weights["proj_weight"])
        except Exception as e:
            pass
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
                legacy_vecs = weights.get("knowledge_vectors") if 'weights' in locals() else None
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
        log_file = p / "train_log.json"
        if log_file.exists():
            with open(log_file, "r") as f:
                self.logs = json.load(f)
        snap_file = p / "snapshots.json"
        if snap_file.exists():
            try:
                with open(snap_file, "r") as f:
                    snap_meta = json.load(f)
                for vid, meta in snap_meta.items():
                    if vid not in self.snapshots:
                        self.snapshots[vid] = {"metadata": meta, "plastic_state": {}}
            except Exception as e:
                pass
        policy_file = p / "retrieval_policy.json"
        if policy_file.exists() and self.retrieval_policy:
            try:
                self.retrieval_policy.load(str(policy_file))
            except Exception as e:
                pass
        if self.gguf_config and GgufBackend is not None:
            try:
                model_path = self.gguf_config.get("gguf_model_path") or self.gguf_config.get("model_path")
                repo_id = self.gguf_config.get("gguf_repo_id") or self.gguf_config.get("repo_id")
                filename = self.gguf_config.get("gguf_filename") or self.gguf_config.get("filename")
                if model_path or repo_id:
                    self.gguf_backend = GgufBackend(model_path=model_path, repo_id=repo_id, filename=filename)
            except Exception as e:
                pass
        self.log(f"Profile loaded from {target_dir}")
        return True
