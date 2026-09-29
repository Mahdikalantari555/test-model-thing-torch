
import time
import json
import threading
import queue
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn.functional as F

# ponytail: memory lifecycle management - versioning, merge, semantic store, contradiction, adaptive retrieval, continuous stream, TTT

class SnapshotManager:
    """Snapshot versioning, branching, and rollback using safetensors + knowledge.db"""
    def __init__(self, base_dir: str = "droids"):
        self.base_dir = Path(base_dir)
        self.snapshots: Dict[str, Dict] = {}  # version_id -> metadata

    def create_snapshot(self, droid, version_id: str, parent_version: Optional[str] = None) -> Dict:
        """Create complete memory snapshot containing plastic RTU state, knowledge store checkpoint, metadata."""
        # Get plastic state
        if hasattr(droid.memory, 'state_dict'):
            plastic_state = droid.memory.state_dict()
        else:
            plastic_state = {}
        
        # Knowledge store checkpoint - we store active count and version
        fact_count = droid.knowledge.active_count() if hasattr(droid, 'knowledge') else 0
        
        # Health metrics if available
        health = {}
        if hasattr(droid, 'get_memory_health'):
            try:
                health = droid.get_memory_health()
            except:
                pass
        
        metadata = {
            "version_id": version_id,
            "timestamp": time.time(),
            "step_count": getattr(droid, 'step_count', 0),
            "fact_count": fact_count,
            "parent_version": parent_version,
            "health_metrics": health,
        }
        
        snapshot = {
            "plastic_state": plastic_state,
            "metadata": metadata,
            # knowledge store is file-based, we would export DB in DroidManager
        }
        
        self.snapshots[version_id] = snapshot
        return snapshot

    def restore_snapshot(self, droid, version_id: str) -> bool:
        """Restore plastic state and knowledge store to prior snapshot version."""
        if version_id not in self.snapshots:
            return False
        snap = self.snapshots[version_id]
        plastic_state = snap.get("plastic_state", {})
        if hasattr(droid.memory, 'load_state_dict') and plastic_state:
            try:
                droid.memory.load_state_dict(plastic_state)
            except Exception as e:
                print(f"Failed to restore plastic state: {e}")
                return False
        # Knowledge store restore would need DB file - handled by DroidManager
        return True

    def list_snapshots(self) -> List[Dict]:
        """List all available snapshots with metadata sorted by timestamp."""
        result = []
        for vid, snap in self.snapshots.items():
            meta = snap.get("metadata", {})
            result.append(meta)
        result.sort(key=lambda x: x.get("timestamp", 0))
        return result

    def branch_from(self, version_id: str, new_version_id: str) -> Optional[Dict]:
        """Create new branch from snapshot."""
        if version_id not in self.snapshots:
            return None
        base = self.snapshots[version_id]
        # Copy plastic state
        new_snap = {
            "plastic_state": base.get("plastic_state", {}).copy() if isinstance(base.get("plastic_state"), dict) else base.get("plastic_state"),
            "metadata": {
                "version_id": new_version_id,
                "timestamp": time.time(),
                "step_count": base.get("metadata", {}).get("step_count", 0),
                "fact_count": base.get("metadata", {}).get("fact_count", 0),
                "parent_version": version_id,
                "branch_from": version_id,
            }
        }
        self.snapshots[new_version_id] = new_snap
        return new_snap


class MergeEngine:
    """Multi-profile memory fusion with conflict detection and resolution."""
    
    @staticmethod
    def detect_conflicts(store_a, store_b) -> List[Dict]:
        """Detect slot clashes between two stores using SVO logic."""
        # Simplified: use existing _is_contradiction logic from droid.py if available
        # For now, detect by checking similar facts with different objects
        conflicts = []
        try:
            # Get all facts from both stores
            facts_a = store_a.get_audit_trail()
            facts_b = store_b.get_audit_trail()
            # For each fact in A, find similar in B
            for fa in facts_a:
                if fa.get("superseded"):
                    continue
                # Find similar in B via embedding? For simplicity, use text overlap
                for fb in facts_b:
                    if fb.get("superseded"):
                        continue
                    # Check if same subject-predicate but different object
                    # Use simple heuristic: high overlap but not identical
                    # Reuse _svo_slots if available
                    try:
                        from src.model.droid import _svo_slots, _is_contradiction
                        if _is_contradiction(fa["text"], fb["text"]):
                            conflicts.append({
                                "type": "SlotClash" if "capital" in fa["text"].lower() or "is" in fa["text"].lower() else "Polarity",
                                "fact_a": fa,
                                "fact_b": fb,
                                "subject": _svo_slots(fa["text"])[0],
                            })
                    except:
                        # Fallback: check if texts share subject but differ
                        if fa["text"][:20].lower() == fb["text"][:20].lower() and fa["text"] != fb["text"]:
                            conflicts.append({
                                "type": "SlotClash",
                                "fact_a": fa,
                                "fact_b": fb,
                            })
        except Exception as e:
            print(f"Conflict detection error: {e}")
        return conflicts

    @staticmethod
    def merge_memories(store_a, store_b, policy: str = "require_manual") -> Dict:
        """Merge two Droid memories by unioning knowledge stores and fusing plastic states."""
        conflicts = MergeEngine.detect_conflicts(store_a, store_b)
        
        # Resolve per policy
        resolutions = []
        for conflict in conflicts:
            fa = conflict["fact_a"]
            fb = conflict["fact_b"]
            if policy == "keep_newest":
                # Keep later timestamp
                keep = fa if fa.get("timestamp",0) > fb.get("timestamp",0) else fb
                discard = fb if keep == fa else fa
                resolutions.append({"keep": keep["id"], "discard": discard["id"], "policy": policy})
            elif policy == "keep_highest_novelty":
                keep = fa if fa.get("novelty",0) > fb.get("novelty",0) else fb
                discard = fb if keep == fa else fa
                resolutions.append({"keep": keep["id"], "discard": discard["id"], "policy": policy})
            elif policy == "keep_both_as_alternatives":
                resolutions.append({"keep": [fa["id"], fb["id"]], "policy": policy, "alternative_of": True})
            else:  # require_manual
                resolutions.append({"conflict": conflict, "policy": "manual_review_needed"})
        
        return {
            "conflicts": conflicts,
            "resolutions": resolutions,
            "policy": policy,
            "conflict_count": len(conflicts),
        }

    @staticmethod
    def fuse_plastic(rtu_a, rtu_b, weight_a: float = 0.5, weight_b: float = 0.5) -> Dict:
        """Fuse plastic associative states by weighted averaging of S matrices and trace vectors."""
        try:
            # Weighted average of S
            S_a = rtu_a.S if hasattr(rtu_a, 'S') else torch.zeros(4,96,96)
            S_b = rtu_b.S if hasattr(rtu_b, 'S') else torch.zeros(4,96,96)
            # Handle shape mismatches
            if S_a.shape != S_b.shape:
                # Use smaller shape or pad
                min_shape = (min(S_a.shape[0], S_b.shape[0]), min(S_a.shape[1], S_b.shape[1]), min(S_a.shape[2], S_b.shape[2]))
                S_a = S_a[:min_shape[0], :min_shape[1], :min_shape[2]]
                S_b = S_b[:min_shape[0], :min_shape[1], :min_shape[2]]
            
            fused_S = S_a * weight_a + S_b * weight_b
            
            # Fuse h_trace
            h_a = rtu_a.h_trace if hasattr(rtu_a, 'h_trace') else rtu_a.states if hasattr(rtu_a, 'states') else torch.zeros(384)
            h_b = rtu_b.h_trace if hasattr(rtu_b, 'h_trace') else rtu_b.states if hasattr(rtu_b, 'states') else torch.zeros(384)
            fused_h = h_a * weight_a + h_b * weight_b
            
            # Fuse momentum
            mom_a = rtu_a.momentum if hasattr(rtu_a, 'momentum') else torch.zeros_like(fused_S)
            mom_b = rtu_b.momentum if hasattr(rtu_b, 'momentum') else torch.zeros_like(fused_S)
            fused_mom = mom_a * weight_a + mom_b * weight_b
            
            return {
                "S": fused_S,
                "h_trace": fused_h,
                "momentum": fused_mom,
                "weight_a": weight_a,
                "weight_b": weight_b,
            }
        except Exception as e:
            return {"error": str(e), "S": None}


class ContradictionResolver:
    """Detection and resolution of conflicting facts."""
    
    def detect_contradictions(self, facts: List[Dict]) -> List[Dict]:
        """Detect contradictions in list of facts."""
        contradictions = []
        for i in range(len(facts)):
            for j in range(i+1, len(facts)):
                try:
                    from src.model.droid import _is_contradiction, _svo_slots
                    if _is_contradiction(facts[i]["text"], facts[j]["text"]):
                        s_new, p_new = _svo_slots(facts[i]["text"])
                        s_old, p_old = _svo_slots(facts[j]["text"])
                        contradictions.append({
                            "fact_a": facts[i],
                            "fact_b": facts[j],
                            "subject": s_new,
                            "predicate": p_new,
                            "type": "SlotClash" if p_new and "of" in p_new else "PolarityInversion"
                        })
                except:
                    continue
        return contradictions

    def resolve(self, contradictions: List[Dict], policy: str = "supersede_newest") -> List[Dict]:
        """Resolve contradictions per policy."""
        resolutions = []
        for c in contradictions:
            fa = c["fact_a"]
            fb = c["fact_b"]
            if policy == "supersede_newest":
                keep = fa if fa.get("timestamp",0) > fb.get("timestamp",0) else fb
                discard = fb if keep == fa else fa
                resolutions.append({"action": "supersede", "keep": keep["id"], "discard": discard["id"]})
            elif policy == "supersede_highest_confidence":
                keep = fa if fa.get("novelty",0) > fb.get("novelty",0) else fb
                discard = fb if keep == fa else fa
                resolutions.append({"action": "supersede", "keep": keep["id"], "discard": discard["id"]})
            elif policy == "mark_both_review":
                resolutions.append({"action": "mark_review", "ids": [fa["id"], fb["id"]]})
            elif policy == "keep_as_alternatives":
                resolutions.append({"action": "keep_both", "ids": [fa["id"], fb["id"]], "alternative": True})
        return resolutions


class SemanticStore:
    """Distilled proposition store separate from episodic - KnowledgeStore extension."""
    
    def __init__(self, knowledge_store):
        self.ks = knowledge_store

    def distill(self, similarity_threshold: float = 0.85) -> List[Dict]:
        """Distill episodic facts into semantic propositions by clustering similar facts."""
        # Get all active facts
        facts = [f for f in self.ks.get_audit_trail() if not f.get("superseded")]
        if not facts:
            return []
        
        # Simple clustering by embedding similarity
        # For now, cluster by text similarity heuristic if no embeddings
        try:
            # Get embeddings
            vecs, ids = self.ks._load_dense()
            if len(ids) == 0:
                return []
            
            # Build id -> fact mapping
            id_to_fact = {f["id"]: f for f in facts}
            
            # Cluster: group facts with cosine >= threshold
            clusters = []
            visited = set()
            for i, fid in enumerate(ids):
                if fid in visited or fid not in id_to_fact:
                    continue
                cluster = [fid]
                visited.add(fid)
                # Find similar
                sims = torch.mv(vecs, vecs[i])
                for j, fid2 in enumerate(ids):
                    if fid2 in visited or fid2 not in id_to_fact:
                        continue
                    if sims[j].item() >= similarity_threshold:
                        cluster.append(fid2)
                        visited.add(fid2)
                clusters.append(cluster)
            
            # For each cluster, produce one semantic proposition (consensus)
            semantic_facts = []
            for cluster in clusters:
                if not cluster:
                    continue
                # Consensus: pick most central fact or longest?
                # For simplicity, pick first fact's text as consensus
                # In real implementation, would resolve contradictions
                cluster_facts = [id_to_fact[fid] for fid in cluster if fid in id_to_fact]
                if not cluster_facts:
                    continue
                # Pick fact with highest novelty or access_count as representative
                rep = max(cluster_facts, key=lambda f: f.get("novelty",0) + f.get("access_count",1)*0.1)
                # Create semantic fact
                # Need embedding for semantic fact - use average of cluster embeddings
                cluster_indices = [ids.index(fid) for fid in cluster if fid in ids]
                if cluster_indices:
                    avg_vec = vecs[cluster_indices].mean(dim=0)
                else:
                    avg_vec = torch.randn(self.ks.dim)
                
                # Check for contradictions within cluster and resolve
                resolver = ContradictionResolver()
                contras = resolver.detect_contradictions(cluster_facts)
                # If contradictions, pick highest confidence (novelty)
                if contras:
                    # Resolve by keeping highest novelty
                    rep = max(cluster_facts, key=lambda f: f.get("novelty",0))
                
                sid = self.ks.insert_semantic_fact(
                    text=rep["text"],
                    source_cluster_ids=cluster,
                    confidence=0.8,  # heuristic
                    vec=avg_vec
                )
                semantic_facts.append({
                    "id": sid,
                    "text": rep["text"],
                    "source_cluster_ids": cluster,
                    "confidence": 0.8,
                })
            
            return semantic_facts
        except Exception as e:
            print(f"Distillation error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def retrieve_semantic(self, query_vec: torch.Tensor, top_k: int = 3) -> List[Dict]:
        """Retrieve from semantic store."""
        return self.ks.recall_semantic(query_vec, top_k=top_k)

    def incremental_distill(self, new_fact_ids: List[int]) -> List[Dict]:
        """Only re-distill clusters overlapping with new facts."""
        # For simplicity, just call full distill
        # In optimized version, would only re-cluster affected areas
        return self.distill()


class RetrievalPolicy:
    """Epsilon-greedy bandit per query type for adaptive retrieval strategy selection."""
    
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
        self.strategies = ["episodic_only", "semantic_primary", "hybrid_rrf", "episodic_recency_weighted", "contextual_boosted"]
        self.query_types = ["definitional", "specific_fact", "temporal", "relational", "ambiguous"]
        # Probabilities per query type: query_type -> strategy -> prob
        self.probs: Dict[str, Dict[str, float]] = {}
        for qt in self.query_types:
            self.probs[qt] = {s: 1.0/len(self.strategies) for s in self.strategies}
        # Default strategies per query type
        self.defaults = {
            "definitional": "semantic_primary",
            "specific_fact": "hybrid_rrf",
            "temporal": "episodic_recency_weighted",
            "relational": "hybrid_rrf",
            "ambiguous": "hybrid_rrf",
        }
        # Rewards tracking
        self.rewards: Dict[str, Dict[str, List[float]]] = {}
        for qt in self.query_types:
            self.rewards[qt] = {s: [] for s in self.strategies}

    def classify_query(self, query: str) -> str:
        """Classify query into types: definitional, specific_fact, temporal, relational, ambiguous."""
        q = query.lower()
        if any(w in q for w in ["what is", "what are", "define", "definition", "meaning"]):
            return "definitional"
        if any(w in q for w in ["when", "before", "after", "recent", "latest", "history"]):
            return "temporal"
        if any(w in q for w in ["how does", "why", "relationship", "between", "compare"]):
            return "relational"
        if len(q.split()) <= 5 and "?" in q:
            return "specific_fact"
        # Check if ambiguous (low content words)
        if len(q.split()) < 3:
            return "ambiguous"
        return "specific_fact"

    def select_strategy(self, query_type: str) -> str:
        """Select strategy per query type with epsilon-greedy exploration."""
        import random
        if random.random() < self.epsilon:
            # Explore: random strategy
            return random.choice(self.strategies)
        # Exploit: best prob
        probs = self.probs.get(query_type, self.probs["specific_fact"])
        # Return strategy with max prob, with tie-breaking to default
        best = max(probs, key=lambda s: probs[s])
        # If default has similar prob, prefer default for stability
        default = self.defaults.get(query_type, "hybrid_rrf")
        if abs(probs.get(default,0) - probs[best]) < 0.05:
            return default
        return best

    def update(self, query_type: str, strategy: str, reward: float):
        """Update strategy probabilities based on feedback (approve=positive, reject=negative)."""
        if query_type not in self.probs:
            query_type = "specific_fact"
        if strategy not in self.probs[query_type]:
            return
        
        # Track reward
        self.rewards[query_type][strategy].append(reward)
        # Keep only last 100 rewards
        if len(self.rewards[query_type][strategy]) > 100:
            self.rewards[query_type][strategy] = self.rewards[query_type][strategy][-100:]
        
        # Update prob: simple moving average towards reward
        # reward in [-1,1], convert to prob delta
        current = self.probs[query_type][strategy]
        # Learning rate 0.1
        new_prob = current + 0.1 * reward
        new_prob = max(0.05, min(0.95, new_prob))
        self.probs[query_type][strategy] = new_prob
        
        # Renormalize others to sum to 1
        total = sum(self.probs[query_type].values())
        for s in self.probs[query_type]:
            self.probs[query_type][s] /= total

    def persist(self, path: str):
        """Persist strategy probabilities per Droid profile."""
        data = {
            "probs": self.probs,
            "rewards": self.rewards,
            "epsilon": self.epsilon,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Restore from disk."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.probs = data.get("probs", self.probs)
            self.rewards = data.get("rewards", self.rewards)
            self.epsilon = data.get("epsilon", self.epsilon)
        except:
            pass


class ContinuousStream:
    """Background learning from interaction stream with rate limiting and novelty filtering."""
    
    def __init__(self, droid, max_facts_per_minute: int = 10, surprise_threshold: float = 0.6):
        self.droid = droid
        self.max_fpm = max_facts_per_minute
        self.surprise_threshold = surprise_threshold
        self.queue = queue.Queue()
        self.running = False
        self.paused = False
        self.thread: Optional[threading.Thread] = None
        self.stats = {
            "facts_processed": 0,
            "facts_absorbed": 0,
            "dropped": 0,
            "queue_depth": 0,
        }
        self.lock = threading.Lock()

    def start(self, log_source: Optional[str] = None):
        """Start background thread."""
        if self.running:
            return
        self.running = True
        self.paused = False
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        # If log_source provided, ingest it
        if log_source:
            self.ingest_file(log_source)

    def _run_loop(self):
        """Background loop processing queue with rate limiting."""
        last_minute = time.time()
        facts_this_minute = 0
        
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
            
            try:
                # Rate limiting: check if we exceeded max_fpm
                now = time.time()
                if now - last_minute >= 60:
                    last_minute = now
                    facts_this_minute = 0
                
                if facts_this_minute >= self.max_fpm:
                    time.sleep(1.0)
                    continue
                
                try:
                    text = self.queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                with self.lock:
                    self.stats["queue_depth"] = self.queue.qsize()
                
                # Process text: extract propositions and teach with higher surprise threshold
                # Simplified: treat each sentence as proposition
                import re
                sentences = re.split(r'(?<=[.!?])\s+', text)
                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) < 10:
                        continue
                    with self.lock:
                        self.stats["facts_processed"] += 1
                    
                    # Check surprise threshold - need embedding
                    try:
                        if hasattr(self.droid, 'anchor') and hasattr(self.droid, 'memory'):
                            emb = self.droid.anchor.embed(sent)
                            if emb.dim() > 1:
                                emb = emb.squeeze(0)
                            _, surprise = self.droid.memory.compute_surprise(emb)
                            if surprise < self.surprise_threshold:
                                continue
                    except:
                        pass
                    
                    # Teach
                    try:
                        result = self.droid.teach(sent, source="continuous_stream")
                        with self.lock:
                            self.stats["facts_absorbed"] += result.get("propositions", 1)
                        facts_this_minute += 1
                    except Exception as e:
                        print(f"Continuous stream teach error: {e}")
                
                self.queue.task_done()
                
            except Exception as e:
                print(f"Continuous stream loop error: {e}")
                time.sleep(0.5)

    def ingest(self, text: str):
        """Ingest text into stream queue."""
        if not self.running or self.paused:
            # If not running, queue anyway for later
            pass
        
        # Rate limiting: if queue too large, drop
        if self.queue.qsize() > 100:
            with self.lock:
                self.stats["dropped"] += 1
            return False
        
        self.queue.put(text)
        with self.lock:
            self.stats["queue_depth"] = self.queue.qsize()
        return True

    def ingest_file(self, path: str):
        """Ingest chat log file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.ingest(line)
        except Exception as e:
            print(f"Failed to ingest file {path}: {e}")

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def get_status(self) -> Dict:
        with self.lock:
            return {
                "running": self.running and not self.paused,
                "paused": self.paused,
                "facts_processed": self.stats["facts_processed"],
                "facts_absorbed": self.stats["facts_absorbed"],
                "queue_depth": self.stats["queue_depth"],
                "dropped": self.stats["dropped"],
            }

    def get_stats(self) -> Dict:
        return self.get_status()
