
import math
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F

# ponytail: combined confidence estimation - retrieval + generation fused via harmonic mean

def retrieval_confidence(similarity: float, surprise: Optional[float] = None, 
                         health_metrics: Optional[Dict[str, float]] = None) -> float:
    """
    Compute retrieval confidence from max semantic similarity, surprise alignment, memory health.
    Returns float [0,1]
    """
    # Base confidence from similarity
    # similarity in [0,1], but we want to map to confidence
    # High similarity >0.8 => high confidence
    conf = similarity  # start with similarity
    
    # Adjust for surprise if available: low surprise (expected) might indicate good alignment?
    # Actually per spec, surprise alignment contributes - but we need to interpret
    # For now, if surprise provided, high surprise reduces confidence slightly (novelty might be less reliable)
    # Or low surprise increases confidence? Let's say surprise in [0,2], 0=expected, 2=very novel
    # For retrieval, if we retrieved a fact that was surprising to RTU, it might be less reliable?
    # We'll implement: confidence *= (1 - 0.1*surprise) clamped
    if surprise is not None:
        surprise_factor = max(0.0, min(1.0, 1.0 - 0.1 * surprise))
        conf = conf * (0.8 + 0.2 * surprise_factor)
    
    # Adjust for health metrics: low interference, high quality => higher confidence
    if health_metrics:
        interference = health_metrics.get("interference", 0.5)
        quality = health_metrics.get("retrieval_quality", 0.5)
        # Low interference good
        interference_factor = 1.0 - interference * 0.3  # if interference 0.7, factor 0.79
        # High quality good
        quality_factor = 0.7 + 0.3 * quality  # quality 0.8 => 0.94
        conf = conf * interference_factor * quality_factor
    
    return max(0.0, min(1.0, conf))

def generation_confidence(token_logprobs: Optional[List[float]] = None,
                          self_consistency: Optional[float] = None,
                          entropy: Optional[float] = None) -> float:
    """
    Compute generation confidence from mean token log-prob, self-consistency, normalized entropy.
    Returns float [0,1]
    """
    conf = 0.5  # default
    
    # From token logprobs: mean logprob closer to 0 => higher confidence
    # logprob in negative, e.g., -0.1 high confidence, -2 low confidence
    if token_logprobs:
        mean_logprob = sum(token_logprobs) / len(token_logprobs)
        # Convert: exp(mean_logprob) is prob, but mean_logprob negative
        # Map: logprob -0.1 => conf ~0.9, -1.0 => 0.5, -2.0 => 0.2
        # Use: conf = exp(mean_logprob) clamped, or 1/(1+exp(-mean_logprob*?))
        # Simple: conf = exp(mean_logprob) where mean_logprob negative, so exp(-0.1)=0.9, exp(-1)=0.37
        prob = math.exp(mean_logprob) if mean_logprob > -10 else 0.0
        conf = prob
    
    # Self-consistency: 3/3 samples agree => high confidence
    # self_consistency in [0,1] where 1 = all agree
    if self_consistency is not None:
        conf = conf * 0.6 + self_consistency * 0.4
    
    # Normalized entropy: low entropy => high confidence
    # entropy in [0,1] normalized, 0 low entropy (confident), 1 high entropy (uncertain)
    if entropy is not None:
        conf = conf * (1.0 - 0.5 * entropy)
    
    return max(0.0, min(1.0, conf))

def fuse_confidence(Cr: float, Cg: float, weights: Optional[Dict[str, float]] = None) -> float:
    """
    Combine retrieval and generation confidence as weighted harmonic mean:
    C = 2 / (1/Cr + 1/Cg) with weights configurable per query type.
    Harmonic mean penalizes imbalance.
    """
    if weights is None:
        weights = {"retrieval": 0.5, "generation": 0.5}
    
    w_r = weights.get("retrieval", 0.5)
    w_g = weights.get("generation", 0.5)
    
    # Avoid division by zero
    eps = 1e-8
    Cr = max(eps, Cr)
    Cg = max(eps, Cg)
    
    # Weighted harmonic mean: C = 1 / (w_r/Cr + w_g/Cg) * (w_r + w_g) ??? 
    # Spec says: C = 2 / (1/Cr + 1/Cg) - simple harmonic mean
    # With weights: we can do weighted harmonic mean
    # For simplicity, implement spec's simple harmonic mean, but allow weights to adjust
    # Weighted version: C = (w_r + w_g) / (w_r/Cr + w_g/Cg)
    
    # If equal weights, this reduces to 2/(1/Cr+1/Cg)
    numerator = w_r + w_g
    denominator = w_r / Cr + w_g / Cg
    
    if denominator < eps:
        return 0.0
    
    fused = numerator / denominator
    return max(0.0, min(1.0, fused))

def estimate_confidence(retrieval_conf: float, generation_conf: float, 
                        weights: Optional[Dict[str, float]] = None) -> float:
    """Alias for fuse_confidence."""
    return fuse_confidence(retrieval_conf, generation_conf, weights)

def confidence_tier(confidence: float) -> str:
    """Define thresholds: high (>0.8)=auto-execute, medium (0.5-0.8)=show with citation, low (<0.5)=defer."""
    if confidence > 0.8:
        return "high"
    elif confidence >= 0.5:
        return "medium"
    else:
        return "low"

def format_low_confidence_response(query: str, facts: List[Dict], confidence: float) -> str:
    """Format response for low confidence: include 'I'm uncertain; here's what I found...' with citations."""
    if not facts:
        return f"I'm uncertain about '{query}'. I don't have relevant facts in memory."
    
    citations = "\n".join([f"- {f.get('text','')} [sim={f.get('similarity',0):.2f}]" for f in facts[:3]])
    return f"I'm uncertain (confidence {confidence:.2f}); here's what I found:\n{citations}"
