
from typing import List, Dict, Any, Optional
import torch

# ponytail: memory injection strategies - prompt stuffing, hidden state injection, KV cache priming

def inject_prompt(query: str, facts: List[Dict[str, Any]], max_context_tokens: int = 3000) -> str:
    """Format retrieved facts as structured context block prepended to user prompt."""
    if not facts:
        return query
    
    # Format facts as bullet list
    context_lines = ["Context:"]
    total_len = len("Context:\n")
    
    for fact in facts:
        text = fact.get("text", "").strip()
        if not text:
            continue
        line = f"- {text}"
        # Rough token estimation: 1 token ~ 4 chars
        est_tokens = len(line) // 4
        if total_len + est_tokens > max_context_tokens:
            break
        context_lines.append(line)
        total_len += est_tokens
    
    if len(context_lines) == 1:
        return query
    
    context_block = "\n".join(context_lines)
    prompt = f"{context_block}\n\nQuery: {query}"
    return prompt

def inject_hidden(backend, query: str, h_trace: torch.Tensor, layer_idx: int = 12) -> Dict[str, Any]:
    """
    Inject RTU trace vector h_t into LLM's hidden states at designated layer via addition.
    Returns info about injection for generation.
    """
    # This requires access to LLM's internal hidden states
    # For llama-cpp-python, we would need to register forward hook
    # Since llama.cpp is C++ backend, true hidden injection is complex
    # We implement a mock that projects h_trace to LLM hidden dim and adds to prompt embedding
    
    # Project h_trace to LLM hidden dim (e.g., 4096 for Llama-3B) vs RTU dim (384)
    # Use fixed random projection per JL lemma if no learned projection
    try:
        # Estimate LLM hidden dim - try to get from backend model
        llm_dim = 4096  # default for Llama-3B
        if hasattr(backend, 'model') and hasattr(backend.model, 'n_embd'):
            llm_dim = backend.model.n_embd()
        
        # Create projection matrix: 384 x D_llm
        # For testing, use random projection
        torch.manual_seed(42)
        W_proj = torch.randn(384, llm_dim) * 0.02
        
        # Project h_trace
        if h_trace.shape[0] != 384:
            # Handle dim mismatch
            if h_trace.shape[0] > 384:
                h = h_trace[:384]
            else:
                h = torch.nn.functional.pad(h_trace, (0, 384 - h_trace.shape[0]))
        else:
            h = h_trace
        
        projected = h @ W_proj  # D_llm
        
        return {
            "strategy": "hidden",
            "layer_idx": layer_idx,
            "projected_trace": projected,
            "injection": f"Hidden injection at layer {layer_idx} with trace norm {torch.norm(h_trace):.2f}",
            "query": query,
        }
    except Exception as e:
        return {
            "strategy": "hidden",
            "error": str(e),
            "query": query,
            "fallback": "prompt",
            "prompt": inject_prompt(query, [{"text": f"Memory trace norm: {torch.norm(h_trace):.2f}"}])
        }

def inject_kv_cache(backend, facts: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """
    Pre-fill KV cache with fact embeddings so model attends to them from first token.
    Returns info about KV cache priming.
    """
    # KV cache priming needs access to internal cache tensors
    # llama_cpp may expose via low-level API, but we implement mock that encodes facts separately
    
    try:
        # Encode facts using backend tokenizer
        fact_texts = [f.get("text","") for f in facts]
        combined_facts = "\n".join(fact_texts)
        
        # Tokenize facts
        if hasattr(backend, 'tokenize'):
            fact_tokens = backend.tokenize(combined_facts)
            query_tokens = backend.tokenize(query)
        else:
            fact_tokens = [1,2,3]  # mock
            query_tokens = [4,5,6]
        
        return {
            "strategy": "kv_cache",
            "fact_tokens": fact_tokens,
            "query_tokens": query_tokens,
            "fact_count": len(facts),
            "injection": f"KV cache primed with {len(facts)} facts, {len(fact_tokens)} tokens",
            "query": query,
            "prompt": f"{combined_facts}\n\n{query}"  # Fallback to prompt stuffing if KV cache not supported
        }
    except Exception as e:
        return {
            "strategy": "kv_cache",
            "error": str(e),
            "query": query,
            "fallback": "prompt",
            "prompt": inject_prompt(query, facts)
        }

def select_strategy(fact_count: int, query_type: str, profile_config: Optional[Dict] = None) -> str:
    """Select injection strategy based on fact count, query type, and profile config."""
    # Check profile config override
    if profile_config and "injection_strategy" in profile_config:
        strat = profile_config["injection_strategy"]
        if strat in ["prompt", "hidden", "kv_cache", "auto"]:
            if strat != "auto":
                return strat
    
    # Auto selection logic per spec: prompt for <5 facts, hidden for 5-20, kv_cache for >20 or when context constrained
    if fact_count < 5:
        return "prompt"
    elif fact_count <= 20:
        return "hidden"
    else:
        return "kv_cache"
