import json
import os
import urllib.request
import urllib.error
from typing import List, Optional, Dict, Any

# ponytail: Stdlib urllib for OpenAI-compatible teacher endpoint; no openai sdk needed.

class OpenAITeacher:
    """
    Teacher agent connecting to any OpenAI-compatible endpoint
    (Local Ollama, LM Studio, vLLM, Groq, OpenRouter) to distill
    complex documents into propositions for Droid absorption or
    synthesize fluent answers from plastic memory.
    """
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "llama3:latest"
    ):
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.model = model

    def test_connection(self, timeout: float = 5.0) -> tuple[bool, str]:
        """Test connectivity and authentication to the provider endpoint."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    return True, f"Connected to {self.model} at {self.base_url}"
                return False, f"HTTP status {resp.status}"
        except urllib.error.HTTPError as e:
            return False, f"HTTP {e.code}: {e.reason}"
        except Exception as e:
            return False, str(e)

    def distill_propositions(self, raw_text: str, timeout: float = 15.0) -> List[str]:
        """Distill raw text into standalone factual propositions."""
        prompt = (
            "Extract 3 to 5 core standalone factual propositions from the following text. "
            "Return ONLY the propositions, one per line without bullet points or numbering.\n\n"
            f"Text:\n{raw_text}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a concise fact distillation engine."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                lines = [line.strip("- *0123456789.").strip() for line in content.splitlines() if line.strip()]
                return lines if lines else [raw_text]
        except Exception:
            # Fallback to local splitting if endpoint is unreachable
            return [raw_text]

    def synthesize_answer(self, query: str, context: Optional[str] = None, droid_name: str = "Droid", timeout: float = 45.0) -> tuple[str, Dict[str, Any]]:
        """
        Synthesize answer given query and retrieved Droid plastic memory context.
        Returns (response_text, provenance_metadata).
        """
        if context:
            system_prompt = (
                f"You are the voice and synthesis layer for {droid_name}, an AI with plastic on-device RTU memory. "
                "Base your answer strictly on the following verified facts retrieved from plastic memory. "
                "Do not state that you are ChatGPT or OpenAI; you are speaking on behalf of this Droid.\n\n"
                f"Verified Plastic Memory:\n{context}"
            )
            mode = "llm_grounded_memory"
        else:
            system_prompt = (
                f"You are {droid_name}, an AI assistant. Answer the user question accurately and concisely. "
                "When asked who you are, identify as this Droid assistant backed by your configured model."
            )
            mode = "llm_general_knowledge"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.3
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                text = result["choices"][0]["message"]["content"]
                meta = {
                    "source": mode,
                    "provider_model": self.model,
                    "endpoint": self.base_url,
                    "has_context": bool(context)
                }
                return text, meta
        except Exception as e:
            if context:
                text = f"{context}\n\n*(Note: LLM provider call failed: {e})*"
                meta = {"source": "plastic_memory_only", "error": str(e)}
                return text, meta
            text = f"I do not have information on this topic and external LLM provider is unavailable ({e})."
            meta = {"source": "unanswered", "error": str(e)}
            return text, meta
