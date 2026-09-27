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

    def synthesize_answer(self, query: str, context: Optional[str] = None, timeout: float = 20.0) -> str:
        """Synthesize fluent answer given query and retrieved Droid plastic memory context."""
        if context:
            system_prompt = (
                "You are an intelligent assistant. Use the following verified memory facts "
                "from your plastic memory bank as ground truth to answer the user query.\n\n"
                f"Memory Context:\n{context}"
            )
        else:
            system_prompt = "You are a helpful and knowledgeable assistant."

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
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            if context:
                return f"{context}\n\n(Note: LLM provider unavailable: {e})"
            return f"I do not have information on this topic and LLM provider is unavailable: {e}"
