
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

# ponytail: GGUF backend using llama-cpp-python with optional GPU offload

class GgufBackend:
    """GGUF model loading, inference, and management via llama.cpp"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 repo_id: Optional[str] = None, 
                 filename: Optional[str] = None,
                 n_ctx: int = 4096,
                 n_gpu_layers: int = -1,
                 verbose: bool = False,
                 **params):
        self.model_path = model_path
        self.repo_id = repo_id
        self.filename = filename
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.params = params
        self.verbose = verbose
        self.model = None
        self.tokenizer = None
        
        # Try to load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        elif repo_id and filename:
            # Try to download and load
            try:
                path = self.download_model(repo_id, filename)
                self.load_model(path)
            except Exception as e:
                if verbose:
                    print(f"Failed to auto-download model: {e}")

    def load_model(self, model_path: str):
        """Load GGUF model from local path using llama-cpp-python."""
        try:
            from llama_cpp import Llama
            self.model = Llama(
                model_path=model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
                **self.params
            )
            self.model_path = model_path
            print(f"Loaded GGUF model from {model_path}")
        except ImportError:
            print("llama-cpp-python not installed, using mock backend for testing")
            self.model = MockLlama(model_path)
        except Exception as e:
            print(f"Failed to load GGUF model {model_path}: {e}")
            # Fallback to mock for testing
            self.model = MockLlama(model_path)

    def generate(self, prompt: str, max_tokens: int = 512, 
                 temperature: float = 0.7, top_p: float = 0.9,
                 top_k: int = 40, repeat_penalty: float = 1.1,
                 stream: bool = False, callback: Optional[Callable] = None) -> str:
        """Generate tokens with optional streaming callback."""
        if self.model is None:
            return f"[Mock generation for: {prompt[:50]}...]"
        
        try:
            # Prepare generation kwargs
            kwargs = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repeat_penalty": repeat_penalty,
            }
            
            if stream and callback:
                # Streaming mode
                full_text = ""
                for output in self.model.create_completion(**kwargs, stream=True):
                    token = output["choices"][0]["text"]
                    full_text += token
                    callback(token)
                return full_text
            else:
                result = self.model.create_completion(**kwargs)
                return result["choices"][0]["text"]
        except Exception as e:
            print(f"Generation error: {e}")
            return f"[Generation failed: {e}]"

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to token ids."""
        if self.model is None:
            # Mock tokenization: split by whitespace and hash
            return [hash(w) % 10000 for w in text.split()]
        try:
            return self.model.tokenize(text.encode('utf-8'))
        except:
            return [ord(c) for c in text[:100]]

    def detokenize(self, tokens: List[int]) -> str:
        """Detokenize token ids to text."""
        if self.model is None:
            return f"[Mock detokenize {len(tokens)} tokens]"
        try:
            return self.model.detokenize(tokens).decode('utf-8', errors='ignore')
        except:
            return "".join([chr(t % 256) for t in tokens])

    def get_kv_cache(self):
        """Get current KV cache (if supported)."""
        if self.model is None:
            return None
        try:
            # llama-cpp-python may expose kv cache via model._model or similar
            # This is version-dependent, return None if not available
            return getattr(self.model, 'kv_cache', None)
        except:
            return None

    def set_kv_cache(self, kv):
        """Set KV cache (if supported)."""
        if self.model is None:
            return
        try:
            # Attempt to set cache
            if hasattr(self.model, 'kv_cache'):
                self.model.kv_cache = kv
        except:
            pass

    def download_model(self, repo_id: str, filename: str, local_dir: str = "models") -> str:
        """Download model from Hugging Face Hub using huggingface_hub."""
        try:
            from huggingface_hub import hf_hub_download
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
            return path
        except ImportError:
            # Fallback: try to use snapshot_download or manual
            raise ImportError("huggingface_hub not installed, cannot download model")
        except Exception as e:
            # If filename contains wildcard, need to list files
            if "*" in filename:
                try:
                    from huggingface_hub import list_repo_files, hf_hub_download
                    files = list_repo_files(repo_id)
                    # Simple glob matching: find file containing pattern without *
                    pattern = filename.replace("*", "")
                    matched = [f for f in files if pattern in f and f.endswith(".gguf")]
                    if matched:
                        # Pick first or smallest?
                        matched.sort()
                        chosen = matched[0]
                        Path(local_dir).mkdir(parents=True, exist_ok=True)
                        path = hf_hub_download(repo_id=repo_id, filename=chosen, local_dir=local_dir)
                        return path
                except Exception as e2:
                    print(f"Wildcard download failed: {e2}")
            raise e


class MockLlama:
    """Mock Llama for testing without actual GGUF model."""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.n_ctx = 4096
    
    def create_completion(self, prompt: str, max_tokens: int = 512, **kwargs):
        # Mock generation: echo prompt with some transformation
        # For testing, return prompt + " [generated]"
        text = f"Based on context, here is synthesis for: {prompt[:100]}... [Mock LLM output with {max_tokens} tokens]"
        return {"choices": [{"text": text}]}
    
    def tokenize(self, text: bytes):
        return [ord(c) % 1000 for c in text.decode('utf-8', errors='ignore')[:100]]
    
    def detokenize(self, tokens):
        return "".join([chr(t % 128) for t in tokens]).encode('utf-8')
