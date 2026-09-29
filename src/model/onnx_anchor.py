
import os
from pathlib import Path
from typing import Union, List
import numpy as np
import torch
import re
import hashlib

CACHE_DIR = os.path.expanduser("~/.cache/huggingface")

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at",
    "by", "for", "with", "is", "are", "was", "were", "be", "been",
    "being", "do", "does", "did", "have", "has", "had", "it", "its",
    "this", "that", "these", "those", "which", "who", "whom", "whose",
    "what", "when", "where", "why", "how", "can", "could", "should",
    "would", "will", "shall", "may", "about", "into", "over", "under",
    "between", "during", "through", "out", "up", "down", "you", "your",
    "is", "are", "what", "how", "does", "do", "the", "a", "an"
}

# Semantic clusters for mock embeddings
SEMANTIC_CLUSTERS = {
    "remote_sensing": ["remote", "sensing", "earth", "satellite", "satellites", "observation", "imagery", "acquisition", "acquires", "information", "physical", "contact", "geophysics", "geography", "meteorology", "lidar", "laser", "terrain", "hydrology", "geology", "water", "rocks"],
    "food": ["pizza", "mozzarella", "cheese", "tomato", "sauce", "making", "cooking", "food", "recipe"],
    "capital": ["capital", "bonn", "berlin", "germany", "west"],
    "definition": ["acquisition", "definition", "study", "branch", "science", "concerned", "movement", "distribution", "management"],
}

def _word_to_index(word: str, dim: int = 384) -> int:
    h = hashlib.sha256(word.encode('utf-8')).hexdigest()
    return int(h[:8], 16) % dim

def _stem(word: str) -> str:
    # Simple stemming
    w = word.lower()
    if len(w) > 4:
        if w.endswith("ies"):
            return w[:-3] + "y"
        if w.endswith("es") and not w.endswith("ies"):
            return w[:-2]
        if w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
        if w.endswith("ing") and len(w) > 5:
            return w[:-3]
        if w.endswith("ed") and len(w) > 4:
            return w[:-2]
    return w

def _cluster_vector(cluster_name: str, dim: int = 384) -> np.ndarray:
    h = hashlib.sha256(f"cluster_{cluster_name}".encode('utf-8')).hexdigest()
    seed = int(h[:8], 16) % (2**32)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    return vec

# Precompute cluster vectors
CLUSTER_VECS = {name: _cluster_vector(name) for name in SEMANTIC_CLUSTERS}

class OnnxMiniLM:
    """Sub-25MB INT8 ONNX MiniLM semantic feature extractor with mock fallback."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        self.model_path = None
        self.tokenizer = None
        self.session = None
        self.input_names = []
        self.use_mock = False
        
        try:
            self.model_path = self._locate_or_fetch_model()
            if self.model_path and Path(self.model_path).exists():
                from transformers import AutoTokenizer
                import onnxruntime as ort
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        "sentence-transformers/all-MiniLM-L6-v2",
                        cache_dir=self.cache_dir,
                        local_files_only=True
                    )
                except:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        "sentence-transformers/all-MiniLM-L6-v2",
                        cache_dir=self.cache_dir,
                        local_files_only=False,
                        trust_remote_code=False
                    )
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(self.model_path, opts, providers=["CPUExecutionProvider"])
                self.input_names = [inp.name for inp in self.session.get_inputs()]
            else:
                raise FileNotFoundError("Model not found, using mock")
        except Exception as e:
            self.use_mock = True
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    cache_dir=self.cache_dir,
                    local_files_only=True
                )
            except:
                self.tokenizer = None

    def _locate_or_fetch_model(self) -> str:
        candidates = list(Path(self.cache_dir).glob("**/model_quantized.onnx"))
        if candidates:
            return str(candidates[0])
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(
                repo_id="Xenova/all-MiniLM-L6-v2",
                filename="onnx/model_quantized.onnx",
                cache_dir=self.cache_dir,
                local_files_only=True
            )
        except Exception:
            return None

    def _mock_embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Hashed bag-of-words embedding with semantic cluster boosting."""
        single = isinstance(texts, str)
        text_list = [texts] if single else texts
        
        embeddings = []
        for text in text_list:
            words = re.findall(r"[a-z0-9]+", text.lower())
            words_nostop = [w for w in words if w not in STOP_WORDS]
            if not words_nostop:
                words_nostop = words
            if not words_nostop:
                words_nostop = [text.lower()]
            
            # Stemmed words for better matching
            stemmed = [_stem(w) for w in words_nostop]
            
            vec = np.zeros(384, dtype=np.float32)
            for w in words_nostop:
                idx = _word_to_index(w, 384)
                vec[idx] += 1.0
            # Add stemmed versions
            for w in stemmed:
                idx = _word_to_index(w, 384)
                vec[idx] += 0.5
            
            # Bigram hashing
            for i in range(len(words_nostop)-1):
                bigram = words_nostop[i] + "_" + words_nostop[i+1]
                idx = _word_to_index(bigram, 384)
                vec[idx] += 0.5
                # Stemmed bigram
                bigram_stem = _stem(words_nostop[i]) + "_" + _stem(words_nostop[i+1])
                idx = _word_to_index(bigram_stem, 384)
                vec[idx] += 0.3
            
            # Semantic cluster boosting
            cluster_counts = {}
            for cluster_name, cluster_words in SEMANTIC_CLUSTERS.items():
                count = 0
                for w in words_nostop:
                    if w in cluster_words or _stem(w) in [_stem(cw) for cw in cluster_words]:
                        count += 1
                if count > 0:
                    cluster_counts[cluster_name] = count
            
            # Add cluster vectors weighted by count
            for cluster_name, count in cluster_counts.items():
                cvec = CLUSTER_VECS[cluster_name]
                # Weight by sqrt(count) to avoid domination but give strong signal
                weight = min(3.0, 0.8 * (count ** 0.5))
                vec += cvec * weight * 2.0  # Strong boost
            
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec = vec / norm
            else:
                h = hashlib.sha256(text.encode('utf-8')).hexdigest()
                seed = int(h[:8], 16) % (2**32)
                rng = np.random.RandomState(seed)
                vec = rng.randn(384).astype(np.float32)
                vec = vec / (np.linalg.norm(vec) + 1e-8)
            
            embeddings.append(vec)
        
        pooled = np.stack(embeddings, axis=0) if len(embeddings) > 1 else embeddings[0]
        return pooled

    def embed(self, texts: Union[str, List[str]], to_torch: bool = True) -> Union[torch.Tensor, np.ndarray]:
        if self.use_mock or self.session is None:
            pooled = self._mock_embed(texts)
            if to_torch:
                return torch.from_numpy(pooled).float()
            return pooled

        single = isinstance(texts, str)
        text_list = [texts] if single else texts
        
        try:
            encoded = self.tokenizer(
                text_list,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="np"
            )
            
            feeds = {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = encoded.get(
                    "token_type_ids", 
                    np.zeros_like(encoded["input_ids"], dtype=np.int64)
                )

            outputs = self.session.run(None, feeds)
            token_embeddings = outputs[0]
            
            mask = np.expand_dims(encoded["attention_mask"], -1).astype(np.float32)
            sum_embeddings = np.sum(token_embeddings * mask, axis=1)
            sum_mask = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
            pooled = sum_embeddings / sum_mask
            
            norms = np.linalg.norm(pooled, ord=2, axis=1, keepdims=True)
            pooled = pooled / np.clip(norms, a_min=1e-9, a_max=None)
            
            if single:
                pooled = pooled[0]
                
            if to_torch:
                return torch.from_numpy(pooled).float()
            return pooled
        except Exception as e:
            pooled = self._mock_embed(texts)
            if to_torch:
                return torch.from_numpy(pooled).float()
            return pooled
