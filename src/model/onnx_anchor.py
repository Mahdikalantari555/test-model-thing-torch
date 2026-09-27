import os
from pathlib import Path
from typing import Union, List
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

CACHE_DIR = os.path.expanduser("~/.cache/huggingface")

class OnnxMiniLM:
    """Sub-25MB INT8 ONNX MiniLM semantic feature extractor."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        self.model_path = self._locate_or_fetch_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=self.cache_dir
        )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.model_path, opts, providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in self.session.get_inputs()]

    def _locate_or_fetch_model(self) -> str:
        # Search existing cached onnx model
        candidates = list(Path(self.cache_dir).glob("**/model_quantized.onnx"))
        if candidates:
            return str(candidates[0])
        
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id="Xenova/all-MiniLM-L6-v2",
            filename="onnx/model_quantized.onnx",
            cache_dir=self.cache_dir
        )

    def embed(self, texts: Union[str, List[str]], to_torch: bool = True) -> Union[torch.Tensor, np.ndarray]:
        single = isinstance(texts, str)
        text_list = [texts] if single else texts
        
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
        token_embeddings = outputs[0]  # (B, SeqLen, 384)
        
        # Mean pooling with attention mask
        mask = np.expand_dims(encoded["attention_mask"], -1).astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask, axis=1)
        sum_mask = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = sum_embeddings / sum_mask
        
        # L2 normalize
        norms = np.linalg.norm(pooled, ord=2, axis=1, keepdims=True)
        pooled = pooled / np.clip(norms, a_min=1e-9, a_max=None)
        
        if single:
            pooled = pooled[0]
            
        if to_torch:
            return torch.from_numpy(pooled).float()
        return pooled
