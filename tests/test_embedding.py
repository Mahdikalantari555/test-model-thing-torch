import math
import pytest
import torch
import torch.nn as nn
from src.model.rtu import Encoder

def test_embedding_constructor():
    dim = 512
    encoder = Encoder(dim=dim, vocab_size=256)
    assert isinstance(encoder.embed, nn.Embedding)
    assert encoder.embed.num_embeddings == 256
    assert encoder.embed.embedding_dim == dim

def test_embedding_forward_shape():
    dim = 512
    encoder = Encoder(dim=dim)
    out = encoder(0)
    assert out.shape == (dim,)
    assert out.dtype == torch.float32

def test_embedding_different_rows():
    encoder = Encoder(dim=512)
    out0 = encoder(0)
    out255 = encoder(255)
    assert not torch.allclose(out0, out255)

def test_embedding_init_std():
    dim = 512
    encoder = Encoder(dim=dim)
    expected_std = 1.0 / math.sqrt(dim)
    actual_std = encoder.embed.weight.std().item()
    # verify normal init with std near 1/sqrt(dim)
    assert math.isclose(actual_std, expected_std, rel_tol=0.15)
