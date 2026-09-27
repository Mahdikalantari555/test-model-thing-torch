import os
import tempfile
import pytest
import torch
from src.data import BytePairDataset, StreamingTextDataset, load_corpus

def test_bytepair_dataset_interface():
    text = "Hello"
    ds = BytePairDataset(text)
    assert len(ds) == 4
    c, n, is_end = ds[0]
    assert c == ord('H')
    assert n == ord('e')
    assert not is_end

    c_last, n_last, is_end_last = ds[-1]
    assert c_last == ord('l')
    assert n_last == ord('o')
    assert is_end_last

def test_bytepair_collate_fn():
    text = "ABCD"
    ds = BytePairDataset(text)
    batch = [ds[0], ds[1]]
    currs, nexts, ends = BytePairDataset.collate_fn(batch)
    assert currs.shape == (2,)
    assert nexts.shape == (2,)
    assert ends.shape == (2,)

def test_streaming_dataset():
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        f.write("Line one\nLine two\n")
        f_name = f.name

    try:
        ds = StreamingTextDataset([f_name], seed=42)
        items = list(ds)
        assert len(items) > 0
        assert all(isinstance(c, int) and isinstance(n, int) for c, n, _ in items)
    finally:
        os.remove(f_name)

def test_load_corpus_named_and_deterministic():
    ds1 = load_corpus("tinystories", seed=42)
    ds2 = load_corpus("tinystories", seed=42)
    assert len(ds1) == len(ds2)
    assert [ds1[i] for i in range(len(ds1))] == [ds2[i] for i in range(len(ds2))]

    wiki_ds = load_corpus("simplewiki", seed=42)
    assert len(wiki_ds) > 0
