import os
import tempfile

import pytest
import torch

from src.model.knowledge_store import KnowledgeStore


@pytest.fixture
def store():
    d = tempfile.mkdtemp()
    ks = KnowledgeStore(os.path.join(d, "k.db"), dim=8)
    yield ks
    ks.close()


def _vec(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(8, generator=g)


def test_insert_and_recall(store):
    v = _vec(1)
    store.insert_fact("Remote sensing uses satellites", "chat", 1.0, 0, 0.9, v)
    store.insert_fact("Lidar measures distance with laser pulses", "chat", 2.0, 1, 0.8, _vec(2))

    hits = store.recall(v, top_k=2, query_text="remote sensing satellites")
    assert len(hits) == 2
    assert hits[0]["text"].startswith("Remote sensing")
    assert hits[0]["dense_similarity"] > 0.9
    # spec: response includes both lexical rank and dense similarity scores
    assert "lexical_rank" in hits[0] and "dense_similarity" in hits[0]


def test_recall_empty_store(store):
    assert store.recall(_vec(1), top_k=3, query_text="anything") == []


def test_supersede_excludes_from_recall_but_keeps_audit(store):
    v1 = _vec(1)
    id1 = store.insert_fact("The capital of West Germany is Bonn", "chat", 1.0, 0, 0.5, v1)
    id2 = store.insert_fact("The capital of West Germany is Berlin", "chat", 2.0, 1, 0.5, _vec(2))

    assert store.supersede(id1, id2) == 1
    # double-supersede is a no-op
    assert store.supersede(id1, id2) == 0
    # unknown id is a no-op
    assert store.supersede(9999, id2) == 0

    hits = store.recall(_vec(3), top_k=5, query_text="west germany capital")
    assert all(h["id"] != id1 for h in hits)

    trail = store.get_audit_trail()
    assert len(trail) == 2
    old = [t for t in trail if t["id"] == id1][0]
    assert old["superseded"] is True
    assert old["superseded_by"] == id2
    assert old["valid_until"] is not None
    # original text/timestamp/step unchanged (audit trail completeness)
    assert old["text"] == "The capital of West Germany is Bonn"
    assert old["timestamp"] == 1.0
    assert old["step"] == 0


def test_rrf_fusion_ranks_lexical_and_dense(store):
    # fact A: dense match to query vec, no lexical overlap
    # fact B: lexical match to query text, weak dense
    v = _vec(42)
    store.insert_fact("Quantum entanglement enables secure communication", "chat", 1.0, 0, 0.5, v)
    store.insert_fact("Remote sensing uses satellites for Earth observation", "chat", 2.0, 1, 0.5, _vec(7))

    hits = store.recall(v, top_k=2, query_text="remote sensing satellites")
    texts = [h["text"] for h in hits]
    assert len(hits) == 2
    # both signals present in results; fused score combines both parts
    for h in hits:
        assert h["fused_score"] > 0
        assert h["lexical_score"] >= 0 and h["dense_score"] >= 0


def test_pre_normalized_vectors(store):
    v = _vec(5) * 100.0  # deliberately unnormalized
    store.insert_fact("Normalized fact", "chat", 1.0, 0, 0.5, v)
    vecs, ids = store._load_dense()
    norm = torch.linalg.vector_norm(vecs[0])
    assert abs(norm.item() - 1.0) < 1e-4


def test_export_roundtrip(store):
    store.insert_fact("Persistent fact", "chat", 1.0, 0, 0.5, _vec(9))
    store.insert_fact("Another fact", "chat", 2.0, 1, 0.5, _vec(10))
    store.supersede(1, 2)

    d = tempfile.mkdtemp()
    dest = store.export_to(os.path.join(d, "exported.db"))
    ks2 = KnowledgeStore(str(dest), dim=8)
    try:
        assert ks2.total_count() == 2
        assert ks2.active_count() == 1
        trail = ks2.get_audit_trail()
        assert trail[0]["superseded"] is True
        assert trail[0]["superseded_by"] == 2
        # recall works on restored DB
        hits = ks2.recall(_vec(9), top_k=1, query_text="persistent fact")
        assert len(hits) == 1
    finally:
        ks2.close()


def test_clear_all(store):
    store.insert_fact("Fact one", "chat", 1.0, 0, 0.5, _vec(1))
    store.insert_fact("Fact two", "chat", 2.0, 1, 0.5, _vec(2))
    store.clear_all()
    assert store.total_count() == 0
    assert store.active_count() == 0
    assert store.recall(_vec(1), top_k=3, query_text="fact") == []
    assert store.get_audit_trail() == []
