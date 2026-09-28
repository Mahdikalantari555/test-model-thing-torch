import shutil
import tempfile
import torch
import pytest
from src.model.droid import DroidEngine

@pytest.fixture
def temp_droid_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)

def test_droid_engine_init():
    engine = DroidEngine(name="test-droid")
    assert engine.name == "test-droid"
    assert engine.dim == 384
    assert engine.step_count == 0
    assert len(engine.knowledge_bank) == 0

def test_droid_engine_teaching_and_recall():
    engine = DroidEngine(name="test-droid")
    rs_text = (
        "Remote sensing is the acquisition of information about an object or phenomenon "
        "without making physical contact with the object, in contrast to on-site observation. "
        "The term is applied especially to acquiring information about Earth and other planets. "
        "Remote sensing is used in numerous fields, including geophysics, geography, and meteorology."
    )
    
    result = engine.teach(rs_text, source="user_text")
    assert result["status"] == "learned"
    assert result["propositions"] >= 3
    assert result["memory_norm"] > 0
    assert engine.step_count >= 3
    assert len(engine.knowledge_bank) >= 3

    # Test recall for remote sensing query
    hits = engine.recall("What is remote sensing?")
    assert len(hits) > 0
    top_hit = hits[0]
    assert "Remote sensing is the acquisition" in top_hit["text"]
    assert top_hit["similarity"] > 0.4

    # Test chat answer
    reply = engine.chat("What is remote sensing?")
    assert "acquisition of information" in reply
    assert "physical contact" in reply

def test_droid_engine_persistence(temp_droid_dir):
    engine1 = DroidEngine(name="droid-persist")
    engine1.teach("Lidar uses laser pulses to measure exact 3D distances on terrain.")
    engine1.save_profile(temp_droid_dir)

    engine2 = DroidEngine(name="fresh-droid")
    loaded = engine2.load_profile(temp_droid_dir)
    assert loaded is True
    assert engine2.name == "droid-persist"
    assert len(engine2.knowledge_bank) == 1
    assert "Lidar uses laser" in engine2.knowledge_bank[0]["text"]
    
    reply = engine2.chat("How does lidar work?")
    assert "laser pulses" in reply


def test_anaphora_resolution_pronoun_anchoring():
    engine = DroidEngine(name="anaphora-droid")
    engine.teach(
        "Remote sensing is the acquisition of information about an object. "
        "It is used in numerous fields, including geophysics and geography.",
        source="user_text",
    )
    texts = [f["text"] for f in engine.knowledge_bank]
    assert len(texts) == 2
    # "It" anchored to the active subject from sentence 1
    assert texts[1].startswith("Remote sensing"), texts[1]
    assert "It is used" not in texts[1]


def test_anaphora_carryover_across_paragraphs():
    engine = DroidEngine(name="carryover-droid")
    engine.teach("Remote sensing is the acquisition of information about Earth.", source="p1")
    engine.teach("It uses satellites to observe the surface.", source="p2")
    texts = [f["text"] for f in engine.knowledge_bank]
    assert texts[-1].startswith("Remote sensing uses"), texts[-1]


def test_anaphora_no_subject_leaves_pronoun():
    engine = DroidEngine(name="no-antecedent-droid")
    engine.teach("It is unclear what this refers to.", source="user_text")
    texts = [f["text"] for f in engine.knowledge_bank]
    # no prior subject -> pronoun untouched
    assert texts[0].startswith("It"), texts[0]


def test_contradiction_supersedes_old_fact():
    engine = DroidEngine(name="tms-droid")
    engine.teach("The capital of West Germany is Bonn.", source="user_text")
    assert engine.knowledge.active_count() == 1

    engine.teach("The capital of West Germany is Berlin.", source="user_text")
    assert engine.knowledge.active_count() == 1
    assert result_superseded(engine, "Bonn")

    # superseded fact excluded from recall, present in audit trail
    hits = engine.recall("capital of West Germany", top_k=5, threshold=0.0)
    assert all("Bonn" not in h["text"] for h in hits)
    trail = engine.knowledge.get_audit_trail()
    assert len(trail) == 2
    old = [t for t in trail if "Bonn" in t["text"]][0]
    assert old["superseded"] is True
    assert old["superseded_by"] is not None


def test_polarity_inversion_detected():
    engine = DroidEngine(name="polarity-droid")
    engine.teach("Remote sensing requires physical contact.", source="user_text")
    engine.teach("Remote sensing does not require physical contact.", source="user_text")
    assert engine.knowledge.active_count() == 1
    assert result_superseded(engine, "requires physical contact.")


def test_unrelated_facts_not_superseded():
    engine = DroidEngine(name="no-clash-droid")
    engine.teach("Remote sensing uses satellites to observe Earth.", source="user_text")
    engine.teach("Lidar measures distance with laser pulses.", source="user_text")
    # different subjects -> no slot clash, both stay active
    assert engine.knowledge.active_count() == 2


def result_superseded(engine, old_text: str) -> bool:
    trail = engine.knowledge.get_audit_trail()
    old = [t for t in trail if old_text in t["text"]]
    return bool(old) and old[0]["superseded"] is True
