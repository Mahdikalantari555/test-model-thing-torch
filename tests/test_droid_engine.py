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
