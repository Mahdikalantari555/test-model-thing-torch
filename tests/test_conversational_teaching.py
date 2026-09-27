import pytest
from src.model.droid import DroidEngine

def test_conversational_teaching_zero_collapse():
    droid = DroidEngine(name="lifelong-droid")

    # Domain 1: Remote sensing
    rs_text = (
        "Remote sensing is the acquisition of information about an object or phenomenon "
        "without making physical contact with the object, in contrast to on-site observation. "
        "The term is applied especially to acquiring information about Earth and other planets."
    )
    res1 = droid.teach(rs_text, source="domain_1")
    assert res1["status"] == "learned"

    # Domain 2: Hydrology
    hydro_text = (
        "Hydrology is the scientific study of the movement, distribution, and management of water "
        "on Earth and other planets, including the water cycle, water resources, and environmental watershed sustainability."
    )
    res2 = droid.teach(hydro_text, source="domain_2")
    assert res2["status"] == "learned"

    # Domain 3: Geology
    geo_text = (
        "Geology is a branch of natural science concerned with Earth and other astronomical objects, "
        "the rocks of which it is composed, and the processes by which they change over time."
    )
    res3 = droid.teach(geo_text, source="domain_3")
    assert res3["status"] == "learned"

    # Test that Domain 1 was NOT forgotten (Zero catastrophic collapse)
    rs_hits = droid.recall("Define remote sensing")
    assert len(rs_hits) > 0
    assert "acquisition of information" in rs_hits[0]["text"]
    assert "physical contact" in rs_hits[0]["text"]

    # Test Domain 2 recall
    hydro_hits = droid.recall("What is hydrology?")
    assert len(hydro_hits) > 0
    assert "study of the movement" in hydro_hits[0]["text"]

    # Test Domain 3 recall
    geo_hits = droid.recall("What does geology study?")
    assert len(geo_hits) > 0
    assert "concerned with Earth" in geo_hits[0]["text"]

def test_conversational_teaching_never_goes_silent():
    droid = DroidEngine(name="robust-droid")
    
    # Train 4 times consecutively (previously caused collapse into silence in naive character model)
    for i in range(4):
        droid.teach(f"Fact number {i}: Synthetic aperture radar can penetrate clouds and rain at frequency band {i}.")

    # Ask 5 diverse queries
    queries = [
        "Can radar penetrate clouds?",
        "What happens in rain?",
        "Tell me about astronomy",
        "Hello droid",
        "Define remote sensing"
    ]
    for q in queries:
        ans = droid.chat(q)
        assert isinstance(ans, str)
        assert len(ans.strip()) > 0, f"Droid went silent on query: {q}"
