from src.model.teacher import OpenAITeacher

def test_teacher_fallback():
    # Use dummy unreachable URL to test fallback
    teacher = OpenAITeacher(base_url="http://127.0.0.1:9999/v1", api_key="test")
    raw = "Remote sensing is observing Earth from orbit."
    props = teacher.distill_propositions(raw, timeout=0.5)
    assert len(props) == 1
    assert props[0] == raw

def test_teacher_connection_test_fallback():
    teacher = OpenAITeacher(base_url="http://127.0.0.1:9999/v1", api_key="test")
    ok, msg = teacher.test_connection(timeout=0.5)
    assert ok is False
    assert len(msg) > 0

def test_teacher_synthesize_answer_fallback():
    teacher = OpenAITeacher(base_url="http://127.0.0.1:9999/v1", api_key="test")
    ans, meta = teacher.synthesize_answer("What is RS?", context="Remote sensing is observing.", timeout=0.5)
    assert "Remote sensing is observing." in ans
    assert meta["source"] == "plastic_memory_only"
