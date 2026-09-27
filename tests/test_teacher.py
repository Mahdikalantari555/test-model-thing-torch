from src.model.teacher import OpenAITeacher

def test_teacher_fallback():
    # Use dummy unreachable URL to test fallback
    teacher = OpenAITeacher(base_url="http://127.0.0.1:9999/v1", api_key="test")
    raw = "Remote sensing is observing Earth from orbit."
    props = teacher.distill_propositions(raw, timeout=0.5)
    assert len(props) == 1
    assert props[0] == raw
