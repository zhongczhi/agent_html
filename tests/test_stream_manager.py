import pytest
from backend.chat.stream_manager import StreamJob, STREAM_REGISTRY, get_or_create_job, get_job, clear_job


def test_create_stream_job():
    job = StreamJob(conversation_id="test-123")
    assert job.conversation_id == "test-123"
    assert job.status == "active"
    assert job.tokens == []


def test_get_or_create_job_creates_new():
    clear_job("new-conv")
    job = get_or_create_job("new-conv", [{"role": "user", "content": "hi"}])
    assert job.conversation_id == "new-conv"
    assert job.status == "active"
    assert "new-conv" in STREAM_REGISTRY


def test_get_or_create_job_returns_existing():
    clear_job("existing-conv")
    job1 = get_or_create_job("existing-conv", [{"role": "user", "content": "hi"}])
    job2 = get_or_create_job("existing-conv", [{"role": "user", "content": "hi"}])
    assert job1 is job2  # Same instance


def test_job_status_transitions():
    job = StreamJob(conversation_id="status-test")
    assert job.status == "active"
    job.mark_completed()
    assert job.status == "completed"


def test_job_mark_failed():
    job = StreamJob(conversation_id="fail-test")
    job.mark_failed("test error")
    assert job.status == "failed"
    assert job.error == "test error"


def test_job_append_token():
    job = StreamJob(conversation_id="token-test")
    job.append_token("Hello")
    job.append_token(" ")
    job.append_token("World")
    assert job.tokens == ["Hello", " ", "World"]
    assert job.get_full_content() == "Hello World"


def test_clear_job():
    clear_job("to-clear")
    job = get_or_create_job("to-clear", [])
    assert "to-clear" in STREAM_REGISTRY
    clear_job("to-clear")
    assert "to-clear" not in STREAM_REGISTRY


def test_get_job_not_found():
    clear_job("nonexistent")
    result = get_job("nonexistent")
    assert result is None