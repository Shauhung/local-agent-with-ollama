import pytest

from local_agent_server import parse_agent_request
from local_ollama_mcp_agent import DEFAULT_MODEL


def test_parse_agent_request_accepts_minimal_payload() -> None:
    agent_request = parse_agent_request({"message": " hello "})

    assert agent_request.message == "hello"
    assert agent_request.model == DEFAULT_MODEL
    assert agent_request.trace_enabled is False
    assert agent_request.session_id == "default"
    assert agent_request.history_limit == 20


def test_parse_agent_request_accepts_model_and_trace() -> None:
    agent_request = parse_agent_request(
        {
            "message": "ping",
            "model": "qwen2.5-coder:7b",
            "trace": True,
            "session_id": "phone",
            "history_limit": 8,
        }
    )

    assert agent_request.message == "ping"
    assert agent_request.model == "qwen2.5-coder:7b"
    assert agent_request.trace_enabled is True
    assert agent_request.session_id == "phone"
    assert agent_request.history_limit == 8


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": ""},
        {"message": ["hello"]},
        {"message": "hello", "model": ""},
        {"message": "hello", "trace": "yes"},
        {"message": "hello", "session_id": ""},
        {"message": "hello", "history_limit": "20"},
    ],
)
def test_parse_agent_request_rejects_invalid_payloads(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        parse_agent_request(payload)
