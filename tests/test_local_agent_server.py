import pytest

from local_agent_server import parse_agent_request
from local_ollama_mcp_agent import DEFAULT_MODEL


def test_parse_agent_request_accepts_minimal_payload() -> None:
    message, model, trace_enabled = parse_agent_request({"message": " hello "})

    assert message == "hello"
    assert model == DEFAULT_MODEL
    assert trace_enabled is False


def test_parse_agent_request_accepts_model_and_trace() -> None:
    message, model, trace_enabled = parse_agent_request(
        {"message": "ping", "model": "qwen2.5-coder:7b", "trace": True}
    )

    assert message == "ping"
    assert model == "qwen2.5-coder:7b"
    assert trace_enabled is True


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": ""},
        {"message": ["hello"]},
        {"message": "hello", "model": ""},
        {"message": "hello", "trace": "yes"},
    ],
)
def test_parse_agent_request_rejects_invalid_payloads(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        parse_agent_request(payload)
