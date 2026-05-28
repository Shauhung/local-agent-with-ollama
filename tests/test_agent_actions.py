import pytest

from local_ollama_mcp_agent import (
    ACTION_RESPONSE_FORMAT,
    ActionValidationError,
    ToolSpec,
    build_ollama_payload,
    build_system_prompt,
    parse_agent_action,
)


def test_parse_final_action() -> None:
    action = parse_agent_action('{"type": "final", "answer": "done"}')

    assert action.type == "final"
    assert action.answer == "done"


def test_parse_tool_call_action() -> None:
    action = parse_agent_action(
        '{"type": "tool_call", "reason": "need files", "tool": "list_files", "arguments": {}}'
    )

    assert action.type == "tool_call"
    assert action.reason == "need files"
    assert action.tool == "list_files"
    assert action.arguments == {}


def test_parse_tool_call_defaults_missing_arguments() -> None:
    action = parse_agent_action('{"type": "tool_call", "tool": "get_time"}')

    assert action.arguments == {}


def test_parse_markdown_json_block() -> None:
    action = parse_agent_action(
        """
        ```json
        {"type": "final", "answer": "ok"}
        ```
        """
    )

    assert action.type == "final"
    assert action.answer == "ok"


def test_parse_json_with_surrounding_text() -> None:
    action = parse_agent_action(
        'I will do this now: {"type": "final", "answer": "ok"}'
    )

    assert action.type == "final"
    assert action.answer == "ok"


def test_rejects_unknown_action_type() -> None:
    with pytest.raises(ActionValidationError, match="Action type"):
        parse_agent_action('{"type": "message", "answer": "hello"}')


def test_rejects_non_object_arguments() -> None:
    with pytest.raises(ActionValidationError, match="arguments"):
        parse_agent_action(
            '{"type": "tool_call", "tool": "list_files", "arguments": []}'
        )


def test_rejects_non_string_final_answer() -> None:
    with pytest.raises(ActionValidationError, match="answer"):
        parse_agent_action('{"type": "final", "answer": 123}')


def test_rejects_non_string_reason() -> None:
    with pytest.raises(ActionValidationError, match="reason"):
        parse_agent_action('{"type": "final", "reason": ["too much"], "answer": "ok"}')


def test_build_ollama_payload_uses_structured_output_schema() -> None:
    messages = [{"role": "user", "content": "hello"}]

    payload = build_ollama_payload("qwen3:14b", messages)

    assert payload["model"] == "qwen3:14b"
    assert payload["messages"] == messages
    assert payload["stream"] is False
    assert payload["format"] == ACTION_RESPONSE_FORMAT


def test_action_response_format_limits_action_types() -> None:
    action_types = set(ACTION_RESPONSE_FORMAT["discriminator"]["mapping"])

    assert action_types == {"tool_call", "final"}


def test_system_prompt_treats_generated_tools_as_review_gated() -> None:
    prompt = build_system_prompt(
        [
            ToolSpec(
                name="create_tool_file",
                description="Create generated tool",
                input_schema={},
            )
        ]
    )

    assert "generated_tools 是實驗區" in prompt
    assert "不能假設 generated_tools 裡的工具已經能在一般任務中直接使用" in prompt
    assert "需要人工 review/promote 後才可正式使用" in prompt


def test_system_prompt_prefers_stock_quote_tool_for_latest_prices() -> None:
    prompt = build_system_prompt(
        [
            ToolSpec(
                name="get_stock_quote",
                description="Get latest stock quote",
                input_schema={},
            )
        ]
    )

    assert "詢問今日或最新股價時，優先使用 get_stock_quote" in prompt
