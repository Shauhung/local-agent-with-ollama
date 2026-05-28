import pytest

from agent_history import ConversationHistory, validate_session_id


def test_conversation_history_round_trips_recent_messages(tmp_path) -> None:
    history = ConversationHistory(tmp_path / "history.sqlite3")

    history.append_message("phone", "user", "hello")
    history.append_message("phone", "assistant", "hi")
    history.append_message("phone", "user", "second")

    assert history.load_recent_messages("phone", 2) == [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "second"},
    ]


def test_conversation_history_is_scoped_by_session(tmp_path) -> None:
    history = ConversationHistory(tmp_path / "history.sqlite3")

    history.append_message("phone", "user", "hello")
    history.append_message("desktop", "user", "different")

    assert history.load_recent_messages("phone", 10) == [
        {"role": "user", "content": "hello"}
    ]


def test_conversation_history_rejects_invalid_role(tmp_path) -> None:
    history = ConversationHistory(tmp_path / "history.sqlite3")

    with pytest.raises(ValueError, match="Invalid message role"):
        history.append_message("phone", "system", "nope")


def test_validate_session_id_rejects_empty_value() -> None:
    with pytest.raises(ValueError, match="session_id"):
        validate_session_id(" ")
