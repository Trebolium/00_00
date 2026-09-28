from unittest.mock import MagicMock

from voice_agent import llm


def test_respond_sends_history_and_returns_stripped_reply(monkeypatch):
    fake_completion = MagicMock()
    fake_completion.choices[0].message.content = "  sure, happy to help  "
    fake_create = MagicMock(return_value=fake_completion)
    monkeypatch.setattr(llm._client.chat.completions, "create", fake_create)

    history = [{"role": "user", "content": "hi"}]
    result = llm.respond(history)

    assert result == "sure, happy to help"
    fake_create.assert_called_once_with(model=llm.LLM_MODEL, messages=history)
