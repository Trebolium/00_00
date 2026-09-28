import json

from voice_agent import transcript


def test_append_writes_one_json_line_per_call(tmp_path, monkeypatch):
    path = tmp_path / "transcript.jsonl"
    monkeypatch.setattr(transcript, "TRANSCRIPT_PATH", path)

    transcript.append("user", "hello there")
    transcript.append("assistant", "hi!")

    lines = path.read_text().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["role"] == "user"
    assert first["text"] == "hello there"
    assert "ts" in first
