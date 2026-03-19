from __future__ import annotations

import argparse
import sys
from types import ModuleType
from unittest import mock

from src import cli, continuous


def test_build_meeting_markdown_renders_required_sections() -> None:
    content = continuous._build_meeting_markdown(
        [
            {"start": 0.0, "end": 1.8, "text": "첫 번째 문장"},
            {"start": 65.0, "end": 70.0, "text": "두 번째 문장"},
        ],
        started_at=0.0,
    )

    assert "# 회의 내용 요약" in content
    assert "# To-do" in content
    assert "# 대화 정리" in content
    assert "# Raw Data" in content
    assert "### Unknown · 00:00:00 ~ 00:00:01" in content
    assert "### Unknown · 00:01:05 ~ 00:01:10" in content
    assert "- [00:00:00 ~ 00:00:01] [Unknown] 첫 번째 문장" in content
    assert "- [00:01:05 ~ 00:01:10] [Unknown] 두 번째 문장" in content


def test_polish_continuous_transcript_creates_meeting_and_polished_files(tmp_path) -> None:
    output_path = tmp_path / "transcript.txt"
    segments = [
        {"start": 0.0, "end": 1.0, "text": "안녕하세여."},
        {"start": 2.0, "end": 4.0, "text": "회의를 시작합니디."},
    ]

    def _fake_polish(
        md_path,
        timeout=None,
        segment_count=0,
        progress_callback=None,
        use_ollama=False,
        ollama_model=None,
    ):
        _ = timeout, progress_callback
        assert md_path.name == "transcript_polished.md"
        assert segment_count == 2
        assert use_ollama is True
        assert ollama_model == "qwen3.5:9b"

        content = md_path.read_text(encoding="utf-8")
        assert "- [00:00:00 ~ 00:00:01] [Unknown] 안녕하세여." in content
        md_path.write_text(
            content.replace("안녕하세여.", "안녕하세요."),
            encoding="utf-8",
        )
        return {"corrected": True, "summarized": False}

    with mock.patch("src.polish.polish_meeting", side_effect=_fake_polish):
        meeting_path, polished_path, result = continuous._polish_continuous_transcript(
            output_path,
            segments,
            started_at=0.0,
            use_ollama=True,
            ollama_model="qwen3.5:9b",
        )

    assert meeting_path.name == "transcript_meeting.md"
    assert polished_path.name == "transcript_polished.md"
    assert meeting_path.exists()
    assert polished_path.exists()
    assert "- [00:00:00 ~ 00:00:01] [Unknown] 안녕하세여." in meeting_path.read_text(encoding="utf-8")
    assert "안녕하세요." in polished_path.read_text(encoding="utf-8")
    assert result == {"corrected": True, "summarized": False}


def test_cmd_live_continuous_passes_polish_options(tmp_path) -> None:
    fake_transcribe = ModuleType("src.transcribe")
    fake_transcribe.save_transcript = lambda *args, **kwargs: None
    fake_transcribe.transcribe_audio = lambda *args, **kwargs: []

    fake_continuous = ModuleType("src.continuous")
    continuous_live = mock.Mock()
    fake_continuous.continuous_live = continuous_live

    args = argparse.Namespace(
        url="https://youtube.com/watch?v=test",
        output=str(tmp_path),
        cpu=False,
        auto_start=False,
        back=0,
        continuous=True,
        fmt="txt",
        chunk_size=120,
        model="tellang/whisper-large-v3-turbo-ko",
        language="ko",
        cookies=None,
        no_polish=False,
        ollama=True,
        ollama_model="qwen3.5:9b",
    )

    with mock.patch.dict(
        sys.modules,
        {
            "src.transcribe": fake_transcribe,
            "src.continuous": fake_continuous,
        },
    ):
        cli._cmd_live(args)

    continuous_live.assert_called_once_with(
        "https://youtube.com/watch?v=test",
        tmp_path / "transcript.txt",
        chunk_seconds=120,
        model_id="tellang/whisper-large-v3-turbo-ko",
        language="ko",
        device=None,
        compute_type=None,
        fmt="txt",
        cookies_path=None,
        use_polish=True,
        use_ollama=True,
        ollama_model="qwen3.5:9b",
    )
