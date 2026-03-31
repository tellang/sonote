"""회의 실시간 전사 E2E 테스트."""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
from httpx import ASGITransport

from src import cli, server


def _build_args(output_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        list_devices=False,
        device=0,
        port=8765,
        output=output_path,
        cpu=True,
        model="large-v3-turbo",
        language="ko",
        no_diarize=True,
        profiles=None,
        chunk=0.5,
        prompt="",
        fixed_chunking=False,
        no_polish=True,
        ollama=False,
        ollama_model=None,
        auto_update=False,
    )


def _make_sine_wave(
    *,
    seconds: float = 0.5,
    sample_rate: int = 16000,
    frequency: float = 440.0,
) -> np.ndarray:
    sample_count = int(sample_rate * seconds)
    timeline = np.arange(sample_count, dtype=np.float32) / float(sample_rate)
    return (0.2 * np.sin(2.0 * np.pi * frequency * timeline)).astype(np.float32)


# -- Fake faster_whisper segment objects --

_FAKE_SEGMENTS = [
    SimpleNamespace(
        start=0.0,
        end=0.2,
        text="첫 번째 발화입니다",
        avg_logprob=-0.1,
        no_speech_prob=0.0,
        compression_ratio=1.0,
        words=[
            SimpleNamespace(start=0.0, end=0.1, word="첫", probability=0.98),
            SimpleNamespace(start=0.1, end=0.2, word="문장", probability=0.97),
        ],
    ),
    SimpleNamespace(
        start=3.5,
        end=3.8,
        text="두 번째 발화입니다",
        avg_logprob=-0.1,
        no_speech_prob=0.0,
        compression_ratio=1.0,
        words=[
            SimpleNamespace(start=3.5, end=3.65, word="두", probability=0.96),
            SimpleNamespace(start=3.65, end=3.8, word="문장", probability=0.95),
        ],
    ),
]


class _FakeWhisperModel:
    """faster_whisper.WhisperModel drop-in replacement for testing."""

    instances: list["_FakeWhisperModel"] = []
    transcribe_calls: list[dict] = []

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.__class__.instances.append(self)

    def transcribe(self, audio, **kwargs):
        self.__class__.transcribe_calls.append(
            {
                "samples": int(audio.shape[0]),
                "kwargs": dict(kwargs),
            }
        )
        info = SimpleNamespace(language="ko", language_probability=0.99)
        return iter(list(_FAKE_SEGMENTS)), info


@pytest.fixture(autouse=True)
def _reset_e2e_state():
    original_api_key = server._api_key
    original_shutdown = server._shutdown_requested
    original_phase = server._postprocess_phase
    original_progress = server._postprocess_progress
    original_status_file = server._postprocess_status_file
    original_startup_phase = server._startup_phase
    original_startup_message = server._startup_message
    original_startup_ready = server._startup_ready
    original_event_loop = server._event_loop
    original_paused = server._paused

    server._api_key = ""
    server._shutdown_requested = False
    server._postprocess_phase = ""
    server._postprocess_progress = 0.0
    server._postprocess_status_file = None
    server._startup_phase = ""
    server._startup_message = ""
    server._startup_ready = True
    server._paused = False
    _FakeWhisperModel.instances = []
    _FakeWhisperModel.transcribe_calls = []

    yield

    server._api_key = original_api_key
    server._shutdown_requested = original_shutdown
    server._postprocess_phase = original_phase
    server._postprocess_progress = original_progress
    server._postprocess_status_file = original_status_file
    server._startup_phase = original_startup_phase
    server._startup_message = original_startup_message
    server._startup_ready = original_startup_ready
    server._event_loop = original_event_loop
    server._paused = original_paused


async def _wait_until_async(
    predicate,
    *,
    timeout: float = 10.0,
    interval: float = 0.05,
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("조건이 제한 시간 내에 충족되지 않았습니다.")


@pytest.mark.anyio
async def test_meeting_pipeline_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_id = "2026-03-13_101500"
    session_dir = tmp_path / "meetings" / "2026-03-13" / "101500"
    output_path = session_dir / "meeting.md"
    audio_chunk = _make_sine_wave()

    capture_ready = threading.Event()
    emit_audio = threading.Event()

    def _fake_capture_audio(**kwargs):
        """Replacement for src.audio_capture.capture_audio.

        Yields exactly one chunk of audio, then waits for shutdown before
        returning so the pipeline can perform its normal exit path.
        """
        capture_ready.set()
        if not emit_audio.wait(timeout=10.0):
            return
        yield audio_chunk
        # Hold the generator open until the pipeline is signalled to stop.
        # The pipeline checks _should_stop() after each yielded chunk; when
        # shutdown is requested the for-loop simply advances to the next
        # iteration which lands here.  We wait briefly then return so that
        # capture_audio (the generator) is exhausted and the pipeline's
        # outer while-loop re-checks _should_stop() and exits.
        stop_event = kwargs.get("stop_event")
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if server.is_shutdown_requested():
                return
            if stop_event is not None and stop_event.is_set():
                return
            time.sleep(0.1)

    monkeypatch.setattr(server, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr("src.audio_capture.capture_audio", _fake_capture_audio)

    # Create a fake faster_whisper module so `from faster_whisper import WhisperModel`
    # inside _cmd_meeting resolves to our fake.
    fake_faster_whisper = MagicMock()
    fake_faster_whisper.WhisperModel = _FakeWhisperModel

    with patch("src.server.run_server"), patch(
        "src.cli._wait_for_server_ready",
    ), patch(
        "src.tray.is_available",
        return_value=False,
    ), patch(
        "webbrowser.open",
    ), patch(
        "os.startfile",
        create=True,
    ), patch(
        "src.runtime_env.bootstrap_nvidia_dll_path",
    ), patch.dict(
        "sys.modules",
        {"faster_whisper": fake_faster_whisper},
    ):
        server._event_loop = asyncio.get_running_loop()
        transport = ASGITransport(app=server.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            meeting_thread = threading.Thread(
                target=cli._cmd_meeting,
                args=(_build_args(output_path),),
                daemon=True,
            )
            meeting_thread.start()

            assert await asyncio.to_thread(
                capture_ready.wait,
                10.0,
            ), "마이크 캡처가 시작되지 않았습니다."

            emit_audio.set()

            # Wait for both transcript segments to arrive via push_transcript_sync
            await _wait_until_async(
                lambda: len(server._transcript_history) >= 2,
                timeout=10.0,
            )

            # Signal the meeting loop to shut down gracefully
            server._shutdown_requested = True

            # Wait for the meeting thread to finish
            await asyncio.to_thread(meeting_thread.join, 10.0)
            assert not meeting_thread.is_alive(), "회의 루프가 종료되지 않았습니다."

            # Verify fake model was used
            assert len(_FakeWhisperModel.instances) == 1
            assert len(_FakeWhisperModel.transcribe_calls) >= 1
            assert _FakeWhisperModel.transcribe_calls[0]["samples"] == audio_chunk.shape[0]

            # Verify transcript history
            history = list(server._transcript_history)
            assert len(history) == 2
            assert history[0]["speaker"] == "화자"
            assert "첫 번째 발화입니다" in history[0]["text"]

            # Verify output files
            session_json = session_dir / "session.json"
            alignment_path = session_dir / "meeting.stt.jsonl"
            audio_path = session_dir / "meeting.audio.wav"

            assert output_path.exists()
            assert session_json.exists()
            assert alignment_path.exists()
            assert audio_path.exists()
            assert audio_path.stat().st_size > 44

            meeting_text = output_path.read_text(encoding="utf-8")
            assert "첫 번째 발화입니다" in meeting_text
            assert "두 번째 발화입니다" in meeting_text

            session_meta = json.loads(session_json.read_text(encoding="utf-8"))
            assert session_meta["segment_count"] == 2
            assert session_meta["speaker_count"] == 1
            assert Path(session_meta["output_path"]) == output_path

            alignment_lines = alignment_path.read_text(encoding="utf-8").splitlines()
            # 2 raw stt_segment entries + 2 processed segment entries = 4
            assert len(alignment_lines) == 4

            # API: session list
            list_response = await client.get("/api/sessions")
            assert list_response.status_code == 200
            sessions = list_response.json()
            assert sessions == [
                {
                    "id": session_id,
                    "date": "2026-03-13",
                    "time": "101500",
                    "duration": session_meta["duration"],
                    "segments": 2,
                    "speakers": 1,
                    "path": str(session_dir),
                }
            ]

            # API: session detail
            detail_response = await client.get(f"/api/sessions/{session_id}")
            assert detail_response.status_code == 200
            detail = detail_response.json()
            assert detail["id"] == session_id
            assert detail["segments"] == 2
            assert detail["speakers"] == 1
            assert detail["transcript_source"] == "meeting.md"
            assert any("첫 번째 발화입니다" in line for line in detail["transcript"])
            assert any("두 번째 발화입니다" in line for line in detail["transcript"])
            assert len(detail["alignment"]) == 2
            assert len(detail["display_segments"]) == 2
            assert len(detail["raw_alignment"]) == 2

            # API: history
            history_response = await client.get("/history")
            assert history_response.status_code == 200
            history_data = history_response.json()
            assert len(history_data) == 2
            assert history_data[0]["speaker"] == "화자"
            assert "첫 번째 발화입니다" in history_data[0]["text"]

            # API: session rotate
            rotate_response = await client.post("/api/sessions/new")
            assert rotate_response.status_code == 200
            assert rotate_response.json()["session_id"].count("_") == 1
            assert server.is_session_rotate_requested() is True
            assert len(server._transcript_history) == 0
            assert server._segment_count == 0
            assert server._speakers == set()

            # API: session delete
            delete_response = await client.delete(f"/api/sessions/{session_id}")
            assert delete_response.status_code == 200
            assert delete_response.json() == {
                "deleted": True,
                "session_id": session_id,
            }
            assert not session_dir.exists()
