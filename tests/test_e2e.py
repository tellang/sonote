"""회의 실시간 전사 E2E 테스트."""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import pytest
from httpx import ASGITransport
from starlette.requests import Request

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


class _FakeWhisperWorker:
    instances: list["_FakeWhisperWorker"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.transcribe_calls: list[dict] = []
        self.__class__.instances.append(self)

    @property
    def is_ready(self) -> bool:
        return self.started

    def start(self) -> None:
        self.started = True

    def transcribe(self, audio_chunk: np.ndarray, **kwargs) -> list[dict]:
        self.transcribe_calls.append(
            {
                "samples": int(audio_chunk.shape[0]),
                "kwargs": dict(kwargs),
            }
        )
        return [
            {
                "start": 0.0,
                "end": 0.2,
                "text": "첫 번째 발화입니다",
                "avg_logprob": -0.1,
                "no_speech_prob": 0.0,
                "compression_ratio": 1.0,
                "words": [
                    {
                        "start": 0.0,
                        "end": 0.1,
                        "word": "첫",
                        "probability": 0.98,
                    },
                    {
                        "start": 0.1,
                        "end": 0.2,
                        "word": "문장",
                        "probability": 0.97,
                    },
                ],
            },
            {
                "start": 3.5,
                "end": 3.8,
                "text": "두 번째 발화입니다",
                "avg_logprob": -0.1,
                "no_speech_prob": 0.0,
                "compression_ratio": 1.0,
                "words": [
                    {
                        "start": 3.5,
                        "end": 3.65,
                        "word": "두",
                        "probability": 0.96,
                    },
                    {
                        "start": 3.65,
                        "end": 3.8,
                        "word": "문장",
                        "probability": 0.95,
                    },
                ],
            },
        ]

    def stop(self) -> None:
        self.stopped = True


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

    server._api_key = ""
    server._shutdown_requested = False
    server._postprocess_phase = ""
    server._postprocess_progress = 0.0
    server._postprocess_status_file = None
    server._startup_phase = ""
    server._startup_message = ""
    server._startup_ready = True
    _FakeWhisperWorker.instances = []

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


async def _wait_until_async(
    predicate,
    *,
    timeout: float = 5.0,
    interval: float = 0.02,
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("조건이 제한 시간 내에 충족되지 않았습니다.")


async def _read_first_transcript_event(client: httpx.AsyncClient) -> dict:
    del client
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/stream",
        "raw_path": b"/stream",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }

    async def _receive() -> dict:
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    response = await server.stream(Request(scope, _receive))
    assert response.media_type == "text/event-stream"

    try:
        event_text = await asyncio.wait_for(anext(response.body_iterator), timeout=5.0)
    finally:
        await response.body_iterator.aclose()

    payload_line = next(
        line for line in event_text.splitlines() if line.startswith("data: ")
    )
    payload = json.loads(payload_line[6:])
    assert "text" in payload
    return payload


@pytest.mark.anyio
async def test_meeting_pipeline_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_id = "2026-03-13_101500"
    session_dir = tmp_path / "meetings" / "2026-03-13" / "101500"
    output_path = session_dir / "meeting.md"
    audio_chunk = _make_sine_wave()

    capture_ready = threading.Event()
    emit_audio = threading.Event()
    stop_event = threading.Event()

    class _FakeInputStream:
        def __init__(self, *args, callback=None, **kwargs):
            self.callback = callback
            self._thread: threading.Thread | None = None

        def __enter__(self):
            capture_ready.set()
            self._thread = threading.Thread(target=self._emit_once, daemon=True)
            self._thread.start()
            return self

        def _emit_once(self) -> None:
            if not emit_audio.wait(timeout=3.0):
                stop_event.set()
                return
            frames = audio_chunk.reshape(-1, 1)
            self.callback(frames, len(frames), {}, None)
            time.sleep(0.3)
            stop_event.set()

        def __exit__(self, exc_type, exc, tb) -> None:
            if self._thread is not None:
                self._thread.join(timeout=1.0)

    monkeypatch.setattr(server, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr("src.audio_capture.sd.InputStream", _FakeInputStream)

    with patch("src.server.run_server"), patch(
        "src.cli._wait_for_server_ready",
    ), patch(
        "src.server.get_audio_device_switch_event",
        return_value=stop_event,
    ), patch(
        "src.server.consume_audio_device_switch",
        return_value=(False, None),
    ), patch(
        "src.whisper_worker.WhisperWorker",
        new=_FakeWhisperWorker,
    ), patch(
        "src.tray.is_available",
        return_value=False,
    ), patch(
        "webbrowser.open",
    ), patch(
        "os.startfile",
        create=True,
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
                3.0,
            ), "마이크 캡처가 시작되지 않았습니다."

            stream_task = asyncio.create_task(_read_first_transcript_event(client))

            emit_audio.set()
            event_payload = await asyncio.wait_for(stream_task, timeout=5.0)

            assert event_payload["speaker"] == "화자"
            assert "첫 번째 발화입니다" in event_payload["text"]

            await asyncio.to_thread(meeting_thread.join, 5.0)
            assert not meeting_thread.is_alive(), "회의 루프가 종료되지 않았습니다."

            await _wait_until_async(lambda: len(server._transcript_history) == 2)

            assert len(_FakeWhisperWorker.instances) == 1
            worker = _FakeWhisperWorker.instances[0]
            assert worker.started is True
            assert worker.stopped is True
            assert len(worker.transcribe_calls) == 1
            assert worker.transcribe_calls[0]["samples"] == audio_chunk.shape[0]

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
            assert len(alignment_lines) == 2

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

            history_response = await client.get("/history")
            assert history_response.status_code == 200
            history = history_response.json()
            assert len(history) == 2
            assert history[0]["speaker"] == "화자"
            assert "첫 번째 발화입니다" in history[0]["text"]

            rotate_response = await client.post("/api/sessions/new")
            assert rotate_response.status_code == 200
            assert rotate_response.json()["session_id"].count("_") == 1
            assert server.is_session_rotate_requested() is True
            assert server._transcript_history == []
            assert server._segment_count == 0
            assert server._speakers == set()

            delete_response = await client.delete(f"/api/sessions/{session_id}")
            assert delete_response.status_code == 200
            assert delete_response.json() == {
                "deleted": True,
                "session_id": session_id,
            }
            assert not session_dir.exists()
