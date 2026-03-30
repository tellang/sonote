from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from starlette.websockets import WebSocket
    from watchdog.observers import Observer


class ServerState:
    """`server.py`의 런타임 mutable 상태를 캡슐화한다."""

    def __init__(
        self,
        *,
        max_transcript_history: int,
        api_key: str | None = None,
    ) -> None:
        # SSE/WebSocket
        self.client_queues: set[asyncio.Queue[dict]] = set()
        self.connected_websockets: set[WebSocket] = set()
        self.transcript_history: deque[dict] = deque(maxlen=max_transcript_history)

        # 세션/전사
        self.start_time: float = 0.0
        self.segment_count: int = 0
        self.speakers: set[str] = set()
        self.event_loop: asyncio.AbstractEventLoop | None = None
        self.paused: bool = True
        self.shutdown_requested: bool = False

        # 키워드
        self.manual_keywords: set[str] = set()
        self.extracted_keywords: set[str] = set()
        self.promoted_keywords: set[str] = set()
        self.blocked_keywords: set[str] = set()
        self.keyword_seen_counts: dict[str, int] = {}
        self.kw_lock = threading.Lock()

        # 모델/프로필
        self.diarizer: Any = None
        self.profiles_path: str | None = None
        self.profiles_lock = asyncio.Lock()

        # API 키
        key = (api_key if api_key is not None else (os.getenv("MEETING_API_KEY") or "")).strip()
        self.api_key: str = key

        # 오디오 디바이스 전환
        self.audio_device_lock = threading.Lock()
        self.current_audio_device: int | None = None
        self.requested_audio_device: int | None = None
        self.audio_device_switching: bool = False
        self.audio_device_error: str = ""
        self.audio_device_switch_event = threading.Event()
        self.audio_device_switch_ts: float = 0.0

        # 상태 표시
        self.startup_phase: str = ""
        self.startup_message: str = ""
        self.startup_ready: bool = False
        self.capture_error: str = ""
        self.capture_error_count: int = 0
        self.voice_active: bool = False
        self.voice_active_ts: float = 0.0
        self.postprocess_phase: str = ""
        self.postprocess_progress: float = 0.0
        self.postprocess_status_file: str | None = None

        # 세션 회전
        self.session_rotate_event = threading.Event()
        self.session_rotate_callback: Callable[[], None] | None = None
        self.session_observer: Observer | None = None

    def set_startup_status(self, phase: str, message: str = "", ready: bool = False) -> None:
        self.startup_phase = phase
        self.startup_message = message
        self.startup_ready = ready

    def set_capture_error(self, error: str, count: int = 0) -> None:
        self.capture_error = error
        self.capture_error_count = count

    def set_voice_active(self, active: bool) -> None:
        self.voice_active = active
        if active:
            self.voice_active_ts = time.time()

    def set_current_audio_device(self, device: int | None, error: str = "") -> None:
        with self.audio_device_lock:
            self.current_audio_device = device
            self.requested_audio_device = None
            self.audio_device_switching = False
            self.audio_device_error = error
            self.audio_device_switch_event.clear()

    def request_audio_device_switch(self, device: int | None) -> bool:
        with self.audio_device_lock:
            if not self.audio_device_switching and device == self.current_audio_device:
                self.audio_device_error = ""
                return False
            self.requested_audio_device = device
            self.audio_device_switching = True
            self.audio_device_error = ""
            self.audio_device_switch_ts = time.time()
            self.audio_device_switch_event.set()
            return True

    def consume_audio_device_switch(self) -> tuple[bool, int | None]:
        with self.audio_device_lock:
            if not self.audio_device_switch_event.is_set():
                return False, None
            return True, self.requested_audio_device

    def get_audio_device_switch_event(self) -> threading.Event:
        return self.audio_device_switch_event

    def is_session_rotate_requested(self) -> bool:
        return self.session_rotate_event.is_set()

    def consume_session_rotate(self) -> bool:
        if self.session_rotate_event.is_set():
            self.session_rotate_event.clear()
            return True
        return False

    def set_session_rotate_callback(self, callback: Callable[[], None]) -> None:
        self.session_rotate_callback = callback

    def request_shutdown(self) -> dict[str, bool | str]:
        self.shutdown_requested = True
        return {"shutdown": True, "message": "저장 후 종료합니다..."}

    def set_diarizer(self, diarizer: Any, profiles_path: str | None = None) -> None:
        self.diarizer = diarizer
        self.profiles_path = profiles_path

    def set_postprocess_status(self, phase: str, progress: float = 0.0) -> None:
        self.postprocess_phase = phase
        self.postprocess_progress = progress

    def set_postprocess_status_file(self, path: str | None) -> None:
        self.postprocess_status_file = path

    def reset_for_new_session(self) -> None:
        self.transcript_history.clear()
        self.segment_count = 0
        self.speakers.clear()
        self.start_time = time.time()
        # 키워드 상태 초기화 (세션 간 오염 방지)
        with self.kw_lock:
            self.manual_keywords.clear()
            self.extracted_keywords.clear()
            self.promoted_keywords.clear()
            self.blocked_keywords.clear()
            self.keyword_seen_counts.clear()
