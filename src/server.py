from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType

from fastapi import FastAPI

from . import _server_impl as _impl
from .web.routes import (
    core as core_routes,
    devices as devices_routes,
    keywords as keywords_routes,
    profiles as profiles_routes,
    sessions as sessions_routes,
    stream as stream_routes,
)
from .web.state import ServerState

MAX_WS_CONNECTIONS = _impl.MAX_WS_CONNECTIONS
MAX_WS_MESSAGE_SIZE = _impl.MAX_WS_MESSAGE_SIZE
MAX_TRANSCRIPT_HISTORY = _impl.MAX_TRANSCRIPT_HISTORY

_state = ServerState(max_transcript_history=MAX_TRANSCRIPT_HISTORY)
_server_state = _state

# 내부 구현 모듈이 참조하는 상태 컨테이너를 facade 상태로 통일한다.
_impl._server_state = _state

_STATE_COMPAT_ATTRS: dict[str, str] = dict(_impl._STATE_COMPAT_ATTRS)
_MODULE_COMPAT_ATTRS: dict[str, str] = {
    "OUTPUT_ROOT": "OUTPUT_ROOT",
    "static_dir": "static_dir",
}


def _build_app() -> FastAPI:
    application = FastAPI(title="회의 실시간 전사", lifespan=_impl.lifespan)
    application.include_router(core_routes.router)
    application.include_router(stream_routes.router)
    application.include_router(keywords_routes.router)
    application.include_router(devices_routes.router)
    application.include_router(sessions_routes.router)
    application.include_router(profiles_routes.router)
    return application


app = _build_app()


def create_app(*, beta_mode: bool | None = None) -> FastAPI:
    """데스크톱/CLI 런처가 재사용할 FastAPI 앱 인스턴스를 반환한다."""
    _impl._apply_beta_mode(beta_mode)
    return app


def is_session_rotate_requested() -> bool:
    """세션 회전 요청 상태 반환."""
    return _state.is_session_rotate_requested()


def consume_session_rotate() -> bool:
    """세션 회전 요청을 소비 (1회성). 요청이 있었으면 True."""
    return _state.consume_session_rotate()


def set_session_rotate_callback(callback: Callable[[], None]) -> None:
    """CLI에서 세션 회전 시 호출할 콜백 등록."""
    _state.set_session_rotate_callback(callback)


def set_current_audio_device(device: int | None, error: str = "") -> None:
    _state.set_current_audio_device(device, error=error)


def request_audio_device_switch(device: int | None) -> bool:
    return _state.request_audio_device_switch(device)


def consume_audio_device_switch() -> tuple[bool, int | None]:
    return _state.consume_audio_device_switch()


def get_audio_device_switch_event():
    return _state.get_audio_device_switch_event()


def set_startup_status(phase: str, message: str = "", ready: bool = False) -> None:
    _state.set_startup_status(phase, message=message, ready=ready)


def set_capture_error(error: str, count: int = 0) -> None:
    """캡처 루프 에러를 상태 API에 노출한다."""
    _state.set_capture_error(error, count=count)


def set_voice_active(active: bool) -> None:
    """음성 감지 상태를 갱신한다."""
    _state.set_voice_active(active)


def is_paused() -> bool:
    """현재 일시정지 상태인지 반환 (CLI 루프에서 참조)."""
    return _state.paused


def toggle_pause_state() -> dict[str, bool | str]:
    """현재 녹음 일시정지 상태를 토글하고 최신 상태를 반환한다."""
    _state.paused = not _state.paused
    state = "paused" if _state.paused else "recording"
    return {"paused": _state.paused, "state": state}


def get_keywords() -> set[str]:
    """현재 initial_prompt에 반영할 활성 키워드 집합 반환."""
    return _state.manual_keywords.union(_state.promoted_keywords)


def is_shutdown_requested() -> bool:
    """종료 요청 상태 반환 (CLI 루프에서 참조)."""
    return _state.shutdown_requested


def request_shutdown() -> dict[str, bool | str]:
    """저장 후 종료 플래그를 설정하고 응답 페이로드를 반환한다."""
    return _state.request_shutdown()


def signal_server_shutdown() -> bool:
    """실행 중인 uvicorn 서버에 graceful 종료 신호를 보낸다."""
    return _impl.signal_server_shutdown()


def set_diarizer(diarizer, profiles_path: str | None = None) -> None:
    """화자 분리기 참조 설정 (CLI에서 호출)."""
    _state.set_diarizer(diarizer, profiles_path=profiles_path)


def set_postprocess_status(phase: str, progress: float = 0.0) -> None:
    """후처리 진행 상태 설정 (웹 UI 실시간 표시용)."""
    _state.set_postprocess_status(phase, progress=progress)


def set_postprocess_status_file(path: str | None) -> None:
    """파일 기반 후처리 상태 경로 설정 (별도 프로세스용)."""
    _state.set_postprocess_status_file(path)


# 라우트 핸들러 re-export (기존 import 경로 호환)
index = core_routes.index
settings_page = core_routes.settings_page
get_settings = core_routes.get_settings
save_settings = core_routes.save_settings
history = core_routes.history
status = core_routes.status
toggle_pause = core_routes.toggle_pause
shutdown = core_routes.shutdown

stream = stream_routes.stream
ws_transcribe = stream_routes.ws_transcribe
push_segment_endpoint = stream_routes.push_segment
load_transcript_file = stream_routes.load_transcript

add_keyword = keywords_routes.add_keyword
remove_keyword = keywords_routes.remove_keyword
list_keywords = keywords_routes.list_keywords

list_devices = devices_routes.list_devices
switch_device = devices_routes.switch_device
list_audio_devices = devices_routes.list_audio_devices

list_sessions = sessions_routes.list_sessions
get_session = sessions_routes.get_session
delete_session = sessions_routes.delete_session
search_session = sessions_routes.search_session
export_session_endpoint = sessions_routes.export_session
new_session = sessions_routes.new_session

speaker_profile_page = profiles_routes.speaker_profile_page
list_speakers = profiles_routes.list_speakers
enroll_speaker = profiles_routes.enroll_speaker
get_profiles = profiles_routes.get_profiles
create_profile = profiles_routes.create_profile
update_profile = profiles_routes.update_profile
delete_profile = profiles_routes.delete_profile
list_unknown_speakers = profiles_routes.list_unknown_speakers
auto_register_speaker = profiles_routes.auto_register_speaker
delete_unknown_speaker = profiles_routes.delete_unknown_speaker


# 서비스/유틸리티 re-export
UnknownSpeakerTracker = _impl.UnknownSpeakerTracker
_unknown_tracker = _impl._unknown_tracker

add_extracted_keywords = _impl.add_extracted_keywords
push_transcript = _impl.push_transcript
push_transcript_sync = _impl.push_transcript_sync
push_correction = _impl.push_correction
push_correction_sync = _impl.push_correction_sync


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    application: FastAPI | None = None,
    beta_mode: bool | None = None,
) -> None:
    """uvicorn으로 서버 실행. threading에서 호출할 수 있도록 uvicorn.run() 사용."""
    _impl.run_server(
        host=host,
        port=port,
        application=application or create_app(beta_mode=beta_mode),
        beta_mode=beta_mode,
    )


class _ServerModule(ModuleType):
    def __getattr__(self, name: str):
        state_attr = _STATE_COMPAT_ATTRS.get(name)
        if state_attr is not None:
            return getattr(_state, state_attr)

        module_attr = _MODULE_COMPAT_ATTRS.get(name)
        if module_attr is not None:
            return getattr(_impl, module_attr)

        if hasattr(_impl, name):
            return getattr(_impl, name)

        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        state_attr = _STATE_COMPAT_ATTRS.get(name)
        if state_attr is not None:
            setattr(_state, state_attr, value)
            return

        module_attr = _MODULE_COMPAT_ATTRS.get(name)
        if module_attr is not None:
            setattr(_impl, module_attr, value)
            return

        if hasattr(_impl, name):
            setattr(_impl, name, value)
            return

        super().__setattr__(name, value)


def _install_state_compat_proxy() -> None:
    module = sys.modules.get(__name__)
    if module is None or isinstance(module, _ServerModule):
        return
    module.__class__ = _ServerModule


_install_state_compat_proxy()


__all__ = [
    "app",
    "create_app",
    "index",
    "settings_page",
    "get_settings",
    "save_settings",
    "stream",
    "ws_transcribe",
    "history",
    "status",
    "toggle_pause",
    "shutdown",
    "list_keywords",
    "add_keyword",
    "remove_keyword",
    "list_devices",
    "switch_device",
    "list_audio_devices",
    "list_sessions",
    "get_session",
    "delete_session",
    "search_session",
    "export_session_endpoint",
    "new_session",
    "speaker_profile_page",
    "list_speakers",
    "enroll_speaker",
    "get_profiles",
    "create_profile",
    "update_profile",
    "delete_profile",
    "list_unknown_speakers",
    "auto_register_speaker",
    "delete_unknown_speaker",
    "push_segment_endpoint",
    "load_transcript_file",
    "push_transcript",
    "push_transcript_sync",
    "push_correction",
    "push_correction_sync",
    "run_server",
    "_client_queues",
    "_connected_websockets",
    "is_paused",
    "toggle_pause_state",
    "get_keywords",
    "add_extracted_keywords",
    "request_audio_device_switch",
    "consume_audio_device_switch",
    "get_audio_device_switch_event",
    "set_current_audio_device",
    "set_startup_status",
    "set_capture_error",
    "set_voice_active",
    "is_shutdown_requested",
    "request_shutdown",
    "signal_server_shutdown",
    "set_diarizer",
    "set_postprocess_status",
    "set_postprocess_status_file",
    "is_session_rotate_requested",
    "consume_session_rotate",
    "set_session_rotate_callback",
    "UnknownSpeakerTracker",
    "_unknown_tracker",
]
