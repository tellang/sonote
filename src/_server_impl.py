from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import ModuleType

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .paths import OUTPUT_ROOT, static_dir
from .web.state import ServerState

_BETA_ENV_KEY = "SONOTE_BETA"

# --- 보안 하드닝 상수 ---
MAX_WS_CONNECTIONS = 50  # WebSocket 최대 동시 연결 수
MAX_WS_MESSAGE_SIZE = 65536  # WebSocket 메시지 최대 크기 (바이트)
MAX_TRANSCRIPT_HISTORY = 10000  # 전사 내역 최대 세그먼트 수

# 서버 런타임 mutable 상태 컨테이너
_server_state = ServerState(max_transcript_history=MAX_TRANSCRIPT_HISTORY)
_uvicorn_server = None
_uvicorn_server_lock = threading.Lock()

_STATE_COMPAT_ATTRS: dict[str, str] = {
    "_client_queues": "client_queues",
    "_connected_websockets": "connected_websockets",
    "_transcript_history": "transcript_history",
    "_start_time": "start_time",
    "_segment_count": "segment_count",
    "_speakers": "speakers",
    "_event_loop": "event_loop",
    "_paused": "paused",
    "_manual_keywords": "manual_keywords",
    "_extracted_keywords": "extracted_keywords",
    "_promoted_keywords": "promoted_keywords",
    "_blocked_keywords": "blocked_keywords",
    "_keyword_seen_counts": "keyword_seen_counts",
    "_kw_lock": "kw_lock",
    "_shutdown_requested": "shutdown_requested",
    "_postprocess_phase": "postprocess_phase",
    "_postprocess_progress": "postprocess_progress",
    "_diarizer": "diarizer",
    "_profiles_path": "profiles_path",
    "_profiles_lock": "profiles_lock",
    "_api_key": "api_key",
    "_audio_device_lock": "audio_device_lock",
    "_current_audio_device": "current_audio_device",
    "_requested_audio_device": "requested_audio_device",
    "_audio_device_switching": "audio_device_switching",
    "_audio_device_error": "audio_device_error",
    "_audio_device_switch_event": "audio_device_switch_event",
    "_audio_device_switch_ts": "audio_device_switch_ts",
    "_startup_phase": "startup_phase",
    "_startup_message": "startup_message",
    "_startup_ready": "startup_ready",
    "_capture_error": "capture_error",
    "_capture_error_count": "capture_error_count",
    "_voice_active": "voice_active",
    "_voice_active_ts": "voice_active_ts",
    "_postprocess_status_file": "postprocess_status_file",
    "_session_rotate_event": "session_rotate_event",
    "_session_rotate_callback": "session_rotate_callback",
    "_session_observer": "session_observer",
}

_SESSION_ID_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{6}")
_SESSION_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SESSION_TIME_RE = re.compile(r"\d{6}")
_RAW_DATA_HEADING_RE = re.compile(r"^#{1,6}\s*Raw Data\s*$", re.IGNORECASE)

# 자동 등록 후보 판별에 필요한 최소 세그먼트 수
_AUTO_REGISTER_MIN_SEGMENTS = 5


class UnknownSpeakerTracker:
    """미등록 화자 추적기 — 프로필에 없는 화자의 임베딩을 누적하고 자동 등록 후보를 관리한다.

    임베딩이 기존 프로필 모두와 유사도 임계값 미만이면 미등록 화자로 판별하고,
    같은 화자의 세그먼트 임베딩을 평균하여 임시 저장한다.
    _AUTO_REGISTER_MIN_SEGMENTS 이상 누적되면 자동 등록 후보로 마킹한다.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # speaker_id → {"embeddings": list[list[float]], "mean": list[float],
        #               "first_seen": str, "last_seen": str, "candidate": bool}
        self._unknown: dict[str, dict] = {}
        self._next_id: int = 1

    def track(self, embedding: list[float], threshold: float = 0.70) -> str | None:
        """임베딩을 기존 프로필과 비교하고, 미등록이면 추적에 추가한다.

        Args:
            embedding: 화자 임베딩 벡터
            threshold: 프로필 매칭 임계값

        Returns:
            미등록 화자 ID (기존 프로필과 매칭되면 None)
        """
        if not embedding or _server_state.diarizer is None:
            return None

        import numpy as np

        # 기존 프로필과 매칭 시도
        matched, _ = _find_matching_profile(embedding, threshold=threshold)
        if matched is not None:
            return None

        target = np.array(embedding, dtype=np.float32)
        now = datetime.now().isoformat()

        with self._lock:
            # 기존 미등록 화자 중 가장 유사한 것과 매칭
            best_id: str | None = None
            best_sim: float = 0.0
            for uid, info in self._unknown.items():
                known = np.array(info["mean"], dtype=np.float32)
                sim = float(_server_state.diarizer._cosine_similarity(target, known))
                if sim > best_sim:
                    best_sim = sim
                    best_id = uid

            if best_id is not None and best_sim >= threshold:
                # 기존 미등록 화자에 임베딩 누적
                info = self._unknown[best_id]
                info["embeddings"].append(embedding)
                # 평균 재계산
                arr = np.array(info["embeddings"], dtype=np.float32)
                info["mean"] = np.mean(arr, axis=0).tolist()
                info["last_seen"] = now
                if len(info["embeddings"]) >= _AUTO_REGISTER_MIN_SEGMENTS:
                    info["candidate"] = True
                return best_id

            # 새 미등록 화자 생성
            new_id = f"unknown_{self._next_id}"
            self._next_id += 1
            self._unknown[new_id] = {
                "embeddings": [embedding],
                "mean": embedding,
                "first_seen": now,
                "last_seen": now,
                "candidate": False,
            }
            return new_id

    def list_unknown(self) -> list[dict]:
        """미등록 화자 목록 반환."""
        with self._lock:
            result = []
            for uid, info in self._unknown.items():
                result.append({
                    "id": uid,
                    "segment_count": len(info["embeddings"]),
                    "first_seen": info["first_seen"],
                    "last_seen": info["last_seen"],
                    "candidate": info["candidate"],
                })
            return result

    def get(self, speaker_id: str) -> dict | None:
        """특정 미등록 화자 정보 반환."""
        with self._lock:
            return self._unknown.get(speaker_id)

    def remove(self, speaker_id: str) -> bool:
        """미등록 화자 삭제. 성공 시 True."""
        with self._lock:
            return self._unknown.pop(speaker_id, None) is not None

    def reset(self) -> None:
        """모든 미등록 화자 데이터 초기화."""
        with self._lock:
            self._unknown.clear()
            self._next_id = 1


# 모듈 레벨 미등록 화자 추적기 인스턴스
_unknown_tracker = UnknownSpeakerTracker()


async def _broadcast_item(item: dict) -> None:
    """모든 연결된 SSE + WebSocket 클라이언트에 이벤트 아이템을 전송한다."""
    # SSE 클라이언트
    for q in list(_server_state.client_queues):
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            pass

    # WebSocket 클라이언트
    if _server_state.connected_websockets:
        # SSE 이벤트 타입이 있으면 type 필드 추가, 없으면 transcript 타입
        if "_type" in item:
            ws_payload = {"type": item["_type"], **item.get("_payload", {})}
        else:
            ws_payload = {"type": "transcript", **item}
        message = json.dumps(ws_payload, ensure_ascii=False)
        dead: list[WebSocket] = []
        for ws in list(_server_state.connected_websockets):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _server_state.connected_websockets.discard(ws)


async def _broadcast_sse(event_type: str, payload: dict) -> None:
    await _broadcast_item({"_type": event_type, "_payload": payload})


def _broadcast_sse_sync(event_type: str, payload: dict) -> None:
    """watchdog 스레드에서 SSE 브로드캐스트를 예약한다."""
    if _server_state.event_loop is not None and _server_state.event_loop.is_running():
        asyncio.run_coroutine_threadsafe(
            _broadcast_sse(event_type, payload),
            _server_state.event_loop,
        )


def _session_payload(
    session_id: str,
    session_dir: Path,
    *,
    changed_file: Path | None = None,
) -> dict:
    date_str, time_str = session_id.split("_", 1)
    payload = {
        "session_id": session_id,
        "date": date_str,
        "time": time_str,
        "path": str(session_dir),
    }
    if changed_file is not None:
        payload["file"] = str(changed_file)
    return payload


def _session_id_from_parts(parts: tuple[str, ...]) -> str | None:
    if len(parts) < 2:
        return None
    date_str, time_str = parts[0], parts[1]
    if not _SESSION_DATE_RE.fullmatch(date_str):
        return None
    if not _SESSION_TIME_RE.fullmatch(time_str):
        return None
    return f"{date_str}_{time_str}"


def _session_info_from_path(path: str | Path) -> tuple[str, Path, Path | None] | None:
    meetings_root = (OUTPUT_ROOT / "meetings").resolve()
    resolved = Path(path).resolve()
    try:
        relative_parts = resolved.relative_to(meetings_root).parts
    except ValueError:
        return None

    session_id = _session_id_from_parts(relative_parts)
    if session_id is None:
        return None

    session_dir = meetings_root / relative_parts[0] / relative_parts[1]
    changed_file = resolved if len(relative_parts) > 2 else None
    return session_id, session_dir, changed_file


def _resolve_session_dir(session_id: str) -> tuple[Path, Path]:
    """세션 ID를 안전한 실제 경로로 변환한다."""
    if not _SESSION_ID_RE.fullmatch(session_id):
        raise HTTPException(status_code=400, detail="세션 ID 형식이 올바르지 않습니다.")

    date_str, time_str = session_id.split("_", 1)
    session_dir = (OUTPUT_ROOT / "meetings" / date_str / time_str).resolve()
    meetings_root = (OUTPUT_ROOT / "meetings").resolve()
    if not str(session_dir).startswith(str(meetings_root)):
        raise HTTPException(status_code=400, detail="잘못된 세션 경로입니다.")
    return session_dir, meetings_root


class _SessionWatchHandler(FileSystemEventHandler):
    """output/meetings 하위 세션 생성/변경을 SSE로 브로드캐스트한다."""

    def _broadcast_session_created(self, path: str) -> None:
        info = _session_info_from_path(path)
        if info is None:
            return
        session_id, session_dir, changed_file = info
        if changed_file is not None:
            return
        _broadcast_sse_sync("session_created", _session_payload(session_id, session_dir))

    def _broadcast_session_updated(self, path: str) -> None:
        info = _session_info_from_path(path)
        if info is None:
            return
        session_id, session_dir, changed_file = info
        if changed_file is None:
            return
        _broadcast_sse_sync(
            "session_updated",
            _session_payload(session_id, session_dir, changed_file=changed_file),
        )

    def on_created(self, event) -> None:
        if event.is_directory:
            self._broadcast_session_created(event.src_path)
            return
        self._broadcast_session_updated(event.src_path)

    def on_modified(self, event) -> None:
        if event.is_directory:
            return
        self._broadcast_session_updated(event.src_path)

    def on_moved(self, event) -> None:
        if event.is_directory:
            self._broadcast_session_created(event.dest_path)
            return
        self._broadcast_session_updated(event.dest_path)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """이벤트 루프를 캡처하고 watchdog observer를 서버 생명주기에 맞춰 관리한다."""
    _server_state.start_time = time.time()
    _server_state.event_loop = asyncio.get_running_loop()

    meetings_root = OUTPUT_ROOT / "meetings"
    meetings_root.mkdir(parents=True, exist_ok=True)

    observer = Observer()
    observer.schedule(_SessionWatchHandler(), str(meetings_root), recursive=True)
    observer.start()
    _server_state.session_observer = observer

    try:
        yield
    finally:
        observer.stop()
        observer.join(timeout=5.0)
        _server_state.session_observer = None


app = FastAPI(title="회의 실시간 전사", lifespan=lifespan)


def _apply_beta_mode(beta_mode: bool | None = None) -> None:
    enabled = os.getenv(_BETA_ENV_KEY) == "1" if beta_mode is None else bool(beta_mode)
    if enabled:
        os.environ[_BETA_ENV_KEY] = "1"
    try:
        from .postprocess import set_beta_mode

        set_beta_mode(enabled)
    except Exception:
        pass


def create_app(*, beta_mode: bool | None = None) -> FastAPI:
    """데스크톱/CLI 런처가 재사용할 FastAPI 앱 인스턴스를 반환한다."""
    _apply_beta_mode(beta_mode)
    return app


def is_session_rotate_requested() -> bool:
    """세션 회전 요청 상태 반환."""
    return _server_state.is_session_rotate_requested()


def consume_session_rotate() -> bool:
    """세션 회전 요청을 소비 (1회성). 요청이 있었으면 True."""
    return _server_state.consume_session_rotate()


def set_session_rotate_callback(callback: Callable[[], None]) -> None:
    """CLI에서 세션 회전 시 호출할 콜백 등록."""
    _server_state.set_session_rotate_callback(callback)


def _verify_api_key(request: Request) -> None:
    """MEETING_API_KEY가 설정된 경우 보호 엔드포인트 인증."""
    if not _server_state.api_key:
        return
    provided = request.headers.get("x-api-key") or request.query_params.get("api_key")
    if provided != _server_state.api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _normalize_keyword(keyword: str) -> str:
    keyword = re.sub(r"\s+", " ", (keyword or "").strip(" ,.;:/"))
    if len(keyword) < 2:
        return ""
    return keyword


def _list_input_devices() -> list[dict]:
    from .audio_capture import list_audio_devices

    return list_audio_devices()


def _resolve_device_name(device: int | None, devices: list[dict]) -> str:
    if device is None:
        return "기본 장치"
    for info in devices:
        if info.get("index") == device:
            return str(info.get("name") or f"장치 {device}")
    return f"장치 {device}"


def _audio_device_payload(devices: list[dict] | None = None) -> dict:
    if devices is None:
        devices = _list_input_devices()
    with _server_state.audio_device_lock:
        # 전환 10초 초과 시 자동 해제 (캡처 루프 에러로 리셋 못 한 경우)
        if _server_state.audio_device_switching and _server_state.audio_device_switch_ts > 0 and (time.time() - _server_state.audio_device_switch_ts) > 10:
            _server_state.audio_device_switching = False
            _server_state.audio_device_switch_event.clear()
        current = _server_state.current_audio_device
        requested = _server_state.requested_audio_device
        switching = _server_state.audio_device_switching
        error = _server_state.audio_device_error
    return {
        "devices": devices,
        "current_device": current,
        "current_name": _resolve_device_name(current, devices),
        "requested_device": requested,
        "requested_name": _resolve_device_name(requested, devices),
        "switching": switching,
        "error": error,
    }


def set_current_audio_device(device: int | None, error: str = "") -> None:
    _server_state.set_current_audio_device(device, error=error)


def request_audio_device_switch(device: int | None) -> bool:
    return _server_state.request_audio_device_switch(device)


def consume_audio_device_switch() -> tuple[bool, int | None]:
    return _server_state.consume_audio_device_switch()


def get_audio_device_switch_event() -> threading.Event:
    return _server_state.get_audio_device_switch_event()


def set_startup_status(phase: str, message: str = "", ready: bool = False) -> None:
    _server_state.set_startup_status(phase, message=message, ready=ready)


def set_capture_error(error: str, count: int = 0) -> None:
    """캡처 루프 에러를 상태 API에 노출한다."""
    _server_state.set_capture_error(error, count=count)


def set_voice_active(active: bool) -> None:
    """음성 감지 상태를 갱신한다."""
    _server_state.set_voice_active(active)


def _keyword_payload() -> dict:
    extracted = sorted(
        kw for kw in _server_state.extracted_keywords
        if kw not in _server_state.manual_keywords and kw not in _server_state.promoted_keywords
    )
    manual = sorted(_server_state.manual_keywords)
    promoted = sorted(_server_state.promoted_keywords)
    visible = sorted(set(manual) | set(promoted) | set(extracted))
    return {
        "keywords": visible,
        "prompt_keywords": sorted(get_keywords()),
        "manual": manual,
        "promoted": promoted,
        "extracted": extracted,
        "blocked": sorted(_server_state.blocked_keywords),
        "counts": {kw: _server_state.keyword_seen_counts.get(kw, 0) for kw in extracted},
    }


def get_keywords_snapshot() -> dict:
    """현재 키워드 상태의 직렬화 가능한 스냅샷을 반환한다 (session.json 저장용)."""
    with _server_state.kw_lock:
        return {
            "manual": sorted(_server_state.manual_keywords),
            "extracted": sorted(_server_state.extracted_keywords),
            "promoted": sorted(_server_state.promoted_keywords),
            "blocked": sorted(_server_state.blocked_keywords),
        }


def add_extracted_keywords(keywords: list[str], promote_threshold: int = 2) -> dict:
    """자동 추출 키워드를 누적하고 반복 확인된 것만 승격한다."""
    changed = False
    with _server_state.kw_lock:
        for raw in keywords:
            keyword = _normalize_keyword(raw)
            if not keyword or keyword in _server_state.blocked_keywords or keyword in _server_state.manual_keywords:
                continue
            prev_count = _server_state.keyword_seen_counts.get(keyword, 0)
            _server_state.keyword_seen_counts[keyword] = prev_count + 1
            if _server_state.keyword_seen_counts[keyword] != prev_count:
                changed = True
            if _server_state.keyword_seen_counts[keyword] >= promote_threshold:
                if keyword not in _server_state.promoted_keywords:
                    changed = True
                _server_state.promoted_keywords.add(keyword)
                _server_state.extracted_keywords.discard(keyword)
            elif keyword not in _server_state.promoted_keywords:
                if keyword not in _server_state.extracted_keywords:
                    changed = True
                _server_state.extracted_keywords.add(keyword)
        payload = _keyword_payload()

    if changed:
        _broadcast_sse_sync("keywords_updated", payload)
    return payload


@app.get("/")
async def index() -> HTMLResponse:
    """static/viewer.html 읽어서 HTMLResponse 반환"""
    viewer_path = static_dir() / "viewer.html"
    html = viewer_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/settings")
async def settings_page() -> HTMLResponse:
    """static/settings.html 설정 페이지 반환."""
    settings_path = static_dir() / "settings.html"
    if not settings_path.exists():
        return HTMLResponse(content="<h1>설정 페이지를 찾을 수 없습니다.</h1>", status_code=404)
    html = settings_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/api/settings")
async def get_settings():
    """현재 설정 반환."""
    try:
        from .config import get_config
        cfg = get_config()
        return cfg.to_dict()
    except Exception:
        return {}


@app.post("/api/settings")
async def save_settings(request: Request):
    """설정 저장."""
    try:
        from .config import get_config
        data = await request.json()
        cfg = get_config()
        for key, value in data.items():
            cfg.set(key, value)
        cfg.save()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/audio-devices")
async def list_audio_devices():
    """사용 가능한 오디오 입력 장치 목록 반환."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        result = []
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                result.append({"id": i, "name": d["name"]})
        return result
    except Exception:
        return []


@app.get("/stream")
async def stream(request: Request) -> StreamingResponse:
    """SSE 엔드포인트 — text/event-stream
    각 이벤트: data: {"speaker": "A", "text": "...", "ts": "HH:MM:SS"}\n\n
    """
    _verify_api_key(request)

    client_queue: asyncio.Queue[dict] = asyncio.Queue()
    _server_state.client_queues.add(client_queue)

    async def event_generator():
        try:
            while True:
                try:
                    item = await asyncio.wait_for(client_queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # 연결 유지를 위한 SSE 코멘트 ping
                    yield ": keep-alive\n\n"
                    continue
                except asyncio.CancelledError:
                    break

                if "_type" in item:
                    event_type = item["_type"]
                    payload = item.get("_payload", item)
                    yield f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    continue

                payload = {
                    "speaker": item.get("speaker", "?"),
                    "text": item.get("text", ""),
                    "ts": item.get("ts", ""),
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        finally:
            _server_state.client_queues.discard(client_queue)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket) -> None:
    """WebSocket 엔드포인트 — 전사 세그먼트 실시간 양방향 통신.

    서버→클라이언트: 전사/교정/세션 이벤트 (SSE와 동일 데이터)
    클라이언트→서버: JSON 메시지 (type: edit_speaker, edit_segment, search, ping)
    하트비트: 30초 ping/pong
    """
    # API 키 인증 (쿼리 파라미터 또는 헤더)
    if _server_state.api_key:
        query_key = websocket.query_params.get("api_key", "")
        header_key = (websocket.headers.get("x-api-key") or "").strip()
        if query_key != _server_state.api_key and header_key != _server_state.api_key:
            await websocket.close(code=4003, reason="인증 실패")
            return

    # 최대 연결 수 제한
    if len(_server_state.connected_websockets) >= MAX_WS_CONNECTIONS:
        await websocket.close(code=4008, reason="연결 수 초과")
        return

    await websocket.accept()
    _server_state.connected_websockets.add(websocket)

    # 연결 시 기존 전사 내역 전송 (최근 1000개만)
    MAX_WS_INITIAL_HISTORY = 1000
    try:
        for item in list(_server_state.transcript_history)[-MAX_WS_INITIAL_HISTORY:]:
            msg = json.dumps({"type": "transcript", **item}, ensure_ascii=False)
            await websocket.send_text(msg)
    except Exception:
        _server_state.connected_websockets.discard(websocket)
        return

    # 하트비트 태스크
    async def _heartbeat() -> None:
        try:
            while True:
                await asyncio.sleep(30)
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json({"type": "ping", "ts": time.time()})
        except Exception:
            pass

    heartbeat_task = asyncio.create_task(_heartbeat())

    try:
        while True:
            data = await websocket.receive_text()
            # 메시지 크기 제한
            if len(data) > MAX_WS_MESSAGE_SIZE:
                await websocket.send_json({"type": "error", "message": "메시지 크기 초과"})
                continue
            try:
                msg = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                await websocket.send_json({"type": "error", "message": "잘못된 JSON 형식"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong", "ts": time.time()})

            elif msg_type == "pong":
                # 클라이언트 pong 응답 — 무시
                pass

            elif msg_type == "edit_speaker":
                # 화자 이름 수정: {type: "edit_speaker", index: N, speaker: "새이름"}
                index = msg.get("index")
                new_speaker = msg.get("speaker", "").strip()
                if isinstance(index, int) and 0 <= index < len(_server_state.transcript_history) and new_speaker:
                    _server_state.transcript_history[index]["speaker"] = new_speaker
                    _server_state.speakers.add(new_speaker)
                    await _broadcast_sse("speaker_edited", {
                        "index": index,
                        "speaker": new_speaker,
                    })

            elif msg_type == "edit_segment":
                # 세그먼트 텍스트 수정: {type: "edit_segment", index: N, text: "수정 텍스트"}
                index = msg.get("index")
                new_text = msg.get("text", "").strip()
                if isinstance(index, int) and 0 <= index < len(_server_state.transcript_history) and new_text:
                    _server_state.transcript_history[index]["text"] = new_text
                    await _broadcast_sse("segment_edited", {
                        "index": index,
                        "text": new_text,
                    })

            elif msg_type == "search":
                # 전사 내역 검색: {type: "search", query: "검색어"}
                query = msg.get("query", "").strip().lower()
                if query:
                    results = []
                    for i, item in enumerate(_server_state.transcript_history):
                        if query in item.get("text", "").lower() or query in item.get("speaker", "").lower():
                            results.append({"index": i, **item})
                    await websocket.send_json({
                        "type": "search_result",
                        "query": query,
                        "results": results,
                        "count": len(results),
                    })

            else:
                await websocket.send_json({"type": "error", "message": f"알 수 없는 메시지 타입: {msg_type}"})

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        heartbeat_task.cancel()
        _server_state.connected_websockets.discard(websocket)


@app.get("/history")
async def history() -> list[dict]:
    """현재 세션의 전체 전사 내역 반환."""
    return list(_server_state.transcript_history)


@app.get("/status")
async def status() -> dict:
    """서버 상태 JSON: elapsed, segments, speakers, paused, postprocess"""
    now = time.time()
    elapsed = int(now - _server_state.start_time) if _server_state.start_time > 0 else 0
    result = {
        "elapsed": elapsed,
        "segments": _server_state.segment_count,
        "speakers": sorted(_server_state.speakers),
        "paused": _server_state.paused,
        "keyword_counts": {
            "manual": len(_server_state.manual_keywords),
            "promoted": len(_server_state.promoted_keywords),
            "extracted": len(_server_state.extracted_keywords),
        },
    }
    if _server_state.startup_phase or not _server_state.startup_ready:
        result["startup"] = {
            "phase": _server_state.startup_phase or "booting",
            "message": _server_state.startup_message or "회의 세션 준비 중...",
            "ready": _server_state.startup_ready,
        }
    # 음성 감지 상태 (2초 이내 활성이면 True)
    result["voice_active"] = _server_state.voice_active and (now - _server_state.voice_active_ts < 2.0)
    if _server_state.capture_error:
        result["capture_error"] = {
            "message": _server_state.capture_error,
            "count": _server_state.capture_error_count,
        }
    if _server_state.postprocess_phase:
        result["postprocess"] = {
            "phase": _server_state.postprocess_phase,
            "progress": round(_server_state.postprocess_progress, 1),
        }
    elif _server_state.postprocess_status_file:
        try:
            pp_text = Path(_server_state.postprocess_status_file).read_text(encoding="utf-8")
            pp_data = json.loads(pp_text)
            result["postprocess"] = {
                "phase": pp_data.get("phase", ""),
                "progress": round(pp_data.get("progress", 0), 1),
            }
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return result


@app.post("/toggle-pause")
async def toggle_pause(request: Request) -> dict:
    """일시정지/재개 토글. 일시정지 시 STT 처리를 건너뛴다."""
    _verify_api_key(request)
    return toggle_pause_state()


@app.post("/add-keyword")
async def add_keyword(request: Request, body: dict) -> dict:
    """실시간 키워드 수동 추가 — initial_prompt에 즉시 반영"""
    _verify_api_key(request)
    raw_keyword = body.get("keyword")
    keyword = _normalize_keyword(raw_keyword) if raw_keyword else None
    if keyword:
        _server_state.blocked_keywords.discard(keyword)
        _server_state.manual_keywords.add(keyword)
        _server_state.extracted_keywords.discard(keyword)
        _server_state.promoted_keywords.discard(keyword)
    return _keyword_payload()


@app.post("/remove-keyword")
@app.delete("/remove-keyword")
async def remove_keyword(request: Request, body: dict) -> dict:
    """실시간 키워드 삭제 — 자동 추출 재유입도 막는다."""
    _verify_api_key(request)
    raw_keyword = body.get("keyword")
    keyword = _normalize_keyword(raw_keyword) if raw_keyword else None
    if keyword:
        was_present = (
            keyword in _server_state.manual_keywords
            or keyword in _server_state.extracted_keywords
            or keyword in _server_state.promoted_keywords
        )
        _server_state.manual_keywords.discard(keyword)
        _server_state.extracted_keywords.discard(keyword)
        _server_state.promoted_keywords.discard(keyword)
        if was_present:
            _server_state.blocked_keywords.add(keyword)
            _server_state.keyword_seen_counts.pop(keyword, None)
    return _keyword_payload()


@app.get("/keywords")
async def list_keywords() -> dict:
    """현재 등록된 키워드 목록과 상태를 반환."""
    return _keyword_payload()


@app.get("/devices")
async def list_devices() -> dict:
    """사용 가능한 입력 장치와 현재 선택 상태를 반환."""
    return _audio_device_payload()


@app.post("/switch-device")
async def switch_device(request: Request, body: dict) -> dict:
    """마이크 입력 장치 전환 요청을 등록한다."""
    _verify_api_key(request)
    raw_device = body.get("device")
    if raw_device in ("", "default"):
        device = None
    else:
        try:
            device = None if raw_device is None else int(raw_device)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="장치 인덱스가 올바르지 않습니다.") from exc

    devices = _list_input_devices()
    valid_indexes = {info["index"] for info in devices}
    if device is not None and device not in valid_indexes:
        raise HTTPException(status_code=400, detail="선택한 입력 장치를 찾을 수 없습니다.")

    request_audio_device_switch(device)
    return _audio_device_payload(devices)


def is_paused() -> bool:
    """현재 일시정지 상태인지 반환 (CLI 루프에서 참조)."""
    return _server_state.paused


def toggle_pause_state() -> dict[str, bool | str]:
    """현재 녹음 일시정지 상태를 토글하고 최신 상태를 반환한다."""
    _server_state.paused = not _server_state.paused
    state = "paused" if _server_state.paused else "recording"
    return {"paused": _server_state.paused, "state": state}


def get_keywords() -> set[str]:
    """현재 initial_prompt에 반영할 활성 키워드 집합 반환."""
    return _server_state.manual_keywords.union(_server_state.promoted_keywords)


@app.post("/shutdown")
async def shutdown(request: Request) -> dict:
    """저장 & 종료 요청. CLI 루프에서 감지하여 graceful shutdown."""
    _verify_api_key(request)
    return request_shutdown()


@app.get("/speakers")
async def list_speakers(request: Request) -> dict:
    """등록된 화자 목록 반환."""
    _verify_api_key(request)
    if _server_state.diarizer is None:
        return {"speakers": [], "available": False}
    names = sorted(_server_state.diarizer._enrolled_names)
    return {"speakers": names, "available": True}


@app.post("/enroll")
async def enroll_speaker(request: Request, body: dict) -> dict:
    """웹 UI에서 화자 등록 — base64 PCM float32 오디오 수신.

    body: {"name": "화자명", "audio": "base64 encoded float32 PCM", "sample_rate": 16000}
    """
    _verify_api_key(request)
    import base64
    import numpy as np

    if _server_state.diarizer is None:
        return {"ok": False, "error": "화자 분리가 비활성화됨 (--no-diarize)"}

    name = (body.get("name") or "").strip()
    if not name:
        return {"ok": False, "error": "이름을 입력해주세요"}

    audio_b64 = body.get("audio", "")
    if not audio_b64:
        return {"ok": False, "error": "오디오 데이터가 없습니다"}

    sample_rate = int(body.get("sample_rate", 16000))

    try:
        audio_bytes = base64.b64decode(audio_b64)
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
    except Exception:
        return {"ok": False, "error": "오디오 디코딩 실패"}

    if len(audio) < sample_rate * 2:
        return {"ok": False, "error": "최소 2초 이상 녹음해주세요"}

    # 유사도 기반 중복 검사
    dup_name, dup_sim = _server_state.diarizer.check_duplicate(audio, sample_rate)
    if dup_name:
        return {
            "ok": False,
            "error": f"'{dup_name}'과(와) 유사합니다 (유사도 {dup_sim:.0%})",
            "duplicate": dup_name,
            "similarity": round(dup_sim, 3),
        }

    # 이름 중복 확인
    if name in _server_state.diarizer._enrolled_names:
        return {
            "ok": False,
            "error": f"'{name}'은(는) 이미 등록되어 있습니다",
            "duplicate": name,
        }

    # 등록
    try:
        _server_state.diarizer.enroll(name, audio, sample_rate)
        if _server_state.profiles_path:
            _server_state.diarizer.save_profiles(_server_state.profiles_path)
        return {
            "ok": True,
            "name": name,
            "speakers": sorted(_server_state.diarizer._enrolled_names),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _session_contains_keyword(session_dir: Path, keyword: str) -> bool:
    """세션의 전사 파일에서 키워드 포함 여부를 확인한다."""
    for candidate in ("meeting.md", "meeting.raw.txt"):
        transcript_file = session_dir / candidate
        if transcript_file.is_file():
            try:
                content = transcript_file.read_text(encoding="utf-8").lower()
                return keyword in content
            except OSError:
                pass
            break

    # JSONL 파일에서도 검색
    jsonl_file = session_dir / "meeting.stt.jsonl"
    if jsonl_file.is_file():
        try:
            for line in jsonl_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    seg = json.loads(line)
                    if keyword in str(seg.get("text", "")).lower():
                        return True
                except (json.JSONDecodeError, ValueError):
                    continue
        except OSError:
            pass

    return False


def _trim_meeting_md_transcript(lines: list[str]) -> list[str]:
    """meeting.md에서 '# Raw Data' 이후 섹션을 제외한다."""
    for idx, line in enumerate(lines):
        if _RAW_DATA_HEADING_RE.match(line.strip()):
            return lines[:idx]
    return lines


def _scan_sessions() -> list[dict]:
    """output/meetings 디렉토리를 스캔하여 세션 메타데이터 목록을 반환한다."""
    meetings_root = OUTPUT_ROOT / "meetings"
    if not meetings_root.is_dir():
        return []

    sessions: list[dict] = []
    for date_dir in meetings_root.iterdir():
        if not date_dir.is_dir():
            continue
        date_str = date_dir.name  # YYYY-MM-DD
        for time_dir in date_dir.iterdir():
            if not time_dir.is_dir():
                continue
            try:
                if not any(time_dir.iterdir()):
                    # 빈 세션 디렉토리는 목록에서 제외 (중복/유령 세션 방지)
                    continue
            except OSError:
                continue
            time_str = time_dir.name  # HHMMSS
            session_id = f"{date_str}_{time_str}"

            session_json = time_dir / "session.json"
            if session_json.is_file():
                # session.json이 있으면 메타데이터 사용
                try:
                    meta = json.loads(session_json.read_text(encoding="utf-8"))
                    seg_count = meta.get("segment_count", 0)
                    if seg_count < 1:
                        continue  # 세그먼트 0개 세션 숨김
                    sessions.append({
                        "id": session_id,
                        "date": date_str,
                        "time": time_str,
                        "duration": meta.get("duration", ""),
                        "segments": seg_count,
                        "speakers": meta.get("speaker_count", 0),
                        "path": str(time_dir),
                    })
                    continue
                except (json.JSONDecodeError, OSError):
                    pass

            # 폴백: 디렉토리 구조 + transcript 파일에서 추출
            segment_count = 0
            for candidate in ("meeting.md", "meeting.raw.txt", "meeting.stt.jsonl"):
                transcript_file = time_dir / candidate
                if transcript_file.is_file():
                    try:
                        lines = transcript_file.read_text(encoding="utf-8").splitlines()
                        segment_count = sum(
                            1 for ln in lines if ln.startswith("- [") or (candidate.endswith(".jsonl") and ln.strip())
                        )
                    except OSError:
                        pass
                    break

            if segment_count < 1:
                continue  # 세그먼트 0개 세션 숨김

            sessions.append({
                "id": session_id,
                "date": date_str,
                "time": time_str,
                "duration": "",
                "segments": segment_count,
                "speakers": 0,
                "path": str(time_dir),
            })

    # 최신순 정렬
    sessions.sort(key=lambda s: s["id"], reverse=True)
    return sessions


@app.get("/api/sessions")
async def list_sessions(
    request: Request,
    q: str | None = Query(None, description="키워드 검색 (세그먼트 text 필드)"),
    from_date: str | None = Query(None, description="시작 날짜 (YYYY-MM-DD)"),
    to_date: str | None = Query(None, description="종료 날짜 (YYYY-MM-DD)"),
) -> list[dict]:
    """세션 목록 반환. output/meetings/{date}/{time} 스캔.

    쿼리 파라미터:
      - q: 키워드 검색 — 전사 텍스트에서 해당 키워드를 포함하는 세션만 반환
      - from_date: 시작 날짜 필터 (YYYY-MM-DD, 이상)
      - to_date: 종료 날짜 필터 (YYYY-MM-DD, 이하)
    파라미터 없으면 전체 반환 (하위호환).
    """
    _verify_api_key(request)
    loop = asyncio.get_running_loop()
    sessions = await loop.run_in_executor(None, _scan_sessions)

    # 날짜 범위 필터링 (세션 디렉토리명 YYYY-MM-DD 기반)
    if from_date:
        sessions = [s for s in sessions if s["date"] >= from_date]
    if to_date:
        sessions = [s for s in sessions if s["date"] <= to_date]

    # 키워드 검색 — 세그먼트 text 필드에서 검색
    if q:
        keyword = q.lower()
        filtered: list[dict] = []
        for session in sessions:
            session_dir = Path(session["path"])
            if _session_contains_keyword(session_dir, keyword):
                filtered.append(session)
        sessions = filtered

    return sessions


@app.get("/api/sessions/{session_id}")
async def get_session(request: Request, session_id: str) -> dict:
    """세션 ID로 상세 정보 + 전사 내용 반환.

    session_id 형식: "YYYY-MM-DD_HHMMSS"
    """
    _verify_api_key(request)
    session_dir, _ = _resolve_session_dir(session_id)
    date_str, time_str = session_id.split("_", 1)
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    # 메타데이터 로드
    meta: dict = {}
    session_json = session_dir / "session.json"
    if session_json.is_file():
        try:
            meta = json.loads(session_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    # 전사 내용 로드 (meeting.md 우선, 폴백으로 meeting.raw.txt)
    transcript_lines: list[str] = []
    transcript_source = ""
    for candidate in ("meeting.md", "meeting.raw.txt"):
        transcript_file = session_dir / candidate
        if transcript_file.is_file():
            try:
                transcript_lines = transcript_file.read_text(encoding="utf-8").splitlines()
                transcript_source = candidate
                if transcript_source == "meeting.md":
                    transcript_lines = _trim_meeting_md_transcript(transcript_lines)
            except OSError:
                pass
            break

    # alignment 로드 (있으면)
    alignment: list[dict] = []
    alignment_file = session_dir / "meeting.stt.jsonl"
    if alignment_file.is_file():
        try:
            for line in alignment_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    # avg_logprob → confidence 변환 (뷰어 low-confidence 표시용)
                    if "confidence" not in entry and "avg_logprob" in entry:
                        lp = entry["avg_logprob"]
                        if isinstance(lp, (int, float)):
                            entry["confidence"] = round(max(0.0, min(1.0, 1.0 + lp)), 3)
                    alignment.append(entry)
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "id": session_id,
        "date": date_str,
        "time": time_str,
        "duration": meta.get("duration", ""),
        "segments": meta.get("segment_count", 0),
        "speakers": meta.get("speaker_count", 0),
        "transcript_source": transcript_source,
        "transcript": transcript_lines,
        "alignment": alignment,
        "keywords": meta.get("keywords", {}),
        "meta": meta,
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(request: Request, session_id: str) -> dict:
    """세션 디렉토리를 삭제하고 session_deleted SSE 이벤트를 전송한다."""
    _verify_api_key(request)

    session_dir, _ = _resolve_session_dir(session_id)
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="이 세션에는 저장된 데이터가 없습니다.")

    shutil.rmtree(session_dir)
    await _broadcast_sse("session_deleted", _session_payload(session_id, session_dir))
    return {"deleted": True, "session_id": session_id}


@app.get("/api/sessions/{session_id}/search")
async def search_session(
    request: Request,
    session_id: str,
    query: str = Query(..., description="검색어"),
    speaker: str | None = Query(None, description="화자 필터"),
    time_start: float | None = Query(None, description="시작 시간(초)"),
    time_end: float | None = Query(None, description="종료 시간(초)"),
    regex: bool = Query(False, description="정규식 모드"),
) -> dict:
    """세션 전사 내용에서 검색 필터 조합 + 정규식 검색.

    query, speaker, time_start, time_end를 AND 조건으로 필터링한다.
    regex=true이면 query를 정규식 패턴으로 처리한다.
    """
    _verify_api_key(request)

    session_dir, _ = _resolve_session_dir(session_id)
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    # 정규식 컴파일 (잘못된 패턴은 400 에러)
    if regex:
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error as exc:
            raise HTTPException(
                status_code=400,
                detail=f"잘못된 정규식 패턴입니다: {exc}",
            ) from exc
    else:
        query_lower = query.lower()

    # alignment(JSONL) 로드
    segments: list[dict] = []
    jsonl_file = session_dir / "meeting.stt.jsonl"
    if jsonl_file.is_file():
        try:
            for line in jsonl_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    segments.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    continue
        except OSError:
            pass

    # JSONL이 없으면 meeting.md에서 파싱
    if not segments:
        md_pattern = re.compile(
            r"^- \[(\d{2}:\d{2}:\d{2})\]\s*(.+?):\s*(.+)$"
        )
        for candidate in ("meeting.md", "meeting.raw.txt"):
            transcript_file = session_dir / candidate
            if transcript_file.is_file():
                try:
                    for line_text in transcript_file.read_text(encoding="utf-8").splitlines():
                        m = md_pattern.match(line_text)
                        if m:
                            ts_str = m.group(1)
                            parts = ts_str.split(":")
                            ts_seconds = (
                                int(parts[0]) * 3600
                                + int(parts[1]) * 60
                                + int(parts[2])
                            )
                            segments.append({
                                "text": m.group(3),
                                "speaker": m.group(2),
                                "start": float(ts_seconds),
                            })
                except OSError:
                    pass
                break

    # AND 조건 필터링
    matches: list[dict] = []
    for idx, seg in enumerate(segments):
        text = str(seg.get("text", ""))
        seg_speaker = str(seg.get("speaker", "?"))
        timestamp = float(seg.get("start", 0.0))

        # 텍스트 검색
        if regex:
            if not pattern.search(text):
                continue
        else:
            if query_lower not in text.lower():
                continue

        # 화자 필터
        if speaker and seg_speaker != speaker:
            continue

        # 시간 범위 필터
        if time_start is not None and timestamp < time_start:
            continue
        if time_end is not None and timestamp > time_end:
            continue

        matches.append({
            "index": idx,
            "text": text,
            "speaker": seg_speaker,
            "timestamp": timestamp,
        })

    return {"matches": matches, "total": len(matches)}


@app.get("/api/sessions/{session_id}/export", response_model=None)
async def export_session_endpoint(
    request: Request,
    session_id: str,
    format: str = Query("txt", description="내보내기 포맷 (txt, md, docx 또는 pdf)"),
) -> PlainTextResponse | StreamingResponse:
    """세션 전사 내용을 TXT/MD/DOCX/PDF 포맷으로 내보낸다."""
    _verify_api_key(request)

    fmt = format.lower()
    if fmt not in ("txt", "md", "docx", "pdf"):
        raise HTTPException(
            status_code=400,
            detail="지원하지 않는 포맷입니다. txt, md, docx 또는 pdf만 가능합니다.",
        )

    session_dir, _ = _resolve_session_dir(session_id)
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    from .export import export_session

    content = export_session(session_dir, fmt)

    if fmt == "pdf":
        import io
        from urllib.parse import quote

        content_bytes = content if isinstance(content, bytes) else content.encode("utf-8")
        encoded_name = quote(f"회의록_{session_id}.pdf")
        return StreamingResponse(
            io.BytesIO(content_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    f"attachment; filename=\"meeting_{session_id}.pdf\"; "
                    f"filename*=UTF-8''{encoded_name}"
                ),
            },
        )

    if fmt == "docx":
        import io
        from urllib.parse import quote

        content_bytes = content if isinstance(content, bytes) else content.encode("utf-8")
        encoded_name = quote(f"회의록_{session_id}.docx")
        return StreamingResponse(
            io.BytesIO(content_bytes),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": (
                    f"attachment; filename=\"meeting_{session_id}.docx\"; "
                    f"filename*=UTF-8''{encoded_name}"
                ),
            },
        )

    media_type = "text/markdown" if fmt == "md" else "text/plain"
    text_content = content if isinstance(content, str) else content.decode("utf-8")
    return PlainTextResponse(content=text_content, media_type=media_type)


@app.post("/api/push-segment")
async def push_segment_endpoint(request: Request) -> dict:
    """단일 세그먼트를 SSE로 push. continuous 모드에서 호출."""
    body = await request.json()
    text = body.get("text", "").strip()
    ts = body.get("ts", "")
    speaker = body.get("speaker", "화자")
    confidence = body.get("confidence")
    if text:
        await push_transcript(speaker, text, ts, confidence=confidence)
    return {"ok": True}


@app.post("/api/load-transcript")
async def load_transcript_file(request: Request) -> dict:
    """스트림 전사 파일을 로드하여 뷰어에 SSE로 푸시한다.

    Body: {"path": "output/transcripts/2026-03-16/transcript.corrected.txt"}
    """
    import re as _re

    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="path 파라미터가 필요합니다.") from exc

    raw_path = body.get("path") if isinstance(body, dict) else None
    path_value = str(raw_path).strip() if raw_path is not None else ""
    if not path_value:
        raise HTTPException(status_code=400, detail="path 파라미터가 필요합니다.")

    normalized_path = os.path.normpath(path_value)
    candidate_path = Path(normalized_path)
    if not candidate_path.is_absolute():
        candidate_path = Path.cwd() / candidate_path

    try:
        file_path = candidate_path.resolve()
    except OSError as exc:
        raise HTTPException(status_code=400, detail="잘못된 파일 경로입니다.") from exc

    output_root = OUTPUT_ROOT.resolve()
    try:
        file_path.relative_to(output_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="output 디렉토리 밖의 파일은 불러올 수 없습니다.",
        ) from exc

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    # [HH:MM ~ HH:MM] 텍스트 형식 파싱
    pattern = _re.compile(r"\[(\d{2}:\d{2})\s*~\s*(\d{2}:\d{2})\]\s*(.*)")
    count = 0
    for line in text.strip().splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        ts_start, _ts_end, segment_text = m.groups()
        await push_transcript("화자", segment_text.strip(), ts_start)
        count += 1

    # startup 상태를 ready로 전환
    set_startup_status("ready", "스트림 전사 로드 완료", ready=True)

    return {"loaded": count, "file": str(file_path)}


def _read_profiles_json() -> dict:
    """화자 프로필 JSON 파일을 읽어 반환한다. 파일이 없으면 빈 구조 반환."""
    profiles_file = Path(_server_state.profiles_path) if _server_state.profiles_path else (OUTPUT_ROOT / "data" / "speakers.json")
    if not profiles_file.is_file():
        return {"speakers": {}}
    try:
        data = json.loads(profiles_file.read_text(encoding="utf-8"))
        if "speakers" not in data:
            data["speakers"] = {}
        return data
    except (json.JSONDecodeError, OSError):
        return {"speakers": {}}


def _write_profiles_json(data: dict) -> None:
    """화자 프로필 JSON 파일에 원자적으로 저장한다 (tempfile + os.replace)."""
    profiles_file = Path(_server_state.profiles_path) if _server_state.profiles_path else (OUTPUT_ROOT / "data" / "speakers.json")
    profiles_file.parent.mkdir(parents=True, exist_ok=True)
    # 임시 파일에 먼저 쓰고 os.replace로 원자적 교체 (쓰기 중 충돌 시 데이터 손실 방지)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(profiles_file.parent), suffix=".tmp", prefix=".profiles_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, str(profiles_file))
    except BaseException:
        # 실패 시 임시 파일 정리
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _validate_profile_name(name: str) -> str:
    """화자 이름 유효성 검사 및 path traversal 방지."""
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="화자 이름은 필수입니다.")
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="잘못된 화자 이름입니다.")
    return name


def _get_audio_duration(file_path: str) -> float:
    """오디오 파일의 재생 시간(초)을 반환한다. 실패 시 0.0."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-show_entries",
                "format=duration", "-of", "csv=p=0", file_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass

    # ffprobe 실패 시 soundfile 시도
    try:
        import soundfile as sf

        info = sf.info(file_path)
        return float(info.duration)
    except Exception:
        pass

    return 0.0


# --- 임베딩 연동 헬퍼 ---
_PROFILE_MATCH_THRESHOLD = 0.70  # 프로필 매칭 코사인 유사도 임계값


def _extract_embedding_from_file(file_path: str) -> list[float]:
    """오디오 파일에서 화자 임베딩 벡터를 추출한다.

    _diarizer가 활성화되어 있으면 pyannote 임베딩을 추출하고,
    비활성화 상태면 빈 리스트를 반환한다.
    """
    if _server_state.diarizer is None:
        return []

    try:
        import numpy as np
        import soundfile as sf

        audio, sample_rate = sf.read(file_path, dtype="float32")

        # 스테레오 → 모노 변환
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        # 최소 1초 이상 필요
        if len(audio) < sample_rate:
            return []

        embedding = _server_state.diarizer._extract_embedding(audio, sample_rate)
        return embedding.tolist()
    except Exception:
        return []


def _find_matching_profile(
    embedding: list[float],
    exclude_name: str = "",
    threshold: float = _PROFILE_MATCH_THRESHOLD,
) -> tuple[str | None, float]:
    """기존 프로필과 임베딩 코사인 유사도를 비교하여 매칭되는 화자를 찾는다.

    Args:
        embedding: 비교할 임베딩 벡터
        exclude_name: 비교에서 제외할 화자 이름 (자기 자신 업데이트 시)
        threshold: 매칭 임계값 (기본 0.70)

    Returns:
        (매칭된 화자 이름, 유사도). 매칭 없으면 (None, 0.0).
    """
    if not embedding or _server_state.diarizer is None:
        return None, 0.0

    import numpy as np

    data = _read_profiles_json()
    speakers = data.get("speakers", {})

    target = np.array(embedding, dtype=np.float32)
    best_name: str | None = None
    best_sim: float = 0.0

    for name, info in speakers.items():
        if name == exclude_name:
            continue
        known_emb = info.get("embedding", [])
        if not known_emb:
            continue
        known = np.array(known_emb, dtype=np.float32)
        sim = float(_server_state.diarizer._cosine_similarity(target, known))
        if sim > best_sim:
            best_sim = sim
            best_name = name

    if best_sim >= threshold:
        return best_name, best_sim
    return None, best_sim


@app.get("/speaker-profile")
async def speaker_profile_page() -> FileResponse:
    """화자 프로필 관리 HTML 페이지 서빙."""
    html_path = static_dir() / "speaker_profile.html"
    if not html_path.is_file():
        raise HTTPException(status_code=404, detail="speaker_profile.html을 찾을 수 없습니다.")
    return FileResponse(str(html_path), media_type="text/html")


@app.get("/api/profiles")
async def get_profiles(request: Request) -> dict:
    """전체 화자 프로필 목록 반환."""
    _verify_api_key(request)
    async with _server_state.profiles_lock:
        data = _read_profiles_json()
    return data


@app.post("/api/profiles")
async def create_profile(
    request: Request,
    name: str = Form(...),
    audio: UploadFile = File(...),
    description: str = Form(""),
) -> dict:
    """새 화자 프로필 추가. multipart/form-data로 name, audio, description 수신."""
    _verify_api_key(request)
    name = _validate_profile_name(name)

    # 오디오 파일 처리 (락 밖에서 I/O 수행)
    audio_dir = OUTPUT_ROOT / "data" / "speaker_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(audio.filename or "audio.wav").suffix or ".wav"
    safe_filename = re.sub(r"[^\w\-.]", "_", name) + ext
    audio_path = audio_dir / safe_filename

    content = await audio.read()
    audio_path.write_bytes(content)

    duration = _get_audio_duration(str(audio_path))

    # 임베딩 추출 (diarizer 활성화 시)
    embedding = _extract_embedding_from_file(str(audio_path))

    async with _server_state.profiles_lock:
        data = _read_profiles_json()
        if name in data["speakers"]:
            raise HTTPException(status_code=409, detail=f"'{name}'은(는) 이미 등록된 화자입니다.")

        # 임베딩 기반 중복 화자 검사 (임계값 0.7)
        if embedding:
            dup_name, dup_sim = _find_matching_profile(embedding)
            if dup_name:
                return {
                    "ok": False,
                    "error": f"'{dup_name}'과(와) 유사합니다 (유사도 {dup_sim:.0%})",
                    "duplicate": dup_name,
                    "similarity": round(dup_sim, 3),
                }

        data["speakers"][name] = {
            "embedding": embedding,
            "enrolled_at": datetime.now().isoformat(),
            "duration_seconds": round(duration, 2),
            "description": description.strip(),
        }
        _write_profiles_json(data)

    # diarizer 런타임에도 등록 (실시간 화자 식별에 즉시 반영)
    if _server_state.diarizer is not None and embedding:
        import numpy as np

        _server_state.diarizer.speaker_embeddings[name] = np.array(embedding, dtype=np.float32)
        _server_state.diarizer._speaker_counts[name] = 0
        _server_state.diarizer._enrolled_names.add(name)
        _server_state.diarizer._profile_mode = True

    return {"ok": True, "name": name, "speakers": data["speakers"]}


@app.put("/api/profiles/{name}")
async def update_profile(
    request: Request,
    name: str,
    description: str = Form(None),
    audio: UploadFile | None = File(None),
) -> dict:
    """화자 프로필 수정. description 변경, audio 재업로드(선택)."""
    _verify_api_key(request)
    name = _validate_profile_name(name)

    # 오디오 I/O는 락 밖에서 수행
    audio_content: bytes | None = None
    if audio is not None:
        audio_content = await audio.read()

    async with _server_state.profiles_lock:
        data = _read_profiles_json()
        if name not in data["speakers"]:
            raise HTTPException(status_code=404, detail=f"'{name}' 화자를 찾을 수 없습니다.")

        speaker = data["speakers"][name]

        if description is not None:
            speaker["description"] = description.strip()

        if audio_content is not None:
            audio_dir = OUTPUT_ROOT / "data" / "speaker_audio"
            audio_dir.mkdir(parents=True, exist_ok=True)

            ext = Path(audio.filename or "audio.wav").suffix or ".wav"
            safe_filename = re.sub(r"[^\w\-.]", "_", name) + ext
            audio_path = audio_dir / safe_filename

            audio_path.write_bytes(audio_content)

            duration = _get_audio_duration(str(audio_path))
            speaker["duration_seconds"] = round(duration, 2)

            # 임베딩 재추출 (diarizer 활성화 시)
            embedding = _extract_embedding_from_file(str(audio_path))

            # 임베딩 기반 중복 화자 검사 (자기 자신 제외)
            if embedding:
                dup_name, dup_sim = _find_matching_profile(embedding, exclude_name=name)
                if dup_name:
                    return {
                        "ok": False,
                        "error": f"'{dup_name}'과(와) 유사합니다 (유사도 {dup_sim:.0%})",
                        "duplicate": dup_name,
                        "similarity": round(dup_sim, 3),
                    }

            speaker["embedding"] = embedding

            # diarizer 런타임에도 갱신 (실시간 화자 식별에 즉시 반영)
            if _server_state.diarizer is not None and embedding:
                import numpy as np

                _server_state.diarizer.speaker_embeddings[name] = np.array(embedding, dtype=np.float32)
                _server_state.diarizer._speaker_counts[name] = 0
                _server_state.diarizer._enrolled_names.add(name)
                _server_state.diarizer._profile_mode = True

        _write_profiles_json(data)

    return {"ok": True, "name": name, "speaker": speaker}


@app.delete("/api/profiles/{name}")
async def delete_profile(request: Request, name: str) -> dict:
    """화자 프로필 삭제."""
    _verify_api_key(request)
    name = _validate_profile_name(name)

    async with _server_state.profiles_lock:
        data = _read_profiles_json()
        if name not in data["speakers"]:
            raise HTTPException(status_code=404, detail=f"'{name}' 화자를 찾을 수 없습니다.")

        del data["speakers"][name]
        _write_profiles_json(data)

    # 오디오 파일도 정리 (락 밖에서 수행)
    audio_dir = OUTPUT_ROOT / "data" / "speaker_audio"
    if audio_dir.is_dir():
        safe_prefix = re.sub(r"[^\w\-.]", "_", name)
        for f in audio_dir.iterdir():
            if f.name.startswith(safe_prefix):
                try:
                    f.unlink()
                except OSError:
                    pass

    return {"ok": True, "deleted": name, "speakers": data["speakers"]}


# --- 미등록 화자 자동 등록 API ---


@app.get("/api/speakers/unknown")
async def list_unknown_speakers(request: Request) -> dict:
    """미등록 화자 목록 반환 (임베딩 수, 첫/마지막 발화 시간, 자동 등록 후보 여부)."""
    _verify_api_key(request)
    if _server_state.diarizer is None:
        return {"unknown_speakers": [], "available": False}
    return {"unknown_speakers": _unknown_tracker.list_unknown(), "available": True}


@app.post("/api/speakers/auto-register")
async def auto_register_speaker(request: Request, body: dict) -> dict:
    """미등록 화자를 프로필에 등록한다.

    body: {"speaker_id": "unknown_1", "name": "화자명(선택)"}
    name 미지정 시 기본 이름 Speaker_N 부여.
    """
    _verify_api_key(request)
    import numpy as np

    if _server_state.diarizer is None:
        raise HTTPException(status_code=400, detail="화자 분리가 비활성화되어 있습니다.")

    speaker_id = (body.get("speaker_id") or "").strip()
    if not speaker_id:
        raise HTTPException(status_code=400, detail="speaker_id는 필수입니다.")

    info = _unknown_tracker.get(speaker_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"미등록 화자 '{speaker_id}'를 찾을 수 없습니다.")

    # 평균 임베딩으로 프로필 등록
    embedding = info["mean"]

    async with _server_state.profiles_lock:
        # 이름 결정 (미지정 시 자동 생성)
        name = (body.get("name") or "").strip()
        if not name:
            data = _read_profiles_json()
            existing_count = len(data.get("speakers", {}))
            name = f"Speaker_{existing_count + 1}"

        name = _validate_profile_name(name)

        # 기존 프로필과 이름 중복 확인
        data = _read_profiles_json()
        if name in data.get("speakers", {}):
            raise HTTPException(status_code=409, detail=f"'{name}'은(는) 이미 등록된 화자입니다.")

        data["speakers"][name] = {
            "embedding": embedding,
            "enrolled_at": datetime.now().isoformat(),
            "description": f"자동 등록 (세그먼트 {len(info['embeddings'])}개 평균)",
        }
        _write_profiles_json(data)

    # diarizer 런타임에도 즉시 반영
    _server_state.diarizer.speaker_embeddings[name] = np.array(embedding, dtype=np.float32)
    _server_state.diarizer._speaker_counts[name] = 0
    _server_state.diarizer._enrolled_names.add(name)
    _server_state.diarizer._profile_mode = True

    # 미등록 목록에서 제거
    _unknown_tracker.remove(speaker_id)

    return {
        "ok": True,
        "name": name,
        "speaker_id": speaker_id,
        "segment_count": len(info["embeddings"]),
    }


@app.delete("/api/speakers/unknown/{speaker_id}")
async def delete_unknown_speaker(request: Request, speaker_id: str) -> dict:
    """미등록 화자를 무시/삭제한다."""
    _verify_api_key(request)

    if "/" in speaker_id or "\\" in speaker_id or ".." in speaker_id:
        raise HTTPException(status_code=400, detail="잘못된 speaker_id입니다.")

    if not _unknown_tracker.remove(speaker_id):
        raise HTTPException(status_code=404, detail=f"미등록 화자 '{speaker_id}'를 찾을 수 없습니다.")

    return {"ok": True, "deleted": speaker_id}


@app.post("/api/sessions/new")
async def new_session(request: Request) -> dict:
    """현재 세션을 저장하고 새 세션을 시작. 모델은 유지."""
    _verify_api_key(request)

    # 콜백은 CLI 스레드에서 플래그 감지 시 실행 (async 블로킹 방지)
    # 서버 상태 리셋 (키워드 포함)
    _server_state.reset_for_new_session()

    # 미등록 화자 임시 데이터 리셋
    _unknown_tracker.reset()

    # 키워드 초기화를 뷰어에 즉시 반영
    await _broadcast_sse("keywords_updated", _keyword_payload())

    # 세션 회전 플래그 설정 (CLI/desktop 루프에서 consume_session_rotate()로 감지 → 콜백 호출)
    _server_state.session_rotate_event.set()

    now = datetime.now()
    new_id = now.strftime("%Y-%m-%d_%H%M%S")

    # 세션 디렉토리는 파이프라인 회전 시 MeetingWriter가 생성 (중복 방지)

    return {"session_id": new_id, "message": "새 세션을 시작합니다."}


def is_shutdown_requested() -> bool:
    """종료 요청 상태 반환 (CLI 루프에서 참조)."""
    return _server_state.shutdown_requested


def request_shutdown() -> dict[str, bool | str]:
    """저장 후 종료 플래그를 설정하고 응답 페이로드를 반환한다."""
    return _server_state.request_shutdown()


def signal_server_shutdown() -> bool:
    """실행 중인 uvicorn 서버에 graceful 종료 신호를 보낸다."""
    with _uvicorn_server_lock:
        server = _uvicorn_server
    if server is None:
        return False
    server.should_exit = True
    return True


def set_diarizer(diarizer, profiles_path: str | None = None) -> None:
    """화자 분리기 참조 설정 (CLI에서 호출)."""
    _server_state.set_diarizer(diarizer, profiles_path=profiles_path)


def set_postprocess_status(phase: str, progress: float = 0.0) -> None:
    """후처리 진행 상태 설정 (웹 UI 실시간 표시용)."""
    _server_state.set_postprocess_status(phase, progress=progress)


def set_postprocess_status_file(path: str | None) -> None:
    """파일 기반 후처리 상태 경로 설정 (별도 프로세스용)."""
    _server_state.set_postprocess_status_file(path)


async def push_transcript(
    speaker: str,
    text: str,
    timestamp: str,
    *,
    count_segment: bool = True,
    confidence: float | None = None,
) -> None:
    """큐에 전사 결과 push (외부에서 호출). confidence: 0~1 (낮을수록 불확실)"""

    normalized_speaker = (speaker or "?").strip() or "?"
    normalized_text = (text or "").strip()
    normalized_timestamp = (timestamp or "").strip()

    if not normalized_timestamp:
        normalized_timestamp = time.strftime("%H:%M:%S", time.localtime())

    if count_segment:
        _server_state.segment_count += 1
    _server_state.speakers.add(normalized_speaker)

    item = {
        "speaker": normalized_speaker,
        "text": normalized_text,
        "ts": normalized_timestamp,
    }
    if confidence is not None:
        item["confidence"] = round(confidence, 3)
    _server_state.transcript_history.append(item)
    await _broadcast_item(item)
    await _broadcast_sse("segment_created", item)


async def push_correction(corrections: list[dict]) -> None:
    """기존 전사 내역을 교정하고 correction SSE 이벤트를 브로드캐스트한다."""
    normalized_corrections: list[dict] = []
    for correction in corrections:
        if not isinstance(correction, dict):
            continue

        index = correction.get("index")
        if not isinstance(index, int):
            continue
        if index < 0 or index >= len(_server_state.transcript_history):
            continue

        original = str(correction.get("original", ""))
        corrected = str(correction.get("corrected", ""))
        if not corrected:
            continue

        _server_state.transcript_history[index]["text"] = corrected
        normalized_corrections.append(
            {
                "index": index,
                "original": original,
                "corrected": corrected,
            }
        )

    if not normalized_corrections:
        return

    await _broadcast_sse("correction", {"corrections": normalized_corrections})


def push_transcript_sync(
    speaker: str,
    text: str,
    timestamp: str,
    *,
    confidence: float | None = None,
) -> None:
    """스레드 안전한 동기 래퍼 — 별도 스레드에서 SSE 큐에 push."""

    # status API의 세그먼트 수를 동기 호출 시점에 즉시 반영
    _server_state.segment_count += 1

    if _server_state.event_loop is not None and _server_state.event_loop.is_running():
        asyncio.run_coroutine_threadsafe(
            push_transcript(speaker, text, timestamp, count_segment=False, confidence=confidence),
            _server_state.event_loop,
        )


def push_correction_sync(corrections: list[dict]) -> None:
    if _server_state.event_loop is not None and _server_state.event_loop.is_running():
        asyncio.run_coroutine_threadsafe(push_correction(corrections), _server_state.event_loop)


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    application: FastAPI | None = None,
    beta_mode: bool | None = None,
) -> None:
    """uvicorn으로 서버 실행. 종료 신호 전달을 위해 Server 인스턴스를 보관한다."""
    import uvicorn

    _apply_beta_mode(beta_mode)
    config = uvicorn.Config(
        application or create_app(beta_mode=beta_mode),
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    with _uvicorn_server_lock:
        global _uvicorn_server
        _uvicorn_server = server
    try:
        server.run()
    finally:
        with _uvicorn_server_lock:
            if _uvicorn_server is server:
                _uvicorn_server = None


class _ServerModule(ModuleType):
    def __getattr__(self, name: str):
        state_attr = _STATE_COMPAT_ATTRS.get(name)
        if state_attr is not None:
            return getattr(_server_state, state_attr)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        state_attr = _STATE_COMPAT_ATTRS.get(name)
        if state_attr is not None:
            setattr(_server_state, state_attr, value)
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
    "stream",
    "status",
    "list_devices",
    "push_transcript",
    "push_transcript_sync",
    "push_correction",
    "push_correction_sync",
    "run_server",
    "_client_queues",
    "_connected_websockets",
    "ws_transcribe",
    "is_paused",
    "toggle_pause_state",
    "add_keyword",
    "remove_keyword",
    "list_keywords",
    "get_keywords",
    "add_extracted_keywords",
    "consume_audio_device_switch",
    "get_audio_device_switch_event",
    "set_current_audio_device",
    "set_startup_status",
    "set_capture_error",
    "set_voice_active",
    "switch_device",
    "shutdown",
    "is_shutdown_requested",
    "request_shutdown",
    "signal_server_shutdown",
    "set_postprocess_status",
    "set_postprocess_status_file",
    "list_sessions",
    "get_session",
    "new_session",
    "search_session",
    "export_session_endpoint",
    "speaker_profile_page",
    "get_profiles",
    "create_profile",
    "update_profile",
    "delete_profile",
    "list_unknown_speakers",
    "auto_register_speaker",
    "delete_unknown_speaker",
    "_unknown_tracker",
    "is_session_rotate_requested",
    "consume_session_rotate",
    "set_session_rotate_callback",
]
