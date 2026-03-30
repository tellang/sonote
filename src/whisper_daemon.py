"""Whisper STT 데몬 — TCP IPC 서버 + VRAM 자동 스케일링.

클래스:
  VRAMScaler       — VRAM 모니터링 + 워커 자동 스케일링
  WhisperServiceDaemon — line-delimited JSON over TCP 소켓 서버
  WhisperRemoteClient  — WhisperWorkerPool duck-type 프록시
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import socket
import socketserver
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .runtime_env import get_available_vram
from .whisper_worker import WhisperWorkerPool

# 보안: 최대 audio 페이로드 100MB (float32 16kHz × ~26분)
_MAX_AUDIO_BYTES = 100 * 1024 * 1024
# 보안: 최대 recv 버퍼 128MB (base64 인코딩 오버헤드 포함)
_MAX_RECV_BYTES = 128 * 1024 * 1024


# ---------------------------------------------------------------------------
# VRAMScaler
# ---------------------------------------------------------------------------

class VRAMScaler:
    """VRAM 여유량을 주기적으로 확인해 워커를 자동 추가/제거한다."""

    def __init__(
        self,
        pool: WhisperWorkerPool,
        check_interval: float = 30.0,
        scale_up_threshold_mb: int = 1200,
        scale_down_threshold_mb: int = 400,
        model_size_mb: int = 800,
        max_workers: int = 4,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self._pool = pool
        self._check_interval = check_interval
        self._scale_up_threshold_mb = scale_up_threshold_mb
        self._scale_down_threshold_mb = scale_down_threshold_mb
        self._model_size_mb = model_size_mb
        self._max_workers = max_workers
        self._cooldown_seconds = cooldown_seconds

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_scale_time: float = 0.0

    # -- public API ----------------------------------------------------------

    def start(self) -> None:
        """데몬 스레드를 시작한다."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="vram-scaler",
        )
        self._thread.start()

    def stop(self) -> None:
        """모니터링을 중단한다."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    # -- internals -----------------------------------------------------------

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(self._check_interval)
            if self._stop_event.is_set():
                break

            now = time.monotonic()
            if now - self._last_scale_time < self._cooldown_seconds:
                continue

            available_mb = get_available_vram() // (1024 * 1024)

            if (
                available_mb > self._scale_up_threshold_mb + self._model_size_mb
                and self._pool.n_workers < self._max_workers
            ):
                self._add_worker()
                self._last_scale_time = now
            elif (
                available_mb < self._scale_down_threshold_mb
                and self._pool.n_workers > 1
            ):
                self._remove_worker()
                self._last_scale_time = now

    def _add_worker(self) -> None:
        """풀의 공개 API를 통해 워커를 추가한다 (_pool_lock 준수)."""
        try:
            self._pool.add_worker()
        except Exception:
            pass

    def _remove_worker(self) -> None:
        """풀의 공개 API를 통해 워커를 제거한다 (_pool_lock 준수)."""
        try:
            self._pool.remove_worker()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WhisperServiceDaemon
# ---------------------------------------------------------------------------

class _DaemonRequestHandler(socketserver.StreamRequestHandler):
    """line-delimited JSON 프로토콜 처리기."""

    def handle(self) -> None:
        server: WhisperServiceDaemon = self.server  # type: ignore[assignment]
        for raw_line in self.rfile:
            line = raw_line.strip()
            if not line:
                continue
            if len(line) > _MAX_RECV_BYTES:
                self._send({"id": None, "status": "error", "error": "payload too large"})
                return

            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                self._send({"id": None, "status": "error", "error": "invalid JSON"})
                continue

            req_id = request.get("id")
            method = request.get("method", "")

            # 토큰 인증 (ping 제외)
            if method != "ping" and request.get("token") != server.auth_token:
                self._send({"id": req_id, "status": "error", "error": "unauthorized"})
                continue

            try:
                if method == "ping":
                    self._send({"id": req_id, "status": "ok", "result": "pong"})

                elif method == "status":
                    pool = server.pool
                    self._send({
                        "id": req_id,
                        "status": "ok",
                        "n_workers": pool.n_workers,
                        "ready_workers": pool.ready_workers,
                        "is_ready": pool.is_ready,
                        "is_alive": pool.is_alive,
                    })

                elif method == "shutdown":
                    self._send({"id": req_id, "status": "ok", "result": "shutting_down"})
                    threading.Thread(
                        target=server.stop, daemon=True, name="daemon-shutdown"
                    ).start()
                    return

                elif method == "transcribe":
                    audio_b64 = request.get("audio_b64", "")
                    if len(audio_b64) > _MAX_AUDIO_BYTES * 4 // 3 + 4:
                        self._send({"id": req_id, "status": "error", "error": "audio too large"})
                        continue
                    kwargs = request.get("kwargs", {})
                    audio_bytes = base64.b64decode(audio_b64)
                    if len(audio_bytes) > _MAX_AUDIO_BYTES:
                        self._send({"id": req_id, "status": "error", "error": "audio too large"})
                        continue
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                    segments = server.pool.transcribe(audio_array, **kwargs)
                    self._send({
                        "id": req_id,
                        "status": "ok",
                        "segments": segments,
                    })

                else:
                    self._send({
                        "id": req_id,
                        "status": "error",
                        "error": f"unknown method: {method}",
                    })

            except Exception as exc:
                self._send({
                    "id": req_id,
                    "status": "error",
                    "error": str(exc),
                })

    def _send(self, obj: dict) -> None:
        try:
            data = json.dumps(obj, ensure_ascii=False) + "\n"
            self.wfile.write(data.encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


class WhisperServiceDaemon(socketserver.ThreadingTCPServer):
    """TCP 소켓 기반 Whisper STT 데몬 서버.

    IPC 프로토콜: line-delimited JSON over TCP.
    Request : ``{"id": "uuid", "method": "transcribe", "audio_b64": "...", "kwargs": {...}}``
    Response: ``{"id": "uuid", "status": "ok", "segments": [...]}``
    추가 methods: ``ping``, ``status``, ``shutdown``
    """

    LOCKFILE: Path = (
        Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))) / "sonote" / "run" / "whisper-daemon.lock"
    )

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        pool: WhisperWorkerPool,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.pool = pool
        self.auth_token: str = secrets.token_urlsafe(32)
        self._scaler: VRAMScaler | None = None
        super().__init__((host, port), _DaemonRequestHandler)

    # -- public API ----------------------------------------------------------

    def start(self) -> int:
        """lockfile을 작성하고 할당된 포트 번호를 반환한다."""
        host = str(self.server_address[0])
        port = int(self.server_address[1])
        self._write_lockfile(host, port)
        return port

    def stop(self) -> None:
        """서버를 종료하고 lockfile을 삭제한다."""
        try:
            self.shutdown()
        except Exception:
            pass
        try:
            self.server_close()
        except Exception:
            pass
        if self._scaler is not None:
            self._scaler.stop()
        try:
            self.pool.stop()
        except Exception:
            pass
        self._remove_lockfile()

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        """블로킹 서빙 루프."""
        super().serve_forever(poll_interval=poll_interval)

    # -- lockfile helpers ----------------------------------------------------

    @staticmethod
    def find_running() -> tuple[str, int, str] | None:
        """lockfile을 읽어서 실행 중인 데몬의 (host, port, token) 을 반환한다.

        데몬이 없거나 응답이 없으면 ``None``.
        """
        lockfile = WhisperServiceDaemon.LOCKFILE
        if not lockfile.exists():
            return None
        try:
            text = lockfile.read_text(encoding="utf-8").strip()
            parts = text.split(":")
            host = parts[0]
            port = int(parts[1])
            token = parts[2] if len(parts) > 2 else ""
        except Exception:
            return None

        # 실제 응답 확인 (ping)
        try:
            sock = socket.create_connection((host, port), timeout=2.0)
            sock.settimeout(2.0)
            try:
                req = json.dumps({"id": str(uuid.uuid4()), "method": "ping"}) + "\n"
                sock.sendall(req.encode("utf-8"))
                resp_line = b""
                while not resp_line.endswith(b"\n"):
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    resp_line += chunk
                    if len(resp_line) > 65536:
                        break
                resp = json.loads(resp_line.decode("utf-8"))
                if resp.get("status") == "ok":
                    return host, port, token
            finally:
                sock.close()
        except Exception:
            # 데몬이 죽어 있으면 stale lockfile 삭제
            try:
                lockfile.unlink(missing_ok=True)
            except Exception:
                pass

        return None

    def _write_lockfile(self, host: str, port: int) -> None:
        self.LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
        self.LOCKFILE.write_text(
            f"{host}:{port}:{self.auth_token}", encoding="utf-8",
        )

    def _remove_lockfile(self) -> None:
        try:
            self.LOCKFILE.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WhisperRemoteClient
# ---------------------------------------------------------------------------

class WhisperRemoteClient:
    """WhisperWorkerPool 과 동일한 duck-type 인터페이스로 데몬에 위임한다.

    Drop-in replacement: 코드에서 ``pool`` 자리에 이 클라이언트를 넣으면
    로컬 모델 로드 없이 데몬의 GPU 워커를 공유할 수 있다.
    """

    def __init__(self, host: str, port: int, token: str = "") -> None:
        self._host = host
        self._port = port
        self._token = token

    # -- duck-type properties (WhisperWorkerPool 호환) -----------------------

    @property
    def is_ready(self) -> bool:
        """ping으로 데몬 응답 여부 확인."""
        try:
            resp = self._call("ping")
            return resp.get("status") == "ok"
        except Exception:
            return False

    @property
    def is_alive(self) -> bool:
        return self.is_ready

    @property
    def n_workers(self) -> int:
        """데몬에 status 조회 후 n_workers 반환."""
        try:
            resp = self._call("status")
            return int(resp.get("n_workers", 0))
        except Exception:
            return 0

    @property
    def ready_workers(self) -> int:
        try:
            resp = self._call("status")
            return int(resp.get("ready_workers", 0))
        except Exception:
            return 0

    # -- duck-type methods ---------------------------------------------------

    def start(self) -> None:
        """no-op — 데몬이 이미 실행 중."""

    def wait_ready(self, timeout: float = 10.0) -> None:
        """데몬이 응답할 때까지 대기."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_ready:
                return
            time.sleep(0.3)
        raise RuntimeError("데몬 응답 타임아웃")

    def transcribe(self, audio_chunk: np.ndarray, **kwargs: Any) -> list[dict]:
        """오디오를 base64로 인코딩해 데몬에 전송, 결과 수신."""
        audio_bytes = audio_chunk.astype(np.float32).tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        resp = self._call("transcribe", audio_b64=audio_b64, kwargs=kwargs)

        if resp.get("status") != "ok":
            raise RuntimeError(f"데몬 전사 실패: {resp.get('error', 'unknown')}")
        return resp.get("segments", [])

    def stop(self) -> None:
        """no-op — 데몬은 계속 살려둔다."""

    # -- internal TCP 통신 ---------------------------------------------------

    def _call(
        self,
        method: str,
        timeout: float = 120.0,
        **extra: Any,
    ) -> dict:
        """데몬에 JSON 요청을 보내고 응답을 반환한다."""
        req: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "method": method,
            "token": self._token,
        }
        req.update(extra)

        sock = socket.create_connection((self._host, self._port), timeout=timeout)
        sock.settimeout(timeout)
        try:
            payload = json.dumps(req, ensure_ascii=False) + "\n"
            sock.sendall(payload.encode("utf-8"))

            # 응답 수신 (line-delimited)
            buf = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    break

            line = buf.split(b"\n", 1)[0]
            return json.loads(line.decode("utf-8"))
        finally:
            sock.close()
