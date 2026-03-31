"""sonote 데스크톱 런처 — FastAPI 서버 + Chrome 브라우저 + 시스템 트레이."""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from typing import Any

from ..config import get_config
from ..runtime.context import is_frozen
from ..server import (
    add_extracted_keywords,
    consume_audio_device_switch,
    consume_session_rotate,
    create_app,
    get_audio_device_switch_event,
    get_keywords_snapshot,
    is_paused,
    is_shutdown_requested,
    push_correction_sync,
    push_transcript_sync,
    request_shutdown,
    run_server,
    signal_server_shutdown,
    set_capture_error,
    set_current_audio_device,
    set_postprocess_status,
    set_startup_status,
    set_voice_active,
    toggle_pause_state,
)
from ..tray import MeetingTray, is_available as tray_available
from .browser import _open_app_mode
from .tray_adapter import start_status_poller, start_tray

_BETA_MODEL_ID = "tellang/whisper-medium-ko-ct2"


def _find_free_port() -> int:
    """로컬 루프백에서 사용 가능한 포트를 찾는다."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _wait_for_server_ready(base_url: str, timeout_seconds: float = 10.0) -> None:
    """로컬 서버가 기동될 때까지 readiness를 짧게 poll한다."""
    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    status_url = f"{base_url}/status"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(status_url, timeout=0.5):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(0.05)
    raise RuntimeError(f"FastAPI 서버가 준비되지 않았습니다: {status_url}")


def _fetch_json(url: str, method: str = "GET") -> dict[str, Any]:
    """내장 서버 JSON 엔드포인트를 호출한다."""
    request = urllib.request.Request(url=url, method=method)
    with urllib.request.urlopen(request, timeout=1.0) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    return data if isinstance(data, dict) else {}


def _format_elapsed(seconds: int) -> str:
    """초 단위를 HH:MM:SS 문자열로 변환한다."""
    total = max(int(seconds), 0)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass(slots=True)
class DesktopController:
    """FastAPI + Chrome 브라우저 + pystray 트레이 컨트롤러."""

    host: str = "127.0.0.1"
    port: int = 0
    beta_mode: bool = False
    base_url: str = field(init=False, default="")
    server_thread: threading.Thread | None = field(init=False, default=None)
    tray: MeetingTray | None = field(init=False, default=None)
    stop_event: threading.Event = field(init=False, default_factory=threading.Event)
    _shutdown_complete: threading.Event = field(init=False, default_factory=threading.Event)
    _server_error: BaseException | None = field(init=False, default=None)
    _meeting_thread: threading.Thread | None = field(init=False, default=None)
    _status_thread: threading.Thread | None = field(init=False, default=None)
    _capture_stop_event: threading.Event | None = field(init=False, default=None)
    _shutdown_lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _shutdown_started: bool = field(init=False, default=False)
    _finalize_lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    # 세션 파일 저장 관리자 (캡처 루프 실행 중 유효)
    _writer: Any = field(init=False, default=None)
    # 캡처 시작 시각 (경과시간 계산용)
    _capture_start_time: float = field(init=False, default=0.0)
    # 실시간 LLM 후처리 (STT 교정 + 키워드 추출)
    _polish_pool: Any = field(init=False, default=None)
    _correct_fn: Any = field(init=False, default=None)
    _extract_kw_fn: Any = field(init=False, default=None)
    _use_ollama: bool = field(init=False, default=False)
    _ollama_model: str = field(init=False, default="gemma3:27b")

    def __post_init__(self) -> None:
        if self.port <= 0:
            self.port = _find_free_port()
        self.base_url = f"http://{self.host}:{self.port}"

    def run(self) -> None:
        """데스크톱 앱 전체를 기동한다."""
        self._start_server()

        # exe 환경에서는 서버 기동이 느릴 수 있으므로 타임아웃 확장
        timeout = 30.0 if is_frozen() else 10.0
        try:
            _wait_for_server_ready(self.base_url, timeout_seconds=timeout)
        except RuntimeError:
            # 서버 스레드에서 발생한 에러 확인
            if self._server_error is not None:
                raise RuntimeError(f"서버 시작 실패: {self._server_error}") from self._server_error
            raise

        # 브라우저를 먼저 열고, 모델 로드는 백그라운드에서 진행
        self._open_browser()

        # 미팅 자동시작: WhisperWorkerPool + 마이크 초기화 (백그라운드 스레드)
        self._meeting_thread = threading.Thread(
            target=self._auto_start_meeting,
            daemon=True,
            name="sonote-auto-start",
        )
        self._meeting_thread.start()

        if tray_available():
            start_tray(self)
            self._status_thread = start_status_poller(
                self,
                fetch_json=_fetch_json,
                format_elapsed=_format_elapsed,
            )
            # 트레이가 메인 루프 역할 — 종료 이벤트 대기
            self.stop_event.wait()
            self._shutdown_complete.wait(timeout=3.5)
        else:
            # 트레이 없으면 콘솔 대기
            print(f"sonote 실행 중: {self.base_url}")
            print("종료하려면 Ctrl+C를 누르세요.")
            try:
                self.stop_event.wait()
                self._shutdown_complete.wait(timeout=3.5)
            except KeyboardInterrupt:
                self.shutdown()

    def _run_server_safe(self, **kwargs: Any) -> None:
        """서버 실행 래퍼 — 예외를 캡처한다."""
        try:
            run_server(**kwargs)
        except Exception as exc:
            self._server_error = exc

    def _start_server(self) -> None:
        """FastAPI 서버를 백그라운드 스레드로 기동한다."""
        application = create_app(beta_mode=self.beta_mode)
        self.server_thread = threading.Thread(
            target=self._run_server_safe,
            kwargs={
                "host": self.host,
                "port": self.port,
                "application": application,
                "beta_mode": self.beta_mode,
            },
            daemon=True,
            name="sonote-fastapi",
        )
        self.server_thread.start()

    def _auto_start_meeting(self) -> None:
        """서버 ready 후 WhisperWorkerPool + 마이크를 자동 초기화하고 캡처→전사→푸시 루프를 실행한다."""
        worker: Any | None = None
        try:
            from ..audio_capture import capture_audio, find_builtin_mic
            from ..runtime_env import detect_device
            from ..whisper_worker import WhisperWorkerPool

            # 마이크 자동 감지
            set_startup_status("device", "가속기 감지 중...")
            mic_device = find_builtin_mic()
            set_current_audio_device(mic_device)

            # 디바이스 감지
            device, compute_type = detect_device()

            # Whisper 워커 풀 시작 (비동기 — 모델 로드는 백그라운드)
            config = get_config()
            default_model = _BETA_MODEL_ID if self.beta_mode else "tellang/whisper-medium-ko-ct2"
            model_id = config.get("model", default_model)
            language = config.get("language", "ko")
            chunk_seconds: float = float(config.get("chunk", 5.0))

            worker = WhisperWorkerPool(
                model_id=model_id,
                language=language,
                device=device,
                compute_type=compute_type,
            )
            set_startup_status(
                "loading_asr",
                f"STT 모델 로드 중 ({device}/{compute_type}, BoN×1→{worker.n_workers})...",
            )
            worker.start()

            # 화자 분리기 (선택)
            diarizer = None
            if config.get("diarize", True):
                try:
                    from ..diarize import SpeakerDiarizer

                    if SpeakerDiarizer.is_available():
                        set_startup_status("loading_diarizer", "화자 분리 모델 로드 중...")
                        diarizer = SpeakerDiarizer(
                            hf_token=os.environ.get("HF_TOKEN"),
                            device=device,
                        )
                except Exception:
                    pass  # 화자 분리 실패 시 무시하고 진행

            # diarizer와 profiles_path를 서버에 연결
            from ..server import set_diarizer
            from ..paths import profiles_db_path

            profiles_path = str(profiles_db_path())
            set_diarizer(diarizer, profiles_path)

            # 모델 로드 완료까지 블로킹 대기 (최대 120초)
            worker.wait_ready(timeout=120.0)
            set_startup_status("ready", "녹음 준비 완료", ready=True)
        except Exception:
            # 자동시작 실패 시에도 데스크톱 앱은 정상 기동
            set_startup_status("ready", "녹음 준비 완료 (자동시작 실패)", ready=True)
            return

        # ── MeetingWriter 초기화 — 세션 파일을 output/meetings/ 에 생성 ──
        from ..meeting_writer import MeetingWriter

        writer = MeetingWriter()
        writer.write_header()
        self._writer = writer
        self._capture_start_time = time.time()

        # ── 실시간 LLM 후처리 초기화 (STT 교정 + 키워드 추출) ──────────────
        self._init_polish(writer)

        # ── 캡처 → 전사 → 푸시 루프 ──────────────────────────────────────
        # 모델 준비 완료 이후 실행. 앱 종료(stop_event) 또는 서버 shutdown 요청까지 계속 반복.
        try:
            self._run_capture_loop(
                worker=worker,
                diarizer=diarizer,
                language=language,
                chunk_seconds=chunk_seconds,
                capture_audio=capture_audio,
            )
        except Exception as exc:
            import traceback
            print(f"[캡처 루프] 예외 발생: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        finally:
            if worker is not None:
                try:
                    worker.stop()
                except Exception as exc:
                    print(f"[desktop][worker] 종료 실패: {exc}", file=sys.stderr)
            # 캡처 루프 종료 시 세션 파일 마무리
            self._finalize_writer(run_postprocess=False)
            # 캡처 루프 종료 → 앱 전체 종료 트리거
            if not self.stop_event.is_set():
                self.shutdown()

    def _run_capture_loop(
        self,
        *,
        worker: Any,
        diarizer: Any,
        language: str,
        chunk_seconds: float,
        capture_audio: Any,
    ) -> None:
        """오디오 캡처→전사→서버 푸시 루프를 공통 파이프라인으로 실행한다."""
        from ..meeting import PipelineAdapter, PipelineContext, run_capture_loop

        capture_switch_event = get_audio_device_switch_event()
        from ..server import get_current_audio_device

        def _on_transcript(payload: dict[str, Any]) -> None:
            push_transcript_sync(payload["speaker"], payload["text"], payload["timestamp"], confidence=payload.get("confidence"))
            if self._writer is not None:
                self._writer.append_segment(
                    payload["speaker"],
                    payload["text"],
                    payload["timestamp"],
                    metadata={
                        "start": payload.get("start"),
                        "end": payload.get("end"),
                        "feedback_text": payload.get("feedback_text"),
                        "confidence": payload.get("confidence"),
                    },
                )

        def _submit_correction(batch: list[str], idx: int) -> Any:
            if self._polish_pool is None or self._correct_fn is None:
                return None
            return self._polish_pool.submit(self._correct_fn, batch, idx, 120)

        def _submit_keyword(text: str) -> None:
            if self._polish_pool is None or self._extract_kw_fn is None:
                return

            def _do_kw_extract(_text: str = text, _fn: Any = self._extract_kw_fn) -> None:
                try:
                    kws = _fn(_text)
                    if kws:
                        payload = add_extracted_keywords(list(kws))
                        promoted = payload.get("promoted") or []
                        extracted = payload.get("extracted") or []
                        print(
                            f"[키워드 추출] promoted={len(promoted)} extracted={len(extracted)}",
                            file=sys.stderr,
                        )
                except Exception as exc:
                    print(f"[키워드 추출] 실패: {exc}", file=sys.stderr)

            self._polish_pool.submit(_do_kw_extract)

        def _preprocess(chunk: Any) -> Any:
            import numpy as _np

            rms = float(_np.sqrt(_np.mean(chunk ** 2)))
            set_voice_active(rms > 0.005)
            return chunk

        def _on_capture_error(
            exc: Exception,
            active_device: int | None,
            error_count: int,
        ) -> tuple[int | None, bool]:
            err_msg = f"device={active_device}: {exc}"
            set_capture_error(err_msg, error_count)
            if error_count <= 5 or error_count % 30 == 0:
                import traceback

                print(
                    f"[캡처 루프] 오류 #{error_count} {err_msg}",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
            if error_count >= 3 and active_device is not None:
                print(
                    f"[캡처 루프] 디바이스 {active_device} 반복 실패 → 기본 디바이스로 폴백",
                    file=sys.stderr,
                )
                set_current_audio_device(None)
                time.sleep(1.0)
                return None, True
            time.sleep(1.0)
            return active_device, True

        context = PipelineContext(
            worker=worker,
            diarizer=diarizer,
            language=language,
            chunk_seconds=chunk_seconds,
            on_transcript=_on_transcript,
            on_correction=push_correction_sync,
            stop_event=self.stop_event,
        )
        adapter = PipelineAdapter(
            capture_audio=capture_audio,
            is_paused=is_paused,
            is_shutdown_requested=is_shutdown_requested,
            consume_audio_device_switch=consume_audio_device_switch,
            set_current_audio_device=set_current_audio_device,
            capture_stop_event=capture_switch_event,
            consume_session_rotate=consume_session_rotate,
            on_session_rotate=self._rotate_session,
            reset_runtime_on_rotate=False,
            on_pause=lambda: set_voice_active(False),
            preprocess_chunk=_preprocess,
            submit_correction_batch=_submit_correction,
            submit_keyword_job=_submit_keyword,
            recent_text_limit=50,
            on_capture_error=_on_capture_error,
            diarize_error_label="desktop",
        )
        self._capture_stop_event = capture_switch_event
        try:
            run_capture_loop(
                context,
                adapter,
                initial_device=get_current_audio_device(),
            )
        finally:
            self._capture_stop_event = None

    def _rotate_session(self) -> None:
        """현재 세션을 마무리하고 후처리를 백그라운드로 시작한 뒤 새 세션 writer를 준비한다."""
        from ..meeting_writer import MeetingWriter

        old_writer = self._writer
        if old_writer is not None:
            elapsed_sec = int(time.time() - self._capture_start_time)
            elapsed_str = _format_elapsed(elapsed_sec)
            seg_count = len(old_writer._segments)
            old_writer.set_keywords(get_keywords_snapshot())
            old_writer.write_footer(
                duration=elapsed_str,
                segment_count=seg_count,
                speaker_count=len(old_writer._speakers),
            )
            old_writer.close()

            # 세그먼트가 있으면 백그라운드 후처리 시작
            if seg_count > 0:
                threading.Thread(
                    target=self._postprocess_session,
                    args=(old_writer.output_path, seg_count),
                    daemon=True,
                    name="sonote-postprocess",
                ).start()

        new_writer = MeetingWriter()
        new_writer.write_header()
        self._writer = new_writer
        self._capture_start_time = time.time()

    def _postprocess_session(self, output_path: Any, seg_count: int) -> None:
        """백그라운드에서 세션 후처리 실행 (STT 교정 + 요약)."""
        from ..polish import is_codex_available, is_gemini_available, is_ollama_available

        if not (
            is_codex_available()
            or is_gemini_available()
            or is_ollama_available(self._ollama_model)
        ):
            set_postprocess_status("done", 100)
            return

        try:
            from ..polish import polish_meeting

            set_postprocess_status("stt", 0)
            print(f"[후처리] 세션 후처리 시작: {output_path}", file=sys.stderr)

            def _progress(phase: str, pct: float) -> None:
                set_postprocess_status(phase, pct)

            results = polish_meeting(
                output_path,
                segment_count=seg_count,
                use_ollama=self._use_ollama,
                ollama_model=self._ollama_model if self._use_ollama else None,
                progress_callback=_progress,
            )
            corrected = results.get("corrected", False)
            summarized = results.get("summarized", False)
            print(
                f"[후처리] 완료 — 교정: {'완료' if corrected else '건너뜀'}, "
                f"요약: {'완료' if summarized else '건너뜀'}",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"[후처리] 실패: {exc}", file=sys.stderr)
        finally:
            set_postprocess_status("done", 100)

    def _open_browser(self, url: str | None = None) -> None:
        """Chrome/Edge 앱 모드로 뷰어를 연다. 실패 시 기본 브라우저 폴백."""
        target = url or self.base_url
        if not _open_app_mode(target):
            webbrowser.open(target)

    def _init_polish(self, writer: Any) -> None:
        """설정 기반으로 실시간 LLM 후처리를 초기화한다 (STT 교정 + 키워드 추출).

        Codex/Ollama 미설치 시에는 graceful skip — 에러 없이 진행.
        """
        from ..polish import is_codex_available, is_ollama_available

        config = get_config()
        engine = config.get("polish_engine", "auto")
        self._ollama_model = config.get("ollama_model", "gemma3:27b")

        # 엔진 선택: auto → Codex 우선, Ollama 폴백
        _has_backend = False
        if engine == "ollama":
            if is_ollama_available(self._ollama_model):
                self._use_ollama = True
                _has_backend = True
            else:
                print(
                    f"[후처리] Ollama 미가용 ({self._ollama_model}) — 실시간 교정 비활성",
                    file=sys.stderr,
                )
        elif engine in ("auto", "codex"):
            if is_codex_available():
                self._use_ollama = False
                _has_backend = True
            elif is_ollama_available(self._ollama_model):
                self._use_ollama = True
                _has_backend = True
                print(
                    f"[후처리] Codex 미설치 → Ollama ({self._ollama_model}) 폴백",
                    file=sys.stderr,
                )
            else:
                print(
                    "[후처리] Codex/Ollama 모두 미가용 — 실시간 교정 비활성",
                    file=sys.stderr,
                )
        # engine == "gemini" 또는 기타: 실시간 교정 미지원 (요약만)

        if not _has_backend:
            return

        from concurrent.futures import ThreadPoolExecutor as _TPE

        self._polish_pool = _TPE(max_workers=2, thread_name_prefix="live-polish")

        if self._use_ollama:
            from ..polish import _correct_batch_ollama, extract_keywords_with_ollama

            _model = self._ollama_model

            def _correct_fn_ollama(batch: list[str], idx: int, t: int = 120) -> tuple:
                return _correct_batch_ollama(batch, idx, _model, t)

            def _extract_kw_fn_ollama(text: str) -> list[str]:
                return extract_keywords_with_ollama(text, model=_model)

            self._correct_fn = _correct_fn_ollama
            self._extract_kw_fn = _extract_kw_fn_ollama
        else:
            from ..polish import _correct_batch, extract_keywords_with_codex

            _work_dir = writer.output_path.parent

            def _correct_fn_codex(batch: list[str], idx: int, t: int = 120) -> tuple:
                return _correct_batch(batch, idx, _work_dir, t)

            def _extract_kw_fn_codex(text: str) -> list[str]:
                return extract_keywords_with_codex(text, _work_dir)

            self._correct_fn = _correct_fn_codex
            self._extract_kw_fn = _extract_kw_fn_codex

        print(
            f"[후처리] 실시간 LLM 교정 활성 "
            f"({'Ollama ' + self._ollama_model if self._use_ollama else 'Codex'})",
            file=sys.stderr,
        )

    def _finalize_writer(self, *, run_postprocess: bool | None = None) -> None:
        """현재 writer를 footer/flush/close까지 완료한다."""
        with self._finalize_lock:
            writer = self._writer
            if writer is None:
                return
            self._writer = None

            pool = self._polish_pool
            self._polish_pool = None

        # 진행 중인 교정 태스크 취소 — 종료 시 새 태스크는 더 이상 불필요
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

        try:
            elapsed_sec = int(time.time() - self._capture_start_time) if self._capture_start_time > 0 else 0
            elapsed_str = _format_elapsed(elapsed_sec)
            seg_count = len(writer._segments)
            writer.set_keywords(get_keywords_snapshot())
            writer.write_footer(
                duration=elapsed_str,
                segment_count=seg_count,
                speaker_count=len(writer._speakers),
            )
            writer.close()
        except Exception as exc:
            print(f"[desktop][writer] 세션 파일 저장 실패: {exc}", file=sys.stderr)
            return

        if run_postprocess is None:
            run_postprocess = not self.stop_event.is_set()
        if not run_postprocess:
            set_postprocess_status("done", 100)
            return

        # 회의 후 LLM 후처리 (STT 교정 + 요약) — 세그먼트가 있을 때만
        if seg_count <= 0:
            set_postprocess_status("done", 100)
            return

        from ..polish import is_codex_available, is_gemini_available, is_ollama_available

        if not (
            is_codex_available()
            or is_gemini_available()
            or is_ollama_available(self._ollama_model)
        ):
            set_postprocess_status("done", 100)
            return

        try:
            from ..polish import polish_meeting

            set_postprocess_status("stt", 0)
            print("[후처리] 회의 후 LLM 후처리 시작...", file=sys.stderr)

            def _progress(phase: str, pct: float) -> None:
                set_postprocess_status(phase, pct)

            results = polish_meeting(
                writer.output_path,
                segment_count=seg_count,
                use_ollama=self._use_ollama,
                ollama_model=self._ollama_model if self._use_ollama else None,
                progress_callback=_progress,
            )
            corrected = results.get("corrected", False)
            summarized = results.get("summarized", False)
            print(
                f"[후처리] 완료 — 교정: {'완료' if corrected else '건너뜀'}, "
                f"요약: {'완료' if summarized else '건너뜀'}",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"[후처리] 실패: {exc}", file=sys.stderr)
        finally:
            set_postprocess_status("done", 100)

    def toggle_recording(self) -> None:
        """트레이의 녹음 시작/중지 메뉴를 일시정지 토글에 매핑한다."""
        state = toggle_pause_state()
        if self.tray is not None:
            self.tray.update_status(paused=bool(state["paused"]), recording=True)

    def shutdown(self) -> None:
        """트레이 종료 메뉴에서 전체 앱을 종료한다."""
        with self._shutdown_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True

        start = time.monotonic()
        deadline = start + 3.0

        self.stop_event.set()
        capture_stop_event = self._capture_stop_event
        if capture_stop_event is not None:
            capture_stop_event.set()
        request_shutdown()

        # 1) 캡처 루프 종료 대기 (최대 3초)
        self._join_thread_until(self._meeting_thread, deadline=deadline)

        # 2) writer footer/flush/close 보장 (후처리는 종료 경로에서 생략)
        if self._meeting_thread is None or not self._meeting_thread.is_alive():
            self._finalize_writer(run_postprocess=False)

        # 3) 트레이 종료
        if self.tray is not None:
            try:
                self.tray.stop()
            except Exception:
                pass

        # 4) uvicorn 서버 shutdown signal
        signal_server_shutdown()

        # 5) 데몬 스레드 정리 (최대 2초, 전체 3초 데드라인 우선)
        daemon_join_budget = min(2.0, max(0.0, deadline - time.monotonic()))
        self._join_daemon_threads(max_wait=daemon_join_budget)
        self._shutdown_complete.set()

        # 6) 3초 내 미종료 시에만 최후 수단 강제 종료
        if time.monotonic() >= deadline and self._has_alive_background_threads():
            self._last_resort_exit()

    def _join_thread_until(self, thread: threading.Thread | None, *, deadline: float) -> None:
        if thread is None:
            return
        if thread is threading.current_thread():
            return
        if not thread.is_alive():
            return
        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0.0:
            return
        thread.join(timeout=remaining)

    def _managed_threads(self) -> list[threading.Thread]:
        threads: list[threading.Thread] = []

        for thread in (self._meeting_thread, self._status_thread, self.server_thread):
            if thread is not None:
                threads.append(thread)

        tray_thread = getattr(self.tray, "_thread", None)
        if isinstance(tray_thread, threading.Thread):
            threads.append(tray_thread)

        for thread in threading.enumerate():
            if thread.daemon and thread.name.startswith("sonote-"):
                threads.append(thread)

        deduped: list[threading.Thread] = []
        seen: set[int] = set()
        for thread in threads:
            identity = id(thread)
            if identity in seen:
                continue
            seen.add(identity)
            deduped.append(thread)
        return deduped

    def _join_daemon_threads(self, *, max_wait: float) -> None:
        join_deadline = time.monotonic() + max(0.0, max_wait)
        current = threading.current_thread()
        for thread in self._managed_threads():
            if thread is current:
                continue
            if not thread.daemon:
                continue
            if not thread.is_alive():
                continue
            remaining = join_deadline - time.monotonic()
            if remaining <= 0.0:
                break
            thread.join(timeout=remaining)

    def _has_alive_background_threads(self) -> bool:
        current = threading.current_thread()
        for thread in self._managed_threads():
            if thread is current:
                continue
            if thread.is_alive():
                return True
        return False

    @staticmethod
    def _last_resort_exit() -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(0)

