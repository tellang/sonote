"""sonote 데스크톱 런처 — FastAPI 서버 + Chrome 브라우저 + 시스템 트레이."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import get_config
from .server import (
    add_extracted_keywords,
    consume_audio_device_switch,
    consume_session_rotate,
    create_app,
    get_audio_device_switch_event,
    is_paused,
    is_shutdown_requested,
    push_correction_sync,
    push_transcript_sync,
    request_shutdown,
    run_server,
    set_capture_error,
    set_current_audio_device,
    set_voice_active,
    set_postprocess_status,
    set_startup_status,
    toggle_pause_state,
)
from .tray import MeetingTray, is_available as tray_available

_BETA_ENV_KEY = "SONOTE_BETA"
_BETA_MODEL_ID = "tellang/whisper-large-v3-turbo-ko"


def _resolve_beta_mode(beta_mode: bool = False) -> bool:
    return bool(beta_mode or os.getenv(_BETA_ENV_KEY) == "1")


def _configure_beta_mode(beta_mode: bool) -> None:
    if beta_mode:
        os.environ[_BETA_ENV_KEY] = "1"
    try:
        from .postprocess import set_beta_mode

        set_beta_mode(beta_mode)
    except Exception:
        pass


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


def _find_app_mode_browser() -> str | None:
    """Chrome 또는 Edge 실행파일 경로를 탐색한다. Edge 우선."""
    if sys.platform != "win32":
        return None

    # Edge 우선, Chrome 차선
    candidates = [
        ("Microsoft\\Edge\\Application\\msedge.exe",),
        ("Google\\Chrome\\Application\\chrome.exe",),
    ]
    search_roots = []
    for env_var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        val = os.environ.get(env_var)
        if val:
            search_roots.append(Path(val))

    for rel_parts in candidates:
        for root in search_roots:
            full = root / rel_parts[0]
            if full.is_file():
                return str(full)

    # PATH에서 탐색
    for name in ("msedge", "chrome"):
        found = shutil.which(name)
        if found:
            return found

    return None


def _open_app_mode(url: str) -> bool:
    """--app=URL 플래그로 Chrome/Edge를 앱 모드로 실행한다. 성공 시 True."""
    browser_path = _find_app_mode_browser()
    if browser_path is None:
        return False
    try:
        subprocess.Popen(
            [browser_path, f"--app={url}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except OSError:
        return False


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
    _server_error: BaseException | None = field(init=False, default=None)
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
        import sys

        self._start_server()

        # exe 환경에서는 서버 기동이 느릴 수 있으므로 타임아웃 확장
        timeout = 30.0 if getattr(sys, "frozen", False) else 10.0
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
        threading.Thread(
            target=self._auto_start_meeting,
            daemon=True,
            name="sonote-auto-start",
        ).start()

        if tray_available():
            self._start_tray()
            self._start_status_poller()
            # 트레이가 메인 루프 역할 — 종료 이벤트 대기
            self.stop_event.wait()
        else:
            # 트레이 없으면 콘솔 대기
            print(f"sonote 실행 중: {self.base_url}")
            print("종료하려면 Ctrl+C를 누르세요.")
            try:
                self.stop_event.wait()
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
        try:
            from .audio_capture import capture_audio, find_builtin_mic
            from .runtime_env import detect_device
            from .whisper_worker import WhisperWorkerPool
            from .postprocess import (
                Segment,
                postprocess,
                is_valid_segment,
                is_hallucination,
                is_looping,
                normalize_feedback_text,
                remove_overlap,
            )

            # 마이크 자동 감지
            set_startup_status("device", "가속기 감지 중...")
            mic_device = find_builtin_mic()
            set_current_audio_device(mic_device)

            # 디바이스 감지
            device, compute_type = detect_device()

            # Whisper 워커 풀 시작 (비동기 — 모델 로드는 백그라운드)
            config = get_config()
            default_model = _BETA_MODEL_ID if self.beta_mode else "large-v3-turbo"
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
                    from .diarize import SpeakerDiarizer

                    if SpeakerDiarizer.is_available():
                        set_startup_status("loading_diarizer", "화자 분리 모델 로드 중...")
                        diarizer = SpeakerDiarizer(
                            hf_token=os.environ.get("HF_TOKEN"),
                            device=device,
                        )
                        if device == "cuda":
                            from .transcribe import _register_cuda_exit_guard

                            _register_cuda_exit_guard()
                except Exception:
                    pass  # 화자 분리 실패 시 무시하고 진행

            # diarizer와 profiles_path를 서버에 연결
            from .server import set_diarizer
            from .paths import profiles_db_path

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
        from .meeting_writer import MeetingWriter

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
                is_valid_segment=is_valid_segment,
                is_hallucination=is_hallucination,
                is_looping=is_looping,
                normalize_feedback_text=normalize_feedback_text,
                remove_overlap=remove_overlap,
                postprocess=postprocess,
                Segment=Segment,
            )
        except Exception as exc:
            import traceback
            print(f"[캡처 루프] 예외 발생: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        finally:
            # 캡처 루프 종료 시 세션 파일 마무리
            self._finalize_writer()
            # 캡처 루프 종료 → 앱 전체 종료 트리거
            self.shutdown()

    def _run_capture_loop(
        self,
        *,
        worker: Any,
        diarizer: Any,
        language: str,
        chunk_seconds: float,
        capture_audio: Any,
        is_valid_segment: Any,
        is_hallucination: Any,
        is_looping: Any,
        normalize_feedback_text: Any,
        remove_overlap: Any,
        postprocess: Any,
        Segment: Any,
    ) -> None:
        """오디오 캡처→전사→서버 푸시 루프. _auto_start_meeting에서 호출된다."""
        # 디바이스 전환 이벤트 (웹 API에서 트리거)
        capture_switch_event = get_audio_device_switch_event()
        # 서버에 설정된 현재 디바이스를 초기값으로 사용
        from .server import _current_audio_device
        active_device = _current_audio_device
        _capture_error_count = 0
        _no_switch = object()
        prev_chunk_text = ""
        _diarize_warned = False

        # 실시간 후처리 상태 (STT 교정 + 키워드 추출)
        _segment_count = 0                      # 누적 세그먼트 수 (배치 트리거용)
        _uncorrected_buffer: list[str] = []     # 아직 교정 미제출 세그먼트 라인
        _correction_futures: list[tuple] = []   # (future, batch) 목록
        _keyword_extract_at = 0                 # 마지막 키워드 추출 시점 세그먼트 수
        _recent_texts: list[str] = []           # 키워드 추출용 최근 텍스트 (최대 50)

        while not self.stop_event.is_set():
            # 디바이스 전환 처리: consume 후 active_device 갱신
            has_switch, next_device = consume_audio_device_switch()
            if has_switch and next_device is not None:
                active_device = next_device
                set_current_audio_device(active_device)

            # 세션 회전 처리: POST /api/sessions/new 호출 시 현재 writer를 마무리하고 후처리 시작
            if consume_session_rotate():
                self._rotate_session()

            requested_device = _no_switch
            try:
                for audio_chunk in capture_audio(
                    chunk_seconds=chunk_seconds,
                    device=active_device,
                    stop_event=capture_switch_event,
                    on_stream_started=lambda d=active_device: set_current_audio_device(d),
                ):
                    # 종료 체크
                    if self.stop_event.is_set() or is_shutdown_requested():
                        return

                    # 디바이스 전환 요청 확인
                    has_switch, next_device = consume_audio_device_switch()
                    if has_switch and next_device != active_device:
                        requested_device = next_device
                        break  # capture_audio 루프 탈출 → 외부 while로 재시작

                    # 세션 회전 처리 (캡처 중에도 즉시 감지)
                    if consume_session_rotate():
                        self._rotate_session()

                    # 일시정지 중이면 캡처는 계속하되 전사 건너뜀
                    if is_paused():
                        set_voice_active(False)
                        continue

                    # 음성 감지 (RMS 에너지 기반)
                    import numpy as _np
                    _rms = float(_np.sqrt(_np.mean(audio_chunk ** 2)))
                    set_voice_active(_rms > 0.005)

                    # STT 전사
                    stt_segments = worker.transcribe(
                        audio_chunk,
                        language=language,
                        beam_size=5,
                        temperature=(0.0, 0.2, 0.4, 0.6),
                        vad_filter=True,
                        vad_parameters={
                            "threshold": 0.45,
                            "min_speech_duration_ms": 250,
                            "min_silence_duration_ms": 500,
                            "speech_pad_ms": 400,
                        },
                        hallucination_silence_threshold=2.0,
                        compression_ratio_threshold=2.4,
                        no_speech_threshold=0.45,
                        log_prob_threshold=-1.0,
                        condition_on_previous_text=True,
                        word_timestamps=True,
                    )

                    # 종료 재확인 (transcribe 블로킹 동안 요청됐을 수 있음)
                    if self.stop_event.is_set() or is_shutdown_requested():
                        return

                    # 화자 분리
                    speaker_segments: list[dict] = []
                    if diarizer is not None:
                        try:
                            speaker_segments = diarizer.identify_speakers_in_chunk(audio_chunk)
                        except Exception as exc:
                            if not _diarize_warned:
                                print(f"[desktop][화자 분리] 세그먼테이션 실패: {exc}", file=sys.stderr)
                                _diarize_warned = True

                    # 세그먼트 필터링 및 화자 매핑
                    timestamp = time.strftime("%H:%M:%S")
                    raw_segments: list[Any] = []
                    for seg in stt_segments:
                        if not is_valid_segment(seg):
                            continue
                        raw_text = (seg.get("text") or "").strip()
                        if not raw_text:
                            continue
                        if is_hallucination(raw_text):
                            continue
                        if is_looping(raw_text):
                            continue
                        text = raw_text
                        if prev_chunk_text:
                            text = remove_overlap(prev_chunk_text, text)
                            if not text:
                                continue

                        # 화자 결정
                        speaker = "화자"
                        if speaker_segments:
                            # 겹치는 구간이 가장 긴 화자 선택
                            seg_start = float(seg.get("start", 0.0))
                            seg_end = float(seg.get("end", 0.0))
                            best_overlap = 0.0
                            for spk_seg in speaker_segments:
                                overlap = max(
                                    0.0,
                                    min(seg_end, spk_seg["end"]) - max(seg_start, spk_seg["start"]),
                                )
                                if overlap > best_overlap:
                                    best_overlap = overlap
                                    speaker = spk_seg["speaker"]
                        elif diarizer is not None:
                            try:
                                speaker = diarizer.identify_speaker(audio_chunk)
                            except Exception:
                                speaker = "?"

                        raw_segments.append(
                            Segment(
                                speaker=speaker,
                                text=text,
                                start=float(seg.get("start", 0.0)),
                                end=float(seg.get("end", 0.0)),
                            )
                        )

                    if not raw_segments:
                        continue

                    # 후처리 (반복 병합 등)
                    processed = postprocess(raw_segments)

                    for p in processed:
                        stripped = normalize_feedback_text(p.text)
                        if not stripped:
                            continue
                        # 서버 SSE 큐에 push (스레드 안전 동기 래퍼)
                        push_transcript_sync(p.speaker, p.text, timestamp)
                        # 디스크에도 세그먼트 기록
                        if self._writer is not None:
                            self._writer.append_segment(p.speaker, p.text, timestamp)

                        # 실시간 후처리용 버퍼에 누적
                        _segment_count += 1
                        if self._polish_pool is not None:
                            _uncorrected_buffer.append(
                                f"- [{timestamp}] [{p.speaker}] {p.text}"
                            )
                        # 키워드 추출용 최근 텍스트 유지 (최대 50개)
                        _recent_texts.append(stripped)
                        if len(_recent_texts) > 50:
                            _recent_texts.pop(0)

                    # 실시간 STT 교정 — 10세그먼트 누적 시 배치 제출
                    if (
                        self._polish_pool is not None
                        and self._correct_fn is not None
                        and len(_uncorrected_buffer) >= 10
                    ):
                        _batch = _uncorrected_buffer[:]
                        _batch_start_idx = _segment_count - len(_batch)
                        _uncorrected_buffer.clear()
                        _fut = self._polish_pool.submit(
                            self._correct_fn, _batch, len(_correction_futures), 120,
                        )

                        def _on_correction_done(
                            future,
                            batch=_batch,
                            start_idx=_batch_start_idx,
                        ) -> None:
                            """교정 완료 콜백 — SSE로 교정 결과 전송."""
                            try:
                                _, ok, corrected = future.result(timeout=0)
                                if ok:
                                    corrections = []
                                    for i, (orig, corr) in enumerate(
                                        zip(batch, corrected)
                                    ):
                                        if orig != corr:
                                            corrections.append(
                                                {
                                                    "index": start_idx + i,
                                                    "original": orig,
                                                    "corrected": corr,
                                                }
                                            )
                                    if corrections:
                                        push_correction_sync(corrections)
                            except Exception:
                                pass

                        _fut.add_done_callback(_on_correction_done)
                        _correction_futures.append((_fut, _batch))

                    # 실시간 도메인 키워드 추출 — 10세그먼트마다 실행
                    if (
                        self._polish_pool is not None
                        and self._extract_kw_fn is not None
                        and _segment_count >= _keyword_extract_at + 10
                    ):
                        _keyword_extract_at = _segment_count
                        _kw_text = " ".join(_recent_texts[-10:])
                        if _kw_text.strip():

                            def _do_kw_extract(
                                _text: str = _kw_text,
                                _fn: Any = self._extract_kw_fn,
                            ) -> None:
                                """키워드 추출 후 서버에 등록."""
                                try:
                                    kws = _fn(_text)
                                    if kws:
                                        payload = add_extracted_keywords(list(kws))
                                        promoted = payload.get("promoted") or []
                                        extracted = payload.get("extracted") or []
                                        print(
                                            f"[키워드 추출] promoted={len(promoted)} "
                                            f"extracted={len(extracted)}",
                                            file=sys.stderr,
                                        )
                                except Exception as e:
                                    print(
                                        f"[키워드 추출] 실패: {e}", file=sys.stderr
                                    )

                            self._polish_pool.submit(_do_kw_extract)

                    prev_chunk_text = " ".join(
                        normalize_feedback_text(p.text) for p in processed
                    )

            except Exception as exc:
                _capture_error_count += 1
                err_msg = f"device={active_device}: {exc}"
                set_capture_error(err_msg, _capture_error_count)
                if _capture_error_count <= 5 or _capture_error_count % 30 == 0:
                    import traceback
                    print(
                        f"[캡처 루프] 오류 #{_capture_error_count} {err_msg}",
                        file=sys.stderr,
                    )
                    traceback.print_exc(file=sys.stderr)
                # 3회 연속 실패 시 기본 디바이스로 폴백
                if _capture_error_count >= 3 and active_device is not None:
                    print(
                        f"[캡처 루프] 디바이스 {active_device} 반복 실패 → 기본 디바이스로 폴백",
                        file=sys.stderr,
                    )
                    active_device = None
                    set_current_audio_device(None)
                    _capture_error_count = 0
                time.sleep(1.0)
                continue

            # 디바이스 전환이 요청된 경우 active_device 갱신 후 재시작
            if requested_device is not _no_switch and requested_device is not None:
                active_device = requested_device
                set_current_audio_device(active_device)

    def _rotate_session(self) -> None:
        """현재 세션을 마무리하고 후처리를 백그라운드로 시작한 뒤 새 세션 writer를 준비한다."""
        from .meeting_writer import MeetingWriter

        old_writer = self._writer
        if old_writer is not None:
            elapsed_sec = int(time.time() - self._capture_start_time)
            elapsed_str = _format_elapsed(elapsed_sec)
            seg_count = len(old_writer._segments)
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
        from .polish import is_codex_available, is_gemini_available, is_ollama_available

        if not (
            is_codex_available()
            or is_gemini_available()
            or is_ollama_available(self._ollama_model)
        ):
            set_postprocess_status("done", 100)
            return

        try:
            from .polish import polish_meeting

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

    def _start_tray(self) -> None:
        """트레이 아이콘을 시작한다."""
        if self.tray is not None:
            return

        self.tray = MeetingTray(
            port=self.port,
            on_shutdown=self.shutdown,
            on_open_viewer=self._open_browser,
            on_open_settings=lambda: self._open_browser(f"{self.base_url}/settings"),
            on_toggle_recording=self.toggle_recording,
            get_toggle_label=lambda paused: "녹음 시작" if paused else "녹음 중지",
        )
        self.tray.start()
        self.tray.update_status(paused=is_paused(), recording=True)

    def _start_status_poller(self) -> None:
        """서버 상태를 읽어 트레이 툴팁/아이콘에 반영한다."""
        thread = threading.Thread(
            target=self._status_poller_loop,
            daemon=True,
            name="sonote-desktop-status",
        )
        thread.start()

    def _status_poller_loop(self) -> None:
        """주기적으로 서버 상태를 조회한다."""
        while not self.stop_event.wait(1.0):
            if self.tray is None:
                continue
            try:
                status = _fetch_json(f"{self.base_url}/status")
            except Exception:
                continue

            speakers = status.get("speakers") or []
            self.tray.update_status(
                elapsed=_format_elapsed(int(status.get("elapsed", 0))),
                segments=int(status.get("segments", 0)),
                speakers=len(speakers) if isinstance(speakers, list) else 0,
                paused=bool(status.get("paused", False)),
                recording=True,
            )

    def _init_polish(self, writer: Any) -> None:
        """설정 기반으로 실시간 LLM 후처리를 초기화한다 (STT 교정 + 키워드 추출).

        Codex/Ollama 미설치 시에는 graceful skip — 에러 없이 진행.
        """
        from .polish import is_codex_available, is_ollama_available

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
            from .polish import _correct_batch_ollama, extract_keywords_with_ollama

            _model = self._ollama_model

            def _correct_fn_ollama(batch: list[str], idx: int, t: int = 120) -> tuple:
                return _correct_batch_ollama(batch, idx, _model, t)

            def _extract_kw_fn_ollama(text: str) -> list[str]:
                return extract_keywords_with_ollama(text, model=_model)

            self._correct_fn = _correct_fn_ollama
            self._extract_kw_fn = _extract_kw_fn_ollama
        else:
            from .polish import _correct_batch, extract_keywords_with_codex

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

    def _finalize_writer(self) -> None:
        """현재 writer를 마무리하고 세션 파일을 디스크에 저장한다.

        1. ThreadPoolExecutor shutdown (대기 최대 8초)
        2. 교정 결과 writer에 반영
        3. footer 기록 및 파일 닫기
        4. polish_meeting() 동기 호출 (요약 + STT 교정)
        """
        writer = self._writer
        if writer is None:
            return
        self._writer = None

        # 진행 중인 교정 태스크 취소 — 종료 시 새 태스크는 더 이상 불필요
        if self._polish_pool is not None:
            pool = self._polish_pool
            self._polish_pool = None
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

        try:
            elapsed_sec = int(time.time() - self._capture_start_time) if self._capture_start_time > 0 else 0
            elapsed_str = _format_elapsed(elapsed_sec)
            seg_count = len(writer._segments)
            writer.write_footer(
                duration=elapsed_str,
                segment_count=seg_count,
                speaker_count=len(writer._speakers),
            )
            writer.close()
        except Exception as exc:
            print(f"[desktop][writer] 세션 파일 저장 실패: {exc}", file=sys.stderr)
            return

        # 회의 후 LLM 후처리 (STT 교정 + 요약) — 세그먼트가 있을 때만
        if seg_count <= 0:
            set_postprocess_status("done", 100)
            return

        from .polish import is_codex_available, is_gemini_available, is_ollama_available

        if not (
            is_codex_available()
            or is_gemini_available()
            or is_ollama_available(self._ollama_model)
        ):
            set_postprocess_status("done", 100)
            return

        try:
            from .polish import polish_meeting

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
        if self.stop_event.is_set():
            return

        self.stop_event.set()
        request_shutdown()

        if self.tray is not None:
            self.tray.stop()

        # uvicorn 서버 + WhisperWorker 서브프로세스 강제 정리
        threading.Thread(target=self._force_exit, daemon=True).start()

    @staticmethod
    def _force_exit() -> None:
        """1초 대기 후 프로세스를 강제 종료한다 (서브프로세스/서버 잔존 방지)."""
        import os
        time.sleep(1.0)
        os._exit(0)


def _check_singleton() -> str | None:
    """이미 실행 중인 sonote 인스턴스가 있으면 그 URL을 반환, 없으면 None."""
    import ctypes
    if os.name != "nt":
        return None
    try:
        mutex = ctypes.windll.kernel32.CreateMutexW(None, False, "sonote_desktop_singleton")
        if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
            # 기존 인스턴스의 포트를 찾아서 브라우저로 열기
            import subprocess
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                if "LISTENING" not in line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                addr = parts[1]
                if "127.0.0.1:" in addr:
                    port = addr.split(":")[-1]
                    try:
                        pid = int(parts[-1])
                        proc_result = subprocess.run(
                            ["tasklist", "/FI", f"PID eq {pid}"],
                            capture_output=True, text=True, timeout=5,
                        )
                        if "sonote" in proc_result.stdout.lower():
                            return f"http://127.0.0.1:{port}"
                    except Exception:
                        continue
            return "already_running"
    except Exception:
        pass
    return None


def run_desktop(*, host: str = "127.0.0.1", port: int = 0, beta_mode: bool = False) -> None:
    """CLI에서 호출하는 데스크톱 런처."""
    resolved_beta_mode = _resolve_beta_mode(beta_mode)
    _configure_beta_mode(resolved_beta_mode)

    existing = _check_singleton()
    if existing and existing != "already_running":
        print(f"sonote가 이미 실행 중입니다: {existing}")
        webbrowser.open(existing)
        return
    elif existing == "already_running":
        print("sonote가 이미 실행 중입니다.")
        return

    controller = DesktopController(host=host, port=port, beta_mode=resolved_beta_mode)
    controller.run()


def main() -> None:
    """직접 실행용 엔트리포인트."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--beta", action="store_true")
    args, _ = parser.parse_known_args()
    if args.smoke_test:
        print("sonote desktop smoke ok")
        return
    run_desktop(beta_mode=args.beta)


if __name__ == "__main__":
    main()
