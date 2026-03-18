"""Whisper STT 워커 프로세스 - CUDA 크래시 격리용 multiprocessing 워커."""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
from queue import Empty
from typing import Any

import numpy as np

_CTRL_HANDLER = None


def _bind_to_job_object(pid: int) -> None:
    """Windows Job Object에 자식 프로세스를 등록한다.

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE 플래그로 부모 프로세스가
    강제 종료(taskkill /F)되어도 자식이 자동으로 같이 죽는다.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32

        # Job Object 생성
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return

        # JOBOBJECT_EXTENDED_LIMIT_INFORMATION 설정
        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        kernel32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )

        # 자식 프로세스를 Job에 할당
        PROCESS_ALL_ACCESS = 0x1F0FFF
        handle = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
        if handle:
            kernel32.AssignProcessToJobObject(job, handle)
            kernel32.CloseHandle(handle)

        # Job handle을 전역에 유지 (GC 방지 — handle 닫히면 자식도 죽음)
        _bind_to_job_object._job_handle = job  # type: ignore[attr-defined]
    except Exception:
        pass


def _worker_loop(
    model_id: str,
    language: str,
    device: str,
    compute_type: str,
    in_q: mp.Queue,
    out_q: mp.Queue,
) -> None:
    """워커 프로세스 메인 루프.

    모델을 로드하고, in_q에서 (audio_chunk, kwargs) 튜플을 받아
    STT 실행 후 결과를 out_q에 dict 리스트로 반환.
    None 수신 시 종료.
    """
    # CUDA DLL 경로 등록 (src/transcribe.py 상단과 동일한 목적)
    global _CTRL_HANDLER
    import os
    import signal
    from .runtime_env import bootstrap_nvidia_dll_path

    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (AttributeError, ValueError):
        pass
    try:
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    except (AttributeError, ValueError):
        pass
    if os.name == "nt":
        try:
            import ctypes

            handler_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)

            # Microsoft 문서 기준으로 NULL + TRUE는 프로세스가 CTRL+C를 무시하게 한다.
            ctypes.windll.kernel32.SetConsoleCtrlHandler(None, True)

            @handler_type
            def _ctrl_handler(ctrl_type):
                _ = ctrl_type
                return True

            ctypes.windll.kernel32.SetConsoleCtrlHandler(_ctrl_handler, True)
            _CTRL_HANDLER = _ctrl_handler
        except Exception:
            pass

    bootstrap_nvidia_dll_path()

    from faster_whisper import WhisperModel

    try:
        model = WhisperModel(model_id, device=device, compute_type=compute_type)
    except Exception as exc:
        out_q.put({"type": "error", "error": f"모델 로드 실패: {exc}"})
        os._exit(1)

    out_q.put({"type": "ready"})  # 모델 로드 완료 신호

    try:
        while True:
            item = in_q.get()
            if item is None:  # 종료 신호
                break

            audio_chunk, transcribe_kwargs = item
            transcribe_kwargs = dict(transcribe_kwargs or {})
            transcribe_kwargs.setdefault("language", language)

            try:
                segments_gen, _ = model.transcribe(audio_chunk, **transcribe_kwargs)
                segments: list[dict[str, Any]] = []
                for seg in segments_gen:
                    seg_dict: dict[str, Any] = {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                        "avg_logprob": seg.avg_logprob,
                        "no_speech_prob": seg.no_speech_prob,
                        "compression_ratio": seg.compression_ratio,
                    }
                    # word_timestamps가 있으면 포함
                    if hasattr(seg, "words") and seg.words:
                        seg_dict["words"] = [
                            {
                                "start": word.start,
                                "end": word.end,
                                "word": word.word,
                                "probability": word.probability,
                            }
                            for word in seg.words
                        ]
                    segments.append(seg_dict)

                out_q.put({"type": "result", "segments": segments})
            except Exception as exc:
                out_q.put({"type": "error", "error": str(exc)})
    except KeyboardInterrupt:
        pass

    # 워커 종료 - CUDA 크래시해도 메인에 영향 없음
    # os._exit(0)으로 즉시 종료 (C++ 소멸자 호출 방지)
    os._exit(0)


class WhisperWorker:
    """Whisper STT 워커 프로세스 관리자."""

    def __init__(
        self,
        model_id: str = "large-v3-turbo",
        language: str = "ko",
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        self._in_q: mp.Queue = mp.Queue()
        self._out_q: mp.Queue = mp.Queue()
        self._process = mp.Process(
            target=_worker_loop,
            args=(
                model_id,
                language,
                device,
                compute_type,
                self._in_q,
                self._out_q,
            ),
            daemon=True,
        )
        self._started = False
        self._ready = False

    def start(self, timeout: float = 60.0) -> None:
        """워커 프로세스 시작 (비동기 — 모델 로드를 기다리지 않고 즉시 반환).

        모델 로드 완료는 ``is_ready`` 프로퍼티로 확인하거나
        ``wait_ready()``로 명시적 대기.
        ``transcribe()``는 내부에서 자동 대기하므로 별도 호출 불필요.
        """
        self._process.start()

        # Windows Job Object: 부모 강제 종료 시 자식도 자동 종료
        if self._process.pid is not None:
            _bind_to_job_object(self._process.pid)

        # atexit: 정상 종료 시 워커 정리 보장
        atexit.register(self.stop)

        self._started = True
        self._ready_timeout = timeout

    @property
    def is_ready(self) -> bool:
        """모델 로드 완료 여부 (논블로킹)."""
        if self._ready:
            return True
        if not self._started:
            return False
        try:
            msg = self._out_q.get_nowait()
            if msg.get("type") == "ready":
                self._ready = True
                return True
            if msg.get("type") == "error":
                raise RuntimeError(f"워커 시작 실패: {msg}")
        except Empty:
            pass
        return False

    def wait_ready(self, timeout: float | None = None) -> None:
        """모델 로드 완료까지 블로킹 대기."""
        if self._ready:
            return
        t = timeout if timeout is not None else getattr(self, "_ready_timeout", 60.0)
        try:
            msg = self._out_q.get(timeout=t)
        except Empty as exc:
            raise RuntimeError("워커 시작 실패: 준비 신호 타임아웃") from exc
        if msg.get("type") != "ready":
            raise RuntimeError(f"워커 시작 실패: {msg}")
        self._ready = True

    def transcribe(
        self,
        audio_chunk: np.ndarray,
        timeout: float = 120.0,
        **kwargs: Any,
    ) -> list[dict]:
        """오디오 청크를 워커에 전송하고 결과 수신."""
        if not self._started:
            raise RuntimeError("워커가 시작되지 않음")

        # 모델 로드 미완료 시 자동 대기
        if not self._ready:
            self.wait_ready()

        self._in_q.put((audio_chunk, kwargs))
        try:
            msg = self._out_q.get(timeout=timeout)
        except Empty as exc:
            raise RuntimeError("STT 실패: 워커 응답 타임아웃") from exc

        if msg.get("type") == "error":
            raise RuntimeError(f"STT 실패: {msg.get('error', 'unknown')}")
        if msg.get("type") != "result":
            raise RuntimeError(f"STT 실패: 알 수 없는 워커 응답 {msg}")
        return msg.get("segments", [])

    def stop(self) -> None:
        """워커 종료 (graceful -> force)."""
        if not self._started:
            return

        try:
            self._in_q.put(None)  # 종료 신호
            self._process.join(timeout=5)
        except Exception:
            pass

        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)

        self._started = False

    @property
    def is_alive(self) -> bool:
        return self._process.is_alive() if self._started else False
