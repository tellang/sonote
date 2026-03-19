"""Whisper STT 워커 프로세스 - CUDA 크래시 격리용 multiprocessing 워커."""

from __future__ import annotations

import atexit
import concurrent.futures
import multiprocessing as mp
import os
import threading
import time
from queue import Empty
from typing import Any

import numpy as np

from .runtime_env import calculate_bon_workers

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

    import threading

    # import + 모델 로드를 모두 스레드에서 실행 (Windows CUDA 서브프로세스 hang 대응)
    # faster_whisper import 자체가 CTranslate2 DLL + CUDA 초기화를 트리거하므로
    # import도 타임아웃 범위 안에 포함해야 한다.
    _model_ref: list = [None]
    _load_error: list = [None]
    _LOAD_TIMEOUT = 120  # 초

    def _import_and_load() -> None:
        try:
            from faster_whisper import WhisperModel
            _model_ref[0] = WhisperModel(model_id, device=device, compute_type=compute_type)
        except Exception as exc:
            _load_error[0] = exc

    loader = threading.Thread(target=_import_and_load, daemon=True)
    loader.start()
    loader.join(timeout=_LOAD_TIMEOUT)

    if loader.is_alive():
        out_q.put({
            "type": "error",
            "error": f"모델 로드 타임아웃 ({_LOAD_TIMEOUT}초) — GPU 상태를 확인하세요. "
                     "다른 프로세스가 GPU를 점유 중이거나 CUDA 초기화 문제일 수 있습니다.",
        })
        raise SystemExit(1)

    if _load_error[0]:
        out_q.put({"type": "error", "error": f"모델 로드 실패: {_load_error[0]}"})
        raise SystemExit(1)

    model = _model_ref[0]
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

    # 워커 종료는 우선 graceful return을 사용한다.
    # 종료 지연/교착은 부모 stop()에서 terminate()로 회수한다.
    return


class WhisperWorker:
    """Whisper STT 워커 프로세스 관리자."""

    def __init__(
        self,
        model_id: str = "tellang/whisper-large-v3-turbo-ko",
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
        # 워커 프로세스가 죽었으면 즉시 에러
        if not self._process.is_alive():
            raise RuntimeError(
                f"워커 프로세스 사망 (exit={self._process.exitcode}). "
                "GPU 메모리 부족이거나 CUDA 오류일 수 있습니다. "
                "다른 sonote/Python 프로세스가 GPU를 점유 중인지 확인하세요."
            )
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


class WhisperWorkerPool:
    """Whisper BoN 워커 풀 관리자."""

    _DEFAULT_TEMPERATURES: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6)

    def __init__(
        self,
        model_id: str = "tellang/whisper-large-v3-turbo-ko",
        language: str = "ko",
        device: str = "cuda",
        compute_type: str = "float16",
        n_workers: int | None = None,
    ) -> None:
        self._model_id = model_id
        self._language = language
        self._device = device
        self._compute_type = compute_type

        if os.getenv("SONOTE_BETA") == "1":
            target_workers = 1
        elif n_workers is None:
            target_workers = calculate_bon_workers()
        else:
            target_workers = int(n_workers)

        safe_workers = max(1, min(target_workers, len(self._DEFAULT_TEMPERATURES)))
        self._workers: list[WhisperWorker] = []
        self._worker_locks: list[threading.Lock] = []
        self._temperatures: list[float] = list(self._DEFAULT_TEMPERATURES[:safe_workers])
        self._started = False
        self._ready = False
        self._target_workers = safe_workers

        try:
            self._workers = [self._new_worker() for _ in range(safe_workers)]
            self._worker_locks = [threading.Lock() for _ in range(safe_workers)]
        except Exception:
            # 풀 구성 실패 시 단일 워커로 폴백
            self._workers = [self._new_worker()]
            self._worker_locks = [threading.Lock()]
            self._temperatures = [self._DEFAULT_TEMPERATURES[0]]
            self._target_workers = 1

    @property
    def n_workers(self) -> int:
        return len(self._workers)

    @property
    def ready_workers(self) -> int:
        """현재 사용 가능한 워커 수."""
        return sum(1 for w in self._workers if w.is_ready)

    @property
    def is_ready(self) -> bool:
        """최소 1개 워커가 준비되면 True."""
        return any(w.is_ready for w in self._workers)

    @property
    def is_alive(self) -> bool:
        if self.n_workers == 1:
            return self._workers[0].is_alive
        return any(worker.is_alive for worker in self._workers)

    def _new_worker(self) -> WhisperWorker:
        return WhisperWorker(
            model_id=self._model_id,
            language=self._language,
            device=self._device,
            compute_type=self._compute_type,
        )

    def _stop_all_workers(self) -> None:
        for worker in self._workers:
            try:
                worker.stop()
            except Exception:
                pass

    def _fallback_to_single_worker(
        self,
        start_worker: bool = False,
        ready_timeout: float = 120.0,
    ) -> None:
        # 풀 오류 시 즉시 단일 워커로 축소
        self._stop_all_workers()
        self._workers = [self._new_worker()]
        self._worker_locks = [threading.Lock()]
        self._temperatures = [self._DEFAULT_TEMPERATURES[0]]
        self._ready = False

        if start_worker:
            self._workers[0].start()
            self._workers[0].wait_ready(timeout=ready_timeout)
            self._started = True

    def start(self) -> None:
        """첫 번째 워커만 즉시 시작하고, 나머지는 백그라운드에서 점진적으로 로드."""
        try:
            # 첫 번째 워커 즉시 시작
            self._workers[0].start()
            self._started = True

            # 나머지 워커는 백그라운드 스레드에서 순차 시작
            if self.n_workers > 1:
                threading.Thread(
                    target=self._start_remaining_workers,
                    daemon=True,
                    name="bon-progressive-loader",
                ).start()
        except Exception:
            self._fallback_to_single_worker(start_worker=True, ready_timeout=120.0)

    def _start_remaining_workers(self) -> None:
        """나머지 워커를 순차적으로 시작 — 각 워커 ready 후 다음 시작."""
        import sys
        for i in range(1, self.n_workers):
            try:
                self._workers[i].start()
                self._workers[i].wait_ready(timeout=120.0)
                print(
                    f"[BoN] 워커 {i+1}/{self.n_workers} 준비 완료 (BoN×{self.ready_workers})",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(f"[BoN] 워커 {i+1} 시작 실패: {exc}", file=sys.stderr)
                break  # 이후 워커도 실패할 가능성 높음 → 현재까지만 사용

    def wait_ready(self, timeout: float = 120.0) -> None:
        """첫 번째 워커만 대기 — 나머지는 백그라운드에서 점진 로드."""
        if not self._started:
            raise RuntimeError("워커 풀이 시작되지 않음")
        if self._ready:
            return

        try:
            self._workers[0].wait_ready(timeout=timeout)
            self._ready = True
        except Exception:
            self._fallback_to_single_worker(start_worker=True, ready_timeout=timeout)
            self._ready = True

    @staticmethod
    def _score_segments(segments: list[dict[str, Any]]) -> float:
        scores: list[float] = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            value = segment.get("avg_logprob")
            if value is None:
                continue
            try:
                scores.append(float(value))
            except (TypeError, ValueError):
                continue
        if not scores:
            return float("-inf")
        return sum(scores) / len(scores)

    @staticmethod
    def _transcribe_with_lock(
        worker: WhisperWorker,
        lock: threading.Lock,
        audio_chunk: np.ndarray,
        kwargs: dict[str, Any],
    ) -> list[dict]:
        try:
            return worker.transcribe(audio_chunk, **kwargs)
        finally:
            lock.release()

    def transcribe(self, audio_chunk: np.ndarray, **kwargs: Any) -> list[dict]:
        if not self._started:
            raise RuntimeError("워커 풀이 시작되지 않음")

        # ready인 워커만 수집
        ready_indices = [i for i, w in enumerate(self._workers) if w.is_ready]

        # ready 워커가 1개면 단일 모드 (오버헤드 없음)
        if len(ready_indices) <= 1:
            idx = ready_indices[0] if ready_indices else 0
            return self._workers[idx].transcribe(audio_chunk, **kwargs)

        selected_workers: list[tuple[int, concurrent.futures.Future[list[dict]]]] = []
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(ready_indices),
            thread_name_prefix="whisper-bon",
        )

        try:
            for idx in ready_indices:
                worker = self._workers[idx]
                lock = self._worker_locks[idx]
                if not lock.acquire(blocking=False):
                    continue

                worker_kwargs = dict(kwargs)
                worker_kwargs["temperature"] = self._temperatures[idx]

                try:
                    future = executor.submit(
                        self._transcribe_with_lock,
                        worker,
                        lock,
                        audio_chunk,
                        worker_kwargs,
                    )
                    selected_workers.append((idx, future))
                except Exception:
                    lock.release()

            if not selected_workers:
                raise RuntimeError("사용 가능한 BoN 워커 없음")

            futures = [future for _, future in selected_workers]
            done, _ = concurrent.futures.wait(
                futures,
                timeout=5.0,
                return_when=concurrent.futures.ALL_COMPLETED,
            )

            ranked_results: list[tuple[float, list[dict]]] = []
            for idx, future in selected_workers:
                if future not in done:
                    continue
                try:
                    segments = future.result()
                except Exception:
                    continue
                ranked_results.append((self._score_segments(segments), segments))

            if not ranked_results:
                raise RuntimeError("BoN 결과 수집 실패")

            ranked_results.sort(key=lambda item: item[0], reverse=True)
            return ranked_results[0][1]
        except Exception:
            # 풀 실행 실패 시 단일 워커로 즉시 폴백
            self._fallback_to_single_worker(start_worker=True, ready_timeout=120.0)
            return self._workers[0].transcribe(audio_chunk, **kwargs)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def stop(self) -> None:
        self._stop_all_workers()
        self._started = False
        self._ready = False
