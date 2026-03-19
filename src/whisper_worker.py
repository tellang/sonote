"""Whisper STT 워커 프로세스 - CUDA 크래시 격리용 multiprocessing 워커."""

from __future__ import annotations

import atexit
import concurrent.futures
import json
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np

from .runtime_env import calculate_bon_workers, get_available_vram

_CTRL_HANDLER = None

# ---------------------------------------------------------------------------
# PID 레지스트리 — 좀비 워커 방지
# ---------------------------------------------------------------------------
_PID_REGISTRY = (
    Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")))
    / "sonote"
    / "run"
    / "workers.json"
)


def _read_pid_registry() -> dict:
    """레지스트리 파일을 읽어 dict로 반환. 실패 시 빈 레지스트리."""
    try:
        return json.loads(_PID_REGISTRY.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "pool_pid": None, "workers": []}


def _write_pid_registry(data: dict) -> None:
    """레지스트리를 원자적으로 갱신 (tmp → rename)."""
    _PID_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    tmp = _PID_REGISTRY.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(_PID_REGISTRY))


def _is_process_alive(pid: int) -> bool:
    """pid 프로세스가 살아있는지 확인."""
    if os.name == "nt":
        try:
            import ctypes

            # PROCESS_QUERY_LIMITED_INFORMATION
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _cleanup_stale_workers() -> int:
    """레지스트리의 좀비 워커를 정리하고 제거 수를 반환."""
    registry = _read_pid_registry()
    pool_pid = registry.get("pool_pid")
    workers = registry.get("workers", [])
    if not workers:
        return 0

    pool_alive = _is_process_alive(pool_pid) if pool_pid else False
    cleaned = 0
    remaining: list[dict] = []

    for w in workers:
        pid = w.get("pid")
        if not pid:
            continue
        if not _is_process_alive(pid):
            cleaned += 1
            continue
        if not pool_alive:
            # 부모 죽었는데 자식 살아있음 → kill
            try:
                if os.name == "nt":
                    import ctypes

                    handle = ctypes.windll.kernel32.OpenProcess(1, False, pid)
                    if handle:
                        ctypes.windll.kernel32.TerminateProcess(handle, 1)
                        ctypes.windll.kernel32.CloseHandle(handle)
                else:
                    os.kill(pid, 9)
                cleaned += 1
            except Exception:
                remaining.append(w)
        else:
            remaining.append(w)

    registry["workers"] = remaining
    if not remaining:
        registry["pool_pid"] = None
    _write_pid_registry(registry)
    return cleaned


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

        # 좀비 워커 정리 (이전 실행 잔류 프로세스)
        cleaned = _cleanup_stale_workers()
        if cleaned:
            import sys
            print(f"[pool] 좀비 워커 {cleaned}개 정리 완료", file=sys.stderr)

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
        self._pool_lock = threading.Lock()  # 동적 스케일링 동기화
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop = threading.Event()

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

            # PID 레지스트리 업데이트
            self._update_pid_registry()

            # 하트비트 데몬 시작
            self._heartbeat_stop.clear()
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
                name="pool-heartbeat",
            )
            self._heartbeat_thread.start()

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
                self._update_pid_registry()
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
        """Duration-weighted avg_logprob 스코어링.

        각 세그먼트의 avg_logprob를 해당 세그먼트 길이(초)로 가중 평균한다.
        짧은 hallucination 세그먼트가 점수를 왜곡하는 것을 방지한다.
        """
        weighted_sum = 0.0
        total_duration = 0.0
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            logprob = segment.get("avg_logprob")
            if logprob is None:
                continue
            try:
                logprob = float(logprob)
            except (TypeError, ValueError):
                continue
            start = float(segment.get("start", 0))
            end = float(segment.get("end", 0))
            duration = max(end - start, 0.01)  # 최소 10ms
            weighted_sum += logprob * duration
            total_duration += duration
        if total_duration <= 0:
            return float("-inf")
        return weighted_sum / total_duration

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
        # 하트비트 데몬 종료
        self._heartbeat_stop.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2)
        self._heartbeat_thread = None

        self._stop_all_workers()
        self._started = False
        self._ready = False

        # PID 레지스트리 정리
        try:
            registry = _read_pid_registry()
            if registry.get("pool_pid") == os.getpid():
                registry["pool_pid"] = None
                registry["workers"] = []
                _write_pid_registry(registry)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # PID 레지스트리 업데이트
    # ------------------------------------------------------------------
    def _update_pid_registry(self) -> None:
        """현재 살아있는 워커 PID를 레지스트리에 기록."""
        try:
            worker_entries = []
            for i, w in enumerate(self._workers):
                if w._started and w._process.pid is not None:
                    worker_entries.append({
                        "pid": w._process.pid,
                        "index": i,
                        "started_at": time.time(),
                    })
            registry = {
                "version": 1,
                "pool_pid": os.getpid(),
                "workers": worker_entries,
            }
            _write_pid_registry(registry)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 하트비트 루프 — 5초 주기로 워커 liveness 체크
    # ------------------------------------------------------------------
    def _heartbeat_loop(self) -> None:
        """데몬 스레드: 죽은 워커 감지 및 정리."""
        import sys
        while not self._heartbeat_stop.wait(timeout=5.0):
            if not self._started:
                break
            with self._pool_lock:
                for i, w in enumerate(self._workers):
                    if not w._started:
                        continue
                    if not w._process.is_alive():
                        print(
                            f"[heartbeat] 워커 {i} (pid={w._process.pid}) 사망 감지, "
                            f"exit={w._process.exitcode}",
                            file=sys.stderr,
                        )
                        # 죽은 워커를 새 워커로 교체
                        try:
                            new_w = self._new_worker()
                            new_w.start()
                            new_w.wait_ready(timeout=60.0)
                            self._workers[i] = new_w
                            self._worker_locks[i] = threading.Lock()
                            print(
                                f"[heartbeat] 워커 {i} 교체 완료",
                                file=sys.stderr,
                            )
                            self._update_pid_registry()
                        except Exception as exc:
                            print(
                                f"[heartbeat] 워커 {i} 교체 실패: {exc}",
                                file=sys.stderr,
                            )

    # ------------------------------------------------------------------
    # 동적 스케일링
    # ------------------------------------------------------------------
    def add_worker(self) -> int:
        """워커 1개 추가. VRAM 부족 시 RuntimeError. 새 워커 인덱스 반환."""
        with self._pool_lock:
            if len(self._workers) >= len(self._DEFAULT_TEMPERATURES):
                raise RuntimeError(
                    f"최대 워커 수({len(self._DEFAULT_TEMPERATURES)})에 도달"
                )
            # VRAM 체크 (모델 1개 로드에 ~800MB 필요)
            vram_mb = get_available_vram() // (1024 * 1024)
            if vram_mb < 800:
                raise RuntimeError(
                    f"VRAM 부족: {vram_mb}MB 남음 (최소 800MB 필요)"
                )

            idx = len(self._workers)
            new_w = self._new_worker()
            self._workers.append(new_w)
            self._worker_locks.append(threading.Lock())
            self._temperatures.append(self._DEFAULT_TEMPERATURES[idx])

            if self._started:
                new_w.start()
                new_w.wait_ready(timeout=120.0)
                self._update_pid_registry()

            return idx

    def remove_worker(self, index: int | None = None) -> None:
        """워커 1개 제거. 최소 1개는 유지. index 미지정 시 마지막 워커 제거."""
        with self._pool_lock:
            if len(self._workers) <= 1:
                raise RuntimeError("최소 1개 워커는 유지해야 합니다")

            if index is None:
                index = len(self._workers) - 1

            if index < 0 or index >= len(self._workers):
                raise IndexError(f"유효하지 않은 워커 인덱스: {index}")

            worker = self._workers.pop(index)
            self._worker_locks.pop(index)
            self._temperatures.pop(index)

            try:
                worker.stop()
            except Exception:
                pass

            if self._started:
                self._update_pid_registry()

    def scale_to(self, n: int) -> None:
        """워커 수를 n개로 조정. 최소 1, 최대 len(_DEFAULT_TEMPERATURES)."""
        n = max(1, min(n, len(self._DEFAULT_TEMPERATURES)))
        current = len(self._workers)

        if n == current:
            return

        if n > current:
            for _ in range(n - current):
                try:
                    self.add_worker()
                except RuntimeError:
                    break  # VRAM 부족 등 → 가능한 만큼만 추가
        else:
            for _ in range(current - n):
                try:
                    self.remove_worker()
                except RuntimeError:
                    break
