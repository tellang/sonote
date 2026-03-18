"""실시간 마이크 오디오 캡처 모듈."""

from collections import deque
from collections.abc import Callable, Generator
import threading

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1

_ring_buffer: deque[np.ndarray] = deque()
_ring_buffer_samples: int = 0
_ring_buffer_max_samples: int = int(SAMPLE_RATE * 30.0)
_ring_buffer_lock = threading.Lock()

__all__ = [
    "SAMPLE_RATE",
    "CHANNELS",
    "list_audio_devices",
    "find_builtin_mic",
    "capture_audio",
    "get_ring_buffer",
]

# 내장 마이크 이름 패턴 (우선순위 순서)
_BUILTIN_MIC_PATTERNS = ["마이크 배열", "인텔", "Intel", "Microphone Array"]
# 외장 장치 제외 패턴
_EXTERNAL_PATTERNS = ["CalDigit", "fifine", "USB", "Bluetooth", "BT LE", "Headphone"]


def _set_ring_buffer_limit(max_samples: int) -> None:
    """링버퍼 최대 샘플 수를 갱신하고 초과분을 제거한다."""
    global _ring_buffer_max_samples

    if max_samples < 1:
        max_samples = 1

    with _ring_buffer_lock:
        _ring_buffer_max_samples = max_samples
        _trim_ring_buffer_locked()


def _trim_ring_buffer_locked() -> None:
    """링버퍼를 최대 길이에 맞게 자른다. (락 내부 전용)"""
    global _ring_buffer_samples

    while _ring_buffer_samples > _ring_buffer_max_samples and _ring_buffer:
        overflow = _ring_buffer_samples - _ring_buffer_max_samples
        head = _ring_buffer[0]

        if head.size <= overflow:
            _ring_buffer.popleft()
            _ring_buffer_samples -= head.size
            continue

        _ring_buffer[0] = head[overflow:]
        _ring_buffer_samples -= overflow
        break


def _append_ring_buffer(samples: np.ndarray) -> None:
    """새 샘플을 링버퍼에 추가하고 길이를 유지한다."""
    global _ring_buffer_samples

    if samples.size == 0:
        return

    with _ring_buffer_lock:
        _ring_buffer.append(samples)
        _ring_buffer_samples += samples.size
        _trim_ring_buffer_locked()


def _reset_ring_buffer() -> None:
    """링버퍼를 비운다."""
    global _ring_buffer_samples

    with _ring_buffer_lock:
        _ring_buffer.clear()
        _ring_buffer_samples = 0


def _pop_from_deque(buffer: deque[np.ndarray], num_samples: int) -> np.ndarray:
    """청크 버퍼에서 지정 샘플 수만큼 꺼낸다."""
    out = np.empty(num_samples, dtype=np.float32)
    copied = 0

    while copied < num_samples and buffer:
        head = buffer[0]
        need = num_samples - copied

        if head.size <= need:
            out[copied:copied + head.size] = head
            copied += head.size
            buffer.popleft()
            continue

        out[copied:copied + need] = head[:need]
        buffer[0] = head[need:]
        copied += need

    return out[:copied]


def _choose_ready_chunk_samples(
    total_samples: int,
    min_chunk_samples: int,
    max_chunk_samples: int,
    silence_run_samples: int,
    silence_hold_samples: int,
) -> int:
    """휴리스틱 청크 절단 시점 계산."""
    if total_samples >= max_chunk_samples:
        return max_chunk_samples
    if total_samples >= min_chunk_samples and silence_run_samples >= silence_hold_samples:
        return total_samples
    return 0


def _count_trailing_silence(samples: np.ndarray, silence_threshold: float) -> int:
    """배열 끝에 연속된 저에너지 샘플 수를 계산한다."""
    if samples.size == 0:
        return 0

    silent = np.abs(samples) <= silence_threshold
    if silent.all():
        return int(samples.size)

    trailing = 0
    for is_silent in silent[::-1]:
        if not is_silent:
            break
        trailing += 1
    return trailing


def _should_emit_chunk(
    buffered_samples: int,
    min_chunk_samples: int,
    max_chunk_samples: int,
    trailing_silence_samples: int,
    silence_trigger_samples: int,
    heuristic_split: bool,
) -> bool:
    """현재 버퍼가 청크 방출 조건을 만족하는지 반환한다."""
    if buffered_samples >= max_chunk_samples:
        return True
    if not heuristic_split:
        return False
    return (
        buffered_samples >= min_chunk_samples
        and trailing_silence_samples >= silence_trigger_samples
    )


def find_builtin_mic() -> int | None:
    """내장 마이크(Intel SST 마이크 배열)를 이름 패턴으로 자동 탐색.

    CalDigit 등 외장 장치가 기본으로 잡혀 있어도 내장 마이크를 우선 선택.
    MME API 장치를 우선하며, 못 찾으면 DirectSound/WASAPI 순서로 탐색.

    Returns:
        내장 마이크 장치 인덱스. 못 찾으면 None (시스템 기본 사용).
    """
    candidates: list[tuple[int, int]] = []  # (우선순위, 장치 인덱스)

    for idx, dev in enumerate(sd.query_devices()):
        if int(dev.get("max_input_channels", 0)) <= 0:
            continue
        name = str(dev.get("name", ""))

        # 외장 장치 제외
        if any(ext.lower() in name.lower() for ext in _EXTERNAL_PATTERNS):
            continue

        # 내장 마이크 패턴 매칭
        if not any(pat.lower() in name.lower() for pat in _BUILTIN_MIC_PATTERNS):
            continue

        # API 우선순위: MME(0) > DirectSound(1) > WASAPI(2) > WDM-KS(3)
        api_name = sd.query_hostapis(dev["hostapi"])["name"]
        if "MME" in api_name:
            priority = 0
        elif "DirectSound" in api_name:
            priority = 1
        elif "WASAPI" in api_name:
            priority = 2
        else:
            priority = 3

        candidates.append((priority, idx))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


# 16kHz 리샘플링을 지원하는 API만 허용 (WASAPI/WDM-KS는 네이티브 레이트만 지원)
_COMPATIBLE_APIS = {"MME", "DirectSound"}


def list_audio_devices() -> list[dict]:
    """사용 가능한 오디오 입력 장치 목록 반환.

    WASAPI/WDM-KS 디바이스는 16kHz 리샘플링을 지원하지 않으므로
    MME/DirectSound API 장치만 반환한다.

    Returns:
        [{"index": int, "name": str, "channels": int, "sample_rate": float, "api": str}, ...]
    """
    devices: list[dict] = []

    for idx, dev in enumerate(sd.query_devices()):
        max_input_channels = int(dev.get("max_input_channels", 0))
        if max_input_channels <= 0:
            continue

        api_info = sd.query_hostapis(dev["hostapi"])
        api_name = api_info.get("name", "")

        # MME/DirectSound만 허용 (16kHz 리샘플링 지원)
        if not any(compat in api_name for compat in _COMPATIBLE_APIS):
            continue

        devices.append(
            {
                "index": idx,
                "name": str(dev.get("name", "")),
                "channels": max_input_channels,
                "sample_rate": float(dev.get("default_samplerate", 0.0)),
                "api": api_name,
            }
        )

    return devices


def capture_audio(
    chunk_seconds: float = 5.0,
    sample_rate: int = 16000,
    device: int | None = None,
    ring_buffer_seconds: float = 30.0,
    heuristic_split: bool = False,
    min_chunk_seconds: float | None = None,
    silence_threshold: float = 0.008,
    silence_duration_seconds: float = 0.45,
    stop_event: threading.Event | None = None,
    on_stream_started: Callable[[], None] | None = None,
) -> Generator[np.ndarray, None, None]:
    """마이크에서 실시간 오디오 청크를 yield하는 generator.

    기본은 고정 길이 청킹이고, heuristic_split=True면 최소 길이 이후
    trailing silence를 감지했을 때 더 이른 시점에 청크를 내보낸다.
    """
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds는 0보다 커야 합니다.")
    if sample_rate <= 0:
        raise ValueError("sample_rate는 0보다 커야 합니다.")
    if ring_buffer_seconds <= 0:
        raise ValueError("ring_buffer_seconds는 0보다 커야 합니다.")
    if min_chunk_seconds is not None and min_chunk_seconds <= 0:
        raise ValueError("min_chunk_seconds는 0보다 커야 합니다.")
    if silence_threshold < 0:
        raise ValueError("silence_threshold는 0 이상이어야 합니다.")
    if silence_duration_seconds <= 0:
        raise ValueError("silence_duration_seconds는 0보다 커야 합니다.")

    max_chunk_samples = int(sample_rate * chunk_seconds)
    min_chunk_samples = int(sample_rate * (min_chunk_seconds or chunk_seconds))
    if min_chunk_samples > max_chunk_samples:
        min_chunk_samples = max_chunk_samples
    ring_samples = int(sample_rate * ring_buffer_seconds)
    silence_trigger_samples = int(sample_rate * silence_duration_seconds)
    if max_chunk_samples < 1:
        raise ValueError("chunk_seconds가 너무 작아 청크 샘플 수가 0입니다.")

    _set_ring_buffer_limit(ring_samples)
    _reset_ring_buffer()

    chunk_ready_event = threading.Event()
    chunk_lock = threading.Lock()
    chunk_buffer: deque[np.ndarray] = deque()
    chunk_buffer_samples = 0
    trailing_silence_samples = 0

    def _audio_callback(
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        nonlocal chunk_buffer_samples, trailing_silence_samples

        _ = frames
        _ = time_info
        _ = status

        mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
        if mono.size == 0:
            return

        trailing = _count_trailing_silence(mono, silence_threshold)
        with chunk_lock:
            chunk_buffer.append(mono)
            chunk_buffer_samples += mono.size
            if trailing == mono.size:
                trailing_silence_samples += mono.size
            else:
                trailing_silence_samples = trailing
            if _should_emit_chunk(
                chunk_buffer_samples,
                min_chunk_samples,
                max_chunk_samples,
                trailing_silence_samples,
                silence_trigger_samples,
                heuristic_split,
            ):
                chunk_ready_event.set()

        _append_ring_buffer(mono)

    with sd.InputStream(
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32",
        device=device,
        callback=_audio_callback,
    ):
        if on_stream_started is not None:
            on_stream_started()
        while True:
            if stop_event is not None and stop_event.is_set():
                return
            if not chunk_ready_event.wait(timeout=0.2):
                continue

            while True:
                if stop_event is not None and stop_event.is_set():
                    return
                with chunk_lock:
                    if not _should_emit_chunk(
                        chunk_buffer_samples,
                        min_chunk_samples,
                        max_chunk_samples,
                        trailing_silence_samples,
                        silence_trigger_samples,
                        heuristic_split,
                    ):
                        chunk_ready_event.clear()
                        break

                    emit_samples = max_chunk_samples
                    if heuristic_split and chunk_buffer_samples < max_chunk_samples:
                        emit_samples = chunk_buffer_samples

                    chunk = _pop_from_deque(chunk_buffer, emit_samples)
                    chunk_buffer_samples -= emit_samples
                    trailing_silence_samples = 0 if chunk_buffer_samples else 0
                    if not _should_emit_chunk(
                        chunk_buffer_samples,
                        min_chunk_samples,
                        max_chunk_samples,
                        trailing_silence_samples,
                        silence_trigger_samples,
                        heuristic_split,
                    ):
                        chunk_ready_event.clear()

                if chunk.size:
                    yield chunk


def get_ring_buffer() -> np.ndarray:
    """현재 링버퍼 내용 반환 (화자 분리 컨텍스트용).

    Returns:
        float32 ndarray, 최대 ring_buffer_seconds 길이
    """
    with _ring_buffer_lock:
        if not _ring_buffer:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(list(_ring_buffer)).astype(np.float32, copy=False)


def _main() -> None:
    """모듈 직접 실행 시 장치 목록을 출력한다."""
    import argparse

    parser = argparse.ArgumentParser(description="오디오 캡처 유틸리티")
    parser.add_argument("--list-devices", action="store_true", help="입력 장치 목록 출력")
    args = parser.parse_args()

    if args.list_devices:
        for dev in list_audio_devices():
            print(
                f"[{dev['index']}] {dev['name']} "
                f"(channels={dev['channels']}, sample_rate={dev['sample_rate']})"
            )


if __name__ == "__main__":
    _main()
