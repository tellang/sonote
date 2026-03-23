"""
faster-whisper 기반 한국어 음성 인식 모듈
GPU(CUDA) 자동 감지, CPU int8 폴백 지원
"""
import atexit
import os
import sys
from pathlib import Path

from .runtime_env import bootstrap_nvidia_dll_path, detect_device

_WHISPER_MODEL_CLASS = None


def _get_whisper_model_class():
    global _WHISPER_MODEL_CLASS
    if _WHISPER_MODEL_CLASS is None:
        bootstrap_nvidia_dll_path()
        from faster_whisper import WhisperModel

        _WHISPER_MODEL_CLASS = WhisperModel
    return _WHISPER_MODEL_CLASS

# --- 도메인 용어 프롬프트 (리서치 v0.0.1 섹션 5-4 #1) ---
# Whisper initial_prompt에 삽입하여 핫워드 오인식 50%+ 감소
DOMAIN_PROMPT = (
    "파이썬 Python, 자바스크립트 JavaScript, "
    "타입스크립트 TypeScript, 리액트 React, 뷰 Vue.js, 스프링 Spring Boot, "
    "장고 Django, FastAPI, Docker, Kubernetes, GitHub, GitLab, "
    "API, REST, GraphQL, WebSocket, Redis, PostgreSQL, MongoDB, "
    "프론트엔드, 백엔드, 풀스택, 마이크로서비스, CI/CD, "
    "Jira, 스크럼 Scrum, 칸반 Kanban, 애자일 Agile, "
    "딥러닝 Deep Learning, 머신러닝 Machine Learning, LLM, RAG, "
    "GPT, Transformer, 파인튜닝 Fine-tuning, 임베딩 Embedding, "
    "vLLM, CUDA, PyTorch, TensorFlow, Hugging Face"
)

# ---------------------------------------------------------------------------
# CTranslate2 CUDA segfault 방지 (Windows)
#
# 근본 원인: CTranslate2 C++ 소멸자가 Python 종료 시 cuBLAS/cuDNN 핸들을
# 해제하려 하지만, Windows 로더가 이미 CUDA DLL을 언로드한 상태여서 segfault 발생.
# (CTranslate2 #1782, faster-whisper #1293, #71 — 미해결 upstream 버그)
#
# 해결: CUDA 모델 로드 시 atexit 핸들러를 등록하여 Python 정리 단계를 건너뜀.
# os._exit(0)은 Python finalizer/atexit/C++ static destructor를 모두 생략.
# CPU 전용 실행에는 영향 없음 (핸들러 미등록).
# ---------------------------------------------------------------------------
_cuda_exit_registered = False
_cuda_exit_enabled = True
_cuda_model_ref = None  # CUDA 모델 참조 유지 — 함수 리턴 후 GC 방지, os._exit(0)으로 정리


def _register_cuda_exit_guard() -> None:
    """CUDA 사용 시 프로세스 종료 segfault 방지 atexit 핸들러 등록 (1회)"""
    global _cuda_exit_registered
    if _cuda_exit_registered:
        return
    _cuda_exit_registered = True

    def _force_exit() -> None:
        if not _cuda_exit_enabled:
            return
        # stdout/stderr 플러시 후 즉시 종료 — Python 정리 단계 전체 생략
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(0)

    atexit.register(_force_exit)


def disable_cuda_exit_guard() -> None:
    """force exit 핸들러 비활성화 — CUDA 모델 해제 후 서브프로세스 실행 시 사용"""
    global _cuda_exit_enabled
    _cuda_exit_enabled = False
def transcribe_audio(
    audio_path: str | Path,
    model_id: str = "tellang/whisper-medium-ko-ct2",
    language: str = "ko",
    device: str | None = None,
    compute_type: str | None = None,
    beam_size: int = 5,
    vad_filter: bool = True,
    min_silence_ms: int = 500,
    initial_prompt: str | None = None,
) -> list[dict]:
    """
    오디오 파일을 텍스트로 변환

    Args:
        audio_path: WAV/MP3 오디오 파일 경로
        model_id: Whisper 모델 ID
        language: 인식 언어 (ko, en, ja 등)
        device: cuda 또는 cpu (None이면 자동 감지)
        compute_type: float16, int8 등 (None이면 자동)
        beam_size: 빔 서치 크기 (클수록 정확, 느림)
        vad_filter: 음성 활동 감지 필터 사용 여부
        min_silence_ms: 최소 침묵 구간 (ms)
        initial_prompt: 도메인 용어 프롬프트 (핫워드 인식 개선)

    Returns:
        [{"start": float, "end": float, "text": str}, ...]
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"오디오 파일 없음: {audio_path}")

    # 디바이스 자동 감지
    if device is None or compute_type is None:
        auto_device, auto_compute = detect_device()
        device = device or auto_device
        compute_type = compute_type or auto_compute

    # 한국어이고 프롬프트 미지정 시 도메인 용어 자동 삽입
    if initial_prompt is None and language == "ko":
        initial_prompt = DOMAIN_PROMPT

    print(f"[모델] {model_id} | {device} ({compute_type})")
    print(f"[입력] {audio_path} ({audio_path.stat().st_size / 1024 / 1024:.1f}MB)")

    # 모델 로드
    whisper_model_class = _get_whisper_model_class()
    model = whisper_model_class(model_id, device=device, compute_type=compute_type)
    if device == "cuda":
        _register_cuda_exit_guard()

    # 트랜스크립션
    vad_params = {"min_silence_duration_ms": min_silence_ms} if vad_filter else None
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        vad_parameters=vad_params,
        word_timestamps=False,
        initial_prompt=initial_prompt,
    )

    print(f"[언어] {info.language} (확률: {info.language_probability:.1%})")

    # 결과 수집
    results = []
    for seg in segments_iter:
        results.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })

    print(f"[완료] {len(results)}개 세그먼트")

    # CUDA 모델 명시 해제 — GC에 의한 segfault 방지
    del model
    import gc
    gc.collect()

    return results


def get_audio_duration(audio_path: str | Path) -> float:
    """ffprobe로 오디오 길이(초) 반환"""
    import subprocess
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(audio_path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(f"오디오 길이 확인 실패: {result.stderr[:200]}")


def split_audio(
    audio_path: str | Path,
    chunk_minutes: int,
    output_dir: Path | None = None,
) -> list[Path]:
    """오디오를 N분 단위 청크 WAV로 분할"""
    import subprocess
    import shutil
    import tempfile

    audio_path = Path(audio_path)
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="lt-chunks-"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg가 설치되어 있지 않습니다")

    pattern = str(output_dir / "chunk_%04d.wav")
    subprocess.run(
        [
            ffmpeg_path,
            "-i", str(audio_path),
            "-f", "segment",
            "-segment_time", str(chunk_minutes * 60),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            pattern, "-y",
        ],
        capture_output=True,
    )

    chunks = sorted(output_dir.glob("chunk_*.wav"))
    if not chunks:
        raise RuntimeError("오디오 분할 실패 — 청크 파일 없음")
    return chunks


def transcribe_chunks(
    audio_path: str | Path,
    chunk_minutes: int = 10,
    model_id: str = "tellang/whisper-medium-ko-ct2",
    language: str = "ko",
    device: str | None = None,
    compute_type: str | None = None,
    beam_size: int = 5,
    vad_filter: bool = True,
    min_silence_ms: int = 500,
    initial_prompt: str | None = None,
) -> list[dict]:
    """
    긴 오디오를 청크 단위로 분할 변환 (진행률 표시)

    Args:
        audio_path: 오디오 파일 경로
        chunk_minutes: 청크 크기 (분)
        initial_prompt: 도메인 용어 프롬프트 (핫워드 인식 개선)
        나머지: transcribe_audio와 동일

    Returns:
        [{"start": float, "end": float, "text": str}, ...]
    """
    import shutil as _shutil

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"오디오 파일 없음: {audio_path}")

    if device is None or compute_type is None:
        auto_dev, auto_comp = detect_device()
        device = device or auto_dev
        compute_type = compute_type or auto_comp

    # 한국어이고 프롬프트 미지정 시 도메인 용어 자동 삽입
    if initial_prompt is None and language == "ko":
        initial_prompt = DOMAIN_PROMPT

    # 오디오 분할
    duration = get_audio_duration(audio_path)
    num_est = max(1, int(duration / (chunk_minutes * 60)) + 1)
    print(f"[청크] {duration / 60:.1f}분 오디오 → ~{num_est}개 청크 ({chunk_minutes}분 단위)")

    chunks = split_audio(audio_path, chunk_minutes)
    print(f"[청크] {len(chunks)}개 청크 생성")
    print(f"[모델] {model_id} | {device} ({compute_type})")

    # 모델 한 번만 로드
    whisper_model_class = _get_whisper_model_class()
    model = whisper_model_class(model_id, device=device, compute_type=compute_type)
    if device == "cuda":
        _register_cuda_exit_guard()

    all_segments = []
    vad_params = {"min_silence_duration_ms": min_silence_ms} if vad_filter else None
    chunk_dur_sec = chunk_minutes * 60

    for i, chunk in enumerate(chunks):
        offset = i * chunk_dur_sec
        offset_min = offset / 60
        print(f"\n[{i + 1}/{len(chunks)}] {chunk.name} (시작: {offset_min:.0f}분)")

        segs_iter, info = model.transcribe(
            str(chunk),
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=vad_params,
            word_timestamps=False,
            initial_prompt=initial_prompt,
        )

        count = 0
        for seg in segs_iter:
            all_segments.append({
                "start": seg.start + offset,
                "end": seg.end + offset,
                "text": seg.text.strip(),
            })
            count += 1
        print(f"  → {count}개 세그먼트")

    # 청크 임시 파일 정리
    chunk_dir = chunks[0].parent
    _shutil.rmtree(chunk_dir, ignore_errors=True)

    # CUDA 모델을 모듈 레벨에 보관 — 함수 리턴 시 GC 소멸자 abort 방지
    # atexit의 os._exit(0)이 Python 정리 단계를 건너뛰어 안전하게 종료
    if device == "cuda":
        global _cuda_model_ref
        _cuda_model_ref = model

    print(f"\n[완료] 총 {len(all_segments)}개 세그먼트")
    return all_segments


def format_timestamp(seconds: float) -> str:
    """초 -> MM:SS 형식"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def save_transcript(
    segments: list[dict],
    output_path: str | Path,
    fmt: str = "txt",
) -> Path:
    """
    트랜스크립션 결과 저장

    Args:
        segments: transcribe_audio 반환값
        output_path: 출력 파일 경로
        fmt: 출력 형식 (txt, srt)
    """
    output_path = Path(output_path)

    if fmt == "srt":
        lines = []
        for i, seg in enumerate(segments, 1):
            start_h = int(seg["start"] // 3600)
            start_m = int((seg["start"] % 3600) // 60)
            start_s = int(seg["start"] % 60)
            start_ms = int((seg["start"] % 1) * 1000)
            end_h = int(seg["end"] // 3600)
            end_m = int((seg["end"] % 3600) // 60)
            end_s = int(seg["end"] % 60)
            end_ms = int((seg["end"] % 1) * 1000)
            lines.append(str(i))
            lines.append(
                f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms:03d}"
                f" --> "
                f"{end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms:03d}"
            )
            lines.append(seg["text"])
            lines.append("")
        text = "\n".join(lines)
    else:
        lines = []
        for seg in segments:
            ts = f"[{format_timestamp(seg['start'])} ~ {format_timestamp(seg['end'])}]"
            lines.append(f"{ts} {seg['text']}")
        text = "\n".join(lines)

    output_path.write_text(text, encoding="utf-8")
    print(f"[저장] {output_path}")
    return output_path
