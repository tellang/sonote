#!/usr/bin/env python3
"""
faster-whisper 모델 성능 벤치마크 스크립트

모델별 로드 시간, GPU 메모리 사용량, 전사 시간, RTF(Real-Time Factor)를 측정한다.
결과를 JSON / 마크다운 테이블로 출력한다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (상위 디렉토리의 src 패키지 임포트용)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.runtime_env import bootstrap_nvidia_dll_path, detect_device

# 기본 벤치마크 모델 목록
DEFAULT_MODELS = ["tiny", "base", "small", "medium", "large-v3-turbo"]


# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    """단일 모델 벤치마크 결과."""

    name: str
    load_time_s: float = 0.0
    memory_mb: float = 0.0
    transcribe_time_s: float = 0.0
    rtf: float = 0.0
    word_count: int = 0
    error: str | None = None


@dataclass
class BenchmarkReport:
    """전체 벤치마크 보고서."""

    device: str = ""
    compute_type: str = ""
    audio_file: str = ""
    audio_duration_s: float = 0.0
    total_elapsed_s: float = 0.0
    models: list[ModelResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GPU 메모리 유틸
# ---------------------------------------------------------------------------


def _get_gpu_memory_mb() -> float:
    """현재 CUDA GPU 할당 메모리(MB) 반환. CUDA 미사용 시 0.0."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def _get_peak_gpu_memory_mb() -> float:
    """피크 CUDA GPU 할당 메모리(MB) 반환. CUDA 미사용 시 0.0."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def _reset_peak_gpu_memory() -> None:
    """피크 GPU 메모리 통계 리셋."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _get_audio_duration(audio_path: Path) -> float:
    """오디오 파일 길이(초)를 반환한다. ffprobe 또는 soundfile 사용."""
    # soundfile 시도
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        return info.duration
    except Exception:
        pass

    # ffprobe 시도
    try:
        import subprocess

        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        pass

    return 0.0


# ---------------------------------------------------------------------------
# 벤치마크 핵심 로직
# ---------------------------------------------------------------------------


def benchmark_single_model(
    model_name: str,
    audio_path: Path | None,
    audio_duration: float,
    device: str,
    compute_type: str,
    dry_run: bool = False,
) -> ModelResult:
    """단일 모델의 로드 시간, 메모리, 전사 성능을 측정한다."""
    result = ModelResult(name=model_name)

    bootstrap_nvidia_dll_path()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        result.error = "faster-whisper 패키지 미설치"
        return result

    # GPU 메모리 리셋
    _reset_peak_gpu_memory()
    mem_before = _get_gpu_memory_mb()

    # 모델 로드 시간 측정
    print(f"\n{'='*60}")
    print(f"[벤치마크] {model_name} | {device} ({compute_type})")
    print(f"{'='*60}")

    try:
        t0 = time.perf_counter()
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        t1 = time.perf_counter()
        result.load_time_s = round(t1 - t0, 3)
    except Exception as e:
        result.error = f"모델 로드 실패: {e}"
        print(f"  [오류] {result.error}")
        return result

    # GPU 메모리 측정 (로드 후)
    mem_after_load = _get_peak_gpu_memory_mb()
    result.memory_mb = round(max(mem_after_load - mem_before, 0.0), 1)

    print(f"  로드 시간: {result.load_time_s:.3f}s")
    print(f"  GPU 메모리: {result.memory_mb:.1f} MB")

    # dry-run이면 전사 스킵
    if dry_run or audio_path is None:
        print("  [dry-run] 전사 스킵")
        del model
        return result

    # 전사 시간 측정
    try:
        t0 = time.perf_counter()
        segments, _info = model.transcribe(
            str(audio_path),
            language="ko",
            beam_size=5,
            vad_filter=True,
        )
        # segments는 제너레이터이므로 소비해야 실제 전사가 수행됨
        words = []
        for seg in segments:
            words.extend(seg.text.strip().split())
        t1 = time.perf_counter()

        result.transcribe_time_s = round(t1 - t0, 3)
        result.word_count = len(words)

        # RTF = 전사 시간 / 오디오 길이 (낮을수록 빠름)
        if audio_duration > 0:
            result.rtf = round(result.transcribe_time_s / audio_duration, 4)

        # 피크 메모리 갱신
        peak_mem = _get_peak_gpu_memory_mb()
        result.memory_mb = round(max(peak_mem - mem_before, result.memory_mb), 1)

        print(f"  전사 시간: {result.transcribe_time_s:.3f}s")
        print(f"  RTF: {result.rtf:.4f}")
        print(f"  단어 수: {result.word_count}")

    except Exception as e:
        result.error = f"전사 실패: {e}"
        print(f"  [오류] {result.error}")

    del model
    return result


def run_benchmark(
    models: list[str],
    audio_path: Path | None,
    device: str,
    compute_type: str,
    dry_run: bool = False,
) -> BenchmarkReport:
    """전체 모델에 대해 벤치마크를 실행한다."""
    report = BenchmarkReport(
        device=device,
        compute_type=compute_type,
    )

    audio_duration = 0.0
    if audio_path is not None:
        report.audio_file = str(audio_path)
        audio_duration = _get_audio_duration(audio_path)
        report.audio_duration_s = round(audio_duration, 2)
        print(f"[오디오] {audio_path} ({audio_duration:.1f}초)")

    total_start = time.perf_counter()

    for model_name in models:
        result = benchmark_single_model(
            model_name=model_name,
            audio_path=audio_path,
            audio_duration=audio_duration,
            device=device,
            compute_type=compute_type,
            dry_run=dry_run,
        )
        report.models.append(result)

    report.total_elapsed_s = round(time.perf_counter() - total_start, 2)
    return report


# ---------------------------------------------------------------------------
# 출력 포맷터
# ---------------------------------------------------------------------------


def report_to_json(report: BenchmarkReport) -> str:
    """벤치마크 보고서를 JSON 문자열로 변환한다."""
    data = {
        "device": report.device,
        "compute_type": report.compute_type,
        "audio_file": report.audio_file,
        "audio_duration_s": report.audio_duration_s,
        "total_elapsed_s": report.total_elapsed_s,
        "models": [asdict(m) for m in report.models],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def report_to_markdown(report: BenchmarkReport) -> str:
    """벤치마크 보고서를 마크다운 테이블로 변환한다."""
    lines: list[str] = []
    lines.append(f"## 벤치마크 결과")
    lines.append(f"")
    lines.append(f"- **디바이스**: {report.device} ({report.compute_type})")
    if report.audio_file:
        lines.append(f"- **오디오**: {report.audio_file} ({report.audio_duration_s}초)")
    lines.append(f"- **전체 소요시간**: {report.total_elapsed_s}초")
    lines.append(f"")
    lines.append(f"| Model | Load (s) | Memory (MB) | Transcribe (s) | RTF | Words | Error |")
    lines.append(f"|-------|----------|-------------|-----------------|-----|-------|-------|")

    for m in report.models:
        error_str = m.error or ""
        lines.append(
            f"| {m.name} | {m.load_time_s:.3f} | {m.memory_mb:.1f} | "
            f"{m.transcribe_time_s:.3f} | {m.rtf:.4f} | {m.word_count} | {error_str} |"
        )

    # 요약 통계 (에러 없는 모델만)
    valid = [m for m in report.models if m.error is None]
    if valid:
        lines.append(f"")
        lines.append(f"### 요약 통계")
        lines.append(f"")
        avg_load = sum(m.load_time_s for m in valid) / len(valid)
        avg_mem = sum(m.memory_mb for m in valid) / len(valid)
        avg_rtf = sum(m.rtf for m in valid) / len(valid) if any(m.rtf > 0 for m in valid) else 0.0
        fastest = min(valid, key=lambda m: m.rtf if m.rtf > 0 else float("inf"))
        lines.append(f"- 평균 로드 시간: {avg_load:.3f}s")
        lines.append(f"- 평균 GPU 메모리: {avg_mem:.1f} MB")
        if avg_rtf > 0:
            lines.append(f"- 평균 RTF: {avg_rtf:.4f}")
        lines.append(f"- 가장 빠른 모델: **{fastest.name}** (RTF {fastest.rtf:.4f})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """커맨드라인 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description="faster-whisper 모델 성능 벤치마크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "사용 예시:\n"
            "  python scripts/benchmark_models.py --audio sample.wav\n"
            "  python scripts/benchmark_models.py --models tiny,base,small --dry-run\n"
            "  python scripts/benchmark_models.py --audio sample.wav --output both --device cuda\n"
        ),
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="쉼표 구분 모델 목록 (기본: %(default)s)",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="벤치마크에 사용할 WAV/MP3 오디오 파일 경로",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="모델 로드 + 메모리만 측정 (전사 스킵)",
    )
    parser.add_argument(
        "--output",
        choices=["json", "markdown", "both"],
        default="both",
        help="결과 출력 형식 (기본: both)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="실행 디바이스 (기본: auto — CUDA 자동 감지)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="결과 파일 저장 경로 (확장자에 따라 .json/.md 자동 저장)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """벤치마크 메인 엔트리포인트."""
    args = parse_args(argv)

    # 모델 목록 파싱
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("[오류] 벤치마크할 모델이 없습니다.", file=sys.stderr)
        sys.exit(1)

    # 오디오 파일 검증
    if args.audio is not None and not args.audio.exists():
        print(f"[오류] 오디오 파일 없음: {args.audio}", file=sys.stderr)
        sys.exit(1)

    if args.audio is None and not args.dry_run:
        print("[경고] --audio 미지정. --dry-run 모드로 전환합니다.")
        args.dry_run = True

    # 디바이스 결정
    if args.device == "auto":
        device, compute_type = detect_device()
    elif args.device == "cuda":
        # CUDA 강제 지정 — 실패 시 CPU 폴백
        try:
            bootstrap_nvidia_dll_path()
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                device, compute_type = "cuda", "float16"
            else:
                print("[경고] CUDA 디바이스 미감지. CPU로 폴백합니다.")
                device, compute_type = "cpu", "int8"
        except Exception:
            print("[경고] CUDA 초기화 실패. CPU로 폴백합니다.")
            device, compute_type = "cpu", "int8"
    else:
        device, compute_type = "cpu", "int8"

    print(f"[설정] 디바이스: {device} ({compute_type})")
    print(f"[설정] 모델: {', '.join(models)}")
    print(f"[설정] dry-run: {args.dry_run}")

    # 벤치마크 실행
    report = run_benchmark(
        models=models,
        audio_path=args.audio,
        device=device,
        compute_type=compute_type,
        dry_run=args.dry_run,
    )

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"[완료] 전체 소요시간: {report.total_elapsed_s}초")
    print(f"{'='*60}\n")

    json_output = report_to_json(report)
    md_output = report_to_markdown(report)

    if args.output in ("json", "both"):
        print(json_output)

    if args.output in ("markdown", "both"):
        print()
        print(md_output)

    # 파일 저장
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        if args.save.suffix == ".json":
            args.save.write_text(json_output, encoding="utf-8")
            print(f"\n[저장] {args.save}")
        elif args.save.suffix == ".md":
            args.save.write_text(md_output, encoding="utf-8")
            print(f"\n[저장] {args.save}")
        else:
            # 확장자 없거나 기타 — 둘 다 저장
            json_path = args.save.with_suffix(".json")
            md_path = args.save.with_suffix(".md")
            json_path.write_text(json_output, encoding="utf-8")
            md_path.write_text(md_output, encoding="utf-8")
            print(f"\n[저장] {json_path}")
            print(f"[저장] {md_path}")


if __name__ == "__main__":
    main()
