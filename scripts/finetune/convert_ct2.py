"""학습된 Whisper 모델 → CTranslate2 변환 (faster-whisper 호환).

faster-whisper는 CTranslate2 포맷만 지원하므로,
HuggingFace 포맷 모델을 변환해야 sonote에서 사용 가능하다.

사용법:
    # LoRA 병합 모델 변환
    python convert_ct2.py --model-dir ./whisper-ko-sonote/merged

    # 출력 경로 지정
    python convert_ct2.py --model-dir ./whisper-ko-sonote/merged --output-dir ./whisper-ko-sonote/ct2

    # int8 양자화 (크기 절반, 약간의 품질 손실)
    python convert_ct2.py --model-dir ./whisper-ko-sonote/merged --quantization int8
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def convert_hf_to_ct2(
    model_dir: Path,
    output_dir: Path,
    quantization: str = "float16",
    force: bool = False,
) -> Path:
    """HuggingFace Whisper 모델 → CTranslate2 변환.

    Args:
        model_dir: HuggingFace 모델 디렉토리 (merged/)
        output_dir: CT2 출력 디렉토리
        quantization: float16, float32, int8, int8_float16
        force: 기존 출력 덮어쓰기

    Returns:
        변환된 CT2 모델 경로
    """
    if output_dir.exists() and not force:
        print(f"[스킵] 이미 존재: {output_dir}")
        print("  덮어쓰기: --force 옵션 사용")
        return output_dir

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)

    print(f"=== CTranslate2 변환 ===")
    print(f"입력: {model_dir}")
    print(f"출력: {output_dir}")
    print(f"양자화: {quantization}")

    # ct2-opus-encoder-decoder-converter 사용
    cmd = [
        sys.executable, "-m", "ctranslate2.converters.transformers",
        "--model", str(model_dir),
        "--output_dir", str(output_dir),
        "--quantization", quantization,
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
    ]

    # 대안: ct2-transformers-converter CLI
    ct2_converter = shutil.which("ct2-transformers-converter")
    if ct2_converter:
        cmd = [
            ct2_converter,
            "--model", str(model_dir),
            "--output_dir", str(output_dir),
            "--quantization", quantization,
            "--copy_files", "tokenizer.json", "preprocessor_config.json",
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[오류] 변환 실패: {e.stderr}")
        print("\n대안 방법:")
        print("  pip install ctranslate2")
        print(f"  ct2-transformers-converter --model {model_dir} --output_dir {output_dir} --quantization {quantization}")
        sys.exit(1)
    except FileNotFoundError:
        print("[오류] ctranslate2가 설치되어 있지 않습니다.")
        print("  pip install ctranslate2")
        sys.exit(1)

    # 변환 검증
    model_bin = output_dir / "model.bin"
    if not model_bin.exists():
        print("[오류] 변환된 model.bin이 없습니다.")
        sys.exit(1)

    size_mb = model_bin.stat().st_size / 1024 / 1024
    print(f"\n[완료] {output_dir}")
    print(f"  model.bin: {size_mb:.0f} MB")
    return output_dir


def verify_model(ct2_dir: Path) -> None:
    """변환된 모델을 faster-whisper로 로드 테스트."""
    print("\n=== faster-whisper 호환성 검증 ===")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[스킵] faster-whisper가 설치되어 있지 않습니다.")
        return

    try:
        model = WhisperModel(str(ct2_dir), device="cpu", compute_type="int8")
        print(f"[성공] faster-whisper 로드 완료: {ct2_dir}")

        # 더미 오디오로 추론 테스트
        import numpy as np
        dummy_audio = np.zeros(16000 * 3, dtype=np.float32)  # 3초 무음
        segments, info = model.transcribe(dummy_audio, language="ko")
        segments_list = list(segments)
        print(f"[성공] 추론 테스트 통과 (더미 3초, {len(segments_list)}개 세그먼트)")

        del model

    except Exception as exc:
        print(f"[실패] faster-whisper 로드 실패: {exc}")
        print("  모델 디렉토리 구조를 확인하세요.")


def print_usage_guide(ct2_dir: Path) -> None:
    """sonote 통합 가이드 출력."""
    print(f"""
=== sonote 통합 가이드 ===

1. 모델 경로를 sonote에 전달:

   # CLI
   sonote live --model {ct2_dir}

   # 또는 환경변수
   export SONOTE_MODEL={ct2_dir}
   sonote live

2. 코드에서 직접 사용:

   from src.whisper_worker import WhisperWorker
   worker = WhisperWorker(model_id="{ct2_dir}")
   worker.start()

3. sonote config에서 기본 모델 변경:
   settings.html → 모델 경로를 '{ct2_dir}'로 설정
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper HF → CTranslate2 변환")
    parser.add_argument(
        "--model-dir", type=Path, required=True,
        help="HuggingFace 모델 디렉토리 (merged/)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="CT2 출력 디렉토리 (기본: {model-dir}/../ct2)",
    )
    parser.add_argument(
        "--quantization", type=str, default="float16",
        choices=["float16", "float32", "int8", "int8_float16"],
        help="양자화 타입",
    )
    parser.add_argument("--force", action="store_true", help="기존 출력 덮어쓰기")
    parser.add_argument("--skip-verify", action="store_true", help="검증 건너뛰기")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    if not model_dir.exists():
        print(f"[오류] 모델 디렉토리 없음: {model_dir}")
        sys.exit(1)

    output_dir = args.output_dir or (model_dir.parent / "ct2")
    output_dir = output_dir.resolve()

    ct2_dir = convert_hf_to_ct2(model_dir, output_dir, args.quantization, args.force)

    if not args.skip_verify:
        verify_model(ct2_dir)

    print_usage_guide(ct2_dir)


if __name__ == "__main__":
    main()
