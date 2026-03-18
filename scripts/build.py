#!/usr/bin/env python3
"""
media-transcriber Windows EXE 빌드 스크립트

사용법:
    uv run python scripts/build.py              # 기본 onedir 빌드
    uv run python scripts/build.py --onefile    # 단일 EXE 빌드
    uv run python scripts/build.py --check      # 빌드 없이 의존성만 검증

출력:
    dist/media-transcriber/ (onedir) 또는 dist/media-transcriber.exe (onefile)
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC_FILE = ROOT / "media-transcriber.spec"
DIST_DIR = ROOT / "dist"
PYPROJECT = ROOT / "pyproject.toml"

# 빌드에 필수인 패키지 목록 (import 이름)
REQUIRED_DEPS = [
    "PyInstaller",
    "faster_whisper",
    "sounddevice",
    "numpy",
    "uvicorn",
    "fastapi",
    "httpx",
    "yt_dlp",
    "watchdog",
]


def _read_version() -> str:
    """pyproject.toml에서 버전 문자열을 읽는다."""
    if not PYPROJECT.exists():
        return "0.0.0"
    text = PYPROJECT.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("version"):
            # version = "1.1.0b1" 형태 파싱
            _, _, value = stripped.partition("=")
            return value.strip().strip('"').strip("'")
    return "0.0.0"


def _check_dependencies() -> list[str]:
    """필수 의존성 존재 여부를 검증하고 누락 목록을 반환한다."""
    missing: list[str] = []
    for pkg in REQUIRED_DEPS:
        spec = importlib.util.find_spec(pkg)
        if spec is None:
            missing.append(pkg)
    return missing


def _check_static_dir() -> bool:
    """static/ 디렉토리에 필수 파일이 있는지 확인한다."""
    static_dir = ROOT / "static"
    if not static_dir.is_dir():
        print("[경고] static/ 디렉토리가 없습니다")
        return False
    required_files = ["viewer.html"]
    ok = True
    for fname in required_files:
        if not (static_dir / fname).exists():
            print(f"[경고] static/{fname} 파일이 없습니다")
            ok = False
    return ok


def _smoke_test(exe_path: Path) -> bool:
    """빌드된 EXE의 기본 실행 테스트 (--help)."""
    print("[검증] EXE 스모크 테스트 (--help) ...")
    try:
        result = subprocess.run(
            [str(exe_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print("[검증] 스모크 테스트 통과")
            return True
        else:
            print(f"[경고] --help 종료 코드: {result.returncode}")
            if result.stderr:
                # 첫 5줄만 출력
                lines = result.stderr.strip().splitlines()[:5]
                for line in lines:
                    print(f"  stderr: {line}")
            return False
    except subprocess.TimeoutExpired:
        print("[경고] 스모크 테스트 타임아웃 (30초)")
        return False
    except Exception as e:
        print(f"[경고] 스모크 테스트 실패: {e}")
        return False


def _format_size(size_bytes: int) -> str:
    """바이트 크기를 사람이 읽기 쉬운 형식으로 변환한다."""
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 ** 3):.2f} GB"
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def _dir_total_size(path: Path) -> int:
    """디렉토리의 총 크기(바이트)를 계산한다."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def cmd_check() -> int:
    """빌드 없이 의존성과 환경만 검증한다."""
    version = _read_version()
    print(f"[검증] media-transcriber v{version}")
    print(f"[검증] Python: {sys.version}")
    print(f"[검증] 프로젝트 루트: {ROOT}")
    print()

    # spec 파일 존재
    if SPEC_FILE.exists():
        print(f"[OK] spec 파일: {SPEC_FILE}")
    else:
        print(f"[실패] spec 파일 없음: {SPEC_FILE}")

    # static/ 디렉토리
    _check_static_dir()

    # 의존성 검증
    print()
    print("[검증] 필수 의존성 확인:")
    missing = _check_dependencies()
    for pkg in REQUIRED_DEPS:
        status = "실패" if pkg in missing else "OK"
        print(f"  [{status}] {pkg}")

    if missing:
        print(f"\n[오류] 누락된 의존성 {len(missing)}개: {', '.join(missing)}")
        print("  설치: uv pip install " + " ".join(missing))
        return 1

    print("\n[완료] 모든 의존성이 확인되었습니다. 빌드 준비 완료.")
    return 0


def cmd_build(onefile: bool = False) -> int:
    """PyInstaller 빌드를 실행한다."""
    version = _read_version()
    mode = "onefile" if onefile else "onedir"
    print(f"[빌드] media-transcriber v{version} ({mode})")
    print(f"[빌드] Python: {sys.version}")
    print(f"[빌드] spec: {SPEC_FILE}")
    print(f"[빌드] 출력: {DIST_DIR}")
    print()

    # spec 파일 확인
    if not SPEC_FILE.exists():
        print(f"[오류] spec 파일을 찾을 수 없습니다: {SPEC_FILE}")
        return 1

    # static/ 확인
    _check_static_dir()

    # 사전 의존성 검증
    print("[빌드] 의존성 검증 중 ...")
    missing = _check_dependencies()
    if missing:
        print(f"[오류] 누락된 필수 의존성: {', '.join(missing)}")
        print("  설치: uv pip install " + " ".join(missing))
        return 1
    print("[빌드] 의존성 검증 통과")
    print()

    # 환경 변수로 빌드 모드 전달 (spec 파일에서 읽음)
    import os

    env = os.environ.copy()
    env["MT_BUILD_MODE"] = mode
    env["MT_VERSION"] = version

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(SPEC_FILE),
        "--clean",
        "--noconfirm",
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(ROOT / "build"),
    ]

    print(f"[빌드] 실행: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(ROOT), env=env)

    if result.returncode != 0:
        print(f"\n[오류] PyInstaller 빌드 실패 (exit code: {result.returncode})")
        print("[진단] 일반적인 원인:")
        print("  - PyInstaller 미설치: uv pip install pyinstaller")
        print("  - hidden import 누락: spec 파일의 hiddenimports 확인")
        print("  - 바이러스 백신이 빌드 파일 차단: 예외 등록 필요")
        print("  - 디스크 공간 부족: dist/, build/ 정리 후 재시도")
        return result.returncode

    # 빌드 결과 확인
    print()
    print("=" * 60)

    if onefile:
        exe_path = DIST_DIR / "media-transcriber.exe"
        if exe_path.exists():
            size = exe_path.stat().st_size
            print(f"[완료] 아티팩트: {exe_path}")
            print(f"[완료] 크기: {_format_size(size)}")
            # 스모크 테스트
            _smoke_test(exe_path)
        else:
            print(f"[오류] EXE 파일이 생성되지 않았습니다: {exe_path}")
            return 1
    else:
        out_dir = DIST_DIR / "media-transcriber"
        exe_path = out_dir / "media-transcriber.exe"
        if out_dir.is_dir() and exe_path.exists():
            total_size = _dir_total_size(out_dir)
            exe_size = exe_path.stat().st_size
            file_count = sum(1 for f in out_dir.rglob("*") if f.is_file())
            print(f"[완료] 아티팩트: {out_dir}/")
            print(f"[완료] EXE 크기: {_format_size(exe_size)}")
            print(f"[완료] 디렉토리 총 크기: {_format_size(total_size)}")
            print(f"[완료] 파일 수: {file_count}")
            # static/ 포함 확인
            if (out_dir / "static" / "viewer.html").exists():
                print("[완료] static/viewer.html 포함 확인")
            else:
                print("[경고] static/viewer.html 미포함 — spec datas 설정 확인 필요")
            # 스모크 테스트
            _smoke_test(exe_path)
        else:
            print(f"[오류] 빌드 출력이 생성되지 않았습니다: {out_dir}")
            return 1

    print("=" * 60)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="media-transcriber PyInstaller 빌드 스크립트",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="빌드 없이 의존성만 검증",
    )
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="단일 EXE 빌드 (기본: onedir)",
    )
    parser.add_argument(
        "--onedir",
        action="store_true",
        default=True,
        help="디렉토리 빌드 (기본값)",
    )

    args = parser.parse_args()

    if args.check:
        return cmd_check()

    # --onefile이 명시되면 onefile 모드, 아니면 onedir
    return cmd_build(onefile=args.onefile)


if __name__ == "__main__":
    sys.exit(main())
