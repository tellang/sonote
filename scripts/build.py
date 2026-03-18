#!/usr/bin/env python3
"""
sonote Windows EXE 빌드 스크립트

사용법:
    uv run python scripts/build.py              # 기본 onedir 빌드
    uv run python scripts/build.py --onefile    # 단일 EXE 빌드
    uv run python scripts/build.py --check      # 빌드 없이 의존성만 검증

출력:
    dist/sonote/ (onedir) 또는 dist/sonote.exe (onefile)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRY_SCRIPT = ROOT / "src" / "__main__.py"
SPEC_FILE = ROOT / "sonote.spec"
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"
PYPROJECT = ROOT / "pyproject.toml"
APP_NAME = "sonote"

REQUIRED_DEPS = [
    "PyInstaller",
    "pystray",
    "PIL",
    "faster_whisper",
    "sounddevice",
    "numpy",
    "uvicorn",
    "fastapi",
    "httpx",
    "yt_dlp",
    "watchdog",
]

HIDDEN_IMPORTS = [
    "pystray._win32",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "watchdog.observers.winapi",
]


def _read_version() -> str:
    """pyproject.toml에서 버전 문자열을 읽는다."""
    if not PYPROJECT.exists():
        return "0.0.0"
    text = PYPROJECT.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("version"):
            _, _, value = stripped.partition("=")
            return value.strip().strip('"').strip("'")
    return "0.0.0"


def _check_dependencies() -> list[str]:
    """필수 의존성 존재 여부를 검증하고 누락 목록을 반환한다."""
    missing: list[str] = []
    for pkg in REQUIRED_DEPS:
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    return missing


def _check_static_dir() -> bool:
    """static/ 디렉토리에 필수 파일이 있는지 확인한다."""
    static_dir = ROOT / "static"
    if not static_dir.is_dir():
        print("[경고] static/ 디렉토리가 없습니다")
        return False

    required_files = [
        "viewer.html",
        "settings.html",
        "speaker_profile.html",
    ]
    ok = True
    for filename in required_files:
        if not (static_dir / filename).exists():
            print(f"[경고] static/{filename} 파일이 없습니다")
            ok = False
    return ok


def _render_spec() -> str:
    """현재 프로젝트 기준 PyInstaller spec 파일 내용을 생성한다."""
    hidden_imports = ",\n    ".join(f'"{name}"' for name in HIDDEN_IMPORTS)
    return f'''# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import os

project_root = Path(r"{ROOT}")
build_mode = os.environ.get("SONOTE_BUILD_MODE", "onedir")
is_onefile = build_mode == "onefile"

hiddenimports = [
    {hidden_imports}
]

datas = [
    (str(project_root / "static"), "static"),
]

a = Analysis(
    [str(project_root / "src" / "__main__.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=["PyQt5", "PyQt6", "PySide2", "PySide6", "cefpython3", "webview", "pywebview"],
    noarchive=False,
)
pyz = PYZ(a.pure)

if is_onefile:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name="sonote",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="sonote",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name="sonote",
    )
'''


def _write_spec() -> Path:
    """최신 sonote.spec 파일을 생성하거나 갱신한다."""
    content = _render_spec()
    SPEC_FILE.write_text(content, encoding="utf-8")
    return SPEC_FILE


def _smoke_test(exe_path: Path) -> bool:
    """빌드된 EXE의 기본 실행 테스트 (--smoke-test)."""
    print("[검증] EXE 스모크 테스트 (--smoke-test) ...")
    try:
        result = subprocess.run(
            [str(exe_path), "--smoke-test"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print("[검증] 스모크 테스트 통과")
            return True

        print(f"[경고] --smoke-test 종료 코드: {result.returncode}")
        if result.stderr:
            for line in result.stderr.strip().splitlines()[:5]:
                print(f"  stderr: {line}")
        return False
    except subprocess.TimeoutExpired:
        print("[경고] 스모크 테스트 타임아웃 (30초)")
        return False
    except Exception as exc:
        print(f"[경고] 스모크 테스트 실패: {exc}")
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
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def cmd_check() -> int:
    """빌드 없이 의존성과 환경만 검증한다."""
    version = _read_version()
    spec_path = _write_spec()

    print(f"[검증] {APP_NAME} v{version}")
    print(f"[검증] Python: {sys.version}")
    print(f"[검증] 프로젝트 루트: {ROOT}")
    print(f"[검증] 엔트리포인트: {ENTRY_SCRIPT}")
    print(f"[검증] spec 파일 갱신: {spec_path}")
    print()

    _check_static_dir()

    print()
    print("[검증] 필수 의존성 확인:")
    missing = _check_dependencies()
    for pkg in REQUIRED_DEPS:
        status = "실패" if pkg in missing else "OK"
        print(f"  [{status}] {pkg}")

    if missing:
        print(f"\n[오류] 누락된 의존성 {len(missing)}개: {', '.join(missing)}")
        print("  설치: uv sync --extra desktop && uv pip install pyinstaller")
        return 1

    print("\n[완료] 모든 의존성이 확인되었습니다. 빌드 준비 완료.")
    return 0


def cmd_build(onefile: bool = False) -> int:
    """PyInstaller 빌드를 실행한다."""
    version = _read_version()
    spec_path = _write_spec()
    mode = "onefile" if onefile else "onedir"

    print(f"[빌드] {APP_NAME} v{version} ({mode})")
    print(f"[빌드] Python: {sys.version}")
    print(f"[빌드] 엔트리포인트: {ENTRY_SCRIPT}")
    print(f"[빌드] spec: {spec_path}")
    print(f"[빌드] 출력: {DIST_DIR}")
    print()

    _check_static_dir()

    print("[빌드] 의존성 검증 중 ...")
    missing = _check_dependencies()
    if missing:
        print(f"[오류] 누락된 필수 의존성: {', '.join(missing)}")
        print("  설치: uv sync --extra desktop && uv pip install pyinstaller")
        return 1
    print("[빌드] 의존성 검증 통과")
    print()

    env = os.environ.copy()
    env["SONOTE_BUILD_MODE"] = mode

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(spec_path),
        "--clean",
        "--noconfirm",
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR),
    ]

    print(f"[빌드] 실행: {' '.join(command)}")
    print()

    result = subprocess.run(command, cwd=str(ROOT), env=env)
    if result.returncode != 0:
        print(f"\n[오류] PyInstaller 빌드 실패 (exit code: {result.returncode})")
        print("[진단] 일반적인 원인:")
        print("  - pywebview/pystray 미설치")
        print("  - hidden import 누락: sonote.spec 확인")
        print("  - 바이러스 백신이 빌드 파일 차단")
        print("  - 디스크 공간 부족")
        return result.returncode

    print()
    print("=" * 60)

    if onefile:
        exe_path = DIST_DIR / f"{APP_NAME}.exe"
        if not exe_path.exists():
            print(f"[오류] EXE 파일이 생성되지 않았습니다: {exe_path}")
            return 1

        print(f"[완료] 아티팩트: {exe_path}")
        print(f"[완료] 크기: {_format_size(exe_path.stat().st_size)}")
        _smoke_test(exe_path)
        print("=" * 60)
        return 0

    out_dir = DIST_DIR / APP_NAME
    exe_path = out_dir / f"{APP_NAME}.exe"
    if not out_dir.is_dir() or not exe_path.exists():
        print(f"[오류] 빌드 출력이 생성되지 않았습니다: {out_dir}")
        return 1

    total_size = _dir_total_size(out_dir)
    file_count = sum(1 for file_path in out_dir.rglob("*") if file_path.is_file())
    print(f"[완료] 아티팩트: {out_dir}/")
    print(f"[완료] EXE 크기: {_format_size(exe_path.stat().st_size)}")
    print(f"[완료] 디렉토리 총 크기: {_format_size(total_size)}")
    print(f"[완료] 파일 수: {file_count}")
    # PyInstaller onedir 빌드 시 static은 _internal/static/에 위치
    static_check_path = out_dir / "_internal" / "static" / "viewer.html"
    # 레거시 경로도 확인 (직접 static/ 하위)
    static_legacy_path = out_dir / "static" / "viewer.html"
    if static_check_path.exists() or static_legacy_path.exists():
        print("[완료] static 리소스 포함 확인")
    else:
        print("[경고] static 리소스가 누락되었습니다")
    _smoke_test(exe_path)
    print("=" * 60)
    return 0


def main() -> int:
    """빌드 스크립트 CLI 엔트리포인트."""
    parser = argparse.ArgumentParser(description="sonote PyInstaller 빌드 스크립트")
    parser.add_argument("--check", action="store_true", help="빌드 없이 의존성만 검증")
    parser.add_argument("--onefile", action="store_true", help="단일 EXE 빌드")
    parser.add_argument(
        "--onedir",
        action="store_true",
        default=True,
        help="디렉토리 빌드 (기본값)",
    )

    args = parser.parse_args()
    if args.check:
        return cmd_check()
    return cmd_build(onefile=args.onefile)


if __name__ == "__main__":
    sys.exit(main())
