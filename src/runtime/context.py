"""실행 컨텍스트 판별 유틸리티."""

from __future__ import annotations

from enum import Enum
import sys
from pathlib import Path


class RunMode(Enum):
    """앱 실행 모드."""

    DEV = "dev"
    EXE_ONEDIR = "exe_onedir"
    EXE_ONEFILE = "exe_onefile"


def _resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def is_frozen() -> bool:
    """PyInstaller frozen 실행 여부를 반환한다."""
    return bool(getattr(sys, "frozen", False))


def get_run_mode() -> RunMode:
    """현재 실행 모드를 반환한다."""
    if not is_frozen():
        return RunMode.DEV

    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return RunMode.EXE_ONEDIR

    exe_dir = _resolve(Path(sys.executable).parent)
    bundle_dir = _resolve(Path(meipass))
    if bundle_dir in {exe_dir, _resolve(exe_dir / "_internal")}:
        return RunMode.EXE_ONEDIR
    return RunMode.EXE_ONEFILE


def get_bundle_dir() -> Path:
    """번들 리소스 디렉토리를 반환한다."""
    if not is_frozen():
        return get_project_root()

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return _resolve(Path(sys.executable).parent)


def get_project_root() -> Path:
    """프로젝트(또는 exe) 루트 디렉토리를 반환한다."""
    if is_frozen():
        return _resolve(Path(sys.executable).parent)
    return _resolve(Path(__file__).parent.parent.parent)
