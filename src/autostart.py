"""Windows 자동 시작 레지스트리 관리."""

from __future__ import annotations

import re
import sys
from pathlib import Path, PureWindowsPath
from typing import Final

try:
    import winreg as _WINREG
except ImportError:  # pragma: no cover - 비 Windows 환경 폴백
    _WINREG = None

APP_NAME: Final[str] = "sonote"
RUN_KEY_PATH: Final[str] = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"
_PYTHON_SUFFIXES: Final[tuple[str, ...]] = (".py", ".pyw")


class AutostartError(RuntimeError):
    """자동 시작 등록 처리 중 발생한 오류."""


def get_exe_path() -> str:
    """현재 실행 파일 경로를 반환한다."""
    if getattr(sys, "frozen", False):
        return _resolve_current_path(sys.executable)

    argv0 = Path(sys.argv[0]).expanduser() if sys.argv and sys.argv[0] else None
    if argv0 and argv0.suffix.lower() in {".exe", *_PYTHON_SUFFIXES}:
        return _resolve_current_path(str(argv0))

    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", "")
    if main_file:
        return _resolve_current_path(main_file)

    return _resolve_current_path(sys.executable)


def register() -> None:
    """현재 실행 경로를 Windows 자동 시작에 등록한다."""
    _ensure_windows()
    command = _build_launch_command()

    try:
        with _WINREG.CreateKeyEx(
            _WINREG.HKEY_CURRENT_USER,
            RUN_KEY_PATH,
            0,
            _WINREG.KEY_SET_VALUE,
        ) as key:
            _WINREG.SetValueEx(key, APP_NAME, 0, _WINREG.REG_SZ, command)
    except PermissionError as exc:
        raise AutostartError("자동 시작 등록 권한이 부족합니다.") from exc
    except OSError as exc:
        raise AutostartError(f"자동 시작 등록에 실패했습니다: {exc}") from exc


def unregister() -> None:
    """Windows 자동 시작 등록을 제거한다."""
    _ensure_windows()

    try:
        with _WINREG.OpenKey(
            _WINREG.HKEY_CURRENT_USER,
            RUN_KEY_PATH,
            0,
            _WINREG.KEY_SET_VALUE,
        ) as key:
            _WINREG.DeleteValue(key, APP_NAME)
    except FileNotFoundError:
        return
    except PermissionError as exc:
        raise AutostartError("자동 시작 해제 권한이 부족합니다.") from exc
    except OSError as exc:
        raise AutostartError(f"자동 시작 해제에 실패했습니다: {exc}") from exc


def is_registered() -> bool:
    """현재 실행 경로가 자동 시작에 등록되어 있는지 확인한다."""
    _ensure_windows()

    try:
        with _WINREG.OpenKey(
            _WINREG.HKEY_CURRENT_USER,
            RUN_KEY_PATH,
            0,
            _WINREG.KEY_READ,
        ) as key:
            value, _ = _WINREG.QueryValueEx(key, APP_NAME)
    except FileNotFoundError:
        return False
    except PermissionError as exc:
        raise AutostartError("자동 시작 상태를 확인할 권한이 부족합니다.") from exc
    except OSError as exc:
        raise AutostartError(f"자동 시작 상태 확인에 실패했습니다: {exc}") from exc

    return _normalize_command(value) == _normalize_command(_build_launch_command())


def _ensure_windows() -> None:
    """Windows 환경과 winreg 가용성을 확인한다."""
    if sys.platform != "win32" or _WINREG is None:
        raise AutostartError("Windows에서만 자동 시작 기능을 사용할 수 있습니다.")


def _build_launch_command() -> str:
    """자동 시작 시 실행할 명령 문자열을 생성한다."""
    exe_path = Path(get_exe_path())

    # 개발 모드에서는 python 인터프리터로 cli 스크립트를 실행한다.
    if exe_path.suffix.lower() in _PYTHON_SUFFIXES:
        python_path = _resolve_current_path(sys.executable)
        return _join_command(str(python_path), str(exe_path), "meeting")

    # EXE 또는 엔트리포인트 런처는 그대로 실행하고 meeting 모드를 붙인다.
    return _join_command(str(exe_path), "meeting")


def _join_command(*parts: str) -> str:
    """공백이 있는 인자를 안전하게 감싸 하나의 명령 문자열로 합친다."""
    return " ".join(_quote_argument(part) for part in parts)


def _quote_argument(value: str) -> str:
    """Windows 명령행 인자를 따옴표로 감싼다."""
    escaped = value.replace('"', '\\"')
    return f'"{escaped}"'


def _normalize_command(value: str) -> str:
    """비교를 위해 공백과 대소문자를 정규화한다."""
    collapsed = re.sub(r"\s+", " ", value.strip())
    return collapsed.casefold()


def _resolve_current_path(value: str) -> str:
    """현재 플랫폼에 맞춰 실행 경로 문자열을 정규화한다."""
    if _looks_like_windows_path(value):
        return str(PureWindowsPath(value))

    return str(Path(value).expanduser().resolve())


def _looks_like_windows_path(value: str) -> bool:
    """Windows 스타일 경로인지 판별한다."""
    return (len(value) >= 2 and value[1] == ":") or ("\\" in value)
