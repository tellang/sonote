"""sonote 데스크톱 런처 thin wrapper."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import webbrowser

from .desktop_app.controller import DesktopController

_BETA_ENV_KEY = "SONOTE_BETA"
_SINGLETON_MUTEX_NAME = "sonote_desktop_singleton"
_ERROR_ALREADY_EXISTS = 183
_SINGLETON_MUTEX_HANDLE: int | None = None


def _resolve_beta_mode(beta_mode: bool = False) -> bool:
    return bool(beta_mode or os.getenv(_BETA_ENV_KEY) == "1")


def _configure_beta_mode(beta_mode: bool) -> None:
    if beta_mode:
        os.environ[_BETA_ENV_KEY] = "1"
    try:
        from .postprocess import set_beta_mode

        set_beta_mode(beta_mode)
    except Exception:
        pass


def _instance_file_path() -> Path:
    return Path.home() / ".sonote" / "instance.json"


def _read_instance_port() -> int | None:
    path = _instance_file_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    port = payload.get("port")
    if isinstance(port, int) and 1 <= port <= 65535:
        return port
    return None


def _write_instance_port(port: int) -> None:
    if port <= 0:
        return

    path = _instance_file_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"port": int(port)}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def _clear_instance_file() -> None:
    try:
        _instance_file_path().unlink(missing_ok=True)
    except OSError:
        pass


def _check_singleton() -> str | None:
    """이미 실행 중인 sonote 인스턴스가 있으면 그 URL을 반환, 없으면 None."""
    import ctypes

    if os.name != "nt":
        return None

    try:
        handle = ctypes.windll.kernel32.CreateMutexW(None, False, _SINGLETON_MUTEX_NAME)
        if handle:
            global _SINGLETON_MUTEX_HANDLE
            _SINGLETON_MUTEX_HANDLE = handle

        if ctypes.windll.kernel32.GetLastError() == _ERROR_ALREADY_EXISTS:
            port = _read_instance_port()
            if port is not None:
                return f"http://127.0.0.1:{port}"
            return "already_running"
    except Exception:
        pass
    return None


def run_desktop(*, host: str = "127.0.0.1", port: int = 0, beta_mode: bool = False) -> None:
    """CLI에서 호출하는 데스크톱 런처."""
    resolved_beta_mode = _resolve_beta_mode(beta_mode)
    _configure_beta_mode(resolved_beta_mode)

    existing = _check_singleton()
    if existing and existing != "already_running":
        print(f"sonote가 이미 실행 중입니다: {existing}")
        webbrowser.open(existing)
        return
    if existing == "already_running":
        print("sonote가 이미 실행 중입니다.")
        return

    controller = DesktopController(host=host, port=port, beta_mode=resolved_beta_mode)
    try:
        if os.name == "nt":
            _write_instance_port(controller.port)
        controller.run()
    finally:
        if os.name == "nt":
            _clear_instance_file()


def main() -> None:
    """직접 실행용 엔트리포인트."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--beta", action="store_true")
    args, _ = parser.parse_known_args()
    if args.smoke_test:
        print("sonote desktop smoke ok")
        return
    run_desktop(beta_mode=args.beta)


if __name__ == "__main__":
    main()
