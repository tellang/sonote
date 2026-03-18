"""브라우저 앱 모드 실행 어댑터."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _find_app_mode_browser() -> str | None:
    """Chrome 또는 Edge 실행파일 경로를 탐색한다. Edge 우선."""
    if sys.platform != "win32":
        return None

    candidates = [
        ("Microsoft\\Edge\\Application\\msedge.exe",),
        ("Google\\Chrome\\Application\\chrome.exe",),
    ]
    search_roots = []
    for env_var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        val = os.environ.get(env_var)
        if val:
            search_roots.append(Path(val))

    for rel_parts in candidates:
        for root in search_roots:
            full = root / rel_parts[0]
            if full.is_file():
                return str(full)

    for name in ("msedge", "chrome"):
        found = shutil.which(name)
        if found:
            return found

    return None


def _open_app_mode(url: str) -> bool:
    """--app=URL 플래그로 Chrome/Edge를 앱 모드로 실행한다. 성공 시 True."""
    browser_path = _find_app_mode_browser()
    if browser_path is None:
        return False
    try:
        subprocess.Popen(
            [browser_path, f"--app={url}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except OSError:
        return False
