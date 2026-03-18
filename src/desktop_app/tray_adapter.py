"""DesktopController의 시스템 트레이 연동 어댑터."""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from ..server import is_paused
from ..tray import MeetingTray


def start_tray(controller: Any) -> None:
    """트레이 아이콘을 시작한다."""
    if controller.tray is not None:
        return

    controller.tray = MeetingTray(
        port=controller.port,
        on_shutdown=controller.shutdown,
        on_open_viewer=controller._open_browser,
        on_open_settings=lambda: controller._open_browser(f"{controller.base_url}/settings"),
        on_toggle_recording=controller.toggle_recording,
        get_toggle_label=lambda paused: "녹음 시작" if paused else "녹음 중지",
    )
    controller.tray.start()
    controller.tray.update_status(paused=is_paused(), recording=True)


def start_status_poller(
    controller: Any,
    *,
    fetch_json: Callable[[str], dict[str, Any]],
    format_elapsed: Callable[[int], str],
) -> threading.Thread:
    """서버 상태를 읽어 트레이 툴팁/아이콘에 반영한다."""
    thread = threading.Thread(
        target=_status_poller_loop,
        kwargs={
            "controller": controller,
            "fetch_json": fetch_json,
            "format_elapsed": format_elapsed,
        },
        daemon=True,
        name="sonote-desktop-status",
    )
    thread.start()
    return thread


def _status_poller_loop(
    controller: Any,
    *,
    fetch_json: Callable[[str], dict[str, Any]],
    format_elapsed: Callable[[int], str],
) -> None:
    """주기적으로 서버 상태를 조회한다."""
    while not controller.stop_event.wait(1.0):
        tray = controller.tray
        if tray is None:
            continue
        try:
            status = fetch_json(f"{controller.base_url}/status")
        except Exception:
            continue

        speakers = status.get("speakers") or []
        tray.update_status(
            elapsed=format_elapsed(int(status.get("elapsed", 0))),
            segments=int(status.get("segments", 0)),
            speakers=len(speakers) if isinstance(speakers, list) else 0,
            paused=bool(status.get("paused", False)),
            recording=True,
        )
