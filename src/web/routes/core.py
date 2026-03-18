from __future__ import annotations

from fastapi import APIRouter

from ... import _server_impl as _impl

router = APIRouter()

index = _impl.index
settings_page = _impl.settings_page
get_settings = _impl.get_settings
save_settings = _impl.save_settings
history = _impl.history
status = _impl.status
toggle_pause = _impl.toggle_pause
shutdown = _impl.shutdown

router.add_api_route("/", index, methods=["GET"])
router.add_api_route("/settings", settings_page, methods=["GET"])
router.add_api_route("/api/settings", get_settings, methods=["GET"])
router.add_api_route("/api/settings", save_settings, methods=["POST"])
router.add_api_route("/history", history, methods=["GET"])
router.add_api_route("/status", status, methods=["GET"])
router.add_api_route("/toggle-pause", toggle_pause, methods=["POST"])
router.add_api_route("/shutdown", shutdown, methods=["POST"])

__all__ = [
    "router",
    "index",
    "settings_page",
    "get_settings",
    "save_settings",
    "history",
    "status",
    "toggle_pause",
    "shutdown",
]
