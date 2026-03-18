from __future__ import annotations

from fastapi import APIRouter

from ... import _server_impl as _impl

router = APIRouter()

list_devices = _impl.list_devices
switch_device = _impl.switch_device
list_audio_devices = _impl.list_audio_devices

router.add_api_route("/devices", list_devices, methods=["GET"])
router.add_api_route("/switch-device", switch_device, methods=["POST"])
router.add_api_route("/api/audio-devices", list_audio_devices, methods=["GET"])

__all__ = [
    "router",
    "list_devices",
    "switch_device",
    "list_audio_devices",
]
