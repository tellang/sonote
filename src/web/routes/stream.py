from __future__ import annotations

from fastapi import APIRouter

from ... import _server_impl as _impl

router = APIRouter()

stream = _impl.stream
ws_transcribe = _impl.ws_transcribe
push_segment = _impl.push_segment_endpoint
load_transcript = _impl.load_transcript_file

router.add_api_route("/stream", stream, methods=["GET"])
router.add_api_websocket_route("/ws/transcribe", ws_transcribe)
router.add_api_route("/api/push-segment", push_segment, methods=["POST"])
router.add_api_route("/api/load-transcript", load_transcript, methods=["POST"])

__all__ = [
    "router",
    "stream",
    "ws_transcribe",
    "push_segment",
    "load_transcript",
]
