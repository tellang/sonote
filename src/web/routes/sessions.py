from __future__ import annotations

from fastapi import APIRouter

from ... import _server_impl as _impl

router = APIRouter()

list_sessions = _impl.list_sessions
get_session = _impl.get_session
delete_session = _impl.delete_session
search_session = _impl.search_session
export_session = _impl.export_session_endpoint
new_session = _impl.new_session

router.add_api_route("/api/sessions", list_sessions, methods=["GET"])
router.add_api_route("/api/sessions/{session_id}", get_session, methods=["GET"])
router.add_api_route("/api/sessions/{session_id}", delete_session, methods=["DELETE"])
router.add_api_route("/api/sessions/{session_id}/search", search_session, methods=["GET"])
router.add_api_route(
    "/api/sessions/{session_id}/export",
    export_session,
    methods=["GET"],
    response_model=None,
)
router.add_api_route("/api/sessions/new", new_session, methods=["POST"])

__all__ = [
    "router",
    "list_sessions",
    "get_session",
    "delete_session",
    "search_session",
    "export_session",
    "new_session",
]
