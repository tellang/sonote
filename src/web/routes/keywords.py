from __future__ import annotations

from fastapi import APIRouter

from ... import _server_impl as _impl

router = APIRouter()

add_keyword = _impl.add_keyword
remove_keyword = _impl.remove_keyword
list_keywords = _impl.list_keywords

router.add_api_route("/add-keyword", add_keyword, methods=["POST"])
router.add_api_route("/remove-keyword", remove_keyword, methods=["POST"])
router.add_api_route("/remove-keyword", remove_keyword, methods=["DELETE"])
router.add_api_route("/keywords", list_keywords, methods=["GET"])

__all__ = [
    "router",
    "add_keyword",
    "remove_keyword",
    "list_keywords",
]
