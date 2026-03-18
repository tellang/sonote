from __future__ import annotations

from fastapi import APIRouter

from ... import _server_impl as _impl

router = APIRouter()

speaker_profile_page = _impl.speaker_profile_page
list_speakers = _impl.list_speakers
enroll_speaker = _impl.enroll_speaker
get_profiles = _impl.get_profiles
create_profile = _impl.create_profile
update_profile = _impl.update_profile
delete_profile = _impl.delete_profile
list_unknown_speakers = _impl.list_unknown_speakers
auto_register_speaker = _impl.auto_register_speaker
delete_unknown_speaker = _impl.delete_unknown_speaker

router.add_api_route("/speaker-profile", speaker_profile_page, methods=["GET"])
router.add_api_route("/speakers", list_speakers, methods=["GET"])
router.add_api_route("/enroll", enroll_speaker, methods=["POST"])
router.add_api_route("/api/profiles", get_profiles, methods=["GET"])
router.add_api_route("/api/profiles", create_profile, methods=["POST"])
router.add_api_route("/api/profiles/{name}", update_profile, methods=["PUT"])
router.add_api_route("/api/profiles/{name}", delete_profile, methods=["DELETE"])
router.add_api_route("/api/speakers/unknown", list_unknown_speakers, methods=["GET"])
router.add_api_route("/api/speakers/auto-register", auto_register_speaker, methods=["POST"])
router.add_api_route("/api/speakers/unknown/{speaker_id}", delete_unknown_speaker, methods=["DELETE"])

__all__ = [
    "router",
    "speaker_profile_page",
    "list_speakers",
    "enroll_speaker",
    "get_profiles",
    "create_profile",
    "update_profile",
    "delete_profile",
    "list_unknown_speakers",
    "auto_register_speaker",
    "delete_unknown_speaker",
]
