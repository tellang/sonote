"""미디어 믹스 제어 모듈 — 녹음 중 다른 앱 오디오 음소거/복원."""

from __future__ import annotations

import logging
import platform
from typing import Any

logger = logging.getLogger(__name__)

# 음소거 대상 앱 키워드 (소문자)
_TARGET_APPS: set[str] = {
    "spotify",
    "chrome",
    "firefox",
    "vlc",
    "musicbee",
    "foobar",
    "foobar2000",
    "itunes",
    "edge",
    "msedge",
    "brave",
    "opera",
}

# 음소거에서 제외할 프로세스 키워드 (소문자)
_EXCLUDE_APPS: set[str] = {
    "sonote",
    "python",
    "pythonw",
}

# 음소거 전 상태 보관 {pid: (name, was_muted, volume_level)}
_saved_states: dict[int, tuple[str, bool, float]] = {}

__all__ = [
    "is_available",
    "get_media_sessions",
    "mute_media_apps",
    "restore_media_state",
]


def _is_windows() -> bool:
    """Windows 플랫폼 여부를 반환한다."""
    return platform.system() == "Windows"


def _try_import_pycaw() -> bool:
    """pycaw 임포트 가능 여부를 확인한다."""
    try:
        import pycaw  # noqa: F401
        return True
    except ImportError:
        return False


def is_available() -> bool:
    """pycaw가 설치되어 있고 Windows인지 확인한다."""
    if not _is_windows():
        return False
    return _try_import_pycaw()


def _is_target_app(name: str) -> bool:
    """음소거 대상 앱인지 확인한다."""
    lower = name.lower()
    # 제외 앱 확인
    for exc in _EXCLUDE_APPS:
        if exc in lower:
            return False
    # 대상 앱 확인
    for target in _TARGET_APPS:
        if target in lower:
            return True
    return False


def _get_audio_sessions_raw() -> list[Any]:
    """pycaw를 사용하여 활성 오디오 세션 목록을 가져온다."""
    from comtypes import CoInitialize, CoUninitialize
    from pycaw.pycaw import AudioUtilities

    CoInitialize()
    try:
        sessions = AudioUtilities.GetAllSessions()
        return list(sessions)
    finally:
        CoUninitialize()


def get_media_sessions() -> list[dict[str, Any]]:
    """활성 오디오 세션 정보를 반환한다.

    Returns:
        각 세션의 {name, pid, muted, volume} 딕셔너리 리스트.
        pycaw 미설치 또는 비-Windows 환경이면 빈 리스트.
    """
    if not is_available():
        logger.warning("미디어 제어 사용 불가 (pycaw 미설치 또는 비-Windows)")
        return []

    result: list[dict[str, Any]] = []
    try:
        sessions = _get_audio_sessions_raw()
        for session in sessions:
            process = session.Process
            if process is None:
                continue

            name = process.name()
            pid = process.pid
            volume = session.SimpleAudioVolume

            result.append({
                "name": name,
                "pid": pid,
                "muted": bool(volume.GetMute()),
                "volume": round(volume.GetMasterVolume(), 2),
            })
    except Exception:
        logger.exception("오디오 세션 조회 실패")

    return result


def mute_media_apps(mute: bool = True) -> dict[str, bool]:
    """대상 미디어 앱을 음소거하거나 해제한다.

    Args:
        mute: True이면 음소거, False이면 해제.

    Returns:
        {앱이름: 성공여부} 딕셔너리. 사용 불가 시 빈 딕셔너리.
    """
    global _saved_states

    if not is_available():
        logger.warning("미디어 제어 사용 불가 — 음소거 스킵")
        return {}

    results: dict[str, bool] = {}
    try:
        sessions = _get_audio_sessions_raw()
        for session in sessions:
            process = session.Process
            if process is None:
                continue

            name = process.name()
            pid = process.pid

            if not _is_target_app(name):
                continue

            volume = session.SimpleAudioVolume
            try:
                if mute:
                    # 음소거 전 현재 상태 저장
                    _saved_states[pid] = (
                        name,
                        bool(volume.GetMute()),
                        round(volume.GetMasterVolume(), 2),
                    )
                    volume.SetMute(1, None)
                else:
                    volume.SetMute(0, None)

                results[name] = True
            except Exception:
                logger.exception("앱 음소거 제어 실패: %s (pid=%d)", name, pid)
                results[name] = False
    except Exception:
        logger.exception("미디어 앱 음소거 처리 실패")

    return results


def restore_media_state() -> None:
    """음소거 전 저장된 상태로 미디어 앱을 복원한다.

    _saved_states에 보관된 (muted, volume) 값을 원래대로 되돌린다.
    복원 후 _saved_states를 비운다.
    """
    global _saved_states

    if not is_available():
        _saved_states.clear()
        return

    if not _saved_states:
        logger.debug("복원할 미디어 상태 없음")
        return

    try:
        sessions = _get_audio_sessions_raw()
        restored_pids: set[int] = set()

        for session in sessions:
            process = session.Process
            if process is None:
                continue

            pid = process.pid
            if pid not in _saved_states:
                continue

            name, was_muted, original_volume = _saved_states[pid]
            volume = session.SimpleAudioVolume

            try:
                volume.SetMute(int(was_muted), None)
                volume.SetMasterVolume(original_volume, None)
                restored_pids.add(pid)
                logger.debug("미디어 상태 복원: %s (pid=%d)", name, pid)
            except Exception:
                logger.exception("미디어 상태 복원 실패: %s (pid=%d)", name, pid)

        not_restored = set(_saved_states.keys()) - restored_pids
        if not_restored:
            logger.warning(
                "복원 불가 세션 (프로세스 종료됨): %s",
                [_saved_states[p][0] for p in not_restored if p in _saved_states],
            )
    except Exception:
        logger.exception("미디어 상태 복원 처리 실패")
    finally:
        _saved_states.clear()
