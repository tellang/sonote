"""Discord Webhook 알림 모듈 — 회의 요약 및 실시간 전사 알림 전송."""

from __future__ import annotations

import os
import threading
from typing import Any

import httpx

__all__ = [
    "is_configured",
    "send_webhook",
    "send_meeting_summary",
    "send_realtime_update",
]

# 실시간 업데이트 디바운싱 간격 (초)
_DEBOUNCE_SECONDS = 5.0

# 디바운싱 배치 버퍼 및 락
_batch_lock = threading.Lock()
_batch_buffer: list[str] = []
_batch_timer: threading.Timer | None = None
_batch_webhook_url: str = ""


def _get_webhook_url(webhook_url: str | None = None) -> str | None:
    """Webhook URL 결정: 인자 > 환경변수 순."""
    if webhook_url:
        return webhook_url
    return os.environ.get("DISCORD_WEBHOOK_URL")


def is_configured(webhook_url: str | None = None) -> bool:
    """Discord Webhook URL이 설정되어 있는지 확인한다."""
    return bool(_get_webhook_url(webhook_url))


def send_webhook(
    webhook_url: str,
    content: str,
    embeds: list[dict[str, Any]] | None = None,
) -> bool:
    """Discord Webhook으로 메시지를 전송한다.

    Args:
        webhook_url: Discord Webhook URL.
        content: 메시지 본문.
        embeds: Discord Embed 객체 리스트 (선택).

    Returns:
        전송 성공 여부.
    """
    payload: dict[str, Any] = {"content": content}
    if embeds:
        payload["embeds"] = embeds

    try:
        resp = httpx.post(
            webhook_url,
            json=payload,
            timeout=10.0,
        )
        if resp.status_code >= 400:
            print(f"[discord] Webhook 전송 실패: HTTP {resp.status_code}")
            return False
        return True
    except httpx.HTTPError as exc:
        print(f"[discord] Webhook 전송 오류: {exc}")
        return False


def send_meeting_summary(
    webhook_url: str,
    summary: str,
    speakers: int,
    elapsed: str,
    session_id: str,
) -> bool:
    """회의 요약을 Discord Embed 형식으로 전송한다.

    Args:
        webhook_url: Discord Webhook URL.
        summary: 회의 요약 텍스트.
        speakers: 참여 화자 수.
        elapsed: 경과 시간 문자열 (예: "01:23:45").
        session_id: 세션 식별자.

    Returns:
        전송 성공 여부.
    """
    embed: dict[str, Any] = {
        "title": "회의 요약",
        "description": summary,
        "color": 0x5865F2,  # Discord 브랜드 블루
        "fields": [
            {"name": "화자 수", "value": str(speakers), "inline": True},
            {"name": "경과 시간", "value": elapsed, "inline": True},
            {"name": "세션 ID", "value": f"`{session_id}`", "inline": False},
        ],
    }
    return send_webhook(webhook_url, content="", embeds=[embed])


def _flush_batch() -> None:
    """배치 버퍼에 쌓인 실시간 업데이트를 한 번에 전송한다."""
    global _batch_timer
    with _batch_lock:
        if not _batch_buffer:
            _batch_timer = None
            return
        lines = list(_batch_buffer)
        _batch_buffer.clear()
        url = _batch_webhook_url
        _batch_timer = None

    # 2000자 Discord 메시지 제한 대응
    message = "\n".join(lines)
    if len(message) > 1900:
        message = message[:1900] + "\n..."

    send_webhook(url, content=message)


def send_realtime_update(
    webhook_url: str,
    text: str,
    speaker: str,
) -> None:
    """실시간 전사 결과를 5초 디바운싱 배치로 전송한다.

    짧은 간격의 호출을 모아서 한 번에 전송하여 Discord Rate Limit을 방지한다.

    Args:
        webhook_url: Discord Webhook URL.
        text: 전사된 텍스트.
        speaker: 화자 이름.
    """
    global _batch_timer, _batch_webhook_url

    line = f"**[{speaker}]** {text}"

    with _batch_lock:
        _batch_webhook_url = webhook_url
        _batch_buffer.append(line)

        # 기존 타이머가 있으면 리셋하지 않고 그대로 둔다 (첫 호출 기준 디바운싱)
        if _batch_timer is None:
            _batch_timer = threading.Timer(_DEBOUNCE_SECONDS, _flush_batch)
            _batch_timer.daemon = True
            _batch_timer.start()
