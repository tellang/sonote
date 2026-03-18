"""설정 영속화 모듈 — ~/.sonote/config.json 기반.

싱글톤 SonoteConfig 인스턴스를 통해 설정을 로드/저장/조회/변경한다.
디바운싱(300ms)으로 빈번한 쓰기를 최적화하고, 스레드 안전을 보장한다.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

__all__ = [
    "SonoteConfig",
    "get_config",
]

# 설정 파일 경로
CONFIG_DIR: Path = Path.home() / ".sonote"
CONFIG_PATH: Path = CONFIG_DIR / "config.json"

# 디바운스 대기 시간 (초)
_DEBOUNCE_SEC: float = 0.3

# 기본 설정값
_DEFAULTS: dict[str, Any] = {
    "microphone": "auto",
    "language": "ko",
    "diarize": True,
    "auto_mute_media": False,
    "discord_webhook_url": "",
    "output_format": "md",
    "polish_engine": "auto",
    "minimize_to_tray": True,
    "auto_start": False,
    "theme": "dark",
}


class SonoteConfig:
    """스레드 안전 설정 관리 클래스.

    첫 실행 시 기본값으로 config.json을 자동 생성한다.
    set() 호출 시 300ms 디바운싱으로 디스크에 저장한다.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path: Path = path or CONFIG_PATH
        self._lock: threading.Lock = threading.Lock()
        self._data: dict[str, Any] = {}
        self._save_timer: threading.Timer | None = None
        self.load()

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """디스크에서 설정을 읽어온다. 파일이 없으면 기본값으로 생성."""
        with self._lock:
            if self._path.exists():
                try:
                    raw = self._path.read_text(encoding="utf-8")
                    stored = json.loads(raw)
                except (json.JSONDecodeError, OSError):
                    stored = {}
                # 기본값 병합 — 새 키가 추가되어도 누락 없이 반영
                self._data = {**_DEFAULTS, **stored}
            else:
                self._data = dict(_DEFAULTS)
                self._write_to_disk()

    def save(self) -> None:
        """즉시 디스크에 저장한다 (디바운스 무시)."""
        with self._lock:
            self._cancel_pending_save()
            self._write_to_disk()

    def get(self, key: str, default: Any = None) -> Any:
        """설정값 조회.

        Args:
            key: 설정 키.
            default: 키가 없을 때 반환할 기본값.

        Returns:
            설정값 또는 default.
        """
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """설정값 변경 후 디바운싱 저장 예약.

        Args:
            key: 설정 키.
            value: 새 값.
        """
        with self._lock:
            self._data[key] = value
            self._schedule_save()

    def to_dict(self) -> dict[str, Any]:
        """현재 설정 전체를 딕셔너리로 반환."""
        with self._lock:
            return dict(self._data)

    def reset(self) -> None:
        """모든 설정을 기본값으로 초기화하고 저장."""
        with self._lock:
            self._data = dict(_DEFAULTS)
            self._cancel_pending_save()
            self._write_to_disk()

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _write_to_disk(self) -> None:
        """잠금 상태에서 호출. 디렉토리 보장 후 JSON 기록."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _schedule_save(self) -> None:
        """디바운싱 타이머 예약. 잠금 상태에서 호출."""
        self._cancel_pending_save()
        self._save_timer = threading.Timer(_DEBOUNCE_SEC, self._debounced_save)
        self._save_timer.daemon = True
        self._save_timer.start()

    def _cancel_pending_save(self) -> None:
        """대기 중인 타이머 취소. 잠금 상태에서 호출."""
        if self._save_timer is not None:
            self._save_timer.cancel()
            self._save_timer = None

    def _debounced_save(self) -> None:
        """타이머 콜백 — 잠금 획득 후 디스크 기록."""
        with self._lock:
            self._save_timer = None
            self._write_to_disk()


# ------------------------------------------------------------------
# 싱글톤
# ------------------------------------------------------------------

_instance: SonoteConfig | None = None
_instance_lock: threading.Lock = threading.Lock()


def get_config() -> SonoteConfig:
    """싱글톤 SonoteConfig 인스턴스를 반환한다."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = SonoteConfig()
    return _instance
