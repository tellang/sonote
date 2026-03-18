"""Windows 시스템 트레이 아이콘 — 회의 녹음 상태 표시 및 제어."""
from __future__ import annotations

import threading
import webbrowser
from typing import Callable

try:
    import pystray
    from PIL import Image, ImageDraw

    _TRAY_AVAILABLE = True
except ImportError:
    _TRAY_AVAILABLE = False

__all__ = ["MeetingTray", "is_available"]


def is_available() -> bool:
    """pystray + Pillow 설치 여부 확인."""
    return _TRAY_AVAILABLE


def _create_icon_image(recording: bool = True, paused: bool = False) -> "Image.Image":
    """트레이 아이콘 이미지 생성 (64x64).

    - 녹음 중: 빨간 원 + 흰색 마이크
    - 일시정지: 노란 원 + 흰색 일시정지 기호
    - 중지됨: 회색 원 + 흰색 사각형
    """
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 배경 원
    if paused:
        bg_color = (255, 180, 0, 255)  # 노란색
    elif recording:
        bg_color = (220, 50, 50, 255)  # 빨간색
    else:
        bg_color = (128, 128, 128, 255)  # 회색

    draw.ellipse([4, 4, size - 4, size - 4], fill=bg_color)

    # 아이콘 심볼
    white = (255, 255, 255, 255)
    if paused:
        # 일시정지 기호 (||)
        draw.rectangle([20, 18, 28, 46], fill=white)
        draw.rectangle([36, 18, 44, 46], fill=white)
    elif recording:
        # 마이크 모양 (간소화)
        # 마이크 몸통
        draw.rounded_rectangle([24, 12, 40, 36], radius=8, fill=white)
        # 마이크 받침대
        draw.arc([18, 20, 46, 44], start=0, end=180, fill=white, width=3)
        # 마이크 막대
        draw.line([32, 44, 32, 52], fill=white, width=3)
        draw.line([24, 52, 40, 52], fill=white, width=3)
    else:
        # 정지 사각형
        draw.rectangle([20, 20, 44, 44], fill=white)

    return img


class MeetingTray:
    """회의 녹음 시스템 트레이 아이콘.

    제공 기능:
    - 녹음/일시정지 상태 아이콘
    - 우클릭 메뉴: 일시정지/재개, 뷰어 열기, 종료
    - 툴팁: 경과시간, 세그먼트 수
    """

    def __init__(
        self,
        port: int = 8000,
        on_toggle_pause: Callable[[], None] | None = None,
        on_shutdown: Callable[[], None] | None = None,
    ) -> None:
        if not _TRAY_AVAILABLE:
            raise ImportError("pystray 또는 Pillow가 설치되어 있지 않습니다.")

        self._port = port
        self._on_toggle_pause = on_toggle_pause
        self._on_shutdown = on_shutdown
        self._recording = True
        self._paused = False
        self._elapsed = ""
        self._segments = 0
        self._speakers = 0
        self._icon: pystray.Icon | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """트레이 아이콘 시작 (데몬 스레드)."""
        self._icon = pystray.Icon(
            name="media-transcriber",
            icon=_create_icon_image(recording=True, paused=False),
            title=self._build_tooltip(),
            menu=self._build_menu(),
        )
        self._thread = threading.Thread(target=self._icon.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """트레이 아이콘 종료."""
        if self._icon:
            try:
                self._icon.stop()
            except Exception:
                pass

    def update_status(
        self,
        elapsed: str = "",
        segments: int = 0,
        speakers: int = 0,
        paused: bool = False,
    ) -> None:
        """상태 업데이트 — 아이콘 + 툴팁 갱신."""
        self._elapsed = elapsed
        self._segments = segments
        self._speakers = speakers
        self._paused = paused

        if self._icon:
            self._icon.icon = _create_icon_image(
                recording=self._recording, paused=self._paused,
            )
            self._icon.title = self._build_tooltip()
            # 메뉴 아이템 텍스트 갱신
            self._icon.menu = self._build_menu()

    def _build_tooltip(self) -> str:
        """툴팁 문자열 생성."""
        status = "⏸ 일시정지" if self._paused else "🎙 녹음 중"
        parts = [f"회의 전사 — {status}"]
        if self._elapsed:
            parts.append(f"경과: {self._elapsed}")
        if self._segments:
            parts.append(f"세그먼트: {self._segments}")
        if self._speakers:
            parts.append(f"화자: {self._speakers}명")
        return " | ".join(parts)

    def _build_menu(self) -> "pystray.Menu":
        """우클릭 메뉴 생성."""
        pause_text = "▶ 재개" if self._paused else "⏸ 일시정지"
        return pystray.Menu(
            pystray.MenuItem(
                f"🎙 회의 전사 ({self._elapsed or '00:00:00'})",
                None,
                enabled=False,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(pause_text, self._handle_toggle_pause),
            pystray.MenuItem("🌐 뷰어 열기", self._handle_open_viewer),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("⏹ 저장 && 종료", self._handle_shutdown),
        )

    def _handle_toggle_pause(self, icon: "pystray.Icon", item: "pystray.MenuItem") -> None:
        """일시정지/재개 토글."""
        if self._on_toggle_pause:
            self._on_toggle_pause()

    def _handle_open_viewer(self, icon: "pystray.Icon", item: "pystray.MenuItem") -> None:
        """브라우저에서 뷰어 열기."""
        webbrowser.open(f"http://localhost:{self._port}")

    def _handle_shutdown(self, icon: "pystray.Icon", item: "pystray.MenuItem") -> None:
        """종료 요청."""
        self._recording = False
        if self._icon:
            self._icon.icon = _create_icon_image(recording=False, paused=False)
            self._icon.title = "회의 전사 — 종료 중..."
        if self._on_shutdown:
            self._on_shutdown()
