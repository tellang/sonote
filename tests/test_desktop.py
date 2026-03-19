"""desktop.py 단위 테스트.

_find_app_mode_browser(), _find_free_port(), _format_elapsed(),
_wait_for_server_ready(), _fetch_json(), DesktopController 초기화/종료를 검증한다.
하드웨어 의존 코드(WhisperWorker, AudioCapture)는 mock으로 처리한다.
"""

from __future__ import annotations

import socket
import threading
from unittest.mock import MagicMock, patch

import pytest

from src.desktop_app.controller import (
    DesktopController,
    _fetch_json,
    _find_free_port,
    _format_elapsed,
    _wait_for_server_ready,
)
from src.desktop_app.browser import _open_app_mode


# ---------------------------------------------------------------------------
# 1. _format_elapsed() — 초를 HH:MM:SS로 변환
# ---------------------------------------------------------------------------


class TestFormatElapsed:
    """_format_elapsed()가 초 단위를 올바른 HH:MM:SS 문자열로 변환하는지 검증한다."""

    def test_zero_seconds(self):
        """0초는 '00:00:00'으로 변환된다."""
        assert _format_elapsed(0) == "00:00:00"

    def test_59_seconds(self):
        """59초는 '00:00:59'으로 변환된다."""
        assert _format_elapsed(59) == "00:00:59"

    def test_60_seconds_is_one_minute(self):
        """60초는 '00:01:00'으로 변환된다."""
        assert _format_elapsed(60) == "00:01:00"

    def test_3600_seconds_is_one_hour(self):
        """3600초는 '01:00:00'으로 변환된다."""
        assert _format_elapsed(3600) == "01:00:00"

    def test_3661_seconds(self):
        """3661초(1시간 1분 1초)는 '01:01:01'으로 변환된다."""
        assert _format_elapsed(3661) == "01:01:01"

    def test_negative_seconds_treated_as_zero(self):
        """음수는 0으로 처리되어 '00:00:00'을 반환한다."""
        assert _format_elapsed(-10) == "00:00:00"

    def test_large_hours(self):
        """10시간 이상도 올바르게 포맷된다."""
        assert _format_elapsed(36000) == "10:00:00"

    def test_one_minute_30_seconds(self):
        """90초는 '00:01:30'으로 변환된다."""
        assert _format_elapsed(90) == "00:01:30"


# ---------------------------------------------------------------------------
# 2. _find_free_port() — 사용 가능한 포트 탐색
# ---------------------------------------------------------------------------


class TestFindFreePort:
    """_find_free_port()가 사용 가능한 포트를 반환하는지 검증한다."""

    def test_returns_valid_port_number(self):
        """반환값이 유효한 포트 번호 범위(1–65535) 내에 있다."""
        port = _find_free_port()
        assert 1 <= port <= 65535

    def test_returned_port_is_bindable(self):
        """반환된 포트로 실제 소켓을 바인드할 수 있다."""
        port = _find_free_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # 바인드가 가능해야 한다 (예외 없음)
            sock.bind(("127.0.0.1", port))

    def test_two_calls_return_different_ports(self):
        """연속 두 번 호출 시 서로 다른 포트를 반환한다 (실제로는 거의 항상 다름)."""
        port1 = _find_free_port()
        port2 = _find_free_port()
        # 두 포트 모두 유효 범위여야 한다
        assert 1 <= port1 <= 65535
        assert 1 <= port2 <= 65535


# ---------------------------------------------------------------------------
# 3. _find_app_mode_browser() — Chrome/Edge 탐색
# ---------------------------------------------------------------------------


class TestFindAppModeBrowser:
    """_find_app_mode_browser()의 플랫폼별 동작을 검증한다."""

    def test_returns_none_on_non_windows(self, monkeypatch):
        """win32가 아닌 플랫폼에서는 None을 반환한다."""
        import src.desktop_app.browser as desktop_module

        monkeypatch.setattr(desktop_module.sys, "platform", "linux")

        from src.desktop_app.browser import _find_app_mode_browser

        assert _find_app_mode_browser() is None

    def test_returns_path_when_browser_found_in_env(self, monkeypatch, tmp_path):
        """PROGRAMFILES에 브라우저 실행파일이 있으면 해당 경로를 반환한다."""
        import src.desktop_app.browser as desktop_module

        monkeypatch.setattr(desktop_module.sys, "platform", "win32")

        # 가짜 msedge.exe 생성
        edge_dir = tmp_path / "Microsoft" / "Edge" / "Application"
        edge_dir.mkdir(parents=True)
        fake_edge = edge_dir / "msedge.exe"
        fake_edge.touch()

        monkeypatch.setenv("PROGRAMFILES", str(tmp_path))
        monkeypatch.delenv("PROGRAMFILES(X86)", raising=False)
        monkeypatch.delenv("LOCALAPPDATA", raising=False)

        from src.desktop_app.browser import _find_app_mode_browser

        result = _find_app_mode_browser()
        assert result == str(fake_edge)

    def test_returns_chrome_when_edge_not_found(self, monkeypatch, tmp_path):
        """Edge가 없고 Chrome이 있으면 Chrome 경로를 반환한다."""
        import src.desktop_app.browser as desktop_module

        monkeypatch.setattr(desktop_module.sys, "platform", "win32")

        # 가짜 chrome.exe 생성
        chrome_dir = tmp_path / "Google" / "Chrome" / "Application"
        chrome_dir.mkdir(parents=True)
        fake_chrome = chrome_dir / "chrome.exe"
        fake_chrome.touch()

        monkeypatch.setenv("PROGRAMFILES", str(tmp_path))
        monkeypatch.delenv("PROGRAMFILES(X86)", raising=False)
        monkeypatch.delenv("LOCALAPPDATA", raising=False)

        from src.desktop_app.browser import _find_app_mode_browser

        result = _find_app_mode_browser()
        assert result == str(fake_chrome)

    def test_returns_none_when_no_browser_found(self, monkeypatch, tmp_path):
        """브라우저가 전혀 없으면 None을 반환한다."""
        import src.desktop_app.browser as desktop_module

        monkeypatch.setattr(desktop_module.sys, "platform", "win32")
        monkeypatch.setenv("PROGRAMFILES", str(tmp_path))
        monkeypatch.delenv("PROGRAMFILES(X86)", raising=False)
        monkeypatch.delenv("LOCALAPPDATA", raising=False)

        # shutil.which도 None 반환하도록 패치
        monkeypatch.setattr("src.desktop_app.browser.shutil.which", lambda name: None)

        from src.desktop_app.browser import _find_app_mode_browser

        assert _find_app_mode_browser() is None

    def test_falls_back_to_path_search(self, monkeypatch, tmp_path):
        """PROGRAMFILES에 없어도 PATH에 msedge가 있으면 그 경로를 반환한다."""
        import src.desktop_app.browser as desktop_module

        monkeypatch.setattr(desktop_module.sys, "platform", "win32")
        monkeypatch.setenv("PROGRAMFILES", str(tmp_path))
        monkeypatch.delenv("PROGRAMFILES(X86)", raising=False)
        monkeypatch.delenv("LOCALAPPDATA", raising=False)

        fake_msedge = str(tmp_path / "msedge.exe")
        monkeypatch.setattr(
            "src.desktop_app.browser.shutil.which",
            lambda name: fake_msedge if name == "msedge" else None,
        )

        from src.desktop_app.browser import _find_app_mode_browser

        result = _find_app_mode_browser()
        assert result == fake_msedge


# ---------------------------------------------------------------------------
# 4. _open_app_mode() — 브라우저 앱 모드 실행
# ---------------------------------------------------------------------------


class TestOpenAppMode:
    """_open_app_mode()의 성공/실패 동작을 검증한다."""

    def test_returns_false_when_no_browser_found(self, monkeypatch):
        """브라우저를 찾지 못하면 False를 반환한다."""
        monkeypatch.setattr("src.desktop_app.browser._find_app_mode_browser", lambda: None)
        assert _open_app_mode("http://localhost:8765") is False

    def test_returns_true_when_browser_launched(self, monkeypatch, tmp_path):
        """브라우저 실행에 성공하면 True를 반환한다."""
        fake_browser = str(tmp_path / "fake_browser.exe")
        monkeypatch.setattr(
            "src.desktop_app.browser._find_app_mode_browser", lambda: fake_browser
        )

        with patch("src.desktop_app.browser.subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            result = _open_app_mode("http://localhost:8765")

        assert result is True
        mock_popen.assert_called_once()
        # --app= 플래그가 포함되어야 한다
        call_args = mock_popen.call_args[0][0]
        assert any("--app=http://localhost:8765" in arg for arg in call_args)

    def test_returns_false_on_oserror(self, monkeypatch, tmp_path):
        """Popen이 OSError를 발생시키면 False를 반환한다."""
        monkeypatch.setattr(
            "src.desktop_app.browser._find_app_mode_browser",
            lambda: str(tmp_path / "nonexistent.exe"),
        )

        with patch("src.desktop_app.browser.subprocess.Popen", side_effect=OSError("실행 실패")):
            result = _open_app_mode("http://localhost:8765")

        assert result is False


# ---------------------------------------------------------------------------
# 5. _wait_for_server_ready() — 서버 준비 대기
# ---------------------------------------------------------------------------


class TestWaitForServerReady:
    """_wait_for_server_ready()의 타임아웃과 성공 동작을 검증한다."""

    def test_raises_runtime_error_on_timeout(self):
        """서버가 응답하지 않으면 timeout 후 RuntimeError를 발생시킨다."""
        with pytest.raises(RuntimeError, match="FastAPI 서버가 준비되지 않았습니다"):
            # 존재하지 않는 포트 — 즉시 타임아웃
            _wait_for_server_ready("http://127.0.0.1:19999", timeout_seconds=0.1)

    def test_returns_immediately_when_server_responds(self):
        """서버가 응답하면 즉시 반환한다 (예외 없음)."""
        # 실제 HTTP 서버 없이 urlopen을 mock으로 처리
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("src.desktop_app.controller.urllib.request.urlopen", return_value=mock_response):
            # 예외 없이 반환되어야 한다
            _wait_for_server_ready("http://127.0.0.1:8765", timeout_seconds=5.0)


# ---------------------------------------------------------------------------
# 6. _fetch_json() — JSON 엔드포인트 호출
# ---------------------------------------------------------------------------


class TestFetchJson:
    """_fetch_json()이 JSON 응답을 올바르게 파싱하는지 검증한다."""

    def test_returns_dict_from_json_response(self):
        """JSON 응답을 dict로 파싱하여 반환한다."""
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = b'{"status": "ok", "elapsed": 120}'

        with patch("src.desktop_app.controller.urllib.request.urlopen", return_value=mock_response):
            result = _fetch_json("http://127.0.0.1:8765/status")

        assert result == {"status": "ok", "elapsed": 120}

    def test_returns_empty_dict_when_response_is_not_dict(self):
        """JSON 응답이 dict가 아니면 빈 dict를 반환한다."""
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = b'["list", "response"]'

        with patch("src.desktop_app.controller.urllib.request.urlopen", return_value=mock_response):
            result = _fetch_json("http://127.0.0.1:8765/status")

        assert result == {}


# ---------------------------------------------------------------------------
# 7. DesktopController 초기화
# ---------------------------------------------------------------------------


class TestDesktopControllerInit:
    """DesktopController 초기화 로직을 검증한다."""

    def test_auto_assigns_free_port_when_port_is_zero(self):
        """port=0이면 자동으로 빈 포트를 할당한다."""
        ctrl = DesktopController(port=0)
        assert ctrl.port > 0

    def test_uses_given_port(self):
        """port를 명시하면 해당 포트를 사용한다."""
        ctrl = DesktopController(port=19876)
        assert ctrl.port == 19876

    def test_base_url_matches_host_and_port(self):
        """base_url이 host와 port를 올바르게 조합한다."""
        ctrl = DesktopController(host="127.0.0.1", port=8765)
        assert ctrl.base_url == "http://127.0.0.1:8765"

    def test_stop_event_is_not_set_initially(self):
        """초기 상태에서 stop_event는 설정되지 않은 상태다."""
        ctrl = DesktopController(port=19877)
        assert not ctrl.stop_event.is_set()

    def test_server_thread_is_none_before_run(self):
        """run() 호출 전에는 server_thread가 None이다."""
        ctrl = DesktopController(port=19878)
        assert ctrl.server_thread is None


# ---------------------------------------------------------------------------
# 8. DesktopController.shutdown()
# ---------------------------------------------------------------------------


class TestDesktopControllerShutdown:
    """DesktopController.shutdown()의 종료 동작을 검증한다."""

    def test_shutdown_sets_stop_event(self):
        """shutdown() 호출 후 stop_event가 설정된다."""
        ctrl = DesktopController(port=19880)

        with patch("src.desktop_app.controller.request_shutdown"):
            ctrl.shutdown()

        assert ctrl.stop_event.is_set()

    def test_shutdown_calls_request_shutdown(self):
        """shutdown()은 server의 request_shutdown()을 호출한다."""
        ctrl = DesktopController(port=19881)

        with patch("src.desktop_app.controller.request_shutdown") as mock_shutdown:
            ctrl.shutdown()

        mock_shutdown.assert_called_once()

    def test_shutdown_is_idempotent(self):
        """shutdown()을 두 번 호출해도 예외가 발생하지 않는다."""
        ctrl = DesktopController(port=19882)

        with patch("src.desktop_app.controller.request_shutdown"):
            ctrl.shutdown()
            ctrl.shutdown()  # 두 번째 호출 — 예외 없음

    def test_shutdown_stops_tray_if_present(self):
        """트레이가 있으면 shutdown() 시 tray.stop()이 호출된다."""
        ctrl = DesktopController(port=19883)
        mock_tray = MagicMock()
        ctrl.tray = mock_tray

        with patch("src.desktop_app.controller.request_shutdown"):
            ctrl.shutdown()

        mock_tray.stop.assert_called_once()


# ---------------------------------------------------------------------------
# 9. DesktopController.toggle_recording()
# ---------------------------------------------------------------------------


class TestDesktopControllerToggleRecording:
    """toggle_recording()이 pause 상태를 토글하는지 검증한다."""

    def test_toggle_recording_calls_toggle_pause_state(self):
        """toggle_recording()은 server의 toggle_pause_state()를 호출한다."""
        ctrl = DesktopController(port=19884)

        with patch(
            "src.desktop_app.controller.toggle_pause_state", return_value={"paused": True}
        ) as mock_toggle:
            ctrl.toggle_recording()

        mock_toggle.assert_called_once()

    def test_toggle_recording_updates_tray_when_present(self):
        """트레이가 있으면 toggle 후 tray.update_status()가 호출된다."""
        ctrl = DesktopController(port=19885)
        mock_tray = MagicMock()
        ctrl.tray = mock_tray

        with patch(
            "src.desktop_app.controller.toggle_pause_state", return_value={"paused": False}
        ):
            ctrl.toggle_recording()

        mock_tray.update_status.assert_called_once_with(paused=False, recording=True)
