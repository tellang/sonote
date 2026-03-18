from __future__ import annotations

import threading
from unittest.mock import MagicMock

import src._server_impl as server_impl
import src.desktop_app.controller as controller_module
from src.desktop_app.controller import DesktopController


class _DummyThread:
    def __init__(self, *, name: str, alive: bool, daemon: bool = True) -> None:
        self.name = name
        self._alive = alive
        self.daemon = daemon
        self.join_calls: list[float | None] = []

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)


def test_desktop_shutdown_graceful_path(monkeypatch) -> None:
    ctrl = DesktopController(port=18765)
    ctrl._meeting_thread = _DummyThread(name="sonote-auto-start", alive=False)
    ctrl._status_thread = _DummyThread(name="sonote-status", alive=False)
    ctrl.server_thread = _DummyThread(name="sonote-fastapi", alive=False)

    writer = MagicMock()
    writer._segments = []
    writer._speakers = set()
    ctrl._writer = writer

    tray = MagicMock()
    tray._thread = _DummyThread(name="pystray", alive=False)
    ctrl.tray = tray

    mock_request_shutdown = MagicMock()
    mock_signal_shutdown = MagicMock(return_value=True)
    monkeypatch.setattr(controller_module, "request_shutdown", mock_request_shutdown)
    monkeypatch.setattr(controller_module, "signal_server_shutdown", mock_signal_shutdown)
    monkeypatch.setattr(controller_module.threading, "enumerate", lambda: [threading.current_thread()])

    ctrl.shutdown()

    assert ctrl.stop_event.is_set()
    assert ctrl._shutdown_complete.is_set()
    mock_request_shutdown.assert_called_once()
    mock_signal_shutdown.assert_called_once()
    tray.stop.assert_called_once()
    writer.write_footer.assert_called_once()
    writer.close.assert_called_once()


def test_desktop_shutdown_uses_last_resort_after_timeout(monkeypatch) -> None:
    ctrl = DesktopController(port=18766)
    ctrl._meeting_thread = _DummyThread(name="sonote-auto-start", alive=True)
    ctrl.server_thread = _DummyThread(name="sonote-fastapi", alive=True)

    mock_request_shutdown = MagicMock()
    mock_signal_shutdown = MagicMock(return_value=True)
    mock_last_resort = MagicMock()

    monkeypatch.setattr(controller_module, "request_shutdown", mock_request_shutdown)
    monkeypatch.setattr(controller_module, "signal_server_shutdown", mock_signal_shutdown)
    monkeypatch.setattr(
        DesktopController,
        "_join_thread_until",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        DesktopController,
        "_join_daemon_threads",
        lambda self, *args, **kwargs: None,
    )
    monkeypatch.setattr(
        DesktopController,
        "_has_alive_background_threads",
        lambda self: True,
    )
    monkeypatch.setattr(DesktopController, "_last_resort_exit", staticmethod(mock_last_resort))

    tick_values = iter([100.0, 103.5, 103.5, 103.5, 103.5, 103.5])

    def _fake_monotonic() -> float:
        try:
            return next(tick_values)
        except StopIteration:
            return 103.5

    monkeypatch.setattr(controller_module.time, "monotonic", _fake_monotonic)

    ctrl.shutdown()

    mock_request_shutdown.assert_called_once()
    mock_signal_shutdown.assert_called_once()
    mock_last_resort.assert_called_once()


def test_signal_server_shutdown_sets_should_exit(monkeypatch) -> None:
    class _StubServer:
        def __init__(self) -> None:
            self.should_exit = False

    server = _StubServer()
    monkeypatch.setattr(server_impl, "_uvicorn_server", server)
    assert server_impl.signal_server_shutdown() is True
    assert server.should_exit is True

    monkeypatch.setattr(server_impl, "_uvicorn_server", None)
    assert server_impl.signal_server_shutdown() is False
