from __future__ import annotations

import ctypes
import json
from types import SimpleNamespace

import src.desktop as desktop


class _Kernel32Stub:
    def __init__(self, *, last_error: int, handle: int = 1) -> None:
        self._last_error = last_error
        self._handle = handle
        self.create_calls: list[tuple[object, object, object]] = []

    def CreateMutexW(self, security_attrs: object, initial_owner: object, name: object) -> int:
        self.create_calls.append((security_attrs, initial_owner, name))
        return self._handle

    def GetLastError(self) -> int:
        return self._last_error


def test_check_singleton_returns_existing_url_from_instance_file(monkeypatch, tmp_path):
    instance_file = tmp_path / ".sonote" / "instance.json"
    instance_file.parent.mkdir(parents=True, exist_ok=True)
    instance_file.write_text(json.dumps({"port": 17890}), encoding="utf-8")

    kernel32 = _Kernel32Stub(last_error=desktop._ERROR_ALREADY_EXISTS)
    monkeypatch.setattr(desktop.os, "name", "nt", raising=False)
    monkeypatch.setattr(desktop, "_instance_file_path", lambda: instance_file)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=kernel32), raising=False)

    result = desktop._check_singleton()

    assert result == "http://127.0.0.1:17890"
    assert kernel32.create_calls[0][2] == desktop._SINGLETON_MUTEX_NAME


def test_check_singleton_returns_already_running_without_port(monkeypatch, tmp_path):
    instance_file = tmp_path / ".sonote" / "instance.json"
    kernel32 = _Kernel32Stub(last_error=desktop._ERROR_ALREADY_EXISTS)
    monkeypatch.setattr(desktop.os, "name", "nt", raising=False)
    monkeypatch.setattr(desktop, "_instance_file_path", lambda: instance_file)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(kernel32=kernel32), raising=False)

    assert desktop._check_singleton() == "already_running"


def test_run_desktop_writes_and_clears_instance_file(monkeypatch, tmp_path):
    instance_file = tmp_path / ".sonote" / "instance.json"

    class _FakeController:
        def __init__(self, *, host: str, port: int, beta_mode: bool) -> None:
            self.port = 23456

        def run(self) -> None:
            payload = json.loads(instance_file.read_text(encoding="utf-8"))
            assert payload["port"] == self.port

    monkeypatch.setattr(desktop.os, "name", "nt", raising=False)
    monkeypatch.setattr(desktop, "_check_singleton", lambda: None)
    monkeypatch.setattr(desktop, "_instance_file_path", lambda: instance_file)
    monkeypatch.setattr(desktop, "_configure_beta_mode", lambda beta_mode: None)
    monkeypatch.setattr(desktop, "DesktopController", _FakeController)

    desktop.run_desktop(port=0, beta_mode=False)

    assert not instance_file.exists()


def test_run_desktop_focuses_existing_instance(monkeypatch):
    opened_urls: list[str] = []

    monkeypatch.setattr(desktop, "_check_singleton", lambda: "http://127.0.0.1:19000")
    monkeypatch.setattr(desktop, "_configure_beta_mode", lambda beta_mode: None)
    monkeypatch.setattr(desktop.webbrowser, "open", opened_urls.append)

    desktop.run_desktop()

    assert opened_urls == ["http://127.0.0.1:19000"]
