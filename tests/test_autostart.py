from __future__ import annotations

import types

import pytest

from src import autostart


class _FakeKey:
    def __init__(self, store: dict[str, str]) -> None:
        self._store = store

    def __enter__(self) -> "_FakeKey":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeWinreg:
    HKEY_CURRENT_USER = object()
    KEY_SET_VALUE = 1
    KEY_READ = 2
    REG_SZ = 1

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def CreateKeyEx(self, hive, path: str, reserved: int, access: int) -> _FakeKey:
        return _FakeKey(self.values)

    def OpenKey(self, hive, path: str, reserved: int, access: int) -> _FakeKey:
        return _FakeKey(self.values)

    def SetValueEx(self, key: _FakeKey, name: str, reserved: int, reg_type: int, value: str) -> None:
        key._store[name] = value

    def DeleteValue(self, key: _FakeKey, name: str) -> None:
        if name not in key._store:
            raise FileNotFoundError(name)
        del key._store[name]

    def QueryValueEx(self, key: _FakeKey, name: str) -> tuple[str, int]:
        if name not in key._store:
            raise FileNotFoundError(name)
        return key._store[name], self.REG_SZ


@pytest.fixture
def fake_windows(monkeypatch: pytest.MonkeyPatch) -> _FakeWinreg:
    fake_winreg = _FakeWinreg()
    monkeypatch.setattr(autostart, "_WINREG", fake_winreg)
    monkeypatch.setattr(autostart.sys, "platform", "win32")
    return fake_winreg


def test_get_exe_path_uses_frozen_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(autostart.sys, "frozen", True, raising=False)
    monkeypatch.setattr(autostart.sys, "executable", "C:/Tools/sonote.exe")

    assert autostart.get_exe_path().endswith("sonote.exe")


def test_get_exe_path_uses_main_script(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(autostart.sys, "frozen", False, raising=False)
    monkeypatch.setattr(autostart.sys, "argv", ["sonote"])
    monkeypatch.setitem(
        autostart.sys.modules,
        "__main__",
        types.SimpleNamespace(__file__="C:/Projects/sonote/src/cli.py"),
    )

    assert autostart.get_exe_path().replace("\\", "/").endswith("src/cli.py")


def test_register_and_status_for_python_script(
    fake_windows: _FakeWinreg,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(autostart.sys, "frozen", False, raising=False)
    monkeypatch.setattr(autostart.sys, "executable", "C:/Python311/python.exe")
    monkeypatch.setattr(autostart.sys, "argv", ["C:/Projects/sonote/src/cli.py"])

    autostart.register()

    assert autostart.APP_NAME in fake_windows.values
    assert fake_windows.values[autostart.APP_NAME] == (
        "\"C:\\Python311\\python.exe\" \"C:\\Projects\\sonote\\src\\cli.py\" \"meeting\""
    )
    assert autostart.is_registered() is True


def test_unregister_removes_registry_value(
    fake_windows: _FakeWinreg,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_windows.values[autostart.APP_NAME] = "\"C:\\sonote.exe\" \"meeting\""
    monkeypatch.setattr(autostart.sys, "frozen", True, raising=False)
    monkeypatch.setattr(autostart.sys, "executable", "C:/sonote.exe")

    autostart.unregister()

    assert autostart.APP_NAME not in fake_windows.values


def test_is_registered_returns_false_for_stale_command(
    fake_windows: _FakeWinreg,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_windows.values[autostart.APP_NAME] = "\"C:\\old-sonote.exe\" \"meeting\""
    monkeypatch.setattr(autostart.sys, "frozen", True, raising=False)
    monkeypatch.setattr(autostart.sys, "executable", "C:/new-sonote.exe")

    assert autostart.is_registered() is False


def test_register_raises_on_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(autostart, "_WINREG", None)
    monkeypatch.setattr(autostart.sys, "platform", "linux")

    with pytest.raises(autostart.AutostartError):
        autostart.register()
