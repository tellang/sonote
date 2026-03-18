"""config.py 단위 테스트.

SonoteConfig의 로드/저장/조회/변경/리셋, 디바운싱, 스레드 안전성,
get_config() 싱글톤 동작을 검증한다.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from src.config import SonoteConfig, _DEFAULTS


# ---------------------------------------------------------------------------
# 1. 초기화 및 기본값
# ---------------------------------------------------------------------------


class TestSonoteConfigInit:
    """SonoteConfig 초기화 동작을 검증한다."""

    def test_creates_config_file_on_first_run(self, tmp_path):
        """설정 파일이 없으면 기본값으로 새 파일을 생성한다."""
        config_path = tmp_path / "config.json"
        assert not config_path.exists()

        SonoteConfig(path=config_path)

        assert config_path.exists()

    def test_default_values_are_written_to_new_file(self, tmp_path):
        """새 파일에는 _DEFAULTS 값이 모두 기록된다."""
        config_path = tmp_path / "config.json"
        SonoteConfig(path=config_path)

        written = json.loads(config_path.read_text(encoding="utf-8"))
        for key, value in _DEFAULTS.items():
            assert written[key] == value

    def test_loads_existing_values(self, tmp_path):
        """기존 파일이 있으면 저장된 값을 읽어온다."""
        config_path = tmp_path / "config.json"
        custom = {**_DEFAULTS, "language": "en", "theme": "light"}
        config_path.write_text(json.dumps(custom), encoding="utf-8")

        cfg = SonoteConfig(path=config_path)
        assert cfg.get("language") == "en"
        assert cfg.get("theme") == "light"

    def test_merges_new_default_keys_with_existing_file(self, tmp_path):
        """기존 파일에 없는 새 기본값 키가 누락 없이 병합된다."""
        config_path = tmp_path / "config.json"
        # 일부 키만 있는 파일 생성
        partial = {"language": "ko"}
        config_path.write_text(json.dumps(partial), encoding="utf-8")

        cfg = SonoteConfig(path=config_path)
        # 기존 키는 유지
        assert cfg.get("language") == "ko"
        # 누락된 기본 키는 채워짐
        assert cfg.get("microphone") == _DEFAULTS["microphone"]
        assert cfg.get("diarize") == _DEFAULTS["diarize"]

    def test_invalid_json_falls_back_to_defaults(self, tmp_path):
        """JSON 파싱 오류가 있으면 기본값으로 폴백한다."""
        config_path = tmp_path / "config.json"
        config_path.write_text("{ invalid json }", encoding="utf-8")

        cfg = SonoteConfig(path=config_path)
        assert cfg.get("language") == _DEFAULTS["language"]

    def test_creates_parent_directory_if_missing(self, tmp_path):
        """설정 파일의 부모 디렉토리가 없으면 자동 생성한다."""
        nested_path = tmp_path / "deep" / "nested" / "config.json"
        assert not nested_path.parent.exists()

        SonoteConfig(path=nested_path)
        assert nested_path.exists()


# ---------------------------------------------------------------------------
# 2. get() / set() / to_dict()
# ---------------------------------------------------------------------------


class TestSonoteConfigGetSet:
    """get(), set(), to_dict() 동작을 검증한다."""

    @pytest.fixture
    def cfg(self, tmp_path):
        return SonoteConfig(path=tmp_path / "config.json")

    def test_get_returns_default_value_for_existing_key(self, cfg):
        """기본 키의 값을 올바르게 반환한다."""
        assert cfg.get("language") == _DEFAULTS["language"]

    def test_get_returns_none_for_unknown_key(self, cfg):
        """존재하지 않는 키는 None을 반환한다."""
        assert cfg.get("nonexistent_key") is None

    def test_get_returns_custom_default_for_unknown_key(self, cfg):
        """존재하지 않는 키에 default 인자를 주면 그 값을 반환한다."""
        assert cfg.get("nonexistent_key", "fallback") == "fallback"

    def test_set_updates_value_in_memory(self, cfg):
        """set() 후 get()으로 즉시 변경된 값을 읽을 수 있다."""
        cfg.set("language", "en")
        assert cfg.get("language") == "en"

    def test_set_new_key_is_retrievable(self, cfg):
        """기본값에 없는 새 키도 set() 후 get()으로 읽을 수 있다."""
        cfg.set("custom_key", 42)
        assert cfg.get("custom_key") == 42

    def test_to_dict_returns_all_keys(self, cfg):
        """to_dict()는 모든 설정 키를 포함한 딕셔너리를 반환한다."""
        result = cfg.to_dict()
        for key in _DEFAULTS:
            assert key in result

    def test_to_dict_returns_copy_not_reference(self, cfg):
        """to_dict()가 반환한 딕셔너리를 수정해도 내부 상태에 영향이 없다."""
        d = cfg.to_dict()
        d["language"] = "MUTATED"
        assert cfg.get("language") != "MUTATED"


# ---------------------------------------------------------------------------
# 3. save() — 즉시 디스크 기록
# ---------------------------------------------------------------------------


class TestSonoteConfigSave:
    """save()의 즉시 저장 동작을 검증한다."""

    def test_save_writes_to_disk_immediately(self, tmp_path):
        """save() 호출 즉시 변경된 값이 파일에 기록된다."""
        config_path = tmp_path / "config.json"
        cfg = SonoteConfig(path=config_path)

        cfg.set("language", "en")
        # 디바운스 타이머가 실행되기 전에 save() 강제 호출
        cfg.save()

        written = json.loads(config_path.read_text(encoding="utf-8"))
        assert written["language"] == "en"

    def test_save_produces_valid_json(self, tmp_path):
        """save() 후 파일이 유효한 JSON이다."""
        config_path = tmp_path / "config.json"
        cfg = SonoteConfig(path=config_path)
        cfg.set("theme", "light")
        cfg.save()

        content = config_path.read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_save_file_ends_with_newline(self, tmp_path):
        """저장된 파일은 개행 문자로 끝난다."""
        config_path = tmp_path / "config.json"
        cfg = SonoteConfig(path=config_path)
        cfg.save()

        content = config_path.read_text(encoding="utf-8")
        assert content.endswith("\n")


# ---------------------------------------------------------------------------
# 4. load() — 디스크 재로드
# ---------------------------------------------------------------------------


class TestSonoteConfigLoad:
    """load()로 디스크 상태를 재반영하는지 검증한다."""

    def test_load_reflects_external_file_changes(self, tmp_path):
        """외부에서 파일을 수정한 후 load()하면 변경이 반영된다."""
        config_path = tmp_path / "config.json"
        cfg = SonoteConfig(path=config_path)

        # 외부에서 파일 직접 수정
        updated = cfg.to_dict()
        updated["language"] = "ja"
        config_path.write_text(json.dumps(updated), encoding="utf-8")

        cfg.load()
        assert cfg.get("language") == "ja"

    def test_load_after_file_deleted_uses_defaults(self, tmp_path):
        """파일 삭제 후 load()하면 기본값으로 초기화되고 새 파일이 생성된다."""
        config_path = tmp_path / "config.json"
        cfg = SonoteConfig(path=config_path)
        cfg.set("language", "en")
        cfg.save()

        # 파일 삭제
        config_path.unlink()
        cfg.load()

        # 기본값으로 복원
        assert cfg.get("language") == _DEFAULTS["language"]
        # 파일이 재생성됨
        assert config_path.exists()


# ---------------------------------------------------------------------------
# 5. reset()
# ---------------------------------------------------------------------------


class TestSonoteConfigReset:
    """reset()이 모든 값을 기본값으로 되돌리는지 검증한다."""

    def test_reset_restores_defaults(self, tmp_path):
        """set()으로 변경한 값이 reset() 후 기본값으로 복원된다."""
        cfg = SonoteConfig(path=tmp_path / "config.json")
        cfg.set("language", "en")
        cfg.set("theme", "light")
        cfg.set("custom_key", "custom")

        cfg.reset()

        assert cfg.get("language") == _DEFAULTS["language"]
        assert cfg.get("theme") == _DEFAULTS["theme"]
        # 기본 키에 없는 커스텀 키는 사라진다
        assert cfg.get("custom_key") is None

    def test_reset_writes_defaults_to_disk(self, tmp_path):
        """reset() 후 파일에도 기본값이 기록된다."""
        config_path = tmp_path / "config.json"
        cfg = SonoteConfig(path=config_path)
        cfg.set("language", "en")
        cfg.reset()

        written = json.loads(config_path.read_text(encoding="utf-8"))
        assert written["language"] == _DEFAULTS["language"]


# ---------------------------------------------------------------------------
# 6. 디바운싱 — set() 후 파일 저장 지연
# ---------------------------------------------------------------------------


class TestSonoteConfigDebounce:
    """디바운싱이 실제로 파일 쓰기를 지연시키는지 검증한다."""

    def test_set_does_not_write_immediately(self, tmp_path):
        """set() 직후에는 아직 파일에 변경이 기록되지 않는다 (디바운스 지연)."""
        config_path = tmp_path / "config.json"
        cfg = SonoteConfig(path=config_path)
        # 초기 상태 파일 기록
        cfg.save()

        cfg.set("language", "en")

        # 타이머가 실행되기 전 즉시 읽으면 이전 값이어야 한다
        written = json.loads(config_path.read_text(encoding="utf-8"))
        assert written["language"] == _DEFAULTS["language"]

    def test_set_writes_after_debounce_delay(self, tmp_path):
        """디바운스 대기 후에는 파일에 변경이 기록된다."""
        from src.config import _DEBOUNCE_SEC

        config_path = tmp_path / "config.json"
        cfg = SonoteConfig(path=config_path)

        cfg.set("language", "en")
        # 디바운스 시간 + 여유
        time.sleep(_DEBOUNCE_SEC + 0.1)

        written = json.loads(config_path.read_text(encoding="utf-8"))
        assert written["language"] == "en"

    def test_rapid_sets_coalesce_into_single_write(self, tmp_path):
        """빠른 연속 set()은 마지막 값 하나만 파일에 기록된다."""
        from src.config import _DEBOUNCE_SEC

        config_path = tmp_path / "config.json"
        cfg = SonoteConfig(path=config_path)

        # 빠르게 여러 번 변경
        for i in range(10):
            cfg.set("language", f"lang_{i}")

        time.sleep(_DEBOUNCE_SEC + 0.1)

        written = json.loads(config_path.read_text(encoding="utf-8"))
        assert written["language"] == "lang_9"


# ---------------------------------------------------------------------------
# 7. 스레드 안전성
# ---------------------------------------------------------------------------


class TestSonoteConfigThreadSafety:
    """다중 스레드에서 동시 get/set 시 데이터 일관성을 검증한다."""

    def test_concurrent_sets_do_not_corrupt_data(self, tmp_path):
        """50개 스레드가 동시에 서로 다른 키를 set해도 데이터가 손상되지 않는다."""
        cfg = SonoteConfig(path=tmp_path / "config.json")
        errors: list[Exception] = []

        def worker(index: int) -> None:
            try:
                cfg.set(f"key_{index}", index)
                val = cfg.get(f"key_{index}")
                # 읽은 값이 정수여야 한다 (손상 없음)
                assert isinstance(val, int)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"스레드 오류 발생: {errors}"

    def test_concurrent_get_does_not_raise(self, tmp_path):
        """다중 스레드에서 동시 get()이 예외 없이 동작한다."""
        cfg = SonoteConfig(path=tmp_path / "config.json")
        errors: list[Exception] = []

        def reader() -> None:
            try:
                for _ in range(100):
                    cfg.get("language")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ---------------------------------------------------------------------------
# 8. get_config() 싱글톤
# ---------------------------------------------------------------------------


class TestGetConfigSingleton:
    """get_config()가 동일한 인스턴스를 반환하는지 검증한다."""

    def test_get_config_returns_same_instance(self):
        """두 번 호출해도 동일한 객체를 반환한다."""
        from src.config import get_config

        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_get_config_returns_sonote_config_instance(self):
        """반환값이 SonoteConfig 인스턴스다."""
        from src.config import get_config

        cfg = get_config()
        assert isinstance(cfg, SonoteConfig)
