"""paths.py 단위 테스트.

project_root(), static_dir(), OUTPUT_ROOT 경로 계산 및
날짜별 디렉토리 생성 함수(meetings_dir, transcripts_dir, audio_dir)와
헬퍼 경로 함수(data_dir, profiles_db_path, speakers_json_path)를 검증한다.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# 1. project_root() — frozen/non-frozen 분기
# ---------------------------------------------------------------------------


class TestProjectRoot:
    """project_root()가 실행 환경에 따라 올바른 경로를 반환하는지 검증한다."""

    def test_non_frozen_returns_parent_of_src(self):
        """일반 Python 환경에서 project_root()는 src/ 의 부모 디렉토리를 반환한다."""
        from src.paths import project_root

        root = project_root()
        # src/paths.py 기준: .parent.parent == 프로젝트 루트
        assert root.is_dir()
        # 반환 경로 아래에 src 디렉토리가 있어야 한다
        assert (root / "src").is_dir()

    def test_non_frozen_path_is_absolute(self):
        """non-frozen 환경에서 반환 경로는 절대경로여야 한다."""
        from src.paths import project_root

        assert project_root().is_absolute()

    def test_frozen_returns_executable_parent(self, monkeypatch, tmp_path):
        """PyInstaller frozen 환경에서는 실행 파일의 부모 디렉토리를 반환한다."""
        # frozen 속성을 흉내냄
        fake_exe = tmp_path / "sonote.exe"
        fake_exe.touch()

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "executable", str(fake_exe))

        # 모듈을 재임포트하지 않고 함수만 재호출 — sys 상태를 monkeypatch가 처리
        from src import paths as paths_module

        result = paths_module.project_root()
        assert result == tmp_path.resolve()


# ---------------------------------------------------------------------------
# 2. static_dir() — frozen/non-frozen 분기
# ---------------------------------------------------------------------------


class TestStaticDir:
    """static_dir()가 실행 환경에 따라 올바른 static 경로를 반환하는지 검증한다."""

    def test_non_frozen_returns_static_under_project_root(self):
        """일반 환경에서 static_dir()는 프로젝트 루트 / static 을 반환한다."""
        from src.paths import project_root, static_dir

        result = static_dir()
        assert result == project_root() / "static"

    def test_non_frozen_path_is_absolute(self):
        """non-frozen 환경에서 반환 경로는 절대경로여야 한다."""
        from src.paths import static_dir

        assert static_dir().is_absolute()

    def test_frozen_returns_meipass_static(self, monkeypatch, tmp_path):
        """PyInstaller frozen 환경에서는 _MEIPASS / static 을 반환한다."""
        fake_meipass = tmp_path / "_MEIPASS"
        fake_meipass.mkdir()

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "_MEIPASS", str(fake_meipass), raising=False)

        from src import paths as paths_module

        result = paths_module.static_dir()
        assert result == fake_meipass / "static"


# ---------------------------------------------------------------------------
# 3. 날짜별 디렉토리 함수 — meetings_dir, transcripts_dir, audio_dir
# ---------------------------------------------------------------------------


class TestDateDirs:
    """날짜별 디렉토리 함수가 올바른 경로를 반환하고 디렉토리를 생성하는지 검증한다."""

    @pytest.fixture(autouse=True)
    def _patch_output_root(self, monkeypatch, tmp_path):
        """OUTPUT_ROOT를 임시 디렉토리로 교체하여 실제 파일시스템 오염을 방지한다."""
        import src.paths as paths_module

        monkeypatch.setattr(paths_module, "OUTPUT_ROOT", tmp_path)
        self.output_root = tmp_path

    def test_meetings_dir_uses_today_when_no_date(self):
        """date 인자 없이 호출하면 오늘 날짜 디렉토리를 반환한다."""
        from src.paths import meetings_dir

        today_str = datetime.now().strftime("%Y-%m-%d")
        result = meetings_dir()
        assert result == self.output_root / "meetings" / today_str

    def test_meetings_dir_uses_given_date(self):
        """date 인자를 주면 해당 날짜 디렉토리를 반환한다."""
        from src.paths import meetings_dir

        fixed = datetime(2024, 3, 15)
        result = meetings_dir(fixed)
        assert result == self.output_root / "meetings" / "2024-03-15"

    def test_meetings_dir_creates_directory(self):
        """meetings_dir() 호출 시 디렉토리가 실제로 생성된다."""
        from src.paths import meetings_dir

        fixed = datetime(2025, 1, 1)
        result = meetings_dir(fixed)
        assert result.is_dir()

    def test_transcripts_dir_uses_given_date(self):
        """transcripts_dir()가 주어진 날짜로 올바른 경로를 반환한다."""
        from src.paths import transcripts_dir

        fixed = datetime(2024, 6, 20)
        result = transcripts_dir(fixed)
        assert result == self.output_root / "transcripts" / "2024-06-20"
        assert result.is_dir()

    def test_audio_dir_uses_given_date(self):
        """audio_dir()가 주어진 날짜로 올바른 경로를 반환한다."""
        from src.paths import audio_dir

        fixed = datetime(2024, 12, 31)
        result = audio_dir(fixed)
        assert result == self.output_root / "audio" / "2024-12-31"
        assert result.is_dir()

    def test_multiple_calls_same_date_return_same_path(self):
        """같은 날짜로 여러 번 호출해도 동일한 경로를 반환한다."""
        from src.paths import meetings_dir

        fixed = datetime(2025, 5, 10)
        assert meetings_dir(fixed) == meetings_dir(fixed)

    def test_date_format_is_yyyy_mm_dd(self):
        """날짜 포맷이 YYYY-MM-DD 형식임을 검증한다."""
        from src.paths import audio_dir

        fixed = datetime(2025, 9, 5)
        result = audio_dir(fixed)
        # 경로 마지막 부분이 '2025-09-05' 형식인지 확인
        assert result.name == "2025-09-05"


# ---------------------------------------------------------------------------
# 4. data_dir(), profiles_db_path(), speakers_json_path()
# ---------------------------------------------------------------------------


class TestDataPaths:
    """data_dir()와 DB/JSON 경로 헬퍼를 검증한다."""

    @pytest.fixture(autouse=True)
    def _patch_output_root(self, monkeypatch, tmp_path):
        import src.paths as paths_module

        monkeypatch.setattr(paths_module, "OUTPUT_ROOT", tmp_path)
        self.output_root = tmp_path

    def test_data_dir_returns_output_root_data(self):
        """data_dir()는 OUTPUT_ROOT / data 를 반환한다."""
        from src.paths import data_dir

        assert data_dir() == self.output_root / "data"

    def test_data_dir_creates_directory(self):
        """data_dir() 호출 시 디렉토리가 생성된다."""
        from src.paths import data_dir

        result = data_dir()
        assert result.is_dir()

    def test_profiles_db_path_returns_correct_filename(self):
        """profiles_db_path()는 data_dir() / profiles.db 를 반환한다."""
        from src.paths import profiles_db_path

        result = profiles_db_path()
        assert result.name == "profiles.db"
        assert result.parent.name == "data"

    def test_speakers_json_path_returns_correct_filename(self):
        """speakers_json_path()는 data_dir() / speakers.json 을 반환한다."""
        from src.paths import speakers_json_path

        result = speakers_json_path()
        assert result.name == "speakers.json"
        assert result.parent.name == "data"

    def test_profiles_db_path_is_under_output_root(self):
        """profiles_db_path()가 OUTPUT_ROOT 하위에 있다."""
        from src.paths import profiles_db_path

        assert str(profiles_db_path()).startswith(str(self.output_root))

    def test_speakers_json_path_is_under_output_root(self):
        """speakers_json_path()가 OUTPUT_ROOT 하위에 있다."""
        from src.paths import speakers_json_path

        assert str(speakers_json_path()).startswith(str(self.output_root))
