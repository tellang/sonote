"""sonote setup 커맨드 테스트.

테스트 계획:
1. setup 모듈 임포트 가능 여부
2. run_setup() JSON 모드 — 핵심 항목 실행 + JSON 출력
3. run_setup() fix 모드 — doctor 연동
4. _load_env_file() — .env 로드
5. _setup_hf_token() — 토큰 설정/영속화
6. CLI 통합: `sonote setup --json` 실행
7. CLI 통합: `sonote setup --help` 도움말
"""

import json
import os
import subprocess
from unittest.mock import patch



class TestSetupImport:
    """모듈 임포트 테스트."""

    def test_import_setup(self):
        from src.setup import run_setup
        assert callable(run_setup)

    def test_import_all_exports(self):
        from src.setup import _SETUP_ITEMS
        assert isinstance(_SETUP_ITEMS, list)
        assert len(_SETUP_ITEMS) > 0


class TestSetupCore:
    """run_setup 핵심 기능 테스트."""

    def test_json_output_structure(self, capsys):
        """JSON 모드 출력 구조 검증."""
        from src.setup import run_setup
        result = run_setup(use_json=True)

        assert "items" in result
        assert "summary" in result
        assert "ok" in result["summary"]
        assert "failed" in result["summary"]
        assert "skipped" in result["summary"]

        # JSON 출력이 stdout에 찍혔는지
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "items" in parsed

    def test_core_items_always_run(self):
        """core 카테고리 항목은 항상 실행."""
        from src.setup import run_setup
        result = run_setup(use_json=True)

        core_items = [r for r in result["items"] if r["status"] != "skipped"]
        core_keys = {r["key"] for r in core_items}
        assert "ffmpeg" in core_keys
        assert "model" in core_keys

    def test_diarize_skipped_by_default(self):
        """diarize 카테고리는 기본적으로 스킵."""
        from src.setup import run_setup
        result = run_setup(use_json=True)

        diarize_items = [r for r in result["items"] if r["key"] == "diarize"]
        assert len(diarize_items) == 1
        assert diarize_items[0]["status"] == "skipped"

    def test_all_flag_runs_everything(self):
        """--all 플래그는 모든 항목 실행 (skipped 없음)."""
        from src.setup import run_setup
        result = run_setup(all_extras=True, use_json=True)

        skipped = [r for r in result["items"] if r["status"] == "skipped"]
        assert len(skipped) == 0

    def test_summary_counts_match(self):
        """summary 카운트가 items와 일치."""
        from src.setup import run_setup
        result = run_setup(use_json=True)

        items = result["items"]
        summary = result["summary"]
        assert summary["ok"] == sum(1 for r in items if r["status"] == "ok")
        assert summary["failed"] == sum(1 for r in items if r["status"] == "failed")
        assert summary["skipped"] == sum(1 for r in items if r["status"] == "skipped")


class TestHfToken:
    """HuggingFace 토큰 설정 테스트."""

    def test_token_already_set(self):
        from src.setup import _setup_hf_token
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test1234abcd"}):
            ok, detail = _setup_hf_token()
            assert ok is True
            assert "이미 설정됨" in detail

    def test_token_set_new(self, tmp_path):
        from src.setup import _setup_hf_token
        with patch.dict(os.environ, {}, clear=True):
            # HOME을 tmp_path로 변경하여 .env 파일 생성 테스트
            with patch("src.setup.Path.home", return_value=tmp_path):
                ok, detail = _setup_hf_token("hf_test_token_12345")
                assert ok is True
                assert "설정 완료" in detail

                # .env 파일 생성 확인
                env_file = tmp_path / ".sonote" / ".env"
                assert env_file.exists()
                content = env_file.read_text()
                assert "HF_TOKEN=hf_test_token_12345" in content

    def test_token_not_set(self):
        from src.setup import _setup_hf_token
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HF_TOKEN", None)
            ok, detail = _setup_hf_token()
            assert ok is False


class TestEnvFile:
    """_load_env_file 테스트."""

    def test_load_env_file(self, tmp_path):
        from src.setup import _load_env_file
        env_dir = tmp_path / ".sonote"
        env_dir.mkdir()
        env_file = env_dir / ".env"
        env_file.write_text("TEST_SONOTE_VAR=hello_world\n")

        with patch.dict(os.environ, {}, clear=True):
            with patch("src.setup.Path.home", return_value=tmp_path):
                _load_env_file()
                assert os.environ.get("TEST_SONOTE_VAR") == "hello_world"

    def test_load_env_no_override(self, tmp_path):
        """기존 환경변수는 덮어쓰지 않음."""
        from src.setup import _load_env_file
        env_dir = tmp_path / ".sonote"
        env_dir.mkdir()
        env_file = env_dir / ".env"
        env_file.write_text("EXISTING_VAR=from_file\n")

        with patch.dict(os.environ, {"EXISTING_VAR": "from_env"}):
            with patch("src.setup.Path.home", return_value=tmp_path):
                _load_env_file()
                assert os.environ["EXISTING_VAR"] == "from_env"


class TestCLIIntegration:
    """CLI 통합 테스트 — sonote 엔트리포인트 사용."""

    def test_setup_help(self):
        """sonote setup --help 실행."""
        result = subprocess.run(
            ["sonote", "setup", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "환경 설정" in result.stdout or "setup" in result.stdout

    def test_setup_json(self):
        """sonote setup --json 실행."""
        result = subprocess.run(
            ["sonote", "setup", "--json"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "items" in data
        assert "summary" in data

    def test_doctor_still_works(self):
        """doctor 커맨드가 여전히 동작."""
        result = subprocess.run(
            ["sonote", "doctor", "--json"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "items" in data
