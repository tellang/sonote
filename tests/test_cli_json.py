"""CLI JSON 출력 및 structured exit code 테스트."""

from __future__ import annotations

import argparse
import json
import sys
from types import ModuleType
from unittest.mock import patch

import pytest

from src import cli


def _run_cli(argv: list[str], capsys: pytest.CaptureFixture[str]) -> dict:
    with patch.object(sys, "argv", ["media-transcriber", *argv]):
        cli.main()
    return json.loads(capsys.readouterr().out)


@pytest.fixture
def fake_transcribe_module() -> ModuleType:
    module = ModuleType("src.transcribe")

    def _transcribe_audio(*args, **kwargs):
        return [{"start": 0.0, "end": 1.0, "text": "테스트"}]

    def _transcribe_chunks(*args, **kwargs):
        return [{"start": 0.0, "end": 1.0, "text": "청크 테스트"}]

    def _save_transcript(*args, **kwargs):
        return None

    module.transcribe_audio = _transcribe_audio
    module.transcribe_chunks = _transcribe_chunks
    module.save_transcript = _save_transcript
    return module


def test_json_flag_outputs_structured_payload(
    capsys: pytest.CaptureFixture[str],
    fake_transcribe_module: ModuleType,
) -> None:
    with patch.dict(sys.modules, {"src.transcribe": fake_transcribe_module}):
        payload = _run_cli(["--json", "transcribe", "sample.wav"], capsys)

    assert set(payload) >= {"status", "command", "data"}
    assert payload["status"] == "success"
    assert payload["command"] == "transcribe"
    assert payload["data"]["segments"] == 1
    assert payload["data"]["format"] == "txt"


def test_schema_subcommand_outputs_all_command_schemas(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _run_cli(["schema"], capsys)

    assert payload["status"] == "success"
    assert payload["command"] == "schema"
    commands = payload["data"]["commands"]
    assert {"transcribe", "live", "download", "probe", "scan", "smart", "profile", "enroll", "meeting", "approve-profiles", "status", "auto", "detect", "map", "approve"} <= set(commands)


def test_schema_specific_command_outputs_single_schema(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _run_cli(["schema", "transcribe"], capsys)

    assert payload["status"] == "success"
    assert payload["command"] == "schema"
    assert payload["data"]["name"].endswith("transcribe")
    parameter_names = {item["name"] for item in payload["data"]["parameters"]}
    assert {"audio", "output", "model", "language", "fmt", "dry_run"} <= parameter_names


def test_invalid_arguments_exit_with_code_2(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.object(
        sys,
        "argv",
        ["media-transcriber", "transcribe", "sample.wav", "--fmt", "docx"],
    ):
        with pytest.raises(SystemExit) as exc:
            cli.main()

    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_dry_run_outputs_configuration_json_without_running_transcription(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.runtime_env as runtime_env

    monkeypatch.setattr(runtime_env, "detect_device", lambda: ("cuda", "float16"))

    with patch.object(cli, "_cmd_transcribe") as handler:
        payload = _run_cli(
            ["--json", "transcribe", "sample.wav", "--dry-run", "--model", "tiny"],
            capsys,
        )

    handler.assert_not_called()
    assert payload["status"] == "success"
    assert payload["command"] == "transcribe"
    assert payload["data"]["dry_run"] is True
    assert payload["data"]["config"]["command"] == "transcribe"
    assert payload["data"]["config"]["model"] == "tiny"
    assert payload["data"]["config"]["device"] == "cuda"


# ===================================================================
# 1) Dynamic Command Surface — check_capabilities 테스트
# ===================================================================


def test_check_capabilities_returns_all_keys() -> None:
    """check_capabilities()가 필수 키를 모두 포함하는지 검증."""
    caps = cli.check_capabilities()
    expected_keys = {
        "cuda_available",
        "ffmpeg_available",
        "diarize_available",
        "yt_dlp_available",
        "ollama_available",
        "tray_available",
    }
    assert expected_keys <= set(caps)
    # 각 값은 bool
    for key in expected_keys:
        assert isinstance(caps[key], bool), f"{key}는 bool이어야 합니다"


def test_schema_includes_capabilities(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """schema 출력에 capabilities 필드가 포함되는지 검증."""
    payload = _run_cli(["schema"], capsys)
    assert payload["status"] == "success"
    assert "capabilities" in payload["data"]
    caps = payload["data"]["capabilities"]
    assert "cuda_available" in caps
    assert "ffmpeg_available" in caps


def test_schema_single_command_includes_capabilities(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """특정 서브커맨드 schema에도 capabilities가 포함되는지 검증."""
    payload = _run_cli(["schema", "transcribe"], capsys)
    assert payload["status"] == "success"
    assert "capabilities" in payload["data"]


# ===================================================================
# 2) Explicit Handoff [MANUAL] — 수동 단계 표시 테스트
# ===================================================================


def test_manual_steps_hf_token_missing_for_meeting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HF_TOKEN 미설정 시 meeting 커맨드에 MANUAL 단계가 포함되는지 검증."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    args = argparse.Namespace(command="meeting", no_diarize=False)
    steps = cli._collect_manual_steps(args)
    assert any(s["tag"] == "MANUAL" and "HF_TOKEN" in s["message"] for s in steps)


def test_manual_steps_empty_when_hf_token_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HF_TOKEN 설정 시 meeting 관련 MANUAL 단계가 없어야 함."""
    monkeypatch.setenv("HF_TOKEN", "hf_test123")
    monkeypatch.setenv("MEETING_API_KEY", "secret")
    args = argparse.Namespace(command="meeting", no_diarize=False)
    steps = cli._collect_manual_steps(args)
    hf_steps = [s for s in steps if "HF_TOKEN" in s["message"]]
    assert len(hf_steps) == 0


def test_manual_steps_ffmpeg_missing_for_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ffmpeg 미설치 시 live 커맨드에 MANUAL 단계 포함."""
    import shutil as _shutil
    monkeypatch.setattr(_shutil, "which", lambda x: None)
    args = argparse.Namespace(command="live")
    steps = cli._collect_manual_steps(args)
    assert any(s["tag"] == "MANUAL" and "ffmpeg" in s["message"] for s in steps)


# ===================================================================
# 3) Preflight Diagnostics — 사전 점검 테스트
# ===================================================================


def test_preflight_check_returns_structure() -> None:
    """preflight_check()가 올바른 구조를 반환하는지 검증."""
    args = argparse.Namespace(command="transcribe", audio="nonexist.wav", cpu=False)
    result = cli.preflight_check(args)
    assert "passed" in result
    assert "checks" in result
    assert "manual_steps" in result
    assert isinstance(result["checks"], list)
    assert isinstance(result["manual_steps"], list)


def test_preflight_transcribe_includes_gpu_check() -> None:
    """transcribe preflight에 gpu 점검이 포함되는지 검증."""
    args = argparse.Namespace(command="transcribe", audio="test.wav", cpu=False)
    result = cli.preflight_check(args)
    check_names = {c["name"] for c in result["checks"]}
    assert "gpu" in check_names


def test_preflight_live_fails_without_ffmpeg_and_ytdlp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """live 커맨드에서 ffmpeg/yt-dlp 미설치 시 preflight 실패."""
    import shutil as _shutil
    monkeypatch.setattr(_shutil, "which", lambda x: None)
    args = argparse.Namespace(command="live")
    result = cli.preflight_check(args)
    assert result["passed"] is False
    failed_names = {c["name"] for c in result["checks"] if not c["passed"]}
    assert "ffmpeg" in failed_names
    assert "yt_dlp" in failed_names


def test_preflight_fail_exits_with_code_3(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """preflight 실패 시 exit code 3으로 종료."""
    import shutil as _shutil
    monkeypatch.setattr(_shutil, "which", lambda x: None)
    with patch.object(sys, "argv", ["media-transcriber", "--json", "live", "https://youtube.com/watch?v=test"]):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == cli.EXIT_PREFLIGHT_FAIL
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["status"] == "error"
    assert payload["code"] == "PREFLIGHT_FAIL"
    assert "diagnostics" in payload["data"]


def test_dry_run_includes_capabilities_and_diagnostics(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dry-run에 capabilities/diagnostics가 포함되는지 검증."""
    import src.runtime_env as runtime_env
    monkeypatch.setattr(runtime_env, "detect_device", lambda: ("cpu", "int8"))

    with patch.object(cli, "_cmd_transcribe") as handler:
        payload = _run_cli(
            ["--json", "transcribe", "sample.wav", "--dry-run"],
            capsys,
        )
    handler.assert_not_called()
    assert payload["data"]["dry_run"] is True
    assert "capabilities" in payload["data"]
    assert "diagnostics" in payload["data"]
    assert "manual_steps" in payload["data"]


# ===================================================================
# 4) Stream-friendly Output (NDJSON) 테스트
# ===================================================================


def test_ndjson_transcribe_outputs_line_delimited_json(
    capsys: pytest.CaptureFixture[str],
    fake_transcribe_module: ModuleType,
) -> None:
    """--ndjson transcribe가 줄 단위 JSON을 출력하는지 검증."""
    with patch.dict(sys.modules, {"src.transcribe": fake_transcribe_module}):
        with patch.object(sys, "argv", ["media-transcriber", "--ndjson", "transcribe", "sample.wav"]):
            cli.main()

    lines = capsys.readouterr().out.strip().split("\n")
    assert len(lines) >= 3  # start + segment(s) + complete

    # 각 줄이 독립 JSON
    parsed = [json.loads(line) for line in lines]

    # 첫 줄: start
    assert parsed[0]["event"] == "start"
    assert parsed[0]["command"] == "transcribe"

    # 중간: segment
    segments = [p for p in parsed if p["event"] == "segment"]
    assert len(segments) >= 1
    assert "text" in segments[0]
    assert "start" in segments[0]
    assert "end" in segments[0]

    # 마지막: complete
    assert parsed[-1]["event"] == "complete"
    assert parsed[-1]["segments"] == 1


def test_ndjson_line_produces_valid_json() -> None:
    """_ndjson_line()이 유효한 JSON을 생성하는지 검증."""
    line = cli._ndjson_line("test_event", key="value", num=42)
    parsed = json.loads(line)
    assert parsed["event"] == "test_event"
    assert parsed["key"] == "value"
    assert parsed["num"] == 42
    # 줄바꿈 없음 (단일 라인)
    assert "\n" not in line


def test_ndjson_manual_steps_output(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    fake_transcribe_module: ModuleType,
) -> None:
    """--ndjson에서 manual_steps가 줄 단위 출력되는지 검증."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("MEETING_API_KEY", raising=False)

    # meeting 커맨드는 복잡하므로 transcribe로 테스트 (manual_steps 없는 정상 케이스)
    with patch.dict(sys.modules, {"src.transcribe": fake_transcribe_module}):
        with patch.object(sys, "argv", ["media-transcriber", "--ndjson", "transcribe", "sample.wav"]):
            cli.main()

    out = capsys.readouterr().out.strip()
    lines = out.split("\n")
    # manual_step 이벤트가 없어야 함 (transcribe에는 수동 단계 없음)
    manual_lines = [json.loads(ln) for ln in lines if json.loads(ln).get("event") == "manual_step"]
    assert len(manual_lines) == 0


# ===================================================================
# 엣지케이스: check_capabilities — detect_device 예외 시 graceful fallback
# ===================================================================


def test_check_capabilities_graceful_when_detect_device_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """detect_device()가 예외를 던져도 check_capabilities()는 정상 반환한다."""
    import src.runtime_env as runtime_env

    monkeypatch.setattr(runtime_env, "detect_device", lambda: (_ for _ in ()).throw(RuntimeError("GPU 폭발")))
    caps = cli.check_capabilities()
    assert caps["cuda_available"] is False
    # 나머지 키도 존재해야 함
    assert "ffmpeg_available" in caps
    assert "diarize_available" in caps


def test_check_capabilities_all_tools_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """모든 외부 도구가 없을 때 모든 값이 False인지 검증."""
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda x: None)
    caps = cli.check_capabilities()
    assert caps["ffmpeg_available"] is False
    assert caps["yt_dlp_available"] is False
    assert caps["ollama_available"] is False


# ===================================================================
# 엣지케이스: _collect_manual_steps — 경계값 테스트
# ===================================================================


def test_manual_steps_enroll_hf_token_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """enroll 커맨드에서 HF_TOKEN 미설정 시 MANUAL 단계가 포함된다."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    args = argparse.Namespace(command="enroll")
    steps = cli._collect_manual_steps(args)
    assert any(s["tag"] == "MANUAL" and "임베딩" in s["message"] for s in steps)


def test_manual_steps_meeting_with_no_diarize_skips_hf_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """meeting + no_diarize=True 시 HF_TOKEN 관련 단계가 없어야 한다."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("MEETING_API_KEY", raising=False)
    args = argparse.Namespace(command="meeting", no_diarize=True)
    steps = cli._collect_manual_steps(args)
    hf_steps = [s for s in steps if "HF_TOKEN" in s["message"]]
    assert len(hf_steps) == 0
    # MEETING_API_KEY 경고는 여전히 있어야 함
    api_steps = [s for s in steps if "MEETING_API_KEY" in s["message"]]
    assert len(api_steps) == 1


def test_manual_steps_unknown_command_returns_empty() -> None:
    """알 수 없는 커맨드에 대해 빈 리스트를 반환한다."""
    args = argparse.Namespace(command="unknown_cmd")
    steps = cli._collect_manual_steps(args)
    assert steps == []


def test_manual_steps_no_command_attr_returns_empty() -> None:
    """command 속성이 없는 Namespace에 대해 빈 리스트를 반환한다."""
    args = argparse.Namespace()
    steps = cli._collect_manual_steps(args)
    assert steps == []


# ===================================================================
# 엣지케이스: preflight_check — meeting에서 ffmpeg 미설치는 passed=True
# ===================================================================


def test_preflight_meeting_passes_without_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """meeting에서 ffmpeg는 선택이므로 미설치여도 passed=True."""
    import shutil as _shutil

    original_which = _shutil.which

    def _mock_which(name: str) -> str | None:
        if name == "ffmpeg":
            return None
        return original_which(name)

    monkeypatch.setattr(_shutil, "which", _mock_which)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    args = argparse.Namespace(command="meeting", no_diarize=False)
    result = cli.preflight_check(args)
    assert result["passed"] is True
    ffmpeg_check = [c for c in result["checks"] if c["name"] == "ffmpeg"]
    assert len(ffmpeg_check) == 1
    assert ffmpeg_check[0]["required"] is False


def test_preflight_enroll_fails_without_hf_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """enroll에서 HF_TOKEN 미설정이면 preflight 실패."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    args = argparse.Namespace(command="enroll")
    result = cli.preflight_check(args)
    assert result["passed"] is False
    failed = [c for c in result["checks"] if c["name"] == "hf_token" and not c["passed"]]
    assert len(failed) == 1


# ===================================================================
# 엣지케이스: _ndjson_line — 한글/특수문자/빈 데이터
# ===================================================================


def test_ndjson_line_with_korean_text() -> None:
    """_ndjson_line()이 한글을 ensure_ascii=False로 올바르게 출력한다."""
    line = cli._ndjson_line("segment", text="안녕하세요 테스트")
    parsed = json.loads(line)
    assert parsed["text"] == "안녕하세요 테스트"
    assert "\\u" not in line  # ensure_ascii=False 확인


def test_ndjson_line_with_empty_data() -> None:
    """추가 데이터 없이 event만 있는 NDJSON 라인."""
    line = cli._ndjson_line("heartbeat")
    parsed = json.loads(line)
    assert parsed == {"event": "heartbeat"}


def test_ndjson_line_with_special_chars() -> None:
    """특수문자(따옴표, 백슬래시)가 올바르게 이스케이프된다."""
    line = cli._ndjson_line("test", text='he said "hello" \\ world')
    parsed = json.loads(line)
    assert parsed["text"] == 'he said "hello" \\ world'


# ===================================================================
# Issue 1: json_output reason enum 테스트
# ===================================================================


def test_json_output_error_includes_reason_enum() -> None:
    """에러 출력에 reason enum이 포함되는지 검증."""
    output = cli.json_output("error", "transcribe", error="파일 없음", code="NOT_FOUND")
    payload = json.loads(output)
    assert payload["reason"] == "notFound"


def test_json_output_error_reason_maps_all_codes() -> None:
    """모든 code → reason 매핑이 올바른지 검증."""
    mappings = {
        "ERROR": "runtimeError",
        "ARG_ERROR": "argError",
        "MODEL_ERROR": "modelError",
        "NOT_FOUND": "notFound",
        "PREFLIGHT_FAIL": "preflightFail",
    }
    for code, expected_reason in mappings.items():
        output = cli.json_output("error", "test", error="msg", code=code)
        payload = json.loads(output)
        assert payload["reason"] == expected_reason, f"code={code}"


def test_json_output_error_without_code_defaults_to_runtime_error() -> None:
    """code 없는 에러 시 reason이 runtimeError로 기본값."""
    output = cli.json_output("error", "test", error="알 수 없는 오류")
    payload = json.loads(output)
    assert payload["reason"] == "runtimeError"


def test_json_output_success_has_no_reason() -> None:
    """성공 출력에는 reason이 없어야 한다."""
    output = cli.json_output("success", "test", data={"key": "value"})
    payload = json.loads(output)
    assert "reason" not in payload


def test_preflight_fail_json_includes_reason(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """preflight 실패 시 JSON 출력에 reason=preflightFail 포함."""
    import shutil as _shutil
    monkeypatch.setattr(_shutil, "which", lambda x: None)
    with patch.object(sys, "argv", ["media-transcriber", "--json", "live", "https://youtube.com/watch?v=test"]):
        with pytest.raises(SystemExit):
            cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["reason"] == "preflightFail"


# ===================================================================
# Issue 2: schema 의존성 정보 테스트
# ===================================================================


def test_schema_live_includes_dependencies(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """live 서브커맨드 schema에 dependencies가 포함되는지 검증."""
    payload = _run_cli(["schema", "live"], capsys)
    assert payload["status"] == "success"
    deps = payload["data"]["dependencies"]
    dep_names = {d["name"] for d in deps}
    assert "ffmpeg" in dep_names
    assert "yt-dlp" in dep_names


def test_schema_transcribe_includes_gpu_dependency(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """transcribe schema에 GPU/CUDA 의존성이 포함되는지 검증."""
    payload = _run_cli(["schema", "transcribe"], capsys)
    deps = payload["data"].get("dependencies", [])
    dep_names = {d["name"] for d in deps}
    assert "GPU/CUDA" in dep_names


def test_schema_all_commands_includes_dependencies(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """전체 schema에서 live 커맨드에 dependencies가 포함되는지 검증."""
    payload = _run_cli(["schema"], capsys)
    live_schema = payload["data"]["commands"]["live"]
    assert "dependencies" in live_schema
    dep_names = {d["name"] for d in live_schema["dependencies"]}
    assert "ffmpeg" in dep_names


def test_schema_profile_has_no_dependencies(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """의존성 없는 서브커맨드(profile)에는 dependencies 키가 없다."""
    payload = _run_cli(["schema", "profile"], capsys)
    assert "dependencies" not in payload["data"]


# ===================================================================
# Issue 3: status 서브커맨드 테스트
# ===================================================================


def test_status_subcommand_outputs_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """status --json이 capabilities/diagnostics를 포함하는 JSON을 출력."""
    payload = _run_cli(["--json", "status"], capsys)
    assert payload["status"] == "success"
    assert payload["command"] == "status"
    data = payload["data"]
    assert "capabilities" in data
    assert "diagnostics" in data
    assert "manual_steps" in data
    assert "passed" in data


def test_status_subcommand_plain_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """status (--json 없이)가 유효한 JSON을 출력."""
    with patch.object(sys, "argv", ["media-transcriber", "status"]):
        cli.main()
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "capabilities" in parsed
    assert "passed" in parsed


def test_status_capabilities_contain_expected_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """status 출력의 capabilities에 필수 키가 모두 존재."""
    payload = _run_cli(["--json", "status"], capsys)
    caps = payload["data"]["capabilities"]
    expected = {"cuda_available", "ffmpeg_available", "diarize_available", "yt_dlp_available", "ollama_available", "tray_available"}
    assert expected <= set(caps)
