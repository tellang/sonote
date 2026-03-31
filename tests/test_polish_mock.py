"""polish.py LLM 호출 mock 테스트 — Ollama/Gemini 응답 시뮬레이션, 캐시, 문맥 전달 검증.

pytest + pytest-asyncio 사용. httpx를 mock하여 외부 의존성 없이 테스트.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from src.polish import (
    _build_domain_keyword_hint,
    _compute_batch_hash,
    _correct_batch_ollama,
    _correction_cache,
    _get_cached_correction,
    _run_ollama,
    _set_cached_correction,
    _run_gemini,
    extract_keywords_with_gemini,
    extract_keywords_with_ollama,
    summarize_with_ollama,
)


# ---------------------------------------------------------------
# Fixture: 테스트 간 캐시 격리
# ---------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_correction_cache():
    """각 테스트 전후로 보정 캐시를 초기화한다."""
    _correction_cache.clear()
    yield
    _correction_cache.clear()


# ---------------------------------------------------------------
# 1. httpx.AsyncClient mock — Ollama 응답 시뮬레이션
# ---------------------------------------------------------------

class TestRunOllama:
    """_run_ollama의 httpx.post mock 테스트.

    polish.py는 함수 내부에서 `import httpx`를 지연 임포트하므로
    'httpx.post'를 직접 mock한다 (모듈 레벨 속성이 아님).
    """

    def test_정상_응답(self):
        """Ollama가 200 + 유효한 JSON을 반환하면 (True, response_text)."""
        import httpx as _httpx

        fake_resp = mock.MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"response": "교정된 텍스트입니다."}

        with mock.patch.object(_httpx, "post", return_value=fake_resp) as mock_post:
            ok, text = _run_ollama("테스트 프롬프트", model="gemma3:27b", timeout=30)

        assert ok is True
        assert text == "교정된 텍스트입니다."
        # httpx.post가 올바른 URL과 payload로 호출되었는지
        call_args = mock_post.call_args
        assert "/api/generate" in call_args.args[0]
        assert call_args.kwargs["json"]["model"] == "gemma3:27b"
        assert call_args.kwargs["json"]["stream"] is False

    def test_타임아웃(self):
        """httpx.post가 타임아웃 예외를 발생시키면 (False, 에러 메시지)."""
        import httpx as _httpx

        with mock.patch.object(
            _httpx, "post",
            side_effect=_httpx.TimeoutException("Connection timed out"),
        ):
            ok, text = _run_ollama("프롬프트", timeout=5)

        assert ok is False
        assert "timed out" in text.lower() or "timeout" in text.lower()

    def test_빈_응답(self):
        """Ollama가 200이지만 response 필드가 빈 문자열이면 (True, '')."""
        import httpx as _httpx

        fake_resp = mock.MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"response": ""}

        with mock.patch.object(_httpx, "post", return_value=fake_resp):
            ok, text = _run_ollama("프롬프트")

        assert ok is True
        assert text == ""

    def test_HTTP_에러_상태코드(self):
        """Ollama가 500 등 에러 코드를 반환하면 (False, 'HTTP 500')."""
        import httpx as _httpx

        fake_resp = mock.MagicMock()
        fake_resp.status_code = 500

        with mock.patch.object(_httpx, "post", return_value=fake_resp):
            ok, text = _run_ollama("프롬프트")

        assert ok is False
        assert "500" in text

    def test_연결_에러(self):
        """httpx.post가 연결 에러를 발생시키면 (False, 에러 메시지)."""
        import httpx as _httpx

        with mock.patch.object(
            _httpx, "post",
            side_effect=ConnectionError("Connection refused"),
        ):
            ok, text = _run_ollama("프롬프트")

        assert ok is False
        assert "refused" in text.lower() or "connection" in text.lower()

    def test_response_키_누락(self):
        """JSON에 'response' 키가 없으면 빈 문자열 반환."""
        import httpx as _httpx

        fake_resp = mock.MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"model": "gemma3:27b"}  # response 키 없음

        with mock.patch.object(_httpx, "post", return_value=fake_resp):
            ok, text = _run_ollama("프롬프트")

        assert ok is True
        assert text == ""  # .get("response", "") 동작


# ---------------------------------------------------------------
# 2. Gemini API 응답 mock 테스트 (summarize_with_gemini는 CLI 기반이므로
#    summarize_with_ollama로 Ollama 요약 경로를 검증)
# ---------------------------------------------------------------

class TestSummarizeWithOllama:
    """summarize_with_ollama의 _run_ollama mock 테스트."""

    def _make_meeting_md(self, tmpdir: str) -> Path:
        """테스트용 회의록 파일 생성."""
        md_path = Path(tmpdir) / "meeting.md"
        md_path.write_text(
            "\n".join([
                "# 회의 내용 요약",
                "",
                "- 일시: 2026-03-14 10:00:00",
                "- 참석자: A, B (2명)",
                "- 총 세그먼트: 3개",
                "- (요약은 회의 후 별도 작성)",
                "",
                "---",
                "",
                "# To-do",
                "",
                "- [ ] (회의 후 별도 작성)",
                "",
                "---",
                "",
                "# Raw Data",
                "",
                "- [00:00:01] [A] 안녕하세요.",
                "- [00:00:05] [B] 프로젝트 일정 논의합시다.",
                "- [00:00:10] [A] 좋습니다.",
            ]),
            encoding="utf-8",
        )
        return md_path

    def test_정상_요약_생성(self):
        """Ollama가 구조화된 요약을 반환하면 파일에 반영된다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = self._make_meeting_md(tmpdir)

            fake_response = (
                "=== 요약 ===\n"
                "## 주제\n프로젝트 일정 논의 회의\n\n"
                "## 참석자 역할\n- A: 진행자\n- B: 참석자\n\n"
                "## 안건별 논의\n### 1. 일정 조율\n- 일정 합의\n\n"
                "## 결정사항\n- 다음 주 착수\n\n"
                "## 결론\n일정 합의 완료\n\n"
                "=== To-do ===\n"
                "- [ ] [A] 일정표 작성 (기한: 다음 주)"
            )

            with mock.patch("src.polish._run_ollama", return_value=(True, fake_response)):
                result = summarize_with_ollama(md_path)

            assert result is True
            content = md_path.read_text(encoding="utf-8")
            # placeholder가 요약으로 교체되었는지
            assert "- (요약은 회의 후 별도 작성)" not in content
            assert "## 주제" in content
            assert "프로젝트 일정 논의 회의" in content
            # To-do도 교체되었는지
            assert "- [ ] (회의 후 별도 작성)" not in content
            assert "일정표 작성" in content

    def test_요약_실패시_파일_미변경(self):
        """_run_ollama가 실패하면 파일이 변경되지 않는다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = self._make_meeting_md(tmpdir)
            original = md_path.read_text(encoding="utf-8")

            with mock.patch("src.polish._run_ollama", return_value=(False, "타임아웃")):
                result = summarize_with_ollama(md_path)

            assert result is False
            assert md_path.read_text(encoding="utf-8") == original

    def test_구분자_없는_응답도_요약_반영(self):
        """'=== 요약 ===' 구분자 없이 텍스트만 오면 요약 블록으로 사용된다."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = self._make_meeting_md(tmpdir)

            plain_summary = "## 주제\n단순 요약 테스트\n## 결론\n끝"
            with mock.patch("src.polish._run_ollama", return_value=(True, plain_summary)):
                result = summarize_with_ollama(md_path)

            assert result is True
            content = md_path.read_text(encoding="utf-8")
            # 구분자 없으면 summary_text = result.strip() → placeholder 교체
            assert "- (요약은 회의 후 별도 작성)" not in content
            assert "단순 요약 테스트" in content

    def test_참석자_정보가_프롬프트에_포함됨(self):
        """메타데이터의 참석자 정보가 Ollama 프롬프트에 반영되는지 검증."""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = self._make_meeting_md(tmpdir)
            captured_prompts: list[str] = []

            def _capture_ollama(prompt: str, model: str = "", timeout: int = 0):
                captured_prompts.append(prompt)
                return True, "=== 요약 ===\n## 주제\n테스트\n=== To-do ===\n- [ ] 없음"

            with mock.patch("src.polish._run_ollama", side_effect=_capture_ollama):
                summarize_with_ollama(md_path)

            assert len(captured_prompts) == 1
            assert "A, B" in captured_prompts[0]


# ---------------------------------------------------------------
# 3. SHA256 캐시 히트/미스 테스트
# ---------------------------------------------------------------

class TestCorrectionCache:
    """SHA256 해시 기반 보정 캐시 동작 검증."""

    def test_캐시_미스_후_저장_후_히트(self):
        """동일 입력의 해시로 캐시 저장 → 조회 시 히트."""
        lines = ["- [00:00:01] [A] 안녕하세여."]
        batch_hash = _compute_batch_hash(lines)

        # 미스
        assert _get_cached_correction(batch_hash) is None

        # 저장
        corrected = ["- [00:00:01] [A] 안녕하세요."]
        _set_cached_correction(batch_hash, corrected)

        # 히트
        assert _get_cached_correction(batch_hash) == corrected

    def test_다른_입력은_캐시_미스(self):
        """다른 텍스트의 해시로는 캐시가 히트하지 않는다."""
        lines_a = ["- [00:00:01] [A] 테스트 문장"]
        lines_b = ["- [00:00:01] [A] 다른 문장"]
        hash_a = _compute_batch_hash(lines_a)
        hash_b = _compute_batch_hash(lines_b)

        _set_cached_correction(hash_a, ["- [00:00:01] [A] 교정된 문장"])

        assert hash_a != hash_b
        assert _get_cached_correction(hash_a) is not None
        assert _get_cached_correction(hash_b) is None

    def test_해시_결정성(self):
        """동일 입력은 항상 동일 해시를 생성한다."""
        lines = ["라인1", "라인2", "라인3"]
        h1 = _compute_batch_hash(lines)
        h2 = _compute_batch_hash(lines)
        assert h1 == h2
        # 수동 검증
        expected = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
        assert h1 == expected

    def test_캐시_최대_500_엔트리_LRU_제거(self):
        """캐시가 500개를 초과하면 가장 오래된 항목이 제거된다."""
        # 500개 채우기
        for i in range(500):
            _set_cached_correction(f"hash_{i:04d}", [f"line_{i}"])
        assert len(_correction_cache) == 500
        assert _get_cached_correction("hash_0000") is not None

        # 501번째 추가 → 가장 오래된 hash_0000 제거
        _set_cached_correction("hash_0500", ["line_500"])
        assert len(_correction_cache) == 500
        assert _get_cached_correction("hash_0000") is None
        assert _get_cached_correction("hash_0500") == ["line_500"]

    def test_correct_batch_ollama_캐시_히트시_LLM_미호출(self):
        """캐시에 결과가 있으면 Ollama API를 호출하지 않는다."""
        lines = ["- [00:00:01] [A] 테스트"]
        batch_hash = _compute_batch_hash(lines)
        cached_result = ["- [00:00:01] [A] 교정됨"]
        _set_cached_correction(batch_hash, cached_result)

        with mock.patch("src.polish._run_ollama") as mock_ollama:
            idx, ok, result = _correct_batch_ollama(lines, batch_idx=0)

        # LLM 호출 없음
        mock_ollama.assert_not_called()
        assert ok is True
        assert result == cached_result


# ---------------------------------------------------------------
# 4. 슬라이딩 윈도우 문맥 전달 검증
# ---------------------------------------------------------------

class TestSlidingWindowContext:
    """_correct_batch_ollama에서 이전/다음 배치 문맥이 프롬프트에 포함되는지 검증."""

    def test_이전_문맥이_프롬프트에_포함됨(self):
        """prev_context 인자가 Ollama 프롬프트에 '[이전 문맥]'으로 포함된다."""
        batch_lines = ["- [00:01:00] [A] 교정 대상 문장"]
        prev_ctx = ["- [00:00:50] [B] 이전 배치 마지막 문장"]
        captured_prompts: list[str] = []

        def _capture(prompt: str, model: str = "", timeout: int = 0):
            captured_prompts.append(prompt)
            # 줄 수 일치하는 응답 반환
            return True, "- [00:01:00] [A] 교정된 문장"

        with mock.patch("src.polish._run_ollama", side_effect=_capture):
            _correct_batch_ollama(
                batch_lines, batch_idx=0,
                prev_context=prev_ctx,
            )

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "이전 문맥" in prompt
        assert "이전 배치 마지막 문장" in prompt

    def test_다음_문맥이_프롬프트에_포함됨(self):
        """next_context 인자가 Ollama 프롬프트에 '[다음 문맥]'으로 포함된다."""
        batch_lines = ["- [00:01:00] [A] 교정 대상"]
        next_ctx = ["- [00:01:10] [B] 다음 배치 첫 문장"]
        captured_prompts: list[str] = []

        def _capture(prompt: str, model: str = "", timeout: int = 0):
            captured_prompts.append(prompt)
            return True, "- [00:01:00] [A] 교정됨"

        with mock.patch("src.polish._run_ollama", side_effect=_capture):
            _correct_batch_ollama(
                batch_lines, batch_idx=0,
                next_context=next_ctx,
            )

        prompt = captured_prompts[0]
        assert "다음 문맥" in prompt
        assert "다음 배치 첫 문장" in prompt

    def test_양방향_문맥_동시_전달(self):
        """prev_context와 next_context가 모두 프롬프트에 포함된다."""
        batch_lines = ["- [00:01:00] [A] 중간 배치"]
        prev_ctx = ["- [00:00:55] [B] 이전"]
        next_ctx = ["- [00:01:05] [C] 다음"]
        captured_prompts: list[str] = []

        def _capture(prompt: str, model: str = "", timeout: int = 0):
            captured_prompts.append(prompt)
            return True, "- [00:01:00] [A] 교정됨"

        with mock.patch("src.polish._run_ollama", side_effect=_capture):
            _correct_batch_ollama(
                batch_lines, batch_idx=1,
                prev_context=prev_ctx, next_context=next_ctx,
            )

        prompt = captured_prompts[0]
        assert "이전 문맥" in prompt and "이전" in prompt
        assert "다음 문맥" in prompt and "다음" in prompt

    def test_문맥_없으면_문맥_블록_생략(self):
        """prev/next_context가 None이면 문맥 관련 텍스트가 프롬프트에 없다."""
        batch_lines = ["- [00:01:00] [A] 단독 배치"]
        captured_prompts: list[str] = []

        def _capture(prompt: str, model: str = "", timeout: int = 0):
            captured_prompts.append(prompt)
            return True, "- [00:01:00] [A] 교정됨"

        with mock.patch("src.polish._run_ollama", side_effect=_capture):
            _correct_batch_ollama(batch_lines, batch_idx=0)

        prompt = captured_prompts[0]
        assert "이전 문맥" not in prompt
        assert "다음 문맥" not in prompt


# ---------------------------------------------------------------
# 5. 도메인 키워드 주입 테스트
# ---------------------------------------------------------------

class TestDomainKeywordInjection:
    """domain_keywords.py 사전이 프롬프트에 반영되는지 검증."""

    def test_build_domain_keyword_hint_비어있지_않음(self):
        """_build_domain_keyword_hint가 비어있지 않은 힌트 문자열을 반환한다."""
        hint = _build_domain_keyword_hint()
        assert len(hint) > 0
        # 키워드가 쉼표로 구분되어야 함
        assert "," in hint

    def test_max_chars_제한_준수(self):
        """max_chars를 초과하지 않는다."""
        hint = _build_domain_keyword_hint(max_chars=50)
        assert len(hint) <= 50

    def test_프롬프트에_전문_용어_힌트_포함(self):
        """_correct_batch_ollama 프롬프트에 도메인 키워드 힌트가 포함된다."""
        batch_lines = ["- [00:00:01] [A] 테스트 문장"]
        captured_prompts: list[str] = []

        def _capture(prompt: str, model: str = "", timeout: int = 0):
            captured_prompts.append(prompt)
            return True, "- [00:00:01] [A] 교정됨"

        with mock.patch("src.polish._run_ollama", side_effect=_capture):
            _correct_batch_ollama(batch_lines, batch_idx=0)

        prompt = captured_prompts[0]
        assert "전문 용어" in prompt
        # DEFAULT_MEETING_PROMPT_KEYWORDS 중 일부가 포함되는지
        # (예: R&R, 스프린트 등 — 첫 200자 내 키워드)
        hint = _build_domain_keyword_hint()
        first_keyword = hint.split(",")[0].strip()
        assert first_keyword in prompt

    def test_extract_keywords_with_ollama_정상(self):
        """Ollama 키워드 추출이 쉼표 구분 목록을 파싱한다."""
        fake_result = "Python, FastAPI, 스프린트, Docker"
        with mock.patch("src.polish._run_ollama", return_value=(True, fake_result)):
            keywords = extract_keywords_with_ollama("회의 텍스트 샘플")

        assert "Python" in keywords
        assert "FastAPI" in keywords
        assert "스프린트" in keywords
        assert "Docker" in keywords

    def test_extract_keywords_ollama_실패시_빈_목록(self):
        """Ollama 호출 실패 시 빈 리스트를 반환한다."""
        with mock.patch("src.polish._run_ollama", return_value=(False, "에러")):
            keywords = extract_keywords_with_ollama("텍스트")

        assert keywords == []


class TestGeminiMcpWarningSanitization:
    """Gemini 출력의 MCP 경고문 제거 회귀를 검증한다."""

    def test_run_gemini_strips_mcp_warning_line(self):
        fake_result = mock.MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = (
            "MCP issues detected. Run /mcp list for status.\n"
            "Python, FastAPI"
        )
        fake_result.stderr = ""

        with mock.patch("src.polish.subprocess.run", return_value=fake_result):
            ok, text = _run_gemini("키워드 추출 프롬프트")

        assert ok is True
        assert "MCP issues detected" not in text
        assert text == "Python, FastAPI"

    def test_extract_keywords_with_gemini_excludes_warning_token(self):
        fake_result = mock.MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = (
            "MCP issues detected. Run /mcp list for status.\n"
            "LangGraph, Whisper, FastAPI"
        )
        fake_result.stderr = ""

        with mock.patch("src.polish.subprocess.run", return_value=fake_result):
            keywords = extract_keywords_with_gemini("회의 텍스트")

        assert "MCP issues detected. Run /mcp list for status." not in keywords
        assert keywords == ["LangGraph", "Whisper", "FastAPI"]


# ---------------------------------------------------------------
# 6. 에러 핸들링 — LLM 타임아웃, 빈 응답, 줄 수 불일치
# ---------------------------------------------------------------

class TestErrorHandling:
    """_correct_batch_ollama의 에러 상황 처리 검증."""

    def test_LLM_타임아웃시_원본_유지(self):
        """Ollama가 타임아웃하면 원본 라인이 그대로 반환된다."""
        original = ["- [00:00:01] [A] 원본 텍스트"]

        with mock.patch("src.polish._run_ollama", return_value=(False, "타임아웃")):
            idx, ok, result = _correct_batch_ollama(original, batch_idx=0)

        assert ok is False
        assert result == original

    def test_빈_응답시_원본_유지(self):
        """Ollama가 빈 문자열을 반환하면 줄 수 불일치로 원본이 유지된다."""
        original = ["- [00:00:01] [A] 원본 텍스트", "- [00:00:05] [B] 두 번째"]

        with mock.patch("src.polish._run_ollama", return_value=(True, "")):
            idx, ok, result = _correct_batch_ollama(original, batch_idx=0)

        # 빈 응답 → result_lines가 0줄 → 줄 수 불일치 → 원본 유지
        assert ok is False
        assert result == original

    def test_줄수_불일치_응답시_원본_유지(self):
        """Ollama가 줄 수가 다른 응답을 반환하면 원본이 유지된다."""
        original = [
            "- [00:00:01] [A] 첫째",
            "- [00:00:05] [B] 둘째",
            "- [00:00:10] [A] 셋째",
        ]
        # 2줄만 반환 (3줄이어야 함)
        bad_response = "- [00:00:01] [A] 교정 첫째\n- [00:00:05] [B] 교정 둘째"

        with mock.patch("src.polish._run_ollama", return_value=(True, bad_response)):
            idx, ok, result = _correct_batch_ollama(original, batch_idx=0)

        assert ok is False
        assert result == original

    def test_줄수_일치_응답시_교정_적용(self):
        """줄 수가 일치하면 교정 결과가 반환되고 캐시에 저장된다."""
        original = [
            "- [00:00:01] [A] 안녕하세여",
            "- [00:00:05] [B] 반갑읍니다",
        ]
        good_response = (
            "- [00:00:01] [A] 안녕하세요\n"
            "- [00:00:05] [B] 반갑습니다"
        )

        with mock.patch("src.polish._run_ollama", return_value=(True, good_response)):
            idx, ok, result = _correct_batch_ollama(original, batch_idx=0)

        assert ok is True
        assert result == ["- [00:00:01] [A] 안녕하세요", "- [00:00:05] [B] 반갑습니다"]

        # 캐시에 저장되었는지
        batch_hash = _compute_batch_hash(original)
        assert _get_cached_correction(batch_hash) == result

    def test_줄수_초과_응답시_원본_유지(self):
        """Ollama가 추가 줄을 생성하면 원본이 유지된다."""
        original = ["- [00:00:01] [A] 단일 라인"]
        extra_response = "- [00:00:01] [A] 교정됨\n- [00:00:01] [A] 추가된 줄\n설명 줄"

        with mock.patch("src.polish._run_ollama", return_value=(True, extra_response)):
            idx, ok, result = _correct_batch_ollama(original, batch_idx=0)

        assert ok is False
        assert result == original


# ---------------------------------------------------------------
# 7. is_ollama_available mock 테스트
# ---------------------------------------------------------------

class TestIsOllamaAvailable:
    """is_ollama_available의 httpx.get mock 테스트."""

    def test_서버_가용_모델_있음(self):
        """Ollama 서버가 200 + 해당 모델이 목록에 있으면 True."""
        import httpx as _httpx
        from src.polish import is_ollama_available

        fake_resp = mock.MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "models": [{"name": "gemma3:27b"}, {"name": "llama3:8b"}],
        }

        with mock.patch.object(_httpx, "get", return_value=fake_resp):
            assert is_ollama_available("gemma3:27b") is True

    def test_서버_가용_모델_없음(self):
        """Ollama 서버에 요청 모델이 없으면 False."""
        import httpx as _httpx
        from src.polish import is_ollama_available

        fake_resp = mock.MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "models": [{"name": "llama3:8b"}],
        }

        with mock.patch.object(_httpx, "get", return_value=fake_resp):
            assert is_ollama_available("gemma3:27b") is False

    def test_서버_미가용(self):
        """Ollama 서버 연결 불가 시 False."""
        import httpx as _httpx
        from src.polish import is_ollama_available

        with mock.patch.object(_httpx, "get", side_effect=ConnectionError("refused")):
            assert is_ollama_available() is False
