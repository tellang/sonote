"""세션 검색 API 테스트.

GET /api/sessions/{session_id}/search 엔드포인트의
정상 검색, 정규식, 잘못된 패턴, 필터 조합, 빈 결과를 검증한다.
"""

import json

import httpx
import pytest
from httpx import ASGITransport

from src import server


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """httpx AsyncClient를 FastAPI 앱에 연결한다."""
    transport = ASGITransport(app=server.app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture(autouse=True)
def _reset_api_key():
    """테스트 중 API 키 인증을 비활성화한다."""
    original = server._api_key
    server._api_key = ""
    yield
    server._api_key = original


def _create_session_with_jsonl(
    root, date_str, time_str, segments,
):
    """테스트용 세션 디렉토리 + JSONL 파일을 생성하는 헬퍼.

    Args:
        root: OUTPUT_ROOT에 해당하는 tmp_path.
        date_str: 날짜 문자열 (YYYY-MM-DD).
        time_str: 시간 문자열 (HHMMSS).
        segments: JSONL에 쓸 세그먼트 dict 리스트.

    Returns:
        생성된 세션 디렉토리 Path.
    """
    session_dir = root / "meetings" / date_str / time_str
    session_dir.mkdir(parents=True, exist_ok=True)

    jsonl_lines = [json.dumps(seg, ensure_ascii=False) for seg in segments]
    (session_dir / "meeting.stt.jsonl").write_text(
        "\n".join(jsonl_lines), encoding="utf-8"
    )
    return session_dir


def _create_session_with_md(root, date_str, time_str, lines):
    """테스트용 세션 디렉토리 + meeting.md 파일을 생성하는 헬퍼."""
    session_dir = root / "meetings" / date_str / time_str
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "meeting.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return session_dir


# 공통 테스트 세그먼트
SAMPLE_SEGMENTS = [
    {"text": "안녕하세요 반갑습니다", "speaker": "화자A", "start": 0.0},
    {"text": "오늘 회의 시작하겠습니다", "speaker": "화자B", "start": 5.0},
    {"text": "첫 번째 안건은 예산입니다", "speaker": "화자A", "start": 10.0},
    {"text": "예산 관련 자료를 공유합니다", "speaker": "화자B", "start": 20.0},
    {"text": "다음 안건으로 넘어가겠습니다", "speaker": "화자C", "start": 30.0},
    {"text": "API 서버 배포 일정을 논의합시다", "speaker": "화자A", "start": 45.0},
    {"text": "테스트 코드 작성이 완료되었습니다", "speaker": "화자B", "start": 60.0},
]

SESSION_ID = "2026-03-14_100000"
DATE_STR = "2026-03-14"
TIME_STR = "100000"


# ---------------------------------------------------------------------------
# 1. 기본 텍스트 검색
# ---------------------------------------------------------------------------


class TestBasicSearch:
    """기본 텍스트(비정규식) 검색을 검증한다."""

    @pytest.mark.asyncio
    async def test_search_returns_matching_segments(self, tmp_path, monkeypatch, client):
        """검색어가 포함된 세그먼트만 반환한다."""
        _create_session_with_jsonl(tmp_path, DATE_STR, TIME_STR, SAMPLE_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search", params={"query": "예산"}
        )
        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 2
        assert all("예산" in m["text"] for m in data["matches"])

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, tmp_path, monkeypatch, client):
        """대소문자 구분 없이 검색한다."""
        _create_session_with_jsonl(tmp_path, DATE_STR, TIME_STR, [
            {"text": "API 서버 배포", "speaker": "A", "start": 0.0},
            {"text": "api 테스트 완료", "speaker": "B", "start": 5.0},
        ])
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search", params={"query": "api"}
        )
        data = response.json()

        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_search_returns_correct_fields(self, tmp_path, monkeypatch, client):
        """응답의 각 매치에 index, text, speaker, timestamp 필드가 포함된다."""
        _create_session_with_jsonl(tmp_path, DATE_STR, TIME_STR, SAMPLE_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search", params={"query": "안녕"}
        )
        data = response.json()

        assert data["total"] >= 1
        match = data["matches"][0]
        assert "index" in match
        assert "text" in match
        assert "speaker" in match
        assert "timestamp" in match
        assert match["index"] == 0
        assert match["speaker"] == "화자A"
        assert match["timestamp"] == 0.0


# ---------------------------------------------------------------------------
# 2. 정규식 검색
# ---------------------------------------------------------------------------


class TestRegexSearch:
    """정규식 모드 검색을 검증한다."""

    @pytest.mark.asyncio
    async def test_regex_search_matches_pattern(self, tmp_path, monkeypatch, client):
        """정규식 패턴으로 매칭되는 세그먼트를 반환한다."""
        _create_session_with_jsonl(tmp_path, DATE_STR, TIME_STR, SAMPLE_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        # "안건.*예산" 패턴: "첫 번째 안건은 예산입니다"만 매칭
        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={"query": "안건.*예산", "regex": True},
        )
        data = response.json()

        assert data["total"] == 1
        assert "안건" in data["matches"][0]["text"]
        assert "예산" in data["matches"][0]["text"]

    @pytest.mark.asyncio
    async def test_regex_invalid_pattern_returns_400(self, tmp_path, monkeypatch, client):
        """잘못된 정규식 패턴은 400 에러와 JSON 메시지를 반환한다."""
        _create_session_with_jsonl(tmp_path, DATE_STR, TIME_STR, SAMPLE_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={"query": "[invalid(", "regex": True},
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "정규식" in detail

    @pytest.mark.asyncio
    async def test_regex_case_insensitive(self, tmp_path, monkeypatch, client):
        """정규식 모드에서도 대소문자 구분 없이 검색한다."""
        _create_session_with_jsonl(tmp_path, DATE_STR, TIME_STR, [
            {"text": "API 서버", "speaker": "A", "start": 0.0},
            {"text": "api 클라이언트", "speaker": "B", "start": 5.0},
        ])
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={"query": "^api", "regex": True},
        )
        data = response.json()

        assert data["total"] == 2


# ---------------------------------------------------------------------------
# 3. 필터 조합 (AND 조건)
# ---------------------------------------------------------------------------


class TestFilterCombinations:
    """화자 + 시간 범위 + 텍스트 필터 AND 조건을 검증한다."""

    @pytest.mark.asyncio
    async def test_speaker_filter(self, tmp_path, monkeypatch, client):
        """화자 필터가 적용되어 해당 화자의 세그먼트만 반환된다."""
        _create_session_with_jsonl(tmp_path, DATE_STR, TIME_STR, SAMPLE_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={"query": "안", "speaker": "화자A"},
        )
        data = response.json()

        assert data["total"] >= 1
        assert all(m["speaker"] == "화자A" for m in data["matches"])

    @pytest.mark.asyncio
    async def test_time_range_filter(self, tmp_path, monkeypatch, client):
        """시간 범위 필터가 적용되어 범위 내 세그먼트만 반환된다."""
        _create_session_with_jsonl(tmp_path, DATE_STR, TIME_STR, SAMPLE_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        # 10초~30초 범위: start가 10, 20, 30인 세그먼트만 대상
        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={"query": "안건", "time_start": 5.0, "time_end": 25.0},
        )
        data = response.json()

        assert data["total"] >= 1
        for m in data["matches"]:
            assert 5.0 <= m["timestamp"] <= 25.0

    @pytest.mark.asyncio
    async def test_combined_speaker_and_time_filter(self, tmp_path, monkeypatch, client):
        """화자 + 시간 범위 + 텍스트 검색을 동시에 적용한다."""
        _create_session_with_jsonl(tmp_path, DATE_STR, TIME_STR, SAMPLE_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        # 화자A + 시간 0~15초 + "안" 검색
        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={
                "query": "안",
                "speaker": "화자A",
                "time_start": 0.0,
                "time_end": 15.0,
            },
        )
        data = response.json()

        assert data["total"] >= 1
        for m in data["matches"]:
            assert m["speaker"] == "화자A"
            assert 0.0 <= m["timestamp"] <= 15.0
            assert "안" in m["text"]

    @pytest.mark.asyncio
    async def test_speaker_filter_no_match(self, tmp_path, monkeypatch, client):
        """존재하지 않는 화자로 필터링하면 빈 결과를 반환한다."""
        _create_session_with_jsonl(tmp_path, DATE_STR, TIME_STR, SAMPLE_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={"query": "안녕", "speaker": "없는화자"},
        )
        data = response.json()

        assert data["total"] == 0
        assert data["matches"] == []


# ---------------------------------------------------------------------------
# 4. 빈 결과
# ---------------------------------------------------------------------------


class TestEmptyResults:
    """검색 결과가 없는 경우를 검증한다."""

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self, tmp_path, monkeypatch, client):
        """매칭되는 세그먼트가 없으면 빈 결과를 반환한다."""
        _create_session_with_jsonl(tmp_path, DATE_STR, TIME_STR, SAMPLE_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={"query": "존재하지않는검색어xyz"},
        )
        data = response.json()

        assert data["total"] == 0
        assert data["matches"] == []

    @pytest.mark.asyncio
    async def test_empty_session_returns_empty(self, tmp_path, monkeypatch, client):
        """전사 파일이 없는 세션에서 검색하면 빈 결과를 반환한다."""
        session_dir = tmp_path / "meetings" / DATE_STR / TIME_STR
        session_dir.mkdir(parents=True)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search", params={"query": "테스트"}
        )
        data = response.json()

        assert data["total"] == 0
        assert data["matches"] == []

    @pytest.mark.asyncio
    async def test_regex_no_match_returns_empty(self, tmp_path, monkeypatch, client):
        """정규식 검색에서 매칭이 없으면 빈 결과를 반환한다."""
        _create_session_with_jsonl(tmp_path, DATE_STR, TIME_STR, SAMPLE_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={"query": "^zzz\\d{10}$", "regex": True},
        )
        data = response.json()

        assert data["total"] == 0
        assert data["matches"] == []


# ---------------------------------------------------------------------------
# 5. 에러 처리
# ---------------------------------------------------------------------------


class TestSearchErrors:
    """검색 API 에러 처리를 검증한다."""

    @pytest.mark.asyncio
    async def test_missing_session_returns_404(self, tmp_path, monkeypatch, client):
        """존재하지 않는 세션은 404를 반환한다."""
        (tmp_path / "meetings").mkdir(parents=True)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search", params={"query": "test"}
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_session_id_returns_400(self, client):
        """잘못된 세션 ID 형식은 400을 반환한다."""
        response = await client.get(
            "/api/sessions/invalid-id/search", params={"query": "test"}
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_query_returns_422(self, tmp_path, monkeypatch, client):
        """query 파라미터가 없으면 422 (유효성 검증 오류)를 반환한다."""
        _create_session_with_jsonl(tmp_path, DATE_STR, TIME_STR, SAMPLE_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(f"/api/sessions/{SESSION_ID}/search")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_requires_api_key_when_set(self, tmp_path, monkeypatch, client):
        """API 키가 설정되면 인증 없이 접근 시 401을 반환한다."""
        _create_session_with_jsonl(tmp_path, DATE_STR, TIME_STR, SAMPLE_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)
        server._api_key = "test-secret"

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search", params={"query": "test"}
        )
        assert response.status_code == 401

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={"query": "test"},
            headers={"x-api-key": "test-secret"},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# 6. meeting.md 폴백
# ---------------------------------------------------------------------------


class TestMdFallback:
    """JSONL이 없을 때 meeting.md에서 파싱하여 검색하는 것을 검증한다."""

    @pytest.mark.asyncio
    async def test_search_falls_back_to_meeting_md(self, tmp_path, monkeypatch, client):
        """JSONL이 없으면 meeting.md를 파싱하여 검색한다."""
        _create_session_with_md(tmp_path, DATE_STR, TIME_STR, [
            "- [00:00:01] 화자A: 안녕하세요 여러분",
            "- [00:00:10] 화자B: 반갑습니다",
            "- [00:01:00] 화자A: 오늘 안건을 시작하겠습니다",
        ])
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search", params={"query": "안"}
        )
        data = response.json()

        assert data["total"] == 2
        # 첫 번째: "안녕하세요 여러분" (timestamp=1초)
        assert data["matches"][0]["speaker"] == "화자A"
        assert data["matches"][0]["timestamp"] == 1.0
        # 두 번째: "오늘 안건을 시작하겠습니다" (timestamp=60초)
        assert data["matches"][1]["timestamp"] == 60.0

    @pytest.mark.asyncio
    async def test_md_search_with_speaker_filter(self, tmp_path, monkeypatch, client):
        """meeting.md 폴백에서도 화자 필터가 동작한다."""
        _create_session_with_md(tmp_path, DATE_STR, TIME_STR, [
            "- [00:00:01] 화자A: 프로젝트 논의",
            "- [00:00:10] 화자B: 프로젝트 일정",
            "- [00:01:00] 화자A: 프로젝트 결론",
        ])
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={"query": "프로젝트", "speaker": "화자A"},
        )
        data = response.json()

        assert data["total"] == 2
        assert all(m["speaker"] == "화자A" for m in data["matches"])
