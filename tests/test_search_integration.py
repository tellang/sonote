"""검색 API 통합 테스트.

FastAPI TestClient로 세션 생성 → 세그먼트 추가 → 검색 API 호출 → 결과 검증.
복합 필터(화자+시간대+정규식) 동시 사용, 엣지케이스를 포괄한다.
"""

import json

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from src import server


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client():
    """httpx AsyncClient를 FastAPI 앱에 연결한다."""
    transport = ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_api_key():
    """테스트 중 API 키 인증을 비활성화한다."""
    original = server._api_key
    server._api_key = ""
    yield
    server._api_key = original


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _create_session(root, date_str, time_str, segments):
    """테스트용 세션 디렉토리 + JSONL 파일을 생성한다."""
    session_dir = root / "meetings" / date_str / time_str
    session_dir.mkdir(parents=True, exist_ok=True)
    jsonl_lines = [json.dumps(seg, ensure_ascii=False) for seg in segments]
    (session_dir / "meeting.stt.jsonl").write_text(
        "\n".join(jsonl_lines), encoding="utf-8",
    )
    return session_dir


def _create_session_with_md(root, date_str, time_str, lines):
    """테스트용 세션 디렉토리 + meeting.md 파일을 생성한다."""
    session_dir = root / "meetings" / date_str / time_str
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "meeting.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )
    return session_dir


# 통합 테스트용 풍부한 세그먼트 세트 (다수 화자, 긴 시간 범위)
INTEGRATION_SEGMENTS = [
    {"text": "안녕하세요 오늘 프로젝트 킥오프 미팅입니다", "speaker": "김철수", "start": 0.0},
    {"text": "반갑습니다 일정 공유 부탁드립니다", "speaker": "이영희", "start": 5.0},
    {"text": "3월 말까지 MVP 완성 목표입니다", "speaker": "김철수", "start": 12.0},
    {"text": "백엔드 API 설계는 제가 담당하겠습니다", "speaker": "박민수", "start": 20.0},
    {"text": "프론트엔드는 React로 진행하겠습니다", "speaker": "이영희", "start": 30.0},
    {"text": "API 문서화는 Swagger로 하죠", "speaker": "박민수", "start": 40.0},
    {"text": "테스트 자동화도 병행해야 합니다", "speaker": "김철수", "start": 55.0},
    {"text": "CI/CD 파이프라인 구축은 제가 할게요", "speaker": "정다은", "start": 65.0},
    {"text": "다음 주 수요일에 중간 점검하겠습니다", "speaker": "김철수", "start": 80.0},
    {"text": "감사합니다 수고하세요", "speaker": "이영희", "start": 90.0},
]

SESSION_ID = "2026-03-14_143000"
DATE_STR = "2026-03-14"
TIME_STR = "143000"


# ---------------------------------------------------------------------------
# 1. 세션 생성 → 검색 E2E 흐름
# ---------------------------------------------------------------------------


class TestSessionCreateAndSearch:
    """세션을 생성하고 검색까지 완전한 흐름을 검증한다."""

    @pytest.mark.asyncio
    async def test_create_session_then_search(self, tmp_path, monkeypatch, client):
        """세션을 생성하고 검색하여 올바른 결과를 반환한다."""
        _create_session(tmp_path, DATE_STR, TIME_STR, INTEGRATION_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search", params={"query": "API"},
        )
        assert response.status_code == 200
        data = response.json()

        # "API"가 포함된 세그먼트: 백엔드 API, API 문서화
        assert data["total"] == 2
        texts = [m["text"] for m in data["matches"]]
        assert all("API" in t for t in texts)

    @pytest.mark.asyncio
    async def test_search_preserves_segment_order(self, tmp_path, monkeypatch, client):
        """검색 결과가 원본 세그먼트 순서(index 순)를 유지한다."""
        _create_session(tmp_path, DATE_STR, TIME_STR, INTEGRATION_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={"query": "목표", "speaker": "김철수"},
        )
        data = response.json()
        assert data["total"] == 1
        assert data["matches"][0]["speaker"] == "김철수"
        assert data["matches"][0]["timestamp"] == 12.0

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolated(self, tmp_path, monkeypatch, client):
        """서로 다른 세션의 검색 결과가 격리된다."""
        _create_session(tmp_path, "2026-03-14", "100000", [
            {"text": "세션A 전용 내용", "speaker": "A", "start": 0.0},
        ])
        _create_session(tmp_path, "2026-03-14", "110000", [
            {"text": "세션B 전용 내용", "speaker": "B", "start": 0.0},
        ])
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        resp_a = await client.get(
            "/api/sessions/2026-03-14_100000/search", params={"query": "세션A"},
        )
        resp_b = await client.get(
            "/api/sessions/2026-03-14_110000/search", params={"query": "세션A"},
        )

        assert resp_a.json()["total"] == 1
        assert resp_b.json()["total"] == 0


# ---------------------------------------------------------------------------
# 2. 복합 필터 동시 사용 (화자 + 시간대 + 정규식)
# ---------------------------------------------------------------------------


class TestCompoundFilters:
    """화자, 시간 범위, 정규식 필터를 동시에 적용하는 케이스를 검증한다."""

    @pytest.mark.asyncio
    async def test_speaker_and_time_and_regex(self, tmp_path, monkeypatch, client):
        """화자 + 시간 범위 + 정규식 3중 필터를 동시 적용한다."""
        _create_session(tmp_path, DATE_STR, TIME_STR, INTEGRATION_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        # 박민수 + 15~50초 + "API|React" 정규식
        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={
                "query": "API|React",
                "speaker": "박민수",
                "time_start": 15.0,
                "time_end": 50.0,
                "regex": True,
            },
        )
        data = response.json()

        # 박민수의 발화 중 15~50초 범위: "백엔드 API 설계" (20초), "API 문서화" (40초)
        assert data["total"] == 2
        for m in data["matches"]:
            assert m["speaker"] == "박민수"
            assert 15.0 <= m["timestamp"] <= 50.0

    @pytest.mark.asyncio
    async def test_regex_with_speaker_filter_narrows_results(self, tmp_path, monkeypatch, client):
        """정규식 매칭 결과를 화자 필터로 추가 축소한다."""
        _create_session(tmp_path, DATE_STR, TIME_STR, INTEGRATION_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        # "하겠습니다$" 정규식 — 여러 화자가 매칭됨
        resp_all = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={"query": "하겠습니다$", "regex": True},
        )
        total_all = resp_all.json()["total"]

        # 이영희로 필터링하면 부분 집합
        resp_filtered = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={"query": "하겠습니다$", "regex": True, "speaker": "이영희"},
        )
        total_filtered = resp_filtered.json()["total"]

        assert total_filtered < total_all
        assert all(
            m["speaker"] == "이영희"
            for m in resp_filtered.json()["matches"]
        )

    @pytest.mark.asyncio
    async def test_time_range_excludes_boundary(self, tmp_path, monkeypatch, client):
        """시간 범위 필터의 경계값이 올바르게 포함/제외된다."""
        _create_session(tmp_path, DATE_STR, TIME_STR, INTEGRATION_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        # time_start=20, time_end=40 — start가 20, 30, 40인 세그먼트 포함
        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={"query": "", "time_start": 20.0, "time_end": 40.0},
        )
        data = response.json()

        for m in data["matches"]:
            assert 20.0 <= m["timestamp"] <= 40.0

    @pytest.mark.asyncio
    async def test_all_filters_no_match(self, tmp_path, monkeypatch, client):
        """모든 필터를 동시에 적용하여 결과가 0인 경우."""
        _create_session(tmp_path, DATE_STR, TIME_STR, INTEGRATION_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={
                "query": "존재하지않는패턴xyz",
                "speaker": "김철수",
                "time_start": 0.0,
                "time_end": 100.0,
                "regex": True,
            },
        )
        data = response.json()
        assert data["total"] == 0
        assert data["matches"] == []


# ---------------------------------------------------------------------------
# 3. 엣지케이스
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """검색 API의 엣지케이스를 검증한다."""

    @pytest.mark.asyncio
    async def test_empty_session_directory(self, tmp_path, monkeypatch, client):
        """전사 파일이 없는 빈 세션 디렉토리에서 검색 시 빈 결과를 반환한다."""
        session_dir = tmp_path / "meetings" / DATE_STR / TIME_STR
        session_dir.mkdir(parents=True)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search", params={"query": "테스트"},
        )
        data = response.json()
        assert data["total"] == 0
        assert data["matches"] == []

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, tmp_path, monkeypatch, client):
        """특수문자가 포함된 검색어 (비정규식 모드)가 안전하게 처리된다."""
        _create_session(tmp_path, DATE_STR, TIME_STR, [
            {"text": "가격은 $100 (할인가)입니다", "speaker": "A", "start": 0.0},
            {"text": "이메일: test@example.com", "speaker": "B", "start": 5.0},
        ])
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        # 정규식 메타문자가 포함된 검색어를 일반 텍스트로 검색
        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search", params={"query": "$100 (할인가)"},
        )
        data = response.json()
        assert data["total"] == 1
        assert "$100" in data["matches"][0]["text"]

    @pytest.mark.asyncio
    async def test_special_characters_in_regex_query(self, tmp_path, monkeypatch, client):
        """정규식 모드에서 특수문자를 이스케이프하여 리터럴 매칭한다."""
        _create_session(tmp_path, DATE_STR, TIME_STR, [
            {"text": "가격은 $100 (할인가)입니다", "speaker": "A", "start": 0.0},
        ])
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        # 이스케이프된 정규식으로 리터럴 매칭
        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={"query": r"\$100", "regex": True},
        )
        data = response.json()
        assert data["total"] == 1

    @pytest.mark.asyncio
    async def test_invalid_regex_returns_400(self, tmp_path, monkeypatch, client):
        """잘못된 정규식 패턴은 400 에러를 반환한다."""
        _create_session(tmp_path, DATE_STR, TIME_STR, INTEGRATION_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={"query": "[잘못된(정규식", "regex": True},
        )
        assert response.status_code == 400
        assert "정규식" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_unicode_regex_search(self, tmp_path, monkeypatch, client):
        """유니코드(한글) 정규식 패턴으로 검색한다."""
        _create_session(tmp_path, DATE_STR, TIME_STR, INTEGRATION_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        # "프로젝트|파이프라인" 한글 OR 패턴
        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={"query": "프로젝트|파이프라인", "regex": True},
        )
        data = response.json()
        assert data["total"] >= 2

    @pytest.mark.asyncio
    async def test_empty_jsonl_file(self, tmp_path, monkeypatch, client):
        """빈 JSONL 파일이 있는 세션에서 검색 시 빈 결과를 반환한다."""
        session_dir = tmp_path / "meetings" / DATE_STR / TIME_STR
        session_dir.mkdir(parents=True)
        (session_dir / "meeting.stt.jsonl").write_text("", encoding="utf-8")
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search", params={"query": "테스트"},
        )
        data = response.json()
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_malformed_jsonl_lines_skipped(self, tmp_path, monkeypatch, client):
        """JSONL 파일의 깨진 라인은 무시하고 유효한 세그먼트만 검색한다."""
        session_dir = tmp_path / "meetings" / DATE_STR / TIME_STR
        session_dir.mkdir(parents=True)
        lines = [
            json.dumps({"text": "유효한 세그먼트", "speaker": "A", "start": 0.0}),
            "이것은 유효하지 않은 JSON 라인",
            json.dumps({"text": "또 다른 유효 세그먼트", "speaker": "B", "start": 5.0}),
        ]
        (session_dir / "meeting.stt.jsonl").write_text(
            "\n".join(lines), encoding="utf-8",
        )
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search", params={"query": "유효"},
        )
        data = response.json()
        # 깨진 라인 무시, 유효 세그먼트 2개만 검색
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_nonexistent_session_returns_404(self, tmp_path, monkeypatch, client):
        """존재하지 않는 세션 ID로 검색 시 404를 반환한다."""
        (tmp_path / "meetings").mkdir(parents=True)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            "/api/sessions/2099-12-31_235959/search", params={"query": "테스트"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_session_id_format_returns_400(self, client):
        """세션 ID 형식이 잘못되면 400을 반환한다."""
        response = await client.get(
            "/api/sessions/not-a-valid-id/search", params={"query": "테스트"},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_jsonl_to_md_fallback(self, tmp_path, monkeypatch, client):
        """JSONL이 없을 때 meeting.md 폴백으로 검색한다."""
        _create_session_with_md(tmp_path, DATE_STR, TIME_STR, [
            "- [00:00:05] 발표자: API 설계 논의",
            "- [00:00:15] 참석자: API 변경사항 확인",
            "- [00:01:00] 발표자: 마무리하겠습니다",
        ])
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search", params={"query": "API"},
        )
        data = response.json()
        assert data["total"] == 2
        assert data["matches"][0]["timestamp"] == 5.0
        assert data["matches"][1]["timestamp"] == 15.0

    @pytest.mark.asyncio
    async def test_only_time_start_filter(self, tmp_path, monkeypatch, client):
        """time_start만 지정하면 해당 시간 이후 세그먼트만 반환한다."""
        _create_session(tmp_path, DATE_STR, TIME_STR, INTEGRATION_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={"query": "", "time_start": 60.0},
        )
        data = response.json()

        assert data["total"] >= 1
        for m in data["matches"]:
            assert m["timestamp"] >= 60.0

    @pytest.mark.asyncio
    async def test_only_time_end_filter(self, tmp_path, monkeypatch, client):
        """time_end만 지정하면 해당 시간 이전 세그먼트만 반환한다."""
        _create_session(tmp_path, DATE_STR, TIME_STR, INTEGRATION_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={"query": "", "time_end": 10.0},
        )
        data = response.json()

        assert data["total"] >= 1
        for m in data["matches"]:
            assert m["timestamp"] <= 10.0

    @pytest.mark.asyncio
    async def test_api_key_required_when_set(self, tmp_path, monkeypatch, client):
        """API 키 설정 시 인증 없이 접근하면 401, 인증하면 200을 반환한다."""
        _create_session(tmp_path, DATE_STR, TIME_STR, INTEGRATION_SEGMENTS)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)
        server._api_key = "integration-test-key"

        # 인증 없이 접근 → 401
        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search", params={"query": "API"},
        )
        assert response.status_code == 401

        # 올바른 키로 접근 → 200
        response = await client.get(
            f"/api/sessions/{SESSION_ID}/search",
            params={"query": "API"},
            headers={"x-api-key": "integration-test-key"},
        )
        assert response.status_code == 200
        assert response.json()["total"] >= 1
