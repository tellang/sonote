"""세션 관리 API + 세션 회전 유틸리티 테스트.

GET /api/sessions, GET /api/sessions/{id}, POST /api/sessions/new 엔드포인트와
consume_session_rotate, set_session_rotate_callback, is_session_rotate_requested
유틸리티 함수를 검증한다.
"""

import asyncio
import json
from types import SimpleNamespace

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


def _create_session_dir(root, date_str, time_str, *, meeting_lines=None, session_meta=None):
    """테스트용 세션 디렉토리 구조를 생성하는 헬퍼.

    Args:
        root: OUTPUT_ROOT에 해당하는 tmp_path.
        date_str: 날짜 문자열 (YYYY-MM-DD).
        time_str: 시간 문자열 (HHMMSS).
        meeting_lines: meeting.md에 쓸 줄 리스트. None이면 파일 미생성.
        session_meta: session.json에 쓸 dict. None이면 파일 미생성.

    Returns:
        생성된 세션 디렉토리 Path.
    """
    session_dir = root / "meetings" / date_str / time_str
    session_dir.mkdir(parents=True, exist_ok=True)

    if meeting_lines is not None:
        (session_dir / "meeting.md").write_text(
            "\n".join(meeting_lines), encoding="utf-8"
        )

    if session_meta is not None:
        (session_dir / "session.json").write_text(
            json.dumps(session_meta, ensure_ascii=False), encoding="utf-8"
        )

    return session_dir


# ---------------------------------------------------------------------------
# 1. GET /api/sessions — 세션 목록
# ---------------------------------------------------------------------------


class TestListSessions:
    """GET /api/sessions 테스트."""

    @pytest.mark.asyncio
    async def test_empty_when_no_meetings_dir(self, tmp_path, monkeypatch, client):
        """meetings 디렉토리가 없으면 빈 목록을 반환한다."""
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get("/api/sessions")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_empty_when_meetings_dir_exists_but_empty(self, tmp_path, monkeypatch, client):
        """meetings 디렉토리가 비어 있으면 빈 목록을 반환한다."""
        (tmp_path / "meetings").mkdir()
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get("/api/sessions")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_returns_sessions_from_directory_structure(self, tmp_path, monkeypatch, client):
        """디렉토리 구조에서 세션 목록을 정확히 스캔한다."""
        _create_session_dir(
            tmp_path, "2026-03-13", "143022",
            meeting_lines=[
                "- [00:00:01] 화자A: 안녕하세요.",
                "- [00:00:05] 화자B: 네, 반갑습니다.",
                "# 이것은 일반 줄",
            ],
        )
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()

        assert len(data) == 1
        session = data[0]
        assert session["id"] == "2026-03-13_143022"
        assert session["date"] == "2026-03-13"
        assert session["time"] == "143022"
        assert session["segments"] == 2  # "- [" 로 시작하는 줄만 카운트

    @pytest.mark.asyncio
    async def test_uses_session_json_when_available(self, tmp_path, monkeypatch, client):
        """session.json이 있으면 해당 메타데이터를 사용한다."""
        _create_session_dir(
            tmp_path, "2026-03-13", "100000",
            session_meta={
                "duration": "01:30:00",
                "segment_count": 42,
                "speaker_count": 3,
            },
        )
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get("/api/sessions")
        data = response.json()

        assert len(data) == 1
        session = data[0]
        assert session["duration"] == "01:30:00"
        assert session["segments"] == 42
        assert session["speakers"] == 3

    @pytest.mark.asyncio
    async def test_sorted_newest_first(self, tmp_path, monkeypatch, client):
        """최신 세션이 먼저 반환된다."""
        _create_session_dir(tmp_path, "2026-03-10", "090000", meeting_lines=[])
        _create_session_dir(tmp_path, "2026-03-13", "143022", meeting_lines=[])
        _create_session_dir(tmp_path, "2026-03-12", "120000", meeting_lines=[])
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get("/api/sessions")
        data = response.json()

        assert len(data) == 3
        assert data[0]["id"] == "2026-03-13_143022"
        assert data[1]["id"] == "2026-03-12_120000"
        assert data[2]["id"] == "2026-03-10_090000"

    @pytest.mark.asyncio
    async def test_ignores_non_directory_files(self, tmp_path, monkeypatch, client):
        """meetings 아래 파일은 무시하고 디렉토리만 스캔한다."""
        (tmp_path / "meetings").mkdir(parents=True)
        (tmp_path / "meetings" / "readme.txt").write_text("ignore me")
        _create_session_dir(tmp_path, "2026-03-13", "100000", meeting_lines=[])
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get("/api/sessions")
        data = response.json()

        assert len(data) == 1
        assert data[0]["id"] == "2026-03-13_100000"

    @pytest.mark.asyncio
    async def test_fallback_to_meeting_md_when_no_session_json(self, tmp_path, monkeypatch, client):
        """session.json이 없으면 meeting.md에서 세그먼트 수를 추출한다."""
        _create_session_dir(
            tmp_path, "2026-03-13", "110000",
            meeting_lines=[
                "- [00:00:01] A: 첫 줄",
                "- [00:00:05] B: 둘째 줄",
                "- [00:00:10] A: 셋째 줄",
            ],
        )
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get("/api/sessions")
        data = response.json()

        assert data[0]["segments"] == 3
        assert data[0]["duration"] == ""
        assert data[0]["speakers"] == 0

    @pytest.mark.asyncio
    async def test_malformed_session_json_falls_back(self, tmp_path, monkeypatch, client):
        """session.json이 손상되면 디렉토리 기반 폴백으로 동작한다."""
        session_dir = tmp_path / "meetings" / "2026-03-13" / "120000"
        session_dir.mkdir(parents=True)
        (session_dir / "session.json").write_text("not valid json", encoding="utf-8")
        (session_dir / "meeting.md").write_text(
            "- [00:00:01] A: 테스트\n", encoding="utf-8"
        )
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get("/api/sessions")
        data = response.json()

        assert len(data) == 1
        assert data[0]["segments"] == 1


# ---------------------------------------------------------------------------
# 2. GET /api/sessions/{session_id} — 세션 상세
# ---------------------------------------------------------------------------


class TestGetSession:
    """GET /api/sessions/{session_id} 테스트."""

    @pytest.mark.asyncio
    async def test_returns_404_for_missing_session(self, tmp_path, monkeypatch, client):
        """없는 세션은 404를 반환한다."""
        (tmp_path / "meetings").mkdir(parents=True)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get("/api/sessions/2026-03-13_999999")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_400_for_invalid_id_format(self, client):
        """잘못된 ID 형식은 400을 반환한다."""
        response = await client.get("/api/sessions/invalid-no-underscore")
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_returns_transcript_lines(self, tmp_path, monkeypatch, client):
        """전사 내용을 줄 단위로 반환한다."""
        lines = [
            "- [00:00:01] 화자A: 안녕하세요.",
            "- [00:00:05] 화자B: 반갑습니다.",
        ]
        _create_session_dir(tmp_path, "2026-03-13", "143022", meeting_lines=lines)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get("/api/sessions/2026-03-13_143022")
        assert response.status_code == 200
        data = response.json()

        assert data["id"] == "2026-03-13_143022"
        assert data["date"] == "2026-03-13"
        assert data["time"] == "143022"
        assert data["transcript"] == lines
        assert data["transcript_source"] == "meeting.md"

    @pytest.mark.asyncio
    async def test_returns_session_meta_from_json(self, tmp_path, monkeypatch, client):
        """session.json 메타데이터가 응답에 포함된다."""
        meta = {"duration": "00:45:00", "segment_count": 10, "speaker_count": 2}
        _create_session_dir(
            tmp_path, "2026-03-13", "150000",
            session_meta=meta,
            meeting_lines=["- [00:00:01] A: test"],
        )
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get("/api/sessions/2026-03-13_150000")
        data = response.json()

        assert data["duration"] == "00:45:00"
        assert data["segments"] == 10
        assert data["speakers"] == 2
        assert data["meta"] == meta

    @pytest.mark.asyncio
    async def test_empty_transcript_when_no_file(self, tmp_path, monkeypatch, client):
        """전사 파일이 없으면 빈 리스트를 반환한다."""
        _create_session_dir(tmp_path, "2026-03-13", "160000")
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get("/api/sessions/2026-03-13_160000")
        data = response.json()

        assert data["transcript"] == []
        assert data["transcript_source"] == ""

    @pytest.mark.asyncio
    async def test_loads_alignment_jsonl(self, tmp_path, monkeypatch, client):
        """meeting.stt.jsonl이 있으면 alignment 데이터를 로드한다."""
        session_dir = _create_session_dir(
            tmp_path, "2026-03-13", "170000",
            meeting_lines=["- [00:00:01] A: 테스트"],
        )
        alignment_data = [
            {"start": 0.0, "end": 1.0, "text": "테스트"},
            {"start": 1.5, "end": 3.0, "text": "두 번째"},
        ]
        jsonl_content = "\n".join(json.dumps(d, ensure_ascii=False) for d in alignment_data)
        (session_dir / "meeting.stt.jsonl").write_text(jsonl_content, encoding="utf-8")
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.get("/api/sessions/2026-03-13_170000")
        data = response.json()

        assert len(data["alignment"]) == 2
        assert data["alignment"][0]["text"] == "테스트"
        assert data["alignment"][1]["start"] == 1.5


# ---------------------------------------------------------------------------
# 3. POST /api/sessions/new — 새 세션 시작
# ---------------------------------------------------------------------------


class TestNewSession:
    """POST /api/sessions/new 테스트."""

    @pytest.mark.asyncio
    async def test_resets_server_state(self, client):
        """서버 상태(전사 내역, 세그먼트 수, 화자)가 리셋된다."""
        # 상태를 먼저 오염시킨다
        server._transcript_history.extend([
            {"speaker": "A", "text": "테스트", "ts": "00:00:01"},
        ])
        server._segment_count = 5
        server._speakers.add("A")
        server._speakers.add("B")

        response = await client.post("/api/sessions/new")
        assert response.status_code == 200

        assert len(server._transcript_history) == 0
        assert server._segment_count == 0
        assert len(server._speakers) == 0

    @pytest.mark.asyncio
    async def test_sets_rotate_flag(self, client):
        """세션 회전 플래그가 설정된다."""
        assert not server._session_rotate_event.is_set()

        response = await client.post("/api/sessions/new")
        assert response.status_code == 200

        assert server._session_rotate_event.is_set()

    @pytest.mark.asyncio
    async def test_does_not_call_callback_directly(self, client):
        """서버는 콜백을 직접 호출하지 않는다 (CLI 스레드에서 처리)."""
        called = []
        server.set_session_rotate_callback(lambda: called.append(True))

        response = await client.post("/api/sessions/new")
        assert response.status_code == 200

        # 서버는 플래그만 설정, 콜백은 CLI 스레드가 consume 시 호출
        assert len(called) == 0
        assert server._session_rotate_event.is_set()

    @pytest.mark.asyncio
    async def test_returns_new_session_id(self, client):
        """응답에 새 세션 ID가 포함된다."""
        response = await client.post("/api/sessions/new")
        data = response.json()

        assert "session_id" in data
        # 형식: YYYY-MM-DD_HHMMSS
        assert "_" in data["session_id"]
        parts = data["session_id"].split("_", 1)
        assert len(parts) == 2
        assert len(parts[0]) == 10  # YYYY-MM-DD
        assert len(parts[1]) == 6   # HHMMSS

    @pytest.mark.asyncio
    async def test_session_new_with_bad_callback_still_succeeds(self, client):
        """콜백 등록 여부와 관계없이 세션 생성이 완료된다."""
        def bad_callback():
            raise RuntimeError("콜백 오류")

        server.set_session_rotate_callback(bad_callback)

        response = await client.post("/api/sessions/new")
        assert response.status_code == 200
        # 서버는 콜백을 호출하지 않으므로 예외 발생 없음
        assert server._session_rotate_event.is_set()
        assert server._segment_count == 0

    @pytest.mark.asyncio
    async def test_requires_api_key_when_set(self, client):
        """API 키가 설정되면 인증 없이 접근 시 401을 반환한다."""
        server._api_key = "test-secret"

        response = await client.post("/api/sessions/new")
        assert response.status_code == 401

        response = await client.post(
            "/api/sessions/new",
            headers={"x-api-key": "test-secret"},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# 4. DELETE /api/sessions/{session_id} — 세션 삭제
# ---------------------------------------------------------------------------


class TestDeleteSession:
    """DELETE /api/sessions/{session_id} 테스트."""

    @pytest.mark.asyncio
    async def test_deletes_session_directory_and_broadcasts_event(
        self, tmp_path, monkeypatch, client
    ):
        """세션 삭제 후 디렉토리가 제거되고 session_deleted 이벤트가 전송된다."""
        _create_session_dir(tmp_path, "2026-03-13", "143022", meeting_lines=["- [00:00:01] A: test"])
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        queue: asyncio.Queue[dict] = asyncio.Queue()
        server._client_queues.add(queue)

        response = await client.delete("/api/sessions/2026-03-13_143022")

        assert response.status_code == 200
        assert response.json() == {"deleted": True, "session_id": "2026-03-13_143022"}
        assert not (tmp_path / "meetings" / "2026-03-13" / "143022").exists()

        item = await asyncio.wait_for(queue.get(), timeout=2.0)
        assert item["_type"] == "session_deleted"
        assert item["_payload"]["session_id"] == "2026-03-13_143022"

        server._client_queues.discard(queue)

    @pytest.mark.asyncio
    async def test_returns_404_for_missing_session(self, tmp_path, monkeypatch, client):
        """없는 세션을 삭제하면 404를 반환한다."""
        (tmp_path / "meetings").mkdir(parents=True)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        response = await client.delete("/api/sessions/2026-03-13_999999")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_400_for_invalid_id_format(self, client):
        """잘못된 세션 ID는 400을 반환한다."""
        response = await client.delete("/api/sessions/invalid-session")
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_requires_api_key_when_set(self, tmp_path, monkeypatch, client):
        """API 키가 설정되면 인증 없이 삭제할 수 없다."""
        _create_session_dir(tmp_path, "2026-03-13", "143022")
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)
        server._api_key = "test-secret"

        response = await client.delete("/api/sessions/2026-03-13_143022")
        assert response.status_code == 401

        response = await client.delete(
            "/api/sessions/2026-03-13_143022",
            headers={"x-api-key": "test-secret"},
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# 5. watchdog 세션 변경 감지
# ---------------------------------------------------------------------------


class TestSessionWatchHandler:
    """watchdog 이벤트 핸들러의 세션 이벤트 브로드캐스트를 검증한다."""

    def test_broadcasts_session_created_for_new_session_directory(self, tmp_path, monkeypatch):
        """YYYY-MM-DD/HHMMSS 디렉토리 생성 시 session_created를 전송한다."""
        meetings_root = tmp_path / "meetings"
        session_dir = meetings_root / "2026-03-13" / "143022"
        session_dir.mkdir(parents=True)
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        calls: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            server,
            "_broadcast_sse_sync",
            lambda event_type, payload: calls.append((event_type, payload)),
        )

        handler = server._SessionWatchHandler()
        handler.on_created(SimpleNamespace(src_path=str(session_dir), is_directory=True))

        assert calls == [
            (
                "session_created",
                {
                    "session_id": "2026-03-13_143022",
                    "date": "2026-03-13",
                    "time": "143022",
                    "path": str(session_dir.resolve()),
                },
            )
        ]

    def test_broadcasts_session_updated_for_file_change(self, tmp_path, monkeypatch):
        """세션 내부 파일 변경 시 session_updated를 전송한다."""
        session_dir = _create_session_dir(tmp_path, "2026-03-13", "143022")
        meeting_file = session_dir / "meeting.md"
        meeting_file.write_text("updated", encoding="utf-8")
        monkeypatch.setattr("src.server.OUTPUT_ROOT", tmp_path)

        calls: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            server,
            "_broadcast_sse_sync",
            lambda event_type, payload: calls.append((event_type, payload)),
        )

        handler = server._SessionWatchHandler()
        handler.on_modified(SimpleNamespace(src_path=str(meeting_file), is_directory=False))

        assert calls == [
            (
                "session_updated",
                {
                    "session_id": "2026-03-13_143022",
                    "date": "2026-03-13",
                    "time": "143022",
                    "path": str(session_dir.resolve()),
                    "file": str(meeting_file.resolve()),
                },
            )
        ]


# ---------------------------------------------------------------------------
# 6. 세션 회전 유틸리티 함수
# ---------------------------------------------------------------------------


class TestSessionRotateUtils:
    """세션 회전 유틸리티 함수 테스트."""

    def test_consume_clears_flag(self):
        """consume 후 플래그가 False로 전환된다."""
        server._session_rotate_event.set()

        result = server.consume_session_rotate()
        assert result is True
        assert not server._session_rotate_event.is_set()

    def test_consume_returns_false_when_not_requested(self):
        """요청이 없으면 False를 반환한다."""
        assert not server._session_rotate_event.is_set()

        result = server.consume_session_rotate()
        assert result is False
        assert not server._session_rotate_event.is_set()

    def test_consume_is_one_shot(self):
        """consume은 1회성이며 두 번째 호출은 False를 반환한다."""
        server._session_rotate_event.set()

        assert server.consume_session_rotate() is True
        assert server.consume_session_rotate() is False

    def test_callback_registration(self):
        """콜백 등록 및 호출이 정상 동작한다."""
        called = []
        server.set_session_rotate_callback(lambda: called.append("invoked"))

        assert server._session_rotate_callback is not None
        server._session_rotate_callback()
        assert called == ["invoked"]

    def test_is_session_rotate_requested_reflects_state(self):
        """is_session_rotate_requested가 현재 플래그 상태를 정확히 반환한다."""
        assert server.is_session_rotate_requested() is False

        server._session_rotate_event.set()
        assert server.is_session_rotate_requested() is True

        server._session_rotate_event.clear()
        assert server.is_session_rotate_requested() is False

    def test_callback_none_by_default(self):
        """초기 상태에서 콜백은 None이다."""
        # conftest autouse fixture가 매 테스트 전에 None으로 초기화
        assert server._session_rotate_callback is None
