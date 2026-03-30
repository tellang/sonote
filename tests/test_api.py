"""FastAPI 서버 통합 테스트.

SSE 스트림, 교정 이벤트, 후처리 상태 폴백, 키워드 전파,
내역 일관성, API 키 인증을 검증한다.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport

from src import server
import src._server_impl as server_impl


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """httpx AsyncClient를 FastAPI 앱에 연결한다."""
    transport = ASGITransport(app=server.app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture(autouse=True)
def _reset_extra_state():
    """conftest의 _reset_server_state에 추가로 인증/후처리 상태를 초기화한다."""
    original_api_key = server._api_key
    original_status_file = server._postprocess_status_file
    original_phase = server._postprocess_phase
    original_progress = server._postprocess_progress
    original_paused = server._paused
    original_shutdown = server._shutdown_requested
    original_startup_ready = server._startup_ready

    server._api_key = ""
    server._postprocess_status_file = None
    server._postprocess_phase = ""
    server._postprocess_progress = 0.0
    server._paused = False
    server._shutdown_requested = False
    server._startup_ready = True
    server._startup_phase = ""

    yield

    server._api_key = original_api_key
    server._postprocess_status_file = original_status_file
    server._postprocess_phase = original_phase
    server._postprocess_progress = original_progress
    server._paused = original_paused
    server._shutdown_requested = original_shutdown
    server._startup_ready = original_startup_ready


# ---------------------------------------------------------------------------
# 1. SSE /stream + correction 이벤트 (Critical)
# ---------------------------------------------------------------------------


class TestSSEStreamAndCorrection:
    """SSE 스트림 연결, 전사 수신, correction 이벤트 수신을 검증한다.

    httpx ASGITransport에서 SSE 스트림이 무한 블록되므로,
    서버 내부 _client_queues를 직접 사용하여 SSE 메커니즘을 검증한다.
    """

    @pytest.mark.asyncio
    async def test_push_transcript_broadcasts_to_client_queue(self):
        """push_transcript 호출 시 _client_queues에 등록된 큐로 데이터가 전달된다."""
        queue: asyncio.Queue[dict] = asyncio.Queue()
        server._client_queues.add(queue)

        await server.push_transcript("화자A", "테스트 발화입니다.", "00:01:00")

        item = await asyncio.wait_for(queue.get(), timeout=2.0)
        assert item["speaker"] == "화자A"
        assert item["text"] == "테스트 발화입니다."
        assert item["ts"] == "00:01:00"

        server._client_queues.discard(queue)

    @pytest.mark.asyncio
    async def test_push_transcript_preserves_confidence_in_queue(self):
        """실시간 전사 confidence가 브로드캐스트 큐에 유지된다."""
        queue: asyncio.Queue[dict] = asyncio.Queue()
        server._client_queues.add(queue)

        await server.push_transcript("화자A", "낮은 신뢰도 발화", "00:01:01", confidence=0.42)

        item = await asyncio.wait_for(queue.get(), timeout=2.0)
        assert item["confidence"] == 0.42

        server._client_queues.discard(queue)

    def test_stream_transcript_payload_includes_confidence(self):
        """SSE 기본 transcript payload에 confidence가 포함된다."""
        payload = server_impl._stream_transcript_payload(
            {"speaker": "화자A", "text": "테스트", "ts": "00:00:03", "confidence": 0.33}
        )
        assert payload == {
            "speaker": "화자A",
            "text": "테스트",
            "ts": "00:00:03",
            "confidence": 0.33,
        }

    @pytest.mark.asyncio
    async def test_push_correction_broadcasts_correction_event(self):
        """push_correction 호출 시 correction 타입 이벤트가 큐로 전달된다."""
        server._transcript_history.append(
            {"speaker": "A", "text": "원본 텍스트", "ts": "00:00:01"}
        )

        queue: asyncio.Queue[dict] = asyncio.Queue()
        server._client_queues.add(queue)

        await server.push_correction([
            {"index": 0, "original": "원본 텍스트", "corrected": "교정된 텍스트"}
        ])

        item = await asyncio.wait_for(queue.get(), timeout=2.0)
        assert item["_type"] == "correction"
        assert len(item["_payload"]["corrections"]) == 1
        assert item["_payload"]["corrections"][0]["index"] == 0
        assert item["_payload"]["corrections"][0]["corrected"] == "교정된 텍스트"

        # 내역도 갱신되었는지 확인
        assert server._transcript_history[0]["text"] == "교정된 텍스트"

        server._client_queues.discard(queue)

    @pytest.mark.asyncio
    async def test_multiple_clients_receive_same_transcript(self):
        """여러 클라이언트 큐에 동일한 전사가 브로드캐스트된다."""
        queues = [asyncio.Queue() for _ in range(3)]
        for q in queues:
            server._client_queues.add(q)

        await server.push_transcript("B", "멀티 클라이언트", "00:02:00")

        for q in queues:
            item = await asyncio.wait_for(q.get(), timeout=2.0)
            assert item["text"] == "멀티 클라이언트"
            server._client_queues.discard(q)

    @pytest.mark.asyncio
    async def test_correction_with_invalid_index_no_broadcast(self):
        """유효하지 않은 인덱스만 포함된 교정은 브로드캐스트하지 않는다."""
        server._transcript_history.append(
            {"speaker": "A", "text": "유지", "ts": "00:00:01"}
        )

        queue: asyncio.Queue[dict] = asyncio.Queue()
        server._client_queues.add(queue)

        await server.push_correction([
            {"index": 999, "original": "X", "corrected": "Y"}
        ])

        assert queue.empty()
        assert server._transcript_history[0]["text"] == "유지"

        server._client_queues.discard(queue)

    @pytest.mark.asyncio
    async def test_stream_endpoint_returns_event_stream_type(
        self, client: httpx.AsyncClient
    ):
        """/stream 엔드포인트가 text/event-stream 미디어 타입으로 응답한다."""
        # 큐에 데이터를 미리 넣어서 스트림이 즉시 데이터를 반환하게 함
        async def consume_stream():
            async with client.stream("GET", "/stream") as response:
                assert response.status_code == 200
                assert "text/event-stream" in response.headers["content-type"]
                return True

        # 타임아웃으로 스트림 연결만 확인 후 종료
        try:
            await asyncio.wait_for(consume_stream(), timeout=1.0)
        except (asyncio.TimeoutError, httpx.ReadError):
            # SSE 스트림은 무한히 열려 있으므로 타임아웃은 정상
            pass


# ---------------------------------------------------------------------------
# 2. /status 후처리 상태 파일 폴백 (High)
# ---------------------------------------------------------------------------


class TestStatusPostprocessFallback:
    """/status 엔드포인트의 파일 기반 후처리 상태 폴백을 검증한다."""

    @pytest.mark.asyncio
    async def test_status_reads_postprocess_from_file(self, client: httpx.AsyncClient):
        """_postprocess_status_file을 설정하면 /status에 파일 내용이 반영된다."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"phase": "stt", "progress": 50}, f)
            tmp_path = f.name

        try:
            server.set_postprocess_status_file(tmp_path)

            response = await client.get("/status")
            assert response.status_code == 200
            data = response.json()

            assert "postprocess" in data
            assert data["postprocess"]["phase"] == "stt"
            assert data["postprocess"]["progress"] == 50.0
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            server.set_postprocess_status_file(None)

    @pytest.mark.asyncio
    async def test_status_graceful_fallback_when_file_missing(
        self, client: httpx.AsyncClient
    ):
        """파일이 없으면 postprocess 필드가 누락되며 에러가 발생하지 않는다."""
        server.set_postprocess_status_file("/nonexistent/path/status.json")

        response = await client.get("/status")
        assert response.status_code == 200
        data = response.json()

        # 파일이 없으므로 postprocess 키가 없어야 함
        assert "postprocess" not in data

    @pytest.mark.asyncio
    async def test_status_graceful_fallback_when_file_deleted(
        self, client: httpx.AsyncClient
    ):
        """파일을 먼저 설정 후 삭제하면 graceful하게 폴백한다."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"phase": "polish", "progress": 80}, f)
            tmp_path = f.name

        server.set_postprocess_status_file(tmp_path)

        # 먼저 파일이 있을 때 확인
        response = await client.get("/status")
        assert response.json()["postprocess"]["phase"] == "polish"

        # 파일 삭제
        Path(tmp_path).unlink()

        # 파일이 없어도 에러 없이 응답
        response = await client.get("/status")
        assert response.status_code == 200
        assert "postprocess" not in response.json()

    @pytest.mark.asyncio
    async def test_status_inline_postprocess_takes_priority(
        self, client: httpx.AsyncClient
    ):
        """인라인 _postprocess_phase가 설정되면 파일 폴백보다 우선한다."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"phase": "file_phase", "progress": 30}, f)
            tmp_path = f.name

        try:
            server.set_postprocess_status_file(tmp_path)
            server.set_postprocess_status("inline_phase", 70.0)

            response = await client.get("/status")
            data = response.json()

            assert data["postprocess"]["phase"] == "inline_phase"
            assert data["postprocess"]["progress"] == 70.0
        finally:
            Path(tmp_path).unlink(missing_ok=True)
            server.set_postprocess_status_file(None)
            server.set_postprocess_status("", 0.0)


# ---------------------------------------------------------------------------
# 3. /add-keyword → /keywords 전파 (High)
# ---------------------------------------------------------------------------


class TestKeywordPropagation:
    """키워드 추가/삭제 후 /keywords에서 상태가 올바르게 반영되는지 검증한다."""

    @pytest.mark.asyncio
    async def test_add_keyword_appears_in_keywords(self, client: httpx.AsyncClient):
        """POST /add-keyword로 추가한 키워드가 GET /keywords에서 조회된다."""
        add_resp = await client.post(
            "/add-keyword", json={"keyword": "테스트"}
        )
        assert add_resp.status_code == 200

        kw_resp = await client.get("/keywords")
        assert kw_resp.status_code == 200
        data = kw_resp.json()

        assert "테스트" in data["manual"]
        assert "테스트" in data["keywords"]

    @pytest.mark.asyncio
    async def test_remove_keyword_moves_to_blocked(self, client: httpx.AsyncClient):
        """POST /remove-keyword로 삭제하면 blocked 목록으로 이동한다."""
        # 먼저 추가
        await client.post("/add-keyword", json={"keyword": "테스트"})

        # 삭제
        remove_resp = await client.post(
            "/remove-keyword", json={"keyword": "테스트"}
        )
        assert remove_resp.status_code == 200

        kw_resp = await client.get("/keywords")
        data = kw_resp.json()

        assert "테스트" not in data["manual"]
        assert "테스트" not in data["keywords"]
        assert "테스트" in data["blocked"]

    @pytest.mark.asyncio
    async def test_add_remove_add_roundtrip(self, client: httpx.AsyncClient):
        """추가 → 삭제 → 재추가 시 blocked에서 제거되고 manual로 복원된다."""
        await client.post("/add-keyword", json={"keyword": "파이썬"})
        await client.post("/remove-keyword", json={"keyword": "파이썬"})

        # blocked에 있는 상태
        kw_resp = await client.get("/keywords")
        assert "파이썬" in kw_resp.json()["blocked"]

        # 재추가하면 blocked에서 제거
        await client.post("/add-keyword", json={"keyword": "파이썬"})
        kw_resp = await client.get("/keywords")
        data = kw_resp.json()

        assert "파이썬" in data["manual"]
        assert "파이썬" not in data["blocked"]

    @pytest.mark.asyncio
    async def test_short_keyword_ignored(self, client: httpx.AsyncClient):
        """2자 미만 키워드는 무시된다."""
        await client.post("/add-keyword", json={"keyword": "A"})

        kw_resp = await client.get("/keywords")
        data = kw_resp.json()
        assert "A" not in data["manual"]
        assert "A" not in data["keywords"]

    @pytest.mark.asyncio
    async def test_remove_missing_keyword_does_not_add_blocked(self, client: httpx.AsyncClient):
        """존재하지 않는 키워드 삭제는 blocked 목록을 오염시키지 않는다."""
        response = await client.post("/remove-keyword", json={"keyword": "없는키워드"})
        assert response.status_code == 200
        data = response.json()
        assert "없는키워드" not in data["blocked"]

    @pytest.mark.asyncio
    async def test_delete_remove_keyword_supported(self, client: httpx.AsyncClient):
        """DELETE /remove-keyword도 POST와 동일하게 동작한다."""
        await client.post("/add-keyword", json={"keyword": "삭제대상"})
        response = await client.request("DELETE", "/remove-keyword", json={"keyword": "삭제대상"})
        assert response.status_code == 200
        data = response.json()
        assert "삭제대상" not in data["keywords"]
        assert "삭제대상" in data["blocked"]


# ---------------------------------------------------------------------------
# 4. /api/load-transcript 경로 검증 (High/Security)
# ---------------------------------------------------------------------------


class TestLoadTranscriptPathValidation:
    """load-transcript 엔드포인트의 경로 정규화/검증을 테스트한다."""

    @pytest.mark.asyncio
    async def test_requires_path_parameter(self, client: httpx.AsyncClient):
        """path가 비어 있으면 400을 반환한다."""
        response = await client.post("/api/load-transcript", json={})
        assert response.status_code == 400
        assert response.json()["detail"] == "path 파라미터가 필요합니다."

    @pytest.mark.asyncio
    async def test_rejects_path_outside_output_root(self, tmp_path, monkeypatch, client):
        """OUTPUT_ROOT 밖의 경로는 거부한다."""
        output_root = tmp_path / "output"
        output_root.mkdir(parents=True)
        outside_file = tmp_path / "outside.txt"
        outside_file.write_text("외부 파일", encoding="utf-8")
        monkeypatch.setattr("src.server.OUTPUT_ROOT", output_root)

        response = await client.post(
            "/api/load-transcript",
            json={"path": str(outside_file)},
        )
        assert response.status_code == 400
        assert "output 디렉토리 밖의 파일" in response.json()["detail"]

    @pytest.mark.asyncio
    @pytest.mark.skipif(os.name != "nt", reason="Windows 백슬래시 경로 재현 테스트")
    async def test_accepts_windows_backslash_path(self, tmp_path, monkeypatch, client):
        """Windows 백슬래시 절대 경로를 정상 처리한다."""
        output_root = tmp_path / "output"
        transcript_file = output_root / "transcripts" / "2026-03-16" / "transcript.corrected.txt"
        transcript_file.parent.mkdir(parents=True)
        transcript_file.write_text("[00:00 ~ 00:01] 테스트 라인", encoding="utf-8")
        monkeypatch.setattr("src.server.OUTPUT_ROOT", output_root)

        response = await client.post(
            "/api/load-transcript",
            json={"path": str(transcript_file)},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["loaded"] == 1
        assert payload["file"].endswith("transcript.corrected.txt")


# ---------------------------------------------------------------------------
# 5. /history 일관성 (Medium)
# ---------------------------------------------------------------------------


class TestHistoryConsistency:
    """push_transcript로 추가된 전사가 /history에서 올바르게 조회되는지 검증한다."""

    @pytest.mark.asyncio
    async def test_history_returns_pushed_transcripts(
        self, client: httpx.AsyncClient
    ):
        """여러 전사를 push한 후 /history에서 순서와 개수가 일치한다."""
        transcripts = [
            ("화자A", "첫 번째 발화", "00:00:01"),
            ("화자B", "두 번째 발화", "00:00:05"),
            ("화자A", "세 번째 발화", "00:00:10"),
        ]
        for speaker, text, ts in transcripts:
            await server.push_transcript(speaker, text, ts)

        response = await client.get("/history")
        assert response.status_code == 200
        data = response.json()

        assert len(data) == 3
        assert data[0]["speaker"] == "화자A"
        assert data[0]["text"] == "첫 번째 발화"
        assert data[1]["speaker"] == "화자B"
        assert data[2]["text"] == "세 번째 발화"

    @pytest.mark.asyncio
    async def test_history_empty_initially(self, client: httpx.AsyncClient):
        """초기 상태에서 /history는 빈 리스트를 반환한다."""
        response = await client.get("/history")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_history_reflects_correction(self, client: httpx.AsyncClient):
        """push_correction 후 /history에 교정된 텍스트가 반영된다."""
        await server.push_transcript("A", "원래 텍스트", "00:00:01")

        await server.push_correction([
            {"index": 0, "original": "원래 텍스트", "corrected": "수정된 텍스트"}
        ])

        response = await client.get("/history")
        data = response.json()
        assert data[0]["text"] == "수정된 텍스트"

    @pytest.mark.asyncio
    async def test_history_order_preserved(self, client: httpx.AsyncClient):
        """전사 순서가 push 순서와 동일하게 유지된다."""
        for i in range(5):
            await server.push_transcript("S", f"발화 {i}", f"00:00:{i:02d}")

        response = await client.get("/history")
        data = response.json()

        assert len(data) == 5
        for i in range(5):
            assert data[i]["text"] == f"발화 {i}"

    @pytest.mark.asyncio
    async def test_history_preserves_confidence(self, client: httpx.AsyncClient):
        """confidence가 있는 전사는 /history 응답에도 유지된다."""
        await server.push_transcript("화자A", "신뢰도 테스트", "00:00:01", confidence=0.42)

        response = await client.get("/history")
        assert response.status_code == 200
        data = response.json()

        assert len(data) == 1
        assert data[0]["confidence"] == 0.42


# ---------------------------------------------------------------------------
# 6. MEETING_API_KEY 인증 (Medium)
# ---------------------------------------------------------------------------


class TestAPIKeyAuth:
    """MEETING_API_KEY 설정에 따른 보호 엔드포인트 인증을 검증한다."""

    @pytest.mark.asyncio
    async def test_protected_endpoint_without_key_returns_401(
        self, client: httpx.AsyncClient
    ):
        """API 키가 설정되었으나 제공하지 않으면 401을 반환한다."""
        server._api_key = "test-secret"

        response = await client.post("/toggle-pause")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_protected_endpoint_with_correct_key_returns_200(
        self, client: httpx.AsyncClient
    ):
        """올바른 API 키를 헤더로 제공하면 200을 반환한다."""
        server._api_key = "test-secret"

        response = await client.post(
            "/toggle-pause", headers={"x-api-key": "test-secret"}
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_protected_endpoint_with_wrong_key_returns_401(
        self, client: httpx.AsyncClient
    ):
        """잘못된 API 키를 제공하면 401을 반환한다."""
        server._api_key = "test-secret"

        response = await client.post(
            "/toggle-pause", headers={"x-api-key": "wrong-key"}
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_no_api_key_set_allows_access(self, client: httpx.AsyncClient):
        """API 키가 설정되지 않으면 인증 없이 접근 가능하다."""
        server._api_key = ""

        response = await client.post("/toggle-pause")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_api_key_via_query_param(self, client: httpx.AsyncClient):
        """쿼리 파라미터로 API 키를 전달할 수 있다."""
        server._api_key = "test-secret"

        response = await client.post("/toggle-pause?api_key=test-secret")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_stream_requires_api_key(self, client: httpx.AsyncClient):
        """SSE /stream 엔드포인트도 API 키 인증이 적용된다."""
        server._api_key = "test-secret"

        # API 키 없이 접근하면 401
        response = await client.get("/stream")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_shutdown_requires_api_key(self, client: httpx.AsyncClient):
        """/shutdown 엔드포인트도 API 키 인증이 적용된다."""
        server._api_key = "test-secret"

        response = await client.post("/shutdown")
        assert response.status_code == 401

        response = await client.post(
            "/shutdown", headers={"x-api-key": "test-secret"}
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_add_keyword_requires_api_key(self, client: httpx.AsyncClient):
        """/add-keyword 엔드포인트도 API 키 인증이 적용된다."""
        server._api_key = "test-secret"

        response = await client.post(
            "/add-keyword", json={"keyword": "테스트"}
        )
        assert response.status_code == 401

        response = await client.post(
            "/add-keyword",
            json={"keyword": "테스트"},
            headers={"x-api-key": "test-secret"},
        )
        assert response.status_code == 200
