"""화자 자동 등록(UnknownSpeakerTracker) 단위 테스트 + API 통합 테스트."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import server
from src.server import UnknownSpeakerTracker


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _make_embedding(dim: int = 192, seed: int = 0) -> list[float]:
    """재현 가능한 랜덤 임베딩 벡터 생성."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)  # 단위 벡터 정규화
    return vec.tolist()


def _make_orthogonal_embedding(dim: int = 192, seed: int = 42) -> list[float]:
    """기존 임베딩과 직교(유사도 ~0)인 벡터 생성."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """코사인 유사도 계산."""
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return dot / norm if norm > 0 else 0.0


def _make_fake_diarizer() -> SimpleNamespace:
    """_diarizer 역할을 하는 가짜 객체 생성."""
    fake = SimpleNamespace(
        speaker_embeddings={},
        _speaker_counts={},
        _enrolled_names=set(),
        _profile_mode=False,
        _cosine_similarity=staticmethod(
            lambda a, b: _cosine_similarity(
                np.asarray(a, dtype=np.float32),
                np.asarray(b, dtype=np.float32),
            )
        ),
    )
    return fake


# ---------------------------------------------------------------------------
# 1. UnknownSpeakerTracker 단위 테스트
# ---------------------------------------------------------------------------

class TestUnknownSpeakerTrackerUnit:
    """UnknownSpeakerTracker 클래스 독립 단위 테스트."""

    def setup_method(self) -> None:
        """각 테스트 전 tracker 초기화."""
        self.tracker = UnknownSpeakerTracker()
        self.fake_diarizer = _make_fake_diarizer()

    # -- track(): 기존 프로필과 유사도 < 0.7 → 미등록 분류 --

    def test_track_returns_unknown_id_when_no_profile_match(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """프로필에 매칭되는 화자가 없으면 unknown_N 형태의 ID를 반환한다."""
        monkeypatch.setattr(server, "_diarizer", self.fake_diarizer)
        monkeypatch.setattr(server, "_find_matching_profile", lambda *a, **kw: (None, 0.0))

        emb = _make_embedding(seed=1)
        result = self.tracker.track(emb)

        assert result is not None
        assert result.startswith("unknown_")

    def test_track_returns_none_when_profile_matched(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """기존 프로필과 매칭되면 None을 반환한다."""
        monkeypatch.setattr(server, "_diarizer", self.fake_diarizer)
        monkeypatch.setattr(server, "_find_matching_profile", lambda *a, **kw: ("Alice", 0.85))

        emb = _make_embedding(seed=1)
        result = self.tracker.track(emb)

        assert result is None

    def test_track_returns_none_when_diarizer_is_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_diarizer가 None이면 None을 반환한다."""
        monkeypatch.setattr(server, "_diarizer", None)

        emb = _make_embedding(seed=1)
        result = self.tracker.track(emb)

        assert result is None

    def test_track_returns_none_for_empty_embedding(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """빈 임베딩이면 None을 반환한다."""
        monkeypatch.setattr(server, "_diarizer", self.fake_diarizer)

        result = self.tracker.track([])
        assert result is None

    # -- track(): 같은 화자 세그먼트 임베딩 평균 누적 --

    def test_track_accumulates_embeddings_for_same_unknown_speaker(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """같은 미등록 화자의 임베딩이 누적되고 평균이 갱신된다."""
        monkeypatch.setattr(server, "_diarizer", self.fake_diarizer)
        monkeypatch.setattr(server, "_find_matching_profile", lambda *a, **kw: (None, 0.0))

        # 같은 시드 → 동일 벡터 → 코사인 유사도 1.0 → 동일 화자
        emb = _make_embedding(seed=10)
        id1 = self.tracker.track(emb)
        id2 = self.tracker.track(emb)

        assert id1 == id2
        info = self.tracker.get(id1)
        assert info is not None
        assert len(info["embeddings"]) == 2

    def test_track_creates_separate_ids_for_different_speakers(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """직교 임베딩은 별도 미등록 화자로 분류된다."""
        monkeypatch.setattr(server, "_diarizer", self.fake_diarizer)
        monkeypatch.setattr(server, "_find_matching_profile", lambda *a, **kw: (None, 0.0))

        emb_a = _make_embedding(dim=192, seed=1)
        emb_b = _make_orthogonal_embedding(dim=192, seed=999)

        id_a = self.tracker.track(emb_a)
        id_b = self.tracker.track(emb_b)

        # 직교 벡터이므로 유사도가 임계값 미만 → 별도 ID
        assert id_a != id_b

    # -- track(): 5개 이상 → candidate: true --

    def test_track_marks_candidate_after_min_segments(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_AUTO_REGISTER_MIN_SEGMENTS(5) 이상 누적 시 candidate가 True가 된다."""
        monkeypatch.setattr(server, "_diarizer", self.fake_diarizer)
        monkeypatch.setattr(server, "_find_matching_profile", lambda *a, **kw: (None, 0.0))

        emb = _make_embedding(seed=7)
        speaker_id = None
        for i in range(server._AUTO_REGISTER_MIN_SEGMENTS):
            speaker_id = self.tracker.track(emb)

        assert speaker_id is not None
        info = self.tracker.get(speaker_id)
        assert info is not None
        assert info["candidate"] is True

    def test_track_not_candidate_before_min_segments(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_AUTO_REGISTER_MIN_SEGMENTS 미만이면 candidate는 False이다."""
        monkeypatch.setattr(server, "_diarizer", self.fake_diarizer)
        monkeypatch.setattr(server, "_find_matching_profile", lambda *a, **kw: (None, 0.0))

        emb = _make_embedding(seed=7)
        speaker_id = None
        for _ in range(server._AUTO_REGISTER_MIN_SEGMENTS - 1):
            speaker_id = self.tracker.track(emb)

        assert speaker_id is not None
        info = self.tracker.get(speaker_id)
        assert info is not None
        assert info["candidate"] is False

    # -- reset(): 모든 임시 데이터 초기화 --

    def test_reset_clears_all_unknown_speakers(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """reset() 호출 시 모든 미등록 화자 데이터가 초기화된다."""
        monkeypatch.setattr(server, "_diarizer", self.fake_diarizer)
        monkeypatch.setattr(server, "_find_matching_profile", lambda *a, **kw: (None, 0.0))

        emb = _make_embedding(seed=1)
        self.tracker.track(emb)
        assert len(self.tracker.list_unknown()) > 0

        self.tracker.reset()

        assert len(self.tracker.list_unknown()) == 0

    def test_reset_resets_id_counter(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """reset() 후 다시 unknown_1부터 시작한다."""
        monkeypatch.setattr(server, "_diarizer", self.fake_diarizer)
        monkeypatch.setattr(server, "_find_matching_profile", lambda *a, **kw: (None, 0.0))

        emb = _make_embedding(seed=1)
        self.tracker.track(emb)
        self.tracker.reset()

        emb2 = _make_orthogonal_embedding(seed=99)
        new_id = self.tracker.track(emb2)
        assert new_id == "unknown_1"

    # -- list_unknown / get / remove --

    def test_list_unknown_returns_correct_structure(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """list_unknown()이 올바른 필드를 가진 딕셔너리 목록을 반환한다."""
        monkeypatch.setattr(server, "_diarizer", self.fake_diarizer)
        monkeypatch.setattr(server, "_find_matching_profile", lambda *a, **kw: (None, 0.0))

        emb = _make_embedding(seed=1)
        self.tracker.track(emb)

        items = self.tracker.list_unknown()
        assert len(items) == 1
        item = items[0]
        assert "id" in item
        assert "segment_count" in item
        assert "first_seen" in item
        assert "last_seen" in item
        assert "candidate" in item

    def test_get_returns_none_for_nonexistent_id(self) -> None:
        """존재하지 않는 ID로 get() 호출 시 None을 반환한다."""
        assert self.tracker.get("unknown_999") is None

    def test_remove_returns_true_for_existing_speaker(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """존재하는 미등록 화자 삭제 시 True를 반환한다."""
        monkeypatch.setattr(server, "_diarizer", self.fake_diarizer)
        monkeypatch.setattr(server, "_find_matching_profile", lambda *a, **kw: (None, 0.0))

        emb = _make_embedding(seed=1)
        speaker_id = self.tracker.track(emb)
        assert self.tracker.remove(speaker_id) is True
        assert self.tracker.get(speaker_id) is None

    def test_remove_returns_false_for_nonexistent_speaker(self) -> None:
        """존재하지 않는 미등록 화자 삭제 시 False를 반환한다."""
        assert self.tracker.remove("unknown_999") is False


# ---------------------------------------------------------------------------
# 2. API 통합 테스트
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_speaker_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """API 테스트 환경 격리: 프로필 저장소 + API 키 해제 + diarizer mock."""
    profiles_path = str(tmp_path / "data" / "speakers.json")
    monkeypatch.setattr(server, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(server, "_profiles_path", profiles_path)
    monkeypatch.setattr(server, "_api_key", "")

    fake_diarizer = _make_fake_diarizer()
    monkeypatch.setattr(server, "_diarizer", fake_diarizer)
    monkeypatch.setattr(server, "_find_matching_profile", lambda *a, **kw: (None, 0.0))

    # 매 테스트마다 tracker 초기화
    server._unknown_tracker.reset()


@pytest.fixture
def client() -> TestClient:
    with TestClient(server.app) as c:
        yield c


@pytest.fixture
def unknown_speaker_id() -> str:
    """tracker에 미등록 화자 1명(세그먼트 5개 이상)을 추가하고 ID를 반환한다."""
    emb = _make_embedding(seed=50)
    speaker_id = None
    for _ in range(6):
        speaker_id = server._unknown_tracker.track(emb)
    return speaker_id


class TestGetUnknownSpeakers:
    """GET /api/speakers/unknown 테스트."""

    def test_returns_empty_list_initially(self, client: TestClient) -> None:
        """미등록 화자가 없으면 빈 목록을 반환한다."""
        resp = client.get("/api/speakers/unknown")
        assert resp.status_code == 200
        data = resp.json()
        assert data["unknown_speakers"] == []
        assert data["available"] is True

    def test_returns_unavailable_when_diarizer_is_none(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """diarizer가 없으면 available: False를 반환한다."""
        monkeypatch.setattr(server, "_diarizer", None)
        resp = client.get("/api/speakers/unknown")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is False
        assert data["unknown_speakers"] == []

    def test_returns_unknown_speakers_after_tracking(
        self, client: TestClient, unknown_speaker_id: str,
    ) -> None:
        """tracker에 미등록 화자 추가 후 목록에 반환된다."""
        resp = client.get("/api/speakers/unknown")
        assert resp.status_code == 200
        speakers = resp.json()["unknown_speakers"]
        assert len(speakers) >= 1
        ids = [s["id"] for s in speakers]
        assert unknown_speaker_id in ids

    def test_unknown_speaker_has_correct_fields(
        self, client: TestClient, unknown_speaker_id: str,
    ) -> None:
        """미등록 화자 응답에 필수 필드가 포함된다."""
        resp = client.get("/api/speakers/unknown")
        speakers = resp.json()["unknown_speakers"]
        speaker = next(s for s in speakers if s["id"] == unknown_speaker_id)
        assert "segment_count" in speaker
        assert "first_seen" in speaker
        assert "last_seen" in speaker
        assert "candidate" in speaker
        assert speaker["candidate"] is True
        assert speaker["segment_count"] == 6


class TestAutoRegisterSpeaker:
    """POST /api/speakers/auto-register 테스트."""

    def test_registers_unknown_speaker_with_given_name(
        self, client: TestClient, unknown_speaker_id: str, tmp_path: Path,
    ) -> None:
        """미등록 화자를 지정 이름으로 프로필에 등록한다."""
        resp = client.post(
            "/api/speakers/auto-register",
            json={"speaker_id": unknown_speaker_id, "name": "김철수"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["name"] == "김철수"
        assert data["speaker_id"] == unknown_speaker_id

        # speakers.json에 실제 저장되었는지 확인
        profiles_file = tmp_path / "data" / "speakers.json"
        assert profiles_file.is_file()
        saved = json.loads(profiles_file.read_text(encoding="utf-8"))
        assert "김철수" in saved["speakers"]
        assert "embedding" in saved["speakers"]["김철수"]

    def test_assigns_default_name_when_not_provided(
        self, client: TestClient, unknown_speaker_id: str,
    ) -> None:
        """name 미지정 시 기본 이름 Speaker_N이 부여된다."""
        resp = client.post(
            "/api/speakers/auto-register",
            json={"speaker_id": unknown_speaker_id},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        # 기존 프로필이 0개 → Speaker_1
        assert data["name"] == "Speaker_1"

    def test_removes_from_unknown_list_after_registration(
        self, client: TestClient, unknown_speaker_id: str,
    ) -> None:
        """등록 후 미등록 목록에서 제거된다."""
        client.post(
            "/api/speakers/auto-register",
            json={"speaker_id": unknown_speaker_id, "name": "등록화자"},
        )
        resp = client.get("/api/speakers/unknown")
        ids = [s["id"] for s in resp.json()["unknown_speakers"]]
        assert unknown_speaker_id not in ids

    def test_updates_diarizer_runtime(
        self, client: TestClient, unknown_speaker_id: str,
    ) -> None:
        """등록 후 diarizer 런타임에 즉시 반영된다."""
        client.post(
            "/api/speakers/auto-register",
            json={"speaker_id": unknown_speaker_id, "name": "런타임화자"},
        )
        assert "런타임화자" in server._diarizer.speaker_embeddings
        assert "런타임화자" in server._diarizer._enrolled_names

    def test_returns_404_for_nonexistent_speaker_id(self, client: TestClient) -> None:
        """존재하지 않는 speaker_id로 등록 시 404를 반환한다."""
        resp = client.post(
            "/api/speakers/auto-register",
            json={"speaker_id": "unknown_999", "name": "없는화자"},
        )
        assert resp.status_code == 404

    def test_returns_400_when_speaker_id_missing(self, client: TestClient) -> None:
        """speaker_id가 없으면 400을 반환한다."""
        resp = client.post(
            "/api/speakers/auto-register",
            json={"name": "이름만"},
        )
        assert resp.status_code == 400

    def test_returns_400_when_diarizer_is_none(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """diarizer 비활성 시 400을 반환한다."""
        monkeypatch.setattr(server, "_diarizer", None)
        resp = client.post(
            "/api/speakers/auto-register",
            json={"speaker_id": "unknown_1", "name": "테스트"},
        )
        assert resp.status_code == 400

    def test_returns_409_for_duplicate_name(
        self, client: TestClient, unknown_speaker_id: str, tmp_path: Path,
    ) -> None:
        """이미 등록된 이름으로 등록 시 409를 반환한다."""
        # 먼저 프로필 파일에 기존 화자 추가
        profiles_dir = tmp_path / "data"
        profiles_dir.mkdir(parents=True, exist_ok=True)
        profiles_file = profiles_dir / "speakers.json"
        profiles_file.write_text(
            json.dumps({"speakers": {"중복이름": {"embedding": []}}}, ensure_ascii=False),
            encoding="utf-8",
        )

        resp = client.post(
            "/api/speakers/auto-register",
            json={"speaker_id": unknown_speaker_id, "name": "중복이름"},
        )
        assert resp.status_code == 409


class TestDeleteUnknownSpeaker:
    """DELETE /api/speakers/unknown/{id} 테스트."""

    def test_deletes_existing_unknown_speaker(
        self, client: TestClient, unknown_speaker_id: str,
    ) -> None:
        """미등록 화자를 성공적으로 삭제한다."""
        resp = client.delete(f"/api/speakers/unknown/{unknown_speaker_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["deleted"] == unknown_speaker_id

    def test_returns_404_for_nonexistent_id(self, client: TestClient) -> None:
        """존재하지 않는 ID로 삭제 시 404를 반환한다."""
        resp = client.delete("/api/speakers/unknown/unknown_999")
        assert resp.status_code == 404

    def test_rejects_path_traversal_in_speaker_id(self, client: TestClient) -> None:
        """speaker_id에 경로 탐색 문자가 포함되면 400을 반환한다."""
        for malicious_id in ["../etc/passwd", "unknown\\1", "..%2F"]:
            resp = client.delete(f"/api/speakers/unknown/{malicious_id}")
            assert resp.status_code in {400, 404, 422}

    def test_speaker_removed_from_list_after_delete(
        self, client: TestClient, unknown_speaker_id: str,
    ) -> None:
        """삭제 후 GET 목록에서 사라진다."""
        client.delete(f"/api/speakers/unknown/{unknown_speaker_id}")
        resp = client.get("/api/speakers/unknown")
        ids = [s["id"] for s in resp.json()["unknown_speakers"]]
        assert unknown_speaker_id not in ids


class TestApiKeyProtection:
    """API 키 인증이 미등록 화자 엔드포인트에도 적용되는지 확인."""

    def test_unknown_speakers_requires_api_key(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """MEETING_API_KEY 설정 시 인증 없는 요청은 401을 반환한다."""
        monkeypatch.setattr(server, "_api_key", "test-secret")

        resp = client.get("/api/speakers/unknown")
        assert resp.status_code == 401

        resp = client.get(
            "/api/speakers/unknown",
            headers={"x-api-key": "test-secret"},
        )
        assert resp.status_code == 200

    def test_auto_register_requires_api_key(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """MEETING_API_KEY 설정 시 auto-register도 인증이 필요하다."""
        monkeypatch.setattr(server, "_api_key", "test-secret")

        resp = client.post(
            "/api/speakers/auto-register",
            json={"speaker_id": "unknown_1", "name": "테스트"},
        )
        assert resp.status_code == 401

    def test_delete_unknown_requires_api_key(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """MEETING_API_KEY 설정 시 삭제도 인증이 필요하다."""
        monkeypatch.setattr(server, "_api_key", "test-secret")

        resp = client.delete("/api/speakers/unknown/unknown_1")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 3. 엣지케이스
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """엣지케이스 테스트."""

    def test_track_with_empty_embedding_list(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """빈 임베딩 배열로 track() 호출 시 None을 반환한다."""
        monkeypatch.setattr(server, "_diarizer", _make_fake_diarizer())
        tracker = UnknownSpeakerTracker()
        assert tracker.track([]) is None

    def test_track_with_numpy_not_available(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """numpy import 실패 시 track()이 예외를 발생시킨다 (빈 임베딩이 아닌 경우)."""
        monkeypatch.setattr(server, "_diarizer", _make_fake_diarizer())
        monkeypatch.setattr(server, "_find_matching_profile", lambda *a, **kw: (None, 0.0))

        tracker = UnknownSpeakerTracker()

        import builtins
        original_import = builtins.__import__

        def mock_import(name: str, *args, **kwargs):
            if name == "numpy":
                raise ImportError("numpy 미설치 mock")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with pytest.raises(ImportError):
                tracker.track([0.1, 0.2, 0.3])

    def test_concurrent_track_calls_are_thread_safe(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """여러 스레드에서 동시에 track()을 호출해도 데이터가 손상되지 않는다."""
        import threading

        monkeypatch.setattr(server, "_diarizer", _make_fake_diarizer())
        monkeypatch.setattr(server, "_find_matching_profile", lambda *a, **kw: (None, 0.0))

        tracker = UnknownSpeakerTracker()
        errors: list[Exception] = []

        def worker(seed: int) -> None:
            try:
                emb = _make_embedding(seed=seed)
                for _ in range(3):
                    tracker.track(emb)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i * 100,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # 각 스레드가 서로 다른 시드 → 서로 다른 미등록 화자 (직교 가능성 높음)
        # 최소 1명 이상의 미등록 화자가 있어야 한다
        assert len(tracker.list_unknown()) >= 1

    def test_session_reset_clears_unknown_tracker(
        self, client: TestClient, unknown_speaker_id: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """새 세션 시작 시 미등록 화자 추적 데이터가 초기화된다."""
        # 세션 회전 이벤트만 확인 (실제 콜백은 CLI에서 처리)
        assert len(server._unknown_tracker.list_unknown()) > 0

        resp = client.post("/api/sessions/new")
        assert resp.status_code == 200

        assert len(server._unknown_tracker.list_unknown()) == 0
