"""화자 프로필 CRUD API 통합 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import server


def _audio_upload(filename: str = "sample.wav", content: bytes = b"fake-audio-bytes") -> dict:
    return {"audio": (filename, content, "audio/wav")}


@pytest.fixture(autouse=True)
def _isolate_profile_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(server, "_profiles_path", str(tmp_path / "data" / "speakers.json"))
    monkeypatch.setattr(server, "_api_key", "")
    monkeypatch.setattr(server, "_get_audio_duration", lambda _: 12.34)


@pytest.fixture
def client() -> TestClient:
    with TestClient(server.app) as test_client:
        yield test_client


@pytest.fixture
def created_profile(client: TestClient) -> str:
    response = client.post(
        "/api/profiles",
        data={"name": "Alice", "description": "초기 설명"},
        files=_audio_upload(),
    )
    assert response.status_code == 200
    return "Alice"


def test_get_profiles_returns_empty_list_initially(client: TestClient) -> None:
    response = client.get("/api/profiles")

    assert response.status_code == 200
    assert response.json() == {"speakers": {}}


def test_post_profiles_creates_profile_from_multipart_form(client: TestClient) -> None:
    response = client.post(
        "/api/profiles",
        data={"name": "Alice", "description": "프로젝트 리드"},
        files=_audio_upload(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["name"] == "Alice"
    assert "Alice" in payload["speakers"]
    assert payload["speakers"]["Alice"]["description"] == "프로젝트 리드"
    assert payload["speakers"]["Alice"]["duration_seconds"] == 12.34


def test_get_profiles_includes_created_profile(client: TestClient, created_profile: str) -> None:
    response = client.get("/api/profiles")

    assert response.status_code == 200
    payload = response.json()
    assert created_profile in payload["speakers"]
    assert payload["speakers"][created_profile]["description"] == "초기 설명"


def test_put_profiles_updates_existing_profile(client: TestClient, created_profile: str) -> None:
    response = client.put(
        f"/api/profiles/{created_profile}",
        data={"description": "업데이트된 설명"},
        files=_audio_upload("updated.wav", b"updated-audio"),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["name"] == created_profile
    assert payload["speaker"]["description"] == "업데이트된 설명"
    assert payload["speaker"]["duration_seconds"] == 12.34
    assert payload["speaker"]["embedding"] == []


def test_delete_profiles_deletes_profile_with_json_response(
    client: TestClient,
    created_profile: str,
) -> None:
    response = client.delete(f"/api/profiles/{created_profile}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["deleted"] == created_profile

    list_response = client.get("/api/profiles")
    assert list_response.status_code == 200
    assert created_profile not in list_response.json()["speakers"]


def test_profiles_require_auth_when_meeting_api_key_is_set(client: TestClient) -> None:
    server._api_key = "test-secret"

    response = client.get("/api/profiles")
    assert response.status_code in {401, 403}

    authorized = client.get("/api/profiles", headers={"x-api-key": "test-secret"})
    assert authorized.status_code == 200


def test_put_missing_profile_returns_404(client: TestClient) -> None:
    response = client.put(
        "/api/profiles/missing-user",
        data={"description": "없음"},
        files=_audio_upload(),
    )

    assert response.status_code == 404


def test_post_profiles_created_profile_appears_in_list(client: TestClient) -> None:
    """POST 생성 후 GET 목록에 해당 화자가 포함된다."""
    client.post(
        "/api/profiles",
        data={"name": "Bob"},
        files=_audio_upload(),
    )

    list_response = client.get("/api/profiles")
    assert list_response.status_code == 200
    assert "Bob" in list_response.json()["speakers"]


def test_put_profiles_updates_description_only(client: TestClient, created_profile: str) -> None:
    """오디오 없이 description만 수정할 수 있다."""
    response = client.put(
        f"/api/profiles/{created_profile}",
        data={"description": "설명만 변경"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["speaker"]["description"] == "설명만 변경"


def test_delete_missing_profile_returns_404(client: TestClient) -> None:
    """존재하지 않는 프로필 삭제 시 404를 반환한다."""
    response = client.delete("/api/profiles/존재하지않는화자")

    assert response.status_code == 404
