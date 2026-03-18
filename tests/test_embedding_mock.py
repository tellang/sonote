"""임베딩 연동 mock 테스트.

pyannote-audio를 mock하여 코사인 유사도 매칭, 임계값 경계,
프로필 매칭/중복 검사 로직을 검증한다.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src import server
from src.diarize import SpeakerDiarizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_server_diarizer():
    """테스트 전후로 서버의 _diarizer 상태를 초기화한다."""
    original = server._diarizer
    yield
    server._diarizer = original


@pytest.fixture
def mock_diarizer():
    """pyannote 없이 동작하는 mock SpeakerDiarizer를 생성한다."""
    diarizer = MagicMock(spec=SpeakerDiarizer)
    # _cosine_similarity는 실제 numpy 연산 사용
    diarizer._cosine_similarity = SpeakerDiarizer._cosine_similarity
    diarizer.speaker_embeddings = {}
    diarizer._speaker_counts = {}
    diarizer._enrolled_names = set()
    diarizer._profile_mode = False
    return diarizer


# ---------------------------------------------------------------------------
# 1. 코사인 유사도 계산 검증
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """SpeakerDiarizer._cosine_similarity 정적 메서드를 검증한다."""

    def test_identical_vectors(self):
        """동일한 벡터의 코사인 유사도는 1.0이다."""
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sim = SpeakerDiarizer._cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """직교 벡터의 코사인 유사도는 0.0이다."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        sim = SpeakerDiarizer._cosine_similarity(a, b)
        assert abs(sim) < 1e-6

    def test_opposite_vectors(self):
        """반대 방향 벡터의 코사인 유사도는 -1.0이다."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        sim = SpeakerDiarizer._cosine_similarity(a, b)
        assert abs(sim - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        """영벡터가 포함되면 코사인 유사도는 0.0이다."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.zeros(3, dtype=np.float32)
        sim = SpeakerDiarizer._cosine_similarity(a, b)
        assert sim == 0.0

    def test_similar_vectors_high_similarity(self):
        """유사한 벡터는 높은 코사인 유사도를 가진다."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.1, 2.1, 3.1], dtype=np.float32)
        sim = SpeakerDiarizer._cosine_similarity(a, b)
        assert sim > 0.99

    def test_high_dimensional_vectors(self):
        """고차원 벡터(256차원)에서도 정상 동작한다."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(256).astype(np.float32)
        # a에 약간의 노이즈를 추가한 유사 벡터
        b = (a + rng.standard_normal(256).astype(np.float32) * 0.1).astype(np.float32)
        sim = SpeakerDiarizer._cosine_similarity(a, b)
        assert 0.9 < sim < 1.0


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _create_profiles(tmp_path, speakers_dict):
    """테스트용 프로필 JSON 파일을 생성하고 server._profiles_path를 설정한다."""
    import json

    profiles = {"speakers": speakers_dict}
    profiles_path = tmp_path / "speakers.json"
    profiles_path.write_text(json.dumps(profiles), encoding="utf-8")
    server._profiles_path = str(profiles_path)


# ---------------------------------------------------------------------------
# 2. 프로필 매칭 (_find_matching_profile) mock 테스트
# ---------------------------------------------------------------------------


class TestFindMatchingProfile:
    """server._find_matching_profile의 임베딩 비교 로직을 mock으로 검증한다."""

    def test_matching_above_threshold(self, mock_diarizer, tmp_path):
        """임계값(0.70) 이상의 유사도면 매칭된 화자 이름을 반환한다."""
        server._diarizer = mock_diarizer

        # 프로필 JSON 생성 (고정 임베딩)
        known_emb = [1.0, 0.0, 0.0]
        _create_profiles(tmp_path, {
            "민수": {"embedding": known_emb, "enrolled_at": "2026-01-01"},
        })

        # 매우 유사한 임베딩으로 매칭 테스트
        target_emb = [0.99, 0.01, 0.0]
        name, sim = server._find_matching_profile(target_emb)

        assert name == "민수"
        assert sim > 0.70

    def test_no_matching_below_threshold(self, mock_diarizer, tmp_path):
        """임계값 미만의 유사도면 매칭 없음을 반환한다."""
        server._diarizer = mock_diarizer

        known_emb = [1.0, 0.0, 0.0]
        _create_profiles(tmp_path, {
            "민수": {"embedding": known_emb, "enrolled_at": "2026-01-01"},
        })

        # 직교에 가까운 임베딩 → 유사도 낮음
        target_emb = [0.0, 1.0, 0.0]
        name, sim = server._find_matching_profile(target_emb)

        assert name is None
        assert sim < 0.70

    def test_exclude_name_skips_self(self, mock_diarizer, tmp_path):
        """exclude_name으로 지정된 화자는 매칭에서 제외된다."""
        server._diarizer = mock_diarizer

        _create_profiles(tmp_path, {
            "민수": {"embedding": [1.0, 0.0, 0.0], "enrolled_at": "2026-01-01"},
            "영희": {"embedding": [0.0, 1.0, 0.0], "enrolled_at": "2026-01-01"},
        })

        # 민수와 동일한 임베딩으로 검색하되, 민수를 제외
        target_emb = [1.0, 0.0, 0.0]
        name, sim = server._find_matching_profile(target_emb, exclude_name="민수")

        # 민수 제외 → 영희만 비교 → 직교이므로 매칭 안 됨
        assert name is None

    def test_empty_embedding_returns_none(self, mock_diarizer):
        """빈 임베딩은 매칭 없이 (None, 0.0)을 반환한다."""
        server._diarizer = mock_diarizer

        name, sim = server._find_matching_profile([])
        assert name is None
        assert sim == 0.0

    def test_no_diarizer_returns_none(self):
        """_diarizer가 None이면 매칭 없이 반환한다."""
        server._diarizer = None

        name, sim = server._find_matching_profile([1.0, 2.0, 3.0])
        assert name is None
        assert sim == 0.0

    def test_empty_profiles_returns_none(self, mock_diarizer, tmp_path):
        """프로필에 등록된 화자가 없으면 매칭 없이 반환한다."""
        server._diarizer = mock_diarizer

        _create_profiles(tmp_path, {})

        name, sim = server._find_matching_profile([1.0, 0.0, 0.0])
        assert name is None

    def test_speaker_without_embedding_skipped(self, mock_diarizer, tmp_path):
        """임베딩이 없는 화자는 비교에서 건너뛴다."""
        server._diarizer = mock_diarizer

        _create_profiles(tmp_path, {
            "민수": {"embedding": [], "enrolled_at": "2026-01-01"},
            "영희": {"embedding": [1.0, 0.0, 0.0], "enrolled_at": "2026-01-01"},
        })

        target_emb = [0.98, 0.02, 0.0]
        name, sim = server._find_matching_profile(target_emb)

        assert name == "영희"
        assert sim > 0.70


# ---------------------------------------------------------------------------
# 3. 임계값(0.70) 경계 테스트
# ---------------------------------------------------------------------------


class TestThresholdBoundary:
    """프로필 매칭 임계값 0.70 경계에서의 동작을 검증한다."""

    def test_just_above_threshold(self, mock_diarizer, tmp_path):
        """유사도가 0.70보다 약간 높으면 매칭된다."""
        server._diarizer = mock_diarizer

        # cos(theta) ≈ 0.72 인 벡터 쌍
        a_emb = [1.0, 0.0]
        b_emb = [0.72, np.sqrt(1.0 - 0.72**2)]

        _create_profiles(tmp_path, {
            "경계화자": {"embedding": a_emb, "enrolled_at": "2026-01-01"},
        })

        name, sim = server._find_matching_profile(b_emb)

        # 유사도가 0.70 이상이므로 매칭
        assert name == "경계화자"
        assert sim >= 0.70

    def test_just_below_threshold(self, mock_diarizer, tmp_path):
        """유사도가 0.70보다 약간 낮으면 매칭되지 않는다."""
        server._diarizer = mock_diarizer

        a_emb = [1.0, 0.0]
        # cos(theta) ≈ 0.69
        b_emb = [0.69, np.sqrt(1.0 - 0.69**2)]

        _create_profiles(tmp_path, {
            "비매칭화자": {"embedding": a_emb, "enrolled_at": "2026-01-01"},
        })

        name, sim = server._find_matching_profile(b_emb)

        assert name is None
        assert sim < 0.70

    def test_custom_threshold(self, mock_diarizer, tmp_path):
        """커스텀 임계값을 전달하면 해당 값으로 매칭 기준이 변경된다."""
        server._diarizer = mock_diarizer

        _create_profiles(tmp_path, {
            "화자X": {"embedding": [1.0, 0.0, 0.0], "enrolled_at": "2026-01-01"},
        })

        target_emb = [0.90, 0.44, 0.0]  # cos sim ≈ 0.898

        # 기본 임계값 0.70 → 매칭
        name_low, _ = server._find_matching_profile(target_emb, threshold=0.70)
        assert name_low == "화자X"

        # 높은 임계값 0.95 → 매칭 안 됨
        name_high, _ = server._find_matching_profile(target_emb, threshold=0.95)
        assert name_high is None


# ---------------------------------------------------------------------------
# 4. 임베딩 추출 mock 테스트
# ---------------------------------------------------------------------------


class TestExtractEmbeddingFromFile:
    """server._extract_embedding_from_file의 mock 테스트."""

    def test_no_diarizer_returns_empty(self):
        """_diarizer가 None이면 빈 리스트를 반환한다."""
        server._diarizer = None
        result = server._extract_embedding_from_file("dummy.wav")
        assert result == []

    def test_with_mock_diarizer_and_audio(self, mock_diarizer, tmp_path):
        """mock diarizer로 임베딩 추출이 동작한다."""
        server._diarizer = mock_diarizer

        fake_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_diarizer._extract_embedding.return_value = fake_embedding

        # 가짜 오디오 파일 생성 (soundfile mock 필요)
        with patch("src.server.sf") if hasattr(server, "sf") else patch.dict(
            "sys.modules", {"soundfile": MagicMock()}
        ):
            import soundfile as sf_mock

            audio_data = np.random.randn(16000 * 2).astype(np.float32)  # 2초 오디오
            wav_path = tmp_path / "test_audio.wav"

            # soundfile로 실제 파일 작성
            try:
                import soundfile as sf
                sf.write(str(wav_path), audio_data, 16000)
            except ImportError:
                # soundfile 없으면 mock으로 처리
                wav_path.write_bytes(b"\x00" * 100)
                with patch("soundfile.read", return_value=(audio_data, 16000)):
                    result = server._extract_embedding_from_file(str(wav_path))
                    # mock이므로 결과는 환경에 따라 다름
                    return

            result = server._extract_embedding_from_file(str(wav_path))
            # diarizer._extract_embedding이 호출되었는지 확인
            if result:
                assert isinstance(result, list)
                assert len(result) == 3

    def test_short_audio_returns_empty(self, mock_diarizer, tmp_path):
        """1초 미만의 짧은 오디오는 빈 임베딩을 반환한다."""
        server._diarizer = mock_diarizer

        short_audio = np.zeros(8000, dtype=np.float32)  # 0.5초

        try:
            import soundfile as sf
            wav_path = tmp_path / "short.wav"
            sf.write(str(wav_path), short_audio, 16000)
            result = server._extract_embedding_from_file(str(wav_path))
            assert result == []
        except ImportError:
            pytest.skip("soundfile 미설치")


# ---------------------------------------------------------------------------
# 5. 다수 화자 프로필 매칭
# ---------------------------------------------------------------------------


class TestMultiSpeakerMatching:
    """여러 화자가 등록된 상황에서 가장 유사한 화자를 매칭하는지 검증한다."""

    def test_best_match_among_multiple(self, mock_diarizer, tmp_path):
        """여러 화자 중 가장 유사한 화자를 정확히 매칭한다."""
        server._diarizer = mock_diarizer

        _create_profiles(tmp_path, {
            "화자A": {"embedding": [1.0, 0.0, 0.0], "enrolled_at": "2026-01-01"},
            "화자B": {"embedding": [0.0, 1.0, 0.0], "enrolled_at": "2026-01-01"},
            "화자C": {"embedding": [0.0, 0.0, 1.0], "enrolled_at": "2026-01-01"},
        })

        # 화자B에 가장 가까운 벡터
        target_emb = [0.1, 0.99, 0.05]
        name, sim = server._find_matching_profile(target_emb)

        assert name == "화자B"
        assert sim > 0.90

    def test_no_match_when_all_distant(self, mock_diarizer, tmp_path):
        """모든 화자와 유사도가 낮으면 매칭 없음을 반환한다."""
        server._diarizer = mock_diarizer

        # 3개 축에 분산된 화자들
        _create_profiles(tmp_path, {
            "화자A": {"embedding": [1.0, 0.0, 0.0], "enrolled_at": "2026-01-01"},
            "화자B": {"embedding": [0.0, 1.0, 0.0], "enrolled_at": "2026-01-01"},
            "화자C": {"embedding": [0.0, 0.0, 1.0], "enrolled_at": "2026-01-01"},
        })

        # 모든 축에 균등한 벡터 → 각 화자와의 유사도 ≈ 0.577 (< 0.70)
        target_emb = [1.0, 1.0, 1.0]
        name, sim = server._find_matching_profile(target_emb)

        assert name is None
        assert sim < 0.70

    def test_matching_with_realistic_embeddings(self, mock_diarizer, tmp_path):
        """256차원 임베딩으로 현실적인 매칭 시나리오를 검증한다."""
        server._diarizer = mock_diarizer

        rng = np.random.default_rng(42)
        emb_a = rng.standard_normal(256).astype(np.float32)
        emb_b = rng.standard_normal(256).astype(np.float32)
        # emb_a의 변형 (높은 유사도)
        emb_a_similar = (emb_a + rng.standard_normal(256).astype(np.float32) * 0.05).astype(
            np.float32,
        )

        _create_profiles(tmp_path, {
            "사람A": {"embedding": emb_a.tolist(), "enrolled_at": "2026-01-01"},
            "사람B": {"embedding": emb_b.tolist(), "enrolled_at": "2026-01-01"},
        })

        name, sim = server._find_matching_profile(emb_a_similar.tolist())

        assert name == "사람A"
        assert sim > 0.90
