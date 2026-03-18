from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from pyannote.audio import Inference, Model
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

    _PYANNOTE_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - 선택 의존성 처리
    torch = None  # type: ignore[assignment]
    Inference = None  # type: ignore[misc,assignment]
    Model = None  # type: ignore[misc,assignment]
    PretrainedSpeakerEmbedding = None  # type: ignore[assignment]
    _PYANNOTE_IMPORT_ERROR = exc

# segmentation Powerset 유틸 (선택 import)
try:
    from pyannote.audio.utils.powerset import Powerset

    _HAS_POWERSET = True
except ImportError:
    Powerset = None  # type: ignore[misc,assignment]
    _HAS_POWERSET = False

__all__ = ["SpeakerDiarizer"]

# --- 화자 분리 하이퍼파라미터 (리서치 v0.0.2 기반) ---
EMA_ALPHA = 0.1               # 임베딩 EMA 가중치 (새 발화 반영 비율)
MAX_SPEAKERS = 8               # 최대 화자 수 제한 (화자 폭발 방지)
BASE_THRESHOLD = 0.65          # 기본 유사도 임계값
THRESHOLD_PER_SPEAKER = 0.02   # 화자 수에 따른 임계값 증분 (6인→0.77)
MIN_DURATION_WEIGHT = 0.3      # 발화 길이 가중치 최솟값 (짧은 발화 과적합 방지)
DURATION_WEIGHT_SCALE = 10.0   # 가중치 정규화 기준 (초)

# --- 모델 설정 ---
EMBEDDING_MODEL_LEGACY = "pyannote/embedding"
EMBEDDING_MODEL_DEFAULT = "pyannote/wespeaker-voxceleb-resnet34-LM"
SEGMENTATION_MODEL = "pyannote/segmentation-3.0"
SEGMENTATION_DURATION = 10.0   # 세그먼테이션 모델 학습 윈도우 (초, 고정)
MIN_SEGMENT_DURATION = 0.5     # 임베딩 추출 최소 구간 (초)
SPEECH_THRESHOLD = 0.5         # 음성 감지 확률 임계값
PROFILE_VERSION = 2            # 프로필 파일 포맷 버전

# --- 한국어 최적화 (SONOTE_BETA=1에서만 활성) ---
KOREAN_SPEECH_THRESHOLD = 0.4
KOREAN_MIN_SEGMENT_DURATION = 0.3
KOREAN_BASE_THRESHOLD = 0.60
KOREAN_EMA_ALPHA = 0.15
KOREAN_BACKCHANNEL_MAX_DURATION = 0.3


def _is_korean_beta_mode() -> bool:
    """한국어 최적화 베타 모드 활성 여부."""
    return os.getenv("SONOTE_BETA") == "1"


def _effective_speech_threshold() -> float:
    """베타 모드에 따라 적용할 음성 판정 임계값."""
    if _is_korean_beta_mode():
        return KOREAN_SPEECH_THRESHOLD
    return SPEECH_THRESHOLD


def _effective_min_segment_duration() -> float:
    """베타 모드에 따라 적용할 최소 세그먼트 길이."""
    if _is_korean_beta_mode():
        return KOREAN_MIN_SEGMENT_DURATION
    return MIN_SEGMENT_DURATION


def _effective_base_threshold() -> float:
    """베타 모드에 따라 적용할 기본 유사도 임계값."""
    if _is_korean_beta_mode():
        return KOREAN_BASE_THRESHOLD
    return BASE_THRESHOLD


def _effective_ema_alpha(default_alpha: float) -> float:
    """베타 모드에서 기본 EMA_ALPHA 값을 한국어 프리셋으로 대체."""
    if _is_korean_beta_mode() and np.isclose(default_alpha, EMA_ALPHA):
        return KOREAN_EMA_ALPHA
    return default_alpha


def _merge_short_backchannel(
    segments: list[dict[str, Any]],
    max_duration: float = KOREAN_BACKCHANNEL_MAX_DURATION,
) -> list[dict[str, Any]]:
    """짧은 추임새 세그먼트를 인접 동일 화자 구간으로 병합.

    0.3초 미만 세그먼트가 앞/뒤 화자 사이에 끼어 있고
    앞/뒤 화자가 동일할 때, 가운데 세그먼트를 별도 화자로 유지하지 않는다.
    """
    if len(segments) < 3:
        return segments

    merged: list[dict[str, Any]] = []
    idx = 0

    while idx < len(segments):
        current = dict(segments[idx])
        current_duration = float(current["end"]) - float(current["start"])

        if 0.0 < current_duration < max_duration and merged and idx + 1 < len(segments):
            next_segment = segments[idx + 1]
            prev_segment = merged[-1]
            if prev_segment["speaker"] == next_segment["speaker"]:
                # 앞뒤 동일 화자면 가운데 짧은 추임새 세그먼트는 흡수
                prev_segment["end"] = max(
                    float(prev_segment["end"]), float(next_segment["end"]),
                )
                idx += 2
                continue

        if merged and merged[-1]["speaker"] == current["speaker"]:
            merged[-1]["end"] = max(float(merged[-1]["end"]), float(current["end"]))
        else:
            merged.append(current)

        idx += 1

    return merged


class SpeakerDiarizer:
    """실시간 화자 분리기 — 세그먼테이션 + 구간별 임베딩 방식

    v0.0.2 개선:
    - segmentation-3.0으로 청크 내 화자 교대 감지
    - 구간별 임베딩 추출 → 글로벌 화자 매칭
    - 임베딩 모델 설정 가능 (기본: wespeaker-resnet34-LM)
    - 기존 identify_speaker() 하위 호환 유지
    """

    def __init__(
        self,
        hf_token: str | None = None,
        device: str = "cuda",
        similarity_threshold: float | None = None,
        ema_alpha: float = EMA_ALPHA,
        max_speakers: int = MAX_SPEAKERS,
        profiles_path: str | Path | None = None,
        embedding_model: str | None = None,
        use_segmentation: bool = True,
    ) -> None:
        """pyannote 모델 로드.

        Args:
            hf_token: HuggingFace 토큰 (HF_TOKEN 환경변수 폴백)
            device: "cuda" 또는 "cpu"
            similarity_threshold: 고정 임계값 (None이면 동적 임계값 사용)
            ema_alpha: EMA 가중치 (0에 가까울수록 기존 임베딩 유지)
            max_speakers: 최대 허용 화자 수
            profiles_path: 사전 등록된 화자 프로필 JSON 경로
            embedding_model: 임베딩 모델 (None이면 프로필 기반 자동 선택)
            use_segmentation: 청크 내 세그먼테이션 활성화 (기본: True)
        """
        if not self.is_available():
            raise ImportError(
                "pyannote-audio가 설치되어 있지 않습니다. "
                "먼저 `pip install pyannote-audio`를 실행하세요."
            ) from _PYANNOTE_IMPORT_ERROR

        if similarity_threshold is not None and not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold는 0.0 이상 1.0 이하여야 합니다.")

        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError("ema_alpha는 0 초과 1.0 이하여야 합니다.")

        requested_device = device.strip().lower()
        if requested_device not in {"cuda", "cpu"}:
            raise ValueError('device는 "cuda" 또는 "cpu"만 허용됩니다.')

        if requested_device == "cuda" and not torch.cuda.is_available():
            requested_device = "cpu"

        self.device: str = requested_device
        self._fixed_threshold: float | None = similarity_threshold
        self._ema_alpha: float = _effective_ema_alpha(ema_alpha)
        self._max_speakers: int = max_speakers
        self._hf_token: str | None = hf_token or os.environ.get("HF_TOKEN")
        self._torch_device = torch.device(self.device)

        self.speaker_embeddings: dict[str, np.ndarray] = {}
        self._speaker_counts: dict[str, int] = {}
        self._embedding_backend: str = ""
        self._embedding_extractor: Any = None
        self._embedding_model_name: str = embedding_model or EMBEDDING_MODEL_DEFAULT
        self._profile_mode: bool = False
        self._enrolled_names: set[str] = set()
        self._unknown_count: int = 0

        # 세그먼테이션 모델 (청크 내 화자 교대 감지용)
        self._use_segmentation: bool = use_segmentation and _HAS_POWERSET
        self._seg_model: Any = None
        self._to_multilabel: Any = None

        # 프로필 로드 (임베딩 모델 자동 선택 포함)
        if profiles_path:
            self._auto_select_embedding_model(profiles_path)

        self._load_embedding_model()

        if self._use_segmentation:
            self._load_segmentation_model()

        if profiles_path:
            self.load_profiles(profiles_path)

    @property
    def similarity_threshold(self) -> float:
        """현재 유사도 임계값 (고정 또는 동적)."""
        if self._fixed_threshold is not None:
            return self._fixed_threshold
        return _effective_base_threshold() + self.get_speaker_count() * THRESHOLD_PER_SPEAKER

    @property
    def segmentation_available(self) -> bool:
        """세그먼테이션 모델 사용 가능 여부."""
        return self._seg_model is not None

    # ---------------------------------------------------------------
    # 공개 API: 화자 식별
    # ---------------------------------------------------------------

    def identify_speaker(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
        duration: float = 0.0,
    ) -> str:
        """오디오 청크의 화자를 식별 (하위 호환 API).

        세그먼테이션 비활성화 시 기존 방식 (전체 청크 → 단일 임베딩).
        세그먼테이션 활성화 시 청크 내 가장 긴 구간의 화자를 반환.

        Args:
            audio_chunk: 오디오 데이터 (float32 ndarray)
            sample_rate: 샘플레이트
            duration: 발화 길이(초) — 짧은 발화의 임베딩 영향 감소

        Returns:
            화자 라벨 ("A", "B", ... 또는 등록 이름)
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate는 양수여야 합니다.")

        # 세그먼테이션 활성화 시 구간별 처리
        if self.segmentation_available:
            segments = self.identify_speakers_in_chunk(audio_chunk, sample_rate)
            if segments:
                # 가장 긴 구간의 화자 반환
                longest = max(segments, key=lambda s: s["end"] - s["start"])
                return longest["speaker"]

            # 세그먼테이션에서 유효 구간을 못 찾으면 기존 방식으로 폴백
        return self._identify_speaker_legacy(audio_chunk, sample_rate, duration)

    def identify_speakers_in_chunk(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
    ) -> list[dict[str, Any]]:
        """청크 내 화자 교대를 감지하여 구간별 화자 라벨을 반환.

        segmentation-3.0으로 화자 변화 시점 감지 → 구간별 임베딩 추출 → 글로벌 매칭.

        Args:
            audio_chunk: 오디오 데이터 (float32 ndarray, 5~15초 권장)
            sample_rate: 샘플레이트

        Returns:
            [{"speaker": "A", "start": 0.0, "end": 3.5}, ...]
            세그먼테이션 불가 시 빈 리스트 반환.
        """
        if not self.segmentation_available:
            return []

        if sample_rate <= 0:
            raise ValueError("sample_rate는 양수여야 합니다.")

        # 1. 세그먼테이션: 화자 교대 시점 + 로컬 화자 구간 추출
        local_segments = self._segment_chunk(audio_chunk, sample_rate)
        if not local_segments:
            return []

        # 2. 구간별 임베딩 추출 + 글로벌 화자 매칭
        results: list[dict[str, Any]] = []
        min_segment_samples = int(_effective_min_segment_duration() * sample_rate)

        for seg in local_segments:
            start_sample = int(seg["start"] * sample_rate)
            end_sample = min(int(seg["end"] * sample_rate), len(audio_chunk))

            if (end_sample - start_sample) < min_segment_samples:
                continue

            seg_audio = audio_chunk[start_sample:end_sample]
            embedding = self._extract_embedding(seg_audio, sample_rate)
            duration = seg["end"] - seg["start"]

            # 글로벌 화자 매칭 (기존 identify_speaker 로직과 동일)
            speaker = self._match_or_create_speaker(embedding, duration)
            results.append({
                "speaker": speaker,
                "start": seg["start"],
                "end": seg["end"],
            })

        if _is_korean_beta_mode():
            return _merge_short_backchannel(results)

        return results

    # ---------------------------------------------------------------
    # 공개 API: 등록/프로필
    # ---------------------------------------------------------------

    def check_duplicate(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
        threshold: float = 0.70,
    ) -> tuple[str | None, float]:
        """등록 전 중복 화자 검사 — 임베딩 유사도 기반.

        Returns:
            (가장 유사한 화자 이름, 유사도). 유사 화자 없으면 (None, 0.0).
        """
        if not self.speaker_embeddings:
            return None, 0.0

        embedding = self._extract_embedding(audio_chunk, sample_rate)
        best_name: str | None = None
        best_sim: float = 0.0
        for name, known in self.speaker_embeddings.items():
            sim = float(self._cosine_similarity(embedding, known))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim >= threshold:
            return best_name, best_sim
        return None, best_sim

    def enroll(
        self,
        name: str,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
    ) -> None:
        """화자 임베딩 사전 등록.

        Args:
            name: 화자 이름 (예: "민수")
            audio_chunk: 오디오 데이터 (float32 ndarray, 10~30초 권장)
            sample_rate: 샘플레이트
        """
        if not name or not name.strip():
            raise ValueError("화자 이름은 비어있을 수 없습니다.")
        name = name.strip()

        embedding = self._extract_embedding(audio_chunk, sample_rate)
        self.speaker_embeddings[name] = embedding
        self._speaker_counts[name] = 0
        self._enrolled_names.add(name)
        self._profile_mode = True

    def save_profiles(self, path: str | Path) -> None:
        """등록된 화자 임베딩을 JSON으로 저장."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 기존 파일이 있으면 로드하여 병합
        existing: dict[str, Any] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                existing = data.get("speakers", {})

        # 등록된 화자만 저장 (unknown 제외)
        speakers: dict[str, Any] = {}
        for name in self._enrolled_names:
            if name in self.speaker_embeddings:
                speakers[name] = {
                    "embedding": self.speaker_embeddings[name].tolist(),
                    "enrolled_at": existing.get(name, {}).get(
                        "enrolled_at", datetime.now().isoformat()
                    ),
                }

        # 기존에 있던 화자 중 이번에 등록하지 않은 화자도 유지
        for name, info in existing.items():
            if name not in speakers:
                speakers[name] = info

        out = {
            "version": PROFILE_VERSION,
            "embedding_model": self._embedding_model_name,
            "speakers": speakers,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    def load_profiles(self, path: str | Path) -> None:
        """JSON에서 화자 임베딩 로드. 프로필 모드 활성화."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"프로필 파일 없음: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        speakers = data.get("speakers", {})
        if not speakers:
            raise ValueError("프로필에 등록된 화자가 없습니다.")

        self.speaker_embeddings.clear()
        self._speaker_counts.clear()
        self._enrolled_names.clear()

        for name, info in speakers.items():
            embedding = np.array(info["embedding"], dtype=np.float32)
            self.speaker_embeddings[name] = embedding
            self._speaker_counts[name] = 0
            self._enrolled_names.add(name)

        self._profile_mode = True
        self._unknown_count = 0

    @staticmethod
    def list_profiles(path: str | Path) -> list[dict[str, str]]:
        """프로필 파일에서 등록된 화자 목록 반환."""
        path = Path(path)
        if not path.exists():
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        result = []
        for name, info in data.get("speakers", {}).items():
            result.append({
                "name": name,
                "enrolled_at": info.get("enrolled_at", "?"),
            })
        return result

    @staticmethod
    def delete_from_profiles(path: str | Path, name: str) -> bool:
        """프로필 파일에서 특정 화자 삭제. 성공 시 True."""
        path = Path(path)
        if not path.exists():
            return False

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        speakers = data.get("speakers", {})
        if name not in speakers:
            return False

        del speakers[name]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True

    def get_speaker_count(self) -> int:
        """현재까지 감지된 화자 수"""
        return len(self.speaker_embeddings)

    def reset(self) -> None:
        """화자 상태 초기화 (임베딩 DB 클리어)"""
        self.speaker_embeddings.clear()
        self._speaker_counts.clear()
        self._enrolled_names.clear()
        self._profile_mode = False
        self._unknown_count = 0

    @staticmethod
    def is_available() -> bool:
        """pyannote-audio 설치 여부 확인"""
        return _PYANNOTE_IMPORT_ERROR is None

    # ---------------------------------------------------------------
    # 내부: 모델 로딩
    # ---------------------------------------------------------------

    def _auto_select_embedding_model(self, profiles_path: str | Path) -> None:
        """프로필 파일에서 임베딩 모델 자동 선택 (하위 호환)."""
        path = Path(profiles_path)
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        profile_version = data.get("version", 1)
        profile_model = data.get("embedding_model", EMBEDDING_MODEL_LEGACY)

        if profile_version < PROFILE_VERSION:
            # v1 프로필: 레거시 모델로 자동 전환
            print(f"[화자 분리] v1 프로필 감지 — {EMBEDDING_MODEL_LEGACY} 사용")
            print("[화자 분리] 새 모델 사용하려면 화자를 다시 등록하세요 (enroll)")
            self._embedding_model_name = EMBEDDING_MODEL_LEGACY
        elif profile_model != self._embedding_model_name:
            # 프로필의 모델과 현재 설정이 다르면 프로필 기준으로 전환
            self._embedding_model_name = profile_model

    def _load_embedding_model(self) -> None:
        """임베딩 모델 로드. 모델명에 따라 적절한 백엔드 선택."""
        kwargs: dict[str, Any] = {"device": self._torch_device}
        if self._hf_token:
            kwargs["use_auth_token"] = self._hf_token

        model_name = self._embedding_model_name

        try:
            self._embedding_extractor = PretrainedSpeakerEmbedding(model_name, **kwargs)
            self._embedding_backend = "pretrained_speaker_embedding"
            return
        except TypeError:
            kwargs.pop("use_auth_token", None)
            if self._hf_token:
                kwargs["token"] = self._hf_token
            try:
                self._embedding_extractor = PretrainedSpeakerEmbedding(model_name, **kwargs)
                self._embedding_backend = "pretrained_speaker_embedding"
                return
            except Exception:
                pass
        except Exception:
            pass

        # Model + Inference 폴백
        if self._hf_token:
            try:
                model = Model.from_pretrained(model_name, token=self._hf_token)
            except TypeError:
                model = Model.from_pretrained(model_name, use_auth_token=self._hf_token)
        else:
            model = Model.from_pretrained(model_name)

        model.to(self._torch_device)  # type: ignore[union-attr]
        model.eval()  # type: ignore[union-attr]
        self._embedding_extractor = Inference(model, window="whole")  # type: ignore[arg-type]
        self._embedding_backend = "inference"

    def _load_segmentation_model(self) -> None:
        """segmentation-3.0 모델 로드 (청크 내 화자 교대 감지용)."""
        if not _HAS_POWERSET:
            return

        try:
            if self._hf_token:
                try:
                    seg_model = Model.from_pretrained(
                        SEGMENTATION_MODEL, token=self._hf_token,
                    )
                except TypeError:
                    seg_model = Model.from_pretrained(
                        SEGMENTATION_MODEL, use_auth_token=self._hf_token,
                    )
            else:
                seg_model = Model.from_pretrained(SEGMENTATION_MODEL)

            seg_model.to(self._torch_device)  # type: ignore[union-attr]
            seg_model.eval()  # type: ignore[union-attr]
            self._seg_model = seg_model

            # Powerset → multi-label 변환기 (최대 3화자, 최대 2명 동시)
            self._to_multilabel = Powerset(
                num_classes=3, max_set_size=2,
            ).to_multilabel
        except Exception as exc:
            print(f"[화자 분리] segmentation-3.0 로드 실패: {exc}")
            print("[화자 분리] 기존 방식(전체 청크 임베딩)으로 동작합니다.")
            self._seg_model = None

    # ---------------------------------------------------------------
    # 내부: 세그먼테이션
    # ---------------------------------------------------------------

    def _segment_chunk(
        self, audio_chunk: np.ndarray, sample_rate: int,
    ) -> list[dict[str, Any]]:
        """세그먼테이션 모델로 청크 내 화자 구간 추출.

        10초 초과 청크는 다중 윈도우로 분할 처리하여
        전체 구간의 화자 정보를 보존한다.

        Args:
            audio_chunk: float32 ndarray, shape (samples,)
            sample_rate: 샘플레이트

        Returns:
            [{"local_speaker": int, "start": float, "end": float}, ...]
        """
        window_samples = int(SEGMENTATION_DURATION * sample_rate)
        chunk_duration = len(audio_chunk) / sample_rate

        # 10초 이하: 단일 윈도우
        if chunk_duration <= SEGMENTATION_DURATION + 0.5:
            return self._segment_window(audio_chunk, sample_rate, 0.0)

        # 10초 초과: 다중 윈도우 (비중첩)
        all_segments: list[dict[str, Any]] = []
        offset = 0

        while offset < len(audio_chunk):
            window = audio_chunk[offset:offset + window_samples]
            time_offset = offset / sample_rate
            segs = self._segment_window(window, sample_rate, time_offset)
            all_segments.extend(segs)
            offset += window_samples

        return all_segments

    def _segment_window(
        self, audio_window: np.ndarray, sample_rate: int, time_offset: float,
    ) -> list[dict[str, Any]]:
        """단일 10초 윈도우에 대한 세그먼테이션 실행.

        Args:
            audio_window: float32 ndarray (최대 10초)
            sample_rate: 샘플레이트
            time_offset: 전체 청크 내에서 이 윈도우의 시작 시간(초)

        Returns:
            [{"local_speaker": int, "start": float, "end": float}, ...]
        """
        target_samples = int(SEGMENTATION_DURATION * sample_rate)
        original_len = len(audio_window)
        actual_duration = original_len / sample_rate

        # 10초로 패딩 또는 트림
        if original_len < target_samples:
            padded = np.pad(audio_window, (0, target_samples - original_len))
        elif original_len > target_samples:
            padded = audio_window[:target_samples]
            actual_duration = SEGMENTATION_DURATION
        else:
            padded = audio_window

        # 세그먼테이션 실행: (1, 1, 160000) → (1, ~625, 7)
        waveform = torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0)
        waveform = waveform.to(self._torch_device)

        with torch.no_grad():
            powerset_output = self._seg_model(waveform)
            probs = self._to_multilabel(powerset_output)

        probs_np = probs[0].cpu().numpy()  # (frames, 3)
        num_frames = probs_np.shape[0]
        frame_dur = SEGMENTATION_DURATION / num_frames

        # 실제 오디오 길이에 해당하는 프레임만 사용
        valid_frames = min(num_frames, int(actual_duration / frame_dur))
        probs_np = probs_np[:valid_frames]

        # 프레임별 주 화자 + 음성 여부 판정
        dominant = probs_np.argmax(axis=1)
        is_speech = probs_np.max(axis=1) > _effective_speech_threshold()

        # 연속 화자 구간 추출
        segments: list[dict[str, Any]] = []
        current_spk = -1
        seg_start = 0.0

        for i in range(len(dominant)):
            t = i * frame_dur

            if not is_speech[i]:
                if current_spk >= 0:
                    segments.append({
                        "local_speaker": current_spk,
                        "start": time_offset + seg_start,
                        "end": time_offset + t,
                    })
                    current_spk = -1
                continue

            spk = int(dominant[i])
            if spk != current_spk:
                if current_spk >= 0:
                    segments.append({
                        "local_speaker": current_spk,
                        "start": time_offset + seg_start,
                        "end": time_offset + t,
                    })
                current_spk = spk
                seg_start = t

        # 마지막 구간 마무리
        if current_spk >= 0:
            segments.append({
                "local_speaker": current_spk,
                "start": time_offset + seg_start,
                "end": time_offset + valid_frames * frame_dur,
            })

        return segments

    # ---------------------------------------------------------------
    # 내부: 화자 매칭
    # ---------------------------------------------------------------

    def _match_or_create_speaker(
        self, embedding: np.ndarray, duration: float = 0.0,
    ) -> str:
        """임베딩을 기존 화자와 매칭하거나 새 화자 생성.

        identify_speaker()와 identify_speakers_in_chunk() 공통 로직.
        """
        duration_weight = max(
            MIN_DURATION_WEIGHT,
            min(duration / DURATION_WEIGHT_SCALE, 1.0),
        ) if duration > 0 else 1.0

        if not self.speaker_embeddings:
            label = self._new_speaker_label()
            self.speaker_embeddings[label] = embedding
            self._speaker_counts[label] = 1
            return label

        best_label: str | None = None
        best_similarity: float = -1.0
        for label, known_embedding in self.speaker_embeddings.items():
            similarity = self._cosine_similarity(embedding, known_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_label = label

        if best_label is None:
            label = self._new_speaker_label()
            self.speaker_embeddings[label] = embedding
            self._speaker_counts[label] = 1
            return label

        threshold = self.similarity_threshold

        if best_similarity >= threshold:
            self._update_embedding(best_label, embedding, duration_weight)
            return best_label

        # MAX_SPEAKERS 도달 시 가장 유사한 기존 화자에 강제 병합
        if self.get_speaker_count() >= self._max_speakers:
            self._update_embedding(best_label, embedding, duration_weight)
            return best_label

        label = self._new_speaker_label()
        self.speaker_embeddings[label] = embedding
        self._speaker_counts[label] = 1
        return label

    def _update_embedding(
        self, label: str, embedding: np.ndarray, duration_weight: float,
    ) -> None:
        """EMA로 화자 임베딩 갱신."""
        alpha = self._ema_alpha * duration_weight
        # 프로필 모드에서 등록된 화자는 EMA 비율 낮춤 (등록 임베딩 보존)
        if self._profile_mode and label in self._enrolled_names:
            alpha *= 0.3
        current = self.speaker_embeddings[label]
        updated = alpha * embedding + (1.0 - alpha) * current
        self.speaker_embeddings[label] = updated.astype(np.float32)
        self._speaker_counts[label] += 1

    def _identify_speaker_legacy(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
        duration: float = 0.0,
    ) -> str:
        """기존 방식: 전체 청크 → 단일 임베딩 → 화자 매칭."""
        embedding = self._extract_embedding(audio_chunk, sample_rate)
        return self._match_or_create_speaker(embedding, duration)

    # ---------------------------------------------------------------
    # 내부: 임베딩 추출
    # ---------------------------------------------------------------

    def _extract_embedding(self, audio_chunk: np.ndarray, sample_rate: int) -> np.ndarray:
        """오디오 청크를 임베딩 벡터(1차원 float32)로 변환."""
        waveform = self._prepare_waveform(audio_chunk)

        if self._embedding_backend == "pretrained_speaker_embedding":
            batch = waveform.unsqueeze(0).to(self._torch_device)
            vector = self._embedding_extractor(batch)
        else:
            audio = {"waveform": waveform, "sample_rate": sample_rate}
            vector = self._embedding_extractor(audio)

        if isinstance(vector, torch.Tensor):
            vector_np = vector.detach().cpu().numpy()
        else:
            vector_np = np.asarray(vector)

        if vector_np.ndim == 0:
            raise RuntimeError("임베딩 추출 결과가 비어 있습니다.")

        if vector_np.ndim > 1:
            vector_np = np.mean(vector_np, axis=0)

        embedding = vector_np.reshape(-1).astype(np.float32)
        if embedding.size == 0:
            raise RuntimeError("유효한 임베딩을 얻지 못했습니다.")
        return embedding

    def _prepare_waveform(self, audio_chunk: np.ndarray) -> torch.Tensor:
        """float32 ndarray를 모델 입력용 torch.Tensor로 변환."""
        waveform_np = np.asarray(audio_chunk, dtype=np.float32)
        if waveform_np.size == 0:
            raise ValueError("audio_chunk가 비어 있습니다.")

        if waveform_np.ndim == 1:
            waveform_np = waveform_np[np.newaxis, :]
        elif waveform_np.ndim == 2:
            if waveform_np.shape[0] <= 8:
                waveform_np = np.mean(waveform_np, axis=0, keepdims=True)
            elif waveform_np.shape[1] <= 8:
                waveform_np = np.mean(waveform_np, axis=1, keepdims=True).T
            else:
                raise ValueError("2차원 audio_chunk의 채널 축을 해석할 수 없습니다.")
        else:
            raise ValueError("audio_chunk는 1차원 또는 2차원 ndarray여야 합니다.")

        return torch.from_numpy(waveform_np)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """코사인 유사도 계산."""
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _new_speaker_label(self) -> str:
        """새 화자 라벨 생성.

        프로필 모드: "unknown_1", "unknown_2", ...
        일반 모드: "A", "B", "C", ...
        """
        if self._profile_mode:
            self._unknown_count += 1
            return f"unknown_{self._unknown_count}"
        return chr(ord("A") + self.get_speaker_count())
