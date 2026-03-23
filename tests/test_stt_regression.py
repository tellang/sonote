"""STT 정확도 회귀 테스트 — 모델 교체 시 정확도 저하를 자동 감지.

tests/fixtures/meeting_15s.wav (7분 지점 15초 미팅 녹음)에 대해
기대 텍스트와 비교하여 CER이 임계치 이하인지 검증한다.
"""
from __future__ import annotations

import os
import re
import pytest
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures"
AUDIO_15S = FIXTURE_DIR / "meeting_15s.wav"

# 기준 전사 결과 (seastar105/whisper-medium-ko-zeroth CT2, 2026-03-20 확인)
GOLDEN_TEXT = "아 그래도 교실이 오늘 조금 조용해서 괜찮은 것 같아요 이때보다 덜"

# CER 임계치: 30% 이하여야 PASS (모델 교체 시 심한 퇴행만 잡는 수준)
CER_THRESHOLD = 0.30

# WER 임계치
WER_THRESHOLD = 0.50


def _cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate (편집 거리 기반)."""
    ref = reference.replace(" ", "")
    hyp = hypothesis.replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0

    # DP 편집 거리
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / n


def _wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / n


# ---------------------------------------------------------------------------
# 유틸 테스트 (항상 실행)
# ---------------------------------------------------------------------------


def test_cer_identical():
    assert _cer("안녕하세요", "안녕하세요") == 0.0


def test_cer_completely_different():
    assert _cer("가나다", "라마바") == 1.0


def test_cer_partial():
    # 1글자 차이 / 3글자 = 0.333...
    assert 0.3 <= _cer("가나다", "가나라") <= 0.34


def test_wer_identical():
    assert _wer("오늘 날씨가 좋습니다", "오늘 날씨가 좋습니다") == 0.0


def test_wer_one_word_wrong():
    # 1/3 = 0.333
    assert 0.3 <= _wer("오늘 날씨가 좋습니다", "오늘 날씨가 나쁩니다") <= 0.34


# ---------------------------------------------------------------------------
# STT 회귀 테스트 (GPU + 모델 필요)
# ---------------------------------------------------------------------------

_requires_gpu = pytest.mark.skipif(
    not os.environ.get("SONOTE_TEST_GPU"),
    reason="GPU STT regression test — set SONOTE_TEST_GPU=1 to enable",
)

_requires_fixture = pytest.mark.skipif(
    not AUDIO_15S.exists(),
    reason=f"Fixture not found: {AUDIO_15S}",
)


@_requires_gpu
@_requires_fixture
def test_stt_cer_below_threshold():
    """현재 기본 모델의 CER이 임계치 이하인지 확인."""
    from src.transcribe import transcribe_audio

    segments = transcribe_audio(str(AUDIO_15S), language="ko")
    hypothesis = " ".join(seg["text"].strip() for seg in segments if seg["text"].strip())

    cer = _cer(GOLDEN_TEXT, hypothesis)
    print(f"\n[STT regression] CER={cer:.3f} (threshold={CER_THRESHOLD})")
    print(f"  golden:     {GOLDEN_TEXT}")
    print(f"  hypothesis: {hypothesis}")
    assert cer <= CER_THRESHOLD, f"CER {cer:.3f} > {CER_THRESHOLD} — STT 정확도 퇴행!"


@_requires_gpu
@_requires_fixture
def test_stt_wer_below_threshold():
    """현재 기본 모델의 WER이 임계치 이하인지 확인."""
    from src.transcribe import transcribe_audio

    segments = transcribe_audio(str(AUDIO_15S), language="ko")
    hypothesis = " ".join(seg["text"].strip() for seg in segments if seg["text"].strip())

    wer = _wer(GOLDEN_TEXT, hypothesis)
    print(f"\n[STT regression] WER={wer:.3f} (threshold={WER_THRESHOLD})")
    assert wer <= WER_THRESHOLD, f"WER {wer:.3f} > {WER_THRESHOLD} — STT 정확도 퇴행!"


@_requires_gpu
@_requires_fixture
def test_stt_no_hallucination():
    """전사 결과에 반복 환각이 없는지 확인."""
    from src.transcribe import transcribe_audio

    segments = transcribe_audio(str(AUDIO_15S), language="ko")
    hypothesis = " ".join(seg["text"].strip() for seg in segments if seg["text"].strip())

    # 같은 3글자 이상 패턴이 5회 이상 반복되면 환각
    for match in re.finditer(r"(.{3,}?)\1{4,}", hypothesis):
        pytest.fail(f"환각 감지: '{match.group(0)[:50]}...' — 반복 패턴: '{match.group(1)}'")


@_requires_gpu
@_requires_fixture
def test_stt_output_not_empty():
    """전사 결과가 비어있지 않은지 확인."""
    from src.transcribe import transcribe_audio

    segments = transcribe_audio(str(AUDIO_15S), language="ko")
    assert len(segments) > 0, "전사 결과가 비어있음 — 모델 로드 또는 VAD 문제"
    total_text = "".join(seg["text"] for seg in segments)
    assert len(total_text) > 5, f"전사 텍스트가 너무 짧음: '{total_text}'"
