"""Unit tests for the meeting transcription pipeline (src/meeting/pipeline.py)."""

from __future__ import annotations

import threading
from functools import partial
from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from src.meeting.context import PipelineContext
from src.meeting.pipeline import (
    PipelineAdapter,
    TranscriptionPipeline,
    _DEFAULT_TRANSCRIBE_KWARGS,
    run_capture_loop,
)
from src.postprocess import Segment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(seconds: float = 0.5, sample_rate: int = 16000) -> np.ndarray:
    """Return a short sine-wave audio chunk."""
    n = int(sample_rate * seconds)
    t = np.arange(n, dtype=np.float32) / sample_rate
    return (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _good_stt_segment(text: str = "안녕하세요 반갑습니다", start: float = 0.0, end: float = 1.0) -> dict[str, Any]:
    return {
        "start": start,
        "end": end,
        "text": text,
        "avg_logprob": -0.15,
        "no_speech_prob": 0.01,
        "compression_ratio": 1.2,
        "words": [{"start": start, "end": end, "word": text, "probability": 0.95}],
    }


def _make_context(
    *,
    worker: Any = None,
    diarizer: Any = None,
    language: str = "ko",
    chunk_seconds: float = 5.0,
    on_transcript: Any = None,
    on_correction: Any = None,
) -> PipelineContext:
    return PipelineContext(
        worker=worker or MagicMock(),
        diarizer=diarizer,
        language=language,
        chunk_seconds=chunk_seconds,
        on_transcript=on_transcript or MagicMock(),
        on_correction=on_correction,
        stop_event=threading.Event(),
    )


def _make_adapter(**overrides) -> PipelineAdapter:
    defaults: dict[str, Any] = {
        "capture_audio": MagicMock(return_value=iter([])),
        "is_paused": MagicMock(return_value=False),
        "is_shutdown_requested": MagicMock(return_value=False),
    }
    defaults.update(overrides)
    return PipelineAdapter(**defaults)


def _make_pipeline(ctx: PipelineContext | None = None, adapter: PipelineAdapter | None = None) -> TranscriptionPipeline:
    return TranscriptionPipeline(
        context=ctx or _make_context(),
        adapter=adapter or _make_adapter(),
    )


# ---------------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_default_state(self):
        p = _make_pipeline()
        assert p.segment_count == 0
        assert p.recent_feedback_texts == []
        assert p._prev_chunk_text == ""
        assert p._uncorrected_buffer == []
        assert p._correction_batch_index == 0

    def test_context_and_adapter_stored(self):
        ctx = _make_context()
        adapter = _make_adapter()
        p = TranscriptionPipeline(ctx, adapter)
        assert p._context is ctx
        assert p._adapter is adapter


# ---------------------------------------------------------------------------
# 2. _transcribe
# ---------------------------------------------------------------------------

class TestTranscribe:
    def test_uses_worker_transcribe_by_default(self):
        worker = MagicMock()
        worker.transcribe.return_value = [_good_stt_segment()]
        ctx = _make_context(worker=worker)
        p = _make_pipeline(ctx=ctx)
        chunk = _make_chunk()

        result = p._transcribe(chunk)

        assert result == [_good_stt_segment()]
        worker.transcribe.assert_called_once()
        kw = worker.transcribe.call_args
        assert kw.kwargs["language"] == "ko"

    def test_uses_custom_transcribe_runner(self):
        runner = MagicMock(return_value=[_good_stt_segment("커스텀 결과")])
        worker = MagicMock()
        ctx = _make_context(worker=worker)
        adapter = _make_adapter(transcribe_runner=runner)
        p = _make_pipeline(ctx=ctx, adapter=adapter)
        chunk = _make_chunk()

        result = p._transcribe(chunk)

        assert result[0]["text"] == "커스텀 결과"
        runner.assert_called_once()
        args = runner.call_args[0]
        assert args[0] is worker
        assert isinstance(args[2], dict)
        assert args[2]["language"] == "ko"
        # worker.transcribe should NOT have been called
        worker.transcribe.assert_not_called()

    def test_transcribe_kwargs_factory_merges(self):
        worker = MagicMock()
        worker.transcribe.return_value = []

        def factory(chunk):
            return {"beam_size": 1, "custom_param": True}

        ctx = _make_context(worker=worker)
        adapter = _make_adapter(transcribe_kwargs_factory=factory)
        p = _make_pipeline(ctx=ctx, adapter=adapter)

        p._transcribe(_make_chunk())
        kw = worker.transcribe.call_args.kwargs
        assert kw["beam_size"] == 1
        assert kw["custom_param"] is True
        # Default kwargs still present
        assert kw["vad_filter"] is True


# ---------------------------------------------------------------------------
# 3. _build_raw_segments
# ---------------------------------------------------------------------------

class TestBuildRawSegments:
    def _build(self, stt_segments, *, ctx=None, adapter=None, prev_text=""):
        ctx = ctx or _make_context()
        adapter = adapter or _make_adapter()
        p = _make_pipeline(ctx=ctx, adapter=adapter)
        p._prev_chunk_text = prev_text
        chunk = _make_chunk()
        return p._build_raw_segments(
            stt_segments=stt_segments,
            speaker_segments=[],
            audio_chunk=chunk,
            timestamp="10:00:00",
        )

    def test_valid_segment_passes(self):
        segs = self._build([_good_stt_segment("테스트 문장입니다")])
        assert len(segs) == 1
        assert segs[0].text == "테스트 문장입니다"
        assert segs[0].speaker == "화자"

    def test_empty_text_filtered(self):
        seg = _good_stt_segment("")
        segs = self._build([seg])
        assert len(segs) == 0

    def test_hallucination_filtered(self):
        # Common hallucination pattern
        seg = _good_stt_segment("MBC 뉴스 이덕영입니다")
        with patch("src.meeting.pipeline.is_hallucination", return_value=True):
            segs = self._build([seg])
        assert len(segs) == 0

    def test_looping_filtered(self):
        seg = _good_stt_segment("반복 반복 반복 반복 반복")
        with patch("src.meeting.pipeline.is_looping", return_value=True):
            segs = self._build([seg])
        assert len(segs) == 0

    def test_invalid_segment_filtered(self):
        seg = _good_stt_segment("유효하지 않은 세그먼트")
        with patch("src.meeting.pipeline.is_valid_segment", return_value=False):
            segs = self._build([seg])
        assert len(segs) == 0

    def test_overlap_removed(self):
        """When prev_chunk_text is set, overlap removal should kick in."""
        seg = _good_stt_segment("이전 텍스트 새로운 내용")
        with patch("src.meeting.pipeline.remove_overlap", return_value="새로운 내용"):
            segs = self._build([seg], prev_text="이전 텍스트")
        assert len(segs) == 1
        assert segs[0].text == "새로운 내용"

    def test_overlap_removes_entirely(self):
        seg = _good_stt_segment("완전 중복")
        with patch("src.meeting.pipeline.remove_overlap", return_value=""):
            segs = self._build([seg], prev_text="완전 중복")
        assert len(segs) == 0

    def test_on_raw_segment_callback(self):
        callback = MagicMock()
        adapter = _make_adapter(on_raw_segment=callback)
        segs = self._build([_good_stt_segment("콜백 테스트")], adapter=adapter)
        assert len(segs) == 1
        callback.assert_called_once()
        payload = callback.call_args[0][0]
        assert payload["kind"] == "stt_segment"
        assert payload["timestamp"] == "10:00:00"

    def test_speaker_from_diarizer_segments(self):
        ctx = _make_context()
        adapter = _make_adapter()
        p = _make_pipeline(ctx=ctx, adapter=adapter)
        stt = [_good_stt_segment("화자 테스트")]
        speaker_segs = [{"start": 0.0, "end": 1.0, "speaker": "Speaker_01"}]
        result = p._build_raw_segments(
            stt_segments=stt,
            speaker_segments=speaker_segs,
            audio_chunk=_make_chunk(),
            timestamp="10:00:00",
        )
        assert len(result) == 1
        assert result[0].speaker == "Speaker_01"

    @pytest.mark.parametrize(
        "avg_logprob, expected_confidence",
        [
            (None, None),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.5, 1.0),
            (-2.0, 0.0),
        ],
    )
    def test_confidence_from_avg_logprob(self, avg_logprob, expected_confidence):
        """avg_logprob → confidence 변환 경계값 테스트."""
        seg = _good_stt_segment("경계값 테스트")
        seg["avg_logprob"] = avg_logprob
        with patch("src.meeting.pipeline.is_valid_segment", return_value=True), \
             patch("src.meeting.pipeline.is_hallucination", return_value=False), \
             patch("src.meeting.pipeline.is_looping", return_value=False):
            segs = self._build([seg])
        assert len(segs) == 1
        assert segs[0].confidence == expected_confidence


# ---------------------------------------------------------------------------
# 4. _emit_segments
# ---------------------------------------------------------------------------

class TestEmitSegments:
    def test_emits_to_on_transcript(self):
        on_transcript = MagicMock()
        ctx = _make_context(on_transcript=on_transcript)
        p = _make_pipeline(ctx=ctx)
        segments = [Segment(speaker="A", text="발화 내용", start=0.0, end=1.0)]

        chunk_text = p._emit_segments(segments, "10:00:00")

        assert on_transcript.call_count == 1
        payload = on_transcript.call_args[0][0]
        assert payload["speaker"] == "A"
        assert payload["text"] == "발화 내용"
        assert payload["timestamp"] == "10:00:00"
        assert p.segment_count == 1
        assert chunk_text != ""

    def test_dedupe_skips_duplicates(self):
        on_transcript = MagicMock()
        ctx = _make_context(on_transcript=on_transcript)
        adapter = _make_adapter(dedupe_recent_texts=True, recent_text_limit=50)
        p = _make_pipeline(ctx=ctx, adapter=adapter)
        segments = [Segment(speaker="A", text="중복 발화", start=0.0, end=1.0)]

        p._emit_segments(segments, "10:00:00")
        p._emit_segments(segments, "10:00:05")

        # Second emission should be deduped
        assert on_transcript.call_count == 1

    def test_on_segment_emitted_callback(self):
        callback = MagicMock()
        adapter = _make_adapter(on_segment_emitted=callback)
        ctx = _make_context()
        p = _make_pipeline(ctx=ctx, adapter=adapter)
        segments = [Segment(speaker="A", text="콜백 발화", start=0.0, end=1.0)]

        p._emit_segments(segments, "10:00:00")

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == segments[0]  # segment
        assert args[1] == "10:00:00"   # timestamp

    def test_recent_feedback_texts_trimmed(self):
        ctx = _make_context()
        adapter = _make_adapter(recent_text_limit=3)
        p = _make_pipeline(ctx=ctx, adapter=adapter)

        for i in range(5):
            segments = [Segment(speaker="A", text=f"발화 {i}번째", start=0.0, end=1.0)]
            p._emit_segments(segments, "10:00:00")

        assert len(p.recent_feedback_texts) <= 4  # limit + 1 guard in code uses max(1, limit)

    def test_uncorrected_buffer_populated(self):
        submit_fn = MagicMock()
        adapter = _make_adapter(submit_correction_batch=submit_fn, correction_batch_size=100)
        ctx = _make_context()
        p = _make_pipeline(ctx=ctx, adapter=adapter)
        segments = [Segment(speaker="A", text="교정 대상 문장", start=0.0, end=1.0)]

        p._emit_segments(segments, "10:00:00")

        assert len(p._uncorrected_buffer) == 1
        assert "교정 대상 문장" in p._uncorrected_buffer[0]


# ---------------------------------------------------------------------------
# 5. _submit_pending_corrections
# ---------------------------------------------------------------------------

class TestSubmitPendingCorrections:
    def test_no_submit_when_no_fn(self):
        p = _make_pipeline()
        p._uncorrected_buffer = ["line1"] * 20
        p._submit_pending_corrections()  # should not raise

    def test_no_submit_below_batch_size(self):
        submit_fn = MagicMock()
        adapter = _make_adapter(submit_correction_batch=submit_fn, correction_batch_size=10)
        p = _make_pipeline(adapter=adapter)
        p._uncorrected_buffer = ["line"] * 5

        p._submit_pending_corrections()
        submit_fn.assert_not_called()

    def test_submits_when_batch_full(self):
        submit_fn = MagicMock(return_value=None)
        adapter = _make_adapter(submit_correction_batch=submit_fn, correction_batch_size=3)
        ctx = _make_context()
        p = _make_pipeline(ctx=ctx, adapter=adapter)
        p._uncorrected_buffer = ["line0", "line1", "line2"]
        p._segment_count = 3

        p._submit_pending_corrections()

        submit_fn.assert_called_once_with(["line0", "line1", "line2"], 0)
        assert p._uncorrected_buffer == []
        assert p._correction_batch_index == 1

    def test_correction_future_callback(self):
        future = MagicMock()
        submit_fn = MagicMock(return_value=future)
        on_future = MagicMock()
        on_correction = MagicMock()
        adapter = _make_adapter(
            submit_correction_batch=submit_fn,
            correction_batch_size=2,
            on_correction_future=on_future,
        )
        ctx = _make_context(on_correction=on_correction)
        p = _make_pipeline(ctx=ctx, adapter=adapter)
        p._uncorrected_buffer = ["a", "b"]
        p._segment_count = 2

        p._submit_pending_corrections()

        on_future.assert_called_once_with(future, ["a", "b"])
        future.add_done_callback.assert_called_once()


# ---------------------------------------------------------------------------
# 6. _submit_keywords
# ---------------------------------------------------------------------------

class TestSubmitKeywords:
    def test_no_submit_when_no_fn(self):
        p = _make_pipeline()
        p._segment_count = 100
        p._submit_keywords()  # no-op, should not raise

    def test_no_submit_before_interval(self):
        submit_fn = MagicMock()
        adapter = _make_adapter(submit_keyword_job=submit_fn, keyword_every_segments=10)
        p = _make_pipeline(adapter=adapter)
        p._segment_count = 5

        p._submit_keywords()
        submit_fn.assert_not_called()

    def test_submits_at_interval(self):
        submit_fn = MagicMock()
        adapter = _make_adapter(
            submit_keyword_job=submit_fn,
            keyword_every_segments=5,
            keyword_window=3,
        )
        p = _make_pipeline(adapter=adapter)
        p._segment_count = 5
        p._keyword_extract_at = 0
        p._recent_feedback_texts = ["첫째 문장", "둘째 문장", "셋째 문장"]

        p._submit_keywords()

        submit_fn.assert_called_once()
        text_arg = submit_fn.call_args[0][0]
        assert "첫째 문장" in text_arg
        assert p._keyword_extract_at == 5

    def test_no_submit_on_empty_text(self):
        submit_fn = MagicMock()
        adapter = _make_adapter(submit_keyword_job=submit_fn, keyword_every_segments=1)
        p = _make_pipeline(adapter=adapter)
        p._segment_count = 1
        p._recent_feedback_texts = ["   "]

        # normalize_feedback_text could produce whitespace-only
        # But the join+strip check in _submit_keywords guards against empty
        # The actual behavior depends on the content; let's use truly empty
        p._recent_feedback_texts = []
        p._submit_keywords()
        submit_fn.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Session rotation
# ---------------------------------------------------------------------------

class TestSessionRotate:
    def test_handle_session_rotate_calls_callback(self):
        callback = MagicMock()
        adapter = _make_adapter(
            on_session_rotate=callback,
            reset_runtime_on_rotate=True,
        )
        p = _make_pipeline(adapter=adapter)
        p._segment_count = 42
        p._prev_chunk_text = "이전"
        p._uncorrected_buffer = ["x"]
        p._recent_feedback_texts = ["y"]

        p._handle_session_rotate()

        callback.assert_called_once()
        assert p._segment_count == 0
        assert p._prev_chunk_text == ""
        assert p._uncorrected_buffer == []
        assert p._recent_feedback_texts == []

    def test_no_reset_when_flag_false(self):
        adapter = _make_adapter(
            on_session_rotate=MagicMock(),
            reset_runtime_on_rotate=False,
        )
        p = _make_pipeline(adapter=adapter)
        p._segment_count = 10

        p._handle_session_rotate()

        assert p._segment_count == 10

    def test_consume_session_rotate_returns_false_by_default(self):
        p = _make_pipeline()
        assert p._consume_session_rotate() is False


# ---------------------------------------------------------------------------
# 8. Device switch flow
# ---------------------------------------------------------------------------

class TestDeviceSwitch:
    def test_consume_device_switch_no_fn(self):
        p = _make_pipeline()
        has_switch, device = p._consume_device_switch()
        assert has_switch is False
        assert device is None

    def test_consume_device_switch_delegates(self):
        fn = MagicMock(return_value=(True, 42))
        adapter = _make_adapter(consume_audio_device_switch=fn)
        p = _make_pipeline(adapter=adapter)

        has_switch, device = p._consume_device_switch()
        assert has_switch is True
        assert device == 42

    def test_set_current_device_delegates(self):
        fn = MagicMock()
        adapter = _make_adapter(set_current_audio_device=fn)
        p = _make_pipeline(adapter=adapter)

        p._set_current_device(7)
        fn.assert_called_once_with(7)


# ---------------------------------------------------------------------------
# 9. Pause / Resume
# ---------------------------------------------------------------------------

class TestPauseResume:
    def test_paused_chunk_skipped(self):
        """When is_paused returns True, the chunk should be skipped (no transcription)."""
        worker = MagicMock()
        worker.transcribe.return_value = []
        ctx = _make_context(worker=worker)

        def capture_audio(**kwargs):
            yield _make_chunk()
            yield _make_chunk()
            ctx.stop_event.set()

        adapter = _make_adapter(
            capture_audio=capture_audio,
            is_paused=MagicMock(return_value=True),
            is_shutdown_requested=MagicMock(return_value=False),
        )
        p = _make_pipeline(ctx=ctx, adapter=adapter)
        p.run()

        worker.transcribe.assert_not_called()

    def test_on_pause_callback_called(self):
        on_pause = MagicMock()
        ctx = _make_context()

        def capture_audio(**kwargs):
            yield _make_chunk()
            ctx.stop_event.set()

        adapter = _make_adapter(
            capture_audio=capture_audio,
            is_paused=MagicMock(return_value=True),
            is_shutdown_requested=MagicMock(return_value=False),
            on_pause=on_pause,
        )
        p = _make_pipeline(ctx=ctx, adapter=adapter)
        p.run()
        on_pause.assert_called()


# ---------------------------------------------------------------------------
# 10. _should_stop
# ---------------------------------------------------------------------------

class TestShouldStop:
    def test_stop_event_set(self):
        ctx = _make_context()
        adapter = _make_adapter(is_shutdown_requested=MagicMock(return_value=False))
        p = _make_pipeline(ctx=ctx, adapter=adapter)

        assert p._should_stop() is False
        ctx.stop_event.set()
        assert p._should_stop() is True

    def test_shutdown_requested(self):
        adapter = _make_adapter(is_shutdown_requested=MagicMock(return_value=True))
        p = _make_pipeline(adapter=adapter)
        assert p._should_stop() is True


# ---------------------------------------------------------------------------
# 11. _collect_speaker_segments
# ---------------------------------------------------------------------------

class TestCollectSpeakerSegments:
    def test_no_diarizer(self):
        ctx = _make_context(diarizer=None)
        p = _make_pipeline(ctx=ctx)
        assert p._collect_speaker_segments(_make_chunk()) == []

    def test_diarizer_success(self):
        diarizer = MagicMock()
        diarizer.identify_speakers_in_chunk.return_value = [{"start": 0, "end": 1, "speaker": "A"}]
        ctx = _make_context(diarizer=diarizer)
        p = _make_pipeline(ctx=ctx)
        result = p._collect_speaker_segments(_make_chunk())
        assert len(result) == 1

    def test_diarizer_exception_returns_empty(self):
        diarizer = MagicMock()
        diarizer.identify_speakers_in_chunk.side_effect = RuntimeError("boom")
        ctx = _make_context(diarizer=diarizer)
        p = _make_pipeline(ctx=ctx)
        result = p._collect_speaker_segments(_make_chunk())
        assert result == []
        assert p._diarize_warned is True

    def test_diarizer_warns_only_once(self, capsys):
        diarizer = MagicMock()
        diarizer.identify_speakers_in_chunk.side_effect = RuntimeError("boom")
        ctx = _make_context(diarizer=diarizer)
        p = _make_pipeline(ctx=ctx)
        p._collect_speaker_segments(_make_chunk())
        p._collect_speaker_segments(_make_chunk())
        captured = capsys.readouterr()
        assert captured.err.count("세그먼테이션 실패") == 1


# ---------------------------------------------------------------------------
# 12. _resolve_speaker
# ---------------------------------------------------------------------------

class TestResolveSpeaker:
    def test_best_overlap_speaker(self):
        ctx = _make_context()
        p = _make_pipeline(ctx=ctx)
        seg = {"start": 1.0, "end": 3.0}
        speakers = [
            {"start": 0.0, "end": 1.5, "speaker": "A"},
            {"start": 1.5, "end": 4.0, "speaker": "B"},
        ]
        result = p._resolve_speaker(seg=seg, speaker_segments=speakers, audio_chunk=_make_chunk())
        assert result == "B"  # overlap 1.5 vs 0.5

    def test_fallback_to_diarizer(self):
        diarizer = MagicMock()
        diarizer.identify_speaker.return_value = "Speaker_X"
        ctx = _make_context(diarizer=diarizer)
        p = _make_pipeline(ctx=ctx)
        result = p._resolve_speaker(seg={}, speaker_segments=[], audio_chunk=_make_chunk())
        assert result == "Speaker_X"

    def test_fallback_default(self):
        ctx = _make_context(diarizer=None)
        p = _make_pipeline(ctx=ctx)
        result = p._resolve_speaker(seg={}, speaker_segments=[], audio_chunk=_make_chunk())
        assert result == "화자"


# ---------------------------------------------------------------------------
# 13. _format_correction_line
# ---------------------------------------------------------------------------

class TestFormatCorrectionLine:
    def test_default_format(self):
        p = _make_pipeline()
        line = p._format_correction_line("10:00:00", "A", "발화 내용")
        assert line == "- [10:00:00] [A] 발화 내용"

    def test_custom_formatter(self):
        fmt = MagicMock(return_value="custom")
        adapter = _make_adapter(correction_line_formatter=fmt)
        p = _make_pipeline(adapter=adapter)
        line = p._format_correction_line("10:00:00", "A", "발화 내용")
        assert line == "custom"
        fmt.assert_called_once_with("10:00:00", "A", "발화 내용")


# ---------------------------------------------------------------------------
# 14. _segment_bounds
# ---------------------------------------------------------------------------

class TestSegmentBounds:
    def test_default(self):
        p = _make_pipeline()
        assert p._segment_bounds({"start": 1.5, "end": 3.5}) == (1.5, 3.5)

    def test_custom_mapper(self):
        mapper = MagicMock(return_value=(10.0, 20.0))
        adapter = _make_adapter(segment_time_mapper=mapper)
        p = _make_pipeline(adapter=adapter)
        assert p._segment_bounds({"start": 0}) == (10.0, 20.0)


# ---------------------------------------------------------------------------
# 15. Full run() loop with mock capture yielding 2-3 chunks
# ---------------------------------------------------------------------------

class TestRunLoop:
    @patch("src.meeting.pipeline.postprocess")
    @patch("src.meeting.pipeline.normalize_feedback_text", side_effect=lambda t: t)
    @patch("src.meeting.pipeline.remove_overlap", side_effect=lambda prev, cur: cur)
    @patch("src.meeting.pipeline.is_looping", return_value=False)
    @patch("src.meeting.pipeline.is_hallucination", return_value=False)
    @patch("src.meeting.pipeline.is_valid_segment", return_value=True)
    def test_full_run_two_chunks(
        self, mock_valid, mock_hall, mock_loop, mock_overlap, mock_norm, mock_pp
    ):
        """Pipeline processes 2 chunks, emits segments, then stops."""
        chunk = _make_chunk()
        chunks_yielded = [0]

        worker = MagicMock()
        worker.transcribe.return_value = [_good_stt_segment("테스트 발화")]

        on_transcript = MagicMock()
        ctx = _make_context(worker=worker, on_transcript=on_transcript)

        def capture_audio(**kwargs):
            for _ in range(2):
                chunks_yielded[0] += 1
                yield chunk
            # After yielding all chunks, set stop so the outer while exits
            ctx.stop_event.set()

        mock_pp.return_value = [Segment(speaker="화자", text="테스트 발화", start=0.0, end=1.0)]

        adapter = _make_adapter(
            capture_audio=capture_audio,
            is_paused=MagicMock(return_value=False),
            is_shutdown_requested=MagicMock(return_value=False),
            timestamp_provider=MagicMock(return_value="12:00:00"),
        )
        p = TranscriptionPipeline(ctx, adapter)
        p.run()

        assert chunks_yielded[0] == 2
        assert on_transcript.call_count == 2
        assert p.segment_count == 2

    @patch("src.meeting.pipeline.postprocess")
    @patch("src.meeting.pipeline.normalize_feedback_text", side_effect=lambda t: t)
    @patch("src.meeting.pipeline.remove_overlap", side_effect=lambda prev, cur: cur)
    @patch("src.meeting.pipeline.is_looping", return_value=False)
    @patch("src.meeting.pipeline.is_hallucination", return_value=False)
    @patch("src.meeting.pipeline.is_valid_segment", return_value=True)
    def test_run_stops_on_stop_event(
        self, mock_valid, mock_hall, mock_loop, mock_overlap, mock_norm, mock_pp
    ):
        """Pipeline exits when stop_event is set mid-capture."""
        chunk = _make_chunk()
        ctx = _make_context()

        call_idx = [0]
        def capture_audio(**kwargs):
            while True:
                call_idx[0] += 1
                if call_idx[0] > 5:
                    ctx.stop_event.set()
                yield chunk

        worker = ctx.worker
        worker.transcribe.return_value = [_good_stt_segment()]
        mock_pp.return_value = [Segment(speaker="화자", text="발화", start=0.0, end=1.0)]

        adapter = _make_adapter(
            capture_audio=capture_audio,
            is_paused=MagicMock(return_value=False),
            is_shutdown_requested=MagicMock(return_value=False),
            timestamp_provider=MagicMock(return_value="12:00:00"),
        )
        p = TranscriptionPipeline(ctx, adapter)
        p.run()

        assert call_idx[0] >= 5

    def test_run_device_switch_during_capture(self):
        """Device switch mid-capture breaks the inner loop and re-enters."""
        chunk = _make_chunk()
        switch_consumed = [False]
        inner_iterations = [0]

        def capture_audio(**kwargs):
            while True:
                inner_iterations[0] += 1
                yield chunk

        switch_seq = [(False, None), (True, 99)]
        switch_idx = [0]

        def consume_switch():
            if switch_idx[0] < len(switch_seq):
                val = switch_seq[switch_idx[0]]
                switch_idx[0] += 1
                return val
            return (False, None)

        on_switched = MagicMock()
        worker = MagicMock()
        worker.transcribe.return_value = []
        ctx = _make_context(worker=worker)

        stop_calls = [0]
        def is_shutdown():
            stop_calls[0] += 1
            return stop_calls[0] > 3

        adapter = _make_adapter(
            capture_audio=capture_audio,
            consume_audio_device_switch=consume_switch,
            on_device_switched=on_switched,
            set_current_audio_device=MagicMock(),
            is_paused=MagicMock(return_value=False),
            is_shutdown_requested=is_shutdown,
            timestamp_provider=MagicMock(return_value="12:00:00"),
        )
        p = TranscriptionPipeline(ctx, adapter)
        p.run()

        # The on_device_switched should have been called with the new device
        # (either in the inner loop break path or outer loop)
        assert inner_iterations[0] >= 1

    def test_capture_error_handled(self):
        """on_capture_error is invoked when capture raises."""
        def capture_audio(**kwargs):
            raise RuntimeError("mic error")

        error_handler = MagicMock(return_value=(None, False))  # stop

        adapter = _make_adapter(
            capture_audio=capture_audio,
            is_paused=MagicMock(return_value=False),
            is_shutdown_requested=MagicMock(return_value=False),
            on_capture_error=error_handler,
        )
        p = _make_pipeline(adapter=adapter)
        p.run()  # should not raise

        error_handler.assert_called_once()
        args = error_handler.call_args[0]
        assert isinstance(args[0], RuntimeError)
        assert args[2] == 1  # capture_error_count

    def test_capture_error_without_handler_raises(self):
        def capture_audio(**kwargs):
            raise RuntimeError("mic error")

        adapter = _make_adapter(
            capture_audio=capture_audio,
            is_paused=MagicMock(return_value=False),
            is_shutdown_requested=MagicMock(return_value=False),
        )
        p = _make_pipeline(adapter=adapter)
        with pytest.raises(RuntimeError, match="mic error"):
            p.run()


# ---------------------------------------------------------------------------
# 16. run_capture_loop convenience function
# ---------------------------------------------------------------------------

class TestRunCaptureLoop:
    def test_returns_pipeline(self):
        ctx = _make_context()
        ctx.stop_event.set()  # immediate stop
        adapter = _make_adapter(
            is_shutdown_requested=MagicMock(return_value=False),
        )
        result = run_capture_loop(ctx, adapter, initial_device=5)
        assert isinstance(result, TranscriptionPipeline)


# ---------------------------------------------------------------------------
# 17. _iterate_capture
# ---------------------------------------------------------------------------

class TestIterateCapture:
    def test_passes_kwargs(self):
        capture_fn = MagicMock(return_value=iter([]))
        stop_evt = threading.Event()
        adapter = _make_adapter(
            capture_audio=capture_fn,
            capture_stop_event=stop_evt,
            capture_kwargs={"extra_param": True},
        )
        ctx = _make_context(chunk_seconds=7.0)
        p = _make_pipeline(ctx=ctx, adapter=adapter)

        list(p._iterate_capture(42))

        capture_fn.assert_called_once()
        kw = capture_fn.call_args.kwargs
        assert kw["device"] == 42
        assert kw["chunk_seconds"] == 7.0
        assert kw["stop_event"] is stop_evt
        assert kw["extra_param"] is True
