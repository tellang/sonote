"""Shared real-time transcription pipeline for CLI/Desktop."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import partial
import sys
import threading
import time
from typing import Any

import numpy as np

from ..postprocess import (
    Segment,
    is_hallucination,
    is_looping,
    is_valid_segment,
    normalize_feedback_text,
    postprocess,
    remove_overlap,
)
from .context import PipelineContext

CaptureAudioFn = Callable[..., Iterable[np.ndarray]]
TranscribeRunner = Callable[[Any, np.ndarray, dict[str, Any]], list[dict[str, Any]]]

_DEFAULT_VAD_PARAMETERS = {
    "threshold": 0.45,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 400,
}

_DEFAULT_TRANSCRIBE_KWARGS: dict[str, Any] = {
    "beam_size": 5,
    "temperature": (0.0, 0.2, 0.4, 0.6),
    "vad_filter": True,
    "vad_parameters": _DEFAULT_VAD_PARAMETERS,
    "hallucination_silence_threshold": 2.0,
    "compression_ratio_threshold": 2.4,
    "no_speech_threshold": 0.45,
    "log_prob_threshold": -1.0,
    "condition_on_previous_text": True,
    "word_timestamps": True,
}


@dataclass(slots=True)
class PipelineAdapter:
    """Host-specific adapter hooks for CLI/Desktop behavior differences."""

    capture_audio: CaptureAudioFn
    is_paused: Callable[[], bool]
    is_shutdown_requested: Callable[[], bool]
    consume_audio_device_switch: Callable[[], tuple[bool, int | None]] | None = None
    set_current_audio_device: Callable[[int | None], None] | None = None
    on_device_switched: Callable[[int | None, int | None], None] | None = None
    capture_stop_event: threading.Event | None = None
    capture_kwargs: dict[str, Any] = field(default_factory=dict)
    consume_session_rotate: Callable[[], bool] | None = None
    on_session_rotate: Callable[[], None] | None = None
    skip_chunk_on_rotate: bool = False
    reset_runtime_on_rotate: bool = True
    on_pause: Callable[[], None] | None = None
    preprocess_chunk: Callable[[np.ndarray], np.ndarray | None] | None = None
    transcribe_kwargs_factory: Callable[[np.ndarray], dict[str, Any]] | None = None
    transcribe_runner: TranscribeRunner | None = None
    timestamp_provider: Callable[[], str] | None = None
    segment_time_mapper: Callable[[dict[str, Any]], tuple[float, float]] | None = None
    on_raw_segment: Callable[[dict[str, Any]], None] | None = None
    on_segment_emitted: Callable[[Segment, str, str, int], None] | None = None
    on_chunk_processed: Callable[[list[Segment], str, int], None] | None = None
    dedupe_recent_texts: bool = False
    recent_text_limit: int = 50
    submit_correction_batch: Callable[[list[str], int], Any] | None = None
    on_correction_future: Callable[[Any, list[str]], None] | None = None
    correction_batch_size: int = 10
    correction_line_formatter: Callable[[str, str, str], str] | None = None
    submit_keyword_job: Callable[[str], None] | None = None
    on_keyword_submitted: Callable[[str, int], None] | None = None
    keyword_every_segments: int = 10
    keyword_window: int = 10
    on_capture_error: Callable[[Exception, int | None, int], tuple[int | None, bool]] | None = None
    diarize_error_label: str = "pipeline"


class TranscriptionPipeline:
    """Reusable streaming transcription loop for CLI and Desktop."""

    def __init__(self, context: PipelineContext, adapter: PipelineAdapter) -> None:
        self._context = context
        self._adapter = adapter
        self._prev_chunk_text = ""
        self._diarize_warned = False
        self._segment_count = 0
        self._uncorrected_buffer: list[str] = []
        self._keyword_extract_at = 0
        self._recent_feedback_texts: list[str] = []
        self._correction_batch_index = 0

    @property
    def segment_count(self) -> int:
        return self._segment_count

    @property
    def recent_feedback_texts(self) -> list[str]:
        return list(self._recent_feedback_texts)

    def run(self, *, initial_device: int | None = None) -> None:
        """Run capture -> transcription -> postprocess loop."""
        active_device = initial_device
        capture_error_count = 0
        _no_switch = object()

        while not self._should_stop():
            has_switch, next_device = self._consume_device_switch()
            if has_switch and next_device is not None:
                previous_device = active_device
                active_device = next_device
                if self._adapter.on_device_switched is not None:
                    self._adapter.on_device_switched(previous_device, active_device)
                self._set_current_device(active_device)

            if self._consume_session_rotate():
                self._handle_session_rotate()

            requested_device: object | int | None = _no_switch
            try:
                for audio_chunk in self._iterate_capture(active_device):
                    if self._should_stop():
                        return

                    has_switch, next_device = self._consume_device_switch()
                    if has_switch and next_device != active_device:
                        requested_device = next_device
                        break
                    if has_switch:
                        self._set_current_device(active_device)

                    if self._consume_session_rotate():
                        self._handle_session_rotate()
                        if self._adapter.skip_chunk_on_rotate:
                            continue

                    if self._adapter.is_paused():
                        if self._adapter.on_pause is not None:
                            self._adapter.on_pause()
                        continue

                    chunk = audio_chunk
                    if self._adapter.preprocess_chunk is not None:
                        chunk = self._adapter.preprocess_chunk(audio_chunk)
                        if chunk is None:
                            continue

                    stt_segments = self._transcribe(chunk)

                    if self._should_stop():
                        return

                    speaker_segments = self._collect_speaker_segments(chunk)
                    timestamp = self._current_timestamp()
                    raw_segments = self._build_raw_segments(
                        stt_segments=stt_segments,
                        speaker_segments=speaker_segments,
                        audio_chunk=chunk,
                        timestamp=timestamp,
                    )
                    if not raw_segments:
                        continue

                    processed = postprocess(raw_segments)
                    if not processed:
                        self._prev_chunk_text = ""
                        continue

                    chunk_text = self._emit_segments(processed, timestamp)
                    self._prev_chunk_text = chunk_text

                    if self._adapter.on_chunk_processed is not None:
                        self._adapter.on_chunk_processed(processed, chunk_text, self._segment_count)

                    self._submit_pending_corrections()
                    self._submit_keywords()

                capture_error_count = 0
            except Exception as exc:
                capture_error_count += 1
                if self._adapter.on_capture_error is None:
                    raise

                next_active, should_continue = self._adapter.on_capture_error(
                    exc,
                    active_device,
                    capture_error_count,
                )
                active_device = next_active
                if should_continue:
                    continue
                return

            if requested_device is not _no_switch and requested_device is not None:
                previous_device = active_device
                active_device = requested_device
                if self._adapter.on_device_switched is not None:
                    self._adapter.on_device_switched(previous_device, active_device)
                self._set_current_device(active_device)

    def _iterate_capture(self, active_device: int | None) -> Iterable[np.ndarray]:
        capture_kwargs = dict(self._adapter.capture_kwargs)
        capture_kwargs.setdefault("chunk_seconds", self._context.chunk_seconds)
        capture_kwargs["device"] = active_device

        if self._adapter.capture_stop_event is not None:
            capture_kwargs["stop_event"] = self._adapter.capture_stop_event

        if self._adapter.set_current_audio_device is not None:
            capture_kwargs["on_stream_started"] = (
                lambda d=active_device: self._adapter.set_current_audio_device(d)
            )

        return self._adapter.capture_audio(**capture_kwargs)

    def _transcribe(self, chunk: np.ndarray) -> list[dict[str, Any]]:
        kwargs = dict(_DEFAULT_TRANSCRIBE_KWARGS)
        kwargs["language"] = self._context.language
        if self._adapter.transcribe_kwargs_factory is not None:
            kwargs.update(self._adapter.transcribe_kwargs_factory(chunk))

        runner = self._adapter.transcribe_runner
        if runner is not None:
            return runner(self._context.worker, chunk, kwargs)
        return self._context.worker.transcribe(chunk, **kwargs)

    def _collect_speaker_segments(self, chunk: np.ndarray) -> list[dict[str, Any]]:
        diarizer = self._context.diarizer
        if diarizer is None:
            return []

        try:
            return diarizer.identify_speakers_in_chunk(chunk)
        except Exception as exc:
            if not self._diarize_warned:
                print(
                    f"[{self._adapter.diarize_error_label}][화자 분리] 세그먼테이션 실패: {exc}",
                    file=sys.stderr,
                )
                self._diarize_warned = True
            return []

    def _build_raw_segments(
        self,
        *,
        stt_segments: list[dict[str, Any]],
        speaker_segments: list[dict[str, Any]],
        audio_chunk: np.ndarray,
        timestamp: str,
    ) -> list[Segment]:
        raw_segments: list[Segment] = []
        for seg in stt_segments:
            if not is_valid_segment(seg):
                continue

            raw_text = (seg.get("text") or "").strip()
            if not raw_text:
                continue
            if is_hallucination(raw_text, language=self._context.language):
                continue
            if is_looping(raw_text):
                continue

            text = raw_text
            if self._prev_chunk_text:
                text = remove_overlap(self._prev_chunk_text, text)
                if not text:
                    continue

            start, end = self._segment_bounds(seg)
            speaker = self._resolve_speaker(
                seg=seg,
                speaker_segments=speaker_segments,
                audio_chunk=audio_chunk,
            )

            feedback_text = normalize_feedback_text(text)
            if self._adapter.on_raw_segment is not None:
                self._adapter.on_raw_segment(
                    {
                        "kind": "stt_segment",
                        "timestamp": timestamp,
                        "speaker": speaker,
                        "raw_text": raw_text,
                        "text": text,
                        "feedback_text": feedback_text,
                        "start": start,
                        "end": end,
                        "avg_logprob": seg.get("avg_logprob"),
                        "no_speech_prob": seg.get("no_speech_prob"),
                        "compression_ratio": seg.get("compression_ratio"),
                        "words": seg.get("words") or [],
                    }
                )

            avg_logprob = seg.get("avg_logprob")
            confidence = (
                round(max(0.0, min(1.0, 1.0 + avg_logprob)), 3)
                if avg_logprob is not None
                else None
            )

            raw_segments.append(
                Segment(
                    speaker=speaker,
                    text=text,
                    start=start,
                    end=end,
                    confidence=confidence,
                )
            )
        return raw_segments

    def _resolve_speaker(
        self,
        *,
        seg: dict[str, Any],
        speaker_segments: list[dict[str, Any]],
        audio_chunk: np.ndarray,
    ) -> str:
        diarizer = self._context.diarizer
        if speaker_segments:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))
            best_speaker = "화자"
            best_overlap = 0.0
            for spk_seg in speaker_segments:
                overlap = max(
                    0.0,
                    min(seg_end, float(spk_seg.get("end", 0.0)))
                    - max(seg_start, float(spk_seg.get("start", 0.0))),
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = str(spk_seg.get("speaker", "화자"))
            return best_speaker

        if diarizer is not None:
            try:
                return str(diarizer.identify_speaker(audio_chunk))
            except Exception:
                return "?"
        return "화자"

    def _segment_bounds(self, seg: dict[str, Any]) -> tuple[float, float]:
        if self._adapter.segment_time_mapper is not None:
            return self._adapter.segment_time_mapper(seg)
        return float(seg.get("start", 0.0)), float(seg.get("end", 0.0))

    def _emit_segments(self, processed: list[Segment], timestamp: str) -> str:
        chunk_feedback_texts = [normalize_feedback_text(seg.text) for seg in processed]
        chunk_text = " ".join(text for text in chunk_feedback_texts if text)

        for segment, stripped in zip(processed, chunk_feedback_texts):
            if not stripped:
                continue
            if self._adapter.dedupe_recent_texts and stripped in self._recent_feedback_texts:
                continue

            payload = {
                "speaker": segment.speaker,
                "text": segment.text,
                "timestamp": timestamp,
                "start": segment.start,
                "end": segment.end,
                "feedback_text": stripped,
            }
            if segment.confidence is not None:
                payload["confidence"] = segment.confidence
            self._context.on_transcript(payload)
            self._segment_count += 1

            if self._adapter.on_segment_emitted is not None:
                self._adapter.on_segment_emitted(segment, timestamp, stripped, self._segment_count)

            self._recent_feedback_texts.append(stripped)
            if len(self._recent_feedback_texts) > max(1, self._adapter.recent_text_limit):
                self._recent_feedback_texts.pop(0)

            if self._adapter.submit_correction_batch is not None:
                self._uncorrected_buffer.append(
                    self._format_correction_line(timestamp, segment.speaker, segment.text)
                )

        return chunk_text

    def _submit_pending_corrections(self) -> None:
        submit_fn = self._adapter.submit_correction_batch
        if submit_fn is None:
            return
        if len(self._uncorrected_buffer) < max(1, self._adapter.correction_batch_size):
            return

        batch = self._uncorrected_buffer[:]
        start_idx = self._segment_count - len(batch)
        self._uncorrected_buffer.clear()

        future = submit_fn(batch, self._correction_batch_index)
        self._correction_batch_index += 1
        if future is None:
            return

        if self._context.on_correction is not None:
            future.add_done_callback(
                partial(self._emit_corrections, batch=batch, start_idx=start_idx)
            )
        if self._adapter.on_correction_future is not None:
            self._adapter.on_correction_future(future, batch)

    def _emit_corrections(self, future: Any, *, batch: list[str], start_idx: int) -> None:
        on_correction = self._context.on_correction
        if on_correction is None:
            return
        try:
            _, ok, corrected = future.result(timeout=0)
            if not ok:
                return
            corrections: list[dict[str, Any]] = []
            for idx, (original_line, corrected_line) in enumerate(zip(batch, corrected)):
                if original_line == corrected_line:
                    continue
                corrections.append(
                    {
                        "index": start_idx + idx,
                        "original": original_line,
                        "corrected": corrected_line,
                    }
                )
            if corrections:
                on_correction(corrections)
        except Exception:
            return

    def _submit_keywords(self) -> None:
        submit_fn = self._adapter.submit_keyword_job
        if submit_fn is None:
            return

        if self._segment_count < self._keyword_extract_at + max(1, self._adapter.keyword_every_segments):
            return

        self._keyword_extract_at = self._segment_count
        window = max(1, self._adapter.keyword_window)
        kw_text = " ".join(self._recent_feedback_texts[-window:])
        if kw_text.strip():
            try:
                submit_fn(kw_text)
            except Exception as exc:
                print(f"[pipeline][keyword] submit 실패: {exc}", file=sys.stderr)
                return
            if self._adapter.on_keyword_submitted is not None:
                try:
                    self._adapter.on_keyword_submitted(kw_text, self._segment_count)
                except Exception as exc:
                    print(f"[pipeline][keyword] callback 실패: {exc}", file=sys.stderr)

    def _format_correction_line(self, timestamp: str, speaker: str, text: str) -> str:
        formatter = self._adapter.correction_line_formatter
        if formatter is not None:
            return formatter(timestamp, speaker, text)
        return f"- [{timestamp}] [{speaker}] {text}"

    def _handle_session_rotate(self) -> None:
        if self._adapter.on_session_rotate is not None:
            self._adapter.on_session_rotate()
        if self._adapter.reset_runtime_on_rotate:
            self._reset_runtime_state()

    def _reset_runtime_state(self) -> None:
        self._prev_chunk_text = ""
        self._segment_count = 0
        self._uncorrected_buffer.clear()
        self._keyword_extract_at = 0
        self._recent_feedback_texts.clear()
        self._correction_batch_index = 0

    def _current_timestamp(self) -> str:
        if self._adapter.timestamp_provider is not None:
            return self._adapter.timestamp_provider()
        return time.strftime("%H:%M:%S")

    def _consume_device_switch(self) -> tuple[bool, int | None]:
        fn = self._adapter.consume_audio_device_switch
        if fn is None:
            return False, None
        return fn()

    def _set_current_device(self, device: int | None) -> None:
        if self._adapter.set_current_audio_device is not None:
            self._adapter.set_current_audio_device(device)

    def _consume_session_rotate(self) -> bool:
        fn = self._adapter.consume_session_rotate
        if fn is None:
            return False
        return bool(fn())

    def _should_stop(self) -> bool:
        return self._context.stop_event.is_set() or self._adapter.is_shutdown_requested()


def run_capture_loop(
    context: PipelineContext,
    adapter: PipelineAdapter,
    *,
    initial_device: int | None = None,
) -> TranscriptionPipeline:
    """Execute the shared capture loop and return pipeline runtime state."""
    pipeline = TranscriptionPipeline(context, adapter)
    pipeline.run(initial_device=initial_device)
    return pipeline
