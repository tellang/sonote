"""Shared meeting transcription runtime context."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import threading
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..diarize import SpeakerDiarizer
    from ..whisper_worker import WhisperWorkerPool


TranscriptPayload = dict[str, Any]
CorrectionPayload = list[dict[str, Any]]


@dataclass(slots=True)
class PipelineContext:
    """Runtime dependency bundle for meeting transcription."""

    worker: WhisperWorkerPool
    diarizer: SpeakerDiarizer | None
    language: str
    chunk_seconds: float
    on_transcript: Callable[[TranscriptPayload], None]
    on_correction: Callable[[CorrectionPayload], None] | None
    stop_event: threading.Event
