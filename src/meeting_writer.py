from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
import json
from pathlib import Path
from typing import Any
import wave

import numpy as np
from src.paths import meetings_dir

__all__ = ["MeetingWriter"]

_DISPLAY_PARAGRAPH_GAP_SECONDS = 1.5
_DISPLAY_PARAGRAPH_MAX_SEGMENTS = 4
_DISPLAY_PARAGRAPH_MAX_CHARS = 280


class MeetingWriter:
    """회의록 파일 저장 관리자 — 노션 양식 Markdown 출력.

    실시간 flush는 .raw.txt / .stt.jsonl에 수행하여 크래시 안전성을 확보하고,
    종료 시 최종 회의록과 세션 메타데이터를 함께 남긴다.
    """

    def __init__(self, output_path: Path | str | None = None) -> None:
        if output_path is None:
            now = datetime.now()
            date_dir = meetings_dir(now)
            session_id = now.strftime("%H%M%S")
            session_dir = date_dir / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            md_path = session_dir / "meeting.md"
        else:
            md_path = Path(output_path)
            if md_path.suffix == ".txt":
                md_path = md_path.with_suffix(".md")
            if md_path.parent != Path(""):
                md_path.parent.mkdir(parents=True, exist_ok=True)

        self._output_path: Path = md_path
        session_dir = md_path.parent
        self._raw_path: Path = session_dir / "meeting.raw.txt"
        self._alignment_path: Path = session_dir / "meeting.stt.jsonl"
        self._session_path: Path = session_dir / "session.json"
        self._profile_review_path: Path = session_dir / "meeting.profile-review.json"
        self._audio_path: Path = session_dir / "meeting.audio.wav"
        self._started_at: datetime = datetime.now()
        self._speakers: set[str] = set()
        self._segments: list[dict[str, Any]] = []
        self._artifacts: dict[str, str] = {}
        self._keywords: dict[str, Any] | None = None
        self._footer_written: bool = False

        self._raw_file = self._raw_path.open("a", encoding="utf-8")
        self._alignment_file = self._alignment_path.open("a", encoding="utf-8")
        self._audio_file = wave.open(str(self._audio_path), "wb")
        self._audio_file.setnchannels(1)
        self._audio_file.setsampwidth(2)
        self._audio_file.setframerate(16000)
        self.set_artifact("session_audio", self._audio_path)

    def append_audio(self, chunk: np.ndarray, sample_rate: int = 16000) -> None:
        """사후 화자 분리용 회의 원본 오디오를 이어 붙여 저장한다."""
        if chunk.size == 0:
            return
        if sample_rate != 16000:
            raise ValueError("MeetingWriter는 현재 16kHz 오디오만 저장합니다.")
        pcm16 = np.clip(chunk, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype("<i2")
        self._audio_file.writeframes(pcm16.tobytes())

    def write_header(self) -> None:
        started_text = self._started_at.strftime("%Y-%m-%d %H:%M:%S")
        self._raw_file.write(f"# started: {started_text}\n")
        self._raw_file.flush()

    def set_keywords(self, keywords: dict[str, Any]) -> None:
        """세션 종료 시 session.json에 포함할 키워드 스냅샷을 저장한다."""
        self._keywords = keywords

    def set_artifact(self, name: str, path: Path | str | None) -> None:
        if not path:
            return
        self._artifacts[name] = str(Path(path))

    def append_alignment(self, payload: dict[str, Any]) -> None:
        self._alignment_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._alignment_file.flush()

    def append_segment(
        self,
        speaker: str,
        text: str,
        timestamp: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        normalized_speaker = speaker.strip() if speaker.strip() else "Unknown"
        self._speakers.add(normalized_speaker)

        record: dict[str, Any] = {
            "timestamp": timestamp,
            "speaker": normalized_speaker,
            "text": text,
        }
        if metadata:
            record.update(metadata)
        self._segments.append(record)

        self._raw_file.write(f"[{timestamp}] [{normalized_speaker}] {text}\n")
        self._raw_file.flush()

        # alignment JSONL에도 기록 (세션 로드 시 구조화된 세그먼트로 사용)
        self._alignment_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._alignment_file.flush()

    def write_profile_review(self, payload: dict[str, Any]) -> Path:
        self._profile_review_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.set_artifact("profile_review", self._profile_review_path)
        return self._profile_review_path

    @staticmethod
    def _coerce_seconds(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_duration_seconds(value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)

        if not isinstance(value, str):
            return 0.0

        text = value.strip()
        if not text:
            return 0.0

        if ":" not in text:
            try:
                return float(text)
            except ValueError:
                return 0.0

        try:
            parts = [float(part) for part in text.split(":")]
        except ValueError:
            return 0.0

        seconds = 0.0
        for part in parts:
            seconds = (seconds * 60.0) + part
        return seconds

    def _normalize_speakers(self, speakers: Any) -> list[str]:
        if speakers is None:
            return sorted(self._speakers)

        if isinstance(speakers, str):
            normalized = speakers.strip()
            return [normalized] if normalized else sorted(self._speakers)

        if isinstance(speakers, (int, float)):
            return sorted(self._speakers)

        if isinstance(speakers, Iterable):
            names = sorted({str(name).strip() for name in speakers if str(name).strip()})
            if names:
                return names

        return sorted(self._speakers)

    def _group_segments_for_display(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for record in self._segments:
            text = str(record.get("text") or "").strip()
            if not text:
                continue

            speaker = str(record.get("speaker") or "Unknown")
            start = self._coerce_seconds(record.get("start"))
            end = self._coerce_seconds(record.get("end"))
            timestamp = str(record.get("timestamp") or "").strip()

            if not groups:
                groups.append(
                    {
                        "speaker": speaker,
                        "start": start,
                        "end": end,
                        "first_timestamp": timestamp,
                        "last_timestamp": timestamp,
                        "segment_count": 1,
                        "char_count": len(text),
                        "texts": [text],
                    }
                )
                continue

            current = groups[-1]
            can_merge = (
                current["speaker"] == speaker
                and current["end"] is not None
                and start is not None
                and end is not None
                and start - current["end"] <= _DISPLAY_PARAGRAPH_GAP_SECONDS
                and current["segment_count"] < _DISPLAY_PARAGRAPH_MAX_SEGMENTS
                and current["char_count"] + 1 + len(text) <= _DISPLAY_PARAGRAPH_MAX_CHARS
            )
            if not can_merge:
                groups.append(
                    {
                        "speaker": speaker,
                        "start": start,
                        "end": end,
                        "first_timestamp": timestamp,
                        "last_timestamp": timestamp,
                        "segment_count": 1,
                        "char_count": len(text),
                        "texts": [text],
                    }
                )
                continue

            current["texts"].append(text)
            current["end"] = end
            current["last_timestamp"] = timestamp
            current["segment_count"] += 1
            current["char_count"] += 1 + len(text)

        return groups

    def apply_segment_corrections(self, corrections: dict[str, str]) -> int:
        applied = 0
        for record in self._segments:
            raw_line = f"- [{record['timestamp']}] [{record['speaker']}] {record['text']}"
            corrected_line = corrections.get(raw_line)
            if not corrected_line:
                continue
            prefix = f"- [{record['timestamp']}] [{record['speaker']}] "
            if not corrected_line.startswith(prefix):
                continue
            corrected_text = corrected_line[len(prefix):].strip()
            if not corrected_text or corrected_text == record["text"]:
                continue
            record["text"] = corrected_text
            applied += 1
        return applied

    def write_session_json(
        self,
        duration: Any,
        segments_count: int,
        speakers: Any,
        keywords: dict[str, Any] | None = None,
    ) -> None:
        ended_at = datetime.now()
        speaker_list = self._normalize_speakers(speakers)
        duration_seconds = self._coerce_duration_seconds(duration)
        duration_text = duration if isinstance(duration, str) else None

        payload: dict[str, Any] = {
            "duration": duration_seconds,
            "segments": int(segments_count),
            "speakers": speaker_list,
            "started_at": self._started_at.isoformat(timespec="seconds"),
            "ended_at": ended_at.isoformat(timespec="seconds"),
            # Backward-compatible aliases for existing readers/tests.
            "segment_count": int(segments_count),
            "speaker_count": len(speaker_list),
            "output_path": str(self._output_path),
            "raw_path": str(self._raw_path),
            "alignment_path": str(self._alignment_path),
            "audio_path": str(self._audio_path),
            "artifacts": self._artifacts,
        }
        if duration_text is not None:
            payload["duration_text"] = duration_text
        kw = keywords or self._keywords
        if kw:
            payload["keywords"] = kw

        try:
            self._session_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"[경고] session.json 저장 실패: {exc}")

    def write_footer(self, duration: str, segment_count: int, speaker_count: int) -> None:
        _ = speaker_count
        computed_count = len(self._speakers)
        speaker_list = sorted(self._speakers)
        started_text = self._started_at.strftime("%Y-%m-%d %H:%M:%S")

        lines: list[str] = [
            "# 회의록",
            "",
            "# 회의 내용 요약",
            "",
            f"- 일시: {started_text}",
            f"- 경과 시간: {duration}",
            f"- 참석자: {', '.join(speaker_list)} ({computed_count}명)",
            f"- 총 세그먼트: {segment_count}개",
            "- (요약은 회의 후 별도 작성)",
            "",
            "---",
            "",
            "# To-do",
            "",
            "- [ ] (회의 후 별도 작성)",
            "",
            "---",
            "",
            "# 대화 정리",
            "",
        ]

        for group in self._group_segments_for_display():
            if group["first_timestamp"] == group["last_timestamp"]:
                time_label = group["first_timestamp"]
            else:
                time_label = f"{group['first_timestamp']} ~ {group['last_timestamp']}"
            lines.extend(
                [
                    f"### {group['speaker']} · {time_label}",
                    "",
                    " ".join(group["texts"]),
                    "",
                ]
            )

        lines.extend(
            [
                "---",
                "",
            "# Raw Data",
            "",
            ]
        )

        for record in self._segments:
            lines.append(f"- [{record['timestamp']}] [{record['speaker']}] {record['text']}")

        lines.append("")
        self._output_path.write_text("\n".join(lines), encoding="utf-8")
        self.write_session_json(duration, segment_count, speaker_list)
        self._footer_written = True

    def close(self) -> None:
        if self._audio_file:
            self._audio_file.close()
        if self._alignment_file and not self._alignment_file.closed:
            self._alignment_file.close()
        if self._raw_file and not self._raw_file.closed:
            self._raw_file.close()
        if self._footer_written and self._raw_path.exists():
            self._raw_path.unlink()

    @property
    def output_path(self) -> Path:
        return self._output_path

    @property
    def alignment_path(self) -> Path:
        return self._alignment_path

    @property
    def session_path(self) -> Path:
        return self._session_path

    @property
    def audio_path(self) -> Path:
        return self._audio_path
