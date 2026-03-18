"""
트랜스크립트 병합 및 중복 제거 모듈
기존 스크립트와 새 스크립트를 텍스트 유사도 기반으로 병합
- 앞에 빠진 구간 (prepend)
- 뒤에 빠진 구간 (append)
- 겹치는 구간 자동 제거 (연속 매칭 기반, 오탐 방지)
"""
import re
from difflib import SequenceMatcher
from pathlib import Path


def load_transcript(path: str | Path) -> list[dict]:
    """
    [MM:SS ~ MM:SS] 형식 txt 파일 → 세그먼트 리스트 파싱

    Returns:
        [{"start": float, "end": float, "text": str}, ...]
    """
    path = Path(path)
    if not path.exists():
        return []

    segments = []
    pat = re.compile(r"\[(\d{2}):(\d{2})\s*~\s*(\d{2}):(\d{2})\]\s*(.*)")

    for line in path.read_text(encoding="utf-8").splitlines():
        m = pat.match(line.strip())
        if m:
            sm, ss, em, es, text = m.groups()
            segments.append({
                "start": int(sm) * 60 + int(ss),
                "end": int(em) * 60 + int(es),
                "text": text.strip(),
            })
    return segments


def _text_similar(a: str, b: str, threshold: float) -> bool:
    """두 텍스트의 유사도가 임계값 이상인지 판별"""
    if not a or not b:
        return False
    # 짧은 텍스트는 완전 일치에 가까워야 함
    if len(a) < 10 or len(b) < 10:
        return SequenceMatcher(None, a, b).ratio() >= max(threshold, 0.85)
    return SequenceMatcher(None, a, b).ratio() >= threshold


def _find_overlap_region(
    existing_texts: list[str],
    new_segments: list[dict],
    threshold: float,
    min_consecutive: int,
) -> tuple[int, int]:
    """
    new_segments에서 existing과 겹치는 연속 구간 찾기.

    Returns:
        (overlap_start, overlap_end) — new_segments의 인덱스 범위
        겹침 없으면 (-1, -1)
    """
    for j in range(len(new_segments)):
        consecutive = 0
        last_matched = j - 1

        for k in range(j, len(new_segments)):
            matched = any(
                _text_similar(et, new_segments[k]["text"], threshold)
                for et in existing_texts
            )
            if matched:
                consecutive += 1
                last_matched = k
            else:
                if consecutive >= min_consecutive:
                    # 갭 허용 (Whisper 세그먼트 누락 가능)
                    if k - last_matched > 2:
                        return (j, last_matched + 1)
                else:
                    # 연속 매칭 부족 — 다음 시작점으로
                    break

        if consecutive >= min_consecutive:
            return (j, last_matched + 1)

    return (-1, -1)


def merge_transcripts(
    existing: list[dict],
    new_segments: list[dict],
    threshold: float = 0.75,
    min_consecutive: int = 2,
) -> list[dict]:
    """
    기존 + 새 세그먼트를 텍스트 유사도로 병합.

    동작:
    1. new_segments에서 existing과 겹치는 연속 구간 찾기
    2. 겹침 이전 → prepend, 겹침 이후 → append
    3. 겹치는 구간은 기존 것을 유지

    Args:
        existing: 기존 세그먼트 리스트
        new_segments: 새로 변환된 세그먼트 리스트
        threshold: 텍스트 유사도 임계값 (0.75 권장)
        min_consecutive: 겹침 인정에 필요한 최소 연속 매칭 수

    Returns:
        병합된 세그먼트 리스트
    """
    if not existing:
        return new_segments
    if not new_segments:
        return existing

    existing_texts = [s["text"] for s in existing]

    ov_start, ov_end = _find_overlap_region(
        existing_texts, new_segments, threshold, min_consecutive,
    )

    # 겹침 없음
    if ov_start == -1:
        if new_segments[-1]["end"] <= existing[0]["start"]:
            print(f"[병합] 겹침 없음 — 새 {len(new_segments)}개를 앞에 추가")
            return new_segments + existing
        print(f"[병합] 겹침 없음 — 새 {len(new_segments)}개를 뒤에 추가")
        return existing + new_segments

    prepend = new_segments[:ov_start]
    append = new_segments[ov_end:]
    overlap_count = ov_end - ov_start

    merged = prepend + existing + append

    parts = []
    if prepend:
        parts.append(f"앞 +{len(prepend)}개")
    parts.append(f"기존 {len(existing)}개 유지")
    if append:
        parts.append(f"뒤 +{len(append)}개")
    if overlap_count > 0:
        parts.append(f"겹침 {overlap_count}개 제거")

    print(f"[병합] {', '.join(parts)} = 총 {len(merged)}개")
    return merged
