from __future__ import annotations

from dataclasses import dataclass
import re

__all__ = [
    "Segment",
    "FILLER_WORDS",
    "HALLUCINATION_TEXTS",
    "CORRECTIONS",
    "remove_fillers",
    "merge_fragments",
    "remove_stutters",
    "add_punctuation",
    "clean_ellipsis",
    "is_hallucination",
    "is_valid_segment",
    "is_looping",
    "remove_overlap",
    "remove_phrase_repeats",
    "correct",
    "normalize_feedback_text",
    "normalize_live_text",
    "postprocess",
]


@dataclass
class Segment:
    """전사 세그먼트"""

    speaker: str
    text: str
    start: float  # 초 단위
    end: float


# 필러 단어 목록
FILLER_WORDS = {"네", "어", "음", "아", "그", "에", "저", "뭐", "이제"}
_ENDING_PUNCTUATION = {".", "!", "?", "…"}
_WORD_REPEAT_PATTERN = re.compile(r"\b(\S+)(?:\s+\1\b)+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MULTISPACE_PATTERN = re.compile(r"\s{2,}")

# 한국어 환각 패턴 사전 — 무음 구간에서 Whisper가 반복 생성하는 패턴
HALLUCINATION_TEXTS = {
    "감사합니다", "구독과 좋아요", "다음 영상에서 만나요",
    "자막 제공", "MBC뉴스", "시청해 주셔서 감사합니다",
    "thanks for watching", "please subscribe", "subtitles by",
}

# 도메인 교정 사전 (리서치 v0.0.1 섹션 5-3 기반 확장)
CORRECTIONS = {
    # 원본 10개
    "파이선": "파이썬", "깃헙": "GitHub",
    "자바스크립트": "JavaScript", "타입스크립트": "TypeScript",
    "리액트": "React", "스프링": "Spring", "도커": "Docker",
    "했읍니다": "했습니다", "됬다": "됐다",
    # 프레임워크/라이브러리
    "장고": "Django", "플라스크": "Flask", "넥스트": "Next.js",
    "뷰제이에스": "Vue.js", "노드": "Node.js", "익스프레스": "Express",
    "스프링부트": "Spring Boot", "쿠버네티스": "Kubernetes",
    # 인프라/도구
    "엔진엑스": "Nginx", "아파치": "Apache", "젠킨스": "Jenkins",
    "깃랩": "GitLab", "도커컴포즈": "Docker Compose",
    # 데이터/AI
    "몽고디비": "MongoDB", "포스트그레스": "PostgreSQL", "레디스": "Redis",
    "파이토치": "PyTorch", "텐서플로": "TensorFlow", "허깅페이스": "Hugging Face",
    # API/프로토콜
    "제이슨": "JSON", "에이피아이": "API", "그래프큐엘": "GraphQL",
    "웹소켓": "WebSocket", "레스트": "REST",
    # 일반 IT 용어
    "프론트앤드": "프론트엔드", "백앤드": "백엔드",
    "데브옵스": "DevOps", "시아이시디": "CI/CD",
}


def remove_phrase_repeats(text: str) -> str:
    """구/절 단위 연속 반복 제거.

    '그 크림. 그 크림. 그 크림.' → '그 크림.'
    '일단. X. X. X. 약간.' → '일단. X. 약간.'
    """
    stripped = text.strip()
    if not stripped:
        return ""

    parts = _SENTENCE_SPLIT.split(stripped)
    if len(parts) <= 1:
        return stripped

    deduped = [parts[0]]
    for part in parts[1:]:
        if part.rstrip(".!? ") == deduped[-1].rstrip(".!? "):
            continue
        deduped.append(part)
    return " ".join(deduped)


def remove_fillers(text: str) -> str:
    """필러 단어 제거. 단독으로 나온 필러만 제거 (문맥 내 사용은 유지).
    예: "네" -> "" (단독), "네 알겠습니다" -> "네 알겠습니다" (문맥)
    """
    stripped = text.strip()
    if not stripped:
        return ""

    # 구두점만 제거한 뒤 단일 필러인지 확인
    normalized = re.sub(r"[^\w가-힣]+", "", stripped)
    if normalized in FILLER_WORDS:
        return ""
    return stripped


def merge_fragments(segments: list[Segment], gap_threshold: float = 3.0) -> list[Segment]:
    """같은 화자의 연속 세그먼트 중 gap_threshold 미만인 것을 하나로 합침."""
    if not segments:
        return []

    merged: list[Segment] = [Segment(**segments[0].__dict__)]
    for current in segments[1:]:
        previous = merged[-1]
        gap = current.start - previous.end
        if current.speaker == previous.speaker and gap < gap_threshold:
            joined_text = " ".join(part for part in [previous.text, current.text] if part).strip()
            previous.text = joined_text
            previous.end = current.end
            continue

        merged.append(Segment(**current.__dict__))
    return merged


def remove_stutters(text: str) -> str:
    """반복/더듬기 제거. "그 그 그래서" -> "그래서", "그래서 그래서" -> "그래서"
    정규식: 연속 동일 어절 감지 후 하나만 유지.
    """
    stripped = text.strip()
    if not stripped:
        return ""

    collapsed = stripped
    while True:
        updated = _WORD_REPEAT_PATTERN.sub(r"\1", collapsed)
        if updated == collapsed:
            break
        collapsed = updated

    tokens = collapsed.split()
    # "그 그래서"처럼 도치된 더듬기 패턴을 보정
    while len(tokens) >= 2 and tokens[0] in FILLER_WORDS and tokens[1].startswith(tokens[0]):
        tokens.pop(0)

    return " ".join(tokens).strip()


def add_punctuation(segments: list[Segment], silence_threshold: float = 3.0) -> list[Segment]:
    """침묵 기반 마침표 삽입. 다음 세그먼트와의 gap이 threshold 이상이면 마침표 추가."""
    if not segments:
        return []

    punctuated = [Segment(**segment.__dict__) for segment in segments]
    for index, segment in enumerate(punctuated[:-1]):
        next_segment = punctuated[index + 1]
        gap = next_segment.start - segment.end
        if gap >= silence_threshold and segment.text and segment.text[-1] not in _ENDING_PUNCTUATION:
            segment.text = f"{segment.text}."
    return punctuated


def is_hallucination(text: str) -> bool:
    """한국어 환각 패턴 감지"""
    stripped = text.strip()
    return stripped in HALLUCINATION_TEXTS or len(stripped) < 2


def is_valid_segment(seg) -> bool:
    """faster-whisper 세그먼트 메타데이터 기반 환각 필터.
    seg는 faster-whisper Segment 객체 또는 동일 키를 가진 dict를 지원한다.
    """
    if isinstance(seg, dict):
        no_speech_prob = seg["no_speech_prob"]
        avg_logprob = seg["avg_logprob"]
        compression_ratio = seg["compression_ratio"]
    else:
        no_speech_prob = seg.no_speech_prob
        avg_logprob = seg.avg_logprob
        compression_ratio = seg.compression_ratio

    if no_speech_prob > 0.6 and avg_logprob < -1.0:
        return False
    if compression_ratio > 2.4:
        return False
    if avg_logprob < -1.5:
        return False
    return True


def is_looping(text: str, phrase_len: int = 3, threshold: int = 3) -> bool:
    """반복 루핑 감지 — 동일 구절이 threshold회 이상 반복되면 True"""
    words = text.split()
    if len(words) < phrase_len * threshold:
        return False
    phrase = tuple(words[:phrase_len])
    count, i = 1, phrase_len
    while i + phrase_len <= len(words):
        if tuple(words[i:i + phrase_len]) == phrase:
            count += 1
            i += phrase_len
            if count >= threshold:
                return True
        else:
            break
    return False


def remove_overlap(prev_text: str, curr_text: str, max_words: int = 10) -> str:
    """이전 청크 끝과 현재 청크 앞의 단어 중복 제거"""
    prev_words = prev_text.split()
    curr_words = curr_text.split()
    for n in range(min(max_words, len(prev_words), len(curr_words)), 0, -1):
        if prev_words[-n:] == curr_words[:n]:
            return " ".join(curr_words[n:])
    return curr_text


def clean_ellipsis(text: str) -> str:
    """과도한 말줄임표(...) 정리.

    Whisper가 한국어 발화 pause마다 삽입하는 ... 을 정리:
    - 연속 점/띄어쓴 점 (. . . / ....) → ...
    - 문장 중간 ... (뒤에 텍스트 이어짐) → 공백
    - 문장 끝 ... → .
    """
    # 연속 점/띄어쓴 점 정규화: . . . / .... 이상 → ...
    text = re.sub(r"(?:\s*\.\s*){3,}", "...", text)
    # 문장 중간 ...: 뒤에 한글/영어/숫자가 이어지면 공백으로 교체
    text = re.sub(r"\.\.\.\s*(?=[가-힣a-zA-Z0-9])", " ", text)
    # 문장 끝 ... 은 마침표 하나로 축약해 프롬프트 토큰 낭비를 줄인다.
    text = re.sub(r"\.\.\.$", ".", text)
    # 정리 후 이중 공백 제거
    return _MULTISPACE_PATTERN.sub(" ", text).strip()


def normalize_live_text(text: str) -> str:
    """실시간 루프에서 쓰는 경량 정규화.

    후처리 전 단계에서 반복/군더더기/말줄임표를 먼저 줄여
    rolling prompt와 키워드 추출이 노이즈를 덜 먹게 한다.
    """
    return clean_ellipsis(remove_phrase_repeats(remove_stutters(remove_fillers(text))))


def normalize_feedback_text(text: str) -> str:
    """프롬프트/키워드 피드백용 추가 정규화."""
    normalized = normalize_live_text(text).replace("...", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def correct(text: str) -> str:
    """도메인 교정 사전 적용"""
    for wrong, right in CORRECTIONS.items():
        text = text.replace(wrong, right)
    return text


def _deduplicate_segments(segments: list[Segment]) -> list[Segment]:
    """연속 동일 텍스트 세그먼트 제거 (화자 무관)."""
    if not segments:
        return []

    result = [segments[0]]
    for seg in segments[1:]:
        if seg.text.strip() == result[-1].text.strip():
            continue
        result.append(seg)
    return result


def postprocess(segments: list[Segment]) -> list[Segment]:
    """전체 후처리 파이프라인: remove_fillers -> remove_stutters -> remove_phrase_repeats -> correct -> 환각/루핑 필터 -> merge -> punctuation -> dedup"""
    cleaned: list[Segment] = []
    for segment in segments:
        text = correct(normalize_live_text(segment.text))
        if not text:
            continue
        if is_hallucination(text):
            continue
        if is_looping(text):
            continue
        cleaned.append(
            Segment(
                speaker=segment.speaker,
                text=text,
                start=segment.start,
                end=segment.end,
            )
        )

    merged = merge_fragments(cleaned)
    # 병합 후 재결합된 구절 반복도 제거
    for seg in merged:
        seg.text = remove_phrase_repeats(seg.text)
    merged = [seg for seg in merged if seg.text]
    punctuated = add_punctuation(merged)
    return _deduplicate_segments(punctuated)
