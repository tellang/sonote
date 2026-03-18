"""sonote 전사 결과 → Whisper 파인튜닝 데이터셋 변환.

sonote output/meetings/ 디렉토리의 기존 전사 결과와
외부 데이터셋(KsponSpeech 등)을 HuggingFace datasets 포맷으로 변환한다.

사용법:
    # sonote 자체 데이터만
    python prepare_data.py --sonote-dir ../output/meetings --output-dir ./data

    # KsponSpeech 추가
    python prepare_data.py --sonote-dir ../output/meetings --kspon-dir /path/to/KsponSpeech --output-dir ./data

    # 교정 사전 자동 적용
    python prepare_data.py --sonote-dir ../output/meetings --apply-corrections --output-dir ./data
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# sonote 교정 사전 재활용
CORRECTIONS = {
    "파이선": "파이썬", "깃헙": "GitHub",
    "자바스크립트": "JavaScript", "타입스크립트": "TypeScript",
    "리액트": "React", "스프링": "Spring", "도커": "Docker",
    "했읍니다": "했습니다", "됬다": "됐다",
    "장고": "Django", "플라스크": "Flask", "넥스트": "Next.js",
    "뷰제이에스": "Vue.js", "노드": "Node.js", "익스프레스": "Express",
    "스프링부트": "Spring Boot", "쿠버네티스": "Kubernetes",
    "엔진엑스": "Nginx", "아파치": "Apache", "젠킨스": "Jenkins",
    "깃랩": "GitLab", "도커컴포즈": "Docker Compose",
    "몽고디비": "MongoDB", "포스트그레스": "PostgreSQL", "레디스": "Redis",
    "파이토치": "PyTorch", "텐서플로": "TensorFlow", "허깅페이스": "Hugging Face",
    "제이슨": "JSON", "에이피아이": "API", "그래프큐엘": "GraphQL",
    "웹소켓": "WebSocket", "레스트": "REST",
    "프론트앤드": "프론트엔드", "백앤드": "백엔드",
    "데브옵스": "DevOps", "시아이시디": "CI/CD",
}

# 환각 텍스트 (학습 데이터에서 제외)
HALLUCINATION_TEXTS = {
    "감사합니다", "구독과 좋아요", "다음 영상에서 만나요",
    "자막 제공", "MBC뉴스", "시청해 주셔서 감사합니다",
    "thanks for watching", "please subscribe", "subtitles by",
    "ご視聴ありがとうございました", "ご視聴ありがとうございます",
}

JAPANESE_KANA_RE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")


def apply_corrections(text: str) -> str:
    """교정 사전 적용."""
    for wrong, right in CORRECTIONS.items():
        text = text.replace(wrong, right)
    return text


def is_valid_text(text: str) -> bool:
    """학습에 부적합한 텍스트 필터링."""
    stripped = text.strip()
    if not stripped or len(stripped) < 2:
        return False
    if stripped in HALLUCINATION_TEXTS:
        return False
    if JAPANESE_KANA_RE.search(stripped):
        return False
    return True


def parse_sonote_session(session_dir: Path) -> list[dict[str, Any]]:
    """sonote 세션 디렉토리에서 오디오+텍스트 쌍 추출.

    session.json에서 세그먼트 정보를 읽고,
    해당 오디오 파일과 매칭한다.
    """
    samples: list[dict[str, Any]] = []

    # session.json 탐색
    session_json = session_dir / "session.json"
    if not session_json.exists():
        return samples

    with open(session_json, "r", encoding="utf-8") as f:
        session = json.load(f)

    # 오디오 파일 탐색
    audio_file = None
    for ext in ["wav", "mp3", "m4a", "ogg", "webm"]:
        candidates = list(session_dir.glob(f"*.{ext}"))
        if candidates:
            audio_file = candidates[0]
            break

    if audio_file is None:
        return samples

    # 세그먼트 추출
    segments = session.get("segments", [])
    if not segments:
        # transcript.txt 폴백
        transcript_file = session_dir / "transcript.txt"
        if transcript_file.exists():
            text = transcript_file.read_text(encoding="utf-8").strip()
            if is_valid_text(text):
                samples.append({
                    "audio_path": str(audio_file),
                    "text": text,
                })
        return samples

    for seg in segments:
        text = seg.get("text", "").strip()
        if not is_valid_text(text):
            continue
        samples.append({
            "audio_path": str(audio_file),
            "text": text,
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
        })

    return samples


def collect_sonote_data(
    meetings_dir: Path,
    do_corrections: bool = False,
) -> list[dict[str, Any]]:
    """sonote output/meetings/ 전체 스캔."""
    all_samples: list[dict[str, Any]] = []

    if not meetings_dir.exists():
        print(f"[경고] 디렉토리 없음: {meetings_dir}")
        return all_samples

    # 날짜/세션 디렉토리 순회
    for date_dir in sorted(meetings_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        for session_dir in sorted(date_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            samples = parse_sonote_session(session_dir)
            all_samples.extend(samples)

    if do_corrections:
        for sample in all_samples:
            sample["text"] = apply_corrections(sample["text"])

    print(f"[sonote] {len(all_samples)}개 샘플 수집 ({meetings_dir})")
    return all_samples


def collect_kspon_data(kspon_dir: Path) -> list[dict[str, Any]]:
    """KsponSpeech 데이터셋 로드 (pcm + txt 쌍)."""
    all_samples: list[dict[str, Any]] = []

    if not kspon_dir.exists():
        print(f"[경고] KsponSpeech 디렉토리 없음: {kspon_dir}")
        return all_samples

    # KsponSpeech 구조: KsponSpeech_N/pcm/*.pcm + txt/*.txt
    for pcm_file in sorted(kspon_dir.rglob("*.pcm")):
        txt_file = pcm_file.with_suffix(".txt")
        # txt가 pcm과 같은 위치 또는 txt/ 디렉토리에 있을 수 있음
        if not txt_file.exists():
            txt_dir = pcm_file.parent.parent / "txt" / pcm_file.with_suffix(".txt").name
            if txt_dir.exists():
                txt_file = txt_dir

        if not txt_file.exists():
            continue

        text = txt_file.read_text(encoding="utf-8").strip()
        # KsponSpeech 노이즈 태그 제거
        text = re.sub(r"\([^)]*\)", "", text)  # (노이즈), (웃음) 등
        text = re.sub(r"[/+*]", "", text)  # 전사 규칙 특수문자
        text = re.sub(r"\s+", " ", text).strip()

        if not is_valid_text(text):
            continue

        all_samples.append({
            "audio_path": str(pcm_file),
            "text": text,
            "source": "kspon",
        })

    print(f"[KsponSpeech] {len(all_samples)}개 샘플 수집")
    return all_samples


def save_dataset(
    samples: list[dict[str, Any]],
    output_dir: Path,
    train_ratio: float = 0.9,
) -> None:
    """학습/검증 분할 후 JSONL로 저장."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 셔플
    import random
    random.seed(42)
    random.shuffle(samples)

    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]

    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"

    for path, data in [(train_path, train_samples), (eval_path, eval_samples)]:
        with open(path, "w", encoding="utf-8") as f:
            for sample in data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n=== 데이터셋 저장 완료 ===")
    print(f"학습: {train_path} ({len(train_samples)}개)")
    print(f"검증: {eval_path} ({len(eval_samples)}개)")
    print(f"총: {len(samples)}개")

    # 통계
    texts = [s["text"] for s in samples]
    avg_len = sum(len(t) for t in texts) / len(texts) if texts else 0
    print(f"평균 텍스트 길이: {avg_len:.0f}자")


def main() -> None:
    parser = argparse.ArgumentParser(description="sonote Whisper 파인튜닝 데이터 준비")
    parser.add_argument(
        "--sonote-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "output" / "meetings",
        help="sonote output/meetings 경로",
    )
    parser.add_argument("--kspon-dir", type=Path, default=None, help="KsponSpeech 경로")
    parser.add_argument("--output-dir", type=Path, default=Path("./data"), help="출력 디렉토리")
    parser.add_argument("--apply-corrections", action="store_true", help="교정 사전 자동 적용")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="학습/검증 분할 비율")
    args = parser.parse_args()

    all_samples: list[dict[str, Any]] = []

    # sonote 데이터 수집
    sonote_samples = collect_sonote_data(args.sonote_dir, args.apply_corrections)
    all_samples.extend(sonote_samples)

    # KsponSpeech 데이터 수집 (선택)
    if args.kspon_dir:
        kspon_samples = collect_kspon_data(args.kspon_dir)
        all_samples.extend(kspon_samples)

    if not all_samples:
        print("[오류] 수집된 샘플이 없습니다.")
        print("  sonote 전사 결과가 output/meetings/에 있는지 확인하세요.")
        sys.exit(1)

    save_dataset(all_samples, args.output_dir, args.train_ratio)


if __name__ == "__main__":
    main()
