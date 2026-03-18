"""파인튜닝 모델 vs pretrained 모델 A/B 평가.

동일 오디오에 대해 두 모델의 CER(Character Error Rate)을 비교하고,
환각 발생 빈도, IT 용어 인식률을 측정한다.

사용법:
    # 기본 (pretrained vs fine-tuned)
    CUDA_VISIBLE_DEVICES=1 python evaluate.py \
        --baseline large-v3-turbo \
        --finetuned ./whisper-ko-sonote/ct2 \
        --eval-data ./data/eval.jsonl

    # 환각 분석만
    CUDA_VISIBLE_DEVICES=1 python evaluate.py \
        --finetuned ./whisper-ko-sonote/ct2 \
        --eval-data ./data/eval.jsonl \
        --hallucination-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# 환각 패턴 (sonote postprocess.py와 동일)
HALLUCINATION_TEXTS = {
    "감사합니다", "구독과 좋아요", "다음 영상에서 만나요",
    "자막 제공", "MBC뉴스", "시청해 주셔서 감사합니다",
    "thanks for watching", "please subscribe", "subtitles by",
    "ご視聴ありがとうございました", "ご視聴ありがとうございます",
}
JAPANESE_KANA_RE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")

# IT 용어 정답 사전 (CORRECTIONS 역방향)
IT_TERMS = {
    "파이썬", "Python", "JavaScript", "TypeScript", "React", "Vue.js",
    "Spring Boot", "Django", "FastAPI", "Docker", "Kubernetes",
    "GitHub", "GitLab", "Nginx", "Jenkins", "MongoDB", "PostgreSQL",
    "Redis", "PyTorch", "TensorFlow", "Hugging Face",
    "API", "REST", "GraphQL", "WebSocket", "JSON",
    "CI/CD", "DevOps", "프론트엔드", "백엔드",
}

# CORRECTIONS 오인식 패턴
CORRECTION_ERRORS = {
    "파이선", "깃헙", "자바스크립트", "타입스크립트", "리액트",
    "장고", "플라스크", "넥스트", "뷰제이에스", "노드",
    "엔진엑스", "젠킨스", "깃랩", "도커컴포즈",
    "몽고디비", "포스트그레스", "레디스", "파이토치", "텐서플로",
    "허깅페이스", "제이슨", "에이피아이", "그래프큐엘", "웹소켓",
    "프론트앤드", "백앤드", "데브옵스", "시아이시디",
}


def transcribe_with_model(
    model_id: str,
    audio_path: str,
    language: str = "ko",
) -> tuple[str, float]:
    """모델로 오디오 전사. (텍스트, 소요시간) 반환."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_id, device="cuda", compute_type="float16")

    start = time.time()
    segments, info = model.transcribe(audio_path, language=language)
    text = " ".join(seg.text.strip() for seg in segments)
    elapsed = time.time() - start

    del model
    return text, elapsed


def compute_cer(prediction: str, reference: str) -> float:
    """CER (Character Error Rate) 계산."""
    try:
        import jiwer
        return jiwer.cer(reference, prediction)
    except ImportError:
        # 간이 CER 계산
        ref_chars = list(reference.replace(" ", ""))
        pred_chars = list(prediction.replace(" ", ""))
        if not ref_chars:
            return 0.0 if not pred_chars else 1.0

        # 레벤슈타인 거리
        m, n = len(ref_chars), len(pred_chars)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if ref_chars[i - 1] == pred_chars[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[m][n] / m


def count_hallucinations(text: str) -> int:
    """환각 패턴 감지 횟수."""
    count = 0
    for pattern in HALLUCINATION_TEXTS:
        count += text.count(pattern)
    if JAPANESE_KANA_RE.search(text):
        count += 1
    return count


def count_it_term_accuracy(text: str) -> tuple[int, int]:
    """IT 용어 정확 인식 / 오인식 횟수."""
    correct = sum(1 for term in IT_TERMS if term in text)
    errors = sum(1 for term in CORRECTION_ERRORS if term in text)
    return correct, errors


def evaluate_pair(
    baseline_id: str,
    finetuned_id: str,
    eval_data_path: Path,
) -> dict[str, Any]:
    """두 모델 A/B 비교 평가."""
    with open(eval_data_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    if not samples:
        print("[오류] 평가 데이터가 비어있습니다.")
        sys.exit(1)

    print(f"평가 샘플: {len(samples)}개")
    print(f"Baseline: {baseline_id}")
    print(f"Fine-tuned: {finetuned_id}")
    print()

    results = {
        "baseline": {"cer": [], "hallucinations": 0, "it_correct": 0, "it_errors": 0, "time": []},
        "finetuned": {"cer": [], "hallucinations": 0, "it_correct": 0, "it_errors": 0, "time": []},
    }

    for i, sample in enumerate(samples):
        audio_path = sample.get("audio_path")
        reference = sample.get("text", "")

        if not audio_path or not Path(audio_path).exists():
            continue

        print(f"[{i+1}/{len(samples)}] {Path(audio_path).name}")

        # Baseline
        try:
            bl_text, bl_time = transcribe_with_model(baseline_id, audio_path)
            bl_cer = compute_cer(bl_text, reference)
            bl_hall = count_hallucinations(bl_text)
            bl_correct, bl_errors = count_it_term_accuracy(bl_text)

            results["baseline"]["cer"].append(bl_cer)
            results["baseline"]["hallucinations"] += bl_hall
            results["baseline"]["it_correct"] += bl_correct
            results["baseline"]["it_errors"] += bl_errors
            results["baseline"]["time"].append(bl_time)
        except Exception as e:
            print(f"  baseline 실패: {e}")
            continue

        # Fine-tuned
        try:
            ft_text, ft_time = transcribe_with_model(finetuned_id, audio_path)
            ft_cer = compute_cer(ft_text, reference)
            ft_hall = count_hallucinations(ft_text)
            ft_correct, ft_errors = count_it_term_accuracy(ft_text)

            results["finetuned"]["cer"].append(ft_cer)
            results["finetuned"]["hallucinations"] += ft_hall
            results["finetuned"]["it_correct"] += ft_correct
            results["finetuned"]["it_errors"] += ft_errors
            results["finetuned"]["time"].append(ft_time)
        except Exception as e:
            print(f"  finetuned 실패: {e}")
            continue

        # 샘플별 비교
        delta_cer = ft_cer - bl_cer
        marker = "▲" if delta_cer > 0 else ("▼" if delta_cer < 0 else "=")
        print(f"  CER: {bl_cer:.3f} → {ft_cer:.3f} ({marker}{abs(delta_cer):.3f})")

    return results


def print_report(results: dict[str, Any]) -> None:
    """평가 결과 리포트 출력."""
    bl = results["baseline"]
    ft = results["finetuned"]

    n = len(bl["cer"])
    if n == 0:
        print("[오류] 평가 결과가 없습니다.")
        return

    bl_cer_avg = np.mean(bl["cer"])
    ft_cer_avg = np.mean(ft["cer"])
    bl_time_avg = np.mean(bl["time"])
    ft_time_avg = np.mean(ft["time"])

    print(f"""
╔══════════════════════════════════════════════════╗
║          sonote Whisper A/B 평가 결과            ║
╠══════════════════════════════════════════════════╣
║                  Baseline    Fine-tuned   Delta  ║
╠──────────────────────────────────────────────────╣
║ CER (평균)     {bl_cer_avg:>8.3f}     {ft_cer_avg:>8.3f}    {ft_cer_avg - bl_cer_avg:>+.3f}  ║
║ 환각 감지       {bl['hallucinations']:>8d}     {ft['hallucinations']:>8d}    {ft['hallucinations'] - bl['hallucinations']:>+5d}  ║
║ IT 용어 정확    {bl['it_correct']:>8d}     {ft['it_correct']:>8d}    {ft['it_correct'] - bl['it_correct']:>+5d}  ║
║ IT 용어 오인식  {bl['it_errors']:>8d}     {ft['it_errors']:>8d}    {ft['it_errors'] - bl['it_errors']:>+5d}  ║
║ 추론 시간 (초)  {bl_time_avg:>8.2f}     {ft_time_avg:>8.2f}    {ft_time_avg - bl_time_avg:>+.2f}  ║
║ 평가 샘플       {n:>8d}                          ║
╚══════════════════════════════════════════════════╝
""")

    # 판정
    improvements = 0
    if ft_cer_avg < bl_cer_avg:
        improvements += 1
        print(f"  [개선] CER: {(bl_cer_avg - ft_cer_avg) / bl_cer_avg * 100:.1f}% 감소")
    if ft["hallucinations"] < bl["hallucinations"]:
        improvements += 1
        print(f"  [개선] 환각: {bl['hallucinations'] - ft['hallucinations']}건 감소")
    if ft["it_errors"] < bl["it_errors"]:
        improvements += 1
        print(f"  [개선] IT 용어 오인식: {bl['it_errors'] - ft['it_errors']}건 감소")

    if improvements >= 2:
        print("\n  결론: 파인튜닝 모델 채택 권장")
    elif improvements == 1:
        print("\n  결론: 부분 개선 — 추가 학습 데이터/에폭 고려")
    else:
        print("\n  결론: 개선 미미 — 학습 데이터 품질/양 재검토 필요")


def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper 파인튜닝 A/B 평가")
    parser.add_argument("--baseline", type=str, default="large-v3-turbo", help="베이스라인 모델 ID")
    parser.add_argument("--finetuned", type=str, required=True, help="파인튜닝 모델 경로 (CT2)")
    parser.add_argument("--eval-data", type=Path, required=True, help="평가 데이터 (eval.jsonl)")
    args = parser.parse_args()

    if not args.eval_data.exists():
        print(f"[오류] 평가 데이터 없음: {args.eval_data}")
        sys.exit(1)

    results = evaluate_pair(args.baseline, args.finetuned, args.eval_data)
    print_report(results)


if __name__ == "__main__":
    main()
