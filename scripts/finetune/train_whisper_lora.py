"""Whisper large-v3-turbo LoRA 파인튜닝 — 한국어 도메인 특화.

H200 NVL (141GB VRAM) 기준 최적화.
Device1 고정: CUDA_VISIBLE_DEVICES=1 python train_whisper_lora.py

사용법:
    # 기본 학습 (LoRA rank 32, 10 에폭)
    CUDA_VISIBLE_DEVICES=1 python train_whisper_lora.py --data-dir ./data

    # 배치 크기/에폭 조정
    CUDA_VISIBLE_DEVICES=1 python train_whisper_lora.py --data-dir ./data --batch-size 32 --epochs 20

    # Full fine-tuning (LoRA 없이)
    CUDA_VISIBLE_DEVICES=1 python train_whisper_lora.py --data-dir ./data --no-lora

    # WandB 모니터링
    CUDA_VISIBLE_DEVICES=1 python train_whisper_lora.py --data-dir ./data --wandb-project sonote-ft
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import evaluate
import numpy as np
import torch
from datasets import Audio, Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)


# --- 설정 ---
MODEL_ID = "openai/whisper-large-v3-turbo"
LANGUAGE = "ko"
TASK = "transcribe"
SAMPLING_RATE = 16000


@dataclass
class TrainConfig:
    """학습 설정."""

    data_dir: Path = field(default_factory=lambda: Path("./data"))
    output_dir: Path = field(default_factory=lambda: Path("./whisper-ko-sonote"))
    model_id: str = MODEL_ID
    batch_size: int = 16
    grad_accum: int = 2
    epochs: int = 10
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    lora_rank: int = 32
    lora_alpha: int = 64
    use_lora: bool = True
    fp16: bool = True
    wandb_project: str | None = None
    max_input_length: int = 30  # 초 — 30초 이상 오디오 필터
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 25


def load_jsonl_dataset(jsonl_path: Path) -> Dataset:
    """JSONL 파일에서 HuggingFace Dataset 생성."""
    records: list[dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # audio_path + text 구조
    audio_paths = [r["audio_path"] for r in records]
    texts = [r["text"] for r in records]

    ds = Dataset.from_dict({"audio": audio_paths, "text": texts})
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    return ds


def prepare_dataset(
    batch: dict[str, Any],
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer,
    max_input_length: int,
) -> dict[str, Any]:
    """배치 전처리 — 오디오 → mel spectrogram, 텍스트 → 토큰."""
    audio = batch["audio"]

    # 오디오 → mel 스펙트로그램
    input_features = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]

    # 길이 필터 (max_input_length 초과 시 제거)
    input_length = len(audio["array"]) / audio["sampling_rate"]
    if input_length > max_input_length:
        return {"input_features": None, "labels": None}

    # 텍스트 → 토큰
    labels = tokenizer(batch["text"]).input_ids

    return {
        "input_features": input_features,
        "labels": labels,
    }


@dataclass
class DataCollator:
    """Whisper 학습용 데이터 콜레이터."""

    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # None 필터 (길이 초과 등)
        features = [f for f in features if f.get("input_features") is not None]
        if not features:
            raise ValueError("유효한 샘플이 없습니다.")

        # input_features 패딩
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # labels 패딩
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # -100으로 패딩 토큰 마스킹 (loss 계산 제외)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # decoder_start_token이 첫 토큰이면 제거
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def compute_metrics(
    pred: Any,
    tokenizer: WhisperTokenizer,
    cer_metric: Any,
) -> dict[str, float]:
    """CER (Character Error Rate) 계산."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # -100 → pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper LoRA 파인튜닝")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./whisper-ko-sonote"))
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--no-lora", action="store_true", help="Full fine-tuning (LoRA 비활성화)")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--max-input-length", type=int, default=30)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        use_lora=not args.no_lora,
        wandb_project=args.wandb_project,
        max_input_length=args.max_input_length,
    )

    print("=== sonote Whisper 파인튜닝 ===")
    print(f"모델: {cfg.model_id}")
    print(f"LoRA: {'rank ' + str(cfg.lora_rank) if cfg.use_lora else '비활성 (Full FT)'}")
    print(f"배치: {cfg.batch_size} × {cfg.grad_accum} (effective: {cfg.batch_size * cfg.grad_accum})")
    print(f"에폭: {cfg.epochs}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 1. 프로세서 로드
    print("\n[1/6] 프로세서 로드...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.model_id)
    tokenizer = WhisperTokenizer.from_pretrained(cfg.model_id, language=LANGUAGE, task=TASK)
    processor = WhisperProcessor.from_pretrained(cfg.model_id, language=LANGUAGE, task=TASK)

    # 2. 데이터 로드
    print("[2/6] 데이터 로드...")
    train_path = cfg.data_dir / "train.jsonl"
    eval_path = cfg.data_dir / "eval.jsonl"

    if not train_path.exists():
        print(f"[오류] 학습 데이터 없음: {train_path}")
        print("먼저 prepare_data.py를 실행하세요.")
        sys.exit(1)

    train_dataset = load_jsonl_dataset(train_path)
    eval_dataset = load_jsonl_dataset(eval_path) if eval_path.exists() else None

    print(f"  학습: {len(train_dataset)}개")
    if eval_dataset:
        print(f"  검증: {len(eval_dataset)}개")

    # 3. 전처리
    print("[3/6] 데이터 전처리...")

    def preprocess(batch: dict) -> dict:
        return prepare_dataset(batch, feature_extractor, tokenizer, cfg.max_input_length)

    train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.filter(lambda x: x["input_features"] is not None)

    if eval_dataset:
        eval_dataset = eval_dataset.map(preprocess, remove_columns=eval_dataset.column_names)
        eval_dataset = eval_dataset.filter(lambda x: x["input_features"] is not None)

    print(f"  전처리 후 학습: {len(train_dataset)}개")

    # 4. 모델 로드
    print("[4/6] 모델 로드...")
    model = WhisperForConditionalGeneration.from_pretrained(cfg.model_id)
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None

    # LoRA 적용
    if cfg.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # 5. 학습 설정
    print("[5/6] 학습 시작...")
    data_collator = DataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    cer_metric = evaluate.load("cer")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        num_train_epochs=cfg.epochs,
        fp16=cfg.fp16,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=cfg.eval_steps if eval_dataset else None,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=3,
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="cer" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
        predict_with_generate=True,
        generation_max_length=225,
        report_to="wandb" if cfg.wandb_project else "none",
        run_name=f"sonote-whisper-lora-r{cfg.lora_rank}" if cfg.wandb_project else None,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    if cfg.wandb_project:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer, cer_metric),
        processing_class=processor.feature_extractor,
    )

    trainer.train()

    # 6. 저장
    print("[6/6] 모델 저장...")
    final_dir = cfg.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    if cfg.use_lora:
        # LoRA 어댑터만 저장
        model.save_pretrained(str(final_dir))
        # 병합된 전체 모델도 저장 (CTranslate2 변환용)
        merged_dir = cfg.output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        processor.save_pretrained(str(merged_dir))
        print(f"  LoRA 어댑터: {final_dir}")
        print(f"  병합 모델: {merged_dir}")
    else:
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        processor.save_pretrained(str(final_dir))
        print(f"  모델: {final_dir}")

    print("\n=== 학습 완료 ===")
    print(f"다음 단계: python convert_ct2.py --model-dir {cfg.output_dir / 'merged' if cfg.use_lora else final_dir}")


if __name__ == "__main__":
    main()
