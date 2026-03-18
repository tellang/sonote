#!/usr/bin/env python3
"""sonote Whisper 파인튜닝 — 올인원 실행 스크립트.

GPU 서버에서 단독 실행 가능. 데이터 다운로드 → 전처리 → 학습 → 변환 → 검증까지 전부 수행.
공개 한국어 음성 데이터셋(zeroth-korean)을 HuggingFace에서 직접 다운로드.

사용법 (GPU 서버 터미널에서):
    CUDA_VISIBLE_DEVICES=1 python run_all.py
    CUDA_VISIBLE_DEVICES=1 python run_all.py --epochs 5 --batch-size 32
    CUDA_VISIBLE_DEVICES=1 python run_all.py --skip-setup  # 환경 이미 구축 시
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# === 설정 ===
WORK_DIR = Path.home() / "sonote-finetune"
DATA_DIR = WORK_DIR / "data"
OUTPUT_DIR = WORK_DIR / "whisper-ko-sonote"
MODEL_ID = "openai/whisper-large-v3-turbo"
LANGUAGE = "ko"

# HuggingFace 공개 한국어 음성 데이터셋 (등록 불필요)
HF_DATASET = "google/fleurs"  # 다국어 음성, 한국어(ko_kr) ~12시간
HF_SUBSET = "ko_kr"


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """셸 명령 실행."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if check and result.returncode != 0:
        print(f"  [실패] exit code: {result.returncode}")
        sys.exit(1)
    return result


def phase_0_setup() -> None:
    """Phase 0: 패키지 설치."""
    print("\n" + "=" * 60)
    print("Phase 0: 환경 설정")
    print("=" * 60)

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "datasets>=2.18.0,<3.0.0",
        "accelerate>=0.29.0",
        "evaluate>=0.4.0",
        "jiwer>=3.0.0",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        "ctranslate2>=4.0.0",
    ]

    for pkg in packages:
        run(f"pip install -q {pkg}", check=False)

    # 검증
    import torch
    print(f"\n  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1024**3
        print(f"  VRAM: {vram:.0f} GB")


def phase_1_data() -> None:
    """Phase 1: 데이터 다운로드 + 전처리."""
    print("\n" + "=" * 60)
    print("Phase 1: 데이터 준비")
    print("=" * 60)

    from datasets import load_dataset, Audio

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_jsonl = DATA_DIR / "train.jsonl"
    eval_jsonl = DATA_DIR / "eval.jsonl"

    if train_jsonl.exists() and eval_jsonl.exists():
        with open(train_jsonl) as f:
            n_train = sum(1 for _ in f)
        with open(eval_jsonl) as f:
            n_eval = sum(1 for _ in f)
        print(f"  [스킵] 데이터 이미 존재 (학습: {n_train}, 검증: {n_eval})")
        return

    print(f"  데이터셋 다운로드: {HF_DATASET} ({HF_SUBSET})")
    ds = load_dataset(HF_DATASET, HF_SUBSET, trust_remote_code=True)

    # FLEURS: train / validation / test 분할 사용
    train_ds = ds.get("train")
    test_ds = ds.get("test", ds.get("validation"))

    if train_ds is None:
        full_ds = ds[list(ds.keys())[0]]
        split = full_ds.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"]
        test_ds = split["test"]

    print(f"  학습: {len(train_ds)}개")
    print(f"  검증: {len(test_ds)}개")

    # JSONL로 저장 (오디오 경로 + 텍스트)
    def save_split(dataset, output_path: Path) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            for i, sample in enumerate(dataset):
                # 오디오 필드 추출 (FLEURS: "audio" 키)
                audio = sample.get("audio", {})
                # FLEURS: "transcription" 키, 일반: "text" 또는 "sentence"
                text = sample.get("transcription", sample.get("text", sample.get("sentence", ""))).strip()

                if not text or len(text) < 2:
                    continue

                # 오디오 배열을 wav로 저장
                audio_dir = DATA_DIR / "audio"
                audio_dir.mkdir(exist_ok=True)
                audio_path = audio_dir / f"{output_path.stem}_{i:06d}.wav"

                if isinstance(audio, dict) and "array" in audio:
                    import soundfile as sf
                    sf.write(
                        str(audio_path),
                        audio["array"],
                        audio.get("sampling_rate", 16000),
                    )
                elif isinstance(audio, dict) and "path" in audio and audio["path"]:
                    audio_path = Path(audio["path"])
                else:
                    continue

                record = {"audio_path": str(audio_path), "text": text}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("  학습 데이터 저장 중...")
    save_split(train_ds, train_jsonl)
    print("  검증 데이터 저장 중...")
    save_split(test_ds, eval_jsonl)

    with open(train_jsonl) as f:
        n = sum(1 for _ in f)
    print(f"  완료: {n}개 학습 샘플")


def phase_2_train(epochs: int = 10, batch_size: int = 16, lora_rank: int = 32) -> None:
    """Phase 2: Whisper LoRA 파인튜닝."""
    print("\n" + "=" * 60)
    print("Phase 2: 학습")
    print("=" * 60)

    import evaluate as eval_lib
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
    from peft import LoraConfig, TaskType, get_peft_model

    # 프로세서
    print("  프로세서 로드...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_ID, language=LANGUAGE, task="transcribe")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task="transcribe")

    # 데이터 로드
    print("  데이터 로드...")
    train_jsonl = DATA_DIR / "train.jsonl"
    eval_jsonl = DATA_DIR / "eval.jsonl"

    def load_jsonl(path: Path) -> Dataset:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        ds = Dataset.from_dict({
            "audio": [r["audio_path"] for r in records],
            "text": [r["text"] for r in records],
        })
        return ds.cast_column("audio", Audio(sampling_rate=16000))

    train_ds = load_jsonl(train_jsonl)
    eval_ds = load_jsonl(eval_jsonl) if eval_jsonl.exists() else None

    print(f"  학습: {len(train_ds)}개")
    if eval_ds:
        print(f"  검증: {len(eval_ds)}개")

    # 전처리
    def preprocess(batch):
        audio = batch["audio"]
        inputs = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"],
        ).input_features[0]

        # 30초 초과 필터
        if len(audio["array"]) / audio["sampling_rate"] > 30:
            return {"input_features": None, "labels": None}

        labels = tokenizer(batch["text"]).input_ids
        return {"input_features": inputs, "labels": labels}

    print("  전처리...")
    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    train_ds = train_ds.filter(lambda x: x["input_features"] is not None)

    if eval_ds:
        eval_ds = eval_ds.map(preprocess, remove_columns=eval_ds.column_names)
        eval_ds = eval_ds.filter(lambda x: x["input_features"] is not None)

    print(f"  전처리 후: {len(train_ds)}개 학습")

    # 모델 + LoRA
    print("  모델 로드 + LoRA...")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.generation_config.language = LANGUAGE
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Data Collator
    class DataCollator:
        def __init__(self, proc, start_id):
            self.proc = proc
            self.start_id = start_id

        def __call__(self, features):
            features = [f for f in features if f.get("input_features") is not None]
            input_feats = [{"input_features": f["input_features"]} for f in features]
            batch = self.proc.feature_extractor.pad(input_feats, return_tensors="pt")

            label_feats = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.proc.tokenizer.pad(label_feats, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100,
            )
            if (labels[:, 0] == self.start_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    collator = DataCollator(processor, model.config.decoder_start_token_id)

    # CER 메트릭
    cer_metric = eval_lib.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"cer": cer_metric.compute(predictions=pred_str, references=label_str)}

    # 학습
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_steps=500,
        num_train_epochs=epochs,
        fp16=True,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=500 if eval_ds else None,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        logging_steps=25,
        load_best_model_at_end=True if eval_ds else False,
        metric_for_best_model="cer" if eval_ds else None,
        greater_is_better=False if eval_ds else None,
        predict_with_generate=True,
        generation_max_length=225,
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics if eval_ds else None,
        processing_class=processor.feature_extractor,
    )

    print(f"\n  학습 시작: {epochs}에폭, 배치 {batch_size}×2")
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    print(f"\n  학습 완료: {elapsed / 60:.1f}분")

    # 저장
    merged_dir = OUTPUT_DIR / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    processor.save_pretrained(str(merged_dir))
    print(f"  병합 모델 저장: {merged_dir}")


def phase_3_convert() -> None:
    """Phase 3: CTranslate2 변환."""
    print("\n" + "=" * 60)
    print("Phase 3: CTranslate2 변환")
    print("=" * 60)

    merged_dir = OUTPUT_DIR / "merged"
    ct2_dir = OUTPUT_DIR / "ct2"

    if ct2_dir.exists():
        print(f"  [스킵] 이미 존재: {ct2_dir}")
        return

    cmd = (
        f"ct2-transformers-converter"
        f" --model {merged_dir}"
        f" --output_dir {ct2_dir}"
        f" --quantization float16"
        f" --copy_files tokenizer.json preprocessor_config.json"
    )
    run(cmd)

    model_bin = ct2_dir / "model.bin"
    if model_bin.exists():
        size_mb = model_bin.stat().st_size / 1024 / 1024
        print(f"  model.bin: {size_mb:.0f} MB")
    print(f"  변환 완료: {ct2_dir}")


def phase_4_verify() -> None:
    """Phase 4: faster-whisper 로드 검증."""
    print("\n" + "=" * 60)
    print("Phase 4: 검증")
    print("=" * 60)

    ct2_dir = OUTPUT_DIR / "ct2"
    if not ct2_dir.exists():
        print("  [스킵] CT2 모델 없음")
        return

    try:
        from faster_whisper import WhisperModel
        import numpy as np

        print("  faster-whisper 로드 테스트...")
        model = WhisperModel(str(ct2_dir), device="cuda", compute_type="float16")

        # 더미 오디오 추론
        dummy = np.random.randn(16000 * 5).astype(np.float32) * 0.01
        segments, info = model.transcribe(dummy, language="ko")
        seg_list = list(segments)
        print(f"  추론 테스트 통과 (더미 5초, {len(seg_list)}개 세그먼트)")

        del model
        print(f"\n  === 파인튜닝 완료 ===")
        print(f"  모델 경로: {ct2_dir}")
        print(f"  sonote에서 사용: sonote live --model {ct2_dir}")

    except ImportError:
        print("  [스킵] faster-whisper 미설치 — 로컬에서 검증 필요")
    except Exception as e:
        print(f"  [경고] 검증 실패: {e}")
        print(f"  모델 경로: {ct2_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="sonote Whisper 파인튜닝 올인원")
    parser.add_argument("--skip-setup", action="store_true", help="Phase 0 건너뛰기")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--only", type=str, default=None, help="특정 Phase만 실행 (0,1,2,3,4)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════╗")
    print("║  sonote Whisper 파인튜닝 — 올인원    ║")
    print("╚══════════════════════════════════════╝")
    print(f"작업 디렉토리: {WORK_DIR}")
    print(f"모델: {MODEL_ID}")
    print(f"데이터: {HF_DATASET}")
    print(f"LoRA rank: {args.lora_rank}, 에폭: {args.epochs}, 배치: {args.batch_size}")

    phases = {
        "0": phase_0_setup,
        "1": phase_1_data,
        "2": lambda: phase_2_train(args.epochs, args.batch_size, args.lora_rank),
        "3": phase_3_convert,
        "4": phase_4_verify,
    }

    if args.only:
        for p in args.only.split(","):
            phases[p.strip()]()
        return

    if not args.skip_setup:
        phase_0_setup()
    phase_1_data()
    phase_2_train(args.epochs, args.batch_size, args.lora_rank)
    phase_3_convert()
    phase_4_verify()


if __name__ == "__main__":
    main()
