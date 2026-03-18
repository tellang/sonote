# sonote Whisper 파인튜닝 파이프라인

Whisper large-v3-turbo를 한국어 도메인(IT/개발 회의)에 특화 파인튜닝한다.

## 대상 GPU

SSAFY H200 NVL (141GB VRAM) — Device1 고정

## 실행 순서

```bash
# GPU 서버 (70.12.130.110) 접속 후

# 0. 이 디렉토리를 GPU 서버로 전송
git clone <repo> && cd sonote/scripts/finetune

# 1. 환경 구축
bash setup_env.sh
conda activate sonote-ft

# 2. 데이터 준비
python prepare_data.py --sonote-dir ../../output/meetings --apply-corrections --output-dir ./data

# 3. 학습 (LoRA, ~2-4시간)
CUDA_VISIBLE_DEVICES=1 python train_whisper_lora.py --data-dir ./data

# 4. CTranslate2 변환
python convert_ct2.py --model-dir ./whisper-ko-sonote/merged

# 5. 평가
CUDA_VISIBLE_DEVICES=1 python evaluate.py \
    --baseline large-v3-turbo \
    --finetuned ./whisper-ko-sonote/ct2 \
    --eval-data ./data/eval.jsonl
```

## 파일 구조

| 파일 | 역할 |
|------|------|
| `setup_env.sh` | conda 환경 + PyTorch/PEFT/CTranslate2 설치 |
| `prepare_data.py` | sonote 전사 결과 → HF datasets JSONL |
| `train_whisper_lora.py` | Whisper LoRA 파인튜닝 (PEFT) |
| `convert_ct2.py` | HF → CTranslate2 변환 (faster-whisper 호환) |
| `evaluate.py` | Baseline vs Fine-tuned A/B 비교 |

## 주요 옵션

### 학습

```bash
# 배치 크기 키우기 (H200 VRAM 여유)
CUDA_VISIBLE_DEVICES=1 python train_whisper_lora.py --batch-size 32

# Full fine-tuning (LoRA 없이)
CUDA_VISIBLE_DEVICES=1 python train_whisper_lora.py --no-lora

# WandB 모니터링
CUDA_VISIBLE_DEVICES=1 python train_whisper_lora.py --wandb-project sonote-ft
```

### 데이터

```bash
# KsponSpeech 추가
python prepare_data.py \
    --sonote-dir ../../output/meetings \
    --kspon-dir /path/to/KsponSpeech \
    --apply-corrections \
    --output-dir ./data
```

### 변환

```bash
# int8 양자화 (크기 절반, 추론 약간 빠름)
python convert_ct2.py --model-dir ./whisper-ko-sonote/merged --quantization int8
```

## sonote 통합

변환 완료 후 sonote에서 파인튜닝 모델 사용:

```bash
# CLI
sonote live --model ./whisper-ko-sonote/ct2

# 또는 환경변수
export SONOTE_MODEL=./whisper-ko-sonote/ct2
sonote live
```
