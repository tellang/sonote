#!/usr/bin/env bash
# sonote Whisper 파인튜닝 — GPU 서버 환경 구축
# 실행: bash setup_env.sh
# 대상: SSAFY GPU 서버 (70.12.130.110, Device1)

set -euo pipefail

ENV_NAME="sonote-ft"
PYTHON_VER="3.11"

echo "=== sonote Whisper 파인튜닝 환경 구축 ==="
echo "GPU 서버: $(hostname)"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'N/A')"

# 1. conda 환경 생성
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[스킵] conda 환경 '${ENV_NAME}' 이미 존재"
else
    echo "[1/5] conda 환경 생성: ${ENV_NAME} (Python ${PYTHON_VER})"
    conda create -n "${ENV_NAME}" python="${PYTHON_VER}" -y
fi

# conda activate를 스크립트 내에서 사용하기 위한 초기화
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# 2. PyTorch + CUDA 12.8
echo "[2/5] PyTorch + CUDA 12.8 설치"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. 학습 프레임워크
echo "[3/5] Transformers + PEFT + datasets 설치"
pip install \
    transformers>=4.40.0 \
    peft>=0.10.0 \
    datasets>=2.18.0 \
    accelerate>=0.29.0 \
    evaluate>=0.4.0 \
    jiwer>=3.0.0 \
    soundfile>=0.12.0 \
    librosa>=0.10.0

# 4. CTranslate2 (faster-whisper 변환용)
echo "[4/5] CTranslate2 설치"
pip install ctranslate2>=4.0.0

# 5. Jupyter 커널 등록
echo "[5/5] Jupyter 커널 등록"
pip install ipykernel
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "sonote-ft (Python ${PYTHON_VER})"

# 6. GPU 검증
echo ""
echo "=== 환경 검증 ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')

import transformers
print(f'Transformers: {transformers.__version__}')

import peft
print(f'PEFT: {peft.__version__}')

import ctranslate2
print(f'CTranslate2: {ctranslate2.__version__}')
"

echo ""
echo "=== 완료 ==="
echo "활성화: conda activate ${ENV_NAME}"
echo "Device1 사용: CUDA_VISIBLE_DEVICES=1 python train_whisper_lora.py"
