# media-transcriber

Korean Speech-to-Text (STT) tool for live streams, meetings, and local audio files.

스트림/회의/로컬 오디오 전사를 위한 한국어 음성 인식(STT) 도구.

## 주요 기능

| 기능 | 설명 |
|------|------|
| **YouTube 라이브 STT** | DVR 기록에서 오디오 추출 → 한국어 텍스트 변환 |
| **스마트 올인원** | BGM/강의 자동 분류 → 음성 블록만 병렬 다운로드 → 변환 → 병합 |
| **연속 실시간 변환** | 라이브 스트림을 청크 단위로 실시간 변환 (Ctrl+C 종료) |
| **트랜스크립트 병합** | 텍스트 유사도 기반 오버랩 자동 제거 |
| **회의 실시간 전사** | 마이크 → 화자 분리 → SSE 자막 + 파일 저장 |
| **후처리 파이프라인** | 타임스탬프 제거, 파편 병합, 구조화 Markdown 문서 생성 |

## 기술 스택

- **STT**: faster-whisper (CTranslate2, large-v3-turbo)
- **GPU**: CUDA float16 (RTX 4070 최적화)
- **오디오**: yt-dlp, ffmpeg, sounddevice
- **웹**: FastAPI + SSE
- **화자 분리**: pyannote-audio (선택)

## 설치

```bash
# PyPI (출시 예정)
pip install media-transcriber

# uv (권장)
uv sync

# 화자 분리 포함
pip install -e ".[diarize]"
```

> ffmpeg 필수: `choco install ffmpeg` (Windows) 또는 `brew install ffmpeg` (macOS)

## CLI 사용법

```bash
media-transcriber <command> [options]
# 또는
python -m src.cli <command>
```

### 로컬 오디오 변환

```bash
# 기본 변환
media-transcriber transcribe audio.wav

# 긴 파일 청크 분할
media-transcriber transcribe long.wav --chunk-minutes 10

# SRT 자막 출력
media-transcriber transcribe audio.wav --fmt srt
```

### YouTube 라이브

```bash
# 올인원 (스캔 → 병렬 다운 → 변환 → 병합)
media-transcriber smart "https://youtube.com/watch?v=XXX"

# 기존 스크립트에 이어붙이기
media-transcriber smart "URL" --resume transcript.txt

# BGM 경계 탐색만
media-transcriber probe "URL"

# 전체 구간 맵핑
media-transcriber scan "URL" --max-back 180 --step 5

# 단순 녹음 + 변환
media-transcriber live "URL" --back 50

# 연속 실시간 변환
media-transcriber live "URL" --continuous --chunk-size 120

# 오디오만 다운로드
media-transcriber download "URL"
```

### 회의 실시간 전사

```bash
# 기본 실행 (마이크 → SSE 자막 + 파일 저장)
media-transcriber meeting

# 화자 분리 비활성화
media-transcriber meeting --no-diarize

# 포트/출력 지정
media-transcriber meeting --port 8080 -o standup.txt

# 마이크 장치 선택
media-transcriber meeting --list-devices
media-transcriber meeting --device 1
```

보안/인증 참고:

- 기본 바인딩은 `127.0.0.1`입니다.
- `MEETING_API_KEY`를 설정하면 보호 엔드포인트 인증이 활성화됩니다.
- 화자 분리 토큰은 `HF_TOKEN` 환경변수로 전달합니다.

## 프로젝트 구조

```
src/
├── cli.py              # CLI 진입점
├── paths.py            # 출력 디렉토리 경로 관리 (유형+날짜 기반)
├── download.py         # YouTube 오디오 다운로드 (yt-dlp + ffmpeg + 병렬 분할)
├── transcribe.py       # Whisper STT (faster-whisper, CUDA/CPU, 청크 분할)
├── probe.py            # BGM↔강의 경계 탐색 (silencedetect + binary search)
├── continuous.py       # 연속 실시간 변환 (ffmpeg 세그먼트 + Whisper)
├── merge.py            # 트랜스크립트 병합 (텍스트 유사도 기반)
├── audio_capture.py    # 마이크 캡처 (sounddevice, 링버퍼)
├── diarize.py          # 화자 분리 (pyannote-audio, 선택적)
├── whisper_worker.py   # CUDA 격리 STT 워커 (multiprocessing)
├── postprocess.py      # 후처리 (필러 제거, 파편 병합)
├── meeting_writer.py   # 회의록 파일 저장
├── polish.py           # LLM 후처리 (Codex STT 교정 + Gemini 요약)
├── server.py           # FastAPI SSE 서버
├── tray.py             # Windows 시스템 트레이
├── db.py               # 영상 프로필 DB (SQLite)
└── __init__.py
static/
└── viewer.html         # 자막 뷰어 (단일 HTML)
output/                 # 출력 (유형+날짜 기반 자동 분류)
├── data/               # 공유 데이터 (profiles.db, speakers.json)
├── meetings/YYYY-MM-DD/  # 회의록 (.md, .raw.txt)
├── transcripts/YYYY-MM-DD/  # 전사 결과
└── audio/YYYY-MM-DD/   # WAV 오디오
```

## 상세 문서

- **[기술 가이드](docs/GUIDE.md)** — 설치, CLI 레퍼런스, 피쳐별 워크플로우, 트러블슈팅, 아키텍처
- **[프로젝트 인덱스](PROJECT_INDEX.md)** — 디렉토리 구조, 의존성 그래프, 설계 특징

## 모델 정보

| 모델 | 한국어 CER | 속도 | 비고 |
|------|-----------|------|------|
| large-v3-turbo **(기본)** | ~16% | 빠름 | CTranslate2 내장 |
| large-v3 | ~11% | 보통 | 제로샷 기준 |
| 한국어 fine-tuned Whisper | ~7.5% | 보통 | CTranslate2 변환 필요 |
| ENERZAi Whisper (Small) | ~6.45% | 빠름 | 1.58-bit 양자화, MIT |
| Conformer-RNNT (NeMo) | ~8% | 보통 | 스트리밍 지원, Apache 2.0 |

> 업계 비교 및 개선 로드맵: [docs/MEETING_STT_RESEARCH_v0.0.1.md](docs/MEETING_STT_RESEARCH_v0.0.1.md)

## License
MIT — see [LICENSE](LICENSE)
