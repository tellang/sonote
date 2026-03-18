---
schema: openCLAW/v1
name: sonote
version: 1.2.0b2
description: AI 에이전트를 위한 실시간 한국어 음성 전사 CLI (Whisper 기반)
---

# sonote

실시간 한국어 음성 전사 CLI. YouTube 라이브, 로컬 오디오, 마이크 입력을 Whisper로 전사하고 구조화된 JSON 출력을 제공한다.

## 실행

```bash
# 로컬 오디오 전사
sonote transcribe audio.wav --json

# YouTube 라이브 전사
sonote live https://youtube.com/watch?v=XXX --back 50 --json

# 회의 실시간 전사
sonote meeting --json

# 설정만 확인 (dry-run)
sonote meeting --dry-run --json

# 오프라인 회의록 변환
sonote meeting --file recording.wav --json
```

## 구조화 출력

모든 커맨드는 `--json` 플래그로 기계 판독 가능한 JSON을 출력한다.

### 성공
```json
{
  "status": "success",
  "command": "transcribe",
  "data": { ... }
}
```

### 실패
```json
{
  "status": "error",
  "command": "transcribe",
  "error": "오디오 파일을 찾을 수 없습니다",
  "code": "NOT_FOUND",
  "reason": "notFound"
}
```

### 스트리밍 (NDJSON)
`--ndjson` 플래그로 줄 단위 JSON 스트리밍 출력:
```
{"event":"start","command":"transcribe","audio":"audio.wav"}
{"event":"segment","start":0.0,"end":5.2,"text":"안녕하세요"}
{"event":"complete","command":"transcribe","segments":42}
```

## 종료코드

| 코드 | reason enum    | 의미 |
|------|---------------|------|
| 0    | —             | 성공 |
| 1    | runtimeError  | 런타임 오류 |
| 2    | argError      | 인수 오류 |
| 3    | modelError    | 모델 로딩/추론 오류 |
| 4    | notFound      | 파일/리소스 미발견 |
| 5    | preflightFail | 사전 점검 실패 |

## 스키마 조회

```bash
# 전체 서브커맨드 스키마 + 런타임 capabilities
sonote schema --json

# 특정 서브커맨드 스키마
sonote schema meeting --json
```

## 주요 커맨드

| 커맨드 | 설명 |
|--------|------|
| `transcribe` | 로컬 오디오 파일 전사 |
| `live` | YouTube 라이브 녹음 + 전사 |
| `meeting` | 마이크 실시간 회의 전사 (SSE + 파일) |
| `auto` | 스캔 → 다운로드 → 전사 올인원 |
| `detect` | 라이브 스트림 강의 시작 지점 탐색 |
| `map` | 라이브 스트림 전체 구간 맵핑 |
| `schema` | 서브커맨드 스키마 JSON 출력 |
| `status` | 환경 진단 (GPU/모델/외부 도구) |
| `doctor` | 종합 환경 진단 + 설치 안내 |
| `setup` | 환경 설정 + 의존성 자동 설치 |
| `enroll` | 화자 목소리 사전 등록 |
| `cookies` | Chrome 쿠키 관리 |
| `desktop` | 네이티브 데스크톱 앱 |

## 필드 선택

`--fields` 플래그로 전사 결과의 출력 필드를 선택할 수 있다 (JSON 모드에서만 유효):

```bash
sonote transcribe audio.wav --json --fields text,start,end
```

## 사전 점검 (Preflight)

모든 커맨드는 실행 전 필수 조건(ffmpeg, yt-dlp, CUDA, HF_TOKEN)을 자동 점검한다. `--dry-run`으로 실제 실행 없이 설정과 점검 결과만 확인 가능.

## 환경 변수

| 변수 | 설명 |
|------|------|
| `HF_TOKEN` | Hugging Face 토큰 (화자 분리) |
| `MEETING_API_KEY` | 회의 API 인증 키 |
| `SONOTE_COOKIES` | cookies.txt 경로 |
| `SONOTE_BETA` | 베타 모드 활성화 (1) |
| `PORT` | 서버 포트 (기본: 8000) |

## 제약 사항

- CUDA GPU 권장 (CPU 폴백 가능하지만 느림)
- YouTube 라이브 기능은 ffmpeg + yt-dlp 필수
- 화자 분리는 HF_TOKEN + pyannote-audio 필요
- Windows 전용 기능: Chrome 쿠키 추출, 시스템 트레이
