"""
연속 라이브 트랜스크립션 모듈
YouTube 라이브 스트림을 청크 단위로 지속 변환

동작:
1. ffmpeg 세그먼트 먹서로 N초 단위 WAV 청크를 자동 생성
2. 완성된 청크를 즉시 Whisper로 변환
3. 결과를 출력 파일에 점진적으로 추가
4. Ctrl+C로 깔끔하게 종료
"""
import json as _json
import subprocess
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path

from .download import get_stream_url
from .transcribe import save_transcript, detect_device, _register_cuda_exit_guard, DOMAIN_PROMPT


def continuous_live(
    video_url: str,
    output_path: str | Path,
    chunk_seconds: int = 10,
    model_id: str = "tellang/whisper-large-v3-turbo-ko",
    language: str = "ko",
    device: str | None = None,
    compute_type: str | None = None,
    fmt: str = "txt",
    format_id: str = "91",
    beam_size: int = 5,
    cookies_path: str | Path | None = None,
):
    """
    연속 라이브 트랜스크립션 실행

    Args:
        video_url: YouTube 라이브 URL
        output_path: 출력 파일 경로
        chunk_seconds: 청크 크기 (초, 기본 10)
        model_id: Whisper 모델
        language: 인식 언어
        device: cuda/cpu (None이면 자동)
        compute_type: float16/int8 (None이면 자동)
        fmt: 출력 형식 (txt/srt)
        format_id: yt-dlp 포맷 ID
        beam_size: 빔 서치 크기
        cookies_path: cookies.txt 경로 (None이면 자동 탐색)
    """
    from faster_whisper import WhisperModel

    output_path = Path(output_path)

    # 디바이스 감지
    if device is None or compute_type is None:
        auto_dev, auto_comp = detect_device()
        device = device or auto_dev
        compute_type = compute_type or auto_comp

    print("=" * 60)
    print("[연속 모드] 라이브 트랜스크립션")
    print(f"  모델: {model_id} | {device} ({compute_type})")
    print(f"  청크: {chunk_seconds}초 ({chunk_seconds // 60}분 {chunk_seconds % 60}초)")
    print(f"  출력: {output_path}")
    print("  종료: Ctrl+C")
    print("=" * 60)

    # 모델 사전 로드 (모든 청크에 재사용)
    print("\n[모델] 로딩 중...")
    model = WhisperModel(model_id, device=device, compute_type=compute_type)
    if device == "cuda":
        _register_cuda_exit_guard()
    print("[모델] 준비 완료")

    # 스트림 URL 추출
    print("[스트림] URL 추출 중...")
    stream_url = get_stream_url(video_url, format_id, cookies_path=cookies_path)
    print("[스트림] 연결 완료\n")

    # 순차 다운로드+변환 방식 (segment muxer 대신 — HLS 라이브 호환)
    chunk_dir = Path(tempfile.mkdtemp(prefix="lt-continuous-"))
    all_segments: list[dict] = []

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg가 설치되어 있지 않습니다")

    print(f"[녹음] 순차 캡처 시작 (청크당 {chunk_seconds}초)\n")

    try:
        idx = 0
        while True:
            chunk_file = chunk_dir / f"chunk_{idx:04d}.wav"

            # 1) ffmpeg로 N초 캡처
            dl_cmd = [
                ffmpeg_path,
                "-live_start_index", "-1",
                "-i", stream_url,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                "-t", str(chunk_seconds),
                str(chunk_file), "-y",
            ]
            dl_proc = subprocess.run(
                dl_cmd, capture_output=True, timeout=chunk_seconds + 30,
            )

            if not chunk_file.exists() or chunk_file.stat().st_size < 2000:
                print(f"--- 청크 #{idx}: 캡처 실패, 재시도 ---")
                time.sleep(2)
                continue

            # 2) Whisper 변환 + SSE push
            _process_chunk(
                model, chunk_file, idx, chunk_seconds,
                language, beam_size, all_segments,
            )

            # 3) 중간 저장
            if all_segments:
                save_transcript(all_segments, output_path, fmt=fmt)

            chunk_file.unlink(missing_ok=True)
            idx += 1

    except KeyboardInterrupt:
        print("\n\n[중지] Ctrl+C — 종료합니다...")

    finally:
        if all_segments:
            save_transcript(all_segments, output_path, fmt=fmt)
            print(f"\n{'=' * 60}")
            print(f"[완료] 총 {len(all_segments)}개 세그먼트 → {output_path}")
        else:
            print("\n[완료] 변환된 세그먼트 없음")

        shutil.rmtree(chunk_dir, ignore_errors=True)


def _process_chunk(
    model,
    chunk_path: Path,
    chunk_idx: int,
    chunk_seconds: int,
    language: str,
    beam_size: int,
    all_segments: list[dict],
):
    """단일 청크를 Whisper로 변환하고 all_segments에 추가"""
    print(f"--- 청크 #{chunk_idx} 변환 중 ---")

    try:
        vad_params = {"min_silence_duration_ms": 500}
        # 한국어 도메인 용어 프롬프트 적용 (리서치 v0.0.1)
        prompt = DOMAIN_PROMPT if language == "ko" else None
        segments_iter, _info = model.transcribe(
            str(chunk_path),
            language=language,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters=vad_params,
            word_timestamps=False,
            initial_prompt=prompt,
        )

        offset = chunk_idx * chunk_seconds
        count = 0

        for seg in segments_iter:
            # avg_logprob(-1.0~0.0)를 0~1 confidence로 선형 매핑하고 범위를 보정
            confidence = max(0.0, min(1.0, 1.0 + seg.avg_logprob))
            entry = {
                "start": seg.start + offset,
                "end": seg.end + offset,
                "text": seg.text.strip(),
                "confidence": confidence,
            }
            all_segments.append(entry)
            count += 1

            # 실시간 콘솔 출력
            ms, ss = divmod(int(entry["start"]), 60)
            me, se = divmod(int(entry["end"]), 60)
            print(f"  [{ms:02d}:{ss:02d} ~ {me:02d}:{se:02d}] {entry['text']}")

            # 서버 SSE로 실시간 push (서버 실행 중이면)
            try:
                payload = _json.dumps({
                    "text": entry["text"],
                    "ts": f"{ms:02d}:{ss:02d}",
                    "speaker": "화자",
                    "confidence": entry.get("confidence"),
                }).encode()
                req = urllib.request.Request(
                    "http://127.0.0.1:8000/api/push-segment",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=2)
            except Exception:
                pass  # 서버 미실행 시 무시

        print(f"--- 청크 #{chunk_idx}: {count}개 세그먼트 (누적 {len(all_segments)}개) ---\n")

    except Exception as e:
        print(f"--- 청크 #{chunk_idx}: 오류 — {e} ---\n")
