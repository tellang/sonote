"""
YouTube 라이브 스트림 오디오 다운로드 모듈
DVR 지원으로 과거 시점부터 오디오 추출 가능
긴 블록은 내부적으로 분할 → 병렬 다운로드 → 결합
"""
import subprocess
import shutil
from pathlib import Path

# 블록 내부 분할 다운로드 설정
SPLIT_THRESHOLD_MIN = 20  # 이보다 긴 블록만 분할
SUB_CHUNK_MIN = 15        # 서브청크 크기 (분)
BOUNDARY_MARGIN_MIN = 5   # 마지막 청크 경계 여유 (분)


def get_stream_url(video_url: str, format_id: str = "91") -> str:
    """yt-dlp로 HLS 스트림 URL 추출"""
    result = subprocess.run(
        ["yt-dlp", "--js-runtimes", "node", "--cookies-from-browser", "chrome", "-f", format_id, "-g", video_url],
        capture_output=True,
        text=True,
    )
    for line in result.stdout.strip().split("\n"):
        if line.startswith("http"):
            return line.strip()
    raise RuntimeError(f"스트림 URL 추출 실패: {result.stderr[:300]}")


def list_formats(video_url: str) -> str:
    """사용 가능한 포맷 목록 조회"""
    result = subprocess.run(
        ["yt-dlp", "--js-runtimes", "node", "--cookies-from-browser", "chrome", "--list-formats", video_url],
        capture_output=True,
        text=True,
    )
    return result.stdout


def _start_ffmpeg(
    ffmpeg_path: str,
    stream_url: str,
    minutes_back: int,
    duration_minutes: int,
    output_path: Path,
    sample_rate: int = 16000,
) -> subprocess.Popen:
    """ffmpeg 프로세스 시작 (공통 로직)"""
    cmd = [ffmpeg_path]
    if minutes_back > 0:
        cmd.extend(["-live_start_index", str(-minutes_back * 60)])
    cmd.extend([
        "-i", stream_url,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        "-t", str(duration_minutes * 60),
        str(output_path), "-y",
    ])
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _make_sub_chunks(
    block_from: int,
    block_to: int,
    chunk_min: int = SUB_CHUNK_MIN,
) -> list[tuple[int, int]]:
    """
    블록을 서브청크로 분할.

    Args:
        block_from: 시작 (minutes_back, 큰 수 = 더 과거)
        block_to: 끝 (minutes_back, 작은 수 = 더 최근)
        chunk_min: 서브청크 크기(분)

    Returns:
        [(back_minutes, duration_minutes), ...] — 시간순 (과거→최근)
    """
    chunks = []
    pos = block_from
    while pos > block_to:
        remaining = pos - block_to
        dur = min(chunk_min, remaining)
        # 마지막 청크에 경계 여유분 추가
        if dur == remaining:
            dur += BOUNDARY_MARGIN_MIN
        chunks.append((pos, dur))
        pos -= min(chunk_min, remaining)
    return chunks


def _concat_wav_files(
    parts: list[Path],
    output: Path,
    ffmpeg_path: str,
) -> bool:
    """여러 WAV 파일을 하나로 결합 (ffmpeg concat demuxer)."""
    if len(parts) == 0:
        return False
    if len(parts) == 1:
        parts[0].rename(output)
        return True

    # concat 리스트 파일 생성 (ffmpeg용)
    list_file = output.parent / f".concat_{output.stem}.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for part in parts:
            # ffmpeg concat은 forward slash 필요 (Windows 호환)
            f.write(f"file '{part.resolve().as_posix()}'\n")

    cmd = [
        ffmpeg_path,
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",  # PCM이므로 재인코딩 불필요
        str(output), "-y",
    ]
    proc = subprocess.run(cmd, capture_output=True)

    # 임시 파일 정리
    list_file.unlink(missing_ok=True)
    for part in parts:
        part.unlink(missing_ok=True)

    return proc.returncode == 0 and output.exists()


def download_live_audio(
    video_url: str,
    output_path: str | Path,
    minutes_back: int = 0,
    duration_minutes: int = 50,
    format_id: str = "91",
    sample_rate: int = 16000,
) -> Path:
    """
    YouTube 라이브 스트림에서 오디오 다운로드

    Args:
        video_url: YouTube URL
        output_path: 출력 WAV 파일 경로
        minutes_back: 현재로부터 N분 전부터 시작 (0이면 현재 시점)
        duration_minutes: 녹음할 분 수
        format_id: yt-dlp 포맷 ID (91=저화질+오디오)
        sample_rate: 출력 샘플레이트 (16000=Whisper 최적)

    Returns:
        출력 파일 Path
    """
    output_path = Path(output_path)
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg가 설치되어 있지 않습니다")

    stream_url = get_stream_url(video_url, format_id)

    proc = _start_ffmpeg(
        ffmpeg_path, stream_url, minutes_back, duration_minutes, output_path, sample_rate,
    )
    _, stderr = proc.communicate()

    if proc.returncode != 0 and not output_path.exists():
        raise RuntimeError(f"ffmpeg 실패: {stderr.decode('utf-8', errors='replace')[-500:]}")

    return output_path


def download_speech_blocks(
    video_url: str,
    blocks: list[dict],
    output_dir: str | Path,
    format_id: str = "91",
    sample_rate: int = 16000,
    chunk_minutes: int = SUB_CHUNK_MIN,
    split_threshold: int = SPLIT_THRESHOLD_MIN,
) -> list[Path]:
    """
    스캔 결과의 음성 블록들을 병렬로 다운로드.
    긴 블록은 내부적으로 분할 → 병렬 다운로드 → 결합.

    Args:
        video_url: YouTube URL
        blocks: scan_stream 결과의 speech 블록 리스트
                [{"type": "speech", "from": 180, "to": 90}, ...]
                from/to는 minutes_back (from > to, from이 더 과거)
        output_dir: 출력 디렉토리
        format_id: yt-dlp 포맷 ID
        sample_rate: 샘플레이트
        chunk_minutes: 블록 내부 분할 단위 (분)
        split_threshold: 이 값(분)보다 긴 블록만 분할

    Returns:
        다운로드된 WAV 파일 경로 리스트 (시간순)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg가 설치되어 있지 않습니다")

    print("[다운로드] 스트림 URL 추출 중...")
    stream_url = get_stream_url(video_url, format_id)

    # 음성 블록만 필터 + 시간순 정렬 (과거→최근)
    speech_blocks = [b for b in blocks if b["type"] == "speech"]
    speech_blocks.sort(key=lambda b: -b["from"])

    if not speech_blocks:
        print("[다운로드] 음성 블록 없음")
        return []

    # 다운로드 계획 수립 + ffmpeg 프로세스 일괄 시작
    # jobs: [(proc, out_path, block_idx, sub_idx)]
    # plan: {block_idx: {"final": Path, "parts": [Path], "split": bool}}
    jobs: list[tuple[subprocess.Popen, Path, int, int]] = []
    plan: dict[int, dict] = {}

    total_procs = 0
    print(f"[다운로드] {len(speech_blocks)}개 음성 블록 분석")

    for i, block in enumerate(speech_blocks):
        back = block["from"]
        block_duration = block["from"] - block["to"]
        final_path = output_dir / f"block_{i:02d}_{back}min.wav"

        if block_duration > split_threshold:
            # 큰 블록: 서브청크로 분할
            sub_chunks = _make_sub_chunks(back, block["to"], chunk_minutes)
            parts = []
            print(
                f"  블록 {i}: {back}~{block['to']}분 전 ({block_duration}분)"
                f" -> {len(sub_chunks)}개 청크 병렬"
            )

            for j, (sub_back, sub_dur) in enumerate(sub_chunks):
                part_path = output_dir / f".block_{i:02d}_p{j:02d}.wav"
                proc = _start_ffmpeg(
                    ffmpeg_path, stream_url, sub_back, sub_dur, part_path, sample_rate,
                )
                jobs.append((proc, part_path, i, j))
                parts.append(part_path)

            plan[i] = {"final": final_path, "parts": parts, "split": True}
            total_procs += len(sub_chunks)
        else:
            # 작은 블록: 단일 다운로드 (경계 여유 포함)
            duration_min = block_duration + BOUNDARY_MARGIN_MIN
            print(f"  블록 {i}: {back}~{block['to']}분 전 ({block_duration}분) -> {final_path.name}")

            proc = _start_ffmpeg(
                ffmpeg_path, stream_url, back, duration_min, final_path, sample_rate,
            )
            jobs.append((proc, final_path, i, -1))
            plan[i] = {"final": final_path, "parts": [final_path], "split": False}
            total_procs += 1

    print(f"[다운로드] {total_procs}개 ffmpeg 프로세스 동시 실행 중...")

    # 모든 프로세스 완료 대기
    for proc, out_path, block_idx, sub_idx in jobs:
        _, stderr = proc.communicate()
        if out_path.exists() and out_path.stat().st_size > 1000:
            size_mb = out_path.stat().st_size / 1024 / 1024
            label = f"블록{block_idx}"
            if sub_idx >= 0:
                label += f"-청크{sub_idx}"
            print(f"  OK {label}: {out_path.name} ({size_mb:.1f}MB)")
        else:
            err = stderr.decode("utf-8", errors="replace")[-200:] if stderr else "unknown"
            label = f"블록{block_idx}" + (f"-청크{sub_idx}" if sub_idx >= 0 else "")
            print(f"  FAIL {label}: {err}")

    # 분할 블록 결합 + 최종 결과 수집
    results: list[Path] = []
    for i in sorted(plan.keys()):
        p = plan[i]
        if p["split"]:
            valid_parts = [pt for pt in p["parts"] if pt.exists() and pt.stat().st_size > 1000]
            if not valid_parts:
                print(f"  SKIP 블록{i}: 유효한 청크 없음")
                continue
            if _concat_wav_files(valid_parts, p["final"], ffmpeg_path):
                size_mb = p["final"].stat().st_size / 1024 / 1024
                print(f"  MERGE 블록{i}: {len(valid_parts)}개 청크 -> {p['final'].name} ({size_mb:.1f}MB)")
                results.append(p["final"])
            else:
                print(f"  FAIL 블록{i}: 결합 실패")
        else:
            if p["final"].exists() and p["final"].stat().st_size > 1000:
                results.append(p["final"])

    print(f"[다운로드] {len(results)}/{len(speech_blocks)}개 블록 완료")
    return results
