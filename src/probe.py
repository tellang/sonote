"""
스트림 음성 구간 탐색 모듈 (BGM vs 강의 경계 탐색)

일반 강의/세미나 스트림 구조:
  [초반 BGM 1~2시간] → [강의 1] → [쉬는시간 BGM] → [강의 2] → ... → [종료 BGM]

제공 기능:
- find_content_start(): 초반 BGM 건너뛰기 (binary search)
- scan_stream(): 전체 구간 맵핑 — 모든 BGM/음성 블록 식별

원리:
- BGM: 연속 오디오, 침묵 구간 거의 없음
- 한국어 음성: 문장 사이 자연스러운 일시정지 → 침묵 구간 다수
- ffmpeg silencedetect로 침묵 패턴 분석 (GPU 불필요, 빠름)
"""
import subprocess
import shutil
import tempfile
from pathlib import Path

from .download import get_stream_url


def _download_probe(
    stream_url: str,
    minutes_back: int,
    probe_seconds: int = 15,
) -> Path | None:
    """특정 시간 오프셋에서 짧은 오디오 프로브 다운로드"""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg가 설치되어 있지 않습니다")

    tmp = Path(tempfile.mktemp(suffix=".wav", prefix="lt-probe-"))

    cmd = [ffmpeg_path]
    if minutes_back > 0:
        cmd.extend(["-live_start_index", str(-minutes_back * 60)])
    cmd.extend([
        "-i", stream_url,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        "-t", str(probe_seconds),
        str(tmp), "-y",
    ])

    try:
        subprocess.run(cmd, capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        return None

    if tmp.exists() and tmp.stat().st_size > 1000:
        return tmp
    return None


def _detect_speech(audio_path: Path, noise_db: int = -35, min_dur: float = 0.3) -> tuple[bool, int]:
    """
    오디오에 음성 패턴이 있는지 감지 (silence gap 분석).

    BGM: 연속 오디오 → 침묵 구간 0~1개
    음성: 문장 사이 일시정지 → 침묵 구간 2개 이상 (15초 기준)

    Returns:
        (has_speech, silence_count)
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return False, 0

    result = subprocess.run(
        [ffmpeg_path, "-i", str(audio_path),
         "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=15,
    )

    silence_count = result.stderr.count("silence_start")
    return silence_count >= 2, silence_count


def find_content_start(
    video_url: str,
    max_back_minutes: int = 180,
    probe_seconds: int = 15,
    format_id: str = "91",
    verbose: bool = True,
    cookies_path: str | Path | None = None,
) -> dict:
    """
    라이브 스트림에서 실제 강의 시작 지점을 탐색.

    Phase 1: 지수 스캔 — 넓은 간격으로 빠르게 프로빙
    Phase 2: 이진 탐색 — BGM↔음성 경계를 ~2분 정밀도로 좁힘

    Args:
        video_url: YouTube 라이브 URL
        max_back_minutes: 최대 탐색 범위 (분)
        probe_seconds: 각 프로브 길이 (초)
        format_id: yt-dlp 포맷 ID
        verbose: 진행 출력 여부
        cookies_path: cookies.txt 경로 (None이면 자동 탐색)

    Returns:
        {
            "speech_start_back": int,  # 음성 시작 지점 (현재에서 N분 전)
            "content_back": int,       # 권장 다운로드 시작 (여유 포함)
            "probes": int,             # 수행한 프로브 수
            "status": str,             # "found" | "all_speech" | "all_bgm" | "no_data"
        }
    """
    _log = print if verbose else lambda *a, **kw: None

    _log("[탐색] 스트림 URL 추출 중...")
    stream_url = get_stream_url(video_url, format_id, cookies_path=cookies_path)

    # Phase 1: 지수 스캔
    checkpoints = [5, 15, 30, 60, 90, 120, 150, 180]
    checkpoints = [c for c in checkpoints if c <= max_back_minutes]

    probe_count = 0
    speech_points: list[int] = []
    bgm_points: list[int] = []
    unavail_points: list[int] = []

    _log(f"\n[Phase 1] 지수 스캔 (최대 {max_back_minutes}분 전)")
    _log("-" * 40)

    for back in checkpoints:
        probe_path = _download_probe(stream_url, back, probe_seconds)
        probe_count += 1

        if probe_path is None:
            _log(f"  {back:3d}분 전: --- 데이터 없음")
            unavail_points.append(back)
            continue

        try:
            has_speech, sil_count = _detect_speech(probe_path)
            label = "음성" if has_speech else "BGM "
            _log(f"  {back:3d}분 전: [{label}] 침묵구간 {sil_count}개")

            if has_speech:
                speech_points.append(back)
            else:
                bgm_points.append(back)
        except Exception as e:
            _log(f"  {back:3d}분 전: [오류] {e}")
        finally:
            if probe_path and probe_path.exists():
                probe_path.unlink(missing_ok=True)

    # 결과 분석
    if not speech_points and not bgm_points:
        _log("\n[결과] 데이터를 가져올 수 없습니다")
        return {"speech_start_back": 0, "content_back": 0,
                "probes": probe_count, "status": "no_data"}

    if not speech_points:
        _log("\n[결과] 탐색 범위 내 음성 구간 없음 (전체 BGM)")
        return {"speech_start_back": 0, "content_back": 0,
                "probes": probe_count, "status": "all_bgm"}

    if not bgm_points:
        earliest = max(speech_points)
        _log(f"\n[결과] 전체 음성 — {earliest}분 전부터 강의")
        return {"speech_start_back": earliest, "content_back": earliest + 2,
                "probes": probe_count, "status": "all_speech"}

    # Phase 2: 이진 탐색 — BGM↔음성 경계
    # 타임라인: [BGM ...] [강의 ...] [현재]
    # minutes_back: 큰 값 = 과거(BGM), 작은 값 = 최근(강의)
    max_speech_back = max(speech_points)    # 음성이 감지된 가장 먼 시점
    further_bgm = [b for b in bgm_points if b > max_speech_back]

    if not further_bgm:
        # BGM이 음성보다 더 최근? 비정상 — 가장 먼 음성 지점 사용
        _log(f"\n[결과] 강의 시작: ~{max_speech_back + 2}분 전")
        return {"speech_start_back": max_speech_back,
                "content_back": max_speech_back + 2,
                "probes": probe_count, "status": "found"}

    lo = max_speech_back       # 음성 확인된 가장 먼 시점
    hi = min(further_bgm)      # 음성 바로 너머의 BGM 시점

    _log(f"\n[Phase 2] 이진 탐색 ({lo}~{hi}분 사이)")
    _log("-" * 40)

    while hi - lo > 2:
        mid = (lo + hi) // 2
        probe_path = _download_probe(stream_url, mid, probe_seconds)
        probe_count += 1

        if probe_path is None:
            _log(f"  {mid:3d}분 전: --- 데이터 없음")
            hi = mid
            continue

        try:
            has_speech, sil_count = _detect_speech(probe_path)
            label = "음성" if has_speech else "BGM "
            _log(f"  {mid:3d}분 전: [{label}] 침묵 {sil_count}개")

            if has_speech:
                lo = mid
            else:
                hi = mid
        except Exception:
            hi = mid
        finally:
            if probe_path and probe_path.exists():
                probe_path.unlink(missing_ok=True)

    # lo = 음성 확인 가장 먼 시점, +2분 여유
    result_back = lo + 2

    _log(f"\n{'=' * 40}")
    _log(f"[결과] 강의 시작: ~{lo}분 전")
    _log(f"[권장] --back {result_back} 로 다운로드")
    _log(f"[프로브] {probe_count}회 수행")

    return {
        "speech_start_back": lo,
        "content_back": result_back,
        "probes": probe_count,
        "status": "found",
    }


def scan_stream(
    video_url: str,
    max_back_minutes: int = 180,
    step_minutes: int = 5,
    probe_seconds: int = 15,
    format_id: str = "91",
    cookies_path: str | Path | None = None,
) -> dict:
    """
    전체 스트림 구간 맵핑 — 모든 BGM/음성 블록 식별.

    일반 강의 스트림 구조:
      [초반 BGM] → [강의1] → [쉬는시간] → [강의2] → [쉬는시간] → ... → [종료 BGM]

    5분 간격으로 전체 스캔하여 구간별 유형(음성/BGM) 파악.

    Args:
        video_url: YouTube 라이브 URL
        max_back_minutes: 최대 탐색 범위 (분)
        step_minutes: 프로브 간격 (분)
        probe_seconds: 각 프로브 길이 (초)
        format_id: yt-dlp 포맷 ID
        cookies_path: cookies.txt 경로 (None이면 자동 탐색)

    Returns:
        {
            "blocks": [{"type": "speech"|"bgm", "from": int, "to": int}, ...],
            "speech_ranges": [(from, to), ...],  # 강의 구간만
            "total_speech_min": int,
            "probes": int,
        }
    """
    print("[스캔] 스트림 URL 추출 중...")
    stream_url = get_stream_url(video_url, format_id, cookies_path=cookies_path)

    checkpoints = list(range(step_minutes, max_back_minutes + 1, step_minutes))
    # 최근→과거 순서로 스캔 (과거가 큰 값)
    # 결과는 시간순 정렬 (과거→최근 = 큰값→작은값)

    results: list[tuple[int, str]] = []
    probe_count = 0

    total = len(checkpoints)
    print(f"[스캔] {total}개 지점 프로빙 ({step_minutes}분 간격, 최대 {max_back_minutes}분)")
    print("-" * 50)

    for i, back in enumerate(checkpoints):
        probe_path = _download_probe(stream_url, back, probe_seconds)
        probe_count += 1

        if probe_path is None:
            print(f"  [{i+1:2d}/{total}] {back:3d}분 전: --- 없음")
            continue

        try:
            has_speech, sil_count = _detect_speech(probe_path)
            label = "음성" if has_speech else "BGM"
            results.append((back, "speech" if has_speech else "bgm"))
            bar = "█" * sil_count + "░" * max(0, 10 - sil_count)
            print(f"  [{i+1:2d}/{total}] {back:3d}분 전: {label:4s} {bar} ({sil_count})")
        except Exception as e:
            print(f"  [{i+1:2d}/{total}] {back:3d}분 전: 오류 {e}")
        finally:
            if probe_path and probe_path.exists():
                probe_path.unlink(missing_ok=True)

    if not results:
        print("\n[결과] 데이터 없음")
        return {"blocks": [], "speech_ranges": [], "total_speech_min": 0,
                "probes": probe_count}

    # 시간순 정렬: 과거(큰 값) → 최근(작은 값)
    results.sort(key=lambda x: -x[0])

    # 연속 블록으로 그룹핑
    blocks: list[dict] = []
    current_type = results[0][1]
    block_start = results[0][0]

    for i in range(1, len(results)):
        back, seg_type = results[i]
        if seg_type != current_type:
            blocks.append({
                "type": current_type,
                "from": block_start,
                "to": results[i - 1][0],
            })
            current_type = seg_type
            block_start = back

    # 마지막 블록
    blocks.append({
        "type": current_type,
        "from": block_start,
        "to": results[-1][0],
    })

    # 음성 구간만 추출
    speech_ranges = [(b["from"], b["to"]) for b in blocks if b["type"] == "speech"]
    total_speech = sum(abs(r[0] - r[1]) + step_minutes for r in speech_ranges)

    # 결과 출력
    print(f"\n{'=' * 50}")
    print("[구간 맵]")
    for b in blocks:
        icon = "🎤" if b["type"] == "speech" else "🎵"
        label = "강의" if b["type"] == "speech" else "BGM "
        print(f"  {icon} {label}  {b['from']:3d}분 전 ~ {b['to']:3d}분 전")

    print(f"\n[요약] 강의 구간: {len(speech_ranges)}개, 총 ~{total_speech}분")

    if speech_ranges:
        # 전체 강의 다운로드 명령 제안
        earliest = max(r[0] for r in speech_ranges)
        latest = min(r[1] for r in speech_ranges)
        duration = earliest - latest + step_minutes
        print(f"[권장] --back {earliest + 2} -d {duration}")

    return {
        "blocks": blocks,
        "speech_ranges": speech_ranges,
        "total_speech_min": total_speech,
        "probes": probe_count,
    }
