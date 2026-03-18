"""자동 업데이트 모듈 — GitHub Releases 기반.

GitHub API로 최신 릴리스를 체크하고, 새 버전이 있으면
EXE를 다운로드하여 현재 실행 파일을 교체한다.

사용 예시:
    from src.updater import check_for_update, download_update, apply_update

    info = check_for_update()
    if info:
        exe_path = download_update(info.download_url, Path("sonote_new.exe"))
        apply_update(exe_path)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

__all__ = [
    "UpdateInfo",
    "UpdateError",
    "check_for_update",
    "download_update",
    "download_update_with_retry",
    "apply_update",
    "get_current_version",
    "start_background_check",
]

# --- 상수 ---
GITHUB_REPO: str = "tellang/sonote"
GITHUB_API_URL: str = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
# 백그라운드 체크 간격: 24시간 (초)
CHECK_INTERVAL_SEC: int = 24 * 60 * 60
# API 요청 타임아웃 (초)
REQUEST_TIMEOUT_SEC: int = 10
# 다운로드 청크 크기 (바이트)
DOWNLOAD_CHUNK_SIZE: int = 65536
# exponential backoff 재시도 설정
RETRY_MAX_ATTEMPTS: int = 5          # check/download 각각 최대 재시도 횟수
RETRY_BASE_SEC: int = 5              # 기본 대기 초 (2^attempt * base)
RETRY_CAP_SEC: int = 300             # 최대 대기 상한 (300초 = 5분)


class UpdateError(Exception):
    """업데이트 관련 예외."""
    pass


@dataclass
class UpdateInfo:
    """최신 릴리스 정보.

    Attributes:
        version: 릴리스 버전 문자열 (예: "1.2.0").
        download_url: EXE 다운로드 URL.
        release_notes: 릴리스 노트 (마크다운).
        published_at: 릴리스 게시 일시 (ISO 8601 문자열).
        checksum_sha256: SHA256 체크섬 (latest.json에 있을 경우, 없으면 None).
    """

    version: str
    download_url: str
    release_notes: str
    published_at: str
    checksum_sha256: Optional[str] = None


# ---------------------------------------------------------------------------
# 버전 비교 유틸리티
# ---------------------------------------------------------------------------

def _parse_version(version_str: str) -> tuple[int, ...]:
    """버전 문자열을 비교 가능한 정수 튜플로 파싱한다.

    pre-release 접미사 처리:
      - 정식 릴리스 "1.1.0"       → (1, 1, 0, 1, 0)   # pre=1 (정식)
      - 베타       "1.1.0b1"      → (1, 1, 0, 0, 1)   # pre=0 (베타)
      - 알파       "1.1.0a2"      → (1, 1, 0, 0, -1)  # pre=0 (알파)
      - RC         "1.1.0rc1"     → (1, 1, 0, 0, 2)   # pre=0 (RC)

    튜플 구조: (major, minor, patch, is_release, pre_rank)
      is_release=1: 정식 릴리스 (a/b/rc 접미사 없음)
      is_release=0: pre-release
    """
    import re

    clean = version_str.lstrip("v").split("+")[0]  # build metadata 제거

    # pre-release 접미사 분리: "1.1.0b2" → base="1.1.0", pre_type="b", pre_num=2
    m = re.match(r"^([\d.]+)(?:(a|b|rc)(\d*))?$", clean)
    if not m:
        # 파싱 불가 시 폴백: 숫자 부분만 추출
        nums = re.findall(r"\d+", clean)
        return tuple(int(n) for n in nums) if nums else (0,)

    base_str, pre_type, pre_num_str = m.group(1), m.group(2), m.group(3)

    # base 버전 파싱 (major, minor, patch)
    base_parts = []
    for part in base_str.split("."):
        try:
            base_parts.append(int(part))
        except ValueError:
            break
    # 최소 3자리 보장
    while len(base_parts) < 3:
        base_parts.append(0)

    if pre_type is None:
        # 정식 릴리스: pre-release보다 항상 높음
        return (*base_parts, 1, 0)

    # pre-release 순위: a < b < rc
    pre_rank_map = {"a": -1, "b": 1, "rc": 2}
    pre_num = int(pre_num_str) if pre_num_str else 0
    pre_rank = pre_rank_map.get(pre_type, 0)

    return (*base_parts, 0, pre_rank * 100 + pre_num)


def _is_newer(latest: str, current: str) -> bool:
    """latest 버전이 current보다 새로운지 비교한다."""
    try:
        return _parse_version(latest) > _parse_version(current)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 현재 버전 조회
# ---------------------------------------------------------------------------

def get_current_version() -> str:
    """현재 앱 버전을 반환한다.

    우선순위:
      1. src/__init__.py의 __version__
      2. 실행 파일 옆 pyproject.toml (개발 환경)
      3. 패키지 메타데이터 (importlib.metadata)
      4. 폴백: "0.0.0"
    """
    # 1) __version__ 속성
    try:
        from src import __version__  # type: ignore[attr-defined]
        if __version__:
            return __version__
    except (ImportError, AttributeError):
        pass

    # 2) importlib.metadata
    try:
        import importlib.metadata
        return importlib.metadata.version("sonote")
    except Exception:
        pass

    # 3) pyproject.toml 직접 파싱 (개발 환경)
    try:
        candidate_paths = [
            # 현재 파일 기준 상위 디렉토리
            Path(__file__).resolve().parent.parent / "pyproject.toml",
            # 실행 파일 기준 (PyInstaller onefile/onedir)
            Path(sys.executable).parent / "pyproject.toml",
        ]
        for pyproject_path in candidate_paths:
            if pyproject_path.exists():
                text = pyproject_path.read_text(encoding="utf-8")
                for line in text.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("version") and "=" in stripped:
                        _, _, value = stripped.partition("=")
                        return value.strip().strip('"').strip("'")
    except Exception:
        pass

    return "0.0.0"


# ---------------------------------------------------------------------------
# GitHub API — 최신 릴리스 체크
# ---------------------------------------------------------------------------

def _find_exe_asset(assets: list[dict]) -> Optional[dict]:
    """릴리스 assets에서 Windows EXE를 찾아 반환한다.

    이름 패턴: sonote*.exe (대소문자 무시)
    """
    for asset in assets:
        name: str = asset.get("name", "").lower()
        if name.startswith("sonote") and name.endswith(".exe"):
            return asset
    # 폴백: .exe 확장자를 가진 첫 번째 파일
    for asset in assets:
        name = asset.get("name", "").lower()
        if name.endswith(".exe"):
            return asset
    return None


def _fetch_checksum_from_latest_json(assets: list[dict], version: str) -> Optional[str]:
    """latest.json asset에서 SHA256 체크섬을 가져온다.

    latest.json 구조:
        {"version": "...", "download_url": "...", "checksum": "sha256:..."}
    """
    for asset in assets:
        if asset.get("name", "").lower() == "latest.json":
            url: str = asset.get("browser_download_url", "")
            if not url:
                return None
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": f"sonote-updater/{get_current_version()}"},
                )
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    checksum = data.get("checksum", "")
                    # "sha256:abcdef..." 또는 "abcdef..." 형식 모두 지원
                    if checksum.startswith("sha256:"):
                        return checksum[7:]
                    return checksum or None
            except Exception:
                return None
    return None


def check_for_update() -> Optional[UpdateInfo]:
    """GitHub API로 최신 릴리스를 체크하고 새 버전이 있으면 UpdateInfo를 반환한다.

    현재 버전보다 최신 버전이 없거나 오류 발생 시 None을 반환한다.

    Returns:
        새 버전이 있으면 UpdateInfo, 없으면 None.

    Raises:
        UpdateError: 네트워크 오류 또는 API 응답 파싱 실패 시.
    """
    current = get_current_version()

    try:
        req = urllib.request.Request(
            GITHUB_API_URL,
            headers={
                "User-Agent": f"sonote-updater/{current}",
                "Accept": "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            if resp.status != 200:
                raise UpdateError(f"GitHub API 응답 오류: HTTP {resp.status}")
            release_data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise UpdateError(f"GitHub API 연결 실패: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise UpdateError(f"GitHub API 응답 파싱 실패: {exc}") from exc

    latest_tag: str = release_data.get("tag_name", "").lstrip("v")
    if not latest_tag:
        raise UpdateError("릴리스 tag_name을 찾을 수 없습니다.")

    # 현재 버전과 비교
    if not _is_newer(latest_tag, current):
        return None

    # EXE asset 탐색
    assets: list[dict] = release_data.get("assets", [])
    exe_asset = _find_exe_asset(assets)
    if not exe_asset:
        raise UpdateError("릴리스에 EXE 파일이 없습니다.")

    download_url: str = exe_asset.get("browser_download_url", "")
    if not download_url or not download_url.startswith("https://"):
        raise UpdateError(f"안전하지 않은 다운로드 URL: {download_url}")

    # latest.json에서 체크섬 조회 (선택적)
    checksum = _fetch_checksum_from_latest_json(assets, latest_tag)

    return UpdateInfo(
        version=latest_tag,
        download_url=download_url,
        release_notes=release_data.get("body", ""),
        published_at=release_data.get("published_at", ""),
        checksum_sha256=checksum,
    )


# ---------------------------------------------------------------------------
# 다운로드
# ---------------------------------------------------------------------------

def download_update(
    url: str,
    dest: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """EXE를 다운로드하고 경로를 반환한다.

    Args:
        url: 다운로드 URL (HTTPS 필수).
        dest: 저장할 파일 경로.
        progress_callback: (downloaded_bytes, total_bytes) 호출 콜백.
                           total_bytes가 0이면 크기 미확인.

    Returns:
        다운로드된 파일 경로 (dest와 동일).

    Raises:
        UpdateError: 비HTTPS URL 또는 다운로드 실패 시.
    """
    # HTTPS 강제
    if not url.startswith("https://"):
        raise UpdateError(f"HTTPS가 아닌 URL은 허용되지 않습니다: {url}")

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": f"sonote-updater/{get_current_version()}"},
        )
        with urllib.request.urlopen(req) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0

            with dest.open("wb") as f:
                while True:
                    chunk = resp.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size)

    except urllib.error.URLError as exc:
        # 불완전한 파일 정리
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise UpdateError(f"다운로드 실패: {exc}") from exc

    return dest


def verify_checksum(file_path: Path, expected_sha256: str) -> bool:
    """파일의 SHA256 체크섬을 검증한다.

    Args:
        file_path: 검증할 파일 경로.
        expected_sha256: 예상 SHA256 hex 문자열.

    Returns:
        체크섬이 일치하면 True.
    """
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(DOWNLOAD_CHUNK_SIZE), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    return actual.lower() == expected_sha256.lower()


# ---------------------------------------------------------------------------
# 업데이트 적용
# ---------------------------------------------------------------------------

def apply_update(new_exe_path: Path) -> None:
    """현재 EXE를 .bak으로 백업하고 새 EXE로 교체한 뒤 앱을 재시작한다.

    Windows에서는 실행 중인 EXE를 직접 교체할 수 없으므로
    배치 스크립트를 통해 프로세스 종료 후 교체한다.

    Args:
        new_exe_path: 새 EXE 파일 경로.

    Raises:
        UpdateError: EXE 탐지 실패 또는 교체 오류.
    """
    # 현재 실행 파일 경로 탐지
    if getattr(sys, "frozen", False):
        # PyInstaller 환경
        current_exe = Path(sys.executable)
    else:
        raise UpdateError(
            "개발 환경에서는 apply_update를 사용할 수 없습니다. "
            "PyInstaller로 빌드된 EXE에서만 동작합니다."
        )

    if not new_exe_path.exists():
        raise UpdateError(f"새 EXE 파일을 찾을 수 없습니다: {new_exe_path}")

    backup_path = current_exe.with_suffix(".bak")

    if sys.platform == "win32":
        _apply_update_windows(current_exe, new_exe_path, backup_path)
    else:
        _apply_update_posix(current_exe, new_exe_path, backup_path)


def _apply_update_windows(current_exe: Path, new_exe: Path, backup_path: Path) -> None:
    """Windows에서 배치 스크립트로 EXE를 교체하고 재시작한다."""
    # 배치 스크립트: 현재 프로세스 종료 대기 → 백업 → 교체 → 재시작
    batch_content = f"""@echo off
:: sonote 자동 업데이트 배치 스크립트
:: 현재 프로세스가 완전히 종료될 때까지 대기
:wait_loop
tasklist /FI "PID eq {os.getpid()}" 2>nul | find /I "{os.getpid()}" >nul
if not errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto wait_loop
)

:: 기존 백업 제거
if exist "{backup_path}" del /f /q "{backup_path}"

:: 현재 EXE를 백업으로 이동
move /y "{current_exe}" "{backup_path}"
if errorlevel 1 (
    echo [업데이트 오류] 현재 EXE 백업 실패
    pause
    exit /b 1
)

:: 새 EXE로 교체
move /y "{new_exe}" "{current_exe}"
if errorlevel 1 (
    echo [업데이트 오류] 새 EXE 교체 실패 — 백업에서 복구 시도
    move /y "{backup_path}" "{current_exe}"
    pause
    exit /b 1
)

:: 앱 재시작
start "" "{current_exe}"

:: 배치 스크립트 자체 삭제
del /f /q "%~f0"
"""
    # 임시 디렉토리에 배치 파일 생성
    tmp_dir = Path(tempfile.gettempdir())
    batch_path = tmp_dir / "sonote_update.bat"
    batch_path.write_text(batch_content, encoding="utf-8")

    # 배치 스크립트를 분리된 프로세스로 실행 (현재 프로세스와 독립)
    subprocess.Popen(
        ["cmd.exe", "/c", str(batch_path)],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
        close_fds=True,
    )

    # 현재 앱 종료
    sys.exit(0)


def _apply_update_posix(current_exe: Path, new_exe: Path, backup_path: Path) -> None:
    """Unix/macOS에서 직접 EXE를 교체하고 재시작한다."""
    try:
        # 기존 백업 제거
        if backup_path.exists():
            backup_path.unlink()

        # 현재 EXE 백업
        shutil.copy2(str(current_exe), str(backup_path))

        # 새 EXE로 교체 (원자적 교체)
        new_exe.replace(current_exe)

        # 실행 권한 부여
        current_exe.chmod(0o755)

    except OSError as exc:
        # 교체 실패 시 백업 복원 시도
        if backup_path.exists() and not current_exe.exists():
            shutil.copy2(str(backup_path), str(current_exe))
        raise UpdateError(f"EXE 교체 실패: {exc}") from exc

    # 앱 재시작 후 종료
    os.execv(str(current_exe), sys.argv)


# ---------------------------------------------------------------------------
# 백그라운드 업데이트 체크
# ---------------------------------------------------------------------------

def _get_last_check_time() -> float:
    """config.json에서 마지막 업데이트 체크 시간을 읽어온다."""
    try:
        from .config import get_config
        config = get_config()
        return float(config.get("last_update_check", 0.0))
    except Exception:
        return 0.0


def _set_last_check_time(timestamp: float) -> None:
    """config.json에 마지막 업데이트 체크 시간을 저장한다."""
    try:
        from .config import get_config
        config = get_config()
        config.set("last_update_check", timestamp)
    except Exception:
        pass


def start_background_check(
    on_update_found: Optional[Callable[[UpdateInfo], None]] = None,
    force: bool = False,
) -> None:
    """백그라운드 스레드에서 업데이트를 체크한다.

    24시간마다 한 번만 체크하며, 새 버전 발견 시 on_update_found 콜백을 호출한다.
    메인 스레드를 차단하지 않는다.

    Args:
        on_update_found: 새 버전 발견 시 호출할 콜백 (UpdateInfo 전달).
                         None이면 콘솔에 메시지 출력.
        force: True이면 마지막 체크 시간 무시하고 즉시 체크.
    """
    def _default_callback(info: UpdateInfo) -> None:
        """기본 콜백 — 콘솔에 업데이트 알림 메시지 출력."""
        print(
            f"\n[sonote 업데이트] 새 버전 {info.version}이 출시되었습니다!\n"
            f"  다운로드: {info.download_url}\n"
            f"  설치: sonote update --install\n",
            file=sys.stderr,
        )

    callback = on_update_found or _default_callback

    def _check_worker() -> None:
        """백그라운드 체크 워커 — exponential backoff 재시도 포함."""
        now = time.time()
        last_check = _get_last_check_time()

        # 24시간 미경과 시 건너뜀 (force가 아닌 경우)
        if not force and (now - last_check) < CHECK_INTERVAL_SEC:
            return

        _set_last_check_time(now)

        # check_for_update() 실패 시 exponential backoff 재시도
        info: Optional[UpdateInfo] = None
        for attempt in range(RETRY_MAX_ATTEMPTS + 1):
            try:
                info = check_for_update()
                break  # 성공 시 루프 탈출
            except UpdateError as exc:
                if attempt >= RETRY_MAX_ATTEMPTS:
                    # 최대 재시도 초과 — 이번 사이클 포기
                    print(
                        f"[sonote 업데이트] 체크 실패 (최대 {RETRY_MAX_ATTEMPTS}회 재시도 후 포기): {exc}",
                        file=sys.stderr,
                    )
                    return
                wait_sec = min(2 ** attempt * RETRY_BASE_SEC, RETRY_CAP_SEC)
                print(
                    f"[sonote 업데이트] 체크 실패 (시도 {attempt + 1}/{RETRY_MAX_ATTEMPTS},"
                    f" {wait_sec}초 후 재시도): {exc}",
                    file=sys.stderr,
                )
                time.sleep(wait_sec)
            except Exception as exc:
                # 예상치 못한 오류는 즉시 포기
                print(f"[sonote 업데이트] 예상치 못한 오류: {exc}", file=sys.stderr)
                return

        if info is None:
            # 최신 버전 — 업데이트 불필요
            return

        # download_update() 실패 시 exponential backoff 재시도
        if not callable(callback):
            return

        # 콜백 내부에서 download_update()를 호출하는 경우를 위해
        # 콜백 자체를 backoff 래퍼로 감싸지 않고, 콜백 호출만 1회 수행한다.
        # (콜백이 download_update를 직접 쓰는 경우 download_update_with_retry 사용 권장)
        try:
            callback(info)
        except Exception as exc:
            print(f"[sonote 업데이트] 콜백 처리 중 오류: {exc}", file=sys.stderr)

    thread = threading.Thread(target=_check_worker, daemon=True, name="sonote-update-check")
    thread.start()


def download_update_with_retry(
    url: str,
    dest: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """download_update()를 exponential backoff로 재시도하는 래퍼.

    Args:
        url: 다운로드 URL (HTTPS 필수).
        dest: 저장할 파일 경로.
        progress_callback: (downloaded_bytes, total_bytes) 호출 콜백.

    Returns:
        다운로드된 파일 경로.

    Raises:
        UpdateError: 최대 재시도 횟수 초과 시.
    """
    for attempt in range(RETRY_MAX_ATTEMPTS + 1):
        try:
            return download_update(url, dest, progress_callback)
        except UpdateError as exc:
            if attempt >= RETRY_MAX_ATTEMPTS:
                raise UpdateError(
                    f"다운로드 실패 (최대 {RETRY_MAX_ATTEMPTS}회 재시도 후 포기): {exc}"
                ) from exc
            wait_sec = min(2 ** attempt * RETRY_BASE_SEC, RETRY_CAP_SEC)
            print(
                f"[sonote 업데이트] 다운로드 실패 (시도 {attempt + 1}/{RETRY_MAX_ATTEMPTS},"
                f" {wait_sec}초 후 재시도): {exc}",
                file=sys.stderr,
            )
            time.sleep(wait_sec)
    # 도달 불가 — 타입 체커를 위한 안전 장치
    raise UpdateError("download_update_with_retry: 예상치 못한 루프 탈출")
