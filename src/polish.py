"""회의록 LLM 후처리 — Codex/Ollama(STT 교정) + Gemini/Ollama(요약 생성)

회의 종료 후 .md 파일을 자동 후처리:
  1. Codex Pro 또는 Ollama: Raw Data의 STT 오인식을 맥락 기반 교정
  2. Gemini 또는 Ollama: 구조화된 요약 + To-do 생성

CLI 미설치 시 Ollama 폴백. --ollama로 Ollama 전용 모드.
--no-polish로 전체 비활성화.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from collections.abc import Callable

__all__ = [
    "polish_meeting",
    "is_codex_available",
    "is_gemini_available",
    "is_ollama_available",
    "extract_keywords_with_codex",
    "extract_keywords_with_ollama",
    "correct_with_ollama_parallel",
    "summarize_with_ollama",
    "_correct_batch",
    "_correct_batch_ollama",
]

# 기본 타임아웃 (초) — 긴 회의록 처리 시 여유 필요
_DEFAULT_TIMEOUT = 300

# --- Ollama 설정 ---
_OLLAMA_DEFAULT_MODEL = "gemma3:27b"
_OLLAMA_BASE_URL = "http://localhost:11434"


def is_codex_available() -> bool:
    """Codex CLI 설치 여부"""
    return shutil.which("codex") is not None


def is_gemini_available() -> bool:
    """Gemini CLI 설치 여부"""
    return shutil.which("gemini") is not None


def is_ollama_available(model: str = _OLLAMA_DEFAULT_MODEL) -> bool:
    """Ollama 서버 + 모델 가용 여부 확인."""
    try:
        import httpx
        resp = httpx.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        # 모델명 부분 매칭 (gemma3:27b → gemma3 포함 여부)
        base = model.split(":")[0]
        return any(base in m for m in models)
    except Exception:
        return False


def _run_cli(args: list[str], timeout: int, cwd: str) -> tuple[bool, str]:
    """CLI 명령 실행 — Windows/Unix 호환.

    Returns:
        (성공 여부, stderr 텍스트)
    """
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            shell=(os.name == "nt"),  # Windows .CMD 배치 파일 실행 지원
        )
        return result.returncode == 0, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, f"타임아웃 ({timeout}초)"
    except (FileNotFoundError, OSError) as e:
        return False, str(e)


def _run_ollama(
    prompt: str,
    model: str = _OLLAMA_DEFAULT_MODEL,
    timeout: int = 120,
) -> tuple[bool, str]:
    """Ollama API 단일 호출.

    Returns:
        (성공 여부, 응답 텍스트 또는 에러 메시지)
    """
    import httpx
    try:
        resp = httpx.post(
            f"{_OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return True, resp.json().get("response", "")
        return False, f"HTTP {resp.status_code}"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------
# Codex 기반 교정/추출
# ---------------------------------------------------------------

def correct_with_codex(md_path: Path, timeout: int = _DEFAULT_TIMEOUT) -> bool:
    """Codex CLI로 Raw Data 섹션의 STT 오인식 교정.

    codex exec 모드로 파일을 직접 수정.
    실행 전 .pre-polish.md 백업 생성. 실패/데이터 손실 시 자동 복원.
    """
    abs_path = md_path.resolve()
    backup_path = abs_path.with_suffix(".pre-polish.md")

    # 백업 생성 — Codex가 파일을 망가뜨려도 복원 가능
    import shutil as _shutil
    _shutil.copy2(abs_path, backup_path)
    original_lines = abs_path.read_text(encoding="utf-8").count("\n")
    prompt = (
        f'회의록 파일 "{abs_path}"의 "# Raw Data" 섹션에서 '
        "STT 오인식을 맥락 기반으로 교정해줘. "
        "동음이의어/유사음 오인식, 기술 용어 오류, 명백한 문법 오류만 수정. "
        "형식 '- [HH:MM:SS] [화자] 텍스트' 유지, 타임스탬프/화자 라벨 수정 금지. "
        "절대 금지: Raw Data 섹션의 줄을 삭제하지 마라. 모든 세그먼트를 유지하고 텍스트만 교정하라. "
        "절대 금지: 파일의 다른 섹션(요약, To-do, 메타데이터)을 수정하거나 삭제하지 마라. "
        "절대 금지: .raw.txt 등 다른 파일을 삭제하지 마라. "
        "교정 후 같은 파일에 저장."
    )
    ok, err = _run_cli(
        [
            "codex", "exec",
            "--profile", "normal",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            prompt,
        ],
        timeout=timeout,
        cwd=str(abs_path.parent),
    )
    if not ok and err:
        print(f"[후처리] Codex 상세: {err[:200]}")

    # 결과 검증 — 줄 수가 절반 이하로 줄면 데이터 손실로 판단하여 복원
    if ok:
        result_lines = abs_path.read_text(encoding="utf-8").count("\n")
        if result_lines < original_lines * 0.5:
            print(f"[후처리] 데이터 손실 감지: {original_lines}줄 → {result_lines}줄. 백업에서 복원.")
            _shutil.copy2(backup_path, abs_path)
            ok = False
        else:
            # 성공 시 백업 삭제
            backup_path.unlink(missing_ok=True)
    else:
        # 실패 시 백업 복원
        if backup_path.exists():
            _shutil.copy2(backup_path, abs_path)
            print("[후처리] 실패 — 백업에서 원본 복원 완료")

    return ok


def _correct_batch(
    batch_lines: list[str],
    batch_idx: int,
    work_dir: Path,
    timeout: int,
) -> tuple[int, bool, list[str]]:
    """단일 배치의 STT 오인식 교정. 임시 파일에서 작업."""
    temp_path = work_dir / f".polish-batch-{batch_idx}.txt"
    temp_path.write_text("\n".join(batch_lines) + "\n", encoding="utf-8")
    original_count = len(batch_lines)

    prompt = (
        f'파일 "{temp_path.name}"의 회의 전사 내용에서 '
        "STT 오인식을 맥락 기반으로 교정해줘. "
        "동음이의어/유사음 오인식, 기술 용어 오류, 명백한 문법 오류만 수정. "
        "형식 '- [HH:MM:SS] [화자] 텍스트' 유지, 타임스탬프/화자 라벨 수정 금지. "
        "절대 금지: 줄을 삭제하거나 추가하지 마라. 줄 수를 반드시 유지하라. "
        "교정 후 같은 파일에 저장."
    )

    ok, err = _run_cli(
        [
            "codex", "exec",
            "--profile", "normal",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            prompt,
        ],
        timeout=timeout,
        cwd=str(work_dir),
    )

    corrected: list[str] = batch_lines  # 기본값: 원본 유지
    if ok and temp_path.exists():
        result_text = temp_path.read_text(encoding="utf-8").strip()
        result_lines = [line for line in result_text.split("\n") if line.strip()]
        # 줄 수 검증 — 불일치 시 원본 유지
        if len(result_lines) == original_count:
            corrected = result_lines
        else:
            print(
                f"[후처리] 배치 {batch_idx}: 줄 수 불일치 "
                f"({original_count} → {len(result_lines)}), 원본 유지"
            )
    elif not ok and err:
        print(f"[후처리] 배치 {batch_idx} Codex 상세: {err[:200]}")

    # 임시 파일 정리
    temp_path.unlink(missing_ok=True)

    return batch_idx, ok, corrected


def correct_with_codex_parallel(
    md_path: Path,
    timeout: int = _DEFAULT_TIMEOUT,
    batch_size: int = 10,
    max_workers: int = 3,
    progress_callback: Callable[[float], None] | None = None,
) -> bool:
    """세그먼트를 배치로 분할하여 Codex를 병렬 호출로 STT 교정.

    기존 correct_with_codex의 병렬 버전.
    Raw Data 섹션을 배치로 나누어 동시에 교정 후 원본에 병합.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    abs_path = md_path.resolve()
    backup_path = abs_path.with_suffix(".pre-polish.md")

    # 백업 생성
    import shutil as _shutil
    _shutil.copy2(abs_path, backup_path)

    content = abs_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    # Raw Data 섹션에서 세그먼트 라인 인덱스 추출
    segment_indices: list[int] = []
    in_raw = False
    for i, line in enumerate(lines):
        if line.strip() == "# Raw Data":
            in_raw = True
            continue
        if in_raw and line.startswith("# "):
            break
        if in_raw and line.startswith("- ["):
            segment_indices.append(i)

    if not segment_indices:
        backup_path.unlink(missing_ok=True)
        return True

    # 배치 분할
    batches: list[list[tuple[int, str]]] = []
    for start in range(0, len(segment_indices), batch_size):
        batch = [(idx, lines[idx]) for idx in segment_indices[start:start + batch_size]]
        batches.append(batch)

    total_batches = len(batches)
    per_batch_timeout = max(120, calc_timeout(batch_size))
    print(
        f"[후처리] {len(segment_indices)}세그먼트 → {total_batches}배치 "
        f"(배치당 {batch_size}, 워커 {max_workers}, 배치 타임아웃 {per_batch_timeout}초)"
    )

    completed = 0
    all_corrected: dict[int, str] = {}
    any_success = False
    work_dir = abs_path.parent

    # 병렬 실행
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for batch_idx, batch in enumerate(batches):
            batch_lines = [text for _, text in batch]
            future = pool.submit(
                _correct_batch, batch_lines, batch_idx, work_dir, per_batch_timeout,
            )
            futures[future] = (batch_idx, batch)

        for future in as_completed(futures):
            batch_idx, batch = futures[future]
            try:
                _, ok, corrected = future.result()
                if ok:
                    any_success = True
                for j, (line_idx, _) in enumerate(batch):
                    if j < len(corrected):
                        all_corrected[line_idx] = corrected[j]
            except Exception as e:
                print(f"[후처리] 배치 {batch_idx} 예외: {e}")

            completed += 1
            pct = completed / total_batches * 100
            print(f"[후처리] STT 교정 진행: {completed}/{total_batches} ({pct:.0f}%)")
            if progress_callback:
                progress_callback(pct)

    # 교정된 라인을 원본에 반영
    if all_corrected:
        for line_idx, corrected_text in all_corrected.items():
            lines[line_idx] = corrected_text
        new_content = "\n".join(lines)
        abs_path.write_text(new_content, encoding="utf-8")

        # 결과 검증
        original_line_count = content.count("\n")
        result_line_count = new_content.count("\n")
        if result_line_count < original_line_count * 0.5:
            print(
                f"[후처리] 데이터 손실 감지: "
                f"{original_line_count}줄 → {result_line_count}줄. 백업에서 복원."
            )
            _shutil.copy2(backup_path, abs_path)
            any_success = False
        else:
            backup_path.unlink(missing_ok=True)
    else:
        _shutil.copy2(backup_path, abs_path)
        print("[후처리] 전체 실패 — 백업에서 원본 복원")

    return any_success


def summarize_with_gemini(md_path: Path, timeout: int = _DEFAULT_TIMEOUT) -> bool:
    """Gemini CLI로 요약 + To-do 생성.

    파일의 placeholder 섹션을 채움.
    """
    abs_path = md_path.resolve()
    prompt = (
        f'회의록 파일 "{abs_path}"을 읽고 요약과 To-do를 생성해줘. '
        '"- (요약은 회의 후 별도 작성)" 줄을 구조화된 요약으로 교체하고 '
        '"- [ ] (회의 후 별도 작성)" 줄을 구체적 행동 항목으로 교체해. '
        "요약은 ## 주제, ## 핵심 논의(번호별 소제목), ## 결론 형식. "
        "Raw Data 섹션과 메타데이터(일시/경과시간/참석자)는 수정 금지. "
        "결과를 같은 파일에 저장."
    )
    ok, err = _run_cli(
        ["gemini", "-m", "gemini-3-flash-preview", "-y", "-p", prompt],
        timeout=timeout,
        cwd=str(abs_path.parent),
    )
    if not ok and err:
        print(f"[후처리] Gemini 상세: {err[:200]}")
    return ok


def calc_timeout(segment_count: int, base: int = 180, per_segment: int = 3) -> int:
    """세그먼트 수에 비례하는 타임아웃 계산.

    기본 180초 + 세그먼트당 3초. 최소 300초.
    예: 156 세그먼트 → 180 + 468 = 648초 (~11분)
    """
    return max(300, base + per_segment * segment_count)


def _cleanup_orphaned_temp_files(work_dir: Path) -> None:
    """이전 polish 실행에서 남은 임시 파일 정리."""
    patterns = [
        ".polish-batch-*.txt",
        "*.pre-polish.md",
        "*.summary.tmp.md",
        ".kw-input.txt",
        ".kw-output.txt",
        "*.postprocess-status.json",
    ]
    for pattern in patterns:
        for f in work_dir.glob(pattern):
            try:
                f.unlink()
            except OSError:
                pass


def extract_keywords_with_codex(
    text: str,
    work_dir: Path,
    timeout: int = 60,
) -> list[str]:
    """회의 전사 텍스트에서 도메인 키워드 추출 (Codex).

    전문 용어, 고유명사, 기술 용어, 프로젝트명 등을 추출하여
    Whisper initial_prompt에 피드백. 녹음 중 실시간 호출용.
    """
    input_path = work_dir / ".kw-input.txt"
    output_path = work_dir / ".kw-output.txt"

    input_path.write_text(text, encoding="utf-8")
    output_path.unlink(missing_ok=True)

    prompt = (
        f'"{input_path.name}"을 읽고 도메인 특화 키워드를 추출해줘. '
        "전문 용어, 고유명사, 기술 용어, 프로젝트명, 약어 등. "
        f'쉼표로 구분된 키워드 목록만 "{output_path.name}"에 저장. '
        "설명이나 부연 없이 키워드만."
    )

    ok, _ = _run_cli(
        [
            "codex", "exec",
            "--profile", "fast",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            prompt,
        ],
        timeout=timeout,
        cwd=str(work_dir),
    )

    keywords: list[str] = []
    if ok and output_path.exists():
        raw = output_path.read_text(encoding="utf-8").strip()
        keywords = [k.strip() for k in raw.replace("\n", ",").split(",") if k.strip()]

    input_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)

    return keywords


# ---------------------------------------------------------------
# Ollama 기반 교정/추출
# ---------------------------------------------------------------

def _correct_batch_ollama(
    batch_lines: list[str],
    batch_idx: int,
    model: str = _OLLAMA_DEFAULT_MODEL,
    timeout: int = 120,
) -> tuple[int, bool, list[str]]:
    """Ollama로 단일 배치 STT 교정."""
    text = "\n".join(batch_lines)
    original_count = len(batch_lines)

    prompt = (
        "다음은 한국어 STT 결과입니다. 오인식된 부분만 교정하세요.\n"
        "형식 '- [HH:MM:SS] [화자] 텍스트' 유지. 타임스탬프/화자 수정 금지.\n"
        "줄 삭제/추가 금지. 줄 수 반드시 유지.\n"
        "교정 결과만 출력:\n\n"
        f"{text}"
    )

    ok, result = _run_ollama(prompt, model, timeout)
    if not ok:
        print(f"[후처리] Ollama 배치 {batch_idx} 실패: {result[:200]}")
        return batch_idx, False, batch_lines

    result_lines = [line for line in result.strip().split("\n") if line.strip()]
    if len(result_lines) == original_count:
        return batch_idx, True, result_lines

    print(
        f"[후처리] Ollama 배치 {batch_idx}: 줄 수 불일치 "
        f"({original_count} → {len(result_lines)}), 원본 유지"
    )
    return batch_idx, False, batch_lines


def correct_with_ollama_parallel(
    md_path: Path,
    timeout: int = _DEFAULT_TIMEOUT,
    batch_size: int = 10,
    max_workers: int = 2,
    model: str = _OLLAMA_DEFAULT_MODEL,
    progress_callback: Callable[[float], None] | None = None,
) -> bool:
    """Ollama로 STT 교정 — 배치 병렬. CPU 추론이므로 워커 2개 기본."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    abs_path = md_path.resolve()
    backup_path = abs_path.with_suffix(".pre-polish.md")

    import shutil as _shutil
    _shutil.copy2(abs_path, backup_path)

    content = abs_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    # Raw Data 섹션에서 세그먼트 라인 인덱스 추출
    segment_indices: list[int] = []
    in_raw = False
    for i, line in enumerate(lines):
        if line.strip() == "# Raw Data":
            in_raw = True
            continue
        if in_raw and line.startswith("# "):
            break
        if in_raw and line.startswith("- ["):
            segment_indices.append(i)

    if not segment_indices:
        backup_path.unlink(missing_ok=True)
        return True

    # 배치 분할
    batches: list[list[tuple[int, str]]] = []
    for start in range(0, len(segment_indices), batch_size):
        batch = [(idx, lines[idx]) for idx in segment_indices[start:start + batch_size]]
        batches.append(batch)

    total_batches = len(batches)
    per_batch_timeout = max(120, timeout // max(total_batches, 1))
    print(
        f"[후처리] Ollama {model}: {len(segment_indices)}세그먼트 → "
        f"{total_batches}배치 (워커 {max_workers})"
    )

    completed_count = 0
    all_corrected: dict[int, str] = {}
    any_success = False

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for batch_idx, batch in enumerate(batches):
            batch_lines_text = [text_val for _, text_val in batch]
            future = pool.submit(
                _correct_batch_ollama, batch_lines_text, batch_idx,
                model, per_batch_timeout,
            )
            futures[future] = (batch_idx, batch)

        for future in as_completed(futures):
            batch_idx, batch = futures[future]
            try:
                _, ok, corrected = future.result()
                if ok:
                    any_success = True
                for j, (line_idx, _) in enumerate(batch):
                    if j < len(corrected):
                        all_corrected[line_idx] = corrected[j]
            except Exception as e:
                print(f"[후처리] Ollama 배치 {batch_idx} 예외: {e}")

            completed_count += 1
            pct = completed_count / total_batches * 100
            print(f"[후처리] Ollama 교정: {completed_count}/{total_batches} ({pct:.0f}%)")
            if progress_callback:
                progress_callback(pct)

    # 교정된 라인을 원본에 반영
    if all_corrected:
        for line_idx, corrected_text in all_corrected.items():
            lines[line_idx] = corrected_text
        new_content = "\n".join(lines)
        abs_path.write_text(new_content, encoding="utf-8")

        if new_content.count("\n") < content.count("\n") * 0.5:
            print("[후처리] 데이터 손실 감지. 백업에서 복원.")
            _shutil.copy2(backup_path, abs_path)
            any_success = False
        else:
            backup_path.unlink(missing_ok=True)
    else:
        _shutil.copy2(backup_path, abs_path)
        print("[후처리] 전체 실패 — 백업에서 원본 복원")

    return any_success


def extract_keywords_with_ollama(
    text: str,
    model: str = _OLLAMA_DEFAULT_MODEL,
    timeout: int = 30,
) -> list[str]:
    """Ollama로 도메인 키워드 추출. 파일 I/O 불필요 — 직접 HTTP 호출."""
    prompt = (
        "다음 회의 전사 텍스트에서 도메인 특화 키워드만 추출하세요.\n"
        "전문 용어, 고유명사, 기술 용어, 프로젝트명, 약어만.\n"
        "쉼표로 구분된 키워드 목록만 출력. 설명 없이.\n\n"
        f"{text}\n\n"
        "키워드:"
    )
    ok, result = _run_ollama(prompt, model, timeout)
    if not ok:
        return []
    return [k.strip() for k in result.replace("\n", ",").split(",") if k.strip()]


def summarize_with_ollama(
    md_path: Path,
    model: str = _OLLAMA_DEFAULT_MODEL,
    timeout: int = _DEFAULT_TIMEOUT,
) -> bool:
    """Ollama로 회의 요약 + To-do 생성."""
    abs_path = md_path.resolve()
    content = abs_path.read_text(encoding="utf-8")

    # Raw Data에서 처음 8000자만 추출 (컨텍스트 제한)
    raw_start = content.find("# Raw Data")
    if raw_start < 0:
        return False
    raw_data = content[raw_start:raw_start + 8000]

    prompt = (
        "다음 회의 녹취록을 분석하여 요약과 To-do를 생성하세요.\n\n"
        "출력 형식 (구분자 포함):\n"
        "=== 요약 ===\n## 주제\n...\n## 핵심 논의\n...\n## 결론\n...\n"
        "=== To-do ===\n- [ ] 항목1\n- [ ] 항목2\n\n"
        f"{raw_data}"
    )
    ok, result = _run_ollama(prompt, model, timeout)
    if not ok:
        print(f"[후처리] Ollama 요약 실패: {result[:200]}")
        return False

    # 요약/To-do 파싱
    summary_text = ""
    todo_text = ""
    if "=== 요약 ===" in result:
        parts = result.split("=== To-do ===")
        summary_text = parts[0].replace("=== 요약 ===", "").strip()
        if len(parts) > 1:
            todo_text = parts[1].strip()
    else:
        summary_text = result.strip()

    if summary_text:
        content = content.replace("- (요약은 회의 후 별도 작성)", summary_text)
    if todo_text:
        content = content.replace("- [ ] (회의 후 별도 작성)", todo_text)

    abs_path.write_text(content, encoding="utf-8")
    return True


# ---------------------------------------------------------------
# 통합 파이프라인
# ---------------------------------------------------------------

def polish_meeting(
    md_path: Path,
    timeout: int | None = None,
    segment_count: int = 0,
    progress_callback: Callable[[str, float], None] | None = None,
    use_ollama: bool = False,
    ollama_model: str | None = None,
) -> dict[str, bool]:
    """회의록 LLM 후처리 파이프라인.

    1단계: STT 교정 — Codex (기본) 또는 Ollama (--ollama / 폴백)
    2단계: 요약 — Gemini (기본) 또는 Ollama (폴백)

    Args:
        md_path: 회의록 파일 경로
        timeout: 직접 지정 타임아웃 (초). None이면 segment_count 기반 자동 계산.
        segment_count: 세그먼트 수 (타임아웃 자동 계산용)
        progress_callback: 진행 콜백 (phase, progress%) — 웹 UI 표시용
        use_ollama: True면 Ollama 우선 사용
        ollama_model: Ollama 모델명 (기본: gemma3:27b)

    Returns:
        {"corrected": bool, "summarized": bool}
    """
    from concurrent.futures import Future, ThreadPoolExecutor, as_completed

    if timeout is None:
        timeout = calc_timeout(segment_count)

    abs_path = md_path.resolve()
    _cleanup_orphaned_temp_files(abs_path.parent)
    model = ollama_model or _OLLAMA_DEFAULT_MODEL
    results: dict[str, bool] = {"corrected": False, "summarized": False}
    print(f"[후처리] 타임아웃: {timeout}초 ({segment_count}세그먼트 기준)")

    correction_available = False
    correction_mode = ""
    if use_ollama:
        if is_ollama_available(model):
            correction_available = True
            correction_mode = "ollama"
        else:
            print(f"[후처리] Ollama 미가용 ({model}) — STT 교정 건너뜀")
    else:
        if is_codex_available():
            correction_available = True
            correction_mode = "codex"
        elif is_ollama_available(model):
            correction_available = True
            correction_mode = "ollama"
            print(f"[후처리] Codex 미설치 → Ollama ({model}) 폴백...")
        else:
            print("[후처리] Codex/Ollama 모두 미가용 — STT 교정 건너뜀")

    summary_available = False
    summary_mode = ""
    if is_gemini_available():
        summary_available = True
        summary_mode = "gemini"
    elif is_ollama_available(model):
        summary_available = True
        summary_mode = "ollama"
    else:
        print("[후처리] Gemini/Ollama 미가용 — 요약 건너뜀")

    def _stt_progress(pct: float) -> None:
        if progress_callback:
            progress_callback("stt", pct)

    def _extract_summary_blocks(content: str) -> tuple[str | None, str | None]:
        summary_match = re.search(
            r"(?s)(# 회의 내용 요약.*?- 총 세그먼트: .*?\n)(.*?)(\n---\n\n# To-do)",
            content,
        )
        todo_match = re.search(
            r"(?s)(# To-do\n\n)(.*?)(\n\n---\n\n# 대화 정리)",
            content,
        )
        summary_block = summary_match.group(2) if summary_match else None
        todo_block = todo_match.group(2) if todo_match else None
        return summary_block, todo_block

    def _merge_summary_blocks(
        target_path: Path,
        summary_block: str | None,
        todo_block: str | None,
    ) -> bool:
        content = target_path.read_text(encoding="utf-8")
        updated = content
        changed = False

        if summary_block is not None:
            replaced, count = re.subn(
                r"(?s)(# 회의 내용 요약.*?- 총 세그먼트: .*?\n)(.*?)(\n---\n\n# To-do)",
                lambda match: str(match.group(1)) + summary_block + str(match.group(3)),
                updated,
                count=1,
            )
            if count:
                updated = replaced
                changed = True

        if todo_block is not None:
            replaced, count = re.subn(
                r"(?s)(# To-do\n\n)(.*?)(\n\n---\n\n# 대화 정리)",
                lambda match: str(match.group(1)) + todo_block + str(match.group(3)),
                updated,
                count=1,
            )
            if count:
                updated = replaced
                changed = True

        if changed:
            target_path.write_text(updated, encoding="utf-8")
        return changed

    def _do_correction() -> bool:
        if progress_callback:
            progress_callback("stt", 0)

        if correction_mode == "codex":
            print("[후처리] STT 교정 (Codex) 병렬 실행 중...")
            corrected_result = correct_with_codex_parallel(
                abs_path,
                timeout=timeout,
                progress_callback=_stt_progress,
            )
        else:
            print(f"[후처리] STT 교정 (Ollama {model}) 실행 중...")
            corrected_result = correct_with_ollama_parallel(
                abs_path,
                timeout=timeout,
                model=model,
                progress_callback=_stt_progress,
            )

        print(f"[후처리] STT 교정 — {'완료' if corrected_result else '실패'}")
        return corrected_result

    # 병렬 실행 전에 요약용 임시 복사본 생성 (Windows 파일 잠금 방지)
    _temp_summary_path: Path | None = None
    if summary_available and correction_available:
        _temp_summary_path = abs_path.with_suffix(".summary.tmp.md")
        shutil.copy2(abs_path, _temp_summary_path)

    def _do_summary() -> dict[str, str | bool | None]:
        if progress_callback:
            progress_callback("summary", 0)

        summary_path = _temp_summary_path if _temp_summary_path else abs_path

        try:
            if summary_mode == "gemini":
                print("[후처리] 요약 생성 (Gemini) 실행 중...")
                summarized_result = summarize_with_gemini(summary_path, timeout=timeout)
            else:
                print(f"[후처리] 요약 생성 (Ollama {model}) 실행 중...")
                summarized_result = summarize_with_ollama(
                    summary_path,
                    model=model,
                    timeout=timeout,
                )

            summary_block = None
            todo_block = None
            if summarized_result and _temp_summary_path is not None and _temp_summary_path.exists():
                summary_block, todo_block = _extract_summary_blocks(
                    _temp_summary_path.read_text(encoding="utf-8")
                )

            if progress_callback:
                progress_callback("summary", 100)
            print(f"[후처리] 요약 생성 — {'완료' if summarized_result else '실패'}")
            return {
                "ok": summarized_result,
                "summary_block": summary_block,
                "todo_block": todo_block,
            }
        finally:
            if _temp_summary_path is not None:
                _temp_summary_path.unlink(missing_ok=True)

    futures: dict[Future, str] = {}
    summary_result: dict[str, str | bool | None] | None = None
    with ThreadPoolExecutor(max_workers=2) as pool:
        if correction_available:
            futures[pool.submit(_do_correction)] = "corrected"
        if summary_available:
            futures[pool.submit(_do_summary)] = "summarized"

        for future in as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
                if key == "summarized":
                    summary_result = result
                else:
                    results[key] = bool(result)
            except Exception as e:
                print(f"[후처리] {key} 실패: {e}")

    if summary_result is not None:
        summary_ok = bool(summary_result.get("ok"))
        if summary_ok and correction_available:
            summary_blk = summary_result.get("summary_block")
            todo_blk = summary_result.get("todo_block")
            results["summarized"] = _merge_summary_blocks(
                abs_path,
                str(summary_blk) if summary_blk is not None else None,
                str(todo_blk) if todo_blk is not None else None,
            )
        else:
            results["summarized"] = summary_ok

    if progress_callback:
        progress_callback("done", 100)

    return results
