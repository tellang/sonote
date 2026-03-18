"""
입력 검증 유틸리티 — Agent DX Safety Rails

AI 에이전트가 전달하는 입력에서 제어문자, 위험한 유니코드,
경로 탈출(path traversal) 등을 차단한다.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "reject_control_chars",
    "reject_dangerous_unicode",
    "sanitize_input",
    "ValidationError",
]


class ValidationError(ValueError):
    """입력 검증 실패 시 발생하는 예외."""

    def __init__(self, message: str, field: str = "", code: str = "INVALID_INPUT"):
        super().__init__(message)
        self.field = field
        self.code = code


# 허용: 탭(\t), 개행(\n), 캐리지리턴(\r)
_ALLOWED_CONTROL = {"\t", "\n", "\r"}

# 위험한 유니코드 카테고리 (BiDi 오버라이드, 서로게이트 등)
_DANGEROUS_CATEGORIES = {"Cc", "Cf", "Cs", "Co"}
# Cf 중 허용할 코드포인트 (BOM, 소프트 하이픈 등은 상황에 따라 허용)
_ALLOWED_CF = {
    "\ufeff",  # BOM
    "\u00ad",  # 소프트 하이픈
}

# BiDi 오버라이드 문자 (명시적 차단)
_BIDI_OVERRIDES = {
    "\u202a",  # LRE
    "\u202b",  # RLE
    "\u202c",  # PDF
    "\u202d",  # LRO
    "\u202e",  # RLO
    "\u2066",  # LRI
    "\u2067",  # RLI
    "\u2068",  # FSI
    "\u2069",  # PDI
    "\u200f",  # RLM
    "\u200e",  # LRM
}

# 경로 탈출 패턴
_PATH_TRAVERSAL_RE = re.compile(r"\.\.[/\\]")


def reject_control_chars(value: str, *, field: str = "") -> str:
    """제어문자(탭/개행/CR 제외)가 포함된 문자열을 거부한다."""
    for ch in value:
        if unicodedata.category(ch) == "Cc" and ch not in _ALLOWED_CONTROL:
            raise ValidationError(
                f"제어문자 감지 (U+{ord(ch):04X})",
                field=field,
                code="CONTROL_CHAR",
            )
    return value


def reject_dangerous_unicode(value: str, *, field: str = "") -> str:
    """BiDi 오버라이드, 서로게이트 등 위험한 유니코드를 거부한다."""
    for ch in value:
        if ch in _BIDI_OVERRIDES:
            raise ValidationError(
                f"BiDi 오버라이드 문자 감지 (U+{ord(ch):04X})",
                field=field,
                code="BIDI_OVERRIDE",
            )
        cat = unicodedata.category(ch)
        if cat in ("Cs", "Co"):
            raise ValidationError(
                f"위험한 유니코드 감지 (U+{ord(ch):04X}, 카테고리={cat})",
                field=field,
                code="DANGEROUS_UNICODE",
            )
    return value


def reject_path_traversal(value: str, *, field: str = "") -> str:
    """경로 탈출(../ 또는 ..\\) 패턴을 거부한다."""
    if _PATH_TRAVERSAL_RE.search(value):
        raise ValidationError(
            "경로 탈출 패턴 감지 (../)",
            field=field,
            code="PATH_TRAVERSAL",
        )
    return value


def sanitize_input(value: str, *, field: str = "") -> str:
    """제어문자 + 위험 유니코드 + 경로 탈출을 한 번에 검증한다."""
    reject_control_chars(value, field=field)
    reject_dangerous_unicode(value, field=field)
    reject_path_traversal(value, field=field)
    return value
