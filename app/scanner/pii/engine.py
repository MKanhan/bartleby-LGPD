"""Lightweight PII scanner — regex + Brazilian checksum validation + context.

Presidio is a heavy dependency (spaCy models, gigabytes of disk). For the MVP we
ship a regex + validator engine that handles every Brazilian identifier the
catalog requires. A Presidio-backed engine can be added later behind the same
interface if the customer needs broader language support.

Sprint 22 adds a contextual gate (see ``context_rules.classify_finding``) that
classifies each match as high/low/discarded, eliminating regex literals and
test-fixture noise from the top-level RIPD section.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.scanner.pii import br_recognizers
from app.scanner.pii.context_rules import classify_finding

_CPF_RE = re.compile(r"(?<!\d)\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?!\d)")
_CNPJ_RE = re.compile(r"(?<!\d)\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}(?!\d)")
_CNH_RE = re.compile(r"(?<!\d)\d{11}(?!\d)")
_TITULO_RE = re.compile(r"(?<!\d)\d{12}(?!\d)")
_CEP_RE = re.compile(r"(?<!\d)\d{5}-\d{3}(?!\d)")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"\+?\d{2}\s?\(?\d{2}\)?\s?9?\d{4}-?\d{4}")

_TEXT_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".md", ".yml", ".yaml", ".json", ".env", ".txt"}
_IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "evals",
    "snapshots",
    "__snapshots__",
    "cassettes",
    "recordings",
    "coverage",
    "htmlcov",
}


@dataclass
class PIIFinding:
    kind: str
    confidence: float
    file: str
    line: int
    snippet: str
    confidence_tier: Literal["high", "low"] = "high"
    context_signals: list[str] = field(default_factory=list)


def _mask_cpf(v: str) -> str:
    return "***.***.***-" + v[-2:] if len(v) >= 2 else "***"


def _mask_cnpj(v: str) -> str:
    return "**.***.***/****-" + v[-2:] if len(v) >= 2 else "****"


def _mask_number(v: str) -> str:
    return v[:2] + "…" + v[-2:] if len(v) > 4 else "…"


def _iter_text_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in _IGNORED_DIRS for part in rel.parts):
            continue
        if p.suffix in _TEXT_EXTS or p.name in {".env", ".env.example"}:
            out.append(p)
    return out


def _line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


_CANDIDATES: tuple[tuple[str, re.Pattern[str], bool, float], ...] = (
    ("cpf", _CPF_RE, True, 0.95),
    ("cnpj", _CNPJ_RE, True, 0.95),
    ("cnh", _CNH_RE, True, 0.85),
    ("titulo_eleitor", _TITULO_RE, True, 0.85),
    ("cep", _CEP_RE, False, 0.7),
    ("email", _EMAIL_RE, False, 0.9),
    ("phone", _PHONE_RE, False, 0.7),
)

_VALIDATORS: dict[str, Callable[[str], bool]] = {
    "cpf": br_recognizers.is_valid_cpf,
    "cnpj": br_recognizers.is_valid_cnpj,
    "cnh": br_recognizers.is_valid_cnh,
    "titulo_eleitor": br_recognizers.is_valid_titulo_eleitor,
}

_MASKERS: dict[str, Callable[[str], str]] = {
    "cpf": _mask_cpf,
    "cnpj": _mask_cnpj,
    "cnh": _mask_number,
    "titulo_eleitor": _mask_number,
    "cep": lambda v: v,
    "email": lambda v: "[email]",
    "phone": lambda v: "[phone]",
}


def scan_directory_for_pii(root: Path, require_context: bool = True) -> list[PIIFinding]:
    """Scan a directory for PII literals.

    Args:
        root: directory to walk.
        require_context: when True (default), each match is classified by
            ``context_rules.classify_finding``; matches with tier "discarded"
            are dropped, "low" findings are tagged for the muted section, and
            "high" findings populate the main RIPD section. Pass False to
            restore the legacy pre-S22 behavior (every match becomes high).
    """
    root = Path(root)
    findings: list[PIIFinding] = []
    for path in _iter_text_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = path.relative_to(root).as_posix()

        for kind, regex, has_checksum, confidence in _CANDIDATES:
            validator = _VALIDATORS.get(kind)
            masker = _MASKERS[kind]
            for m in regex.finditer(text):
                value = m.group()
                if validator is not None and not validator(value):
                    continue
                line_no = _line_of(text, m.start())
                if require_context:
                    tier, signals = classify_finding(
                        text=text,
                        line_no=line_no,
                        file_path=rel,
                        kind=kind,
                        has_checksum=has_checksum,
                    )
                    if tier == "discarded":
                        continue
                else:
                    tier, signals = "high", []
                findings.append(
                    PIIFinding(
                        kind=kind,
                        confidence=confidence,
                        file=rel,
                        line=line_no,
                        snippet=masker(value),
                        confidence_tier=tier,
                        context_signals=signals,
                    )
                )

    return findings
