"""Context heuristics that classify PII matches as high/low/discarded.

The PII engine emits a finding only after a regex+checksum match. This module
adds the second gate: inspect the surrounding lines and the file path, then
decide whether the match is actually a real data treatment site or noise from
fixtures, regex literals or generic utility code.

Three outcomes:
- ``high``     — the match sits next to a keyword (variable name, comment,
                 label string) and there is no test-marker or test-path signal
                 around it. Renders in the main RIPD section.
- ``low``      — the keyword is present but compromised (test file path,
                 sample/example marker in comment, regex artifact in line) OR
                 there is no keyword but a checksum was validated. Renders in a
                 muted secondary section.
- ``discarded``— no keyword and either an explicit ruído signal (test file,
                 regex artifact, sample marker) or no checksum to anchor it.
                 The finding is dropped before reaching the report.
"""

from __future__ import annotations

import re
from typing import Literal

Tier = Literal["high", "low", "discarded"]

KEYWORDS_BY_KIND: dict[str, tuple[str, ...]] = {
    "cpf": ("cpf", "tax_id", "taxid", "documento"),
    "cnpj": ("cnpj", "tax_id", "taxid"),
    "cnh": ("cnh", "habilitacao", "habilitação", "driver_license", "drivers_license"),
    "titulo_eleitor": ("titulo", "título", "eleitor", "voter"),
    "pis": ("pis", "nis", "pasep"),
    "cep": ("cep", "zip", "postal", "endereco", "endereço", "address"),
    "email": ("email", "e-mail", "mail", "contato", "contact"),
    "phone": ("phone", "telefone", "celular", "mobile", "fone", "whatsapp"),
}

TEST_MARKERS: tuple[str, ...] = (
    "sample",
    "example",
    "exemplo",
    "fake",
    "dummy",
    "mock",
    "fixture",
    "placeholder",
)

TEST_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:^|/)tests?(?:/|$)"),
    re.compile(r"(?:^|/)fixtures?(?:/|$)"),
    re.compile(r"(?:^|/)test_[^/]+\.(?:py|js|jsx|ts|tsx)$"),
    re.compile(r"_test\.(?:py|js|jsx|ts|tsx)$"),
    re.compile(r"\.spec\.(?:js|jsx|ts|tsx)$"),
    re.compile(r"\.test\.(?:js|jsx|ts|tsx)$"),
    re.compile(r"(?:^|/)conftest\.py$"),
)

# Project-metadata paths (docs/, examples/, README*, setup.py, pyproject.toml,
# package.json, …). Matches downgrade to low: maintainer contact emails and
# placeholder sample data in these files are *real* PII literals but they are
# NOT data treated by the agent — they describe the project itself.
PROJECT_META_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:^|/)docs?(?:/|$)"),
    re.compile(r"(?:^|/)examples?(?:/|$)"),
    re.compile(r"(?:^|/)samples?(?:/|$)"),
    re.compile(r"(?:^|/)demos?(?:/|$)"),
    re.compile(
        r"(?:^|/)(?:README|LICENSE|CHANGELOG|CONTRIBUTING|"
        r"CODE_OF_CONDUCT|SECURITY|AUTHORS|MAINTAINERS|"
        r"HISTORY|NEWS|NOTICE)(?:\.[^/]*)?$",
        re.IGNORECASE,
    ),
    re.compile(r"(?:^|/)setup\.(?:py|cfg)$"),
    re.compile(r"(?:^|/)pyproject\.toml$"),
    re.compile(r"(?:^|/)package\.json$"),
)

# Same-line artifacts that suggest the match is a regex/pattern literal, not
# real personal data. The class character ``[0-9]``, an escape like ``\d`` or
# ``\.``, or the Python raw-string prefix ``r"`` / ``r'``.
_REGEX_ARTIFACT_RE = re.compile(r"""\\d|\\\.|\[0-9\]|\br['"]""")

_COMMENT_PREFIX_RE = re.compile(r"^\s*(?:#|//|/\*|\*)\s*(.+)$")
_WINDOW_BEFORE = 5
_WINDOW_AFTER = 5
_MARKER_WINDOW = 3


def _is_test_path(file_path: str) -> bool:
    norm = file_path.replace("\\", "/")
    return any(p.search(norm) for p in TEST_PATH_PATTERNS)


def _is_meta_path(file_path: str) -> bool:
    norm = file_path.replace("\\", "/")
    return any(p.search(norm) for p in PROJECT_META_PATH_PATTERNS)


def _line_has_var_marker(line: str) -> str | None:
    """Catch variable-name markers like ``sample_data = ...`` or ``EXAMPLE: ...``.

    A literal in a fixture-like assignment is fixture data, not real PII —
    this complements the comment marker check for source files that omit
    explanatory comments.
    """
    lower = line.lower()
    for marker in TEST_MARKERS:
        pattern = rf"(?<![a-z0-9])({re.escape(marker)})\w*\s*[=:]"
        if re.search(pattern, lower):
            return marker
    return None


def _line_has_keyword(line: str, keywords: tuple[str, ...]) -> str | None:
    """Match keyword with snake_case-friendly boundaries.

    Treats ``_`` and ``-`` as word separators so ``customer_cpf`` and
    ``user-email`` are detected; ``cpfcheck`` is rejected.
    """
    lower = line.lower()
    for kw in keywords:
        kw_norm = kw.lower()
        pattern = rf"(?<![a-z0-9]){re.escape(kw_norm)}(?![a-z0-9])"
        if re.search(pattern, lower):
            return kw
    return None


def _line_has_marker(line: str) -> str | None:
    m = _COMMENT_PREFIX_RE.match(line)
    target = (m.group(1) if m else line).lower()
    for marker in TEST_MARKERS:
        if re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", target):
            return marker
    return None


def classify_finding(
    text: str,
    line_no: int,
    file_path: str,
    kind: str,
    has_checksum: bool,
) -> tuple[Tier, list[str]]:
    """Classify a PII match as high/low/discarded with audit signals.

    Args:
        text: full file content (so the function can slice context).
        line_no: 1-based line number of the match.
        file_path: path relative to scan root (forward slashes preferred).
        kind: the PII category emitted by the recognizer.
        has_checksum: True for CPF/CNPJ/CNH/título/PIS (regex+validator),
            False for CEP/email/phone (regex-only).
    """
    signals: list[str] = []
    keywords = KEYWORDS_BY_KIND.get(kind, ())

    lines = text.split("\n")
    idx = max(0, line_no - 1)
    window_start = max(0, idx - _WINDOW_BEFORE)
    window_end = min(len(lines), idx + _WINDOW_AFTER + 1)
    window = lines[window_start:window_end]
    match_line = lines[idx] if idx < len(lines) else ""

    keyword_hit: str | None = None
    for w_line in window:
        kw = _line_has_keyword(w_line, keywords)
        if kw:
            keyword_hit = kw
            break
    if keyword_hit:
        signals.append(f"keyword:{keyword_hit}")

    marker_hit: str | None = None
    marker_signal_kind = "marker"
    marker_start = max(0, idx - _MARKER_WINDOW)
    marker_end = min(len(lines), idx + _MARKER_WINDOW + 1)
    for m_line in lines[marker_start:marker_end]:
        if _COMMENT_PREFIX_RE.match(m_line):
            mk = _line_has_marker(m_line)
            kind_sig = "marker"
        else:
            mk = _line_has_var_marker(m_line)
            kind_sig = "var_marker"
        if mk:
            marker_hit = mk
            marker_signal_kind = kind_sig
            break
    if marker_hit:
        signals.append(f"{marker_signal_kind}:{marker_hit}")

    test_path = _is_test_path(file_path)
    if test_path:
        signals.append("test_path")

    meta_path = _is_meta_path(file_path)
    if meta_path:
        signals.append("meta_path")

    regex_artifact = bool(_REGEX_ARTIFACT_RE.search(match_line))
    if regex_artifact:
        signals.append("regex_artifact")

    has_compromise = marker_hit is not None or test_path or meta_path or regex_artifact

    if keyword_hit and not has_compromise:
        return "high", signals
    if keyword_hit:
        return "low", signals
    if has_compromise:
        return "discarded", signals
    if has_checksum:
        signals.append("checksum_only")
        return "low", signals
    return "discarded", signals
