"""Inspect .env, dependency manifests, Dockerfile for config-level facts."""

from __future__ import annotations

import re
from pathlib import Path

_RETENTION_RE = re.compile(
    r"^\s*([A-Z0-9_]*(?:TTL|RETENTION|MAX_AGE|EXPIRE|EXPIRY|HISTORY_LIMIT)[A-Z0-9_]*)\s*[:=]\s*(\S+)",
    re.MULTILINE,
)
_PROVIDER_KEYS = [
    ("anthropic", re.compile(r"^\s*ANTHROPIC_API_KEY", re.MULTILINE)),
    ("openai", re.compile(r"^\s*OPENAI_API_KEY", re.MULTILINE)),
    (
        "google",
        re.compile(r"^\s*(GOOGLE_API_KEY|GEMINI_API_KEY|GOOGLE_GENAI_API_KEY)", re.MULTILINE),
    ),
    ("azure_openai", re.compile(r"^\s*AZURE_OPENAI", re.MULTILINE)),
    ("mistral", re.compile(r"^\s*MISTRAL_API_KEY", re.MULTILINE)),
]
_BASE_URL_RE = re.compile(
    r"^\s*([A-Z0-9_]*(?:BASE_URL|ENDPOINT)[A-Z0-9_]*)\s*[:=]\s*(\S+)",
    re.MULTILINE,
)
_NAMES_SCANNED = {
    ".env",
    ".env.example",
    ".env.sample",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "setup.py",
    "Pipfile",
    "Dockerfile",
    "package.json",
}


def scan_config(root: Path) -> dict:
    root = Path(root)
    findings: dict = {
        "providers_declared": [],
        "base_urls": [],
        "retention_hints": [],
        "files_seen": [],
    }
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.name not in _NAMES_SCANNED:
            continue
        rel = candidate.relative_to(root).as_posix()
        findings["files_seen"].append(rel)
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for provider, pattern in _PROVIDER_KEYS:
            if pattern.search(text) and provider not in findings["providers_declared"]:
                findings["providers_declared"].append(provider)
        for m in _BASE_URL_RE.finditer(text):
            findings["base_urls"].append(
                {"file": rel, "var": m.group(1), "value": m.group(2).strip("\"'")}
            )
        for m in _RETENTION_RE.finditer(text):
            findings["retention_hints"].append(
                {"file": rel, "var": m.group(1), "value": m.group(2).strip("\"'")}
            )
    return findings
