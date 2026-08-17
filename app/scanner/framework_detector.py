"""Heuristic agent-framework identification from manifest + imports."""

from __future__ import annotations

import re
from pathlib import Path

_SIGNATURES: dict[str, list[str]] = {
    "langgraph": [r"\bfrom\s+langgraph\b", r"\blanggraph\s*[=><~]"],
    "langchain": [
        r"\bfrom\s+langchain(_[a-z]+)?\b",
        r"\blangchain[-_]",
    ],
    "crewai": [r"\bfrom\s+crewai\b", r"\bcrewai\s*[=><~]"],
    "pydantic_ai": [r"\bfrom\s+pydantic_ai\b"],
    "anthropic_sdk": [r"\bfrom\s+anthropic\b", r"^anthropic\s*[=><~]"],
    "openai_sdk": [r"\bfrom\s+openai\b", r"^openai\s*[=><~]"],
    "google_sdk": [r"\bfrom\s+google\.genai\b", r"\bgoogle-genai\b"],
}


def _iter_relevant_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in ("*.py", "pyproject.toml", "requirements*.txt", "setup.py", "Pipfile"):
        files.extend(p for p in root.rglob(pattern) if ".git" not in p.parts)
    return files


def detect_framework(root: Path) -> tuple[str, float]:
    """Return (framework_name, confidence in [0,1])."""
    scores: dict[str, int] = dict.fromkeys(_SIGNATURES.keys(), 0)
    total_hits = 0
    for path in _iter_relevant_files(Path(root)):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for framework, patterns in _SIGNATURES.items():
            for pat in patterns:
                hits = len(re.findall(pat, text, flags=re.MULTILINE))
                if hits:
                    scores[framework] += hits
                    total_hits += hits

    if total_hits == 0:
        return ("unknown", 0.0)

    # Prefer higher-level frameworks when present alongside SDKs they wrap.
    priority = ["langgraph", "crewai", "pydantic_ai", "langchain"]
    for framework in priority:
        if scores[framework] > 0:
            return (framework, min(1.0, scores[framework] / max(total_hits, 1) + 0.25))

    best = max(scores.items(), key=lambda kv: kv[1])
    if best[1] == 0:
        return ("custom", 0.3)
    return (best[0], min(1.0, best[1] / total_hits))
