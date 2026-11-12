"""Heuristic agent-framework identification from manifest + imports (Python and, since spec 69,
JS/TS: package.json and import specifiers)."""

from __future__ import annotations

import re
from pathlib import Path

# A quoted JS package name matches both `import ... from "x"` and a package.json dependency.
_JS_PKG = r"""['"]{}['"/]"""

_SIGNATURES: dict[str, list[str]] = {
    "langgraph": [
        r"\bfrom\s+langgraph\b",
        r"\blanggraph\s*[=><~]",
        _JS_PKG.format("@langchain/langgraph"),
    ],
    "langchain": [
        r"\bfrom\s+langchain(_[a-z]+)?\b",
        r"\blangchain[-_]",
        _JS_PKG.format("@langchain/core"),
        r"""['"]langchain/""",
    ],
    "crewai": [r"\bfrom\s+crewai\b", r"\bcrewai\s*[=><~]"],
    "pydantic_ai": [r"\bfrom\s+pydantic_ai\b"],
    "haystack": [r"\bfrom\s+haystack\b", r"\bhaystack-ai\b"],
    "llamaindex": [
        r"\bfrom\s+llama_index\b",
        _JS_PKG.format("llamaindex"),
        r"""['"]@llamaindex/""",
    ],
    "mastra": [r"""['"]@mastra/"""],
    "openai_agents": [_JS_PKG.format("@openai/agents")],
    "vercel_ai_sdk": [_JS_PKG.format("ai"), r"""['"]@ai-sdk/"""],
    "anthropic_sdk": [r"\bfrom\s+anthropic\b", r"^anthropic\s*[=><~]"],
    "openai_sdk": [r"\bfrom\s+openai\b", r"^openai\s*[=><~]"],
    "google_sdk": [r"\bfrom\s+google\.genai\b", r"\bgoogle-genai\b"],
}

_SKIP_DIRS = {".git", "node_modules", ".next", "dist", "build", ".venv", "venv"}


def _iter_relevant_files(root: Path) -> list[Path]:
    files: list[Path] = []
    patterns = (
        "*.py",
        "pyproject.toml",
        "requirements*.txt",
        "setup.py",
        "Pipfile",
        "package.json",
        "*.ts",
        "*.tsx",
        "*.js",
        "*.mjs",
    )
    for pattern in patterns:
        files.extend(
            p
            for p in root.rglob(pattern)
            if not any(part in _SKIP_DIRS for part in p.relative_to(root).parts)
        )
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
    priority = [
        "langgraph",
        "crewai",
        "pydantic_ai",
        "mastra",
        "openai_agents",
        "haystack",
        "llamaindex",
        "langchain",
        "vercel_ai_sdk",
    ]
    for framework in priority:
        if scores[framework] > 0:
            return (framework, min(1.0, scores[framework] / max(total_hits, 1) + 0.25))

    best = max(scores.items(), key=lambda kv: kv[1])
    if best[1] == 0:
        return ("custom", 0.3)
    return (best[0], min(1.0, best[1] / total_hits))
