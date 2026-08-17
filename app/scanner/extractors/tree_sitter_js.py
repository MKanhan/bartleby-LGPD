"""tree-sitter-based scanner for JavaScript and TypeScript agents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_javascript
import tree_sitter_typescript
from tree_sitter import Language, Parser

from app.scanner.schema import Entrypoint, Operation, ToolDef

_JS_LANG = Language(tree_sitter_javascript.language())
_TS_LANG = Language(tree_sitter_typescript.language_typescript())
_TSX_LANG = Language(tree_sitter_typescript.language_tsx())

_JS_EXTS = {".js", ".jsx", ".mjs", ".cjs"}
_TS_EXTS = {".ts"}
_TSX_EXTS = {".tsx"}

# Framework/library heuristics — matched against raw text.
_VERCEL_SIG = re.compile(r"\bfrom\s+['\"](ai|@ai-sdk/[^'\"]+)['\"]")
_ANTHROPIC_SIG = re.compile(r"\bfrom\s+['\"]@anthropic-ai/sdk['\"]")
_OPENAI_SIG = re.compile(r"\bfrom\s+['\"]openai['\"]")
_GOOGLE_SIG = re.compile(r"\bfrom\s+['\"]@google/genai['\"]")

_LLM_CALL_RE = re.compile(
    r"\b(generateText|streamText|generateObject|messages\.create|chat\.completions\.create|embed|embedMany|generate_content)\s*\("
)
_TOOL_RE = re.compile(r"\btool\s*\(\s*\{")
_NEXTJS_HANDLER_RE = re.compile(
    r"export\s+(?:async\s+)?function\s+(GET|POST|PUT|PATCH|DELETE|OPTIONS)\s*\("
)
_EXPRESS_ROUTE_RE = re.compile(r"\bapp\.(get|post|put|patch|delete|use)\s*\(\s*['\"]([^'\"]+)['\"]")


@dataclass
class _JsFacts:
    entrypoints: list[Entrypoint] = field(default_factory=list)
    operations: list[Operation] = field(default_factory=list)
    tools: list[ToolDef] = field(default_factory=list)


def _language_for(path: Path) -> Language | None:
    if path.suffix in _JS_EXTS:
        return _JS_LANG
    if path.suffix in _TS_EXTS:
        return _TS_LANG
    if path.suffix in _TSX_EXTS:
        return _TSX_LANG
    return None


def _parse(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _provider_hint(text: str) -> str | None:
    if _VERCEL_SIG.search(text):
        return "vercel_ai_sdk"
    if _ANTHROPIC_SIG.search(text):
        return "anthropic"
    if _OPENAI_SIG.search(text):
        return "openai"
    if _GOOGLE_SIG.search(text):
        return "google"
    return None


# S23 — best-effort context capture via regex. Without a real AST walk we can
# only spot the *nearest* function/const declaration before the match and a
# `//` comment on the immediately previous line.
_FUNC_DECL_RE = re.compile(
    r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=|(?:async\s+)?(\w+)\s*\([^)]*\)\s*=>)"
)


def _enclosing_function_js(text: str, char_idx: int) -> str | None:
    before = text[:char_idx]
    last: re.Match[str] | None = None
    for m in _FUNC_DECL_RE.finditer(before):
        last = m
    if last is None:
        return None
    return last.group(1) or last.group(2) or last.group(3)


def _prev_comment_js(text: str, char_idx: int) -> str | None:
    line_start = text.rfind("\n", 0, char_idx) + 1
    prev_block = text[:line_start].rstrip("\n")
    if not prev_block:
        return None
    prev_line = prev_block.rsplit("\n", 1)[-1].strip()
    if prev_line.startswith("//"):
        return prev_line.lstrip("/").strip() or None
    return None


def analyze_js_file(path: Path, root: Path) -> _JsFacts:
    language = _language_for(path)
    if language is None:
        return _JsFacts()
    text = _parse(path)

    # tree-sitter parse to confirm syntactic validity — we use regex for extraction
    # because the grammar varies heavily across TS/TSX variants. Parse failures
    # downgrade us to "no facts" rather than crashing the scanner.
    parser = Parser(language)
    try:
        parser.parse(text.encode())
    except Exception:
        return _JsFacts()

    rel = path.relative_to(root).as_posix()
    facts = _JsFacts()
    provider = _provider_hint(text)

    for m in _NEXTJS_HANDLER_RE.finditer(text):
        facts.entrypoints.append(
            Entrypoint(
                kind="nextjs_route",
                name=m.group(1),
                file=rel,
                line=_line_of(text, m.start()),
                extra={"method": m.group(1)},
            )
        )
    for m in _EXPRESS_ROUTE_RE.finditer(text):
        facts.entrypoints.append(
            Entrypoint(
                kind="express_route",
                name=m.group(2),
                file=rel,
                line=_line_of(text, m.start()),
                extra={"method": m.group(1).upper()},
            )
        )

    for m in _LLM_CALL_RE.finditer(text):
        symbol = m.group(1)
        kind = "embed" if "embed" in symbol.lower() else "invoke_llm"
        facts.operations.append(
            Operation(
                kind=kind,
                provider=provider,
                model=None,
                file=rel,
                line=_line_of(text, m.start()),
                snippet=symbol,
                extra={"symbol": symbol},
                enclosing_function=_enclosing_function_js(text, m.start()),
                prev_comment=_prev_comment_js(text, m.start()),
                # return_var requires real AST — best-effort regex would mis-match.
                return_var=None,
            )
        )

    for m in _TOOL_RE.finditer(text):
        facts.tools.append(
            ToolDef(
                name="tool(...)",
                description="",
                parameters=[],
                file=rel,
                line=_line_of(text, m.start()),
            )
        )

    return facts


def iter_js_files(root: Path) -> list[Path]:
    ignored = {".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build", ".next"}
    exts = _JS_EXTS | _TS_EXTS | _TSX_EXTS
    return [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in exts
        and not any(part in ignored for part in p.relative_to(root).parts)
    ]
