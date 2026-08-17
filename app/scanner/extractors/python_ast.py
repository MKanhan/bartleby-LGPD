"""Python AST extractors — entry points, LLM calls, tool definitions."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from app.scanner.schema import Entrypoint, ExternalCall, Operation, ToolDef

# name_of_call_target → (operation_kind, provider_guess)
_CALL_TARGETS: dict[str, tuple[str, str]] = {
    # Anthropic SDK
    "messages.create": ("invoke_llm", "anthropic"),
    "messages.stream": ("invoke_llm", "anthropic"),
    # OpenAI SDK
    "chat.completions.create": ("invoke_llm", "openai"),
    "responses.create": ("invoke_llm", "openai"),
    "embeddings.create": ("embed", "openai"),
    "images.generate": ("invoke_llm", "openai"),
    # Google GenAI
    "generate_content": ("invoke_llm", "google"),
    "embed_content": ("embed", "google"),
    # LangChain runnable interface
    "invoke": ("invoke_llm", "langchain"),
    "ainvoke": ("invoke_llm", "langchain"),
    "stream": ("invoke_llm", "langchain"),
    "astream": ("invoke_llm", "langchain"),
    # LangChain embeddings
    "embed_query": ("embed", "langchain"),
    "embed_documents": ("embed", "langchain"),
    "aembed_query": ("embed", "langchain"),
    "aembed_documents": ("embed", "langchain"),
}

_ENTRY_DECORATORS: dict[str, str] = {
    "app.post": "fastapi",
    "app.get": "fastapi",
    "app.put": "fastapi",
    "app.delete": "fastapi",
    "app.patch": "fastapi",
    "router.post": "fastapi",
    "router.get": "fastapi",
    "router.put": "fastapi",
    "router.delete": "fastapi",
    "router.patch": "fastapi",
    "app.route": "flask",
    "app.command": "typer",
    "app.callback": "typer",
    "app.task": "celery",
    "shared_task": "celery",
}

_LAMBDA_HANDLERS = {"lambda_handler", "handler"}

_HTTP_LIBS = {"httpx", "requests", "urllib"}


@dataclass
class _FileFacts:
    entrypoints: list[Entrypoint] = field(default_factory=list)
    operations: list[Operation] = field(default_factory=list)
    tools: list[ToolDef] = field(default_factory=list)
    external_calls: list[ExternalCall] = field(default_factory=list)


def _attr_dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    cur: ast.AST | None = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    elif isinstance(cur, ast.Call):
        inner = _attr_dotted(cur.func)
        if inner:
            parts.append(inner)
    else:
        return None
    return ".".join(reversed(parts))


def _literal_kwarg(call: ast.Call, name: str) -> str | None:
    for kw in call.keywords:
        if (
            kw.arg == name
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            return kw.value.value
    return None


def _ends_with(dotted: str, target: str) -> bool:
    if dotted == target:
        return True
    return dotted.endswith("." + target)


def _snippet(src_lines: list[str], line: int, span: int = 1) -> str:
    start = max(line - 1, 0)
    end = min(line + span - 1, len(src_lines))
    return "\n".join(src_lines[start:end]).strip()


def _build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    """Map child node id() → parent node, for context lookup during a single walk."""
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    return parents


def _enclosing_function(node: ast.AST, parents: dict[int, ast.AST]) -> str | None:
    cur: ast.AST | None = parents.get(id(node))
    while cur is not None:
        if isinstance(cur, ast.FunctionDef | ast.AsyncFunctionDef):
            return cur.name
        cur = parents.get(id(cur))
    return None


def _return_var(call: ast.Call, parents: dict[int, ast.AST]) -> str | None:
    """Variable name when `Call` is the value of an Assign/AnnAssign (Await unwrapped)."""
    parent = parents.get(id(call))
    if isinstance(parent, ast.Await):
        parent = parents.get(id(parent))
    if isinstance(parent, ast.Assign) and parent.targets:
        target = parent.targets[0]
        if isinstance(target, ast.Name):
            return target.id
        if isinstance(target, ast.Attribute):
            return target.attr
        if isinstance(target, ast.Tuple) and target.elts:
            first = target.elts[0]
            if isinstance(first, ast.Name):
                return first.id
    if isinstance(parent, ast.AnnAssign) and isinstance(parent.target, ast.Name):
        return parent.target.id
    return None


def _prev_comment(src_lines: list[str], line: int) -> str | None:
    """Strip-and-return the closest `#` comment within 1-2 lines before `line` (1-based)."""
    for offset in (2, 3):
        idx = line - offset
        if 0 <= idx < len(src_lines):
            stripped = src_lines[idx].strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip() or None
            if stripped:
                return None
    return None


def _extract_tool_definition(node: ast.FunctionDef | ast.AsyncFunctionDef) -> ToolDef | None:
    for deco in node.decorator_list:
        dotted = _attr_dotted(deco if not isinstance(deco, ast.Call) else deco.func)
        if dotted and _ends_with(dotted, "tool"):
            description = ast.get_docstring(node) or ""
            params = [a.arg for a in node.args.args if a.arg != "self"]
            return ToolDef(
                name=node.name,
                description=description,
                parameters=params,
                file="",
                line=node.lineno,
            )
    return None


def _decorator_matches_entrypoint(deco: ast.AST) -> tuple[str, str] | None:
    call_target = deco.func if isinstance(deco, ast.Call) else deco
    dotted = _attr_dotted(call_target)
    if not dotted:
        return None
    for target, kind in _ENTRY_DECORATORS.items():
        if _ends_with(dotted, target):
            return (kind, target)
    return None


def _infer_http_call(call: ast.Call) -> tuple[str, str] | None:
    dotted = _attr_dotted(call.func)
    if not dotted:
        return None
    for lib in _HTTP_LIBS:
        if lib in dotted:
            last = dotted.split(".")[-1]
            if last in {"get", "post", "put", "delete", "patch", "request"}:
                return (last.upper(), lib)
    return None


def _url_arg(call: ast.Call) -> str | None:
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value
    for kw in call.keywords:
        if kw.arg == "url" and isinstance(kw.value, ast.Constant):
            return str(kw.value.value)
    return None


def analyze_python_file(path: Path, root: Path) -> _FileFacts:
    source = path.read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return _FileFacts()

    rel = path.relative_to(root).as_posix()
    facts = _FileFacts()
    parents = _build_parent_map(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            tool = _extract_tool_definition(node)
            if tool is not None:
                tool.file = rel
                facts.tools.append(tool)
            for deco in node.decorator_list:
                match = _decorator_matches_entrypoint(deco)
                if match is not None:
                    kind, rule = match
                    facts.entrypoints.append(
                        Entrypoint(
                            kind=kind,
                            name=node.name,
                            file=rel,
                            line=node.lineno,
                            extra={"decorator": rule},
                        )
                    )
            if node.name in _LAMBDA_HANDLERS and _has_lambda_signature(node):
                facts.entrypoints.append(
                    Entrypoint(kind="aws_lambda", name=node.name, file=rel, line=node.lineno)
                )
        elif isinstance(node, ast.If) and _is_main_guard(node):
            facts.entrypoints.append(
                Entrypoint(kind="script", name="__main__", file=rel, line=node.lineno)
            )
        elif isinstance(node, ast.Call):
            dotted = _attr_dotted(node.func)
            if not dotted:
                continue
            for target, (kind, provider) in _CALL_TARGETS.items():
                if _ends_with(dotted, target):
                    model = _literal_kwarg(node, "model")
                    facts.operations.append(
                        Operation(
                            kind=kind,
                            provider=provider,
                            model=model,
                            file=rel,
                            line=node.lineno,
                            snippet=_snippet(lines, node.lineno),
                            enclosing_function=_enclosing_function(node, parents),
                            prev_comment=_prev_comment(lines, node.lineno),
                            return_var=_return_var(node, parents),
                        )
                    )
                    break
            http = _infer_http_call(node)
            if http is not None:
                method, lib = http
                facts.external_calls.append(
                    ExternalCall(
                        url=_url_arg(node),
                        method=method,
                        library=lib,
                        file=rel,
                        line=node.lineno,
                    )
                )

    return facts


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if not (isinstance(test.left, ast.Name) and test.left.id == "__name__"):
        return False
    return any(isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators)


def _has_lambda_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    names = [a.arg for a in node.args.args]
    return names[:2] == ["event", "context"]


def iter_python_files(root: Path) -> list[Path]:
    ignored = {".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build"}
    return [
        p
        for p in root.rglob("*.py")
        if not any(part in ignored for part in p.relative_to(root).parts)
    ]
