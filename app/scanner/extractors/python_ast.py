"""Python AST extractors — entry points, tool definitions, external HTTP calls.

LLM calls moved to `llm_calls.py` (spec 65): the provider comes from evidence in the code, never
from a method name. `analyze_python_file` still returns them for a single file, which is what the
unit tests exercise; the orchestrator runs the repository-wide analysis instead.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from app.scanner.schema import Entrypoint, Operation, ToolDef

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


@dataclass
class _FileFacts:
    entrypoints: list[Entrypoint] = field(default_factory=list)
    operations: list[Operation] = field(default_factory=list)
    tools: list[ToolDef] = field(default_factory=list)


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


def analyze_python_file(path: Path, root: Path, *, include_llm: bool = True) -> _FileFacts:
    source = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return _FileFacts()

    rel = path.relative_to(root).as_posix()
    facts = _FileFacts()

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

    if include_llm:
        from app.scanner.extractors.llm_calls import analyze_repo
        from app.scanner.py_index import RepoIndex

        facts.operations.extend(analyze_repo(RepoIndex.build(root, [path])))
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
