"""Extractors for vector stores, memory stores, and telemetry libraries (Python).

Matching is driven by which libraries are imported in the file, not by bare symbol names —
"Client" alone is ambiguous between LangSmith, Chroma, Qdrant and a dozen others. Since spec 69
the libraries come from the catalogs, never from a table in code: memory and telemetry from
`integracoes.yml`, vector stores from `armazenamentos.yml` (the same entries `datastores.py`
reads). A call to a name imported from a catalog module is the integration; a later
`receiver.method(...)` on what that call returned becomes a write/read operation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from app.scanner import integration_catalog as ic
from app.scanner import store_catalog as sc
from app.scanner.schema import Operation

# method on a known receiver -> (operation kind, category it is valid for)
_METHOD_OPERATIONS: dict[str, tuple[str, str]] = {
    "upsert": ("vector_store_write", "vector"),
    "add": ("vector_store_write", "vector"),
    "add_documents": ("vector_store_write", "vector"),
    "add_texts": ("vector_store_write", "vector"),
    "query": ("vector_store_read", "vector"),
    "search": ("vector_store_read", "vector"),
    "similarity_search": ("vector_store_read", "vector"),
    "hset": ("memory_persist", "memory"),
    "hget": ("memory_persist", "memory"),
    "set": ("memory_persist", "memory"),
}


@dataclass
class _IntegrationFacts:
    vector_ops: list[Operation] = field(default_factory=list)
    memory_ops: list[Operation] = field(default_factory=list)
    telemetry_ops: list[Operation] = field(default_factory=list)


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


def _collect_bindings(tree: ast.AST) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """(imported name -> (module, original name), module alias -> module).

    `from redis import Redis as R` -> {"R": ("redis", "Redis")}; `import redis.asyncio` binds
    `redis` (Python binds the top-level name); `import redis.asyncio as r` binds `r` to the full path.
    """
    symbols: dict[str, tuple[str, str]] = {}
    modules: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                symbols[alias.asname or alias.name] = (node.module, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    modules[alias.asname] = alias.name
                else:
                    top = alias.name.split(".")[0]
                    modules[top] = top
    return symbols, modules


def _category(kind: str) -> str:
    if kind.startswith("vector"):
        return "vector"
    return "memory" if kind == "memory_persist" else "telemetry"


def _resolve(
    func: ast.expr, symbols: dict[str, tuple[str, str]], modules: dict[str, str]
) -> tuple[str, str] | None:
    """(module, name) a call target comes from, when it comes from an import."""
    if isinstance(func, ast.Name):
        return symbols.get(func.id)
    dotted = _attr_dotted(func) if isinstance(func, ast.Attribute) else None
    if not dotted or "." not in dotted:
        return None
    head, *middle, name = dotted.split(".")
    if head in modules:
        return ".".join([modules[head], *middle]), name
    if head in symbols:  # from langchain_community import chat_message_histories as h
        module, original = symbols[head]
        return ".".join([module, original, *middle]), name
    return None


def _classify(
    call: ast.Call, symbols: dict[str, tuple[str, str]], modules: dict[str, str]
) -> tuple[str, str] | None:
    """(operation kind, integration or store id) for a call into a catalog library, or None."""
    target = _resolve(call.func, symbols, modules)
    if target is None:
        return None
    module, name = target
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        owner = symbols.get(call.func.value.id)
        if owner is not None:  # a class method builds the class: AsyncMemory.from_config(...)
            module, name = owner
    integration = ic.python_symbol(module, name)
    if integration is not None:
        return integration.kind, integration.id
    store = sc.python_class(module, name) or sc.python_module(module)
    if store is not None and store.is_vector:
        return "vector_store_init", store.id
    return None


def _learn_assignments(
    tree: ast.AST, symbols: dict[str, tuple[str, str]], modules: dict[str, str]
) -> dict[str, tuple[str, str]]:
    """Local name -> (kind, id) of the integration it holds, through chained calls."""
    origins: dict[str, tuple[str, str]] = {}
    assigns = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    for _ in range(3):  # fixed point; modules are small
        grew = False
        for node in assigns:
            hit = _rhs_origin(node.value, symbols, modules, origins)
            if hit is None:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and origins.get(target.id) != hit:
                    origins[target.id] = hit
                    grew = True
        if not grew:
            break
    return origins


def _rhs_origin(
    node: ast.AST,
    symbols: dict[str, tuple[str, str]],
    modules: dict[str, str],
    origins: dict[str, tuple[str, str]],
) -> tuple[str, str] | None:
    if isinstance(node, ast.Call):
        direct = _classify(node, symbols, modules)
        if direct is not None:
            return direct
        if isinstance(node.func, ast.Attribute):
            base = _base_name(node.func)
            if base is not None:
                return origins.get(base)
    elif isinstance(node, ast.Attribute):
        base = _base_name(node)
        if base is not None:
            return origins.get(base)
    return None


def _base_name(attr: ast.Attribute) -> str | None:
    cur: ast.AST = attr
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    return cur.id if isinstance(cur, ast.Name) else None


def _method_on_receiver(
    call: ast.Call, origins: dict[str, tuple[str, str]]
) -> tuple[str, str] | None:
    func = call.func
    if not isinstance(func, ast.Attribute):
        return None
    base = _base_name(func)
    origin = origins.get(base) if base else None
    method = _METHOD_OPERATIONS.get(func.attr)
    if origin is None or method is None:
        return None
    kind, valid_for = method
    if _category(origin[0]) != valid_for:
        return None
    return kind, origin[1]


def analyze_integrations(path: Path, root: Path) -> _IntegrationFacts:
    src = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return _IntegrationFacts()

    rel = path.relative_to(root).as_posix()
    facts = _IntegrationFacts()
    symbols, modules = _collect_bindings(tree)
    origins = _learn_assignments(tree, symbols, modules)

    def add(kind: str, provider: str, line: int, symbol: str) -> None:
        op = Operation(kind=kind, provider=provider, file=rel, line=line, extra={"symbol": symbol})
        category = _category(kind)
        if category == "vector":
            facts.vector_ops.append(op)
        elif category == "memory":
            facts.memory_ops.append(op)
        else:
            facts.telemetry_ops.append(op)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            hit = _classify(node, symbols, modules) or _method_on_receiver(node, origins)
            if hit is not None:
                add(hit[0], hit[1], node.lineno, _attr_dotted(node.func) or "?")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # an env switch the code reads or sets (os.environ["LANGSMITH_TRACING"] = "true")
            integration = ic.by_env(node.value)
            if integration is not None:
                add(integration.kind, integration.id, node.lineno, f"variável {node.value}")
    return facts
