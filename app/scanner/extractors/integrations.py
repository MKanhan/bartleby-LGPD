"""Extractors for vector stores, memory stores, and telemetry libraries.

Matching is driven by which libraries are imported in the file, not by bare
symbol names — "Client" alone is ambiguous between LangSmith, Chroma, Qdrant,
and a dozen others. The scanner tracks `from X import Y` bindings and only
classifies calls whose library of origin is a known LGPD-relevant integration.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from app.scanner.schema import Operation

# (module_prefix, symbol) → (kind, provider)
_LIBRARY_SYMBOLS: dict[tuple[str, str], tuple[str, str]] = {
    # vector stores
    ("pinecone", "Pinecone"): ("vector_store_init", "pinecone"),
    ("weaviate", "Client"): ("vector_store_init", "weaviate"),
    ("qdrant_client", "QdrantClient"): ("vector_store_init", "qdrant"),
    ("chromadb", "Client"): ("vector_store_init", "chroma"),
    ("chromadb", "PersistentClient"): ("vector_store_init", "chroma"),
    ("faiss", "IndexFlatL2"): ("vector_store_init", "faiss"),
    ("faiss", "IndexFlatIP"): ("vector_store_init", "faiss"),
    # memory stores
    ("redis", "Redis"): ("memory_persist", "redis"),
    ("redis", "StrictRedis"): ("memory_persist", "redis"),
    ("redis.asyncio", "Redis"): ("memory_persist", "redis_asyncio"),
    ("langchain.memory", "ConversationBufferMemory"): ("memory_persist", "langchain"),
    ("langchain_community.chat_message_histories", "RedisChatMessageHistory"): (
        "memory_persist",
        "langchain_redis",
    ),
    # telemetry
    ("langsmith", "Client"): ("telemetry", "langsmith"),
    ("langfuse", "Langfuse"): ("telemetry", "langfuse"),
    ("langsmith.run_helpers", "traceable"): ("telemetry", "langsmith"),
    ("langfuse.decorators", "observe"): ("telemetry", "langfuse"),
}

# method_name → (kind, provider_guess_when_library_known)
_METHOD_OPERATIONS: dict[str, tuple[str, str]] = {
    "upsert": ("vector_store_write", "vector_store"),
    "query": ("vector_store_read", "vector_store"),
    "search": ("vector_store_read", "vector_store"),
    "add": ("vector_store_write", "vector_store"),
    "hset": ("memory_persist", "redis"),
    "hget": ("memory_persist", "redis"),
    "set": ("memory_persist", "redis"),
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


def _collect_bindings(tree: ast.AST) -> tuple[dict[str, str], dict[str, str]]:
    """Return (name_to_library, call_target_to_library) maps for this module.

    - name_to_library: `from redis import Redis as R` → {"R": "redis"}
    - aliased_modules: `import redis` → {"redis": "redis"}; `import redis as r` → {"r": "redis"}
    """
    symbol_library: dict[str, str] = {}
    module_alias: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imported_name = alias.asname or alias.name
                symbol_library[imported_name] = module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                alias_name = alias.asname or alias.name.split(".")[0]
                module_alias[alias_name] = alias.name.split(".")[0]

    return symbol_library, module_alias


def _classify_call(
    call: ast.Call,
    symbol_library: dict[str, str],
    module_alias: dict[str, str],
) -> tuple[str, str, str] | None:
    """Return (category, kind, provider) for a call, or None.

    category ∈ {"vector", "memory", "telemetry"}.
    """
    func = call.func
    if isinstance(func, ast.Name):
        module = symbol_library.get(func.id)
        if module is None:
            return None
        hit = _LIBRARY_SYMBOLS.get((module, func.id))
        if hit is None:
            return None
        kind, provider = hit
        category = _category_of_kind(kind)
        return (category, kind, provider)

    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        mod_alias = module_alias.get(func.value.id)
        if mod_alias is not None:
            hit = _LIBRARY_SYMBOLS.get((mod_alias, func.attr))
            if hit is not None:
                kind, provider = hit
                return (_category_of_kind(kind), kind, provider)

    return None


def _category_of_kind(kind: str) -> str:
    if kind.startswith("vector"):
        return "vector"
    if kind.startswith("memory"):
        return "memory"
    return "telemetry"


def _method_call_for_known_receiver(
    call: ast.Call,
    receiver_origins: dict[str, str],
) -> tuple[str, str, str] | None:
    """Classify a `receiver.method(...)` call when we know the receiver's library."""
    func = call.func
    if not isinstance(func, ast.Attribute):
        return None
    receiver_name = None
    if isinstance(func.value, ast.Name):
        receiver_name = func.value.id
    elif isinstance(func.value, ast.Attribute) and isinstance(func.value.value, ast.Name):
        receiver_name = func.value.value.id
    if receiver_name is None:
        return None
    origin = receiver_origins.get(receiver_name)
    if origin is None:
        return None
    method_hit = _METHOD_OPERATIONS.get(func.attr)
    if method_hit is None:
        return None
    kind, fallback_provider = method_hit
    provider = origin
    return (_category_of_kind(kind), kind, provider)


def _learn_assignments(
    tree: ast.AST,
    symbol_library: dict[str, str],
    module_alias: dict[str, str],
) -> dict[str, str]:
    """Map local variable names to a provider, propagating through chained calls."""
    origins: dict[str, str] = {}

    assigns = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]

    for _ in range(3):  # fixed-point iterations are cheap on small modules
        grew = False
        for node in assigns:
            provider = _infer_rhs_provider(node.value, symbol_library, module_alias, origins)
            if provider is None:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and origins.get(target.id) != provider:
                    origins[target.id] = provider
                    grew = True
        if not grew:
            break

    return origins


def _infer_rhs_provider(
    node: ast.AST,
    symbol_library: dict[str, str],
    module_alias: dict[str, str],
    origins: dict[str, str],
) -> str | None:
    if isinstance(node, ast.Call):
        direct = _classify_call(node, symbol_library, module_alias)
        if direct is not None:
            return direct[2]
        func = node.func
        if isinstance(func, ast.Attribute):
            base = _base_name(func)
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
    if isinstance(cur, ast.Name):
        return cur.id
    return None


def analyze_integrations(path: Path, root: Path) -> _IntegrationFacts:
    src = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return _IntegrationFacts()

    rel = path.relative_to(root).as_posix()
    facts = _IntegrationFacts()

    symbol_library, module_alias = _collect_bindings(tree)
    receiver_origins = _learn_assignments(tree, symbol_library, module_alias)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        direct = _classify_call(node, symbol_library, module_alias)
        if direct is not None:
            category, kind, provider = direct
        else:
            indirect = _method_call_for_known_receiver(node, receiver_origins)
            if indirect is None:
                continue
            category, kind, provider = indirect

        op = Operation(
            kind=kind,
            provider=provider,
            file=rel,
            line=node.lineno,
            extra={"symbol": _attr_dotted(node.func) or "?"},
        )
        if category == "vector":
            facts.vector_ops.append(op)
        elif category == "memory":
            facts.memory_ops.append(op)
        else:
            facts.telemetry_ops.append(op)

    return facts
