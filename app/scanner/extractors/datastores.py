"""Where the data is kept: databases and vector stores in Python code (spec 65).

Evidence, all looked up in armazenamentos.yml:
- an import that by itself proves the store (`import sqlite3`, `from psycopg_pool import ...`);
- a literal connection URL passed to a call (`create_engine("sqlite:///app.db")`), or the literal
  start of an f-string URL (`f"mysql+pymysql://{user}@..."`: the scheme is fixed even if the rest
  is not);
- Django's `DATABASES[...]["ENGINE"]`;
- a vector-store class of the catalog (`Chroma` from `langchain_chroma`), also when built by a
  class method (`Chroma.from_documents(...)`);
- a vector-store config dict (`{"vector_store": {"provider": "pgvector"}}`, as mem0 uses).

Relational and document stores become `datastore` operations (provider None, `extra.store`);
vector stores become `vector_store_init` with the store as provider, like the existing extractor.
One operation per (file, store).
"""

from __future__ import annotations

import ast

from app.scanner import store_catalog as sc
from app.scanner.extractors.py_bindings import split_module_name
from app.scanner.extractors.py_values import resolve
from app.scanner.py_index import PyModule, RepoIndex
from app.scanner.schema import Operation


def _op(mod: PyModule, store: sc.Store, line: int, origin: str) -> Operation:
    snippet = mod.lines[line - 1].strip() if 0 < line <= len(mod.lines) else ""
    if store.is_vector:
        return Operation(
            kind="vector_store_init",
            provider=store.id,
            file=mod.rel,
            line=line,
            snippet=snippet,
            extra={"store": store.id, "store_origin": origin},
        )
    return Operation(
        kind="datastore",
        provider=None,
        file=mod.rel,
        line=line,
        snippet=snippet,
        extra={"store": store.id, "store_origin": origin},
    )


def analyze_module(mod: PyModule, index: RepoIndex) -> list[Operation]:
    found: dict[str, Operation] = {}

    def add(store: sc.Store | None, line: int, origin: str) -> None:
        if store is not None and store.id not in found:
            found[store.id] = _op(mod, store, line, origin)

    for node in ast.walk(mod.tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(sc.python_module(alias.name), node.lineno, f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            add(sc.python_module(node.module), node.lineno, f"from {node.module} import ...")
            for alias in node.names:  # from google.cloud import firestore
                submodule = f"{node.module}.{alias.name}"
                add(sc.python_module(submodule), node.lineno, f"import {submodule}")
        elif isinstance(node, ast.Call):
            qualified = mod.imports.qualify(node.func)
            if qualified:
                head, _, last = qualified.rpartition(".")
                if head and last in sc.load_catalog().construtores:
                    qualified = head
                module, name = split_module_name(qualified)
                add(sc.python_class(module, name), node.lineno, f"classe {qualified}")
                for store, cls in sc.python_conditional_classes(module, name):
                    value = _required_value(node, cls, mod, index)
                    if value is not None:
                        add(store, node.lineno, f"{qualified}({value!r})")
            for arg in [
                *node.args[:1],
                *(kw.value for kw in node.keywords if kw.arg in ("url", "dsn")),
            ]:
                value = _fstring_head(arg) or resolve(arg, mod, index).value
                if value and "://" in value:
                    add(sc.by_url(value), node.lineno, f"URL de conexão {value.split('://')[0]}://")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if (
                len(node.value) > 12
            ):  # spec 73: cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                add(sc.by_sql(node.value), node.lineno, "SQL em string do código")
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if not (isinstance(key, ast.Constant) and isinstance(value, ast.Constant)):
                    continue
                if key.value == "ENGINE" and isinstance(value.value, str):
                    add(
                        sc.by_django_engine(value.value),
                        node.lineno,
                        f"DATABASES ENGINE {value.value}",
                    )
                if (
                    key.value == "provider"
                    and isinstance(value.value, str)
                    and _inside_vector_config(node, mod)
                ):
                    store = sc.by_config_provider(value.value)
                    if store is not None and store.is_vector:
                        add(
                            store,
                            node.lineno,
                            f"configuração de vector store provider={value.value}",
                        )
    return list(found.values())


def _required_value(call: ast.Call, cls: sc.PyClass, mod: PyModule, index: RepoIndex) -> str | None:
    """The literal argument that makes this call the store, or None when it does not match."""
    found = None
    for arg_name, allowed in cls.requer.items():
        expr = next((kw.value for kw in call.keywords if kw.arg == arg_name), None)
        if expr is None and call.args:
            expr = call.args[0]
        value = resolve(expr, mod, index).value if expr is not None else None
        if value not in allowed:
            return None
        found = value
    return found


def _fstring_head(node: ast.expr) -> str | None:
    """The literal text an f-string starts with, up to its first placeholder."""
    if not isinstance(node, ast.JoinedStr):
        return None
    head = ""
    for part in node.values:
        if not (isinstance(part, ast.Constant) and isinstance(part.value, str)):
            break
        head += part.value
    return head or None


def _inside_vector_config(node: ast.Dict, mod: PyModule) -> bool:
    parent = mod.parents.get(id(node))
    if isinstance(parent, ast.Dict):
        for key, value in zip(parent.keys, parent.values, strict=True):
            if value is node and isinstance(key, ast.Constant) and key.value == "vector_store":
                return True
    return False


def analyze_repo(index: RepoIndex) -> list[Operation]:
    ops: list[Operation] = []
    for mod in index.modules.values():
        ops.extend(analyze_module(mod, index))
    return ops
