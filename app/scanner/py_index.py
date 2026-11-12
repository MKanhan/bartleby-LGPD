"""Repository-wide index for the Python scanner (spec 65).

One parse per file, shared by every extractor. The index answers the cross-file questions a
single-file pass cannot: what a module-level constant holds, which config attribute has which
default, which module an import points at, and which functions return an LLM client.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.scanner.extractors.py_bindings import ImportTable, split_module_name


@dataclass
class PyModule:
    rel: str
    dotted: str
    tree: ast.Module
    lines: list[str]
    imports: ImportTable
    parents: dict[int, ast.AST]
    consts: dict[str, ast.expr] = field(default_factory=dict)
    const_lines: dict[str, int] = field(default_factory=dict)


@dataclass
class ConfigDefault:
    attr: str
    expr: ast.expr
    module: PyModule
    line: int
    scope: ast.AST  # the class body or __init__ the value lives in


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    out: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            out[id(child)] = parent
    return out


def _module_dotted(rel: str) -> str:
    path = rel[:-3] if rel.endswith(".py") else rel
    parts = [p for p in path.split("/") if p]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


class RepoIndex:
    def __init__(self) -> None:
        self.modules: dict[str, PyModule] = {}
        self._by_key: dict[str, list[str]] = {}
        self.config_defaults: dict[str, list[ConfigDefault]] = {}
        # (module rel, function name) -> client description, filled by the LLM extractor pre-pass
        self.client_factories: dict[tuple[str, str], Any] = {}
        # (module rel, function name) of functions that wrap a client they receive as an argument
        # (`def build_agent(model): return Agent(model=model)`), filled by the same pre-pass
        self.client_wrappers: set[tuple[str, str]] = set()
        # files Python could not parse (spec 74): they yield no fact, so the document says so
        self.unparsed: list[str] = []

    # --- building -----------------------------------------------------------------------------

    @classmethod
    def build(cls, root: Path, files: list[Path]) -> RepoIndex:
        index = cls()
        for path in files:
            index.add(path, root)
        return index

    def add(self, path: Path, root: Path) -> PyModule | None:
        source = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            self.unparsed.append(path.relative_to(root).as_posix())
            return None
        rel = path.relative_to(root).as_posix()
        mod = PyModule(
            rel=rel,
            dotted=_module_dotted(rel),
            tree=tree,
            lines=source.splitlines(),
            imports=ImportTable.from_tree(tree),
            parents=_parents(tree),
        )
        for stmt in tree.body:
            targets: list[ast.expr] = []
            value: ast.expr | None = None
            if isinstance(stmt, ast.Assign):
                targets, value = stmt.targets, stmt.value
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                targets, value = [stmt.target], stmt.value
            for t in targets:
                if isinstance(t, ast.Name) and value is not None:
                    mod.consts[t.id] = value
                    mod.const_lines[t.id] = stmt.lineno
        self._index_config(mod)
        self.modules[rel] = mod
        parts = mod.dotted.split(".")
        for i in range(len(parts)):
            self._by_key.setdefault(".".join(parts[i:]), []).append(rel)
        return mod

    def _index_config(self, mod: PyModule) -> None:
        for node in ast.walk(mod.tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    and stmt.value
                ):
                    self._add_default(stmt.target.id, stmt.value, mod, stmt.lineno, node)
                elif isinstance(stmt, ast.Assign):
                    for t in stmt.targets:
                        if isinstance(t, ast.Name):
                            self._add_default(t.id, stmt.value, mod, stmt.lineno, node)
                elif isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                    for sub in ast.walk(stmt):
                        if isinstance(sub, ast.Assign):
                            for t in sub.targets:
                                if (
                                    isinstance(t, ast.Attribute)
                                    and isinstance(t.value, ast.Name)
                                    and t.value.id == "self"
                                ):
                                    self._add_default(t.attr, sub.value, mod, sub.lineno, stmt)

    def _add_default(
        self, attr: str, expr: ast.expr, mod: PyModule, line: int, scope: ast.AST
    ) -> None:
        self.config_defaults.setdefault(attr, []).append(
            ConfigDefault(attr, expr, mod, line, scope)
        )

    # --- lookups ------------------------------------------------------------------------------

    def find_module(self, dotted: str) -> PyModule | None:
        """Module for an import path; exact suffix key, and only when unambiguous."""
        rels = self._by_key.get(dotted)
        if rels and len(rels) == 1:
            return self.modules[rels[0]]
        return None

    def resolve_import(self, qualified: str) -> tuple[PyModule, str] | None:
        """`pkg.constants.MODEL` → (module of pkg/constants.py, "MODEL") when it is in the repo."""
        module, name = split_module_name(qualified)
        found = self.find_module(module) if module else None
        if found is None:
            return None
        return found, name

    def module_for(self, qualified: str) -> PyModule | None:
        """Module object when `qualified` itself names a module of the repo (`import constants`)."""
        return self.find_module(qualified)
