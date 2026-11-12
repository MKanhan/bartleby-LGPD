"""Repository index for JavaScript/TypeScript (spec 69) — the mirror of `py_index`.

Every JS/TS file is parsed once; each file records what it exports (ESM and CommonJS). An import
is followed to the file it names (relative path, extension or `index.*`, and the `baseUrl`/`paths`
aliases of `tsconfig.json`/`jsconfig.json`), and a name is always resolved in the module that
defines it — never by a bare name somewhere in the repository. What does not close stays
unresolved, with its origin, exactly as in Python.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from app.scanner import provider_catalog as pc
from app.scanner import store_catalog as sc

if TYPE_CHECKING:
    from tree_sitter import Node

    from app.scanner.extractors.tree_sitter_js import _FileJS

_RESOLVE_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts")
_MAX_DEPTH = 8
_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


class JsIndex:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[str, _FileJS] = {}
        # (config dir rel, baseUrl rel or None, {pattern: [targets]})
        self._aliases: list[tuple[str, str | None, dict[str, list[str]]]] = []
        self.ai_bindings: dict[str, str] = {}  # wrangler binding -> provider id (Workers AI)
        # spec 73: functions a framework injects with no import (NuxtHub), valid because the
        # project declares the package: name -> provider id / store id
        self.global_providers: dict[str, str] = {}
        self.global_stores: dict[str, str] = {}

    @classmethod
    def build(cls, root: Path, paths: list[Path]) -> JsIndex:
        from app.scanner.config_scan import wrangler_ai_bindings
        from app.scanner.extractors.tree_sitter_js import _parse_file

        index = cls(root)
        index._load_aliases()
        index.ai_bindings = wrangler_ai_bindings(root)
        declared = index._declared_packages()
        index.global_providers = pc.js_global_calls(declared)
        index.global_stores = sc.js_global_calls(declared)
        for path in paths:
            js = _parse_file(path, root, index)
            if js is not None:
                js.collect()
                index.files[js.rel] = js
        return index

    def _declared_packages(self) -> set[str]:
        """Dependencies declared by any package.json of the repository."""
        out: set[str] = set()
        for cfg in self.root.rglob("package.json"):
            if "node_modules" in cfg.parts:
                continue
            try:
                data = json.loads(cfg.read_text(encoding="utf-8", errors="ignore"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            for section in ("dependencies", "devDependencies", "peerDependencies"):
                deps = data.get(section)
                if isinstance(deps, dict):
                    out |= set(deps)
        return out

    # --- module resolution ----------------------------------------------------------------------

    def _load_aliases(self) -> None:
        for name in ("tsconfig.json", "jsconfig.json"):
            for cfg in self.root.rglob(name):
                if "node_modules" in cfg.parts:
                    continue
                try:
                    raw = cfg.read_text(encoding="utf-8", errors="ignore")
                    data = json.loads(_TRAILING_COMMA_RE.sub(r"\1", _COMMENT_RE.sub("", raw)))
                except (ValueError, OSError):
                    continue
                opts = (data or {}).get("compilerOptions") or {}
                base = opts.get("baseUrl")
                paths = {k: list(v) for k, v in (opts.get("paths") or {}).items() if v}
                if base is None and not paths:
                    continue
                cfg_dir = cfg.parent.relative_to(self.root).as_posix()
                cfg_dir = "" if cfg_dir == "." else cfg_dir
                base_rel = _join(cfg_dir, base) if base is not None else None
                self._aliases.append((cfg_dir, base_rel, paths))
        # the deepest config wins for files under it
        self._aliases.sort(key=lambda a: -len(a[0]))

    def _file_for(self, candidate: str) -> str | None:
        """Repository file for an import path without extension, or with a .js that means .ts."""
        cand = candidate.rstrip("/")
        if cand in self.files:
            return cand
        stem = cand
        for ext in (".js", ".jsx", ".mjs", ".cjs"):
            if cand.endswith(ext):
                stem = cand[: -len(ext)]
                break
        for ext in _RESOLVE_EXTS:
            if stem + ext in self.files:
                return stem + ext
        for ext in _RESOLVE_EXTS:
            if f"{cand}/index{ext}" in self.files:
                return f"{cand}/index{ext}"
        return None

    def resolve_specifier(self, from_rel: str, spec: str) -> str | None:
        """The repository file an import specifier names, or None (a package, or not found)."""
        if spec.startswith("."):
            base = PurePosixPath(from_rel).parent.as_posix()
            return self._file_for(_join("" if base == "." else base, spec))
        for cfg_dir, base_rel, paths in self._aliases:
            if cfg_dir and not from_rel.startswith(cfg_dir + "/"):
                continue
            root_rel = base_rel if base_rel is not None else cfg_dir
            for pattern, targets in paths.items():
                if pattern.endswith("*") and spec.startswith(pattern[:-1]):
                    rest = spec[len(pattern) - 1 :]
                    for target in targets:
                        hit = self._file_for(_join(root_rel, target.replace("*", rest)))
                        if hit:
                            return hit
                elif spec == pattern:
                    for target in targets:
                        hit = self._file_for(_join(root_rel, target))
                        if hit:
                            return hit
            if base_rel is not None:
                hit = self._file_for(_join(base_rel, spec))
                if hit:
                    return hit
        return None

    # --- exports --------------------------------------------------------------------------------

    def export(
        self, rel: str, name: str, depth: int = 0, seen: set[tuple[str, str]] | None = None
    ) -> tuple[_FileJS, Node] | None:
        """(module, value node) that `name` exported by `rel` stands for, following re-exports."""
        seen = seen if seen is not None else set()
        if depth > _MAX_DEPTH or (rel, name) in seen or rel not in self.files:
            return None
        seen.add((rel, name))
        f = self.files[rel]
        if name in f.exports:
            value = f.exports[name]
            if value.type == "identifier":
                return f.lookup(value, depth + 1) or (f, value)
            return f, value
        if name in f.export_locals:
            return f.lookup_name(f.export_locals[name], depth + 1)
        if name in f.reexports:
            source, original = f.reexports[name]
            target = self.resolve_specifier(rel, source)
            return self.export(target, original, depth + 1, seen) if target else None
        for source in f.star_sources:
            target = self.resolve_specifier(rel, source)
            hit = self.export(target, name, depth + 1, seen) if target else None
            if hit is not None:
                return hit
        return None


def _join(base: str, rel: str) -> str:
    parts: list[str] = [p for p in base.split("/") if p and p != "."]
    for p in rel.split("/"):
        if p in ("", "."):
            continue
        if p == "..":
            if parts:
                parts.pop()
            continue
        parts.append(p)
    return "/".join(parts)
