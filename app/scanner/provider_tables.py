"""Tables of provider addresses in the code (spec 73).

A constant dict or object whose values are the addresses of two or more catalog providers is a
menu: the code names the options and something at run time (a setting, the request) picks one.
The scanner reads the menu, not the order, so none of them is claimed: the documents list them
as candidates the controller must confirm, in the section of facts the code does not fix.
"""

from __future__ import annotations

import ast

from app.scanner import provider_catalog as pc
from app.scanner.py_index import RepoIndex


def providers_of(urls: list[str]) -> list[str]:
    """Catalog provider ids for the addresses of one table, in order, when they are two or more."""
    found: list[str] = []
    for url in urls:
        hit = pc.by_host(url) if "://" in url else None
        if hit is not None and hit.id not in found:
            found.append(hit.id)
    return found if len(found) >= 2 else []


def python_tables(index: RepoIndex) -> list[dict]:
    out: list[dict] = []
    for mod in index.modules.values():
        for node in ast.walk(mod.tree):
            if not isinstance(node, ast.Dict):
                continue
            urls = [
                v.value
                for v in node.values
                if isinstance(v, ast.Constant) and isinstance(v.value, str)
            ]
            providers = providers_of(urls)
            if not providers:
                continue
            holder = mod.parents.get(id(node))
            target: ast.expr | None = None
            if isinstance(holder, ast.Assign) and holder.targets:
                target = holder.targets[0]
            elif isinstance(holder, ast.AnnAssign):
                target = holder.target
            name = target.id if isinstance(target, ast.Name) else ""
            out.append(
                {"file": mod.rel, "line": node.lineno, "nome": name, "provedores": providers}
            )
    return out
