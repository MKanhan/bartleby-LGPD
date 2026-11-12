"""Outbound HTTP calls of Python code (spec 71).

Every `requests`, `httpx`, `aiohttp` or `urllib` call, with the URL read as far as the code fixes
it (literal, constant, config default, f-string pieces). The host is what the documents list as an
external service; a URL the code does not fix is kept with `url=None`, so the documents can count
it. Relative URLs (the system's own routes) are dropped.
"""

from __future__ import annotations

import ast

from app.scanner import integration_catalog as ic
from app.scanner.extractors.py_bindings import dotted
from app.scanner.extractors.py_values import resolve
from app.scanner.py_index import PyModule, RepoIndex
from app.scanner.schema import ExternalCall

# Marks a part of the URL the code does not fix.
UNKNOWN = "\x00"

_LIBS = ("requests", "httpx", "aiohttp", "urllib")
_METHODS = {"get", "post", "put", "patch", "delete", "request", "stream", "urlopen"}
# constructors whose instances make the calls: `session = aiohttp.ClientSession()`
_SESSIONS = {
    "requests.Session",
    "requests.session",
    "httpx.Client",
    "httpx.AsyncClient",
    "aiohttp.ClientSession",
}


def url_text(expr: ast.expr | None, mod: PyModule, index: RepoIndex) -> str:
    """The URL with every part the code does not fix replaced by UNKNOWN."""
    if expr is None:
        return UNKNOWN
    if isinstance(expr, ast.JoinedStr):
        parts = []
        for v in expr.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                parts.append(url_text(v.value, mod, index))
        return "".join(parts)
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        return url_text(expr.left, mod, index) + url_text(expr.right, mod, index)
    r = resolve(expr, mod, index)
    return r.value if r.value else UNKNOWN


def classify(url: str) -> str | None:
    """'' for a relative URL (the system itself), None when the host is not fixed, else the URL."""
    if url.startswith(UNKNOWN):
        return None
    if "://" not in url:
        return ""
    host = url.split("://", 1)[1].split("/", 1)[0]
    return None if not host or UNKNOWN in host else url


def _sessions(mod: PyModule) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(mod.tree):
        pairs: list[tuple[ast.expr, ast.expr]] = []
        if isinstance(node, ast.Assign):
            pairs = [(t, node.value) for t in node.targets]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            pairs = [(node.target, node.value)]
        elif isinstance(node, ast.With | ast.AsyncWith):
            pairs = [(i.optional_vars, i.context_expr) for i in node.items if i.optional_vars]
        for target, value in pairs:
            if isinstance(value, ast.Await):
                value = value.value
            if isinstance(value, ast.Call) and mod.imports.qualify(value.func) in _SESSIONS:
                name = dotted(target)
                if name:
                    names.add(name)
    return names


def _library_services(mod: PyModule) -> list[ExternalCall]:
    """Spec 73: an imported library that is itself an external service (integracoes.yml,
    `tipo: servico`), or whose credential the code reads by name. One entry per (file, service)."""
    found: dict[str, ExternalCall] = {}
    for node in ast.walk(mod.tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            service = ic.service_by_env(node.value)
            if service is not None and service.id not in found:
                found[service.id] = ExternalCall(
                    url=f"https://{service.host}",
                    method="BIBLIOTECA",
                    file=mod.rel,
                    line=node.lineno,
                    library=f"variável {node.value}",
                )
            continue
        if isinstance(node, ast.Import):
            imports = [(alias.name, []) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imports = [(node.module, [alias.name for alias in node.names])]
        else:
            continue
        for module, names in imports:
            service = ic.python_service(module, names)
            if service is not None and service.id not in found:
                found[service.id] = ExternalCall(
                    url=f"https://{service.host}",
                    method="BIBLIOTECA",
                    file=mod.rel,
                    line=node.lineno,
                    library=module,
                )
    return list(found.values())


def analyze_repo(index: RepoIndex) -> list[ExternalCall]:
    out: list[ExternalCall] = []
    for mod in index.modules.values():
        out.extend(_library_services(mod))
        sessions = _sessions(mod)
        for node in ast.walk(mod.tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute | ast.Name
            ):
                continue
            method = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            if method not in _METHODS:
                continue
            qualified = mod.imports.qualify(node.func) or ""
            if qualified.split(".")[0] in _LIBS:
                library = qualified.split(".")[0]
            elif isinstance(node.func, ast.Attribute) and dotted(node.func.value) in sessions:
                library = "sessao"
            else:
                continue
            args = list(node.args)
            if method == "request" and args:
                args = args[1:]  # request(method, url)
            expr = (
                args[0] if args else next((k.value for k in node.keywords if k.arg == "url"), None)
            )
            url = classify(url_text(expr, mod, index))
            if url == "":
                continue
            out.append(
                ExternalCall(
                    url=url,
                    method=method.upper(),
                    file=mod.rel,
                    line=node.lineno,
                    library=library,
                )
            )
    return out
