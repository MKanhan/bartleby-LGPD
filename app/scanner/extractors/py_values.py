"""Resolve a Python expression to a static string value, and say where it comes from (spec 65).

The scanner may only state a model or a base URL as fact when the code fixes it. Everything else
comes back with an origin the document can print ("variável de ambiente OPENAI_MODEL_NAME",
"escolha do usuário na interface", "settings.llm_model — padrão em config/settings.py:28").

Classes:
  literal      fixed in the code: a literal, or a module constant (also imported from another
               module of the repo) holding one
  padrao       only an overridable default exists: config attribute, os.getenv with a default,
               a function parameter's default — `value` carries the default, it is not the fact
  runtime      chosen at run time: environment variable without default, user input, parameter
               without default
  desconhecido the scanner cannot follow it
  ausente      no expression at all (the SDK default applies)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from app.scanner.extractors.py_bindings import dotted
from app.scanner.py_index import PyModule, RepoIndex

_MAX_DEPTH = 8
_ENV_GETTERS = {"os.getenv", "getenv", "os.environ.get", "environ.get"}
_USER_INPUT_PREFIXES = ("st.", "streamlit.", "gr.", "gradio.", "click.prompt", "typer.prompt")


@dataclass(frozen=True)
class Resolved:
    value: str | None
    cls: str
    origin: str | None

    @property
    def is_fact(self) -> bool:
        return self.cls == "literal" and self.value is not None


ABSENT = Resolved(None, "ausente", "não informado no código (vale o padrão do serviço)")
# Spec 73: a configuration field whose default is None — nothing is set unless someone
# configures it. For a base_url this means "the service's own address, overridable".
DEFAULT_NONE = "padrao_ausente"


def _short(node: ast.AST) -> str:
    try:
        text = ast.unparse(node)
    except Exception:  # pragma: no cover - unparse is total on parsed trees
        return "?"
    return text if len(text) <= 70 else text[:67] + "..."


def _enclosing(node: ast.AST, mod: PyModule, kinds: tuple[type, ...]) -> ast.AST | None:
    cur = mod.parents.get(id(node))
    while cur is not None:
        if isinstance(cur, kinds):
            return cur
        cur = mod.parents.get(id(cur))
    return None


def _param_default(
    func: ast.FunctionDef | ast.AsyncFunctionDef, name: str
) -> tuple[bool, ast.expr | None]:
    """(is_param, default_expr) for a parameter of `func`."""
    args = func.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for a, d in zip(positional, defaults, strict=True):
        if a.arg == name:
            return True, d
    for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        if a.arg == name:
            return True, d
    return False, None


def _last_local_assign(func: ast.AST, name: str, before: int) -> ast.expr | None:
    best: tuple[int, ast.expr] | None = None
    for sub in ast.walk(func):
        if isinstance(sub, ast.Assign) and sub.lineno < before:
            for t in sub.targets:
                if (
                    isinstance(t, ast.Name)
                    and t.id == name
                    and (best is None or sub.lineno > best[0])
                ):
                    best = (sub.lineno, sub.value)
        elif (
            isinstance(sub, ast.AnnAssign)
            and sub.value is not None
            and isinstance(sub.target, ast.Name)
            and sub.target.id == name
            and sub.lineno < before
            and (best is None or sub.lineno > best[0])
        ):
            best = (sub.lineno, sub.value)
    return best[1] if best else None


def _as_padrao(r: Resolved, origin: str) -> Resolved:
    if r.value is None:
        return Resolved(None, r.cls if r.cls != "literal" else "desconhecido", r.origin or origin)
    return Resolved(r.value, "padrao", origin)


def resolve(node: ast.AST | None, mod: PyModule, index: RepoIndex, depth: int = 0) -> Resolved:
    if node is None:
        return ABSENT
    if depth > _MAX_DEPTH:
        return Resolved(None, "desconhecido", _short(node))

    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return Resolved(node.value, "literal", None)
        if node.value is None:
            return ABSENT
        return Resolved(None, "desconhecido", _short(node))

    if isinstance(node, ast.Name):
        return _resolve_name(node, mod, index, depth)

    if isinstance(node, ast.Attribute):
        return _resolve_attribute(node, mod, index, depth)

    if isinstance(node, ast.Subscript):
        return _resolve_subscript(node, mod, index, depth)

    if isinstance(node, ast.Call):
        return _resolve_call(node, mod, index, depth)

    if isinstance(node, ast.IfExp):
        a = resolve(node.body, mod, index, depth + 1)
        b = resolve(node.orelse, mod, index, depth + 1)
        if a.is_fact and b.is_fact and a.value == b.value:
            return a
        return Resolved(None, "runtime", f"depende de condição: {_short(node.test)}")

    if isinstance(node, ast.JoinedStr):
        return Resolved(None, "runtime", "texto montado em tempo de execução")

    return Resolved(None, "desconhecido", _short(node))


def _resolve_name(node: ast.Name, mod: PyModule, index: RepoIndex, depth: int) -> Resolved:
    name = node.id
    func = _enclosing(node, mod, (ast.FunctionDef, ast.AsyncFunctionDef))
    if func is not None:
        local = _last_local_assign(func, name, getattr(node, "lineno", 10**9))
        if local is not None:
            return resolve(local, mod, index, depth + 1)
        is_param, default = _param_default(func, name)  # type: ignore[arg-type]
        if is_param:
            fname = getattr(func, "name", "?")
            if default is None:
                return Resolved(None, "runtime", f"parâmetro {name} de {fname}()")
            r = resolve(default, mod, index, depth + 1)
            return _as_padrao(r, f"padrão do parâmetro {name} de {fname}()")

    if name in mod.consts:
        r = resolve(mod.consts[name], mod, index, depth + 1)
        if r.is_fact:
            return Resolved(
                r.value, "literal", f"constante {name} ({mod.rel}:{mod.const_lines[name]})"
            )
        return r

    qualified = mod.imports.names.get(name)
    if qualified:
        hit = index.resolve_import(qualified)
        if hit is not None:
            target, attr = hit
            if attr in target.consts:
                r = resolve(target.consts[attr], target, index, depth + 1)
                if r.is_fact:
                    return Resolved(
                        r.value,
                        "literal",
                        f"constante {attr} ({target.rel}:{target.const_lines[attr]})",
                    )
                return r
    return Resolved(None, "desconhecido", name)


def _resolve_attribute(
    node: ast.Attribute, mod: PyModule, index: RepoIndex, depth: int
) -> Resolved:
    text = dotted(node) or _short(node)

    # self.x — follow the assignment inside the class (usually in __init__).
    if isinstance(node.value, ast.Name) and node.value.id == "self":
        cls = _enclosing(node, mod, (ast.ClassDef,))
        if cls is not None:
            for sub in ast.walk(cls):
                if isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if (
                            isinstance(t, ast.Attribute)
                            and isinstance(t.value, ast.Name)
                            and t.value.id == "self"
                            and t.attr == node.attr
                        ):
                            return resolve(sub.value, mod, index, depth + 1)

    # module.CONSTANT — a module of the repo imported by name.
    qualified = mod.imports.qualify(node)
    if qualified:
        hit = index.resolve_import(qualified)
        if hit is not None:
            target, attr = hit
            if attr in target.consts:
                r = resolve(target.consts[attr], target, index, depth + 1)
                if r.is_fact:
                    return Resolved(
                        r.value,
                        "literal",
                        f"constante {attr} ({target.rel}:{target.const_lines[attr]})",
                    )
                return r

    # settings.X / config.X / self.settings.X — an overridable configuration default.
    candidates = _config_candidates(node, mod, index)
    if candidates:
        c = candidates[0]
        results = [resolve(k.expr, k.module, index, depth + 1) for k in candidates]
        r = results[0]
        if all((x.value, x.cls) == (r.value, r.cls) for x in results):
            if r.value is not None:
                return Resolved(r.value, "padrao", f"{text} — padrão em {c.module.rel}:{c.line}")
            if r.cls == "runtime":
                return Resolved(None, "runtime", f"{text} ({r.origin})")
            if r.cls == "ausente" and _is_none(c.expr):
                return Resolved(
                    None, DEFAULT_NONE, f"{text} — padrão None em {c.module.rel}:{c.line}"
                )
    return Resolved(None, "desconhecido", text)


def _is_none(expr: ast.expr) -> bool:
    if isinstance(expr, ast.Constant):
        return expr.value is None
    if isinstance(expr, ast.Call) and (dotted(expr.func) or "").endswith("Field"):
        default = expr.args[0] if expr.args else None
        for kw in expr.keywords:
            if kw.arg == "default":
                default = kw.value
        return isinstance(default, ast.Constant) and default.value is None
    return False


def _annotation_name(annotation: ast.expr | None) -> str | None:
    """`ModelSpec`, `cfg.ModelSpec`, `"ModelSpec"`, `ModelSpec | None` -> 'ModelSpec'."""
    if annotation is None:
        return None
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value.split(".")[-1].split("|")[0].strip() or None
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return _annotation_name(annotation.left)
    if isinstance(annotation, ast.Attribute):
        return annotation.attr
    return annotation.id if isinstance(annotation, ast.Name) else None


def _config_candidates(node: ast.Attribute, mod: PyModule, index: RepoIndex) -> list:
    """Configuration defaults `x.attr` may stand for. One class declares the field: that one.
    Several: the class the parameter `x` is annotated with (`spec: ModelSpec`), else all of
    them — the caller only uses them when they agree."""
    candidates = index.config_defaults.get(node.attr, [])
    if len(candidates) <= 1 or not isinstance(node.value, ast.Name):
        return candidates
    func = _enclosing(node, mod, (ast.FunctionDef, ast.AsyncFunctionDef))
    if func is not None:
        args = func.args
        for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            if a.arg == node.value.id:
                wanted = _annotation_name(a.annotation)
                typed = [
                    c
                    for c in candidates
                    if isinstance(c.scope, ast.ClassDef) and c.scope.name == wanted
                ]
                if typed:
                    return typed
    return candidates


def _resolve_subscript(
    node: ast.Subscript, mod: PyModule, index: RepoIndex, depth: int
) -> Resolved:
    base_text = dotted(node.value)
    key = node.slice
    if base_text in ("os.environ", "environ") and isinstance(key, ast.Constant):
        return Resolved(None, "runtime", f"variável de ambiente {key.value}")
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        container = _container_value(node.value, mod, index, depth)
        if isinstance(container, ast.Dict):
            for k, v in zip(container.keys, container.values, strict=True):
                if isinstance(k, ast.Constant) and k.value == key.value:
                    r = resolve(v, mod, index, depth + 1)
                    return _as_padrao(r, f"{_short(node)} — valor de configuração")
    return Resolved(None, "desconhecido", _short(node))


def _container_value(node: ast.AST, mod: PyModule, index: RepoIndex, depth: int) -> ast.AST | None:
    if isinstance(node, ast.Dict):
        return node
    if isinstance(node, ast.Name) and node.id in mod.consts:
        return mod.consts[node.id]
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        inner = _container_value(node.value, mod, index, depth + 1)
        if isinstance(inner, ast.Dict):
            for k, v in zip(inner.keys, inner.values, strict=True):
                if isinstance(k, ast.Constant) and k.value == node.slice.value:
                    return v
    return None


def _resolve_call(node: ast.Call, mod: PyModule, index: RepoIndex, depth: int) -> Resolved:
    callee = dotted(node.func) or ""
    if callee in _ENV_GETTERS:
        var = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else "?"
        default: ast.expr | None = node.args[1] if len(node.args) > 1 else None
        for kw in node.keywords:
            if kw.arg == "default":
                default = kw.value
        if default is None:
            return Resolved(None, "runtime", f"variável de ambiente {var}")
        r = resolve(default, mod, index, depth + 1)
        if r.value is None:
            return Resolved(None, "runtime", f"variável de ambiente {var}")
        return Resolved(r.value, "padrao", f"variável de ambiente {var} (padrão {r.value!r})")
    if callee.endswith("Field") or callee == "Field":
        default = node.args[0] if node.args else None
        for kw in node.keywords:
            if kw.arg == "default":
                default = kw.value
        return resolve(default, mod, index, depth + 1) if default is not None else ABSENT
    if callee.startswith(_USER_INPUT_PREFIXES) or callee == "input":
        return Resolved(None, "runtime", f"escolha do usuário na interface ({callee})")
    return Resolved(None, "desconhecido", _short(node))
