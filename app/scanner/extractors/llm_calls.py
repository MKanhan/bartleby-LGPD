"""LLM operations in Python code — provider and model by evidence, never by method name (spec 65).

How it works, per module:
1. A *client* is a call to a catalog constructor (`Groq()`, `ChatGroq(...)`, `genai.GenerativeModel`,
   `boto3.client("bedrock-runtime")`), a router (`crewai.LLM("groq/...")`), or a call to a repo
   function that returns a client. Its provider comes from evidence layers: base_url/host, then
   Azure evidence, then the class. An unresolvable base_url on an OpenAI-style client yields
   `openai_compatible`, never "openai".
2. Clients propagate (taint) through assignments (also conditional ones and `self.x`), methods that
   return the client (`start_chat`, `bind_tools`), `prompt | llm`, and chain/agent constructors
   that receive it. Only a call made on a tainted receiver is an LLM call; an unknown receiver's
   `.invoke` (a graph, a retriever, a tool) is not.
3. The operation sits at the call site. A client with no recognised call gets one operation at its
   constructor (`extra.via = "construtor"`), emitted after every module was seen, since a factory
   in one module may be used in another.
4. The model is a fact only when the code fixes it; otherwise the operation carries its origin.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from app.scanner import provider_catalog as pc
from app.scanner.extractors.http_calls import UNKNOWN as _URL_GAP
from app.scanner.extractors.http_calls import url_text
from app.scanner.extractors.py_bindings import dotted, split_module_name
from app.scanner.extractors.py_values import ABSENT, DEFAULT_NONE, Resolved, resolve
from app.scanner.py_index import PyModule, RepoIndex
from app.scanner.schema import Operation

_HTTP_LIBS = ("requests", "httpx", "aiohttp")
_ROUTER_FUNCTIONS = ("completion", "acompletion", "embedding")
_HTTP_WRITE = {"post", "request"}


@dataclass(eq=False)
class Client:
    provider: str
    provider_origin: str
    kind: str
    model: Resolved
    module: PyModule
    node: ast.Call
    intermediario: str | None = None
    used: bool = False
    # spec 73: how sure the provider is (provider_padrao), copied into the operation's extra
    marks: dict[str, Any] = field(default_factory=dict)

    def derive(self, model: Resolved) -> Client:
        """A view of the same client with a model fixed later (e.g. `chats.create(model=...)`)."""
        if model.cls == "ausente":
            return self
        twin = Client(
            self.provider,
            self.provider_origin,
            self.kind,
            model,
            self.module,
            self.node,
            self.intermediario,
            marks=self.marks,
        )
        twin._parent = self  # type: ignore[attr-defined]
        return twin

    def mark_used(self) -> None:
        self.used = True
        parent = getattr(self, "_parent", None)
        if parent is not None:
            parent.mark_used()


@dataclass
class _Ambiguous:
    keys: set[str] = field(default_factory=set)


def _scope_of(node: ast.AST, mod: PyModule, kinds: tuple[type, ...]) -> int:
    cur = mod.parents.get(id(node))
    while cur is not None:
        if isinstance(cur, kinds):
            return id(cur)
        cur = mod.parents.get(id(cur))
    return 0


def _func_scope(node: ast.AST, mod: PyModule) -> int:
    return _scope_of(node, mod, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))


def _class_scope(node: ast.AST, mod: PyModule) -> int:
    return _scope_of(node, mod, (ast.ClassDef,))


def _kw(call: ast.Call, names: list[str] | tuple[str, ...]) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg in names:
            return kw.value
    return None


def _model_expr(call: ast.Call, model_pos: int | None = None) -> ast.expr | None:
    expr = _kw(call, pc.load_catalog().model_kwargs)
    if expr is None and model_pos is not None and len(call.args) > model_pos:
        expr = call.args[model_pos]
    return expr


class ModuleLLM:
    """Clients, taint and operations for one module. Instances live for the whole scan."""

    def __init__(self, mod: PyModule, index: RepoIndex) -> None:
        self.mod = mod
        self.index = index
        self.clients: list[Client] = []
        self._by_node: dict[int, Client] = {}
        self._taint: dict[tuple[int, str], Client | _Ambiguous] = {}
        self._module_azure = self._module_api_type_is_azure()

    # --- clients --------------------------------------------------------------------------------

    def _qualify(self, expr: ast.AST) -> str | None:
        """Qualified dotted path of a callee, following what plain imports cannot see:
        a module-level alias (`openai = promptlayer.openai`), names that arrive through
        `from config import *`, and catalog wrappers that re-export a provider module."""
        q = self.mod.imports.qualify(expr)
        if q is None:
            q = self._qualify_via_assignment(expr)
        return pc.unalias(q) if q else None

    def _qualify_via_assignment(self, expr: ast.AST) -> str | None:
        parts: list[str] = []
        cur = expr
        while isinstance(cur, ast.Attribute):
            parts.insert(0, cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Call):
            # boto3.Session().client("bedrock-runtime") — spec 73
            base = pc.instance_module(self.mod.imports.qualify(cur.func) or "")
            return ".".join([base, *parts]) if base else None
        if not isinstance(cur, ast.Name):
            return None
        held = self._held_value(cur)
        if isinstance(held, ast.Call):
            base = pc.instance_module(self.mod.imports.qualify(held.func) or "")
            if base:
                return ".".join([base, *parts])
        sources = [self.mod] + [
            m for m in (self.index.find_module(s) for s in self.mod.imports.stars) if m is not None
        ]
        for source in sources:
            value = source.consts.get(cur.id)
            if isinstance(value, ast.Attribute | ast.Name):
                base = source.imports.qualify(value)
                if base:
                    return ".".join([base, *parts])
        return None

    def _held_value(self, name: ast.Name) -> ast.expr | None:
        """What a plain name was assigned: in its own function first, then at module level."""
        cur = self.mod.parents.get(id(name))
        while cur is not None and not isinstance(cur, ast.FunctionDef | ast.AsyncFunctionDef):
            cur = self.mod.parents.get(id(cur))
        if cur is not None:
            for sub in ast.walk(cur):
                if isinstance(sub, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == name.id for t in sub.targets
                ):
                    return sub.value
        return self.mod.consts.get(name.id)

    def _module_api_type_is_azure(self) -> bool:
        az = pc.get("azure_openai")
        want = az.evidencia.api_type if az and az.evidencia else None
        for node in ast.walk(self.mod.tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if self._qualify(t) == "openai.api_type":
                        r = resolve(node.value, self.mod, self.index)
                        if r.value and r.value.lower() == want:
                            return True
        return False

    def _azure_kwargs(self, call: ast.Call) -> list[str]:
        az = pc.get("azure_openai")
        names = az.evidencia.kwargs if az and az.evidencia else []
        return [kw.arg for kw in call.keywords if kw.arg in names]

    def _base_url(self, call: ast.Call) -> tuple[ast.expr | None, Resolved]:
        names = pc.load_catalog().base_url_kwargs
        expr = _kw(call, names)
        if expr is None:
            # one level down: OpenAIModel(..., provider=OpenAIProvider(base_url=...))
            for kw in call.keywords:
                if isinstance(kw.value, ast.Call):
                    expr = _kw(kw.value, names)
                    if expr is not None:
                        break
        return expr, resolve(expr, self.mod, self.index) if expr is not None else ABSENT

    def _provider_for(
        self, call: ast.Call, provider: pc.Provider
    ) -> tuple[str, str, dict[str, Any]]:
        """(provider id, origin, marks) by evidence layers for a client constructed by `call`.
        Spec 73: a provider that comes from an overridable default (a default address, or an
        address field whose default is None) is claimed and marked configurable."""
        expr, url = self._base_url(call)
        default = {"provider_padrao": True}
        if expr is not None and provider.id in ("openai", "azure_openai"):
            if url.value:
                hit = pc.by_host(url.value)
                tail = "" if url.cls == "literal" else f" ({url.origin})"
                marks = default if url.cls == "padrao" else {}
                if hit is not None:
                    return hit.id, f"host do base_url {url.value}{tail}", marks
                if pc.azure_path_in(url.value):
                    return "azure_openai", f"caminho Azure no base_url {url.value}", marks
                return (
                    pc.OPENAI_COMPATIBLE,
                    f"base_url {url.value} não é de provedor catalogado",
                    {},
                )
            if url.cls != DEFAULT_NONE:
                return pc.OPENAI_COMPATIBLE, f"base_url não resolvido: {url.origin}", {}
        azure_kw = self._azure_kwargs(call)
        if provider.id == "openai" and (azure_kw or self._module_azure):
            why = f"argumento {azure_kw[0]}" if azure_kw else "openai.api_type = azure"
            return "azure_openai", why, {}
        qualified = self._qualify(call.func) or dotted(call.func) or "?"
        if url.cls == DEFAULT_NONE:
            return provider.id, f"classe {qualified}; {url.origin}", default
        return provider.id, f"classe {qualified}", {}

    def client_for_call(self, call: ast.Call) -> Client | None:
        """Client created by `call` (constructor, router or repo factory), cached per node."""
        key = id(call)
        cached = self._by_node.get(key)
        if cached is not None:
            return cached
        # "Not a client" is not cached: a factory of another module may only be known later.
        client = self._make_client(call)
        if isinstance(client, _Ambiguous):
            return client  # type: ignore[return-value]
        if client is not None:
            self._by_node[key] = client
            if client.node is call:
                self.clients.append(client)
        return client

    def _make_client(self, call: ast.Call) -> Client | None:
        qualified = self._qualify(call.func)
        if qualified:
            module, name = split_module_name(qualified)
            hit = pc.python_class(module, name)
            if hit is not None:
                provider, cls = hit
                if not self._requirements_met(call, cls):
                    return None
                pid, origin, marks = self._provider_for(call, provider)
                kind = "embed" if cls.kind == "embed" else "invoke_llm"
                model_expr = _kw(call, (cls.model_kwarg,)) if cls.model_kwarg else None
                if model_expr is None:
                    model_expr = _model_expr(call, cls.model_pos)
                model = resolve(model_expr, self.mod, self.index)
                return Client(pid, origin, kind, model, self.mod, call, marks=marks)
            if pc.is_router(qualified) and qualified.split(".")[-1] not in _ROUTER_FUNCTIONS:
                return self._router_client(call, qualified)
        return self._factory_client(call)

    def _requirements_met(self, call: ast.Call, cls: pc.PyClass) -> bool:
        for arg_name, allowed in cls.requer.items():
            expr = _kw(call, (arg_name,)) or (call.args[0] if call.args else None)
            value = resolve(expr, self.mod, self.index).value
            if value not in allowed:
                return False
        return True

    def _router_client(self, call: ast.Call, qualified: str) -> Client | None:
        expr = _model_expr(call, 0)
        passed = self.taint_of(expr)
        if isinstance(passed, Client):
            return passed  # Agent(model=OpenAIModel(...)) — the model object is the client
        if isinstance(passed, _Ambiguous):
            origin = f"mais de um cliente possível: {', '.join(sorted(passed.keys))}"
            return Client(pc.UNRESOLVED, origin, "invoke_llm", ABSENT, self.mod, call)
        r = resolve(expr, self.mod, self.index)
        # spec 71: a provider named by argument (init_chat_model(model, model_provider="openai"))
        named = resolve(_kw(call, pc.load_catalog().router_provider_kwargs), self.mod, self.index)
        if named.value:
            hit = pc.by_prefix(f"{named.value}:{r.value or 'x'}")
            if hit is not None:
                tail = "" if named.cls == "literal" else f" ({named.origin})"
                return Client(
                    hit[0].id,
                    f"provedor {named.value!r}{tail} no roteador {qualified}",
                    "embed" if qualified.endswith("init_embeddings") else "invoke_llm",
                    r,
                    self.mod,
                    call,
                )
        base_expr, url = self._base_url(call)
        if base_expr is not None and url.value:
            host_hit = pc.by_host(url.value)
            if host_hit is not None:
                split = pc.by_prefix(r.value) if r.value else None
                model = Resolved(split[1], r.cls, r.origin) if split else r
                tail = "" if url.cls == "literal" else f" ({url.origin})"
                return Client(
                    host_hit.id,
                    f"host do base_url {url.value}{tail} ({qualified})",
                    "invoke_llm",
                    model,
                    self.mod,
                    call,
                )
        if r.value:
            # a model that is only a default names the provider as configurable (spec 73)
            marks = {"provider_padrao": True} if r.cls == "padrao" else {}
            hit = pc.by_prefix(r.value)
            if hit is not None:
                provider, model = hit
                return Client(
                    provider.id,
                    f"prefixo {provider.id} em {r.value!r} ({qualified})",
                    "invoke_llm",
                    Resolved(model, r.cls, r.origin),
                    self.mod,
                    call,
                    marks=marks,
                )
            bare = pc.by_bare_model(r.value) if base_expr is None else None
            if bare is not None:
                return Client(
                    bare.id,
                    f"nome de modelo {r.value!r} sem prefixo: o roteador {qualified} o envia a "
                    f"{bare.nome}",
                    "embed"
                    if qualified.endswith(("embedding", "init_embeddings"))
                    else "invoke_llm",
                    r,
                    self.mod,
                    call,
                    marks=marks,
                )
        return Client(
            pc.UNRESOLVED,
            f"modelo do roteador {qualified} não resolvido: {r.origin}",
            "invoke_llm",
            r,
            self.mod,
            call,
        )

    def _repo_function(self, call: ast.Call) -> tuple[str, str] | None:
        """(module rel, function name) when `call` calls a function defined in the repo."""
        if isinstance(call.func, ast.Name) and any(
            isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == call.func.id
            for n in self.mod.tree.body
        ):
            return (self.mod.rel, call.func.id)
        qualified = self._qualify(call.func)
        if qualified:
            hit = self.index.resolve_import(qualified)
            if hit is not None:
                return (hit[0].rel, hit[1])
        return None

    def _factory_client(self, call: ast.Call) -> Client | None:
        if isinstance(call.func, ast.Name):
            local = self.index.client_factories.get((self.mod.rel, call.func.id))
            if local is not None:
                return local
        qualified = self._qualify(call.func)
        if qualified:
            hit = self.index.resolve_import(qualified)
            if hit is not None:
                target, name = hit
                return self.index.client_factories.get((target.rel, name))
        return None

    # --- taint ----------------------------------------------------------------------------------

    def _keys_for_target(self, target: ast.AST, at: ast.AST) -> list[tuple[int, str]]:
        if isinstance(target, ast.Name):
            return [(_func_scope(at, self.mod), target.id)]
        text = dotted(target)
        if text is None:
            return []
        if text.startswith("self."):
            return [(_class_scope(at, self.mod), text)]
        return [(0, text)]

    def _lookup(self, expr: ast.AST) -> Client | _Ambiguous | None:
        if isinstance(expr, ast.Name):
            for scope in (_func_scope(expr, self.mod), 0):
                hit = self._taint.get((scope, expr.id))
                if hit is not None:
                    return hit
            return None
        text = dotted(expr)
        if text is None:
            return None
        if text.startswith("self."):
            return self._taint.get((_class_scope(expr, self.mod), text))
        return self._taint.get((0, text))

    def taint_of(self, expr: ast.AST | None) -> Client | _Ambiguous | None:
        if expr is None:
            return None
        if isinstance(expr, ast.Await):
            return self.taint_of(expr.value)
        if isinstance(expr, ast.IfExp):
            return self.taint_of(expr.body) or self.taint_of(expr.orelse)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
            return self.taint_of(expr.right) or self.taint_of(expr.left)
        if isinstance(expr, ast.Name | ast.Attribute):
            return self._lookup(expr)
        if isinstance(expr, ast.Call):
            made = self.client_for_call(expr)
            if made is not None:
                return made
            if isinstance(expr.func, ast.Attribute):
                receiver = self.taint_of(expr.func.value)
                if isinstance(receiver, Client) and pc.is_propagating_method(expr.func.attr):
                    return receiver.derive(resolve(_model_expr(expr), self.mod, self.index))
                if receiver is not None and pc.is_propagating_method(expr.func.attr):
                    return receiver
            target = self._repo_function(expr)
            if target is not None and target in self.index.client_wrappers:
                for arg in [*expr.args, *(kw.value for kw in expr.keywords)]:
                    hit = self.taint_of(arg)
                    if hit is not None:
                        return hit
            callee = dotted(expr.func) or ""
            if pc.is_propagating_callee(callee.split(".")[-1]):
                for arg in [*expr.args, *(kw.value for kw in expr.keywords)]:
                    hit = self.taint_of(arg)
                    if hit is not None:
                        return hit
        return None

    def learn(self) -> None:
        """Propagate clients through assignments until nothing changes (cheap per module)."""
        assigns: list[tuple[ast.AST, ast.expr]] = []
        for node in ast.walk(self.mod.tree):
            if isinstance(node, ast.Assign):
                assigns.extend((t, node.value) for t in node.targets)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                assigns.append((node.target, node.value))
            elif isinstance(node, ast.With | ast.AsyncWith):
                for item in node.items:
                    if item.optional_vars is not None:
                        assigns.append((item.optional_vars, item.context_expr))
        for _ in range(4):
            grew = False
            for target, value in assigns:
                hit = self.taint_of(value)
                if hit is None:
                    continue
                for key in self._keys_for_target(target, value):
                    grew |= self._set_taint(key, hit)
            if not grew:
                break

    def _set_taint(self, key: tuple[int, str], hit: Client | _Ambiguous) -> bool:
        """Record what a name holds; two different providers under one name are ambiguous."""
        current = self._taint.get(key)
        if current is None:
            self._taint[key] = hit
            return True
        if current is hit or isinstance(current, _Ambiguous):
            return False
        if (
            isinstance(hit, Client)
            and isinstance(current, Client)
            and hit.provider == current.provider
        ):
            return False
        self._taint[key] = _Ambiguous({key[1]})
        return True

    def _repo_class(self, call: ast.Call) -> tuple[PyModule, ast.ClassDef] | None:
        """(module, class) when `call` constructs a class defined in the repository."""
        target, name = self.mod, call.func.id if isinstance(call.func, ast.Name) else None
        if name is None or not any(
            isinstance(n, ast.ClassDef) and n.name == name for n in self.mod.tree.body
        ):
            qualified = self._qualify(call.func)
            hit = self.index.resolve_import(qualified) if qualified else None
            if hit is None:
                return None
            target, name = hit
        for node in target.tree.body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                return target, node
        return None

    def inject(self, analyzers: dict[str, ModuleLLM]) -> bool:
        """Spec 73: a client handed to the constructor of a repository class lives on in the
        attribute the class stores it in (`Service(client=c)` with `self.client = client` in
        `__init__`, or a dataclass field): calls on `self.client` in that class are its calls."""
        grew = False
        for node in ast.walk(self.mod.tree):
            if not isinstance(node, ast.Call) or not (node.args or node.keywords):
                continue
            hit = self._repo_class(node)
            if hit is None:
                continue
            target, cls = hit
            for param, expr in _bound_arguments(cls, node).items():
                client = self.taint_of(expr)
                if client is None:
                    continue
                for attr in _attributes_fed_by(cls, param):
                    key = (id(cls), f"self.{attr}")
                    grew |= analyzers[target.rel]._set_taint(key, client)
        return grew

    def factories(self) -> dict[tuple[str, str], Client | _Ambiguous]:
        """Functions of this module whose return value is a client.

        A function that returns clients of different providers (configuration picks one at run
        time) is ambiguous: its callers get `nao_resolvido`, and each constructor still shows up
        as its own operation.
        """
        out: dict[tuple[str, str], Client | _Ambiguous] = {}
        for node in ast.walk(self.mod.tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            found: list[Client] = []
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and sub.value is not None:
                    hit = self.taint_of(sub.value)
                    if isinstance(hit, Client):
                        found.append(hit)
            providers = {c.provider for c in found}
            if len(providers) == 1:
                out[(self.mod.rel, node.name)] = found[0]
            elif len(providers) > 1:
                out[(self.mod.rel, node.name)] = _Ambiguous(
                    {f"{node.name}() devolve {' ou '.join(sorted(providers))}"}
                )
        return out

    def wrappers(self) -> set[tuple[str, str]]:
        """Functions that return a chain/agent built from one of their parameters, or the parameter
        itself — calling them with a client argument yields that client."""
        out: set[tuple[str, str]] = set()
        for node in ast.walk(self.mod.tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            params = {
                a.arg for a in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            }
            params.discard("self")
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Return) or sub.value is None:
                    continue
                value = sub.value
                if isinstance(value, ast.Name) and value.id in params:
                    out.add((self.mod.rel, node.name))
                elif isinstance(value, ast.Call):
                    callee = (dotted(value.func) or "").split(".")[-1]
                    args = [*value.args, *(kw.value for kw in value.keywords)]
                    if pc.is_propagating_callee(callee) and any(
                        isinstance(a, ast.Name) and a.id in params for a in args
                    ):
                        out.add((self.mod.rel, node.name))
        return out

    # --- operations -----------------------------------------------------------------------------

    def operations(self) -> list[Operation]:
        # Register every catalog constructor, even one never assigned nor returned, so that an
        # unused client still becomes a constructor-level operation.
        for node in ast.walk(self.mod.tree):
            if isinstance(node, ast.Call):
                self.client_for_call(node)
        ops: list[Operation] = []
        for node in ast.walk(self.mod.tree):
            if not isinstance(node, ast.Call):
                continue
            op = self._call_operation(node)
            if op is not None:
                ops.append(op)
        return ops

    def _call_operation(self, call: ast.Call) -> Operation | None:
        func = call.func
        # 1) call on a tainted receiver: client.chat.completions.create(...), chain.invoke(...)
        if isinstance(func, ast.Attribute):
            path: list[str] = []
            cur: ast.AST = func
            while isinstance(cur, ast.Attribute):
                path.insert(0, cur.attr)
                receiver = self.taint_of(cur.value)
                if receiver is not None:
                    kind = pc.call_kind(".".join(path))
                    if kind is None:
                        return None
                    return self._op_on_receiver(call, receiver, kind, ".".join(path))
                cur = cur.value
        # 2) a tainted object called directly: llm("prompt")
        if isinstance(func, ast.Name):
            receiver = self._lookup(func)
            if receiver is not None:
                return self._op_on_receiver(call, receiver, "invoke_llm", func.id)
        qualified = self._qualify(func)
        if qualified:
            # 3) module function that is itself the call: ollama.chat, openai.ChatCompletion.create
            provider = pc.python_function(qualified)
            if provider is not None:
                return self._module_function_op(call, provider, qualified)
            # 4) router function: litellm.completion(model="groq/...")
            if pc.is_router(qualified) and qualified.split(".")[-1] in _ROUTER_FUNCTIONS:
                client = self._router_client(call, qualified)
                if client is not None:
                    client.mark_used()
                    kind = "embed" if qualified.endswith("embedding") else "invoke_llm"
                    return self._make_op(
                        call,
                        client.provider,
                        client.provider_origin,
                        "embed" if client.kind == "embed" else kind,
                        client.model,
                        "funcao",
                        extra=dict(client.marks),
                    )
        # 5) direct HTTP to a catalog host
        return self._http_op(call, qualified or dotted(func) or "")

    def _op_on_receiver(
        self, call: ast.Call, receiver: Client | _Ambiguous, kind: str, method: str
    ) -> Operation:
        call_model = resolve(_model_expr(call, pc.call_model_pos(method)), self.mod, self.index)
        if isinstance(receiver, _Ambiguous):
            origin = f"mais de um cliente possível: {', '.join(sorted(receiver.keys))}"
            return self._make_op(call, pc.UNRESOLVED, origin, kind, call_model, "chamada")
        receiver.mark_used()
        model = call_model if call_model.cls != "ausente" else receiver.model
        elsewhere = pc.call_model_origin(method)
        if model.cls == "ausente" and elsewhere:
            model = Resolved(None, "ausente", elsewhere)
        if receiver.kind == "embed":
            kind = "embed"
        elif model.is_fact and pc.is_embedding_model(receiver.provider, str(model.value)):
            kind = "embed"  # spec 73: Bedrock invoke_model with a Titan embedding model
        extra: dict[str, Any] = dict(receiver.marks)
        if method.endswith("audio.transcriptions.create"):
            extra["modalidade"] = "audio"
        stored = pc.call_stores_data(method)
        if stored:
            extra["guarda_no_provedor"] = stored
        return self._make_op(
            call,
            receiver.provider,
            receiver.provider_origin,
            kind,
            model,
            "chamada",
            receiver.intermediario,
            extra,
        )

    def _module_function_op(
        self, call: ast.Call, provider: pc.Provider, qualified: str
    ) -> Operation:
        pid, origin = provider.id, f"função {qualified}"
        azure_kw = self._azure_kwargs(call)
        if provider.id == "openai" and (azure_kw or self._module_azure):
            pid = "azure_openai"
            origin = f"argumento {azure_kw[0]}" if azure_kw else "openai.api_type = azure"
        model_expr = _model_expr(call)
        if pid == "azure_openai" and model_expr is None:
            # Azure names a deployment (engine/deployment_id), not the model behind it.
            deployment = resolve(_kw(call, ("deployment_id", "engine")), self.mod, self.index)
            model = Resolved(
                None, "runtime", f"deployment Azure: {deployment.value or deployment.origin}"
            )
        else:
            # SDK 0.x: `engine` is the model id for plain OpenAI calls.
            model = resolve(model_expr or _kw(call, ("engine",)), self.mod, self.index)
        kind = "embed" if "embed" in qualified.lower() else "invoke_llm"
        return self._make_op(call, pid, origin, kind, model, "funcao")

    def _http_op(self, call: ast.Call, callee: str) -> Operation | None:
        parts = callee.split(".")
        if not parts or parts[0] not in _HTTP_LIBS or parts[-1] not in _HTTP_WRITE:
            return None
        url_expr = call.args[0] if call.args else _kw(call, ("url",))
        if url_expr is None:
            return None
        provider_id, origin, url = self._provider_from_url(url_expr)
        if provider_id is None:
            return None
        # spec 73: the URL itself may name the model (.../models/gemini-2.0-flash:generateContent)
        named = pc.model_in_url(provider_id, url, _URL_GAP)
        model = Resolved(named, "literal", "caminho da URL") if named else ABSENT
        return self._make_op(call, provider_id, origin, "invoke_llm", model, "http")

    def _provider_from_url(self, expr: ast.expr) -> tuple[str | None, str, str]:
        """(provider, origin, the URL with what the code does not fix replaced by a gap). The
        pieces of an f-string or of a concatenation are read one by one (constants included)."""
        url = url_text(expr, self.mod, self.index)
        shown = url.replace(_URL_GAP, "")
        if pc.azure_path_in(shown):
            return "azure_openai", f"caminho Azure em {shown!r}", url
        host = url.split("://", 1)[1].split("/", 1)[0] if "://" in url else _URL_GAP
        hit = pc.by_host(shown) if _URL_GAP not in host else None
        return (hit.id, f"host em {shown!r}", url) if hit else (None, "", url)

    def _make_op(
        self,
        call: ast.Call,
        provider: str,
        provider_origin: str,
        kind: str,
        model: Resolved,
        via: str,
        intermediario: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Operation:
        return make_operation(
            self.mod, call, provider, provider_origin, kind, model, via, intermediario, extra
        )


def _init_of(cls: ast.ClassDef) -> ast.FunctionDef | None:
    return next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None
    )


def _bound_arguments(cls: ast.ClassDef, call: ast.Call) -> dict[str, ast.expr]:
    """Parameter name -> argument expression for a constructor call. With no `__init__`, the
    annotated fields of the class body are the parameters (dataclass, pydantic model)."""
    init = _init_of(cls)
    if init is not None:
        names = [a.arg for a in [*init.args.posonlyargs, *init.args.args]][1:]  # drop self
        keywords = set(names) | {a.arg for a in init.args.kwonlyargs}
    else:
        names = [
            n.target.id
            for n in cls.body
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
        ]
        keywords = set(names)
    bound: dict[str, ast.expr] = {}
    for name, arg in zip(names, call.args, strict=False):
        if not isinstance(arg, ast.Starred):
            bound[name] = arg
    for kw in call.keywords:
        if kw.arg in keywords:
            bound[kw.arg] = kw.value
    return bound


def _attributes_fed_by(cls: ast.ClassDef, param: str) -> list[str]:
    """Attributes of `self` that receive the parameter as it came (`self.x = param`)."""
    init = _init_of(cls)
    if init is None:
        return [param]
    out: list[str] = []
    for node in ast.walk(init):
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        if not (isinstance(value, ast.Name) and value.id == param):
            continue
        for t in targets:
            if (
                isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
            ):
                out.append(t.attr)
    return out


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    cached = getattr(tree, "_bartleby_parents", None)
    if cached is None:
        cached = {
            id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
        }
        tree._bartleby_parents = cached  # type: ignore[attr-defined]
    return cached


def _imported_elsewhere(name: str, client: Client, index: RepoIndex) -> bool:
    """Is a module-level name of the client's module imported by another module (by name or `*`)?"""
    stem = client.module.rel.removesuffix(".py").replace("/", ".")
    for mod in index.modules.values():
        if mod is client.module:
            continue
        for node in ast.walk(mod.tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (stem == node.module or stem.endswith("." + node.module))
                and any(a.name in (name, "*") for a in node.names)
            ):
                return True
    return False


def _flows(client: Client, index: RepoIndex) -> bool:
    """Does a client that is never called reach anything? A constructor assigned to a plain name
    counts only if that name is handed on — as an argument, a return value, an assignment, an
    element of a collection, a `with` target — or imported by another module. A name never read,
    or read only for its attributes and non-LLM methods (`llm.metadata`, `client.list()`), sends
    no personal data to the provider. (LLM calls on it would already have marked it used.)"""
    tree = client.module.tree
    parents = _parents(tree)
    holder = parents.get(id(client.node))
    if not (
        isinstance(holder, ast.Assign)
        and len(holder.targets) == 1
        and isinstance(holder.targets[0], ast.Name)
    ):
        return True  # passed inline, returned, stored in an attribute...: it goes somewhere
    name = holder.targets[0].id
    if isinstance(parents.get(id(holder)), ast.Module) and _imported_elsewhere(name, client, index):
        return True
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)):
            continue
        if isinstance(parents.get(id(node)), ast.Attribute):
            continue  # an attribute or a non-LLM method of it
        return True
    return False


def make_operation(
    mod: PyModule,
    node: ast.Call,
    provider: str,
    provider_origin: str,
    kind: str,
    model: Resolved,
    via: str,
    intermediario: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Operation:
    from app.scanner.extractors.python_ast import (
        _enclosing_function,
        _prev_comment,
        _return_var,
        _snippet,
    )

    data: dict[str, Any] = {"via": via, "provider_origin": provider_origin}
    if intermediario:
        data["intermediario"] = intermediario
    if model.is_fact:
        if model.origin:
            data["model_origin"] = model.origin
    else:
        data["model_origin"] = model.origin or "não resolvido"
        if model.cls == "padrao" and model.value:
            data["model_padrao"] = model.value
    data.update(extra or {})
    return Operation(
        kind=kind,
        provider=provider,
        model=model.value if model.is_fact else None,
        file=mod.rel,
        line=node.lineno,
        snippet=_snippet(mod.lines, node.lineno),
        enclosing_function=_enclosing_function(node, mod.parents),
        prev_comment=_prev_comment(mod.lines, node.lineno),
        return_var=_return_var(node, mod.parents),
        extra=data,
    )


def analyze_repo(index: RepoIndex) -> list[Operation]:
    """LLM operations for every module of the index, plus constructor fallbacks."""
    analyzers = {rel: ModuleLLM(mod, index) for rel, mod in index.modules.items()}
    for an in analyzers.values():
        index.client_wrappers |= an.wrappers()
    # Factories can call other factories: settle them in a few rounds before reading calls.
    for _ in range(3):
        before = len(index.client_factories)
        for an in analyzers.values():
            an.learn()
            index.client_factories.update(an.factories())
        if len(index.client_factories) == before:
            break
    for an in analyzers.values():
        an.learn()
    # spec 73: clients handed to constructors of repository classes, then what follows from them
    for _ in range(3):
        grew = [an.inject(analyzers) for an in analyzers.values()]  # every module, no short cut
        if not any(grew):
            break
        for an in analyzers.values():
            an.learn()
    ops: list[Operation] = []
    for an in analyzers.values():
        ops.extend(an.operations())
    seen: set[int] = set()
    for an in analyzers.values():
        for client in an.clients:
            if client.used or id(client) in seen:
                continue
            seen.add(id(client))
            if not _flows(client, index):
                continue  # spec 71: built and never handed anywhere — no data reaches the provider
            ops.append(
                make_operation(
                    client.module,
                    client.node,
                    client.provider,
                    client.provider_origin,
                    client.kind,
                    client.model,
                    "construtor",
                    client.intermediario,
                    dict(client.marks),
                )
            )
    return ops
