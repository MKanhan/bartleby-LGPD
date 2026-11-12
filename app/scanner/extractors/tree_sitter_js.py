"""JavaScript/TypeScript extractor — facts from the tree-sitter syntax tree (spec 65).

What is read from the tree: imports (ESM and `require`), string constants, `new X(...)` clients,
calls made on those clients, AI SDK calls (`generateText({ model: openai("gpt-4o") })`, gateway
strings like "openai/gpt-4o"), direct HTTP calls to catalog hosts, and datastore packages. The
provider always comes from the catalog (provedores.yml / armazenamentos.yml), never from a method
name. Since spec 69 the files are read together through `js_index` (constants, objects, clients
and factories followed to the module that defines them), memory and telemetry come from
integracoes.yml, and routes and tools (HTTP frameworks, Next, Workers, MCP; AI SDK, LangChain.js,
Agents SDK) come from the tree with name, description and parameters.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tree_sitter_javascript
import tree_sitter_typescript
from tree_sitter import Language, Node, Parser

from app.scanner import integration_catalog as ic
from app.scanner import provider_catalog as pc
from app.scanner import store_catalog as sc
from app.scanner.context import context_of
from app.scanner.extractors.http_calls import UNKNOWN as UNKNOWN_URL
from app.scanner.extractors.http_calls import classify as classify_url
from app.scanner.schema import Entrypoint, ExternalCall, Operation, ToolDef

if TYPE_CHECKING:
    from app.scanner.js_index import JsIndex

_JS_LANG = Language(tree_sitter_javascript.language())
_TS_LANG = Language(tree_sitter_typescript.language_typescript())
_TSX_LANG = Language(tree_sitter_typescript.language_tsx())

_JS_EXTS = {".js", ".jsx", ".mjs", ".cjs"}
_TS_EXTS = {".ts"}
_TSX_EXTS = {".tsx"}

# The AI SDK core functions; the provider is whatever their `model` resolves to.
_AI_SDK_PACKAGE = "ai"
_AI_SDK_CALLS = {
    "generateText": "invoke_llm",
    "streamText": "invoke_llm",
    "generateObject": "invoke_llm",
    "streamObject": "invoke_llm",
    "embed": "embed",
    "embedMany": "embed",
}
_GATEWAY = "vercel_ai_gateway"
_SCOPE_TYPES = {
    "function_declaration",
    "function_expression",
    "arrow_function",
    "method_definition",
    "function",
    "generator_function_declaration",
}
_FETCH_CALLEES = {"fetch", "$fetch", "ofetch", "globalThis.fetch", "window.fetch"}
_HTTP_CLIENT_METHODS = {"get", "post", "put", "patch", "delete", "request"}
_HTTP_CALLEES = {"fetch", "axios", "axios.post", "axios.request", "ky.post", "got.post"}

_FUNC_DECL_RE = re.compile(
    r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=|(?:async\s+)?(\w+)\s*\([^)]*\)\s*=>)"
)


@dataclass
class _JsFacts:
    entrypoints: list[Entrypoint] = field(default_factory=list)
    operations: list[Operation] = field(default_factory=list)
    tools: list[ToolDef] = field(default_factory=list)
    external_calls: list[ExternalCall] = field(default_factory=list)
    provider_tables: list[dict] = field(default_factory=list)


@dataclass
class _Client:
    provider: str
    origin: str
    model: str | None
    model_origin: str | None
    used: bool = False
    line: int = 0
    node: Node | None = None
    # spec 73: extra facts about how sure the provider is (provider_padrao, model_padrao)
    marks: dict[str, Any] = field(default_factory=dict)


def _language_for(path: Path) -> Language | None:
    if path.suffix in _JS_EXTS:
        return _JS_LANG
    if path.suffix in _TS_EXTS:
        return _TS_LANG
    if path.suffix in _TSX_EXTS:
        return _TSX_LANG
    return None


def _text(node: Node | None) -> str:
    return node.text.decode("utf-8", errors="ignore") if node is not None and node.text else ""


def _walk(node: Node) -> Iterator[Node]:
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(reversed(cur.children))


def _string_value(node: Node | None) -> str | None:
    if node is None:
        return None
    if node.type == "string":
        return "".join(_text(c) for c in node.children if c.type == "string_fragment")
    if node.type == "template_string" and not any(
        c.type == "template_substitution" for c in node.children
    ):
        return "".join(_text(c) for c in node.children if c.type == "string_fragment")
    return None


def _first_string_arg(call: Node) -> str | None:
    args = call.child_by_field_name("arguments")
    return _string_value(args.named_children[0]) if args and args.named_children else None


def _short(text: str, limit: int = 90) -> str:
    """Code shown as an origin: whitespace collapsed, cut on a word with an ellipsis (spec 71: the
    old fixed slice cut string literals in half, like `|| "`)."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0] + " …"


def _scope_id(node: Node) -> int:
    cur = node.parent
    while cur is not None:
        if cur.type in _SCOPE_TYPES:
            return cur.id
        cur = cur.parent
    return 0


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _enclosing_function_js(text: str, char_idx: int) -> str | None:
    last: re.Match[str] | None = None
    for m in _FUNC_DECL_RE.finditer(text[:char_idx]):
        last = m
    if last is None:
        return None
    return last.group(1) or last.group(2) or last.group(3)


def _prev_comment_js(text: str, char_idx: int) -> str | None:
    line_start = text.rfind("\n", 0, char_idx) + 1
    prev_block = text[:line_start].rstrip("\n")
    if not prev_block:
        return None
    prev_line = prev_block.rsplit("\n", 1)[-1].strip()
    if prev_line.startswith("//"):
        return prev_line.lstrip("/").strip() or None
    return None


# spec 69: routes and tools from the syntax tree
_HTTP_METHODS = {"get", "post", "put", "patch", "delete", "all", "options", "head"}
_SERVER_PACKAGES = ("express", "fastify", "hono", "koa", "@koa/router", "koa-router", "elysia")
_ROUTER_NAMES = {"app", "router", "server", "api", "route", "routes", "fastify"}
_HTTP_CLIENT_PACKAGES = ("axios", "ky", "got", "superagent", "node-fetch", "undici")
_NEXT_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
_TOOL_PACKAGES = ("ai", "@openai/agents", "@langchain/core/tools", "langchain/tools")

_TS_WRAPPERS = {
    "as_expression",
    "satisfies_expression",
    "non_null_expression",
    "parenthesized_expression",
    "type_assertion",
}
_FUNCTION_TYPES = {"function_declaration", "function_expression", "arrow_function", "function"}


def _unwrap(node: Node | None) -> Node | None:
    """`"gpt-4o" as const`, `(x)`, `x!` -> the expression inside."""
    while node is not None and node.type in _TS_WRAPPERS and node.named_children:
        node = node.named_children[0]
    return node


_ABSENT = {"null", "undefined"}
_LLM_KINDS = {"invoke_llm", "embed"}


def _client_expr(value: Node | None) -> Node | None:
    """The `new X(...)` a name holds: directly, or as the only real branch of a ternary whose
    other branch is null/undefined (spec 73: `KEY ? new OpenAI({...}) : null`)."""
    value = _unwrap(value)
    if value is None:
        return None
    if value.type == "new_expression":
        return value
    if value.type == "ternary_expression":
        branches = [
            _unwrap(value.child_by_field_name("consequence")),
            _unwrap(value.child_by_field_name("alternative")),
        ]
        built = [b for b in branches if b is not None and b.type == "new_expression"]
        absent = [b for b in branches if b is not None and _text(b) in _ABSENT]
        if len(built) == 1 and len(absent) == 1:
            return built[0]
    return None


def _returned_expr(fn: Node) -> Node | None:
    """What a function returns: the arrow body, or the first `return` of its own body."""
    body = fn.child_by_field_name("body")
    if body is None:
        return None
    if body.type != "statement_block":
        return body
    stack = list(reversed(body.children))
    while stack:
        cur = stack.pop()
        if cur.type == "return_statement":
            return cur.named_children[0] if cur.named_children else None
        if cur.type in _FUNCTION_TYPES or cur.type in _SCOPE_TYPES:
            continue  # a nested function's return is not this one's
        stack.extend(reversed(cur.children))
    return None


class _FileJS:
    def __init__(
        self, path: Path, root: Path, text: str, tree_root: Node, index: JsIndex | None = None
    ) -> None:
        self.index = index
        self.has_error = tree_root.has_error
        # spec 69: what this module exports, for the repository index
        self.exports: dict[str, Node] = {}  # exported name -> value node
        self.export_locals: dict[str, str] = {}  # export { a as b } -> b: a
        self.reexports: dict[str, tuple[str, str]] = {}  # export { a } from "./x" -> a: (x, a)
        self.star_sources: list[str] = []  # export * from "./x"
        self.client_by_node: dict[int, _Client] = {}
        self.rel = path.relative_to(root).as_posix()
        self.text = text
        self.root = tree_root
        self.lines = text.splitlines()
        self.imports: dict[str, tuple[str, str]] = {}  # local name -> (package, imported name)
        self.import_lines: dict[str, int] = {}  # package -> first line
        self.consts: dict[str, Node] = {}
        # (enclosing function node id or 0, name) -> value: the same name in two functions
        # (`const url = ...` for two providers) must not collapse into one
        self.scoped: dict[tuple[int, str], Node] = {}
        self.clients: dict[str, _Client] = {}
        self.ops: list[Operation] = []
        self.external: list[ExternalCall] = []
        # spec 73: calls that build a provider's model, and the ones an operation resolved through
        self.created_models: list[tuple[Node, tuple[str, str, str | None, str | None], str]] = []
        self.factory_marks: dict[int, dict[str, Any]] = {}  # factory call node id -> marks
        self.global_stores_seen: set[str] = set()
        self.consumed_models: set[int] = set()

    # --- collection -----------------------------------------------------------------------------

    def collect(self) -> None:
        for node in _walk(self.root):
            if node.type == "import_statement":
                self._import(node)
            elif node.type == "variable_declarator":
                self._declarator(node)
            elif node.type == "assignment_expression":
                self._field_client(node)
                self._commonjs_export(node)
            elif node.type == "export_statement":
                self._export(node)
            elif node.type == "call_expression":
                self._then_import(node)

    def _export(self, node: Node) -> None:
        source = _string_value(node.child_by_field_name("source"))
        decl = node.child_by_field_name("declaration")
        value = node.child_by_field_name("value")
        is_default = any(c.type == "default" for c in node.children)
        if decl is not None:
            if decl.type in ("lexical_declaration", "variable_declaration"):
                for d in decl.named_children:
                    name, val = d.child_by_field_name("name"), d.child_by_field_name("value")
                    if d.type == "variable_declarator" and name is not None and val is not None:
                        self.exports[_text(name)] = val
            else:
                name = decl.child_by_field_name("name")
                if name is not None:
                    self.exports[_text(name)] = decl
                if is_default:
                    self.exports["default"] = decl
            return
        if is_default and value is not None:
            self.exports["default"] = value
            return
        clause = next((c for c in node.children if c.type == "export_clause"), None)
        if clause is not None:
            for spec in clause.named_children:
                if spec.type != "export_specifier":
                    continue
                name = _text(spec.child_by_field_name("name"))
                alias = spec.child_by_field_name("alias")
                exported = _text(alias) if alias else name
                if source:
                    self.reexports[exported] = (source, name)
                else:
                    self.export_locals[exported] = name
            return
        if source and any(c.type == "*" for c in node.children):
            self.star_sources.append(source)

    def _commonjs_export(self, node: Node) -> None:
        """`module.exports = {...}`, `module.exports = x`, `module.exports.a = x`, `exports.a = x`."""
        left = _text(node.child_by_field_name("left"))
        right = _unwrap(node.child_by_field_name("right"))
        if right is None:
            return
        if left == "module.exports":
            if right.type == "object":
                for pair in right.named_children:
                    if pair.type == "pair":
                        key = _text(pair.child_by_field_name("key")).strip("\"'")
                        self.exports[key] = pair.child_by_field_name("value")
                    elif pair.type == "shorthand_property_identifier":
                        self.export_locals[_text(pair)] = _text(pair)
            self.exports.setdefault("default", right)
        elif left.startswith(("module.exports.", "exports.")):
            self.exports[left.rsplit(".", 1)[1]] = right

    def _field_client(self, node: Node) -> None:
        """`this.client = new OpenAI(...)`: the client lives in a field, named `this.client`."""
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if (
            left is not None
            and right is not None
            and _client_expr(right) is not None
            and left.type == "member_expression"
            and _text(left.child_by_field_name("object")) == "this"
        ):
            self.consts.setdefault(_text(left), right)

    def _import(self, node: Node) -> None:
        package = _string_value(node.child_by_field_name("source"))
        if not package:
            return
        self.import_lines.setdefault(package, node.start_point[0] + 1)
        for clause in node.children:
            if clause.type != "import_clause":
                continue
            for part in clause.children:
                if part.type == "identifier":
                    self.imports[_text(part)] = (package, "default")
                elif part.type == "namespace_import":
                    for c in part.children:
                        if c.type == "identifier":
                            self.imports[_text(c)] = (package, "*")
                elif part.type == "named_imports":
                    for spec in part.children:
                        if spec.type == "import_specifier":
                            name = _text(spec.child_by_field_name("name"))
                            alias = spec.child_by_field_name("alias")
                            self.imports[_text(alias) if alias else name] = (package, name)

    def _declarator(self, node: Node) -> None:
        name_node = node.child_by_field_name("name")
        value = node.child_by_field_name("value")
        if name_node is None or value is None:
            return
        while (
            value.type in ("await_expression", "parenthesized_expression") and value.named_children
        ):
            value = value.named_children[0]
        loader = (
            _text(value.child_by_field_name("function")) if value.type == "call_expression" else ""
        )
        if loader in ("require", "import"):
            # spec 73: `await import("pkg")` binds names like `require("pkg")` does; the bare
            # name is the module namespace (`mod.createOpenAI`), not a default export
            package = _first_string_arg(value)
            if package:
                self.import_lines.setdefault(package, node.start_point[0] + 1)
                self._bind_import(name_node, package, "*" if loader == "import" else "default")
            return
        if name_node.type == "identifier":
            name = _text(name_node)
            scope = _scope_id(node)
            self.scoped[(scope, name)] = value
            if scope == 0 or name not in self.consts:
                self.consts[name] = value

    def _bind_import(self, pattern: Node, package: str, bare: str) -> None:
        """Names a `require`/`import()` result is bound to: `x`, `{ a, b: c }`."""
        if pattern.type == "identifier":
            self.imports[_text(pattern)] = (package, bare)
        elif pattern.type == "object_pattern":
            for c in pattern.named_children:
                if c.type == "shorthand_property_identifier_pattern":
                    self.imports[_text(c)] = (package, _text(c))
                elif c.type == "pair_pattern":
                    key, local = c.child_by_field_name("key"), c.child_by_field_name("value")
                    if key is not None and local is not None and local.type == "identifier":
                        self.imports[_text(local)] = (package, _text(key))

    def _then_import(self, node: Node) -> None:
        """`import("pkg").then((m) => ...)` / `.then(({ a }) => ...)` (spec 73)."""
        func = node.child_by_field_name("function")
        if func is None or func.type != "member_expression":
            return
        if _text(func.child_by_field_name("property")) != "then":
            return
        source = func.child_by_field_name("object")
        if (
            source is None
            or source.type != "call_expression"
            or _text(source.child_by_field_name("function")) != "import"
        ):
            return
        package = _first_string_arg(source)
        args = node.child_by_field_name("arguments")
        callback = args.named_children[0] if args and args.named_children else None
        if not package or callback is None or callback.type not in _FUNCTION_TYPES:
            return
        self.import_lines.setdefault(package, node.start_point[0] + 1)
        param = callback.child_by_field_name("parameter")
        if param is None:
            params = callback.child_by_field_name("parameters")
            first = params.named_children[0] if params and params.named_children else None
            # TS wraps each parameter: required_parameter(pattern: ...)
            param = first.child_by_field_name("pattern") or first if first is not None else None
        if param is not None:
            self._bind_import(param, package, "*")

    def import_of(self, name: str) -> tuple[str, str] | None:
        """(package, imported name) for `name` or for `ns.name` when `ns` is a module namespace."""
        if name in self.imports:
            return self.imports[name]
        head, _, last = name.rpartition(".")
        if head in self.imports and self.imports[head][1] == "*" and "." not in head:
            return self.imports[head][0], last
        return None

    # --- routes and tools (spec 69) -------------------------------------------------------------

    def _imports_from(self, packages: tuple[str, ...]) -> bool:
        return any(
            p == pkg or p.startswith(pkg + "/") for p in self.import_lines for pkg in packages
        )

    def _schema_keys(self, node: Node | None, depth: int = 0) -> list[str]:
        """Parameter names of a tool schema: z.object({...}), a raw zod shape, jsonSchema({...})
        or a JSON schema object with `properties`, following constants."""
        node = _unwrap(node)
        if node is None or depth > 4:
            return []
        if node.type == "identifier":
            hit = self.lookup(node)
            return hit[0]._schema_keys(hit[1], depth + 1) if hit is not None else []
        if node.type == "call_expression":
            args = node.child_by_field_name("arguments")
            first = args.named_children[0] if args and args.named_children else None
            return self._schema_keys(first, depth + 1)
        if node.type == "member_expression":  # z.object({...}).describe(...) / .strict()
            obj = node.child_by_field_name("object")
            return self._schema_keys(obj, depth + 1)
        obj = self.as_object(node)
        if obj is None:
            return []
        props = self._object_prop(obj, ("properties",))
        if props is not None and props.type == "object":
            obj = props
        keys: list[str] = []
        for pair in obj.named_children:
            if pair.type == "pair":
                keys.append(_text(pair.child_by_field_name("key")).strip("\"'"))
            elif pair.type == "shorthand_property_identifier":
                keys.append(_text(pair))
        return keys

    def _string_prop(self, obj: Node | None, name: str) -> str:
        value, _ = self.resolve_string(self._object_prop(obj, (name,)))
        return value or ""

    def _tool(
        self, node: Node, name: str, obj: Node | None, schema_keys: tuple[str, ...]
    ) -> ToolDef:
        schema = next(
            (s for s in (self._object_prop(obj, (k,)) for k in schema_keys) if s is not None), None
        )
        return ToolDef(
            name=name,
            description=self._string_prop(obj, "description"),
            parameters=self._schema_keys(schema),
            file=self.rel,
            line=node.start_point[0] + 1,
        )

    def entry_facts(self) -> tuple[list[Entrypoint], list[ToolDef]]:
        entries: list[Entrypoint] = []
        tools: list[ToolDef] = []
        server_file = self._imports_from(_SERVER_PACKAGES)
        rel_parts = self.rel.split("/")
        stem = rel_parts[-1].rsplit(".", 1)[0]
        is_next_route = stem == "route" and "app" in rel_parts[:-1]
        is_pages_api = "pages" in rel_parts and "api" in rel_parts

        def add(kind: str, name: str, node: Node, **extra: str) -> None:
            entries.append(
                Entrypoint(
                    kind=kind, name=name, file=self.rel, line=node.start_point[0] + 1, extra=extra
                )
            )

        for node in self.root.named_children:
            if node.type != "export_statement":
                continue
            decl = node.child_by_field_name("declaration")
            value = node.child_by_field_name("value")
            names: list[str] = []
            if decl is not None and decl.type == "function_declaration":
                names = [_text(decl.child_by_field_name("name"))]
            elif decl is not None and decl.type in ("lexical_declaration", "variable_declaration"):
                names = [_text(d.child_by_field_name("name")) for d in decl.named_children]
            if is_next_route:
                for n in names:
                    if n in _NEXT_METHODS:
                        add("nextjs_route", f"{n} {self._next_path()}", node, method=n)
            if any(c.type == "default" for c in node.children):
                target = _unwrap(value) if value is not None else decl
                if is_pages_api and target is not None:
                    add("nextjs_api", self._next_path(), node)
                obj = self.as_object(target) if target is not None else None
                if obj is not None and (
                    self._object_prop(obj, ("fetch",)) is not None
                    or any(
                        c.type == "method_definition"
                        and _text(c.child_by_field_name("name")) == "fetch"
                        for c in obj.named_children
                    )
                ):
                    add("workers_fetch", "fetch", node)

        for node in _walk(self.root):
            if node.type == "call_expression":
                self._route_or_tool(node, server_file, add, tools)
            elif node.type == "new_expression":
                ctor = _text(node.child_by_field_name("constructor"))
                if ctor in ("DynamicStructuredTool", "DynamicTool", "StructuredTool"):
                    args = node.child_by_field_name("arguments")
                    obj = (
                        self.as_object(args.named_children[0])
                        if args and args.named_children
                        else None
                    )
                    tools.append(
                        self._tool(node, self._string_prop(obj, "name") or ctor, obj, ("schema",))
                    )
        return entries, tools

    def _next_path(self) -> str:
        parts = self.rel.split("/")
        start = (
            parts.index("app") + 1
            if "app" in parts
            else (parts.index("pages") + 1 if "pages" in parts else 0)
        )
        segs = [s for s in parts[start:-1] if not (s.startswith("(") and s.endswith(")"))]
        if parts[-1].rsplit(".", 1)[0] not in ("route", "index"):
            segs.append(parts[-1].rsplit(".", 1)[0])
        return "/" + "/".join(segs)

    def _route_or_tool(self, node: Node, server_file: bool, add, tools: list[ToolDef]) -> None:
        func = node.child_by_field_name("function")
        args = node.child_by_field_name("arguments")
        argv = list(args.named_children) if args is not None else []
        callee = _text(func)
        # tools: AI SDK / Agents SDK `tool({...})`, LangChain.js `tool(fn, {...})`
        if func is not None and func.type == "identifier" and callee == "tool":
            package = self.imports.get("tool", ("", ""))[0]
            if package and not any(
                package == p or package.startswith(p + "/") for p in _TOOL_PACKAGES
            ):
                return
            if (
                argv
                and argv[0].type == "object"
                or (argv and self.as_object(argv[0]) is not None and len(argv) == 1)
            ):
                obj = self.as_object(argv[0])
                parent = node.parent
                name = self._string_prop(obj, "name")
                if not name and parent is not None and parent.type == "pair":
                    name = _text(parent.child_by_field_name("key")).strip("\"'")
                if not name and parent is not None and parent.type == "variable_declarator":
                    name = _text(parent.child_by_field_name("name"))
                tools.append(
                    self._tool(node, name or "tool", obj, ("inputSchema", "parameters", "schema"))
                )
            elif len(argv) >= 2:
                obj = self.as_object(argv[1])
                tools.append(
                    self._tool(node, self._string_prop(obj, "name") or "tool", obj, ("schema",))
                )
            return
        if func is None or func.type != "member_expression":
            return
        method = _text(func.child_by_field_name("property"))
        receiver = _text(func.child_by_field_name("object"))
        root_name = receiver.split(".")[0]
        # MCP server tools: server.tool(name, [description], [shape], handler) / registerTool(name, {...})
        if method in ("tool", "registerTool") and argv:
            name, _ = self.resolve_string(argv[0])
            if name:
                description, schema = "", None
                if method == "registerTool" and len(argv) > 1:
                    obj = self.as_object(argv[1])
                    description = self._string_prop(obj, "description")
                    schema = self._object_prop(obj, ("inputSchema",))
                else:
                    rest = argv[1:-1]
                    if rest:
                        d, _ = self.resolve_string(rest[0])
                        if d is not None:
                            description, rest = d, rest[1:]
                    schema = rest[0] if rest else None
                tools.append(
                    ToolDef(
                        name=name,
                        description=description,
                        parameters=self._schema_keys(schema),
                        file=self.rel,
                        line=node.start_point[0] + 1,
                    )
                )
            return
        # HTTP routes: app.post("/chat", ...), router.get(...), fastify.route({ method, url })
        imported_from = self.imports.get(root_name, ("", ""))[0]
        if any(
            imported_from == p or imported_from.startswith(p + "/") for p in _HTTP_CLIENT_PACKAGES
        ):
            return
        if not (server_file or root_name in _ROUTER_NAMES):
            return
        if method in _HTTP_METHODS and argv:
            path, _ = self.resolve_string(argv[0])
            if path and path.startswith("/"):
                add("http_route", f"{method.upper()} {path}", node, method=method.upper())
        elif method == "route" and argv:
            obj = self.as_object(argv[0])
            url = self._string_prop(obj, "url") or self._string_prop(obj, "path")
            verb = self._string_prop(obj, "method")
            if url.startswith("/"):
                add(
                    "http_route",
                    f"{(verb or 'ALL').upper()} {url}",
                    node,
                    method=(verb or "ALL").upper(),
                )

    # --- resolution -----------------------------------------------------------------------------

    def const_of(self, ident: Node) -> Node | None:
        """Value of the constant an identifier refers to, nearest enclosing function first."""
        name = _text(ident)
        cur = ident.parent
        while cur is not None:
            if cur.type in _SCOPE_TYPES and (cur.id, name) in self.scoped:
                return self.scoped[(cur.id, name)]
            cur = cur.parent
        return self.scoped.get((0, name)) or self.consts.get(name)

    def lookup_name(self, name: str, depth: int = 0) -> tuple[_FileJS, Node] | None:
        """(module, value) of a top-level name of this module: a constant, a function, or what an
        import of it stands for in the module that defines it (spec 69)."""
        local = self.scoped.get((0, name)) or self.consts.get(name)
        if local is None:
            local = self._top_level_function(name)
        if local is not None:
            return self, local
        if self.index is not None and name in self.imports and depth < 8:
            package, imported = self.imports[name]
            target = self.index.resolve_specifier(self.rel, package)
            if target is not None and imported != "*":
                return self.index.export(target, imported, depth + 1)
        return None

    def lookup(self, ident: Node, depth: int = 0) -> tuple[_FileJS, Node] | None:
        """(module, value) an identifier refers to: nearest scope first, then imports."""
        local = self.const_of(ident)
        if local is not None:
            return self, local
        return self.lookup_name(_text(ident), depth)

    def _top_level_function(self, name: str) -> Node | None:
        for child in self.root.named_children:
            node = (
                child.child_by_field_name("declaration")
                if child.type == "export_statement"
                else child
            )
            if (
                node is not None
                and node.type == "function_declaration"
                and _text(node.child_by_field_name("name")) == name
            ):
                return node
        return None

    def member_target(self, node: Node, depth: int = 0) -> tuple[_FileJS, Node] | None:
        """(module, value) of `A.B.C` when A is a constant object (here or imported) or a namespace
        import: `AI_CONFIG.MODELS.CHAT` -> the literal it holds."""
        props: list[str] = []
        cur: Node | None = node
        while cur is not None and cur.type == "member_expression":
            props.append(_text(cur.child_by_field_name("property")))
            cur = _unwrap(cur.child_by_field_name("object"))
        if cur is None or cur.type != "identifier" or depth > 8:
            return None
        props.reverse()
        name = _text(cur)
        hit: tuple[_FileJS, Node] | None = None
        package, imported = self.imports.get(name, ("", ""))
        target = (
            self.index.resolve_specifier(self.rel, package)
            if self.index is not None and imported in ("*", "default") and props
            else None
        )
        if target is not None and (
            imported == "*" or "default" not in self.index.files[target].exports
        ):
            # namespace import, or `require()` of a module that only has named exports
            # (`exports.EMBED = ...`): the first property is the export's name
            hit = self.index.export(target, props.pop(0), depth + 1)
        elif imported != "*":
            hit = self.lookup(cur, depth + 1)
        for prop in props:
            if hit is None:
                return None
            owner, value = hit
            obj = owner.as_object(value)
            prop_node = owner._object_prop(obj, (prop,))
            if prop_node is None:
                return None
            if prop_node.type in ("identifier", "shorthand_property_identifier"):
                hit = owner.lookup_name(_text(prop_node), depth + 1)
            else:
                hit = (owner, prop_node)
        return hit

    def resolve_string(self, node: Node | None, depth: int = 0) -> tuple[str | None, str | None]:
        """(value, origin) — value only when fixed in the code."""
        if node is None:
            return None, "não informado no código (vale o padrão do serviço)"
        node = _unwrap(node)
        literal = _string_value(node)
        if literal is not None:
            return literal, None
        if node.type == "identifier" and depth < 5:
            name = _text(node)
            hit = self.lookup(node)
            if hit is not None:
                owner, value_node = hit
                value, origin = owner.resolve_string(value_node, depth + 1)
                return value, origin or (f"constante {name}" if value else None)
            return None, f"variável {name}"
        if (
            node.type == "member_expression"
            and depth < 5
            and not _text(node).startswith("process.env.")
        ):
            hit = self.member_target(node)
            if hit is not None:
                owner, value_node = hit
                value, origin = owner.resolve_string(value_node, depth + 1)
                if value is not None:
                    return value, origin or f"constante {_text(node)[:60]}"
        if node.type == "binary_expression" and _text(node.child_by_field_name("operator")) in (
            "||",
            "??",
        ):
            left = _text(node.child_by_field_name("left"))[:50]
            fallback, _ = self.resolve_string(node.child_by_field_name("right"), depth + 1)
            if fallback is not None:
                return None, f"{left} (padrão {fallback!r})"
            return None, _short(_text(node))
        txt = _text(node)
        if txt.startswith("process.env."):
            return None, f"variável de ambiente {txt.removeprefix('process.env.')}"
        return None, _short(txt) or "não resolvido"

    def url_text(
        self, node: Node | None, depth: int = 0, unknown: str = "", defaults: bool = True
    ) -> str:
        """Every part of a URL expression the code fixes: literals, constants, template pieces and
        the literal fallback of `a || 'x'`. Unknown parts become `unknown`. Used to find a host.
        With `defaults=False` a fallback does not count: only what cannot be overridden."""
        if node is None or depth > 6:
            return unknown
        literal = _string_value(node)
        if literal is not None:
            return literal
        if node.type == "template_string":
            parts: list[str] = []
            for c in node.children:
                if c.type == "string_fragment":
                    parts.append(_text(c))
                elif c.type == "template_substitution" and c.named_children:
                    parts.append(self.url_text(c.named_children[0], depth + 1, unknown, defaults))
            return "".join(parts)
        if node.type == "identifier":
            hit = self.lookup(node)
            return (
                hit[0].url_text(hit[1], depth + 1, unknown, defaults)
                if hit is not None
                else unknown
            )
        if node.type == "member_expression" and not _text(node).startswith("process.env."):
            hit = self.member_target(node)
            return (
                hit[0].url_text(hit[1], depth + 1, unknown, defaults)
                if hit is not None
                else unknown
            )
        if node.type == "parenthesized_expression" and node.named_children:
            return self.url_text(node.named_children[0], depth + 1, unknown, defaults)
        if node.type == "binary_expression":
            op = _text(node.child_by_field_name("operator"))
            left = self.url_text(node.child_by_field_name("left"), depth + 1, unknown, defaults)
            right = self.url_text(node.child_by_field_name("right"), depth + 1, unknown, defaults)
            if op in ("||", "??"):
                if not defaults:
                    return left if left else unknown
                return left if left and left != unknown else (right or unknown)
            if op == "+":
                return left + right
        return unknown

    def _default_host_marks(self, node: Node | None, provider_id: str) -> dict[str, Any]:
        """Spec 73: the host comes only from an overridable fallback (`process.env.X || "https://
        api.groq.com"`): the provider is claimed and marked configurable."""
        fixed = self.url_text(node, defaults=False)
        hit = pc.by_host(fixed) if "://" in fixed else None
        return {} if hit is not None and hit.id == provider_id else {"provider_padrao": True}

    def _external_call(self, node: Node, callee: str) -> None:
        """Spec 71: an outbound HTTP call, listed by host in the documents. A relative URL is the
        system's own route; a URL the code does not fix is kept with url=None, to be counted."""
        parts = callee.split(".")
        if callee not in _FETCH_CALLEES and not (
            parts[0] in _HTTP_CLIENT_PACKAGES
            and parts[0] in self.imports
            and (len(parts) == 1 or parts[-1] in _HTTP_CLIENT_METHODS)
        ):
            return
        args = node.child_by_field_name("arguments")
        first = args.named_children[0] if args and args.named_children else None
        if first is not None and first.type == "object":  # axios({ url })
            first = self._object_prop(first, ("url",))
        url = classify_url(self.url_text(first, unknown=UNKNOWN_URL))
        if url == "":
            return
        self.external.append(
            ExternalCall(
                url=url,
                method=parts[-1].upper() if len(parts) > 1 else "FETCH",
                file=self.rel,
                line=node.start_point[0] + 1,
                library=parts[0],
            )
        )

    def as_object(self, node: Node | None, depth: int = 0) -> Node | None:
        """An object literal, directly or through a constant."""
        node = _unwrap(node)
        if node is None or depth > 5:
            return None
        if node.type == "object":
            return node
        if node.type == "identifier":
            hit = self.lookup(node)
            # an object from another module is read in that module (its own constants)
            return hit[0].as_object(hit[1], depth + 1) if hit is not None else None
        if node.type == "member_expression":
            hit = self.member_target(node)
            return hit[0].as_object(hit[1], depth + 1) if hit is not None else None
        return None

    def body_model(self, args: Node | None) -> tuple[str | None, str | None]:
        """Model sent in an HTTP body: `axios.post(url, { model })` or
        `fetch(url, { body: JSON.stringify({ model }) })`, following constants."""
        if args is None or len(args.named_children) < 2:
            return None, None
        second = self.as_object(args.named_children[1])
        payload = second
        body = self._object_prop(second, ("body",))
        if body is not None and body.type == "call_expression":
            inner = body.child_by_field_name("arguments")
            payload = (
                self.as_object(inner.named_children[0]) if inner and inner.named_children else None
            )
        model_node = self._object_prop(payload, ("model",))
        if model_node is None:
            return None, None
        return self.resolve_string(model_node)

    def _object_prop(self, obj: Node | None, names: list[str] | tuple[str, ...]) -> Node | None:
        if obj is None or obj.type != "object":
            return None
        for pair in obj.named_children:
            if pair.type == "pair":
                key = _text(pair.child_by_field_name("key")).strip("\"'")
                if key in names:
                    return pair.child_by_field_name("value")
            elif pair.type == "shorthand_property_identifier" and _text(pair) in names:
                return pair
        return None

    def _first_object_arg(self, call: Node) -> Node | None:
        args = call.child_by_field_name("arguments")
        if args is None:
            return None
        return next((a for a in args.named_children if a.type == "object"), None)

    def _model_from_expr(
        self, expr: Node | None, depth: int = 0
    ) -> tuple[str, str, str | None, str | None, str | None, dict[str, Any]]:
        """(provider, provider_origin, model, model_origin, intermediario, marks) for an AI SDK
        `model:`. `marks` goes into the operation's extra (spec 73: a default is configurable)."""
        expr = _unwrap(expr)
        if expr is None:
            return (
                pc.UNRESOLVED,
                "argumento model ausente",
                None,
                "não informado no código",
                None,
                {},
            )
        if (
            expr.type == "binary_expression"
            and _text(expr.child_by_field_name("operator")) in ("??", "||")
            and depth < 6
        ):
            # spec 73: `opts.model ?? openai("gpt-4o")`. What the left side fixes wins; otherwise
            # the default names the provider (claimed, marked configurable) and its model stays
            # an origin, never a fact (D-64).
            left_node = expr.child_by_field_name("left")
            left = self._model_from_expr(left_node, depth + 1)
            if left[0] != pc.UNRESOLVED:
                # the fallback never runs for what the left side fixes: it is not a fact
                for inner in _walk(expr.child_by_field_name("right")):
                    self.consumed_models.add(inner.id)
                return left
            pid, porigin, model, morigin, inter, marks = self._model_from_expr(
                expr.child_by_field_name("right"), depth + 1
            )
            if pid != pc.UNRESOLVED:
                shown = _short(_text(left_node), 50)
                marks = {**marks, "provider_padrao": True}
                if model:
                    marks["model_padrao"] = model
                return (
                    pid,
                    f"padrão de {shown}: {porigin}",
                    None,
                    f"{shown} (padrão {model!r})" if model else morigin,
                    inter,
                    marks,
                )
        if expr.type in ("identifier", "member_expression") and depth < 6:
            hit = (
                self.lookup(expr)
                if expr.type == "identifier"
                else (None if _text(expr).startswith("process.env.") else self.member_target(expr))
            )
            if hit is not None and hit[1].type not in _FUNCTION_TYPES:
                return hit[0]._model_from_expr(hit[1], depth + 1)
        if expr.type == "call_expression" and depth < 6:
            # model: getModel() -> what the function returns, in the module that defines it
            fn = expr.child_by_field_name("function")
            hit = self.lookup(fn) if fn is not None and fn.type == "identifier" else None
            if hit is not None and hit[1].type in _FUNCTION_TYPES:
                ret = _returned_expr(hit[1])
                if ret is not None:
                    return hit[0]._model_from_expr(ret, depth + 1)
        if expr.type == "call_expression":
            made = self._provider_model_call(expr)
            if made is not None:
                self.consumed_models.add(expr.id)
                return (*made, None, self._marks_of_model_call(expr))
        value, origin = self.resolve_string(expr)
        if value:
            hit = pc.by_prefix(value)
            if hit is not None:
                provider, model = hit
                return (
                    provider.id,
                    f"prefixo {provider.id} em {value!r} (gateway do AI SDK)",
                    model,
                    None,
                    _GATEWAY,
                    {},
                )
            return (
                pc.UNRESOLVED,
                f"modelo {value!r} sem provedor identificável",
                None,
                f"valor {value!r}",
                None,
                {},
            )
        return pc.UNRESOLVED, f"modelo não resolvido: {origin}", None, origin, None, {}

    def _provider_model_call(self, expr: Node) -> tuple[str, str, str | None, str | None] | None:
        """(provider, provider_origin, model, model_origin) when the call builds a model of a
        catalog provider: `openai("gpt-4o")` from the provider's package, or a call on a factory
        instance (`const groq = createOpenAI({ baseURL }); groq("llama-3.3")`)."""
        func = expr.child_by_field_name("function")
        root_name = _text(func).split(".")[0]
        args = expr.child_by_field_name("arguments")
        first = args.named_children[0] if args and args.named_children else None
        if root_name in self.imports:
            package, _ = self.imports[root_name]
            provider = pc.js_package(package)
            if provider is not None and not self._is_factory_call(func):
                model, origin = self.resolve_string(first)
                return provider.id, f"pacote {package}", model, origin
        factory_ref = func
        if (
            func is not None
            and func.type == "member_expression"
            and _text(func.child_by_field_name("property")) in pc.load_catalog().js_factory_methods
        ):
            factory_ref = func.child_by_field_name("object")  # openai.embedding(...)
        factory = (
            self.const_of(factory_ref)
            if factory_ref is not None and factory_ref.type == "identifier"
            else None
        )
        if factory is not None and factory.type == "call_expression":
            pid_origin = self._factory_provider(factory)
            if pid_origin is not None:
                model, origin = self.resolve_string(first)
                return pid_origin[0], pid_origin[1], model, origin
        return None

    def _marks_of_model_call(self, expr: Node) -> dict[str, Any]:
        """Marks of the factory a model call goes through (a default baseURL host)."""
        func = expr.child_by_field_name("function")
        ref = func.child_by_field_name("object") if func.type == "member_expression" else func
        factory = self.const_of(ref) if ref is not None and ref.type == "identifier" else None
        return dict(self.factory_marks.get(factory.id, {})) if factory is not None else {}

    def _note_created_model(self, node: Node) -> None:
        """Spec 73: remember every call that builds a provider's model with the AI SDK shape
        (`openai("gpt-4o")`, `factory(modelId)`), so one that no operation accounts for is not
        lost (the model reached the call through an options object, an adapter, a wrapper)."""
        func = node.child_by_field_name("function")
        if func is None or func.type not in ("identifier", "member_expression"):
            return
        name = _text(func)
        root = name.split(".")[0]
        direct = func.type == "identifier"
        method = name.rsplit(".", 1)[-1] in pc.load_catalog().js_factory_methods
        if not (direct or (method and name.count(".") == 1)):
            return
        if root in self.imports:
            # only a provider factory exported by an AI SDK provider package builds a model
            package, imported = self.imports[root]
            provider = pc.js_package(package)
            if provider is None or imported not in provider.js.fabricas:
                return
            if imported.startswith("create"):
                return  # createOpenAI({...}) builds the factory; its instance builds the model
        made = self._provider_model_call(node)
        if made is not None and made[0] not in (pc.UNRESOLVED, pc.OPENAI_COMPATIBLE):
            self.created_models.append((node, made, "embed" if "mbed" in name else "invoke_llm"))

    def unexplained_models(self, explained: set[str]) -> None:
        """Emit one operation per provider whose model this file builds and no operation of the
        repository accounts for. A model that is only the default of `??`/`||` keeps the rule of
        the configurable default: provider claimed and marked, model as origin."""
        done: set[tuple[str, str]] = set()
        for node, (pid, porigin, model, morigin), kind in self.created_models:
            if node.id in self.consumed_models or pid in explained or (pid, kind) in done:
                continue
            done.add((pid, kind))
            marks: dict[str, Any] = self._marks_of_model_call(node)
            parent = node.parent
            while parent is not None and parent.type in _TS_WRAPPERS:
                parent = parent.parent
            if (
                parent is not None
                and parent.type == "binary_expression"
                and _text(parent.child_by_field_name("operator")) in ("??", "||")
                and parent.child_by_field_name("left").id != node.id
            ):
                shown = _short(_text(parent.child_by_field_name("left")), 50)
                marks["provider_padrao"] = True
                if model:
                    marks["model_padrao"] = model
                porigin = f"padrão de {shown}: {porigin}"
                model, morigin = None, f"{shown} (padrão {model!r})" if model else morigin
            self._emit(node, kind, pid, porigin, model, morigin, "modelo_criado", marks=marks)

    # --- operations -----------------------------------------------------------------------------

    def _factory_provider(self, call: Node) -> tuple[str, str] | None:
        """Provider of `createOpenAI({ baseURL })`-style factories: the package, overridden by the
        host of a baseURL the code fixes (an OpenAI-shaped factory pointed at Groq is Groq)."""
        name = _text(call.child_by_field_name("function"))
        imported = self.import_of(name)
        if imported is None:
            return None
        package = imported[0]
        provider = pc.js_package(package)
        if provider is None:
            return None
        opts = self._first_object_arg(call)
        base = self._object_prop(opts, pc.load_catalog().base_url_kwargs)
        if base is not None and provider.id == "openai":
            url = self.url_text(base)
            hit = pc.by_host(url) if url else None
            if hit is not None:
                self.factory_marks[call.id] = self._default_host_marks(base, hit.id)
                return hit.id, f"host do baseURL {url} ({name})"
            return (
                pc.OPENAI_COMPATIBLE,
                f"baseURL {url or _text(base)[:60]} não é de provedor catalogado",
            )
        return provider.id, f"fábrica {name} de {package}"

    def _is_factory_call(self, func: Node | None) -> bool:
        """`mod.createOpenAI` on a module namespace builds a factory; it is not a model call."""
        name = _text(func)
        return "." in name and name.rsplit(".", 1)[1].startswith("create")

    def _client_from_new(self, node: Node) -> _Client | None:
        ctor = _text(node.child_by_field_name("constructor"))
        imported = self.import_of(ctor)
        if imported is None:
            return None
        package = imported[0]
        if pc.is_js_model_agent(package):
            # Mastra: new Agent({ model: openai("gpt-4o") }) — the model expression decides
            opts = next(
                (
                    a
                    for a in (node.child_by_field_name("arguments") or node).named_children
                    if a.type == "object"
                ),
                None,
            )
            pid, porigin, model, morigin, _inter, marks = self._model_from_expr(
                self._object_prop(opts, ("model",))
            )
            return _Client(
                pid, porigin, model, morigin, line=node.start_point[0] + 1, node=node, marks=marks
            )
        provider = pc.js_package(package)
        if provider is None:
            return None
        args = node.child_by_field_name("arguments")
        obj = next((a for a in args.named_children if a.type == "object"), None) if args else None
        pid, origin = provider.id, f"classe {ctor} de {package}"
        marks: dict[str, Any] = {}
        base = self._object_prop(obj, pc.load_catalog().base_url_kwargs)
        if base is not None and provider.id == "openai":
            value = self.url_text(base) or None
            why = _text(base)[:60]
            hit = pc.by_host(value) if value else None
            if hit is not None:
                pid, origin = hit.id, f"host do baseURL {value}"
                marks = self._default_host_marks(base, hit.id)
            else:
                pid, origin = (
                    pc.OPENAI_COMPATIBLE,
                    f"baseURL {value or why} não é de provedor catalogado",
                )
        model, model_origin = self.resolve_string(
            self._object_prop(obj, pc.load_catalog().model_kwargs)
        )
        return _Client(
            pid, origin, model, model_origin, line=node.start_point[0] + 1, node=node, marks=marks
        )

    def run(self) -> None:
        self.make_clients()
        self.calls()
        self.finish()

    def make_clients(self) -> None:
        # 1) clients assigned to names: const openai = new OpenAI(...)
        for name, value in list(self.consts.items()):
            built = _client_expr(value)
            if built is not None:
                client = self._client_from_new(built)
                if client is not None:
                    self.clients[name] = client
                    self.client_by_node[value.id] = client

    def calls(self) -> None:
        # 2) calls
        for node in _walk(self.root):
            if node.type == "call_expression":
                self._call(node)
                self._note_created_model(node)

    def finish(self) -> None:
        # 3) clients never called become constructor-level operations
        for client in self.clients.values():
            if not client.used and client.node is not None:
                self._emit(
                    client.node,
                    "invoke_llm",
                    client.provider,
                    client.origin,
                    client.model,
                    client.model_origin,
                    "construtor",
                    marks=client.marks,
                )
        # 4) datastores and vector stores by package
        seen: set[str] = set()
        for package, line in self.import_lines.items():
            store = sc.js_package(package)
            if store is None or store.id in seen:
                continue
            seen.add(store.id)
            kind = "vector_store_init" if store.is_vector else "datastore"
            self.ops.append(
                Operation(
                    kind=kind,
                    provider=store.id if store.is_vector else None,
                    file=self.rel,
                    line=line,
                    snippet=self.lines[line - 1].strip() if line <= len(self.lines) else "",
                    extra={"store": store.id, "store_origin": f"pacote {package}"},
                )
            )
        self._sql_stores(seen)
        # 5) conversation memory and telemetry by package (spec 69, integracoes.yml)
        seen_integrations: set[str] = set()
        for package, line in self.import_lines.items():
            integration = ic.js_package(package)
            if integration is None or integration.id in seen_integrations:
                continue
            seen_integrations.add(integration.id)
            self._integration_op(integration, line, f"pacote {package}")
        for package, name in self.imports.values():
            integration = ic.js_name(package, name)
            if integration is None or integration.id in seen_integrations:
                continue
            seen_integrations.add(integration.id)
            line = self.import_lines.get(package, 1)
            self._integration_op(integration, line, f"{name} de {package}")
        # 6) libraries that are themselves an external service (spec 73)
        seen_services: set[str] = set()
        for package, line in self.import_lines.items():
            service = ic.js_service(package)
            if service is None or service.id in seen_services:
                continue
            seen_services.add(service.id)
            self.external.append(
                ExternalCall(
                    url=f"https://{service.host}",
                    method="BIBLIOTECA",
                    file=self.rel,
                    line=line,
                    library=package,
                )
            )
        for node in _walk(self.root):
            if node.type == "member_expression" and _text(node).startswith("process.env."):
                service = ic.service_by_env(_text(node).removeprefix("process.env."))
                if service is not None and service.id not in seen_services:
                    seen_services.add(service.id)
                    self.external.append(
                        ExternalCall(
                            url=f"https://{service.host}",
                            method="BIBLIOTECA",
                            file=self.rel,
                            line=node.start_point[0] + 1,
                            library=_text(node),
                        )
                    )
                integration = ic.by_env(_text(node).removeprefix("process.env."))
                if integration is not None and integration.id not in seen_integrations:
                    seen_integrations.add(integration.id)
                    self._integration_op(integration, node.start_point[0] + 1, _text(node))

    def _global_store(self, node: Node, name: str) -> None:
        store = sc.get(self.index.global_stores[name]) if self.index is not None else None
        if store is None or store.id in self.global_stores_seen:
            return
        self.global_stores_seen.add(store.id)
        line = node.start_point[0] + 1
        self.ops.append(
            Operation(
                kind="vector_store_init" if store.is_vector else "datastore",
                provider=store.id if store.is_vector else None,
                file=self.rel,
                line=line,
                snippet=self.lines[line - 1].strip()[:200] if line <= len(self.lines) else "",
                extra={"store": store.id, "store_origin": f"{name}() injetado pelo framework"},
            )
        )

    def provider_tables(self) -> list[dict]:
        """Spec 73: object literals whose values are addresses of two or more catalog providers
        (`const UPSTREAMS = { anthropic: "https://api.anthropic.com", openai: "https://..." }`)."""
        from app.scanner.provider_tables import providers_of

        out: list[dict] = []
        for node in _walk(self.root):
            if node.type != "object":
                continue
            urls = [
                value
                for pair in node.named_children
                if pair.type == "pair"
                for value in [_string_value(_unwrap(pair.child_by_field_name("value")))]
                if value
            ]
            providers = providers_of(urls)
            if not providers:
                continue
            holder = node.parent
            while holder is not None and holder.type in _TS_WRAPPERS:
                holder = holder.parent
            name = (
                _text(holder.child_by_field_name("name"))
                if holder is not None and holder.type == "variable_declarator"
                else ""
            )
            out.append(
                {
                    "file": self.rel,
                    "line": node.start_point[0] + 1,
                    "nome": name,
                    "provedores": providers,
                }
            )
        return out

    def _sql_stores(self, seen: set[str]) -> None:
        """Spec 73: SQL inside a string of the code that proves a store (pgvector)."""
        for node in _walk(self.root):
            if (
                node.type not in ("string", "template_string")
                or node.end_byte - node.start_byte < 14
            ):
                continue
            store = sc.by_sql(_text(node))
            if store is None or store.id in seen:
                continue
            seen.add(store.id)
            line = node.start_point[0] + 1
            self.ops.append(
                Operation(
                    kind="vector_store_init" if store.is_vector else "datastore",
                    provider=store.id if store.is_vector else None,
                    file=self.rel,
                    line=line,
                    snippet=self.lines[line - 1].strip()[:200] if line <= len(self.lines) else "",
                    extra={"store": store.id, "store_origin": "SQL em string do código"},
                )
            )

    def _integration_op(self, integration: ic.Integration, line: int, symbol: str) -> None:
        self.ops.append(
            Operation(
                kind=integration.kind,
                provider=integration.id,
                file=self.rel,
                line=line,
                snippet=self.lines[line - 1].strip()[:200] if line <= len(self.lines) else "",
                extra={"symbol": symbol},
            )
        )

    def _call(self, node: Node) -> None:
        func = node.child_by_field_name("function")
        callee = _text(func)
        self._external_call(node, callee)
        root_name = callee.split(".")[0].split("(")[0]
        if root_name == "this" and callee.count(".") >= 2:
            root_name = ".".join(callee.split(".")[:2])  # this.client.responses.create

        # AI SDK core function imported from "ai"
        if callee in self.imports and self.imports[callee][0] == _AI_SDK_PACKAGE:
            kind = _AI_SDK_CALLS.get(self.imports[callee][1])
            if kind:
                obj = self._first_object_arg(node)
                provider, origin, model, model_origin, inter, marks = self._model_from_expr(
                    self._object_prop(obj, ("model",))
                )
                self._emit(
                    node, kind, provider, origin, model, model_origin, "chamada", inter, marks
                )
            return

        # call on a client: openai.chat.completions.create({...}), ollama.chat({...})
        if func is not None and func.type == "member_expression" and "." in callee:
            kind = pc.call_kind(callee.split(".", 1)[1])
            client = self.clients.get(root_name) if kind else None
            if kind and client is None and root_name in self.imports and self.index is not None:
                # spec 69: a client created in another module and imported here
                hit = self.lookup_name(root_name)
                if hit is not None:
                    client = hit[0].client_by_node.get(hit[1].id)
            if kind and client is None and root_name in self.imports:
                package, imported = self.imports[root_name]
                provider = pc.js_package(package)
                if (
                    provider is not None
                    and imported in ("default", "*")
                    and root_name not in provider.js.classes
                ):
                    client = _Client(provider.id, f"módulo {package}", None, None)
            if kind and client is not None:
                client.used = True
                obj = self._first_object_arg(node)
                model_node = self._object_prop(obj, pc.load_catalog().model_kwargs)
                if model_node is not None:
                    model, model_origin = self.resolve_string(model_node)
                else:
                    model, model_origin = client.model, client.model_origin
                elsewhere = pc.call_model_origin(callee.split(".", 1)[1])
                stored = pc.call_stores_data(callee.split(".", 1)[1])
                if model is None and elsewhere and model_node is None:
                    model_origin = elsewhere
                if model is None and model_origin is None:
                    model_origin = "não informado no código (vale o padrão do serviço)"
                self._emit(
                    node,
                    kind,
                    client.provider,
                    client.origin,
                    model,
                    model_origin,
                    "chamada",
                    marks=(
                        {**(client.marks or {}), "guarda_no_provedor": stored}
                        if stored
                        else client.marks
                    ),
                )
                return

        # spec 73: functions injected with no import (NuxtHub). hubAI().run(model, ...) is the
        # Workers AI call; hubDatabase(), hubVectorize(), hubBlob() are where the data is kept.
        if self.index is not None and func is not None:
            if func.type == "identifier" and callee in self.index.global_stores:
                self._global_store(node, callee)
            target = (
                func.child_by_field_name("object") if func.type == "member_expression" else None
            )
            injected = (
                _text(target.child_by_field_name("function"))
                if target is not None and target.type == "call_expression"
                else ""
            )
            if (
                injected in self.index.global_providers
                and _text(func.child_by_field_name("property")) == "run"
            ):
                args = node.child_by_field_name("arguments")
                first = args.named_children[0] if args and args.named_children else None
                model, model_origin = self.resolve_string(first)
                self._emit(
                    node,
                    "invoke_llm",
                    self.index.global_providers[injected],
                    f"{injected}() injetado pelo framework",
                    model,
                    model_origin,
                    "chamada",
                )
                return

        # Workers AI binding: env.AI.run("@cf/meta/llama-3.1-8b-instruct", {...}) (spec 69)
        parts = callee.split(".")
        if (
            self.index is not None
            and len(parts) >= 2
            and parts[-1] == "run"
            and parts[-2] in self.index.ai_bindings
        ):
            args = node.child_by_field_name("arguments")
            first = args.named_children[0] if args and args.named_children else None
            model, model_origin = self.resolve_string(first)
            self._emit(
                node,
                "invoke_llm",
                self.index.ai_bindings[parts[-2]],
                f"binding {parts[-2]} do wrangler.toml",
                model,
                model_origin,
                "chamada",
            )
            return

        # direct HTTP to a catalog host
        if callee in _HTTP_CALLEES:
            args = node.child_by_field_name("arguments")
            first = args.named_children[0] if args and args.named_children else None
            url = self.url_text(first)
            if not url:
                return
            model, model_origin = self.body_model(args)
            gapped = self.url_text(first, unknown=UNKNOWN_URL)
            if pc.azure_path_in(url):
                self._emit(
                    node,
                    "invoke_llm",
                    "azure_openai",
                    f"caminho Azure em {url!r}",
                    model,
                    model_origin or "deployment em variável",
                    "http",
                )
                return
            hit = pc.by_host(url) if "://" in url else None
            if hit is not None:
                if model is None:
                    # spec 73: .../models/gemini-2.0-flash:generateContent
                    named = pc.model_in_url(hit.id, gapped, UNKNOWN_URL)
                    if named:
                        model, model_origin = named, "caminho da URL"
                self._emit(
                    node,
                    "invoke_llm",
                    hit.id,
                    f"host em {url!r}",
                    model,
                    model_origin or "não informado no código",
                    "http",
                    marks=self._default_host_marks(first, hit.id),
                )

    def _emit(
        self,
        node: Node,
        kind: str,
        provider: str,
        provider_origin: str,
        model: str | None,
        model_origin: str | None,
        via: str,
        intermediario: str | None = None,
        marks: dict[str, Any] | None = None,
    ) -> None:
        extra: dict[str, Any] = {"via": via, "provider_origin": provider_origin, **(marks or {})}
        if kind == "invoke_llm" and model and pc.is_embedding_model(provider, model):
            kind = "embed"  # spec 73: the same call serves chat and embeddings (Workers AI)
        if intermediario:
            extra["intermediario"] = intermediario
        if model is None:
            extra["model_origin"] = model_origin or "não resolvido"
        elif model_origin:
            extra["model_origin"] = model_origin
        line = node.start_point[0] + 1
        offset = node.start_byte
        self.ops.append(
            Operation(
                kind=kind,
                provider=provider,
                model=model,
                file=self.rel,
                line=line,
                snippet=self.lines[line - 1].strip()[:200] if line <= len(self.lines) else "",
                extra=extra,
                enclosing_function=_enclosing_function_js(self.text, offset),
                prev_comment=_prev_comment_js(self.text, offset),
            )
        )


def _parse_file(path: Path, root: Path, index: JsIndex | None = None) -> _FileJS | None:
    language = _language_for(path)
    if language is None:
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    # tree-sitter recovers from syntax errors; a partial tree still yields its facts.
    tree = Parser(language).parse(text.encode())
    js = _FileJS(path, root, text, tree.root_node, index)
    js._tree = tree  # keep the tree alive as long as its nodes are used
    return js


def _entry_facts(js: _FileJS) -> _JsFacts:
    entrypoints, tools = js.entry_facts()
    return _JsFacts(entrypoints=entrypoints, tools=tools)


def analyze_js_repo(root: Path, paths: list[Path]) -> list[_JsFacts]:
    """All JS/TS files of a repository, read together (spec 69): clients of every module first,
    then every call — so a client exported by one module and called in another counts as used —
    then what was never called."""
    from app.scanner.js_index import JsIndex

    index = JsIndex.build(root, paths)
    files = list(index.files.values())
    for js in files:
        js.make_clients()
    for js in files:
        js.calls()
    # spec 73: a model built for a provider that no operation of the repository reaches
    # (a call in a test or an example does not account for what product code builds)
    explained = {
        str(op.provider)
        for js in files
        if not context_of(js.rel)
        for op in js.ops
        if op.kind in _LLM_KINDS
    }
    for js in files:
        js.unexplained_models(explained)
    out: list[_JsFacts] = []
    for js in files:
        js.finish()
        facts = _entry_facts(js)
        for op in js.ops:
            if js.has_error:
                op.extra["arvore_parcial"] = True
        facts.operations.extend(js.ops)
        facts.external_calls.extend(js.external)
        facts.provider_tables.extend(js.provider_tables())
        out.append(facts)
    return out


def analyze_js_file(path: Path, root: Path) -> _JsFacts:
    """One file on its own (tests and single-file callers)."""
    results = analyze_js_repo(root, [path])
    return results[0] if results else _JsFacts()


def iter_js_files(root: Path) -> list[Path]:
    ignored = {".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build", ".next"}
    exts = _JS_EXTS | _TS_EXTS | _TSX_EXTS
    return [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in exts
        and not p.name.endswith(".d.ts")
        and not any(part in ignored for part in p.relative_to(root).parts)
    ]
