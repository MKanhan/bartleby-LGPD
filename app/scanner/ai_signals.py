"""Indícios de uso de IA no repositório — a rede contra a omissão silenciosa (spec 65).

O scanner pode não reconhecer a forma como um código chama um modelo (biblioteca fora do catálogo,
URL guardada numa variável). Quando isso acontece, o documento não pode sair como se o sistema
não usasse IA. Aqui juntamos indícios baratos e independentes do extrator:
- host de provedor (do catálogo, ou de `sinais_de_ia.hosts`) escrito em qualquer string do código;
- biblioteca de IA importada ou declarada como dependência (`sinais_de_ia.python` / `.js`);
- credencial de provedor declarada na configuração (já em `raw_facts["config"]`).
Spec 67: a lista de bibliotecas conhecidas não fecha o problema, porque cada projeto traz uma que
ninguém listou. Por isso todo pacote de terceiro importado em código de produto passa pelo
vocabulário de `vocabulario_de_aviso` (radicais de IA e de armazenamento), e classes de framework
nomeadas por papel (`langchain_community.embeddings.X`) também contam. O que já virou fato é
descontado depois, em `facts.coverage_alerts`.
Cada indício guarda uma evidência (arquivo:linha) para o documento citar. Teste, exemplo e docs
ficam fora, pela mesma regra das operações.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

from app.scanner import integration_catalog as ic
from app.scanner import provider_catalog as pc
from app.scanner.context import context_of

_CODE_EXTS = {".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}
_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build", ".next"}
_PY_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.MULTILINE)
_URL_HOST_RE = re.compile(r"https?://([a-z0-9.-]+(?::\d+)?)(/[^\s\"'`<>)\\]*)?", re.IGNORECASE)
_STDLIB = set(sys.stdlib_module_names)


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _matches(name: str, signals: list[str]) -> str | None:
    n = _norm(name)
    for s in signals:
        sn = _norm(s)
        if sn.endswith("/") and n.startswith(sn):
            return s.rstrip("/")
        if n == sn or n.startswith(sn + ".") or n.startswith(sn + "/") or n.startswith(sn + "-"):
            return s
    return None


def _host_signal(host: str) -> tuple[str, str] | None:
    """(key, display) for a host that belongs to an AI service."""
    hit = pc.by_host(host)
    if hit is not None:
        return hit.id, hit.nome
    bare = host.split(":")[0].lower()
    for known, display in pc.load_catalog().sinais_de_ia.hosts.items():
        if bare == known or bare.endswith("." + known):
            return known, display
    return None


def ai_host(host: str) -> tuple[str, str] | None:
    """Public view of `_host_signal`: the documents leave AI hosts out of the external services."""
    return _host_signal(host)


_COMMENT_STARTS = ("#", "//", "*", "/*", "<!--")


def _in_comment(text: str, pos: int) -> bool:
    """The match sits on a line that is a comment from its first character."""
    start = text.rfind("\n", 0, pos) + 1
    return text[start:pos].lstrip().startswith(_COMMENT_STARTS)


def _code_files(root: Path) -> list[Path]:
    return [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in _CODE_EXTS
        and not any(part in _SKIP_DIRS for part in p.relative_to(root).parts)
    ]


def _manifest_deps(root: Path) -> list[tuple[str, str, str]]:
    """(language, dependency, file) from requirements*, pyproject.toml and package.json."""
    out: list[tuple[str, str, str]] = []
    for p in root.rglob("*"):
        if not p.is_file() or any(part in _SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        rel = p.relative_to(root).as_posix()
        if p.name.startswith("requirements") and p.suffix == ".txt":
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                name = re.split(r"[\s<>=!~\[;]", line.strip(), maxsplit=1)[0]
                if name and not name.startswith(("#", "-")):
                    out.append(("python", name, rel))
        elif p.name == "pyproject.toml":
            for m in re.finditer(
                r"""["']([A-Za-z0-9_.\-]+)\s*(?:[<>=!~\[;]|["'])""",
                p.read_text(encoding="utf-8", errors="ignore"),
            ):
                out.append(("python", m.group(1), rel))
        elif p.name == "package.json":
            try:
                data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            except ValueError:
                continue
            for key in ("dependencies", "devDependencies", "peerDependencies"):
                for dep in data.get(key) or {}:
                    out.append(("js", dep, rel))
    return out


def _first(store: dict[str, dict[str, str]], key: str, value: dict[str, str]) -> None:
    if key not in store:
        store[key] = value


def _py_package(module: str, namespaces: list[str]) -> str:
    parts = module.split(".")
    return ".".join(parts[:2]) if parts[0] in namespaces and len(parts) > 1 else parts[0]


def _py_libraries(module: str, names: list[str], namespaces: list[str]) -> list[str]:
    """The names a reader knows the libraries by: the top-level module, or three segments in a
    namespace that many libraries share (google.cloud.bigquery, azure.storage.blob). In
    `from google.cloud import bigquery` the third segment is the imported name."""
    parts = module.split(".")
    if parts[0] not in namespaces:
        return [parts[0]]
    if len(parts) >= 3 or not names:
        return [".".join(parts[:3])]
    return [".".join([*parts, name][:3]) for name in names]


# Node's own modules, importable without the `node:` prefix: not third-party libraries.
_NODE_BUILTINS = frozenset(
    "assert async_hooks buffer child_process cluster console constants crypto dgram "
    "diagnostics_channel dns domain events fs http http2 https inspector module net os path "
    "perf_hooks process punycode querystring readline repl stream string_decoder sys timers tls "
    "trace_events tty url util v8 vm wasi worker_threads zlib".split()
)


def _js_package(spec: str) -> str | None:
    if spec.startswith((".", "/", "~", "#", "node:")) or spec.startswith("@/"):
        return None
    parts = spec.split("/")
    return "/".join(parts[:2]) if spec.startswith("@") and len(parts) > 1 else parts[0]


def _local_names(root: Path) -> set[str]:
    """Names a Python import may take from the repository itself (module files and folders)."""
    out: set[str] = set()
    for p in root.rglob("*"):
        if any(part in _SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        if p.is_dir():
            out.add(p.name)
        elif p.suffix == ".py":
            out.add(p.stem)
    return out


def _own_js_packages(root: Path) -> set[str]:
    """Names the repository gives its own packages (a workspace imports them like any other)."""
    out: set[str] = set()
    for p in root.rglob("package.json"):
        if any(part in _SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        try:
            name = json.loads(p.read_text(encoding="utf-8", errors="ignore")).get("name")
        except (ValueError, AttributeError):
            continue
        if isinstance(name, str) and name:
            out.add(name)
    return out


def _role_module(module: str, roles: pc.Roles) -> bool:
    """True when `module` is a framework module that holds models or stores, not an abstraction."""
    norm = module.replace("/", ".")
    if any(
        norm == a.replace("/", ".") or norm.startswith(a.replace("/", ".") + ".")
        for a in roles.raizes_abstratas
    ):
        return False
    root = norm.split(".")[0]
    in_framework = any(
        root == f or root.startswith(f + "_") or module.startswith(f) for f in roles.frameworks
    )
    segments = norm.split(".")
    return (
        in_framework
        and any(seg in roles.segmentos for seg in segments)
        and not any(seg in roles.segmentos_excluidos for seg in segments)
    )


def _py_imports(text: str) -> list[tuple[str, list[str], int]]:
    """(module, imported names, line) for absolute imports; empty when the file does not parse."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    out: list[tuple[str, list[str], int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend((a.name, [], node.lineno) for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append((node.module, [a.name for a in node.names], node.lineno))
    return out


def _js_imports(path: Path, text: str) -> list[tuple[str, list[str], int]]:
    """(module, imported names, line) read from the syntax tree (spec 73): `import ... from`,
    `export ... from`, `require("x")` and `import("x")`. An `import` written inside a string —
    a template of generated code — is text, not an import."""
    from tree_sitter import Parser

    from app.scanner.extractors.tree_sitter_js import _language_for, _string_value, _text, _walk

    language = _language_for(path)
    if language is None:
        return []
    tree = Parser(language).parse(text.encode())
    out: list[tuple[str, list[str], int]] = []
    for node in _walk(tree.root_node):
        line = node.start_point[0] + 1
        if node.type in ("import_statement", "export_statement"):
            module = _string_value(node.child_by_field_name("source"))
            if module:
                names = [
                    _text(spec.child_by_field_name("name"))
                    for spec in _walk(node)
                    if spec.type == "import_specifier"
                ]
                out.append((module, names, line))
        elif node.type == "call_expression":
            if _text(node.child_by_field_name("function")) not in ("require", "import"):
                continue
            args = node.child_by_field_name("arguments")
            module = _string_value(args.named_children[0]) if args and args.named_children else None
            if not module:
                continue
            names = []
            holder = node.parent
            while holder is not None and holder.type in (
                "await_expression",
                "parenthesized_expression",
            ):
                holder = holder.parent
            pattern = holder.child_by_field_name("name") if holder is not None else None
            if holder is not None and holder.type == "variable_declarator" and pattern is not None:
                names = [
                    _text(c)
                    for c in _walk(pattern)
                    if c.type in ("shorthand_property_identifier_pattern", "property_identifier")
                ]
            out.append((module, names, line))
    return out


def collect(root: Path) -> dict[str, dict]:
    """Signs of AI and of storage in the repository's product code.

    - `provedores`: {id_or_host: "Nome — evidência"} — AI service hosts written in the code;
    - `pacotes`: {package: {tipo, evidencia}} — third-party packages imported in product code whose
      name is in the AI or storage vocabulary (or in the known AI library list);
    - `manifesto`: {package: {tipo, evidencia}} — the same, for dependencies only declared;
    - `classes`: {name: {pacote, modulo, evidencia}} — framework components named by role;
    - `enderecos`: {host: "file:line"} — other hosts written in product code whose address has
      the shape of an API (spec 74), outside comment lines. A plain link is not a service;
    - `bibliotecas`: {"python": [...], "js": [...]} — every third-party library imported in
      product code (spec 74). `facts.unclassified_libraries` keeps the ones no catalog knows.
    """
    catalog = pc.load_catalog()
    signals = catalog.sinais_de_ia
    vocab = catalog.vocabulario_de_aviso
    providers: dict[str, str] = {}
    packages: dict[str, dict[str, str]] = {}
    manifest: dict[str, dict[str, str]] = {}
    classes: dict[str, dict[str, str]] = {}
    addresses: dict[str, str] = {}
    libraries: dict[str, set[str]] = {"python": set(), "js": set()}
    root = Path(root)
    local = _local_names(root)
    own_packages = _own_js_packages(root)

    def kind(package: str, full: str, known: list[str]) -> tuple[str, dict[str, str]] | None:
        """(key, {tipo, radical}) for a package, or None when it is neither AI nor storage."""
        if package.lower() in vocab.neutros:
            return None
        found = vocab.kind_of(package)
        if found and found[0] == "armazenamento":
            return package, {"tipo": found[0], "radical": found[1]}
        lib = _matches(full, known)
        if lib:
            return lib, {"tipo": "ia", "radical": lib.rstrip("/").lower()}
        return (package, {"tipo": found[0], "radical": found[1]}) if found else None

    for path in _code_files(root):
        rel = path.relative_to(root).as_posix()
        if context_of(rel):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for m in _URL_HOST_RE.finditer(text):
            hit = _host_signal(m.group(1))
            if hit is not None and hit[0] not in providers:
                line = text.count("\n", 0, m.start()) + 1
                providers[hit[0]] = f"{hit[1]} — endereço em {rel}:{line}"
            host = m.group(1).split(":")[0].lower().strip(".")
            if (
                hit is None
                and host not in addresses
                and ic.api_shaped(m.group(1), m.group(2) or "")
                and not _in_comment(text, m.start())
            ):
                addresses[host] = f"{rel}:{text.count(chr(10), 0, m.start()) + 1}"
        is_py = path.suffix == ".py"
        for module, names, line in _py_imports(text) if is_py else _js_imports(path, text):
            if is_py:
                top = module.split(".")[0]
                if top in _STDLIB or top in local:
                    continue
                package: str | None = _py_package(module, vocab.namespaces)
            else:
                package = _js_package(module)
            if package is None:
                continue
            if is_py:
                libraries["python"].update(_py_libraries(module, names, vocab.namespaces))
            elif (
                package not in _NODE_BUILTINS
                and package not in own_packages
                and ".config." not in path.name
            ):
                # a build or lint config (vite.config.ts) imports tooling, not what the product uses
                libraries["js"].add(package)
            where = f"import em {rel}:{line}"
            hit = kind(package, module, signals.python if is_py else signals.js)
            if hit is not None:
                _first(packages, hit[0], {**hit[1], "evidencia": where})
            if _role_module(module, vocab.papeis):
                for name in names:
                    if name[:1].isupper():
                        _first(
                            classes,
                            name,
                            {"pacote": package, "modulo": module, "evidencia": where},
                        )
    for lang, dep, rel in _manifest_deps(root):
        if context_of(rel):
            continue
        hit = kind(dep, dep, signals.python if lang == "python" else signals.js)
        if hit is not None:
            _first(manifest, hit[0], {**hit[1], "evidencia": f"dependência em {rel}"})
    return {
        "provedores": providers,
        "pacotes": packages,
        "manifesto": manifest,
        "classes": classes,
        "enderecos": addresses,
        "bibliotecas": {lang: sorted(names) for lang, names in libraries.items()},
    }
