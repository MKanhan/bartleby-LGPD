"""Shared reading of scan results for everything downstream (spec 65).

Risk triggers, the stack sentence, activity names and the documents' coverage section all read
operations through these functions, so one rule decides: what counts as product code, how a
provider is named, what is an external provider, and which facts stayed unresolved.
"""

from __future__ import annotations

import ipaddress
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from app.scanner import integration_catalog as ic
from app.scanner import provider_catalog as pc
from app.scanner import store_catalog as sc
from app.scanner.context import context_of
from app.scanner.schema import Operation, ScanOutput

LLM_KINDS = frozenset({"invoke_llm", "embed"})
VECTOR_KINDS = frozenset({"vector_store_init", "vector_store_write", "vector_store_read"})
NON_CLAIMS = frozenset({None, "", pc.UNRESOLVED, pc.OPENAI_COMPATIBLE})


def is_product(op: Operation) -> bool:
    """False for operations in tests, examples or docs: they are not what the agent does."""
    return not op.extra.get("contexto")


def product_operations(scan: ScanOutput) -> list[Operation]:
    return [op for op in scan.operations if is_product(op)]


UNSEEN_CALL = frozenset({"construtor", "modelo_criado"})


def provider_marks(op: Operation) -> tuple[str, ...]:
    """Spec 73: how sure the provider is. A provider that comes from an overridable default is
    claimed and marked as configurable. A provider known only because the code builds its client
    or its model, with no call the scanner could follow, is claimed and marked as such. Empty when
    the code fixes the provider at a call site."""
    if op.provider in NON_CLAIMS:
        return ()
    marks = []
    if op.extra.get("provider_padrao"):
        marks.append("padrão do código, configurável")
    if op.extra.get("via") in UNSEEN_CALL:
        marks.append("chamada não identificada")
    return tuple(marks)


def provider_mark(op: Operation) -> str:
    return "; ".join(provider_marks(op))


def provider_label(op: Operation) -> str:
    """How the document names who processes the data for this operation."""
    if op.kind in LLM_KINDS:
        name = pc.display_name(op.provider)
        if op.extra.get("intermediario") == "vercel_ai_gateway":
            name = f"{name} via Vercel AI Gateway"
        mark = provider_mark(op)
        return f"{name}, {mark}" if mark else name
    store_id = op.extra.get("store") or op.provider
    if store_id and sc.get(str(store_id)):
        return sc.display_name(str(store_id))
    if op.provider and ic.get(str(op.provider)):
        return ic.display_name(str(op.provider))
    return op.provider or "n/d"


def is_external(op: Operation) -> bool:
    """Data leaves the controller's environment towards a third party.

    LLM operations count unless the provider runs locally; an unresolved provider counts
    (conservative: the risk is listed until someone shows otherwise). Datastores never count here:
    their risk is retention, not transfer. Vector stores count unless local; memory and telemetry
    count unless their catalog entry is local (spec 69: a Redis in the controller's own
    environment is not a transfer).
    """
    if not is_product(op) or op.kind == "datastore":
        return False
    if op.kind in LLM_KINDS:
        return not pc.is_local(op.provider)
    if op.kind in VECTOR_KINDS:
        store = sc.get(str(op.provider)) if op.provider else None
        return bool(op.provider) and not (store is not None and store.local)
    if op.kind in ("memory_persist", "telemetry"):
        integration = ic.get(str(op.provider)) if op.provider else None
        return bool(op.provider) and not (integration is not None and integration.local)
    return bool(op.provider)


@dataclass
class ExternalServices:
    """Spec 71: hosts outside the AI catalog that receive data, and calls whose host the code
    does not fix."""

    hosts: list[tuple[str, list[str]]] = field(default_factory=list)  # (host, ["file:line"])
    unresolved: int = 0
    # spec 74: hosts written in product code that no followed call reaches (host, "file:line")
    unseen: list[tuple[str, str]] = field(default_factory=list)


def _host(url: str) -> str:
    host = url.split("://", 1)[1].split("/", 1)[0].split("?", 1)[0]
    return host.rsplit("@", 1)[-1].split(":", 1)[0].lower()


def _local(host: str) -> bool:
    """localhost, a private or loopback address, or a bare name (a compose service)."""
    if "." not in host or host.endswith((".local", ".internal", ".localhost")):
        return True
    try:
        return ipaddress.ip_address(host).is_private or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def external_services(scan: ScanOutput) -> ExternalServices:
    from app.scanner.ai_signals import ai_host

    by_host: dict[str, list[str]] = {}
    unresolved = 0
    for call in scan.external_calls:
        if context_of(call.file):
            continue
        if call.url is None:
            unresolved += 1
            continue
        host = _host(call.url)
        if not host or _local(host):
            continue
        # an AI host is a provider, not a service, unless a catalog library reaches it as one
        if ai_host(host) is not None and ic.service_host(host) is None:
            continue
        where = f"{call.file}:{call.line}"
        if where not in by_host.setdefault(host, []):
            by_host[host].append(where)
    written: dict[str, str] = (scan.raw_facts.get("ai_signals") or {}).get("enderecos", {})
    unseen = [
        (host, where)
        for host, where in sorted(written.items())
        if host not in by_host and not _local(host) and not ic.ignored_host(host)
    ]
    return ExternalServices(sorted(by_host.items()), unresolved, unseen)


def operation_share(ops: list[Operation]) -> dict[str, float]:
    """Share of LLM operations per identified provider (sums to 1 when there is any)."""
    providers = [
        op.provider
        for op in ops
        if op.kind in LLM_KINDS and op.provider not in NON_CLAIMS and is_product(op)
    ]
    if not providers:
        return {}
    total = len(providers)
    return {str(p): c / total for p, c in Counter(providers).items()}


def providers_identified(ops: list[Operation]) -> list[str]:
    """'Groq (sede: EUA)'-style entries for every LLM provider evidenced in product code."""
    marks: dict[str, set[tuple[str, ...]]] = {}
    for op in ops:
        if op.kind not in LLM_KINDS or op.provider in NON_CLAIMS or not is_product(op):
            continue
        if pc.get(str(op.provider)) is not None:
            marks.setdefault(str(op.provider), set()).add(provider_marks(op))
    out: list[str] = []
    for provider_id in sorted(marks):
        p = pc.get(provider_id)
        assert p is not None
        details = ["executado localmente" if p.local else f"sede: {p.sede}" if p.sede else ""]
        # The mark shows only when no call site fixes this provider outright.
        if () not in marks[provider_id]:
            details += sorted({m for group in marks[provider_id] for m in group})
        text = "; ".join(d for d in details if d)
        out.append(f"{p.nome} ({text})" if text else p.nome)
    return out


def datastores(ops: list[Operation]) -> list[str]:
    names: dict[str, str] = {}
    for op in ops:
        if not is_product(op):
            continue
        store_id = (
            op.extra.get("store")
            if op.kind == "datastore"
            else op.provider
            if op.kind in VECTOR_KINDS
            else None
        )
        store = sc.get(str(store_id)) if store_id else None
        if store is not None:
            names[store.id] = store.nome
    return [names[k] for k in sorted(names)]


def unresolved_facts(ops: list[Operation]) -> list[str]:
    """One line per LLM operation whose provider or model the code does not fix."""
    out: list[str] = []
    for op in ops:
        if op.kind not in LLM_KINDS or not is_product(op):
            continue
        where = f"{op.file}:{op.line}"
        if op.provider in NON_CLAIMS:
            out.append(
                f"{where} — provedor: {pc.display_name(op.provider)} ({op.extra.get('provider_origin', 'origem não registrada')})"
            )
        if not op.model:
            origin = op.extra.get("model_origin", "não resolvido")
            default = op.extra.get("model_padrao")
            tail = f"; padrão no código: {default}" if default else ""
            out.append(f"{where} — modelo não fixado no código: {origin}{tail}")
    return out


def provider_storage(ops: list[Operation]) -> list[str]:
    """Spec 76: what the code keeps at a provider, by a call that stores it there (an Assistants
    thread holds the conversation). One line per provider and kind of data, with the places."""
    places: dict[tuple[str, str], list[str]] = {}
    for op in ops:
        stored = op.extra.get("guarda_no_provedor")
        if stored and op.provider:
            places.setdefault((pc.display_name(str(op.provider)), str(stored)), []).append(
                f"{op.file}:{op.line}"
            )
    return [
        f"{name} guarda {what} — {', '.join(where[:3])}{' e outros' if len(where) > 3 else ''}"
        for (name, what), where in sorted(places.items())
    ]


def candidate_providers(scan: ScanOutput) -> set[str]:
    """Spec 73: providers the code lists in a table of addresses and picks at run time. They are
    options, not facts: nothing here is claimed."""
    tables = scan.raw_facts.get("tabelas_de_provedores") or []
    return {str(p) for t in tables for p in t["provedores"]}


def candidate_lines(scan: ScanOutput) -> list[str]:
    """One line per table, for the section of facts the code does not fix."""
    out: list[str] = []
    for t in scan.raw_facts.get("tabelas_de_provedores") or []:
        names = [pc.display_name(p) for p in t["provedores"]]
        listed = f"{', '.join(names[:-1])} ou {names[-1]}"
        table = f"a tabela {t['nome']}" if t["nome"] else "uma tabela de endereços"
        out.append(
            f"{t['file']}:{t['line']} — provedor escolhido em tempo de execução: {table} prevê "
            f"{listed}. Confirme qual está em uso."
        )
    return out


# `neutros` (spec 75): a library the catalog names as neither AI nor storage is classified too
_NAME_KEYS = frozenset(
    {"modulos", "modules", "pacotes", "orm", "analisadas", "python", "js", "neutros"}
)


def _names_in(node: object, under: str = "") -> set[str]:
    """Every module or package name a catalog lists, whatever the entry it sits in."""
    if isinstance(node, dict):
        return {n for key, value in node.items() for n in _names_in(value, str(key))}
    if isinstance(node, list):
        if under in _NAME_KEYS:
            return {str(v).rstrip("/").lower() for v in node if isinstance(v, str)}
        return {n for value in node for n in _names_in(value, under)}
    return set()


@lru_cache
def _catalog_names() -> frozenset[str]:
    catalogs = (pc.load_catalog(), sc.load_catalog(), ic.load_catalog())
    return frozenset(n for c in catalogs for n in _names_in(c.model_dump()))


def _catalog_knows(library: str) -> bool:
    """The library, one of its modules or the package it belongs to is named in a catalog."""
    lib = library.lower()
    return any(
        lib == n or n.startswith((lib + ".", lib + "/")) or lib.startswith((n + ".", n + "/"))
        for n in _catalog_names()
    )


def unclassified_libraries(scan: ScanOutput) -> dict[str, list[str]]:
    """Spec 74: third-party libraries the product code imports that no catalog knows and no
    warning names. Not a finding: the list lets a reviewer spot a service the scanner misses."""
    signals = scan.raw_facts.get("ai_signals") or {}
    warned = {_squash(name) for name in signals.get("pacotes", {})}
    out: dict[str, list[str]] = {}
    for lang, names in (signals.get("bibliotecas") or {}).items():
        kept = [n for n in names if not _catalog_knows(n) and _squash(n) not in warned]
        if kept:
            out[lang] = sorted(kept, key=str.lower)
    return out


def ignored_count(scan: ScanOutput) -> int:
    return sum(1 for op in scan.operations if not is_product(op))


def _key(name: str) -> str:
    """Package and module names compare equal: google-generativeai == google.generativeai."""
    return name.lower().rstrip("/").replace("-", "_").replace(".", "_").replace("/", "_")


def _squash(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def _found_names(ops: list[Operation]) -> tuple[str, set[str]]:
    """What the found facts account for: their origin text, and the catalog modules, packages and
    class names of every provider and store that became a fact."""
    origins = " ".join(
        str(op.extra.get(k, "")) for op in ops for k in ("provider_origin", "store_origin")
    )
    ids = {str(op.provider) for op in ops if op.provider not in NON_CLAIMS}
    # spec 73: an OpenAI-compatible call is made with the OpenAI SDK (only its classes and
    # factories yield it): the library has calls, whoever the host turns out to be
    if any(op.provider == pc.OPENAI_COMPATIBLE for op in ops):
        ids.add("openai")
    ids |= {str(op.extra["store"]) for op in ops if op.extra.get("store")}
    names: set[str] = set()
    for p in pc.load_catalog().providers:
        if p.id in ids:
            names |= {m for cls in p.python.classes for m in cls.modules}
            names |= {cls.name for cls in p.python.classes}
            names |= set(p.python.funcoes) | set(p.js.pacotes)
    for i in ic.load_catalog().integracoes:
        if i.id in ids:
            names |= set(i.python.modulos) | set(i.js.pacotes)
            names |= {m for cls in i.python.classes for m in cls.modules}
            names |= {cls.name for cls in i.python.classes}
    for s in sc.load_catalog().stores:
        if s.id in ids:
            names |= set(s.python.modulos) | set(s.js.pacotes)
            names |= {m for cls in s.python.classes for m in cls.modules}
            names |= {cls.name for cls in s.python.classes}
    return _key(origins), {_key(n) for n in names}


def _explained(name: str, origins: str, names: set[str]) -> bool:
    key = _key(name)
    if key in origins:
        return True
    return any(n == key or n.startswith(key + "_") or key.startswith(n + "_") for n in names)


def _framework_analyzed(lib: str) -> bool:
    key = _key(lib)
    return any(
        key == _key(a) or key.startswith(_key(a) + "_")
        for a in pc.load_catalog().sinais_de_ia.analisadas
    )


HEADLINE = (
    "Há indícios de uso de IA no projeto, mas nenhuma chamada a modelo de linguagem foi "
    "identificada. Este documento pode estar incompleto quanto ao uso de IA: confirme "
    "com a equipe técnica antes da assinatura."
)


UNREAD_PREFIX = "O scanner não conseguiu ler"
UNREAD_SHOWN = 10
# spec 76: product code in a language the scanner does not read. Names no fact and presumes no AI.
UNREAD_LANGUAGE_PREFIX = "O repositório tem código em linguagem que o scanner não lê"


@dataclass
class CoverageState:
    """What the scanner saw but could not turn into facts, read once for every document section
    and for the risk engine (spec 68): the alerts, whether there are signs of AI with no LLM call,
    and which external catalog providers show up only as signs (credential or host)."""

    alerts: list[str] = field(default_factory=list)
    ai_without_call: bool = False
    presumed_external: dict[str, str] = field(default_factory=dict)  # provider id -> evidence

    @property
    def has_gaps(self) -> bool:
        return bool(self.alerts)


def coverage_alerts(scan: ScanOutput) -> list[str]:
    return coverage_state(scan).alerts


def coverage_state(scan: ScanOutput) -> CoverageState:
    """What the document must warn about: signs of AI or storage the scanner did not turn into facts.

    The guard against silent omission (spec 65, widened in 67): a provider host or credential with
    no identified call; a package whose name says AI or storage, imported in product code, that no
    fact accounts for; a framework component named by role (an embeddings class, a document store)
    that no fact accounts for; and — loudest — signs of AI with no LLM call identified at all.
    Dependencies only declared, never imported, count only in that last case.
    """
    ops = product_operations(scan)
    identified = {
        str(op.provider) for op in ops if op.kind in LLM_KINDS and op.provider not in NON_CLAIMS
    }
    has_llm = any(op.kind in LLM_KINDS for op in ops)
    signals = scan.raw_facts.get("ai_signals") or {}
    host_signals: dict[str, str] = signals.get("provedores", {})
    packages: dict[str, dict[str, str]] = dict(signals.get("pacotes", {}))
    classes: dict[str, dict[str, str]] = signals.get("classes", {})
    if not has_llm:
        # pip and npm names may differ from the import only in case and punctuation
        # (SpeechRecognition / speech_recognition): one package, one warning
        # and a declared dependency whose word already shows in an imported package or component
        # (opensearch-haystack / OpenSearchDocumentStore) adds nothing
        seen = {_squash(name) for name in packages}
        imported = _squash(" ".join([*packages, *classes]))
        for name, info in signals.get("manifesto", {}).items():
            if info.get("radical") and _squash(info["radical"]) in imported:
                continue
            if _squash(name) not in seen:
                seen.add(_squash(name))
                packages[name] = info
    declared = (scan.raw_facts.get("config") or {}).get("providers_declared", [])
    origins, names = _found_names(ops)
    if any(op.kind == "datastore" for op in ops):
        names |= {_key(o) for o in pc.load_catalog().vocabulario_de_aviso.orm}

    alerts: list[str] = []
    presumed: dict[str, str] = {}
    ai_signs = False
    candidates = candidate_providers(scan)
    for key in sorted(host_signals):
        if key not in identified:
            ai_signs = True
            if pc.get(key) is not None and not pc.is_local(key):
                presumed[key] = host_signals[key]
            if key in candidates:
                continue  # spec 73: already listed, with the other options of its table
            alerts.append(f"Indício de {host_signals[key]}, sem chamada identificada no código.")
    # spec 75: the alert names the variable that gave the provider away
    declared_vars: dict[str, str] = (scan.raw_facts.get("config") or {}).get("provider_vars", {})
    variables = {
        pid: f"{declared_vars[pid]}, declarada na configuração"
        if pid in declared_vars
        else "declarada na configuração"
        for pid in declared
    }
    for pid in sorted(variables):
        if pid not in identified and pid not in host_signals:
            ai_signs = True
            if pc.get(pid) is not None and not pc.is_local(pid):
                presumed[pid] = f"{pc.display_name(pid)} — variável de configuração"
            alerts.append(
                f"Variável de configuração de {pc.display_name(pid)} ({variables[pid]}), sem "
                "chamada identificada no código."
            )

    unexplained_classes: dict[str, list[str]] = {}
    for cls in sorted(classes):
        info = classes[cls]
        if not _explained(cls, origins, names):
            unexplained_classes.setdefault(info["pacote"], []).append(cls)
    for name in sorted(packages):
        info = packages[name]
        own = unexplained_classes.pop(name, [])
        if not own and (_framework_analyzed(name) or _explained(name, origins, names)):
            continue
        if info["tipo"] == "ia":
            ai_signs = True
            what = "parece ligada a IA, mas o scanner não identificou chamadas feitas por ela"
        else:
            what = "parece guardar dados, mas o scanner não identificou o armazenamento"
        tail = f" Componentes usados: {', '.join(own)}." if own else ""
        alerts.append(
            f"A biblioteca {name} aparece no projeto ({info['evidencia']}) e {what}.{tail}"
        )
    for package, found in sorted(unexplained_classes.items()):
        ai_signs = True
        where = classes[found[0]]["evidencia"]
        alerts.append(
            f"O projeto usa {', '.join(found)} ({where}), de {package}, e o scanner não analisa o "
            "que esses componentes fazem com os dados."
        )
    unread: list[str] = scan.raw_facts.get("arquivos_nao_lidos") or []
    if unread:
        shown = ", ".join(unread[:UNREAD_SHOWN])
        more = f" e mais {len(unread) - UNREAD_SHOWN}" if len(unread) > UNREAD_SHOWN else ""
        alerts.append(
            f"{UNREAD_PREFIX} {len(unread)} arquivo(s) Python por erro de sintaxe; o que eles "
            f"fazem não entra nesta análise: {shown}{more}."
        )
    languages: dict[str, int] = scan.raw_facts.get("linguagens_nao_lidas") or {}
    if languages:
        counts = [
            f"{n} {'arquivo' if n == 1 else 'arquivos'} em {name}" for name, n in languages.items()
        ]
        listed = ", ".join(counts[:-1]) + " e " + counts[-1] if len(counts) > 1 else counts[0]
        alerts.append(
            f"{UNREAD_LANGUAGE_PREFIX}: {listed}. O que esse código faz não está neste documento."
        )
    ai_without_call = not has_llm and ai_signs
    if ai_without_call:
        alerts.insert(0, HEADLINE)
    return CoverageState(alerts, ai_without_call, presumed)
