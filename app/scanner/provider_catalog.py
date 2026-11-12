"""Load the LLM provider catalog (app/templates/catalogs/provedores.yml) — spec 65.

The scanner never infers a provider from a method name; it asks this catalog which provider
a module, class, JS package, host, environment variable or "provider/model" prefix belongs to.
Adding a provider is a YAML entry, never an `if` in the extractors.
"""

from __future__ import annotations

import fnmatch
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field

_CATALOG_PATH = Path(__file__).parent.parent / "templates" / "catalogs" / "provedores.yml"

UNRESOLVED = "nao_resolvido"
OPENAI_COMPATIBLE = "openai_compatible"
_SPECIAL_NAMES = {
    UNRESOLVED: "provedor não identificado",
    OPENAI_COMPATIBLE: "API compatível com OpenAI (host não identificado)",
}


class PyClass(BaseModel):
    name: str
    modules: list[str]
    kind: str = "invoke_llm"
    model_pos: int | None = None
    model_kwarg: str | None = None
    requer: dict[str, list[str]] = Field(default_factory=dict)
    # Spec 73: `name` is a class method of any class of the module (`X.from_pretrained`);
    # `exceto` lists the classes it is not a model for (tokenizers, processors).
    exceto: list[str] = Field(default_factory=list)


class PySpec(BaseModel):
    classes: list[PyClass] = Field(default_factory=list)
    funcoes: list[str] = Field(default_factory=list)


class JsGlobal(BaseModel):
    """A function the framework injects with no import (Nuxt auto-import): `hubAI()`. It only
    counts when the package that provides it is declared in a package.json (spec 73)."""

    chamada: str
    pacote: str


class JsSpec(BaseModel):
    pacotes: list[str] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    fabricas: list[str] = Field(default_factory=list)
    globais: list[JsGlobal] = Field(default_factory=list)


class AzureEvidence(BaseModel):
    kwargs: list[str] = Field(default_factory=list)
    api_type: str | None = None
    caminho_url: str | None = None


class Provider(BaseModel):
    id: str
    nome: str
    sede: str | None = None
    local: bool = False
    python: PySpec = Field(default_factory=PySpec)
    js: JsSpec = Field(default_factory=JsSpec)
    hosts: list[str] = Field(default_factory=list)
    env: list[str] = Field(default_factory=list)
    prefixos: list[str] = Field(default_factory=list)
    evidencia: AzureEvidence | None = None
    wrangler: list[str] = Field(default_factory=list)
    # Spec 73: model id prefixes that are embedding models (the call method does not tell)
    modelos_embed: list[str] = Field(default_factory=list)
    # Spec 73: where the model sits in the URL of the provider's HTTP API (one capture group)
    url_modelo: str | None = None


class Routers(BaseModel):
    python: list[str] = Field(default_factory=list)


class AiSignals(BaseModel):
    python: list[str] = Field(default_factory=list)
    js: list[str] = Field(default_factory=list)
    analisadas: list[str] = Field(default_factory=list)
    hosts: dict[str, str] = Field(default_factory=dict)


class Vocabulary(BaseModel):
    """Words that mark a package as AI or storage, whole token (`palavras`) or substring (`trechos`)."""

    palavras: list[str] = Field(default_factory=list)
    trechos: list[str] = Field(default_factory=list)

    def hit(self, name: str) -> str | None:
        low = name.lower()
        tokens = set(re.split(r"[-_./@\s]+", low))
        for w in self.palavras:
            if w in tokens:
                return w
        for w in self.trechos:
            if w in low:
                return w
        return None


class Roles(BaseModel):
    """Framework components named by role: a class imported from a module with one of these
    segments (`langchain_community.embeddings`) is a model or a store, known or not."""

    frameworks: list[str] = Field(default_factory=list)
    segmentos: list[str] = Field(default_factory=list)
    segmentos_excluidos: list[str] = Field(default_factory=list)
    raizes_abstratas: list[str] = Field(default_factory=list)


class WarningVocabulary(BaseModel):
    ia: Vocabulary = Field(default_factory=Vocabulary)
    armazenamento: Vocabulary = Field(default_factory=Vocabulary)
    papeis: Roles = Field(default_factory=Roles)
    namespaces: list[str] = Field(default_factory=list)
    neutros: list[str] = Field(default_factory=list)
    orm: list[str] = Field(default_factory=list)

    def kind_of(self, package: str) -> tuple[str, str] | None:
        """(`ia` or `armazenamento`, the word that matched) for a package name, or None.
        Storage wins: `opensearch-haystack` and `langchain-chroma` keep data, whatever drives them."""
        word = self.armazenamento.hit(package)
        if word:
            return "armazenamento", word
        word = self.ia.hit(package)
        return ("ia", word) if word else None


class Catalog(BaseModel):
    providers: list[Provider]
    sinais_de_ia: AiSignals = Field(default_factory=AiSignals)
    vocabulario_de_aviso: WarningVocabulary = Field(default_factory=WarningVocabulary)
    js_agentes_com_modelo: list[str] = Field(default_factory=list)
    router_provider_kwargs: list[str] = Field(default_factory=list)
    js_factory_methods: list[str] = Field(default_factory=list)
    model_kwargs: list[str]
    base_url_kwargs: list[str]
    call_methods: dict[str, list[str]]
    call_model_pos: dict[str, int] = Field(default_factory=dict)
    call_model_origin: dict[str, str] = Field(default_factory=dict)
    propagating_methods: list[str]
    propagating_callees: list[str]
    routers: Routers = Field(default_factory=Routers)
    module_aliases: dict[str, str] = Field(default_factory=dict)
    # Spec 73
    instancias_de_modulo: dict[str, str] = Field(default_factory=dict)
    router_modelos_sem_prefixo: dict[str, list[str]] = Field(default_factory=dict)


@lru_cache
def load_catalog() -> Catalog:
    data = yaml.safe_load(_CATALOG_PATH.read_text(encoding="utf-8"))
    catalog = Catalog.model_validate(data)
    ids = [p.id for p in catalog.providers]
    if len(ids) != len(set(ids)):
        raise ValueError(f"provedores.yml: id repetido em {ids}")
    return catalog


@lru_cache
def _by_id() -> dict[str, Provider]:
    return {p.id: p for p in load_catalog().providers}


def get(provider_id: str) -> Provider | None:
    return _by_id().get(provider_id)


def provider_ids() -> list[str]:
    return [p.id for p in load_catalog().providers]


def display_name(provider_id: str | None) -> str:
    if not provider_id:
        return _SPECIAL_NAMES[UNRESOLVED]
    if provider_id in _SPECIAL_NAMES:
        return _SPECIAL_NAMES[provider_id]
    p = get(provider_id)
    return p.nome if p else provider_id


def is_local(provider_id: str | None) -> bool:
    p = get(provider_id) if provider_id else None
    return bool(p and p.local)


def is_llm_provider(provider_id: str | None) -> bool:
    """True for a catalog provider id or one of the two explicit non-claims."""
    return provider_id in _by_id() or provider_id in _SPECIAL_NAMES


# --- Python ------------------------------------------------------------------------------------


def _module_matches(module: str, candidates: list[str]) -> bool:
    return any(module == c or module.startswith(c + ".") for c in candidates)


def python_class(module: str, name: str) -> tuple[Provider, PyClass] | None:
    """Provider + class spec for `from <module> import <name>` (or `module.name`)."""
    owner = module.rsplit(".", 1)[-1]
    for p in load_catalog().providers:
        for cls in p.python.classes:
            if (
                cls.name == name
                and _module_matches(module, cls.modules)
                and owner not in cls.exceto
            ):
                return p, cls
    return None


def instance_module(qualified: str) -> str | None:
    """Module an instance of this class stands for: `boto3.Session()` answers `.client(...)`
    like `boto3` itself (spec 73)."""
    return load_catalog().instancias_de_modulo.get(qualified)


def is_embedding_model(provider_id: str, model: str) -> bool:
    """Spec 73: the model family decides the operation when the call method is the same for
    chat and embeddings (Bedrock `invoke_model`)."""
    p = get(provider_id)
    return bool(p and any(model.startswith(prefix) for prefix in p.modelos_embed))


@lru_cache
def _bare_model_res() -> list[tuple[re.Pattern[str], str]]:
    return [
        (re.compile(rx), pid)
        for pid, patterns in load_catalog().router_modelos_sem_prefixo.items()
        for rx in patterns
    ]


def model_in_url(provider_id: str, url: str, unknown: str = "") -> str | None:
    """Model the URL itself names (`.../models/gemini-2.0-flash:generateContent`), by the
    provider's declared pattern; None when that piece is not fixed by the code."""
    p = get(provider_id)
    if p is None or not p.url_modelo:
        return None
    m = re.search(p.url_modelo, url)
    if m is None or not m.group(1) or (unknown and unknown in m.group(1)):
        return None
    return m.group(1)


def js_global_calls(declared_packages: set[str]) -> dict[str, str]:
    """{function name: provider id} for injected functions whose package the project declares."""
    return {
        g.chamada: p.id
        for p in load_catalog().providers
        for g in p.js.globais
        if g.pacote in declared_packages
    }


def by_bare_model(model: str) -> Provider | None:
    """Provider a router picks for a model name with no prefix (LiteLLM sends `gpt-4o` to
    OpenAI) — spec 73. Only names whose routing is unambiguous are in the catalog."""
    for rx, pid in _bare_model_res():
        if rx.search(model):
            return get(pid)
    return None


def unalias(qualified: str) -> str:
    """Map a wrapper's re-export to the provider module (`promptlayer.openai.X` → `openai.X`)."""
    for prefix in sorted(load_catalog().module_aliases, key=len, reverse=True):
        if qualified == prefix or qualified.startswith(prefix + "."):
            return load_catalog().module_aliases[prefix] + qualified[len(prefix) :]
    return qualified


def python_function(dotted: str) -> Provider | None:
    """Provider for a module-level function that is itself the LLM call (e.g. `ollama.chat`)."""
    for p in load_catalog().providers:
        if dotted in p.python.funcoes:
            return p
    return None


@lru_cache
def _call_method_index() -> list[tuple[str, str]]:
    pairs = [(m, kind) for kind, methods in load_catalog().call_methods.items() for m in methods]
    # Longest suffix first: "chat.completions.create" must win over "create"-like shorter ones.
    return sorted(pairs, key=lambda mk: -len(mk[0]))


def call_kind(attr_path: str) -> str | None:
    """Operation kind when `attr_path` (the attribute chain after the client) is an LLM call."""
    for method, kind in _call_method_index():
        if attr_path == method or attr_path.endswith("." + method):
            return kind
    return None


def by_wrangler_section(section: str) -> Provider | None:
    for p in load_catalog().providers:
        if section in p.wrangler:
            return p
    return None


def is_js_model_agent(package: str) -> bool:
    return any(
        package == p or package.startswith(p + "/") for p in load_catalog().js_agentes_com_modelo
    )


def call_model_pos(attr_path: str) -> int | None:
    """Position of the model argument for call methods that take it positionally."""
    for method, pos in load_catalog().call_model_pos.items():
        if attr_path == method or attr_path.endswith("." + method):
            return pos
    return None


def call_model_origin(attr_path: str) -> str | None:
    """Where the model lives when the call does not take it (spec 75: the Assistants API keeps
    it in the assistant configured at the provider)."""
    dotted = f".{attr_path}."
    for prefix, origin in load_catalog().call_model_origin.items():
        if f".{prefix}." in dotted:
            return origin
    return None


def is_propagating_method(attr_path: str) -> bool:
    return any(
        attr_path == m or attr_path.endswith("." + m) for m in load_catalog().propagating_methods
    )


@lru_cache
def _propagating_callee_res() -> list[re.Pattern[str]]:
    return [re.compile(rx) for rx in load_catalog().propagating_callees]


def is_propagating_callee(name: str) -> bool:
    return any(rx.search(name) for rx in _propagating_callee_res())


def is_router(dotted: str) -> bool:
    return dotted in load_catalog().routers.python


# --- JS ----------------------------------------------------------------------------------------


def js_package(package: str) -> Provider | None:
    for p in load_catalog().providers:
        if package in p.js.pacotes:
            return p
    return None


# --- Shared ------------------------------------------------------------------------------------


def _host_of(url_or_host: str) -> str:
    text = url_or_host.strip()
    if "://" not in text:
        text = "http://" + text
    netloc = urlparse(text).netloc.lower()
    return netloc.split("@")[-1]


def by_host(url_or_host: str) -> Provider | None:
    """Provider whose catalog host matches the URL's host (with `*.` wildcard and port)."""
    netloc = _host_of(url_or_host)
    host_only = netloc.split(":")[0]
    for p in load_catalog().providers:
        for pattern in p.hosts:
            target = netloc if ":" in pattern else host_only
            if fnmatch.fnmatch(target, pattern):
                return p
    return None


def azure_path_in(text: str) -> bool:
    az = get("azure_openai")
    path = az.evidencia.caminho_url if az and az.evidencia else None
    return bool(path and path in text)


def by_prefix(model_string: str) -> tuple[Provider, str] | None:
    """Split "provider/model" or "provider:model" when the prefix is a catalog provider."""
    for sep in ("/", ":"):
        if sep in model_string:
            prefix, rest = model_string.split(sep, 1)
            for p in load_catalog().providers:
                if prefix.lower() in p.prefixos and rest:
                    return p, rest
    return None


@lru_cache
def _env_res() -> list[tuple[re.Pattern[str], str]]:
    return [(re.compile(rf"\b{rx}\b"), p.id) for p in load_catalog().providers for rx in p.env]


def providers_in_env_text(text: str) -> list[str]:
    """Provider ids whose environment-variable patterns appear in `text` (config files)."""
    found = {pid for rx, pid in _env_res() if rx.search(text)}
    return sorted(found)


@lru_cache
def _env_full_res() -> list[tuple[re.Pattern[str], str]]:
    return [(re.compile(rx), p.id) for p in load_catalog().providers for rx in p.env]


def provider_for_env_var(name: str) -> str | None:
    """Provider id when a declared variable name fully matches a catalog env pattern."""
    for rx, pid in _env_full_res():
        if rx.fullmatch(name):
            return pid
    return None
