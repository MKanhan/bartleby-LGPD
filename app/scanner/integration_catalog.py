"""Load the integrations catalog (app/templates/catalogs/integracoes.yml) — spec 69.

Conversation memory and telemetry, for Python and JS, as data: the extractors read it, the
documents name the integration by it, and `facts.is_external` asks it whether one runs locally.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

_CATALOG_PATH = Path(__file__).parent.parent / "templates" / "catalogs" / "integracoes.yml"

KIND_OF = {"memoria": "memory_persist", "telemetria": "telemetry"}
# Spec 73: a library that is itself an external service (googletrans sends text to Google
# Translate). It is not an operation: it joins the external services list by its host.
SERVICE = "servico"


class PyClass(BaseModel):
    name: str
    modules: list[str]


class PySpec(BaseModel):
    modulos: list[str] = Field(default_factory=list)
    classes: list[PyClass] = Field(default_factory=list)


class JsName(BaseModel):
    name: str
    pacotes: list[str]


class JsSpec(BaseModel):
    pacotes: list[str] = Field(default_factory=list)
    # a name imported from a package that has many other things (MemorySaver from the LangGraph root)
    nomes: list[JsName] = Field(default_factory=list)


class Integration(BaseModel):
    id: str
    nome: str
    tipo: str
    local: bool = False
    python: PySpec = Field(default_factory=PySpec)
    js: JsSpec = Field(default_factory=JsSpec)
    env: list[str] = Field(default_factory=list)
    host: str = ""  # tipo servico: where the data goes

    @property
    def kind(self) -> str:
        return KIND_OF[self.tipo]

    @property
    def is_service(self) -> bool:
        return self.tipo == SERVICE


class WrittenAddresses(BaseModel):
    """Spec 74: when a host that is only written in the code counts as a service."""

    host: list[str] = Field(default_factory=list)  # regex over the host
    caminho: list[str] = Field(default_factory=list)  # regex over the path
    ignorados: list[str] = Field(default_factory=list)  # hosts (and their subdomains) left out


class Catalog(BaseModel):
    integracoes: list[Integration]
    enderecos_no_codigo: WrittenAddresses = Field(default_factory=lambda: WrittenAddresses())


@lru_cache
def load_catalog() -> Catalog:
    data = yaml.safe_load(_CATALOG_PATH.read_text(encoding="utf-8"))
    catalog = Catalog.model_validate(data)
    ids = [i.id for i in catalog.integracoes]
    if len(ids) != len(set(ids)):
        raise ValueError(f"integracoes.yml: id repetido em {ids}")
    bad = [i.id for i in catalog.integracoes if i.tipo not in KIND_OF and not i.is_service]
    if bad:
        raise ValueError(f"integracoes.yml: tipo inválido em {bad}")
    no_host = [i.id for i in catalog.integracoes if i.is_service != bool(i.host)]
    if no_host:
        raise ValueError(f"integracoes.yml: host é obrigatório e exclusivo de serviço em {no_host}")
    return catalog


def _integrations() -> list[Integration]:
    """Memory and telemetry: what becomes an operation."""
    return [i for i in load_catalog().integracoes if not i.is_service]


def _services() -> list[Integration]:
    return [i for i in load_catalog().integracoes if i.is_service]


def ids() -> list[str]:
    """Memory and telemetry ids (services are not operations)."""
    return [i.id for i in _integrations()]


def get(integration_id: str) -> Integration | None:
    for i in load_catalog().integracoes:
        if i.id == integration_id:
            return i
    return None


def display_name(integration_id: str) -> str:
    i = get(integration_id)
    return i.nome if i else integration_id


def _module_matches(module: str, candidates: list[str]) -> bool:
    return any(module == c or module.startswith(c + ".") for c in candidates)


def python_symbol(module: str, name: str) -> Integration | None:
    """The integration a name imported from `module` belongs to: a listed class first, then any
    name from a listed module."""
    for i in _integrations():
        for cls in i.python.classes:
            if cls.name == name and _module_matches(module, cls.modules):
                return i
    for i in _integrations():
        if _module_matches(module, i.python.modulos):
            return i
    return None


def _package_matches(package: str, candidates: list[str]) -> bool:
    return any(
        package == p or (p.endswith("/") and package.startswith(p)) or package.startswith(p + "/")
        for p in candidates
    )


def js_package(package: str) -> Integration | None:
    for i in _integrations():
        if _package_matches(package, i.js.pacotes):
            return i
    return None


def js_name(package: str, name: str) -> Integration | None:
    """The integration a name imported from `package` belongs to (`js.nomes`)."""
    for i in _integrations():
        for n in i.js.nomes:
            if n.name == name and package in n.pacotes:
                return i
    return None


def by_env(var: str) -> Integration | None:
    for i in _integrations():
        if var in i.env:
            return i
    return None


def python_service(module: str, names: list[str]) -> Integration | None:
    """The external service an import stands for: a listed module, or a listed class among the
    imported names."""
    for i in _services():
        if _module_matches(module, i.python.modulos):
            return i
        for cls in i.python.classes:
            if cls.name in names and _module_matches(module, cls.modules):
                return i
    return None


def js_service(package: str) -> Integration | None:
    for i in _services():
        if _package_matches(package, i.js.pacotes):
            return i
    return None


def service_by_env(var: str) -> Integration | None:
    return next((i for i in _services() if var in i.env), None)


def service_host(host: str) -> Integration | None:
    return next((i for i in _services() if i.host == host), None)


def ignored_host(host: str) -> bool:
    """Spec 74: a host written in the code that is a link, not a service that receives data."""
    bare = host.split(":")[0].lower()
    ignored = load_catalog().enderecos_no_codigo.ignorados
    return any(bare == known or bare.endswith("." + known) for known in ignored)


@lru_cache
def _address_res() -> tuple[list[re.Pattern[str]], list[re.Pattern[str]]]:
    rules = load_catalog().enderecos_no_codigo
    return (
        [re.compile(r, re.IGNORECASE) for r in rules.host],
        [re.compile(r, re.IGNORECASE) for r in rules.caminho],
    )


def api_shaped(host: str, path: str) -> bool:
    """Spec 74: the address looks like an endpoint a program calls, not a page a person opens."""
    hosts, paths = _address_res()
    return any(r.search(host) for r in hosts) or any(r.search(path) for r in paths)
