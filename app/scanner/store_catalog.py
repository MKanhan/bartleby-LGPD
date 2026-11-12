"""Load the datastore catalog (app/templates/catalogs/armazenamentos.yml) — spec 65."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

_CATALOG_PATH = Path(__file__).parent.parent / "templates" / "catalogs" / "armazenamentos.yml"


class PyClass(BaseModel):
    name: str
    modules: list[str]
    # Spec 73: argument values that make the call this store (boto3.client("dynamodb")), like
    # the provider catalog's `requer`. The first positional argument stands for the keyword.
    requer: dict[str, list[str]] = Field(default_factory=dict)


class PySpec(BaseModel):
    modulos: list[str] = Field(default_factory=list)
    classes: list[PyClass] = Field(default_factory=list)


class JsGlobal(BaseModel):
    """A function the framework injects with no import (`hubDatabase()`), valid only when the
    package that provides it is declared in a package.json (spec 73)."""

    chamada: str
    pacote: str


class JsSpec(BaseModel):
    pacotes: list[str] = Field(default_factory=list)
    globais: list[JsGlobal] = Field(default_factory=list)


class Store(BaseModel):
    id: str
    nome: str
    tipo: str
    local: bool = False
    python: PySpec = Field(default_factory=PySpec)
    url: list[str] = Field(default_factory=list)
    django: list[str] = Field(default_factory=list)
    config: list[str] = Field(default_factory=list)
    wrangler: list[str] = Field(default_factory=list)
    js: JsSpec = Field(default_factory=JsSpec)
    # Spec 73: `provider` of a Prisma datasource; SQL that proves the store (regex, any case)
    prisma: list[str] = Field(default_factory=list)
    sql: list[str] = Field(default_factory=list)

    @property
    def is_vector(self) -> bool:
        return self.tipo == "vetorial"


class Catalog(BaseModel):
    stores: list[Store]
    construtores: list[str] = Field(default_factory=list)


@lru_cache
def load_catalog() -> Catalog:
    data = yaml.safe_load(_CATALOG_PATH.read_text(encoding="utf-8"))
    catalog = Catalog.model_validate(data)
    ids = [s.id for s in catalog.stores]
    if len(ids) != len(set(ids)):
        raise ValueError(f"armazenamentos.yml: id repetido em {ids}")
    return catalog


def get(store_id: str) -> Store | None:
    return next((s for s in load_catalog().stores if s.id == store_id), None)


def display_name(store_id: str) -> str:
    s = get(store_id)
    return s.nome if s else store_id


def _module_matches(module: str, candidates: list[str]) -> bool:
    return any(module == c or module.startswith(c + ".") for c in candidates)


def python_module(module: str) -> Store | None:
    for s in load_catalog().stores:
        if _module_matches(module, s.python.modulos):
            return s
    return None


def python_class(module: str, name: str) -> Store | None:
    """Store for a class that proves it by name alone (no `requer`)."""
    for s in load_catalog().stores:
        for cls in s.python.classes:
            if cls.name == name and not cls.requer and _module_matches(module, cls.modules):
                return s
    return None


def python_conditional_classes(module: str, name: str) -> list[tuple[Store, PyClass]]:
    """Catalog classes that are a store only with the right argument (`requer`)."""
    return [
        (s, cls)
        for s in load_catalog().stores
        for cls in s.python.classes
        if cls.name == name and cls.requer and _module_matches(module, cls.modules)
    ]


def by_url(url: str) -> Store | None:
    """Store for a literal connection URL (`sqlite:///x.db`, `postgresql+psycopg://...`)."""
    if "://" not in url:
        return None
    scheme = url.split("://", 1)[0].lower()
    base = scheme.split("+", 1)[0]  # mysql+mysqlconnector is still MySQL
    for s in load_catalog().stores:
        if scheme in s.url or base in s.url:
            return s
    return None


def by_django_engine(engine: str) -> Store | None:
    for s in load_catalog().stores:
        if engine in s.django:
            return s
    return None


def by_config_provider(value: str) -> Store | None:
    for s in load_catalog().stores:
        if value.lower() in s.config:
            return s
    return None


def by_wrangler_section(section: str) -> Store | None:
    for s in load_catalog().stores:
        if section in s.wrangler:
            return s
    return None


def by_prisma_provider(value: str) -> Store | None:
    for s in load_catalog().stores:
        if value.lower() in s.prisma:
            return s
    return None


@lru_cache
def _sql_res() -> list[tuple[re.Pattern[str], str]]:
    return [(re.compile(rx, re.IGNORECASE), s.id) for s in load_catalog().stores for rx in s.sql]


def by_sql(text: str) -> Store | None:
    """Store a SQL statement proves (`CREATE EXTENSION vector` is pgvector)."""
    for rx, store_id in _sql_res():
        if rx.search(text):
            return get(store_id)
    return None


def js_global_calls(declared_packages: set[str]) -> dict[str, str]:
    """{function name: store id} for injected functions whose package the project declares."""
    return {
        g.chamada: s.id
        for s in load_catalog().stores
        for g in s.js.globais
        if g.pacote in declared_packages
    }


def js_package(package: str) -> Store | None:
    for s in load_catalog().stores:
        if package in s.js.pacotes:
            return s
    return None
