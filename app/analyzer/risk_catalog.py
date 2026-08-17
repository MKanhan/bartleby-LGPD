"""Load the risk catalog YAML and expose Pydantic models."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

_CATALOG_PATH = Path(__file__).parent.parent / "templates" / "catalogs" / "riscos.yml"


class Mitigation(BaseModel):
    text: str
    probability_reduction: int = 1
    impact_reduction: int = 0


class Triggers(BaseModel):
    operations: list[str] = Field(default_factory=list)
    has_pii: bool = False
    has_external_provider: bool = False
    has_telemetry: bool = False
    has_memory_store: bool = False
    has_vector_store: bool = False
    always: bool = False


class CatalogEntry(BaseModel):
    id: str
    categoria: str
    descricao: str
    artigos: list[str]
    probabilidade_baseline: int
    impacto_baseline: int
    triggers: Triggers
    mitigacoes: list[Mitigation]


@lru_cache
def load_catalog() -> list[CatalogEntry]:
    data = yaml.safe_load(_CATALOG_PATH.read_text(encoding="utf-8"))
    return [CatalogEntry.model_validate(entry) for entry in data["risks"]]
