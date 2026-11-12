"""Product code in languages the scanner does not read (spec 76).

The scanner reads Python and JavaScript/TypeScript. A repository may keep part of the system in
another language; counting those files lets the document say that part was not analysed, instead
of describing the rest as if it were the whole.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

from app.scanner.context import context_of

_CATALOG_PATH = Path(__file__).parent.parent / "templates" / "catalogs" / "linguagens_nao_lidas.yml"


class Catalog(BaseModel):
    linguagens: dict[str, list[str]]
    pastas_ignoradas: list[str]


@lru_cache
def load_catalog() -> Catalog:
    return Catalog.model_validate(yaml.safe_load(_CATALOG_PATH.read_text(encoding="utf-8")))


@lru_cache
def _by_extension() -> dict[str, str]:
    return {ext: name for name, exts in load_catalog().linguagens.items() for ext in exts}


def count(root: Path) -> dict[str, int]:
    """{language: files} for product code the scanner does not read, most files first."""
    root = Path(root)
    skipped = set(load_catalog().pastas_ignoradas)
    by_ext = _by_extension()
    found: Counter[str] = Counter()
    for path in root.rglob("*"):
        language = by_ext.get(path.suffix.lower())
        if language is None or not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in skipped for part in rel.parts) or context_of(rel.as_posix()):
            continue
        found[language] += 1
    return dict(sorted(found.items(), key=lambda kv: (-kv[1], kv[0])))
