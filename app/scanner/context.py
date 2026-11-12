"""Which files are not the product: tests, examples, docs (spec 65).

One rule for the extractors' operations and for the coverage signals, so a library seen only in a
test never becomes a warning and a call seen only in an example never becomes a fact.
"""

from __future__ import annotations

_CONTEXT_DIRS = {
    "tests": "teste",
    "test": "teste",
    "__tests__": "teste",
    "examples": "exemplo",
    "example": "exemplo",
    "docs": "docs",
    "doc": "docs",
}


def context_of(rel: str) -> str | None:
    """`teste` / `exemplo` / `docs` for code that is not the product; None otherwise."""
    parts = rel.split("/")
    for part in parts[:-1]:
        if part.lower() in _CONTEXT_DIRS:
            return _CONTEXT_DIRS[part.lower()]
    name = parts[-1].lower()
    if name.startswith("test_") or name.endswith(
        ("_test.py", ".test.ts", ".test.tsx", ".test.js", ".spec.ts", ".spec.js")
    ):
        return "teste"
    return None
