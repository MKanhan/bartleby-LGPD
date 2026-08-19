"""Scan artifact — the result of a scan, portable, without the source tree.

Why this exists is commercial, not technical. The segment the free CI gate recruits — software
houses shipping AI — protects its source code, so "suba o repositório na nuvem" (SaaS) is a
non-starter and the on-prem tier is a bigger commitment than a first purchase deserves. The scan
is already the only thing the paid pipeline consumes downstream (`analyzer.pipeline.run_analysis`
scans a directory and never touches the filesystem again), so the source tree is not needed to
produce the RIPD/ROPA/Mapa. This module makes that fact usable: the client runs the free,
deterministic, offline gate on their own machine, exports one JSON, and only that JSON travels.

Two properties make the artifact something a security-minded buyer can accept:

- **Auditable** — it is plain, indented JSON. Anyone can open it and see exactly what would leave
  the machine before deciding to send it.
- **Redactable** — `redact()` strips every field that carries verbatim source text. Redaction is
  *score-invariant*: `risk_engine.score()` reads structure (operation kinds, PII kinds, stores,
  external calls), never the snippets, so a redacted artifact yields the same risk register. What
  it costs is the semantic labelling of operations (`analyzer.operation_classifier` reads
  `snippet` / `prev_comment` for a human label), which degrades to the generic SDK-based label.

Open-core: this module ships in the public gate package (`app/scanner` is copied whole by
`scripts/build_gate.py`), so it must stay free of any paid-world import — no `app.core`,
no `app.llm`, no `app.db`. Hence `datetime.now(UTC)` here instead of `app.core.clock.utcnow`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from app.scanner.schema import ScanOutput

# Bump the major only on a breaking change; `loads()` refuses anything else so a stale producer
# fails loudly instead of silently feeding a half-understood scan into a legal document.
FORMAT = "bartleby.scan/1"

# Exactly the fields that carry verbatim source text. Redaction removes these and nothing else —
# structure, file paths, line numbers, symbol names and config facts survive (they are what the
# documents are made of). A team that also wants those gone can edit the JSON: it is theirs.
_OPERATION_TEXT_FIELDS = ("snippet", "prev_comment")
_PII_TEXT_FIELDS = ("snippet",)


class ScanArtifactError(ValueError):
    """Raised when a file is not a readable Bartleby scan artifact."""


class ScanArtifact(BaseModel):
    """The scan plus the provenance a reader needs to trust it."""

    format: str = FORMAT
    generated_at: str = ""
    producer: str = "bartleby check"
    source_label: str = ""  # what was scanned, as the client named it (path / repo URL)
    redacted: bool = False
    scan: ScanOutput = Field(default_factory=ScanOutput)


def redact(scan: ScanOutput) -> ScanOutput:
    """Return a copy of `scan` with every verbatim source line removed.

    Score-invariant by construction: only text fields are cleared, and `risk_engine.score()`
    reads none of them. See the module docstring for what redaction costs.
    """
    out = scan.model_copy(deep=True)
    for op in out.operations:
        op.snippet = ""
        op.prev_comment = None
    out.pii_findings = [
        {k: v for k, v in finding.items() if k not in _PII_TEXT_FIELDS}
        for finding in out.pii_findings
    ]
    return out


def build(
    scan: ScanOutput,
    *,
    source_label: str = "",
    redacted: bool = False,
    producer: str = "bartleby check",
) -> ScanArtifact:
    """Wrap a scan into an artifact. Pass `redacted=True` only for an already-redacted scan."""
    return ScanArtifact(
        format=FORMAT,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        producer=producer,
        source_label=source_label,
        redacted=redacted,
        scan=scan,
    )


def dumps(artifact: ScanArtifact) -> str:
    """Serialize as indented UTF-8 JSON — readable by the person deciding whether to send it."""
    return json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, indent=2)


def loads(text: str) -> ScanArtifact:
    """Parse an artifact, raising `ScanArtifactError` with a human reason on anything else."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ScanArtifactError(f"não é um JSON válido: {e}") from e
    if not isinstance(data, dict):
        raise ScanArtifactError("esperado um objeto JSON no topo do arquivo")
    fmt = data.get("format")
    if fmt != FORMAT:
        raise ScanArtifactError(f"formato desconhecido: {fmt!r} (esperado {FORMAT!r})")
    try:
        return ScanArtifact.model_validate(data)
    except ValidationError as e:
        raise ScanArtifactError(f"artefato inválido: {e}") from e


def write_file(path: str | Path, artifact: ScanArtifact) -> Path:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dumps(artifact), encoding="utf-8")
    return p


def read_file(path: str | Path) -> ScanArtifact:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise ScanArtifactError(f"não consegui ler {path}: {e}") from e
    return loads(text)
