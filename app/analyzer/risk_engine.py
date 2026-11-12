"""Match and score risks from a ScanOutput."""

from __future__ import annotations

import math
import re

from pydantic import BaseModel, Field

from app.analyzer.risk_catalog import CatalogEntry, Mitigation, load_catalog
from app.scanner.context import context_of
from app.scanner.facts import CoverageState, coverage_state, is_external, product_operations
from app.scanner.schema import ScanOutput


def _level(prob: int, impact: int) -> str:
    score = prob * impact
    if score >= 20:
        return "Crítico"
    if score >= 10:
        return "Alto"
    if score >= 5:
        return "Médio"
    return "Baixo"


def _clamp(v: int) -> int:
    return max(1, min(5, v))


# Residual scoring guards (spec 56). Mitigations cannot drop a risk arbitrarily, and an inherently
# high risk cannot be presented as residual-Baixo. Conservative, defensible defaults (tunable later
# with legal review). Without these the residual collapsed to "Baixo" for nearly every risk.
_MAX_PROB_REDUCTION = 2
_MAX_IMPACT_REDUCTION = 1
_TRATAMENTO = {"Crítico": "evitar", "Alto": "mitigar", "Médio": "mitigar", "Baixo": "aceitar"}


def _floor_residual_prob(prob_res: int, imp_res: int, nivel_inerente: str) -> int:
    """An inherently Alto/Crítico risk keeps a residual floor of at least "Médio": bump the residual
    probability just enough that prob*impact >= 5, so the document never presents a high-inherent risk
    as residual-Baixo on unverified controls (and the matrix cell, colored by p*i, stays consistent)."""
    if nivel_inerente in ("Alto", "Crítico"):
        return _clamp(max(prob_res, math.ceil(5 / imp_res)))
    return prob_res


class ScoredRisk(BaseModel):
    id: str
    categoria: str
    descricao: str
    artigos: list[str]
    probabilidade_inerente: int
    impacto_inerente: int
    nivel_inerente: str
    probabilidade_residual: int
    impacto_residual: int
    nivel_residual: str
    mitigacoes: list[str]
    tratamento: str = "mitigar"
    responsavel: str = "[a completar pelo controlador]"
    prazo: str = "[a completar pelo controlador]"
    # Spec 61: (file, line, kind) of the scan operations / PII findings that fired this risk's
    # triggers. Optional with a default → ScanResult JSON persisted pre-S61 loads clean, like the
    # S23/S43 back-compat fields. The linter/SARIF surface reads it; the residual model ignores it.
    evidencia: list[dict] = Field(default_factory=list)
    # Spec 68: fired only by signs the scanner could not turn into facts (a credential, a host, an
    # AI library with no call found). Kept on purpose — a governance document errs on the side of
    # the risk — and marked so the reviewer knows the trigger is presumed. Default False keeps
    # persisted registers loading clean.
    presumido: bool = False


class RiskRegister(BaseModel):
    risks: list[ScoredRisk] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)


def _scan_flags(scan: ScanOutput) -> dict[str, bool]:
    # Spec 65: tests/examples/docs are not the agent; external = facts.is_external (local
    # providers and datastores do not count, an unresolved LLM provider does).
    ops = product_operations(scan)
    operation_kinds = {op.kind for op in ops}
    if _product_tools(scan):
        # a tool the model can call is an action with side effects (AI-012), whether or not a
        # call to it shows up as an operation
        operation_kinds.add("tool_call")
    return {
        "operation_kinds": operation_kinds,
        "has_pii": any(f.get("confidence_tier", "high") == "high" for f in scan.pii_findings),
        "has_external_provider": any(is_external(op) for op in ops),
        "has_telemetry": any(op.kind == "telemetry" for op in ops),
        "has_memory_store": any(op.kind == "memory_persist" for op in ops),
        "has_vector_store": any(
            op.kind in {"vector_store_init", "vector_store_write", "vector_store_read"}
            for op in ops
        ),
    }


def _presumed_flags(flags: dict, state: CoverageState) -> dict:
    """The fact flags plus what the coverage signs presume: an LLM call when there are signs of AI
    and none was found, an external provider when one shows up only as a credential or a host."""
    presumed = dict(flags)
    if state.ai_without_call:
        presumed["operation_kinds"] = set(flags["operation_kinds"]) | {"invoke_llm"}
    if state.presumed_external:
        presumed["has_external_provider"] = True
    return presumed


_SIGN_WHERE_RE = re.compile(
    r"(?:em|import em|dependência em) ([^\s:,()]+\.[A-Za-z0-9]+)(?::(\d+))?"
)


def _sign_evidence(state: CoverageState) -> list[dict]:
    """Where the signs behind a presumed risk are: file and line when the alert names them."""
    texts = list(state.presumed_external.values()) or state.alerts[1:] or state.alerts
    out: list[dict] = []
    for text in texts:
        m = _SIGN_WHERE_RE.search(text)
        item = {"file": m.group(1) if m else "", "line": int(m.group(2) or 0) if m else 0}
        item["kind"] = "indicio"
        if item not in out:
            out.append(item)
    return out


def _product_tools(scan: ScanOutput) -> list:
    return [t for t in scan.tools if not context_of(t.file)]


def _flag_match(triggers, flags: dict) -> bool:
    return any(
        [
            triggers.has_pii and flags["has_pii"],
            triggers.has_external_provider and flags["has_external_provider"],
            triggers.has_telemetry and flags["has_telemetry"],
            triggers.has_memory_store and flags["has_memory_store"],
            triggers.has_vector_store and flags["has_vector_store"],
        ]
    )


def _entry_matches(entry: CatalogEntry, flags: dict) -> bool:
    t = entry.triggers
    if t.always:
        return True
    if t.operations and set(t.operations) & flags["operation_kinds"]:
        return True
    return _flag_match(t, flags)


_VECTOR_KINDS = {"vector_store_init", "vector_store_write", "vector_store_read"}


def _entry_evidence(entry: CatalogEntry, scan: ScanOutput) -> list[dict]:
    """Collect the (file, line, kind) of the scan items that fired this entry's triggers.

    Mirrors _scan_flags / _entry_matches: the same operations and PII findings that made the entry
    match are the evidence a linter points at. Deterministic, dedup'd, order-stable. An ``always``
    entry (or one whose declared triggers left no scan item) yields ``[]`` — the SARIF then anchors
    the finding at the repo root and the Markdown omits the line. Never invents a location.
    """
    t = entry.triggers
    seen: set[tuple[str, int, str]] = set()
    ev: list[dict] = []

    def _add(file: str, line: int, kind: str) -> None:
        key = (file, line, kind)
        if key not in seen:
            seen.add(key)
            ev.append({"file": file, "line": line, "kind": kind})

    if t.operations:
        opset = set(t.operations)
        for op in product_operations(scan):
            if op.kind in opset:
                _add(op.file, op.line, op.kind)
        if "tool_call" in opset:
            for tool in _product_tools(scan):
                _add(tool.file, tool.line, f"tool:{tool.name}")
    if t.has_pii:
        for f in scan.pii_findings:
            if f.get("confidence_tier", "high") == "high":
                _add(f.get("file", ""), int(f.get("line", 0)), f"pii:{f.get('kind', '')}")
    if t.has_external_provider:
        for op in product_operations(scan):
            if is_external(op):
                _add(op.file, op.line, op.kind)
    if t.has_telemetry:
        for op in product_operations(scan):
            if op.kind == "telemetry":
                _add(op.file, op.line, op.kind)
    if t.has_memory_store:
        for op in product_operations(scan):
            if op.kind == "memory_persist":
                _add(op.file, op.line, op.kind)
    if t.has_vector_store:
        for op in product_operations(scan):
            if op.kind in _VECTOR_KINDS:
                _add(op.file, op.line, op.kind)
    return ev


def _apply_mitigations(prob: int, impact: int, mitigations: list[Mitigation]) -> tuple[int, int]:
    # Cumulative reductions are capped per factor (spec 56) — no control set drops a risk more than
    # ~2 probability bands / 1 impact band. Prevents the 5x4 -> 1x4 collapse the old sum produced.
    prob_red = max(0, min(_MAX_PROB_REDUCTION, sum(m.probability_reduction for m in mitigations)))
    impact_red = max(0, min(_MAX_IMPACT_REDUCTION, sum(m.impact_reduction for m in mitigations)))
    return _clamp(prob - prob_red), _clamp(impact - impact_red)


def score(scan: ScanOutput) -> RiskRegister:
    flags = _scan_flags(scan)
    state = coverage_state(scan)
    presumed_flags = _presumed_flags(flags, state)
    scored: list[ScoredRisk] = []

    for entry in load_catalog():
        by_fact = _entry_matches(entry, flags)
        if not by_fact and not _entry_matches(entry, presumed_flags):
            continue

        prob_inh = entry.probabilidade_baseline
        imp_inh = entry.impacto_baseline
        lvl_inh = _level(prob_inh, imp_inh)

        prob_res, imp_res = _apply_mitigations(prob_inh, imp_inh, entry.mitigacoes)
        prob_res = _floor_residual_prob(prob_res, imp_res, lvl_inh)
        lvl_res = _level(prob_res, imp_res)

        scored.append(
            ScoredRisk(
                id=entry.id,
                categoria=entry.categoria,
                descricao=entry.descricao,
                artigos=entry.artigos,
                probabilidade_inerente=prob_inh,
                impacto_inerente=imp_inh,
                nivel_inerente=lvl_inh,
                probabilidade_residual=prob_res,
                impacto_residual=imp_res,
                nivel_residual=lvl_res,
                mitigacoes=[m.text for m in entry.mitigacoes],
                tratamento=_TRATAMENTO[lvl_res],
                evidencia=_entry_evidence(entry, scan) if by_fact else _sign_evidence(state),
                presumido=not by_fact,
            )
        )

    summary: dict[str, int] = {}
    for r in scored:
        summary[r.nivel_residual] = summary.get(r.nivel_residual, 0) + 1

    return RiskRegister(risks=scored, summary=summary)
