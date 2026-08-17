"""CI gate — scan + score only, no document generation, no license required.

The gate is the *free tier* of the CI wedge. It answers one question on every pull
request: did this change introduce LGPD-relevant data handling whose worst finding
severity sits at or above a threshold the team chose? It never generates the
RIPD/ROPA/Risk-Map — that remediation is the licensed ``analyze`` path. So the gate runs
deterministically, offline, with no API key and no license, which is exactly what makes
it safe to drop into someone else's CI.

Design notes:
- Severity ladder mirrors ``risk_engine._level`` output ("Baixo" < "Médio" < "Alto" <
  "Crítico"). The gate compares the *worst inherent finding severity* present against the
  threshold — the same severity the findings list and SARIF show, so the linter enforces
  on what it displays. The mitigated *residual* stays in the licensed ``analyze`` path.
- ``--fail-on none`` (default) never breaches — the gate reports but never fails the build,
  so adoption is zero-risk and a team can watch it for a few PRs before turning on
  enforcement. Reversibility by default.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.analyzer.risk_engine import RiskRegister, ScoredRisk

# Ordinal ladder — index is severity. Must match risk_engine._level() outputs.
LEVELS: list[str] = ["Baixo", "Médio", "Alto", "Crítico"]
_ORDINAL: dict[str, int] = {name: i for i, name in enumerate(LEVELS)}

# CLI-friendly aliases (ascii + english) → canonical level, or None to disable the gate.
_ALIASES: dict[str, str | None] = {
    "none": None,
    "nenhum": None,
    "off": None,
    "baixo": "Baixo",
    "low": "Baixo",
    "medio": "Médio",
    "médio": "Médio",
    "medium": "Médio",
    "alto": "Alto",
    "high": "Alto",
    "critico": "Crítico",
    "crítico": "Crítico",
    "critical": "Crítico",
}


class ThresholdError(ValueError):
    """Raised when --fail-on gets a value that is not a known level."""


def parse_threshold(value: str) -> str | None:
    """Map a user string to a canonical level, or None (gate disabled). Raises on unknown."""
    key = (value or "").strip().lower()
    if key not in _ALIASES:
        allowed = "none, baixo, medio, alto, critico"
        raise ThresholdError(f"nível inválido para --fail-on: {value!r} (use um de: {allowed})")
    return _ALIASES[key]


@dataclass
class GateResult:
    counts: dict[str, int]  # level -> count, always all four keys present
    total: int
    worst: str | None  # highest level with count > 0, or None when no risks
    threshold: str | None  # canonical level enforced, or None when gate disabled
    breached: bool  # True when worst >= threshold

    @property
    def exit_code(self) -> int:
        return 4 if self.breached else 0


def evaluate(register: RiskRegister, threshold: str | None) -> GateResult:
    """Compare the worst *inherent* finding severity against the threshold.

    The gate is a linter: it enforces on the severity it shows — the inherent level of
    each fired risk — not on the mitigated *residual* that the S56 model compresses down
    to Médio. So --fail-on alto/critico actually fire, and the counts table, the findings
    list, and the SARIF all speak one severity axis. The residual model is untouched: it
    stays in the licensed ``analyze`` path (RIPD/Risk-Map), which the gate never runs.
    """
    counts = {lvl: 0 for lvl in LEVELS}
    for r in register.risks:
        if r.nivel_inerente in counts:
            counts[r.nivel_inerente] += 1
    total = sum(counts.values())
    present = [lvl for lvl in LEVELS if counts[lvl] > 0]
    worst = present[-1] if present else None
    breached = (
        threshold is not None and worst is not None and _ORDINAL[worst] >= _ORDINAL[threshold]
    )
    return GateResult(
        counts=counts, total=total, worst=worst, threshold=threshold, breached=breached
    )


_EMOJI = {"Baixo": "🟢", "Médio": "🟡", "Alto": "🟠", "Crítico": "🔴"}


def render_markdown(
    result: GateResult,
    *,
    framework: str,
    operations: int,
    pii: int,
    source: str,
) -> str:
    """A PR-friendly Markdown summary. Wire the CLI's --summary-md to $GITHUB_STEP_SUMMARY."""
    verdict = (
        f"❌ **Gate falhou** — severidade `{result.worst}` ≥ limite `{result.threshold}`"
        if result.breached
        else (
            "✅ **Gate ok** — nenhum risco no ou acima do limite " f"`{result.threshold}`"
            if result.threshold
            else "🔵 **Somente relatório** — gate desligado (`--fail-on none`)"
        )
    )
    lines = [
        "## Bartleby — LGPD gate",
        "",
        verdict,
        "",
        f"- **Fonte:** `{source}`",
        f"- **Framework detectado:** {framework}",
        f"- **Operações de tratamento:** {operations}",
        f"- **Achados de PII:** {pii}",
        f"- **Riscos totais:** {result.total}",
        "",
        "| Severidade | Achados |",
        "| --- | ---: |",
    ]
    for lvl in reversed(LEVELS):
        lines.append(f"| {_EMOJI[lvl]} {lvl} | {result.counts[lvl]} |")
    lines += [
        "",
        "<sub>Gate determinístico (sem LLM, sem envio de código). "
        "Gere RIPD/ROPA/Mapa de Riscos com `bartleby analyze` — "
        "[Bartleby](https://bartleby.com.br).</sub>",
        "",
    ]
    return "\n".join(lines)


# --- Linter surface (spec 61): named findings + SARIF -----------------------
#
# One severity axis across the whole free tier: the counts table, this findings list, the SARIF,
# and the --fail-on decision all read the *inherent* level — the honest, un-flattened severity the
# engine computes. (The mitigated *residual* the S56 model compresses to Médio is not a CI concern;
# it stays in the licensed analyze path / RIPD.) A risk that fires shows its true 🟠 Alto here and
# fails the gate at --fail-on alto. Both surfaces are pure and offline.

_SARIF_LEVEL: dict[str, str] = {
    "Crítico": "error",
    "Alto": "error",
    "Médio": "warning",
    "Baixo": "note",
}
_TOOL_URI = "https://bartleby.com.br"


def _by_inherent_severity(register: RiskRegister) -> list[ScoredRisk]:
    """Risks worst-inherent-first, ties stable by catalog order."""
    return sorted(register.risks, key=lambda r: _ORDINAL.get(r.nivel_inerente, 0), reverse=True)


def _artigos(risk: ScoredRisk) -> str:
    return ", ".join(risk.artigos) if risk.artigos else "—"


def render_findings(register: RiskRegister) -> str:
    """A named-finding list, worst inherent severity first — what makes the gate read as a linter.

    `AI-002` 🟠 **Alto** — <descrição> · LGPD art. X, Y
      - Evidência: `file:line` (+N)
      - Remediação: <primeira mitigação>

    Returns "" when there are no risks, so the caller can skip the section entirely.
    """
    risks = _by_inherent_severity(register)
    if not risks:
        return ""
    lines = ["### Achados (linter LGPD)", ""]
    for r in risks:
        emoji = _EMOJI.get(r.nivel_inerente, "")
        lines.append(
            f"- `{r.id}` {emoji} **{r.nivel_inerente}** — {r.descricao} · LGPD {_artigos(r)}"
        )
        if r.evidencia:
            first = r.evidencia[0]
            loc = f"`{first.get('file', '')}:{first.get('line', 0)}`"
            if len(r.evidencia) > 1:
                loc += f" (+{len(r.evidencia) - 1})"
            lines.append(f"  - Evidência: {loc}")
        if r.mitigacoes:
            lines.append(f"  - Remediação: {r.mitigacoes[0]}")
    lines.append("")
    return "\n".join(lines)


def to_sarif(register: RiskRegister, root: str = ".") -> dict:
    """SARIF 2.1.0 — GitHub code-scanning consumes it and renders each result as an inline PR
    annotation on the exact line. Severity (`level`) maps from the inherent level. A finding with
    matched evidence anchors at `file:line`; without it, at the repo root. No location is invented.
    """
    rules: dict[str, dict] = {}
    results: list[dict] = []
    for r in register.risks:
        if r.id not in rules:
            rules[r.id] = {
                "id": r.id,
                "name": r.categoria,
                "shortDescription": {"text": r.descricao},
                "fullDescription": {"text": f"{r.descricao} — LGPD {_artigos(r)}"},
                "helpUri": _TOOL_URI,
                "properties": {"lgpd_artigos": r.artigos, "nivel_inerente": r.nivel_inerente},
            }
        locations: list[dict] = []
        for ev in r.evidencia:
            uri = ev.get("file") or ""
            if not uri:
                continue
            locations.append(
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": uri},
                        "region": {"startLine": max(1, int(ev.get("line", 1) or 1))},
                    }
                }
            )
        if not locations:
            locations.append({"physicalLocation": {"artifactLocation": {"uri": root}}})
        results.append(
            {
                "ruleId": r.id,
                "level": _SARIF_LEVEL.get(r.nivel_inerente, "warning"),
                "message": {
                    "text": f"{r.id} [{r.nivel_inerente}] {r.descricao} — LGPD {_artigos(r)}"
                },
                "locations": locations,
            }
        )
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "Bartleby",
                        "informationUri": _TOOL_URI,
                        "rules": list(rules.values()),
                    }
                },
                "results": results,
            }
        ],
    }
