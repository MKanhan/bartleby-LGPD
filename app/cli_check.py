"""bartleby check — the free CI gate (scan + risk scoring; no LLM, no license, no documents).

This module is shared by two CLIs:
- the full on-prem CLI (`app/cli.py`, `analyze` + `check`) — the Studio/Enterprise product;
- the free gate-only CLI shipped in the public open-core package (`bartleby-LGPD`).

Keep it free of any analyze/LLM/generator/license import — importing this module must stay light
(only scanner + risk_engine + gate). The check UX lives here so it never drifts between the free
and paid builds. See `specs/62_opencore_gate_carveout.md`.

`--scan-out` exports the scan artifact (`app/scanner/artifact.py`): the free side of the
"documents without the source" path — the client keeps the repository and sends one auditable
JSON. Exporting stays offline and license-free; only `analyze` (paid) consumes it. See
`specs/63_scan_artifact_no_source.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from app.analyzer.risk_engine import score
from app.gate import (
    ThresholdError,
    evaluate,
    parse_threshold,
    render_findings,
    render_markdown,
    to_sarif,
)
from app.scanner import scan_directory
from app.scanner.artifact import build as build_artifact
from app.scanner.artifact import redact
from app.scanner.artifact import write_file as write_artifact
from app.scanner.coverage import is_scan_empty
from app.scanner.ingest import UnsafeGitUrlError, ingest_git, ingest_zip, validate_git_url


def _resolve_source(source: str, work_dir: Path) -> Path:
    """Return a directory to scan: a local dir as-is, a .zip extracted, or a git URL cloned."""
    p = Path(source)
    if p.is_dir():
        return p
    if p.is_file() and p.suffix.lower() == ".zip":
        return ingest_zip(p, work_dir)
    if "://" in source or source.startswith("git@") or source.startswith("ext::"):
        validate_git_url(source)  # raises UnsafeGitUrlError
        return ingest_git(source, work_dir)
    raise FileNotFoundError(
        f"fonte não encontrada / não reconhecida: {source} "
        "(esperado: um diretório, um arquivo .zip, ou uma URL git https)"
    )


def run_check(args: argparse.Namespace) -> int:
    """CI gate: scan + score only. No LLM, no license, no documents generated.

    Exit codes: 0 pass · 2 bad source / bad threshold · 3 empty scan · 4 gate breached.
    """
    try:
        threshold = parse_threshold(args.fail_on)
    except ThresholdError as e:
        print(str(e), file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="bartleby-check-") as tmp:
        try:
            source_path = _resolve_source(args.source, Path(tmp))
        except UnsafeGitUrlError as e:
            print(f"URL git rejeitada: {e}", file=sys.stderr)
            return 2
        except (FileNotFoundError, OSError) as e:
            print(str(e), file=sys.stderr)
            return 2

        scan = scan_directory(source_path)

    if is_scan_empty(scan):
        print(
            "Análise vazia: nenhuma operação de tratamento de dados detectada.",
            file=sys.stderr,
        )
        return 3

    register = score(scan)
    result = evaluate(register, threshold)

    if args.summary_md:
        md = render_markdown(
            result,
            framework=scan.agent_framework,
            operations=len(scan.operations),
            pii=len(scan.pii_findings),
            source=args.source,
        )
        findings = render_findings(register)
        if findings:
            md = md + findings + "\n"
        # append (GitHub's $GITHUB_STEP_SUMMARY is append-only); create if missing.
        with open(args.summary_md, "a", encoding="utf-8") as fh:
            fh.write(md)

    if args.sarif:
        sarif = to_sarif(register, root=args.source)
        with open(args.sarif, "w", encoding="utf-8") as fh:
            json.dump(sarif, fh, ensure_ascii=False, indent=2)

    # The scan artifact: everything the paid pipeline needs to write the RIPD/ROPA/Mapa, and
    # nothing else — so a team can buy the documents without ever shipping the repository.
    # Still free, still offline: exporting is a file write, not a call home.
    if getattr(args, "scan_out", None):
        exported = redact(scan) if getattr(args, "redact", False) else scan
        write_artifact(
            args.scan_out,
            build_artifact(
                exported, source_label=args.source, redacted=bool(getattr(args, "redact", False))
            ),
        )

    if args.json:
        payload = {
            "framework": scan.agent_framework,
            "operations": len(scan.operations),
            "pii_findings": len(scan.pii_findings),
            "counts": result.counts,
            "total": result.total,
            "worst": result.worst,
            "threshold": result.threshold,
            "breached": result.breached,
        }
        if getattr(args, "scan_out", None):
            payload["scan_out"] = str(args.scan_out)
            payload["redacted"] = bool(getattr(args, "redact", False))
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"Bartleby — gate LGPD ({scan.agent_framework})")
        print(f"  Operações : {len(scan.operations)}")
        print(f"  PII       : {len(scan.pii_findings)} ocorrência(s)")
        print(f"  Riscos    : {result.total}  (pior severidade: {result.worst or '—'})")
        for lvl in ("Crítico", "Alto", "Médio", "Baixo"):
            print(f"    {lvl:8s} {result.counts[lvl]}")
        if getattr(args, "scan_out", None):
            marca = " (redigido)" if getattr(args, "redact", False) else ""
            print(f"  Artefato  : {args.scan_out}{marca}")
        if result.breached:
            print(
                f"  GATE      : FALHOU — risco {result.worst} ≥ limite {result.threshold}",
                file=sys.stderr,
            )
        elif result.threshold:
            print(f"  GATE      : ok (limite {result.threshold})")
        else:
            print("  GATE      : somente relatório (--fail-on none)")

    return result.exit_code


def add_check_subcommand(sub: argparse._SubParsersAction) -> None:
    """Register the `check` subcommand on a subparsers action (shared by both CLIs)."""
    ck = sub.add_parser(
        "check",
        help="gate de CI: scan + risco, sem gerar documentos (sem LLM, sem licença)",
    )
    ck.add_argument("source", help="diretório, arquivo .zip, ou URL git https")
    ck.add_argument(
        "--fail-on",
        default="none",
        help="falha o build se a pior severidade do achado atingir o nível "
        "(none|baixo|medio|alto|critico; default: none)",
    )
    ck.add_argument(
        "--summary-md",
        default=None,
        help="anexa um resumo Markdown a este arquivo (aponte para $GITHUB_STEP_SUMMARY)",
    )
    ck.add_argument(
        "--sarif",
        default=None,
        help="escreve os achados em SARIF 2.1.0 neste arquivo "
        "(o GitHub code-scanning anota o PR na linha exata)",
    )
    ck.add_argument(
        "--scan-out",
        dest="scan_out",
        default=None,
        help="exporta o resultado do scan (JSON) neste arquivo — é o que o Bartleby precisa "
        "para gerar RIPD/ROPA/Mapa sem receber o código-fonte",
    )
    ck.add_argument(
        "--redact",
        action="store_true",
        help="no artefato exportado, remove as linhas de código (mantém estrutura, caminhos e "
        "severidade; não altera o risco calculado)",
    )
    ck.add_argument("--json", action="store_true", help="imprime o resultado em JSON")
