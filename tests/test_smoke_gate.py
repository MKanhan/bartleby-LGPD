"""Smoke test do gate — roda `bartleby check` contra a fixture ACME (fictícia)."""

import json
import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "demo_acme_triage_agent"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "app.cli", "check", str(FIXTURE), *args],
        capture_output=True,
        text=True,
    )


def test_gate_reprova_em_alto():
    assert _run("--fail-on", "alto").returncode == 4


def test_gate_passa_em_critico():
    # ACME tem severidade máxima Alto (sem Crítico) — a escada discrimina.
    assert _run("--fail-on", "critico").returncode == 0


def test_sarif_tem_regra_ai(tmp_path):
    out = tmp_path / "gate.sarif"
    _run("--sarif", str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    rule_ids = [r["ruleId"] for r in data["runs"][0]["results"]]
    assert any(rid.startswith("AI-") for rid in rule_ids)


def test_scan_out_exporta_artefato(tmp_path):
    out = tmp_path / "scan.json"
    assert _run("--scan-out", str(out)).returncode == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["format"] == "bartleby.scan/1"
    assert data["scan"]["operations"]


def test_redact_tira_o_codigo(tmp_path):
    out = tmp_path / "scan.json"
    assert _run("--scan-out", str(out), "--redact").returncode == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["redacted"] is True
    assert all(op["snippet"] == "" for op in data["scan"]["operations"])
