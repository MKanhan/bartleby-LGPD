"""Top-level scanner pipeline — produces a ScanOutput from a work directory."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from app.scanner.config_scan import scan_config
from app.scanner.extractors.integrations import analyze_integrations
from app.scanner.extractors.python_ast import analyze_python_file, iter_python_files
from app.scanner.extractors.tree_sitter_js import analyze_js_file, iter_js_files
from app.scanner.framework_detector import detect_framework
from app.scanner.pii.engine import scan_directory_for_pii
from app.scanner.schema import ScanOutput


def scan_directory(root: Path) -> ScanOutput:
    root = Path(root)
    framework, confidence = detect_framework(root)
    out = ScanOutput(agent_framework=framework, confidence=confidence)

    py_files = iter_python_files(root)
    for path in py_files:
        py_facts = analyze_python_file(path, root)
        out.entrypoints.extend(py_facts.entrypoints)
        out.operations.extend(py_facts.operations)
        out.tools.extend(py_facts.tools)
        out.external_calls.extend(py_facts.external_calls)

        integ = analyze_integrations(path, root)
        out.operations.extend(integ.vector_ops)
        out.operations.extend(integ.memory_ops)
        out.operations.extend(integ.telemetry_ops)
        out.vector_stores.extend(op.model_dump() for op in integ.vector_ops)
        out.memory_stores.extend(op.model_dump() for op in integ.memory_ops)

    js_files = iter_js_files(root)
    for path in js_files:
        js_facts = analyze_js_file(path, root)
        out.entrypoints.extend(js_facts.entrypoints)
        out.operations.extend(js_facts.operations)
        out.tools.extend(js_facts.tools)

    out.pii_findings = [f.__dict__ for f in scan_directory_for_pii(root)]
    out.raw_facts["config"] = scan_config(root)
    out.raw_facts["files_scanned_py"] = len(py_files)
    out.raw_facts["files_scanned_js"] = len(js_files)

    providers = [op.provider for op in out.operations if op.provider]
    if providers:
        total = len(providers)
        out.operation_share = {p: c / total for p, c in Counter(providers).items()}
    return out
