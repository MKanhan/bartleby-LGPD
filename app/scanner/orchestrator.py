"""Top-level scanner pipeline — produces a ScanOutput from a work directory."""

from __future__ import annotations

from pathlib import Path

from app.scanner import store_catalog
from app.scanner.ai_signals import collect as collect_ai_signals
from app.scanner.config_scan import scan_config
from app.scanner.context import context_of
from app.scanner.extractors.datastores import analyze_repo as analyze_datastores
from app.scanner.extractors.http_calls import analyze_repo as analyze_http_calls
from app.scanner.extractors.integrations import analyze_integrations
from app.scanner.extractors.llm_calls import analyze_repo as analyze_llm_repo
from app.scanner.extractors.python_ast import analyze_python_file, iter_python_files
from app.scanner.extractors.tree_sitter_js import analyze_js_repo, iter_js_files
from app.scanner.facts import operation_share
from app.scanner.framework_detector import detect_framework
from app.scanner.pii.engine import scan_directory_for_pii
from app.scanner.provider_tables import python_tables
from app.scanner.py_index import RepoIndex
from app.scanner.schema import Operation, ScanOutput


def scan_directory(root: Path) -> ScanOutput:
    root = Path(root)
    framework, confidence = detect_framework(root)
    out = ScanOutput(agent_framework=framework, confidence=confidence)

    py_files = iter_python_files(root)
    index = RepoIndex.build(root, py_files)
    out.operations.extend(analyze_llm_repo(index))
    out.external_calls.extend(analyze_http_calls(index))
    provider_tables = python_tables(index)
    for path in py_files:
        py_facts = analyze_python_file(path, root, include_llm=False)
        out.entrypoints.extend(py_facts.entrypoints)
        out.tools.extend(py_facts.tools)

        integ = analyze_integrations(path, root)
        out.operations.extend(integ.vector_ops)
        out.operations.extend(integ.memory_ops)
        out.operations.extend(integ.telemetry_ops)
        out.vector_stores.extend(op.model_dump() for op in integ.vector_ops)
        out.memory_stores.extend(op.model_dump() for op in integ.memory_ops)

    # Spec 65: databases and catalog vector stores. A vector store the integrations extractor
    # already saw in the same file is not repeated.
    seen_vectors = {(op.file, op.provider) for op in out.operations if op.kind.startswith("vector")}
    for op in analyze_datastores(index):
        if op.kind.startswith("vector"):
            if (op.file, op.provider) in seen_vectors:
                continue
            out.vector_stores.append(op.model_dump())
        out.operations.append(op)

    js_files = iter_js_files(root)
    for js_facts in analyze_js_repo(root, js_files):
        out.entrypoints.extend(js_facts.entrypoints)
        out.operations.extend(js_facts.operations)
        out.tools.extend(js_facts.tools)
        out.external_calls.extend(js_facts.external_calls)
        provider_tables.extend(js_facts.provider_tables)
    # spec 73: providers the code lists in a table and picks at run time — candidates, not facts
    out.raw_facts["tabelas_de_provedores"] = [
        t for t in provider_tables if not context_of(t["file"])
    ]

    out.pii_findings = [f.__dict__ for f in scan_directory_for_pii(root)]
    out.raw_facts["config"] = scan_config(root)
    # Spec 66: bindings declared in configuration (wrangler.toml) are where data is kept too;
    # spec 73: so are a Prisma datasource and a SQL file that proves a store.
    for decl in out.raw_facts["config"].get("stores_declared", []):
        store = store_catalog.get(decl["store"])
        if store is None:
            continue
        out.operations.append(
            Operation(
                kind="vector_store_init" if store.is_vector else "datastore",
                provider=store.id if store.is_vector else None,
                file=decl["file"],
                line=decl["line"],
                snippet=decl["snippet"],
                extra={"store": store.id, "store_origin": decl["origin"]},
            )
        )
    out.raw_facts["ai_signals"] = collect_ai_signals(root)
    out.raw_facts["files_scanned_py"] = len(py_files)
    out.raw_facts["arquivos_nao_lidos"] = sorted(f for f in index.unparsed if not context_of(f))
    out.raw_facts["files_scanned_js"] = len(js_files)

    for op in out.operations:
        context = context_of(op.file)
        if context:
            op.extra["contexto"] = context

    out.operations = _one_item_per_store(out.operations)
    out.operation_share = operation_share(out.operations)
    return out


def _one_item_per_store(ops: list[Operation]) -> list[Operation]:
    """One operation per store and kind in the repository (spec 66).

    A database imported by five files is one place where data is kept, not five activities.
    The first product-code occurrence stays (test/example code only if nothing else); the other
    places go to `extra.arquivos`. Vector-store reads and writes are left as they are.
    """
    kept: list[Operation] = []
    groups: dict[tuple[str, str], list[Operation]] = {}
    for op in ops:
        if op.kind in ("datastore", "vector_store_init"):
            key = (op.kind, str(op.extra.get("store") or op.provider))
            groups.setdefault(key, []).append(op)
        else:
            kept.append(op)
    for items in groups.values():
        items.sort(key=lambda o: (bool(o.extra.get("contexto")), o.file, o.line))
        primary = items[0]
        others = sorted({f"{o.file}:{o.line}" for o in items[1:] if not o.extra.get("contexto")})
        if others:
            primary.extra["arquivos"] = others
        kept.append(primary)
    return kept
