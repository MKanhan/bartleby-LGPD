"""Inspect .env, dependency manifests, Dockerfile for config-level facts."""

from __future__ import annotations

import re
from pathlib import Path

from app.scanner import provider_catalog as pc
from app.scanner import store_catalog as sc

_RETENTION_RE = re.compile(
    r"^\s*([A-Z0-9_]*(?:TTL|RETENTION|MAX_AGE|EXPIRE|EXPIRY|HISTORY_LIMIT)[A-Z0-9_]*)\s*[:=]\s*(\S+)",
    re.MULTILINE,
)
# A declared variable: `KEY=...` in .env, `export KEY=`, `ENV KEY=`/`ARG KEY` in a Dockerfile.
# Which provider a variable belongs to comes from provedores.yml (spec 65).
_DECLARED_VAR_RE = re.compile(
    r"^\s*(?:export\s+|ENV\s+|ARG\s+)?([A-Z][A-Z0-9_]*)\s*(?:[:=]|$)", re.MULTILINE
)
_BASE_URL_RE = re.compile(
    r"^\s*([A-Z0-9_]*(?:BASE_URL|ENDPOINT)[A-Z0-9_]*)\s*[:=]\s*(\S+)",
    re.MULTILINE,
)
_NAMES_SCANNED = {
    ".env",
    ".env.example",
    ".env.sample",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "setup.py",
    "Pipfile",
    "Dockerfile",
    "package.json",
    "wrangler.toml",
}
# Cloudflare declares bindings as array tables: [[vectorize]], [[d1_databases]] (spec 66).
_WRANGLER_SECTION_RE = re.compile(r"^[ \t]*\[\[[ \t]*([a-z0-9_]+)[ \t]*\]\]", re.MULTILINE)
# spec 69: single tables with a binding (`[ai]` / `binding = "AI"`) name a provider the code calls
_WRANGLER_TABLE_RE = re.compile(r"^[ \t]*\[[ \t]*([a-z0-9_]+)[ \t]*\][ \t]*$", re.MULTILINE)
_WRANGLER_BINDING_RE = re.compile(
    r'^[ \t]*binding[ \t]*=[ \t]*["\']([A-Za-z0-9_]+)["\']', re.MULTILINE
)


def wrangler_ai_bindings(root: Path) -> dict[str, str]:
    """{binding name: provider id} from `[section]` tables of wrangler.toml files whose section
    belongs to a catalog provider (Workers AI: `[ai]`)."""
    out: dict[str, str] = {}
    for cfg in Path(root).rglob("wrangler.toml"):
        if "node_modules" in cfg.parts:
            continue
        text = cfg.read_text(encoding="utf-8", errors="ignore")
        tables = list(_WRANGLER_TABLE_RE.finditer(text))
        for i, m in enumerate(tables):
            provider = pc.by_wrangler_section(m.group(1))
            if provider is None:
                continue
            end = tables[i + 1].start() if i + 1 < len(tables) else len(text)
            b = _WRANGLER_BINDING_RE.search(text, m.end(), end)
            if b is not None:
                out[b.group(1)] = provider.id
    return out


# spec 73: the datasource of a Prisma schema names the database; a SQL file may prove a store
_PRISMA_DATASOURCE_RE = re.compile(
    r"^[ \t]*datasource\s+\w+\s*\{[^}]*?^[ \t]*provider\s*=\s*\"([a-z0-9_]+)\"",
    re.MULTILINE | re.DOTALL,
)
_IGNORED_DIRS = {"node_modules", ".git", ".venv", "venv", "dist", "build", ".next"}


def _declared_stores(rel: str, name: str, text: str) -> list[dict]:
    out: list[dict] = []
    if name == "wrangler.toml":
        for m in _WRANGLER_SECTION_RE.finditer(text):
            store = sc.by_wrangler_section(m.group(1))
            if store is not None:
                section = f"[[{m.group(1)}]]"
                out.append(
                    {
                        "file": rel,
                        "line": text.count("\n", 0, m.start()) + 1,
                        "store": store.id,
                        "section": m.group(1),
                        "snippet": section,
                        "origin": f"{rel} {section}",
                    }
                )
    elif name.endswith(".prisma"):
        for m in _PRISMA_DATASOURCE_RE.finditer(text):
            store = sc.by_prisma_provider(m.group(1))
            if store is not None:
                out.append(
                    {
                        "file": rel,
                        "line": text.count("\n", 0, m.start(1)) + 1,
                        "store": store.id,
                        "snippet": f'provider = "{m.group(1)}"',
                        "origin": f"datasource do Prisma em {rel} (provider {m.group(1)})",
                    }
                )
    elif name.endswith(".sql"):
        store = sc.by_sql(text)
        if store is not None:
            out.append(
                {
                    "file": rel,
                    "line": 1,
                    "store": store.id,
                    "snippet": "",
                    "origin": f"SQL em {rel}",
                }
            )
    return out


def scan_config(root: Path) -> dict:
    root = Path(root)
    findings: dict = {
        "providers_declared": [],
        "provider_vars": {},  # spec 75: {provider id: the variable that names it}
        "base_urls": [],
        "retention_hints": [],
        "stores_declared": [],
        "files_seen": [],
    }
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        declares_store = candidate.suffix in (".prisma", ".sql")
        if candidate.name not in _NAMES_SCANNED and not declares_store:
            continue
        rel = candidate.relative_to(root).as_posix()
        if declares_store:
            # not a configuration file: only what it says about where data is kept
            if _IGNORED_DIRS.isdisjoint(candidate.relative_to(root).parts):
                text = candidate.read_text(encoding="utf-8", errors="ignore")
                findings["stores_declared"].extend(_declared_stores(rel, candidate.name, text))
            continue
        findings["files_seen"].append(rel)
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in _DECLARED_VAR_RE.finditer(text):
            provider = pc.provider_for_env_var(m.group(1))
            if provider:
                findings["provider_vars"].setdefault(provider, m.group(1))
            if provider and provider not in findings["providers_declared"]:
                findings["providers_declared"].append(provider)
        for m in _BASE_URL_RE.finditer(text):
            value = m.group(2).strip("\"'")
            hit = pc.by_host(value) if "://" in value or ":" in value else None
            findings["base_urls"].append(
                {
                    "file": rel,
                    "var": m.group(1),
                    "value": value,
                    "provider": hit.id if hit else None,
                }
            )
            if hit is not None:
                findings["provider_vars"].setdefault(hit.id, m.group(1))
            if hit is not None and hit.id not in findings["providers_declared"]:
                findings["providers_declared"].append(hit.id)
        findings["stores_declared"].extend(_declared_stores(rel, candidate.name, text))
        for m in _RETENTION_RE.finditer(text):
            findings["retention_hints"].append(
                {"file": rel, "var": m.group(1), "value": m.group(2).strip("\"'")}
            )
    return findings
