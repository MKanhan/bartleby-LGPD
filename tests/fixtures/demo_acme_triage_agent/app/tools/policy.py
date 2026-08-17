"""Policy lookup tool — calls internal CRM."""

from __future__ import annotations

import httpx

CRM_BASE = "https://crm.acme.example/api"


def lookup_policy_status(apolice: str) -> str:
    r = httpx.get(f"{CRM_BASE}/policies/{apolice}", timeout=10.0)
    r.raise_for_status()
    return r.json().get("status", "desconhecido")
