"""Human review tool — opens a ticket for an analyst."""

from __future__ import annotations

import httpx

QUEUE_BASE = "https://queue.acme.example/api"


def request_human_review(claim_id: str, reason: str) -> dict:
    r = httpx.post(
        f"{QUEUE_BASE}/reviews",
        json={"claim_id": claim_id, "reason": reason[:500]},
        timeout=10.0,
    )
    r.raise_for_status()
    return r.json()
