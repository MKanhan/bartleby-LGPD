"""Detect whether a scan recovered enough signal to justify rendering reports."""

from __future__ import annotations

from app.scanner.schema import ScanOutput

CONFIDENCE_FLOOR = 0.3


def is_scan_empty(scan: ScanOutput) -> bool:
    """True when the scan produced no actionable structural evidence.

    PII regex hits without surrounding operations or entry points are noise
    (a string that looks like an email is not evidence of processing). Same for
    raw_facts — they may contain config without behavior.
    """
    return (
        len(scan.operations) == 0
        and len(scan.entrypoints) == 0
        and scan.confidence < CONFIDENCE_FLOOR
    )
