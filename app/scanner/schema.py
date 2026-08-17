"""Pydantic models shared across ingest → extractors → orchestrator."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Entrypoint(BaseModel):
    kind: str
    name: str
    file: str
    line: int
    extra: dict[str, str] = Field(default_factory=dict)


class Operation(BaseModel):
    kind: str  # invoke_llm | embed | tool_call | vector_store_write | vector_store_read | memory_persist | telemetry | external_call
    provider: str | None = None
    model: str | None = None
    file: str
    line: int
    snippet: str = ""
    extra: dict[str, str] = Field(default_factory=dict)
    # Sprint 23: local context used by operation_classifier to produce a
    # semantic label instead of a generic SDK-based one. All optional so
    # ScanResult JSON written pre-S23 still loads cleanly.
    enclosing_function: str | None = None
    prev_comment: str | None = None
    return_var: str | None = None


class ToolDef(BaseModel):
    name: str
    description: str = ""
    parameters: list[str] = Field(default_factory=list)
    file: str
    line: int


class ExternalCall(BaseModel):
    url: str | None = None
    method: str = "GET"
    file: str
    line: int
    library: str = "httpx"


class ScanOutput(BaseModel):
    agent_framework: str = "unknown"
    # DEPRECATED (Spec 43): import-composition ratio — epistemically misleading
    # on the RIPD. Kept so ScanResult JSON persisted pre-S43 still loads; new
    # templates use operation_share instead.
    confidence: float = 0.0
    entrypoints: list[Entrypoint] = Field(default_factory=list)
    operations: list[Operation] = Field(default_factory=list)
    tools: list[ToolDef] = Field(default_factory=list)
    external_calls: list[ExternalCall] = Field(default_factory=list)
    pii_findings: list[dict] = Field(default_factory=list)
    vector_stores: list[dict] = Field(default_factory=list)
    memory_stores: list[dict] = Field(default_factory=list)
    raw_facts: dict = Field(default_factory=dict)
    # Spec 43: share of provider-attributed operations per provider (sums to
    # ~1.0); {} when no operation carries a provider. Drives the RIPD stack
    # sentence, replacing the confidence number.
    operation_share: dict[str, float] = Field(default_factory=dict)
