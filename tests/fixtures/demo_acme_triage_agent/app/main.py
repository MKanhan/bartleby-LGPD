"""ACME triage backend — entry point."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr, Field

from app.agents.triage import triage_claim
from app.models.claim import Claim, ClaimResponse

app = FastAPI(title="ACME Seguros — Triagem de Sinistros")


class TriageRequest(BaseModel):
    cpf: str = Field(..., description="CPF do segurado")
    nome_completo: str
    email: EmailStr
    telefone: str
    apolice: str
    descricao_sinistro: str


@app.post("/triage", response_model=ClaimResponse)
async def submit_claim(req: TriageRequest) -> ClaimResponse:
    claim = Claim(
        cpf=req.cpf,
        nome=req.nome_completo,
        email=req.email,
        telefone=req.telefone,
        apolice=req.apolice,
        descricao=req.descricao_sinistro,
    )
    if not claim.cpf:
        raise HTTPException(status_code=400, detail="CPF é obrigatório")
    return await triage_claim(claim)


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "service": os.getenv("LANGCHAIN_PROJECT", "acme-triage")}
