"""Pydantic models for claims."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, EmailStr


class Claim(BaseModel):
    cpf: str
    nome: str
    email: EmailStr
    telefone: str
    apolice: str
    descricao: str
    data_nascimento: date | None = None


class ClaimResponse(BaseModel):
    claim_id: str
    classificacao: str
    encaminhamento: str
    requer_revisao_humana: bool
    cidato_email_disparado: bool
