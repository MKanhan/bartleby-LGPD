"""Triage agent — orchestrates LLM call + tool dispatch + memory."""

from __future__ import annotations

import os
import uuid

from anthropic import Anthropic
from langsmith import Client
from openai import OpenAI

from app.memory.redis_memory import append_message, get_history
from app.models.claim import Claim, ClaimResponse
from app.retrieval.pinecone_index import retrieve_similar_claims
from app.tools.notify import notify_claimant_email
from app.tools.policy import lookup_policy_status
from app.tools.review import request_human_review

_ls = Client(api_key=os.environ["LANGCHAIN_API_KEY"])
_openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
_anthropic = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")
_ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")


SYSTEM_PROMPT = """\
Você é o agente de triagem de sinistros da ACME Seguros.
Receba a descrição do segurado, contexto histórico (sinistros similares),
e produza uma classificação preliminar + recomendação de encaminhamento.

Você está autorizado a acessar dados pessoais do segurado (nome, CPF,
e-mail, telefone, apólice). Esses dados serão tratados sob a base legal
do Art. 7º, V (execução do contrato de seguro) e Art. 7º, IX (legítimo
interesse na detecção de fraudes).
"""


async def triage_claim(claim: Claim) -> ClaimResponse:
    history = get_history(claim.cpf)
    similar = retrieve_similar_claims(claim.descricao, top_k=5)
    policy = lookup_policy_status(claim.apolice)

    user_msg = (
        f"Segurado: {claim.nome} (CPF {claim.cpf}, apólice {claim.apolice})\n"
        f"Descrição: {claim.descricao}\n"
        f"Histórico recente:\n{history}\n\n"
        f"Apólice: {policy}\n"
        f"Sinistros similares (RAG): {similar}\n"
    )
    append_message(claim.cpf, "user", user_msg)

    try:
        completion = _openai.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        llm_text = completion.choices[0].message.content
        provider_used = "openai"
    except Exception:
        message = _anthropic.messages.create(
            model=_ANTHROPIC_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        llm_text = message.content[0].text
        provider_used = "anthropic"

    append_message(claim.cpf, "assistant", llm_text)

    requires_human = "REVISAO_HUMANA" in (llm_text or "").upper()
    if requires_human:
        request_human_review(claim_id=claim.cpf, reason=llm_text)

    notify_sent = notify_claimant_email(
        to=claim.email,
        nome=claim.nome,
        classification=llm_text or "—",
    )

    return ClaimResponse(
        claim_id=str(uuid.uuid4()),
        classificacao=(llm_text or "n/a").split("\n", 1)[0],
        encaminhamento=f"via {provider_used}",
        requer_revisao_humana=requires_human,
        cidato_email_disparado=notify_sent,
    )
