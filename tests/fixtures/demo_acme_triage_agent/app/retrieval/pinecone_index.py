"""Pinecone vector store — historical claim retrieval."""

from __future__ import annotations

import os

from openai import OpenAI
from pinecone import Pinecone

_pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
_index = _pc.Index(os.environ.get("PINECONE_INDEX", "acme-claims-prod"))
_openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _embed(text: str) -> list[float]:
    resp = _openai.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return resp.data[0].embedding


def retrieve_similar_claims(query: str, top_k: int = 5) -> list[str]:
    vec = _embed(query)
    res = _index.query(vector=vec, top_k=top_k, include_metadata=True)
    return [
        match.metadata.get("descricao", "") for match in res.matches if match.metadata is not None
    ]


def upsert_claim(claim_id: str, descricao: str, metadata: dict) -> None:
    vec = _embed(descricao)
    _index.upsert(vectors=[(claim_id, vec, metadata)])
