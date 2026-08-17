"""Conversation memory — Redis-backed, per CPF, with TTL."""

from __future__ import annotations

import json
import os

import redis
from langchain_community.chat_message_histories import RedisChatMessageHistory

_TTL = int(os.getenv("CHAT_HISTORY_TTL", "86400"))
_redis = redis.Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)


def _key(cpf: str) -> str:
    return f"acme:triage:history:{cpf}"


def get_history(cpf: str) -> str:
    raw = _redis.lrange(_key(cpf), 0, 19)
    return "\n".join(raw) if raw else "(sem histórico anterior)"


def append_message(cpf: str, role: str, text: str) -> None:
    history = RedisChatMessageHistory(
        session_id=cpf,
        url=os.environ["REDIS_URL"],
        ttl=_TTL,
    )
    if role == "user":
        history.add_user_message(text)
    else:
        history.add_ai_message(text)
    _redis.lpush(_key(cpf), json.dumps({"role": role, "text": text[:200]}))
    _redis.expire(_key(cpf), _TTL)
