"""LLM busy flag helpers.

Purpose: allow the backend to pause ML auto-predictions while an LLM explanation
is running, to reduce resource contention.

We store a short-lived per-machine flag in Redis (cache DB).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import json

import redis.asyncio as redis

from config import settings


LLM_BUSY_TTL_SECONDS = 60 * 5  # 5 minutes (failsafe: expires if worker crashes)


def _busy_key(machine_id: str) -> str:
    mid = (machine_id or "").strip()[:128] or "unknown"
    return f"pm:llm_busy:{mid}"


async def _get_redis() -> redis.Redis:
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_CACHE_DB,
        decode_responses=True,
    )


async def set_llm_busy(
    machine_id: str,
    task_id: Optional[str] = None,
    client_id: Optional[str] = None,
    ttl_seconds: int = LLM_BUSY_TTL_SECONDS,
) -> None:
    mid = (machine_id or "").strip()
    if not mid:
        return

    payload: Dict[str, Any] = {
        "machine_id": mid,
        "task_id": (task_id or "").strip() or None,
        "client_id": (client_id or "").strip() or None,
        "started_at": datetime.utcnow().isoformat() + "Z",
    }

    r: Optional[redis.Redis] = None
    try:
        r = await _get_redis()
        await r.setex(_busy_key(mid), max(10, int(ttl_seconds)), json.dumps(payload, default=str))
    finally:
        if r is not None:
            try:
                await r.close()
            except Exception:
                pass


async def clear_llm_busy(machine_id: str) -> None:
    mid = (machine_id or "").strip()
    if not mid:
        return

    r: Optional[redis.Redis] = None
    try:
        r = await _get_redis()
        await r.delete(_busy_key(mid))
    finally:
        if r is not None:
            try:
                await r.close()
            except Exception:
                pass


async def is_llm_busy(machine_id: str) -> bool:
    mid = (machine_id or "").strip()
    if not mid:
        return False

    r: Optional[redis.Redis] = None
    try:
        r = await _get_redis()
        return bool(await r.exists(_busy_key(mid)))
    except Exception:
        # If Redis isn't reachable, don't block predictions.
        return False
    finally:
        if r is not None:
            try:
                await r.close()
            except Exception:
                pass
