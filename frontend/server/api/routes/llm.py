"""\
LLM API Routes

Provides endpoints for generating natural-language explanations of ML predictions.

Design goals:
- Do not block normal API traffic with long-running CPU inference.
- Use Celery tasks for LLM generation and poll for results.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from typing import Any, Dict, Optional
from datetime import datetime
import logging

import json
import time

import redis.asyncio as redis

from celery_app import celery_app
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/llm", tags=["LLM"])


# Cooldown between starts of LLM explanation generation per client+machine.
# Goal: avoid hammering a slow CPU-only model and keep other workloads responsive.
LLM_EXPLAIN_COOLDOWN_SECONDS = 120
LLM_INFLIGHT_TTL_SECONDS = 60 * 15  # keep inflight mapping for 15 minutes
LLM_LAST_TTL_SECONDS = 60 * 60 * 24  # 24h


async def _get_redis() -> redis.Redis:
    # Use DB 3 (API cache) so we don't mix with Celery broker/result DBs.
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_CACHE_DB,
        decode_responses=True,
    )


def _client_key(request: Request) -> str:
    cid = (request.headers.get("x-pm-client-id") or "").strip()
    if cid:
        return cid[:128]
    client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}"


def _normalize_use_case(use_case: str) -> str:
    uc = (use_case or "").strip().lower()
    if not uc:
        return "default"
    return uc[:64]


def _redis_key_inflight(client_key: str, machine_id: str, use_case: str) -> str:
    mid = (machine_id or "").strip()[:128] or "unknown"
    uc = _normalize_use_case(use_case)
    return f"llm:explain:inflight:{client_key}:{mid}:{uc}"


def _redis_key_last(client_key: str, machine_id: str, use_case: str) -> str:
    mid = (machine_id or "").strip()[:128] or "unknown"
    uc = _normalize_use_case(use_case)
    return f"llm:explain:last:{client_key}:{mid}:{uc}"


def _is_terminal_state(state: str) -> bool:
    return state in {"SUCCESS", "FAILURE", "REVOKED"}


@router.post(
    "/explain",
    summary="Start LLM explanation generation (async)",
)
async def start_explain(request: Request, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Start an LLM explanation job.

    The frontend should POST the prediction context and then poll `/api/llm/explain/{task_id}`.
    """
    machine_id = str(payload.get("machine_id") or "").strip()
    # Major workflow change: always run a single combined explanation.
    # This reduces latency and avoids partial/mismatched outputs.
    requested_use_case = str(payload.get("use_case") or "default")
    use_case = "combined"
    client_key = _client_key(request)

    # If the client provides a run_id, always explain the stored run.
    # This prevents mismatches when the UI has moved on to a new snapshot.
    run_id = str(payload.get("run_id") or "").strip()
    if run_id:
        try:
            from services.history_store import get_run

            run = await get_run(run_id)
            if run:
                # Override payload with the canonical stored context.
                payload = dict(payload)
                payload["run_id"] = run.get("run_id") or run_id
                payload["machine_id"] = run.get("machine_id") or machine_id
                payload["data_stamp"] = run.get("data_stamp")
                payload["sensor_data"] = run.get("sensor_data") or {}
                payload["predictions"] = run.get("predictions") or {}
                machine_id = str(payload.get("machine_id") or "").strip()
        except Exception:
            # Best-effort: if lookup fails, fall back to provided payload.
            pass

    # Include client_id for WebSocket push updates (Redis pub/sub)
    # Keep the original payload shape otherwise.
    payload = dict(payload)
    payload.setdefault("client_id", client_key)
    payload["use_case"] = "combined"
    if requested_use_case and requested_use_case.strip().lower() not in {"combined", "default", "auto"}:
        payload.setdefault("requested_use_case", str(requested_use_case)[:64])

    redis_client: Optional[redis.Redis] = None
    try:
        from services.llm_busy import set_llm_busy

        redis_client = await _get_redis()

        inflight_key = _redis_key_inflight(client_key, machine_id, use_case)
        last_key = _redis_key_last(client_key, machine_id, use_case)

        # If an explanation is already in-flight for this client+machine, return it.
        existing_raw = await redis_client.get(inflight_key)
        if existing_raw:
            try:
                existing = json.loads(existing_raw)
                existing_task_id = str(existing.get("task_id") or "").strip()
            except Exception:
                existing_task_id = ""

            if existing_task_id:
                state = celery_app.AsyncResult(existing_task_id).state
                if not _is_terminal_state(state):
                    # Ensure the machine is marked busy (best-effort) while the task is inflight.
                    await set_llm_busy(machine_id, task_id=existing_task_id, client_id=client_key)
                    return {
                        "success": True,
                        "task_id": existing_task_id,
                        "status": state,
                        "submitted_at": existing.get("submitted_at") or datetime.utcnow().isoformat(),
                        "deduped": True,
                        "use_case": _normalize_use_case(use_case),
                    }
                # Terminal state: clear inflight mapping.
                await redis_client.delete(inflight_key)

        # Cooldown enforcement (between starts).
        last_started_raw = await redis_client.get(last_key)
        if last_started_raw:
            try:
                last_started = float(last_started_raw)
            except Exception:
                last_started = 0.0
            now = time.time()
            elapsed = now - last_started
            if elapsed < LLM_EXPLAIN_COOLDOWN_SECONDS:
                retry_after = int(LLM_EXPLAIN_COOLDOWN_SECONDS - elapsed)
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "LLMCooldown",
                        "detail": f"LLM explanation is rate-limited for '{_normalize_use_case(use_case)}'. Try again in ~{retry_after}s.",
                        "retry_after": retry_after,
                        "use_case": _normalize_use_case(use_case),
                    },
                )

        # Route LLM work to a dedicated queue so it doesn't block GAN/ML workers.
        task = celery_app.send_task(
            "tasks.llm_tasks.generate_explanation",
            args=[payload],
            queue="llm",
        )

        # Mark machine busy so ML auto-predictions can pause until LLM completes.
        await set_llm_busy(machine_id, task_id=task.id, client_id=client_key)

        submitted_at = datetime.utcnow().isoformat()
        await redis_client.setex(
            inflight_key,
            LLM_INFLIGHT_TTL_SECONDS,
            json.dumps({"task_id": task.id, "submitted_at": submitted_at}, default=str),
        )
        await redis_client.setex(last_key, LLM_LAST_TTL_SECONDS, str(time.time()))

        return {
            "success": True,
            "task_id": task.id,
            "status": "PENDING",
            "submitted_at": submitted_at,
            "use_case": _normalize_use_case(use_case),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enqueue LLM explanation task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start explanation task: {str(e)}")
    finally:
        if redis_client is not None:
            try:
                await redis_client.close()
            except Exception:
                pass


@router.get(
    "/explain/{task_id}",
    summary="Get LLM explanation task status/result",
)
async def get_explain(task_id: str) -> Dict[str, Any]:
    """Poll an LLM explanation job."""
    try:
        task_result = celery_app.AsyncResult(task_id)

        response: Dict[str, Any] = {
            "task_id": task_id,
            "status": task_result.status,
            "result": None,
            "error": None,
            "completed_at": None,
        }

        if task_result.state == "SUCCESS":
            response["result"] = task_result.result
            response["completed_at"] = datetime.utcnow().isoformat()

        elif task_result.state == "FAILURE":
            response["error"] = str(task_result.info)
            response["completed_at"] = datetime.utcnow().isoformat()

        elif task_result.state == "PROGRESS":
            # Optional: task may publish progress in `info`.
            info = task_result.info
            if isinstance(info, dict):
                response["progress"] = info

        return response

    except Exception as e:
        logger.error(f"Failed to get LLM task status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")


@router.get(
    "/info",
    summary="Get LLM runtime capability info",
)
async def llm_info() -> Dict[str, Any]:
    """Lightweight info endpoint to answer: is the LLM using GPU or CPU?

    Note: This reports capability/config hints without forcing a model load.
    Actual runtime compute is most reliably known once the LLM has been loaded.
    """
    info: Dict[str, Any] = {
        "llama_cpp_installed": False,
        "llama_cpp_version": None,
        "supports_gpu_offload": None,
        "compute_hint": "unknown",
    }

    try:
        import importlib.metadata as md
        info["llama_cpp_version"] = md.version("llama-cpp-python")
        info["llama_cpp_installed"] = True
    except Exception:
        pass

    try:
        from llama_cpp import llama_cpp as _llama_cpp
        fn = getattr(_llama_cpp, "llama_supports_gpu_offload", None)
        if callable(fn):
            info["supports_gpu_offload"] = bool(fn())
            info["compute_hint"] = "gpu" if info["supports_gpu_offload"] else "cpu"
    except Exception:
        # If we can't import llama_cpp, keep unknown.
        pass

    return info
