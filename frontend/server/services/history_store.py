from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis.asyncio as redis

from config import settings
from services.audit_csv import append_event


REDIS_DB = 3  # dedicated DB for dashboard runtime state (snapshots + runs)

# Keep this bounded so it can't grow forever on a long-running machine.
MAX_SNAPSHOTS_PER_MACHINE = 60 * 60 * 24 // 5  # 24h at 5s cadence ~= 17,280
MAX_RUNS_PER_MACHINE = 24 * 60 * 60 // 150    # 24h at 150s cadence ~= 576

SNAPSHOT_TTL_SECONDS = 60 * 60 * 24 * 2  # 48h
RUN_TTL_SECONDS = 60 * 60 * 24 * 7       # 7 days


_redis_client: Optional[redis.Redis] = None


def _utc_now_stamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _snapshots_key(machine_id: str) -> str:
    return f"pm:snapshots:{machine_id}"


def _runs_key(machine_id: str) -> str:
    return f"pm:runs:{machine_id}"


def _run_key(run_id: str) -> str:
    return f"pm:run:{run_id}"


def _run_by_stamp_key(machine_id: str, data_stamp: str) -> str:
    # data_stamp contains ':' which is fine for redis keys.
    return f"pm:run_by_stamp:{machine_id}:{data_stamp}"


async def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
        )
    return _redis_client


@dataclass
class SnapshotItem:
    machine_id: str
    data_stamp: str
    sensor_data: Dict[str, Any]


@dataclass
class RunRecord:
    run_id: str
    machine_id: str
    data_stamp: str
    created_at: str
    sensor_data: Dict[str, Any]
    predictions: Dict[str, Any]
    llm: Dict[str, Any]
    run_type: str = "explanation"  # "prediction" or "explanation"


def _compact_predictions_for_storage(predictions: Dict[str, Any]) -> Dict[str, Any]:
    """Make predictions safe/fast to store and serve in run history.

    Some predictors (notably time-series) can produce very large arrays.
    Persisting them in every run record makes `/api/ml/runs/{run_id}` heavy
    and can cause client-side request timeouts.
    """
    if not isinstance(predictions, dict):
        return {}

    out = dict(predictions)
    ts = out.get("timeseries")
    if isinstance(ts, dict):
        ts_compact = dict(ts)
        # Drop large forecast arrays; keep summary + metadata.
        ts_compact.pop("forecasts", None)
        out["timeseries"] = ts_compact
    return out


async def add_snapshot(machine_id: str, sensor_data: Dict[str, Any], data_stamp: Optional[str] = None) -> SnapshotItem:
    stamp = (data_stamp or _utc_now_stamp()).strip() or _utc_now_stamp()
    payload = {
        "machine_id": machine_id,
        "data_stamp": stamp,
        "sensor_data": sensor_data or {},
    }

    r = await _get_redis()
    key = _snapshots_key(machine_id)
    # Avoid duplicate rows when the caller polls multiple times within the same data_stamp.
    # If the latest snapshot has the same stamp, overwrite it instead of pushing a new entry.
    try:
        latest_raw = await r.lindex(key, 0)
        if latest_raw:
            latest_obj = json.loads(latest_raw)
            latest_stamp = str(latest_obj.get("data_stamp") or "").strip()
            if latest_stamp == stamp:
                pipe = r.pipeline()
                pipe.lset(key, 0, json.dumps(payload, default=str))
                pipe.expire(key, SNAPSHOT_TTL_SECONDS)
                await pipe.execute()
                return SnapshotItem(machine_id=machine_id, data_stamp=stamp, sensor_data=sensor_data or {})
    except Exception:
        # Best-effort only; fall back to pushing.
        pass

    pipe = r.pipeline()
    pipe.lpush(key, json.dumps(payload, default=str))
    pipe.ltrim(key, 0, MAX_SNAPSHOTS_PER_MACHINE - 1)
    pipe.expire(key, SNAPSHOT_TTL_SECONDS)
    await pipe.execute()

    # Permanent audit log (best-effort)
    append_event(
        record_type="snapshot",
        machine_id=machine_id,
        data_stamp=stamp,
        payload={"sensor_data": sensor_data or {}},
    )

    return SnapshotItem(machine_id=machine_id, data_stamp=stamp, sensor_data=sensor_data or {})


async def get_latest_snapshot(machine_id: str) -> Optional[SnapshotItem]:
    r = await _get_redis()
    raw = await r.lindex(_snapshots_key(machine_id), 0)
    if not raw:
        return None
    obj = json.loads(raw)
    return SnapshotItem(
        machine_id=str(obj.get("machine_id") or machine_id),
        data_stamp=str(obj.get("data_stamp") or ""),
        sensor_data=obj.get("sensor_data") or {},
    )


async def list_snapshots(machine_id: str, limit: int = 500) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit or 500), 5000))

    r = await _get_redis()
    raws = await r.lrange(_snapshots_key(machine_id), 0, limit - 1)
    snapshots: List[Dict[str, Any]] = []
    pipe = r.pipeline()
    stamps: List[str] = []
    seen: set[str] = set()
    for raw in raws:
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        stamp = str(obj.get("data_stamp") or "").strip()
        if not stamp:
            continue
        if stamp in seen:
            continue
        seen.add(stamp)
        stamps.append(stamp)
        pipe.get(_run_by_stamp_key(machine_id, stamp))
        snapshots.append(
            {
                "machine_id": machine_id,
                "data_stamp": stamp,
                "sensor_data": obj.get("sensor_data") or {},
            }
        )

    run_ids: List[Optional[str]] = []
    if stamps:
        run_ids = await pipe.execute()

    for idx, run_id in enumerate(run_ids):
        snapshots[idx]["run_id"] = run_id
        snapshots[idx]["has_run"] = bool(run_id)
        snapshots[idx]["run_type"] = None  # will be populated below if run exists

    # Best-effort: mark whether an LLM explanation exists for each snapshot's run.
    # Also capture run_type for UI display.
    details_pipe = r.pipeline()
    details_map: List[int] = []
    for idx, run_id in enumerate(run_ids):
        if run_id:
            details_pipe.get(_run_key(str(run_id)))
            details_map.append(idx)

    details: List[Optional[str]] = []
    if details_map:
        details = await details_pipe.execute()

    for idx in range(len(snapshots)):
        snapshots[idx]["has_explanation"] = False

    for pos, raw in enumerate(details):
        snap_idx = details_map[pos]
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        
        # Extract run_type
        snapshots[snap_idx]["run_type"] = obj.get("run_type", "explanation")
        
        llm = obj.get("llm") or {}
        has_summary = False
        if isinstance(llm, dict):
            for v in llm.values():
                if not isinstance(v, dict):
                    continue
                summary = v.get("summary")
                if isinstance(summary, str) and summary.strip():
                    has_summary = True
                    break
        snapshots[snap_idx]["has_explanation"] = has_summary

    return snapshots


async def get_run_id_for_stamp(machine_id: str, data_stamp: str) -> Optional[str]:
    stamp = (data_stamp or "").strip()
    if not stamp:
        return None
    r = await _get_redis()
    val = await r.get(_run_by_stamp_key(machine_id, stamp))
    return val or None


async def create_run(machine_id: str, snapshot: SnapshotItem, predictions: Dict[str, Any], run_type: str = "explanation") -> RunRecord:
    """
    Create a run record.
    
    run_type: "prediction" for ML-only (auto), "explanation" for full with LLM
    """
    run_id = str(uuid.uuid4())
    created_at = _utc_now_stamp()
    run_type = run_type if run_type in ("prediction", "explanation") else "explanation"

    compact_predictions = _compact_predictions_for_storage(predictions or {})
    record: Dict[str, Any] = {
        "run_id": run_id,
        "machine_id": machine_id,
        "data_stamp": snapshot.data_stamp,
        "created_at": created_at,
        "sensor_data": snapshot.sensor_data or {},
        "predictions": compact_predictions,
        "llm": {},
        "run_type": run_type,
    }

    r = await _get_redis()
    pipe = r.pipeline()
    pipe.set(_run_key(run_id), json.dumps(record, default=str), ex=RUN_TTL_SECONDS)
    pipe.lpush(_runs_key(machine_id), run_id)
    pipe.ltrim(_runs_key(machine_id), 0, MAX_RUNS_PER_MACHINE - 1)
    pipe.expire(_runs_key(machine_id), RUN_TTL_SECONDS)
    pipe.set(_run_by_stamp_key(machine_id, snapshot.data_stamp), run_id, ex=RUN_TTL_SECONDS)
    await pipe.execute()

    # Permanent audit log (best-effort)
    payload_predictions = predictions or {}
    try:
        # Guard against extremely large payloads
        raw_len = len(json.dumps(payload_predictions, default=str))
        if raw_len > 2_000_000:
            payload_predictions = compact_predictions
    except Exception:
        payload_predictions = compact_predictions

    append_event(
        record_type="run_created",
        machine_id=machine_id,
        data_stamp=snapshot.data_stamp,
        run_id=run_id,
        payload={
            "sensor_data": snapshot.sensor_data or {},
            "predictions": payload_predictions,
            "run_type": run_type,
        },
    )

    return RunRecord(
        run_id=run_id,
        machine_id=machine_id,
        data_stamp=snapshot.data_stamp,
        created_at=created_at,
        sensor_data=snapshot.sensor_data or {},
        predictions=compact_predictions,
        llm={},
        run_type=run_type,
    )


async def list_runs(machine_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit or 200), 2000))

    r = await _get_redis()
    run_ids = await r.lrange(_runs_key(machine_id), 0, limit - 1)
    if not run_ids:
        return []

    pipe = r.pipeline()
    for rid in run_ids:
        pipe.get(_run_key(rid))
    raws = await pipe.execute()

    out: List[Dict[str, Any]] = []
    for raw in raws:
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        out.append(
            {
                "run_id": obj.get("run_id"),
                "machine_id": obj.get("machine_id"),
                "data_stamp": obj.get("data_stamp"),
                "created_at": obj.get("created_at"),
                "run_type": obj.get("run_type", "explanation"),
                "has_details": True,
            }
        )
    return out


async def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    r = await _get_redis()
    raw = await r.get(_run_key(run_id))
    if not raw:
        return None
    obj = json.loads(raw)
    # Serve a compact shape even for older stored runs.
    preds = obj.get("predictions")
    if isinstance(preds, dict) and isinstance(preds.get("timeseries"), dict):
        preds["timeseries"].pop("forecasts", None)
    return obj


async def update_run_llm(run_id: str, use_case: str, summary: str, compute: str = "unknown", task_id: str = "") -> None:
    uc = (use_case or "").strip().lower() or "auto"
    r = await _get_redis()
    raw = await r.get(_run_key(run_id))
    if not raw:
        return
    obj = json.loads(raw)
    llm = obj.get("llm") or {}
    llm[uc] = {
        "summary": summary,
        "compute": compute,
        "task_id": (task_id or "").strip(),
        "updated_at": _utc_now_stamp(),
    }
    obj["llm"] = llm
    # If an LLM summary exists, treat this run as an "explanation" run for UI labeling.
    obj["run_type"] = "explanation"
    await r.set(_run_key(run_id), json.dumps(obj, default=str), ex=RUN_TTL_SECONDS)

    append_event(
        record_type="llm_update",
        machine_id=str(obj.get("machine_id") or ""),
        data_stamp=str(obj.get("data_stamp") or ""),
        run_id=str(run_id),
        task_id=(task_id or "").strip(),
        payload={"use_case": uc, "compute": compute, "summary": summary},
    )


async def create_run_prediction_only(machine_id: str, snapshot: SnapshotItem, predictions: Dict[str, Any]) -> RunRecord:
    """
    Convenience wrapper for creating a prediction-only run (no LLM).
    Used by auto_prediction_service for automatic ML runs every Nth data point.
    """
    return await create_run(machine_id, snapshot, predictions, run_type="prediction")


async def clear_machine_history(machine_id: str) -> Dict[str, int]:
    """Delete all stored snapshots and runs for a machine.

    This is used by the dashboard's "Delete All Prediction History" button.
    It clears:
    - snapshot list
    - run id list
    - run details records
    - stamp -> run_id mappings

    Returns counts (best-effort) of keys deleted.
    """
    mid = (machine_id or "").strip()
    if not mid:
        return {"deleted": 0}

    r = await _get_redis()

    # Gather stamps from snapshots so we can delete stamp->run mappings without SCAN.
    stamps: List[str] = []
    try:
        raws = await r.lrange(_snapshots_key(mid), 0, -1)
        for raw in raws:
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            stamp = str(obj.get("data_stamp") or "").strip()
            if stamp:
                stamps.append(stamp)
    except Exception:
        stamps = []

    # Gather run ids so we can delete run detail keys.
    run_ids: List[str] = []
    try:
        run_ids = [str(x) for x in (await r.lrange(_runs_key(mid), 0, -1)) if x]
    except Exception:
        run_ids = []

    pipe = r.pipeline()
    for stamp in stamps:
        pipe.delete(_run_by_stamp_key(mid, stamp))
    for rid in run_ids:
        pipe.delete(_run_key(rid))

    pipe.delete(_snapshots_key(mid))
    pipe.delete(_runs_key(mid))

    deleted_counts = await pipe.execute()
    deleted = 0
    for v in deleted_counts:
        try:
            deleted += int(v or 0)
        except Exception:
            continue

    return {
        "deleted": deleted,
        "stamps": len(stamps),
        "runs": len(run_ids),
    }
