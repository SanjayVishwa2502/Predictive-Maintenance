"""\
LLM Celery Tasks

Runs long LLM inference (CPU or GPU) outside of request handlers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import time

import json
from datetime import datetime

import redis as sync_redis

from celery_app import celery_app
from tasks import LoggingTask
from config import settings
from services.audit_csv import append_event
from services.wide_dataset_csv import append_llm_row

from pathlib import Path


def _update_run_llm_best_effort(run_id: Any, use_case: str, summary: str, compute: str, task_id: str) -> None:
    """Best-effort persistence of LLM output into the run record (Redis DB 3)."""
    rid = str(run_id or "").strip()
    if not rid:
        return
    try:
        r = sync_redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=3,
            decode_responses=True,
        )
        key = f"pm:run:{rid}"
        raw = r.get(key)
        if not raw:
            return
        obj = json.loads(raw)
        llm = obj.get("llm") or {}
        uc = (use_case or "").strip().lower() or "auto"
        llm[uc] = {
            "summary": summary,
            "compute": compute or "unknown",
            "task_id": (task_id or "").strip(),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        obj["llm"] = llm
        # Keep existing TTL (best-effort).
        ttl = r.ttl(key)
        if ttl is None or ttl <= 0:
            r.set(key, json.dumps(obj, default=str))
        else:
            r.set(key, json.dumps(obj, default=str), ex=int(ttl))
    except Exception:
        return


_EXPLAINER = None


def _get_explainer():
    """Lazy-load the MLExplainer once per worker process."""
    global _EXPLAINER
    if _EXPLAINER is not None:
        return _EXPLAINER

    # Importing these can be expensive; keep it lazy.
    from LLM.api.explainer import MLExplainer

    _EXPLAINER = MLExplainer()
    return _EXPLAINER


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _coerce_predictions(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build a unified `predictions` object from any legacy/single-use-case payload.

    The combined LLM prompt expects a single object containing the available
    predictions (classification/rul/anomaly/timeseries). Some callers (e.g. the
    modal) send only a subset of fields.
    """
    base: Dict[str, Any] = {}

    raw = payload.get("predictions")
    if isinstance(raw, dict) and raw:
        # If it already looks like our unified structure, keep it.
        base.update(raw)

    # Classification-like fields
    cls: Dict[str, Any] = {}
    if payload.get("health_state") is not None:
        cls["health_state"] = payload.get("health_state")
    if payload.get("predicted_failure_type") is not None:
        cls["failure_type"] = payload.get("predicted_failure_type")
    fp = _safe_float(payload.get("failure_probability"))
    if fp is not None:
        cls["failure_probability"] = fp
    conf = _safe_float(payload.get("confidence"))
    if conf is not None:
        cls["confidence"] = conf
    if cls:
        base.setdefault("classification", cls)

    # RUL
    rul_hours = _safe_float(payload.get("rul_hours"))
    if rul_hours is not None:
        base.setdefault("rul", {"rul_hours": rul_hours, "confidence": conf})

    # Anomaly
    anomaly_score = _safe_float(payload.get("anomaly_score"))
    abnormal_sensors = payload.get("abnormal_sensors")
    detection_method = payload.get("detection_method")
    if anomaly_score is not None or abnormal_sensors or detection_method is not None:
        base.setdefault(
            "anomaly",
            {
                "anomaly_score": float(anomaly_score or 0.0),
                "abnormal_sensors": abnormal_sensors or {},
                "detection_method": str(detection_method or "unknown"),
            },
        )

    # Time-series / forecast
    forecast_summary = payload.get("forecast_summary")
    if forecast_summary is not None:
        base.setdefault("timeseries", {"forecast_summary": str(forecast_summary), "confidence": conf})

    return base


def _publish_llm_event(client_id: str, payload: Dict[str, Any]) -> None:
    """Best-effort pub/sub event for UI push updates (Redis DB 2)."""
    cid = (client_id or "").strip()
    if not cid:
        return
    try:
        r = sync_redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=2,
            decode_responses=True,
        )
        channel = f"llm:events:{cid[:128]}"
        r.publish(channel, json.dumps(payload, default=str))
    except Exception:
        # Never fail the task because pub/sub broadcast failed.
        return


def _baseline_ranges_from_profile(machine_id: str, sensor_data: Dict[str, Any]) -> str:
    """Best-effort: derive baseline normal ranges from the GAN machine profile.

    For printers (e.g. Ender3) nozzle temps like ~210C are normal; without this context
    the LLM tends to apply generic industrial thresholds.
    """
    mid = (machine_id or "").strip()
    if not mid:
        return "unavailable"

    try:
        # Note: Settings paths (e.g. GAN_ROOT_PATH) are configured relative to the
        # frontend/server directory (where config.py and .env live), not repo root.
        server_root = Path(__file__).resolve().parents[1]

        gan_root_cfg = Path(str(settings.GAN_ROOT_PATH or "GAN"))
        gan_root = gan_root_cfg if gan_root_cfg.is_absolute() else (server_root / gan_root_cfg).resolve()

        profile_path = gan_root / "data" / "real_machines" / "profiles" / f"{mid}.json"
        if not profile_path.exists():
            return "unavailable"

        obj = json.loads(profile_path.read_text(encoding="utf-8"))
        baseline = obj.get("baseline_normal_operation") or {}

        # Flatten baseline blocks into a {sensor_name: spec_dict} map.
        sensor_specs: Dict[str, Any] = {}
        for _block_name, block in (baseline or {}).items():
            if isinstance(block, dict):
                for sensor_name, spec in block.items():
                    if isinstance(spec, dict):
                        sensor_specs[str(sensor_name)] = spec

        if not sensor_specs:
            return "unavailable"

        # Only include sensors actually present in current data.
        current_keys = {str(k) for k in (sensor_data or {}).keys()}

        lines = []
        for name, spec in sensor_specs.items():
            if name not in current_keys:
                continue
            unit = str(spec.get("unit") or "")
            # Avoid encoding artifacts ("Â°C") and keep prompts ASCII-friendly.
            unit = unit.replace("Â°C", "C").replace("°C", "C").replace("Â°F", "F").replace("°F", "F")
            parts = []
            for key in ("min", "typical", "max", "alarm", "trip"):
                if key in spec and spec.get(key) is not None:
                    parts.append(f"{key}={spec.get(key)}")
            if not parts:
                continue
            suffix = f" {unit}" if unit else ""
            lines.append(f"{name}: {', '.join(parts)}{suffix}")

        return "\n".join(lines) if lines else "unavailable"
    except Exception:
        return "unavailable"


@celery_app.task(bind=True, base=LoggingTask, name="tasks.llm_tasks.generate_explanation")
def generate_explanation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate an explanation from prediction context.

    Payload is passed from the frontend modal and may contain:
    - machine_id
    - health_state
    - confidence
    - failure_probability
    - predicted_failure_type
    - rul_hours
    - sensor_data
    """
    started = time.time()

    machine_id = str(payload.get("machine_id") or "")
    client_id = str(payload.get("client_id") or "")
    session_id = str(payload.get("session_id") or "")
    run_id = payload.get("run_id")
    celery_task_id = str(getattr(getattr(self, "request", None), "id", "") or "")
    requested_use_case = str(payload.get("use_case") or "").strip().lower() or "auto"
    # Major workflow change: we only emit a single combined LLM output.
    # Any incoming use_case is treated as combined for consistency and speed.
    use_case = "combined"
    data_stamp = payload.get("data_stamp")
    health_state = payload.get("health_state")
    confidence = _safe_float(payload.get("confidence"))
    failure_probability = _safe_float(payload.get("failure_probability"))
    predicted_failure_type = payload.get("predicted_failure_type")
    rul_hours = _safe_float(payload.get("rul_hours"))
    sensor_data = payload.get("sensor_data") or {}

    anomaly_score = _safe_float(payload.get("anomaly_score"))
    abnormal_sensors = payload.get("abnormal_sensors") or {}
    detection_method = payload.get("detection_method")

    forecast_summary = payload.get("forecast_summary")
    predictions = _coerce_predictions(payload)

    # Best-effort progress update (optional)
    try:
        self.update_state(
            state="PROGRESS",
            meta={"stage": "initializing", "machine_id": machine_id, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))},
        )
    except Exception:
        pass

    # Fallback-safe defaults
    result: Dict[str, Any] = {
        "summary": "",
        "risk_factors": [],
        "recommendations": [],
        "technical_details": None,
        "confidence_analysis": None,
        "use_case": use_case,
        "data_stamp": data_stamp,
        "run_id": run_id,
        "machine_id": machine_id,
        "compute": "unknown",
    }

    # If the LLM stack isn't installed/configured, return a clear message instead of crashing.
    try:
        explainer = _get_explainer()

        # Best-effort compute/device metadata (llama-cpp / CPU vs GPU)
        try:
            llm = getattr(explainer, "llm", None)
            if llm is not None and hasattr(llm, "get_runtime_info"):
                info = llm.get_runtime_info() or {}
                if isinstance(info, dict):
                    result["compute"] = info.get("compute") or result["compute"]
        except Exception:
            pass

        if True:
            try:
                self.update_state(state="PROGRESS", meta={"stage": "generating", "kind": "combined"})
            except Exception:
                pass
            llm_out = explainer.explain_combined_run(
                machine_id=machine_id,
                predictions=predictions,
                sensor_data=sensor_data,
                baseline_ranges=_baseline_ranges_from_profile(machine_id, sensor_data),
            )

        explanation_text = str(llm_out.get("explanation") or "")
        # Some stacks occasionally return mojibake for degree symbols (e.g. "Â°C").
        # Normalize output to keep the dashboard readable.
        explanation_text = (
            explanation_text.replace("Â°C", " C")
            .replace("°C", " C")
            .replace("Â°F", " F")
            .replace("°F", " F")
        )
        sources = llm_out.get("sources") or []
        out_conf = llm_out.get("confidence")

        elapsed = time.time() - started

        # Map to the frontend modal's expected shape.
        result["summary"] = explanation_text
        result["risk_factors"] = []
        result["recommendations"] = []
        result["technical_details"] = (
            f"Machine: {machine_id}\n"
            f"Use case: {use_case}\n"
            f"Requested use case: {requested_use_case}\n"
            f"Data stamp: {data_stamp}\n"
            f"Compute: {result.get('compute')}\n"
            f"Sources: {sources}\n"
            f"Elapsed: {elapsed:.1f}s"
        )
        if out_conf is not None:
            result["confidence_analysis"] = f"Model-reported confidence: {out_conf}"

        # Push event for the dashboard (no polling) when available.
        _publish_llm_event(
            client_id,
            {
                "type": "llm_explanation",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                **result,
            },
        )

        # Persist into run details if this task is associated with a stored run.
        # Persist under a stable key for combined runs.
        _update_run_llm_best_effort(
            run_id,
            "combined",
            result.get("summary") or "",
            result.get("compute") or "unknown",
            celery_task_id,
        )

        # Wide CSV dataset capture (Excel-friendly, per session). Best-effort only.
        try:
            if session_id.strip():
                append_llm_row(
                    machine_id=machine_id,
                    session_id=session_id,
                    data_stamp=str(data_stamp or ""),
                    run_id=str(run_id or ""),
                    task_id=celery_task_id,
                    llm_summary=str(result.get("summary") or ""),
                    client_id=client_id,
                )
        except Exception:
            pass

        append_event(
            record_type="llm_task_succeeded",
            machine_id=machine_id,
            data_stamp=str(data_stamp or ""),
            run_id=str(run_id or ""),
            task_id=celery_task_id,
            payload={"use_case": use_case, "compute": result.get("compute"), "summary": result.get("summary")},
        )

        return result

    except Exception as e:
        elapsed = time.time() - started
        result["summary"] = (
            "LLM explanation is currently unavailable on this server.\n\n"
            f"Reason: {type(e).__name__}: {e}\n\n"
            "If you're running on CPU, loading the LLM may take a long time and requires the LLM dependencies (RAG index, sentence-transformers, faiss, llama engine)."
        )
        result["technical_details"] = f"Elapsed: {elapsed:.1f}s"

        _publish_llm_event(
            client_id,
            {
                "type": "llm_explanation",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                **result,
            },
        )
        _update_run_llm_best_effort(
            run_id,
            "combined",
            result.get("summary") or "",
            result.get("compute") or "unknown",
            celery_task_id,
        )

        # Wide CSV dataset capture (Excel-friendly, per session). Best-effort only.
        try:
            if session_id.strip():
                append_llm_row(
                    machine_id=machine_id,
                    session_id=session_id,
                    data_stamp=str(data_stamp or ""),
                    run_id=str(run_id or ""),
                    task_id=celery_task_id,
                    llm_summary=str(result.get("summary") or ""),
                    client_id=client_id,
                )
        except Exception:
            pass
        append_event(
            record_type="llm_task_failed",
            machine_id=machine_id,
            data_stamp=str(data_stamp or ""),
            run_id=str(run_id or ""),
            task_id=celery_task_id,
            payload={"use_case": use_case, "compute": result.get("compute"), "summary": result.get("summary")},
        )
        return result
