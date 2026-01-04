"""
Auto Prediction Service

Runs ML predictions automatically every Nth data point from the logger.
No LLM explanations - just fast ML inference (classification, anomaly, timeseries).

This service is triggered by the status endpoint whenever it receives new printer data.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Per-machine counters: how many data points since last prediction
_data_point_counters: Dict[str, int] = {}

# Track which machines have had their models preloaded
_preloaded_machines: set = set()

# Interval: run predictions every Nth data point
PREDICTION_INTERVAL = 5


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def should_run_prediction(machine_id: str) -> bool:
    """Check if we should run a prediction for this machine (every Nth data point)."""
    mid = (machine_id or "").strip()
    if not mid:
        return False
    
    count = _data_point_counters.get(mid, 0) + 1
    _data_point_counters[mid] = count
    
    if count >= PREDICTION_INTERVAL:
        _data_point_counters[mid] = 0
        return True
    
    return False


def get_counter(machine_id: str) -> int:
    """Get current counter value for a machine."""
    return _data_point_counters.get((machine_id or "").strip(), 0)


def reset_counter(machine_id: str) -> None:
    """Reset counter for a machine (e.g., when machine is deselected)."""
    mid = (machine_id or "").strip()
    if mid in _data_point_counters:
        del _data_point_counters[mid]


async def run_prediction_only(
    machine_id: str,
    sensor_data: Dict[str, Any],
    data_stamp: str,
) -> Optional[Dict[str, Any]]:
    """
    Run ML predictions only (no LLM explanation).
    
    Returns a dict with run_id, predictions, etc. or None if prediction failed.
    """
    from services.ml_manager import ml_manager
    from services.history_store import SnapshotItem, create_run_prediction_only
    
    mid = (machine_id or "").strip()
    if not mid:
        return None
    
    if not sensor_data:
        logger.warning(f"No sensor data for prediction: {mid}")
        return None
    
    try:
        # Create snapshot for this prediction
        snapshot = SnapshotItem(
            machine_id=mid,
            data_stamp=data_stamp,
            sensor_data=sensor_data,
        )
        
        # Run all available ML predictions
        predictions: Dict[str, Any] = {}
        
        # Classification
        try:
            cls = ml_manager.predict_classification(machine_id=mid, sensor_data=sensor_data)
            predictions["classification"] = asdict(cls)
        except Exception as e:
            predictions["classification"] = {"error": str(e)}
        
        # RUL (skip if no model)
        try:
            regression_path = ml_manager.models_dir / "regression" / mid
            if not ml_manager._has_model_artifacts(regression_path):
                predictions["rul"] = {
                    "skipped": True,
                    "reason": "no_rul_model",
                    "message": "RUL: n/a (no RUL model for this machine)",
                }
            else:
                rul = ml_manager.predict_rul(machine_id=mid, sensor_data=sensor_data)
                predictions["rul"] = asdict(rul)
        except Exception as e:
            predictions["rul"] = {"error": str(e)}
        
        # Anomaly
        try:
            anomaly_input = sensor_data
            # Printers benefit from lightweight derived features (deltas, short-window stats)
            # and from using recent history rather than a single point.
            if mid.startswith("printer_"):
                from services.history_store import list_snapshots

                snaps = await list_snapshots(mid, limit=120)
                # snaps are newest-first; build oldest-first series
                series = list(reversed(snaps))
                # Append current point if needed
                if not series or str(series[-1].get("data_stamp") or "") != str(data_stamp):
                    series.append({"data_stamp": data_stamp, "sensor_data": sensor_data})

                def _get_float(s: dict, k: str) -> Optional[float]:
                    try:
                        v = (s.get("sensor_data") or {}).get(k)
                        return float(v) if v is not None else None
                    except Exception:
                        return None

                bed_vals = [v for v in (_get_float(s, "bed_temp_c") for s in series) if v is not None]
                noz_vals = [v for v in (_get_float(s, "nozzle_temp_c") for s in series) if v is not None]

                # 60-second-ish window (best-effort, based on sample count)
                bed_tail = bed_vals[-12:] if len(bed_vals) >= 12 else bed_vals
                noz_tail = noz_vals[-12:] if len(noz_vals) >= 12 else noz_vals

                bed = float(sensor_data.get("bed_temp_c")) if sensor_data.get("bed_temp_c") is not None else None
                noz = float(sensor_data.get("nozzle_temp_c")) if sensor_data.get("nozzle_temp_c") is not None else None
                bed_prev = bed_vals[-2] if len(bed_vals) >= 2 else None
                noz_prev = noz_vals[-2] if len(noz_vals) >= 2 else None

                anomaly_input = dict(sensor_data)
                if bed is not None and bed_prev is not None:
                    anomaly_input["bed_temp_c_delta"] = float(bed - bed_prev)
                if noz is not None and noz_prev is not None:
                    anomaly_input["nozzle_temp_c_delta"] = float(noz - noz_prev)
                if bed_tail:
                    anomaly_input["bed_temp_c_mean_60s"] = float(sum(bed_tail) / len(bed_tail))
                    try:
                        import numpy as _np
                        anomaly_input["bed_temp_c_std_60s"] = float(_np.std(_np.array(bed_tail, dtype=float)))
                    except Exception:
                        pass
                if noz_tail:
                    anomaly_input["nozzle_temp_c_mean_60s"] = float(sum(noz_tail) / len(noz_tail))
                    try:
                        import numpy as _np
                        anomaly_input["nozzle_temp_c_std_60s"] = float(_np.std(_np.array(noz_tail, dtype=float)))
                    except Exception:
                        pass

            ano = ml_manager.predict_anomaly(machine_id=mid, sensor_data=anomaly_input)
            predictions["anomaly"] = asdict(ano)
        except Exception as e:
            predictions["anomaly"] = {"error": str(e)}

        # Time-series
        try:
            ts_input = sensor_data
            # For timeseries, use a rolling window of recent snapshots so the model
            # can align to recent behavior and produce non-constant summaries.
            from services.history_store import list_snapshots
            snaps = await list_snapshots(mid, limit=500)
            series = list(reversed(snaps))
            # Convert to a list of {timestamp, <sensor...>} rows.
            rows = []
            for s in series:
                stamp = str(s.get("data_stamp") or "").strip()
                sd = s.get("sensor_data") or {}
                if not stamp or not isinstance(sd, dict) or not sd:
                    continue
                row = {"timestamp": stamp}
                row.update(sd)
                rows.append(row)
            # Add current point (ensures latest is included)
            if data_stamp:
                row = {"timestamp": data_stamp}
                row.update(sensor_data)
                rows.append(row)
            ts_input = rows if rows else sensor_data

            ts = ml_manager.predict_timeseries(machine_id=mid, sensor_data=ts_input)
            predictions["timeseries"] = asdict(ts)
        except Exception as e:
            predictions["timeseries"] = {"error": str(e)}
        
        # Store as prediction-only run
        run = await create_run_prediction_only(mid, snapshot, predictions)
        
        logger.info(f"[OK] Prediction-only run created for {mid}: {run.run_id}")
        
        return {
            "run_id": run.run_id,
            "machine_id": mid,
            "data_stamp": run.data_stamp,
            "created_at": run.created_at,
            "run_type": "prediction",
            "predictions": predictions,
        }
        
    except Exception as e:
        logger.error(f"Prediction-only run failed for {mid}: {e}")
        return None


def preload_models_for_machine(machine_id: str) -> bool:
    """
    Preload ML models for a machine when it's selected.
    This warms up the model cache for faster predictions.
    
    Only runs once per machine until reset.
    Returns True if models were loaded successfully.
    """
    from services.ml_manager import ml_manager
    
    mid = (machine_id or "").strip()
    if not mid:
        return False
    
    # Skip if already preloaded this session
    if mid in _preloaded_machines:
        return True
    
    try:
        # The IntegratedPredictionSystem caches models after first use.
        # We do a quick "warm-up" by checking model availability.
        classification_path = ml_manager.models_dir / "classification" / mid
        anomaly_path = ml_manager.models_dir / "anomaly" / mid
        timeseries_path = ml_manager.models_dir / "timeseries" / mid
        
        has_cls = ml_manager._has_model_artifacts(classification_path)
        has_ano = ml_manager._has_model_artifacts(anomaly_path)
        has_ts = ml_manager._has_model_artifacts(timeseries_path)
        
        logger.info(
            f"[PRELOAD] Models for {mid}: "
            f"classification={has_cls}, anomaly={has_ano}, timeseries={has_ts}"
        )
        
        # Mark as preloaded
        _preloaded_machines.add(mid)
        
        # Reset counter when machine is selected
        reset_counter(mid)
        
        return has_cls or has_ano or has_ts
        
    except Exception as e:
        logger.error(f"Failed to preload models for {mid}: {e}")
        return False
