from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from celery_app import celery_app

from services.ml_manager import ml_manager
from services.sensor_simulator import get_simulator
from services.history_store import SnapshotItem, add_snapshot, create_run, get_latest_snapshot
from services.wide_dataset_csv import append_run_row
from services.printer_log_reader import read_latest_printer_reading
from services.llm_busy import set_llm_busy

logger = logging.getLogger(__name__)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class AutoPredictionRunner:
    """Backend-side scheduler for running prediction+LLM cycles.

    The dashboard (or a script) registers a machine+interval, and the backend
    creates a run every N seconds using the latest stored snapshot.
    """

    def __init__(self):
        self._tasks: Dict[str, asyncio.Task] = {}
        self._intervals: Dict[str, int] = {}
        self._last_run: Dict[str, Optional[str]] = {}
        # When auto-runs are initiated from the dashboard, capture the browser client_id
        # so LLM task completion events publish to the same WS channel the UI listens on.
        self._client_ids: Dict[str, str] = {}

    def status(self, machine_id: str) -> Dict[str, Any]:
        mid = (machine_id or "").strip()
        running = mid in self._tasks and not self._tasks[mid].done()
        return {
            "machine_id": mid,
            "running": running,
            "interval_seconds": self._intervals.get(mid),
            "last_run_at": self._last_run.get(mid),
            "client_id": self._client_ids.get(mid),
        }

    async def start(self, machine_id: str, interval_seconds: int = 150, client_id: Optional[str] = None) -> Dict[str, Any]:
        mid = (machine_id or "").strip()
        if not mid:
            raise ValueError("machine_id is required")

        cid = (client_id or "").strip()
        if cid:
            self._client_ids[mid] = cid[:128]

        interval = max(30, int(interval_seconds or 150))

        # Restart if already running.
        await self.stop(mid)

        self._intervals[mid] = interval
        self._tasks[mid] = asyncio.create_task(self._loop(mid))

        return self.status(mid)

    async def stop(self, machine_id: str) -> Dict[str, Any]:
        mid = (machine_id or "").strip()
        task = self._tasks.get(mid)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._tasks.pop(mid, None)
        self._client_ids.pop(mid, None)
        return self.status(mid)

    async def run_once(
        self,
        machine_id: str,
        client_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        mid = (machine_id or "").strip()
        if not mid:
            raise ValueError("machine_id is required")

        cid = (client_id or "").strip()
        if cid:
            self._client_ids[mid] = cid[:128]

        # Always record a fresh snapshot for this run.
        # If the UI isn't polling, the last snapshot may not change; repeated runs would reuse
        # the same data_stamp and can cause stamp->run_id mapping churn (making the UI appear
        # like LLM is perpetually "pending" for that stamp).
        prior_snapshot = await get_latest_snapshot(mid)
        
        # PRIORITY 1: Try to read REAL-TIME sensor data from the printer's CSV log
        # This is the datalogger output that contains current/live readings
        fresh_sensor_data: Optional[Dict[str, Any]] = None
        try:
            printer_reading = read_latest_printer_reading(mid)
            if printer_reading is not None:
                fresh_sensor_data = printer_reading.to_sensors()
                logger.info(f"[run_once] Using real-time printer data for {mid}: {fresh_sensor_data}")
        except Exception as e:
            logger.warning(f"[run_once] Failed to read real-time printer data for {mid}: {e}")
        
        # PRIORITY 2: Fall back to simulator if no real-time data available
        if not fresh_sensor_data:
            simulator = get_simulator()
            if not simulator.is_running(mid):
                try:
                    simulator.start_simulation(mid)
                except Exception:
                    pass
            try:
                fresh_sensor_data = simulator.get_current_readings(mid) if simulator.is_running(mid) else None
                if fresh_sensor_data:
                    logger.info(f"[run_once] Using simulator data for {mid}: {fresh_sensor_data}")
            except Exception:
                fresh_sensor_data = None

        # PRIORITY 3: Fall back to prior snapshot if all else fails
        if not fresh_sensor_data:
            fresh_sensor_data = (prior_snapshot.sensor_data if prior_snapshot else {}) or {}
            logger.info(f"[run_once] Using prior snapshot data for {mid}: {fresh_sensor_data}")

        snapshot = await add_snapshot(mid, sensor_data=fresh_sensor_data, data_stamp=_utc_stamp())

        predictions = await self._predict_all(mid, snapshot)
        run = await create_run(mid, snapshot, predictions)

        # Wide CSV dataset capture (Excel-friendly, per session). Best-effort only.
        try:
            sid = (session_id or "").strip()
            if sid:
                append_run_row(
                    machine_id=mid,
                    session_id=sid,
                    data_stamp=run.data_stamp,
                    run_id=run.run_id,
                    sensor_data=snapshot.sensor_data or {},
                    predictions=predictions or {},
                    client_id=(cid or ""),
                )
        except Exception:
            pass

        # Enqueue LLM explanations (async).
        # Prefer a dashboard/browser client_id so WS push updates reach the UI.
        client_id = self._client_ids.get(mid) or f"auto:{mid}"[:128]
        base_payload = {
            "machine_id": mid,
            "client_id": client_id,
            "session_id": (session_id or "").strip(),
            "run_id": run.run_id,
            "data_stamp": run.data_stamp,
            "sensor_data": snapshot.sensor_data or {},
        }

        payload = {**base_payload, "use_case": "combined", "predictions": predictions}
        task = celery_app.send_task(
            "tasks.llm_tasks.generate_explanation",
            args=[payload],
            queue="llm",
        )

        # Mark machine as LLM-busy so frontend shows loading state
        await set_llm_busy(mid, task_id=task.id, client_id=client_id)

        self._last_run[mid] = _utc_stamp()
        return {
            "run_id": run.run_id,
            "machine_id": mid,
            "data_stamp": run.data_stamp,
            "created_at": run.created_at,
        }

    async def _loop(self, machine_id: str) -> None:
        mid = machine_id
        interval = self._intervals.get(mid, 150)
        logger.info(f"AutoPredictionRunner started for {mid} (interval={interval}s)")
        try:
            while True:
                try:
                    await self.run_once(mid)
                except Exception as e:
                    logger.error(f"AutoPredictionRunner run failed for {mid}: {e}")
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info(f"AutoPredictionRunner stopped for {mid}")
            raise

    async def _predict_all(self, machine_id: str, snapshot: SnapshotItem) -> Dict[str, Any]:
        sensor_data = snapshot.sensor_data or {}

        out: Dict[str, Any] = {}

        # Classification
        try:
            cls = ml_manager.predict_classification(machine_id=machine_id, sensor_data=sensor_data)
            out["classification"] = asdict(cls)
        except Exception as e:
            out["classification"] = {"error": str(e)}

        # RUL
        try:
            regression_path = ml_manager.models_dir / "regression" / machine_id
            if not ml_manager._has_model_artifacts(regression_path):
                out["rul"] = {
                    "skipped": True,
                    "reason": "no_rul_model",
                    "message": "RUL: n/a (no RUL model for this machine)",
                }
            else:
                rul = ml_manager.predict_rul(machine_id=machine_id, sensor_data=sensor_data)
                out["rul"] = asdict(rul)
        except Exception as e:
            out["rul"] = {"error": str(e)}

        # Anomaly
        try:
            ano = ml_manager.predict_anomaly(machine_id=machine_id, sensor_data=sensor_data)
            out["anomaly"] = asdict(ano)
        except Exception as e:
            out["anomaly"] = {"error": str(e)}

        # Time-series
        try:
            ts = ml_manager.predict_timeseries(machine_id=machine_id, sensor_data=sensor_data)
            out["timeseries"] = asdict(ts)
        except Exception as e:
            out["timeseries"] = {"error": str(e)}

        return out

    def _llm_payloads_from_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        payloads: Dict[str, Dict[str, Any]] = {}

        cls = predictions.get("classification") or {}
        if isinstance(cls, dict) and not cls.get("error"):
            payloads["classification"] = {
                "predicted_failure_type": cls.get("failure_type"),
                "failure_probability": cls.get("failure_probability"),
                "confidence": cls.get("confidence"),
            }

        rul = predictions.get("rul") or {}
        if isinstance(rul, dict) and not rul.get("error"):
            payloads["rul"] = {
                "rul_hours": rul.get("rul_hours"),
                "confidence": rul.get("confidence"),
            }

        ano = predictions.get("anomaly") or {}
        if isinstance(ano, dict) and not ano.get("error"):
            payloads["anomaly"] = {
                "anomaly_score": ano.get("anomaly_score"),
                "abnormal_sensors": ano.get("abnormal_sensors") or {},
                "detection_method": ano.get("detection_method"),
            }

        ts = predictions.get("timeseries") or {}
        if isinstance(ts, dict) and not ts.get("error"):
            payloads["timeseries"] = {
                "forecast_summary": ts.get("forecast_summary"),
                "confidence": ts.get("confidence"),
            }

        return payloads


auto_prediction_runner = AutoPredictionRunner()
