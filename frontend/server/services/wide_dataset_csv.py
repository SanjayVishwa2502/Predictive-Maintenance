from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from config import settings


_BASE_HEADER = [
    "logged_at_utc",
    "record_type",
    "machine_id",
    "session_id",
    "client_id",
    "data_stamp",
    "run_id",
    "task_id",
    # Common prediction fields
    "classification_failure_type",
    "classification_failure_probability",
    "classification_confidence",
    "rul_hours",
    "rul_days",
    "rul_urgency",
    "rul_confidence",
    "anomaly_is_anomaly",
    "anomaly_score",
    "anomaly_method",
    "timeseries_forecast_summary",
    "timeseries_confidence",
    # LLM
    "llm_summary",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _audit_dir() -> Path:
    configured = getattr(settings, "AUDIT_CSV_DIR", "reports/audit_csv")
    return Path(configured)


def _safe_part(val: str) -> str:
    raw = (val or "").strip() or "unknown"
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in raw)


def _file_path(machine_id: str, session_id: str) -> Path:
    mid = _safe_part(machine_id)
    sid = _safe_part(session_id or "session")
    return _audit_dir() / f"wide_dataset_{mid}_{sid}.csv"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _lock_file(f) -> None:
    try:
        if os.name == "nt":
            import msvcrt  # type: ignore

            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl  # type: ignore

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    except Exception:
        return


def _unlock_file(f) -> None:
    try:
        if os.name == "nt":
            import msvcrt  # type: ignore

            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl  # type: ignore

            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception:
        return


def _read_existing_header(path: Path) -> Optional[list[str]]:
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            return next(reader, None)
    except Exception:
        return None


def _write_row(path: Path, header: list[str], row: Dict[str, Any]) -> None:
    _ensure_parent(path)
    file_exists = path.exists()
    try:
        with open(path, "a", encoding="utf-8", newline="") as f:
            _lock_file(f)
            try:
                writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            finally:
                _unlock_file(f)
    except Exception:
        # Must never break predictions.
        return


def append_snapshot_row(
    *,
    machine_id: str,
    session_id: str,
    data_stamp: str,
    sensor_data: Dict[str, Any],
    client_id: str = "",
    run_id: str = "",
) -> None:
    """Append one sensor datapoint row with one column per sensor.

    This is intended to be Excel-friendly: each sensor is its own CSV column.
    """
    sid = (session_id or "").strip() or "session"
    path = _file_path(machine_id, sid)

    existing_header = _read_existing_header(path)
    if existing_header:
        header = list(existing_header)
    else:
        sensor_cols = sorted({str(k) for k in (sensor_data or {}).keys()})
        header = _BASE_HEADER + sensor_cols

    row: Dict[str, Any] = {
        "logged_at_utc": _utc_now_iso(),
        "record_type": "snapshot",
        "machine_id": (machine_id or "").strip(),
        "session_id": sid,
        "client_id": (client_id or "").strip(),
        "data_stamp": (data_stamp or "").strip(),
        "run_id": (run_id or "").strip(),
        "task_id": "",
        "llm_summary": "",
    }

    for k, v in (sensor_data or {}).items():
        row[str(k)] = v

    _write_row(path, header, row)


def append_run_row(
    *,
    machine_id: str,
    session_id: str,
    data_stamp: str,
    run_id: str,
    sensor_data: Dict[str, Any],
    predictions: Dict[str, Any],
    client_id: str = "",
) -> None:
    sid = (session_id or "").strip() or "session"
    path = _file_path(machine_id, sid)

    existing_header = _read_existing_header(path)
    if existing_header:
        header = list(existing_header)
    else:
        sensor_cols = sorted({str(k) for k in (sensor_data or {}).keys()})
        header = _BASE_HEADER + sensor_cols

    cls = (predictions or {}).get("classification") or {}
    rul = (predictions or {}).get("rul") or {}
    ano = (predictions or {}).get("anomaly") or {}
    ts = (predictions or {}).get("timeseries") or {}

    row: Dict[str, Any] = {
        "logged_at_utc": _utc_now_iso(),
        "record_type": "run_created",
        "machine_id": (machine_id or "").strip(),
        "session_id": sid,
        "client_id": (client_id or "").strip(),
        "data_stamp": (data_stamp or "").strip(),
        "run_id": (run_id or "").strip(),
        "task_id": "",
        "classification_failure_type": cls.get("failure_type"),
        "classification_failure_probability": cls.get("failure_probability"),
        "classification_confidence": cls.get("confidence"),
        "rul_hours": rul.get("rul_hours"),
        "rul_days": rul.get("rul_days"),
        "rul_urgency": rul.get("urgency"),
        "rul_confidence": rul.get("confidence"),
        "anomaly_is_anomaly": ano.get("is_anomaly"),
        "anomaly_score": ano.get("anomaly_score"),
        "anomaly_method": ano.get("detection_method"),
        "timeseries_forecast_summary": ts.get("forecast_summary"),
        "timeseries_confidence": ts.get("confidence"),
        "llm_summary": "",
    }

    for k, v in (sensor_data or {}).items():
        row[str(k)] = v

    _write_row(path, header, row)


def append_llm_row(
    *,
    machine_id: str,
    session_id: str,
    data_stamp: str,
    run_id: str,
    task_id: str,
    llm_summary: str,
    client_id: str = "",
) -> None:
    sid = (session_id or "").strip() or "session"
    path = _file_path(machine_id, sid)

    existing_header = _read_existing_header(path)
    header = list(existing_header) if existing_header else list(_BASE_HEADER)

    row: Dict[str, Any] = {
        "logged_at_utc": _utc_now_iso(),
        "record_type": "llm_update",
        "machine_id": (machine_id or "").strip(),
        "session_id": sid,
        "client_id": (client_id or "").strip(),
        "data_stamp": (data_stamp or "").strip(),
        "run_id": (run_id or "").strip(),
        "task_id": (task_id or "").strip(),
        "llm_summary": (llm_summary or "").strip(),
    }

    _write_row(path, header, row)


def latest_wide_dataset_path(machine_id: str, session_id: str) -> Optional[Path]:
    """Return the expected wide dataset path for (machine_id, session_id) if it exists."""
    sid = (session_id or "").strip()
    if not sid:
        return None
    p = _file_path(machine_id, sid)
    return p if p.exists() else None
