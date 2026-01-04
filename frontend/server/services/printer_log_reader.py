"""Printer log reader

Reads latest temperatures from clean CSV logs produced by 3Dprinterdata/datalogger.py.

This is intentionally lightweight (no pandas) and is used by the ML dashboard
"machine status" endpoint to provide real streamed values instead of validation
simulation.
"""

from __future__ import annotations

import os
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class PrinterReading:
    timestamp: str
    bed_temp_c: float
    nozzle_temp_c: float

    def to_sensors(self) -> Dict[str, float]:
        return {
            "bed_temp_c": float(self.bed_temp_c),
            "nozzle_temp_c": float(self.nozzle_temp_c),
        }


def _resolve_log_dir() -> Path:
    raw = str(os.environ.get("PM_PRINTER_LOG_DIR", "")).strip()
    if not raw:
        return PROJECT_ROOT / "3Dprinterdata"
    p = Path(raw)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _candidate_files(machine_id: str) -> list[Path]:
    log_dir = _resolve_log_dir()
    candidates: list[Path] = []

    mid = (machine_id or "").strip()
    if mid:
        candidates.append(log_dir / f"{mid}_temps_clean.csv")

    # Backward-compatible single-printer default
    candidates.append(log_dir / "ender3_temps_clean.csv")

    return candidates


def read_latest_printer_reading(machine_id: str) -> Optional[PrinterReading]:
    """Read the last valid row from the machine's clean temp CSV.

    Returns None if no suitable file exists or no valid data rows exist.
    """
    for path in _candidate_files(machine_id):
        if not path.exists() or not path.is_file():
            continue

        try:
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)

            # Expect header + data. Walk backwards to find last non-empty data row.
            for row in reversed(rows[1:]):
                if not row or len(row) < 3:
                    continue
                ts = (row[0] or "").strip()
                try:
                    bed = float(row[1])
                    noz = float(row[2])
                except Exception:
                    continue
                if not ts:
                    continue
                return PrinterReading(timestamp=ts, bed_temp_c=bed, nozzle_temp_c=noz)
        except Exception:
            continue

    return None


def reading_to_status_fields(reading: PrinterReading) -> dict:
    """Convert a PrinterReading into fields compatible with MachineStatusResponse."""
    # Parse the timestamp from CSV (assumed to be local time without timezone)
    # For API consistency, keep the local timestamp as-is with Z suffix
    # The key is that the SAME timestamp appears in both data_stamp and for run lookups
    try:
        local_dt = datetime.fromisoformat(reading.timestamp)
        last_update = local_dt
        # Keep the original timestamp for data_stamp (just add Z for format consistency)
        stamp = reading.timestamp
        if stamp and not stamp.endswith("Z"):
            stamp = f"{stamp}Z"
        # Calculate data age in seconds
        data_age_seconds = (datetime.now() - local_dt).total_seconds()
    except Exception:
        stamp = reading.timestamp
        if stamp and not stamp.endswith("Z"):
            stamp = f"{stamp}Z"
        last_update = datetime.now()
        data_age_seconds = 0.0  # Unknown age, assume fresh

    return {
        "latest_sensors": reading.to_sensors(),
        "last_update": last_update,
        "data_stamp": stamp,
        "sensor_count": 2,
        "is_running": True,
        "data_age_seconds": max(0.0, data_age_seconds),
    }
