"""Extract a clean temperature dataset from an Ender 3 OctoPrint MQTT/event log.

Input format (repo root): ender3_continuous_log.csv
Columns: Timestamp,Topic,Payload

Output: a "normal" dataset with ONLY:
- timestamp
- bed_temp
- nozzle_temp

Notes:
- The raw log records bed/nozzle temps on separate lines.
- This script emits a unified time series (one row per temp record) using
  forward-fill of the latest known bed/nozzle temps.
- Rows are only emitted after BOTH bed and nozzle have been seen at least once.

Usage (from repo root):
  python data_ingestion/scripts/extract_ender3_temps.py

Optional args:
  --input  path/to/ender3_continuous_log.csv
    --output path/to/ender3_temps_clean.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


BED_TOPIC = "octoPrint/temperature/bed"
NOZZLE_TOPIC = "octoPrint/temperature/tool0"


def _find_project_root(start: Path) -> Path:
    """Find repo root by looking for known top-level folders/files."""
    start = start.resolve()
    for parent in [start, *start.parents]:
        if (parent / "data_ingestion").exists() and (parent / "frontend").exists():
            return parent
    return start


def _parse_timestamp(ts: str) -> str:
    """Normalize timestamp to ISO 8601 (seconds precision)."""
    ts = (ts or "").strip()
    # Example: 2025-12-23 15:24:06
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        return dt.isoformat(timespec="seconds")
    except Exception:
        return ts


def _safe_json(payload: str) -> Optional[Dict[str, Any]]:
    payload = (payload or "").strip()
    if not payload or payload[0] != "{":
        return None
    try:
        obj = json.loads(payload)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_actual(payload_obj: Dict[str, Any]) -> Optional[float]:
    try:
        val = payload_obj.get("actual", None)
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


@dataclass
class TempState:
    bed_temp: Optional[float] = None
    bed_line: Optional[int] = None
    nozzle_temp: Optional[float] = None
    nozzle_line: Optional[int] = None

    def ready(self) -> bool:
        return self.bed_temp is not None and self.nozzle_temp is not None


def extract_clean_dataset(input_csv: Path, output_csv: Path) -> Tuple[int, int]:
    """Extract cleaned dataset.

    Returns:
        (rows_written, temp_records_seen)
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    state = TempState()
    rows_written = 0
    temp_records_seen = 0

    with input_csv.open("r", newline="", encoding="utf-8") as f_in, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as f_out:
        reader = csv.reader(f_in)
        header = next(reader, None)
        if not header:
            raise ValueError("Input CSV is empty")

        try:
            ts_idx = header.index("Timestamp")
            topic_idx = header.index("Topic")
            payload_idx = header.index("Payload")
        except ValueError as exc:
            raise ValueError(f"Unexpected CSV header. Expected Timestamp,Topic,Payload. Got: {header}") from exc

        writer = csv.DictWriter(
            f_out,
            fieldnames=[
                "timestamp",
                "bed_temp",
                "nozzle_temp",
            ],
        )
        writer.writeheader()

        for file_line_no, row in enumerate(reader, start=2):
            if not row or len(row) <= max(ts_idx, topic_idx, payload_idx):
                continue

            topic = (row[topic_idx] or "").strip()
            if topic not in (BED_TOPIC, NOZZLE_TOPIC):
                continue

            payload = row[payload_idx]
            payload_obj = _safe_json(payload)
            if payload_obj is None:
                continue

            actual = _extract_actual(payload_obj)
            if actual is None:
                continue

            temp_records_seen += 1

            if topic == BED_TOPIC:
                state.bed_temp = actual
                state.bed_line = file_line_no
            elif topic == NOZZLE_TOPIC:
                state.nozzle_temp = actual
                state.nozzle_line = file_line_no

            # Emit only once both sensors have been observed at least once.
            if not state.ready():
                continue

            timestamp_iso = _parse_timestamp(row[ts_idx])

            writer.writerow(
                {
                    "timestamp": timestamp_iso,
                    "bed_temp": f"{state.bed_temp:.3f}" if state.bed_temp is not None else "",
                    "nozzle_temp": f"{state.nozzle_temp:.3f}" if state.nozzle_temp is not None else "",
                }
            )
            rows_written += 1

    return rows_written, temp_records_seen


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Ender 3 bed/nozzle temps into a clean dataset")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to ender3_continuous_log.csv (default: repo root / ender3_continuous_log.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: data_ingestion/processed/ender3_temps_clean.csv)",
    )
    args = parser.parse_args()

    root = _find_project_root(Path(__file__).parent)

    input_csv = Path(args.input) if args.input else (root / "ender3_continuous_log.csv")
    output_csv = Path(args.output) if args.output else (root / "data_ingestion" / "processed" / "ender3_temps_clean.csv")

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv.as_posix()}")

    rows_written, temp_seen = extract_clean_dataset(input_csv=input_csv, output_csv=output_csv)

    print("=" * 80)
    print("ENDER3 TEMPERATURE EXTRACTION")
    print("=" * 80)
    print(f"Input:  {input_csv.as_posix()}")
    print(f"Output: {output_csv.as_posix()}")
    print(f"Temp records seen (bed+nozzle): {temp_seen:,}")
    print(f"Rows written (unified dataset): {rows_written:,}")


if __name__ == "__main__":
    main()
