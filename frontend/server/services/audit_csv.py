from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from config import settings


# Session identifier (time-based) to separate datasets across server restarts.
# This prevents mixing previous sessions into the same CSV file.
_SESSION_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _audit_enabled() -> bool:
    # Safe default: enabled
    return bool(getattr(settings, "AUDIT_CSV_ENABLED", True))


def _audit_dir() -> Path:
    # Default to workspace-level reports directory.
    configured = getattr(settings, "AUDIT_CSV_DIR", "reports/audit_csv")
    return Path(configured)


def _safe_machine_id(machine_id: str) -> str:
    mid = (machine_id or "unknown").strip() or "unknown"
    # Keep filenames safe-ish
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in mid)


def _file_path(machine_id: str) -> Path:
    return _audit_dir() / f"prediction_history_{_safe_machine_id(machine_id)}_{_SESSION_ID}.csv"


def list_datasets(machine_id: str) -> list[dict[str, Any]]:
    """List available CSV datasets for this machine (newest first)."""
    mid = _safe_machine_id(machine_id)
    base = _audit_dir()
    try:
        paths = sorted(base.glob(f"prediction_history_{mid}_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for p in paths:
        try:
            st = p.stat()
            out.append(
                {
                    "filename": p.name,
                    "path": str(p),
                    "size_bytes": int(st.st_size),
                    "modified_at": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                }
            )
        except Exception:
            continue
    return out


def latest_dataset_path(machine_id: str) -> Optional[Path]:
    datasets = list_datasets(machine_id)
    if not datasets:
        return None
    try:
        return Path(datasets[0]["path"])
    except Exception:
        return None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _lock_file(f):
    """Best-effort cross-process advisory lock.

    On Windows, uses msvcrt.locking.
    On Unix, uses fcntl.flock.
    """
    try:
        if os.name == "nt":
            import msvcrt  # type: ignore

            # Lock 1 byte (file must be opened in a mode that allows it)
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl  # type: ignore

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    except Exception:
        return


def _unlock_file(f):
    try:
        if os.name == "nt":
            import msvcrt  # type: ignore

            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl  # type: ignore

            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception:
        return


_HEADER = [
    "logged_at_utc",
    "record_type",
    "machine_id",
    "data_stamp",
    "run_id",
    "task_id",
    "payload_json",
]


def append_event(
    *,
    record_type: str,
    machine_id: str,
    data_stamp: Optional[str] = None,
    run_id: Optional[str] = None,
    task_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    if not _audit_enabled():
        return

    path = _file_path(machine_id)
    _ensure_parent(path)

    row = {
        "logged_at_utc": _utc_now_iso(),
        "record_type": (record_type or "unknown").strip(),
        "machine_id": (machine_id or "").strip(),
        "data_stamp": (data_stamp or "").strip(),
        "run_id": (run_id or "").strip(),
        "task_id": (task_id or "").strip(),
        "payload_json": json.dumps(payload or {}, ensure_ascii=False, default=str),
    }

    try:
        file_exists = path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            _lock_file(f)
            try:
                writer = csv.DictWriter(f, fieldnames=_HEADER)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            finally:
                _unlock_file(f)
    except Exception:
        # Audit logging must never break predictions.
        return
