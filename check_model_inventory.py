"""Model inventory + integrity checker.

Scans:
- ml_models/models/<model_type>/<machine_id>/
- ml_models/reports/performance_metrics/<machine_id>_*_report.json

Reports per-machine per-model-type status: missing | available | corrupted
Optionally deletes corrupted artifacts.

Usage examples:
  python check_model_inventory.py
  python check_model_inventory.py --json inventory.json
  python check_model_inventory.py --machine chiller_trane_rtac_001
  python check_model_inventory.py --delete-corrupted
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_ROOT = PROJECT_ROOT / "ml_models" / "models"
REPORTS_ROOT = PROJECT_ROOT / "ml_models" / "reports" / "performance_metrics"

MODEL_TYPES = ("classification", "regression", "anomaly", "timeseries")


def report_path(model_type: str, machine_id: str) -> Path:
    if model_type == "classification":
        return REPORTS_ROOT / f"{machine_id}_classification_report.json"
    if model_type == "regression":
        return REPORTS_ROOT / f"{machine_id}_regression_report.json"
    if model_type == "anomaly":
        return REPORTS_ROOT / f"{machine_id}_comprehensive_anomaly_report.json"
    if model_type == "timeseries":
        return REPORTS_ROOT / f"{machine_id}_timeseries_report.json"
    raise ValueError(f"Unknown model_type: {model_type}")


def model_dir(model_type: str, machine_id: str) -> Path:
    return MODELS_ROOT / model_type / machine_id


def dir_non_empty(path: Path) -> bool:
    try:
        return path.is_dir() and any(path.iterdir())
    except Exception:
        return path.is_dir()


@dataclass
class Status:
    status: str  # missing | available | corrupted
    model_dir: str
    report_path: str
    issues: List[str]


def compute_status(model_type: str, machine_id: str) -> Status:
    mdir = model_dir(model_type, machine_id)
    rpath = report_path(model_type, machine_id)

    issues: List[str] = []

    if not mdir.exists():
        return Status(
            status="missing",
            model_dir=str(mdir),
            report_path=str(rpath),
            issues=["model_dir_missing"],
        )

    if not dir_non_empty(mdir):
        issues.append("model_dir_empty")

    if not rpath.exists():
        issues.append("report_missing")
    else:
        try:
            with open(rpath, "r", encoding="utf-8") as f:
                json.load(f)
        except Exception:
            issues.append("report_invalid_json")

    if not issues:
        return Status(
            status="available",
            model_dir=str(mdir),
            report_path=str(rpath),
            issues=[],
        )

    return Status(
        status="corrupted",
        model_dir=str(mdir),
        report_path=str(rpath),
        issues=issues,
    )


def discover_machine_ids() -> List[str]:
    ids = set()
    for model_type in MODEL_TYPES:
        type_dir = MODELS_ROOT / model_type
        if not type_dir.exists():
            continue
        try:
            for entry in type_dir.iterdir():
                if entry.is_dir():
                    ids.add(entry.name)
        except Exception:
            continue
    return sorted(ids)


def delete_artifacts(model_type: str, machine_id: str) -> Tuple[bool, bool]:
    mdir = model_dir(model_type, machine_id)
    rpath = report_path(model_type, machine_id)

    deleted_model_dir = False
    deleted_report_file = False

    if mdir.exists() and mdir.is_dir():
        shutil.rmtree(mdir)
        deleted_model_dir = True

    if rpath.exists() and rpath.is_file():
        rpath.unlink()
        deleted_report_file = True

    return deleted_model_dir, deleted_report_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Check ML model artifact availability/integrity")
    parser.add_argument("--machine", help="Only check this machine_id", default=None)
    parser.add_argument("--json", dest="json_path", help="Write JSON report to this path", default=None)
    parser.add_argument(
        "--delete-corrupted",
        action="store_true",
        help="Delete corrupted model directories + report files (DANGEROUS)",
    )

    args = parser.parse_args()

    machine_ids = [args.machine] if args.machine else discover_machine_ids()

    inventory: Dict[str, Dict[str, Dict]] = {}

    any_corrupted = False
    any_missing = False

    for machine_id in machine_ids:
        inventory[machine_id] = {}
        for model_type in MODEL_TYPES:
            st = compute_status(model_type, machine_id)
            inventory[machine_id][model_type] = {
                "status": st.status,
                "model_dir": st.model_dir,
                "report_path": st.report_path,
                "issues": st.issues,
            }

            if st.status == "corrupted":
                any_corrupted = True
                if args.delete_corrupted:
                    deleted_model_dir, deleted_report_file = delete_artifacts(model_type, machine_id)
                    inventory[machine_id][model_type]["deleted_model_dir"] = deleted_model_dir
                    inventory[machine_id][model_type]["deleted_report_file"] = deleted_report_file

            if st.status == "missing":
                any_missing = True

    # Console output (compact)
    for machine_id in machine_ids:
        print(f"\n{machine_id}")
        for model_type in MODEL_TYPES:
            entry = inventory[machine_id][model_type]
            status = entry["status"]
            issues = entry.get("issues") or []
            issues_txt = f" ({', '.join(issues)})" if issues else ""
            print(f"  - {model_type}: {status}{issues_txt}")

    if args.json_path:
        out_path = Path(args.json_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"machines": inventory}, f, indent=2)
        print(f"\nWrote JSON report: {out_path}")

    # Non-zero exit if anything wrong (useful for CI/scripts)
    if any_corrupted:
        return 2
    if any_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
