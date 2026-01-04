"""Printer anomaly training (Ender 3)

Trains a lightweight anomaly detector for printer temperature telemetry.

Why this exists:
- The generic anomaly pipeline expects many industrial features (power_rating_kw,
  rated_speed_rpm, etc.). For printers, those are not present in streaming data,
  so inference pads missing values with zeros -> outputs look "hardcoded".

This script trains an IsolationForest + a simple statistical Z-score detector on
printer temps (+ deltas), and saves artifacts in the same format expected by
`ml_models/scripts/inference/predict_anomaly.py`.

Outputs:
- ml_models/models/anomaly/<machine_id>/all_detectors.pkl
- ml_models/models/anomaly/<machine_id>/preprocessing.pkl
- ml_models/models/anomaly/<machine_id>/features.json
- ml_models/reports/performance_metrics/<machine_id>_comprehensive_anomaly_report.json
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_printer_csv(machine_id: str) -> Optional[Path]:
    candidates = [
        PROJECT_ROOT / "3Dprinterdata" / f"{machine_id}_temps_clean.csv",
        PROJECT_ROOT / "3Dprinterdata" / "ender3_temps_clean.csv",
        PROJECT_ROOT / "3Dprinterdata" / "printer_creality_ender3_001_temps_clean.csv",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def _load_printer_df(machine_id: str) -> pd.DataFrame:
    path = _resolve_printer_csv(machine_id)
    if path is None:
        raise FileNotFoundError(
            "Printer temp CSV not found. Expected one of: "
            f"3Dprinterdata/{machine_id}_temps_clean.csv, 3Dprinterdata/ender3_temps_clean.csv"
        )

    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"CSV missing 'timestamp' column: {path}")

    # Normalize column names if needed
    rename = {}
    if "bed_temp_c" not in df.columns:
        for c in df.columns:
            if c.lower() in {"bed", "bed_temp", "bed_temp_c"}:
                rename[c] = "bed_temp_c"
    if "nozzle_temp_c" not in df.columns:
        for c in df.columns:
            if c.lower() in {"nozzle", "tool0", "nozzle_temp", "nozzle_temp_c"}:
                rename[c] = "nozzle_temp_c"
    if rename:
        df = df.rename(columns=rename)

    needed = ["bed_temp_c", "nozzle_temp_c"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column '{col}': {path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df["bed_temp_c"] = pd.to_numeric(df["bed_temp_c"], errors="coerce")
    df["nozzle_temp_c"] = pd.to_numeric(df["nozzle_temp_c"], errors="coerce")
    df = df.dropna(subset=needed)

    return df


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["bed_temp_c"] = df["bed_temp_c"].astype(float)
    out["nozzle_temp_c"] = df["nozzle_temp_c"].astype(float)

    # Deltas help catch sudden jumps/runaway heating.
    out["bed_temp_c_delta"] = out["bed_temp_c"].diff().fillna(0.0)
    out["nozzle_temp_c_delta"] = out["nozzle_temp_c"].diff().fillna(0.0)

    # Short rolling stats (approx ~1 minute if data is ~5s cadence; best-effort)
    w = 12
    out["bed_temp_c_mean_60s"] = out["bed_temp_c"].rolling(w, min_periods=1).mean()
    out["bed_temp_c_std_60s"] = out["bed_temp_c"].rolling(w, min_periods=1).std().fillna(0.0)
    out["nozzle_temp_c_mean_60s"] = out["nozzle_temp_c"].rolling(w, min_periods=1).mean()
    out["nozzle_temp_c_std_60s"] = out["nozzle_temp_c"].rolling(w, min_periods=1).std().fillna(0.0)

    return out


def train_printer_anomaly(machine_id: str, contamination: float = 0.02) -> dict:
    df = _load_printer_df(machine_id)
    feat = _build_features(df)

    feature_cols = list(feat.columns)
    X = feat.values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=200,
        contamination=float(contamination),
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(Xs)

    # Simple z-score detector stored as "zscore"; inference treats it specially.
    # We store the scaled data mean/std in a small shim dict.
    z = {
        "mean": Xs.mean(axis=0).tolist(),
        "std": (Xs.std(axis=0) + 1e-10).tolist(),
        "threshold": 3.0,
    }

    save_dir = PROJECT_ROOT / "ml_models" / "models" / "anomaly" / machine_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # all_detectors.pkl in the format expected by AnomalyPredictor
    joblib.dump(
        {
            "detectors": {
                "isolation_forest": iso,
                # keep key name exactly 'zscore' so inference applies its fast path
                "zscore": z,
            },
            "scaler": scaler,
        },
        save_dir / "all_detectors.pkl",
    )

    joblib.dump(
        {
            "scaler": scaler,
            "feature_cols": feature_cols,
        },
        save_dir / "preprocessing.pkl",
    )

    with open(save_dir / "features.json", "w", encoding="utf-8") as f:
        json.dump({"features": feature_cols}, f, indent=2)

    # Minimal report to satisfy the dashboard inventory expectations
    report = {
        "machine_id": machine_id,
        "task_type": "printer_anomaly_detection",
        "timestamp": datetime.now().isoformat(),
        "data_info": {
            "n_samples": int(len(df)),
            "time_start": str(df["timestamp"].min()),
            "time_end": str(df["timestamp"].max()),
            "n_features": int(len(feature_cols)),
            "features": feature_cols,
        },
        "model": {
            "type": "isolation_forest + zscore",
            "contamination": float(contamination),
        },
        "notes": [
            "Trained on printer temperature telemetry only.",
            "Uses deltas and short-window rolling stats to reduce 'hardcoded' outputs.",
        ],
    }

    report_path = PROJECT_ROOT / "ml_models" / "reports" / "performance_metrics" / f"{machine_id}_comprehensive_anomaly_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {"model_dir": str(save_dir), "report_path": str(report_path), "report": report}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train printer anomaly model")
    parser.add_argument("--machine_id", required=True)
    parser.add_argument("--contamination", type=float, default=0.02)
    args = parser.parse_args()

    result = train_printer_anomaly(args.machine_id, contamination=args.contamination)
    print(json.dumps(result, indent=2))
