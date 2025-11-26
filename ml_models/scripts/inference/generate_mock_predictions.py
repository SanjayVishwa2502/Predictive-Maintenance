"""
MOCK PREDICTION GENERATOR FOR TESTING
=====================================
Generates realistic test predictions without loading actual models.
Perfect for testing LLM integration pipeline.
"""

import json
import random
from pathlib import Path
from datetime import datetime, timedelta

# Priority machines
MACHINES = [
    "motor_siemens_1la7_001",
    "motor_abb_m3bp_002", 
    "pump_grundfos_cr3_004",
    "compressor_atlas_copco_ga30_001",
    "cooling_tower_bac_vti_018"
]

# Realistic failure types per machine
FAILURE_TYPES = {
    "motor": ["bearing_wear", "overheating", "electrical_fault", "normal"],
    "pump": ["seal_leak", "cavitation", "bearing_wear", "normal"],
    "compressor": ["valve_failure", "overheating", "oil_contamination", "normal"],
    "cooling_tower": ["fan_imbalance", "scale_buildup", "normal"]
}

# Sensor names per machine type
SENSORS = {
    "motor": {
        "bearing_de_temp_C": (45, 85),
        "bearing_nde_temp_C": (45, 85),
        "winding_temp_C": (50, 95),
        "rms_velocity_mm_s": (1, 15),
        "current_A": (20, 60)
    },
    "pump": {
        "bearing_temp_C": (40, 80),
        "motor_winding_temp_C": (45, 90),
        "overall_rms_mm_s": (1, 12),
        "current_a": (15, 50),
        "discharge_pressure_bar": (3, 8)
    },
    "compressor": {
        "discharge_air_temp_C": (50, 120),
        "oil_temp_C": (40, 90),
        "rms_velocity_mm_s": (2, 18),
        "discharge_pressure_bar": (5, 10),
        "current_A": (25, 70)
    },
    "cooling_tower": {
        "fan_bearing_temp_C": (35, 75),
        "fan_bearing_vibration_mm_s": (1, 10)
    }
}


def get_machine_type(machine_id):
    """Extract machine type from ID"""
    if "motor" in machine_id:
        return "motor"
    elif "pump" in machine_id:
        return "pump"
    elif "compressor" in machine_id:
        return "compressor"
    elif "cooling" in machine_id or "tower" in machine_id:
        return "cooling_tower"
    return "motor"


def generate_sensor_readings(machine_id, is_failing=False):
    """Generate realistic sensor readings"""
    machine_type = get_machine_type(machine_id)
    sensors = SENSORS.get(machine_type, SENSORS["motor"])
    
    readings = {}
    for sensor, (min_val, max_val) in sensors.items():
        if is_failing:
            # Elevated values for failures
            value = random.uniform(max_val * 0.8, max_val * 1.1)
        else:
            # Normal operating range
            value = random.uniform(min_val, max_val * 0.6)
        readings[sensor] = round(value, 2)
    
    return readings


def generate_classification_prediction(machine_id, sample_num):
    """Generate classification prediction"""
    machine_type = get_machine_type(machine_id)
    failure_types = FAILURE_TYPES.get(machine_type, FAILURE_TYPES["motor"])
    
    # 70% normal, 30% failure
    is_failing = random.random() < 0.3
    failure_type = random.choice(failure_types[:-1]) if is_failing else "normal"
    
    confidence = random.uniform(0.75, 0.95)
    failure_prob = 1 - confidence if failure_type == "normal" else confidence
    
    # Generate all probabilities
    all_probs = {ft: 0.0 for ft in failure_types}
    all_probs[failure_type] = confidence
    remaining = 1 - confidence
    for ft in failure_types:
        if ft != failure_type:
            all_probs[ft] = remaining / (len(failure_types) - 1)
    
    return {
        "machine_id": machine_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_type": "classification",
        "prediction": {
            "failure_probability": round(failure_prob, 4),
            "failure_type": failure_type,
            "confidence": round(confidence, 4),
            "all_probabilities": {k: round(v, 4) for k, v in all_probs.items()}
        },
        "sensor_readings": generate_sensor_readings(machine_id, is_failing),
        "model_info": {
            "model_path": f"ml_models/models/classification/{machine_id}",
            "model_type": "AutoGluon TabularPredictor",
            "best_model": "WeightedEnsemble_L2"
        }
    }


def generate_anomaly_prediction(machine_id, sample_num):
    """Generate anomaly detection prediction"""
    # 80% normal, 20% anomalous
    is_anomaly = random.random() < 0.2
    
    # Generate detector votes
    detectors = {
        "isolation_forest": random.choice([-1, 1]),
        "one_class_svm": random.choice([-1, 1]),
        "lof": random.choice([-1, 1]),
        "elliptic_envelope": random.choice([-1, 1]),
        "zscore": random.choice([-1, 1]),
        "iqr": random.choice([-1, 1]),
        "modified_zscore": random.choice([-1, 1]),
        "ensemble_voting": -1 if is_anomaly else 1
    }
    
    anomaly_votes = sum(1 for v in detectors.values() if v == -1)
    confidence = anomaly_votes / len(detectors) if is_anomaly else (8 - anomaly_votes) / 8
    
    return {
        "machine_id": machine_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_type": "anomaly_detection",
        "prediction": {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(random.uniform(0.6, 0.95) if is_anomaly else random.uniform(0.05, 0.4), 4),
            "confidence": round(confidence, 4),
            "severity": "high" if is_anomaly else "normal",
            "detector_votes": detectors,
            "consensus": f"{anomaly_votes}/8 detectors flagged anomaly"
        },
        "sensor_readings": generate_sensor_readings(machine_id, is_anomaly),
        "model_info": {
            "model_path": f"ml_models/models/anomaly/{machine_id}",
            "num_detectors": 8,
            "ensemble_method": "voting"
        }
    }


def generate_rul_prediction(machine_id, sample_num):
    """Generate RUL regression prediction"""
    # Random RUL between 24 hours and 2000 hours
    rul_hours = random.uniform(24, 2000)
    rul_days = rul_hours / 24
    
    # Determine urgency
    if rul_hours < 24:
        urgency = "critical"
    elif rul_hours < 72:
        urgency = "high"
    elif rul_hours < 168:
        urgency = "medium"
    else:
        urgency = "low"
    
    estimated_failure = datetime.utcnow() + timedelta(hours=rul_hours)
    
    return {
        "machine_id": machine_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_type": "rul_regression",
        "prediction": {
            "rul_hours": round(rul_hours, 2),
            "rul_days": round(rul_days, 2),
            "estimated_failure_date": estimated_failure.isoformat() + "Z",
            "confidence": round(random.uniform(0.75, 0.95), 4),
            "urgency": urgency,
            "maintenance_window": {
                "earliest": (datetime.utcnow() + timedelta(hours=rul_hours * 0.7)).isoformat() + "Z",
                "latest": (datetime.utcnow() + timedelta(hours=rul_hours * 0.9)).isoformat() + "Z",
                "recommended": (datetime.utcnow() + timedelta(hours=rul_hours * 0.8)).isoformat() + "Z"
            }
        },
        "sensor_readings": generate_sensor_readings(machine_id, rul_hours < 200),
        "model_info": {
            "model_path": f"ml_models/models/rul/{machine_id}",
            "model_type": "AutoGluon TabularPredictor",
            "best_model": "WeightedEnsemble_L2"
        }
    }


def generate_timeseries_prediction(machine_id, sample_num):
    """Generate time-series forecast prediction"""
    machine_type = get_machine_type(machine_id)
    sensors = SENSORS.get(machine_type, SENSORS["motor"])
    
    # Generate 24-hour forecast for each sensor
    forecasts = {}
    for sensor, (min_val, max_val) in sensors.items():
        base_value = random.uniform(min_val, max_val * 0.5)
        trend = random.uniform(-0.5, 0.5)  # Small trend
        
        yhat = []
        yhat_lower = []
        yhat_upper = []
        timestamps = []
        
        for hour in range(24):
            value = base_value + (trend * hour) + random.uniform(-2, 2)
            yhat.append(round(value, 2))
            yhat_lower.append(round(value - random.uniform(3, 5), 2))
            yhat_upper.append(round(value + random.uniform(3, 5), 2))
            timestamps.append((datetime.utcnow() + timedelta(hours=hour+1)).isoformat() + "Z")
        
        forecasts[sensor] = {
            "yhat": yhat,
            "yhat_lower": yhat_lower,
            "yhat_upper": yhat_upper,
            "ds": timestamps
        }
    
    # Check for concerning trends
    concerning_trends = []
    for sensor, forecast in forecasts.items():
        if "temp" in sensor.lower():
            if max(forecast["yhat"]) > 80:
                concerning_trends.append(f"{sensor}: Peak value {max(forecast['yhat']):.1f}°C (high)")
        elif "vibration" in sensor.lower() or "velocity" in sensor.lower():
            if max(forecast["yhat"]) > 10:
                concerning_trends.append(f"{sensor}: Peak vibration {max(forecast['yhat']):.1f} mm/s (elevated)")
    
    if not concerning_trends:
        concerning_trends = ["No concerning trends detected"]
    
    return {
        "machine_id": machine_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_type": "timeseries_forecast",
        "prediction": {
            "forecast_horizon_hours": 24,
            "forecast_start": (datetime.utcnow() + timedelta(hours=1)).isoformat() + "Z",
            "forecast_end": (datetime.utcnow() + timedelta(hours=25)).isoformat() + "Z",
            "confidence": round(random.uniform(0.75, 0.90), 4),
            "forecast_summary": "Stable sensor readings expected over next 24 hours" if not concerning_trends[0].startswith("No") else "All sensors within normal operating ranges",
            "concerning_trends": concerning_trends,
            "maintenance_window": "Optimal window: Hour 0-6 (lowest sensor activity)",
            "detailed_forecast": forecasts
        },
        "historical_data_points": 168,
        "model_info": {
            "model_path": f"ml_models/models/timeseries/{machine_id}",
            "prediction_length": 24,
            "forecasted_sensors": list(sensors.keys())
        }
    }


def main():
    """Generate all 100 mock predictions"""
    print("="*70)
    print(" MOCK PREDICTION GENERATOR - TESTING")
    print("="*70)
    print()
    print("Generating 100 realistic test predictions...")
    print(f"  - 25 Classification predictions")
    print(f"  - 25 Anomaly predictions")
    print(f"  - 25 RUL predictions")
    print(f"  - 25 Time-series forecasts")
    print()
    
    # Setup output directories
    base_path = Path(__file__).parent.parent.parent / "outputs" / "predictions"
    
    for model_type in ["classification", "anomaly", "rul", "timeseries"]:
        output_dir = base_path / model_type
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate predictions
    stats = {
        "classification": 0,
        "anomaly": 0,
        "rul": 0,
        "timeseries": 0
    }
    
    for machine_id in MACHINES:
        print(f"\nGenerating predictions for {machine_id}...")
        
        # Classification
        predictions = []
        for i in range(5):
            predictions.append(generate_classification_prediction(machine_id, i))
        
        output_file = base_path / "classification" / f"{machine_id}_predictions.json"
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        stats["classification"] += 5
        print(f"  ✓ Classification: 5 predictions")
        
        # Anomaly
        predictions = []
        for i in range(5):
            predictions.append(generate_anomaly_prediction(machine_id, i))
        
        output_file = base_path / "anomaly" / f"{machine_id}_predictions.json"
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        stats["anomaly"] += 5
        print(f"  ✓ Anomaly: 5 predictions")
        
        # RUL
        predictions = []
        for i in range(5):
            predictions.append(generate_rul_prediction(machine_id, i))
        
        output_file = base_path / "rul" / f"{machine_id}_predictions.json"
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        stats["rul"] += 5
        print(f"  ✓ RUL: 5 predictions")
        
        # TimeSeries
        predictions = []
        for i in range(5):
            predictions.append(generate_timeseries_prediction(machine_id, i))
        
        output_file = base_path / "timeseries" / f"{machine_id}_predictions.json"
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        stats["timeseries"] += 5
        print(f"  ✓ TimeSeries: 5 predictions")
    
    # Generate summary
    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_predictions": 100,
        "successful": 100,
        "failed": 0,
        "success_rate": 100.0,
        "model_types": {
            "classification": {"success": stats["classification"], "failed": 0},
            "anomaly": {"success": stats["anomaly"], "failed": 0},
            "rul": {"success": stats["rul"], "failed": 0},
            "timeseries": {"success": stats["timeseries"], "failed": 0}
        },
        "output_directories": {
            "classification": str(base_path / "classification"),
            "anomaly": str(base_path / "anomaly"),
            "rul": str(base_path / "rul"),
            "timeseries": str(base_path / "timeseries")
        },
        "note": "Mock predictions generated for testing LLM integration pipeline"
    }
    
    summary_file = base_path / "batch_generation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print()
    print("="*70)
    print(" ✓ GENERATION COMPLETE")
    print("="*70)
    print()
    print(f"Total Predictions: 100")
    print(f"  - Classification: {stats['classification']}")
    print(f"  - Anomaly: {stats['anomaly']}")
    print(f"  - RUL: {stats['rul']}")
    print(f"  - TimeSeries: {stats['timeseries']}")
    print()
    print(f"Success Rate: 100.0%")
    print()
    print(f"Output Location: {base_path}")
    print(f"Summary: {summary_file}")
    print()
    print("✓ Ready for LLM integration testing!")
    print()


if __name__ == "__main__":
    main()
