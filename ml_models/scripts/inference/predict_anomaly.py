"""
Anomaly Detection Model Inference Script
Phase 3.5.0: ML Model Inference Pipeline

Loads trained anomaly detection models and generates predictions
for unusual behavior detection in industrial machines

Industrial-grade implementation with:
- Multiple detector support (IsolationForest, OneClassSVM, LocalOutlierFactor, Z-Score)
- Ensemble scoring
- Anomaly severity classification
- JSON output format for LLM integration
"""

import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))


class AnomalyPredictor:
    """Loads and runs inference with trained anomaly detection models"""
    
    def __init__(self, machine_id: str, models_dir: Optional[Path] = None):
        """
        Initialize anomaly predictor
        
        Args:
            machine_id: Machine identifier (e.g., 'motor_siemens_1la7_001')
            models_dir: Path to models directory (optional, uses default)
        """
        self.machine_id = machine_id
        self.models_dir = models_dir or PROJECT_ROOT / "ml_models" / "models" / "anomaly"
        self.model_path = self.models_dir / machine_id
        self.detectors = {}
        self.preprocessing = None
        self.feature_names = None
        
        self._load_models()
    
    def _load_models(self):
        """Load trained anomaly detection models"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model path not found: {self.model_path}")
            
            print(f"Loading anomaly detection models for {self.machine_id}...")
            
            # Load all detectors using joblib (sklearn models)
            import joblib
            import sys
            
            # Add ml_models to path to handle any custom imports
            ml_models_path = str(Path(__file__).resolve().parents[2])
            if ml_models_path not in sys.path:
                sys.path.insert(0, ml_models_path)
            
            all_detectors_path = self.model_path / "all_detectors.pkl"
            if all_detectors_path.exists():
                model_data = joblib.load(all_detectors_path)
                
                # Check if it's the new format with dict structure
                if isinstance(model_data, dict) and 'detectors' in model_data:
                    self.detectors = model_data['detectors']
                    self.preprocessing = model_data.get('scaler')  # Use scaler as preprocessing
                    print(f"✓ Loaded {len(self.detectors)} detectors: {list(self.detectors.keys())}")
                else:
                    self.detectors = model_data
                    print(f"✓ Loaded {len(self.detectors)} detectors")
            else:
                # Load individual detectors
                detector_files = {
                    'isolation_forest': 'isolation_forest.pkl',
                    'one_class_svm': 'one_class_svm.pkl',
                    'local_outlier_factor': 'local_outlier_factor.pkl',
                    'zscore': 'zscore.pkl'
                }
                
                for name, filename in detector_files.items():
                    detector_path = self.model_path / filename
                    if detector_path.exists():
                        self.detectors[name] = joblib.load(detector_path)
                        print(f"✓ Loaded {name}")
            
            if not self.detectors:
                raise RuntimeError("No anomaly detectors found")
            
            # Load preprocessing if not already loaded
            if self.preprocessing is None:
                preprocessing_path = self.model_path / "preprocessing.pkl"
                if preprocessing_path.exists():
                    self.preprocessing = joblib.load(preprocessing_path)
                    print("✓ Loaded preprocessing pipeline")
            
            # Load feature names
            features_path = self.model_path / "features.json"
            if features_path.exists():
                with open(features_path, 'r') as f:
                    self.feature_names = json.load(f)
                print(f"✓ Loaded {len(self.feature_names)} feature names")
            
            print("✓ Anomaly detection models loaded successfully\n")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load anomaly detection models: {e}")
    
    def predict(self, sensor_data: Dict[str, float]) -> Dict:
        """
        Run anomaly detection prediction
        
        Args:
            sensor_data: Dictionary of sensor readings
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Validate input
            if not sensor_data:
                raise ValueError("sensor_data cannot be empty")
            
            # Convert to DataFrame
            df = pd.DataFrame([sensor_data])
            
            # Preprocess if preprocessing pipeline exists
            if self.preprocessing is not None:
                X = self.preprocessing.transform(df)
            else:
                X = df.values
            
            # Run all detectors and collect scores
            detector_scores = {}
            detector_predictions = {}
            abnormal_detectors = []
            
            for detector_name, detector in self.detectors.items():
                try:
                    # Get anomaly prediction (-1 = anomaly, 1 = normal)
                    if detector_name == 'zscore':
                        # Z-score based detection
                        z_scores = np.abs(X)
                        anomaly_score = np.max(z_scores)
                        prediction = -1 if anomaly_score > 3.0 else 1
                    else:
                        prediction = detector.predict(X)[0]
                        
                        # Get anomaly score if available
                        if hasattr(detector, 'decision_function'):
                            anomaly_score = -detector.decision_function(X)[0]
                        elif hasattr(detector, 'score_samples'):
                            anomaly_score = -detector.score_samples(X)[0]
                        else:
                            anomaly_score = 1.0 if prediction == -1 else 0.0
                    
                    detector_scores[detector_name] = float(anomaly_score)
                    detector_predictions[detector_name] = int(prediction)
                    
                    if prediction == -1:
                        abnormal_detectors.append(detector_name)
                
                except Exception as e:
                    print(f"⚠️  {detector_name} failed: {e}")
                    detector_scores[detector_name] = 0.0
                    detector_predictions[detector_name] = 1
            
            # Calculate ensemble anomaly score (average of normalized scores)
            if detector_scores:
                # Normalize scores to 0-1 range
                scores_array = np.array(list(detector_scores.values()))
                if scores_array.max() > 0:
                    normalized_scores = scores_array / scores_array.max()
                else:
                    normalized_scores = scores_array
                
                ensemble_score = float(np.mean(normalized_scores))
            else:
                ensemble_score = 0.0
            
            # Determine anomaly severity
            if ensemble_score > 0.8:
                severity = "critical"
            elif ensemble_score > 0.6:
                severity = "high"
            elif ensemble_score > 0.4:
                severity = "medium"
            elif ensemble_score > 0.2:
                severity = "low"
            else:
                severity = "normal"
            
            # Identify abnormal sensors (sensors with highest contribution to anomaly)
            abnormal_sensors = self._identify_abnormal_sensors(sensor_data, ensemble_score)
            
            # Determine detection method
            if len(abnormal_detectors) == len(self.detectors):
                detection_method = "Ensemble (all detectors)"
            elif abnormal_detectors:
                detection_method = f"Ensemble ({', '.join(abnormal_detectors)})"
            else:
                detection_method = "None (normal behavior)"
            
            # Construct result
            result = {
                "machine_id": self.machine_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model_type": "anomaly_detection",
                "prediction": {
                    "is_anomaly": ensemble_score > 0.5,
                    "anomaly_score": round(ensemble_score, 4),
                    "severity": severity,
                    "detection_method": detection_method,
                    "detector_scores": {k: round(v, 4) for k, v in detector_scores.items()},
                    "detector_predictions": detector_predictions,
                    "abnormal_sensors": abnormal_sensors
                },
                "sensor_readings": {k: round(float(v), 2) for k, v in sensor_data.items()},
                "model_info": {
                    "model_path": str(self.model_path),
                    "num_detectors": len(self.detectors),
                    "detectors": list(self.detectors.keys())
                }
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Anomaly detection failed: {e}")
    
    def _identify_abnormal_sensors(self, sensor_data: Dict[str, float], 
                                   anomaly_score: float) -> List[str]:
        """
        Identify which sensors are contributing most to anomaly
        
        Args:
            sensor_data: Dictionary of sensor readings
            anomaly_score: Overall anomaly score
            
        Returns:
            List of abnormal sensor descriptions
        """
        abnormal = []
        
        # Simple heuristic: identify sensors with extreme values
        for sensor, value in sensor_data.items():
            if 'temp' in sensor.lower():
                if value > 85:
                    abnormal.append(f"{sensor}: {value:.1f}°C (high)")
                elif value < 35:
                    abnormal.append(f"{sensor}: {value:.1f}°C (low)")
            elif 'velocity' in sensor.lower() or 'vibration' in sensor.lower():
                if value > 10:
                    abnormal.append(f"{sensor}: {value:.1f} mm/s (high)")
            elif 'current' in sensor.lower():
                if value > 45:
                    abnormal.append(f"{sensor}: {value:.1f} A (high)")
                elif value < 5:
                    abnormal.append(f"{sensor}: {value:.1f} A (low)")
            elif 'voltage' in sensor.lower():
                if value > 415 or value < 385:
                    abnormal.append(f"{sensor}: {value:.1f} V (out of range)")
        
        # If no specific sensors identified but anomaly detected
        if not abnormal and anomaly_score > 0.5:
            abnormal.append("Multiple sensors showing unusual patterns")
        
        return abnormal
    
    def predict_batch(self, sensor_data_list: List[Dict[str, float]]) -> List[Dict]:
        """
        Run batch anomaly detection
        
        Args:
            sensor_data_list: List of sensor reading dictionaries
            
        Returns:
            List of prediction result dictionaries
        """
        results = []
        for i, sensor_data in enumerate(sensor_data_list):
            try:
                result = self.predict(sensor_data)
                results.append(result)
            except Exception as e:
                print(f"⚠️  Prediction {i+1} failed: {e}")
                results.append({
                    "machine_id": self.machine_id,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "model_type": "anomaly_detection",
                    "error": str(e)
                })
        
        return results


def load_sample_data(machine_id: str, num_samples: int = 10) -> List[Dict[str, float]]:
    """
    Load sample data from GAN synthetic dataset
    
    Args:
        machine_id: Machine identifier
        num_samples: Number of samples to load
        
    Returns:
        List of sensor reading dictionaries
    """
    try:
        # Try to load from GAN synthetic data (parquet format)
        data_path = PROJECT_ROOT / "GAN" / "data" / "synthetic" / machine_id / "train.parquet"
        
        if not data_path.exists():
            print(f"⚠️  Synthetic data not found: {data_path}")
            print("Generating test sensor data...")
            return generate_test_data(machine_id, num_samples)
        
        print(f"Loading sample data from: {data_path}")
        df = pd.read_parquet(data_path)
        
        # Remove target columns if present (keep timestamp as it may be a feature)
        target_cols = ['failure_mode', 'rul', 'rul_category']
        df = df.drop(columns=[col for col in target_cols if col in df.columns], errors='ignore')
        
        # Sample: mix of normal and potential anomalies
        # Get some normal samples
        normal_samples = df.sample(n=num_samples//2, random_state=42)
        
        # Create some anomalous samples by perturbing normal data
        anomalous_samples = df.sample(n=num_samples - num_samples//2, random_state=43).copy()
        
        # Perturb temperature and vibration sensors
        for col in anomalous_samples.columns:
            if 'temp' in col.lower():
                anomalous_samples[col] *= 1.3  # 30% increase
            elif 'velocity' in col.lower() or 'vibration' in col.lower():
                anomalous_samples[col] *= 2.0  # 100% increase
        
        # Combine
        combined_df = pd.concat([normal_samples, anomalous_samples], ignore_index=True)
        
        # Convert timestamp to Unix timestamp (numeric) if present
        if 'timestamp' in combined_df.columns:
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp']).astype('int64') // 10**9
        
        # Convert to list of dictionaries
        sensor_data_list = combined_df.to_dict('records')
        
        print(f"✓ Loaded {len(sensor_data_list)} samples ({num_samples//2} normal, {num_samples - num_samples//2} anomalous)")
        return sensor_data_list
        
    except Exception as e:
        print(f"⚠️  Error loading sample data: {e}")
        print("Generating test sensor data...")
        return generate_test_data(machine_id, num_samples)


def generate_test_data(machine_id: str, num_samples: int = 10) -> List[Dict[str, float]]:
    """Generate test data with mix of normal and anomalous readings"""
    from predict_classification import generate_random_sensor_data
    
    # Generate half normal, half anomalous
    normal_samples = generate_random_sensor_data(machine_id, num_samples//2)
    anomalous_samples = generate_random_sensor_data(machine_id, num_samples - num_samples//2)
    
    # Make anomalous samples actually anomalous
    for sample in anomalous_samples:
        for key in sample:
            if 'temp' in key.lower():
                sample[key] *= 1.4  # 40% increase
            elif 'velocity' in key.lower() or 'vibration' in key.lower():
                sample[key] *= 2.5  # 150% increase
    
    return normal_samples + anomalous_samples


def main():
    """Main function for testing anomaly detection inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Anomaly Detection Model Inference')
    parser.add_argument('--machine_id', type=str, required=True,
                       help='Machine identifier (e.g., motor_siemens_1la7_001)')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of predictions to generate (default: 5)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for predictions')
    
    args = parser.parse_args()
    
    print("="*60)
    print("ANOMALY DETECTION MODEL INFERENCE")
    print("="*60)
    print(f"Machine: {args.machine_id}")
    print(f"Samples: {args.num_samples}\n")
    
    try:
        # Initialize predictor
        predictor = AnomalyPredictor(args.machine_id)
        
        # Load sample data
        sensor_data_list = load_sample_data(args.machine_id, args.num_samples)
        
        # Run predictions
        print(f"\nGenerating {len(sensor_data_list)} predictions...")
        results = predictor.predict_batch(sensor_data_list)
        
        # Save results
        output_dir = Path(args.output_dir) if args.output_dir else \
                     PROJECT_ROOT / "ml_models" / "outputs" / "predictions" / "anomaly"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{args.machine_id}_predictions.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Predictions saved to: {output_file}")
        
        # Display summary
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        
        print(f"Total predictions: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}\n")
        
        if successful:
            anomalies = [r for r in successful if r['prediction']['is_anomaly']]
            normal = [r for r in successful if not r['prediction']['is_anomaly']]
            
            print(f"Anomalies detected: {len(anomalies)}")
            print(f"Normal behavior: {len(normal)}\n")
            
            print("Sample predictions:")
            for i, result in enumerate(successful[:3], 1):
                pred = result['prediction']
                print(f"\n{i}. {result['machine_id']}")
                print(f"   Is Anomaly: {pred['is_anomaly']}")
                print(f"   Anomaly Score: {pred['anomaly_score']:.4f}")
                print(f"   Severity: {pred['severity']}")
                print(f"   Detection Method: {pred['detection_method']}")
                if pred['abnormal_sensors']:
                    print(f"   Abnormal Sensors: {pred['abnormal_sensors'][0]}")
        
        print("\n" + "="*60)
        print("✓ ANOMALY DETECTION INFERENCE COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
