"""
Classification Model Inference Script
Phase 3.5.0: ML Model Inference Pipeline

Loads trained AutoGluon classification models and generates predictions
for failure classification (bearing_wear, overheating, electrical_fault, normal)

Industrial-grade implementation with:
- Error handling and validation
- Logging and monitoring
- Confidence score extraction
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


class ClassificationPredictor:
    """Loads and runs inference with trained classification models"""
    
    def __init__(self, machine_id: str, models_dir: Optional[Path] = None):
        """
        Initialize classification predictor
        
        Args:
            machine_id: Machine identifier (e.g., 'motor_siemens_1la7_001')
            models_dir: Path to models directory (optional, uses default)
        """
        self.machine_id = machine_id
        self.models_dir = models_dir or PROJECT_ROOT / "ml_models" / "models" / "classification"
        self.model_path = self.models_dir / machine_id
        self.model = None
        self.feature_names = None
        self.label_mapping = {
            0: "normal",
            1: "bearing_wear",
            2: "overheating", 
            3: "electrical_fault"
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load trained AutoGluon classification model"""
        try:
            from autogluon.tabular import TabularPredictor
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            print(f"Loading classification model for {self.machine_id}...")
            self.model = TabularPredictor.load(str(self.model_path))
            
            # Extract feature names from model
            self.feature_names = self.model.feature_metadata.get_features()
            
            print(f"[OK] Model loaded successfully")
            print(f"[OK] Features: {len(self.feature_names)}")
            print(f"[OK] Label column: {self.model.label}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load classification model: {e}")
    
    def predict(self, sensor_data: Dict[str, float]) -> Dict:
        """
        Run classification prediction
        
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
            
            # Filter and align features
            if hasattr(self, 'feature_names') and self.feature_names:
                # Ensure all required features are present
                missing_features = [f for f in self.feature_names if f not in df.columns]
                if missing_features:
                    # print(f"⚠️ Warning: {len(missing_features)} features missing, filling with 0.0")
                    for f in missing_features:
                        df[f] = 0.0
                
                # Select only required columns (ignores extra columns like timestamp)
                df = df[self.feature_names]
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(df)
            
            # Get predicted class
            prediction = self.model.predict(df)
            predicted_class = prediction.iloc[0]
            
            # Get failure type and confidence
            if isinstance(predicted_class, (int, np.integer)):
                failure_type = self.label_mapping.get(predicted_class, "unknown")
            else:
                failure_type = predicted_class
            
            # Extract confidence scores for all classes
            class_probabilities = {}
            for col in probabilities.columns:
                if isinstance(col, (int, np.integer)):
                    class_name = self.label_mapping.get(col, f"class_{col}")
                else:
                    class_name = col
                class_probabilities[class_name] = float(probabilities[col].iloc[0])
            
            # Get failure probability (1 - normal probability)
            failure_prob = 1.0 - class_probabilities.get("normal", 0.0)
            confidence = class_probabilities.get(failure_type, 0.0)
            
            # Construct result
            result = {
                "machine_id": self.machine_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model_type": "classification",
                "prediction": {
                    "failure_probability": round(failure_prob, 4),
                    "failure_type": failure_type,
                    "confidence": round(confidence, 4),
                    "all_probabilities": {k: round(v, 4) for k, v in class_probabilities.items()}
                },
                "sensor_readings": {k: round(float(v), 2) for k, v in sensor_data.items()},
                "model_info": {
                    "model_path": str(self.model_path),
                    "best_model": self.model.model_best if hasattr(self.model, 'model_best') else "unknown",
                    "num_features": len(self.feature_names)
                }
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def predict_batch(self, sensor_data_list: List[Dict[str, float]]) -> List[Dict]:
        """
        Run batch predictions
        
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
                    "model_type": "classification",
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
            print("Generating random sensor data for testing...")
            return generate_random_sensor_data(machine_id, num_samples)
        
        print(f"Loading sample data from: {data_path}")
        df = pd.read_parquet(data_path)
        
        # Remove target columns if present (keep timestamp and rul as they may be features)
        target_cols = ['failure_mode', 'rul_category']
        df = df.drop(columns=[col for col in target_cols if col in df.columns], errors='ignore')
        
        # Sample random rows
        if len(df) > num_samples:
            df = df.sample(n=num_samples, random_state=42)
        
        # Convert timestamp to Unix timestamp (numeric) if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**9
        
        # Convert to list of dictionaries
        sensor_data_list = df.to_dict('records')
        
        print(f"✓ Loaded {len(sensor_data_list)} samples")
        return sensor_data_list
        
    except Exception as e:
        print(f"⚠️  Error loading sample data: {e}")
        print("Generating random sensor data for testing...")
        return generate_random_sensor_data(machine_id, num_samples)


def generate_random_sensor_data(machine_id: str, num_samples: int = 10) -> List[Dict[str, float]]:
    """
    Generate random sensor data for testing when real data not available
    
    Args:
        machine_id: Machine identifier
        num_samples: Number of samples to generate
        
    Returns:
        List of sensor reading dictionaries
    """
    # Load metadata to get sensor names
    metadata_path = PROJECT_ROOT / "GAN" / "metadata" / f"{machine_id}_metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        sensor_columns = list(metadata['columns'].keys())
    else:
        # Default sensors for motors
        sensor_columns = [
            'bearing_de_temp_C', 'bearing_nde_temp_C', 'winding_temp_C',
            'casing_temp_C', 'ambient_temp_C', 'rms_velocity_mm_s',
            'peak_velocity_mm_s', 'current_100pct_load_A', 'voltage_phase_to_phase_V'
        ]
    
    np.random.seed(42)
    sensor_data_list = []
    
    for _ in range(num_samples):
        sensor_data = {}
        for col in sensor_columns:
            # Generate realistic values based on sensor type
            if 'temp' in col.lower():
                sensor_data[col] = round(np.random.uniform(40, 80), 2)
            elif 'velocity' in col.lower() or 'vibration' in col.lower():
                sensor_data[col] = round(np.random.uniform(0.5, 15.0), 2)
            elif 'current' in col.lower():
                sensor_data[col] = round(np.random.uniform(10, 50), 2)
            elif 'voltage' in col.lower():
                sensor_data[col] = round(np.random.uniform(380, 420), 2)
            elif 'frequency' in col.lower():
                sensor_data[col] = round(np.random.uniform(50, 200), 2)
            elif 'power_factor' in col.lower():
                sensor_data[col] = round(np.random.uniform(0.7, 0.95), 3)
            elif 'efficiency' in col.lower():
                sensor_data[col] = round(np.random.uniform(0.85, 0.95), 3)
            elif 'sound' in col.lower():
                sensor_data[col] = round(np.random.uniform(60, 85), 2)
            else:
                sensor_data[col] = round(np.random.uniform(0, 100), 2)
        
        sensor_data_list.append(sensor_data)
    
    return sensor_data_list


def main():
    """Main function for testing classification inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Classification Model Inference')
    parser.add_argument('--machine_id', type=str, required=True,
                       help='Machine identifier (e.g., motor_siemens_1la7_001)')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of predictions to generate (default: 5)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for predictions (default: ml_models/outputs/predictions/classification)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CLASSIFICATION MODEL INFERENCE")
    print("="*60)
    print(f"Machine: {args.machine_id}")
    print(f"Samples: {args.num_samples}\n")
    
    try:
        # Initialize predictor
        predictor = ClassificationPredictor(args.machine_id)
        
        # Load sample data
        sensor_data_list = load_sample_data(args.machine_id, args.num_samples)
        
        # Run predictions
        print(f"\nGenerating {len(sensor_data_list)} predictions...")
        results = predictor.predict_batch(sensor_data_list)
        
        # Save results
        output_dir = Path(args.output_dir) if args.output_dir else \
                     PROJECT_ROOT / "ml_models" / "outputs" / "predictions" / "classification"
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
            print("Sample predictions:")
            for i, result in enumerate(successful[:3], 1):
                pred = result['prediction']
                print(f"\n{i}. {result['machine_id']}")
                print(f"   Failure Type: {pred['failure_type']}")
                print(f"   Failure Probability: {pred['failure_probability']*100:.1f}%")
                print(f"   Confidence: {pred['confidence']*100:.1f}%")
        
        print("\n" + "="*60)
        print("✓ CLASSIFICATION INFERENCE COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
