"""
Integration with Phase 2 ML models
Phase 3.5.2: ML Model Integration (Days 3-5)

This module integrates the MLExplainer with actual ML models to provide
end-to-end prediction + explanation capabilities.

Supports:
- Classification: Real AutoGluon models (10 machines)
- RUL Regression: Real AutoGluon models (10 machines)
- Anomaly Detection: Mock predictions (models need window-based inference)
- Time-Series Forecast: Mock predictions (models need Prophet refit fix)
"""
import pickle
import pandas as pd
from pathlib import Path
from explainer import MLExplainer
import sys
import warnings
warnings.filterwarnings('ignore')

# Add ml_models path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "ml_models" / "scripts" / "inference"))

# Import existing predictors
from predict_classification import ClassificationPredictor
from predict_rul import RULPredictor


class IntegratedPredictionSystem:
    """
    Unified system for ML predictions + LLM explanations
    
    Combines:
    - Phase 2 ML models (classification, RUL, anomaly, timeseries)
    - Phase 3 MLExplainer (LLM + RAG)
    
    Returns predictions with human-readable explanations for technicians.
    """
    
    def __init__(self):
        """Initialize explainer and model cache"""
        print("\n" + "="*70)
        print("Initializing IntegratedPredictionSystem")
        print("="*70)
        
        print("\n[1/2] Loading MLExplainer (LLM + RAG)...")
        self.explainer = MLExplainer()
        
        print("[2/2] Preparing model cache...")
        # Cache for loaded models (lazy loading)
        self.models = {
            'classification': {},  # Will load on demand
            'regression': {},      # Will load on demand
            'anomaly': {},         # Not implemented yet (using mock)
            'timeseries': {}       # Not implemented yet (using mock)
        }
        self.models_dir = PROJECT_ROOT / "ml_models" / "models"
        
        print("\n" + "="*70)
        print("✓ IntegratedPredictionSystem Ready")
        print("="*70)
        print()
    
    def predict_with_explanation(self, machine_id, sensor_data, model_type='all'):
        """
        Run ML prediction and generate LLM explanation
        
        Args:
            machine_id: Machine identifier (e.g., "motor_siemens_1la7_001")
            sensor_data: Dict of sensor readings
            model_type: 'classification', 'regression', 'anomaly', 'timeseries', or 'all'
        
        Returns:
            Dict with predictions and explanations for each model type
            
        Example:
            {
                'classification': {
                    'prediction': {...},
                    'explanation': {...}
                },
                'regression': {...}
            }
        """
        print(f"\n{'='*70}")
        print(f"Running integrated prediction for {machine_id}")
        print(f"Model types: {model_type}")
        print(f"{'='*70}\n")
        
        results = {}
        
        # Classification
        if model_type in ['classification', 'all']:
            try:
                print(f"[Classification] Running prediction...")
                pred = self.predict_classification(machine_id, sensor_data)
                
                print(f"[Classification] Generating explanation...")
                explanation = self.explainer.explain_classification(
                    machine_id=machine_id,
                    failure_prob=pred['failure_probability'],
                    failure_type=pred['failure_type'],
                    sensor_data=pred['sensor_readings'],  # Use actual sensor readings from prediction
                    confidence=pred['confidence']
                )
                
                results['classification'] = {
                    'prediction': pred,
                    'explanation': explanation
                }
                print(f"[Classification] ✓ Complete\n")
                
            except Exception as e:
                print(f"[Classification] ✗ Error: {e}\n")
                results['classification'] = {'error': str(e)}
        
        # Regression (RUL)
        if model_type in ['regression', 'rul', 'all']:
            try:
                print(f"[RUL] Running prediction...")
                rul = self.predict_rul(machine_id, sensor_data)
                
                print(f"[RUL] Generating explanation...")
                explanation = self.explainer.explain_rul(
                    machine_id=machine_id,
                    rul_hours=rul['rul_hours'],
                    sensor_data=rul['sensor_readings'],  # Use actual sensor readings from prediction
                    confidence=rul['confidence']
                )
                
                results['regression'] = {
                    'prediction': rul,
                    'explanation': explanation
                }
                print(f"[RUL] ✓ Complete\n")
                
            except Exception as e:
                print(f"[RUL] ✗ Error: {e}\n")
                results['regression'] = {'error': str(e)}
        
        # Anomaly (Mock - models need window-based inference)
        if model_type in ['anomaly', 'all']:
            try:
                print(f"[Anomaly] Running detection (MOCK - see Phase 3.5.0 exceptions)...")
                anomaly = self.detect_anomaly(machine_id, sensor_data)
                
                if anomaly['is_anomaly']:
                    print(f"[Anomaly] Generating explanation...")
                    explanation = self.explainer.explain_anomaly(
                        machine_id=machine_id,
                        anomaly_score=anomaly['score'],
                        abnormal_sensors=anomaly['abnormal_sensors'],
                        detection_method=anomaly['method']
                    )
                    
                    results['anomaly'] = {
                        'prediction': anomaly,
                        'explanation': explanation
                    }
                    print(f"[Anomaly] ✓ Complete\n")
                else:
                    results['anomaly'] = {
                        'prediction': anomaly,
                        'explanation': {'note': 'No anomaly detected - explanation not needed'}
                    }
                    print(f"[Anomaly] ✓ No anomaly detected\n")
                    
            except Exception as e:
                print(f"[Anomaly] ✗ Error: {e}\n")
                results['anomaly'] = {'error': str(e)}
        
        # Time-Series Forecast (Mock - models need Prophet refit fix)
        if model_type in ['timeseries', 'forecast', 'all']:
            try:
                print(f"[TimeSeries] Running forecast (MOCK - see Phase 3.5.0 exceptions)...")
                forecast = self.predict_forecast(machine_id, sensor_data)
                
                print(f"[TimeSeries] Generating explanation...")
                explanation = self.explainer.explain_forecast(
                    machine_id=machine_id,
                    forecast_summary=forecast['forecast_summary'],
                    confidence=forecast['confidence']
                )
                
                results['timeseries'] = {
                    'prediction': forecast,
                    'explanation': explanation
                }
                print(f"[TimeSeries] ✓ Complete\n")
                
            except Exception as e:
                print(f"[TimeSeries] ✗ Error: {e}\n")
                results['timeseries'] = {'error': str(e)}
        
        print(f"{'='*70}")
        print(f"✓ Integrated prediction complete ({len(results)} model types)")
        print(f"{'='*70}\n")
        
        return results
    
    def predict_classification(self, machine_id: str, sensor_data: dict = None) -> dict:
        """
        Run classification prediction using real AutoGluon model
        
        Args:
            machine_id: Machine identifier
            sensor_data: Dict of sensor readings (optional - will load from GAN data if not provided)
            
        Returns:
            Dict with failure type, probability, and confidence
        """
        # Load model if not cached
        if machine_id not in self.models['classification']:
            model_path = self.models_dir / "classification" / machine_id
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Classification model not found for {machine_id}. "
                    f"Expected: {model_path}"
                )
            
            self.models['classification'][machine_id] = ClassificationPredictor(machine_id)
        
        # If no sensor data provided, load from GAN synthetic dataset
        if sensor_data is None:
            sensor_data = self._load_sample_data(machine_id, num_samples=1)[0]
        
        # Run prediction
        predictor = self.models['classification'][machine_id]
        result = predictor.predict(sensor_data)
        
        return {
            'failure_type': result['prediction']['failure_type'],
            'failure_probability': result['prediction']['failure_probability'],
            'confidence': result['prediction']['confidence'],
            'all_probabilities': result['prediction']['all_probabilities'],
            'sensor_readings': result['sensor_readings']
        }
    
    def predict_rul(self, machine_id: str, sensor_data: dict = None) -> dict:
        """
        Run RUL regression prediction using real AutoGluon model
        
        Args:
            machine_id: Machine identifier
            sensor_data: Dict of sensor readings (optional - will load from GAN data if not provided)
            
        Returns:
            Dict with RUL hours and confidence
        """
        # Load model if not cached
        if machine_id not in self.models['regression']:
            model_path = self.models_dir / "regression" / machine_id
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"RUL model not found for {machine_id}. "
                    f"Expected: {model_path}"
                )
            
            self.models['regression'][machine_id] = RULPredictor(machine_id)
        
        # If no sensor data provided, load from GAN synthetic dataset
        if sensor_data is None:
            sensor_data = self._load_sample_data(machine_id, num_samples=1)[0]
        
        # Run prediction
        predictor = self.models['regression'][machine_id]
        result = predictor.predict(sensor_data)
        
        return {
            'rul_hours': result['prediction']['rul_hours'],
            'rul_days': result['prediction']['rul_days'],
            'confidence': result['prediction'].get('confidence', 0.85),
            'estimated_failure_date': result['prediction'].get('estimated_failure_date'),
            'maintenance_window': result['prediction'].get('maintenance_window'),
            'urgency': result['prediction'].get('urgency'),
            'sensor_readings': result['sensor_readings']
        }
    
    def detect_anomaly(self, machine_id: str, sensor_data: dict = None) -> dict:
        """
        Run anomaly detection (MOCK - real models need window-based inference)
        
        Note: This is a mock implementation. Real anomaly models require:
        - Window-based data loading (not single-point)
        - Feature engineering (rolling means, lags, 388 features)
        - See Phase 3.5.0 exception documentation
        
        Args:
            machine_id: Machine identifier
            sensor_data: Dict of sensor readings (optional - will load from GAN if not provided)
            
        Returns:
            Dict with anomaly score and abnormal sensors
        """
        # If no sensor data provided, load from GAN synthetic dataset
        if sensor_data is None:
            sensor_data = self._load_sample_data(machine_id, num_samples=1)[0]
        
        # Mock implementation - generate realistic anomaly based on sensor values
        import random
        
        # Check if any sensor is significantly elevated
        thresholds = {
            'vibration': 10.0,
            'temperature': 75.0,
            'current': 50.0,
            'pressure': 8.0,
            'rms_velocity_mm_s': 10.0,
            'bearing_de_temp_C': 70.0,
            'bearing_nde_temp_C': 70.0,
            'winding_temp_C': 80.0
        }
        
        abnormal_sensors = {}
        for sensor, value in sensor_data.items():
            threshold = thresholds.get(sensor, 100.0)
            if value > threshold * 0.8:  # 80% of threshold
                abnormal_sensors[sensor] = value
        
        is_anomaly = len(abnormal_sensors) > 0
        anomaly_score = min(0.95, len(abnormal_sensors) * 0.25 + random.uniform(0.1, 0.3))
        
        return {
            'is_anomaly': is_anomaly,
            'score': anomaly_score if is_anomaly else 0.0,
            'abnormal_sensors': abnormal_sensors,
            'method': 'Isolation Forest (Mock)',
            'note': 'Mock prediction - real models need window-based inference (Phase 3.5.0 exception)'
        }
    
    def predict_forecast(self, machine_id: str, sensor_data: dict = None) -> dict:
        """
        Run time-series forecasting (MOCK - real models need Prophet refit fix)
        
        Note: This is a mock implementation. Real Prophet models require:
        - Removing refit logic (models already fitted)
        - Proper time-series data structure
        - See Phase 3.5.0 exception documentation
        
        Args:
            machine_id: Machine identifier
            sensor_data: Dict of sensor readings (optional - will load from GAN if not provided)
            
        Returns:
            Dict with forecast summary and confidence
        """
        # If no sensor data provided, load from GAN synthetic dataset
        if sensor_data is None:
            sensor_data = self._load_sample_data(machine_id, num_samples=1)[0]
        
        import random
        
        # Mock forecast based on current sensor values
        temp_change = random.choice([3, 5, 8, -2, -3])
        vib_trend = "upward" if sensor_data.get('vibration', sensor_data.get('rms_velocity_mm_s', 0)) > 8 else "stable"
        
        if 'temperature' in sensor_data or 'winding_temp_C' in sensor_data:
            temp_key = 'temperature' if 'temperature' in sensor_data else 'winding_temp_C'
            current_temp = sensor_data[temp_key]
            
            forecast_summary = (
                f"Temperature predicted to {'rise' if temp_change > 0 else 'drop'} by "
                f"{abs(temp_change)}°C over next 24 hours (current: {current_temp:.1f}°C). "
                f"Vibration trending {vib_trend}."
            )
        else:
            forecast_summary = f"Sensor trends stable. Vibration {vib_trend}. No significant changes expected."
        
        return {
            'forecast_summary': forecast_summary,
            'confidence': 0.85,
            'forecast_horizon': '24 hours',
            'note': 'Mock prediction - real models need Prophet refit fix (Phase 3.5.0 exception)'
        }
    
    def _load_sample_data(self, machine_id: str, num_samples: int = 1) -> list:
        """
        Load sample sensor data from GAN synthetic dataset
        
        Args:
            machine_id: Machine identifier
            num_samples: Number of samples to load
            
        Returns:
            List of sensor reading dictionaries (with all features)
        """
        data_path = PROJECT_ROOT / "GAN" / "data" / "synthetic" / machine_id / "train.parquet"
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Synthetic training data not found for {machine_id}. "
                f"Expected: {data_path}"
            )
        
        # Load parquet data
        df = pd.read_parquet(data_path)
        
        # Remove target columns
        target_cols = ['failure_mode', 'rul_category', 'failure_status']
        df = df.drop(columns=[col for col in target_cols if col in df.columns], errors='ignore')
        
        # Sample random rows
        samples = df.sample(n=min(num_samples, len(df)))
        
        # Convert timestamp to Unix format
        if 'timestamp' in samples.columns:
            samples['timestamp'] = samples['timestamp'].astype('int64') // 10**9
        
        # Convert to list of dictionaries
        return samples.to_dict('records')


# Test the integrated system
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING IntegratedPredictionSystem - Phase 3.5.2")
    print("="*70)
    
    # Initialize system (loads LLM + RAG)
    system = IntegratedPredictionSystem()
    
    # Test with motor_siemens_1la7_001
    test_machine = "motor_siemens_1la7_001"
    
    print(f"\nTest Machine: {test_machine}")
    print(f"NOTE: Using real sensor data from GAN synthetic dataset\n")
    
    # Run integrated prediction (will load sensor data from GAN automatically)
    results = system.predict_with_explanation(
        machine_id=test_machine,
        sensor_data=None,  # Will auto-load from GAN dataset
        model_type='all'
    )
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for model_type, result in results.items():
        print(f"\n{model_type.upper()}:")
        print("-"*70)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Prediction: {result['prediction']}")
            if 'explanation' in result:
                exp = result['explanation']
                if isinstance(exp, dict) and 'explanation' in exp:
                    print(f"\nExplanation Preview:")
                    print(exp['explanation'][:300] + "...")
                else:
                    print(f"\nExplanation: {exp}")
    
    print("\n" + "="*70)
    print("✓ TEST COMPLETE")
    print("="*70)
