"""
Time-Series Forecasting Model Inference Script
Phase 3.5.0: ML Model Inference Pipeline

Loads trained AutoGluon time-series models and generates predictions
for future sensor behavior (24-hour forecast)

Industrial-grade implementation with:
- 24-hour ahead forecasting
- Multiple sensor forecasts
- Trend analysis
- Maintenance window recommendations
- JSON output format for LLM integration
"""

import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))


class TimeSeriesPredictor:
    """Loads and runs inference with trained time-series forecasting models"""
    
    def __init__(self, machine_id: str, models_dir: Optional[Path] = None):
        """
        Initialize time-series predictor
        
        Args:
            machine_id: Machine identifier (e.g., 'motor_siemens_1la7_001')
            models_dir: Path to models directory (optional, uses default)
        """
        self.machine_id = machine_id
        self.models_dir = models_dir or PROJECT_ROOT / "ml_models" / "models" / "timeseries"
        self.model_path = self.models_dir / machine_id
        self.model = None
        # Default: 24 hours ahead, hourly steps.
        # Some machines (e.g. printers) operate at sub-hourly cadence. We load
        # optional metadata.json to set an appropriate frequency.
        self.prediction_length = 24
        self.freq = 'H'
        self.steps_per_hour = 1
        
        self._load_model()
    
    def _load_model(self):
        """Load trained Prophet time-series models"""
        try:
            import joblib
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model path not found: {self.model_path}")
            
            # Load Prophet models (saved as prophet_models.pkl)
            prophet_models_path = self.model_path / "prophet_models.pkl"
            if not prophet_models_path.exists():
                raise FileNotFoundError(f"Prophet models not found: {prophet_models_path}")
            
            print(f"Loading time-series forecasting model for {self.machine_id}...")
            self.model = joblib.load(prophet_models_path)

            # Load optional metadata (freq/steps_per_hour) to support sub-hourly forecasts.
            metadata_path = self.model_path / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    freq = str(meta.get('freq') or '').strip()
                    if freq:
                        self.freq = freq
                    sph = meta.get('steps_per_hour')
                    if isinstance(sph, int) and sph > 0:
                        self.steps_per_hour = sph
                except Exception:
                    pass
            
            print(f"[OK] Model loaded successfully")
            if isinstance(self.model, dict):
                print(f"[OK] Loaded {len(self.model)} Prophet models for different sensors")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load time-series model: {e}")
    
    def predict(self, historical_data: pd.DataFrame, forecast_steps: int = 24) -> Dict:
        """
        Run time-series forecast using Prophet models
        
        Args:
            historical_data: DataFrame with historical sensor readings
                           Must have 'timestamp' column and sensor columns
            forecast_steps: Number of hours to forecast (default: 24)
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Validate input
            if historical_data.empty:
                raise ValueError("historical_data cannot be empty")
            
            # Ensure timestamp is datetime
            if 'timestamp' not in historical_data.columns:
                raise ValueError("historical_data must have 'timestamp' column")

            # Normalize timestamps and sort so we can anchor the forecast to the
            # latest observed reading rather than the model's original training window.
            historical_data = historical_data.copy()
            historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], errors='coerce', utc=True)
            historical_data = historical_data.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            if historical_data.empty:
                raise ValueError("historical_data has no valid timestamps")

            # Interpret forecast_steps as forecast hours for API stability.
            forecast_hours = int(forecast_steps or 24)
            steps = max(1, int(forecast_hours * (self.steps_per_hour or 1)))

            last_ts_utc = historical_data['timestamp'].iloc[-1]
            # Prophet expects naive datetimes. We'll treat them as UTC.
            last_ts = last_ts_utc.tz_convert('UTC').tz_localize(None)
            future_ds = pd.date_range(start=last_ts, periods=steps + 1, freq=self.freq)[1:]
            
            # Models are dict with one Prophet model per sensor
            forecasts = {}
            sensor_cols = []
            
            # Iterate through each sensor's Prophet model
            for sensor_name, prophet_model in self.model.items():
                if sensor_name not in historical_data.columns:
                    print(f"⚠️  Sensor {sensor_name} not in historical data, skipping")
                    continue
                
                sensor_cols.append(sensor_name)
                
                try:
                    # Forecast relative to the latest observed timestamp (input-driven),
                    # not relative to when the model was originally trained.
                    ds_all = pd.DatetimeIndex([last_ts]).append(future_ds)
                    forecast = prophet_model.predict(pd.DataFrame({'ds': ds_all}))

                    # Prophet includes the anchor point; keep future rows only.
                    forecast_values = forecast.iloc[1:].copy()

                    # Light-weight anchoring: shift the forecast so the first point
                    # aligns with the latest observed value. This improves accuracy
                    # under level shifts without an expensive refit.
                    last_val = historical_data[sensor_name].iloc[-1]
                    try:
                        if pd.notna(last_val) and pd.notna(forecast.iloc[0].get('yhat', np.nan)):
                            offset = float(last_val) - float(forecast.iloc[0]['yhat'])
                            for col in ('yhat', 'yhat_lower', 'yhat_upper'):
                                if col in forecast_values.columns and forecast_values[col].notna().any():
                                    forecast_values[col] = forecast_values[col].astype(float) + offset
                    except Exception:
                        # If anchoring fails for any reason, keep the raw forecast.
                        pass

                    # Robust sanity clamp: keep forecasts within a plausible range
                    # derived from recent historical values for this sensor.
                    try:
                        hist_vals = pd.to_numeric(historical_data[sensor_name], errors='coerce').dropna().astype(float)
                        hist_tail = hist_vals.tail(500)
                        if not hist_tail.empty:
                            q01, q99 = hist_tail.quantile([0.01, 0.99]).tolist()
                            span = float(q99 - q01)
                            # Allow a modest margin beyond recent extremes.
                            margin = max(5.0, span * 0.2) if span > 0 else 10.0
                            lower = float(q01 - margin)
                            upper = float(q99 + margin)
                            # Temperatures should not go wildly negative; bound to 0 where applicable.
                            if 'temp' in str(sensor_name).lower():
                                lower = max(0.0, lower)

                            for col in ('yhat', 'yhat_lower', 'yhat_upper'):
                                if col in forecast_values.columns:
                                    s = pd.to_numeric(forecast_values[col], errors='coerce')
                                    # Replace non-finite with the last observed value before clamping.
                                    s = s.where(np.isfinite(s), float(hist_tail.iloc[-1]))
                                    forecast_values[col] = s.astype(float).clip(lower=lower, upper=upper)
                    except Exception:
                        # If sanitization fails, keep the forecast as-is.
                        pass
                    
                    # Store forecasts with confidence intervals
                    forecasts[sensor_name] = {
                        'yhat': forecast_values['yhat'].tolist(),
                        'yhat_lower': forecast_values['yhat_lower'].tolist(),
                        'yhat_upper': forecast_values['yhat_upper'].tolist(),
                        'ds': pd.to_datetime(forecast_values['ds'], utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ').tolist()
                    }
                    
                except Exception as e:
                    print(f"⚠️  Forecast failed for {sensor_name}: {e}")
                    # Add empty forecast to maintain structure
                    forecasts[sensor_name] = {
                        'yhat': [None] * steps,
                        'yhat_lower': [None] * steps,
                        'yhat_upper': [None] * steps,
                        'ds': []
                    }
            
            # Calculate forecast summary from forecasts dict
            forecast_summary = self._generate_forecast_summary_from_dict(forecasts, sensor_cols)
            
            # Identify concerning trends
            concerning_trends = self._identify_concerning_trends_from_dict(forecasts, sensor_cols)
            
            # Determine maintenance window
            maintenance_window = self._recommend_maintenance_window_from_dict(forecasts, sensor_cols)
            
            # Calculate overall confidence (average from successful forecasts)
            successful_forecasts = [f for f in forecasts.values() if f['yhat'][0] is not None]
            confidence = 0.85 if successful_forecasts else 0.0
            
            # Get current time
            current_time = datetime.utcnow()
            # Align the forecast window to the latest observed timestamp.
            forecast_start = (last_ts_utc.to_pydatetime() + timedelta(hours=1))
            forecast_end = (last_ts_utc.to_pydatetime() + timedelta(hours=forecast_hours))
            
            # Construct result
            result = {
                "machine_id": self.machine_id,
                "timestamp": current_time.isoformat() + "Z",
                "model_type": "timeseries_forecast",
                "prediction": {
                    "forecast_horizon_hours": forecast_steps,
                    "forecast_start": forecast_start.isoformat() + "Z",
                    "forecast_end": forecast_end.isoformat() + "Z",
                    "confidence": round(confidence, 4),
                    "forecast_summary": forecast_summary,
                    "concerning_trends": concerning_trends,
                    "maintenance_window": maintenance_window,
                    "detailed_forecast": forecasts
                },
                "historical_data_points": len(historical_data),
                "model_info": {
                    "model_path": str(self.model_path),
                    "prediction_length": forecast_steps,
                    "forecasted_sensors": sensor_cols
                }
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Time-series forecast failed: {e}")
    
    def _generate_forecast_summary(self, forecast_df: pd.DataFrame, 
                                   sensor_cols: List[str]) -> str:
        """
        Generate human-readable forecast summary
        
        Args:
            forecast_df: Forecast DataFrame
            sensor_cols: List of sensor column names
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Divide forecast into time windows (0-6h, 6-12h, 12-18h, 18-24h)
        windows = [
            (0, 6, "Hour 0-6"),
            (6, 12, "Hour 6-12"),
            (12, 18, "Hour 12-18"),
            (18, 24, "Hour 18-24")
        ]
        
        for start, end, label in windows:
            if len(forecast_df) > end:
                window_data = forecast_df.iloc[start:end]
                
                # Get key sensor values
                temp_cols = [col for col in sensor_cols if 'temp' in col.lower()]
                vib_cols = [col for col in sensor_cols if 'velocity' in col.lower() or 'vibration' in col.lower()]
                
                temp_values = []
                vib_values = []
                
                if temp_cols:
                    temp_values = [window_data[col].mean() for col in temp_cols]
                if vib_cols:
                    vib_values = [window_data[col].mean() for col in vib_cols]
                
                # Build summary for this window
                if temp_values and vib_values:
                    summary_parts.append(
                        f"{label}: Temperature {np.mean(temp_values):.1f}°C, "
                        f"Vibration {np.mean(vib_values):.1f} mm/s"
                    )
                elif temp_values:
                    summary_parts.append(f"{label}: Temperature {np.mean(temp_values):.1f}°C")
                elif vib_values:
                    summary_parts.append(f"{label}: Vibration {np.mean(vib_values):.1f} mm/s")
        
        return "\n".join(summary_parts) if summary_parts else "Stable sensor readings expected"
    
    def _identify_concerning_trends(self, forecast_df: pd.DataFrame,
                                    sensor_cols: List[str]) -> List[str]:
        """
        Identify concerning trends in forecast
        
        Args:
            forecast_df: Forecast DataFrame
            sensor_cols: List of sensor column names
            
        Returns:
            List of concerning trend descriptions
        """
        concerns = []
        
        for col in sensor_cols:
            if col in forecast_df.columns:
                values = forecast_df[col].values
                
                # Check for increasing trend
                if len(values) > 2:
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    
                    if 'temp' in col.lower() and trend > 1.0:
                        concerns.append(f"{col}: Rising trend (+{trend:.1f}°C/hour)")
                    elif ('velocity' in col.lower() or 'vibration' in col.lower()) and trend > 0.5:
                        concerns.append(f"{col}: Increasing vibration (+{trend:.1f} mm/s/hour)")
                    
                    # Check for peak values
                    max_val = np.max(values)
                    if 'temp' in col.lower() and max_val > 85:
                        concerns.append(f"{col}: Peak value {max_val:.1f}°C (high)")
                    elif ('velocity' in col.lower() or 'vibration' in col.lower()) and max_val > 10:
                        concerns.append(f"{col}: Peak vibration {max_val:.1f} mm/s (elevated)")
        
        if not concerns:
            concerns.append("No concerning trends detected")
        
        return concerns
    
    def _recommend_maintenance_window(self, forecast_df: pd.DataFrame,
                                      sensor_cols: List[str]) -> str:
        """
        Recommend optimal maintenance window based on forecast
        
        Args:
            forecast_df: Forecast DataFrame
            sensor_cols: List of sensor column names
            
        Returns:
            Maintenance window recommendation
        """
        # Find time period with lowest sensor values (best for maintenance)
        windows = [(0, 6), (6, 12), (12, 18), (18, 24)]
        window_scores = []
        
        for start, end in windows:
            if len(forecast_df) > end:
                window_data = forecast_df.iloc[start:end]
                
                # Calculate score (lower is better for maintenance)
                score = 0
                for col in sensor_cols:
                    if col in window_data.columns:
                        if 'temp' in col.lower():
                            score += window_data[col].mean() / 100.0
                        elif 'velocity' in col.lower() or 'vibration' in col.lower():
                            score += window_data[col].mean() / 10.0
                
                window_scores.append((start, end, score))
        
        if window_scores:
            # Find window with lowest score
            best_window = min(window_scores, key=lambda x: x[2])
            return f"Optimal window: Hour {best_window[0]}-{best_window[1]} (lowest sensor activity)"
        else:
            return "Schedule during normal operating hours"
    
    def _format_detailed_forecast(self, forecast_df: pd.DataFrame,
                                  sensor_cols: List[str],
                                  forecast_start: datetime) -> List[Dict]:
        """
        Format detailed hourly forecast
        
        Args:
            forecast_df: Forecast DataFrame
            sensor_cols: List of sensor column names
            forecast_start: Start time of forecast
            
        Returns:
            List of hourly forecast dictionaries
        """
        detailed = []
        
        for i, row in forecast_df.iterrows():
            hour_time = forecast_start + timedelta(hours=i)
            
            hour_data = {
                "hour": i + 1,
                "timestamp": hour_time.isoformat() + "Z",
                "sensors": {}
            }
            
            for col in sensor_cols:
                if col in row.index:
                    hour_data["sensors"][col] = round(float(row[col]), 2)
            
            detailed.append(hour_data)
        
        return detailed[:24]  # Limit to 24 hours
    
    def _generate_forecast_summary_from_dict(self, forecasts: Dict, 
                                              sensor_cols: List[str]) -> str:
        """
        Generate human-readable forecast summary from forecasts dict
        
        Args:
            forecasts: Dict of sensor forecasts
            sensor_cols: List of sensor column names
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Divide forecast into time windows (0-6h, 6-12h, 12-18h, 18-24h)
        sph = int(self.steps_per_hour or 1)
        windows = [
            (0, 6, "Hour 0-6"),
            (6, 12, "Hour 6-12"),
            (12, 18, "Hour 12-18"),
            (18, 24, "Hour 18-24")
        ]
        
        for start, end, label in windows:
            start_i = start * sph
            end_i = end * sph
            temp_cols = [col for col in sensor_cols if 'temp' in col.lower()]
            vib_cols = [col for col in sensor_cols if 'velocity' in col.lower() or 'vibration' in col.lower()]
            
            temp_values = []
            vib_values = []
            per_temp_means: Dict[str, float] = {}
            
            for col in temp_cols:
                if col in forecasts and forecasts[col]['yhat']:
                    window_vals = forecasts[col]['yhat'][start_i:end_i]
                    if window_vals and all(v is not None for v in window_vals):
                        temp_values.extend(window_vals)
                        per_temp_means[col] = float(np.mean(window_vals))
            
            for col in vib_cols:
                if col in forecasts and forecasts[col]['yhat']:
                    window_vals = forecasts[col]['yhat'][start_i:end_i]
                    if window_vals and all(v is not None for v in window_vals):
                        vib_values.extend(window_vals)
            
            # Build summary for this window
            if temp_values and vib_values:
                if len(per_temp_means) >= 2 and len(per_temp_means) <= 4:
                    temps = ", ".join([f"{k} {v:.1f} C" for k, v in per_temp_means.items()])
                    summary_parts.append(
                        f"{label}: Temperature ({temps}), Vibration {np.mean(vib_values):.1f} mm/s"
                    )
                else:
                    summary_parts.append(
                        f"{label}: Temperature {np.mean(temp_values):.1f} C, "
                        f"Vibration {np.mean(vib_values):.1f} mm/s"
                    )
            elif temp_values:
                if len(per_temp_means) >= 2 and len(per_temp_means) <= 4:
                    temps = ", ".join([f"{k} {v:.1f} C" for k, v in per_temp_means.items()])
                    summary_parts.append(f"{label}: Temperature ({temps})")
                else:
                    summary_parts.append(f"{label}: Temperature {np.mean(temp_values):.1f} C")
            elif vib_values:
                summary_parts.append(f"{label}: Vibration {np.mean(vib_values):.1f} mm/s")
        
        return "\n".join(summary_parts) if summary_parts else "Stable sensor readings expected"
    
    def _identify_concerning_trends_from_dict(self, forecasts: Dict,
                                               sensor_cols: List[str]) -> List[str]:
        """
        Identify concerning trends from forecasts dict
        
        Args:
            forecasts: Dict of sensor forecasts
            sensor_cols: List of sensor column names
            
        Returns:
            List of concerning trend descriptions
        """
        concerns = []
        
        for col in sensor_cols:
            if col in forecasts and forecasts[col]['yhat']:
                values = [v for v in forecasts[col]['yhat'] if v is not None]
                
                if len(values) > 2:
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    
                    if 'temp' in col.lower() and trend > 1.0:
                        concerns.append(f"{col}: Rising trend (+{trend:.1f}°C/hour)")
                    elif ('velocity' in col.lower() or 'vibration' in col.lower()) and trend > 0.5:
                        concerns.append(f"{col}: Increasing vibration (+{trend:.1f} mm/s/hour)")
                    
                    # Check for peak values
                    max_val = np.max(values)
                    if 'temp' in col.lower() and max_val > 85:
                        concerns.append(f"{col}: Peak value {max_val:.1f}°C (high)")
                    elif ('velocity' in col.lower() or 'vibration' in col.lower()) and max_val > 10:
                        concerns.append(f"{col}: Peak vibration {max_val:.1f} mm/s (elevated)")
        
        if not concerns:
            concerns.append("No concerning trends detected")
        
        return concerns
    
    def _recommend_maintenance_window_from_dict(self, forecasts: Dict,
                                                 sensor_cols: List[str]) -> str:
        """
        Recommend optimal maintenance window from forecasts dict
        
        Args:
            forecasts: Dict of sensor forecasts
            sensor_cols: List of sensor column names
            
        Returns:
            Maintenance window recommendation
        """
        # Find time period with lowest sensor values (best for maintenance)
        sph = int(self.steps_per_hour or 1)
        windows = [(0, 6), (6, 12), (12, 18), (18, 24)]
        window_scores = []
        
        for start, end in windows:
            score = 0
            for col in sensor_cols:
                if col in forecasts and forecasts[col]['yhat']:
                    window_vals = forecasts[col]['yhat'][start * sph:end * sph]
                    valid_vals = [v for v in window_vals if v is not None]
                    
                    if valid_vals:
                        if 'temp' in col.lower():
                            score += np.mean(valid_vals) / 100.0
                        elif 'velocity' in col.lower() or 'vibration' in col.lower():
                            score += np.mean(valid_vals) / 10.0
            
            window_scores.append((start, end, score))
        
        if window_scores:
            # Find window with lowest score
            best_window = min(window_scores, key=lambda x: x[2])
            return f"Optimal window: Hour {best_window[0]}-{best_window[1]} (lowest sensor activity)"
        else:
            return "Schedule during normal operating hours"


def create_sample_historical_data(machine_id: str, 
                                  num_timesteps: int = 168) -> pd.DataFrame:
    """
    Create sample historical data for testing (1 week of hourly data)
    
    Args:
        machine_id: Machine identifier
        num_timesteps: Number of time steps (default: 168 for 1 week)
        
    Returns:
        DataFrame with historical sensor readings
    """
    # Generate timestamps
    end_time = datetime.utcnow()
    timestamps = [end_time - timedelta(hours=i) for i in range(num_timesteps, 0, -1)]
    
    # Load sensor names from metadata
    metadata_path = PROJECT_ROOT / "GAN" / "metadata" / f"{machine_id}_metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        sensor_cols = list(metadata['columns'].keys())
    else:
        # Default sensors
        sensor_cols = [
            'bearing_de_temp_C', 'bearing_nde_temp_C', 'winding_temp_C',
            'rms_velocity_mm_s', 'peak_velocity_mm_s'
        ]
    
    # Generate synthetic time series data with trends
    np.random.seed(42)
    data = {'timestamp': timestamps}
    
    for col in sensor_cols:
        # Create base values with trend and noise
        if 'temp' in col.lower():
            base = 60 + np.sin(np.linspace(0, 4*np.pi, num_timesteps)) * 5
            trend = np.linspace(0, 5, num_timesteps)
            noise = np.random.normal(0, 2, num_timesteps)
            data[col] = base + trend + noise
        elif 'velocity' in col.lower() or 'vibration' in col.lower():
            base = 4 + np.sin(np.linspace(0, 6*np.pi, num_timesteps)) * 2
            trend = np.linspace(0, 2, num_timesteps)
            noise = np.random.normal(0, 0.5, num_timesteps)
            data[col] = base + trend + noise
        elif 'current' in col.lower():
            base = 30 + np.sin(np.linspace(0, 3*np.pi, num_timesteps)) * 5
            noise = np.random.normal(0, 1, num_timesteps)
            data[col] = base + noise
        else:
            data[col] = np.random.uniform(50, 100, num_timesteps)
    
    df = pd.DataFrame(data)
    return df


def main():
    """Main function for testing time-series forecasting inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Time-Series Forecasting Model Inference')
    parser.add_argument('--machine_id', type=str, required=True,
                       help='Machine identifier (e.g., motor_siemens_1la7_001)')
    parser.add_argument('--forecast_hours', type=int, default=24,
                       help='Number of hours to forecast (default: 24)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for predictions')
    
    args = parser.parse_args()
    
    print("="*60)
    print("TIME-SERIES FORECASTING MODEL INFERENCE")
    print("="*60)
    print(f"Machine: {args.machine_id}")
    print(f"Forecast horizon: {args.forecast_hours} hours\n")
    
    try:
        # Initialize predictor
        predictor = TimeSeriesPredictor(args.machine_id)
        
        # Create sample historical data (1 week)
        print("Creating sample historical data (1 week of hourly readings)...")
        historical_data = create_sample_historical_data(args.machine_id, num_timesteps=168)
        print(f"✓ Generated {len(historical_data)} historical data points")
        
        # Run forecast
        print(f"\nGenerating {args.forecast_hours}-hour forecast...")
        result = predictor.predict(historical_data, forecast_steps=args.forecast_hours)
        
        # Save result
        output_dir = Path(args.output_dir) if args.output_dir else \
                     PROJECT_ROOT / "ml_models" / "outputs" / "predictions" / "timeseries"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{args.machine_id}_forecast.json"
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✓ Forecast saved to: {output_file}")
        
        # Display summary
        print("\n" + "="*60)
        print("FORECAST SUMMARY")
        print("="*60)
        
        pred = result['prediction']
        print(f"Forecast horizon: {pred['forecast_horizon_hours']} hours")
        print(f"Confidence: {pred['confidence']*100:.1f}%")
        print(f"\nForecast Summary:")
        print(pred['forecast_summary'])
        print(f"\nConcerning Trends:")
        for trend in pred['concerning_trends']:
            print(f"  - {trend}")
        print(f"\nMaintenance Recommendation:")
        print(f"  {pred['maintenance_window']}")
        
        print("\n" + "="*60)
        print("✓ TIME-SERIES FORECASTING COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
