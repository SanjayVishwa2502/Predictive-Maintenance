"""
ML Manager Service - Phase 3.7.3 Day 15.1
FastAPI service layer for ML predictions and machine monitoring

Integrates with:
- IntegratedPredictionSystem (LLM/api/ml_integration.py)
- Phase 2 ML models (classification, RUL, anomaly, timeseries)
- Machine metadata from GAN/metadata/

Provides:
- Single-machine predictions
- Model metadata retrieval
- Health status monitoring
- Graceful error handling
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Ensure the repo root is importable so we can import `LLM.*` without
# colliding with the backend's own `api.*` package.
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ClassificationResult:
    """Classification prediction result"""
    machine_id: str
    failure_type: str
    confidence: float
    failure_probability: float
    all_probabilities: Dict[str, float]
    explanation: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class RULResult:
    """Remaining Useful Life prediction result"""
    machine_id: str
    rul_hours: float
    rul_days: float
    urgency: str
    confidence: float
    explanation: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class AnomalyResult:
    """Anomaly detection result"""
    machine_id: str
    is_anomaly: bool
    anomaly_score: float
    detection_method: str
    abnormal_sensors: Dict[str, float]
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class TimeSeriesResult:
    """Time-series forecast result"""
    machine_id: str
    forecast_summary: str
    confidence: float
    forecast_horizon: str
    forecasts: Dict[str, Any]
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class MachineInfo:
    """Machine metadata"""
    machine_id: str
    display_name: str
    category: str
    manufacturer: str
    model: str
    sensor_count: int
    has_classification_model: bool
    has_regression_model: bool
    has_anomaly_model: bool
    has_timeseries_model: bool


class MLManager:
    """
    Centralized ML model management service
    
    Singleton pattern for efficient resource management.
    Loads IntegratedPredictionSystem on demand.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize ML Manager (singleton)"""
        if MLManager._initialized:
            return
        
        MLManager._initialized = True
        logger.info("Initializing MLManager...")
        
        # Paths
        self.project_root = PROJECT_ROOT
        self.models_dir = self.project_root / "ml_models" / "models"
        self.gan_metadata_dir = self.project_root / "GAN" / "metadata"
        
        # IntegratedPredictionSystem (lazy loaded)
        self.integrated_system = None
        self._load_integrated_system()
        
        # Machine metadata cache
        self.machine_metadata: Dict[str, Dict] = {}
        self._load_machine_metadata()
        
        logger.info(f"[OK] Loaded metadata for {len(self.machine_metadata)} machines")
        logger.info("[OK] MLManager initialized successfully")
    
    def _load_integrated_system(self):
        """Load IntegratedPredictionSystem (LLM + ML models)"""
        try:
            from LLM.api.ml_integration import IntegratedPredictionSystem
            self.integrated_system = IntegratedPredictionSystem()
            logger.info("[OK] IntegratedPredictionSystem loaded successfully")
        except Exception as e:
            logger.warning(f"[WARN] Failed to load IntegratedPredictionSystem: {e}")
            logger.warning("[WARN] ML predictions will not be available. Server will continue with limited functionality.")
            self.integrated_system = None
    
    def _load_machine_metadata(self):
        """Load machine metadata from GAN/metadata/"""
        try:
            if not self.gan_metadata_dir.exists():
                logger.warning(f"Metadata directory not found: {self.gan_metadata_dir}")
                return
            
            # Only load enriched ML metadata files.
            # The folder can also contain transient profile staging files (e.g. *_profile_temp.json)
            # and raw profiles; those should not appear in the ML machine selector.
            for metadata_file in self.gan_metadata_dir.glob("*_metadata.json"):
                try:
                    # Extract machine_id from filename (e.g., "motor_siemens_1la7_001_metadata.json" -> "motor_siemens_1la7_001")
                    machine_id = metadata_file.stem.replace('_metadata', '')
                    
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        
                        # Extract sensor list from columns
                        sensors = []
                        if 'columns' in metadata:
                            # Exclude non-sensor / target columns from sensor inventory.
                            # Some metadata files include fields like timestamp or RUL targets.
                            skip_names = {
                                'timestamp', 'time', 'datetime', 'date',
                                'rul', 'RUL', 'remaining_useful_life',
                                'machine_id', 'id', 'split', 'label', 'target',
                            }
                            sensors = [
                                {
                                    'name': col_name,
                                    'type': col_data.get('sdtype', 'numerical'),
                                    'unit': self._infer_unit_from_name(col_name)
                                }
                                for col_name, col_data in metadata['columns'].items()
                                if col_name not in skip_names
                                and col_data.get('sdtype') not in {'datetime', 'id'}
                            ]
                        
                        # Build enriched metadata
                        enriched_metadata = {
                            'machine_id': machine_id,
                            'display_name': self._format_display_name(machine_id),
                            'category': self._infer_category(machine_id),
                            'manufacturer': self._extract_manufacturer(machine_id),
                            'model': self._extract_model(machine_id),
                            'sensors': sensors,
                            'raw_metadata': metadata
                        }
                        
                        self.machine_metadata[machine_id] = enriched_metadata
                        
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {metadata_file.name}: {e}")
        except Exception as e:
            logger.error(f"Error loading machine metadata: {e}")
    
    def _format_display_name(self, machine_id: str) -> str:
        """Convert machine_id to readable display name"""
        # Convert underscores to spaces and title case
        parts = machine_id.split('_')
        return ' '.join(word.upper() if len(word) <= 3 else word.title() for word in parts)
    
    def _infer_category(self, machine_id: str) -> str:
        """Infer machine category from machine_id"""
        category_map = {
            'motor': 'motor',
            'pump': 'pump',
            'compressor': 'compressor',
            'fan': 'fan',
            'cnc': 'cnc',
            'robot': 'robot',
            'conveyor': 'conveyor',
            'hydraulic': 'hydraulic',
            'cooling': 'cooling_system',
            'transformer': 'transformer',
            'turbofan': 'turbine',
            'induction': 'motor'
        }
        
        for key, value in category_map.items():
            if machine_id.startswith(key):
                return value
        
        return 'unknown'
    
    def _extract_manufacturer(self, machine_id: str) -> str:
        """Extract manufacturer from machine_id"""
        parts = machine_id.split('_')
        if len(parts) >= 2:
            # Second part is usually manufacturer
            manufacturer = parts[1].upper()
            # Handle multi-word manufacturers
            if len(parts) >= 3 and parts[2].lower() in ['copco', 'papst', 'mori', 'rand']:
                manufacturer += ' ' + parts[2].title()
            return manufacturer
        return 'Unknown'
    
    def _extract_model(self, machine_id: str) -> str:
        """Extract model from machine_id"""
        parts = machine_id.split('_')
        # Model is typically after manufacturer
        if len(parts) >= 3:
            # Find where the model starts (after manufacturer)
            model_parts = []
            for i, part in enumerate(parts[2:], start=2):
                if part.isdigit() or part.startswith('00'):
                    break
                model_parts.append(part.upper())
            return ' '.join(model_parts) if model_parts else 'Unknown'
        return 'Unknown'
    
    def _infer_unit_from_name(self, sensor_name: str) -> str:
        """Infer measurement unit from sensor name"""
        sensor_lower = sensor_name.lower()
        
        # Extract unit from sensor name (e.g., "bearing_temp_C" -> "°C")
        if '_c' in sensor_lower and sensor_name.endswith('_C'):
            return '°C'
        elif '_f' in sensor_lower and sensor_name.endswith('_F'):
            return '°F'
        elif '_mm_s' in sensor_lower:
            return 'mm/s'
        elif '_hz' in sensor_lower:
            return 'Hz'
        elif '_a' in sensor_lower and sensor_name.endswith('_A'):
            return 'A'
        elif '_v' in sensor_lower and sensor_name.endswith('_V'):
            return 'V'
        elif '_bar' in sensor_lower:
            return 'bar'
        elif '_psi' in sensor_lower:
            return 'psi'
        elif '_rpm' in sensor_lower:
            return 'RPM'
        elif '_dba' in sensor_lower:
            return 'dBA'
        elif '_m3_h' in sensor_lower:
            return 'm³/h'
        elif '_kw' in sensor_lower:
            return 'kW'
        elif 'pct' in sensor_lower or 'percent' in sensor_lower:
            return '%'
        
        return ''

    def _has_model_artifacts(self, model_path: Path) -> bool:
        """Best-effort check that a model directory exists and is non-empty."""
        try:
            return model_path.is_dir() and any(model_path.iterdir())
        except Exception:
            return model_path.is_dir()

    def _discover_machine_ids_from_models_dir(self) -> List[str]:
        """Return machine ids that have at least one model directory on disk."""
        machine_ids = set()
        for model_type in ("classification", "regression", "anomaly", "timeseries"):
            type_dir = self.models_dir / model_type
            if not type_dir.exists():
                continue
            try:
                for entry in type_dir.iterdir():
                    if entry.is_dir():
                        machine_ids.add(entry.name)
            except Exception:
                continue
        return sorted(machine_ids)
    
    def get_machines(self) -> List[MachineInfo]:
        """
        Get list of all available machines with model status
        
        Returns:
            List of MachineInfo objects
        """
        machines: List[MachineInfo] = []

        # Include both metadata-defined machines and machines that only exist on disk.
        machine_ids = set(self.machine_metadata.keys())
        machine_ids.update(self._discover_machine_ids_from_models_dir())

        for machine_id in sorted(machine_ids):
            metadata = self.machine_metadata.get(machine_id, {})

            classification_path = self.models_dir / "classification" / machine_id
            regression_path = self.models_dir / "regression" / machine_id
            anomaly_path = self.models_dir / "anomaly" / machine_id
            timeseries_path = self.models_dir / "timeseries" / machine_id

            sensors = metadata.get("sensors", [])
            sensor_count = len(sensors) if sensors else 1  # Ensure at least 1 for Pydantic validation

            display_name = metadata.get("display_name") or self._format_display_name(machine_id)
            category = metadata.get("category") or self._infer_category(machine_id)
            manufacturer = metadata.get("manufacturer") or self._extract_manufacturer(machine_id)
            model = metadata.get("model") or self._extract_model(machine_id)

            machine_info = MachineInfo(
                machine_id=machine_id,
                display_name=display_name,
                category=category,
                manufacturer=manufacturer,
                model=model,
                sensor_count=sensor_count,
                has_classification_model=self._has_model_artifacts(classification_path),
                has_regression_model=self._has_model_artifacts(regression_path),
                has_anomaly_model=self._has_model_artifacts(anomaly_path),
                has_timeseries_model=self._has_model_artifacts(timeseries_path),
            )
            machines.append(machine_info)

        return machines
    
    def get_machine_info(self, machine_id: str) -> Optional[MachineInfo]:
        """
        Get information for a specific machine
        
        Args:
            machine_id: Machine identifier
            
        Returns:
            MachineInfo object or None if not found
        """
        metadata = self.machine_metadata.get(machine_id)

        classification_path = self.models_dir / "classification" / machine_id
        regression_path = self.models_dir / "regression" / machine_id
        anomaly_path = self.models_dir / "anomaly" / machine_id
        timeseries_path = self.models_dir / "timeseries" / machine_id

        # If we have neither metadata nor any model artifacts, treat as unknown.
        if metadata is None and not any(
            self._has_model_artifacts(p)
            for p in (classification_path, regression_path, anomaly_path, timeseries_path)
        ):
            return None

        metadata = metadata or {}

        return MachineInfo(
            machine_id=machine_id,
            display_name=metadata.get("display_name") or self._format_display_name(machine_id),
            category=metadata.get("category") or self._infer_category(machine_id),
            manufacturer=metadata.get("manufacturer") or self._extract_manufacturer(machine_id),
            model=metadata.get("model") or self._extract_model(machine_id),
            sensor_count=len(metadata.get("sensors", [])) or 1,
            has_classification_model=self._has_model_artifacts(classification_path),
            has_regression_model=self._has_model_artifacts(regression_path),
            has_anomaly_model=self._has_model_artifacts(anomaly_path),
            has_timeseries_model=self._has_model_artifacts(timeseries_path),
        )
    
    def predict_classification(
        self, 
        machine_id: str, 
        sensor_data: Optional[Dict[str, float]]
    ) -> ClassificationResult:
        """
        Run classification prediction for a machine
        
        Args:
            machine_id: Machine identifier
            sensor_data: Dictionary of sensor readings
            
        Returns:
            ClassificationResult with prediction and explanation
            
        Raises:
            RuntimeError: If IntegratedPredictionSystem not available
            ValueError: If machine_id not found
        """
        if self.integrated_system is None:
            raise RuntimeError(
                "IntegratedPredictionSystem not available. "
                "ML predictions are disabled. Check server logs for details."
            )
        
        classification_path = self.models_dir / "classification" / machine_id
        has_model = self._has_model_artifacts(classification_path)
        
        logger.info(f"Running classification prediction for {machine_id} (model={'yes' if has_model else 'no'})")
        
        try:
            # If we have a trained model, use it. Otherwise, fall back to a lightweight heuristic
            # so the frontend doesn't hard-fail for machines without models.
            if has_model:
                # NOTE: Keep this endpoint fast. LLM explanations are generated separately
                # via /api/llm/explain (async Celery worker) and shown in the UI.
                pred = self.integrated_system.predict_classification(
                    machine_id=machine_id,
                    sensor_data=sensor_data,
                )
                return ClassificationResult(
                    machine_id=machine_id,
                    failure_type=pred.get('failure_type', 'unknown'),
                    confidence=float(pred.get('confidence', 0.0) or 0.0),
                    failure_probability=float(pred.get('failure_probability', 0.0) or 0.0),
                    all_probabilities=pred.get('all_probabilities', {}) or {},
                    explanation='',
                )

            # ---------- Fallback path (no model artifacts) ----------
            values = sensor_data or {}
            # Heuristic type detection based on common sensor names.
            sensor_lower = {str(k).lower(): float(v) for k, v in values.items() if v is not None}

            temp_vals = [v for k, v in sensor_lower.items() if 'temp' in k]
            vib_vals = [v for k, v in sensor_lower.items() if 'vib' in k or 'vibration' in k]
            elec_vals = [v for k, v in sensor_lower.items() if 'current' in k or 'amp' in k or 'voltage' in k]

            max_val = max([abs(v) for v in sensor_lower.values()], default=0.0)
            # Simple bounded risk score in [0.05, 0.95]
            failure_probability = max(0.05, min(0.95, max_val / (max_val + 50.0) if max_val > 0 else 0.1))

            failure_type = 'normal'
            if temp_vals and max(temp_vals) >= 80:
                failure_type = 'overheating'
            elif vib_vals and max(vib_vals) >= 10:
                failure_type = 'bearing_wear'
            elif elec_vals and max(elec_vals) >= 20:
                failure_type = 'electrical_fault'
            elif failure_probability >= 0.35:
                failure_type = 'bearing_wear'

            # Distribute probabilities across known labels.
            base = {
                'normal': max(0.0, 1.0 - failure_probability),
                'bearing_wear': 0.0,
                'overheating': 0.0,
                'electrical_fault': 0.0,
            }
            base[failure_type] = max(base.get(failure_type, 0.0), failure_probability)
            s = sum(base.values()) or 1.0
            all_probabilities = {k: float(v / s) for k, v in base.items()}

            confidence = 0.55
            explanation_text = (
                f"Heuristic prediction (no trained model found for {machine_id}).\n\n"
                f"Predicted failure type: {failure_type}\n"
                f"Failure probability: {failure_probability:.2f}\n"
                "LLM explanation is generated separately."
            )

            return ClassificationResult(
                machine_id=machine_id,
                failure_type=failure_type,
                confidence=confidence,
                failure_probability=float(all_probabilities.get(failure_type, failure_probability)),
                all_probabilities=all_probabilities,
                explanation=explanation_text,
            )
            
        except Exception as e:
            logger.error(f"Classification prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_rul(
        self, 
        machine_id: str, 
        sensor_data: Optional[Dict[str, float]]
    ) -> RULResult:
        """
        Run Remaining Useful Life prediction for a machine
        
        Args:
            machine_id: Machine identifier
            sensor_data: Dictionary of sensor readings
            
        Returns:
            RULResult with prediction and explanation
            
        Raises:
            RuntimeError: If IntegratedPredictionSystem not available
            ValueError: If machine_id not found
        """
        if self.integrated_system is None:
            raise RuntimeError(
                "IntegratedPredictionSystem not available. "
                "ML predictions are disabled. Check server logs for details."
            )
        
        # Allow predictions even if GAN metadata is missing, as long as model artifacts exist.
        regression_path = self.models_dir / "regression" / machine_id
        if not self._has_model_artifacts(regression_path):
            raise FileNotFoundError(
                f"RUL model not found or empty for {machine_id}. Expected: {regression_path}"
            )
        
        logger.info(f"Running RUL prediction for {machine_id}")
        
        try:
            # IMPORTANT: do NOT generate LLM explanations in the request path.
            # Explanations are generated asynchronously via Celery (/api/llm/explain).
            pred = self.integrated_system.predict_rul(machine_id=machine_id, sensor_data=sensor_data)

            rul_hours = float(pred.get('rul_hours', 0.0) or 0.0)
            rul_days = rul_hours / 24.0
            
            # Determine urgency
            if rul_hours < 24:
                urgency = "critical"
            elif rul_hours < 72:
                urgency = "high"
            elif rul_hours < 168:
                urgency = "medium"
            else:
                urgency = "low"
            
            return RULResult(
                machine_id=machine_id,
                rul_hours=rul_hours,
                rul_days=round(rul_days, 2),
                urgency=urgency,
                confidence=float(pred.get('confidence', 0.0) or 0.0),
                explanation=''
            )
        except Exception as e:
            logger.error(f"RUL prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_anomaly(
        self,
        machine_id: str,
        sensor_data: Optional[Dict[str, float]],
    ) -> AnomalyResult:
        """Run anomaly detection (prediction-only)."""
        if self.integrated_system is None:
            raise RuntimeError(
                "IntegratedPredictionSystem not available. "
                "ML predictions are disabled. Check server logs for details."
            )

        anomaly_path = self.models_dir / "anomaly" / machine_id
        has_model = self._has_model_artifacts(anomaly_path)
        logger.info(f"Running anomaly detection for {machine_id} (model={'yes' if has_model else 'no'})")

        try:
            if not has_model:
                return AnomalyResult(
                    machine_id=machine_id,
                    is_anomaly=False,
                    anomaly_score=0.0,
                    detection_method="unavailable",
                    abnormal_sensors={},
                )

            pred = self.integrated_system.detect_anomaly(machine_id=machine_id, sensor_data=sensor_data)
            return AnomalyResult(
                machine_id=machine_id,
                is_anomaly=bool(pred.get("is_anomaly", False)),
                anomaly_score=float(pred.get("score", 0.0) or 0.0),
                detection_method=str(pred.get("method") or "unknown"),
                abnormal_sensors=pred.get("abnormal_sensors") or {},
            )
        except Exception as e:
            logger.error(f"Anomaly prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_timeseries(
        self,
        machine_id: str,
        sensor_data: Optional[Dict[str, float]],
    ) -> TimeSeriesResult:
        """Run time-series forecast (prediction-only)."""
        if self.integrated_system is None:
            raise RuntimeError(
                "IntegratedPredictionSystem not available. "
                "ML predictions are disabled. Check server logs for details."
            )

        timeseries_path = self.models_dir / "timeseries" / machine_id
        has_model = self._has_model_artifacts(timeseries_path)
        logger.info(f"Running timeseries forecast for {machine_id} (model={'yes' if has_model else 'no'})")

        try:
            if not has_model:
                return TimeSeriesResult(
                    machine_id=machine_id,
                    forecast_summary="Time-series model not available for this machine.",
                    confidence=0.0,
                    forecast_horizon="",
                    forecasts={},
                )

            pred = self.integrated_system.predict_forecast(machine_id=machine_id, sensor_data=sensor_data)
            return TimeSeriesResult(
                machine_id=machine_id,
                forecast_summary=str(pred.get("forecast_summary") or "Forecast generated"),
                confidence=float(pred.get("confidence", 0.0) or 0.0),
                forecast_horizon=str(pred.get("forecast_horizon") or ""),
                forecasts=pred.get("forecasts") or {},
            )
        except Exception as e:
            logger.error(f"Timeseries prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def get_health(self) -> Dict[str, Any]:
        """
        Get ML service health status
        
        Returns:
            Dictionary with service health information
        """
        integrated_ready = self.integrated_system is not None
        health: Dict[str, Any] = {
            'status': 'healthy' if integrated_ready else 'degraded',
            'models_loaded': {
                'classification': 0,
                'regression': 0,
                'anomaly': 0,
                'timeseries': 0,
            },
            'llm_status': 'available' if integrated_ready else 'unavailable',
            'gpu_available': False,
            'gpu_info': None,
            'integrated_system_ready': integrated_ready,
            'models_directory': str(self.models_dir),
            'models_directory_exists': self.models_dir.exists(),
            'machines_loaded': len(self.machine_metadata),
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add model counts if available
        if self.models_dir.exists():
            classification_models = list((self.models_dir / "classification").glob("*"))
            regression_models = list((self.models_dir / "regression").glob("*"))
            anomaly_models = list((self.models_dir / "anomaly").glob("*"))
            timeseries_models = list((self.models_dir / "timeseries").glob("*"))

            health['models_loaded'] = {
                'classification': len([m for m in classification_models if m.is_dir()]),
                'regression': len([m for m in regression_models if m.is_dir()]),
                'anomaly': len([m for m in anomaly_models if m.is_dir()]),
                'timeseries': len([m for m in timeseries_models if m.is_dir()]),
            }
        
        # Add GPU info if available
        try:
            import torch
            health['gpu_available'] = bool(torch.cuda.is_available())
            if health['gpu_available']:
                health['gpu_info'] = {
                    'name': torch.cuda.get_device_name(0),
                    'cuda_version': torch.version.cuda,
                }
        except Exception:
            pass
        
        return health
    
    def reload_metadata(self):
        """Reload machine metadata from disk"""
        logger.info("Reloading machine metadata...")
        self.machine_metadata.clear()
        self._load_machine_metadata()
        logger.info(f"[OK] Reloaded metadata for {len(self.machine_metadata)} machines")


# Create singleton instance
ml_manager = MLManager()
