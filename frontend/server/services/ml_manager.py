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
LLM_PATH = PROJECT_ROOT / "LLM"
sys.path.insert(0, str(LLM_PATH))


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
            from api.ml_integration import IntegratedPredictionSystem
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
        
        # Allow predictions even if GAN metadata is missing, as long as model artifacts exist.
        classification_path = self.models_dir / "classification" / machine_id
        if not self._has_model_artifacts(classification_path):
            raise FileNotFoundError(
                f"Classification model not found or empty for {machine_id}. Expected: {classification_path}"
            )
        
        logger.info(f"Running classification prediction for {machine_id}")
        
        try:
            # Run prediction with explanation
            result = self.integrated_system.predict_with_explanation(
                machine_id=machine_id,
                sensor_data=sensor_data,
                model_type='classification'
            )
            
            classification_block = result.get('classification', {}) if isinstance(result, dict) else {}
            if isinstance(classification_block, dict) and classification_block.get('error'):
                raise RuntimeError(classification_block.get('error'))

            pred = classification_block.get('prediction', {}) if isinstance(classification_block, dict) else {}
            explanation_block = classification_block.get('explanation', {}) if isinstance(classification_block, dict) else {}
            explanation_text = (
                explanation_block.get('summary')
                if isinstance(explanation_block, dict)
                else (classification_block.get('explanation') if isinstance(classification_block, dict) else '')
            )

            return ClassificationResult(
                machine_id=machine_id,
                failure_type=pred.get('failure_type', 'unknown'),
                confidence=float(pred.get('confidence', 0.0) or 0.0),
                failure_probability=float(pred.get('failure_probability', 0.0) or 0.0),
                all_probabilities=pred.get('all_probabilities', {}) or {},
                explanation=explanation_text or '',
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
            # Run prediction with explanation
            result = self.integrated_system.predict_with_explanation(
                machine_id=machine_id,
                sensor_data=sensor_data,
                model_type='regression'
            )
            
            regression_block = result.get('regression', {}) if isinstance(result, dict) else {}
            if isinstance(regression_block, dict) and regression_block.get('error'):
                raise RuntimeError(regression_block.get('error'))

            pred = regression_block.get('prediction', {}) if isinstance(regression_block, dict) else {}
            explanation_block = regression_block.get('explanation', {}) if isinstance(regression_block, dict) else {}
            explanation_text = (
                explanation_block.get('summary')
                if isinstance(explanation_block, dict)
                else (regression_block.get('explanation') if isinstance(regression_block, dict) else '')
            )

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
                explanation=explanation_text or ''
            )
        except Exception as e:
            logger.error(f"RUL prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def get_health(self) -> Dict[str, Any]:
        """
        Get ML service health status
        
        Returns:
            Dictionary with service health information
        """
        health = {
            'status': 'healthy' if self.integrated_system else 'degraded',
            'ml_system_available': self.integrated_system is not None,
            'models_directory': str(self.models_dir),
            'models_directory_exists': self.models_dir.exists(),
            'machines_loaded': len(self.machine_metadata),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add model counts if available
        if self.models_dir.exists():
            classification_models = list((self.models_dir / "classification").glob("*"))
            regression_models = list((self.models_dir / "regression").glob("*"))
            
            health['model_counts'] = {
                'classification': len([m for m in classification_models if m.is_dir()]),
                'regression': len([m for m in regression_models if m.is_dir()]),
                'anomaly': 0,  # Not yet implemented
                'timeseries': 0  # Not yet implemented
            }
        
        # Add GPU info if IntegratedPredictionSystem loaded
        if self.integrated_system:
            try:
                import torch
                health['gpu_available'] = torch.cuda.is_available()
                if torch.cuda.is_available():
                    health['gpu_info'] = {
                        'name': torch.cuda.get_device_name(0),
                        'cuda_version': torch.version.cuda
                    }
            except:
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
