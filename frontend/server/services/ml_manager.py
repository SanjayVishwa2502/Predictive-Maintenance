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
    
    def get_machines(self) -> List[MachineInfo]:
        """
        Get list of all available machines with model status
        
        Returns:
            List of MachineInfo objects
        """
        machines = []
        
        # Get machines from metadata
        for machine_id, metadata in self.machine_metadata.items():
            # Check which models exist
            classification_path = self.models_dir / "classification" / machine_id
            regression_path = self.models_dir / "regression" / machine_id
            
            sensors = metadata.get('sensors', [])
            sensor_count = len(sensors) if sensors else 1  # Ensure at least 1 for Pydantic validation
            
            machine_info = MachineInfo(
                machine_id=machine_id,
                display_name=metadata.get('display_name', machine_id),
                category=metadata.get('category', 'unknown'),
                manufacturer=metadata.get('manufacturer', 'Unknown'),
                model=metadata.get('model', 'Unknown'),
                sensor_count=sensor_count,
                has_classification_model=classification_path.exists(),
                has_regression_model=regression_path.exists(),
                has_anomaly_model=False,  # Not yet implemented
                has_timeseries_model=False  # Not yet implemented
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
        if machine_id not in self.machine_metadata:
            return None
        
        metadata = self.machine_metadata[machine_id]
        classification_path = self.models_dir / "classification" / machine_id
        regression_path = self.models_dir / "regression" / machine_id
        
        return MachineInfo(
            machine_id=machine_id,
            display_name=metadata.get('display_name', machine_id),
            category=metadata.get('category', 'unknown'),
            manufacturer=metadata.get('manufacturer', 'Unknown'),
            model=metadata.get('model', 'Unknown'),
            sensor_count=len(metadata.get('sensors', [])),
            has_classification_model=classification_path.exists(),
            has_regression_model=regression_path.exists(),
            has_anomaly_model=False,
            has_timeseries_model=False
        )
    
    def predict_classification(
        self, 
        machine_id: str, 
        sensor_data: Dict[str, float]
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
        
        if machine_id not in self.machine_metadata:
            raise ValueError(f"Machine not found: {machine_id}")
        
        logger.info(f"Running classification prediction for {machine_id}")
        
        try:
            # Run prediction with explanation
            result = self.integrated_system.predict_with_explanation(
                machine_id=machine_id,
                sensor_data=sensor_data,
                model_type='classification'
            )
            
            # Extract classification results
            pred = result.get('prediction', {})
            classification = pred.get('classification', {})
            
            return ClassificationResult(
                machine_id=machine_id,
                failure_type=classification.get('health_state', 'unknown'),
                confidence=classification.get('confidence', 0.0),
                all_probabilities=classification.get('probabilities', {}),
                explanation=result.get('explanation', '')
            )
        except Exception as e:
            logger.error(f"Classification prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_rul(
        self, 
        machine_id: str, 
        sensor_data: Dict[str, float]
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
        
        if machine_id not in self.machine_metadata:
            raise ValueError(f"Machine not found: {machine_id}")
        
        logger.info(f"Running RUL prediction for {machine_id}")
        
        try:
            # Run prediction with explanation
            result = self.integrated_system.predict_with_explanation(
                machine_id=machine_id,
                sensor_data=sensor_data,
                model_type='regression'
            )
            
            # Extract RUL results
            pred = result.get('prediction', {})
            rul = pred.get('rul', {})
            
            rul_hours = rul.get('predicted_rul', 0.0)
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
                confidence=rul.get('confidence', 0.0),
                explanation=result.get('explanation', '')
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
