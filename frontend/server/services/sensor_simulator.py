"""
Sensor Data Simulator Service

Simulates real-time sensor data streaming by reading from validation datasets.
Iterates through val.parquet files row-by-row to simulate time-based sensor readings.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import pandas as pd
from threading import Lock

logger = logging.getLogger(__name__)


class SensorSimulator:
    """
    Manages simulated sensor data streaming for multiple machines.
    
    Uses validation datasets (val.parquet) to avoid contamination with training data.
    Each machine maintains its own iteration state.
    """
    
    def __init__(self, synthetic_data_path: str = "GAN/data/synthetic"):
        """
        Initialize the sensor simulator.
        
        Args:
            synthetic_data_path: Path to synthetic data directory
        """
        # Get absolute path relative to the project root
        server_dir = Path(__file__).resolve().parents[1]  # Go up to server/
        project_root = server_dir.parent.parent  # Go up to frontend/, then to project root
        self.synthetic_data_path = project_root / synthetic_data_path
        self._simulations: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        
        logger.info(f"SensorSimulator initialized with data path: {self.synthetic_data_path}")
    
    def start_simulation(self, machine_id: str) -> bool:
        """
        Start sensor simulation for a machine.
        
        Args:
            machine_id: Machine identifier
            
        Returns:
            True if simulation started successfully, False otherwise
        """
        with self._lock:
            try:
                # Check if already simulating
                if machine_id in self._simulations:
                    logger.warning(f"Simulation already running for {machine_id}")
                    return True
                
                # Load validation dataset
                val_path = self.synthetic_data_path / machine_id / "val.parquet"
                
                if not val_path.exists():
                    logger.error(f"Validation dataset not found: {val_path}")
                    return False
                
                # Load parquet file
                df = pd.read_parquet(val_path)
                
                if df.empty:
                    logger.error(f"Empty validation dataset for {machine_id}")
                    return False
                
                # Remove non-sensor columns if they exist
                exclude_cols = ['timestamp', 'machine_id', 'rul', 'failure_type', 'health_state']
                sensor_cols = [col for col in df.columns if col not in exclude_cols]
                
                if not sensor_cols:
                    logger.error(f"No sensor columns found in dataset for {machine_id}")
                    return False
                
                # Initialize simulation state
                self._simulations[machine_id] = {
                    'dataframe': df[sensor_cols],
                    'current_index': 0,
                    'total_rows': len(df),
                    'sensor_columns': sensor_cols,
                    'started_at': datetime.now(),
                    'last_update': datetime.now(),
                    'cycle_count': 0  # Track how many times we've looped through the dataset
                }
                
                logger.info(
                    f"[OK] Started simulation for {machine_id}: "
                    f"{len(sensor_cols)} sensors, {len(df)} samples"
                )
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to start simulation for {machine_id}: {e}", exc_info=True)
                return False
    
    def stop_simulation(self, machine_id: str) -> bool:
        """
        Stop sensor simulation for a machine.
        
        Args:
            machine_id: Machine identifier
            
        Returns:
            True if simulation stopped, False if not running
        """
        with self._lock:
            if machine_id in self._simulations:
                del self._simulations[machine_id]
                logger.info(f"[OK] Stopped simulation for {machine_id}")
                return True
            else:
                logger.warning(f"No active simulation for {machine_id}")
                return False
    
    def get_current_readings(self, machine_id: str) -> Optional[Dict[str, float]]:
        """
        Get current sensor readings for a machine.
        
        Returns the current row and advances to the next row for the next call.
        Automatically loops back to the beginning when reaching the end of the dataset.
        
        Args:
            machine_id: Machine identifier
            
        Returns:
            Dictionary of sensor readings or None if simulation not running
        """
        with self._lock:
            if machine_id not in self._simulations:
                logger.debug(f"No active simulation for {machine_id}")
                return None
            
            sim = self._simulations[machine_id]
            
            try:
                # Get current row
                current_row = sim['dataframe'].iloc[sim['current_index']]
                
                # Convert to dictionary (handling any NaN values)
                sensor_data = current_row.to_dict()
                sensor_data = {k: float(v) if pd.notna(v) else 0.0 for k, v in sensor_data.items()}
                
                # Advance to next row
                sim['current_index'] += 1
                
                # Loop back to beginning if we've reached the end
                if sim['current_index'] >= sim['total_rows']:
                    sim['current_index'] = 0
                    sim['cycle_count'] += 1
                    logger.info(f"Simulation cycle {sim['cycle_count']} completed for {machine_id}, restarting from row 0")
                
                # Update timestamp
                sim['last_update'] = datetime.now()
                
                logger.debug(
                    f"Sensor reading for {machine_id}: "
                    f"row {sim['current_index']}/{sim['total_rows']} "
                    f"(cycle {sim['cycle_count']})"
                )
                
                return sensor_data
                
            except Exception as e:
                logger.error(f"Failed to get readings for {machine_id}: {e}", exc_info=True)
                return None
    
    def get_simulation_status(self, machine_id: str) -> Optional[Dict[str, Any]]:
        """
        Get simulation status for a machine.
        
        Args:
            machine_id: Machine identifier
            
        Returns:
            Status dictionary or None if not running
        """
        with self._lock:
            if machine_id not in self._simulations:
                return None
            
            sim = self._simulations[machine_id]
            
            return {
                'machine_id': machine_id,
                'is_running': True,
                'current_row': sim['current_index'],
                'total_rows': sim['total_rows'],
                'progress_percent': (sim['current_index'] / sim['total_rows']) * 100,
                'sensor_count': len(sim['sensor_columns']),
                'started_at': sim['started_at'].isoformat(),
                'last_update': sim['last_update'].isoformat(),
                'cycle_count': sim['cycle_count']
            }
    
    def is_running(self, machine_id: str) -> bool:
        """
        Check if simulation is running for a machine.
        
        Args:
            machine_id: Machine identifier
            
        Returns:
            True if simulation is active
        """
        with self._lock:
            return machine_id in self._simulations
    
    def get_active_simulations(self) -> list[str]:
        """
        Get list of machines with active simulations.
        
        Returns:
            List of machine IDs
        """
        with self._lock:
            return list(self._simulations.keys())
    
    def stop_all_simulations(self) -> int:
        """
        Stop all active simulations.
        
        Returns:
            Number of simulations stopped
        """
        with self._lock:
            count = len(self._simulations)
            self._simulations.clear()
            logger.info(f"[OK] Stopped all simulations ({count} total)")
            return count


# Global singleton instance
_simulator_instance: Optional[SensorSimulator] = None


def get_simulator() -> SensorSimulator:
    """
    Get the global sensor simulator instance.
    
    Returns:
        SensorSimulator instance
    """
    global _simulator_instance
    
    if _simulator_instance is None:
        _simulator_instance = SensorSimulator()
    
    return _simulator_instance
