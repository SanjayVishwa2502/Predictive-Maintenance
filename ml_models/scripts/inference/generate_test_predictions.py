"""
Batch Test Prediction Generator
Phase 3.5.0: ML Model Inference Pipeline - Task 2

Generates 100 test predictions across all 4 model types:
- 25 Classification predictions (5 machines × 5 samples)
- 25 Anomaly detection predictions (5 machines × 5 samples)
- 25 RUL regression predictions (5 machines × 5 samples)
- 25 Time-series forecasts (5 machines × 5 samples)

Output: Organized JSON files for LLM integration testing
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

# Import all 4 predictor classes and data loading functions
from ml_models.scripts.inference.predict_classification import ClassificationPredictor, load_sample_data as load_classification_data
from ml_models.scripts.inference.predict_anomaly import AnomalyPredictor, load_sample_data as load_anomaly_data
from ml_models.scripts.inference.predict_rul import RULPredictor, load_sample_data as load_rul_data
from ml_models.scripts.inference.predict_timeseries import TimeSeriesPredictor, create_sample_historical_data


class BatchPredictionGenerator:
    """Generates batch predictions for all 4 model types"""
    
    # Priority machines for testing
    PRIORITY_MACHINES = [
        'motor_siemens_1la7_001',
        'motor_abb_m3bp_002',
        'pump_grundfos_cr3_004',
        'compressor_atlas_copco_ga30_001',
        'cooling_tower_bac_vti_018'
    ]
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize batch generator
        
        Args:
            output_dir: Output directory for predictions
        """
        self.output_dir = output_dir or PROJECT_ROOT / "ml_models" / "outputs" / "predictions"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.classification_dir = self.output_dir / "classification"
        self.anomaly_dir = self.output_dir / "anomaly"
        self.rul_dir = self.output_dir / "rul"
        self.timeseries_dir = self.output_dir / "timeseries"
        
        for dir in [self.classification_dir, self.anomaly_dir, self.rul_dir, self.timeseries_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'classification': {'success': 0, 'failed': 0, 'errors': []},
            'anomaly': {'success': 0, 'failed': 0, 'errors': []},
            'rul': {'success': 0, 'failed': 0, 'errors': []},
            'timeseries': {'success': 0, 'failed': 0, 'errors': []}
        }
    
    def generate_classification_predictions(self, num_samples: int = 5) -> Dict:
        """
        Generate classification predictions for all machines
        
        Args:
            num_samples: Number of samples per machine
            
        Returns:
            Statistics dictionary
        """
        print("\n" + "="*60)
        print("GENERATING CLASSIFICATION PREDICTIONS")
        print("="*60)
        
        for machine_id in self.PRIORITY_MACHINES:
            print(f"\nProcessing {machine_id}...")
            
            try:
                # Initialize predictor
                predictor = ClassificationPredictor(machine_id)
                
                # Load real sample data (returns list of dicts with timestamp and rul)
                test_data_list = load_classification_data(machine_id, num_samples)
                
                # Run predictions
                predictions = predictor.predict_batch(test_data_list)
                
                # Save predictions
                output_file = self.classification_dir / f"{machine_id}_predictions.json"
                with open(output_file, 'w') as f:
                    json.dump(predictions, f, indent=2)
                
                print(f"  ✓ Generated {len(predictions)} predictions")
                print(f"  ✓ Saved to: {output_file.name}")
                
                self.stats['classification']['success'] += len(predictions)
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                self.stats['classification']['failed'] += num_samples
                self.stats['classification']['errors'].append(f"{machine_id}: {str(e)}")
        
        return self.stats['classification']
    
    def generate_anomaly_predictions(self, num_samples: int = 5) -> Dict:
        """
        Generate anomaly detection predictions for all machines
        
        Args:
            num_samples: Number of samples per machine
            
        Returns:
            Statistics dictionary
        """
        print("\n" + "="*60)
        print("GENERATING ANOMALY DETECTION PREDICTIONS")
        print("="*60)
        
        for machine_id in self.PRIORITY_MACHINES:
            print(f"\nProcessing {machine_id}...")
            
            try:
                # Initialize predictor
                predictor = AnomalyPredictor(machine_id)
                
                # Load real sample data (returns list of dicts with timestamp)
                test_data_list = load_anomaly_data(machine_id, num_samples)
                
                # Run predictions
                predictions = predictor.predict_batch(test_data_list)
                
                # Save predictions
                output_file = self.anomaly_dir / f"{machine_id}_predictions.json"
                with open(output_file, 'w') as f:
                    json.dump(predictions, f, indent=2)
                
                print(f"  ✓ Generated {len(predictions)} predictions")
                print(f"  ✓ Saved to: {output_file.name}")
                
                self.stats['anomaly']['success'] += len(predictions)
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                self.stats['anomaly']['failed'] += num_samples
                self.stats['anomaly']['errors'].append(f"{machine_id}: {str(e)}")
        
        return self.stats['anomaly']
    
    def generate_rul_predictions(self, num_samples: int = 5) -> Dict:
        """
        Generate RUL regression predictions for all machines
        
        Args:
            num_samples: Number of samples per machine
            
        Returns:
            Statistics dictionary
        """
        print("\n" + "="*60)
        print("GENERATING RUL REGRESSION PREDICTIONS")
        print("="*60)
        
        for machine_id in self.PRIORITY_MACHINES:
            print(f"\nProcessing {machine_id}...")
            
            try:
                # Initialize predictor
                predictor = RULPredictor(machine_id)
                
                # Load real sample data (returns list of dicts with timestamp, no rul)
                test_data_list = load_rul_data(machine_id, num_samples)
                
                # Run predictions
                predictions = predictor.predict_batch(test_data_list)
                
                # Save predictions
                output_file = self.rul_dir / f"{machine_id}_predictions.json"
                with open(output_file, 'w') as f:
                    json.dump(predictions, f, indent=2)
                
                print(f"  ✓ Generated {len(predictions)} predictions")
                print(f"  ✓ Saved to: {output_file.name}")
                
                self.stats['rul']['success'] += len(predictions)
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                self.stats['rul']['failed'] += num_samples
                self.stats['rul']['errors'].append(f"{machine_id}: {str(e)}")
        
        return self.stats['rul']
    
    def generate_timeseries_forecasts(self, num_samples: int = 5) -> Dict:
        """
        Generate time-series forecasts for all machines
        
        Args:
            num_samples: Number of forecasts per machine
            
        Returns:
            Statistics dictionary
        """
        print("\n" + "="*60)
        print("GENERATING TIME-SERIES FORECASTS")
        print("="*60)
        
        for machine_id in self.PRIORITY_MACHINES:
            print(f"\nProcessing {machine_id}...")
            
            try:
                # Initialize predictor
                predictor = TimeSeriesPredictor(machine_id)
                
                # Generate forecasts (use different historical windows)
                predictions = []
                for i in range(num_samples):
                    # Create historical data
                    historical_data = create_sample_historical_data(machine_id, num_timesteps=168)
                    
                    # Run forecast
                    result = predictor.predict(historical_data, forecast_steps=24)
                    predictions.append(result)
                
                # Save predictions
                output_file = self.timeseries_dir / f"{machine_id}_predictions.json"
                with open(output_file, 'w') as f:
                    json.dump(predictions, f, indent=2)
                
                print(f"  ✓ Generated {len(predictions)} forecasts")
                print(f"  ✓ Saved to: {output_file.name}")
                
                self.stats['timeseries']['success'] += len(predictions)
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                self.stats['timeseries']['failed'] += num_samples
                self.stats['timeseries']['errors'].append(f"{machine_id}: {str(e)}")
        
        return self.stats['timeseries']
    
    def generate_all_predictions(self, samples_per_machine: int = 5):
        """
        Generate all 100 predictions (25 per model type)
        
        Args:
            samples_per_machine: Number of samples per machine (default: 5)
        """
        print("\n" + "="*70)
        print(" BATCH PREDICTION GENERATION - PHASE 3.5.0 TASK 2")
        print("="*70)
        print(f"\nTarget: 100 predictions total")
        print(f"  - 25 Classification predictions ({len(self.PRIORITY_MACHINES)} machines × {samples_per_machine} samples)")
        print(f"  - 25 Anomaly predictions ({len(self.PRIORITY_MACHINES)} machines × {samples_per_machine} samples)")
        print(f"  - 25 RUL predictions ({len(self.PRIORITY_MACHINES)} machines × {samples_per_machine} samples)")
        print(f"  - 25 Time-series forecasts ({len(self.PRIORITY_MACHINES)} machines × {samples_per_machine} samples)")
        print(f"\nPriority Machines:")
        for i, machine in enumerate(self.PRIORITY_MACHINES, 1):
            print(f"  {i}. {machine}")
        
        start_time = datetime.now()
        
        # Generate all 4 types
        self.generate_classification_predictions(samples_per_machine)
        self.generate_anomaly_predictions(samples_per_machine)
        self.generate_rul_predictions(samples_per_machine)
        self.generate_timeseries_forecasts(samples_per_machine)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Generate summary report
        self._generate_summary_report(duration)
    
    def _generate_summary_report(self, duration: float):
        """
        Generate summary report
        
        Args:
            duration: Total execution time in seconds
        """
        print("\n" + "="*70)
        print(" BATCH GENERATION SUMMARY")
        print("="*70)
        
        total_success = 0
        total_failed = 0
        
        for model_type, stats in self.stats.items():
            print(f"\n{model_type.upper()}:")
            print(f"  ✓ Success: {stats['success']}")
            print(f"  ❌ Failed: {stats['failed']}")
            
            if stats['errors']:
                print(f"  Errors:")
                for error in stats['errors']:
                    print(f"    - {error}")
            
            total_success += stats['success']
            total_failed += stats['failed']
        
        print("\n" + "-"*70)
        print(f"TOTAL PREDICTIONS: {total_success + total_failed}")
        print(f"  ✓ Success: {total_success}")
        print(f"  ❌ Failed: {total_failed}")
        print(f"  Success Rate: {(total_success/(total_success+total_failed)*100):.1f}%")
        print(f"\nExecution Time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Average Time per Prediction: {duration/(total_success+total_failed):.2f} seconds")
        
        print("\n" + "-"*70)
        print("OUTPUT LOCATIONS:")
        print(f"  Classification: {self.classification_dir}")
        print(f"  Anomaly: {self.anomaly_dir}")
        print(f"  RUL: {self.rul_dir}")
        print(f"  Time-Series: {self.timeseries_dir}")
        
        # Save summary to JSON
        summary = {
            'timestamp': datetime.utcnow().isoformat() + "Z",
            'total_predictions': total_success + total_failed,
            'successful': total_success,
            'failed': total_failed,
            'success_rate': round(total_success/(total_success+total_failed)*100, 2),
            'execution_time_seconds': round(duration, 2),
            'avg_time_per_prediction': round(duration/(total_success+total_failed), 2),
            'model_types': self.stats,
            'output_directories': {
                'classification': str(self.classification_dir),
                'anomaly': str(self.anomaly_dir),
                'rul': str(self.rul_dir),
                'timeseries': str(self.timeseries_dir)
            }
        }
        
        summary_file = self.output_dir / "batch_generation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Summary saved to: {summary_file}")
        
        print("\n" + "="*70)
        print("✓ BATCH GENERATION COMPLETE")
        print("="*70)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch Test Prediction Generator')
    parser.add_argument('--samples_per_machine', type=int, default=5,
                       help='Number of samples per machine (default: 5)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for predictions')
    parser.add_argument('--model_type', type=str, default='all',
                       choices=['all', 'classification', 'anomaly', 'rul', 'timeseries'],
                       help='Model type to generate predictions for (default: all)')
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        output_dir = Path(args.output_dir) if args.output_dir else None
        generator = BatchPredictionGenerator(output_dir)
        
        # Generate predictions
        if args.model_type == 'all':
            generator.generate_all_predictions(args.samples_per_machine)
        elif args.model_type == 'classification':
            generator.generate_classification_predictions(args.samples_per_machine)
        elif args.model_type == 'anomaly':
            generator.generate_anomaly_predictions(args.samples_per_machine)
        elif args.model_type == 'rul':
            generator.generate_rul_predictions(args.samples_per_machine)
        elif args.model_type == 'timeseries':
            generator.generate_timeseries_forecasts(args.samples_per_machine)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
