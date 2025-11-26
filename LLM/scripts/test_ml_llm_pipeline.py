"""
ML → LLM PIPELINE TESTER
========================
Tests the complete pipeline from ML predictions to LLM explanations.

Pipeline Flow:
1. Load ML predictions (from mock or real models)
2. Retrieve RAG context
3. Format prompt with ML + RAG
4. Generate LLM explanation
5. Save results

Usage:
    python test_ml_llm_pipeline.py --num_samples 5 --model_types classification anomaly rul timeseries
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from LLM.api.inference_service import UnifiedInferenceService


class MLLLMPipelineTester:
    """Tests complete ML → LLM pipeline"""
    
    def __init__(self, use_mock: bool = True):
        """
        Initialize pipeline tester
        
        Args:
            use_mock: If True, use mock predictions. If False, use real ML models.
        """
        self.use_mock = use_mock
        self.predictions_dir = PROJECT_ROOT / "ml_models" / "outputs" / "predictions"
        self.output_dir = PROJECT_ROOT / "LLM" / "outputs" / "explanations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize unified service
        print("\n" + "="*70)
        print(" INITIALIZING ML → LLM PIPELINE")
        print("="*70)
        print(f"\nMode: {'Mock Predictions' if use_mock else 'Real ML Models'}")
        
        try:
            self.service = UnifiedInferenceService()
            print("✓ Unified Inference Service initialized")
        except Exception as e:
            print(f"⚠️  Service initialization warning: {e}")
            print("   (This is expected - full implementation in Phase 3.5.1)")
            self.service = None
    
    def load_predictions(self, model_type: str, num_samples: int = None) -> List[Dict]:
        """
        Load ML predictions from files
        
        Args:
            model_type: 'classification', 'anomaly', 'rul', or 'timeseries'
            num_samples: Number of samples to load (None = all)
            
        Returns:
            List of prediction dictionaries
        """
        type_dir = self.predictions_dir / model_type
        
        if not type_dir.exists():
            print(f"⚠️  No predictions found for {model_type}")
            return []
        
        predictions = []
        for pred_file in type_dir.glob("*_predictions.json"):
            with open(pred_file, 'r') as f:
                file_predictions = json.load(f)
                predictions.extend(file_predictions)
                
                if num_samples and len(predictions) >= num_samples:
                    predictions = predictions[:num_samples]
                    break
        
        return predictions
    
    def simulate_rag_retrieval(self, prediction: Dict) -> List[str]:
        """
        Simulate RAG context retrieval
        (In Phase 3.5.1, this will use actual FAISS index)
        
        Args:
            prediction: ML prediction dictionary
            
        Returns:
            List of relevant context documents
        """
        machine_id = prediction.get('machine_id', 'unknown')
        model_type = prediction.get('model_type', 'unknown')
        
        # Simulated context documents
        if model_type == 'classification':
            failure_type = prediction.get('prediction', {}).get('failure_type', 'unknown')
            return [
                f"Machine {machine_id}: {failure_type} failure symptoms include elevated temperature and abnormal vibration patterns.",
                f"Historical data shows {failure_type} failures typically occur after 500-1000 operating hours.",
                f"Recommended action for {failure_type}: Schedule maintenance within 48 hours to prevent catastrophic failure."
            ]
        
        elif model_type == 'anomaly_detection':
            is_anomaly = prediction.get('prediction', {}).get('is_anomaly', False)
            return [
                f"Machine {machine_id}: {'Anomalous' if is_anomaly else 'Normal'} operating pattern detected.",
                f"Anomaly detection baseline: established from 10,000+ hours of normal operation.",
                f"Action: {'Immediate inspection required' if is_anomaly else 'Continue monitoring'}."
            ]
        
        elif model_type == 'rul_regression':
            rul_hours = prediction.get('prediction', {}).get('rul_hours', 0)
            return [
                f"Machine {machine_id}: Estimated remaining useful life {rul_hours:.0f} hours.",
                f"RUL prediction based on degradation curve analysis and sensor trend forecasting.",
                f"Maintenance scheduling: Plan intervention at 70-80% of predicted RUL for optimal cost-efficiency."
            ]
        
        elif model_type == 'timeseries_forecast':
            return [
                f"Machine {machine_id}: 24-hour sensor forecasting indicates upcoming operational trends.",
                f"Forecast accuracy: ±5% for temperature, ±10% for vibration based on historical validation.",
                f"Proactive monitoring: Use forecasts to schedule preventive maintenance during low-activity windows."
            ]
        
        return ["Generic maintenance guidance for industrial equipment."]
    
    def simulate_prompt_formatting(self, prediction: Dict, context: List[str]) -> str:
        """
        Simulate prompt formatting
        (In Phase 3.5.1, this will use actual prompt templates)
        
        Args:
            prediction: ML prediction dictionary
            context: RAG context documents
            
        Returns:
            Formatted prompt string
        """
        machine_id = prediction.get('machine_id', 'unknown')
        model_type = prediction.get('model_type', 'unknown')
        pred_data = prediction.get('prediction', {})
        sensors = prediction.get('sensor_readings', {})
        
        # Build prompt
        prompt = f"""You are an expert industrial maintenance technician analyzing predictive maintenance data.

MACHINE: {machine_id}
ANALYSIS TYPE: {model_type}

SENSOR READINGS:
"""
        for sensor, value in sensors.items():
            prompt += f"  - {sensor}: {value}\n"
        
        prompt += f"\nML PREDICTION:\n{json.dumps(pred_data, indent=2)}\n"
        
        prompt += "\nRELEVANT CONTEXT:\n"
        for i, doc in enumerate(context, 1):
            prompt += f"{i}. {doc}\n"
        
        prompt += """
INSTRUCTIONS:
Provide a concise maintenance explanation (<200 words) covering:
1. What the data indicates
2. Root cause analysis
3. Immediate actions required
4. Preventive recommendations
5. Safety considerations

Keep technical but understandable for maintenance technicians.
"""
        
        return prompt
    
    def simulate_llm_generation(self, prompt: str, prediction: Dict) -> str:
        """
        Simulate LLM explanation generation
        (In Phase 3.5.1, this will use actual Llama model)
        
        Args:
            prompt: Formatted prompt
            prediction: Original prediction for context
            
        Returns:
            Generated explanation
        """
        model_type = prediction.get('model_type', 'unknown')
        pred_data = prediction.get('prediction', {})
        machine_id = prediction.get('machine_id', 'unknown')
        
        # Simulated explanations based on prediction data
        if model_type == 'classification':
            failure_type = pred_data.get('failure_type', 'unknown')
            confidence = pred_data.get('confidence', 0.0)
            
            if failure_type == 'normal':
                return f"""**Status**: Machine {machine_id} is operating normally with {confidence*100:.1f}% confidence.

**Analysis**: Sensor readings are within acceptable parameters. No immediate failure indicators detected. All subsystems functioning as expected.

**Root Cause**: N/A - preventive monitoring active.

**Immediate Actions**: 
- Continue routine monitoring
- No maintenance intervention required

**Preventive Recommendations**:
- Schedule routine inspection in 2 weeks
- Monitor vibration trends for early warning signs
- Maintain lubrication schedule

**Safety**: No safety concerns. Equipment safe for operation."""
            else:
                return f"""**Status**: {failure_type.replace('_', ' ').title()} failure predicted with {confidence*100:.1f}% confidence.

**Analysis**: Abnormal sensor patterns indicate developing {failure_type.replace('_', ' ')} issue. Temperature and vibration levels exceeding normal thresholds.

**Root Cause**: Likely degradation of critical components based on sensor signature matching historical failure patterns.

**Immediate Actions**:
- Schedule maintenance within 48 hours
- Reduce operational load to 70% if possible
- Increase monitoring frequency to hourly

**Preventive Recommendations**:
- Replace affected components during scheduled maintenance
- Inspect adjacent systems for secondary damage
- Update maintenance records

**Safety**: MODERATE RISK - Avoid continuous high-load operation until serviced."""
        
        elif model_type == 'anomaly_detection':
            is_anomaly = pred_data.get('is_anomaly', False)
            score = pred_data.get('anomaly_score', 0.0)
            
            if is_anomaly:
                return f"""**Status**: Anomalous operation detected on {machine_id} (anomaly score: {score:.2f}).

**Analysis**: Multiple detectors flagged unusual operating patterns not seen in baseline training. Sensor readings deviate from established normal behavior.

**Root Cause**: Potential undiagnosed issue - could be sensor drift, process change, or early-stage component degradation.

**Immediate Actions**:
- Visual inspection within 24 hours
- Verify sensor calibration
- Review recent operational changes
- Compare with adjacent equipment

**Preventive Recommendations**:
- Document anomaly details for pattern analysis
- If persistent, schedule diagnostic testing
- Consider baseline recalibration if process changed

**Safety**: LOW-MODERATE RISK - Investigation required but no immediate danger."""
            else:
                return f"""**Status**: {machine_id} operating normally (anomaly score: {score:.2f}).

**Analysis**: All 8 detection algorithms confirm normal operation. Sensor patterns match established baseline behavior.

**Root Cause**: N/A - equipment within normal operating envelope.

**Immediate Actions**: None required - continue standard monitoring.

**Preventive Recommendations**:
- Maintain current monitoring schedule
- Update baseline if operational parameters change
- Archive current data for future anomaly detection

**Safety**: No concerns - equipment operating safely."""
        
        elif model_type == 'rul_regression':
            rul_hours = pred_data.get('rul_hours', 0)
            rul_days = pred_data.get('rul_days', 0)
            urgency = pred_data.get('urgency', 'unknown')
            
            # Prepare conditional strings
            urgent_action = 'URGENT: Schedule maintenance immediately' if urgency in ['critical', 'high'] else 'Plan maintenance within maintenance window'
            load_action = 'Reduce load if possible' if urgency == 'critical' else 'Continue normal operation until scheduled maintenance'
            
            if urgency == 'critical':
                safety_msg = 'CRITICAL - Failure imminent, minimize operation'
            elif urgency == 'high':
                safety_msg = 'HIGH - Schedule maintenance soon'
            elif urgency == 'medium':
                safety_msg = 'MODERATE - Plan maintenance proactively'
            else:
                safety_msg = 'LOW - Routine maintenance sufficient'
            
            return f"""**Status**: {machine_id} has {rul_hours:.0f} hours ({rul_days:.1f} days) remaining useful life.

**Analysis**: Degradation trend analysis predicts component failure in {rul_days:.0f} days. Current sensor trends indicate steady decline in performance metrics.

**Root Cause**: Normal wear and tear progression based on cumulative operating hours and load patterns.

**Immediate Actions**:
- {urgent_action}
- Order replacement parts now
- {load_action}

**Preventive Recommendations**:
- Perform full inspection during maintenance
- Replace not just primary but adjacent wear components
- Update maintenance schedule based on actual vs predicted life
- Implement condition monitoring post-maintenance

**Safety**: {safety_msg}"""
        
        elif model_type == 'timeseries_forecast':
            forecast_hours = pred_data.get('forecast_horizon_hours', 24)
            concerning = pred_data.get('concerning_trends', [])
            
            has_concerns = concerning and concerning[0] != "No concerning trends detected"
            
            # Prepare conditional strings
            status_desc = 'indicates concerning trends' if has_concerns else 'shows stable operation'
            analysis_desc = 'elevated sensor values in upcoming hours' if has_concerns else 'normal operating conditions continuing'
            root_cause = 'Trend analysis suggests increasing thermal or mechanical stress' if has_concerns else 'Equipment maintaining steady-state operation'
            
            if has_concerns:
                actions = "- Monitor forecasted parameters closely\n- Consider load reduction during peak forecast periods\n- Prepare for potential intervention"
            else:
                actions = "- Continue standard monitoring\n- No immediate intervention required\n- Use forecast for maintenance scheduling"
            
            recommendations = 'Review operating conditions if trends worsen' if has_concerns else 'Maintain current operational parameters'
            safety_msg = 'MODERATE - Monitor forecasted conditions' if has_concerns else 'LOW - No safety concerns in forecast period'
            
            return f"""**Status**: {forecast_hours}-hour forecast for {machine_id} {status_desc}.

**Analysis**: Predictive models forecast {analysis_desc}. Temperature and vibration projections based on historical patterns.

**Root Cause**: {root_cause}.

**Immediate Actions**:
{actions}

**Preventive Recommendations**:
- Schedule maintenance during forecasted low-activity windows (hours 0-6)
- {recommendations}
- Update forecast models with actual outcomes for accuracy improvement

**Safety**: {safety_msg}."""
        
        return "Explanation generation in progress..."
    
    def test_single_prediction(self, prediction: Dict, save: bool = True) -> Dict:
        """
        Test pipeline on a single prediction
        
        Args:
            prediction: ML prediction dictionary
            save: Whether to save results
            
        Returns:
            Complete result with explanation
        """
        machine_id = prediction.get('machine_id', 'unknown')
        model_type = prediction.get('model_type', 'unknown')
        
        print(f"\n{'─'*70}")
        print(f"Testing: {machine_id} - {model_type}")
        print(f"{'─'*70}")
        
        # Step 1: RAG Retrieval (simulated)
        print("📚 Retrieving RAG context...")
        context = self.simulate_rag_retrieval(prediction)
        print(f"   ✓ Retrieved {len(context)} context documents")
        
        # Step 2: Format Prompt (simulated)
        print("📝 Formatting prompt...")
        prompt = self.simulate_prompt_formatting(prediction, context)
        print(f"   ✓ Prompt formatted ({len(prompt)} chars)")
        
        # Step 3: Generate Explanation (simulated)
        print("🤖 Generating LLM explanation...")
        explanation = self.simulate_llm_generation(prompt, prediction)
        print(f"   ✓ Explanation generated ({len(explanation)} chars)")
        
        # Construct result
        result = {
            'request_id': f"{machine_id}_{model_type}_{int(datetime.utcnow().timestamp())}",
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'machine_id': machine_id,
            'model_type': model_type,
            'ml_prediction': prediction.get('prediction', {}),
            'sensor_readings': prediction.get('sensor_readings', {}),
            'rag_context': context,
            'explanation': explanation,
            'metadata': {
                'use_mock': self.use_mock,
                'context_docs': len(context),
                'prompt_length': len(prompt),
                'explanation_length': len(explanation)
            }
        }
        
        # Save if requested
        if save:
            output_file = self.output_dir / model_type / f"{machine_id}_explanation.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"   ✓ Saved to: {output_file.name}")
        
        return result
    
    def test_all_predictions(self, model_types: List[str] = None, 
                           num_samples: int = None):
        """
        Test pipeline on all predictions
        
        Args:
            model_types: List of model types to test (None = all)
            num_samples: Number of samples per type (None = all)
        """
        if model_types is None:
            model_types = ['classification', 'anomaly', 'rul', 'timeseries']
        
        print("\n" + "="*70)
        print(" TESTING ML → LLM PIPELINE")
        print("="*70)
        print(f"\nModel Types: {', '.join(model_types)}")
        print(f"Samples per type: {num_samples if num_samples else 'All'}")
        print()
        
        results = []
        stats = {mt: {'success': 0, 'failed': 0} for mt in model_types}
        
        for model_type in model_types:
            print(f"\n{'═'*70}")
            print(f" {model_type.upper()}")
            print(f"{'═'*70}")
            
            # Load predictions
            predictions = self.load_predictions(model_type, num_samples)
            print(f"\n✓ Loaded {len(predictions)} predictions")
            
            # Test each prediction
            for i, prediction in enumerate(predictions, 1):
                try:
                    result = self.test_single_prediction(prediction)
                    results.append(result)
                    stats[model_type]['success'] += 1
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    stats[model_type]['failed'] += 1
        
        # Print summary
        print("\n" + "="*70)
        print(" PIPELINE TEST SUMMARY")
        print("="*70)
        print()
        
        total_success = sum(s['success'] for s in stats.values())
        total_failed = sum(s['failed'] for s in stats.values())
        
        for model_type, s in stats.items():
            print(f"{model_type:20} Success: {s['success']:3}  Failed: {s['failed']:3}")
        
        print()
        print(f"{'─'*70}")
        print(f"TOTAL                Success: {total_success:3}  Failed: {total_failed:3}")
        print(f"Success Rate: {total_success/(total_success+total_failed)*100:.1f}%")
        print()
        print(f"Output Directory: {self.output_dir}")
        print()
        
        # Save summary
        summary = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'total_predictions': total_success + total_failed,
            'successful': total_success,
            'failed': total_failed,
            'success_rate': total_success/(total_success+total_failed)*100 if total_success+total_failed > 0 else 0,
            'model_types': stats,
            'note': 'Simulated pipeline test - Phase 3.5.1 will use real RAG + LLM'
        }
        
        summary_file = self.output_dir / 'pipeline_test_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Summary saved to: {summary_file.name}")
        print()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test ML → LLM Pipeline')
    parser.add_argument('--model_types', nargs='+', 
                       choices=['classification', 'anomaly', 'rul', 'timeseries'],
                       help='Model types to test (default: all)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples per model type (default: all)')
    parser.add_argument('--use_real', action='store_true',
                       help='Use real ML models instead of mock predictions')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = MLLLMPipelineTester(use_mock=not args.use_real)
    
    # Run tests
    tester.test_all_predictions(
        model_types=args.model_types,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
