"""
Test all 4 MLExplainer methods
Phase 3.5.1: Comprehensive Testing

This script tests:
1. explain_classification() - Failure prediction explanations
2. explain_rul() - Remaining useful life explanations
3. explain_anomaly() - Anomaly detection explanations
4. explain_forecast() - Time-series forecast explanations
"""
import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root / "LLM" / "api"))

from explainer import MLExplainer
import time


def test_all_methods():
    """Test all 4 explanation methods"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE MLExplainer TEST - Phase 3.5.1")
    print("="*70)
    
    # Initialize explainer (loads LLM + RAG)
    explainer = MLExplainer()
    
    results = {}
    
    # ==================== TEST 1: Classification ====================
    print("\n" + "="*70)
    print("TEST 1: FAILURE CLASSIFICATION")
    print("="*70)
    
    start = time.time()
    result_classification = explainer.explain_classification(
        machine_id="motor_siemens_1la7_001",
        failure_prob=0.87,
        failure_type="bearing_wear",
        sensor_data={
            'vibration': 12.5,
            'temperature': 78.0,
            'current': 45.2
        },
        confidence=0.92
    )
    elapsed_classification = time.time() - start
    
    print("\n" + "-"*70)
    print("EXPLANATION:")
    print("-"*70)
    print(result_classification['explanation'])
    print("\n" + "-"*70)
    print(f"Sources: {result_classification['sources']}")
    print(f"Confidence: {result_classification['confidence']:.1%}")
    print(f"Total Time: {elapsed_classification:.1f}s")
    print("="*70)
    
    results['classification'] = {
        'success': True,
        'time': elapsed_classification,
        'words': len(result_classification['explanation'].split())
    }
    
    # ==================== TEST 2: RUL Regression ====================
    print("\n" + "="*70)
    print("TEST 2: REMAINING USEFUL LIFE (RUL)")
    print("="*70)
    
    start = time.time()
    result_rul = explainer.explain_rul(
        machine_id="pump_grundfos_cr3_004",
        rul_hours=156.5,
        sensor_data={
            'pressure': 2.1,
            'flow_rate': 145.0,
            'vibration': 8.3,
            'temperature': 65.0
        },
        confidence=0.89
    )
    elapsed_rul = time.time() - start
    
    print("\n" + "-"*70)
    print("EXPLANATION:")
    print("-"*70)
    print(result_rul['explanation'])
    print("\n" + "-"*70)
    print(f"Sources: {result_rul['sources']}")
    print(f"Confidence: {result_rul['confidence']:.1%}")
    print(f"Total Time: {elapsed_rul:.1f}s")
    print("="*70)
    
    results['rul'] = {
        'success': True,
        'time': elapsed_rul,
        'words': len(result_rul['explanation'].split())
    }
    
    # ==================== TEST 3: Anomaly Detection ====================
    print("\n" + "="*70)
    print("TEST 3: ANOMALY DETECTION")
    print("="*70)
    
    start = time.time()
    result_anomaly = explainer.explain_anomaly(
        machine_id="compressor_atlas_copco_ga30_001",
        anomaly_score=0.78,
        abnormal_sensors={
            'vibration': 15.2,
            'temperature': 92.0,
            'pressure': 8.5
        },
        detection_method="Isolation Forest",
        threshold=0.5
    )
    elapsed_anomaly = time.time() - start
    
    print("\n" + "-"*70)
    print("EXPLANATION:")
    print("-"*70)
    print(result_anomaly['explanation'])
    print("\n" + "-"*70)
    print(f"Sources: {result_anomaly['sources']}")
    print(f"Anomaly Score: {result_anomaly['anomaly_score']:.2f}")
    print(f"Threshold: {result_anomaly['threshold']:.2f}")
    print(f"Total Time: {elapsed_anomaly:.1f}s")
    print("="*70)
    
    results['anomaly'] = {
        'success': True,
        'time': elapsed_anomaly,
        'words': len(result_anomaly['explanation'].split())
    }
    
    # ==================== TEST 4: Time-Series Forecast ====================
    print("\n" + "="*70)
    print("TEST 4: TIME-SERIES FORECAST")
    print("="*70)
    
    start = time.time()
    result_forecast = explainer.explain_forecast(
        machine_id="cooling_tower_bac_vti_018",
        forecast_summary="Temperature predicted to rise by 8°C over next 24 hours. Vibration trending upward (current: 6.2, forecast: 9.5). Pressure stable.",
        confidence=0.85
    )
    elapsed_forecast = time.time() - start
    
    print("\n" + "-"*70)
    print("EXPLANATION:")
    print("-"*70)
    print(result_forecast['explanation'])
    print("\n" + "-"*70)
    print(f"Sources: {result_forecast['sources']}")
    print(f"Confidence: {result_forecast['confidence']:.1%}")
    print(f"Total Time: {elapsed_forecast:.1f}s")
    print("="*70)
    
    results['forecast'] = {
        'success': True,
        'time': elapsed_forecast,
        'words': len(result_forecast['explanation'].split())
    }
    
    # ==================== SUMMARY ====================
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total_time = sum(r['time'] for r in results.values())
    total_words = sum(r['words'] for r in results.values())
    
    print(f"\nTests Completed: 4/4 ✓")
    print(f"\nPer-Method Results:")
    print("-"*70)
    for method, data in results.items():
        print(f"  {method:15} | Time: {data['time']:5.1f}s | Words: {data['words']:3d} | Status: ✓")
    
    print("\n" + "-"*70)
    print(f"Total Execution Time: {total_time:.1f}s")
    print(f"Average Time/Method:  {total_time/4:.1f}s")
    print(f"Total Words Generated: {total_words}")
    print(f"Average Words/Method:  {total_words/4:.0f}")
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED - Phase 3.5.1 Complete!")
    print("="*70)
    
    return results


if __name__ == "__main__":
    try:
        results = test_all_methods()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
