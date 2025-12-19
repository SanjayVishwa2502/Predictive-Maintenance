"""
ML API Endpoint Testing Script
Phase 3.7.3 Day 15.2

Tests all 6 ML API endpoints:
1. GET /api/ml/machines - List machines
2. GET /api/ml/machines/{id}/status - Get machine status
3. POST /api/ml/predict/classification - Run classification
4. POST /api/ml/predict/rul - Run RUL prediction
5. GET /api/ml/machines/{id}/history - Get prediction history
6. GET /api/ml/health - Health check

Usage:
    python test_ml_api.py
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "frontend" / "server"))

from api.routes.ml import (
    list_machines,
    get_machine_status,
    predict_classification,
    predict_rul,
    get_prediction_history,
    health_check
)
from api.models.ml import PredictionRequest


def print_header(title: str):
    """Print formatted test header"""
    print("\n" + "="*70)
    print(f"TEST: {title}")
    print("="*70)


def print_success(message: str):
    """Print success message"""
    print(f"✅ {message}")


def print_error(message: str):
    """Print error message"""
    print(f"❌ {message}")


async def test_list_machines():
    """Test 1: List all machines"""
    print_header("List All Machines")
    
    try:
        response = await list_machines()
        
        print(f"Total machines: {response.total}")
        print(f"\nFirst 5 machines:")
        for machine in response.machines[:5]:
            print(f"  - {machine.machine_id}")
            print(f"    Category: {machine.category}")
            print(f"    Sensors: {machine.sensor_count}")
            print(f"    Classification model: {machine.has_classification_model}")
        
        print_success(f"Retrieved {response.total} machines")
        return True
        
    except Exception as e:
        print_error(f"Failed: {e}")
        return False


async def test_machine_status():
    """Test 2: Get machine status"""
    print_header("Get Machine Status")
    
    machine_id = "motor_siemens_1la7_001"
    
    try:
        response = await get_machine_status(machine_id)
        
        print(f"Machine: {response.machine_id}")
        print(f"Running: {response.is_running}")
        print(f"Sensor count: {response.sensor_count}")
        print(f"Last update: {response.last_update}")
        print(f"\nSample sensors:")
        for i, (name, value) in enumerate(list(response.latest_sensors.items())[:5]):
            print(f"  {name}: {value}")
        
        print_success(f"Status retrieved for {machine_id}")
        return True
        
    except Exception as e:
        print_error(f"Failed: {e}")
        return False


async def test_classification_prediction():
    """Test 3: Run classification prediction"""
    print_header("Classification Prediction")
    
    machine_id = "motor_siemens_1la7_001"
    
    try:
        # Get current sensor data
        status = await get_machine_status(machine_id)
        sensor_data = status.latest_sensors
        
        # Create request
        request = PredictionRequest(
            machine_id=machine_id,
            sensor_data=sensor_data
        )
        
        # Run prediction
        print(f"Running prediction with {len(sensor_data)} sensors...")
        response = await predict_classification(request)
        
        print(f"\nPrediction Results:")
        print(f"  Failure type: {response.prediction.failure_type}")
        print(f"  Confidence: {response.prediction.confidence:.2%}")
        print(f"  Failure probability: {response.prediction.failure_probability:.2%}")
        print(f"\nAll probabilities:")
        for failure_type, prob in response.prediction.all_probabilities.items():
            print(f"  {failure_type}: {prob:.2%}")
        
        print(f"\nExplanation:")
        print(f"  Summary: {response.explanation.summary[:100]}...")
        print(f"  Risk factors: {len(response.explanation.risk_factors)}")
        print(f"  Recommendations: {len(response.explanation.recommendations)}")
        
        print_success("Classification prediction completed")
        return True
        
    except Exception as e:
        print_error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_rul_prediction():
    """Test 4: Run RUL prediction"""
    print_header("RUL Prediction")
    
    machine_id = "motor_siemens_1la7_001"
    
    try:
        # Get current sensor data
        status = await get_machine_status(machine_id)
        sensor_data = status.latest_sensors
        
        # Create request
        request = PredictionRequest(
            machine_id=machine_id,
            sensor_data=sensor_data
        )
        
        # Run prediction
        print(f"Running RUL prediction...")
        response = await predict_rul(request)
        
        print(f"\nRUL Results:")
        print(f"  RUL: {response.prediction.rul_hours:.1f} hours ({response.prediction.rul_days:.1f} days)")
        print(f"  Urgency: {response.prediction.urgency}")
        print(f"  Maintenance window: {response.prediction.maintenance_window}")
        print(f"  Confidence: {response.prediction.confidence:.2%}")
        
        print(f"\nCritical sensors:")
        for sensor in response.prediction.critical_sensors:
            print(f"  {sensor.name}: {sensor.value} (severity: {sensor.severity})")
        
        print(f"\nExplanation:")
        print(f"  Summary: {response.explanation.summary[:100]}...")
        
        print_success("RUL prediction completed")
        return True
        
    except Exception as e:
        print_error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_prediction_history():
    """Test 5: Get prediction history"""
    print_header("Prediction History")
    
    machine_id = "motor_siemens_1la7_001"
    
    try:
        response = await get_prediction_history(machine_id, limit=10)
        
        print(f"Machine: {response.machine_id}")
        print(f"Total predictions: {response.total}")
        print(f"Showing: {len(response.predictions)}")
        
        if response.predictions:
            print(f"\nRecent predictions:")
            for pred in response.predictions[:3]:
                print(f"  {pred.timestamp}: {pred.failure_type} (confidence: {pred.confidence:.2%})")
        else:
            print("  No predictions yet")
        
        print_success("Prediction history retrieved")
        return True
        
    except Exception as e:
        print_error(f"Failed: {e}")
        return False


async def test_health_check():
    """Test 6: Service health check"""
    print_header("Service Health Check")
    
    try:
        response = await health_check()
        
        print(f"Status: {response.status}")
        print(f"\nModels loaded:")
        print(f"  Classification: {response.models_loaded.classification}")
        print(f"  Regression: {response.models_loaded.regression}")
        print(f"  Anomaly: {response.models_loaded.anomaly}")
        print(f"  Timeseries: {response.models_loaded.timeseries}")
        
        print(f"\nLLM status: {response.llm_status}")
        print(f"GPU available: {response.gpu_available}")
        if response.gpu_info:
            print(f"GPU: {response.gpu_info.name} (CUDA {response.gpu_info.cuda_version})")
        
        print(f"IntegratedPredictionSystem ready: {response.integrated_system_ready}")
        
        print_success("Health check completed")
        return True
        
    except Exception as e:
        print_error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("ML API ENDPOINT TESTING")
    print("Testing 6 endpoints with motor_siemens_1la7_001")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("List Machines", await test_list_machines()))
    results.append(("Machine Status", await test_machine_status()))
    results.append(("Classification Prediction", await test_classification_prediction()))
    results.append(("RUL Prediction", await test_rul_prediction()))
    results.append(("Prediction History", await test_prediction_history()))
    results.append(("Health Check", await test_health_check()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! ML API is ready.")
        print("\nNext steps:")
        print("1. Start FastAPI server: uvicorn main:app --reload")
        print("2. Open Swagger UI: http://localhost:8000/docs")
        print("3. Test WebSocket: Open websocket_test.html")
    else:
        print("\n⚠️ Some tests failed. Check error messages above.")


if __name__ == "__main__":
    asyncio.run(main())
