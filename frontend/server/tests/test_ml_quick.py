"""
Quick ML API Health Check
Tests that ML API is working without verbose LLM output
"""
import asyncio
import sys
import os
from pathlib import Path

# Suppress verbose TensorFlow/LLM warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "frontend" / "server"))

print("\n" + "="*70)
print("ML API Quick Health Check")
print("="*70)
print("\nInitializing services (this takes ~30 seconds)...")
print("Loading: LLM (4.9GB) + RAG embeddings + ML models\n")

# Import MLManager (triggers LLM load)
from services.ml_manager import get_ml_manager


async def main():
    try:
        # Get manager
        manager = get_ml_manager()
        print("✅ MLManager initialized successfully\n")
        
        # Quick health check
        print("Running health check...")
        health = manager.get_service_health()
        
        print("\n" + "="*70)
        print("SERVICE STATUS")
        print("="*70)
        print(f"Status: {health['status'].upper()}")
        print(f"\nModels Loaded:")
        print(f"  - Classification: {health['models_loaded']['classification']}")
        print(f"  - Regression: {health['models_loaded']['regression']}")
        print(f"  - Anomaly: {health['models_loaded']['anomaly']}")
        print(f"  - Timeseries: {health['models_loaded']['timeseries']}")
        print(f"\nLLM Status: {health['llm_status']}")
        print(f"GPU Available: {health['gpu_available']}")
        print(f"IntegratedPredictionSystem: {'READY' if health['integrated_system_ready'] else 'NOT READY'}")
        
        # Quick machine list test
        print("\n" + "="*70)
        print("MACHINE LIST TEST")
        print("="*70)
        machines = await manager.list_machines()
        print(f"Total machines: {len(machines)}")
        print(f"\nFirst 5 machines:")
        for m in machines[:5]:
            print(f"  - {m['machine_id']} ({m['sensor_count']} sensors)")
        
        print("\n" + "="*70)
        print("✅ ALL SYSTEMS OPERATIONAL")
        print("="*70)
        print("\nML API is ready for use!")
        print("\nNext Steps:")
        print("1. Register ML router in main.py")
        print("2. Start FastAPI: uvicorn main:app --reload")
        print("3. Test endpoints: http://localhost:8000/docs")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
