"""
Simple test that waits for worker to process
"""
import time
from tasks.test_task import add

print("="*70)
print("CELERY TASK TEST - WITH WORKER RUNNING")
print("="*70)
print("\nMake sure Celery worker is running in another terminal:")
print("  celery -A celery_app worker --loglevel=info --pool=solo")
print("\n" + "="*70)

input("\nPress ENTER when worker is ready...")

print("\nSending task to worker...")
result = add.delay(10, 20)
print(f"Task ID: {result.id}")
print(f"Initial state: {result.state}")

print("\nWaiting for result (10 second timeout)...")
try:
    value = result.get(timeout=10)
    print(f"✅ SUCCESS! Result: {value}")
    print(f"Final state: {result.state}")
except Exception as e:
    print(f"❌ FAILED: {type(e).__name__}: {e}")
    print(f"Final state: {result.state}")
