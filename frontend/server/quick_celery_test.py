"""
Quick test to check if Celery worker can discover and execute tasks.
"""
import sys
sys.path.insert(0, ".")

# First, let's check if celery_app can find tasks
from celery_app import celery_app

print("=" * 70)
print("CELERY CONFIGURATION CHECK")
print("=" * 70)
print(f"Broker URL: {celery_app.conf.broker_url}")
print(f"Result Backend: {celery_app.conf.result_backend}")
print(f"\nRegistered Tasks:")
for task_name in sorted(celery_app.tasks.keys()):
    if not task_name.startswith('celery.'):
        print(f"  - {task_name}")

print("\n" + "=" * 70)
print("TESTING TASK EXECUTION")
print("=" * 70)

# Import task directly
from tasks.test_task import add

# Test 1: Direct execution (no Celery worker needed)
print("\nTest 1: Direct Execution (no worker)")
result = add(2, 3)
print(f"add(2, 3) = {result}")

# Test 2: Check if we can create a task signature
print("\nTest 2: Task Signature")
task_signature = add.s(4, 5)
print(f"Task signature created: {task_signature}")

# Test 3: Try to send task to worker (will timeout if worker not running)
print("\nTest 3: Async Execution (requires worker)")
print("Sending task to worker...")
try:
    task = add.delay(10, 20)
    print(f"Task sent! Task ID: {task.id}")
    print(f"Task state: {task.state}")
    print("Waiting for result (5 second timeout)...")
    result = task.get(timeout=5)
    print(f"✅ Result received: {result}")
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nPossible issues:")
    print("  1. Celery worker not running")
    print("  2. Worker can't connect to Redis")
    print("  3. Tasks not registered with worker")
    print("\nTo start worker, run:")
    print("  celery -A celery_app worker --loglevel=info --pool=solo")
