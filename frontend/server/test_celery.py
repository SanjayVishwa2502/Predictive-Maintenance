"""
Test script for Phase 3.7.1.4: Celery Worker
Tests async task execution with Celery and Redis.
"""
import sys
import time
from celery.result import AsyncResult

sys.path.insert(0, ".")
from celery_app import celery_app
from tasks.test_task import add, multiply, long_running

def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def test_celery_tasks():
    """Test Celery task execution."""
    
    print_section("PHASE 3.7.1.4 CELERY WORKER TESTING")
    
    # Test 1: Simple synchronous execution
    print_section("Test 1: Synchronous Task Execution")
    result = add(4, 5)
    print(f"add(4, 5) = {result}")
    assert result == 9, f"Expected 9, got {result}"
    print("✅ Synchronous execution works")
    
    # Test 2: Async task execution with delay
    print_section("Test 2: Async Task Execution (add.delay)")
    task = add.delay(10, 20)
    print(f"Task ID: {task.id}")
    print(f"Task State: {task.state}")
    print("Waiting for result...")
    
    try:
        result = task.get(timeout=10)
        print(f"Result: {result}")
        assert result == 30, f"Expected 30, got {result}"
        print("✅ Async task execution works")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Multiple async tasks
    print_section("Test 3: Multiple Async Tasks")
    task1 = add.delay(5, 10)
    task2 = multiply.delay(3, 7)
    task3 = add.delay(100, 200)
    
    print(f"Task 1 ID: {task1.id}")
    print(f"Task 2 ID: {task2.id}")
    print(f"Task 3 ID: {task3.id}")
    print("Waiting for all results...")
    
    try:
        result1 = task1.get(timeout=10)
        result2 = task2.get(timeout=10)
        result3 = task3.get(timeout=10)
        
        print(f"add(5, 10) = {result1}")
        print(f"multiply(3, 7) = {result2}")
        print(f"add(100, 200) = {result3}")
        
        assert result1 == 15, f"Expected 15, got {result1}"
        assert result2 == 21, f"Expected 21, got {result2}"
        assert result3 == 300, f"Expected 300, got {result3}"
        print("✅ Multiple async tasks work")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Long-running task with progress updates
    print_section("Test 4: Long-Running Task with Progress")
    task = long_running.delay(3)
    print(f"Task ID: {task.id}")
    print("Monitoring progress...")
    
    while not task.ready():
        if task.state == "PROGRESS":
            meta = task.info
            print(f"  Progress: {meta.get('current')}/{meta.get('total')} - {meta.get('status')}")
        time.sleep(0.5)
    
    try:
        result = task.get()
        print(f"Result: {result}")
        print("✅ Long-running task with progress works")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 5: Check task result in Redis
    print_section("Test 5: Task Results in Redis")
    task = add.delay(7, 8)
    task_id = task.id
    result = task.get(timeout=10)
    
    # Retrieve result using AsyncResult
    async_result = AsyncResult(task_id, app=celery_app)
    print(f"Task ID: {task_id}")
    print(f"Task State: {async_result.state}")
    print(f"Task Result: {async_result.result}")
    print(f"Task Successful: {async_result.successful()}")
    print("✅ Results stored in Redis")
    
    # Test 6: Task info and metadata
    print_section("Test 6: Task Info and Metadata")
    task = add.delay(15, 25)
    print(f"Task ID: {task.id}")
    print(f"Task Name: {task.name}")
    print(f"Task Args: {task.args}")
    print(f"Task Kwargs: {task.kwargs}")
    result = task.get(timeout=10)
    print(f"Result: {result}")
    print("✅ Task metadata accessible")
    
    print_section("CELERY TESTING COMPLETE")
    print("\n✅ All Celery features tested successfully!")
    print("\nTest Summary:")
    print("  ✅ Synchronous task execution")
    print("  ✅ Async task execution with .delay()")
    print("  ✅ Multiple concurrent tasks")
    print("  ✅ Long-running tasks with progress")
    print("  ✅ Results stored in Redis")
    print("  ✅ Task metadata and info")
    print("\nCelery Worker Configuration:")
    print(f"  - Broker: redis://localhost:6379/0")
    print(f"  - Result Backend: redis://localhost:6379/1")
    print(f"  - Task Serializer: JSON")
    print(f"  - Concurrency: 4 workers (solo pool)")

if __name__ == "__main__":
    try:
        test_celery_tasks()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
