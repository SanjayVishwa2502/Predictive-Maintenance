"""
Quick test to see task failure details
"""
from celery_app import celery_app
from tasks.test_task import add

print("Sending task...")
result = add.delay(5, 10)
print(f"Task ID: {result.id}")
print(f"Task state: {result.state}")

if result.failed():
    print(f"\n❌ Task failed!")
    print(f"Error: {result.info}")
    print(f"Traceback: {result.traceback}")
else:
    print(f"Task result: {result.result}")
