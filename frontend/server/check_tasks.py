"""
Check registered tasks
"""
from celery_app import celery_app

print("Registered tasks:")
for task_name in sorted(celery_app.tasks.keys()):
    print(f"  - {task_name}")

print(f"\nTotal: {len(celery_app.tasks)} tasks")

# Try to get the task directly
print("\nTrying to get 'tasks.test_task.add' task...")
try:
    task = celery_app.tasks.get('tasks.test_task.add')
    if task:
        print(f"✅ Found task: {task}")
        print(f"   Name: {task.name}")
        print(f"   Type: {type(task)}")
    else:
        print("❌ Task not found in registry")
except Exception as e:
    print(f"❌ Error: {e}")
