"""
Test task execution modes
"""
from celery_app import celery_app
from tasks.test_task import add
import redis

print("="*70)
print("CONFIGURATION CHECK")
print("="*70)
print(f"task_always_eager: {celery_app.conf.task_always_eager}")
print(f"task_eager_propagates: {celery_app.conf.task_eager_propagates}")
print(f"broker_url: {celery_app.conf.broker_url}")
print(f"result_backend: {celery_app.conf.result_backend}")

print("\n" + "="*70)
print("REDIS BACKEND CHECK")
print("="*70)
try:
    # Check broker Redis (db=0)
    broker_redis = redis.Redis(host='localhost', port=6379, db=0)
    print(f"✅ Broker Redis (db=0) ping: {broker_redis.ping()}")
    
    # Check backend Redis (db=1)
    backend_redis = redis.Redis(host='localhost', port=6379, db=1)
    print(f"✅ Backend Redis (db=1) ping: {backend_redis.ping()}")
except Exception as e:
    print(f"❌ Redis error: {e}")

print("\n" + "="*70)
print("TASK EXECUTION TEST")
print("="*70)

print("\n1. Direct execution (should work):")
try:
    result = add(2, 3)
    print(f"   ✅ add(2, 3) = {result}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n2. Async execution with .delay():")
try:
    async_result = add.delay(5, 10)
    print(f"   Task ID: {async_result.id}")
    print(f"   Task state: {async_result.state}")
    
    if async_result.failed():
        print(f"   ❌ Task failed immediately")
        print(f"   Error info: {async_result.info}")
        
        # Try to get more details
        try:
            _ = async_result.result
        except Exception as e:
            print(f"   Exception: {type(e).__name__}: {e}")
    else:
        print(f"   ✅ Task state: {async_result.state}")
        
except Exception as e:
    print(f"   ❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Apply async (alternative method):")
try:
    async_result = add.apply_async(args=[7, 8])
    print(f"   Task ID: {async_result.id}")
    print(f"   Task state: {async_result.state}")
    
    if async_result.failed():
        print(f"   ❌ Task failed")
        try:
            _ = async_result.result
        except Exception as e:
            print(f"   Exception: {type(e).__name__}: {e}")
    else:
        print(f"   ✅ Task state: {async_result.state}")
        
except Exception as e:
    print(f"   ❌ Error: {type(e).__name__}: {e}")
