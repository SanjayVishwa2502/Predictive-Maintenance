"""
Redis and Celery diagnostics
"""
import redis
from celery_app import celery_app as app

def check_redis_connection():
    """Check if Redis is accessible"""
    print("\n" + "="*70)
    print("REDIS CONNECTION CHECK")
    print("="*70)
    
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        if r.ping():
            print("✅ Redis is running and accessible")
            
            # Check for Celery keys
            keys = r.keys('celery*')
            print(f"\nCelery keys in Redis: {len(keys)}")
            for key in keys[:10]:  # Show first 10
                print(f"  - {key.decode()}")
            
            # Check queue length
            queue_length = r.llen('celery')
            print(f"\nMessages in 'celery' queue: {queue_length}")
            
            # Check if there are any task results
            result_keys = r.keys('celery-task-meta-*')
            print(f"Task results in Redis: {len(result_keys)}")
            
            return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def check_celery_config():
    """Check Celery configuration"""
    print("\n" + "="*70)
    print("CELERY CONFIGURATION")
    print("="*70)
    
    print(f"Broker: {app.conf.broker_url}")
    print(f"Backend: {app.conf.result_backend}")
    print(f"Task serializer: {app.conf.task_serializer}")
    print(f"Result serializer: {app.conf.result_serializer}")
    print(f"Accept content: {app.conf.accept_content}")
    print(f"Task routes: {app.conf.task_routes}")
    print(f"Task default queue: {app.conf.task_default_queue}")
    print(f"Task default exchange: {app.conf.task_default_exchange}")
    print(f"Task default routing key: {app.conf.task_default_routing_key}")

def test_task_send():
    """Test sending a task"""
    print("\n" + "="*70)
    print("TASK SENDING TEST")
    print("="*70)
    
    from tasks.test_task import add
    
    print("\n1. Sending task with .delay()...")
    result = add.delay(5, 10)
    print(f"   Task ID: {result.id}")
    print(f"   Task state: {result.state}")
    
    print("\n2. Checking if task appears in Redis...")
    r = redis.Redis(host='localhost', port=6379, db=0)
    queue_length = r.llen('celery')
    print(f"   Queue length: {queue_length}")
    
    if queue_length > 0:
        print("   ✅ Task was queued in Redis")
        
        # Peek at the message
        message = r.lindex('celery', -1)
        if message:
            print(f"   Message preview: {message[:100]}")
    else:
        print("   ❌ Task was NOT queued in Redis")
    
    print("\n3. Checking task routing...")
    print(f"   Task name: {add.name}")
    print(f"   Task queue: {add.queue}")
    
    # Check if task should go to a specific queue based on routing
    if app.conf.task_routes:
        for pattern, route in app.conf.task_routes.items():
            print(f"   Route pattern '{pattern}': {route}")

if __name__ == "__main__":
    check_redis_connection()
    check_celery_config()
    test_task_send()
    
    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)
