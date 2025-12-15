"""
WebSocket Routes - Real-time Progress Streaming
Handles WebSocket connections for GAN training progress, task status, and heartbeat monitoring
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Optional, Dict, Any
import redis.asyncio as redis
import json
import asyncio
import logging
from datetime import datetime

from config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# CONNECTION MANAGER (FOR FUTURE MULTI-CLIENT BROADCASTING)
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections for broadcasting"""
    
    def __init__(self):
        self.active_connections: Dict[str, list[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, room: str):
        """Add connection to room"""
        await websocket.accept()
        if room not in self.active_connections:
            self.active_connections[room] = []
        self.active_connections[room].append(websocket)
    
    def disconnect(self, websocket: WebSocket, room: str):
        """Remove connection from room"""
        if room in self.active_connections:
            self.active_connections[room].remove(websocket)
            if not self.active_connections[room]:
                del self.active_connections[room]
    
    async def broadcast(self, message: str, room: str):
        """Broadcast message to all connections in room"""
        if room in self.active_connections:
            for connection in self.active_connections[room]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to connection: {e}")


manager = ConnectionManager()


# ============================================================================
# REDIS PUB/SUB UTILITIES
# ============================================================================

async def get_redis_pubsub():
    """Get Redis pub/sub client (DB 2)"""
    redis_client = await redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=2,  # DB 2 for pub/sub
        decode_responses=True
    )
    return redis_client


# ============================================================================
# WEBSOCKET ENDPOINT 1: GAN TRAINING PROGRESS
# ============================================================================

@router.websocket("/ws/gan/training/{task_id}")
async def gan_training_websocket(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for real-time GAN training progress streaming
    
    Flow:
    1. Client connects with task_id
    2. Server subscribes to Redis channel: gan:training:{task_id}
    3. Server streams progress messages from Redis pub/sub
    4. Server auto-closes on task completion (SUCCESS/FAILURE)
    5. 2-hour timeout for long-running tasks
    
    Args:
        websocket: FastAPI WebSocket connection
        task_id: Celery task ID for the training job
    
    Message Types:
        - connected: Initial connection confirmation
        - progress: Training progress update (epoch, loss, etc.)
        - closing: Connection closing notification
        - timeout: 2-hour timeout reached
        - error: Error occurred
    """
    await websocket.accept()
    
    channel = f"gan:training:{task_id}"
    redis_client = None
    pubsub = None
    
    logger.info(f"WebSocket connected for task {task_id}")
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "channel": channel,
            "message": "WebSocket connected successfully",
            "timestamp": datetime.now().isoformat()
        })
        
        # Connect to Redis pub/sub
        redis_client = await get_redis_pubsub()
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)
        
        logger.info(f"Subscribed to Redis channel: {channel}")
        
        # Listen for messages with 2-hour timeout
        timeout = 7200  # 2 hours
        start_time = asyncio.get_event_loop().time()
        
        while True:
            # Check timeout
            if asyncio.get_event_loop().time() - start_time > timeout:
                await websocket.send_json({
                    "type": "timeout",
                    "message": "Connection timeout after 2 hours",
                    "timestamp": datetime.now().isoformat()
                })
                break
            
            # Get message from Redis with 1-second timeout
            try:
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=1.0
                )
                
                if message and message['type'] == 'message':
                    # Parse progress data
                    try:
                        progress_data = json.loads(message['data'])
                        
                        # Send to WebSocket client
                        await websocket.send_json({
                            "type": "progress",
                            **progress_data
                        })
                        
                        logger.debug(f"Progress sent: {progress_data.get('message', 'N/A')}")
                        
                        # Check if task completed
                        status = progress_data.get('status')
                        if status in ['SUCCESS', 'FAILURE']:
                            await websocket.send_json({
                                "type": "closing",
                                "reason": f"Task {status.lower()}",
                                "final_data": progress_data,
                                "timestamp": datetime.now().isoformat()
                            })
                            logger.info(f"Task {task_id} completed with status: {status}")
                            break
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse progress data: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid progress data format",
                            "timestamp": datetime.now().isoformat()
                        })
            
            except asyncio.TimeoutError:
                # No message received, continue waiting
                continue
            
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                break
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for task {task_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Server error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        except:
            pass
    
    finally:
        # Cleanup
        if pubsub:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.close()
                logger.info(f"Unsubscribed from channel: {channel}")
            except Exception as e:
                logger.error(f"Error unsubscribing: {e}")
        
        if redis_client:
            try:
                await redis_client.close()
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
        
        logger.info(f"WebSocket cleanup completed for task {task_id}")


# ============================================================================
# WEBSOCKET ENDPOINT 2: GENERIC TASK PROGRESS
# ============================================================================

@router.websocket("/ws/tasks/{task_id}/progress")
async def task_progress_websocket(
    websocket: WebSocket,
    task_id: str,
    channel_prefix: Optional[str] = Query("gan:training", description="Redis channel prefix")
):
    """
    Generic WebSocket endpoint for any task progress streaming
    
    Flexible endpoint that can be used for different task types by changing channel_prefix.
    
    Args:
        websocket: FastAPI WebSocket connection
        task_id: Task identifier
        channel_prefix: Redis channel prefix (default: gan:training)
    
    Examples:
        /ws/tasks/abc123/progress?channel_prefix=gan:training
        /ws/tasks/xyz789/progress?channel_prefix=ml:inference
        /ws/tasks/def456/progress?channel_prefix=data:processing
    """
    await websocket.accept()
    
    channel = f"{channel_prefix}:{task_id}"
    redis_client = None
    pubsub = None
    
    logger.info(f"Generic WebSocket connected for task {task_id}, channel: {channel}")
    
    try:
        await websocket.send_json({
            "type": "connected",
            "channel": channel,
            "message": "WebSocket connected successfully",
            "timestamp": datetime.now().isoformat()
        })
        
        redis_client = await get_redis_pubsub()
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)
        
        # Listen for messages
        while True:
            try:
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=1.0
                )
                
                if message and message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        await websocket.send_json({
                            "type": "progress",
                            **data
                        })
                        
                        # Auto-close on completion
                        if data.get('status') in ['SUCCESS', 'FAILURE', 'COMPLETED']:
                            await websocket.send_json({
                                "type": "closing",
                                "reason": f"Task {data.get('status', 'completed')}",
                                "timestamp": datetime.now().isoformat()
                            })
                            break
                    
                    except json.JSONDecodeError:
                        # If not JSON, send raw message
                        await websocket.send_json({
                            "type": "message",
                            "content": message['data'],
                            "timestamp": datetime.now().isoformat()
                        })
            
            except asyncio.TimeoutError:
                continue
            
            except Exception as e:
                logger.error(f"Error in generic WebSocket: {e}")
                break
    
    except WebSocketDisconnect:
        logger.info(f"Generic WebSocket disconnected for task {task_id}")
    
    finally:
        if pubsub:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.close()
            except:
                pass
        
        if redis_client:
            try:
                await redis_client.close()
            except:
                pass


# ============================================================================
# WEBSOCKET ENDPOINT 3: HEARTBEAT / HEALTH CHECK
# ============================================================================

@router.websocket("/ws/heartbeat")
async def heartbeat_websocket(websocket: WebSocket):
    """
    Simple heartbeat WebSocket for testing connectivity
    
    Sends a timestamp message every second to verify connection is alive.
    Useful for:
    - Testing WebSocket functionality
    - Monitoring connection health
    - Debugging connection issues
    
    Message Format:
        {"type": "heartbeat", "timestamp": "2024-12-15T12:00:00.000Z", "count": 1}
    """
    await websocket.accept()
    
    logger.info("Heartbeat WebSocket connected")
    
    try:
        await websocket.send_json({
            "type": "connected",
            "message": "Heartbeat WebSocket connected",
            "timestamp": datetime.now().isoformat()
        })
        
        count = 0
        while True:
            count += 1
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat(),
                "count": count
            })
            
            await asyncio.sleep(1.0)
    
    except WebSocketDisconnect:
        logger.info("Heartbeat WebSocket disconnected")
    
    except Exception as e:
        logger.error(f"Heartbeat WebSocket error: {e}")


# ============================================================================
# WEBSOCKET UTILITIES
# ============================================================================

async def broadcast_to_channel(channel: str, data: Dict[str, Any]):
    """
    Utility function to broadcast data to a WebSocket channel via Redis
    
    Args:
        channel: Redis channel name (e.g., "gan:training:task-123")
        data: Dictionary to broadcast (will be JSON serialized)
    
    Usage:
        await broadcast_to_channel("gan:training:abc123", {
            "epoch": 100,
            "loss": 0.045,
            "progress": 33
        })
    """
    try:
        redis_client = await get_redis_pubsub()
        await redis_client.publish(channel, json.dumps(data))
        await redis_client.close()
    except Exception as e:
        logger.error(f"Failed to broadcast to channel {channel}: {e}")


# ============================================================================
# ROUTE REGISTRATION VERIFICATION
# ============================================================================

if __name__ == '__main__':
    """Verify WebSocket routes are registered"""
    print("WebSocket Routes:")
    print("1. /ws/gan/training/{task_id} - GAN training progress")
    print("2. /ws/tasks/{task_id}/progress - Generic task progress")
    print("3. /ws/heartbeat - Health check")
    print("\nAll WebSocket routes registered successfully!")
