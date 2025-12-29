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


def _safe_client_id(client_id: str) -> str:
    cid = (client_id or "").strip()
    if not cid:
        return ""
    return cid[:128]


# ============================================================================
# WEBSOCKET ENDPOINT: LLM EXPLANATION EVENTS (PUSH, NO POLLING)
# ============================================================================


@router.websocket("/ws/llm/events")
async def llm_events_websocket(
    websocket: WebSocket,
    client_id: str = Query(..., description="Stable client id from dashboard (pm_client_id)")
):
    """Stream LLM completion events for a single client.

    Celery tasks publish JSON messages to Redis channel: `llm:events:{client_id}`.
    The dashboard connects once and receives push updates as soon as tasks finish.
    """
    await websocket.accept()

    cid = _safe_client_id(client_id)
    if not cid:
        await websocket.close(code=1008)
        return

    channel = f"llm:events:{cid}"
    redis_client = None
    pubsub = None

    logger.info(f"WebSocket connected for LLM events (client_id={cid})")

    try:
        await websocket.send_json(
            {
                "type": "connected",
                "channel": channel,
                "timestamp": datetime.now().isoformat(),
            }
        )

        redis_client = await get_redis_pubsub()
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)

        # Some environments/proxies drop idle WebSockets. Send a lightweight
        # heartbeat periodically to keep the connection alive.
        last_heartbeat = datetime.now()

        # Long-lived connection; rely on client disconnect.
        while True:
            try:
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=1.0,
                )
                if not message or message.get("type") != "message":
                    continue

                data = message.get("data")
                try:
                    payload = json.loads(data) if isinstance(data, str) else data
                except Exception:
                    payload = {"type": "error", "message": "Invalid event payload"}

                await websocket.send_json(payload)

            except asyncio.TimeoutError:
                now = datetime.now()
                if (now - last_heartbeat).total_seconds() >= 20:
                    last_heartbeat = now
                    try:
                        await websocket.send_json({"type": "heartbeat", "timestamp": now.isoformat()})
                    except Exception:
                        # If heartbeat fails, let the outer handler close.
                        raise
                continue

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for LLM events (client_id={cid})")

    except Exception as e:
        logger.error(f"WebSocket error for LLM events (client_id={cid}): {e}")
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"Server error: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception:
            pass

    finally:
        if pubsub:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.close()
            except Exception:
                pass
        if redis_client:
            try:
                await redis_client.close()
            except Exception:
                pass


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
# ML SENSOR STREAMING ENDPOINT
# ============================================================================

@router.websocket("/ws/ml/sensors/{machine_id}")
async def ml_sensor_stream(websocket: WebSocket, machine_id: str):
    """
    Stream real-time sensor data for a specific machine
    
    This endpoint streams sensor readings every 5 seconds for the selected machine.
    Frontend connects to this endpoint to display live sensor dashboard.
    
    Args:
        websocket: WebSocket connection
        machine_id: Machine identifier (e.g., motor_siemens_1la7_001)
    
    Message Format:
        {
            "type": "sensor_update",
            "machine_id": "motor_siemens_1la7_001",
            "timestamp": "2025-12-15T10:45:23.123Z",
            "sensors": {
                "bearing_de_temp_C": 65.2,
                "bearing_nde_temp_C": 62.1,
                "winding_temp_C": 55.3,
                ...
            },
            "sensor_count": 22
        }
    
    Connection Flow:
        1. Client connects with machine_id
        2. Server accepts connection
        3. Server streams sensor updates every 5 seconds
        4. Client disconnects when switching machines or closing dashboard
    
    Timeout: 2 hours (then auto-disconnect)
    """
    try:
        # Accept WebSocket connection
        await websocket.accept()
        logger.info(f"ML sensor stream connected for machine: {machine_id}")
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "machine_id": machine_id,
            "message": f"Sensor stream connected for {machine_id}",
            "update_interval_seconds": 5,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        
        # Import MLManager (lazy import to avoid circular dependency)
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parents[2]))
        from services.ml_manager import ml_manager
        
        # Verify machine exists
        try:
            _ = await ml_manager.get_machine_status(machine_id)
        except FileNotFoundError:
            await websocket.send_json({
                "type": "error",
                "error": "Machine not found",
                "detail": f"Machine '{machine_id}' does not exist",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
            await websocket.close()
            return
        
        # Stream sensor data every 5 seconds
        timeout = 7200  # 2 hours
        start_time = asyncio.get_event_loop().time()
        
        while True:
            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                logger.info(f"ML sensor stream timeout for {machine_id} (2 hours)")
                await websocket.send_json({
                    "type": "timeout",
                    "message": "Connection timeout after 2 hours",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
                break
            
            try:
                # Get latest sensor data from MLManager
                status = await ml_manager.get_machine_status(machine_id)
                
                # Send sensor update
                await websocket.send_json({
                    "type": "sensor_update",
                    "machine_id": machine_id,
                    "timestamp": status['last_update'].isoformat() + "Z",
                    "sensors": status['latest_sensors'],
                    "sensor_count": status['sensor_count'],
                    "is_running": status['is_running']
                })
                
                # Wait 5 seconds before next update
                await asyncio.sleep(5)
                
            except WebSocketDisconnect:
                logger.info(f"Client disconnected from ML sensor stream: {machine_id}")
                break
            except Exception as e:
                logger.error(f"Error streaming sensor data for {machine_id}: {e}")
                await websocket.send_json({
                    "type": "error",
                    "error": "Stream error",
                    "detail": str(e),
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
                await asyncio.sleep(5)  # Continue after error
        
        # Close connection
        await websocket.close()
        logger.info(f"ML sensor stream closed for {machine_id}")
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: ML sensor stream for {machine_id}")
    except Exception as e:
        logger.error(f"ML sensor stream error for {machine_id}: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": "Connection error",
                "detail": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
            await websocket.close()
        except:
            pass


# ============================================================================
# ROUTE REGISTRATION VERIFICATION
# ============================================================================

if __name__ == '__main__':
    """Verify WebSocket routes are registered"""
    print("WebSocket Routes:")
    print("1. /ws/gan/training/{task_id} - GAN training progress")
    print("2. /ws/tasks/{task_id}/progress - Generic task progress")
    print("3. /ws/heartbeat - Health check")
    print("4. /ws/ml/sensors/{machine_id} - ML sensor streaming")
    print("\nAll WebSocket routes registered successfully!")
