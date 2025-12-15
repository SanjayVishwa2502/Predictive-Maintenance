# PHASE 3.7: COMPREHENSIVE DASHBOARD IMPLEMENTATION
**Full System Integration & Frontend Development**
**Duration:** 6 weeks  
**Goal:** Build unified web dashboard integrating GAN, ML, and LLM subsystems  
**Status:** 🟡 READY TO START (November 27, 2025)

---

## Overview

**What Phase 3.7 Does:**
- Provides web interface for all predictive maintenance operations
- Integrates GAN (data generation), ML (predictions), LLM (explanations)
- Enables new machine onboarding via wizard interface
- Displays real-time predictions for entire fleet (26 machines)
- Generates maintenance reports with AI explanations

**Deliverables:**
- React frontend with 10+ pages across 3 modules
- FastAPI backend with RESTful API
- PostgreSQL database for metadata
- Celery workers for async tasks
- Docker deployment configuration
- Complete API documentation (Swagger)

**Architecture:**
```
Frontend (React) ↔ Backend API (FastAPI) ↔ Existing Systems (GAN/ML/LLM)
       ↓                    ↓                         ↓
  Browser UI         Celery Workers           File System (Parquet/Models)
                           ↓
                    Redis + PostgreSQL
```

---

## Scope Adjustment: VLM Handled Externally

- **Status:** VLM (Vision / VLM integration) is being developed and owned by a colleague and is *out of scope* for the Phase 3.7 MVP. Do not implement VLM-specific UI or ingestion pipelines in this phase — instead, provide lightweight integration hooks (API endpoints and a data contract) so the VLM work can be connected later.
- **Action:** Add a `vlm/` API contract doc and a `TODO` in code comments noting the external owner and expected integration points (embedding format, REST endpoint, auth). Keep VLM fields optional in database schemas and UI models.

## Recommended Frontend Subfolders (Minimal, GAN-First)

- `frontend/client/` (React app)
  - `src/modules/gan/` — GAN-focused pages, components, and hooks (onboarding, seed generation, training status, synthetic data viewer)
  - `src/modules/ml/` — placeholder components for future ML pages (keep minimal stubs; defer full implementation)
  - `src/modules/llm/` — placeholders for explanation panels and RAG integrations (deferred)
  - `src/components/` — shared UI components (charts, tables, forms)
  - `src/services/` — client-side API wrappers; include `ganApi.ts` with endpoint contracts

- `frontend/server/` (FastAPI app)
  - `api/routes/gan.py` — GAN endpoints (train, generate, status, validate)
  - `api/contracts/` — JSON schema or OpenAPI snippets describing VLM contract (kept optional)
  - `tasks/gan_tasks.py` — Celery tasks for long-running GAN jobs
  - `services/gan_manager.py` — orchestration calling existing `GAN/` scripts

---

## Prerequisites

**Completed Phases:**
- ✅ Phase 1: GAN data generation (26 machines operational)
- ✅ Phase 2: ML models trained (4 model types functional)
- ✅ Phase 3.0-3.6: LLM integration complete (GPU-accelerated)

**Hardware Requirements:**
- ✅ RTX 4070 GPU (for LLM inference)
- ✅ 16GB RAM minimum
- ✅ 100GB disk space (for frontend build + databases)

**Software Stack:**
- Python 3.11+
- Node.js 18+ (for React frontend)
- PostgreSQL 15+
- Redis 7+
- Docker + Docker Compose

**Existing Codebase Status:**
- ✅ `GAN/`: 18 scripts operational, 26 machines validated
- ✅ `ml_models/`: 4 inference classes working
- ✅ `LLM/`: `IntegratedPredictionSystem` functional with GPU acceleration
- ✅ GPU Fix: CUDA DLL injection working (~26 tok/s)

---

## Project Structure

**New Directories to Create:**
```
frontend/
├── client/                  # React frontend
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
│
├── server/                  # FastAPI backend
│   ├── api/
│   ├── tasks/
│   ├── db/
│   ├── main.py
│   └── requirements.txt
│
├── docker-compose.yml
├── nginx.conf
└── .env.example
```

**Integration Points:**
- `frontend/server/api/services/gan_manager.py` → Calls `GAN/scripts/*.py`
- `frontend/server/api/services/ml_manager.py` → Imports `ml_models/scripts/inference/*.py`
- `frontend/server/api/services/llm_manager.py` → Imports `LLM/api/ml_integration.py`

---

## PHASE 3.7.1: FOUNDATION SETUP
**Duration:** Week 1 (Days 1-7)  
**Goal:** Setup development environment, database, and authentication

### Phase 3.7.1.1: Project Initialization (Days 1-2)

**Tasks:**
- [✅] Initialize React project with Vite + TypeScript
- [✅] Initialize FastAPI project structure
- [✅] Setup version control (.gitignore for node_modules, __pycache__)
- [✅] Create environment variable templates (.env.example)
- [✅] Install base dependencies

**Frontend Dependencies (package.json):**
- react ^18.2.0
- react-router-dom ^6.20.0
- @mui/material ^5.14.0
- @tanstack/react-query ^5.0.0
- zustand ^4.4.0
- chart.js ^4.4.0
- axios ^1.6.0
- typescript ^5.3.0

**Backend Dependencies (requirements.txt):**
- fastapi ^0.104.0
- uvicorn[standard] ^0.24.0
- sqlalchemy ^2.0.0
- psycopg2-binary ^2.9.9
- celery ^5.3.0
- redis ^5.0.0
- pydantic ^2.5.0
- python-jose[cryptography] ^3.3.0

**Expected Output:**
- ✅ `frontend/client/` directory with React app
- ✅ `frontend/server/` directory with FastAPI app
- ✅ Both projects install dependencies successfully

---

### Phase 3.7.1.2: Database Setup (Days 3-4)

**Tasks:**
- [✅] Install PostgreSQL 15+ (or use Docker)
- [✅] Create database: `predictive_maintenance`
- [✅] Create SQLAlchemy models for 5 tables
- [✅] Create Alembic migration scripts
- [✅] Run initial migration
- [✅] Setup Redis for Celery

**Database Schema:**

**Table 1: machines**
- `id` (UUID, PK)
- `machine_id` (VARCHAR, unique)
- `machine_type` (VARCHAR)
- `manufacturer` (VARCHAR)
- `model` (VARCHAR)
- `metadata_path` (VARCHAR)
- `created_at` (TIMESTAMP)
- `updated_at` (TIMESTAMP)

**Table 2: gan_training_jobs**
- `id` (UUID, PK)
- `machine_id` (VARCHAR, FK)
- `epochs` (INT)
- `status` (VARCHAR: pending|running|completed|failed)
- `loss_history` (JSON)
- `started_at` (TIMESTAMP)
- `completed_at` (TIMESTAMP)

**Table 3: predictions**
- `id` (UUID, PK)
- `machine_id` (VARCHAR, FK)
- `prediction_type` (VARCHAR: classification|rul|anomaly|timeseries)
- `input_data` (JSON)
- `prediction_result` (JSON)
- `confidence` (FLOAT)
- `timestamp` (TIMESTAMP)

**Table 4: explanations**
- `id` (UUID, PK)
- `prediction_id` (UUID, FK)
- `explanation_text` (TEXT)
- `recommendations` (JSON)
- `created_at` (TIMESTAMP)

**Table 5: model_versions**
- `id` (UUID, PK)
- `model_type` (VARCHAR)
- `version` (VARCHAR)
- `file_path` (VARCHAR)
- `metrics` (JSON: accuracy, f1, etc.)
- `trained_at` (TIMESTAMP)
- `is_active` (BOOLEAN)

**File:** `frontend/server/db/models.py`
**File:** `frontend/server/db/crud.py` (CRUD operations)
**File:** `frontend/server/database.py` (connection setup)

**Migration Commands:**
```powershell
cd frontend/server
alembic init alembic
alembic revision --autogenerate -m "Initial schema"
alembic upgrade head
```

**Expected Output:**
- ✅ Database created with 5 tables
- ✅ Alembic migrations folder created
- ✅ Redis running on localhost:6379

---

### Phase 3.7.1.3: Authentication Setup (Days 5-6)

**Tasks:**
- [❌] Create User model (id, username, hashed_password, role)
- [❌] Implement password hashing (passlib + bcrypt)
- [❌] Implement JWT token generation (python-jose)
- [❌] Create auth endpoints: /register, /login, /refresh
- [❌] Create authentication middleware
- [❌] Add role-based access control (admin, operator, viewer)

**API Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/auth/register` | POST | Create new user |
| `/api/auth/login` | POST | Get JWT token |
| `/api/auth/refresh` | POST | Refresh token |
| `/api/auth/me` | GET | Get current user |

**JWT Payload:**
```
{
  "sub": "user_id",
  "username": "admin",
  "role": "admin",
  "exp": 1732723200
}
```

**File:** `frontend/server/api/routes/auth.py`
**File:** `frontend/server/utils/security.py` (password hashing, JWT)
**File:** `frontend/server/dependencies.py` (get_current_user)

**Frontend Integration:**
- Store JWT in localStorage
- Add Authorization header to all API calls
- Create login page component
- Create protected route wrapper

**Expected Output:**
- ✅ User registration working
- ✅ Login returns JWT token
- ✅ Protected endpoints verify token
- ✅ Frontend redirects to login if unauthorized

---

### Phase 3.7.1.4: Celery Worker Setup (Day 7)

**Tasks:**
- [✅] Configure Celery app with Redis broker
- [✅] Create worker configuration
- [✅] Test basic task execution
- [✅] Setup Flower for monitoring

**File:** `frontend/server/celery_app.py`
**File:** `frontend/server/tasks/__init__.py`

**Worker Configuration:**
- Broker: Redis (localhost:6379/0)
- Result backend: Redis (localhost:6379/1)
- Task serializer: JSON
- Concurrency: 4 workers

**Test Task:**
- Create simple task: `tasks/test_task.py`
- Function: `add(x, y)` returns x + y
- Test async execution: `add.delay(4, 5)`

**Start Worker Command:**
```powershell
cd frontend/server
celery -A celery_app worker --loglevel=info --pool=solo
```

**Expected Output:**
- ✅ Celery worker starts without errors
- ✅ Test task executes successfully
- ✅ Task results stored in Redis

---

## PHASE 3.7.2: GAN INTEGRATION
**Duration:** Week 2 (Days 8-14)  
**Goal:** Build GAN module with New Machine Wizard

### Phase 3.7.2.1: GAN Manager Service - Industrial Grade (Days 8-9) ✅ UPGRADED

**Industrial-Grade Implementation:**

**File:** `GAN/services/gan_manager.py` (~550 lines)

**Architecture Patterns Applied:**
- ✅ **Singleton Pattern** - Single instance for resource efficiency
- ✅ **LRU Caching** - Cache up to 5 TVAE models in memory
- ✅ **Result Classes** - Structured output objects
- ✅ **Comprehensive Logging** - All operations logged
- ✅ **Error Handling** - Defensive programming with detailed exceptions
- ✅ **Performance Tracking** - Operation counters and metrics

**Result Classes (Dataclasses):**

```python
@dataclass
class SeedGenerationResult:
    machine_id: str
    samples_generated: int
    file_path: str
    file_size_mb: float
    generation_time_seconds: float
    timestamp: str
    
    def to_dict() -> Dict

@dataclass
class SyntheticGenerationResult:
    machine_id: str
    train_samples: int
    val_samples: int
    test_samples: int
    train_file: str
    val_file: str
    test_file: str
    generation_time_seconds: float
    timestamp: str
    
    def to_dict() -> Dict

@dataclass
class TVAEModelMetadata:
    machine_id: str
    model_path: str
    is_trained: bool
    epochs: int
    loss: Optional[float]
    training_time_seconds: Optional[float]
    trained_at: Optional[str]
    num_features: int
    
    def to_dict() -> Dict
```

**GANManager Class (Singleton):**

```python
class GANManager:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize once"""
        if GANManager._initialized:
            return
        GANManager._initialized = True
        
        # Base paths
        self.gan_root = Path("GAN")
        self.models_path = self.gan_root / "models"
        self.seed_data_path = self.gan_root / "seed_data"
        self.synthetic_data_path = self.gan_root / "data"
        self.metadata_path = self.gan_root / "metadata"
        
        # Model cache (LRU, max 5)
        self.model_cache: Dict[str, Any] = {}
        self.max_cache_size = 5
        
        # Performance tracking
        self.operation_count = 0
        self.seed_generations = 0
        self.synthetic_generations = 0
        self.model_trainings = 0
```

**Core Methods:**

| Method | Purpose | Returns | Error Handling |
|--------|---------|---------|----------------|
| `generate_seed_data(machine_id, samples)` | Generate temporal seed data | `SeedGenerationResult` | ValueError, FileNotFoundError, RuntimeError |
| `train_tvae_model(machine_id, epochs)` | Train TVAE model | `TVAEModelMetadata` | ValueError, FileNotFoundError, RuntimeError |
| `generate_synthetic_data(machine_id, train, val, test)` | Generate datasets | `SyntheticGenerationResult` | ValueError, FileNotFoundError, RuntimeError |
| `get_model_metadata(machine_id)` | Get TVAE model info | `TVAEModelMetadata` | FileNotFoundError |
| `list_available_machines()` | List all machines | `List[str]` | None |
| `get_statistics()` | Get service stats | `Dict[str, Any]` | None |
| `clear_cache()` | Clear model cache | `None` | None |
| `_load_tvae_model(machine_id)` | Load model with LRU cache | TVAE model | FileNotFoundError, RuntimeError |

**Input Validation:**

```python
def generate_seed_data(self, machine_id: str, samples: int = 10000):
    if not machine_id:
        raise ValueError("machine_id cannot be empty")
    if samples <= 0:
        raise ValueError("samples must be positive")
    # ... implementation
```

**Performance Features:**

1. **LRU Caching:**
   ```python
   @lru_cache(maxsize=5)
   def _load_tvae_model(self, machine_id: str):
       # Load model from disk
       # Cached for future use
   ```

2. **Performance Tracking:**
   ```python
   self.operation_count += 1
   self.seed_generations += 1
   # Track all operations
   ```

3. **Statistics API:**
   ```python
   def get_statistics(self) -> Dict:
       return {
           'total_operations': self.operation_count,
           'seed_generations': self.seed_generations,
           'synthetic_generations': self.synthetic_generations,
           'model_trainings': self.model_trainings,
           'cached_models': len(self.model_cache),
           'available_machines': len(self.list_available_machines())
       }
   ```

**Logging:**

```python
logger.info(f"Generating {samples} seed samples for {machine_id}...")
logger.info(f"✅ Seed data generated: {file_size_mb:.2f} MB in {time:.2f}s")
logger.error(f"Seed generation failed: {e}")
```

**Integration with GAN Scripts:**

```python
from GAN.scripts.create_temporal_seed_data import generate_temporal_seed_data
from GAN.scripts.retrain_tvae_temporal import retrain_tvae_temporal
from GAN.scripts.generate_from_temporal_tvae import generate_from_temporal_tvae
```

**Singleton Instance Export:**

```python
# Create singleton instance
gan_manager = GANManager()
```

**Expected Output:**
- ✅ GANManager singleton initialized
- ✅ LRU cache functional (max 5 models)
- ✅ All methods return structured result objects
- ✅ Comprehensive error handling with specific exceptions
- ✅ All operations logged with timestamps
- ✅ Performance metrics tracked
- ✅ Unit tests coverage >80%

**Quality Metrics:**
- **Lines of Code:** ~550 (well-structured)
- **Methods:** 8 public + 1 private (cached)
- **Result Classes:** 3 dataclasses
- **Error Types:** 3 (ValueError, FileNotFoundError, RuntimeError)
- **Logging:** INFO for operations, ERROR for failures
- **Cache:** LRU with max 5 models

---

### Phase 3.7.2.2: GAN API Routes - Professional Implementation (Days 10-11) ✅ UPGRADED

**Professional API Implementation:**

**File:** `frontend/server/api/routes/gan.py` (700+ lines with enhancements)
**File:** `frontend/server/api/models/gan.py` (400+ lines)

**Architecture Enhancements:**
- ✅ **Rate Limiting** - 100 requests/minute per IP (Redis-based)
- ✅ **Response Caching** - 30s TTL for list endpoints (Redis)
- ✅ **Comprehensive Error Handling** - HTTP status codes with detailed messages
- ✅ **OpenAPI Documentation** - Complete Swagger UI with examples
- ✅ **Input Validation** - Pydantic models with field validators
- ✅ **Dependency Injection** - Clean separation of concerns
- ✅ **Integration Tests** - >80% coverage

**API Endpoints (13 Total):**

#### Profile Management

| Endpoint | Method | Rate Limited | Cached | Purpose |
|----------|--------|--------------|--------|---------|
| `/api/gan/templates` | GET | ✅ | ✅ (30s) | List all machine profile templates |
| `/api/gan/templates/{machine_type}` | GET | ✅ | ✅ (30s) | Get template for specific machine type |
| `/api/gan/templates/{machine_type}/download` | GET | ✅ | ❌ | Download template file |
| `/api/gan/profiles/upload` | POST | ✅ | ❌ | Upload profile (JSON/YAML/Excel) |
| `/api/gan/profiles/{id}/validate` | POST | ✅ | ❌ | Validate uploaded profile |
| `/api/gan/profiles/{id}/edit` | PUT | ✅ | ❌ | Edit profile after validation |

#### Machine Management

| Endpoint | Method | Rate Limited | Cached | Purpose |
|----------|--------|--------------|--------|---------|
| `/api/gan/machines` | POST | ✅ | ❌ | Create machine from profile |
| `/api/gan/machines` | GET | ✅ | ✅ (30s) | List all machines |
| `/api/gan/machines/{id}` | GET | ✅ | ✅ (30s) | Get machine details |
| `/api/gan/machines/{id}/status` | GET | ✅ | ❌ | Get workflow status |
| `/api/gan/machines/{id}` | DELETE | ✅ | ❌ | Delete machine and data |

#### Workflow Operations

| Endpoint | Method | Rate Limited | Cached | Purpose |
|----------|--------|--------------|--------|---------|
| `/api/gan/machines/{id}/seed` | POST | ✅ | ❌ | Generate seed data (sync) |
| `/api/gan/machines/{id}/train` | POST | ✅ | ❌ | Train TVAE (async/Celery) |
| `/api/gan/machines/{id}/generate` | POST | ✅ | ❌ | Generate synthetic data |
| `/api/gan/machines/{id}/validate` | GET | ✅ | ❌ | Validate data quality |

#### Monitoring

| Endpoint | Method | Rate Limited | Cached | Purpose |
|----------|--------|--------------|--------|---------|
| `/api/gan/tasks/{task_id}` | GET | ✅ | ❌ | Get Celery task status |
| `/api/gan/health` | GET | ❌ | ❌ | Service health check |

**Enhanced Pydantic Models with Validation:**

```python
# Request Models with Field Validators
class ProfileUploadRequest(BaseModel):
    machine_id: str = Field(..., min_length=3, max_length=100, 
                           pattern="^[a-z0-9_]+$")
    machine_type: str = Field(..., min_length=2)
    sensors: List[SensorConfig] = Field(..., min_items=1, max_items=50)
    
    @validator('machine_id')
    def validate_machine_id_format(cls, v):
        if not v.islower():
            raise ValueError('machine_id must be lowercase')
        return v

class SeedGenerationRequest(BaseModel):
    samples: int = Field(10000, ge=1000, le=100000,
                        description="Number of samples (1K-100K)")

class TrainingRequest(BaseModel):
    epochs: int = Field(300, ge=50, le=1000,
                       description="Training epochs (50-1000)")
    batch_size: int = Field(500, ge=100, le=2000)

class GenerationRequest(BaseModel):
    train_samples: int = Field(35000, ge=1000, le=100000)
    val_samples: int = Field(7500, ge=100, le=20000)
    test_samples: int = Field(7500, ge=100, le=20000)
    
    @validator('val_samples', 'test_samples')
    def validate_split_ratio(cls, v, values):
        train = values.get('train_samples', 0)
        if v > train * 0.5:
            raise ValueError('Val/Test samples too large relative to train')
        return v

# Response Models with Examples
class SeedGenerationResponse(BaseModel):
    machine_id: str
    samples_generated: int
    file_path: str
    file_size_mb: float
    generation_time_seconds: float
    timestamp: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "machine_id": "motor_siemens_1la7_001",
                "samples_generated": 10000,
                "file_path": "GAN/seed_data/motor_siemens_1la7_001_temporal_seed.parquet",
                "file_size_mb": 2.45,
                "generation_time_seconds": 12.34,
                "timestamp": "2024-12-13T16:30:00Z"
            }
        }

class TrainingResponse(BaseModel):
    success: bool
    machine_id: str
    task_id: str
    epochs: int
    estimated_time_minutes: float
    websocket_url: str
    message: str
    
    class Config:
        schema_extra = {
            "example": {
                "success": true,
                "machine_id": "motor_siemens_1la7_001",
                "task_id": "a1b2c3d4-e5f6-7890",
                "epochs": 300,
                "estimated_time_minutes": 4.0,
                "websocket_url": "/ws/gan/training/a1b2c3d4-e5f6-7890",
                "message": "Training started successfully"
            }
        }
```

**Rate Limiting Implementation:**

```python
from fastapi import Request, HTTPException
import redis.asyncio as redis

RATE_LIMIT = 100  # requests per minute
RATE_WINDOW = 60  # seconds

async def rate_limiter(request: Request):
    """Rate limiter dependency"""
    client_ip = request.client.host
    key = f"ratelimit:gan:{client_ip}"
    
    redis_client = await get_redis()
    count = await redis_client.incr(key)
    
    if count == 1:
        await redis_client.expire(key, RATE_WINDOW)
    
    if count > RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again in a minute."
        )

# Apply to endpoints
@router.post("/machines/{id}/train", dependencies=[Depends(rate_limiter)])
async def train_model(...):
    ...
```

**Error Handling with HTTP Status Codes:**

```python
@router.post("/machines/{id}/seed")
async def generate_seed(machine_id: str, request: SeedGenerationRequest):
    try:
        result = gan_manager.generate_seed_data(machine_id, request.samples)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
```

**OpenAPI Documentation:**

```python
@router.post(
    "/machines/{machine_id}/train",
    response_model=TrainingResponse,
    summary="Train TVAE Model",
    description="""
    Train TVAE model on temporal seed data (asynchronous via Celery).
    
    **Requirements:**
    - Machine must exist
    - Seed data must be generated first
    
    **Returns:**
    - Celery task_id for progress tracking
    - WebSocket URL for real-time updates
    - Estimated completion time
    
    **Typical Duration:** ~4 minutes for 300 epochs
    """,
    responses={
        200: {"description": "Training started successfully"},
        400: {"description": "Seed data not found"},
        404: {"description": "Machine not found"},
        429: {"description": "Rate limit exceeded"}
    }
)
async def train_model(...):
    ...
```

**Integration Tests:**

**File:** `frontend/server/tests/test_gan_api.py`

```python
class TestGANApi(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
    
    @patch('frontend.server.api.routes.gan.gan_manager')
    def test_generate_seed_success(self, mock_gan_manager):
        mock_gan_manager.generate_seed_data.return_value = SeedGenerationResult(...)
        
        response = self.client.post(
            "/api/gan/machines/motor_001/seed",
            json={"samples": 10000}
        )
        
        assert response.status_code == 200
        assert response.json()["samples_generated"] == 10000
    
    def test_rate_limiting(self):
        # Make 101 requests
        for i in range(101):
            response = self.client.get("/api/gan/machines")
        
        assert response.status_code == 429
```

**Expected Output:**
- ✅ All 17 endpoints accessible via Swagger UI (`/docs`)
- ✅ Rate limiting active (100 req/min per IP)
- ✅ Response caching for list endpoints (30s TTL)
- ✅ Input validation with helpful error messages
- ✅ HTTP status codes correct (200, 400, 404, 429, 500)
- ✅ OpenAPI documentation complete with examples
- ✅ Integration tests coverage >80%
- ✅ Response times <500ms (excluding training)

---

### Phase 3.7.2.3: GAN Celery Tasks (Day 12) ✅ COMPLETE

**Tasks:**
- [✅] Create async task for TVAE training
- [✅] Create async task for data generation
- [✅] Implement progress broadcasting to Redis
- [✅] Add task result storage

**Completed Files:**
- `tasks/gan_tasks.py` (450+ lines) - 3 Celery tasks with progress tracking
- `api/models/gan.py` - Added TaskStatusResponse model
- `api/routes/gan.py` - Updated training endpoint, added task status endpoint
- `test_gan_tasks.py` - Test script for verification

**File:** `frontend/server/tasks/gan_tasks.py`

**Task 1: train_tvae_task(machine_id, epochs)** ✅
- Calls `GANManager.train_tvae_model()` with streaming
- Parses epoch/loss from script output
- Broadcasts progress to Redis every 10 epochs
- Stores final result with model path, training time, final loss
- Full error handling and progress tracking

**Task 2: generate_data_task(machine_id, samples)** ✅
- Calls `GANManager.generate_synthetic_data()`
- Updates progress through 3 stages
- Returns file paths and statistics
- Broadcasts completion/failure to Redis

**Task 3: generate_seed_data_task(machine_id, samples)** ✅
- Simple wrapper for seed generation
- No streaming needed (fast operation)

**Progress Broadcasting:** ✅
- Channel: `gan:training:{task_id}`
- Message format: `{task_id, timestamp, epoch, loss, progress, status, metadata}`
- Redis DB 2 for pub/sub
- broadcast_progress() helper function
- ProgressTask base class with update_progress() method

**Task Status Tracking:** ✅
- GET `/api/gan/tasks/{task_id}` endpoint
- get_task_status() helper function
- TaskStatusResponse Pydantic model
- Supports polling for frontend

**API Integration:** ✅
- POST `/api/gan/machines/{id}/train` now uses train_tvae_task.delay()
- Returns real Celery task_id
- Checks machine exists and seed data available before starting

**Expected Output:**
- ✅ Training task runs in background
- ✅ Progress updates broadcast to Redis every 10 epochs
- ✅ Task status available via GET /api/gan/tasks/{task_id}
- ✅ Task completes successfully with model path and statistics

---

### Phase 3.7.2.4: GAN WebSocket Handler (Day 13) ✅ COMPLETE

**Tasks:**
- [✅] Create WebSocket endpoint for training progress
- [✅] Subscribe to Redis channel
- [✅] Stream updates to client
- [✅] Handle connection errors

**Completed Files:**
- `api/routes/websocket.py` (350+ lines) - WebSocket router with 3 endpoints
- `main.py` - Added WebSocket router registration
- `websocket_test.html` - Interactive test client with progress bar

**Endpoint:** `/ws/gan/training/{task_id}`

**File:** `frontend/server/api/routes/websocket.py`

**WebSocket Endpoints:** ✅

1. **`/ws/gan/training/{task_id}`** - GAN training progress stream
   - Subscribes to Redis channel `gan:training:{task_id}`
   - Streams real-time progress updates
   - Closes on task completion/failure
   - 2-hour timeout

2. **`/ws/tasks/{task_id}/progress?channel_prefix=...`** - Generic task progress
   - Flexible endpoint for any task type
   - Configurable channel prefix

3. **`/ws/heartbeat`** - Health check/connectivity test
   - Sends timestamp every second
   - Useful for testing WebSocket functionality

**WebSocket Flow:** ✅
1. Client connects with task_id
2. Server accepts connection and subscribes to Redis channel `gan:training:{task_id}`
3. Server streams progress messages from Redis pub/sub
4. Server auto-closes on task completion (SUCCESS/FAILURE)
5. Server handles disconnects and cleanup

**Message Types:**
```json
{
  "type": "connected",
  "channel": "gan:training:550e8400-...",
  "message": "WebSocket connected successfully"
}

{
  "type": "progress",
  "current": 150,
  "total": 300,
  "progress": 50,
  "status": "RUNNING",
  "machine_id": "motor_siemens_001",
  "stage": "training",
  "epoch": 150,
  "loss": 0.0452,
  "message": "Epoch 150/300, Loss: 0.0452"
}

{
  "type": "closing",
  "reason": "Task success",
  "timestamp": "2024-12-03T10:45:23.123Z"
}
```

**Features Implemented:** ✅
- Async Redis pub/sub with `redis.asyncio`
- Automatic subscription cleanup on disconnect
- 2-hour connection timeout
- Error handling and logging
- Multiple message types (connected, progress, closing, timeout, error)
- Auto-close on task completion
- ConnectionManager class (for future multi-client broadcasting)

**Test Client:** ✅
- `websocket_test.html` - Beautiful interactive UI
- Real-time progress bar
- Epoch/loss metrics display
- Scrollable message logs
- Connect/disconnect controls
- Heartbeat test button
- Auto-reconnect (3 attempts)

**Expected Output:**
- ✅ WebSocket connection established in <100ms
- ✅ Progress messages streamed in real-time
- ✅ Connection closes gracefully on task completion
- ✅ Automatic cleanup of Redis subscriptions

---



## PHASE 3.7.3: ML DASHBOARD WITH PROFESSIONAL UI/UX
**Duration:** Week 3 (Days 15-21) - Extended: 5-6 days  
**Goal:** Build production-grade ML prediction dashboard with modern professional design

### Executive Overview

**Objective:** Develop a production-grade Machine Learning Dashboard that integrates trained ML models with a modern, professional user interface for real-time predictive maintenance monitoring across 26 industrial machines.

**Timeline:** 5-6 days (Extended for professional design implementation)  
**Team Size:** 1-2 developers  
**Technology Stack:** React 19 + TypeScript, FastAPI, MUI v7, Recharts, Framer Motion

**Success Criteria:**
- ✅ All 26 machines display real-time health predictions
- ✅ Classification accuracy visible with confidence scores ≥85%
- ✅ Professional UI matching enterprise design standards
- ✅ Response time <500ms for predictions
- ✅ LLM-powered explanations for all predictions

---

## 🎨 Design System Specifications

### Color Palette (Professional Dark Theme)

**Primary Colors:**
- Primary Blue: `#667eea`
- Primary Purple: `#764ba2`
- Gradient: `linear-gradient(135deg, #667eea 0%, #764ba2 100%)`

**Status Colors:**
- Healthy (State 0): `#10b981` (Green)
- Degrading (State 1): `#fbbf24` (Yellow)
- Warning (State 2): `#f97316` (Orange)
- Critical (State 3): `#ef4444` (Red)

**Background Palette:**
- App Background: `#0f172a`
- Sidebar: `linear-gradient(180deg, #1a1a2e 0%, #16213e 100%)`
- Card Background: `#1f2937`
- Header: `#16213e`

**Typography:**
- Font Family: 'Inter', sans-serif
- Heading Sizes: H1(32px), H2(24px), H3(18px)
- Body: 16px, Small: 14px, Tiny: 12px

---

## 📐 Layout Architecture

### Sidebar Navigation (240px width)
```
┌─────────────────────┐
│  🔧 Predictive      │ ← Logo (80px height)
│     Maintenance     │
├─────────────────────┤
│  📊 Dashboard       │ ← Active (gradient bg)
│  📁 Data Management │
│     └ Machines      │
│     └ New Machine   │
│     └ Data Explorer │
│  📈 Analytics       │
│  🤖 ML Models       │
│  💬 AI Assistant    │
│  ⚙️ Settings        │
└─────────────────────┘
```

### Main Content Layout
```
┌─────────────────────────────────────────────────────┐
│  Breadcrumb: Dashboard / Machine Health    [🔍]👤  │ ← Header (60px)
├─────────────────────────────────────────────────────┤
│  Fleet Health Overview                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐│
│  │    20    │ │    3     │ │    2     │ │   1    ││
│  │ Healthy  │ │Degrading │ │ Warning  │ │Critical││
│  │  ↑ +2    │ │  ↑ +1    │ │  → 0     │ │ ↑ +1   ││
│  └──────────┘ └──────────┘ └──────────┘ └────────┘│
├─────────────────────────────────────────────────────┤
│  [Search] [Filter: All ▼] [Sort: Status ▼]         │
│                                                      │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────┐│
│  │Motor Siemens │ │Pump Grundfos │ │CNC DMG Mori ││
│  │1LA7 001      │ │CR3 004       │ │NLX 010      ││
│  │              │ │              │ │             ││
│  │● Healthy 95% │ │● Degrad. 87% │ │● Healthy 92%││
│  │🌡️45°C 📊2.1mm│ │🌡️62°C 📊4.3mm│ │🌡️41°C 📊1.8││
│  │[Details][AI] │ │[Details][AI] │ │[Details][AI]││
│  └──────────────┘ └──────────────┘ └─────────────┘│
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Phased Implementation Plan

### **Phase 1: Backend Infrastructure (Day 15-16)**

#### Day 15.1: ML Manager Service Development

**File:** `frontend/server/api/services/ml_manager.py`

**Tasks:**
- [ ] Create `MLManager` class with singleton pattern
- [ ] Implement model loading from `ml_models/models/classification/`
- [ ] Add LRU cache for model instances (max 5 models in memory)
- [ ] Implement graceful error handling for missing models
- [ ] Add logging for all operations
- [ ] Write unit tests (>80% coverage)

**Class Structure:**
```python
class MLManager:
    """
    Centralized ML model management service
    Handles model loading, caching, and inference
    """
    def __init__(self):
        self.classification_models = {}  # Cache loaded models
        self.anomaly_models = {}
        self.model_metadata = {}
        
    def load_models(self) -> Dict[str, bool]:
        """Load all trained ML models into memory"""
        
    def predict_classification(
        self, 
        machine_id: str, 
        sensor_data: Dict[str, float]
    ) -> ClassificationResult:
        """Run classification inference"""
        
    def predict_anomaly(
        self, 
        machine_id: str, 
        sensor_data: Dict[str, float]
    ) -> AnomalyResult:
        """Run anomaly detection"""
        
    def get_model_info(self, machine_id: str) -> ModelMetadata:
        """Retrieve model metadata"""
```

**Dependencies:**
```python
from ml_models.scripts.inference.predict_classification import ClassificationInference
from ml_models.scripts.inference.predict_anomaly import AnomalyInference
import joblib
from functools import lru_cache
```

**Success Metrics:**
- Model loading time: <2 seconds per model
- Inference time: <100ms per prediction
- Memory usage: <2GB for 5 cached models

**Expected Output:**
- ✅ MLManager loads all models successfully
- ✅ All 4 prediction types working
- ✅ Error handling for missing models

---

#### Day 15.2: ML API Endpoints Implementation

**File:** `frontend/server/api/routes/ml.py`

**API Specification:**

```python
# POST /api/ml/predict/classification
{
  "machine_id": "motor_siemens_1la7_001",
  "sensor_data": {
    "winding_temp_C": 45.2,
    "bearing_vibration_mm_s": 2.1,
    "current_phase_A_A": 12.5,
    ...
  }
}

# Response
{
  "machine_id": "motor_siemens_1la7_001",
  "health_state": 0,  # 0=Healthy, 1=Degrading, 2=Warning, 3=Critical
  "health_label": "Healthy",
  "confidence": 0.95,
  "predicted_at": "2024-12-13T15:11:19+05:30",
  "model_version": "v1.0.0",
  "inference_time_ms": 87
}
```

**Endpoints to Implement:**

1. **POST** `/api/ml/predict/classification` - Health state prediction
2. **POST** `/api/ml/predict/anomaly` - Anomaly detection
3. **POST** `/api/ml/predict/batch` - Batch predictions (all machines)
4. **GET** `/api/ml/models` - List available models
5. **GET** `/api/ml/models/{machine_id}` - Model metadata
6. **GET** `/api/ml/health` - Service health check

**Tasks:**
- [ ] Implement all 6 endpoints with Pydantic models
- [ ] Add input validation (sensor value ranges)
- [ ] Implement rate limiting (100 requests/minute)
- [ ] Add API response caching (30 seconds TTL)
- [ ] Comprehensive error handling with HTTP status codes
- [ ] OpenAPI documentation (Swagger UI)
- [ ] Integration tests for all endpoints

**Pydantic Models:**

**File:** `frontend/server/api/models/ml.py`

```python
class PredictionRequest(BaseModel):
    machine_id: str
    sensor_data: Dict[str, float]

class ClassificationResponse(BaseModel):
    machine_id: str
    health_state: int  # 0-3
    health_label: str
    confidence: float
    predicted_at: datetime
    model_version: str
    inference_time_ms: int
```

**Expected Output:**
- ✅ All 6 endpoints accessible via Swagger UI
- ✅ Predictions logged to database
- ✅ Response times < 500ms

---

### **Phase 2: Design System & Theme Setup (Day 16)**

#### Day 16.1: MUI Theme Configuration

**File:** `frontend/client/src/theme/professionalTheme.ts`

**Tasks:**
- [ ] Create custom MUI theme with design system colors
- [ ] Configure typography (Inter font family)
- [ ] Set up component overrides (Button, Card, etc.)
- [ ] Define breakpoints (mobile: 640px, tablet: 1024px, desktop: 1280px)
- [ ] Configure dark mode palette
- [ ] Create theme provider wrapper

**Code Structure:**
```typescript
export const professionalTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#667eea' },
    secondary: { main: '#764ba2' },
    success: { main: '#10b981' },
    warning: { main: '#fbbf24' },
    error: { main: '#ef4444' },
    background: {
      default: '#0f172a',
      paper: '#1f2937',
    },
  },
  typography: {
    fontFamily: 'Inter, sans-serif',
    h1: { fontSize: 32, fontWeight: 700 },
    h2: { fontSize: 24, fontWeight: 600 },
    h3: { fontSize: 18, fontWeight: 500 },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          backgroundImage: 'none',
        },
      },
    },
  },
});
```

**Expected Output:**
- ✅ Theme applied globally
- ✅ All components use design system colors
- ✅ Dark mode working

---

#### Day 16.2: Global Styles & Animations

**File:** `frontend/client/src/styles/global.css`

**Tasks:**
- [ ] Import Inter font from Google Fonts
- [ ] Define CSS custom properties for colors
- [ ] Create reusable animation keyframes (fadeIn, slideUp, pulse)
- [ ] Configure glassmorphism utilities
- [ ] Set up responsive grid system classes

**Animations:**
```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

@keyframes slideUp {
  from { transform: translateY(20px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

.glassmorphism {
  background: rgba(31, 41, 55, 0.6);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
}
```

**Expected Output:**
- ✅ Animations working smoothly
- ✅ Glassmorphism effects visible
- ✅ Responsive utilities functional

---

### **Phase 3: Core Frontend Components (Day 17-18)**

#### Day 17.1: Fleet Overview Cards Component

**File:** `frontend/client/src/modules/ml/components/FleetOverviewCards.tsx`

**Component Specifications:**

```typescript
interface FleetOverviewCardsProps {
  fleetStats: {
    healthy: number;
    degrading: number;
    warning: number;
    critical: number;
  };
  trends: {
    healthy: number;  // +2, -1, 0
    degrading: number;
    warning: number;
    critical: number;
  };
}
```

**Features:**
- [ ] Responsive 4-column grid (2x2 on tablet, 1x4 on mobile)
- [ ] Animated counters (count-up animation 1.5s)
- [ ] Trend indicators with arrows (↑ ↓ →)
- [ ] Click to filter machines by status
- [ ] Glassmorphism card design
- [ ] Pulse animation on critical status

**Styling:**
- Card height: 140px
- Border-radius: 12px
- Background: `rgba(31, 41, 55, 0.6)` with backdrop-blur
- Border: 1px solid matching status color
- Box-shadow: `0 4px 12px rgba(0,0,0,0.3)`

**Expected Output:**
- ✅ Cards render with correct styling
- ✅ Counters animate smoothly
- ✅ Responsive on all screen sizes

---

#### Day 17.2: Machine Status Card Component

**File:** `frontend/client/src/modules/ml/components/MachineStatusCard.tsx`

**Component Interface:**
```typescript
interface MachineStatusCardProps {
  machineId: string;
  machineName: string;
  healthState: 0 | 1 | 2 | 3;
  healthLabel: string;
  confidence: number;
  sensors: Array<{
    name: string;
    value: number;
    unit: string;
    icon: string;
  }>;
  lastUpdated: Date;
  onViewDetails: (machineId: string) => void;
  onExplain: (machineId: string) => void;
}
```

**Card Layout:**
```
┌─────────────────────────────┐
│ Motor Siemens 1LA7 001  95% │ ← Name + Confidence badge
│                              │
│ ● Healthy                    │ ← Status dot + label
│                              │
│ 🌡️ Temp: 45°C  📊 Vib: 2.1mm│ ← Key sensors
│ ⚡ Current: 12.5A            │
│                              │
│ Updated: 2 min ago           │
│                              │
│ [View Details]  [AI Explain] │ ← Action buttons
└─────────────────────────────┘
```

**Tasks:**
- [ ] Build card layout with MUI Card component
- [ ] Add status dot with pulse animation (critical state)
- [ ] Confidence badge (top-right corner)
- [ ] Sensor metrics display (max 3 sensors)
- [ ] Last updated timestamp (relative time)
- [ ] Two action buttons with icons
- [ ] Hover effect (scale 1.02, shadow increase)
- [ ] Click to expand for full details
- [ ] Skeleton loader for loading state

**Responsive Behavior:**
- Desktop: 380px width
- Tablet: 45% width
- Mobile: 100% width

**Expected Output:**
- ✅ Card renders correctly
- ✅ Animations smooth
- ✅ All interactions working

---

#### Day 17.3: Machine Grid Component

**File:** `frontend/client/src/modules/ml/components/MachineGrid.tsx`

**Features:**
- [ ] Responsive grid (3 cols desktop, 2 tablet, 1 mobile)
- [ ] Search bar (filter by machine name)
- [ ] Status filter dropdown (All, Healthy, Degrading, Warning, Critical)
- [ ] Sort options (Status, Name, Confidence)
- [ ] Pagination (12 machines per page)
- [ ] Lazy loading with IntersectionObserver
- [ ] Empty state illustration
- [ ] Loading state with skeleton cards

**Grid Configuration:**
```css
display: grid;
grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
gap: 20px;
padding: 24px;
```

**Expected Output:**
- ✅ Grid layout responsive
- ✅ Filters working correctly
- ✅ Pagination functional

---

#### Day 18.1: LLM Explanation Modal

**File:** `frontend/client/src/modules/ml/components/LLMExplanationModal.tsx`

**Features:**
- [ ] Full-screen modal with backdrop blur
- [ ] Fetch explanation from `/api/llm/explain`
- [ ] Markdown rendering for formatted text
- [ ] Sections: Summary, Risk Factors, Recommendations
- [ ] Copy to clipboard button
- [ ] Loading state with animated skeleton
- [ ] Error state with retry button
- [ ] Close animation (fade out)

**API Call:**
```typescript
const fetchExplanation = async (
  machineId: string,
  predictionData: PredictionResult
) => {
  const response = await fetch('/api/llm/explain', {
    method: 'POST',
    body: JSON.stringify({
      machine_id: machineId,
      health_state: predictionData.health_state,
      confidence: predictionData.confidence,
      sensor_data: predictionData.sensor_data
    })
  });
  return await response.json();
};
```

**Expected Output:**
- ✅ Modal opens smoothly
- ✅ Explanations load correctly
- ✅ Markdown rendering works

---

### **Phase 4: ML Dashboard Page Assembly (Day 18-19)**

#### Day 18.2: Main Dashboard Page

**File:** `frontend/client/src/pages/MLDashboardPage.tsx`

**Page Structure:**
```typescript
export const MLDashboardPage: React.FC = () => {
  const [fleetStats, setFleetStats] = useState(null);
  const [machines, setMachines] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedMachine, setSelectedMachine] = useState(null);
  
  // Fetch all predictions every 30 seconds
  useEffect(() => {
    const interval = setInterval(fetchAllPredictions, 30000);
    return () => clearInterval(interval);
  }, []);
  
  return (
    <Container maxWidth="xl">
      <PageHeader title="Machine Health Dashboard" />
      
      <FleetOverviewCards 
        fleetStats={fleetStats} 
        trends={trends} 
      />
      
      <MachineGrid 
        machines={machines}
        onViewDetails={handleViewDetails}
        onExplain={handleExplain}
      />
      
      {selectedMachine && (
        <LLMExplanationModal 
          machine={selectedMachine}
          onClose={() => setSelectedMachine(null)}
        />
      )}
    </Container>
  );
};
```

**Tasks:**
- [ ] Build page layout with proper spacing
- [ ] Implement data fetching with React Query
- [ ] Add auto-refresh (30-second interval)
- [ ] Implement search/filter/sort logic
- [ ] Connect all components
- [ ] Add page transitions (Framer Motion)
- [ ] Error boundary for fault tolerance
- [ ] Accessibility (ARIA labels, keyboard navigation)

**Expected Output:**
- ✅ Dashboard displays all 26 machines
- ✅ Real-time updates working
- ✅ All interactions functional

---

#### Day 19.1: Real-Time Updates & Polling

**Features:**
- [ ] WebSocket connection for live updates (future)
- [ ] HTTP polling fallback (current: 30s interval)
- [ ] Optimistic UI updates
- [ ] Background sync when tab inactive
- [ ] Connection status indicator
- [ ] Offline mode handling

**Expected Output:**
- ✅ Auto-refresh working
- ✅ No performance degradation
- ✅ Connection status visible

---

### **Phase 5: Integration & Testing (Day 19-21)**

#### Day 19.2: Backend-Frontend Integration

**Tasks:**
- [ ] Connect all API endpoints
- [ ] Test batch predictions for 26 machines
- [ ] Verify WebSocket real-time updates
- [ ] Load testing (100 concurrent users)
- [ ] API response time optimization
- [ ] Error handling for network failures

**Expected Output:**
- ✅ All APIs working end-to-end
- ✅ Performance meets targets
- ✅ Error handling robust

---

#### Day 20: Quality Assurance

**Testing Checklist:**
- [ ] Unit tests (>80% coverage)
- [ ] Integration tests (API + Frontend)
- [ ] E2E tests with Playwright (happy path)
- [ ] Accessibility testing (WCAG 2.1 AA)
- [ ] Cross-browser testing (Chrome, Firefox, Safari)
- [ ] Mobile responsive testing
- [ ] Performance testing (Lighthouse score >90)

**Expected Output:**
- ✅ All tests passing
- ✅ No critical bugs
- ✅ Performance optimized

---

#### Day 21: Documentation & Deployment Prep

**Deliverables:**
- [ ] API documentation (Swagger/OpenAPI)
- [ ] Component Storybook
- [ ] User guide (screenshots + instructions)
- [ ] Developer README
- [ ] Deployment checklist
- [ ] Rollback plan

**Expected Output:**
- ✅ Documentation complete
- ✅ Deployment ready
- ✅ Team trained

---

## 📊 Success Metrics & KPIs

### Performance Targets
- **Page Load Time:** <2 seconds
- **API Response Time:** <500ms (95th percentile)
- **Prediction Inference:** <100ms per machine
- **UI Frame Rate:** 60 FPS (no jank)

### Quality Targets
- **Test Coverage:** >80%
- **Lighthouse Score:** >90
- **Accessibility:** WCAG 2.1 AA compliant
- **Browser Support:** Last 2 versions of major browsers

### Business Metrics
- **User Adoption:** 100% of maintenance team using dashboard within 1 week
- **Prediction Accuracy:** >90% match with actual failures
- **Time to Insight:** <10 seconds from data to prediction

---

## 🎯 Deliverables Summary

### Backend (Python/FastAPI)
1. ✅ `ml_manager.py` - ML service layer
2. ✅ `routes/ml.py` - 6 REST API endpoints
3. ✅ Unit tests + Integration tests

### Frontend (React/TypeScript)
4. ✅ `professionalTheme.ts` - MUI theme
5. ✅ `FleetOverviewCards.tsx` - Status summary
6. ✅ `MachineStatusCard.tsx` - Individual machine card
7. ✅ `MachineGrid.tsx` - Grid layout with filters
8. ✅ `LLMExplanationModal.tsx` - AI explanations
9. ✅ `MLDashboardPage.tsx` - Main dashboard page

### Documentation
10. ✅ API documentation (OpenAPI spec)
11. ✅ Component documentation (Storybook)
12. ✅ User guide
13. ✅ Deployment guide

---

## PHASE 3.7.4: LLM INTEGRATION
**Duration:** Week 4 (Days 22-28)  
**Goal:** Build LLM explanation module with chat interface

### Phase 3.7.4.1: LLM Manager Service (Days 22-23)

**Tasks:**
- [❌] Create LLMManager class in `frontend/server/api/services/llm_manager.py`
- [❌] Import `IntegratedPredictionSystem` from `LLM/api/ml_integration.py`
- [❌] Initialize with GPU acceleration
- [❌] Implement explanation methods
- [❌] Add streaming support

**LLMManager Methods:**

| Method | Purpose | Returns |
|--------|---------|---------|
| `explain_prediction(pred_type, pred_data)` | Generate explanation | Explanation text |
| `chat(message, context)` | Conversational Q&A | Response text |
| `analyze_batch(predictions)` | Batch analysis | Array of explanations |
| `generate_report(machine_id, period)` | PDF report generation | File path |

**Integration:**
- Uses existing `IntegratedPredictionSystem` class
- GPU acceleration via CUDA DLL injection
- Token streaming for chat interface

**File:** `frontend/server/api/services/llm_manager.py` (150+ lines)

**Expected Output:**
- ✅ LLMManager initializes successfully
- ✅ Explanations generated correctly
- ✅ GPU acceleration working (~26 tok/s)

---

### Phase 3.7.4.2: LLM API Routes (Days 24-25)

**Tasks:**
- [❌] Create FastAPI router in `frontend/server/api/routes/llm.py`
- [❌] Implement 4 API endpoints
- [❌] Add streaming endpoint for chat
- [❌] Integrate with LLMManager service

**API Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/llm/explain` | POST | Get prediction explanation |
| `/api/llm/chat` | POST | Send chat message |
| `/api/llm/analyze` | POST | Batch analysis |
| `/api/llm/report` | POST | Generate PDF report |

**Pydantic Models:**

**File:** `frontend/server/api/models/llm.py`

**ExplainRequest:**
- prediction_type: str
- prediction_data: dict
- machine_id: str

**ExplainResponse:**
- success: bool
- explanation: str
- recommendations: List[str]
- risk_factors: List[str]

**ChatRequest:**
- message: str
- context: dict (optional)

**ChatResponse:**
- success: bool
- response: str
- timestamp: datetime

**File:** `frontend/server/api/routes/llm.py` (150+ lines)

**Expected Output:**
- ✅ Explanation endpoint working
- ✅ Chat endpoint responding
- ✅ Responses under 3 seconds

---

### Phase 3.7.4.3: LLM WebSocket for Chat (Day 26)

**Tasks:**
- [❌] Create WebSocket endpoint `/ws/llm/chat`
- [❌] Implement token streaming
- [❌] Handle context management
- [❌] Add conversation history

**WebSocket Flow:**
1. Client sends message
2. Server streams tokens as generated
3. Client displays in real-time
4. Conversation context preserved

**Message Format:**
```
{
  type: "token",
  content: "High bearing ",
  done: false
}
```

**Expected Output:**
- ✅ Chat streaming functional
- ✅ Tokens display in real-time
- ✅ Context preserved across messages

---

### Phase 3.7.4.4: Frontend LLM Components (Days 27-28)

**Tasks:**
- [❌] Create ExplanationPanel component
- [❌] Create ChatMessage component
- [❌] Create ChatInterface page
- [❌] Create AIExplainer page
- [❌] Integrate with LLM API

**Components:**

**ExplanationPanel.tsx:**
- Displays explanation text
- Shows recommendations list
- Risk factors highlighted
- Export PDF button

**ChatMessage.tsx:**
- User message bubble
- AI response bubble
- Timestamp display
- Markdown rendering

**ChatInterface.tsx:**
- Message input field
- Message history
- WebSocket connection
- Auto-scroll to bottom

**AIExplainer.tsx:**
- Left: Prediction details
- Right: ExplanationPanel
- Bottom: Chat button
- Export options

**Expected Output:**
- ✅ Explanations display correctly
- ✅ Chat interface functional
- ✅ Streaming working smoothly

---

## PHASE 3.7.5: DASHBOARD & UI POLISH
**Duration:** Week 5 (Days 29-35)  
**Goal:** Build main dashboard and finalize UI

### Phase 3.7.5.1: Main Dashboard Page (Days 29-30)

**Tasks:**
- [❌] Create Dashboard page (landing page)
- [❌] Add fleet overview cards
- [❌] Add prediction grid (26 machines)
- [❌] Add recent alerts list
- [❌] Add quick actions

**Dashboard Layout:**
- Top: 4 metric cards (Total, Healthy, At-Risk, Failed)
- Middle: Prediction grid (6x5 grid for 26 machines)
- Right: Recent alerts sidebar
- Bottom: Quick actions bar

**Data Sources:**
- `/api/ml/predict/classification` (batch)
- `/api/gan/machines` (machine list)
- `/api/llm/explain` (for alerts)

**Auto-refresh:** Every 30 seconds

**Expected Output:**
- ✅ Dashboard displays all 26 machines
- ✅ Status colors accurate
- ✅ Auto-refresh working

---

### Phase 3.7.5.2: Navigation & Layout (Days 31-32)

**Tasks:**
- [❌] Create Header component (top bar)
- [❌] Create Sidebar component (navigation)
- [❌] Setup React Router routes
- [❌] Add breadcrumb navigation
- [❌] Add user profile dropdown

**Navigation Structure:**
```
Dashboard
├── GAN Module
│   ├── New Machine Wizard
│   ├── Data Explorer
│   └── Batch Operations
├── ML Module
│   ├── Prediction Dashboard
│   ├── Model Performance
│   └── Feature Importance
└── LLM Module
    ├── AI Explainer
    ├── Chat Interface
    └── Report Generator
```

**Sidebar Features:**
- Collapsible menu
- Active route highlighting
- Icon + text labels
- Dark/Light mode toggle

**Expected Output:**
- ✅ Navigation working across all pages
- ✅ Breadcrumbs update correctly
- ✅ Sidebar state persists

---

### Phase 3.7.5.3: Material-UI Theming (Day 33)

**Tasks:**
- [❌] Create custom MUI theme
- [❌] Define color palette (industrial)
- [❌] Add dark mode support
- [❌] Apply theme globally

**Color Palette:**
- Primary: Blue (#1976d2)
- Secondary: Amber (#ffa726)
- Success: Green (#4caf50)
- Warning: Orange (#ff9800)
- Error: Red (#f44336)
- Background (Dark): #121212

**Theme File:** `client/src/theme.ts`

**Expected Output:**
- ✅ Consistent styling across app
- ✅ Dark mode toggle working
- ✅ Colors match design system

---

### Phase 3.7.5.4: Error Handling & Loading States (Day 34)

**Tasks:**
- [❌] Create ErrorBoundary component
- [❌] Create LoadingSpinner component
- [❌] Add error handling to API calls
- [❌] Add toast notifications (react-toastify)
- [❌] Add retry logic for failed requests

**Error Handling:**
- Network errors: Display toast + retry button
- API errors: Display error message
- 401 errors: Redirect to login
- 500 errors: Show error boundary

**Loading States:**
- Skeleton loaders for data fetching
- Progress bars for file uploads
- Spinners for async operations

**Expected Output:**
- ✅ Errors handled gracefully
- ✅ Loading states display correctly
- ✅ User experience smooth

---

### Phase 3.7.5.5: Responsive Design (Day 35)

**Tasks:**
- [❌] Test on mobile (375px, 768px, 1024px)
- [❌] Add responsive breakpoints
- [❌] Fix layout issues
- [❌] Test on different browsers

**Responsive Breakpoints:**
- Mobile: < 768px (single column)
- Tablet: 768px - 1024px (2 columns)
- Desktop: > 1024px (full layout)

**Expected Output:**
- ✅ App usable on mobile devices
- ✅ Layout adapts correctly
- ✅ No horizontal scrolling

---

## PHASE 3.7.6: TESTING & DEPLOYMENT
**Duration:** Week 6 (Days 36-42)  
**Goal:** Comprehensive testing and production deployment

### Phase 3.7.6.1: Backend Unit Tests (Days 36-37)

**Tasks:**
- [❌] Write tests for GANManager (5 test cases)
- [❌] Write tests for MLManager (4 test cases)
- [❌] Write tests for LLMManager (3 test cases)
- [❌] Write tests for API routes (20 test cases)
- [❌] Achieve 80%+ code coverage

**Testing Framework:** pytest + httpx

**Test Files:**
- `frontend/server/tests/test_gan_manager.py`
- `frontend/server/tests/test_ml_manager.py`
- `frontend/server/tests/test_llm_manager.py`
- `frontend/server/tests/test_routes.py`

**Run Tests:**
```powershell
cd frontend/server
pytest --cov=api tests/
```

**Expected Output:**
- ✅ All tests passing
- ✅ Coverage > 80%

---

### Phase 3.7.6.2: Frontend Component Tests (Day 38)

**Tasks:**
- [❌] Write tests for MachineForm (5 test cases)
- [❌] Write tests for ProgressTracker (3 test cases)
- [❌] Write tests for PredictionCard (4 test cases)
- [❌] Write tests for ChatInterface (3 test cases)

**Testing Framework:** React Testing Library + Vitest

**Test Files:**
- `frontend/client/src/components/gan/__tests__/MachineForm.test.tsx`
- `frontend/client/src/components/ml/__tests__/PredictionCard.test.tsx`

**Run Tests:**
```powershell
cd frontend/client
npm run test
```

**Expected Output:**
- ✅ All component tests passing
- ✅ No console errors

---

### Phase 3.7.6.3: E2E Tests (Day 39)

**Tasks:**
- [❌] Write E2E test for wizard workflow
- [❌] Write E2E test for prediction flow
- [❌] Write E2E test for chat interface
- [❌] Run tests in headless mode

**Testing Framework:** Playwright

**Test Scenarios:**
1. Complete wizard (5 steps)
2. Make prediction → get explanation
3. Chat with AI assistant

**Run Tests:**
```powershell
cd frontend/client
npx playwright test
```

**Expected Output:**
- ✅ All E2E tests passing
- ✅ No UI regressions

---

### Phase 3.7.6.4: Docker Configuration (Days 40-41)

**Tasks:**
- [❌] Create Dockerfile for frontend
- [❌] Create Dockerfile for backend
- [❌] Create docker-compose.yml
- [❌] Configure Nginx reverse proxy
- [❌] Test multi-container setup

**Services:**
- frontend: Nginx + React build
- backend: FastAPI + Uvicorn
- celery-worker: Background tasks
- redis: Task broker
- postgres: Database

**File:** `frontend/docker-compose.yml`

**Build & Run:**
```powershell
cd frontend
docker-compose up --build
```

**Expected Output:**
- ✅ All containers start successfully
- ✅ App accessible at http://localhost
- ✅ API accessible at http://localhost/api

---

### Phase 3.7.6.5: Production Deployment (Day 42)

**Tasks:**
- [❌] Create .env.production file
- [❌] Setup HTTPS (SSL certificates)
- [❌] Configure CORS properly
- [❌] Add rate limiting
- [❌] Setup logging (file rotation)
- [❌] Create backup scripts
- [❌] Deploy to production server

**Environment Variables:**
```
DATABASE_URL=postgresql://user:pass@postgres:5432/pdm
REDIS_URL=redis://redis:6379/0
SECRET_KEY=<generated-key>
ALLOWED_ORIGINS=https://yourdomain.com
```

**Monitoring:**
- FastAPI `/metrics` endpoint
- Celery Flower: http://localhost:5555
- Frontend error logging (Sentry)

**Expected Output:**
- ✅ App deployed successfully
- ✅ HTTPS working
- ✅ All features functional in production

---

## Success Criteria

**Technical Metrics:**
- [ ] API response time < 200ms
- [ ] WebSocket latency < 50ms
- [ ] System uptime > 99.5%
- [ ] Test coverage > 80%

**User Experience:**
- [ ] Wizard completion rate > 90%
- [ ] Prediction accuracy displayed correctly
- [ ] Explanations are human-readable
- [ ] Chat responses < 3 seconds

**Business Metrics:**
- [ ] Time to add new machine < 30 minutes
- [ ] Prediction coverage: 100% of fleet
- [ ] Report generation < 30 seconds

---

## Documentation Deliverables

**By Phase End:**
1. [ ] API Documentation (Swagger UI auto-generated)
2. [ ] User Guide (PDF with screenshots)
3. [ ] Developer Setup Guide (README)
4. [ ] Deployment Guide (Docker commands)
5. [ ] Video Demo (5-minute walkthrough)

---

## Next Steps

**Immediate Actions:**
1. Review and approve this plan
2. Setup development environment
3. Create GitHub project board
4. Begin Phase 3.7.1.1 (Project Initialization)

**Decision Points:**
- [ ] Approve tech stack (React + FastAPI)
- [ ] Confirm authentication (JWT vs. none)
- [ ] Confirm deployment target (local vs. cloud)

---

**END OF PHASE 3.7 PLAN**

This comprehensive plan integrates all three subsystems (GAN, ML, LLM) with detailed step-by-step tasks.

---

## 📑 Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Subsystem Integration Points](#2-subsystem-integration-points)
3. [Backend API Design](#3-backend-api-design)
4. [Frontend Pages & Components](#4-frontend-pages--components)
5. [Phase Implementation Roadmap](#5-phase-implementation-roadmap)
6. [Database Schema](#6-database-schema)
7. [Security & Authentication](#7-security--authentication)
8. [Testing & Validation](#8-testing--validation)
9. [Deployment Strategy](#9-deployment-strategy)
10. [Success Metrics](#10-success-metrics)

---

## 1. 🏛️ System Architecture

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      FRONTEND (React + TypeScript)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ GAN Module   │  │  ML Module   │  │  LLM Module  │         │
│  │ - New Machine│  │ - Predictions│  │ - Explainer  │         │
│  │ - Training   │  │ - Model Mgmt │  │ - Chat       │         │
│  │ - Validation │  │ - Monitoring │  │ - Reports    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ↕ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND API (FastAPI)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ GAN Routes   │  │  ML Routes   │  │  LLM Routes  │         │
│  │ /api/gan/*   │  │ /api/ml/*    │  │ /api/llm/*   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ GAN Manager  │  │ ML Manager   │  │ LLM Manager  │         │
│  │ (Scripts)    │  │ (Classes)    │  │ (API)        │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                    EXISTING SYSTEMS                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ GAN/         │  │ ml_models/   │  │ LLM/         │         │
│  │ - Scripts    │  │ - Models     │  │ - API        │         │
│  │ - Models     │  │ - Scalers    │  │ - RAG        │         │
│  │ - Data       │  │ - Config     │  │ - Llama      │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ PostgreSQL   │  │ Redis        │  │ File Storage │         │
│  │ (Metadata)   │  │ (Tasks/Cache)│  │ (Parquet)    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18 + TypeScript | Component-based UI |
| | Material-UI v5 | UI components library |
| | Zustand | State management |
| | React Query | Data fetching/caching |
| | Chart.js + D3.js | Visualizations |
| | React Router v6 | Navigation |
| **Backend** | FastAPI | REST API framework |
| | Celery | Background task queue |
| | Redis | Task broker + caching |
| | PostgreSQL | Relational database |
| | SQLAlchemy | ORM |
| | Pydantic | Data validation |
| **Deployment** | Docker + Compose | Containerization |
| | Nginx | Reverse proxy |
| | Gunicorn/Uvicorn | WSGI/ASGI server |

---

## 2. 🔗 Subsystem Integration Points

### 2.1 GAN Subsystem

**Directory:** `GAN/`

**Integration Approach:** Subprocess execution (scripts are standalone)

**Key Files:**
- `GAN/scripts/*.py` - All 18 GAN scripts
- `GAN/metadata/*.json` - Machine configurations
- `GAN/models/*.pkl` - TVAE models
- `GAN/data/synthetic/*.parquet` - Generated datasets

**API Endpoints:**
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/gan/machines` | POST | Create machine profile |
| `/api/gan/machines/{id}/seed` | POST | Generate seed data |
| `/api/gan/machines/{id}/train` | POST | Train TVAE model |
| `/api/gan/machines/{id}/generate` | POST | Generate synthetic data |
| `/api/gan/machines/{id}/validate` | GET | Validate data quality |
| `/api/gan/machines` | GET | List all machines |
| `/ws/gan/training/{task_id}` | WebSocket | Training progress stream |

**Frontend Pages:**
- **New Machine Wizard:** 5-step workflow
- **Data Explorer:** Visualize parquet files
- **Batch Operations:** Validate all 26 machines

---

### 2.2 ML Models Subsystem

**Directory:** `ml_models/`

**Integration Approach:** Direct class import (already modular)

**Key Files:**
- `ml_models/scripts/inference/*.py` - 4 model inference classes
  - `classification_predictor.py`
  - `regression_predictor.py`
  - `anomaly_detector.py`
  - `timeseries_forecaster.py`
- `ml_models/models/*.pkl` - Trained models
- `ml_models/models/*.joblib` - Scalers
- `ml_models/config/model_paths.py` - Model registry

**API Endpoints:**
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/ml/predict/classification` | POST | Binary health classification |
| `/api/ml/predict/rul` | POST | RUL regression |
| `/api/ml/predict/anomaly` | POST | Anomaly detection |
| `/api/ml/predict/timeseries` | POST | Time series forecasting |
| `/api/ml/models` | GET | List available models |
| `/api/ml/models/{type}/metrics` | GET | Model performance metrics |
| `/api/ml/models/{type}/train` | POST | Retrain model (background) |
| `/ws/ml/training/{task_id}` | WebSocket | Training progress |

**Frontend Pages:**
- **Prediction Dashboard:** Real-time predictions for all 26 machines
- **Model Performance:** Accuracy, F1, MAE, RMSE charts
- **Model Training:** Retrain models with new data
- **Feature Importance:** SHAP values visualization

---

### 2.3 LLM Subsystem

**Directory:** `LLM/`

**Integration Approach:** API class import (in-memory instances)

**Key Files:**
- `LLM/api/ml_integration.py` - `IntegratedPredictionSystem`
- `LLM/api/explainer.py` - `MLExplainer`
- `LLM/scripts/inference/llama_engine.py` - Core LLM engine
- `LLM/scripts/rag/retriever.py` - RAG retriever
- `LLM/data/manuals/*.txt` - Equipment manuals

**API Endpoints:**
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/llm/explain` | POST | Get ML prediction explanation |
| `/api/llm/chat` | POST | Conversational Q&A |
| `/api/llm/analyze` | POST | Batch analysis with explanations |
| `/api/llm/report` | POST | Generate PDF report |
| `/ws/llm/stream` | WebSocket | Streaming chat responses |

**Frontend Pages:**
- **AI Explainer:** Show predictions + natural language explanations
- **Chat Interface:** Conversational assistant for maintenance queries
- **Report Generator:** Auto-generate maintenance reports

---

## 3. 🔌 Backend API Design

### 3.1 Directory Structure

```
frontend/
└── server/                          # FastAPI backend
    ├── main.py                      # FastAPI app entry point
    ├── config.py                    # Environment settings
    ├── database.py                  # PostgreSQL connection
    ├── dependencies.py              # Shared dependencies
    │
    ├── api/
    │   ├── routes/
    │   │   ├── gan.py               # GAN endpoints
    │   │   ├── ml.py                # ML endpoints
    │   │   ├── llm.py               # LLM endpoints
    │   │   ├── dashboard.py         # Dashboard data aggregation
    │   │   └── websocket.py         # WebSocket handlers
    │   │
    │   ├── services/
    │   │   ├── gan_manager.py       # GAN script executor
    │   │   ├── ml_manager.py        # ML model wrapper
    │   │   ├── llm_manager.py       # LLM API wrapper
    │   │   └── data_loader.py       # Parquet/metadata reader
    │   │
    │   └── models/                  # Pydantic models
    │       ├── gan.py               # GAN request/response schemas
    │       ├── ml.py                # ML schemas
    │       ├── llm.py               # LLM schemas
    │       └── common.py            # Shared types
    │
    ├── tasks/                       # Celery workers
    │   ├── gan_tasks.py             # TVAE training, data generation
    │   ├── ml_tasks.py              # Model retraining
    │   └── report_tasks.py          # PDF generation
    │
    ├── db/
    │   ├── models.py                # SQLAlchemy models
    │   └── crud.py                  # Database operations
    │
    └── utils/
        ├── validators.py            # Input validation
        ├── formatters.py            # Data formatting
        └── file_utils.py            # File I/O helpers
```

### 3.2 Core Service Classes

**GANManager (server/api/services/gan_manager.py):**
- Executes GAN scripts via subprocess
- Parses stdout for progress tracking
- Returns standardized ScriptResult objects
- Methods: create_profile, generate_seed, train_tvae, generate_data, validate

**MLManager (server/api/services/ml_manager.py):**
- Imports inference classes from `ml_models/`
- Loads models + scalers on initialization
- Provides unified predict() interface
- Methods: predict_classification, predict_rul, detect_anomaly, forecast_timeseries

**LLMManager (server/api/services/llm_manager.py):**
- Wraps `IntegratedPredictionSystem` from `LLM/api/ml_integration.py`
- Manages GPU-accelerated inference
- Methods: explain_prediction, chat, generate_report

### 3.3 Request/Response Schemas

**Example: ML Prediction Request**

```
POST /api/ml/predict/classification

Request Body:
{
  "machine_id": "motor_siemens_1la7_001",
  "sensor_data": {
    "bearing_temp_C": 75.2,
    "vibration_rms_mm_s": 3.4,
    "current_A": 12.1,
    ...
  }
}

Response:
{
  "success": true,
  "prediction": {
    "health_status": "unhealthy",
    "confidence": 0.92,
    "timestamp": "2025-11-27T10:30:00Z"
  },
  "explanation": {
    "summary": "High bearing temperature and vibration indicate bearing wear...",
    "risk_factors": ["bearing_temp_C", "vibration_rms_mm_s"],
    "recommendations": ["Schedule bearing inspection within 48 hours"]
  }
}
```

---

## 4. 🎨 Frontend Pages & Components

### 4.1 Directory Structure

```
frontend/
└── client/                          # React frontend
    ├── src/
    │   ├── App.tsx                  # Main app component
    │   ├── index.tsx                # Entry point
    │   │
    │   ├── pages/
    │   │   ├── Dashboard.tsx        # **Main landing page**
    │   │   │
    │   │   ├── gan/
    │   │   │   ├── NewMachineWizard.tsx
    │   │   │   ├── DataExplorer.tsx
    │   │   │   └── BatchOperations.tsx
    │   │   │
    │   │   ├── ml/
    │   │   │   ├── PredictionDashboard.tsx
    │   │   │   ├── ModelPerformance.tsx
    │   │   │   ├── ModelTraining.tsx
    │   │   │   └── FeatureImportance.tsx
    │   │   │
    │   │   └── llm/
    │   │       ├── AIExplainer.tsx
    │   │       ├── ChatInterface.tsx
    │   │       └── ReportGenerator.tsx
    │   │
    │   ├── components/
    │   │   ├── common/
    │   │   │   ├── Header.tsx
    │   │   │   ├── Sidebar.tsx
    │   │   │   ├── LoadingSpinner.tsx
    │   │   │   └── ErrorBoundary.tsx
    │   │   │
    │   │   ├── gan/
    │   │   │   ├── MachineForm.tsx
    │   │   │   ├── ProgressTracker.tsx
    │   │   │   ├── ValidationDisplay.tsx
    │   │   │   └── DataPlotter.tsx
    │   │   │
    │   │   ├── ml/
    │   │   │   ├── PredictionCard.tsx
    │   │   │   ├── MetricsChart.tsx
    │   │   │   ├── ConfusionMatrix.tsx
    │   │   │   └── SHAPPlot.tsx
    │   │   │
    │   │   └── llm/
    │   │       ├── ExplanationPanel.tsx
    │   │       ├── ChatMessage.tsx
    │   │       └── ReportPreview.tsx
    │   │
    │   ├── services/
    │   │   ├── ganAPI.ts            # GAN API client
    │   │   ├── mlAPI.ts             # ML API client
    │   │   ├── llmAPI.ts            # LLM API client
    │   │   └── websocket.ts         # WebSocket manager
    │   │
    │   ├── store/
    │   │   ├── ganStore.ts          # GAN state
    │   │   ├── mlStore.ts           # ML state
    │   │   ├── llmStore.ts          # LLM state
    │   │   └── uiStore.ts           # UI state
    │   │
    │   ├── types/
    │   │   ├── gan.ts               # GAN TypeScript types
    │   │   ├── ml.ts                # ML types
    │   │   └── llm.ts               # LLM types
    │   │
    │   └── utils/
    │       ├── validators.ts
    │       ├── formatters.ts
    │       └── constants.ts
    │
    ├── public/
    │   └── index.html
    │
    ├── package.json
    ├── vite.config.ts
    └── tsconfig.json
```

### 4.2 Key Frontend Pages

**1. Dashboard (Main Landing Page)**

**Purpose:** Unified view of all 26 machines

**Components:**
- Fleet overview cards (total machines, healthy, at-risk, failed)
- Real-time prediction grid (26 machines with health status)
- Recent alerts list
- Quick actions (New Machine, Run Prediction, Generate Report)

**Data Sources:**
- `/api/ml/predict/classification` (batch for all machines)
- `/api/gan/machines` (machine list)
- `/api/llm/explain` (for alerts)

---

**2. GAN Module - New Machine Wizard**

**Purpose:** 5-step workflow to add new machine

**Steps:**
1. Profile Creation (MachineForm component)
2. Seed Generation (progress indicator)
3. TVAE Training (ProgressTracker with WebSocket)
4. Data Generation (file statistics)
5. Validation (ValidationDisplay)

**State:** Zustand wizard store (current step, machine_id, results)

---

**3. ML Module - Prediction Dashboard**

**Purpose:** Real-time predictions for fleet

**Components:**
- Machine selector dropdown
- Prediction type tabs (Classification, RUL, Anomaly, Timeseries)
- PredictionCard showing result + confidence
- Historical predictions chart
- Feature importance plot (SHAP)

**Data Flow:**
1. User selects machine + prediction type
2. Fetch latest parquet data via `/api/ml/predict/*`
3. Display prediction + call `/api/llm/explain` for natural language
4. Show ExplanationPanel with recommendations

---

**4. LLM Module - AI Explainer**

**Purpose:** Show ML predictions with human-readable explanations

**Layout:**
- Left panel: Machine + prediction details
- Right panel: LLM explanation (streaming via WebSocket)
- Bottom: Recommendations list
- Action buttons: Export PDF, Chat with AI

**Integration:**
- Receives prediction from ML module
- Calls `/api/llm/explain` with prediction context
- Streams response via `/ws/llm/stream`

---

## 5. 📅 Phase Implementation Roadmap

### 5.1 Phase Breakdown

**PHASE 3.7.1: Foundation Setup (Week 1)**
- [ ] Initialize React + TypeScript project with Vite
- [ ] Setup FastAPI project structure
- [ ] Configure PostgreSQL + Redis with Docker Compose
- [ ] Create database schema (machines, predictions, training_jobs)
- [ ] Setup Celery workers
- [ ] Build authentication system (JWT)

**PHASE 3.7.2: GAN Integration (Week 2)**
- [ ] Backend: Implement GANManager service
- [ ] Backend: Create GAN routes (`/api/gan/*`)
- [ ] Backend: Setup Celery tasks for training
- [ ] Frontend: Build MachineForm component
- [ ] Frontend: Build ProgressTracker with WebSocket
- [ ] Frontend: Build New Machine Wizard page
- [ ] Frontend: Build Data Explorer page
- [ ] Test: Full GAN workflow (create → train → validate)

**PHASE 3.7.3: ML Integration (Week 3)**
- [ ] Backend: Implement MLManager service
- [ ] Backend: Create ML routes (`/api/ml/*`)
- [ ] Backend: Add model retraining Celery tasks
- [ ] Frontend: Build PredictionCard component
- [ ] Frontend: Build MetricsChart component
- [ ] Frontend: Build Prediction Dashboard page
- [ ] Frontend: Build Model Performance page
- [ ] Test: All 4 prediction types with 26 machines

**PHASE 3.7.4: LLM Integration (Week 4)**
- [ ] Backend: Implement LLMManager service
- [ ] Backend: Create LLM routes (`/api/llm/*`)
- [ ] Backend: Setup streaming WebSocket for chat
- [ ] Backend: Add PDF report generation task
- [ ] Frontend: Build ExplanationPanel component
- [ ] Frontend: Build ChatMessage component
- [ ] Frontend: Build AI Explainer page
- [ ] Frontend: Build Chat Interface page
- [ ] Test: Prediction + explanation pipeline

**PHASE 3.7.5: Dashboard & UI Polish (Week 5)**
- [ ] Frontend: Build main Dashboard page
- [ ] Frontend: Integrate all modules into unified nav
- [ ] Frontend: Add Material-UI theme (dark/light mode)
- [ ] Frontend: Build Header + Sidebar components
- [ ] Frontend: Add error handling + loading states
- [ ] Backend: Add comprehensive logging
- [ ] Backend: Implement rate limiting
- [ ] Test: End-to-end user workflows

**PHASE 3.7.6: Testing & Deployment (Week 6)**
- [ ] Write unit tests (backend services)
- [ ] Write component tests (React Testing Library)
- [ ] Write E2E tests (Playwright)
- [ ] Create Docker images (frontend, backend, workers)
- [ ] Write docker-compose.yml
- [ ] Configure Nginx reverse proxy
- [ ] Setup environment variables
- [ ] Deploy to staging environment
- [ ] Performance testing (load testing with Locust)
- [ ] Production deployment

---

## 6. 🗄️ Database Schema

### 6.1 PostgreSQL Tables

**machines** (GAN-generated machines)
```
id: UUID (PK)
machine_id: VARCHAR (unique, e.g., "motor_siemens_1la7_001")
machine_type: VARCHAR
manufacturer: VARCHAR
model: VARCHAR
metadata_path: VARCHAR
created_at: TIMESTAMP
updated_at: TIMESTAMP
```

**gan_training_jobs** (TVAE training history)
```
id: UUID (PK)
machine_id: VARCHAR (FK)
epochs: INT
status: VARCHAR (pending, running, completed, failed)
loss_history: JSON
started_at: TIMESTAMP
completed_at: TIMESTAMP
```

**predictions** (ML prediction logs)
```
id: UUID (PK)
machine_id: VARCHAR (FK)
prediction_type: VARCHAR (classification, rul, anomaly, timeseries)
input_data: JSON
prediction_result: JSON
confidence: FLOAT
timestamp: TIMESTAMP
```

**explanations** (LLM explanations cache)
```
id: UUID (PK)
prediction_id: UUID (FK)
explanation_text: TEXT
recommendations: JSON
created_at: TIMESTAMP
```

**model_versions** (ML model tracking)
```
id: UUID (PK)
model_type: VARCHAR
version: VARCHAR
file_path: VARCHAR
metrics: JSON (accuracy, f1, etc.)
trained_at: TIMESTAMP
is_active: BOOLEAN
```

---

## 7. 🔐 Security & Authentication

### 7.1 Authentication Strategy

**Option 1: Simple (PoC/Demo)**
- No authentication (local deployment only)
- Use for development/testing

**Option 2: JWT-based (Production)**
- User registration/login
- JWT tokens for API access
- Role-based access control (admin, operator, viewer)

**Libraries:**
- `python-jose` (JWT handling)
- `passlib` (password hashing)
- `python-multipart` (form data)

**Endpoints:**
- `POST /api/auth/register`
- `POST /api/auth/login` → returns JWT token
- `POST /api/auth/refresh` → refresh token

**Frontend:**
- Store JWT in localStorage
- Include in Authorization header: `Bearer <token>`
- Redirect to login on 401 responses

### 7.2 API Security

- **CORS:** Configure allowed origins
- **Rate Limiting:** 100 requests/minute per IP
- **Input Validation:** Pydantic models for all requests
- **SQL Injection:** Use SQLAlchemy ORM (parameterized queries)
- **File Upload:** Validate file types, size limits

---

## 8. 🧪 Testing & Validation

### 8.1 Backend Testing

**Unit Tests:**
- Test GANManager methods with mocked subprocess
- Test MLManager predictions with sample data
- Test LLMManager with mocked LLM responses
- Test database CRUD operations

**Integration Tests:**
- Test full GAN workflow (profile → train → validate)
- Test ML prediction pipeline
- Test LLM explanation generation
- Test WebSocket connections

**Tools:**
- `pytest` for test framework
- `pytest-asyncio` for async tests
- `httpx` for API testing

### 8.2 Frontend Testing

**Component Tests:**
- Test form validation (MachineForm)
- Test chart rendering (MetricsChart)
- Test WebSocket updates (ProgressTracker)

**E2E Tests:**
- Test complete wizard flow
- Test prediction workflow
- Test chat interface

**Tools:**
- `React Testing Library`
- `Playwright` for E2E

### 8.3 Performance Testing

**Load Testing:**
- Simulate 100 concurrent users
- Test prediction API throughput
- Test WebSocket scalability

**Tool:** Locust (Python-based load testing)

---

## 9. 🚀 Deployment Strategy

### 9.1 Docker Compose Architecture

**Services:**
```yaml
services:
  frontend:
    # Nginx + React build
    ports: 80:80
  
  backend:
    # FastAPI + Uvicorn
    ports: 8000:8000
    
  celery-worker:
    # Background tasks
    
  redis:
    # Task broker + cache
    
  postgres:
    # Database
    
  nginx:
    # Reverse proxy
```

### 9.2 Environment Variables

**Backend (.env):**
```
DATABASE_URL=postgresql://user:pass@postgres:5432/pdm
REDIS_URL=redis://redis:6379/0
GAN_ROOT=/app/GAN
ML_MODELS_ROOT=/app/ml_models
LLM_ROOT=/app/LLM
SECRET_KEY=<random-key>
```

**Frontend (.env):**
```
VITE_API_URL=http://localhost:8000
```

### 9.3 Production Considerations

**Scaling:**
- Multiple Celery workers for parallel training
- Redis cluster for high availability
- PostgreSQL read replicas for analytics

**Monitoring:**
- FastAPI `/metrics` endpoint (Prometheus)
- Celery Flower for task monitoring
- Frontend error logging (Sentry)

**Backup:**
- Daily PostgreSQL backups
- Model file versioning (S3/MinIO)

---

## 10. 📊 Success Metrics

### 10.1 Technical Metrics

- **API Response Time:** < 200ms for predictions
- **WebSocket Latency:** < 50ms for training updates
- **System Uptime:** > 99.5%
- **Database Query Time:** < 100ms average
- **Frontend Load Time:** < 2s initial load

### 10.2 User Experience Metrics

- **Wizard Completion Rate:** > 90%
- **Prediction Accuracy Display:** Real-time updates
- **Explanation Quality:** Human-readable, actionable
- **Report Generation Time:** < 30s for PDF

### 10.3 Business Metrics

- **Time to Add New Machine:** < 30 minutes (vs. manual setup)
- **Prediction Coverage:** 100% of fleet (26 machines)
- **Alert Response Time:** < 5 minutes from prediction to notification

---

## 11. 📚 Documentation Deliverables

By end of Phase 3.7, deliver:

1. **API Documentation:** Auto-generated OpenAPI (Swagger UI)
2. **User Guide:** PDF with screenshots for each workflow
3. **Developer Guide:** Setup instructions, architecture diagrams
4. **Deployment Guide:** Docker commands, environment setup
5. **Video Demo:** 5-minute walkthrough of all features

---

## 12. 🎯 Next Steps

**Immediate Actions:**
1. Review and approve this implementation plan
2. Setup development environment (Docker, Node, Python)
3. Create GitHub project board for task tracking
4. Begin Phase 3.7.1 (Foundation Setup)

**Decision Points:**
- [ ] Confirm tech stack (React + FastAPI)
- [ ] Authentication strategy (none vs. JWT)
- [ ] Deployment target (local vs. cloud)
- [ ] Scope adjustments (which features are MVP vs. nice-to-have)

---

**End of Document**

This comprehensive plan integrates all three subsystems (GAN, ML, LLM) into a unified dashboard with clear implementation phases.
