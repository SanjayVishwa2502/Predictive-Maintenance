# PHASE 3.7.8: ML MODEL TRAINING INTEGRATION
**Complete Frontend Implementation for Model Training Workflow**  
**Duration:** 1.5 weeks (Days 1-10)  
**Goal:** Enable users to train ML models from dashboard using synthetic data  
**Status:** 🟡 READY TO START (December 19, 2025)

---

## Overview

**What Phase 3.7.8 Does:**
- Adds ML Model Training capability to the dashboard
- Enables training of 4 model types (Classification, Regression, Anomaly, Time-Series)
- Provides real-time progress tracking during training
- Displays training results and model metrics
- Integrates with existing GAN synthetic data workflow
- **NEW:** Separate navigation panel feature (not embedded in Predictions page)

**Key User Flow:**
```
1. User navigates to "Model Training" in nav panel
2. Selects machine with synthetic data
3. Chooses model types to train (ALL 4 REQUIRED for complete ML functionality)
4. Configures training parameters (optional)
5. Starts training job (recommended: batch train all 4 models)
6. Monitors real-time progress for each model
7. Views results and downloads models
8. Newly trained models appear in Predictions page
```

**⚠️ CRITICAL: All 4 Model Types Required**

Each model type serves a **distinct, non-overlapping purpose**:

| Model Type | Specific Purpose | Why It's Essential |
|------------|------------------|-------------------|
| **Classification** | Binary failure prediction (normal/failure) | Answers: "Will this machine fail soon?" |
| **Regression (RUL)** | Remaining Useful Life in hours | Answers: "How much time until failure?" |
| **Anomaly Detection** | Unusual sensor behavior detection | Answers: "Is current behavior abnormal?" |
| **Time-Series** | Future sensor value forecasting | Answers: "What will sensors look like tomorrow?" |

**Complete ML System = All 4 Models Trained**

Without all 4 models, the predictive maintenance system is incomplete:
- ❌ Only Classification → Can't estimate time-to-failure
- ❌ Only RUL → Can't detect unexpected anomalies
- ❌ Only Anomaly → Can't predict normal degradation patterns
- ❌ Only Time-Series → Can't classify current health state

✅ **Recommended Workflow:** Always train all 4 models using "Train All Models" batch option

**Architecture:**
```
Frontend (React)     Backend API          Celery Workers       Training Scripts
    ↓                    ↓                      ↓                    ↓
ModelTrainingView → /api/ml/train/* → ml_training_tasks.py → train_*.py
    ↓                    ↓                      ↓                    ↓
Progress Monitor  ← Redis Pub/Sub ← Progress Broadcast ← AutoGluon/Prophet
    ↓
Results Display  ← GET /api/ml/tasks/{task_id} ← Task Result ← Model Artifacts
```

---

## Backend Testing Completed ✅

**Testing Summary (December 19, 2025):**
- ✅ All 4 training scripts verified working
- ✅ Test machine: `cnc_dmg_mori_nlx_010`
- ✅ Total training time: ~4.5 minutes
- ✅ Total model size: ~1.27 GB
- ✅ 100% success rate (4/4 model types)

**Test Results:**

| Model Type | Time | Size | Status |
|------------|------|------|--------|
| Classification | 46 sec | 306 MB | ✅ F1: 0.75 |
| Regression (RUL) | 185 sec | 935 MB | ✅ R²: 1.0 |
| Anomaly Detection | 10 sec | 24 MB | ✅ F1: 1.0 |
| Time-Series | 11 sec | 8 MB | ✅ MAPE: 10.71% |

**See:** `ML_BACKEND_TRAINING_TEST_REPORT.md` for detailed results

---

## Prerequisites

**Completed Work:**
- ✅ Training scripts operational: `ml_models/scripts/training/*.py`
- ✅ Synthetic data available: `GAN/data/synthetic/{machine_id}/`
- ✅ AutoGluon 1.4.0 installed and working
- ✅ Prophet library installed for time-series
- ✅ MLflow tracking configured
- ✅ Celery infrastructure from GAN integration (Phase 3.7.2)
- ✅ Backend ML Manager service (Phase 3.7.3)

**Training Scripts Validated:**
- `train_classification_fast.py` - Binary failure prediction
- `train_regression_fast.py` - RUL estimation
- `train_anomaly_comprehensive.py` - Ensemble anomaly detection
- `train_timeseries.py` - Prophet-based forecasting

---

## Integration Points

**Reuse from Existing Phases:**
- Celery infrastructure (Phase 3.7.1.4)
- Redis pub/sub pattern (Phase 3.7.2.3)
- Progress monitoring pattern (Phase 3.7.2.4)
- Navigation panel (Phase 3.7.6.1)
- MLManager service (Phase 3.7.3)

**New Components:**
- ML Training API routes
- ML Training Celery tasks
- ModelTrainingView component
- Training progress monitor
- Results dashboard

---

## PHASE 3.7.8.1: BACKEND API ROUTES
**Duration:** Days 1-2  
**Goal:** Create REST API endpoints for ML training operations

### Tasks

**Backend Files:**
- [⏳] Create `frontend/server/api/routes/ml_training.py`
- [⏳] Add training endpoints to main router
- [⏳] Create request/response models in `api/models/ml_training.py`

---

### API Endpoints Design

**Start endpoints base path:** `/api/ml/train`  
**Task polling base path (recommended):** `/api/ml/tasks`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/ml/train/classification` | POST | Start classification training |
| `/api/ml/train/regression` | POST | Start RUL regression training |
| `/api/ml/train/anomaly` | POST | Start anomaly detection training |
| `/api/ml/train/timeseries` | POST | Start time-series training |
| `/api/ml/train/batch` | POST | Train all 4 models sequentially |
| `/api/ml/tasks/{task_id}` | GET | Get task status (same shape as GAN `TaskStatusResponse`) |
| `/api/ml/tasks/{task_id}/cancel` | POST | Cancel running task |
| `/api/ml/train/machines/available` | GET | (Optional) list machines with synthetic data (can reuse `/api/gan/machines` + filter) |

---

### Request Models

**File:** `frontend/server/api/models/ml_training.py`

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class TrainingRequest(BaseModel):
    machine_id: str = Field(..., description="Machine identifier")
    time_limit: Optional[int] = Field(900, description="Training time limit in seconds")
    
class BatchTrainingRequest(BaseModel):
    machine_id: str
    model_types: List[str] = Field(
        default=["classification", "regression", "anomaly", "timeseries"],
        description="Models to train"
    )
    time_limit_per_model: Optional[int] = 900
```

---

### Response Models

**Design goal (matches current frontend use-cases):**
Return a Celery `task_id` from the start endpoints, then let the frontend poll a status endpoint that returns the **same** `TaskStatusResponse` shape currently used for GAN tasks.

```python
from pydantic import BaseModel

class StartTrainingResponse(BaseModel):
  success: bool
  machine_id: str
  task_id: str
  message: str

# Task polling response should mirror:
# `frontend/client/src/modules/ml/types/gan.types.ts::TaskStatusResponse`
# - status: PENDING|STARTED|PROGRESS|SUCCESS|FAILURE|RETRY|REVOKED
# - progress: { progress_percent, stage, message, ... }
# - logs, error, result, started_at, completed_at
```

---

### Example Endpoint Implementation

**File:** `frontend/server/api/routes/ml_training.py`

```python
from fastapi import APIRouter, HTTPException
from ..models.ml_training import TrainingRequest, StartTrainingResponse
from tasks.ml_training_tasks import train_classification

router = APIRouter(prefix="/api/ml/train", tags=["ML Training"])

@router.post("/classification", response_model=StartTrainingResponse)
async def start_classification_training(request: TrainingRequest):
    """Start classification model training job"""
    
    # Validate machine has synthetic data
    data_path = f"GAN/data/synthetic/{request.machine_id}/train.parquet"
    if not os.path.exists(data_path):
        raise HTTPException(
            status_code=404,
            detail=f"Synthetic data not found for {request.machine_id}"
        )
    
    # Start Celery task
    task = train_classification.delay(
        machine_id=request.machine_id,
        time_limit=request.time_limit
    )
    
    return StartTrainingResponse(
      success=True,
      task_id=task.id,
      machine_id=request.machine_id,
      message="Classification training task started"
    )
```

---

### Expected Output

- ✅ 8 new API endpoints created
- ✅ Request/response models defined
- ✅ Validation for synthetic data existence
- ✅ Integration with Celery tasks
- ✅ Swagger documentation auto-generated

---

## PHASE 3.7.8.2: CELERY TRAINING TASKS
**Duration:** Days 3-4  
**Goal:** Implement async training tasks with progress tracking

### Tasks

**Backend Files:**
- [⏳] Create `frontend/server/tasks/ml_training_tasks.py`
- [⏳] Implement 4 training task functions
- [⏳] Add progress broadcasting to Redis
- [⏳] Handle error cases and cleanup

---

### Training Task Pattern

**File:** `frontend/server/tasks/ml_training_tasks.py`

**Following GAN Task Pattern:**
- Progress broadcasting via Redis pub/sub
- Structured logging with task_id
- Error handling with cleanup
- MLflow experiment tracking
- Model artifact validation

```python
from celery import Task
from celery_app import celery_app
import subprocess
import json
from pathlib import Path

@celery_app.task(bind=True, name="ml.train_classification")
def train_classification(self: Task, machine_id: str, time_limit: int = 900):
    """
    Train classification model using existing training script
    
    Progress stages:
    1. Validating data (0-10%)
    2. Loading data (10-20%)
    3. Training models (20-90%)
    4. Evaluating results (90-95%)
    5. Saving model (95-100%)
    """
    task_id = self.request.id
    
    # Stage 1: Validate data
    broadcast_progress(
        task_id, machine_id, 10, 100, 
        "RUNNING", "Validating synthetic data..."
    )
    
    # Stage 2: Execute training script (scripts remain the single source of truth)
    script_path = Path("ml_models/scripts/training/train_classification_fast.py")
    cmd = [
      "python",
      str(script_path),
      "--machine_id",
      machine_id,
      "--time_limit",
      str(time_limit),
    ]

    broadcast_progress(task_id, machine_id, 20, 100, "RUNNING", "Launching training script...")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
      broadcast_progress(task_id, machine_id, 100, 100, "FAILURE", "Training failed", error=proc.stderr)
      raise RuntimeError(proc.stderr)

    # Convention: scripts write a JSON report under the model directory
    report_path = f"ml_models/models/classification/{machine_id}/training_report.json"
    broadcast_progress(task_id, machine_id, 100, 100, "SUCCESS", "Training completed successfully")

    return {
      "status": "completed",
      "machine_id": machine_id,
      "model_type": "classification",
      "report_path": report_path,
      "stdout": proc.stdout,
    }
```

---

### Progress Broadcasting

**Reusing GAN Pattern:**

```python
def broadcast_progress(
    task_id: str,
    machine_id: str,
    current: int,
    total: int,
    status: str,
    message: str,
    **metadata
):
    """Broadcast to Redis channel: ml:training:{task_id}"""
    redis_client = get_redis_pubsub()
    
    progress_data = {
        'task_id': task_id,
        'machine_id': machine_id,
        'progress': (current / total) * 100,
        'status': status,
        'message': message,
        **metadata
    }
    
    channel = f"ml:training:{task_id}"
    redis_client.publish(channel, json.dumps(progress_data))
```

---

### Task Registry

**All 4 Training Tasks (Each Required for Complete ML System):**

```python
@celery_app.task(name="ml.train_classification")
def train_classification(machine_id: str, time_limit: int = 900):
    """
    REQUIRED: Trains binary failure classifier (normal vs failure)
    Purpose: Answers "Will this machine fail soon?"
    Output: Classification model for health state prediction
    """
    ...

@celery_app.task(name="ml.train_regression")
def train_regression(machine_id: str, time_limit: int = 900):
    """
    REQUIRED: Trains RUL regression model
    Purpose: Answers "How much time until failure?"
    Output: Regression model for remaining useful life estimation
    """
    ...

@celery_app.task(name="ml.train_anomaly")
def train_anomaly(machine_id: str, contamination: float = 0.1):
    """
    REQUIRED: Trains ensemble anomaly detectors
    Purpose: Answers "Is current behavior abnormal?"
    Output: Anomaly detection models (LOF, Isolation Forest, etc.)
    """
    ...

@celery_app.task(name="ml.train_timeseries")
def train_timeseries(machine_id: str, forecast_hours: int = 24):
    """
    REQUIRED: Trains Prophet time-series forecasters
    Purpose: Answers "What will sensors look like tomorrow?"
    Output: Time-series models for future sensor predictions
    """
    ...

@celery_app.task(name="ml.train_batch")
def train_batch(machine_id: str, model_types: List[str] = None):
    """
    RECOMMENDED: Trains all 4 models sequentially for complete ML system
    Default: Trains all 4 model types (classification, regression, anomaly, timeseries)
    Total time: ~4.5 minutes
    """
    if model_types is None:
        model_types = ["classification", "regression", "anomaly", "timeseries"]
    
    results = {}
    for model_type in model_types:
        task = globals()[f"train_{model_type}"]
        result = task(machine_id)
        results[model_type] = result
    
    return {
        "machine_id": machine_id,
        "models_trained": len(model_types),
        "complete_system": len(model_types) == 4,
        "results": results
    }
```

---

### Expected Output

- ✅ 5 Celery tasks registered (4 single + 1 batch)
- ✅ Progress updates broadcast every 5-10 seconds
- ✅ Training logs captured and stored
- ✅ Models saved to correct directories
- ✅ Error handling with rollback

---

## PHASE 3.7.8.3: FRONTEND NAV INTEGRATION
**Duration:** Day 5  
**Goal:** Add "Model Training" to navigation panel

### Tasks

**Frontend Files:**
- [⏳] Update `frontend/client/src/modules/ml/context/DashboardContext.tsx` (add new view id)
- [⏳] Update `frontend/client/src/modules/ml/components/NavigationPanel.tsx` (add nav option)
- [⏳] Update `frontend/client/src/pages/MLDashboardPage.tsx` (render training view when selected)

---

### Navigation Panel Update

**Important architectural note (matches current codebase):**
This dashboard is **view-based** (controlled by `selectedView` in `DashboardContext`) rather than route-based.  
So adding “Model Training” means:
- add a new `DashboardView` id
- add a new nav option with that id
- render the training view inside `MLDashboardPage.tsx`

**File:** `frontend/client/src/modules/ml/components/NavigationPanel.tsx`

**Add to Navigation Items:**

```typescript
// Add a new option that uses a view id (not a route)
{
  id: 'training',
  label: 'Model Training',
  icon: <SchoolIcon />,
  description: 'Train all 4 ML models from synthetic data',
  dividerAfter: true,
}
```

**Icon Suggestion:**
```typescript
import SchoolIcon from '@mui/icons-material/School';  // Education/training
// OR
import PsychologyIcon from '@mui/icons-material/Psychology';  // AI brain
// OR
import BuildIcon from '@mui/icons-material/Build';  // Build/construction
```

---

### Dashboard View Wiring

**File:** `frontend/client/src/modules/ml/context/DashboardContext.tsx`

Add the new view id:

```typescript
export type DashboardView =
  | 'predictions'
  | 'gan'
  | 'training' // NEW
  | 'history'
  | 'reports'
  | 'tasks'
  | 'datasets'
  | 'settings';
```

**File:** `frontend/client/src/pages/MLDashboardPage.tsx`

- Add `training` to the view title switch (if used)
- Render training when `selectedView === 'training'`

---

### Expected Output

- ✅ "Model Training" appears in nav panel
- ✅ Clicking switches the dashboard to the `training` view
- ✅ Active state highlights when on training page
- ✅ Icon matches design system

---

## PHASE 3.7.8.4: MODEL TRAINING VIEW (MAIN UI)
**Duration:** Days 6-7  
**Goal:** Build main training interface with machine selector and model type chooser

### Tasks

**Frontend Files:**
- [⏳] Create `frontend/client/src/modules/ml/components/training/ModelTrainingView.tsx`
- [⏳] Reuse `frontend/client/src/modules/ml/components/MachineSelector.tsx` (existing)
- [⏳] Create `frontend/client/src/modules/ml/components/training/ModelTypeSelector.tsx`
- [⏳] Create `frontend/client/src/modules/ml/components/training/TrainingConfigForm.tsx`
- [⏳] Create `frontend/client/src/modules/ml/api/mlTrainingApi.ts` (axios client, parallel to `ganApi.ts`)
- [⏳] Implement training start logic (register task via `TaskSessionContext`)

---

### Main View Structure

**File:** `frontend/client/src/modules/ml/components/training/ModelTrainingView.tsx`

**Machine list source (matches current dashboard use-cases):**
Use `ganApi.getMachines()` and filter to `machine.status.has_synthetic_data === true`. This aligns with the existing “Downloads” view, which already treats GAN outputs (synthetic parquet splits) as the source-of-truth.

```typescript
export default function ModelTrainingView() {
  const { setSelectedView } = useDashboard();
  const { registerRunningTask } = useTaskSession();

  // MachineSelector expects Machine[] (see existing `MachineSelector.tsx`)
  const [machines, setMachines] = useState<Machine[]>([]);
  const [selectedMachineId, setSelectedMachineId] = useState<string | null>(null);
  const [selectedModels, setSelectedModels] = useState<Set<ModelType>>(new Set());
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      const machineList = await ganApi.getMachines();
      const withSynthetic = (machineList.machine_details || [])
        .filter((m) => m.status?.has_synthetic_data)
        .map((m) => ({
          machine_id: m.machine_id,
          display_name: `${m.manufacturer} ${m.model}`,
          category: m.machine_type,
          manufacturer: m.manufacturer,
          model: m.model,
          sensor_count: m.num_sensors,
          // Optional: populate from /api/ml/machines if you want accurate badges
          has_classification_model: false,
          has_regression_model: false,
          has_anomaly_model: false,
          has_timeseries_model: false,
        }));
      setMachines(withSynthetic);
    })();
  }, []);

  const handleStartTraining = async () => {
    if (!selectedMachineId) return;
    const resp = await mlTrainingApi.startTraining({
      machine_id: selectedMachineId,
      model_types: Array.from(selectedModels),
    });
    // Keep `kind: 'train'` for compatibility with current TaskSessionContext;
    // optionally extend TaskKind to distinguish ML-vs-GAN training.
    registerRunningTask({ task_id: resp.task_id, machine_id: selectedMachineId, kind: 'train' });
    setActiveTaskId(resp.task_id);
  };

  if (machines.length === 0) {
    return (
      <Alert severity="info">
        No machines have synthetic training data yet. Run the GAN wizard first.
        <Button onClick={() => setSelectedView('gan')}>Open GAN Wizard</Button>
      </Alert>
    );
  }

  return (
    <Box>
      <MachineSelector machines={machines} selectedMachineId={selectedMachineId} onSelect={setSelectedMachineId} />
      <ModelTypeSelector selected={selectedModels} onChange={setSelectedModels} />
      <TrainingConfigForm />
      <Button variant="contained" onClick={handleStartTraining}>Start Training</Button>
      {activeTaskId && <TrainingProgressMonitor taskId={activeTaskId} />}
    </Box>
  );
}
```

**Recommended gating UX:**
- If the selected machine does not have synthetic data, disable “Start Training” and show a short message.
- Provide a one-click action that switches to GAN (`setSelectedView('gan')`).

---

### Model Type Selector Component

**File:** `frontend/client/src/modules/ml/components/training/ModelTypeSelector.tsx`

**⚠️ IMPORTANT: All 4 models should be selected by default!**

```typescript
const modelTypes = [
  {
    type: 'classification',
    label: 'Classification (Failure Prediction)',
    icon: <CategoryIcon />,
    description: 'Binary classification: normal vs. failure',
    purpose: 'Answers: Will this machine fail?',
    estimatedTime: '~1 minute',
    required: true  // Essential for complete ML system
  },
  {
    type: 'regression',
    label: 'Regression (RUL Estimation)',
    icon: <TrendingUpIcon />,
    description: 'Predict remaining useful life in hours',
    purpose: 'Answers: How much time until failure?',
    estimatedTime: '~3 minutes',
    required: true  // Essential for complete ML system
  },
  {
    type: 'anomaly',
    label: 'Anomaly Detection',
    icon: <WarningIcon />,
    description: 'Ensemble anomaly detectors',
    purpose: 'Answers: Is current behavior abnormal?',
    estimatedTime: '~10 seconds',
    required: true  // Essential for complete ML system
  },
  {
    type: 'timeseries',
    label: 'Time-Series Forecast',
    icon: <ShowChartIcon />,
    description: '24-hour ahead sensor predictions',
    purpose: 'Answers: What will sensors look like tomorrow?',
    estimatedTime: '~10 seconds',
    required: true  // Essential for complete ML system
  }
];

export function ModelTypeSelector({ selected, onChange }) {
  // Initialize with all models selected by default
  useEffect(() => {
    if (selected.size === 0) {
      const allModels = new Set(modelTypes.map(m => m.type));
      onChange(allModels);
    }
  }, []);
  
  return (
    <Box>
      {/* Quick Action: Train All (Recommended) */}
      <Alert severity="info" sx={{ mb: 2 }}>
        <AlertTitle>Recommended: Train All 4 Models</AlertTitle>
        Each model serves a distinct purpose. Training all 4 provides complete 
        predictive maintenance coverage (~4.5 minutes total).
      </Alert>
      
      <Button 
        variant="contained" 
        fullWidth 
        sx={{ mb: 2 }}
        onClick={() => {
          const allModels = new Set(modelTypes.map(m => m.type));
          onChange(allModels);
        }}
      >
        ✓ Select All 4 Models (Recommended)
      </Button>
      
      <Grid container spacing={2}>
        {modelTypes.map(model => (
          <Grid item xs={12} md={6} key={model.type}>
            <Card 
              variant={selected.has(model.type) ? "outlined" : "elevation"}
              sx={{ 
                cursor: 'pointer',
                borderColor: model.required ? 'primary.main' : undefined,
                borderWidth: selected.has(model.type) ? 2 : 1
              }}
              onClick={() => toggleModel(model.type)}
            >
              <CardContent>
                <Box display="flex" alignItems="center" gap={1}>
                  <Checkbox 
                    checked={selected.has(model.type)} 
                    color="primary"
                  />
                  {model.icon}
                  <Typography variant="h6">{model.label}</Typography>
                  {model.required && (
                    <Chip 
                      label="REQUIRED" 
                      size="small" 
                      color="primary"
                    />
                  )}
                </Box>
                <Typography variant="body2" color="text.secondary">
                  {model.description}
                </Typography>
                <Typography 
                  variant="body2" 
                  color="primary" 
                  sx={{ mt: 1, fontWeight: 'bold' }}
                >
                  {model.purpose}
                </Typography>
                <Chip 
                  label={model.estimatedTime} 
                  size="small" 
                  sx={{ mt: 1 }}
                />
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
      
      {/* Warning if not all selected */}
      {selected.size < 4 && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          ⚠️ Incomplete ML System: {4 - selected.size} model(s) not selected. 
          For full predictive maintenance capability, train all 4 models.
        </Alert>
      )}
    </Box>
  );
}
```

---

### Training Config Form

**Advanced Options (Collapsible):**

```typescript
export function TrainingConfigForm({ config, onChange }) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  return (
    <Box>
      <FormControl fullWidth>
        <InputLabel>Time Limit</InputLabel>
        <Select value={config.timeLimit} onChange={...}>
          <MenuItem value={900}>15 minutes (Fast - Pi)</MenuItem>
          <MenuItem value={1800}>30 minutes (Standard)</MenuItem>
          <MenuItem value={3600}>60 minutes (High Quality)</MenuItem>
        </Select>
      </FormControl>
      
      <Accordion expanded={showAdvanced}>
        <AccordionSummary>Advanced Options</AccordionSummary>
        <AccordionDetails>
          {/* Preset selector */}
          {/* Excluded models */}
          {/* Cross-validation folds */}
        </AccordionDetails>
      </Accordion>
    </Box>
  );
}
```

---

### Expected Output

- ✅ Machine selector dropdown populated with machines where `status.has_synthetic_data === true`
- ✅ **All 4 model type cards selected by default** (recommended workflow)
- ✅ "Select All 4 Models" quick action button prominently displayed
- ✅ Warning message if user deselects any model (incomplete ML system)
- ✅ Each card shows specific purpose ("Answers: Will machine fail?")
- ✅ Configuration form with sensible defaults (15 min, Pi-compatible)
- ✅ Start button shows "Train All 4 Models" when all selected
- ✅ Estimated total time displayed: "~4.5 minutes for complete system"

---

## PHASE 3.7.8.5: PROGRESS MONITOR COMPONENT
**Duration:** Days 8-9  
**Goal:** Training progress display using the existing polling pattern (same UX as GAN tasks)

### Tasks

**Frontend Files:**
- [⏳] Create `frontend/client/src/modules/ml/components/training/TrainingProgressMonitor.tsx`
- [⏳] Implement progress bar with stages
- [⏳] Add live log viewer (render `TaskStatusResponse.logs`)
- [⏳] Display current metrics (F1, R², etc.)
- [⏳] Handle completion/error states
- [⏳] (If needed) extend `TaskKind` in `TaskSessionContext.tsx` to include `ml_train`

---

### Progress Monitor Design

**File:** `frontend/client/src/modules/ml/components/training/TrainingProgressMonitor.tsx`

**Layout:**
```
┌─────────────────────────────────────────┐
│ Training: cnc_dmg_mori_nlx_010          │
│ Model: Classification                    │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 75%        │
│ Current: Training XGBoost model...      │
│ Estimated time remaining: 2 minutes     │
├─────────────────────────────────────────┤
│ [Live Training Logs]                    │
│ > Loading data... ✓                     │
│ > Training LightGBM... ✓ (F1: 0.74)     │
│ > Training RandomForest... ⏳            │
├─────────────────────────────────────────┤
│ [Models Leaderboard]                    │
│ 1. LightGBM      F1: 0.7470            │
│ 2. RandomForest  F1: 0.7517 (training) │
└─────────────────────────────────────────┘
```

---

### Implementation

```typescript
export function TrainingProgressMonitor({ taskId }: { taskId: string }) {
  const [status, setStatus] = useState<TaskStatusResponse | null>(null);
  const { updateTaskFromStatus } = useTaskSession();

  useEffect(() => {
    const interval = setInterval(async () => {
      const next = await mlTrainingApi.getTaskStatus(taskId);
      setStatus(next);
      updateTaskFromStatus(taskId, next);

      if (next.status === 'SUCCESS' || next.status === 'FAILURE' || next.status === 'REVOKED') {
        clearInterval(interval);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [taskId, updateTaskFromStatus]);

  const progress = status?.progress?.progress_percent ?? 0;
  const message = status?.progress?.message || status?.progress?.stage || '';

  return (
    <Paper sx={{ p: 2, mt: 2 }}>
      <Typography variant="h6">Training Progress</Typography>

      <LinearProgress variant="determinate" value={progress} sx={{ my: 2 }} />
      <Typography variant="body2" color="text.secondary">{message}</Typography>

      {typeof status?.logs === 'string' && status.logs.length > 0 && (
        <Paper variant="outlined" sx={{ p: 2, mt: 2, maxHeight: 300, overflow: 'auto' }}>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>Logs</Typography>
          <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
            {status.logs}
          </Typography>
        </Paper>
      )}

      {status?.status === 'SUCCESS' && <Alert severity="success">Training completed.</Alert>}
      {status?.status === 'FAILURE' && <Alert severity="error">Training failed: {status.error}</Alert>}
    </Paper>
  );
}
```

---

### Expected Output

- ✅ Progress bar updates every 2 seconds
- ✅ Current training step displayed
- ✅ Live logs scroll automatically
- ✅ Success/error alerts shown
- ✅ Cancel button to abort training

---

## PHASE 3.7.8.6: RESULTS DASHBOARD
**Duration:** Day 10  
**Goal:** Display training results and model performance metrics

### Tasks

**Frontend Files:**
- [⏳] Create `frontend/client/src/modules/ml/components/training/TrainingResultsDashboard.tsx`
- [⏳] Display performance metrics (accuracy, F1, R², RMSE)
- [⏳] Show model leaderboard
- [⏳] Add download model button
- [⏳] Link to Predictions view for testing (`setSelectedView('predictions')`)

---

### Results Dashboard Design

**File:** `frontend/client/src/modules/ml/components/training/TrainingResultsDashboard.tsx`

```typescript
export function TrainingResultsDashboard({ 
  taskId,
  machineId, 
  modelType 
}: ResultsProps) {
  const { setSelectedView } = useDashboard();
  const [results, setResults] = useState<TrainingResults | null>(null);

  useEffect(() => {
    (async () => {
      const r = await mlTrainingApi.getTrainingResults(taskId);
      setResults(r);
    })();
  }, [taskId]);
  
  return (
    <Grid container spacing={3}>
      {/* Summary Cards */}
      <Grid item xs={12} md={3}>
        <Card>
          <CardContent>
            <Typography variant="h4">{results.accuracy}</Typography>
            <Typography color="text.secondary">Accuracy</Typography>
          </CardContent>
        </Card>
      </Grid>
      
      {/* Model Leaderboard */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardHeader title="Model Performance Ranking" />
          <Table>
            <TableBody>
              {results.models.map((model, idx) => (
                <TableRow key={model.name}>
                  <TableCell>{idx + 1}</TableCell>
                  <TableCell>{model.name}</TableCell>
                  <TableCell>{model.score.toFixed(4)}</TableCell>
                  <TableCell>
                    {idx === 0 && <Chip label="Best" color="success" />}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Card>
      </Grid>
      
      {/* Confusion Matrix (for classification) */}
      {modelType === 'classification' && (
        <Grid item xs={12} md={6}>
          <ConfusionMatrix data={results.confusion_matrix} />
        </Grid>
      )}
      
      {/* Feature Importance */}
      <Grid item xs={12}>
        <FeatureImportanceChart features={results.feature_importance} />
      </Grid>
      
      {/* Actions */}
      <Grid item xs={12}>
        <ButtonGroup>
          <Button startIcon={<DownloadIcon />}>
            Download Model
          </Button>
          <Button 
            startIcon={<PlayArrowIcon />}
            onClick={() => setSelectedView('predictions')}
          >
            Test Predictions
          </Button>
          <Button startIcon={<ShareIcon />}>
            Share Report
          </Button>
        </ButtonGroup>
      </Grid>
    </Grid>
  );
}
```

---

### Expected Output

- ✅ Training metrics displayed in cards
- ✅ Model leaderboard with rankings
- ✅ Confusion matrix visualization (classification)
- ✅ Feature importance bar chart
- ✅ Download/test/share actions

---

## Integration Checklist

### Backend Integration

- [ ] API routes integrated in main FastAPI app
- [ ] **All 4 individual training endpoints implemented**
- [ ] **Batch training endpoint (trains all 4) implemented**
- [ ] Celery tasks registered in worker
- [ ] Redis pub/sub channels configured
- [ ] **All 4 training scripts callable from tasks**
- [ ] Error handling and logging
- [ ] MLflow experiment tracking
- [ ] Model artifact validation
- [ ] **Validation: All 4 models exist after batch training**

### Frontend Integration

- [ ] Navigation panel updated
- [ ] `training` view wired into `MLDashboardPage.tsx`
- [ ] Machine selector fetches correct data
- [ ] **Model type selector shows all 4 models with distinct purposes**
- [ ] **All 4 models selected by default**
- [ ] **Warning shown if any model deselected**
- [ ] Training config form working
- [ ] **Progress monitor tracks all 4 models individually**
- [ ] Results dashboard displays metrics
- [ ] **Results show performance for all trained models**
- [ ] Links to Predictions page

### Testing (Critical: All 4 Models)

- [ ] **Test Classification training** (binary failure prediction)
- [ ] **Test Regression training** (RUL estimation)
- [ ] **Test Anomaly training** (unusual behavior detection)
- [ ] **Test Time-Series training** (future forecasting)
- [ ] **Test batch training (all 4 models together)** ⭐ PRIMARY USE CASE
- [ ] Monitor progress in real-time for each model
- [ ] Cancel training job (mid-execution)
- [ ] View results after completion (all 4 models)
- [ ] **Verify all 4 models appear in Predictions page**
- [ ] Download trained models
- [ ] **Test predictions work with all 4 model types**
- [ ] Error handling (no synthetic data, partial failures)
- [ ] **Verify incomplete training warning if <4 models selected**

---

## Expected Deliverables

### Backend

1. **API Routes File:** `frontend/server/api/routes/ml_training.py` (~200 lines)
2. **Celery Tasks File:** `frontend/server/tasks/ml_training_tasks.py` (~400 lines)
3. **Request/Response Models:** `frontend/server/api/models/ml_training.py` (~100 lines)

### Frontend

1. **API Client:** `frontend/client/src/modules/ml/api/mlTrainingApi.ts` (~150 lines)
2. **Main View:** `frontend/client/src/modules/ml/components/training/ModelTrainingView.tsx` (~300 lines)
3. **Progress Monitor:** `frontend/client/src/modules/ml/components/training/TrainingProgressMonitor.tsx` (~200 lines)
4. **Results Dashboard:** `frontend/client/src/modules/ml/components/training/TrainingResultsDashboard.tsx` (~250 lines)
5. **Model Type Selector:** `frontend/client/src/modules/ml/components/training/ModelTypeSelector.tsx` (~150 lines)
6. **Training Config Form:** `frontend/client/src/modules/ml/components/training/TrainingConfigForm.tsx` (~100 lines)

### Documentation

1. **API Documentation:** Auto-generated Swagger/OpenAPI
2. **User Guide:** How to train models from dashboard
3. **Training Report:** Generated after each training job

---

## Success Criteria

✅ **User can train all 4 model types from dashboard**  
✅ **Real-time progress updates every 2-5 seconds**  
✅ **Training completes in ~4-5 minutes (all 4 models)**  
✅ **Results dashboard shows comprehensive metrics**  
✅ **Newly trained models immediately available in Predictions**  
✅ **Error handling for missing data, failed training**  
✅ **Training jobs can be cancelled mid-execution**  
✅ **Multiple users can train different machines simultaneously**

---

## Performance Targets

| Metric | Target | Actual (Backend Test) |
|--------|--------|----------------------|
| Classification Training | <2 min | 46 sec ✅ |
| Regression Training | <5 min | 3.09 min ✅ |
| Anomaly Training | <1 min | 10 sec ✅ |
| Timeseries Training | <1 min | 11 sec ✅ |
| **Total (All 4)** | **<10 min** | **4.5 min ✅** |

---

## Raspberry Pi Considerations

**Model Size Limits:**
- Classification: 306 MB ✅ OK
- Regression: 935 MB ⚠️ Large (consider compression)
- Anomaly: 24 MB ✅ Perfect
- Timeseries: 8 MB ✅ Perfect

**Recommendations:**
- Add model compression option in config form
- Warning if model >500 MB for Pi deployment
- Suggest reduced `num_bag_folds` for large models

---

## Future Enhancements (Out of Scope)

- [ ] Hyperparameter tuning UI
- [ ] AutoML mode (automatic model selection)
- [ ] Training history comparison
- [ ] Model versioning system
- [ ] A/B testing between models
- [ ] Distributed training across multiple machines
- [ ] Custom model upload
- [ ] Transfer learning from pre-trained models

---

## Related Documents

- **Training Workflow:** `ML_TRAINING_WORKFLOW.md`
- **Backend Test Report:** `ML_BACKEND_TRAINING_TEST_REPORT.md`
- **ML System Overview:** `ML_PART_OVERVIEW.md`
- **GAN Integration:** `PHASE_3.7.2_GAN_INTEGRATION.md` (pattern reference)
- **ML Dashboard:** `PHASE_3.7.3_ML_DASHBOARD.md` (predictions page)

---

## Timeline Summary

| Phase | Days | Deliverables |
|-------|------|--------------|
| 3.7.8.1: API Routes | 1-2 | 8 endpoints, request/response models |
| 3.7.8.2: Celery Tasks | 3-4 | 5 training tasks, progress broadcasting |
| 3.7.8.3: Nav Integration | 5 | Navigation panel update, routing |
| 3.7.8.4: Main UI | 6-7 | ModelTrainingView, selectors, config form |
| 3.7.8.5: Progress Monitor | 8-9 | Real-time progress UI, logs viewer |
| 3.7.8.6: Results Dashboard | 10 | Metrics display, leaderboard, actions |

**Total Duration:** 10 days (1.5 weeks)

---

**Document Version:** 1.0  
**Last Updated:** December 19, 2025  
**Status:** 🟡 READY TO START  
**Prerequisites:** ✅ Backend training validated, synthetic data available, Celery infrastructure ready
