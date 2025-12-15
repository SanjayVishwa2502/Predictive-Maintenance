# PHASE 3.7.3: ML DASHBOARD ARCHITECTURE REVISION
**Comprehensive Workflow Assessment & Design Corrections**  
**Date:** November 27, 2025  
**Status:** 🔴 CRITICAL ARCHITECTURE CHANGE REQUIRED

---

## 🚨 CRITICAL DISCOVERY: SINGLE-MACHINE OPERATION

### Original Assumption (INCORRECT)
- **Plan assumption:** 26 machines running simultaneously
- **Dashboard design:** Fleet monitoring with real-time status for all machines
- **Components planned:** `FleetOverviewCards`, `MachineGrid` (3-column grid showing all 26)
- **API design:** Batch predictions, fleet-wide metrics, simultaneous monitoring

### Reality Clarified by User
> **"understand this each an every time only a machine will be intended to run"**

**ACTUAL REQUIREMENT:**
- ✅ **ONLY ONE MACHINE RUNS AT A TIME** (e.g., CNC DMG Mori NLX 010)
- ✅ No fleet-wide monitoring needed
- ✅ No batch predictions across 26 machines
- ✅ Focus: Real-time monitoring of SINGLE ACTIVE machine
- ✅ User selects machine → monitors → switches when machine changes

**Impact:** ~60% of Phase 3.7.3 ML Dashboard plan is **architecturally incorrect**

---

## 📊 ML/LLM WORKFLOW ANALYSIS

### 1. Existing Infrastructure Discovery

#### **1.1 ML Inference Scripts** (`ml_models/scripts/inference/`)
**Discovered 4 predictor classes:**

| Predictor | File | Purpose | Output Structure |
|-----------|------|---------|------------------|
| **ClassificationPredictor** | `predict_classification.py` (367 lines) | Predicts failure type | `failure_type`, `confidence`, `all_probabilities` |
| **RULPredictor** | `predict_rul.py` (387 lines) | Estimates remaining useful life | `rul_hours`, `rul_days`, `urgency`, `maintenance_window` |
| **AnomalyPredictor** | `predict_anomaly.py` | Detects anomalies via Isolation Forest | `is_anomaly`, `score`, `abnormal_sensors` |
| **TimeSeriesPredictor** | `predict_timeseries.py` | Forecasts future sensor values | `forecast_summary`, `confidence` |

**Key Technical Details:**

**ClassificationPredictor:**
```python
# Returns:
{
    "failure_probability": 0.15,          # 1 - normal_prob
    "failure_type": "bearing_wear",       # normal, bearing_wear, overheating, electrical_fault
    "confidence": 0.92,                   # Probability of predicted class
    "all_probabilities": {                # All class probabilities
        "normal": 0.85,
        "bearing_wear": 0.08,
        "overheating": 0.04,
        "electrical_fault": 0.03
    },
    "sensor_readings": {...},             # Input sensor data
    "model_info": {
        "path": "ml_models/models/classification/motor_siemens_1la7_001",
        "best_model": "WeightedEnsemble_L2",
        "num_features": 22
    }
}
```

**RULPredictor:**
```python
# Returns:
{
    "rul_hours": 156.3,                   # Remaining useful life in hours
    "rul_days": 6.51,                     # Remaining useful life in days
    "urgency": "medium",                  # critical, high, medium, low
    "maintenance_window": "within 3 days", # Human-readable recommendation
    "critical_sensors": [                 # Sensors most likely degrading
        {"name": "bearing_de_temp_C", "value": 75.2, "severity": "high"},
        {"name": "vibration_mm_s", "value": 12.1, "severity": "medium"}
    ],
    "estimated_failure_date": "2025-12-04T10:45:00Z",
    "confidence": 0.85
}
```

#### **1.2 LLM Integration System** (`LLM/api/ml_integration.py` - 515 lines)

**IntegratedPredictionSystem Class:**
```python
class IntegratedPredictionSystem:
    """Singleton wrapping all ML predictors + LLM explainer"""
    
    def __init__(self):
        self.explainer = MLExplainer()  # LLM + RAG system
        self.models = {
            'classification': {},  # Lazy-loaded AutoGluon models
            'regression': {},      # Lazy-loaded RUL models
            'anomaly': {},         # Isolation Forest models
            'timeseries': {}       # Time-series forecast models
        }
    
    def predict_with_explanation(self, machine_id, sensor_data, model_type='all'):
        """
        Unified prediction + explanation pipeline
        
        Returns:
        {
            'classification': {
                'prediction': {...},    # ML prediction
                'explanation': {...}    # LLM explanation
            },
            'regression': {...},
            'anomaly': {...},
            'timeseries': {...}
        }
        """
```

**Workflow:**
1. User provides `machine_id` + `sensor_data`
2. System loads models (lazy loading, cached)
3. Runs ML predictions (classification, RUL, anomaly, timeseries)
4. LLM generates human-readable explanations
5. Returns structured JSON with predictions + explanations

**GPU Acceleration:**
- CUDA DLL injection for Windows compatibility
- LLaMA-based LLM running at ~26 tokens/sec
- Models cached to prevent reload overhead

#### **1.3 Machine Configuration System** (`ml_models/config/model_config.py`)

**Key Discoveries:**
- **26 machines total** (not 27 as initially documented)
- **Sensor counts vary:** 1-22 sensors per machine
- **Research-based configurations:** Must be preserved (NOT modified)
- **Machine categories:** Motors, pumps, compressors, CNCs, robots, fans, etc.

**Example Sensor Profiles:**
| Machine | Sensors | Category |
|---------|---------|----------|
| `motor_siemens_1la7_001` | 22 | Very High |
| `cnc_dmg_mori_nlx_010` | 2 | Low |
| `compressor_atlas_copco_ga30_001` | 10 | High |
| `hydraulic_parker_hpu_012` | 3 | Low |

**Critical:** Each machine has unique sensor configuration stored in `GAN/metadata/{machine_id}_metadata.json`

#### **1.4 Model Storage** (`ml_models/models/`)

**Classification Models Trained (10 machines):**
```
ml_models/models/classification/
├── cnc_dmg_mori_nlx_010/
├── compressor_atlas_copco_ga30_001/
├── compressor_ingersoll_rand_2545_009/
├── cooling_tower_bac_vti_018/
├── generic_all_machines/
├── hydraulic_beckwood_press_011/
├── motor_abb_m3bp_002/
├── motor_siemens_1la7_001/
├── motor_weg_w22_003/
├── pooled_test_3_machines/
├── pump_flowserve_ansi_005/
└── pump_grundfos_cr3_004/
```

**Not all 26 machines have trained models yet** (only 10 have classification models)

---

## ❌ DESIGN CONFLICTS IN PHASE 3.7.3 PLAN

### **Conflict 1: Fleet-Wide Monitoring (INCORRECT)**

**Plan Specification (Lines 1260-1310):**
```typescript
// ❌ INCORRECT: FleetOverviewCards component
interface FleetOverviewCardsProps {
  fleetStats: {
    healthy: number;        // Count across 26 machines
    degrading: number;
    warning: number;
    critical: number;
  };
}
```

**Reality:** No fleet stats needed - only ONE machine runs at a time

**Required Change:** Remove `FleetOverviewCards` component entirely

---

### **Conflict 2: Machine Grid (3-Column Display - INCORRECT)**

**Plan Specification (Lines 1395-1420):**
```typescript
// ❌ INCORRECT: MachineGrid showing all 26 machines
<MachineGrid 
  machines={machines}  // All 26 machines displayed
  onViewDetails={handleViewDetails}
/>

// Grid configuration:
grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
// Shows 3 columns on desktop, 2 on tablet
```

**Reality:** No grid needed - user selects ONE machine from dropdown

**Required Change:** Replace with `MachineSelector` dropdown component

---

### **Conflict 3: Batch Prediction API (INCORRECT)**

**Plan Specification (Lines 1098-1150):**
```python
# ❌ INCORRECT: Batch prediction endpoint
POST /api/ml/predict/batch
{
  "machine_ids": ["motor_001", "pump_004", ...],  # Multiple machines
  "sensor_data": {...}
}

# Response: Array of predictions for all machines
```

**Reality:** No batch predictions - predict for ONE active machine only

**Required Change:** Single-machine prediction endpoint:
```python
POST /api/ml/predict/classification
{
  "machine_id": "motor_siemens_1la7_001",  # ONE machine
  "sensor_data": {
    "bearing_de_temp_C": 65.2,
    "vibration_mm_s": 3.4,
    ...
  }
}
```

---

### **Conflict 4: Real-Time Fleet Monitoring (INCORRECT)**

**Plan Specification (Lines 1450-1500):**
```typescript
// ❌ INCORRECT: Fetch all 26 machine predictions every 30 seconds
useEffect(() => {
  const interval = setInterval(fetchAllPredictions, 30000);
  return () => clearInterval(interval);
}, []);

const fetchAllPredictions = async () => {
  const response = await fetch('/api/ml/predict/batch', {
    method: 'POST',
    body: JSON.stringify({ machine_ids: ALL_26_MACHINES })
  });
};
```

**Reality:** Only fetch predictions for ONE selected machine

**Required Change:**
```typescript
// ✅ CORRECT: Monitor only selected machine
useEffect(() => {
  if (selectedMachineId) {
    const interval = setInterval(() => fetchPrediction(selectedMachineId), 10000);
    return () => clearInterval(interval);
  }
}, [selectedMachineId]);
```

---

### **Conflict 5: Auto-Refresh for 26 Machines (INCORRECT)**

**Plan Specification:** Refresh all 26 machine statuses every 30 seconds

**Reality:** No need - refresh only when machine is actively selected

**Required Change:** WebSocket or polling for SINGLE machine only

---

## ✅ CORRECTED ARCHITECTURE

### **Correct Architecture Pattern: Single-Machine Monitoring**

```
┌─────────────────────────────────────────────────────────┐
│              ML DASHBOARD (REVISED DESIGN)              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  🔧 Select Machine: [▼ Motor Siemens 1LA7 001    ]     │  ← Dropdown selector
│                                                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │         REAL-TIME SENSOR MONITORING               │ │
│  ├───────────────────────────────────────────────────┤ │
│  │  🌡️ Bearing Temp: 65.2°C    📊 Vibration: 3.4mm │ │
│  │  ⚡ Current: 12.5A           🔋 Voltage: 410V     │ │
│  │  Last Update: 5 seconds ago                       │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │         MACHINE HEALTH PREDICTION                 │ │
│  ├───────────────────────────────────────────────────┤ │
│  │  Status: 🟢 Healthy                               │ │
│  │  Confidence: 95%                                  │ │
│  │  Failure Type: Normal                             │ │
│  │  RUL: 156 hours (6.5 days)                        │ │
│  │  [🤖 Get AI Explanation]                          │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │         SENSOR TREND CHARTS (Last 10 min)         │ │
│  ├───────────────────────────────────────────────────┤ │
│  │  📈 Line charts for key sensors                   │ │
│  │  (Temperature, Vibration, Current)                │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │         PREDICTION HISTORY (Last 100)             │ │
│  ├───────────────────────────────────────────────────┤ │
│  │  Table: Timestamp | Status | RUL | Confidence     │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ REVISED COMPONENTS

### **NEW Components (Replace Incorrect Ones)**

#### **1. MachineSelector Component** (REPLACES FleetOverviewCards + MachineGrid)

**File:** `frontend/client/src/modules/ml/components/MachineSelector.tsx`

**Purpose:** Dropdown to select ONE active machine

**Interface:**
```typescript
interface MachineSelectorProps {
  machines: Machine[];  // All 26 machines
  selectedMachineId: string | null;
  onSelect: (machineId: string) => void;
}

interface Machine {
  machine_id: string;
  display_name: string;
  category: string;
  sensor_count: number;
  has_trained_model: boolean;
}
```

**Features:**
- ✅ Searchable dropdown (Autocomplete)
- ✅ Group by category (Motors, Pumps, CNCs, etc.)
- ✅ Show sensor count + model status
- ✅ Highlight machines with trained models
- ✅ Disabled state for machines without models

**Example:**
```tsx
<MachineSelector
  machines={allMachines}
  selectedMachineId="motor_siemens_1la7_001"
  onSelect={(id) => setSelectedMachine(id)}
/>
```

---

#### **2. SensorDashboard Component** (NEW)

**File:** `frontend/client/src/modules/ml/components/SensorDashboard.tsx`

**Purpose:** Real-time sensor value display for selected machine

**Interface:**
```typescript
interface SensorDashboardProps {
  machineId: string;
  sensorData: Record<string, number>;  // {"bearing_temp_C": 65.2, ...}
  lastUpdated: Date;
}
```

**Features:**
- ✅ Grid of sensor cards (4 columns)
- ✅ Color-coded values (green/yellow/red based on thresholds)
- ✅ Live update indicator (pulsing dot)
- ✅ Last update timestamp
- ✅ Auto-refresh every 10 seconds

**Example:**
```tsx
<SensorDashboard
  machineId="motor_siemens_1la7_001"
  sensorData={latestSensors}
  lastUpdated={new Date()}
/>
```

---

#### **3. SensorCharts Component** (NEW)

**File:** `frontend/client/src/modules/ml/components/SensorCharts.tsx`

**Purpose:** Time-series line charts for key sensors (last 10 minutes)

**Interface:**
```typescript
interface SensorChartsProps {
  machineId: string;
  sensorHistory: SensorReading[];  // Array of timestamped readings
  selectedSensors: string[];       // ["bearing_temp_C", "vibration_mm_s"]
}

interface SensorReading {
  timestamp: Date;
  values: Record<string, number>;
}
```

**Features:**
- ✅ Multi-line chart (Recharts library)
- ✅ X-axis: Time (last 10 minutes)
- ✅ Y-axis: Sensor value
- ✅ Toggle sensors on/off
- ✅ Zoom and pan controls
- ✅ Auto-scroll as new data arrives

---

#### **4. PredictionCard Component** (REVISED)

**File:** `frontend/client/src/modules/ml/components/PredictionCard.tsx`

**Purpose:** Display ML prediction results for selected machine

**Interface:**
```typescript
interface PredictionCardProps {
  machineId: string;
  prediction: {
    failure_type: string;
    confidence: number;
    rul_hours: number;
    rul_days: number;
    urgency: string;
  };
  onExplain: () => void;
}
```

**Features:**
- ✅ Status badge (Healthy/Degrading/Warning/Critical)
- ✅ Confidence percentage
- ✅ RUL countdown (hours/days)
- ✅ Urgency indicator
- ✅ "Get AI Explanation" button
- ✅ Last prediction timestamp

---

#### **5. PredictionHistory Component** (NEW)

**File:** `frontend/client/src/modules/ml/components/PredictionHistory.tsx`

**Purpose:** Table showing last 100 predictions for selected machine

**Interface:**
```typescript
interface PredictionHistoryProps {
  machineId: string;
  predictions: HistoricalPrediction[];
}

interface HistoricalPrediction {
  timestamp: Date;
  failure_type: string;
  confidence: number;
  rul_hours: number;
}
```

**Features:**
- ✅ Paginated table (10 rows per page)
- ✅ Sort by timestamp
- ✅ Filter by failure type
- ✅ Export to CSV
- ✅ Color-coded status

---

### **REMOVED Components (From Original Plan)**

| Component | Reason for Removal |
|-----------|-------------------|
| `FleetOverviewCards` | No fleet monitoring - only 1 machine at a time |
| `MachineGrid` | No grid needed - dropdown selector instead |
| Batch prediction endpoints | No batch predictions - single machine only |
| Fleet statistics API | No fleet stats needed |
| Auto-refresh all 26 machines | Only refresh selected machine |

---

## 🔌 REVISED API SPECIFICATION

### **Backend Service: MLManager** (`frontend/server/services/ml_manager.py`)

**Purpose:** Wrapper around `IntegratedPredictionSystem` for FastAPI integration

**Implementation:**
```python
from LLM.api.ml_integration import IntegratedPredictionSystem

class MLManager:
    """FastAPI-compatible wrapper for ML/LLM predictions"""
    
    def __init__(self):
        self.integrated_system = IntegratedPredictionSystem()
    
    async def predict_for_machine(
        self, 
        machine_id: str, 
        sensor_data: Dict[str, float],
        model_types: List[str] = ['classification', 'regression']
    ) -> Dict:
        """
        Run predictions for a single machine
        
        Args:
            machine_id: Machine identifier
            sensor_data: Current sensor readings
            model_types: Which models to run (default: classification + RUL)
        
        Returns:
            Dict with predictions and explanations
        """
        return self.integrated_system.predict_with_explanation(
            machine_id=machine_id,
            sensor_data=sensor_data,
            model_type='all' if 'all' in model_types else ','.join(model_types)
        )
    
    async def get_machine_status(self, machine_id: str) -> Dict:
        """
        Get current status of a machine (is it running? latest sensors?)
        
        Returns:
            {
                "machine_id": "motor_siemens_1la7_001",
                "is_running": True,
                "latest_sensors": {...},
                "last_update": "2025-11-27T10:30:00Z"
            }
        """
        # Implementation: Check if recent sensor data exists
        # This would check database or cache for latest readings
        pass
    
    async def get_prediction_history(
        self, 
        machine_id: str, 
        limit: int = 100
    ) -> List[Dict]:
        """
        Get historical predictions for a machine
        
        Returns:
            List of predictions sorted by timestamp (newest first)
        """
        # Implementation: Query PostgreSQL for stored predictions
        pass
```

---

### **API Routes: ml.py** (`frontend/server/api/routes/ml.py`)

**File:** `frontend/server/api/routes/ml.py`

**Endpoints (6 TOTAL - NOT batch-focused):**

#### **1. List All Machines**
```python
GET /api/ml/machines
Response:
{
  "machines": [
    {
      "machine_id": "motor_siemens_1la7_001",
      "display_name": "Motor Siemens 1LA7 001",
      "category": "motor",
      "sensor_count": 22,
      "has_classification_model": True,
      "has_regression_model": True,
      "has_anomaly_model": False,
      "has_timeseries_model": False
    },
    ...
  ],
  "total": 26
}
```

#### **2. Get Machine Status**
```python
GET /api/ml/machines/{machine_id}/status
Response:
{
  "machine_id": "motor_siemens_1la7_001",
  "is_running": True,
  "latest_sensors": {
    "bearing_de_temp_C": 65.2,
    "vibration_mm_s": 3.4,
    "current_A": 12.5,
    ...
  },
  "last_update": "2025-11-27T10:45:00Z"
}
```

#### **3. Run Classification Prediction**
```python
POST /api/ml/predict/classification
Request:
{
  "machine_id": "motor_siemens_1la7_001",
  "sensor_data": {
    "bearing_de_temp_C": 65.2,
    "vibration_mm_s": 3.4,
    "current_A": 12.5,
    ...
  }
}

Response:
{
  "machine_id": "motor_siemens_1la7_001",
  "prediction": {
    "failure_type": "normal",
    "confidence": 0.95,
    "failure_probability": 0.05,
    "all_probabilities": {
      "normal": 0.95,
      "bearing_wear": 0.03,
      "overheating": 0.01,
      "electrical_fault": 0.01
    }
  },
  "explanation": {
    "summary": "Machine is operating normally with 95% confidence...",
    "risk_factors": [],
    "recommendations": ["Continue normal operation", "Monitor bearing temperature"]
  },
  "timestamp": "2025-11-27T10:45:23Z"
}
```

#### **4. Run RUL Prediction**
```python
POST /api/ml/predict/rul
Request: (same as classification)

Response:
{
  "machine_id": "motor_siemens_1la7_001",
  "prediction": {
    "rul_hours": 156.3,
    "rul_days": 6.51,
    "urgency": "medium",
    "maintenance_window": "within 3 days",
    "critical_sensors": [...]
  },
  "explanation": {...},
  "timestamp": "2025-11-27T10:45:23Z"
}
```

#### **5. Get Prediction History**
```python
GET /api/ml/machines/{machine_id}/history?limit=100&model_type=classification
Response:
{
  "machine_id": "motor_siemens_1la7_001",
  "predictions": [
    {
      "timestamp": "2025-11-27T10:45:00Z",
      "failure_type": "normal",
      "confidence": 0.95,
      "rul_hours": 156.3
    },
    ...
  ],
  "total": 100
}
```

#### **6. Health Check**
```python
GET /api/ml/health
Response:
{
  "status": "healthy",
  "models_loaded": {
    "classification": 10,
    "regression": 8,
    "anomaly": 5,
    "timeseries": 3
  },
  "llm_status": "operational",
  "gpu_available": True
}
```

---

### **WebSocket Endpoint: Real-Time Sensor Stream**

**File:** `frontend/server/api/routes/websocket.py`

**Endpoint:**
```python
WS /ws/ml/sensors/{machine_id}

# Client connects:
ws = new WebSocket('ws://localhost:8000/ws/ml/sensors/motor_siemens_1la7_001')

# Server sends every 5 seconds:
{
  "machine_id": "motor_siemens_1la7_001",
  "timestamp": "2025-11-27T10:45:23Z",
  "sensors": {
    "bearing_de_temp_C": 65.2,
    "vibration_mm_s": 3.4,
    ...
  }
}
```

**Implementation:**
```python
@router.websocket("/ws/ml/sensors/{machine_id}")
async def sensor_stream(websocket: WebSocket, machine_id: str):
    await websocket.accept()
    
    try:
        while True:
            # Fetch latest sensor data (from database or mock)
            sensor_data = await ml_manager.get_latest_sensors(machine_id)
            
            await websocket.send_json({
                "machine_id": machine_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "sensors": sensor_data
            })
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except WebSocketDisconnect:
        print(f"Client disconnected from {machine_id} sensor stream")
```

---

## 📋 REVISED IMPLEMENTATION PLAN

### **Phase 3.7.3: ML Dashboard (CORRECTED) - 4 Days**

#### **Day 19.1: MLManager Service (Backend)**

**Tasks:**
- [ ] Create `frontend/server/services/ml_manager.py`
- [ ] Wrap `IntegratedPredictionSystem` from LLM module
- [ ] Implement `predict_for_machine()` method
- [ ] Implement `get_machine_status()` method
- [ ] Implement `get_prediction_history()` method
- [ ] Add error handling and logging

**Files:**
- `frontend/server/services/ml_manager.py` (300 lines)

**Expected Output:**
- ✅ MLManager class operational
- ✅ Can load models on demand
- ✅ Returns predictions in correct format

---

#### **Day 19.2: ML API Routes (Backend)**

**Tasks:**
- [ ] Create `frontend/server/api/routes/ml.py`
- [ ] Implement 6 API endpoints:
  1. `GET /api/ml/machines` - List all 26 machines
  2. `GET /api/ml/machines/{id}/status` - Machine status
  3. `POST /api/ml/predict/classification` - Classification prediction
  4. `POST /api/ml/predict/rul` - RUL prediction
  5. `GET /api/ml/machines/{id}/history` - Prediction history
  6. `GET /api/ml/health` - Service health
- [ ] Add request validation (Pydantic models)
- [ ] Add response schemas
- [ ] Test all endpoints with Swagger UI

**Files:**
- `frontend/server/api/routes/ml.py` (600 lines)
- `frontend/server/api/schemas/ml.py` (200 lines)

**Expected Output:**
- ✅ All 6 endpoints operational
- ✅ Swagger docs updated
- ✅ Predictions working for test machine

---

#### **Day 20.1: Frontend Components (Part 1)**

**Tasks:**
- [ ] Create `MachineSelector` component
- [ ] Create `SensorDashboard` component
- [ ] Create `PredictionCard` component
- [ ] Set up React Query for data fetching
- [ ] Add WebSocket hook for real-time sensors

**Files:**
- `frontend/client/src/modules/ml/components/MachineSelector.tsx` (150 lines)
- `frontend/client/src/modules/ml/components/SensorDashboard.tsx` (200 lines)
- `frontend/client/src/modules/ml/components/PredictionCard.tsx` (180 lines)
- `frontend/client/src/hooks/useWebSocket.ts` (100 lines)

**Expected Output:**
- ✅ MachineSelector dropdown working
- ✅ Sensors display in real-time
- ✅ Predictions display correctly

---

#### **Day 20.2: Frontend Components (Part 2)**

**Tasks:**
- [ ] Create `SensorCharts` component (time-series graphs)
- [ ] Create `PredictionHistory` component (table)
- [ ] Create `LLMExplanationModal` component
- [ ] Integrate Recharts library for line charts

**Files:**
- `frontend/client/src/modules/ml/components/SensorCharts.tsx` (250 lines)
- `frontend/client/src/modules/ml/components/PredictionHistory.tsx` (180 lines)
- `frontend/client/src/modules/ml/components/LLMExplanationModal.tsx` (200 lines)

**Expected Output:**
- ✅ Charts display sensor trends
- ✅ History table paginated
- ✅ Explanation modal functional

---

#### **Day 21: ML Dashboard Page Assembly**

**Tasks:**
- [ ] Create `MLDashboardPage.tsx`
- [ ] Integrate all components
- [ ] Add state management (React Query)
- [ ] Implement WebSocket connection
- [ ] Add auto-refresh logic
- [ ] Add error boundaries
- [ ] Test with multiple machines

**Files:**
- `frontend/client/src/pages/MLDashboardPage.tsx` (300 lines)

**Expected Output:**
- ✅ Complete dashboard functional
- ✅ Machine selection working
- ✅ Real-time updates working
- ✅ Predictions accurate

---

## 🔍 DATA FLOW (CORRECTED)

### **Single-Machine Monitoring Workflow**

```
User Action: Select machine from dropdown
          ↓
┌─────────────────────────────────────────────────┐
│ 1. User selects "Motor Siemens 1LA7 001"       │
└─────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────┐
│ 2. Frontend establishes WebSocket connection   │
│    WS /ws/ml/sensors/motor_siemens_1la7_001    │
└─────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────┐
│ 3. Backend sends sensor data every 5 seconds   │
│    {timestamp, bearing_temp, vibration, ...}    │
└─────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────┐
│ 4. Frontend displays sensors in real-time      │
│    SensorDashboard component updates           │
└─────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────┐
│ 5. User clicks "Run Prediction" button         │
└─────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────┐
│ 6. POST /api/ml/predict/classification         │
│    {machine_id, sensor_data}                    │
└─────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────┐
│ 7. MLManager.predict_for_machine()             │
│    → IntegratedPredictionSystem                 │
│    → ClassificationPredictor.predict()          │
│    → RULPredictor.predict()                     │
│    → MLExplainer.explain()                      │
└─────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────┐
│ 8. Response with prediction + explanation      │
│    {failure_type, confidence, rul, explanation} │
└─────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────┐
│ 9. Frontend displays results                   │
│    PredictionCard component updates            │
└─────────────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────────────┐
│ 10. User clicks "Get AI Explanation"           │
│     LLMExplanationModal opens                   │
└─────────────────────────────────────────────────┘
```

---

## 🚨 CRITICAL PRESERVATION RULES

### **1. Sensor Configurations (DO NOT MODIFY)**

**Location:** `GAN/metadata/{machine_id}_metadata.json`

**Example:**
```json
{
  "machine_id": "motor_siemens_1la7_001",
  "columns": {
    "bearing_de_temp_C": {"mean": 55.0, "std": 5.0, "min": 40.0, "max": 70.0},
    "bearing_nde_temp_C": {"mean": 52.0, "std": 4.5, "min": 38.0, "max": 68.0},
    ...
  }
}
```

**RULE:** These configurations are **research-based** - never modify without user approval

---

### **2. Model Paths (DO NOT CHANGE)**

**Location:** `ml_models/models/{model_type}/{machine_id}/`

**RULE:** Do not rename or move model directories

---

### **3. AutoGluon Models (DO NOT RETRAIN)**

**RULE:** Existing models are already trained - use them as-is

---

## 📊 SUMMARY OF CHANGES

### **Components REMOVED**
| Component | Lines | Reason |
|-----------|-------|--------|
| `FleetOverviewCards.tsx` | ~150 | No fleet monitoring |
| `MachineGrid.tsx` | ~250 | No grid needed |
| Batch prediction API | ~200 | No batch predictions |

### **Components ADDED**
| Component | Lines | Purpose |
|-----------|-------|---------|
| `MachineSelector.tsx` | ~150 | Dropdown to select ONE machine |
| `SensorDashboard.tsx` | ~200 | Real-time sensor display |
| `SensorCharts.tsx` | ~250 | Time-series line charts |
| `PredictionHistory.tsx` | ~180 | Historical predictions table |

### **API Changes**
| Original | Revised | Reason |
|----------|---------|--------|
| `POST /api/ml/predict/batch` | ❌ REMOVED | No batch predictions |
| `GET /api/ml/fleet/stats` | ❌ REMOVED | No fleet stats |
| `WS /ws/ml/fleet` | ❌ REMOVED | No fleet monitoring |
| — | ✅ `GET /api/ml/machines` | List all machines |
| — | ✅ `GET /api/ml/machines/{id}/status` | Machine status |
| — | ✅ `POST /api/ml/predict/classification` | Single prediction |
| — | ✅ `WS /ws/ml/sensors/{id}` | Real-time sensors |

---

## ✅ NEXT STEPS

1. **User Review:** Approve corrected architecture
2. **Implementation:** Follow revised 4-day plan (Days 19-21)
3. **Testing:** Verify single-machine workflow
4. **Phase 3.7.4:** LLM integration (after ML dashboard complete)
5. **Phase 3.7.5:** GAN frontend + UI polish

---

## 📝 OPEN QUESTIONS

1. **Sensor Data Source:** Where do real-time sensor readings come from?
   - Option A: Simulated/mock data for MVP
   - Option B: Stored historical data from GAN
   - Option C: Real SCADA/IoT integration (future)

2. **Prediction Frequency:** How often should predictions run?
   - Recommendation: On-demand (user clicks button)
   - Alternative: Auto-predict every 30 seconds

3. **Prediction Storage:** Should predictions be saved to database?
   - Recommendation: Yes (for history table)
   - Alternative: Redis cache only (ephemeral)

---

**Document Complete** ✅  
**Author:** Phase 3.7.3 Architecture Revision  
**Date:** November 27, 2025  
**Status:** 🔴 AWAITING USER APPROVAL
