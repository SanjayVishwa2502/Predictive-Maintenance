# PHASE 3.7.3: ML DASHBOARD IMPLEMENTATION GUIDE
**Single-Machine Monitoring Architecture**  
**Start Date:** December 15, 2025  
**Duration:** 4 Days (Day 17-20)  
**Status:** 🟢 READY TO START

---

## 🎯 QUICK START

### What We're Building
A real-time machine health monitoring dashboard that:
- Monitors **ONE machine at a time** (user selects from dropdown)
- Streams real-time sensor data via WebSocket
- Runs ML predictions on-demand (classification + RUL)
- Displays time-series sensor charts (last 10 minutes)
- Shows prediction history in table format
- Provides LLM explanations for predictions

### Key Architecture Points
- ✅ **Single machine focus** (not fleet-wide)
- ✅ Uses existing `IntegratedPredictionSystem` (LLM/api/ml_integration.py)
- ✅ 10 trained models available (AutoGluon classification)
- ✅ 26 machines total, 1-22 sensors each
- ✅ GAN backend complete (no changes needed)

---

## 📅 IMPLEMENTATION SCHEDULE

### **Day 17: Frontend Components (Part 1)** - December 16, 2025

#### Morning (Day 17.1): MachineSelector Component
**File:** `frontend/client/src/modules/ml/components/MachineSelector.tsx`
**Duration:** 3 hours

**Tasks:**
1. Create component file
2. Implement Material-UI Autocomplete
3. Add category grouping (Motors, Pumps, CNCs, etc.)
4. Display sensor count + model availability badges
5. Add search functionality
6. Test with all 26 machines

**API Call:**
```typescript
const { data: machines } = useQuery('machines', async () => {
  const res = await fetch('/api/ml/machines');
  return await res.json();
});
```

**Acceptance Criteria:**
- ✅ Dropdown displays all 26 machines
- ✅ Search filters by name/category
- ✅ Badges show model availability
- ✅ Selection triggers callback

---

#### Afternoon (Day 17.2): SensorDashboard Component
**File:** `frontend/client/src/modules/ml/components/SensorDashboard.tsx`
**Duration:** 4 hours

**Tasks:**
1. Create component file
2. Build responsive grid (4 cols → 2 → 1)
3. Design SensorCard subcomponent
4. Add color-coded thresholds (green/yellow/red)
5. Implement live update indicator
6. Connect to WebSocket data

**Props:**
```typescript
interface SensorDashboardProps {
  machineId: string;
  sensorData: Record<string, number>;
  lastUpdated: Date;
  loading?: boolean;
}
```

**Acceptance Criteria:**
- ✅ All sensors display in grid
- ✅ Colors match thresholds
- ✅ Updates in real-time
- ✅ Responsive on mobile

---

### **Day 17 Continued: Frontend Components (Part 2)**

#### Evening (Day 17.3): PredictionCard Component
**File:** `frontend/client/src/modules/ml/components/PredictionCard.tsx`
**Duration:** 3 hours

**Tasks:**
1. Create component file
2. Design card layout (health status + RUL)
3. Add probability bars for all failure types
4. Implement "Run Prediction" button
5. Add "AI Explanation" button
6. Connect to prediction API

**Layout:**
```
┌─────────────────────────────────────┐
│   MACHINE HEALTH PREDICTION         │
│ ┌────────────┐  ┌────────────────┐  │
│ │ 🟢 Healthy │  │ ⏱️ 156 hours   │  │
│ │ Conf: 95%  │  │ (6.5 days)     │  │
│ └────────────┘  └────────────────┘  │
│                                     │
│ Failure Probabilities:              │
│ Normal:       ████████████████ 95% │
│ Bearing Wear: ██░░░░░░░░░░░░░ 3%  │
│ [🤖 Get AI Explanation] [View Hist]│
└─────────────────────────────────────┘
```

**Acceptance Criteria:**
- ✅ Displays prediction results
- ✅ Status colors accurate
- ✅ Buttons functional
- ✅ Loading states work

---

#### Evening (Day 17.4): SensorCharts Component
**File:** `frontend/client/src/modules/ml/components/SensorCharts.tsx`
**Duration:** 3 hours

**Tasks:**
1. Install Recharts: `npm install recharts`
2. Create component file
3. Implement multi-line chart
4. Add sensor selection dropdown (max 5 sensors)
5. Implement rolling 10-minute window
6. Add zoom/pan controls
7. Export to CSV functionality

**Libraries:**
```bash
npm install recharts date-fns
```

**Acceptance Criteria:**
- ✅ Chart displays time-series data
- ✅ Multiple sensors can be plotted
- ✅ Auto-scrolls as new data arrives
- ✅ Zoom/pan functional
- ✅ Export generates CSV

---

### **Day 18: Completion + Integration** - December 17, 2025

#### Morning (Day 18.1): PredictionHistory Component
**File:** `frontend/client/src/modules/ml/components/PredictionHistory.tsx`
**Duration:** 3 hours

**Tasks:**
1. Create component file
2. Implement Material-UI DataGrid
3. Add pagination (10 rows per page)
4. Implement sort by column
5. Add date filter
6. Export to CSV

**API Call:**
```typescript
const { data } = useQuery(['history', machineId], async () => {
  const res = await fetch(`/api/ml/machines/${machineId}/history?limit=100`);
  return await res.json();
});
```

**Acceptance Criteria:**
- ✅ Table displays 100 predictions
- ✅ Pagination works
- ✅ Sorting functional
- ✅ Export generates CSV

---

#### Afternoon (Day 18.2): MLDashboardPage Assembly
**File:** `frontend/client/src/pages/MLDashboardPage.tsx`
**Duration:** 4 hours

**Tasks:**
1. Create page file
2. Set up React Query for data fetching
3. Implement WebSocket hook
4. Connect all components
5. Add error boundaries
6. Implement empty state (no machine selected)
7. Add LLMExplanationModal integration

**WebSocket Hook:**
```typescript
const useWebSocket = (url: string | null) => {
  const [lastMessage, setLastMessage] = useState(null);
  const [connected, setConnected] = useState(false);
  
  useEffect(() => {
    if (!url) return;
    
    const ws = new WebSocket(url);
    ws.onopen = () => setConnected(true);
    ws.onmessage = (event) => setLastMessage(JSON.parse(event.data));
    ws.onerror = () => setConnected(false);
    
    return () => ws.close();
  }, [url]);
  
  return { connected, lastMessage };
};
```

**Acceptance Criteria:**
- ✅ All components integrated
- ✅ WebSocket connection stable
- ✅ Machine selection works
- ✅ Predictions run successfully
- ✅ Empty state displays
- ✅ Error handling works

---

### **Day 19: Backend Implementation** - December 18, 2025

#### Morning (Day 19.1): MLManager Service
**File:** `frontend/server/services/ml_manager.py`
**Duration:** 3 hours

**Tasks:**
1. Create service file
2. Import `IntegratedPredictionSystem` from LLM module
3. Implement `list_machines()` method
4. Implement `get_machine_status()` method
5. Implement `predict_classification()` method
6. Implement `predict_rul()` method
7. Add error handling and logging

**Key Code:**
```python
import sys
from pathlib import Path

# Add LLM module to path
sys.path.append(str(Path(__file__).parents[2] / "LLM"))
from api.ml_integration import IntegratedPredictionSystem

class MLManager:
    def __init__(self):
        self.integrated_system = IntegratedPredictionSystem()
    
    async def predict_classification(self, machine_id: str, sensor_data: dict):
        result = self.integrated_system.predict_with_explanation(
            machine_id=machine_id,
            sensor_data=sensor_data,
            model_type='classification'
        )
        return result['classification']
```

**Acceptance Criteria:**
- ✅ Can load IntegratedPredictionSystem
- ✅ Lists all 26 machines correctly
- ✅ Predictions work for test machines
- ✅ Error handling catches exceptions

---

#### Afternoon (Day 19.2): ML API Routes
**File:** `frontend/server/api/routes/ml.py`
**Duration:** 4 hours

**Tasks:**
1. Create routes file
2. Implement 6 REST endpoints:
   - GET `/api/ml/machines`
   - GET `/api/ml/machines/{id}/status`
   - POST `/api/ml/predict/classification`
   - POST `/api/ml/predict/rul`
   - GET `/api/ml/machines/{id}/history`
   - GET `/api/ml/health`
3. Implement WebSocket endpoint: `/ws/ml/sensors/{id}`
4. Add request/response schemas (Pydantic)
5. Test all endpoints in Swagger UI
6. Create HTML test client for WebSocket

**WebSocket Implementation:**
```python
@router.websocket("/ws/ml/sensors/{machine_id}")
async def sensor_stream(websocket: WebSocket, machine_id: str):
    await websocket.accept()
    
    try:
        while True:
            status = await ml_manager.get_machine_status(machine_id)
            await websocket.send_json({
                "machine_id": machine_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "sensors": status["latest_sensors"]
            })
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        print(f"Client disconnected from {machine_id}")
```

**Acceptance Criteria:**
- ✅ All 6 REST endpoints operational
- ✅ WebSocket streams data every 5 seconds
- ✅ Swagger docs display correctly
- ✅ Test client can connect to WebSocket
- ✅ Predictions return correct format

**Test Commands:**
```bash
# Start backend server
cd frontend/server
uvicorn main:app --reload

# Test REST endpoint
curl http://localhost:8000/api/ml/machines

# Test prediction
curl -X POST http://localhost:8000/api/ml/predict/classification \
  -H "Content-Type: application/json" \
  -d '{"machine_id": "motor_siemens_1la7_001", "sensor_data": {...}}'
```

---

### **Day 20: Testing + Polish** - December 19, 2025

#### Morning (Day 20.1): End-to-End Testing
**Duration:** 3 hours

**Test Scenarios:**
1. ✅ Select machine from dropdown
2. ✅ WebSocket connects and streams sensors
3. ✅ Sensors display in real-time
4. ✅ Click "Run Prediction" → Results appear
5. ✅ Click "AI Explanation" → Modal opens
6. ✅ View prediction history → Table loads
7. ✅ Export history to CSV → File downloads
8. ✅ Switch machines → Data updates
9. ✅ Disconnect WebSocket → Reconnects automatically
10. ✅ Mobile responsive → All layouts work

**Bug Tracking:**
- Document all bugs in GitHub Issues
- Fix critical bugs immediately
- Defer minor bugs to Phase 3.7.4

---

#### Afternoon (Day 20.2): Performance Optimization
**Duration:** 3 hours

**Tasks:**
1. Add React.memo() to sensor cards
2. Debounce machine selector search
3. Lazy load Recharts library
4. Add service worker for offline support
5. Optimize WebSocket message size
6. Add loading skeletons
7. Implement error boundaries

**Performance Targets:**
- Initial load: < 2 seconds
- WebSocket latency: < 50ms
- Prediction response: < 3 seconds
- Chart rendering: < 500ms

---

## 🔧 TECHNICAL SETUP

### Prerequisites
```bash
# Check versions
node --version  # Should be 18+
python --version  # Should be 3.11+

# Activate Python environment
cd "C:/Projects/Predictive Maintenance"
.\venv\Scripts\Activate.ps1

# Verify services running
# PostgreSQL: Port 5433
# Redis: Port 6379
# FastAPI: http://localhost:8000
```

### Frontend Setup
```bash
cd frontend/client

# Install dependencies (if not done)
npm install

# Install new libraries
npm install recharts date-fns @mui/x-data-grid

# Start development server
npm run dev
# Opens at http://localhost:5173
```

### Backend Setup
```bash
cd frontend/server

# Verify dependencies
pip list | grep -E "fastapi|uvicorn|websockets"

# Start server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Test endpoints
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/api/ml/machines
```

---

## 📊 DATA FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────┐
│                    USER WORKFLOW                            │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. User opens ML Dashboard Page                            │
│    → Fetches all 26 machines from GET /api/ml/machines     │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. User selects "Motor Siemens 1LA7 001" from dropdown     │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Frontend establishes WebSocket connection               │
│    ws://localhost:8000/ws/ml/sensors/motor_siemens_1la7_001│
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Backend sends sensor data every 5 seconds               │
│    {timestamp, bearing_temp_C, vibration_mm_s, current_A...}│
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. SensorDashboard displays real-time values               │
│    SensorCharts plots time-series (last 10 minutes)        │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. User clicks "Run Prediction" button                     │
│    → POST /api/ml/predict/classification                   │
│    → MLManager → IntegratedPredictionSystem                 │
│    → ClassificationPredictor.predict()                      │
│    → RULPredictor.predict()                                 │
│    → MLExplainer.explain()                                  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. PredictionCard displays results                         │
│    Status: 🟢 Healthy (95% confidence)                     │
│    RUL: 156 hours (6.5 days)                               │
│    Urgency: Medium                                          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. User clicks "Get AI Explanation"                        │
│    → LLMExplanationModal opens                              │
│    → Displays LLM-generated explanation                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🐛 TROUBLESHOOTING

### Issue: WebSocket won't connect
**Solution:**
1. Check FastAPI server is running: `http://localhost:8000/docs`
2. Verify WebSocket endpoint exists: `/ws/ml/sensors/{machine_id}`
3. Check browser console for errors
4. Test with HTML WebSocket client first

### Issue: Predictions fail with 500 error
**Solution:**
1. Check machine has trained model: `ml_models/models/classification/{machine_id}/`
2. Verify sensor data format matches model expectations
3. Check FastAPI logs for detailed error
4. Test with known working machine: `motor_siemens_1la7_001`

### Issue: Sensor charts not updating
**Solution:**
1. Verify WebSocket messages arriving (browser DevTools → Network → WS)
2. Check `sensorHistory` state updating in React
3. Ensure Recharts `data` prop receiving new values
4. Add console.log in useEffect to debug

### Issue: IntegratedPredictionSystem import fails
**Solution:**
1. Verify LLM module path added: `sys.path.append(str(Path(__file__).parents[2] / "LLM"))`
2. Check LLM/api/ml_integration.py exists
3. Verify Python virtual environment activated
4. Test import manually in Python REPL

---

## ✅ ACCEPTANCE CHECKLIST

### Day 17 Completion Criteria
- [ ] MachineSelector displays all 26 machines
- [ ] SensorDashboard shows real-time sensors
- [ ] PredictionCard displays results
- [ ] SensorCharts plots time-series data

### Day 18 Completion Criteria
- [ ] PredictionHistory table functional
- [ ] MLDashboardPage integrates all components
- [ ] WebSocket connection stable
- [ ] Empty state displays correctly

### Day 19 Completion Criteria
- [ ] MLManager service operational
- [ ] All 6 REST endpoints working
- [ ] WebSocket endpoint streaming
- [ ] Predictions return correct format
- [ ] Swagger UI displays correctly

### Day 20 Completion Criteria
- [ ] End-to-end tests passing
- [ ] Performance targets met
- [ ] Mobile responsive
- [ ] No critical bugs

### Phase 3.7.3 COMPLETE
- [ ] All components functional
- [ ] Backend integrated with IntegratedPredictionSystem
- [ ] Real-time monitoring working
- [ ] Predictions accurate
- [ ] UI polished and responsive

---

## 📝 NOTES & REMINDERS

### Configuration Files to Preserve
- **DO NOT MODIFY:** `GAN/metadata/{machine_id}_metadata.json` (research-based)
- **DO NOT MODIFY:** `ml_models/config/model_config.py` (sensor configurations)
- **DO NOT RENAME:** Model directories in `ml_models/models/`

### Data Sources
- **Sensor data:** Mock generated from metadata (for MVP)
- **Predictions:** Real ML models (AutoGluon)
- **Explanations:** Real LLM (GPU-accelerated)

### Future Enhancements (Phase 3.7.4+)
- Real SCADA/IoT sensor integration
- Database storage for predictions
- Email/SMS alerts for critical predictions
- Batch prediction mode (optional)
- Multi-machine comparison view

---

## 🚀 QUICK REFERENCE

### API Endpoints
```
GET    /api/ml/machines                    # List all machines
GET    /api/ml/machines/{id}/status        # Machine status
POST   /api/ml/predict/classification      # Run prediction
POST   /api/ml/predict/rul                 # RUL prediction
GET    /api/ml/machines/{id}/history       # Prediction history
GET    /api/ml/health                      # Service health
WS     /ws/ml/sensors/{id}                 # Real-time sensors
```

### Key Files
```
frontend/client/src/
├── modules/ml/components/
│   ├── MachineSelector.tsx          (Day 17.1)
│   ├── SensorDashboard.tsx          (Day 17.2)
│   ├── PredictionCard.tsx           (Day 17.3)
│   ├── SensorCharts.tsx             (Day 17.4)
│   └── PredictionHistory.tsx        (Day 18.1)
├── pages/
│   └── MLDashboardPage.tsx          (Day 18.2)

frontend/server/
├── services/
│   └── ml_manager.py                (Day 19.1)
└── api/routes/
    └── ml.py                        (Day 19.2)
```

### Testing URLs
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- WebSocket Test: ws://localhost:8000/ws/ml/sensors/motor_siemens_1la7_001

---

**Document Complete** ✅  
**Ready to Start:** December 15, 2025  
**Estimated Completion:** December 19, 2025  
**Contact:** See main project README for support
