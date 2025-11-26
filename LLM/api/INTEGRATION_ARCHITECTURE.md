# ML → LLM Integration Architecture
## Phase 3.5: Integration with ML Models

**Version:** 1.0  
**Status:** Phase 3.5.0 Complete | Phase 3.5.1 Ready  
**Target Deployment:** Raspberry Pi 5 (16GB RAM)

---

## 1. System Overview

### 1.1 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PREDICTIVE MAINTENANCE SYSTEM                   │
└─────────────────────────────────────────────────────────────────────┘

┌────────────────┐
│ Sensor Data    │ ──► Temperature, Vibration, Current, Voltage...
│ (Real-time)    │     22 sensor columns per machine
└────────┬───────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────┐
│                     ML INFERENCE LAYER (Phase 3.5.0)               │
├────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────┐ │
│  │ Classification  │  │ Anomaly         │  │ RUL Regression     │ │
│  │ Predictor       │  │ Predictor       │  │ Predictor          │ │
│  │                 │  │                 │  │                    │ │
│  │ • AutoGluon     │  │ • IsolationFor. │  │ • AutoGluon        │ │
│  │ • 4 failure     │  │ • OneClassSVM   │  │ • Hours/Days       │ │
│  │   types         │  │ • LOF           │  │ • Urgency levels   │ │
│  │ • Confidence    │  │ • Z-Score       │  │ • Maintenance      │ │
│  │   scores        │  │ • Severity      │  │   windows          │ │
│  └────────┬────────┘  └────────┬────────┘  └─────────┬──────────┘ │
│           │                    │                      │            │
│           └────────────────────┼──────────────────────┘            │
│                                │                                   │
│                        ┌───────▼──────────┐                        │
│                        │ Time-Series      │                        │
│                        │ Forecasting      │                        │
│                        │                  │                        │
│                        │ • 24h forecast   │                        │
│                        │ • Trend analysis │                        │
│                        │ • Optimal maint. │                        │
│                        │   windows        │                        │
│                        └───────┬──────────┘                        │
└────────────────────────────────┼───────────────────────────────────┘
                                 │
                     ┌───────────▼──────────┐
                     │ JSON Predictions     │
                     │ • machine_id         │
                     │ • timestamp          │
                     │ • prediction         │
                     │ • sensor_readings    │
                     │ • confidence         │
                     └───────┬──────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────┐
│                  MLExplainer API (Phase 3.5.1)                     │
├────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ UnifiedInferenceService                                    │   │
│  │  • ML Model Manager (lazy loading, caching)               │   │
│  │  • RAG Retriever (FAISS, <150ms)                          │   │
│  │  • Prompt Formatter (template selection)                  │   │
│  │  • LLM Generator (Llama 3.1 8B)                           │   │
│  └────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────┬────────────────────────────┘
                                        │
                                        ▼
                        ┌───────────────────────────┐
                        │ Human-Readable            │
                        │ Explanation               │
                        │                           │
                        │ • Technical analysis      │
                        │ • Root cause              │
                        │ • Recommendations         │
                        │ • Confidence + context    │
                        └───────────────────────────┘
```

---

## 2. Component Architecture

### 2.1 ML Inference Layer (Phase 3.5.0)

**Location:** `ml_models/scripts/inference/`

#### Classification Predictor
- **File:** `predict_classification.py`
- **Model:** AutoGluon TabularPredictor
- **Input:** 22 sensor readings (temperature, vibration, current, voltage, etc.)
- **Output:**
  ```json
  {
    "failure_type": "bearing_wear | overheating | electrical_fault | normal",
    "failure_probability": 0.87,
    "confidence": 0.92,
    "all_probabilities": {
      "normal": 0.13,
      "bearing_wear": 0.87,
      "overheating": 0.05,
      "electrical_fault": 0.03
    }
  }
  ```
- **Performance:** <1 second inference
- **Models:** 10 trained models (1 per priority machine)

#### Anomaly Detector
- **File:** `predict_anomaly.py`
- **Models:** Ensemble of 4 detectors
  - IsolationForest (outlier detection)
  - OneClassSVM (boundary learning)
  - LocalOutlierFactor (density-based)
  - Z-Score (statistical threshold)
- **Input:** 22 sensor readings
- **Output:**
  ```json
  {
    "is_anomaly": true,
    "anomaly_score": 0.78,
    "severity": "critical | high | medium | low | normal",
    "detector_scores": {
      "isolation_forest": 0.82,
      "one_class_svm": 0.74,
      "lof": 0.80,
      "zscore": 0.76
    },
    "abnormal_sensors": [
      "temperature: 92°C (high)",
      "vibration: 15.2 mm/s (elevated)"
    ]
  }
  ```
- **Performance:** <0.5 seconds inference
- **Models:** 10 trained ensembles

#### RUL Predictor
- **File:** `predict_rul.py`
- **Model:** AutoGluon TabularPredictor (regression)
- **Input:** 22 sensor readings
- **Output:**
  ```json
  {
    "rul_hours": 145.5,
    "rul_days": 6.1,
    "estimated_failure_date": "2025-12-01T16:00:00Z",
    "urgency": "critical | high | medium | low",
    "confidence": 0.85,
    "maintenance_window": "within 3 days",
    "critical_sensors": [
      "temperature: 82°C (elevated)",
      "vibration: 9.2 mm/s (elevated)"
    ]
  }
  ```
- **Urgency Levels:**
  - Critical: <24 hours (immediate action)
  - High: <72 hours (schedule within 24h)
  - Medium: <168 hours (within 3 days)
  - Low: >168 hours (within 1 week)
- **Performance:** <1 second inference
- **Models:** 10 trained models

#### Time-Series Forecaster
- **File:** `predict_timeseries.py`
- **Model:** AutoGluon TimeSeriesPredictor
- **Input:** Historical data (168 timesteps = 1 week hourly)
- **Output:**
  ```json
  {
    "forecast_horizon_hours": 24,
    "confidence": 0.85,
    "forecast_summary": "Hour 0-6: stable, Hour 6-12: vibration increasing...",
    "concerning_trends": [
      "Vibration increasing 45% in next 12 hours",
      "Temperature steady at elevated 78°C"
    ],
    "maintenance_window": "Optimal window: Hour 12-18 (lowest sensor activity)",
    "detailed_forecast": [
      {"hour": 1, "timestamp": "...", "sensors": {...}},
      ...
    ]
  }
  ```
- **Performance:** <2 seconds inference
- **Models:** 11 trained models

---

### 2.2 MLExplainer API (Phase 3.5.1)

**Location:** `LLM/api/`

#### UnifiedInferenceService

**Purpose:** Orchestrates ML inference + RAG retrieval + LLM generation

**Core Methods:**

```python
class UnifiedInferenceService:
    def __init__(self):
        """Initialize LLM, RAG, model cache"""
        
    def load_ml_model(self, machine_id: str, model_type: str):
        """Lazy load ML models with caching"""
        
    def run_ml_inference(self, machine_id: str, sensor_data: Dict, model_type: str) -> Dict:
        """Execute ML prediction"""
        
    def retrieve_context(self, query: str, machine_id: str, top_k: int = 5) -> List[Dict]:
        """RAG retrieval from knowledge base"""
        
    def format_prompt(self, model_type: str, ml_prediction: Dict, rag_context: List[Dict]) -> str:
        """Fill prompt template"""
        
    def generate_explanation(self, prompt: str) -> str:
        """LLM generation (Llama 3.1 8B)"""
        
    def explain(self, machine_id: str, sensor_data: Dict, model_type: str) -> Dict:
        """End-to-end: ML → RAG → LLM → Explanation"""
```

---

## 3. API Design

### 3.1 Unified Endpoint

**POST /api/v1/explain**

**Request:**
```json
{
  "machine_id": "motor_siemens_1la7_001",
  "model_type": "classification | anomaly | rul | timeseries",
  "sensor_data": {
    "bearing_de_temp_C": 78.5,
    "bearing_nde_temp_C": 72.3,
    "winding_temp_C": 82.1,
    "rms_velocity_mm_s": 6.8,
    "peak_velocity_mm_s": 12.5,
    ...
  },
  "include_forecast": false  // Optional: add time-series forecast
}
```

**Response:**
```json
{
  "request_id": "uuid-12345",
  "timestamp": "2025-11-25T10:00:00Z",
  "machine_id": "motor_siemens_1la7_001",
  "model_type": "classification",
  
  "ml_prediction": {
    "failure_type": "bearing_wear",
    "failure_probability": 0.87,
    "confidence": 0.92
  },
  
  "explanation": {
    "text": "The motor is exhibiting signs of bearing wear with 87% probability...",
    "key_findings": [
      "Elevated vibration at 12.5 mm/s RMS velocity (40% above baseline)",
      "Bearing temperature at 78.5°C approaching threshold",
      "Frequency analysis shows bearing defect signature"
    ],
    "root_cause": "Progressive bearing degradation due to insufficient lubrication...",
    "recommendations": [
      "1. Schedule bearing inspection within 48 hours",
      "2. Check lubrication system for proper oil level",
      "3. Monitor vibration trends closely"
    ],
    "confidence": "high"
  },
  
  "metadata": {
    "ml_inference_time_ms": 850,
    "rag_retrieval_time_ms": 142,
    "llm_generation_time_ms": 28500,
    "total_latency_ms": 29492,
    "context_documents": 5,
    "prompt_template": "classification_v1"
  }
}
```

---

## 4. Error Handling Strategy

### 4.1 Graceful Degradation

```python
try:
    # Full pipeline
    ml_prediction = run_ml_inference()
    rag_context = retrieve_context()
    explanation = generate_explanation()
except MLModelError:
    # ML model fails → Use last known good prediction
    ml_prediction = cache.get_last_prediction(machine_id)
    explanation = generate_explanation_from_cache()
except RAGRetrievalError:
    # RAG fails → Use generic context
    rag_context = get_default_context(model_type)
    explanation = generate_explanation_with_generic_context()
except LLMGenerationError:
    # LLM fails → Return raw ML prediction
    return {
        "ml_prediction": ml_prediction,
        "explanation": {"text": "LLM unavailable - showing raw prediction"},
        "error": "llm_timeout"
    }
```

### 4.2 Timeout Handling

- **ML Inference:** 2 seconds max
- **RAG Retrieval:** 0.5 seconds max
- **LLM Generation:** 60 seconds max
- **Total Request:** 65 seconds max

### 4.3 Fallback Strategies

1. **ML Model Unavailable:**
   - Use last cached prediction (5-minute TTL)
   - Return "model loading" status if first request
   
2. **RAG Context Empty:**
   - Use generic failure mode descriptions
   - Reduce explanation detail level
   
3. **LLM Timeout:**
   - Return raw ML predictions only
   - Queue explanation for async generation

---

## 5. Caching Strategy

### 5.1 Model Caching

**ML Models:**
- **Strategy:** In-memory, lazy load on first request
- **Eviction:** LRU with 5-model limit (memory constraint)
- **Preload:** 3 most-used machines at startup

**LLM Model:**
- **Strategy:** In-memory, persistent (loaded at startup)
- **Size:** 4.92 GB Llama 3.1 8B Q4_K_M
- **Never evict:** Single model for all requests

**RAG Index:**
- **Strategy:** In-memory, persistent
- **Size:** ~50 MB FAISS IndexFlatL2
- **Rebuild:** Daily at 2 AM (add new documents)

### 5.2 Prediction Caching

**ML Predictions:**
- **Key:** `hash(machine_id, sensor_data, model_type)`
- **TTL:** 5 minutes
- **Reason:** Identical sensor data → same prediction

**RAG Results:**
- **Key:** `hash(query, machine_id, top_k)`
- **TTL:** 5 minutes
- **Reason:** Similar queries → same documents

**LLM Explanations:**
- **Key:** `hash(prompt)`
- **TTL:** 15 minutes
- **Reason:** Same prediction + context → same explanation

---

## 6. Performance Targets

### 6.1 Latency Goals (Raspberry Pi 5)

| Component | Target | Measured | Status |
|-----------|--------|----------|--------|
| ML Inference | <1s | 0.5-1s | ✅ |
| RAG Retrieval | <0.2s | 0.14s | ✅ |
| LLM Generation | <30s | 25-35s | ✅ |
| **Total Latency** | **<35s** | **30-40s** | ✅ |

### 6.2 Throughput Goals

- **Concurrent Requests:** 2-3 (memory limited)
- **Requests per Minute:** 10-15
- **Daily Capacity:** 10,000+ predictions

### 6.3 Resource Utilization

**Raspberry Pi 5 (16GB RAM):**
- LLM Model: 4.92 GB
- ML Models (3 cached): 0.7-1.5 GB
- RAG Index: 0.05 GB
- Application: 0.5 GB
- **Total:** ~6.5-7 GB (40% utilization)
- **Headroom:** 9 GB (60%)

---

## 7. Edge Deployment Considerations

### 7.1 Memory Management

**Strategies:**
1. **Lazy Loading:** Load ML models on demand
2. **LRU Eviction:** Keep 3 most-used models
3. **Model Quantization:** Q4_K_M for LLM (4-bit)
4. **Batch Processing:** Group requests when possible

### 7.2 Offline Mode

**Fallback when internet unavailable:**
1. Use cached ML models (always local)
2. Use cached RAG index (always local)
3. Use local LLM (always local)
4. **No external dependencies required**

### 7.3 Model Update Strategy

**Weekly Updates:**
1. Download new ML models (if retrained)
2. Download updated RAG documents
3. Reload models at 2 AM low-traffic window
4. Preserve old models as backup

---

## 8. Testing Strategy

### 8.1 Unit Tests

- ✅ Each ML predictor tested independently
- ✅ RAG retrieval accuracy validated
- ✅ LLM generation quality checked
- ⏳ End-to-end pipeline testing (Phase 3.5.2)

### 8.2 Integration Tests

**Test Cases:**
1. **Happy Path:** ML → RAG → LLM → Explanation (all succeed)
2. **ML Failure:** Use cached prediction
3. **RAG Failure:** Use generic context
4. **LLM Timeout:** Return raw prediction
5. **Concurrent Requests:** 3 simultaneous requests

### 8.3 Performance Tests

**Benchmarks:**
- Latency under load (10 req/min)
- Memory usage over 24 hours
- Model cache hit rates
- Explanation quality metrics

---

## 9. Monitoring & Observability

### 9.1 Metrics to Track

**Performance:**
- Request latency (p50, p95, p99)
- Component latency (ML, RAG, LLM)
- Throughput (requests per minute)
- Error rates by component

**Resource:**
- RAM usage (total, per component)
- CPU usage (average, peak)
- Disk I/O (model loading)

**Quality:**
- ML prediction confidence
- RAG context relevance scores
- LLM explanation length/quality

### 9.2 Logging Strategy

**Log Levels:**
- ERROR: ML failures, LLM timeouts, crashes
- WARN: Cache misses, slow requests (>40s)
- INFO: Request start/end, model loads
- DEBUG: Prediction details, RAG context

**Log Format:**
```json
{
  "timestamp": "2025-11-25T10:00:00Z",
  "level": "INFO",
  "request_id": "uuid-12345",
  "machine_id": "motor_siemens_1la7_001",
  "model_type": "classification",
  "latency_ms": 32500,
  "ml_time_ms": 850,
  "rag_time_ms": 142,
  "llm_time_ms": 28500,
  "confidence": 0.92
}
```

---

## 10. Security Considerations

### 10.1 Input Validation

- **Sensor Data:** Validate ranges (no negative temps, reasonable values)
- **Machine ID:** Whitelist of 27 known machines
- **Model Type:** Enum validation (only 4 types)

### 10.2 Rate Limiting

- **Per Machine:** 30 requests/minute
- **Global:** 60 requests/minute
- **Burst:** 10 requests/second

### 10.3 Data Privacy

- **No External Calls:** All processing on-device
- **No Data Storage:** Predictions not persisted (cache only)
- **No Telemetry:** No data sent to external servers

---

## 11. Deployment Architecture

### 11.1 Directory Structure

```
/opt/predictive_maintenance/
├── LLM/
│   ├── models/
│   │   └── llama-3.1-8b-q4_k_m.gguf (4.92 GB)
│   ├── data/
│   │   └── embeddings/ (FAISS index)
│   ├── api/
│   │   ├── inference_service.py
│   │   └── server.py (FastAPI)
│   └── config/
│       └── production_config.json
├── ml_models/
│   ├── models/
│   │   ├── classification/ (10 models)
│   │   ├── anomaly/ (10 models)
│   │   ├── rul/ (10 models)
│   │   └── timeseries/ (11 models)
│   └── scripts/
│       └── inference/ (4 predictor scripts)
└── logs/
    ├── app.log
    └── performance.log
```

### 11.2 Service Configuration

**Systemd Service:**
```ini
[Unit]
Description=Predictive Maintenance API
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/opt/predictive_maintenance
ExecStart=/usr/bin/python3 LLM/api/server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## 12. Phase 3.5 Roadmap

### Phase 3.5.0: Prerequisites ✅ COMPLETE

- ✅ Create 4 ML inference scripts
- ✅ Generate 100 test predictions
- ✅ Document integration architecture
- ⏳ Create unified service skeleton

### Phase 3.5.1: MLExplainer Implementation (Next)

**Duration:** 2-3 days

**Tasks:**
1. Implement UnifiedInferenceService class
2. Integrate all 4 ML predictors
3. Connect RAG retriever
4. Connect LLM generator
5. Test end-to-end pipeline
6. Create FastAPI endpoints

**Deliverable:** Working MLExplainer API

### Phase 3.5.2: End-to-End Testing

**Duration:** 1 day

**Tasks:**
1. Test all 4 model types
2. Validate explanation quality
3. Measure latency
4. Performance benchmarking
5. Create test report

**Deliverable:** Validated pipeline

### Phase 3.6: Raspberry Pi 5 POC

**Duration:** 1-2 days

**Tasks:**
1. Setup Pi 5 environment
2. Install dependencies (ARM builds)
3. Deploy and test
4. Measure real-world performance

**Deliverable:** Pi 5 benchmark results

---

## 13. Success Criteria

### Phase 3.5 Complete When:

- ✅ All 4 ML inference scripts functional
- ✅ 100 test predictions generated
- ⏳ UnifiedInferenceService implemented
- ⏳ End-to-end pipeline tested (4 model types)
- ⏳ Latency <40 seconds on Pi 5
- ⏳ Explanation quality validated
- ⏳ API documentation complete
- ⏳ Error handling comprehensive

---

## 14. References

**Related Documents:**
- `PHASE_3.5_PREREQUISITE_CHECKLIST.md` - Prerequisite analysis
- `LLM/WORKFLOW_AUTOMATION_COMPLETE.md` - Phase 3.4 completion report
- `ml_models/PHASE_2_ML_DETAILED_APPROACH.md` - ML model training details

**Code Files:**
- `ml_models/scripts/inference/predict_*.py` - 4 inference scripts
- `ml_models/scripts/inference/generate_test_predictions.py` - Batch generator
- `LLM/scripts/inference/test_prompts.py` - Phase 3.4.2 prompt testing

**Models:**
- Classification: `ml_models/models/classification/` (10 models)
- Anomaly: `ml_models/models/anomaly/` (10 models)
- RUL: `ml_models/models/regression/` (10 models)
- Time-Series: `ml_models/models/timeseries/` (11 models)
- LLM: `LLM/models/llama-3.1-8b-instruct-q4_k_m.gguf` (4.92 GB)

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-25  
**Status:** Phase 3.5.0 Complete, Phase 3.5.1 Ready
