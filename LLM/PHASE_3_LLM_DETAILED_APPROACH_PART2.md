# PHASE 3 PART 2: LLM EXPLANATIONS (INTEGRATION & DEPLOYMENT)
**Duration:** 4 weeks (Part 2: 2 weeks)  
**Goal:** Integrate LLM with ML models and deploy to production  
**Status:** 🔄 READY TO START (November 24, 2025)

---

## Overview

**Part 2 Scope (This Document):**
- Phase 3.5: Integration with ML Models
- Phase 3.6: Testing & Validation
- Phase 3.7: Production Deployment
- Phase 3.8: Monitoring & Maintenance

**Prerequisites from Part 1:**
- ✅ RAG system with FAISS index (127+ docs)
- ✅ Llama 3.1 8B installed and tested
- ✅ Synthetic knowledge base created
- ✅ Prompt templates ready

---

## PHASE 3.5: Integration with ML Models
**Duration:** Week 5 (Days 1-5)  
**Goal:** Connect LLM to ML prediction outputs

### Phase 3.5.1: API Design (Days 1-2)

**Create unified explanation API:**

**Script:** `LLM/api/explainer.py`
```python
"""
Unified LLM explainer for all ML model types
"""
from pathlib import Path
import sys
sys.path.append('../scripts/inference')
sys.path.append('../scripts/rag')
sys.path.append('../config')

from test_llama import LlamaInference
from retriever import MachineDocRetriever
from prompts import *

class MLExplainer:
    def __init__(self):
        self.llm = LlamaInference()
        self.retriever = MachineDocRetriever()
    
    def explain_classification(self, machine_id, failure_prob, failure_type, 
                               sensor_data, confidence=0.9):
        """Explain failure classification prediction"""
        
        # Retrieve relevant context
        query = f"{failure_type} symptoms in {machine_id}"
        rag_results = self.retriever.retrieve(query, machine_id, top_k=2)
        rag_context = "\n\n".join([r['doc'][:400] for r in rag_results])
        
        # Format sensor readings
        sensor_str = "\n".join([f"{k}: {v:.2f}" for k, v in sensor_data.items()])
        
        # Fill prompt
        user_message = FAILURE_CLASSIFICATION_PROMPT.format(
            machine_id=machine_id,
            failure_probability=failure_prob,
            failure_type=failure_type,
            confidence=confidence,
            sensor_readings=sensor_str,
            rag_context=rag_context
        )
        
        # Generate explanation
        explanation = self.llm.generate(SYSTEM_PROMPT, user_message, max_tokens=300)
        
        return {
            'explanation': explanation,
            'sources': [r['machine_id'] for r in rag_results],
            'confidence': confidence
        }
    
    def explain_rul(self, machine_id, rul_hours, sensor_data, confidence=0.9):
        """Explain RUL prediction"""
        
        query = f"RUL estimation maintenance for {machine_id}"
        rag_results = self.retriever.retrieve(query, machine_id, top_k=2)
        rag_context = "\n\n".join([r['doc'][:400] for r in rag_results])
        
        sensor_str = "\n".join([f"{k}: {v:.2f}" for k, v in sensor_data.items()])
        
        user_message = RUL_REGRESSION_PROMPT.format(
            machine_id=machine_id,
            rul_hours=rul_hours,
            rul_days=rul_hours / 24,
            confidence=confidence,
            sensor_readings=sensor_str,
            rag_context=rag_context
        )
        
        explanation = self.llm.generate(SYSTEM_PROMPT, user_message, max_tokens=300)
        
        return {
            'explanation': explanation,
            'sources': [r['machine_id'] for r in rag_results],
            'confidence': confidence
        }
    
    def explain_anomaly(self, machine_id, anomaly_score, abnormal_sensors, 
                        detection_method, threshold=0.5):
        """Explain anomaly detection"""
        
        query = f"anomaly investigation for {machine_id}"
        rag_results = self.retriever.retrieve(query, machine_id, top_k=2)
        rag_context = "\n\n".join([r['doc'][:400] for r in rag_results])
        
        sensor_str = "\n".join([f"{k}: {v:.2f}" for k, v in abnormal_sensors.items()])
        
        user_message = ANOMALY_DETECTION_PROMPT.format(
            machine_id=machine_id,
            anomaly_score=anomaly_score,
            detection_method=detection_method,
            abnormal_sensors=sensor_str,
            rag_context=rag_context
        )
        
        explanation = self.llm.generate(SYSTEM_PROMPT, user_message, max_tokens=300)
        
        return {
            'explanation': explanation,
            'sources': [r['machine_id'] for r in rag_results],
            'anomaly_score': anomaly_score,
            'threshold': threshold
        }
    
    def explain_forecast(self, machine_id, forecast_summary, confidence=0.85):
        """Explain time-series forecast"""
        
        query = f"sensor forecasting for {machine_id}"
        rag_results = self.retriever.retrieve(query, machine_id, top_k=2)
        rag_context = "\n\n".join([r['doc'][:400] for r in rag_results])
        
        user_message = TIMESERIES_FORECAST_PROMPT.format(
            machine_id=machine_id,
            forecast_summary=forecast_summary,
            confidence=confidence,
            rag_context=rag_context
        )
        
        explanation = self.llm.generate(SYSTEM_PROMPT, user_message, max_tokens=250)
        
        return {
            'explanation': explanation,
            'sources': [r['machine_id'] for r in rag_results],
            'confidence': confidence
        }

# Test
if __name__ == "__main__":
    explainer = MLExplainer()
    
    # Test classification
    result = explainer.explain_classification(
        machine_id="motor_siemens_1la7_001",
        failure_prob=0.87,
        failure_type="bearing_wear",
        sensor_data={'vibration': 12.5, 'temperature': 78.0, 'current': 45.2},
        confidence=0.92
    )
    
    print("\n=== CLASSIFICATION EXPLANATION ===")
    print(result['explanation'])
    print(f"\nSources: {result['sources']}")
```

**Run test:**
```powershell
cd LLM/api
python explainer.py
```

---

### Phase 3.5.2: ML Model Integration (Days 3-5)

**Connect to existing ML models:**

**Script:** `LLM/api/ml_integration.py`
```python
"""
Integration with Phase 2 ML models
"""
import pickle
import pandas as pd
from pathlib import Path
from explainer import MLExplainer

class IntegratedPredictionSystem:
    def __init__(self):
        self.explainer = MLExplainer()
        self.models = self.load_models()
    
    def load_models(self):
        """Load trained ML models from Phase 2"""
        models_dir = Path("../../ml_models/models/trained")
        
        return {
            'classification': {}, # Load per machine
            'regression': {},
            'anomaly': {},
            'timeseries': {}
        }
    
    def predict_with_explanation(self, machine_id, sensor_data, model_type='all'):
        """
        Run ML prediction and generate LLM explanation
        
        Args:
            machine_id: Machine identifier
            sensor_data: Dict of sensor readings
            model_type: 'classification', 'regression', 'anomaly', 'timeseries', or 'all'
        
        Returns:
            Dict with predictions and explanations
        """
        results = {}
        
        # Classification
        if model_type in ['classification', 'all']:
            pred = self.predict_classification(machine_id, sensor_data)
            explanation = self.explainer.explain_classification(
                machine_id=machine_id,
                failure_prob=pred['probability'],
                failure_type=pred['failure_type'],
                sensor_data=sensor_data,
                confidence=pred['confidence']
            )
            results['classification'] = {
                'prediction': pred,
                'explanation': explanation
            }
        
        # Regression (RUL)
        if model_type in ['regression', 'all']:
            rul = self.predict_rul(machine_id, sensor_data)
            explanation = self.explainer.explain_rul(
                machine_id=machine_id,
                rul_hours=rul['rul_hours'],
                sensor_data=sensor_data,
                confidence=rul['confidence']
            )
            results['regression'] = {
                'prediction': rul,
                'explanation': explanation
            }
        
        # Anomaly
        if model_type in ['anomaly', 'all']:
            anomaly = self.detect_anomaly(machine_id, sensor_data)
            if anomaly['is_anomaly']:
                explanation = self.explainer.explain_anomaly(
                    machine_id=machine_id,
                    anomaly_score=anomaly['score'],
                    abnormal_sensors=anomaly['abnormal_sensors'],
                    detection_method=anomaly['method']
                )
                results['anomaly'] = {
                    'prediction': anomaly,
                    'explanation': explanation
                }
        
        return results
    
    def predict_classification(self, machine_id, sensor_data):
        """Mock classification prediction"""
        return {
            'failure_type': 'bearing_wear',
            'probability': 0.87,
            'confidence': 0.92
        }
    
    def predict_rul(self, machine_id, sensor_data):
        """Mock RUL prediction"""
        return {
            'rul_hours': 156.5,
            'confidence': 0.89
        }
    
    def detect_anomaly(self, machine_id, sensor_data):
        """Mock anomaly detection"""
        return {
            'is_anomaly': True,
            'score': 0.78,
            'abnormal_sensors': {'vibration': 12.5, 'temperature': 78.0},
            'method': 'Isolation Forest'
        }

# Test
if __name__ == "__main__":
    system = IntegratedPredictionSystem()
    
    results = system.predict_with_explanation(
        machine_id="motor_siemens_1la7_001",
        sensor_data={'vibration': 12.5, 'temperature': 78.0, 'current': 45.2},
        model_type='all'
    )
    
    print("\n=== INTEGRATED PREDICTIONS ===")
    for model_type, result in results.items():
        print(f"\n{model_type.upper()}:")
        print(f"Prediction: {result['prediction']}")
        print(f"Explanation: {result['explanation']['explanation'][:200]}...")
```

**Deliverables:**
- ✅ MLExplainer class (4 methods)
- ✅ Integration with Phase 2 models
- ✅ Unified prediction + explanation API

---

## PHASE 3.6: Testing & Validation
**Duration:** Week 6 (Days 1-5)  
**Goal:** Validate explanation quality and accuracy

### Phase 3.6.1: Explanation Quality Metrics (Days 1-3)

**Evaluate LLM output quality:**

**Script:** `LLM/scripts/validation/test_explanation_quality.py`
```python
"""
Test explanation quality with automated metrics
"""
from api.explainer import MLExplainer
import json

def test_explanation_quality():
    """Test multiple scenarios and evaluate"""
    
    explainer = MLExplainer()
    
    test_cases = [
        {
            'machine_id': 'motor_siemens_1la7_001',
            'failure_prob': 0.92,
            'failure_type': 'bearing_wear',
            'sensor_data': {'vibration': 15.2, 'temperature': 85.0},
            'expected_keywords': ['bearing', 'vibration', 'maintenance', 'replace']
        },
        {
            'machine_id': 'pump_grundfos_cr3_004',
            'failure_prob': 0.68,
            'failure_type': 'cavitation',
            'sensor_data': {'pressure': 2.1, 'flow_rate': 145.0},
            'expected_keywords': ['cavitation', 'pressure', 'inspect']
        }
    ]
    
    results = []
    
    for case in test_cases:
        explanation = explainer.explain_classification(
            machine_id=case['machine_id'],
            failure_prob=case['failure_prob'],
            failure_type=case['failure_type'],
            sensor_data=case['sensor_data']
        )
        
        # Check quality metrics
        text = explanation['explanation'].lower()
        
        quality = {
            'machine_id': case['machine_id'],
            'length': len(text.split()),
            'keywords_found': sum(1 for kw in case['expected_keywords'] if kw in text),
            'keywords_total': len(case['expected_keywords']),
            'has_action': any(word in text for word in ['replace', 'inspect', 'check', 'monitor']),
            'has_safety': any(word in text for word in ['safety', 'shutdown', 'caution', 'risk']),
            'concise': len(text.split()) <= 250
        }
        
        quality['score'] = (
            (quality['keywords_found'] / quality['keywords_total']) * 0.4 +
            (1.0 if quality['has_action'] else 0.0) * 0.3 +
            (1.0 if quality['concise'] else 0.0) * 0.2 +
            (1.0 if quality['has_safety'] else 0.0) * 0.1
        )
        
        results.append(quality)
        
        print(f"\n{case['machine_id']}: Score {quality['score']:.2f}")
        print(f"  Words: {quality['length']}")
        print(f"  Keywords: {quality['keywords_found']}/{quality['keywords_total']}")
        print(f"  Action: {'✓' if quality['has_action'] else '✗'}")
        print(f"  Safety: {'✓' if quality['has_safety'] else '✗'}")
    
    # Save results
    with open('../../reports/explanation_quality_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    avg_score = sum(r['score'] for r in results) / len(results)
    print(f"\n=== AVERAGE QUALITY SCORE: {avg_score:.2f} ===")

if __name__ == "__main__":
    test_explanation_quality()
```

**Run:**
```powershell
cd LLM/scripts/validation
python test_explanation_quality.py
```

---

### Phase 3.6.2: Integration Testing (Days 4-5)

**End-to-end testing:**

**Script:** `LLM/scripts/validation/test_integration.py`
```python
"""
Integration tests for ML + LLM pipeline
"""
from api.ml_integration import IntegratedPredictionSystem
import time

def test_e2e_pipeline():
    """Test complete pipeline"""
    
    system = IntegratedPredictionSystem()
    
    # Test data
    test_machines = [
        'motor_siemens_1la7_001',
        'pump_grundfos_cr3_004',
        'compressor_atlas_copco_ga30_001'
    ]
    
    test_sensors = {
        'vibration': 10.5,
        'temperature': 72.0,
        'current': 42.0,
        'voltage': 400.0
    }
    
    print("=== INTEGRATION TEST ===\n")
    
    for machine_id in test_machines:
        print(f"Testing {machine_id}...")
        
        start = time.time()
        results = system.predict_with_explanation(
            machine_id=machine_id,
            sensor_data=test_sensors,
            model_type='all'
        )
        elapsed = time.time() - start
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Models run: {len(results)}")
        print(f"  Explanations generated: {sum(1 for r in results.values() if 'explanation' in r)}")
        
        # Check latency
        assert elapsed < 15.0, f"Pipeline too slow: {elapsed:.2f}s"
        
        print("  ✓ PASS\n")
    
    print("=== ALL TESTS PASSED ===")

if __name__ == "__main__":
    test_e2e_pipeline()
```

**Run:**
```powershell
cd LLM/scripts/validation
python test_integration.py
```

**Deliverables:**
- ✅ Quality metrics for explanations
- ✅ Integration tests passing
- ✅ Latency <15 sec for full pipeline

---

## PHASE 3.7: Production Deployment
**Duration:** Week 7 (Days 1-5)  
**Goal:** Deploy FastAPI server with LLM explanations

### Phase 3.7.1: FastAPI Server (Days 1-3)

**Create production API:**

**Script:** `LLM/api/main.py`
```python
"""
FastAPI server for ML predictions with LLM explanations
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
from ml_integration import IntegratedPredictionSystem

app = FastAPI(title="Predictive Maintenance LLM API", version="1.0")

# Initialize system
system = IntegratedPredictionSystem()

class PredictionRequest(BaseModel):
    machine_id: str
    sensor_data: Dict[str, float]
    model_type: Optional[str] = 'all'

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Get ML predictions with LLM explanations
    
    Example:
    ```
    {
        "machine_id": "motor_siemens_1la7_001",
        "sensor_data": {"vibration": 12.5, "temperature": 78.0},
        "model_type": "all"
    }
    ```
    """
    try:
        results = system.predict_with_explanation(
            machine_id=request.machine_id,
            sensor_data=request.sensor_data,
            model_type=request.model_type
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain/classification")
async def explain_classification(
    machine_id: str,
    failure_prob: float,
    failure_type: str,
    sensor_data: Dict[str, float]
):
    """Generate explanation for classification prediction"""
    try:
        explanation = system.explainer.explain_classification(
            machine_id=machine_id,
            failure_prob=failure_prob,
            failure_type=failure_type,
            sensor_data=sensor_data
        )
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain/rul")
async def explain_rul(
    machine_id: str,
    rul_hours: float,
    sensor_data: Dict[str, float]
):
    """Generate explanation for RUL prediction"""
    try:
        explanation = system.explainer.explain_rul(
            machine_id=machine_id,
            rul_hours=rul_hours,
            sensor_data=sensor_data
        )
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "llm": "loaded", "models": "ready"}

@app.get("/")
async def root():
    """API info"""
    return {
        "name": "Predictive Maintenance LLM API",
        "version": "1.0",
        "endpoints": ["/predict", "/explain/classification", "/explain/rul", "/health"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Run server:**
```powershell
cd LLM/api
python main.py
```

**Test API:**
```powershell
# Test prediction
Invoke-WebRequest -Uri "http://localhost:8000/predict" -Method POST -ContentType "application/json" -Body '{"machine_id":"motor_siemens_1la7_001","sensor_data":{"vibration":12.5,"temperature":78.0}}'

# Test health
Invoke-WebRequest -Uri "http://localhost:8000/health"
```

---

### Phase 3.7.2: Docker Deployment (Days 4-5)

**Containerize application:**

**File:** `LLM/Dockerfile`
```dockerfile
FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**File:** `LLM/requirements.txt`
```
fastapi==0.104.1
uvicorn==0.24.0
llama-cpp-python==0.2.20
sentence-transformers==2.2.2
faiss-cpu==1.7.4
pydantic==2.4.2
numpy==1.24.3
```

**Install llama-cpp-python with GPU support:**
```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

**Build and run:**
```powershell
cd LLM

# Build image
docker build -t predictive-maintenance-llm:v1.0 .

# Run container
docker run -d -p 8000:8000 --gpus all predictive-maintenance-llm:v1.0
```

**Deliverables:**
- ✅ FastAPI server with 5+ endpoints
- ✅ Dockerized deployment
- ✅ GPU support configured

---

## PHASE 3.8: Monitoring & Maintenance
**Duration:** Week 8 (Days 1-5)  
**Goal:** Setup monitoring and logging

### Phase 3.8.1: Logging Infrastructure (Days 1-3)

**Add comprehensive logging:**

**Script:** `LLM/api/logger.py`
```python
"""
Logging for LLM API
"""
import logging
from datetime import datetime
import json

def setup_logger():
    """Configure logging"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../logs/llm_api.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('LLM_API')

def log_prediction(machine_id, model_type, latency, success):
    """Log prediction event"""
    
    logger = logging.getLogger('LLM_API')
    
    event = {
        'timestamp': datetime.now().isoformat(),
        'machine_id': machine_id,
        'model_type': model_type,
        'latency_sec': latency,
        'success': success
    }
    
    logger.info(f"PREDICTION: {json.dumps(event)}")

def log_explanation(machine_id, explanation_length, sources_count, latency):
    """Log explanation generation"""
    
    logger = logging.getLogger('LLM_API')
    
    event = {
        'timestamp': datetime.now().isoformat(),
        'machine_id': machine_id,
        'explanation_words': explanation_length,
        'sources': sources_count,
        'latency_sec': latency
    }
    
    logger.info(f"EXPLANATION: {json.dumps(event)}")
```

**Update main.py to add logging:**
```python
# Add to main.py
from logger import setup_logger, log_prediction, log_explanation
import time

logger = setup_logger()

@app.post("/predict")
async def predict(request: PredictionRequest):
    start = time.time()
    success = False
    
    try:
        results = system.predict_with_explanation(...)
        success = True
        return results
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        latency = time.time() - start
        log_prediction(request.machine_id, request.model_type, latency, success)
```

---

### Phase 3.8.2: Performance Monitoring (Days 4-5)

**Track metrics:**

**Script:** `LLM/scripts/monitoring/analyze_logs.py`
```python
"""
Analyze API logs for performance metrics
"""
import json
from pathlib import Path
from datetime import datetime, timedelta

def analyze_logs(log_file='../../logs/llm_api.log', hours=24):
    """Analyze recent logs"""
    
    predictions = []
    explanations = []
    
    cutoff = datetime.now() - timedelta(hours=hours)
    
    with open(log_file) as f:
        for line in f:
            if 'PREDICTION:' in line:
                data = json.loads(line.split('PREDICTION: ')[1])
                if datetime.fromisoformat(data['timestamp']) > cutoff:
                    predictions.append(data)
            
            elif 'EXPLANATION:' in line:
                data = json.loads(line.split('EXPLANATION: ')[1])
                if datetime.fromisoformat(data['timestamp']) > cutoff:
                    explanations.append(data)
    
    # Calculate metrics
    if predictions:
        avg_latency = sum(p['latency_sec'] for p in predictions) / len(predictions)
        success_rate = sum(1 for p in predictions if p['success']) / len(predictions)
        
        print(f"\n=== LAST {hours}H METRICS ===")
        print(f"Total Predictions: {len(predictions)}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Avg Latency: {avg_latency:.2f}s")
        print(f"Explanations Generated: {len(explanations)}")
        
        if explanations:
            avg_exp_len = sum(e['explanation_words'] for e in explanations) / len(explanations)
            avg_exp_latency = sum(e['latency_sec'] for e in explanations) / len(explanations)
            print(f"Avg Explanation Length: {avg_exp_len:.0f} words")
            print(f"Avg Explanation Latency: {avg_exp_latency:.2f}s")

if __name__ == "__main__":
    analyze_logs(hours=24)
```

**Run:**
```powershell
cd LLM/scripts/monitoring
python analyze_logs.py
```

---

## Deliverables Summary (Part 2)

**Phase 3.5: Integration**
- ✅ MLExplainer API (4 methods)
- ✅ IntegratedPredictionSystem connecting ML + LLM
- ✅ Tested with Phase 2 models

**Phase 3.6: Testing**
- ✅ Explanation quality metrics
- ✅ Integration tests passing
- ✅ Latency validated (<15s)

**Phase 3.7: Deployment**
- ✅ FastAPI server with 5+ endpoints
- ✅ Docker container ready
- ✅ GPU support configured

**Phase 3.8: Monitoring**
- ✅ Logging infrastructure
- ✅ Performance monitoring scripts
- ✅ Log analysis tools

---

## Complete Phase 3 Summary

**Total Duration:** 8 weeks  
**Status:** ✅ READY TO IMPLEMENT

**Part 1 Deliverables:**
- RAG system with 127+ documents
- Llama 3.1 8B installed (llama.cpp GGUF, ~3GB VRAM)
- Synthetic knowledge base
- 4 prompt templates
- No transformers dependency (avoids conflicts!)

**Part 2 Deliverables:**
- ML integration API
- Quality validation framework
- Production FastAPI server
- Docker deployment
- Monitoring infrastructure

**Final Folder Structure:**
```
LLM/
├── api/
│   ├── main.py (FastAPI server)
│   ├── explainer.py (MLExplainer)
│   ├── ml_integration.py (IntegratedPredictionSystem)
│   └── logger.py
├── config/
│   ├── llama_config.json
│   └── prompts.py
├── data/
│   ├── knowledge_base/
│   │   ├── machines/ (27 .txt)
│   │   └── failure_cases.json
│   └── embeddings/
│       ├── machines.index
│       └── metadata.pkl
├── logs/
│   └── llm_api.log
├── models/
│   └── llama-3.1-8b-instruct-4bit/
├── scripts/
│   ├── preprocessing/
│   ├── rag/
│   ├── inference/
│   ├── validation/
│   └── monitoring/
├── reports/
│   └── explanation_quality_report.json
├── Dockerfile
├── requirements.txt
├── PHASE_3_LLM_DETAILED_APPROACH_PART1.md
└── PHASE_3_LLM_DETAILED_APPROACH_PART2.md
```

**Performance Targets (with llama.cpp):**
- ✅ Explanation generation: <3 seconds (faster with llama.cpp!)
- ✅ RAG retrieval: <100ms
- ✅ Full pipeline: <10 seconds
- ✅ VRAM usage: ~3GB (fits RTX 4070 8GB with 5GB headroom!)
- ✅ Explanation quality: >0.75 score

**Ready for Phase 4 (Optional):** Vision-Language Models for visual inspection
