# Complete Workflow: Development to Production

**Date:** November 25, 2025  
**Status:** Phase 3.3.1 Complete (Synthetic data stage)

---

## Two-Stage Approach

### STAGE 1: Development (Current - Phases 3.1-3.4)
**Purpose:** Build and test LLM system with synthetic data

```
📊 Synthetic Random Data
    ↓
💾 Store in failure_cases.json (100+ examples)
    ↓
🧮 Embed in FAISS index (RAG knowledge base)
    ↓
🤖 LLM learns explanation patterns
    ↓
✅ Test prompts and tune responses
```

**Why Synthetic Data?**
- Safe testing environment
- No dependency on real sensor data
- Can generate diverse failure scenarios
- Validates LLM/RAG architecture works

---

### STAGE 2: Production (Future - Phase 3.5+)
**Purpose:** Full automation with real ML predictions

```
🔧 REAL PRODUCTION WORKFLOW (FULLY AUTOMATED):

1. Raspberry Pi 5
   - Collects sensor data (vibration, temp, current, voltage)
   - Every 1 minute (configurable)
   ↓

2. Data Preprocessing
   - Normalize sensor values
   - Create feature windows
   - Prepare for ML models
   ↓

3. ML Model Predictions (Phase 2 Models)
   
   A. Classification Model (failure_classifier.pkl)
      Output: {
        "failure_type": "bearing_wear",
        "probability": 0.87,
        "confidence": 0.92
      }
   
   B. Regression Model (rul_regressor.pkl)
      Output: {
        "rul_hours": 145.3,
        "rul_days": 6.05
      }
   
   C. Anomaly Detection (anomaly_detector.pkl)
      Output: {
        "anomaly_score": 0.73,
        "is_anomaly": true,
        "method": "Isolation Forest"
      }
   
   D. Time-Series Forecast (timeseries_forecaster.pkl)
      Output: {
        "forecast_24h": [temp values...],
        "peak_time": "14:30",
        "peak_value": 92.5
      }
   ↓

4. RAG Retrieval System (Phase 3.1)
   - Query: "bearing_wear symptoms motor_siemens_1la7_001"
   - Search FAISS index
   - Retrieve top 3 relevant documents
   - Returns: Machine specs, failure modes, maintenance procedures
   ↓

5. LLM Explanation Generation (Phase 3.2 + 3.4)
   
   Input to Llama 3.1 8B:
   {
     "system_prompt": "You are a maintenance expert...",
     "ml_predictions": {ML outputs from step 3},
     "rag_context": {Retrieved docs from step 4},
     "prompt_template": FAILURE_CLASSIFICATION_PROMPT
   }
   
   LLM Output:
   "⚠️ BEARING WEAR ALERT - Motor Siemens 1LA7 001
   
   Your motor bearings show signs of wear with 87% confidence.
   
   KEY INDICATORS:
   - Vibration: 12.5 mm/s (normal: <5 mm/s) ⬆️ 150% above normal
   - Temperature: 78°C (normal: <65°C) ⬆️ 20% above normal
   
   REMAINING TIME: ~6 days (145 hours) before failure
   
   IMMEDIATE ACTIONS:
   1. Schedule bearing replacement within 5 days
   2. Increase monitoring frequency to every 30 minutes
   3. Reduce load by 20% until maintenance
   
   COST IMPACT:
   - Planned maintenance: $2,500
   - Emergency failure: $6,250 (2.5x more expensive!)
   
   SAFETY: Wear PPE, lockout/tagout before inspection"
   ↓

6. User Interface Display
   - Web dashboard (FastAPI + React)
   - Mobile app notifications
   - Email alerts for critical issues
   - Export reports (PDF/Excel)
```

---

## Complete Integration Flow

### Phase 3.5: Integration Layer (Next Phase)

**File:** `LLM/scripts/integration/ml_to_llm_pipeline.py`

```python
"""
Production pipeline: ML Predictions → LLM Explanations
"""
from pathlib import Path
import pickle
import pandas as pd
from rag.retriever import MachineDocRetriever
from inference.test_llama import LlamaInference
from config.prompts import FAILURE_CLASSIFICATION_PROMPT, SYSTEM_PROMPT


class ProductionPipeline:
    def __init__(self):
        # Load Phase 2 ML models
        models_dir = Path("../../../ml_models/models")
        self.classifier = pickle.load(open(models_dir / "failure_classifier.pkl", 'rb'))
        self.regressor = pickle.load(open(models_dir / "rul_regressor.pkl", 'rb'))
        self.anomaly_detector = pickle.load(open(models_dir / "anomaly_detector.pkl", 'rb'))
        
        # Load Phase 3 LLM components
        self.retriever = MachineDocRetriever()
        self.llm = LlamaInference()
    
    def process_sensor_data(self, sensor_data, machine_id):
        """
        Complete pipeline: Sensors → ML → LLM → Explanation
        
        Args:
            sensor_data: Dict with sensor readings
            machine_id: Machine identifier
        
        Returns:
            Human-readable maintenance explanation
        """
        # Step 1: ML Predictions
        ml_predictions = self._get_ml_predictions(sensor_data)
        
        # Step 2: RAG Retrieval
        rag_context = self._retrieve_context(ml_predictions, machine_id)
        
        # Step 3: LLM Explanation
        explanation = self._generate_explanation(
            ml_predictions, rag_context, machine_id
        )
        
        return {
            'ml_predictions': ml_predictions,
            'explanation': explanation,
            'timestamp': pd.Timestamp.now()
        }
    
    def _get_ml_predictions(self, sensor_data):
        """Run all ML models"""
        features = self._preprocess(sensor_data)
        
        return {
            'classification': {
                'failure_type': self.classifier.predict(features)[0],
                'probability': self.classifier.predict_proba(features).max()
            },
            'rul': {
                'hours': self.regressor.predict(features)[0],
                'days': self.regressor.predict(features)[0] / 24
            },
            'anomaly': {
                'score': self.anomaly_detector.decision_function(features)[0],
                'is_anomaly': self.anomaly_detector.predict(features)[0] == -1
            }
        }
    
    def _retrieve_context(self, ml_predictions, machine_id):
        """RAG context retrieval"""
        query = f"{ml_predictions['classification']['failure_type']} symptoms"
        results = self.retriever.retrieve(query, machine_id, top_k=3)
        return "\n".join([r['doc'][:300] for r in results])
    
    def _generate_explanation(self, ml_predictions, rag_context, machine_id):
        """LLM explanation generation"""
        user_message = FAILURE_CLASSIFICATION_PROMPT.format(
            machine_id=machine_id,
            failure_probability=ml_predictions['classification']['probability'],
            failure_type=ml_predictions['classification']['failure_type'],
            confidence=ml_predictions['classification']['probability'],
            sensor_readings="...",  # Format from ml_predictions
            rag_context=rag_context
        )
        
        return self.llm.generate(SYSTEM_PROMPT, user_message)


# Usage in production
if __name__ == "__main__":
    pipeline = ProductionPipeline()
    
    # Real sensor data from Raspberry Pi
    sensor_data = {
        'vibration': 12.5,
        'temperature': 78.0,
        'current': 42.3,
        'voltage': 405.2
    }
    
    result = pipeline.process_sensor_data(
        sensor_data, 
        machine_id="motor_siemens_1la7_001"
    )
    
    print(result['explanation'])
```

---

## Data Flow Summary

### Current (Phase 3.3.1)
```
generate_failure_cases.py
    ↓
failure_cases.json (100 synthetic examples)
    ↓
Used for LLM training/testing
```

### Production (Phase 3.5+)
```
Raspberry Pi Sensors
    ↓
ML Models (Phase 2)
    ↓
ProductionPipeline
    ↓
RAG + LLM (Phase 3)
    ↓
Human Explanation
    ↓
User Interface
```

---

## Validation: Is This Fully Automated?

✅ **YES!** Here's the complete automation:

1. **Data Collection:** Raspberry Pi automatically reads sensors (no human)
2. **ML Predictions:** Models automatically predict failures (no human)
3. **RAG Retrieval:** Automatically fetches relevant docs (no human)
4. **LLM Generation:** Automatically generates explanation (no human)
5. **User Display:** Automatically shows in dashboard (no human)

**Human Only Needed For:**
- Reading the explanation
- Deciding when to schedule maintenance
- Performing actual maintenance work

---

## Next Steps

**Phase 3.3.2:** Embed these 100 failure cases in FAISS
**Phase 3.4:** Create prompt templates for each ML model type
**Phase 3.5:** Build the ProductionPipeline (integration layer)
**Phase 3.6:** Test end-to-end with real ML predictions
**Phase 3.7:** Deploy FastAPI server + web UI
**Phase 3.8:** Monitor and improve

---

## Current Progress

- ✅ Phase 3.1: RAG infrastructure (27 machine docs)
- ✅ Phase 3.2: Llama 3.1 8B installed (CPU mode)
- ✅ Phase 3.3.1: 100 synthetic failure cases generated
- ⏳ Phase 3.3.2: Embed failure cases (next)
- ⏳ Phase 3.4: Prompt engineering
- ⏳ Phase 3.5: ML integration (PRODUCTION PIPELINE)

**Note:** Phase 3.5 is where synthetic data gets replaced with real ML predictions!
