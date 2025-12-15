# 🏛️ PREDICTIVE MAINTENANCE SYSTEM: FULL INTEGRATION SPECIFICATION
**Phase 3.7: Dashboard & Backend Architecture**
**Date:** November 27, 2025
**Status:** Blueprint for Implementation

---

## 1. 🗺️ System Overview

This document provides a comprehensive mapping of the entire project codebase to the proposed Dashboard. It details how the Frontend (Streamlit) interacts with the three core backend subsystems: **GAN (Data Generation)**, **ML (Prediction)**, and **LLM (Explanation)**.

### 1.1 The "Traditional Frontend" Architecture
Unlike a simple API wrapper, this dashboard acts as a **System Orchestrator**. It directly interfaces with the file system and executes Python scripts to perform heavy lifting (training, generation), while using memory-resident classes for real-time inference.

```mermaid
graph TD
    User[👤 User] <--> Dashboard[🖥️ Dashboard (Streamlit)]
    
    subgraph "Subsystem 1: GAN (The Factory)"
        Dashboard -->|1. Config| Meta[📝 Metadata JSON]
        Dashboard -->|2. Exec| Seed[🌱 Seed Generator]
        Dashboard -->|3. Train| TVAE[🧠 TVAE Model]
        Dashboard -->|4. Gen| SynData[💾 Synthetic Parquet]
    end
    
    subgraph "Subsystem 2: ML (The Brain)"
        SynData -->|Input| TrainML[🏋️ Training Scripts]
        TrainML -->|Save| Models[📦 Pickle Models]
        Models -->|Load| Predictor[🔮 IntegratedPredictionSystem]
        Dashboard <-->|Real-time| Predictor
    end
    
    subgraph "Subsystem 3: LLM (The Voice)"
        Predictor -->|Context| RAG[📚 RAG Retriever]
        RAG -->|Prompt| Llama[🦙 Llama 3.1 Engine]
        Llama -->|Text| Predictor
    end
```

---

## 2. 🏭 Subsystem 1: GAN (New Machine Onboarding)

**Goal:** Allow users to add a new machine (e.g., "Pump_005") and generate training data for it without writing code.

### 2.1 Script Inventory & Dashboard Integration

| Step | Script Path | Functionality | Dashboard Action |
| :--- | :--- | :--- | :--- |
| **1. Profile** | `GAN/scripts/create_metadata.py` | Creates `{id}_metadata.json` defining sensors & physics. | **Form Input:** User fills "Machine Type", "Sensors". Dashboard saves JSON to `GAN/metadata/`. |
| **2. Seed** | `GAN/scripts/generate_seed_from_profile.py` | Generates physics-based degradation patterns (CSV). | **Button:** "Generate Seed". Dashboard runs `subprocess.run(['python', 'generate_seed...', '--id', machine_id])`. |
| **3. Train** | `GAN/scripts/train_tvae_machine.py` | Trains a TVAE model on the seed data. | **Button:** "Train GAN". Dashboard streams stdout to a progress bar. |
| **4. Generate** | `GAN/scripts/generate_synthetic_data.py` | Uses TVAE to generate 35k+ rows of synthetic data. | **Button:** "Generate Data". Dashboard waits for `train.parquet` to appear. |
| **5. Validate** | `GAN/validate_new_machine.py` | Checks data quality (monotonicity, RUL trends). | **Visual:** Dashboard parses the script output ("✅ PASS") and shows distribution plots. |

### 2.2 Data Flow (New Machine)
1.  **User** inputs: `Machine ID: pump_005`, `Sensors: [vibration, temp]`.
2.  **Dashboard** writes: `GAN/metadata/pump_005_metadata.json`.
3.  **Script** (`generate_seed`) reads JSON -> writes `GAN/seed_data/pump_005_seed.csv`.
4.  **Script** (`train_tvae`) reads CSV -> saves model `GAN/models/pump_005_tvae.pkl`.
5.  **Script** (`generate_synthetic`) uses PKL -> writes `GAN/data/synthetic/pump_005/train.parquet`.

---

## 3. 🧠 Subsystem 2: ML Models (Training & Inference)

**Goal:** Train predictive models on the GAN data and run real-time inference.

### 3.1 Training Integration (Admin Panel)

The dashboard will allow "Retraining" models when new data is generated.

| Model Type | Script Path | Dashboard Action |
| :--- | :--- | :--- |
| **Classification** | `ml_models/scripts/training/train_classification_fast.py` | **Button:** "Retrain Classifier". Runs script with `--machine_id`. |
| **Regression (RUL)** | `ml_models/scripts/training/train_regression_fast.py` | **Button:** "Retrain RUL". Runs script. |
| **Anomaly** | `ml_models/scripts/training/train_anomaly_comprehensive.py` | **Button:** "Retrain Anomaly". Runs script. |
| **Forecasting** | `ml_models/scripts/training/train_timeseries.py` | **Button:** "Retrain Forecast". Runs script. |

### 3.2 Inference Integration (Real-Time View)

The dashboard does **not** call scripts for inference (too slow). It imports the Python class directly.

*   **Core Class:** `IntegratedPredictionSystem`
*   **File:** `LLM/api/ml_integration.py`
*   **Usage:**
    ```python
    # Dashboard Code (app.py)
    from LLM.api.ml_integration import IntegratedPredictionSystem
    
    # Initialize once (caches models)
    if 'predictor' not in st.session_state:
        st.session_state.predictor = IntegratedPredictionSystem()
    
    # Run every 30s
    result = st.session_state.predictor.predict_with_explanation(
        machine_id="motor_001",
        sensor_data=current_sensor_row
    )
    ```

---

## 4. 🗣️ Subsystem 3: LLM & RAG (The Explainer)

**Goal:** Provide human-readable context to the ML predictions.

### 4.1 Component Breakdown

1.  **Llama Engine (`LLM/scripts/inference/llama_engine.py`):**
    *   **Role:** Wraps `llama-cpp-python`.
    *   **Dashboard Integration:** Loaded automatically by `IntegratedPredictionSystem`.
    *   **Hardware:** Uses the RTX 4070 (via CUDA DLL injection in `ml_integration.py`).

2.  **RAG Retriever (`LLM/scripts/rag/retriever.py`):**
    *   **Role:** Searches `LLM/data/embeddings/` (FAISS index) for machine manuals.
    *   **Input:** "High vibration in motor_001".
    *   **Output:** "Manual Section 4.2: Check bearing lubrication."

3.  **Explainer API (`LLM/api/explainer.py`):**
    *   **Role:** The "Prompt Engineer".
    *   **Logic:**
        1.  Receives ML Prediction (e.g., "Failure Prob: 85%").
        2.  Calls Retriever for context.
        3.  Fills the Prompt Template (`LLM/config/prompts.py`).
        4.  Calls Llama Engine.
        5.  Returns text to Dashboard.

---

## 5. 🖥️ Dashboard Implementation Plan

### 5.1 Directory Structure
We will create a clean `dashboard/` directory to house the frontend logic, keeping it separate from the heavy backend scripts.

```
dashboard/
├── app.py                  # Main Entry Point (Streamlit)
├── pages/
│   ├── 1_🏠_Fleet_Overview.py
│   ├── 2_🔍_Machine_Inspector.py
│   └── 3_➕_New_Machine.py
├── utils/
│   ├── backend.py          # Imports IntegratedPredictionSystem
│   ├── gan_manager.py      # subprocess calls to GAN scripts
│   ├── ml_manager.py       # subprocess calls to ML training scripts
│   └── data_loader.py      # Reads Parquet files for simulation
└── assets/                 # Logos, CSS
```

### 5.2 The "Simulation Loop"
Since we are in PoC mode (Phase 3.7), we simulate real-time data.

1.  **Data Source:** `GAN/data/synthetic/{machine_id}/test.parquet`.
2.  **Mechanism:**
    *   Dashboard loads the parquet file into memory (cached).
    *   A "Tick" counter increments every 30 seconds.
    *   Row `[Tick]` is extracted and sent to `IntegratedPredictionSystem`.
    *   The result (Prediction + Explanation) is displayed.

### 5.3 Visual Elements Mapping

| Dashboard Element | Source Script/Data |
| :--- | :--- |
| **Live Chart** | `GAN/data/synthetic/.../test.parquet` (Historical window) |
| **Failure Gauge** | `predict_classification.py` (via `IntegratedPredictionSystem`) |
| **RUL Countdown** | `predict_rul.py` (via `IntegratedPredictionSystem`) |
| **Anomaly Alert** | `predict_anomaly.py` (via `IntegratedPredictionSystem`) |
| **"AI Diagnosis" Text** | `LLM/api/explainer.py` -> `llama_engine.py` |
| **"Train" Progress Bar** | stdout from `GAN/scripts/train_tvae_machine.py` |

---

## 6. 🚀 Execution Strategy

1.  **Environment Setup:** Ensure `streamlit`, `plotly`, and `sdmetrics` are installed.
2.  **Backend Bridge:** Create `dashboard/utils/backend.py` to safely import the complex `LLM` modules without path errors.
3.  **Page 1 (Inspector):** Build the single-machine view first (highest value).
4.  **Page 2 (Fleet):** Build the aggregation view.
5.  **Page 3 (Onboarding):** Implement the form-to-script wiring.

This architecture ensures the Dashboard is a **true interface** to the underlying powerful engines, not just a static display.
