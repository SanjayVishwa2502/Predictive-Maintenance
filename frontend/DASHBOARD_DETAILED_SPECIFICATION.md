# 📱 PREDICTIVE MAINTENANCE DASHBOARD: DETAILED SPECIFICATION
**Phase 3.7: Comprehensive UI/UX & Functional Requirements**
**Date:** November 27, 2025
**Version:** 1.0

---

## 1. 📖 Introduction & Scope

This document serves as the **Master Specification** for the Predictive Maintenance Dashboard. It translates the high-level architecture into granular functional requirements, user interface elements, and interaction flows.

**Primary Goal:** Provide a unified interface for:
1.  **Monitoring:** Real-time health status of the 26-machine fleet.
2.  **Diagnosis:** AI-driven insights (ML predictions + LLM explanations).
3.  **Management:** Onboarding new machines via the GAN workflow.

**Target Audience:**
*   **Maintenance Technician:** Focuses on "Which machine is broken?" and "What do I do?".
*   **System Administrator:** Focuses on "Adding new machines" and "System health".

---

## 2. 👤 User Personas & Use Cases

### 2.1 Persona A: The Maintenance Technician (Alex)
*   **Goal:** Minimize downtime.
*   **Pain Point:** Overwhelmed by raw sensor data; needs actionable advice.
*   **Use Case 1 (Morning Check):** Opens dashboard, filters for "Critical" machines, reads the LLM explanation to understand the root cause, and schedules maintenance.
*   **Use Case 2 (Live Monitoring):** Watches a specific motor during a stress test (simulation), observing real-time vibration spikes.

### 2.2 Persona B: The System Architect (Sam)
*   **Goal:** Scale the system.
*   **Pain Point:** Manually running scripts to add new machines is error-prone.
*   **Use Case 3 (Onboarding):** Uses the "New Machine Wizard" to define a new CNC machine profile, triggers the GAN training, and validates the synthetic data quality visually.

---

## 3. 🖥️ Dashboard Modules & UI Elements

The application will be built using **Streamlit** with a sidebar navigation layout.

### 3.1 🎨 Global Elements
*   **Sidebar Navigation:**
    *   🏠 **Fleet Overview**
    *   🔍 **Machine Inspector**
    *   ➕ **New Machine Wizard**
    *   ⚙️ **Settings / Debug**
*   **Header:**
    *   System Status Indicator (Backend Connection).
    *   "Last Updated" Timestamp (syncs with 30s heartbeat).

---

### 3.2 🏠 Module 1: Fleet Overview (The "Command Center")

**Functional Requirement:** Display high-level status of all assets to allow quick triage.

**UI Elements:**
1.  **KPI Banner:**
    *   *Total Machines:* Count (e.g., 26).
    *   *Critical Alerts:* Count of machines with Failure Prob > 80% or RUL < 24h.
    *   *System Health:* Average Confidence Score of ML models.
2.  **Filter & Sort Toolbar:**
    *   *Search:* Text input for Machine ID.
    *   *Filter:* Dropdown (All, Critical, Warning, Healthy).
    *   *Sort By:* RUL (Ascending), Failure Prob (Descending).
3.  **Asset Grid (Card View):**
    *   Each card represents one machine.
    *   **Visuals:**
        *   **Header:** Machine ID & Type Icon (Motor, Pump, etc.).
        *   **Status Badge:** Color-coded (🔴 Critical, 🟡 Warning, 🟢 Healthy).
        *   **Mini-Sparkline:** 24h trend of the most critical sensor (e.g., Vibration).
        *   **Key Metric:** "RUL: 45 Days" or "Prob: 92%".
    *   **Interaction:** Clicking a card redirects to *Machine Inspector* with that machine selected.

**Data Source:**
*   Aggregated latest prediction from `IntegratedPredictionSystem` for all active machines.

---

### 3.3 🔍 Module 2: Machine Inspector (The "Diagnosis Engine")

**Functional Requirement:** Provide deep, explainable insights for a single asset.

**Layout:** Split View (Left: Data/Visuals, Right: AI Insights).

**Left Column: Real-Time Telemetry**
1.  **Live Sensor Charts:**
    *   *Library:* Plotly (interactive).
    *   *Content:* Multi-line chart showing key sensors (e.g., `bearing_de_temp`, `rms_velocity`) over the last window (e.g., 1 hour).
    *   *Update:* Appends new data point every 30s.
2.  **Digital Twin / Sensor Table:**
    *   Current values of all sensors with units.
    *   Highlight "Abnormal" sensors (flagged by Anomaly Detection model) in Red.

**Right Column: AI Intelligence**
1.  **Prediction Summary (The "What"):**
    *   **Failure Classification:**
        *   *Display:* Top predicted failure mode (e.g., "Bearing Wear").
        *   *Visual:* Probability Bar (0-100%).
    *   **RUL Estimation:**
        *   *Display:* Estimated Remaining Useful Life.
        *   *Visual:* Gauge Chart (Green -> Red zones).
    *   **Anomaly Status:**
        *   *Display:* "Normal" or "Anomaly Detected".
        *   *Visual:* Status Indicator.
2.  **LLM Explanation (The "Why"):**
    *   *Component:* Scrollable Text Area / Markdown.
    *   *Content:* The output from `MLExplainer`.
    *   *Format:* Structured text (Diagnosis, Reasoning, Recommendation).
    *   *Feature:* "Regenerate Explanation" button.
3.  **Forecast (The "Future"):**
    *   *Visual:* Prophet forecast plot (next 24h) for critical sensors.
    *   *Text:* Summary of expected trends.

**Simulation Controls (Footer):**
*   **Simulation Mode:** Toggle (Real-time Stream vs. Manual Step).
*   **Fault Injection:** Buttons to force specific sensor patterns (e.g., "Simulate Overheating") to test model response.

---

### 3.4 ➕ Module 3: New Machine Wizard (The "Builder")

**Functional Requirement:** GUI wrapper for the GAN workflow (`GAN/WORKFLOW_TEST_NEW_MACHINE.md`).

**Step-by-Step Wizard Flow:**

**Step 1: Profile Configuration**
*   **Inputs:**
    *   Machine ID (Text, unique).
    *   Machine Type (Dropdown: Motor, Pump, CNC, etc.).
    *   Manufacturer/Model (Text).
    *   Operational Params (Rated Power, Speed, Voltage).
*   **Sensor Definition:**
    *   Dynamic list to add sensors.
    *   Fields: Name, Unit, Type (Numerical/Categorical).
*   **Action:** Generates `metadata/{id}_metadata.json`.

**Step 2: Seed Generation (Physics)**
*   **Display:** Explanation of physics-based degradation patterns.
*   **Action:** Button "Generate Seed Data".
*   **Backend:** Runs `validate_new_machine.py` (Step 1 logic).
*   **Feedback:** Success message showing number of seed samples created.

**Step 3: Model Training (TVAE)**
*   **Display:** Training configuration (Epochs, Batch Size - with defaults).
*   **Action:** Button "Start Training".
*   **Backend:** Triggers TVAE training script.
*   **Feedback:**
    *   Real-time log stream (capturing stdout).
    *   Progress bar (estimated time).

**Step 4: Validation & Deployment**
*   **Action:** Button "Generate Synthetic Data".
*   **Backend:** Generates `train.parquet` / `test.parquet`.
*   **Visual Validation:** Display distribution plots (Real vs. Synthetic) using `sdmetrics` reports if available, or simple histograms.
*   **Final Action:** "Add to Fleet" (Updates the global machine list).

---

## 4. ⚙️ Technical Implementation Details

### 4.1 The "30-Second Heartbeat" (PoC Simulation)
Since we don't have live IoT sensors yet, we simulate them using the GAN data.

*   **Session State:**
    *   `st.session_state['simulation_index']`: Tracks the current row index in the parquet file.
    *   `st.session_state['last_update']`: Timestamp of last refresh.
*   **Logic:**
    *   Use `streamlit-autorefresh` component.
    *   Interval: 30,000ms.
    *   On Refresh: Increment `simulation_index`, read next row from `GAN/data/synthetic/{id}/test.parquet`, push to `IntegratedPredictionSystem`.

### 4.2 Data Structures
*   **Machine Registry:** A simple JSON/Dict loaded at startup listing all available machines and their metadata paths.
*   **History Buffer:** A deque (size ~100) in session state to store recent sensor values for plotting the live charts.

### 4.3 Error Handling
*   **Model Not Found:** If a machine is selected but no ML model exists, show a friendly "Model Training Required" state instead of crashing.
*   **LLM Timeout:** If Llama 3.1 takes too long (>30s), display "Analyzing..." and update asynchronously (or show cached result).

---

## 5. 📅 Development Phases (Micro-Roadmap)

1.  **Phase 3.7.1: Skeleton & Navigation**
    *   Setup `dashboard/` folder.
    *   Implement Sidebar and empty pages.
2.  **Phase 3.7.2: The Backend Bridge**
    *   Connect `IntegratedPredictionSystem` to Streamlit.
    *   Implement the Data Loader for Parquet files.
3.  **Phase 3.7.3: Machine Inspector (Core)**
    *   Build the Real-time charts.
    *   Display ML predictions.
    *   Display LLM text.
4.  **Phase 3.7.4: Fleet Overview**
    *   Build the aggregation logic.
    *   Create the Card Grid.
5.  **Phase 3.7.5: New Machine Wizard**
    *   Implement the multi-step form.
    *   Connect to GAN scripts.
