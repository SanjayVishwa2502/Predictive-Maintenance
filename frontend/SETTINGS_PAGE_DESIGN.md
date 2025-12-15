# Settings Page Design Specification

## Overview
A centralized configuration hub for managing the entire Predictive Maintenance system, covering General application settings, GAN parameters, ML model thresholds, LLM integration, and System maintenance.

## Architecture
The settings page will be divided into 5 main tabs:

### 1. General Settings
*   **Theme & Display:**
    *   Dark/Light mode toggle.
    *   Density settings (Compact/Comfortable).
    *   Dashboard refresh rate (e.g., 30s, 1m, 5m).
*   **Notifications:**
    *   Enable/Disable toast notifications.
    *   Email alert configuration (SMTP settings).
    *   Alert severity thresholds (Info, Warning, Critical).

### 2. GAN Settings (Generative Adversarial Networks)
*   **Training Defaults:**
    *   Default Epochs (e.g., 300).
    *   Batch Size.
    *   Learning Rate.
*   **Data Generation:**
    *   Default Sample Sizes (Train/Test/Val split).
    *   Output format (Parquet/CSV).
*   **Validation Thresholds:**
    *   Minimum KS-Test Score for acceptance.
    *   Maximum Drift tolerance.

### 3. ML Settings (Machine Learning)
*   **Model Management:**
    *   Active Model Version selection.
    *   Auto-retraining triggers (e.g., "Retrain if accuracy drops below 85%").
*   **Prediction Thresholds:**
    *   "At Risk" probability threshold (e.g., > 0.7).
    *   "Failure" probability threshold (e.g., > 0.9).
    *   RUL Warning limit (e.g., < 48 hours).

### 4. LLM Settings (Large Language Models)
*   **Provider Config:**
    *   Local GPU vs Cloud API toggle.
    *   Model Selection (Llama 3, Mistral, etc.).
    *   Temperature / Max Tokens.
*   **Prompt Engineering:**
    *   System Prompt template editor.
    *   RAG Context window size.
*   **Performance:**
    *   GPU Memory limit.
    *   Token streaming toggle.

### 5. System Settings
*   **Maintenance:**
    *   Clear Cache (Redis/Browser).
    *   Prune old logs.
    *   Database backup trigger.
*   **User Management:**
    *   Manage Users & Roles (Admin, Operator, Viewer).
    *   API Key management.
*   **About:**
    *   System Version.
    *   Component Health Status.

## Implementation Plan
1.  **Store:** Create `settingsStore.ts` (Zustand) with persistence.
2.  **Component:** Create `SettingsPage.tsx` using MUI Tabs.
3.  **Backend:** Create `/api/settings` endpoints to persist critical configs to DB/File.
