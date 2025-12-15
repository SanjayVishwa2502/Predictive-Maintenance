# 🏭 GAN BACKEND INTEGRATION REFERENCE
**Frontend Module:** GAN Manager
**Date:** November 27, 2025
**Purpose:** Complete mapping of GAN scripts to Frontend UI actions

---

## 1. 📋 Overview

This document serves as the **definitive reference** for integrating the GAN subsystem into the Frontend Dashboard. It lists every script in `GAN/scripts/`, its purpose, parameters, outputs, and how it should be exposed in the UI.

**Directory Structure:**
```
GAN/
├── scripts/                    # All backend Python scripts
├── metadata/                   # Machine profile JSON files
├── seed_data/                  # Physics-based CSV files
├── models/                     # Trained TVAE .pkl files
├── data/synthetic/             # Generated parquet datasets
└── validate_new_machine.py     # Root-level validation script
```

---

## 2. 🆕 NEW MACHINE WORKFLOW (Primary User Journey)

### 2.1 Step 1: Create Machine Profile

**Script:** `GAN/scripts/create_metadata.py`

**Purpose:** Generate a machine configuration JSON file defining sensors, operational parameters, and failure modes.

**Input Parameters:**
- `machine_id` (str): Unique identifier (e.g., "pump_atlas_copco_005")
- `machine_type` (str): Category (motor, pump, cnc, fan, compressor, conveyor, hydraulic, cooling_tower)
- `manufacturer` (str): Maker name
- `model` (str): Model number
- `sensors` (list): Array of sensor definitions
  - Each sensor: `{name, unit, type, description}`
- `operational_parameters` (dict): Rated power, speed, voltage, etc.
- `failure_modes` (list): Expected failure types with descriptions

**Output:**
- File: `GAN/metadata/{machine_id}_metadata.json`

**Frontend Implementation:**
```python
# UI: Multi-step form with dynamic sensor addition
def create_machine_profile(machine_config: dict):
    """
    Args:
        machine_config: {
            'machine_id': 'pump_005',
            'machine_type': 'pump',
            'sensors': [
                {'name': 'vibration_rms', 'unit': 'mm/s', 'type': 'numerical'},
                {'name': 'temperature', 'unit': 'C', 'type': 'numerical'}
            ],
            ...
        }
    """
    subprocess.run([
        'python', 'GAN/scripts/create_metadata.py',
        '--machine_id', machine_config['machine_id'],
        '--machine_type', machine_config['machine_type'],
        '--config', json.dumps(machine_config)
    ])
```

**UI Elements:**
- Text Input: Machine ID
- Dropdown: Machine Type (predefined list)
- Dynamic List: "Add Sensor" button to build sensor array
- Submit Button: "Create Profile"

---

### 2.2 Step 2: Generate Physics-Based Seed Data

**Script:** `GAN/scripts/generate_seed_from_profile.py`

**Purpose:** Create initial degradation patterns using physics-based algorithms (exponential decay for RUL, Brownian motion for sensors).

**Input Parameters:**
- `machine_id` (str): The machine to generate seed for
- `num_samples` (int): Default 10,000

**Output:**
- File: `GAN/seed_data/{machine_id}_temporal_seed.csv`
- Columns: `timestamp`, `rul`, `sensor_1`, `sensor_2`, ..., `failure_mode`

**Frontend Implementation:**
```python
def generate_seed_data(machine_id: str, num_samples: int = 10000):
    """
    Triggers seed generation subprocess.
    Returns: Success boolean and path to CSV file.
    """
    result = subprocess.run([
        'python', 'GAN/scripts/generate_seed_from_profile.py',
        '--machine_id', machine_id,
        '--num_samples', str(num_samples)
    ], capture_output=True, text=True)
    
    return result.returncode == 0, result.stdout
```

**UI Elements:**
- Button: "Generate Seed Data"
- Number Input: Sample Count (default 10,000)
- Status Indicator: ✅ "Seed data created: 10,000 rows"

---

### 2.3 Step 3: Train TVAE Model

**Script:** `GAN/scripts/train_tvae_machine.py`

**Purpose:** Train a Temporal Variational Autoencoder on the seed data to learn sensor correlations and temporal patterns.

**Input Parameters:**
- `machine_id` (str): The machine to train for
- `epochs` (int): Training iterations (default 300)
- `batch_size` (int): Default 500

**Output:**
- File: `GAN/models/{machine_id}_tvae_temporal.pkl`
- Training logs (stdout): Loss values per epoch

**Frontend Implementation:**
```python
def train_tvae_model(machine_id: str, epochs: int = 300):
    """
    Runs TVAE training with real-time progress tracking.
    Streams stdout to dashboard progress bar.
    """
    process = subprocess.Popen([
        'python', 'GAN/scripts/train_tvae_machine.py',
        '--machine_id', machine_id,
        '--epochs', str(epochs)
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Stream output line-by-line
    for line in process.stdout:
        if "Epoch" in line:
            # Parse: "Epoch 150/300 - Loss: 0.045"
            current_epoch = extract_epoch(line)
            yield current_epoch / epochs  # Return progress percentage
```

**UI Elements:**
- Button: "Start Training"
- Progress Bar: 0-100% with estimated time
- Log Viewer: Scrollable text area showing "Epoch 1/300..."
- Training Config: Slider for Epochs (100-500)

**Special Handling:**
- **Long-Running Process:** Use `st.spinner` or background thread
- **GPU Alert:** Display warning if CUDA not available

---

### 2.4 Step 4: Generate Synthetic Dataset

**Script:** `GAN/scripts/generate_synthetic_data.py`

**Purpose:** Use the trained TVAE model to generate large-scale synthetic datasets (35K train, 7.5K val, 7.5K test).

**Input Parameters:**
- `machine_id` (str): The machine to generate data for
- `num_samples` (int): Total samples to generate (default 50,000)
- `split_ratio` (tuple): Train/Val/Test split (default 0.7, 0.15, 0.15)

**Output:**
- Files:
  - `GAN/data/synthetic/{machine_id}/train.parquet`
  - `GAN/data/synthetic/{machine_id}/val.parquet`
  - `GAN/data/synthetic/{machine_id}/test.parquet`

**Frontend Implementation:**
```python
def generate_synthetic_data(machine_id: str, num_samples: int = 50000):
    """
    Generates synthetic data using trained TVAE model.
    Returns: Dict with file paths and row counts.
    """
    subprocess.run([
        'python', 'GAN/scripts/generate_synthetic_data.py',
        '--machine_id', machine_id,
        '--num_samples', str(num_samples)
    ])
    
    # Verify output files
    train_path = f'GAN/data/synthetic/{machine_id}/train.parquet'
    if Path(train_path).exists():
        df = pd.read_parquet(train_path)
        return {'success': True, 'train_rows': len(df)}
```

**UI Elements:**
- Button: "Generate Dataset"
- Number Input: Total Samples (default 50,000)
- File Status Table:
  - train.parquet: ✅ 35,000 rows
  - val.parquet: ✅ 7,500 rows
  - test.parquet: ✅ 7,500 rows

---

### 2.5 Step 5: Validate Data Quality

**Script:** `GAN/validate_new_machine.py`

**Purpose:** Run quality checks on the generated data (timestamp monotonicity, RUL decay, feature distributions).

**Input Parameters:**
- `machine_id` (str): The machine to validate

**Output:**
- Console output with Pass/Fail status
- Metrics: RUL decay percentage, timestamp sorting, feature ranges

**Frontend Implementation:**
```python
def validate_machine_data(machine_id: str):
    """
    Runs validation checks and parses output.
    Returns: Dict with validation results.
    """
    result = subprocess.run([
        'python', 'GAN/validate_new_machine.py',
        '--machine_id', machine_id
    ], capture_output=True, text=True)
    
    # Parse output
    output = result.stdout
    validation_status = {
        'timestamp_sorted': '✅' in output and 'Timestamp sorted: True' in output,
        'rul_decreasing': extract_percentage(output, 'RUL decreasing'),
        'overall_pass': '✅ PASS' in output
    }
    return validation_status
```

**UI Elements:**
- Button: "Validate Data"
- Results Card:
  - ✅ Timestamp Sorted
  - ✅ RUL Decreasing (98.5%)
  - ✅ Feature Ranges Valid
  - Overall Status: **PASS**

---

## 3. 📊 DATA MANAGEMENT (Admin Functions)

### 3.1 Batch Generation

**Script:** `GAN/scripts/generate_all_synthetic_data.py`

**Purpose:** Regenerate datasets for all machines in the fleet (used after GAN code updates).

**Input:** None (reads all metadata files)

**Frontend Implementation:**
```python
def regenerate_all_data():
    """
    Admin function: Regenerate all 26 machines.
    WARNING: Takes ~2-3 hours.
    """
    subprocess.Popen([
        'python', 'GAN/scripts/generate_all_synthetic_data.py'
    ])
```

**UI Elements:**
- Button: "Regenerate All Data" (Admin Panel, with confirmation dialog)
- Warning: "This will take 2-3 hours"

---

### 3.2 Batch Training

**Script:** `GAN/scripts/train_all_remaining_machines.py`

**Purpose:** Train TVAE models for any metadata files that don't have models yet.

**Frontend Implementation:**
```python
def train_remaining_machines():
    """
    Finds metadata files without models and trains them.
    """
    subprocess.run([
        'python', 'GAN/scripts/train_all_remaining_machines.py'
    ])
```

**UI Elements:**
- Button: "Train Missing Models"
- Discovery List: Shows machines needing training (e.g., "Found 3 machines without models")

---

### 3.3 System Health Check

**Script:** `GAN/scripts/validate_all_models.py`

**Purpose:** Check if all machines have valid metadata, models, and data files.

**Frontend Implementation:**
```python
def check_system_health():
    """
    Returns: Dict with status for each machine.
    {
        'motor_001': {'metadata': True, 'model': True, 'data': True},
        'pump_002': {'metadata': True, 'model': False, 'data': False}
    }
    """
    result = subprocess.run([
        'python', 'GAN/scripts/validate_all_models.py'
    ], capture_output=True, text=True)
    
    # Parse output into structured format
    return parse_validation_output(result.stdout)
```

**UI Elements:**
- Dashboard Widget: "System Health"
- Grid showing all 26 machines with status icons:
  - 🟢 Fully Ready
  - 🟡 Model Missing
  - 🔴 Data Missing

---

## 4. 📈 DATA VISUALIZATION (Read-Only)

### 4.1 Read Machine Profile

**File:** `GAN/metadata/{machine_id}_metadata.json`

**Frontend Implementation:**
```python
def load_machine_profile(machine_id: str):
    """
    Loads and parses the JSON metadata.
    Returns: Dict with machine specs.
    """
    with open(f'GAN/metadata/{machine_id}_metadata.json') as f:
        return json.load(f)
```

**UI Elements:**
- Machine Detail Page: Display card showing manufacturer, model, sensors

---

### 4.2 Read Synthetic Data

**File:** `GAN/data/synthetic/{machine_id}/train.parquet`

**Frontend Implementation:**
```python
def load_synthetic_data(machine_id: str, split: str = 'train'):
    """
    Loads parquet file for plotting.
    Args:
        split: 'train', 'val', or 'test'
    Returns: DataFrame
    """
    path = f'GAN/data/synthetic/{machine_id}/{split}.parquet'
    return pd.read_parquet(path)
```

**UI Elements:**
- Live Charts: Plot sensor values over time
- Data Table: Show recent 100 rows

---

## 5. 🔧 HELPER SCRIPTS (Background Tasks)

### 5.1 Test ML Readiness

**Script:** `GAN/scripts/test_ml_readiness.py`

**Purpose:** Check if a machine's data is ready for ML model training (correct columns, no NaNs).

**Usage:** Called internally before passing data to ML training scripts.

---

### 5.2 Validate Temporal Seed

**Script:** `GAN/scripts/validate_temporal_seed_data.py`

**Purpose:** Check seed data quality before TVAE training.

**Usage:** Automatically called after Step 2 (Seed Generation).

---

## 6. 🎯 FRONTEND IMPLEMENTATION CHECKLIST

- [ ] **GAN Manager Module** (`gan_manager.py`): Python class wrapping all subprocess calls
- [ ] **New Machine Wizard** (5-page form)
- [ ] **Admin Panel** (Batch operations)
- [ ] **System Health Widget** (Status grid)
- [ ] **Data Viewer** (Parquet reader + Plotly charts)
- [ ] **Progress Tracking** (Real-time stdout parsing)

---

## 7. 📦 Dependencies

**Python Packages:**
- `subprocess` (stdlib)
- `pandas` (reading parquet)
- `json` (reading metadata)
- `pathlib` (file handling)
- `streamlit` (UI framework)

**Backend Requirements:**
- All GAN scripts must be executable from the project root
- Python environment must have `sdv`, `pandas`, `numpy` installed

---

## 8. 🚨 Error Handling

**Common Scenarios:**
1. **Script Not Found:** Check if `GAN/scripts/{script}.py` exists
2. **CUDA Error:** TVAE training fails without GPU → Show fallback message
3. **File Not Found:** Metadata missing → Redirect to "Create Profile" step
4. **Invalid Data:** Validation fails → Show error details and "Regenerate" button

**Frontend Pattern:**
```python
try:
    result = subprocess.run([...], check=True, capture_output=True, text=True)
    st.success("Operation completed!")
except subprocess.CalledProcessError as e:
    st.error(f"Script failed: {e.stderr}")
```

---

**End of Document**
