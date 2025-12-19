# Machine Profile Validation System

## Overview

**Critical Security Feature**: Comprehensive validation system to prevent duplicate machines, data conflicts, and TVAE training errors.

**Status**: ✅ **IMPLEMENTED AND ACTIVE**  
**Date**: December 17, 2025  
**Component**: `frontend/server/api/services/profile_validator.py`

---

## Problem Statement

### Issue Reported by User
> "I uploaded a machine brother speedio that is already in our workflow and as you see it passed validation!! That should never happen. Already existing machines may create chaos inside it and so please formulate all the validation details according to the requirements by the TVAE for training. Such irregular validations may cause errors in the later training."

### Critical Risks Without Validation
1. **Duplicate Machines**: Same machine_id uploaded multiple times
2. **Data Conflicts**: Overlapping training data causing TVAE confusion
3. **Training Errors**: Invalid sensor configurations breaking TVAE
4. **System Chaos**: Inconsistent machine metadata across workflow stages

---

## Validation Architecture

### MachineProfileValidator Class

**Location**: `frontend/server/api/services/profile_validator.py`

**Key Features**:
- ✅ Duplicate machine detection across 4 GAN directories
- ✅ Required fields validation (machine_id, manufacturer, model, category)
- ✅ Sensor count validation (minimum 1, recommended 5+)
- ✅ TVAE compatibility checks
- ✅ Profile structure validation against template
- ✅ RUL parameter validation
- ✅ machine_id format validation

---

## Validation Rules

### 1. Machine ID Uniqueness (CRITICAL)

**Severity**: ERROR (blocking)

**Checks**:
- GAN/metadata/*.json (SDV metadata files)
- GAN/data/real_machines/profiles/*.json (machine profiles)
- GAN/data/synthetic/*/  (synthetic data directories)
- GAN/models/*/  (trained model directories)

**Error Message**:
```
CRITICAL: Machine ID 'cnc_brother_speedio_001' already exists in the system.
This would cause data conflicts and training errors.
Please use a different machine ID or update the existing machine.
```

**Example**:
- ❌ cnc_brother_speedio_001 (already exists)
- ✅ cnc_brother_speedio_002 (unique)

---

### 2. Required Fields

**Severity**: ERROR (blocking)

**Fields**:
- `machine_id`: Format: `<type>_<manufacturer>_<model>_<id>`
- `manufacturer`: Manufacturer name (e.g., "Siemens", "DMG MORI")
- `model`: Model number (e.g., "1LA7 113-4AA60", "SPEEDIO S700X1")
- `category`: Machine category (e.g., "motor", "pump", "cnc")

**Example Error**:
```json
{
  "severity": "error",
  "field": "manufacturer",
  "message": "Manufacturer name is required"
}
```

---

### 3. Sensor Configuration

**Severity**: ERROR if < 1 sensor, WARNING if < 5 sensors

**Requirements**:
- **Minimum**: 1 sensor (CRITICAL for TVAE)
- **Recommended**: 5+ sensors (better predictions)
- **Maximum**: 50 sensors (performance)

**Sensor Structure** (Option 1 - Simplified):
```json
{
  "sensors": [
    {
      "name": "bearing_temp_C",
      "min_value": 40,
      "max_value": 85,
      "unit": "°C"
    },
    {
      "name": "vibration_rms_mm_s",
      "min_value": 0.7,
      "max_value": 4.5,
      "unit": "mm/s"
    }
  ]
}
```

**Sensor Structure** (Option 2 - Baseline Operation):
```json
{
  "baseline_normal_operation": {
    "temperature": {
      "bearing_temp_C": {
        "min": 40,
        "typical": 55,
        "max": 70,
        "alarm": 85,
        "trip": 95,
        "unit": "°C"
      }
    },
    "vibration": {
      "overall_rms_mm_s": {
        "min": 0.7,
        "typical": 1.2,
        "max": 1.8,
        "unit": "mm/s"
      }
    }
  }
}
```

---

### 4. TVAE Compatibility

**Severity**: ERROR or WARNING depending on issue

**Checks**:
1. **RUL Range Validation**:
   - `rul_max` must be > `rul_min`
   - Example: rul_min=0, rul_max=1000 ✅

2. **Degradation States**:
   - Minimum: 2 states (ERROR if < 2)
   - Recommended: 3-5 states
   - Maximum: 10 states (WARNING if > 10)

3. **Numeric Sensors**:
   - All sensors must have numeric ranges
   - Required for TVAE normalization

**Example**:
```json
{
  "rul_min": 0,
  "rul_max": 1000,
  "degradation_states": 4
}
```

---

### 5. Machine ID Format

**Severity**: ERROR (blocking)

**Format**: `<type>_<manufacturer>_<model>_<sequential_id>`

**Rules**:
- Lowercase only
- Underscores to separate parts
- No consecutive underscores (`__`)
- No spaces
- Alphanumeric characters only

**Examples**:
- ✅ motor_siemens_1la7_001
- ✅ cnc_dmg_mori_nlx_010
- ✅ pump_grundfos_cr3_004
- ❌ Motor_Siemens_001 (not lowercase)
- ❌ motor__siemens_001 (consecutive underscores)
- ❌ motor siemens 001 (spaces)

---

## API Integration

### Upload Endpoint

**POST** `/api/gan/profiles/upload`

**Request Body** (ProfileUploadRequest):
```json
{
  "machine_id": "motor_abb_m3bp_005",
  "machine_type": "motor",
  "manufacturer": "ABB",
  "model": "M3BP 160 MLB 4",
  "sensors": [
    {
      "name": "winding_temp_C",
      "display_name": "Winding Temperature",
      "unit": "°C",
      "min_value": 20.0,
      "max_value": 120.0,
      "sensor_type": "temperature",
      "is_critical": true
    }
  ],
  "degradation_states": 4,
  "rul_min": 0,
  "rul_max": 1000
}
```

**Response** (ProfileUploadResponse):
```json
{
  "success": true,
  "profile_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "machine_id": "motor_abb_m3bp_005",
  "message": "Profile uploaded successfully. Validation required before machine creation.",
  "validation_required": true
}
```

---

### Validation Endpoint

**POST** `/api/gan/profiles/{profile_id}/validate`

**Request Body** (ProfileValidationRequest):
```json
{
  "profile_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "strict": true
}
```

**Response** (ProfileValidationResponse):
```json
{
  "valid": false,
  "profile_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "machine_id": "cnc_brother_speedio_001",
  "can_proceed": false,
  "message": "[ERROR] Validation failed with 1 errors and 1 warnings. Fix errors before proceeding.",
  "issues": [
    {
      "severity": "error",
      "field": "machine_id",
      "message": "CRITICAL: Machine ID 'cnc_brother_speedio_001' already exists in the system. This would cause data conflicts and training errors. Please use a different machine ID or update the existing machine."
    },
    {
      "severity": "warning",
      "field": "machine_id",
      "message": "Found similar machine IDs: cnc_brother_speedio_001. Verify this is not a duplicate or typo."
    }
  ]
}
```

---

### Creation Endpoint (Final Validation)

**POST** `/api/gan/machines?profile_id={profile_id}`

**Final Duplicate Check**: Even if validation is bypassed, creation endpoint performs final duplicate check:

```python
# CRITICAL: Final validation before creation
validator = MachineProfileValidator(gan_metadata_dir=metadata_path)

if machine_id in validator.existing_machines:
    raise HTTPException(
        status_code=400,
        detail="[DUPLICATE ERROR] Machine already exists in the system"
    )
```

---

## Test Results

### Test 1: Brother Speedio (Duplicate Detection)

**Input**:
```json
{
  "machine_id": "cnc_brother_speedio_001",
  "manufacturer": "Brother Industries",
  "model": "SPEEDIO S700X1",
  "category": "CNC Vertical Machining Center"
}
```

**Result**: ❌ FAIL (Expected)

**Issues**:
- 🔴 **ERROR**: Machine ID already exists in system
- ⚠️ **WARNING**: Found similar machine IDs
- ℹ️ **INFO**: Sensors will be extracted from baseline_normal_operation

**Outcome**: ✅ **Validation working correctly** - Prevented duplicate machine

---

### Test 2: Valid New Machine

**Input**:
```json
{
  "machine_id": "test_new_machine_999",
  "manufacturer": "Test Manufacturer",
  "model": "Test Model 123",
  "category": "test_equipment",
  "sensors": [
    {"name": "temp_sensor_1", "min_value": 0, "max_value": 100},
    {"name": "vibration_sensor_1", "min_value": 0, "max_value": 50},
    {"name": "pressure_sensor_1", "min_value": 0, "max_value": 200}
  ]
}
```

**Result**: ✅ PASS

**Issues**:
- ⚠️ **WARNING**: Only 3 sensors (recommend 5+)
- ℹ️ **INFO**: baseline_normal_operation section missing

**Outcome**: ✅ **Validation working correctly** - Allowed valid unique machine

---

### Test 3: Invalid Machine (Missing Required Fields)

**Input**:
```json
{
  "machine_id": "test_invalid_machine"
  // Missing: manufacturer, model, category, sensors
}
```

**Result**: ❌ FAIL (Expected)

**Issues**:
- 🔴 **ERROR**: Manufacturer name required
- 🔴 **ERROR**: Model number required
- 🔴 **ERROR**: Category required
- 🔴 **ERROR**: No sensors defined (CRITICAL for TVAE)

**Outcome**: ✅ **Validation working correctly** - Prevented invalid machine

---

## Existing Machines Detection

**System discovers 33 existing machines across 4 locations**:

### Locations Checked:
1. **GAN/metadata/** (29 machines with SDV metadata)
   - motor_siemens_1la7_001_metadata.json
   - cnc_dmg_mori_nlx_010_metadata.json
   - pump_grundfos_cr3_004_metadata.json
   - etc.

2. **GAN/data/real_machines/profiles/** (4 machines with full profiles)
   - cnc_brother_speedio_001.json ⚠️
   - (other profile files)

3. **GAN/data/synthetic/** (machines with synthetic data)
   - cnc_brother_speedio_001_synthetic_temporal/
   - (other synthetic data directories)

4. **GAN/models/** (machines with trained models)
   - (trained model directories)

**Total Unique Machines**: 33 (includes Brother Speedio)

---

## Validation Severity Levels

### 🔴 ERROR (Blocking)
- Machine cannot be created
- `can_proceed = false`
- Must fix before proceeding

**Examples**:
- Duplicate machine_id
- Missing required fields
- Invalid RUL range
- No sensors defined

### ⚠️ WARNING (Non-blocking)
- Machine can be created
- `can_proceed = true`
- Recommended to fix

**Examples**:
- Few sensors (< 5)
- Similar machine_ids found
- High sensor count (> 50)
- Missing RUL parameters

### ℹ️ INFO (Informational)
- Machine can be created
- `can_proceed = true`
- Optional improvements

**Examples**:
- Missing baseline_normal_operation
- Missing specifications
- Sensors extracted from alternative format

---

## Usage Guide

### For Developers

**Import and Use Validator**:
```python
from api.services.profile_validator import MachineProfileValidator

# Initialize
validator = MachineProfileValidator()

# Validate profile
is_valid, issues, can_proceed = validator.validate_profile(
    profile_data=profile_dict,
    strict=True
)

# Check result
if not is_valid:
    for issue in issues:
        if issue.severity == "error":
            print(f"ERROR: {issue.field} - {issue.message}")
```

### For API Users

**1. Upload Profile**:
```bash
POST /api/gan/profiles/upload
{
  "machine_id": "motor_new_001",
  "manufacturer": "Siemens",
  "model": "1LA7 090-4AA60",
  "sensors": [...]
}
```

**2. Validate Profile**:
```bash
POST /api/gan/profiles/{profile_id}/validate
{
  "profile_id": "uuid-from-upload",
  "strict": true
}
```

**3. Check Validation Result**:
- If `valid = true` → Proceed to creation
- If `valid = false` → Fix errors in `issues` array

**4. Create Machine** (only if validated):
```bash
POST /api/gan/machines?profile_id={profile_id}
```

---

## Template Compliance

### Machine Profile Template

**Reference**: `machine_profile_template (1).json`

**Required Sections**:
- ✅ machine_id
- ✅ manufacturer
- ✅ model
- ✅ category
- ✅ baseline_normal_operation OR sensors
- ⚠️ specifications (recommended)
- ⚠️ rul_min, rul_max (recommended)
- ⚠️ degradation_states (recommended)

**Flexible Parsing**:
- Accepts multiple JSON formats
- Auto-extracts sensors from baseline_normal_operation
- Infers units from sensor names
- Normalizes field names

---

## Error Prevention

### What Validation Prevents

1. **Duplicate Machine Chaos**:
   - ❌ Same machine_id uploaded twice
   - ✅ Comprehensive duplicate detection

2. **TVAE Training Errors**:
   - ❌ No sensors defined
   - ❌ Invalid RUL ranges
   - ❌ Non-numeric sensor data
   - ✅ TVAE compatibility validation

3. **Data Conflicts**:
   - ❌ Overlapping machine data
   - ❌ Inconsistent metadata
   - ✅ Strict field validation

4. **System Inconsistency**:
   - ❌ Invalid machine_id formats
   - ❌ Missing required fields
   - ✅ Format standardization

---

## Monitoring & Logging

### Validation Logs

**Backend Log** (`logs/backend_*.log`):
```
2025-12-17 10:22:50 - api.routes.gan - INFO - Validated profile uuid-123 for machine cnc_brother_speedio_001: FAIL
2025-12-17 10:22:51 - api.routes.gan - INFO - Validated profile uuid-456 for machine motor_new_001: PASS
```

### Validation Metrics

**To Monitor**:
- Validation pass rate
- Common error types
- Duplicate detection rate
- Sensor count distribution

---

## Future Enhancements

### Planned Improvements
1. ✅ Multi-location duplicate detection (DONE)
2. ✅ TVAE compatibility validation (DONE)
3. ⏳ Database integration for profile storage
4. ⏳ Real-time validation in upload form
5. ⏳ Validation history tracking
6. ⏳ Machine similarity scoring
7. ⏳ Auto-correction suggestions

---

## Summary

### ✅ Validation System Status

**Component Status**:
- ✅ MachineProfileValidator class: IMPLEMENTED
- ✅ Validation endpoint: ACTIVE
- ✅ Creation endpoint guard: ACTIVE
- ✅ Duplicate detection: WORKING (4 locations)
- ✅ Required fields check: WORKING
- ✅ TVAE compatibility: WORKING
- ✅ Test coverage: VERIFIED

**Test Results**:
- ✅ Brother Speedio duplicate: CORRECTLY REJECTED
- ✅ Valid new machine: CORRECTLY ACCEPTED
- ✅ Invalid machine: CORRECTLY REJECTED

**System Impact**:
- **Before**: Any machine could be uploaded (chaos risk)
- **After**: Only valid, unique machines allowed (system integrity)

**User Request**: ✅ **FULLY ADDRESSED**

The validation system now prevents irregular validations that could cause TVAE training errors, exactly as requested by the user.

---

**Documentation Version**: 1.0  
**Last Updated**: December 17, 2025  
**Maintained By**: Predictive Maintenance Team
