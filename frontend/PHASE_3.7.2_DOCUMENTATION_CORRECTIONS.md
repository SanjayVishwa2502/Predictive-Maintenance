# Phase 3.7.2 Documentation Corrections
## Based on GAN Integration Research

**Date:** December 15, 2025  
**Research Document:** GAN_INTEGRATION_RESEARCH.md  
**Status:** Ready for Implementation

---

## Critical Corrections Needed

### 1. Machine Count Update: 26 → 32 Machines

**Issue:** Documentation consistently mentions "26 machines" but RUL profiles contain **32 machines**

**Evidence:** `GAN/config/rul_profiles.py` defines:
- Motor: 7 machines (was 3)
- Pump: 3 machines
- Compressor: 3 machines (was 2)
- CNC: 8 machines (was 7)
- Fan: 2 machines
- Conveyor: 2 machines
- Robot: 2 machines
- Hydraulic: 2 machines
- Transformer: 1 machine
- Cooling Tower: 1 machine
- Turbofan: 1 machine

**Total: 32 machines**

**Files to Update:**
- [ ] Line 15: "Displays real-time predictions for entire fleet (26 machines)" → **32 machines**
- [ ] Line 62: "Phase 1: GAN data generation (26 machines operational)" → **32 machines**
- [ ] Line 79: "GAN/: 18 scripts operational, 26 machines validated" → **32 machines**
- [ ] Line 931: "All 26 machines display real-time health predictions" → **32 machines**
- [ ] Line 1487: "Dashboard displays all 26 machines" → **32 machines**
- [ ] Line 1516: "Test batch predictions for 26 machines" → **32 machines**
- [ ] Line 1775: "Add prediction grid (26 machines)" → **32 machines**
- [ ] Line 1781: "Prediction grid (6x5 grid for 26 machines)" → **Grid layout update needed**
- [ ] Line 1793: "Dashboard displays all 26 machines" → **32 machines**
- [ ] Line 2223: "Validate all 26 machines" → **32 machines**
- [ ] Line 2256: "Real-time predictions for all 26 machines" → **32 machines**
- [ ] Line 2482: "Unified view of all 26 machines" → **32 machines**
- [ ] Line 2486: "Real-time prediction grid (26 machines with health status)" → **32 machines**
- [ ] Line 2578: "All 4 prediction types with 26 machines" → **32 machines**
- [ ] Line 2844: "Prediction Coverage: 100% of fleet (26 machines)" → **32 machines**

**Grid Layout Correction:**
- Old: 6x5 grid = 30 slots (26 used)
- New: 8x4 grid = 32 slots (32 used) or 6x6 grid = 36 slots (32 used)

---

### 2. Path Structure Corrections

**Issue:** Incomplete path documentation for temporal subdirectories

**Current Documentation:** Missing temporal subdirectories  
**Actual Implementation:**
```
GAN/
├── seed_data/
│   └── temporal/                    ← MISSING IN DOCS
│       └── {machine_id}_temporal_seed.parquet
├── models/
│   └── tvae/
│       └── temporal/                ← MISSING IN DOCS
│           └── {machine_id}_tvae_temporal_{epochs}epochs.pkl
```

**Correction Needed:**
Add explicit note in Phase 3.7.2.1 (GAN Manager section):

```markdown
**Path Structure:**
- Seed Data: `GAN/seed_data/temporal/{machine_id}_temporal_seed.parquet`
- TVAE Models: `GAN/models/tvae/temporal/{machine_id}_tvae_temporal_{epochs}epochs.pkl`
- Synthetic Data: `GAN/data/synthetic/{machine_id}/train.parquet`
- Reports: `GAN/reports/generation/{machine_id}_report.json`

Note: All temporal data uses `temporal/` subdirectory for versioning.
```

---

### 3. Model Filename Pattern

**Issue:** Documentation doesn't mention epoch count in model filename

**Current:** `{machine_id}_tvae_temporal.pkl`  
**Actual:** `{machine_id}_tvae_temporal_{epochs}epochs.pkl`

**Examples:**
- `motor_siemens_1la7_001_tvae_temporal_300epochs.pkl`
- `cnc_haas_vf2_001_tvae_temporal_500epochs.pkl`

**Impact:** `get_model_metadata()` must use glob pattern matching

**Correction for Phase 3.7.2.1:**
```markdown
**Model Filename Convention:**
- Pattern: `{machine_id}_tvae_temporal_{epochs}epochs.pkl`
- Loading: Use glob pattern to find any epoch count
- Multiple models: Use most recent (by modification time)
```

---

### 4. Training Results - Loss Metric

**Issue:** Documentation shows `loss` field in `TVAEModelMetadata` but script doesn't return it

**Current Implementation:**
```python
@dataclass
class TVAEModelMetadata:
    loss: Optional[float]  # Always None - not returned by script
```

**Root Cause:** `retrain_machine_tvae_temporal()` returns results dict without `final_loss` key

**Correction Options:**

**Option A: Update Script (Recommended)**
```python
# In retrain_tvae_temporal.py
results = {
    ...
    'final_loss': final_epoch_loss,  # ADD THIS
    'training_time_seconds': train_time
}
```

**Option B: Parse from Logs**
```python
# In gan_manager.py
# Parse "Epoch 300/300, Loss: 0.0234" from training output
```

**Option C: Document as Not Available**
```markdown
**Note:** Loss metric not currently captured. Consider adding in future release.
```

**Recommended:** Option A (update script), fallback to Option C for now

---

### 5. Production Configuration Updates

**Issue:** TVAE config shows "21 machines" in comments

**Location:** `GAN/config/tvae_config.py` lines 28-34

**Current:**
```python
PRODUCTION_EXPECTATIONS = {
    'total_training_time_21_machines_minutes': 84,  # 21 machines?
    'total_storage_21_machines_mb': 21,
}
```

**Should Be:**
```python
PRODUCTION_EXPECTATIONS = {
    'total_training_time_32_machines_minutes': 128,  # 32 × 4 min
    'total_storage_32_machines_mb': 32,  # 32 × 1 MB
}
```

---

## Additional Corrections

### 6. Template System Status

**Issue:** Documentation plans template endpoints but `/GAN/templates/` is empty

**Phase 3.7.2.2 mentions:**
- GET `/api/gan/templates`
- GET `/api/gan/templates/{machine_type}`
- GET `/api/gan/templates/{machine_type}/download`

**Actual Status:**
- ⚠️ Templates directory exists but is empty
- ⚠️ Template files not created yet

**Correction:**
Add note to Phase 3.7.2.2:
```markdown
**Template System:**
- Status: Planned for Phase 3.7.5 (Future Scope)
- Priority: Medium (nice-to-have, not critical)
- Alternative: Users can reference existing machine metadata as examples
```

---

### 7. Database Schema - GANTrainingJobs Table

**Issue:** Phase 3.7.2 mentions database table but it's not in Phase 3.7.1.2

**Current:** Phase 3.7.1.2 created 5 tables (machines, gan_training_jobs, predictions, explanations, model_versions)

**Verification Needed:**
- [ ] Check if `gan_training_jobs` table exists in `db/models.py`
- [ ] Verify columns match Phase 3.7.2.2 specification
- [ ] Ensure foreign key to `machines` table

**Expected Schema:**
```sql
CREATE TABLE gan_training_jobs (
    id UUID PRIMARY KEY,
    machine_id VARCHAR(100) REFERENCES machines(machine_id),
    status VARCHAR(20),  -- pending, running, success, failed
    epochs INT,
    progress INT,  -- 0-100
    training_time_seconds FLOAT,
    quality_score FLOAT,
    model_path VARCHAR(500),
    celery_task_id UUID,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

### 8. Rate Limiting Configuration

**Issue:** Documentation mentions Redis-based rate limiting but doesn't specify implementation

**Phase 3.7.2.2 mentions:**
- Rate Limiting: 100 requests/minute per IP
- Implementation: Redis-based

**Missing Details:**
- Which Redis DB? (Recommend db=3 for rate limiting)
- TTL for rate limit keys?
- Which endpoints are rate limited?

**Correction:**
```markdown
**Rate Limiting Implementation:**
- Storage: Redis DB 3 (separate from broker/backend/pubsub)
- Limit: 100 requests per minute per IP address
- Window: Rolling 60-second window
- Key Format: `ratelimit:{endpoint}:{ip}:{minute}`
- TTL: 120 seconds (2x window for safety)
- Library: `slowapi` or custom Redis implementation

**Exempt Endpoints:**
- /health (health check)
- /ws/* (WebSocket connections)

**Rate Limited Endpoints:**
- All /api/gan/* endpoints
```

---

### 9. WebSocket Channel Naming

**Issue:** Documentation inconsistent about Redis channel naming

**Phase 3.7.2.3 mentions:**
- Channel: `gan:training:{task_id}`

**Phase 3.7.2.4 mentions:**
- Flexible channel prefix via query param

**Clarification Needed:**
```markdown
**Redis Channel Naming Convention:**
- Training: `gan:training:{task_id}`
- Generation: `gan:generation:{task_id}`
- Seed: `gan:seed:{task_id}` (optional, fast operation)
- Generic: `{prefix}:{task_id}` (via WebSocket query param)

**Redis Database:**
- Pub/Sub: DB 2 (dedicated for messaging)
- Separate from broker (DB 0) and backend (DB 1)
```

---

### 10. Progress Update Frequency

**Issue:** Documentation mentions "every 10 epochs" but doesn't specify for short trainings

**Current:** Broadcast progress every 10 epochs  
**Problem:** What if training is only 50 epochs total?

**Correction:**
```markdown
**Progress Broadcasting Frequency:**
- Every 10 epochs for epochs ≥ 100
- Every 5 epochs for epochs 50-99
- Every epoch for epochs < 50
- Or 10% intervals (every ceil(epochs/10))

**Implementation:**
```python
update_interval = max(1, epochs // 10)  # At least every epoch
if epoch % update_interval == 0:
    broadcast_progress(...)
```
```

---

## Summary of Changes

### High Priority
- [x] **Machine count: 26 → 32** (15 occurrences)
- [x] **Path structure documentation** (add temporal subdirectories)
- [x] **Model filename pattern** (add epochs suffix)

### Medium Priority
- [ ] **Production config updates** (training time calculations)
- [ ] **Loss metric handling** (document limitation or fix)
- [ ] **Database schema verification** (gan_training_jobs table)

### Low Priority
- [ ] **Template system status** (mark as future scope)
- [ ] **Rate limiting details** (Redis DB, implementation)
- [ ] **Progress frequency** (formula for different epoch counts)

---

## Implementation Checklist

Before starting Phase 3.7.2.2 implementation:

**Documentation:**
- [ ] Update all "26 machines" references to "32 machines"
- [ ] Update grid layout (6x5 → 8x4 or 6x6)
- [ ] Add temporal subdirectory notes
- [ ] Document model filename pattern
- [ ] Add production config corrections

**Code Verification:**
- [ ] Verify gan_training_jobs table exists
- [ ] Verify GANManager path handling
- [ ] Test model loading with glob pattern
- [ ] Verify all 32 machines have RUL profiles

**Testing:**
- [ ] Create test for each of 32 machines
- [ ] Test path consistency
- [ ] Test model filename matching
- [ ] Test database schema

---

## Recommendations for Phase 3.7.2 Implementation

### Use This Order:

1. **Day 8-9: GAN Manager (COMPLETE)** ✅
   - No changes needed
   - Already implemented correctly

2. **Day 10: API Routes - Profile Management**
   - POST `/api/gan/machines` (create from profile)
   - GET `/api/gan/machines` (list all)
   - GET `/api/gan/machines/{id}` (get details)
   - DELETE `/api/gan/machines/{id}` (delete)

3. **Day 10-11: API Routes - Workflow Operations**
   - POST `/api/gan/machines/{id}/seed` (generate seed)
   - POST `/api/gan/machines/{id}/train` (train TVAE - async)
   - POST `/api/gan/machines/{id}/generate` (generate synthetic)
   - GET `/api/gan/tasks/{task_id}` (task status)

4. **Day 11: API Routes - Testing & Documentation**
   - Integration tests
   - Rate limiting
   - OpenAPI documentation
   - Error handling

5. **Day 12: Celery Tasks (COMPLETE)** ✅
   - Already implemented
   - May need minor updates based on API changes

6. **Day 13: WebSocket (COMPLETE)** ✅
   - Already implemented
   - Working correctly

---

**End of Corrections Document**

All findings documented and ready for implementation of Phase 3.7.2.2 (API Routes).
