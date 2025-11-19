# Handoff Checklist - RUL & Phase 1.5
**For: Colleague (i7-14650HX + RTX 4060)**  
**Date: November 18, 2025**

---

## 📋 Pre-Work Checklist

Before you start coding:

- [ ] Read `COLLEAGUE_HANDOFF_RUL_AND_PHASE_1.5.md` (30 min)
- [ ] Read `QUICK_START_COLLEAGUE.md` (5 min)
- [ ] Understand the blocker: No RUL = No regression models
- [ ] Verify GPU working: `nvidia-smi` shows RTX 4060
- [ ] Activate venv: `& "..\venv\Scripts\activate"`
- [ ] Check existing data: `ls GAN/data/synthetic/*/train.parquet`

---

## ✅ Task 1: RUL Generation (Day 1-3)

### Day 1: Implementation
- [ ] Create `scripts/add_rul_to_datasets.py` (copy from handoff doc)
- [ ] Create `scripts/add_rul_metadata.py` (copy from handoff doc)
- [ ] Test on 1 machine: `python scripts/add_rul_to_datasets.py --machine_id motor_siemens_1la7_001`
- [ ] Validate output:
  - [ ] 'rul' column exists
  - [ ] RUL range: 0 to ~1000 hours
  - [ ] No NaN/negative values
  - [ ] Mean RUL ~500 hours

### Day 2: Testing & Refinement
- [ ] Test on 2 more machines (different types: pump, compressor)
- [ ] Check RUL makes sense:
  - [ ] Decreases over time (sorted)
  - [ ] Correlates with sensor readings
  - [ ] Has realistic variance
- [ ] Adjust algorithm if needed (sensor weights, noise)
- [ ] Document any changes in report

### Day 3: Batch Processing
- [ ] Run metadata update: `python scripts/add_rul_metadata.py`
- [ ] Batch process all 21 machines: `python scripts/add_rul_to_datasets.py`
- [ ] Expected time: 1-2 hours
- [ ] Verify report: `reports/rul_generation_report.json`
- [ ] Check for errors: Should show 21/21 successful

**Validation:**
```powershell
# Quick check on all machines
cd GAN/data/synthetic
Get-ChildItem -Directory | ForEach-Object {
    $df = python -c "import pandas as pd; df = pd.read_parquet('$($_.Name)/train.parquet'); print(f'{df.shape[1]} cols, RUL: {\"rul\" in df.columns}')"
    Write-Host "$($_.Name): $df"
}
# Expected: "24 cols, RUL: True" for all machines
```

---

## ✅ Task 2: Phase 1.5 Scripts (Day 4-6)

### Day 4: Core Scripts
- [ ] Create `PHASE_1.5_NEW_MACHINE_GUIDE.md` (copy from handoff doc)
- [ ] Create `scripts/add_new_machine.py` (main automation)
- [ ] Test structure: `python scripts/add_new_machine.py --help`
- [ ] Verify imports work (no errors)

### Day 5: Supporting Files
- [ ] Create `scripts/validate_new_machine.py`
- [ ] Create `templates/machine_metadata_template.json`
- [ ] Add example seed data for testing (optional)

### Day 6: End-to-End Testing
- [ ] Create test machine seed data:
  ```powershell
  # Copy from existing machine and rename
  cp seed_data/motor_siemens_1la7_001_seed.parquet seed_data/test_motor_001_seed.parquet
  ```
- [ ] Run full workflow: `python scripts/add_new_machine.py --machine_id test_motor_001`
- [ ] Expected output:
  - [ ] Metadata created
  - [ ] TVAE trained (2-3 min on your GPU)
  - [ ] Datasets generated (50K rows)
  - [ ] RUL labels added automatically
  - [ ] Quality score > 0.85
  - [ ] No errors
- [ ] Clean up test: `Remove-Item data/synthetic/test_motor_001 -Recurse`

---

## 📦 Final Deliverables Checklist

### Files Created (8 new files):
- [ ] `scripts/add_rul_to_datasets.py`
- [ ] `scripts/add_rul_metadata.py`
- [ ] `scripts/add_new_machine.py`
- [ ] `scripts/validate_new_machine.py`
- [ ] `PHASE_1.5_NEW_MACHINE_GUIDE.md`
- [ ] `templates/machine_metadata_template.json`
- [ ] `reports/rul_generation_report.json` (auto-generated)
- [ ] `reports/<test_machine>_onboarding_report.json` (auto-generated)

### Files Modified (21 machines):
- [ ] All metadata files: `metadata/*_metadata.json` (21 files)
  - [ ] Each has `rul_parameters` section
- [ ] All train datasets: `data/synthetic/*/train.parquet` (21 files)
  - [ ] Each has 'rul' column (24 columns total)
- [ ] All val datasets: `data/synthetic/*/val.parquet` (21 files)
  - [ ] Each has 'rul' column
- [ ] All test datasets: `data/synthetic/*/test.parquet` (21 files)
  - [ ] Each has 'rul' column

### Validation Passed:
- [ ] No errors in `rul_generation_report.json`
- [ ] All 21 machines show "successful": true
- [ ] Test machine onboarding completed successfully
- [ ] Quality scores maintained (>0.85 for all)
- [ ] RUL statistics look realistic (no outliers)

---

## 🧪 Quality Checks

### Manual Spot Checks (Pick 3 random machines):

**Machine 1: Motor**
```powershell
cd GAN/data/synthetic/motor_siemens_1la7_001
python -c "import pandas as pd; df = pd.read_parquet('train.parquet'); print(df.columns.tolist()); print(df['rul'].describe())"
```
Expected:
- 24 columns (23 sensors + rul)
- RUL: min=0, max~1000, mean~500

**Machine 2: Pump**
```powershell
cd GAN/data/synthetic/pump_grundfos_cr3_004
python -c "import pandas as pd; df = pd.read_parquet('train.parquet'); print(df['rul'].describe())"
```
Expected:
- RUL: min=0, max~800, mean~400

**Machine 3: Compressor**
```powershell
cd GAN/data/synthetic/compressor_atlas_copco_ga30_001
python -c "import pandas as pd; df = pd.read_parquet('train.parquet'); print(df['rul'].describe())"
```
Expected:
- RUL: min=0, max~1200, mean~600

### Automated Checks:

```powershell
# Run validation script
cd GAN/scripts
python validate_new_machine.py --check_all
```

Expected output:
```
Validating 21 machines...
✅ motor_siemens_1la7_001 - PASS
✅ motor_abb_m3bp_002 - PASS
✅ motor_weg_w22_003 - PASS
[... 18 more ...]
────────────────────────────────
Summary: 21/21 PASSED
```

---

## 📊 Performance Benchmarks (Your System)

Based on your i7-14650HX + RTX 4060:

| Task | Expected Time | GPU Usage |
|------|---------------|-----------|
| Single machine RUL | 2-3 seconds | 0% (CPU) |
| All 21 machines RUL | 1-2 hours | 0% (I/O bound) |
| TVAE training (new machine) | 2-3 minutes | 60-80% |
| Quality validation | 5-10 seconds | 0% (CPU) |
| Full onboarding | 3-4 minutes | 60-80% |

If slower: Check disk I/O, background processes.  
If CUDA errors: Reduce batch_size to 250.

---

## 🆘 Common Issues & Solutions

### Issue 1: "CUDA out of memory"
```python
# In add_new_machine.py, line ~80:
batch_size=250  # Reduce from 500
```

### Issue 2: "Quality score < 0.75"
```python
# In add_new_machine.py, line ~75:
epochs=500  # Increase from 300
```

### Issue 3: "RUL values all same"
```python
# In add_rul_to_datasets.py, check:
# - Sensor columns detected (should see list printed)
# - Time index calculated (np.linspace output)
# - Noise added (np.random.normal)
```

### Issue 4: "Import errors"
```powershell
# Ensure in GAN directory:
cd "C:\Projects\Predictive Maintenance\GAN"
# Activate venv:
& "..\venv\Scripts\activate"
# Check packages:
pip list | Select-String "pandas|numpy|sdv"
```

### Issue 5: "Path not found"
```python
# All scripts should use:
project_root = Path(__file__).parent.parent
# Not hardcoded paths like "C:\Projects\..."
```

---

## 📤 Handoff Process

### When You're Done (Day 7):

1. **Self-Review:**
   - [ ] Run all validation checks
   - [ ] Review `rul_generation_report.json`
   - [ ] Test new machine onboarding
   - [ ] No errors/warnings

2. **Package Preparation:**
   ```powershell
   # Create archive
   cd "C:\Projects\Predictive Maintenance"
   Compress-Archive -Path "GAN" -DestinationPath "GAN_with_RUL_$(Get-Date -Format 'yyyyMMdd').zip"
   ```

3. **Handoff Document:**
   Create `HANDOFF_NOTES.md` with:
   - What you changed
   - Any issues encountered
   - Test results
   - Recommendations

4. **Send to Original Developer:**
   - [ ] GAN/ folder (or zip)
   - [ ] HANDOFF_NOTES.md
   - [ ] `reports/rul_generation_report.json`
   - [ ] Confirmation all tests passed

---

## ✅ Success Criteria

You're done when:

1. **RUL Generation:**
   - ✅ All 21 machines have 'rul' column
   - ✅ RUL values range 0 to max_rul
   - ✅ No NaN, no negative values
   - ✅ Quality maintained (>0.85)
   - ✅ Report shows 21/21 successful

2. **Phase 1.5 Automation:**
   - ✅ All 4 scripts created
   - ✅ Documentation complete
   - ✅ Test machine onboarded successfully
   - ✅ No errors in workflow

3. **Validation:**
   - ✅ Spot checks passed (3+ machines)
   - ✅ Automated validation passed
   - ✅ Quality scores maintained
   - ✅ Performance benchmarks met

4. **Handoff:**
   - ✅ Package created
   - ✅ Handoff notes written
   - ✅ Ready to send back

---

## 📞 Need Help?

**Resources:**
1. Main guide: `COLLEAGUE_HANDOFF_RUL_AND_PHASE_1.5.md` (has all code)
2. Quick ref: `QUICK_START_COLLEAGUE.md`
3. Status: `PHASE_STATUS_AND_BLOCKERS.md`

**If stuck:**
1. Check troubleshooting in main guide
2. Test on 1 machine first
3. Review error messages carefully
4. Document issue in handoff notes

**Communication:**
- Daily update: "Completed X/21 machines"
- Report issues immediately
- Ask questions via Copilot

---

**Timeline:** 7 days  
**Your System:** More than capable (i7-14650HX + RTX 4060)  
**Complexity:** Medium (well-documented, code provided)  
**Impact:** CRITICAL (unblocks regression training)

**Good luck! 🚀**

---

_Last updated: November 18, 2025_  
_For questions, refer to main handoff document_
