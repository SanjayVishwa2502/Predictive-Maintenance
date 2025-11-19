# Quick Start - RUL & Phase 1.5 Tasks
**⚡ Fast reference for colleague working on GAN enhancements**

---

## 🎯 Your Two Tasks

### Task 1: Add RUL Labels (CRITICAL - 3 days)
**Problem:** No RUL (Remaining Useful Life) labels in synthetic data  
**Solution:** Add 'rul' column to all 21 machines

### Task 2: Phase 1.5 Scripts (2-3 days)
**Problem:** No automated workflow for new machines  
**Solution:** Create onboarding automation

---

## 🚀 Quick Commands

### Setup (First Time)
```powershell
cd "C:\Projects\Predictive Maintenance\GAN"
& "..\venv\Scripts\activate"
pip install pandas numpy pyarrow  # If not already installed
```

### Task 1: Generate RUL Labels
```powershell
# Test on 1 machine first
python scripts/add_rul_to_datasets.py --machine_id motor_siemens_1la7_001

# If successful, process all 21 machines
python scripts/add_rul_to_datasets.py
```

### Task 2: Test New Machine Workflow
```powershell
# After creating scripts, test end-to-end
python scripts/add_new_machine.py --machine_id test_motor_001
```

---

## 📁 Files You'll Create

### RUL Task:
- `scripts/add_rul_to_datasets.py` (main script)
- `scripts/add_rul_metadata.py` (update metadata)

### Phase 1.5 Task:
- `PHASE_1.5_NEW_MACHINE_GUIDE.md` (documentation)
- `scripts/add_new_machine.py` (automation)
- `scripts/validate_new_machine.py` (validation)
- `templates/machine_metadata_template.json` (template)

**📖 Full implementation code is in:** `COLLEAGUE_HANDOFF_RUL_AND_PHASE_1.5.md`

---

## ✅ Validation Checklist

### Before You Send Back:

**RUL Task:**
- [ ] All 21 machines have 'rul' column
- [ ] RUL values: 0 to max_rul (no negatives/NaN)
- [ ] Report generated: `reports/rul_generation_report.json`
- [ ] Tested on 2 machines manually

**Phase 1.5 Task:**
- [ ] Successfully onboarded 1 test machine
- [ ] All 4 scripts created and working
- [ ] Documentation complete
- [ ] No errors in test run

---

## 🚨 Critical Rules

### ✅ DO:
- Work ONLY in `GAN/` folder
- Test on 1 machine first
- Create backups (automatic in script)
- Use GPU for TVAE training (`cuda=True`)

### ❌ DON'T:
- Touch `ml_models/` directory
- Delete synthetic data without backup
- Change paths or structure
- Skip validation steps

---

## 📊 Expected Results

### RUL Statistics (Example: Motor)
```
Min RUL:    0.00 hours (end of life)
Max RUL: 1000.00 hours (brand new)
Mean RUL:  481.47 hours (mid-life)
Std Dev:   290.15 hours (realistic variance)
```

### Processing Time (Your RTX 4060)
- Single machine RUL: ~2 seconds
- All 21 machines: ~1-2 hours (mostly I/O)
- TVAE training (new machine): ~2-3 minutes

---

## 🆘 Troubleshooting

### Issue: CUDA out of memory
```python
# In add_new_machine.py, reduce:
batch_size=250  # Instead of 500
```

### Issue: Quality score < 0.75
```python
# Increase training:
epochs=500  # Instead of 300
```

### Issue: RUL values look wrong
```python
# Check sensor detection in output
# Should see: "Temperature: 5 columns, Vibration: 3 columns"
# If missing, adjust column detection logic
```

---

## 📞 Questions?

**Full Documentation:**
- Main guide: `COLLEAGUE_HANDOFF_RUL_AND_PHASE_1.5.md`
- Status tracking: `PHASE_STATUS_AND_BLOCKERS.md`
- Original GAN doc: `PHASE_1_GAN_DETAILED_APPROACH.md`

**Contact:** Via Copilot session (same account)

---

## ⏱️ Timeline

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | RUL implementation | Scripts created |
| 3 | RUL batch processing | All 21 machines done |
| 4-5 | Phase 1.5 scripts | Automation complete |
| 6 | Testing & validation | End-to-end test |
| 7 | **Send back complete GAN/ folder** | ✅ Done |

**Your deadline: ~1 week**

---

**🎯 Goal:** Add RUL to all datasets + automate new machine onboarding  
**📦 Deliverable:** Complete GAN/ folder with RUL labels  
**🚀 Impact:** Unblocks regression training (Phase 2.3.1)

Good luck! 🍀
