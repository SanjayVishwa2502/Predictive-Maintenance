# Training Script Organization

This folder contains multiple training approaches optimized for different deployment scenarios.

---

## 🚀 Current PoC Deployment (Raspberry Pi 4)

### Active Scripts:
- **`train_classification_fast.py`** ⭐ **CURRENT USE**
  - Target: Raspberry Pi 4 (ARM CPU, 4-8 GB RAM, no GPU)
  - Time: 15 minutes per machine
  - Models: LightGBM, RandomForest, XGBoost, CatBoost only
  - Model size: 5-10 MB per machine
  - Excluded: Neural networks (NN_TORCH, FASTAI, XT, KNN)
  - Configuration: `config/fast_training_config.py`
  - Documentation: `RASPBERRY_PI_POC_GUIDE.md`

- **`batch_train_classification.py`** ⭐ **CURRENT USE**
  - Batch training for 10 priority machines
  - Uses `train_classification_fast.py` internally
  - Total time: ~2.5 hours (10 machines × 15 min)

---

## 🔥 Future Production Deployment (NVIDIA Jetson Orin Nano)

### Future Scripts (When Scaling to Production):
- **`train_classification_full.py`** (NOT YET CREATED)
  - Target: Jetson Orin Nano (ARM CPU + GPU, 8 GB RAM, CUDA support)
  - Time: 60 minutes per machine (best quality)
  - Models: ALL models including neural networks (NN_TORCH, FASTAI)
  - Model size: 50-100 MB per machine
  - Presets: 'best_quality' for maximum accuracy
  - GPU acceleration: ENABLED
  - Expected F1: >0.95 (higher than Pi models)
  
- **`train_classification_per_machine_gpu.py`** (NOT YET CREATED)
  - Same as above but with GPU-optimized hyperparameters
  - Batch size tuning for Jetson GPU
  - Mixed precision training (FP16)
  - TensorRT optimization

---

## 📊 Comparison: Raspberry Pi vs Jetson Orin Nano

| Feature | **Raspberry Pi 4** (PoC) | **Jetson Orin Nano** (Production) |
|---------|--------------------------|-----------------------------------|
| **CPU** | ARM Cortex-A72 (1.5 GHz) | ARM Cortex-A78AE (2.0 GHz) |
| **GPU** | ❌ None | ✅ NVIDIA Ampere (1024 CUDA cores) |
| **RAM** | 4-8 GB | 8 GB |
| **CUDA** | ❌ Not supported | ✅ Supported |
| **Training Time** | 15 min/machine | 60 min/machine |
| **Model Types** | Tree models only | All models (including NNs) |
| **Model Size** | 5-10 MB | 50-100 MB |
| **Accuracy** | F1 >0.90 | F1 >0.95 |
| **Inference** | 20-50 ms | 5-10 ms |
| **Power** | 5-10W | 15-25W |
| **Cost** | $75-100 | $400-500 |

---

## 🎯 Recommended Migration Path

### Phase 1: PoC on Raspberry Pi (CURRENT)
```powershell
# Use fast training (15 min/machine)
python scripts/train_classification_fast.py --machine_id motor_siemens_1la7_001

# Batch training for 10 machines
python scripts/batch_train_classification.py
```
- **Goal:** Validate concept with lightweight models
- **Timeline:** 2.5 hours total training
- **Deployment:** Raspberry Pi 4
- **Models:** Tree-based only (5-10 MB)

### Phase 2: Production on Jetson Orin Nano (FUTURE)
```powershell
# Create full training script (when ready)
# python scripts/train_classification_full.py --machine_id motor_siemens_1la7_001

# Use GPU acceleration + neural networks
# python scripts/batch_train_classification_gpu.py
```
- **Goal:** Maximum accuracy for production deployment
- **Timeline:** 10 hours total training (60 min/machine)
- **Deployment:** Jetson Orin Nano
- **Models:** All models including neural networks (50-100 MB)

---

## 📝 Notes for Future Development

### When to Create Full Training Scripts:
1. ✅ **PoC Successful** - Raspberry Pi models validated in field
2. ✅ **Budget Approved** - Jetson Orin Nano hardware procured
3. ✅ **Accuracy Requirements** - Need F1 >0.95 (higher than Pi)
4. ✅ **Inference Speed** - Need <10ms latency (vs 20-50ms on Pi)

### What to Include in Full Scripts:
- Enable neural networks: Remove `excluded_model_types`
- GPU configuration: `num_gpus=1`, CUDA optimization
- Best quality preset: `presets='best_quality'`
- Longer training: `time_limit=3600` (60 minutes)
- Stacking: `num_stack_levels=1` for ensemble
- Bagging: `num_bag_folds=5-8` for better generalization
- TensorRT: Post-training optimization for Jetson
- Mixed precision: FP16 training for speed

### Configuration Files to Add:
- `config/jetson_training_config.py` - GPU-optimized settings
- `config/tensorrt_config.py` - Inference optimization
- `JETSON_DEPLOYMENT_GUIDE.md` - Deployment instructions

---

## 🔧 Current File Structure

```
scripts/
├── README_TRAINING_APPROACHES.md          ⭐ This file
├── train_classification_fast.py          ⭐ Current (Pi 4)
├── batch_train_classification.py         ⭐ Current (Pi 4)
├── feature_engineering.py                ✅ Shared
├── validate_setup_phase_2_2_2.py         ✅ Validation
└── [Future GPU scripts will go here]     🔮 When scaling

config/
├── fast_training_config.py               ⭐ Current (Pi 4)
├── model_config.py                       ✅ Shared
└── [jetson_training_config.py]           🔮 Future

docs/
├── RASPBERRY_PI_POC_GUIDE.md             ⭐ Current
└── [JETSON_DEPLOYMENT_GUIDE.md]          🔮 Future
```

---

## ✅ Summary

- **Current Focus:** Raspberry Pi 4 PoC with fast, lightweight models
- **Active Scripts:** `train_classification_fast.py` + `batch_train_classification.py`
- **Future Path:** Jetson Orin Nano with full models (neural networks + GPU)
- **No Scripts Deleted:** Fast scripts remain for Pi, full scripts will be added later
- **Clear Separation:** Each approach documented with target hardware and use case

**Next Steps:**
1. Complete Pi PoC training (2.5 hours)
2. Validate in field
3. If successful → Create Jetson scripts for production scaling
