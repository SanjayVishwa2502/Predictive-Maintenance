# Phase 2.2.3: Classification Model Validation Report

**Validation Date:** November 21, 2025

## Executive Summary

- **Total Models Validated:** 10
- **Models ≥ 0.70 F1:** 10/10 (100%)
- **Models ≥ 0.85 F1:** 2/10 (20%)
- **Average F1 Score:** 0.7695
- **Average Accuracy:** 0.9460
- **F1 Score Range:** 0.7173 - 0.8598
- **Total Model Size:** 2583.81 MB (2.52 GB)
- **Total Training Time:** 6.30 minutes (0.10 hours)
- **Pi-Compatible Models:** 9/10 (90%)

## Performance by Machine

| Machine | F1 Score | Accuracy | Precision | Recall | Size (MB) | Training (min) | Pi-Compatible | Best Model |
|---------|----------|----------|-----------|--------|-----------|----------------|---------------|------------|
| 🥇 compressor\_atlas\_copco\_ga30\_001 | 0.8598 | 0.9491 | 0.9436 | 0.7896 | 242.34 | 0.72 | ✅ | RandomForestGini_BAG_L1 |
| 🥇 motor\_siemens\_1la7\_001 | 0.8548 | 0.9460 | 0.9430 | 0.7816 | 255.93 | 0.89 | ✅ | LightGBM_BAG_L1 |
| ✅ hydraulic\_beckwood\_press\_011 | 0.8486 | 0.9443 | 0.9257 | 0.7833 | 262.06 | 0.60 | ✅ | LightGBMLarge_BAG_L1 |
| ✅ motor\_abb\_m3bp\_002 | 0.7598 | 0.9519 | 0.9533 | 0.6316 | 237.58 | 0.69 | ✅ | RandomForestEntr_BAG_L1 |
| ✅ pump\_flowserve\_ansi\_005 | 0.7432 | 0.9469 | 0.9260 | 0.6207 | 257.22 | 0.65 | ✅ | RandomForestEntr_BAG_L1 |
| ✅ pump\_grundfos\_cr3\_004 | 0.7427 | 0.9468 | 0.9474 | 0.6108 | 231.34 | 0.65 | ❌ | WeightedEnsemble_L2 |
| ✅ cnc\_dmg\_mori\_nlx\_010 | 0.7273 | 0.9448 | 0.9452 | 0.5910 | 294.92 | 0.42 | ✅ | LightGBM_BAG_L1 |
| ✅ motor\_weg\_w22\_003 | 0.7230 | 0.9423 | 0.8870 | 0.6102 | 246.69 | 0.70 | ✅ | LightGBMLarge_BAG_L1 |
| ✅ compressor\_ingersoll\_rand\_2545\_009 | 0.7184 | 0.9444 | 0.9500 | 0.5776 | 251.40 | 0.52 | ✅ | LightGBMXT_BAG_L1 |
| ✅ cooling\_tower\_bac\_vti\_018 | 0.7173 | 0.9435 | 0.9260 | 0.5854 | 304.33 | 0.46 | ✅ | LightGBM_BAG_L1 |

## Top 3 Performing Models

🥇 **compressor_atlas_copco_ga30_001**
- F1 Score: 0.8598
- Accuracy: 0.9491
- Training Time: 0.72 minutes
- Model Size: 242.34 MB
- Best Model: RandomForestGini_BAG_L1

🥈 **motor_siemens_1la7_001**
- F1 Score: 0.8548
- Accuracy: 0.9460
- Training Time: 0.89 minutes
- Model Size: 255.93 MB
- Best Model: LightGBM_BAG_L1

🥉 **hydraulic_beckwood_press_011**
- F1 Score: 0.8486
- Accuracy: 0.9443
- Training Time: 0.60 minutes
- Model Size: 262.06 MB
- Best Model: LightGBMLarge_BAG_L1


## Summary

✅ **All 10 models meet the minimum F1 ≥ 0.70 requirement**

- Average F1: 0.7695 (exceeds 0.70 by 9.9%)
- Training efficiency: 6.30 minutes total
- Pi-compatible: 9/10 models
- Storage usage: 2.52 GB

## Next Steps

1. ✅ **Phase 2.2.3 Complete** - All 10 classification models validated
2. 🔄 **Phase 2.3** - Train regression models (RUL prediction)
3. 🔄 **Phase 2.4** - Train anomaly detection models
4. 🔄 **Phase 2.5** - Train time-series forecasting models
5. 🔄 **Phase 2.6** - Edge optimization (model compression)

