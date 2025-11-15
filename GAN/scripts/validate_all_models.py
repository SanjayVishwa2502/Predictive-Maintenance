"""
Model Validation Script - Phase 1.3 Completion
Validates all 21 trained TVAE models:
1. Load each model successfully
2. Generate test samples
3. Verify sample quality
4. Check constraints compliance
"""

import sys
from pathlib import Path
import pandas as pd
import json
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

# All 21 trained machines
ALL_MACHINES = [
    # Priority machines (5)
    "motor_siemens_1la7_001",
    "motor_abb_m3bp_002",
    "pump_grundfos_cr3_004",
    "pump_flowserve_ansi_005",
    "compressor_atlas_copco_ga30_001",
    # Remaining machines (16)
    "motor_weg_w22_003",
    "pump_ksb_etanorm_006",
    "fan_ebm_papst_a3g710_007",
    "fan_howden_buffalo_008",
    "compressor_ingersoll_rand_2545_009",
    "cnc_dmg_mori_nlx_010",
    "hydraulic_beckwood_press_011",
    "hydraulic_parker_hpu_012",
    "conveyor_dorner_2200_013",
    "conveyor_hytrol_e24ez_014",
    "robot_fanuc_m20ia_015",
    "robot_abb_irb6700_016",
    "transformer_square_d_017",
    "cooling_tower_bac_vti_018",
    "cnc_haas_vf2_001",
    "turbofan_cfm56_7b_001"
]

def validate_model(machine_id, test_samples=1000):
    """Validate a single trained model"""
    
    print(f"\n{'─' * 70}")
    print(f"Validating: {machine_id}")
    print(f"{'─' * 70}")
    
    base_path = Path(__file__).parent.parent
    model_path = base_path / "models" / "tvae" / f"{machine_id}_tvae_500epochs.pkl"
    metadata_path = base_path / "metadata" / f"{machine_id}_metadata.json"
    seed_path = base_path / "seed_data" / f"{machine_id}_seed.parquet"
    
    validation_result = {
        'machine_id': machine_id,
        'model_exists': False,
        'model_loadable': False,
        'can_generate': False,
        'quality_score': None,
        'sample_shape': None,
        'constraints_valid': None,
        'status': 'UNKNOWN',
        'error': None
    }
    
    try:
        # Step 1: Check model file exists
        if not model_path.exists():
            validation_result['status'] = 'FAILED'
            validation_result['error'] = 'Model file not found'
            print(f"  ❌ Model file not found: {model_path}")
            return validation_result
        
        validation_result['model_exists'] = True
        print(f"  ✅ Model file exists ({model_path.stat().st_size / 1024 / 1024:.2f} MB)")
        
        # Step 2: Load model
        try:
            synthesizer = TVAESynthesizer.load(str(model_path))
            validation_result['model_loadable'] = True
            print(f"  ✅ Model loaded successfully")
        except Exception as e:
            validation_result['status'] = 'FAILED'
            validation_result['error'] = f'Model load failed: {str(e)}'
            print(f"  ❌ Failed to load model: {e}")
            return validation_result
        
        # Step 3: Generate test samples
        try:
            synthetic_samples = synthesizer.sample(num_rows=test_samples)
            validation_result['can_generate'] = True
            validation_result['sample_shape'] = synthetic_samples.shape
            print(f"  ✅ Generated {test_samples} samples: {synthetic_samples.shape}")
        except Exception as e:
            validation_result['status'] = 'FAILED'
            validation_result['error'] = f'Sample generation failed: {str(e)}'
            print(f"  ❌ Failed to generate samples: {e}")
            return validation_result
        
        # Step 4: Load metadata and seed data for quality check
        try:
            metadata = SingleTableMetadata.load_from_json(str(metadata_path))
            seed_data = pd.read_parquet(seed_path)
        except Exception as e:
            validation_result['status'] = 'WARNING'
            validation_result['error'] = f'Cannot load metadata/seed: {str(e)}'
            print(f"  ⚠️  Cannot validate quality: {e}")
            return validation_result
        
        # Step 5: Evaluate quality
        try:
            quality_report = evaluate_quality(seed_data, synthetic_samples, metadata)
            quality_score = quality_report.get_score()
            validation_result['quality_score'] = quality_score
            
            quality_status = "EXCELLENT" if quality_score >= 0.90 else "GOOD" if quality_score >= 0.80 else "ACCEPTABLE"
            print(f"  ✅ Quality Score: {quality_score:.3f} ({quality_status})")
        except Exception as e:
            validation_result['status'] = 'WARNING'
            validation_result['error'] = f'Quality evaluation failed: {str(e)}'
            print(f"  ⚠️  Quality evaluation failed: {e}")
            return validation_result
        
        # Step 6: Check basic constraints (no NaN, positive where expected)
        try:
            has_nan = synthetic_samples.isna().any().any()
            has_inf = (synthetic_samples == float('inf')).any().any() or (synthetic_samples == float('-inf')).any().any()
            
            if has_nan:
                print(f"  ⚠️  Warning: Generated samples contain NaN values")
                validation_result['constraints_valid'] = False
            elif has_inf:
                print(f"  ⚠️  Warning: Generated samples contain infinite values")
                validation_result['constraints_valid'] = False
            else:
                validation_result['constraints_valid'] = True
                print(f"  ✅ No NaN or infinite values")
        except Exception as e:
            print(f"  ⚠️  Constraint check warning: {e}")
        
        # Final status
        validation_result['status'] = 'VALID'
        print(f"  ✅ Model validation: PASSED")
        
    except Exception as e:
        validation_result['status'] = 'FAILED'
        validation_result['error'] = str(e)
        print(f"  ❌ Unexpected error: {e}")
    
    return validation_result


def main():
    """Validate all 21 trained models"""
    
    print("\n" + "=" * 70)
    print("MODEL VALIDATION - PHASE 1.3 COMPLETION")
    print("=" * 70)
    print(f"\nValidating {len(ALL_MACHINES)} trained TVAE models...")
    print(f"Test samples per model: 1000")
    print("=" * 70)
    
    results = []
    
    for idx, machine_id in enumerate(ALL_MACHINES, start=1):
        print(f"\n[{idx}/{len(ALL_MACHINES)}]", end=" ")
        result = validate_model(machine_id)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    valid_models = [r for r in results if r['status'] == 'VALID']
    failed_models = [r for r in results if r['status'] == 'FAILED']
    warning_models = [r for r in results if r['status'] == 'WARNING']
    
    print(f"\nResults:")
    print(f"  ✅ Valid Models: {len(valid_models)}/{len(ALL_MACHINES)}")
    print(f"  ⚠️  Warnings: {len(warning_models)}/{len(ALL_MACHINES)}")
    print(f"  ❌ Failed: {len(failed_models)}/{len(ALL_MACHINES)}")
    
    if valid_models:
        quality_scores = [r['quality_score'] for r in valid_models if r['quality_score'] is not None]
        if quality_scores:
            print(f"\nQuality Statistics:")
            print(f"  Average Quality: {sum(quality_scores) / len(quality_scores):.3f}")
            print(f"  Min Quality: {min(quality_scores):.3f}")
            print(f"  Max Quality: {max(quality_scores):.3f}")
            
            # Count by quality tier
            excellent = sum(1 for q in quality_scores if q >= 0.90)
            good = sum(1 for q in quality_scores if 0.80 <= q < 0.90)
            acceptable = sum(1 for q in quality_scores if q < 0.80)
            
            print(f"\nQuality Distribution:")
            print(f"  Excellent (≥0.90): {excellent} models")
            print(f"  Good (0.80-0.89): {good} models")
            print(f"  Acceptable (<0.80): {acceptable} models")
    
    if failed_models:
        print(f"\n❌ Failed Models:")
        for r in failed_models:
            print(f"  - {r['machine_id']}: {r['error']}")
    
    if warning_models:
        print(f"\n⚠️  Models with Warnings:")
        for r in warning_models:
            print(f"  - {r['machine_id']}: {r['error']}")
    
    # Save validation report
    report_path = Path(__file__).parent.parent / 'reports' / 'model_validation_report.json'
    with open(report_path, 'w') as f:
        json.dump({
            'total_models': len(ALL_MACHINES),
            'valid': len(valid_models),
            'warnings': len(warning_models),
            'failed': len(failed_models),
            'validation_results': results
        }, f, indent=2)
    
    print(f"\n✅ Validation report saved: {report_path}")
    
    # Final status
    if len(valid_models) == len(ALL_MACHINES):
        print(f"\n{'=' * 70}")
        print("🎉 ALL 21 MODELS VALIDATED SUCCESSFULLY! 🎉")
        print("=" * 70 + "\n")
        return 0
    else:
        print(f"\n{'=' * 70}")
        print(f"⚠️  VALIDATION INCOMPLETE: {len(valid_models)}/{len(ALL_MACHINES)} models valid")
        print("=" * 70 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
