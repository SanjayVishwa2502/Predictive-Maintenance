"""
Comprehensive Machine Data Verification Script
Checks all machines for seed data, models, and synthetic data
Shows EXACTLY what exists and what the paths are
"""

from pathlib import Path
import json

# Project paths
PROJECT_ROOT = Path(__file__).parent
GAN_ROOT = PROJECT_ROOT / "GAN"

# Data directories
PROFILES_DIR = GAN_ROOT / "data" / "real_machines" / "profiles"
METADATA_DIR = GAN_ROOT / "metadata"
SEED_DATA_DIR = GAN_ROOT / "seed_data" / "temporal"
MODELS_DIR = GAN_ROOT / "models" / "tvae" / "temporal"
SYNTHETIC_DIR = GAN_ROOT / "data" / "synthetic"

print("=" * 80)
print("COMPREHENSIVE MACHINE DATA VERIFICATION")
print("=" * 80)
print(f"\n📁 GAN Root: {GAN_ROOT}")
print(f"📁 Profiles: {PROFILES_DIR}")
print(f"📁 Metadata: {METADATA_DIR}")
print(f"📁 Seed Data: {SEED_DATA_DIR}")
print(f"📁 Models: {MODELS_DIR}")
print(f"📁 Synthetic: {SYNTHETIC_DIR}")
print("\n" + "=" * 80)

# Get all machine profiles
profile_files = sorted(PROFILES_DIR.glob("*.json"))
print(f"\n✅ Found {len(profile_files)} machine profiles\n")

# Check each machine
results = []
for profile_path in profile_files:
    machine_id = profile_path.stem
    
    # Check all artifacts
    metadata_file = METADATA_DIR / f"{machine_id}_metadata.json"
    seed_file = SEED_DATA_DIR / f"{machine_id}_temporal_seed.parquet"
    model_files = list(MODELS_DIR.glob(f"{machine_id}_tvae_temporal_*.pkl"))
    synthetic_machine_dir = SYNTHETIC_DIR / machine_id
    
    # Check synthetic data files
    train_file = synthetic_machine_dir / "train.parquet"
    val_file = synthetic_machine_dir / "val.parquet"
    test_file = synthetic_machine_dir / "test.parquet"
    
    has_metadata = metadata_file.exists()
    has_seed = seed_file.exists()
    has_model = len(model_files) > 0
    has_synthetic = train_file.exists() and val_file.exists() and test_file.exists()
    
    # Determine completion status
    if has_metadata and has_seed and has_model and has_synthetic:
        status = "✅ COMPLETE"
        status_code = 4
    elif has_metadata and has_seed and has_model:
        status = "⚠️  NEEDS SYNTHETIC DATA"
        status_code = 3
    elif has_metadata and has_seed:
        status = "⚠️  NEEDS TRAINING"
        status_code = 2
    elif has_metadata:
        status = "⚠️  NEEDS SEED DATA"
        status_code = 1
    else:
        status = "❌ NO METADATA"
        status_code = 0
    
    results.append({
        'machine_id': machine_id,
        'status': status,
        'status_code': status_code,
        'has_metadata': has_metadata,
        'has_seed': has_seed,
        'has_model': has_model,
        'model_count': len(model_files),
        'has_synthetic': has_synthetic,
        'metadata_path': str(metadata_file) if has_metadata else None,
        'seed_path': str(seed_file) if has_seed else None,
        'model_paths': [str(m) for m in model_files] if model_files else [],
        'synthetic_path': str(synthetic_machine_dir) if has_synthetic else None,
    })

# Sort by status (complete first, then by machine_id)
results.sort(key=lambda x: (-x['status_code'], x['machine_id']))

# Print summary
print("\n" + "=" * 80)
print("MACHINE STATUS SUMMARY")
print("=" * 80)

complete = sum(1 for r in results if r['status_code'] == 4)
need_synthetic = sum(1 for r in results if r['status_code'] == 3)
need_training = sum(1 for r in results if r['status_code'] == 2)
need_seed = sum(1 for r in results if r['status_code'] == 1)
no_metadata = sum(1 for r in results if r['status_code'] == 0)

print(f"\n✅ Complete (all 4 stages):        {complete:2d} machines")
print(f"⚠️  Need Synthetic Data:           {need_synthetic:2d} machines")
print(f"⚠️  Need Training:                 {need_training:2d} machines")
print(f"⚠️  Need Seed Data:                {need_seed:2d} machines")
print(f"❌ No Metadata:                    {no_metadata:2d} machines")
print(f"\n📊 Total:                          {len(results):2d} machines")

# Detailed listing
print("\n" + "=" * 80)
print("DETAILED MACHINE STATUS")
print("=" * 80)

for i, r in enumerate(results, 1):
    print(f"\n{i:2d}. {r['status']} | {r['machine_id']}")
    print(f"    Metadata: {'✓' if r['has_metadata'] else '✗'}")
    print(f"    Seed Data: {'✓' if r['has_seed'] else '✗'}")
    print(f"    TVAE Model: {'✓' if r['has_model'] else '✗'} ({r['model_count']} files)")
    print(f"    Synthetic Data: {'✓' if r['has_synthetic'] else '✗'}")
    
    if r['model_paths']:
        for model_path in r['model_paths']:
            epochs = model_path.split('_')[-1].replace('epochs.pkl', '')
            print(f"        Model: {epochs} epochs")

# Show incomplete machines that need attention
incomplete = [r for r in results if r['status_code'] < 4]
if incomplete:
    print("\n" + "=" * 80)
    print("MACHINES NEEDING ATTENTION")
    print("=" * 80)
    
    for r in incomplete:
        print(f"\n🔧 {r['machine_id']}")
        if not r['has_seed']:
            print(f"   → Run: Generate Seed Data (50,000 samples)")
        elif not r['has_model']:
            print(f"   → Run: Train TVAE Model (500 epochs)")
        elif not r['has_synthetic']:
            print(f"   → Run: Generate Synthetic Data (70/15/15 split)")

# Export to JSON for dashboard use
output_file = PROJECT_ROOT / "machine_verification_report.json"
with open(output_file, 'w') as f:
    json.dump({
        'summary': {
            'total': len(results),
            'complete': complete,
            'need_synthetic': need_synthetic,
            'need_training': need_training,
            'need_seed': need_seed,
            'no_metadata': no_metadata,
        },
        'machines': results,
    }, f, indent=2)

print("\n" + "=" * 80)
print(f"📄 Full report saved to: {output_file}")
print("=" * 80)
