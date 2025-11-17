# ml_models/scripts/validate_setup_phase_2_2_2.py
# Validation script for Phase 2.2.2: Per-Machine Classification Training
# Checks all prerequisites before starting training

from pathlib import Path
import sys
import pandas as pd

def validate_priority_machines_file():
    """Validate priority_10_machines.txt exists and is correctly formatted"""
    print("\n1. Checking priority_10_machines.txt...")
    
    machines_file = Path('config/priority_10_machines.txt')
    if not machines_file.exists():
        print("   ❌ File not found: config/priority_10_machines.txt")
        return False, []
    
    with open(machines_file, 'r') as f:
        machines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"   ✅ Found {len(machines)} machines")
    for i, machine_id in enumerate(machines, 1):
        print(f"      {i}. {machine_id}")
    
    return True, machines

def validate_synthetic_data(machines):
    """Validate synthetic data exists for all machines"""
    print("\n2. Checking synthetic data availability...")
    
    gan_data_path = Path('../GAN/data/synthetic')
    if not gan_data_path.exists():
        print(f"   ❌ GAN data directory not found: {gan_data_path}")
        return False
    
    missing_machines = []
    valid_machines = []
    
    for machine_id in machines:
        machine_dir = gan_data_path / machine_id
        
        # Check if directory exists
        if not machine_dir.exists():
            missing_machines.append(machine_id)
            print(f"   ❌ {machine_id}: Directory not found")
            continue
        
        # Check required files
        required_files = ['train.parquet', 'val.parquet', 'test.parquet']
        missing_files = []
        
        for file_name in required_files:
            if not (machine_dir / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            missing_machines.append(machine_id)
            print(f"   ❌ {machine_id}: Missing files: {', '.join(missing_files)}")
        else:
            # Check data shapes
            try:
                train_df = pd.read_parquet(machine_dir / 'train.parquet')
                val_df = pd.read_parquet(machine_dir / 'val.parquet')
                test_df = pd.read_parquet(machine_dir / 'test.parquet')
                
                print(f"   ✅ {machine_id}: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}, features={len(train_df.columns)}")
                valid_machines.append(machine_id)
            except Exception as e:
                missing_machines.append(machine_id)
                print(f"   ❌ {machine_id}: Error loading data: {str(e)}")
    
    if missing_machines:
        print(f"\n   ⚠️ {len(missing_machines)} machines have data issues:")
        for machine_id in missing_machines:
            print(f"      - {machine_id}")
        return False
    
    print(f"\n   ✅ All {len(valid_machines)} machines have valid synthetic data")
    return True

def validate_scripts():
    """Validate training scripts exist"""
    print("\n3. Checking training scripts...")
    
    required_scripts = [
        'scripts/train_classification_per_machine.py',
        'scripts/batch_train_classification.py'
    ]
    
    all_exist = True
    for script_path in required_scripts:
        script_file = Path(script_path)
        if script_file.exists():
            print(f"   ✅ {script_path}")
        else:
            print(f"   ❌ {script_path} - NOT FOUND")
            all_exist = False
    
    return all_exist

def validate_directories():
    """Validate required directories exist"""
    print("\n4. Checking directory structure...")
    
    required_dirs = [
        'models/classification',
        'reports/performance_metrics',
        'config'
    ]
    
    for dir_path in required_dirs:
        dir_obj = Path(dir_path)
        dir_obj.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {dir_path}")
    
    return True

def validate_config():
    """Validate model_config.py has required settings"""
    print("\n5. Checking configuration...")
    
    try:
        sys.path.append(str(Path('.').absolute()))
        from config.model_config import AUTOGLUON_CONFIG
        
        # Check classification config
        if 'classification' in AUTOGLUON_CONFIG:
            config = AUTOGLUON_CONFIG['classification']
            print(f"   ✅ Classification config found")
            print(f"      - Time limit: {config.get('time_limit', 'N/A')}s")
            print(f"      - Presets: {config.get('presets', 'N/A')}")
            print(f"      - Eval metric: {config.get('eval_metric', 'N/A')}")
            return True
        else:
            print(f"   ❌ Classification config not found in AUTOGLUON_CONFIG")
            return False
    except Exception as e:
        print(f"   ❌ Error loading config: {str(e)}")
        return False

def estimate_training_time(num_machines, time_limit_per_machine=3600):
    """Estimate total training time"""
    print("\n6. Training time estimates...")
    
    total_seconds = num_machines * time_limit_per_machine
    total_hours = total_seconds / 3600
    
    print(f"   Time limit per machine: {time_limit_per_machine}s ({time_limit_per_machine/60:.0f} minutes)")
    print(f"   Number of machines: {num_machines}")
    print(f"   Total estimated time: {total_hours:.1f} hours ({total_seconds/60:.0f} minutes)")
    print(f"   Expected completion: ~{total_hours:.1f} hours from start")
    
    # Storage estimates
    model_size_mb = 50  # Estimated size per model
    total_storage_mb = num_machines * model_size_mb
    print(f"\n   Storage estimates:")
    print(f"   Model size per machine: ~{model_size_mb} MB")
    print(f"   Total storage needed: ~{total_storage_mb} MB (~{total_storage_mb/1024:.1f} GB)")

def main():
    """Run all validation checks"""
    print("\n" + "=" * 70)
    print("PHASE 2.2.2 SETUP VALIDATION")
    print("Per-Machine Classification Training")
    print("=" * 70)
    
    all_checks_passed = True
    
    # 1. Check priority machines file
    machines_ok, machines = validate_priority_machines_file()
    if not machines_ok:
        all_checks_passed = False
    
    # 2. Check synthetic data
    if machines:
        data_ok = validate_synthetic_data(machines)
        if not data_ok:
            all_checks_passed = False
    else:
        all_checks_passed = False
    
    # 3. Check scripts
    scripts_ok = validate_scripts()
    if not scripts_ok:
        all_checks_passed = False
    
    # 4. Check directories
    dirs_ok = validate_directories()
    if not dirs_ok:
        all_checks_passed = False
    
    # 5. Check config
    config_ok = validate_config()
    if not config_ok:
        all_checks_passed = False
    
    # 6. Estimates
    if machines:
        estimate_training_time(len(machines))
    
    # Final summary
    print("\n" + "=" * 70)
    if all_checks_passed:
        print("✅ ALL VALIDATION CHECKS PASSED")
        print("=" * 70)
        print("\nReady to start training! Run one of:")
        print("\n1. Single machine training:")
        print("   python scripts/train_classification_per_machine.py --machine_id motor_siemens_1la7_001")
        print("\n2. Batch training (all 10 machines):")
        print("   python scripts/batch_train_classification.py --machines_file config/priority_10_machines.txt")
        print("\n" + "=" * 70 + "\n")
        return 0
    else:
        print("❌ VALIDATION FAILED - Please fix errors above before training")
        print("=" * 70 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
