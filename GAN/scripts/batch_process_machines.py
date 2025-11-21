"""
Batch process remaining machines with progress tracking
"""
import subprocess
import sys
from pathlib import Path

machines = [
    'cnc_okuma_lb3000_001',
    'cnc_dmg_mori_ntx_001'
]

python_exe = 'C:/GAN/Modules/Scripts/python.exe'
script_path = 'c:/GAN/scripts/add_new_machine.py'

print("\n" + "="*70)
print("BATCH PROCESSING REMAINING MACHINES")
print("="*70)

for i, machine_id in enumerate(machines, 1):
    print(f"\n[{i}/{len(machines)}] Processing: {machine_id}")
    print("-" * 70)
    
    cmd = [python_exe, script_path, '--machine_id', machine_id, '--epochs', '100', '--batch_size', '500']
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✅ {machine_id} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ {machine_id} failed with error code {e.returncode}")
        continue

print("\n" + "="*70)
print("BATCH PROCESSING COMPLETE")
print("="*70)
