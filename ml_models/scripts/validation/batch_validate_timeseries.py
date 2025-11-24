"""
Batch Industrial Validation for All Time-Series Models
Runs comprehensive validation on all 10 trained machines
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime


# Priority machines (same as training)
PRIORITY_MACHINES = [
    'motor_siemens_1la7_001',
    'motor_abb_m3bp_002',
    'motor_weg_w22_003',
    'pump_grundfos_cr3_004',
    'pump_flowserve_ansi_005',
    'compressor_atlas_copco_ga30_001',
    'compressor_ingersoll_rand_2545_009',
    'cnc_dmg_mori_nlx_010',
    'hydraulic_beckwood_press_011',
    'cooling_tower_bac_vti_018'
]


def validate_machine(machine_id, python_path, script_path):
    """Validate a single machine"""
    print(f"\n{'='*80}")
    print(f"VALIDATING: {machine_id}")
    print(f"{'='*80}\n")
    
    start_time = datetime.now()
    
    try:
        cmd = [python_path, str(script_path), '--machine_id', machine_id]
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=300)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if result.returncode == 0:
            # Load validation report to get grade
            report_file = Path(f'../../reports/validation/{machine_id}_timeseries_industrial_validation.json')
            if report_file.exists():
                with open(report_file, 'r') as f:
                    report = json.load(f)
                    grade = report.get('overall_grade', 'N/A')
                    status = report.get('overall_status', 'Unknown')
            else:
                grade = 'N/A'
                status = 'Report not found'
            
            print(f"\n✅ SUCCESS: {machine_id}")
            print(f"   Grade: {grade}")
            print(f"   Time: {duration:.1f}s")
            
            return {
                'machine_id': machine_id,
                'status': 'success',
                'validation_time_seconds': duration,
                'grade': grade,
                'grade_status': status,
                'timestamp': datetime.now().isoformat()
            }
        else:
            print(f"\n❌ FAILED: {machine_id}")
            return {
                'machine_id': machine_id,
                'status': 'failed',
                'error': 'Validation script returned non-zero exit code',
                'timestamp': datetime.now().isoformat()
            }
            
    except subprocess.TimeoutExpired:
        print(f"\n⏱️ TIMEOUT: {machine_id}")
        return {
            'machine_id': machine_id,
            'status': 'timeout',
            'error': 'Validation exceeded 5 minute timeout',
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"\n❌ ERROR: {machine_id} - {str(e)}")
        return {
            'machine_id': machine_id,
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def main():
    print("="*80)
    print("BATCH TIME-SERIES MODEL VALIDATION")
    print("="*80)
    print(f"Machines to validate: {len(PRIORITY_MACHINES)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Setup paths
    python_path = sys.executable
    script_path = Path(__file__).parent / 'validate_timeseries_industrial.py'
    
    print(f"\nPython: {python_path}")
    print(f"Script: {script_path}")
    
    # Validate all machines
    results = []
    successful = 0
    failed = 0
    
    batch_start = datetime.now()
    
    for i, machine_id in enumerate(PRIORITY_MACHINES, 1):
        print(f"\n{'#'*80}")
        print(f"[{i}/{len(PRIORITY_MACHINES)}] Processing: {machine_id}")
        print(f"{'#'*80}")
        
        result = validate_machine(machine_id, python_path, script_path)
        results.append(result)
        
        if result['status'] == 'success':
            successful += 1
        else:
            failed += 1
    
    batch_end = datetime.now()
    total_time = (batch_end - batch_start).total_seconds() / 60
    
    # Save batch report
    report = {
        'batch_start': batch_start.isoformat(),
        'batch_end': batch_end.isoformat(),
        'total_time_minutes': total_time,
        'total_machines': len(PRIORITY_MACHINES),
        'successful': successful,
        'failed': failed,
        'results': results
    }
    
    report_path = Path('../../reports/validation/batch_timeseries_validation_report.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH VALIDATION COMPLETE")
    print("="*80)
    print(f"Total Time: {total_time:.2f} minutes")
    print(f"Successful: {successful}/{len(PRIORITY_MACHINES)}")
    print(f"Failed: {failed}/{len(PRIORITY_MACHINES)}")
    print(f"Success Rate: {successful/len(PRIORITY_MACHINES)*100:.1f}%")
    print(f"\nReport saved: {report_path}")
    print("="*80)
    
    # Show grade distribution
    if successful > 0:
        print("\n✅ Validation Results:")
        grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0, 'N/A': 0}
        
        for r in results:
            if r['status'] == 'success':
                grade = r.get('grade', 'N/A')
                grade_counts[grade] = grade_counts.get(grade, 0) + 1
                print(f"  {r['machine_id']:40s} Grade: {grade}")
        
        print(f"\nGrade Distribution:")
        for grade in ['A', 'B', 'C', 'D', 'F']:
            if grade_counts[grade] > 0:
                print(f"  Grade {grade}: {grade_counts[grade]} machines")
    
    # Show failed machines
    if failed > 0:
        print("\n❌ Failed Validations:")
        for r in results:
            if r['status'] != 'success':
                print(f"  {r['machine_id']:40s} {r.get('error', 'Unknown error')}")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
