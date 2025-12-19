"""
Batch Generation Script - Phase 1.6 Week 4
Generate 50K samples for all 26 machines from retrained temporal TVAE models

Expected time: ~15 minutes per machine = ~6.5 hours total
"""

import json
import time
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from GAN.config.rul_profiles import RUL_PROFILES
from scripts.generate_from_temporal_tvae import generate_temporal_data


def get_all_machines():
    """Extract all machine IDs from RUL_PROFILES"""
    machines = []
    for category_data in RUL_PROFILES.values():
        machines.extend(category_data['machines'])
    return machines


def batch_generate_all_machines(num_samples=50000, start_from=None):
    """
    Generate synthetic data for all 26 machines
    
    Args:
        num_samples: Samples per machine (default: 50000)
        start_from: Optional machine ID to resume from
    
    Returns:
        dict: Batch generation results
    """
    
    print("\n" + "=" * 80)
    print("PHASE 1.6 WEEK 4: BATCH SYNTHETIC DATA GENERATION")
    print("=" * 80)
    print(f"\nBatch started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get all machines
    machines = get_all_machines()
    print(f"\nTotal machines: {len(machines)}")
    print(f"Samples per machine: {num_samples:,}")
    print(f"Total samples: {len(machines) * num_samples:,}")
    
    # Handle resume capability
    start_index = 0
    if start_from:
        if start_from in machines:
            start_index = machines.index(start_from)
            print(f"\n[RESUME] Starting from machine {start_index + 1}: {start_from}")
            machines = machines[start_index:]
        else:
            print(f"\n[WARNING] Machine '{start_from}' not found, starting from beginning")
    
    # Initialize tracking
    batch_start = time.time()
    results = {
        'batch_start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_samples_per_machine': num_samples,
        'total_machines': len(machines),
        'successful': [],
        'failed': [],
        'timing': {},
        'total_samples': {},
        'total_size_mb': {},
        'rul_correlations': {}
    }
    
    print(f"\n{'=' * 80}")
    print("STARTING BATCH GENERATION")
    print(f"{'=' * 80}\n")
    
    # Process each machine
    for idx, machine_id in enumerate(machines, start=start_index + 1):
        print(f"\n{'#' * 80}")
        print(f"# MACHINE {idx}/{len(machines) + start_index}: {machine_id}")
        print(f"{'#' * 80}")
        
        machine_start = time.time()
        
        try:
            # Generate data
            machine_result = generate_temporal_data(machine_id, num_samples=num_samples)
            
            machine_elapsed = time.time() - machine_start
            
            # Record success
            results['successful'].append(machine_id)
            results['timing'][machine_id] = {
                'seconds': round(machine_elapsed, 2),
                'generation_seconds': machine_result['generation_time_seconds']
            }
            results['total_samples'][machine_id] = machine_result['samples_generated']
            results['total_size_mb'][machine_id] = machine_result['total_size_mb']
            
            if machine_result.get('rul_correlations'):
                results['rul_correlations'][machine_id] = {
                    'average': machine_result['avg_rul_correlation'],
                    'negative_percentage': machine_result['negative_correlation_pct']
                }
            
            # Progress update
            elapsed_total = time.time() - batch_start
            avg_per_machine = elapsed_total / (idx - start_index)
            remaining = len(machines) + start_index - idx
            
            print(f"\n[PROGRESS] {idx}/{len(machines) + start_index} machines completed")
            print(f"  Success: {len(results['successful'])}, Failed: {len(results['failed'])}")
            print(f"  Time for this machine: {machine_elapsed:.0f}s")
            print(f"  Average per machine: {avg_per_machine:.0f}s")
            print(f"  Total elapsed: {elapsed_total/60:.1f} min")
            
            if remaining > 0:
                est_remaining = avg_per_machine * remaining
                print(f"  Estimated remaining: {est_remaining/60:.1f} min")
                completion_time = datetime.now().timestamp() + est_remaining
                print(f"  Estimated completion: {datetime.fromtimestamp(completion_time).strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            machine_elapsed = time.time() - machine_start
            
            print(f"\n[ERROR] Failed to generate data for {machine_id}")
            print(f"Error: {e}")
            
            results['failed'].append({
                'machine_id': machine_id,
                'error': str(e),
                'elapsed_seconds': round(machine_elapsed, 2)
            })
            
            print(f"\n[CONTINUE] Proceeding to next machine...")
    
    # Calculate final statistics
    batch_elapsed = time.time() - batch_start
    results['batch_end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results['total_elapsed_seconds'] = round(batch_elapsed, 2)
    results['total_elapsed_minutes'] = round(batch_elapsed / 60, 2)
    results['total_elapsed_hours'] = round(batch_elapsed / 3600, 2)
    results['success_rate'] = round(len(results['successful']) / (len(machines) + start_index) * 100, 1)
    
    # Aggregate statistics
    if results['total_samples']:
        results['aggregate_statistics'] = {
            'total_samples_generated': sum(results['total_samples'].values()),
            'total_size_mb': round(sum(results['total_size_mb'].values()), 2),
            'avg_generation_time_seconds': round(
                sum(t['generation_seconds'] for t in results['timing'].values()) / len(results['timing']), 2
            )
        }
    
    # RUL correlation statistics
    if results['rul_correlations']:
        avg_corrs = [c['average'] for c in results['rul_correlations'].values() if c['average']]
        neg_pcts = [c['negative_percentage'] for c in results['rul_correlations'].values() if c['negative_percentage']]
        
        if avg_corrs:
            results['rul_statistics'] = {
                'average_correlation': round(sum(avg_corrs) / len(avg_corrs), 3),
                'average_negative_percentage': round(sum(neg_pcts) / len(neg_pcts), 1),
                'all_negative': sum(1 for p in neg_pcts if p == 100.0),
                'mostly_negative': sum(1 for p in neg_pcts if p >= 80.0)
            }
    
    # Save report
    project_root = Path(__file__).parent.parent
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    report_filename = "batch_generation_report.json"
    report_path = reports_dir / report_filename
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print final summary
    print(f"\n\n{'=' * 80}")
    print("BATCH GENERATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nCompletion time: {results['batch_end_time']}")
    print(f"Total duration: {batch_elapsed/60:.1f} minutes ({batch_elapsed/3600:.2f} hours)")
    print(f"\nResults:")
    print(f"  Successful: {len(results['successful'])}/{len(machines) + start_index} ({results['success_rate']}%)")
    print(f"  Failed: {len(results['failed'])}")
    
    if results['failed']:
        print(f"\n  Failed machines:")
        for failure in results['failed']:
            print(f"    - {failure['machine_id']}: {failure['error']}")
    
    if 'aggregate_statistics' in results:
        print(f"\nAggregate Statistics:")
        print(f"  Total samples: {results['aggregate_statistics']['total_samples_generated']:,}")
        print(f"  Total size: {results['aggregate_statistics']['total_size_mb']:.2f} MB")
        print(f"  Avg generation time: {results['aggregate_statistics']['avg_generation_time_seconds']:.2f}s per machine")
    
    if 'rul_statistics' in results:
        print(f"\nRUL Correlation Statistics:")
        print(f"  Average Correlation: {results['rul_statistics']['average_correlation']}")
        print(f"  Average Negative %: {results['rul_statistics']['average_negative_percentage']}%")
        print(f"  All Negative (100%): {results['rul_statistics']['all_negative']}/{len(results['rul_correlations'])}")
        print(f"  Mostly Negative (>80%): {results['rul_statistics']['mostly_negative']}/{len(results['rul_correlations'])}")
    
    print(f"\nReport saved: {report_filename}")
    print(f"Individual reports: reports/generation/*.json")
    print(f"Synthetic data: data/synthetic/*/{{train,val,test}}.parquet")
    
    print(f"\n{'=' * 80}")
    print("Next Steps:")
    print("  1. Review batch generation report")
    print("  2. Verify all datasets in data/synthetic/")
    print("  3. Run comprehensive verification script")
    print("  4. Document Phase 1.6 completion")
    print(f"{'=' * 80}\n")
    
    return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch generate synthetic data for all 26 machines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 50K samples per machine (default)
  python batch_generate_from_temporal_tvae.py
  
  # Generate custom number of samples
  python batch_generate_from_temporal_tvae.py --num_samples 100000
  
  # Resume from specific machine
  python batch_generate_from_temporal_tvae.py --start_from pump_grundfos_cr3_004

Expected time: ~15 minutes per machine = ~6.5 hours total
        """
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=50000,
        help='Samples per machine (default: 50000)'
    )
    
    parser.add_argument(
        '--start_from',
        type=str,
        default=None,
        help='Resume from specific machine'
    )
    
    args = parser.parse_args()
    
    try:
        results = batch_generate_all_machines(
            num_samples=args.num_samples,
            start_from=args.start_from
        )
        
        # Exit code based on success rate
        if results['success_rate'] == 100.0:
            sys.exit(0)
        elif results['success_rate'] >= 90.0:
            sys.exit(1)
        else:
            sys.exit(2)
        
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Batch generation interrupted by user")
        print("You can resume by using --start_from <machine_id>")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Batch generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
