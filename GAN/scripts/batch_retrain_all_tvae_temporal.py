"""
Batch TVAE Temporal Retraining - Phase 1.6 Weeks 2-3
Retrain all 26 TVAE models with temporal seed data sequentially

This script automates the retraining process for all machines:
- Loads machine list from rul_profiles.py
- Sequentially retrains each machine (GPU can only handle one at a time)
- Tracks progress and timing
- Handles errors gracefully (continues to next machine on failure)
- Generates consolidated report at the end

Expected: ~52 hours total (2 hours × 26 machines with 500 epochs)
Recommendation: Run overnight/weekend
"""

import json
import time
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from GAN.config.rul_profiles import RUL_PROFILES
from GAN.config.tvae_config import TVAE_CONFIG
from scripts.retrain_tvae_temporal import retrain_machine_tvae_temporal


def get_all_machines():
    """
    Extract all machine IDs from RUL_PROFILES
    
    Returns:
        list: All 26 machine IDs
    """
    machines = []
    for category_data in RUL_PROFILES.values():
        machines.extend(category_data['machines'])
    return machines


def format_duration(seconds):
    """
    Format duration in human-readable format
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        str: Formatted duration (e.g., "2h 15m 30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def estimate_completion_time(elapsed_per_machine, remaining_machines):
    """
    Estimate when batch will complete
    
    Args:
        elapsed_per_machine: Average seconds per machine so far
        remaining_machines: Number of machines left
    
    Returns:
        str: Estimated completion time
    """
    remaining_seconds = elapsed_per_machine * remaining_machines
    completion_time = datetime.now() + timedelta(seconds=remaining_seconds)
    return completion_time.strftime('%Y-%m-%d %H:%M:%S')


def batch_retrain_all_machines(test_mode=False, start_from=None, batch_size=None):
    """
    Retrain all 26 machines sequentially
    
    Args:
        test_mode: If True, use 10 epochs for quick testing
        start_from: Optional machine ID to start from (resume capability)
        batch_size: Optional batch size override (default: from config)
    
    Returns:
        dict: Batch retraining results
    """
    
    print("\n" + "=" * 80)
    print("PHASE 1.6 WEEKS 2-3: BATCH TVAE TEMPORAL RETRAINING")
    print("=" * 80)
    print(f"\nBatch started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get all machines
    machines = get_all_machines()
    print(f"\nTotal machines: {len(machines)}")
    
    if test_mode:
        print("\n[TEST MODE] Using 10 epochs for quick validation")
        print("Expected time: ~1-2 minutes per machine (~30-50 minutes total)")
    else:
        print("\n[PRODUCTION MODE] Using 500 epochs for high-quality training")
        print("Expected time: ~2 hours per machine (~52 hours total)")
        print("\nRecommendation: Run overnight/weekend to minimize wait time")
    
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
        'test_mode': test_mode,
        'total_machines': len(machines),
        'successful': [],
        'failed': [],
        'timing': {},
        'quality_scores': {},
        'rul_correlations': {}
    }
    
    # Configuration
    config = TVAE_CONFIG.copy()
    
    if batch_size is not None:
        config['batch_size'] = batch_size
        print(f"\n[CONFIG] Batch size override: {batch_size}")
    
    print(f"\n{'=' * 80}")
    print("STARTING SEQUENTIAL RETRAINING")
    print(f"{'=' * 80}\n")
    
    # Process each machine
    for idx, machine_id in enumerate(machines, start=start_index + 1):
        print(f"\n{'#' * 80}")
        print(f"# MACHINE {idx}/{len(machines) + start_index}: {machine_id}")
        print(f"{'#' * 80}")
        
        machine_start = time.time()
        
        try:
            # Retrain machine
            machine_result = retrain_machine_tvae_temporal(machine_id, config, test_mode=test_mode)
            
            machine_elapsed = time.time() - machine_start
            
            # Record success
            results['successful'].append(machine_id)
            results['timing'][machine_id] = {
                'seconds': round(machine_elapsed, 2),
                'minutes': round(machine_elapsed / 60, 2),
                'hours': round(machine_elapsed / 3600, 2),
                'training_seconds': machine_result['training_time_seconds']
            }
            results['quality_scores'][machine_id] = machine_result['quality_score']
            
            if machine_result.get('rul_correlation'):
                results['rul_correlations'][machine_id] = {
                    'average': machine_result['rul_correlation']['average'],
                    'negative_percentage': machine_result['rul_correlation']['negative_percentage']
                }
            
            # Progress update
            elapsed_total = time.time() - batch_start
            avg_per_machine = elapsed_total / (idx - start_index)
            remaining = len(machines) + start_index - idx
            
            print(f"\n[PROGRESS] {idx}/{len(machines) + start_index} machines completed")
            print(f"  Success: {len(results['successful'])}, Failed: {len(results['failed'])}")
            print(f"  Time for this machine: {format_duration(machine_elapsed)}")
            print(f"  Average per machine: {format_duration(avg_per_machine)}")
            print(f"  Total elapsed: {format_duration(elapsed_total)}")
            
            if remaining > 0:
                est_remaining = avg_per_machine * remaining
                print(f"  Estimated remaining: {format_duration(est_remaining)}")
                print(f"  Estimated completion: {estimate_completion_time(avg_per_machine, remaining)}")
            
        except Exception as e:
            machine_elapsed = time.time() - machine_start
            
            print(f"\n[ERROR] Failed to retrain {machine_id}")
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
    
    # Quality statistics
    if results['quality_scores']:
        scores = list(results['quality_scores'].values())
        results['quality_statistics'] = {
            'average': round(sum(scores) / len(scores), 3),
            'min': round(min(scores), 3),
            'max': round(max(scores), 3),
            'above_0.70': sum(1 for s in scores if s >= 0.70),
            'above_0.80': sum(1 for s in scores if s >= 0.80),
            'above_0.90': sum(1 for s in scores if s >= 0.90)
        }
    
    # RUL correlation statistics
    if results['rul_correlations']:
        avg_corrs = [c['average'] for c in results['rul_correlations'].values()]
        neg_pcts = [c['negative_percentage'] for c in results['rul_correlations'].values()]
        
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
    
    report_filename = f"batch_tvae_temporal_retraining_{'test' if test_mode else 'production'}.json"
    report_path = reports_dir / report_filename
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print final summary
    print(f"\n\n{'=' * 80}")
    print("BATCH RETRAINING COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nCompletion time: {results['batch_end_time']}")
    print(f"Total duration: {format_duration(batch_elapsed)}")
    print(f"\nResults:")
    print(f"  Successful: {len(results['successful'])}/{len(machines) + start_index} ({results['success_rate']}%)")
    print(f"  Failed: {len(results['failed'])}")
    
    if results['failed']:
        print(f"\n  Failed machines:")
        for failure in results['failed']:
            print(f"    - {failure['machine_id']}: {failure['error']}")
    
    if 'quality_statistics' in results:
        print(f"\nQuality Statistics:")
        print(f"  Average Score: {results['quality_statistics']['average']}")
        print(f"  Range: [{results['quality_statistics']['min']}, {results['quality_statistics']['max']}]")
        print(f"  Above 0.70: {results['quality_statistics']['above_0.70']}/{len(results['quality_scores'])}")
        print(f"  Above 0.80: {results['quality_statistics']['above_0.80']}/{len(results['quality_scores'])}")
        print(f"  Above 0.90: {results['quality_statistics']['above_0.90']}/{len(results['quality_scores'])}")
    
    if 'rul_statistics' in results:
        print(f"\nRUL Correlation Statistics:")
        print(f"  Average Correlation: {results['rul_statistics']['average_correlation']}")
        print(f"  Average Negative %: {results['rul_statistics']['average_negative_percentage']}%")
        print(f"  All Negative (100%): {results['rul_statistics']['all_negative']}/{len(results['rul_correlations'])}")
        print(f"  Mostly Negative (>80%): {results['rul_statistics']['mostly_negative']}/{len(results['rul_correlations'])}")
    
    print(f"\nReport saved: {report_filename}")
    print(f"Individual reports: reports/tvae_temporal/*.json")
    
    print(f"\n{'=' * 80}")
    print("Next Steps:")
    print("  1. Review batch report for any failures")
    print("  2. Verify all models in models/tvae/temporal/")
    print("  3. Proceed to Phase 1.6 Week 4: Generate 50K samples per machine")
    print(f"{'=' * 80}\n")
    
    return results


def main():
    """Main entry point for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch retrain all 26 TVAE models with temporal seed data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Production mode (500 epochs, ~52 hours)
  python batch_retrain_all_tvae_temporal.py
  
  # Test mode (10 epochs, ~30-50 minutes)
  python batch_retrain_all_tvae_temporal.py --test
  
  # Resume from specific machine (if previous run failed)
  python batch_retrain_all_tvae_temporal.py --start_from pump_grundfos_cr3_004

Recommendations:
  - Run production mode overnight/weekend
  - Monitor GPU temperature and utilization
  - Check progress periodically via reports/tvae_temporal/
  - Use --test mode first to verify everything works
        """
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: 10 epochs per machine (~1-2 min each, ~30-50 min total)'
    )
    
    parser.add_argument(
        '--start_from',
        type=str,
        default=None,
        help='Resume from specific machine (e.g., pump_grundfos_cr3_004)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Override batch size (default: 500 from config)'
    )
    
    args = parser.parse_args()
    
    try:
        results = batch_retrain_all_machines(test_mode=args.test, start_from=args.start_from, batch_size=args.batch_size)
        
        # Exit code based on success rate
        if results['success_rate'] == 100.0:
            sys.exit(0)  # All successful
        elif results['success_rate'] >= 90.0:
            sys.exit(1)  # Mostly successful (some failures)
        else:
            sys.exit(2)  # Many failures
        
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Batch retraining interrupted by user")
        print("You can resume by using --start_from <machine_id>")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Batch retraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
