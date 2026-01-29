#!/usr/bin/env python3
"""
Monte Carlo Surface Analysis Runner

Run a full Monte Carlo grid search over perturbation parameters.

Usage:
    python CURSOR_run_surface_full_200k.py --repo "path/to/backtest/export" --n_per_cell 200000 --jobs 8

Examples:
    # Full 200K run
    python CURSOR_run_surface_full_200k.py --repo "." --n_per_cell 200000 --jobs 8

    # Quick test with 1000 permutations
    python CURSOR_run_surface_full_200k.py --repo "." --n_per_cell 1000 --jobs 4

    # Check status of existing run
    python CURSOR_run_surface_full_200k.py --repo "." --run_name mc_surface_full_200k_20240115 --status_only

    # Resume interrupted run
    python CURSOR_run_surface_full_200k.py --repo "." --run_name mc_surface_full_200k_20240115 --resume
"""

import argparse
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wale_montecarlo.models import RunConfig
from wale_montecarlo.runner import MonteCarloRunner, get_run_status
from wale_montecarlo.grid import get_grid_summary, estimate_runtime


def main():
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo surface analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Path to backtest export directory (contains trade_list.csv)"
    )

    parser.add_argument(
        "--n_per_cell",
        type=int,
        default=200000,
        help="Number of permutations per cell (default: 200000)"
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )

    parser.add_argument(
        "--fixed_delay",
        type=int,
        default=1,
        help="Fix delay parameter to this value (default: 1)"
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this run (auto-generated if not provided)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: repo/backtest/out/montecarlo/)"
    )

    parser.add_argument(
        "--status_only",
        action="store_true",
        help="Only show status, don't run"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing progress"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show configuration without running"
    )

    args = parser.parse_args()

    # Validate input directory
    repo_path = os.path.abspath(args.repo)
    trade_list_path = os.path.join(repo_path, "trade_list.csv")

    if not os.path.exists(trade_list_path):
        print(f"Error: trade_list.csv not found at {trade_list_path}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_base = os.path.abspath(args.output_dir)
    else:
        output_base = os.path.join(repo_path, "backtest", "out", "montecarlo")

    # Generate or use run name
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"mc_surface_full_{args.n_per_cell // 1000}k_{timestamp}"

    output_dir = os.path.join(output_base, run_name)

    # Status only mode
    if args.status_only:
        status = get_run_status(output_dir)
        print(f"\n=== Run Status: {run_name} ===")
        print(f"Exists: {status.get('exists', False)}")
        print(f"Complete: {status.get('is_complete', False)}")
        if 'cells_completed' in status:
            print(f"Cells: {status['cells_completed']}/{status.get('cells_total', '?')}")
        if 'pct_complete' in status:
            print(f"Progress: {status['pct_complete']:.1f}%")
        if 'last_heartbeat' in status:
            print(f"Last heartbeat: {status['last_heartbeat']}")
        return

    # Create run config
    config = RunConfig(
        input_dir=repo_path,
        output_dir=output_dir,
        n_per_cell=args.n_per_cell,
        n_jobs=args.jobs,
        fixed_delay=args.fixed_delay
    )

    # Dry run mode
    if args.dry_run:
        from wale_montecarlo.grid import generate_grid
        cells = generate_grid(config)
        summary = get_grid_summary(cells)
        runtime = estimate_runtime(len(cells), args.n_per_cell)

        print(f"\n=== Dry Run Configuration ===")
        print(f"Input: {repo_path}")
        print(f"Output: {output_dir}")
        print(f"\n=== Grid Summary ===")
        print(f"Total cells: {summary['total_cells']}")
        print(f"Permutations per cell: {args.n_per_cell:,}")
        print(f"Total permutations: {runtime['total_permutations']:,}")
        print(f"\n=== Runtime Estimates ===")
        print(f"Single core: {runtime['single_core_hours']:.1f} hours")
        print(f"8 cores: {runtime['8_core_hours']:.1f} hours")
        print(f"\n=== Parameter Ranges ===")
        for key, values in summary['dimensions'].items():
            print(f"  {key}: {values} values")
        return

    # Run
    print(f"\n=== Starting Monte Carlo Run ===")
    print(f"Input: {repo_path}")
    print(f"Output: {output_dir}")
    print(f"Permutations per cell: {args.n_per_cell:,}")
    print(f"Workers: {args.jobs}")
    print()

    runner = MonteCarloRunner(config)

    if not runner.setup():
        print("Error: Setup failed")
        sys.exit(1)

    success = runner.run(resume=args.resume)

    status = runner.get_status()
    print(f"\n=== Final Status ===")
    print(f"Cells completed: {status['cells_completed']}/{status['cells_total']}")
    print(f"Cells failed: {status['cells_failed']}")
    print(f"Permutations: {status['perms_completed']:,}/{status['perms_total']:,}")

    if success:
        print("\nRun completed successfully!")
        print(f"Results at: {output_dir}")
    else:
        print("\nRun completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
