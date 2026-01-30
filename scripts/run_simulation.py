#!/usr/bin/env python3
"""
Monte Carlo Simulation CLI

Usage:
    python scripts/run_simulation.py --trades examples/sample_trade_list.csv --n_per_cell 1000 --jobs 4
    python scripts/run_simulation.py --trades path/to/trades.csv --resume mc_run_20260129
    python scripts/run_simulation.py --status mc_run_20260129
"""

import argparse
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wale_montecarlo.models import RunConfig
from wale_montecarlo.runner import MonteCarloRunner, get_run_status
from wale_montecarlo.grid import get_grid_summary


def main():
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo stress-testing simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--trades",
        type=str,
        help="Path to trade_list.csv file"
    )

    parser.add_argument(
        "--n_per_cell",
        type=int,
        default=1000,
        help="Number of permutations per cell (default: 1000)"
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume existing run by name"
    )

    parser.add_argument(
        "--status",
        type=str,
        default=None,
        help="Check status of existing run"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: ./montecarlo_output)"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show configuration without running"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=['explore', 'focus', 'full'],
        default=None,
        help="Grid mode: explore (128 cells, 20K perms), focus (200 cells, 100K perms), full (6048 cells, 200K perms)"
    )

    args = parser.parse_args()

    # Apply mode presets if specified
    if args.mode:
        from wale_montecarlo.config import get_grid_config, DEFAULT_CONFIG
        
        mode_perms = DEFAULT_CONFIG['perms_per_mode']
        if args.n_per_cell == 1000:  # Default wasn't overridden
            args.n_per_cell = mode_perms.get(args.mode, 50000)
        
        # Store grid config for later use
        args.grid_config = get_grid_config(args.mode)
        print(f"Using {args.mode} mode: {args.n_per_cell:,} perms/cell")

    # Status check mode
    if args.status:
        output_dir = args.status
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), "montecarlo_output", output_dir)
        
        status = get_run_status(output_dir)
        print(f"\n=== Run Status: {args.status} ===")
        print(f"Exists: {status.get('exists', False)}")
        print(f"Complete: {status.get('is_complete', False)}")
        if 'cells_completed' in status:
            print(f"Cells: {status['cells_completed']}/{status.get('cells_total', '?')}")
        if 'pct_complete' in status:
            print(f"Progress: {status['pct_complete']:.1f}%")
        if 'last_heartbeat' in status:
            print(f"Last heartbeat: {status['last_heartbeat']}")
        return

    # Require trades path for run modes
    if not args.trades and not args.resume:
        parser.error("--trades is required unless using --resume")

    # Resolve trades path
    if args.trades:
        trades_path = os.path.abspath(args.trades)
        if not os.path.exists(trades_path):
            print(f"Error: Trade file not found: {trades_path}")
            sys.exit(1)
        input_dir = os.path.dirname(trades_path)
    else:
        input_dir = os.getcwd()

    # Determine output directory
    if args.output_dir:
        output_base = os.path.abspath(args.output_dir)
    else:
        output_base = os.path.join(os.getcwd(), "montecarlo_output")

    # Generate or use run name
    if args.resume:
        run_name = args.resume
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"mc_run_{timestamp}"

    output_dir = os.path.join(output_base, run_name)

    # Create run config
    config = RunConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        n_per_cell=args.n_per_cell,
        n_jobs=args.jobs
    )

    # Dry run mode
    if args.dry_run:
        from wale_montecarlo.grid import generate_grid
        cells = generate_grid(config)
        summary = get_grid_summary(cells)

        print(f"\n=== Dry Run Configuration ===")
        print(f"Input: {input_dir}")
        print(f"Trades: {args.trades}")
        print(f"Output: {output_dir}")
        print(f"\n=== Grid Summary ===")
        print(f"Total cells: {summary['total_cells']}")
        print(f"Permutations per cell: {args.n_per_cell:,}")
        print(f"Total permutations: {summary['total_cells'] * args.n_per_cell:,}")
        print(f"\n=== Parameter Ranges ===")
        for key, values in summary.get('dimensions', {}).items():
            print(f"  {key}: {values} values")
        return

    # Run or resume
    print(f"\n=== Starting Monte Carlo Simulation ===")
    print(f"Trades: {args.trades}")
    print(f"Output: {output_dir}")
    print(f"Permutations per cell: {args.n_per_cell:,}")
    print(f"Workers: {args.jobs}")
    print()

    runner = MonteCarloRunner(config)

    if not runner.setup():
        print("Error: Setup failed")
        sys.exit(1)

    success = runner.run(resume=bool(args.resume))

    status = runner.get_status()
    print(f"\n=== Final Status ===")
    print(f"Cells completed: {status['cells_completed']}/{status['cells_total']}")
    print(f"Cells failed: {status['cells_failed']}")
    print(f"Permutations: {status['perms_completed']:,}/{status['perms_total']:,}")

    if success:
        print("\nSimulation completed successfully!")
        print(f"Results at: {output_dir}")
    else:
        print("\nSimulation completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
