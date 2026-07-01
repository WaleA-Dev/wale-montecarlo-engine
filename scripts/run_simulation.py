#!/usr/bin/env python3
"""
Monte Carlo Simulation CLI

Usage:
    python scripts/run_simulation.py --trades examples/sample_trade_list.csv --n_per_cell 1000 --jobs 4
    python scripts/run_simulation.py --trades examples/sample_trade_list.csv --mode explore --jobs 4
    python scripts/run_simulation.py --resume mc_run_20260129_012613
    python scripts/run_simulation.py --status mc_run_20260129_012613
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wale_montecarlo.models import RunConfig, ShuffleMode, BootstrapMode
from wale_montecarlo.runner import MonteCarloRunner, get_run_status
from wale_montecarlo.grid import get_grid_summary


def apply_mode_grid(config: RunConfig, grid: dict) -> None:
    """Apply a mode preset grid (from config.defaults) onto a RunConfig."""
    config.p_skip_values = list(grid['p_skip'])
    config.slip_values = list(grid['slip'])
    config.delay_values = list(grid['delay'])
    config.shuffle_modes = [ShuffleMode(m) for m in grid['shuffle']]
    config.bootstrap_modes = [BootstrapMode(m) for m in grid['bootstrap']]
    config.block_len_values = list(grid['block_len'])


def load_manifest(output_dir: str) -> dict:
    """Load run_manifest.json for an existing run (empty dict if missing)."""
    manifest_path = os.path.join(output_dir, "aggregated", "run_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo stress-testing simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--trades",
        type=str,
        help="Path to the trade list CSV (any filename works)"
    )

    parser.add_argument(
        "--n_per_cell",
        type=int,
        default=None,
        help="Number of permutations per cell (default: 1000, or the mode preset)"
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
        help="Resume existing run by name (trades path is read from the run manifest)"
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
        "--fixed_delay",
        type=int,
        default=None,
        help="Fix the delay dimension to a single value (shrinks the grid)"
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
        help="Grid preset: explore (128 cells, 20K perms), focus (64 cells, 100K perms), full (6048 cells, 200K perms)"
    )

    args = parser.parse_args()

    # Status check mode
    if args.status:
        output_dir = args.status
        if not os.path.isabs(output_dir):
            base = os.path.abspath(args.output_dir) if args.output_dir else \
                os.path.join(os.getcwd(), "montecarlo_output")
            output_dir = os.path.join(base, output_dir)

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

    # Require trades path for new runs (resume reads it from the manifest)
    if not args.trades and not args.resume:
        parser.error("--trades is required unless using --resume or --status")

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

    # Resolve trades path (from CLI or, on resume, from the run manifest)
    manifest = load_manifest(output_dir) if args.resume else {}
    trades_path = None
    if args.trades:
        trades_path = os.path.abspath(args.trades)
    elif manifest:
        trades_path = manifest.get("config", {}).get("trades_path")

    if not trades_path or not os.path.exists(trades_path):
        print(f"Error: Trade file not found: {trades_path or args.trades}")
        sys.exit(1)

    input_dir = os.path.dirname(trades_path)

    # Permutations per cell: CLI > mode preset > manifest (resume) > 1000
    n_per_cell = args.n_per_cell
    if n_per_cell is None and args.mode:
        from wale_montecarlo.config import get_perms_for_mode
        n_per_cell = get_perms_for_mode(args.mode)
    if n_per_cell is None and manifest:
        n_per_cell = manifest.get("config", {}).get("n_per_cell")
    if n_per_cell is None:
        n_per_cell = 1000

    # Create run config
    config = RunConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        trades_path=trades_path,
        n_per_cell=n_per_cell,
        n_jobs=args.jobs,
        fixed_delay=args.fixed_delay,
    )

    # Apply mode preset grid (this is what actually shrinks the grid)
    if args.mode:
        from wale_montecarlo.config import get_grid_config
        apply_mode_grid(config, get_grid_config(args.mode))
        print(f"Using {args.mode} mode: {n_per_cell:,} perms/cell")
    elif args.resume and manifest.get("grid"):
        # Restore the original grid so a resumed run never changes shape
        grid = manifest["grid"]
        config.p_skip_values = grid.get("p_skip_values", config.p_skip_values)
        config.slip_values = grid.get("slip_values", config.slip_values)
        config.delay_values = grid.get("delay_values", config.delay_values)
        config.shuffle_modes = [ShuffleMode(m) for m in grid.get("shuffle_modes", [])] or config.shuffle_modes
        config.bootstrap_modes = [BootstrapMode(m) for m in grid.get("bootstrap_modes", [])] or config.bootstrap_modes
        config.block_len_values = grid.get("block_len_values", config.block_len_values)

    # Dry run mode
    if args.dry_run:
        from wale_montecarlo.grid import generate_grid
        cells = generate_grid(config)
        summary = get_grid_summary(cells)

        print(f"\n=== Dry Run Configuration ===")
        print(f"Input: {input_dir}")
        print(f"Trades: {trades_path}")
        print(f"Output: {output_dir}")
        print(f"\n=== Grid Summary ===")
        print(f"Total cells: {summary['total_cells']}")
        print(f"Permutations per cell: {n_per_cell:,}")
        print(f"Total permutations: {summary['total_cells'] * n_per_cell:,}")
        print(f"\n=== Parameter Ranges ===")
        for key, values in summary.get('dimensions', {}).items():
            print(f"  {key}: {values} values")
        return

    # Run or resume
    print(f"\n=== Starting Monte Carlo Simulation ===")
    print(f"Trades: {trades_path}")
    print(f"Output: {output_dir}")
    print(f"Permutations per cell: {n_per_cell:,}")
    print(f"Workers: {args.jobs}")
    print()

    runner = MonteCarloRunner(config)

    if not runner.setup():
        print("Error: Setup failed")
        sys.exit(1)

    # Always resume-safe: existing per-cell progress is deduped and reused
    success = runner.run(resume=True)

    status = runner.get_status()
    print(f"\n=== Final Status ===")
    print(f"Cells completed: {status['cells_completed']}/{status['cells_total']}")
    print(f"Cells failed: {status['cells_failed']}")
    print(f"Permutations: {status['perms_completed']:,}/{status['perms_total']:,}")

    if success:
        print("\nSimulation completed successfully!")
        print(f"Results at: {output_dir}")
        print(f"Grid summary: {os.path.join(output_dir, 'aggregated', 'grid_summary.csv')}")
    else:
        print("\nSimulation completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
