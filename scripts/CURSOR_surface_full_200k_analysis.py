#!/usr/bin/env python3
"""
Monte Carlo Surface Analysis Script

Analyze completed Monte Carlo run and generate decision report.

Usage:
    python CURSOR_surface_full_200k_analysis.py --run_dir "path/to/run"

Examples:
    # Analyze run and generate report
    python CURSOR_surface_full_200k_analysis.py --run_dir "backtest/out/montecarlo/mc_surface_full_200k_20240115"

    # Generate tables only
    python CURSOR_surface_full_200k_analysis.py --run_dir "." --tables_only

    # Export top cells to CSV
    python CURSOR_surface_full_200k_analysis.py --run_dir "." --export_csv
"""

import argparse
import os
import sys
import csv
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wale_montecarlo.analysis.report import generate_decision_report, _load_all_summaries
from wale_montecarlo.analysis.robust_score import get_top_cells, rank_cells_by_robust_score
from wale_montecarlo.analysis.pareto import find_pareto_front_2d, find_pareto_front_3d
from wale_montecarlo.analysis.clustering import find_plateau_clusters


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Monte Carlo surface results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to completed Monte Carlo run directory"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for analysis (default: run_dir/aggregated/analysis)"
    )

    parser.add_argument(
        "--baseline_pf",
        type=float,
        default=1.0,
        help="Baseline profit factor for comparison (default: 1.0)"
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=50,
        help="Number of top cells to include in report (default: 50)"
    )

    parser.add_argument(
        "--tables_only",
        action="store_true",
        help="Generate CSV tables only, skip full report"
    )

    parser.add_argument(
        "--export_csv",
        action="store_true",
        help="Export all results to CSV"
    )

    args = parser.parse_args()

    # Validate run directory
    run_dir = os.path.abspath(args.run_dir)
    if not os.path.exists(run_dir):
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    # Check for completion
    done_file = os.path.join(run_dir, "aggregated", "DONE.txt")
    if not os.path.exists(done_file):
        print("Warning: Run may not be complete (DONE.txt not found)")

    # Set output directory
    if args.output:
        output_dir = os.path.abspath(args.output)
    else:
        output_dir = os.path.join(run_dir, "aggregated", "analysis")

    os.makedirs(output_dir, exist_ok=True)
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    print(f"=== Analyzing: {run_dir} ===")

    # Load summaries
    print("Loading cell summaries...")
    summaries = _load_all_summaries(run_dir)

    if not summaries:
        print("Error: No cell summaries found")
        sys.exit(1)

    print(f"Loaded {len(summaries)} cell summaries")

    # Generate tables
    print("\nGenerating tables...")

    # Top 50 by robust score
    top_cells = get_top_cells(summaries, n=args.top_n, baseline_pf=args.baseline_pf)
    _save_top_cells_csv(top_cells, os.path.join(tables_dir, f"top_{args.top_n}_by_robust_score.csv"))
    print(f"  - top_{args.top_n}_by_robust_score.csv")

    # Pareto front 2D
    pareto_2d = find_pareto_front_2d(summaries)
    _save_pareto_csv(pareto_2d, os.path.join(tables_dir, "pareto_front_pf_vs_maxdd.csv"))
    print(f"  - pareto_front_pf_vs_maxdd.csv ({len(pareto_2d)} cells)")

    # Pareto front 3D
    pareto_3d = find_pareto_front_3d(summaries)
    _save_pareto_csv(pareto_3d, os.path.join(tables_dir, "pareto_front_multidim.csv"))
    print(f"  - pareto_front_multidim.csv ({len(pareto_3d)} cells)")

    # Plateau clusters
    clusters = find_plateau_clusters(summaries)
    _save_clusters_csv(clusters, os.path.join(tables_dir, "plateau_clusters.csv"))
    print(f"  - plateau_clusters.csv ({len(clusters)} clusters)")

    if args.export_csv:
        # Export all summaries
        _save_all_summaries_csv(summaries, os.path.join(tables_dir, "all_cells.csv"))
        print(f"  - all_cells.csv ({len(summaries)} cells)")

    if not args.tables_only:
        # Generate full report
        print("\nGenerating decision report...")
        report_path = os.path.join(output_dir, "SURFACE_FULL_DECISION_REPORT.md")
        report = generate_decision_report(run_dir, report_path, args.baseline_pf)

        if report.startswith("Error"):
            print(f"Warning: {report}")
        else:
            print(f"  - SURFACE_FULL_DECISION_REPORT.md")

    print(f"\n=== Analysis complete ===")
    print(f"Output: {output_dir}")

    # Quick summary
    print(f"\n=== Quick Summary ===")
    if top_cells:
        best = top_cells[0]
        print(f"Best cell: {best['cell_id'][:40]}...")
        print(f"  Robust score: {best['robust_score']:.3f}")
        print(f"  PF (P50/P05): {best['pf_p50']:.2f} / {best['pf_p05']:.2f}")
        print(f"  Max DD (P95): {best['max_dd_p95']:.1f}%")
        print(f"  Category: {best['category']}")


def _save_top_cells_csv(cells: list, path: str):
    """Save top cells to CSV."""
    if not cells:
        return

    fieldnames = [
        'rank', 'cell_id', 'robust_score', 'category',
        'pf_p50', 'pf_p05', 'return_p50', 'max_dd_p95',
        'p_skip', 'slip_dollars', 'delay_bars_max',
        'shuffle_mode', 'bootstrap_mode', 'block_len'
    ]

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, cell in enumerate(cells, 1):
            row = {
                'rank': i,
                'cell_id': cell['cell_id'],
                'robust_score': cell['robust_score'],
                'category': cell['category'],
                'pf_p50': cell['pf_p50'],
                'pf_p05': cell['pf_p05'],
                'return_p50': cell['return_p50'],
                'max_dd_p95': cell['max_dd_p95'],
                'p_skip': cell['config']['p_skip'],
                'slip_dollars': cell['config']['slip_dollars'],
                'delay_bars_max': cell['config']['delay_bars_max'],
                'shuffle_mode': cell['config']['shuffle_mode'],
                'bootstrap_mode': cell['config']['bootstrap_mode'],
                'block_len': cell['config']['block_len']
            }
            writer.writerow(row)


def _save_pareto_csv(cells: list, path: str):
    """Save Pareto front to CSV."""
    if not cells:
        with open(path, 'w') as f:
            f.write("No Pareto-optimal cells found\n")
        return

    fieldnames = [
        'cell_id', 'pf_p50', 'max_dd_p50', 'return_p50',
        'p_skip', 'slip_dollars', 'delay_bars_max'
    ]

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for cell in cells:
            row = {
                'cell_id': cell.cell_id,
                'pf_p50': cell.profit_factor.p50,
                'max_dd_p50': cell.max_drawdown.p50,
                'return_p50': cell.total_return.p50,
                'p_skip': cell.config.p_skip,
                'slip_dollars': cell.config.slip_dollars,
                'delay_bars_max': cell.config.delay_bars_max
            }
            writer.writerow(row)


def _save_clusters_csv(clusters: list, path: str):
    """Save plateau clusters to CSV."""
    if not clusters:
        with open(path, 'w') as f:
            f.write("No plateau clusters found\n")
        return

    fieldnames = [
        'cluster_id', 'n_cells', 'robust_score_mean', 'robust_score_std',
        'pf_p50_mean', 'dd_p50_mean', 'p_skip_range', 'slip_range'
    ]

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, cluster in enumerate(clusters, 1):
            p_skip_range = cluster['parameter_ranges']['p_skip']
            slip_range = cluster['parameter_ranges']['slip_dollars']

            row = {
                'cluster_id': i,
                'n_cells': cluster['n_cells'],
                'robust_score_mean': cluster['robust_score_mean'],
                'robust_score_std': cluster['robust_score_std'],
                'pf_p50_mean': cluster['pf_p50_mean'],
                'dd_p50_mean': cluster['dd_p50_mean'],
                'p_skip_range': f"{min(p_skip_range)}-{max(p_skip_range)}" if p_skip_range else "",
                'slip_range': f"{min(slip_range)}-{max(slip_range)}" if slip_range else ""
            }
            writer.writerow(row)


def _save_all_summaries_csv(summaries: list, path: str):
    """Save all cell summaries to CSV."""
    fieldnames = [
        'cell_id', 'robust_score', 'n_permutations',
        'pf_p05', 'pf_p50', 'pf_p95',
        'dd_p05', 'dd_p50', 'dd_p95',
        'return_p05', 'return_p50', 'return_p95',
        'p_skip', 'slip_dollars', 'delay_bars_max',
        'shuffle_mode', 'bootstrap_mode', 'block_len'
    ]

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for summary in summaries:
            row = {
                'cell_id': summary.cell_id,
                'robust_score': summary.robust_score,
                'n_permutations': summary.n_permutations,
                'pf_p05': summary.profit_factor.p05,
                'pf_p50': summary.profit_factor.p50,
                'pf_p95': summary.profit_factor.p95,
                'dd_p05': summary.max_drawdown.p05,
                'dd_p50': summary.max_drawdown.p50,
                'dd_p95': summary.max_drawdown.p95,
                'return_p05': summary.total_return.p05,
                'return_p50': summary.total_return.p50,
                'return_p95': summary.total_return.p95,
                'p_skip': summary.config.p_skip,
                'slip_dollars': summary.config.slip_dollars,
                'delay_bars_max': summary.config.delay_bars_max,
                'shuffle_mode': summary.config.shuffle_mode.value,
                'bootstrap_mode': summary.config.bootstrap_mode.value,
                'block_len': summary.config.block_len
            }
            writer.writerow(row)


if __name__ == "__main__":
    main()
