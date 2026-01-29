"""
Cell worker for Monte Carlo simulations.

Runs permutations for a single cell with:
- Deterministic seeding
- Resume support
- Periodic checkpointing
- Timeout handling
"""

import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Callable

from .models import (
    Trade, CellConfig, OHLCData, EquityCurve,
    PermutationResult, CellSummary, QuantileStats
)
from .seeding import get_rng_for_permutation
from .perturbations import apply_all_perturbations
from .metrics import compute_all_metrics
from .io import (
    load_metrics_compact, save_metrics_compact,
    atomic_write_json, ensure_cell_dir
)

import numpy as np


# Checkpoint every N permutations
CHECKPOINT_INTERVAL = 1000

# Default timeout per permutation (seconds)
DEFAULT_PERM_TIMEOUT = 1.0

# Log progress every N permutations
LOG_INTERVAL = 10000


logger = logging.getLogger(__name__)


def run_cell(
    cell_config: CellConfig,
    trades: List[Trade],
    n_perms: int,
    output_dir: str,
    ohlc_data: Optional[OHLCData] = None,
    initial_equity: float = 100000.0,
    resume: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Tuple[List[PermutationResult], CellSummary]:
    """
    Run all permutations for a single cell.

    Args:
        cell_config: Configuration for this cell
        trades: Original trade list
        n_perms: Target number of permutations
        output_dir: Directory for cell output
        ohlc_data: Optional OHLC data for delay modeling
        initial_equity: Starting equity for metrics
        resume: If True, resume from existing progress
        progress_callback: Called with (completed, total) periodically

    Returns:
        Tuple of (all results, cell summary)
    """
    cell_id = cell_config.to_cell_id()
    cell_dir = ensure_cell_dir(output_dir, cell_id)
    metrics_path = os.path.join(cell_dir, "metrics_compact.csv")
    progress_path = os.path.join(cell_dir, "progress.json")
    summary_path = os.path.join(cell_dir, "summary.json")
    log_path = os.path.join(cell_dir, "logs.txt")

    # Set up logging for this cell
    _setup_cell_logging(log_path)

    logger.info(f"Starting cell: {cell_id}")
    logger.info(f"Target permutations: {n_perms}")

    # Resume support: load existing results
    existing_results = []
    start_perm = 0

    if resume and os.path.exists(metrics_path):
        existing_results, max_idx = load_metrics_compact(metrics_path)
        start_perm = max_idx + 1
        logger.info(f"Resuming from perm {start_perm} ({len(existing_results)} existing)")

    if start_perm >= n_perms:
        logger.info("Cell already complete")
        summary = compute_cell_summary(cell_id, cell_config, existing_results)
        return existing_results, summary

    # Run remaining permutations
    results = list(existing_results)  # Copy existing
    batch_results = []  # Buffer for checkpointing

    start_time = time.time()
    last_checkpoint = start_time
    last_log = start_time

    for perm_idx in range(start_perm, n_perms):
        # Get seeded RNG for this permutation
        rng = get_rng_for_permutation(cell_id, perm_idx)

        # Apply perturbations
        perturbed = apply_all_perturbations(
            trades, cell_config, rng, ohlc_data
        )

        # Compute metrics
        perm_result = compute_all_metrics(perturbed, perm_idx, initial_equity)
        batch_results.append(perm_result)
        results.append(perm_result)

        # Progress callback
        if progress_callback and (perm_idx + 1) % 100 == 0:
            progress_callback(perm_idx + 1, n_perms)

        # Periodic checkpoint
        current_time = time.time()
        if (perm_idx + 1) % CHECKPOINT_INTERVAL == 0 or current_time - last_checkpoint > 30:
            _checkpoint(metrics_path, batch_results, progress_path, perm_idx + 1, n_perms, start_time)
            batch_results = []
            last_checkpoint = current_time

        # Periodic logging
        if (perm_idx + 1) % LOG_INTERVAL == 0:
            elapsed = current_time - start_time
            rate = (perm_idx - start_perm + 1) / elapsed if elapsed > 0 else 0
            logger.info(f"Progress: {perm_idx + 1}/{n_perms} ({rate:.0f} perms/sec)")

    # Final checkpoint
    if batch_results:
        _checkpoint(metrics_path, batch_results, progress_path, n_perms, n_perms, start_time)

    # Compute summary statistics
    summary = compute_cell_summary(cell_id, cell_config, results)
    atomic_write_json(summary_path, summary.to_dict())

    elapsed = time.time() - start_time
    logger.info(f"Cell complete: {n_perms} perms in {elapsed:.1f}s")

    return results, summary


def run_cell_simple(
    cell_config: CellConfig,
    trades: List[Trade],
    n_perms: int,
    ohlc_data: Optional[OHLCData] = None,
    initial_equity: float = 100000.0
) -> List[PermutationResult]:
    """
    Simplified cell runner without file I/O (for testing).

    Args:
        cell_config: Configuration for this cell
        trades: Original trade list
        n_perms: Number of permutations
        ohlc_data: Optional OHLC data
        initial_equity: Starting equity

    Returns:
        List of PermutationResult objects
    """
    cell_id = cell_config.to_cell_id()
    results = []

    for perm_idx in range(n_perms):
        rng = get_rng_for_permutation(cell_id, perm_idx)
        perturbed = apply_all_perturbations(trades, cell_config, rng, ohlc_data)
        result = compute_all_metrics(perturbed, perm_idx, initial_equity)
        results.append(result)

    return results


def _checkpoint(
    metrics_path: str,
    batch_results: List[PermutationResult],
    progress_path: str,
    completed: int,
    total: int,
    start_time: float
) -> None:
    """Save checkpoint to disk."""
    if batch_results:
        save_metrics_compact(metrics_path, batch_results, append=True)

    elapsed = time.time() - start_time
    progress = {
        "completed": completed,
        "total": total,
        "pct": (completed / total) * 100 if total > 0 else 0,
        "elapsed_seconds": elapsed,
        "last_update": datetime.now().isoformat()
    }
    atomic_write_json(progress_path, progress)


def _setup_cell_logging(log_path: str) -> None:
    """Set up file logging for cell."""
    handler = logging.FileHandler(log_path, mode='a')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def compute_cell_summary(
    cell_id: str,
    config: CellConfig,
    results: List[PermutationResult]
) -> CellSummary:
    """
    Compute summary statistics for a cell.

    Args:
        cell_id: Cell identifier
        config: Cell configuration
        results: All permutation results

    Returns:
        CellSummary with quantiles and statistics
    """
    if not results:
        # Return empty summary
        empty_stats = QuantileStats(0, 0, 0, 0, 0, 0, 0)
        return CellSummary(
            cell_id=cell_id,
            config=config,
            n_permutations=0,
            total_return=empty_stats,
            max_drawdown=empty_stats,
            profit_factor=empty_stats,
            worst_month=empty_stats
        )

    # Extract metric arrays
    returns = np.array([r.total_return_pct for r in results])
    drawdowns = np.array([r.max_drawdown_pct for r in results])
    pfs = np.array([r.profit_factor for r in results])
    worst_months = np.array([r.worst_month_pct for r in results])

    # Handle infinities in profit factor
    pfs = np.clip(pfs, -999, 999)

    def compute_quantile_stats(arr: np.ndarray) -> QuantileStats:
        return QuantileStats(
            p05=float(np.percentile(arr, 5)),
            p25=float(np.percentile(arr, 25)),
            p50=float(np.percentile(arr, 50)),
            p75=float(np.percentile(arr, 75)),
            p95=float(np.percentile(arr, 95)),
            mean=float(np.mean(arr)),
            std=float(np.std(arr))
        )

    return CellSummary(
        cell_id=cell_id,
        config=config,
        n_permutations=len(results),
        total_return=compute_quantile_stats(returns),
        max_drawdown=compute_quantile_stats(drawdowns),
        profit_factor=compute_quantile_stats(pfs),
        worst_month=compute_quantile_stats(worst_months)
    )


def get_cell_status(cell_dir: str, target_perms: int) -> dict:
    """
    Get status of a cell from its output directory.

    Args:
        cell_dir: Path to cell directory
        target_perms: Expected number of permutations

    Returns:
        Status dictionary
    """
    metrics_path = os.path.join(cell_dir, "metrics_compact.csv")
    progress_path = os.path.join(cell_dir, "progress.json")
    summary_path = os.path.join(cell_dir, "summary.json")

    status = {
        "exists": os.path.exists(cell_dir),
        "has_metrics": os.path.exists(metrics_path),
        "has_summary": os.path.exists(summary_path),
        "completed": 0,
        "target": target_perms,
        "is_complete": False
    }

    if os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                progress = json.load(f)
                status["completed"] = progress.get("completed", 0)
        except Exception:
            pass

    if os.path.exists(metrics_path):
        try:
            results, max_idx = load_metrics_compact(metrics_path)
            status["completed"] = len(results)
        except Exception:
            pass

    status["is_complete"] = status["completed"] >= target_perms

    return status
