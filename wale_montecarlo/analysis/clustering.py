"""
Plateau clustering for Monte Carlo results.

Identifies stable parameter regions where similar robust scores
are achieved. Large stable clusters indicate robustness to
parameter choice - the strategy works across a range of conditions.
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np

from ..models import CellSummary, CellConfig


def find_plateau_clusters(
    cell_summaries: List[CellSummary],
    tolerance: float = 0.10,
    min_cluster_size: int = 3
) -> List[Dict]:
    """
    Identify plateau clusters of cells with similar robust scores.

    Cells are clustered if their robust scores are within `tolerance`
    of each other. Adjacent cells in parameter space are grouped.

    Args:
        cell_summaries: List of CellSummary objects
        tolerance: Maximum relative difference to be in same cluster (0.10 = 10%)
        min_cluster_size: Minimum cells to form a cluster

    Returns:
        List of cluster dictionaries with stats
    """
    if not cell_summaries:
        return []

    # Sort by robust score
    sorted_cells = sorted(
        cell_summaries,
        key=lambda c: c.robust_score,
        reverse=True
    )

    # Greedy clustering: start from highest score
    clusters = []
    assigned = set()

    for cell in sorted_cells:
        if cell.cell_id in assigned:
            continue

        # Start new cluster
        cluster_cells = [cell]
        cluster_score = cell.robust_score
        assigned.add(cell.cell_id)

        # Find cells within tolerance
        for other in sorted_cells:
            if other.cell_id in assigned:
                continue

            if cluster_score == 0:
                if other.robust_score == 0:
                    cluster_cells.append(other)
                    assigned.add(other.cell_id)
            else:
                rel_diff = abs(other.robust_score - cluster_score) / cluster_score
                if rel_diff <= tolerance:
                    cluster_cells.append(other)
                    assigned.add(other.cell_id)

        if len(cluster_cells) >= min_cluster_size:
            clusters.append(_compute_cluster_stats(cluster_cells))

    return clusters


def _compute_cluster_stats(cells: List[CellSummary]) -> Dict:
    """Compute statistics for a cluster of cells."""
    robust_scores = [c.robust_score for c in cells]
    pf_p50s = [c.profit_factor.p50 for c in cells]
    dd_p50s = [c.max_drawdown.p50 for c in cells]

    # Find parameter ranges in cluster
    configs = [c.config for c in cells]
    p_skips = set(c.p_skip for c in configs)
    slips = set(c.slip_dollars for c in configs)
    delays = set(c.delay_bars_max for c in configs)
    shuffles = set(c.shuffle_mode.value for c in configs)
    bootstraps = set(c.bootstrap_mode.value for c in configs)

    return {
        'n_cells': len(cells),
        'cell_ids': [c.cell_id for c in cells],
        'robust_score_mean': float(np.mean(robust_scores)),
        'robust_score_std': float(np.std(robust_scores)),
        'robust_score_range': (min(robust_scores), max(robust_scores)),
        'pf_p50_mean': float(np.mean(pf_p50s)),
        'dd_p50_mean': float(np.mean(dd_p50s)),
        'parameter_ranges': {
            'p_skip': sorted(p_skips),
            'slip_dollars': sorted(slips),
            'delay_bars_max': sorted(delays),
            'shuffle_modes': sorted(shuffles),
            'bootstrap_modes': sorted(bootstraps)
        }
    }


def find_stable_regions(
    cell_summaries: List[CellSummary],
    parameter: str,
    metric: str = 'robust_score'
) -> Dict:
    """
    Find stable regions along a single parameter dimension.

    Args:
        cell_summaries: List of CellSummary objects
        parameter: Parameter to analyze ('p_skip', 'slip_dollars', etc.)
        metric: Metric to assess stability

    Returns:
        Dictionary with stability analysis
    """
    if not cell_summaries:
        return {}

    # Group by parameter value
    groups = defaultdict(list)
    for cell in cell_summaries:
        config = cell.config
        if parameter == 'p_skip':
            val = config.p_skip
        elif parameter == 'slip_dollars':
            val = config.slip_dollars
        elif parameter == 'delay_bars_max':
            val = config.delay_bars_max
        elif parameter == 'shuffle_mode':
            val = config.shuffle_mode.value
        elif parameter == 'bootstrap_mode':
            val = config.bootstrap_mode.value
        elif parameter == 'block_len':
            val = config.block_len
        else:
            continue

        groups[val].append(cell)

    # Compute stats per group
    results = {}
    for val, cells in groups.items():
        if metric == 'robust_score':
            values = [c.robust_score for c in cells]
        elif metric == 'profit_factor':
            values = [c.profit_factor.p50 for c in cells]
        elif metric == 'max_drawdown':
            values = [c.max_drawdown.p50 for c in cells]
        else:
            values = [c.robust_score for c in cells]

        results[val] = {
            'n_cells': len(cells),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'cv': float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0
        }

    return {
        'parameter': parameter,
        'groups': results,
        'most_stable': min(results.items(), key=lambda x: x[1]['cv'])[0] if results else None,
        'best_mean': max(results.items(), key=lambda x: x[1]['mean'])[0] if results else None
    }


def compute_parameter_sensitivity(
    cell_summaries: List[CellSummary]
) -> Dict[str, float]:
    """
    Compute sensitivity of robust score to each parameter.

    Higher sensitivity means the parameter has more impact on results.

    Args:
        cell_summaries: List of CellSummary objects

    Returns:
        Dictionary mapping parameter name to sensitivity score
    """
    parameters = ['p_skip', 'slip_dollars', 'delay_bars_max', 'shuffle_mode', 'bootstrap_mode']

    sensitivities = {}

    for param in parameters:
        analysis = find_stable_regions(cell_summaries, param)

        if analysis and 'groups' in analysis:
            means = [g['mean'] for g in analysis['groups'].values()]
            if len(means) > 1:
                # Sensitivity = range of means / overall mean
                overall_mean = np.mean(means)
                if overall_mean != 0:
                    sensitivity = (max(means) - min(means)) / overall_mean
                else:
                    sensitivity = 0.0
            else:
                sensitivity = 0.0
        else:
            sensitivity = 0.0

        sensitivities[param] = float(sensitivity)

    return sensitivities


def find_optimal_parameter_region(
    cell_summaries: List[CellSummary],
    top_n: int = 20
) -> Dict:
    """
    Find the parameter region containing the best cells.

    Analyzes the top N cells to find common parameter ranges.

    Args:
        cell_summaries: List of CellSummary objects
        top_n: Number of top cells to analyze

    Returns:
        Dictionary with optimal parameter ranges
    """
    if not cell_summaries:
        return {}

    # Get top cells by robust score
    sorted_cells = sorted(
        cell_summaries,
        key=lambda c: c.robust_score,
        reverse=True
    )[:top_n]

    configs = [c.config for c in sorted_cells]

    return {
        'n_cells_analyzed': len(sorted_cells),
        'optimal_ranges': {
            'p_skip': {
                'min': min(c.p_skip for c in configs),
                'max': max(c.p_skip for c in configs),
                'most_common': _most_common([c.p_skip for c in configs])
            },
            'slip_dollars': {
                'min': min(c.slip_dollars for c in configs),
                'max': max(c.slip_dollars for c in configs),
                'most_common': _most_common([c.slip_dollars for c in configs])
            },
            'delay_bars_max': {
                'min': min(c.delay_bars_max for c in configs),
                'max': max(c.delay_bars_max for c in configs),
                'most_common': _most_common([c.delay_bars_max for c in configs])
            },
            'shuffle_mode': _most_common([c.shuffle_mode.value for c in configs]),
            'bootstrap_mode': _most_common([c.bootstrap_mode.value for c in configs]),
            'block_len': _most_common([c.block_len for c in configs])
        },
        'top_cell_scores': [c.robust_score for c in sorted_cells[:5]]
    }


def _most_common(values: list):
    """Find most common value in list."""
    if not values:
        return None
    counts = defaultdict(int)
    for v in values:
        counts[v] += 1
    return max(counts.items(), key=lambda x: x[1])[0]
