"""
Pareto front analysis for Monte Carlo results.

Identifies cells on the efficient frontier:
- 2D: Maximize PF while minimizing MaxDD
- 3D: Add total return as third objective

Pareto-optimal cells represent the best trade-offs.
"""

from typing import List, Tuple, Optional
import numpy as np

from ..models import CellSummary


def find_pareto_front_2d(
    cell_summaries: List[CellSummary],
    maximize_metric: str = 'profit_factor',
    minimize_metric: str = 'max_drawdown'
) -> List[CellSummary]:
    """
    Find 2D Pareto front (efficient frontier).

    A cell is Pareto-optimal if no other cell is better on both metrics.

    Args:
        cell_summaries: List of CellSummary objects
        maximize_metric: Metric to maximize (default: profit_factor P50)
        minimize_metric: Metric to minimize (default: max_drawdown P50)

    Returns:
        List of Pareto-optimal cells
    """
    if not cell_summaries:
        return []

    # Extract metric values
    points = []
    for summary in cell_summaries:
        if maximize_metric == 'profit_factor':
            val_max = summary.profit_factor.p50
        elif maximize_metric == 'total_return':
            val_max = summary.total_return.p50
        else:
            val_max = summary.profit_factor.p50

        if minimize_metric == 'max_drawdown':
            val_min = summary.max_drawdown.p50
        elif minimize_metric == 'worst_month':
            val_min = abs(summary.worst_month.p50)
        else:
            val_min = summary.max_drawdown.p50

        points.append((val_max, val_min, summary))

    # Find Pareto front
    pareto = []
    for i, (x1, y1, s1) in enumerate(points):
        is_dominated = False

        for j, (x2, y2, s2) in enumerate(points):
            if i == j:
                continue

            # s2 dominates s1 if s2 is >= on max metric AND <= on min metric
            # with at least one strict inequality
            if x2 >= x1 and y2 <= y1:
                if x2 > x1 or y2 < y1:
                    is_dominated = True
                    break

        if not is_dominated:
            pareto.append(s1)

    return pareto


def find_pareto_front_3d(
    cell_summaries: List[CellSummary],
    metric1: str = 'profit_factor',
    metric2: str = 'max_drawdown',
    metric3: str = 'total_return'
) -> List[CellSummary]:
    """
    Find 3D Pareto front.

    Maximizes metric1 and metric3, minimizes metric2.

    Args:
        cell_summaries: List of CellSummary objects
        metric1: First metric to maximize
        metric2: Metric to minimize
        metric3: Second metric to maximize

    Returns:
        List of Pareto-optimal cells
    """
    if not cell_summaries:
        return []

    def get_metric(summary: CellSummary, name: str, p: str = 'p50') -> float:
        """Extract metric value from summary."""
        if name == 'profit_factor':
            stats = summary.profit_factor
        elif name == 'max_drawdown':
            stats = summary.max_drawdown
        elif name == 'total_return':
            stats = summary.total_return
        elif name == 'worst_month':
            stats = summary.worst_month
        else:
            return 0.0

        return getattr(stats, p, stats.p50)

    points = []
    for summary in cell_summaries:
        m1 = get_metric(summary, metric1)  # maximize
        m2 = get_metric(summary, metric2)  # minimize
        m3 = get_metric(summary, metric3)  # maximize
        points.append((m1, m2, m3, summary))

    pareto = []
    for i, (x1, y1, z1, s1) in enumerate(points):
        is_dominated = False

        for j, (x2, y2, z2, s2) in enumerate(points):
            if i == j:
                continue

            # s2 dominates s1 if better or equal on all, strictly better on at least one
            # For maximize metrics: higher is better
            # For minimize metrics: lower is better
            better_m1 = x2 >= x1
            better_m2 = y2 <= y1  # minimize
            better_m3 = z2 >= z1

            strictly_better = (x2 > x1) or (y2 < y1) or (z2 > z1)

            if better_m1 and better_m2 and better_m3 and strictly_better:
                is_dominated = True
                break

        if not is_dominated:
            pareto.append(s1)

    return pareto


def compute_pareto_ranking(
    cell_summaries: List[CellSummary],
    maximize_metric: str = 'profit_factor',
    minimize_metric: str = 'max_drawdown'
) -> List[Tuple[CellSummary, int]]:
    """
    Compute Pareto ranking (layer assignment).

    Layer 0 = Pareto front
    Layer 1 = Pareto front after removing layer 0
    etc.

    Args:
        cell_summaries: List of CellSummary objects
        maximize_metric: Metric to maximize
        minimize_metric: Metric to minimize

    Returns:
        List of (summary, rank) tuples
    """
    if not cell_summaries:
        return []

    remaining = list(cell_summaries)
    ranked = []
    layer = 0

    while remaining:
        front = find_pareto_front_2d(remaining, maximize_metric, minimize_metric)

        if not front:
            # Remaining cells are all equivalent
            for s in remaining:
                ranked.append((s, layer))
            break

        for s in front:
            ranked.append((s, layer))
            remaining.remove(s)

        layer += 1

    return ranked


def get_pareto_front_stats(pareto_cells: List[CellSummary]) -> dict:
    """
    Get statistics about the Pareto front.

    Args:
        pareto_cells: List of Pareto-optimal cells

    Returns:
        Dictionary of statistics
    """
    if not pareto_cells:
        return {
            'n_cells': 0,
            'pf_range': (0, 0),
            'dd_range': (0, 0),
            'return_range': (0, 0)
        }

    pfs = [c.profit_factor.p50 for c in pareto_cells]
    dds = [c.max_drawdown.p50 for c in pareto_cells]
    returns = [c.total_return.p50 for c in pareto_cells]

    return {
        'n_cells': len(pareto_cells),
        'pf_range': (min(pfs), max(pfs)),
        'pf_mean': np.mean(pfs),
        'dd_range': (min(dds), max(dds)),
        'dd_mean': np.mean(dds),
        'return_range': (min(returns), max(returns)),
        'return_mean': np.mean(returns)
    }


def filter_pareto_by_constraints(
    pareto_cells: List[CellSummary],
    min_pf: Optional[float] = None,
    max_dd: Optional[float] = None,
    min_return: Optional[float] = None
) -> List[CellSummary]:
    """
    Filter Pareto front by additional constraints.

    Args:
        pareto_cells: Pareto-optimal cells
        min_pf: Minimum profit factor
        max_dd: Maximum drawdown
        min_return: Minimum return

    Returns:
        Filtered list
    """
    result = []

    for cell in pareto_cells:
        include = True

        if min_pf is not None and cell.profit_factor.p50 < min_pf:
            include = False
        if max_dd is not None and cell.max_drawdown.p50 > max_dd:
            include = False
        if min_return is not None and cell.total_return.p50 < min_return:
            include = False

        if include:
            result.append(cell)

    return result
