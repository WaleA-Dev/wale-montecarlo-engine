"""
Convergence and advanced analysis utilities.

Functions for analyzing convergence, Pareto ranking, and confidence intervals.
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional


def convergence_analysis(
    values: List[float],
    metric_fn: Callable[[List[float]], float],
    checkpoints: List[int] = None
) -> List[Dict]:
    """
    Compute metric at each checkpoint and measure stability.
    
    Useful for determining how many permutations are actually needed
    for stable estimates.
    
    Args:
        values: List of metric values from permutations
        metric_fn: Function to compute aggregate metric (e.g., np.median)
        checkpoints: Sample sizes to evaluate (default: 1K to 200K)
    
    Returns:
        List of dicts with n, value, and pct_error_vs_final
    """
    if checkpoints is None:
        checkpoints = [1000, 5000, 10000, 25000, 50000, 100000, 200000]
    
    if len(values) < 2:
        return []
    
    final_value = metric_fn(values)
    
    convergence = []
    for n in checkpoints:
        if n > len(values):
            break
        subset_value = metric_fn(values[:n])
        
        if abs(final_value) > 1e-10:
            pct_error = abs(subset_value - final_value) / abs(final_value) * 100
        else:
            pct_error = 0.0 if abs(subset_value) < 1e-10 else 100.0
        
        convergence.append({
            'n': n,
            'value': float(subset_value),
            'pct_error_vs_final': float(pct_error)
        })
    
    return convergence


def percentile_confidence_interval(
    data: List[float],
    percentile: float,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = None
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for a percentile.
    
    Args:
        data: List of values
        percentile: Percentile to compute CI for (0-100)
        confidence: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(data) < 2:
        val = data[0] if data else 0.0
        return (val, val)
    
    rng = np.random.default_rng(seed)
    data_arr = np.array(data)
    
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data_arr, size=len(data_arr), replace=True)
        bootstrap_estimates.append(np.percentile(sample, percentile))
    
    alpha = 1 - confidence
    lower = float(np.percentile(bootstrap_estimates, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2)))
    
    return (lower, upper)


def pareto_rank(cells: List[Dict], objectives: List[str] = None) -> List[int]:
    """
    Rank cells by Pareto dominance.
    
    A cell is dominated if another cell beats it on ALL objectives.
    Lower rank = better (0 = Pareto-optimal).
    
    Args:
        cells: List of cell dicts with metric values
        objectives: List of metric names to maximize (default: pf_p50, neg_maxdd_p95)
    
    Returns:
        List of ranks (0 = Pareto-optimal, higher = more dominated)
    """
    if objectives is None:
        objectives = ['pf_p50', 'neg_maxdd_p95']
    
    n = len(cells)
    if n == 0:
        return []
    
    # Convert to array for vectorized operations
    values = np.zeros((n, len(objectives)))
    for i, cell in enumerate(cells):
        for j, obj in enumerate(objectives):
            if obj == 'neg_maxdd_p95':
                # Negate maxdd so higher is better
                values[i, j] = -cell.get('max_dd_p95', cell.get('maxdd_p95', 0))
            else:
                values[i, j] = cell.get(obj, 0)
    
    ranks = []
    for i in range(n):
        n_dominating = 0
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j >= i on all objectives and j > i on at least one
            if np.all(values[j] >= values[i]) and np.any(values[j] > values[i]):
                n_dominating += 1
        ranks.append(n_dominating)
    
    return ranks


def find_pareto_front(cells: List[Dict], objectives: List[str] = None) -> List[Dict]:
    """
    Find the Pareto-optimal front (cells with rank 0).
    
    Args:
        cells: List of cell dicts with metric values
        objectives: List of metric names to maximize
    
    Returns:
        List of Pareto-optimal cells
    """
    ranks = pareto_rank(cells, objectives)
    return [cell for cell, rank in zip(cells, ranks) if rank == 0]


def estimate_required_permutations(
    values: List[float],
    target_error_pct: float = 1.0,
    percentile: float = 50
) -> int:
    """
    Estimate how many permutations are needed for a target error level.
    
    Uses the convergence pattern to extrapolate.
    
    Args:
        values: Current values from permutations
        target_error_pct: Target percentage error (default 1%)
        percentile: Which percentile to estimate for
    
    Returns:
        Estimated number of permutations needed
    """
    # Error scales as 1/sqrt(n) for percentiles
    # Use current data to estimate the variance coefficient
    
    n_current = len(values)
    if n_current < 1000:
        return 200000  # Default to max if insufficient data
    
    # Compute error at half the data
    half_n = n_current // 2
    full_value = np.percentile(values, percentile)
    half_value = np.percentile(values[:half_n], percentile)
    
    if abs(full_value) < 1e-10:
        return n_current  # Can't estimate
    
    current_error = abs(half_value - full_value) / abs(full_value) * 100
    
    # Error at n: error(n) = k / sqrt(n)
    # k = error * sqrt(n)
    k = current_error * np.sqrt(half_n)
    
    # Required n: n = (k / target_error)^2
    required_n = (k / target_error_pct) ** 2
    
    # Cap at reasonable values
    required_n = max(n_current, min(int(required_n), 1000000))
    
    return required_n


def compute_overfit_score(
    baseline_pf: float,
    stressed_pf_p50: float,
    severe_stressed_pf_p50: float = None
) -> Dict:
    """
    Compute comprehensive overfitting metrics.
    
    Args:
        baseline_pf: Original backtest profit factor
        stressed_pf_p50: Median PF at moderate stress (p_skip=0.05, slip=$100, delay=1)
        severe_stressed_pf_p50: Optional median PF at severe stress
    
    Returns:
        Dict with overfit_score, degradation_rate, classification, and recommendation
    """
    if baseline_pf <= 0:
        return {
            'overfit_score': 1.0,
            'degradation_rate': 1.0,
            'classification': 'Highly Overfit',
            'recommendation': 'Strategy has no edge. Redesign completely.'
        }
    
    degradation_rate = (baseline_pf - stressed_pf_p50) / baseline_pf
    degradation_rate = max(0.0, min(1.0, degradation_rate))
    
    # Classify based on stressed performance
    if stressed_pf_p50 >= 1.5:
        classification = 'Robust'
        recommendation = 'Trade with confidence. Consider full position size.'
    elif stressed_pf_p50 >= 1.0:
        classification = 'Fragile'
        recommendation = 'Reduce position size by 50%. Monitor performance closely.'
    elif stressed_pf_p50 >= 0.8:
        classification = 'Overfit'
        recommendation = 'Do not trade live. Edge disappears under realistic conditions.'
    else:
        classification = 'Highly Overfit'
        recommendation = 'Redesign strategy. Backtest results are not reliable.'
    
    result = {
        'overfit_score': degradation_rate,
        'degradation_rate': degradation_rate,
        'classification': classification,
        'recommendation': recommendation,
        'baseline_pf': baseline_pf,
        'stressed_pf_p50': stressed_pf_p50,
    }
    
    if severe_stressed_pf_p50 is not None:
        severe_degradation = (baseline_pf - severe_stressed_pf_p50) / baseline_pf
        result['severe_degradation_rate'] = max(0.0, min(1.0, severe_degradation))
        result['severe_stressed_pf_p50'] = severe_stressed_pf_p50
    
    return result
