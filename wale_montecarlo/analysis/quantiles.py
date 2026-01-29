"""
Quantile analysis for Monte Carlo results.

Computes distribution statistics avoiding extreme outliers.
Uses P05/P50/P95 instead of min/max for robustness.
"""

from typing import List, Dict, Tuple
import numpy as np

from ..models import PermutationResult, QuantileStats


def compute_quantiles(
    values: np.ndarray,
    quantiles: List[float] = [0.05, 0.25, 0.50, 0.75, 0.95]
) -> Dict[str, float]:
    """
    Compute specified quantiles of a distribution.

    Args:
        values: Array of values
        quantiles: List of quantiles to compute (0-1)

    Returns:
        Dictionary mapping quantile name to value
    """
    if len(values) == 0:
        return {f"p{int(q*100):02d}": 0.0 for q in quantiles}

    result = {}
    for q in quantiles:
        key = f"p{int(q*100):02d}"
        result[key] = float(np.percentile(values, q * 100))

    return result


def compute_distribution_stats(values: np.ndarray) -> QuantileStats:
    """
    Compute full distribution statistics.

    Args:
        values: Array of values

    Returns:
        QuantileStats with P05, P25, P50, P75, P95, mean, std
    """
    if len(values) == 0:
        return QuantileStats(0, 0, 0, 0, 0, 0, 0)

    return QuantileStats(
        p05=float(np.percentile(values, 5)),
        p25=float(np.percentile(values, 25)),
        p50=float(np.percentile(values, 50)),
        p75=float(np.percentile(values, 75)),
        p95=float(np.percentile(values, 95)),
        mean=float(np.mean(values)),
        std=float(np.std(values))
    )


def compute_all_metric_quantiles(
    results: List[PermutationResult]
) -> Dict[str, QuantileStats]:
    """
    Compute quantile statistics for all metrics.

    Args:
        results: List of PermutationResult objects

    Returns:
        Dictionary mapping metric name to QuantileStats
    """
    if not results:
        empty = QuantileStats(0, 0, 0, 0, 0, 0, 0)
        return {
            'total_return_pct': empty,
            'max_drawdown_pct': empty,
            'profit_factor': empty,
            'worst_month_pct': empty,
            'sharpe_ratio': empty,
            'win_rate': empty
        }

    # Extract arrays
    returns = np.array([r.total_return_pct for r in results])
    drawdowns = np.array([r.max_drawdown_pct for r in results])
    pfs = np.clip(np.array([r.profit_factor for r in results]), -999, 999)
    worst_months = np.array([r.worst_month_pct for r in results])
    sharpes = np.array([r.sharpe_ratio for r in results])
    win_rates = np.array([r.win_rate for r in results])

    return {
        'total_return_pct': compute_distribution_stats(returns),
        'max_drawdown_pct': compute_distribution_stats(drawdowns),
        'profit_factor': compute_distribution_stats(pfs),
        'worst_month_pct': compute_distribution_stats(worst_months),
        'sharpe_ratio': compute_distribution_stats(sharpes),
        'win_rate': compute_distribution_stats(win_rates)
    }


def compute_tail_risk_metrics(
    results: List[PermutationResult]
) -> Dict[str, float]:
    """
    Compute tail risk metrics for decision making.

    These metrics focus on worst-case scenarios.

    Args:
        results: List of PermutationResult objects

    Returns:
        Dictionary of tail risk metrics
    """
    if not results:
        return {
            'return_p05': 0.0,
            'return_p95': 0.0,
            'drawdown_p95': 0.0,
            'pf_p05': 0.0,
            'prob_dd_gt_20': 0.0,
            'prob_dd_gt_40': 0.0,
            'prob_negative_return': 0.0
        }

    returns = np.array([r.total_return_pct for r in results])
    drawdowns = np.array([r.max_drawdown_pct for r in results])
    pfs = np.clip(np.array([r.profit_factor for r in results]), -999, 999)

    return {
        'return_p05': float(np.percentile(returns, 5)),
        'return_p95': float(np.percentile(returns, 95)),
        'drawdown_p95': float(np.percentile(drawdowns, 95)),
        'pf_p05': float(np.percentile(pfs, 5)),
        'prob_dd_gt_20': float(np.mean(drawdowns > 20)),
        'prob_dd_gt_40': float(np.mean(drawdowns > 40)),
        'prob_negative_return': float(np.mean(returns < 0))
    }


def compare_to_baseline(
    perturbed_results: List[PermutationResult],
    baseline_pf: float,
    baseline_return: float
) -> Dict[str, float]:
    """
    Compare perturbed results to baseline metrics.

    Args:
        perturbed_results: Monte Carlo results
        baseline_pf: Baseline profit factor
        baseline_return: Baseline total return

    Returns:
        Comparison metrics
    """
    if not perturbed_results:
        return {
            'pf_degradation_pct': 100.0,
            'return_degradation_pct': 100.0,
            'prob_beat_baseline_pf': 0.0,
            'prob_beat_baseline_return': 0.0
        }

    pfs = np.array([r.profit_factor for r in perturbed_results])
    returns = np.array([r.total_return_pct for r in perturbed_results])

    median_pf = np.median(pfs)
    median_return = np.median(returns)

    pf_degradation = ((baseline_pf - median_pf) / baseline_pf * 100
                      if baseline_pf != 0 else 0)
    return_degradation = ((baseline_return - median_return) / abs(baseline_return) * 100
                         if baseline_return != 0 else 0)

    return {
        'pf_degradation_pct': pf_degradation,
        'return_degradation_pct': return_degradation,
        'prob_beat_baseline_pf': float(np.mean(pfs >= baseline_pf)),
        'prob_beat_baseline_return': float(np.mean(returns >= baseline_return))
    }
