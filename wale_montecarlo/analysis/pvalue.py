"""
P-value testing for Monte Carlo results.

Tests whether perturbations significantly degrade performance.
Uses Bonferroni correction for multiple comparisons.
"""

from typing import List, Optional
import numpy as np

from ..models import PermutationResult


def compute_pvalue(
    results: List[PermutationResult],
    baseline_pf: float,
    metric: str = 'profit_factor'
) -> float:
    """
    Compute p-value testing if perturbations matter.

    Null hypothesis: Perturbations don't affect performance
    Test: What fraction of permutations achieve >= baseline?

    A low p-value means perturbations significantly degrade performance.

    Args:
        results: List of PermutationResult objects
        baseline_pf: Baseline value to compare against
        metric: Which metric to test ('profit_factor', 'total_return_pct', etc.)

    Returns:
        P-value (0 to 1)
    """
    if not results:
        return 1.0

    # Extract metric values
    if metric == 'profit_factor':
        values = np.array([r.profit_factor for r in results])
    elif metric == 'total_return_pct':
        values = np.array([r.total_return_pct for r in results])
    elif metric == 'sharpe_ratio':
        values = np.array([r.sharpe_ratio for r in results])
    else:
        values = np.array([r.profit_factor for r in results])

    # Fraction that meet or exceed baseline
    n_success = np.sum(values >= baseline_pf)
    pvalue = n_success / len(values)

    return float(pvalue)


def apply_bonferroni(pvalue: float, n_tests: int) -> float:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        pvalue: Raw p-value
        n_tests: Number of tests performed

    Returns:
        Corrected p-value (capped at 1.0)
    """
    if n_tests <= 0:
        return pvalue

    corrected = pvalue * n_tests
    return min(corrected, 1.0)


def compute_pvalue_two_tailed(
    results: List[PermutationResult],
    baseline_pf: float,
    metric: str = 'profit_factor'
) -> float:
    """
    Compute two-tailed p-value.

    Tests if results are significantly different from baseline
    (either better or worse).

    Args:
        results: List of PermutationResult objects
        baseline_pf: Baseline value
        metric: Which metric to test

    Returns:
        Two-tailed p-value
    """
    if not results:
        return 1.0

    if metric == 'profit_factor':
        values = np.array([r.profit_factor for r in results])
    elif metric == 'total_return_pct':
        values = np.array([r.total_return_pct for r in results])
    else:
        values = np.array([r.profit_factor for r in results])

    mean_val = np.mean(values)

    # Count how many are at least as extreme as baseline
    if mean_val >= baseline_pf:
        # Baseline is below mean - count fraction >= baseline
        n_extreme = np.sum(values >= baseline_pf)
    else:
        # Baseline is above mean - count fraction <= baseline
        n_extreme = np.sum(values <= baseline_pf)

    # Two-tailed: double the one-tail probability
    pvalue = 2 * min(n_extreme / len(values), 1 - n_extreme / len(values))

    return float(min(pvalue, 1.0))


def compute_confidence_interval(
    results: List[PermutationResult],
    metric: str = 'profit_factor',
    confidence: float = 0.95
) -> tuple:
    """
    Compute confidence interval for a metric.

    Args:
        results: List of PermutationResult objects
        metric: Which metric
        confidence: Confidence level (e.g., 0.95 for 95%)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if not results:
        return (0.0, 0.0)

    if metric == 'profit_factor':
        values = np.array([r.profit_factor for r in results])
    elif metric == 'total_return_pct':
        values = np.array([r.total_return_pct for r in results])
    elif metric == 'max_drawdown_pct':
        values = np.array([r.max_drawdown_pct for r in results])
    else:
        values = np.array([r.profit_factor for r in results])

    alpha = 1 - confidence
    lower_q = alpha / 2 * 100
    upper_q = (1 - alpha / 2) * 100

    lower = float(np.percentile(values, lower_q))
    upper = float(np.percentile(values, upper_q))

    return (lower, upper)


def is_significant(
    results: List[PermutationResult],
    baseline_pf: float,
    alpha: float = 0.05,
    n_tests: int = 1
) -> bool:
    """
    Test if performance degradation is statistically significant.

    Args:
        results: List of PermutationResult objects
        baseline_pf: Baseline profit factor
        alpha: Significance level
        n_tests: Number of tests for Bonferroni correction

    Returns:
        True if degradation is significant
    """
    pvalue = compute_pvalue(results, baseline_pf)
    pvalue_corrected = apply_bonferroni(pvalue, n_tests)

    return pvalue_corrected < alpha


def compute_effect_size(
    results: List[PermutationResult],
    baseline_pf: float
) -> float:
    """
    Compute Cohen's d effect size for performance difference.

    Args:
        results: Monte Carlo results
        baseline_pf: Baseline value

    Returns:
        Cohen's d (small=0.2, medium=0.5, large=0.8)
    """
    if not results:
        return 0.0

    values = np.array([r.profit_factor for r in results])
    mean_val = np.mean(values)
    std_val = np.std(values)

    if std_val == 0:
        return 0.0

    cohens_d = (baseline_pf - mean_val) / std_val

    return float(cohens_d)
