"""
Robust scoring for Monte Carlo results.

Combines performance and significance into a single metric.
Formula: robust_score = PF_P50 × (1 - p_corrected)

This ensures both factors are strong - a high PF with poor
significance or vice versa will be penalized.
"""

from typing import List, Dict, Tuple
import numpy as np

from ..models import PermutationResult, CellSummary


def compute_robust_score(
    pf_p50: float,
    p_corrected: float
) -> float:
    """
    Compute robust score from median profit factor and p-value.

    Formula: PF_P50 * (1 - p_corrected)

    High score requires both:
    - High median profit factor
    - Low p-value (high significance)

    Args:
        pf_p50: Median profit factor (P50)
        p_corrected: Corrected p-value

    Returns:
        Robust score (higher is better)
    """
    # Ensure p_corrected is in valid range
    p_corrected = max(0.0, min(1.0, p_corrected))

    # Cap profit factor at reasonable value
    pf_p50 = max(0.0, min(999.0, pf_p50))

    return pf_p50 * (1 - p_corrected)


def compute_robust_score_v3(
    pf_p50: float,
    p_corrected: float,
    maxdd_p95: float = 0.0,
    dd_penalty_start: float = 0.20,
    dd_penalty_end: float = 0.60
) -> float:
    """
    Compute robust score v3 with gated multiplicative formula.
    
    Formula: max(0, PF - 1.0) * (1 - p_val) * dd_penalty
    
    Key properties:
    - PF <= 1.0 (breakeven or losing) scores exactly 0
    - Only excess return above breakeven contributes
    - Significance scales the excess return
    - High drawdown reduces or zeros the score
    
    Args:
        pf_p50: Median profit factor (P50)
        p_corrected: Corrected p-value
        maxdd_p95: 95th percentile max drawdown (0 to 1, e.g. 0.25 = 25%)
        dd_penalty_start: Drawdown at which penalty starts (default 20%)
        dd_penalty_end: Drawdown at which score becomes 0 (default 60%)
    
    Returns:
        Robust score v3 (higher is better, 0 = no edge)
    """
    # Gated: only excess return above breakeven counts
    excess_return = max(0.0, pf_p50 - 1.0)
    
    # Significance component
    p_corrected = max(0.0, min(1.0, p_corrected))
    significance = 1.0 - p_corrected
    
    # Drawdown penalty: linear from dd_penalty_start to dd_penalty_end
    # Score = 1.0 at dd_penalty_start, 0.0 at dd_penalty_end
    if maxdd_p95 <= dd_penalty_start:
        dd_penalty = 1.0
    elif maxdd_p95 >= dd_penalty_end:
        dd_penalty = 0.0
    else:
        dd_range = dd_penalty_end - dd_penalty_start
        dd_penalty = (dd_penalty_end - maxdd_p95) / dd_range
    
    return excess_return * significance * dd_penalty


def classify_overfit(
    baseline_pf: float,
    stressed_pf_p50: float,
    robust_threshold: float = 1.5,
    fragile_threshold: float = 1.0
) -> str:
    """
    Classify strategy overfitting based on degradation under stress.
    
    Args:
        baseline_pf: Original backtest profit factor
        stressed_pf_p50: Median PF at moderate stress (p_skip=0.05, slip=$100, delay=1)
        robust_threshold: PF threshold for 'Robust' classification
        fragile_threshold: PF threshold for 'Fragile' classification
    
    Returns:
        Classification: 'Robust', 'Fragile', 'Overfit', or 'Highly Overfit'
    """
    # Calculate degradation rate
    if baseline_pf > 0:
        degradation = (baseline_pf - stressed_pf_p50) / baseline_pf
    else:
        degradation = 1.0
    
    # Classify based on stressed performance
    if stressed_pf_p50 >= robust_threshold:
        return 'Robust'
    elif stressed_pf_p50 >= fragile_threshold:
        return 'Fragile'
    elif stressed_pf_p50 >= 0.8:  # Still slightly profitable
        return 'Overfit'
    else:
        return 'Highly Overfit'


def compute_degradation_rate(
    baseline_pf: float,
    stressed_pf_p50: float
) -> float:
    """
    Compute degradation rate (how much edge is lost under stress).
    
    Args:
        baseline_pf: Original backtest profit factor
        stressed_pf_p50: Median PF under stress
    
    Returns:
        Degradation rate (0 to 1, higher = more degradation)
    """
    if baseline_pf <= 0:
        return 1.0
    
    degradation = (baseline_pf - stressed_pf_p50) / baseline_pf
    return max(0.0, min(1.0, degradation))


def compute_robust_score_from_results(
    results: List[PermutationResult],
    baseline_pf: float,
    n_cells: int = 1
) -> Tuple[float, Dict]:
    """
    Compute robust score directly from permutation results.

    Args:
        results: List of PermutationResult objects
        baseline_pf: Baseline profit factor for p-value calculation
        n_cells: Total cells for Bonferroni correction

    Returns:
        Tuple of (robust_score, detail_dict)
    """
    from .pvalue import compute_pvalue, apply_bonferroni

    if not results:
        return 0.0, {'pf_p50': 0.0, 'pvalue_raw': 1.0, 'pvalue_corrected': 1.0}

    # Compute P50 profit factor
    pfs = np.array([r.profit_factor for r in results])
    pf_p50 = float(np.percentile(pfs, 50))

    # Compute p-value
    pvalue_raw = compute_pvalue(results, baseline_pf)
    pvalue_corrected = apply_bonferroni(pvalue_raw, n_cells)

    score = compute_robust_score(pf_p50, pvalue_corrected)

    details = {
        'pf_p50': pf_p50,
        'pvalue_raw': pvalue_raw,
        'pvalue_corrected': pvalue_corrected,
        'robust_score': score
    }

    return score, details


def rank_cells_by_robust_score(
    cell_summaries: List[CellSummary],
    baseline_pf: float
) -> List[Tuple[CellSummary, float]]:
    """
    Rank cells by robust score (highest first).

    Args:
        cell_summaries: List of CellSummary objects
        baseline_pf: Baseline PF for reference

    Returns:
        List of (summary, score) tuples, sorted descending
    """
    n_cells = len(cell_summaries)

    scored = []
    for summary in cell_summaries:
        score = compute_robust_score(
            summary.profit_factor.p50,
            summary.pvalue_corrected
        )
        scored.append((summary, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored


def compute_weighted_robust_score(
    pf_p50: float,
    max_dd_p95: float,
    p_corrected: float,
    pf_weight: float = 0.5,
    dd_weight: float = 0.3,
    sig_weight: float = 0.2
) -> float:
    """
    Compute weighted robust score including drawdown penalty.

    Incorporates:
    - Profit factor (positive)
    - Max drawdown P95 (negative - penalize high drawdowns)
    - Statistical significance (positive)

    Args:
        pf_p50: Median profit factor
        max_dd_p95: 95th percentile max drawdown
        p_corrected: Corrected p-value
        pf_weight: Weight for profit factor component
        dd_weight: Weight for drawdown penalty
        sig_weight: Weight for significance

    Returns:
        Weighted robust score
    """
    # Normalize components
    pf_component = min(pf_p50, 10.0) / 10.0  # Cap at 10, normalize to 0-1
    dd_component = 1 - min(max_dd_p95 / 100.0, 1.0)  # Lower DD is better
    sig_component = 1 - p_corrected  # Lower p-value is better

    score = (
        pf_weight * pf_component +
        dd_weight * dd_component +
        sig_weight * sig_component
    )

    # Scale to make it more interpretable
    return score * 10


def categorize_robustness(
    robust_score: float,
    pf_p05: float,
    max_dd_p95: float
) -> str:
    """
    Categorize strategy robustness level.

    Args:
        robust_score: Computed robust score
        pf_p05: 5th percentile profit factor
        max_dd_p95: 95th percentile max drawdown

    Returns:
        Category string: 'excellent', 'good', 'moderate', 'weak', 'poor'
    """
    # Must have positive edge even in worst cases
    if pf_p05 < 1.0:
        return 'poor'

    # Must not have catastrophic drawdown risk
    if max_dd_p95 > 50:
        if robust_score < 1.0:
            return 'poor'
        return 'weak'

    # Score-based categorization
    if robust_score >= 2.0 and pf_p05 >= 1.5:
        return 'excellent'
    elif robust_score >= 1.5 and pf_p05 >= 1.2:
        return 'good'
    elif robust_score >= 1.0 and pf_p05 >= 1.0:
        return 'moderate'
    elif robust_score >= 0.5:
        return 'weak'
    else:
        return 'poor'


def get_top_cells(
    cell_summaries: List[CellSummary],
    n: int = 50,
    baseline_pf: float = 1.0
) -> List[Dict]:
    """
    Get top N cells by robust score with detailed info.

    Args:
        cell_summaries: List of CellSummary objects
        n: Number of top cells to return
        baseline_pf: Baseline for reference

    Returns:
        List of dictionaries with cell info
    """
    ranked = rank_cells_by_robust_score(cell_summaries, baseline_pf)

    results = []
    for summary, score in ranked[:n]:
        category = categorize_robustness(
            score,
            summary.profit_factor.p05,
            summary.max_drawdown.p95
        )

        results.append({
            'cell_id': summary.cell_id,
            'config': summary.config.to_dict(),
            'robust_score': score,
            'category': category,
            'pf_p50': summary.profit_factor.p50,
            'pf_p05': summary.profit_factor.p05,
            'return_p50': summary.total_return.p50,
            'max_dd_p95': summary.max_drawdown.p95,
            'pvalue_corrected': summary.pvalue_corrected,
            'n_permutations': summary.n_permutations
        })

    return results
