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

    Formula: PF_P50 × (1 - p_corrected)

    High score requires both:
    - High median profit factor
    - Low p-value (high significance)

    Args:
        pf_p50: Median profit factor (P50)
        p_corrected: Bonferroni-corrected p-value

    Returns:
        Robust score (higher is better)
    """
    # Ensure p_corrected is in valid range
    p_corrected = max(0.0, min(1.0, p_corrected))

    # Cap profit factor at reasonable value
    pf_p50 = max(0.0, min(999.0, pf_p50))

    return pf_p50 * (1 - p_corrected)


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
