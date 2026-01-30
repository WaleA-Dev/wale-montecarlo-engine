"""
Statistical analysis for Monte Carlo results.

Provides tools for:
- Quantile analysis
- P-value testing (BH and Bonferroni correction)
- Robust scoring (v1 and v3 with drawdown penalty)
- Pareto front identification
- Plateau clustering
- Convergence analysis
- Overfitting classification
- Decision report generation
"""

from .quantiles import compute_quantiles, compute_distribution_stats
from .pvalue import compute_pvalue, apply_bonferroni, benjamini_hochberg, apply_correction
from .robust_score import (
    compute_robust_score,
    compute_robust_score_v3,
    classify_overfit,
    compute_degradation_rate,
    rank_cells_by_robust_score,
)
from .pareto import find_pareto_front_2d, find_pareto_front_3d
from .clustering import find_plateau_clusters
from .report import generate_decision_report
from .convergence import (
    convergence_analysis,
    percentile_confidence_interval,
    pareto_rank,
    find_pareto_front,
    compute_overfit_score,
)
from .interactions import (
    analyze_interactions,
    interaction_adjusted_prediction,
    summarize_interactions,
)

__all__ = [
    'compute_quantiles',
    'compute_distribution_stats',
    'compute_pvalue',
    'apply_bonferroni',
    'benjamini_hochberg',
    'apply_correction',
    'compute_robust_score',
    'compute_robust_score_v3',
    'classify_overfit',
    'compute_degradation_rate',
    'rank_cells_by_robust_score',
    'find_pareto_front_2d',
    'find_pareto_front_3d',
    'find_plateau_clusters',
    'generate_decision_report',
    'convergence_analysis',
    'percentile_confidence_interval',
    'pareto_rank',
    'find_pareto_front',
    'compute_overfit_score',
    'analyze_interactions',
    'interaction_adjusted_prediction',
    'summarize_interactions',
]

