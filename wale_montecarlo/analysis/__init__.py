"""
Statistical analysis for Monte Carlo results.

Provides tools for:
- Quantile analysis
- P-value testing
- Robust scoring
- Pareto front identification
- Plateau clustering
- Decision report generation
"""

from .quantiles import compute_quantiles, compute_distribution_stats
from .pvalue import compute_pvalue, apply_bonferroni
from .robust_score import compute_robust_score, rank_cells_by_robust_score
from .pareto import find_pareto_front_2d, find_pareto_front_3d
from .clustering import find_plateau_clusters
from .report import generate_decision_report

__all__ = [
    'compute_quantiles',
    'compute_distribution_stats',
    'compute_pvalue',
    'apply_bonferroni',
    'compute_robust_score',
    'rank_cells_by_robust_score',
    'find_pareto_front_2d',
    'find_pareto_front_3d',
    'find_plateau_clusters',
    'generate_decision_report',
]
