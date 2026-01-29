"""
Perturbation models for Monte Carlo stress testing.

This package implements five core perturbation types:
1. Trade Skipping - Randomly skip trades (technical issues, outages)
2. Slippage - Add random execution costs
3. Execution Delay - Delay entries by N bars
4. Sequence Shuffling - Randomize trade order
5. Bootstrap Resampling - Resample with replacement

Plus state-dependent multipliers for volatility/drawdown awareness.
"""

from .skip import apply_skip
from .slippage import apply_slippage
from .delay import apply_delay
from .shuffle import apply_shuffle
from .bootstrap import apply_bootstrap
from .state_dependent import (
    compute_volatility_multiplier,
    compute_drawdown_multiplier
)
from .pipeline import apply_all_perturbations

__all__ = [
    'apply_skip',
    'apply_slippage',
    'apply_delay',
    'apply_shuffle',
    'apply_bootstrap',
    'compute_volatility_multiplier',
    'compute_drawdown_multiplier',
    'apply_all_perturbations',
]
