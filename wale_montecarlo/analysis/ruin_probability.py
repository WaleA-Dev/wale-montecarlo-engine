"""
Ruin probability analysis.

Estimates the probability of hitting various drawdown thresholds
given a trade history and starting capital.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class RuinResult:
    """Results from ruin probability analysis."""
    starting_capital: float
    
    # Probability of hitting each threshold
    prob_10pct_dd: float  # 10% drawdown
    prob_20pct_dd: float  # 20% drawdown
    prob_30pct_dd: float  # 30% drawdown
    prob_50pct_dd: float  # 50% drawdown (typical "ruin")
    
    # Custom threshold if provided
    custom_threshold: Optional[float]
    prob_custom: Optional[float]
    
    # Expected stats
    expected_max_dd_pct: float
    expected_max_dd_p95_pct: float
    
    # Recommended minimum capital
    recommended_capital: float  # Capital where P(50% DD) < 5%
    
    # Raw data
    max_dd_distribution: np.ndarray
    n_simulations: int


def estimate_ruin_probability(
    trades: List,
    starting_capital: float,
    ruin_threshold: float = 0.50,
    n_simulations: int = 10000,
    seed: Optional[int] = None
) -> RuinResult:
    """
    Estimate probability of hitting drawdown thresholds.
    
    Args:
        trades: List of Trade objects with pnl attribute
        starting_capital: Starting account value
        ruin_threshold: Custom drawdown threshold (default: 50%)
        n_simulations: Number of Monte Carlo paths
        seed: Random seed for reproducibility
    
    Returns:
        RuinResult with probabilities and recommendations
    """
    if not trades:
        raise ValueError("No trades provided")
    if starting_capital <= 0:
        raise ValueError("Starting capital must be positive")
    
    rng = np.random.default_rng(seed)
    n_trades = len(trades)
    
    # Extract PnL values
    pnls = np.array([t.pnl for t in trades])
    
    # Track maximum drawdowns across simulations
    max_dd_pcts = np.zeros(n_simulations)
    
    for i in range(n_simulations):
        # Resample trades with replacement
        indices = rng.choice(n_trades, size=n_trades, replace=True)
        resampled_pnls = pnls[indices]
        
        # Build equity curve starting at capital
        equity = starting_capital + np.cumsum(resampled_pnls)
        equity = np.concatenate([[starting_capital], equity])
        
        # Compute drawdown as percentage of peak
        running_max = np.maximum.accumulate(equity)
        drawdowns_pct = (running_max - equity) / running_max
        max_dd_pcts[i] = np.max(drawdowns_pct)
    
    # Compute probabilities for standard thresholds
    prob_10 = np.mean(max_dd_pcts >= 0.10)
    prob_20 = np.mean(max_dd_pcts >= 0.20)
    prob_30 = np.mean(max_dd_pcts >= 0.30)
    prob_50 = np.mean(max_dd_pcts >= 0.50)
    prob_custom = np.mean(max_dd_pcts >= ruin_threshold) if ruin_threshold not in [0.10, 0.20, 0.30, 0.50] else None
    
    # Estimate recommended capital (where P(50% DD) < 5%)
    # Binary search for the right capital level
    recommended = find_safe_capital(trades, pnls, rng, n_simulations // 10)
    
    return RuinResult(
        starting_capital=starting_capital,
        prob_10pct_dd=prob_10,
        prob_20pct_dd=prob_20,
        prob_30pct_dd=prob_30,
        prob_50pct_dd=prob_50,
        custom_threshold=ruin_threshold if prob_custom is not None else None,
        prob_custom=prob_custom,
        expected_max_dd_pct=np.median(max_dd_pcts),
        expected_max_dd_p95_pct=np.percentile(max_dd_pcts, 95),
        recommended_capital=recommended,
        max_dd_distribution=max_dd_pcts,
        n_simulations=n_simulations
    )


def find_safe_capital(
    trades: List,
    pnls: np.ndarray,
    rng: np.random.Generator,
    n_sims: int,
    target_prob: float = 0.05,
    target_dd: float = 0.50
) -> float:
    """
    Find minimum capital where P(DD > target_dd) < target_prob.
    
    Uses binary search to find the appropriate capital level.
    """
    n_trades = len(pnls)
    
    # Estimate max potential loss
    max_loss = abs(np.min(np.cumsum(pnls)))
    
    # Search range
    low = max_loss * 1.5  # Minimum reasonable capital
    high = max_loss * 10  # Upper bound
    
    for _ in range(10):  # Binary search iterations
        mid = (low + high) / 2
        
        # Test this capital level
        hit_ruin = 0
        for _ in range(n_sims):
            indices = rng.choice(n_trades, size=n_trades, replace=True)
            equity = mid + np.cumsum(pnls[indices])
            equity = np.concatenate([[mid], equity])
            running_max = np.maximum.accumulate(equity)
            max_dd = np.max((running_max - equity) / running_max)
            if max_dd >= target_dd:
                hit_ruin += 1
        
        prob = hit_ruin / n_sims
        
        if prob > target_prob:
            low = mid  # Need more capital
        else:
            high = mid  # Can use less
    
    return high  # Return conservative estimate


def format_ruin_summary(result: RuinResult) -> str:
    """Format ruin analysis as markdown summary."""
    lines = [
        "## Ruin Probability Analysis",
        "",
        f"**Starting Capital:** ${result.starting_capital:,.0f}",
        f"**Simulations:** {result.n_simulations:,}",
        "",
        "### Drawdown Probabilities",
        "",
        "| Threshold | Probability |",
        "|-----------|-------------|",
        f"| 10% DD | {result.prob_10pct_dd:.1%} |",
        f"| 20% DD | {result.prob_20pct_dd:.1%} |",
        f"| 30% DD | {result.prob_30pct_dd:.1%} |",
        f"| 50% DD (Ruin) | {result.prob_50pct_dd:.1%} |",
        "",
        "### Expected Drawdown",
        f"- **Median Max DD:** {result.expected_max_dd_pct:.1%}",
        f"- **P95 Max DD:** {result.expected_max_dd_p95_pct:.1%}",
        "",
        "### Recommendation",
        f"**Minimum Safe Capital:** ${result.recommended_capital:,.0f}",
        f"(Capital where P(50% DD) < 5%)",
    ]
    return "\n".join(lines)
