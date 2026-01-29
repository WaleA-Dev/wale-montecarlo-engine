"""
Trade skipping perturbation.

Simulates trades that would have been missed due to:
- Technical issues (system outages, connectivity)
- Risk management limits being hit
- Manual intervention or hesitation
- Broker execution failures

Uses Bernoulli probability per trade.
"""

from typing import List
import numpy as np

from ..models import Trade


def apply_skip(
    trades: List[Trade],
    p_skip: float,
    rng: np.random.Generator
) -> List[Trade]:
    """
    Randomly skip trades with probability p_skip.

    Each trade independently has probability p_skip of being removed.
    This simulates real-world scenarios where not all signals are executed.

    Args:
        trades: List of Trade objects
        p_skip: Probability of skipping each trade (0.0 to 0.10 typical)
        rng: Seeded numpy random generator

    Returns:
        New list of trades with some randomly removed

    Example:
        With p_skip=0.05, approximately 5% of trades will be skipped.
        A 100-trade backtest might have 95 trades remaining.
    """
    if p_skip <= 0.0:
        return [t.copy() for t in trades]

    if p_skip >= 1.0:
        return []

    # Generate Bernoulli mask
    n = len(trades)
    keep_mask = rng.random(n) > p_skip

    # Filter trades
    return [t.copy() for t, keep in zip(trades, keep_mask) if keep]


def apply_skip_targeted(
    trades: List[Trade],
    p_skip: float,
    rng: np.random.Generator,
    skip_winners: bool = False
) -> List[Trade]:
    """
    Skip trades with bias toward winners or losers.

    Simulates scenarios where certain types of trades are more likely
    to be missed (e.g., hesitating on winners, or system crashes during
    volatile moves that tend to be losers).

    Args:
        trades: List of Trade objects
        p_skip: Base probability of skipping
        rng: Seeded numpy random generator
        skip_winners: If True, winners are more likely to be skipped;
                     if False (default), losers are more likely

    Returns:
        New list of trades with biased skipping applied
    """
    if p_skip <= 0.0:
        return [t.copy() for t in trades]

    result = []
    for trade in trades:
        is_winner = trade.pnl > 0

        # Adjust skip probability based on outcome
        if skip_winners:
            # 1.5x more likely to skip winners
            adj_p = p_skip * 1.5 if is_winner else p_skip * 0.5
        else:
            # 1.5x more likely to skip losers
            adj_p = p_skip * 0.5 if is_winner else p_skip * 1.5

        # Cap at 1.0
        adj_p = min(adj_p, 1.0)

        if rng.random() > adj_p:
            result.append(trade.copy())

    return result
