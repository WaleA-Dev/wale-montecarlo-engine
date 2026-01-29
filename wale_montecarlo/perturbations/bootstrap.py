"""
Bootstrap resampling perturbation.

Tests statistical significance by resampling trades with replacement:
- trade_bootstrap: Individual trades resampled
- block_bootstrap: Blocks of trades resampled (preserves autocorrelation)

Bootstrap creates "synthetic" histories to estimate confidence intervals
and test if observed performance could be due to chance.
"""

from typing import List
import numpy as np

from ..models import Trade, BootstrapMode


def apply_bootstrap(
    trades: List[Trade],
    mode: BootstrapMode,
    block_len: int,
    rng: np.random.Generator
) -> List[Trade]:
    """
    Bootstrap resample trades with replacement.

    Modes:
    - none: No resampling
    - trade_bootstrap: Resample individual trades
    - block_bootstrap: Resample blocks (preserves local correlation)

    The resulting sample has the same number of trades as input,
    but some trades appear multiple times and others not at all.

    Args:
        trades: List of Trade objects
        mode: BootstrapMode enum value
        block_len: Block size for block_bootstrap
        rng: Seeded numpy random generator

    Returns:
        New list of bootstrapped trades (same length as input)
    """
    if len(trades) == 0:
        return []

    if mode == BootstrapMode.NONE:
        return [t.copy() for t in trades]

    if mode == BootstrapMode.TRADE_BOOTSTRAP:
        return _apply_trade_bootstrap(trades, rng)

    if mode == BootstrapMode.BLOCK_BOOTSTRAP:
        return _apply_block_bootstrap(trades, block_len, rng)

    # Default: no bootstrap
    return [t.copy() for t in trades]


def _apply_trade_bootstrap(
    trades: List[Trade],
    rng: np.random.Generator
) -> List[Trade]:
    """
    Resample individual trades with replacement.

    Each position in the result is independently sampled from the
    original trades. This breaks temporal structure but tests if
    the average trade characteristics produce similar results.
    """
    n = len(trades)

    # Sample n indices with replacement
    indices = rng.choice(n, size=n, replace=True)

    # Build result with copied trades
    result = []
    for i, idx in enumerate(indices):
        new_trade = trades[idx].copy()
        # Assign new trade_id to distinguish resampled trades
        new_trade.trade_id = i
        result.append(new_trade)

    return result


def _apply_block_bootstrap(
    trades: List[Trade],
    block_len: int,
    rng: np.random.Generator
) -> List[Trade]:
    """
    Resample blocks of trades with replacement.

    Preserves autocorrelation structure within blocks while
    testing if the observed sequence of blocks matters.

    The total number of trades returned approximately matches
    the input count.
    """
    if block_len <= 0:
        block_len = 1

    n = len(trades)

    # Create blocks
    blocks = []
    for i in range(0, n, block_len):
        block = trades[i:i + block_len]
        blocks.append(block)

    n_blocks = len(blocks)
    if n_blocks == 0:
        return []

    # Calculate how many blocks we need to approximately match n trades
    avg_block_size = n / n_blocks
    n_blocks_needed = int(np.ceil(n / avg_block_size))

    # Sample blocks with replacement
    block_indices = rng.choice(n_blocks, size=n_blocks_needed, replace=True)

    # Build result
    result = []
    trade_id = 0
    for block_idx in block_indices:
        for trade in blocks[block_idx]:
            new_trade = trade.copy()
            new_trade.trade_id = trade_id
            result.append(new_trade)
            trade_id += 1

    # Trim to original length if we overshot
    if len(result) > n:
        result = result[:n]

    return result


def apply_bootstrap_stratified(
    trades: List[Trade],
    rng: np.random.Generator
) -> List[Trade]:
    """
    Stratified bootstrap: separately resample winners and losers.

    Preserves the win/loss ratio while resampling within each group.
    This tests if the observed performance depends on the specific
    winning/losing trades or just the overall distribution.

    Args:
        trades: List of Trade objects
        rng: Seeded numpy random generator

    Returns:
        Stratified bootstrapped trades
    """
    if len(trades) == 0:
        return []

    # Separate winners and losers
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]

    n_winners = len(winners)
    n_losers = len(losers)

    result = []
    trade_id = 0

    # Bootstrap winners
    if n_winners > 0:
        winner_indices = rng.choice(n_winners, size=n_winners, replace=True)
        for idx in winner_indices:
            new_trade = winners[idx].copy()
            new_trade.trade_id = trade_id
            result.append(new_trade)
            trade_id += 1

    # Bootstrap losers
    if n_losers > 0:
        loser_indices = rng.choice(n_losers, size=n_losers, replace=True)
        for idx in loser_indices:
            new_trade = losers[idx].copy()
            new_trade.trade_id = trade_id
            result.append(new_trade)
            trade_id += 1

    # Shuffle combined result to mix winners and losers
    rng.shuffle(result)

    return result


def apply_bootstrap_moving_block(
    trades: List[Trade],
    block_len: int,
    rng: np.random.Generator
) -> List[Trade]:
    """
    Moving block bootstrap (overlapping blocks).

    Unlike standard block bootstrap which uses non-overlapping blocks,
    this samples from all possible starting positions, creating
    overlapping blocks. This provides more blocks to choose from
    and better preserves the time series structure.

    Args:
        trades: List of Trade objects
        block_len: Size of each block
        rng: Seeded numpy random generator

    Returns:
        Moving-block bootstrapped trades
    """
    if len(trades) == 0:
        return []

    if block_len <= 0:
        block_len = 1

    n = len(trades)

    # Number of possible starting positions
    n_starts = n - block_len + 1
    if n_starts <= 0:
        # Block longer than trade list - just copy
        return [t.copy() for t in trades]

    # Calculate how many blocks we need
    n_blocks_needed = int(np.ceil(n / block_len))

    result = []
    trade_id = 0

    for _ in range(n_blocks_needed):
        # Random starting position
        start = rng.integers(0, n_starts)

        # Extract block
        for i in range(start, min(start + block_len, n)):
            new_trade = trades[i].copy()
            new_trade.trade_id = trade_id
            result.append(new_trade)
            trade_id += 1

    # Trim to original length
    if len(result) > n:
        result = result[:n]

    return result
