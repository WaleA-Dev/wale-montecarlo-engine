"""
Trade sequence shuffling perturbation.

Tests sensitivity to trade order by:
- Full random permutation (permute)
- Block-based permutation (block_permute)

This reveals if performance depends on specific trade sequences
or if the edge exists regardless of timing.
"""

from typing import List
import numpy as np

from ..models import Trade, ShuffleMode


def apply_shuffle(
    trades: List[Trade],
    mode: ShuffleMode,
    block_len: int,
    rng: np.random.Generator
) -> List[Trade]:
    """
    Shuffle trade sequence according to specified mode.

    Modes:
    - none: Keep original order
    - permute: Full random shuffle
    - block_permute: Shuffle blocks of block_len trades (preserves local structure)

    Args:
        trades: List of Trade objects
        mode: ShuffleMode enum value
        block_len: Block size for block_permute mode
        rng: Seeded numpy random generator

    Returns:
        New list of trades in shuffled order
    """
    if len(trades) <= 1:
        return [t.copy() for t in trades]

    if mode == ShuffleMode.NONE:
        return [t.copy() for t in trades]

    if mode == ShuffleMode.PERMUTE:
        return _apply_full_permute(trades, rng)

    if mode == ShuffleMode.BLOCK_PERMUTE:
        return _apply_block_permute(trades, block_len, rng)

    # Default: no shuffle
    return [t.copy() for t in trades]


def _apply_full_permute(
    trades: List[Trade],
    rng: np.random.Generator
) -> List[Trade]:
    """
    Fully randomize trade order.

    Every trade has equal probability of appearing at any position.
    """
    n = len(trades)
    indices = rng.permutation(n)
    return [trades[i].copy() for i in indices]


def _apply_block_permute(
    trades: List[Trade],
    block_len: int,
    rng: np.random.Generator
) -> List[Trade]:
    """
    Shuffle blocks of trades while preserving order within blocks.

    This tests sequence dependency while maintaining some temporal structure.
    Trades within a block stay together, but blocks are randomized.

    Example with block_len=3 and 9 trades:
    Original: [0,1,2], [3,4,5], [6,7,8]
    Shuffled: [3,4,5], [6,7,8], [0,1,2] (blocks randomized)
    """
    if block_len <= 0:
        block_len = 1

    n = len(trades)

    # Create blocks
    blocks = []
    for i in range(0, n, block_len):
        block = trades[i:i + block_len]
        blocks.append(block)

    # Shuffle blocks
    n_blocks = len(blocks)
    block_indices = rng.permutation(n_blocks)
    shuffled_blocks = [blocks[i] for i in block_indices]

    # Flatten back to trade list
    result = []
    for block in shuffled_blocks:
        result.extend([t.copy() for t in block])

    return result


def apply_shuffle_partial(
    trades: List[Trade],
    shuffle_fraction: float,
    rng: np.random.Generator
) -> List[Trade]:
    """
    Partially shuffle trades - only a fraction are moved.

    Useful for testing mild sequence dependency.

    Args:
        trades: List of Trade objects
        shuffle_fraction: Fraction of trades to shuffle (0.0 to 1.0)
        rng: Seeded numpy random generator

    Returns:
        New list with partial shuffling
    """
    if shuffle_fraction <= 0.0 or len(trades) <= 1:
        return [t.copy() for t in trades]

    if shuffle_fraction >= 1.0:
        return _apply_full_permute(trades, rng)

    n = len(trades)
    result = [t.copy() for t in trades]

    # Select fraction of positions to shuffle
    n_shuffle = max(2, int(n * shuffle_fraction))
    positions = rng.choice(n, size=n_shuffle, replace=False)

    # Shuffle just those positions
    shuffled_positions = rng.permutation(positions)

    # Create mapping
    temp = [result[pos] for pos in positions]
    for i, pos in enumerate(positions):
        result[pos] = temp[shuffled_positions[i] - positions[0]]

    return result


def apply_shuffle_adjacent(
    trades: List[Trade],
    swap_prob: float,
    rng: np.random.Generator
) -> List[Trade]:
    """
    Randomly swap adjacent trades.

    Simulates small timing variations where trade order might flip.

    Args:
        trades: List of Trade objects
        swap_prob: Probability of swapping each adjacent pair
        rng: Seeded numpy random generator

    Returns:
        New list with adjacent swaps applied
    """
    if swap_prob <= 0.0 or len(trades) <= 1:
        return [t.copy() for t in trades]

    result = [t.copy() for t in trades]

    # Walk through pairs and potentially swap
    i = 0
    while i < len(result) - 1:
        if rng.random() < swap_prob:
            # Swap trades at i and i+1
            result[i], result[i + 1] = result[i + 1], result[i]
            i += 2  # Skip the pair we just swapped
        else:
            i += 1

    return result
