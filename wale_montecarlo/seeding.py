"""
Deterministic seeding for reproducible Monte Carlo simulations.

This module implements the seeding scheme described in the documentation,
ensuring that:
1. Same cell_id + perm_index always produces identical results
2. Different cells/permutations get statistically independent sequences
3. Seeds are crash-safe and resume-compatible

The seeding scheme uses SHA256 for cell seeds and a prime multiplier
for per-permutation seeds.
"""

import hashlib
from typing import Tuple
import numpy as np


# Prime multiplier for permutation seed generation
# Chosen to be large prime that spreads seeds well across uint32 space
PERM_SEED_MULTIPLIER = 1000003

# Modulus for seed calculation (2^32 for numpy compatibility)
SEED_MODULUS = 2**32


def compute_cell_seed(cell_id: str) -> int:
    """
    Compute deterministic seed for a cell based on its ID.

    Uses SHA256 hash of the cell_id string, taking first 8 hex chars
    as a 32-bit integer. This ensures:
    - Same cell_id always produces same seed
    - Different cell_ids get well-distributed seeds
    - Seed fits in numpy's uint32 range

    Args:
        cell_id: Unique string identifier for the cell
                 (e.g., "skip0.05_slip100_delay1_shufnone_bootnone_blk10")

    Returns:
        32-bit unsigned integer seed
    """
    hash_bytes = hashlib.sha256(cell_id.encode('utf-8')).hexdigest()
    return int(hash_bytes[:8], 16)


def compute_perm_seed(cell_seed: int, perm_index: int) -> int:
    """
    Compute deterministic seed for a specific permutation within a cell.

    Uses linear congruential formula with prime multiplier:
        perm_seed = (cell_seed + perm_index * PRIME) mod 2^32

    This ensures:
    - Same cell_seed + perm_index always produces same result
    - Sequential perm_indices get well-separated seeds
    - No collisions within reasonable perm_index ranges

    Args:
        cell_seed: Base seed for the cell (from compute_cell_seed)
        perm_index: Index of this permutation (0 to n_perms-1)

    Returns:
        32-bit unsigned integer seed
    """
    return (cell_seed + perm_index * PERM_SEED_MULTIPLIER) % SEED_MODULUS


def get_rng_for_permutation(cell_id: str, perm_index: int) -> np.random.Generator:
    """
    Get a numpy random generator seeded for a specific permutation.

    Convenience function combining cell seed and perm seed computation
    with numpy Generator creation.

    Args:
        cell_id: Unique string identifier for the cell
        perm_index: Index of this permutation

    Returns:
        Seeded numpy random Generator instance
    """
    cell_seed = compute_cell_seed(cell_id)
    perm_seed = compute_perm_seed(cell_seed, perm_index)
    return np.random.default_rng(perm_seed)


def get_seeds_for_cell(cell_id: str, n_perms: int, start_perm: int = 0) -> Tuple[int, list]:
    """
    Get all permutation seeds for a cell.

    Useful for pre-computing seeds or verification.

    Args:
        cell_id: Unique string identifier for the cell
        n_perms: Number of permutations to generate seeds for
        start_perm: Starting permutation index (for resume support)

    Returns:
        Tuple of (cell_seed, list of perm_seeds)
    """
    cell_seed = compute_cell_seed(cell_id)
    perm_seeds = [
        compute_perm_seed(cell_seed, i)
        for i in range(start_perm, start_perm + n_perms)
    ]
    return cell_seed, perm_seeds


def verify_seed_uniqueness(cell_id: str, n_perms: int) -> bool:
    """
    Verify that all permutation seeds for a cell are unique.

    Used for testing and validation.

    Args:
        cell_id: Cell identifier to check
        n_perms: Number of permutations to verify

    Returns:
        True if all seeds are unique, False otherwise
    """
    _, perm_seeds = get_seeds_for_cell(cell_id, n_perms)
    return len(perm_seeds) == len(set(perm_seeds))


def seed_info(cell_id: str, perm_index: int) -> dict:
    """
    Get detailed seed information for debugging.

    Args:
        cell_id: Cell identifier
        perm_index: Permutation index

    Returns:
        Dictionary with cell_seed, perm_seed, and verification info
    """
    cell_seed = compute_cell_seed(cell_id)
    perm_seed = compute_perm_seed(cell_seed, perm_index)

    return {
        "cell_id": cell_id,
        "perm_index": perm_index,
        "cell_seed": cell_seed,
        "cell_seed_hex": f"0x{cell_seed:08x}",
        "perm_seed": perm_seed,
        "perm_seed_hex": f"0x{perm_seed:08x}",
    }
