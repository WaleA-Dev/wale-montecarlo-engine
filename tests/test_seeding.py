"""Tests for seeding module."""

import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wale_montecarlo.seeding import (
    compute_cell_seed,
    compute_perm_seed,
    get_rng_for_permutation,
    get_seeds_for_cell,
    verify_seed_uniqueness,
    seed_info
)


class TestCellSeed:
    """Tests for cell seed computation."""

    def test_deterministic(self):
        """Same cell_id should produce same seed."""
        cell_id = "skip0.05_slip100_delay1_shufnone_bootnone_blk10"

        seed1 = compute_cell_seed(cell_id)
        seed2 = compute_cell_seed(cell_id)

        assert seed1 == seed2

    def test_different_ids_different_seeds(self):
        """Different cell_ids should produce different seeds."""
        id1 = "skip0.05_slip100_delay1_shufnone_bootnone_blk10"
        id2 = "skip0.10_slip200_delay2_shufpermute_bootnone_blk10"

        seed1 = compute_cell_seed(id1)
        seed2 = compute_cell_seed(id2)

        assert seed1 != seed2

    def test_seed_in_valid_range(self):
        """Seed should fit in uint32."""
        cell_id = "test_cell_id"
        seed = compute_cell_seed(cell_id)

        assert 0 <= seed < 2**32


class TestPermSeed:
    """Tests for permutation seed computation."""

    def test_deterministic(self):
        """Same cell_seed + perm_index should produce same seed."""
        cell_seed = 12345
        perm_index = 100

        seed1 = compute_perm_seed(cell_seed, perm_index)
        seed2 = compute_perm_seed(cell_seed, perm_index)

        assert seed1 == seed2

    def test_different_indices_different_seeds(self):
        """Different perm_indices should produce different seeds."""
        cell_seed = 12345

        seeds = [compute_perm_seed(cell_seed, i) for i in range(100)]

        assert len(set(seeds)) == 100

    def test_seed_distribution(self):
        """Seeds should be well-distributed."""
        cell_seed = 12345
        seeds = [compute_perm_seed(cell_seed, i) for i in range(10000)]

        # Check standard deviation is reasonably high
        std = np.std(seeds)
        assert std > 1e9  # Should be well-spread over uint32 range


class TestRngForPermutation:
    """Tests for RNG generation."""

    def test_deterministic_rng(self):
        """Same cell_id + perm_index should produce same random sequence."""
        cell_id = "test_cell"
        perm_index = 42

        rng1 = get_rng_for_permutation(cell_id, perm_index)
        rng2 = get_rng_for_permutation(cell_id, perm_index)

        # Generate same sequence of random numbers
        vals1 = [rng1.random() for _ in range(10)]
        vals2 = [rng2.random() for _ in range(10)]

        assert vals1 == vals2

    def test_different_permutations_different_sequences(self):
        """Different permutations should produce different sequences."""
        cell_id = "test_cell"

        rng1 = get_rng_for_permutation(cell_id, 0)
        rng2 = get_rng_for_permutation(cell_id, 1)

        vals1 = [rng1.random() for _ in range(10)]
        vals2 = [rng2.random() for _ in range(10)]

        assert vals1 != vals2


class TestSeedUniqueness:
    """Tests for seed uniqueness verification."""

    def test_unique_seeds(self):
        """Verify all seeds are unique for typical cell."""
        cell_id = "skip0.05_slip100_delay1_shufnone_bootnone_blk10"
        n_perms = 10000

        assert verify_seed_uniqueness(cell_id, n_perms)

    def test_many_permutations(self):
        """Verify uniqueness holds for large number of permutations."""
        cell_id = "test_cell"
        n_perms = 100000

        # This should still be unique due to prime multiplier
        assert verify_seed_uniqueness(cell_id, n_perms)


class TestSeedInfo:
    """Tests for seed debugging info."""

    def test_info_structure(self):
        """Verify seed_info returns expected structure."""
        cell_id = "test_cell"
        perm_index = 42

        info = seed_info(cell_id, perm_index)

        assert 'cell_id' in info
        assert 'perm_index' in info
        assert 'cell_seed' in info
        assert 'perm_seed' in info
        assert 'cell_seed_hex' in info
        assert 'perm_seed_hex' in info

    def test_info_values_consistent(self):
        """Verify info values are consistent with direct computation."""
        cell_id = "test_cell"
        perm_index = 42

        info = seed_info(cell_id, perm_index)

        assert info['cell_seed'] == compute_cell_seed(cell_id)
        assert info['perm_seed'] == compute_perm_seed(info['cell_seed'], perm_index)
