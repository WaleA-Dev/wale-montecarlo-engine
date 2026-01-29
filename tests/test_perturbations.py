"""Tests for perturbation modules."""

import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wale_montecarlo.models import ShuffleMode, BootstrapMode
from wale_montecarlo.perturbations.skip import apply_skip
from wale_montecarlo.perturbations.slippage import apply_slippage
from wale_montecarlo.perturbations.delay import apply_delay
from wale_montecarlo.perturbations.shuffle import apply_shuffle
from wale_montecarlo.perturbations.bootstrap import apply_bootstrap
from wale_montecarlo.perturbations.pipeline import apply_all_perturbations


class TestSkip:
    """Tests for trade skipping perturbation."""

    def test_no_skip(self, sample_trades_small):
        """Zero skip probability should return all trades."""
        rng = np.random.default_rng(42)
        result = apply_skip(sample_trades_small, 0.0, rng)

        assert len(result) == len(sample_trades_small)

    def test_full_skip(self, sample_trades_small):
        """100% skip probability should return empty list."""
        rng = np.random.default_rng(42)
        result = apply_skip(sample_trades_small, 1.0, rng)

        assert len(result) == 0

    def test_partial_skip(self, sample_trades):
        """50% skip should remove approximately half the trades."""
        rng = np.random.default_rng(42)
        result = apply_skip(sample_trades, 0.5, rng)

        # Allow some variance
        assert 30 <= len(result) <= 70

    def test_skip_deterministic(self, sample_trades_small):
        """Same seed should produce same result."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        result1 = apply_skip(sample_trades_small, 0.5, rng1)
        result2 = apply_skip(sample_trades_small, 0.5, rng2)

        assert len(result1) == len(result2)

    def test_skip_creates_copies(self, sample_trades_small):
        """Skipping should return copies, not references."""
        rng = np.random.default_rng(42)
        result = apply_skip(sample_trades_small, 0.0, rng)

        # Modify result
        result[0].pnl = 999999

        # Original should be unchanged
        assert sample_trades_small[0].pnl != 999999


class TestSlippage:
    """Tests for slippage perturbation."""

    def test_no_slippage(self, sample_trades_small):
        """Zero slippage should not change PnL."""
        rng = np.random.default_rng(42)
        result = apply_slippage(sample_trades_small, 0.0, rng)

        for orig, new in zip(sample_trades_small, result):
            assert orig.pnl == new.pnl

    def test_slippage_reduces_pnl(self, sample_trades_small):
        """Slippage should reduce PnL."""
        rng = np.random.default_rng(42)
        result = apply_slippage(sample_trades_small, 100.0, rng)

        total_orig = sum(t.pnl for t in sample_trades_small)
        total_new = sum(t.pnl for t in result)

        # Slippage always hurts
        assert total_new < total_orig

    def test_slippage_bounded(self, sample_trades_small):
        """Slippage should be bounded by max value."""
        rng = np.random.default_rng(42)
        max_slip = 50.0
        result = apply_slippage(sample_trades_small, max_slip, rng)

        for orig, new in zip(sample_trades_small, result):
            slip = orig.pnl - new.pnl
            assert 0 <= slip <= max_slip

    def test_slippage_deterministic(self, sample_trades_small):
        """Same seed should produce same result."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        result1 = apply_slippage(sample_trades_small, 100.0, rng1)
        result2 = apply_slippage(sample_trades_small, 100.0, rng2)

        for r1, r2 in zip(result1, result2):
            assert r1.pnl == r2.pnl


class TestDelay:
    """Tests for execution delay perturbation."""

    def test_no_delay(self, sample_trades_small):
        """Zero delay should not change trades."""
        rng = np.random.default_rng(42)
        result = apply_delay(sample_trades_small, 0, rng)

        assert len(result) == len(sample_trades_small)

    def test_delay_approximate_mode(self, sample_trades_small):
        """Delay without OHLC should use approximate mode."""
        rng = np.random.default_rng(42)
        result = apply_delay(sample_trades_small, 2, rng, ohlc_data=None)

        # Should still return trades
        assert len(result) > 0

    def test_delay_hurts_performance(self, sample_trades):
        """Delay should not improve overall performance."""
        rng = np.random.default_rng(42)
        result = apply_delay(sample_trades, 2, rng)

        # Conservative: delays only hurt
        total_orig = sum(t.pnl for t in sample_trades)
        total_new = sum(t.pnl for t in result)

        assert total_new <= total_orig


class TestShuffle:
    """Tests for sequence shuffling perturbation."""

    def test_no_shuffle(self, sample_trades_small):
        """None mode should preserve order."""
        rng = np.random.default_rng(42)
        result = apply_shuffle(sample_trades_small, ShuffleMode.NONE, 10, rng)

        for orig, new in zip(sample_trades_small, result):
            assert orig.trade_id == new.trade_id

    def test_permute_changes_order(self, sample_trades_small):
        """Permute should change trade order."""
        rng = np.random.default_rng(42)
        result = apply_shuffle(sample_trades_small, ShuffleMode.PERMUTE, 10, rng)

        # Order should be different (very unlikely to be same)
        orig_ids = [t.trade_id for t in sample_trades_small]
        new_ids = [t.trade_id for t in result]

        assert orig_ids != new_ids
        assert sorted(orig_ids) == sorted(new_ids)  # Same trades, different order

    def test_block_permute(self, sample_trades):
        """Block permute should shuffle blocks."""
        rng = np.random.default_rng(42)
        result = apply_shuffle(sample_trades, ShuffleMode.BLOCK_PERMUTE, 10, rng)

        # Should have same number of trades
        assert len(result) == len(sample_trades)

    def test_shuffle_preserves_trade_count(self, sample_trades_small):
        """Shuffle should not change number of trades."""
        rng = np.random.default_rng(42)
        result = apply_shuffle(sample_trades_small, ShuffleMode.PERMUTE, 10, rng)

        assert len(result) == len(sample_trades_small)


class TestBootstrap:
    """Tests for bootstrap resampling perturbation."""

    def test_no_bootstrap(self, sample_trades_small):
        """None mode should return copies."""
        rng = np.random.default_rng(42)
        result = apply_bootstrap(sample_trades_small, BootstrapMode.NONE, 10, rng)

        assert len(result) == len(sample_trades_small)

    def test_trade_bootstrap_same_length(self, sample_trades_small):
        """Trade bootstrap should return same number of trades."""
        rng = np.random.default_rng(42)
        result = apply_bootstrap(sample_trades_small, BootstrapMode.TRADE_BOOTSTRAP, 10, rng)

        assert len(result) == len(sample_trades_small)

    def test_bootstrap_allows_duplicates(self, sample_trades_small):
        """Bootstrap should allow duplicate trades."""
        rng = np.random.default_rng(42)
        result = apply_bootstrap(sample_trades_small, BootstrapMode.TRADE_BOOTSTRAP, 10, rng)

        # With replacement, some trades may appear multiple times
        # Check by looking at PnL distribution
        orig_pnls = sorted([t.pnl for t in sample_trades_small])
        new_pnls = sorted([t.pnl for t in result])

        # Likely to have some duplicates
        assert orig_pnls != new_pnls or len(set(new_pnls)) < len(new_pnls)

    def test_block_bootstrap(self, sample_trades):
        """Block bootstrap should preserve local structure."""
        rng = np.random.default_rng(42)
        result = apply_bootstrap(sample_trades, BootstrapMode.BLOCK_BOOTSTRAP, 10, rng)

        # Approximately same length
        assert abs(len(result) - len(sample_trades)) <= 10


class TestPipeline:
    """Tests for full perturbation pipeline."""

    def test_pipeline_baseline(self, sample_trades_small, baseline_config):
        """Baseline config should not perturb."""
        rng = np.random.default_rng(42)
        result = apply_all_perturbations(sample_trades_small, baseline_config, rng)

        assert len(result) == len(sample_trades_small)

        total_orig = sum(t.pnl for t in sample_trades_small)
        total_new = sum(t.pnl for t in result)
        assert total_orig == total_new

    def test_pipeline_moderate(self, sample_trades, moderate_config):
        """Moderate config should perturb trades."""
        rng = np.random.default_rng(42)
        result = apply_all_perturbations(sample_trades, moderate_config, rng)

        # Should have fewer trades (skip)
        assert len(result) < len(sample_trades)

    def test_pipeline_stress(self, sample_trades, stress_config):
        """Stress config should significantly perturb."""
        rng = np.random.default_rng(42)
        result = apply_all_perturbations(sample_trades, stress_config, rng)

        # Significant reduction in trades and performance
        total_orig = sum(t.pnl for t in sample_trades)
        total_new = sum(t.pnl for t in result)

        assert total_new < total_orig

    def test_pipeline_deterministic(self, sample_trades, moderate_config):
        """Same seed should produce same result."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        result1 = apply_all_perturbations(sample_trades, moderate_config, rng1)
        result2 = apply_all_perturbations(sample_trades, moderate_config, rng2)

        assert len(result1) == len(result2)

        for r1, r2 in zip(result1, result2):
            assert r1.pnl == r2.pnl
