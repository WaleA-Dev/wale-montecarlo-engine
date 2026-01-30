"""
Synthetic validation tests for Monte Carlo engine.

Tests that the engine recovers correct answers on data with known distributions.
"""

import pytest
import numpy as np
from typing import List

from wale_montecarlo.models import Trade, TradeSide
from wale_montecarlo.perturbations.skip import apply_skip
from wale_montecarlo.perturbations.slippage import apply_slippage
from wale_montecarlo.perturbations.shuffle import apply_shuffle


def generate_constant_trades(n: int = 100, pnl: float = 100.0) -> List[Trade]:
    """Generate trades with constant PnL for testing."""
    from datetime import datetime
    
    trades = []
    for i in range(n):
        trades.append(Trade(
            entry_time=datetime(2023, 1, (i % 28) + 1, 10, 0, 0),
            exit_time=datetime(2023, 1, (i % 28) + 1, 14, 0, 0),
            entry_price=100.0,
            exit_price=100.0 + pnl / 10,  # Assuming 10 contracts
            pnl=pnl,
            side=TradeSide.LONG,
            qty=10.0
        ))
    return trades


def generate_random_trades(n: int = 100, mean_pnl: float = 100.0, std_pnl: float = 200.0, seed: int = 42) -> List[Trade]:
    """Generate trades with random PnL for testing."""
    from datetime import datetime
    
    np.random.seed(seed)
    pnls = np.random.normal(mean_pnl, std_pnl, n)
    
    trades = []
    for i, pnl in enumerate(pnls):
        trades.append(Trade(
            entry_time=datetime(2023, 1, (i % 28) + 1, 10, 0, 0),
            exit_time=datetime(2023, 1, (i % 28) + 1, 14, 0, 0),
            entry_price=100.0,
            exit_price=100.0 + pnl / 10,
            pnl=float(pnl),
            side=TradeSide.LONG,
            qty=10.0
        ))
    return trades


class TestZeroPerturbation:
    """With all perturbations at zero, output should exactly match input."""
    
    def test_skip_zero_preserves_all(self):
        """p_skip=0 should preserve all trades."""
        trades = generate_constant_trades(100, pnl=100.0)
        rng = np.random.default_rng(42)
        
        result = apply_skip(trades, p_skip=0.0, rng=rng)
        
        assert len(result) == 100
        assert sum(t.pnl for t in result) == sum(t.pnl for t in trades)
    
    def test_slippage_zero_preserves_pnl(self):
        """slip_max=0 should not change PnL."""
        trades = generate_constant_trades(100, pnl=100.0)
        original_total = sum(t.pnl for t in trades)
        rng = np.random.default_rng(42)
        
        result = apply_slippage(trades, slip_dollars=0.0, rng=rng)
        result_total = sum(t.pnl for t in result)
        
        assert result_total == original_total


class TestSkipDistribution:
    """Skip should remove trades according to specified probability."""
    
    def test_skip_50_percent_removes_half(self):
        """p_skip=0.5 should remove approximately 50% of trades."""
        trades = generate_constant_trades(1000, pnl=100.0)
        
        # Run multiple times and average
        trade_counts = []
        for seed in range(100):
            rng = np.random.default_rng(seed)
            result = apply_skip(trades, p_skip=0.5, rng=rng)
            trade_counts.append(len(result))
        
        mean_remaining = np.mean(trade_counts)
        # Should be around 500 (50% of 1000)
        assert 450 < mean_remaining < 550
    
    def test_skip_affects_total_pnl(self):
        """Skipping trades should reduce total PnL proportionally."""
        trades = generate_constant_trades(100, pnl=100.0)
        baseline_total = sum(t.pnl for t in trades)  # 10,000
        
        # Run many trials
        totals = []
        for seed in range(500):
            rng = np.random.default_rng(seed)
            result = apply_skip(trades, p_skip=0.5, rng=rng)
            totals.append(sum(t.pnl for t in result))
        
        mean_total = np.mean(totals)
        # Should be around 5,000 (50% of 10,000)
        assert 4500 < mean_total < 5500


class TestSlippageDistribution:
    """Slippage should reduce PnL by expected amount."""
    
    def test_slippage_reduces_pnl(self):
        """Average slippage should be slip_max / 2 (uniform distribution)."""
        trades = generate_constant_trades(100, pnl=1000.0)
        baseline_total = sum(t.pnl for t in trades)  # 100,000
        
        # Run many trials with slip_max=100
        totals = []
        for seed in range(500):
            rng = np.random.default_rng(seed)
            result = apply_slippage(trades, slip_dollars=100.0, rng=rng)
            totals.append(sum(t.pnl for t in result))
        
        mean_total = np.mean(totals)
        expected_slippage = 100 * 50  # 100 trades * avg $50 slippage
        expected_total = baseline_total - expected_slippage  # 95,000
        
        # Allow 5% tolerance
        assert 0.95 * expected_total < mean_total < 1.05 * expected_total


class TestShufflePreservesDistribution:
    """Shuffling should not change the distribution of returns."""
    
    def test_shuffle_preserves_total_pnl(self):
        """Shuffle should not change total PnL."""
        trades = generate_random_trades(100, mean_pnl=100.0, std_pnl=200.0)
        baseline_total = sum(t.pnl for t in trades)
        rng = np.random.default_rng(42)
        
        result = apply_shuffle(trades, mode='permute', block_len=10, rng=rng)
        result_total = sum(t.pnl for t in result)
        
        assert np.isclose(result_total, baseline_total, rtol=1e-10)
    
    def test_shuffle_changes_order(self):
        """Shuffle should change trade order."""
        trades = generate_random_trades(100)
        rng = np.random.default_rng(42)
        
        result = apply_shuffle(trades, mode='permute', block_len=10, rng=rng)
        
        # At least some trades should be in different positions
        same_position = sum(
            1 for i, (t1, t2) in enumerate(zip(trades, result))
            if t1.pnl == t2.pnl
        )
        assert same_position < len(trades)  # Not all in same position


class TestRobustScoreV3:
    """Test the gated multiplicative robust score."""
    
    def test_breakeven_scores_zero(self):
        """PF = 1.0 should score exactly 0."""
        from wale_montecarlo.analysis.robust_score import compute_robust_score_v3
        
        score = compute_robust_score_v3(pf_p50=1.0, p_corrected=0.01, maxdd_p95=0.1)
        assert score == 0.0
    
    def test_losing_scores_zero(self):
        """PF < 1.0 should score exactly 0."""
        from wale_montecarlo.analysis.robust_score import compute_robust_score_v3
        
        score = compute_robust_score_v3(pf_p50=0.8, p_corrected=0.01, maxdd_p95=0.1)
        assert score == 0.0
    
    def test_high_dd_scores_zero(self):
        """Drawdown at penalty_end should score 0."""
        from wale_montecarlo.analysis.robust_score import compute_robust_score_v3
        
        score = compute_robust_score_v3(pf_p50=3.0, p_corrected=0.01, maxdd_p95=0.60)
        assert score == 0.0
    
    def test_good_strategy_scores_high(self):
        """Good PF, low p-value, low DD should score high."""
        from wale_montecarlo.analysis.robust_score import compute_robust_score_v3
        
        score = compute_robust_score_v3(pf_p50=3.0, p_corrected=0.01, maxdd_p95=0.15)
        # (3.0 - 1.0) * (1 - 0.01) * 1.0 = 1.98
        assert 1.9 < score < 2.1


class TestOverfitClassification:
    """Test overfitting classification."""
    
    def test_robust_classification(self):
        """High stressed PF should classify as Robust."""
        from wale_montecarlo.analysis.robust_score import classify_overfit
        
        result = classify_overfit(baseline_pf=3.0, stressed_pf_p50=2.0)
        assert result == 'Robust'
    
    def test_fragile_classification(self):
        """Moderate stressed PF should classify as Fragile."""
        from wale_montecarlo.analysis.robust_score import classify_overfit
        
        result = classify_overfit(baseline_pf=3.0, stressed_pf_p50=1.3)
        assert result == 'Fragile'
    
    def test_overfit_classification(self):
        """Low stressed PF should classify as Overfit."""
        from wale_montecarlo.analysis.robust_score import classify_overfit
        
        result = classify_overfit(baseline_pf=3.0, stressed_pf_p50=0.9)
        assert result == 'Overfit'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
