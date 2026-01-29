"""
Unified test suite for the Wale Monte Carlo Backtesting Engine.

This file consolidates all core tests as specified in the prompt.
Run with: python -m pytest tests/test_engine.py -v
"""

import pytest
import tempfile
import shutil
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wale_montecarlo.models import (
    Trade, TradeSide, CellConfig, ShuffleMode, BootstrapMode, RunConfig
)
from wale_montecarlo.io import load_trade_list, save_metrics_compact, load_metrics_compact
from wale_montecarlo.metrics import (
    compute_equity_curve, compute_total_return_pct, compute_max_drawdown_pct,
    compute_profit_factor, compute_all_metrics
)
from wale_montecarlo.perturbations.skip import apply_skip
from wale_montecarlo.perturbations.slippage import apply_slippage
from wale_montecarlo.perturbations.delay import apply_delay
from wale_montecarlo.perturbations.pipeline import apply_all_perturbations
from wale_montecarlo.seeding import compute_cell_seed, compute_perm_seed, get_rng_for_permutation
from wale_montecarlo.worker import run_cell_simple


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_trades():
    """Generate 20 sample trades for testing."""
    trades = []
    start_date = datetime(2024, 1, 15, 9, 30)
    
    # Mix of winners and losers (~55% win rate)
    pnl_values = [
        662.50, -312.50, 625.00, -337.50, 762.50, 375.00, -412.50, 737.50,
        -475.00, 712.50, 412.50, -462.50, 662.50, -387.50, 612.50, 612.50,
        -362.50, 587.50, 687.50, -487.50
    ]
    
    for i, pnl in enumerate(pnl_values):
        entry_time = start_date + timedelta(days=i // 3, hours=(i % 3) * 2)
        exit_time = entry_time + timedelta(hours=1)
        
        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=5250 + i * 5,
            exit_price=5250 + i * 5 + (pnl / 50),  # point_value = 50
            pnl=pnl,
            qty=1,
            side=TradeSide.LONG,
            trade_id=i
        )
        trades.append(trade)
    
    return trades


@pytest.fixture
def tmp_output_dir():
    """Create temporary output directory."""
    tmp_dir = tempfile.mkdtemp(prefix="mc_test_")
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================================
# Test: Trade Loading
# ============================================================================

def test_trade_loading():
    """Load sample trades, verify validation passes."""
    # Get path to sample trade list
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(script_dir, "..", "examples", "sample_trade_list.csv")
    
    if os.path.exists(sample_path):
        trades = load_trade_list(sample_path)
        
        assert len(trades) == 20, f"Expected 20 trades, got {len(trades)}"
        assert all(isinstance(t, Trade) for t in trades)
        assert all(t.pnl is not None for t in trades)
        
        # Check win rate is approximately 55%
        winners = sum(1 for t in trades if t.pnl > 0)
        win_rate = winners / len(trades)
        assert 0.45 <= win_rate <= 0.65, f"Win rate {win_rate} outside expected range"
    else:
        pytest.skip("Sample trade list not found")


# ============================================================================
# Test: Perturbation - Skip
# ============================================================================

def test_perturbation_skip(sample_trades):
    """p_skip=0.5 should drop roughly half the trades."""
    rng = np.random.default_rng(42)
    
    result = apply_skip(sample_trades, p_skip=0.5, rng=rng)
    
    # Allow statistical variance
    assert 5 <= len(result) <= 15, f"Expected ~10 trades, got {len(result)}"
    
    # Verify trades are preserved correctly
    assert all(isinstance(t, Trade) for t in result)


# ============================================================================
# Test: Perturbation - Slippage
# ============================================================================

def test_perturbation_slippage(sample_trades):
    """Slippage should reduce total PnL."""
    rng = np.random.default_rng(42)
    
    original_pnl = sum(t.pnl for t in sample_trades)
    result = apply_slippage(sample_trades, slip_dollars=100.0, rng=rng)
    slipped_pnl = sum(t.pnl for t in result)
    
    # Slippage always hurts
    assert slipped_pnl < original_pnl, "Slippage should reduce total PnL"
    
    # Check each trade was reduced (not increased)
    for orig, slipped in zip(sample_trades, result):
        assert slipped.pnl <= orig.pnl, "Slippage should never improve a trade"


# ============================================================================
# Test: Perturbation - Delay Only Hurts
# ============================================================================

def test_perturbation_delay_only_hurts(sample_trades):
    """Verify delay never improves a trade."""
    rng = np.random.default_rng(42)
    
    original_pnl = sum(t.pnl for t in sample_trades)
    result = apply_delay(sample_trades, delay_bars_max=2, rng=rng)
    delayed_pnl = sum(t.pnl for t in result)
    
    # Delay should not improve overall performance
    assert delayed_pnl <= original_pnl, "Delay should never improve total PnL"


# ============================================================================
# Test: Metrics Calculation
# ============================================================================

def test_metrics_calculation():
    """Known inputs should produce known outputs."""
    # Create simple trades with predictable metrics
    trades = [
        Trade(datetime(2024, 1, 1), datetime(2024, 1, 1), 100, 102, 200, 1, TradeSide.LONG, 0),
        Trade(datetime(2024, 1, 2), datetime(2024, 1, 2), 100, 101, 100, 1, TradeSide.LONG, 1),
        Trade(datetime(2024, 1, 3), datetime(2024, 1, 3), 100, 99, -100, 1, TradeSide.LONG, 2),
        Trade(datetime(2024, 1, 4), datetime(2024, 1, 4), 100, 102, 200, 1, TradeSide.LONG, 3),
    ]
    
    curve = compute_equity_curve(trades, initial_equity=10000)
    
    # Check return
    return_pct = compute_total_return_pct(curve)
    expected_return = (400 / 10000) * 100  # 4%
    assert abs(return_pct - expected_return) < 0.1
    
    # Check profit factor: gross_profit / gross_loss = 500 / 100 = 5.0
    pf = compute_profit_factor(trades)
    assert abs(pf - 5.0) < 0.1


# ============================================================================
# Test: Single Simulation
# ============================================================================

def test_single_simulation(sample_trades):
    """Run one sim, verify output structure."""
    result = compute_all_metrics(sample_trades, perm_index=0)
    
    # Verify result has all required fields
    assert hasattr(result, 'perm_index')
    assert hasattr(result, 'total_return_pct')
    assert hasattr(result, 'max_drawdown_pct')
    assert hasattr(result, 'profit_factor')
    assert hasattr(result, 'win_rate')
    assert hasattr(result, 'n_trades')
    assert hasattr(result, 'total_pnl')
    
    # Verify values are reasonable
    assert result.perm_index == 0
    assert result.n_trades == len(sample_trades)
    assert result.total_pnl == sum(t.pnl for t in sample_trades)


# ============================================================================
# Test: Cell Runner
# ============================================================================

def test_cell_runner(sample_trades, tmp_output_dir):
    """Run 100 perms for one cell, verify output files."""
    config = CellConfig(
        p_skip=0.02,
        slip_dollars=50.0,
        delay_bars_max=1,
        shuffle_mode=ShuffleMode.NONE,
        bootstrap_mode=BootstrapMode.NONE,
        block_len=10
    )
    
    n_perms = 100
    results = run_cell_simple(
        cell_config=config,
        trades=sample_trades,
        n_perms=n_perms
    )
    
    assert len(results) == n_perms, f"Expected {n_perms} results, got {len(results)}"
    
    # Verify all have unique perm_index
    indices = [r.perm_index for r in results]
    assert len(set(indices)) == n_perms, "All perm_index values should be unique"


# ============================================================================
# Test: Seeding Determinism
# ============================================================================

def test_seeding_determinism(sample_trades):
    """Same seed should produce identical results."""
    config = CellConfig(
        p_skip=0.05,
        slip_dollars=100.0,
        delay_bars_max=1,
        shuffle_mode=ShuffleMode.PERMUTE,
        bootstrap_mode=BootstrapMode.NONE,
        block_len=10
    )
    
    # Run with same seed twice
    results1 = run_cell_simple(config, sample_trades, n_perms=50)
    results2 = run_cell_simple(config, sample_trades, n_perms=50)
    
    # Should produce identical results
    for r1, r2 in zip(results1, results2):
        assert r1.total_pnl == r2.total_pnl, "Same seed should produce identical PnL"
        assert r1.n_trades == r2.n_trades, "Same seed should produce identical trade count"


# ============================================================================
# Test: Checkpoint Resume
# ============================================================================

def test_checkpoint_resume(tmp_output_dir, sample_trades):
    """Simulate crash, resume, verify no duplicates."""
    from wale_montecarlo.models import PermutationResult
    
    metrics_path = os.path.join(tmp_output_dir, "metrics_compact.csv")
    
    # Simulate partial run - write 50 results
    partial_results = []
    for i in range(50):
        result = PermutationResult(
            perm_index=i,
            total_return_pct=10.0 + np.random.randn(),
            max_drawdown_pct=5.0 + abs(np.random.randn()),
            profit_factor=1.5 + np.random.randn() * 0.2,
            win_rate=0.55,
            worst_month_pct=-2.0,
            sharpe_ratio=1.0,
            n_trades=18,
            total_pnl=5000.0
        )
        partial_results.append(result)
    
    save_metrics_compact(metrics_path, partial_results, append=False)
    
    # Load and verify
    loaded, max_idx = load_metrics_compact(metrics_path)
    
    assert len(loaded) == 50, f"Expected 50 results, got {len(loaded)}"
    assert max_idx == 49, f"Expected max_idx=49, got {max_idx}"
    
    # Simulate resume - write more results starting at 50
    resume_results = []
    for i in range(50, 100):
        result = PermutationResult(
            perm_index=i,
            total_return_pct=10.0 + np.random.randn(),
            max_drawdown_pct=5.0 + abs(np.random.randn()),
            profit_factor=1.5 + np.random.randn() * 0.2,
            win_rate=0.55,
            worst_month_pct=-2.0,
            sharpe_ratio=1.0,
            n_trades=18,
            total_pnl=5000.0
        )
        resume_results.append(result)
    
    save_metrics_compact(metrics_path, resume_results, append=True)
    
    # Load final and verify no duplicates
    final_results, final_max = load_metrics_compact(metrics_path)
    
    assert len(final_results) == 100, f"Expected 100 results, got {len(final_results)}"
    
    # Verify all indices are unique
    indices = [r.perm_index for r in final_results]
    assert len(set(indices)) == 100, "All perm_index values should be unique"


# ============================================================================
# Test: End-to-End
# ============================================================================

def test_end_to_end(sample_trades, tmp_output_dir):
    """Mini grid (2×2), 50 perms each, verify completion."""
    # Create a minimal grid
    configs = [
        CellConfig(p_skip=0.0, slip_dollars=0.0, delay_bars_max=0,
                   shuffle_mode=ShuffleMode.NONE, bootstrap_mode=BootstrapMode.NONE, block_len=10),
        CellConfig(p_skip=0.05, slip_dollars=0.0, delay_bars_max=0,
                   shuffle_mode=ShuffleMode.NONE, bootstrap_mode=BootstrapMode.NONE, block_len=10),
        CellConfig(p_skip=0.0, slip_dollars=50.0, delay_bars_max=0,
                   shuffle_mode=ShuffleMode.NONE, bootstrap_mode=BootstrapMode.NONE, block_len=10),
        CellConfig(p_skip=0.05, slip_dollars=50.0, delay_bars_max=0,
                   shuffle_mode=ShuffleMode.NONE, bootstrap_mode=BootstrapMode.NONE, block_len=10),
    ]
    
    n_perms = 50
    all_results = []
    
    for config in configs:
        results = run_cell_simple(config, sample_trades, n_perms=n_perms)
        all_results.extend(results)
    
    # Verify total results
    assert len(all_results) == len(configs) * n_perms
    
    # Verify baseline (no perturbation) has highest return
    baseline_results = all_results[:n_perms]
    avg_baseline_return = np.mean([r.total_return_pct for r in baseline_results])
    
    # Stressed results should have lower average return
    stressed_results = all_results[3 * n_perms:]  # Last cell with both perturbations
    avg_stressed_return = np.mean([r.total_return_pct for r in stressed_results])
    
    assert avg_stressed_return < avg_baseline_return, \
        "Stressed scenario should have lower returns than baseline"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
