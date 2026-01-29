"""Tests for metrics calculation."""

import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wale_montecarlo.metrics import (
    compute_equity_curve,
    compute_total_return_pct,
    compute_max_drawdown_pct,
    compute_profit_factor,
    compute_worst_month_pct,
    compute_sharpe_ratio,
    compute_win_rate,
    compute_all_metrics
)
from wale_montecarlo.models import Trade, TradeSide, EquityPoint, EquityCurve
from datetime import datetime, timedelta


class TestEquityCurve:
    """Tests for equity curve computation."""

    def test_empty_trades(self):
        """Empty trade list should return initial equity."""
        curve = compute_equity_curve([], initial_equity=100000)

        assert curve.initial_equity == 100000
        assert len(curve.points) == 1

    def test_single_trade(self):
        """Single trade should update equity."""
        trade = Trade(
            entry_time=datetime(2024, 1, 1, 10, 0),
            exit_time=datetime(2024, 1, 1, 12, 0),
            entry_price=100,
            exit_price=105,
            pnl=500,
            qty=100,
            side=TradeSide.LONG
        )

        curve = compute_equity_curve([trade], initial_equity=100000)

        assert curve.final_equity == 100500

    def test_multiple_trades(self, sample_trades_small):
        """Multiple trades should accumulate PnL."""
        curve = compute_equity_curve(sample_trades_small, initial_equity=100000)

        expected_final = 100000 + sum(t.pnl for t in sample_trades_small)
        assert abs(curve.final_equity - expected_final) < 0.01


class TestTotalReturn:
    """Tests for total return calculation."""

    def test_positive_return(self):
        """Positive return should be calculated correctly."""
        points = [
            EquityPoint(datetime(2024, 1, 1), 100000),
            EquityPoint(datetime(2024, 1, 2), 150000),
        ]
        curve = EquityCurve(points=points, initial_equity=100000)

        return_pct = compute_total_return_pct(curve)

        assert abs(return_pct - 50.0) < 0.01

    def test_negative_return(self):
        """Negative return should be calculated correctly."""
        points = [
            EquityPoint(datetime(2024, 1, 1), 100000),
            EquityPoint(datetime(2024, 1, 2), 80000),
        ]
        curve = EquityCurve(points=points, initial_equity=100000)

        return_pct = compute_total_return_pct(curve)

        assert abs(return_pct - (-20.0)) < 0.01

    def test_zero_return(self):
        """No change should return zero."""
        points = [
            EquityPoint(datetime(2024, 1, 1), 100000),
            EquityPoint(datetime(2024, 1, 2), 100000),
        ]
        curve = EquityCurve(points=points, initial_equity=100000)

        return_pct = compute_total_return_pct(curve)

        assert abs(return_pct) < 0.01


class TestMaxDrawdown:
    """Tests for max drawdown calculation."""

    def test_no_drawdown(self):
        """Monotonically increasing equity should have zero drawdown."""
        points = [
            EquityPoint(datetime(2024, 1, i), 100000 + i * 1000)
            for i in range(1, 11)
        ]
        curve = EquityCurve(points=points, initial_equity=100000)

        dd = compute_max_drawdown_pct(curve)

        assert dd < 0.01

    def test_simple_drawdown(self):
        """Simple peak-to-trough should be calculated correctly."""
        points = [
            EquityPoint(datetime(2024, 1, 1), 100000),
            EquityPoint(datetime(2024, 1, 2), 120000),  # Peak
            EquityPoint(datetime(2024, 1, 3), 96000),   # Trough (20% DD)
            EquityPoint(datetime(2024, 1, 4), 110000),
        ]
        curve = EquityCurve(points=points, initial_equity=100000)

        dd = compute_max_drawdown_pct(curve)

        assert abs(dd - 20.0) < 0.01

    def test_multiple_drawdowns(self):
        """Should find the maximum drawdown."""
        points = [
            EquityPoint(datetime(2024, 1, 1), 100000),
            EquityPoint(datetime(2024, 1, 2), 110000),
            EquityPoint(datetime(2024, 1, 3), 99000),   # 10% DD from 110k
            EquityPoint(datetime(2024, 1, 4), 130000),
            EquityPoint(datetime(2024, 1, 5), 91000),   # 30% DD from 130k
            EquityPoint(datetime(2024, 1, 6), 120000),
        ]
        curve = EquityCurve(points=points, initial_equity=100000)

        dd = compute_max_drawdown_pct(curve)

        assert abs(dd - 30.0) < 0.01


class TestProfitFactor:
    """Tests for profit factor calculation."""

    def test_all_winners(self):
        """All winning trades should return infinity."""
        trades = [
            Trade(datetime.now(), datetime.now(), 100, 101, 100, 1, TradeSide.LONG),
            Trade(datetime.now(), datetime.now(), 100, 102, 200, 1, TradeSide.LONG),
        ]

        pf = compute_profit_factor(trades)

        assert pf == float('inf')

    def test_all_losers(self):
        """All losing trades should return zero."""
        trades = [
            Trade(datetime.now(), datetime.now(), 100, 99, -100, 1, TradeSide.LONG),
            Trade(datetime.now(), datetime.now(), 100, 98, -200, 1, TradeSide.LONG),
        ]

        pf = compute_profit_factor(trades)

        assert pf == 0.0

    def test_balanced_trades(self):
        """Balanced trades should have PF close to 1."""
        trades = [
            Trade(datetime.now(), datetime.now(), 100, 101, 100, 1, TradeSide.LONG),
            Trade(datetime.now(), datetime.now(), 100, 99, -100, 1, TradeSide.LONG),
        ]

        pf = compute_profit_factor(trades)

        assert abs(pf - 1.0) < 0.01

    def test_profitable_system(self, sample_trades_small):
        """Sample trades should have reasonable PF."""
        pf = compute_profit_factor(sample_trades_small)

        # Should be positive
        assert pf > 0


class TestWinRate:
    """Tests for win rate calculation."""

    def test_all_winners(self):
        """All winners should return 1.0."""
        trades = [
            Trade(datetime.now(), datetime.now(), 100, 101, 100, 1, TradeSide.LONG),
            Trade(datetime.now(), datetime.now(), 100, 102, 200, 1, TradeSide.LONG),
        ]

        wr = compute_win_rate(trades)

        assert wr == 1.0

    def test_all_losers(self):
        """All losers should return 0.0."""
        trades = [
            Trade(datetime.now(), datetime.now(), 100, 99, -100, 1, TradeSide.LONG),
            Trade(datetime.now(), datetime.now(), 100, 98, -200, 1, TradeSide.LONG),
        ]

        wr = compute_win_rate(trades)

        assert wr == 0.0

    def test_half_and_half(self):
        """50/50 should return 0.5."""
        trades = [
            Trade(datetime.now(), datetime.now(), 100, 101, 100, 1, TradeSide.LONG),
            Trade(datetime.now(), datetime.now(), 100, 99, -100, 1, TradeSide.LONG),
        ]

        wr = compute_win_rate(trades)

        assert abs(wr - 0.5) < 0.01


class TestAllMetrics:
    """Tests for compute_all_metrics."""

    def test_returns_permutation_result(self, sample_trades_small):
        """Should return PermutationResult object."""
        result = compute_all_metrics(sample_trades_small, perm_index=42)

        assert result.perm_index == 42
        assert hasattr(result, 'total_return_pct')
        assert hasattr(result, 'max_drawdown_pct')
        assert hasattr(result, 'profit_factor')
        assert hasattr(result, 'win_rate')

    def test_empty_trades(self):
        """Empty trades should return zero metrics."""
        result = compute_all_metrics([], perm_index=0)

        assert result.total_return_pct == 0.0
        assert result.max_drawdown_pct == 0.0
        assert result.profit_factor == 0.0
        assert result.n_trades == 0

    def test_n_trades_correct(self, sample_trades_small):
        """n_trades should match input."""
        result = compute_all_metrics(sample_trades_small, perm_index=0)

        assert result.n_trades == len(sample_trades_small)

    def test_total_pnl_correct(self, sample_trades_small):
        """total_pnl should match sum of trade PnLs."""
        result = compute_all_metrics(sample_trades_small, perm_index=0)

        expected = sum(t.pnl for t in sample_trades_small)
        assert abs(result.total_pnl - expected) < 0.01
