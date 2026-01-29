"""
Pytest fixtures for wale_montecarlo tests.

Provides sample data for testing:
- Trade lists
- Equity curves
- OHLC data
- Cell configurations
"""

import pytest
from datetime import datetime, timedelta
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wale_montecarlo.models import (
    Trade, TradeSide, EquityCurve, EquityPoint,
    OHLCData, OHLCBar, CellConfig, ShuffleMode, BootstrapMode
)


@pytest.fixture
def sample_trades():
    """Generate sample trade list with 100 trades."""
    trades = []
    start_date = datetime(2024, 1, 1, 9, 30, 0)

    np.random.seed(42)

    for i in range(100):
        # Entry and exit times
        entry_time = start_date + timedelta(days=i, hours=np.random.randint(0, 6))
        exit_time = entry_time + timedelta(hours=np.random.randint(1, 8))

        # Prices
        entry_price = 100 + np.random.randn() * 5
        exit_price = entry_price + np.random.randn() * 2

        # Side
        side = TradeSide.LONG if np.random.random() > 0.3 else TradeSide.SHORT

        # PnL
        if side == TradeSide.LONG:
            pnl = (exit_price - entry_price) * 100
        else:
            pnl = (entry_price - exit_price) * 100

        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            qty=100,
            side=side,
            trade_id=i
        )
        trades.append(trade)

    return trades


@pytest.fixture
def sample_trades_small():
    """Generate small trade list with 10 trades for quick tests."""
    trades = []
    start_date = datetime(2024, 1, 1, 9, 30, 0)

    pnl_values = [100, -50, 200, -75, 150, -100, 300, -25, 50, 175]

    for i, pnl in enumerate(pnl_values):
        entry_time = start_date + timedelta(days=i)
        exit_time = entry_time + timedelta(hours=2)

        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=100,
            exit_price=101 if pnl > 0 else 99,
            pnl=pnl,
            qty=100,
            side=TradeSide.LONG,
            trade_id=i
        )
        trades.append(trade)

    return trades


@pytest.fixture
def sample_equity_curve():
    """Generate sample equity curve."""
    points = []
    start_date = datetime(2024, 1, 1)
    equity = 100000.0

    np.random.seed(42)

    for i in range(100):
        time = start_date + timedelta(days=i)
        equity += np.random.randn() * 500
        points.append(EquityPoint(time=time, equity=equity))

    return EquityCurve(points=points, initial_equity=100000.0)


@pytest.fixture
def sample_ohlc_data():
    """Generate sample OHLC data."""
    bars = []
    start_date = datetime(2024, 1, 1, 9, 30, 0)
    price = 100.0

    np.random.seed(42)

    for i in range(500):
        time = start_date + timedelta(hours=i)

        # Random walk
        change = np.random.randn() * 0.5
        open_price = price
        close_price = price + change

        high = max(open_price, close_price) + abs(np.random.randn() * 0.2)
        low = min(open_price, close_price) - abs(np.random.randn() * 0.2)

        bar = OHLCBar(
            time=time,
            open=open_price,
            high=high,
            low=low,
            close=close_price,
            volume=np.random.randint(1000, 10000)
        )
        bars.append(bar)
        price = close_price

    return OHLCData(bars=bars)


@pytest.fixture
def baseline_config():
    """Create baseline config with no perturbations."""
    return CellConfig(
        p_skip=0.0,
        slip_dollars=0.0,
        delay_bars_max=0,
        shuffle_mode=ShuffleMode.NONE,
        bootstrap_mode=BootstrapMode.NONE,
        block_len=10
    )


@pytest.fixture
def moderate_config():
    """Create moderate perturbation config."""
    return CellConfig(
        p_skip=0.05,
        slip_dollars=100.0,
        delay_bars_max=1,
        shuffle_mode=ShuffleMode.PERMUTE,
        bootstrap_mode=BootstrapMode.NONE,
        block_len=10
    )


@pytest.fixture
def stress_config():
    """Create high-stress config."""
    return CellConfig(
        p_skip=0.10,
        slip_dollars=300.0,
        delay_bars_max=3,
        shuffle_mode=ShuffleMode.PERMUTE,
        bootstrap_mode=BootstrapMode.TRADE_BOOTSTRAP,
        block_len=10
    )


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "mc_output"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def sample_trade_list_csv(tmp_path):
    """Create sample trade_list.csv file."""
    import csv

    csv_path = tmp_path / "trade_list.csv"

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl', 'qty', 'side'])

        start_date = datetime(2024, 1, 1, 9, 30, 0)
        for i in range(50):
            entry = start_date + timedelta(days=i)
            exit = entry + timedelta(hours=2)
            pnl = 100 if i % 3 != 0 else -50

            writer.writerow([
                entry.strftime('%Y-%m-%d %H:%M:%S'),
                exit.strftime('%Y-%m-%d %H:%M:%S'),
                100.0,
                101.0 if pnl > 0 else 99.0,
                pnl,
                100,
                'long'
            ])

    return str(csv_path)
