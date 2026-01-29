"""
Wale Monte Carlo Backtesting Engine

A production-grade Monte Carlo simulation framework for stress-testing
trading strategies under realistic market conditions.

This engine systematically perturbs backtest results through:
- Trade skipping (p_skip)
- Slippage costs (slip_dollars)
- Execution delays (delay_bars_max)
- Sequence shuffling (shuffle_mode)
- Bootstrap resampling (bootstrap_mode)

Usage:
    from wale_montecarlo import MonteCarloRunner, RunConfig

    config = RunConfig(
        input_dir="backtest/export",
        output_dir="backtest/montecarlo",
        n_per_cell=200000,
        n_jobs=8
    )

    runner = MonteCarloRunner(config)
    runner.setup()
    runner.run()
"""

__version__ = "1.0.0"
__author__ = "Wale"

# Core models
from .models import (
    Trade,
    TradeSide,
    EquityCurve,
    EquityPoint,
    OHLCData,
    OHLCBar,
    CellConfig,
    PermutationResult,
    CellSummary,
    QuantileStats,
    BaselineMetrics,
    RunConfig,
    ShuffleMode,
    BootstrapMode,
)

# Seeding
from .seeding import (
    compute_cell_seed,
    compute_perm_seed,
    get_rng_for_permutation,
)

# I/O
from .io import (
    load_trade_list,
    load_equity_curve,
    load_ohlc_data,
    load_baseline_report,
    save_metrics_compact,
    load_metrics_compact,
)

# Metrics
from .metrics import (
    compute_equity_curve,
    compute_total_return_pct,
    compute_max_drawdown_pct,
    compute_profit_factor,
    compute_all_metrics,
)

# Grid
from .grid import (
    generate_grid,
    generate_grid_default,
    filter_grid,
    get_grid_summary,
)

# Runner
from .runner import (
    MonteCarloRunner,
    run_surface,
    get_run_status,
)

# Worker
from .worker import (
    run_cell,
    run_cell_simple,
)

__all__ = [
    # Version
    "__version__",
    # Models
    "Trade",
    "TradeSide",
    "EquityCurve",
    "EquityPoint",
    "OHLCData",
    "OHLCBar",
    "CellConfig",
    "PermutationResult",
    "CellSummary",
    "QuantileStats",
    "BaselineMetrics",
    "RunConfig",
    "ShuffleMode",
    "BootstrapMode",
    # Seeding
    "compute_cell_seed",
    "compute_perm_seed",
    "get_rng_for_permutation",
    # I/O
    "load_trade_list",
    "load_equity_curve",
    "load_ohlc_data",
    "load_baseline_report",
    "save_metrics_compact",
    "load_metrics_compact",
    # Metrics
    "compute_equity_curve",
    "compute_total_return_pct",
    "compute_max_drawdown_pct",
    "compute_profit_factor",
    "compute_all_metrics",
    # Grid
    "generate_grid",
    "generate_grid_default",
    "filter_grid",
    "get_grid_summary",
    # Runner
    "MonteCarloRunner",
    "run_surface",
    "get_run_status",
    # Worker
    "run_cell",
    "run_cell_simple",
]
