"""
Data models for the Monte Carlo backtesting engine.

This module defines all the core data structures used throughout the engine,
including trade records, configuration objects, and result containers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Literal
from enum import Enum


class ShuffleMode(str, Enum):
    """Trade sequence shuffling modes."""
    NONE = "none"
    PERMUTE = "permute"
    BLOCK_PERMUTE = "block_permute"


class BootstrapMode(str, Enum):
    """Bootstrap resampling modes."""
    NONE = "none"
    TRADE_BOOTSTRAP = "trade_bootstrap"
    BLOCK_BOOTSTRAP = "block_bootstrap"


class TradeSide(str, Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"


@dataclass
class Trade:
    """
    Represents a single trade from the backtest.

    Attributes:
        entry_time: When the trade was entered
        exit_time: When the trade was exited
        entry_price: Price at entry
        exit_price: Price at exit
        pnl: Profit/loss in dollars
        qty: Quantity traded
        side: Long or short
        trade_id: Optional unique identifier
    """
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    pnl: float
    qty: float
    side: TradeSide
    trade_id: Optional[int] = None

    def copy(self) -> 'Trade':
        """Create a deep copy of this trade."""
        return Trade(
            entry_time=self.entry_time,
            exit_time=self.exit_time,
            entry_price=self.entry_price,
            exit_price=self.exit_price,
            pnl=self.pnl,
            qty=self.qty,
            side=self.side,
            trade_id=self.trade_id
        )

    @property
    def r_value(self) -> float:
        """Estimate R-value (risk unit) based on entry price movement."""
        if self.entry_price == 0:
            return 0.0
        price_change = abs(self.exit_price - self.entry_price)
        return price_change / self.entry_price


@dataclass
class EquityPoint:
    """Single point on the equity curve."""
    time: datetime
    equity: float


@dataclass
class EquityCurve:
    """
    Time series of portfolio equity.

    Attributes:
        points: List of (time, equity) points
        initial_equity: Starting equity value
    """
    points: List[EquityPoint]
    initial_equity: float = 100000.0

    @property
    def times(self) -> List[datetime]:
        return [p.time for p in self.points]

    @property
    def values(self) -> List[float]:
        return [p.equity for p in self.points]

    @property
    def final_equity(self) -> float:
        return self.points[-1].equity if self.points else self.initial_equity


@dataclass
class OHLCBar:
    """Single OHLC bar."""
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


@dataclass
class OHLCData:
    """
    OHLC price data for realistic delay modeling.

    Attributes:
        bars: List of OHLC bars sorted by time
    """
    bars: List[OHLCBar]

    def get_bar_at(self, time: datetime) -> Optional[OHLCBar]:
        """Find the bar at or just before the given time."""
        for bar in reversed(self.bars):
            if bar.time <= time:
                return bar
        return None

    def get_bar_after(self, time: datetime, n_bars: int = 1) -> Optional[OHLCBar]:
        """Get the bar n_bars after the given time."""
        found_idx = None
        for i, bar in enumerate(self.bars):
            if bar.time >= time:
                found_idx = i
                break

        if found_idx is None:
            return None

        target_idx = found_idx + n_bars
        if target_idx < len(self.bars):
            return self.bars[target_idx]
        return None


@dataclass
class CellConfig:
    """
    Configuration for a single grid cell.

    Each cell represents a unique combination of perturbation parameters.

    Attributes:
        p_skip: Probability of skipping each trade (0.0 to 0.10)
        slip_dollars: Maximum slippage in dollars per trade
        delay_bars_max: Maximum execution delay in bars
        shuffle_mode: How to shuffle trade sequence
        bootstrap_mode: Bootstrap resampling method
        block_len: Block length for block-based operations
    """
    p_skip: float = 0.0
    slip_dollars: float = 0.0
    delay_bars_max: int = 0
    shuffle_mode: ShuffleMode = ShuffleMode.NONE
    bootstrap_mode: BootstrapMode = BootstrapMode.NONE
    block_len: int = 10

    def to_cell_id(self) -> str:
        """Generate unique cell identifier string."""
        return (
            f"skip{self.p_skip:.2f}_"
            f"slip{self.slip_dollars:.0f}_"
            f"delay{self.delay_bars_max}_"
            f"shuf{self.shuffle_mode.value}_"
            f"boot{self.bootstrap_mode.value}_"
            f"blk{self.block_len}"
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "p_skip": self.p_skip,
            "slip_dollars": self.slip_dollars,
            "delay_bars_max": self.delay_bars_max,
            "shuffle_mode": self.shuffle_mode.value,
            "bootstrap_mode": self.bootstrap_mode.value,
            "block_len": self.block_len
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'CellConfig':
        """Create from dictionary."""
        return cls(
            p_skip=d["p_skip"],
            slip_dollars=d["slip_dollars"],
            delay_bars_max=d["delay_bars_max"],
            shuffle_mode=ShuffleMode(d["shuffle_mode"]),
            bootstrap_mode=BootstrapMode(d["bootstrap_mode"]),
            block_len=d["block_len"]
        )


@dataclass
class PermutationResult:
    """
    Results from a single permutation run.

    Attributes:
        perm_index: Unique index of this permutation (0 to n_perms-1)
        total_return_pct: Total return as percentage
        max_drawdown_pct: Maximum drawdown as percentage
        profit_factor: Gross profit / gross loss
        worst_month_pct: Worst monthly return as percentage
        sharpe_ratio: Risk-adjusted return
        win_rate: Fraction of winning trades
        n_trades: Number of trades after perturbations
        total_pnl: Total profit/loss in dollars
    """
    perm_index: int
    total_return_pct: float
    max_drawdown_pct: float
    profit_factor: float
    worst_month_pct: float
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0
    total_pnl: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV output."""
        return {
            "perm_index": self.perm_index,
            "total_return_pct": self.total_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "profit_factor": self.profit_factor,
            "worst_month_pct": self.worst_month_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "n_trades": self.n_trades,
            "total_pnl": self.total_pnl
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'PermutationResult':
        """Create from dictionary."""
        return cls(
            perm_index=int(d["perm_index"]),
            total_return_pct=float(d["total_return_pct"]),
            max_drawdown_pct=float(d["max_drawdown_pct"]),
            profit_factor=float(d["profit_factor"]),
            worst_month_pct=float(d["worst_month_pct"]),
            sharpe_ratio=float(d.get("sharpe_ratio", 0.0)),
            win_rate=float(d.get("win_rate", 0.0)),
            n_trades=int(d.get("n_trades", 0)),
            total_pnl=float(d.get("total_pnl", 0.0))
        )


@dataclass
class QuantileStats:
    """Quantile statistics for a metric."""
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    mean: float
    std: float


@dataclass
class CellSummary:
    """
    Summary statistics for a completed cell.

    Contains quantile distributions and p-values for all metrics.
    """
    cell_id: str
    config: CellConfig
    n_permutations: int

    # Quantile stats for each metric
    total_return: QuantileStats
    max_drawdown: QuantileStats
    profit_factor: QuantileStats
    worst_month: QuantileStats

    # P-value testing
    pvalue_raw: float = 1.0
    pvalue_corrected: float = 1.0

    # Robust score
    robust_score: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON output."""
        return {
            "cell_id": self.cell_id,
            "config": self.config.to_dict(),
            "n_permutations": self.n_permutations,
            "total_return": {
                "p05": self.total_return.p05,
                "p25": self.total_return.p25,
                "p50": self.total_return.p50,
                "p75": self.total_return.p75,
                "p95": self.total_return.p95,
                "mean": self.total_return.mean,
                "std": self.total_return.std
            },
            "max_drawdown": {
                "p05": self.max_drawdown.p05,
                "p25": self.max_drawdown.p25,
                "p50": self.max_drawdown.p50,
                "p75": self.max_drawdown.p75,
                "p95": self.max_drawdown.p95,
                "mean": self.max_drawdown.mean,
                "std": self.max_drawdown.std
            },
            "profit_factor": {
                "p05": self.profit_factor.p05,
                "p25": self.profit_factor.p25,
                "p50": self.profit_factor.p50,
                "p75": self.profit_factor.p75,
                "p95": self.profit_factor.p95,
                "mean": self.profit_factor.mean,
                "std": self.profit_factor.std
            },
            "worst_month": {
                "p05": self.worst_month.p05,
                "p25": self.worst_month.p25,
                "p50": self.worst_month.p50,
                "p75": self.worst_month.p75,
                "p95": self.worst_month.p95,
                "mean": self.worst_month.mean,
                "std": self.worst_month.std
            },
            "pvalue_raw": self.pvalue_raw,
            "pvalue_corrected": self.pvalue_corrected,
            "robust_score": self.robust_score
        }


@dataclass
class BaselineMetrics:
    """
    Baseline metrics from the original backtest.

    Used for p-value calculations to test if perturbations matter.
    """
    total_return_pct: float
    max_drawdown_pct: float
    profit_factor: float
    worst_month_pct: float
    sharpe_ratio: float = 0.0
    n_trades: int = 0


@dataclass
class RunConfig:
    """
    Configuration for a full Monte Carlo run.

    Attributes:
        input_dir: Path to backtest export directory
        output_dir: Path for Monte Carlo results
        n_per_cell: Number of permutations per cell
        n_jobs: Number of parallel workers
        fixed_delay: If set, fix delay parameter to this value
        grid_filters: Optional filters to reduce grid size
    """
    input_dir: str
    output_dir: str
    n_per_cell: int = 200000
    n_jobs: int = 8
    fixed_delay: Optional[int] = None
    grid_filters: Optional[Dict] = None

    # Grid parameter ranges
    p_skip_values: List[float] = field(
        default_factory=lambda: [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    )
    slip_values: List[float] = field(
        default_factory=lambda: [0, 25, 50, 75, 100, 150, 200, 300]
    )
    delay_values: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3]
    )
    shuffle_modes: List[ShuffleMode] = field(
        default_factory=lambda: [ShuffleMode.NONE, ShuffleMode.PERMUTE, ShuffleMode.BLOCK_PERMUTE]
    )
    bootstrap_modes: List[BootstrapMode] = field(
        default_factory=lambda: [BootstrapMode.NONE, BootstrapMode.TRADE_BOOTSTRAP, BootstrapMode.BLOCK_BOOTSTRAP]
    )
    block_len_values: List[int] = field(
        default_factory=lambda: [5, 10, 20]
    )


@dataclass
class CellProgress:
    """Progress tracking for a single cell."""
    cell_id: str
    status: Literal["pending", "running", "completed", "failed"]
    n_completed: int = 0
    n_target: int = 0
    last_update: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class RunProgress:
    """Overall progress for the Monte Carlo run."""
    run_name: str
    start_time: datetime
    cells_total: int
    cells_completed: int = 0
    cells_running: int = 0
    cells_failed: int = 0
    perms_completed: int = 0
    perms_total: int = 0
    last_heartbeat: Optional[datetime] = None
