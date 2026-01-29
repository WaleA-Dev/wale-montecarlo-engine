"""
Performance metrics calculation for Monte Carlo simulations.

Computes key trading metrics from perturbed trade lists:
- Total return percentage
- Maximum drawdown percentage
- Profit factor
- Worst month percentage
- Sharpe ratio
- Win rate

These metrics are calculated for each permutation and then
aggregated to form distributions.
"""

from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

from .models import Trade, EquityCurve, EquityPoint, PermutationResult


def compute_equity_curve(
    trades: List[Trade],
    initial_equity: float = 100000.0
) -> EquityCurve:
    """
    Compute equity curve from trade list.

    Equity increases/decreases by trade PnL at each trade's exit time.

    Args:
        trades: List of Trade objects (should be time-sorted)
        initial_equity: Starting equity value

    Returns:
        EquityCurve object with equity at each trade exit
    """
    if len(trades) == 0:
        return EquityCurve(
            points=[EquityPoint(time=datetime.now(), equity=initial_equity)],
            initial_equity=initial_equity
        )

    # Sort by exit time
    sorted_trades = sorted(trades, key=lambda t: t.exit_time)

    points = [EquityPoint(time=sorted_trades[0].entry_time, equity=initial_equity)]
    current_equity = initial_equity

    for trade in sorted_trades:
        current_equity += trade.pnl
        points.append(EquityPoint(time=trade.exit_time, equity=current_equity))

    return EquityCurve(points=points, initial_equity=initial_equity)


def compute_total_return_pct(equity_curve: EquityCurve) -> float:
    """
    Compute total return as percentage.

    Args:
        equity_curve: Equity curve

    Returns:
        Total return as percentage (e.g., 50.0 for 50% gain)
    """
    if equity_curve.initial_equity == 0:
        return 0.0

    final = equity_curve.final_equity
    initial = equity_curve.initial_equity

    return ((final - initial) / initial) * 100.0


def compute_max_drawdown_pct(equity_curve: EquityCurve) -> float:
    """
    Compute maximum drawdown as percentage.

    Drawdown is the peak-to-trough decline in equity.

    Args:
        equity_curve: Equity curve

    Returns:
        Maximum drawdown as positive percentage (e.g., 20.0 for 20% drawdown)
    """
    if len(equity_curve.points) == 0:
        return 0.0

    equities = np.array([p.equity for p in equity_curve.points])

    # Running maximum
    running_max = np.maximum.accumulate(equities)

    # Drawdown at each point
    drawdowns = (running_max - equities) / running_max
    drawdowns = np.nan_to_num(drawdowns, nan=0.0)

    max_dd = np.max(drawdowns) * 100.0

    return max_dd


def compute_profit_factor(trades: List[Trade]) -> float:
    """
    Compute profit factor (gross profit / gross loss).

    Args:
        trades: List of Trade objects

    Returns:
        Profit factor (>1 is profitable, <1 is losing)
    """
    if len(trades) == 0:
        return 0.0

    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def compute_worst_month_pct(
    equity_curve: EquityCurve,
    trades: List[Trade] = None
) -> float:
    """
    Compute worst monthly return as percentage.

    Groups equity changes by calendar month and finds the worst.

    Args:
        equity_curve: Equity curve
        trades: Optional trade list for more accurate monthly grouping

    Returns:
        Worst month return as percentage (negative for losses)
    """
    if len(equity_curve.points) < 2:
        return 0.0

    # Group by year-month
    monthly_returns = defaultdict(float)

    points = equity_curve.points
    for i in range(1, len(points)):
        month_key = (points[i].time.year, points[i].time.month)
        start_equity = points[i - 1].equity
        end_equity = points[i].equity

        if start_equity > 0:
            pct_change = ((end_equity - start_equity) / start_equity) * 100.0
            monthly_returns[month_key] += pct_change

    if not monthly_returns:
        return 0.0

    worst = min(monthly_returns.values())
    return worst


def compute_sharpe_ratio(
    equity_curve: EquityCurve,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Compute annualized Sharpe ratio.

    Args:
        equity_curve: Equity curve
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Annualized Sharpe ratio
    """
    if len(equity_curve.points) < 2:
        return 0.0

    equities = np.array([p.equity for p in equity_curve.points])

    # Calculate returns
    returns = np.diff(equities) / equities[:-1]
    returns = np.nan_to_num(returns, nan=0.0)

    if len(returns) == 0:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns)

    if std_return == 0:
        return 0.0

    # Annualize
    excess_return = mean_return - (risk_free_rate / periods_per_year)
    sharpe = (excess_return / std_return) * np.sqrt(periods_per_year)

    return sharpe


def compute_win_rate(trades: List[Trade]) -> float:
    """
    Compute win rate (fraction of winning trades).

    Args:
        trades: List of Trade objects

    Returns:
        Win rate as decimal (0.0 to 1.0)
    """
    if len(trades) == 0:
        return 0.0

    winners = sum(1 for t in trades if t.pnl > 0)
    return winners / len(trades)


def compute_avg_win_loss_ratio(trades: List[Trade]) -> float:
    """
    Compute average win / average loss ratio.

    Args:
        trades: List of Trade objects

    Returns:
        Ratio of average winning trade to average losing trade
    """
    winners = [t.pnl for t in trades if t.pnl > 0]
    losers = [abs(t.pnl) for t in trades if t.pnl < 0]

    if not losers:
        return float('inf') if winners else 0.0

    avg_win = np.mean(winners) if winners else 0.0
    avg_loss = np.mean(losers)

    if avg_loss == 0:
        return float('inf') if avg_win > 0 else 0.0

    return avg_win / avg_loss


def compute_calmar_ratio(
    equity_curve: EquityCurve,
    periods_per_year: int = 252
) -> float:
    """
    Compute Calmar ratio (annualized return / max drawdown).

    Args:
        equity_curve: Equity curve
        periods_per_year: Periods per year for annualization

    Returns:
        Calmar ratio
    """
    total_return = compute_total_return_pct(equity_curve)
    max_dd = compute_max_drawdown_pct(equity_curve)

    if max_dd == 0:
        return float('inf') if total_return > 0 else 0.0

    # Estimate holding period in years
    if len(equity_curve.points) >= 2:
        days = (equity_curve.points[-1].time - equity_curve.points[0].time).days
        years = max(days / 365.0, 1/365.0)  # At least 1 day
    else:
        years = 1.0

    annualized_return = total_return / years

    return annualized_return / max_dd


def compute_sortino_ratio(
    equity_curve: EquityCurve,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Compute Sortino ratio (return / downside deviation).

    Unlike Sharpe, only penalizes downside volatility.

    Args:
        equity_curve: Equity curve
        risk_free_rate: Annual risk-free rate
        periods_per_year: Periods per year

    Returns:
        Annualized Sortino ratio
    """
    if len(equity_curve.points) < 2:
        return 0.0

    equities = np.array([p.equity for p in equity_curve.points])
    returns = np.diff(equities) / equities[:-1]
    returns = np.nan_to_num(returns, nan=0.0)

    mean_return = np.mean(returns)

    # Downside returns only
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        return float('inf') if mean_return > 0 else 0.0

    downside_std = np.std(downside_returns)

    if downside_std == 0:
        return float('inf') if mean_return > 0 else 0.0

    excess_return = mean_return - (risk_free_rate / periods_per_year)
    sortino = (excess_return / downside_std) * np.sqrt(periods_per_year)

    return sortino


def compute_all_metrics(
    trades: List[Trade],
    perm_index: int,
    initial_equity: float = 100000.0
) -> PermutationResult:
    """
    Compute all metrics for a permutation.

    This is the main function called for each Monte Carlo permutation.

    Args:
        trades: Perturbed trade list
        perm_index: Index of this permutation
        initial_equity: Starting equity

    Returns:
        PermutationResult with all computed metrics
    """
    if len(trades) == 0:
        return PermutationResult(
            perm_index=perm_index,
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
            profit_factor=0.0,
            worst_month_pct=0.0,
            sharpe_ratio=0.0,
            win_rate=0.0,
            n_trades=0,
            total_pnl=0.0
        )

    # Compute equity curve once
    equity_curve = compute_equity_curve(trades, initial_equity)

    # Calculate all metrics
    total_return = compute_total_return_pct(equity_curve)
    max_drawdown = compute_max_drawdown_pct(equity_curve)
    profit_factor = compute_profit_factor(trades)
    worst_month = compute_worst_month_pct(equity_curve, trades)
    sharpe = compute_sharpe_ratio(equity_curve)
    win_rate = compute_win_rate(trades)
    total_pnl = sum(t.pnl for t in trades)

    # Handle infinities
    if profit_factor == float('inf'):
        profit_factor = 999.0  # Cap at large value
    if profit_factor == float('-inf'):
        profit_factor = 0.0

    return PermutationResult(
        perm_index=perm_index,
        total_return_pct=total_return,
        max_drawdown_pct=max_drawdown,
        profit_factor=profit_factor,
        worst_month_pct=worst_month,
        sharpe_ratio=sharpe,
        win_rate=win_rate,
        n_trades=len(trades),
        total_pnl=total_pnl
    )


def compute_baseline_metrics(
    trades: List[Trade],
    initial_equity: float = 100000.0
) -> Dict[str, float]:
    """
    Compute baseline metrics from original (unperturbed) trades.

    Args:
        trades: Original trade list
        initial_equity: Starting equity

    Returns:
        Dictionary of metric name -> value
    """
    result = compute_all_metrics(trades, perm_index=0, initial_equity=initial_equity)

    return {
        'total_return_pct': result.total_return_pct,
        'max_drawdown_pct': result.max_drawdown_pct,
        'profit_factor': result.profit_factor,
        'worst_month_pct': result.worst_month_pct,
        'sharpe_ratio': result.sharpe_ratio,
        'win_rate': result.win_rate,
        'n_trades': result.n_trades,
        'total_pnl': result.total_pnl
    }
