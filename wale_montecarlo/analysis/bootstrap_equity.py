"""
Bootstrap equity curve analysis.

Industry-standard Monte Carlo simulation that resamples trades
to generate confidence intervals on returns, drawdown, and Sharpe ratio.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass

# Import from existing modules
try:
    from ..models import Trade
    from ..metrics import compute_metrics
except ImportError:
    # Allow standalone testing
    pass


@dataclass
class BootstrapResult:
    """Results from bootstrap analysis."""
    # Returns
    total_return_mean: float
    total_return_ci: Tuple[float, float]  # 95% CI
    cagr_mean: float
    cagr_ci: Tuple[float, float]
    
    # Risk
    max_drawdown_median: float
    max_drawdown_p75: float
    max_drawdown_p95: float
    
    # Risk-adjusted
    sharpe_mean: float
    sharpe_ci: Tuple[float, float]
    
    # Win rate
    win_rate: float
    profit_factor: float
    
    # Raw distributions for plotting
    equity_curves: np.ndarray  # Shape: (n_samples, n_trades+1)
    return_distribution: np.ndarray
    drawdown_distribution: np.ndarray
    
    # Metadata
    n_trades: int
    n_samples: int


def bootstrap_equity_curves(
    trades: List['Trade'],
    n_samples: int = 10000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None
) -> BootstrapResult:
    """
    Bootstrap resample trades to build equity curve distributions.
    
    Args:
        trades: List of Trade objects with pnl attribute
        n_samples: Number of bootstrap resamples (default: 10,000)
        confidence_level: CI level (default: 0.95 for 95% CI)
        seed: Random seed for reproducibility
    
    Returns:
        BootstrapResult with confidence intervals and distributions
    """
    if not trades:
        raise ValueError("No trades provided")
    
    rng = np.random.default_rng(seed)
    n_trades = len(trades)
    
    # Extract PnL values
    pnls = np.array([t.pnl for t in trades])
    
    # Compute basic stats on original data
    wins = pnls > 0
    win_rate = np.mean(wins)
    gross_profit = np.sum(pnls[wins]) if np.any(wins) else 0
    gross_loss = abs(np.sum(pnls[~wins])) if np.any(~wins) else 1e-10
    profit_factor = min(gross_profit / gross_loss, 999.0)
    
    # Bootstrap resample
    equity_curves = np.zeros((n_samples, n_trades + 1))
    total_returns = np.zeros(n_samples)
    max_drawdowns = np.zeros(n_samples)
    sharpe_ratios = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Resample trades with replacement
        indices = rng.choice(n_trades, size=n_trades, replace=True)
        resampled_pnls = pnls[indices]
        
        # Build equity curve (starting at 0)
        equity = np.concatenate([[0], np.cumsum(resampled_pnls)])
        equity_curves[i] = equity
        
        # Total return
        total_returns[i] = equity[-1]
        
        # Max drawdown (as fraction of peak, or absolute if no gains)
        running_max = np.maximum.accumulate(equity)
        drawdowns = running_max - equity
        max_drawdowns[i] = np.max(drawdowns)
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        if np.std(resampled_pnls) > 0:
            daily_sharpe = np.mean(resampled_pnls) / np.std(resampled_pnls)
            sharpe_ratios[i] = daily_sharpe * np.sqrt(252)
        else:
            sharpe_ratios[i] = 0.0
    
    # Compute confidence intervals
    alpha = 1 - confidence_level
    ci_low = alpha / 2 * 100
    ci_high = (1 - alpha / 2) * 100
    
    return_ci = (np.percentile(total_returns, ci_low), 
                 np.percentile(total_returns, ci_high))
    sharpe_ci = (np.percentile(sharpe_ratios, ci_low),
                 np.percentile(sharpe_ratios, ci_high))
    
    # CAGR calculation (simplified - assumes trades span 1 year)
    # For proper CAGR, need trade dates
    mean_return = np.mean(total_returns)
    cagr_mean = mean_return  # Simplified
    cagr_ci = return_ci  # Simplified
    
    return BootstrapResult(
        total_return_mean=mean_return,
        total_return_ci=return_ci,
        cagr_mean=cagr_mean,
        cagr_ci=cagr_ci,
        max_drawdown_median=np.percentile(max_drawdowns, 50),
        max_drawdown_p75=np.percentile(max_drawdowns, 75),
        max_drawdown_p95=np.percentile(max_drawdowns, 95),
        sharpe_mean=np.mean(sharpe_ratios),
        sharpe_ci=sharpe_ci,
        win_rate=win_rate,
        profit_factor=profit_factor,
        equity_curves=equity_curves,
        return_distribution=total_returns,
        drawdown_distribution=max_drawdowns,
        n_trades=n_trades,
        n_samples=n_samples
    )


def compute_percentile_curves(
    equity_curves: np.ndarray,
    percentiles: List[int] = [5, 25, 50, 75, 95]
) -> Dict[int, np.ndarray]:
    """
    Compute percentile equity curves for plotting confidence bands.
    
    Args:
        equity_curves: Shape (n_samples, n_points)
        percentiles: Which percentiles to compute
    
    Returns:
        Dict mapping percentile -> equity curve array
    """
    result = {}
    for p in percentiles:
        result[p] = np.percentile(equity_curves, p, axis=0)
    return result


def format_bootstrap_summary(result: BootstrapResult) -> str:
    """Format bootstrap results as markdown summary."""
    lines = [
        "## Bootstrap Analysis Summary",
        "",
        f"**Trades analyzed:** {result.n_trades}",
        f"**Bootstrap samples:** {result.n_samples:,}",
        "",
        "### Returns",
        f"- **Expected Return:** ${result.total_return_mean:,.2f}",
        f"- **95% CI:** ${result.total_return_ci[0]:,.2f} to ${result.total_return_ci[1]:,.2f}",
        "",
        "### Risk",
        f"- **Max Drawdown (P50):** ${result.max_drawdown_median:,.2f}",
        f"- **Max Drawdown (P95):** ${result.max_drawdown_p95:,.2f}",
        "",
        "### Risk-Adjusted",
        f"- **Sharpe Ratio:** {result.sharpe_mean:.2f} ({result.sharpe_ci[0]:.2f} to {result.sharpe_ci[1]:.2f})",
        "",
        "### Trade Stats",
        f"- **Win Rate:** {result.win_rate:.1%}",
        f"- **Profit Factor:** {result.profit_factor:.2f}",
    ]
    return "\n".join(lines)
