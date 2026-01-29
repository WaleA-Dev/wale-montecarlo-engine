"""
State-dependent perturbation multipliers.

Adjusts perturbation intensity based on market conditions:
- Volatility-aware: Higher slippage/delay during volatile periods
- Drawdown-aware: Increased costs during drawdowns

This makes perturbations more realistic by accounting for
adverse conditions often being correlated.
"""

from typing import List, Optional
import numpy as np

from ..models import Trade, EquityCurve, OHLCData


def compute_volatility_multiplier(
    trades: List[Trade],
    ohlc_data: Optional[OHLCData] = None,
    window: int = 20,
    base_vol: Optional[float] = None
) -> List[float]:
    """
    Compute volatility-based multipliers for each trade.

    Higher multiplier during volatile periods (slippage/delay worse).
    Lower multiplier during calm periods.

    Args:
        trades: List of Trade objects
        ohlc_data: OHLC data for volatility calculation
        window: Lookback window for volatility (bars)
        base_vol: Baseline volatility (if None, uses median)

    Returns:
        List of multipliers (one per trade), typically 0.5 to 2.0
    """
    if len(trades) == 0:
        return []

    if ohlc_data is None or len(ohlc_data.bars) < window:
        # No OHLC data - return uniform multipliers
        return [1.0] * len(trades)

    # Calculate rolling volatility for each bar
    closes = np.array([bar.close for bar in ohlc_data.bars])
    returns = np.diff(np.log(closes))

    # Rolling std dev
    vol_series = []
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        window_returns = returns[start:i + 1]
        if len(window_returns) > 0:
            vol = np.std(window_returns)
        else:
            vol = 0.0
        vol_series.append(vol)

    vol_series = np.array(vol_series)

    # Use median as baseline if not provided
    if base_vol is None:
        base_vol = np.median(vol_series[vol_series > 0]) if np.any(vol_series > 0) else 0.01

    if base_vol <= 0:
        base_vol = 0.01

    # Map each trade to a volatility multiplier
    multipliers = []
    bar_times = [bar.time for bar in ohlc_data.bars]

    for trade in trades:
        # Find closest bar to trade entry
        closest_idx = _find_closest_bar_index(trade.entry_time, bar_times)

        if closest_idx is not None and closest_idx < len(vol_series):
            trade_vol = vol_series[closest_idx]
            mult = trade_vol / base_vol
            # Clamp to reasonable range
            mult = np.clip(mult, 0.5, 3.0)
        else:
            mult = 1.0

        multipliers.append(mult)

    return multipliers


def compute_drawdown_multiplier(
    trades: List[Trade],
    equity_curve: Optional[EquityCurve] = None,
    max_mult: float = 2.0
) -> List[float]:
    """
    Compute drawdown-based multipliers for each trade.

    Higher multiplier during drawdowns (costs worse when already losing).
    This simulates:
    - Tighter stops during drawdowns (higher skip rate)
    - Worse fills due to fear/urgency
    - Increased market stress

    Args:
        trades: List of Trade objects
        equity_curve: Equity curve to measure drawdown
        max_mult: Maximum multiplier at deep drawdown

    Returns:
        List of multipliers (one per trade), 1.0 to max_mult
    """
    if len(trades) == 0:
        return []

    if equity_curve is None or len(equity_curve.points) == 0:
        return [1.0] * len(trades)

    # Calculate running max and drawdown for each equity point
    equities = np.array([p.equity for p in equity_curve.points])
    times = [p.time for p in equity_curve.points]

    running_max = np.maximum.accumulate(equities)
    drawdowns = (running_max - equities) / running_max  # As positive fraction

    # Map each trade to drawdown multiplier
    multipliers = []

    for trade in trades:
        # Find closest equity point to trade entry
        closest_idx = _find_closest_time_index(trade.entry_time, times)

        if closest_idx is not None:
            dd = drawdowns[closest_idx]
            # Linear scaling: 0% dd -> 1.0, 20% dd -> max_mult
            # Capped at 20% drawdown for full effect
            dd_capped = min(dd, 0.20)
            mult = 1.0 + (max_mult - 1.0) * (dd_capped / 0.20)
        else:
            mult = 1.0

        multipliers.append(mult)

    return multipliers


def compute_combined_multiplier(
    trades: List[Trade],
    ohlc_data: Optional[OHLCData] = None,
    equity_curve: Optional[EquityCurve] = None,
    vol_weight: float = 0.5,
    dd_weight: float = 0.5
) -> List[float]:
    """
    Compute combined volatility + drawdown multiplier.

    Args:
        trades: List of Trade objects
        ohlc_data: OHLC data for volatility
        equity_curve: Equity curve for drawdown
        vol_weight: Weight for volatility component
        dd_weight: Weight for drawdown component

    Returns:
        Combined multipliers (one per trade)
    """
    vol_mults = compute_volatility_multiplier(trades, ohlc_data)
    dd_mults = compute_drawdown_multiplier(trades, equity_curve)

    # Weighted combination
    combined = []
    for vm, dm in zip(vol_mults, dd_mults):
        combined_mult = vol_weight * vm + dd_weight * dm
        combined.append(combined_mult)

    return combined


def compute_regime_multiplier(
    trades: List[Trade],
    ohlc_data: Optional[OHLCData] = None,
    n_regimes: int = 3
) -> List[float]:
    """
    Compute regime-based multipliers using volatility clustering.

    Identifies low/medium/high volatility regimes and assigns
    multipliers accordingly:
    - Low vol: 0.5x (better fills)
    - Medium vol: 1.0x (normal)
    - High vol: 2.0x (worse fills)

    Args:
        trades: List of Trade objects
        ohlc_data: OHLC data for regime detection
        n_regimes: Number of regimes (default 3)

    Returns:
        Regime-based multipliers
    """
    if len(trades) == 0:
        return []

    if ohlc_data is None or len(ohlc_data.bars) < 20:
        return [1.0] * len(trades)

    # Calculate volatility
    closes = np.array([bar.close for bar in ohlc_data.bars])
    returns = np.diff(np.log(closes))

    # Rolling volatility
    window = 20
    vol_series = []
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        vol_series.append(np.std(returns[start:i + 1]) if i >= start else 0)

    vol_series = np.array(vol_series)

    # Define regime boundaries using quantiles
    q33 = np.percentile(vol_series[vol_series > 0], 33) if np.any(vol_series > 0) else 0.01
    q67 = np.percentile(vol_series[vol_series > 0], 67) if np.any(vol_series > 0) else 0.02

    # Map to multipliers
    regime_mults = {
        'low': 0.5,
        'medium': 1.0,
        'high': 2.0
    }

    bar_times = [bar.time for bar in ohlc_data.bars]
    multipliers = []

    for trade in trades:
        closest_idx = _find_closest_bar_index(trade.entry_time, bar_times)

        if closest_idx is not None and closest_idx < len(vol_series):
            vol = vol_series[closest_idx]
            if vol < q33:
                mult = regime_mults['low']
            elif vol < q67:
                mult = regime_mults['medium']
            else:
                mult = regime_mults['high']
        else:
            mult = 1.0

        multipliers.append(mult)

    return multipliers


def _find_closest_bar_index(target_time, bar_times) -> Optional[int]:
    """Find index of bar closest to target time."""
    if not bar_times:
        return None

    # Binary search for efficiency
    for i, bt in enumerate(bar_times):
        if bt >= target_time:
            return i

    return len(bar_times) - 1


def _find_closest_time_index(target_time, times) -> Optional[int]:
    """Find index of time point closest to target."""
    return _find_closest_bar_index(target_time, times)
