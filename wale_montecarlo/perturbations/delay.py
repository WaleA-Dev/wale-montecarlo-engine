"""
Execution delay perturbation.

Simulates delayed trade entries due to:
- System latency
- Manual confirmation delays
- Broker execution queues
- Network issues

Two modes:
- OHLC-based: Uses actual open prices at delayed bars (realistic)
- Approximate: Estimates price impact without OHLC data

Conservative approach: delays only hurt, never help.
Adverse impact capped at 0.5R per trade.
"""

from typing import List, Optional
import numpy as np

from ..models import Trade, OHLCData, TradeSide


# Maximum adverse price movement as fraction of entry (proxy for 0.5R)
MAX_ADVERSE_FRACTION = 0.005  # 0.5% as proxy for 0.5R


def apply_delay(
    trades: List[Trade],
    delay_bars_max: int,
    rng: np.random.Generator,
    ohlc_data: Optional[OHLCData] = None
) -> List[Trade]:
    """
    Apply random execution delay to trade entries.

    Each trade's entry is delayed by 0 to delay_bars_max bars.
    The new entry price uses the open of the delayed bar (if OHLC available)
    or an estimated adverse price movement.

    Key behaviors:
    - Delays only hurt performance, never help
    - Adverse price impact is capped at 0.5R equivalent
    - If delayed bar is after exit, trade is skipped

    Args:
        trades: List of Trade objects
        delay_bars_max: Maximum delay in bars (0 = no delay)
        rng: Seeded numpy random generator
        ohlc_data: Optional OHLC data for realistic pricing

    Returns:
        New list of trades with delayed entries
    """
    if delay_bars_max <= 0 or len(trades) == 0:
        return [t.copy() for t in trades]

    result = []
    n = len(trades)

    # Generate delays for all trades at once
    delays = rng.integers(0, delay_bars_max + 1, n)

    for trade, delay in zip(trades, delays):
        if delay == 0:
            result.append(trade.copy())
            continue

        new_trade = trade.copy()

        if ohlc_data is not None:
            # Use actual OHLC data for realistic delay modeling
            delayed_bar = ohlc_data.get_bar_after(trade.entry_time, delay)

            if delayed_bar is None:
                # No data for delayed bar - skip trade
                continue

            # Check if delayed entry is after original exit
            if delayed_bar.time >= trade.exit_time:
                # Would have missed the trade entirely
                continue

            # Use delayed bar's open as new entry price
            delayed_price = delayed_bar.open

        else:
            # Approximate mode: estimate adverse price movement
            delayed_price = _estimate_delayed_price(trade, delay, rng)

        # Calculate price impact
        price_change = delayed_price - trade.entry_price

        # Ensure delay only hurts (conservative)
        if trade.side == TradeSide.LONG:
            # For longs, delayed entry should be at higher price (worse)
            if delayed_price < trade.entry_price:
                delayed_price = trade.entry_price
                price_change = 0
        else:
            # For shorts, delayed entry should be at lower price (worse)
            if delayed_price > trade.entry_price:
                delayed_price = trade.entry_price
                price_change = 0

        # Cap adverse impact at 0.5R equivalent
        max_adverse = trade.entry_price * MAX_ADVERSE_FRACTION
        if abs(price_change) > max_adverse:
            if trade.side == TradeSide.LONG:
                delayed_price = trade.entry_price + max_adverse
            else:
                delayed_price = trade.entry_price - max_adverse
            price_change = delayed_price - trade.entry_price

        # Update trade with delayed entry
        new_trade.entry_price = delayed_price

        # Adjust PnL for the worse entry price
        # For longs: PnL = (exit - entry) * qty
        # Higher entry = lower PnL
        pnl_impact = price_change * trade.qty
        if trade.side == TradeSide.LONG:
            new_trade.pnl = trade.pnl - pnl_impact
        else:
            new_trade.pnl = trade.pnl + pnl_impact

        result.append(new_trade)

    return result


def _estimate_delayed_price(
    trade: Trade,
    delay: int,
    rng: np.random.Generator
) -> float:
    """
    Estimate entry price after delay when OHLC data not available.

    Uses random walk model with adverse bias.
    """
    # Base volatility estimate (2% per bar is aggressive)
    vol_per_bar = 0.002

    # Random walk with slight adverse bias
    total_move = 0.0
    for _ in range(delay):
        # Slightly biased toward adverse direction
        move = rng.normal(0.001, vol_per_bar)  # Small positive bias
        total_move += move

    # Apply to entry price
    if trade.side == TradeSide.LONG:
        # For longs, price tends to move up (adverse)
        return trade.entry_price * (1 + abs(total_move))
    else:
        # For shorts, price tends to move down (adverse)
        return trade.entry_price * (1 - abs(total_move))


def apply_delay_deterministic(
    trades: List[Trade],
    delay_bars: int,
    ohlc_data: Optional[OHLCData] = None
) -> List[Trade]:
    """
    Apply fixed delay to all trades (no randomness).

    Useful for analyzing specific delay scenarios.

    Args:
        trades: List of Trade objects
        delay_bars: Exact delay in bars for all trades
        ohlc_data: Optional OHLC data for realistic pricing

    Returns:
        New list of trades with fixed delay applied
    """
    # Create a dummy RNG that always returns the same value
    dummy_rng = np.random.default_rng(0)

    if delay_bars <= 0:
        return [t.copy() for t in trades]

    result = []

    for trade in trades:
        new_trade = trade.copy()

        if ohlc_data is not None:
            delayed_bar = ohlc_data.get_bar_after(trade.entry_time, delay_bars)

            if delayed_bar is None or delayed_bar.time >= trade.exit_time:
                continue

            delayed_price = delayed_bar.open
        else:
            # For deterministic mode without OHLC, use simple adverse estimate
            adverse_move = trade.entry_price * 0.001 * delay_bars
            if trade.side == TradeSide.LONG:
                delayed_price = trade.entry_price + adverse_move
            else:
                delayed_price = trade.entry_price - adverse_move

        # Calculate and cap impact
        price_change = delayed_price - trade.entry_price
        max_adverse = trade.entry_price * MAX_ADVERSE_FRACTION

        if trade.side == TradeSide.LONG:
            if price_change < 0:
                price_change = 0
            elif price_change > max_adverse:
                price_change = max_adverse
        else:
            if price_change > 0:
                price_change = 0
            elif price_change < -max_adverse:
                price_change = -max_adverse

        new_trade.entry_price = trade.entry_price + price_change

        pnl_impact = price_change * trade.qty
        if trade.side == TradeSide.LONG:
            new_trade.pnl = trade.pnl - pnl_impact
        else:
            new_trade.pnl = trade.pnl + pnl_impact

        result.append(new_trade)

    return result
