"""
Slippage perturbation.

Simulates additional execution costs due to:
- Bid-ask spread crossing
- Market impact from order size
- Adverse fills (buying into strength, selling into weakness)
- Price movement during order routing

Three modes available:
- Dollar-based: Fixed dollar amount per trade
- R-based: Fraction of initial risk (R)
- Percentage-based: Percentage of trade value
"""

from typing import List, Optional
import numpy as np

from ..models import Trade


def apply_slippage(
    trades: List[Trade],
    slip_dollars: float,
    rng: np.random.Generator,
    mode: str = "dollar"
) -> List[Trade]:
    """
    Apply random slippage cost to each trade.

    Slippage is drawn uniformly from [0, slip_dollars] and subtracted
    from each trade's PnL. Slippage is always adverse (reduces profits,
    increases losses).

    Args:
        trades: List of Trade objects
        slip_dollars: Maximum slippage in dollars per trade
        rng: Seeded numpy random generator
        mode: Slippage calculation mode ("dollar", "r_based", "percent")
              For "dollar" mode, slip_dollars is the max dollar amount.
              For other modes, slip_dollars is max percent/R-multiple.

    Returns:
        New list of trades with slippage applied to PnL
    """
    if slip_dollars <= 0.0 or len(trades) == 0:
        return [t.copy() for t in trades]

    result = []
    n = len(trades)

    # Generate all slippage values at once for efficiency
    slippage_values = rng.uniform(0, slip_dollars, n)

    for trade, slip in zip(trades, slippage_values):
        new_trade = trade.copy()

        if mode == "dollar":
            # Direct dollar slippage
            new_trade.pnl -= slip
        elif mode == "r_based":
            # Slippage as fraction of R (estimated from entry price)
            r_estimate = abs(trade.entry_price * 0.02)  # 2% as proxy for R
            new_trade.pnl -= slip * r_estimate
        elif mode == "percent":
            # Slippage as percentage of trade value
            trade_value = abs(trade.entry_price * trade.qty)
            new_trade.pnl -= (slip / 100.0) * trade_value
        else:
            new_trade.pnl -= slip  # Default to dollar

        result.append(new_trade)

    return result


def apply_slippage_variable(
    trades: List[Trade],
    base_slip: float,
    volatility_multipliers: Optional[List[float]],
    rng: np.random.Generator
) -> List[Trade]:
    """
    Apply slippage that varies with market conditions.

    Higher slippage during volatile periods when fills are worse.

    Args:
        trades: List of Trade objects
        base_slip: Base slippage in dollars
        volatility_multipliers: Per-trade multipliers (e.g., from vol regime)
                               If None, uses uniform slippage
        rng: Seeded numpy random generator

    Returns:
        New list of trades with variable slippage applied
    """
    if base_slip <= 0.0 or len(trades) == 0:
        return [t.copy() for t in trades]

    result = []
    n = len(trades)

    if volatility_multipliers is None:
        volatility_multipliers = [1.0] * n

    for trade, vol_mult in zip(trades, volatility_multipliers):
        new_trade = trade.copy()

        # Adjust slippage by volatility multiplier
        max_slip = base_slip * vol_mult
        slip = rng.uniform(0, max_slip)
        new_trade.pnl -= slip

        result.append(new_trade)

    return result


def apply_slippage_asymmetric(
    trades: List[Trade],
    slip_entry: float,
    slip_exit: float,
    rng: np.random.Generator
) -> List[Trade]:
    """
    Apply different slippage for entries vs exits.

    Entry slippage is always adverse (buy higher, sell lower).
    Exit slippage is also always adverse (sell lower, cover higher).

    Args:
        trades: List of Trade objects
        slip_entry: Max slippage on entry in dollars
        slip_exit: Max slippage on exit in dollars
        rng: Seeded numpy random generator

    Returns:
        New list of trades with asymmetric slippage
    """
    if (slip_entry <= 0.0 and slip_exit <= 0.0) or len(trades) == 0:
        return [t.copy() for t in trades]

    result = []

    for trade in trades:
        new_trade = trade.copy()

        # Entry slippage (adverse to direction)
        entry_slip = rng.uniform(0, slip_entry) if slip_entry > 0 else 0

        # Exit slippage (adverse to direction)
        exit_slip = rng.uniform(0, slip_exit) if slip_exit > 0 else 0

        # Total slippage is sum of entry and exit costs
        total_slip = entry_slip + exit_slip
        new_trade.pnl -= total_slip

        result.append(new_trade)

    return result
