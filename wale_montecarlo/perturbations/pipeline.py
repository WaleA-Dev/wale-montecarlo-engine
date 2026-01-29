"""
Perturbation pipeline - composes all perturbations in sequence.

The order of perturbations matters:
1. Skip (removes trades first)
2. Slippage (modifies PnL)
3. Delay (modifies entry price/PnL)
4. Shuffle (reorders remaining trades)
5. Bootstrap (resamples trades)

State-dependent multipliers can be applied to slippage and delay.
"""

from typing import List, Optional
import numpy as np

from ..models import Trade, CellConfig, OHLCData, EquityCurve
from .skip import apply_skip
from .slippage import apply_slippage, apply_slippage_variable
from .delay import apply_delay
from .shuffle import apply_shuffle
from .bootstrap import apply_bootstrap
from .state_dependent import (
    compute_volatility_multiplier,
    compute_drawdown_multiplier,
    compute_combined_multiplier
)


def apply_all_perturbations(
    trades: List[Trade],
    config: CellConfig,
    rng: np.random.Generator,
    ohlc_data: Optional[OHLCData] = None,
    equity_curve: Optional[EquityCurve] = None,
    use_state_dependent: bool = False
) -> List[Trade]:
    """
    Apply all perturbations in sequence according to config.

    Perturbation order:
    1. Trade skipping (p_skip)
    2. Slippage (slip_dollars)
    3. Execution delay (delay_bars_max)
    4. Sequence shuffling (shuffle_mode)
    5. Bootstrap resampling (bootstrap_mode)

    Args:
        trades: Original list of Trade objects
        config: CellConfig with perturbation parameters
        rng: Seeded numpy random generator
        ohlc_data: Optional OHLC data for realistic delay modeling
        equity_curve: Optional equity curve for state-dependent multipliers
        use_state_dependent: Whether to apply volatility/drawdown multipliers

    Returns:
        New list of perturbed trades
    """
    if len(trades) == 0:
        return []

    # 1. Apply trade skipping
    result = apply_skip(trades, config.p_skip, rng)

    if len(result) == 0:
        return []

    # 2. Apply slippage
    if use_state_dependent and (ohlc_data is not None or equity_curve is not None):
        # State-dependent slippage
        multipliers = compute_combined_multiplier(
            result, ohlc_data, equity_curve
        )
        result = apply_slippage_variable(
            result, config.slip_dollars, multipliers, rng
        )
    else:
        # Standard slippage
        result = apply_slippage(result, config.slip_dollars, rng)

    # 3. Apply execution delay
    result = apply_delay(result, config.delay_bars_max, rng, ohlc_data)

    if len(result) == 0:
        return []

    # 4. Apply sequence shuffling
    result = apply_shuffle(result, config.shuffle_mode, config.block_len, rng)

    # 5. Apply bootstrap resampling
    result = apply_bootstrap(result, config.bootstrap_mode, config.block_len, rng)

    return result


def apply_perturbations_selective(
    trades: List[Trade],
    config: CellConfig,
    rng: np.random.Generator,
    skip_perturbations: Optional[List[str]] = None,
    ohlc_data: Optional[OHLCData] = None
) -> List[Trade]:
    """
    Apply perturbations selectively, skipping specified types.

    Useful for ablation studies to understand which perturbations matter most.

    Args:
        trades: Original list of Trade objects
        config: CellConfig with perturbation parameters
        rng: Seeded numpy random generator
        skip_perturbations: List of perturbation names to skip
                           ('skip', 'slippage', 'delay', 'shuffle', 'bootstrap')
        ohlc_data: Optional OHLC data

    Returns:
        Perturbed trades with specified perturbations skipped
    """
    if skip_perturbations is None:
        skip_perturbations = []

    skip_set = set(p.lower() for p in skip_perturbations)
    result = [t.copy() for t in trades]

    # 1. Skip
    if 'skip' not in skip_set and config.p_skip > 0:
        result = apply_skip(result, config.p_skip, rng)

    if len(result) == 0:
        return []

    # 2. Slippage
    if 'slippage' not in skip_set and config.slip_dollars > 0:
        result = apply_slippage(result, config.slip_dollars, rng)

    # 3. Delay
    if 'delay' not in skip_set and config.delay_bars_max > 0:
        result = apply_delay(result, config.delay_bars_max, rng, ohlc_data)

    if len(result) == 0:
        return []

    # 4. Shuffle
    if 'shuffle' not in skip_set:
        result = apply_shuffle(result, config.shuffle_mode, config.block_len, rng)

    # 5. Bootstrap
    if 'bootstrap' not in skip_set:
        result = apply_bootstrap(result, config.bootstrap_mode, config.block_len, rng)

    return result


def estimate_perturbation_impact(
    trades: List[Trade],
    config: CellConfig,
    n_samples: int = 100,
    seed: int = 42
) -> dict:
    """
    Estimate the impact of each perturbation type separately.

    Runs Monte Carlo simulation isolating each perturbation to
    understand individual contribution to variance.

    Args:
        trades: Original trades
        config: Cell configuration
        n_samples: Number of samples per perturbation type
        seed: Random seed for reproducibility

    Returns:
        Dictionary with impact statistics per perturbation type
    """
    from ..metrics import compute_all_metrics

    base_rng = np.random.default_rng(seed)

    results = {}

    perturbation_types = ['skip', 'slippage', 'delay', 'shuffle', 'bootstrap']

    for ptype in perturbation_types:
        pnl_samples = []

        for i in range(n_samples):
            rng = np.random.default_rng(base_rng.integers(0, 2**32))

            # Apply only this perturbation type
            all_except = [p for p in perturbation_types if p != ptype]
            perturbed = apply_perturbations_selective(
                trades, config, rng, skip_perturbations=all_except
            )

            if len(perturbed) > 0:
                total_pnl = sum(t.pnl for t in perturbed)
                pnl_samples.append(total_pnl)

        if pnl_samples:
            results[ptype] = {
                'mean_pnl': np.mean(pnl_samples),
                'std_pnl': np.std(pnl_samples),
                'min_pnl': np.min(pnl_samples),
                'max_pnl': np.max(pnl_samples),
                'cv': np.std(pnl_samples) / abs(np.mean(pnl_samples)) if np.mean(pnl_samples) != 0 else float('inf')
            }
        else:
            results[ptype] = {'mean_pnl': 0, 'std_pnl': 0, 'cv': 0}

    return results


def create_baseline_config() -> CellConfig:
    """Create a baseline config with no perturbations."""
    from ..models import ShuffleMode, BootstrapMode
    return CellConfig(
        p_skip=0.0,
        slip_dollars=0.0,
        delay_bars_max=0,
        shuffle_mode=ShuffleMode.NONE,
        bootstrap_mode=BootstrapMode.NONE,
        block_len=10
    )


def create_stress_config() -> CellConfig:
    """Create a high-stress config for worst-case testing."""
    from ..models import ShuffleMode, BootstrapMode
    return CellConfig(
        p_skip=0.10,
        slip_dollars=300.0,
        delay_bars_max=3,
        shuffle_mode=ShuffleMode.PERMUTE,
        bootstrap_mode=BootstrapMode.TRADE_BOOTSTRAP,
        block_len=10
    )
