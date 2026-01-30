"""
Predefined stress scenarios for strategy testing.

Instead of grid searching 6,048 parameter combinations,
use 3 industry-standard scenarios: optimistic, realistic, pessimistic.
"""

from typing import List, Dict, NamedTuple
from dataclasses import dataclass


@dataclass
class StressScenario:
    """A predefined stress test scenario."""
    name: str
    description: str
    
    # Perturbation parameters
    skip_rate: float      # Probability of missing a trade
    slippage_dollars: float  # Per-trade slippage cost
    delay_bars: int       # Execution delay in bars
    
    # Additional costs
    commission_per_trade: float = 0.0


# Standard scenarios
SCENARIOS = {
    'optimistic': StressScenario(
        name='Optimistic',
        description='Best-case execution with minimal friction',
        skip_rate=0.00,
        slippage_dollars=25.0,
        delay_bars=0,
        commission_per_trade=2.0
    ),
    
    'realistic': StressScenario(
        name='Realistic',
        description='Typical real-world execution conditions',
        skip_rate=0.02,  # 2% of trades missed
        slippage_dollars=50.0,
        delay_bars=1,
        commission_per_trade=5.0
    ),
    
    'pessimistic': StressScenario(
        name='Pessimistic',
        description='Adverse conditions stress test',
        skip_rate=0.05,  # 5% of trades missed
        slippage_dollars=100.0,
        delay_bars=2,
        commission_per_trade=10.0
    ),
    
    'extreme': StressScenario(
        name='Extreme',
        description='Worst-case scenario',
        skip_rate=0.10,  # 10% of trades missed
        slippage_dollars=200.0,
        delay_bars=3,
        commission_per_trade=15.0
    )
}


@dataclass
class ScenarioResult:
    """Results from running a stress scenario."""
    scenario: StressScenario
    
    # Returns
    total_return: float
    total_return_after_costs: float
    
    # Risk
    max_drawdown: float
    
    # Metrics
    profit_factor: float
    win_rate: float
    sharpe_ratio: float
    
    # Trade stats
    trades_executed: int
    trades_skipped: int


def apply_scenario(
    trades: List,
    scenario: StressScenario,
    seed: int = 42
) -> ScenarioResult:
    """
    Apply a stress scenario to a trade list.
    
    Args:
        trades: List of Trade objects
        scenario: StressScenario to apply
        seed: Random seed
    
    Returns:
        ScenarioResult with adjusted metrics
    """
    import numpy as np
    
    rng = np.random.default_rng(seed)
    n_trades = len(trades)
    
    # Determine which trades are skipped
    skip_mask = rng.random(n_trades) < scenario.skip_rate
    trades_skipped = np.sum(skip_mask)
    trades_executed = n_trades - trades_skipped
    
    # Adjust PnL for executed trades
    adjusted_pnls = []
    for i, trade in enumerate(trades):
        if skip_mask[i]:
            continue  # Trade skipped
        
        # Apply slippage (always negative)
        slippage = rng.uniform(0, scenario.slippage_dollars)
        
        # Apply commission
        total_cost = slippage + scenario.commission_per_trade
        
        adjusted_pnl = trade.pnl - total_cost
        adjusted_pnls.append(adjusted_pnl)
    
    if not adjusted_pnls:
        return ScenarioResult(
            scenario=scenario,
            total_return=0.0,
            total_return_after_costs=0.0,
            max_drawdown=0.0,
            profit_factor=0.0,
            win_rate=0.0,
            sharpe_ratio=0.0,
            trades_executed=0,
            trades_skipped=n_trades
        )
    
    pnls = np.array(adjusted_pnls)
    
    # Calculate metrics
    total_return = sum(t.pnl for t in trades)
    total_return_after_costs = np.sum(pnls)
    
    # Equity curve and drawdown
    equity = np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity)
    drawdowns = running_max - equity
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    # Win rate
    wins = pnls > 0
    win_rate = np.mean(wins)
    
    # Profit factor
    gross_profit = np.sum(pnls[wins]) if np.any(wins) else 0
    gross_loss = abs(np.sum(pnls[~wins])) if np.any(~wins) else 1e-10
    profit_factor = min(gross_profit / gross_loss, 999.0)
    
    # Sharpe ratio (annualized)
    if np.std(pnls) > 0:
        sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    return ScenarioResult(
        scenario=scenario,
        total_return=total_return,
        total_return_after_costs=total_return_after_costs,
        max_drawdown=max_drawdown,
        profit_factor=profit_factor,
        win_rate=win_rate,
        sharpe_ratio=sharpe,
        trades_executed=trades_executed,
        trades_skipped=trades_skipped
    )


def run_all_scenarios(trades: List) -> Dict[str, ScenarioResult]:
    """Run all standard scenarios and return results."""
    results = {}
    for name, scenario in SCENARIOS.items():
        results[name] = apply_scenario(trades, scenario)
    return results


def format_scenario_comparison(results: Dict[str, ScenarioResult]) -> str:
    """Format scenario comparison as markdown table."""
    lines = [
        "## Stress Scenario Comparison",
        "",
        "| Scenario | Return | After Costs | Max DD | PF | Sharpe |",
        "|----------|--------|-------------|--------|-----|--------|",
    ]
    
    for name in ['optimistic', 'realistic', 'pessimistic', 'extreme']:
        if name in results:
            r = results[name]
            lines.append(
                f"| {r.scenario.name} | ${r.total_return:,.0f} | "
                f"${r.total_return_after_costs:,.0f} | ${r.max_drawdown:,.0f} | "
                f"{r.profit_factor:.2f} | {r.sharpe_ratio:.2f} |"
            )
    
    return "\n".join(lines)
