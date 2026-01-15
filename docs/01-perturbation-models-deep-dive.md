# Perturbation Models: A Deep Dive

This document explains the mathematical and practical details of each perturbation model in the Monte Carlo engine.

---

## Table of Contents

1. [Philosophy of Perturbation](#philosophy-of-perturbation)
2. [Trade Skipping Model](#trade-skipping-model)
3. [Slippage Model](#slippage-model)
4. [Execution Delay Model](#execution-delay-model)
5. [Sequence Shuffling](#sequence-shuffling)
6. [Bootstrap Resampling](#bootstrap-resampling)
7. [State-Dependent Models](#state-dependent-models)
8. [Combining Perturbations](#combining-perturbations)

---

## Philosophy of Perturbation

### Why Perturb at All?

Every backtest makes implicit assumptions:
- "I will execute every signal"
- "My fills will be at the exact historical price"
- "There will be no delays"
- "The order of trades doesn't matter"

These assumptions are always wrong in live trading. Perturbation stress-tests your strategy by systematically violating each assumption.

### Conservative Bias

Our perturbations are **conservative** — they can only hurt your results, never help them.

Example: In the delay model, if delaying your entry would have given you a *better* fill price, we ignore the delay and keep the original (worse) price. This is intentional. In live trading, you might occasionally get lucky with delays, but you shouldn't rely on it.

### Independence

Each perturbation is applied independently to each trade. This might seem harsh (unlikely that *every* trade has slippage), but it's mathematically cleaner and more conservative.

---

## Trade Skipping Model

### Mathematical Formulation

For each trade $i$ in the trade list:
$$
\text{executed}_i = \begin{cases}
1 & \text{if } U_i > p_{\text{skip}} \\
0 & \text{if } U_i \leq p_{\text{skip}}
\end{cases}
$$

Where $U_i \sim \text{Uniform}(0, 1)$ is an independent random draw.

### Implementation Details

```python
mask = rng.random(n_trades) >= p_skip
returns[~mask] = 0.0
pnl[~mask] = 0.0
```

When a trade is skipped:
- Its return contribution is 0
- Its PnL contribution is 0
- The equity curve continues from the previous value

### Minimum Trades Constraint

To prevent degenerate simulations, we enforce a minimum number of executed trades:

```python
attempts = 0
min_trades = 30  # configurable
while mask.sum() < min_trades and attempts < 50:
    mask = rng.random(n_trades) >= p_skip
    attempts += 1
```

If after 50 attempts we can't get enough trades, the simulation proceeds anyway but is flagged.

### Practical Interpretation

| p_skip | Expected Trades Skipped (of 100) | Real-World Analog |
|--------|----------------------------------|-------------------|
| 0.01   | 1                                | Occasional connection drop |
| 0.02   | 2                                | Monthly infrastructure issue |
| 0.05   | 5                                | Frequent problems |
| 0.10   | 10                               | Severe reliability issues |

---

## Slippage Model

### Mathematical Formulation

We support three slippage modes:

#### Dollar Slippage (Recommended)
$$
\text{slip}_i = U_i \cdot \text{slip\_max} \cdot m_i
$$

Where:
- $U_i \sim \text{Uniform}(0, 1)$
- $\text{slip\_max}$ is the maximum slippage in dollars
- $m_i$ is a state-dependent multiplier (see State-Dependent Models)

#### R-Based Slippage
$$
\text{slip}_i = U_i \cdot r_{\text{max}} \cdot R_i \cdot m_i
$$

Where $R_i$ is the risk (R-value) for trade $i$ in dollars.

#### Percentage Slippage
$$
\text{slip}_i = U_i \cdot \text{pct\_max} \cdot V_i \cdot m_i
$$

Where $V_i$ is the entry value (notional) for trade $i$.

### Application

Slippage is subtracted from PnL:
$$
\text{PnL}_{\text{after}} = \text{PnL}_{\text{before}} - \text{slip}_i
$$

### Why Uniform Distribution?

We use uniform rather than normal/log-normal because:
1. Simpler to parameterize (just one number: max)
2. Conservative (puts weight on larger slippages)
3. No negative slippage (you never get *better* than expected)

### Practical Calibration

For index futures (NQ, ES):
- **Tight market**: $25-50
- **Normal conditions**: $50-100
- **Volatile periods**: $100-200
- **Extreme conditions**: $200-300

For individual stocks, multiply by typical bid-ask spread / 2.

---

## Execution Delay Model

This is the most sophisticated perturbation model. It simulates what happens when your orders don't fill at the intended price because of latency.

### Two Modes

#### 1. OHLC-Based Delay (Preferred)

When OHLC data is available, we use actual bar prices:

```python
# Original entry at bar idx_entry
# Delay by k bars
new_entry_bar = idx_entry + k
entry_price = ohlc_open[new_entry_bar]
```

We use the **open price** of the delayed bar because:
- It's the most realistic "you get filled at market open"
- It avoids lookahead bias (we don't know the close/high/low in advance)

#### 2. Approximate Delay (Fallback)

When no OHLC data is available, we sample from the strategy's historical bar returns:

```python
# Sample k bar returns from the historical distribution
sampled_returns = rng.choice(bar_returns, size=k, replace=True)
combined_return = np.prod(1 + sampled_returns) - 1
```

This approximates "what would have happened if we entered k bars late."

### Conservative Constraint

**Key rule**: Delay can only hurt, never help.

```python
if new_pnl > original_pnl:
    new_pnl = original_pnl  # Keep original (worse) outcome
```

This prevents the simulation from accidentally rewarding delays.

### Adverse Cap

To prevent unrealistic catastrophic delay impacts:

```python
# Cap the adverse impact at delay_adverse_cap_r × R
if new_pnl < original_pnl - cap_r * R_dollars:
    new_pnl = original_pnl - cap_r * R_dollars
```

Default: `delay_adverse_cap_r = 0.5` (delay can cost at most 0.5R per trade).

### Both Sides vs One Side

- **`both`** (default): Both entry and exit can be delayed independently
- **`one_side`**: Either entry OR exit is delayed, not both

Mathematical difference:
- `both`: Expected delay = 2 × delay_max / 2 = delay_max bars total
- `one_side`: Expected delay = delay_max / 2 bars total

---

## Sequence Shuffling

### Why Shuffle?

The order of trades matters for:
- **Drawdown timing**: A streak of losses hits harder psychologically and financially
- **Compounding**: Early wins compound more than late wins
- **Path dependence**: The same total return can come from very different equity paths

### Modes

#### Full Permutation (`permute`)

```python
idx = rng.permutation(n_trades)
returns = returns[idx]
pnl = pnl[idx]
```

This completely randomizes trade order while preserving the set of trades.

#### Block Permutation (`block_permute`)

```python
def block_permutation(n, block_len):
    blocks = [np.arange(i, min(i + block_len, n)) 
              for i in range(0, n, block_len)]
    rng.shuffle(blocks)
    return np.concatenate(blocks)
```

This shuffles blocks of trades, preserving local structure (autocorrelation within blocks) while randomizing global order.

### Interpretation

If your strategy's metrics are:
- **Stable under permute**: Not dependent on lucky ordering
- **Unstable under permute**: May be benefiting from specific trade sequences

Block permutation is more realistic — real trading strategies often have clustered behavior.

---

## Bootstrap Resampling

### Why Bootstrap?

Bootstrap answers: "What if we had drawn a different sample from the same underlying process?"

This is useful for:
- Estimating confidence intervals
- Understanding sampling variability
- Detecting strategies that depend on specific trades

### Trade Bootstrap

```python
idx = rng.integers(0, n_trades, size=n_trades)
returns = returns[idx]
pnl = pnl[idx]
```

Sample trades with replacement:
- Some trades appear multiple times
- Some trades never appear
- Total number of trades stays the same

### Block Bootstrap

```python
def block_bootstrap(n, block_len):
    max_start = n - block_len
    out = []
    while len(out) < n:
        start = rng.integers(0, max_start + 1)
        out.extend(range(start, start + block_len))
    return np.array(out[:n])
```

Sample blocks with replacement:
- Preserves autocorrelation structure
- More realistic for time series data

### When to Use

- **Trade bootstrap**: When trades are approximately independent
- **Block bootstrap**: When there's serial correlation (trend-following strategies)

---

## State-Dependent Models

### Motivation

Real slippage and delays aren't uniform — they're worse during volatile periods and drawdowns.

### State Variables

We compute two state variables for each trade:

#### Volatility State (`vol_pct`)
Rolling volatility percentile:
```python
vol = returns.rolling(lookback).std()
vol_pct = vol.rank(pct=True)  # 0 to 1
```

#### Drawdown State (`dd_norm`)
Normalized drawdown severity:
```python
running_max = np.maximum.accumulate(equity)
dd = (equity - running_max) / running_max
dd_norm = |dd| / max(|dd|)  # 0 to 1
```

### State-Dependent Multipliers

For slippage:
```python
multiplier = 1 + intensity  # Where intensity = vol_pct, dd_norm, or average
slip_actual = slip_base * multiplier
```

For delay probability:
```python
if rng.random() > intensity:
    skip_delay = True  # No delay applied
```

### Models

| Model | Intensity | Effect |
|-------|-----------|--------|
| `none` | 1.0 | Uniform (no state dependence) |
| `vol` | vol_pct | More slippage/delay during volatile periods |
| `dd` | dd_norm | More slippage/delay during drawdowns |
| `vol_dd` | 0.5 × vol_pct + 0.5 × dd_norm | Combined |

---

## Combining Perturbations

### Order of Operations

Perturbations are applied in this order:

1. **Skip**: Decide which trades are executed
2. **Delay**: Adjust entry/exit prices for timing
3. **Slippage**: Subtract slippage cost from PnL
4. **Shuffle**: Reorder trades
5. **Bootstrap**: Resample trades

### Why This Order?

- Skip happens first because delay/slippage don't apply to skipped trades
- Delay before slippage because delay affects the base price
- Shuffle before bootstrap because we shuffle the original set, then resample
- Bootstrap last because it changes which trades exist

### Independence

Each perturbation is applied independently:
- A trade can be skipped AND (if not skipped) have slippage AND delay
- The parameters are not correlated

### Combinatorial Explosion

With all perturbations active, the space of possible outcomes is enormous:
- Each trade: skip/execute × delay_bars × slippage_amount
- Plus global: shuffle × bootstrap

This is why we need 200,000 permutations — to adequately sample this space.

---

## Summary

| Model | What It Tests | Key Parameter | Default |
|-------|---------------|---------------|---------|
| Skip | Execution reliability | p_skip | 0-10% |
| Slippage | Market impact | slip_dollars | 0-$300 |
| Delay | Timing risk | delay_bars_max | 0-3 bars |
| Shuffle | Sequence dependence | mode | none/permute/block |
| Bootstrap | Sampling variability | mode | none/trade/block |

The combination of all these creates a comprehensive stress test that exposes hidden fragilities in your strategy.
