# PineScript to Python Backtesting Engine

A production-grade engine that converts TradingView PineScript strategies to Python, enabling local backtesting with exact trade-by-trade validation against TradingView's results.

---

## What This Engine Does

1. **Parses PineScript** - Extracts strategy parameters, indicator logic, and entry/exit rules from your `.pine` files
2. **Rebuilds in Python** - Implements equivalent indicator calculations (EMA, ATR, ADX, etc.) matching PineScript's math exactly
3. **Runs Backtests** - Executes the strategy on OHLC data with proper fill timing, commission, and position sizing
4. **Validates Output** - Compares trade-by-trade results against TradingView's Excel export to ensure accuracy
5. **Feeds Monte Carlo** - Produces trade lists that can be stress-tested with the Monte Carlo engine

---

## Why Build This?

TradingView's backtester is great for visualization but has limitations:

- **No programmatic access** - You cannot run thousands of parameter combinations
- **No Monte Carlo** - TradingView shows one equity curve, not a distribution
- **No custom robustness tests** - Walk-forward, CPCV, permutation tests are impossible
- **Limited export** - Getting raw trade data requires manual Excel downloads

This engine solves all of that while maintaining TradingView-equivalent accuracy.

---

## Architecture Overview

```
[PineScript File] --> [Parser] --> [StrategyParams]
                                         |
[OHLC CSV Data] --> [Indicator Engine] --+--> [Backtest Engine] --> [Trade List]
                                                                          |
[TradingView Excel Export] --> [Validator] <------------------------------+
                                    |
                              [Validation Report]
                                    |
                              [Monte Carlo Engine]
```

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| Parameter Parser | `backtest_engine.py` | Extracts inputs from PineScript |
| Indicator Library | `backtest_engine.py` | EMA, RMA, ATR, ADX implementations |
| Signal Logic | `backtest_engine.py` | Entry/exit condition evaluation |
| Backtest Loop | `backtest_engine.py` | Position management, fill execution |
| Trade Validator | `backtest_engine.py` | Compare vs TradingView exports |
| Monte Carlo | `run_montecarlo.py` | Stress testing suite |

---

## Indicator Implementations

The engine implements PineScript indicators with exact mathematical equivalence:

### Exponential Moving Average (EMA)

```python
def ema(series: np.ndarray, length: int) -> np.ndarray:
    alpha = 2.0 / (length + 1)
    # Initialize with SMA of first 'length' values
    # Then apply recursive: out[i] = alpha * series[i] + (1 - alpha) * out[i-1]
```

**Key detail:** PineScript initializes EMA with an SMA seed. Many Python implementations skip this, causing divergence on early bars.

### Running Moving Average (RMA)

```python
def rma(series: np.ndarray, length: int) -> np.ndarray:
    alpha = 1.0 / length  # Different from EMA!
    # Same recursive structure as EMA but with alpha = 1/length
```

**Key detail:** RMA uses `alpha = 1/length`, not `2/(length+1)`. This is used internally by ATR and ADX.

### Average True Range (ATR)

```python
def atr(high, low, close, length):
    tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
    return rma(tr, length)  # NOT ema!
```

**Key detail:** TradingView's ATR uses RMA, not EMA. Using EMA will cause drift over time.

### Average Directional Index (ADX)

The ADX implementation follows PineScript exactly:
1. Calculate +DM and -DM
2. Smooth with RMA
3. Compute +DI and -DI
4. Calculate DX
5. Smooth DX with RMA to get ADX

---

## Fill Timing Model

The engine matches TradingView's default execution model:

| Event | When Evaluated | When Filled |
|-------|----------------|-------------|
| Entry Signal | Bar N close | Bar N+1 open |
| Exit Signal | Bar N close | Bar N+1 open |
| Stop Loss | Bar N | Bar N (intrabar at stop price) |
| Trailing Stop | Bar N | Bar N (intrabar at trail price) |

**Important:** Signals are computed on bar close but executed on next bar open. This is PineScript's `process_orders_on_close=false` behavior.

---

## Validation Against TradingView

The engine performs trade-by-trade validation:

### What Gets Compared

| Field | Tolerance | Notes |
|-------|-----------|-------|
| Entry Time | Exact | Must match to the bar |
| Exit Time | Exact | Must match to the bar |
| Entry Price | 0.01 | Allows for rounding |
| Exit Price | 0.01 | Allows for rounding |
| Exit Signal | Exact | SL, Trail, PT, OB, etc. |
| PnL | 2% | Accounts for commission differences |
| PnL % | 2% | Same tolerance |

### Open Trade Handling

If validation fails only on the final trade and that trade is still open:

```python
if is_last_trade and excel_exit_signal == "Open" and our_exit_signal == "Open":
    if excel_exit_time > last_csv_time:
        # Ignore this trade - dataset ended before trade closed
        return PASS
```

This handles the common case where TradingView shows an open position but our CSV data ends before the trade closes.

### Sharpe Ratio Matching

TradingView's Sharpe calculation is not publicly documented. The engine tries multiple methods:

1. Daily equity returns (simple)
2. Daily equity returns (log)
3. Bar-by-bar returns
4. Trade-by-trade returns
5. Excess returns vs risk-free

It selects the variant that best matches the TradingView target and reports which was chosen.

---

## Common Accuracy Issues and Fixes

### Issue 1: EMA Drift on Early Bars

**Symptom:** First few trades have slightly different entry/exit prices.

**Cause:** Incorrect EMA initialization.

**Fix:** Initialize EMA with the SMA of the first `length` values, then apply the recursive formula. Do not start from bar 0.

### Issue 2: ATR Using Wrong Smoothing

**Symptom:** Stop loss triggers at wrong prices.

**Cause:** Using EMA instead of RMA for ATR.

**Fix:** ATR must use RMA (alpha = 1/length), not EMA (alpha = 2/(length+1)).

### Issue 3: Fill Price Timing

**Symptom:** Entry/exit prices are one bar off.

**Cause:** Filling on signal bar instead of next bar.

**Fix:** Store pending orders and execute them on the next bar's open price.

### Issue 4: Overnight Gap Handling

**Symptom:** Stop losses trigger at wrong prices during gaps.

**Cause:** Not checking if gap crosses the stop level.

**Fix:** When a bar opens beyond the stop price, fill at the stop price (conservative) or open price (realistic) based on your assumptions.

### Issue 5: Position Sizing Rounding

**Symptom:** Small PnL differences accumulate over time.

**Cause:** Different rounding in quantity calculation.

**Fix:** Use `int(floor(equity / cost))` to match TradingView's behavior.

### Issue 6: Commission Model

**Symptom:** PnL differs by a consistent small percentage.

**Cause:** Commission applied differently (entry only vs entry+exit, percent vs fixed).

**Fix:** Match TradingView's commission setting exactly. Default is percent of trade value on both entry and exit.

---

## Testing Your Implementation

### Step 1: Run Against Known Good Data

```bash
python backtest_engine.py \
    --csv "your_ohlc_data.csv" \
    --pine "your_strategy.pine" \
    --excel "tradingview_export.xlsx" \
    --run_step1 true
```

### Step 2: Check Validation Report

Look at `backtest/out/validation_report.txt`:

```
=== VALIDATION REPORT ===
Trade matching: PASS - All closed trades matched.
Closed trades matched: 77/77

Sharpe Ratio:
  Target (Excel): 0.33
  Best match: daily_log = 0.32 (within 3.0% tolerance)
  Method selected: daily_log
```

### Step 3: Compare Trade Lists

Export both trade lists and diff them:

```bash
# Our trades
cat backtest/out/research_outputs/step1/trade_list.csv

# Compare entry times, exit times, PnL
```

### Step 4: Visual Inspection

Check equity curves match:
- `backtest/out/plots/step1_equity_drawdown.png`
- Compare against TradingView's equity curve visually

---

## Debugging Mismatches

### Trade Count Mismatch

If you have a different number of trades:

1. **Check data alignment** - Does your CSV cover the same date range as TradingView?
2. **Check timezone** - TradingView uses exchange timezone. Your CSV should match.
3. **Check for missing bars** - Weekend/holiday gaps can cause issues.

### Specific Trade Mismatch

If trade N differs:

1. Print the indicator values for that bar
2. Compare entry conditions step by step
3. Check if it is a boundary condition (first bar after warmup, etc.)

### Systematic Drift

If errors accumulate over time:

1. Check indicator initialization (EMA seed)
2. Check for floating point accumulation
3. Verify ATR is using RMA not EMA

---

## Integration with Monte Carlo

Once validation passes, use the trade list for stress testing:

```bash
# Run Monte Carlo with 200k permutations per cell
python CURSOR_run_surface_full_200k.py \
    --repo "path/to/your/data" \
    --n_per_cell 200000 \
    --jobs 8
```

The Monte Carlo engine reads:
- `trade_list.csv` - Your validated trade sequence
- `equity_curve.csv` - For baseline metrics
- `step1_report.txt` - For baseline profit factor

---

## File Structure

```
your_project/
├── backtest_engine.py          # Core engine
├── run_montecarlo.py           # Monte Carlo runner
├── your_strategy.pine          # Your PineScript (private)
├── your_ohlc_data.csv          # OHLC price data
├── tradingview_export.xlsx     # TradingView validation export
└── backtest/
    └── out/
        ├── validation_report.txt
        ├── research_outputs/
        │   └── step1/
        │       ├── trade_list.csv
        │       ├── equity_curve.csv
        │       └── step1_report.txt
        └── plots/
            ├── step1_equity_drawdown.png
            └── step1_monthly_returns.png
```

---

## Performance Considerations

| Data Size | Expected Runtime | Notes |
|-----------|-----------------|-------|
| 1,000 bars | < 1 second | Quick testing |
| 10,000 bars | 2-5 seconds | Typical daily data |
| 50,000 bars | 10-30 seconds | Multi-year intraday |
| 100,000+ bars | 1-2 minutes | Long history |

The bottleneck is usually indicator calculation, not the backtest loop itself.

---

## Summary

This engine bridges the gap between TradingView's user-friendly interface and the rigorous analysis needed for production trading. By maintaining trade-by-trade accuracy, you can trust that the Monte Carlo stress tests and robustness checks reflect your actual strategy.

Key principles:
1. **Match PineScript math exactly** - Especially EMA initialization and ATR smoothing
2. **Match fill timing** - Signal on close, fill on next open
3. **Validate before trusting** - Always compare against TradingView exports
4. **Document discrepancies** - Some differences are unavoidable; know what they are
