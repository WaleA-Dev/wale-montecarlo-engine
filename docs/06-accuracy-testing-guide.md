# Accuracy Testing Guide

How to verify your PineScript to Python conversion produces identical results to TradingView.

---

## The Testing Philosophy

**Goal:** Every closed trade should match TradingView exactly. If it does not, something is wrong.

We do not aim for "close enough" - we aim for exact replication. Small differences compound over time and can completely change Monte Carlo results.

---

## Pre-Testing Checklist

Before running any tests, verify:

### 1. Data Alignment

```
[ ] CSV start date matches TradingView chart start
[ ] CSV end date matches TradingView chart end
[ ] Bar count matches (check for missing weekends/holidays)
[ ] Timezone is consistent (UTC vs exchange local)
[ ] OHLC values match TradingView to 2 decimal places
```

### 2. Strategy Settings

```
[ ] All input parameters match between Pine and Python
[ ] Commission rate matches
[ ] Initial capital matches
[ ] Order size percentage matches
[ ] Pyramiding setting matches (usually disabled)
```

### 3. Data Quality

```
[ ] No NaN values in OHLC columns
[ ] High >= Low for all bars
[ ] High >= Open and High >= Close for all bars
[ ] Low <= Open and Low <= Close for all bars
[ ] Timestamps are monotonically increasing
```

---

## The Testing Process

### Phase 1: Indicator Verification

Before testing the full strategy, verify individual indicators:

**Test 1: EMA Verification**
```python
# Compare your EMA output to TradingView's
# Export EMA values from TradingView using plot() and data export
# Compare first 100 values
for i in range(100):
    assert abs(our_ema[i] - tv_ema[i]) < 0.0001, f"EMA mismatch at bar {i}"
```

**Test 2: ATR Verification**
```python
# ATR is critical for stop loss calculations
# Verify it uses RMA not EMA
for i in range(100):
    assert abs(our_atr[i] - tv_atr[i]) < 0.0001, f"ATR mismatch at bar {i}"
```

**Test 3: Custom Oscillator Verification**
```python
# If your strategy uses custom indicators, verify each one
# This is where most bugs hide
```

### Phase 2: Signal Verification

Before testing fills, verify the signals fire on the same bars:

**Test: Entry Signal Comparison**
```python
# Export entry bars from TradingView
# Compare to your signal generation
our_entries = [i for i in range(len(df)) if entry_signal(df, i, params)]
tv_entries = [...]  # From TradingView export

assert our_entries == tv_entries, f"Entry signal mismatch"
```

### Phase 3: Full Trade Comparison

Run the full backtest and compare trade-by-trade:

```bash
python backtest_engine.py \
    --csv data.csv \
    --pine strategy.pine \
    --excel tradingview_export.xlsx \
    --run_step1 true
```

Check `validation_report.txt` for results.

---

## Common Failure Modes

### Failure Mode 1: Off-by-One Bar

**Symptom:** Every trade entry/exit is exactly one bar early or late.

**Causes:**
- Signal evaluated on wrong bar (close vs open)
- Fill executed on wrong bar (same bar vs next bar)
- Index off-by-one in loop

**Debug:**
```python
# Print signal and fill timing
print(f"Signal on bar {signal_bar}, time={df.loc[signal_bar, 'time']}")
print(f"Fill on bar {fill_bar}, time={df.loc[fill_bar, 'time']}")
```

### Failure Mode 2: First Trade Mismatch

**Symptom:** First trade differs, rest are fine.

**Causes:**
- Indicator warmup period insufficient
- EMA not initialized with SMA seed
- Strategy requires N bars of history before first signal

**Debug:**
```python
# Check indicator NaN values
print(f"First non-NaN EMA: bar {np.argmax(~np.isnan(df['ema200']))}")
```

### Failure Mode 3: Stop Loss Trigger Price

**Symptom:** Exit price differs on stop loss trades.

**Causes:**
- Gap handling (open beyond stop vs fill at stop)
- ATR calculation mismatch
- Trailing stop activation logic

**Debug:**
```python
# Print stop calculation for the problematic trade
print(f"Entry: {trade.entry_price}")
print(f"ATR at entry: {atr_at_entry}")
print(f"Stop level: {stop_level}")
print(f"Bar that triggered stop: low={bar_low}, high={bar_high}")
```

### Failure Mode 4: Trailing Stop Activation

**Symptom:** Some trades exit via trailing stop when they should not (or vice versa).

**Causes:**
- Trailing activation threshold differs
- Trail offset calculation differs
- Trail updates on close vs high

**Debug:**
```python
# Track trailing stop state through the trade
for bar in trade_bars:
    print(f"Bar {bar}: high={high}, trail_active={trail_active}, trail_level={trail}")
```

### Failure Mode 5: Cumulative Drift

**Symptom:** Trades match early in the backtest but diverge later.

**Causes:**
- Floating point accumulation in indicators
- Position size depends on equity which diverges
- Small PnL differences compound

**Debug:**
```python
# Compare equity at specific points
checkpoints = [100, 500, 1000, 5000]
for cp in checkpoints:
    print(f"Bar {cp}: Our equity={our_equity[cp]}, TV equity={tv_equity[cp]}")
```

---

## Diagnostic Tools

### Trade Diff Report

Generate a detailed comparison:

```python
def generate_trade_diff(our_trades, tv_trades):
    for i, (ours, theirs) in enumerate(zip(our_trades, tv_trades)):
        diffs = []
        if ours['entry_time'] != theirs['entry_time']:
            diffs.append(f"entry_time: {ours['entry_time']} vs {theirs['entry_time']}")
        if ours['exit_time'] != theirs['exit_time']:
            diffs.append(f"exit_time: {ours['exit_time']} vs {theirs['exit_time']}")
        if abs(ours['entry_price'] - theirs['entry_price']) > 0.01:
            diffs.append(f"entry_price: {ours['entry_price']} vs {theirs['entry_price']}")
        if abs(ours['pnl'] - theirs['pnl']) > 1.0:
            diffs.append(f"pnl: {ours['pnl']:.2f} vs {theirs['pnl']:.2f}")
        
        if diffs:
            print(f"\nTrade {i+1} MISMATCH:")
            for d in diffs:
                print(f"  {d}")
```

### Indicator Snapshot

Dump indicator values at a specific bar:

```python
def indicator_snapshot(df, bar_idx):
    print(f"=== Bar {bar_idx}: {df.loc[bar_idx, 'time']} ===")
    print(f"OHLC: {df.loc[bar_idx, 'open']:.2f} / {df.loc[bar_idx, 'high']:.2f} / "
          f"{df.loc[bar_idx, 'low']:.2f} / {df.loc[bar_idx, 'close']:.2f}")
    print(f"EMA21: {df.loc[bar_idx, 'pivot']:.4f}")
    print(f"ATR14: {df.loc[bar_idx, 'atr14']:.4f}")
    print(f"EMA200: {df.loc[bar_idx, 'ema200']:.4f}")
    print(f"Oscillator: {df.loc[bar_idx, 'osc']:.4f}")
```

### Position State Logger

Track position through time:

```python
def log_position_state(position, bar_idx, df):
    row = df.loc[bar_idx]
    print(f"Bar {bar_idx}: close={row['close']:.2f}")
    if position:
        print(f"  Position: qty={position.qty}, entry={position.entry_price:.2f}")
        print(f"  Unrealized PnL: {(row['close'] - position.entry_price) * position.qty:.2f}")
        if position.trail_stop:
            print(f"  Trail stop: {position.trail_stop:.2f}")
    else:
        print(f"  No position")
```

---

## Acceptance Criteria

### Minimum Acceptance (Research Use)

For Monte Carlo analysis, you need:
- All closed trades match entry/exit times exactly
- Entry/exit prices within 0.01
- PnL within 2% per trade
- Total PnL within 1%

### Strict Acceptance (Production Use)

For live trading signals, you need:
- All of the above PLUS
- Sharpe ratio within 5%
- Max drawdown within 1%
- Monthly returns match within 2%

### When to Accept Differences

Some differences are acceptable if documented:

1. **Open trade at dataset end** - TradingView may show a trade that has not closed yet
2. **Sharpe methodology** - TradingView's exact formula is proprietary
3. **Rounding on very small values** - Floating point differences on sub-penny amounts

---

## Regression Testing

Once you achieve validation, prevent regressions:

### Save Golden Output

```bash
cp backtest/out/research_outputs/step1/trade_list.csv tests/golden_trades.csv
cp backtest/out/validation_report.txt tests/golden_validation.txt
```

### Automated Test

```python
def test_regression():
    # Run backtest
    result = run_backtest(df, params, config)
    trades = export_trades(result)
    
    # Load golden
    golden = pd.read_csv("tests/golden_trades.csv")
    
    # Compare
    assert len(trades) == len(golden), "Trade count changed"
    
    for i, (ours, gold) in enumerate(zip(trades.iterrows(), golden.iterrows())):
        assert ours['entry_time'] == gold['entry_time'], f"Trade {i} entry time changed"
        assert abs(ours['pnl'] - gold['pnl']) < 1.0, f"Trade {i} PnL changed"
```

### Run On Every Change

```bash
# Add to your workflow
python -m pytest tests/test_regression.py
```

---

## When You Cannot Match Exactly

Sometimes perfect matching is impossible:

### Undocumented TradingView Behavior

TradingView occasionally has undocumented quirks:
- Specific fill timing on limit orders
- Intrabar stop loss execution
- Commission rounding

**Solution:** Document the difference, measure the impact, accept if impact is small.

### Data Differences

If your CSV has different bars than TradingView:
- Missing bars (holidays, data gaps)
- Different bar boundaries (session times)
- Timezone issues

**Solution:** Fix the data first. Do not proceed until data matches.

### Strategy Ambiguity

If the PineScript has ambiguous logic:
- Conditions that could evaluate differently
- Order of operations unclear

**Solution:** Test both interpretations, pick the one that matches, document.

---

## Summary

Accuracy testing is the foundation of trust in your backtest. Without it, Monte Carlo results are meaningless - garbage in, garbage out.

The process:
1. Verify data alignment first
2. Test indicators in isolation
3. Test signals in isolation
4. Test full trades
5. Document any accepted differences
6. Set up regression tests

Do not skip these steps. The time invested here saves weeks of debugging later.
