# Wale Monte Carlo - Strategy Stress Lab

> Drop in a trade list. Get a verdict: does your strategy's edge survive reality?

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

If you run a systematic strategy, your backtest is one path through history - and
usually a lucky one. This tool resamples, reorders, and degrades that path tens of
thousands of times to answer the questions that actually matter:

- **Is the edge real,** or an artifact of one lucky trade sequence?
- **What drawdown should I actually expect** - not the one backtest path, but the distribution?
- **Does the edge survive execution friction** (missed trades, slippage) scaled to *your* trade size?
- **How much capital do I need** so a normal losing streak doesn't ruin me?

Everything runs locally. Your trades never leave your machine.

---

## Quick Start

### Option 1 - Desktop app (no Python required)

Download `WaleMonteCarlo.exe` from [Releases](https://github.com/WaleA-Dev/wale-montecarlo-engine/releases)
and double-click it. The app opens in its own window. Drag in a trade list CSV,
set your capital, hit **Run stress test**. You get:

- A **verdict plate**: Robust, Moderate, Fragile, Overfit, or Insufficient Data, with the specific reasons as pass/fail flags
- **Equity cone** - your backtest path against the 5th-95th percentile band of 10,000 resampled paths
- **Final P&L distribution** and probability of ending at a loss
- **Max-drawdown distribution** plus a *luck detector* (was your smooth equity curve just lucky ordering?)
- **Ruin ladder** - probability of hitting 10/20/30/40/50% drawdowns at your capital
- **Execution stress scenarios** with friction scaled to your median trade notional
- One-click **standalone HTML report** you can share or archive

### Option 2 - From source

```bash
pip install -r requirements.txt
python app.py                      # desktop launcher (opens browser)
# or
python -m wale_montecarlo serve    # same UI, explicit port control
```

### Option 3 - CLI report

```bash
python -m wale_montecarlo analyze trades.csv --capital 100000 --output report.html
```

### Build the exe yourself

```bash
pip install pyinstaller
pyinstaller WaleMonteCarlo.spec --noconfirm     # -> dist/WaleMonteCarlo.exe
```

### Run tests

```bash
python -m pytest tests/ -q                      # unit tests
python scripts/independent_validation.py        # audits the running app
```

Note for the exe: Windows SmartScreen may warn on first run because the binary
is not code-signed. Click "More info", then "Run anyway". The app makes no
network calls; everything stays on your machine.

---

## Supported Trade List Formats

Format detection is automatic - drop the file in as-is:

| Format | How it's recognized |
|--------|--------------------|
| **TradingView "List of trades" export** | Two rows per trade (`Entry long` / `Exit long`), `Net P&L USD` - the CSV you get from the Strategy Tester's export button |
| **Native format** | Columns `entry_time, exit_time, entry_price, exit_price, pnl, side, quantity` |
| **Generic broker export** | Any CSV with a recognizable P&L column (`pnl`, `profit`, `Profit/Loss`, `realized pnl`, …); dates optional |

Currency symbols, thousands separators, parenthesized negatives, BOMs, and open
(incomplete) trades are all handled. If a file can't be parsed, the error tells
you exactly which columns were found and what was expected.

---

## What the Analysis Does (and the statistics behind it)

**1. Bootstrap resampling.** Trades are resampled with replacement 10,000 times
(fixed position size, additive P&L). This produces distributions - not point
estimates - for final P&L, profit factor, and max drawdown. If 30% of resamples
lose money, your edge is fragile no matter how good the single backtest looked.

**2. Shuffle test (luck detector).** Trades are permuted without replacement.
Total P&L is invariant; only the *path* changes. If your original ordering's max
drawdown sits in the bottom 10% of shuffled orderings, your smooth equity curve
was partly lucky sequencing - expect the shuffled-median drawdown going forward.

**3. Ruin analysis.** From the drawdown distribution: probability of hitting
10/20/30/40/50% drawdowns at your capital, plus the recommended capital that
keeps P(ruin) under 5% (computed from the dollar-drawdown distribution, which is
conservative).

**4. Execution stress.** Four scenarios (optimistic → extreme) combining missed
trades (0-10%) and per-trade friction. Friction is scaled to *your* strategy -
basis points of median trade notional when prices/quantities are available,
fraction of average |P&L| otherwise - so a $500-notional stock strategy isn't
judged with futures-sized dollar slippage. Each scenario runs across 2,000 seeds;
the verdict uses medians, not a single lucky draw.

**5. Composite verdict.** Sample size, bootstrap loss probability, profit-factor
degradation under friction, ruin risk, and ordering luck combine into one
classification with human-readable flags. The scoring is transparent - every
flag states the number that triggered it.

**Honest-stats notes:** Sharpe is annualized by your actual trade frequency
(sqrt of trades per year), not a blanket sqrt(252). CAGR comes from the real
date span. Profit factor is capped at 999 to keep all-winner samples finite.

---

## Assumptions and Limitations

Read this before trusting any output. Every tool in this category makes the
same core assumptions, and most don't tell you.

- **Trades are treated as independent draws.** Bootstrap resampling assumes
  your trades are exchangeable. If your strategy's results depend heavily on
  regime (2022 bear market trades vs 2024 trending trades), resampling mixes
  those regimes and can understate clustering risk.
- **Closed trades only.** Open positions in TradingView exports carry
  unrealized P&L and are excluded, with a warning telling you how much was
  left out. If most of your backtest profit sits in one open trade, the
  verdict will reflect the closed trades, which is the honest read.
- **Fixed position size.** P&L is resampled in dollars, matching a
  fixed-quantity backtest. Compounding or volatility-scaled sizing is not
  modeled.
- **Fewer than 30 trades gets a capped verdict, fewer than 10 gets no verdict
  at all.** No statistical method can separate skill from luck at that sample
  size, so the tool refuses to pretend otherwise.
- **Friction scenarios are estimates.** Slippage in basis points of notional
  is a reasonable model for liquid instruments, but nothing replaces measured
  live fills.

The verdict is a screening tool. It can tell you a strategy is fragile. It
cannot tell you a strategy is guaranteed to work.

---

## Research Grid Engine (advanced CLI)

Beyond the Stress Lab, the repo contains a research-grade grid engine that sweeps
the full perturbation surface (skip × slippage × delay × shuffle × bootstrap -
up to 6,048 cells × 200K permutations) with crash-safe resume. The rest of this
document covers that engine.

```powershell
python scripts/run_simulation.py --trades your_trades.csv --mode explore --jobs 8
python scripts/run_simulation.py --trades your_trades.csv --status mc_run_YYYYMMDD_HHMMSS
```

---

## The Philosophy

### Why Monte Carlo?

Traditional backtesting gives you a single path through history. But that path is just *one* of many possibilities. What if:

- Your broker had a 30-second outage and you missed an entry?
- Slippage was 2x worse during a volatility spike?
- Your limit order filled 1 bar late?

Monte Carlo simulation lets us explore the space of "what could have happened" by systematically perturbing the historical record in realistic ways.

### Why This Design?

I built this engine after getting burned by strategies that looked great in backtests but fell apart in live trading. The problems were always the same:

1. **Slippage was underestimated**  -  Real fills are worse than historical prices
2. **Execution delays killed edge**  -  Even 1-bar delays can destroy mean-reversion strategies
3. **Lucky trade sequences**  -  The exact ordering of wins/losses mattered more than I thought
4. **Missing trades**  -  Technical issues, connection drops, risk limits  -  trades get skipped

This engine forces you to confront all of these before you risk real capital.

### The Core Insight

A strategy is robust if it works across a *range* of adverse conditions, not just the idealized backtest. We don't care about the mean outcome - we care about the **tail risk**.

That's why we focus on metrics like:
- **P95 Max Drawdown**  -  What's the worst drawdown in 95% of scenarios?
- **P05 Total Return**  -  What's the floor on returns?
- **P(MaxDD > 40%)**  -  How likely is a catastrophic drawdown?

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT DATA                               │
├─────────────────────────────────────────────────────────────────┤
│  trade list CSV (--trades)                 │  OHLC data (opt)   │
│  - entry/exit times                        │  - for delay model │
│  - prices, PnL, side, quantity             │  - databento, etc  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PERTURBATION ENGINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│   │ p_skip   │  │ slippage │  │  delay   │  │   shuffle    │    │
│   │ 0-10%    │  │ $0-$300  │  │ 0-3 bars │  │ permute/block│    │
│   └──────────┘  └──────────┘  └──────────┘  └──────────────┘    │
│                                                                 │
│   ┌──────────────┐                                              │
│   │  bootstrap   │  ← Resampling with replacement               │
│   │ trade/block  │                                              │
│   └──────────────┘                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GRID SEARCH ENGINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each parameter combination (cell):                         │
│    → Run N permutations (e.g., 200,000)                         │
│    → Compute distribution of outcomes                           │
│    → Save metrics_compact.csv                                   │
│                                                                 │
│  Grid dimensions:                                               │
│    p_skip:     [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]       │
│    slip:       [0, 25, 50, 75, 100, 150, 200, 300]              │
│    delay:      [0, 1, 2, 3]                                     │
│    shuffle:    [none, permute, block_permute]                   │
│    bootstrap:  [none, trade_bootstrap, block_bootstrap]         │
│    block_len:  [5, 10, 20]                                      │
│                                                                 │
│  Total cells: 7 × 8 × 4 × 3 × 3 × 3 = 6,048 (unfiltered)        │
│  With delay=1: ~1,500 cells                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYSIS ENGINE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Robust Score = PF_P50 × (1 - P_value_corrected)                │
│                                                                 │
│  Outputs:                                                       │
│    - Ranking by robust score                                    │
│    - Pareto fronts (PF vs MaxDD, multi-dimensional)             │
│    - Plateau clusters (stable parameter regions)                │
│    - Decision-grade markdown report                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Databento Integration (OHLC Data)

The engine uses OHLC (Open-High-Low-Close) data to model realistic execution delays. When you delay a trade entry by N bars, the engine looks up the actual price at that future bar instead of guessing.

### Why You Need OHLC Data

Without OHLC data, the engine estimates delay impact using a statistical model. With OHLC data, it uses real historical prices, which is more accurate for stress testing.

### Setting Up Databento

1. **Get an API key** from [databento.com](https://databento.com)

2. **Set your API key** (pick one method):

   **Option A: Environment variable (recommended)**
   ```powershell
   # PowerShell
   $env:DATABENTO_API_KEY = "db-your-key-here"
   ```
   
   **Option B: Command line argument**
   ```powershell
   python scripts/fetch_ohlc.py --key db-your-key-here
   ```

3. **Fetch OHLC data:**
   ```bash
   python scripts/fetch_ohlc.py --symbol NQ --start 2023-01-01 --end 2026-01-29
   ```

4. **Run your simulation** (it automatically uses `ohlc.csv` if present next to your trade list):
   ```bash
   python scripts/run_simulation.py --trades trade_list.csv --n_per_cell 1000
   ```

### fetch_ohlc.py Options

| Option | Description |
|--------|-------------|
| `--key` | Databento API key (or use DATABENTO_API_KEY env var) |
| `--symbol` | Futures symbol (default: NQ) |
| `--start` | Start date YYYY-MM-DD |
| `--end` | End date YYYY-MM-DD |
| `--output` | Output file (default: ohlc.csv) |
| `--schema` | Bar size: ohlcv-1m, ohlcv-1h, ohlcv-1d (default: ohlcv-1h) |

---

## Perturbation Models

### 1. Trade Skipping (`p_skip`)

**What it simulates:** Technical issues, broker outages, risk limit breaches, manual overrides.

**How it works:** Each trade is independently skipped with probability `p_skip`. If a trade is skipped, its return and PnL contribute 0 to the equity curve for that simulation.

**Typical values:**
| p_skip | Interpretation |
|--------|----------------|
| 0.00   | Perfect execution (baseline) |
| 0.01   | 1% of trades missed  -  minor issues |
| 0.02   | 2% missed  -  occasional problems |
| 0.05   | 5% missed  -  significant reliability issues |
| 0.10   | 10% missed  -  severe infrastructure problems |

**What to look for:** If your strategy collapses at p_skip=0.02, it's probably over-optimized to a specific sequence of trades.

---

### 2. Slippage (`slip_dollars`)

**What it simulates:** Market impact, bid-ask spread, partial fills, adverse price movement during execution.

**How it works:** Each executed trade incurs a random slippage cost uniformly distributed between 0 and `slip_dollars_max`. This cost is subtracted from the trade PnL.

**Typical values:**
| Slip ($) | Interpretation |
|----------|----------------|
| 0        | Zero slippage (unrealistic) |
| 25       | Minimal slippage  -  very liquid markets |
| 50       | Typical for liquid index futures |
| 100      | Moderate slippage  -  less liquid conditions |
| 200-300  | High slippage  -  volatile/illiquid conditions |

**What to look for:** Many strategies that look great at slip=$0 become unprofitable at slip=$100. If your edge disappears with realistic slippage, it wasn't a real edge.

---

### 3. Execution Delay (`delay_bars_max`)

**What it simulates:** Order routing latency, queue position, limit order non-fills, manual confirmation delays.

**How it works:** Entry and exit prices are shifted by 0 to `delay_bars_max` bars. The engine uses actual OHLC data to find the realistic fill price at the delayed timestamp.

**Key constraint:** Delay can only *hurt*  -  if the delayed fill would be better, we keep the original price. This is conservative and realistic (you rarely get *better* fills due to delays).

**OHLC Model:** When OHLC data is available, we use the actual open prices at the delayed bar. This is more realistic than statistical approximations.

**Typical values:**
| Delay | Interpretation |
|-------|----------------|
| 0     | Instant execution (unrealistic for retail) |
| 1     | 1-bar delay  -  typical for manual/slow execution |
| 2     | 2-bar delay  -  poor infrastructure |
| 3     | 3-bar delay  -  severe issues |

**What to look for:** Mean-reversion strategies are especially sensitive to delay. A strategy that works at delay=0 but fails at delay=1 is probably capturing spurious patterns.

---

### 4. Sequence Shuffling (`shuffle_mode`)

**What it simulates:** Dependence on the exact ordering of trades.

**How it works:**
- **`none`**: Original trade order preserved
- **`permute`**: Trades randomly reordered (full shuffle)
- **`block_permute`**: Blocks of N trades shuffled (preserves some local structure)

**Why it matters:** If your equity curve looks smooth only because wins and losses happened to alternate nicely, shuffling will expose that luck. A robust strategy should have similar distributions under different orderings.

---

### 5. Bootstrap Resampling (`bootstrap_mode`)

**What it simulates:** Drawing from the same underlying distribution with replacement  -  "what if we had different trades from the same strategy?"

**How it works:**
- **`none`**: Use original trades
- **`trade_bootstrap`**: Sample trades with replacement (some trades appear multiple times, some never)
- **`block_bootstrap`**: Sample blocks of N trades with replacement (preserves autocorrelation)

**Why I think it matters:** Bootstrap lets us estimate confidence intervals around metrics. If the P50 profit factor varies wildly under bootstrap, we don't have enough trades to draw reliable conclusions.

---

## Grid Search System

### Why Grid Search?

We don't just run one set of perturbation parameters  -  we run *all* combinations. This creates a **surface** that shows how strategy performance degrades across the parameter space.

### Default Grid (Full Surface)

```python
grid = {
    "p_skip":    [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10],  # 7 values
    "slip":      [0, 25, 50, 75, 100, 150, 200, 300],          # 8 values
    "delay":     [0, 1, 2, 3],                                  # 4 values
    "shuffle":   ["none", "permute", "block_permute"],         # 3 values
    "bootstrap": ["none", "trade_bootstrap", "block_bootstrap"], # 3 values
    "block_len": [5, 10, 20],                                  # 3 values
}
# Total: 7 × 8 × 4 × 3 × 3 × 3 = 6,048 cells
```

With `delay` fixed to 1 (the recommended default), this reduces to ~1,500 cells.

### Cell Identification

Each cell gets a human-readable ID built from its parameter values:

```
skip0.03_slip50_delay0_shufpermute_boottrade_bootstrap_blk10
  → p_skip=0.03, slip=$50, delay=0 bars,
    shuffle="permute", bootstrap="trade_bootstrap", block_len=10
```

The per-cell output directory is `per_cell/cell_<id>/`.

### Permutations Per Cell

We run 200,000 permutations per cell by default. Why so many?

- **Convergence**: Statistical estimates stabilize around 100K-200K samples
- **Tail estimation**: To reliably estimate P95/P99, you need many samples
- **P-value precision**: With 200K samples, we can detect effects as small as 0.01%

**Total simulations for full grid:** 1,500 cells × 200,000 perms = **300 million simulations**

This takes ~24-48 hours on a Ryzen 7700X with 8 parallel workers.

---

## Resume & Correctness Guarantees

### The Problem Solved

The original runner had a critical bug: on resume after a crash, it would restart from the wrong position and produce **duplicate permutations**. This inflated `n_perms_done` beyond the target and corrupted the statistics.

### The Fix: Source of Truth

**`metrics_compact.csv` is the SOURCE OF TRUTH, not `progress.json`.**

On every resume:
1. Read `metrics_compact.csv`
2. Dedupe by `perm_index`, keeping first occurrence
3. Count unique `perm_index` values → `n_done`
4. Find `max(perm_index)` → resume starts at `max + 1`
5. Rewrite deduped CSV atomically

### Guarantees

| Guarantee | Implementation |
|-----------|----------------|
| **Exactly N unique permutations** | Resume at `max(perm_index) + 1`; truncate if exceeds |
| **No duplicates** | Dedupe on every resume; first occurrence wins |
| **Atomic writes** | All files use `tmp` + `replace` pattern |
| **Crash-safe** | Can Ctrl+C anytime; progress persists |
| **Deterministic** | Same seed + cell_id + perm_index = same result |

### Seeding Scheme

```
cell_seed = int(sha256(cell_id)[:8], 16)
perm_seed = (cell_seed + perm_index × 1000003) mod 2^32
```

This ensures:
- Different cells get different seed ranges
- Same cell always produces same results
- No seed collisions across cells

---

## Statistical Analysis

### Robust Score

```
Robust Score = PF_P50 × (1 - P_value_corrected)
```

Where:
- **PF_P50**: Median profit factor across all permutations
- **P_value_raw**: Fraction of permutations where PF ≥ baseline_PF
- **P_value_corrected**: Bonferroni-corrected p-value (× number of cells)

A high robust score means: "This parameter combination produces good median performance AND the result is statistically unlikely to be due to chance."

### Pareto Front Analysis

We compute Pareto-optimal cells along multiple dimensions:

**2D Front (PF vs MaxDD):**
- X-axis: Profit Factor P50 (higher is better)
- Y-axis: Max Drawdown P95 (lower is better)
- Pareto-optimal: No other cell dominates on both metrics

**3D Front (PF × Return vs MaxDD):**
- Adds Return P50 as a third objective
- Identifies cells that balance all three

### Plateau Clustering

Clusters cells with similar robust scores to identify **stable parameter regions**. If a cluster contains cells with p_skip ranging from 0.02-0.05 and all have similar scores, that's a robust region  -  your results aren't sensitive to exact parameter choice.

---

## File Structure

```
montecarlo_output/mc_run_<timestamp>/
│
├── aggregated/
│   ├── run_manifest.json      # Complete run configuration
│   │   {
│   │     "run_name": "mc_run_20260701_164335",
│   │     "created": "...",
│   │     "config": {"input_dir": "...", "trades_path": "...", "n_per_cell": 200000, ...},
│   │     "baseline": {"profit_factor": 2.30, "total_return_pct": 4.21, ...},
│   │     "seed_scheme": "...",
│   │     "resume_scheme": "...",
│   │     "grid": {"total_cells": 6048, ...}
│   │   }
│   │
│   ├── progress.csv           # Status of all cells
│   │   cell_id, status, perms_completed, perms_target, pct, ...
│   │
│   ├── grid_summary.csv       # Summary stats for all completed cells
│   │   cell_id, pf_p05, pf_p50, pf_p95, ret_p50, maxdd_p95,
│   │   pvalue_raw, pvalue_corrected, robust_score, ...
│   │
│   ├── heartbeat.json         # Updated every 30 seconds (and at completion)
│   │   {
│   │     "run_name": "mc_run_20260701_164335",
│   │     "timestamp": "2026-07-01T16:43:35",
│   │     "cells_completed": 847,
│   │     "cells_total": 6048,
│   │     "perms_completed": 169400000,
│   │     "perms_total": 1209600000,
│   │     "pct_complete": 14.0
│   │   }
│   │
│   ├── DONE.txt               # Sentinel file (written when complete)
│   │
│   └── analysis/              # Created by scripts/analyze_run.py
│       ├── SURFACE_FULL_DECISION_REPORT.md
│       └── tables/
│           ├── top_50_by_robust_score.csv
│           ├── pareto_front_pf_vs_maxdd.csv
│           ├── pareto_front_multidim.csv
│           ├── plateau_clusters.csv
│           └── all_cells.csv           (with --export_csv)
│
└── per_cell/
    └── cell_<id>/
        ├── metrics_compact.csv   # THE SOURCE OF TRUTH
        │   perm_index, seed, n_trades_executed, total_return_pct,
        │   max_drawdown_pct, profit_factor, worst_month_pct, ...
        │   (exactly n_per_cell rows when complete)
        │
        ├── progress.json         # Advisory progress (not authoritative)
        │
        ├── summary.json          # Final statistics
        │   {
        │     "profit_factor": {"p05": 1.2, "p50": 2.8, "p95": 7.1, ...},
        │     "total_return": {...}, "max_drawdown": {...}, "worst_month": {...},
        │     "pvalue_raw": 0.023,
        │     "pvalue_corrected": 0.34,
        │     "robust_score": 1.84
        │   }
        │
        └── logs.txt              # Human-readable log for this cell
```

---

## Command Reference

### Simulation Runner

```bash
python scripts/run_simulation.py [OPTIONS]

Required (new runs):
  --trades PATH         Path to trade list CSV (any filename; must have
                        entry_time, exit_time, pnl columns)

Optional:
  --mode MODE           Grid preset: explore (128 cells, 20K perms),
                        focus (64 cells, 100K perms), full (6048 cells, 200K perms)
  --n_per_cell INT      Permutations per cell (overrides the mode preset; default 1000)
  --jobs INT            Parallel workers (default: 4)
  --output_dir PATH     Custom output base directory (default: ./montecarlo_output)
  --fixed_delay INT     Fix the delay dimension to one value (shrinks the grid)

Resume/Status (no --trades needed):
  --resume RUN_NAME     Resume an interrupted run (config restored from manifest)
  --status RUN_NAME     Check progress of existing run

Preview:
  --dry_run             Show config and grid size without running
```

### Example Commands

```bash
# Start a new run with 100k permutations per cell on the full grid
python scripts/run_simulation.py --trades trade_list.csv --n_per_cell 100000 --jobs 8

# Check status
python scripts/run_simulation.py --status mc_run_20260129_012613

# Resume interrupted run
python scripts/run_simulation.py --resume mc_run_20260129_012613

# Dry run to see grid dimensions
python scripts/run_simulation.py --trades trade_list.csv --mode explore --dry_run
```

### Analysis Script

```bash
python scripts/analyze_run.py --run_dir montecarlo_output/<run_name> [--export_csv] [--top_n 50]
```

Outputs go to `<run_dir>/aggregated/analysis/` (decision report + ranking tables).

### Surface Runner (repo-style layout)

`scripts/run_surface.py` is an alternative entry point that expects a directory
containing `trade_list.csv` (the "repo" layout) and defaults to `delay=1`,
producing a 1,512-cell grid:

```bash
python scripts/run_surface.py --repo path/to/export --n_per_cell 200000 --jobs 8
```


---

## **FAQ**

### Q: How long does a full run take?

**A:** With 1,500 cells × 200,000 permutations on a Ryzen 7700X (8 workers):
- ~200-400 simulations/second per worker
- ~24-48 hours total

You can reduce this by:
- Fewer permutations (50K is often sufficient for initial exploration)
- Fewer cells (fix more dimensions)
- More cores (scales linearly)

### Q: Can I stop and resume?

**A:** Yes. Hit Ctrl+C anytime. Progress is saved per-cell. Resume with `python scripts/run_simulation.py --resume <run_name>` - the trades path and grid are restored from the run manifest.

### Q: What if I see duplicates in my old runs?

**A:** The new code auto-dedupes on resume. Just run the command again and it will clean up existing data.

### Q: How do I interpret the results?

**A:** Focus on:
1. **Robust Score > 1.5**: Generally indicates edge survives stress
2. **MaxDD P95 < 40%**: Tail risk is manageable
3. **Plateau stability**: Score doesn't change much with small parameter changes

### Q: What input data do I need?

**A:** Minimum:
- A trade list CSV: entry/exit times, prices, PnL (see [Trade List Format](#trade-list-format))

Optional:
- `ohlc.csv` or `price_data.csv` next to your trade list: OHLC data for realistic delay modeling (see [Databento Integration](#databento-integration-ohlc-data))

Baseline metrics for p-values are computed automatically from your unperturbed trade list and stored in `run_manifest.json`.

### Q: Why 200,000 permutations?

**A:** Statistical convergence. Key percentiles (P05, P50, P95) stabilize around 100K. We use 200K for extra precision on tail estimates and reliable p-values... and just for fun :)

---

## Contributing

This is a personal research tool, but if you find bugs or have improvements, feel free to open an issue or PR.

## License

CC BY-NC-SA 4.0. You can use, modify, and share this code, but NOT for commercial purposes or sale. If you build on it, share your improvements under the same license. No warranty. Use at your own risk.

---
