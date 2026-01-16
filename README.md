# Wale Monte Carlo Backtesting Engine

> A battle-tested, production-grade Monte Carlo simulation framework for stress-testing trading strategies under realistic market conditions.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

## What This Is (And Why It Matters)

If you're running a systematic trading strategy, you've probably asked yourself: *"Is my backtest too good to be true?"* *"Is my strategy overfit?"*

The honest answer is usually: *yes, probably.*

This engine exists to answer a harder question: **"What happens when everything that can go wrong, does go wrong?"**

We do this by running hundreds of thousands of Monte Carlo simulations that systematically degrade your backtest in realistic ways:
- What if you miss trades due to technical issues?
- What if slippage is worse than expected?
- What if execution is delayed by a few bars?
- What if the sequence of trades was just lucky?

The result isn't a single equity curve - it's a *distribution* of outcomes. And from that distribution, you can make actual decisions about position sizing, risk limits, and whether this strategy is worth trading at all.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [The Philosophy](#the-philosophy)
3. [Architecture Overview](#architecture-overview)
4. [Perturbation Models](#perturbation-models)
5. [Grid Search System](#grid-search-system)
6. [Resume & Correctness Guarantees](#resume--correctness-guarantees)
7. [Statistical Analysis](#statistical-analysis)
8. [File Structure](#file-structure)
9. [Command Reference](#command-reference)
10. [FAQ](#faq)

### Additional Documentation

- [PineScript to Python Engine](docs/05-pinescript-to-python-engine.md) - Converting TradingView strategies
- [Accuracy Testing Guide](docs/06-accuracy-testing-guide.md) - Validating your implementation
- [CLI Reference](docs/07-cli-reference.md) - Full command-line documentation

---

## Quick Start

### Prerequisites

```bash
pip install numpy pandas matplotlib
```

### Run Your First Simulation

```powershell
# Set deterministic seed for reproducibility
$env:PYTHONHASHSEED = "0"

# Run the full 200K permutation grid
python CURSOR_run_surface_full_200k.py `
    --repo "C:\path\to\your\strategy" `
    --n_per_cell 200000 `
    --jobs 8
```

### Check Progress While Running

```powershell
# Quick status
python CURSOR_run_surface_full_200k.py --repo "." --run_name <your_run> --status_only

# Watch heartbeat
Get-Content "backtest\out\montecarlo\<run>\aggregated\heartbeat.json"
```

### Analyze Results

```powershell
python CURSOR_surface_full_200k_analysis.py --run_dir "backtest\out\montecarlo\<run>"
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
│                        INPUT DATA                                │
├─────────────────────────────────────────────────────────────────┤
│  trade_list.csv      │  equity_curve.csv   │  OHLC data (opt)   │
│  - entry/exit times  │  - bar-by-bar equity│  - for delay model │
│  - prices, PnL       │  - timestamps       │                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PERTURBATION ENGINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│   │ p_skip   │  │ slippage │  │  delay   │  │   shuffle    │   │
│   │ 0-10%    │  │ $0-$300  │  │ 0-3 bars │  │ permute/block│   │
│   └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│                                                                  │
│   ┌──────────────┐                                              │
│   │  bootstrap   │  ← Resampling with replacement               │
│   │ trade/block  │                                              │
│   └──────────────┘                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GRID SEARCH ENGINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  For each parameter combination (cell):                         │
│    → Run N permutations (e.g., 200,000)                         │
│    → Compute distribution of outcomes                           │
│    → Save metrics_compact.csv                                   │
│                                                                  │
│  Grid dimensions:                                                │
│    p_skip:     [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]       │
│    slip:       [0, 25, 50, 75, 100, 150, 200, 300]              │
│    delay:      [0, 1, 2, 3]                                     │
│    shuffle:    [none, permute, block_permute]                   │
│    bootstrap:  [none, trade_bootstrap, block_bootstrap]         │
│    block_len:  [5, 10, 20]                                      │
│                                                                  │
│  Total cells: 7 × 8 × 4 × 3 × 3 × 3 = 6,048 (unfiltered)       │
│  With delay=1: ~1,500 cells                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYSIS ENGINE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Robust Score = PF_P50 × (1 - P_value_corrected)                │
│                                                                  │
│  Outputs:                                                        │
│    - Ranking by robust score                                    │
│    - Pareto fronts (PF vs MaxDD, multi-dimensional)             │
│    - Plateau clusters (stable parameter regions)                │
│    - Decision-grade markdown report                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

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

Each cell gets a unique ID based on its parameter indices:
```
cell_3_2_0_1_1 = {
    p_skip_idx: 3 → p_skip=0.03
    slip_idx: 2 → slip=50
    delay_idx: 0 → delay=0
    shuffle_idx: 1 → shuffle="permute"
    bootstrap_idx: 1 → bootstrap="trade_bootstrap"
}
```

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
cell_seed_base = (global_seed + sha256(cell_id)[:8] % seed_stride) mod 2^32
perm_seed = (cell_seed_base + perm_index × 1000003) mod 2^32
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
backtest/out/montecarlo/mc_surface_full_200k_<timestamp>/
│
├── aggregated/
│   ├── run_manifest.json      # Complete run configuration
│   │   {
│   │     "run_name": "mc_surface_full_200k_20260115",
│   │     "total_cells": 1512,
│   │     "n_per_cell_target": 200000,
│   │     "global_seed": 1337,
│   │     "seed_scheme": "...",
│   │     "resume_scheme": "...",
│   │     ...
│   │   }
│   │
│   ├── progress.csv           # Status of all cells
│   │   cell_id, status, n_done, n_target, pct_done, ...
│   │
│   ├── grid_summary.csv       # Summary stats for complete cells
│   │   cell_id, pf_p50, maxdd_p95, ret_p50, robust_score, ...
│   │
│   ├── heartbeat.json         # Updated every 30 seconds
│   │   {
│   │     "timestamp": "2026-01-15T14:30:00",
│   │     "cells_complete": 847,
│   │     "cells_total": 1512,
│   │     "sims_done_total_estimate": 169400000
│   │   }
│   │
│   ├── DONE.txt               # Sentinel file (written when complete)
│   │
│   └── analysis/              # Created by analysis script
│       ├── SURFACE_FULL_200K_DECISION_REPORT.md
│       └── tables/
│           ├── top_50_by_robust_score.csv
│           ├── pareto_front_pf_vs_maxdd.csv
│           ├── pareto_front_multidim.csv
│           └── plateau_clusters.csv
│
└── per_cell/
    └── cell_<id>/
        ├── metrics_compact.csv   # THE SOURCE OF TRUTH
        │   perm_index, total_return_pct, max_drawdown_pct, profit_factor, ...
        │   (exactly 200,000 rows when complete)
        │
        ├── progress.json         # Advisory progress (not authoritative)
        │   {
        │     "n_done": 200000,
        │     "n_target": 200000,
        │     "runtime_seconds": 1847.3,
        │     ...
        │   }
        │
        ├── summary.json          # Final statistics
        │   {
        │     "quantiles": {"profit_factor": {"p05": 1.2, "p50": 2.8, "p95": 7.1}},
        │     "p_value": {"raw": 0.023, "corrected": 0.34},
        │     "robust_score": {"value": 1.84},
        │     ...
        │   }
        │
        └── logs.txt              # Human-readable log
            [2026-01-15T14:00:00] START: cell=3_2_0_1_1 target=200000
            [2026-01-15T14:30:47] COMPLETE: n_done=200000
```

---

## Command Reference

### Main Runner

```powershell
python CURSOR_run_surface_full_200k.py [OPTIONS]

Required:
  --repo PATH           Repository root containing backtest data

Optional:
  --run_name NAME       Resume existing run (auto-generated if omitted)
  --n_per_cell INT      Permutations per cell (default: 200000)
  --jobs INT            Parallel processes (default: min(8, CPU cores))
  --checkpoint_every INT  Checkpoint interval (default: 2000)
  --base_seed INT       Global seed (default: 1337)
  
Grid Filters:
  --fixed_delay INT     Fix delay to single value (default: 1)
  --slip_min FLOAT      Minimum slip dollars (default: 0)
  --slip_max FLOAT      Maximum slip dollars (default: 300)
  --include_zero_slip   Include slip=0 in grid (default: True)
  
Modes:
  --status_only         Print status and exit
  --skip_extension      Skip top-k extension phase (default: True)
```

### Analysis Script

```powershell
python CURSOR_surface_full_200k_analysis.py [OPTIONS]

Required:
  --run_dir PATH        Path to completed run directory

Optional:
  --top_n INT           Number of top cells to include in report (default: 50)
```

---

## PineScript to Python Engine

This project includes a complete PineScript to Python conversion engine that enables local backtesting with TradingView-equivalent accuracy.

### What It Does

1. **Parses PineScript files** - Extracts strategy parameters, presets, and input values
2. **Implements indicators in Python** - EMA, RMA, ATR, ADX with exact PineScript math
3. **Runs backtests locally** - Full position management with proper fill timing
4. **Validates against TradingView** - Trade-by-trade comparison with Excel exports

### Why This Matters

Without accurate conversion, Monte Carlo results are meaningless. If your Python engine does not match TradingView exactly, you are stress-testing the wrong strategy.

The engine has been validated to match TradingView trade-by-trade on:
- Entry/exit times (exact bar match)
- Entry/exit prices (within 0.01 tolerance)
- PnL per trade (within 2% tolerance)
- Total equity (within 1% tolerance)

### Key Implementation Details

| Indicator | Critical Detail |
|-----------|-----------------|
| EMA | Must initialize with SMA seed, not from bar 0 |
| RMA | Uses alpha = 1/length, different from EMA |
| ATR | Uses RMA smoothing, not EMA |
| Fill Timing | Signal on close, execute on next bar open |

For complete documentation, see [PineScript to Python Engine](docs/05-pinescript-to-python-engine.md).

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

**A:** Yes. Hit Ctrl+C anytime. Progress is saved per-cell. Run the same command with `--run_name <existing_name>` to resume.

### Q: What if I see duplicates in my old runs?

**A:** The new code auto-dedupes on resume. Just run the command again and it will clean up existing data.

### Q: How do I interpret the results?

**A:** Focus on:
1. **Robust Score > 1.5**: Generally indicates edge survives stress
2. **MaxDD P95 < 40%**: Tail risk is manageable
3. **Plateau stability**: Score doesn't change much with small parameter changes

### Q: What input data do I need?

**A:** Minimum:
- `trade_list.csv`: Entry/exit times, prices, PnL
- `equity_curve.csv`: Bar-by-bar equity

Recommended:
- `jan_2_data_to_now.csv`: OHLC data for realistic delay modeling
- `step1_report.txt`: Baseline metrics for p-value calculation

### Q: Why 200,000 permutations?

**A:** Statistical convergence. Key percentiles (P05, P50, P95) stabilize around 100K. We use 200K for extra precision on tail estimates and reliable p-values... and just for fun :)

---

## Contributing

This is a personal research tool, but if you find bugs or have improvements, feel free to open an issue or PR.

## License

CC BY-NC-SA 4.0. You can use, modify, and share this code, but NOT for commercial purposes or sale. If you build on it, share your improvements under the same license. No warranty. Use at your own risk.

---
