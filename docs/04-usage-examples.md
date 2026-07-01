# Usage Examples: Real-World Scenarios

This document provides practical examples for common use cases.

---

## Prerequisites

### Required Files

All you need is a trade list CSV (any filename, passed via `--trades`):

```csv
entry_time,exit_time,entry_price,exit_price,pnl,side,quantity,symbol
2024-01-15 09:30:00,2024-01-15 10:45:00,15234.5,15267.25,654.00,long,2,NQ
2024-01-15 11:15:00,2024-01-15 11:58:00,15289.0,15245.75,-864.50,short,2,NQ
```

Optionally, place `ohlc.csv` (or `price_data.csv`) in the same directory as
your trade list to enable the OHLC-based delay model (see
`scripts/fetch_ohlc.py` for fetching data from Databento).

A ready-made sample lives at `examples/sample_trade_list.csv`.

---

## Example 1: Quick Exploration Run

Start here. 128 cells covering the main perturbation dimensions.

```bash
python scripts/run_simulation.py \
    --trades examples/sample_trade_list.csv \
    --mode explore \
    --jobs 8
```

`--mode explore` defaults to 20,000 permutations per cell. Add
`--n_per_cell 500` for a fast smoke test (finishes in seconds).

### What Happens

1. **Initialization**:
   - Loads your trade list and computes baseline metrics (PF, return, drawdown)
   - Builds the grid (128 cells in explore mode)
   - Creates `montecarlo_output/mc_run_<timestamp>/`

2. **Processing**:
   - Spawns parallel worker processes
   - Each worker processes cells and streams results to disk
   - Heartbeat updated every 30 seconds

3. **Output**:
   - Per-cell: `metrics_compact.csv`, `summary.json`, `progress.json`, `logs.txt`
   - Aggregated: `progress.csv`, `grid_summary.csv`, `heartbeat.json`, `run_manifest.json`
   - Completion: `DONE.txt` sentinel

---

## Example 2: Full 200K Grid Run

The complete stress test with all 6,048 parameter combinations.

```bash
python scripts/run_simulation.py \
    --trades your_trades.csv \
    --mode full \
    --jobs 8 \
    > console_log.txt 2>&1
```

For the repo-style layout (a directory containing `trade_list.csv`), the
surface runner fixes `delay=1` and runs 1,512 cells:

```bash
python scripts/run_surface.py --repo path/to/export --n_per_cell 200000 --jobs 8
```

### Expected Duration

Highly hardware dependent. On an 8-core machine expect hours for the full
grid at 200K perms/cell; use `--dry_run` to see the total permutation count
and a runtime estimate before committing.

---

## Example 3: Resume After Interruption

If the run was interrupted (Ctrl+C, power loss, crash):

```bash
python scripts/run_simulation.py --resume mc_run_20260701_164335
```

No `--trades` needed — the trades path, permutation target, and grid are
restored from `run_manifest.json`.

### What Happens

1. Detects the existing run directory
2. For each cell:
   - Reads `metrics_compact.csv` (the source of truth)
   - Dedupes by `perm_index` (first occurrence wins)
   - Truncates any rows past the target
   - Resumes from `max(perm_index) + 1`
3. Cells that are already complete just re-emit their `summary.json`

Resumed results are bit-identical to an uninterrupted run because every
permutation's seed depends only on `(cell_id, perm_index)`.

---

## Example 4: Quick Status Check

While a run is in progress:

```bash
python scripts/run_simulation.py --status mc_run_20260701_164335
```

Output:

```
=== Run Status: mc_run_20260701_164335 ===
Exists: True
Complete: False
Cells: 84/128
Progress: 65.6%
Last heartbeat: 2026-07-01T16:43:35
```

### Alternative: Watch the Heartbeat

```bash
# One-shot
cat montecarlo_output/mc_run_*/aggregated/heartbeat.json

# Continuous monitoring (every 30 seconds)
watch -n 30 cat montecarlo_output/mc_run_*/aggregated/heartbeat.json
```

---

## Example 5: Analyze a Completed Run

After the run finishes (`DONE.txt` appears):

```bash
python scripts/analyze_run.py \
    --run_dir montecarlo_output/mc_run_20260701_164335 \
    --export_csv
```

### Output Files

```
aggregated/analysis/
├── SURFACE_FULL_DECISION_REPORT.md    # Main report
└── tables/
    ├── top_50_by_robust_score.csv
    ├── pareto_front_pf_vs_maxdd.csv
    ├── pareto_front_multidim.csv
    ├── plateau_clusters.csv
    └── all_cells.csv                  (with --export_csv)
```

The report ranks cells by robust score, identifies Pareto-optimal
parameter combinations, and clusters stable plateaus.

---

## Example 6: Single Cell Deep Dive

To investigate a specific parameter combination:

### Step 1: Identify the Cell

From `grid_summary.csv` or the analysis report, find the cell_id:

```
skip0.03_slip50_delay1_shufpermute_boottrade_bootstrap_blk10
```

### Step 2: Inspect Its Files

```bash
cell=montecarlo_output/mc_run_*/per_cell/cell_skip0.03_slip50_delay1_shufpermute_boottrade_bootstrap_blk10
cat $cell/summary.json
head $cell/metrics_compact.csv
```

### Step 3: Distribution Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(f"{cell_dir}/metrics_compact.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

df["profit_factor"].hist(bins=100, ax=axes[0])
axes[0].set_title("Profit Factor Distribution")
axes[0].axvline(df["profit_factor"].median(), color='red', label='Median')

df["max_drawdown_pct"].hist(bins=100, ax=axes[1])
axes[1].set_title("Max Drawdown Distribution")

df["total_return_pct"].hist(bins=100, ax=axes[2])
axes[2].set_title("Total Return Distribution")

plt.tight_layout()
plt.savefig("cell_distributions.png")
```

---

## Example 7: Compare Two Runs

If you have multiple runs with different configurations:

```python
import pandas as pd

run1 = pd.read_csv("montecarlo_output/run_a/aggregated/grid_summary.csv")
run2 = pd.read_csv("montecarlo_output/run_b/aggregated/grid_summary.csv")

merged = run1.merge(run2, on="cell_id", suffixes=("_a", "_b"))

improved = merged[merged["robust_score_b"] > merged["robust_score_a"]]
print(f"Improved: {len(improved)} cells")
print(f"Correlation: {merged['robust_score_a'].corr(merged['robust_score_b']):.3f}")
```

---

## Example 8: Export for External Analysis

```python
import pandas as pd

df = pd.read_csv("montecarlo_output/mc_run_.../aggregated/grid_summary.csv")
df.to_excel("grid_summary.xlsx", index=False)
```

Or use `--export_csv` on the analysis script to get `all_cells.csv` with
every metric for every cell in one file.

---

## Troubleshooting

### Error: Trade file not found

```
Error: Trade file not found: /path/to/trades.csv
```

**Fix**: Check the `--trades` path. Relative paths are resolved from your
current working directory.

### Delay model falls back to statistical mode

If no OHLC file is found next to your trade list, delays are modeled
statistically instead of with real prices. Place `ohlc.csv` in the same
directory as your trade list (see `scripts/fetch_ohlc.py`).

### Warning: duplicates repaired on resume

```
Repaired metrics_compact.csv (duplicates=True, truncated=0)
```

This means a previous run wrote overlapping rows (e.g. crash mid-write).
The engine dedupes automatically; results remain correct.

### Run seems stuck

Check the heartbeat:

```bash
cat montecarlo_output/mc_run_*/aggregated/heartbeat.json
```

If `timestamp` hasn't updated in >60 seconds, check:
1. Is the process still running?
2. Is the disk full?
3. Did a cell crash? (Check `per_cell/cell_*/logs.txt`)

---

## Best Practices

1. **Start with `--mode explore`** and a small `--n_per_cell`, then scale up

2. **Redirect output to a file** for long runs:
   ```bash
   python scripts/run_simulation.py ... > console_log.txt 2>&1
   ```

3. **Monitor the heartbeat** periodically during long runs

4. **Run the analysis immediately after completion** while context is fresh

5. **Use `--resume <run_name>`** after any interruption — never delete a
   partial run, the engine repairs and continues it

6. **Use an SSD** if possible — lots of small file I/O
