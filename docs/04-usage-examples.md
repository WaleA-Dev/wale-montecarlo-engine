# Usage Examples: Real-World Scenarios

This document provides practical examples for common use cases.

---

## Prerequisites

### Required Files

Your strategy output folder should contain:

```
backtest/out/export_YYYYMMDD_HHMMSS/
├── trade_list.csv       # Required
├── equity_curve.csv     # Required  
└── step1_report.txt     # Optional but recommended
```

Plus OHLC data in the repo root:
```
jan_2_data_to_now.csv    # Required for delay model
```

### trade_list.csv Format

```csv
entry_time,exit_time,entry_price,exit_price,pnl,qty,side
2024-01-15T09:30:00,2024-01-15T10:45:00,15234.5,15267.25,654.00,2,long
2024-01-15T11:15:00,2024-01-15T11:58:00,15289.0,15245.75,-864.50,2,short
...
```

### equity_curve.csv Format

```csv
time,equity
2024-01-15T09:30:00,100000.00
2024-01-15T09:31:00,100012.50
2024-01-15T09:32:00,99987.25
...
```

---

## Example 1: Full 200K Grid Run

The complete stress test with all parameter combinations.

### Command

```powershell
# Set up environment
$env:PYTHONHASHSEED = "0"
cd "C:\Users\wale\wale_backtest\codex\1.7.26 data"

# Start the run
python CURSOR_run_surface_full_200k.py `
    --repo "." `
    --n_per_cell 200000 `
    --jobs 8 `
    *> "backtest\out\montecarlo\console_log.txt"
```

### What Happens

1. **Initialization**:
   - Loads trade_list.csv, equity_curve.csv, OHLC data
   - Builds grid: ~1,500 cells (with delay=1 fixed)
   - Creates output directory with timestamp

2. **Processing**:
   - Spawns 8 parallel worker processes
   - Each worker processes cells sequentially
   - Progress logged every 60 seconds
   - Heartbeat updated every 30 seconds

3. **Output**:
   - Per-cell: metrics_compact.csv, summary.json, progress.json, logs.txt
   - Aggregated: progress.csv, grid_summary.csv, heartbeat.json
   - Completion: DONE.txt sentinel

### Expected Duration

For 1,500 cells × 200,000 permutations:
- Ryzen 7700X (8 cores): ~24-48 hours
- Ryzen 9 5900X (12 cores): ~16-32 hours
- Intel i9-13900K (24 cores): ~8-16 hours

---

## Example 2: Resume After Interruption

If the run was interrupted (Ctrl+C, power loss, crash):

### Command

```powershell
# Use the SAME run_name as before
python CURSOR_run_surface_full_200k.py `
    --repo "C:\Users\wale\wale_backtest\codex\1.7.26 data" `
    --run_name mc_surface_full_200k_20260115_1234 `
    --n_per_cell 200000 `
    --jobs 8
```

### What Happens

1. Detects existing run directory
2. For each cell:
   - Reads metrics_compact.csv
   - Dedupes by perm_index
   - Finds max_perm_index completed
   - Resumes from max + 1
3. Skips cells that already have summary.json

### Verification

```powershell
# Check how many cells resumed vs started fresh
Select-String "RESUME" "backtest\out\montecarlo\mc_surface_full_200k_*\per_cell\*\logs.txt" | Measure-Object
```

---

## Example 3: Quick Status Check

While a run is in progress:

### Command

```powershell
python CURSOR_run_surface_full_200k.py `
    --repo "." `
    --run_name mc_surface_full_200k_20260115_1234 `
    --status_only
```

### Output

```
============================================================
STATUS: mc_surface_full_200k_20260115_1234
============================================================
Cells complete: 847/1512
Simulations: 169,400,000/302,400,000 (56.0%)
Last heartbeat: 2026-01-15T14:30:00
Current cell: 3_5_0_2_1
============================================================
```

### Alternative: Watch Heartbeat

```powershell
# One-shot
Get-Content "backtest\out\montecarlo\mc_surface_full_200k_*\aggregated\heartbeat.json" | ConvertFrom-Json

# Continuous monitoring (every 30 seconds)
while ($true) {
    Clear-Host
    Get-Content "backtest\out\montecarlo\mc_surface_full_200k_*\aggregated\heartbeat.json" | ConvertFrom-Json
    Start-Sleep 30
}
```

---

## Example 4: Analyze Completed Run

After the run finishes (DONE.txt appears):

### Command

```powershell
python CURSOR_surface_full_200k_analysis.py `
    --run_dir "C:\Users\wale\wale_backtest\codex\1.7.26 data\backtest\out\montecarlo\mc_surface_full_200k_20260115_1234"
```

### Output Files

```
aggregated/analysis/
├── SURFACE_FULL_200K_DECISION_REPORT.md    # Main report
└── tables/
    ├── top_50_by_robust_score.csv
    ├── pareto_front_pf_vs_maxdd.csv
    ├── pareto_front_multidim.csv
    ├── plateau_clusters.csv
    ├── all_cells_verified.csv
    └── dedupe_summary.csv
```

### View Report

```powershell
# Open in default markdown viewer
Start-Process "backtest\out\montecarlo\mc_surface_full_200k_*\aggregated\analysis\SURFACE_FULL_200K_DECISION_REPORT.md"

# Or view in terminal
Get-Content "backtest\out\montecarlo\mc_surface_full_200k_*\aggregated\analysis\SURFACE_FULL_200K_DECISION_REPORT.md"
```

---

## Example 5: Custom Grid (Faster Exploration)

For initial exploration with fewer permutations:

### Command

```powershell
python CURSOR_run_surface_full_200k.py `
    --repo "." `
    --n_per_cell 50000 `
    --jobs 8 `
    --slip_min 25 `
    --slip_max 150 `
    --no_zero_slip
```

### Changes from Default

| Parameter | Default | This Run |
|-----------|---------|----------|
| n_per_cell | 200,000 | 50,000 |
| slip range | 0-300 | 25-150 |
| zero slip | Included | Excluded |

### Expected Duration

~6-12 hours (4x fewer perms, fewer slip values)

---

## Example 6: Single Cell Deep Dive

To investigate a specific parameter combination:

### Step 1: Identify the Cell

From the analysis report, find the cell_id you want to examine:
```
Top candidate: cell_3_2_0_1_1
Parameters: p_skip=0.03, slip=$50, shuffle=permute, bootstrap=trade_bootstrap
```

### Step 2: Extract Metrics

```powershell
$cellDir = "backtest\out\montecarlo\mc_surface_full_200k_*\per_cell\cell_3_2_0_1_1"

# Summary statistics
Get-Content "$cellDir\summary.json" | ConvertFrom-Json

# Raw metrics (200K rows)
Import-Csv "$cellDir\metrics_compact.csv" | Measure-Object -Property profit_factor -Average -Maximum -Minimum
```

### Step 3: Distribution Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("per_cell/cell_3_2_0_1_1/metrics_compact.csv")

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

### Step 1: Load Summaries

```python
import pandas as pd

run1 = pd.read_csv("mc_run_conservative/aggregated/grid_summary.csv")
run2 = pd.read_csv("mc_run_aggressive/aggregated/grid_summary.csv")

# Merge on cell_id
merged = run1.merge(run2, on="cell_id", suffixes=("_cons", "_agg"))
```

### Step 2: Compare

```python
# Cells that improved
improved = merged[merged["robust_score_agg"] > merged["robust_score_cons"]]
print(f"Improved: {len(improved)} cells")

# Cells that got worse
worse = merged[merged["robust_score_agg"] < merged["robust_score_cons"]]
print(f"Degraded: {len(worse)} cells")

# Correlation
print(f"Correlation: {merged['robust_score_cons'].corr(merged['robust_score_agg']):.3f}")
```

---

## Example 8: Export for External Analysis

Export data for use in R, Julia, or other tools:

### Export Grid Summary to Excel

```python
import pandas as pd

df = pd.read_csv("aggregated/grid_summary.csv")
df.to_excel("grid_summary.xlsx", index=False)
```

### Export All Cell Metrics

```powershell
# Create combined CSV with all cells
$output = "all_cells_metrics.csv"
$first = $true

Get-ChildItem "per_cell\cell_*\metrics_compact.csv" | ForEach-Object {
    $df = Import-Csv $_
    $cellId = $_.Directory.Name -replace "cell_", ""
    
    # Add cell_id column
    $df | ForEach-Object { $_ | Add-Member -NotePropertyName "cell_id" -NotePropertyValue $cellId -PassThru }
    
    if ($first) {
        $df | Export-Csv $output -NoTypeInformation
        $first = $false
    } else {
        $df | Export-Csv $output -NoTypeInformation -Append
    }
}
```

---

## Troubleshooting

### Error: OHLC data not found

```
ERROR: OHLC CSV not found. Specify --ohlc_csv or place jan_2_data_to_now.csv in repo root.
```

**Fix**: Ensure your OHLC file exists and matches the expected format.

### Error: trade_list.csv not found

```
ERROR: trade_list.csv not found at C:\...\backtest\out\...\trade_list.csv
```

**Fix**: Specify `--input_dir` pointing to your export folder, or ensure the auto-detection finds your latest export.

### Warning: Many duplicates dropped

If you see messages about thousands of duplicates:
```
[2026-01-15T14:30:00] DEDUPE: Removed 15,000 duplicate rows
```

This indicates previous resume issues. The system handles it automatically, but the run took longer than necessary.

### Run seems stuck

Check heartbeat:
```powershell
Get-Content "aggregated\heartbeat.json"
```

If `timestamp` hasn't updated in >60 seconds, something is wrong. Check:
1. Process still running? (Task Manager)
2. Disk full?
3. Cell crashed? (Check per_cell/cell_*/logs.txt)

---

## Best Practices

1. **Always use `$env:PYTHONHASHSEED = "0"`** for reproducibility

2. **Redirect output to file** for long runs:
   ```powershell
   *> "console_log.txt"
   ```

3. **Start with fewer permutations** (50K) for exploration, then scale up

4. **Monitor heartbeat** periodically during long runs

5. **Run analysis immediately after completion** while context is fresh

6. **Save your command** so you can resume with exact same parameters

7. **Use SSD** if possible  -  lots of small file I/O
