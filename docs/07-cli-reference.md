# CLI Reference

Command-line interface documentation for the Wale Backtesting Engine.

---

## Core Commands

### Backtest Engine

The main backtest runner that converts PineScript to Python and validates against TradingView.

```bash
python backtest_engine.py [OPTIONS]
```

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `--csv` | Path to OHLC CSV data file |
| `--pine` | Path to PineScript strategy file |

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--excel` | None | TradingView Excel export for validation |
| `--run_step1` | false | Generate Step 1 research outputs |
| `--run_step2` | false | Run robustness tests |
| `--holdout_months` | 6 | Out-of-sample holdout period |
| `--initial_capital` | 100000 | Starting capital |
| `--commission_pct` | 0.1 | Commission as percent of trade value |

#### Examples

**Basic backtest:**
```bash
python backtest_engine.py \
    --csv jan_2_data_to_now.csv \
    --pine strategy.pine
```

**With validation:**
```bash
python backtest_engine.py \
    --csv jan_2_data_to_now.csv \
    --pine strategy.pine \
    --excel tradingview_export.xlsx \
    --run_step1 true
```

**Full research run:**
```bash
python backtest_engine.py \
    --csv jan_2_data_to_now.csv \
    --pine strategy.pine \
    --excel tradingview_export.xlsx \
    --run_step1 true \
    --run_step2 true \
    --holdout_months 6
```

---

### Monte Carlo Surface Runner

Run full grid search Monte Carlo stress tests.

```bash
python CURSOR_run_surface_full_200k.py [OPTIONS]
```

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `--repo` | Path to repository root containing data |

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--run_name` | Auto-generated | Name for this run (for resume) |
| `--n_per_cell` | 200000 | Permutations per grid cell |
| `--jobs` | min(8, CPU cores) | Parallel worker processes |
| `--checkpoint_every` | 2000 | Save progress every N sims |
| `--base_seed` | 1337 | Random seed for reproducibility |
| `--fixed_delay` | 1 | Execution delay in bars |
| `--slip_min` | 0.0 | Minimum slippage to test |
| `--slip_max` | 300.0 | Maximum slippage to test |
| `--status_only` | false | Print status and exit |

#### Examples

**Start a new run:**
```bash
python CURSOR_run_surface_full_200k.py \
    --repo "C:\Users\wale\wale_backtest\codex\1.7.26 data" \
    --n_per_cell 200000 \
    --jobs 8
```

**Resume an interrupted run:**
```bash
python CURSOR_run_surface_full_200k.py \
    --repo "C:\Users\wale\wale_backtest\codex\1.7.26 data" \
    --run_name mc_surface_full_200k_20260115_125153
```

**Quick test with fewer permutations:**
```bash
python CURSOR_run_surface_full_200k.py \
    --repo "C:\Users\wale\wale_backtest\codex\1.7.26 data" \
    --n_per_cell 1000 \
    --jobs 4
```

**Check status of running job:**
```bash
python CURSOR_run_surface_full_200k.py \
    --repo "C:\Users\wale\wale_backtest\codex\1.7.26 data" \
    --status_only
```

---

### Monte Carlo Analysis

Analyze completed Monte Carlo runs.

```bash
python CURSOR_surface_full_200k_analysis.py [OPTIONS]
```

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `--run_dir` | Path to completed MC run directory |

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--top_n` | 50 | Number of top cells to report |

#### Examples

**Run analysis:**
```bash
python CURSOR_surface_full_200k_analysis.py \
    --run_dir "backtest/out/montecarlo/mc_surface_full_200k_20260115_125153"
```

---

## Utility Commands

### Check Run Status

Monitor a running Monte Carlo job:

```powershell
# PowerShell
Get-Content "backtest\out\montecarlo\mc_surface_full_200k_*\aggregated\heartbeat.json"
```

```bash
# Bash
cat backtest/out/montecarlo/mc_surface_full_200k_*/aggregated/heartbeat.json
```

Output:
```json
{
  "timestamp": "2026-01-16T10:06:34",
  "run_name": "mc_surface_full_200k_20260115_125153",
  "current_cell_id": "3_3_0_1_1",
  "cells_complete": 1468,
  "cells_total": 1512,
  "sims_done_total_estimate": 302400000,
  "uptime_seconds": 76481
}
```

### Count Completed Cells

```powershell
# PowerShell
(Get-ChildItem "backtest\out\montecarlo\mc_*\per_cell\*\summary.json").Count
```

### View Grid Summary

```powershell
# PowerShell - Top 10 by robust score
Import-Csv "backtest\out\montecarlo\mc_*\aggregated\grid_summary.csv" |
    Sort-Object {[double]$_.robust_score} -Descending |
    Select-Object -First 10 |
    Format-Table cell_id, pf_p50, maxdd_p95, robust_score
```

---

## Output Locations

### Backtest Engine Outputs

```
backtest/out/
├── validation_report.txt       # TradingView comparison results
├── research_outputs/
│   ├── step1/
│   │   ├── trade_list.csv      # All trades with entry/exit details
│   │   ├── equity_curve.csv    # Bar-by-bar equity
│   │   └── step1_report.txt    # Performance summary
│   └── step2/
│       └── ...                 # Robustness test results
└── plots/
    ├── step1_equity_drawdown.png
    ├── step1_monthly_returns.png
    └── step1_trade_pnl_hist.png
```

### Monte Carlo Outputs

```
backtest/out/montecarlo/mc_surface_full_200k_<timestamp>/
├── aggregated/
│   ├── run_manifest.json       # Run configuration
│   ├── grid_summary.csv        # All cells summary stats
│   ├── heartbeat.json          # Current progress
│   ├── progress.csv            # Detailed progress log
│   └── DONE.txt                # Completion sentinel
└── per_cell/
    └── cell_<p_skip>_<slip>_<shuffle>_<bootstrap>_<block>/
        ├── summary.json        # Cell statistics
        ├── metrics_compact.csv # All 200K sim results
        ├── progress.json       # Cell progress
        └── logs.txt            # Execution log
```

---

## Environment Setup

### Required Python Packages

```bash
pip install numpy pandas scipy openpyxl matplotlib
```

### Recommended: Create Virtual Environment

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### Check Installation

```bash
python -c "import numpy, pandas, scipy; print('OK')"
```

---

## Troubleshooting

### "Module not found" Errors

```bash
# Ensure you are in the correct directory
cd "C:\Users\wale\wale_backtest\codex\1.7.26 data"

# Ensure virtual environment is activated
.\venv\Scripts\activate
```

### "Permission denied" on Windows

Run PowerShell as Administrator, or use:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Out of Memory During Monte Carlo

Reduce parallel workers:
```bash
python CURSOR_run_surface_full_200k.py --jobs 4
```

### Stuck Monte Carlo Run

1. Check if processes are still running:
```powershell
Get-Process python
```

2. If stuck, kill and resume:
```powershell
Get-Process python | Stop-Process -Force
# Then run the same command again - it will resume
```

---

## Quick Reference Card

| Task | Command |
|------|---------|
| Run backtest | `python backtest_engine.py --csv data.csv --pine strat.pine` |
| Validate vs TV | Add `--excel export.xlsx --run_step1 true` |
| Start MC run | `python CURSOR_run_surface_full_200k.py --repo .` |
| Resume MC run | Add `--run_name <run_name>` |
| Check MC status | `cat aggregated/heartbeat.json` |
| Analyze MC results | `python CURSOR_surface_full_200k_analysis.py --run_dir <dir>` |
