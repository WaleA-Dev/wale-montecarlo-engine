# Changelog

## 2026-07-03 — v2.0: Strategy Stress Lab

### Added
- **Desktop app**: `WaleMonteCarlo.exe` (PyInstaller one-file, ~24 MB) — double-click,
  browser opens, drag in a CSV. Built from `WaleMonteCarlo.spec`.
- **Web UI** (`wale_montecarlo/webapp/`): dark dashboard with verdict plate,
  equity cone, P&L / drawdown distributions, ruin ladder, stress-scenario table,
  and one-click standalone HTML report export. Chart.js bundled locally — fully offline.
- **Universal CSV ingestion** (`wale_montecarlo/ingest.py`): auto-detects
  TradingView "List of trades" exports (two rows per trade, `Net P&L USD`),
  the native format, and generic broker exports with any P&L-like column.
  Handles BOM, currency symbols, parenthesized negatives, open trades.
- **Vectorized MC engine** (`wale_montecarlo/mc.py`): bootstrap + shuffle +
  ruin + stress in chunked numpy. 925 trades × 10K samples in ~1s (previously
  minutes in Python loops).
- **Shuffle luck detector**: percentile of the original ordering's max drawdown
  within the permutation distribution.
- **Composite verdict** with transparent pass/fail flags.
- `python -m wale_montecarlo serve` CLI command.
- 35 new tests (ingestion + vectorized engine): 117 total.

### Fixed
- Stress friction now scales to the strategy (bps of median trade notional, or
  fraction of avg |P&L|) instead of fixed futures-sized dollar slippage that
  misjudged small-notional strategies.
- Sharpe annualized by actual trade frequency instead of a blanket sqrt(252).
- CAGR computed from the real date span (was returning raw dollar total).
- Overfit verdict now uses medians across 2,000 seeds instead of a single seed.
- Vectorized engine cross-validated against the legacy loop implementation
  (quantiles agree within Monte Carlo noise).

## 2026-01-16

### Fixed: Worker hang on completion

**Problem:** The 200k grid runner would hang indefinitely when worker processes completed their metrics but crashed before writing `summary.json`. The main process waited forever on `as_completed(futures)` with no timeout.

**Solution:**
1. Added 10-minute timeout to `future.result()` calls
2. Added final sweep that detects orphaned cells (complete metrics, missing summary) and regenerates summaries from the raw data

**Files changed:**
- `CURSOR_run_surface_full_200k.py`

**Patch:**
```python
# Before (line 1265):
status_msg, is_complete, n_done = future.result()

# After:
status_msg, is_complete, n_done = future.result(timeout=600)
```

Plus added ~40 lines of orphan sweep logic after the main executor loop.
