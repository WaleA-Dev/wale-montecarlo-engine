# Changelog

## 2026-07-03 - v2.0.1: honesty and hardening pass

- Verdicts now refuse to overpromise: under 10 trades returns "Insufficient
  Data" instead of a grade, and under 30 trades caps at "Moderate" no matter
  how clean the numbers look. A single winning trade used to rate "Robust",
  which was embarrassing.
- Data-quality warnings (like excluded open trades) now appear as flags on
  the verdict plate instead of fine print at the bottom of the page.
- Memory use is bounded for large files: a 20,000-trade list peaked at
  2.3 GB before, 0.26 GB now. Chunk sizes scale with trade count across
  bootstrap, shuffle, and stress simulations.
- Verified the full desktop flow by driving the built exe with real mouse
  input: sample analysis, dashboard render, and report download through the
  native save dialog.
- Removed the stale IMPLEMENTATION_PLAN.md and cleaned up typography across
  the repo.
- README now has an "Assumptions and Limitations" section that says plainly
  what this tool can and cannot tell you.

## 2026-07-03 - v2.0: Strategy Stress Lab

### Added
- **Desktop app**: `WaleMonteCarlo.exe` (PyInstaller one-file, ~27 MB) - double-click
  and a native application window opens (WebView2 via pywebview, custom icon,
  no console, no browser). Falls back to the default browser if WebView2 is
  unavailable. Built from `WaleMonteCarlo.spec`.
- **Web UI** (`wale_montecarlo/webapp/`): dark dashboard with verdict plate,
  equity cone, P&L / drawdown distributions, ruin ladder, stress-scenario table,
  and one-click standalone HTML report export. Chart.js bundled locally - fully offline.
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

### Validated
- `scripts/independent_validation.py`: zero-dependency audit (pure stdlib, no
  numpy, no shared code) that drives the live app over HTTP and checks it
  against (a) TradingView's own Cumulative P&L column, (b) exact brute-force
  enumeration of all 6^6 bootstrap draws and all 7! orderings, (c) closed-form
  expectations with 4-sigma bands, (d) determinism and cross-seed convergence.
  23/23 checks pass against the shipped exe.

### Fixed
- Open positions in TradingView exports (Exit rows with Signal=Open carrying
  unrealized mark-to-market P&L) are now excluded with an explicit warning --
  previously a still-open trade could dominate every statistic (caught by the
  independent audit: one open SCHD trade carried $110k unrealized).
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
