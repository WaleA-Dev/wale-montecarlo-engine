# Wale Monte Carlo Engine - Implementation Plan

## Overview

This plan outlines the implementation of the Monte Carlo backtesting engine based on the existing comprehensive documentation. The engine stress-tests trading strategies by systematically perturbing backtest results across multiple dimensions.

---

## Current State

| Component | Status |
|-----------|--------|
| Documentation | Complete (README + 4 detailed docs) |
| Python Module | Empty (`__init__.py` only) |
| Core Engine | Not implemented |
| Runner Scripts | Not implemented |
| Analysis Scripts | Not implemented |
| Tests | Empty directory |

---

## Implementation Phases

### Phase 1: Core Data Structures & I/O

**Goal**: Establish the foundation for loading/saving data

#### Tasks:
1. **Create `models.py`** - Data classes for:
   - `Trade` (entry_time, exit_time, entry_price, exit_price, pnl, qty, side)
   - `EquityCurve` (time, equity)
   - `OHLCData` (time, open, high, low, close)
   - `CellConfig` (p_skip, slip_dollars, delay_bars_max, shuffle_mode, bootstrap_mode, block_len)
   - `PermutationResult` (perm_index, total_return_pct, max_drawdown_pct, profit_factor, worst_month_pct, etc.)
   - `CellSummary` (quantiles, p-values, statistics)

2. **Create `io.py`** - File I/O functions:
   - `load_trade_list(path) -> List[Trade]`
   - `load_equity_curve(path) -> EquityCurve`
   - `load_ohlc_data(path) -> OHLCData`
   - `load_baseline_report(path) -> BaselineMetrics`
   - `save_metrics_compact(path, results)` - atomic write with tmp+replace
   - `load_metrics_compact(path) -> List[PermutationResult]`
   - `save_cell_summary(path, summary)`
   - `save_run_manifest(path, config)`

3. **Create `seeding.py`** - Deterministic seeding:
   - `compute_cell_seed(cell_id: str) -> int` - SHA256-based
   - `compute_perm_seed(cell_seed: int, perm_index: int) -> int` - prime multiplier (1000003)

---

### Phase 2: Perturbation Models

**Goal**: Implement all 5 perturbation types as documented

#### Tasks:
1. **Create `perturbations/skip.py`** - Trade skipping:
   - `apply_skip(trades, p_skip, rng) -> List[Trade]`
   - Bernoulli probability per trade
   - Range: 0-10%

2. **Create `perturbations/slippage.py`** - Slippage costs:
   - `apply_slippage(trades, slip_dollars, rng) -> List[Trade]`
   - Uniform random $0 to slip_dollars per trade
   - Subtracted from PnL (always hurts)

3. **Create `perturbations/delay.py`** - Execution delay:
   - `apply_delay(trades, delay_bars_max, ohlc_data, rng) -> List[Trade]`
   - Random delay 0 to delay_bars_max bars
   - OHLC-based: use actual open price at delayed bar
   - Fallback: approximate mode without OHLC
   - Cap adverse impact at 0.5R per trade

4. **Create `perturbations/shuffle.py`** - Sequence shuffling:
   - `apply_shuffle(trades, mode, block_len, rng) -> List[Trade]`
   - Modes: `none`, `permute`, `block_permute`
   - Block permute: shuffle blocks of `block_len` trades

5. **Create `perturbations/bootstrap.py`** - Bootstrap resampling:
   - `apply_bootstrap(trades, mode, block_len, rng) -> List[Trade]`
   - Modes: `none`, `trade_bootstrap`, `block_bootstrap`
   - Resample with replacement

6. **Create `perturbations/state_dependent.py`** - State-aware multipliers:
   - `compute_volatility_multiplier(trades, window=20)`
   - `compute_drawdown_multiplier(equity_curve)`
   - Applied to slippage and delay during volatile/drawdown periods

7. **Create `perturbations/pipeline.py`** - Compose all perturbations:
   - `apply_all_perturbations(trades, config, ohlc, rng) -> List[Trade]`
   - Order: skip -> slippage -> delay -> shuffle -> bootstrap

---

### Phase 3: Metrics Calculation

**Goal**: Compute performance metrics from perturbed trades

#### Tasks:
1. **Create `metrics.py`**:
   - `compute_equity_curve(trades) -> EquityCurve`
   - `compute_total_return_pct(equity_curve) -> float`
   - `compute_max_drawdown_pct(equity_curve) -> float`
   - `compute_profit_factor(trades) -> float`
   - `compute_worst_month_pct(equity_curve) -> float`
   - `compute_sharpe_ratio(equity_curve) -> float`
   - `compute_win_rate(trades) -> float`
   - `compute_all_metrics(trades) -> PermutationResult`

---

### Phase 4: Grid Search Engine

**Goal**: Implement the parallel grid search system

#### Tasks:
1. **Create `grid.py`** - Grid generation:
   - `generate_grid(config) -> List[CellConfig]`
   - Default ranges:
     - p_skip: [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
     - slip: [0, 25, 50, 75, 100, 150, 200, 300]
     - delay: [0, 1, 2, 3] (often fixed to 1)
     - shuffle: ["none", "permute", "block_permute"]
     - bootstrap: ["none", "trade_bootstrap", "block_bootstrap"]
     - block_len: [5, 10, 20]
   - `filter_grid(cells, filters) -> List[CellConfig]`
   - `cell_to_id(config) -> str` - unique identifier

2. **Create `worker.py`** - Single cell worker:
   - `run_cell(cell_config, trades, ohlc, n_perms, start_perm=0) -> List[PermutationResult]`
   - Resume support: start from `max(existing_perm_index) + 1`
   - Atomic writes every N permutations (checkpointing)
   - Timeout handling to prevent hangs

3. **Create `runner.py`** - Parallel orchestrator:
   - `run_surface(config, n_jobs) -> RunResult`
   - Multiprocessing pool with `n_jobs` workers
   - Progress tracking with heartbeat (every 30s)
   - Resume logic: read `metrics_compact.csv`, dedupe, continue
   - Create output directory structure

---

### Phase 5: Statistical Analysis

**Goal**: Implement analysis methods from docs

#### Tasks:
1. **Create `analysis/quantiles.py`**:
   - `compute_quantiles(results, qs=[0.05, 0.50, 0.95]) -> Dict`
   - For each metric: total_return, max_drawdown, profit_factor, worst_month

2. **Create `analysis/pvalue.py`**:
   - `compute_pvalue(results, baseline_pf) -> float`
   - Fraction of perms with PF >= baseline
   - `apply_bonferroni(pvalue, n_tests) -> float`

3. **Create `analysis/robust_score.py`**:
   - `compute_robust_score(pf_p50, p_corrected) -> float`
   - Formula: `PF_P50 * (1 - p_corrected)`

4. **Create `analysis/pareto.py`**:
   - `find_pareto_front_2d(cells, metric1, metric2) -> List[CellConfig]`
   - `find_pareto_front_3d(cells, m1, m2, m3) -> List[CellConfig]`
   - Maximize PF, minimize MaxDD, maximize Return

5. **Create `analysis/clustering.py`**:
   - `find_plateau_clusters(cells, tolerance=0.10) -> List[Cluster]`
   - Group cells with similar robust scores
   - Large clusters = robust parameter regions

6. **Create `analysis/report.py`**:
   - `generate_decision_report(run_dir) -> str`
   - Markdown format with tables and recommendations

---

### Phase 6: CLI Scripts

**Goal**: Create user-facing command-line tools

#### Tasks:
1. **Create `CURSOR_run_surface_full_200k.py`**:
   ```
   Arguments:
   --repo: Path to backtest export
   --n_per_cell: Permutations per cell (default: 200000)
   --jobs: Parallel workers (default: CPU count)
   --fixed_delay: Fix delay parameter (optional)
   --run_name: Name for this run
   --status_only: Just show progress, don't run
   --resume: Continue previous run
   ```

2. **Create `CURSOR_surface_full_200k_analysis.py`**:
   ```
   Arguments:
   --run_dir: Path to completed run
   --output: Output directory for analysis
   --top_n: Number of top cells to report (default: 50)
   ```

3. **Create `CURSOR_single_cell_deep_dive.py`**:
   ```
   Arguments:
   --cell_dir: Path to single cell results
   --visualize: Generate plots
   ```

---

### Phase 7: Testing

**Goal**: Ensure correctness and reliability

#### Tasks:
1. **Unit Tests** (`tests/`):
   - `test_perturbations.py` - Each perturbation in isolation
   - `test_metrics.py` - Metric calculations
   - `test_seeding.py` - Determinism verification
   - `test_io.py` - File read/write roundtrips
   - `test_resume.py` - Resume correctness (no duplicates)

2. **Integration Tests**:
   - `test_single_cell.py` - Full cell run with small N
   - `test_grid_small.py` - Small grid (3x3) verification

3. **Fixtures**:
   - Create sample `trade_list.csv` (100 trades)
   - Create sample `equity_curve.csv`
   - Create sample OHLC data

---

### Phase 8: Package Setup

**Goal**: Make installable and distributable

#### Tasks:
1. **Create `pyproject.toml`**:
   - Dependencies: numpy, pandas, tqdm, multiprocessing
   - Entry points for CLI scripts
   - Python >= 3.9

2. **Create `requirements.txt`**:
   - numpy>=1.21
   - pandas>=1.3
   - tqdm>=4.62
   - scipy>=1.7 (for statistics)

3. **Update `wale_montecarlo/__init__.py`**:
   - Export main classes and functions
   - Version number

---

## File Structure (Final)

```
wale-montecarlo-engine/
├── README.md
├── CHANGELOG.md
├── LICENSE
├── IMPLEMENTATION_PLAN.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── wale_montecarlo/
│   ├── __init__.py
│   ├── models.py
│   ├── io.py
│   ├── seeding.py
│   ├── metrics.py
│   ├── grid.py
│   ├── worker.py
│   ├── runner.py
│   │
│   ├── perturbations/
│   │   ├── __init__.py
│   │   ├── skip.py
│   │   ├── slippage.py
│   │   ├── delay.py
│   │   ├── shuffle.py
│   │   ├── bootstrap.py
│   │   ├── state_dependent.py
│   │   └── pipeline.py
│   │
│   └── analysis/
│       ├── __init__.py
│       ├── quantiles.py
│       ├── pvalue.py
│       ├── robust_score.py
│       ├── pareto.py
│       ├── clustering.py
│       └── report.py
│
├── scripts/
│   ├── CURSOR_run_surface_full_200k.py
│   ├── CURSOR_surface_full_200k_analysis.py
│   └── CURSOR_single_cell_deep_dive.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py (fixtures)
│   ├── test_perturbations.py
│   ├── test_metrics.py
│   ├── test_seeding.py
│   ├── test_io.py
│   ├── test_resume.py
│   ├── test_single_cell.py
│   └── test_grid_small.py
│
└── docs/
    ├── 01-perturbation-models-deep-dive.md
    ├── 02-resume-correctness-technical.md
    ├── 03-statistical-analysis.md
    └── 04-usage-examples.md
```

---

## Implementation Order

| Step | Phase | Priority | Est. Complexity |
|------|-------|----------|-----------------|
| 1 | Phase 1: Models & I/O | Critical | Medium |
| 2 | Phase 2: Perturbations | Critical | High |
| 3 | Phase 3: Metrics | Critical | Medium |
| 4 | Phase 4: Grid Engine | Critical | High |
| 5 | Phase 6: CLI (basic) | High | Medium |
| 6 | Phase 7: Tests (core) | High | Medium |
| 7 | Phase 5: Analysis | Medium | Medium |
| 8 | Phase 6: CLI (full) | Medium | Low |
| 9 | Phase 8: Package | Low | Low |
| 10 | Phase 7: Tests (full) | Low | Medium |

---

## Key Implementation Notes

### Resume Correctness (Critical)
Per `02-resume-correctness-technical.md`:
- `metrics_compact.csv` is the **SOURCE OF TRUTH**
- On resume: dedupe by `perm_index`, keep first occurrence
- Resume at `max(perm_index) + 1`
- Use atomic writes: write to `.tmp`, then `os.replace()`

### Seeding Scheme (Critical)
```python
import hashlib

def compute_cell_seed(cell_id: str) -> int:
    return int(hashlib.sha256(cell_id.encode()).hexdigest()[:8], 16)

def compute_perm_seed(cell_seed: int, perm_index: int) -> int:
    return (cell_seed + perm_index * 1000003) % (2**32)
```

### Perturbation Order
1. Skip (removes trades)
2. Slippage (modifies PnL)
3. Delay (modifies entry price/PnL)
4. Shuffle (reorders trades)
5. Bootstrap (resamples trades)

### Performance Targets
- 200,000 permutations per cell
- ~1,500 cells (with delay=1 fixed)
- 300 million total simulations
- Target: 24-48 hours on 8-core CPU

---

## Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
tqdm>=4.62.0
scipy>=1.7.0
```

---

## Success Criteria

1. **Correctness**: Seeding produces deterministic results
2. **Resume Safety**: No duplicate permutations after crash/resume
3. **Performance**: Can complete 200K perms/cell in reasonable time
4. **Analysis**: Generates decision report with robust scores, Pareto fronts, clusters
5. **Usability**: CLI scripts work as documented in `04-usage-examples.md`

---

## Next Steps

Begin with **Phase 1: Core Data Structures & I/O**, implementing:
1. `models.py` - All data classes
2. `io.py` - File loading/saving with atomic writes
3. `seeding.py` - Deterministic seed generation

Then proceed to **Phase 2: Perturbations** which is the heart of the engine.
