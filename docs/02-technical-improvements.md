# Technical Improvements & Performance Optimization

> Addressing statistical rigor, convergence analysis, perturbation interactions, **overfitting detection**, and the elephant in the room: why your simulation is crawling.

---

## Table of Contents

1. [The Performance Problem (Read This First)](#1-the-performance-problem-read-this-first)
2. [Robust Score: Why This Formula](#2-robust-score-why-this-formula)
3. [Multiple Testing Correction: Bonferroni vs FDR](#3-multiple-testing-correction-bonferroni-vs-fdr)
4. [Convergence Analysis: How Many Permutations](#4-convergence-analysis-how-many-permutations)
5. [Perturbation Interactions: The Full Model](#5-perturbation-interactions-the-full-model)
6. [Overfitting Detection: The Core Purpose](#6-overfitting-detection-the-core-purpose)
7. [Validation: Synthetic Ground Truth Tests](#7-validation-synthetic-ground-truth-tests)

---

## 1. The Performance Problem (Read This First)

### Current State: The Math Doesn't Add Up

You've been running for **60,000 seconds (~16.7 hours)** and completed **259 cells**.

Let's do the math:
- Time per cell: 60,000 / 259 ≈ **232 seconds per cell**
- Total cells (default grid): ~1,500 (with delay fixed) or ~6,048 (full)
- Estimated total time at current rate: 1,500 × 232 = **348,000 seconds ≈ 4 days**

That's... not great. But it's also not catastrophically slow for 200K permutations per cell. The real question is: **are you getting 200K permutations per cell, or dying earlier?**

### Diagnosis: What's Actually Happening

Check your cell progress:

```bash
# Look at a completed cell
type montecarlo_output\mc_run_XXXXXX\per_cell\cell_0_0_0_0_0\progress.json
```

If `n_done` is much less than `n_target`, you have a different problem (crashes, memory issues).

If `n_done` equals `n_target`, the bottleneck is raw computation.

### The Real Bottleneck: Permutation Count × Cell Count

| Configuration | Cells | Perms/Cell | Total Sims | Est. Time (8 cores) |
|--------------|-------|------------|------------|---------------------|
| Current (full grid, 200K) | 1,500 | 200,000 | 300M | 4-5 days |
| Reduced perms (50K) | 1,500 | 50,000 | 75M | 1 day |
| Reduced grid + perms | 400 | 50,000 | 20M | 6-8 hours |
| Minimal viable | 100 | 20,000 | 2M | 30-60 min |

### Solution 1: Reduce Grid Dimensionality (Recommended)

The 6,048-cell grid is overkill for initial exploration. Most of the information is in a much smaller subspace.

**Proposed "Exploration Grid":**

```python
exploration_grid = {
    "p_skip":    [0.00, 0.02, 0.05, 0.10],     # 4 values (was 7)
    "slip":      [0, 50, 100, 200],             # 4 values (was 8)
    "delay":     [0, 1],                        # 2 values (was 4)
    "shuffle":   ["none", "permute"],           # 2 values (was 3)
    "bootstrap": ["none", "trade_bootstrap"],   # 2 values (was 3)
    "block_len": [10],                          # 1 value (was 3)
}
# Total: 4 × 4 × 2 × 2 × 2 × 1 = 128 cells
```

Run this first with 50K permutations:
- 128 cells × 50,000 = **6.4M simulations**
- Estimated time: **2-4 hours**

Then, if you find interesting regions, run a focused "deep dive" grid around those parameters.

### Solution 2: Smarter Parallelization

Your current setup uses 8 workers. But are they efficient?

**Check CPU utilization:**
```powershell
# PowerShell: watch CPU usage
while ($true) { Get-Process python | Select-Object CPU; Start-Sleep 2 }
```

**Common issues:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| 8 cores but <50% total CPU | GIL contention | Use `multiprocessing` not `threading` |
| High CPU but slow progress | Small batch sizes | Increase batch size per worker |
| Memory growing over time | Not releasing results | Write to disk incrementally |
| One core at 100%, others idle | Sequential bottleneck | Check for synchronization locks |

**Recommended worker configuration:**

```python
# In run_simulation.py or config
WORKER_CONFIG = {
    "n_workers": min(cpu_count() - 1, 12),  # Leave 1 core for OS
    "batch_size": 1000,                      # Perms per batch before sync
    "write_interval": 10000,                 # Write to disk every N perms
    "memory_limit_gb": 0.5,                  # Per-worker memory cap
}
```

### Solution 3: Tiered Execution Strategy

**Solving the time problem may require one solution or a combination of all of them. Read the above solutions carefully and make an informed decision based on your specific constraints (available time, hardware, precision requirements).**

**Phase 1: Scout Run (2-4 hours)**
- 128-cell exploration grid
- 20,000 permutations per cell
- Goal: Identify which parameter regions matter

**Phase 2: Focused Run (4-8 hours)**
- 200-400 cells around interesting regions
- 100,000 permutations per cell
- Goal: Get reliable estimates for decision-making

**Phase 3: Publication Run (optional, 24-48 hours)**
- Full grid
- 200,000 permutations per cell
- Goal: Comprehensive surface for documentation

Most users only need Phase 1 + Phase 2. Phase 3 is for when you're writing up results or need extreme tail precision.

### Implementation: New CLI Flags

Add these to `run_simulation.py`:

```bash
# Quick exploration (recommended starting point)
python scripts/run_simulation.py --trades trades.csv --mode explore

# Focused analysis (after reviewing exploration results)
python scripts/run_simulation.py --trades trades.csv --mode focus \
    --p_skip_range 0.02,0.08 --slip_range 50,150

# Full publication run
python scripts/run_simulation.py --trades trades.csv --mode full --n_per_cell 200000
```

---

## 2. Robust Score: Why This Formula

### The Original Formula

```
Robust Score = PF_P50 × (1 - P_value_corrected)
```

### Honest Assessment: The Formula Is Somewhat Arbitrary

The robust score formula was chosen for practical reasons, not derived from first principles. This section documents the reasoning and acknowledges limitations.

### Why Multiplication (Not Addition)?

**Multiplicative form enforces joint requirements:**

| PF_P50 | P-value | Additive (w=0.5) | Multiplicative |
|--------|---------|------------------|----------------|
| 3.0 | 0.01 | 0.5×3 + 0.5×0.99 = 2.0 | 3.0 × 0.99 = 2.97 |
| 3.0 | 0.50 | 0.5×3 + 0.5×0.50 = 1.75 | 3.0 × 0.50 = 1.50 |
| 1.0 | 0.01 | 0.5×1 + 0.5×0.99 = 1.0 | 1.0 × 0.99 = 0.99 |
| 0.5 | 0.01 | 0.5×0.5 + 0.5×0.99 = 0.75 | 0.5 × 0.99 = 0.50 |

With multiplication:
- **Bad performance (PF < 1) gets penalized regardless of significance**
- **Good performance but high p-value gets heavily penalized**
- You need BOTH to score well

With addition, a mediocre profit factor can be offset by a strong p-value, producing a high score for a strategy that isn't actually profitable. This fundamentally misrepresents what "robust" should mean.

### Deep Analysis: Scoring Function Alternatives

The choice of scoring function determines what the engine optimizes for. Here's a rigorous comparison of the options:

#### Option 1: Multiplicative (Current)

```
Score = PF_P50 × (1 - p_value)
```

| PF_P50 | p-value | Score | Interpretation |
|--------|---------|-------|----------------|
| 3.0 | 0.01 | 2.97 | Excellent: high performance, highly significant |
| 3.0 | 0.50 | 1.50 | Mediocre: high performance, not significant |
| 1.0 | 0.01 | 0.99 | Poor: breakeven, but "significant" breakeven |
| 0.8 | 0.01 | 0.79 | Bad: losing strategy, significance irrelevant |

**Properties:**
- Zero p-value → Score = PF (pure performance)
- PF = 1.0 → Score ≤ 1.0 (breakeven strategies capped)
- PF < 1.0 → Score < 1.0 (losing strategies always score poorly)

**Failure mode:** A strategy with PF=1.5 and p=0.01 scores 1.485, while PF=2.5 and p=0.40 scores 1.50. The latter is arguably better (higher actual performance) but scores the same.

#### Option 2: Additive (Weighted Sum)

```
Score = w₁ × PF_P50 + w₂ × (1 - p_value)
```

With w₁ = w₂ = 0.5:

| PF_P50 | p-value | Score | Problem |
|--------|---------|-------|---------|
| 3.0 | 0.01 | 2.00 | - |
| 3.0 | 0.50 | 1.75 | - |
| 1.0 | 0.01 | 1.00 | Breakeven scores same as PF=2.0, p=0.50 |
| 0.5 | 0.01 | 0.75 | **Losing strategy scores > 0** |

**Fatal flaw:** Losing strategies (PF < 1.0) can achieve positive scores if p-value is low enough. This is nonsensical-statistical significance of a losing strategy is not valuable.

#### Option 3: Gated Multiplicative (Recommended)

```
Score = max(0, PF_P50 - 1.0) × (1 - p_value)
```

This subtracts 1.0 from PF before multiplying, so only the **excess return** is scored:

| PF_P50 | p-value | Score | Interpretation |
|--------|---------|-------|----------------|
| 3.0 | 0.01 | 1.98 | Strong edge, highly significant |
| 3.0 | 0.50 | 1.00 | Strong edge, not significant |
| 1.5 | 0.01 | 0.50 | Moderate edge, highly significant |
| 1.0 | 0.01 | 0.00 | No edge-score is zero regardless of significance |
| 0.8 | 0.01 | 0.00 | Losing-score is zero |

**Properties:**
- Breakeven (PF=1.0) always scores zero
- Losing strategies (PF<1.0) always score zero
- Only excess returns above breakeven contribute to score
- Significance scales the excess return

**This is the mathematically correct formulation.** Statistical significance of "no edge" or "negative edge" should contribute nothing.

#### Option 4: Threshold-Gated

```
Score = PF_P50 × (1 - p_value)  if p_value < α and PF_P50 > min_pf
Score = 0                        otherwise
```

With α = 0.05 and min_pf = 1.2:

| PF_P50 | p-value | Score | Interpretation |
|--------|---------|-------|----------------|
| 3.0 | 0.01 | 2.97 | Passes both gates |
| 3.0 | 0.10 | 0.00 | Fails significance gate |
| 1.1 | 0.01 | 0.00 | Fails minimum PF gate |
| 0.8 | 0.01 | 0.00 | Fails minimum PF gate |

**Properties:**
- Binary gates eliminate marginal strategies
- No partial credit for "almost significant" or "almost profitable"
- Clear decision boundaries

**Drawback:** Loses ranking granularity. All strategies below threshold score zero, making it hard to compare "bad" strategies.

#### Option 5: Geometric Mean

```
Score = √(PF_P50 × (1 - p_value))
```

| PF_P50 | p-value | Score | Interpretation |
|--------|---------|-------|----------------|
| 3.0 | 0.01 | 1.72 | - |
| 3.0 | 0.50 | 1.22 | - |
| 1.0 | 0.01 | 1.00 | Breakeven scores 1.0 |
| 0.5 | 0.01 | 0.70 | **Losing strategy scores > 0** |

**Same fatal flaw as additive:** Losing strategies score above zero.

### Recommendation: Gated Multiplicative with Drawdown Penalty

```python
def robust_score_v3(pf_p50, p_value_corrected, maxdd_p95):
    """
    Final recommended scoring function.
    
    Components:
    1. Excess return: (PF - 1.0) - only profit above breakeven counts
    2. Significance: (1 - p_value) - statistical confidence
    3. Drawdown penalty: penalize high tail risk
    
    Properties:
    - PF ≤ 1.0 → Score = 0 (no edge = no score)
    - High p-value → Score reduced proportionally
    - High drawdown → Score reduced or zeroed
    """
    # Component 1: Excess return (gated at zero)
    excess_return = max(0, pf_p50 - 1.0)
    
    # Component 2: Statistical significance
    significance = 1 - p_value_corrected
    
    # Component 3: Drawdown penalty
    # Linear penalty: 1.0 at 20% DD, 0.0 at 60% DD
    dd_penalty = np.clip((0.60 - maxdd_p95) / 0.40, 0, 1)
    
    # Final score
    return excess_return * significance * dd_penalty
```

**Score interpretation:**

| Score Range | Meaning |
|-------------|---------|
| > 1.5 | Excellent: strong edge, significant, controlled risk |
| 1.0 - 1.5 | Good: meaningful edge worth trading |
| 0.5 - 1.0 | Marginal: edge exists but modest |
| 0.1 - 0.5 | Weak: barely profitable or high uncertainty |
| < 0.1 | Reject: no tradeable edge |

### Why Gated Multiplicative Is Correct

The fundamental insight is that **profit factor is a ratio, not a difference**. A PF of 1.0 means "break even"-there is no edge. Statistical significance of "no edge" is meaningless.

By subtracting 1.0 before multiplying:
- We measure **edge magnitude** (excess return over breakeven)
- We scale edge by **confidence** (significance)
- We penalize **tail risk** (drawdown)

This produces a score that answers: **"How much confident, risk-adjusted edge does this strategy have?"**

### Why P50 (Not P05 or Mean)?

| Percentile | What It Measures | Problem |
|------------|------------------|---------|
| Mean | Average outcome | Sensitive to outliers; one amazing run skews everything |
| P05 | Worst 5% of outcomes | Too pessimistic; optimizing for disaster |
| P50 | Median outcome | "What typically happens"; robust to outliers |
| P95 | Best 5% of outcomes | Too optimistic; chasing upside |

**P50 is the "honest expectation"** - if you ran this strategy many times, P50 is roughly what you'd typically see.

For risk management, P05 (tail risk) should also be examined, but for ranking parameter combinations, P50 is the correct central tendency measure.

### Alternative: Multi-Objective Scoring (Pareto Ranking)

If collapsing to a single score feels reductive, use **Pareto ranking** instead:

```python
def pareto_rank(cells):
    """
    Rank cells by Pareto dominance.
    A cell is dominated if another cell beats it on ALL objectives.
    """
    objectives = ['pf_p50', 'neg_maxdd_p95', 'sharpe_p50']  # maximize all
    
    ranks = []
    for cell in cells:
        n_dominating = sum(
            1 for other in cells 
            if all(other[obj] >= cell[obj] for obj in objectives)
            and any(other[obj] > cell[obj] for obj in objectives)
        )
        ranks.append(n_dominating)
    
    return ranks  # 0 = Pareto-optimal, higher = more dominated
```

This preserves the full trade-off structure without forcing a single ranking.

---

## 3. Multiple Testing Correction: Bonferroni vs FDR

### The Problem

With 6,048 cells, Bonferroni correction multiplies every p-value by 6,048. This means:
- A raw p-value of 0.001 becomes corrected p-value of 6.048 (capped at 1.0)
- **Nothing is significant** unless raw p-value < 0.05/6048 ≈ 0.0000083

This is almost certainly too conservative for our use case.

### Why Bonferroni Is Wrong Here

Bonferroni controls the **Family-Wise Error Rate (FWER)**: the probability of making *any* false positive.

But we don't care about avoiding all false positives. We care about:
1. Finding the best parameter regions (ranking)
2. Having some confidence that good results aren't pure luck

For this, we want to control the **False Discovery Rate (FDR)**: the *proportion* of our discoveries that are false.

### Benjamini-Hochberg FDR Control

```python
def benjamini_hochberg(p_values, alpha=0.05):
    """
    Adjust p-values using Benjamini-Hochberg procedure.
    Controls FDR at level alpha.
    
    Returns adjusted p-values (q-values).
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]
    
    # BH adjustment
    adjusted = np.zeros(n)
    for i, idx in enumerate(sorted_indices):
        rank = i + 1
        adjusted[idx] = min(1.0, sorted_pvals[i] * n / rank)
    
    # Enforce monotonicity (larger raw p → larger adjusted p)
    for i in range(n - 2, -1, -1):
        adjusted[sorted_indices[i]] = min(
            adjusted[sorted_indices[i]], 
            adjusted[sorted_indices[i + 1]]
        )
    
    return adjusted
```

### Comparison

| Method | What It Controls | 6,048 tests, raw p=0.001 | Use Case |
|--------|------------------|--------------------------|----------|
| None | Nothing | 0.001 | Never |
| Bonferroni | FWER | 1.0 (capped) | Medical trials |
| Holm | FWER (less conservative) | ~0.8 | Critical decisions |
| BH | FDR | ~0.006 | Exploratory analysis |
| BY | FDR (conservative) | ~0.05 | Dependent tests |

### Recommendation: Use BH with α=0.10

```python
# In analysis code
from scipy.stats import false_discovery_control

def compute_adjusted_pvalues(raw_pvalues, method='bh'):
    """
    Compute adjusted p-values using specified method.
    
    Methods:
    - 'bonferroni': Family-wise error rate (very conservative)
    - 'bh': Benjamini-Hochberg FDR (recommended)
    - 'by': Benjamini-Yekutieli FDR (for dependent tests)
    """
    if method == 'bonferroni':
        return np.minimum(raw_pvalues * len(raw_pvalues), 1.0)
    elif method == 'bh':
        return false_discovery_control(raw_pvalues, method='bh')
    elif method == 'by':
        return false_discovery_control(raw_pvalues, method='by')
```

### Updated Robust Score with BH Correction

```python
def robust_score_v3(pf_p50, raw_p_value, maxdd_p95, all_raw_p_values):
    """
    Robust score using BH-corrected p-values.
    """
    # Get BH-adjusted p-value for this cell
    adjusted_p = benjamini_hochberg_single(raw_p_value, all_raw_p_values)
    
    # Rest of scoring
    perf = pf_p50 * (1 - adjusted_p)
    dd_penalty = np.clip(1 - (maxdd_p95 - 0.20) / 0.40, 0, 1)
    
    return perf * dd_penalty
```

---

## 4. Convergence Analysis: How Many Permutations

### The Claim Under Scrutiny

> "Statistical estimates stabilize around 100K-200K samples"

This is hand-wavy. Let's be precise.

### Convergence Rates by Metric

| Metric | Convergence Rate | 10K Error | 50K Error | 200K Error |
|--------|------------------|-----------|-----------|------------|
| Mean | O(1/√n) | ±3.2% | ±1.4% | ±0.7% |
| Median (P50) | O(1/√n) | ±4.0% | ±1.8% | ±0.9% |
| P05 | O(1/√n) × √(0.05×0.95) | ±6.8% | ±3.0% | ±1.5% |
| P95 | O(1/√n) × √(0.05×0.95) | ±6.8% | ±3.0% | ±1.5% |
| P99 | O(1/√n) × √(0.01×0.99) | ±10.0% | ±4.5% | ±2.2% |
| Max | O(log(n)/n) | Very noisy | Still noisy | Somewhat stable |

**Key insight:** Tail percentiles (P05, P95, P99) converge **slower** than the median because there's less data in the tails.

### Empirical Convergence Test

Run this on your actual data to validate:

```python
def convergence_analysis(cell_results, metric_fn, checkpoints=[1000, 5000, 10000, 25000, 50000, 100000, 200000]):
    """
    Compute metric at each checkpoint and measure stability.
    """
    final_value = metric_fn(cell_results)
    
    convergence = []
    for n in checkpoints:
        if n > len(cell_results):
            break
        subset_value = metric_fn(cell_results[:n])
        pct_error = abs(subset_value - final_value) / abs(final_value) * 100
        convergence.append({
            'n': n,
            'value': subset_value,
            'pct_error_vs_final': pct_error
        })
    
    return convergence

# Example usage
results = load_cell_results('cell_0_0_0_0_0')
pf_values = [r['profit_factor'] for r in results]

print("P50 Convergence:")
for c in convergence_analysis(pf_values, lambda x: np.percentile(x, 50)):
    print(f"  n={c['n']:>7}: {c['value']:.3f} (error: {c['pct_error_vs_final']:.2f}%)")

print("\nP95 Convergence:")
for c in convergence_analysis(pf_values, lambda x: np.percentile(x, 95)):
    print(f"  n={c['n']:>7}: {c['value']:.3f} (error: {c['pct_error_vs_final']:.2f}%)")
```

### Recommended Permutation Counts

| Goal | Metric Focus | Recommended N | Rationale |
|------|--------------|---------------|-----------|
| Quick exploration | P50 only | 10,000 - 20,000 | <2% error on median |
| Standard analysis | P50, P05, P95 | 50,000 - 100,000 | <2% error on P05/P95 |
| Publication quality | All percentiles + p-values | 200,000 | <1% error everywhere |
| Extreme tail (P99) | Tail risk analysis | 500,000+ | Still ~1.5% error |

### Confidence Intervals for Percentiles

For formal analysis, compute confidence intervals:

```python
def percentile_confidence_interval(data, percentile, confidence=0.95):
    """
    Bootstrap confidence interval for a percentile.
    """
    n_bootstrap = 1000
    bootstrap_estimates = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_estimates.append(np.percentile(sample, percentile))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
    
    return lower, upper

# Example
pf_values = [...]  # Your 200K profit factor values
p50 = np.percentile(pf_values, 50)
ci_lower, ci_upper = percentile_confidence_interval(pf_values, 50)
print(f"PF P50 = {p50:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
```

---

## 5. Perturbation Interactions: The Full Model

### The Concern

> "You apply skip, slippage, delay, shuffle, and bootstrap together. But are these effects additive? Multiplicative? Do they have interaction terms?"

This is a legitimate concern. Let's formalize it.

### The Implicit Model

Currently, the engine applies perturbations sequentially:

```python
def perturb_trades(trades, p_skip, slip_max, delay_max, shuffle_mode, bootstrap_mode):
    # Step 1: Bootstrap (resampling)
    trades = apply_bootstrap(trades, bootstrap_mode)
    
    # Step 2: Shuffle (reordering)
    trades = apply_shuffle(trades, shuffle_mode)
    
    # Step 3: Skip (random deletion)
    trades = apply_skip(trades, p_skip)
    
    # Step 4: Slippage (cost deduction)
    trades = apply_slippage(trades, slip_max)
    
    # Step 5: Delay (price adjustment)
    trades = apply_delay(trades, delay_max)
    
    return trades
```

### Do These Interact?

**Yes, but in predictable ways:**

| Interaction | Type | Effect |
|-------------|------|--------|
| skip × slippage | Independent | Slippage only applies to executed trades |
| skip × delay | Independent | Delay only applies to executed trades |
| slippage × delay | Additive-ish | Both reduce PnL; no interaction term |
| shuffle × bootstrap | Compositional | Bootstrap changes the trade set, then shuffle reorders |
| bootstrap × skip | Order matters | Bootstrap first → different trades get skipped |

### The Correct Interaction Model

For the metrics we care about (total return, max drawdown, profit factor), the effects are roughly **multiplicative** on returns and **path-dependent** on drawdown.

**Return decomposition:**

```
Final_Return ≈ Baseline_Return 
             × (1 - p_skip)           # Fewer trades
             × (1 - slippage_impact)  # Slippage cost
             × (1 - delay_impact)     # Delay cost
             × shuffle_factor         # Usually ~1 (changes variance, not mean)
             × bootstrap_factor       # Usually ~1 (resampling)
```

**Drawdown is harder:** It's path-dependent, so shuffle and bootstrap can have large effects even when they don't change the mean.

### Quantifying Interactions: ANOVA Decomposition

To actually measure interaction effects, run an ANOVA-style analysis:

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

def analyze_interactions(grid_summary_df):
    """
    Fit a linear model with interaction terms to understand
    how perturbations combine.
    """
    # Create dummy variables for categorical factors
    df = grid_summary_df.copy()
    df['shuffle_permute'] = (df['shuffle_mode'] == 'permute').astype(int)
    df['bootstrap_trade'] = (df['bootstrap_mode'] == 'trade_bootstrap').astype(int)
    
    # Fit model with main effects and 2-way interactions
    formula = """
        pf_p50 ~ p_skip + slip + delay 
               + shuffle_permute + bootstrap_trade
               + p_skip:slip + p_skip:delay + slip:delay
               + p_skip:shuffle_permute + slip:shuffle_permute
    """
    
    model = ols(formula, data=df).fit()
    
    print(model.summary())
    
    # Extract interaction significance
    interactions = {
        name: {'coef': model.params[name], 'pval': model.pvalues[name]}
        for name in model.params.index if ':' in name
    }
    
    return interactions

# If interactions are significant (p < 0.05), they matter.
# If not, you can treat perturbations as approximately independent.
```

### What This Means for Users

**If interactions are small (<10% of main effects):**
- You can interpret parameters independently
- The grid search is valid as-is

**If interactions are large:**
- Report them in the analysis
- Consider interaction-aware scoring:

```python
def interaction_adjusted_score(cell_params, interaction_model):
    """
    Adjust predicted performance for interaction effects.
    """
    main_effect = (
        baseline_pf 
        - cell_params['p_skip'] * skip_coef
        - cell_params['slip'] * slip_coef
        - cell_params['delay'] * delay_coef
    )
    
    interaction_effect = (
        cell_params['p_skip'] * cell_params['slip'] * skip_slip_interaction
        + cell_params['p_skip'] * cell_params['delay'] * skip_delay_interaction
        # ... etc
    )
    
    return main_effect + interaction_effect
```

---

## 6. Overfitting Detection: The Core Purpose

### Why This Engine Exists

The entire point of this Monte Carlo framework is to answer one question: **Is this strategy overfit?**

A strategy is overfit when its backtest performance is primarily explained by:
- Lucky trade sequencing
- Curve-fitting to specific historical patterns
- Unrealistic execution assumptions
- Insufficient out-of-sample validation

The perturbation models are not arbitrary stress tests-they are **systematic probes for overfitting**.

### How Each Perturbation Detects Overfitting

| Perturbation | What It Probes | Overfit Signal |
|--------------|----------------|----------------|
| **Skip (p_skip)** | Dependence on specific trades | Performance collapses when key trades are removed |
| **Slippage** | Unrealistic execution assumptions | Edge disappears with realistic transaction costs |
| **Delay** | Timing precision requirements | Strategy needs impossibly precise execution |
| **Shuffle** | Sequence dependence | Performance varies wildly with trade ordering |
| **Bootstrap** | Sample size adequacy | High variance indicates insufficient trade count |

### The Overfit Signature

A strategy exhibits classic overfitting when:

```
Overfit Score = (Baseline_PF - Stressed_PF_P50) / Baseline_PF

If Overfit Score > 0.50 → Strategy loses >50% of edge under stress
If Overfit Score > 0.75 → Strategy is almost certainly overfit
```

**Concrete example:**

| Metric | Baseline | p_skip=0.05 | slip=$100 | delay=1 bar | Combined |
|--------|----------|-------------|-----------|-------------|----------|
| Profit Factor | 3.2 | 2.9 | 2.1 | 1.8 | 1.3 |
| Overfit Score | - | 9% | 34% | 44% | **59%** |

This strategy loses 59% of its edge under realistic adverse conditions. That's a red flag.

### The OHLC Requirement for Delay Testing

**OHLC data is mandatory for delay perturbation.** The engine assumes Databento (or equivalent) provides continuous price data.

The delay model works as follows:

```python
def apply_delay_with_ohlc(trade, delay_bars, ohlc_data):
    """
    Apply execution delay using actual historical prices.
    
    This is the ONLY supported delay model. No statistical approximations.
    
    Parameters:
    -----------
    trade : dict
        Original trade with entry_time, exit_time, entry_price, exit_price
    delay_bars : int
        Number of bars to delay execution
    ohlc_data : DataFrame
        OHLC price data indexed by timestamp
    
    Returns:
    --------
    dict : Trade with adjusted prices based on delayed execution
    """
    if delay_bars == 0:
        return trade
    
    trade = trade.copy()
    
    # Find delayed entry bar
    delayed_entry_time = get_bar_offset(trade['entry_time'], delay_bars, ohlc_data)
    delayed_entry_price = ohlc_data.loc[delayed_entry_time, 'open']
    
    # Find delayed exit bar  
    delayed_exit_time = get_bar_offset(trade['exit_time'], delay_bars, ohlc_data)
    delayed_exit_price = ohlc_data.loc[delayed_exit_time, 'open']
    
    # Delay can only hurt: take the worse price
    if trade['side'] == 'long':
        trade['entry_price'] = max(trade['entry_price'], delayed_entry_price)
        trade['exit_price'] = min(trade['exit_price'], delayed_exit_price)
    else:
        trade['entry_price'] = min(trade['entry_price'], delayed_entry_price)
        trade['exit_price'] = max(trade['exit_price'], delayed_exit_price)
    
    # Recalculate PnL
    trade['pnl'] = calculate_pnl(trade)
    
    return trade
```

**If OHLC data is missing for a trade's time range, the simulation fails with an error.** There is no fallback. This is intentional-delay impact without real prices is guesswork.

### Quantifying Overfitting: The Degradation Surface

The grid search produces a **degradation surface** showing how performance decays across parameter space:

```
                    Slippage ($)
                 0      50     100    200
              ┌──────┬──────┬──────┬──────┐
         0.00 │ 3.2  │ 2.8  │ 2.1  │ 1.4  │
p_skip   0.02 │ 3.0  │ 2.6  │ 1.9  │ 1.2  │
         0.05 │ 2.7  │ 2.3  │ 1.6  │ 0.9  │
         0.10 │ 2.3  │ 1.9  │ 1.3  │ 0.7  │
              └──────┴──────┴──────┴──────┘
              
              Profit Factor P50 (delay=1, shuffle=permute)
```

**Reading the surface:**
- **Gradual degradation** (top-left to bottom-right) = Robust strategy
- **Cliff at specific parameter** = Strategy depends on that assumption
- **Rapid collapse everywhere** = Severely overfit

### Decision Framework

Based on the degradation surface, classify the strategy:

| Classification | Criteria | Recommendation |
|----------------|----------|----------------|
| **Robust** | PF_P50 > 1.5 at (p_skip=0.05, slip=$100, delay=1) | Trade with confidence |
| **Fragile** | PF_P50 drops below 1.5 at moderate stress | Reduce position size 50% |
| **Overfit** | PF_P50 < 1.0 at moderate stress | Do not trade |
| **Highly Overfit** | PF_P50 < 1.0 at minimal stress (p_skip=0.02, slip=$50) | Redesign strategy |

### Reporting Overfit Metrics

Every analysis report should include:

```markdown
## Overfitting Assessment

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Baseline Profit Factor | 3.2 | Raw backtest performance |
| Stressed PF (moderate) | 1.8 | p_skip=0.05, slip=$100, delay=1 |
| Stressed PF (severe) | 0.9 | p_skip=0.10, slip=$200, delay=2 |
| Degradation Rate | 44% | (Baseline - Moderate) / Baseline |
| Overfit Classification | **Fragile** | Proceed with caution |

### Key Vulnerabilities
1. **Slippage sensitivity**: Edge halves at $100 slippage
2. **Delay sensitivity**: Mean-reversion component fails with 1-bar delay
3. **Trade concentration**: 15% of PnL comes from 3 trades
```

---

## 7. Validation: Synthetic Ground Truth Tests

### The Problem

> "Does the engine recover the right answer on synthetic data where you know the true distribution?"

This is crucial. Without validation, we don't know if bugs exist.

### Test Suite: Synthetic Scenarios

#### Test 1: Zero Perturbation = Exact Baseline

```python
def test_zero_perturbation():
    """
    With all perturbations at zero, output should exactly match input.
    """
    trades = generate_synthetic_trades(n=100, seed=42)
    baseline_pf = compute_profit_factor(trades)
    baseline_return = compute_total_return(trades)
    
    results = run_monte_carlo(
        trades,
        n_perms=1000,
        p_skip=0.0,
        slip_max=0.0,
        delay_max=0,
        shuffle='none',
        bootstrap='none'
    )
    
    # All permutations should be identical
    assert np.allclose(results['profit_factor'], baseline_pf)
    assert np.allclose(results['total_return'], baseline_return)
    assert np.std(results['profit_factor']) < 1e-10
```

#### Test 2: Known Skip Distribution

```python
def test_skip_distribution():
    """
    With p_skip=0.5, we should skip ~50% of trades.
    This has a known effect on expected returns.
    """
    # Create trades with constant PnL
    trades = [{'pnl': 100, ...} for _ in range(100)]
    baseline_total = 10000  # 100 trades × $100
    
    results = run_monte_carlo(
        trades,
        n_perms=10000,
        p_skip=0.5,
        slip_max=0.0,
        delay_max=0,
        shuffle='none',
        bootstrap='none'
    )
    
    # Expected return should be ~50% of baseline
    mean_return = np.mean(results['total_return'])
    assert 4500 < mean_return < 5500  # Within 10% of expected
    
    # Distribution should be binomial-ish
    # Var(return) = n × p × (1-p) × pnl² = 100 × 0.5 × 0.5 × 100² = 250,000
    # SD = 500
    std_return = np.std(results['total_return'])
    assert 400 < std_return < 600
```

#### Test 3: Known Slippage Impact

```python
def test_slippage_distribution():
    """
    With slip_max=$100, average slippage should be ~$50 per trade.
    """
    trades = [{'pnl': 1000, ...} for _ in range(100)]
    
    results = run_monte_carlo(
        trades,
        n_perms=10000,
        p_skip=0.0,
        slip_max=100.0,  # Uniform [0, 100]
        delay_max=0,
        shuffle='none',
        bootstrap='none'
    )
    
    # Expected total slippage: 100 trades × $50 avg = $5000
    baseline_return = 100000
    expected_return = 100000 - 5000  # = 95000
    
    mean_return = np.mean(results['total_return'])
    assert 94000 < mean_return < 96000
```

#### Test 4: Shuffle Preserves Distribution (Under Independence)

```python
def test_shuffle_preserves_distribution():
    """
    Shuffling shouldn't change the distribution of total returns
    if trades are independent.
    """
    # IID trades
    np.random.seed(42)
    trades = [{'pnl': np.random.normal(100, 200), ...} for _ in range(100)]
    
    results_none = run_monte_carlo(
        trades, n_perms=10000, shuffle='none', bootstrap='none'
    )
    
    results_shuffle = run_monte_carlo(
        trades, n_perms=10000, shuffle='permute', bootstrap='none'
    )
    
    # Distributions should be statistically indistinguishable
    from scipy.stats import ks_2samp
    stat, pval = ks_2samp(results_none['total_return'], results_shuffle['total_return'])
    assert pval > 0.01  # Not significantly different
```

#### Test 5: Bootstrap Variance

```python
def test_bootstrap_increases_variance():
    """
    Bootstrap resampling should NOT change the mean but SHOULD
    change the variance (typically increases it).
    """
    trades = generate_synthetic_trades(n=50, seed=42)
    baseline_mean = np.mean([t['pnl'] for t in trades]) * 50
    
    results_none = run_monte_carlo(
        trades, n_perms=10000, bootstrap='none'
    )
    
    results_boot = run_monte_carlo(
        trades, n_perms=10000, bootstrap='trade_bootstrap'
    )
    
    # Means should be similar
    assert abs(np.mean(results_none['total_return']) - np.mean(results_boot['total_return'])) < baseline_mean * 0.05
    
    # Bootstrap variance should typically be equal or larger
    # (This is a soft check; depends on underlying distribution)
    var_none = np.var(results_none['total_return'])
    var_boot = np.var(results_boot['total_return'])
    print(f"Variance ratio (boot/none): {var_boot / var_none:.2f}")
```

### Running the Validation Suite

Add to `tests/test_validation.py`:

```python
import pytest
from wale_montecarlo import run_monte_carlo, generate_synthetic_trades

class TestSyntheticValidation:
    
    def test_zero_perturbation(self):
        # ... implementation
    
    def test_skip_distribution(self):
        # ... implementation
    
    # ... etc

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

Run with:
```bash
python -m pytest tests/test_validation.py -v
```

### Expected Validation Results

| Test | Expected Outcome | If Fails |
|------|------------------|----------|
| Zero perturbation | Exact match | Bug in perturbation logic |
| Skip distribution | Mean ≈ 50% baseline | Bug in skip probability |
| Slippage distribution | Mean reduced by avg slippage | Bug in slippage calculation |
| Shuffle preserves dist | KS test p > 0.01 | Bug in shuffle implementation |
| Bootstrap variance | Var(boot) ≥ Var(none) | Bug or edge case |

---

## Summary: Action Items

### Immediate (Fix Performance)

1. **Add `--mode explore` flag** with reduced grid (128 cells, 20K perms)
2. **Verify worker utilization** and fix any parallelization issues
3. **Choose execution tier** based on time constraints and precision needs

### Short-Term (Statistical Rigor)

4. **Switch to BH correction** from Bonferroni
5. **Add drawdown penalty** to robust score
6. **Add overfit classification** to analysis output (Robust/Fragile/Overfit/Highly Overfit)

### Medium-Term (Validation)

7. **Implement synthetic validation test suite**
8. **Add convergence diagnostics** to output
9. **Run ANOVA on interactions** and report in analysis

### Optional (Publication Quality)

10. **Add confidence intervals** to all reported percentiles
11. **Create convergence plots** as part of analysis output
12. **Document interaction effects** in the methodology section

---

## Appendix: Recommended Default Configuration

```python
# config/defaults.py

DEFAULT_CONFIG = {
    # Performance
    'n_workers': 'auto',  # cpu_count() - 1
    'batch_size': 1000,
    'default_perms': 50000,  # Balanced default for most use cases
    
    # Grid (exploration mode)
    'exploration_grid': {
        'p_skip': [0.0, 0.02, 0.05, 0.10],
        'slip': [0, 50, 100, 200],
        'delay': [0, 1],
        'shuffle': ['none', 'permute'],
        'bootstrap': ['none', 'trade_bootstrap'],
        'block_len': [10],
    },
    
    # Statistical
    'p_value_correction': 'bh',  # Changed from 'bonferroni'
    'confidence_level': 0.95,
    
    # Scoring (Gated Multiplicative - see Section 2)
    'robust_score_version': 'v3',  # Gated multiplicative with drawdown penalty
    'maxdd_penalty_start': 0.20,
    'maxdd_penalty_end': 0.60,
    
    # Overfitting Detection
    'overfit_thresholds': {
        'robust': 1.5,          # PF_P50 > 1.5 at moderate stress
        'fragile': 1.0,         # PF_P50 between 1.0-1.5 at moderate stress
        'overfit': 0.0,         # PF_P50 < 1.0 at moderate stress
    },
    'moderate_stress_params': {
        'p_skip': 0.05,
        'slip': 100,
        'delay': 1,
    },
    
    # OHLC Data (required for delay model)
    'ohlc_source': 'databento',  # No fallback - OHLC is mandatory
    'fail_on_missing_ohlc': True,
}
```
