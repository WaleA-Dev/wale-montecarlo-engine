# Statistical Analysis: Methods and Interpretation

This document explains the statistical methods used in the Monte Carlo analysis engine and how to interpret the results.

---

## Table of Contents

1. [Distribution Metrics](#distribution-metrics)
2. [Robust Score](#robust-score)
3. [P-Value Calculation](#p-value-calculation)
4. [Pareto Front Analysis](#pareto-front-analysis)
5. [Plateau Clustering](#plateau-clustering)
6. [Interpreting Results](#interpreting-results)

---

## Distribution Metrics

For each cell (parameter combination), we run N permutations and collect:

- **total_return_pct**: Cumulative return over the backtest period
- **max_drawdown_pct**: Maximum peak-to-trough decline
- **profit_factor**: Gross profit / Gross loss
- **worst_month_pct**: Worst single-month return
- **trades**: Number of trades executed (after skipping)

From these N values, we compute distributional summaries.

### Quantiles

```python
quantiles = np.nanpercentile(values, [5, 50, 95])
```

| Quantile | Symbol | Interpretation |
|----------|--------|----------------|
| P05 | 5th percentile | "Bad case" — only 5% of outcomes are worse |
| P50 | Median | "Typical case" — half above, half below |
| P95 | 95th percentile | "Good case" — only 5% of outcomes are better |

**Why P05/P95 instead of min/max?**

Min and max are dominated by outliers. P05/P95 give you the "edges" of the realistic distribution without extreme noise.

### Mean and Standard Deviation

```python
stats = {
    "mean": np.nanmean(values),
    "std": np.nanstd(values)
}
```

Useful for:
- **Mean**: Central tendency (but can be skewed by outliers)
- **Std**: Dispersion — high std means high uncertainty

---

## Robust Score

The robust score combines performance and statistical significance into a single metric.

### Formula

$$
\text{Robust Score} = \text{PF}_{P50} \times (1 - p_{\text{corrected}})
$$

Where:
- $\text{PF}_{P50}$ = Median profit factor across permutations
- $p_{\text{corrected}}$ = Bonferroni-corrected p-value

### Intuition

The robust score is high when:
1. **Median PF is high**: The strategy typically produces good results
2. **P-value is low**: The results are unlikely to be due to chance

It's low when:
- PF is low (strategy doesn't work)
- P-value is high (could be luck)
- Either factor is bad (multiplicative penalty)

### Why Multiplicative?

Additive combination (PF + significance) doesn't capture the interaction correctly. A strategy with PF=10 but p=0.9 shouldn't score well, and neither should one with PF=1.01 and p=0.0001.

Multiplication ensures **both** factors need to be good.

---

## P-Value Calculation

### What We're Testing

**Null hypothesis**: The perturbation parameters don't matter — results are as good as baseline.

**Alternative**: Perturbation degrades performance below baseline.

### Method

We use a **tail probability test**:

$$
p_{\text{raw}} = \frac{\#\{\text{PF}_i \geq \text{PF}_{\text{baseline}}\}}{N}
$$

This is the fraction of permutations where the perturbed strategy achieved profit factor at least as good as the unperturbed baseline.

### Example

- Baseline PF = 3.5
- Run 200,000 permutations with perturbations
- 4,600 of them have PF ≥ 3.5
- p_raw = 4,600 / 200,000 = 0.023

**Interpretation**: Only 2.3% of perturbed runs match or beat the baseline.

### Bonferroni Correction

When testing multiple cells, we need to correct for multiple comparisons:

$$
p_{\text{corrected}} = \min(1.0, p_{\text{raw}} \times N_{\text{cells}})
$$

If testing 1,500 cells:
- p_raw = 0.023
- p_corrected = min(1.0, 0.023 × 1500) = 1.0

This is very conservative — it's saying "with 1,500 tests, a raw p of 0.023 could easily be chance."

### Interpretation Guidelines

| p_corrected | Interpretation |
|-------------|----------------|
| < 0.05 | Strong evidence against null (rare with many cells) |
| 0.05-0.20 | Moderate evidence |
| 0.20-0.50 | Weak evidence |
| > 0.50 | Little evidence against null |

---

## Pareto Front Analysis

### What is a Pareto Front?

Given multiple objectives, a solution is **Pareto-optimal** if no other solution is better in *all* objectives.

### 2D Pareto Front: PF vs MaxDD

**Objectives**:
- Maximize: Profit Factor P50
- Minimize: Max Drawdown P95

**Algorithm**:
```python
def pareto_front_2d(df, maximize_col, minimize_col):
    df = df.sort_values(maximize_col, ascending=False)
    
    pareto_points = []
    min_so_far = float("inf")
    
    for idx, row in df.iterrows():
        if row[minimize_col] < min_so_far:
            pareto_points.append(idx)
            min_so_far = row[minimize_col]
    
    return df.loc[pareto_points]
```

**Visual interpretation**:

```
    PF_P50 ↑
         │     * ←── Pareto optimal
         │   *
         │ *    *
         │      *
         └─────────→ MaxDD_P95
               (lower is better)
```

Points on the frontier represent "you can't improve one without worsening the other."

### 3D Pareto Front

**Objectives**:
- Maximize: Profit Factor P50
- Maximize: Return P50
- Minimize: Max Drawdown P95

**Definition**: Point A dominates point B if A is ≥ B on all objectives and > B on at least one.

A point is Pareto-optimal if no other point dominates it.

### Interpretation

The Pareto front shows you the **efficient frontier** — the best possible tradeoffs. Cells *not* on the front are strictly dominated.

---

## Plateau Clustering

### What is a Plateau?

A **plateau** is a region of parameter space where the metric stays relatively stable. If cells with p_skip from 0.02-0.05 all have similar robust scores, that's a plateau.

### Why Care?

Plateaus indicate **robustness to parameter choice**. If your results only hold for p_skip=0.0317 exactly, you're probably overfitting. If they hold for p_skip ∈ [0.02, 0.06], you've found something real.

### Algorithm

```python
def cluster_plateaus(df, metric_col, threshold_pct=0.1):
    df = df.sort_values(metric_col, ascending=False)
    
    clusters = []
    current_cluster = 0
    prev_value = None
    
    for idx, row in df.iterrows():
        value = row[metric_col]
        if prev_value is None:
            clusters.append(current_cluster)
        else:
            rel_diff = abs(value - prev_value) / prev_value
            if rel_diff > threshold_pct:
                current_cluster += 1
            clusters.append(current_cluster)
        prev_value = value
    
    df["cluster"] = clusters
    return df
```

### Threshold

Default: 10% (`threshold_pct=0.1`)

Two cells are in the same cluster if their metric differs by less than 10%.

### Output

| Cluster | N Cells | Robust Score (Mean±Std) | Parameter Ranges |
|---------|---------|-------------------------|------------------|
| 0 | 12 | 1.84 ± 0.05 | p_skip: 0.02-0.05, slip: 25-75 |
| 1 | 8 | 1.62 ± 0.08 | p_skip: 0.01-0.03, slip: 100-150 |
| 2 | 47 | 1.31 ± 0.12 | Various |

---

## Interpreting Results

### Reading the Decision Report

#### Section 1: Verification

First, verify all cells completed correctly:
- **Cells Verified**: Should equal total cells
- **Duplicates Dropped**: Small number is normal on resume

If verification fails, the results aren't trustworthy.

#### Section 2: Top Candidates

The table shows the best cells by robust score:

| Rank | Cell ID | PF_P50 | MaxDD_P95 | Ret_P50 | Robust Score |
|------|---------|--------|-----------|---------|--------------|
| 1 | 3_2_0_1_1 | 2.83 | 28.4% | 456% | 1.87 |

**What to look for**:
- **PF_P50 > 2.0**: Strategy doubles its wins vs losses
- **MaxDD_P95 < 40%**: Drawdowns are manageable in 95% of cases
- **Robust Score > 1.5**: Combination of good performance and significance

#### Section 3: Pareto Front

The Pareto front shows optimal tradeoffs:

- **Top-left of 2D front**: Best PF for given DD tolerance
- **Multi-dimensional front**: Balanced across all objectives

If the same cell appears in both Pareto front AND top robust score, that's a strong candidate.

#### Section 4: Plateau Clusters

Look for:
- **Large clusters**: Many cells with similar performance = stable region
- **Tight std**: Low variance within cluster = predictable behavior
- **Sensible parameter ranges**: The parameters in the cluster should make intuitive sense

### Red Flags

🚩 **Robust score drops sharply with small perturbations**: Over-optimized
🚩 **P-value correction makes everything insignificant**: Not enough edge
🚩 **Best cell is an outlier**: Might be lucky
🚩 **MaxDD_P95 > 50%**: Tail risk is dangerous
🚩 **Return distribution is bimodal**: Strategy has two very different regimes

### Green Flags

✅ **Stable plateau in top 10%**: Robust to parameter choice
✅ **Multiple Pareto-optimal cells with similar parameters**: Consistent edge
✅ **Low raw p-values across many cells**: Real effect
✅ **MaxDD_P95 similar to baseline**: Perturbations don't blow up risk
✅ **Block bootstrap ≈ trade bootstrap**: Results don't depend on sequence

---

## Practical Recommendations

### How to Pick a Cell

1. **Start with Pareto front**: These are provably efficient
2. **Check robust score ranking**: Confirms statistical significance
3. **Find the plateau**: Pick from the middle of a stable region
4. **Verify parameters make sense**: Don't pick p_skip=0.00 — that's unrealistic

### What Parameters to Use Live

Generally:
- **p_skip**: Use the middle of your plateau, not the edge
- **slip**: Use the pessimistic end of your acceptable range
- **delay**: Use 1 bar unless you have very fast execution

### Position Sizing

Use the stress-tested metrics for sizing:
- **Kelly fraction**: Use P05 return, not mean
- **Max position**: Based on P95 drawdown you can tolerate
- **Stop-out level**: Set based on worst observed outcomes

---

## Summary Table

| Metric | Good | Concerning | Critical |
|--------|------|------------|----------|
| Robust Score | > 1.5 | 1.0-1.5 | < 1.0 |
| PF_P50 | > 2.0 | 1.5-2.0 | < 1.5 |
| MaxDD_P95 | < 35% | 35-50% | > 50% |
| p_corrected | < 0.3 | 0.3-0.6 | > 0.6 |
| Plateau size | > 10 cells | 5-10 cells | < 5 cells |

These thresholds are guidelines, not rules. The right values depend on your risk tolerance and strategy type.
