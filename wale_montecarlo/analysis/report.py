"""
Decision report generation for Monte Carlo results.

Creates comprehensive Markdown reports with:
- Executive summary
- Top cells by robust score
- Pareto front analysis
- Parameter sensitivity
- Recommendations
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from ..models import CellSummary, CellConfig
from ..io import load_metrics_compact, get_cell_dir
from .quantiles import compute_tail_risk_metrics
from .robust_score import get_top_cells, categorize_robustness
from .pareto import find_pareto_front_2d, get_pareto_front_stats
from .clustering import find_plateau_clusters, compute_parameter_sensitivity


def generate_decision_report(
    run_dir: str,
    output_path: Optional[str] = None,
    baseline_pf: float = 1.0
) -> str:
    """
    Generate comprehensive decision report from Monte Carlo run.

    Args:
        run_dir: Path to completed run directory
        output_path: Optional path to save report
        baseline_pf: Baseline profit factor for reference

    Returns:
        Markdown report string
    """
    # Load run data
    manifest_path = os.path.join(run_dir, "aggregated", "run_manifest.json")
    if not os.path.exists(manifest_path):
        return "Error: Run manifest not found"

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load cell summaries
    summaries = _load_all_summaries(run_dir)

    if not summaries:
        return "Error: No cell summaries found"

    # Generate report sections
    report = []

    report.append(_generate_header(manifest))
    report.append(_generate_executive_summary(summaries, baseline_pf))
    report.append(_generate_top_cells_table(summaries, baseline_pf))
    report.append(_generate_pareto_analysis(summaries))
    report.append(_generate_sensitivity_analysis(summaries))
    report.append(_generate_cluster_analysis(summaries))
    report.append(_generate_recommendations(summaries, baseline_pf))

    report_text = "\n\n".join(report)

    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_text)

    return report_text


def _load_all_summaries(run_dir: str) -> List[CellSummary]:
    """Load all cell summaries from run directory."""
    from ..models import QuantileStats

    summaries = []
    per_cell_dir = os.path.join(run_dir, "per_cell")

    if not os.path.exists(per_cell_dir):
        return summaries

    for cell_name in os.listdir(per_cell_dir):
        cell_dir = os.path.join(per_cell_dir, cell_name)
        summary_path = os.path.join(cell_dir, "summary.json")

        if os.path.exists(summary_path):
            try:
                with open(summary_path) as f:
                    data = json.load(f)

                # Parse summary
                config = CellConfig.from_dict(data['config'])

                def parse_quantile_stats(d: dict) -> QuantileStats:
                    return QuantileStats(
                        p05=d['p05'], p25=d['p25'], p50=d['p50'],
                        p75=d['p75'], p95=d['p95'],
                        mean=d['mean'], std=d['std']
                    )

                summary = CellSummary(
                    cell_id=data['cell_id'],
                    config=config,
                    n_permutations=data['n_permutations'],
                    total_return=parse_quantile_stats(data['total_return']),
                    max_drawdown=parse_quantile_stats(data['max_drawdown']),
                    profit_factor=parse_quantile_stats(data['profit_factor']),
                    worst_month=parse_quantile_stats(data['worst_month']),
                    pvalue_raw=data.get('pvalue_raw', 1.0),
                    pvalue_corrected=data.get('pvalue_corrected', 1.0),
                    robust_score=data.get('robust_score', 0.0)
                )
                summaries.append(summary)

            except Exception as e:
                continue

    return summaries


def _generate_header(manifest: dict) -> str:
    """Generate report header."""
    return f"""# Monte Carlo Surface Analysis Report

**Run Name:** {manifest.get('run_name', 'Unknown')}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Run Configuration

| Parameter | Value |
|-----------|-------|
| Total Cells | {manifest.get('grid', {}).get('total_cells', 'N/A')} |
| Permutations per Cell | {manifest.get('config', {}).get('n_per_cell', 'N/A'):,} |
| Parallel Workers | {manifest.get('config', {}).get('n_jobs', 'N/A')} |
| Fixed Delay | {manifest.get('config', {}).get('fixed_delay', 'None')} |

---"""


def _generate_executive_summary(
    summaries: List[CellSummary],
    baseline_pf: float
) -> str:
    """Generate executive summary section."""
    n_cells = len(summaries)

    # Categorize cells
    categories = {}
    for s in summaries:
        cat = categorize_robustness(
            s.robust_score,
            s.profit_factor.p05,
            s.max_drawdown.p95
        )
        categories[cat] = categories.get(cat, 0) + 1

    # Overall stats
    robust_scores = [s.robust_score for s in summaries]
    pf_p50s = [s.profit_factor.p50 for s in summaries]

    return f"""## Executive Summary

### Overall Assessment

Analyzed **{n_cells:,} cells** across the perturbation parameter space.

| Robustness Category | Cell Count | Percentage |
|---------------------|------------|------------|
| Excellent | {categories.get('excellent', 0)} | {categories.get('excellent', 0)/n_cells*100:.1f}% |
| Good | {categories.get('good', 0)} | {categories.get('good', 0)/n_cells*100:.1f}% |
| Moderate | {categories.get('moderate', 0)} | {categories.get('moderate', 0)/n_cells*100:.1f}% |
| Weak | {categories.get('weak', 0)} | {categories.get('weak', 0)/n_cells*100:.1f}% |
| Poor | {categories.get('poor', 0)} | {categories.get('poor', 0)/n_cells*100:.1f}% |

### Key Metrics

- **Best Robust Score:** {max(robust_scores):.3f}
- **Median Robust Score:** {sorted(robust_scores)[len(robust_scores)//2]:.3f}
- **Best PF (P50):** {max(pf_p50s):.2f}
- **Median PF (P50):** {sorted(pf_p50s)[len(pf_p50s)//2]:.2f}

---"""


def _generate_top_cells_table(
    summaries: List[CellSummary],
    baseline_pf: float,
    n: int = 20
) -> str:
    """Generate top cells table."""
    top = get_top_cells(summaries, n=n, baseline_pf=baseline_pf)

    rows = []
    for i, cell in enumerate(top, 1):
        rows.append(
            f"| {i} | {cell['robust_score']:.3f} | {cell['category']} | "
            f"{cell['pf_p50']:.2f} | {cell['pf_p05']:.2f} | "
            f"{cell['max_dd_p95']:.1f}% | {cell['config']['p_skip']:.2f} | "
            f"{cell['config']['slip_dollars']:.0f} |"
        )

    table = "\n".join(rows)

    return f"""## Top {n} Cells by Robust Score

| Rank | Score | Category | PF P50 | PF P05 | DD P95 | Skip | Slip |
|------|-------|----------|--------|--------|--------|------|------|
{table}

---"""


def _generate_pareto_analysis(summaries: List[CellSummary]) -> str:
    """Generate Pareto front analysis."""
    pareto = find_pareto_front_2d(summaries)
    stats = get_pareto_front_stats(pareto)

    pareto_rows = []
    for cell in sorted(pareto, key=lambda c: c.profit_factor.p50, reverse=True)[:10]:
        pareto_rows.append(
            f"| {cell.cell_id[:30]}... | {cell.profit_factor.p50:.2f} | "
            f"{cell.max_drawdown.p50:.1f}% | {cell.total_return.p50:.1f}% |"
        )

    table = "\n".join(pareto_rows)

    return f"""## Pareto Front Analysis

The Pareto front contains **{stats['n_cells']} cells** representing the best
trade-offs between profit factor and max drawdown.

### Pareto Front Statistics

| Metric | Min | Max | Mean |
|--------|-----|-----|------|
| Profit Factor | {stats['pf_range'][0]:.2f} | {stats['pf_range'][1]:.2f} | {stats.get('pf_mean', 0):.2f} |
| Max Drawdown | {stats['dd_range'][0]:.1f}% | {stats['dd_range'][1]:.1f}% | {stats.get('dd_mean', 0):.1f}% |
| Total Return | {stats['return_range'][0]:.1f}% | {stats['return_range'][1]:.1f}% | {stats.get('return_mean', 0):.1f}% |

### Top Pareto-Optimal Cells

| Cell ID | PF P50 | DD P50 | Return P50 |
|---------|--------|--------|------------|
{table}

---"""


def _generate_sensitivity_analysis(summaries: List[CellSummary]) -> str:
    """Generate parameter sensitivity analysis."""
    sensitivities = compute_parameter_sensitivity(summaries)

    # Sort by sensitivity
    sorted_sens = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)

    rows = []
    for param, sens in sorted_sens:
        impact = "High" if sens > 0.3 else "Medium" if sens > 0.1 else "Low"
        rows.append(f"| {param} | {sens:.3f} | {impact} |")

    table = "\n".join(rows)

    return f"""## Parameter Sensitivity Analysis

How much does each parameter affect the robust score?

| Parameter | Sensitivity | Impact |
|-----------|-------------|--------|
{table}

**Interpretation:**
- High sensitivity (>0.3): Parameter choice significantly affects results
- Medium sensitivity (0.1-0.3): Moderate impact on results
- Low sensitivity (<0.1): Results are stable across parameter values

---"""


def _generate_cluster_analysis(summaries: List[CellSummary]) -> str:
    """Generate plateau cluster analysis."""
    clusters = find_plateau_clusters(summaries)

    if not clusters:
        return """## Plateau Cluster Analysis

No significant plateau clusters found.

---"""

    rows = []
    for i, cluster in enumerate(clusters[:5], 1):
        rows.append(
            f"| {i} | {cluster['n_cells']} | {cluster['robust_score_mean']:.3f} | "
            f"±{cluster['robust_score_std']:.3f} |"
        )

    table = "\n".join(rows)

    return f"""## Plateau Cluster Analysis

Found **{len(clusters)} plateau clusters** (regions with similar robust scores).
Large clusters indicate stable parameter regions.

### Top Clusters

| Cluster | Cells | Mean Score | Std |
|---------|-------|------------|-----|
{table}

---"""


def _generate_recommendations(
    summaries: List[CellSummary],
    baseline_pf: float
) -> str:
    """Generate recommendations section."""
    # Find best cell
    best = max(summaries, key=lambda s: s.robust_score)

    # Find most stable region
    clusters = find_plateau_clusters(summaries)
    largest_cluster = max(clusters, key=lambda c: c['n_cells']) if clusters else None

    # Assess overall robustness
    n_good = sum(1 for s in summaries if s.robust_score >= 1.0)
    pct_good = n_good / len(summaries) * 100

    recs = []

    if pct_good >= 50:
        recs.append("- **Strategy appears robust** across most parameter combinations")
    elif pct_good >= 20:
        recs.append("- **Strategy shows moderate robustness** - some parameter sensitivity exists")
    else:
        recs.append("- **Strategy shows high parameter sensitivity** - results vary significantly")

    if best.profit_factor.p05 >= 1.5:
        recs.append("- **Strong edge maintained** even in adverse conditions (P05 PF > 1.5)")
    elif best.profit_factor.p05 >= 1.0:
        recs.append("- **Edge preserved** but diminished under stress (P05 PF > 1.0)")
    else:
        recs.append("- **Edge at risk** under stress conditions (P05 PF < 1.0)")

    if best.max_drawdown.p95 <= 30:
        recs.append("- **Drawdown risk acceptable** (P95 DD ≤ 30%)")
    elif best.max_drawdown.p95 <= 50:
        recs.append("- **Moderate drawdown risk** (P95 DD ≤ 50%)")
    else:
        recs.append("- **High drawdown risk** under stress (P95 DD > 50%)")

    if largest_cluster and largest_cluster['n_cells'] >= 10:
        recs.append(f"- **Stable region found** with {largest_cluster['n_cells']} cells")

    rec_text = "\n".join(recs)

    return f"""## Recommendations

### Best Configuration

```
Cell ID: {best.cell_id}
Robust Score: {best.robust_score:.3f}
PF (P50/P05): {best.profit_factor.p50:.2f} / {best.profit_factor.p05:.2f}
Max DD (P50/P95): {best.max_drawdown.p50:.1f}% / {best.max_drawdown.p95:.1f}%
Return (P50): {best.total_return.p50:.1f}%
```

### Key Findings

{rec_text}

### Suggested Actions

1. Focus on parameter combinations within the top cluster region
2. Run additional permutations on the top 10 cells for higher precision
3. Consider the trade-off between profit factor and drawdown on the Pareto front
4. Test the best configuration with out-of-sample data

---

*Report generated by Wale Monte Carlo Engine*
"""
