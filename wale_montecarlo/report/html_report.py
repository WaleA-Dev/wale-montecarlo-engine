"""
HTML report generator.

Creates a visual, shareable report from Monte Carlo analysis results.
"""

from typing import Dict, Optional
from datetime import datetime
import base64
import io

# Try to import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def create_equity_curve_plot(equity_curves: 'np.ndarray') -> str:
    """Create equity curve plot with confidence bands, return as base64."""
    if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
        return ""
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Compute percentiles
    p5 = np.percentile(equity_curves, 5, axis=0)
    p25 = np.percentile(equity_curves, 25, axis=0)
    p50 = np.percentile(equity_curves, 50, axis=0)
    p75 = np.percentile(equity_curves, 75, axis=0)
    p95 = np.percentile(equity_curves, 95, axis=0)
    
    x = range(len(p50))
    
    # Plot confidence bands
    ax.fill_between(x, p5, p95, alpha=0.2, color='blue', label='90% CI')
    ax.fill_between(x, p25, p75, alpha=0.3, color='blue', label='50% CI')
    ax.plot(x, p50, color='blue', linewidth=2, label='Median')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative P&L ($)')
    ax.set_title('Equity Curve (Bootstrap 95% Confidence Band)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_drawdown_histogram(drawdowns: 'np.ndarray') -> str:
    """Create drawdown distribution histogram, return as base64."""
    if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
        return ""
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.hist(drawdowns, bins=50, color='red', alpha=0.7, edgecolor='darkred')
    ax.axvline(np.percentile(drawdowns, 50), color='blue', linestyle='--', 
               label=f'Median: ${np.percentile(drawdowns, 50):,.0f}')
    ax.axvline(np.percentile(drawdowns, 95), color='red', linestyle='--',
               label=f'P95: ${np.percentile(drawdowns, 95):,.0f}')
    
    ax.set_xlabel('Maximum Drawdown ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Max Drawdown Distribution')
    ax.legend()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_html_report(
    bootstrap_result: 'BootstrapResult',
    ruin_result: Optional['RuinResult'] = None,
    scenario_results: Optional[Dict] = None,
    title: str = "Monte Carlo Analysis Report"
) -> str:
    """
    Generate a complete HTML report.
    
    Args:
        bootstrap_result: Results from bootstrap_equity_curves()
        ruin_result: Optional results from estimate_ruin_probability()
        scenario_results: Optional dict from run_all_scenarios()
        title: Report title
    
    Returns:
        Complete HTML document as string
    """
    
    # Generate plots
    equity_plot_b64 = create_equity_curve_plot(bootstrap_result.equity_curves)
    dd_plot_b64 = create_drawdown_histogram(bootstrap_result.drawdown_distribution)
    
    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #1a1a2e; margin-bottom: 20px; }}
        h2 {{ color: #16213e; margin: 30px 0 15px; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; }}
        .card {{ 
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .metric {{
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #1a1a2e; }}
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        img {{ max-width: 100%; height: auto; border-radius: 6px; }}
        .timestamp {{ color: #888; font-size: 12px; margin-top: 30px; text-align: center; }}
    </style>
</head>
<body>
    <h1>📊 {title}</h1>
    
    <div class="card">
        <h2>Key Metrics</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-value {'positive' if bootstrap_result.total_return_mean > 0 else 'negative'}">
                    ${bootstrap_result.total_return_mean:,.0f}
                </div>
                <div class="metric-label">Expected Return</div>
            </div>
            <div class="metric">
                <div class="metric-value">{bootstrap_result.sharpe_mean:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric">
                <div class="metric-value negative">${bootstrap_result.max_drawdown_p95:,.0f}</div>
                <div class="metric-label">Max DD (P95)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{bootstrap_result.win_rate:.0%}</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{bootstrap_result.profit_factor:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            <div class="metric">
                <div class="metric-value">{bootstrap_result.n_trades}</div>
                <div class="metric-label">Trades</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>Equity Curve</h2>
        <p>Confidence bands from {bootstrap_result.n_samples:,} bootstrap resamples.</p>
        {'<img src="data:image/png;base64,' + equity_plot_b64 + '" alt="Equity Curve">' if equity_plot_b64 else '<p>Matplotlib not available for plotting.</p>'}
    </div>
    
    <div class="card">
        <h2>Return Confidence Intervals</h2>
        <table>
            <tr><th>Metric</th><th>Value</th><th>95% CI</th></tr>
            <tr>
                <td>Total Return</td>
                <td>${bootstrap_result.total_return_mean:,.0f}</td>
                <td>${bootstrap_result.total_return_ci[0]:,.0f} to ${bootstrap_result.total_return_ci[1]:,.0f}</td>
            </tr>
            <tr>
                <td>Sharpe Ratio</td>
                <td>{bootstrap_result.sharpe_mean:.2f}</td>
                <td>{bootstrap_result.sharpe_ci[0]:.2f} to {bootstrap_result.sharpe_ci[1]:.2f}</td>
            </tr>
        </table>
    </div>
    
    <div class="card">
        <h2>Drawdown Analysis</h2>
        <table>
            <tr><th>Percentile</th><th>Max Drawdown</th></tr>
            <tr><td>Median (P50)</td><td>${bootstrap_result.max_drawdown_median:,.0f}</td></tr>
            <tr><td>P75</td><td>${bootstrap_result.max_drawdown_p75:,.0f}</td></tr>
            <tr><td>P95</td><td>${bootstrap_result.max_drawdown_p95:,.0f}</td></tr>
        </table>
        {'<img src="data:image/png;base64,' + dd_plot_b64 + '" alt="Drawdown Distribution">' if dd_plot_b64 else ''}
    </div>
"""
    
    # Add ruin analysis if available
    if ruin_result:
        html += f"""
    <div class="card">
        <h2>Ruin Probability</h2>
        <p>Starting capital: ${ruin_result.starting_capital:,.0f}</p>
        <table>
            <tr><th>Drawdown Threshold</th><th>Probability</th></tr>
            <tr><td>10% Drawdown</td><td>{ruin_result.prob_10pct_dd:.1%}</td></tr>
            <tr><td>20% Drawdown</td><td>{ruin_result.prob_20pct_dd:.1%}</td></tr>
            <tr><td>30% Drawdown</td><td>{ruin_result.prob_30pct_dd:.1%}</td></tr>
            <tr><td>50% Drawdown (Ruin)</td><td class="negative">{ruin_result.prob_50pct_dd:.1%}</td></tr>
        </table>
        <p><strong>Recommended Minimum Capital:</strong> ${ruin_result.recommended_capital:,.0f}</p>
    </div>
"""
    
    # Add scenario comparison if available
    if scenario_results:
        html += """
    <div class="card">
        <h2>Stress Scenarios</h2>
        <table>
            <tr><th>Scenario</th><th>Return</th><th>After Costs</th><th>Max DD</th><th>Sharpe</th></tr>
"""
        for name in ['optimistic', 'realistic', 'pessimistic', 'extreme']:
            if name in scenario_results:
                r = scenario_results[name]
                html += f"""
            <tr>
                <td>{r.scenario.name}</td>
                <td>${r.total_return:,.0f}</td>
                <td>${r.total_return_after_costs:,.0f}</td>
                <td>${r.max_drawdown:,.0f}</td>
                <td>{r.sharpe_ratio:.2f}</td>
            </tr>
"""
        html += """
        </table>
    </div>
"""
    
    # Close HTML
    html += f"""
    <p class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</body>
</html>
"""
    
    return html


def save_report(html: str, filepath: str) -> None:
    """Save HTML report to file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
