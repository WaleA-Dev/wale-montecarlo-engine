"""
Monte Carlo Trading Strategy Analyzer

Simple CLI for industry-standard Monte Carlo simulation.

Usage:
    python -m wale_montecarlo analyze trades.csv
    python -m wale_montecarlo analyze trades.csv --capital 100000 --output report.html
"""

import argparse
import sys
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo Trading Strategy Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Serve command (web UI)
    serve_parser = subparsers.add_parser('serve', help='Launch the web UI (Strategy Stress Lab)')
    serve_parser.add_argument('--port', type=int, default=8742, help='Port (default: 8742)')
    serve_parser.add_argument('--no-browser', action='store_true', help="Don't open a browser")

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Full Monte Carlo analysis')
    analyze_parser.add_argument('trades', type=str, help='Path to trades CSV file')
    analyze_parser.add_argument('--capital', type=float, default=100000,
                               help='Starting capital for ruin analysis (default: 100000)')
    analyze_parser.add_argument('--samples', type=int, default=10000,
                               help='Number of bootstrap samples (default: 10000)')
    analyze_parser.add_argument('--output', '-o', type=str, default=None,
                               help='Output HTML report path (default: trades_report.html)')
    analyze_parser.add_argument('--no-stress', action='store_true',
                               help='Skip stress scenario analysis')
    
    # Quick command
    quick_parser = subparsers.add_parser('quick', help='Quick summary (1000 samples)')
    quick_parser.add_argument('trades', type=str, help='Path to trades CSV file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'serve':
        import threading, webbrowser
        from .webapp import create_app
        url = f"http://127.0.0.1:{args.port}"
        if not args.no_browser:
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        print(f"Strategy Stress Lab running at {url}  (Ctrl+C to quit)")
        create_app().run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
        return

    # Import here to avoid circular imports
    from .io import load_trade_list
    from .analysis.bootstrap_equity import bootstrap_equity_curves, format_bootstrap_summary
    from .analysis.ruin_probability import estimate_ruin_probability, format_ruin_summary
    from .analysis.stress_scenarios import run_all_scenarios, format_scenario_comparison, compute_overfit_score, format_overfit_summary
    from .report.html_report import generate_html_report, save_report
    
    # Load trades
    trades_path = Path(args.trades).resolve()
    if not trades_path.exists():
        print(f"Error: File not found: {trades_path}")
        sys.exit(1)
    
    print(f"Loading trades from: {trades_path}")
    trades = load_trade_list(str(trades_path))
    print(f"Loaded {len(trades)} trades")
    
    if len(trades) < 5:
        print("Warning: Very small sample size. Results may not be reliable.")
    
    # Run analysis based on command
    if args.command == 'quick':
        n_samples = 1000
        capital = 100000
        run_stress = False
    else:
        n_samples = args.samples
        capital = args.capital
        run_stress = not args.no_stress
    
    # Bootstrap analysis
    print(f"\nRunning bootstrap analysis ({n_samples:,} samples)...")
    bootstrap = bootstrap_equity_curves(trades, n_samples=n_samples, seed=42)
    print(format_bootstrap_summary(bootstrap))
    
    # Ruin analysis
    print(f"\nRunning ruin probability analysis...")
    ruin = estimate_ruin_probability(trades, starting_capital=capital, seed=42)
    print(format_ruin_summary(ruin))
    
    # Stress scenarios and overfit detection
    scenarios = None
    overfit = None
    if run_stress:
        print(f"\nRunning stress scenarios...")
        scenarios = run_all_scenarios(trades)
        print(format_scenario_comparison(scenarios))
        
        # Compute overfit score
        overfit = compute_overfit_score(scenarios)
        print(f"\n{format_overfit_summary(overfit, scenarios)}")
    
    # Generate HTML report
    if args.command == 'analyze':
        output_path = args.output or trades_path.stem + '_report.html'
        output_path = Path(output_path).resolve()
        
        print(f"\nGenerating HTML report: {output_path}")
        html = generate_html_report(
            bootstrap_result=bootstrap,
            ruin_result=ruin,
            scenario_results=scenarios,
            overfit_score=overfit,
            title=f"Monte Carlo Analysis: {trades_path.name}"
        )
        save_report(html, str(output_path))
        print(f"Report saved to: {output_path}")
    
    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
