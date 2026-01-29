#!/usr/bin/env python3
"""
Fetch OHLC data from Databento for Monte Carlo delay modeling.

This script downloads historical OHLC data for futures contracts (default: NQ)
which is used by the Monte Carlo engine to model realistic execution delays.

SETUP:
------
1. Get a Databento API key from https://databento.com
2. Set it as an environment variable:
   
   Windows (PowerShell):
       $env:DATABENTO_API_KEY = "your-api-key-here"
   
   Linux/Mac:
       export DATABENTO_API_KEY="your-api-key-here"

3. Or pass it as a command line argument (not recommended for security):
       python fetch_ohlc.py --key your-api-key-here

USAGE:
------
    # Using environment variable (recommended)
    python scripts/fetch_ohlc.py --symbol NQ --start 2023-01-01 --end 2026-01-29
    
    # Using command line key (less secure)
    python scripts/fetch_ohlc.py --key db-xxx --symbol NQ

OUTPUT:
-------
    Creates ohlc.csv in the current directory with columns:
    time, open, high, low, close, volume
"""
import os
import sys
import argparse
import databento as db
import pandas as pd


def get_contracts(symbol: str, start_year: int, end_year: int):
    """Generate list of futures contracts for a symbol."""
    # Futures months: H=March, M=June, U=September, Z=December
    months = [('H', 3), ('M', 6), ('U', 9), ('Z', 12)]
    contracts = []
    
    for year in range(start_year, end_year + 1):
        year_suffix = str(year)[-1]  # Last digit of year
        for month_code, month_num in months:
            sym = f"{symbol}{month_code}{year_suffix}"
            # Approximate contract active period
            start_month = month_num - 3 if month_num > 3 else 12 + month_num - 3
            start_yr = year if month_num > 3 else year - 1
            end_month = month_num
            
            start_date = f"{start_yr}-{start_month:02d}-15"
            end_date = f"{year}-{end_month:02d}-15"
            contracts.append((sym, start_date, end_date))
    
    return contracts


def main():
    parser = argparse.ArgumentParser(
        description="Fetch OHLC data from Databento for Monte Carlo simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--key', type=str, help='Databento API key (or set DATABENTO_API_KEY env var)')
    parser.add_argument('--symbol', type=str, default='NQ', help='Futures symbol (default: NQ)')
    parser.add_argument('--start', type=str, default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2026-01-29', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='ohlc.csv', help='Output file path')
    parser.add_argument('--schema', type=str, default='ohlcv-1h', help='OHLC schema (ohlcv-1m, ohlcv-1h, ohlcv-1d)')
    
    args = parser.parse_args()
    
    # Get API key from arg or environment
    api_key = args.key or os.environ.get('DATABENTO_API_KEY')
    if not api_key:
        print("ERROR: No Databento API key provided.")
        print()
        print("Set it as an environment variable:")
        print("  PowerShell: $env:DATABENTO_API_KEY = 'your-key'")
        print("  Bash:       export DATABENTO_API_KEY='your-key'")
        print()
        print("Or pass it as an argument:")
        print("  python fetch_ohlc.py --key your-key")
        sys.exit(1)
    
    print(f"Connecting to Databento...")
    client = db.Historical(api_key)
    
    # Parse years from dates
    start_year = int(args.start.split('-')[0])
    end_year = int(args.end.split('-')[0])
    
    contracts = get_contracts(args.symbol, start_year, end_year)
    print(f"Fetching {args.symbol} OHLC data ({len(contracts)} contracts)...")
    
    all_dfs = []
    for sym, start, end in contracts:
        print(f"  {sym}: {start} to {end}...", end=" ")
        try:
            data = client.timeseries.get_range(
                dataset='GLBX.MDP3',
                symbols=[sym],
                schema=args.schema,
                start=start,
                end=end,
            )
            df = data.to_df()
            if len(df) > 0:
                all_dfs.append(df)
                print(f"{len(df)} bars")
            else:
                print("no data")
        except Exception as e:
            print(f"error: {e}")
    
    if not all_dfs:
        print("No data retrieved. Check your API key and date range.")
        sys.exit(1)
    
    # Combine and dedupe
    combined = pd.concat(all_dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    
    # Convert to engine format (remove timezone)
    ohlc = pd.DataFrame()
    ohlc['time'] = pd.to_datetime(combined.index).tz_localize(None)
    ohlc['open'] = combined['open'].values
    ohlc['high'] = combined['high'].values
    ohlc['low'] = combined['low'].values
    ohlc['close'] = combined['close'].values
    ohlc['volume'] = combined['volume'].values
    
    ohlc.to_csv(args.output, index=False)
    print(f"\nSaved {len(ohlc)} bars to {args.output}")
    print(f"Date range: {ohlc['time'].min()} to {ohlc['time'].max()}")


if __name__ == '__main__':
    main()
