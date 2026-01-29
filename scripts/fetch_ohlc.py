#!/usr/bin/env python3
"""
Fetch NQ OHLC data from Databento for Monte Carlo simulation.
API key is passed as command line argument, NOT stored in file.
"""
import sys
import databento as db
import pandas as pd

# Get API key from command line
api_key = sys.argv[1] if len(sys.argv) > 1 else None
if not api_key:
    print("Usage: python fetch_ohlc.py <DATABENTO_API_KEY>")
    sys.exit(1)

client = db.Historical(api_key)

print('Fetching NQ OHLC data by contract...')

# Define contracts covering trade date range (2023-01 to 2026-01)
contracts = [
    ('NQH3', '2022-12-15', '2023-03-15'),
    ('NQM3', '2023-03-15', '2023-06-15'),
    ('NQU3', '2023-06-15', '2023-09-15'),
    ('NQZ3', '2023-09-15', '2023-12-15'),
    ('NQH4', '2023-12-15', '2024-03-15'),
    ('NQM4', '2024-03-15', '2024-06-15'),
    ('NQU4', '2024-06-15', '2024-09-15'),
    ('NQZ4', '2024-09-15', '2024-12-15'),
    ('NQH5', '2024-12-15', '2025-03-15'),
    ('NQM5', '2025-03-15', '2025-06-15'),
    ('NQU5', '2025-06-15', '2025-09-15'),
    ('NQZ5', '2025-09-15', '2025-12-15'),
    ('NQH6', '2025-12-15', '2026-02-01'),
]

all_dfs = []

for sym, start, end in contracts:
    print(f'  Fetching {sym} from {start} to {end}...')
    try:
        data = client.timeseries.get_range(
            dataset='GLBX.MDP3',
            symbols=[sym],
            schema='ohlcv-1h',
            start=start,
            end=end,
        )
        df = data.to_df()
        if len(df) > 0:
            all_dfs.append(df)
            print(f'    Got {len(df)} bars')
    except Exception as e:
        print(f'    Error: {e}')

if all_dfs:
    combined = pd.concat(all_dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    
    # Convert to engine format
    ohlc = pd.DataFrame()
    ohlc['time'] = combined.index
    ohlc['open'] = combined['open'].values
    ohlc['high'] = combined['high'].values
    ohlc['low'] = combined['low'].values
    ohlc['close'] = combined['close'].values
    ohlc['volume'] = combined['volume'].values
    
    ohlc.to_csv('ohlc.csv', index=False)
    print(f'\nSaved {len(ohlc)} total bars to ohlc.csv')
    print(f'Date range: {ohlc["time"].min()} to {ohlc["time"].max()}')
else:
    print('No data retrieved')
