"""
File I/O operations for the Monte Carlo backtesting engine.

This module handles all file reading and writing, including:
- Loading backtest data (trades, equity curve, OHLC)
- Saving/loading permutation results
- Atomic writes for crash safety
- Resume support via metrics_compact.csv

CRITICAL: metrics_compact.csv is the SOURCE OF TRUTH for resume operations.
Always use atomic writes (tmp + replace) for crash safety.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Tuple
import csv
import re

from .models import (
    Trade, TradeSide, EquityCurve, EquityPoint, OHLCData, OHLCBar,
    CellConfig, PermutationResult, CellSummary, BaselineMetrics,
    RunConfig, CellProgress, RunProgress
)


def parse_datetime(s: str) -> datetime:
    """Parse datetime string in various formats."""
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: {s}")


def load_trade_list(path: str) -> List[Trade]:
    """
    Load trade list from CSV file.

    Expected columns: entry_time, exit_time, entry_price, exit_price, pnl, qty, side

    Args:
        path: Path to trade_list.csv

    Returns:
        List of Trade objects sorted by entry_time
    """
    trades = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Normalize column names (lowercase, strip spaces)
            row = {k.lower().strip(): v for k, v in row.items()}

            # Parse side
            side_str = row.get('side', 'long').lower().strip()
            side = TradeSide.LONG if side_str in ('long', 'buy', '1') else TradeSide.SHORT

            trade = Trade(
                entry_time=parse_datetime(row['entry_time']),
                exit_time=parse_datetime(row['exit_time']),
                entry_price=float(row['entry_price']),
                exit_price=float(row['exit_price']),
                pnl=float(row['pnl']),
                qty=float(row.get('qty', row.get('quantity', 1))),
                side=side,
                trade_id=i
            )
            trades.append(trade)

    # Sort by entry time
    trades.sort(key=lambda t: t.entry_time)
    return trades


def load_equity_curve(path: str) -> EquityCurve:
    """
    Load equity curve from CSV file.

    Expected columns: time, equity

    Args:
        path: Path to equity_curve.csv

    Returns:
        EquityCurve object
    """
    points = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.lower().strip(): v for k, v in row.items()}
            point = EquityPoint(
                time=parse_datetime(row['time']),
                equity=float(row['equity'])
            )
            points.append(point)

    # Sort by time
    points.sort(key=lambda p: p.time)

    # Initial equity is first point or default
    initial = points[0].equity if points else 100000.0

    return EquityCurve(points=points, initial_equity=initial)


def load_ohlc_data(path: str) -> OHLCData:
    """
    Load OHLC price data from CSV file.

    Expected columns: time, open, high, low, close, [volume]

    Args:
        path: Path to OHLC CSV file

    Returns:
        OHLCData object
    """
    bars = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.lower().strip(): v for k, v in row.items()}
            bar = OHLCBar(
                time=parse_datetime(row.get('time', row.get('date', row.get('datetime')))),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']) if 'volume' in row else None
            )
            bars.append(bar)

    # Sort by time
    bars.sort(key=lambda b: b.time)

    return OHLCData(bars=bars)


def load_baseline_report(path: str) -> Optional[BaselineMetrics]:
    """
    Parse baseline metrics from step1_report.txt.

    This file contains the original backtest metrics used for p-value calculation.

    Args:
        path: Path to step1_report.txt

    Returns:
        BaselineMetrics object or None if file not found/parseable
    """
    if not os.path.exists(path):
        return None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse metrics using regex patterns
        def extract_float(pattern: str) -> float:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(',', '').replace('%', ''))
            return 0.0

        def extract_int(pattern: str) -> int:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return int(match.group(1).replace(',', ''))
            return 0

        return BaselineMetrics(
            total_return_pct=extract_float(r'total[_ ]?return[:\s]+([+-]?[\d,.]+)%?'),
            max_drawdown_pct=extract_float(r'max[_ ]?drawdown[:\s]+([+-]?[\d,.]+)%?'),
            profit_factor=extract_float(r'profit[_ ]?factor[:\s]+([+-]?[\d,.]+)'),
            worst_month_pct=extract_float(r'worst[_ ]?month[:\s]+([+-]?[\d,.]+)%?'),
            sharpe_ratio=extract_float(r'sharpe[_ ]?ratio[:\s]+([+-]?[\d,.]+)'),
            n_trades=extract_int(r'(?:total[_ ]?)?trades[:\s]+(\d+)')
        )
    except Exception:
        return None


def atomic_write(path: str, content: str) -> None:
    """
    Write content to file atomically using tmp + replace pattern.

    This ensures crash safety: either the old file exists or the new file
    exists completely, never a partial write.

    Args:
        path: Target file path
        content: Content to write
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        # Atomic replace
        shutil.move(tmp_path, path)
    except Exception:
        # Clean up temp file on error
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def atomic_write_json(path: str, data: dict) -> None:
    """Write JSON data atomically."""
    content = json.dumps(data, indent=2, default=str)
    atomic_write(path, content)


def save_metrics_compact(path: str, results: List[PermutationResult], append: bool = False) -> None:
    """
    Save permutation results to metrics_compact.csv.

    This is the SOURCE OF TRUTH for resume operations.

    Args:
        path: Path to metrics_compact.csv
        results: List of PermutationResult objects
        append: If True, append to existing file; if False, overwrite
    """
    if not results:
        return

    fieldnames = [
        'perm_index', 'total_return_pct', 'max_drawdown_pct', 'profit_factor',
        'worst_month_pct', 'sharpe_ratio', 'win_rate', 'n_trades', 'total_pnl'
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = 'a' if append and path.exists() else 'w'
    write_header = not (append and path.exists())

    # Use atomic write for full rewrites, direct write for appends
    if not append:
        lines = []
        lines.append(','.join(fieldnames))
        for r in results:
            lines.append(','.join(str(r.to_dict()[f]) for f in fieldnames))
        atomic_write(path, '\n'.join(lines) + '\n')
    else:
        with open(path, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for r in results:
                writer.writerow(r.to_dict())


def load_metrics_compact(path: str) -> Tuple[List[PermutationResult], int]:
    """
    Load permutation results from metrics_compact.csv.

    Performs deduplication by perm_index, keeping first occurrence.
    This is critical for resume correctness.

    Args:
        path: Path to metrics_compact.csv

    Returns:
        Tuple of (list of unique PermutationResult, max_perm_index seen)
    """
    if not os.path.exists(path):
        return [], -1

    results = []
    seen_indices = set()
    max_index = -1

    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            perm_index = int(row['perm_index'])

            # Dedupe: keep first occurrence only
            if perm_index in seen_indices:
                continue
            seen_indices.add(perm_index)

            max_index = max(max_index, perm_index)
            results.append(PermutationResult.from_dict(row))

    # Sort by perm_index for consistency
    results.sort(key=lambda r: r.perm_index)

    return results, max_index


def save_cell_summary(path: str, summary: CellSummary) -> None:
    """
    Save cell summary statistics to JSON.

    Args:
        path: Path to summary.json
        summary: CellSummary object
    """
    atomic_write_json(path, summary.to_dict())


def load_cell_summary(path: str) -> Optional[CellSummary]:
    """Load cell summary from JSON."""
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_run_manifest(path: str, config: RunConfig, cells: List[CellConfig]) -> None:
    """
    Save run configuration manifest.

    Args:
        path: Path to run_manifest.json
        config: RunConfig object
        cells: List of CellConfig objects
    """
    manifest = {
        "run_name": os.path.basename(config.output_dir),
        "created": datetime.now().isoformat(),
        "config": {
            "input_dir": config.input_dir,
            "n_per_cell": config.n_per_cell,
            "n_jobs": config.n_jobs,
            "fixed_delay": config.fixed_delay,
            "grid_filters": config.grid_filters,
        },
        "grid": {
            "total_cells": len(cells),
            "p_skip_values": config.p_skip_values,
            "slip_values": config.slip_values,
            "delay_values": config.delay_values if config.fixed_delay is None else [config.fixed_delay],
            "shuffle_modes": [m.value for m in config.shuffle_modes],
            "bootstrap_modes": [m.value for m in config.bootstrap_modes],
            "block_len_values": config.block_len_values,
        },
        "cells": [c.to_dict() for c in cells]
    }
    atomic_write_json(path, manifest)


def save_progress(path: str, progress: RunProgress, cell_progress: List[CellProgress]) -> None:
    """
    Save current run progress.

    Args:
        path: Path to progress.csv
        progress: Overall run progress
        cell_progress: Per-cell progress
    """
    lines = ['cell_id,status,n_completed,n_target,last_update,error_message']
    for cp in cell_progress:
        lines.append(
            f'{cp.cell_id},{cp.status},{cp.n_completed},{cp.n_target},'
            f'{cp.last_update.isoformat() if cp.last_update else ""},'
            f'{cp.error_message or ""}'
        )
    atomic_write(path, '\n'.join(lines) + '\n')


def save_heartbeat(path: str, progress: RunProgress) -> None:
    """
    Save heartbeat file for monitoring.

    Args:
        path: Path to heartbeat.json
        progress: Current run progress
    """
    heartbeat = {
        "run_name": progress.run_name,
        "timestamp": datetime.now().isoformat(),
        "cells_total": progress.cells_total,
        "cells_completed": progress.cells_completed,
        "cells_running": progress.cells_running,
        "cells_failed": progress.cells_failed,
        "perms_completed": progress.perms_completed,
        "perms_total": progress.perms_total,
        "pct_complete": (progress.perms_completed / progress.perms_total * 100
                        if progress.perms_total > 0 else 0),
        "elapsed_seconds": (datetime.now() - progress.start_time).total_seconds()
    }
    atomic_write_json(path, heartbeat)


def save_done_sentinel(path: str) -> None:
    """Create DONE.txt sentinel file indicating run completion."""
    atomic_write(path, f"Completed at {datetime.now().isoformat()}\n")


def ensure_output_structure(output_dir: str) -> Dict[str, str]:
    """
    Create output directory structure for a Monte Carlo run.

    Returns dict of key paths.
    """
    output_dir = Path(output_dir)
    dirs = {
        "root": str(output_dir),
        "aggregated": str(output_dir / "aggregated"),
        "analysis": str(output_dir / "aggregated" / "analysis"),
        "tables": str(output_dir / "aggregated" / "analysis" / "tables"),
        "per_cell": str(output_dir / "per_cell"),
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    return dirs


def get_cell_dir(output_dir: str, cell_id: str) -> str:
    """Get path to a specific cell's output directory."""
    return str(Path(output_dir) / "per_cell" / f"cell_{cell_id}")


def ensure_cell_dir(output_dir: str, cell_id: str) -> str:
    """Create and return path to a cell's output directory."""
    cell_dir = get_cell_dir(output_dir, cell_id)
    os.makedirs(cell_dir, exist_ok=True)
    return cell_dir
