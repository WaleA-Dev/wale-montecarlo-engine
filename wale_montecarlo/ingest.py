"""
Universal trade-list CSV ingestion.

Auto-detects and parses:
  1. TradingView "List of trades" exports (two rows per trade: Entry + Exit)
  2. Native format (entry_time, exit_time, entry_price, exit_price, pnl, ...)
  3. Generic broker exports (any CSV with a recognizable P&L column)

Design goals:
  - Zero configuration: drag a file in, it loads or it explains exactly why not.
  - Tolerant of BOMs, currency symbols, thousands separators, parenthesized
    negatives, blank lines, and open (incomplete) trades.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .models import Trade, TradeSide


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------

@dataclass
class TradeData:
    """Parsed trade list plus everything the analysis engine needs."""
    pnls: List[float]
    entry_times: List[Optional[datetime]]
    exit_times: List[Optional[datetime]]
    entry_prices: List[Optional[float]]
    exit_prices: List[Optional[float]]
    qtys: List[Optional[float]]
    sides: List[str]                      # "long" / "short"
    notionals: List[Optional[float]]      # entry_price * qty when known
    source_format: str = "unknown"        # tradingview | native | generic
    symbol: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.pnls)

    @property
    def has_dates(self) -> bool:
        return any(t is not None for t in self.exit_times)

    @property
    def median_notional(self) -> Optional[float]:
        vals = sorted(n for n in self.notionals if n)
        if not vals:
            return None
        mid = len(vals) // 2
        return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2

    def to_trades(self) -> List[Trade]:
        """Convert to legacy Trade objects for existing engine modules."""
        out = []
        fallback = datetime(2000, 1, 1)
        for i in range(len(self.pnls)):
            out.append(Trade(
                entry_time=self.entry_times[i] or fallback,
                exit_time=self.exit_times[i] or self.entry_times[i] or fallback,
                entry_price=self.entry_prices[i] or 0.0,
                exit_price=self.exit_prices[i] or 0.0,
                pnl=self.pnls[i],
                qty=self.qtys[i] or 1.0,
                side=TradeSide.SHORT if self.sides[i] == "short" else TradeSide.LONG,
                trade_id=i,
            ))
        return out


class IngestError(ValueError):
    """Raised when a file cannot be parsed as a trade list."""


# --------------------------------------------------------------------------
# Low-level parsing helpers
# --------------------------------------------------------------------------

_DT_FORMATS = [
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M", "%Y-%m-%d", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
    "%m/%d/%Y", "%d/%m/%Y %H:%M", "%m/%d/%y %H:%M", "%m/%d/%y",
    "%b %d, %Y %H:%M", "%b %d, %Y",
]


def parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    for fmt in _DT_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    # ISO with timezone (strip it)
    try:
        return datetime.fromisoformat(s.replace("Z", "")).replace(tzinfo=None)
    except ValueError:
        return None


_NUM_CLEAN = re.compile(r"[,$\s%]")


def parse_num(s: Optional[str]) -> Optional[float]:
    """Parse '($1,234.56)' / '1,234.56' / '-12.3%' style numbers."""
    if s is None:
        return None
    s = str(s).strip()
    if not s or s.lower() in ("n/a", "na", "-", "--", "null", "none", ""):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = _NUM_CLEAN.sub("", s.strip("()"))
    if not s or s in ("-",):
        return None
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return None


def _norm(h: str) -> str:
    """Normalize a header: strip BOM, lowercase, collapse non-alphanumerics."""
    return re.sub(r"[^a-z0-9%#]+", " ", h.replace("﻿", "").lower()).strip()


def _find_col(headers: Dict[str, str], *candidates: str) -> Optional[str]:
    """Find original header name whose normalized form matches or contains a candidate."""
    for cand in candidates:
        if cand in headers:
            return headers[cand]
    for cand in candidates:
        for norm, orig in headers.items():
            if cand in norm:
                return orig
    return None


def _read_rows(text: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read CSV text into row dicts. Returns (rows, original_headers)."""
    # Sniff delimiter (comma, semicolon, tab)
    sample = text[:4096]
    delim = ","
    if sample.count(";") > sample.count(",") and sample.count(";") > 2:
        delim = ";"
    elif sample.count("\t") > sample.count(",") and sample.count("\t") > 2:
        delim = "\t"
    reader = csv.DictReader(io.StringIO(text), delimiter=delim)
    if not reader.fieldnames:
        raise IngestError("File appears to be empty.")
    headers = [h or "" for h in reader.fieldnames]
    rows = [row for row in reader if any((v or "").strip() for v in row.values())]
    return rows, headers


# --------------------------------------------------------------------------
# Format detectors / parsers
# --------------------------------------------------------------------------

def _parse_tradingview(rows: List[Dict[str, str]], headers: List[str]) -> TradeData:
    """
    TradingView "List of trades" export. Two rows per trade number:
    one with Type like "Entry long", one with "Exit long".
    P&L columns repeat on both rows.
    """
    hmap = {_norm(h): h for h in headers}
    col_trade = _find_col(hmap, "trade #", "trade")
    col_type = _find_col(hmap, "type")
    col_time = _find_col(hmap, "date and time", "date time", "date/time", "date")
    col_signal = _find_col(hmap, "signal")
    col_price = _find_col(hmap, "price usd", "price")
    col_qty = _find_col(hmap, "position size qty", "contracts", "qty", "quantity", "position size")
    col_pnl = _find_col(hmap, "net p&l usd", "net p l usd", "profit usd", "net p l", "p&l usd", "profit")
    # Avoid grabbing the percent column
    if col_pnl and "%" in col_pnl:
        col_pnl = None
        for norm, orig in hmap.items():
            if ("p l" in norm or "profit" in norm) and "%" not in norm and "cum" not in norm:
                col_pnl = orig
                break
    if not (col_trade and col_type and col_pnl):
        raise IngestError("Not a TradingView trade list (missing Trade #/Type/P&L columns).")

    groups: Dict[str, Dict[str, Dict[str, str]]] = {}
    order: List[str] = []
    for row in rows:
        tid = (row.get(col_trade) or "").strip()
        if not tid:
            continue
        typ = (row.get(col_type) or "").strip().lower()
        kind = "entry" if typ.startswith("entry") else "exit" if typ.startswith("exit") else None
        if kind is None:
            continue
        if tid not in groups:
            groups[tid] = {}
            order.append(tid)
        groups[tid][kind] = row
        if kind == "entry":
            groups[tid]["side_row"] = row  # side comes from entry type

    data = TradeData([], [], [], [], [], [], [], [], source_format="tradingview")
    skipped_open = 0
    open_unrealized = 0.0
    for tid in order:
        g = groups[tid]
        entry, exit_ = g.get("entry"), g.get("exit")
        if entry is None or exit_ is None:
            skipped_open += 1
            continue
        pnl = parse_num(exit_.get(col_pnl)) or parse_num(entry.get(col_pnl))
        exit_sig = ((exit_.get(col_signal) or "") if col_signal else "").strip().lower()
        if exit_sig == "open":
            # Still-open position: TradingView exports it as an "Exit" row
            # marked Signal=Open with UNREALIZED mark-to-market P&L.
            # Counting it as a completed trade would corrupt every statistic.
            skipped_open += 1
            open_unrealized += pnl or 0.0
            continue
        if pnl is None:
            data.warnings.append(f"Trade {tid}: missing P&L, skipped.")
            continue
        typ = (entry.get(col_type) or "").lower()
        side = "short" if "short" in typ else "long"
        eprice = parse_num(entry.get(col_price)) if col_price else None
        xprice = parse_num(exit_.get(col_price)) if col_price else None
        qty = parse_num(entry.get(col_qty)) if col_qty else None
        data.pnls.append(pnl)
        data.entry_times.append(parse_dt(entry.get(col_time)) if col_time else None)
        data.exit_times.append(parse_dt(exit_.get(col_time)) if col_time else None)
        data.entry_prices.append(eprice)
        data.exit_prices.append(xprice)
        data.qtys.append(qty)
        data.sides.append(side)
        data.notionals.append(eprice * qty if (eprice and qty) else None)

    if skipped_open:
        note = (f" (unrealized P&L ${open_unrealized:,.2f} not counted)"
                if open_unrealized else "")
        data.warnings.append(
            f"{skipped_open} open/incomplete trade(s) excluded{note}.")
    if not data.pnls:
        raise IngestError("TradingView file recognized but no completed trades found.")
    _sort_by_exit(data)
    return data


def _parse_native(rows: List[Dict[str, str]], headers: List[str]) -> TradeData:
    """Native format: entry_time, exit_time, entry_price, exit_price, pnl, side, qty."""
    hmap = {_norm(h): h for h in headers}
    col_et = _find_col(hmap, "entry time", "entry date", "open time", "open date")
    col_xt = _find_col(hmap, "exit time", "exit date", "close time", "close date")
    col_pnl = _find_col(hmap, "pnl", "net p&l", "profit loss", "profit/loss", "profit", "p l", "gain")
    if not (col_et and col_xt and col_pnl):
        raise IngestError("Not native format.")
    col_ep = _find_col(hmap, "entry price", "open price")
    col_xp = _find_col(hmap, "exit price", "close price")
    col_qty = _find_col(hmap, "qty", "quantity", "contracts", "size", "shares")
    col_side = _find_col(hmap, "side", "direction", "type")
    col_sym = _find_col(hmap, "symbol", "ticker")

    data = TradeData([], [], [], [], [], [], [], [], source_format="native")
    for i, row in enumerate(rows):
        pnl = parse_num(row.get(col_pnl))
        if pnl is None:
            data.warnings.append(f"Row {i + 2}: unparseable P&L, skipped.")
            continue
        side_raw = ((row.get(col_side) or "long") if col_side else "long").lower()
        eprice = parse_num(row.get(col_ep)) if col_ep else None
        qty = parse_num(row.get(col_qty)) if col_qty else None
        data.pnls.append(pnl)
        data.entry_times.append(parse_dt(row.get(col_et)))
        data.exit_times.append(parse_dt(row.get(col_xt)))
        data.entry_prices.append(eprice)
        data.exit_prices.append(parse_num(row.get(col_xp)) if col_xp else None)
        data.qtys.append(qty)
        data.sides.append("short" if ("short" in side_raw or "sell" in side_raw) else "long")
        data.notionals.append(eprice * qty if (eprice and qty) else None)
        if col_sym and not data.symbol:
            data.symbol = (row.get(col_sym) or "").strip() or None

    if not data.pnls:
        raise IngestError("No valid trades found in native-format file.")
    _sort_by_exit(data)
    return data


def _parse_generic(rows: List[Dict[str, str]], headers: List[str]) -> TradeData:
    """Last resort: find one P&L-like column, one optional date column."""
    hmap = {_norm(h): h for h in headers}
    col_pnl = _find_col(
        hmap, "net p&l", "net p l", "realized pnl", "realized p l", "pnl",
        "profit loss", "profit/loss", "p l", "profit", "gain loss", "gain", "amount",
    )
    if col_pnl and "%" in col_pnl:
        col_pnl = None
    if not col_pnl:
        found = ", ".join(h for h in headers if h)
        raise IngestError(
            "Could not find a P&L column. Columns found: "
            f"[{found}]. Expected something like 'pnl', 'profit', 'Net P&L USD'."
        )
    col_date = _find_col(hmap, "exit time", "close time", "date and time", "date time",
                         "exit date", "close date", "date", "time")
    col_side = _find_col(hmap, "side", "direction")
    col_qty = _find_col(hmap, "qty", "quantity", "contracts", "size", "shares")

    data = TradeData([], [], [], [], [], [], [], [], source_format="generic")
    bad = 0
    for row in rows:
        pnl = parse_num(row.get(col_pnl))
        if pnl is None:
            bad += 1
            continue
        side_raw = ((row.get(col_side) or "long") if col_side else "long").lower()
        data.pnls.append(pnl)
        dt = parse_dt(row.get(col_date)) if col_date else None
        data.entry_times.append(None)
        data.exit_times.append(dt)
        data.entry_prices.append(None)
        data.exit_prices.append(None)
        data.qtys.append(parse_num(row.get(col_qty)) if col_qty else None)
        data.sides.append("short" if ("short" in side_raw or "sell" in side_raw) else "long")
        data.notionals.append(None)
    if bad:
        data.warnings.append(f"{bad} row(s) had unparseable P&L and were skipped.")
    if not data.pnls:
        raise IngestError(f"Found P&L column '{col_pnl}' but no parseable values.")
    _sort_by_exit(data)
    return data


def _sort_by_exit(data: TradeData) -> None:
    """Sort all parallel arrays chronologically by exit (then entry) time."""
    n = len(data.pnls)
    if n == 0 or not data.has_dates:
        return
    far_future = datetime(9999, 1, 1)

    def key(i: int):
        return (data.exit_times[i] or data.entry_times[i] or far_future,
                data.entry_times[i] or far_future)

    idx = sorted(range(n), key=key)
    for name in ("pnls", "entry_times", "exit_times", "entry_prices",
                 "exit_prices", "qtys", "sides", "notionals"):
        arr = getattr(data, name)
        setattr(data, name, [arr[i] for i in idx])


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def load_trades_text(text: str) -> TradeData:
    """Parse trade list from CSV text, auto-detecting the format."""
    text = text.replace("﻿", "")
    rows, headers = _read_rows(text)
    if not rows:
        raise IngestError("No data rows found in file.")

    norm_headers = {_norm(h) for h in headers}

    # TradingView: has a Type column with Entry/Exit values and a trade-number column
    if any("type" == h or h.startswith("type") for h in norm_headers) and \
       any("trade" in h for h in norm_headers):
        type_col = next(h for h in headers if _norm(h).startswith("type"))
        vals = {(r.get(type_col) or "").strip().lower() for r in rows[:20]}
        if any(v.startswith("entry") or v.startswith("exit") for v in vals):
            return _parse_tradingview(rows, headers)

    # Native: explicit entry/exit time columns
    if any("entry" in h and "time" in h for h in norm_headers) or \
       any("entry" in h and "date" in h for h in norm_headers):
        try:
            return _parse_native(rows, headers)
        except IngestError:
            pass

    return _parse_generic(rows, headers)


def load_trades_file(path: str) -> TradeData:
    """Parse trade list from a CSV file on disk."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        return load_trades_text(f.read())
