"""Tests for universal CSV ingestion (wale_montecarlo.ingest)."""

import pytest

from wale_montecarlo.ingest import (
    IngestError, load_trades_text, parse_dt, parse_num,
)


NATIVE_CSV = """entry_time,exit_time,entry_price,exit_price,pnl,symbol,side,quantity
2023-01-19 12:30:00,2023-01-23 15:30:00,100.0,110.0,1000.0,NQ,long,1
2023-02-10 12:30:00,2023-03-23 09:30:00,110.0,105.0,-500.0,NQ,long,1
"""

# Real TradingView layout: exit row listed before entry row, BOM, P&L on both
TV_CSV = "﻿" + """Trade #,Type,Date and time,Signal,Price USD,Position size (qty),Position size (value),Net P&L USD,Net P&L %,Favorable excursion USD,Favorable excursion %,Adverse excursion USD,Adverse excursion %,Cumulative P&L USD,Cumulative P&L %
2,Exit short,2023-02-01 10:30,Stop,95.0,10,1000.0,-50.0,-5.0,1,1,-2,-2,-30.0,-3
2,Entry short,2023-01-20 09:30,Short,100.0,10,1000.0,-50.0,-5.0,1,1,-2,-2,-30.0,-3
1,Exit long,2023-01-15 21:30,Trail,12.46,100,1246.0,20.0,1.6,4,4,-5,-5,20.0,2
1,Entry long,2023-01-14 03:30,Long,12.01,100,1201.0,20.0,1.6,4,4,-5,-5,20.0,2
"""

GENERIC_CSV = """Date,Description,Profit/Loss
01/05/2023,Trade A,"$1,250.00"
01/09/2023,Trade B,($400.00)
01/12/2023,Trade C,300
"""


class TestParsers:
    def test_parse_num_currency(self):
        assert parse_num("$1,234.56") == 1234.56

    def test_parse_num_paren_negative(self):
        assert parse_num("($400.00)") == -400.0

    def test_parse_num_junk(self):
        assert parse_num("N/A") is None
        assert parse_num("") is None

    def test_parse_dt_tv_format(self):
        dt = parse_dt("2023-01-14 03:30")
        assert dt is not None and dt.hour == 3


class TestNative:
    def test_loads(self):
        d = load_trades_text(NATIVE_CSV)
        assert d.source_format == "native"
        assert len(d) == 2
        assert d.pnls == [1000.0, -500.0]
        assert d.notionals[0] == 100.0

    def test_to_trades_roundtrip(self):
        trades = load_trades_text(NATIVE_CSV).to_trades()
        assert trades[0].pnl == 1000.0
        assert trades[0].side.value == "long"


class TestTradingView:
    def test_pairs_entry_exit_rows(self):
        d = load_trades_text(TV_CSV)
        assert d.source_format == "tradingview"
        assert len(d) == 2

    def test_sorted_by_exit_time(self):
        d = load_trades_text(TV_CSV)
        assert d.exit_times[0] < d.exit_times[1]
        assert d.pnls == [20.0, -50.0]

    def test_side_detection(self):
        d = load_trades_text(TV_CSV)
        assert d.sides == ["long", "short"]

    def test_notional_from_price_qty(self):
        d = load_trades_text(TV_CSV)
        assert d.notionals[0] == pytest.approx(12.01 * 100)

    def test_open_trade_skipped(self):
        # Entry with no exit row = open position; must be excluded with warning
        open_tail = "3,Entry long,2023-03-01 09:30,Long,50.0,10,500.0,,,,,,,,\n"
        d = load_trades_text(TV_CSV + open_tail)
        assert len(d) == 2
        assert any("open" in w.lower() or "incomplete" in w.lower()
                   for w in d.warnings)

    def test_open_signal_exit_row_excluded(self):
        # TV exports still-open positions as an Exit row with Signal=Open and
        # UNREALIZED P&L - must not be counted as a completed trade
        open_pair = (
            "3,Exit long,2023-03-05 09:30,Open,60.0,10,600.0,110129.91,5.0,1,1,-2,-2,99.0,9\n"
            "3,Entry long,2023-03-01 09:30,Long,50.0,10,500.0,110129.91,5.0,1,1,-2,-2,99.0,9\n"
        )
        d = load_trades_text(TV_CSV + open_pair)
        assert len(d) == 2
        assert 110129.91 not in d.pnls
        assert any("unrealized" in w.lower() for w in d.warnings)


class TestGeneric:
    def test_finds_pnl_column(self):
        d = load_trades_text(GENERIC_CSV)
        assert d.source_format == "generic"
        assert d.pnls == [1250.0, -400.0, 300.0]

    def test_unparseable_raises_with_columns_listed(self):
        with pytest.raises(IngestError, match="Columns found"):
            load_trades_text("a,b,c\n1,2,3\n")

    def test_empty_file_raises(self):
        with pytest.raises(IngestError):
            load_trades_text("")
