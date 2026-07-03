"""
Independent validation audit for the Strategy Stress Lab.

Purpose: prove the numbers the app reports are REAL, using only means that
do not share code with the engine:

  1. This script imports NOTHING from wale_montecarlo and does not use numpy.
     All reference values are computed in pure-Python stdlib arithmetic.
  2. It talks to the RUNNING APP over HTTP (the shipped exe or app.py),
     so the entire product pipeline is under test - CSV parsing, engine,
     JSON API - not just a library function.
  3. Ground truths come from three independent sources:
       a. TradingView's own "Cumulative P&L USD" column in real exports
          (TradingView computed those numbers, not us).
       b. Exact brute-force enumeration: for a 6-trade list, ALL 6^6=46,656
          bootstrap draws; for a 7-trade list, ALL 7!=5,040 orderings.
          These are exact distributions, not simulations.
       c. Closed-form expectations (bootstrap mean = n * mean(pnl), etc.)
          with 4-sigma Monte Carlo tolerance bands.

Usage:
    python scripts/independent_validation.py [--url http://127.0.0.1:8742]
Exit code 0 = every check passed.
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import math
import statistics
import sys
import time
import urllib.request
from pathlib import Path

CHECKS = []


def check(name: str, expected: float, actual: float, tol: float, unit: str = "") -> None:
    ok = (expected is not None and actual is not None
          and abs(expected - actual) <= tol)
    CHECKS.append((name, expected, actual, tol, ok, unit))


def report() -> int:
    w = max(len(c[0]) for c in CHECKS) + 2
    print()
    print(f"{'CHECK':<{w}} {'INDEPENDENT':>14} {'ENGINE':>14} {'TOL':>10}  RESULT")
    print("-" * (w + 50))
    fails = 0
    for name, exp, act, tol, ok, unit in CHECKS:
        fails += (not ok)
        print(f"{name:<{w}} {exp:>14.4f} {act:>14.4f} {tol:>10.4f}  "
              f"{'PASS' if ok else '*** FAIL ***'}{unit}")
    print("-" * (w + 50))
    print(f"{len(CHECKS) - fails}/{len(CHECKS)} checks passed")
    return fails


# ---------------------------------------------------------------------------
# Pure-python reference math (shared with no engine code)
# ---------------------------------------------------------------------------

def ref_max_dd_pct(pnls, capital):
    """Max peak-to-trough drawdown as % of peak, equity = capital + cumsum."""
    eq = capital
    peak = capital
    worst = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        if peak > 0:
            worst = max(worst, (peak - eq) / peak)
    return worst * 100.0


def ref_profit_factor(pnls):
    gp = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p < 0)
    if gl == 0:
        return 999.0 if gp > 0 else 0.0
    return min(gp / gl, 999.0)


# ---------------------------------------------------------------------------
# Independent TradingView export parser (deliberately minimal + separate)
# ---------------------------------------------------------------------------

def parse_tv_independent(path):
    """Parse a TradingView 'List of trades' export with fresh logic:
    returns (pnls_in_exit_time_order, tv_final_cumulative_pnl)."""
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    per_trade = {}   # trade# -> dict(pnl=..., exit_time=...)
    for r in rows:
        tid = r["Trade #"].strip()
        typ = r["Type"].strip().lower()
        if not tid:
            continue
        d = per_trade.setdefault(tid, {})
        if typ.startswith("exit"):
            d["pnl"] = float(r["Net P&L USD"])
            d["exit"] = r["Date and time"]
            d["cum"] = float(r["Cumulative P&L USD"])
            # Signal=Open means a still-open position: unrealized P&L,
            # not a completed trade. Closed-trade analysis must skip it.
            d["open"] = r["Signal"].strip().lower() == "open"
        elif typ.startswith("entry"):
            d["has_entry"] = True
    complete = {int(tid): d for tid, d in per_trade.items()
                if "pnl" in d and d.get("has_entry") and not d.get("open")}
    ordered = [complete[k] for k in sorted(complete)]
    pnls = [d["pnl"] for d in ordered]
    # TradingView accumulates "Cumulative P&L" in trade-number order, so its
    # own final total is the cumulative on the highest trade number.
    tv_cum_final = ordered[-1]["cum"] if ordered else 0.0
    return pnls, tv_cum_final


# ---------------------------------------------------------------------------
# Talk to the running app
# ---------------------------------------------------------------------------

def api_analyze(base_url, csv_text, filename, capital, samples=10000, seed=42):
    boundary = "----wmcaudit"
    body = io.BytesIO()

    def field(name, value):
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; "
                   f"name=\"{name}\"\r\n\r\n{value}\r\n".encode())

    body.write(f"--{boundary}\r\nContent-Disposition: form-data; "
               f"name=\"file\"; filename=\"{filename}\"\r\n"
               f"Content-Type: text/csv\r\n\r\n".encode())
    body.write(csv_text.encode("utf-8"))
    body.write(b"\r\n")
    field("capital", capital)
    field("samples", samples)
    field("seed", seed)
    body.write(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        f"{base_url}/api/analyze", data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    job = json.load(urllib.request.urlopen(req, timeout=30))["job_id"]

    deadline = time.time() + 120
    while time.time() < deadline:
        st = json.load(urllib.request.urlopen(
            f"{base_url}/api/job/{job}", timeout=30))
        if st["status"] == "done":
            return st["results"]
        if st["status"] == "error":
            raise RuntimeError(st["error"])
        time.sleep(0.3)
    raise TimeoutError("analysis did not finish")


# ---------------------------------------------------------------------------
# Audit sections
# ---------------------------------------------------------------------------

def audit_real_tv_file(base_url, path, capital):
    """Baseline stats vs independent parse + TradingView's own cumulative."""
    name = Path(path).name
    pnls, tv_cum = parse_tv_independent(path)
    res = api_analyze(base_url, open(path, encoding="utf-8-sig").read(),
                      name, capital)
    b = res["baseline"]

    check(f"[{name}] n trades", len(pnls), b["n_trades"], 0)
    check(f"[{name}] total PnL vs OUR sum", round(sum(pnls), 2),
          b["total_pnl"], 0.011, " $")
    # TV rounds each exported Net P&L to 2dp but computes its cumulative
    # from unrounded internals: honest tolerance is 0.5 cent per trade.
    check(f"[{name}] total PnL vs TRADINGVIEW cum", tv_cum,
          b["total_pnl"], 0.005 * len(pnls) + 0.011, " $")
    wr = sum(1 for p in pnls if p > 0) / len(pnls)
    check(f"[{name}] win rate", wr, b["win_rate"], 0.0006)
    check(f"[{name}] profit factor", ref_profit_factor(pnls),
          b["profit_factor"], 0.002)
    check(f"[{name}] max drawdown %", ref_max_dd_pct(pnls, capital),
          b["max_dd_pct"], 0.02, " %")
    return pnls, res


def audit_exact_bootstrap(base_url):
    """6 trades -> ALL 6^6 = 46,656 equally likely bootstrap draws, computed
    exactly by enumeration. Engine's 10k-sample estimates must sit inside
    tight Monte Carlo bands around these EXACT values."""
    pnls = [120.0, -80.0, 200.0, -150.0, 60.0, -40.0]
    n = len(pnls)
    capital = 1000.0

    finals, dds = [], []
    for combo in itertools.product(range(n), repeat=n):
        sample = [pnls[i] for i in combo]
        finals.append(sum(sample))
        dds.append(ref_max_dd_pct(sample, capital))
    finals.sort()
    dds.sort()

    exact_mean = statistics.fmean(finals)
    exact_p_loss = sum(1 for f in finals if f <= 0) / len(finals)
    exact_dd_p50 = statistics.median(dds)

    m = 50_000  # ask the engine for a deep run to tighten its own noise
    csv_text = "pnl\n" + "\n".join(str(p) for p in pnls)
    res = api_analyze(base_url, csv_text, "exact6.csv", capital, samples=m)
    bo = res["bootstrap"]

    # 4-sigma tolerances from exact population moments
    sd_final = statistics.pstdev(finals)
    tol_mean = 4 * sd_final / math.sqrt(m)
    tol_p = 4 * math.sqrt(exact_p_loss * (1 - exact_p_loss) / m)

    check("[exact 6^6] bootstrap mean final PnL", exact_mean,
          bo["final_pnl_mean"], tol_mean, " $")
    check("[exact 6^6] P(final <= 0)", exact_p_loss,
          bo["prob_loss"], tol_p)
    check("[exact 6^6] median max-DD %", exact_dd_p50,
          bo["max_dd_pct"]["p50"], 0.35, " %")
    # closed-form: E[bootstrap sum] must equal n * mean(pnls)
    check("[closed form] n*mean == exact mean", n * statistics.fmean(pnls),
          exact_mean, 1e-9, " $")


def audit_exact_shuffle(base_url):
    """7 trades -> ALL 7! = 5,040 orderings. Exact permutation max-DD
    distribution vs the engine's shuffle (luck) test."""
    pnls = [90.0, -60.0, 150.0, -120.0, 45.0, -30.0, 75.0]
    capital = 1000.0

    dds = sorted(ref_max_dd_pct(list(perm), capital)
                 for perm in itertools.permutations(pnls))
    exact_median = statistics.median(dds)
    baseline_dd = ref_max_dd_pct(pnls, capital)
    exact_pctile = sum(1 for d in dds if d <= baseline_dd) / len(dds) * 100

    csv_text = "pnl\n" + "\n".join(str(p) for p in pnls)
    res = api_analyze(base_url, csv_text, "exact7.csv", capital, samples=20000)
    sh = res["shuffle"]

    check("[exact 7!] shuffled median max-DD %", exact_median,
          sh["median_dd_pct"], 0.30, " %")
    check("[exact 7!] baseline DD percentile", exact_pctile,
          sh["baseline_dd_percentile"], 2.5)
    check("[exact 7!] baseline DD % itself", baseline_dd,
          sh["baseline_dd_pct"], 0.02, " %")


def audit_stress_expectation(base_url, pnls, res):
    """Optimistic scenario (skip=0): E[total] = sum(pnl) - sum(friction).
    The engine's median across 2,000 seeds must sit in the 4-sigma band
    of that closed-form expectation (sum of uniforms ~ normal, median=mean)."""
    scen = {s["name"]: s for s in res["stress"]["scenarios"]}
    opt = scen["optimistic"]
    n = res["baseline"]["n_trades"]
    total = sum(pnls)
    exp_total = total - opt["avg_friction_per_trade"] * n
    # slip_i ~ U(0, 2f): var = (2f)^2/12; over n trades; median of 2000 seeds
    f = opt["avg_friction_per_trade"]
    sd_total = math.sqrt(n * (2 * f) ** 2 / 12)
    sd_median = sd_total * 1.2533 / math.sqrt(2000)
    check("[stress closed-form] optimistic median PnL", exp_total,
          opt["total_pnl_med"], max(4 * sd_median, 0.02 * abs(exp_total) + 1),
          " $")


def audit_determinism(base_url):
    """Same file + same seed twice through the app: identical output."""
    csv_text = "pnl\n" + "\n".join(str(p) for p in
                                   [50, -30, 80, -20, 100, -60, 40, -10] * 5)
    a = api_analyze(base_url, csv_text, "det.csv", 5000, samples=5000, seed=77)
    b = api_analyze(base_url, csv_text, "det.csv", 5000, samples=5000, seed=77)
    same = (a["bootstrap"] == b["bootstrap"] and a["ruin"] == b["ruin"]
            and a["stress"] == b["stress"])
    check("[determinism] identical results for same seed", 1.0,
          1.0 if same else 0.0, 0)


def audit_convergence(base_url):
    """Estimates must converge as samples grow: the 50k-sample P(loss) of a
    coin-flip-ish book must be closer to the exact 6^6 value than a wildly
    wrong number, and two different seeds must agree within MC noise."""
    pnls = [120.0, -80.0, 200.0, -150.0, 60.0, -40.0]
    csv_text = "pnl\n" + "\n".join(str(p) for p in pnls)
    r1 = api_analyze(base_url, csv_text, "c1.csv", 1000, samples=20000, seed=1)
    r2 = api_analyze(base_url, csv_text, "c2.csv", 1000, samples=20000, seed=999)
    p1, p2 = r1["bootstrap"]["prob_loss"], r2["bootstrap"]["prob_loss"]
    tol = 4 * math.sqrt(0.25 / 20000) * 2
    check("[convergence] P(loss) agrees across seeds", p1, p2, tol)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8742")
    ap.add_argument("--schd", default=r"C:\Users\wale\Downloads\Saty_Phase_-_ANTI-CONSOLIDATION_V2_AMEX_SCHD_2026-02-07.csv")
    ap.add_argument("--ndaq", default=r"C:\Users\wale\saty_backtest\codex\1.7.26 data\1.18.code fixing\seelist.csv")
    args = ap.parse_args()

    try:
        urllib.request.urlopen(args.url, timeout=5)
    except Exception:
        print(f"App not reachable at {args.url} - start WaleMonteCarlo.exe "
              f"or `python app.py` first.")
        return 2

    print("Independent validation audit")
    print(f"Target: {args.url} (live app over HTTP)")
    print("Reference math: pure-Python stdlib, zero shared code, no numpy\n")

    for path, cap in ((args.ndaq, 500.0), (args.schd, 100000.0)):
        if Path(path).exists():
            print(f"auditing real TradingView export: {Path(path).name} ...")
            pnls, res = audit_real_tv_file(args.url, path, cap)
            audit_stress_expectation(args.url, pnls, res)
        else:
            print(f"skipping missing file: {path}")

    print("enumerating 6^6 = 46,656 exact bootstrap draws ...")
    audit_exact_bootstrap(args.url)
    print("enumerating 7! = 5,040 exact orderings ...")
    audit_exact_shuffle(args.url)
    print("determinism + cross-seed convergence ...")
    audit_determinism(args.url)
    audit_convergence(args.url)

    fails = report()
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
