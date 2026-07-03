"""
Vectorized Monte Carlo analysis engine.

Single entry point: run_full_analysis(TradeData, ...) -> JSON-serializable dict.

All simulation is numpy-vectorized and chunked so memory stays bounded
regardless of trade count or sample count. This is the engine behind the
web UI and the standalone HTML report.

Statistical notes
-----------------
- Bootstrap: trades resampled with replacement (fixed position size, additive
  PnL). Answers "what other outcomes were consistent with this trade
  distribution?"
- Shuffle (luck detector): trades permuted without replacement. Total PnL is
  invariant; only path/drawdown changes. If the original ordering's max
  drawdown sits in a low percentile of the shuffled distribution, the smooth
  original equity curve was partly lucky sequencing.
- Ruin: P(max relative drawdown >= threshold). Recommended capital uses the
  dollar-drawdown distribution: c >= q95(dollarDD) / threshold, which is
  conservative (peak equity >= starting capital).
- Stress: friction scales to the strategy itself - basis points of trade
  notional when prices/quantities are known, otherwise a fraction of the
  average absolute trade PnL. A fixed dollar default calibrated to futures
  would misjudge small-notional strategies.
- Sharpe: annualized with the strategy's actual trade frequency
  (sqrt(trades_per_year)), not a blanket sqrt(252).
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import numpy as np

from .ingest import TradeData

ProgressFn = Callable[[str, float], None]

_CHUNK = 1000          # samples per vectorized chunk
_MAX_CURVE_PTS = 400   # downsampled x-resolution for cone/spaghetti
_N_SPAGHETTI = 40      # individual sample paths sent to the UI
_PF_CAP = 999.0


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _r(x, nd=4):
    """JSON-safe rounded float."""
    if x is None:
        return None
    x = float(x)
    if math.isnan(x):
        return None
    if math.isinf(x):
        return _PF_CAP if x > 0 else -_PF_CAP
    return round(x, nd)


def _profit_factor(pnls: np.ndarray) -> float:
    gp = pnls[pnls > 0].sum()
    gl = -pnls[pnls < 0].sum()
    if gl <= 0:
        return _PF_CAP if gp > 0 else 0.0
    return min(gp / gl, _PF_CAP)


def _pf_rows(adj: np.ndarray) -> np.ndarray:
    """Profit factor per row of a (m, n) PnL matrix."""
    pos = np.where(adj > 0, adj, 0.0).sum(axis=1)
    neg = -np.where(adj < 0, adj, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pf = np.where(neg > 0, pos / np.maximum(neg, 1e-12),
                      np.where(pos > 0, _PF_CAP, 0.0))
    return np.minimum(pf, _PF_CAP)


def _dd_stats_rows(pnl_matrix: np.ndarray, capital: float):
    """Per-row (dollar max DD, relative max DD) for a (m, n) PnL matrix."""
    eq = capital + np.cumsum(pnl_matrix, axis=1)
    eq = np.concatenate([np.full((eq.shape[0], 1), capital), eq], axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    dd_dollar = (peak - eq).max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (peak - eq) / np.maximum(peak, 1e-9)
    dd_rel = rel.max(axis=1)
    return dd_dollar, dd_rel


def _max_consecutive_losses(pnls: np.ndarray) -> int:
    worst = run = 0
    for p in pnls:
        run = run + 1 if p < 0 else 0
        worst = max(worst, run)
    return worst


def _hist(values: np.ndarray, bins: int = 48) -> Dict:
    counts, edges = np.histogram(values, bins=bins)
    return {"counts": counts.tolist(),
            "edges": [_r(e, 2) for e in edges.tolist()]}


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

def _baseline(data: TradeData, capital: float) -> Dict:
    pnls = np.asarray(data.pnls, dtype=np.float64)
    n = len(pnls)
    wins = pnls > 0
    losses = pnls < 0
    total = pnls.sum()
    win_rate = wins.mean() if n else 0.0
    avg_win = pnls[wins].mean() if wins.any() else 0.0
    avg_loss = -pnls[losses].mean() if losses.any() else 0.0

    dd_dollar, dd_rel = _dd_stats_rows(pnls[None, :], capital)

    years = None
    trades_per_year = None
    cagr = None
    start, end = None, None
    if data.has_dates:
        times = [t for t in data.entry_times + data.exit_times if t is not None]
        start, end = min(times), max(times)
        span_days = max((end - start).days, 1)
        years = span_days / 365.25
        trades_per_year = n / years if years > 0 else None
        if years >= 0.25 and capital > 0 and (capital + total) > 0:
            cagr = (capital + total) / capital
            cagr = cagr ** (1 / years) - 1

    # Per-trade Sharpe annualized by actual trade frequency
    tpy = trades_per_year if trades_per_year and trades_per_year > 0 else 252.0
    rets = pnls / capital if capital > 0 else pnls
    sharpe = None
    if n > 1 and rets.std(ddof=1) > 0:
        sharpe = rets.mean() / rets.std(ddof=1) * math.sqrt(tpy)

    # Kelly criterion from win rate + payoff ratio
    kelly = None
    if avg_loss > 0 and 0 < win_rate < 1:
        payoff = avg_win / avg_loss if avg_loss else 0.0
        if payoff > 0:
            kelly = max(win_rate - (1 - win_rate) / payoff, 0.0)

    # Equity curve of the original sequence (downsampled for display)
    eq = capital + np.cumsum(pnls)
    eq = np.concatenate([[capital], eq])
    step = max(1, len(eq) // _MAX_CURVE_PTS)
    idx = list(range(0, len(eq), step))
    if idx[-1] != len(eq) - 1:
        idx.append(len(eq) - 1)

    return {
        "n_trades": n,
        "total_pnl": _r(total, 2),
        "total_return_pct": _r(total / capital * 100, 2) if capital > 0 else None,
        "win_rate": _r(win_rate),
        "profit_factor": _r(_profit_factor(pnls), 3),
        "avg_win": _r(avg_win, 2),
        "avg_loss": _r(avg_loss, 2),
        "expectancy": _r(pnls.mean(), 2) if n else None,
        "max_dd_dollar": _r(float(dd_dollar[0]), 2),
        "max_dd_pct": _r(float(dd_rel[0]) * 100, 2),
        "max_consec_losses": _max_consecutive_losses(pnls),
        "sharpe": _r(sharpe, 3),
        "sharpe_basis": ("trade frequency" if trades_per_year else "assumed 252/yr"),
        "cagr_pct": _r(cagr * 100, 2) if cagr is not None else None,
        "years": _r(years, 2),
        "trades_per_year": _r(trades_per_year, 1),
        "date_start": start.isoformat() if start else None,
        "date_end": end.isoformat() if end else None,
        "kelly_pct": _r(kelly * 100, 1) if kelly is not None else None,
        "equity_curve": {
            "x": idx,
            "y": [_r(float(eq[i]), 2) for i in idx],
        },
        "best_trade": _r(pnls.max(), 2) if n else None,
        "worst_trade": _r(pnls.min(), 2) if n else None,
    }


def _bootstrap(pnls: np.ndarray, capital: float, n_samples: int,
               rng: np.random.Generator, progress: Optional[ProgressFn]) -> Dict:
    n = len(pnls)
    step = max(1, (n + 1) // _MAX_CURVE_PTS)
    cols = np.arange(0, n + 1, step)
    if cols[-1] != n:
        cols = np.append(cols, n)

    finals = np.empty(n_samples)
    dd_dollar = np.empty(n_samples)
    dd_rel = np.empty(n_samples)
    pf = np.empty(n_samples)
    sharpe_dist = np.empty(n_samples)
    curve_cols = np.empty((n_samples, len(cols)))

    done = 0
    while done < n_samples:
        m = min(_CHUNK, n_samples - done)
        idx = rng.integers(0, n, size=(m, n))
        sampled = pnls[idx]
        eq = capital + np.cumsum(sampled, axis=1)
        eq = np.concatenate([np.full((m, 1), capital), eq], axis=1)
        peak = np.maximum.accumulate(eq, axis=1)
        dd_dollar[done:done + m] = (peak - eq).max(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd_rel[done:done + m] = ((peak - eq) / np.maximum(peak, 1e-9)).max(axis=1)
        finals[done:done + m] = eq[:, -1] - capital
        pf[done:done + m] = _pf_rows(sampled)
        mu = sampled.mean(axis=1)
        sd = sampled.std(axis=1, ddof=1) if n > 1 else np.ones(m)
        sharpe_dist[done:done + m] = np.where(sd > 0, mu / np.maximum(sd, 1e-12), 0.0)
        curve_cols[done:done + m] = eq[:, cols]
        done += m
        if progress:
            progress("bootstrap", done / n_samples)

    pcts = {p: np.percentile(curve_cols, p, axis=0) for p in (5, 25, 50, 75, 95)}
    spaghetti = curve_cols[:_N_SPAGHETTI]

    q = lambda a, p: float(np.percentile(a, p))
    return {
        "n_samples": n_samples,
        "final_pnl": {f"p{p:02d}": _r(q(finals, p), 2) for p in (5, 25, 50, 75, 95)},
        "final_pnl_mean": _r(finals.mean(), 2),
        "prob_loss": _r(float((finals <= 0).mean())),
        "pf": {f"p{p:02d}": _r(q(pf, p), 3) for p in (5, 50, 95)},
        "max_dd_pct": {f"p{p:02d}": _r(q(dd_rel, p) * 100, 2)
                       for p in (5, 25, 50, 75, 95)},
        "max_dd_dollar": {f"p{p:02d}": _r(q(dd_dollar, p), 2)
                          for p in (50, 95, 99)},
        "return_hist": _hist(finals),
        "dd_hist": _hist(dd_rel * 100),
        "cone": {
            "x": cols.tolist(),
            **{f"p{p:02d}": [_r(v, 2) for v in pcts[p]] for p in (5, 25, 50, 75, 95)},
        },
        "spaghetti": [[_r(v, 2) for v in row] for row in spaghetti],
        "_dd_dollar_dist": dd_dollar,   # internal, stripped before JSON
        "_dd_rel_dist": dd_rel,
    }


def _shuffle_test(pnls: np.ndarray, capital: float, baseline_dd_pct: float,
                  n_samples: int, rng: np.random.Generator,
                  progress: Optional[ProgressFn]) -> Dict:
    n = len(pnls)
    dd_rel = np.empty(n_samples)
    done = 0
    while done < n_samples:
        m = min(_CHUNK, n_samples - done)
        block = np.tile(pnls, (m, 1))
        block = rng.permuted(block, axis=1)
        _, rel = _dd_stats_rows(block, capital)
        dd_rel[done:done + m] = rel
        done += m
        if progress:
            progress("shuffle", done / n_samples)

    dd_pct = dd_rel * 100
    # Percentile of the original ordering's DD within the shuffled distribution.
    # Low value => original sequence had unusually small DD => lucky ordering.
    luck_pctile = float((dd_pct <= baseline_dd_pct).mean()) * 100
    return {
        "n_samples": n_samples,
        "median_dd_pct": _r(float(np.median(dd_pct)), 2),
        "p95_dd_pct": _r(float(np.percentile(dd_pct, 95)), 2),
        "baseline_dd_pct": _r(baseline_dd_pct, 2),
        "baseline_dd_percentile": _r(luck_pctile, 1),
        "lucky_ordering": bool(luck_pctile < 10.0),
        "dd_hist": _hist(dd_pct),
    }


# Stress scenario definitions: (name, description, skip_rate, friction_bps,
# friction_frac_of_avg_abs_pnl). bps used when notional known, frac otherwise.
_SCENARIOS = [
    ("optimistic", "Near-perfect execution", 0.00, 1.0, 0.02),
    ("realistic", "Typical live conditions", 0.02, 5.0, 0.08),
    ("pessimistic", "Adverse conditions", 0.05, 15.0, 0.20),
    ("extreme", "Everything goes wrong", 0.10, 30.0, 0.40),
]


def _stress(data: TradeData, capital: float, n_seeds: int,
            rng: np.random.Generator, progress: Optional[ProgressFn]) -> Dict:
    pnls = np.asarray(data.pnls, dtype=np.float64)
    n = len(pnls)

    med_notional = data.median_notional
    if med_notional:
        notionals = np.array([x if x else med_notional for x in data.notionals])
        friction_base = notionals / 10_000.0     # $ per bp, per trade
        friction_mode = "bps_of_notional"
    else:
        avg_abs = float(np.abs(pnls).mean()) or 1.0
        friction_base = np.full(n, avg_abs)      # $ per unit fraction
        friction_mode = "fraction_of_avg_pnl"

    out = {"friction_mode": friction_mode,
           "median_notional": _r(med_notional, 2) if med_notional else None,
           "scenarios": []}

    for si, (name, desc, skip, bps, frac) in enumerate(_SCENARIOS):
        unit = bps if friction_mode == "bps_of_notional" else frac
        per_trade_friction = friction_base * unit   # mean friction $ per trade

        keep = rng.random((n_seeds, n)) >= skip
        slip = rng.uniform(0.0, 2.0, size=(n_seeds, n)) * per_trade_friction
        adj = np.where(keep, pnls - slip, 0.0)

        totals = adj.sum(axis=1)
        pf = _pf_rows(adj)
        _, dd_rel = _dd_stats_rows(adj, capital)

        out["scenarios"].append({
            "name": name,
            "description": desc,
            "skip_rate": skip,
            "friction_label": (f"{bps:g} bps of notional"
                               if friction_mode == "bps_of_notional"
                               else f"{frac:.0%} of avg |PnL|"),
            "avg_friction_per_trade": _r(float(per_trade_friction.mean()), 2),
            "total_pnl_med": _r(float(np.median(totals)), 2),
            "total_pnl_p05": _r(float(np.percentile(totals, 5)), 2),
            "pf_med": _r(float(np.median(pf)), 3),
            "max_dd_pct_med": _r(float(np.median(dd_rel)) * 100, 2),
            "max_dd_pct_p95": _r(float(np.percentile(dd_rel, 95)) * 100, 2),
            "prob_loss": _r(float((totals <= 0).mean())),
        })
        if progress:
            progress("stress", (si + 1) / len(_SCENARIOS))

    return out


def _verdict(baseline: Dict, bootstrap: Dict, shuffle: Dict, stress: Dict,
             n_trades: int) -> Dict:
    flags: List[Dict] = []
    scen = {s["name"]: s for s in stress["scenarios"]}
    pf_opt = scen["optimistic"]["pf_med"] or 0.0
    pf_real = scen["realistic"]["pf_med"] or 0.0
    pf_pess = scen["pessimistic"]["pf_med"] or 0.0

    degradation = (pf_opt - pf_real) / pf_opt if pf_opt > 0 else 1.0
    degradation_pess = (pf_opt - pf_pess) / pf_opt if pf_opt > 0 else 1.0

    score = 0  # higher is worse

    if n_trades < 30:
        flags.append({"level": "warn", "text":
                      f"Only {n_trades} trades - too few for statistically reliable "
                      "conclusions. Treat every number here as a rough sketch."})
        score += 1
    if bootstrap["prob_loss"] is not None and bootstrap["prob_loss"] > 0.20:
        flags.append({"level": "bad", "text":
                      f"{bootstrap['prob_loss']:.0%} of bootstrap resamples end at a loss. "
                      "The edge is not consistent."})
        score += 2
    elif bootstrap["prob_loss"] is not None and bootstrap["prob_loss"] > 0.05:
        flags.append({"level": "warn", "text":
                      f"{bootstrap['prob_loss']:.0%} of bootstrap resamples end at a loss."})
        score += 1
    if degradation > 0.5:
        flags.append({"level": "bad", "text":
                      f"Profit factor drops {degradation:.0%} under realistic execution "
                      "friction. The edge may be smaller than the costs of trading it."})
        score += 3
    elif degradation > 0.25:
        flags.append({"level": "warn", "text":
                      f"Profit factor drops {degradation:.0%} under realistic friction - "
                      "meaningful execution sensitivity."})
        score += 2
    elif degradation > 0.10:
        flags.append({"level": "info", "text":
                      f"Profit factor drops {degradation:.0%} under realistic friction - "
                      "modest execution sensitivity."})
        score += 1
    if scen["realistic"]["prob_loss"] and scen["realistic"]["prob_loss"] > 0.25:
        flags.append({"level": "bad", "text":
                      f"Under realistic conditions, {scen['realistic']['prob_loss']:.0%} "
                      "of simulated runs lose money."})
        score += 2
    if shuffle["lucky_ordering"]:
        flags.append({"level": "warn", "text":
                      "The original trade ordering's drawdown is in the bottom "
                      f"{shuffle['baseline_dd_percentile']:.0f}% of shuffled orderings - "
                      "the smooth historical equity curve was partly lucky sequencing. "
                      f"Expect drawdowns nearer {shuffle['median_dd_pct']:.1f}% going forward."})
        score += 1
    if (baseline["profit_factor"] or 0) < 1.0:
        flags.append({"level": "bad", "text":
                      "Baseline profit factor is below 1.0 - the strategy lost money "
                      "even before stress testing."})
        score += 4

    if score >= 6:
        cls, emoji = "Overfit / Not Tradable", "X"
        summary = ("This strategy's backtest performance is unlikely to survive real "
                   "trading conditions.")
    elif score >= 4:
        cls, emoji = "Fragile", "!"
        summary = ("The edge exists but is highly sensitive to execution quality and "
                   "sequencing. Trade small, monitor closely.")
    elif score >= 2:
        cls, emoji = "Moderate", "~"
        summary = ("Reasonable robustness with some caveats worth understanding "
                   "before sizing up.")
    else:
        cls, emoji = "Robust", "+"
        summary = ("The edge survives realistic execution stress and does not depend "
                   "on lucky trade ordering.")

    return {
        "classification": cls,
        "emoji": emoji,
        "summary": summary,
        "score": score,
        "pf_degradation_realistic": _r(degradation),
        "pf_degradation_pessimistic": _r(degradation_pess),
        "flags": flags,
    }


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def run_full_analysis(
    data: TradeData,
    capital: float = 100_000.0,
    n_samples: int = 10_000,
    ruin_threshold: float = 0.50,
    n_stress_seeds: int = 2_000,
    seed: int = 42,
    progress: Optional[ProgressFn] = None,
) -> Dict:
    """Run the complete Monte Carlo analysis. Returns JSON-serializable dict."""
    if len(data) == 0:
        raise ValueError("No trades to analyze.")
    if capital <= 0:
        raise ValueError("Starting capital must be positive.")

    pnls = np.asarray(data.pnls, dtype=np.float64)
    rng = np.random.default_rng(seed)

    if progress:
        progress("baseline", 1.0)
    baseline = _baseline(data, capital)

    boot = _bootstrap(pnls, capital, n_samples, rng, progress)

    n_shuffle = min(n_samples, 5000)
    shuffle = _shuffle_test(pnls, capital, baseline["max_dd_pct"] or 0.0,
                            n_shuffle, rng, progress)

    # Ruin: probabilities from relative-DD dist; capital rec from dollar-DD dist
    dd_rel = boot.pop("_dd_rel_dist")
    dd_dollar = boot.pop("_dd_dollar_dist")
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]
    if ruin_threshold not in thresholds:
        thresholds.append(ruin_threshold)
    ruin = {
        "capital": capital,
        "threshold": ruin_threshold,
        "prob_by_threshold": [
            {"dd_pct": int(t * 100), "prob": _r(float((dd_rel >= t).mean()))}
            for t in sorted(thresholds)
        ],
        "prob_ruin": _r(float((dd_rel >= ruin_threshold).mean())),
        "recommended_capital": _r(
            float(np.percentile(dd_dollar, 95)) / ruin_threshold, 0),
        "recommended_capital_note":
            f"Capital at which P(max drawdown >= {ruin_threshold:.0%}) stays "
            "under 5% (conservative, dollar-drawdown based).",
    }
    if progress:
        progress("ruin", 1.0)

    stress = _stress(data, capital, n_stress_seeds, rng, progress)
    verdict = _verdict(baseline, boot, shuffle, stress, len(data))

    if progress:
        progress("done", 1.0)

    return {
        "meta": {
            "n_trades": len(data),
            "source_format": data.source_format,
            "symbol": data.symbol,
            "capital": capital,
            "n_samples": n_samples,
            "n_stress_seeds": n_stress_seeds,
            "seed": seed,
            "warnings": data.warnings,
        },
        "baseline": baseline,
        "bootstrap": boot,
        "shuffle": shuffle,
        "ruin": ruin,
        "stress": stress,
        "verdict": verdict,
    }
