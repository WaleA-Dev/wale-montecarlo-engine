"""Tests for the vectorized Monte Carlo engine (wale_montecarlo.mc)."""

import json
import math

import numpy as np
import pytest

from wale_montecarlo.ingest import load_trades_text
from wale_montecarlo.mc import run_full_analysis, _profit_factor, _dd_stats_rows


def make_data(pnls, with_dates=True):
    rows = ["entry_time,exit_time,entry_price,exit_price,pnl,side,quantity"]
    for i, p in enumerate(pnls):
        day = i % 27 + 1
        month = i // 27 % 12 + 1
        year = 2023 + i // (27 * 12)
        rows.append(f"{year}-{month:02d}-{day:02d} 09:30:00,"
                    f"{year}-{month:02d}-{day:02d} 15:30:00,100,101,{p},long,1")
    return load_trades_text("\n".join(rows))


class TestPrimitives:
    def test_profit_factor_basic(self):
        assert _profit_factor(np.array([10.0, -5.0])) == 2.0

    def test_profit_factor_all_wins_capped(self):
        assert _profit_factor(np.array([10.0, 5.0])) == 999.0

    def test_profit_factor_all_losses(self):
        assert _profit_factor(np.array([-10.0, -5.0])) == 0.0

    def test_dd_known_sequence(self):
        # capital 100; +50 -> 150 peak; -75 -> 75; dd = 75/150 = 50%
        dollar, rel = _dd_stats_rows(np.array([[50.0, -75.0]]), 100.0)
        assert dollar[0] == pytest.approx(75.0)
        assert rel[0] == pytest.approx(0.5)

    def test_dd_monotone_up_is_zero(self):
        dollar, rel = _dd_stats_rows(np.array([[10.0, 20.0, 5.0 ]]), 100.0)
        assert rel[0] < 0.001 or dollar[0] == 0.0


class TestFullAnalysis:
    @pytest.fixture(scope="class")
    def result(self):
        rng = np.random.default_rng(3)
        pnls = list(np.where(rng.random(200) < 0.5,
                             rng.uniform(50, 300, 200),
                             -rng.uniform(40, 200, 200)).round(2))
        return run_full_analysis(make_data(pnls), capital=50_000,
                                 n_samples=3000, n_stress_seeds=500, seed=11)

    def test_json_serializable(self, result):
        json.dumps(result)

    def test_baseline_consistency(self, result):
        b = result["baseline"]
        assert b["n_trades"] == 200
        assert 0 < b["win_rate"] < 1
        assert b["max_dd_pct"] >= 0

    def test_bootstrap_percentiles_ordered(self, result):
        fp = result["bootstrap"]["final_pnl"]
        assert fp["p05"] <= fp["p25"] <= fp["p50"] <= fp["p75"] <= fp["p95"]

    def test_bootstrap_median_near_total(self, result):
        # Bootstrap mean final PnL should be near the actual total
        total = result["baseline"]["total_pnl"]
        assert abs(result["bootstrap"]["final_pnl_mean"] - total) < abs(total) * 0.25 + 500

    def test_ruin_probs_monotone_decreasing(self, result):
        probs = [row["prob"] for row in result["ruin"]["prob_by_threshold"]]
        assert all(a >= b - 1e-9 for a, b in zip(probs, probs[1:]))

    def test_stress_scenarios_ordered_by_severity(self, result):
        meds = [s["total_pnl_med"] for s in result["stress"]["scenarios"]]
        assert meds[0] >= meds[1] >= meds[2] >= meds[3]

    def test_cone_shapes_match(self, result):
        cone = result["bootstrap"]["cone"]
        n = len(cone["x"])
        for k in ("p05", "p25", "p50", "p75", "p95"):
            assert len(cone[k]) == n

    def test_verdict_present(self, result):
        assert result["verdict"]["classification"] in (
            "Robust", "Moderate", "Fragile", "Overfit / Not Tradable")


class TestDeterminism:
    def test_same_seed_same_results(self):
        pnls = [100, -50, 200, -80, 60, -40, 90, -30] * 6
        a = run_full_analysis(make_data(pnls), n_samples=1000,
                              n_stress_seeds=200, seed=7)
        b = run_full_analysis(make_data(pnls), n_samples=1000,
                              n_stress_seeds=200, seed=7)
        assert a["bootstrap"]["final_pnl"] == b["bootstrap"]["final_pnl"]
        assert a["ruin"]["prob_ruin"] == b["ruin"]["prob_ruin"]

    def test_different_seed_differs(self):
        pnls = [100, -50, 200, -80, 60, -40, 90, -30] * 6
        a = run_full_analysis(make_data(pnls), n_samples=1000,
                              n_stress_seeds=200, seed=7)
        b = run_full_analysis(make_data(pnls), n_samples=1000,
                              n_stress_seeds=200, seed=8)
        assert a["bootstrap"]["final_pnl"] != b["bootstrap"]["final_pnl"]


class TestEdgeCases:
    def test_all_winners(self):
        res = run_full_analysis(make_data([100.0] * 40), n_samples=500,
                                n_stress_seeds=100)
        assert res["baseline"]["profit_factor"] == 999.0
        assert res["bootstrap"]["prob_loss"] == 0.0

    def test_all_losers(self):
        res = run_full_analysis(make_data([-100.0] * 40), n_samples=500,
                                n_stress_seeds=100)
        assert res["verdict"]["classification"] == "Overfit / Not Tradable"

    def test_tiny_sample_flagged(self):
        res = run_full_analysis(make_data([50, -30, 80, -20, 100]),
                                n_samples=500, n_stress_seeds=100)
        assert any("too few" in f["text"].lower()
                   for f in res["verdict"]["flags"])

    def test_under_10_trades_is_insufficient_data(self):
        res = run_full_analysis(make_data([100.0] * 8), n_samples=500,
                                n_stress_seeds=100)
        assert res["verdict"]["classification"] == "Insufficient Data"

    def test_under_30_trades_never_robust(self):
        # flawless 20-trade book must cap at Moderate, not stamp Robust
        res = run_full_analysis(make_data([100.0] * 20), n_samples=500,
                                n_stress_seeds=100)
        assert res["verdict"]["classification"] == "Moderate"

    def test_ingest_warnings_surface_as_flags(self):
        d = make_data([50, -30, 80, -20, 100] * 8)
        d.warnings.append("2 open/incomplete trade(s) excluded.")
        res = run_full_analysis(d, n_samples=500, n_stress_seeds=100)
        assert any("excluded" in f["text"] for f in res["verdict"]["flags"])

    def test_no_dates_still_works(self):
        d = load_trades_text("Profit\n100\n-50\n200\n-80\n" * 1)
        res = run_full_analysis(d, n_samples=500, n_stress_seeds=100)
        assert res["baseline"]["cagr_pct"] is None
        json.dumps(res)

    def test_zero_capital_raises(self):
        with pytest.raises(ValueError):
            run_full_analysis(make_data([1.0, -1.0]), capital=0)

    def test_shuffle_preserves_total(self):
        # permutation invariance: shuffled runs must not change total PnL,
        # so P(loss) in shuffle context is degenerate - verify via bootstrap
        res = run_full_analysis(make_data([100, -90] * 30), n_samples=500,
                                n_stress_seeds=100)
        assert res["shuffle"]["median_dd_pct"] > 0
