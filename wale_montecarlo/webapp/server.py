"""
Flask server for the Monte Carlo Strategy Stress Lab.

Routes
------
GET  /                   dashboard UI
POST /api/analyze        multipart upload (file + params) -> {job_id}
GET  /api/job/<id>       job status / progress / results
GET  /api/report/<id>    standalone self-contained HTML report download
GET  /api/sample         run the bundled sample trade list (demo mode)

Analysis runs on a background thread per job; jobs live in memory
(this is a single-user desktop app, not a hosted service).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Dict

from flask import Flask, jsonify, render_template, request, Response

from ..ingest import IngestError, load_trades_text
from ..mc import run_full_analysis

_JOBS: Dict[str, Dict] = {}
_JOBS_LOCK = threading.Lock()
_MAX_JOBS = 20

_STAGE_LABELS = {
    "baseline": "Computing baseline statistics",
    "bootstrap": "Bootstrapping equity paths",
    "shuffle": "Shuffling trade order (luck test)",
    "ruin": "Estimating ruin probabilities",
    "stress": "Running stress scenarios",
    "done": "Finished",
}


def _prune_jobs() -> None:
    with _JOBS_LOCK:
        if len(_JOBS) <= _MAX_JOBS:
            return
        by_age = sorted(_JOBS.items(), key=lambda kv: kv[1]["created"])
        for k, _ in by_age[: len(_JOBS) - _MAX_JOBS]:
            _JOBS.pop(k, None)


def _run_job(job_id: str, text: str, filename: str, capital: float,
             n_samples: int, ruin_threshold: float, seed: int) -> None:
    job = _JOBS[job_id]

    def progress(stage: str, frac: float) -> None:
        job["stage"] = stage
        job["stage_label"] = _STAGE_LABELS.get(stage, stage)
        job["stage_pct"] = round(frac * 100)

    try:
        t0 = time.time()
        data = load_trades_text(text)
        job["n_trades"] = len(data)
        results = run_full_analysis(
            data,
            capital=capital,
            n_samples=n_samples,
            ruin_threshold=ruin_threshold,
            seed=seed,
            progress=progress,
        )
        results["meta"]["filename"] = filename
        results["meta"]["elapsed_seconds"] = round(time.time() - t0, 2)
        job["results"] = results
        job["status"] = "done"
    except IngestError as e:
        job["status"] = "error"
        job["error"] = str(e)
    except Exception as e:  # surface anything unexpected to the UI
        job["status"] = "error"
        job["error"] = f"Analysis failed: {e}"


def create_app() -> Flask:
    base = Path(__file__).parent
    app = Flask(
        __name__,
        template_folder=str(base / "templates"),
        static_folder=str(base / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB uploads

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/analyze")
    def analyze():
        f = request.files.get("file")
        if f is None or not f.filename:
            return jsonify({"error": "No file uploaded."}), 400
        try:
            text = f.read().decode("utf-8-sig", errors="replace")
        except Exception:
            return jsonify({"error": "Could not read file as text/CSV."}), 400

        def _num(name, default, lo, hi, cast=float):
            try:
                v = cast(request.form.get(name, default))
            except (TypeError, ValueError):
                v = default
            return min(max(v, lo), hi)

        capital = _num("capital", 100_000.0, 1.0, 1e12)
        n_samples = _num("samples", 10_000, 500, 100_000, int)
        ruin = _num("ruin_threshold", 50.0, 5.0, 95.0) / 100.0
        seed = _num("seed", 42, 0, 2**31 - 1, int)

        job_id = uuid.uuid4().hex[:12]
        _JOBS[job_id] = {
            "status": "running", "created": time.time(),
            "stage": "parse", "stage_label": "Parsing trade list",
            "stage_pct": 0, "error": None, "results": None,
        }
        _prune_jobs()
        threading.Thread(
            target=_run_job,
            args=(job_id, text, f.filename, capital, n_samples, ruin, seed),
            daemon=True,
        ).start()
        return jsonify({"job_id": job_id})

    @app.get("/api/job/<job_id>")
    def job_status(job_id: str):
        job = _JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown job."}), 404
        out = {
            "status": job["status"],
            "stage": job["stage"],
            "stage_label": job["stage_label"],
            "stage_pct": job["stage_pct"],
            "error": job["error"],
        }
        if job["status"] == "done":
            out["results"] = job["results"]
        return jsonify(out)

    @app.get("/api/report/<job_id>")
    def report(job_id: str):
        job = _JOBS.get(job_id)
        if job is None or job["status"] != "done":
            return jsonify({"error": "No completed analysis for this id."}), 404
        html = build_standalone_report(job["results"], base)
        fname = (job["results"]["meta"].get("filename") or "analysis").rsplit(".", 1)[0]
        return Response(
            html, mimetype="text/html",
            headers={"Content-Disposition":
                     f'attachment; filename="{fname}_stress_report.html"'},
        )

    @app.get("/api/sample")
    def sample():
        """Analyze the bundled example so first-time users see the product."""
        sample_path = _find_sample()
        if sample_path is None:
            return jsonify({"error": "Sample data not found."}), 404
        text = sample_path.read_text(encoding="utf-8-sig")
        job_id = uuid.uuid4().hex[:12]
        _JOBS[job_id] = {
            "status": "running", "created": time.time(),
            "stage": "parse", "stage_label": "Parsing trade list",
            "stage_pct": 0, "error": None, "results": None,
        }
        threading.Thread(
            target=_run_job,
            args=(job_id, text, sample_path.name, 100_000.0, 10_000, 0.5, 42),
            daemon=True,
        ).start()
        return jsonify({"job_id": job_id})

    return app


def _find_sample() -> Path | None:
    here = Path(__file__).parent
    for cand in (
        here / "static" / "sample_trades.csv",
        here.parent.parent / "examples" / "sample_trade_list.csv",
    ):
        if cand.exists():
            return cand
    return None


def build_standalone_report(results: Dict, base: Path) -> str:
    """
    Build a fully self-contained HTML report: same dashboard renderer,
    with CSS / Chart.js / app.js and the results JSON inlined.
    Works offline, single file, shareable.
    """
    static = base / "static"
    css = (static / "style.css").read_text(encoding="utf-8")
    chartjs = (static / "chart.umd.js").read_text(encoding="utf-8")
    appjs = (static / "app.js").read_text(encoding="utf-8")
    payload = json.dumps(results).replace("</", "<\\/")
    title = results["meta"].get("filename") or "Monte Carlo Report"
    generated = time.strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stress Report - {title}</title>
<style>{css}</style>
</head>
<body class="report-mode">
<header class="topbar">
  <div class="brand">
    <span class="brand-mark">MC</span>
    <span class="brand-name">Strategy Stress Lab</span>
    <span class="brand-sub">standalone report &middot; generated {generated}</span>
  </div>
</header>
<main id="results" class="results" hidden></main>
<script>{chartjs}</script>
<script>window.EMBEDDED_RESULTS = {payload};</script>
<script>{appjs}</script>
</body>
</html>"""
