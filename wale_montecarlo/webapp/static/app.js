/* Strategy Stress Lab — frontend
   Runs in two modes:
   - app mode: upload → poll job → render results
   - report mode: window.EMBEDDED_RESULTS present → render immediately  */

(function () {
  "use strict";

  // ------------------------------------------------------------ palette
  const C = {
    page: "#0d0d0d", surface: "#1a1a19", ink: "#ffffff", ink2: "#c3c2b7",
    mut: "#898781", grid: "#2c2c2a", baseline: "#383835",
    blue: "#3987e5", aqua: "#199e70", yellow: "#c98500", red: "#e66767",
    good: "#0ca30c", warn: "#fab219", serious: "#ec835a", critical: "#d03b3b",
  };
  const MONO = '"Cascadia Code", Consolas, ui-monospace, monospace';

  if (window.Chart) {
    Chart.defaults.color = C.mut;
    Chart.defaults.borderColor = C.grid;
    Chart.defaults.font.family = MONO;
    Chart.defaults.font.size = 11;
    const instant = window.EMBEDDED_RESULTS ||
      matchMedia("(prefers-reduced-motion: reduce)").matches;
    Chart.defaults.animation = instant ? false : { duration: 350 };
    Chart.defaults.plugins.tooltip.backgroundColor = "#222221";
    Chart.defaults.plugins.tooltip.borderColor = "rgba(255,255,255,0.14)";
    Chart.defaults.plugins.tooltip.borderWidth = 1;
    Chart.defaults.plugins.tooltip.titleColor = C.ink;
    Chart.defaults.plugins.tooltip.bodyColor = C.ink2;
    Chart.defaults.plugins.tooltip.padding = 10;
    Chart.defaults.plugins.tooltip.cornerRadius = 6;
    Chart.defaults.plugins.legend.labels.boxWidth = 14;
    Chart.defaults.plugins.legend.labels.boxHeight = 3;
  }

  // Vertical annotation lines on histograms
  const vlinePlugin = {
    id: "vlines",
    afterDatasetsDraw(chart, _args, opts) {
      if (!opts || !opts.lines) return;
      const { ctx, chartArea, scales } = chart;
      for (const ln of opts.lines) {
        const x = scales.x.getPixelForValue(ln.x);
        if (x < chartArea.left - 2 || x > chartArea.right + 2) continue;
        ctx.save();
        ctx.strokeStyle = ln.color; ctx.lineWidth = 1.4;
        ctx.setLineDash(ln.dash || [5, 4]);
        ctx.beginPath();
        ctx.moveTo(x, chartArea.top); ctx.lineTo(x, chartArea.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = ln.color;
        ctx.font = `10px ${MONO}`;
        ctx.textAlign = ln.align || "left";
        const tx = x + (ln.align === "right" ? -5 : 5);
        ctx.fillText(ln.label, tx, chartArea.top + (ln.dy || 10));
        ctx.restore();
      }
    },
  };

  // Direct value labels at the end of horizontal bars
  const barLabelPlugin = {
    id: "barLabels",
    afterDatasetsDraw(chart, _args, opts) {
      if (!opts || !opts.enabled) return;
      const { ctx } = chart;
      const meta = chart.getDatasetMeta(0);
      const data = chart.data.datasets[0].data;
      ctx.save();
      ctx.font = `11px ${MONO}`;
      ctx.textBaseline = "middle";
      meta.data.forEach((bar, i) => {
        const v = data[i];
        ctx.fillStyle = C.ink2;
        ctx.textAlign = "left";
        ctx.fillText(fmtPct(v, 1), bar.x + 7, bar.y);
      });
      ctx.restore();
    },
  };
  if (window.Chart) Chart.register(vlinePlugin, barLabelPlugin);

  // ------------------------------------------------------------ helpers
  const $ = (sel) => document.querySelector(sel);
  const el = (tag, cls, html) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html !== undefined) e.innerHTML = html;
    return e;
  };
  const fmtMoney = (v, dp = 0) => {
    if (v === null || v === undefined) return "—";
    const sign = v < 0 ? "−$" : "$";
    return sign + Math.abs(v).toLocaleString("en-US", {
      minimumFractionDigits: dp, maximumFractionDigits: dp });
  };
  const fmtPct = (v, dp = 1) =>
    (v === null || v === undefined) ? "—" : (v * 100).toFixed(dp) + "%";
  const fmtNum = (v, dp = 2) =>
    (v === null || v === undefined) ? "—" : Number(v).toFixed(dp);
  const esc = (s) => String(s).replace(/[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  let charts = [];
  function newChart(canvas, cfg) {
    const ch = new Chart(canvas, cfg);
    charts.push(ch);
    return ch;
  }
  function destroyCharts() {
    charts.forEach((c) => c.destroy());
    charts = [];
  }

  // ------------------------------------------------------------ rendering
  const VERDICT_COLOR = {
    "Robust": C.good, "Moderate": C.warn,
    "Fragile": C.serious, "Overfit / Not Tradable": C.critical,
  };

  function render(res) {
    destroyCharts();
    const root = $("#results");
    root.innerHTML = "";

    renderVerdict(root, res);
    renderTiles(root, res);

    // charts row 1: equity cone + return distribution
    const row1 = el("div", "grid-2");
    row1.appendChild(chartCard("Equity paths", coneSub(res),
      (box) => coneChart(box, res), "tall"));
    row1.appendChild(chartCard("Final P&L distribution",
      `${res.bootstrap.n_samples.toLocaleString()} resamples · ` +
      `${fmtPct(res.bootstrap.prob_loss)} end at a loss`,
      (box) => returnHistChart(box, res), "tall"));
    root.appendChild(row1);

    // charts row 2: drawdown dist + ruin ladder
    const row2 = el("div", "grid-2b");
    row2.appendChild(chartCard("Max drawdown distribution",
      "Bootstrap resamples · dashed markers: your backtest vs shuffled orderings",
      (box) => ddHistChart(box, res)));
    row2.appendChild(chartCard("Drawdown probabilities",
      `Chance of hitting each drawdown at ${fmtMoney(res.ruin.capital)} capital`,
      (box) => ruinChart(box, res)));
    root.appendChild(row2);

    renderStressTable(root, res);
    renderMeta(root, res);

    root.hidden = false;
  }

  function renderVerdict(root, res) {
    const v = res.verdict;
    const color = VERDICT_COLOR[v.classification] || C.mut;
    const plate = el("section", "verdict");
    plate.style.setProperty("--verdict-color", color);

    const stamp = el("div", "verdict-stamp");
    stamp.appendChild(el("div", "verdict-eyebrow", "verdict"));
    stamp.appendChild(el("div", "verdict-word", esc(v.classification)));
    stamp.appendChild(el("div", "verdict-summary", esc(v.summary)));
    plate.appendChild(stamp);

    const flags = el("div", "verdict-flags");
    const items = v.flags && v.flags.length ? v.flags
      : [{ level: "ok", text: "No red flags. Edge survives resampling, reordering and realistic execution friction." }];
    for (const f of items) {
      const row = el("div", `flag ${f.level}`);
      const tag = { bad: "FAIL", warn: "WARN", info: "NOTE", ok: "PASS" }[f.level] || "NOTE";
      row.appendChild(el("span", "flag-dot", tag));
      row.appendChild(el("span", "", esc(f.text)));
      flags.appendChild(row);
    }
    plate.appendChild(flags);
    root.appendChild(plate);
  }

  function tile(label, value, cls, note) {
    const t = el("div", "tile");
    t.appendChild(el("div", "t-label", label));
    t.appendChild(el("div", `t-value${cls ? " " + cls : ""}`, value));
    if (note) t.appendChild(el("div", "t-note", note));
    return t;
  }

  function renderTiles(root, res) {
    const b = res.baseline, bo = res.bootstrap, ru = res.ruin;
    const wrap = el("section", "tiles");
    const pnlCls = (b.total_pnl ?? 0) >= 0 ? "pos" : "neg";
    wrap.appendChild(tile("Net P&L", fmtMoney(b.total_pnl), pnlCls,
      b.total_return_pct !== null ? fmtNum(b.total_return_pct, 1) + "% on capital" : ""));
    wrap.appendChild(tile("Profit factor", b.profit_factor >= 999 ? "∞" : fmtNum(b.profit_factor),
      b.profit_factor >= 1 ? "" : "neg", "gross win / gross loss"));
    wrap.appendChild(tile("Win rate", fmtPct(b.win_rate, 0), "",
      `${b.n_trades} trades`));
    wrap.appendChild(tile("Max drawdown", fmtNum(b.max_dd_pct, 1) + "%",
      (b.max_dd_pct ?? 0) > 30 ? "neg" : "", "your backtest path"));
    if (b.cagr_pct !== null && b.cagr_pct !== undefined)
      wrap.appendChild(tile("CAGR", fmtNum(b.cagr_pct, 1) + "%",
        b.cagr_pct >= 0 ? "pos" : "neg", `${fmtNum(b.years, 1)} yrs of data`));
    if (b.sharpe !== null && b.sharpe !== undefined)
      wrap.appendChild(tile("Sharpe", fmtNum(b.sharpe), "",
        `annualized · ${b.sharpe_basis}`));
    wrap.appendChild(tile("P(loss)", fmtPct(bo.prob_loss, 1),
      bo.prob_loss > 0.1 ? "neg" : "", "of bootstrap resamples"));
    wrap.appendChild(tile("Capital for safety", fmtMoney(ru.recommended_capital), "",
      `keeps P(DD ≥ ${Math.round(ru.threshold * 100)}%) < 5%`));
    if (b.kelly_pct !== null && b.kelly_pct !== undefined)
      wrap.appendChild(tile("Kelly fraction", fmtNum(b.kelly_pct, 1) + "%", "",
        "half-Kelly is safer: " + fmtNum(b.kelly_pct / 2, 1) + "%"));
    root.appendChild(wrap);
  }

  function chartCard(title, sub, build, tall) {
    const card = el("section", "card");
    card.appendChild(el("h2", "", title));
    if (sub) card.appendChild(el("div", "card-sub", sub));
    const box = el("div", "chart-box" + (tall ? " tall" : ""));
    const canvas = document.createElement("canvas");
    box.appendChild(canvas);
    card.appendChild(box);
    build(canvas);
    return card;
  }

  function coneSub(res) {
    const bo = res.bootstrap;
    return `Median ${fmtMoney(bo.final_pnl.p50)} · 90% of outcomes between ` +
      `${fmtMoney(bo.final_pnl.p05)} and ${fmtMoney(bo.final_pnl.p95)}`;
  }

  // ---- chart builders -------------------------------------------------
  function coneChart(canvas, res) {
    const cone = res.bootstrap.cone;
    const actual = res.baseline.equity_curve;
    const labels = cone.x;
    const band = (data, fill, color) => ({
      data, fill, borderWidth: 0, pointRadius: 0,
      backgroundColor: color, tension: 0.2, label: "_band",
    });
    newChart(canvas, {
      type: "line",
      data: {
        labels,
        datasets: [
          { data: cone.p95, fill: false, borderWidth: 0, pointRadius: 0, label: "_hide" },
          band(cone.p05, "-1", "rgba(57,135,229,0.12)"),
          { data: cone.p75, fill: false, borderWidth: 0, pointRadius: 0, label: "_hide" },
          band(cone.p25, "-1", "rgba(57,135,229,0.20)"),
          { data: cone.p50, label: "Median resample", borderColor: C.blue,
            borderWidth: 2, pointRadius: 0, tension: 0.2 },
          { data: interp(actual.x, actual.y, labels), label: "Your backtest",
            borderColor: C.ink2, borderDash: [6, 4], borderWidth: 1.6,
            pointRadius: 0, tension: 0.2 },
        ],
      },
      options: {
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { title: { display: true, text: "trade #" },
               ticks: { maxTicksLimit: 9 }, grid: { display: false } },
          y: { ticks: { callback: (v) => compactMoney(v) },
               grid: { color: C.grid } },
        },
        plugins: {
          legend: { labels: { filter: (item) => !item.text.startsWith("_") } },
          tooltip: {
            filter: (item) => !item.dataset.label.startsWith("_"),
            callbacks: { label: (item) =>
              ` ${item.dataset.label}: ${fmtMoney(item.parsed.y)}` },
          },
        },
      },
    });
  }

  function interp(xs, ys, targetXs) {
    // Actual curve is sampled at its own indices; align to cone x-labels.
    const out = [];
    let j = 0;
    for (const tx of targetXs) {
      while (j < xs.length - 1 && xs[j + 1] <= tx) j++;
      if (tx <= xs[0]) { out.push(ys[0]); continue; }
      if (j >= xs.length - 1) { out.push(ys[ys.length - 1]); continue; }
      const t = (tx - xs[j]) / (xs[j + 1] - xs[j] || 1);
      out.push(ys[j] + t * (ys[j + 1] - ys[j]));
    }
    return out;
  }

  function compactMoney(v) {
    const a = Math.abs(v);
    if (a >= 1e12) return "$" + (v / 1e12).toFixed(1) + "T";
    if (a >= 1e9) return "$" + (v / 1e9).toFixed(1) + "B";
    if (a >= 1e6) return "$" + (v / 1e6).toFixed(1) + "M";
    if (a >= 1e3) return "$" + (v / 1e3).toFixed(0) + "k";
    return "$" + Math.round(v);
  }

  function histCenters(hist) {
    const e = hist.edges, out = [];
    for (let i = 0; i < e.length - 1; i++) out.push((e[i] + e[i + 1]) / 2);
    return out;
  }

  function returnHistChart(canvas, res) {
    const hist = res.bootstrap.return_hist;
    const centers = histCenters(hist);
    newChart(canvas, {
      type: "bar",
      data: {
        labels: centers,
        datasets: [{
          data: hist.counts,
          backgroundColor: centers.map((c) => (c < 0 ? C.red : C.blue)),
          borderRadius: 3, barPercentage: 1, categoryPercentage: 0.92,
        }],
      },
      options: {
        maintainAspectRatio: false,
        scales: {
          x: { type: "linear", min: hist.edges[0], max: hist.edges[hist.edges.length - 1],
               title: { display: true, text: "final P&L ($)" },
               ticks: { maxTicksLimit: 7, callback: (v) => compactMoney(v) },
               grid: { display: false } },
          y: { display: false },
        },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: {
            title: (items) => "≈ " + compactMoney(items[0].parsed.x),
            label: (item) => ` ${item.parsed.y} of ${res.bootstrap.n_samples.toLocaleString()} runs`,
          } },
          vlines: { lines: [
            { x: 0, color: C.mut, label: "break-even", dash: [3, 3], dy: 12 },
            { x: res.bootstrap.final_pnl.p05, color: C.serious, label: "P5", dy: 26, align: "right" },
          ] },
        },
      },
    });
  }

  function ddHistChart(canvas, res) {
    const hist = res.bootstrap.dd_hist;
    const centers = histCenters(hist);
    const lines = [
      { x: res.shuffle.baseline_dd_pct, color: C.ink2, label: "your path", dy: 12 },
      { x: res.shuffle.median_dd_pct, color: C.yellow, label: "shuffled median", dy: 28 },
    ];
    newChart(canvas, {
      type: "bar",
      data: {
        labels: centers,
        datasets: [{
          data: hist.counts, backgroundColor: C.red,
          borderRadius: 3, barPercentage: 1, categoryPercentage: 0.92,
        }],
      },
      options: {
        maintainAspectRatio: false,
        scales: {
          x: { type: "linear", min: hist.edges[0], max: hist.edges[hist.edges.length - 1],
               title: { display: true, text: "max drawdown (% of peak equity)" },
               ticks: { maxTicksLimit: 8, callback: (v) => v + "%" },
               grid: { display: false } },
          y: { display: false },
        },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: {
            title: (items) => "≈ " + items[0].parsed.x.toFixed(1) + "% drawdown",
            label: (item) => ` ${item.parsed.y} of ${res.bootstrap.n_samples.toLocaleString()} runs`,
          } },
          vlines: { lines },
        },
      },
    });
  }

  function ruinChart(canvas, res) {
    const rows = res.ruin.prob_by_threshold;
    const sevColor = (p) =>
      p >= 0.5 ? C.critical : p >= 0.2 ? C.serious : p >= 0.05 ? C.warn : C.good;
    newChart(canvas, {
      type: "bar",
      data: {
        labels: rows.map((r) => "≥ " + r.dd_pct + "% DD"),
        datasets: [{
          data: rows.map((r) => r.prob),
          backgroundColor: rows.map((r) => sevColor(r.prob)),
          borderRadius: 4, barPercentage: 0.65,
        }],
      },
      options: {
        indexAxis: "y",
        maintainAspectRatio: false,
        scales: {
          x: { min: 0, max: 1.15,
               ticks: { callback: (v) => (v <= 1 ? Math.round(v * 100) + "%" : ""),
                        stepSize: 0.25 },
               grid: { color: C.grid } },
          y: { grid: { display: false }, ticks: { color: C.ink2 } },
        },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (item) =>
            ` ${fmtPct(item.parsed.x, 1)} of simulated runs hit this drawdown` } },
          barLabels: { enabled: true },
        },
      },
    });
  }

  function renderStressTable(root, res) {
    const card = el("section", "card");
    card.appendChild(el("h2", "", "Execution stress scenarios"));
    const fm = res.stress.friction_mode === "bps_of_notional"
      ? `Friction scaled to your trade size (median notional ${fmtMoney(res.stress.median_notional)})`
      : "Friction scaled to your average trade P&L (no price/size data in file)";
    card.appendChild(el("div", "card-sub",
      fm + ` · ${res.meta.n_stress_seeds.toLocaleString()} seeds per scenario`));

    const t = el("table", "stress-table");
    t.innerHTML = `<thead><tr>
      <th>Scenario</th><th>Missed trades</th><th>Friction / trade</th>
      <th>Median P&L</th><th>P5 P&L</th><th>Median PF</th>
      <th>Median max DD</th><th>P(loss)</th></tr></thead>`;
    const tb = el("tbody");
    for (const s of res.stress.scenarios) {
      const tr = el("tr");
      const pnlCls = (s.total_pnl_med ?? 0) >= 0 ? "pos" : "neg";
      const p5Cls = (s.total_pnl_p05 ?? 0) >= 0 ? "pos" : "neg";
      tr.innerHTML =
        `<td>${esc(cap(s.name))}<span class="scen-desc">${esc(s.description)}</span></td>` +
        `<td>${(s.skip_rate * 100).toFixed(0)}%</td>` +
        `<td>${fmtMoney(s.avg_friction_per_trade, 2)}</td>` +
        `<td class="${pnlCls}">${fmtMoney(s.total_pnl_med)}</td>` +
        `<td class="${p5Cls}">${fmtMoney(s.total_pnl_p05)}</td>` +
        `<td>${s.pf_med >= 999 ? "∞" : fmtNum(s.pf_med)}</td>` +
        `<td>${fmtNum(s.max_dd_pct_med, 1)}%</td>` +
        `<td class="${s.prob_loss > 0.25 ? "neg" : ""}">${fmtPct(s.prob_loss, 1)}</td>`;
      tb.appendChild(tr);
    }
    t.appendChild(tb);
    card.appendChild(t);
    root.appendChild(card);
  }

  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

  function renderMeta(root, res) {
    const m = res.meta, b = res.baseline;
    const span = b.date_start
      ? ` · ${b.date_start.slice(0, 10)} → ${b.date_end.slice(0, 10)}`
      : "";
    const warn = m.warnings && m.warnings.length
      ? ` · ${m.warnings.join(" ")}` : "";
    root.appendChild(el("div", "meta-line",
      `${esc(m.filename || "trades")} · ${m.n_trades} trades${span} · ` +
      `${m.n_samples.toLocaleString()} simulations · seed ${m.seed}` +
      `${m.elapsed_seconds ? " · " + m.elapsed_seconds + "s" : ""}${esc(warn)}`));
  }

  // ------------------------------------------------------------ app mode
  const isReport = !!window.EMBEDDED_RESULTS;
  if (isReport) {
    render(window.EMBEDDED_RESULTS);
    return;
  }

  let selectedFile = null;
  let currentJob = null;

  const dz = $("#dropzone"), fileInput = $("#file-input");
  const btnRun = $("#btn-run"), btnSample = $("#btn-sample");
  const btnExport = $("#btn-export"), btnNew = $("#btn-new");

  function setFile(f) {
    selectedFile = f;
    $("#dz-file").textContent = `${f.name} · ${(f.size / 1024).toFixed(1)} KB`;
    $("#dz-file").hidden = false;
    btnRun.disabled = false;
  }

  dz.addEventListener("click", () => fileInput.click());
  dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) setFile(fileInput.files[0]);
  });
  ["dragover", "dragenter"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); }));
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) setFile(f);
  });

  $("#settings").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!selectedFile) return;
    const fd = new FormData();
    fd.append("file", selectedFile);
    fd.append("capital", $("#capital").value);
    fd.append("samples", $("#samples").value);
    fd.append("ruin_threshold", $("#ruin").value);
    try {
      const r = await fetch("/api/analyze", { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "Upload failed");
      startPolling(j.job_id);
    } catch (err) { toast(err.message); }
  });

  btnSample.addEventListener("click", async () => {
    try {
      const r = await fetch("/api/sample");
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "Sample unavailable");
      startPolling(j.job_id);
    } catch (err) { toast(err.message); }
  });

  const STAGES = ["baseline", "bootstrap", "shuffle", "ruin", "stress"];
  const STAGE_TEXT = {
    baseline: "computing baseline statistics",
    bootstrap: "bootstrapping equity paths",
    shuffle: "shuffling trade order — luck test",
    ruin: "estimating ruin probabilities",
    stress: "running execution stress scenarios",
  };

  function startPolling(jobId) {
    currentJob = jobId;
    show("progress");
    const log = $("#prog-log");
    const fill = $("#prog-fill");
    log.textContent = "";

    const timer = setInterval(async () => {
      let j;
      try {
        const r = await fetch(`/api/job/${jobId}`);
        j = await r.json();
        if (!r.ok) throw new Error(j.error || "Job lost");
      } catch (err) {
        clearInterval(timer); toast(err.message); show("setup"); return;
      }
      const stageIdx = Math.max(STAGES.indexOf(j.stage), 0);
      const lines = STAGES.map((s, i) => {
        if (i < stageIdx || j.status === "done")
          return `<span class="done">  ✓ ${STAGE_TEXT[s]}</span>`;
        if (i === stageIdx)
          return `<span class="active">  ▸ ${STAGE_TEXT[s]} ${j.stage_pct ? j.stage_pct + "%" : ""}</span>`;
        return `<span>    ${STAGE_TEXT[s]}</span>`;
      });
      log.innerHTML = lines.join("\n");
      fill.style.width = j.status === "done" ? "100%"
        : `${(stageIdx / STAGES.length) * 100 + (j.stage_pct || 0) / STAGES.length}%`;

      if (j.status === "done") {
        clearInterval(timer);
        setTimeout(() => { render(j.results); show("results"); }, 250);
      } else if (j.status === "error") {
        clearInterval(timer);
        toast(j.error || "Analysis failed");
        show("setup");
      }
    }, 250);
  }

  function show(view) {
    $("#setup").hidden = view !== "setup";
    $("#progress").hidden = view !== "progress";
    $("#results").hidden = view !== "results";
    btnExport.hidden = btnNew.hidden = view !== "results";
    window.scrollTo(0, 0);
  }

  btnExport.addEventListener("click", () => {
    if (currentJob) window.location.href = `/api/report/${currentJob}`;
  });
  btnNew.addEventListener("click", () => {
    selectedFile = null; fileInput.value = "";
    $("#dz-file").hidden = true; btnRun.disabled = true;
    show("setup");
  });

  // Demo/deep-link: /?sample=1 runs the bundled sample immediately
  if (new URLSearchParams(location.search).get("sample")) btnSample.click();

  let toastTimer = null;
  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, 6000);
  }
})();
