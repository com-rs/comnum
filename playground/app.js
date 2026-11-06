// The page of the playground: a CodeMirror editor of the specification, the menu of
// the specifications of the repository, a worker that runs the WebAssembly engine, and
// a Plotly chart of the metric against a variable of the grid, the expected values of
// the expectations (published curves, closed forms) drawn over the simulated points.

import { basicSetup, EditorView } from "https://esm.sh/codemirror@6.0.2";
import { yaml as yamlLanguage } from "https://esm.sh/@codemirror/lang-yaml@6.1.2";
import { setDiagnostics } from "https://esm.sh/@codemirror/lint@6.8.5";

const $ = (id) => document.getElementById(id);
const ui = {
  select: $("spec-select"), run: $("run"), stop: $("stop"), status: $("status"),
  error: $("error"), xAxis: $("x-axis"), metric: $("metric"), logY: $("log-y"),
  plot: $("plot"), version: $("version"), caption: $("caption"), terminal: $("terminal"), clear: $("clear"),
};

// ---------------------------------------------------------------- the editor

const editor = new EditorView({
  doc: "",
  extensions: [
    basicSetup,
    yamlLanguage(),
    EditorView.updateListener.of((u) => { if (u.docChanged) onEdit(); }),
  ],
  parent: $("editor"),
});

const text = () => editor.state.doc.toString();

function setText(s) {
  editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: s } });
}

/// The span of the editor an error message points at: the line and column of a YAML
/// error, or the item of the link a block error names, else nothing.
function errorSpan(message) {
  const doc = editor.state.doc;
  let m = /at line (\d+) column (\d+)/.exec(message);
  if (m) {
    const line = doc.line(Math.min(+m[1], doc.lines));
    const from = Math.min(line.from + Math.max(+m[2] - 1, 0), line.to);
    return { from, to: Math.max(from, Math.min(from + 1, line.to)) };
  }
  m = /link block (\d+)/.exec(message);
  if (m) {
    const lines = doc.toString().split("\n");
    let inLink = false, n = 0;
    for (let i = 0; i < lines.length; i++) {
      const l = lines[i];
      if (/^link\s*:/.test(l)) { inLink = true; continue; }
      if (inLink && /^\S/.test(l)) break;
      if (inLink && /^\s*-\s/.test(l) && ++n === +m[1]) {
        const line = doc.line(i + 1);
        return { from: line.from, to: line.to };
      }
    }
  }
  return null;
}

function showError(message) {
  ui.error.hidden = !message;
  ui.error.textContent = message || "";
  const span = message ? errorSpan(message) : null;
  const diagnostics = span ? [{ ...span, severity: "error", message }] : [];
  editor.dispatch(setDiagnostics(editor.state, diagnostics));
}

// ---------------------------------------------------------------- the worker

let worker = null;
let specs = [];
let inspected = null;   // the last successful inspection of the text
let inspectId = 0;
let rows = [];          // the completed conditions of the run in progress
let running = false;
let runStart = 0;
let optimised = false;  // whether the browser has installed its optimised code for the module
let probes = [];        // the times of the probes, ms
let probeTimer = null;
let probeStart = 0;

function startWorker() {
  worker = new Worker(`worker.js${location.search}`, { type: "module" });
  worker.onmessage = onMessage;
  worker.onerror = (e) => {
    setStatus(`the engine failed to load: ${e.message || e}`);
  };
}

function setStatus(message, busy = false) {
  ui.status.textContent = message;
  ui.status.classList.toggle("running", busy);
}

function onMessage(e) {
  const m = e.data;
  if (m.type === "ready") {
    ui.version.textContent = `v${m.version}`;
    if (specs.length === 0) {
      specs = m.specs;
      fillMenu();
      const wanted = new URLSearchParams(location.search).get("spec");
      const first = specs.find((s) => s.path === wanted || s.name === wanted) || specs[0];
      ui.select.value = first.path;
      setText(first.text);
    }
    ui.run.disabled = false;
    setStatus("ready");
    startProbing();
    onEdit();
  } else if (m.type === "probe") {
    onProbe(m.ms);
  } else if (m.type === "inspect") {
    if (m.id !== inspectId) return;
    if (m.ok) {
      inspected = m.data;
      showError(null);
      fillPlotControls();
      ui.caption.textContent = inspected.description || inspected.name;
      draw();
    } else {
      showError(m.error);
    }
  } else if (m.type === "condition") {
    rows.push(m.row);
    const n = inspected ? inspected.conditions.length : "?";
    setStatus(`${rows.length}/${n} conditions, ${elapsed()} s`, true);
    log(conditionLine(m.row, n));
    draw();
  } else if (m.type === "batch") {
    const p = m.progress;
    const n = inspected ? inspected.conditions.length : "?";
    setStatus(`condition ${p.index + 1}/${n}: ${p.trials} trials, ${elapsed()} s`, true);
  } else if (m.type === "done") {
    finish(m.data);
  } else if (m.type === "error") {
    finishWithError(m.error);
  }
}

const elapsed = () => ((performance.now() - runStart) / 1e3).toFixed(1);

// ---------------------------------------------------------------- the probes

// V8 runs a fresh module in its baseline tier, about five times slower than its
// optimised code, which it compiles in the background over tens of seconds (and
// caches for the next visit); a campaign started before that runs to its end in the
// baseline tier. A short probe every 1.5 s while the page is idle sees the switch.
const PROBE_EVERY_MS = 1500;
const PROBE_FOR_S = 120;

function startProbing() {
  stopProbing();
  optimised = false;
  probes = [];
  probeStart = performance.now();
  probeTimer = setInterval(() => {
    if (!running && worker) worker.postMessage({ type: "probe" });
  }, PROBE_EVERY_MS);
}

function stopProbing() {
  clearInterval(probeTimer);
  probeTimer = null;
}

function onProbe(ms) {
  probes.push(ms);
  const first = Math.min(...probes.slice(0, 2));
  const since = (performance.now() - probeStart) / 1e3;
  if (probes.length >= 3 && ms < first / 2.5) {
    optimised = true;
    stopProbing();
    if (!running) setStatus(`ready, engine optimised (${since.toFixed(0)} s after load)`);
  } else if (since > PROBE_FOR_S) {
    stopProbing();
    if (!running) setStatus("ready");
  } else if (!running) {
    setStatus(`ready · the browser is still optimising the engine (${since.toFixed(0)} s)…`);
  }
}

// ---------------------------------------------------------------- the menu

function fillMenu() {
  const groups = [["Examples", specs.filter((s) => !s.validation)], ["Validation", specs.filter((s) => s.validation)]];
  ui.select.replaceChildren();
  for (const [label, items] of groups) {
    const g = document.createElement("optgroup");
    g.label = label;
    for (const s of items) {
      const o = document.createElement("option");
      o.value = s.path;
      o.textContent = s.description ? `${s.name} — ${s.description}` : s.name;
      g.appendChild(o);
    }
    ui.select.appendChild(g);
  }
}

ui.select.addEventListener("change", () => {
  const s = specs.find((x) => x.path === ui.select.value);
  if (s) {
    rows = [];
    setText(s.text);
    history.replaceState(null, "", `?spec=${encodeURIComponent(s.name)}`);
  }
});

// ---------------------------------------------------------------- editing

let editTimer = null;
function onEdit() {
  clearTimeout(editTimer);
  editTimer = setTimeout(() => {
    if (!worker) return;
    inspectId += 1;
    worker.postMessage({ type: "inspect", id: inspectId, yaml: text() });
  }, 300);
}

// ---------------------------------------------------------------- running

ui.run.addEventListener("click", () => {
  if (running || !worker) return;
  rows = [];
  running = true;
  runStart = performance.now();
  ui.run.disabled = true;
  ui.stop.disabled = false;
  showError(null);
  setStatus("running…", true);
  if (inspected) {
    const n = inspected.conditions.length;
    log(`$ comnum run ${inspected.name}.yaml --threads 1`, "dim");
    log(`[${inspected.name}] ${n} conditions, seed ${inspected.seed}, 1 thread, stop: ${inspected.stop}`);
  }
  if (!optimised && probeTimer) {
    log("note: the browser has not finished optimising the engine: this run uses its baseline compiler, about five times slower", "dim");
  }
  worker.postMessage({ type: "run", yaml: text() });
  draw();
});

ui.stop.addEventListener("click", () => {
  if (!running) return;
  worker.terminate();
  running = false;
  ui.run.disabled = true;
  ui.stop.disabled = true;
  setStatus("stopped, reloading the engine…");
  stopProbing();
  startWorker();
});

function finish(data) {
  running = false;
  ui.run.disabled = false;
  ui.stop.disabled = true;
  rows = data.results;
  const n = rows.length;
  const trials = rows.reduce((a, r) => a + (r.trials || 0), 0);
  setStatus(`done: ${n} conditions, ${trials} trials in ${data.wall_s.toFixed(1)} s on one thread`);
  draw();
  log(`${n} conditions, ${trials} trials, ${data.wall_s.toFixed(2)} s on one thread`, "dim");
  showChecks(data.checks);
}

function finishWithError(message) {
  running = false;
  ui.run.disabled = false;
  ui.stop.disabled = true;
  setStatus("error");
  showError(message);
  log(`error: ${message}`, "err");
}

// ---------------------------------------------------------------- the plot

const RATE = /^(ber|ser|bler|fer|per|wer)\b/;

/// The default x of a specification: the `along` of a curve or skip rule if the
/// text gives one, else the last axis with several numeric values.
function defaultAxis(axes, conditions) {
  const m = /along:\s*(\w+)/.exec(text());
  if (m && axes.includes(m[1])) return m[1];
  for (let i = axes.length - 1; i >= 0; i--) {
    const values = new Set(conditions.map((c) => c[axes[i]]));
    if (values.size > 1 && conditions.every((c) => typeof c[axes[i]] === "number")) return axes[i];
  }
  return axes[axes.length - 1];
}

function fillPlotControls() {
  const { axes, conditions, metrics } = inspected;
  const keepX = ui.xAxis.value, keepY = ui.metric.value;
  ui.xAxis.replaceChildren(...axes.map((a) => new Option(a, a)));
  ui.metric.replaceChildren(...metrics.map((m) => new Option(m, m)));
  ui.xAxis.value = axes.includes(keepX) ? keepX : defaultAxis(axes, conditions);
  if (metrics.includes(keepY)) ui.metric.value = keepY;
  else { ui.metric.value = metrics[0]; ui.logY.checked = RATE.test(metrics[0] || ""); }
}

for (const el of [ui.xAxis, ui.metric]) el.addEventListener("change", draw);
ui.metric.addEventListener("change", () => { ui.logY.checked = RATE.test(ui.metric.value); draw(); });
ui.logY.addEventListener("change", draw);

const fmt = (v) => (typeof v === "number" ? +v.toPrecision(6) : v);

/// The key of the series a condition belongs to: its values on every axis but x.
function seriesKey(cond, axes, x) {
  return axes.filter((a) => a !== x).map((a) => `${a}=${fmt(cond[a])}`).join(", ");
}

function draw() {
  if (!inspected || typeof Plotly === "undefined") return;
  const { axes, conditions, expectations, name } = inspected;
  const x = ui.xAxis.value, metric = ui.metric.value;
  const byIndex = new Map(conditions.map((c) => [c.index, c]));
  const series = new Map();
  const seriesOf = (key) => {
    if (!series.has(key)) series.set(key, { key, sim: [], expected: new Map() });
    return series.get(key);
  };
  for (const c of conditions) seriesOf(seriesKey(c, axes, x));
  for (const r of rows) {
    if (!(metric in r) || r.trials === 0) continue;
    const c = byIndex.get(r.index) || r;
    seriesOf(seriesKey(c, axes, x)).sim.push({ x: c[x], y: r[metric], lo: r[`${metric}_lo`], hi: r[`${metric}_hi`] });
  }
  for (const e of expectations) {
    if (e.metric !== metric) continue;
    for (const p of e.points) {
      const c = byIndex.get(p.index);
      if (!c) continue;
      const s = seriesOf(seriesKey(c, axes, x));
      const label = e.name || (e.kind === "theory" ? "theory" : "expected");
      if (!s.expected.has(label)) s.expected.set(label, { kind: e.kind, source: e.source, points: [] });
      s.expected.get(label).points.push({ x: c[x], y: p.expected });
    }
  }
  const palette = ["#e8590c", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#d62728", "#7f7f7f", "#bcbd22", "#e377c2"];
  const traces = [];
  const shown = new Set();   // the expectations already in the legend: one entry each
  let i = 0;
  for (const s of series.values()) {
    const color = palette[i++ % palette.length];
    const label = s.key || name;
    s.sim.sort((a, b) => a.x - b.x);
    traces.push({
      name: label, legendgroup: label, x: s.sim.map((p) => p.x), y: s.sim.map((p) => p.y),
      mode: "lines+markers", type: "scatter", line: { color, width: 2 }, marker: { color, size: 7 },
      error_y: {
        type: "data", symmetric: false, visible: s.sim.some((p) => p.hi > p.y),
        array: s.sim.map((p) => p.hi - p.y),
        // a lower bound at zero has no logarithm: no lower bar then
        arrayminus: s.sim.map((p) => (ui.logY.checked && !(p.lo > 0) ? 0 : p.y - p.lo)),
        color, thickness: 1,
      },
      hovertemplate: `${label}<br>${x}=%{x}<br>${metric}=%{y:.4g}<extra></extra>`,
    });
    for (const [elabel, e] of s.expected) {
      e.points.sort((a, b) => a.x - b.x);
      const theory = e.kind === "theory";
      traces.push({
        name: elabel, legendgroup: elabel, showlegend: !shown.has(elabel), x: e.points.map((p) => p.x), y: e.points.map((p) => p.y),
        mode: theory ? "lines" : "markers", type: "scatter",
        line: { color, width: 1.5, dash: "dash" },
        marker: { color, size: 10, symbol: "diamond-open", line: { width: 2, color } },
        hovertemplate: `${elabel}${s.key ? ` (${s.key})` : ""}<br>${e.source || ""}<br>${x}=%{x}<br>${metric}=%{y:.4g}<extra></extra>`,
      });
      shown.add(elabel);
    }
  }
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const fg = dark ? "#e6edf3" : "#1f2328", grid = dark ? "#30363d" : "#e4e8ec";
  const layout = {
    margin: { l: 56, r: 12, t: 16, b: 48 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: fg, size: 12 },
    xaxis: { title: { text: x }, gridcolor: grid, zeroline: false },
    yaxis: { title: { text: metric }, type: ui.logY.checked ? "log" : "linear", gridcolor: grid, zeroline: false, exponentformat: "power" },
    legend: { orientation: "h", y: -0.18, font: { size: 11 } },
    hovermode: "closest",
  };
  // the mode bar overlaps the title on a phone: zooming by pinch is enough there
  const config = { responsive: true, displaylogo: false, displayModeBar: innerWidth >= 700, modeBarButtonsToRemove: ["lasso2d", "select2d"] };
  Plotly.react(ui.plot, traces, layout, config);
}

addEventListener("resize", () => { if (ui.plot.data) Plotly.Plots.resize(ui.plot); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);

// ---------------------------------------------------------------- the terminal

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/// Append a line to the terminal, with the class of its colour.
function log(line, cls = "") {
  const atBottom = ui.terminal.scrollTop + ui.terminal.clientHeight >= ui.terminal.scrollHeight - 4;
  ui.terminal.insertAdjacentHTML("beforeend", `<span class="${cls}">${esc(line)}</span>\n`);
  if (atBottom) ui.terminal.scrollTop = ui.terminal.scrollHeight;
}

ui.clear.addEventListener("click", () => { ui.terminal.textContent = ""; });

/// The line of a completed condition, as the progress bar of `comnum run` prints it.
function conditionLine(r, n) {
  const keys = inspected ? inspected.axes : Object.keys(r).filter((k) => !["index", "status", "trials", "wall_s"].includes(k) && !/_(lo|hi|count|total)$/.test(k));
  const label = keys.map((k) => `${k}=${fmt(r[k])}`).join(", ");
  const metrics = Object.keys(r)
    .filter((k) => `${k}_lo` in r)
    .map((k) => `${k} = ${r[k].toExponential(4)} [${r[`${k}_lo`].toExponential(2)}, ${r[`${k}_hi`].toExponential(2)}]`);
  return `[${r.index + 1}/${n}] (${label}) ${r.status}: ${r.trials} trials, ${r.wall_s.toFixed(2)} s${metrics.length ? "  " : ""}${metrics.join(", ")}`;
}

/// The checks, as `comnum verify` prints them.
function showChecks(checks) {
  if (!checks || checks.length === 0) return;
  const failed = checks.filter((c) => !c.passed).length;
  log(`${inspected ? inspected.name : ""}: ${checks.length} checks, ${failed} failed`, failed ? "fail" : "ok");
  for (const c of checks) {
    const line = `  ${c.passed ? "ok  " : "FAIL"} ${c.name} | ${c.condition} | ${c.metric}: expected ${c.expected.toExponential(4)}, measured ${c.value.toExponential(4)} [${c.lo.toExponential(3)}, ${c.hi.toExponential(3)}]`;
    log(line, c.passed ? "" : "fail");
  }
  const sources = [...new Set(checks.map((c) => c.source).filter(Boolean))];
  if (sources.length) log(`  sources: ${sources.join("; ")}`, "dim");
}

// ---------------------------------------------------------------- start

startWorker();

// A hook for scripts and the browser console: the text of the editor, and a run.
window.comnumPlayground = { text, setText, run: () => ui.run.click(), results: () => rows };
