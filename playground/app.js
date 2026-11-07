// The page of the playground: a CodeMirror editor of the specification, an explorer of
// the specifications of the repository, a worker that runs the WebAssembly engine, a
// Plotly chart of the metric against a variable of the grid, the expected values of
// the expectations (published curves, closed forms) drawn over the simulated points,
// and a terminal that prints the lines of `comnum run` and `comnum verify`.

import { basicSetup, EditorView } from "https://esm.sh/codemirror@6.0.2";
import { yaml as yamlLanguage } from "https://esm.sh/@codemirror/lang-yaml@6.1.2";
import { setDiagnostics } from "https://esm.sh/@codemirror/lint@6.8.5";
import { oneDark } from "https://esm.sh/@codemirror/theme-one-dark@6.1.2";

const $ = (id) => document.getElementById(id);
const ui = {
  list: $("spec-list"), run: $("run"), stop: $("stop"), status: $("status"), engine: $("engine"),
  error: $("error"), xAxis: $("x-axis"), metric: $("metric"), logY: $("log-y"),
  plot: $("plot"), plotTabs: $("plot-tabs"), chain: $("chain"), plotControls: document.querySelector(".plot-controls"), version: $("version"), caption: $("caption"), terminal: $("terminal"), clear: $("clear"),
  tabName: $("tab-name"), share: $("share"), openName: $("open-name"), theme: $("theme"),
  sidebar: $("sidebar"), backdrop: $("sidebar-backdrop"), explorerOpen: $("explorer-toggle"), explorerClose: $("explorer-close"),
  splitH: $("split-h"), splitV: $("split-v"), workbench: document.querySelector(".workbench"), editorArea: document.querySelector(".editor-area"),
};

// ---------------------------------------------------------------- the theme

const systemDark = matchMedia("(prefers-color-scheme: dark)");
const isDark = () => {
  const t = document.documentElement.dataset.theme;
  return t ? t === "dark" : systemDark.matches;
};

ui.theme.addEventListener("click", () => {
  const next = isDark() ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("comnum-theme", next); } catch (e) { /* no storage */ }
  onThemeChange();
});
systemDark.addEventListener("change", onThemeChange);

function onThemeChange() {
  rebuildEditor();
  draw();
}

// The colours of the page, read from its style sheet so that the plot follows the theme.
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---------------------------------------------------------------- the editor

let editor = null;

function makeEditor(doc) {
  return new EditorView({
    doc,
    extensions: [
      basicSetup,
      yamlLanguage(),
      ...(isDark() ? [oneDark] : []),
      EditorView.updateListener.of((u) => { if (u.docChanged) onEdit(); }),
    ],
    parent: $("editor"),
  });
}

/// A new editor with the same text: the theme of CodeMirror is an extension of the view.
function rebuildEditor() {
  const doc = text();
  editor.destroy();
  editor = makeEditor(doc);
  if (lastError) showError(lastError);
}

editor = makeEditor("");

const text = () => editor.state.doc.toString();

function setText(s) {
  editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: s } });
}

$("editor").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); ui.run.click(); }
});

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

let lastError = null;

function showError(message) {
  lastError = message || null;
  ui.error.hidden = !message;
  ui.error.textContent = message || "";
  const span = message ? errorSpan(message) : null;
  const diagnostics = span ? [{ ...span, severity: "error", message }] : [];
  editor.dispatch(setDiagnostics(editor.state, diagnostics));
}

// ---------------------------------------------------------------- the worker

let worker = null;
let specs = [];
let current = null;     // the specification of the explorer the editor holds
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
      fillExplorer();
      const wanted = new URLSearchParams(location.search).get("spec");
      open(specs.find((s) => s.path === wanted || s.name === wanted) || specs[0], false);
      openShared();
    }
    ui.run.disabled = false;
    setStatus("ready");
    startProbing();
    onEdit();
  } else if (m.type === "probe") {
    onProbe(m.ms);
  } else if (m.type === "trace") {
    if (m.ok) {
      traces = m.data;
      log(`traced ${traceNote}: ${traces.map((t) => t.label).join(", ")}`, "dim");
    } else {
      traces = null;
      traceNote = `trace: ${m.error}`;
      log(`error: ${traceNote}`, "err");
    }
    draw();
  } else if (m.type === "inspect") {
    if (m.id !== inspectId) return;
    if (m.ok) {
      inspected = m.data;
      showError(null);
      fillPlotControls();
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
  ui.engine.textContent = "engine: optimising…";
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
    ui.engine.textContent = `engine optimised (${since.toFixed(0)} s after load)`;
    if (!running) setStatus("ready, engine optimised");
  } else if (since > PROBE_FOR_S) {
    stopProbing();
    ui.engine.textContent = "";
    if (!running) setStatus("ready");
  } else if (!running) {
    ui.engine.textContent = `engine: optimising… ${since.toFixed(0)} s`;
  }
}

// ---------------------------------------------------------------- the explorer

function fillExplorer() {
  const groups = [["examples", "specs", specs.filter((s) => !s.validation)], ["validation", "specs/validation", specs.filter((s) => s.validation)]];
  ui.list.replaceChildren();
  for (const [key, label, items] of groups) {
    const head = document.createElement("button");
    head.className = "tree-group";
    head.textContent = label;
    head.setAttribute("aria-expanded", "true");
    head.addEventListener("click", () => head.setAttribute("aria-expanded", head.getAttribute("aria-expanded") !== "true"));
    const box = document.createElement("div");
    box.className = "tree-items";
    box.dataset.group = key;
    for (const s of items) {
      const b = document.createElement("button");
      b.className = "tree-item";
      b.dataset.path = s.path;
      b.title = s.description ? `${s.path}: ${s.description}` : s.path;
      b.innerHTML = `<span class="name">${esc(s.name)}.yaml</span><span class="desc">${esc(s.description || "")}</span>`;
      b.addEventListener("click", () => { open(s, true); closeExplorer(); });
      box.appendChild(b);
    }
    ui.list.append(head, box);
  }
}

/// Load a specification of the explorer into the editor.
function open(s, pushUrl) {
  current = s;
  rows = [];
  traces = null;
  traceNote = "";
  picked = false;
  view = { kind: "curves", key: null };
  setText(s.text);
  ui.tabName.textContent = `${s.name}.yaml`;
  ui.openName.textContent = s.path;
  for (const b of ui.list.querySelectorAll(".tree-item")) b.setAttribute("aria-current", b.dataset.path === s.path);
  if (pushUrl) history.replaceState(null, "", `?spec=${encodeURIComponent(s.name)}`);
}

const openExplorer = () => { ui.sidebar.classList.add("open"); ui.backdrop.hidden = false; };
const closeExplorer = () => { ui.sidebar.classList.remove("open"); ui.backdrop.hidden = true; };
ui.explorerOpen.addEventListener("click", openExplorer);
ui.explorerClose.addEventListener("click", closeExplorer);
ui.backdrop.addEventListener("click", closeExplorer);

// ---------------------------------------------------------------- sharing

// A link carries the text of the specification in its fragment, `#yaml=<payload>`: the
// payload is `z` and the text compressed (deflate) in URL-safe base64, or `t` and the
// text itself in base64 where the browser cannot compress. The fragment never reaches
// a server; a shared specification opens in the editor and runs only on Run.

const toBase64Url = (bytes) => {
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
};
const fromBase64Url = (s) => Uint8Array.from(atob(s.replace(/-/g, "+").replace(/_/g, "/")), (c) => c.charCodeAt(0));

async function pipe(bytes, stream) {
  return new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(stream)).arrayBuffer());
}

async function encodeSpec(yaml) {
  const bytes = new TextEncoder().encode(yaml);
  if (typeof CompressionStream === "undefined") return `t${toBase64Url(bytes)}`;
  return `z${toBase64Url(await pipe(bytes, new CompressionStream("deflate-raw")))}`;
}

async function decodeSpec(payload) {
  const bytes = fromBase64Url(payload.slice(1));
  const raw = payload[0] === "z" ? await pipe(bytes, new DecompressionStream("deflate-raw")) : bytes;
  return new TextDecoder().decode(raw);
}

ui.share.addEventListener("click", async () => {
  const url = `${location.origin}${location.pathname}#yaml=${await encodeSpec(text())}`;
  history.replaceState(null, "", url);
  try {
    await navigator.clipboard.writeText(url);
    setStatus(`link copied (${url.length} characters): it opens this specification in the editor`);
  } catch (e) {
    setStatus("link in the address bar: copy it from there");
  }
  log(`share: ${url}`, "dim");
});

/// Open the specification of the fragment of the address, if it carries one.
async function openShared() {
  const m = /^#yaml=([A-Za-z0-9_-]+)$/.exec(location.hash);
  if (!m) return;
  try {
    const yaml = await decodeSpec(m[1]);
    const name = (/^name:\s*(\S+)/m.exec(yaml) || [])[1] || "shared";
    open({ path: `shared link: ${name}.yaml`, name, text: yaml }, false);
    ui.tabName.textContent = `${name}.yaml`;
    log(`opened a shared specification (${name}); press Run to run it`, "dim");
  } catch (e) {
    showError(`the shared link could not be read: ${e.message || e}`);
  }
}

addEventListener("hashchange", openShared);

// ---------------------------------------------------------------- the sashes

/// Drag a sash: `apply(delta)` receives the movement in pixels since the pointer went down.
function sash(el, axis, apply) {
  el.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    el.setPointerCapture(e.pointerId);
    el.classList.add("dragging");
    const start = axis === "x" ? e.clientX : e.clientY;
    const move = (ev) => apply((axis === "x" ? ev.clientX : ev.clientY) - start);
    const up = () => {
      el.classList.remove("dragging");
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      el.removeEventListener("pointercancel", up);
      if (ui.plot.data) Plotly.Plots.resize(ui.plot);
    };
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
  });
}

sash(ui.splitH, "x", (() => {
  let base = 0;
  return (delta) => {
    if (delta === 0 || base === 0) base = ui.editorArea.querySelector(".editor-pane").getBoundingClientRect().width - delta;
    const total = ui.editorArea.getBoundingClientRect().width;
    const w = Math.min(Math.max(base + delta, 240), total - 240);
    ui.editorArea.style.setProperty("--editor-w", `${w}px`);
    if (ui.plot.data) Plotly.Plots.resize(ui.plot);
  };
})());

sash(ui.splitV, "y", (() => {
  let base = 0;
  return (delta) => {
    if (delta === 0 || base === 0) base = document.querySelector(".panel").getBoundingClientRect().height + delta;
    const total = ui.workbench.getBoundingClientRect().height;
    const h = Math.min(Math.max(base - delta, 60), total - 160);
    ui.workbench.style.setProperty("--panel-h", `${h}px`);
  };
})());

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
  if (running || !worker || ui.run.disabled) return;
  rows = [];
  running = true;
  runStart = performance.now();
  ui.run.disabled = true;
  ui.stop.disabled = false;
  showError(null);
  setStatus("running…", true);
  if (inspected) {
    const n = inspected.conditions.length;
    log(`$ comnum run ${inspected.name}.yaml --threads 1`, "prompt");
    log(`[${inspected.name}] ${n} conditions, seed ${inspected.seed}, 1 thread, stop: ${inspected.stop}`);
  }
  if (!optimised && probeTimer) {
    log("note: the browser has not finished optimising the engine: this run uses its baseline compiler, about five times slower", "dim");
  }
  requestTrace();
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
  log("^C stopped", "dim");
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

ui.xAxis.addEventListener("change", draw);
ui.metric.addEventListener("change", () => { ui.logY.checked = RATE.test(ui.metric.value); draw(); });
ui.logY.addEventListener("change", draw);

const fmt = (v) => (typeof v === "number" ? +v.toPrecision(6) : v);

/// The key of the series a condition belongs to: its values on every axis but x.
function seriesKey(cond, axes, x) {
  return axes.filter((a) => a !== x).map((a) => `${a}=${fmt(cond[a])}`).join(", ");
}

/// The colours of the curves: the six of the logo, then four more.
function palette() {
  return ["--blue", "--orange", "--green", "--violet", "--amber", "--red"].map(cssVar)
    .concat(["#17a2b8", "#8c564b", "#7f7f7f", "#e377c2"]);
}

/// The keys of the series of the grid for the chosen x, in the order of the conditions.
function curveKeys() {
  const { axes, conditions } = inspected;
  return [...new Set(conditions.map((c) => seriesKey(c, axes, ui.xAxis.value)))];
}

/// The labels of the `scope` blocks of the link.
const scopeLabels = () => (inspected.link || []).filter((b) => b.kind === "scope").map((b) => b.label);

function draw() {
  if (!inspected || typeof Plotly === "undefined") return;
  const keys = curveKeys();
  const p = palette();
  const colorOf = new Map(keys.map((k, j) => [k, p[j % p.length]]));
  renderTabs(keys, colorOf);
  const chain = view.kind === "chain";
  ui.chain.hidden = !chain;
  ui.plot.hidden = chain;
  ui.plotControls.hidden = view.kind !== "curves";
  if (chain) drawChain();
  else if (view.kind === "scope") drawScope();
  else drawCurves(colorOf);
}

function drawCurves(colorOf) {
  ui.caption.textContent = inspected.description || inspected.name;
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
  const traces = [];
  const shown = new Set();   // the expectations already in the legend: one entry each
  for (const s of series.values()) {
    if (view.key !== null && s.key !== view.key) continue;
    const color = colorOf.get(s.key) || palette()[0];
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
  const layout = {
    ...baseLayout(),
    margin: { l: 56, r: 16, t: 12, b: 48 },
    xaxis: { ...axisStyle(), title: { text: x } },
    yaxis: { ...axisStyle(), title: { text: metric }, type: ui.logY.checked ? "log" : "linear", exponentformat: "power" },
  };
  Plotly.react(ui.plot, traces, layout, plotConfig());
}

/// The layout every figure shares: transparent, the colours of the theme.
function baseLayout() {
  return {
    margin: { l: 56, r: 16, t: 12, b: 48 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: cssVar("--fg"), size: 12 },
    legend: { orientation: "h", y: -0.18, font: { size: 11 } },
    hovermode: "closest",
  };
}

const axisStyle = () => ({ gridcolor: cssVar("--grid"), zeroline: false, linecolor: cssVar("--grid"), tickcolor: cssVar("--muted") });

// the mode bar is in the way on a phone: zooming by pinch is enough there
const plotConfig = () => ({ responsive: true, displaylogo: false, displayModeBar: innerWidth >= 700, modeBarButtonsToRemove: ["lasso2d", "select2d"] });

addEventListener("resize", () => { if (ui.plot.data) Plotly.Plots.resize(ui.plot); });

// ---------------------------------------------------------------- the tabs of the plot

// What the right pane shows: the curves (all, or the series of `key` alone), the chain
// of blocks, or the signal of a scope.
let view = { kind: "curves", key: null };
let picked = false;        // whether the user chose the tab (else the page chooses)
let tabSignature = "";     // what the tabs were built for
let traces = null;         // the signals of the scopes of the last traced trial
let traceNote = "";        // the condition traced, or the error of the trace

function setView(v, byUser) {
  view = v;
  if (byUser) picked = true;
  draw();
}

/// The tabs: the curves (one per series when the grid gives several, after All), the
/// chain, then one per scope.
function renderTabs(keys, colorOf) {
  const several = keys.length > 1;
  const scopes = scopeLabels();
  if (view.kind === "curves" && view.key !== null && !keys.includes(view.key)) view = { kind: "curves", key: null };
  if (view.kind === "scope" && !scopes.includes(view.label)) view = { kind: "curves", key: null };
  // a specification with scopes and one condition opens on its chain, then its first scope
  if (!picked && scopes.length && inspected.conditions.length === 1) {
    view = traces && traces.length ? { kind: "scope", label: view.kind === "scope" ? view.label : traces[0].label } : { kind: "chain" };
  }
  const tabs = [];
  if (!(scopes.length && inspected.conditions.length === 1)) {
    tabs.push({ v: { kind: "curves", key: null }, label: several ? "All" : "Plot", icon: "plot" });
    if (several) for (const k of keys) tabs.push({ v: { kind: "curves", key: k }, label: k, color: colorOf.get(k) });
  }
  tabs.push({ v: { kind: "chain" }, label: "Chain", icon: "chain" });
  for (const l of scopes) tabs.push({ v: { kind: "scope", label: l }, label: l, icon: "scope" });
  const signature = JSON.stringify(tabs.map((t) => [t.label, t.color || ""]));
  if (signature !== tabSignature) {
    tabSignature = signature;
    ui.plotTabs.replaceChildren(...tabs.map((t) => {
      const b = document.createElement("button");
      b.className = "tab";
      b.setAttribute("role", "tab");
      b.dataset.view = JSON.stringify(t.v);
      const mark = document.createElement("span");
      if (t.color) {
        mark.className = "swatch";
        mark.style.background = t.color;
      } else {
        mark.className = `tab-icon ${t.icon}`;
      }
      b.append(mark, t.label);
      b.addEventListener("click", () => setView(t.v, true));
      return b;
    }));
  }
  const current = JSON.stringify(view);
  for (const b of ui.plotTabs.children) {
    const on = b.dataset.view === current;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", on);
    if (on) b.scrollIntoView({ block: "nearest", inline: "nearest" });
  }
}

// ---------------------------------------------------------------- the chain

const shortValue = (v) => {
  const t = typeof v === "string" ? v : JSON.stringify(v);
  return t.length > 28 ? `${t.slice(0, 27)}…` : t;
};

/// The chain: the blocks of the link on one horizontal line, joined by arrows, and the
/// scopes hung under the arrow where they take the signal. A drawing only.
function drawChain() {
  const scopes = inspected.link.filter((b) => b.kind === "scope").length;
  ui.caption.textContent = `${inspected.link.length - scopes} blocks${scopes ? `, ${scopes} scopes` : ""}`;
  // the scopes between two blocks belong to the arrow between them
  const mains = [], taps = [[]];
  for (const b of inspected.link) {
    if (b.kind === "scope") taps[taps.length - 1].push(b);
    else { mains.push(b); taps.push([]); }
  }
  const box = (b, cls) => {
    const el = document.createElement("div");
    el.className = cls;
    const kind = b.kind === "scope" ? `scope · ${b.params.view || "constellation"}` : b.kind;
    const params = Object.entries(b.params).filter(([k]) => !(b.kind === "scope" && k === "view"));
    el.innerHTML = `<span class="chain-kind">${esc(kind)}</span>`
      + (b.label !== b.kind ? `<span class="chain-label">${esc(b.label)}</span>` : "")
      + params.map(([k, v]) => `<span class="chain-param"><span>${esc(k)}</span> ${esc(shortValue(v))}</span>`).join("");
    el.title = JSON.stringify(b.params);
    return el;
  };
  const arrow = (hung) => {
    const el = document.createElement("div");
    el.className = `chain-link${hung.length ? " tapped" : ""}`;
    const line = document.createElement("div");
    line.className = "chain-arrow";
    const col = document.createElement("div");
    col.className = "chain-taps";
    col.append(...hung.map((b) => box(b, "chain-block scope")));
    el.append(line, col);
    return el;
  };
  const items = [];
  if (taps[0].length) items.push(arrow(taps[0]));
  mains.forEach((b, i) => {
    items.push(box(b, "chain-block"));
    const next = taps[i + 1];
    if (i < mains.length - 1 || next.length) items.push(arrow(next));
  });
  ui.chain.replaceChildren(...items);
}

// ---------------------------------------------------------------- the scopes

/// The signal of a scope: constellation, spectrum, eye or time.
function drawScope() {
  const t = traces && traces.find((x) => x.label === view.label);
  const layout = baseLayout();
  if (!t) {
    ui.caption.textContent = traceNote || "press Run: the page traces one trial and draws the signal of every scope";
    Plotly.react(ui.plot, [], { ...layout, xaxis: { ...axisStyle(), visible: false }, yaxis: { ...axisStyle(), visible: false } }, plotConfig());
    return;
  }
  const colors = palette();
  const n = t.streams[0] ? t.streams[0].re.length : 0;
  const multi = t.streams.length > 1;
  const name = (j) => (multi ? `stream ${j}` : t.label);
  const sps = t.sps > 0 ? t.sps : 1;
  const where = `${t.label} · ${t.view} · ${n} samples, ${fmt(sps)} per symbol${traceNote ? ` · ${traceNote}` : ""}`;
  ui.caption.textContent = where;
  let data = [];
  if (t.view === "constellation") {
    data = t.streams.map((s, j) => ({
      name: name(j), x: s.re, y: s.im, mode: "markers", type: "scatter",
      marker: { color: colors[j % colors.length], size: 3, opacity: 0.6 },
      hovertemplate: "%{x:.3f} %{y:+.3f}j<extra></extra>",
    }));
    layout.xaxis = { ...axisStyle(), title: { text: "in phase" }, zeroline: true, zerolinecolor: cssVar("--grid") };
    layout.yaxis = { ...axisStyle(), title: { text: "quadrature" }, scaleanchor: "x", scaleratio: 1, zeroline: true, zerolinecolor: cssVar("--grid") };
  } else if (t.view === "spectrum" && t.spectrum) {
    data = t.spectrum.psd.map((d, j) => ({
      name: name(j), x: t.spectrum.f, y: d.map((v) => 10 * Math.log10(v)), mode: "lines", type: "scatter",
      line: { color: colors[j % colors.length], width: 1.5 },
      hovertemplate: "f = %{x:.3f} R<sub>s</sub><br>%{y:.1f} dB<extra></extra>",
    }));
    layout.xaxis = { ...axisStyle(), title: { text: "frequency / symbol rate" } };
    layout.yaxis = { ...axisStyle(), title: { text: "power spectral density (dB)" } };
  } else if (t.view === "eye") {
    const k = Math.round(sps);
    if (k < 2) {
      ui.caption.textContent = `${where} · the eye needs more than one sample per symbol: put the scope before the decimation`;
    } else {
      // segments of two symbols, at most 200, the symbol instants (the phase of largest
      // power, as the max-variance decimation picks it) at 0.5 and 1.5
      const s = t.streams[0];
      const power = new Array(k).fill(0);
      s.re.forEach((v, i) => { power[i % k] += v * v + s.im[i] * s.im[i]; });
      const phase = power.indexOf(Math.max(...power));
      const first = (((phase - Math.floor(k / 2)) % k) + k) % k;
      const x = [], y = [];
      for (let start = first + k, m = 0; start + 2 * k < s.re.length && m < 200; start += k, m++) {
        for (let i = 0; i <= 2 * k; i++) { x.push(i / k); y.push(s.re[start + i]); }
        x.push(null); y.push(null);
      }
      data = [{ name: "real part", x, y, mode: "lines", type: "scatter", line: { color: colors[0], width: 1 }, opacity: 0.35, hoverinfo: "skip" }];
    }
    layout.xaxis = { ...axisStyle(), title: { text: "time (symbols)" } };
    layout.yaxis = { ...axisStyle(), title: { text: "real part" } };
  } else {
    const s = t.streams[0];
    const x = s.re.map((_, i) => i / sps);
    data = [
      { name: "real", x, y: s.re, mode: "lines", type: "scatter", line: { color: colors[0], width: 1.2 } },
      { name: "imaginary", x, y: s.im, mode: "lines", type: "scatter", line: { color: colors[1], width: 1.2 } },
    ];
    layout.xaxis = { ...axisStyle(), title: { text: "time (symbols)" }, range: [0, Math.min(n / sps, 64)] };
    layout.yaxis = { ...axisStyle(), title: { text: "amplitude" } };
  }
  layout.showlegend = data.length > 1;
  Plotly.react(ui.plot, data, layout, plotConfig());
}

/// Trace the first trial of the first condition, before the campaign: the worker
/// answers in order, so the scopes are drawn before the first point of the curves.
function requestTrace() {
  if (!inspected || !scopeLabels().length) return;
  const c = inspected.conditions[0];
  const label = c ? inspected.axes.map((a) => `${a}=${fmt(c[a])}`).join(", ") : "";
  traceNote = `trial 0${label ? ` of (${label})` : ""}`;
  worker.postMessage({ type: "trace", yaml: text(), condition: 0 });
}

// ---------------------------------------------------------------- the terminal

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/// Append a line to the terminal, with the class of its colour.
function log(line, cls = "") {
  const atBottom = ui.terminal.scrollTop + ui.terminal.clientHeight >= ui.terminal.scrollHeight - 4;
  ui.terminal.insertAdjacentHTML("beforeend", `<span class="${cls}">${esc(line)}</span>\n`);
  if (atBottom) ui.terminal.scrollTop = ui.terminal.scrollHeight;
}

ui.clear.addEventListener("click", () => { ui.terminal.textContent = ""; });

/// A number as `comnum run` prints it; JSON has no infinity, an unbounded interval
/// arrives as null.
const exp = (v, digits, missing = "nan") => (typeof v === "number" ? v.toExponential(digits) : missing);

/// The line of a completed condition, as the progress bar of `comnum run` prints it.
function conditionLine(r, n) {
  const keys = inspected ? inspected.axes : Object.keys(r).filter((k) => !["index", "status", "trials", "wall_s"].includes(k) && !/_(lo|hi|count|total)$/.test(k));
  const label = keys.map((k) => `${k}=${fmt(r[k])}`).join(", ");
  const metrics = Object.keys(r)
    .filter((k) => `${k}_lo` in r)
    .map((k) => `${k} = ${exp(r[k], 4)} [${exp(r[`${k}_lo`], 2, "-inf")}, ${exp(r[`${k}_hi`], 2, "inf")}]`);
  return `[${r.index + 1}/${n}] (${label}) ${r.status}: ${r.trials} trials, ${r.wall_s.toFixed(2)} s${metrics.length ? "  " : ""}${metrics.join(", ")}`;
}

/// The checks, as `comnum verify` prints them.
function showChecks(checks) {
  if (!checks || checks.length === 0) return;
  const failed = checks.filter((c) => !c.passed).length;
  log(`$ comnum verify ${inspected ? inspected.name : "spec"}.yaml`, "prompt");
  log(`${inspected ? inspected.name : ""}: ${checks.length} checks, ${failed} failed`, failed ? "fail" : "ok");
  for (const c of checks) {
    const line = `  ${c.passed ? "ok  " : "FAIL"} ${c.name} | ${c.condition} | ${c.metric}: expected ${exp(c.expected, 4)}, measured ${exp(c.value, 4)} [${exp(c.lo, 3, "-inf")}, ${exp(c.hi, 3, "inf")}]`;
    log(line, c.passed ? "" : "fail");
  }
  const sources = [...new Set(checks.map((c) => c.source).filter(Boolean))];
  if (sources.length) log(`  sources: ${sources.join("; ")}`, "dim");
}

// ---------------------------------------------------------------- start

startWorker();

// A hook for scripts and the browser console: the text of the editor, and a run.
window.comnumPlayground = { text, setText, run: () => ui.run.click(), results: () => rows, open: (name) => { const s = specs.find((x) => x.name === name); if (s) open(s, true); } };
