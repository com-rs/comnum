// The Web Worker of the playground: loads the WebAssembly module once, then answers the
// page's messages. A run blocks this worker, not the page; the module calls back for
// every completed condition and for the progress of the batches, relayed as messages.

import init, { inspect, run, specs, trace, version } from "./pkg/comnum.js";

// A short campaign, run on request to measure the speed of the module: V8 starts a
// module in its baseline tier (Liftoff) and installs the optimised code (TurboFan) in
// the background, function by function, tens of seconds later for a module this size,
// never inside a call already running; the page probes until the probe speeds up.
const PROBE = `name: probe
grid: {ebn0_db: [0, 4, 8]}
link:
  - {type: source, constellation: QPSK, n_symbols: 1000}
  - {type: awgn_ebn0, ebn0_db: $ebn0_db}
metrics: {report: [ber]}
stop: {max_trials: 8}
`;

// `?nocache=1` on the page fetches the module under a new URL every time, past the
// browser's caches (to measure a cold start).
const params = new URLSearchParams(self.location.search);
const wasmUrl = new URL("./pkg/comnum_bg.wasm" + (params.get("nocache") ? `?v=${Date.now()}` : ""), import.meta.url);
// A module that does not load (not found, served with another type, out of memory) is
// told to the page: the rejection of a promise does not reach `Worker.onerror`.
const ready = init({ module_or_path: wasmUrl })
  .then(() => {
    postMessage({ type: "ready", version: version(), specs: JSON.parse(specs()) });
    return true;
  })
  .catch((err) => {
    postMessage({ type: "failed", error: String(err) });
    return false;
  });

onmessage = async (e) => {
  if (!(await ready)) return;
  const { type, yaml, id } = e.data;
  if (type === "inspect") {
    try {
      postMessage({ type: "inspect", id, ok: true, data: JSON.parse(inspect(yaml)) });
    } catch (err) {
      postMessage({ type: "inspect", id, ok: false, error: String(err) });
    }
  } else if (type === "trace") {
    try {
      postMessage({ type: "trace", ok: true, data: JSON.parse(trace(yaml, e.data.condition || 0)) });
    } catch (err) {
      postMessage({ type: "trace", ok: false, error: String(err) });
    }
  } else if (type === "probe") {
    const t0 = performance.now();
    try {
      run(PROBE, () => {}, () => {});
      postMessage({ type: "probe", ms: performance.now() - t0 });
    } catch (err) {
      // a probe that fails measures nothing: the page stops probing
      postMessage({ type: "probe", failed: true });
    }
  } else if (type === "run") {
    try {
      const out = run(
        yaml,
        (row) => postMessage({ type: "condition", row: JSON.parse(row) }),
        (progress) => postMessage({ type: "batch", progress: JSON.parse(progress) }),
      );
      postMessage({ type: "done", data: JSON.parse(out) });
    } catch (err) {
      postMessage({ type: "error", error: String(err) });
    }
  }
};
