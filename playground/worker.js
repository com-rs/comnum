// The Web Worker of the playground: loads the WebAssembly module once, then answers the
// page's messages. A run blocks this worker, not the page; the module calls back for
// every completed condition and for the progress of the batches, relayed as messages.

import init, { inspect, run, specs, version } from "./pkg/comnum.js";

const ready = init().then(() => {
  postMessage({ type: "ready", version: version(), specs: JSON.parse(specs()) });
});

onmessage = async (e) => {
  await ready;
  const { type, yaml, id } = e.data;
  if (type === "inspect") {
    try {
      postMessage({ type: "inspect", id, ok: true, data: JSON.parse(inspect(yaml)) });
    } catch (err) {
      postMessage({ type: "inspect", id, ok: false, error: String(err) });
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
