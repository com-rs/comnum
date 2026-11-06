# comnum playground

A static web page where a comnum specification (the YAML format of `comnum-spec`, run
by `comnum run` and `comnum.run_spec`) is edited in a code editor and run in the
browser: the Rust engine compiled to WebAssembly, the curves plotted as the conditions
complete, and, for the validation specifications of `specs/validation/`, the published
points of the reference drawn over the simulated curve. A terminal panel prints the
lines of `comnum run` as the conditions complete and those of `comnum verify` at the
end.

The engine is the one of the native build (the same runner, blocks and seeds), on the
one thread of a Web Worker: a point computed in the page is the point of `comnum run`,
to the last bit. Nothing leaves the browser.

## Build

The bindings are the crate `bindings/comnum-wasm` (wasm-bindgen). They need the target
`wasm32-unknown-unknown` and [wasm-pack](https://github.com/drager/wasm-pack):

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack      # or: brew install wasm-pack binaryen
just playground              # wasm-pack build … --out-dir playground/pkg
```

`just playground` runs

```bash
wasm-pack build bindings/comnum-wasm --target web --release --out-dir ../../playground/pkg --out-name comnum
```

which writes `playground/pkg/` (not tracked): `comnum.js`, the module loader, and
`comnum_bg.wasm`, about 2 MB (0.8 MB compressed), optimised by `wasm-opt` when it is
installed. The specifications of `specs/` and `specs/validation/` are embedded in the
module by the build script of the crate (`capture_16qam`, which reads a file, is left
out): rebuild after changing one.

## Serve

The page is static, but a module worker needs HTTP (not `file://`):

```bash
just playground-serve        # python3 -m http.server 8765 --directory playground
```

then open <http://localhost:8765/>. `?spec=<name>` opens a specification of the menu.
CodeMirror 6 (the editor) and Plotly (the plot) are loaded from a CDN; everything else
is in this folder: `index.html`, `style.css`, `app.js` (the page), `worker.js` (the
worker that loads the module and runs the campaign), `pkg/` (built).

## Files and API

The module exports (see `pkg/comnum.d.ts`):

- `specs()`: the specifications of the repository, as JSON (`path`, `name`,
  `description`, `validation`, `text`);
- `inspect(yaml)`: what a specification will run, before it runs: its conditions, its
  metrics, the expected values of its expectations on every condition (the published
  points and closed forms the page draws);
- `run(yaml, on_condition, on_batch)`: the campaign, every completed condition and
  the progress of the batches relayed to the callbacks as JSON, the results and the
  checks of the expectations returned at the end.

A long run blocks the worker only; **Stop** terminates it and loads the module again.
In the console, `comnumPlayground.setText(yaml)` and `comnumPlayground.run()` script
the page.

## Speed

The browser runs the trials one after the other on one thread, with the native
kernels. On a laptop (Apple M-series), with the page visible, Chrome takes 1.8 to 2
times the native one-thread time: a sweep of 11 points of QPSK over AWGN with 10⁶
bits per point 0.18 s against 0.105 s for `comnum run --threads 1`, the example
`qam_awgn` (33 conditions, 24 000 trials of 4096 symbols) 3.6 s against 2.0 s,
`proakis_awgn` 2.5 s against 1.3 s. A hidden tab is throttled by the browser (up to
5 times slower was observed): keep the page in front while it runs. The validation
specifications that take minutes natively on every core (`hager2021_fig5`,
`dvbs2_table13`) take as many minutes times the core count here: use the short ones
(`proakis_awgn`, `msk_bpsk`, `thompson2008_eq15`) or lower `max_trials`.
