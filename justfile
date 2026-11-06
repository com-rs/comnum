# Development tasks of comnum (https://github.com/casey/just).
# `cargo` without `-p` works on the default members: every crate but the Python
# extension, which maturin builds.

# Windows runners of CI run the recipes with the bash of Git
set windows-shell := ["bash", "-cu"]

# The specifications of specs/validation that `just check` and CI verify (the only list:
# CI runs `just verify-ci`)
ci_specs := "specs/validation/proakis_awgn.yaml specs/validation/information_rates.yaml specs/validation/hager2021_fig5.yaml specs/validation/msk_bpsk.yaml specs/validation/thompson2008_eq15.yaml"

# The Python environment .venv at the versions of uv.lock: the package, built by maturin,
# with the dependency groups test and lint (`uv sync --group docs` adds the documentation;
# `uv lock --upgrade` moves the versions, a deliberate act)
[doc('The Python environment .venv at the versions of uv.lock')]
setup-py:
    uv sync --locked

# Build the Python extension in the virtual environment
develop:
    maturin develop --release

# Rust and Python tests (doctests included)
test: develop
    cargo test
    PYTHONWARNDEFAULTENCODING=1 python -m pytest

# The statistical validation (slow): simulated curves against closed forms
validation:
    cargo test --release -p comnum-validation -- --ignored

# The checks of CI, in its order: run them before declaring a task done (CI also runs
# the Python tests, `just test`, on three systems, plus `just deny` and the MSRV build).
# CI runs the three parts as they are, so that a check cannot differ between the two.
[doc('The checks of CI, in its order: run them before declaring a task done')]
check: check-rust check-tests check-python

# Formatting, lints and the Rust documentation (CI: job rust)
check-rust:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    RUSTDOCFLAGS="--html-in-header {{justfile_directory()}}/docs/katex-header.html -D warnings" cargo doc --workspace --no-deps --exclude comnum-py
    for c in crates/*/; do cmp docs/katex-header.html "$c/katex-header.html"; done

# The Rust tests and the validation specifications of ci_specs (CI: job test)
check-tests: && verify-ci
    cargo test

# The validation specifications that CI verifies
verify-ci:
    cargo run --release -p comnum-spec --bin comnum -- verify {{ci_specs}} --out target/results

# The Python lints and every generated file up to date (CI: job python, on Linux); a
# generated file out of date is rewritten by `just regen`
[doc('Python lints and every generated file up to date (CI: job python)')]
check-python:
    python tools/published_files.py --check
    ruff check python tests docs/python tools benchmarks/libraries
    ruff format --check python tests docs/python tools benchmarks/libraries
    mypy
    python tools/api_index.py --check
    python tools/api_index.py --duplicates --check
    python tools/literature_table.py --check
    python tools/maturity_table.py --check
    python tools/cards.py --check
    python tools/test_layout.py --check
    python tools/benchmarks_page.py --check
    python tools/highlights.py --check
    python tools/gen_stubs.py --check
    python tools/gen_blocks.py --check

# Rewrite every generated file, in the order they depend on each other: the stub and
# the classes of comnum.blocks (from the extension, rebuilt), docs/API.md (which reads
# them), the literature and maturity tables, the benchmark page and the highlights
[doc('Rewrite every generated file (stub, blocks, API index, tables, highlights)')]
regen: develop
    python tools/gen_stubs.py
    python tools/gen_blocks.py
    python tools/api_index.py
    python tools/literature_table.py
    python tools/maturity_table.py
    python tools/benchmarks_page.py
    python tools/highlights.py

# Licences and advisories of the dependencies (needs cargo-deny)
deny:
    cargo deny check

# The WebAssembly module of the browser playground (needs the target
# wasm32-unknown-unknown and wasm-pack): playground/pkg/, not tracked
[doc('The WebAssembly module of the browser playground (wasm-pack)')]
playground:
    wasm-pack build bindings/comnum-wasm --target web --release --out-dir ../../playground/pkg --out-name comnum

# Serve the playground locally (a module worker needs HTTP): http://localhost:8765/
playground-serve:
    python3 -m http.server 8765 --directory playground

# Rerun the comparison of one validation card (validation/cards/<id>.yaml): its Python
# function, Rust test or specification
[doc('Rerun the comparison of one validation card')]
card id:
    python tools/cards.py run {{id}}

# Regenerate docs/API.md, the index of every public item (read it before adding one)
api-index:
    python tools/api_index.py

# Regenerate docs/LITERATURE.md, the comparisons with the literature, from
# validation/literature.yaml
[doc('Regenerate docs/LITERATURE.md from validation/literature.yaml')]
literature:
    python tools/literature_table.py

# Regenerate docs/MATURITY.md and the maturity tables of README.md and
# docs/python/maturity.rst from validation/maturity.yaml
[doc('Regenerate docs/MATURITY.md and the maturity tables')]
maturity:
    python tools/maturity_table.py

# Regenerate the stub of the extension and the classes of comnum.blocks (after changing
# a binding or a block; the extension built)
[doc('Regenerate the stub and the classes of comnum.blocks (extension rebuilt)')]
gen-python: develop
    python tools/gen_stubs.py
    python tools/gen_blocks.py

# The names defined in several modules: candidates for a shared helper
api-duplicates:
    python tools/api_index.py --duplicates

# Refresh the KaTeX header shipped by every crate (docs.rs builds crates in isolation)
katex:
    for c in crates/*/; do cp docs/katex-header.html "$c/katex-header.html"; done

# The examples (results under target/results/)
examples:
    cargo run --release -p comnum --example ber_awgn
    cargo run --release -p comnum --example coherent_link
    cargo run --release -p comnum --example ssfm_dbp

# Criterion benchmarks of the main blocks
bench-rust:
    cargo bench -p comnum

# The documentation of the Python package, with the gallery (docs/python/_build/html)
docs-py: develop
    sphinx-build -W --keep-going -b html docs/python docs/python/_build/html

# Open the Rust documentation
doc:
    cargo doc --workspace --no-deps --exclude comnum-py --open

# Run every specification of specs/ (reports under target/results/)
specs:
    cargo run --release -p comnum-spec -- run specs/*.yaml --out target/results

# Check every specification (one trial per condition)
check-specs:
    cargo run --release -p comnum-spec -- check specs/*.yaml

# Check the results against the literature (specs/validation)
validate:
    cargo run --release -p comnum-spec -- verify specs/validation/*.yaml --out target/results

# Results, speed and footprint against komm, OptiCommPy and Sionna
# (benchmarks/libraries/README.md: a venv with benchmarks/libraries/requirements.txt, on a
# quiet machine, about 15 minutes); regenerates docs/python/benchmarks.rst, the
# highlights of README.md and docs/python/index.rst, and docs/python/why.rst
[doc('Results, speed and footprint against komm, OptiCommPy and Sionna')]
bench-libraries:
    cd benchmarks/libraries && python run_all.py && python footprint.py && python ../../tools/benchmarks_page.py && python ../../tools/highlights.py

# The Linux x86_64 wheel of this checkout for a GPU machine, built here (maturin with
# the zig linker, `uv pip install ziglang` and `rustup target add x86_64-unknown-linux-gnu`
# once), in dist/: send it to the machine and run benchmarks/libraries/runpod.sh --wheel
[doc('The Linux x86_64 wheel of this checkout, for a GPU machine (dist/)')]
runpod-wheel:
    rustup target add x86_64-unknown-linux-gnu
    maturin build --release --zig --target x86_64-unknown-linux-gnu -m bindings/comnum-py/Cargo.toml -o dist

# Fetch the results of benchmarks/libraries/runpod.sh from the machine (its SSH host and
# port, printed by the script; RunPod: the pod's public IP and TCP port 22) into
# benchmarks/libraries/results/<name>/ and render the benchmark page
[doc('Fetch the results of runpod.sh (host, ssh port, name) and render the page')]
runpod-fetch host port name="runpod":
    scp -P {{port}} root@{{host}}:comnum/results-{{name}}.tar.gz /tmp/results-{{name}}.tar.gz
    tar -xzf /tmp/results-{{name}}.tar.gz -C benchmarks/libraries/results
    python tools/benchmarks_page.py
    @echo "benchmarks/libraries/results/{{name}}/*.json: git add -f them, then python tools/highlights.py"

# Speed against OptiCommPy (writes benchmarks/opticommpy/results.md)
bench: develop
    python benchmarks/opticommpy/bench_opticommpy.py

# Rewrite the synthetic capture of examples/data/
make-capture:
    cargo run --release -p comnum --example make_capture -- examples/data
