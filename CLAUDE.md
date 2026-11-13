# comnum

Link-level simulation of digital communication systems (wireless and optical) in Rust,
with Python bindings. The architecture, naming, documentation, quality and testing
rules are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); its exceptions are recorded
in [docs/adr/](docs/adr/). Read the guide before adding a block or a crate.

## The two rules to keep in mind

- **Dependencies point downward only.** Layers: 0 `comnum-core`, `comnum-math`; 1 `comnum-dsp`;
  2 `comnum-fec`, `comnum-mapping`, `comnum-waveform`, `comnum-mimo`, `comnum-channel`,
  `comnum-hardware`, `comnum-metrics`; 3 `comnum-sim`; 4 `comnum-spec`, `comnum-systems`, `comnum`.
  Crates of one layer never depend on each other, except `comnum-waveform`,
  `comnum-metrics` (ADR 0007) and `comnum-mimo` (ADR 0008) towards `comnum-mapping`,
  `comnum-waveform → comnum-mimo` (ADR 0009) and `comnum-spec → comnum-systems` (ADR 0010). A new
  edge needs an ADR.
- **Placement:** put code by what it does in the chain; if it depends on a specific
  waveform, it lives under that waveform; if something else could reuse it, it moves
  one layer down. Numerical method without communication meaning → `comnum-math`; type or
  trait every block needs → `comnum-core`; medium → `comnum-channel`; component →
  `comnum-hardware`.

## Before adding a function or a block

Follow these steps, in order, and write them down in the reply (or in the brief of a
subagent) before writing the code:

1. **Read the prototypes.** Search [docs/API.md](docs/API.md), the generated index of
   every public item (signature and first line of documentation), for what the feature
   needs: the kernels, closed forms, containers and blocks that exist already.
2. **Say what is reused.** List the existing items the feature will call; reuse the
   prototype that exists, or move it one layer down when another crate needs it,
   instead of writing it again.
3. **Place it.** Name the crate and module by the placement rule above, and the ADR if
   a new edge is needed.
4. **Name the reference.** When the feature comes from a paper or a standard, name it so
   that it can be downloaded into `references/<theme>/` and the code checked against its
   figures and tables (a validation card, `specs/validation/`, tests named after the
   reference).
5. **Check.** `just api-index` regenerates the index; `just check` fails on a name
   defined in several modules that `docs/api-duplicates.txt` does not accept (factor
   it, or add it with the reason it is not a copy).

## Before declaring a task done

Run `just check`: the checks of CI, in its order (CI runs its three parts,
`check-rust`, `check-tests`, `check-python`, as they are). A generated file out of date
is rewritten by `just regen`.

- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test` (the default members: every crate but the Python extension)
- `comnum verify` of the validation specifications that CI checks (`ci_specs` of the
  `justfile`, the only list)
- `cargo doc --workspace --no-deps --exclude comnum-py` with the KaTeX header and
  `-D warnings`
- every crate ships the same `katex-header.html` as `docs/katex-header.html`
- `python tools/published_files.py --check` (no PDF, no file of `references/`, no reading
  card of `validation/readings/` and no capture is tracked, whatever its name)
- `ruff check` and `ruff format --check` of `python tests docs/python tools benchmarks/libraries`
- `mypy` (`[tool.mypy]` of `pyproject.toml`: the types of the package and of the stub
  `python/comnum/_comnum.pyi`, with the tests as callers of the stub: a method stubbed
  `-> None` whose value is used fails)
- `python3 tools/api_index.py --check` (docs/API.md is up to date) and `--duplicates
  --check` (no name defined in more modules than docs/api-duplicates.txt accepts)
- `python tools/literature_table.py --check` (docs/LITERATURE.md is up to date and every
  test named in `validation/literature.yaml` exists)
- `python tools/maturity_table.py --check` (docs/MATURITY.md and the maturity tables of
  docs/python/maturity.rst and the summary of README.md are generated from `validation/maturity.yaml`,
  and every test it cites exists)
- `python tools/cards.py --check` (every validation card of `validation/cards/` points at
  an existing runner, literature entry and theme folder; `just card <id>` reruns one)
- `python tools/test_layout.py --check` (every test in the place `docs/ARCHITECTURE.md`, "Where a
  test goes", gives it: validation tests in a theme folder declared by its `main.rs`, no
  exploration among the tests, oracle tests in `tests/oracle/`)
- `python tools/highlights.py --check` (the highlights of README.md and
  docs/python/index.rst and the page docs/python/why.rst are computed from the
  repository: the manifests, the benchmark, the validation files)
- `python3 tools/benchmarks_page.py --check` (docs/python/benchmarks.rst is generated from
  the recorded results of `benchmarks/libraries/results/`)
- `python tools/gen_stubs.py --check` (the type stubs `python/comnum/_comnum.pyi` match the
  extension) and `python tools/gen_blocks.py --check` (`python/comnum/blocks.py`, the classes of the
  declarative blocks, matches the registry)
- `python tools/gen_namespaces.py --check` (the thematic namespaces `python/comnum/<theme>.py`,
  `comnum.theory`, `comnum.optical`…, match the sections of `docs/python/api/index.rst`)

CI also runs the Python tests (`just test`: `maturin develop`, then `python -m pytest`;
`just setup-py` makes `.venv` with `uv sync --locked`), `just deny` and the MSRV build.
Statistical validation: `cargo test --release -p comnum-validation -- --ignored`.

## Notes

- comnum contains no block of another project: an external crate adds its blocks with
  `comnum_spec::blocks::register` and ships its own binary (ADR 0006).
- Random blocks take `&mut SimRng` (xoshiro256**, ADR 0002); results must not depend on
  the thread count. Kernels that produce published results keep their order of
  operations: no `mul_add` or `f64::midpoint` (ADR 0005).
- Examples and demonstrations on measured signals (HackRF or SigMF captures) call the
  blocks of comnum only: a missing receiver step is added to the library (placed,
  tested, validated) before the example uses it. Captures of broadcast programmes are
  never committed.
- Every formula is LaTeX in the rustdoc (`$…$`, `$$…$$`, braced subscripts, `\\` before
  punctuation); every crate ships `katex-header.html` (`just katex` refreshes them).
