# Contributing to comnum

Thank you for helping. comnum aims to be a reference link-level simulator: a
contribution is judged first on the evidence that it is right. This page says how to
work on the code, how a block is validated, what the test suites cover, and how
decisions are recorded.

## Good first contributions

- **Reproduce a figure of a paper.** The shortest way into the project, and the most
  useful: a self-contained task with a clear end. The orange and red rows of
  [`docs/LITERATURE.md`](docs/LITERATURE.md) list what is not reproduced yet; a paper is read into a card of its
  values, and [Validating a block](#validating-a-block-against-a-reference) below how
  the comparison is recorded. The issues labelled `validation` are such tasks.
- **A measured signal.** A recording of a software radio (SigMF) that one of the
  receivers decodes, or does not: the LTE cell search, the FM receiver, the receiver
  of `specs/capture_16qam.yaml`. Captures of broadcast programmes are not committed;
  the numbers and the figure are.
- **A gallery example** of a classical result that the gallery does not have yet,
  against its closed form.
- **A disagreement** with a reference or with another tool, reported with the
  specification that reproduces it (the issue template *Validation discrepancy*).

## Before writing code

1. Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): the layers of the workspace,
   the placement rules (where a function belongs), the naming conventions, the
   documentation template and the tests every block family must pass. Its exceptions
   are the decisions of [`docs/adr/`](docs/adr/).
2. Search [`docs/API.md`](docs/API.md), the generated index of every public item,
   before adding a function or a block: it may exist under another name, or a shared
   helper may already do part of the work (`just api-duplicates` lists the names
   defined in several modules).
3. For a change of some size, open an issue first, so that its place and its
   reference can be agreed on.

Two rules matter most: dependencies point downward only (layer 0 `comnum-core`,
`comnum-math`; 1 `comnum-dsp`; 2 the link blocks; 3 `comnum-sim`; 4 `comnum-spec`, `comnum-systems`,
`comnum`), and code is placed by what it does in the chain.

## Setting up

```bash
git clone https://github.com/com-rs/comnum && cd comnum
cargo test                                   # the Rust crates
just setup-py                                # .venv at the versions of uv.lock (needs uv)
source .venv/bin/activate
just test                                    # builds the extension, then Rust and Python tests
```

[`just`](https://github.com/casey/just) runs the tasks of the [`justfile`](justfile).

## Before opening a pull request

Run `just check`. It runs what CI runs, in its order: `cargo fmt --check`, clippy with
`-D warnings`, the tests, `comnum verify` of the validation specifications, the
documentation with `-D warnings` and its KaTeX header, `ruff`, `mypy`, and the checks
that every generated file is up to date (`just regen` rewrites them: `docs/API.md`,
the stubs, the block classes, the literature and maturity tables, the highlights). CI
also runs the Python tests on Linux, macOS and Windows, `cargo-deny`, the MSRV build
and, on a pull request, `cargo-semver-checks` against the base branch.

- Every public item is documented with the formula it implements, in LaTeX
  (`$…$`, `$$…$$`), and its reference.
- Random blocks take `&mut SimRng`; results must not depend on the number of threads
  (ADR 0002). Kernels that produce published results keep their order of operations:
  no `mul_add` or `f64::midpoint` (ADR 0005).
- Invalid inputs are errors (`Error::Config`, `Error::Shape`), never panics, in
  everything a user reaches; a kernel may document a precondition under `# Panics`
  (ADR 0016). `tests/test_no_panic.py` calls every Python function with degenerate
  arguments.
- Add an entry to [`CHANGELOG.md`](CHANGELOG.md) (Keep a Changelog), with a migration
  note for a breaking change (`docs/ARCHITECTURE.md`, *API stability*).

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/), scoped by crate or
area, as in `git log`: `feat(fec): layered min-sum LDPC decoder`,
`fix(spec): number the trials so that dump saves them by index`,
`test(ce_ofdm): …`, `docs(python): …`, `refactor: …`, `ci: …`. The subject says what
the change does; the body says why, and names the reference a result was checked
against. `main` is always green.

## Validating a block against a reference

A block is merged when it is correct by construction (unit and property tests), by
physics (invariants) and by statistics (reference results). The last step ties it to
the literature:

1. **Cite the reference** in the rustdoc of the item: authors, title, venue, year, and
   the equation, figure or table that the block reproduces. A standard is cited with
   its number, version and section. The documents themselves are not in the
   repository (keep yours under `references/<theme>/`, ignored by git; prefer an
   open-access version).
2. **Check it**, with a validation card in [`validation/cards/`](validation/cards/):
   the source with its locator, the expected values with their tolerance and why, the
   findings, and the runner that measures the values again:
   - `function`: a closed form or a deterministic function of the package (the card
     is the test, run by `tests/test_cards.py`);
   - `spec`: a specification under [`specs/validation/`](specs/validation/) with an
     `expect` section, run by `comnum verify`; add it to `ci_specs` of the `justfile`
     if it runs in a few seconds;
   - `test`: a Rust test of `validation/tests/<theme>/<key>.rs`, named after the
     reference, for a chain that needs code or a numerical property; long runs are
     `#[ignore]` (`cargo test --release -p comnum-validation -- --ignored`, or the
     workflow `validation.yml`, started by hand).
   A comparison with a reference tool records the tool's output once
   (`tests/oracle/`, `validation/aff3ct/`) with its version: the tool is never a
   dependency.
3. **Record the evidence** in `validation/literature.yaml` (the document: what was
   reproduced, with the numbers, the tests, the badge, what is not covered, the
   remarks on the document) and `validation/maturity.yaml` (the module: its evidence
   and its gaps), then `just regen`: the pages `docs/LITERATURE.md`, `docs/MATURITY.md`
   and the tables of the README and of the documentation are generated from them. A
   disagreement is a finding: explain it in the card, or report it as an issue.

## The test suites

| Suite | Runs | Covers |
|---|---|---|
| `cargo test` | every push | the unit behaviour of every block, the doctests, the end-to-end checks of each crate (error rates against the theory, round trips, invariants), the literature tests (`mod literature`: tables and closed forms of the documents verbatim) |
| `comnum verify` of `ci_specs` | every push | the validation specifications that run in a few seconds |
| `python -m pytest` | every push, on three systems | the Python API (`tests/test_api.py`: docstring examples, conversions, errors, the generator parity with Rust), the chains and the identity of the two runners (`test_chain.py`), the specifications (`test_spec.py`), the standards (`test_standards.py`), the classical results of the textbooks (`test_performance.py`, marked `slow` on two of the three runners), the validation cards of kind `function` (`test_cards.py`), no panic on degenerate arguments (`test_no_panic.py`), the recorded oracles (`tests/oracle/`: OptiCommPy, GNPy, py3gpp, Sionna, sigmf-python) |
| `cargo test --release -p comnum-validation -- --ignored` | by hand, before a release | the statistical comparisons with published figures, tables and AFF3CT runs that take minutes |
| `just check` | before every commit | everything above that is quick, and the generated files |

A result is checked in one place: a Python test stays only when it is the only caller
of a binding, or when its Rust counterpart is `#[ignore]`.

## The test oracles

`tests/oracle/test_oracle.py` compares the deterministic blocks with OptiCommPy and
`tests/oracle/test_gnpy_oracle.py` the Raman amplification with GNPy. Their outputs are
recorded fixtures of `tests/oracle/` (`<name>.json`, with the tool, its version and
the date, and `<name>.npz` for the arrays), so the tests need neither tool: they are
never imported by the package (OptiCommPy is GPL-3.0). To re-record them after a
change of the tool or of the inputs:

```bash
pip install --group oracle
COMNUM_RECORD_ORACLE=1 python -m pytest tests/oracle/test_oracle.py tests/oracle/test_gnpy_oracle.py
```

Re-recording is a deliberate act: the commit message says why and names the version
of the tool. Re-recording an unchanged oracle produces no diff.

## Architecture decision records

An ADR is required for a new crate, a new dependency between crates of one layer, a
change to a core trait, or an exception to `docs/ARCHITECTURE.md`. Add
`docs/adr/NNNN-short-title.md` (the next number), one page: a status line (accepted,
with the date; superseded by a later ADR when it is), the context, the decision and
its consequences. An ADR is never deleted: a later one supersedes it, and both say
so.

## Reporting problems

A disagreement with a reference is a bug: open a "Validation discrepancy" issue with
the specification that reproduces it. Security issues are reported privately (see
[`SECURITY.md`](SECURITY.md)). Everyone taking part follows the
[code of conduct](CODE_OF_CONDUCT.md).

## Licence

Contributions are released under the licence of the project, MIT or Apache-2.0 at
the user's option, without additional terms (Apache-2.0, section 5).
