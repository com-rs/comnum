# comnum for programs and agents

comnum simulates digital communication links (wireless and optical) and checks every
block against a closed form, a table of a standard, a published figure or a reference
tool. Use it to get numbers you can trust (an error rate with its confidence interval,
an operating point) and to check numbers produced elsewhere. Write a declarative
specification, check it, run it, read one line of JSON. `comnum guide` prints this page.

## Install

```text
cargo install comnum-spec     # the command `comnum`
pip install comnum            # the Python package: comnum.run_spec(dict | path)
```

Both are prebuilt or compile once; nothing runs at import time.

## The loop

1. `comnum blocks --json` lists the blocks: name, summary, ports (`needs`, `gives`),
   `required` parameters and the JSON Schema of every parameter, with its unit
   (`x-unit`). `comnum blocks NAME --json` gives one block. `comnum schema` gives the
   JSON Schema of a block of a link.
2. Write the specification (YAML or JSON). Pass `-` to read it from standard input.
3. `comnum check SPEC --json`: builds every condition and runs one trial, writes
   nothing, takes a fraction of a second. Fix the errors it reports before running.
4. `comnum run SPEC --json`: one line of JSON per specification on standard output.
5. `comnum verify SPEC --json` when the specification has an `expect` section (a closed
   form or a published curve): every check with its expected and measured values.

Every `--json` command prints only JSON on standard output, one document per line, and
exits with 0 (success), 1 (a failure, the document says which) or 2 (a usage error).

## A specification

```yaml
name: qpsk_awgn
description: QPSK over AWGN
seed: 1
grid: {ebn0_db: [0, 2, 4, 6, 8]}
link:
  - {type: source, constellation: qpsk, n_symbols: 1000}
  - {type: awgn_ebn0, ebn0_db: $ebn0_db}
  - demap
metrics: {report: [ber, ser]}
stop: {min_errors: {ber: 200}, max_trials: 2000}
```

- `grid`: every axis is a list, `{from, to, step}` or `{linspace: [start, stop, count]}`;
  the conditions are their product. `$name` in a block reads the value of an axis.
  `cases: [{...}, {...}]` adds named variants (another constellation, another receiver).
- `link`: the blocks in order; a block without parameters is its name (`demap`).
- `metrics.report`: `ber`, `ser`, `bler`, `coded_ber` (with `fec_decode`), `snr_db`,
  `snr_eff_db`, `evm`, `mi`, `gmi`, `ngmi`, `air`.
- `stop`: `max_trials`, `min_trials`, `min_errors: {metric: n}`,
  `precision: {metric: relative half-width}`, `max_time_s`; the condition stops when any
  bound is met. Always give `max_trials`: a rule on errors alone never ends at a low
  error rate.
- `seed`: the results are the same on every machine and for any number of threads
  (except a condition cut by a time budget, see below).

A coded link (a regular LDPC code built by progressive edge growth, 16-QAM, bit
interleaving):

```yaml
name: ldpc_16qam
description: LDPC (1024, 512) with 16-QAM over AWGN
grid: {ebn0_db: {from: 2.5, to: 4.0, step: 0.5}}
link:
  - {type: fec_encode, code: {type: ldpc, n: 1024, k: 512, seed: 1}, blocks: 8, interleave: true}
  - {type: map, constellation: 16-QAM}
  - {type: awgn_ebn0, ebn0_db: $ebn0_db}
  - demap
  - {type: fec_decode, code: {type: ldpc, n: 1024, k: 512, seed: 1}, interleave: true}
metrics: {report: [ber, coded_ber, bler]}
stop: {min_errors: {bler: 100}, max_trials: 400}
```

## The JSON of a run

```text
{"ok": true, "command": "run", "spec": "qpsk_awgn", "budget_s": 300.0,
 "provenance": {"comnum": "0.2.0", "seed": 1, "threads": 0, "confidence": 0.95,
                "fingerprint": "<FNV-1a of the specification>", "reports": {...}},
 "conditions": [{"condition": {"ebn0_db": 4}, "status": "completed", "trials": 50,
                 "wall_s": 0.002, "time_limited": false,
                 "metrics": {"ber": {"value": 0.0126, "lo": 0.0119, "hi": 0.0133,
                                     "count": 1261, "total": 100000}}}],
 "wall_s": 0.01}
```

`lo` and `hi` are the confidence interval at `confidence`; `count` and `total` are the
errors and the trials of a rate. Report the interval with the value. The CSV and JSON
reports of the run are also written (`--out DIR`, `results/` by default).

## Time budget

With `--json`, a run stops after 300 s (`--max-seconds S` to change it, `--max-seconds 0`
for no limit), spread over its conditions. A condition the budget stopped before its
rule says `"time_limited": true`: its estimate is valid but less precise than asked, and
it depends on the machine. Start with a small grid and `max_trials`, then widen.

## Errors

```text
{"ok": false, "command": "check", "file": "-",
 "error": {"kind": "config", "message": "... unknown block \"awgn_ebno\" ...",
           "hint": "`comnum blocks --json` lists the blocks and their parameters",
           "suggestion": "type: awgn_ebn0"}}
```

`kind` is `usage`, `config`, `shape`, `numerical`, `condition` or `io`. `suggestion`
gives the closest block or parameter name for a misspelling, or the block that takes a
parameter given to the wrong one.

## Pitfalls

- `awgn` takes `snr_db` (an SNR in a bandwidth, at the samples); `awgn_ebn0` takes
  `ebn0_db` (per information bit, the matched-filter definition). They differ by the
  rate and the samples per symbol: use `awgn_ebn0` to compare with an E_b/N_0 curve.
- Units are in the names: `_db`, `_dbm`, `_hz`, `_km`, `_s`. A rate is a fraction, not a
  percentage.
- With fewer than about 10 errors (`count`) an error rate is coarse: raise
  `max_trials` or `min_errors`, or report the upper bound `hi`.
- Validation: what each block is checked against, with the tolerance, is in
  `docs/LITERATURE.md` and `docs/MATURITY.md` of the repository
  (https://github.com/com-rs/comnum).
