# comnum-sim

[![crates.io](https://img.shields.io/crates/v/comnum-sim.svg)](https://crates.io/crates/comnum-sim)
[![docs.rs](https://img.shields.io/docsrs/comnum-sim)](https://docs.rs/comnum-sim)

Part of [comnum](https://github.com/com-rs/comnum), link-level simulation of digital communication systems,
wireless and optical. Layer 3 (simulation) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/comnum/blob/main/docs/ARCHITECTURE.md)).

The Monte Carlo engine: a scenario (one random trial of a link) run on every condition of a parameter grid, as many times as a stopping rule requires, on a thread pool, with every metric reported with a confidence interval. The results do not depend on the number of threads.

Most users depend on the facade [`comnum`](https://crates.io/crates/comnum), which
re-exports this crate as `comnum::sim`. The API is documented on
[docs.rs/comnum-sim](https://docs.rs/comnum-sim), with the formula that every item implements.

## Organisation

A **scenario** (one random trial of a link) runs on every **condition** of a **grid**,
as many times as a **stopping rule** requires, in parallel on every core, and every
metric is reported with a confidence interval. The organisation follows PyTorch
Lightning:

| PyTorch Lightning | `comnum::sim` | Role |
|---|---|---|
| `LightningModule` | `Scenario` | `setup` once per condition, `trial` many times |
| `self.log(...)` | `Record` | error counts (Wilson intervals) or values (normal intervals) |
| `Trainer` | `Runner` | batches of trials on a thread pool |
| `seed_everything` | `Runner::seed` | trial `t` of condition `c` draws from `stream(seed, name, c, t)` |
| `max_epochs`, `EarlyStopping` | `Stop` | `max_trials`, `min_errors`, `precision`, `max_time`, combined with `or`/`and` |
| `Callback` | `Callback` | `ProgressBar`, `SkipAfter` (skip the rest of a sweep below a BER floor) |
| `Logger` | `Logger` | `CsvLogger` (a row per condition as it completes), `JsonLogger` |
| `fast_dev_run` | `Runner::fast_dev_run` | one batch per condition |

The records of a batch are merged in trial order, so the results are identical whatever
the number of threads.

```rust
use comnum::prelude::*;
use comnum::sim::scenario::from_fn;

let qpsk = from_fn("qpsk", |cond, rng, rec| {
    let c = Constellation::qam(4)?.normalized();
    let bits = source::bits(2000, rng);
    let rx = awgn::awgn_ebn0(&c.map_bits(&bits)?, cond.f64("ebn0_db")?, 2.0, 1.0, rng);
    rec.errors("ber", metrics::bit_errors(&c.demap_bits(&rx), &bits)? as u64, 2000);
    Ok(())
});
let results = Runner::new()
    .seed(7)
    .stop(Stop::min_errors("ber", 500).or(Stop::max_trials(10_000)))
    .callback(ProgressBar::default())
    .callback(SkipAfter::below("ber", 1e-6, "ebn0_db"))
    .logger(CsvLogger::new("results/qpsk.csv"))
    .run(&qpsk, &Grid::new().axis("ebn0_db", [0.0, 2.0, 4.0, 6.0, 8.0]))?;
```

The Python package runs trials written in Python on the same runner (`comnum.sim`),
with the same streams, statistics, stopping rules and loggers.

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/comnum/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/comnum/blob/main/LICENSE-APACHE), shipped in the package).
