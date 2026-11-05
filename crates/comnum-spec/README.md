# comnum-spec

[![crates.io](https://img.shields.io/crates/v/comnum-spec.svg)](https://crates.io/crates/comnum-spec)
[![docs.rs](https://img.shields.io/docsrs/comnum-spec)](https://docs.rs/comnum-spec)

Part of [comnum](https://github.com/com-rs/comnum), link-level simulation of digital communication systems,
wireless and optical. Layer 4 (integration) of the workspace, whose dependencies
point downward only ([architecture](https://github.com/com-rs/comnum/blob/main/docs/ARCHITECTURE.md)).

Declarative campaigns: a YAML, JSON or TOML file describes the grid, the link as a list of blocks, the metrics, the stopping rule and the results expected from the literature. The crate also reads and writes recordings (SigMF, raw IQ, NPY) and provides the `comnum` command:

```bash
cargo install comnum-spec
comnum blocks              # the blocks of a link and their parameters
comnum run campaign.yaml   # reports under results/
comnum verify check.yaml   # fails if a result leaves its reference
```

Most users depend on the facade [`comnum`](https://crates.io/crates/comnum), which
re-exports this crate as `comnum::spec`. The API is documented on
[docs.rs/comnum-spec](https://docs.rs/comnum-spec), with the formula that every item implements.

## Blocks from other crates

A project built on comnum (a learned receiver, a model that depends on unpublished
code) adds its own blocks to the declarative links without comnum depending on it
([ADR 0006](https://github.com/com-rs/comnum/blob/main/docs/adr/0006-external-blocks.md)).
`comnum_spec::blocks::register` adds a block to the registry: it is then listed by
`comnum blocks`, described by the JSON Schema, checked by `comnum check` and built like
the others; `comnum_spec::cli::main` is the command line as a function, so that the
crate ships its own binary with its blocks:

```rust
use comnum_spec::blocks::{register, Extension};

fn main() -> std::process::ExitCode {
    register(Extension { info: my_block_info(), build: |params, label, ctx| build_my_block(params, label, ctx) })
        .expect("a block of that name exists already");
    comnum_spec::cli::main(std::env::args().skip(1))
}
```

## Licence

MIT or Apache-2.0, at your option ([`LICENSE-MIT`](https://github.com/com-rs/comnum/blob/main/LICENSE-MIT),
[`LICENSE-APACHE`](https://github.com/com-rs/comnum/blob/main/LICENSE-APACHE), shipped in the package).
