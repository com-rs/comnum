# comnum — Architecture & Engineering Guide

Vincent Choqueuse · revised 30 October 2026

This guide describes the workspace as it is: its layers, where code goes, how it is
named, documented, tested and released. Its exceptions are recorded in the
architecture decision records of [`adr/`](adr/); when the code and the guide disagree,
the guide is wrong and is fixed.

## Purpose and design principles

comnum is a Rust workspace for link-level simulation of digital communication systems,
wireless and optical, with Python bindings (the package `comnum`) and a command line
that runs declarative campaigns (`comnum`). It targets research-grade correctness and
reproducibility first, then performance, then ergonomics.

Six principles govern every design decision. When a new feature does not fit cleanly,
revisit the principles before bending the structure.

1. **Organise by function, never by domain.** A module answers "what does this block
   do in the chain?" (encode, map, shape, propagate, impair, estimate, detect,
   measure). Domains such as optics or a standard only appear as submodules
   (`channel::optical`) or as assembled systems (`comnum-systems`).
2. **Multi-stream containers.** The containers of `comnum-core` (`Bits`, `Llr`,
   `Symbols`, `Signal`) carry an explicit stream dimension, SISO being one stream, and
   the verb traits work on them, so MIMO and polarisation multiplexing reuse the same
   blocks. Most kernels are written on one stream (`&[C64]`, `&[u8]`, `&[f64]`); the
   multi-stream blocks apply them stream by stream (`Signal::map_streams`).
3. **Dependencies point downward only.** Foundation crates know nothing about the
   crates above them. Two crates of the same layer do not depend on each other unless
   an ADR records the edge (four do, ADRs 0007–0010). If a block needs a sibling, the
   shared part moves one layer down.
4. **Propagation and hardware are separate concerns.** `comnum-channel` models the
   medium; `comnum-hardware` models component non-idealities. Any impairment combines
   with any channel.
5. **Physics is explicit.** A quantity carries its unit in its type or in its name.
   The unit newtypes of `comnum-core` (`Db`, `EbN0`, `EsN0`, `Hertz`, `Seconds`) are used
   by the verb-trait blocks, and the decibel conversions of plain numbers (`db2lin`,
   `lin2db`, …) are defined once next to them (`comnum_core::units`, re-exported by
   `comnum_math`); the functions and configuration fields of the numerical
   kernels take `f64` named with their unit (`ebn0_db`, `alpha_db_km`, `fs`,
   `linewidth_hz`, `beta2_s2_km`).
6. **Every result is reproducible.** Randomness is seeded and derived from the trial,
   never from the worker; every JSON report records the library version, the commit
   and the configuration (with its seed) that produced it.

## Workspace layout

The workspace has five layers: foundation, signal processing, link blocks, simulation
and integration. Every crate is published under the `com-` prefix and re-exported by
the facade `comnum`, so users write `comnum::fec::ldpc` rather than `comnum_fec::ldpc`.

```text
comnum/
├── Cargo.toml                 # [workspace]: members, shared lints, dependency versions
├── crates/
│   │  ── Layer 0: foundation ──────────────────────────────────────────────
│   ├── comnum-core/              # signal, traits, units, rng, error
│   ├── comnum-math/              # special, quadrature, log_domain, stats, fft, linalg,
│   │                          # nearest, geometry, nlse, gf2, gf2m, bigint, bits, search,
│   │                          # sequences
│   │  ── Layer 1: generic signal processing ───────────────────────────────
│   ├── comnum-dsp/
│   │   ├── filter  resample  signal  window  interpolation
│   │   ├── sync/              # carrier, timing, frame
│   │   ├── equalizer          # the n×n butterfly: CMA, RDE, NLMS, DD-LMS
│   │   ├── estimation         # LS, LMMSE (waveform-agnostic)
│   │   ├── nonlinear          # Bussgang's decomposition of memoryless nonlinearities
│   │   ├── spectrum  cyclo    # Welch's PSD; cyclic autocorrelation, symbol rate
│   │   └── compensation/      # cd, dbp, dpd
│   │  ── Layer 2: link blocks ─────────────────────────────────────────────
│   ├── comnum-fec/               # crc, scrambling, interleaving, block, conv, puncture,
│   │   │                      # turbo, ldpc, polar, rs, bch, algebraic, ccdm, bounds
│   │   └── nr/                # 5G NR LDPC and polar codes (TS 38.212: ldpc, polar, tables)
│   ├── comnum-mapping/           # constellation, demapper, differential, source, theory,
│   │                          # bicm, imperfect_csi
│   ├── comnum-waveform/
│   │   ├── ofdm/              # grid, estimation, sync, detection, link, correction, nr
│   │   ├── papr               # distribution and reduction of the PAPR
│   │   ├── ce_ofdm/           # constant-envelope OFDM (gamp, theory)
│   │   ├── cpm/               # continuous-phase modulation (pulse, detection)
│   │   ├── css/               # chirp spread spectrum, LoRa (theory)
│   │   ├── fbmc/              # FBMC/OQAM
│   │   └── ufmc  otfs         # universal filtered multicarrier; OTFS
│   ├── comnum-mimo/              # detection, message_passing, receiver, stbc, precoding,
│   │                          # capacity
│   ├── comnum-channel/
│   │   ├── awgn  fading  spatial
│   │   ├── multipath          # TDL profiles, the delay–Doppler channel and its model
│   │   ├── cdl/               # 3GPP TR 38.901 (tables, array)
│   │   ├── owc/               # optical wireless: indoor (rooms, VLC), atmospheric (FSO)
│   │   └── optical/           # fiber, manakov, pmd, edfa, gn, raman/   [feature: optical]
│   ├── comnum-hardware/
│   │   ├── amplifier  oscillator  iq  converter
│   │   └── optical            # laser, modulators, photodiode, receiver [feature: optical]
│   ├── comnum-metrics/           # errors, information, spectrum, blind, stability
│   │  ── Layer 3: simulation ──────────────────────────────────────────────
│   ├── comnum-sim/               # scenario, grid, runner, stopping, callbacks, loggers, stats,
│   │                          # json
│   │  ── Layer 4: integration ─────────────────────────────────────────────
│   ├── comnum-spec/              # spec, params, link, metrics, run, blocks/, verify, search,
│   │                          # io, cli, the `comnum` binary
│   ├── comnum-systems/           # dvbs2/ (with fec), pas
│   └── comnum/                 # facade: re-exports, prelude, examples, benchmarks
├── bindings/comnum-py/           # PyO3 extension of the Python package (built by maturin)
├── python/comnum/              # the Python package: chains, results, plots, generated blocks
├── validation/                # statistical validation (crate comnum-validation): tests/, cards/
│                              # (validation cards), aff3ct/ fixtures, literature.yaml,
│                              # maturity.yaml
├── specs/                     # specifications; specs/validation/: checked against the literature
├── tests/                     # Python tests; tests/oracle/: recorded OptiCommPy, GNPy, py3gpp
│                              # and Sionna outputs
├── examples/                  # runnable examples of the facade; data/: a synthetic SigMF
│                              # capture (`just make-capture`)
├── benchmarks/                # rust/: criterion benchmarks of the facade; opticommpy/: speed
│                              # against OptiCommPy; aff3ct/: speed against AFF3CT;
│                              # libraries/: results and speed against komm, OptiCommPy, Sionna
├── tools/                     # generators and checks: API.md, the stub, comnum.blocks,
│                              # LITERATURE.md, MATURITY.md, the cards, the benchmarks page
├── references/                # the documents the code is checked against (local, ignored by git)
└── docs/                      # this guide, adr/, API.md, LITERATURE.md, MATURITY.md, audit/,
                               # katex-header.html, python/ (Sphinx)
```

## Crate reference

Each crate owns one responsibility and states what it must not contain. The "Must not
contain" column is as binding as the first one.

| Crate | Layer | Owns | Must not contain | Depends on |
| --- | --- | --- | --- | --- |
| `comnum-core` | 0 | Signal containers, verb traits, unit newtypes and the decibel conversions, the generator and its streams, the error type | Any algorithm; any heavy dependency | `num-complex`, `ndarray`, `thiserror` |
| `comnum-math` | 0 | Special functions (`erfc`, `erfcinv`, `qfunc`, `qfuncinv`, `diric`), Simpson, Gauss–Hermite and Gauss–Legendre quadrature, FFT plans and their unitary and centred forms, complex linear algebra (tridiagonal systems included), vectors of 3-D space, nearest-point search, the NLSE split-step solvers, GF(2) and GF(2^m), exact integers, Gray code, LFSR sequences, bisection and golden-section search, decibels, entropy | Communication semantics (no "symbol", no "channel") | `comnum-core`, `rustfft` |
| `comnum-dsp` | 1 | Waveform-agnostic signal processing: filters, resampling, cyclic prefixes and frequency shifts, synchronisation, equalisation, estimation, interpolation, spectral estimation, cyclostationary analysis, memoryless nonlinearities driven by Gaussian signals (Bussgang), compensation, predistortion | Code that assumes a specific waveform or standard | 0 |
| `comnum-fec` | 2 | Channel codes (the 5G NR LDPC and polar codes included), CRC, scrambling, interleaving, puncturing, distribution matching, finite-length bounds | Modulation or channel knowledge beyond LLR input | 0 |
| `comnum-mapping` | 2 | Constellations, labelling, hard and soft demapping, differential PSK, sources, the closed forms of error and information rates (BICM cutoff rates, imperfect channel estimates included) | Pulse shaping or framing | 0 |
| `comnum-waveform` | 2 | Time–frequency structure: OFDM and its grid, PAPR, CE-OFDM, CPM, CSS, FBMC/OQAM, UFMC, OTFS | Channel models; MIMO detection algorithms | 0, 1, `comnum-mapping`, `comnum-mimo` (ADR 0009) |
| `comnum-mimo` | 2 | Detection (message passing included), the receiver of a link, space-time codes, precoding, capacity | Waveform-specific resource mapping | 0, `comnum-mapping` (ADR 0008) |
| `comnum-channel` | 2 | Propagation media: AWGN, fading, spatial correlation, TDL and CDL multipath, delay–Doppler channels, optical fibre, optical wireless (indoor rooms, free-space optics) | Component non-idealities | 0, 1 |
| `comnum-hardware` | 2 | Transmitter and receiver non-idealities, electrical and optical: amplifiers, oscillators, IQ imbalance, converters, lasers, modulators, photodiodes | Propagation effects | 0, 1 |
| `comnum-metrics` | 2 | Error counts and rates, SNR, EVM, information rates, metrics of spectra (ACLR), blind estimates (cumulants, modulation classification, the SNR), the frequency stability of oscillators (Allan variances) | Simulation control flow | 0, 1, `comnum-mapping` (ADR 0007) |
| `comnum-sim` | 3 | Monte Carlo engine, grids, stopping rules, callbacks, loggers, statistics | Physical-layer algorithms | 0 (`rayon`) |
| `comnum-spec` | 4 | Specifications, their blocks and registry, verification, operating-point search, recordings I/O, the `comnum` command (ADR 0004) | Algorithms of the blocks (they live in layers 0–2) | 0–3, `comnum-systems` (ADR 0010); `serde`, `serde_json`, `serde_norway` (YAML), `toml` |
| `comnum-systems` | 4 | Chains of standards and assemblies of several layer-2 crates: `dvbs2` (with its FEC), `pas` | New algorithms (they belong in layers 1–2) | 0–2 |
| `comnum` | 4 | The facade: re-exports, prelude, features `optical` and `spec` | Any code of its own | all |

`comnum-math` depends on `comnum-core` for `C64` only; both are layer 0. Layer 2 has four
sibling edges, all recorded: `comnum-metrics → comnum-mapping` (ADR 0007), `comnum-mimo →
comnum-mapping` (ADR 0008), `comnum-waveform → comnum-mapping` and `comnum-waveform → comnum-mimo`
(ADR 0009). In layer 4, `comnum-spec → comnum-systems` (ADR 0010). Any new edge between two
crates of one layer requires an ADR.

The Python extension (`bindings/comnum-py`) depends on the facade and adds no algorithm:
it converts arrays and calls the crates. `validation/` (the crate `comnum-validation`,
not published) depends on the facade.

## Placement rules

One rule decides where new code goes: **place it by what it does in the chain; if it
depends on a specific waveform, it lives under that waveform; if something else could
reuse it, it moves one layer down.**

Apply the questions in order and stop at the first "yes":

1. Is it a numerical method with no communication meaning? → `comnum-math`.
2. Is it a type or trait every block needs? → `comnum-core`.
3. Does it only make sense for one waveform or standard, or does it assemble several
   crates of layer 2? → that waveform's submodule, or `comnum-systems`.
4. Does it model the medium, or a component? → `comnum-channel` or `comnum-hardware`.
5. Otherwise → the functional crate matching its role in the chain.

Before adding a function, search [`API.md`](API.md), the generated index of every
public item: reuse the prototype that exists, or move it one layer down.

The following cases are settled. Treat them as precedent.

| Item | Location | Rationale |
| --- | --- | --- |
| NLSE split-step solvers (scalar, Manakov) | `comnum_math::nlse` | Shared by forward propagation and back-propagation; avoids a `dsp` → `channel` edge |
| Fibre channel | `comnum_channel::optical::fiber` | Thin wrapper supplying physical parameters (β₂, γ, α) to the solver |
| Digital back-propagation | `comnum_dsp::compensation::dbp` | Receiver processing; runs the same solver with negated parameters |
| Chromatic dispersion compensation | `comnum_dsp::compensation::cd` | An all-pass frequency-domain filter, not an optical concept |
| Digital predistortion | `comnum_dsp::compensation::dpd` | Receiver-side estimation of an inverse; the amplifier models stay in `comnum_hardware::amplifier` |
| Laser phase noise | `comnum_hardware::optical::Laser` | Reuses the Wiener model of `comnum_hardware::oscillator` |
| IQ imbalance | `comnum_hardware::iq` | A component of the transmitter or receiver |
| CE-OFDM | `comnum_waveform::ce_ofdm` | A waveform: a real OFDM signal driving a phase modulator |
| LS / LMMSE estimators | `comnum_dsp::estimation` | Waveform-agnostic |
| Interpolation between known positions (linear, quadratic, spline, band-limited, cubic Lagrange) | `comnum_dsp::interpolation` | Waveform-agnostic; shared by the pilot interpolation of OFDM and the timing recovery |
| Pilot interpolation on a resource grid | `comnum_waveform::ofdm::estimation` | Depends on the grid layout |
| MIMO detectors (ZF, MMSE, SIC, ML, sphere, K-best) | `comnum_mimo::detection` | Operate on a generic $y = H x + w$ model |
| The receiver of a link (layers, Alamouti, soft outputs) | `comnum_mimo::receiver` | Waveform-agnostic; used by the OFDM link and the `mimo_rx` block |
| Per-resource-element MIMO detection | `comnum_waveform::ofdm::{detection, link}` | Thin wrappers applying `comnum_mimo` across the grid (ADR 0009) |
| Polarisation demultiplexing | `comnum_dsp::equalizer` (2×2 CMA/RDE/LMS) | Polarisation multiplexing is 2×2 MIMO; no optical-specific equaliser |
| Soft demapper (LLR) | `comnum_mapping::demapper` | Inverse of the mapper, owned by the same crate |
| Closed forms of error and information rates | `comnum_mapping::theory` | Properties of a constellation and its labels (ADR 0007), re-exported by `comnum_metrics` |
| Gray code, bit ↔ index conversions | `comnum_math::bits` | No communication meaning; shared by mapping, coding and shaping |
| GF(2^m) arithmetic | `comnum_math::gf2m` | Shared by the BCH and Reed–Solomon codes |
| DVB-S2 BCH and LDPC codes and their tables | `comnum_systems::dvbs2::fec` | Only make sense for one standard |
| Probabilistic amplitude shaping | `comnum_systems::pas` | Assembles the CCDM and the LDPC code of `comnum-fec` and a PAM of `comnum-mapping` |
| `ber_to_q_db` | `comnum_metrics::errors` | A way of reporting a measured error rate |
| Bussgang's decomposition, the output series of a nonlinearity driven by a Gaussian signal | `comnum_dsp::nonlinear` | Shared by the clipping of OFDM (`comnum_waveform::papr`) and the amplifier models (`comnum_hardware::amplifier`), which are siblings of layer 2 |
| Welch's PSD and the averaged periodogram | `comnum_dsp::spectrum` | Spectral estimation, used by tests of the waveforms; `comnum_metrics::spectrum` re-exports them next to the ACLR |
| Cyclic prefix, frequency shift, unitary and centred FFT | `comnum_dsp::signal`, `comnum_math::fft` | Shared by OFDM, CE-OFDM, NR, OTFS and the oscillator impairments |
| Delay–Doppler channel and its random model | `comnum_channel::multipath::{DelayDoppler, DelayDopplerModel}` | A medium; OTFS only takes its paths (`Otfs::effective_channel`) |
| SER of $M$ orthogonal signals detected noncoherently | `comnum_mapping::theory::ser_orthogonal_noncoherent` | Shared by noncoherent FSK and chirp spread spectrum |
| Mutual information over fading (CM and BICM capacities) | `comnum_mapping::theory::{constellation_mi, constellation_gmi}` | One kernel for the closed forms over AWGN and Rayleigh fading; `comnum_mapping::bicm` keeps the cutoff rates |
| Hard decisions of LLRs | `comnum_core::signal::hard_decisions` | The sign convention of LLRs is fixed in `comnum-core`; used by codes, demappers and blocks |
| Indoor optical channel (room, reflections, illuminance) and free-space optics (turbulence, pointing errors, fog) | `comnum_channel::owc` | Media: the waveforms and receivers take a realisation (an impulse response, a gain) |
| PIN-FET receiver noise (shot, thermal) | `comnum_hardware::optical::PinFetReceiver` | A component of the receiver, next to the photodiodes |
| Gamma variates | `comnum_core::rng::Rng::gamma` | Random variates are drawn by the generator every block takes |
| A random QPSK point (pilots, preambles) | `comnum_mapping::source::qpsk` | A constellation point is a source of symbols, not a variate of the generator |
| Decibel conversions (`db2lin`, `lin2db`, `dbm2w`, …) | `comnum_core::units`, re-exported by `comnum_math` | One definition for the newtypes (`Db::to_linear`) and the plain numbers; `comnum-core` cannot depend on `comnum-math` |
| Moments and normalised correlation of complex sequences | `comnum_math::stats::{Moments, Correlation}` | Shared by the cumulants and M2M4 (`comnum_metrics::blind`), the CMA radius (`comnum_dsp::equalizer`), the frame synchronisation and the effective SNR |
| The raised-cosine spectrum | `comnum_dsp::filter::raised_cosine_spectrum` | Shared by `rrc_spectrum` and the WDM spectra of the GN model (`comnum_channel::optical::gn`) |
| Modified Bessel function $K_{\nu}$ | `comnum_math::{bessel_k, ln_bessel_k}` | A special function without communication meaning |
| BPSK-over-AWGN LLRs of the code tests | `comnum_validation::bpsk_awgn_llr` | A test link, not a channel model of the library |
| BPSK symbols $1 - 2b$ and their LLRs $2y/\sigma^{2}$ | `comnum_mapping::{constellation::bpsk, demapper::bpsk_llr}` | A modulation: `comnum-fec` holds none |
| The log-sums of the label halves of a bit (the bitwise demapper) | `comnum_mapping::demapper::LabelHalves` | One kernel for the demapper, the bitwise closed forms and the bit-metric rates of `comnum_metrics` (ADR 0007) |
| Soft output of a code (a posteriori LLRs of the information bits) | `comnum_fec::block::SoftDecode` | Implemented by the LDPC, convolutional (BCJR), turbo and NR LDPC codes and passed through the punctured codes |
| The bits a rate matching sends (puncturing, circular buffers) | `comnum_fec::puncture::BitSelection` | `select` on the encoder side, `accumulate_into` (LLRs summed over the copies) on the decoder side |
| Maximum Doppler shift of a speed | `comnum_channel::multipath::max_doppler_hz` | A property of the medium, used by the CDL channel and the tests |
| Spherical angles to a direction | `comnum_math::geometry::from_spherical` | No communication meaning; shared by the CDL arrays and the indoor optical channel |
| The loop over the spans of an optical link (amplifiers, Raman) | `comnum_channel::optical::fiber::amplified_spans` (crate-private) | One loop for `Ssfm`, `ManakovSsfm` and `pmd::ManakovPmd` |

## Naming conventions

Names follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/naming.html),
plus the domain rules below.

| Element | Rule | Good | Avoid |
| --- | --- | --- | --- |
| Crate | `com-` prefix, kebab-case, singular | `comnum-channel` | `com-channels`, `comchannel` |
| Module | snake\_case, singular noun, no prefix | `fec::ldpc`, `hardware::amplifier` | `fec::ldpc_codes`, `hw::amp` |
| Trait | Verb for the action the block performs | `Encode`, `Map`, `Propagate`, `Impair`, `Detect` | `Encoder` (clashes with concrete types), `ChannelTrait` |
| Concrete type | Noun, no module stutter | `fading::Rayleigh`, `awgn::Awgn`, `stbc::Ostbc` | `fading::RayleighFading` |
| Configuration | A struct with public fields and a `Default` or a constructor, changed by struct update | `FiberParams { n_spans: 10, ..FiberParams::default() }` | positional constructors with 6 `f64` |
| Parser of a name | `FromStr` when the name alone gives the value; `from_name(name, …)` when it needs other arguments or builds a structure | `"eva".parse::<multipath::Profile>()`, `DetectorKind::from_name("kbest", 8)`, `Crc::from_name("crc24a")` | ad-hoc `match` on strings in the bindings or the specifications |
| Unit | In the type (`Db`, `EbN0`, `EsN0`, `Hertz`, `Seconds`) or in the name | `ebn0_db`, `fc_hz`, `alpha_db_km` | bare `f64` named `snr` |
| Function | snake\_case verb or noun of the quantity; `_db` / `_lin` suffix on numeric helpers | `theory_ber`, `qfunc`, `snr_noise_variance` | `BerBpsk`, `calc_ber` |
| Feature flag | Lowercase domain or capability | `optical`, `spec` | `enable-optical`, `OPTICAL` |
| Error | The shared `comnum_core::Error` (ADR 0003) | `Error::Config`, `Error::Shape` | `String` errors, `Box<dyn Error>` in public API |

Some types kept the names of the first API and stutter (`ldpc::LdpcCode`,
`fiber::FiberParams`, `pmd::PmdLink`, `cpm::CpmConfig`); new types do not.

Use the notation of the reference literature in parameter names when it is unambiguous
and documented: `beta2`, `gamma`, `n_fft`, `n_cp`, `rolloff`. Never use single letters
in public signatures except for mathematical objects named that way in the docs (`h`
for a channel matrix is acceptable inside a function body, not as a field name). The
Python API shares one vocabulary with the block parameters of the specifications
(`fs`, `fc_hz`, `doppler_hz`, `noise_var`, `esn0_db`, `n_rx`, `n_tx`).

### Units in names

A public `f64` that has a unit carries it, in its type (`Hertz`, `Db`, `EbN0`) or in a
suffix. This covers struct fields, methods that return a quantity, block parameters of
the specifications, Python parameters and Rust parameters.

| Suffix | Unit |
| --- | --- |
| `_hz` | Hz |
| `_s` | s |
| `_m`, `_km` | m, km |
| `_mps` | m/s |
| `_rad` | rad |
| `_db`, `_dbm` | dB, dBm |
| `_w` | W |
| `_a` | A |
| `_k` | K |
| `_lin` | a linear ratio that could be mistaken for dB, such as a linear SNR |

Names that stay bare: the vocabulary documented below (`fs` and `sample_rate` in Hz,
the rate of a signal or of a channel as `Signal::sample_rate` gives it; `rolloff`,
`beta2`, `gamma`, `noise_var`, linear by its name), counts and dimensionless
quantities. A
quantity in another unit says so too (`spacing_wavelengths`, `coherence_samples`).
A function of the sampling grid names its argument by what it is: `fftfreq(n, spacing_s)`
takes the sample spacing $1/f_s$ (s), `angular_frequencies(n, fs)` the rate (Hz).

### One word per notion

| Notion | Word | Not |
| --- | --- | --- |
| samples per symbol of a pulse | `sps` | `samples_per_symbol` |
| FFT oversampling factor (OFDM, CE-OFDM) | `oversampling` | |
| roll-off | `rolloff` | `alpha` |
| noise variance | `noise_var` | `sigma2` |
| fixed iteration count | `iterations` | |
| iteration cap with early stopping | `max_iterations` | |
| subcarriers | `n_subcarriers` | `n_data` |
| data resource elements | `n_data` | |
| data symbols per antenna of an OFDM frame | `n_data_symbols` | `n_data`, and `n_symbols`, which counts OFDM symbols |

`iterations` and `max_iterations` are chosen by meaning: a decoder that stops when its
syndrome is zero has a cap (`LdpcCode`, the NR LDPC code). CPM keeps `sps`, its shaping
rate.

### Code dimensions

Arguments keep the order of the reference: (n, k) for an (n, k) block code; (K, E) for
the 3GPP rate-matched codes, named `k` and `e`, never `n`. Swapping two integers would
break callers silently, so an order is never changed to make two families agree;
instead every name says what it is.

### One choice, one enum

A choice between two algorithms is an enum, never a boolean whose polarity differs
from one function to the next: `Jacobian { Exact, MaxLog }` (of `comnum_core::traits`,
re-exported as `comnum_math::Jacobian`) is the choice of the Jacobian logarithm in
`max_star`, the BCJR, the turbo decoder, `Constellation::llr` and the `Demap` trait. A specification may keep a boolean key (`max_log`) mapped to the
enum.

### Constructors

- At most three positional arguments.
- Beyond that, a parameter struct with public fields, a `Default` and a
  `check -> Result`; building is `new(params) -> Result`, or struct update followed by
  `check` for a block whose fields are its parameters (`Rapp`, `Saleh`, `Mzm`, `Ufmc`):
  `TdlParams`, `CdlParams` (with `ClusterTable` for `Cdl::custom`), `PortEstimation`,
  `FrameDetection`, `message_passing::MessagePassing`.
- A boolean argument becomes an enum or a field.

### Container accessors

- `stream(i)` returns a slice on every container, `stream_view(i)` the `ndarray` view.
- The length of a stream is `len()`, and the resize is `resize(..)`, everywhere.

### Verbs

The names of the verb traits (`modulate`, `demodulate`, `map`, `demap`, `encode`,
`decode`, `detect`, `propagate`, `impair`) are reserved for their meaning. Bits to
symbols outside the trait is `map_bits`, symbols to bits by hard decision
`demap_bits`; a second meaning of a name gets another name
(`CeOfdm::demodulate_with_receiver`, `Ccdm::match_bits`). A closed form says its SNR
convention in its name when it is not Eb/N0 (`theory::ser_esn0`).

### Python

- Signals and multi-antenna arrays are stream-major, `(n_streams, n)`, everywhere; a
  channel matrix is `(n_rx, n_tx)`, a stack of them `(n, n_rx, n_tx)`, one per channel
  use. The received streams of a MIMO detector are `(n_rx, n)` and its decisions
  `(n_tx, n)`; its LLRs are the LLR sequence of every stream, `(n_tx, n · log2 M)`. A
  dual-polarisation field is a `(2, n)` array.
- A time-frequency grid is a `(n_subcarriers, n_symbols)` array, a column per OFDM
  symbol, and its resource elements are `(k, l)` pairs, an `(n, 2)` array, so that
  `grid[idx[:, 0], idx[:, 1]]` reads them. The functions that read a grid also take
  the symbols of `Ofdm.demodulate` (symbol after symbol, the order of the crates).
- The random generator is the last positional parameter, named `rng`; an optional
  generator is a keyword (`propagate(x, fs, *, rng=None)`).
- Tuning parameters are keyword-only.
- A function returns one type, whatever its arguments: a second result is a second
  function (`Ofdm.modulate` and `modulate_with`, `papr_db` and `papr_db_blocks`,
  `mimo_channel` and `mimo_channels`); an array keeps its dimensions (a SISO frequency
  response is `(n_bins, 1, 1)`). The closed forms that follow the shape of their SNR
  argument, a float for a number and an array for an array, are the exception, as
  NumPy's functions are.
- A closed form is named `theory_*` and accepts arrays for its SNR argument.
- Each item has the name of its Rust item, unless the Python idiom requires otherwise
  (`__repr__`, `n_streams` properties). The Python names flatten the modules: a
  function whose Rust name is only clear inside its module takes the module in its
  name (`gn_closed_form`, `fog_attenuation_db_per_km`, `theory_ergodic_capacity` for
  `capacity::ergodic_rayleigh`), and the parameters of a class that has a block in the
  specifications are those of the block (`Ofdm(n_fft, n_subcarriers)` as `ofdm_tx`,
  where the Rust field is `n_active`).
- The choice of the Jacobian logarithm is the boolean keyword `max_log`, as the
  specifications write it (`Constellation.llr`, `Dvbs2.llr`, `TurboCode`); the prior
  probabilities of the points are `px`, everywhere.

## Core types and traits

`comnum-core` defines the vocabulary every other crate speaks: typed containers, verb
traits, unit newtypes, a single generator type and the error type. It contains no
algorithm.

### Containers

All containers are two-dimensional, streams × elements. SISO is `n_streams == 1`.

```rust
use ndarray::Array2;
use num_complex::Complex64;

/// Hard bits, one row per stream. Values are 0 or 1.
pub struct Bits(Array2<u8>);

/// Log-likelihood ratios, one row per stream (sign convention below).
pub struct Llr(Array2<f64>);

/// Complex symbols at one sample per symbol.
pub struct Symbols(Array2<Complex64>);

/// The indices of symbols (a chirp of CSS, a level of CPM), one row per stream.
pub struct Labels(Array2<usize>);

/// Complex baseband waveform with its sample rate.
pub struct Signal {
    data: Array2<Complex64>, // [n_streams, n_samples]
    sample_rate: Hertz,
}
```

`DualPol` holds the two polarisations of an optical field.

The LLR sign convention is fixed library-wide. A positive LLR favours bit 0:

```latex
L(b) = \ln \frac{P(b = 0 \mid y)}{P(b = 1 \mid y)}
```

### Traits

One verb trait per role: `Encode`, `Decode`, `Map`, `Demap`, `Modulate`, `Demodulate`,
`Precode`, `Detect`, `Propagate`, `Impair`. Methods taking `&self` are pure; `&mut self`
signals internal state (memory, adaptation). Every trait has an `_into` method writing
into a caller-owned buffer, which it resizes and fills in place (never replaces), so a
loop that passes the same buffer allocates no output once its shape is stable; scratch
buffers may allocate unless a profile shows them (ADR 0011). All but `Precode` and
`Detect` also have an allocating method that calls it.

```rust
pub trait Encode {
    /// Code rate k/n.
    fn rate(&self) -> f64;
    fn encode_into(&self, info: &Bits, out: &mut Bits) -> Result<()>;
    fn encode(&self, info: &Bits) -> Result<Bits> { /* allocates, calls encode_into */ }
}

pub trait Decode {
    fn decode_into(&mut self, llr: &Llr, out: &mut Bits) -> Result<()>;
    fn decode(&mut self, llr: &Llr) -> Result<Bits> { /* … */ }
}

pub trait Map    { fn map_into(&self, bits: &Bits, out: &mut Symbols) -> Result<()>; /* map */ }

/// The log-priors of the points (uniform if `None`) and the Jacobian logarithm.
pub struct Demapping<'a> { pub log_priors: Option<&'a [f64]>, pub jacobian: Jacobian }

pub trait Demap {
    fn demap_with_into(&self, y: &Symbols, noise_var: f64, demapping: Demapping<'_>, out: &mut Llr) -> Result<()>;
    fn demap_into(&self, y: &Symbols, noise_var: f64, out: &mut Llr) -> Result<()> { /* default Demapping */ }
    /* demap_with, demap */
}

/// Complex symbols by default; `Labels` for the waveforms that carry an index (ADR 0012).
pub trait Modulate<In = Symbols>   { fn modulate_into(&self, x: &In, out: &mut Signal) -> Result<()>; /* modulate */ }
pub trait Demodulate<Out = Symbols> { fn demodulate_into(&mut self, y: &Signal, out: &mut Out) -> Result<()>; /* demodulate */ }

pub trait Detect { fn detect_into(&mut self, y: &Symbols, noise_var: f64, out: &mut Symbols) -> Result<()>; }

/// Propagation through a medium. Randomness is always explicit.
pub trait Propagate {
    fn propagate_into(&mut self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<()>;
}

/// A component non-ideality (amplifier, oscillator, IQ imbalance…).
pub trait Impair {
    fn impair_into(&mut self, x: &Signal, rng: &mut SimRng, out: &mut Signal) -> Result<()>;
}
```

Implementations: every `BlockCode` of `comnum-fec` (`Encode`, `Decode`, by the macro
`container_traits!`): `ConvolutionalCode`, `TurboCode`, `PuncturedTurbo`,
`Punctured<C>`, `LdpcCode`, `PolarCode`, `ReedSolomon`, `Bch`, `NrLdpc`, `NrPolar`
(`Dvbs2Fec` of `comnum-systems` is a `BlockCode` only); `Constellation` (`Map`, `Demap`,
with priors and the Jacobian through `Demapping`); `Ofdm`, `Ufmc`, `Otfs`, `CeOfdm`,
`Fbmc` (`Modulate`, `Demodulate`) and `Css`, `Cpm` (`Modulate<Labels>`,
`Demodulate<Labels>`), at the sample rate of the modem, checked on the input; `Eigenmode`, `LinearPrecoder`,
`Ostbc` (`Precode`); `detection::Coherent` (`Detect`); `Awgn`, `Noise`, `Rayleigh`,
`Mimo` (`fading`), `Tdl`, `DelayDopplerModel` (`multipath`), `Cdl`, `Fso`
(`owc::atmospheric`), `Ssfm` (`optical::fiber`) (`Propagate`); `Rapp`, `Saleh`,
`MemoryPolynomial`, `PhaseNoise`, `Cfo`, `TxIq`, `RxIq`, `Adc`, `Dac` (`Impair`). The
random linear channels of `comnum-channel` also share `fading::Fading` (`Mimo`, `Tdl`,
`DelayDopplerModel`, `Cdl`): a realisation drawn as `multipath::Taps` for a number of
samples, applied with the checks of streams and sample rate; their `Propagate` is its
`fade_into`. The other blocks expose slice-based functions and methods.

### Randomness

There is no global or thread-local generator anywhere in the library. Every random
block receives `&mut SimRng`, a concrete type alias, which keeps traits object-safe and
calls monomorphic. `SimRng` is xoshiro256\*\* seeded through SplitMix64 (ADR 0002).

```rust
pub type SimRng = comnum_core::rng::Rng; // xoshiro256**

/// The stream of one trial: independent of every other (seed, experiment, condition, trial).
pub fn stream(base_seed: u64, experiment: &str, condition: u64, trial: u64) -> Rng { /* … */ }

/// Independent, reproducible stream for one batch of frames: stream(master_seed, "", 0, batch).
pub fn batch_rng(master_seed: u64, batch: u64) -> SimRng { /* … */ }
```

The runner of `comnum-sim` gives trial `t` of condition `c` of the scenario `name` the
stream `stream(seed, name, c, t)`, so results do not depend on the thread count, nor on
the batch size when the stopping rule bounds the trials (the last batch ends at
`max_trials`); an error or precision target is checked after every batch.

### Units

Quantities that are routinely confused get their own type. Conversions are explicit
methods, never `From` impls that hide a formula.

```rust
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Db(pub f64);

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct EbN0(pub Db);

impl EbN0 {
    /// Converts to Es/N0 for `bits_per_symbol` coded bits at code rate `rate` (an `f64`:
    /// the entropy of a shaped constellation is not an integer).
    pub fn to_es_n0(self, bits_per_symbol: f64, rate: f64) -> EsN0 { /* … */ }
}
```

The decibel conversions of plain numbers (`db2lin`, `lin2db`, `db2amp`, `amp2db`,
`dbm2w`, `w2dbm`, `db2np`, `np2db`) are defined in `comnum_core::units` and re-exported by
`comnum_math` (`comnum_math::db2lin` is that function); the methods of the newtypes call
them, so that a value in dB converts to the same bits whichever way it is written.
The kernels keep taking `f64` named with their unit; the newtypes are for the blocks.

The conversion it implements, with k bits per symbol and code rate R:

```latex
\left.\frac{E_s}{N_0}\right|_{\mathrm{dB}} = \left.\frac{E_b}{N_0}\right|_{\mathrm{dB}} + 10 \log_{10}(k R)
```

### Configuration

A block is configured by a struct with public fields, built by a constructor or
`Default` and adjusted by struct-update syntax; its fields are checked when the block
is built or used, and an invalid value returns `Error::Config`:

```rust
let fiber = FiberParams { span_km: 100.0, n_spans: 10, ..FiberParams::default() };
let ofdm = Ofdm { n_cp: 16, dft_spread: true, ..Ofdm::new(256, 200)? };
```

A modem that samples a filter once at construction (`Fbmc`, its prototype) keeps its
fields private behind getters, so that the cached taps cannot disagree with them; so do
the codes whose tables derive from their parameters (`Trellis`, `Interleaver`,
`TurboCode`, `Punctured`, `NrLdpc`, `NrPolar`), whose decoder settings are a struct of
public fields (`LdpcDecoder`, `TurboDecoder`, reached by `decoder_mut()` on a wrapping
code).
`CpmConfig::build` is the one configuration validated by a separate `build` step,
since it tabulates the phase response. The enums that will gain variants are
`#[non_exhaustive]` (`comnum_core::Error`, `equalizer::Algorithm`, `filter::{PulseShape,
LowPass}`, `ConstellationKind`, `CheckRule`, `DetectorKind`, `multipath::Profile`,
`cdl::{Profile, Pattern}`, `fiber::{Amplification, Precision}`, `PhaseOrigin`,
`StepRule`, `PilotPattern`, `FrequencyInterpolation`, `ce_ofdm::{Unwrap, Receiver}`,
`papr::{Partition, Companding}`, `cpm::FrequencyPulse`, `fbmc::Prototype`,
`css::Waveform`, `message_passing::Prior`, `atmospheric::Turbulence`, the DVB-S2
`FrameSize`, `CodeRate` and `Modulation`); the configuration structs are not.

## Documentation standards

Every public item is documented, every formula is written in LaTeX, and every example
compiles and runs as a test. `missing_docs` and `cargo doc -D warnings` enforce the
first; the doctests the last.

### Tooling

- **API reference:** rustdoc, with KaTeX injected through `--html-in-header`
  (`.cargo/config.toml` for local builds). docs.rs builds each crate in isolation, so
  each crate ships its own copy of `docs/katex-header.html` (`just katex` refreshes
  them, CI checks they are identical).
- **Index of the public API:** [`API.md`](API.md), every public item of the crates,
  of the bindings and of the Python package with its signature and the first line of
  its documentation, generated by `tools/api_index.py` (`just api-index`).
- **Python API:** Google-style docstrings written on the PyO3 bindings and in
  `python/comnum`, rendered with Sphinx (PyData theme, `sphinx.ext.napoleon`,
  autosummary) in `docs/python`, with tutorials and a gallery of classical results run
  when the documentation is built (`sphinx-gallery`). Python docstrings are written for
  Python users, not copied from rustdoc. The stub `python/comnum/_comnum.pyi` is
  generated from them (`tools/gen_stubs.py`).

```toml
# .cargo/config.toml (local builds)
[build]
rustdocflags = ["--html-in-header", "docs/katex-header.html"]

# crates/<crate>/Cargo.toml (docs.rs builds)
[package.metadata.docs.rs]
rustdoc-args = ["--html-in-header", "katex-header.html"]
all-features = true
```

### Item template

Sections appear in this order and are omitted only when empty: summary line,
description, `# Model`, `# Errors`, `# Panics`, `# Examples`, `# References`.

````rust
/// Additive white Gaussian noise channel.
///
/// Adds circularly-symmetric complex Gaussian noise to every stream independently.
/// The noise power is set from `E_s/N_0` and the measured average power of each
/// stream (at one sample per symbol), so the input need not be normalised.
///
/// # Model
///
/// $$
/// y_{k} = x_{k} + n_{k}, \qquad n_{k} \sim \mathcal{CN}(0, N_{0})
/// $$
///
/// with $N_{0} = \bar{E}_{s} / (E_{s}/N_{0})$, where $\bar{E}_{s}$ is the average
/// symbol energy of the stream at one sample per symbol.
///
/// # Errors
///
/// [`Propagate::propagate_into`] returns [`Error::Shape`] if `x` has no samples.
///
/// # Examples
///
/// ```
/// use comnum_core::prelude::*;
/// use comnum_channel::awgn::Awgn;
///
/// let mut rng = batch_rng(42, 0);
/// let mut ch = Awgn::new(EsN0(Db(10.0)));
/// let x = Signal::from_stream(vec![C64::new(1.0, 0.0); 1024], Hertz(1.0))?;
/// let y = ch.propagate(&x, &mut rng)?;
/// assert_eq!(y.len(), 1024);
/// # Ok::<(), comnum_core::Error>(())
/// ```
///
/// # References
///
/// 1. J. G. Proakis and M. Salehi, *Digital Communications*, 5th ed.,
///    McGraw-Hill, 2008.
pub struct Awgn { /* … */ }
````

A comparison with a publication or a standard cites its authors, venue, year and the
figure, table or equation it reproduces; a standard is cited with its number, version
and section. The documents themselves are not in the repository: a validation card
names its file under `references/<theme>/`, a local folder ignored by git.

### Module documentation

Each module's `//!` header gives an overview, the signal model with its notation, a
table of the types it exports, and references. The crate root has a table of its
public modules (`| [`module`] | content |`), which lists all of them.

### LaTeX in rustdoc

Markdown is parsed before KaTeX runs, so a few rules prevent silent corruption:

- Put display math in `$$` on its own lines; inline math in single `$`.
- Brace every subscript (`E_{b}`, `x_{k}`) so pairs of underscores are never read as
  emphasis.
- Double the backslash before punctuation, which Markdown would otherwise consume:
  write `\\\\` in the source for a LaTeX line break `\\`, and `\\{` for a literal brace
  `\{`. Backslashes before letters (`\frac`, `\mathcal`) need no doubling.
- Check every new formula in the rendered page (`just doc`) before merging.

### Prose

British spelling (identifiers keep their spelling: `normalized`, `Equalizer`).

## Code quality standards

Lints are declared once at the workspace root and inherited by every crate (`[lints]
workspace = true`); the exceptions are listed and justified in ADR 0005. CI runs clippy
with `-D warnings`, so a warning blocks a merge.

```toml
# Cargo.toml (workspace root)
[workspace.package]
version = "0.2.0"              # one version for every crate
edition = "2024"
rust-version = "1.85"          # MSRV, tested in CI
license = "MIT OR Apache-2.0"

[workspace.lints.rust]
missing_docs = "warn"
unsafe_code = "forbid"
unreachable_pub = "warn"

[workspace.lints.clippy]
pedantic = { level = "warn", priority = -1 }
cast_precision_loss = "allow"  # usize -> f64 is pervasive in numeric code
unwrap_used = "deny"
expect_used = "warn"
allow_attributes_without_reason = "warn"
# … the numeric exceptions of ADR 0005 (suboptimal_flops, manual_midpoint, casts, …)
```

### File size

A source file holds one notion. Past about 1 200 lines (`tools/test_layout.py` warns,
without failing), it becomes a folder whose `mod.rs` keeps the module documentation,
the shared definitions and the tests, and whose submodules hold one part each
(`comnum_channel::multipath::{taps, tdl, delay_doppler, echoes}`,
`comnum_math::nlse::manakov`, `comnum_fec::ccsds::{ldpc, turbo}`). The submodules are
private and re-exported (`pub use tdl::{Tdl, TdlParams};`), so that the public paths do
not change; they import by name (`use super::{Taps, LOS_DOPPLER_RATIO};`, the prelude of
`comnum_core` being the only glob), and what the tests of `mod.rs` need is `pub(super)`.

### Errors and panics

- One error type, `comnum_core::Error` (`thiserror`, `#[non_exhaustive]`: `Config`,
  `Shape`, `Numerical`, `Condition`, `Io`), shared by every crate and re-exported as
  `comnum::Error` (ADR 0003). No `String` errors, no `Box<dyn Error>` in public
  signatures.
- `Error::Shape` is for lengths: two inputs that disagree, or an input empty where
  samples are needed; it gives the lengths received (`comnum_core::ensure_shape`,
  `same_len`). `Error::Config` is for values: outside their domain, not finite, or an
  unknown name. Every function follows this rule, and its `# Errors` documentation
  says the same.
- Invalid user input returns an error; it never panics. Panics are reserved for broken
  internal invariants and for the length and index preconditions of kernels and
  accessors called in hot loops by blocks that check their inputs once (ADR 0016); both
  are documented under `# Panics`. Nothing the Python bindings or the specifications
  call panics on the user's values (`tests/test_no_panic.py`).
- A configuration is checked when the block is built or used, once; hot loops never
  re-validate. The Python bindings map `Config`, `Shape` and `Condition` to `ValueError`,
  `Numerical` to `ArithmeticError`, `Io` to `OSError`.

### Numerics

- `f64` throughout. A lower-precision path is added only with a measured need and a
  check proving it holds (`Ssfm` with `Precision::F32`, compared with `f64` in its
  tests).
- Probabilities and LLR combinations are computed in the log domain (`log_sum_exp`,
  `max_star`, `ln_1p_exp`) to avoid underflow at high SNR.
- Results do not depend on the number of threads. Random streams are derived from the
  trial index, never from the worker id, so a run on 1 core and on 16 cores is
  bit-identical (`crates/comnum/tests/simulation.rs`).
- Kernels that produce published results keep their order of operations: no
  `mul_add`, no `f64::midpoint` (ADR 0005).
- Floating-point comparisons in tests use explicit tolerances written in the assertion,
  with the reason (sampling error, precision of a published value) when it is not
  obvious.

### Performance

- No allocation inside the Monte Carlo loop where it matters: the verb traits have
  `_into` methods that reuse their output, callers reuse buffers, and blocks whose
  scratch shows in a profile keep a workspace (ADR 0011).
- Parallelism lives in `comnum-sim` only (rayon over the trials of a batch). Blocks are
  single-threaded, which avoids nested parallelism.
- FFT plans and code matrices are built once at construction and cached in the block.
- Profile before optimising (`samply`, `cargo flamegraph`, or `profile: true` in a
  specification for the time of every block). An optimisation states its gain,
  measured with the criterion benchmarks (`benchmarks/rust/blocks.rs`) or
  `benchmarks/opticommpy/bench_opticommpy.py`.

### API stability

- Semantic versioning; one version for every crate; the changelog lists the changes of
  every release, with a migration note for a breaking change. From 0.2.0, the first
  public release, CI runs `cargo-semver-checks` on every pull request against the base
  branch (job `semver` of `ci.yml`): a change that breaks the public API of a crate
  fails unless the major version (the minor, below 1.0) is bumped.
- The Python API has the same contract: a renamed or removed name, a changed default
  or return value is a breaking change with a migration note; a new keyword argument
  with a default is not. The names follow `docs/ARCHITECTURE.md` (Naming
  conventions) and the plan of 30 October 2026 (`docs/plans/`); until 0.2.0 they move
  without deprecated aliases.
- The specification files carry no version field yet: the schema (`comnum schema`,
  `spec_schema`) is versioned with the crates, and a specification of an older version
  that no longer builds is a breaking change of `comnum-spec`.
- The enums that will grow are `#[non_exhaustive]` (see Configuration).
- `unsafe` is forbidden workspace-wide. The Python bindings do not inherit the
  workspace lints (PyO3's macros expand to `unsafe` code, ADR 0005); a future exception
  in the library requires its own module, a `// SAFETY:` comment on every block, and
  Miri in CI.

### Automation

`just check` runs the checks of CI, in this order (run it before declaring a task
done). It is three recipes, which the jobs of `.github/workflows/ci.yml` run as they
are, so that a check cannot pass locally and fail in CI or the reverse:

- `just check-rust`: `cargo fmt --all --check`; `cargo clippy --workspace --all-targets
  -- -D warnings`; `cargo doc --workspace --no-deps --exclude comnum-py` with
  `RUSTDOCFLAGS="--html-in-header $PWD/docs/katex-header.html -D warnings"`; every crate
  ships the same `katex-header.html` as `docs/katex-header.html` (`cmp`);
- `just check-tests`: `cargo test` (the default members: every crate but the Python
  extension, doctests and the quick tests of `comnum-validation` included), then `just
  verify-ci`, `comnum verify` of the validation specifications of `ci_specs` of the
  `justfile`, the only list (`proakis_awgn`, `information_rates`, `hager2021_fig5`,
  `msk_bpsk`, `thompson2008_eq15`);
- `just check-python`: `ruff check` and `ruff format --check` of `python tests
  docs/python tools benchmarks/libraries`; `mypy` (`[tool.mypy]` of `pyproject.toml`: the
  types of the package and of the generated stub, the tests checked as callers of the
  stub only, so a method stubbed `-> None` whose value is used fails); then every
  generated file up to date and every registry consistent: `api_index.py --check`
  (`docs/API.md`) and `--duplicates --check` (no public name defined in several modules
  that `docs/api-duplicates.txt` does not accept, or in more modules than it accepts:
  factor it, or add it with its count and the reason it is not a copy),
  `literature_table.py --check` and `maturity_table.py --check` (and every test they
  cite exists), `cards.py --check` (and a test named after a reference loads its card,
  `validation/cards/uncarded.txt` apart), `test_layout.py --check` ("Where a
  test goes"), `benchmarks_page.py --check`, `highlights.py --check`, `gen_stubs.py
  --check` and `gen_blocks.py --check` (they need the extension built).

`just regen` rewrites every generated file in the order they depend on each other (the
extension rebuilt, the stub and `comnum.blocks`, `docs/API.md`, the literature and
maturity tables, the benchmark page, the highlights): the answer to any "out of date"
of `just check-python`.

The Python environment is `.venv`, made by `just setup-py` (`uv sync --locked`): the
package built by maturin and the dependency groups `test` and `lint` at the versions of
`uv.lock`, the same in CI; `uv sync --group docs` adds the documentation, `--group
oracle` the tools of the recorded oracles. Moving the versions (`uv lock --upgrade`) is
a deliberate act, in a commit of its own.

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests, with read-only
permissions; it runs the quick tests only:

| Job | Runs on | Steps |
| --- | --- | --- |
| `rust` | Linux | `just check-rust` |
| `test` | Linux, macOS, Windows | `just check-tests` |
| `msrv` | Linux, Rust 1.85 | `cargo build` |
| `deny` | Linux | `cargo deny check` (licences, advisories, `deny.toml`) |
| `python` | Linux, macOS, Windows × Python 3.11, 3.12, 3.13 | `uv sync --locked`; on Linux with 3.12 only: `just check-python`; then `python -m pytest` |
| `docs-python` | Linux | `uv sync --locked --group docs`; `sphinx-build -W --keep-going` of `docs/python`; the site is uploaded on pushes to `main` |
| `deploy-docs` | Linux, pushes to `main` | publishes the site to GitHub Pages |

`.github/workflows/validation.yml` runs the long validations and the benchmarks, and
only when started by hand (`workflow_dispatch`): `cargo test --release -p
comnum-validation -- --ignored`, with an optional filter on the test names, and, when its
switch is set, `cargo bench -p comnum`. The ignored tests never run automatically, and
the benchmarks are run, not compared: no regression threshold is enforced.

`.github/workflows/release.yml` runs on a tag `vX.Y.Z` and on demand:

1. `version`: the tag equals `workspace.package.version`, `CITATION.cff` gives the
   same version, and the section of `CHANGELOG.md` whose heading contains `[X.Y.Z]`
   becomes the release notes;
2. `crates` (tag only): `cargo publish --workspace`, in the order of the dependencies;
3. `wheels`: one abi3 wheel (CPython 3.11 and later) per platform: manylinux and
   musllinux on x86_64 and aarch64, macOS on x86_64 and aarch64, Windows x64; `sdist`;
4. `install`: the wheels install from the built files alone on Linux, macOS and
   Windows (Python 3.11 and 3.13) and run a link;
5. `testpypi` (on demand only, the dry run to do before the first tag): the wheels and
   the sdist go to TestPyPI, and nothing else is published;
6. `pypi` (tag only, trusted publishing) and `github-release` (the notes and the
   wheels).

Generated files are committed and checked:

| File | Generator | Check |
| --- | --- | --- |
| `docs/API.md` | `tools/api_index.py` (`just api-index`; `--duplicates` lists the names defined in several modules) | `--check` |
| `python/comnum/_comnum.pyi` | `tools/gen_stubs.py`, from the built extension | `--check` |
| `python/comnum/blocks.py` | `tools/gen_blocks.py`, from the block registry (`just gen-python` runs both) | `--check` |
| `docs/LITERATURE.md` | `tools/literature_table.py`, from `validation/literature.yaml` (`just literature`) | `--check` |
| `docs/MATURITY.md`, the maturity tables of `README.md` and `docs/python/maturity.rst` | `tools/maturity_table.py`, from `validation/maturity.yaml` (`just maturity`) | `--check` |
| `docs/python/benchmarks.rst` | `tools/benchmarks_page.py`, from `benchmarks/libraries/results/` (`just bench-libraries`) | `--check` |
| the highlights of `README.md` and `docs/python/index.rst`, `docs/python/why.rst` | `tools/highlights.py`, from the manifests, `benchmarks/libraries/results/` (with `footprint.json`) and the validation files | `--check` |
| `crates/*/katex-header.html` | `just katex` | `cmp` |
| `tests/oracle/*` | `tests/oracle/__init__.py` with `COMNUM_RECORD_ORACLE=1` (dependency group `oracle`) | the tests read them |
| `validation/aff3ct/*.json` | `tools/record_aff3ct.py`, with AFF3CT installed | the tests read them |
| `benchmarks/opticommpy/results.md` | `benchmarks/opticommpy/bench_opticommpy.py` (`just bench`) | none |

## Testing and validation

A block is merged when it is correct by construction (unit tests), correct by physics
(invariants) and correct against an independent reference. The evidence for every
module is summarised in [`MATURITY.md`](MATURITY.md).

| Level | Where | Checks | Runs |
| --- | --- | --- | --- |
| Unit | `#[cfg(test)] mod tests` of every module | Edge cases, error paths, round trips and identities over seeded random inputs, known small vectors | `cargo test`, every push |
| Literature | `#[cfg(test)] mod literature` of the module, one test named after each reference | The tables, test vectors and closed forms of a document of `references/` for one item, deterministic and under a second, verbatim | `cargo test`, every push |
| Doctest | rustdoc | Every documented example compiles and runs | `cargo test`, every push |
| Integration | `crates/comnum/tests/` (`links`, `simulation`, `specs`), `crates/comnum-spec/tests/extension.rs` | End-to-end links against closed forms, the runner (thread independence, stopping rules), every specification runs one trial per condition | `cargo test`, every push |
| Specifications | `specs/validation/*.yaml` and `specs/*.yaml` with an `expect` section | Simulated results against closed forms, tables of standards and published curves | `comnum verify`: five in CI; the others with `just validate`, `validation/tests/simulation/specs.rs` or `tests/test_performance.py` |
| Statistical | `validation/tests/<theme>/<key>.rs` (crate `comnum-validation`, one test binary per theme) | Monte Carlo curves against closed forms, published figures and tables, standards and the industry, and recorded AFF3CT runs | quick tests with `cargo test`, every push; `#[ignore]` ones on demand (`just card <id>`, `cargo test --release -p comnum-validation -- --ignored`, or `validation.yml` started by hand) |
| Oracle | `tests/oracle/test_*.py`, the recordings beside them | The blocks against the recorded outputs of OptiCommPy, GNPy, py3gpp, Sionna and GNU Radio on the same inputs | `python -m pytest`, every push |
| Classical results | `tests/test_performance.py` and the gallery of `docs/python` | Textbook results end to end from Python | `python -m pytest`; the gallery when the documentation is built |
| Benchmark | `benchmarks/rust/blocks.rs` (criterion), `benchmarks/opticommpy/bench_opticommpy.py`, `benchmarks/libraries/` | Throughput; speed and agreement against OptiCommPy; results and speed against komm, OptiCommPy and Sionna | on demand (`validation.yml` with its benchmarks switch, `just bench-rust`, `just bench`, `just bench-libraries`); not compared with a threshold |

There are no property-based tests (`proptest`): the round trips and identities are
unit tests on inputs drawn from seeded generators.

### Properties every block family must satisfy

- **FEC:** noiseless encode → decode returns the input; the codeword satisfies the
  parity checks (H c = 0).
- **Mapping:** hard demapping of noiseless symbols returns the bits; constellations
  have unit average energy once normalised.
- **Waveforms:** demodulate(modulate(x)) = x for an ideal channel; OFDM keeps unit
  power.
- **Channels:** a noiseless, distortionless configuration is the identity; lossless
  SSFM conserves energy; a fundamental soliton propagates without changing shape.
- **Hardware:** zero-impairment configurations are the identity.

### Classes of evidence

A result is checked against one of four kinds of reference, from the strongest:

1. **The recorded output of a reference implementation** on the same inputs or the
   same configuration: OptiCommPy and GNPy (`tests/oracle/`), AFF3CT
   (`validation/aff3ct/`). The fixtures record the tool, its version and the date.
2. **A table of a standard, reproduced verbatim**, or the result it specifies: the CDL
   and TDL profiles, the CRC polynomials, the puncturing patterns, Table 13 of
   EN 302 307-1.
3. **A published figure or table**: the points read on the figure, or the values of
   the table, with the tolerance the reading allows.
4. **A closed form**: the error rate, capacity or distribution that theory gives for
   the configuration.

A validation card names its class in `evidence` (`validation/cards/README.md`):
`recording` (1), `table` (2, or 3 for a table of a paper, and the numbers printed in
the text), `curve` (3, the points read on a figure), `closed_form` (4), and `claim`, an
approximate statement of the text checked as a range, weaker than a curve; where the
values come from is `source.kind` (`paper`, `standard`, `tool`). `cards.py --check`
holds a recording to a tool and every tolerance but an exact one to its `why`;
`LITERATURE.md` shows the evidence of the cards of each document.

`MATURITY.md` grades every module by the strongest evidence it has: **A** for 1–3
(and a claim of a document), **B** for 4 only, **C** for self-consistency only (round
trips, invariants). The grade is computed by `tools/maturity_table.py` from the kind
that every item of evidence of `validation/maturity.yaml` declares (the card kinds, and
`self`); an item that cites a card has one of the kinds of that card. It is
generated, with the maturity tables of the README and of the Python documentation,
from `validation/maturity.yaml` (`just maturity`); `LITERATURE.md` is generated from
`validation/literature.yaml` (`just literature`). Both group their rows by the themes
of the documentation (foundations, coding and shaping, waveforms, MIMO, wireless
channels, optical, hardware impairments, receiver DSP, metrics and analysis, systems
and standards, simulation and specifications), as the Python API reference and the
gallery do.

### Specifications with expectations

An `expect` entry of a specification gives the expected result as a value, a curve
read from a publication (`curve: {along, x, y}`) or a closed form (`theory:`), with
its `source` and a tolerance (`atol`, `rtol`, or `interval`: the theory must lie in
the confidence interval of the estimate); `reference` compares a metric with another
metric of the same condition within a `factor`. `comnum verify` runs the campaign,
prints every check and fails if one fails; a specification with a `search` section
checks its operating points.

### The oracle fixtures

`tests/oracle/test_oracle.py` compares every deterministic block shared with OptiCommPy, and
`tests/oracle/test_gnpy_oracle.py` the Raman amplification and the generalised GN model with
GNPy. The outputs of the tools are recorded once into `tests/oracle/` (`<name>.json`
with the tool, its version, the versions of Python and NumPy and the date; the arrays of
more than 64 values in `<name>.npz`) by `tests/oracle/__init__.py`; the tests read them and skip
when a file is missing, so neither tool is needed. Re-recording
(`pip install --group oracle`, `COMNUM_RECORD_ORACLE=1 python -m pytest
tests/oracle/test_oracle.py tests/oracle/test_gnpy_oracle.py`) is a deliberate act, named in the
commit message with the version of the tool; re-recording an unchanged oracle
produces no diff.

### The AFF3CT fixtures

`tools/record_aff3ct.py` runs AFF3CT (Cassagne et al., *SoftwareX* 10, 2019) on 17
configurations over BPSK and AWGN (Reed–Solomon, BCH, the (133, 171) convolutional
code, LDPC codes with flooding and layered decoders and the DVB-S2 short code, polar
SC and CA-SCL, LTE turbo) and records, for every $E_b/N_0$, its frames, bit and frame
errors with the command line in `validation/aff3ct/<name>.json`. The LDPC matrices and
polar frozen sets given to AFF3CT are written by comnum
(`COMNUM_RECORD_AFF3CT=1 cargo test -p comnum-validation --test aff3ct`).
`validation/tests/coding/aff3ct.rs` simulates the same configurations: a point passes when
the 99 % Wilson intervals of the two frame error rates overlap and the bit error rates
are within the burst-aware bound of the test. The quick configurations run with
`cargo test` (with at most 10 000 frames and 100 frame errors per point;
`COMNUM_VALIDATION_FULL=1` lifts the limit); the long ones are `#[ignore]`d.

### Where a test goes

Every test has one place, chosen by what it compares with; `tools/test_layout.py`
(run by `just check`) fails on a file out of place.

| The test compares | Place | Named |
| --- | --- | --- |
| an item with itself: round trips, identities, error paths | `#[cfg(test)] mod tests` of its module | a sentence |
| one item with the table, test vector or closed form of one document, deterministically, in under a second | `#[cfg(test)] mod literature` of its module (it may use private items) | `<author>_<year>_<locator>` |
| a chain with a paper, a standard or the industry, or any Monte Carlo result | `validation/tests/<theme>/<key>.rs`, the values in a card of `validation/cards/` | the file after the reference (`<firstauthor><year>`, `ts38101_4`), the tests as sentences |
| the blocks with the recorded output of another tool | `tests/oracle/test_<tool or theme>.py`, the recordings in `tests/oracle/` | the tool |
| a link of the declarative blocks with its expectations | `specs/validation/<name>.yaml` | the reference |
| the Python API (bindings, chains, specifications, the I/O) | `tests/test_<area>.py` | the area |

A result is checked in one place. A Python test does not repeat a check that CI runs
elsewhere (a specification of `ci_specs`, a validation test that is not `#[ignore]`, a
card): it checks that the binding reaches the block, and keeps a statistical check only
when it is the only caller of that binding or when the Rust check is a campaign. The
long Python tests carry the marker `slow` and run on one runner of CI.

The themes of `validation/tests/` are those of the documentation: `coding`, `modulation`,
`multicarrier`, `mimo`, `synchronisation`, `wireless_channels`, `fiber`,
`optical_wireless`, `hardware`, `systems`, `simulation`. Each folder is one test binary
(`main.rs` declares one module per file), so that the eleven link once against the
library instead of seventy times; `cargo test -p comnum-validation --test <theme>`
runs one theme. What the tests of several files share is a function of the library
`validation/src/` (the acceptance rules, the loops until a number of errors, the
harness of a standard test such as `comnum_validation::ts38101_4`), documented and
without `unwrap`. A run that explores rather than checks (variants of the conditions,
a sweep printed for reading) is an example, `validation/examples/<key>_<what>.rs`,
never an ignored test.

### Statistical validation

`validation/tests/<theme>/` holds the Monte Carlo checks, one file per reference document
(`multicarrier/raviteja2019.rs`, `coding/tal2015.rs`, `coding/caire1998.rs`, … named
after its first author and year, listed with what they reproduce in
`validation/literature.yaml` and [`LITERATURE.md`](LITERATURE.md)) and the files below,
which compare with several documents or with closed forms. The helpers they share live
in `validation/src/`: the acceptance rule (`within_three_sigma`), the loops until a
number of errors (`until_errors` on `comnum_sim::Runner`, the only place a validation
test runs in parallel, and `until_errors_with`, sequential), the BPSK link of the code
tests (`bpsk_awgn_llr`, `coded_frame`, `fer`), the crossing of an error rate by
log-linear interpolation (`crossing`, `sweep_crossing`, `log_interp`), the information
rates of shaped ASK (`comnum_validation::shaping`) and the harness of TS 38.101-4
(`comnum_validation::ts38101_4`); the tests call the library for the rest (bit errors,
sources, noise, spectra). A comparison written as data is a validation card of
`validation/cards/` (`validation/cards/README.md`), checked by `tools/cards.py` and
rerun by `just card <id>`.

A test that reproduces values read on a published figure or table holds them in a
card: the test only measures (`Card::check`), the expected values, their tolerances and
the reasons live in the YAML. A test may stay card-less, with its tolerance in the
assertion and the reason beside it, when it checks a closed form (a simulated error
rate within three standard errors of its formula), an invariant or a self-consistency
(a round trip, a symmetry, the agreement of two computations of one quantity), a
recorded reference implementation (`coding/aff3ct.rs`), a claim with no value to read
(an ordering, a diversity order, a slope), or a table reproduced exactly (`assert_eq!`).

- `modulation/closed_forms.rs`: BPSK over AWGN and flat Rayleigh fading; each point
  runs until 100 errors and passes within three standard errors of the closed form
  (`comnum_validation::within_three_sigma`);
- `mimo/mimo.rs`: zero forcing and Alamouti against maximal-ratio combining, and the
  diversity orders of the detectors against Table X of Yang & Hanzo (2015);
- `hardware/converters.rs`, `synchronisation/blind.rs`, `modulation/css.rs`,
  `multicarrier/ce_ofdm.rs`, `mimo/message_passing.rs`, `coding/nr_codes.rs`: the
  converters, blind estimation, chirp spread spectrum, constant-envelope OFDM, LAMA and
  OAMP, the 5G NR codes, each against several papers or closed forms;
- `coding/aff3ct.rs`: the channel codes against AFF3CT;
- `simulation/specs.rs`: the quick specifications of `specs/validation/` that CI does
  not verify, and the check of the list `ci_specs` that CI verifies.

## Versioning and workflow

All crates share one version, set in `[workspace.package]`, and are released together
with a Git tag `vX.Y.Z` (`release.yml`). Papers depend on a tag, never on a branch.

### Records

- **Changelog:** `CHANGELOG.md` in [Keep a Changelog](https://keepachangelog.com/)
  format, one section per release (`## [X.Y.Z] — date`), which the release workflow
  publishes as the release notes.
- **Architecture decision records:** `docs/adr/NNNN-title.md`, one page each (status,
  context, decision, consequences). An ADR is required for a new crate, a new
  dependency between crates of one layer, a change to a core trait, or an exception to
  this guide; it is never deleted, a later one supersedes it.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) scoped by
  crate or area, for example `feat(fec): layered min-sum LDPC decoder`. `main` is
  always green.

### Adding a block

1. Locate it with the placement rules (and [`API.md`](API.md)); record an ADR if none
   applies.
2. Implement it: a configuration struct checked when used, the role trait with its
   `_into` method where one applies, `&mut SimRng` for randomness.
3. Document it with the item template: model in LaTeX, errors, example, references.
4. Expose it to the specifications with a wrapper in `comnum_spec::blocks`: its
   parameters declared once with `block_params!` (type, default, unit, accepted values,
   bounds, meaning) and its ports with `block_info!` (the facts of the frame it needs
   and gives). The parser, `comnum blocks`, the Python classes of `comnum.blocks`
   (`tools/gen_blocks.py`) and the JSON Schema of `comnum schema` all come from this
   declaration; debug builds check the ports while the block runs. A block of another
   crate joins the registry with `comnum_spec::blocks::register` (ADR 0006).
5. Add unit and invariant tests.
6. Check it against its reference, in the place that "Where a test goes" gives: a
   `mod literature` test, a test of `validation/tests/<theme>/`, or a specification of
   `specs/validation/` with `expect`; its entry in `validation/maturity.yaml` and, for a
   document, in `validation/literature.yaml` (`just maturity`, `just literature`).
7. Add the Python binding if users need it (`bindings/comnum-py`), then `just gen-python`
   and `just api-index`, and list it in its theme of `docs/python/api/index.rst` (a test
   fails otherwise).
8. Update the changelog.

### Working with a coding agent

- This guide is `docs/ARCHITECTURE.md`. The root `CLAUDE.md` (read by every agent,
  `AGENTS.md` points to it) stays short: it points to the guide and to `docs/API.md`,
  restates the dependency rule and the placement rule, and lists what `just check`
  runs.
- A personal skill teaches the public API to sessions working in other repositories;
  it points to `examples/` as the canonical usage and is updated with each release.
- Paper repositories read the library and need explicit approval to edit it. Library
  changes made from a paper session must be generic, tested and committed in the
  library repository.
