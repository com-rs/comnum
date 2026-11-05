//! # comnum — link-level simulation of digital communication systems
//!
//! The facade of the comnum workspace: every crate is re-exported under a short
//! name, so that users write `comnum::channel::awgn` rather than `comnum_channel::awgn`.
//! This crate contains no code of its own (see `docs/ARCHITECTURE.md`).
//!
//! | Module | Crate | Layer | Content |
//! |---|---|---|---|
//! | [`core`] | `comnum-core` | 0 | signal containers, verb traits, unit newtypes, the random generator, the error type |
//! | [`math`] | `comnum-math` | 0 | special functions, quadrature, FFT plans, linear algebra, nearest-point search, the NLSE split-step solvers, GF(2) and GF(2^m), exact integers, Gray code, bisection, regula falsi and golden-section search, the geometry of 3-D space |
//! | [`dsp`] | `comnum-dsp` | 1 | filtering, windows, resampling, synchronisation, adaptive equalisation, dispersion compensation, digital back-propagation, digital predistortion, LS and LMMSE channel estimation |
//! | [`fec`] | `comnum-fec` | 2 | channel coding: CRC, scrambling, interleaving, convolutional codes (Viterbi, BCJR) and puncturing, turbo, LDPC, polar, Reed–Solomon and BCH codes, constant-composition distribution matching, finite-length bounds |
//! | [`mapping`] | `comnum-mapping` | 2 | constellations, Gray labelling, hard and soft demapping, differential PSK, sources, closed-form error and information rates |
//! | [`waveform`] | `comnum-waveform` | 2 | OFDM and DFT-spread OFDM, resource grids and pilots, channel estimation, Schmidl–Cox synchronisation, MIMO-OFDM detection, 5G NR numerology and grid, PAPR and its reduction (clipping and filtering, SLM, PTS, tone reservation, ACE, companding), constant-envelope OFDM, continuous-phase modulation, chirp spread spectrum (LoRa), FBMC/OQAM, UFMC, OTFS |
//! | [`mimo`] | `comnum-mimo` | 2 | MIMO detection (ZF, MMSE, SIC, ML, sphere, K-best), the receiver of a link, space-time block codes, precoding, capacity |
//! | [`channel`] | `comnum-channel` | 2 | AWGN; the `Fading` trait; flat fading (SISO and MIMO, Rayleigh and Rician), antenna correlation; tapped-delay-line multipath (EPA, EVA, ETU, ITU) with Doppler; the clustered delay lines of 3GPP TR 38.901; optical fibre: linear, split-step Fourier, Manakov-PMD, EDFA, Raman amplification and ISRS, the GN and EGN models; optical wireless: indoor rooms with reflections, free-space optics with turbulence and pointing errors |
//! | [`hardware`] | `comnum-hardware` | 2 | power amplifiers (Rapp, Saleh, memory polynomial), phase noise and frequency offset, IQ imbalance; lasers, modulators, photodiodes, coherent receiver |
//! | [`metrics`] | `comnum-metrics` | 2 | error rates, SNR, EVM, information rates, PAPR and CCDF, power spectral density, ACLR, blind estimation (cumulants, modulation classification, SNR), frequency stability of oscillators (Allan variances) |
//! | [`sim`] | `comnum-sim` | 3 | the Monte Carlo framework: grids, scenarios, parallel runner, stopping rules, loggers, statistics |
//! | `spec`, `io` | `comnum-spec` | 4 | declarative campaigns (YAML, JSON, TOML): the blocks of optical, wireless and coded links, checks against the literature, operating-point search, the `comnum` command; recordings I/O (feature `spec`) |
//! | [`systems`] | `comnum-systems` | 4 | chains of standards: DVB-S2 (ETSI EN 302 307-1) with its BCH and LDPC codes; the 5G NR PDSCH with HARQ (TS 38.211, 38.212, 38.214); the LTE cell search and MIB (TS 36.211, 36.212); probabilistic amplitude shaping; the WDM transmitter |
//!
//! # Features
//!
//! - `optical` (default): `channel::optical` and `hardware::optical`;
//! - `spec` (default): `comnum::spec` and `comnum::io`.
//!
//! Blocks of other crates join the declarative links through
//! `comnum::spec::blocks::register` (`docs/adr/0006`).
//!
//! # A Monte Carlo campaign in a few lines
//!
//! ```
//! use comnum::prelude::*;
//!
//! /// 16-QAM over AWGN: the scenario is one trial of a block of symbols.
//! struct Qam16Awgn;
//!
//! impl Scenario for Qam16Awgn {
//!     type State = Constellation;
//!     fn name(&self) -> &str { "qam16-awgn" }
//!     fn setup(&self, _: &Condition) -> comnum::Result<Constellation> {
//!         Ok(Constellation::qam(16)?.normalized())
//!     }
//!     fn trial(&self, c: &Constellation, cond: &Condition, rng: &mut Rng, rec: &mut Record)
//!         -> comnum::Result<()> {
//!         let ebn0 = cond.f64("ebn0_db")?;
//!         let bits = source::bits(4 * 1000, rng);
//!         let tx = c.map_bits(&bits)?;
//!         let rx = awgn::awgn_ebn0(&tx, ebn0, c.bits_per_symbol() as f64, 1.0, rng);
//!         let errors = metrics::bit_errors(&c.demap_bits(&rx), &bits)?;
//!         rec.errors("ber", errors as u64, bits.len() as u64);
//!         Ok(())
//!     }
//! }
//!
//! let grid = Grid::new().axis("ebn0_db", [4.0, 6.0]);
//! let results = Runner::new()
//!     .seed(1)
//!     .batch(8)
//!     .stop(Stop::min_errors("ber", 100).or(Stop::max_trials(200)))
//!     .run(&Qam16Awgn, &grid)
//!     .unwrap();
//! let ber = results[1].metric("ber").unwrap().value();
//! let theory = metrics::theory_ber(ConstellationKind::Qam, 16, 6.0).unwrap();
//! assert!((ber / theory - 1.0).abs() < 0.3);
//! ```
//!
//! # The same link with the multi-stream blocks
//!
//! ```
//! use comnum::prelude::*;
//! use comnum::channel::awgn::Awgn;
//!
//! let c = Constellation::qam(4)?.normalized();
//! let mut rng = batch_rng(42, 0);
//! let bits = Bits::from_streams(&[source::bits(2000, &mut rng), source::bits(2000, &mut rng)])?;
//! let x = c.map(&bits)?;                                         // 2 streams of QPSK
//! let y = Awgn::new(EsN0(Db(20.0))).propagate(&Signal::new(x.as_array().clone(), Hertz(1.0)), &mut rng)?;
//! let llr = c.demap(&Symbols::new(y.as_array().clone()), 0.01)?;
//! assert_eq!((llr.n_streams(), llr.len()), (2, 2000));
//! # Ok::<(), comnum::Error>(())
//! ```

pub use comnum_channel as channel;
pub use comnum_core as core;
pub use comnum_dsp as dsp;
pub use comnum_fec as fec;
pub use comnum_hardware as hardware;
pub use comnum_mapping as mapping;
pub use comnum_math as math;
pub use comnum_metrics as metrics;
pub use comnum_mimo as mimo;
pub use comnum_sim as sim;
#[cfg(feature = "spec")]
pub use comnum_spec as spec;
#[cfg(feature = "spec")]
pub use comnum_spec::io;
pub use comnum_systems as systems;
pub use comnum_waveform as waveform;

pub use comnum_core::{C64, Error, Result};

/// The items used by most simulations.
pub mod prelude {
    pub use comnum_channel::awgn;
    pub use comnum_channel::fading::Fading;
    #[cfg(feature = "optical")]
    pub use comnum_channel::optical::fiber::{self, FiberParams};
    pub use comnum_core::prelude::*;
    pub use comnum_mapping::constellation::{Constellation, ConstellationKind};
    pub use comnum_mapping::source;
    pub use comnum_math::{db2lin, dbm2w, lin2db, w2dbm};
    pub use comnum_metrics as metrics;
    pub use comnum_sim::prelude::*;
}
