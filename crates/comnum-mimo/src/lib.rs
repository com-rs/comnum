//! # comnum-mimo — multiple-antenna processing
//!
//! Detection, space-time coding, precoding and capacity on the generic flat model
//! $y = H x + w$, with $H$ the $n_{r} \times n_{t}$ channel matrix. The channel models
//! themselves (Rayleigh and Rician fading, antenna correlation) are in `comnum-channel`
//! (`fading::Mimo`, `spatial`); waveform-specific resource mapping (per-subcarrier
//! detection of MIMO-OFDM) belongs to `comnum-waveform`.
//!
//! | Module | Content |
//! |---|---|
//! | [`detection`] | MR (matched filter), ZF, MMSE, ordered SIC (V-BLAST), exhaustive ML, sphere decoder, K-best; hard and soft outputs, the estimates of the symbols ([`detection::Estimates`]) |
//! | [`message_passing`] | LAMA (complex Bayesian AMP) and OAMP for large systems, their state evolution and the recovery thresholds of LAMA |
//! | [`receiver`] | the receiver of a link: a detector of [`detection`] by name, applied channel use after channel use (spatial layers or Alamouti blocks), with soft outputs |
//! | [`stbc`] | orthogonal space-time block codes: Alamouti, the rate-3/4 codes for 3 and 4 antennas; linear decoding |
//! | [`precoding`] | SVD (eigenmode) transmission with water-filling, ZF and regularised ZF precoders |
//! | [`estimation`] | training-based channel estimation on $S = H P + V$: LS, scaled LS, LMMSE and relaxed MMSE, their MSEs, the orthogonal and MMSE-optimal training, the BLUE combination of training periods (Biguesh and Gershman 2006) |
//! | [`massive`] | the uplink of multi-user massive MIMO: perfect or estimated CSI, the closed-form lower bounds on the rates of MR and ZF receivers (Ngo, Larsson and Marzetta 2013) |
//! | [`capacity`] | capacity of a channel realisation with equal power or water-filling; the ergodic and outage capacities of i.i.d. Rayleigh fading, the eigenvalue density of a Wishart matrix |
//!
//! `comnum-mimo` uses the constellations of `comnum-mapping` (ADR 0008).

pub mod capacity;
pub mod detection;
pub mod estimation;
pub mod massive;
pub mod message_passing;
pub mod precoding;
pub mod receiver;
pub mod stbc;
