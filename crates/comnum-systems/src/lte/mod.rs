//! The LTE downlink as a terminal reads it first: the cell search and the master
//! information block of the broadcast channel, on measured or simulated signals.
//!
//! | Module | Content |
//! |---|---|
//! | [`bch`] | the BCH: CRC masked by the number of ports, tail-biting convolutional code, rate matching (TS 36.212 Sec. 5.3.1) |
//! | [`mib`] | the fields of the master information block (TS 36.331 Sec. 6.2.2) |
//! | [`cell`] | the frames of a cell: PSS, SSS, CRS and PBCH on their grids, for simulation and tests |
//! | [`search`] | the cell search (PSS, frequency offset, SSS) and the decoding of the MIB at 1.92 MS/s |
//!
//! The signals and the grid live in `comnum_waveform::ofdm::lte`, the codes in
//! `comnum_fec::conv` and `comnum_fec::crc`, the front end that brings a capture to
//! 1.92 MS/s in `comnum_dsp::frontend`; this module assembles them.
//!
//! # References
//!
//! 1. 3GPP TS 36.211 V18.0.2, *E-UTRA; Physical channels and modulation*.
//! 2. 3GPP TS 36.212 V18.1.0, *E-UTRA; Multiplexing and channel coding*.
//! 3. 3GPP TS 36.331, *E-UTRA; Radio Resource Control (RRC); Protocol specification*.

pub mod bch;
pub mod cell;
pub mod mib;
pub mod search;
