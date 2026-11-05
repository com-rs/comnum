//! The PDSCH of 5G NR: modulation, MCS and transport block size, the transport channel
//! with HARQ, and the slot of a single-layer PDSCH with its DM-RS and precoding.
//!
//! | Module | Content |
//! |---|---|
//! | [`modulation`] | the modulation mapper of TS 38.211 Sec. 5.1 (QPSK to 1024-QAM) |
//! | [`mcs`] | the MCS tables and the transport block size of TS 38.214 Sec. 5.1.3 |
//! | [`pdsch`] | one slot of a single-layer PDSCH: resource mapping, scrambling, DM-RS, the precoding of every PRG, and a receiver estimating the precoded channel by robust 2-D Wiener filtering |
//! | [`transport`] | TB CRC, code block segmentation, LDPC and rate matching (TS 38.212 Sec. 7.2), the soft buffers of a HARQ process |
//!
//! The numerology, the OFDM carrier, the DM-RS sequences and the codes live in
//! `comnum_waveform::ofdm::nr` and `comnum_fec::nr`; this module assembles them.
//!
//! # References
//!
//! 1. 3GPP TS 38.211 V18.10.0, *NR; Physical channels and modulation*, Sec. 5.1, 7.3.1.
//! 2. 3GPP TS 38.212 V18.5.0, *NR; Multiplexing and channel coding*, Sec. 5.2.2, 5.4.2,
//!    7.2.
//! 3. 3GPP TS 38.214 V19.5.0, *NR; Physical layer procedures for data*, Sec. 5.1.3.

pub mod mcs;
pub mod modulation;
pub mod pdsch;
pub mod transport;
