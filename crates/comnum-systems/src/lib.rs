//! # comnum-systems — reference chains of standards
//!
//! Standard-compliant chains assembled from the blocks of layers 0 to 2. New algorithms
//! do not belong here: they go to the functional crates.
//!
//! | Module | Standard |
//! |---|---|
//! | [`ais`] | AIS (ITU-R M.1371-5): the GMSK/NRZI/HDLC packets of Annex 2, a receiver for a capture, the messages of Annex 8, the `!AIVDM` sentences of NMEA 0183 |
//! | [`ccsds`] | CCSDS 131.0-B-5 TM synchronization: attached sync markers, pseudo-randomizers, the convolutional code (G2 inverted, Table 3-1), channel access data units; its codes are `comnum_fec::ccsds` |
//! | [`dvb`] | the channel coding DVB-S and DVB-T share (ETSI EN 300 421, EN 300 744): energy dispersal, RS(204, 188), convolutional interleaver, punctured inner code |
//! | [`dvbs`] | DVB-S (ETSI EN 300 421): the outer coding of [`dvb`], the punctured inner code on the Gray QPSK of its Figure 5, a frame-aligned receiver |
//! | [`dvbt`] | DVB-T (ETSI EN 300 744): the outer coding of [`dvb`], the inner interleaver, the constellations of every hierarchy, the TPS and the OFDM frame of `comnum_waveform::ofdm::dvbt`, a receiver that knows the channel |
//! | [`dvbs2`] | DVB-S2 (ETSI EN 302 307-1): BBFRAME, BCH and LDPC ([`dvbs2::fec`]), bit interleaving, QPSK to 32APSK, PL framing with pilots and Gold scrambling |
//! | [`hdlc`] | HDLC framing (ISO/IEC 13239): the flag, bit stuffing, the CRC-16/X-25 frame check sequence, NRZI |
//! | [`fm`] | FM sound broadcasting (ITU-R BS.450-4): the stereophonic multiplex, the RDS (NRSC-4 / EN 50067): its demodulator, block code, groups and messages |
//! | [`lte`] | the LTE downlink (3GPP TS 36.211, 36.212): the cell search and the master information block of the broadcast channel, a cell's frames for simulation |
//! | [`nr`] | 5G NR (3GPP TS 38.211, 38.212, 38.214): the modulation mapper, the MCS tables and the transport block size, the transport channel with HARQ soft combining |
//! | [`pas`] | probabilistic amplitude shaping (Böcherer, Steiner & Schulte 2015): a constant-composition matcher, Gray PAM and a systematic LDPC code whose parity bits are the signs |
//! | [`wdm`] | a WDM transmitter (OptiCommPy's `simpleWDMTx`): pulse shaping, lasers with phase noise and IQ modulators on a grid of channels |
//!
//! A chain of another standard that assembles several crates of layer 2 has its place
//! here (see the placement rules of `docs/ARCHITECTURE.md`); the 5G NR numerology, grid
//! and codes live in `comnum-waveform` and `comnum-fec`, which own them.

pub mod ais;
pub mod ccsds;
pub mod dvb;
pub mod dvbs;
pub mod dvbs2;
pub mod dvbt;
pub mod fm;
pub mod hdlc;
pub mod lte;
pub mod nr;
pub mod pas;
pub mod wdm;
