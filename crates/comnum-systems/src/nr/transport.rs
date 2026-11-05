//! The transport channel of the PDSCH: transport block CRC, code block segmentation, LDPC
//! encoding and rate matching (TS 38.212 Sec. 7.2), and the soft buffers of a HARQ process.

use comnum_core::{Result, ensure, ensure_shape};
use comnum_fec::block::BlockCode;
use comnum_fec::crc::Crc;
use comnum_fec::nr::NrLdpc;
use std::collections::HashMap;

/// The segmentation of a transport block of `tbs` bits at the target code rate `rate`
/// (TS 38.212 Sec. 7.2.1, 7.2.2 and 5.2.2).
///
/// The TB CRC is CRC24A for $A > 3824$, CRC16 otherwise; base graph 2 when $A \le 292$,
/// or $A \le 3824$ and $R \le 0.67$, or $R \le 0.25$, base graph 1 otherwise; with
/// $B = A + L$ and $K_{cb}$ = 8448 (base graph 1) or 3840 (base graph 2), one code block
/// when $B \le K_{cb}$, otherwise $C = \lceil B/(K_{cb} - 24) \rceil$ blocks of
/// $K' = (B + 24C)/C$ bits each ending in a CRC24B.
#[derive(Clone, Debug)]
pub struct TransportBlock {
    tbs: usize,
    crc: Crc,
    base_graph: u8,
    c: usize,
    k_prime: usize,
    /// The codes of every `(E, rv, Qm)`, built once.
    codes: HashMap<(usize, u8, usize), NrLdpc>,
}

impl TransportBlock {
    /// The segmentation of a block of `tbs` bits at the target code rate `rate`.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for an empty block, a rate outside $(0, 1)$, or a block whose
    /// segmentation does not divide evenly (a TBS not produced by TS 38.214).
    pub fn new(tbs: usize, rate: f64) -> Result<TransportBlock> {
        ensure(tbs >= 1, || "the transport block needs at least one bit".into())?;
        ensure(rate > 0.0 && rate < 1.0, || format!("the code rate must lie in (0, 1), got {rate}"))?;
        let crc = if tbs > 3824 { Crc::crc24a() } else { Crc::crc16() };
        let base_graph = if tbs <= 292 || (tbs <= 3824 && rate <= 0.67) || rate <= 0.25 { 2 } else { 1 };
        let b = tbs + crc.width() as usize;
        let k_cb = if base_graph == 1 { 8448 } else { 3840 };
        let (c, b_prime) = if b <= k_cb {
            (1, b)
        } else {
            let c = b.div_ceil(k_cb - 24);
            (c, b + 24 * c)
        };
        ensure(b_prime % c == 0, || format!("B' = {b_prime} is not a multiple of C = {c}"))?;
        Ok(TransportBlock { tbs, crc, base_graph, c, k_prime: b_prime / c, codes: HashMap::new() })
    }

    /// The transport block size $A$.
    pub fn tbs(&self) -> usize {
        self.tbs
    }

    /// The base graph, 1 or 2.
    pub fn base_graph(&self) -> u8 {
        self.base_graph
    }

    /// The number of code blocks $C$.
    pub fn code_blocks(&self) -> usize {
        self.c
    }

    /// The bits per code block $K'$, its CRC included.
    pub fn code_block_bits(&self) -> usize {
        self.k_prime
    }

    /// The rate-matched lengths $E_{r}$ of the code blocks for $G$ channel bits,
    /// $Q_{m}$ bits per symbol and `layers` layers (Sec. 5.4.2.1): the first
    /// $C - (G/(N_{L} Q_{m}) \bmod C)$ blocks get $N_{L} Q_{m} \lfloor G/(N_{L} Q_{m} C)
    /// \rfloor$ bits, the others $N_{L} Q_{m} \lceil G/(N_{L} Q_{m} C) \rceil$.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] unless $G$ is a positive multiple of $N_{L} Q_{m}$.
    pub fn rate_matched_lengths(&self, g: usize, qm: usize, layers: usize) -> Result<Vec<usize>> {
        let unit = qm * layers;
        ensure(unit >= 1 && g >= unit && g % unit == 0, || {
            format!("G = {g} is not a positive multiple of Qm NL = {unit}")
        })?;
        let symbols = g / unit;
        let floor_blocks = self.c - symbols % self.c;
        Ok((0..self.c)
            .map(|r| if r < floor_blocks { unit * (symbols / self.c) } else { unit * symbols.div_ceil(self.c) })
            .collect())
    }

    /// The code of one code block rate-matched to `e` bits with the redundancy version
    /// `rv` and the bit interleaver of `qm` bits.
    fn code(&mut self, e: usize, rv: u8, qm: usize) -> Result<&NrLdpc> {
        let (k, bg) = (self.k_prime, self.base_graph);
        Ok(match self.codes.entry((e, rv, qm)) {
            std::collections::hash_map::Entry::Occupied(o) => o.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => v.insert(NrLdpc::with_options(k, e, Some(bg), rv, qm)?),
        })
    }

    /// The $G$ channel bits of the transport block `tb` (TS 38.212 Sec. 7.2): TB CRC,
    /// segmentation and CRC24B, LDPC encoding, rate matching with the redundancy version
    /// `rv` and bit interleaving, code block concatenation.
    ///
    /// # Errors
    /// [`comnum_core::Error::Shape`] unless `tb` has `tbs` bits; those of
    /// [`TransportBlock::rate_matched_lengths`].
    pub fn encode(&mut self, tb: &[u8], g: usize, qm: usize, layers: usize, rv: u8) -> Result<Vec<u8>> {
        ensure_shape(tb.len() == self.tbs, || {
            format!("the transport block has {} bits, expected {}", tb.len(), self.tbs)
        })?;
        let b = self.crc.attach(tb);
        let per = if self.c == 1 { self.k_prime } else { self.k_prime - 24 };
        let lengths = self.rate_matched_lengths(g, qm, layers)?;
        let mut out = Vec::with_capacity(g);
        for (r, &e) in lengths.iter().enumerate() {
            let chunk = &b[r * per..(r + 1) * per];
            let block = if self.c == 1 { chunk.to_vec() } else { Crc::crc24b().attach(chunk) };
            out.extend(self.code(e, rv, qm)?.encode_block(&block)?);
        }
        Ok(out)
    }
}

/// The soft buffers of one HARQ process: one per code block, the LLRs of every
/// transmission of the transport block added on the bits of the full codeword.
#[derive(Clone, Debug)]
pub struct HarqProcess {
    buffers: Vec<Vec<f64>>,
    transmissions: usize,
}

impl HarqProcess {
    /// An empty process for the transport block `tb`.
    ///
    /// # Errors
    /// Those of the LDPC code of the block.
    pub fn new(tb: &mut TransportBlock) -> Result<HarqProcess> {
        // the buffer has the length of the full codeword, the same for every E and rv
        let n = tb.code(tb.k_prime.max(24) * 3, 0, 1)?.code().n();
        Ok(HarqProcess { buffers: vec![vec![0.0; n]; tb.c], transmissions: 0 })
    }

    /// The transmissions combined so far.
    pub fn transmissions(&self) -> usize {
        self.transmissions
    }

    /// Add the $G$ LLRs of one transmission of `tb` with the redundancy version `rv`
    /// (positive for a 0, as `comnum_mapping` demaps).
    ///
    /// # Errors
    /// [`comnum_core::Error::Shape`] unless `llr` has $G$ values; those of
    /// [`TransportBlock::rate_matched_lengths`].
    pub fn add(&mut self, tb: &mut TransportBlock, llr: &[f64], qm: usize, layers: usize, rv: u8) -> Result<()> {
        let lengths = tb.rate_matched_lengths(llr.len(), qm, layers)?;
        let mut start = 0;
        for (r, &e) in lengths.iter().enumerate() {
            tb.code(e, rv, qm)?.combine(&llr[start..start + e], &mut self.buffers[r])?;
            start += e;
        }
        self.transmissions += 1;
        Ok(())
    }

    /// Decode the buffers: the transport block when every code block CRC (for $C > 1$)
    /// and the TB CRC pass, `None` otherwise.
    ///
    /// # Errors
    /// Those of the LDPC decoder.
    pub fn decode(&self, tb: &mut TransportBlock) -> Result<Option<Vec<u8>>> {
        let c = tb.c;
        let mut b = Vec::with_capacity(tb.tbs + tb.crc.width() as usize);
        // any E decodes the buffer: the code is the same, only the selection differs
        let code = tb.code(tb.k_prime.max(24) * 3, 0, 1)?;
        for buffer in &self.buffers {
            let block = code.decode_buffer(buffer)?;
            if c > 1 {
                if !Crc::crc24b().check(&block) {
                    return Ok(None);
                }
                b.extend_from_slice(&block[..block.len() - 24]);
            } else {
                b.extend(block);
            }
        }
        Ok(tb.crc.check(&b).then(|| b[..tb.tbs].to_vec()))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code: a failure is a test failure")]
    use super::*;

    fn bits(n: usize, seed: u64) -> Vec<u8> {
        let mut rng = comnum_core::rng::Rng::new(seed);
        (0..n).map(|_| u8::from(rng.uniform() < 0.5)).collect()
    }

    #[test]
    fn the_segmentation_of_the_frcs_of_ts_38101_4() {
        // (TBS, R, C) of R.PDSCH.1-1.1, 1-1.2, 1-2.1, 1-2.2, 1-3.1, 1-4.1, 2-1.1
        for (a, r, c) in [
            (3904, 308.0 / 1024.0, 1),
            (480, 308.0 / 1024.0, 1),
            (13064, 490.0 / 1024.0, 2),
            (26120, 490.0 / 1024.0, 4),
            (42016, 517.0 / 1024.0, 5),
            (45096, 841.0 / 1024.0, 6),
            (40976, 517.0 / 1024.0, 5),
        ] {
            assert_eq!(TransportBlock::new(a, r).unwrap().code_blocks(), c, "TBS {a}");
        }
        let tb = TransportBlock::new(480, 308.0 / 1024.0).unwrap();
        assert_eq!((tb.base_graph(), tb.code_block_bits()), (2, 496));
    }

    #[test]
    fn the_rate_matched_lengths_share_g() {
        let tb = TransportBlock::new(13064, 490.0 / 1024.0).unwrap();
        assert_eq!(tb.rate_matched_lengths(27456, 4, 1).unwrap(), vec![13728, 13728]);
        // 26208 / 4 = 6552 symbols over 2 blocks; and an odd count: 6553 symbols
        assert_eq!(tb.rate_matched_lengths(26212, 4, 1).unwrap(), vec![13104, 13108]);
        assert!(tb.rate_matched_lengths(27455, 4, 1).is_err());
    }

    #[test]
    fn a_noiseless_transport_block_round_trips_and_harq_combines() {
        for (a, r, g, qm) in [(480, 308.0 / 1024.0, 1584, 2), (13064, 490.0 / 1024.0, 27456, 4)] {
            let mut tb = TransportBlock::new(a, r).unwrap();
            let info = bits(a, a as u64);
            let mut harq = HarqProcess::new(&mut tb).unwrap();
            let c = tb.encode(&info, g, qm, 1, 0).unwrap();
            assert_eq!(c.len(), g);
            let llr: Vec<f64> = c.iter().map(|b| if *b == 0 { 2.0 } else { -2.0 }).collect();
            harq.add(&mut tb, &llr, qm, 1, 0).unwrap();
            assert_eq!(harq.decode(&mut tb).unwrap(), Some(info.clone()));
            // a corrupted block fails the CRC
            let mut bad = HarqProcess::new(&mut tb).unwrap();
            let wrong: Vec<f64> = llr.iter().map(|v| -v).collect();
            bad.add(&mut tb, &wrong, qm, 1, 0).unwrap();
            assert_eq!(bad.decode(&mut tb).unwrap(), None);
        }
    }
}
