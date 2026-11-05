//! The modulation and coding schemes and the transport block size of the PDSCH (TS
//! 38.214 Sec. 5.1.3).

use comnum_core::{Error, Result, ensure};

/// An MCS index table of the PDSCH (TS 38.214 Sec. 5.1.3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum McsTable {
    /// Table 5.1.3.1-1, up to 64-QAM ("64QAM" of TS 38.101-4).
    Qam64,
    /// Table 5.1.3.1-2, up to 256-QAM.
    Qam256,
    /// Table 5.1.3.1-3, the low spectral efficiencies ("64QAMLowSE").
    Qam64LowSe,
    /// Table 5.1.3.1-4, up to 1024-QAM.
    Qam1024,
}

/// A modulation and coding scheme: $Q_{m}$ bits per symbol and the target code rate
/// $R \times 1024$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mcs {
    /// Bits per modulation symbol $Q_{m}$.
    pub qm: usize,
    /// Target code rate times 1024 (682.5, 916.5, 805.5 and 900.5 are not integers).
    pub rate_x1024: f64,
}

impl Mcs {
    /// The target code rate $R$.
    pub fn rate(&self) -> f64 {
        self.rate_x1024 / 1024.0
    }
}

/// `(Q_m, R x 1024)` of the non-reserved indices of the four tables, verbatim.
const TABLE_1: [(usize, f64); 29] = [
    (2, 120.0),
    (2, 157.0),
    (2, 193.0),
    (2, 251.0),
    (2, 308.0),
    (2, 379.0),
    (2, 449.0),
    (2, 526.0),
    (2, 602.0),
    (2, 679.0),
    (4, 340.0),
    (4, 378.0),
    (4, 434.0),
    (4, 490.0),
    (4, 553.0),
    (4, 616.0),
    (4, 658.0),
    (6, 438.0),
    (6, 466.0),
    (6, 517.0),
    (6, 567.0),
    (6, 616.0),
    (6, 666.0),
    (6, 719.0),
    (6, 772.0),
    (6, 822.0),
    (6, 873.0),
    (6, 910.0),
    (6, 948.0),
];
const TABLE_2: [(usize, f64); 28] = [
    (2, 120.0),
    (2, 193.0),
    (2, 308.0),
    (2, 449.0),
    (2, 602.0),
    (4, 378.0),
    (4, 434.0),
    (4, 490.0),
    (4, 553.0),
    (4, 616.0),
    (4, 658.0),
    (6, 466.0),
    (6, 517.0),
    (6, 567.0),
    (6, 616.0),
    (6, 666.0),
    (6, 719.0),
    (6, 772.0),
    (6, 822.0),
    (6, 873.0),
    (8, 682.5),
    (8, 711.0),
    (8, 754.0),
    (8, 797.0),
    (8, 841.0),
    (8, 885.0),
    (8, 916.5),
    (8, 948.0),
];
const TABLE_3: [(usize, f64); 29] = [
    (2, 30.0),
    (2, 40.0),
    (2, 50.0),
    (2, 64.0),
    (2, 78.0),
    (2, 99.0),
    (2, 120.0),
    (2, 157.0),
    (2, 193.0),
    (2, 251.0),
    (2, 308.0),
    (2, 379.0),
    (2, 449.0),
    (2, 526.0),
    (2, 602.0),
    (4, 340.0),
    (4, 378.0),
    (4, 434.0),
    (4, 490.0),
    (4, 553.0),
    (4, 616.0),
    (6, 438.0),
    (6, 466.0),
    (6, 517.0),
    (6, 567.0),
    (6, 616.0),
    (6, 666.0),
    (6, 719.0),
    (6, 772.0),
];
const TABLE_4: [(usize, f64); 27] = [
    (2, 120.0),
    (2, 193.0),
    (2, 449.0),
    (4, 378.0),
    (4, 490.0),
    (4, 616.0),
    (6, 466.0),
    (6, 517.0),
    (6, 567.0),
    (6, 616.0),
    (6, 666.0),
    (6, 719.0),
    (6, 772.0),
    (6, 822.0),
    (6, 873.0),
    (8, 682.5),
    (8, 711.0),
    (8, 754.0),
    (8, 797.0),
    (8, 841.0),
    (8, 885.0),
    (8, 916.5),
    (8, 948.0),
    (10, 805.5),
    (10, 853.0),
    (10, 900.5),
    (10, 948.0),
];

/// Table 5.1.3.2-1: the TBS for $N_{info} \le 3824$, verbatim.
pub const TBS_TABLE: [usize; 93] = [
    24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 208, 224, 240,
    256, 272, 288, 304, 320, 336, 352, 368, 384, 408, 432, 456, 480, 504, 528, 552, 576, 608, 640, 672, 704, 736, 768,
    808, 848, 888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 1256, 1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672,
    1736, 1800, 1864, 1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472, 2536, 2600, 2664, 2728, 2792, 2856, 2976, 3104,
    3240, 3368, 3496, 3624, 3752, 3824,
];

/// The MCS `index` of `table` (TS 38.214 Tables 5.1.3.1-1 to 5.1.3.1-4).
///
/// # Errors
/// [`Error::Config`] for a reserved or out-of-range index (the reserved indices of the
/// retransmissions keep the TBS of the first transmission, Sec. 5.1.3.2 step 5).
///
/// # Examples
/// ```
/// use comnum_systems::nr::mcs::{mcs, McsTable};
/// let m = mcs(McsTable::Qam64, 4)?;               // R.PDSCH.1-1.x of TS 38.101-4
/// assert_eq!((m.qm, m.rate_x1024), (2, 308.0));
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn mcs(table: McsTable, index: usize) -> Result<Mcs> {
    let rows: &[(usize, f64)] = match table {
        McsTable::Qam64 => &TABLE_1,
        McsTable::Qam256 => &TABLE_2,
        McsTable::Qam64LowSe => &TABLE_3,
        McsTable::Qam1024 => &TABLE_4,
    };
    rows.get(index).map(|&(qm, rate_x1024)| Mcs { qm, rate_x1024 }).ok_or_else(|| {
        Error::Config(format!("MCS {index} of {table:?} is reserved or out of range (0 to {})", rows.len() - 1))
    })
}

/// $\lfloor \log_{2} x \rfloor$ of $x \ge 1$.
fn floor_log2(x: f64) -> u32 {
    (x.floor() as u64).ilog2()
}

/// The transport block size of a PDSCH (TS 38.214 Sec. 5.1.3.2): `n_prb` PRBs of
/// `n_symb` symbols, `n_dmrs` DM-RS REs per PRB (the CDM groups without data included),
/// the overhead `n_oh` per PRB, the MCS `mcs` and `layers` layers.
///
/// $N_{RE} = \min(156, 12 N_{symb} - N_{DMRS} - N_{oh})\\, n_{PRB}$,
/// $N_{info} = N_{RE} R Q_{m} \upsilon$; for $N_{info} \le 3824$,
/// $N'_{info} = \max(24, 2^{n} \lfloor N_{info}/2^{n} \rfloor)$,
/// $n = \max(3, \lfloor \log_{2} N_{info} \rfloor - 6)$, and the smallest TBS of
/// [`TBS_TABLE`] not below it; otherwise
/// $N'_{info} = \max(3840, 2^{n}\, \mathrm{round}((N_{info} - 24)/2^{n}))$ (ties up),
/// $n = \lfloor \log_{2}(N_{info} - 24) \rfloor - 5$, and
/// $8 C \lceil (N'_{info} + 24)/(8C) \rceil - 24$ with $C = \lceil (N'_{info} + 24)/3816 \rceil$
/// for $R \le 1/4$, $C = \lceil (N'_{info} + 24)/8424 \rceil$ for $N'_{info} > 8424$,
/// and $C = 1$ otherwise. The arithmetic is exact: $N_{info}$ is a multiple of $1/2048$.
///
/// # Errors
/// [`Error::Config`] for no PRB, symbol or layer, or no RE left for data.
///
/// # Examples
/// ```
/// use comnum_systems::nr::mcs::{mcs, transport_block_size, McsTable};
/// // R.PDSCH.1-1.2 of TS 38.101-4: 6 PRBs, 12 symbols, 12 DM-RS REs, MCS 4, 1 layer
/// assert_eq!(transport_block_size(6, 12, 12, 0, mcs(McsTable::Qam64, 4)?, 1)?, 480);
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn transport_block_size(
    n_prb: usize,
    n_symb: usize,
    n_dmrs: usize,
    n_oh: usize,
    mcs: Mcs,
    layers: usize,
) -> Result<usize> {
    ensure(n_prb >= 1 && n_symb >= 1 && layers >= 1, || "at least one PRB, symbol and layer".into())?;
    let n_re_prime = (12 * n_symb).checked_sub(n_dmrs + n_oh).filter(|n| *n > 0);
    let n_re_prime = n_re_prime.ok_or_else(|| Error::Config("no resource element left for data".into()))?;
    let n_re = n_re_prime.min(156) * n_prb;
    // exact: N_RE (2 R 1024) Q_m v / 2048
    let n_info = (n_re as f64) * (2.0 * mcs.rate_x1024) * (mcs.qm * layers) as f64 / 2048.0;
    if n_info <= 3824.0 {
        let n = floor_log2(n_info).saturating_sub(6).max(3);
        let step = f64::from(1u32 << n);
        let n_info_q = (step * (n_info / step).floor()).max(24.0);
        return TBS_TABLE
            .iter()
            .copied()
            .find(|&t| t as f64 >= n_info_q)
            .ok_or_else(|| Error::Config(format!("no TBS for N'_info = {n_info_q}")));
    }
    let n = floor_log2(n_info - 24.0) - 5;
    let step = 2f64.powi(n as i32);
    let n_info_q = (step * ((n_info - 24.0) / step + 0.5).floor()).max(3840.0);
    let per = |c: f64| 8.0 * c * ((n_info_q + 24.0) / (8.0 * c)).ceil() - 24.0;
    let t = if mcs.rate_x1024 <= 256.0 {
        per(((n_info_q + 24.0) / 3816.0).ceil())
    } else if n_info_q > 8424.0 {
        per(((n_info_q + 24.0) / 8424.0).ceil())
    } else {
        per(1.0)
    };
    Ok(t as usize)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code: a failure is a test failure")]
    use super::*;

    #[test]
    fn the_tables_have_their_spectral_efficiencies() {
        // Table 5.1.3.1-1: 0.6016 at MCS 4, 5.5547 at MCS 28; Table 5.1.3.1-2: 7.4063 at 27
        let se = |t, i| {
            let m = mcs(t, i).unwrap();
            m.qm as f64 * m.rate()
        };
        assert!((se(McsTable::Qam64, 4) - 0.6016).abs() < 5e-5);
        assert!((se(McsTable::Qam64, 28) - 5.5547).abs() < 5e-5);
        assert!((se(McsTable::Qam256, 27) - 7.4063).abs() < 5e-5);
        assert!((se(McsTable::Qam1024, 26) - 9.2578).abs() < 5e-5);
        assert!(mcs(McsTable::Qam64, 29).is_err());
        assert!(mcs(McsTable::Qam256, 28).is_err());
        assert_eq!(TBS_TABLE[37], 480);
    }

    #[test]
    fn the_tbs_of_the_frcs_of_ts_38101_4_annex_a() {
        let q64 = |i| mcs(McsTable::Qam64, i).unwrap();
        // (PRBs, DM-RS REs, MCS, layers, TBS) of R.PDSCH.1-1.1, 1-1.2, 1-2.1, 1-2.2, 1-3.1,
        // 1-4.1 and 2-1.1 (12 symbols, no overhead), Tables A.3.2.1.1-1 to -4, A.3.2.1.2-1
        let frcs = [
            (52, 18, q64(4), 1, 3904),
            (6, 12, q64(4), 1, 480),
            (52, 12, q64(13), 1, 13064),
            (52, 12, q64(13), 2, 26120),
            (52, 12, q64(19), 2, 42016),
            (52, 12, mcs(McsTable::Qam256, 24).unwrap(), 1, 45096),
            (51, 12, q64(19), 2, 40976),
        ];
        for (prb, dmrs, m, v, want) in frcs {
            assert_eq!(transport_block_size(prb, 12, dmrs, 0, m, v).unwrap(), want, "{prb} PRB, MCS {m:?}, {v} layers");
        }
        assert!(transport_block_size(1, 1, 12, 0, q64(4), 1).is_err());
    }
}
