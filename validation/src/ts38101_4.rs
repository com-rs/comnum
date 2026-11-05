//! The harness of the PDSCH conformance test 1-2 of 3GPP TS 38.101-4 (clause 5.2.2.1.1):
//! the slot of R.PDSCH.1-1.2, the HARQ timeline, the channel applied per subcarrier and the
//! receivers, shared by `tests/systems/ts38101_4.rs` (the card `ts38101_4_pdsch`) and the
//! example `examples/ts38101_4_alignment.rs` (the conditions of the RAN4 alignment). The
//! link and its parameters are described in the test.

use comnum::channel::multipath::{Profile, Tdl};
use comnum::core::{Error, Result};
use comnum::math::C64;
use comnum::prelude::*;
use comnum::systems::nr::mcs::{McsTable, mcs};
use comnum::systems::nr::pdsch::{Pdsch, PdschReceiver, SYMBOLS_PER_SLOT};
use comnum::systems::nr::transport::{HarqProcess, TransportBlock};
use comnum::waveform::ofdm::nr::PdschDmrs;

/// The subcarrier spacing of the test (Hz).
pub const SCS: f64 = 15e3;
/// The FFT size of the channel's sampling rate: delays on a grid of 1/(15 kHz × 2^16).
pub const N_CHANNEL: usize = 1 << 16;
/// The redundancy versions of the transmissions of a transport block (Table 5.2-1).
pub const RV: [u8; 4] = [0, 2, 3, 1];

/// The PDSCH of R.PDSCH.1-1.2 in the slot `slot` of the frame (6 PRBs from PRB 23 of 52,
/// QPSK, DM-RS in symbols 2 and 11, 2 ports, bundles of 2 PRBs).
pub fn pdsch(slot: u32) -> Pdsch {
    Pdsch {
        n_rb_grid: 52,
        prb_start: 23,
        n_prb: 6,
        first_symbol: 2,
        n_symbols: 12,
        dmrs: PdschDmrs { config_type: 1, type_a_position: 2, additional_position: 1, n_id: 0, slot },
        n_ports: 2,
        prg_size: 2,
        qm: 2,
        rnti: 1,
        n_id: 0,
        subcarrier_spacing_hz: SCS,
    }
}

/// The conditions of a run: those of TS 38.101-4, or those of the RAN4 alignment of
/// R4-1811723 (RAN4 #88, August 2018).
#[derive(Clone, Copy)]
pub struct Conditions {
    /// The channel profile.
    pub profile: Profile,
    /// The transmitter EVM (0.06 in R4-1811723 for QPSK to 64-QAM).
    pub evm: f64,
    /// The TRS symbols.
    pub trs_symbols: [usize; 2],
    /// The receiver.
    pub receiver: Rx,
}

/// The receiver of a run.
#[derive(Clone, Copy)]
pub enum Rx {
    /// The robust Wiener estimator designed for delays up to `max_delay_s`.
    Estimated {
        /// The largest delay the estimator expects (s).
        max_delay_s: f64,
    },
    /// The precoded channel known (genie).
    Genie,
}

/// The conditions of TS 38.101-4: TDLC300 (Table B.2.1.1-4), no EVM, TRS in 6 and 10.
pub const TS_38101_4: Conditions = Conditions {
    profile: Profile::TdlC300,
    evm: 0.0,
    trs_symbols: [6, 10],
    receiver: Rx::Estimated { max_delay_s: CYCLIC_PREFIX_S },
};

/// The cyclic prefix at 15 kHz (s), the delay bound of the robust estimator.
pub const CYCLIC_PREFIX_S: f64 = 4.69e-6;

/// The TRS REs of slots 10 and 11 of every 20 in the allocation.
pub fn trs(slot: usize, p: &Pdsch, symbols: [usize; 2]) -> Vec<(usize, usize)> {
    if slot % 20 != 10 && slot % 20 != 11 {
        return Vec::new();
    }
    let (a, b) = (p.prb_start * 12, (p.prb_start + p.n_prb) * 12);
    symbols.iter().flat_map(|&l| (a..b).step_by(4).map(move |k| (l, k))).collect()
}

/// The fraction of the PDSCH slots that deliver a transport block over `drops`
/// realisations of `slots` slots at `snr_db`.
///
/// # Errors
/// Those of the PDSCH chain (a configuration the blocks reject).
pub fn throughput(c: Conditions, n_rx: usize, snr_db: f64, drops: usize, slots: usize, seed: u64) -> Result<f64> {
    let m = mcs(McsTable::Qam64, 4)?;
    let mut tb = TransportBlock::new(480, m.rate())?;
    let tdl = Tdl::new(&c.profile.tdl_params(100.0, SCS * N_CHANNEL as f64, n_rx, 2)?)?;
    let noise_var = 10f64.powf(-snr_db / 10.0);
    let mut receiver = PdschReceiver::new(
        match c.receiver {
            Rx::Estimated { max_delay_s } => max_delay_s,
            Rx::Genie => CYCLIC_PREFIX_S,
        },
        100.0,
    );
    let mut rng = SimRng::new(seed);
    let (mut delivered, mut used) = (0usize, 0usize);
    let k_grid = 52 * 12;
    for _ in 0..drops {
        let times: Vec<f64> = (0..slots)
            .flat_map(|s| (0..SYMBOLS_PER_SLOT).map(move |l| (s as f64 + (l as f64 + 0.5) / 14.0) * 1e-3))
            .collect();
        let taps = tdl.draw_at(&times, &mut rng);
        // processes: (transport block, buffers, transmissions)
        let mut procs: Vec<Option<(Vec<u8>, HarqProcess)>> = vec![None, None, None, None];
        let mut next = 0;
        for s in 0..slots {
            if s % 20 == 0 {
                continue;
            }
            let p = pdsch((s % 10) as u32);
            let reserved = trs(s, &p, c.trs_symbols);
            let g = p.g(&reserved)?;
            let slot_proc = &mut procs[next % 4];
            next += 1;
            used += 1;
            if slot_proc.is_none() {
                let info: Vec<u8> = (0..480).map(|_| u8::from(rng.uniform() < 0.5)).collect();
                *slot_proc = Some((info, HarqProcess::new(&mut tb)?));
            }
            let Some((info, harq)) = slot_proc.as_mut() else { continue };
            let rv = RV[harq.transmissions()];
            let bits = tb.encode(info, g, 2, 1, rv)?;
            let precoders: Vec<usize> = (0..p.n_prgs()).map(|_| (rng.uniform() * 4.0) as usize % 4).collect();
            let mut x = p.transmit(&bits, &precoders, &reserved)?;
            // the EVM of the transmitter: white noise on every non-empty RE of every port,
            // of variance EVM^2 times the energy of the RE
            if c.evm > 0.0 {
                for v in x.iter_mut().flatten().filter(|v| v.norm_sqr() > 0.0) {
                    *v += rng.complex_gaussian() * (c.evm * v.norm());
                }
            }
            // the channel per subcarrier of the allocation, then the noise
            let mut y = vec![vec![C64::new(0.0, 0.0); SYMBOLS_PER_SLOT * k_grid]; n_rx];
            let ks: Vec<usize> = (p.prb_start * 12..(p.prb_start + p.n_prb) * 12).collect();
            let bins: Vec<usize> = ks.iter().map(|&k| (k + N_CHANNEL - k_grid / 2) % N_CHANNEL).collect();
            for l in 0..SYMBOLS_PER_SLOT {
                let h = taps.frequency_response(s * SYMBOLS_PER_SLOT + l, &bins, N_CHANNEL);
                for (b, &k) in ks.iter().enumerate() {
                    for (r, yr) in y.iter_mut().enumerate() {
                        let i = l * k_grid + k;
                        yr[i] = (0..2).map(|t| h[(b * n_rx + r) * 2 + t] * x[t][i]).sum::<C64>()
                            + rng.complex_gaussian() * noise_var.sqrt();
                    }
                }
            }
            let llr = match c.receiver {
                Rx::Estimated { .. } => p.receive(&y, noise_var, &mut receiver, &reserved)?,
                Rx::Genie => {
                    // the precoded channel: the noiseless received RE over the layer symbol,
                    // x_0 / w_0 with w_0 = 1/sqrt(2) for the two ports of every precoder
                    let mut known = vec![vec![C64::new(0.0, 0.0); SYMBOLS_PER_SLOT * k_grid]; n_rx];
                    for l in 0..SYMBOLS_PER_SLOT {
                        let h = taps.frequency_response(s * SYMBOLS_PER_SLOT + l, &bins, N_CHANNEL);
                        for (b, &k) in ks.iter().enumerate() {
                            let i = l * k_grid + k;
                            let sym = x[0][i] * std::f64::consts::SQRT_2;
                            if sym.norm_sqr() == 0.0 {
                                continue;
                            }
                            for (r, kr) in known.iter_mut().enumerate() {
                                kr[i] = (0..2).map(|t| h[(b * n_rx + r) * 2 + t] * x[t][i]).sum::<C64>() / sym;
                            }
                        }
                    }
                    p.receive_known(&y, &known, noise_var, &reserved)?
                }
            };
            harq.add(&mut tb, &llr, 2, 1, rv)?;
            let ok = harq.decode(&mut tb)?.is_some_and(|d| d == *info);
            if ok {
                delivered += 1;
            }
            if ok || harq.transmissions() == RV.len() {
                *slot_proc = None;
            }
        }
    }
    Ok(delivered as f64 / used as f64)
}

/// The throughput at every SNR of `snrs` and the SNR (dB) where it crosses `target`,
/// linearly interpolated.
///
/// # Errors
/// [`Error::Numerical`] when the grid does not bracket the target; those of
/// [`throughput`].
pub fn crossing(n_rx: usize, snrs: &[f64], target: f64) -> Result<(Vec<f64>, f64)> {
    crossing_with(TS_38101_4, n_rx, snrs, target)
}

/// [`crossing`] under the conditions `c`.
///
/// # Errors
/// As [`crossing`].
pub fn crossing_with(c: Conditions, n_rx: usize, snrs: &[f64], target: f64) -> Result<(Vec<f64>, f64)> {
    let t = snrs.iter().map(|&s| throughput(c, n_rx, s, 20, 200, 38_101)).collect::<Result<Vec<f64>>>()?;
    println!("{n_rx} RX: {:?}", snrs.iter().zip(&t).map(|(s, v)| format!("{s} dB {v:.3}")).collect::<Vec<_>>());
    let i = t
        .windows(2)
        .position(|w| w[0] < target && w[1] >= target)
        .ok_or_else(|| Error::Numerical(format!("the grid {snrs:?} does not bracket {target}")))?;
    let snr = snrs[i] + (target - t[i]) / (t[i + 1] - t[i]) * (snrs[i + 1] - snrs[i]);
    Ok((t, snr))
}
