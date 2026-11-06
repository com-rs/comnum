//! Signal-to-interference ratio of a multicarrier scheme with a one-tap equaliser over a
//! doubly-selective channel, in closed form from its pulses (Nissel, Schwarz & Rupp 2017,
//! Sec. IV).
//!
//! # Model
//!
//! The transmit pulses $\mathbf{g}_{j}$ (the columns of $\mathbf{G}$) carry uncorrelated
//! unit-power symbols $x_{j}$ through a time-variant channel $\[\mathbf{H}\]_{n,n'} = h\[n -
//! n', n\]$ and the receiver projects on the pulse $\mathbf{q}_{i}$ of the wanted symbol
//! (their eq. (34)):
//!
//! $$
//! y_{i} = \sum_{j} \underbrace{\mathbf{q}_{i}^{H} \mathbf{H} \mathbf{g}_{j}}_{c_{j}}\, x_{j},
//! \qquad \[\boldsymbol\Gamma\]_{j,j'} = \mathrm{E}\\{c_{j} c_{j'}^{*}\\}.
//! $$
//!
//! The channel is WSSUS: taps of delays $d_{m}$ (samples) and powers $p_{m}$, independent,
//! each with the Clarke/Jakes correlation $J_{0}(2\pi \nu_{\max} \Delta t\,(n - n'))$,
//! $\nu_{\max} \Delta t$ the maximum Doppler shift in cycles per sample. $J_{0}(x) =
//! \frac{1}{Q}\sum_{q} \cos(x \cos\theta_{q})$ on the midpoint rule, exactly to the
//! machine precision for $Q \ge 64 + 2 x_{\max}$ (`comnum_math::bessel_j0`), so the
//! channel is the finite sum of sinusoids $h\[m, n\] = \sum_{q} \sqrt{p_{m}/Q}\,
//! e^{j 2\pi \nu_{\max} \Delta t \cos\theta_{q}\, n} z_{m,q}$, $z$ i.i.d. unit complex
//! Gaussian, and $\boldsymbol\Gamma = \boldsymbol\Omega \boldsymbol\Omega^{H}$ (their eq.
//! (37)) with the explicit factor
//!
//! $$
//! \[\boldsymbol\Omega\]_{j,(m,q)} = \sqrt{\frac{p_{m}}{Q}} \sum_{n} q_{i}^{*}\[n\]\,
//! e^{j 2\pi \nu_{\max}\Delta t \cos\theta_{q}\, n}\, g_{j}\[n - d_{m}\].
//! $$
//!
//! For complex symbols (OFDM, FBMC-QAM), their eq. (35),
//! $\mathrm{SIR}_{i} = \[\boldsymbol\Gamma\]_{i,i} / (\mathrm{tr}\,\boldsymbol\Gamma -
//! \[\boldsymbol\Gamma\]_{i,i})$ ([`DoublySelective::sir_qam`]). For the real symbols of
//! FBMC-OQAM the one-tap equaliser removes the phase of the wanted coefficient and keeps
//! the real part: each column of $\boldsymbol\Omega$ is turned by the phase of its entry
//! in row $i$ (eq. (38); a zero entry is left as it is), $\tilde{\boldsymbol\Gamma}_{i} =
//! \Re\\{\tilde{\boldsymbol\Omega}_{i}\\} \Re\\{\tilde{\boldsymbol\Omega}_{i}\\}^{T}$ (eq.
//! (39)) and the SIR is the same ratio of $\tilde{\boldsymbol\Gamma}_{i}$ (eq. (40),
//! [`DoublySelective::sir_oqam`]). The receiver knows the one tap exactly; the noise is
//! zero; the pulses are the sampled ones of the modems (`Ofdm::pulse`, `Ofdm::window`,
//! `Fbmc::pulse`), the delays rounded to the sampling grid.
//!
//! When the transmit pulses form a complete orthonormal basis (OFDM without prefix, every
//! subcarrier of every symbol) the received power is the channel's whatever the delays and
//! the Doppler, by Parseval: $\mathrm{tr}\,\boldsymbol\Gamma = \sum_{m} p_{m}
//! \|\mathbf{q}_{i}\|^{2}$, and the SIR needs the wanted coefficient only,
//! $\mathrm{SIR}_{i} = \[\boldsymbol\Gamma\]_{i,i} / (\|\mathbf{q}_{i}\|^{2} -
//! \[\boldsymbol\Gamma\]_{i,i})$ ([`DoublySelective::sir_complete`]). The same holds for
//! the real symbols of FBMC-OQAM on its whole lattice, a complete real-orthonormal frame
//! ($\Re\{\mathbf{G}^{H}\mathbf{G}\} = \mathbf{I}$, their eq. (25)), since $\sum_{j}
//! (\Re\langle \mathbf{y}, \mathbf{g}_{j}\rangle)^{2} = \|\mathbf{y}\|^{2}$ for every
//! $\mathbf{y}$ and the phases of eq. (38) leave $\[\tilde{\boldsymbol\Gamma}_{i}\]_{i,i} =
//! \[\boldsymbol\Gamma\]_{i,i}$: no truncation of the interferers is needed, where the sums
//! of eqs. (35) and (40) over a finite set miss the interference the channel spreads far
//! from the wanted symbol (a rectangular window spreads its ISI evenly over every
//! subcarrier).
//!
//! # Examples
//! ```
//! use comnum_waveform::sir::DoublySelective;
//! use comnum_waveform::ofdm::Ofdm;
//! // CP-OFDM over a static two-path channel inside the prefix: no interference
//! let ofdm = Ofdm { n_cp: 8, oversampling: 1, ..Ofdm::new(32, 32)? };
//! let ch = DoublySelective::new(&[(0.0, 0.0), (3.0, -6.0)], 1.0, 0.0)?;
//! let tx: Vec<_> = (0..3).flat_map(|n| (0..32).map(move |i| (i, n))).map(|(i, n)| ofdm.pulse(i, n)).collect();
//! let sir = ch.sir_qam(&ofdm.window(5, 1), &tx, 32 + 5);
//! assert!(sir > 1e20);
//! // without the prefix the paths beyond the window interfere
//! let ofdm = Ofdm { oversampling: 1, ..Ofdm::new(32, 32)? };
//! let tx: Vec<_> = (0..3).flat_map(|n| (0..32).map(move |i| (i, n))).map(|(i, n)| ofdm.pulse(i, n)).collect();
//! let sir = ch.sir_qam(&ofdm.window(5, 1), &tx, 32 + 5);
//! assert!(sir > 1.0 && sir < 1e3);
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//! 1. R. Nissel, S. Schwarz and M. Rupp, "Filter bank multicarrier modulation schemes for
//!    future mobile communications," *IEEE J. Sel. Areas Commun.*, 35(8), 1768–1782,
//!    2017: Sec. IV, eqs. (34)–(40).

use comnum_core::{Result, ensure};
use comnum_math::{C64, cis};

/// A pulse on its support: `(first sample, values)`, as the modems return them.
pub type Pulse = (usize, Vec<C64>);

/// A WSSUS channel of discrete taps with Jakes' Doppler, on the sampling grid (see the
/// module).
#[derive(Clone, Debug, PartialEq)]
pub struct DoublySelective {
    /// Delays of the taps (samples), distinct and increasing.
    delays: Vec<usize>,
    /// Powers of the taps (linear, unit sum).
    powers: Vec<f64>,
    /// The maximum Doppler shift in cycles per sample, $\nu_{\max}\Delta t$.
    doppler: f64,
}

impl DoublySelective {
    /// The channel of the taps `(delay in s, power in dB)` sampled at `sample_rate` (Hz),
    /// the delays rounded to the nearest sample (taps that fall on the same sample add
    /// their powers) and the powers normalised to a unit sum, with the maximum Doppler
    /// shift `max_doppler_hz` (Hz, Jakes' spectrum; 0 for a static channel).
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] without taps, for a negative delay, Doppler or sample rate, or a
    /// power that is not finite.
    pub fn new(taps: &[(f64, f64)], sample_rate: f64, max_doppler_hz: f64) -> Result<DoublySelective> {
        ensure(!taps.is_empty(), || "a channel needs at least one tap".into())?;
        ensure(sample_rate > 0.0 && max_doppler_hz >= 0.0, || {
            format!("sample rate {sample_rate} Hz and Doppler {max_doppler_hz} Hz: positive and non-negative")
        })?;
        let mut rounded: Vec<(usize, f64)> = Vec::with_capacity(taps.len());
        for &(delay_s, power_db) in taps {
            ensure(delay_s >= 0.0 && power_db.is_finite(), || {
                format!("tap at {delay_s} s, {power_db} dB: a non-negative delay and a finite power")
            })?;
            let d = (delay_s * sample_rate).round() as usize;
            let p = 10f64.powf(power_db / 10.0);
            match rounded.iter_mut().find(|(dd, _)| *dd == d) {
                Some((_, pp)) => *pp += p,
                None => rounded.push((d, p)),
            }
        }
        rounded.sort_by_key(|&(d, _)| d);
        let total: f64 = rounded.iter().map(|(_, p)| p).sum();
        Ok(DoublySelective {
            delays: rounded.iter().map(|&(d, _)| d).collect(),
            powers: rounded.iter().map(|&(_, p)| p / total).collect(),
            doppler: max_doppler_hz / sample_rate,
        })
    }

    /// Delays of the taps on the sampling grid (samples).
    pub fn delays(&self) -> &[usize] {
        &self.delays
    }

    /// Powers of the taps (linear, unit sum).
    pub fn powers(&self) -> &[f64] {
        &self.powers
    }

    /// The maximum Doppler shift in cycles per sample.
    pub fn doppler(&self) -> f64 {
        self.doppler
    }

    /// The number of Doppler nodes of the sum of sinusoids for pulses spanning `span`
    /// samples: $64 + 2 x_{\max}$, $x_{\max} = 2\pi \nu_{\max} \Delta t \cdot$ `span`
    /// (1 for a static channel).
    fn nodes(&self, span: usize) -> usize {
        if self.doppler == 0.0 {
            1
        } else {
            64 + (2.0 * std::f64::consts::TAU * self.doppler * span as f64).ceil() as usize
        }
    }

    /// The factor $\boldsymbol\Omega$ of the module: one row per transmit pulse of `tx`,
    /// the columns the (tap, Doppler node) pairs, for the receive pulse `rx`.
    fn omega(&self, rx: &Pulse, tx: &[Pulse]) -> Vec<Vec<C64>> {
        let (rx_start, rx_values) = rx;
        let rx_end = rx_start + rx_values.len();
        let span = tx.iter().map(|(s, v)| s + v.len()).max().unwrap_or(0).max(rx_end);
        let q = self.nodes(span);
        let omegas: Vec<f64> = (0..q)
            .map(|k| {
                let theta = std::f64::consts::TAU * (k as f64 + 0.5) / q as f64;
                std::f64::consts::TAU * self.doppler * theta.cos()
            })
            .collect();
        let scale: Vec<f64> = self.powers.iter().map(|p| (p / q as f64).sqrt()).collect();
        tx.iter()
            .map(|(g_start, g)| {
                let mut row = Vec::with_capacity(self.delays.len() * q);
                for (m, &d) in self.delays.iter().enumerate() {
                    // the product sequence c[n] = conj(q_i[n]) g_j[n - d] on the overlap
                    let start = (*rx_start).max(g_start + d);
                    let end = rx_end.min(g_start + d + g.len());
                    if start >= end {
                        row.extend(std::iter::repeat_n(C64::new(0.0, 0.0), q));
                        continue;
                    }
                    let c: Vec<C64> =
                        (start..end).map(|n| rx_values[n - rx_start].conj() * g[n - d - g_start]).collect();
                    for &w in &omegas {
                        let mut acc = C64::new(0.0, 0.0);
                        // the phase of the sample `start`, then one rotation per sample
                        let step = cis(w);
                        let mut rot = cis(w * start as f64);
                        for v in &c {
                            acc += v * rot;
                            rot *= step;
                        }
                        row.push(acc * scale[m]);
                    }
                }
                row
            })
            .collect()
    }

    /// The mean power $\[\boldsymbol\Gamma\]_{i,i} = \mathrm{E}|\mathbf{q}^{H}\mathbf{H}\mathbf{g}|^{2}$
    /// of the one-tap coefficient of the pulse `tx` received on `rx`, for a channel of unit
    /// power.
    pub fn coefficient_power(&self, rx: &Pulse, tx: &Pulse) -> f64 {
        self.omega(rx, std::slice::from_ref(tx))[0].iter().map(C64::norm_sqr).sum()
    }

    /// The SIR (linear) of the symbol carried by `tx`, received on `rx`, when the transmit
    /// pulses form a complete orthonormal basis (OFDM without prefix) or, for real symbols,
    /// a complete real-orthonormal frame (FBMC-OQAM on its whole lattice):
    /// $\[\boldsymbol\Gamma\]_{i,i} / (\|\mathbf{q}\|^{2} - \[\boldsymbol\Gamma\]_{i,i})$
    /// (see the module).
    pub fn sir_complete(&self, rx: &Pulse, tx: &Pulse) -> f64 {
        let signal = self.coefficient_power(rx, tx);
        let total: f64 = rx.1.iter().map(C64::norm_sqr).sum();
        let interference = (total - signal).max(0.0);
        if interference == 0.0 { f64::INFINITY } else { signal / interference }
    }

    /// The SIR (linear) of the complex symbol carried by `tx[wanted]`, received on `rx`
    /// with a one-tap equaliser, the other pulses of `tx` interfering (eq. (35)).
    ///
    /// # Panics
    /// If `wanted` is not an index of `tx`.
    pub fn sir_qam(&self, rx: &Pulse, tx: &[Pulse], wanted: usize) -> f64 {
        let omega = self.omega(rx, tx);
        let gamma: Vec<f64> = omega.iter().map(|row| row.iter().map(C64::norm_sqr).sum()).collect();
        ratio(&gamma, wanted)
    }

    /// The SIR (linear) of the real symbol carried by `tx[wanted]`, received on `rx`, the
    /// one tap's phase removed and the real part kept (eqs. (38)–(40)).
    ///
    /// # Panics
    /// If `wanted` is not an index of `tx`.
    pub fn sir_oqam(&self, rx: &Pulse, tx: &[Pulse], wanted: usize) -> f64 {
        let omega = self.omega(rx, tx);
        let phases: Vec<C64> = omega[wanted]
            .iter()
            .map(|v| if v.norm() > 0.0 { v.conj() / v.norm() } else { C64::new(1.0, 0.0) })
            .collect();
        let gamma: Vec<f64> =
            omega.iter().map(|row| row.iter().zip(&phases).map(|(v, p)| (v * p).re.powi(2)).sum()).collect();
        ratio(&gamma, wanted)
    }
}

/// $\Gamma_{ii} / (\mathrm{tr}\,\Gamma - \Gamma_{ii})$ from the diagonal of $\Gamma$.
fn ratio(gamma: &[f64], wanted: usize) -> f64 {
    let total: f64 = gamma.iter().sum();
    let signal = gamma[wanted];
    let interference = (total - signal).max(0.0);
    if interference == 0.0 { f64::INFINITY } else { signal / interference }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fbmc::{Fbmc, Prototype};
    use crate::ofdm::Ofdm;
    use crate::ofdm::budget::jakes_ici_sir;

    /// Flat Jakes fading on CP-OFDM: the matrix SIR is the closed form of eq. (42) at
    /// $f_{D} T_{u}$ (the FFT of the window spans $T_{u}$).
    #[test]
    fn flat_jakes_fading_on_cp_ofdm_gives_the_closed_form_sir() {
        let n = 64;
        let ofdm = Ofdm { n_cp: 4, oversampling: 1, ..Ofdm::new(n, n).unwrap() };
        for fd_tu in [0.02, 0.1] {
            // the Doppler per sample: the useful symbol spans fft_len samples
            let ch = DoublySelective::new(&[(0.0, 0.0)], 1.0, fd_tu / ofdm.fft_len() as f64).unwrap();
            let tx: Vec<Pulse> =
                (0..3).flat_map(|k| (0..n).map(move |i| (i, k))).map(|(i, k)| ofdm.pulse(i, k)).collect();
            let sir = ch.sir_qam(&ofdm.window(n / 2, 1), &tx, n + n / 2);
            let want = jakes_ici_sir(fd_tu);
            assert!((sir / want - 1.0).abs() < 2e-3, "{fd_tu}: {sir} against {want}");
        }
    }

    /// A static channel inside the prefix leaves CP-OFDM free of interference and
    /// FBMC/OQAM at its intrinsic level (PHYDYAS: a residual of a few 1e-4 per neighbour).
    #[test]
    fn static_channels_give_no_interference_beyond_the_prototype() {
        let ch = DoublySelective::new(&[(0.0, 0.0), (2.0, -3.0), (5.0, -10.0)], 1.0, 0.0).unwrap();
        let ofdm = Ofdm { n_cp: 6, oversampling: 1, ..Ofdm::new(32, 32).unwrap() };
        let tx: Vec<Pulse> = (0..3).flat_map(|k| (0..32).map(move |i| (i, k))).map(|(i, k)| ofdm.pulse(i, k)).collect();
        assert!(ch.sir_qam(&ofdm.window(3, 1), &tx, 32 + 3) > 1e20);
        let flat = DoublySelective::new(&[(0.0, 0.0)], 1.0, 0.0).unwrap();
        let fbmc = Fbmc::new(16, 4).unwrap();
        let tx: Vec<Pulse> =
            (0..17).flat_map(|n| (0..16).map(move |m| (m, n))).map(|(m, n)| fbmc.pulse(m, n)).collect();
        let sir = flat.sir_oqam(&fbmc.pulse(8, 8), &tx, 8 * 16 + 8);
        assert!(sir > 1e5, "{sir}");
        // the complex SIR of the OQAM lattice is about 1: the imaginary interference
        let complex = flat.sir_qam(&fbmc.pulse(8, 8), &tx, 8 * 16 + 8);
        assert!(complex > 0.5 && complex < 2.0, "{complex}");
    }

    /// With every subcarrier of OFDM, or the whole OQAM lattice, the sums of eqs. (35) and
    /// (40) equal the Parseval form of `sir_complete`.
    #[test]
    fn the_complete_basis_gives_the_sum_over_every_interferer() {
        let ch = DoublySelective::new(&[(0.0, 0.0), (3.0, -4.0), (7.0, -9.0)], 1.0, 2e-3).unwrap();
        let n = 32;
        let ofdm = Ofdm { oversampling: 1, n_cp: 0, ..Ofdm::new(n, n).unwrap() };
        let tx: Vec<Pulse> = (0..3).flat_map(|k| (0..n).map(move |i| (i, k))).map(|(i, k)| ofdm.pulse(i, k)).collect();
        let rx = ofdm.window(n / 2, 1);
        let full = ch.sir_qam(&rx, &tx, n + n / 2);
        let parseval = ch.sir_complete(&rx, &tx[n + n / 2]);
        assert!((full / parseval - 1.0).abs() < 1e-9, "{full} {parseval}");
        let fbmc = Fbmc::with_prototype(16, 4, Prototype::Hermite).unwrap();
        let tx: Vec<Pulse> =
            (0..17).flat_map(|k| (0..16).map(move |m| (m, k))).map(|(m, k)| fbmc.pulse(m, k)).collect();
        let rx = fbmc.pulse(8, 8);
        let full = ch.sir_oqam(&rx, &tx, 8 * 16 + 8);
        let parseval = ch.sir_complete(&rx, &rx);
        assert!((full / parseval - 1.0).abs() < 1e-4, "{full} {parseval}");
    }

    #[test]
    fn taps_are_rounded_merged_and_normalised() {
        let ch = DoublySelective::new(&[(0.0, 0.0), (0.4e-6, -3.0), (0.6e-6, -3.0)], 2e6, 100.0).unwrap();
        assert_eq!(ch.delays(), &[0, 1]);
        assert!((ch.powers().iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((ch.powers()[1] / ch.powers()[0] - 10f64.powf(-0.3) * 2.0).abs() < 1e-9);
        assert!((ch.doppler() - 5e-5).abs() < 1e-18);
        assert!(DoublySelective::new(&[], 1.0, 0.0).is_err());
        assert!(DoublySelective::new(&[(0.0, 0.0)], 1.0, -1.0).is_err());
    }
}
