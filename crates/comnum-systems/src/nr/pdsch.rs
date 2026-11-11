//! One slot of a single-layer PDSCH on its resource grid (TS 38.211 Sec. 7.3.1), and a
//! receiver that estimates the precoded channel on the DM-RS.
//!
//! # Transmitter
//!
//! The $G$ channel bits of the slot are scrambled by the Gold sequence of
//! $c_{\mathrm{init}} = n_{\mathrm{RNTI}} 2^{15} + q 2^{14} + n_{\mathrm{ID}}$ (Sec. 7.3.1.1,
//! $q = 0$), mapped by [`super::modulation::constellation`], and placed frequency first
//! on the REs of the allocation that carry neither DM-RS nor a reserved RE (Sec.
//! 7.3.1.5): every subcarrier of the PDSCH symbols, the DM-RS symbols included on the
//! subcarriers of the CDM groups with data (one CDM group without data: the odd
//! subcarriers of configuration type 1). The DM-RS of port 1000 (`PdschDmrs`) and the data
//! are precoded by the same $W_{n}$ in every precoding resource block group (PRG, Sec.
//! 5.1.2.3 of TS 38.214: groups of $P$ PRBs counted from common resource block 0), the
//! $n$ of each PRG chosen by the caller (TS 38.101-4 draws it at random): for two ports,
//! $W_{n} = \frac{1}{\sqrt{2}}(1, j^{n})^{T}$ (TS 38.214 Table 5.2.2.2.1-1, one layer).
//! The DM-RS has the energy of the data (one CDM group without data, Table 4.1-1).
//!
//! # Receiver
//!
//! The receiver does not know the precoder: in every PRG it estimates the channel
//! including the precoder, per receive antenna, from the LS estimates
//! $y\\, x^{*}$ on the DM-RS REs, by the 2-D Wiener interpolator of
//! `comnum_waveform::ofdm::wiener` designed for a uniform delay spectrum over
//! `max_delay_s` and a Jakes Doppler spectrum of `max_doppler_hz` (a robust design: the
//! receiver knows bounds, not the channel), combines the antennas by maximum ratio, and
//! computes the LLRs with the effective noise of every RE, then descrambles them.
//!
//! # References
//!
//! 1. 3GPP TS 38.211 V18.10.0, Sec. 7.3.1 (PDSCH), 7.4.1.1 (DM-RS).
//! 2. 3GPP TS 38.214 V19.5.0, Sec. 5.1.2.3 (PRB bundling), Table 4.1-1, Table 5.2.2.2.1-1.
//! 3. P. Hoeher, S. Kaiser and P. Robertson, "Two-dimensional pilot-symbol-aided channel
//!    estimation by Wiener filtering," *Proc. IEEE ICASSP*, 1997.

use super::modulation::constellation;
use comnum_core::traits::Jacobian;
use comnum_core::{Result, ensure, ensure_shape};
use comnum_fec::scrambling::Scrambler;
use comnum_mapping::constellation::Constellation;
use comnum_math::C64;
use comnum_waveform::ofdm::nr::{PdschDmrs, SUBCARRIERS_PER_RB};
use comnum_waveform::ofdm::wiener::{Correlation2d, DelaySpectrum, DopplerSpectrum, Wiener2d, WienerPlan};
use std::collections::HashMap;

/// Symbols of a slot (normal cyclic prefix).
pub const SYMBOLS_PER_SLOT: usize = 14;

/// A single-layer PDSCH and its slot (see the module).
#[derive(Clone, Debug)]
pub struct Pdsch {
    /// Resource blocks of the carrier grid (52 for 10 MHz at 15 kHz).
    pub n_rb_grid: usize,
    /// First PRB of the allocation (common resource block index).
    pub prb_start: usize,
    /// PRBs of the allocation.
    pub n_prb: usize,
    /// First PDSCH symbol $S$.
    pub first_symbol: usize,
    /// PDSCH symbols $L$.
    pub n_symbols: usize,
    /// The DM-RS (port 1000, configuration type 1, one CDM group without data).
    pub dmrs: PdschDmrs,
    /// Transmit antenna ports, 1 or 2.
    pub n_ports: usize,
    /// PRBs per precoding resource block group $P$ (2 or 4).
    pub prg_size: usize,
    /// Bits per symbol $Q_{m}$.
    pub qm: usize,
    /// $n_{\mathrm{RNTI}}$ of the scrambling.
    pub rnti: u32,
    /// $n_{\mathrm{ID}}$ of the scrambling.
    pub n_id: u32,
    /// Subcarrier spacing (Hz).
    pub subcarrier_spacing_hz: f64,
}

/// The receiver of a [`Pdsch`]: the robust bounds of its channel estimator.
#[derive(Clone, Debug)]
pub struct PdschReceiver {
    /// The largest delay the estimator expects (s), e.g. the cyclic prefix.
    pub max_delay_s: f64,
    /// The largest Doppler shift the estimator expects (Hz).
    pub max_doppler_hz: f64,
    /// The interpolators of every PRG width and noise level, built once.
    plans: HashMap<(usize, u64), (WienerPlan, Vec<f64>)>,
}

impl PdschReceiver {
    /// A receiver whose estimator expects delays up to `max_delay_s` and Doppler shifts up
    /// to `max_doppler_hz`.
    pub fn new(max_delay_s: f64, max_doppler_hz: f64) -> PdschReceiver {
        PdschReceiver { max_delay_s, max_doppler_hz, plans: HashMap::new() }
    }
}

impl Pdsch {
    fn k(&self) -> usize {
        SUBCARRIERS_PER_RB * self.n_rb_grid
    }

    /// Check the parameters (every method that maps or receives does).
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for an allocation outside the grid or the slot, a
    /// number of ports other than 1 or 2, a DM-RS configuration other than type 1 or a
    /// PRG without PRB.
    pub fn check(&self) -> Result<()> {
        ensure(self.n_prb >= 1 && self.prb_start.checked_add(self.n_prb).is_some_and(|e| e <= self.n_rb_grid), || {
            format!("the allocation {}+{} leaves the grid of {} RBs", self.prb_start, self.n_prb, self.n_rb_grid)
        })?;
        ensure(self.first_symbol + self.n_symbols <= SYMBOLS_PER_SLOT && self.n_symbols >= 1, || {
            "the PDSCH symbols leave the slot".into()
        })?;
        ensure(self.n_ports == 1 || self.n_ports == 2, || format!("1 or 2 ports, got {}", self.n_ports))?;
        ensure(self.dmrs.config_type == 1, || "DM-RS configuration type 1 only".into())?;
        ensure(self.prg_size >= 1, || "a PRG of at least one PRB".into())
    }

    /// The DM-RS symbols of the slot.
    ///
    /// # Errors
    /// Those of `PdschDmrs::symbols`.
    pub fn dmrs_symbols(&self) -> Result<Vec<usize>> {
        self.dmrs.symbols()
    }

    /// The data REs `(symbol, subcarrier)` of the slot in their mapping order (frequency
    /// first), the REs of `reserved` excluded.
    ///
    /// # Errors
    /// Those of [`Pdsch::dmrs_symbols`] and of the configuration.
    pub fn data_res(&self, reserved: &[(usize, usize)]) -> Result<Vec<(usize, usize)>> {
        self.check()?;
        let dmrs = self.dmrs_symbols()?;
        let k0 = self.prb_start * SUBCARRIERS_PER_RB;
        let mut out = Vec::new();
        for l in self.first_symbol..self.first_symbol + self.n_symbols {
            let on_dmrs = dmrs.contains(&l);
            for k in k0..k0 + self.n_prb * SUBCARRIERS_PER_RB {
                if (on_dmrs && k % 2 == 0) || reserved.contains(&(l, k)) {
                    continue;
                }
                out.push((l, k));
            }
        }
        Ok(out)
    }

    /// The channel bits $G$ of the slot.
    ///
    /// # Errors
    /// Those of [`Pdsch::data_res`].
    pub fn g(&self, reserved: &[(usize, usize)]) -> Result<usize> {
        Ok(self.data_res(reserved)?.len() * self.qm)
    }

    /// The PRGs of the allocation: their first PRB and their width.
    fn prgs(&self) -> Vec<(usize, usize)> {
        let p = self.prg_size;
        let (a, b) = (self.prb_start, self.prb_start + self.n_prb);
        let mut out = Vec::new();
        let mut s = a;
        while s < b {
            let end = ((s / p + 1) * p).min(b);
            out.push((s, end - s));
            s = end;
        }
        out
    }

    /// The number of PRGs of the allocation (the precoders [`Pdsch::transmit`] takes).
    pub fn n_prgs(&self) -> usize {
        if self.prg_size == 0 { 0 } else { self.prgs().len() }
    }

    fn precoder(&self, n: usize) -> Vec<C64> {
        if self.n_ports == 1 {
            return vec![C64::new(1.0, 0.0)];
        }
        let jn = [C64::new(1.0, 0.0), C64::new(0.0, 1.0), C64::new(-1.0, 0.0), C64::new(0.0, -1.0)][n % 4];
        let s = std::f64::consts::FRAC_1_SQRT_2;
        vec![C64::new(s, 0.0), jn * s]
    }

    fn scrambler(&self) -> Scrambler {
        Scrambler { c_init: (self.rnti << 15) + self.n_id }
    }

    /// The grids `[port][l * K + k]` of the slot carrying the $G$ channel bits `bits`,
    /// precoded in PRG $i$ by $W_{n_{i}}$, `precoders[i]` $= n_{i}$ (0 to 3; ignored with one
    /// port), the REs of `reserved` left empty.
    ///
    /// # Errors
    /// [`comnum_core::Error::Shape`] unless `bits` has $G$ values and `precoders` one per
    /// PRG; those of [`Pdsch::data_res`].
    pub fn transmit(&self, bits: &[u8], precoders: &[usize], reserved: &[(usize, usize)]) -> Result<Vec<Vec<C64>>> {
        let res = self.data_res(reserved)?;
        ensure_shape(bits.len() == res.len() * self.qm, || {
            format!("{} bits for G = {}", bits.len(), res.len() * self.qm)
        })?;
        let prgs = self.prgs();
        ensure_shape(precoders.len() == prgs.len(), || {
            format!("{} precoders for {} PRGs", precoders.len(), prgs.len())
        })?;
        let c = constellation(self.qm)?;
        let symbols = c.map_bits(&self.scrambler().scramble(bits))?;
        let k = self.k();
        let mut grid = vec![vec![C64::new(0.0, 0.0); SYMBOLS_PER_SLOT * k]; self.n_ports];
        let w_of = |sc: usize| -> Vec<C64> {
            let prb = sc / SUBCARRIERS_PER_RB;
            let i = prgs.iter().position(|&(s, n)| prb >= s && prb < s + n).unwrap_or(0);
            self.precoder(precoders[i])
        };
        for (&(l, sc), x) in res.iter().zip(&symbols) {
            for (p, w) in w_of(sc).iter().enumerate() {
                grid[p][l * k + sc] = x * w;
            }
        }
        for l in self.dmrs_symbols()? {
            let values = self.dmrs.values(self.n_rb_grid, l);
            for (m, sc) in self.dmrs.subcarriers(self.n_rb_grid).into_iter().enumerate() {
                let prb = sc / SUBCARRIERS_PER_RB;
                if prb < self.prb_start || prb >= self.prb_start + self.n_prb {
                    continue;
                }
                for (p, w) in w_of(sc).iter().enumerate() {
                    grid[p][l * k + sc] = values[m] * w;
                }
            }
        }
        Ok(grid)
    }

    /// The $G$ descrambled LLRs (positive for a 0) of the received grids `rx`
    /// (`[antenna][l * K + k]`) at the noise variance `noise_var` per RE and antenna.
    ///
    /// # Errors
    /// [`comnum_core::Error::Shape`] for grids of another size; those of the estimator and
    /// of [`Pdsch::data_res`].
    pub fn receive(
        &self,
        rx: &[Vec<C64>],
        noise_var: f64,
        receiver: &mut PdschReceiver,
        reserved: &[(usize, usize)],
    ) -> Result<Vec<f64>> {
        let k = self.k();
        ensure_shape(!rx.is_empty() && rx.iter().all(|g| g.len() == SYMBOLS_PER_SLOT * k), || {
            format!("the received grids must have {} values", SYMBOLS_PER_SLOT * k)
        })?;
        ensure(noise_var > 0.0, || format!("the noise variance must be positive, got {noise_var}"))?;
        let res = self.data_res(reserved)?;
        let dmrs_symbols = self.dmrs_symbols()?;
        let dmrs_sc = self.dmrs.subcarriers(self.n_rb_grid);
        let values: Vec<Vec<C64>> = dmrs_symbols.iter().map(|&l| self.dmrs.values(self.n_rb_grid, l)).collect();
        // h[antenna][l * K + k] on the data REs, and the estimation MSE
        let mut h = vec![vec![C64::new(0.0, 0.0); SYMBOLS_PER_SLOT * k]; rx.len()];
        let mut mse = vec![0.0; SYMBOLS_PER_SLOT * k];
        for (s, n) in self.prgs() {
            let k_lo = s * SUBCARRIERS_PER_RB;
            let width = n * SUBCARRIERS_PER_RB;
            let pilots: Vec<(usize, usize)> = dmrs_symbols
                .iter()
                .flat_map(|&l| (0..width).filter(|i| (k_lo + i) % 2 == 0).map(move |i| (l, i)))
                .collect();
            let targets: Vec<(usize, usize)> =
                res.iter().filter(|&&(_, sc)| sc >= k_lo && sc < k_lo + width).map(|&(l, sc)| (l, sc - k_lo)).collect();
            let key = (n, noise_var.to_bits());
            if !receiver.plans.contains_key(&key) {
                let ts = 1.0 / self.subcarrier_spacing_hz * 15.0 / 14.0;
                let corr = Correlation2d {
                    fd_ts: receiver.max_doppler_hz * ts,
                    doppler: DopplerSpectrum::Jakes,
                    delay: DelaySpectrum::Uniform { tau_df: receiver.max_delay_s * self.subcarrier_spacing_hz },
                };
                let wiener = Wiener2d::new(pilots.clone(), corr, noise_var, pilots.len())?;
                let errors = wiener.mse(&targets)?;
                receiver.plans.insert(key, (wiener.plan(&targets)?, errors));
            }
            let (plan, errors) = &receiver.plans[&key];
            for (a, y) in rx.iter().enumerate() {
                let ls: Vec<C64> = pilots
                    .iter()
                    .map(|&(l, i)| {
                        let sc = k_lo + i;
                        let d = dmrs_symbols.iter().position(|&x| x == l).unwrap_or(0);
                        let m = dmrs_sc.iter().position(|&x| x == sc).unwrap_or(0);
                        y[l * k + sc] * values[d][m].conj()
                    })
                    .collect();
                for (&(l, i), est) in targets.iter().zip(plan.apply(&ls)?) {
                    h[a][l * k + k_lo + i] = est;
                }
            }
            for (&(l, i), e) in targets.iter().zip(errors) {
                mse[l * k + k_lo + i] = *e;
            }
        }
        self.combine(rx, &h, &mse, noise_var, &res)
    }

    /// [`Pdsch::receive`] with the precoded channel `channel` (`[antenna][l * K + k]`, the
    /// channel times the precoder of the RE) known: the genie-aided receiver, the bound of
    /// the estimated one.
    ///
    /// # Errors
    /// [`comnum_core::Error::Shape`] for grids of another size; those of
    /// [`Pdsch::data_res`].
    pub fn receive_known(
        &self,
        rx: &[Vec<C64>],
        channel: &[Vec<C64>],
        noise_var: f64,
        reserved: &[(usize, usize)],
    ) -> Result<Vec<f64>> {
        let size = SYMBOLS_PER_SLOT * self.k();
        ensure_shape(!rx.is_empty() && rx.len() == channel.len(), || "one channel grid per receive antenna".into())?;
        ensure_shape(rx.iter().chain(channel).all(|g| g.len() == size), || {
            format!("the grids must have {size} values")
        })?;
        ensure(noise_var > 0.0, || format!("the noise variance must be positive, got {noise_var}"))?;
        let res = self.data_res(reserved)?;
        self.combine(rx, channel, &vec![0.0; size], noise_var, &res)
    }

    /// Maximum ratio combining of the antennas with the channel `h` of estimation error
    /// `mse` per antenna, and the descrambled LLRs: $z = \sum_{a} \hat{h}_{a}^{*} y_{a} / g$,
    /// $g = \sum_{a} |\hat{h}_{a}|^{2}$, of noise $(\sigma^{2} + \mathrm{MSE})/g$ (the error
    /// $h_{a} - \hat{h}_{a}$ times the unit-energy symbol adds its variance to the noise).
    fn combine(
        &self,
        rx: &[Vec<C64>],
        h: &[Vec<C64>],
        mse: &[f64],
        noise_var: f64,
        res: &[(usize, usize)],
    ) -> Result<Vec<f64>> {
        let k = self.k();
        let (mut z, mut var) = (Vec::with_capacity(res.len()), Vec::with_capacity(res.len()));
        for &(l, sc) in res {
            let i = l * k + sc;
            let g: f64 = h.iter().map(|ha| ha[i].norm_sqr()).sum::<f64>().max(1e-12);
            let num: C64 = h.iter().zip(rx).map(|(ha, y)| ha[i].conj() * y[i]).sum();
            z.push(num / g);
            var.push((noise_var + mse[i]) / g);
        }
        let c: Constellation = constellation(self.qm)?;
        let llr = c.llr_with_variances(&z, &var, None, Jacobian::Exact)?;
        Ok(self.scrambler().descramble_llr(&llr))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code: a failure is a test failure")]
    use super::*;

    fn frc_1_1_2() -> Pdsch {
        Pdsch {
            n_rb_grid: 52,
            prb_start: 23,
            n_prb: 6,
            first_symbol: 2,
            n_symbols: 12,
            dmrs: PdschDmrs { config_type: 1, type_a_position: 2, additional_position: 1, n_id: 0, slot: 1 },
            n_ports: 2,
            prg_size: 2,
            qm: 2,
            rnti: 1,
            n_id: 0,
            subcarrier_spacing_hz: 15e3,
        }
    }

    #[test]
    fn the_slot_of_r_pdsch_1_1_2_has_1584_channel_bits_and_four_prgs() {
        let p = frc_1_1_2();
        assert_eq!(p.g(&[]).unwrap(), 1584);
        assert_eq!(p.prgs(), vec![(23, 1), (24, 2), (26, 2), (28, 1)]);
        // 6 TRS REs per PRB removed in slots 10 and 11: 1512 bits
        let trs: Vec<(usize, usize)> =
            [6, 10].iter().flat_map(|&l| (23 * 12..29 * 12).step_by(4).map(move |k| (l, k))).collect();
        assert_eq!(p.g(&trs).unwrap(), 1512);
    }

    #[test]
    fn a_flat_channel_without_noise_gives_the_bits_back() {
        let p = frc_1_1_2();
        let g = p.g(&[]).unwrap();
        let bits: Vec<u8> = (0..g).map(|i| ((i * 7) % 3 == 0) as u8).collect();
        let grids = p.transmit(&bits, &[0, 1, 2, 3], &[]).unwrap();
        // two antennas, a different flat channel per link
        let hs = [[C64::new(0.8, 0.3), C64::new(-0.2, 0.5)], [C64::new(0.1, -0.9), C64::new(0.6, 0.2)]];
        let rx: Vec<Vec<C64>> = hs
            .iter()
            .map(|hr| (0..grids[0].len()).map(|i| hr[0] * grids[0][i] + hr[1] * grids[1][i]).collect())
            .collect();
        let mut receiver = PdschReceiver::new(4.7e-6, 100.0);
        let llr = p.receive(&rx, 1e-3, &mut receiver, &[]).unwrap();
        let decided: Vec<u8> = llr.iter().map(|v| u8::from(*v < 0.0)).collect();
        assert_eq!(decided, bits);
    }
}
