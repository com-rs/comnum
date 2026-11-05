//! Two-dimensional Wiener interpolation of pilots in time and frequency (Hoeher,
//! Kaiser & Robertson 1997).
//!
//! The channel $h_{l,k}$ of symbol $l$ and subcarrier $k$ is observed at the pilots
//! $\mathcal{P}$, $\tilde{h}_{l',k'} = h_{l',k'} + n_{l',k'}$ (the LS estimates, noise of
//! variance $\sigma^{2}/E_{p}$), and every position is estimated by a linear filter of
//! the $N_{\mathrm{tap}}$ pilots $\mathcal{T}(l, k) \subset \mathcal{P}$ nearest to it,
//! their eqs. (2) and (9),
//!
//! $$
//! \hat{h}_{l,k} = \underline{w}^{T} \underline{\tilde{h}}, \qquad
//! \underline{w}^{T} = \underline{\theta}^{T}\, \Phi^{-1}, \qquad
//! J_{o}(l, k) = 1 - \underline{\theta}^{T} \Phi^{-1} \underline{\theta}^{*},
//! $$
//!
//! with $\theta = E[h_{l,k} \tilde{h}^{*}]$ over the taps and $\Phi = E[\tilde{h}
//! \tilde{h}^{H}] = R + \frac{\sigma^{2}}{E_{p}} I$ (eqs. (6)–(7), (12)): the linear MMSE
//! of [`comnum_dsp::estimation::lmmse`] on the taps. For a WSSUS channel the correlation
//! is separable (eq. (15)),
//!
//! $$
//! R(\Delta l, \Delta k) = r_{t}(\Delta l)\, r_{f}(\Delta k),
//! $$
//!
//! the time correlation $r_{t}$ of the Doppler spectrum at the symbol period $T_{s}$
//! (uniform: $\mathrm{si}(2\pi f_{D} T_{s} \Delta l)$; Jakes: $J_{0}(2\pi f_{D} T_{s}
//! \Delta l)$) and the frequency correlation $r_{f}$ of the delay power spectrum at the
//! subcarrier spacing $\Delta F$ (uniform on $[-\tau_{\max}, \tau_{\max}]$:
//! $\mathrm{si}(2\pi \tau_{\max} \Delta F \Delta k)$, or a [`PowerDelayProfile`]). The
//! taps are the pilots nearest in the weighted distance $|\Delta l| f_{D} T_{s} +
//! |\Delta k| \tau \Delta F$ (their Sec. 5, method B), ties broken by the smaller time
//! distance then the pilot order. With pilots in one symbol only, the filter is the 1-D
//! Wiener filter in frequency of their Fig. 2.
//!
//! The sampling theorem of the grid (their eq. (17)): pilots every $N_{t}$ symbols and
//! $N_{f}$ subcarriers resolve the channel if $f_{D} T_{s} N_{t} \le 1/2$ and $\tau_{\max}
//! \Delta F N_{f} \le 1/2$ (two-sided spreads), $1/4$ for a twofold oversampling.
//!
//! # Examples
//! ```
//! use comnum_waveform::ofdm::wiener::{Correlation2d, DelaySpectrum, DopplerSpectrum, Wiener2d};
//! // pilots every 3 symbols and 6 subcarriers on a block of 12 x 24
//! let pilots: Vec<(usize, usize)> =
//!     (0..12).step_by(3).flat_map(|l| (0..24).step_by(6).map(move |k| (l, k))).collect();
//! let corr = Correlation2d { fd_ts: 0.0832, doppler: DopplerSpectrum::Uniform, delay: DelaySpectrum::Uniform { tau_df: 0.0417 } };
//! let w = Wiener2d::new(pilots, corr, 0.01, 10)?;
//! let mse = w.mse(&[(1, 2), (0, 0)])?;
//! assert!(mse[1] < 0.01 && mse[0] < 0.05);
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # The robust estimator of Li, Cimini & Sollenberger (1998)
//!
//! An estimator designed for the worst channel of maximum Doppler $f_{\max}$ and delay
//! spread $t_{\max}$, a flat band-limited Doppler spectrum and a uniform delay profile
//! on the $K_{o} = \lfloor K t_{\max}/T_{s} \rfloor + 1$ taps of the DFT grid between 0 and
//! $t_{\max}$ (their eq. (56) prints $\lceil K t_{\max}/T_{s} \rceil$, one tap fewer when
//! $K t_{\max}/T_{s}$ is an integer; their figures follow the count of the taps), filters every kept tap in time with the temporal Wiener filter of the
//! correlation $\mathrm{sinc}(n\omega_{d}/\pi)$, $\omega_{d} = 2\pi f_{\max} T_{f}$ (block
//! period $T_{f}$). Its MSE, normalised to the channel power, is for an infinite filter
//! (eq. (57))
//!
//! $$
//! \overline{\mathrm{MMSE}} = \frac{K_{o}\rho}{K}\left(1 - \left[\frac{\pi K}{K_{o} \rho
//! \omega_{d}} + 1\right]^{-\omega_{d}/\pi}\right),
//! $$
//!
//! $\rho$ the noise to signal ratio, and for $L$ taps $\overline{\mathrm{MSE}} = K_{o} c_{0}
//! \rho / K$ with $(c_{0}, \dots, c_{L-1})^{T} = (\bar{R}_{t} + \frac{K_{o}\rho}{K}
//! I)^{-1} \bar{r}_{t}$ (eqs. (69)–(74)): [`RobustDesign::mse`] and [`RobustDesign::fir`].
//! It bounds the MSE of every channel within the design (their eqs. (34)–(36)).
//!
//! # References
//!
//! 1. P. Hoeher, S. Kaiser and P. Robertson, "Two-dimensional pilot-symbol-aided channel
//!    estimation by Wiener filtering," *Proc. IEEE ICASSP*, vol. 3, 1845–1848, 1997.
//! 2. Y. Li, L. J. Cimini and N. R. Sollenberger, "Robust channel estimation for OFDM
//!    systems with rapid dispersive fading channels," *IEEE Trans. Commun.*, 46(7),
//!    902–915, 1998.

use super::estimation::PowerDelayProfile;
use comnum_core::{Result, ensure, ensure_shape};
use comnum_dsp::estimation::{Lmmse, lmmse_mse};
use comnum_math::linalg::CMat;
use comnum_math::{C64, bessel_j0, sinc};
use std::f64::consts::PI;

/// The Doppler spectrum of every path, of maximum Doppler $f_{D}$.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DopplerSpectrum {
    /// Uniform on $[-f_{D}, f_{D}]$: $r_{t}(\Delta l) = \mathrm{si}(2\pi f_{D} T_{s} \Delta l)$
    /// (Hoeher et al. 1997, eq. (15)).
    Uniform,
    /// Jakes (Clarke): $r_{t}(\Delta l) = J_{0}(2\pi f_{D} T_{s} \Delta l)$.
    Jakes,
}

/// The delay power spectrum of the channel.
#[derive(Clone, Debug, PartialEq)]
pub enum DelaySpectrum {
    /// Uniform on $[-\tau_{\max}, \tau_{\max}]$, `tau_df` = $\tau_{\max} \Delta F$:
    /// $r_{f}(\Delta k) = \mathrm{si}(2\pi \tau_{\max} \Delta F \Delta k)$ (eq. (15)).
    Uniform {
        /// The one-sided maximum delay over the useful duration, $\tau_{\max} \Delta F$.
        tau_df: f64,
    },
    /// A delay profile on the sampling grid of the symbol
    /// ([`PowerDelayProfile::correlation`]).
    Profile(PowerDelayProfile),
}

/// The separable time-frequency correlation $r_{t}(\Delta l)\, r_{f}(\Delta k)$ of a
/// WSSUS channel of unit power (see the module).
#[derive(Clone, Debug, PartialEq)]
pub struct Correlation2d {
    /// The maximum Doppler over the symbol rate, $f_{D} T_{s}$.
    pub fd_ts: f64,
    /// The Doppler spectrum.
    pub doppler: DopplerSpectrum,
    /// The delay power spectrum.
    pub delay: DelaySpectrum,
}

impl Correlation2d {
    /// $E[h_{l + \Delta l, k + \Delta k}\, h_{l,k}^{*}]$.
    pub fn at(&self, dl: isize, dk: isize) -> C64 {
        let x = 2.0 * self.fd_ts * dl as f64;
        let rt = match self.doppler {
            DopplerSpectrum::Uniform => sinc(x),
            DopplerSpectrum::Jakes => bessel_j0(PI * x),
        };
        let rf = match &self.delay {
            DelaySpectrum::Uniform { tau_df } => C64::new(sinc(2.0 * tau_df * dk as f64), 0.0),
            DelaySpectrum::Profile(p) => p.correlation(dk as f64),
        };
        rf * rt
    }

    /// The weight of a subcarrier in the distance of the tap selection: $\tau_{\max}
    /// \Delta F$, or the largest delay of a profile over its FFT size.
    fn frequency_weight(&self) -> f64 {
        match &self.delay {
            DelaySpectrum::Uniform { tau_df } => *tau_df,
            DelaySpectrum::Profile(p) => p.delays().iter().fold(0.0f64, |m, &d| m.max(d)) / p.n_fft() as f64,
        }
    }
}

/// The 2-D Wiener interpolator of a set of pilots (see the module).
#[derive(Clone, Debug, PartialEq)]
pub struct Wiener2d {
    pilots: Vec<(usize, usize)>,
    corr: Correlation2d,
    noise_to_pilot: f64,
    n_taps: usize,
}

impl Wiener2d {
    /// The interpolator of the pilots at `(symbol, subcarrier)` positions, for the
    /// correlation `corr`, the noise of the LS estimates `noise_to_pilot` =
    /// $\sigma^{2}/E_{p}$ and $N_{\mathrm{tap}}$ = `n_taps` pilots per position (at most
    /// the number of pilots).
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] without pilots, for no tap or more taps than
    /// pilots, a negative noise or Doppler.
    pub fn new(
        pilots: Vec<(usize, usize)>,
        corr: Correlation2d,
        noise_to_pilot: f64,
        n_taps: usize,
    ) -> Result<Wiener2d> {
        ensure(!pilots.is_empty(), || "the interpolator needs pilots".into())?;
        ensure((1..=pilots.len()).contains(&n_taps), || {
            format!("{n_taps} taps for {} pilots: between 1 and the number of pilots", pilots.len())
        })?;
        ensure(noise_to_pilot >= 0.0 && corr.fd_ts >= 0.0, || "the noise and the Doppler must be non-negative".into())?;
        Ok(Wiener2d { pilots, corr, noise_to_pilot, n_taps })
    }

    /// The pilots, `(symbol, subcarrier)`.
    pub fn pilots(&self) -> &[(usize, usize)] {
        &self.pilots
    }

    /// The pilots per estimated position, $N_{\mathrm{tap}}$.
    pub fn n_taps(&self) -> usize {
        self.n_taps
    }

    /// The pilots (indices into the pilot list) used for the position `(l, k)`: the
    /// $N_{\mathrm{tap}}$ nearest in $|\Delta l| f_{D} T_{s} + |\Delta k| \tau \Delta F$.
    pub fn taps(&self, l: usize, k: usize) -> Vec<usize> {
        let (wt, wf) = (self.corr.fd_ts, self.corr.frequency_weight());
        let mut order: Vec<(f64, usize, usize)> = self
            .pilots
            .iter()
            .enumerate()
            .map(|(i, &(pl, pk))| {
                let (dl, dk) = (pl.abs_diff(l), pk.abs_diff(k));
                (dl as f64 * wt + dk as f64 * wf, dl, i)
            })
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        // the selected pilots in their own order: positions with the same taps share a
        // factorisation ([`Wiener2d::plan`])
        let mut taps: Vec<usize> = order.into_iter().take(self.n_taps).map(|(_, _, i)| i).collect();
        taps.sort_unstable();
        taps
    }

    /// The positions of `targets` grouped by their taps: `(taps, indices into targets)`,
    /// in the order of their first position.
    fn groups(&self, targets: &[(usize, usize)]) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut groups: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
        let mut index: std::collections::HashMap<Vec<usize>, usize> = std::collections::HashMap::new();
        for (t, &(l, k)) in targets.iter().enumerate() {
            let taps = self.taps(l, k);
            if let Some(&g) = index.get(&taps) {
                groups[g].1.push(t);
            } else {
                index.insert(taps.clone(), groups.len());
                groups.push((taps, vec![t]));
            }
        }
        groups
    }

    /// The correlations of the positions `members` of `targets` with the taps (one row
    /// each) and of the taps.
    fn group_system(&self, targets: &[(usize, usize)], members: &[usize], taps: &[usize]) -> (CMat, CMat) {
        let d = |a: usize, b: usize| a as isize - b as isize;
        let theta = CMat::from_shape_fn((members.len(), taps.len()), |(i, j)| {
            let ((l, k), (pl, pk)) = (targets[members[i]], self.pilots[taps[j]]);
            self.corr.at(d(l, pl), d(k, pk))
        });
        let r = CMat::from_shape_fn((taps.len(), taps.len()), |(i, j)| {
            let ((il, ik), (jl, jk)) = (self.pilots[taps[i]], self.pilots[taps[j]]);
            self.corr.at(d(il, jl), d(ik, jk))
        });
        (theta, r)
    }

    /// The estimator of the positions `targets`, factored once per set of taps: apply it
    /// to the LS estimates of every symbol that shares the pilots and the correlation.
    ///
    /// # Errors
    /// [`comnum_core::Error::Numerical`] for a singular system.
    pub fn plan(&self, targets: &[(usize, usize)]) -> Result<WienerPlan> {
        let groups = self
            .groups(targets)
            .into_iter()
            .map(|(taps, members)| {
                let (theta, r) = self.group_system(targets, &members, &taps);
                Ok((taps, members, Lmmse::new(theta, &r, self.noise_to_pilot)?))
            })
            .collect::<Result<_>>()?;
        Ok(WienerPlan { n_pilots: self.pilots.len(), n_targets: targets.len(), groups })
    }

    /// The estimates at the positions `targets` from the LS estimates `h_ls` at the
    /// pilots (in their order).
    ///
    /// # Errors
    /// [`comnum_core::Error::Shape`] unless `h_ls` has one value per pilot,
    /// [`comnum_core::Error::Numerical`] for a singular system.
    pub fn estimate(&self, h_ls: &[C64], targets: &[(usize, usize)]) -> Result<Vec<C64>> {
        self.plan(targets)?.apply(h_ls)
    }

    /// The mean square error $J_{o}$ of the estimate at every position of `targets`
    /// (eq. (12)), for a channel of the assumed correlation.
    ///
    /// # Errors
    /// [`comnum_core::Error::Numerical`] for a singular system.
    pub fn mse(&self, targets: &[(usize, usize)]) -> Result<Vec<f64>> {
        let mut out = vec![0.0; targets.len()];
        for (taps, members) in self.groups(targets) {
            let (theta, r) = self.group_system(targets, &members, &taps);
            let mse = lmmse_mse(&theta, &r, self.noise_to_pilot, &vec![1.0; members.len()])?;
            for (m, v) in members.iter().zip(mse) {
                out[*m] = v;
            }
        }
        Ok(out)
    }
}

/// The [`Wiener2d`] estimator of a set of positions, one factorisation per set of taps
/// ([`Wiener2d::plan`]).
#[derive(Clone, Debug)]
pub struct WienerPlan {
    n_pilots: usize,
    n_targets: usize,
    groups: Vec<(Vec<usize>, Vec<usize>, Lmmse)>,
}

impl WienerPlan {
    /// The estimates at the positions of the plan from the LS estimates `h_ls` at the
    /// pilots (in their order).
    ///
    /// # Errors
    /// [`comnum_core::Error::Shape`] unless `h_ls` has one value per pilot.
    pub fn apply(&self, h_ls: &[C64]) -> Result<Vec<C64>> {
        ensure_shape(h_ls.len() == self.n_pilots, || {
            format!("{} LS estimates for {} pilots", h_ls.len(), self.n_pilots)
        })?;
        let mut out = vec![C64::new(0.0, 0.0); self.n_targets];
        for (taps, members, lmmse) in &self.groups {
            let obs: Vec<C64> = taps.iter().map(|&t| h_ls[t]).collect();
            for (m, v) in members.iter().zip(lmmse.apply(&obs)?) {
                out[*m] = v;
            }
        }
        Ok(out)
    }
}

/// The robust estimator of Li, Cimini & Sollenberger (1998) for $K$ tones (see the
/// module): the design delay spread spans $K_{o}$ taps of the DFT grid, the design
/// Doppler is $\omega_{d} = 2\pi f_{\max} T_{f}$.
///
/// # Examples
/// ```
/// use comnum_waveform::ofdm::wiener::RobustDesign;
/// // 120 tones of 160 us, blocks of 200 us, 20 us and 40 Hz: the taps 0 to 15 (their Fig. 4)
/// let d = RobustDesign::new(120, 20e-6 / 160e-6, 40.0 * 200e-6)?;
/// assert_eq!(d.k_o, 16);
/// let mse_db = 10.0 * d.mse(0.1).log10();                 // SNR 10 dB
/// assert!((mse_db + 27.7).abs() < 0.1);
/// let (c, mse50) = d.fir(50, 0.1)?;
/// assert_eq!(c.len(), 50);
/// assert!(mse50 > d.mse(0.1));                           // a finite filter does worse
/// # Ok::<(), comnum_core::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RobustDesign {
    /// Tones $K$.
    pub k: usize,
    /// Taps of the design delay spread on the DFT grid, $K_{o}$.
    pub k_o: usize,
    /// The design Doppler $\omega_{d} = 2\pi f_{\max} T_{f}$ (rad per block).
    pub omega_d: f64,
}

impl RobustDesign {
    /// The design of `k` tones for a delay spread `tmax_ts` (over the useful duration
    /// $T_{s}$: the $K_{o} = \lfloor K t_{\max}/T_{s} \rfloor + 1$ taps of the grid from 0 to
    /// $t_{\max}$, at most $K$) and a maximum Doppler
    /// `fmax_tf` (over the block rate, $f_{\max} T_{f}$).
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] without tones, for a negative spread or a Doppler
    /// outside $[0, 1/2]$.
    pub fn new(k: usize, tmax_ts: f64, fmax_tf: f64) -> Result<RobustDesign> {
        ensure(k > 0, || "the design needs tones".into())?;
        ensure(tmax_ts >= 0.0 && (0.0..=0.5).contains(&fmax_tf), || {
            format!("the delay spread must be non-negative and f_max T_f in [0, 1/2], got {tmax_ts} and {fmax_tf}")
        })?;
        let k_o = ((k as f64 * tmax_ts).floor() as usize + 1).min(k);
        Ok(RobustDesign { k, k_o, omega_d: 2.0 * PI * fmax_tf })
    }

    /// The MSE of the infinite-length estimator at the noise to signal ratio `rho`,
    /// eq. (57) (normalised to the power of the channel).
    pub fn mse(&self, rho: f64) -> f64 {
        let a = self.k_o as f64 * rho / self.k as f64;
        if self.omega_d == 0.0 {
            return 0.0;
        }
        a * (1.0 - (PI / (a * self.omega_d) + 1.0).powf(-self.omega_d / PI))
    }

    /// The coefficients $c_{0}, \dots, c_{L-1}$ of the $L$-tap temporal filter and its MSE
    /// $K_{o} c_{0} \rho / K$ at the noise to signal ratio `rho` (eqs. (69)–(74)).
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for no tap, [`comnum_core::Error::Numerical`] for a
    /// singular system.
    pub fn fir(&self, taps: usize, rho: f64) -> Result<(Vec<f64>, f64)> {
        ensure(taps > 0, || "the filter needs at least one tap".into())?;
        let a = self.k_o as f64 * rho / self.k as f64;
        let r = |n: usize| sinc(n as f64 * self.omega_d / PI);
        let m =
            CMat::from_shape_fn((taps, taps), |(i, j)| C64::new(r(i.abs_diff(j)) + if i == j { a } else { 0.0 }, 0.0));
        let b = CMat::from_shape_fn((taps, 1), |(i, _)| C64::new(r(i), 0.0));
        let c: Vec<f64> =
            comnum_math::linalg::solve_hpd(&m.view(), &b.view())?.column(0).iter().map(|v| v.re).collect();
        let mse = a * c[0];
        Ok((c, mse))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corr() -> Correlation2d {
        Correlation2d { fd_ts: 0.05, doppler: DopplerSpectrum::Uniform, delay: DelaySpectrum::Uniform { tau_df: 0.04 } }
    }

    #[test]
    fn a_pilot_alone_is_estimated_with_the_mmse_of_its_own_noise() {
        let w = Wiener2d::new(vec![(0, 0)], corr(), 0.1, 1).unwrap();
        // one observation of variance 1 + 0.1: J = 1 - 1/1.1
        assert!((w.mse(&[(0, 0)]).unwrap()[0] - (1.0 - 1.0 / 1.1)).abs() < 1e-12);
        let h = w.estimate(&[C64::new(1.1, 0.0)], &[(0, 0)]).unwrap();
        assert!((h[0] - C64::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn more_taps_never_increase_the_error_and_the_taps_are_the_nearest() {
        let pilots: Vec<(usize, usize)> =
            (0..9).step_by(3).flat_map(|l| (0..30).step_by(6).map(move |k| (l, k))).collect();
        let mut last = f64::INFINITY;
        for n in [1, 4, 9, 15] {
            let j = Wiener2d::new(pilots.clone(), corr(), 0.01, n).unwrap().mse(&[(4, 9)]).unwrap()[0];
            assert!(j <= last + 1e-12, "{n} taps: {j} after {last}");
            last = j;
        }
        let w = Wiener2d::new(pilots.clone(), corr(), 0.01, 2).unwrap();
        let near: Vec<(usize, usize)> = w.taps(3, 7).into_iter().map(|i| pilots[i]).collect();
        // |dl| 0.05 + |dk| 0.04: (3, 6) at 0.04, (0, 6) at 0.19, before (3, 12) at 0.20;
        // the taps in the order of the pilots
        assert_eq!(near, vec![(0, 6), (3, 6)]);
    }

    #[test]
    fn the_robust_fir_tends_to_eq_57_and_one_tap_is_the_static_mmse() {
        let d = RobustDesign::new(120, 20e-6 / 160e-6, 0.008).unwrap();
        let rho = 0.1;
        let a = d.k_o as f64 * rho / 120.0;
        assert!((d.fir(1, rho).unwrap().1 - a / (1.0 + a)).abs() < 1e-15);
        // the filter converges slowly to eq. (57): 0.8 dB short at 100 taps in their Fig. 4
        let long = d.fir(400, rho).unwrap().1;
        assert!(long > d.mse(rho) && long < 1.1 * d.mse(rho), "{long} {}", d.mse(rho));
        assert!(RobustDesign::new(0, 0.1, 0.01).is_err() && RobustDesign::new(8, 0.1, 0.6).is_err());
        assert!(d.fir(0, rho).is_err());
    }

    #[test]
    fn invalid_configurations_are_errors() {
        assert!(Wiener2d::new(vec![], corr(), 0.1, 1).is_err());
        assert!(Wiener2d::new(vec![(0, 0)], corr(), 0.1, 2).is_err());
        assert!(Wiener2d::new(vec![(0, 0)], corr(), -0.1, 1).is_err());
        let w = Wiener2d::new(vec![(0, 0), (0, 6)], corr(), 0.1, 1).unwrap();
        assert!(w.estimate(&[C64::new(1.0, 0.0)], &[(0, 1)]).is_err());
    }
}
