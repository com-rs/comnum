//! The steps of the split-step solvers and the sampling of their fields: the rules that
//! choose the step ([`StepRule`], [`StepUpdate`]), the distributions of the steps of a
//! span ([`StepDistribution`], [`logarithmic_steps`]), where the nonlinear phase of a
//! step is taken ([`NonlinearStep`]), and the conditions on the sampling rate and the
//! time window of a field ([`SamplingCheck`], [`sampling_check`], [`signal_bandwidth`]).
//! The references are those of the parent module.

use super::Medium;
use crate::C64;
use crate::fft;
use comnum_core::{Error, Result};

/// The step size of [`ManakovSplitStep`](super::ManakovSplitStep).
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum StepRule {
    /// Steps of a fixed length (km); the last step of a span is shortened.
    Fixed(f64),
    /// Steps whose largest nonlinear phase $\max_{t} \phi(t)$ is this rotation (rad).
    NonlinearPhase(f64),
    /// This number of steps over every propagated length (a span), distributed
    /// logarithmically ([`logarithmic_steps`], Bosco et al. 2000, eq. (6)): every step
    /// carries the same integrated power, short where the power is high. Without loss,
    /// equal steps.
    Logarithmic(usize),
    /// The four-wave-mixing-aware rule of Musetti, Serena and Bononi (2018): the first step
    /// of every span bounds the phase mismatch of the four-wave mixing across the signal
    /// bandwidth $B$ (eq. (13)),
    ///
    /// $$
    /// h_{1} = \frac{\Phi_{\mathrm{FWM}}}{|\beta_{2}| (2\pi B)^{2}},
    /// $$
    ///
    /// and the next steps grow with the attenuation by `update`. The rule does not depend on
    /// the power: the error it leaves on the SNR is fixed by $\Phi_{\mathrm{FWM}}$ (about
    /// 0.01 dB for 20 rad with the constant local error, 4 rad with the nonlinear-phase
    /// update, Sec. IV-B).
    Fwm {
        /// $\Phi_{\mathrm{FWM}}$ (rad).
        phase_rad: f64,
        /// $B$ (Hz): the bandwidth of the signal; `None` measures it on the fields at the
        /// start of the propagation ([`signal_bandwidth`]).
        bandwidth_hz: Option<f64>,
        /// The growth of the steps along a span.
        update: StepUpdate,
        /// Also bound the largest nonlinear phase of a step to this rotation (rad), the
        /// rule of [`StepRule::NonlinearPhase`]: the four-wave-mixing rule alone lets the
        /// steps of a single channel exceed its dispersion length at high power.
        max_nl_phase_rad: Option<f64>,
    },
}

/// The growth of the steps of [`StepRule::Fwm`] along a span, with the power attenuation
/// $\alpha$ (Musetti et al. 2018, eqs. (9)–(10)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum StepUpdate {
    /// The constant local error of Zhang and Hayee for the symmetric step,
    /// $h_{k+1} = h_{k} e^{\alpha h_{k} / 3}$ (eq. (10)).
    #[default]
    ConstantLocalError,
    /// The update of the nonlinear-phase criterion without the fluctuations of the peak
    /// power, $L_{\mathrm{eff}}(h_{k+1}) = L_{\mathrm{eff}}(h_{k}) e^{\alpha h_{k}}$ (eq. (9), the
    /// logarithmic step of Bosco et al. 2000).
    NonlinearPhase,
}

/// The distribution of the steps of a span of a back-propagation
/// (`comnum_dsp::compensation::dbp::Dbp::distribution`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum StepDistribution {
    /// Equal steps.
    #[default]
    Uniform,
    /// The logarithmic steps of [`logarithmic_steps`] (Bosco et al. 2000, eq. (6)).
    Logarithmic,
    /// The modified logarithmic steps of Zhang, Li and Dong (2013, eq. (2)): the steps of
    /// [`logarithmic_steps`] for the attenuation times this adjusting factor (0.4
    /// recommended), less concentrated at the start of the span. As described by Häger
    /// and Pfister (2021, Sec. IV-C), whose Example 1 the tests check: two steps over a
    /// span of 100 km at 0.2 dB/km are about 30 and 70 km (15 and 85 km with the factor 1).
    ModifiedLogarithmic(f64),
}

impl StepDistribution {
    /// The same distribution with the adjusting factor `factor` when it is
    /// [`StepDistribution::ModifiedLogarithmic`] (the others have none): the factor of a
    /// specification or a keyword given apart from the name of the distribution.
    pub fn with_factor(self, factor: f64) -> StepDistribution {
        match self {
            StepDistribution::ModifiedLogarithmic(_) => StepDistribution::ModifiedLogarithmic(factor),
            d => d,
        }
    }

    /// The `n` steps of a span of `length_km` with the power attenuation `alpha_np_km`
    /// (Np/km), in the order of the propagation.
    ///
    /// # Errors
    /// As [`logarithmic_steps`].
    pub fn steps(self, alpha_np_km: f64, length_km: f64, n: usize) -> Result<Vec<f64>> {
        match self {
            StepDistribution::Uniform => logarithmic_steps(0.0, length_km, n),
            StepDistribution::Logarithmic => logarithmic_steps(alpha_np_km, length_km, n),
            StepDistribution::ModifiedLogarithmic(factor) => {
                if !(factor > 0.0 && factor.is_finite()) {
                    return Err(Error::Config(format!("the adjusting factor must be positive, got {factor}")));
                }
                logarithmic_steps(alpha_np_km * factor, length_km, n)
            }
        }
    }
}

impl std::str::FromStr for StepDistribution {
    type Err = Error;

    /// `uniform`, `logarithmic` or `modified_logarithmic` (the recommended factor 0.4;
    /// case-insensitive).
    fn from_str(s: &str) -> Result<StepDistribution> {
        match s.to_ascii_lowercase().as_str() {
            "uniform" => Ok(StepDistribution::Uniform),
            "logarithmic" => Ok(StepDistribution::Logarithmic),
            "modified_logarithmic" => Ok(StepDistribution::ModifiedLogarithmic(0.4)),
            _ => Err(Error::Config(format!(
                "the step distribution must be uniform, logarithmic or modified_logarithmic, got {s:?}"
            ))),
        }
    }
}

/// The end of the step $k$ of $n$ logarithmic steps over `length_km` (km from the start):
/// $z_{k} = -\ln(1 - k\sigma)/\alpha$, $\sigma = (1 - e^{-\alpha L})/n$, $z_{n} = L$ exactly;
/// $kL/n$ without loss.
pub(super) fn logarithmic_boundary(alpha_np_km: f64, length_km: f64, n: usize, k: usize) -> f64 {
    let a = alpha_np_km;
    if k >= n {
        return length_km;
    }
    if a * length_km < 1e-12 {
        return length_km * k as f64 / n as f64;
    }
    let sigma = -(-a * length_km).exp_m1() / n as f64;
    -(-(k as f64) * sigma).ln_1p() / a
}

/// The `n` steps (km) of the logarithmic step-size distribution of Bosco et al. (2000,
/// eq. (6)) over a fibre of `length_km` with the power attenuation $\alpha$ =
/// `alpha_np_km` (Np/km), in the order of the propagation:
///
/// $$
/// \Delta z_{k} = -\frac{1}{\alpha} \ln \frac{1 - k\sigma}{1 - (k - 1)\sigma}, \qquad
/// \sigma = \frac{1 - e^{-\alpha L}}{n}, \qquad k = 1 \dots n
/// $$
///
/// (the paper writes the field attenuation, half of $\alpha$). Every step carries the same
/// integrated power, $\int e^{-\alpha z} dz = \sigma/\alpha$ (the same effective length): the
/// steps are short at the start of the span, where the power is high. The phases of the
/// spurious four-wave mixing of the split-step method then add incoherently, so that its
/// efficiency stays below $1/n$ of the true one instead of the peaks of equal steps
/// (Bosco et al., eq. (5) and Fig. 3). Without loss ($\alpha L < 10^{-12}$) the steps are
/// equal; they add up to `length_km` exactly.
///
/// # Errors
/// [`Error::Config`] for no step, a length that is not positive and finite, or an
/// attenuation that is negative or not finite.
///
/// # Examples
/// ```
/// use comnum_math::nlse::logarithmic_steps;
/// let alpha = 0.2 / (10.0 * std::f64::consts::E.log10()); // 0.2 dB/km in Np/km
/// let h = logarithmic_steps(alpha, 80.0, 4).unwrap();
/// assert!(h[0] < h[1] && h[1] < h[2] && h[2] < h[3]);
/// assert!((h.iter().sum::<f64>() - 80.0).abs() < 1e-12);
/// // the same effective length (1 - e^{-a h}) e^{-a z} / a for every step
/// let mut z = 0.0;
/// let leff: Vec<f64> = h.iter().map(|&s| { let l = (1.0 - (-alpha * s).exp()) * (-alpha * z).exp() / alpha; z += s; l }).collect();
/// assert!(leff.iter().all(|l| (l - leff[0]).abs() < 1e-9));
/// ```
///
/// # References
///
/// 1. G. Bosco, A. Carena, V. Curri, R. Gaudino, P. Poggiolini and S. Benedetto,
///    "Suppression of spurious tones induced by the split-step method in fiber systems
///    simulation," *IEEE Photon. Technol. Lett.*, 12(5), 489–491, 2000, eq. (6).
pub fn logarithmic_steps(alpha_np_km: f64, length_km: f64, n: usize) -> Result<Vec<f64>> {
    if n == 0 {
        return Err(Error::Config("at least one step".into()));
    }
    if !(length_km > 0.0 && length_km.is_finite()) {
        return Err(Error::Config(format!("the length must be positive and finite, got {length_km} km")));
    }
    if !(alpha_np_km >= 0.0 && alpha_np_km.is_finite()) {
        return Err(Error::Config(format!("the attenuation must be non-negative and finite, got {alpha_np_km} Np/km")));
    }
    let z: Vec<f64> = (0..=n).map(|k| logarithmic_boundary(alpha_np_km, length_km, n, k)).collect();
    Ok(z.windows(2).map(|w| w[1] - w[0]).collect())
}

impl StepRule {
    /// The default rule: Musetti et al. for an SNR error of about 0.01 dB in WDM
    /// ($\Phi_{\mathrm{FWM}}$ = 20 rad, the constant local error, the bandwidth measured on
    /// the fields), and at most 0.02 rad of nonlinear phase per step, which a single
    /// channel at high power needs: one DP-16QAM channel of 32 GBd over 10 × 80 km at
    /// 2 to 8 dBm gets 0.3 to 0.7 dB too much SNR with the four-wave-mixing rule alone,
    /// the converged values with both (`validation/cards/musetti2018_fig7.yaml`).
    pub const FWM_DEFAULT: StepRule = StepRule::Fwm {
        phase_rad: 20.0,
        bandwidth_hz: None,
        update: StepUpdate::ConstantLocalError,
        max_nl_phase_rad: Some(2e-2),
    };
}

/// The bandwidth (Hz) of the fields `fields` sampled at `fs`: the span of the frequencies
/// whose power spectral density, summed over the fields, exceeds $10^{-4}$ of its peak
/// ($-40$ dB). Measured on a clean signal (before any amplifier noise, which is white).
pub fn signal_bandwidth(fields: &[&[C64]], fs: f64) -> f64 {
    let Some(n) = fields.first().map(|f| f.len()) else { return 0.0 };
    if n == 0 {
        return 0.0;
    }
    let mut psd = vec![0.0; n];
    for f in fields {
        for (p, v) in psd.iter_mut().zip(fft::fft(f)) {
            *p += v.norm_sqr();
        }
    }
    let peak = psd.iter().copied().fold(0.0, f64::max);
    if peak == 0.0 {
        return 0.0;
    }
    // the bins in increasing frequency, from -fs/2
    let signed = |k: usize| if k < n.div_ceil(2) { k as f64 } else { k as f64 - n as f64 };
    let above: Vec<f64> = psd.iter().enumerate().filter(|(_, p)| **p > 1e-4 * peak).map(|(k, _)| signed(k)).collect();
    let (lo, hi) = above.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &k| (a.min(k), b.max(k)));
    (hi - lo + 1.0) * fs / n as f64
}

/// The nonlinear step of [`ManakovSplitStep`](super::ManakovSplitStep).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum NonlinearStep {
    /// The phase of the power at the middle of the step, after the first half of the
    /// dispersion, as [`SplitStep`](super::SplitStep) (eq. (3) of Sinkin et al. 2003): no iteration.
    #[default]
    Midpoint,
    /// OptiCommPy's `manakovSSF`: the phase of the trapezoidal mean of the power at the
    /// two ends of the step, iterated until the field converges. When channels walk off
    /// within a step, the mean smooths the intensity that drives the cross-phase
    /// modulation (see [`ManakovSplitStep`](super::ManakovSplitStep)).
    Trapezoidal,
    /// Optilux's `fiber` (Shao, Liang and Kumar 2014): the phase of the power at the
    /// start of the nonlinear block over the effective length of the step,
    ///
    /// $$
    /// \phi = \gamma P\, L_{\mathrm{eff}}(h), \qquad
    /// L_{\mathrm{eff}}(h) = \frac{1 - e^{-\alpha h}}{\alpha},
    /// $$
    ///
    /// which integrates the attenuation within the step exactly: the first half step
    /// carries the dispersion only, the attenuation of the whole step follows the phase
    /// with the second half of the dispersion. No iteration. Over a span of constant
    /// power the phases of the steps add up to $\gamma P L_{\mathrm{eff}}(L)$ whatever the
    /// step; with a distributed gain the medium carries no attenuation and the scheme is
    /// [`NonlinearStep::Midpoint`]. At 1 km steps its field is closer to the
    /// step-converged solution than the midpoint one on a 16-QAM channel at 4 dBm over
    /// 10 × 80 km (`docs/python/user_guide/fibre_validation.rst`, section 3).
    EffectiveLength,
}

impl std::str::FromStr for NonlinearStep {
    type Err = Error;

    /// `midpoint`, `trapezoidal` or `effective_length` (case-insensitive).
    fn from_str(s: &str) -> Result<NonlinearStep> {
        match s.to_ascii_lowercase().as_str() {
            "trapezoidal" => Ok(NonlinearStep::Trapezoidal),
            "midpoint" => Ok(NonlinearStep::Midpoint),
            "effective_length" => Ok(NonlinearStep::EffectiveLength),
            _ => Err(Error::Config(format!(
                "the nonlinear step must be midpoint, trapezoidal or effective_length, got {s:?}"
            ))),
        }
    }
}

/// The sampling conditions of a split-step simulation of a signal of bandwidth $B$
/// over a dispersive link (Musetti et al. 2018, Appendix B): the sampling rate must hold
/// the four-wave mixing that falls on the signal, at least $\frac{3}{2} B$ (all of the
/// first-order mixing needs $3 B$), and the block must be longer than the walk-off
/// across $B$ over the whole link, $|\beta_{2}| \, 2\pi B \, L$, or the circular
/// convolution wraps the fast channels onto the slow ones. [`sampling_check`] computes
/// them; [`SamplingCheck::is_sufficient`] says whether both hold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingCheck {
    /// The bandwidth of the launched signal (Hz, [`signal_bandwidth`]).
    pub bandwidth_hz: f64,
    /// The sampling rate (Hz).
    pub rate_hz: f64,
    /// The smallest rate that holds the four-wave mixing on the signal, $\frac{3}{2} B$ (Hz).
    pub min_rate_hz: f64,
    /// The duration of the block (s).
    pub window_s: f64,
    /// The walk-off across the bandwidth over the link (s).
    pub walkoff_s: f64,
}

impl SamplingCheck {
    /// Whether the rate and the block are sufficient.
    pub fn is_sufficient(&self) -> bool {
        self.rate_hz >= self.min_rate_hz && self.window_s >= self.walkoff_s
    }

    /// Nothing when [`SamplingCheck::is_sufficient`].
    ///
    /// # Errors
    /// [`Error::Config`] naming what is missing otherwise.
    pub fn ensure(&self) -> Result<()> {
        if self.is_sufficient() {
            return Ok(());
        }
        let mut what = Vec::new();
        if self.rate_hz < self.min_rate_hz {
            what.push(format!(
                "the sampling rate {:.1} GHz is below 3/2 of the signal bandwidth {:.1} GHz ({:.1} GHz): the four-wave mixing aliases onto the signal (Musetti et al. 2018, Table I)",
                self.rate_hz / 1e9,
                self.bandwidth_hz / 1e9,
                self.min_rate_hz / 1e9
            ));
        }
        if self.window_s < self.walkoff_s {
            what.push(format!(
                "the block of {:.2} ns is shorter than the walk-off {:.2} ns across the bandwidth over the link: the circular convolution wraps the channels (Musetti et al. 2018, Appendix B)",
                self.window_s * 1e9,
                self.walkoff_s * 1e9
            ));
        }
        Err(Error::Config(format!("insufficient sampling: {} (check_sampling: false to run anyway)", what.join("; "))))
    }
}

/// The [`SamplingCheck`] of `fields` sampled at `fs` through `length_km` of `medium`.
pub fn sampling_check(fields: &[&[C64]], fs: f64, medium: &Medium, length_km: f64) -> SamplingCheck {
    let b = signal_bandwidth(fields, fs);
    let n = fields.first().map_or(0, |f| f.len()) as f64;
    SamplingCheck {
        bandwidth_hz: b,
        rate_hz: fs,
        min_rate_hz: 1.5 * b,
        window_s: n / fs,
        walkoff_s: medium.beta2_s2_km.abs() * 2.0 * std::f64::consts::PI * b * length_km,
    }
}

#[cfg(test)]
mod literature {
    use super::*;

    /// The efficiency of the fictitious four-wave mixing of a split step (Bosco et al.
    /// 2000, eq. (5)): $\eta'' = |\sum_{n} \gamma_{n} e^{j \phi_{n}}|^{2}/(1 - e^{-\alpha L})^{2}$ with
    /// $\gamma_{n} = (1 - e^{-\alpha \Delta z_{n}}) e^{-\alpha z_{n-1}}$ and the phase mismatch
    /// $\phi_{n} = \Delta\beta\, z_{n-1}$ accumulated by the start of the step.
    fn fictitious_fwm(alpha: f64, steps: &[f64], dbeta: f64) -> f64 {
        let total = 1.0 - (-alpha * steps.iter().sum::<f64>()).exp();
        let mut z = 0.0;
        let mut sum = C64::new(0.0, 0.0);
        for &h in steps {
            let g = (1.0 - (-alpha * h).exp()) * (-alpha * z).exp();
            sum += C64::from_polar(g, dbeta * z);
            z += h;
        }
        sum.norm_sqr() / (total * total)
    }

    /// Bosco et al. (2000), Sec. III and Fig. 3 (L = 50 km, 0.25 dB/km, K = 50): with the
    /// logarithmic steps every $\gamma_{n}$ is the same (eq. (6) imposes it), so that beyond
    /// the true efficiency the fictitious one stays near the floor $1/K$; with equal steps
    /// it has peaks of 1, the first at $\Delta\beta \Delta z = 2\pi$ (the "unrealistic
    /// pumps" of Fig. 2).
    #[test]
    fn bosco_2000_fig_3_logarithmic_steps_keep_the_fictitious_fwm_near_1_over_k() {
        let alpha = 0.25 / (10.0 * std::f64::consts::E.log10());
        let (l, k) = (50.0, 50usize);
        let log = logarithmic_steps(alpha, l, k).unwrap();
        let uniform = vec![l / k as f64; k];
        // gamma_n constant for the logarithmic steps
        let mut z = 0.0;
        let gammas: Vec<f64> = log
            .iter()
            .map(|&h| {
                let g = (1.0 - (-alpha * h).exp()) * (-alpha * z).exp();
                z += h;
                g
            })
            .collect();
        assert!(gammas.iter().all(|g| (g - gammas[0]).abs() < 1e-12 * gammas[0]));
        // the phase mismatch from 2 to 40 rad/km: past the decay of the true efficiency
        let grid: Vec<f64> = (0..3800).map(|i| 2.0 + i as f64 * 0.01).collect();
        let eta_log: Vec<f64> = grid.iter().map(|&b| fictitious_fwm(alpha, &log, b)).collect();
        let eta_uni: Vec<f64> = grid.iter().map(|&b| fictitious_fwm(alpha, &uniform, b)).collect();
        let peak_uniform = fictitious_fwm(alpha, &uniform, 2.0 * std::f64::consts::PI / (l / k as f64));
        let mean_log = eta_log.iter().sum::<f64>() / grid.len() as f64;
        let max_log = eta_log.iter().copied().fold(0.0, f64::max);
        let max_uni = eta_uni.iter().copied().fold(0.0, f64::max);
        assert!((peak_uniform - 1.0).abs() < 1e-9, "the peak of equal steps {peak_uniform}");
        assert!(max_uni > 0.99, "equal steps {max_uni}");
        assert!(
            (mean_log * k as f64 - 1.0).abs() < 0.2,
            "the mean of the logarithmic steps {mean_log}, floor {}",
            1.0 / k as f64
        );
        assert!(max_log < 0.2, "the largest of the logarithmic steps {max_log}");
    }

    /// Häger & Pfister (2021), Sec. IV-C, Example 1: the modified logarithmic heuristic
    /// of Zhang et al. (2013, eq. (2)) with the adjusting factor 0.4 cuts a span of 100 km
    /// (0.2 dB/km, Table I) into steps of about 30 and 70 km, applied from the end of the
    /// span by the back-propagation ($\delta_{1} = \delta_{3} \approx 70$ km, $\delta_{2} = \delta_{4} \approx 30$ km).
    #[test]
    fn hager_2021_example_1_modified_logarithmic_steps() {
        let alpha = 0.2 / (10.0 * std::f64::consts::E.log10());
        let h = StepDistribution::ModifiedLogarithmic(0.4).steps(alpha, 100.0, 2).unwrap();
        assert!((h[0] - 30.0).abs() < 1.0 && (h[1] - 70.0).abs() < 1.0, "{h:?}");
        let bosco = StepDistribution::Logarithmic.steps(alpha, 100.0, 2).unwrap();
        assert!((bosco[0] - 14.8).abs() < 0.1, "{bosco:?}");
        assert_eq!(
            "modified_logarithmic".parse::<StepDistribution>().unwrap(),
            StepDistribution::ModifiedLogarithmic(0.4)
        );
        assert!(StepDistribution::ModifiedLogarithmic(0.0).steps(alpha, 100.0, 2).is_err());
    }
}
