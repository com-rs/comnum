//! Digital back-propagation.
//!
//! The receiver runs the split-step solver of [`comnum_math::nlse`] through the negated
//! medium, span after span in reverse order, removing the amplifier gain first. Given the
//! [`Steps`] of the forward model ([`Dbp::forward`]) and as many steps per span, it
//! uses those steps and inverts the forward split step exactly without noise.
//!
//! # Model
//!
//! With coarser steps of length $\delta$, the nonlinear phase of a step is
//! $\xi \gamma |A|^{2} L_{\mathrm{eff}}(\delta)$ with the effective length of the symmetric step
//!
//! $$
//! L_{\mathrm{eff}}(\delta) = \frac{2 \sinh(\alpha \delta / 2)}{\alpha}
//! $$
//!
//! (the power at the middle of the step times $L_{\mathrm{eff}}$ is the power integrated over
//! the step); with the step length itself, one step per span would apply about 60 % of
//! the nonlinear phase of an 80 km span. The scale $\xi$ is the classical tuning of DBP
//! with few steps per span, where the full phase over-compensates at high symbol rates
//! and an optimised $\xi < 1$ performs better; $\xi = 0$ gives the linear
//! back-propagation.
//!
//! The steps of a span are equal, or distributed logarithmically ([`Dbp::distribution`],
//! [`comnum_math::nlse::logarithmic_steps`], Bosco et al. 2000): every step then
//! carries the same integrated power of the forward span, and the back-propagation
//! applies them from the end of the span, the short steps where the power was high last.
//! With few steps per span the nonlinear phase is then placed where it was accumulated.
//!
//! [`Dbp::run_dual`] back-propagates the two polarisations of a dual-polarisation link
//! through the Manakov equation (nonlinear coefficient $\frac{8}{9}\gamma$ on the total
//! power), with the same steps and the same effective length.
//!
//! # Long steps: a step over several spans and a filter on the power
//!
//! With [`Dbp::steps_per_link`] the link of $N$ spans is back-propagated in $M$ equal
//! symmetric steps that may cover several spans (or a fraction of one): the amplifiers
//! restore the loss of their span, so the power repeats from span to span, and a step of
//! $s$ spans carries the nonlinear phase of $s$ spans, $\xi \gamma |A|^{2} s
//! L_{\mathrm{eff}}$, $L_{\mathrm{eff}} = (1 - e^{-\alpha L})/\alpha$ of a span $L$, applied at
//! its middle (the position Rafique et al. 2011 found best, their Fig. 5). The steps
//! themselves are lossless: the gain of the amplifiers is assumed to restore the loss of
//! the span exactly.
//!
//! With such steps the power of a sample is no longer that of its neighbours along the
//! step: the dispersion has spread it. [`Dbp::power_filter`] takes the phase on the power
//! filtered by real taps $c_{k}$ (see `SplitStep::with_power_filter`),
//! $\xi \gamma s L_{\mathrm{eff}} \sum_{k} c_{k} |A\[n - k\]|^{2}$: the weighted DBP of Rafique
//! et al. (2011, eq. (2), Gaussian weights) and the enhanced split step of Häger & Pfister
//! (2021, eq. (9)). [`gaussian_power_filter`] gives the Gaussian weights, of unit sum.
//!
//! # Examples
//! ```
//! use comnum_dsp::compensation::dbp::Dbp;
//! use comnum_math::nlse::{Medium, SplitStep, Steps};
//! use comnum_math::C64;
//! let m = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
//! let x: Vec<C64> = (0..128).map(|n| C64::new(1e-2 * (n % 3) as f64, 0.0)).collect();
//! let mut y = x.clone();
//! let forward = Steps::new(40, 2.0);                         // one 80 km span
//! SplitStep::new(&m, forward, 128, 64e9).run(&mut y);
//! let dbp = Dbp { forward: Some(forward), steps_per_span: 40, ..Dbp::new(m, 80.0, 1, 1.0) };
//! let z = dbp.run(&y, 64e9).unwrap();
//! assert!(x.iter().zip(&z).all(|(a, b)| (a - b).norm() < 1e-12));
//! ```
//!
//! # References
//!
//! 1. E. Ip and J. M. Kahn, "Compensation of dispersion and nonlinear impairments
//!    using digital backpropagation," *J. Lightw. Technol.*, 26(20), 2008.
//! 2. D. Rafique, M. Mussolin, M. Forzati, J. Mårtensson, M. N. Chugtai and A. D. Ellis,
//!    "Compensation of intra-channel nonlinear fibre impairments using simplified
//!    digital back-propagation algorithm," *Opt. Express*, 19(10), 9453–9460, 2011.
//! 3. C. Häger and H. D. Pfister, "Physics-based deep learning for fiber-optic
//!    communication systems," *IEEE J. Sel. Areas Commun.*, 39(1), 280–294, 2021.

use comnum_core::{Result, ensure};
use comnum_math::C64;
use comnum_math::nlse::{Medium, SplitStep, StepDistribution, Steps};

/// Digital back-propagation of a link of `n_spans` identical spans.
#[derive(Clone, Debug, PartialEq)]
pub struct Dbp {
    /// The forward medium of the link (it is negated inside).
    pub medium: Medium,
    /// Length of a span (km).
    pub span_km: f64,
    /// Number of spans.
    pub n_spans: usize,
    /// Field gain of the amplifier after every span (removed before each span).
    pub span_gain: f64,
    /// Symmetric steps per span.
    pub steps_per_span: usize,
    /// Scale `ξ ≥ 0` of the nonlinear phase.
    pub xi: f64,
    /// The steps of a span of the forward model, if known: with as many steps per span
    /// (`steps_per_span == forward.n_steps`, compared as integers) and uniform steps, the
    /// back-propagation uses exactly those steps and their nonlinear length (the exact
    /// inverse).
    pub forward: Option<Steps>,
    /// The distribution of the steps of a span: equal steps, the logarithmic steps of
    /// Bosco et al. (2000), short where the power of the forward propagation is high, or
    /// their modification by Zhang et al. (2013).
    pub distribution: StepDistribution,
    /// The nonlinear length of a coarse step: its effective length (the default), or the
    /// step itself (the convention of Häger & Pfister 2021).
    pub nonlinear_length: NonlinearLength,
    /// `Some(m)`: `m` equal steps for the whole link instead of [`Dbp::steps_per_span`]
    /// (see the module): a step covers `n_spans / m` spans.
    pub steps_per_link: Option<usize>,
    /// The real taps of the filter on the power of every nonlinear step, centred (see the
    /// module); empty for the instantaneous power.
    pub power_filter: Vec<f64>,
}

/// The Gaussian weights $c_{k} \propto e^{-k^{2}/(2\sigma^{2})}$, $k = -K, \dots, K$, of the
/// weighted DBP (Rafique et al. 2011, eq. (3), the profile they found best), `n_taps =
/// 2K + 1` of them, of standard deviation `sigma_samples` (samples), normalised to a unit
/// sum: a constant power gets the phase of the plain step.
///
/// # Errors
/// [`comnum_core::Error::Config`] for an even number of taps or a standard deviation
/// that is not positive.
///
/// # Examples
/// ```
/// use comnum_dsp::compensation::dbp::gaussian_power_filter;
/// let c = gaussian_power_filter(5, 1.0)?;
/// assert!((c.iter().sum::<f64>() - 1.0).abs() < 1e-15 && c[2] > c[1] && c[1] == c[3]);
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn gaussian_power_filter(n_taps: usize, sigma_samples: f64) -> Result<Vec<f64>> {
    ensure(n_taps % 2 == 1, || format!("the filter on the power needs an odd number of taps, got {n_taps}"))?;
    ensure(sigma_samples > 0.0 && sigma_samples.is_finite(), || {
        format!("the standard deviation must be positive, got {sigma_samples}")
    })?;
    let half = (n_taps / 2) as f64;
    let c: Vec<f64> = (0..n_taps)
        .map(|i| {
            let k = i as f64 - half;
            (-k * k / (2.0 * sigma_samples * sigma_samples)).exp()
        })
        .collect();
    let sum: f64 = c.iter().sum();
    Ok(c.iter().map(|v| v / sum).collect())
}

/// The filter on the power of [`Dbp::power_filter`] from a number of taps and an
/// optional standard deviation (samples), as the blocks and the bindings take them:
/// none for a single tap without a deviation, [`gaussian_power_filter`] otherwise.
///
/// # Errors
/// [`comnum_core::Error::Config`] for more than one tap without a standard deviation,
/// and as [`gaussian_power_filter`].
///
/// # Examples
/// ```
/// use comnum_dsp::compensation::dbp::power_filter;
/// assert!(power_filter(1, None)?.is_empty() && power_filter(3, Some(1.0))?.len() == 3);
/// assert!(power_filter(3, None).is_err());
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn power_filter(n_taps: usize, sigma_samples: Option<f64>) -> Result<Vec<f64>> {
    match (n_taps, sigma_samples) {
        (1, None) => Ok(Vec::new()),
        (n_taps, Some(sigma)) => gaussian_power_filter(n_taps, sigma),
        (n_taps, None) => Err(comnum_core::Error::Config(format!(
            "{n_taps} taps of the filter on the power need its standard deviation (power_filter_sigma)"
        ))),
    }
}

/// The nonlinear length of a coarse step of the back-propagation (with the steps of the
/// forward model the back-propagation uses theirs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum NonlinearLength {
    /// The effective length of the symmetric step, $2\sinh(\alpha\delta/2)/\alpha$: the power at
    /// the middle of the step times it is the power integrated over the step.
    #[default]
    Effective,
    /// The step $\delta$ itself: the power at the middle of the step times the step (the
    /// midpoint rule), the symmetric steps of the code of Häger & Pfister (2021, LDBP,
    /// `ssfm_parameters`, the attenuation carried by the nonlinear coefficients). Both
    /// conventions agree as the step shrinks; on a step of 70 km at 0.2 dB/km this one
    /// gives 33 % less phase than `Effective`.
    Step,
}

impl std::str::FromStr for NonlinearLength {
    type Err = comnum_core::Error;

    /// `effective` or `step` (case-insensitive).
    fn from_str(s: &str) -> Result<NonlinearLength> {
        match s.to_ascii_lowercase().as_str() {
            "effective" => Ok(NonlinearLength::Effective),
            "step" => Ok(NonlinearLength::Step),
            _ => Err(comnum_core::Error::Config(format!("the nonlinear length must be effective or step, got {s:?}"))),
        }
    }
}

impl Dbp {
    /// One step per span, the full nonlinear phase, the effective length.
    pub fn new(medium: Medium, span_km: f64, n_spans: usize, span_gain: f64) -> Dbp {
        Dbp {
            medium,
            span_km,
            n_spans,
            span_gain,
            steps_per_span: 1,
            xi: 1.0,
            forward: None,
            distribution: StepDistribution::Uniform,
            nonlinear_length: NonlinearLength::Effective,
            steps_per_link: None,
            power_filter: Vec::new(),
        }
    }

    /// The forward steps, when they are those of the back-propagation.
    fn exact(&self) -> Option<Steps> {
        match self.distribution {
            StepDistribution::Uniform => self.forward.filter(|f| f.n_steps == self.steps_per_span),
            _ => None,
        }
    }

    /// The step length (km): that of the forward model with as many steps, the span
    /// divided by the steps otherwise.
    pub fn step_km(&self) -> f64 {
        match self.exact() {
            Some(f) => f.step_km,
            None => self.span_km / self.steps_per_span as f64,
        }
    }

    /// The steps of the whole link (see the module), lossless.
    fn link_steps(&self, m: usize, n: usize, fs: f64) -> Result<SplitStep> {
        ensure(m >= 1, || "at least one step for the link".into())?;
        // the forward steps, those of a span, do not apply to steps over the link
        ensure(self.distribution == StepDistribution::Uniform, || {
            "steps for the whole link are equal: no other distribution".into()
        })?;
        ensure(self.nonlinear_length == NonlinearLength::Effective, || {
            "steps for the whole link take the effective length of the spans they cover".into()
        })?;
        let a = self.medium.alpha_np_km;
        // the gain that restores a span, or the forward steps of a span when they
        // overrun it ([`Dbp::step_km`])
        let restores = |km: f64| (self.span_gain / (a / 2.0 * km).exp() - 1.0).abs() < 1e-9;
        ensure(restores(self.span_km) || restores(self.steps_per_span as f64 * self.step_km()), || {
            format!(
                "steps over several spans assume amplifiers that restore the loss of their span: a field gain of {}, got {}",
                (a / 2.0 * self.span_km).exp(),
                self.span_gain
            )
        })?;
        let spans = self.n_spans as f64 / m as f64;
        let h = spans * self.span_km;
        let leff = if a * self.span_km < 1e-12 { self.span_km } else { (1.0 - (-a * self.span_km).exp()) / a };
        let lossless = Medium { alpha_np_km: 0.0, ..self.medium };
        let steps = vec![(h, self.xi * spans * leff); m];
        self.filtered(SplitStep::with_lengths(&lossless.negated(), &steps, n, fs))
    }

    /// The steps with the filter on the power, when there is one.
    fn filtered(&self, steps: SplitStep) -> Result<SplitStep> {
        if self.power_filter.is_empty() { Ok(steps) } else { steps.with_power_filter(&self.power_filter) }
    }

    /// The steps of the back-propagation for a block of `n` samples at `fs`: those of
    /// the whole link with [`Dbp::steps_per_link`], those of one span otherwise (an
    /// empty block has nothing to propagate).
    fn plan(&self, n: usize, fs: f64) -> Result<SplitStep> {
        ensure(self.xi >= 0.0 && self.xi.is_finite(), || {
            format!("the nonlinear scale must be non-negative and finite, got {}", self.xi)
        })?;
        match self.steps_per_link {
            Some(m) => self.link_steps(m, n, fs),
            None => self.steps(n, fs),
        }
    }

    /// The steps of one span.
    fn steps(&self, n: usize, fs: f64) -> Result<SplitStep> {
        ensure(self.steps_per_span >= 1, || "at least one step per span".into())?;
        let a = self.medium.alpha_np_km;
        // the effective length of a symmetric step of h km (the power at its middle times
        // it is the power integrated over the step)
        let leff = |h: f64| match self.nonlinear_length {
            NonlinearLength::Step => h,
            _ if a * h < 1e-12 => h,
            _ => 2.0 * (a * h / 2.0).sinh() / a,
        };
        if self.distribution != StepDistribution::Uniform && self.steps_per_span > 1 {
            // the steps of the forward span, applied from its end: the longest first
            let forward = self.distribution.steps(a, self.span_km, self.steps_per_span)?;
            let steps: Vec<(f64, f64)> = forward.iter().rev().map(|&h| (h, self.xi * leff(h))).collect();
            return self.filtered(SplitStep::with_lengths(&self.medium.negated(), &steps, n, fs));
        }
        let h = self.step_km();
        // nonlinear length of a step: that of the forward model for the exact inverse,
        // the effective length of the symmetric step otherwise
        let h_nl = self.xi
            * match self.exact() {
                Some(f) => f.nl_length_km,
                None => leff(h),
            };
        let steps = Steps { n_steps: self.steps_per_span, step_km: h, nl_length_km: h_nl };
        self.filtered(SplitStep::new(&self.medium.negated(), steps, n, fs))
    }

    /// Back-propagate a field (√W) sampled at `fs`.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for zero steps, a `xi` negative or not finite, a
    /// filter on the power of an even number of taps, of a tap that is not finite or
    /// longer than the block, and steps for the whole link with another distribution,
    /// the nonlinear length [`NonlinearLength::Step`] or a gain that does not restore
    /// the loss of a span (they ignore [`Dbp::forward`]).
    pub fn run(&self, x: &[C64], fs: f64) -> Result<Vec<C64>> {
        let steps = self.plan(x.len(), fs)?;
        let mut e = x.to_vec();
        if self.steps_per_link.is_some() {
            steps.run(&mut e);
            return Ok(e);
        }
        for _ in 0..self.n_spans {
            for v in &mut e {
                *v /= self.span_gain;
            }
            steps.run(&mut e);
        }
        Ok(e)
    }

    /// Back-propagate both polarisations (√W) of a dual-polarisation link sampled at
    /// `fs` through the Manakov equation (nonlinear coefficient $\frac{8}{9}\gamma$ on
    /// the total power), in place.
    ///
    /// # Errors
    /// As [`Dbp::run`], and [`comnum_core::Error::Shape`] for polarisations of
    /// different lengths.
    pub fn run_dual(&self, x: &mut [C64], y: &mut [C64], fs: f64) -> Result<()> {
        comnum_core::same_len(x.len(), y.len(), "the polarisations")?;
        let steps = self.plan(x.len(), fs)?;
        if self.steps_per_link.is_some() {
            steps.run_dual(x, y);
            return Ok(());
        }
        for _ in 0..self.n_spans {
            for v in x.iter_mut().chain(y.iter_mut()) {
                *v /= self.span_gain;
            }
            steps.run_dual(x, y);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use comnum_math::nlse::{ManakovSplitStep, StepRule};

    #[test]
    fn steps_over_the_link_refuse_what_they_cannot_honour() {
        let m = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let ideal = (0.046_f64 / 2.0 * 80.0).exp();
        let x = vec![C64::new(1e-2, 0.0); 64];
        let link = Dbp { steps_per_link: Some(2), ..Dbp::new(m, 80.0, 4, ideal) };
        assert!(link.run(&x, 1e11).is_ok());
        // a negative or undefined nonlinear scale, on this path as on the span steps
        for xi in [-1.0, f64::NAN] {
            assert!(Dbp { xi, ..link.clone() }.run(&x, 1e11).is_err());
            assert!(Dbp { xi, ..Dbp::new(m, 80.0, 4, ideal) }.run(&x, 1e11).is_err());
        }
        // the midpoint rule is a convention of the span steps
        assert!(Dbp { nonlinear_length: NonlinearLength::Step, ..link.clone() }.run(&x, 1e11).is_err());
        // a filter longer than the block
        assert!(Dbp { power_filter: gaussian_power_filter(65, 3.0).unwrap(), ..link }.run(&x, 1e11).is_err());
    }

    #[test]
    fn a_unit_filter_on_the_power_is_the_plain_step_to_the_last_bit() {
        let m = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let x: Vec<C64> = (0..256).map(|n| comnum_math::cis(0.37 * n as f64) * (1e-2 * (1 + n % 5) as f64)).collect();
        for steps in [1usize, 3] {
            let plain = Dbp { steps_per_span: steps, ..Dbp::new(m, 80.0, 3, 1.0) };
            let unit = Dbp { power_filter: vec![1.0], ..plain.clone() };
            assert_eq!(plain.run(&x, 64e9).unwrap(), unit.run(&x, 64e9).unwrap());
        }
        // an even number of taps is refused
        let bad = Dbp { power_filter: vec![0.5, 0.5], ..Dbp::new(m, 80.0, 3, 1.0) };
        assert!(bad.run(&x, 64e9).is_err());
        assert!(gaussian_power_filter(4, 1.0).is_err() && gaussian_power_filter(5, 0.0).is_err());
    }

    #[test]
    fn steps_over_the_link_carry_the_nonlinear_phase_of_its_spans() {
        // a constant field without dispersion: the phase removed is xi gamma P N L_eff,
        // whatever the number of steps and with any filter of unit sum
        let a = 0.2 / (10.0 * std::f64::consts::E.log10());
        let m = Medium { alpha_np_km: a, beta2_s2_km: 0.0, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let g = (a / 2.0 * 80.0).exp();
        let p0 = 4e-3;
        let leff = (1.0 - (-a * 80.0).exp()) / a;
        let expected = -0.7 * 1.3 * p0 * 25.0 * leff;
        let x = vec![C64::new(p0.sqrt(), 0.0); 64];
        for m_steps in [1usize, 4, 25, 100] {
            for filter in [Vec::new(), gaussian_power_filter(41, 6.0).unwrap()] {
                let dbp =
                    Dbp { steps_per_link: Some(m_steps), xi: 0.7, power_filter: filter, ..Dbp::new(m, 80.0, 25, g) };
                let y = dbp.run(&x, 64e9).unwrap();
                assert!((y[10].arg() - expected).abs() < 1e-12, "{m_steps} steps: {} vs {expected}", y[10].arg());
                assert!((y[10].norm_sqr() / p0 - 1.0).abs() < 1e-12);
            }
        }
        // the amplifiers must restore the loss of their span
        let dbp = Dbp { steps_per_link: Some(4), ..Dbp::new(m, 80.0, 25, 1.0) };
        assert!(dbp.run(&x, 64e9).is_err());
        let dbp =
            Dbp { steps_per_link: Some(4), distribution: StepDistribution::Logarithmic, ..Dbp::new(m, 80.0, 25, g) };
        assert!(dbp.run(&x, 64e9).is_err());
    }

    #[test]
    fn one_step_per_span_over_the_link_is_the_lossless_equivalent_of_the_span_steps() {
        // without dispersion within a span the order of loss and phase does not matter:
        // the steps over the link give the field of one effective-length step per span
        let a = 0.2 / (10.0 * std::f64::consts::E.log10());
        let m = Medium { alpha_np_km: a, beta2_s2_km: 0.0, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let g = (a / 2.0 * 80.0).exp();
        let x: Vec<C64> = (0..128).map(|n| comnum_math::cis(0.2 * n as f64) * (2e-2 * (1 + n % 3) as f64)).collect();
        let per_span = Dbp { nonlinear_length: NonlinearLength::Effective, ..Dbp::new(m, 80.0, 5, g) };
        let over_link = Dbp { steps_per_link: Some(5), ..Dbp::new(m, 80.0, 5, g) };
        let (u, v) = (per_span.run(&x, 64e9).unwrap(), over_link.run(&x, 64e9).unwrap());
        // the effective length of the symmetric step (2 sinh(aL/2)/a, at the middle) and
        // the span's (1 - e^{-aL})/a, at its start: the same phase once the power is that
        // of the start of the span
        let ratio = (1.0 - (-a * 80.0).exp()) / a / (2.0 * (a * 40.0).sinh() / a * (-a * 40.0).exp());
        assert!((ratio - 1.0).abs() < 1e-12);
        assert!(u.iter().zip(&v).all(|(p, q)| (p - q).norm() < 1e-12 * p.norm().max(1e-3)));
    }

    #[test]
    fn a_wider_filter_on_the_power_gives_long_steps_a_better_back_propagation() {
        // 25 x 80 km, a field of 10.7 GHz at 2 dBm, 4 steps for the link: the weighted step
        // back-propagates the forward field better than the instantaneous power (Rafique
        // et al. 2011, Fig. 3: Gaussian weights better than the delta for few steps); the
        // residual is measured after the common phase is removed
        use comnum_core::rng::Rng;
        let a = 0.2 / (10.0 * std::f64::consts::E.log10());
        let m = Medium { alpha_np_km: a, beta2_s2_km: -21.683e-24, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let (n, fs, spans) = (4096usize, 64e9, 25usize);
        let g = (a / 2.0 * 80.0).exp();
        let mut rng = Rng::new(3);
        let plan = comnum_math::fft::plan(n);
        let mut x: Vec<C64> = (0..n).map(|_| rng.complex_gaussian()).collect();
        plan.forward(&mut x);
        for (k, v) in x.iter_mut().enumerate() {
            if (k.min(n - k) as f64) / n as f64 * fs > 5.35e9 {
                *v = C64::new(0.0, 0.0);
            }
        }
        plan.inverse(&mut x);
        let pw = x.iter().map(C64::norm_sqr).sum::<f64>() / n as f64;
        for v in &mut x {
            *v *= (1e-3 * 10f64.powf(0.2) / pw).sqrt();
        }
        let forward = comnum_math::nlse::SplitStep::new(&m, Steps::new(200, 0.4), n, fs);
        let mut y = x.clone();
        for _ in 0..spans {
            forward.run(&mut y);
            for v in &mut y {
                *v *= g;
            }
        }
        let residual = |filter: Vec<f64>, xi: f64| {
            let dbp = Dbp { steps_per_link: Some(4), xi, power_filter: filter, ..Dbp::new(m, 80.0, spans, g) };
            let z = dbp.run(&y, fs).unwrap();
            // the common phase removed, as a receiver does (least-squares rotation)
            let c: C64 = z.iter().zip(&x).map(|(p, q)| q * p.conj()).sum();
            let r = c / c.norm();
            let e: f64 = z.iter().zip(&x).map(|(p, q)| (p * r - q).norm_sqr()).sum();
            e / x.iter().map(C64::norm_sqr).sum::<f64>()
        };
        let best = |filter: &dyn Fn() -> Vec<f64>| {
            [0.2, 0.4, 0.6, 0.8, 1.0].iter().map(|&xi| residual(filter(), xi)).fold(f64::INFINITY, f64::min)
        };
        // at its best scale: linear 0.426, the instantaneous power 0.206, Gaussian weights
        // of 5 samples (about one symbol at 6 samples per symbol) 0.071
        let delta = best(&Vec::new);
        let weighted = best(&|| gaussian_power_filter(41, 5.0).unwrap());
        let linear = residual(Vec::new(), 0.0);
        assert!(
            weighted < delta / 2.0 && delta < linear / 2.0,
            "linear {linear:.3e}, delta {delta:.3e}, weighted {weighted:.3e}"
        );
    }

    #[test]
    fn an_empty_field_is_back_propagated_to_an_empty_field_instead_of_panicking() {
        let m = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        assert_eq!(Dbp::new(m, 80.0, 1, 1.0).run(&[], 64e9).unwrap(), Vec::<C64>::new());
        let (mut x, mut y) = (Vec::new(), Vec::new());
        Dbp { steps_per_span: 4, ..Dbp::new(m, 80.0, 2, 3.0) }.run_dual(&mut x, &mut y, 64e9).unwrap();
        assert!(x.is_empty() && y.is_empty());
    }

    #[test]
    fn the_forward_steps_are_recognised_by_their_number_not_by_a_floor_of_lengths() {
        // 80 km in 115 steps: 80/115 is not representable, and the floor of 80 over it
        // may be 114
        let m = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let forward = Steps::new(115, 80.0 / 115.0);
        let x: Vec<C64> = (0..64).map(|n| C64::new(1e-2 * (n % 5) as f64, 1e-3 * (n % 2) as f64)).collect();
        let mut y = x.clone();
        SplitStep::new(&m, forward, 64, 64e9).run(&mut y);
        let dbp = Dbp { forward: Some(forward), steps_per_span: 115, ..Dbp::new(m, 80.0, 1, 1.0) };
        assert_eq!(dbp.step_km(), forward.step_km);
        let z = dbp.run(&y, 64e9).unwrap();
        assert!(x.iter().zip(&z).all(|(a, b)| (a - b).norm() < 1e-12));
        // another number of steps: the span divided by the steps
        assert_eq!(Dbp { steps_per_span: 4, ..dbp }.step_km(), 20.0);
    }

    #[test]
    fn manakov_dbp_undoes_the_nonlinearity_of_two_polarisations() {
        let m = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.04e-23, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let n = 1024;
        let w = |k: usize, f: f64, ph: f64| (2.0 * std::f64::consts::PI * f * k as f64 / n as f64 + ph).sin();
        let x0: Vec<C64> = (0..n).map(|k| C64::new(w(k, 31.0, 0.2), w(k, 17.0, 1.0)) * 0.1).collect();
        let y0: Vec<C64> = (0..n).map(|k| C64::new(w(k, 23.0, 2.0), w(k, 41.0, 0.5)) * 0.1).collect();
        let (mut x, mut y) = (x0.clone(), y0.clone());
        let fwd = ManakovSplitStep {
            rule: StepRule::Fixed(0.1),
            max_iterations: 20,
            tolerance: 1e-12,
            ..ManakovSplitStep::new(m)
        };
        let fs = 128e9;
        let g = (m.alpha_np_km / 2.0 * 80.0).exp();
        for _ in 0..2 {
            fwd.run(&mut x, &mut y, 80.0, fs).unwrap();
            for v in x.iter_mut().chain(y.iter_mut()) {
                *v *= g;
            }
        }
        let err = |xi: f64, steps: usize| {
            let (mut u, mut v) = (x.clone(), y.clone());
            Dbp { steps_per_span: steps, xi, ..Dbp::new(m, 80.0, 2, g) }.run_dual(&mut u, &mut v, fs).unwrap();
            let e: f64 = u.iter().zip(&x0).chain(v.iter().zip(&y0)).map(|(a, b)| (a - b).norm_sqr()).sum();
            let p: f64 = x0.iter().chain(&y0).map(C64::norm_sqr).sum();
            (e / p).sqrt()
        };
        let (linear, full) = (err(0.0, 800), err(1.0, 800));
        assert!(full < 1e-3 && full < linear / 50.0, "{full} vs {linear}");
    }

    /// A span of 80 km at 6 dBm back-propagated with few steps: one logarithmic step is
    /// the uniform step, and with 2 and 4 steps the logarithmic steps leave less error
    /// (at the full nonlinear phase, xi = 1); with 16 the uniform steps are better (the
    /// last logarithmic steps are long: Sinkin et al. 2003 find the method inefficient
    /// in general). Measured: 2 steps 0.070 against 0.090, 4 steps 0.032 against 0.045,
    /// 16 steps 0.0058 against 0.0015.
    #[test]
    fn logarithmic_steps_back_propagate_a_span_better_with_few_steps_only() {
        use comnum_core::rng::Rng;
        let alpha = 0.2 / (10.0 * std::f64::consts::E.log10());
        let m = Medium { alpha_np_km: alpha, beta2_s2_km: -21.683e-24, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let (n, fs) = (4096usize, 128e9);
        let mut rng = Rng::new(5);
        // a band-limited random field (32 GHz) at 6 dBm
        let plan = comnum_math::fft::plan(n);
        let mut x: Vec<C64> = (0..n).map(|_| rng.complex_gaussian()).collect();
        plan.forward(&mut x);
        for (k, v) in x.iter_mut().enumerate() {
            if (k.min(n - k) as f64) / n as f64 * fs > 16e9 {
                *v = C64::new(0.0, 0.0);
            }
        }
        plan.inverse(&mut x);
        let pw = x.iter().map(C64::norm_sqr).sum::<f64>() / n as f64;
        let scale = (10f64.powf(0.6) * 1e-3 / pw).sqrt();
        for v in &mut x {
            *v *= scale;
        }
        let mut y = x.clone();
        ManakovSplitStep { rule: StepRule::Fixed(0.01), ..ManakovSplitStep::new(m) }
            .run_fields(&mut [&mut y], 80.0, fs, &comnum_math::nlse::PropagationOptions::default())
            .unwrap();
        let g = (alpha * 40.0).exp();
        for v in &mut y {
            *v *= g;
        }
        let residual = |steps: usize, distribution: StepDistribution| {
            let z = Dbp { steps_per_span: steps, distribution, ..Dbp::new(m, 80.0, 1, g) }.run(&y, fs).unwrap();
            let e: f64 = z.iter().zip(&x).map(|(a, b)| (a - b).norm_sqr()).sum();
            (e / x.iter().map(C64::norm_sqr).sum::<f64>()).sqrt()
        };
        let (u1, l1) = (residual(1, StepDistribution::Uniform), residual(1, StepDistribution::Logarithmic));
        assert!((u1 - l1).abs() < 1e-12 * u1, "{u1} {l1}");
        for steps in [2, 4] {
            let (u, l) = (residual(steps, StepDistribution::Uniform), residual(steps, StepDistribution::Logarithmic));
            assert!(l < 0.85 * u, "{steps} steps: logarithmic {l}, uniform {u}");
        }
        let (u16, l16) = (residual(16, StepDistribution::Uniform), residual(16, StepDistribution::Logarithmic));
        assert!(u16 < l16, "16 steps: uniform {u16}, logarithmic {l16}");
        assert_eq!("logarithmic".parse::<StepDistribution>().unwrap(), StepDistribution::Logarithmic);
        assert!("log".parse::<StepDistribution>().is_err());
    }

    /// The nonlinear length `Step` gives a step the phase of the power at its middle
    /// times the step (Häger & Pfister's convention), `Effective` the power integrated over
    /// the step: on a constant field without dispersion, one step of 80 km, the two phases
    /// are in the ratio 80 / (2 sinh(40 alpha) / alpha).
    #[test]
    fn the_step_nonlinear_length_is_the_midpoint_rule() {
        let alpha = 0.2 / (10.0 * std::f64::consts::E.log10());
        let m = Medium { alpha_np_km: alpha, beta2_s2_km: 0.0, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let x = vec![C64::new(1e-3, 0.0); 64];
        let phase = |nl: NonlinearLength| {
            let d = Dbp { nonlinear_length: nl, ..Dbp::new(m, 80.0, 1, 1.0) };
            d.run(&x, 64e9).unwrap()[0].arg()
        };
        let (effective, step) = (phase(NonlinearLength::Effective), phase(NonlinearLength::Step));
        let ratio = 80.0 / (2.0 * (alpha * 40.0).sinh() / alpha);
        assert!(effective < 0.0 && effective.abs() < 0.1, "{effective}");
        assert!((step / effective - ratio).abs() < 1e-9, "{} against {ratio}", step / effective);
        assert_eq!("step".parse::<NonlinearLength>().unwrap(), NonlinearLength::Step);
        assert!("midpoint".parse::<NonlinearLength>().is_err());
    }
}
