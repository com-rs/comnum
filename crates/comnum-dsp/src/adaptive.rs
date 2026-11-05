//! The pieces shared by the adaptive algorithms: error laws and update rules of the
//! adaptive filters, loop filters of the tracking loops.
//!
//! # Adaptive filters
//!
//! An adaptive transversal filter $y = \underline{h}^{T} \underline{x}$ descends a criterion
//! by the stochastic gradient
//!
//! $$
//! \underline{h} \leftarrow \underline{h} + \mu_{k}\, g(e)\, \underline{x}^{*},
//! $$
//!
//! where two independent choices fix the algorithm:
//!
//! - the **error law** ([`ErrorLaw`]) says what is minimised, the error $e$ of the
//!   output $y$: the constant modulus $(R - |y|^{2})\, y$ ([`ConstantModulus`], Godard),
//!   the nearest radius $(\hat{R}^{2} - |y|^{2})\, y$ ([`RadiusDirected`]), a known
//!   symbol $s - y$ ([`Reference`]) or the decision $\hat{s} - y$ ([`Decision`]);
//! - the **update rule** ([`UpdateRule`]) says how far to move: a constant step
//!   ($\mu_{k} = \mu$, LMS), a step normalised by the energy of the input
//!   ($\mu / \|\underline{x}\|^{2}$, NLMS), or the sign of the error ($g(e) =
//!   \operatorname{sgn} \mathrm{Re}\, e + j \operatorname{sgn} \mathrm{Im}\, e$, sign-error
//!   LMS, of cheaper hardware), the error itself otherwise.
//!
//! [`crate::equalizer::AdaptiveEqualizer`] runs any law with any rule on its `n × n`
//! butterfly ([`crate::equalizer::AdaptiveEqualizer::run_with`]); its [`Algorithm`]s are
//! the usual pairs: CMA and RDE with LMS, NLMS on a reference, DD-LMS.
//!
//! # Tracking loops
//!
//! A tracking loop (carrier phase, symbol timing) feeds the output $e_{k}$ of its error
//! detector to a [`LoopFilter`] whose output steers its oscillator: the
//! proportional-integral filter $u_{k} = k_{p} e_{k} + \sum_{i \le k} k_{i} e_{i}$
//! ([`ProportionalIntegral`], the Gardner loop of [`crate::sync::timing`]) or its bilinear
//! (Tustin) discretisation $u_{k} = u_{k-1} + b_{1} e_{k-1} + b_{2} e_{k}$
//! ([`BilinearPi`], the decision-directed PLL of [`crate::sync::carrier::ddpll`]).
//!
//! # Examples
//! ```
//! use comnum_dsp::adaptive::{ConstantModulus, ErrorLaw, LoopFilter, ProportionalIntegral, UpdateRule};
//! use comnum_math::C64;
//! // the CMA error vanishes on the circle of radius sqrt(R)
//! let cma = ConstantModulus { radius: 1.0 };
//! assert_eq!(cma.error(C64::new(0.0, 1.0), 0, 0).1, 0.0);
//! // one LMS step towards the error
//! let mut h = vec![C64::new(0.0, 0.0); 2];
//! UpdateRule::Lms.update(&mut h, &[C64::new(1.0, 0.0), C64::new(0.0, 1.0)], C64::new(0.5, 0.0), 0.1);
//! assert_eq!(h, vec![C64::new(0.05, 0.0), C64::new(0.0, -0.05)]);
//! // a PI filter integrates a constant error
//! let mut f = ProportionalIntegral::new(0.1, 0.01);
//! assert!((f.update(1.0) - 0.11).abs() < 1e-15 && (f.update(1.0) - 0.12).abs() < 1e-15);
//! ```
//!
//! # References
//!
//! 1. D. N. Godard, "Self-recovering equalization and carrier tracking in
//!    two-dimensional data communication systems," *IEEE Trans. Commun.*, 28(11),
//!    1867–1875, 1980.
//! 2. S. Haykin, *Adaptive Filter Theory*, 5th ed., Pearson, 2014, chs. 5–7 (LMS, NLMS)
//!    and the sign-error algorithm.
//! 3. F. M. Gardner, *Phaselock Techniques*, 3rd ed., Wiley, 2005 (loop filters).

use crate::equalizer::Algorithm;
use comnum_core::{Result, ensure};
use comnum_math::C64;
use comnum_math::nearest::Slicer;
use comnum_math::stats::Moments;

/// What an adaptive filter minimises: the error of an output.
pub trait ErrorLaw {
    /// The error $e$ of the output `y` of stream `stream` at symbol `k`, and the squared
    /// error reported (the criterion at this output).
    fn error(&self, y: C64, stream: usize, k: usize) -> (C64, f64);
}

/// The constant modulus criterion (Godard): $e = (R - |y|^{2})\, y$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstantModulus {
    /// The modulus $R = E|s|^{4}/E|s|^{2}$ of the constellation.
    pub radius: f64,
}

impl ConstantModulus {
    /// The criterion of a constellation (any scale: normalised to unit energy inside).
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for an empty constellation.
    pub fn for_constellation(points: &[C64]) -> Result<ConstantModulus> {
        ensure(!points.is_empty(), || "empty constellation".into())?;
        let m = Moments::sample(&crate::signal::pnorm(points));
        Ok(ConstantModulus { radius: m.m42 / m.m21 })
    }
}

impl ErrorLaw for ConstantModulus {
    fn error(&self, y: C64, _: usize, _: usize) -> (C64, f64) {
        let d = self.radius - y.norm_sqr();
        (y * d, d * d)
    }
}

/// The radius-directed criterion: $e = (\hat{R}^{2} - |y|^{2})\, y$ with $\hat{R}$ the
/// radius of the constellation nearest to $|y|$ (multi-ring constellations).
#[derive(Clone, Debug, PartialEq)]
pub struct RadiusDirected {
    /// The radii of the constellation, increasing.
    pub radii: Vec<f64>,
}

impl RadiusDirected {
    /// The criterion of a constellation (normalised to unit energy inside).
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for an empty constellation.
    pub fn for_constellation(points: &[C64]) -> Result<RadiusDirected> {
        ensure(!points.is_empty(), || "empty constellation".into())?;
        let mut radii: Vec<f64> = crate::signal::pnorm(points).iter().map(|p| p.norm()).collect();
        radii.sort_by(f64::total_cmp);
        radii.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        Ok(RadiusDirected { radii })
    }
}

impl ErrorLaw for RadiusDirected {
    fn error(&self, y: C64, _: usize, _: usize) -> (C64, f64) {
        let r = y.norm();
        let rr = self.radii.iter().copied().min_by(|a, b| (a - r).abs().total_cmp(&(b - r).abs())).unwrap_or(1.0);
        let d = rr * rr - y.norm_sqr();
        (y * d, d * d)
    }
}

/// The data-aided criterion: $e = s - y$ with the known symbols $s$ of every stream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reference<'a> {
    /// The transmitted symbols of every stream.
    pub symbols: &'a [Vec<C64>],
}

impl ErrorLaw for Reference<'_> {
    fn error(&self, y: C64, stream: usize, k: usize) -> (C64, f64) {
        let d = self.symbols[stream][k] - y;
        (d, d.norm_sqr())
    }
}

/// The decision-directed criterion: $e = \hat{s} - y$ with $\hat{s}$ the point of the
/// constellation nearest to $y$.
#[derive(Clone, Debug)]
pub struct Decision {
    points: Vec<C64>,
    slicer: Slicer,
}

impl Decision {
    /// The criterion of a constellation (normalised to unit energy inside).
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for an empty constellation.
    pub fn for_constellation(points: &[C64]) -> Result<Decision> {
        ensure(!points.is_empty(), || "empty constellation".into())?;
        let points = crate::signal::pnorm(points);
        Ok(Decision { slicer: Slicer::new(&points), points })
    }
}

impl ErrorLaw for Decision {
    fn error(&self, y: C64, _: usize, _: usize) -> (C64, f64) {
        let d = self.points[self.slicer.nearest(y)] - y;
        (d, d.norm_sqr())
    }
}

/// How an adaptive filter moves its taps along the error (see the module).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UpdateRule {
    /// A constant step: $\underline{h} \leftarrow \underline{h} + \mu e \underline{x}^{*}$.
    #[default]
    Lms,
    /// A step normalised by the energy of the input: $\mu / \|\underline{x}\|^{2}$.
    Normalized,
    /// The sign of the error: $\mu\, (\operatorname{sgn}\mathrm{Re}\,e + j
    /// \operatorname{sgn}\mathrm{Im}\,e)\, \underline{x}^{*}$.
    SignError,
}

impl std::str::FromStr for UpdateRule {
    type Err = comnum_core::Error;

    /// `lms`, `nlms` (`normalized`) or `sign` (`sign-error`), any case.
    fn from_str(s: &str) -> Result<UpdateRule> {
        match s.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "lms" => Ok(UpdateRule::Lms),
            "nlms" | "normalized" | "normalised" => Ok(UpdateRule::Normalized),
            "sign" | "signerror" => Ok(UpdateRule::SignError),
            other => Err(comnum_core::Error::Config(format!("unknown update rule {other:?} (lms, nlms, sign)"))),
        }
    }
}

/// The floor of the input energy of a normalised step (an input of zeros).
const NORMALIZED_FLOOR: f64 = 1e-300;

impl UpdateRule {
    /// The step for an input of energy `energy` (used by [`UpdateRule::Normalized`] only).
    pub fn step_size(self, mu: f64, energy: f64) -> f64 {
        match self {
            UpdateRule::Normalized => mu / energy.max(NORMALIZED_FLOOR),
            UpdateRule::Lms | UpdateRule::SignError => mu,
        }
    }

    /// The error the taps follow: `e`, or its sign for [`UpdateRule::SignError`].
    pub fn driving_error(self, e: C64) -> C64 {
        match self {
            UpdateRule::SignError => C64::new(sign(e.re), sign(e.im)),
            UpdateRule::Lms | UpdateRule::Normalized => e,
        }
    }

    /// Whether the step needs the energy of the input.
    pub fn needs_energy(self) -> bool {
        self == UpdateRule::Normalized
    }

    /// One step of the taps along the input `x` (its first `taps.len()` samples):
    /// $h_{t} \leftarrow h_{t} + g(e)\, x_{t}^{*}\, \mu_{k}$ with the step `step`
    /// ([`UpdateRule::step_size`]).
    pub fn update(self, taps: &mut [C64], x: &[C64], e: C64, step: f64) {
        let g = self.driving_error(e);
        for (t, xv) in taps.iter_mut().zip(x) {
            *t += g * xv.conj() * step;
        }
    }
}

/// The sign of `v` (0 for 0).
fn sign(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

impl Algorithm {
    /// The update rule of the algorithm: [`UpdateRule::Normalized`] for `Nlms`, LMS for the
    /// others.
    pub fn update_rule(self) -> UpdateRule {
        if self == Algorithm::Nlms { UpdateRule::Normalized } else { UpdateRule::Lms }
    }
}

/// The loop filter of a tracking loop: the error of the detector in, the control of the
/// oscillator out.
pub trait LoopFilter {
    /// The control $u_{k}$ after the error $e_{k}$.
    fn update(&mut self, e: f64) -> f64;
}

/// The proportional-integral filter $u_{k} = k_{p} e_{k} + I_{k}$, $I_{k} = I_{k-1} +
/// k_{i} e_{k}$ (the Gardner loop).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProportionalIntegral {
    /// Proportional gain $k_{p}$.
    pub kp: f64,
    /// Integral gain $k_{i}$.
    pub ki: f64,
    integral: f64,
}

impl ProportionalIntegral {
    /// The filter of gains `kp`, `ki`, its integrator at 0.
    pub fn new(kp: f64, ki: f64) -> ProportionalIntegral {
        ProportionalIntegral { kp, ki, integral: 0.0 }
    }
}

impl LoopFilter for ProportionalIntegral {
    fn update(&mut self, e: f64) -> f64 {
        self.integral += self.ki * e;
        self.kp * e + self.integral
    }
}

/// The bilinear (Tustin) proportional-integral filter $u_{k} = u_{k-1} + b_{1} e_{k-1} +
/// b_{2} e_{k}$ (the decision-directed PLL).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BilinearPi {
    /// The coefficient of the previous error $b_{1}$.
    pub b1: f64,
    /// The coefficient of the current error $b_{2}$.
    pub b2: f64,
    previous: f64,
    output: f64,
}

impl BilinearPi {
    /// The filter of coefficients `b1`, `b2`, at rest.
    pub fn new(b1: f64, b2: f64) -> BilinearPi {
        BilinearPi { b1, b2, previous: 0.0, output: 0.0 }
    }

    /// The bilinear transform of the active PI filter $(1 + s\tau_{2})/(s\tau_{1})$ at the
    /// sampling period `ts`: $b_{1,2} = \frac{T_{s}}{2\tau_{1}}(1 \mp
    /// \cot\frac{T_{s}}{2\tau_{2}})$ (OptiCommPy `ddpll`).
    pub fn active_pi(ts: f64, tau1: f64, tau2: f64) -> BilinearPi {
        let cot = 1.0 / (ts / (2.0 * tau2)).tan();
        BilinearPi::new(ts / (2.0 * tau1) * (1.0 - cot), ts / (2.0 * tau1) * (1.0 + cot))
    }
}

impl LoopFilter for BilinearPi {
    fn update(&mut self, e: f64) -> f64 {
        self.output += self.b1 * self.previous + self.b2 * e;
        self.previous = e;
        self.output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_laws_vanish_on_their_targets() {
        let qpsk: Vec<C64> =
            [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)].iter().map(|&(a, b)| C64::new(a, b)).collect();
        let p = crate::signal::pnorm(&qpsk);
        let cma = ConstantModulus::for_constellation(&qpsk).unwrap();
        let rde = RadiusDirected::for_constellation(&qpsk).unwrap();
        let dd = Decision::for_constellation(&qpsk).unwrap();
        for v in &p {
            assert!(cma.error(*v, 0, 0).1 < 1e-24 && rde.error(*v, 0, 0).1 < 1e-24 && dd.error(*v, 0, 0).1 < 1e-24);
        }
        let s = vec![vec![p[2]]];
        assert_eq!(Reference { symbols: &s }.error(p[2], 0, 0), (C64::new(0.0, 0.0), 0.0));
        assert!(
            ConstantModulus::for_constellation(&[]).is_err()
                && Decision::for_constellation(&[]).is_err()
                && RadiusDirected::for_constellation(&[]).is_err()
        );
    }

    #[test]
    fn the_update_rules_scale_and_sign_the_update() {
        assert_eq!(UpdateRule::Normalized.step_size(0.1, 4.0), 0.025);
        assert_eq!(UpdateRule::Lms.step_size(0.1, 4.0), 0.1);
        assert_eq!(UpdateRule::SignError.driving_error(C64::new(-0.3, 2.0)), C64::new(-1.0, 1.0));
        assert!(UpdateRule::Normalized.needs_energy() && !UpdateRule::SignError.needs_energy());
        assert_eq!("Sign-Error".parse::<UpdateRule>().unwrap(), UpdateRule::SignError);
        assert!("rls".parse::<UpdateRule>().is_err());
    }

    #[test]
    fn the_bilinear_filter_is_the_ddpll_recursion() {
        let (b1, b2) = (0.2, 0.3);
        let mut f = BilinearPi::new(b1, b2);
        let e = [1.0, -0.5, 0.25];
        let (mut uf, mut prev) = (0.0, 0.0);
        for &v in &e {
            uf += b1 * prev + b2 * v;
            prev = v;
            assert_eq!(f.update(v), uf);
        }
    }
}
