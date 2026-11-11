//! The symmetric split-step Fourier solvers of the nonlinear Schrödinger equation:
//! scalar ([`SplitStep`], [`SplitStepF32`], both built from the [`Steps`] of a span)
//! and dual-polarisation Manakov ([`ManakovSplitStep`]).
//!
//! The same solver propagates a field through a fibre (`comnum_channel::optical::fiber`)
//! and back-propagates it at the receiver (`comnum_dsp::compensation::dbp`, with the
//! coefficients negated), so it lives here, below both.
//!
//! # Model
//!
//! $$
//! \frac{\partial A}{\partial z} = -\frac{\alpha}{2} A
//!   - j \frac{\beta_{2}}{2} \frac{\partial^{2} A}{\partial t^{2}}
//!   + j \gamma |A|^{2} A
//!     $$
//!
//! Each step of length $h$ is a half linear step, the nonlinear phase
//! $\gamma |A|^{2} h_{\mathrm{NL}}$ and a half linear step, with the linear operator applied
//! in the frequency domain:
//!
//! $$
//! \tilde{A}(\omega) \leftarrow \tilde{A}(\omega)\\, e^{-\alpha h / 4}\\, e^{j \beta_{2} \omega^{2} h / 4}
//! $$
//!
//! The nonlinear length $h_{\mathrm{NL}}$ is the step itself for the forward model; a
//! back-propagation with coarse steps uses the effective length of the step instead.
//!
//! # The Manakov equation
//!
//! Averaged over the fast random rotations of the birefringence, the two polarisations
//! $A_{x}$, $A_{y}$ obey
//!
//! $$
//! \frac{\partial A_{x,y}}{\partial z} = -\frac{\alpha}{2} A_{x,y} -
//!   j \frac{\beta_{2}}{2} \frac{\partial^{2} A_{x,y}}{\partial t^{2}} +
//!   j \frac{8}{9} \gamma \left(|A_{x}|^{2} + |A_{y}|^{2}\right) A_{x,y}.
//! $$
//!
//! [`ManakovSplitStep`] is the symmetric scheme of OptiCommPy's `manakovSSF`: a half
//! linear step, the nonlinear phase
//!
//! $$
//! \phi(t) = \frac{8}{9} \gamma\, \frac{P(z, t) + P(z + h, t)}{2}\, h, \qquad
//! P = |A_{x}|^{2} + |A_{y}|^{2},
//! $$
//!
//! (the trapezoidal rule of OptiCommPy, iterated until the step converges since
//! $P(z + h)$ depends on the step itself, [`NonlinearStep::Trapezoidal`]; by default the
//! power at the middle of the step, [`NonlinearStep::Midpoint`], as [`SplitStep`]; or
//! Optilux's scheme, the power at the start of the nonlinear block over the effective
//! length of the step, [`NonlinearStep::EffectiveLength`]) and a
//! half linear step. The step is fixed, adapted so
//! that the largest nonlinear phase of a step is a given rotation (the nonlinear phase
//! rotation method of Sinkin et al., which concentrates the steps where the power is
//! high, at the beginning of the spans), chosen by the four-wave-mixing rule of Musetti
//! et al., or distributed logarithmically ([`logarithmic_steps`], Bosco et al.: every step
//! carries the same integrated power). The logarithmic steps keep the spurious
//! four-wave mixing of the split step near $1/K$ of the true one; with few steps per span
//! they also place the nonlinear phase of a back-propagation better, with many they leave
//! more error than equal steps (the long last steps).
//!
//! # Examples
//! ```
//! use comnum_math::nlse::{Medium, SplitStep, Steps};
//! use comnum_math::C64;
//! // without loss and nonlinearity, forward then negated propagation is the identity
//! let m = Medium { alpha_np_km: 0.0, beta2_s2_km: -2.17e-23, gamma_w_km: 0.0, beta3_s3_km: 0.0 };
//! let x: Vec<C64> = (0..64).map(|n| C64::new((n % 3) as f64, (n % 2) as f64)).collect();
//! let mut e = x.clone();
//! SplitStep::new(&m, Steps::new(8, 10.0), 64, 64e9).run(&mut e);
//! SplitStep::new(&m.negated(), Steps::new(8, 10.0), 64, 64e9).run(&mut e);
//! assert!(x.iter().zip(&e).all(|(a, b)| (a - b).norm() < 1e-9));
//! ```
//!
//! # References
//!
//! 1. G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed., Academic Press, 2013.
//! 2. E. Ip and J. M. Kahn, "Compensation of dispersion and nonlinear impairments
//!    using digital backpropagation," *J. Lightw. Technol.*, 26(20), 2008.
//! 3. D. Marcuse, C. R. Menyuk and P. K. A. Wai, "Application of the Manakov-PMD
//!    equation to studies of signal propagation in optical fibers with randomly varying
//!    birefringence," *J. Lightw. Technol.*, 15(9), 1735–1746, 1997.
//! 4. O. V. Sinkin, R. Holzlöhner, J. Zweck and C. R. Menyuk, "Optimization of the
//!    split-step Fourier method in modeling optical-fiber communications systems," *J.
//!    Lightw. Technol.*, 21(1), 61–68, 2003.
//! 5. G. Bosco, A. Carena, V. Curri, R. Gaudino, P. Poggiolini and S. Benedetto,
//!    "Suppression of spurious tones induced by the split-step method in fiber systems
//!    simulation," *IEEE Photon. Technol. Lett.*, 12(5), 489–491, 2000.
//! 6. J. Zhang, X. Li and Z. Dong, "Digital nonlinear compensation based on the modified
//!    logarithmic step size," *J. Lightw. Technol.*, 31(22), 3546–3555, 2013 (not in the
//!    references: its eq. (2) as Häger & Pfister describe it).
//! 7. C. Häger and H. D. Pfister, "Physics-based deep learning for fiber-optic
//!    communication systems," *IEEE J. Sel. Areas Commun.*, 39(1), 280–294, 2021.
//! 8. J. Shao, X. Liang and S. Kumar, "Comparison of split-step Fourier schemes for
//!    simulating fiber optic communication systems," *IEEE Photonics J.*, 6(4),
//!    7200515, 2014 (the scheme of Optilux's `fiber`, [`NonlinearStep::EffectiveLength`]).

use crate::fft::{self, Fft};
use crate::{C64, cis};
use comnum_core::{Error, Result};
use num_complex::Complex32 as C32;

mod manakov;

pub use manakov::{Birefringence, ManakovSplitStep, PropagationOptions};

/// The coefficients of the scalar NLSE, per kilometre.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Medium {
    /// Power attenuation `α` (Np/km).
    pub alpha_np_km: f64,
    /// Group-velocity dispersion `β₂` (s²/km).
    pub beta2_s2_km: f64,
    /// Nonlinear coefficient `γ` (1/(W·km)).
    pub gamma_w_km: f64,
    /// Third-order dispersion `β₃` (s³/km), 0 for none.
    pub beta3_s3_km: f64,
}

impl Default for Medium {
    /// No loss, no dispersion, no nonlinearity.
    fn default() -> Medium {
        Medium { alpha_np_km: 0.0, beta2_s2_km: 0.0, gamma_w_km: 0.0, beta3_s3_km: 0.0 }
    }
}

impl Medium {
    /// The medium with `α`, `β₂` and `γ` reversed: propagating through it undoes a
    /// propagation through `self` (digital back-propagation).
    pub fn negated(&self) -> Medium {
        Medium {
            alpha_np_km: -self.alpha_np_km,
            beta2_s2_km: -self.beta2_s2_km,
            gamma_w_km: -self.gamma_w_km,
            beta3_s3_km: -self.beta3_s3_km,
        }
    }

    /// The linear response `e^{-αL/2} e^{j(β₂ω²/2 + β₃ω³/6)L}` of `length_km` of the
    /// medium at every angular frequency (loss and dispersion of the field).
    pub fn linear_response(&self, omega: &[f64], length_km: f64) -> Vec<C64> {
        let mut out = vec![C64::new(0.0, 0.0); omega.len()];
        self.linear_response_into(omega, length_km, &mut out);
        out
    }

    /// [`Medium::linear_response`] into `out` (one value per angular frequency).
    fn linear_response_into(&self, omega: &[f64], length_km: f64, out: &mut [C64]) {
        let (a, b2, b3) = (self.alpha_np_km, self.beta2_s2_km, self.beta3_s3_km);
        let g = (-(a / 2.0) * length_km).exp();
        if b3 == 0.0 {
            for (o, wk) in out.iter_mut().zip(omega) {
                *o = g * cis(b2 / 2.0 * wk * wk * length_km);
            }
        } else {
            for (o, wk) in out.iter_mut().zip(omega) {
                *o = g * cis((b2 / 2.0 + b3 / 6.0 * wk) * wk * wk * length_km);
            }
        }
    }

    /// The half linear step `e^{-αh/4} e^{jβ₂ω²h/4}` at every angular frequency.
    fn half_step(&self, omega: &[f64], h: f64) -> Vec<C64> {
        self.linear_response(omega, h / 2.0)
    }

    /// [`Medium::half_step`] into `out`.
    fn half_step_into(&self, omega: &[f64], h: f64, out: &mut [C64]) {
        self.linear_response_into(omega, h / 2.0, out);
    }
}

/// The steps of one span: `n_steps` symmetric steps of `step_km`, the nonlinear phase
/// of each over `nl_length_km` (the step itself for the forward model, the effective
/// length of a coarse step for a back-propagation). Shared by [`SplitStep`] and
/// [`SplitStepF32`], and given by a forward model to its back-propagation
/// (`comnum_dsp::compensation::dbp::Dbp::forward`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Steps {
    /// Number of steps.
    pub n_steps: usize,
    /// Length of a step (km).
    pub step_km: f64,
    /// Nonlinear length of a step (km).
    pub nl_length_km: f64,
}

impl Steps {
    /// The steps of the forward model: `n_steps` of `step_km`, whose nonlinear length is
    /// the step.
    pub fn new(n_steps: usize, step_km: f64) -> Steps {
        Steps { n_steps, step_km, nl_length_km: step_km }
    }

    /// The length covered, $n h$ (km).
    pub fn length_km(&self) -> f64 {
        self.n_steps as f64 * self.step_km
    }
}

/// The [`Steps`] of one span in double precision, or steps of different lengths
/// ([`SplitStep::with_lengths`]).
#[derive(Clone, Debug)]
pub struct SplitStep {
    /// The half linear steps, one per distinct step length.
    lins: Vec<Vec<C64>>,
    /// Every step in order: the index of its half linear step, its length and its
    /// nonlinear length (km).
    schedule: Vec<(usize, f64, f64)>,
    /// `None` for an empty block (nothing to propagate).
    plan: Option<Fft>,
    gamma_w_km: f64,
    steps: Steps,
}

impl SplitStep {
    /// The `steps` through `medium` for a block of `n` samples at `fs` (no plan for an
    /// empty block: the steps then leave it empty).
    pub fn new(medium: &Medium, steps: Steps, n: usize, fs: f64) -> SplitStep {
        let omega = fft::angular_frequencies(n, fs);
        SplitStep {
            lins: vec![medium.half_step(&omega, steps.step_km)],
            schedule: vec![(0, steps.step_km, steps.nl_length_km); steps.n_steps],
            plan: (n > 0).then(|| fft::plan(n)),
            gamma_w_km: medium.gamma_w_km,
            steps,
        }
    }

    /// Steps of different lengths: `steps` holds, in the order they are applied, the
    /// length of every step and its nonlinear length (km), such as the logarithmic steps
    /// of [`logarithmic_steps`] with their effective lengths. [`SplitStep::steps`] then
    /// gives their number, their mean length and their mean nonlinear length.
    pub fn with_lengths(medium: &Medium, steps: &[(f64, f64)], n: usize, fs: f64) -> SplitStep {
        let omega = fft::angular_frequencies(n, fs);
        let mut lengths: Vec<f64> = Vec::new();
        let mut lins = Vec::new();
        let mut schedule = Vec::with_capacity(steps.len());
        for &(h, nl) in steps {
            let i = lengths.iter().position(|&l| l == h).unwrap_or_else(|| {
                lengths.push(h);
                lins.push(medium.half_step(&omega, h));
                lengths.len() - 1
            });
            schedule.push((i, h, nl));
        }
        let k = steps.len().max(1) as f64;
        let total: f64 = steps.iter().map(|s| s.0).sum();
        let total_nl: f64 = steps.iter().map(|s| s.1).sum();
        SplitStep {
            lins,
            schedule,
            plan: (n > 0).then(|| fft::plan(n)),
            gamma_w_km: medium.gamma_w_km,
            steps: Steps { n_steps: steps.len(), step_km: total / k, nl_length_km: total_nl / k },
        }
    }

    /// The steps (for [`SplitStep::with_lengths`], their number and mean lengths).
    pub fn steps(&self) -> Steps {
        self.steps
    }

    /// Propagate `e` (√W, time domain) through the steps, in place.
    ///
    /// # Panics
    /// If `e` has another length than the block of [`SplitStep::new`].
    pub fn run(&self, e: &mut [C64]) {
        let Some(p) = self.plan.as_ref() else {
            assert!(e.is_empty(), "a field for an empty block");
            return;
        };
        let g = self.gamma_w_km;
        p.forward(e);
        for &(i, _, h) in &self.schedule {
            mul(e, &self.lins[i]);
            p.inverse(e);
            for v in e.iter_mut() {
                *v *= cis(g * v.norm_sqr() * h);
            }
            p.forward(e);
            mul(e, &self.lins[i]);
        }
        p.inverse(e);
    }

    /// The steps for the Manakov equation: both polarisations (√W, time domain, equal
    /// lengths) in place, with the nonlinear phase
    /// $\frac{8}{9} \gamma (|A_{x}|^{2} + |A_{y}|^{2}) h_{\mathrm{NL}}$ at the middle of every
    /// step (the back-propagation of a dual-polarisation link).
    ///
    /// # Panics
    /// If the polarisations have different lengths, or another length than the block of
    /// [`SplitStep::new`].
    pub fn run_dual(&self, x: &mut [C64], y: &mut [C64]) {
        assert_eq!(x.len(), y.len(), "the polarisations must have the same length");
        let Some(p) = self.plan.as_ref() else {
            assert!(x.is_empty(), "a field for an empty block");
            return;
        };
        p.forward(x);
        p.forward(y);
        for &(i, _, h) in &self.schedule {
            mul(x, &self.lins[i]);
            mul(y, &self.lins[i]);
            p.inverse(x);
            p.inverse(y);
            manakov_rotation(x, y, 8.0 / 9.0 * self.gamma_w_km * h);
            p.forward(x);
            p.forward(y);
            mul(x, &self.lins[i]);
            mul(y, &self.lins[i]);
        }
        p.inverse(x);
        p.inverse(y);
    }

    /// [`SplitStep::run`] with a distributed gain: before the nonlinear phase and after
    /// it, every half step also multiplies the spectrum by `gain(z_0, z_1, g)`, the gain
    /// of the field at every frequency (FFT order) from $z_{0}$ to $z_{1}$ (km, from the
    /// start of the steps). The medium then carries no attenuation: `gain` holds the
    /// loss as well (the power profile of a Raman-pumped span, `comnum_channel::optical::raman`).
    /// As in [`SplitStep::run`], the field stays in the frequency domain from the end of
    /// a step to the start of the next: two FFTs per step.
    ///
    /// # Panics
    /// If `e` has another length than the block of [`SplitStep::new`].
    pub fn run_with_gain(&self, e: &mut [C64], gain: FieldGain<'_>) {
        let Some(p) = self.plan.as_ref() else {
            assert!(e.is_empty(), "a field for an empty block");
            return;
        };
        let g = self.gamma_w_km;
        let mut buf = vec![0.0; e.len()];
        let mut z0 = 0.0;
        p.forward(e);
        for (k, &(i, step, h)) in self.schedule.iter().enumerate() {
            // the start of the step: k steps of the uniform steps (as they always were
            // computed), the sum of the steps otherwise
            if self.lins.len() == 1 {
                z0 = k as f64 * step;
            }
            mul(e, &self.lins[i]);
            gain(z0, z0 + step / 2.0, &mut buf);
            scale(e, &buf);
            p.inverse(e);
            for v in e.iter_mut() {
                *v *= cis(g * v.norm_sqr() * h);
            }
            p.forward(e);
            mul(e, &self.lins[i]);
            gain(z0 + step / 2.0, z0 + step, &mut buf);
            scale(e, &buf);
            z0 += step;
        }
        p.inverse(e);
    }
}

/// The gain of the field at every frequency bin (FFT order) between two positions of a
/// span (km): `gain(z_0, z_1, out)` writes $\sqrt{G(z_{1}, f) / G(z_{0}, f)}$ into `out`.
pub type FieldGain<'a> = &'a dyn Fn(f64, f64, &mut [f64]);

/// [`SplitStep`] in single precision: the field, the FFTs, the linear operator and the
/// nonlinear phase are computed in `f32` (about twice as fast), with the same [`Steps`]
/// (a back-propagation included); the plans and their work space are cached per thread
/// as those of [`fft::plan`].
///
/// The rounding error grows like `ε √K` with `ε ≈ 6·10⁻⁸` and `K` the number of
/// steps: below `10⁻⁴` relative for thousands of steps.
#[derive(Clone)]
pub struct SplitStepF32 {
    lin: Vec<C32>,
    lin_n: Vec<C32>,
    /// `None` for an empty block.
    plan: Option<fft::FftF32>,
    gamma_h: f32,
    steps: Steps,
}

impl std::fmt::Debug for SplitStepF32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SplitStepF32")
            .field("n", &self.lin.len())
            .field("gamma_h", &self.gamma_h)
            .field("steps", &self.steps)
            .finish_non_exhaustive()
    }
}

impl SplitStepF32 {
    /// The `steps` through `medium` for a block of `n` samples at `fs` (no plan for an
    /// empty block).
    pub fn new(medium: &Medium, steps: Steps, n: usize, fs: f64) -> SplitStepF32 {
        let omega = fft::angular_frequencies(n, fs);
        let half = medium.half_step(&omega, steps.step_km);
        // the 1/N of the inverse transform is folded into the first half step
        let inv_n = 1.0 / n.max(1) as f64;
        let lin = half.iter().map(|v| C32::new(v.re as f32, v.im as f32)).collect();
        let lin_n = half.iter().map(|v| v * inv_n).map(|v| C32::new(v.re as f32, v.im as f32)).collect();
        SplitStepF32 {
            lin,
            lin_n,
            plan: (n > 0).then(|| fft::plan_f32(n)),
            gamma_h: (medium.gamma_w_km * steps.nl_length_km) as f32,
            steps,
        }
    }

    /// The steps.
    pub fn steps(&self) -> Steps {
        self.steps
    }

    /// Propagate `e` through the steps, in place.
    ///
    /// # Panics
    /// If `e` has another length than the block of [`SplitStepF32::new`].
    pub fn run(&self, e: &mut [C32]) {
        let Some(p) = self.plan.as_ref() else {
            assert!(e.is_empty(), "a field for an empty block");
            return;
        };
        assert_eq!(e.len(), self.lin.len(), "FFT length mismatch");
        let n = e.len();
        p.forward(e);
        for _ in 0..self.steps.n_steps {
            e.iter_mut().zip(&self.lin_n).for_each(|(u, l)| *u *= l);
            p.inverse_unnormalized(e);
            for v in e.iter_mut() {
                let (sn, cs) = (self.gamma_h * v.norm_sqr()).sin_cos();
                *v *= C32::new(cs, sn);
            }
            p.forward(e);
            e.iter_mut().zip(&self.lin).for_each(|(u, l)| *u *= l);
        }
        p.inverse_unnormalized(e);
        let s = 1.0 / n as f32;
        for v in e.iter_mut() {
            *v *= s;
        }
    }
}

/// The step size of [`ManakovSplitStep`].
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
pub(crate) fn logarithmic_boundary(alpha_np_km: f64, length_km: f64, n: usize, k: usize) -> f64 {
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

/// The nonlinear step of [`ManakovSplitStep`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum NonlinearStep {
    /// The phase of the power at the middle of the step, after the first half of the
    /// dispersion, as [`SplitStep`] (eq. (3) of Sinkin et al. 2003): no iteration.
    #[default]
    Midpoint,
    /// OptiCommPy's `manakovSSF`: the phase of the trapezoidal mean of the power at the
    /// two ends of the step, iterated until the field converges. When channels walk off
    /// within a step, the mean smooths the intensity that drives the cross-phase
    /// modulation (see [`ManakovSplitStep`]).
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

/// The nonlinear step of the Manakov equation with a phase that does not change during
/// the step: both polarisations rotated by the same phase,
///
/// $$
/// A_{x,y}(t) \leftarrow A_{x,y}(t)\, e^{j k (|A_{x}(t)|^{2} + |A_{y}(t)|^{2})},
/// $$
///
/// with $k = \frac{8}{9} \gamma h_{\mathrm{NL}}$ (negated to undo it). The step of
/// [`SplitStep::run_dual`] and of the Manakov-PMD equation of
/// `comnum_channel::optical::pmd::ManakovPmd`.
///
/// # Panics
/// If the polarisations have different lengths.
///
/// # Examples
/// ```
/// use comnum_math::{C64, nlse::manakov_rotation};
/// let (mut x, mut y) = ([C64::new(1.0, 0.0)], [C64::new(0.0, 1.0)]);
/// manakov_rotation(&mut x, &mut y, std::f64::consts::FRAC_PI_4);
/// assert!((x[0] - C64::new(0.0, 1.0)).norm() < 1e-15 && (y[0] - C64::new(-1.0, 0.0)).norm() < 1e-15);
/// ```
pub fn manakov_rotation(x: &mut [C64], y: &mut [C64], k: f64) {
    assert_eq!(x.len(), y.len(), "the polarisations must have the same length");
    for (u, v) in x.iter_mut().zip(y.iter_mut()) {
        let e = cis(k * (u.norm_sqr() + v.norm_sqr()));
        *u *= e;
        *v *= e;
    }
}

fn mul(a: &mut [C64], b: &[C64]) {
    for (u, v) in a.iter_mut().zip(b) {
        *u *= v;
    }
}

fn scale(a: &mut [C64], b: &[f64]) {
    for (u, v) in a.iter_mut().zip(b) {
        *u *= v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_field_propagates_to_an_empty_field_instead_of_panicking() {
        let m = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let steps = Steps { n_steps: 3, step_km: 1.0, nl_length_km: 0.8 };
        let s = SplitStep::new(&m, steps, 0, 64e9);
        let (mut x, mut y): (Vec<C64>, Vec<C64>) = (Vec::new(), Vec::new());
        s.run(&mut x);
        s.run_dual(&mut x, &mut y);
        s.run_with_gain(&mut x, &|_, _, _| {});
        SplitStepF32::new(&m, steps, 0, 64e9).run(&mut []);
        assert!(x.is_empty() && y.is_empty() && s.steps() == steps);
        assert!((steps.length_km() - 3.0).abs() < 1e-15);
    }

    #[test]
    fn a_uniform_gain_of_the_loss_is_the_attenuation_of_the_medium() {
        // run_with_gain on the lossless medium with the field gain e^{-α Δz / 2} at every
        // frequency is run on the lossy medium (to the rounding of the factorisation)
        let m = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let lossless = Medium { alpha_np_km: 0.0, ..m };
        let x: Vec<C64> = (0..256).map(|n| cis(0.3 * n as f64) * (1e-2 * (1 + n % 5) as f64)).collect();
        let steps = Steps::new(40, 0.5);
        let (mut a, mut b) = (x.clone(), x);
        SplitStep::new(&m, steps, 256, 64e9).run(&mut a);
        let loss = |z0: f64, z1: f64, out: &mut [f64]| out.fill((-(0.046 / 2.0) * (z1 - z0)).exp());
        SplitStep::new(&lossless, steps, 256, 64e9).run_with_gain(&mut b, &loss);
        let peak = a.iter().map(|v| v.norm()).fold(0.0, f64::max);
        assert!(a.iter().zip(&b).all(|(u, v)| (u - v).norm() < 1e-13 * peak));
    }

    #[test]
    fn lossless_split_step_conserves_energy() {
        let m = Medium { alpha_np_km: 0.0, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let mut e: Vec<C64> = (0..256).map(|n| cis(0.1 * n as f64) * (1e-2 * (n % 7) as f64)).collect();
        let e0: f64 = e.iter().map(C64::norm_sqr).sum();
        SplitStep::new(&m, Steps::new(80, 1.0), 256, 64e9).run(&mut e);
        let e1: f64 = e.iter().map(C64::norm_sqr).sum();
        assert!((e1 / e0 - 1.0).abs() < 1e-12);
    }

    /// Two polarisations of about 30 GHz of bandwidth at 256 GS/s, of mean power 10 mW.
    fn dual_field(n: usize) -> (Vec<C64>, Vec<C64>) {
        let w = |k: usize, f: f64, ph: f64| (2.0 * std::f64::consts::PI * f * k as f64 / n as f64 + ph).sin();
        let x = (0..n)
            .map(|k| C64::new(w(k, 61.0, 0.3), w(k, 23.0, 1.1)) * (0.07 * (1.0 + 0.5 * w(k, 7.0, 0.0))))
            .collect();
        let y = (0..n)
            .map(|k| C64::new(w(k, 41.0, 2.0), w(k, 89.0, 0.4)) * (0.07 * (1.0 + 0.5 * w(k, 5.0, 1.0))))
            .collect();
        (x, y)
    }

    #[test]
    fn manakov_with_one_polarisation_is_the_scalar_equation_with_8_9_gamma() {
        let m = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let (x0, _) = dual_field(512);
        let (mut x, mut y) = (x0.clone(), vec![C64::new(0.0, 0.0); 512]);
        let s = ManakovSplitStep {
            rule: StepRule::Fixed(0.01),
            max_iterations: 20,
            tolerance: 1e-12,
            ..ManakovSplitStep::new(m)
        };
        s.run(&mut x, &mut y, 20.0, 128e9).unwrap();
        let mut e = x0;
        let scalar = Medium { gamma_w_km: 8.0 / 9.0 * 1.3, ..m };
        SplitStep::new(&scalar, Steps::new(2000, 0.01), 512, 128e9).run(&mut e);
        let err = x.iter().zip(&e).map(|(u, v)| (u - v).norm()).fold(0.0, f64::max);
        let peak = e.iter().map(|v| v.norm()).fold(0.0, f64::max);
        assert!(err < 1e-4 * peak, "{}", err / peak);
        assert!(y.iter().all(|v| v.norm() == 0.0));
    }

    #[test]
    fn manakov_rejects_a_step_that_does_not_advance_instead_of_looping() {
        let m = Medium { alpha_np_km: 0.0, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        for rule in [
            StepRule::Fixed(0.0),
            StepRule::Fixed(-1.0),
            StepRule::Fixed(f64::NAN),
            StepRule::NonlinearPhase(0.0),
            StepRule::NonlinearPhase(-2e-2),
        ] {
            let (mut x, mut y) = dual_field(64);
            let s = ManakovSplitStep { rule, ..ManakovSplitStep::new(m) };
            assert!(matches!(s.run(&mut x, &mut y, 1.0, 256e9), Err(Error::Config(_))), "{rule:?}");
        }
        let s = ManakovSplitStep::new(m);
        let (mut x, mut y) = dual_field(64);
        assert!(matches!(s.run(&mut x, &mut y, f64::INFINITY, 256e9), Err(Error::Config(_))));
        assert!(matches!(s.run(&mut x, &mut y[..32], 1.0, 256e9), Err(Error::Shape(_))));
        assert_eq!(s.run(&mut x, &mut y, 0.0, 256e9).unwrap(), 0);
        // an infinite power: a nonlinear step of zero length
        x[0] = C64::new(f64::INFINITY, 0.0);
        assert!(matches!(s.run(&mut x, &mut y, 1.0, 256e9), Err(Error::Numerical(_))));
    }

    #[test]
    fn manakov_conserves_the_total_energy_and_converges_with_the_step() {
        let m = Medium { alpha_np_km: 0.0, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let run = |rule| {
            let (mut x, mut y) = dual_field(1024);
            let s = ManakovSplitStep { rule, ..ManakovSplitStep::new(m) };
            let steps = s.run(&mut x, &mut y, 80.0, 256e9).unwrap();
            (x, y, steps)
        };
        let (x0, y0) = dual_field(1024);
        let e0: f64 = x0.iter().chain(&y0).map(C64::norm_sqr).sum();
        let (xa, ya, na) = run(StepRule::NonlinearPhase(2e-2));
        let (xb, yb, nb) = run(StepRule::NonlinearPhase(2e-4));
        let e1: f64 = xa.iter().chain(&ya).map(C64::norm_sqr).sum();
        assert!((e1 / e0 - 1.0).abs() < 1e-10);
        assert!(nb > 50 * na);
        // second order: the error of the coarse solution is small against the fine one
        let err: f64 = xa.iter().zip(&xb).chain(ya.iter().zip(&yb)).map(|(u, v)| (u - v).norm_sqr()).sum();
        assert!((err / e0).sqrt() < 1e-3, "{}", (err / e0).sqrt());
    }

    #[test]
    fn the_fwm_rule_follows_eq_13_of_musetti_and_grows_with_the_loss() {
        // two tones 200 GHz apart: a bandwidth of 200 GHz plus one bin
        let (n, fs) = (1024, 1.024e12);
        let tone =
            |f: f64| -> Vec<C64> { (0..n).map(|k| cis(2.0 * std::f64::consts::PI * f * k as f64 / fs)).collect() };
        let x: Vec<C64> = tone(-100e9).iter().zip(tone(100e9)).map(|(a, b)| a + b).collect();
        let y = vec![C64::new(0.0, 0.0); n];
        assert!((signal_bandwidth(&[&x, &y], fs) - 201e9).abs() < 1.0);
        assert_eq!(signal_bandwidth(&[&y], fs), 0.0);
        // the first step of eq. (13): 20 rad over 350 GHz of SMF is 190 m
        let m = Medium {
            alpha_np_km: 0.2 / 10.0 * std::f64::consts::LN_10,
            beta2_s2_km: -2.1683e-23,
            gamma_w_km: 1.3,
            beta3_s3_km: 0.0,
        };
        let s = ManakovSplitStep {
            rule: StepRule::Fwm {
                phase_rad: 20.0,
                bandwidth_hz: Some(350e9),
                update: StepUpdate::ConstantLocalError,
                max_nl_phase_rad: None,
            },
            ..ManakovSplitStep::new(m)
        };
        let h1 = s.step_length(&[0.0], 0.0, 100.0, 100.0, None, 350e9).unwrap().0;
        assert!((h1 - 20.0 / (2.1683e-23 * (2.0 * std::f64::consts::PI * 350e9).powi(2))).abs() < 1e-12);
        assert!((h1 - 0.190).abs() < 0.001, "{h1}");
        // the constant local error stretches the step by e^{α h / 3}; without loss it stays
        let h2 = s.step_length(&[0.0], h1, 100.0, h1 + 100.0, Some((h1, h1)), 350e9).unwrap().0;
        assert!((h2 / h1 - (m.alpha_np_km * h1 / 3.0).exp()).abs() < 1e-12);
        let lossless = ManakovSplitStep { medium: Medium { alpha_np_km: 0.0, ..m }, ..s };
        assert_eq!(lossless.step_length(&[0.0], h1, 100.0, h1 + 100.0, Some((h1, h1)), 350e9).unwrap().0, h1);
        // the nonlinear-phase update keeps L_eff(h) e^{-α z}: it grows faster
        let nlp = ManakovSplitStep {
            rule: StepRule::Fwm {
                phase_rad: 20.0,
                bandwidth_hz: Some(350e9),
                update: StepUpdate::NonlinearPhase,
                max_nl_phase_rad: None,
            },
            ..s
        };
        let h2n = nlp.step_length(&[0.0], h1, 100.0, h1 + 100.0, Some((h1, h1)), 350e9).unwrap().0;
        let leff = |h: f64| (1.0 - (-m.alpha_np_km * h).exp()) / m.alpha_np_km;
        assert!((leff(h2n) / leff(h1) - (m.alpha_np_km * h1).exp()).abs() < 1e-12);
        assert!(h2n > h2);
        // a non-positive phase or bandwidth is refused
        for rule in [
            StepRule::Fwm {
                phase_rad: 0.0,
                bandwidth_hz: None,
                update: StepUpdate::ConstantLocalError,
                max_nl_phase_rad: None,
            },
            StepRule::Fwm {
                phase_rad: 20.0,
                bandwidth_hz: Some(0.0),
                update: StepUpdate::ConstantLocalError,
                max_nl_phase_rad: None,
            },
            StepRule::Fwm {
                phase_rad: 20.0,
                bandwidth_hz: None,
                update: StepUpdate::ConstantLocalError,
                max_nl_phase_rad: Some(0.0),
            },
        ] {
            let (mut a, mut b) = (x.clone(), y.clone());
            assert!(ManakovSplitStep { rule, ..s }.run(&mut a, &mut b, 1.0, fs).is_err());
        }
    }

    #[test]
    fn the_third_order_dispersion_adds_a_cubic_phase() {
        let m = Medium { beta2_s2_km: -2.17e-23, beta3_s3_km: 1.3e-37, ..Medium::default() };
        let omega = fft::angular_frequencies(64, 64e9);
        let r = m.linear_response(&omega, 50.0);
        for (v, w) in r.iter().zip(&omega) {
            let phase = (m.beta2_s2_km / 2.0 * w * w + m.beta3_s3_km / 6.0 * w * w * w) * 50.0;
            assert!((v - cis(phase)).norm() < 1e-12);
        }
        // negated, the response undoes itself
        let back = m.negated().linear_response(&omega, 50.0);
        assert!(r.iter().zip(&back).all(|(a, b)| ((a * b) - C64::new(1.0, 0.0)).norm() < 1e-12));
    }

    #[test]
    fn one_field_is_the_scalar_split_step_and_fixed_steps_are_equal() {
        let m = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let (x0, _) = dual_field(1024);
        let fs = 256e9;
        // 10 km with steps of at most 3 km: four steps of 2.5 km
        let mut scalar = x0.clone();
        SplitStep::new(&m, Steps::new(4, 2.5), 1024, fs).run(&mut scalar);
        let engine = ManakovSplitStep { rule: StepRule::Fixed(3.0), ..ManakovSplitStep::new(m) };
        let mut x = x0.clone();
        assert_eq!(engine.run_fields(&mut [&mut x], 10.0, fs, &PropagationOptions::default()).unwrap(), 4);
        let e: f64 = scalar.iter().map(C64::norm_sqr).sum();
        let d: f64 = x.iter().zip(&scalar).map(|(u, v)| (u - v).norm_sqr()).sum();
        assert!((d / e).sqrt() < 1e-12, "{}", (d / e).sqrt());
    }

    #[test]
    fn the_birefringence_of_the_sections_is_their_lumped_response() {
        // without nonlinearity: the product of D_k R_k at every frequency
        let m = Medium { alpha_np_km: 0.0, beta2_s2_km: 0.0, gamma_w_km: 0.0, beta3_s3_km: 0.0 };
        let (n, fs) = (256, 256e9);
        let (x0, y0) = dual_field(n);
        let rot = |t: f64, p: f64| -> [C64; 4] {
            let (s, c) = t.sin_cos();
            [C64::new(c, 0.0), -cis(-p) * s, cis(p) * s, C64::new(c, 0.0)]
        };
        let rotations = [rot(0.3, 0.1), rot(1.1, -0.7), rot(-0.4, 2.0)];
        let dgd = [3e-12, 1e-12, 2e-12];
        let b = Birefringence { section_km: 2.0, rotations: &rotations, dgd_s_per_km: &dgd };
        let engine = ManakovSplitStep { rule: StepRule::Fixed(0.7), ..ManakovSplitStep::new(m) };
        let (mut x, mut y) = (x0.clone(), y0.clone());
        let options = PropagationOptions { gain: None, birefringence: Some(b) };
        let steps = engine.run_fields(&mut [&mut x, &mut y], 6.0, fs, &options).unwrap();
        assert_eq!(steps, 9); // three sections of three steps
        // the lumped response in the frequency domain
        let omega = fft::angular_frequencies(n, fs);
        let (fx, fy) = (fft::fft(&x0), fft::fft(&y0));
        let mut ex = vec![C64::new(0.0, 0.0); n];
        let mut ey = ex.clone();
        for k in 0..n {
            let (mut a, mut c) = (fx[k], fy[k]);
            for (r, t) in rotations.iter().zip(&dgd) {
                (a, c) = (r[0] * a + r[1] * c, r[2] * a + r[3] * c);
                let ph = 0.5 * omega[k] * t * 2.0;
                (a, c) = (a * cis(-ph), c * cis(ph));
            }
            ex[k] = a;
            ey[k] = c;
        }
        let (ex, ey) = (fft::ifft(&ex), fft::ifft(&ey));
        let e: f64 = ex.iter().chain(&ey).map(C64::norm_sqr).sum();
        let d: f64 = x.iter().zip(&ex).chain(y.iter().zip(&ey)).map(|(u, v)| (u - v).norm_sqr()).sum();
        assert!((d / e).sqrt() < 1e-12, "{}", (d / e).sqrt());
        // invalid inputs
        let mut z = x0.clone();
        assert!(engine.run_fields(&mut [&mut z], 6.0, fs, &options).is_err());
        assert!(engine.run_fields(&mut [&mut x, &mut y], 5.0, fs, &options).is_err());
        let (mut a, mut b2, mut c) = (x0.clone(), x0.clone(), x0);
        assert!(engine.run_fields(&mut [&mut a, &mut b2, &mut c], 1.0, fs, &PropagationOptions::default()).is_err());
        assert!(engine.run_fields(&mut [], 1.0, fs, &PropagationOptions::default()).is_err());
    }

    #[test]
    fn the_fwm_rule_keeps_its_step_across_the_sections_of_the_birefringence() {
        // a step cut at the end of a section does not restart the rule from the cut step:
        // sections with identity rotations and no DGD leave the number of steps near that
        // of the span without sections (one cut step per section at most), and the field
        let m = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let (n, fs) = (1024, 256e9);
        let (x0, y0) = dual_field(n);
        let one = C64::new(1.0, 0.0);
        let zero = C64::new(0.0, 0.0);
        for update in [StepUpdate::ConstantLocalError, StepUpdate::NonlinearPhase] {
            for alpha in [0.046, 0.0] {
                let m = Medium { alpha_np_km: alpha, ..m };
                let rule = StepRule::Fwm { phase_rad: 20.0, bandwidth_hz: Some(200e9), update, max_nl_phase_rad: None };
                let engine = ManakovSplitStep { rule, ..ManakovSplitStep::new(m) };
                let (mut x, mut y) = (x0.clone(), y0.clone());
                let plain = engine.run(&mut x, &mut y, 80.0, fs).unwrap();
                for sections in [10usize, 16, 80] {
                    let rotations = vec![[one, zero, zero, one]; sections];
                    let dgd = vec![0.0; sections];
                    let b =
                        Birefringence { section_km: 80.0 / sections as f64, rotations: &rotations, dgd_s_per_km: &dgd };
                    let options = PropagationOptions { gain: None, birefringence: Some(b) };
                    let (mut u, mut v) = (x0.clone(), y0.clone());
                    let steps = engine.run_fields(&mut [&mut u, &mut v], 80.0, fs, &options).unwrap();
                    assert!(steps <= plain + sections, "{update:?} alpha {alpha}: {steps} steps against {plain}");
                    let e: f64 = x.iter().chain(&y).map(C64::norm_sqr).sum();
                    let d: f64 = u.iter().zip(&x).chain(v.iter().zip(&y)).map(|(a, b)| (a - b).norm_sqr()).sum();
                    assert!((d / e).sqrt() < 2e-2, "{}", (d / e).sqrt());
                }
            }
        }
    }

    #[test]
    fn the_nonlinear_phase_bounds_the_step_of_a_negated_medium() {
        // the bound is on the magnitude of the phase: the medium of a back-propagation
        // (negative gamma) is bounded as the forward one
        let m = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let (x0, y0) = dual_field(1024);
        for rule in [StepRule::NonlinearPhase(2e-2), StepRule::FWM_DEFAULT] {
            let forward = ManakovSplitStep { rule, ..ManakovSplitStep::new(m) };
            let backward = ManakovSplitStep { rule, ..ManakovSplitStep::new(Medium { gamma_w_km: -1.3, ..m }) };
            let step = |phi: f64| forward.step_length(&[0.0, phi], 0.0, 80.0, 80.0, None, 10e9).unwrap().0;
            assert_eq!(step(-0.5), step(0.5));
            assert!((step(-0.5) - 0.04).abs() < 1e-15, "{}", step(-0.5));
            let (mut x, mut y) = (x0.clone(), y0.clone());
            let f = forward.run(&mut x, &mut y, 20.0, 256e9).unwrap();
            let (mut x, mut y) = (x0.clone(), y0.clone());
            let b = backward.run(&mut x, &mut y, 20.0, 256e9).unwrap();
            assert!(f > 20 && b > f / 4, "{rule:?}: {f} steps forward, {b} with the negated medium");
        }
    }

    #[test]
    fn the_midpoint_manakov_step_is_the_scalar_split_step() {
        // one polarisation, gamma scaled by 9/8: the same scheme as SplitStep
        let m = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let (x0, _) = dual_field(1024);
        let (h, n_steps, fs) = (0.5, 160, 256e9);
        let mut scalar = x0.clone();
        SplitStep::new(&m, Steps::new(n_steps, h), 1024, fs).run(&mut scalar);
        let solver = ManakovSplitStep {
            rule: StepRule::Fixed(h),
            nonlinear: NonlinearStep::Midpoint,
            ..ManakovSplitStep::new(Medium { gamma_w_km: 9.0 / 8.0 * m.gamma_w_km, ..m })
        };
        let (mut x, mut y) = (x0, vec![C64::new(0.0, 0.0); 1024]);
        assert_eq!(solver.run(&mut x, &mut y, n_steps as f64 * h, fs).unwrap(), n_steps);
        let e: f64 = scalar.iter().map(C64::norm_sqr).sum();
        let d: f64 = x.iter().zip(&scalar).map(|(u, v)| (u - v).norm_sqr()).sum();
        assert!((d / e).sqrt() < 1e-12, "{}", (d / e).sqrt());
        assert!(y.iter().all(|v| v.norm() == 0.0));
        assert_eq!("Midpoint".parse::<NonlinearStep>().unwrap(), NonlinearStep::Midpoint);
        assert_eq!("effective_length".parse::<NonlinearStep>().unwrap(), NonlinearStep::EffectiveLength);
        assert!("euler".parse::<NonlinearStep>().is_err());
    }

    /// The effective-length step of Optilux: on a constant field (no dispersion acts)
    /// the phases of the steps add up to $\gamma P L_{\mathrm{eff}}(L)$ exactly whatever the
    /// step, where the midpoint step leaves its discretisation error; without loss it
    /// is the midpoint step bit for bit; with loss and dispersion it is within the step
    /// error of the other two schemes, and the adaptive loop gives the fixed path's field.
    #[test]
    fn the_effective_length_step_integrates_the_attenuation_of_the_step_exactly() {
        let m = Medium { alpha_np_km: 0.046, beta2_s2_km: 0.0, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let (p, n, fs) = (5e-3_f64, 64, 32e9);
        let x0 = vec![C64::new(p.sqrt(), 0.0); n];
        let expected = m.gamma_w_km * p * crate::effective_length(m.alpha_np_km, 80.0);
        let run = |nonlinear: NonlinearStep, h: f64| {
            // one field: the scalar equation, the phase γ P L (8/9 γ with two)
            let mut x = x0.clone();
            let solver = ManakovSplitStep { rule: StepRule::Fixed(h), nonlinear, ..ManakovSplitStep::new(m) };
            solver.run_fields(&mut [&mut x], 80.0, fs, &PropagationOptions::default()).unwrap();
            x
        };
        for h in [0.5, 4.0, 20.0, 80.0] {
            let phase = run(NonlinearStep::EffectiveLength, h)[0].arg();
            assert!((phase - expected).abs() < 1e-12, "step {h}: {phase} against {expected}");
        }
        // the midpoint step discretises L_eff: 1.6 % off with one step of the span
        let midpoint = run(NonlinearStep::Midpoint, 80.0)[0].arg();
        assert!((midpoint - expected).abs() > 1e-2 * expected);
        // without loss the two schemes are the same, bit for bit
        let lossless = Medium { alpha_np_km: 0.0, beta2_s2_km: -2.17e-23, ..m };
        let (xd, _) = dual_field(256);
        let run_lossless = |nonlinear: NonlinearStep| {
            let (mut x, mut y) = (xd.clone(), vec![C64::new(0.0, 0.0); 256]);
            ManakovSplitStep { rule: StepRule::Fixed(1.0), nonlinear, ..ManakovSplitStep::new(lossless) }
                .run(&mut x, &mut y, 10.0, 256e9)
                .unwrap();
            x
        };
        assert_eq!(run_lossless(NonlinearStep::EffectiveLength), run_lossless(NonlinearStep::Midpoint));
        // with loss and dispersion: within the step error of the other schemes, and the
        // adaptive loop (the nonlinear-phase rule with an unreachable bound) takes the
        // same steps as the fixed path and gives its field to the rounding
        let lossy = Medium { alpha_np_km: 0.046, beta2_s2_km: -2.17e-23, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let nmse = |a: &[C64], b: &[C64]| {
            let d: f64 = a.iter().zip(b).map(|(u, v)| (u - v).norm_sqr()).sum();
            d / b.iter().map(C64::norm_sqr).sum::<f64>()
        };
        let run_lossy = |rule: StepRule, nonlinear: NonlinearStep| {
            let (mut x, mut y) = (xd.clone(), vec![C64::new(0.0, 0.0); 256]);
            let steps = ManakovSplitStep { rule, nonlinear, ..ManakovSplitStep::new(lossy) }
                .run(&mut x, &mut y, 80.0, 256e9)
                .unwrap();
            (x, steps)
        };
        let fine = run_lossy(StepRule::Fixed(0.05), NonlinearStep::Midpoint).0;
        let (eff, steps) = run_lossy(StepRule::Fixed(1.0), NonlinearStep::EffectiveLength);
        assert_eq!(steps, 80);
        let mid = run_lossy(StepRule::Fixed(1.0), NonlinearStep::Midpoint).0;
        let trap = run_lossy(StepRule::Fixed(1.0), NonlinearStep::Trapezoidal).0;
        // 3.83e-5, 3.83e-5 and 2.8e-4 on this strong field
        for (name, field) in [("effective length", &eff), ("midpoint", &mid), ("trapezoidal", &trap)] {
            assert!(nmse(field, &fine) < 1e-3, "{name}: {}", nmse(field, &fine));
        }
        let (adaptive, steps) = run_lossy(StepRule::NonlinearPhase(1e9), NonlinearStep::EffectiveLength);
        assert_eq!(steps, 1);
        let (one, _) = run_lossy(StepRule::Fixed(80.0), NonlinearStep::EffectiveLength);
        assert!(nmse(&adaptive, &one) < 1e-28, "{}", nmse(&adaptive, &one));
    }

    #[test]
    fn fundamental_soliton_keeps_its_shape() {
        // T0 = 10 ps, β₂ = -20 ps²/km, P0 = |β₂| / (γ T0²): |A(z, t)| = |A(0, t)|
        let (t0, b2, gamma): (f64, f64, f64) = (10e-12, -20e-24, 1.0);
        let p0 = b2.abs() / (gamma * t0 * t0);
        let (n, fs) = (1024, 2e12);
        let a0: Vec<C64> = (0..n)
            .map(|k| {
                let t = (k as f64 - n as f64 / 2.0) / fs;
                C64::new(p0.sqrt() / (t / t0).cosh(), 0.0)
            })
            .collect();
        let m = Medium { alpha_np_km: 0.0, beta2_s2_km: b2, gamma_w_km: gamma, beta3_s3_km: 0.0 };
        let mut a = a0.clone();
        // two dispersion lengths (L_D = T0²/|β₂| = 5 km)
        SplitStep::new(&m, Steps::new(1000, 0.01), n, fs).run(&mut a);
        let err = a.iter().zip(&a0).map(|(u, v)| (u.norm() - v.norm()).abs()).fold(0.0, f64::max);
        assert!(err < 1e-3 * p0.sqrt(), "relative envelope error {}", err / p0.sqrt());
    }

    /// The logarithmic steps are eq. (6) of Bosco et al. (2000) written with the field
    /// attenuation; one step is the span, no loss gives equal steps, and the inputs are
    /// checked.
    #[test]
    fn logarithmic_steps_are_eq_6_of_bosco_and_check_their_inputs() {
        let alpha = 0.25 / (10.0 * std::f64::consts::E.log10()); // 0.25 dB/km, Fig. 3
        let (l, k) = (50.0, 50usize);
        let h = logarithmic_steps(alpha, l, k).unwrap();
        // eq. (6): dz_n = -1/(2 a_f) ln((1 - n s)/(1 - (n - 1) s)), s = (1 - e^{-2 a_f L})/K
        let af = alpha / 2.0;
        let sigma = (1.0 - (-2.0 * af * l).exp()) / k as f64;
        for (n, hn) in h.iter().enumerate() {
            let n = (n + 1) as f64;
            let bosco = -1.0 / (2.0 * af) * ((1.0 - n * sigma) / (1.0 - (n - 1.0) * sigma)).ln();
            assert!((hn - bosco).abs() < 1e-10 * bosco, "step {n}: {hn} against {bosco}");
        }
        assert!((h.iter().sum::<f64>() - l).abs() < 1e-12 * l);
        assert_eq!(logarithmic_steps(alpha, 80.0, 1).unwrap(), vec![80.0]);
        assert!(logarithmic_steps(0.0, 80.0, 4).unwrap().iter().all(|&s| (s - 20.0).abs() < 1e-12));
        assert!(logarithmic_steps(alpha, 80.0, 0).is_err());
        assert!(logarithmic_steps(alpha, 0.0, 4).is_err());
        assert!(logarithmic_steps(-1.0, 80.0, 4).is_err());
        assert_eq!(StepDistribution::Uniform.steps(alpha, 80.0, 4).unwrap(), vec![20.0; 4]);
    }

    /// The fast path of the Manakov solver with the logarithmic rule, on one field, is the
    /// scalar split step with the same steps; with equal steps `with_lengths` is `new`; the
    /// adaptive loop (trapezoidal phase) takes the same number of steps.
    #[test]
    fn the_logarithmic_rule_runs_the_steps_of_logarithmic_steps() {
        let alpha = 0.2 / (10.0 * std::f64::consts::E.log10());
        let m = Medium { alpha_np_km: alpha, beta2_s2_km: -21.683e-24, gamma_w_km: 1.3, beta3_s3_km: 0.0 };
        let (n, fs) = (512usize, 64e9);
        let w = |k: usize, f: f64, ph: f64| (2.0 * std::f64::consts::PI * f * k as f64 / n as f64 + ph).sin();
        let x0: Vec<C64> = (0..n).map(|k| C64::new(w(k, 31.0, 0.2), w(k, 17.0, 1.0)) * 0.05).collect();
        let mut a = x0.clone();
        let taken = ManakovSplitStep { rule: StepRule::Logarithmic(7), ..ManakovSplitStep::new(m) }
            .run_fields(&mut [&mut a], 80.0, fs, &PropagationOptions::default())
            .unwrap();
        assert_eq!(taken, 7);
        let steps: Vec<(f64, f64)> = logarithmic_steps(alpha, 80.0, 7).unwrap().iter().map(|&h| (h, h)).collect();
        let mut b = x0.clone();
        SplitStep::with_lengths(&m, &steps, n, fs).run(&mut b);
        assert!(a.iter().zip(&b).all(|(u, v)| (u - v).norm() < 1e-12), "fast path against with_lengths");
        let mut c = x0.clone();
        let taken = ManakovSplitStep {
            rule: StepRule::Logarithmic(7),
            nonlinear: NonlinearStep::Trapezoidal,
            ..ManakovSplitStep::new(m)
        }
        .run_fields(&mut [&mut c], 80.0, fs, &PropagationOptions::default())
        .unwrap();
        assert_eq!(taken, 7);
        // equal steps through with_lengths: the steps of new, bit for bit
        let (mut d, mut e) = (x0.clone(), x0.clone());
        SplitStep::new(&m, Steps::new(4, 20.0), n, fs).run(&mut d);
        SplitStep::with_lengths(&m, &[(20.0, 20.0); 4], n, fs).run(&mut e);
        assert_eq!(d, e);
        assert!(
            ManakovSplitStep { rule: StepRule::Logarithmic(0), ..ManakovSplitStep::new(m) }
                .run_fields(&mut [&mut d], 80.0, fs, &PropagationOptions::default())
                .is_err()
        );
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
