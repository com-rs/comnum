//! The split-step solver of the Manakov equation, [`ManakovSplitStep`]: two polarisations,
//! the nonlinear rotation and the birefringence of the fibre.

use super::{
    FieldGain, Medium, NonlinearStep, StepRule, StepUpdate, logarithmic_boundary, manakov_rotation, mul, scale,
    signal_bandwidth,
};
use crate::fft;
use crate::{C64, cis, effective_length};
use comnum_core::{Error, Result, ensure, same_len};

/// The symmetric split-step solver of the Manakov equation (see the module).
///
/// With [`NonlinearStep::Trapezoidal`] (OptiCommPy's scheme) the solver is second order
/// but its error on a collision of two solitons 800 GHz apart is 6.4 times that of
/// [`NonlinearStep::Midpoint`] (the default) at the same steps (`validation/tests/fiber/sinkin2003.rs`),
/// and it underestimates the cross-channel interference of a WDM link whose channels
/// walk off within a step. [`NonlinearStep::EffectiveLength`] (Optilux's scheme) applies
/// the phase of the power at the start of the nonlinear block over the effective length
/// of the step, the attenuation of the step after it: it reproduces Optilux's fields
/// (`tests/oracle/test_optilux_oracle.py`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ManakovSplitStep {
    /// The fibre.
    pub medium: Medium,
    /// The step size.
    pub rule: StepRule,
    /// The nonlinear step.
    pub nonlinear: NonlinearStep,
    /// Largest number of trapezoidal iterations per step.
    pub max_iterations: usize,
    /// Convergence tolerance of the iterations: the relative change of the fields.
    pub tolerance: f64,
}

/// The birefringence of a span cut into sections of equal length (the PMD of
/// `comnum_channel::optical::pmd::PmdLink`): section $k$ starts with the frequency-independent
/// rotation `rotations[k]` (a row-major 2 × 2 matrix applied to $(A_{x}, A_{y})$) and
/// then delays the two polarisations of its frame by $\mp \tau'_{k} z / 2$, its DGD per
/// kilometre `dgd_s_per_km[k]` distributed along it. Without nonlinearity a span is the
/// lumped response $\prod_{k} D_{k}(\omega) R_{k}$, $D_{k}$ the DGD of the whole section.
#[derive(Clone, Copy, Debug)]
pub struct Birefringence<'a> {
    /// The length of a section (km).
    pub section_km: f64,
    /// The rotation at the start of every section.
    pub rotations: &'a [[C64; 4]],
    /// The DGD of every section per kilometre (s/km).
    pub dgd_s_per_km: &'a [f64],
}

/// What a propagation adds to the loss and dispersion of the medium (see
/// [`ManakovSplitStep::run_fields`]).
#[derive(Clone, Copy, Default)]
pub struct PropagationOptions<'a> {
    /// A distributed gain (see [`SplitStep::run_with_gain`](super::SplitStep::run_with_gain)): the medium then carries no
    /// attenuation, the gain holds the power profile of the span.
    pub gain: Option<FieldGain<'a>>,
    /// The birefringence of the span (two polarisations only).
    pub birefringence: Option<Birefringence<'a>>,
}

impl std::fmt::Debug for PropagationOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PropagationOptions")
            .field("gain", &self.gain.map(|_| "FieldGain"))
            .field("birefringence", &self.birefringence)
            .finish()
    }
}

/// The buffers of a propagation of [`ManakovSplitStep`], allocated once per run.
struct Work {
    /// The angular frequencies of the bins.
    omega: Vec<f64>,
    /// $P(z)$, the total power at the start of the step.
    p0: Vec<f64>,
    /// The total power of the last iteration.
    p1: Vec<f64>,
    /// The phase per kilometre of the trapezoid.
    phi: Vec<f64>,
    /// The scalar half linear step.
    lin: Vec<C64>,
    /// The two half steps of every field (the DGD of a section, a distributed gain).
    half1: Vec<Vec<C64>>,
    half2: Vec<Vec<C64>>,
    gain: Vec<f64>,
    /// The fields after the first half step, of the current iteration, and of the last
    /// iteration (at the start of a step, the field at z).
    h: Vec<Vec<C64>>,
    f: Vec<Vec<C64>>,
    c: Vec<Vec<C64>>,
}

impl ManakovSplitStep {
    /// The step rule of Musetti et al. ([`StepRule::FWM_DEFAULT`]: about 0.01 dB of error
    /// on the SNR whatever the power), the nonlinear phase at the middle of the step; with
    /// [`NonlinearStep::Trapezoidal`], OptiCommPy's iterations: at most 10, tolerance
    /// $10^{-5}$. OptiCommPy's own step, 0.02 rad of nonlinear phase, is
    /// `StepRule::NonlinearPhase(2e-2)`: it depends on the power and is not converged in
    /// every WDM link (`specs/raman_wdm.yaml`).
    pub fn new(medium: Medium) -> ManakovSplitStep {
        ManakovSplitStep {
            medium,
            rule: StepRule::FWM_DEFAULT,
            nonlinear: NonlinearStep::Midpoint,
            max_iterations: 10,
            tolerance: 1e-5,
        }
    }

    /// Propagate the two polarisations (√W, time domain, equal lengths) over
    /// `length_km` of fibre sampled at `fs`, in place: the number of steps (none for a
    /// length that is not positive).
    ///
    /// # Errors
    /// As [`ManakovSplitStep::run_fields`].
    pub fn run(&self, x: &mut [C64], y: &mut [C64], length_km: f64, fs: f64) -> Result<usize> {
        self.run_fields(&mut [x, y], length_km, fs, &PropagationOptions::default())
    }

    /// [`ManakovSplitStep::run`] with a distributed gain (see
    /// [`SplitStep::run_with_gain`](super::SplitStep::run_with_gain)): the medium carries no attenuation, `gain` holds the
    /// power profile of the span.
    ///
    /// # Errors
    /// As [`ManakovSplitStep::run_fields`].
    pub fn run_with_gain(
        &self,
        x: &mut [C64],
        y: &mut [C64],
        length_km: f64,
        fs: f64,
        gain: FieldGain<'_>,
    ) -> Result<usize> {
        self.run_fields(&mut [x, y], length_km, fs, &PropagationOptions { gain: Some(gain), birefringence: None })
    }

    /// Propagate one field (the scalar NLSE, nonlinear coefficient $\gamma$) or two (the
    /// Manakov equation, $\frac{8}{9}\gamma$) over `length_km` sampled at `fs`, in place,
    /// with the additions of `options`: the engine of every forward propagation
    /// (`docs/adr/0014-one-fibre-propagator.md`). The steps follow [`ManakovSplitStep::rule`]
    /// and are cut at the boundaries of the sections of a birefringence; a fixed step
    /// covers every stretch with equal steps of at most that length. Returns the number
    /// of steps.
    ///
    /// # Errors
    /// [`Error::Shape`] for no field, more than two, fields of different lengths, a
    /// birefringence with one field or sections that do not cover the length;
    /// [`Error::Config`] for a fixed step, a rotation or a phase that is not positive or
    /// a length that is not finite; [`Error::Numerical`] if a step vanishes (a non-finite
    /// field).
    pub fn run_fields(
        &self,
        fields: &mut [&mut [C64]],
        length_km: f64,
        fs: f64,
        options: &PropagationOptions<'_>,
    ) -> Result<usize> {
        let nf = fields.len();
        if nf == 0 || nf > 2 {
            return Err(Error::Shape(format!("a propagation takes one or two fields, got {nf}")));
        }
        let n = fields[0].len();
        if nf == 2 {
            same_len(n, fields[1].len(), "the polarisations")?;
        }
        self.rule_parameter()?;
        ensure(length_km.is_finite(), || format!("the length must be finite, got {length_km} km"))?;
        if let Some(b) = &options.birefringence {
            check_birefringence(b, nf, length_km)?;
        }
        if n == 0 || length_km <= 0.0 {
            return Ok(0);
        }
        if let (
            StepRule::Fixed(_) | StepRule::Logarithmic(_),
            NonlinearStep::Midpoint | NonlinearStep::EffectiveLength,
        ) = (self.rule, self.nonlinear)
        {
            return Ok(self.run_fixed_midpoint(fields, length_km, fs, options));
        }
        let p = fft::plan(n);
        let zero = C64::new(0.0, 0.0);
        let mut w = Work {
            omega: fft::angular_frequencies(n, fs),
            p0: vec![0.0; n],
            p1: vec![0.0; n],
            phi: vec![0.0; n],
            lin: vec![zero; n],
            half1: vec![vec![zero; n]; nf],
            half2: vec![vec![zero; n]; nf],
            gain: vec![0.0; n],
            h: vec![vec![zero; n]; nf],
            f: vec![vec![zero; n]; nf],
            c: fields.iter().map(|u| u.to_vec()).collect(),
        };
        let k = if nf == 2 { 8.0 / 9.0 } else { 1.0 } * self.medium.gamma_w_km;
        let bandwidth = match self.rule {
            StepRule::Fwm { bandwidth_hz: Some(b), .. } => b,
            StepRule::Fwm { bandwidth_hz: None, .. } => {
                let views: Vec<&[C64]> = fields.iter().map(|u| &**u).collect();
                signal_bandwidth(&views, fs)
            }
            _ => 0.0,
        };
        let mut z = 0.0;
        let mut steps = 0;
        let mut previous = None;
        // the section of the birefringence the steps are in, and the next boundary
        let mut section: Option<usize> = None;
        while z < length_km {
            let mut stretch_end = length_km;
            if let Some(b) = &options.birefringence {
                let current = ((z / b.section_km) * (1.0 + 1e-12)).floor() as usize;
                let current = current.min(b.rotations.len() - 1);
                if section != Some(current) {
                    // the rotation at the start of the section, on the fields and on the
                    // field of the last iteration (a unitary, local change of frame)
                    let (a, rest) = fields.split_at_mut(1);
                    rotate(&b.rotations[current], a[0], rest[0]);
                    let (a, rest) = w.c.split_at_mut(1);
                    rotate(&b.rotations[current], &mut a[0], &mut rest[0]);
                    section = Some(current);
                }
                stretch_end = (((current + 1) as f64) * b.section_km).min(length_km);
            }
            total_power_of(fields, &mut w.p0);
            total_power_of(&w.c, &mut w.p1);
            trapezoid(&w.p0, &w.p1, k, &mut w.phi);
            let (h, rule) = self.step_length(&w.phi, z, stretch_end - z, length_km, previous, bandwidth)?;
            // the step of the rule and the length advanced: a step cut at the end of a
            // section of the birefringence does not restart the rule from the cut step
            previous = Some((rule, h));
            self.half_steps(&mut w, z, h, section, options);
            for (o, u) in w.h.iter_mut().zip(fields.iter()) {
                o.copy_from_slice(u);
            }
            for (u, half) in w.h.iter_mut().zip(&w.half1) {
                p.forward(u);
                mul(u, half);
                p.inverse(u);
            }
            if matches!(self.nonlinear, NonlinearStep::Midpoint | NonlinearStep::EffectiveLength) {
                // the phase of the power after the first half of the dispersion (and, for
                // the midpoint step, of the attenuation)
                kerr_rotation(&mut w.h, k * self.nonlinear_length(h));
                for (u, half) in w.h.iter_mut().zip(&w.half2) {
                    p.forward(u);
                    mul(u, half);
                    p.inverse(u);
                }
                for ((u, c), hh) in fields.iter_mut().zip(w.c.iter_mut()).zip(&w.h) {
                    u.copy_from_slice(hh);
                    c.copy_from_slice(hh);
                }
            } else {
                for it in 0..self.max_iterations.max(1) {
                    for (o, v) in w.f.iter_mut().zip(&w.h) {
                        for ((a, b), ph) in o.iter_mut().zip(v).zip(&w.phi) {
                            *a = b * cis(ph * h);
                        }
                    }
                    for (u, half) in w.f.iter_mut().zip(&w.half2) {
                        p.forward(u);
                        mul(u, half);
                        p.inverse(u);
                    }
                    let diff: f64 =
                        w.f.iter().zip(&w.c).flat_map(|(a, b)| a.iter().zip(b)).map(|(u, v)| (u - v).norm_sqr()).sum();
                    let norm: f64 = w.c.iter().flatten().map(C64::norm_sqr).sum();
                    std::mem::swap(&mut w.c, &mut w.f);
                    if norm == 0.0 || (diff / norm).sqrt() < self.tolerance || it + 1 == self.max_iterations {
                        break;
                    }
                    total_power_of(&w.c, &mut w.p1);
                    trapezoid(&w.p0, &w.p1, k, &mut w.phi);
                }
                for (u, c) in fields.iter_mut().zip(&w.c) {
                    u.copy_from_slice(c);
                }
            }
            z += h;
            steps += 1;
        }
        Ok(steps)
    }

    /// [`ManakovSplitStep::run_fields`] with fixed steps of at most `hf` and the phase of
    /// the midpoint or the effective-length step (no iteration): the fields stay in the
    /// frequency domain from the second half step of a step to the first of the next (two
    /// FFTs per step and field, as [`SplitStep`]), and come back to the time domain only
    /// for the nonlinear phase and at the boundaries of the sections.
    fn run_fixed_midpoint(
        &self,
        fields: &mut [&mut [C64]],
        length_km: f64,
        fs: f64,
        options: &PropagationOptions<'_>,
    ) -> usize {
        let (nf, n) = (fields.len(), fields[0].len());
        let p = fft::plan(n);
        let zero = C64::new(0.0, 0.0);
        let mut w = Work {
            omega: fft::angular_frequencies(n, fs),
            p0: Vec::new(),
            p1: Vec::new(),
            phi: Vec::new(),
            lin: vec![zero; n],
            half1: vec![vec![zero; n]; nf],
            half2: vec![vec![zero; n]; nf],
            gain: vec![0.0; n],
            h: fields.iter().map(|u| u.to_vec()).collect(),
            f: Vec::new(),
            c: Vec::new(),
        };
        let k = if nf == 2 { 8.0 / 9.0 } else { 1.0 } * self.medium.gamma_w_km;
        // the stretches: the sections of the birefringence, or the whole length
        let (n_stretches, stretch_km) = match &options.birefringence {
            Some(b) => (b.rotations.len(), b.section_km),
            None => (1, length_km),
        };
        for u in &mut w.h {
            p.forward(u);
        }
        let mut steps = 0;
        for s in 0..n_stretches {
            let z_start = s as f64 * stretch_km;
            let z_end = if s + 1 == n_stretches { length_km } else { (s + 1) as f64 * stretch_km };
            if let Some(b) = &options.birefringence {
                // the rotation at the start of the section, in the frequency domain (it does
                // not depend on the frequency)
                let (a, rest) = w.h.split_at_mut(1);
                rotate(&b.rotations[s], &mut a[0], &mut rest[0]);
            }
            let section = options.birefringence.is_some().then_some(s);
            let stretch = self.predetermined_steps(z_start, z_end, length_km);
            // the half steps without the gain, which multiplies after them; computed again
            // only when the step length changes (never for fixed steps)
            let mut current = f64::NAN;
            let mut z_log = z_start;
            for (i, &h) in stretch.iter().enumerate() {
                if h.to_bits() != current.to_bits() {
                    self.half_steps(&mut w, z_start, h, section, &PropagationOptions { gain: None, ..*options });
                    current = h;
                }
                // fixed steps: the positions as they were always computed
                let z = match self.rule {
                    StepRule::Logarithmic(_) => z_log,
                    _ => z_start + i as f64 * h,
                };
                z_log += h;
                for (u, half) in w.h.iter_mut().zip(&w.half1) {
                    mul(u, half);
                }
                if let Some(g) = options.gain {
                    g(z, z + h / 2.0, &mut w.gain);
                    for u in &mut w.h {
                        scale(u, &w.gain);
                    }
                }
                for u in &mut w.h {
                    p.inverse(u);
                }
                kerr_rotation_scalar_order(&mut w.h, k, self.nonlinear_length(h));
                for (u, half) in w.h.iter_mut().zip(&w.half2) {
                    p.forward(u);
                    mul(u, half);
                }
                if let Some(g) = options.gain {
                    g(z + h / 2.0, z + h, &mut w.gain);
                    for u in &mut w.h {
                        scale(u, &w.gain);
                    }
                }
                steps += 1;
            }
        }
        for (u, spectrum) in fields.iter_mut().zip(w.h.iter_mut()) {
            p.inverse(spectrum);
            u.copy_from_slice(spectrum);
        }
        steps
    }

    /// The length over which the nonlinear phase of a step of `h` is applied: `h`, or
    /// its effective length $(1 - e^{-\alpha h})/\alpha$ for [`NonlinearStep::EffectiveLength`]
    /// (the expression of Optilux; $h$ without attenuation).
    fn nonlinear_length(&self, h: f64) -> f64 {
        match self.nonlinear {
            NonlinearStep::EffectiveLength => {
                let a = self.medium.alpha_np_km;
                if a >= 0.0 { effective_length(a, h) } else { -(-a * h).exp_m1() / a }
            }
            NonlinearStep::Midpoint | NonlinearStep::Trapezoidal => h,
        }
    }

    /// The two half linear steps of every field for the step from `z` of length `h`: the
    /// loss and dispersion of the medium, the DGD of the section, the distributed gain.
    /// With [`NonlinearStep::EffectiveLength`] the first half step carries no loss and the
    /// second the loss of the whole step, $e^{-\alpha h / 2}$ on the field.
    fn half_steps(&self, w: &mut Work, z: f64, h: f64, section: Option<usize>, options: &PropagationOptions<'_>) {
        let effective = self.nonlinear == NonlinearStep::EffectiveLength;
        if effective {
            Medium { alpha_np_km: 0.0, ..self.medium }.half_step_into(&w.omega, h, &mut w.lin);
        } else {
            self.medium.half_step_into(&w.omega, h, &mut w.lin);
        }
        let loss = (-(self.medium.alpha_np_km / 2.0) * h).exp();
        let nf = w.half1.len();
        let dgd = match (options.birefringence, section) {
            (Some(b), Some(s)) => b.dgd_s_per_km[s],
            _ => 0.0,
        };
        for i in 0..nf {
            // the frame delays x by +τ'h/4 and y by -τ'h/4 over a half step
            let sign = if i == 0 { -1.0 } else { 1.0 };
            for ((a, b), (l, wk)) in w.half1[i].iter_mut().zip(w.half2[i].iter_mut()).zip(w.lin.iter().zip(&w.omega)) {
                let d = if dgd == 0.0 { *l } else { l * cis(sign * 0.25 * wk * dgd * h) };
                *a = d;
                *b = if effective { d * loss } else { d };
            }
        }
        if let Some(g) = options.gain {
            for (half, z0, z1) in [(&mut w.half1, z, z + h / 2.0), (&mut w.half2, z + h / 2.0, z + h)] {
                g(z0, z1, &mut w.gain);
                for v in half.iter_mut() {
                    for (o, b) in v.iter_mut().zip(&w.gain) {
                        *o *= b;
                    }
                }
            }
        }
    }

    /// The steps of the rules whose steps are known in advance ([`StepRule::Fixed`],
    /// [`StepRule::Logarithmic`]) over the stretch from `z_start` to `z_end` of a
    /// propagation of `length_km`: the fixed steps of the stretch, or the logarithmic
    /// steps of the whole length cut at the ends of the stretch.
    fn predetermined_steps(&self, z_start: f64, z_end: f64, length_km: f64) -> Vec<f64> {
        match self.rule {
            StepRule::Logarithmic(n) => {
                let a = self.medium.alpha_np_km;
                let eps = 1e-12 * length_km;
                let mut z = vec![z_start];
                z.extend(
                    (1..n)
                        .map(|k| logarithmic_boundary(a, length_km, n, k))
                        .filter(|&b| b > z_start + eps && b < z_end - eps),
                );
                z.push(z_end);
                z.windows(2).map(|p| p[1] - p[0]).collect()
            }
            StepRule::Fixed(hf) => {
                let (n, h) = equal_steps(z_end - z_start, hf);
                vec![h; n]
            }
            _ => unreachable!("only the rules of predetermined steps"),
        }
    }

    /// The parameter of the step rule, checked: the fixed step (km), the rotation or the
    /// phase of the four-wave mixing (rad), the number of logarithmic steps.
    fn rule_parameter(&self) -> Result<f64> {
        let (step, what) = match self.rule {
            StepRule::Fixed(h) => (h, "fixed step (km)"),
            StepRule::NonlinearPhase(r) => (r, "nonlinear rotation (rad)"),
            StepRule::Logarithmic(n) => (n as f64, "number of logarithmic steps"),
            StepRule::Fwm { phase_rad, bandwidth_hz, max_nl_phase_rad, .. } => {
                if let Some(b) = bandwidth_hz {
                    ensure(b > 0.0 && b.is_finite(), || format!("the bandwidth must be positive, got {b} Hz"))?;
                }
                if let Some(r) = max_nl_phase_rad {
                    ensure(r > 0.0, || format!("the nonlinear rotation must be positive, got {r}"))?;
                }
                (phase_rad, "four-wave-mixing phase (rad)")
            }
        };
        ensure(step > 0.0, || format!("the {what} must be positive, got {step}"))?;
        Ok(step)
    }

    /// The first step (km) of [`StepRule::Fwm`] for the bandwidth `b` (Hz), eq. (13) of
    /// Musetti et al. (the whole length without dispersion or bandwidth).
    fn fwm_first_step(&self, phase_rad: f64, b: f64) -> f64 {
        let w = 2.0 * std::f64::consts::PI * b;
        let h = phase_rad / (self.medium.beta2_s2_km.abs() * w * w);
        if h.is_finite() { h } else { f64::INFINITY }
    }

    /// The length of the next step at `z` with `rest` km left in the stretch, in a
    /// propagation of `length_km`, and the phase `phi` (rad/km) of the step. A fixed step
    /// `h` covers the rest of the stretch with $\lceil \text{rest} / h \rceil$ equal steps;
    /// a logarithmic step ends at the next boundary of the steps of the whole length.
    /// Returns the step and the step of the rule before the end of the stretch cut it
    /// (the same but for [`StepRule::Fwm`], which grows its next step from `previous`:
    /// the last step of the rule and the length then advanced).
    ///
    /// # Errors
    /// [`Error::Numerical`] for a step that does not advance `z`.
    pub(super) fn step_length(
        &self,
        phi: &[f64],
        z: f64,
        rest: f64,
        length_km: f64,
        previous: Option<(f64, f64)>,
        bandwidth: f64,
    ) -> Result<(f64, f64)> {
        let mut rule = None;
        let h = match self.rule {
            StepRule::Fixed(hf) => equal_steps(rest, hf).1,
            StepRule::Logarithmic(n) => {
                let a = self.medium.alpha_np_km;
                let eps = 1e-12 * length_km;
                let next = (1..=n)
                    .map(|k| logarithmic_boundary(a, length_km, n, k))
                    .find(|&b| b > z + eps)
                    .unwrap_or(length_km);
                (next - z).min(rest)
            }
            StepRule::NonlinearPhase(rot) => {
                let m = phi.iter().fold(0.0, |m: f64, p| m.max(p.abs()));
                if m > 0.0 && rest >= rot / m { rot / m } else { rest }
            }
            StepRule::Fwm { phase_rad, update, max_nl_phase_rad, .. } => {
                let a = self.medium.alpha_np_km;
                let h = match previous {
                    None => self.fwm_first_step(phase_rad, bandwidth),
                    Some((hk, _)) if a <= 0.0 => hk,
                    // `dz`, the length advanced since the step `hk` of the rule (`hk`
                    // itself but after a step cut at the end of a stretch)
                    Some((hk, dz)) => match update {
                        StepUpdate::ConstantLocalError => hk * (a * dz / 3.0).exp(),
                        StepUpdate::NonlinearPhase => {
                            // L_eff(h') = L_eff(h) e^{α dz}, h' = -ln(1 - α L_eff(h')) / α
                            let l = (1.0 - (-a * hk).exp()) / a * (a * dz).exp();
                            if a * l < 1.0 { -(1.0 - a * l).ln() / a } else { f64::INFINITY }
                        }
                    },
                };
                let m = phi.iter().fold(0.0, |m: f64, p| m.max(p.abs()));
                let nl = match max_nl_phase_rad {
                    Some(rot) if m > 0.0 => rot / m,
                    _ => f64::INFINITY,
                };
                rule = Some(h.min(nl));
                h.min(nl).min(rest)
            }
        };
        // a step that does not advance (an infinite power, a step below the precision
        // of z) would never reach the end of the fibre
        if z + h <= z || h.is_nan() {
            return Err(Error::Numerical(format!("the split step vanishes at z = {z} km (step {h} km)")));
        }
        Ok((h, rule.unwrap_or(h)))
    }
}

/// The steps that cover `rest` km with steps of at most `hf`: their number and length,
/// `hf` itself when it divides `rest` (within $10^{-9}$), `rest / n` otherwise (the steps
/// of `comnum_channel::optical::fiber::Ssfm`).
fn equal_steps(rest: f64, hf: f64) -> (usize, f64) {
    let ratio = rest / hf;
    let whole = ratio.round();
    if whole >= 1.0 && (whole - ratio).abs() <= 1e-9 * ratio {
        (whole as usize, hf)
    } else {
        let n = ratio.ceil().max(1.0);
        (n as usize, rest / n)
    }
}

/// [`Error::Shape`] unless a birefringence applies to two fields and its sections cover
/// `length_km`.
fn check_birefringence(b: &Birefringence<'_>, nf: usize, length_km: f64) -> Result<()> {
    if nf != 2 {
        return Err(Error::Shape("a birefringence needs two polarisations".into()));
    }
    let n = b.rotations.len();
    let covers = b.section_km > 0.0 && ((n as f64) * b.section_km - length_km).abs() <= 1e-9 * length_km.max(1.0);
    if n == 0 || b.dgd_s_per_km.len() != n || !covers {
        return Err(Error::Shape(format!(
            "{n} sections of {} km and {} DGDs for a length of {length_km} km",
            b.section_km,
            b.dgd_s_per_km.len()
        )));
    }
    Ok(())
}

/// The rotation `m` (row-major) applied to $(A_{x}, A_{y})$ at every sample.
fn rotate(m: &[C64; 4], x: &mut [C64], y: &mut [C64]) {
    for (u, v) in x.iter_mut().zip(y.iter_mut()) {
        let (a, b) = (m[0] * *u + m[1] * *v, m[2] * *u + m[3] * *v);
        *u = a;
        *v = b;
    }
}

/// The total power $\sum_{i} |A_{i}|^{2}$ of the fields at every sample.
fn total_power_of<T: AsRef<[C64]>>(fields: &[T], out: &mut [f64]) {
    out.fill(0.0);
    for f in fields {
        for (o, v) in out.iter_mut().zip(f.as_ref()) {
            *o += v.norm_sqr();
        }
    }
}

/// [`kerr_rotation`] of the phase $k |A|^{2} h$ computed in the order of [`SplitStep`] for
/// one field ($k |A|^{2}$ times $h$), and of [`manakov_rotation`] ($k h$ first) for two:
/// the fixed steps give the fields of the former solvers bit for bit.
fn kerr_rotation_scalar_order(fields: &mut [Vec<C64>], k: f64, h: f64) {
    match fields {
        [x] => {
            for v in x.iter_mut() {
                *v *= cis(k * v.norm_sqr() * h);
            }
        }
        [x, y] => manakov_rotation(x, y, k * h),
        _ => {}
    }
}

/// The Kerr rotation $A_{i} \leftarrow A_{i} e^{j k \sum_{i} |A_{i}|^{2}}$ of one field or
/// two ([`manakov_rotation`]).
fn kerr_rotation(fields: &mut [Vec<C64>], k: f64) {
    match fields {
        [x] => {
            for v in x.iter_mut() {
                *v *= cis(k * v.norm_sqr());
            }
        }
        [x, y] => manakov_rotation(x, y, k),
        _ => {}
    }
}

/// The phase per kilometre of the trapezoidal rule, $\phi = k (P_{0} + P_{1}) / 2$.
fn trapezoid(p0: &[f64], p1: &[f64], k: f64, phi: &mut [f64]) {
    for (f, (u, v)) in phi.iter_mut().zip(p0.iter().zip(p1)) {
        *f = k * (u + v) / 2.0;
    }
}
