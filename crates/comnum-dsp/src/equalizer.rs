//! Adaptive equalisation (OptiCommPy `optic.dsp.equalization`).
//!
//! Polarisation demultiplexing is the `2×2` case: there is no optical-specific
//! equaliser.
//!
//! # Adaptive MIMO equaliser
//!
//! [`AdaptiveEqualizer`] is the `n×n` butterfly FIR equaliser of OptiCommPy's
//! `mimoAdaptEqualizer` (time domain, one update per output symbol). The input of `n`
//! modes sampled at `sps` samples per symbol is padded with `⌊L/2⌋` zeros at both ends;
//! output symbol `k` of mode `m` is
//!
//! `y_m[k] = Σ_N Σ_t h_{m,N}[t] x_N[k·sps + t]`,
//!
//! with the centre taps of the diagonal filters initialised at 1. After every output the
//! taps follow `h_{m,N} ← h_{m,N} + μ e_m x_N*` (a stochastic gradient) with the error
//! of the current [`Algorithm`]:
//!
//! - `Cma` (Godard): `e = (R - |y|²) y`, `R = E|s|⁴/E|s|²`;
//! - `Rde` (radius-directed): `e = (R̂² - |y|²) y` with `R̂` the constellation radius
//!   nearest to `|y|`;
//! - `Nlms` (data-aided): `e = (s - y)/‖x‖²` with the reference symbols `s`;
//! - `DdLms` (decision-directed): `e = ŝ - y` with `ŝ` the nearest constellation point.
//!
//! A [`Schedule`] runs several algorithms in sequence (for instance CMA for
//! pre-convergence, then DD-LMS), over several passes of the block if wanted.
//!
//! # Block least-squares equaliser
//!
//! [`ls_equalizer`] is the data-aided equaliser of a block: the FIR filter of $2L + 1$
//! taps at `sps` samples per symbol that minimises $\sum_{k} |x_{k} - \mathbf{w}^{T}
//! \mathbf{r}_{k}|^{2}$ over the known symbols $x_{k}$, $\mathbf{r}_{k}$ the samples of
//! the block (periodic) around the centre of symbol $k$: the normal equations
//! $(\sum_{k} \mathbf{r}_{k}^{*}\mathbf{r}_{k}^{T})\mathbf{w} = \sum_{k} \mathbf{r}_{k}^{*}
//! x_{k}$ solved by Cholesky (the Wiener solution of the block, an upper bound of any
//! adaptive equaliser trained on it). It gives one output sample per symbol, as the
//! fractionally spaced linear equaliser of a coherent receiver (Ip & Kahn, *J. Lightw.
//! Technol.* 26(20), 2008, Fig. 3: residual dispersion, the anti-aliasing filter and the
//! matched filter in one).
//!
//! # The singularity of blind 2 × 2 equalisation
//!
//! Blind criteria such as the CMA are satisfied by any output of constant modulus: both
//! outputs of a polarisation demultiplexer can converge to the same polarisation (the
//! singularity of Kikuchi's analysis), most often when the input polarisations are
//! mixed near 45°. [`Singularity::Kikuchi`] is his two-stage initialisation: during the
//! first `n` output symbols only output 0 adapts; then the filters of output 1 are set
//! to the unitary complement of those of output 0, as the inverse of a unitary Jones
//! matrix requires,
//!
//! $$
//! h_{yx}\[t\] = -h_{xy}^{*}\[L - 1 - t\], \qquad h_{yy}\[t\] = h_{xx}^{*}\[L - 1 - t\],
//! $$
//!
//! (the conjugate of the frequency response, the taps reversed about the centre), so
//! that output 1 starts on the other polarisation; all filters adapt from then on.
//!
//! # Examples
//! ```
//! use comnum_dsp::equalizer::{AdaptiveEqualizer, Algorithm};
//! use comnum_core::rng::Rng;
//! use comnum_math::C64;
//! // QPSK through a 2-tap channel at 1 sample per symbol, equalised blindly by CMA
//! let qpsk: Vec<C64> = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
//!     .iter().map(|&(a, b)| C64::new(a, b) / 2f64.sqrt()).collect();
//! let mut rng = Rng::new(1);
//! let s: Vec<C64> = (0..4000).map(|_| qpsk[rng.below(4)]).collect();
//! let x: Vec<C64> = (0..s.len()).map(|n| s[n] + if n > 0 { s[n - 1] * 0.3 } else { C64::new(0.0, 0.0) }).collect();
//! let eq = AdaptiveEqualizer { n_taps: 11, sps: 1, mu: 1e-3, ..AdaptiveEqualizer::new(Algorithm::Cma) };
//! let out = eq.run(&[x], None, &qpsk).unwrap();
//! let tail = &out.symbols[0][3000..];
//! let spread = tail.iter().map(|v| (v.norm_sqr() - 1.0).abs()).sum::<f64>() / tail.len() as f64;
//! assert!(spread < 0.05);                     // constant modulus restored
//! ```

use crate::adaptive::{ConstantModulus, Decision, ErrorLaw, RadiusDirected, Reference, UpdateRule};
use comnum_core::{Error, Result, ensure, ensure_shape};
use comnum_math::C64;
use comnum_math::linalg::{CMat, solve_hpd};
use comnum_math::stats::Correlation;

/// An update rule of the adaptive equaliser.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    /// Constant modulus algorithm (blind).
    Cma,
    /// Radius-directed equalisation (blind, multi-ring constellations).
    Rde,
    /// Normalised LMS on reference symbols (data-aided).
    Nlms,
    /// Decision-directed LMS.
    DdLms,
}

impl std::str::FromStr for Algorithm {
    type Err = Error;

    /// `cma`, `rde`, `nlms` or `dd-lms` (`dd_lms`, `ddlms`, `dd`), case-insensitive.
    fn from_str(s: &str) -> Result<Algorithm> {
        match s.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "cma" => Ok(Algorithm::Cma),
            "rde" => Ok(Algorithm::Rde),
            "nlms" => Ok(Algorithm::Nlms),
            "ddlms" | "dd" => Ok(Algorithm::DdLms),
            other => Err(Error::Config(format!("unknown algorithm {other:?} (cma, rde, nlms, dd-lms)"))),
        }
    }
}

/// The protection of a blind 2 × 2 equaliser against the singularity (see the module).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Singularity {
    /// No protection: the filters start from the identity.
    #[default]
    None,
    /// Kikuchi's two-stage initialisation: output 0 adapts alone during this number of
    /// output symbols, then output 1 starts as its unitary complement.
    Kikuchi(usize),
}

/// A sequence of `(algorithm, number of output symbols)`; the last entry runs to the end
/// of the block, and every pass of the block restarts the sequence.
pub type Schedule = Vec<(Algorithm, usize)>;

/// The `n×n` butterfly adaptive equaliser (see the module documentation).
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveEqualizer {
    /// Number of taps of every filter.
    pub n_taps: usize,
    /// Input samples per symbol (the output is at one sample per symbol).
    pub sps: usize,
    /// Step size `μ`.
    pub mu: f64,
    /// Step sizes of the stages of the schedule (the last one holds for the following
    /// stages, as OptiCommPy's list of `mu`); empty: `mu` for every stage.
    pub stage_mu: Vec<f64>,
    /// The algorithms and their durations.
    pub schedule: Schedule,
    /// Number of passes over the block (the taps are kept from one pass to the next).
    pub passes: usize,
    /// The protection against the singularity (2 × 2 only).
    pub singularity: Singularity,
    /// The update rule of every stage instead of that of its algorithm (for instance
    /// [`UpdateRule::SignError`]: the sign of the error of CMA or DD-LMS); `None` keeps
    /// the rule of the algorithm ([`Algorithm::update_rule`]).
    pub update_rule: Option<UpdateRule>,
}

impl AdaptiveEqualizer {
    /// 15 taps, 2 samples per symbol, `μ = 10⁻³`, one pass of `alg`.
    pub fn new(alg: Algorithm) -> AdaptiveEqualizer {
        AdaptiveEqualizer {
            n_taps: 15,
            sps: 2,
            mu: 1e-3,
            stage_mu: Vec::new(),
            schedule: vec![(alg, usize::MAX)],
            passes: 1,
            singularity: Singularity::None,
            update_rule: None,
        }
    }
}

/// The output of [`AdaptiveEqualizer::run`].
#[derive(Clone, Debug, PartialEq)]
pub struct EqualizerOutput {
    /// Equalised symbols of every mode (last pass).
    pub symbols: Vec<Vec<C64>>,
    /// Final taps, `taps[m + N·n]` the filter from input mode `N` to output mode `m`.
    pub taps: Vec<Vec<C64>>,
    /// Squared error `|e|²` of every mode and output symbol (last pass).
    pub error: Vec<Vec<f64>>,
}

impl EqualizerOutput {
    /// The correlation of the two outputs above which a blind 2 × 2 equaliser is taken
    /// to have converged to the singularity ([`EqualizerOutput::output_correlation`]).
    pub const SINGULAR_CORRELATION: f64 = 0.5;

    /// The magnitude of the correlation coefficient of the two outputs of a 2 × 2
    /// equaliser over the second half of the block,
    ///
    /// $$
    /// \rho = \frac{\left|\sum_{k} y_{0}\[k\]\, y_{1}^{*}\[k\]\right|}
    /// {\sqrt{\sum_{k} |y_{0}\[k\]|^{2} \sum_{k} |y_{1}\[k\]|^{2}}},
    /// $$
    ///
    /// near 1 when both outputs converged to the same polarisation (the singularity, see
    /// the module), near 0 when they carry independent streams (0 when an output has no
    /// energy, [`Correlation::rho`]); `None` with another number of outputs.
    ///
    /// # Examples
    /// ```
    /// use comnum_dsp::equalizer::EqualizerOutput;
    /// use comnum_math::C64;
    /// let a = vec![C64::new(1.0, 0.0), C64::new(0.0, 1.0), C64::new(-1.0, 0.0), C64::new(0.0, -1.0)];
    /// let out = EqualizerOutput { symbols: vec![a.clone(), a], taps: vec![], error: vec![] };
    /// assert!((out.output_correlation().unwrap() - 1.0).abs() < 1e-15);
    /// ```
    pub fn output_correlation(&self) -> Option<f64> {
        let [a, b] = self.symbols.as_slice() else { return None };
        let h = a.len() / 2;
        Some(Correlation::between(&a[h..], &b[h..]).rho())
    }
}

impl AdaptiveEqualizer {
    /// Equalise the modes `x` (equal lengths; `Vec`s or slices). `reference` gives the
    /// transmitted symbols of every mode (needed by `Nlms`); `points` are the
    /// constellation points (normalised to unit average energy inside), for the radii and
    /// the decisions.
    ///
    /// # Errors
    /// [`Error::Config`] for an invalid configuration or a missing reference,
    /// [`Error::Shape`] for modes of different lengths, a block shorter than the
    /// equaliser or a reference too short.
    pub fn run<M: AsRef<[C64]>>(
        &self,
        x: &[M],
        reference: Option<&[Vec<C64>]>,
        points: &[C64],
    ) -> Result<EqualizerOutput> {
        ensure(!self.schedule.is_empty(), || "empty schedule".into())?;
        let (padded, n_out) = self.pad(x)?;
        let n = x.len();
        if self.schedule.iter().any(|(a, _)| *a == Algorithm::Nlms) {
            let r = reference.ok_or_else(|| Error::Config("NLMS needs reference symbols".into()))?;
            ensure_shape(r.len() == n && r.iter().all(|v| v.len() >= n_out), || {
                format!(
                    "the reference needs {n} modes of at least {n_out} symbols, got {} modes of {:?}",
                    r.len(),
                    r.iter().map(Vec::len).collect::<Vec<_>>()
                )
            })?;
        }
        let laws = Laws::new(points, reference)?;
        let stages: Vec<(&dyn ErrorLaw, UpdateRule, usize)> = self
            .schedule
            .iter()
            .map(|&(alg, n)| (laws.of(alg), self.update_rule.unwrap_or_else(|| alg.update_rule()), n))
            .collect();
        self.adapt(&padded, n_out, &stages)
    }

    /// Equalise the modes `x` with any error law and update rule
    /// ([`crate::adaptive`]), for the whole block and every pass (the schedule and the
    /// stage steps of the equaliser are ignored, `mu` is the step).
    ///
    /// # Errors
    /// As [`AdaptiveEqualizer::run`] for the shapes and the configuration.
    ///
    /// # Examples
    /// ```
    /// use comnum_dsp::adaptive::{ConstantModulus, UpdateRule};
    /// use comnum_dsp::equalizer::{AdaptiveEqualizer, Algorithm};
    /// use comnum_math::C64;
    /// // a sign-error CMA: the sign of the Godard error, a cheaper update
    /// let x: Vec<C64> = (0..2000).map(|n| C64::from_polar(1.0, 0.7 * f64::from(n))).collect();
    /// let eq = AdaptiveEqualizer { n_taps: 5, sps: 1, mu: 1e-3, ..AdaptiveEqualizer::new(Algorithm::Cma) };
    /// let out = eq.run_with(&[x], &ConstantModulus { radius: 1.0 }, UpdateRule::SignError)?;
    /// assert!((out.symbols[0][1999].norm() - 1.0).abs() < 0.05);
    /// # Ok::<(), comnum_core::Error>(())
    /// ```
    pub fn run_with<M: AsRef<[C64]>>(&self, x: &[M], law: &dyn ErrorLaw, rule: UpdateRule) -> Result<EqualizerOutput> {
        let (padded, n_out) = self.pad(x)?;
        let eq = AdaptiveEqualizer { stage_mu: vec![], ..self.clone() };
        eq.adapt(&padded, n_out, &[(law, rule, usize::MAX)])
    }

    /// The modes padded with `⌊L/2⌋` zeros at both ends, and the number of outputs.
    fn pad<M: AsRef<[C64]>>(&self, x: &[M]) -> Result<(Vec<Vec<C64>>, usize)> {
        let n = x.len();
        ensure(n >= 1, || "at least one mode".into())?;
        ensure(self.n_taps >= 1 && self.sps >= 1, || "n_taps and sps must be positive".into())?;
        let len = x[0].as_ref().len();
        ensure_shape(x.iter().all(|m| m.as_ref().len() == len), || {
            format!("the modes have different lengths: {:?}", x.iter().map(|m| m.as_ref().len()).collect::<Vec<_>>())
        })?;
        let pad = self.n_taps / 2;
        let padded: Vec<Vec<C64>> = x
            .iter()
            .map(|m| {
                let mut p = vec![C64::new(0.0, 0.0); pad];
                p.extend_from_slice(m.as_ref());
                p.extend(std::iter::repeat_n(C64::new(0.0, 0.0), pad));
                p
            })
            .collect();
        let plen = padded[0].len();
        ensure_shape(plen >= self.n_taps, || {
            format!("a block of {len} samples is shorter than the {} taps of the equaliser", self.n_taps)
        })?;
        Ok((padded, (plen - self.n_taps) / self.sps + 1))
    }

    /// The adaptation over the stages `(law, rule, number of outputs)` (the last one to
    /// the end of the block), every pass restarting them.
    fn adapt(
        &self,
        padded: &[Vec<C64>],
        n_out: usize,
        stages: &[(&dyn ErrorLaw, UpdateRule, usize)],
    ) -> Result<EqualizerOutput> {
        let n = padded.len();
        let kikuchi = match self.singularity {
            Singularity::Kikuchi(k) => {
                ensure(n == 2, || "the Kikuchi initialisation is for 2 × 2 equalisers".into())?;
                Some(k)
            }
            Singularity::None => None,
        };
        let mut h = Butterfly::identity(n, self.n_taps);
        let mut out = vec![vec![C64::new(0.0, 0.0); n_out]; n];
        let mut err = vec![vec![0.0; n_out]; n];
        let mut y = vec![C64::new(0.0, 0.0); n];
        let mut norm2 = vec![0.0; n];
        for pass in 0..self.passes.max(1) {
            let mut stage = 0usize;
            let mut left = stages[0].2;
            for k in 0..n_out {
                while left == 0 && stage + 1 < stages.len() {
                    stage += 1;
                    left = stages[stage].2;
                }
                left = left.saturating_sub(1);
                let (law, rule, _) = stages[stage];
                let mu = self.stage_mu.get(stage).or(self.stage_mu.last()).copied().unwrap_or(self.mu);
                // the second stage of Kikuchi's initialisation: output 1 as the unitary
                // complement of output 0
                let alone = pass == 0 && kikuchi.is_some_and(|k0| k < k0);
                if pass == 0 && kikuchi == Some(k) {
                    h.unitary_complement();
                }
                let base = k * self.sps;
                h.filter(padded, base, &mut y);
                if rule.needs_energy() {
                    for (e, xin) in norm2.iter_mut().zip(padded) {
                        *e = xin[base..base + self.n_taps].iter().map(C64::norm_sqr).sum();
                    }
                }
                for mo in 0..n {
                    let ym = y[mo];
                    let (e, e_abs2) = law.error(ym, mo, k);
                    out[mo][k] = ym;
                    err[mo][k] = e_abs2;
                    if alone && mo > 0 {
                        continue;
                    }
                    for (mi, xin) in padded.iter().enumerate() {
                        let step = rule.step_size(mu, norm2[mi]);
                        rule.update(h.tap_mut(mo, mi), &xin[base..], e, step);
                    }
                }
            }
        }
        Ok(EqualizerOutput { symbols: out, taps: h.taps, error: err })
    }
}

/// The filters of the `n×n` butterfly: `taps[mo + mi·n]` from input `mi` to output `mo`
/// (the layout of [`EqualizerOutput::taps`]).
struct Butterfly {
    n: usize,
    taps: Vec<Vec<C64>>,
}

impl Butterfly {
    /// The identity: the centre taps of the diagonal filters at 1.
    fn identity(n: usize, n_taps: usize) -> Butterfly {
        let mut taps = vec![vec![C64::new(0.0, 0.0); n_taps]; n * n];
        for m in 0..n {
            taps[m + m * n][n_taps / 2] = C64::new(1.0, 0.0);
        }
        Butterfly { n, taps }
    }

    fn tap(&self, mo: usize, mi: usize) -> &[C64] {
        &self.taps[mo + mi * self.n]
    }

    fn tap_mut(&mut self, mo: usize, mi: usize) -> &mut [C64] {
        &mut self.taps[mo + mi * self.n]
    }

    /// The outputs `y_m = Σ_N Σ_t h_{m,N}[t] x_N[base + t]`.
    fn filter(&self, padded: &[Vec<C64>], base: usize, y: &mut [C64]) {
        for (mo, ym) in y.iter_mut().enumerate() {
            *ym = C64::new(0.0, 0.0);
            for (mi, xin) in padded.iter().enumerate() {
                for (t, x) in self.tap(mo, mi).iter().zip(&xin[base..]) {
                    *ym += t * x;
                }
            }
        }
    }

    /// Kikuchi's second stage (2 × 2): $h_{yx}[t] = -h_{xy}^{*}[L - 1 - t]$,
    /// $h_{yy}[t] = h_{xx}^{*}[L - 1 - t]$.
    fn unitary_complement(&mut self) {
        let (xx, xy) = (self.tap(0, 0).to_vec(), self.tap(0, 1).to_vec());
        let l = xx.len();
        for t in 0..l {
            self.tap_mut(1, 0)[t] = -xy[l - 1 - t].conj();
            self.tap_mut(1, 1)[t] = xx[l - 1 - t].conj();
        }
    }
}

/// The error laws of the [`Algorithm`]s (see [`crate::adaptive`]).
struct Laws<'a> {
    cma: ConstantModulus,
    rde: RadiusDirected,
    decision: Decision,
    reference: Option<Reference<'a>>,
}

impl<'a> Laws<'a> {
    fn new(points: &[C64], reference: Option<&'a [Vec<C64>]>) -> Result<Laws<'a>> {
        Ok(Laws {
            cma: ConstantModulus::for_constellation(points)?,
            rde: RadiusDirected::for_constellation(points)?,
            decision: Decision::for_constellation(points)?,
            reference: reference.map(|symbols| Reference { symbols }),
        })
    }

    /// The law of an algorithm (the reference checked before for `Nlms`).
    fn of(&self, alg: Algorithm) -> &dyn ErrorLaw {
        match alg {
            Algorithm::Cma => &self.cma,
            Algorithm::Rde => &self.rde,
            Algorithm::DdLms => &self.decision,
            Algorithm::Nlms => match &self.reference {
                Some(r) => r,
                None => unreachable!("checked before the passes"),
            },
        }
    }
}

/// The data-aided least-squares FIR equaliser of a periodic block (see the module):
/// `r` at `sps` samples per symbol (`r.len() == sps * x.len()`), the known symbols `x`,
/// the taps spanning `half_span` symbols on each side of the centre of every symbol,
/// sample `centre` (0 to `sps - 1`) of its `sps`; returns the taps and the output, one
/// sample per symbol.
///
/// # Errors
/// [`comnum_core::Error::Shape`] if `r` does not hold `sps` samples per symbol of `x`;
/// [`comnum_core::Error::Config`] for `sps = 0`, `centre >= sps` or a span of more taps
/// than symbols; [`comnum_core::Error::Numerical`] if the normal equations are singular
/// (a block without energy; a diagonal loading of $10^{-9}$ of the mean diagonal makes a
/// short block solvable).
///
/// # Examples
/// ```
/// use comnum_dsp::equalizer::ls_equalizer;
/// use comnum_math::C64;
/// // BPSK at 2 samples per symbol through a 2-tap channel: the equaliser inverts it
/// let x: Vec<C64> = (0..400).map(|k| C64::new(if (k * k * 7 + k / 3) % 5 < 2 { 1.0 } else { -1.0 }, 0.0)).collect();
/// let mut r = vec![C64::new(0.0, 0.0); 800];
/// for k in 0..400 {
///     r[2 * k] += x[k];
///     r[2 * k + 1] += x[k] * 0.5;
///     r[(2 * k + 3) % 800] += x[k] * 0.3;
/// }
/// let (taps, y) = ls_equalizer(&r, 2, &x, 3, 0)?;
/// assert_eq!(taps.len(), 13);
/// assert!(y.iter().zip(&x).all(|(a, b)| (a - b).norm() < 1e-9));
/// # Ok::<(), comnum_core::Error>(())
/// ```
pub fn ls_equalizer(r: &[C64], sps: usize, x: &[C64], half_span: usize, centre: usize) -> Result<(Vec<C64>, Vec<C64>)> {
    ensure(sps >= 1 && centre < sps, || {
        format!("sps must be positive and the centre below it, got {sps} and {centre}")
    })?;
    ensure_shape(r.len() == sps * x.len(), || {
        format!("{} samples is not {sps} per symbol of {} symbols", r.len(), x.len())
    })?;
    let n_taps = 2 * half_span * sps + 1;
    ensure(n_taps <= r.len(), || format!("{n_taps} taps for a block of {} samples", r.len()))?;
    let n = r.len();
    let at = |k: usize, t: usize| r[(k * sps + centre + n + t - half_span * sps) % n];
    let mut a = CMat::zeros((n_taps, n_taps));
    let mut b = CMat::zeros((n_taps, 1));
    for (k, xk) in x.iter().enumerate() {
        let v: Vec<C64> = (0..n_taps).map(|t| at(k, t)).collect();
        for i in 0..n_taps {
            let ci = v[i].conj();
            b[[i, 0]] += ci * xk;
            for j in 0..n_taps {
                a[[i, j]] += ci * v[j];
            }
        }
    }
    // a diagonal loading of 1e-9 of the mean diagonal keeps a block of few symbols solvable
    let load = 1e-9 * (0..n_taps).map(|i| a[[i, i]].re).sum::<f64>() / n_taps as f64;
    for i in 0..n_taps {
        a[[i, i]] += load;
    }
    let w = solve_hpd(&a.view(), &b.view())?;
    let taps: Vec<C64> = (0..n_taps).map(|i| w[[i, 0]]).collect();
    let y = (0..x.len()).map(|k| (0..n_taps).map(|t| taps[t] * at(k, t)).sum()).collect();
    Ok((taps, y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use comnum_core::rng::Rng;

    #[test]
    fn a_reference_shorter_than_the_signal_stops_the_training() {
        use crate::adaptive::{Reference, UpdateRule};
        let x: Vec<C64> = (0..64).map(|n| C64::from_polar(1.0, 0.7 * f64::from(n))).collect();
        let symbols = vec![x[..4].to_vec()];
        let eq = AdaptiveEqualizer { n_taps: 5, sps: 1, mu: 1e-2, ..AdaptiveEqualizer::new(Algorithm::Nlms) };
        let out = eq.run_with(&[x], &Reference { symbols: &symbols }, UpdateRule::Normalized).unwrap();
        assert_eq!(out.symbols[0].len(), 64);
        assert!(out.symbols[0].iter().all(|v| v.re.is_finite() && v.im.is_finite()));
    }

    #[test]
    fn nlms_converges_to_the_inverse_and_dd_lms_tracks() {
        // 16-QAM, unit energy
        let c: Vec<C64> =
            (0..16).map(|i| C64::new(f64::from(2 * (i % 4) - 3), f64::from(2 * (i / 4) - 3)) / 10f64.sqrt()).collect();
        let mut rng = Rng::new(3);
        let s: Vec<C64> = (0..6000).map(|_| c[rng.below(16)]).collect();
        let x: Vec<C64> = (0..s.len())
            .map(|n| {
                s[n] * C64::new(0.9, 0.1) + if n > 0 { s[n - 1] * C64::new(0.2, -0.1) } else { C64::new(0.0, 0.0) }
            })
            .collect();
        let eq = AdaptiveEqualizer {
            n_taps: 9,
            sps: 1,
            mu: 2e-2,
            stage_mu: Vec::new(),
            schedule: vec![(Algorithm::Nlms, 2000), (Algorithm::DdLms, usize::MAX)],
            passes: 1,
            singularity: Singularity::None,
            update_rule: None,
        };
        let out = eq.run(&[x], Some(std::slice::from_ref(&s)), &c).unwrap();
        let tail = 5000..6000;
        let mse: f64 = tail.clone().map(|k| (out.symbols[0][k] - s[k]).norm_sqr()).sum::<f64>() / 1000.0;
        assert!(mse < 1e-3, "mse {mse}");
    }

    #[test]
    fn kikuchi_initialisation_avoids_the_singularity_of_the_45_degree_rotation() {
        let qpsk: Vec<C64> = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
            .iter()
            .map(|&(a, b)| C64::new(a, b) / 2f64.sqrt())
            .collect();
        let (c, sn) = (std::f64::consts::FRAC_PI_4.cos(), std::f64::consts::FRAC_PI_4.sin());
        let corr = |kikuchi: Singularity, seed: u64| {
            let mut rng = Rng::new(seed);
            let n = 6000;
            let a: Vec<C64> = (0..n).map(|_| qpsk[rng.below(4)]).collect();
            let b: Vec<C64> = (0..n).map(|_| qpsk[rng.below(4)]).collect();
            let x: Vec<C64> = a.iter().zip(&b).map(|(u, v)| u * c - v * sn + rng.complex_gaussian() * 0.05).collect();
            let y: Vec<C64> = a.iter().zip(&b).map(|(u, v)| u * sn + v * c + rng.complex_gaussian() * 0.05).collect();
            let eq = AdaptiveEqualizer {
                n_taps: 5,
                sps: 1,
                mu: 2e-3,
                singularity: kikuchi,
                ..AdaptiveEqualizer::new(Algorithm::Cma)
            };
            let out = eq.run(&[x, y], None, &qpsk).unwrap();
            let (u, v) = (&out.symbols[0][4000..], &out.symbols[1][4000..]);
            let cc: C64 = u.iter().zip(v).map(|(p, q)| p * q.conj()).sum();
            let e = |w: &[C64]| w.iter().map(C64::norm_sqr).sum::<f64>();
            cc.norm() / (e(u) * e(v)).sqrt()
        };
        // from the identity, many realisations end with both outputs on one
        // polarisation (correlation above 0.5); none with Kikuchi's initialisation
        let singular = |init: Singularity| (0..20).filter(|&s| corr(init, s) > 0.5).count();
        assert!(singular(Singularity::None) >= 5);
        assert_eq!(singular(Singularity::Kikuchi(2000)), 0);
    }
}
