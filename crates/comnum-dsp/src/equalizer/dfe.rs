//! The decision-feedback equaliser: a feedforward FIR on the samples and a feedback FIR
//! on the past decisions, adapted by a stochastic gradient (Proakis & Salehi, *Digital
//! Communications*, 5th ed., Sec. 9.5-1; Austin 1967). The feedforward filter is that of
//! [`AdaptiveEqualizer`](super::AdaptiveEqualizer), one stream; the feedback filter
//! cancels the post-cursor interference from the symbols already decided, without
//! enhancing the noise as a linear equaliser must on a channel with a spectral null.
//!
//! # Model
//!
//! Output symbol $k$ of the input $x$ at `sps` samples per symbol is
//!
//! $$
//! y_{k} = \sum_{t=0}^{N_{f}-1} f_{t}\, x\[k \cdot \mathrm{sps} + t\] -
//! \sum_{j=1}^{N_{b}} b_{j}\, \hat{s}_{k-j},
//! $$
//!
//! with $\hat{s}_{k}$ the decision on $y_{k}$ (the point of the constellation nearest to
//! it), or the known symbol $s_{k}$ during the first `training` symbols. Both filters
//! follow the error $e_{k} = \hat{s}_{k} - y_{k}$ by the [`UpdateRule`] of the
//! equaliser, $f \leftarrow f + \mu e_{k} x^{*}$, $b \leftarrow b - \mu e_{k}
//! \hat{s}^{*}$ (the feedback input being $-\hat{s}$). The input is padded with $\lfloor
//! N_{f}/2 \rfloor$ zeros at both ends and the centre tap of $f$ starts at 1, as the
//! linear equaliser does. The decisions and the symbols are on the scale of the points
//! given (no normalisation).
//!
//! # Examples
//! ```
//! use comnum_dsp::equalizer::dfe::Dfe;
//! use comnum_core::rng::Rng;
//! use comnum_math::C64;
//! // 2-PAM through 1 + 0.9 z⁻¹ (a near null at Nyquist), decided after a training block
//! let pts = [C64::new(-1.0, 0.0), C64::new(1.0, 0.0)];
//! let mut rng = Rng::new(2);
//! let idx: Vec<usize> = (0..3000).map(|_| rng.below(2)).collect();
//! let s: Vec<C64> = idx.iter().map(|&i| pts[i]).collect();
//! let x: Vec<C64> = (0..s.len()).map(|n| s[n] + if n > 0 { s[n - 1] * 0.9 } else { C64::new(0.0, 0.0) }).collect();
//! let dfe = Dfe { mu: 5e-3, training: 500, ..Dfe::new(7, 2) };
//! let out = dfe.run(&x, Some(&s), &pts)?;
//! let errors = out.decisions[1000..].iter().zip(&idx[1000..]).filter(|(a, b)| a != b).count();
//! assert_eq!(errors, 0);
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//! 1. J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008,
//!    Sec. 9.5 (decision-feedback equalization) and Sec. 10.2 (its adaptation).
//! 2. M. E. Austin, "Decision-feedback equalization for digital communication over
//!    dispersive channels," MIT Research Laboratory of Electronics, Tech. Rep. 437, 1967.

use crate::adaptive::UpdateRule;
use comnum_core::{Result, ensure, ensure_shape};
use comnum_math::C64;
use comnum_math::nearest::Slicer;

/// A decision-feedback equaliser (see the module).
#[derive(Clone, Debug, PartialEq)]
pub struct Dfe {
    /// Taps $N_{f}$ of the feedforward filter.
    pub n_feedforward: usize,
    /// Taps $N_{b}$ of the feedback filter (0 for a linear equaliser).
    pub n_feedback: usize,
    /// Input samples per symbol.
    pub sps: usize,
    /// Step size $\mu$.
    pub mu: f64,
    /// The update rule of both filters.
    pub rule: UpdateRule,
    /// Number of symbols decided from the reference (data-aided) before the decisions
    /// drive the adaptation; no more than the reference holds.
    pub training: usize,
}

/// The output of [`Dfe::run`].
#[derive(Clone, Debug, PartialEq)]
pub struct DfeOutput {
    /// The equalised symbols $y_{k}$, one per output symbol.
    pub symbols: Vec<C64>,
    /// The decisions $\hat{s}_{k}$ as indices of the constellation (the reference
    /// symbols during the training, by their nearest point).
    pub decisions: Vec<usize>,
    /// The final feedforward taps $f$.
    pub feedforward: Vec<C64>,
    /// The final feedback taps $b$ ($y = f \cdot x - \sum_{j} b_{j} \hat{s}_{k-j}$).
    pub feedback: Vec<C64>,
    /// The squared error $|e_{k}|^{2}$ of every output.
    pub error: Vec<f64>,
}

impl Dfe {
    /// A DFE of `n_feedforward` and `n_feedback` taps at 1 sample per symbol, $\mu =
    /// 10^{-3}$, LMS, decision-directed from the first symbol.
    pub fn new(n_feedforward: usize, n_feedback: usize) -> Dfe {
        Dfe { n_feedforward, n_feedback, sps: 1, mu: 1e-3, rule: UpdateRule::Lms, training: 0 }
    }

    /// Equalise `x` with the constellation `points`; `reference` holds the transmitted
    /// symbols of the training (at least `training` of them).
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for no feedforward tap, no sample per symbol, a
    /// non-positive step, an empty constellation, or a training longer than the
    /// reference; [`comnum_core::Error::Shape`] for an input shorter than the filter.
    pub fn run(&self, x: &[C64], reference: Option<&[C64]>, points: &[C64]) -> Result<DfeOutput> {
        ensure(self.n_feedforward >= 1 && self.sps >= 1, || {
            "the DFE needs a feedforward tap and a sample per symbol".into()
        })?;
        ensure(self.mu > 0.0 && self.mu.is_finite(), || format!("the step size {} must be positive", self.mu))?;
        ensure(!points.is_empty(), || "empty constellation".into())?;
        let n_ref = reference.map_or(0, <[C64]>::len);
        ensure(self.training <= n_ref, || {
            format!("a training of {} symbols with a reference of {n_ref}", self.training)
        })?;
        let pad = self.n_feedforward / 2;
        let mut padded = vec![C64::new(0.0, 0.0); pad];
        padded.extend_from_slice(x);
        padded.extend(std::iter::repeat_n(C64::new(0.0, 0.0), pad));
        ensure_shape(padded.len() >= self.n_feedforward, || {
            format!("a block of {} samples is shorter than the {} taps of the equaliser", x.len(), self.n_feedforward)
        })?;
        let n_out = (padded.len() - self.n_feedforward) / self.sps + 1;
        let slicer = Slicer::new(points);
        let mut f = vec![C64::new(0.0, 0.0); self.n_feedforward];
        f[pad] = C64::new(1.0, 0.0);
        let mut b = vec![C64::new(0.0, 0.0); self.n_feedback];
        // the feedback input: minus the past decisions, the most recent first
        let mut past = vec![C64::new(0.0, 0.0); self.n_feedback];
        let mut out = DfeOutput {
            symbols: Vec::with_capacity(n_out),
            decisions: Vec::with_capacity(n_out),
            feedforward: vec![],
            feedback: vec![],
            error: Vec::with_capacity(n_out),
        };
        for k in 0..n_out {
            let xs = &padded[k * self.sps..k * self.sps + self.n_feedforward];
            let y: C64 = f.iter().zip(xs).map(|(a, v)| a * v).sum::<C64>()
                + b.iter().zip(&past).map(|(a, v)| a * v).sum::<C64>();
            let target = match reference {
                Some(r) if k < self.training => r[k],
                _ => points[slicer.nearest(y)],
            };
            let e = target - y;
            let energy = if self.rule.needs_energy() {
                xs.iter().map(C64::norm_sqr).sum::<f64>() + past.iter().map(C64::norm_sqr).sum::<f64>()
            } else {
                0.0
            };
            let step = self.rule.step_size(self.mu, energy);
            self.rule.update(&mut f, xs, e, step);
            self.rule.update(&mut b, &past, e, step);
            out.symbols.push(y);
            out.decisions.push(slicer.nearest(target));
            out.error.push(e.norm_sqr());
            if self.n_feedback > 0 {
                past.rotate_right(1);
                past[0] = -target;
            }
        }
        out.feedforward = f;
        out.feedback = b;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use comnum_core::rng::Rng;

    fn pam(m: usize) -> Vec<C64> {
        (0..m).map(|i| C64::new(2.0 * i as f64 - (m as f64 - 1.0), 0.0)).collect()
    }

    #[test]
    fn without_feedback_taps_the_dfe_is_the_linear_equaliser() {
        let pts = pam(4);
        let mut rng = Rng::new(1);
        let s: Vec<C64> = (0..400).map(|_| pts[rng.below(4)]).collect();
        let x: Vec<C64> =
            (0..s.len()).map(|n| s[n] + if n > 0 { s[n - 1] * 0.2 } else { C64::new(0.0, 0.0) }).collect();
        let dfe = Dfe { mu: 2e-3, training: 400, ..Dfe::new(5, 0) };
        let out = dfe.run(&x, Some(&s), &pts).unwrap();
        let eq = crate::equalizer::AdaptiveEqualizer {
            n_taps: 5,
            sps: 1,
            mu: 2e-3,
            ..crate::equalizer::AdaptiveEqualizer::new(crate::equalizer::Algorithm::Nlms)
        };
        let reference = vec![s.clone()];
        let lin = eq
            .run_with(std::slice::from_ref(&x), &crate::adaptive::Reference { symbols: &reference }, UpdateRule::Lms)
            .unwrap();
        assert_eq!(out.symbols.len(), lin.symbols[0].len());
        assert!(out.symbols.iter().zip(&lin.symbols[0]).all(|(a, b)| (a - b).norm() < 1e-12));
        assert_eq!(out.feedback.len(), 0);
    }

    #[test]
    fn the_feedback_cancels_a_post_cursor_the_linear_equaliser_cannot() {
        // 1 + z⁻¹: a null at Nyquist, no linear inverse; the DFE learns b = [1]
        let pts = pam(2);
        let mut rng = Rng::new(4);
        let idx: Vec<usize> = (0..4000).map(|_| rng.below(2)).collect();
        let s: Vec<C64> = idx.iter().map(|&i| pts[i]).collect();
        let x: Vec<C64> = (0..s.len()).map(|n| s[n] + if n > 0 { s[n - 1] } else { C64::new(0.0, 0.0) }).collect();
        // one feedforward tap: the only solution is f = 1, b = [1, 0]
        let one = Dfe { mu: 3e-3, training: 1000, ..Dfe::new(1, 2) };
        let out = one.run(&x, Some(&s), &pts).unwrap();
        assert!((out.feedback[0] - C64::new(1.0, 0.0)).norm() < 0.05, "{:?}", out.feedback);
        assert!(out.feedback[1].norm() < 0.05);
        assert!((out.feedforward[0] - C64::new(1.0, 0.0)).norm() < 0.05);
        let dfe = Dfe { mu: 3e-3, training: 1000, ..Dfe::new(5, 2) };
        let out = dfe.run(&x, Some(&s), &pts).unwrap();
        let errors = out.decisions[2000..].iter().zip(&idx[2000..]).filter(|(a, b)| a != b).count();
        assert_eq!(errors, 0);
        let tail = out.error[3000..].iter().sum::<f64>() / 1000.0;
        assert!(tail < 1e-2, "{tail}");
        // normalised and sign-error updates converge too
        for rule in [UpdateRule::Normalized, UpdateRule::SignError] {
            let d = Dfe { rule, mu: if rule == UpdateRule::Normalized { 0.05 } else { 1e-3 }, ..dfe.clone() };
            let o = d.run(&x, Some(&s), &pts).unwrap();
            assert_eq!(o.decisions[3000..], idx[3000..], "{rule:?}");
        }
    }

    #[test]
    fn invalid_settings_are_refused() {
        let pts = pam(2);
        let x = vec![C64::new(1.0, 0.0); 10];
        assert!(Dfe::new(0, 1).run(&x, None, &pts).is_err());
        assert!(Dfe { mu: 0.0, ..Dfe::new(3, 1) }.run(&x, None, &pts).is_err());
        assert!(Dfe::new(3, 1).run(&x, None, &[]).is_err());
        assert!(Dfe { training: 4, ..Dfe::new(3, 1) }.run(&x, Some(&x[..2]), &pts).is_err());
        assert!(Dfe::new(31, 1).run(&[], None, &pts).is_err());
        assert!(Dfe::new(3, 1).run(&x, None, &pts).is_ok());
    }
}
