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
//! it), or the known symbol $s_{k}$ during the first `training` symbols. The two filters
//! are one tap vector $[f;\, -b]$ on the regressor $[x;\, \hat{s}]$ and follow the error
//! $e_{k} = \hat{s}_{k} - y_{k}$ by the [`UpdateRule`] of the equaliser (a stochastic
//! gradient). The input is padded with $\lfloor N_{f}/2 \rfloor$ zeros at both ends and
//! the centre tap of $f$ starts at 1, as the linear equaliser does. The decisions and the
//! symbols are on the scale of the points given (no normalisation).
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
//! let x = comnum_dsp::filter::convolve_full(&[C64::new(1.0, 0.0), C64::new(0.9, 0.0)], &s)[..s.len()].to_vec();
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
use crate::equalizer::pad_block;
use comnum_core::{Result, ensure};
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
        let (padded, n_out) = pad_block(x, self.n_feedforward, self.sps)?;
        let (nf, nb) = (self.n_feedforward, self.n_feedback);
        let slicer = Slicer::new(points);
        // the regressor [x window ; past decisions, the most recent first] and its taps
        // [f ; -b], the centre tap of f at 1
        let mut regressor = vec![C64::new(0.0, 0.0); nf + nb];
        let mut taps = vec![C64::new(0.0, 0.0); nf + nb];
        taps[nf / 2] = C64::new(1.0, 0.0);
        let (mut symbols, mut decisions, mut error) =
            (Vec::with_capacity(n_out), Vec::with_capacity(n_out), Vec::with_capacity(n_out));
        for k in 0..n_out {
            regressor[..nf].copy_from_slice(&padded[k * self.sps..k * self.sps + nf]);
            let y: C64 = taps.iter().zip(&regressor).map(|(a, v)| a * v).sum();
            let index = match reference {
                Some(r) if k < self.training => slicer.nearest(r[k]),
                _ => slicer.nearest(y),
            };
            let target = points[index];
            let e = target - y;
            let energy = if self.rule.needs_energy() { regressor.iter().map(C64::norm_sqr).sum() } else { 0.0 };
            self.rule.update(&mut taps, &regressor, e, self.rule.step_size(self.mu, energy));
            symbols.push(y);
            decisions.push(index);
            error.push(e.norm_sqr());
            if nb > 0 {
                regressor[nf..].rotate_right(1);
                regressor[nf] = target;
            }
        }
        let feedback = taps[nf..].iter().map(|v| -v).collect();
        taps.truncate(nf);
        Ok(DfeOutput { symbols, decisions, feedforward: taps, feedback, error })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equalizer::pam;
    use crate::filter::convolve_full;
    use comnum_core::rng::Rng;

    /// `s` through the causal channel `h`, the tail of the convolution dropped.
    fn through(s: &[C64], h: &[C64]) -> Vec<C64> {
        convolve_full(h, s)[..s.len()].to_vec()
    }

    #[test]
    fn without_feedback_taps_the_dfe_is_the_linear_equaliser() {
        let pts = pam(4);
        let mut rng = Rng::new(1);
        let s: Vec<C64> = (0..400).map(|_| pts[rng.below(4)]).collect();
        let x = through(&s, &[C64::new(1.0, 0.0), C64::new(0.2, 0.0)]);
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
        let x = through(&s, &[C64::new(1.0, 0.0), C64::new(1.0, 0.0)]);
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
        // the normalised step converges too (the rules themselves are tested in `adaptive`)
        let o = Dfe { rule: UpdateRule::Normalized, mu: 0.05, ..dfe }.run(&x, Some(&s), &pts).unwrap();
        assert_eq!(o.decisions[3000..], idx[3000..]);
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
