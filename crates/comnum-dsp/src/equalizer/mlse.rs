//! Maximum-likelihood sequence estimation over a known channel of $L$ symbol-spaced
//! taps: the Viterbi algorithm on the trellis of the channel memory (Forney 1972), with
//! the soft outputs of Hagenauer and Hoeher (1989) for the decoder that follows.
//!
//! # Model
//!
//! The samples at one per symbol are $r_{k} = \sum_{l=0}^{L-1} h_{l} s_{k-l} + n_{k}$
//! with $s_{k}$ in a constellation of $M$ points and white Gaussian noise. The state of
//! the trellis is the $L - 1$ previous symbols ($M^{L-1}$ states), the branch of the
//! symbol $s_{k}$ has the metric
//!
//! $$
//! \gamma_{k}(s, u) = -\left| r_{k} - h_{0} a_{u} - \sum_{l=1}^{L-1} h_{l}\,
//! a_{\mathrm{digit}_{l}(s)} \right|^{2},
//! $$
//!
//! and the path of largest accumulated metric is the maximum-likelihood sequence (the
//! Euclidean-distance form of Forney's detector, his whitened matched filter being the
//! sampled channel here). The past before the block is unknown: every state starts
//! with the same metric; the end is free. The log-likelihood difference between the
//! decided symbol and its best competitor is $\Delta_{k}/(2\sigma^{2})$, $\sigma^{2}$
//! the variance of the complex noise ($\Delta_{k}$ the metric difference of
//! [`comnum_math::viterbi::sova`]): for two points it is the magnitude of the
//! log-likelihood ratio of the symbol.
//!
//! # Examples
//! ```
//! use comnum_dsp::equalizer::mlse::Mlse;
//! use comnum_math::C64;
//! // 2-PAM through 1 + 0.5 z⁻¹, noiseless
//! let pts = [C64::new(-1.0, 0.0), C64::new(1.0, 0.0)];
//! let idx = [0usize, 1, 1, 0, 1, 0, 0, 1];
//! let mut r = Vec::new();
//! for k in 0..idx.len() {
//!     let past = if k > 0 { pts[idx[k - 1]] } else { C64::new(0.0, 0.0) };
//!     r.push(pts[idx[k]] + past * 0.5);
//! }
//! let mlse = Mlse::new(&[C64::new(1.0, 0.0), C64::new(0.5, 0.0)], &pts)?;
//! assert_eq!(mlse.detect(&r)?, idx);
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//! 1. G. D. Forney, "Maximum-likelihood sequence estimation of digital sequences in the
//!    presence of intersymbol interference," *IEEE Trans. Inf. Theory* 18(3), 363–378,
//!    1972.
//! 2. J. Hagenauer and P. Hoeher, "A Viterbi algorithm with soft-decision outputs and its
//!    applications," *Proc. IEEE GLOBECOM*, 1680–1686, 1989.

use comnum_core::{Result, ensure};
use comnum_math::C64;
use comnum_math::viterbi::{self as vit, StateMachine};

/// The maximum-likelihood sequence detector of a known channel (see the module).
#[derive(Clone, Debug, PartialEq)]
pub struct Mlse {
    taps: Vec<C64>,
    points: Vec<C64>,
    machine: StateMachine,
    /// The noiseless sample of every branch, `s * M + u`.
    expected: Vec<C64>,
}

impl Mlse {
    /// The detector of the channel `taps` ($h_{0}, \dots, h_{L-1}$, symbol-spaced) and
    /// the constellation `points`.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for no tap, fewer than two points, or a trellis of
    /// more than $2^{31}$ branches ($M^{L}$).
    pub fn new(taps: &[C64], points: &[C64]) -> Result<Mlse> {
        ensure(!taps.is_empty(), || "the channel needs a tap".into())?;
        let m = points.len();
        let machine = StateMachine::shift_register(m, taps.len() - 1)?;
        let sm = &machine;
        let expected = (0..sm.n_states())
            .flat_map(|s| {
                (0..m).map(move |u| {
                    taps[0] * points[u]
                        + taps[1..].iter().enumerate().map(|(l, h)| h * points[sm.digit(s, l + 1)]).sum::<C64>()
                })
            })
            .collect();
        Ok(Mlse { taps: taps.to_vec(), points: points.to_vec(), machine, expected })
    }

    /// The channel taps.
    pub fn taps(&self) -> &[C64] {
        &self.taps
    }

    /// The constellation.
    pub fn points(&self) -> &[C64] {
        &self.points
    }

    /// Number of states $M^{L-1}$ of the trellis.
    pub fn n_states(&self) -> usize {
        self.machine.n_states()
    }

    fn gammas<'a>(&'a self, r: &'a [C64]) -> impl FnMut(usize, &mut [f64]) + 'a {
        move |k, g| {
            for (gv, e) in g.iter_mut().zip(&self.expected) {
                *gv = -(r[k] - e).norm_sqr();
            }
        }
    }

    /// The maximum-likelihood sequence of the samples `r` (one per symbol), as indices
    /// of the constellation.
    ///
    /// # Errors
    /// None for a non-empty input (an empty one gives an empty sequence).
    pub fn detect(&self, r: &[C64]) -> Result<Vec<usize>> {
        let start = vec![0.0; self.machine.n_states()];
        Ok(vit::viterbi(&self.machine, r.len(), &start, self.gammas(r), None)?.0)
    }

    /// The sequence of [`Mlse::detect`] and the reliability of every symbol, the
    /// log-likelihood difference $\Delta_{k}/(2\sigma^{2})$ with its best competitor
    /// (see the module), for the noise variance `noise_var` $\sigma^{2}$ of the samples.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for a non-positive noise variance.
    pub fn detect_soft(&self, r: &[C64], noise_var: f64) -> Result<(Vec<usize>, Vec<f64>)> {
        ensure(noise_var > 0.0, || format!("the noise variance {noise_var} must be positive"))?;
        let start = vec![0.0; self.machine.n_states()];
        let (path, delta) = vit::sova(&self.machine, r.len(), &start, self.gammas(r), None)?;
        Ok((path, delta.iter().map(|d| d / (2.0 * noise_var)).collect()))
    }

    /// The log-likelihood ratios $\ln[P(s_{k} = a_{0}) / P(s_{k} = a_{1})]$ of a binary
    /// constellation from [`Mlse::detect_soft`]: positive for the first point.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] if the constellation has not two points, or as
    /// [`Mlse::detect_soft`].
    pub fn llr(&self, r: &[C64], noise_var: f64) -> Result<Vec<f64>> {
        ensure(self.points.len() == 2, || format!("LLRs need two points, not {}", self.points.len()))?;
        let (path, rel) = self.detect_soft(r, noise_var)?;
        Ok(path.iter().zip(&rel).map(|(&u, &d)| if u == 0 { d } else { -d }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use comnum_core::rng::Rng;
    use comnum_math::qfunc;

    fn pam(m: usize) -> Vec<C64> {
        (0..m).map(|i| C64::new(2.0 * i as f64 - (m as f64 - 1.0), 0.0)).collect()
    }

    fn channel(s: &[C64], h: &[C64]) -> Vec<C64> {
        (0..s.len()).map(|k| h.iter().enumerate().filter(|(l, _)| *l <= k).map(|(l, hv)| hv * s[k - l]).sum()).collect()
    }

    #[test]
    fn the_noiseless_sequence_is_recovered_for_pam4_over_three_taps() {
        let pts = pam(4);
        let h = [C64::new(1.0, 0.0), C64::new(0.6, 0.1), C64::new(-0.3, 0.0)];
        let mlse = Mlse::new(&h, &pts).unwrap();
        assert_eq!(mlse.n_states(), 16);
        let mut rng = Rng::new(9);
        let idx: Vec<usize> = (0..500).map(|_| rng.below(4)).collect();
        let s: Vec<C64> = idx.iter().map(|&i| pts[i]).collect();
        let r = channel(&s, &h);
        // the unknown past before the block costs nothing here (the first symbols see
        // the ISI of a zero past, which the all-equal start metrics absorb)
        let out = mlse.detect(&r).unwrap();
        assert_eq!(out, idx);
        assert!(Mlse::new(&[], &pts).is_err());
        assert!(Mlse::new(&h, &pts[..1]).is_err());
        assert!(Mlse::new(&vec![C64::new(1.0, 0.0); 40], &pts).is_err());
        assert_eq!(mlse.detect(&[]).unwrap().len(), 0);
    }

    #[test]
    fn the_error_rate_sits_at_the_matched_filter_bound_at_high_snr() {
        // 2-PAM over 1 + 0.5 z⁻¹: the minimum distance of an error event is that of a
        // single error, d² = 4‖h‖² (the double error gives 6), so the MLSE reaches the
        // matched-filter bound Q(‖h‖/σ) up to its multiplicity (Forney 1972, Sec. V)
        let pts = pam(2);
        let h = [C64::new(1.0, 0.0), C64::new(0.5, 0.0)];
        let mlse = Mlse::new(&h, &pts).unwrap();
        let mut rng = Rng::new(21);
        let n = 200_000;
        let idx: Vec<usize> = (0..n).map(|_| rng.below(2)).collect();
        let s: Vec<C64> = idx.iter().map(|&i| pts[i]).collect();
        let sigma = 0.36;
        let r: Vec<C64> = channel(&s, &h).iter().map(|v| v + C64::new(sigma * rng.gaussian(), 0.0)).collect();
        let out = mlse.detect(&r).unwrap();
        let errors = out.iter().zip(&idx).filter(|(a, b)| a != b).count();
        let ber = errors as f64 / n as f64;
        let mfb = qfunc((h[0].norm_sqr() + h[1].norm_sqr()).sqrt() / sigma);
        assert!(ber >= 0.8 * mfb && ber <= 2.5 * mfb, "{ber} against the bound {mfb}");
        // a symbol-by-symbol slicer on the same samples does far worse
        let slicer = comnum_math::nearest::Slicer::new(&pts);
        let hard = r.iter().zip(&idx).filter(|(v, b)| slicer.nearest(**v) != **b).count() as f64 / n as f64;
        assert!(hard > 5.0 * ber, "{hard} against {ber}");
    }

    #[test]
    fn the_soft_outputs_follow_the_decisions_and_scale_with_the_noise() {
        let pts = pam(2);
        let h = [C64::new(1.0, 0.0), C64::new(0.4, 0.0)];
        let mlse = Mlse::new(&h, &pts).unwrap();
        let mut rng = Rng::new(5);
        let idx: Vec<usize> = (0..2000).map(|_| rng.below(2)).collect();
        let s: Vec<C64> = idx.iter().map(|&i| pts[i]).collect();
        let r: Vec<C64> = channel(&s, &h).iter().map(|v| v + C64::new(0.3 * rng.gaussian(), 0.0)).collect();
        let hard = mlse.detect(&r).unwrap();
        let (path, rel) = mlse.detect_soft(&r, 0.09).unwrap();
        assert_eq!(path, hard);
        assert!(rel.iter().all(|v| *v >= 0.0));
        let llr = mlse.llr(&r, 0.09).unwrap();
        assert!(llr.iter().zip(&path).all(|(l, &u)| (*l >= 0.0) == (u == 0)));
        // a wrong decision is less reliable on average than a right one
        let (mut wrong, mut right) = ((0.0, 0usize), (0.0, 0usize));
        for ((&p, &i), &d) in path.iter().zip(&idx).zip(&rel) {
            if p == i {
                right = (right.0 + d, right.1 + 1);
            } else {
                wrong = (wrong.0 + d, wrong.1 + 1);
            }
        }
        let (mean_wrong, mean_right) = (wrong.0 / wrong.1 as f64, right.0 / right.1 as f64);
        assert!(wrong.1 > 0 && mean_wrong < 0.3 * mean_right);
        // twice the noise variance halves the log-likelihood differences
        let (_, half) = mlse.detect_soft(&r, 0.18).unwrap();
        assert!(rel.iter().zip(&half).all(|(a, b)| (a - 2.0 * b).abs() < 1e-9 * a.max(1.0)));
        assert!(mlse.detect_soft(&r, 0.0).is_err());
        assert!(Mlse::new(&h, &pam(4)).unwrap().llr(&r, 1.0).is_err());
    }
}
