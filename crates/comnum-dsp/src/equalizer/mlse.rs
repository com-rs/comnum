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
//! decided symbol and its best competitor is $\Delta_{k}/\sigma^{2}$ with $\Delta_{k}$
//! the metric difference of [`comnum_math::viterbi::sova`] and $\sigma^{2} = E|n_{k}|^{2}$
//! the variance of the complex noise, the convention of the demappers of comnum (a real
//! channel passes twice the variance of its real noise); for two points it is the
//! magnitude of the log-likelihood ratio of the symbol.
//!
//! A channel that is not linear in the symbols (the photocurrent of a dispersed
//! intensity modulation, a square law) has the same trellis: its noiseless sample
//! depends on the current and the $L - 1$ previous symbols through a table, which
//! [`Mlse::from_table`] takes and [`Mlse::fit`] measures on a training sequence (the
//! stored signal vectors of Winters & Gitlin 1990, Sec. III-C). The worst-case eye of
//! such a detector over a window of $W$ samples is its minimum Euclidean distance
//! between two paths that leave a state with different symbols (their eq. (17)),
//! [`Mlse::min_distance`]; for $W = 1$ it is the eye of a slicer that knows the past
//! symbols (a decision feedback without error).
//!
//! The Euclidean metric is the log-likelihood of Gaussian noise. After a square-law
//! photodiode with optical amplifier noise the sample is a noncentral chi-square
//! variable, whose log-likelihood Foggi, Forestieri, Colavolpe and Prati (2006) write in
//! closed form (their eq. (12), the modified Bessel function approximated by its
//! asymptote): for a noiseless sample $s > 0$, a received sample $z \ge 0$, the noise
//! variance $N_{0}$ of every complex degree of freedom and $\nu$ such degrees of freedom
//! (the ratio $2B_{o}/B_{e}$ of the noise bandwidths of the optical and electrical
//! filters),
//!
//! $$
//! \lambda(z, s) = 2\sqrt{z s} - s - \frac{N_{0}}{2}\left(\nu - \frac{1}{2}\right)\ln s,
//! $$
//!
//! [`Mlse::chi_square_metric`], which [`Mlse::detect_with`] takes in place of the
//! Euclidean one.
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
//! 3. J. H. Winters, R. D. Gitlin, "Electrical signal processing techniques in
//!    long-haul, fiber-optic systems," *Proc. IEEE ICC*, 397–403, 1990.
//! 4. T. Foggi, E. Forestieri, G. Colavolpe, G. Prati, "Maximum-likelihood sequence
//!    detection with closed-form metrics in OOK optical systems impaired by GVD and
//!    PMD," *J. Lightw. Technol.* 24(8), 3073–3087, 2006.

use comnum_core::{Result, ensure, ensure_shape, same_len};
use comnum_math::C64;
use comnum_math::viterbi::{self as vit, StateMachine};

/// The maximum-likelihood sequence detector of a known channel (see the module).
#[derive(Clone, Debug, PartialEq)]
pub struct Mlse {
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
        Ok(Mlse { machine, expected })
    }

    /// The detector of a channel given by its table `expected` of noiseless samples,
    /// one per branch `s * M + u` of the shift register of `memory` symbols over `m`
    /// points ([`comnum_math::viterbi::StateMachine::shift_register`]: `s` holds the
    /// previous symbols, the most recent in the least significant digit, `u` the current
    /// one).
    ///
    /// # Errors
    /// [`comnum_core::Error::Shape`] if the table has not $M^{L}$ entries;
    /// [`comnum_core::Error::Config`] as [`Mlse::new`].
    pub fn from_table(expected: Vec<C64>, m: usize, memory: usize) -> Result<Mlse> {
        let machine = StateMachine::shift_register(m, memory)?;
        same_len(expected.len(), machine.n_states() * m, "the table and the branches")?;
        Ok(Mlse { machine, expected })
    }

    /// The detector fitted on the samples `r` of the known symbols `indices` (indices of
    /// `m` points): the table of [`Mlse::from_table`] is the mean of the samples of every
    /// pattern of `memory + 1` symbols (the least-squares table of a channel that need not
    /// be linear); the first `memory` samples, whose past is unknown, are skipped.
    ///
    /// # Errors
    /// [`comnum_core::Error::Shape`] for lengths that differ or an index at or above `m`;
    /// [`comnum_core::Error::Config`] for a pattern that never occurs, or as [`Mlse::new`].
    pub fn fit(r: &[C64], indices: &[usize], m: usize, memory: usize) -> Result<Mlse> {
        same_len(r.len(), indices.len(), "samples and symbols")?;
        ensure_shape(indices.iter().all(|&i| i < m), || format!("a symbol index is not below {m}"))?;
        let machine = StateMachine::shift_register(m, memory)?;
        let n_states = machine.n_states();
        let mut sum = vec![C64::new(0.0, 0.0); n_states * m];
        let mut count = vec![0usize; n_states * m];
        let mut state = 0;
        for (k, (&x, &u)) in r.iter().zip(indices).enumerate() {
            if k >= memory {
                sum[state * m + u] += x;
                count[state * m + u] += 1;
            }
            state = machine.next(state, u);
        }
        ensure(count.iter().all(|&c| c > 0), || {
            format!("every pattern of {} symbols must occur in the training", memory + 1)
        })?;
        let expected = sum.iter().zip(&count).map(|(s, &c)| s / c as f64).collect();
        Ok(Mlse { machine, expected })
    }

    /// The noiseless sample of every branch `s * M + u` (the table of [`Mlse::from_table`]).
    pub fn expected(&self) -> &[C64] {
        &self.expected
    }

    /// The minimum Euclidean distance between the sample vectors of two paths of
    /// `window` steps that leave the same state with different symbols, whatever follows
    /// (see the module): the worst-case eye of a detector that decides a symbol from the
    /// next `window` samples, over every state.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for a window of 0 or of more than $2^{20}$ paths.
    pub fn min_distance(&self, window: usize) -> Result<f64> {
        let m = self.machine.n_inputs();
        let n_paths = m.checked_pow(window as u32).filter(|&n| window >= 1 && n <= 1 << 20);
        let n_paths =
            n_paths.ok_or_else(|| comnum_core::Error::Config(format!("a window of {window} steps is not usable")))?;
        let mut best = f64::INFINITY;
        let mut vectors = vec![C64::new(0.0, 0.0); window];
        let mut paths: Vec<(usize, Vec<C64>)> = Vec::with_capacity(n_paths);
        for s0 in 0..self.machine.n_states() {
            paths.clear();
            for p in 0..n_paths {
                let (mut s, mut code) = (s0, p);
                for v in &mut vectors {
                    let u = code % m;
                    code /= m;
                    *v = self.expected[s * m + u];
                    s = self.machine.next(s, u);
                }
                paths.push((p % m, vectors.clone()));
            }
            for (i, (u, a)) in paths.iter().enumerate() {
                for (u2, b) in &paths[i + 1..] {
                    if u != u2 {
                        let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y).norm_sqr()).sum();
                        best = best.min(d);
                    }
                }
            }
        }
        Ok(best.sqrt())
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

    /// The sequence of largest accumulated branch metric `metric(sample, expected)`
    /// over the samples `r`, for a noise that is not Gaussian (see the module); with
    /// the negative squared distance it is [`Mlse::detect`].
    ///
    /// # Errors
    /// None for a non-empty input.
    pub fn detect_with(&self, r: &[C64], metric: impl Fn(C64, C64) -> f64) -> Result<Vec<usize>> {
        let start = vec![0.0; self.machine.n_states()];
        let gammas = |k: usize, g: &mut [f64]| {
            for (gv, e) in g.iter_mut().zip(&self.expected) {
                *gv = metric(r[k], *e);
            }
        };
        Ok(vit::viterbi(&self.machine, r.len(), &start, gammas, None)?.0)
    }

    /// The branch metric of a noncentral chi-square sample (see the module; Foggi et
    /// al. 2006, eq. (12)) on the real parts, for the noise variance `noise_var` $N_{0}$
    /// of every complex degree of freedom and `dof` $\nu$ such degrees: a negative sample
    /// counts as zero, and every expected sample must be positive (a finite extinction
    /// ratio).
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for a non-positive noise variance or number of
    /// degrees of freedom, or an expected sample that is not positive.
    pub fn chi_square_metric(&self, noise_var: f64, dof: f64) -> Result<impl Fn(C64, C64) -> f64 + use<>> {
        ensure(noise_var > 0.0 && dof > 0.0, || {
            "the noise variance and the degrees of freedom must be positive".into()
        })?;
        ensure(self.expected.iter().all(|e| e.re > 0.0), || {
            "the chi-square metric needs positive expected samples (a finite extinction ratio)".into()
        })?;
        let c = noise_var / 2.0 * (dof - 0.5);
        Ok(move |z: C64, s: C64| 2.0 * (z.re.max(0.0) * s.re).sqrt() - s.re - c * s.re.ln())
    }

    /// The sequence of [`Mlse::detect`] and the reliability of every symbol, the
    /// log-likelihood difference $\Delta_{k}/\sigma^{2}$ with its best competitor (see the
    /// module), for the variance `noise_var` $\sigma^{2} = E|n|^{2}$ of the complex noise.
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] for a non-positive noise variance.
    pub fn detect_soft(&self, r: &[C64], noise_var: f64) -> Result<(Vec<usize>, Vec<f64>)> {
        ensure(noise_var > 0.0, || format!("the noise variance {noise_var} must be positive"))?;
        let start = vec![0.0; self.machine.n_states()];
        let (path, delta) = vit::sova(&self.machine, r.len(), &start, self.gammas(r), None)?;
        Ok((path, delta.iter().map(|d| d / noise_var).collect()))
    }

    /// The log-likelihood ratios $\ln[P(s_{k} = a_{0}) / P(s_{k} = a_{1})]$ of a binary
    /// constellation from [`Mlse::detect_soft`] ([`comnum_math::viterbi::binary_llr`]).
    ///
    /// # Errors
    /// [`comnum_core::Error::Config`] if the constellation has not two points, or as
    /// [`Mlse::detect_soft`].
    pub fn llr(&self, r: &[C64], noise_var: f64) -> Result<Vec<f64>> {
        let m = self.machine.n_inputs();
        ensure(m == 2, || format!("LLRs need two points, not {m}"))?;
        let (path, rel) = self.detect_soft(r, noise_var)?;
        Ok(vit::binary_llr(&path, &rel))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equalizer::pam;
    use comnum_core::rng::Rng;

    #[test]
    fn the_chi_square_metric_decides_as_the_euclidean_one_at_high_snr_and_checks_its_inputs() {
        // on-off keying through 1 + 0.3 z⁻¹ on powers (a square-law channel of positive
        // samples), weak noise: both metrics recover the sequence; the chi-square one
        // refuses a zero expected sample
        let pts = [C64::new(0.1, 0.0), C64::new(1.0, 0.0)];
        let taps = [C64::new(1.0, 0.0), C64::new(0.3, 0.0)];
        let mlse = Mlse::new(&taps, &pts).unwrap();
        let mut rng = Rng::new(5);
        let idx: Vec<usize> = (0..300).map(|_| usize::from(rng.bit())).collect();
        let r: Vec<C64> = (0..idx.len())
            .map(|k| {
                let past = if k > 0 { taps[1] * pts[idx[k - 1]] } else { C64::new(0.0, 0.0) };
                taps[0] * pts[idx[k]] + past + C64::new(0.02 * rng.gaussian(), 0.0)
            })
            .collect();
        let metric = mlse.chi_square_metric(0.01, 2.0).unwrap();
        assert_eq!(mlse.detect_with(&r, metric).unwrap(), idx);
        assert_eq!(mlse.detect_with(&r, |z, s| -(z - s).norm_sqr()).unwrap(), mlse.detect(&r).unwrap());
        assert!(mlse.chi_square_metric(0.0, 2.0).is_err());
        let zero = Mlse::new(&taps, &[C64::new(0.0, 0.0), C64::new(1.0, 0.0)]).unwrap();
        assert!(zero.chi_square_metric(0.01, 2.0).is_err());
    }

    #[test]
    fn the_fitted_table_is_the_linear_channel_and_the_distance_its_eye() {
        // 2-PAM through 1 + 0.5 z⁻¹: the table fitted on every pattern is the one of `new`,
        // the window-1 distance is the level gap of a known past, 2 |h0| = 2, and a long
        // window reaches Forney's minimum distance of the single error event, 2 sqrt(1.25)
        let pts = pam(2);
        let taps = [C64::new(1.0, 0.0), C64::new(0.5, 0.0)];
        let idx: Vec<usize> = [0, 0, 1, 1, 0, 1, 0, 0, 1, 1].to_vec();
        let r: Vec<C64> = (0..idx.len())
            .map(|k| taps[0] * pts[idx[k]] + if k > 0 { taps[1] * pts[idx[k - 1]] } else { C64::new(0.0, 0.0) })
            .collect();
        let fitted = Mlse::fit(&r, &idx, 2, 1).unwrap();
        let exact = Mlse::new(&taps, &pts).unwrap();
        assert!(fitted.expected().iter().zip(exact.expected()).all(|(a, b)| (a - b).norm() < 1e-12));
        assert!((exact.min_distance(1).unwrap() - 2.0).abs() < 1e-12);
        assert!((exact.min_distance(6).unwrap() - 2.0 * 1.25f64.sqrt()).abs() < 1e-12);
        assert_eq!(Mlse::from_table(exact.expected().to_vec(), 2, 1).unwrap(), exact);
        assert!(Mlse::from_table(vec![C64::new(0.0, 0.0); 3], 2, 1).is_err());
        assert!(Mlse::fit(&r[..3], &idx[..3], 2, 1).is_err());
        assert!(exact.min_distance(0).is_err());
    }
    use comnum_math::qfunc;

    /// `s` through the causal channel `h`, the tail of the convolution dropped.
    fn channel(s: &[C64], h: &[C64]) -> Vec<C64> {
        crate::filter::convolve_full(h, s)[..s.len()].to_vec()
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
        // real noise of variance 0.09: the complex variance E|n|² passed is 0.18
        let (path, rel) = mlse.detect_soft(&r, 0.18).unwrap();
        assert_eq!(path, hard);
        assert!(rel.iter().all(|v| *v >= 0.0));
        let llr = mlse.llr(&r, 0.18).unwrap();
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
        let (_, half) = mlse.detect_soft(&r, 0.36).unwrap();
        assert!(rel.iter().zip(&half).all(|(a, b)| (a - 2.0 * b).abs() < 1e-9 * a.max(1.0)));
        assert!(mlse.detect_soft(&r, 0.0).is_err());
        assert!(Mlse::new(&h, &pam(4)).unwrap().llr(&r, 1.0).is_err());
    }
}
