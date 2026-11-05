//! Density evolution of regular LDPC ensembles under message passing (Richardson &
//! Urbanke 2001): the expected fraction of incorrect messages on a cycle-free Tanner
//! graph, iteration after iteration, and the threshold of the channel parameter below
//! which it tends to zero.
//!
//! For the $(d_{v}, d_{c})$-regular ensemble, an output-symmetric binary-input channel
//! and a symmetric decoder, the all-one codeword may be assumed (Lemma 1 of the paper)
//! and the densities of the messages evolve deterministically:
//!
//! - **hard decisions on the BSC** ([`hard_decision_error`], [`hard_decision_threshold`]):
//!   Gallager's algorithm A, eq. (6),
//!   $$p^{(\ell)} = \epsilon - \epsilon \left[\frac{1 + (1 - 2p^{(\ell-1)})^{d_{c}-1}}{2}\right]^{d_{v}-1} + (1 - \epsilon) \left[\frac{1 - (1 - 2p^{(\ell-1)})^{d_{c}-1}}{2}\right]^{d_{v}-1},$$
//!   algorithm B (Example 4: the received bit is flipped when at least $b_{\ell}$ of the
//!   $d_{v} - 1$ incoming check messages disagree, $b_{\ell}$ chosen by Gallager's rule),
//!   and algorithm E (Example 5: ternary messages $\{-1, 0, 1\}$, the received bit weighted
//!   by $w^{(\ell)}$ chosen to maximise the capacity (7) of the induced binary symmetric
//!   erasure channel);
//! - **belief propagation** ([`BpDensityEvolution`]): the densities of the
//!   log-likelihood ratios on a uniform grid; at a variable node the channel density
//!   convolved with $d_{v} - 1$ check densities (eq. (9), by FFT), at a check node the
//!   $d_{c} - 1$ variable densities combined by $m = 2 \operatorname{atanh}\left(\tanh\frac{a}{2} \tanh\frac{b}{2}\right)$
//!   pair by pair on the grid (the discretized density evolution of Chung, Forney,
//!   Richardson & Urbanke, *IEEE Commun. Lett.* 5(2), 2001; the paper computes the same
//!   density with the transform of eq. (8)). An LLR of zero counts as half an error.
//!
//! The threshold is found by bisection ([`comnum_math::bisect`]) of the predicate "the
//! fraction of incorrect messages falls below the target". With a finite grid and a
//! finite number of iterations the computed threshold is an estimate; Richardson &
//! Urbanke give theirs rounded down (Tables I and II).
//!
//! # Examples
//! ```
//! use comnum_fec::density_evolution::{hard_decision_threshold, HardDecoder, RegularEnsemble};
//! // Gallager A on the (3, 6)-regular ensemble: 0.0395 (Table I prints 0.04)
//! let e = RegularEnsemble::new(3, 6)?;
//! let t = hard_decision_threshold(&e, HardDecoder::GallagerA)?;
//! assert!((t - 0.0395).abs() < 5e-4, "{t}");
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//! 1. T. J. Richardson, R. L. Urbanke, "The capacity of low-density parity-check codes
//!    under message-passing decoding," *IEEE Trans. Inf. Theory* 47(2), 599–618, 2001:
//!    eq. (6), Examples 3–5, 10–11, Tables I–II.
//! 2. S.-Y. Chung, G. D. Forney, T. J. Richardson, R. Urbanke, "On the design of
//!    low-density parity-check codes within 0.0045 dB of the Shannon limit," *IEEE Commun.
//!    Lett.* 5(2), 58–60, 2001: the discretized density evolution of the check node.

use comnum_core::{Error, Result};
use comnum_math::fft::{fft, ifft};
use comnum_math::{C64, bisect, entropy, qfunc};

/// The $(d_{v}, d_{c})$-regular LDPC ensemble: every variable node of degree $d_{v}$,
/// every check node of degree $d_{c}$, design rate $1 - d_{v}/d_{c}$.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegularEnsemble {
    /// Variable-node degree $d_{v}$.
    pub dv: usize,
    /// Check-node degree $d_{c}$.
    pub dc: usize,
}

impl RegularEnsemble {
    /// The ensemble of degrees `dv` and `dc`.
    ///
    /// # Errors
    /// [`Error::Config`] unless $2 \le d_{v} < d_{c}$.
    pub fn new(dv: usize, dc: usize) -> Result<RegularEnsemble> {
        if dv < 2 || dc <= dv {
            return Err(Error::Config(format!("a regular ensemble needs 2 <= dv < dc, got ({dv}, {dc})")));
        }
        Ok(RegularEnsemble { dv, dc })
    }

    /// The design rate $1 - d_{v}/d_{c}$.
    pub fn rate(&self) -> f64 {
        1.0 - self.dv as f64 / self.dc as f64
    }
}

/// A hard-decision message-passing decoder on the binary symmetric channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HardDecoder {
    /// Gallager's algorithm A (Example 3, eq. (6)).
    GallagerA,
    /// Gallager's algorithm B (Example 4).
    GallagerB,
    /// The ternary decoder with erasures in the messages (Example 5).
    AlgorithmE,
}

/// The binomial coefficient as a float.
fn binomial(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |c, i| c * (n - i) as f64 / (i + 1) as f64)
}

/// The fraction of incorrect variable-to-check messages after `iterations` iterations of
/// `decoder` on the BSC of crossover probability `eps` (an erasure counts as half an
/// error), on a cycle-free graph of the ensemble.
///
/// # Errors
/// [`Error::Config`] unless $0 \le \epsilon \le 1/2$.
pub fn hard_decision_error(
    ensemble: &RegularEnsemble,
    decoder: HardDecoder,
    eps: f64,
    iterations: usize,
) -> Result<f64> {
    if !(0.0..=0.5).contains(&eps) {
        return Err(Error::Config(format!("the crossover probability must be in [0, 1/2], got {eps}")));
    }
    let (dv, dc) = (ensemble.dv, ensemble.dc);
    match decoder {
        HardDecoder::GallagerA | HardDecoder::GallagerB => {
            let mut p = eps;
            for _ in 0..iterations {
                let x = (1.0 - 2.0 * p).powi(dc as i32 - 1);
                let (agree, disagree) = ((1.0 + x) / 2.0, (1.0 - x) / 2.0);
                // b: the number of disagreeing checks that flips the received bit; A: all
                // of them; B: the smallest b of Gallager's rule (Example 4)
                let b = match decoder {
                    HardDecoder::GallagerA => dv - 1,
                    _ => (1..dv)
                        .find(|&b| {
                            let e = 2 * b as i32 - dv as i32 + 1;
                            e > 0 && (1.0 - eps) / eps <= (agree / disagree).powi(e)
                        })
                        .unwrap_or(dv - 1),
                };
                let tail = |u: f64, v: f64| {
                    (b..dv).map(|k| binomial(dv - 1, k) * u.powi(k as i32) * v.powi((dv - 1 - k) as i32)).sum::<f64>()
                };
                p = eps - eps * tail(agree, disagree) + (1.0 - eps) * tail(disagree, agree);
                if p < 1e-300 {
                    return Ok(0.0);
                }
            }
            Ok(p)
        }
        HardDecoder::AlgorithmE => Ok(algorithm_e(dv, dc, eps, iterations)),
    }
}

/// Algorithm E (Example 5): the probabilities $(p_{1}, p_{-1}, p_{0})$ of a correct, an
/// incorrect and an erased variable-to-check message, iterated; returns $p_{-1} + p_{0}/2$.
fn algorithm_e(dv: usize, dc: usize, eps: f64, iterations: usize) -> f64 {
    let n = dv - 1;
    let (mut p1, mut pm, mut p0): (f64, f64, f64) = (1.0 - eps, eps, 0.0);
    for _ in 0..iterations {
        // check node
        let (s, d) = ((p1 + pm).powi(dc as i32 - 1), (p1 - pm).powi(dc as i32 - 1));
        // q0 = 1 - (1 - p0)^(dc - 1) taken as the complement, so that the three sum to one:
        // a total off by rounding grows by (dc - 1)(dv - 1) per iteration
        let _ = p0;
        let (q1, qm) = (((s + d) / 2.0).max(0.0), ((s - d) / 2.0).max(0.0));
        let q0 = (1.0 - q1 - qm).max(0.0);
        // variable node: for every weight, the trinomial distribution of i - j over the
        // n incoming messages (i correct, j incorrect), then the sign of w m0 + i - j
        let mut best: Option<(f64, f64, f64, f64)> = None;
        for w in 1..=dv as i64 {
            let (mut n1, mut nm, mut n0) = (0.0, 0.0, 0.0);
            for i in 0..=n {
                for j in 0..=n - i {
                    let pr = binomial(n, i)
                        * binomial(n - i, j)
                        * q1.powi(i as i32)
                        * qm.powi(j as i32)
                        * q0.powi((n - i - j) as i32);
                    let diff = i as i64 - j as i64;
                    for (m0, pm0) in [(1i64, 1.0 - eps), (-1, eps)] {
                        let v = w * m0 + diff;
                        let mass = pr * pm0;
                        match v.signum() {
                            1 => n1 += mass,
                            -1 => nm += mass,
                            _ => n0 += mass,
                        }
                    }
                }
            }
            // eq. (7): 1 - p0 + h(p0) - h(p0, p_-1), the capacity of the binary symmetric
            // erasure channel of these probabilities
            let (n1, nm, n0): (f64, f64, f64) = (n1.max(0.0), nm.max(0.0), n0.max(0.0));
            let cap = 1.0 - n0 + entropy(&[n0, (1.0 - n0).max(0.0)]) - entropy(&[n0, nm, n1]);
            if best.is_none_or(|b| cap > b.0) {
                best = Some((cap, n1, nm, n0));
            }
        }
        let Some((_, a, b, c)) = best else { break };
        let total = a + b + c;
        (p1, pm, p0) = (a / total, b / total, c / total);
        if pm + p0 / 2.0 < 1e-300 {
            return 0.0;
        }
    }
    pm + p0 / 2.0
}

/// The largest crossover probability for which `decoder` drives the fraction of
/// incorrect messages to zero on the ensemble (Table I of Richardson & Urbanke): a
/// bisection of $[0, 1/2]$ to $10^{-6}$, each point iterated until the fraction falls
/// below $10^{-12}$ (at most $10^{5}$ iterations).
///
/// # Errors
/// As [`hard_decision_error`] (never for this range).
pub fn hard_decision_threshold(ensemble: &RegularEnsemble, decoder: HardDecoder) -> Result<f64> {
    let converges = |eps: f64| hard_decision_error(ensemble, decoder, eps, 100_000).is_ok_and(|p| p < 1e-12);
    Ok(bisect(converges, 0.0, 0.5, 20))
}

/// A binary-input memoryless output-symmetric channel, the input of
/// [`BpDensityEvolution`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BinaryInputChannel {
    /// The binary symmetric channel of crossover probability `crossover` (Example 1).
    Bsc {
        /// Crossover probability $\epsilon$.
        crossover: f64,
    },
    /// The binary-input AWGN channel $y = x + z$, $x \in \{\pm 1\}$,
    /// $z \sim \mathcal{N}(0, \sigma^{2})$ (Example 2).
    Biawgn {
        /// Noise standard deviation $\sigma$.
        sigma: f64,
    },
}

/// The outcome of a run of density evolution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensityEvolutionRun {
    /// Fraction of incorrect variable-to-check messages at the last iteration (an LLR of
    /// zero counts as half).
    pub error: f64,
    /// Iterations run.
    pub iterations: usize,
    /// Whether the fraction fell below the target.
    pub converged: bool,
}

/// Discretized density evolution of belief propagation (see the module).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BpDensityEvolution {
    /// Step of the LLR grid (nominal: the step is adjusted so that the LLR of the BSC
    /// falls on the grid).
    pub llr_step: f64,
    /// Largest LLR of the grid; larger magnitudes are clipped to it.
    pub llr_max: f64,
    /// Largest number of iterations.
    pub max_iterations: usize,
    /// The fraction of incorrect messages below which the run has converged.
    pub target: f64,
}

impl Default for BpDensityEvolution {
    /// A step of 0.05, LLRs up to 25, 3000 iterations, a target of $10^{-7}$.
    fn default() -> Self {
        BpDensityEvolution { llr_step: 0.05, llr_max: 25.0, max_iterations: 3000, target: 1e-7 }
    }
}

/// The grid of the LLRs and the table of the check-node operation on it.
struct Grid {
    /// Half-width $K$: the LLRs are $k \Delta$, $-K \le k \le K$.
    half: usize,
    /// For every pair of bins, the bin of $2 \operatorname{atanh}(\tanh(a/2) \tanh(b/2))$.
    table: Vec<u32>,
}

impl Grid {
    fn new(step: f64, half: usize) -> Grid {
        let n = 2 * half + 1;
        let th: Vec<f64> = (0..n).map(|i| ((i as f64 - half as f64) * step / 2.0).tanh()).collect();
        let mut table = vec![0u32; n * n];
        for i in 0..n {
            for j in 0..n {
                let t = (th[i] * th[j]).clamp(-1.0, 1.0);
                let m = 2.0 * t.atanh();
                let k = if m.is_finite() { (m / step).round() } else { m.signum() * half as f64 };
                table[i * n + j] = (k.clamp(-(half as f64), half as f64) + half as f64) as u32;
            }
        }
        Grid { half, table }
    }

    fn len(&self) -> usize {
        2 * self.half + 1
    }

    /// The density of the check-node combination of two independent messages.
    fn boxplus(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        let n = self.len();
        let mut out = vec![0.0; n];
        for (i, &pa) in a.iter().enumerate() {
            if pa == 0.0 {
                continue;
            }
            let row = &self.table[i * n..(i + 1) * n];
            for (&k, &pb) in row.iter().zip(b) {
                out[k as usize] += pa * pb;
            }
        }
        out
    }

    /// The density of the sum of independent messages of densities `parts` (indices add),
    /// clipped to the grid.
    fn convolve(&self, parts: &[&[f64]]) -> Vec<f64> {
        let n = self.len();
        let full = parts.len() * (n - 1) + 1;
        let size = full.next_power_of_two();
        let mut acc = vec![C64::new(1.0, 0.0); size];
        for p in parts {
            let mut x = vec![C64::new(0.0, 0.0); size];
            for (xi, &pi) in x.iter_mut().zip(p.iter()) {
                *xi = C64::new(pi, 0.0);
            }
            for (a, f) in acc.iter_mut().zip(fft(&x)) {
                *a *= f;
            }
        }
        let y = ifft(&acc);
        // index t of the full convolution is the LLR (t - parts.len() * half) step
        let offset = parts.len() * self.half;
        let mut out = vec![0.0; n];
        for (t, v) in y.iter().take(full).enumerate() {
            let k = (t as i64 - offset as i64).clamp(-(self.half as i64), self.half as i64);
            out[(k + self.half as i64) as usize] += v.re.max(0.0);
        }
        let total: f64 = out.iter().sum();
        for v in &mut out {
            *v /= total;
        }
        out
    }

    /// Fraction of incorrect messages: the negative LLRs and half the zero bin.
    fn error(&self, p: &[f64]) -> f64 {
        p[..self.half].iter().sum::<f64>() + 0.5 * p[self.half]
    }
}

impl BpDensityEvolution {
    /// The grid step and the density of the channel LLRs on it.
    fn channel(&self, channel: BinaryInputChannel) -> Result<(f64, usize, Vec<f64>)> {
        if !(self.llr_step > 0.0 && self.llr_max > self.llr_step) {
            return Err(Error::Config(format!(
                "density evolution needs 0 < llr_step < llr_max, got {} and {}",
                self.llr_step, self.llr_max
            )));
        }
        match channel {
            BinaryInputChannel::Bsc { crossover: eps } => {
                if !(eps > 0.0 && eps < 0.5) {
                    return Err(Error::Config(format!("the crossover probability must be in (0, 1/2), got {eps}")));
                }
                // the step that puts the LLR log((1 - eps)/eps) on a bin (Example 10)
                let l0 = ((1.0 - eps) / eps).ln();
                let bins = (l0 / self.llr_step).round().max(1.0);
                let step = l0 / bins;
                let half = (self.llr_max / step).round() as usize;
                let half = half.max(bins as usize);
                let mut p = vec![0.0; 2 * half + 1];
                p[half + bins as usize] = 1.0 - eps;
                p[half - bins as usize] = eps;
                Ok((step, half, p))
            }
            BinaryInputChannel::Biawgn { sigma } => {
                if !(sigma > 0.0 && sigma.is_finite()) {
                    return Err(Error::Config(format!("the noise standard deviation must be positive, got {sigma}")));
                }
                // LLR 2y/sigma^2 ~ N(2/sigma^2, 4/sigma^2) for the input +1 (Example 11)
                let (mu, sd) = (2.0 / (sigma * sigma), 2.0 / sigma);
                let step = self.llr_step;
                let half = (self.llr_max / step).round() as usize;
                let cdf = |x: f64| 1.0 - qfunc((x - mu) / sd);
                let p = (0..=2 * half)
                    .map(|i| {
                        let k = i as f64 - half as f64;
                        let lo = if i == 0 { f64::NEG_INFINITY } else { (k - 0.5) * step };
                        let hi = if i == 2 * half { f64::INFINITY } else { (k + 0.5) * step };
                        let c = |x: f64| if x.is_infinite() { if x > 0.0 { 1.0 } else { 0.0 } } else { cdf(x) };
                        (c(hi) - c(lo)).max(0.0)
                    })
                    .collect();
                Ok((step, half, p))
            }
        }
    }

    /// Run density evolution of belief propagation on the ensemble and the channel until
    /// the fraction of incorrect messages falls below [`target`](Self::target), stops
    /// decreasing, or [`max_iterations`](Self::max_iterations) is reached.
    ///
    /// # Errors
    /// [`Error::Config`] for an invalid grid or channel parameter.
    pub fn run(&self, ensemble: &RegularEnsemble, channel: BinaryInputChannel) -> Result<DensityEvolutionRun> {
        let (step, half, p0) = self.channel(channel)?;
        let grid = Grid::new(step, half);
        let mut p = p0.clone();
        let mut last = grid.error(&p);
        for it in 1..=self.max_iterations {
            // check node: d_c - 1 variable messages
            let mut q = p.clone();
            for _ in 1..ensemble.dc - 1 {
                q = grid.boxplus(&q, &p);
            }
            // variable node: the channel and d_v - 1 check messages, eq. (9)
            let mut parts: Vec<&[f64]> = vec![&p0];
            parts.extend(std::iter::repeat_n(q.as_slice(), ensemble.dv - 1));
            p = grid.convolve(&parts);
            let e = grid.error(&p);
            if e < self.target {
                return Ok(DensityEvolutionRun { error: e, iterations: it, converged: true });
            }
            // a fixed point: the fraction no longer decreases
            if e >= last - 1e-15 * last.max(1e-300) && it > 1 {
                return Ok(DensityEvolutionRun { error: e, iterations: it, converged: false });
            }
            last = e;
        }
        Ok(DensityEvolutionRun { error: last, iterations: self.max_iterations, converged: false })
    }

    /// The threshold of a family of channels `channel(t)` that degrades as `t` grows,
    /// between `lo` (converges) and `hi` (does not), by `steps` bisection steps.
    ///
    /// # Errors
    /// As [`run`](Self::run) for a parameter of the bracket.
    pub fn threshold(
        &self,
        ensemble: &RegularEnsemble,
        channel: impl Fn(f64) -> BinaryInputChannel,
        lo: f64,
        hi: f64,
        steps: usize,
    ) -> Result<f64> {
        let mut err = None;
        let t = bisect(
            |x| match self.run(ensemble, channel(x)) {
                Ok(r) => r.converged,
                Err(e) => {
                    err = Some(e);
                    false
                }
            },
            lo,
            hi,
            steps,
        );
        match err {
            Some(e) => Err(e),
            None => Ok(t),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_ensemble_needs_two_le_dv_lt_dc() {
        assert!(RegularEnsemble::new(1, 6).is_err() && RegularEnsemble::new(6, 6).is_err());
        assert!((RegularEnsemble::new(3, 6).unwrap().rate() - 0.5).abs() < 1e-15);
        let e = RegularEnsemble::new(3, 6).unwrap();
        assert!(hard_decision_error(&e, HardDecoder::GallagerA, 0.6, 10).is_err());
        let bp = BpDensityEvolution::default();
        assert!(bp.run(&e, BinaryInputChannel::Bsc { crossover: 0.0 }).is_err());
        assert!(bp.run(&e, BinaryInputChannel::Biawgn { sigma: -1.0 }).is_err());
    }

    /// For $d_{v} = 3$ algorithms A and B coincide (Table I caption); the noiseless channel
    /// has no error; above the threshold the fraction stays at a fixed point.
    #[test]
    fn gallager_a_and_b_coincide_for_dv_3_and_have_fixed_points_above_threshold() {
        let e = RegularEnsemble::new(3, 6).unwrap();
        for eps in [0.03, 0.05] {
            let a = hard_decision_error(&e, HardDecoder::GallagerA, eps, 2000).unwrap();
            let b = hard_decision_error(&e, HardDecoder::GallagerB, eps, 2000).unwrap();
            assert_eq!(a, b);
        }
        assert_eq!(hard_decision_error(&e, HardDecoder::GallagerA, 0.0, 10).unwrap(), 0.0);
        assert!(hard_decision_error(&e, HardDecoder::GallagerA, 0.05, 2000).unwrap() > 0.01);
    }

    /// The check and variable maps preserve the total probability and the symmetry
    /// $P(-m) = e^{-m} P(m)$ of LLR densities, approximately on the grid.
    #[test]
    fn the_grid_operations_keep_the_mass_and_the_density_symmetry() {
        let bp = BpDensityEvolution::default();
        let (step, half, p0) = bp.channel(BinaryInputChannel::Biawgn { sigma: 0.9 }).unwrap();
        let g = Grid::new(step, half);
        let q = g.boxplus(&p0, &p0);
        let v = g.convolve(&[&p0, &q, &q]);
        for d in [&p0, &q, &v] {
            assert!((d.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        // symmetry at a few bins of the check output
        for k in [5usize, 20, 40] {
            let (pos, neg) = (q[half + k], q[half - k]);
            let m = k as f64 * step;
            assert!((neg - (-m).exp() * pos).abs() < 0.05 * pos, "bin {k}: {neg} against {}", (-m).exp() * pos);
        }
    }

    /// On a cycle-free graph a channel far better than the threshold converges in a few
    /// iterations, a channel far worse never does.
    #[test]
    fn bp_density_evolution_converges_well_below_the_threshold_only() {
        let e = RegularEnsemble::new(3, 6).unwrap();
        let bp = BpDensityEvolution { llr_step: 0.1, ..BpDensityEvolution::default() };
        let good = bp.run(&e, BinaryInputChannel::Biawgn { sigma: 0.7 }).unwrap();
        assert!(good.converged && good.iterations < 30, "{good:?}");
        let bad = bp.run(&e, BinaryInputChannel::Biawgn { sigma: 1.0 }).unwrap();
        assert!(!bad.converged && bad.error > 0.05, "{bad:?}");
    }

    /// Gallager's algorithm B is A with the flipping rule of Example 4: it only helps for
    /// $d_{v} > 3$; algorithm E with the weights that maximise (7) beats both (Table I).
    #[test]
    fn algorithm_b_and_e_improve_on_a_as_in_table_i() {
        let e48 = RegularEnsemble::new(4, 8).unwrap();
        let a = hard_decision_threshold(&e48, HardDecoder::GallagerA).unwrap();
        let b = hard_decision_threshold(&e48, HardDecoder::GallagerB).unwrap();
        let ee = hard_decision_threshold(&e48, HardDecoder::AlgorithmE).unwrap();
        assert!((a - 0.0476).abs() < 2e-4 && (b - 0.0517).abs() < 2e-4 && (ee - 0.0595).abs() < 2e-4, "{a} {b} {ee}");
        // (3, 6): Example 5 finds the weights 2 then 1 and a threshold near 0.07
        let e36 = RegularEnsemble::new(3, 6).unwrap();
        let ee = hard_decision_threshold(&e36, HardDecoder::AlgorithmE).unwrap();
        assert!((0.070..0.0715).contains(&ee), "{ee}");
        // below capacity: eps_opt = 0.11 at rate 1/2
        assert!(ee < 0.11);
    }
}
