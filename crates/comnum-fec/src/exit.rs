//! Extrinsic information transfer (EXIT) charts of convolutional constituent codes (ten
//! Brink 2001): the mutual information between the bits and their LLRs, the transfer
//! curve $I_{E} = T(I_{A})$ of a BCJR decoder fed with Gaussian a priori LLRs, and the
//! pinch-off $E_{b}/N_{0}$ of a parallel concatenation, below which the decoding tunnel
//! between the two curves closes.
//!
//! The a priori LLRs are modelled as $A = \mu_{A} x + n_{A}$ with $x = \pm 1$,
//! $n_{A} \sim \mathcal{N}(0, \sigma_{A}^{2})$ and $\mu_{A} = \sigma_{A}^{2}/2$ (eqs. (9)–(11));
//! their mutual information with the bits is $J(\sigma_{A})$ (eq. (14)),
//! $$J(\sigma) = 1 - \int \frac{e^{-(\xi - \sigma^{2}/2)^{2}/(2\sigma^{2})}}{\sqrt{2\pi\sigma^{2}}}\log_{2}(1 + e^{-\xi})\,d\xi$$
//! ([`j_function`], Gauss–Hermite with 64 nodes; [`j_inverse`] by bisection). The
//! extrinsic LLRs of the decoder are $E = D - A - Z$ (eq. (19)), $D$ the a posteriori,
//! $Z$ the channel LLRs of the systematic bits; their mutual information is estimated by
//! the time average $1 - \frac1N\sum_{k}\log_{2}(1 + e^{-x_{k}E_{k}})$
//! ([`llr_mutual_information`]), exact in the mean for consistent LLRs (the paper uses
//! histograms of $E$). [`ExitCurve::measure`] simulates a recursive systematic code on the
//! BPSK AWGN channel with its parity punctured by a period, and [`pinch_off_db`]
//! bisects the $E_{b}/N_{0}$ at which $T_{2}(T_{1}(I)) > I$ stops holding on
//! $[0, I_{\max}]$, the curves measured on common random numbers.
//!
//! # Examples
//! ```
//! use comnum_fec::exit::{j_function, j_inverse};
//! // J(0) = 0, J grows to 1; J and its inverse agree
//! assert!(j_function(0.0).abs() < 1e-12 && j_function(20.0) > 0.999_99);
//! assert!((j_inverse(j_function(1.7)) - 1.7).abs() < 1e-9);
//! ```
//!
//! # References
//! 1. S. ten Brink, "Convergence behavior of iteratively decoded parallel concatenated
//!    codes," *IEEE Trans. Commun.* 49(10), 1727–1737, 2001: eqs. (9)–(19), Tables II, IV.

use crate::bounds::bpsk_noise_variance;
use crate::conv::{Trellis, bcjr};
use comnum_core::rng::Rng;
use comnum_core::{Error, Result};
use comnum_math::{Jacobian, bisect, gaussian_nodes, ln_1p_exp};
use std::f64::consts::LN_2;

/// $J(\sigma)$, the mutual information between a bit and a consistent Gaussian LLR of
/// standard deviation $\sigma$ and mean $\sigma^{2}/2$ (ten Brink 2001, eq. (14)).
pub fn j_function(sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return 0.0;
    }
    let mu = sigma * sigma / 2.0;
    let loss: f64 =
        gaussian_nodes(sigma * sigma, false, 64).iter().map(|(w, p)| p * ln_1p_exp(-(mu + w.re)) / LN_2).sum();
    (1.0 - loss).clamp(0.0, 1.0)
}

/// The inverse of [`j_function`] on $[0, 1)$, by bisection of $\sigma$ on $[0, 60]$.
pub fn j_inverse(mi: f64) -> f64 {
    if mi <= 0.0 {
        return 0.0;
    }
    bisect(|s| j_function(s) < mi, 0.0, 60.0, 80)
}

/// The mutual information between the bits `bits` (0 or 1) and their LLRs `llr`
/// (positive for 0), by the time average $1 - \frac1N\sum\log_{2}(1 + e^{-x_{k}L_{k}})$,
/// $x_{k} = 1 - 2b_{k}$.
///
/// # Errors
/// [`Error::Shape`] for lengths that differ or none.
pub fn llr_mutual_information(bits: &[u8], llr: &[f64]) -> Result<f64> {
    if bits.len() != llr.len() || bits.is_empty() {
        return Err(Error::Shape(format!("{} bits for {} LLRs", bits.len(), llr.len())));
    }
    let loss: f64 = bits
        .iter()
        .zip(llr)
        .map(|(&b, &l)| {
            let x = if b == 0 { 1.0 } else { -1.0 };
            ln_1p_exp(-x * l) / LN_2
        })
        .sum();
    Ok(1.0 - loss / bits.len() as f64)
}

/// The transfer curve of a constituent decoder on a grid of a priori informations.
#[derive(Clone, Debug, PartialEq)]
pub struct ExitCurve {
    /// The a priori informations $I_{A}$, increasing.
    pub apriori: Vec<f64>,
    /// The extrinsic informations $I_{E} = T(I_{A})$.
    pub extrinsic: Vec<f64>,
}

impl ExitCurve {
    /// Measure the transfer curve of the recursive systematic code `trellis` (rate $1/2$:
    /// systematic and parity bits) whose parity is kept once every `parity_period` bits
    /// (2: a rate-2/3 constituent), at the $E_{b}/N_{0}$ `ebn0_db` of the concatenated
    /// code of rate `rate`, on a block of `n_bits` information bits (unterminated,
    /// log-MAP BCJR), at the a priori informations `apriori`. The bits, the channel and
    /// the a priori noise are drawn from `seed`: two curves of the same seed share them.
    ///
    /// # Errors
    /// [`Error::Config`] for a trellis that is not of rate 1/2, a period of 0, no bit, or
    /// a rate outside $(0, 1]$; as [`bcjr`].
    pub fn measure(
        trellis: &Trellis,
        parity_period: usize,
        ebn0_db: f64,
        rate: f64,
        n_bits: usize,
        apriori: &[f64],
        seed: u64,
    ) -> Result<ExitCurve> {
        if trellis.n_out() != 2 || parity_period == 0 || n_bits == 0 || !(rate > 0.0 && rate <= 1.0) {
            return Err(Error::Config(format!(
                "an EXIT curve needs a rate-1/2 recursive code, a period >= 1, bits and a rate in (0, 1], got n_out {}, period {parity_period}, {n_bits} bits, rate {rate}",
                trellis.n_out()
            )));
        }
        let mut rng = Rng::new(seed);
        let bits: Vec<u8> = (0..n_bits).map(|_| rng.bit()).collect();
        let coded = trellis.encode(&bits, false);
        let sd = bpsk_noise_variance(ebn0_db, rate).sqrt();
        let llr: Vec<f64> = coded
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                let x = if b == 0 { 1.0 } else { -1.0 };
                let y = x + sd * rng.gaussian();
                // the parity of step k = i / 2 is punctured unless k is a multiple of the period
                if i % 2 == 1 && (i / 2) % parity_period != 0 { 0.0 } else { 2.0 * y / (sd * sd) }
            })
            .collect();
        let noise: Vec<f64> = (0..n_bits).map(|_| rng.gaussian()).collect();
        let extrinsic = apriori
            .iter()
            .map(|&ia| {
                let sa = j_inverse(ia);
                let a: Vec<f64> = bits
                    .iter()
                    .zip(&noise)
                    .map(|(&b, &n)| {
                        let x = if b == 0 { 1.0 } else { -1.0 };
                        sa * sa / 2.0 * x + sa * n
                    })
                    .collect();
                let d = bcjr(trellis, &llr, &a, false, Jacobian::Exact)?;
                let e: Vec<f64> = (0..n_bits).map(|k| d[k] - a[k] - llr[2 * k]).collect();
                llr_mutual_information(&bits, &e)
            })
            .collect::<Result<Vec<f64>>>()?;
        Ok(ExitCurve { apriori: apriori.to_vec(), extrinsic })
    }

    /// $T(I)$ by linear interpolation on the grid (clamped at its ends).
    pub fn transfer(&self, i: f64) -> f64 {
        let x = &self.apriori;
        if i <= x[0] {
            return self.extrinsic[0];
        }
        for k in 1..x.len() {
            if i <= x[k] {
                let t = (i - x[k - 1]) / (x[k] - x[k - 1]);
                return self.extrinsic[k - 1] + t * (self.extrinsic[k] - self.extrinsic[k - 1]);
            }
        }
        self.extrinsic[x.len() - 1]
    }

    /// Whether the tunnel between this curve (first decoder) and `second` is open: the
    /// alternating passes $I \to T_{2}(T_{1}(I))$ from $I = 0$ reach `target`.
    pub fn tunnel_open(&self, second: &ExitCurve, target: f64) -> bool {
        let mut i = 0.0;
        for _ in 0..10_000 {
            let next = second.transfer(self.transfer(i));
            if next >= target {
                return true;
            }
            if next <= i + 1e-9 {
                return false;
            }
            i = next;
        }
        false
    }
}

/// The pinch-off $E_{b}/N_{0}$ (dB) of the parallel concatenation of `first` and
/// `second` (rate `rate`, parities punctured by `parity_period`): the bisection, `steps`
/// steps on $[$`lo`$,$ `hi`$]$, of the $E_{b}/N_{0}$ at which the passes reach $I$ = `end`
/// (no intersection of the curves before the end of the chart; the curve of a code of
/// small memory reaches 1 only at $I_{A} = 1$, so `end` stays below 1: 0.97), every curve measured with [`ExitCurve::measure`] on `n_bits` bits and the
/// same `seed`, on the grid $I_{A} = 0, 0.02, \dots, 0.98$ and 0.995.
///
/// # Errors
/// As [`ExitCurve::measure`].
#[allow(clippy::too_many_arguments, reason = "the parameters of the measurement and of the search")]
pub fn pinch_off_db(
    first: &Trellis,
    second: &Trellis,
    parity_period: usize,
    rate: f64,
    n_bits: usize,
    seed: u64,
    (lo, hi): (f64, f64),
    steps: usize,
    end: f64,
) -> Result<f64> {
    let grid: Vec<f64> = (0..50).map(|k| f64::from(k) * 0.02).chain(std::iter::once(0.995)).collect();
    let mut err = None;
    let t = bisect(
        |db| {
            let curves = ExitCurve::measure(first, parity_period, db, rate, n_bits, &grid, seed)
                .and_then(|c1| Ok((c1, ExitCurve::measure(second, parity_period, db, rate, n_bits, &grid, seed)?)));
            match curves {
                // below: the tunnel is closed
                Ok((c1, c2)) => !c1.tunnel_open(&c2, end),
                Err(e) => {
                    err = Some(e);
                    false
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The estimator gives J(sigma) on LLRs drawn from the a priori model, and 0 on
    /// LLRs that carry no information.
    #[test]
    fn the_estimator_recovers_j_on_gaussian_llrs() {
        let mut rng = Rng::new(9);
        for sigma in [0.5, 1.5, 3.0] {
            let bits: Vec<u8> = (0..200_000).map(|_| rng.bit()).collect();
            let llr: Vec<f64> = bits
                .iter()
                .map(|&b| {
                    let x = if b == 0 { 1.0 } else { -1.0 };
                    sigma * sigma / 2.0 * x + sigma * rng.gaussian()
                })
                .collect();
            let mi = llr_mutual_information(&bits, &llr).unwrap();
            assert!((mi - j_function(sigma)).abs() < 0.005, "{sigma}: {mi} {}", j_function(sigma));
        }
        assert!(llr_mutual_information(&[0, 1], &[0.0, 0.0]).unwrap().abs() < 1e-15);
        assert!(llr_mutual_information(&[0], &[]).is_err());
    }

    /// A transfer curve rises with the a priori information, and a stronger channel lifts
    /// it; with full a priori information the extrinsic information is close to 1.
    #[test]
    fn transfer_curves_rise_with_the_apriori_and_the_channel() {
        let t = Trellis::recursive(3, 0o7, 0o5).unwrap();
        let grid = [0.0, 0.5, 0.99];
        let low = ExitCurve::measure(&t, 2, 0.5, 0.5, 20_000, &grid, 1).unwrap();
        let high = ExitCurve::measure(&t, 2, 2.0, 0.5, 20_000, &grid, 1).unwrap();
        assert!(low.extrinsic.windows(2).all(|w| w[1] > w[0]), "{low:?}");
        assert!(high.extrinsic.iter().zip(&low.extrinsic).all(|(h, l)| h > l));
        assert!(high.extrinsic[2] > 0.95);
        assert!(ExitCurve::measure(&t, 0, 1.0, 0.5, 100, &grid, 1).is_err());
    }
}
