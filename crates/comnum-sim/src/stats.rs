//! Running statistics with confidence intervals.
//!
//! - [`ErrorRate`] counts errors among trials of a Bernoulli event (bit, symbol or block
//!   errors). Its estimate is $\hat{p} = k/n$ and its interval the **Wilson score
//!   interval** at confidence $1 - \alpha$,
//!
//!   $$
//!   \frac{\hat{p} + \frac{z^{2}}{2n} \pm z \sqrt{\frac{\hat{p}(1 - \hat{p})}{n} + \frac{z^{2}}{4n^{2}}}}{1 + \frac{z^{2}}{n}},
//!   \qquad z = \Phi^{-1}(1 - \alpha/2),
//!   $$
//!
//!   which, unlike the normal approximation, stays inside $[0, 1]$ and is informative
//!   with few or no errors (with $k = 0$ its upper end is $z^{2}/(n + z^{2})$). The other
//!   intervals of a proportion compared by Brown, Cai & DasGupta (2001) are available
//!   through [`ErrorRate::interval`] and [`BinomialInterval`]: Wald, Agresti–Coull,
//!   Jeffreys and Clopper–Pearson, with their exact [`coverage`] and
//!   [`expected_length`].
//! - [`Mean`] accumulates real values with Welford's algorithm (numerically stable
//!   running mean and variance) and merges two accumulators exactly (Chan, Golub and
//!   LeVeque); its interval is the normal interval of the mean,
//!   $\bar{x} \pm z s/\sqrt{n}$.
//! - [`Estimate`] is either of the two, with the same interface.
//!
//! # Examples
//! ```
//! use comnum_sim::stats::{ErrorRate, Mean, Estimate, z_score};
//! let mut e = ErrorRate::default();
//! e.add(3, 1000);
//! e.add(7, 1000);
//! assert_eq!(e.rate(), 5e-3);
//! let (lo, hi) = e.wilson(0.95);
//! assert!(lo < 5e-3 && 5e-3 < hi);
//! let mut m = Mean::default();
//! for v in [1.0, 2.0, 3.0, 4.0] { m.add(v); }
//! assert_eq!((m.mean(), m.variance()), (2.5, 5.0 / 3.0));
//! assert!((z_score(0.95) - 1.959964).abs() < 1e-5);
//! let est = Estimate::Rate(e);
//! assert_eq!(est.value(), 5e-3);
//! ```

//!
//! # References
//!
//! 1. L. D. Brown, T. T. Cai, A. DasGupta, "Interval estimation for a binomial
//!    proportion," *Statistical Science* 16(2), 101–133, 2001: eqs. (1), (4), (5),
//!    (6)–(8) and Sec. 4.2.1 (the intervals), Sec. 2 (the coverage), Sec. 3.3 (the
//!    expected length).

use comnum_core::{Error, Result};
use comnum_math::{beta_inv, ln_binomial_pmf, qfuncinv};

/// The quantile $z = \Phi^{-1}(1 - \alpha/2) = Q^{-1}(\alpha/2)$ of a two-sided
/// confidence level $1 - \alpha$ (1.96 for 0.95); the level is clamped to $[0, 1)$ (the
/// runner refuses a level outside $(0, 1)$, see
/// [`check_confidence`](crate::stopping::check_confidence)).
pub fn z_score(confidence: f64) -> f64 {
    qfuncinv((1.0 - confidence.clamp(0.0, 1.0 - 1e-16)) / 2.0)
}

/// Errors among trials of a Bernoulli event.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ErrorRate {
    /// Number of errors $k$.
    pub errors: u64,
    /// Number of trials $n$.
    pub total: u64,
}

impl ErrorRate {
    /// Add `errors` errors among `total` new trials.
    pub fn add(&mut self, errors: u64, total: u64) {
        self.errors += errors;
        self.total += total;
    }

    /// Merge another count.
    pub fn merge(&mut self, o: &ErrorRate) {
        self.add(o.errors, o.total);
    }

    /// $k/n$ (`NaN` without trials).
    pub fn rate(&self) -> f64 {
        if self.total == 0 { f64::NAN } else { self.errors as f64 / self.total as f64 }
    }

    /// Wilson score interval at the given confidence (`(0, 1)` without trials): eq. (4)
    /// of Brown et al. (2001), [`BinomialInterval::Wilson`].
    pub fn wilson(&self, confidence: f64) -> (f64, f64) {
        if self.total == 0 {
            return (0.0, 1.0);
        }
        let n = self.total as f64;
        let p = self.rate();
        let z = z_score(confidence);
        let z2 = z * z;
        let centre = (p + z2 / (2.0 * n)) / (1.0 + z2 / n);
        let half = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / (1.0 + z2 / n);
        // without errors the lower end is exactly 0: `centre - half` would leave a rounding residue
        let lower = if self.errors == 0 { 0.0 } else { (centre - half).max(0.0) };
        (lower, (centre + half).min(1.0))
    }

    /// The interval of the rate by `method` at the given confidence (`(0, 1)` without
    /// trials; see [`BinomialInterval`]: the Wald and Agresti–Coull ends may leave
    /// $[0, 1]$, as the paper defines them).
    ///
    /// # Examples
    /// ```
    /// use comnum_sim::stats::{BinomialInterval, ErrorRate};
    /// let r = ErrorRate { errors: 3, total: 7 };
    /// // Table 5 of Brown et al. (2001): the Jeffreys interval of 3 successes out of 7
    /// let (lo, hi) = r.interval(BinomialInterval::Jeffreys, 0.95);
    /// assert!((lo - 0.139).abs() < 1e-3 && (hi - 0.766).abs() < 1e-3);
    /// // Clopper-Pearson contains Jeffreys, Agresti-Coull contains Wilson
    /// let (cl, ch) = r.interval(BinomialInterval::ClopperPearson, 0.95);
    /// assert!(cl < lo && hi < ch);
    /// ```
    pub fn interval(&self, method: BinomialInterval, confidence: f64) -> (f64, f64) {
        if self.total == 0 {
            return (0.0, 1.0);
        }
        let (k, n) = (self.errors as f64, self.total as f64);
        let alpha = 1.0 - confidence;
        let z = z_score(confidence);
        match method {
            BinomialInterval::Wald => {
                let p = k / n;
                let half = z * (p * (1.0 - p) / n).sqrt();
                (p - half, p + half)
            }
            BinomialInterval::Wilson => self.wilson(confidence),
            BinomialInterval::AgrestiCoull => {
                let nt = n + z * z;
                let pt = (k + z * z / 2.0) / nt;
                let half = z * (pt * (1.0 - pt) / nt).sqrt();
                (pt - half, pt + half)
            }
            BinomialInterval::Jeffreys => {
                let lo = if self.errors == 0 { 0.0 } else { beta_inv(alpha / 2.0, k + 0.5, n - k + 0.5) };
                let hi =
                    if self.errors == self.total { 1.0 } else { beta_inv(1.0 - alpha / 2.0, k + 0.5, n - k + 0.5) };
                (lo, hi)
            }
            BinomialInterval::ClopperPearson => {
                let lo = if self.errors == 0 { 0.0 } else { beta_inv(alpha / 2.0, k, n - k + 1.0) };
                let hi = if self.errors == self.total { 1.0 } else { beta_inv(1.0 - alpha / 2.0, k + 1.0, n - k) };
                (lo, hi)
            }
        }
    }
}

/// A confidence interval of a binomial proportion $p$ from $k$ successes among $n$
/// trials, $\hat{p} = k/n$, $z = \Phi^{-1}(1 - \alpha/2)$ (Brown, Cai & DasGupta 2001,
/// with their equations):
///
/// - [`Wald`](Self::Wald), eq. (1): $\hat{p} \pm z \sqrt{\hat{p}(1 - \hat{p})/n}$, the
///   textbook interval, whose coverage oscillates far below $1 - \alpha$ (a single
///   point at $k = 0$ or $n$);
/// - [`Wilson`](Self::Wilson), eq. (4): the score interval (the default of the runner);
/// - [`AgrestiCoull`](Self::AgrestiCoull), eq. (5): the Wald interval of
///   $\tilde{p} = (k + z^{2}/2)/(n + z^{2})$ with $\tilde{n} = n + z^{2}$, which contains
///   the Wilson interval;
/// - [`Jeffreys`](Self::Jeffreys), eqs. (6)–(8): the $\alpha/2$ and $1 - \alpha/2$
///   quantiles of $\mathrm{Beta}(k + 1/2, n - k + 1/2)$, 0 at $k = 0$ and 1 at $k = n$;
/// - [`ClopperPearson`](Self::ClopperPearson), Sec. 4.2.1: the quantiles of
///   $\mathrm{Beta}(k, n - k + 1)$ and $\mathrm{Beta}(k + 1, n - k)$, the "exact"
///   interval whose coverage never falls below $1 - \alpha$.
///
/// The Wald and Agresti–Coull intervals are returned as the paper defines them, not
/// truncated to $[0, 1]$: their expected lengths are those of its Sec. 3.3 (Fig. 8), and
/// truncating them would not change their coverage of $p \in (0, 1)$.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinomialInterval {
    /// The standard interval, eq. (1).
    Wald,
    /// The score interval, eq. (4).
    Wilson,
    /// The Agresti–Coull interval, eq. (5).
    AgrestiCoull,
    /// The equal-tailed Jeffreys interval, eqs. (6)–(8).
    Jeffreys,
    /// The Clopper–Pearson interval, Sec. 4.2.1.
    ClopperPearson,
}

impl BinomialInterval {
    /// The five intervals.
    pub const ALL: [BinomialInterval; 5] = [
        BinomialInterval::Wald,
        BinomialInterval::Wilson,
        BinomialInterval::AgrestiCoull,
        BinomialInterval::Jeffreys,
        BinomialInterval::ClopperPearson,
    ];

    /// The name of the interval: `wald`, `wilson`, `agresti_coull`, `jeffreys` or
    /// `clopper_pearson`.
    pub fn name(self) -> &'static str {
        match self {
            BinomialInterval::Wald => "wald",
            BinomialInterval::Wilson => "wilson",
            BinomialInterval::AgrestiCoull => "agresti_coull",
            BinomialInterval::Jeffreys => "jeffreys",
            BinomialInterval::ClopperPearson => "clopper_pearson",
        }
    }
}

impl std::str::FromStr for BinomialInterval {
    type Err = Error;

    /// The interval of a [`BinomialInterval::name`] (any case, `-` for `_`).
    fn from_str(s: &str) -> Result<Self> {
        let key = s.to_ascii_lowercase().replace('-', "_");
        BinomialInterval::ALL.into_iter().find(|m| m.name() == key).ok_or_else(|| {
            let names: Vec<&str> = BinomialInterval::ALL.iter().map(|m| m.name()).collect();
            Error::Config(format!("unknown binomial interval {s:?}: one of {}", names.join(", ")))
        })
    }
}

/// The intervals of $k = 0, \dots, n$ successes and their binomial probabilities.
fn intervals_and_probabilities(method: BinomialInterval, n: u64, p: f64, confidence: f64) -> Vec<((f64, f64), f64)> {
    let ln_pmf = ln_binomial_pmf(n as usize, p);
    (0..=n)
        .map(|k| (ErrorRate { errors: k, total: n }.interval(method, confidence), ln_pmf[k as usize].exp()))
        .collect()
}

/// The exact coverage probability $C(p, n) = P_{p}(L(X) \le p \le U(X))$ of an interval
/// for $X \sim \mathrm{Bin}(n, p)$: the sum of the binomial probabilities of the counts
/// whose interval contains $p$ (Brown et al. 2001, Sec. 2), no simulation.
///
/// # Examples
/// ```
/// use comnum_sim::stats::{BinomialInterval, coverage};
/// // Table 1 of Brown et al. (2001): the Wald interval covers p = 1/2 with 0.891 at n = 10
/// let c = coverage(BinomialInterval::Wald, 10, 0.5, 0.95);
/// assert!((c - 0.891).abs() < 5e-4);
/// ```
pub fn coverage(method: BinomialInterval, n: u64, p: f64, confidence: f64) -> f64 {
    intervals_and_probabilities(method, n, p, confidence)
        .into_iter()
        .filter(|((lo, hi), _)| *lo <= p && p <= *hi)
        .map(|(_, w)| w)
        .sum()
}

/// The expected length $\sum_{k} (U(k) - L(k)) P_{p}(X = k)$ of an interval for
/// $X \sim \mathrm{Bin}(n, p)$ (Brown et al. 2001, Sec. 3.3), with the ends of
/// [`ErrorRate::interval`].
pub fn expected_length(method: BinomialInterval, n: u64, p: f64, confidence: f64) -> f64 {
    intervals_and_probabilities(method, n, p, confidence).into_iter().map(|((lo, hi), w)| (hi - lo) * w).sum()
}

/// Running mean and variance of real values (Welford).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[allow(clippy::struct_field_names, reason = "the running mean of Welford's algorithm")]
pub struct Mean {
    n: u64,
    mean: f64,
    m2: f64,
}

impl Mean {
    /// Add one value.
    pub fn add(&mut self, x: f64) {
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (x - self.mean);
    }

    /// Merge another accumulator (exact, Chan et al.).
    pub fn merge(&mut self, o: &Mean) {
        if o.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = *o;
            return;
        }
        let n = self.n + o.n;
        let d = o.mean - self.mean;
        self.mean += d * o.n as f64 / n as f64;
        self.m2 += o.m2 + d * d * self.n as f64 * o.n as f64 / n as f64;
        self.n = n;
    }

    /// Number of values.
    pub fn count(&self) -> u64 {
        self.n
    }

    /// Mean (`NaN` without values).
    pub fn mean(&self) -> f64 {
        if self.n == 0 { f64::NAN } else { self.mean }
    }

    /// Unbiased sample variance (`NaN` with fewer than two values).
    pub fn variance(&self) -> f64 {
        if self.n < 2 { f64::NAN } else { self.m2 / (self.n - 1) as f64 }
    }

    /// Sample standard deviation.
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Standard error of the mean $s/\sqrt{n}$.
    pub fn sem(&self) -> f64 {
        self.std() / (self.n as f64).sqrt()
    }

    /// Normal confidence interval of the mean, $\bar{x} \pm z s/\sqrt{n}$ (unbounded with fewer
    /// than two values, so that a precision rule never stops on a single value).
    pub fn interval(&self, confidence: f64) -> (f64, f64) {
        if self.n < 2 {
            return (f64::NEG_INFINITY, f64::INFINITY);
        }
        let h = z_score(confidence) * self.sem();
        (self.mean - h, self.mean + h)
    }
}

/// The estimate of one metric: an error rate or a mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Estimate {
    /// Error counts.
    Rate(ErrorRate),
    /// Values.
    Mean(Mean),
}

impl Estimate {
    /// The point estimate (rate or mean).
    pub fn value(&self) -> f64 {
        match self {
            Estimate::Rate(r) => r.rate(),
            Estimate::Mean(m) => m.mean(),
        }
    }

    /// The confidence interval (Wilson or normal).
    pub fn interval(&self, confidence: f64) -> (f64, f64) {
        match self {
            Estimate::Rate(r) => r.wilson(confidence),
            Estimate::Mean(m) => m.interval(confidence),
        }
    }

    /// Half-width of the interval relative to the estimate (`∞` for a zero estimate).
    pub fn relative_precision(&self, confidence: f64) -> f64 {
        let (lo, hi) = self.interval(confidence);
        let v = self.value().abs();
        if v == 0.0 || !v.is_finite() { f64::INFINITY } else { 0.5 * (hi - lo) / v }
    }

    /// Number of errors (rates) or of values (means).
    pub fn count(&self) -> u64 {
        match self {
            Estimate::Rate(r) => r.errors,
            Estimate::Mean(m) => m.count(),
        }
    }

    /// Number of trials (rates) or of values (means).
    pub fn total(&self) -> u64 {
        match self {
            Estimate::Rate(r) => r.total,
            Estimate::Mean(m) => m.count(),
        }
    }

    /// Merge an estimate of the same kind (a mismatch keeps `self` unchanged and
    /// returns `false`).
    pub fn merge(&mut self, o: &Estimate) -> bool {
        match (self, o) {
            (Estimate::Rate(a), Estimate::Rate(b)) => {
                a.merge(b);
                true
            }
            (Estimate::Mean(a), Estimate::Mean(b)) => {
                a.merge(b);
                true
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_intervals_nest_as_brown_cai_dasgupta_state() {
        // Sec. 3.1.2: Agresti-Coull contains Wilson; Sec. 4.3: Clopper-Pearson contains Jeffreys
        for n in 1..=60 {
            for k in 0..=n {
                let r = ErrorRate { errors: k, total: n };
                let w = r.interval(BinomialInterval::Wilson, 0.95);
                assert_eq!(w, r.wilson(0.95));
                let ac = r.interval(BinomialInterval::AgrestiCoull, 0.95);
                assert!(ac.0 <= w.0 + 1e-15 && w.1 <= ac.1 + 1e-15, "n {n} k {k}: {ac:?} {w:?}");
                let j = r.interval(BinomialInterval::Jeffreys, 0.95);
                let cp = r.interval(BinomialInterval::ClopperPearson, 0.95);
                assert!(cp.0 <= j.0 && j.1 <= cp.1, "n {n} k {k}: {cp:?} {j:?}");
            }
        }
    }

    #[test]
    fn clopper_pearson_covers_at_least_its_level() {
        // Sec. 4.2.1: the coverage of Clopper-Pearson never falls below 1 - alpha
        for n in [5, 17, 50] {
            for i in 1..200 {
                let p = f64::from(i) / 200.0;
                let c = coverage(BinomialInterval::ClopperPearson, n, p, 0.95);
                assert!(c >= 0.95 - 1e-12, "n {n} p {p}: {c}");
            }
        }
    }

    #[test]
    fn the_names_parse_back_and_unknown_ones_are_errors() {
        for m in BinomialInterval::ALL {
            assert_eq!(m.name().parse::<BinomialInterval>().unwrap(), m);
        }
        assert_eq!("Clopper-Pearson".parse::<BinomialInterval>().unwrap(), BinomialInterval::ClopperPearson);
        assert!("exact".parse::<BinomialInterval>().is_err());
        assert_eq!(ErrorRate::default().interval(BinomialInterval::Wald, 0.95), (0.0, 1.0));
    }

    #[test]
    fn wilson_with_no_error_and_merges() {
        let r = ErrorRate { errors: 0, total: 1000 };
        let (lo, hi) = r.wilson(0.95);
        assert_eq!(lo, 0.0);
        assert!((hi - 3.83e-3).abs() < 1e-4, "{hi}");
        let mut a = Mean::default();
        let mut b = Mean::default();
        let mut all = Mean::default();
        for i in 0..100 {
            let v = (i as f64 * 0.37).sin();
            if i < 30 {
                a.add(v);
            } else {
                b.add(v);
            }
            all.add(v);
        }
        a.merge(&b);
        assert!((a.mean() - all.mean()).abs() < 1e-14 && (a.variance() - all.variance()).abs() < 1e-14);
    }
}
