//! The confidence intervals of a binomial proportion of `comnum::sim::stats`
//! (`ErrorRate::interval`, `BinomialInterval`, `coverage`, `expected_length`) against
//! L. D. Brown, T. T. Cai and A. DasGupta, "Interval estimation for a binomial
//! proportion," *Statistical Science* 16(2), 101–133, 2001.
//!
//! Every value of the paper is an exact probability, not a simulation (Sec. 2.1): the
//! coverage $C(p, n) = \sum_{x} \mathbf{1}\{L(x) \le p \le U(x)\} \binom{n}{x} p^{x}
//! (1 - p)^{n - x}$ of an interval for $X \sim \mathrm{Bin}(n, p)$, its expected length
//! $\sum_{x} (U(x) - L(x)) \binom{n}{x} p^{x} (1 - p)^{n - x}$, or the ends of the
//! interval. The comparisons, at $1 - \alpha = 0.95$ unless stated:
//!
//! - the Wald interval, eq. (1): Table 1 ($p = 0.5$, $10 \le n \le 50$), Table 2 and
//!   Sec. 1 ($p = 0.005$ up to $n = 1876$), Sec. 1 ($n = 100$, $p = 0.106$ and 0.107),
//!   Example 1 ($p = 0.2$), Example 5 (99 %, $n = 20$: the coverage averaged over $p$,
//!   by the midpoint rule on 20 000 points);
//! - the limits as $n \to \infty$ with $np = \gamma$ fixed, where $X$ becomes
//!   $\mathrm{Poisson}(\gamma)$ and the intervals of $\gamma = np$ become
//!   $X \pm \kappa\sqrt{X}$ (Wald) and $X + \kappa^{2}/2 \pm \kappa\sqrt{X + \kappa^{2}/4}$
//!   (Wilson, eq. (4) multiplied by $n$): the bound of eq. (3) (Table 3), the infimum of
//!   the coverage over $\gamma \ge 1$, 5 and 10 (Sec. 3.2; the coverage jumps at the ends
//!   of the intervals, so the infimum is taken just beyond every end) and the spikes of
//!   the Wilson interval at $\gamma = 0.1765$ (95 %) and 0.1174 (99 %) (Sec. 4.1.1);
//! - the expected lengths of Fig. 8 ($n = 25$): the values of $p$ where the shortest of
//!   the Wald, Wilson and Jeffreys intervals changes (Sec. 3.3), by bisection;
//! - the Jeffreys interval, eqs. (6)–(8): the 252 pairs of limits of Table 5
//!   ($7 \le n \le 30$).
//!
//! The values are in the cards `validation/cards/brown2001_coverage.yaml` and
//! `validation/cards/brown2001_table5.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::math::{bisect, ln_gamma};
use comnum::sim::stats::{BinomialInterval, ErrorRate, coverage, expected_length, z_score};
use comnum_validation::card::Card;
use serde_json::json;

/// $P(X = x)$ for $X \sim \mathrm{Poisson}(\gamma)$.
fn poisson(gamma: f64, x: u32) -> f64 {
    (-gamma + f64::from(x) * gamma.ln() - ln_gamma(f64::from(x) + 1.0)).exp()
}

/// The counts beyond which the Poisson probabilities of $\gamma \le 60$ are negligible.
const COUNTS: u32 = 200;

/// The interval of $\gamma = np$ for $X$ successes as $n \to \infty$.
type Limit = fn(f64, f64) -> (f64, f64);

/// Eq. (1) times $n$: $X \pm \kappa \sqrt{X}$.
fn wald_limit(x: f64, kappa: f64) -> (f64, f64) {
    (x - kappa * x.sqrt(), x + kappa * x.sqrt())
}

/// Eq. (4) times $n$: $X + \kappa^{2}/2 \pm \kappa \sqrt{X + \kappa^{2}/4}$.
fn wilson_limit(x: f64, kappa: f64) -> (f64, f64) {
    let half = kappa * (x + kappa * kappa / 4.0).sqrt();
    (x + kappa * kappa / 2.0 - half, x + kappa * kappa / 2.0 + half)
}

/// The limiting coverage of $\gamma$: the Poisson probability of the counts whose
/// interval contains it.
fn limit_coverage(interval: Limit, kappa: f64, gamma: f64) -> f64 {
    (0..COUNTS)
        .filter(|&x| {
            let (lo, hi) = interval(f64::from(x), kappa);
            lo <= gamma && gamma <= hi
        })
        .map(|x| poisson(gamma, x))
        .sum()
}

/// The infimum of the limiting coverage over $\gamma_{0} \le \gamma \le 60$: the coverage
/// is constant between the ends of the intervals and jumps at them, so the infimum is
/// reached just beside an end (or at $\gamma_{0}$).
fn limit_infimum(interval: Limit, kappa: f64, gamma0: f64) -> f64 {
    let mut candidates = vec![gamma0];
    for x in 0..COUNTS {
        let (lo, hi) = interval(f64::from(x), kappa);
        for end in [lo, hi] {
            candidates.extend([end - 1e-9, end + 1e-9].into_iter().filter(|g| (gamma0..=60.0).contains(g)));
        }
    }
    candidates.into_iter().map(|g| limit_coverage(interval, kappa, g)).fold(f64::INFINITY, f64::min)
}

/// The right side of eq. (3): $P(a_{\gamma} < \mathrm{Poisson}(\gamma) \le b_{\gamma})$.
fn eq_3_bound(gamma: f64, kappa: f64) -> f64 {
    let root = kappa * (kappa * kappa + 4.0 * gamma).sqrt();
    let a = ((kappa * kappa + 2.0 * gamma - root) / 2.0).floor() as u32;
    let b = ((kappa * kappa + 2.0 * gamma + root) / 2.0).floor() as u32;
    (a + 1..=b).map(|x| poisson(gamma, x)).sum()
}

/// The $p$ of $(a, b)$ where the expected lengths of two intervals at $n = 25$ cross.
fn length_crossing(first: BinomialInterval, second: BinomialInterval, a: f64, b: f64) -> f64 {
    let diff = |p: f64| expected_length(first, 25, p, 0.95) - expected_length(second, 25, p, 0.95);
    let below_sign = diff(a) > 0.0;
    bisect(|p| (diff(p) > 0.0) == below_sign, a, b, 60)
}

#[test]
fn brown_cai_dasgupta_2001_coverage_of_the_wald_and_wilson_intervals() {
    let mut card = Card::load("brown2001_coverage").unwrap();
    let wald = |n: u64, p: f64| coverage(BinomialInterval::Wald, n, p, 0.95);
    // Table 1
    for n in [17, 20, 25, 30, 35, 37, 42, 44, 49, 10, 12, 13, 15, 18, 23, 28, 33, 40] {
        card.check(&json!({"interval": "wald", "n": n, "p": 0.5}), wald(n, 0.5));
    }
    // Table 2 and Sec. 1
    for n in [591, 592, 953, 954, 1279, 1583, 1876] {
        card.check(&json!({"interval": "wald", "n": n, "p": 0.005}), wald(n, 0.005));
    }
    for (n, p) in [(100, 0.106), (100, 0.107), (30, 0.2), (98, 0.2)] {
        card.check(&json!({"interval": "wald", "n": n, "p": p}), wald(n, p));
    }
    // Example 5: the average over p of the coverage at 99 %, n = 20
    let points = 20_000;
    let average = (0..points)
        .map(|i| coverage(BinomialInterval::Wald, 20, (f64::from(i) + 0.5) / f64::from(points), 0.99))
        .sum::<f64>()
        / f64::from(points);
    card.check(&json!({"interval": "wald", "n": 20, "confidence": 0.99, "quantity": "average"}), average);
    // Table 3: eq. (3); Sec. 3.2: the limiting infima
    let (k95, k99) = (z_score(0.95), z_score(0.99));
    for gamma in [5.0, 7.0, 10.0] {
        card.check(&json!({"quantity": "eq_3", "gamma": gamma}), eq_3_bound(gamma, k95));
    }
    for gamma in [5.0, 10.0] {
        card.check(
            &json!({"interval": "wald", "quantity": "limit_infimum", "gamma": gamma}),
            limit_infimum(wald_limit, k95, gamma),
        );
    }
    for gamma in [1.0, 5.0, 10.0] {
        card.check(
            &json!({"interval": "wilson", "quantity": "limit_infimum", "gamma": gamma}),
            limit_infimum(wilson_limit, k95, gamma),
        );
    }
    // Sec. 4.1.1: the spikes of the Wilson interval
    for (confidence, kappa, gamma) in [(0.95, k95, 0.1765), (0.99, k99, 0.1174)] {
        card.check(
            &json!({"interval": "wilson", "quantity": "limit_spike", "confidence": confidence, "gamma": gamma}),
            limit_coverage(wilson_limit, kappa, gamma),
        );
    }
    // Sec. 3.3, Fig. 8: where the shortest interval changes at n = 25
    card.check(
        &json!({"quantity": "shortest", "from": "wald", "to": "jeffreys"}),
        length_crossing(BinomialInterval::Wald, BinomialInterval::Jeffreys, 0.05, 0.19),
    );
    card.check(
        &json!({"quantity": "shortest", "from": "jeffreys", "to": "wilson"}),
        length_crossing(BinomialInterval::Jeffreys, BinomialInterval::Wilson, 0.19, 0.3),
    );
    card.finish().unwrap();
}

#[test]
fn brown_cai_dasgupta_2001_table_5_jeffreys_limits() {
    let mut card = Card::load("brown2001_table5").unwrap();
    for n in 7..=30u64 {
        for x in 0..=n.div_ceil(2) {
            let (lo, hi) = ErrorRate { errors: x, total: n }.interval(BinomialInterval::Jeffreys, 0.95);
            card.check(&json!({"n": n, "x": x, "end": "lower"}), lo);
            card.check(&json!({"n": n, "x": x, "end": "upper"}), hi);
        }
    }
    card.finish().unwrap();
}
