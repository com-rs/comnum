//! The differential group delay of `comnum_channel::optical::pmd::PmdLink` against the
//! statistics of G. J. Foschini and C. D. Poole, "Statistical theory of polarization
//! dispersion in single mode fibers," *J. Lightw. Technol.* 9(11), 1439–1456, 1991.
//!
//! Over long fibres the polarisation dispersion vector $\Omega$ tends to a three-dimensional
//! Brownian motion with i.i.d. Gaussian components (Sec. V): its length, the DGD $\tau =
//! |\Omega|$, has the Maxwellian density $p(\tau) = \sqrt{2/\pi}\,\tau^{2}\sigma^{-3}
//! e^{-\tau^{2}/2\sigma^{2}}$ of Fig. 3 ($\sigma$ the standard deviation of one component),
//! with $E\tau = 2\sigma\sqrt{2/\pi}$ and $E\tau^{2} = 3\sigma^{2}$, so that $E\tau/\sqrt{E\tau^{2}}
//! = \sqrt{8/(3\pi)}$; and the rms DGD grows as the square root of the length (eq. (18) for
//! $\sigma^{2}l \gg 1$, Fig. 1). The model of comnum is a concatenation of birefringent
//! sections with uniform rotations between them (the coarse-step model of Marcuse,
//! Menyuk & Wai 1997), whose PMD vector is the random walk of the section vectors: the
//! asymptotic regime of the paper. The DGD is the length of the output PMD vector of
//! `PmdLink::lumped_response` (eq. (5.19) of Gordon & Kogelnik 2000,
//! `comnum_validation::pmd::pmd_vectors`).
//!
//! The cards are `validation/cards/foschini1991_fig3.yaml` (the Maxwellian law: the
//! mean, the ratio of the mean to the rms, the quartiles, the density at the mode of
//! Fig. 3 for a mean of 21 ps, the components) and `validation/cards/foschini1991_eq18.yaml`
//! (the square-root growth: the mean DGD of `PmdLink::random` over 1, 2 and 4 spans
//! against the PMD parameter times the square root of the length).

use comnum::channel::optical::fiber::FiberParams;
use comnum::channel::optical::pmd::PmdLink;
use comnum::core::rng::Rng;
use comnum::math::special::erf;
use comnum_validation::card::Card;
use comnum_validation::pmd::pmd_vectors;
use serde_json::json;
use std::f64::consts::PI;

/// The mean DGD of Fig. 3 (s).
const MEAN_DGD: f64 = 21e-12;
/// Sections of the links of Fig. 3.
const SECTIONS: usize = 20;
/// Links drawn.
const LINKS: usize = 20_000;

fn norm(a: &[f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// The Maxwell CDF $F(\tau) = \operatorname{erf}(\tau/\sqrt{2}\sigma) - \sqrt{2/\pi}\,(\tau/\sigma)
/// e^{-\tau^{2}/2\sigma^{2}}$.
fn maxwell_cdf(tau: f64, sigma: f64) -> f64 {
    let u = tau / sigma;
    erf(u / 2f64.sqrt()) - (2.0 / PI).sqrt() * u * (-u * u / 2.0).exp()
}

/// The Maxwell density of Fig. 3 (1/s).
fn maxwell_pdf(tau: f64, sigma: f64) -> f64 {
    (2.0 / PI).sqrt() * tau * tau / sigma.powi(3) * (-tau * tau / (2.0 * sigma * sigma)).exp()
}

/// The quantile `p` of the Maxwell law of parameter `sigma`, by bisection.
fn maxwell_quantile(p: f64, sigma: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, 10.0 * sigma);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if maxwell_cdf(mid, sigma) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[test]
#[ignore = "statistical validation: run with --release --ignored"]
fn the_dgd_of_the_sections_is_maxwellian() {
    let mut card = Card::load("foschini1991_fig3").unwrap();
    let mut rng = Rng::new(1991);
    let mut dgd = Vec::with_capacity(LINKS);
    // the sum and the sum of squares of every component of the input PMD vector: the
    // first section starts with a uniform rotation, so the input frame is isotropic; the
    // output one is not (no rotation after the last section: its DGD lies along the
    // first Stokes axis of the output frame, a mean of tau_N on that component)
    let mut comp = [[0.0f64; 2]; 3];
    for _ in 0..LINKS {
        let link = PmdLink::with_mean_dgd(MEAN_DGD, SECTIONS, &mut rng);
        let (tin, tout, _) = pmd_vectors(&link, 0.0);
        dgd.push(norm(&tout));
        for (c, v) in comp.iter_mut().zip(tin) {
            c[0] += v;
            c[1] += v * v;
        }
    }
    let n = LINKS as f64;
    let mean = dgd.iter().sum::<f64>() / n;
    let rms = (dgd.iter().map(|t| t * t).sum::<f64>() / n).sqrt();
    let sigma = rms / 3f64.sqrt(); // the component s.d. of a Maxwellian of that rms
    println!(
        "mean DGD {:.2} ps (target {:.0}), rms {:.2} ps, mean/rms {:.4} (Maxwell: {:.4})",
        mean * 1e12,
        MEAN_DGD * 1e12,
        rms * 1e12,
        mean / rms,
        (8.0 / (3.0 * PI)).sqrt()
    );
    card.check(&json!({"quantity": "mean_over_target"}), mean / MEAN_DGD);
    card.check(&json!({"quantity": "mean_over_rms"}), mean / rms);
    for p in [0.25, 0.5, 0.75] {
        let q = maxwell_quantile(p, sigma);
        let below = dgd.iter().filter(|&&t| t < q).count() as f64 / n;
        println!("fraction below the Maxwell quantile {p}: {below:.4}");
        card.check(&json!({"quantity": "fraction_below_quantile", "p": p}), below);
    }
    // the density at the mode of Fig. 3 (sqrt(2) sigma = 16.7 ps for a mean of 21 ps):
    // a bin of 1 ps, in 1/ps
    let mode = 2f64.sqrt() * MEAN_DGD * (PI / 8.0).sqrt();
    let bin = 1e-12;
    let count = dgd.iter().filter(|&&t| (t - mode).abs() < bin / 2.0).count() as f64;
    let density = count / n / bin * 1e-12;
    println!(
        "density at the mode {:.1} ps: {density:.4} per ps (Maxwell of the same mean: {:.4})",
        mode * 1e12,
        maxwell_pdf(mode, MEAN_DGD * (PI / 8.0).sqrt()) * 1e-12
    );
    card.check(&json!({"quantity": "density_at_mode_per_ps"}), density);
    // the components: zero mean and equal variances, E tau^2 / 3 each
    for (k, c) in comp.iter().enumerate() {
        let m = c[0] / n;
        let var = c[1] / n - m * m;
        println!("component {k}: mean/rms {:.4}, variance over E tau^2 / 3: {:.4}", m / rms, var / (rms * rms / 3.0));
        card.check(&json!({"quantity": "component_mean_over_rms", "component": k}), m / rms);
        card.check(&json!({"quantity": "component_variance_ratio", "component": k}), var / (rms * rms / 3.0));
    }
    card.finish().unwrap();
}

#[test]
#[ignore = "statistical validation: run with --release --ignored"]
fn the_rms_dgd_grows_as_the_square_root_of_the_length() {
    let mut card = Card::load("foschini1991_eq18").unwrap();
    let mut rng = Rng::new(18);
    let pmd = 0.1; // ps per sqrt(km)
    let links = 4000;
    let mut rms_of = Vec::new();
    for n_spans in [1usize, 2, 4] {
        let fiber = FiberParams { n_spans, ..FiberParams::default() };
        let length_km = fiber.span_km * n_spans as f64;
        let (mut sum, mut sq) = (0.0, 0.0);
        for _ in 0..links {
            let link = PmdLink::random(&fiber, pmd, 20, &mut rng).unwrap();
            let t = norm(&pmd_vectors(&link, 0.0).1) * 1e12;
            sum += t;
            sq += t * t;
        }
        let (mean, rms) = (sum / links as f64, (sq / links as f64).sqrt());
        rms_of.push(rms);
        println!(
            "{n_spans} span(s), {length_km:.0} km: mean DGD {mean:.3} ps, rms {rms:.3} ps; mean over pmd sqrt(L) = {:.4}",
            mean / (pmd * length_km.sqrt())
        );
        card.check(
            &json!({"quantity": "mean_over_pmd_sqrt_length", "spans": n_spans}),
            mean / (pmd * length_km.sqrt()),
        );
    }
    card.check(&json!({"quantity": "rms_ratio", "spans": "2 over 1"}), rms_of[1] / rms_of[0]);
    card.check(&json!({"quantity": "rms_ratio", "spans": "4 over 1"}), rms_of[2] / rms_of[0]);
    card.finish().unwrap();
}
