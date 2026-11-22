//! The optically amplified on–off keying receiver and its maximum-likelihood sequence
//! detector against T. Foggi, E. Forestieri, G. Colavolpe and G. Prati,
//! "Maximum-likelihood sequence detection with closed-form metrics in OOK optical
//! systems impaired by GVD and PMD," *J. Lightw. Technol.* 24(8), 3073–3087, 2006:
//! the chain of their Fig. 1 (NRZ OOK of 20 dB extinction ratio, ASE on two
//! polarisations of two-sided density $N_{0}$ each, a 4th-order Butterworth optical
//! filter of 3-dB bandwidth $1.9/T$, the square law $|s_{1} + n_{1}|^{2} + |s_{2} +
//! n_{2}|^{2}$, a 5th-order Bessel electrical filter of $0.75/T$, one sample per bit;
//! Sec. VI), with $E_{b}/N_{0}$ the mean energy per bit over $N_{0}$ (Sec. III).
//!
//! Fig. 6, DGD = 0 (no intersymbol interference but the filters'): the BER of the
//! receiver against $E_{b}/N_{0}$, where the MLSD and the FFE + DFE coincide with the
//! threshold receiver; here the threshold receiver with its optimum threshold, and the
//! MLSD of 8 states (`Mlse::fit` on the noiseless samples) with the Euclidean metric and
//! with the chi-square metric of eq. (12) (`Mlse::chi_square_metric`, $N_{0}$ and $\nu$
//! from the mean and variance of the samples without signal, eqs. (24)–(25) at $s_{k} =
//! 0$). The optical filter is a band-pass of 4th order: on the complex envelope, its
//! low-pass prototype of 2nd order and cut-off $0.95/T$ (a 4th-order low-pass on the
//! envelope costs 0.65 dB more, see the card). The sampling instant is the one of the
//! smallest error rate of the threshold receiver over three bit periods (the group
//! delay of the filters); $2^{19}$ bits per point.
//!
//! Sec. VI-D, Fig. 15, at the error rates a simulation reaches: the same receivers at
//! $E_{b}/N_{0}$ = 14.4 dB over the chromatic dispersion index $\gamma$ = 0.2, 0.3 and
//! 0.4 ($\gamma = -2\pi\beta_{2} L R_{b}^{2}$, standard fibre), the MLSD of 16 states
//! fitted on the dispersed noiseless samples and sampled at the threshold receiver's
//! best instant; printed and recorded as findings (the figure's penalties are at
//! $10^{-12}$, from the bounds of Sec. V, which are not implemented).
//! The values and tolerances are in the card `validation/cards/foggi2006_fig6.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::C64;
use comnum::channel::awgn::add_noise;
use comnum::channel::optical::fiber::{FiberParams, linear_fiber_channel};
use comnum::core::rng::Rng;
use comnum::dsp::equalizer::mlse::Mlse;
use comnum::dsp::filter::Iir;
use comnum::math::db2lin;
use comnum_validation::card::Card;
use serde_json::json;

const BIT_RATE: f64 = 10e9;
const SPS: usize = 16;
const FS: f64 = BIT_RATE * SPS as f64;
const EXTINCTION_DB: f64 = 20.0;
const MEMORY: usize = 3;

struct Receiver {
    optical: Iir,
    electrical: Iir,
}

impl Receiver {
    fn new() -> Receiver {
        Receiver {
            // a 4th-order band-pass of 3-dB bandwidth 1.9/T: on the envelope, the 2nd-order
            // low-pass prototype of cut-off 0.95/T
            optical: Iir::butterworth(2, 0.95 * BIT_RATE, FS).unwrap(),
            electrical: Iir::bessel(5, 0.75 * BIT_RATE, FS).unwrap(),
        }
    }

    /// The photocurrent of the field `x` (one polarisation) with ASE of variance
    /// `noise_var` per complex sample on both polarisations (none when zero).
    fn photocurrent(&self, x: &[C64], noise_var: f64, rng: &mut Rng) -> Vec<f64> {
        let mut pol_x = x.to_vec();
        let mut pol_y = vec![C64::new(0.0, 0.0); x.len()];
        if noise_var > 0.0 {
            add_noise(&mut pol_x, noise_var, rng);
            add_noise(&mut pol_y, noise_var, rng);
        }
        let (fx, fy) = (self.optical.filter(&pol_x), self.optical.filter(&pol_y));
        let power: Vec<C64> = fx.iter().zip(&fy).map(|(a, b)| C64::new(a.norm_sqr() + b.norm_sqr(), 0.0)).collect();
        self.electrical.filter(&power).iter().map(|v| v.re).collect()
    }
}

/// The NRZ field of `bits`: powers 1 and 10^(-2) (20 dB of extinction ratio).
fn field(bits: &[u8]) -> Vec<C64> {
    let p0 = db2lin(-EXTINCTION_DB);
    bits.iter().flat_map(|&b| std::iter::repeat_n(C64::new(if b == 1 { 1.0 } else { p0.sqrt() }, 0.0), SPS)).collect()
}

/// The mean energy per bit of the NRZ signal (the powers 1 and 10^-2, equally likely).
fn energy_per_bit() -> f64 {
    (1.0 + db2lin(-EXTINCTION_DB)) / 2.0 / BIT_RATE
}

/// One sample per bit from the sample `phase` on (the group delay of the two filters
/// exceeds a bit: the phase spans several), as many as the bits of `i` minus the delay.
fn sampled(i: &[f64], phase: usize) -> Vec<f64> {
    i.iter().skip(phase).step_by(SPS).take(i.len() / SPS - phase.div_ceil(SPS)).copied().collect()
}

/// The error rate of the best threshold on `y` (a sweep of 200 thresholds).
fn threshold_ber(y: &[f64], bits: &[u8]) -> f64 {
    let (lo, hi) = y.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    (1..200)
        .map(|k| {
            let th = lo + (hi - lo) * k as f64 / 200.0;
            y.iter().zip(bits).filter(|(v, b)| u8::from(**v > th) != **b).count() as f64 / y.len() as f64
        })
        .fold(f64::INFINITY, f64::min)
}

/// The sampling phase (over three bit periods, the group delay of the filters) of the
/// smallest error rate of a receiver.
fn best_phase(i: &[f64], ber: impl Fn(&[f64]) -> f64) -> usize {
    (0..3 * SPS).min_by(|&a, &b| ber(&sampled(i, a)).total_cmp(&ber(&sampled(i, b)))).unwrap()
}

/// The MLSD fitted on the noiseless samples `clean` of the bits, with the chi-square
/// metric of the moments of the dark samples `dark`.
fn fitted_mlsd(clean: &[f64], idx: &[usize], memory: usize) -> Mlse {
    Mlse::fit(&clean.iter().map(|&v| C64::new(v, 0.0)).collect::<Vec<_>>(), &idx[..clean.len()], 2, memory).unwrap()
}

fn chi_square_ber(mlse: &Mlse, y: &[f64], dark: &[f64], bits: &[u8]) -> f64 {
    let mean = dark.iter().sum::<f64>() / dark.len() as f64;
    let var = dark.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / dark.len() as f64;
    let r: Vec<C64> = y.iter().map(|&v| C64::new(v, 0.0)).collect();
    errors(&mlse.detect_with(&r, mlse.chi_square_metric(var / mean, mean * mean / var).unwrap()).unwrap(), bits)
}

fn errors(decided: &[usize], bits: &[u8]) -> f64 {
    decided.iter().zip(bits).filter(|(d, b)| **d != usize::from(**b)).count() as f64 / decided.len() as f64
}

#[test]
fn foggi_2006_fig_6_the_amplified_ook_receiver_without_dispersion() {
    let mut card = Card::load("foggi2006_fig6").unwrap();
    let rx = Receiver::new();
    let mut rng = Rng::new(2006);
    let n_bits = 1 << 19;
    let bits: Vec<u8> = (0..n_bits).map(|_| rng.bit()).collect();
    let idx: Vec<usize> = bits.iter().map(|&b| usize::from(b)).collect();
    let x = field(&bits);
    let noise_var_of = |ebn0_db: f64| energy_per_bit() / db2lin(ebn0_db) * FS;
    let short = 1 << 14;
    let fiber = FiberParams { n_spans: 1, alpha_db_km: 0.0, slope_ps_nm2_km: 0.0, ..FiberParams::default() };
    // the chain at the index gamma: the receivers at their own best sampling phase (the
    // threshold receiver's, and the chi-square MLSD's), chosen on the first 2^14 bits at
    // 14.4 dB; the MLSD tables from the noiseless samples at that phase
    for gamma in [0.0, 0.2, 0.3, 0.4] {
        let length_km = gamma / (2.0 * std::f64::consts::PI * fiber.beta2_s2_km().abs() * BIT_RATE * BIT_RATE);
        let signal = if gamma > 0.0 { linear_fiber_channel(&x, &fiber, length_km, FS) } else { x.clone() };
        let memory = if gamma > 0.0 { 4 } else { MEMORY };
        let probe = rx.photocurrent(&signal[..short * SPS], noise_var_of(14.4), &mut Rng::new(1));
        let dark_probe = rx.photocurrent(&vec![C64::new(0.0, 0.0); short * SPS], noise_var_of(14.4), &mut Rng::new(2));
        let clean = rx.photocurrent(&signal[..short * SPS], 0.0, &mut rng);
        let phase_th = best_phase(&probe, |y| threshold_ber(y, &bits));
        let mlse_at = |phase: usize| fitted_mlsd(&sampled(&clean, phase), &idx, memory);
        // the MLSD phase: the smallest chi-square error rate with its table refitted per phase
        let phase_ml = (0..3 * SPS)
            .min_by(|&a, &b| {
                let f =
                    |ph: usize| chi_square_ber(&mlse_at(ph), &sampled(&probe, ph), &sampled(&dark_probe, ph), &bits);
                f(a).total_cmp(&f(b))
            })
            .unwrap();
        let mlse = mlse_at(phase_ml);
        let points: &[f64] = if gamma == 0.0 { &[10.4, 12.6, 14.4] } else { &[14.4] };
        for &ebn0_db in points {
            let noise_var = noise_var_of(ebn0_db);
            let i = rx.photocurrent(&signal, noise_var, &mut rng);
            let dark = rx.photocurrent(&vec![C64::new(0.0, 0.0); short * SPS], noise_var, &mut rng);
            let ber_th = threshold_ber(&sampled(&i, phase_th), &bits);
            let y = sampled(&i, phase_ml);
            let ber_chi = chi_square_ber(&mlse, &y, &sampled(&dark, phase_ml), &bits);
            let r: Vec<C64> = y.iter().map(|&v| C64::new(v, 0.0)).collect();
            let ber_gauss = errors(&mlse.detect(&r).unwrap(), &bits);
            println!(
                "gamma {gamma} ({length_km:.1} km), Eb/N0 {ebn0_db} dB (phases {phase_th}, {phase_ml}): threshold {ber_th:.2e}, MLSD chi-square {ber_chi:.2e}, Euclidean {ber_gauss:.2e}"
            );
            if gamma == 0.0 {
                card.check(&json!({"receiver": "threshold", "ebn0_db": ebn0_db}), ber_th);
                let inputs = json!({"receiver": "mlsd_chi_square", "ebn0_db": ebn0_db});
                if card.expected(&inputs).is_some() {
                    card.check(&inputs, ber_chi);
                }
            }
        }
    }
    card.finish().unwrap();
}
