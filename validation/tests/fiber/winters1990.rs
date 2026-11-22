//! Electrical equalisation of the chromatic dispersion of a direct-detection NRZ link
//! against J. H. Winters and R. D. Gitlin, "Electrical signal processing techniques in
//! long-haul, fiber-optic systems," *Proc. IEEE ICC*, 397–403, 1990, Fig. 8: the
//! optical power penalty of the worst-case eye opening (eqs. (7)–(8), no noise) versus
//! the distance, for no equalisation, a 6-tap synchronous linear equaliser (LE, with the
//! noise enhancement of eq. (9)), nonlinear cancellation (NLC, eqs. (10)–(12): a
//! feedforward tap on the next sample and a look-up table of the previous decided bits,
//! a decision feedback equaliser) and maximum-likelihood detection (MLD, eq. (17): the
//! minimum Euclidean distance between the stored signal vectors of two patterns that
//! differ in the decided bit, `Mlse::min_distance`), all with $N = 6$ bits.
//!
//! The link of the paper: 8 Gb/s NRZ, external modulation (no chirp) through a
//! first-order RC transmit filter of 32 GHz, chromatic dispersion, direct detection,
//! a 3-pole Butterworth receiver filter, a 64-bit sequence holding every 6-bit pattern
//! (a de Bruijn sequence here). The paper gives neither the dispersion nor the
//! wavelength nor the bandwidth of the receiver filter: here $D$ = 17 ps/(nm km) at
//! 1552 nm (`FiberParams::default`) and a Butterworth of the bit rate ("nearly ideal"),
//! and the distance axis is calibrated on the unequalised curve: every receiver is
//! measured at the distance of this link whose unequalised penalty is the one the
//! figure shows at 100, 150, 200 and 250 km (1.5, 3.2, 5.7 and 10 dB), which absorbs
//! the unknown product of the dispersion and the square of the bit rate. The sampling
//! instant is chosen per receiver and distance for the widest eye, as the paper
//! samples "at the peak opening of the eye". The LE taps are the least squares of the
//! samples on the ideal levels (the fixed point of the paper's zero-forcing algorithm
//! with an open eye) over the best split of precursor and postcursor taps; the NLC has
//! one future tap and 4 previous bits (the paper's "at most 1 future bit"); the MLD
//! has 2 previous bits and a window of 4 samples (the split of its $N$ = 6 is not
//! given). The DFE of comnum (`Dfe`: 2 feedforward taps, 4 linear feedback taps, NLMS
//! on the known bits) is measured on the same samples and compared with the NLC
//! curve. The values and tolerances are in the card
//! `validation/cards/winters1990_fig8.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::C64;
use comnum::channel::optical::fiber::{FiberParams, linear_fiber_channel};
use comnum::dsp::adaptive::UpdateRule;
use comnum::dsp::equalizer::dfe::Dfe;
use comnum::dsp::equalizer::mlse::Mlse;
use comnum::dsp::filter::Iir;
use comnum::math::linalg::lstsq;
use comnum::metrics::eye::eye_openings;
use comnum_validation::card::Card;
use ndarray::Array2;
use serde_json::json;

const BIT_RATE: f64 = 8e9;
const SPS: usize = 16;
const FS: f64 = BIT_RATE * SPS as f64;
const ORDER: usize = 6;
const N_BITS: usize = 1 << ORDER;
const RX_BANDWIDTH: f64 = BIT_RATE;
/// Previous bits of the NLC table and of the MLD state; the MLD window is the other 6 - N1 - 1 bits.
const N1: usize = 4;
const N1_MLD: usize = 2;

/// The binary de Bruijn sequence of order `ORDER` (every pattern once, the greedy
/// "prefer one" construction).
fn de_bruijn() -> Vec<usize> {
    let mut seq = vec![0usize; ORDER];
    let mut seen = [false; N_BITS];
    seen[0] = true;
    while seq.len() < N_BITS + ORDER {
        let base = seq[seq.len() - ORDER + 1..].iter().fold(0, |p, &b| (p << 1) | b);
        let bit = usize::from(!seen[(base << 1) | 1]);
        seen[(base << 1) | bit] = true;
        seq.push(bit);
    }
    seq[ORDER..].to_vec()
}

/// A filter from rest applied to a periodic signal: three periods, the last kept.
fn periodic(filter: &Iir, x: &[C64]) -> Vec<C64> {
    let n = x.len();
    let y = filter.filter(&[x, x, x].concat());
    y[2 * n..].to_vec()
}

/// The photocurrent of the link over `length_km`, one period of the sequence.
fn photocurrent(bits: &[usize], length_km: f64) -> Vec<f64> {
    let drive: Vec<C64> = bits.iter().flat_map(|&b| std::iter::repeat_n(C64::new(b as f64, 0.0), SPS)).collect();
    let tx = Iir::butterworth(1, 32e9, FS).unwrap();
    let field: Vec<C64> = periodic(&tx, &drive).iter().map(|p| C64::new(p.re.max(0.0).sqrt(), 0.0)).collect();
    let fiber = FiberParams {
        n_spans: 1,
        span_km: length_km.max(1e-9),
        alpha_db_km: 0.0,
        slope_ps_nm2_km: 0.0,
        ..FiberParams::default()
    };
    let out = if length_km > 0.0 { linear_fiber_channel(&field, &fiber, length_km, FS) } else { field };
    let power: Vec<C64> = out.iter().map(|e| C64::new(e.norm_sqr(), 0.0)).collect();
    let rx = Iir::butterworth(3, RX_BANDWIDTH, FS).unwrap();
    periodic(&rx, &power).iter().map(|v| v.re).collect()
}

/// The samples of bit `k` at the phase `phase` (any delay: the sequence is periodic).
fn sampled(i: &[f64], phase: usize) -> Vec<f64> {
    (0..N_BITS).map(|k| i[(k * SPS + phase) % i.len()]).collect()
}

/// The eye of a receiver, maximised over the sampling phase (two bit periods).
fn best_eye(i: &[f64], receiver: impl Fn(&[f64]) -> f64) -> f64 {
    (0..2 * SPS).map(|phase| receiver(&sampled(i, phase))).fold(f64::NEG_INFINITY, f64::max).max(0.0)
}

fn eye(y: &[f64], bits: &[usize]) -> f64 {
    eye_openings(y, bits, 2).unwrap()[0]
}

/// Least squares of `y_k` on the regressors `rows[k]` against the levels `bits`.
fn fit(rows: &[Vec<f64>], bits: &[usize]) -> Vec<f64> {
    let a = Array2::from_shape_vec((rows.len(), rows[0].len()), rows.concat()).unwrap();
    let b: Vec<f64> = bits.iter().map(|&v| v as f64).collect();
    lstsq(&a.view(), &b).unwrap()
}

/// The LE: 6 synchronous taps, `pre` of them on future samples, by least squares;
/// the eye divided by the noise enhancement sqrt(Σ c²) (eq. (9)).
fn linear_equalizer(y: &[f64], bits: &[usize], pre: usize) -> f64 {
    let n = y.len();
    let rows: Vec<Vec<f64>> = (0..n).map(|k| (0..6).map(|j| y[(k + n + pre - j) % n]).collect()).collect();
    let c = fit(&rows, bits);
    let out: Vec<f64> = rows.iter().map(|r| r.iter().zip(&c).map(|(a, b)| a * b).sum()).collect();
    eye(&out, bits) / c.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// The NLC: the current and the next sample, and a look-up table of the `N1` previous
/// bits (eqs. (10)–(12)), by least squares; the eye of its output.
fn nonlinear_cancellation(y: &[f64], bits: &[usize]) -> f64 {
    let n = y.len();
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|k| {
            let past = (1..=N1).fold(0, |p, j| (p << 1) | bits[(k + n - j) % n]);
            let mut r = vec![y[k], y[(k + 1) % n]];
            r.extend((0..1 << N1).map(|t| if t == past { 1.0 } else { 0.0 }));
            r
        })
        .collect();
    let c = fit(&rows, bits);
    let out: Vec<f64> = rows.iter().map(|r| r.iter().zip(&c).map(|(a, b)| a * b).sum()).collect();
    eye(&out, bits)
}

/// The MLD: the stored vectors of `N1_MLD` previous bits over a window of the other
/// bits of `N` = 6 (eq. (17)): the minimum distance of the table-driven detector.
fn maximum_likelihood(y: &[f64], bits: &[usize]) -> f64 {
    let r: Vec<C64> = y.iter().map(|&v| C64::new(v, 0.0)).collect();
    // the table needs the past of every sample: two periods, the first as the past
    let rr = [r.clone(), r].concat();
    let bb = [bits.to_vec(), bits.to_vec()].concat();
    let n2 = 6 - 1 - N1_MLD;
    Mlse::fit(&rr, &bb, 2, N1_MLD + n2).unwrap().min_distance(n2 + 1).unwrap()
}

/// comnum's DFE on the samples shifted by one (the window holds the current and the
/// next sample), 4 feedback taps, NLMS on the known bits over 40 periods: the eye of
/// its output on the last period.
fn dfe(y: &[f64], bits: &[usize]) -> f64 {
    let n = y.len();
    let periods = 40;
    let x: Vec<C64> = (0..periods * n).map(|k| C64::new(y[(k + 1) % n], 0.0)).collect();
    let reference: Vec<C64> = (0..periods * n).map(|k| C64::new(bits[k % n] as f64, 0.0)).collect();
    let eq =
        Dfe { n_feedforward: 2, n_feedback: N1, sps: 1, mu: 0.05, rule: UpdateRule::Normalized, training: periods * n };
    let out = eq.run(&x, Some(&reference), &[C64::new(0.0, 0.0), C64::new(1.0, 0.0)]).unwrap();
    let last: Vec<f64> = out.symbols[(periods - 1) * n..].iter().take(n).map(|v| v.re).collect();
    eye(&last, bits)
}

/// The penalty of the unequalised receiver read on the figure at 100 to 250 km.
const PAPER_NONE_DB: [(f64, f64); 4] = [(100.0, 1.5), (150.0, 3.2), (200.0, 5.7), (250.0, 10.0)];

#[test]
fn winters_1990_fig_8_eye_penalties_of_the_equalisers_against_chromatic_dispersion() {
    let mut card = Card::load("winters1990_fig8").unwrap();
    let bits = de_bruijn();
    let i0 = photocurrent(&bits, 0.0);
    let y_ref = best_eye(&i0, |y| eye(y, &bits));
    let penalty = |e: f64| -10.0 * (e / y_ref).max(1e-9).log10();
    // the distance axis calibrated on the unequalised curve: the distance of the
    // simulation whose unequalised penalty is the one read at every distance of the paper
    let grid: Vec<(f64, f64)> = (0..=60)
        .map(|k| {
            let km = 5.0 * k as f64;
            (km, penalty(best_eye(&photocurrent(&bits, km), |y| eye(y, &bits))))
        })
        .collect();
    println!("km (paper) -> km (here): none / LE / NLC / MLD / comnum DFE (dB)");
    for (km_paper, target) in PAPER_NONE_DB {
        let (above, _) = grid.iter().enumerate().find(|(_, (_, p))| *p >= target).unwrap();
        let ((k1, p1), (k0, p0)) = (grid[above], grid[above - 1]);
        let km = k0 + (k1 - k0) * (target - p0) / (p1 - p0);
        let i = photocurrent(&bits, km);
        let none = best_eye(&i, |y| eye(y, &bits));
        let le = (0..6).map(|pre| best_eye(&i, |y| linear_equalizer(y, &bits, pre))).fold(0.0, f64::max);
        let nlc = best_eye(&i, |y| nonlinear_cancellation(y, &bits));
        let mld = best_eye(&i, |y| maximum_likelihood(y, &bits));
        let dfe = best_eye(&i, |y| dfe(y, &bits));
        let p = [none, le, nlc, mld, dfe].map(penalty);
        println!("{km_paper:>3} -> {km:>5.1}: {:.2} / {:.2} / {:.2} / {:.2} / {:.2}", p[0], p[1], p[2], p[3], p[4]);
        for (name, v) in ["none", "le", "nlc", "mld", "dfe"].iter().zip(p) {
            let inputs = json!({"receiver": name, "km": km_paper});
            if card.expected(&inputs).is_some() {
                card.check(&inputs, v);
            }
        }
    }
    card.finish().unwrap();
}
