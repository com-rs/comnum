//! The blind phase search (`comnum_dsp::sync::carrier::bps`) with the differential quadrant
//! code of square QAM (`comnum_mapping::differential::QuadrantQam`) against T. Pfau,
//! S. Hoffmann and R. Noé, "Hardware-efficient coherent digital receiver concept with
//! feedforward carrier recovery for M-QAM constellations," *J. Lightw. Technol.* 27(8),
//! 2009, Sec. V.
//!
//! The link of the paper: unit-energy square QAM whose quadrant is differentially
//! encoded, Wiener phase noise of increments of variance $2\pi \Delta f T_{s}$ (eq. (2)),
//! white Gaussian noise, the blind phase search over $2N + 1 = 19$ symbols with $B = 32$
//! test phases for 4- and 16-QAM and $B = 64$ for 64- and 256-QAM, the phase unwrapped
//! with a period of $\pi/2$ (the quadrant jumps of eq. (12)), hard decisions and the
//! differential decoding of the quadrants. The penalty is the $E_{s}/N_{0}$ at a BER of
//! $10^{-3}$ above that of Gray QAM given by eq. (17) of the paper; it includes the cost
//! of the differential code.
//!
//! - Fig. 6: the penalty without phase noise, 0.7, 0.6, 0.5 and 0.55 dB.
//! - Table I: the linewidth times symbol duration of a 1 dB penalty, $4.1 \cdot 10^{-4}$,
//!   $1.4 \cdot 10^{-4}$, $4.0 \cdot 10^{-5}$ and $8.0 \cdot 10^{-6}$: the penalty measured
//!   there.
//!
//! The paper simulates 200 000 symbols per point; here 10⁶. The values read and the
//! tolerances are in the card `validation/cards/pfau2009_table1.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::C64;
use comnum::channel::awgn::add_noise;
use comnum::core::rng::Rng;
use comnum::dsp::sync::carrier::{bps, derotate, unwrap};
use comnum::hardware::converter::Quantizer;
use comnum::hardware::oscillator::{PhaseOrigin, wiener_phase};
use comnum::mapping::differential::QuadrantQam;
use comnum::mapping::source;
use comnum::math::{cis, db2lin, lin2db, qfuncinv};
use comnum_validation::card::Card;
use comnum_validation::sweep_crossing;
use serde_json::json;
use std::f64::consts::{FRAC_PI_2, PI};

const SYMBOLS: usize = 1_000_000;
const HALF_WIDTH: usize = 9;

/// The $E_{s}/N_{0}$ (dB) at which Gray square QAM reaches `ber`, eq. (17) of the paper.
fn theory_esn0_db(m: usize, ber: f64) -> f64 {
    let (mf, k) = (m as f64, (m as f64).log2());
    let q = qfuncinv(k / 2.0 / (1.0 - 1.0 / mf.sqrt()) * (1.0 - (1.0 - ber).sqrt()));
    lin2db((mf - 1.0) / 3.0 * q * q)
}

/// The number of test phases of the paper.
fn test_phases(m: usize) -> usize {
    if m <= 16 { 32 } else { 64 }
}

/// The receiver of the penalty.
#[derive(Clone, Copy, Debug)]
enum Receiver {
    /// The blind phase search with `B` test phases (Sec. III-A).
    Search,
    /// The carrier phase known exactly and rounded to the grid of `B` test phases: the
    /// "ideal carrier recovery" of Fig. 6, whose penalty is the differential quadrant
    /// code and the phase quantisation alone (Sec. V-A).
    Ideal,
}

/// One link of the paper: the constellation, the phase noise, the receiver and, for
/// Fig. 10, a converter of `bits` bits and full scale `full_scale` times the outermost
/// coordinate of the constellation on the in-phase and quadrature components.
#[derive(Clone, Copy, Debug)]
struct Link {
    m: usize,
    dfts: f64,
    b: usize,
    receiver: Receiver,
    adc: Option<(u32, f64)>,
}

impl Link {
    /// The link of Table I: the search with the number of test phases of the paper.
    fn paper(m: usize, dfts: f64) -> Link {
        Link { m, dfts, b: test_phases(m), receiver: Receiver::Search, adc: None }
    }

    /// The BER at `esn0_db`.
    fn ber(&self, esn0_db: f64, rng: &mut Rng) -> f64 {
        let qq = QuadrantQam::new(self.m).unwrap();
        let points = qq.points();
        let bits = source::bits(SYMBOLS * qq.bits_per_symbol(), rng);
        let x = qq.map_bits(&bits).unwrap();
        // the carrier phase starts anywhere: without phase noise it would otherwise sit
        // on the grid of the test phases and the quantisation of the phase cost nothing
        let phi0 = 2.0 * PI * rng.uniform();
        let mut phase = wiener_phase((2.0 * PI * self.dfts).sqrt(), x.len(), PhaseOrigin::Zero, rng);
        for p in &mut phase {
            *p += phi0;
        }
        let mut r: Vec<C64> = x.iter().zip(&phase).map(|(v, p)| v * cis(*p)).collect();
        add_noise(&mut r, 1.0 / db2lin(esn0_db), rng);
        if let Some((bits, full_scale)) = self.adc {
            let outer = points.iter().map(|p| p.re.abs()).fold(0.0, f64::max);
            Quantizer::new(bits, full_scale * outer).unwrap().apply(&mut r);
        }
        let grid = FRAC_PI_2 / self.b as f64;
        let derotated = match self.receiver {
            Receiver::Search => {
                let theta = unwrap(&bps(&r, HALF_WIDTH, &points, self.b).unwrap(), FRAC_PI_2);
                derotate(&r, &theta, 1.0).unwrap()
            }
            Receiver::Ideal => {
                let theta: Vec<f64> = phase.iter().map(|p| (p / grid).round() * grid).collect();
                derotate(&r, &theta, -1.0).unwrap()
            }
        };
        let decided = qq.demap_bits(&derotated).unwrap();
        let errors = decided.iter().zip(&bits).filter(|(a, b)| a != b).count();
        errors as f64 / bits.len() as f64
    }

    /// The penalty (dB) at a BER of $10^{-3}$: the $E_{s}/N_{0}$ of the crossing,
    /// interpolated between points 0.2 dB apart, above eq. (17).
    fn penalty(&self, rng: &mut Rng) -> f64 {
        let theory = theory_esn0_db(self.m, 1e-3);
        let crossing = sweep_crossing(theory, 0.2, 20, 1e-3, |_, esn0| self.ber(esn0, rng)).unwrap();
        crossing - theory
    }
}

/// The penalty (dB) of the link of Table I at a BER of $10^{-3}$.
fn penalty(m: usize, dfts: f64, rng: &mut Rng) -> f64 {
    Link::paper(m, dfts).penalty(rng)
}

#[test]
fn eq_17_is_the_gray_ber_of_square_qam() {
    // QPSK: BER = Q(√(Es/N0)) up to 2(1 - √(1 - BER)) ≈ BER, so 10⁻³ at Q⁻¹(10⁻³)²
    assert!((theory_esn0_db(4, 1e-3) - lin2db(qfuncinv(1e-3).powi(2))).abs() < 1e-3);
    assert!((theory_esn0_db(16, 1e-3) - 16.54).abs() < 0.01);
}

#[test]
#[ignore = "validation against the literature: run with --release --ignored"]
fn the_linewidth_tolerance_follows_table_i_of_pfau_2009() {
    let mut card = Card::load("pfau2009_table1").unwrap();
    let mut rng = Rng::new(2009);
    for (m, table) in [(4, 4.1e-4), (16, 1.4e-4), (64, 4.0e-5), (256, 8.0e-6)] {
        let (p0, p1) = (penalty(m, 0.0, &mut rng), penalty(m, table, &mut rng));
        println!("{m}-QAM: penalty {p0:.2} dB without phase noise, {p1:.2} dB at {table:e}");
        card.check(&json!({"m": m, "quantity": "penalty_without_phase_noise"}), p0);
        card.check(&json!({"m": m, "quantity": "penalty_at_table_i"}), p1);
    }
    card.finish().unwrap();
}

#[test]
#[ignore = "validation against the literature: run with --release --ignored"]
fn the_penalty_against_the_test_phases_follows_fig_6_of_pfau_2009() {
    let mut card = Card::load("pfau2009_fig6").unwrap();
    let mut rng = Rng::new(2009);
    for (m, log2_b) in [(4, 3), (4, 4), (4, 5), (4, 6), (4, 8), (16, 4), (16, 5), (16, 6), (16, 8)].into_iter().chain([
        (64, 4),
        (64, 5),
        (64, 6),
        (64, 8),
        (256, 5),
        (256, 6),
        (256, 7),
        (256, 8),
    ]) {
        for (receiver, name) in [(Receiver::Search, "search"), (Receiver::Ideal, "ideal")] {
            let link = Link { b: 1 << log2_b, receiver, ..Link::paper(m, 0.0) };
            let p = link.penalty(&mut rng);
            println!("{m}-QAM, log2 B = {log2_b}, {name}: penalty {p:.2} dB");
            card.check(&json!({"m": m, "log2_b": log2_b, "receiver": name}), p);
        }
    }
    card.finish().unwrap();
}

#[test]
#[ignore = "validation against the literature: run with --release --ignored"]
fn the_penalty_against_the_linewidth_follows_fig_7_of_pfau_2009() {
    let mut card = Card::load("pfau2009_fig7").unwrap();
    let mut rng = Rng::new(2009);
    let points: [(usize, &[f64]); 4] = [
        (4, &[1e-6, 1e-5, 3e-5, 1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3]),
        (16, &[1e-6, 1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 1.4e-4, 2e-4, 3e-4]),
        (64, &[1e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 4e-5, 6e-5, 8e-5]),
        (256, &[1e-6, 2e-6, 3e-6, 5e-6, 7e-6, 1e-5, 1.5e-5]),
    ];
    for (m, dfts) in points {
        for &d in dfts {
            let p = Link::paper(m, d).penalty(&mut rng);
            println!("{m}-QAM, df.Ts = {d:e}: penalty {p:.2} dB");
            card.check(&json!({"m": m, "dfts": d}), p);
        }
    }
    card.finish().unwrap();
}
