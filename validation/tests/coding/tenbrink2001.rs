//! EXIT charts of parallel concatenated codes (`comnum::fec::exit`) against S. ten Brink,
//! "Convergence behavior of iteratively decoded parallel concatenated codes," *IEEE
//! Trans. Commun.* 49(10), 1727–1737, 2001:
//!
//! - Table II: the pinch-off of the rate-1/2 PCC of the (037, 021) code (the turbo code of
//!   Berrou et al.), 0.53 dB by the EXIT chart, 0.537 dB by density evolution;
//! - Table IV: the pinch-offs of the rate-1/2 PCCs of the recursive codes of memory 1 to 6
//!   (C1 = (03, 02) to C6 = (0147, 0117)), the diagonal (symmetric PCCs) and the pairs
//!   with C2 = (07, 05).
//!
//! Each constituent is a rate-1/2 recursive systematic code with its parity punctured to
//! rate 2/3 (one parity bit in two), BPSK on AWGN at the $E_{b}/N_{0}$ of the rate-1/2 PCC,
//! log-MAP BCJR, Gaussian a priori LLRs of $I_{A} = J(\sigma_{A})$; the extrinsic information
//! by the time average of the LLRs of $2 \cdot 10^{5}$ bits; the pinch-off by bisection to
//! 0.005 dB, every curve on the same random numbers. The paper does not give its puncturing
//! pattern (alternate parities assumed) nor its search. The values and tolerances are in
//! `validation/cards/tenbrink2001_pinch_off.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::fec::conv::Trellis;
use comnum::fec::exit::pinch_off_db;
use comnum_validation::card::Card;
use serde_json::json;

/// The codes of Table IV: (name, constraint length, feedback, forward), octal.
const CODES: [(&str, u32, u32, u32); 6] = [
    ("C1", 2, 0o3, 0o2),
    ("C2", 3, 0o7, 0o5),
    ("C3", 4, 0o13, 0o15),
    ("C4", 5, 0o23, 0o37),
    ("C5", 6, 0o67, 0o45),
    ("C6", 7, 0o147, 0o117),
];

fn trellis(name: &str) -> Trellis {
    let (_, k, fb, fw) = CODES.iter().find(|c| c.0 == name).copied().unwrap();
    Trellis::recursive(k, fb, fw).unwrap()
}

#[test]
#[ignore = "statistical validation: run with --ignored (a few minutes)"]
fn ten_brink_2001_pinch_off_limits() {
    let mut card = Card::load("tenbrink2001_pinch_off").unwrap();
    let n = 200_000;
    let t = std::time::Instant::now();
    let berrou = Trellis::recursive(5, 0o37, 0o21).unwrap();
    let p = pinch_off_db(&berrou, &berrou, 2, 0.5, n, 2001, (0.0, 1.5), 9, 0.97).unwrap();
    println!("(037, 021): {p:.3} dB ({:.0} s)", t.elapsed().as_secs_f64());
    card.check(&json!({"table": "II", "code": "037_021"}), p);
    for inputs in card.inputs_matching(&json!({"table": "IV"})) {
        let (a, b) = (inputs["first"].as_str().unwrap(), inputs["second"].as_str().unwrap());
        let t = std::time::Instant::now();
        let p = pinch_off_db(&trellis(a), &trellis(b), 2, 0.5, n, 2001, (0.0, 2.5), 9, 0.97).unwrap();
        println!("{a}/{b}: {p:.3} dB ({:.0} s)", t.elapsed().as_secs_f64());
        card.check(&inputs, p);
    }
    card.finish().unwrap();
}
