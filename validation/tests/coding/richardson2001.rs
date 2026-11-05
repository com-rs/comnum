//! Density evolution of regular LDPC ensembles (`comnum::fec::density_evolution`) against
//! T. J. Richardson and R. L. Urbanke, "The capacity of low-density parity-check codes
//! under message-passing decoding," *IEEE Trans. Inf. Theory* 47(2), 599–618, 2001.
//!
//! - Table I: the thresholds $\epsilon^{*}$ of the BSC for the six $(d_{v}, d_{c})$
//!   ensembles of rates 1/4 to 1/2 under Gallager's algorithms A (eq. (6)) and B
//!   (Example 4), the ternary algorithm E (Example 5, the weights maximising eq. (7))
//!   and belief propagation (Example 10), and the column $\epsilon_{\mathrm{opt}}$ at
//!   which the capacity (3) equals the rate;
//! - Table II: the thresholds $\sigma^{*}$ of the binary-input AWGN channel under belief
//!   propagation (Example 11) and the column $\sigma_{\mathrm{opt}}$ of the capacity (4).
//!
//! The hard-decision thresholds are scalar recursions, bisected to $10^{-6}$; the
//! capacity columns are closed forms. Belief propagation runs discretized density
//! evolution (LLR step 0.05, LLRs to 25, the check node pair by pair; the paper computes
//! the same densities with the transform of eq. (8)) and bisects the threshold to
//! $10^{-4}$: about five minutes, a campaign. The tests measure; the printed values, their
//! precision and the findings are in the cards `validation/cards/richardson2001_table1.yaml`
//! and `richardson2001_bp.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::fec::bounds::biawgn_capacity;
use comnum::fec::density_evolution::{
    BinaryInputChannel, BpDensityEvolution, HardDecoder, RegularEnsemble, hard_decision_threshold,
};
use comnum::math::{bisect, entropy};
use comnum_validation::card::Card;
use serde_json::json;

/// The six ensembles of Tables I-III, in the order of the tables.
const ENSEMBLES: [(usize, usize); 6] = [(3, 6), (4, 8), (5, 10), (3, 5), (4, 6), (3, 4)];

#[test]
fn richardson_urbanke_2001_table_1_hard_decisions_and_capacities() {
    let mut card = Card::load("richardson2001_table1").unwrap();
    for (dv, dc) in ENSEMBLES {
        let e = RegularEnsemble::new(dv, dc).unwrap();
        for (name, d) in [("A", HardDecoder::GallagerA), ("B", HardDecoder::GallagerB), ("E", HardDecoder::AlgorithmE)]
        {
            let t = hard_decision_threshold(&e, d).unwrap();
            card.check(&json!({"table": "I", "decoder": name, "dv": dv, "dc": dc}), t);
        }
        let r = e.rate();
        // eq. (3): 1 - h(eps) = r; eq. (4): C_BIAWGN(sigma) = r
        let eps_opt = bisect(|x| 1.0 - entropy(&[x, 1.0 - x]) > r, 1e-9, 0.5, 60);
        let sigma_opt = bisect(|s| biawgn_capacity(s * s) > r, 0.1, 5.0, 60);
        card.check(&json!({"table": "I", "quantity": "eps_opt", "dv": dv, "dc": dc}), eps_opt);
        card.check(&json!({"table": "II", "quantity": "sigma_opt", "dv": dv, "dc": dc}), sigma_opt);
    }
    card.finish().unwrap();
}

#[test]
#[ignore = "statistical validation: run with --ignored (about five minutes)"]
fn richardson_urbanke_2001_tables_1_and_2_belief_propagation() {
    let mut card = Card::load("richardson2001_bp").unwrap();
    let bp = BpDensityEvolution::default();
    let printed = [(0.88, 0.084), (0.83, 0.076), (0.79, 0.068), (1.0, 0.113), (1.01, 0.116), (1.26, 0.167)];
    for ((dv, dc), (s0, e0)) in ENSEMBLES.into_iter().zip(printed) {
        let e = RegularEnsemble::new(dv, dc).unwrap();
        let sigma = bp.threshold(&e, |s| BinaryInputChannel::Biawgn { sigma: s }, s0 - 0.04, s0 + 0.04, 10).unwrap();
        let eps = bp.threshold(&e, |x| BinaryInputChannel::Bsc { crossover: x }, e0 - 0.02, e0 + 0.02, 10).unwrap();
        println!("({dv}, {dc}): sigma* {sigma:.4}, eps* {eps:.4}");
        card.check(&json!({"table": "II", "channel": "biawgn", "dv": dv, "dc": dc}), sigma);
        card.check(&json!({"table": "I", "channel": "bsc", "dv": dv, "dc": dc}), eps);
    }
    card.finish().unwrap();
}
