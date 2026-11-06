//! Gallager codes under sum-product decoding (`comnum::fec::ldpc`,
//! `comnum::fec::density_evolution`) against D. J. C. MacKay, "Good error-correcting codes
//! based on very sparse matrices," *IEEE Trans. Inf. Theory* 45(2), 399–431, 1999:
//!
//! - Fig. 6(b): the zero in $f$ of the correction effect (57) of the simple decoder, the
//!   dotted curves, at rates 1/2 ($t = 3$ to 6) and 1/4 ($t = 3$);
//! - Fig. 6: the transition of belief propagation on the infinite tree, about 0.075 for
//!   $t = 4$, $t_{r} = 8$ (caption of Fig. 6(a)), and between the last decoded marker (0.08)
//!   and the first failure (0.09) for $t = 3$ at rate 1/2, by the discretized density
//!   evolution of `BpDensityEvolution` (the paper: a Monte Carlo of the tree);
//! - Fig. 11: the bit error rate of the (504, 252) and (1008, 504) codes on the Gaussian
//!   channel, with codes of column weight 3 built by progressive edge growth
//!   (`LdpcCode::peg`: the paper's construction 1A is not reproduced, PEG avoids short
//!   cycles), flooding sum-product, at most 500 iterations, until 100 frame errors.
//!
//! The values read on the figures, the tolerances and the findings are in
//! `validation/cards/mackay1999.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::fec::bounds::bpsk_noise_variance;
use comnum::fec::density_evolution::{
    BinaryInputChannel, BpDensityEvolution, RegularEnsemble, correction_effect_threshold,
};
use comnum::fec::ldpc::{CheckRule, LdpcCode, LdpcDecoder};
use comnum::prelude::*;
use comnum_validation::card::Card;
use comnum_validation::{MIN_ERRORS, coded_frame, until_errors_with};
use serde_json::json;

#[test]
#[ignore = "statistical validation: run with --ignored (a few minutes)"]
fn mackay_1999_figs_6_and_11() {
    let mut card = Card::load("mackay1999").unwrap();
    for (t, tr) in [(3, 6), (4, 8), (5, 10), (6, 12), (3, 4)] {
        let f = correction_effect_threshold(&RegularEnsemble::new(t, tr).unwrap());
        card.check(&json!({"figure": "6b", "t": t, "tr": tr}), f);
    }
    let bp = BpDensityEvolution::default();
    for (t, tr, lo, hi) in [(4, 8, 0.06, 0.09), (3, 6, 0.07, 0.10)] {
        let e = RegularEnsemble::new(t, tr).unwrap();
        let f = bp.threshold(&e, |x| BinaryInputChannel::Bsc { crossover: x }, lo, hi, 8).unwrap();
        card.check(&json!({"figure": "6", "t": t, "tr": tr}), f);
    }
    let mut rng = Rng::new(1999);
    for (n, k) in [(504, 252), (1008, 504)] {
        let mut code = LdpcCode::peg(n, k, 3, 1).unwrap();
        code.decoder = LdpcDecoder { rule: CheckRule::SumProduct, layered: false, max_iterations: 500 };
        for inputs in card.inputs_matching(&json!({"figure": 11, "n": n})) {
            let ebn0 = inputs["ebn0_db"].as_f64().unwrap();
            let noise_var = bpsk_noise_variance(ebn0, 0.5);
            // until 100 frame errors (the rule), counting the bit errors of the same frames
            let mut bits = (0u64, 0u64);
            let count = until_errors_with(&mut rng, MIN_ERRORS, 200_000, |rng| {
                let e = coded_frame(&code, noise_var, rng)?;
                bits.0 += e.bit_errors;
                bits.1 += k as u64;
                Ok((u64::from(e.frame_error), 1))
            })
            .unwrap();
            let frames = count.trials;
            let ber = bits.0 as f64 / bits.1 as f64;
            println!("({n}, {k}) {ebn0} dB: BER {ber:.3e} over {frames} frames");
            card.check(&inputs, ber);
        }
    }
    card.finish().unwrap();
}
