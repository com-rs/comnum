//! The linear multiuser receivers of DS-CDMA with random signatures
//! (`comnum::waveform::dsss::{Dsss, theory}`) against D. N. C. Tse and S. V. Hanly,
//! "Linear multiuser receivers: effective interference, effective bandwidth and user
//! capacity," *IEEE Trans. Inf. Theory* 45(2), 641–657, 1999:
//!
//! - Fig. 1: the SIR of the MMSE receiver of user 1 over 100 draws of random $\pm 1$
//!   signatures, $N = 32$, 64 and 128 chips, $P/\sigma^{2} = 20$ dB, against the
//!   asymptotic limit of eq. (9) at the loads $\alpha = K/N$ of 0.25, 0.5, 1 and 1.5: the
//!   mean (in dB) of the draws and their standard deviation, which shrinks with $N$;
//! - Fig. 2: the SIRs of the 80 users of one draw at $N = 128$, around the limit;
//! - Proposition 3.3 and Theorem 7.2: the same comparison for the conventional receiver
//!   and for the decorrelator at $N = 128$.
//!
//! The SIRs are those of `Dsss::sinr`, the output of the linear detectors of
//! `comnum::mimo::detection` on the signature matrix; the limits are
//! `theory::sinr_large_system`. The expected values (the limits) and the tolerances are
//! in the card `validation/cards/tse1999_fig1.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::core::rng::Rng;
use comnum::mapping::constellation::Constellation;
use comnum::math::lin2db;
use comnum::mimo::detection::{Criterion, Linear};
use comnum::waveform::dsss::Dsss;
use comnum_validation::card::Card;
use serde_json::json;

/// The noise variance of $P/\sigma^{2} = 20$ dB.
const NOISE_VAR: f64 = 0.01;

/// A link of `k` users with random binary signatures of `n` chips.
fn random_link(k: usize, n: usize, rng: &mut Rng) -> Dsss {
    let codes: Vec<Vec<u8>> = (0..k).map(|_| (0..n).map(|_| rng.bit()).collect()).collect();
    Dsss::new(&codes).unwrap()
}

/// The mean and the standard deviation (dB) of the SIR of user 1 over `draws` links.
fn sir_of_user_1(linear: &Linear, k: usize, n: usize, draws: usize, rng: &mut Rng) -> (f64, f64) {
    let db: Vec<f64> = (0..draws).map(|_| lin2db(random_link(k, n, rng).sinr(linear, NOISE_VAR).unwrap()[0])).collect();
    let mean = db.iter().sum::<f64>() / draws as f64;
    let var = db.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / draws as f64;
    (mean, var.sqrt())
}

#[test]
fn tse_hanly_1999_figs_1_and_2_the_sir_of_random_signatures_tends_to_its_limit() {
    let mut card = Card::load("tse1999_fig1").unwrap();
    let mut rng = Rng::new(1999);
    let bpsk = Constellation::psk(2).unwrap();
    let mmse = Linear { constellation: bpsk.clone(), criterion: Criterion::Mmse };
    let mut spread = Vec::new();
    for n in [32usize, 64, 128] {
        for alpha in [0.25, 0.5, 1.0, 1.5] {
            let k = (alpha * n as f64) as usize;
            let (mean, std) = sir_of_user_1(&mmse, k, n, 100, &mut rng);
            println!("MMSE, N = {n}, alpha = {alpha}: {mean:.2} dB, standard deviation {std:.2} dB");
            card.check(&json!({"figure": 1, "receiver": "mmse", "n": n, "alpha": alpha}), mean);
            spread.push((n, alpha, std));
        }
    }
    // the spread around the limit shrinks with the spreading factor at every load
    for alpha in [0.25, 0.5, 1.0, 1.5] {
        let at = |n: usize| spread.iter().find(|s| s.0 == n && s.1 == alpha).unwrap().2;
        assert!(at(128) < at(64) && at(64) < at(32), "alpha {alpha}");
        card.check(&json!({"figure": 1, "quantity": "standard deviation at N = 128 (dB)", "alpha": alpha}), at(128));
    }
    // Fig. 2: the 80 users of one draw at N = 128
    let sir: Vec<f64> =
        random_link(80, 128, &mut rng).sinr(&mmse, NOISE_VAR).unwrap().into_iter().map(lin2db).collect();
    let mean = sir.iter().sum::<f64>() / 80.0;
    let (low, high) = sir.iter().fold((f64::MAX, f64::MIN), |(l, h), v| (l.min(*v), h.max(*v)));
    println!("Fig. 2: mean {mean:.2} dB, from {low:.2} to {high:.2} dB");
    card.check(&json!({"figure": 2, "quantity": "mean over the 80 users (dB)"}), mean);
    card.check(&json!({"figure": 2, "quantity": "lowest of the 80 users (dB)"}), low);
    card.check(&json!({"figure": 2, "quantity": "highest of the 80 users (dB)"}), high);
    // the conventional receiver and the decorrelator at N = 128
    for (name, criterion) in [("mr", Criterion::Mr), ("zf", Criterion::Zf)] {
        let linear = Linear { constellation: bpsk.clone(), criterion };
        for alpha in [0.25, 0.5] {
            let (mean, _) = sir_of_user_1(&linear, (alpha * 128.0) as usize, 128, 100, &mut rng);
            card.check(&json!({"figure": 1, "receiver": name, "n": 128, "alpha": alpha}), mean);
        }
    }
    card.finish().unwrap();
}
