//! Downlink NOMA with randomly deployed users (`comnum::mimo::noma`) against Z. Ding,
//! Z. Yang, P. Fan and H. V. Poor, "On the performance of non-orthogonal multiple access
//! in 5G systems with randomly deployed users," *IEEE Signal Process. Lett.* 21(12),
//! 1501–1505, 2014:
//!
//! - the table of Fig. 3 (outage at $\rho = 80$ dB, $\alpha = 3$, two users, $a = (4/5, 1/5)$,
//!   four cases of radius and rates): the "anal." rows against the high-SNR form (11)
//!   with $N = 10$ Gauss–Chebyshev nodes, the "simul." rows against the exact outage (9)
//!   and the conventional row against the orthogonal random user;
//! - Fig. 1(a): the outage of the two users and of the orthogonal user from 10 to 35 dB
//!   for (3 m, $\alpha = 2$) and (5 m, $\alpha = 3$), $\tilde R = (0.1, 0.5)$, by a Monte Carlo
//!   that drops the users, orders them and runs SIC on the rates of eq. (2), written
//!   here without the thresholds of the module ($10^{6}$ drops per point).
//!
//! The values of the paper, the tolerances and the findings are in
//! `validation/cards/ding2014_outage.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::mimo::noma::{NomaDownlink, RandomDisc, orthogonal_outage};
use comnum::prelude::*;
use comnum_validation::card::Card;
use serde_json::json;

/// Outages of the two NOMA users and of the orthogonal random user by Monte Carlo.
fn simulate(disc: &RandomDisc, snr: f64, a: [f64; 2], r: [f64; 2], drops: usize, rng: &mut Rng) -> [f64; 3] {
    let rate = |h: f64, a: f64, rest: f64| (1.0 + snr * h * a / (snr * h * rest + 1.0)).log2();
    let mut out = [0usize; 3];
    for _ in 0..drops {
        let mut g: Vec<f64> = (0..2)
            .map(|_| {
                let d = disc.radius * rng.uniform().sqrt();
                -rng.uniform().ln() / (1.0 + d.powf(disc.alpha))
            })
            .collect();
        // the orthogonal user: one user at random gets the sum rate
        if (1.0 + snr * g[0]).log2() < r[0] + r[1] {
            out[2] += 1;
        }
        g.sort_by(f64::total_cmp);
        // user 1 decodes its message with user 2's as noise
        if rate(g[0], a[0], a[1]) < r[0] {
            out[0] += 1;
        }
        // user 2 decodes user 1's message (SIC), then its own
        if rate(g[1], a[0], a[1]) < r[0] || rate(g[1], a[1], 0.0) < r[1] {
            out[1] += 1;
        }
    }
    out.map(|c| c as f64 / drops as f64)
}

#[test]
#[ignore = "statistical validation: run with --ignored (under 1 s)"]
fn ding_2014_fig_3_table_and_fig_1a() {
    let mut card = Card::load("ding2014_outage").unwrap();
    let a = vec![0.8, 0.2];
    let cases = [(1, 80.0, 0.1, 0.5), (2, 80.0, 0.2, 0.6), (3, 100.0, 0.1, 0.5), (4, 100.0, 0.2, 0.6)];
    for (case, radius, r1, r2) in cases {
        let disc = RandomDisc { radius, alpha: 3.0 };
        let noma = NomaDownlink::new(a.clone(), vec![r1, r2]).unwrap();
        let snr = 1e8;
        let exact = noma.outage(&disc, snr).unwrap();
        let high = noma.outage_high_snr(&disc, snr, 10).unwrap();
        for u in 0..2 {
            card.check(&json!({"figure": 3, "row": format!("user{}_anal", u + 1), "case": case}), high[u] * 1e4);
            card.check(&json!({"figure": 3, "row": format!("user{}_simul", u + 1), "case": case}), exact[u] * 1e4);
        }
        let conv = orthogonal_outage(&disc, snr, r1 + r2).unwrap();
        card.check(&json!({"figure": 3, "row": "conventional", "case": case}), conv * 1e4);
    }
    // the points of Fig. 1(a) that the card holds, one simulation per disc and SNR
    let mut rng = Rng::new(2014);
    let mut cache: Vec<((u64, u64, u64), [f64; 3])> = Vec::new();
    for inputs in card.inputs_matching(&json!({"figure": "1a"})) {
        let (radius, alpha, db) =
            (inputs["radius"].as_f64().unwrap(), inputs["alpha"].as_f64().unwrap(), inputs["snr_db"].as_f64().unwrap());
        let key = (radius.to_bits(), alpha.to_bits(), db.to_bits());
        let p = if let Some((_, p)) = cache.iter().find(|(k, _)| *k == key) {
            *p
        } else {
            let disc = RandomDisc { radius, alpha };
            let p = simulate(&disc, 10f64.powf(db / 10.0), [0.8, 0.2], [0.1, 0.5], 1_000_000, &mut rng);
            cache.push((key, p));
            p
        };
        let k = ["user1", "user2", "conventional"].iter().position(|c| *c == inputs["curve"]).unwrap();
        card.check(&inputs, p[k].log10());
    }
    card.finish().unwrap();
}
