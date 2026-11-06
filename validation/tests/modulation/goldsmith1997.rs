//! Variable-rate variable-power MQAM over Rayleigh fading (`comnum::mapping::link_adaptation`)
//! against A. J. Goldsmith and S.-G. Chua, "Variable-rate variable-power MQAM for fading
//! channels," *IEEE Trans. Commun.* 45(10), 1218–1230, 1997:
//!
//! - Fig. 3: the capacity with water-filling in time (eq. (7)) and the efficiency of
//!   continuous-rate MQAM at the BERs $10^{-3}$ and $10^{-6}$ (eq. (25)) from 5 to 25 dB;
//! - Fig. 5: the five-region discrete-rate MQAM ($M \in \{0, 2, 4, 16, 64\}$) at $10^{-3}$
//!   (eqs. (31)–(32));
//! - Table II: the average fade region durations of the five regions at 10 and 20 dB,
//!   $f_{D} = 100$ Hz (eqs. (34)–(35)), and the threshold $\gamma_{K}^{*}$ of eq. (32);
//! - Fig. 10: the BER under a constant estimation error of the SNR (eq. (42)).
//!
//! All are closed forms over Rayleigh fading (exponential integrals), evaluated in
//! microseconds. The values read on the figures, the table verbatim, the tolerances and
//! the findings are in `validation/cards/goldsmith1997_adaptive_mqam.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::mapping::link_adaptation::{
    estimation_error_ber, rayleigh_adaptive_mqam_efficiency, rayleigh_capacity_water_filling, rayleigh_discrete_mqam,
    rayleigh_region_durations,
};
use comnum_validation::card::Card;
use serde_json::json;

fn lin(db: f64) -> f64 {
    10f64.powf(db / 10.0)
}

#[test]
fn goldsmith_chua_1997_figs_3_5_10_and_table_ii() {
    let mut card = Card::load("goldsmith1997_adaptive_mqam").unwrap();
    for db in [5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0] {
        let s = lin(db);
        card.check(
            &json!({"figure": 3, "curve": "capacity", "snr_db": db}),
            rayleigh_capacity_water_filling(s).unwrap(),
        );
        for ber in [1e-3, 1e-6] {
            let r = rayleigh_adaptive_mqam_efficiency(s, ber).unwrap();
            card.check(&json!({"figure": 3, "curve": "mqam", "ber": ber, "snr_db": db}), r);
        }
    }
    let five = [2, 4, 16, 64];
    for db in [7.0, 10.0, 13.0, 16.0, 20.0, 25.0] {
        let d = rayleigh_discrete_mqam(lin(db), 1e-3, &five).unwrap();
        card.check(&json!({"figure": 5, "curve": "five_regions", "snr_db": db}), d.efficiency);
        let c = rayleigh_adaptive_mqam_efficiency(lin(db), 1e-3).unwrap();
        card.check(&json!({"figure": 5, "curve": "continuous", "snr_db": db}), c);
    }
    for db in [10.0, 20.0] {
        let d = rayleigh_discrete_mqam(lin(db), 1e-3, &five).unwrap();
        card.check(&json!({"table": "II", "quantity": "gamma_k", "snr_db": db}), d.gamma_k);
        let t = rayleigh_region_durations(&d, lin(db), 100.0).unwrap();
        for (j, tj) in t.iter().enumerate() {
            card.check(&json!({"table": "II", "region": j, "snr_db": db}), tj * 1e3);
        }
    }
    for (eps_db, ber0) in [(1.0, 1e-3), (5.0, 1e-3), (0.5, 1e-6), (5.0, 1e-6)] {
        card.check(&json!({"figure": 10, "eps_db": eps_db, "ber0": ber0}), estimation_error_ber(ber0, lin(eps_db)));
    }
    card.finish().unwrap();
}
