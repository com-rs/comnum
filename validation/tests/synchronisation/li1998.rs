//! The robust channel estimator of OFDM (`comnum::waveform::ofdm::wiener::RobustDesign`)
//! against Y. Li, L. J. Cimini and N. R. Sollenberger, "Robust channel estimation for
//! OFDM systems with rapid dispersive fading channels," *IEEE Trans. Commun.* 46(7),
//! 902–915, 1998: the MSE of the estimator designed for a flat band-limited Doppler
//! spectrum and a uniform delay profile, at an SNR of 10 dB ($\rho = 0.1$), $K = 120$
//! tones, $T_{s} = 160$ µs, $T_{f} = 200$ µs, $K_{o} = \lfloor K t_{d}/T_{s} \rfloor + 1$ (the taps
//! from 0 to $t_{d}$; eq. (56) prints the ceiling, one fewer at 20 and 40 µs):
//!
//! - Fig. 3(a): the infinite-length estimator, eq. (57), against the Doppler for delay
//!   spreads of 0 to 40 µs;
//! - Fig. 4(a), (c): the $L$-tap estimator, eqs. (69)–(74), against $L$ for 10, 40 and
//!   200 Hz at 20 µs, and for 5 and 40 µs at 40 Hz.
//!
//! The figures give neither $K$ (128 or 120) nor the SNR of Fig. 4: $K = 120$ places the
//! drops of Fig. 3(b) where they are, and 10 dB reproduces Fig. 4 (see the card
//! `validation/cards/li1998.yaml`).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::waveform::ofdm::wiener::RobustDesign;
use comnum_validation::card::Card;
use serde_json::json;

const RHO: f64 = 0.1;

fn design(td_us: f64, fd_hz: f64) -> RobustDesign {
    RobustDesign::new(120, td_us / 160.0, fd_hz * 200e-6).unwrap()
}

#[test]
fn li_cimini_sollenberger_1998_figs_3_and_4() {
    let mut card = Card::load("li1998").unwrap();
    for td in [0.0, 5.0, 10.0, 20.0, 40.0] {
        for fd in [1.0, 10.0, 100.0, 200.0] {
            card.check(&json!({"figure": "3a", "td_us": td, "fd_hz": fd}), 10.0 * design(td, fd).mse(RHO).log10());
        }
    }
    for (td, fd) in [(20.0, 10.0), (20.0, 40.0), (20.0, 200.0), (5.0, 40.0), (40.0, 40.0)] {
        for taps in [1, 2, 3, 5, 10, 20, 50, 100] {
            let mse = design(td, fd).fir(taps, RHO).unwrap().1;
            card.check(&json!({"figure": "4", "td_us": td, "fd_hz": fd, "taps": taps}), 10.0 * mse.log10());
        }
    }
    card.finish().unwrap();
}
