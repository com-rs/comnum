//! The rank-p LMMSE channel estimator of OFDM (`comnum::dsp::estimation::LowRank`) and the
//! channel of random delays (`comnum::waveform::ofdm::estimation::RandomDelayProfile`)
//! against O. Edfors, M. Sandell, J.-J. van de Beek, S. K. Wilson and P. O. Börjesson,
//! "OFDM channel estimation by singular value decomposition," the conference version,
//! *Proc. IEEE VTC* 1996, 923–927 (the journal version, *IEEE Trans. Commun.* 46(7),
//! 1998, is not in hand; the figures are numbered as in the conference paper).
//!
//! $N = 64$ tones, a cyclic prefix of $L = 4$ samples, 16-QAM ($\beta = 17/9$):
//!
//! - the "uniform" channel: impulses of equal power, delays uniform over the prefix (its
//!   correlation does not depend on the number of impulses $M$);
//! - the "synchronized" channel: an impulse at 0 and $M - 1$ impulses uniform over the
//!   prefix, of power $e^{-\tau/\tau_{\mathrm{rms}}}$, $\tau_{\mathrm{rms}} = 1$ sample. The
//!   paper does not give $M$; $M = 5$ reproduces the floors of Fig. 5 within 2 % (the
//!   card `validation/cards/edfors1998.yaml`).
//!
//! The comparisons: the eigenvalues of the correlations (Fig. 4), the MSE of the rank-5,
//! 6 and 7 estimators on the synchronized channel and their floors (Fig. 5, eqs. (10)
//! and (11)), and the MSE of the rank-8 estimator with the designs and channels crossed
//! (Fig. 6), the design SNR equal to the channel SNR; every MSE relative to
//! $E\|h\|^{2} = N$.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::dsp::estimation::{LowRank, ls_noise_factor};
use comnum::math::C64;
use comnum::math::linalg::CMat;
use comnum::waveform::ofdm::estimation::RandomDelayProfile;
use comnum_validation::card::Card;
use serde_json::json;

const N: usize = 64;

fn channel(name: &str) -> CMat {
    let p = match name {
        "uniform" => RandomDelayProfile { paths: 5, max_delay: 4.0, tau_rms: None, first_at_zero: false, n_fft: N },
        _ => RandomDelayProfile { paths: 5, max_delay: 4.0, tau_rms: Some(1.0), first_at_zero: true, n_fft: N },
    };
    p.correlation_matrix(N)
}

fn qam16() -> Vec<C64> {
    (0..16).map(|i| C64::new(f64::from(2 * (i % 4) - 3), f64::from(2 * (i / 4) - 3))).collect()
}

#[test]
fn edfors_1996_figs_4_to_6() {
    let mut card = Card::load("edfors1998").unwrap();
    let beta = ls_noise_factor(&qam16()).unwrap();
    card.check(&json!({"quantity": "beta"}), beta);
    let noise = |snr_db: f64| beta / 10f64.powf(snr_db / 10.0);
    // Fig. 4
    for (name, ks) in [("uniform", 0..9), ("synchronized", 2..7)] {
        let est = LowRank::new(&channel(name), 1.0, 1).unwrap();
        for k in ks {
            card.check(&json!({"figure": 4, "channel": name, "k": k}), est.eigenvalues()[k] / N as f64);
        }
    }
    // Fig. 5
    let sync = channel("synchronized");
    for p in [5, 6, 7] {
        for snr in [0.0, 10.0, 15.0, 20.0, 25.0, 29.6] {
            let est = LowRank::new(&sync, noise(snr), p).unwrap();
            card.check(&json!({"figure": 5, "rank": p, "snr_db": snr}), est.mse(&sync, noise(snr)).unwrap() / N as f64);
        }
        let est = LowRank::new(&sync, noise(30.0), p).unwrap();
        card.check(&json!({"figure": 5, "rank": p, "quantity": "floor"}), est.mse_floor(&sync).unwrap() / N as f64);
    }
    // Fig. 6
    for (design, truth) in [
        ("synchronized", "uniform"),
        ("uniform", "synchronized"),
        ("uniform", "uniform"),
        ("synchronized", "synchronized"),
    ] {
        for snr in [0.3, 10.6, 19.5, 29.7] {
            let est = LowRank::new(&channel(design), noise(snr), 8).unwrap();
            let mse = est.mse(&channel(truth), noise(snr)).unwrap() / N as f64;
            card.check(&json!({"figure": 6, "design": design, "channel": truth, "snr_db": snr}), mse);
        }
    }
    card.finish().unwrap();
}
