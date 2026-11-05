//! The two-dimensional Wiener interpolation of pilots (`comnum::waveform::ofdm::wiener`)
//! against P. Hoeher, S. Kaiser and P. Robertson, "Two-dimensional pilot-symbol-aided
//! channel estimation by Wiener filtering," *Proc. IEEE ICASSP*, vol. 3, 1845–1848, 1997.
//!
//! The channel is WSSUS with uniform Doppler and delay spectra (their eq. (15)):
//! $f_{D} = 320$ Hz, $T_{s} = 260$ µs ($f_{D} T_{s} = 0.0832$), $\tau_{\max} = 10$ µs,
//! $\Delta F = 1/240$ µs ($\tau_{\max} \Delta F = 0.041667$); a block of $K = 58$ symbols and
//! $L = 115$ subcarriers. The average MSE over every position of the block (pilots and
//! edges included, their Sec. 6.2) is the closed form of their eq. (12), the covariances
//! known:
//!
//! - Fig. 2: 1-D filtering in frequency, pilots every 6 subcarriers of a symbol, 3, 5, 7
//!   and 9 taps (the nearest pilots);
//! - Fig. 4: 2-D filtering on pilots every 3 symbols and 6 subcarriers (400 pilots,
//!   $N_{t} N_{f} = 18$), 10, 25 and 100 taps chosen by the weighted distance
//!   $|\Delta l| f_{D} T_{s} + |\Delta k| \tau_{\max} \Delta F$ (their selection B).
//!
//! The readings of the figures (on the grid of the scan, about 0.05 decade) are in the
//! card `validation/cards/hoeher1997_fig2.yaml` and `hoeher1997_fig4.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::waveform::ofdm::wiener::{Correlation2d, DelaySpectrum, DopplerSpectrum, Wiener2d};
use comnum_validation::card::Card;
use serde_json::json;

const SYMBOLS: usize = 58;
const SUBCARRIERS: usize = 115;

fn correlation() -> Correlation2d {
    Correlation2d {
        fd_ts: 320.0 * 260e-6,
        doppler: DopplerSpectrum::Uniform,
        delay: DelaySpectrum::Uniform { tau_df: 10e-6 / 240e-6 },
    }
}

/// The average MSE over `positions` of the interpolator of `pilots` with `taps` taps at
/// the SNR `esn0_db`.
fn average_mse(pilots: &[(usize, usize)], positions: &[(usize, usize)], taps: usize, esn0_db: f64) -> f64 {
    let w = Wiener2d::new(pilots.to_vec(), correlation(), 10f64.powf(-esn0_db / 10.0), taps).unwrap();
    let mse = w.mse(positions).unwrap();
    mse.iter().sum::<f64>() / mse.len() as f64
}

#[test]
fn hoeher_1997_fig_2_one_dimensional_filtering() {
    let mut card = Card::load("hoeher1997_fig2").unwrap();
    let pilots: Vec<(usize, usize)> = (0..SUBCARRIERS).step_by(6).map(|k| (0, k)).collect();
    let positions: Vec<(usize, usize)> = (0..SUBCARRIERS).map(|k| (0, k)).collect();
    for (taps, last) in [(3, 40.0), (5, 40.0), (9, 35.0)] {
        for snr in [0.0, 10.0, 20.0, 30.0, last] {
            card.check(&json!({"taps": taps, "esn0_db": snr}), average_mse(&pilots, &positions, taps, snr));
        }
    }
    card.finish().unwrap();
}

#[test]
fn hoeher_1997_fig_4_two_dimensional_filtering() {
    let mut card = Card::load("hoeher1997_fig4").unwrap();
    let pilots: Vec<(usize, usize)> =
        (0..SYMBOLS).step_by(3).flat_map(|l| (0..SUBCARRIERS).step_by(6).map(move |k| (l, k))).collect();
    assert_eq!(pilots.len(), 400);
    let positions: Vec<(usize, usize)> = (0..SYMBOLS).flat_map(|l| (0..SUBCARRIERS).map(move |k| (l, k))).collect();
    for (taps, snrs) in
        [(10, vec![0.0, 10.0, 20.0, 30.0, 40.0]), (25, vec![0.0, 10.0, 20.0, 30.0]), (100, vec![0.0, 12.0, 30.0])]
    {
        for snr in snrs {
            card.check(&json!({"taps": taps, "esn0_db": snr}), average_mse(&pilots, &positions, taps, snr));
        }
    }
    card.finish().unwrap();
}
