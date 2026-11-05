//! The SNR-loss budget of an OFDM receiver (`comnum::waveform::ofdm::budget`) and the
//! sampling clock offset (`comnum::hardware::oscillator::sampling_offset`) against
//! M. Speth, S. A. Fechtel, G. Fock and H. Meyr, "Optimum receiver design for wireless
//! broad-band systems using OFDM — Part I," *IEEE Trans. Commun.* 47(11), 1668–1677,
//! 1999:
//!
//! - Fig. 7: the largest $f_{D} T_{u}$ for an SNR loss of 0.1 to 3 dB at an SNR of 10 to
//!   30 dB, read on the figure, against the bound of eqs. (47) and (51);
//! - Fig. 2: the channel-estimator gain for a loss, read on the figure, against eq. (20);
//!   the numbers of Secs. IV-A and IV-B (0.41 dB at $G = 10$, $G = 43$ for 0.1 dB, the
//!   timing noise 16.4 dB below the channel noise for 0.1 dB);
//! - Fig. 3, AWGN: the power of the interference of an FFT window late by
//!   $n_{\varepsilon}$ samples ($N = 2048$, $K = 1705$, no guard interval), simulated with
//!   `Ofdm` (oversampling 1) on 40 symbols of QPSK: the power of
//!   $z_{k} - \frac{N - n_{\varepsilon}}{N} a_{k} e^{j2\pi k n_{\varepsilon}/N}$ (eqs. (27)–(29)),
//!   against the simulated points read on the figure and against eq. (30);
//! - eq. (47): the interference of fading within a symbol, simulated with a `Tdl` of one
//!   tap with a Jakes spectrum (64 sinusoids), 256 subcarriers all active, the channel
//!   averaged over the symbol as the perfect estimate (Sec. IV-D), 400 symbols per
//!   Doppler;
//! - eqs. (38)–(40): the rotation of the subcarriers by a sampling clock offset of
//!   $\zeta = 10^{-3}$, simulated on `Ofdm` (256 subcarriers, 200 active, a guard of 32,
//!   oversampling 4) through `sampling_offset`: the slope across $k$ of the phase
//!   increment from one symbol to the next, $2\pi \zeta (N + N_{g})/N$ per subcarrier.
//!
//! The values are in the card `validation/cards/speth1999.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::channel::multipath::{Tdl, TdlParams};
use comnum::core::rng::Rng;
use comnum::hardware::oscillator::sampling_offset;
use comnum::math::C64;
use comnum::waveform::ofdm::Ofdm;
use comnum::waveform::ofdm::budget::{estimator_gain, estimator_loss_db, max_doppler, noise_budget};
use comnum_validation::card::Card;
use serde_json::json;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

/// QPSK symbols of unit energy.
fn qpsk(n: usize, rng: &mut Rng) -> Vec<C64> {
    (0..n)
        .map(|_| {
            let re = if rng.uniform() < 0.5 { FRAC_1_SQRT_2 } else { -FRAC_1_SQRT_2 };
            let im = if rng.uniform() < 0.5 { FRAC_1_SQRT_2 } else { -FRAC_1_SQRT_2 };
            C64::new(re, im)
        })
        .collect()
}

/// Fig. 3: the interference power of a window `offset` samples late, AWGN channel.
fn timing_interference(offset: usize) -> f64 {
    let ofdm = Ofdm { oversampling: 1, ..Ofdm::new(2048, 1705).unwrap() };
    let (n, k, symbols) = (2048, 1705, 41);
    let a = qpsk(k * symbols, &mut Rng::new(19));
    let x = ofdm.modulate(&a).unwrap();
    let mut power = 0.0;
    for l in 0..symbols - 1 {
        let z = ofdm.to_active(&x[l * n + offset..l * n + offset + n]).unwrap();
        let alpha = (n - offset) as f64 / n as f64;
        for (i, zi) in z.iter().enumerate() {
            let rot = C64::from_polar(alpha, 2.0 * PI * ofdm.subcarrier(i) as f64 * offset as f64 / n as f64);
            power += (zi - rot * a[l * k + i]).norm_sqr();
        }
    }
    power / ((symbols - 1) * k) as f64
}

/// Eq. (47): the interference of fading within a symbol of maximum Doppler `fd_tu`.
fn doppler_interference(fd_tu: f64) -> f64 {
    let n = 256;
    let ofdm = Ofdm { oversampling: 1, ..Ofdm::new(n, n).unwrap() };
    // one sample per second: T_u = n s
    let p =
        TdlParams { taps: vec![(0.0, 0.0)], doppler_hz: fd_tu / n as f64, sample_rate: 1.0, ..TdlParams::default() };
    let mut tdl = Tdl::new(&p).unwrap();
    tdl.sinusoids = 64;
    let mut rng = Rng::new(23);
    let (mut noise, mut gain) = (0.0, 0.0);
    let trials = 400;
    for _ in 0..trials {
        let a = qpsk(n, &mut rng);
        let x = ofdm.to_time(&a).unwrap();
        let taps = tdl.draw(n, &mut rng);
        let y = taps.apply(&[x]).unwrap().remove(0);
        let h_mean = (0..n).map(|t| taps.coefficient(0, t, 0, 0)).sum::<C64>() / n as f64;
        gain += (0..n).map(|t| taps.coefficient(0, t, 0, 0).norm_sqr()).sum::<f64>() / n as f64;
        let z = ofdm.to_active(&y).unwrap();
        noise += z.iter().zip(&a).map(|(zi, ai)| (zi - h_mean * ai).norm_sqr()).sum::<f64>() / n as f64;
    }
    noise / gain
}

/// Eq. (40): the slope across the subcarriers of the phase increment from one symbol
/// to the next under a sampling clock offset `zeta`.
fn clock_offset_phase_slope(zeta: f64) -> f64 {
    let ofdm = Ofdm { n_cp: 32, ..Ofdm::new(256, 200).unwrap() };
    let (k, symbols) = (200, 8);
    let a = qpsk(k * symbols, &mut Rng::new(29));
    let y = sampling_offset(&ofdm.modulate(&a).unwrap(), zeta);
    let z = ofdm.demodulate(&y).unwrap();
    // the phase increment of every subcarrier, averaged over the symbols, against k
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for i in 0..k {
        let inc = (1..symbols)
            .map(|l| (z[l * k + i] / a[l * k + i]) * (z[(l - 1) * k + i] / a[(l - 1) * k + i]).conj())
            .sum::<C64>();
        let kf = ofdm.subcarrier(i) as f64;
        sxy += kf * inc.arg();
        sxx += kf * kf;
    }
    sxy / sxx
}

#[test]
fn speth_1999_snr_loss_budget() {
    let mut card = Card::load("speth1999").unwrap();
    let db = |v: f64| 10f64.powf(v / 10.0);
    // Fig. 7
    for snr_db in [10.0, 15.0, 20.0, 25.0, 30.0] {
        for loss_db in [0.1, 0.5, 1.0, 2.0, 3.0] {
            card.check(
                &json!({"quantity": "max_doppler", "snr_db": snr_db, "loss_db": loss_db}),
                max_doppler(db(snr_db), loss_db),
            );
        }
    }
    // Fig. 2 and the text
    for loss_db in [0.1, 0.5, 1.0, 2.0, 3.0] {
        card.check(&json!({"quantity": "estimator_gain", "loss_db": loss_db}), estimator_gain(loss_db));
    }
    card.check(&json!({"quantity": "estimator_loss_db", "gain": 10.0}), estimator_loss_db(10.0));
    // Sec. IV-B: the timing noise must stay 16.4 dB below the channel noise for 0.1 dB
    card.check(&json!({"quantity": "timing_margin_db", "loss_db": 0.1}), -10.0 * noise_budget(1.0, 0.1).log10());
    // Fig. 3, AWGN
    for offset in [2, 4, 6, 8, 10, 20, 30] {
        card.check(&json!({"quantity": "timing_interference", "offset": offset}), timing_interference(offset));
    }
    // eq. (47)
    for fd_tu in [0.05, 0.1] {
        card.check(&json!({"quantity": "doppler_interference", "fd_tu": fd_tu}), doppler_interference(fd_tu));
    }
    // eq. (40)
    card.check(&json!({"quantity": "clock_offset_phase_slope", "zeta": 1.0e-3}), clock_offset_phase_slope(1e-3));
    card.finish().unwrap();
}
