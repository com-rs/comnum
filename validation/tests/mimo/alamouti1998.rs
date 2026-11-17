//! The two-branch transmit diversity scheme (`comnum::mimo::stbc::Ostbc::alamouti`)
//! against S. M. Alamouti, "A simple transmit diversity technique for wireless
//! communications," *IEEE J. Sel. Areas Commun.* 16(8), 1451–1458, 1998, Fig. 4: the BER
//! of uncoded coherent BPSK over i.i.d. Rayleigh fading with two transmit antennas and
//! one or two receive antennas, under the assumptions of its Sec. IV: the two antennas
//! share the power of the single antenna of MRRC (each radiates $E_{b}/2$), every
//! transmit-receive pair fades independently with the same mean power, constant over the
//! two symbol periods of a block, and the receiver knows the channel.
//!
//! The test simulates the scheme at the values of $E_{b}/N_{0}$ where the curves of the
//! figure cross $10^{-2.5}$ to $10^{-5.5}$ (read on the grid) and measures the BER there,
//! until 400 errors; the readings, their tolerance and the reason for it are in the card
//! `validation/cards/alamouti1998_fig4.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::channel::awgn::add_noise;
use comnum::channel::fading::Mimo;
use comnum::mapping::constellation::bpsk;
use comnum::mimo::stbc::Ostbc;
use comnum::prelude::*;
use comnum_validation::card::Card;
use comnum_validation::until_errors;
use ndarray::Array2;
use serde_json::json;

/// The BER of the scheme with `n_rx` receive antennas at `ebn0_db` (the energy of a bit
/// split over the two antennas), until 400 errors.
fn ber(n_rx: usize, ebn0_db: f64, seed: u64) -> f64 {
    let code = Ostbc::alamouti();
    let ch = Mimo::rayleigh(n_rx, 2);
    // every symbol carries one bit and the encoder gives each antenna half of its
    // energy (the closed form of `mimo.rs` is MRC of 2 n_rx branches at Eb/N0 / 2):
    // the noise of a receive antenna is 1/(Eb/N0)
    let n0 = 1.0 / db2lin(ebn0_db);
    let count = until_errors(seed, 400, u64::MAX, 1, |rng| {
        let (mut errors, mut total) = (0u64, 0u64);
        for _ in 0..20_000 {
            let bits = [rng.bit(), rng.bit()];
            let s: Vec<C64> = bpsk(&bits).iter().map(|&v| C64::new(v, 0.0)).collect();
            let x = code.encode(&s)?;
            let h = ch.draw(rng)?;
            let mut y: Array2<C64> = h.dot(&x);
            add_noise(y.as_slice_mut().expect("a standard layout"), n0, rng);
            let est = code.decode(&h.view(), &y.view(), n0)?.symbols;
            errors += est.iter().zip(bits).filter(|(e, b)| (e.re < 0.0) != (*b == 1)).count() as u64;
            total += 2;
        }
        Ok((errors, total))
    })
    .unwrap();
    count.errors as f64 / count.total as f64
}

#[test]
#[ignore = "statistical validation: run with --ignored"]
fn alamouti_1998_fig_4() {
    let mut card = Card::load("alamouti1998_fig4").unwrap();
    for (i, inputs) in card.inputs_matching(&json!({})).into_iter().enumerate() {
        let n_rx = inputs["n_rx"].as_u64().unwrap() as usize;
        let ebn0_db = inputs["ebn0_db"].as_f64().unwrap();
        card.check(&inputs, ber(n_rx, ebn0_db, 1998 + i as u64));
    }
    card.finish().unwrap();
}
