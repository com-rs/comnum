//! Recursive clipping and active constellation extension through a Rapp amplifier
//! against Figs. 8 and 9 of A. Saul, "Peak reduction for OFDM by shaping the clipping
//! noise," *Proc. IEEE VTC2004-Fall*, vol. 1, 443–447, 2004.
//!
//! The link of Table I: OFDM of $N = 1024$ subcarriers, QPSK (Fig. 8) or 16-QAM (Fig. 9),
//! uncoded, oversampled $L = 2$ for the peak reduction, a guard interval of 11.1 %
//! (114 samples of the 1024) with the raised-cosine window of eq. (1) over the guard
//! interval (`Ofdm::windowed`, a roll-off of 11.1 %), then the Rapp SSPA of eq. (2) with
//! $p = 10$ driven at an output back-off of 6 dB (eq. (3), `Rapp::scale_for_output_backoff`),
//! then AWGN. The peak reduction is recursive clipping (`ClipFilter`: clip at the
//! clipping ratio CR of eq. (4), filter out of band, twice) or the clipping-based ACE of
//! Krongold & Jones (`Ace`, thirty iterations of clipping at CR with the projected noise
//! added with a unit step: the paper's two iterations use the optimised step of the smart
//! gradient projection), for CR from 12 dB down to 0 dB by 0.5 dB. Each point of the figures is the pair (the
//! $E_{b}/N_{0}$ loss at a BER of $10^{-3}$ against the undistorted link, the PSD at the
//! normalised frequency $f/f_{s} = 0.6$ after the amplifier, in dB below the in-band
//! density), $f_{s} = N \Delta f$ the Nyquist rate of the subcarrier grid (not defined in
//! the paper). The PSD is Welch's estimate with a 120 dB Dolph–Chebyshev window of 4096
//! samples (`welch`); the loss is the crossing of the BER of 100 blocks (`sweep_crossing`,
//! a step of 0.2 dB) with the received symbols scaled by the least-squares gain on the
//! known data (the AGC of a conventional receiver). The clipping ratio of every marker of
//! the figures is not labelled, so the comparison is between curves: at the PSD of every
//! published point the loss of comnum's curve (linear interpolation along the clipping
//! ratios, the first crossing from 12 dB) is compared with the published loss. The values
//! read on the figures are in the cards
//! `validation/cards/saul2004_fig8.yaml` and `saul2004_fig9.yaml`.
//!
//! Not in the paper: the number of OFDM symbols, the PSD estimator, the receiver's gain
//! control, the window's exact shape (its roll-off is taken over the whole guard interval).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::C64;
use comnum::core::rng::Rng;
use comnum::dsp::spectrum::welch;
use comnum::dsp::window::chebyshev;
use comnum::hardware::amplifier::Rapp;
use comnum::mapping::constellation::Constellation;
use comnum::metrics::errors::ber;
use comnum::waveform::ofdm::{NoReduction, Ofdm, PaprReduction};
use comnum::waveform::papr::{Ace, ClipFilter};
use comnum_validation::card::Card;
use comnum_validation::sweep_crossing;
use serde_json::json;

const N: usize = 1024;
const L: usize = 2;
const N_CP: usize = 114; // 11.1 % of N
const RAMP: usize = L * N_CP; // the roll-off of 11.1 % over the guard interval
const BLOCKS: usize = 100;
const OBO_DB: f64 = 6.0;
const NFFT: usize = 4096;
const F_READ: f64 = 0.6;
const TARGET_BER: f64 = 1e-3;
/// Iterations of the clipping-based ACE (unit step): the two SGP iterations of the paper
/// converge in two steps, the plain projection needs about thirty (see the card).
const ACE_ITERATIONS: usize = 30;
/// The clipping ratios, 12 dB down to 0 by 0.5 dB (the curves are convex in the
/// (loss, PSD) plane: a finer grid than the figure's markers for the interpolation).
const CR_DB: [f64; 25] = [
    12.0, 11.5, 11.0, 10.5, 10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5,
    1.0, 0.5, 0.0,
];

fn ofdm() -> Ofdm {
    Ofdm { oversampling: L, n_cp: N_CP, ..Ofdm::new(N, N).unwrap() }
}

fn rapp() -> Rapp {
    Rapp { smoothness: 10.0, ..Rapp::default() }
}

/// The amplified stream of `bits` through the window and the Rapp SSPA at `OBO_DB`,
/// with the peak reduction `reduction`; without the amplifier if `amplified` is false.
fn transmit(c: &Constellation, bits: &[u8], reduction: &dyn PaprReduction, amplified: bool) -> Vec<C64> {
    let ofdm = ofdm();
    let (x, _) = ofdm.modulate_with(&c.map_bits(bits).unwrap(), reduction).unwrap();
    let y = ofdm.windowed(&x, RAMP).unwrap();
    if !amplified {
        return y;
    }
    let pa = rapp();
    let v = pa.scale_for_output_backoff(&y, OBO_DB).unwrap();
    pa.amplify(&y.iter().map(|s| s * v).collect::<Vec<_>>()).unwrap()
}

/// The PSD at $f/f_{s}$ = `F_READ` in dB below the in-band density ($|f| < 0.4 f_{s}$).
fn psd_dbc(z: &[C64]) -> f64 {
    let (f, p) = welch(z, 2.0, NFFT, Some(&chebyshev(NFFT, 120.0)), NFFT / 2).unwrap();
    let (mut sum, mut n) = (0.0, 0);
    for (fi, pi) in f.iter().zip(&p) {
        if fi.abs() < 0.4 {
            sum += pi;
            n += 1;
        }
    }
    let in_band = sum / n as f64;
    let k = f.iter().enumerate().min_by(|a, b| (a.1 - F_READ).abs().total_cmp(&(b.1 - F_READ).abs())).unwrap().0;
    let out = p[k - 2..=k + 2].iter().sum::<f64>() / 5.0;
    10.0 * (out / in_band).log10()
}

/// The BER of the stream `z` at `ebn0_db`: the noise variance per sample from the mean
/// power of the stream and its $K \log_{2} M$ bits per block of $L(N + N_{\mathrm{cp}})$
/// samples; the received symbols scaled by the least-squares gain on the known symbols.
fn ber_at(c: &Constellation, z: &[C64], bits: &[u8], ebn0_db: f64, seed: u64) -> f64 {
    let ofdm = ofdm();
    let power = z.iter().map(C64::norm_sqr).sum::<f64>() / z.len() as f64;
    let bits_per_block = (N * c.bits_per_symbol()) as f64;
    let sigma2 = power * ofdm.block_len() as f64 / bits_per_block / 10f64.powf(ebn0_db / 10.0);
    let mut rng = Rng::new(seed);
    let r: Vec<C64> = z.iter().map(|s| s + rng.complex_gaussian() * sigma2.sqrt()).collect();
    let d = ofdm.demodulate(&r).unwrap();
    let x = c.map_bits(bits).unwrap();
    let num: C64 = x.iter().zip(&d).map(|(a, b)| a.conj() * b).sum();
    let den: f64 = x.iter().map(C64::norm_sqr).sum();
    let g = num / den;
    let eq: Vec<C64> = d.iter().map(|v| v / g).collect();
    ber(&c.demap_bits(&eq), bits).unwrap()
}

/// The $E_{b}/N_{0}$ (dB) at which the BER of `z` crosses the target, up to 16 dB above
/// `start_db` (`None` beyond: an error floor).
fn crossing(c: &Constellation, z: &[C64], bits: &[u8], start_db: f64) -> Option<f64> {
    sweep_crossing(start_db, 0.2, 81, TARGET_BER, |k, e| ber_at(c, z, bits, e, 1000 + k as u64))
}

/// The loss of comnum's curve `points` (clipping ratio order, (loss, psd)) at the PSD
/// `psd`: the first segment, from 12 dB down, whose PSD crosses it, interpolated.
fn loss_at_psd(points: &[(f64, f64)], psd: f64) -> f64 {
    for w in points.windows(2) {
        let ((l0, p0), (l1, p1)) = (w[0], w[1]);
        if (p0 >= psd && p1 <= psd) || (p0 <= psd && p1 >= psd) {
            return if p0 == p1 { l0 } else { l0 + (l1 - l0) * (psd - p0) / (p1 - p0) };
        }
    }
    f64::NAN
}

/// One figure: the two techniques over the clipping ratios, against its card.
fn figure(mut card: Card, m: usize, seed: u64, start_db: f64) {
    let c = Constellation::qam(m).unwrap().normalized();
    let mut rng = Rng::new(seed);
    let bits: Vec<u8> = (0..BLOCKS * N * c.bits_per_symbol()).map(|_| rng.below(2) as u8).collect();
    let linear = transmit(&c, &bits, &NoReduction, false);
    let reference = crossing(&c, &linear, &bits, start_db).unwrap();
    let unclipped = transmit(&c, &bits, &NoReduction, true);
    println!(
        "{m}-QAM: undistorted crossing {reference:.2} dB, PSD of the windowed signal {:.1} dBc; the amplifier alone: loss {:.2} dB, PSD {:.1} dBc",
        psd_dbc(&linear),
        crossing(&c, &unclipped, &bits, start_db).unwrap() - reference,
        psd_dbc(&unclipped)
    );
    let mut curves: Vec<(&str, Vec<(f64, f64)>)> = Vec::new();
    for technique in ["clip_filter", "ace"] {
        println!("{technique}:  CR (dB)   loss (dB)   PSD (dBc)");
        let mut points = Vec::new();
        for cr in CR_DB {
            let reduction: Box<dyn PaprReduction> = match technique {
                "clip_filter" => Box::new(ClipFilter { ratio_db: cr, iterations: 2 }),
                _ => Box::new(Ace { constellation: c.clone(), ratio_db: cr, iterations: ACE_ITERATIONS, step: 1.0 }),
            };
            let z = transmit(&c, &bits, reduction.as_ref(), true);
            let psd = psd_dbc(&z);
            match crossing(&c, &z, &bits, start_db) {
                Some(e) => {
                    let loss = e - reference;
                    println!("              {cr:4.1}      {loss:6.2}      {psd:6.1}");
                    points.push((loss, psd));
                }
                None => println!(
                    "              {cr:4.1}      (no crossing below {:.0} dB: an error floor)      {psd:6.1}",
                    start_db + 16.0
                ),
            }
        }
        curves.push((technique, points));
    }
    for (technique, points) in &curves {
        for inputs in card.inputs_matching(&json!({"technique": technique})) {
            if let Some(psd) = inputs.get("psd_dbc").and_then(serde_json::Value::as_f64) {
                card.check(&inputs, loss_at_psd(points, psd));
            } else if inputs.get("claim").and_then(serde_json::Value::as_str) == Some("psd_floor_dbc") {
                card.check(&inputs, points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min));
            }
        }
    }
    card.finish().unwrap();
}

#[test]
#[ignore = "validation against the literature: a few minutes, run with --release --ignored"]
fn the_clipping_and_ace_curves_of_qpsk_follow_fig_8_of_saul_2004() {
    figure(Card::load("saul2004_fig8").unwrap(), 4, 2004, 5.0);
}

#[test]
#[ignore = "validation against the literature: a few minutes, run with --release --ignored"]
fn the_clipping_and_ace_curves_of_16qam_follow_fig_9_of_saul_2004() {
    figure(Card::load("saul2004_fig9").unwrap(), 16, 2005, 9.0);
}
