//! Digital back-propagation with one step per span against Fig. 12 of E. Ip and
//! J. M. Kahn, "Compensation of dispersion and nonlinear impairments using digital
//! backpropagation," *J. Lightw. Technol.* 26(20), 3416–3425, 2008.
//!
//! The link of Table I without DCF: $N$ spans of 80 km of SMF ($\alpha = 0.2$ dB/km,
//! $D = 17$ ps/nm/km, $\gamma = 1.3$ W⁻¹km⁻¹), an EDFA of noise figure 5 dB after each,
//! 21.4 Gb/s 50 % RZ-QPSK (10.7 GBd, the field of a sinusoidal pulse carver, `PulseShape::Rz`)
//! at 0 dBm,
//! 32 768 symbols in a periodic block. The receiver (Fig. 3(b)) filters by a fifth-order
//! Butterworth at 40 % of its sampling rate $M/K \cdot R_{s}$ ($M/K = 2$ and 3), samples,
//! back-propagates (BP-1S: `Ssfm::backpropagate_scaled` with one step per span and the
//! scale $\xi$ of the nonlinear phase optimised over a grid, as the paper does; BP-SSFM10:
//! ten steps per span; linear equalisation: $\xi = 0$), equalises by the fractionally
//! spaced linear equaliser of the paper's receiver (`ls_equalizer`, the least-squares
//! taps of the block on the known symbols: residual dispersion, the anti-aliasing filter
//! and the matched filter in one, one sample per symbol), and measures the standard deviation of the phase error after the constant phase and gain
//! are removed (`comnum_metrics::phase_error_std`). The AWGN limit is
//! $\sigma_{\varepsilon} = 1/\sqrt{2\,\mathrm{SNR}}$ with the SNR of the accumulated ASE
//! (`ase_density`) in the symbol-rate bandwidth, and the same receiver on the linear
//! fibre ($\gamma = 0$) measures it. The forward propagation runs once per five spans
//! and is shared by the receivers; the values read on the figure are in
//! the card `validation/cards/ip2008_fig12.yaml`.
//!
//! Not in the paper: the forward step (2 km here, 12 samples per symbol), the wavelength (193.1 THz here), the
//! carrier recovery (a constant phase here, as the paper neglects laser phase noise),
//! and the value of $\xi$ at every point (searched over 0.4 to 1.0 by 0.1 and reported).

#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::C64;
use comnum::channel::optical::edfa::ase_density;
use comnum::channel::optical::fiber::{FiberParams, Ssfm};
use comnum::core::rng::Rng;
use comnum::dsp::equalizer::ls_equalizer;
use comnum::dsp::filter::{Butterworth, PulseShape, fir_filter, pulse_shape};
use comnum::dsp::resample::{downsample, upsample};
use comnum::dsp::signal::pnorm;
use comnum::mapping::constellation::Constellation;
use comnum::math::{db2lin, dbm2w};
use comnum::metrics::phase_error_std;
use comnum_validation::card::Card;
use serde_json::json;

const RS: f64 = 10.7e9;
const SPS: usize = 12;
const FS: f64 = RS * SPS as f64;
const SYMBOLS: usize = 32_768;
const POWER_DBM: f64 = 0.0;
const SPANS_PER_CHUNK: usize = 5;
const CHUNKS: usize = 16; // 5 to 80 spans
const XI_GRID: [f64; 7] = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

/// One row of the results: spans, M/K, linear equalisation, BP-1S and its xi, BP-SSFM10
/// and its xi, the measured AWGN limit.
type Row = (usize, usize, f64, f64, f64, f64, f64, f64);

fn fiber(n_spans: usize, gamma_w_km: f64) -> FiberParams {
    FiberParams {
        n_spans,
        alpha_db_km: 0.2,
        d_ps_nm_km: 17.0,
        gamma_w_km,
        span_km: 80.0,
        nf_db: 5.0,
        ..FiberParams::default()
    }
}

/// The 50 % RZ-QPSK field at `SPS` samples per symbol, 0 dBm, and its symbols.
fn transmit(rng: &mut Rng) -> (Vec<C64>, Vec<C64>) {
    let c = Constellation::qam(4).unwrap().normalized();
    let idx: Vec<usize> = (0..SYMBOLS).map(|_| rng.below(4)).collect();
    let x = c.symbols(&idx);
    let h = pulse_shape(PulseShape::Rz, SPS, 0).unwrap();
    let tx: Vec<C64> = pnorm(&fir_filter(&h, &upsample(&x, SPS))).iter().map(|v| v * dbm2w(POWER_DBM).sqrt()).collect();
    (tx, x)
}

/// The receiver of Fig. 3(b) at `mk` samples per symbol on the field `y` (at `FS`) of
/// `n_spans` spans: the periodic steady state of the Butterworth filter, the samples at
/// `offset`, the back-propagation with `steps` per span and the scale `xi`, then the
/// fractionally spaced linear equaliser (the least-squares equaliser of the block on the
/// known symbols, 4 symbols on each side); returns the decision samples.
fn receive(y: &[C64], n_spans: usize, gamma_w_km: f64, mk: usize, steps: usize, xi: f64, x: &[C64]) -> Vec<C64> {
    let fs_rx = RS * mk as f64;
    let aa = Butterworth::lowpass(5, 0.4 * fs_rx, FS).unwrap();
    // the steady state of the causal filter on the periodic block: two periods, keep the second
    let twice: Vec<C64> = y.iter().chain(y).copied().collect();
    let filtered = &aa.filter(&twice)[y.len()..];
    let r = downsample(filtered, SPS / mk, 1);
    let link = Ssfm::new(fiber(n_spans, gamma_w_km), 2.0);
    let z = link.backpropagate_scaled(&r, fs_rx, steps, xi).unwrap();
    ls_equalizer(&z, mk, x, 4, 0).unwrap().1
}

/// The phase error s.d. of the best `xi` of the grid, and that `xi`.
fn best_over_xi(y: &[C64], n_spans: usize, mk: usize, steps: usize, x: &[C64]) -> (f64, f64) {
    XI_GRID
        .iter()
        .map(|&xi| (phase_error_std(&receive(y, n_spans, 1.3, mk, steps, xi, x), x).unwrap(), xi))
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap()
}

/// The AWGN limit in closed form: the ASE of `n_spans` EDFAs in the symbol-rate
/// bandwidth against the launch power.
fn awgn_limit(n_spans: usize) -> f64 {
    let f = fiber(n_spans, 1.3);
    let gain = db2lin(f.alpha_db_km * f.span_km);
    let ase = n_spans as f64 * ase_density(gain, db2lin(f.nf_db), f.fc_hz) * RS;
    let snr = dbm2w(POWER_DBM) / ase;
    1.0 / (2.0 * snr).sqrt()
}

#[test]
#[ignore = "validation against the literature: a few minutes, run with --release --ignored"]
fn the_phase_error_of_bp_1s_against_the_distance_follows_fig_12() {
    let mut card = Card::load("ip2008_fig12").unwrap();
    let mut rng = Rng::new(2008);
    let (tx, x) = transmit(&mut rng);
    // the forward propagations, five spans at a time: nonlinear, and linear for the measured AWGN limit
    let mut snapshots: Vec<(usize, Vec<C64>, Vec<C64>)> = Vec::new();
    let (mut y_nl, mut y_lin) = (tx.clone(), tx.clone());
    let mut rng_lin = Rng::new(2008 + 1);
    // the sampling check of Ssfm measures the occupied band of the RZ signal as the whole
    // block (its sidelobes fall slowly); the power at the edge of the band is reported
    // below and the propagation runs at 12 samples per symbol as the paper's receivers do
    let edge = {
        let spectrum = comnum::math::fft::fft(&tx);
        let n = spectrum.len();
        let peak = spectrum.iter().map(C64::norm_sqr).fold(0.0, f64::max);
        let edge: f64 = spectrum[n / 2 - n / 48..n / 2 + n / 48].iter().map(C64::norm_sqr).fold(0.0, f64::max);
        10.0 * (edge / peak).log10()
    };
    println!(
        "the spectrum of the RZ-QPSK signal at the edge of the band (±{:.0} GHz): {edge:.0} dB below its peak",
        FS / 2e9
    );
    let link = |gamma| Ssfm { check_sampling: false, ..Ssfm::new(fiber(SPANS_PER_CHUNK, gamma), 2.0) };
    for k in 1..=CHUNKS {
        link(1.3).propagate(&mut y_nl, FS, Some(&mut rng)).unwrap();
        link(0.0).propagate(&mut y_lin, FS, Some(&mut rng_lin)).unwrap();
        snapshots.push((k * SPANS_PER_CHUNK, y_nl.clone(), y_lin.clone()));
    }
    // the receivers of every distance, in parallel
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let mut results: Vec<Vec<Row>> = vec![Vec::new(); snapshots.len()];
    let chunk = snapshots.len().div_ceil(threads);
    std::thread::scope(|s| {
        for (snaps, out) in snapshots.chunks(chunk).zip(results.chunks_mut(chunk)) {
            let x = &x;
            s.spawn(move || {
                for ((n, y, y_lin), o) in snaps.iter().zip(out.iter_mut()) {
                    for mk in [2usize, 3] {
                        let lin = phase_error_std(&receive(y, *n, 1.3, mk, 1, 0.0, x), x).unwrap();
                        let (bp1, xi1) = best_over_xi(y, *n, mk, 1, x);
                        let (bp10, xi10) = best_over_xi(y, *n, mk, 10, x);
                        let awgn = phase_error_std(&receive(y_lin, *n, 0.0, mk, 1, 0.0, x), x).unwrap();
                        o.push((*n, mk, lin, bp1, xi1, bp10, xi10, awgn));
                    }
                }
            });
        }
    });
    println!("spans  M/K   lin.eq   BP-1S (xi)   BP-SSFM10 (xi)   AWGN measured / closed form");
    for (n, mk, lin, bp1, xi1, bp10, xi10, awgn) in results.into_iter().flatten() {
        let limit = awgn_limit(n);
        println!("{n:5}  {mk}    {lin:.4}   {bp1:.4} ({xi1:.1})   {bp10:.4} ({xi10:.1})   {awgn:.4} / {limit:.4}");
        let panel = if mk == 2 { "a" } else { "b" };
        let mut check = |inputs: serde_json::Value, v: f64| {
            if card.expected(&inputs).is_some() {
                card.check(&inputs, v);
            }
        };
        check(json!({"panel": panel, "receiver": "lin_eq", "spans": n}), lin);
        check(json!({"panel": panel, "receiver": "bp_1s", "spans": n}), bp1);
        check(json!({"panel": panel, "receiver": "bp_ssfm10", "spans": n}), bp10);
        check(json!({"panel": panel, "receiver": "awgn_measured", "spans": n}), awgn);
        if mk == 2 {
            check(json!({"receiver": "awgn_closed_form", "spans": n}), limit);
        }
        if n == 80 {
            // the claims of Sec. III-B: BP-SSFM10 "about 2 dB better" than BP-1S at M/K = 3, "no observable difference" at M/K = 2
            card.check(&json!({"claim": "bp_ssfm10_over_bp_1s", "panel": panel, "spans": n}), bp10 / bp1);
        }
    }
    card.finish().unwrap();
}
