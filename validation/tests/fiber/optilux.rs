//! The scalar split-step Fourier method (`comnum_channel::optical::fiber::Ssfm`) against
//! the `fiber` of Optilux (P. Serena, University of Parma, the release of April 2021,
//! GPL-3.0, <https://optilux.sourceforge.io/>), run under Octave by
//! `tools/record_optilux.py` on the same input field and recorded in
//! `validation/optilux/ssfm_256.json` (the outputs of the tool, not its code): 16-QAM at
//! 32 GBd, root-raised cosine of roll-off 0.1, 4 samples per symbol, 4 dBm, 2⁸ symbols;
//! spans of 80 km of standard fibre (0.2 dB/km, 17 ps/(nm·km), no slope, 1.3 /(W·km) at
//! 193.1 THz), a fixed symmetric step of 1 km and a noiseless amplification of `αL` at
//! the end of every span. Three cases: one span without the Kerr effect, one span with
//! it, the ten spans of the benchmark scenario (`benchmarks/libraries/s5_ssfm.py`).
//!
//! The test measures the normalised mean square error of comnum's field against the
//! recorded one, $\mathrm{NMSE} = \lVert u - u_{\mathrm{Optilux}} \rVert^{2} /
//! \lVert u_{\mathrm{Optilux}} \rVert^{2}$, at the same step on the same field; the
//! expected value is 0 (the same solution) and the tolerance of every case is the
//! difference of the schemes measured, written on the card
//! `validation/cards/optilux_ssfm.yaml`. Without the Kerr effect the two solvers apply
//! the same dispersion operator and attenuation and agree to the arithmetic. With it
//! they differ by the nonlinear step: comnum rotates the phase by `γ|A|²h` with the
//! field at the middle of the step, Optilux by `γ|A|² L_eff(h)` with
//! $L_{\mathrm{eff}}(h) = (1 - e^{-\alpha h})/\alpha$, the attenuation of the step applied
//! around its nonlinear block (Shao, Liang & Kumar, *IEEE Photonics J.* 6(4), 2014,
//! cited by its help). Optilux writes the equation with the opposite sign of the
//! imaginary unit: the recorder propagated the conjugate of the field and conjugated the
//! result; its fields are in √mW, recorded in √W.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use std::path::PathBuf;

use comnum::C64;
use comnum::channel::optical::fiber::{Amplification, FiberParams, Ssfm};
use comnum_validation::card::Card;
use serde_json::{Value, json};

/// The recorded fields of `validation/optilux/ssfm_256.json`, by case.
fn recorded() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("optilux").join("ssfm_256.json");
    serde_json::from_str(&std::fs::read_to_string(&path).expect("validation/optilux/ssfm_256.json is missing")).unwrap()
}

fn field(doc: &Value, name: &str) -> Vec<C64> {
    doc["fields"][name]
        .as_array()
        .unwrap()
        .iter()
        .map(|z| C64::new(z[0].as_f64().unwrap(), z[1].as_f64().unwrap()))
        .collect()
}

/// comnum's field after `n_spans` spans of the recorded link, with `gamma_scale` times
/// its nonlinear coefficient, at the recorded step.
fn propagate(doc: &Value, input: &[C64], n_spans: usize, gamma_scale: f64) -> Vec<C64> {
    let link = &doc["link"];
    let fiber = FiberParams {
        alpha_db_km: link["alpha_db_km"].as_f64().unwrap(),
        d_ps_nm_km: link["d_ps_nm_km"].as_f64().unwrap(),
        gamma_w_km: gamma_scale * link["gamma_w_km"].as_f64().unwrap(),
        span_km: link["span_km"].as_f64().unwrap(),
        n_spans,
        fc_hz: link["fc_hz"].as_f64().unwrap(),
        ..FiberParams::default()
    };
    let mut ssfm = Ssfm::new(fiber, link["step_km"].as_f64().unwrap());
    ssfm.amplification = Amplification::Ideal;
    let mut u = input.to_vec();
    ssfm.propagate(&mut u, link["fs_hz"].as_f64().unwrap(), None).unwrap();
    u
}

fn nmse(u: &[C64], reference: &[C64]) -> f64 {
    let err: f64 = u.iter().zip(reference).map(|(a, b)| (a - b).norm_sqr()).sum();
    err / reference.iter().map(C64::norm_sqr).sum::<f64>()
}

#[test]
fn the_split_step_of_optilux_on_the_same_field() {
    let doc = recorded();
    let input = field(&doc, "input");
    let mut card = Card::load("optilux_ssfm").unwrap();
    for (case, n_spans, gamma_scale) in [("linear_1span", 1, 0.0), ("kerr_1span", 1, 1.0), ("kerr_10spans", 10, 1.0)] {
        let steps: Vec<u64> =
            doc["steps_per_span"][case].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        assert_eq!(steps.len(), n_spans, "{case}: the recorded spans");
        let ours = propagate(&doc, &input, n_spans, gamma_scale);
        let e = nmse(&ours, &field(&doc, case));
        println!("{case}: {n_spans} span(s), Optilux steps per span {steps:?}, NMSE {e:.3e}");
        card.check(&json!({"case": case}), e);
    }
    card.finish().unwrap();
}

// ------------------------------------------------------------------ the amplified link

use comnum::channel::optical::edfa::Edfa;
use comnum::core::rng::Rng;
use comnum::dsp::filter::rrc_spectrum;
use comnum::dsp::resample::{downsample, upsample};
use comnum::mapping::constellation::{Constellation, indices_to_bits};
use comnum::mapping::source::symbols;
use comnum::math::fft::{fft, ifft};
use comnum::metrics::{bit_errors, effective_snr_db, ls_gain};

/// The link of Optilux's example 21, recorded in `validation/optilux/ase_link.json`:
/// polarisation-multiplexed QPSK at 10 GBd (root-raised cosine of roll-off 0.01, 32
/// samples per symbol, 0.5 mW per polarisation) through 50 spans of 150 km at
/// 0.2 dB/km without dispersion nor Kerr effect, each followed by an amplifier of 30 dB
/// gain and 6 dB noise figure at 1550 nm; a root-raised-cosine matched filter, the
/// sample at the symbol instant, the least-squares gain, hard decisions. The test
/// measures, over `BLOCKS` independent blocks of 1024 symbols per polarisation (every
/// span's noise drawn anew), the SNR (dB, the data-aided Es/N0 of
/// `comnum::metrics::effective_snr_db`) and the bit error rate after the spans the
/// card names.
const SYMBOLS: usize = 1024;
const SPS: usize = 32;
const SYMBOL_RATE: f64 = 10e9;
const ROLLOFF: f64 = 0.01;
const POWER_W: f64 = 0.5e-3;
const SPANS: usize = 50;
const BLOCKS: usize = 200;

fn ase_link(doc: &Value) -> (Vec<(usize, f64, f64)>, Vec<usize>) {
    let link = &doc["link"];
    let loss_field = 10f64.powf(-link["alpha_db_km"].as_f64().unwrap() * link["span_km"].as_f64().unwrap() / 20.0);
    let edfa = Edfa {
        gain_db: link["amplifier_gain_db"].as_f64().unwrap(),
        nf_db: link["noise_figure_db"].as_f64().unwrap(),
        fc_hz: 299_792_458.0 / (link["wavelength_nm"].as_f64().unwrap() * 1e-9),
    };
    let fs = SYMBOL_RATE * SPS as f64;
    let n = SYMBOLS * SPS;
    let spectrum = rrc_spectrum(SYMBOLS, SPS, ROLLOFF);
    let shape = |x: &[C64]| -> Vec<C64> {
        let mut xf = fft(x);
        for (v, h) in xf.iter_mut().zip(&spectrum) {
            *v *= *h;
        }
        ifft(&xf)
    };
    let c = Constellation::qam(4).unwrap().normalized();
    let checkpoints: Vec<usize> = (10..=SPANS).step_by(10).collect();
    let mut rng = Rng::new(1);
    // per checkpoint: the sum of 1/SNR, the bit errors, the bits
    let mut inv_snr = vec![0.0; checkpoints.len()];
    let mut errors = vec![0usize; checkpoints.len()];
    let mut bits = vec![0usize; checkpoints.len()];
    for _ in 0..BLOCKS {
        for _pol in 0..2 {
            let (labels, points) = symbols(&c, SYMBOLS, None, &mut rng).unwrap();
            let tx_bits = indices_to_bits(&labels, c.bits_per_symbol());
            let mut x = shape(&upsample(&points, SPS));
            let power: f64 = x.iter().map(C64::norm_sqr).sum::<f64>() / n as f64;
            let scale = (POWER_W / power).sqrt();
            for v in &mut x {
                *v *= scale;
            }
            for span in 1..=SPANS {
                for v in &mut x {
                    *v *= loss_field;
                }
                edfa.apply(&mut x, fs, &mut rng).unwrap();
                if let Some(i) = checkpoints.iter().position(|&s| s == span) {
                    let r = downsample(&shape(&x), SPS, 0);
                    let g = ls_gain(&r, &points).unwrap();
                    let z: Vec<C64> = r.iter().map(|v| g * v).collect();
                    inv_snr[i] += 10f64.powf(-effective_snr_db(&r, &points).unwrap() / 10.0);
                    errors[i] += bit_errors(&c.demap_bits(&z), &tx_bits).unwrap();
                    bits[i] += tx_bits.len();
                }
            }
        }
    }
    let rows = checkpoints
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, -10.0 * (inv_snr[i] / (2 * BLOCKS) as f64).log10(), errors[i] as f64 / bits[i] as f64))
        .collect();
    (rows, errors)
}

#[test]
fn the_amplified_link_of_example_21() {
    let doc: Value = serde_json::from_str(
        &std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("optilux").join("ase_link.json"))
            .expect("validation/optilux/ase_link.json is missing"),
    )
    .unwrap();
    let mut card = Card::load("optilux_ase_link").unwrap();
    let (rows, errors) = ase_link(&doc);
    for (i, &(span, snr_db, ber)) in rows.iter().enumerate() {
        let rec = &doc["spans_recorded"][span - 1];
        println!(
            "{span} spans: SNR {snr_db:.2} dB (Optilux {:.2}, theory {:.2}); BER {ber:.3e} ({} errors; Optilux {:.3e}, theory {:.3e})",
            rec["snr_db"].as_f64().unwrap(),
            rec["snr_theory_db"].as_f64().unwrap(),
            errors[i],
            rec["ber"].as_f64().unwrap(),
            rec["ber_theory"].as_f64().unwrap()
        );
        card.check(&json!({"spans": span, "quantity": "snr_db"}), snr_db);
        card.check(&json!({"spans": span, "quantity": "snr_theory_db"}), snr_db);
        if span >= 20 {
            card.check(&json!({"spans": span, "quantity": "ber_theory"}), ber);
        }
    }
    card.finish().unwrap();
}
