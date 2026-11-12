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

// ------------------------------------------------------- the dispersion-managed link

use comnum::channel::optical::manakov::ManakovSsfm;
use comnum::core::signal::DualPol;
use comnum::math::nlse::{NonlinearStep, StepRule, StepUpdate};

/// The four-wave-mixing phase of Optilux's `fiber` without a `bandwidth` field: its
/// default of 20 rad (Musetti et al. 2018, for the signal bandwidth) times 1.5², the
/// correction it applies for taking the sampling bandwidth instead.
const OPTILUX_FWM_PHASE_RAD: f64 = 20.0 * 1.5 * 1.5;

/// One fibre of the recorded dispersion-managed link (`validation/optilux/dm_link.json`)
/// as a single span of comnum's Manakov link without amplifier, its nonlinear
/// coefficient $\gamma = 2\pi n_{2} / (\lambda A_{\mathrm{eff}})$ and Optilux's
/// nonlinear step on the effective length of the step.
fn dm_fibre(doc: &Value, name: &str, rule: StepRule) -> ManakovSsfm {
    let link = &doc["link"];
    let f = &link[name];
    let lambda_m = link["wavelength_nm"].as_f64().unwrap() * 1e-9;
    let gamma_w_m = 2.0 * std::f64::consts::PI * link["n2_m2_w"].as_f64().unwrap()
        / (lambda_m * f["aeff_um2"].as_f64().unwrap() * 1e-12);
    let fiber = FiberParams {
        alpha_db_km: f["alpha_db_km"].as_f64().unwrap(),
        d_ps_nm_km: f["d_ps_nm_km"].as_f64().unwrap(),
        slope_ps_nm2_km: f["slope_ps_nm2_km"].as_f64().unwrap(),
        gamma_w_km: gamma_w_m * 1e3,
        span_km: f["length_km"].as_f64().unwrap(),
        n_spans: 1,
        fc_hz: 299_792_458.0 / lambda_m,
        ..FiberParams::default()
    };
    ManakovSsfm {
        rule,
        nonlinear: NonlinearStep::EffectiveLength,
        amplification: Amplification::None,
        check_sampling: false,
        ..ManakovSsfm::new(fiber)
    }
}

/// comnum's field after `spans` spans of the recorded dispersion-managed link, the
/// transmission fibre with the step rule `rule`, and its number of steps per span.
fn dm_link(doc: &Value, rule: StepRule, spans: usize) -> (Vec<C64>, usize) {
    let fs = doc["link"]["fs_hz"].as_f64().unwrap();
    let transmission = dm_fibre(doc, "transmission", rule);
    let compensation = dm_fibre(doc, "compensation", StepRule::Fixed(1.0));
    let gain = 10f64.powf(doc["link"]["amplifier_gain_db"].as_f64().unwrap() / 20.0);
    let x = field(doc, "input");
    let mut u = DualPol { y: vec![C64::new(0.0, 0.0); x.len()], x };
    let mut steps = 0;
    for _ in 0..spans {
        steps = transmission.propagate(&mut u, fs, None).unwrap();
        compensation.propagate(&mut u, fs, None).unwrap();
        for v in u.x.iter_mut().chain(u.y.iter_mut()) {
            *v *= gain;
        }
    }
    assert!(u.y.iter().all(|v| v.norm_sqr() == 0.0), "no PMD: the y polarisation stays empty");
    (u.x, steps)
}

/// Optilux's example 19 on the recorded input field: five spans of 100 km of
/// transmission fibre and 1 km of compensating fibre followed by a noiseless gain of
/// both losses. With Optilux's `fiber` in its symmetric step, comnum's Manakov split
/// step takes the same steps (the four-wave-mixing rule of Musetti et al. 2018 with
/// the phase `20 × 1.5²` rad over the sampling bandwidth and the constant local error
/// in the transmission fibre, one step of 1 km in the compensating fibre): the test
/// measures the normalised mean square error of the fields after one span and after
/// five, and the number of steps. The example's own step is asymmetric, which comnum
/// does not have: its field after five spans is compared with comnum's at steps of
/// 10 m (the converged solution), as is the symmetric one. The card
/// `validation/cards/optilux_dm_link.yaml` holds the tolerances.
#[test]
fn the_dispersion_managed_link_of_example_19() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("optilux").join("dm_link.json");
    let doc: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("validation/optilux/dm_link.json is missing"))
            .unwrap();
    let mut card = Card::load("optilux_dm_link").unwrap();
    let fwm = StepRule::Fwm {
        phase_rad: OPTILUX_FWM_PHASE_RAD,
        bandwidth_hz: Some(doc["link"]["fs_hz"].as_f64().unwrap()),
        update: StepUpdate::ConstantLocalError,
        max_nl_phase_rad: None,
    };
    for spans in [1, 5] {
        let (u, steps) = dm_link(&doc, fwm, spans);
        let e = nmse(&u, &field(&doc, &format!("span_{spans}_symm")));
        println!(
            "symmetric steps, {spans} span(s): {steps} steps (Optilux {}), NMSE {e:.3e}",
            doc["steps"]["symm"]["transmission"]
        );
        card.check(&json!({"spans": spans, "step": "symm", "quantity": "nmse"}), e);
        card.check(&json!({"spans": spans, "step": "symm", "quantity": "steps_transmission"}), steps as f64);
    }
    let (converged, _) = dm_link(&doc, StepRule::Fixed(0.01), 5);
    for step in ["asymm", "symm"] {
        let e = nmse(&field(&doc, &format!("span_5_{step}")), &converged);
        println!("Optilux {step} after 5 spans against comnum at 10 m steps: NMSE {e:.3e}");
        card.check(&json!({"spans": 5, "step": step, "quantity": "nmse_converged"}), e);
    }
    card.finish().unwrap();
}

// ------------------------------------------------------------ the WDM back to back

use comnum::dsp::signal::freq_shift;

/// Optilux's example 16 in comnum: five channels of polarisation-multiplexed 16-QAM at
/// 32 GBd spaced by 37.5 GHz (root-raised cosine of roll-off 0.1 on the periodic block,
/// 32 samples per symbol, 1 mW per polarisation, 1024 symbols), multiplexed back to
/// back; per channel the frequency shift to the baseband, the matched root-raised
/// cosine, the sample at the symbol instant and the data-aided SNR of both
/// polarisations (`comnum::metrics::effective_snr_db`, the inverse SNRs averaged). The
/// SNR that Optilux records (`validation/optilux/wdm_b2b.json`) is its numerical floor,
/// which the example notes; the card `validation/cards/optilux_wdm_b2b.yaml` bounds
/// comnum's by it.
#[test]
fn the_wdm_back_to_back_of_example_16() {
    const SYMBOLS: usize = 1024;
    const SPS: usize = 32;
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("optilux").join("wdm_b2b.json");
    let doc: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("validation/optilux/wdm_b2b.json is missing"))
            .unwrap();
    let mut card = Card::load("optilux_wdm_b2b").unwrap();
    let (symbol_rate, spacing, power_w) = (32e9, 37.5e9, 1e-3);
    let fs = symbol_rate * SPS as f64;
    let spectrum = rrc_spectrum(SYMBOLS, SPS, 0.1);
    let shape = |x: &[C64]| -> Vec<C64> {
        let mut xf = fft(x);
        for (v, h) in xf.iter_mut().zip(&spectrum) {
            *v *= *h;
        }
        ifft(&xf)
    };
    let c = Constellation::qam(16).unwrap().normalized();
    let mut rng = Rng::new(1);
    let offsets: Vec<f64> = (0..5).map(|k| (k as f64 - 2.0) * spacing).collect();
    let mut sent = Vec::new();
    let mut wdm = [vec![C64::new(0.0, 0.0); SYMBOLS * SPS], vec![C64::new(0.0, 0.0); SYMBOLS * SPS]];
    for &f in &offsets {
        let mut pols = Vec::new();
        for field in &mut wdm {
            let (_, points) = symbols(&c, SYMBOLS, None, &mut rng).unwrap();
            let x = shape(&upsample(&points, SPS));
            let scale = (power_w / (x.iter().map(C64::norm_sqr).sum::<f64>() / x.len() as f64)).sqrt();
            for (w, v) in field.iter_mut().zip(freq_shift(&x, f, fs)) {
                *w += v * scale;
            }
            pols.push(points);
        }
        sent.push(pols);
    }
    for (k, &f) in offsets.iter().enumerate() {
        let inv_snr: f64 = wdm
            .iter()
            .zip(&sent[k])
            .map(|(field, points)| {
                let r = downsample(&shape(&freq_shift(field, -f, fs)), SPS, 0);
                10f64.powf(-effective_snr_db(&r, points).unwrap() / 10.0)
            })
            .sum::<f64>()
            / 2.0;
        let snr_db = -10.0 * inv_snr.log10();
        println!("channel {}: SNR {snr_db:.1} dB (Optilux {:.2})", k + 1, doc["snr_db"][k].as_f64().unwrap());
        card.check(&json!({"channel": k + 1}), snr_db);
    }
    card.finish().unwrap();
}

// ------------------------------------------------------------------- the DPSK link

use comnum::dsp::filter::moving_average;
use comnum::hardware::optical::{Photodiode, balanced_photodiode, delay_interferometer};
use comnum::mapping::differential::Dpsk;

/// Optilux's example 15 in comnum: binary DPSK at 10 GBd with the rectangular pulse of
/// one symbol (Optilux's `costails` of roll-off 0.01 at 32 samples per symbol is that
/// rectangle), 1 mW, complex white Gaussian noise of variance `10^(-SNR/10) f_s/R` per
/// sample (the SNR over the symbol rate, `E_s/N_0 = E_b/N_0`); the integrator over one
/// symbol (`moving_average` of 32 samples, the `movavg` optical filter of bandwidth `2R`),
/// the one-symbol delay interferometer (`delay_interferometer`) and the ideal balanced
/// photodiode, the sample at the middle of every symbol and the sign decision. The test
/// measures the bit error rate over `BLOCKS` blocks of 1024 bits at every SNR of the
/// recording `validation/optilux/dpsk.json`; the card
/// `validation/cards/optilux_dpsk.yaml` compares it with the example's theory
/// `0.5 exp(-SNR)` and with Optilux's estimate from bit 3 on.
#[test]
fn the_dpsk_link_of_example_15() {
    const BITS: usize = 1024;
    const SPS: usize = 32;
    const BLOCKS: usize = 400;
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("optilux").join("dpsk.json");
    let doc: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("validation/optilux/dpsk.json is missing"))
            .unwrap();
    let mut card = Card::load("optilux_dpsk").unwrap();
    let dpsk = Dpsk::new(2).unwrap();
    let power_w: f64 = 1e-3;
    let pd = Photodiode::ideal();
    let mut rng = Rng::new(15);
    for point in doc["points"].as_array().unwrap() {
        let snr_db = point["snr_db"].as_f64().unwrap();
        let sigma = (power_w * 10f64.powf(-snr_db / 10.0) * SPS as f64).sqrt();
        let (mut errors, mut bits) = (0usize, 0usize);
        for _ in 0..BLOCKS {
            let tx: Vec<u8> = (0..BITS).map(|_| u8::from(rng.uniform() < 0.5)).collect();
            let s = dpsk.map_bits(&tx).unwrap();
            let e: Vec<C64> = s
                .iter()
                .flat_map(|v| std::iter::repeat_n(*v * power_w.sqrt(), SPS))
                .map(|v| v + rng.complex_gaussian() * sigma)
                .collect();
            let r = moving_average(&e, SPS);
            let [plus, minus] = delay_interferometer(&r, SPS).unwrap();
            let i = balanced_photodiode(&plus, &minus, &pd, 1.0, &mut rng).unwrap();
            // the symbol k >= 1 against the symbol k - 1: the bit k - 1
            let indices: Vec<usize> = (1..s.len()).map(|k| usize::from(i[k * SPS + SPS / 2] < 0.0)).collect();
            errors += bit_errors(&dpsk.bits(&indices).unwrap(), &tx).unwrap();
            bits += BITS;
        }
        let ber = errors as f64 / bits as f64;
        let optilux = point["errors_from_bit_3"].as_f64().unwrap() / point["bits_from_bit_3"].as_f64().unwrap();
        println!(
            "SNR {snr_db} dB: BER {ber:.3e} ({errors} errors); theory {:.3e}; Optilux from bit 3 {optilux:.3e}, the example's count {:.3e}",
            point["ber_theory"].as_f64().unwrap(),
            point["ber"].as_f64().unwrap()
        );
        card.check(&json!({"snr_db": snr_db, "quantity": "ber_theory"}), ber);
        card.check(&json!({"snr_db": snr_db, "quantity": "ber_optilux"}), ber);
    }
    card.finish().unwrap();
}
