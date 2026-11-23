//! The Manakov-PMD channel of `comnum::channel::optical::pmd::ManakovPmd` and the two
//! digital back-propagation baselines of Fig. 8 of R. M. Bütler, C. Häger, H. D. Pfister,
//! G. Liga and A. Alvarado, "Model-based machine learning for joint digital
//! backpropagation and PMD compensation," *J. Lightw. Technol.* 39(4), 949–959, 2021
//! (arXiv:2010.12313).
//!
//! The link of its Table II: one channel of Gaussian symbols at 32 GBd, root-raised-cosine
//! pulses of roll-off 0.01 at 6 samples per symbol, 10 spans of 100 km ($\alpha = 0.2$
//! dB/km, $\beta_{2} = -21.67$ ps²/km, $\gamma = 1.2$ W⁻¹km⁻¹ at 1550 nm), an EDFA of 4.5
//! dB noise figure after every span, the PMD parameter 0.2 ps/√km in sections of 0.1 km
//! (1000 per span, each a uniform SU(2) rotation and a Gaussian DGD, its Sec. III-B), the
//! forward split step cut at the sections. The receiver of its Fig. 4: the field resampled
//! to 2 samples per symbol, then
//!
//! - *DBP without PMD*: the link without PMD and its dual-polarisation back-propagation
//!   with 1000 steps per span (`Dbp::run_dual`), the ceiling of its Fig. 8;
//! - *DBP with PMD*: the link with PMD, the same back-propagation, then the ideal
//!   frequency-domain PMD equaliser $J(\omega)^{-1}$ (`ManakovPmd::equalize_pmd`), the
//!   paper's "ideal lumped PMD compensator";
//! - *DBP with the PMD known*: the exact inverse of the sections
//!   (`ManakovPmd::backpropagate_with_pmd`), the distributed compensation the paper's
//!   learned model approaches (not a curve of the paper);
//!
//! the matched filter, one sample per symbol, and the effective SNR of its eq. (17)
//! (here after the least-squares gain, which removes the phase offset as its genie does),
//! averaged in the linear domain over the polarisations and the PMD realisations. Without
//! Kerr effect the SNR is the closed form of the ASE of the ten amplifiers in the 32 GHz of
//! the matched filter, the dashed line of its Fig. 8.
//!
//! The card `validation/cards/butler2021_fig8.yaml` holds the markers of the figure read
//! from its vector paths; `just card butler2021_fig8` runs the campaign (the forward link
//! takes 10 000 steps per point).
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::C64;
use comnum::channel::optical::edfa::ase_density;
use comnum::channel::optical::fiber::FiberParams;
use comnum::channel::optical::manakov::ManakovSsfm;
use comnum::channel::optical::pmd::{ManakovPmd, PmdLink};
use comnum::core::rng::Rng;
use comnum::core::signal::DualPol;
use comnum::dsp::filter::rrc_spectrum;
use comnum::dsp::resample::resample_fft;
use comnum::math::fft::{fft, ifft};
use comnum::math::{db2lin, dbm2w, lin2db};
use comnum::metrics::effective_snr_db;
use comnum_validation::card::Card;
use serde_json::json;

const RATE: f64 = 32e9;
const ROLLOFF: f64 = 0.01;
const SPS_TX: usize = 6;
const SPS_RX: usize = 2;
const SYMBOLS: usize = 4096;
const SPANS: usize = 10;
const SPAN_KM: f64 = 100.0;
const SECTIONS_PER_SPAN: usize = 1000;
const DBP_STEPS_PER_SPAN: usize = 1000;
const PMD_PS_SQRT_KM: f64 = 0.2;
const BETA2_PS2_KM: f64 = -21.67;
const WAVELENGTH_M: f64 = 1550e-9;
/// The PMD realisations averaged (the paper: 40).
const REALISATIONS: usize = 4;
/// The powers of the figure (dBm) measured.
const POWERS_DBM: [f64; 7] = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

/// The link of Table II: $D = -2\pi c \beta_{2} / \lambda^{2}$ at 1550 nm (16.99 ps/(nm·km)).
fn fiber() -> FiberParams {
    let c = 299_792_458.0;
    // β₂ in s²/m, D in s/m², then ps/(nm·km)
    let d_s_m2 = -2.0 * std::f64::consts::PI * c * BETA2_PS2_KM * 1e-27 / (WAVELENGTH_M * WAVELENGTH_M);
    FiberParams {
        n_spans: SPANS,
        span_km: SPAN_KM,
        alpha_db_km: 0.2,
        gamma_w_km: 1.2,
        d_ps_nm_km: d_s_m2 * 1e6,
        fc_hz: c / WAVELENGTH_M,
        nf_db: 4.5,
        slope_ps_nm2_km: 0.0,
    }
}

/// Gaussian symbols of unit power and their root-raised-cosine waveform on the periodic
/// block at `sps` samples per symbol.
fn shape(symbols: &[C64], sps: usize) -> Vec<C64> {
    let spectrum = fft(symbols);
    let s = rrc_spectrum(SYMBOLS, sps, ROLLOFF);
    let up: Vec<C64> = (0..SYMBOLS * sps).map(|k| spectrum[k % SYMBOLS] * s[k]).collect();
    ifft(&up)
}

/// The two polarisations of Gaussian symbols at the total power `power_dbm` and the
/// symbols sent.
fn launch(rng: &mut Rng, power_dbm: f64) -> (DualPol, [Vec<C64>; 2]) {
    let sx: Vec<C64> = (0..SYMBOLS).map(|_| rng.complex_gaussian()).collect();
    let sy: Vec<C64> = (0..SYMBOLS).map(|_| rng.complex_gaussian()).collect();
    let mut u = DualPol::new(shape(&sx, SPS_TX), shape(&sy, SPS_TX)).unwrap();
    let scale = (dbm2w(power_dbm) / u.power()).sqrt();
    u.scale(scale);
    (u, [sx, sy])
}

/// The receiver after the back-propagation: matched filter, one sample per symbol (taken
/// in the time domain: a band-limited decimation would drop the excess band of the
/// raised cosine instead of folding it, an ISI floor 31 dB below the signal), the
/// effective SNR (linear) of both polarisations.
fn effective_snr(u: &DualPol, sent: &[Vec<C64>; 2]) -> f64 {
    let s = rrc_spectrum(SYMBOLS, SPS_RX, ROLLOFF);
    let snr = |z: &[C64], x: &[C64]| {
        let spectrum: Vec<C64> = fft(z).iter().zip(&s).map(|(v, g)| v * g).collect();
        let filtered = ifft(&spectrum);
        let y: Vec<C64> = (0..SYMBOLS).map(|k| filtered[k * SPS_RX]).collect();
        db2lin(effective_snr_db(&y, x).unwrap())
    };
    (snr(&u.x, &sent[0]) + snr(&u.y, &sent[1])) / 2.0
}

/// The field at the receiver's 2 samples per symbol.
fn at_receiver(u: &DualPol) -> DualPol {
    DualPol::new(resample_fft(&u.x, SYMBOLS * SPS_RX), resample_fft(&u.y, SYMBOLS * SPS_RX)).unwrap()
}

/// The three receivers of one realisation of the PMD at one power: DBP then the ideal
/// lumped equaliser, and the exact inverse of the sections; `None` for the link without
/// PMD, whose DBP alone is the ceiling.
fn measure(link: &ManakovPmd, power_dbm: f64, seed: u64) -> (f64, f64) {
    let mut rng = Rng::new(seed);
    let (mut u, sent) = launch(&mut rng, power_dbm);
    let fs_tx = RATE * SPS_TX as f64;
    let fs_rx = RATE * SPS_RX as f64;
    link.propagate(&mut u, fs_tx, Some(&mut rng)).unwrap();
    let received = at_receiver(&u);

    let dbp = ManakovSsfm::new(link.fiber.clone()).dbp(DBP_STEPS_PER_SPAN, 1.0);
    let mut lumped = received.clone();
    dbp.run_dual(&mut lumped.x, &mut lumped.y, fs_rx).unwrap();
    link.equalize_pmd(&mut lumped, fs_rx);

    let mut known = received;
    link.backpropagate_with_pmd(&mut known, fs_rx);
    (effective_snr(&lumped, &sent), effective_snr(&known, &sent))
}

/// The ASE-limited SNR of one polarisation (dB): $P/2$ over the noise of the amplifiers
/// in the bandwidth of the matched filter.
fn linear_snr_db(power_dbm: f64) -> f64 {
    let f = fiber();
    let ase = SPANS as f64 * ase_density(db2lin(f.span_gain_db()), db2lin(f.nf_db), f.fc_hz) * RATE;
    lin2db(dbm2w(power_dbm) / 2.0 / ase)
}

/// Check the case when the card has it (the card decides which points are compared).
fn check(card: &mut Card, inputs: &serde_json::Value, measured: f64) {
    if card.expected(inputs).is_some() {
        card.check(inputs, measured);
    }
}

#[test]
#[ignore = "campaign: about 15 minutes in release; `just card butler2021_fig8`"]
fn fig_8_dbp_with_and_without_pmd() {
    let mut card = Card::load("butler2021_fig8").unwrap();
    let fiber = fiber();
    assert!((fiber.beta2_ps2_km() - BETA2_PS2_KM).abs() < 1e-2, "β₂ of the link: {} ps²/km", fiber.beta2_ps2_km());
    let no_pmd = ManakovPmd::new(fiber.clone(), PmdLink::none(&fiber, SECTIONS_PER_SPAN).unwrap());
    let links: Vec<ManakovPmd> = (0..REALISATIONS as u64)
        .map(|k| {
            let mut rng = Rng::new(1000 + k);
            ManakovPmd::new(
                fiber.clone(),
                PmdLink::random(&fiber, PMD_PS_SQRT_KM, SECTIONS_PER_SPAN, &mut rng).unwrap(),
            )
        })
        .collect();
    let dgd: Vec<String> = links.iter().map(|l| format!("{:.2}", l.pmd.rms_dgd_s() * 1e12)).collect();
    let mean_dgd = PMD_PS_SQRT_KM * fiber.total_km().sqrt();
    println!(
        "rms DGD of the sections of the realisations (ps): {}; the mean DGD of the link {mean_dgd:.2} ps, its rms {:.2} ps",
        dgd.join(", "),
        mean_dgd * (3.0 * std::f64::consts::PI / 8.0).sqrt()
    );

    // without Kerr effect: the closed form of the ASE
    let linear = ManakovPmd::new(FiberParams { gamma_w_km: 0.0, ..fiber.clone() }, links[0].pmd.clone());
    let (snr_lin, _) = measure(&linear, 8.0, 7);
    println!("linear transmission at 8 dBm: {:.2} dB, closed form {:.2} dB", lin2db(snr_lin), linear_snr_db(8.0));
    check(&mut card, &json!({"receiver": "linear", "power_dbm": 8.0}), lin2db(snr_lin));
    check(&mut card, &json!({"receiver": "linear_closed_form", "power_dbm": 8.0}), linear_snr_db(8.0));

    println!("{:>6} {:>12} {:>12} {:>12} {:>12}", "P", "no PMD", "PMD+lumped", "PMD known", "std (dB)");
    for p in POWERS_DBM {
        let (no, _) = measure(&no_pmd, p, 1);
        let runs: Vec<(f64, f64)> = links.iter().enumerate().map(|(k, l)| measure(l, p, 1 + k as u64)).collect();
        let lumped_db: Vec<f64> = runs.iter().map(|r| lin2db(r.0)).collect();
        let lumped = runs.iter().map(|r| r.0).sum::<f64>() / runs.len() as f64;
        let known = runs.iter().map(|r| r.1).sum::<f64>() / runs.len() as f64;
        let mean = lumped_db.iter().sum::<f64>() / runs.len() as f64;
        let std = (lumped_db.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (runs.len() - 1) as f64).sqrt();
        println!("{p:>6.1} {:>12.2} {:>12.2} {:>12.2} {std:>12.2}", lin2db(no), lin2db(lumped), lin2db(known));
        check(&mut card, &json!({"receiver": "dbp_no_pmd", "power_dbm": p}), lin2db(no));
        check(&mut card, &json!({"receiver": "dbp_pmd_lumped", "power_dbm": p}), lin2db(lumped));
        check(
            &mut card,
            &json!({"receiver": "dbp_pmd_known_minus_no_pmd", "power_dbm": p}),
            lin2db(known) - lin2db(no),
        );
    }
    card.finish().unwrap();
}
