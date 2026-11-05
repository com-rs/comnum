//! The closed form of the ISRS GN model (D. Semrau, R. I. Killey, P. Bayvel, "A
//! closed-form approximation of the Gaussian noise model in the presence of
//! inter-channel stimulated Raman scattering," *J. Lightw. Technol.* 37(9), 1924–1936,
//! 2019; the equation and figure numbers are those of the preprint arXiv 1808.07940v2)
//! against the dashed curves of its Figs. 4a, 4b and 5, read from the vector paths of
//! the figures to 0.02 dB.
//!
//! The link of Table I: 251 channels of 40 GBd (roll-off 0.01 %, bandwidth 40.004 GHz)
//! on a 40.005 GHz grid, 10.05 THz around 1550 nm, spans of 100 km of SMF (0.2 dB/km,
//! $D$ = 17 ps/(nm·km), $S$ = 0.067 ps/(nm²·km), $\gamma$ = 1.2 /(W·km)), a Raman gain
//! slope $C_{r}$ = 0.028 /(W·km·THz). The quantity is the NLI coefficient
//! $\eta_{n}(f_{i})$ (dB, 1/W²) of `isrs_closed_form_eta`: eq. (10) for the SPM (without
//! its last term, see the function) and eq. (11) summed over the 250 interferers, combined
//! by eq. (5).
//!
//! - **Fig. 4a**: $\eta_{1}$ after one span against the channel frequency, without ISRS
//!   ($C_{r} = 0$), at 0 dBm/ch. and at 2 dBm/ch. (`semrau2019_fig4a`).
//! - **Fig. 4b**: the change of $\eta_{1}$ by ISRS against the power transfer
//!   $\Delta\rho(L)$ of eq. (3), $4.3 P_{\mathrm{tot}} C_{r} L_{\mathrm{eff}} B_{\mathrm{tot}}$,
//!   for the channels nearest to −4500, −2000, 0, 2000 and 4500 GHz (`semrau2019_fig4b`).
//! - **Fig. 5**: $\eta_{6}$ after 6 × 100 km at 0 dBm/ch., without (a) and with (b) ISRS,
//!   the SPM accumulated coherently with $\varepsilon$ = 0.15 (Poggiolini 2012 eq. (22)
//!   over the channel bandwidth, `coherence_factor`: 0.151) or incoherently
//!   ($\varepsilon$ = 0), and the claim of Sec. III-A that the incoherent form costs
//!   0.2 dB on average (`semrau2019_fig5`).
//!
//! What the closed form neglects (Sec. II-C): four-wave mixing between interferers, the
//! XPM of the nearest interferers (assumption $|f_{k} - f_{i}| \gg B_{k}/2$), the ISRS
//! beyond first order and its change of $\varepsilon$. The numerical ISRS GN integral of
//! `GnModel` on this spectrum (a grid of 10 MHz over 10 THz, the scale of the link
//! function of the far pairs) is not run here: its cost is recorded on the cards.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::channel::optical::fiber::FiberParams;
use comnum::channel::optical::gn::{WdmSpectrum, coherence_factor, isrs_closed_form_eta, isrs_closed_form_terms};
use comnum::math::{SPEED_OF_LIGHT, dbm2w, lin2db};
use comnum_validation::card::Card;
use serde_json::json;

/// The Raman gain slope of Table I (1/W/km/THz).
const C_R: f64 = 0.028;
/// The optical bandwidth of Table I (THz), for eq. (3).
const B_TOT_THZ: f64 = 10.05;
const N_CH: usize = 251;
const SPACING: f64 = 40.005e9;
const RS: f64 = 40e9;
const ROLLOFF: f64 = 1e-4;

fn table_i(n_spans: usize) -> FiberParams {
    FiberParams {
        span_km: 100.0,
        n_spans,
        alpha_db_km: 0.2,
        d_ps_nm_km: 17.0,
        slope_ps_nm2_km: 0.067,
        gamma_w_km: 1.2,
        fc_hz: SPEED_OF_LIGHT / 1550e-9,
        ..FiberParams::default()
    }
}

fn spectrum(power_w: f64) -> WdmSpectrum {
    let freqs = (0..N_CH).map(|k| (k as f64 - 125.0) * SPACING).collect();
    WdmSpectrum::uniform(freqs, power_w, RS, ROLLOFF)
}

/// The channel nearest to `f_hz`.
fn channel(f_hz: f64) -> usize {
    ((f_hz / SPACING).round() as i64 + 125) as usize
}

#[test]
fn the_closed_form_of_eta_1_follows_the_dashed_curves_of_fig_4a() {
    let fiber = table_i(1);
    let mut card = Card::load("semrau2019_fig4a").unwrap();
    for (curve, p_dbm, c_r) in [("no_isrs", 0.0, 0.0), ("0dbm", 0.0, C_R), ("2dbm", 2.0, C_R)] {
        let s = spectrum(dbm2w(p_dbm));
        for inputs in card.inputs_matching(&json!({"curve": curve})) {
            let f_thz = inputs["f_thz"].as_f64().unwrap();
            let ch = channel(f_thz * 1e12);
            let (spm, xpm) = isrs_closed_form_terms(&fiber, &s, ch, c_r).unwrap();
            let eta = isrs_closed_form_eta(&fiber, &s, ch, c_r, 0.0).unwrap();
            println!(
                "{curve} {f_thz:+.2} THz (channel {ch} at {:+.4} THz): SPM {:.2} dB, XPM {:.2} dB, eta_1 {:.2} dB",
                s.freqs_hz[ch] / 1e12,
                lin2db(spm),
                lin2db(xpm),
                lin2db(eta)
            );
            card.check(&inputs, lin2db(eta));
        }
    }
    card.finish().unwrap();
}

#[test]
fn the_change_of_eta_1_by_isrs_follows_the_dashed_curves_of_fig_4b() {
    let fiber = table_i(1);
    let l_eff = fiber.effective_length_km();
    let mut card = Card::load("semrau2019_fig4b").unwrap();
    for inputs in card.inputs_matching(&json!({})) {
        let f_ghz = inputs["channel_ghz"].as_f64().unwrap();
        let delta_rho_db = inputs["delta_rho_db"].as_f64().unwrap();
        let ch = channel(f_ghz * 1e9);
        // eq. (3): the total launch power of the power transfer
        let p_tot = delta_rho_db / (4.3 * C_R * l_eff * B_TOT_THZ);
        let with = isrs_closed_form_eta(&fiber, &spectrum(p_tot / N_CH as f64), ch, C_R, 0.0).unwrap();
        let without = isrs_closed_form_eta(&fiber, &spectrum(1e-3), ch, 0.0, 0.0).unwrap();
        let delta = lin2db(with / without);
        println!(
            "channel {ch} at {:+.1} GHz, delta_rho {delta_rho_db} dB (P_tot {:.1} dBm): {delta:+.2} dB",
            (ch as f64 - 125.0) * SPACING / 1e9,
            comnum::math::w2dbm(p_tot)
        );
        card.check(&inputs, delta);
    }
    card.finish().unwrap();
}

#[test]
fn the_accumulation_over_six_spans_follows_the_dashed_curves_of_fig_5() {
    let fiber = table_i(6);
    let eps = coherence_factor(&fiber, RS * (1.0 + ROLLOFF));
    println!("coherence factor of Poggiolini 2012 eq. (22) over 40.004 GHz, 6 spans: {eps:.4} (the paper: 0.15)");
    let s = spectrum(1e-3);
    let mut card = Card::load("semrau2019_fig5").unwrap();
    card.check(&json!({"claim": "epsilon"}), eps);
    for (panel, c_r) in [("a", 0.0), ("b", C_R)] {
        let mut gaps = Vec::new();
        for inputs in card.inputs_matching(&json!({"panel": panel, "accumulation": "coherent"})) {
            let f_thz = inputs["f_thz"].as_f64().unwrap();
            let ch = channel(f_thz * 1e12);
            let coherent = lin2db(isrs_closed_form_eta(&fiber, &s, ch, c_r, 0.15).unwrap());
            let incoherent = lin2db(isrs_closed_form_eta(&fiber, &s, ch, c_r, 0.0).unwrap());
            println!("panel {panel}, {f_thz:+.2} THz: coherent {coherent:.2} dB, incoherent {incoherent:.2} dB");
            card.check(&inputs, coherent);
            card.check(&json!({"panel": panel, "accumulation": "incoherent", "f_thz": f_thz}), incoherent);
            gaps.push(coherent - incoherent);
        }
        let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
        println!("panel {panel}: the incoherent form costs {mean:.3} dB on average over the points of the card");
        card.check(&json!({"claim": "incoherent_cost_db", "panel": panel}), mean);
    }
    card.finish().unwrap();
}
