//! DVB-T (`comnum_systems::dvbt`) against ETSI EN 300 744 V1.6.2, *Digital Video
//! Broadcasting (DVB); Framing structure, channel coding and modulation for digital
//! terrestrial television*, 2015, Annex A, Table A.1 (non-hierarchical transmission).
//!
//! Table A.1 gives the C/N at which the BER after the Viterbi decoder is $2 \cdot
//! 10^{-4}$ ("QEF after Reed-Solomon") for QPSK, 16-QAM and 64-QAM at the five code rates,
//! over the Gaussian channel and the fixed (Ricean, F1) and portable (Rayleigh, P1)
//! channels of Annex B, in the 8K mode, "assuming perfect channel estimation and without
//! phase noise", the C/N "calculated over the transmitted 6 817 OFDM carriers". The values
//! are printed in italics: approximate (NOTE 1).
//!
//! The link: bursts of random transport packets through [`Dvbt`] (energy dispersal,
//! RS(204, 188), the outer interleaver, the punctured inner code, the inner interleaver,
//! the constellations of Figure 9, the frame with its pilots and TPS) in the 8K mode, the
//! cells of every carrier multiplied by the gain $H(k'/T_{U})$ of the static channel
//! (`Echoes::dvbt_f1`, `Echoes::dvbt_p1`, Table B.1; 1 for the Gaussian channel, $T_{U}$ =
//! 896 µs), white Gaussian noise of the variance of a carrier at the C/N (the mean power of
//! the carriers, pilots boosted, over the C/N: [`Dvbt::carrier_noise_var`]), exact LLRs
//! with the known gains, soft Viterbi decoding. The guard interval is longer than every
//! echo, so the channel acts carrier by carrier. The BER is counted on the bytes entering
//! the inner coder; the C/N of the crossing is interpolated between points 0.2 dB apart
//! with $10^{6}$ bits at least per point. The values and tolerances are in the card
//! `validation/cards/dvbt_tablea1.yaml`.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]

use comnum::C64;
use comnum::channel::awgn::add_noise;
use comnum::channel::multipath::Echoes;
use comnum::core::rng::Rng;
use comnum::systems::dvb::{self, InnerRate};
use comnum::systems::dvbt::{Dvbt, Modulation};
use comnum::waveform::ofdm::dvbt::{Guard, Mode, elementary_period};
use comnum_validation::card::Card;
use comnum_validation::sweep_crossing;
use serde_json::json;

const MIN_BITS: usize = 1_000_000;
const TARGET: f64 = 2e-4;

/// The gains of the 6817 carriers of the 8K mode in an 8 MHz channel.
fn carrier_gains(echoes: Option<&Echoes>) -> Vec<C64> {
    let tu = 8192.0 * elementary_period(8e6);
    let centre = (6817 - 1) / 2;
    match echoes {
        None => vec![C64::new(1.0, 0.0); 6817],
        Some(e) => {
            let f: Vec<f64> = (0..6817).map(|k| (k as f64 - centre as f64) / tu).collect();
            e.frequency_response(&f)
        }
    }
}

/// The BER after the Viterbi decoder at the C/N `cn_db`.
fn viterbi_ber(link: &Dvbt, h: &[C64], cn_db: f64, rng: &mut Rng) -> f64 {
    let noise_var = link.carrier_noise_var(cn_db);
    let n_packets = 200;
    let (mut errors, mut bits) = (0usize, 0usize);
    while bits < MIN_BITS {
        let tx = dvb::burst_encode(&dvb::random_packets(n_packets, rng)).unwrap();
        let mut cells = link.cells(&tx).unwrap();
        for symbol in &mut cells {
            for (c, g) in symbol.iter_mut().zip(h) {
                *c *= g;
            }
            add_noise(symbol, noise_var, rng);
        }
        let rx = link.inner_decode(&cells, n_packets, noise_var, Some(h)).unwrap();
        errors += tx.iter().zip(&rx).map(|(a, b)| (a ^ b).count_ones() as usize).sum::<usize>();
        bits += 8 * tx.len();
    }
    errors as f64 / bits as f64
}

#[test]
#[ignore = "validation against the standard: run with --release --ignored"]
fn the_required_cn_follows_table_a1_of_en_300_744() {
    let mut card = Card::load("dvbt_tablea1").unwrap();
    let mut rng = Rng::new(744);
    let channels: [(&str, Option<Echoes>); 3] =
        [("gaussian", None), ("f1", Some(Echoes::dvbt_f1())), ("p1", Some(Echoes::dvbt_p1()))];
    for (channel, echoes) in &channels {
        let h = carrier_gains(echoes.as_ref());
        for modulation in [Modulation::Qpsk, Modulation::Qam16, Modulation::Qam64] {
            for rate in InnerRate::ALL {
                let link = Dvbt::new(Mode::K8, Guard::G1_4, modulation, rate).unwrap();
                let inputs = json!({"channel": channel, "constellation": modulation.name(), "rate": rate.name()});
                let table = card.expected(&inputs).unwrap();
                let start = table - 2.0;
                let crossing =
                    sweep_crossing(start, 0.2, 40, TARGET, |_, cn| viterbi_ber(&link, &h, cn, &mut rng)).unwrap();
                println!("{channel} {} {}: C/N {crossing:.2} dB (Table A.1: {table})", modulation.name(), rate.name());
                card.check(&inputs, crossing);
            }
        }
    }
    card.finish().unwrap();
}
