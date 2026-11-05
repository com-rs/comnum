//! Find an LTE cell in a recording and decode its master information block, with the
//! blocks of comnum only: the front end (`comnum::dsp::frontend::select_channel`), the
//! cell search and the PBCH receiver (`comnum::systems::lte::search`).
//!
//! ```text
//! cargo run --release -p comnum --features spec --example lte_mib -- <recording> <cell_hz>
//! cargo run --release -p comnum --features spec --example lte_mib
//! ```
//!
//! `<recording>` is a SigMF recording (`.sigmf-meta`, `.sigmf-data`, any sample format)
//! whose metadata give the sampling rate (`core:sample_rate`) and the tuning frequency
//! (`core:frequency` of the first capture), and `<cell_hz>` the centre of the LTE carrier
//! (Hz; a multiple of 100 kHz). The output of `hackrf_transfer -r capture.sigmf-data -f
//! 775000000 -s 15360000` becomes one with the file `capture.sigmf-meta`
//!
//! ```text
//! {"global": {"core:datatype": "ci8", "core:sample_rate": 15360000, "core:version": "1.2.0"},
//!  "captures": [{"core:sample_start": 0, "core:frequency": 775000000}], "annotations": []}
//! ```
//!
//! Without arguments, the example makes a recording: 20 frames of a 10 MHz cell of two
//! ports at 15.36 MS/s, tuned 2 MHz below the carrier, the oscillator of the receiver
//! 17 ppm off (in frequency and in sampling rate), at 10 dB of SNR.
//!
//! The example prints the cell, the frequency offset, the MIB of every frame, and the
//! error of the oscillator of the receiver twice: from the frequency offset of the carrier
//! and from the samples it counts per frame (one oscillator drives the mixer and the
//! converter of a software radio, so that the two agree).

use comnum::dsp::frontend::select_channel;
use comnum::io::read_sigmf;
use comnum::prelude::*;
use comnum::systems::lte::cell::Downlink;
use comnum::systems::lte::mib::{Mib, PhichDuration, PhichResource};
use comnum::systems::lte::search::{SEARCH_RATE, SearchConfig, cell_search, decode_mib};

/// A recording of a synthetic cell: (samples, sampling rate, tuning frequency, carrier).
fn synthetic() -> Result<(Vec<C64>, f64, f64, f64)> {
    let (fs, carrier, tuning, ppm) = (15.36e6, 806e6, 804e6, -17e-6);
    let mib = Mib {
        n_rb: 50,
        phich_duration: PhichDuration::Normal,
        phich_resource: PhichResource::OneSixth,
        sfn_msb: 0,
        rest: [0; 10],
    };
    let cell = Downlink { n_cell_id: 123, mib, ports: 2 };
    let mut rng = Rng::new(806);
    let h = [C64::new(0.8, 0.3), C64::new(-0.2, 0.5)];
    let mut x = Vec::new();
    for sfn in 700..720 {
        let ports = cell.frame_samples(sfn, 1024, Some(&mut rng))?;
        x.extend((0..ports[0].len()).map(|n| ports[0][n] * h[0] + ports[1][n] * h[1]));
    }
    // the receiver samples at fs (1 + ppm) and mixes with tuning (1 + ppm): the carrier
    // lands at carrier - tuning (1 + ppm), its samples read with a linear interpolation
    let n_out = (x.len() as f64 * (1.0 + ppm)) as usize - 1;
    let offset = carrier - tuning * (1.0 + ppm);
    let power = x.iter().map(C64::norm_sqr).sum::<f64>() / x.len() as f64;
    let sigma = (power / 10.0).sqrt();
    let y = (0..n_out)
        .map(|m| {
            let t = m as f64 / (1.0 + ppm);
            let (i, f) = (t.floor() as usize, t - t.floor());
            let v = x[i] * (1.0 - f) + x[i + 1] * f;
            let phase = 2.0 * std::f64::consts::PI * offset * m as f64 / (fs * (1.0 + ppm));
            v * C64::from_polar(1.0, phase) + rng.complex_gaussian() * sigma
        })
        .collect();
    // the receiver believes that it samples at fs, tuned to `tuning`
    Ok((y, fs, tuning, carrier))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (x, fs, tuning, carrier) = if args.len() >= 2 {
        let r = read_sigmf(&args[0])?;
        let fs = r.sample_rate.ok_or_else(|| Error::Config("the recording has no core:sample_rate".into()))?;
        let fc = r.frequency.ok_or_else(|| Error::Config("the capture has no core:frequency".into()))?;
        let carrier: f64 = args[1].parse().map_err(|_| Error::Config(format!("not a frequency: {}", args[1])))?;
        (r.channel(0), fs, fc, carrier)
    } else {
        println!("no recording given: a synthetic one (cell 123, 10 MHz, 2 ports, 17 ppm, 10 dB)");
        synthetic()?
    };
    println!(
        "{} samples at {:.3} MS/s, tuned to {:.3} MHz, carrier at {:.3} MHz",
        x.len(),
        fs / 1e6,
        tuning / 1e6,
        carrier / 1e6
    );
    let y = select_channel(&x, fs, carrier - tuning, SEARCH_RATE, 0.6e6, 129)?;
    let cells = cell_search(&y, &SearchConfig { max_cfo_hz: 25e3, ..SearchConfig::default() })?;
    if cells.is_empty() {
        println!("no cell found");
        return Ok(());
    }
    for cell in &cells {
        println!(
            "\ncell {} (PSS {:.2}, SSS {:.2}): offset {:+.0} Hz ({:+.2} ppm of the carrier), first frame at sample {}",
            cell.n_cell_id,
            cell.pss_metric,
            cell.sss_metric,
            cell.cfo_hz,
            1e6 * cell.cfo_hz / carrier,
            cell.frame_start
        );
        let frames = decode_mib(&y, cell, 4)?;
        // the frames whose subframe 0 (1920 samples) is in the capture
        let total = (y.len() - cell.frame_start - 1920) / 19_200 + 1;
        println!("MIB decoded in {} of {total} frames", frames.len());
        for f in frames.iter().take(4) {
            println!(
                "  SFN {:4}  {} ports  {} RB  PHICH {:?}, Ng = {:.3}",
                f.sfn,
                f.ports,
                f.mib.n_rb,
                f.mib.phich_duration,
                f.mib.phich_resource.ng()
            );
        }
        if let (Some(a), Some(b)) = (frames.first(), frames.last()) {
            let consistent = frames
                .iter()
                .all(|f| (f.sfn + 1024 - a.sfn) % 1024 == ((f.start - a.start) as f64 / 19_200.0).round() as usize);
            println!("  the frame numbers follow the frames: {consistent}");
            if b.start > a.start {
                // the samples per frame against the 19 200 of 10 ms at 1.92 MS/s
                let per_frame = (b.start - a.start) as f64 / ((b.sfn + 1024 - a.sfn) % 1024) as f64;
                // a receiver whose oscillator runs ε fast counts 19 200 (1 + ε) samples per
                // frame and sees the carrier ε f_c below where it is
                println!(
                    "  the oscillator of the receiver: {:+.1} ppm from its {per_frame:.2} samples per frame, {:+.1} ppm from the carrier offset",
                    1e6 * (per_frame / 19_200.0 - 1.0),
                    -1e6 * cell.cfo_hz / carrier
                );
            }
        }
    }
    Ok(())
}
