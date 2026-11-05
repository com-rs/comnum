//! Decode the AIS messages of ships in a recording, with the blocks of comnum only: the
//! front end (`comnum::dsp::frontend::select_channel`) brings each of the two channels
//! to baseband, and `comnum::systems::ais::receive` demodulates the GMSK, finds the
//! HDLC frames and decodes the messages of ITU-R M.1371-5.
//!
//! ```text
//! cargo run --release -p comnum --features spec --example ais -- <recording>
//! cargo run --release -p comnum --features spec --example ais
//! ```
//!
//! `<recording>` is a SigMF recording (`.sigmf-meta`, `.sigmf-data`) whose metadata give
//! the sampling rate (`core:sample_rate`) and the tuning frequency (`core:frequency`),
//! for instance `hackrf_transfer -r ais.sigmf-data -f 162000000 -s 2000000 -n 60000000
//! -l 32 -g 32 -a 1` (30 s around 162 MHz, the two channels 25 kHz below and above) with
//! the file `ais.sigmf-meta`
//!
//! ```text
//! {"global": {"core:datatype": "ci8", "core:sample_rate": 2000000, "core:version": "1.2.0"},
//!  "captures": [{"core:sample_start": 0, "core:frequency": 162000000}], "annotations": []}
//! ```
//!
//! Without arguments, the example makes a recording of three packets of a simulated ship
//! on channel A at 10 dB of SNR, tuned 25 kHz below the channel.
//!
//! The example prints, for each channel, the packets received with their `!AIVDM`
//! sentence and the fields of the message (the ships' identities and positions are
//! those of the recording: do not commit a recording of real traffic).

use comnum::dsp::frontend::select_channel;
use comnum::io::read_sigmf;
use comnum::prelude::*;
use comnum::systems::ais::{self, message::Message, nmea};

/// A recording of a simulated ship: (samples, sampling rate, tuning frequency).
fn synthetic() -> Result<(Vec<C64>, f64, f64)> {
    let (fs, tuning) = (2e6, ais::CHANNEL_A_HZ - 25e3);
    let p = ais::message::Position {
        lon_deg: Some(-9.8765),
        lat_deg: Some(46.1234),
        accurate: true,
        sog_knots: Some(8.4),
        cog_deg: Some(271.5),
        heading_deg: Some(270),
        timestamp: 42,
    };
    let messages: Vec<Vec<u8>> = (0..3).map(|k| ais::message::position_report_bits(200_000_000 + k, 0, &p)).collect();
    // the packets at 16 samples per bit, then at 2 MS/s, 25 kHz above the tuning frequency
    let base = ais::transmit(&messages, ais::FRONT_END_RATE)?;
    let up = comnum::dsp::resample::resample_fft(&base, (base.len() as f64 * fs / ais::FRONT_END_RATE) as usize);
    let mut rng = Rng::new(1371);
    let sigma = (fs / ais::CHANNEL_WIDTH_HZ / 10.0).sqrt();
    let y = up
        .iter()
        .enumerate()
        .map(|(n, v)| {
            v * C64::from_polar(1.0, 2.0 * std::f64::consts::PI * 25e3 * n as f64 / fs) + rng.complex_gaussian() * sigma
        })
        .collect();
    Ok((y, fs, tuning))
}

fn describe(m: &Message) -> String {
    match m {
        Message::PositionReport { msg_type, mmsi, nav_status, position, .. } => {
            format!("type {msg_type} position report, MMSI {mmsi}, status {nav_status}, {}", position_text(position))
        }
        Message::BaseStation { mmsi, utc, position, .. } => format!(
            "type 4 base station, MMSI {mmsi}, {:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC, {}",
            utc.0,
            utc.1,
            utc.2,
            utc.3,
            utc.4,
            utc.5,
            position_text(position)
        ),
        Message::StaticAndVoyage { mmsi, shipname, callsign, ship_type, destination, draught_m, .. } => format!(
            "type 5 static data, MMSI {mmsi}, {shipname:?} ({callsign}), ship type {ship_type}, draught {draught_m} m, destination {destination:?}"
        ),
        Message::ClassBPosition { mmsi, position, .. } => {
            format!("type 18 class B position, MMSI {mmsi}, {}", position_text(position))
        }
        Message::ClassBName { mmsi, shipname, .. } => format!("type 24 class B name, MMSI {mmsi}, {shipname:?}"),
        Message::ClassBStatic { mmsi, ship_type, callsign, .. } => {
            format!("type 24 class B static data, MMSI {mmsi}, ship type {ship_type}, call sign {callsign:?}")
        }
        Message::Other { msg_type, mmsi, .. } => format!("type {msg_type}, MMSI {mmsi}"),
    }
}

fn position_text(p: &ais::message::Position) -> String {
    let coord = |v: Option<f64>| v.map_or("n/a".to_string(), |x| format!("{x:.4}"));
    format!(
        "lat {} lon {}, {} kn, course {}, heading {}",
        coord(p.lat_deg),
        coord(p.lon_deg),
        p.sog_knots.map_or("n/a".to_string(), |v| format!("{v:.1}")),
        p.cog_deg.map_or("n/a".to_string(), |v| format!("{v:.1}°")),
        p.heading_deg.map_or("n/a".to_string(), |v| format!("{v}°")),
    )
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (x, fs, tuning) = if let Some(path) = args.first() {
        let r = read_sigmf(path)?;
        let fs = r.sample_rate.ok_or_else(|| Error::Config("the recording has no core:sample_rate".into()))?;
        let fc = r.frequency.ok_or_else(|| Error::Config("the recording has no core:frequency".into()))?;
        (r.channel(0), fs, fc)
    } else {
        synthetic()?
    };
    println!("{:.2} s at {:.3} MS/s, tuned to {:.3} MHz", x.len() as f64 / fs, fs / 1e6, tuning / 1e6);
    let mut total = 0;
    for (name, channel) in [("A", ais::CHANNEL_A_HZ), ("B", ais::CHANNEL_B_HZ)] {
        let offset = channel - tuning;
        if offset.abs() > fs / 2.0 - ais::CHANNEL_WIDTH_HZ {
            println!("channel {name} ({:.3} MHz) is outside the recording", channel / 1e6);
            continue;
        }
        let y = select_channel(&x, fs, offset, ais::FRONT_END_RATE, ais::CHANNEL_WIDTH_HZ / 2.0, 129)?;
        let packets = ais::receive(&y, ais::FRONT_END_RATE)?;
        println!(
            "\nchannel {name} ({:.3} MHz, {:+.0} Hz from the tuning): {} packets",
            channel / 1e6,
            offset,
            packets.len()
        );
        for p in &packets {
            let t = p.start_bit as f64 / ais::BIT_RATE;
            for s in nmea::aivdm(&p.bits, name.chars().next().unwrap_or('A'), 0) {
                println!("  {t:7.3} s  {s}");
            }
            match &p.message {
                Ok(m) => println!("           {}", describe(m)),
                Err(e) => println!("           not decoded: {e}"),
            }
        }
        total += packets.len();
    }
    println!("\n{total} packets with a correct frame check sequence");
    Ok(())
}
