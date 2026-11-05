//! FM sound broadcasting: the stereophonic multiplex of ITU-R BS.450-4 and the radio
//! data system (RDS) of its 57 kHz subcarrier.
//!
//! | Module | Content |
//! |---|---|
//! | [`mpx`] | the multiplex of the pilot-tone system: transmitter and stereo decoder, the pilot, the levels of BS.450 |
//! | [`rds`] | the RDS: biphase symbols and their demodulator, the block code and its offsets, the groups, the programme service name, the radiotext and the clock time |
//!
//! A receiver brings the station to baseband ([`comnum_dsp::frontend::select_channel`],
//! at [`mpx::MPX_RATE`]), takes its instantaneous frequency
//! ([`comnum_dsp::fm::fm_discriminator`]), then decodes the audio
//! ([`mpx::stereo_decode`]) and the data ([`rds::receive`]).
//!
//! # Examples
//! ```
//! use comnum_dsp::fm::{fm_discriminator, fm_modulate, Emphasis};
//! use comnum_systems::fm::{mpx, rds};
//! let mut bits = Vec::new();
//! for seg in 0..4u16 {
//!     let b4 = u16::from_be_bytes([b"RADIO 95"[2 * seg as usize], b"RADIO 95"[2 * seg as usize + 1]]);
//!     bits.extend(rds::group_bits([0xF201, seg, 0xE0CD, b4], false));
//! }
//! let d = rds::differential_encode(&bits.repeat(4));
//! let n = (0.5 * mpx::MPX_RATE) as usize;
//! let silence = vec![0.0; n];
//! let m = mpx::multiplex(&silence, &silence, Some(&d), mpx::MPX_RATE, &mpx::Levels::default(), Emphasis::EUROPE)?;
//! let x = fm_modulate(&m, mpx::MPX_RATE, 1.0);              // the multiplex in Hz of deviation
//! let (_, station) = rds::receive(&fm_discriminator(&x, mpx::MPX_RATE), mpx::MPX_RATE)?;
//! assert_eq!((station.pi, station.ps.as_str()), (Some(0xF201), "RADIO 95"));
//! # Ok::<(), comnum_core::Error>(())
//! ```
//!
//! # References
//!
//! 1. Recommendation ITU-R BS.450-4, *Transmission standards for FM sound broadcasting
//!    at VHF*, 2019.
//! 2. NRSC-4, *United States RBDS Standard*, April 1998 (with EN 50067).

pub mod mpx;
pub mod rds;

#[cfg(test)]
mod tests {
    use super::{mpx, rds};
    use comnum_core::rng::Rng;
    use comnum_dsp::fm::{Emphasis, fm_discriminator, fm_modulate};
    use comnum_dsp::frontend::select_channel;
    use comnum_math::C64;
    use std::f64::consts::PI;

    /// The groups of a station: its name (0A), a radiotext (2A) and the clock (4A).
    fn station_bits(pi: u16, ps: [u8; 8], text: &[u8; 16]) -> Vec<u8> {
        let mut bits = Vec::new();
        for seg in 0..4u16 {
            let b4 = u16::from_be_bytes([ps[2 * seg as usize], ps[2 * seg as usize + 1]]);
            bits.extend(rds::group_bits([pi, (1 << 10) | (10 << 5) | seg, 0xE0CD, b4], false));
        }
        for seg in 0..4u16 {
            let c = &text[4 * seg as usize..4 * seg as usize + 4];
            let b2 = (2 << 12) | (10 << 5) | seg;
            bits.extend(rds::group_bits(
                [pi, b2, u16::from_be_bytes([c[0], c[1]]), u16::from_be_bytes([c[2], c[3]])],
                false,
            ));
        }
        // 61347 = 3 November 2026, 21:07 UTC, +1 h
        let b2 = (4 << 12) | (10 << 5) | (61_347u32 >> 15) as u16;
        let b3 = (((0xEFA3_u32 & 0x7FFF) as u16) << 1) | 1;
        bits.extend(rds::group_bits([pi, b2, b3, (0b0101 << 12) | (7 << 6) | 2], false));
        bits
    }

    #[test]
    fn a_station_through_a_software_radio_gives_its_audio_and_its_data_back() {
        let fs_mpx = mpx::MPX_RATE;
        let n = (1.5 * fs_mpx) as usize;
        let left: Vec<f64> = (0..n).map(|i| 0.6 * (2.0 * PI * 1e3 * i as f64 / fs_mpx).sin()).collect();
        let right: Vec<f64> = (0..n).map(|i| 0.3 * (2.0 * PI * 2.5e3 * i as f64 / fs_mpx).sin()).collect();
        let d = rds::differential_encode(&station_bits(0xF20C, *b"COMNUM 1", b"Hello from comnu"));
        let m = mpx::multiplex(&left, &right, Some(&d), fs_mpx, &mpx::Levels::default(), Emphasis::EUROPE).unwrap();
        // the carrier 300 kHz above the tuning (2.4 MS/s, by 2.4e6/228e3 = 200/19), 2 kHz
        // off, at a CNR of 25 dB in 200 kHz
        let up: Vec<C64> = fm_modulate(&m, fs_mpx, 1.0);
        let x0 = comnum_dsp::frontend::resample_rational(&up, fs_mpx, 2.4e6, 110e3, 129).unwrap();
        let mut rng = Rng::new(959);
        let sigma = (1.0f64 / 10f64.powf(2.5) * 2.4e6 / 200e3 / 2.0).sqrt();
        let x: Vec<C64> = x0
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v * C64::from_polar(1.0, 2.0 * PI * 302e3 * i as f64 / 2.4e6)
                    + C64::new(rng.gaussian(), rng.gaussian()) * sigma
            })
            .collect();
        let y = select_channel(&x, 2.4e6, 300e3, fs_mpx, 110e3, 129).unwrap();
        let f = fm_discriminator(&y, fs_mpx);
        let (groups, st) = rds::receive(&f, fs_mpx).unwrap();
        assert!(groups.len() >= 12, "{} groups", groups.len());
        assert_eq!((st.pi, st.ps.as_str(), st.pty), (Some(0xF20C), "COMNUM 1", Some(10)));
        assert_eq!(st.radiotext, "Hello from comnu");
        let ct = st.clock.unwrap();
        assert_eq!((ct.year, ct.month, ct.day, ct.hour, ct.minute), (2026, 11, 3, 21, 7));
        assert!(st.valid_blocks.0 * 100 >= 95 * st.valid_blocks.1, "{:?}", st.valid_blocks);
        // the audio, the 2 kHz offset removed by the mean of the discriminator
        let mean = f.iter().sum::<f64>() / f.len() as f64;
        let f0: Vec<f64> = f.iter().map(|v| v - mean).collect();
        let a = mpx::stereo_decode(&f0, fs_mpx, Emphasis::EUROPE, 48e3).unwrap();
        assert!(a.stereo);
        let rms = |x: &[f64]| (x[24_000..48_000].iter().map(|v| v * v).sum::<f64>() / 24_000.0).sqrt();
        assert!((rms(&a.left) / (0.6 / 2f64.sqrt()) - 1.0).abs() < 0.05, "{}", rms(&a.left));
        assert!((rms(&a.right) / (0.3 / 2f64.sqrt()) - 1.0).abs() < 0.05, "{}", rms(&a.right));
    }
}
