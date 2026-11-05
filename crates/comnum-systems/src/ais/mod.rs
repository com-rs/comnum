//! AIS, the automatic identification system of ships (Recommendation ITU-R M.1371-5):
//! the transmission of Annex 2 (GMSK at 9600 bit/s with NRZI, HDLC framing) on the two
//! channels AIS 1 and AIS 2, a receiver for a capture, the messages of Annex 8
//! ([`message`]) and the `!AIVDM` sentences of NMEA 0183 ([`nmea`]).
//!
//! | Module | Content |
//! |---|---|
//! | [`message`] | the fields of the position reports (1, 2, 3, 18), the base station report (4), the static and voyage data (5), the class B static data (24) |
//! | [`nmea`] | the six-bit armouring, the checksum and the fragments of the `!AIVDM` sentences |
//!
//! # Model
//!
//! A transmission packet (Annex 2, Sec. 3.3.2) is a training sequence of 24 alternating
//! bits, the HDLC frame of the message (the flag `01111110`, the bytes of the message
//! least significant bit first with bit stuffing, the 16-bit frame check sequence, the
//! flag: [`crate::hdlc`]) and a buffer; the bits are NRZI coded and modulate a
//! GMSK carrier at 9600 bit/s with $BT = 0.4$ (the receiver's filter has $BT = 0.5$)
//! and a modulation index of $1/2$, that is a frequency deviation of ±2400 Hz
//! (Sec. 3.3.3). The channels are AIS 1 at 161.975 MHz and AIS 2 at 162.025 MHz, 25 kHz
//! wide (Sec. 2.1). The message bits are the bytes of the frame most significant bit
//! first; [`message::decode`] reads them.
//!
//! The receiver of a channel at baseband ([`demodulate`]): a low-pass on the samples,
//! the instantaneous frequency by the polar discriminator
//! ([`comnum_dsp::fm::fm_discriminator`]), an integrator over half a bit, the frequency
//! offset of the transmitter removed by a moving average, a bit clock recovered from
//! the zero crossings ([`clock_from_zero_crossings`]: an oscillator of period one bit
//! pulled towards every transition, as the receivers of the open-source decoders do),
//! the sign of the sample at the middle of the bit; then NRZI decoding and the HDLC
//! frames ([`receive`]). The Gaussian filter of the receiver of the standard ($BT =
//! 0.5$) is not used after the discriminator: on the module's test it costs 1 dB. The channel is brought to baseband by
//! [`comnum_dsp::frontend::select_channel`] at [`FRONT_END_RATE`].
//!
//! # Examples
//! ```
//! use comnum_systems::ais::{self, message::{Position, position_report_bits}};
//! let p = Position { lon_deg: Some(-9.88), lat_deg: Some(46.12), accurate: true, sog_knots: Some(8.0), cog_deg: Some(90.0), heading_deg: Some(92), timestamp: 10 };
//! let bits = position_report_bits(200000001, 0, &p);
//! let x = ais::transmit(&[bits.clone()], ais::FRONT_END_RATE).unwrap();
//! let received = ais::receive(&x, ais::FRONT_END_RATE).unwrap();
//! assert_eq!(received.len(), 1);
//! assert_eq!(received[0].bits, bits);
//! assert_eq!(received[0].message.as_ref().unwrap().mmsi(), 200000001);
//! ```
//!
//! # References
//!
//! 1. Recommendation ITU-R M.1371-5, *Technical characteristics for an automatic
//!    identification system using time division multiple access in the VHF maritime
//!    mobile band*, 2014: Annex 2, Secs. 2.1 (channels), 3.3.2 (packet format, NRZI,
//!    bit stuffing), 3.3.3 (modulation: GMSK, 9600 bit/s, $BT$ 0.4 / 0.5, index 0.5);
//!    Annex 8 (messages).
//! 2. IEC 61162-1:2016, the `VDM` sentence.

pub mod message;
pub mod nmea;

use crate::hdlc;
use comnum_core::{Result, ensure, ensure_shape};
use comnum_dsp::filter::{LowPass, fir_filter, low_pass_fir};
use comnum_dsp::fm::fm_discriminator;
use comnum_math::C64;
use comnum_waveform::cpm::CpmConfig;
use message::Message;

/// The bit rate (bit/s).
pub const BIT_RATE: f64 = 9600.0;
/// The channel AIS 1 (Hz).
pub const CHANNEL_A_HZ: f64 = 161.975e6;
/// The channel AIS 2 (Hz).
pub const CHANNEL_B_HZ: f64 = 162.025e6;
/// The width of a channel (Hz).
pub const CHANNEL_WIDTH_HZ: f64 = 25e3;
/// $BT$ of the transmitter's Gaussian filter.
pub const BT_TX: f64 = 0.4;
/// The frequency deviation of the GMSK, $h R_{b}/2$ (Hz).
pub const DEVIATION_HZ: f64 = BIT_RATE / 4.0;
/// The training sequence: 24 alternating bits.
pub const TRAINING_BITS: usize = 24;
/// The sampling rate of a channel at baseband for [`receive`]: 16 samples per bit.
pub const FRONT_END_RATE: f64 = 16.0 * BIT_RATE;

/// The samples per bit of `fs`: an even integer of 4 at least.
fn samples_per_bit(fs: f64) -> Result<usize> {
    let sps = fs / BIT_RATE;
    ensure(sps >= 4.0 && (sps - sps.round()).abs() < 1e-9 && sps.round() as usize % 2 == 0, || {
        format!("the sampling rate must be an even multiple of {BIT_RATE} Hz of 4 samples per bit at least, got {fs}")
    })?;
    Ok(sps.round() as usize)
}

/// The bytes of the message `bits` (most significant bit first), the last byte padded
/// with zeros.
pub fn message_bytes(bits: &[u8]) -> Vec<u8> {
    bits.chunks(8).map(|c| c.iter().enumerate().fold(0u8, |b, (i, &v)| b | ((v & 1) << (7 - i)))).collect()
}

/// The message bits of the bytes of a frame, most significant bit first.
pub fn message_bits(bytes: &[u8]) -> Vec<u8> {
    bytes.iter().flat_map(|&b| (0..8).rev().map(move |i| (b >> i) & 1)).collect()
}

/// The NRZI line levels of the packet of one message: the training sequence, the HDLC
/// frame of its bytes, and 8 bits of buffer.
pub fn packet_levels(bits: &[u8]) -> Vec<u8> {
    let mut data: Vec<u8> = (0..TRAINING_BITS).map(|i| (i % 2) as u8).collect();
    data.extend(hdlc::frame(&message_bytes(bits)));
    data.extend([1u8; 8]);
    hdlc::nrzi_encode(&data, 0)
}

/// The GMSK signal of the packets of `messages` (bit strings), one after the other with
/// a silence of 32 bits before each, at `fs` (an even multiple of 9600 Hz): unit
/// envelope during a packet.
///
/// # Errors
/// [`comnum_core::Error::Config`] for a sampling rate that is not an even multiple of
/// the bit rate of 4 samples per bit at least.
pub fn transmit(messages: &[Vec<u8>], fs: f64) -> Result<Vec<C64>> {
    let sps = samples_per_bit(fs)?;
    let cpm = CpmConfig::gmsk(BT_TX, 3, sps).build()?;
    let mut out = Vec::new();
    for m in messages {
        out.extend(vec![C64::new(0.0, 0.0); 32 * sps]);
        let symbols: Vec<usize> = packet_levels(m).iter().map(|&l| usize::from(l & 1)).collect();
        out.extend(cpm.modulate(&symbols)?);
    }
    out.extend(vec![C64::new(0.0, 0.0); 32 * sps]);
    Ok(out)
}

/// The bits of a baseband data signal `x` at `sps` samples per bit by a clock recovered
/// from its zero crossings: a numerically controlled oscillator of period one bit whose
/// phase is pulled by `alpha` towards every zero crossing (the transitions of the data,
/// nominally at phase 0), the sign of the sample taken at phase one half.
pub fn clock_from_zero_crossings(x: &[f64], sps: f64, alpha: f64) -> Vec<u8> {
    let mut bits = Vec::with_capacity(x.len() / sps as usize + 1);
    let mut phase = 0.0;
    let mut sampled = false;
    for w in x.windows(2) {
        phase += 1.0 / sps;
        if phase >= 1.0 {
            phase -= 1.0;
            sampled = false;
        }
        if (w[0] > 0.0) != (w[1] > 0.0) {
            let e = if phase < 0.5 { phase } else { phase - 1.0 };
            phase -= alpha * e;
            if phase < 0.0 {
                phase += 1.0;
                sampled = true;
            }
        }
        if !sampled && phase >= 0.5 {
            bits.push(u8::from(w[1] > 0.0));
            sampled = true;
        }
    }
    bits
}

/// The line levels of a channel at baseband (`x` at `fs`, an even multiple of 9600 Hz:
/// [`FRONT_END_RATE`]), one per bit, by the receiver of the module: a low-pass of
/// 6 kHz (64 taps) on the samples, the polar discriminator bounded to twice the
/// deviation, an integrator over half a bit, the frequency offset of the transmitter
/// removed by a moving average over 256 bits weighted by the power of the signal (the
/// silence between the packets does not count), and the clock of
/// [`clock_from_zero_crossings`] with a gain of 0.3. Over a 25 kHz channel of white
/// noise, every packet of the module's test is received at 15 dB of SNR and 9 of 10 at
/// 10 dB.
///
/// # Errors
/// [`comnum_core::Error::Config`] for the sampling rate, [`comnum_core::Error::Shape`]
/// for fewer than 64 bits of signal.
pub fn demodulate(x: &[C64], fs: f64) -> Result<Vec<u8>> {
    let sps = samples_per_bit(fs)?;
    ensure_shape(x.len() >= 64 * sps, || format!("{} samples is less than 64 bits at {fs} Hz", x.len()))?;
    let x = fir_filter(&low_pass_fir(6e3, fs, 64, LowPass::Rect), x);
    let freq: Vec<f64> =
        fm_discriminator(&x, fs).iter().map(|v| v.clamp(-2.0 * DEVIATION_HZ, 2.0 * DEVIATION_HZ)).collect();
    let boxcar = vec![2.0 / sps as f64; sps / 2];
    let filtered: Vec<f64> =
        fir_filter(&boxcar, &freq.iter().map(|&v| C64::new(v, 0.0)).collect::<Vec<_>>()).iter().map(|v| v.re).collect();
    let power: Vec<f64> = x.iter().map(C64::norm_sqr).collect();
    let window = 256 * sps;
    let mut centred = Vec::with_capacity(filtered.len());
    let (mut acc, mut weight) = (0.0, 0.0);
    for (i, v) in filtered.iter().enumerate() {
        acc += v * power[i];
        weight += power[i];
        if i >= window {
            acc -= filtered[i - window] * power[i - window];
            weight -= power[i - window];
        }
        centred.push(v - if weight > 0.0 { acc / weight } else { 0.0 });
    }
    Ok(clock_from_zero_crossings(&centred, sps as f64, 0.3))
}

/// A received packet.
#[derive(Clone, Debug, PartialEq)]
pub struct Received {
    /// The bit of the line where the opening flag starts.
    pub start_bit: usize,
    /// The message bits (most significant bit first; the padding of the last byte
    /// included).
    pub bits: Vec<u8>,
    /// The decoded message, or why it could not be decoded.
    pub message: std::result::Result<Message, String>,
}

/// The packets of a channel at baseband: [`demodulate`], NRZI decoding, the HDLC frames
/// whose frame check sequence is correct ([`hdlc::frames`]) with at least the 38 bits
/// of a message header, decoded by [`message::decode`].
///
/// # Errors
/// As [`demodulate`].
pub fn receive(x: &[C64], fs: f64) -> Result<Vec<Received>> {
    let levels = demodulate(x, fs)?;
    Ok(packets(&hdlc::nrzi_decode(&levels)))
}

/// The packets of a stream of NRZI-decoded bits (see [`receive`]).
pub fn packets(bits: &[u8]) -> Vec<Received> {
    hdlc::frames_of_at_least(bits, 5)
        .into_iter()
        .filter(|f| f.fcs_ok)
        .map(|f| {
            let bits = message_bits(&f.payload);
            let message = message::decode(&bits).map_err(|e| e.to_string());
            Received { start_bit: f.start, bits, message }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use comnum_core::rng::Rng;
    use message::{Position, position_report_bits};

    fn report(mmsi: u32) -> Vec<u8> {
        let p = Position {
            lon_deg: Some(-9.8765),
            lat_deg: Some(46.1234),
            accurate: true,
            sog_knots: Some(12.3),
            cog_deg: Some(271.5),
            heading_deg: Some(270),
            timestamp: 42,
        };
        position_report_bits(mmsi, 0, &p)
    }

    #[test]
    fn message_bits_and_bytes_round_trip() {
        let bits = report(200_000_001);
        assert_eq!(message_bits(&message_bytes(&bits)), bits);
        assert_eq!(message_bytes(&bits).len(), 21);
    }

    #[test]
    fn packets_survive_noise_a_frequency_offset_and_either_polarity() {
        let fs = FRONT_END_RATE;
        let messages = vec![report(200_000_001), report(200_000_002), report(200_000_003)];
        let x = transmit(&messages, fs).unwrap();
        let mut rng = Rng::new(1371);
        // 15 dB of SNR in the 25 kHz channel, 300 Hz of offset, inverted spectrum
        let sigma = (fs / CHANNEL_WIDTH_HZ / 10f64.powf(1.5)).sqrt();
        let y: Vec<C64> = x
            .iter()
            .enumerate()
            .map(|(n, v)| {
                let offset = C64::from_polar(1.0, 2.0 * std::f64::consts::PI * 300.0 * n as f64 / fs);
                (v * offset + rng.complex_gaussian() * sigma).conj()
            })
            .collect();
        let r = receive(&y, fs).unwrap();
        let got: Vec<u32> = r.iter().filter_map(|p| p.message.as_ref().ok().map(Message::mmsi)).collect();
        assert_eq!(got, vec![200_000_001, 200_000_002, 200_000_003]);
        assert_eq!(r[0].bits, messages[0]);
        assert!(receive(&x[..100], fs).is_err());
        assert!(transmit(&messages, 50e3).is_err());
        // at 10 dB, 9 of 10 packets of random bits over 100 (the sensitivity of the standard: 8 of 10)
        let mut rng = Rng::new(10);
        let random: Vec<Vec<u8>> = (0..100).map(|_| (0..168).map(|_| rng.below(2) as u8).collect()).collect();
        let x = transmit(&random, fs).unwrap();
        let sigma = (fs / CHANNEL_WIDTH_HZ / 10.0).sqrt();
        let y: Vec<C64> = x.iter().map(|v| v + rng.complex_gaussian() * sigma).collect();
        let ok = receive(&y, fs).unwrap().iter().filter(|p| random.contains(&p.bits)).count();
        println!("packets received at 10 dB: {ok} of 100");
        assert!(ok >= 85, "{ok} of 100 packets at 10 dB");
    }
}
