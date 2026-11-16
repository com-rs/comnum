//! A text as the source of a link, and orthogonal FSK: the chain of a message played
//! on a sound card.
//!
//! `text_source` replaces the random symbols of `source` by a text: the bits of its
//! UTF-8 bytes, or of the HDLC frame that carries them
//! ([`comnum_systems::hdlc::frame`]: flags, a check sequence and bit stuffing, which a
//! receiver finds in a stream of bits with `comnum.hdlc_frames`), mapped to the points
//! of a constellation, the bits of a point being the binary digits of its index.
//! `fsk_tx` replaces the symbols by the tones of [`comnum_waveform::fsk`], the index of
//! a point choosing its tone, and `fsk_rx` decides them back, aligned on the symbols.

use super::{Block, ConstellationParam, Extension, Frame, parse_constellation};
use comnum_core::rng::Rng;
use comnum_core::{Result, ensure};
use comnum_mapping::constellation::Constellation;
use comnum_math::bits::{bits_to_indices, text_to_bits};
use comnum_systems::hdlc;
use comnum_waveform::fsk::Fsk;

/// The blocks of this module: their documentation and their builders.
pub(super) fn blocks() -> Vec<Extension> {
    vec![
        builtin!(
            Source, start "text_source",
            "A text as the bits and symbols of a block: its UTF-8 bytes in an HDLC frame (or bare), the same in every trial.",
            TextSourceSpec,
            needs [],
            gives [Bits, Symbols, Constellation, Samples],
            |q, _, _| Ok(Box::new(TextSource::new(&q)?)),
        ),
        builtin!(
            Waveform, "fsk_tx",
            "Orthogonal FSK modulator: the index of every symbol of the constellation chooses one of its M tones, a unit-envelope signal at sps samples per symbol.",
            FskSpec,
            needs [Samples, Constellation],
            gives [],
            |q, _, _| Ok(Box::new(FskTx(q))),
        ),
        builtin!(
            Waveform, "fsk_rx",
            "Noncoherent FSK detector, aligned on the symbols: the tone of largest energy, back to the symbols of the constellation.",
            FskSpec,
            needs [Samples, Constellation],
            gives [],
            |q, _, _| Ok(Box::new(FskRx(q))),
        ),
    ]
}

block_params! {
    struct TextSourceSpec {
        text: String ["Hello, world!"], "the message (UTF-8)";
        constellation: ConstellationParam ["4-PAM"],
            "the constellation the bits are mapped to, the bits of a point being the binary digits of its index (for fsk_tx only its order matters: `4-PAM` is four tones)";
        framing: String ["hdlc"] one_of ["hdlc", "none"],
            "hdlc: the bytes in an HDLC frame (flags, CRC-16, bit stuffing), found in a stream of bits by its flags; none: the bits of the bytes, most significant first";
        flags: usize [4], "flags sent before the frame (hdlc), for the receiver to settle";
        symbol_rate: f64 [1.0] "Bd" > 0, "symbol rate; 1 keeps a normalised time axis";
    }
}

struct TextSource {
    c: Constellation,
    bits: Vec<u8>,
    indices: Vec<usize>,
    symbol_rate: f64,
}

impl TextSource {
    fn new(p: &TextSourceSpec) -> Result<TextSource> {
        let c = parse_constellation(p.constellation.json())?;
        let mut bits = if p.framing == "hdlc" {
            let mut bits = hdlc::FLAG.repeat(p.flags);
            bits.extend(hdlc::frame(p.text.as_bytes()));
            bits
        } else {
            text_to_bits(&p.text)
        };
        ensure(!bits.is_empty(), || "text_source: an empty text without framing has no bit".into())?;
        // zeros up to a whole number of symbols
        let per_symbol = c.bits_per_symbol();
        bits.resize(bits.len().div_ceil(per_symbol) * per_symbol, 0);
        let indices = bits_to_indices(&bits, per_symbol)?;
        Ok(TextSource { c, bits, indices, symbol_rate: p.symbol_rate })
    }
}

impl Block for TextSource {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let symbols = self.c.symbols(&self.indices);
        *s = Frame {
            trial: s.trial,
            traces: s.traces.take(),
            bits: self.bits.clone(),
            indices: self.indices.clone(),
            samples: symbols.clone(),
            excluded: vec![false; symbols.len()],
            symbols,
            sps: 1.0,
            symbol_rate: self.symbol_rate,
            constellation: Some(self.c.clone()),
            ..Frame::default()
        };
        Ok(())
    }
}

block_params! {
    struct FskSpec {
        sps: usize [16] >= 2, "samples per symbol";
        spacing: usize [1] >= 1, "cycles per symbol between two adjacent tones (M · spacing ≤ sps)";
    }
}

impl FskSpec {
    /// The modem of the order of the constellation `c`.
    fn fsk(&self, c: &Constellation, block: &str) -> Result<Fsk> {
        Fsk::new(c.order(), self.sps, self.spacing).map_err(|e| e.context(block))
    }
}

struct FskTx(FskSpec);

impl Block for FskTx {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        s.require_one_sps("fsk_tx")?;
        let c = s.constellation("fsk_tx")?;
        let fsk = self.0.fsk(c, "fsk_tx")?;
        s.samples = fsk.modulate(&c.decide(&s.samples))?;
        s.sps = self.0.sps as f64;
        Ok(())
    }
}

struct FskRx(FskSpec);

impl Block for FskRx {
    fn apply(&self, s: &mut Frame, _: &mut Rng) -> Result<()> {
        let c = s.constellation("fsk_rx")?;
        ensure(s.sps == self.0.sps as f64, || {
            format!("fsk_rx: expects {} samples per symbol (as fsk_tx), the signal has {}", self.0.sps, s.sps)
        })?;
        let fsk = self.0.fsk(c, "fsk_rx")?;
        s.samples = c.symbols(&fsk.demodulate(&s.samples)?);
        s.sps = 1.0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::{BuildContext, build};
    use comnum_core::rng::Rng;
    use comnum_systems::hdlc;
    use serde_json::json;

    #[test]
    fn a_text_goes_through_fsk_and_comes_back_as_its_frame() {
        let ctx = BuildContext::default();
        let mut rng = Rng::new(1);
        let mut s = super::Frame::default();
        build(&json!({"type": "text_source", "text": "été ✓"}), &ctx).unwrap().apply(&mut s, &mut rng).unwrap();
        // four flags, the frame, zeros to a whole number of symbols of two bits
        let frame = hdlc::frame("été ✓".as_bytes());
        assert_eq!(s.bits[..32], hdlc::FLAG.repeat(4));
        assert_eq!(s.bits[32..32 + frame.len()], frame);
        assert_eq!(s.samples.len(), (32 + frame.len()).div_ceil(2));
        assert_eq!(s.indices[0], 1); // the bits 0, 1 of the flag
        let sent = s.samples.clone();
        build(&json!({"type": "fsk_tx", "sps": 24, "spacing": 2}), &ctx).unwrap().apply(&mut s, &mut rng).unwrap();
        assert_eq!((s.samples.len(), s.sps), (sent.len() * 24, 24.0));
        assert!(s.samples.iter().all(|v| (v.norm() - 1.0).abs() < 1e-12));
        build(&json!({"type": "fsk_rx", "sps": 24, "spacing": 2}), &ctx).unwrap().apply(&mut s, &mut rng).unwrap();
        assert_eq!(s.samples, sent);
        // the receiver of a recording: the frame is found in the bits and intact
        let found = hdlc::frames(&s.bits);
        assert!(found.iter().any(|f| f.fcs_ok && f.payload == "été ✓".as_bytes()));
    }

    #[test]
    fn the_bare_bits_and_the_errors() {
        let ctx = BuildContext::default();
        let mut rng = Rng::new(1);
        let mut s = super::Frame::default();
        let bare = json!({"type": "text_source", "text": "A", "framing": "none", "constellation": "BPSK"});
        build(&bare, &ctx).unwrap().apply(&mut s, &mut rng).unwrap();
        assert_eq!(s.bits, [0, 1, 0, 0, 0, 0, 0, 1]);
        assert!(build(&json!({"type": "text_source", "text": "", "framing": "none"}), &ctx).is_err());
        // two tones do not fit eight samples at a spacing of five
        let tx = build(&json!({"type": "fsk_tx", "sps": 8, "spacing": 5}), &ctx).unwrap();
        assert!(tx.apply(&mut s, &mut rng).unwrap_err().to_string().contains("fsk_tx"));
        build(&json!({"type": "fsk_tx", "sps": 8}), &ctx).unwrap().apply(&mut s, &mut rng).unwrap();
        let rx = build(&json!({"type": "fsk_rx", "sps": 4}), &ctx).unwrap();
        assert!(rx.apply(&mut s, &mut rng).unwrap_err().to_string().contains("samples per symbol"));
    }
}
