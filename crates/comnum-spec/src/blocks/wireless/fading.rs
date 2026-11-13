//! The fading channels: `rayleigh`, `flat_fading`, `tdl` and `cdl`.

use super::Csi;
use crate::blocks::{Block, Frame, NameOr, Named};
use comnum_channel::cdl::{Array, Cdl, CdlParams, Profile as CdlProfile};
use comnum_channel::fading::{FlatFading, Mimo, flat_gains};
use comnum_channel::multipath::{Profile, Tdl, TdlParams};
use comnum_channel::spatial::{Kronecker, exponential_correlation};
use comnum_core::Result;
use comnum_core::rng::Rng;
use comnum_math::C64;

// ----------------------------------------------------------------------- rayleigh

block_params! {
    pub(super) struct RayleighSpec {
        n_rx: usize [1] >= 1, "receive antennas";
        coherence_samples: usize [0] "samples", "samples per channel matrix (0: one matrix per trial, 1: fast fading)";
        k_factor: f64 [0.0] >= 0, "Rician factor K (linear), 0 for Rayleigh fading";
        rx_correlation: f64 [0.0] > -1 < 1, "exponential correlation ρ of the receive antennas";
        tx_correlation: f64 [0.0] > -1 < 1, "exponential correlation ρ of the transmit antennas";
        seed: Option<u64> [absent "the stream of the trial"],
            "draw the channel from its own stream: the same realisation in every condition and trial, to compare receivers on one channel";
    }
}

pub(super) struct Rayleigh {
    pub(super) p: RayleighSpec,
}

/// A fading channel on every stream of the frame: the transmitted power, summed over
/// the antennas, becomes the reference of the noise blocks, and `channel(x, fs, rng)`
/// gives the received streams and the channel that a genie-aided receiver knows.
fn fade(
    s: &mut Frame,
    rng: &mut Rng,
    channel: impl FnOnce(&[Vec<C64>], f64, &mut Rng) -> Result<(Vec<Vec<C64>>, Csi)>,
) -> Result<()> {
    s.channel.faded_power = Some(s.mean_power() * s.n_streams() as f64);
    let fs = s.fs();
    let x = s.take_streams();
    let (y, csi) = channel(&x, fs, rng)?;
    s.set_streams(y);
    s.channel.csi = Some(csi);
    Ok(())
}

impl Block for Rayleigh {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        fade(s, rng, |x, _, rng| {
            let (nt, nr, n) = (x.len(), self.p.n_rx, x[0].len());
            let correlation = if self.p.rx_correlation != 0.0 || self.p.tx_correlation != 0.0 {
                Some(Kronecker::new(
                    exponential_correlation(nr, C64::new(self.p.rx_correlation, 0.0)),
                    exponential_correlation(nt, C64::new(self.p.tx_correlation, 0.0)),
                )?)
            } else {
                None
            };
            let coh = if self.p.coherence_samples == 0 { n.max(1) } else { self.p.coherence_samples };
            let model = Mimo { n_rx: nr, n_tx: nt, k_factor: self.p.k_factor, los: None, correlation, coherence: coh };
            let (y, hs) = match self.p.seed {
                Some(seed) => model.apply(x, &mut Rng::new(seed))?,
                None => model.apply(x, rng)?,
            };
            Ok((y, Csi::Flat { h: hs, coherence: coh }))
        })
    }
}

// -------------------------------------------------------------------- flat_fading

block_params! {
    pub(super) struct FlatFadingSpec {
        fading: String ["rayleigh"] one_of ["rayleigh", "rice", "nakagami", "hoyt"],
            "the distribution of the gain, of unit mean power (Simon & Alouini 1998, Table 2)";
        parameter: Option<f64> [absent "none for rayleigh"] >= 0,
            "the Rician factor K (linear) of rice, m >= 1/2 of nakagami, q in [0, 1] of hoyt";
        coherence_samples: usize [0] "samples", "samples per gain (0: one gain per trial, 1: fast fading)";
        seed: Option<u64> [absent "the stream of the trial"],
            "draw the gains from their own stream: the same realisation in every condition and trial";
    }
}

pub(super) struct FlatFadingBlock {
    fading: FlatFading,
    coherence: usize,
    seed: Option<u64>,
}

impl FlatFadingBlock {
    pub(super) fn new(p: &FlatFadingSpec) -> Result<FlatFadingBlock> {
        let parameter = if p.fading == "rayleigh" { None } else { p.parameter };
        Ok(FlatFadingBlock {
            fading: FlatFading::from_name(&p.fading, parameter).map_err(|e| e.context("flat_fading"))?,
            coherence: p.coherence_samples,
            seed: p.seed,
        })
    }
}

impl Block for FlatFadingBlock {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        fade(s, rng, |x, _, rng| {
            let n = x[0].len();
            let coh = if self.coherence == 0 { n.max(1) } else { self.coherence };
            // every stream by its own gain: a diagonal matrix per coherence block
            let mut own = self.seed.map(Rng::new);
            let rng = own.as_mut().unwrap_or(rng);
            let gains = x.iter().map(|_| flat_gains(self.fading, n, coh, rng)).collect::<Result<Vec<_>>>()?;
            let y = x.iter().zip(&gains).map(|(xs, g)| xs.iter().zip(g).map(|(v, h)| v * h).collect()).collect();
            let h = (0..n.div_ceil(coh).max(1))
                .map(|b| {
                    comnum_math::linalg::CMat::from_shape_fn((x.len(), x.len()), |(r, t)| {
                        if r == t {
                            gains[r].get(b * coh).copied().unwrap_or(C64::new(1.0, 0.0))
                        } else {
                            C64::new(0.0, 0.0)
                        }
                    })
                })
                .collect();
            Ok((y, Csi::Flat { h, coherence: coh }))
        })
    }
}

// ---------------------------------------------------------------------------- tdl

block_params! {
    pub(super) struct TdlSpec {
        profile: NameOr<ExponentialSpec> ["EPA"],
            "EPA, EVA, ETU (3GPP TS 36.101), pedestrian_a, pedestrian_b, vehicular_a, vehicular_b (ITU-R M.1225, Tables 4 and 5), TDLA30, TDLB100, TDLC300, TDLD30 (3GPP TS 38.101-4, Annex B.2.1), TDL-A to TDL-E (3GPP TR 38.901, Tables 7.7.2-1 to 7.7.2-5, scaled to delay_spread_s), or {rms_delay_s, resolution_s} (exponential)";
        delay_spread_s: f64 [100e-9] "s" >= 0, "RMS delay spread of TDL-A to TDL-E (TR 38.901 §7.7.3); the other profiles ignore it";
        doppler_hz: f64 [0.0] "Hz" >= 0, "maximum Doppler shift; 0 for a channel constant over the trial";
        n_rx: usize [1] >= 1, "receive antennas";
        seed: Option<u64> [absent "the stream of the trial"],
            "draw the channel from its own stream: the same realisation in every condition and trial, to compare receivers on one channel";
    }
}

block_params! {
    /// The exponential power-delay profile of `tdl`.
    struct ExponentialSpec {
        rms_delay_s: f64 [required] "s" > 0, "RMS delay spread";
        resolution_s: f64 [required] "s" > 0, "spacing of the taps";
    }
}

pub(super) struct TdlBlock {
    taps: Vec<(f64, f64)>,
    specular: Option<(f64, f64)>,
    doppler: f64,
    n_rx: usize,
    seed: Option<u64>,
}

impl TdlBlock {
    pub(super) fn new(p: &TdlSpec) -> Result<TdlBlock> {
        let profile = match p.profile.get("tdl profile")? {
            Named::Name(name) => name.parse::<Profile>()?.with_delay_spread(p.delay_spread_s),
            Named::Params(e) => Profile::Exponential { rms_delay_s: e.rms_delay_s, resolution_s: e.resolution_s },
        };
        Ok(TdlBlock {
            taps: profile.try_taps()?,
            specular: profile.specular(),
            doppler: p.doppler_hz,
            n_rx: p.n_rx,
            seed: p.seed,
        })
    }
}

impl Block for TdlBlock {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        fade(s, rng, |x, fs, rng| {
            let tdl = Tdl::new(&TdlParams {
                taps: self.taps.clone(),
                specular: self.specular,
                doppler_hz: self.doppler,
                sample_rate: fs,
                n_rx: self.n_rx,
                n_tx: x.len(),
            })?;
            let taps = match self.seed {
                Some(seed) => tdl.draw(x[0].len(), &mut Rng::new(seed)),
                None => tdl.draw(x[0].len(), rng),
            };
            Ok((taps.apply(x)?, Csi::Taps(taps)))
        })
    }
}

// ---------------------------------------------------------------------------- cdl

block_params! {
    pub(super) struct CdlSpec {
        profile: String ["CDL-A"] one_of ["CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E"],
            "the profile of TR 38.901 Tables 7.7.1-1 to 7.7.1-5 (A, B, C: NLOS; D, E: LOS)";
        delay_spread_s: f64 [100e-9] "s" >= 0, "desired RMS delay spread (TR 38.901 §7.7.3)";
        n_rx: usize [1] >= 1, "receive antennas, a vertical uniform linear array";
        spacing_wavelengths: f64 [0.5] "wavelengths" > 0, "element spacing of both arrays";
        velocity_mps: f64 [0.0] "m/s" >= 0, "speed of the receiver along the array axis; 0 for a channel constant over the trial";
        fc_hz: f64 [3.5e9] "Hz" > 0, "carrier frequency (the wavelength of the Doppler shifts)";
    }
}

pub(super) struct CdlBlock {
    profile: CdlProfile,
    delay_spread: f64,
    n_rx: usize,
    spacing: f64,
    velocity: f64,
    carrier: f64,
}

impl CdlBlock {
    pub(super) fn new(p: &CdlSpec) -> Result<CdlBlock> {
        Ok(CdlBlock {
            profile: p.profile.parse()?,
            delay_spread: p.delay_spread_s,
            n_rx: p.n_rx,
            spacing: p.spacing_wavelengths,
            velocity: p.velocity_mps,
            carrier: p.fc_hz,
        })
    }
}

impl Block for CdlBlock {
    fn apply(&self, s: &mut Frame, rng: &mut Rng) -> Result<()> {
        fade(s, rng, |x, fs, rng| {
            let tx = Array::linear(x.len(), self.spacing);
            let params = CdlParams {
                delay_spread_s: self.delay_spread,
                sample_rate: fs,
                tx,
                rx: Array::linear(self.n_rx, self.spacing),
                velocity_mps: [0.0, self.velocity, 0.0],
                fc_hz: self.carrier,
            };
            let cdl = Cdl::new(self.profile, params)?;
            let taps = cdl.draw(x[0].len(), rng)?;
            Ok((taps.apply(x)?, Csi::Taps(taps)))
        })
    }
}
