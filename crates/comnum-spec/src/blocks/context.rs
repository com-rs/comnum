//! What the builders need besides the parameters of a block: [`BuildContext`], with
//! the fibre and Raman parameters of the specification.

use super::Params;
use super::schema::Mapping;
use comnum_channel::optical::fiber::FiberParams;
use comnum_channel::optical::raman::{Direction, Raman, RamanGain, RamanPump};
use comnum_core::{Error, Result};
use serde_json::{Map, Value as Json};
use std::path::PathBuf;

/// The meaning of the `fiber` parameter of the optical blocks.
pub(super) const FIBER_DOC: &str = "fibre parameters, merged over the `fiber` of the specification (see FiberParams); its `raman` mapping adds Raman pumps and ISRS";

/// What the builders of the blocks need besides their parameters.
#[derive(Clone, Debug, Default)]
pub struct BuildContext {
    /// The `fiber` mapping of the specification (defaults of every fibre block).
    pub fiber: Option<Json>,
    /// The directory of the specification (relative paths of files).
    pub base_dir: Option<PathBuf>,
    /// The name of the campaign (the files a block writes).
    pub name: Option<String>,
    /// The label of the condition the link is built for (the files a block writes).
    pub condition: Option<String>,
}

impl BuildContext {
    /// The fibre parameters of a block: the defaults of [`FiberParams`], then the
    /// specification's `fiber`, then the block's `fiber`.
    ///
    /// # Errors
    /// [`Error::Config`] for parameters that are not a valid fibre mapping.
    pub fn fiber(&self, block: Option<&Json>) -> Result<FiberParams> {
        Ok(FiberSpec::parse("fiber", Json::Object(self.fiber_map(block)?))?.fiber_params())
    }

    /// The Raman amplification of a block: the `raman` mapping of its merged `fiber`
    /// (see the optical blocks), `None` without one.
    ///
    /// # Errors
    /// [`Error::Config`] for a `raman` mapping that is not valid.
    pub fn raman(&self, block: Option<&Json>) -> Result<Option<Raman>> {
        let f = FiberSpec::parse("fiber", Json::Object(self.fiber_map(block)?))?;
        f.raman.map(|r| r.parse("raman")?.raman()).transpose()
    }

    fn fiber_map(&self, block: Option<&Json>) -> Result<Map<String, Json>> {
        merge_fiber(self.fiber.as_ref(), block)
    }

    /// A path relative to the specification.
    pub fn path(&self, p: &str) -> PathBuf {
        let p = PathBuf::from(p);
        match &self.base_dir {
            Some(d) if p.is_relative() => d.join(p),
            _ => p,
        }
    }
}

/// The fibre mapping of a block (or of a theory) merged over that of the
/// specification: the keys of `block` replace those of `spec`.
///
/// # Errors
/// [`Error::Config`] for a value that is not a mapping.
pub(crate) fn merge_fiber(spec: Option<&Json>, block: Option<&Json>) -> Result<Map<String, Json>> {
    let mut m = Map::new();
    for src in [spec, block].into_iter().flatten() {
        match src {
            Json::Object(o) => m.extend(o.clone()),
            Json::Null => {}
            other => return Err(Error::Config(format!("fiber must be a mapping, got {other}"))),
        }
    }
    Ok(m)
}

block_params! {
    /// The `fiber` mapping of a specification or of a block: the fields of
    /// [`FiberParams`], their defaults its own, and the Raman pumps of every span.
    pub(crate) struct FiberSpec {
        alpha_db_km: f64 [FiberParams::default().alpha_db_km => "0.2"] "dB/km", "attenuation α";
        d_ps_nm_km: f64 [FiberParams::default().d_ps_nm_km => "17.0"] "ps/(nm km)",
            "dispersion parameter D (positive for standard fibre at 1550 nm)";
        gamma_w_km: f64 [FiberParams::default().gamma_w_km => "1.3"] "1/(W km)", "nonlinear coefficient γ";
        span_km: f64 [FiberParams::default().span_km => "80.0"] "km", "length of one span";
        n_spans: usize [FiberParams::default().n_spans => "1"], "number of spans";
        fc_hz: f64 [FiberParams::default().fc_hz => "193.1e12"] "Hz", "carrier frequency";
        nf_db: f64 [FiberParams::default().nf_db => "4.5"] "dB", "noise figure of the amplifiers";
        slope_ps_nm2_km: f64 [FiberParams::default().slope_ps_nm2_km => "0.0"] "ps/(nm² km)",
            "dispersion slope S (the third-order dispersion), 0 for none";
        raman: Option<Mapping<RamanSpec>> [absent "no Raman pump"],
            "Raman amplification of every span (see comnum_channel::optical::raman)";
    }
}

impl FiberSpec {
    /// The fibre parameters.
    fn fiber_params(&self) -> FiberParams {
        FiberParams {
            alpha_db_km: self.alpha_db_km,
            d_ps_nm_km: self.d_ps_nm_km,
            gamma_w_km: self.gamma_w_km,
            span_km: self.span_km,
            n_spans: self.n_spans,
            fc_hz: self.fc_hz,
            nf_db: self.nf_db,
            slope_ps_nm2_km: self.slope_ps_nm2_km,
        }
    }
}

block_params! {
    /// The `raman` mapping of a fibre: the pumps and the settings of [`Raman`], their
    /// defaults its own.
    pub(crate) struct RamanSpec {
        pumps: Vec<Mapping<PumpSpec>> [serde_json::json!([]) => "[]"], "the pumps of every span, any number, co- or counter-propagating";
        aeff_um2: f64 [Raman::default().aeff_um2 => "80.0"] "µm²" > 0, "effective area";
        aeff_spectrum: Option<Mapping<AeffSpectrumSpec>> [absent "aeff_um2 at every frequency"],
            "the effective area against frequency (linear between the points, constant beyond); two waves couple through the mean of their areas";
        alpha_spectrum: Option<Mapping<AlphaSpectrumSpec>> [absent "alpha_db_km of the fibre at every frequency"],
            "the attenuation against frequency (linear between the points, constant beyond), of the signals and of the pumps without their own alpha_db_km";
        temperature_k: f64 [Raman::default().temperature_k => "300.0"] "K" > 0,
            "temperature, for the spontaneous Raman noise";
        isrs: bool [Raman::default().isrs => "true"], "the channels also exchange power (inter-channel stimulated Raman scattering)";
        step_km: f64 [Raman::default().step_km => "0.1"] "km" > 0, "grid of the power equations";
        slice_hz: f64 [Raman::default().slice_hz => "25e9"] "Hz" > 0, "slices of the spectrum of the field";
        gain: Option<Mapping<GainSpec>> [absent "standard single-mode fibre"], "the Raman gain curve";
        rayleigh_db_km: Option<f64> [absent "no double Rayleigh backscattering"] "dB/km",
            "Rayleigh backscatter coupling of the signals, 10 log10 of its value in 1/km (about -40 for transmission fibre): the double Rayleigh backscattering is counted as a noise";
    }
}

block_params! {
    /// A pump of the `raman` mapping.
    pub(crate) struct PumpSpec {
        wavelength_nm: Option<f64> [absent "frequency_thz"] "nm" > 0, "vacuum wavelength (or frequency_thz)";
        frequency_thz: Option<f64> [absent "wavelength_nm"] "THz" > 0, "frequency (or wavelength_nm)";
        power_mw: f64 [required] "mW" >= 0, "power launched at its end of the span";
        direction: String ["counter"] one_of ["co", "forward", "counter", "backward"],
            "co (forward, launched at z = 0) or counter (backward, launched at z = L)";
        alpha_db_km: Option<f64> [absent "that of the fibre"] "dB/km", "attenuation at the pump wavelength";
    }
}

block_params! {
    /// The effective area against frequency of the `raman` mapping.
    pub(crate) struct AeffSpectrumSpec {
        frequency_thz: Vec<f64> [required] "THz", "increasing frequencies of the points";
        aeff_um2: Vec<f64> [required] "µm²", "effective areas at the frequencies";
    }
}

block_params! {
    /// The attenuation against frequency of the `raman` mapping.
    pub(crate) struct AlphaSpectrumSpec {
        frequency_thz: Vec<f64> [required] "THz", "increasing frequencies of the points";
        alpha_db_km: Vec<f64> [required] "dB/km", "attenuations at the frequencies";
    }
}

/// The `(Hz, value)` points of a spectrum of the `raman` mapping.
///
/// # Errors
/// [`Error::Config`] when the two lists differ in length.
fn spectrum_points(name: &str, frequency_thz: &[f64], values: &[f64]) -> Result<Vec<(f64, f64)>> {
    if frequency_thz.len() != values.len() {
        return Err(Error::Config(format!(
            "raman {name}: {} frequencies for {} values",
            frequency_thz.len(),
            values.len()
        )));
    }
    Ok(frequency_thz.iter().zip(values).map(|(f, v)| (f * 1e12, *v)).collect())
}

block_params! {
    /// A measured Raman gain curve of the `raman` mapping.
    pub(crate) struct GainSpec {
        offsets_thz: Vec<f64> [required] "THz", "frequency offsets of the curve";
        gain_m_w: Vec<f64> [required] "m/W", "the gain coefficient times the effective area, g_R A_eff, at the offsets";
        reference_thz: f64 [required] "THz", "pump frequency of the measurement";
        pump_scaling: bool [true], "scale the efficiency of a pump with its frequency, f_p/f_ref (false: the offset alone, the coupling of GNPy up to version 2.4)";
    }
}

impl RamanSpec {
    /// The Raman amplification.
    fn raman(&self) -> Result<Raman> {
        let d = Raman::default();
        let pumps = self.pumps.iter().map(|p| p.parse("raman pump")?.pump()).collect::<Result<_>>()?;
        let gain = match &self.gain {
            None => d.gain.clone(),
            Some(g) => {
                let g = g.parse("raman gain")?;
                RamanGain::new(g.offsets_thz.iter().map(|f| f * 1e12).collect(), g.gain_m_w, g.reference_thz * 1e12)?
                    .with_pump_scaling(g.pump_scaling)
            }
        };
        let aeff_spectrum_um2 = match &self.aeff_spectrum {
            None => Vec::new(),
            Some(a) => {
                let a = a.parse("raman aeff_spectrum")?;
                spectrum_points("aeff_spectrum", &a.frequency_thz, &a.aeff_um2)?
            }
        };
        let alpha_spectrum_db_km = match &self.alpha_spectrum {
            None => Vec::new(),
            Some(a) => {
                let a = a.parse("raman alpha_spectrum")?;
                spectrum_points("alpha_spectrum", &a.frequency_thz, &a.alpha_db_km)?
            }
        };
        Ok(Raman {
            pumps,
            gain,
            aeff_um2: self.aeff_um2,
            aeff_spectrum_um2,
            alpha_spectrum_db_km,
            temperature_k: self.temperature_k,
            isrs: self.isrs,
            step_km: self.step_km,
            slice_hz: self.slice_hz,
            rayleigh_db_km: self.rayleigh_db_km,
            ..d
        })
    }
}

impl PumpSpec {
    /// The pump.
    fn pump(&self) -> Result<RamanPump> {
        let direction = match self.direction.as_str() {
            "co" | "forward" => Direction::Co,
            _ => Direction::Counter,
        };
        let frequency_hz = match (self.wavelength_nm, self.frequency_thz) {
            (Some(w), None) => comnum_math::nm2hz(w),
            (None, Some(f)) => f * 1e12,
            _ => return Err(Error::Config("raman: a pump needs wavelength_nm or frequency_thz".into())),
        };
        Ok(RamanPump { frequency_hz, power_w: self.power_mw * 1e-3, direction, alpha_db_km: self.alpha_db_km })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::ParamDefault;

    #[test]
    fn the_defaults_of_the_fibre_and_of_raman_are_those_of_com_channel() {
        let f = BuildContext::default().fiber(None).unwrap();
        assert_eq!(f, FiberParams::default());
        let r = RamanSpec::parse("raman", serde_json::json!({})).unwrap().raman().unwrap();
        assert_eq!(r, Raman::default());
        // the documented text is the value
        for p in FiberSpec::params().into_iter().chain(RamanSpec::params()) {
            if let ParamDefault::Value(v, text) = p.default {
                let shown: f64 = text.parse().unwrap_or(f64::NAN);
                let ok = v
                    .as_f64()
                    .map_or(serde_json::from_str::<serde_json::Value>(&text).ok().as_ref() == Some(&v), |x| x == shown);
                assert!(ok, "{}: {text} for {v}", p.name);
            }
        }
        // the choices of a pump are read without case
        let p = PumpSpec::parse("pump", serde_json::json!({"wavelength_nm": 1450, "power_mw": 100, "direction": "CO"}));
        assert_eq!(p.unwrap().pump().unwrap().direction, Direction::Co);
    }

    #[test]
    fn the_spectra_of_the_raman_mapping_are_read_in_hertz() {
        let r = RamanSpec::parse(
            "raman",
            serde_json::json!({
                "aeff_spectrum": {"frequency_thz": [185.0, 220.0], "aeff_um2": [86.0, 72.0]},
                "alpha_spectrum": {"frequency_thz": [185.0, 193.0, 205.0], "alpha_db_km": [0.2, 0.19, 0.24]}
            }),
        )
        .unwrap()
        .raman()
        .unwrap();
        assert_eq!(r.aeff_spectrum_um2, vec![(185e12, 86.0), (220e12, 72.0)]);
        assert_eq!(r.alpha_spectrum_db_km, vec![(185e12, 0.2), (193e12, 0.19), (205e12, 0.24)]);
        // two lists of different lengths
        let bad = RamanSpec::parse(
            "raman",
            serde_json::json!({"alpha_spectrum": {"frequency_thz": [185.0], "alpha_db_km": []}}),
        );
        assert!(bad.unwrap().raman().is_err());
    }
}
