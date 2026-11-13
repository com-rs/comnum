//! The closed forms of the `theory` of an expectation, one variant per `function`, each
//! with its own parameters (an unknown parameter is an error).

use crate::blocks::{BuildContext, parse_constellation};
use comnum_channel::optical::gn::{FormatMoments, GnModel, WdmSpectrum};
use comnum_core::{Error, Result, ensure};
use comnum_mapping::constellation::{Constellation, ConstellationKind};
use comnum_mapping::theory;
use comnum_math::{db2lin, lin2db};
use comnum_metrics as metrics;
use serde::{Deserialize, Deserializer};
use serde_json::Value as Json;

/// A closed form of an expectation, its `function` the tag of its parameters.
#[derive(Deserialize, Debug)]
#[serde(tag = "function", rename_all = "snake_case")]
pub(super) enum Theory {
    /// The BER of Gray-labelled `kind`-`m` over AWGN.
    Ber(ErrorRate),
    /// The SER of `kind`-`m` over AWGN.
    Ser(ErrorRate),
    /// The mutual information of `kind`-`m` over AWGN (bits/symbol).
    Mi(Information),
    /// The generalised mutual information (bits/symbol).
    Gmi(Information),
    /// The $E_{b}/N_{0}$ (dB) of a BER `target`.
    Ebn0ForBer(Inverse),
    /// The $E_{b}/N_{0}$ (dB) of an SER `target`.
    Ebn0ForSer(Inverse),
    /// The SNR $E_{s}/N_{0}$ (dB) of a BER `target`.
    SnrForBer(Inverse),
    /// The SNR $E_{s}/N_{0}$ (dB) of an SER `target`.
    SnrForSer(Inverse),
    /// The SNR (dB) of the centre channel of a WDM link by the GN model.
    GnSnrDb(Gn),
    /// The BER of BPSK or QPSK with maximal-ratio combining of Rayleigh branches.
    BerRayleigh(Rayleigh),
    /// The BER of a coherent binary signal over flat fading, by the MGF of the SNR.
    BerFading(FadingTheory),
    /// The ideal $E_{s}/N_{0}$ (dB) of a DVB-S2 MODCOD (Table 13 of EN 302 307-1).
    Dvbs2Esn0Db(Dvbs2),
}

/// A value of a type parsed by its `FromStr` (without case, for the families of
/// constellations).
fn parsed<'de, D: Deserializer<'de>, T: std::str::FromStr<Err = Error>>(d: D) -> std::result::Result<T, D::Error> {
    String::deserialize(d)?.parse().map_err(serde::de::Error::custom)
}

fn d_qam() -> ConstellationKind {
    ConstellationKind::Qam
}
fn d_m() -> usize {
    4
}
fn d_points() -> usize {
    comnum_mapping::theory::DEFAULT_QUADRATURE_POINTS
}
fn d_one() -> usize {
    1
}
fn d_true() -> bool {
    true
}

/// The parameters of `ber` and `ser`.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct ErrorRate {
    #[serde(default = "d_qam", deserialize_with = "parsed")]
    kind: ConstellationKind,
    #[serde(default = "d_m")]
    m: usize,
    #[serde(default)]
    ebn0_db: Option<f64>,
    #[serde(default)]
    snr_db: Option<f64>,
}

/// The parameters of `mi` and `gmi`.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct Information {
    #[serde(default = "d_qam", deserialize_with = "parsed")]
    kind: ConstellationKind,
    #[serde(default = "d_m")]
    m: usize,
    #[serde(default)]
    ebn0_db: Option<f64>,
    #[serde(default)]
    snr_db: Option<f64>,
    /// Gauss–Hermite points per dimension.
    #[serde(default = "d_points")]
    points: usize,
    /// The Maxwell–Boltzmann priors of a shaped source.
    #[serde(default)]
    mb_lambda: Option<f64>,
}

/// The parameters of the inverses `ebn0_for_ber`, `ebn0_for_ser`, `snr_for_ber` and
/// `snr_for_ser`.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct Inverse {
    #[serde(default = "d_qam", deserialize_with = "parsed")]
    kind: ConstellationKind,
    #[serde(default = "d_m")]
    m: usize,
    target: f64,
}

/// The parameters of `gn_snr_db`.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct Gn {
    /// Merged over the `fiber` of the specification, as for the fibre blocks.
    #[serde(default)]
    fiber: Option<Json>,
    n_channels: usize,
    spacing_hz: f64,
    symbol_rate: f64,
    #[serde(default)]
    rolloff: f64,
    power_dbm: f64,
    #[serde(default = "d_true")]
    coherent: bool,
    /// A constellation (with `mb_lambda` for shaped symbols): the EGN model.
    #[serde(default)]
    format: Option<Json>,
    #[serde(default)]
    mb_lambda: Option<f64>,
}

/// The parameters of `ber_rayleigh`.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct Rayleigh {
    #[serde(default = "d_qam", deserialize_with = "parsed")]
    kind: ConstellationKind,
    #[serde(default = "d_m")]
    m: usize,
    #[serde(default)]
    ebn0_db: Option<f64>,
    #[serde(default)]
    snr_db: Option<f64>,
    #[serde(default)]
    offset_db: f64,
    #[serde(default = "d_one")]
    branches: usize,
}

/// The parameters of `ber_fading`.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct FadingTheory {
    fading: String,
    #[serde(default)]
    parameter: Option<f64>,
    ebn0_db: f64,
    #[serde(default = "d_g")]
    g: f64,
    #[serde(default = "d_one")]
    branches: usize,
}

fn d_g() -> f64 {
    1.0
}

/// The parameters of `dvbs2_esn0_db`.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub(super) struct Dvbs2 {
    modulation: String,
    rate: String,
    #[serde(default)]
    offset_db: f64,
}

/// $E_{b}/N_{0}$ (dB) from `ebn0_db`, or from `snr_db` minus $10 \log_{10} \log_{2} M$.
fn ebn0_of(ebn0_db: Option<f64>, snr_db: Option<f64>, m: usize, name: &str) -> Result<f64> {
    ebn0_db
        .or(snr_db.map(|s| s - lin2db((m as f64).log2())))
        .ok_or_else(|| Error::Config(format!("expectation {name:?}: the theory needs ebn0_db or snr_db")))
}

/// The largest order of a constellation and the largest number of quadrature points or
/// diversity branches of a closed form.
const MAX_ORDER: usize = 1 << 16;
const MAX_POINTS: usize = 10_000;

impl Theory {
    /// The value of the closed form (`name`: the expectation, for the messages).
    pub(super) fn value(&self, name: &str) -> Result<f64> {
        let bits = |m: usize| lin2db((m as f64).log2());
        Ok(match self {
            Theory::Ber(t) => metrics::theory_ber(t.kind, t.m, ebn0_of(t.ebn0_db, t.snr_db, t.m, name)?)?,
            Theory::Ser(t) => metrics::theory_ser(t.kind, t.m, ebn0_of(t.ebn0_db, t.snr_db, t.m, name)?)?,
            Theory::Mi(t) | Theory::Gmi(t) => {
                // sizes that a closed form is evaluated with: a typing error is refused
                // instead of a computation without end
                ensure(t.m <= MAX_ORDER && t.points <= MAX_POINTS, || {
                    format!(
                        "expectation {name:?}: an order of at most {MAX_ORDER} and at most {MAX_POINTS} quadrature \
                         points, got m = {} and points = {}",
                        t.m, t.points
                    )
                })?;
                // Maxwell-Boltzmann priors on the unit-energy uniform constellation, as
                // the source with mb_lambda
                let px = match t.mb_lambda {
                    Some(l) => Some(comnum_mapping::source::maxwell_boltzmann(
                        &Constellation::new(t.kind, t.m)?.normalized(),
                        l,
                    )),
                    None => None,
                };
                let snr = t.snr_db.or(t.ebn0_db.map(|e| e + bits(t.m))).ok_or_else(|| {
                    Error::Config(format!("expectation {name:?}: the theory needs snr_db or ebn0_db"))
                })?;
                let f = if matches!(self, Theory::Mi(_)) { metrics::theory_mi } else { metrics::theory_gmi };
                f(t.kind, t.m, snr, px.as_deref(), t.points)?
            }
            Theory::Ebn0ForBer(t) => theory::ebn0_for_ber(t.kind, t.m, t.target)?,
            Theory::Ebn0ForSer(t) => theory::ebn0_for_ser(t.kind, t.m, t.target)?,
            Theory::SnrForBer(t) => theory::ebn0_for_ber(t.kind, t.m, t.target)? + bits(t.m),
            Theory::SnrForSer(t) => theory::ebn0_for_ser(t.kind, t.m, t.target)? + bits(t.m),
            Theory::GnSnrDb(t) => t.snr_db()?,
            Theory::BerRayleigh(t) => {
                ensure(t.m <= 4 && matches!(t.kind, ConstellationKind::Qam | ConstellationKind::Psk), || {
                    format!("expectation {name:?}: ber_rayleigh is for BPSK and QPSK")
                })?;
                ensure(t.branches <= MAX_POINTS, || {
                    format!("expectation {name:?}: at most {MAX_POINTS} branches, got {}", t.branches)
                })?;
                theory::ber_rayleigh_mrc(db2lin(ebn0_of(t.ebn0_db, t.snr_db, t.m, name)? + t.offset_db), t.branches)?
            }
            Theory::BerFading(t) => {
                ensure(t.branches <= MAX_POINTS, || {
                    format!("expectation {name:?}: at most {MAX_POINTS} branches, got {}", t.branches)
                })?;
                // rayleigh and awgn take no parameter: one written for the others is ignored
                let parameter = if matches!(t.fading.as_str(), "rayleigh" | "awgn") { None } else { t.parameter };
                let fading = theory::SnrFading::from_name(&t.fading, parameter)?;
                theory::ber_binary_fading(t.g, fading, t.ebn0_db, t.branches)?
            }
            Theory::Dvbs2Esn0Db(t) => {
                comnum_systems::dvbs2::Modcod::from_name(&t.modulation, &t.rate)?.threshold_db() + t.offset_db
            }
        })
    }
}

impl Gn {
    /// The SNR (dB) of the centre channel: its fibre (merged over that of the
    /// specification when the specification was parsed, see [`super::resolve_fiber`]),
    /// with its Raman pumps; the EGN model with a `format`.
    fn snr_db(&self) -> Result<f64> {
        let ctx = BuildContext::default();
        let fiber = ctx.fiber(self.fiber.as_ref())?;
        let freqs = comnum_dsp::signal::wdm_grid(self.n_channels, self.spacing_hz);
        let p = comnum_math::dbm2w(self.power_dbm);
        let egn = match &self.format {
            None => None,
            Some(Json::String(s)) if s.eq_ignore_ascii_case("gaussian") => Some(FormatMoments::GAUSSIAN),
            Some(v) => {
                // the priors of the source with mb_lambda: on the unit-energy constellation
                let c = parse_constellation(v)?.normalized();
                let px = self.mb_lambda.map(|l| comnum_mapping::source::maxwell_boltzmann(&c, l));
                Some(FormatMoments::of(c.points(), px.as_deref())?)
            }
        };
        let model =
            GnModel { coherent: self.coherent, raman: ctx.raman(self.fiber.as_ref())?, egn, ..GnModel::new(fiber) };
        let snr = model.snr(&WdmSpectrum::uniform(freqs, p, self.symbol_rate, self.rolloff), self.n_channels / 2)?;
        Ok(lin2db(snr))
    }
}
