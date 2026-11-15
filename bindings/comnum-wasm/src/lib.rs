//! WebAssembly bindings of comnum for the browser playground.
//!
//! Five functions, exported by `wasm-bindgen` to the page (`playground/`): [`specs`],
//! the specifications of the repository; [`inspect`], what a specification will run,
//! before it runs (its link, its conditions, its metrics, the published and theoretical
//! points of its expectations); [`run`], the campaign itself on the calling thread, a
//! JavaScript callback receiving every condition as it completes and the progress of
//! every batch; [`trace`], the signals at the `scope` blocks of one trial; [`wav`], a
//! traced signal as the WAV file of a sound card. Every value crosses to JavaScript as
//! JSON text, but the bytes of the WAV file.
//!
//! The engine is the one of the native build: the same runner of `comnum_sim`, the
//! same blocks and seeds, so that a point computed in the browser is the point of
//! `comnum run` to the last bit; in the browser the trials of a batch run one after the
//! other on the one thread of the worker (the records are merged in trial order on
//! every target, see `comnum_sim::runner`).

use comnum::sim::callbacks::{Action, Callback};
use comnum::sim::clock::{self, Stopwatch};
use comnum::sim::grid::Value;
use comnum::sim::runner::ConditionResult;
use comnum::sim::stopping::Progress;
use comnum::spec::metrics::Metrics;
use comnum::spec::verify::{self, Expectation};
use comnum::spec::{RunOptions, Spec, SpecScenario};
use serde_json::{Map, Value as Json, json};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;

include!(concat!(env!("OUT_DIR"), "/specs.rs"));

/// The seconds between two progress messages within a condition.
const PROGRESS_EVERY_S: f64 = 0.25;

thread_local! {
    /// The JavaScript callbacks of the run in progress: `(on_condition, on_batch)`.
    static CALLBACKS: RefCell<Option<(js_sys::Function, js_sys::Function)>> = const { RefCell::new(None) };
}

/// The clock of the browser, in seconds: registered once, before the first run.
fn now_s() -> f64 {
    js_sys::Date::now() / 1e3
}

/// The error of a `Result` as the message JavaScript receives.
fn js_error(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// A value of a condition as JSON.
fn value_json(v: &Value) -> Json {
    match v {
        Value::F64(x) => json!(x),
        Value::I64(x) => json!(x),
        Value::Str(s) => json!(s),
        Value::Bool(b) => json!(b),
    }
}

/// A condition result as a flat JSON object: the row of the CSV report
/// (`ConditionResult::row`).
fn row_json(r: &ConditionResult, confidence: f64) -> Json {
    Json::Object(r.row(confidence).into_iter().map(|(k, v)| (k, value_json(&v))).collect())
}

/// The version of the library.
#[wasm_bindgen]
pub fn version() -> String {
    comnum::sim::VERSION.to_string()
}

/// The specifications of the repository, as JSON: a list of objects `path`, `name`,
/// `description`, `tutorial` (whether it lives in `specs/tutorial/`, one trial seen
/// along the chain), `validation` (whether it lives in `specs/validation/`, with
/// expectations) and `text`.
#[wasm_bindgen]
pub fn specs() -> String {
    let items: Vec<Json> = SPECS
        .iter()
        .map(|(path, text)| {
            let (name, description) = Spec::from_yaml(text)
                .map(|s| (s.name, s.description))
                .unwrap_or_else(|_| (path.to_string(), String::new()));
            json!({
                "path": path,
                "name": name,
                "description": description,
                "tutorial": path.starts_with("specs/tutorial/"),
                "validation": path.starts_with("specs/validation/"),
                "text": text,
            })
        })
        .collect();
    Json::Array(items).to_string()
}

/// What a specification will run, as JSON, before it runs: `name`, `description`,
/// `seed`, `link` (one object per block: `kind`, `label`, as the profile names it, and
/// `params`, the mapping of the block without its type and label), `axes` (the names of the variables of the grid), `conditions` (one object
/// per condition, with `index` and the variables), `metrics` (the names reported),
/// `stop` (the stopping rule as text), `expectations` (one object per expectation:
/// `name`, `source`, `metric`, `kind` (`curve`, `theory` or `value`) and `points`, the
/// expected value on every condition it applies to, as `{index, expected}`).
///
/// # Errors
/// The message of an invalid specification (YAML, unknown field, invalid expectation).
#[wasm_bindgen]
pub fn inspect(yaml: &str) -> Result<String, JsValue> {
    let spec = Spec::from_yaml(yaml).map_err(js_error)?;
    let conditions = spec.conditions();
    let axes: Vec<&str> = conditions.first().map(|c| c.keys().collect()).unwrap_or_default();
    let metrics = Metrics::from_json(&spec.metrics).map_err(js_error)?;
    let mut expectations = Vec::new();
    for e in &spec.expect {
        let x = Expectation::from_json(e).map_err(js_error)?;
        let kind =
            ["curve", "theory", "value", "reference"].into_iter().find(|k| e.get(k).is_some()).unwrap_or("value");
        let mut points = Vec::new();
        for c in &conditions {
            if let Some(v) = x.expected_on(c).map_err(js_error)? {
                points.push(json!({"index": c_index(c, &conditions), "expected": v}));
            }
        }
        expectations.push(json!({
            "name": x.name(),
            "source": x.source(),
            "metric": x.metric(),
            "kind": kind,
            "points": points,
        }));
    }
    let conds: Vec<Json> = conditions
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut o: Map<String, Json> = c.entries().iter().map(|(k, v)| (k.clone(), value_json(v))).collect();
            o.insert("index".into(), json!(i));
            Json::Object(o)
        })
        .collect();
    Ok(json!({
        "name": spec.name,
        "description": spec.description,
        "seed": spec.seed,
        "link": link_json(&spec.link).map_err(js_error)?,
        "axes": axes,
        "conditions": conds,
        "metrics": metrics.names(),
        "stop": spec.stop.to_string(),
        "expectations": expectations,
    })
    .to_string())
}

/// The blocks of a link as the page draws them: their type, label and parameters.
fn link_json(link: &[Json]) -> comnum::core::Result<Json> {
    let labels = comnum::spec::link_labels(link)?;
    let blocks = link
        .iter()
        .zip(labels)
        .map(|(b, label)| {
            let (kind, params) = match b {
                Json::String(k) => (k.clone(), Map::new()),
                Json::Object(o) => {
                    let kind = o.get("type").and_then(Json::as_str).unwrap_or_default().to_string();
                    let params = o
                        .iter()
                        .filter(|(k, _)| *k != "type" && *k != "label")
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    (kind, params)
                }
                _ => (String::new(), Map::new()),
            };
            json!({"kind": kind, "label": label, "params": params})
        })
        .collect();
    Ok(Json::Array(blocks))
}

/// The signals at the `scope` blocks of the first trial of condition `condition`, as
/// JSON: one object per scope, in the order of the link, with `label`, `view`, `sps`,
/// `symbol_rate`, `streams` (one `{re, im}` per antenna, the first samples) and, for
/// the view `spectrum`, `spectrum` (`f`, in units of the symbol rate, and `psd`, one
/// density per stream). The trial is that of a campaign (`comnum_spec::Spec::trace`).
///
/// # Errors
/// The message of a specification that does not parse, of a condition out of the grid
/// or of an error of the link.
#[wasm_bindgen]
pub fn trace(yaml: &str, condition: usize) -> Result<String, JsValue> {
    let spec = Spec::from_yaml(yaml).map_err(js_error)?;
    let traces = spec.trace(condition, 0).map_err(js_error)?;
    let items: Vec<Json> = traces
        .iter()
        .map(|t| {
            let streams: Vec<Json> = t
                .streams
                .iter()
                .map(|x| json!({"re": x.iter().map(|z| z.re).collect::<Vec<_>>(), "im": x.iter().map(|z| z.im).collect::<Vec<_>>()}))
                .collect();
            let spectrum = t.spectrum.as_ref().map(|(f, d)| json!({"f": f, "psd": d}));
            json!({
                "label": t.label,
                "view": t.view,
                "sps": t.sps,
                "symbol_rate": t.symbol_rate,
                "streams": streams,
                "spectrum": spectrum,
            })
        })
        .collect();
    Ok(Json::Array(items).to_string())
}

/// The peak of the samples of an exported WAV file, of a full scale of 1.
const WAV_PEAK: f64 = 0.8;

/// The silence before and after the signal of an exported WAV file (s).
const WAV_SILENCE_S: f64 = 0.25;

/// The bytes of [`wav`], with the errors of the library.
fn wav_bytes(re: &[f64], im: &[f64], fs: f64, carrier_hz: f64, audio_rate: u32) -> comnum::Result<Vec<u8>> {
    if re.len() != im.len() {
        return Err(comnum::Error::Shape(format!("{} real parts for {} imaginary parts", re.len(), im.len())));
    }
    let x: Vec<comnum::math::C64> = re.iter().zip(im).map(|(a, b)| comnum::math::C64::new(*a, *b)).collect();
    let signal = comnum::dsp::audio::upconvert(&x, fs, carrier_hz, f64::from(audio_rate))?;
    let peak = signal.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let gain = if peak > 0.0 { WAV_PEAK / peak } else { 0.0 };
    let silence = vec![0.0; (WAV_SILENCE_S * f64::from(audio_rate)) as usize];
    let mut samples = silence.clone();
    samples.extend(signal.iter().map(|v| v * gain));
    samples.extend(silence);
    comnum::dsp::audio::wav_encode(&samples, audio_rate)
}

/// The WAV file (mono, 16-bit PCM at `audio_rate`) that plays a traced signal on a
/// sound card: the complex baseband samples `re + j im`, taken as sampled at `fs` (Hz),
/// on the carrier `carrier_hz` (`comnum::dsp::audio::upconvert`), scaled to a peak of
/// 0.8 of the full scale, between two silences of a quarter of a second. A recording
/// of it comes back to baseband with `comnum.audio_downconvert(recording, carrier_hz,
/// fs)` of the Python package. The page only exports: it reads no recording.
///
/// # Errors
/// The message of a band `carrier_hz ± fs/2` that does not fit under half the audio
/// rate, or of rates that are not whole numbers of hertz.
#[wasm_bindgen]
pub fn wav(re: &[f64], im: &[f64], fs: f64, carrier_hz: f64, audio_rate: u32) -> Result<Vec<u8>, JsValue> {
    wav_bytes(re, im, fs, carrier_hz, audio_rate).map_err(js_error)
}

/// The index of a condition in the list (the conditions are compared by their entries).
fn c_index(c: &comnum::sim::grid::Condition, all: &[comnum::sim::grid::Condition]) -> usize {
    all.iter().position(|x| x.entries() == c.entries()).unwrap_or(0)
}

/// The callback of a run: every completed condition and the progress of every batch go
/// to the JavaScript functions of [`CALLBACKS`].
struct Relay {
    confidence: f64,
    last: Option<Stopwatch>,
}

impl Relay {
    fn call(which: usize, arg: &str) {
        CALLBACKS.with(|c| {
            if let Some((on_condition, on_batch)) = c.borrow().as_ref() {
                let f = if which == 0 { on_condition } else { on_batch };
                let _ = f.call1(&JsValue::NULL, &JsValue::from_str(arg));
            }
        });
    }
}

impl Callback for Relay {
    fn on_condition_start(&mut self, _: usize, _: &comnum::sim::grid::Condition) -> Action {
        self.last = Some(Stopwatch::start());
        Action::Continue
    }

    fn on_batch_end(&mut self, p: &Progress<'_>) -> Action {
        if self.last.is_some_and(|t| t.elapsed().as_secs_f64() >= PROGRESS_EVERY_S) {
            self.last = Some(Stopwatch::start());
            let arg = json!({"index": p.condition, "trials": p.trials, "elapsed_s": p.elapsed.as_secs_f64()});
            Relay::call(1, &arg.to_string());
        }
        Action::Continue
    }

    fn on_condition_end(&mut self, r: &ConditionResult) -> Action {
        Relay::call(0, &row_json(r, self.confidence).to_string());
        Action::Continue
    }
}

/// Run a specification on the calling thread. `on_condition` receives every completed
/// condition as JSON text (the row of the CSV report: `index`, the variables, `status`,
/// `trials`, `wall_s`, then `<metric>`, `<metric>_lo`, `<metric>_hi`, `<metric>_count`,
/// `<metric>_total` for every metric); `on_batch` receives, at most four times a second,
/// `{index, trials, elapsed_s}` within a condition. The result is JSON: `results` (the
/// rows), `checks` (the expectations checked, as in the report of `comnum verify`:
/// `name`, `source`, `condition`, `metric`, `expected`, `value`, `lo`, `hi`, `passed`;
/// empty without expectations), `passed` (every check passed) and `wall_s`.
///
/// # Errors
/// The message of an invalid specification, of a block or of a check.
#[wasm_bindgen]
pub fn run(yaml: &str, on_condition: &js_sys::Function, on_batch: &js_sys::Function) -> Result<String, JsValue> {
    clock::set_clock(now_s);
    let spec = Spec::from_yaml(yaml).map_err(js_error)?;
    let opts = RunOptions { threads: Some(1), progress: Some(false), ..RunOptions::default() };
    CALLBACKS.with(|c| *c.borrow_mut() = Some((on_condition.clone(), on_batch.clone())));
    let t = Stopwatch::start();
    let outcome = spec
        .runner(&opts)
        .callback(Relay { confidence: spec.confidence, last: None })
        .run(&SpecScenario::new(&spec), &spec.grid);
    CALLBACKS.with(|c| *c.borrow_mut() = None);
    let results = outcome.map_err(js_error)?;
    let checks = if spec.expect.is_empty() {
        Vec::new()
    } else {
        verify::verify(&spec.expect, &results, spec.confidence).map_err(js_error)?
    };
    let rows: Vec<Json> = results.iter().map(|r| row_json(r, spec.confidence)).collect();
    let checks_json: Vec<Json> = checks
        .iter()
        .map(|c| {
            json!({
                "name": c.name, "source": c.source, "condition": c.condition, "metric": c.metric,
                "expected": c.expected, "value": c.measured.value, "lo": c.measured.lo, "hi": c.measured.hi,
                "passed": c.passed,
            })
        })
        .collect();
    Ok(json!({
        "results": rows,
        "checks": checks_json,
        "passed": checks.iter().all(|c| c.passed),
        "wall_s": t.elapsed().as_secs_f64(),
    })
    .to_string())
}

#[cfg(test)]
mod tests {
    use super::wav_bytes;
    use comnum::dsp::audio::{downconvert, wav_decode};

    #[test]
    fn an_exported_signal_comes_back_from_its_wav_file() {
        // 64 symbols of a QPSK-like signal at 8 samples per symbol, played at 100 baud
        let (re, im): (Vec<f64>, Vec<f64>) = (0..512)
            .map(|n| (if (n / 8) % 2 == 0 { 1.0 } else { -1.0 }, if (n / 16) % 2 == 0 { 1.0 } else { -1.0 }))
            .unzip();
        let bytes = wav_bytes(&re, &im, 800.0, 6000.0, 48_000).unwrap();
        let wav = wav_decode(&bytes).unwrap();
        // a quarter of a second of silence on each side of 0.64 s of signal
        assert_eq!((wav.sample_rate, wav.channels[0].len()), (48_000, 12_000 + 30_720 + 12_000));
        assert!(wav.channels[0][..11_000].iter().all(|v| *v == 0.0));
        let peak = wav.channels[0].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!((peak - 0.8).abs() < 1e-3);
        // back at baseband, the samples sent to a scale
        let y = downconvert(&wav.channels[0][12_000..42_720], 48_000.0, 6000.0, 800.0).unwrap();
        let scale = y[100].re / re[100];
        let (a, b) = (y[300], y[404]);
        assert!((a.re - scale * re[300]).abs() < 0.05 * scale && (b.im - scale * im[404]).abs() < 0.05 * scale);
        assert!(wav_bytes(&re, &im[..10], 800.0, 6000.0, 48_000).is_err());
        assert!(wav_bytes(&re, &im, 800.0, 300.0, 48_000).is_err());
    }

    /// What the page does with `specs/tutorial/13_text_fsk.yaml`, then what its header
    /// tells the reader to do in Python: the text comes back from the exported file.
    #[test]
    fn the_text_of_the_tutorial_comes_back_from_the_exported_wav_file() {
        use comnum::spec::Spec;
        use comnum::systems::hdlc;
        use comnum::waveform::fsk::Fsk;
        let yaml = include_str!("../../../specs/tutorial/13_text_fsk.yaml");
        let traces = Spec::from_yaml(yaml).unwrap().trace(0, 0).unwrap();
        let signal = traces.iter().find(|t| t.label == "message").unwrap();
        let (re, im): (Vec<f64>, Vec<f64>) = signal.streams[0].iter().map(|z| (z.re, z.im)).unzip();
        // 100 baud at the 24 samples per symbol of the scope, a carrier of 6 kHz
        let fs = 100.0 * signal.sps;
        let wav = wav_decode(&wav_bytes(&re, &im, fs, 6000.0, 48_000).unwrap()).unwrap();
        let y = downconvert(&wav.channels[0], 48_000.0, 6000.0, fs).unwrap();
        let fsk = Fsk::new(4, 24, 2).unwrap();
        let decided = fsk.demodulate(&y[fsk.timing(&y).unwrap()..]).unwrap();
        let bits: Vec<u8> = decided.iter().flat_map(|d| [(d >> 1) as u8, (d & 1) as u8]).collect();
        let text = "Bonjour ! Ce message est passé par le haut-parleur.";
        assert!(hdlc::frames(&bits).iter().any(|f| f.fcs_ok && f.payload == text.as_bytes()));
    }
}
