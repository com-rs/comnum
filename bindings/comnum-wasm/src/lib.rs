//! WebAssembly bindings of comnum for the browser playground.
//!
//! Four functions, exported by `wasm-bindgen` to the page (`playground/`): [`specs`],
//! the specifications of the repository; [`inspect`], what a specification will run,
//! before it runs (its link, its conditions, its metrics, the published and theoretical
//! points of its expectations); [`run`], the campaign itself on the calling thread, a
//! JavaScript callback receiving every condition as it completes and the progress of
//! every batch; [`trace`], the signals at the `scope` blocks of one trial. Every value
//! crosses to JavaScript as JSON text.
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
/// `description`, `validation` (whether it lives in `specs/validation/`, with
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
