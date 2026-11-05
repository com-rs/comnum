//! The output of `comnum --json` for programs and agents: one compact JSON document per
//! specification on standard output, results with their confidence intervals and their
//! provenance, errors with a stable kind, a hint and a suggestion, and the time budget
//! of `--max-seconds`.
//!
//! A run is a line
//!
//! ```text
//! {"ok": true, "command": "run", "spec": NAME, "provenance": {...}, "budget_s": S,
//!  "conditions": [{"condition": {...}, "status": ..., "trials": N, "wall_s": T,
//!                  "time_limited": false, "metrics": {"ber": {"value": ..., "lo": ...,
//!                  "hi": ..., "count": ..., "total": ...}}}], "wall_s": T}
//! ```
//!
//! and an error a line `{"ok": false, "command": ..., "file": ..., "error": {"kind": ...,
//! "message": ..., "hint": ..., "suggestion": ...}}`, the kind one of `usage`, `config`,
//! `shape`, `numerical`, `condition` and `io`.

use crate::blocks;
use crate::search::SearchResult;
use crate::verify::Check;
use crate::{RunOptions, Spec};
use comnum_core::Error;
use comnum_sim::loggers::value_json;
use comnum_sim::runner::ConditionResult;
use comnum_sim::stopping::{Progress, Stop};
use serde_json::{Map, Value as Json, json};
use std::time::Duration;

/// The time budget of a run in JSON mode when `--max-seconds` is not given (s).
pub const DEFAULT_BUDGET_S: f64 = 300.0;

/// The budget of a run (s): `--max-seconds` (0 for none), [`DEFAULT_BUDGET_S`] in JSON
/// mode, none otherwise.
pub fn budget(max_seconds: Option<f64>, json: bool) -> Option<f64> {
    match max_seconds {
        Some(s) if s > 0.0 => Some(s),
        Some(_) => None,
        None => json.then_some(DEFAULT_BUDGET_S),
    }
}

/// Spread a budget of `seconds` over the conditions (and the evaluations of a search) of
/// `spec`: its stopping rule becomes the rule of the specification or the share of the
/// budget of every condition, whichever comes first. Returns the rule of the
/// specification, to tell afterwards which conditions the budget cut short.
pub fn apply_budget(spec: &mut Spec, seconds: f64) -> Stop {
    let original = spec.stop.clone();
    let evaluations = spec.search.as_ref().map_or(1, |s| s.max_evaluations);
    let n = (spec.conditions().len() * evaluations).max(1);
    spec.stop = original.clone().or(Stop::max_time(Duration::from_secs_f64(seconds / n as f64)));
    original
}

/// The provenance of a run: the version of comnum, the seed, the threads, the
/// confidence, a fingerprint of the specification and the reports written.
pub fn provenance(spec: &Spec, opts: &RunOptions) -> Json {
    json!({
        "comnum": env!("CARGO_PKG_VERSION"),
        "seed": spec.seed,
        "threads": opts.threads.unwrap_or(spec.threads),
        "confidence": spec.confidence,
        "fingerprint": fingerprint(&spec.source.to_string()),
        "reports": {
            "csv": opts.csv.as_ref().map(|p| p.display().to_string()),
            "json": opts.json.as_ref().map(|p| p.display().to_string()),
        },
    })
}

/// FNV-1a (64 bits) of a text, in hexadecimal: the same specification gives the same
/// fingerprint on every machine.
pub fn fingerprint(text: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// One condition of a run: its values, how it ended, its metrics with their intervals,
/// and whether the budget stopped it before the rule of the specification (`original`)
/// was met.
pub fn condition(r: &ConditionResult, confidence: f64, original: Option<&Stop>) -> Json {
    let values: Map<String, Json> = r
        .condition
        .entries()
        .iter()
        .map(|(k, v)| (k.clone(), serde_json::from_str(&value_json(v)).unwrap_or(Json::Null)))
        .collect();
    let metrics: Map<String, Json> = r
        .record
        .iter()
        .map(|(k, e)| {
            let (lo, hi) = e.interval(confidence);
            (k.clone(), json!({"value": finite(e.value()), "lo": finite(lo), "hi": finite(hi), "count": e.count(), "total": e.total()}))
        })
        .collect();
    let time_limited = original.is_some_and(|s| {
        let p = Progress {
            condition: r.index,
            trials: r.trials,
            record: &r.record,
            elapsed: Duration::from_secs_f64(r.wall_s),
        };
        r.status.as_str() == "completed" && !s.is_met(&p)
    });
    json!({
        "condition": values,
        "status": r.status.as_str(),
        "trials": r.trials,
        "wall_s": r.wall_s,
        "time_limited": time_limited,
        "metrics": metrics,
    })
}

/// A number, or null when it is not finite (JSON has no infinity).
fn finite(x: f64) -> Json {
    if x.is_finite() { json!(x) } else { Json::Null }
}

/// One operating point of a search.
pub fn operating_point(r: &SearchResult) -> Json {
    let values: Map<String, Json> = r
        .condition
        .entries()
        .iter()
        .map(|(k, v)| (k.clone(), serde_json::from_str(&value_json(v)).unwrap_or(Json::Null)))
        .collect();
    json!({
        "condition": values,
        "along": r.along,
        "metric": r.metric,
        "target": r.target,
        "value": finite(r.value),
        "lo": finite(r.lo),
        "hi": finite(r.hi),
        "converged": r.converged,
        "evaluations": r.evaluations.len(),
        "trials": r.evaluations.iter().map(|e| e.trials).sum::<u64>(),
    })
}

/// One check of `verify`.
pub fn check_json(c: &Check) -> Json {
    json!({
        "name": c.name,
        "source": c.source,
        "condition": c.condition,
        "metric": c.metric,
        "expected": finite(c.expected),
        "measured": finite(c.measured.value),
        "lo": finite(c.measured.lo),
        "hi": finite(c.measured.hi),
        "passed": c.passed,
    })
}

/// The kind of an error: `config`, `shape`, `numerical`, `condition`, `io` (`other` for a
/// kind added later).
pub fn error_kind(e: &Error) -> &'static str {
    match e {
        Error::Config(_) => "config",
        Error::Shape(_) => "shape",
        Error::Numerical(_) => "numerical",
        Error::Condition(_) => "condition",
        Error::Io(_) => "io",
        _ => "other",
    }
}

/// An error as JSON: its kind, its message, a hint (the command that documents what was
/// wrong) and, for a misspelt block or parameter, the closest name.
pub fn error(kind: &str, message: &str) -> Json {
    let (hint, suggestion) = advice(message);
    json!({"kind": kind, "message": message, "hint": hint, "suggestion": suggestion})
}

/// The hint and the suggestion for an error message.
fn advice(message: &str) -> (String, Option<String>) {
    let all = blocks::registry();
    if let Some(name) = quoted_after(message, "unknown block \"", '"') {
        let names: Vec<&str> = all.iter().map(|b| b.name).collect();
        let best = closest(&name, &names).map(|b| format!("type: {b}"));
        return ("`comnum blocks --json` lists the blocks and their parameters".into(), best);
    }
    if let (Some(block), Some(field)) =
        (quoted_after(message, "block \"", '"'), quoted_after(message, "unknown field `", '`'))
    {
        let hint = format!("`comnum blocks {block} --json` documents the parameters of {block}");
        let own: Vec<&str> =
            all.iter().find(|b| b.name == block).map(|b| b.params.iter().map(|p| p.name).collect()).unwrap_or_default();
        if let Some(best) = closest(&field, &own) {
            return (hint, Some(format!("{best} (a parameter of {block})")));
        }
        // the parameter of another block: the block meant may be that one
        let others: Vec<&str> = all
            .iter()
            .filter(|b| b.name != block && b.params.iter().any(|p| p.name == field))
            .map(|b| b.name)
            .collect();
        return (hint, others.first().map(|b| format!("type: {b} (it takes {field})")));
    }
    if ["YAML:", "JSON:", "TOML:"].iter().any(|f| message.contains(f)) {
        return (
            "a specification is a mapping with name, grid, link (a list of blocks), metrics and stop; `comnum schema` gives the JSON Schema of a block".into(),
            None,
        );
    }
    if ["unknown option", "unknown command", "no specification given", "needs a", "is not a number", "must be a number"]
        .iter()
        .any(|m| message.contains(m))
    {
        return ("`comnum --help` prints the usage".into(), None);
    }
    if message.contains("No such file") || message.contains("not found") {
        return ("check the path of the specification, or pass `-` to read it from standard input".into(), None);
    }
    ("`comnum check SPEC` builds every condition and runs one trial, without writing anything".into(), None)
}

/// The text between `prefix` and the next `end` in `message`.
fn quoted_after(message: &str, prefix: &str, end: char) -> Option<String> {
    let start = message.find(prefix)? + prefix.len();
    let rest = &message[start..];
    Some(rest[..rest.find(end)?].to_string())
}

/// The name of `names` closest to `word` by edit distance, if within a third of its
/// length (at least 2).
fn closest<'a>(word: &str, names: &[&'a str]) -> Option<&'a str> {
    let limit = (word.chars().count() / 3).max(2);
    names.iter().map(|n| (edit_distance(word, n), *n)).filter(|(d, _)| *d <= limit).min().map(|(_, n)| n)
}

/// The Levenshtein distance between two words.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cur = row[j + 1];
            row[j + 1] = (prev + usize::from(ca != *cb)).min(row[j] + 1).min(cur + 1);
            prev = cur;
        }
    }
    row[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_misspelt_block_or_parameter_gets_the_closest_name() {
        let e = error("config", "link block 2: unknown block \"awgnn\": one of source, awgn");
        assert_eq!(e["suggestion"], "type: awgn");
        // a parameter of another block: the block that takes it
        let e = error("config", "link block 2: block \"awgn\": unknown field `ebn0_db`, expected one of `snr_db`");
        assert_eq!(e["suggestion"], "type: awgn_ebn0 (it takes ebn0_db)");
        let e = error("config", "link block 2: block \"awgn\": unknown field `snr_bd`, expected one of `snr_db`");
        assert_eq!(e["suggestion"], "snr_db (a parameter of awgn)");
        let e = error("config", "b.yaml: YAML: did not find expected node content");
        assert!(e["hint"].as_str().unwrap().contains("comnum schema") && e["suggestion"].is_null());
        assert!(error("usage", "unknown option --bogus")["hint"].as_str().unwrap().contains("--help"));
    }

    #[test]
    fn the_budget_is_300_s_in_json_mode_and_0_removes_it() {
        assert_eq!(budget(None, true), Some(DEFAULT_BUDGET_S));
        assert_eq!(budget(None, false), None);
        assert_eq!(budget(Some(12.0), false), Some(12.0));
        assert_eq!(budget(Some(0.0), true), None);
    }

    #[test]
    fn the_fingerprint_is_fnv_1a() {
        // the test vectors of FNV-1a 64
        assert_eq!(fingerprint(""), "cbf29ce484222325");
        assert_eq!(fingerprint("a"), "af63dc4c8601ec8c");
        assert_eq!(edit_distance("kitten", "sitting"), 3);
    }
}
