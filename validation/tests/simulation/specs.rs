//! The quick validation specifications of `specs/validation/` that CI does not verify
//! with `comnum verify` (the list `ci_specs` of the justfile, which CI runs with `just
//! verify-ci`), run by `comnum_spec::verify::verify_spec` as `comnum verify` runs them;
//! and the check of that list.
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "test code: a failure is a test failure")]
#![cfg(feature = "spec")]

use comnum::spec::{RunOptions, Spec, verify};
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn check(name: &str) {
    let spec = Spec::from_path(root().join("specs/validation").join(name)).unwrap();
    let (_, checks) = verify::verify_spec(&spec, &RunOptions::default()).unwrap();
    assert!(!checks.is_empty(), "{name}: nothing checked");
    let failed: Vec<String> = checks
        .iter()
        .filter(|c| !c.passed)
        .map(|c| format!("{} | {}: expected {}, measured {:?}", c.name, c.condition, c.expected, c.measured))
        .collect();
    assert!(failed.is_empty(), "{name}:\n{}", failed.join("\n"));
}

/// The specifications that CI verifies: `ci_specs` of the justfile.
fn ci_specs() -> Vec<String> {
    let justfile = std::fs::read_to_string(root().join("justfile")).unwrap();
    let line = justfile.lines().find(|l| l.starts_with("ci_specs :=")).expect("ci_specs in the justfile");
    line.split('"').nth(1).unwrap().split_whitespace().map(String::from).collect()
}

#[test]
fn the_specifications_that_ci_verifies_exist_and_are_verified_once() {
    let specs = ci_specs();
    assert!(!specs.is_empty(), "no specification in ci_specs");
    for s in &specs {
        assert!(root().join(s).is_file(), "{s} does not exist");
    }
    // the specifications of this file are not verified by CI a second time
    for name in QUICK {
        assert!(!specs.iter().any(|s| s.ends_with(name)), "{name} is verified by CI already");
    }
}

/// The quick specifications that CI does not verify.
const QUICK: [&str; 1] = ["required_snr.yaml"];

#[test]
fn operating_points_follow_the_inverted_closed_forms() {
    check(QUICK[0]);
}
