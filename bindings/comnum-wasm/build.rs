//! Embeds the specifications of `specs/` (the examples) and `specs/validation/` (the
//! validation specifications) in the module, so that the playground lists them without
//! any build step beyond `wasm-pack`: `SPECS` is a list of `(path, text)`, in the order
//! of the menu, the validation specifications after the examples. A specification that
//! reads a file (a `capture` block) is left out: the browser has no file.

use std::fs;
use std::path::Path;

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../specs");
    let mut entries = Vec::new();
    for dir in ["", "validation"] {
        let dir = root.join(dir);
        println!("cargo:rerun-if-changed={}", dir.display());
        let mut files: Vec<_> = fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "yaml"))
            .collect();
        files.sort();
        for f in files {
            println!("cargo:rerun-if-changed={}", f.display());
            let text = fs::read_to_string(&f).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
            if text.contains("type: capture") {
                continue;
            }
            let rel = f.strip_prefix(root.parent().unwrap_or(&root)).unwrap_or(&f).display().to_string();
            entries.push(format!("    ({rel:?}, include_str!({:?})),", f.display()));
        }
    }
    let out = Path::new(&std::env::var("OUT_DIR").unwrap_or_default()).join("specs.rs");
    let src = format!(
        "/// The specifications of the repository: `(path, text)`.\npub(crate) const SPECS: &[(&str, &str)] = &[\n{}\n];\n",
        entries.join("\n")
    );
    fs::write(&out, src).unwrap_or_else(|e| panic!("{}: {e}", out.display()));
    println!("cargo:rerun-if-changed=build.rs");
}
