//! # vokra-parity
//!
//! Numerical-parity test harness for Vokra (NFR-QL-01). It hosts a tiny,
//! dependency-free fixture reader (the workspace keeps zero third-party crates,
//! so no `serde` / JSON) plus the per-op parity suites under `tests/`.
//!
//! Fixtures are produced by `tests/parity/gen_parity_fixtures.py` and committed
//! under `fixtures/m0-04/`, so CI runs `cargo test` only. The M0-04 criteria
//! and regeneration steps are in `tests/parity/README.md`.
//!
//! Parity suites by owning WP: M0-04 (STFT/Mel — this crate's current
//! fixtures), M0-05 (Silero VAD), M0-06 (Whisper base), M0-07 (piper-plus TTS).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// FP32 absolute tolerance mandated by NFR-QL-01.
pub const FP32_ATOL: f64 = 0.01;

/// One parsed fixture file: a flat `key = value` map (see the module docs of
/// `gen_parity_fixtures.py` for the format). Array values are whitespace-
/// separated floats.
#[derive(Debug, Clone)]
pub struct Fixture {
    /// Path the fixture was loaded from (for diagnostics).
    pub path: PathBuf,
    fields: HashMap<String, String>,
}

impl Fixture {
    /// Parses a fixture file. Blank lines and `#` comments are ignored.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let text = fs::read_to_string(path)?;
        let mut fields = HashMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                fields.insert(k.trim().to_owned(), v.trim().to_owned());
            }
        }
        Ok(Self {
            path: path.to_owned(),
            fields,
        })
    }

    /// Returns the raw string for `key`.
    ///
    /// # Panics
    ///
    /// Panics if `key` is absent (a malformed fixture is a test bug).
    pub fn get(&self, key: &str) -> &str {
        self.fields
            .get(key)
            .unwrap_or_else(|| panic!("fixture {:?} missing key `{key}`", self.path))
    }

    /// Returns the raw string for `key`, if present.
    pub fn try_get(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(String::as_str)
    }

    /// Parses `key` as a `usize`.
    pub fn usize(&self, key: &str) -> usize {
        self.get(key)
            .parse()
            .unwrap_or_else(|_| panic!("fixture {:?} key `{key}` is not usize", self.path))
    }

    /// Parses `key` as an `f64`. Uses locale-independent [`str::parse`]
    /// (NFR-RL-01: never `strtod`).
    pub fn f64(&self, key: &str) -> f64 {
        self.get(key)
            .parse()
            .unwrap_or_else(|_| panic!("fixture {:?} key `{key}` is not f64", self.path))
    }

    /// Parses `key` as a whitespace-separated `Vec<f32>`. Uses locale-
    /// independent [`str::parse`] (NFR-RL-01: never `strtod`).
    pub fn floats(&self, key: &str) -> Vec<f32> {
        self.get(key)
            .split_whitespace()
            .map(|t| {
                t.parse::<f32>().unwrap_or_else(|_| {
                    panic!("fixture {:?} key `{key}`: bad float {t:?}", self.path)
                })
            })
            .collect()
    }
}

/// The `fixtures/m0-04` root inside this crate.
pub fn fixtures_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("m0-04")
}

/// Loads every `*.txt` fixture in `fixtures/m0-04/<subdir>`, sorted by file
/// name. Returns an empty vector if the directory does not exist (so
/// env-gated `#[ignore]` suites can skip cleanly).
pub fn load_dir(subdir: &str) -> Vec<Fixture> {
    let dir = fixtures_root().join(subdir);
    let Ok(entries) = fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    paths.sort();
    paths
        .iter()
        .map(|p| Fixture::load(p).unwrap_or_else(|e| panic!("load {p:?}: {e}")))
        .collect()
}

/// Asserts `got` matches `expected` element-wise within [`FP32_ATOL`].
///
/// # Panics
///
/// Panics with the offending index / values on length or tolerance mismatch.
pub fn assert_close(got: &[f32], expected: &[f32], context: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{context}: length {} != expected {}",
        got.len(),
        expected.len()
    );
    let atol = FP32_ATOL as f32;
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert!(
            (g - e).abs() <= atol,
            "{context}: index {i}: {g} vs {e} (atol {atol})"
        );
    }
}
