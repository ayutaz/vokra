//! `MisakiG2p` — a text-to-phoneme-ids adapter for Vokra's native Kokoro TTS,
//! backed by the upstream Python [`misaki`](https://github.com/hexgrad/misaki)
//! package (the same G2P Kokoro-82M was trained against).
//!
//! # Why not pure Rust?
//!
//! misaki is Python-only upstream. Re-implementing it in Rust would drift
//! from Kokoro's training phoneme distribution by construction — subtle
//! stress / vowel-reduction differences that a synthetic re-implementation
//! cannot detect, because the reference IS the Python code. This wrapper
//! shells out to the real thing.
//!
//! # Pipeline
//!
//! ```text
//! text --(subprocess: python misaki_bridge.py)--> IPA phoneme string
//!      --(this crate: KokoroConfig.phoneme_symbols lookup)--> phoneme ids
//!      --> KokoroTts::synthesize_phonemes --> PCM
//! ```
//!
//! # Zero-dependency posture
//!
//! This crate lives outside the Vokra root workspace (see `Cargo.toml`), so
//! its transitive Python-adjacent deps (none today, but even a future
//! `serde_json` would count) cannot leak into `Cargo.lock`. The Kokoro
//! runtime it drives (`vokra-models::kokoro`) stays zero-dependency.
//!
//! # Fail-closed
//!
//! An unknown language, a missing misaki install, or a misaki return whose
//! phoneme character is not in `KokoroConfig::phoneme_symbols` all surface
//! as a loud `VokraError::InvalidArgument` (FR-EX-08: never a silent skip).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use vokra_core::{Result, VokraError};
use vokra_models::kokoro::KokoroTts;

/// The five misaki language modules — one per upstream sub-package. Anything
/// outside this set is out of scope for the bridge (`en` / `en-gb` are two
/// tunings of the same `misaki.en.G2P`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MisakiLang {
    /// American English (default: `en.G2P(trf=False, british=False)`).
    En,
    /// British English (`en.G2P(trf=False, british=True)`).
    EnGb,
    /// Japanese (`misaki.ja.G2P`).
    Ja,
    /// Mandarin Chinese (`misaki.zh.G2P`).
    Zh,
    /// Korean (`misaki.ko.G2P`).
    Ko,
}

impl MisakiLang {
    /// The exact string the Python bridge expects on `--lang`.
    pub fn as_arg(self) -> &'static str {
        match self {
            Self::En => "en",
            Self::EnGb => "en-gb",
            Self::Ja => "ja",
            Self::Zh => "zh",
            Self::Ko => "ko",
        }
    }

    /// Parse from the CLI flag (case-insensitive, `_` and `-` normalised).
    pub fn parse(s: &str) -> Result<Self> {
        let norm: String = s
            .trim()
            .to_ascii_lowercase()
            .chars()
            .map(|c| if c == '_' { '-' } else { c })
            .collect();
        match norm.as_str() {
            "en" | "english" | "en-us" => Ok(Self::En),
            "en-gb" | "british" => Ok(Self::EnGb),
            "ja" | "japanese" | "jp" => Ok(Self::Ja),
            "zh" | "chinese" | "cn" | "mandarin" => Ok(Self::Zh),
            "ko" | "korean" | "kr" => Ok(Self::Ko),
            _ => Err(VokraError::InvalidArgument(format!(
                "unknown misaki language {s:?}; expected one of en / en-gb / ja / zh / ko"
            ))),
        }
    }
}

/// A shell-out to the misaki Python bridge, wired to a specific Kokoro
/// GGUF's phoneme symbol table.
pub struct MisakiG2p {
    python: PathBuf,
    script: PathBuf,
    /// Maps a Kokoro phoneme *symbol string* (usually a single character, but
    /// upstream's table also stores the marker tokens `_`, `^`, `$` etc.) to
    /// its 0-based id. Duplicate symbols in the source table are rejected at
    /// construction time so a misconverted GGUF cannot silently pick one.
    symbol_to_id: HashMap<String, i64>,
}

impl MisakiG2p {
    /// Bind to a loaded [`KokoroTts`]. `python` defaults to `python3` on the
    /// PATH; pass an explicit path (e.g. a virtualenv's) when the system
    /// interpreter cannot find `misaki`.
    ///
    /// The bridge script is located relative to this crate's manifest dir
    /// (`../python/misaki_bridge.py`), which the standard `cargo run` path
    /// resolves. When the binary is copied elsewhere, use
    /// [`Self::from_kokoro_with_paths`] to override.
    pub fn from_kokoro(kokoro: &KokoroTts, python: Option<PathBuf>) -> Result<Self> {
        Self::from_kokoro_with_paths(kokoro, python, None)
    }

    /// Full-control constructor: caller supplies both the interpreter and
    /// the bridge script path.
    pub fn from_kokoro_with_paths(
        kokoro: &KokoroTts,
        python: Option<PathBuf>,
        script: Option<PathBuf>,
    ) -> Result<Self> {
        let python = python.unwrap_or_else(|| PathBuf::from("python3"));
        let script = script.unwrap_or_else(default_script_path);
        let symbol_to_id = build_symbol_map(&kokoro.config().phoneme_symbols)?;
        Ok(Self {
            python,
            script,
            symbol_to_id,
        })
    }

    /// Text → phoneme ids for the target `lang`. Fails loudly on any missing
    /// misaki, any error the upstream G2P raises, or any phoneme character
    /// misaki produced that the Kokoro voice's table does not carry.
    pub fn phonemize(&self, text: &str, lang: MisakiLang) -> Result<Vec<i64>> {
        let phonemes = self.phonemize_string(text, lang)?;
        self.phonemes_to_ids(&phonemes)
    }

    /// Text → IPA phoneme string (the raw misaki output). Useful for tests
    /// and for `--dump` flows where the id mapping is inspected separately.
    pub fn phonemize_string(&self, text: &str, lang: MisakiLang) -> Result<String> {
        let output = Command::new(&self.python)
            .arg(&self.script)
            .arg("--lang")
            .arg(lang.as_arg())
            .arg("--text")
            .arg(text)
            .output()
            .map_err(VokraError::Io)?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(VokraError::InvalidArgument(format!(
                "misaki bridge exit {}: {}",
                output.status.code().unwrap_or(-1),
                stderr.trim(),
            )));
        }
        let stdout = std::str::from_utf8(&output.stdout).map_err(|e| {
            VokraError::InvalidArgument(format!("misaki bridge stdout is not UTF-8: {e}"))
        })?;
        parse_phonemes_field(stdout)
    }

    /// Map an IPA phoneme string to the Kokoro id sequence — char-by-char,
    /// matching upstream's char-level tokeniser over `phoneme_symbols`.
    /// Missing characters produce a loud error (never a silent drop).
    pub fn phonemes_to_ids(&self, phonemes: &str) -> Result<Vec<i64>> {
        let mut ids = Vec::with_capacity(phonemes.chars().count());
        for c in phonemes.chars() {
            let key = c.to_string();
            let id = self.symbol_to_id.get(&key).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "misaki produced phoneme {c:?} (U+{u:04X}) not in Kokoro's phoneme_symbols; \
                     either the voice was converted without --config (missing symbol table) \
                     or misaki emitted a symbol outside this Kokoro voice's training set",
                    u = c as u32,
                ))
            })?;
            ids.push(*id);
        }
        Ok(ids)
    }

    /// Number of distinct symbols known to this bridge's Kokoro voice. Small
    /// diagnostic — the `--dump` CLI reports it so a misconverted voice is
    /// obvious.
    pub fn symbol_count(&self) -> usize {
        self.symbol_to_id.len()
    }
}

/// Locates `python/misaki_bridge.py` relative to this crate's manifest dir
/// (`CARGO_MANIFEST_DIR`). Correct for `cargo run` and for a locally-built
/// binary while the source tree is intact.
fn default_script_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join("misaki_bridge.py")
}

/// Builds a `symbol → id` map from a Kokoro `phoneme_symbols` array. Empty
/// slots (unused id gaps) are skipped; a duplicate symbol is a hard error.
fn build_symbol_map(symbols: &[String]) -> Result<HashMap<String, i64>> {
    let mut map: HashMap<String, i64> = HashMap::with_capacity(symbols.len());
    for (i, s) in symbols.iter().enumerate() {
        if s.is_empty() {
            continue;
        }
        if let Some(prev) = map.insert(s.clone(), i as i64) {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro phoneme_symbols has duplicate entry {s:?} at ids {prev} and {i}"
            )));
        }
    }
    if map.is_empty() {
        return Err(VokraError::InvalidArgument(
            "kokoro phoneme_symbols is empty — was the voice converted with --config?".to_owned(),
        ));
    }
    Ok(map)
}

/// Parse the single-line JSON produced by `python/misaki_bridge.py`:
/// `{"lang":"…","phonemes":"…"}`. We do this by hand to keep the crate free
/// of a `serde_json` dependency (Cargo.lock is isolated but even isolated
/// deps have a maintenance cost; the shape here is fixed and small).
fn parse_phonemes_field(stdout: &str) -> Result<String> {
    const KEY: &str = "\"phonemes\":\"";
    let start = stdout.find(KEY).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "misaki bridge stdout missing `\"phonemes\":\"` key: {stdout:?}"
        ))
    })?;
    let after = &stdout[start + KEY.len()..];
    // misaki IPA output does not contain ASCII `"`, and the JSON writer runs
    // with `ensure_ascii=False`, so no `\uXXXX` escapes appear either. The
    // first `"` is the value terminator; anything else is a bridge bug.
    let end = after.find('"').ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "misaki bridge stdout has unterminated `phonemes` value: {stdout:?}"
        ))
    })?;
    Ok(after[..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lang_parse_accepts_common_spellings() {
        for (s, expected) in [
            ("en", MisakiLang::En),
            ("EN", MisakiLang::En),
            ("English", MisakiLang::En),
            ("en-us", MisakiLang::En),
            ("en-gb", MisakiLang::EnGb),
            ("british", MisakiLang::EnGb),
            ("ja", MisakiLang::Ja),
            ("Japanese", MisakiLang::Ja),
            ("jp", MisakiLang::Ja),
            ("zh", MisakiLang::Zh),
            ("mandarin", MisakiLang::Zh),
            ("ko", MisakiLang::Ko),
            ("Korean", MisakiLang::Ko),
        ] {
            assert_eq!(MisakiLang::parse(s).unwrap(), expected, "{s}");
        }
    }

    #[test]
    fn lang_parse_rejects_unknown() {
        assert!(MisakiLang::parse("de").is_err());
        assert!(MisakiLang::parse("").is_err());
        assert!(MisakiLang::parse("ipa").is_err());
    }

    #[test]
    fn symbol_map_builds_from_canonical_kokoro_prefix() {
        // The canonical Kokoro symbol table starts with `["_", "^", "$", "a", …]`;
        // we only exercise the mechanics here.
        let symbols: Vec<String> = ["_", "^", "$", "a", "b", "c"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let map = build_symbol_map(&symbols).unwrap();
        assert_eq!(map.get("_"), Some(&0));
        assert_eq!(map.get("^"), Some(&1));
        assert_eq!(map.get("$"), Some(&2));
        assert_eq!(map.get("a"), Some(&3));
        assert_eq!(map.get("c"), Some(&5));
    }

    #[test]
    fn symbol_map_rejects_duplicates() {
        let symbols: Vec<String> = ["a", "b", "a"].iter().map(|s| (*s).to_owned()).collect();
        let err = build_symbol_map(&symbols).unwrap_err();
        assert!(err.to_string().contains("duplicate entry"), "{err}");
    }

    #[test]
    fn symbol_map_rejects_empty_table() {
        let empty: Vec<String> = Vec::new();
        let err = build_symbol_map(&empty).unwrap_err();
        assert!(
            err.to_string().contains("phoneme_symbols is empty"),
            "{err}"
        );
    }

    #[test]
    fn symbol_map_skips_empty_slots() {
        // Missing ids in the middle stay unmapped (kokoro leaves them as `""`).
        let symbols: Vec<String> = ["_", "", "a", ""].iter().map(|s| (*s).to_owned()).collect();
        let map = build_symbol_map(&symbols).unwrap();
        assert_eq!(map.get("_"), Some(&0));
        assert_eq!(map.get("a"), Some(&2));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn parse_phonemes_field_extracts_from_canonical_output() {
        let stdout = r#"{"lang": "en", "phonemes": "həˈloʊ"}"#;
        // Note the space after `:` in this canonical output — Python's default
        // `json.dumps` output. We accept both spaced and unspaced variants:
        // the KEY search uses no space, so this test purposely exercises the
        // unspaced form the Python bridge emits.
        let stdout_unspaced = r#"{"lang":"en","phonemes":"həˈloʊ"}"#;
        assert_eq!(parse_phonemes_field(stdout_unspaced).unwrap(), "həˈloʊ");
        // Spaced form should error out cleanly rather than misparse; the
        // Python side is under our control and emits the unspaced form.
        assert!(parse_phonemes_field(stdout).is_err());
    }

    #[test]
    fn parse_phonemes_field_flags_missing_key() {
        let stdout = r#"{"lang":"en","tokens":[]}"#;
        let err = parse_phonemes_field(stdout).unwrap_err();
        assert!(err.to_string().contains("missing"), "{err}");
    }

    #[test]
    fn parse_phonemes_field_flags_unterminated_value() {
        // No closing `"` — should error, not run off the end of the string.
        let stdout = "\"phonemes\":\"həˈloʊ";
        let err = parse_phonemes_field(stdout).unwrap_err();
        assert!(err.to_string().contains("unterminated"), "{err}");
    }

    #[test]
    fn phonemes_to_ids_maps_characters_and_flags_unknowns() {
        let symbol_to_id: HashMap<String, i64> = [
            ("h".to_owned(), 10),
            ("ə".to_owned(), 11),
            ("ˈ".to_owned(), 12),
            ("l".to_owned(), 13),
            ("o".to_owned(), 14),
            ("ʊ".to_owned(), 15),
        ]
        .into_iter()
        .collect();
        let g2p = MisakiG2p {
            python: PathBuf::from("python3"),
            script: PathBuf::new(),
            symbol_to_id,
        };
        assert_eq!(
            g2p.phonemes_to_ids("həˈloʊ").unwrap(),
            vec![10, 11, 12, 13, 14, 15]
        );
        // Unknown char = loud error naming the offender.
        let err = g2p.phonemes_to_ids("hɛlloʊ").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("ɛ") || msg.contains("U+025B"), "{msg}");
    }
}
