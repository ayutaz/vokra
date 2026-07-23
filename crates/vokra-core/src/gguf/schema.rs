//! GGUF **schema generation** stamping and staleness reporting.
//!
//! # Why this exists
//!
//! A Vokra GGUF is not just tensors — loaders depend on `vokra.*` metadata
//! *groups* being present. When a converter learns to emit a new group, files
//! produced before that change are still structurally valid GGUFs: they parse,
//! they carry the tensors the older loader wanted, and nothing about them looks
//! wrong. They are simply **incomplete for the current code**.
//!
//! That failure mode was observed for real (2026-07-22). A cached `mimi.gguf`
//! carried 319 tensors and 9 metadata keys — only
//! `vokra.mimi.{n_codebooks,codebook_size,d_model}`, with no
//! `vokra.mimi.seanet.*` group. Re-converting the same upstream checkpoint with
//! the current converter produced 603 tensors and 36 keys with a bindable PCM
//! neural chain. One consumer rejected the stale file loudly; another silently
//! fell back to a synthesized bridge and produced audio with no real semantics.
//! Nothing anywhere said "this file is from an older converter".
//!
//! # What this module does — and deliberately does not do
//!
//! It records a **generation number** ([`SCHEMA_VERSION`]) in every GGUF the
//! converter writes, and lets a loader ask what generation a file is
//! ([`schema_version`], [`describe`]).
//!
//! It does **not** hard-fail on old files. Every GGUF converted before this
//! landed lacks the key, so refusing them would break every existing artifact
//! for a diagnostic. [`SchemaGeneration::PreStamping`] is a first-class answer.
//! Enforcement belongs where a loader knows a specific group is required — see
//! [`stale_group_hint`], which turns "this group is missing" into a message
//! that names the generation and tells the user to re-convert.
//!
//! # Bumping
//!
//! Raise [`SCHEMA_VERSION`] when a converter begins emitting a group that
//! loaders may come to depend on, and add a line to its changelog. Do not tie
//! it to `CARGO_PKG_VERSION`: that string has been `0.1.0-alpha.0` since M0 and
//! would have been identical on both sides of the incident above.

use alloc::borrow::ToOwned;
use alloc::format;
use alloc::string::String;

use crate::gguf::chunks::{KEY_SCHEMA_PRODUCER, KEY_SCHEMA_VERSION};
use crate::gguf::{GgufFile, GgufMetadataValue};

/// Current Vokra GGUF schema generation.
///
/// | gen | meaning |
/// |-----|---------|
/// | (absent) | pre-stamping: any GGUF converted before 2026-07-22 |
/// | 1 | first stamped generation — Mimi dual-write (`vokra.mimi.seanet.*` + the `mimi.enc.*`/`mimi.dec.*` neural chain) is present whenever the upstream checkpoint carries it |
///
/// Bump this **and document the row** when a converter starts emitting a group
/// loaders may depend on.
pub const SCHEMA_VERSION: u32 = 1;

/// What generation a GGUF was written by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaGeneration {
    /// No `vokra.schema.version` key: written before schema stamping existed.
    /// Not an error — but a file that is missing a group the current code
    /// expects is very likely stale rather than malformed.
    PreStamping,
    /// Written by a stamped converter, at this generation.
    Stamped(u32),
}

impl SchemaGeneration {
    /// Whether this file predates the current [`SCHEMA_VERSION`] — i.e. it may
    /// lack groups the current loaders expect.
    #[must_use]
    pub fn is_older_than_current(self) -> bool {
        match self {
            Self::PreStamping => true,
            Self::Stamped(v) => v < SCHEMA_VERSION,
        }
    }

    /// Short human-readable form for error and log messages.
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::PreStamping => "pre-stamping (no vokra.schema.version)".to_owned(),
            Self::Stamped(v) => format!("schema gen {v}"),
        }
    }
}

/// Reads the schema generation a GGUF was written at.
///
/// A non-`UINT32` value is treated as [`SchemaGeneration::PreStamping`]: the
/// key is diagnostic, and a malformed diagnostic must not be able to fail a
/// load that would otherwise succeed.
#[must_use]
pub fn schema_version(file: &GgufFile) -> SchemaGeneration {
    match file.get(KEY_SCHEMA_VERSION) {
        Some(GgufMetadataValue::U32(v)) => SchemaGeneration::Stamped(*v),
        _ => SchemaGeneration::PreStamping,
    }
}

/// The producer string, when the file carries one (diagnostics only).
#[must_use]
pub fn producer(file: &GgufFile) -> Option<&str> {
    match file.get(KEY_SCHEMA_PRODUCER) {
        Some(GgufMetadataValue::String(s)) => Some(s.as_str()),
        _ => None,
    }
}

/// One-line provenance summary for logs and error messages, e.g.
/// `"schema gen 1 (vokra-convert 0.1.0-alpha.0)"`.
#[must_use]
pub fn describe(file: &GgufFile) -> String {
    match producer(file) {
        Some(p) => format!("{} ({p})", schema_version(file).label()),
        None => schema_version(file).label(),
    }
}

/// Builds the message a loader should use when a **required metadata group is
/// absent**, attributing it to converter age when that is the likely cause.
///
/// This is the enforcement half of schema stamping: the generation number on
/// its own only tells you how old a file is, which nobody reads. Folding it
/// into the error a missing group already produces is what makes staleness
/// visible at the moment it actually costs something.
///
/// `group` is the missing key prefix (e.g. `"vokra.mimi.seanet."`), and
/// `re_convert` the command that regenerates the file.
#[must_use]
pub fn stale_group_hint(file: &GgufFile, group: &str, re_convert: &str) -> String {
    let generation = schema_version(file);
    let provenance = describe(file);
    if generation.is_older_than_current() {
        format!(
            "the `{group}*` metadata group is missing and this GGUF is \
             {provenance}, older than the current schema gen {SCHEMA_VERSION} — \
             it was almost certainly written by an earlier converter. \
             Re-convert it: {re_convert}"
        )
    } else {
        format!(
            "the `{group}*` metadata group is missing even though this GGUF is \
             {provenance} (current). This is not converter age — the source \
             checkpoint most likely does not carry that component. \
             Regenerate with: {re_convert}"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufBuilder;

    /// Hand-rolls GGUF bytes **without** [`GgufBuilder`], which is the only way
    /// to obtain an unstamped file now that the writer stamps unconditionally.
    /// This is what a real pre-2026-07-22 artifact on disk looks like.
    fn raw_gguf(kvs: &[(&str, &str)]) -> GgufFile {
        const STRING: u32 = 8;
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes()); // version
        out.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, v) in kvs {
            out.extend_from_slice(&(k.len() as u64).to_le_bytes());
            out.extend_from_slice(k.as_bytes());
            out.extend_from_slice(&STRING.to_le_bytes());
            out.extend_from_slice(&(v.len() as u64).to_le_bytes());
            out.extend_from_slice(v.as_bytes());
        }
        GgufFile::parse(out).expect("hand-rolled GGUF must parse")
    }

    /// Every GGUF converted before stamping existed lacks the key. Reporting
    /// that as `PreStamping` — rather than erroring, or defaulting to 0 and
    /// pretending it is a real generation — is what keeps those artifacts
    /// loadable instead of breaking every file already on disk.
    #[test]
    fn an_unstamped_file_is_pre_stamping_not_an_error() {
        let f = raw_gguf(&[("vokra.model.arch", "mimi")]);
        assert_eq!(schema_version(&f), SchemaGeneration::PreStamping);
        assert!(schema_version(&f).is_older_than_current());
        assert_eq!(producer(&f), None);
        assert!(describe(&f).contains("pre-stamping"), "{}", describe(&f));
    }

    /// The stamp is written at the writer's single choke point, so **no**
    /// builder-produced GGUF can escape unstamped — that universality is the
    /// property that makes staleness detectable at all.
    #[test]
    fn every_builder_written_gguf_is_stamped() {
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "mimi");
        let f = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert_eq!(
            schema_version(&f),
            SchemaGeneration::Stamped(SCHEMA_VERSION)
        );
        assert!(!schema_version(&f).is_older_than_current());
        let p = producer(&f).expect("producer stamped");
        assert!(
            p.contains(env!("CARGO_PKG_NAME")) && p.contains(env!("CARGO_PKG_VERSION")),
            "producer must identify the build that wrote the bytes: {p}"
        );
    }

    /// A caller cannot forge the stamp: whatever it supplies is dropped and
    /// replaced by the writer's own values. A stamp that could be set by the
    /// caller would describe intent, not provenance, and would be worthless as
    /// staleness evidence.
    #[test]
    fn a_caller_supplied_stamp_cannot_spoof_the_writer() {
        let mut b = GgufBuilder::new();
        b.add_u32(KEY_SCHEMA_VERSION, 9_999);
        b.add_string(KEY_SCHEMA_PRODUCER, "definitely-not-vokra 42");
        let f = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert_eq!(
            schema_version(&f),
            SchemaGeneration::Stamped(SCHEMA_VERSION),
            "the writer's generation must win over a caller-supplied one"
        );
        let p = producer(&f).expect("producer stamped");
        assert!(
            !p.contains("definitely-not-vokra"),
            "a forged producer must not survive: {p}"
        );
    }

    /// A malformed diagnostic key must not be able to fail a load that would
    /// otherwise succeed — it degrades to `PreStamping`.
    #[test]
    fn a_wrong_typed_version_key_degrades_instead_of_failing() {
        let f = raw_gguf(&[(KEY_SCHEMA_VERSION, "not-a-number")]);
        assert_eq!(schema_version(&f), SchemaGeneration::PreStamping);
    }

    /// The hint must separate the two causes: an old file (re-converting fixes
    /// it) from a current file whose source checkpoint simply lacks the
    /// component (re-converting will not). Blaming staleness for a missing
    /// component would send the user down the wrong path.
    #[test]
    fn the_stale_hint_separates_converter_age_from_a_missing_component() {
        let cmd = "vokra-cli convert --model mimi --input <ckpt> --output <out.gguf>";

        let old = raw_gguf(&[("vokra.model.arch", "mimi")]);
        let msg = stale_group_hint(&old, "vokra.mimi.seanet.", cmd);
        assert!(
            msg.contains("vokra.mimi.seanet.*"),
            "names the group: {msg}"
        );
        assert!(
            msg.contains("earlier converter") && msg.contains(cmd),
            "an old file must be attributed to converter age and give the fix: {msg}"
        );

        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "mimi");
        let current = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let msg = stale_group_hint(&current, "vokra.mimi.seanet.", cmd);
        assert!(
            msg.contains("not converter age"),
            "a current file must NOT be blamed on staleness: {msg}"
        );
    }
}
