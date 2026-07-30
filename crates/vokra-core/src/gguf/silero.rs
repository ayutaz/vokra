//! Silero VAD release-tag metadata (v5 / v6.2.1 provenance).
//!
//! The Silero VAD subgraph (M0-05 / `vokra-vad-micro`) is architecturally
//! stable across upstream tags v5 and v6.2.1 — this was verified against
//! primary source at both tags on 2026-07-30:
//!
//! - **`snakers4/silero-vad` `src/silero_vad/tinygrad_model.py` @ v6.2.1**
//!   declares `Conv1d(1, 258, k=256, stride=128)` (pseudo-STFT for 16 kHz),
//!   `Conv1d(129, 128, k=3, padding=1)` (encoder 0), `Conv1d(128, 64, k=3,
//!   stride=2, padding=1)` (encoder 1), `Conv1d(64, 64, k=3, stride=2,
//!   padding=1)` (encoder 2), `Conv1d(64, 128, k=3, stride=1, padding=1)`
//!   (encoder 3), `LSTMCell(128, 128)`, `Conv1d(128, 1, 1)` (head) —
//!   bit-identical to the v5 SPEC in `crates/vokra-models/src/silero_vad/`.
//! - **`utils_vad.py`** at both tags has identical inference geometry:
//!   `num_samples = 512 if sr == 16000 else 256`, `context_size = 64 if sr ==
//!   16000 else 32`, state shape `torch.zeros((2, batch_size, 128))`.
//! - **`silero_vad.onnx` git blob sha** differs between the tags (v5.1.2 =
//!   `b3e3a900…` / v6.0 = `cb605195…` / v6.2.1 = `80c5592e…`) yet the file
//!   size is identical (2 327 524 bytes at every tag): retrained weights,
//!   same topology.
//!
//! What the release tag therefore controls is **provenance and license
//! attribution**, not tensor names or shapes: the fixture GGUF, the parity
//! reference txt and the license-audit sign-off row all key on the exact
//! upstream tag. This module owns the two-way mapping between the
//! [`chunks::KEY_SILERO_VERSION`] string and the [`SileroVariant`] enum, plus
//! the fail-closed reader and the converter helper.
//!
//! # Fail-closed contract (FR-EX-08)
//!
//! - **Absent key** → [`SileroVariant::V5`]. Every GGUF converted before this
//!   key existed (including the committed fixture
//!   `tests/parity/silero_vad/silero-vad-v5.gguf`) is treated as v5, so
//!   loading the pre-tagging fleet keeps working with no re-conversion.
//! - **Known tag** (`"v5"`, `"v6.2.1"`) → the matching variant.
//! - **Unknown tag** (`"v7"`, `"v6.1"`, anything else) → hard
//!   [`crate::VokraError::ModelLoad`], never a silent V5 fallback. A tag we do
//!   not recognize may imply a topology change this build cannot honor
//!   (CLAUDE.md "ハルシネーション厳禁"), and downstream fallback to v5 in that
//!   case would misread the weights.

#[cfg(not(feature = "std"))]
use alloc::format;

use crate::gguf::{GgufFile, chunks};
use crate::{Result, VokraError};

/// Which Silero VAD release the weights come from.
///
/// The enum tags provenance; both variants currently share the same forward
/// (topology is identical across v5 and v6.2.1 per upstream, documented at
/// the module level). Adding a future variant that *does* diverge in shape
/// changes the loader (per-variant shape checks) and the forward (branch on
/// [`SileroVariant`]), and is a deliberate future decision — no silent
/// fallback exists to hide the mismatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SileroVariant {
    /// Upstream `snakers4/silero-vad` v5 (tags `v5.0` / `v5.1` / `v5.1.1` /
    /// `v5.1.2`, June-October 2024). The default when the metadata key is
    /// absent (backward compatibility with pre-tagging GGUFs).
    V5,
    /// Upstream `snakers4/silero-vad` v6.2.1 (released 2026-02-24: "Make ONNX
    /// Runtime optional"; architecture unchanged from v6.0). Retrained weights
    /// over the v5 topology — same tensor names, same shapes.
    V6_2_1,
}

/// Canonical string used in [`chunks::KEY_SILERO_VERSION`] for
/// [`SileroVariant::V5`].
pub const TAG_V5: &str = "v5";

/// Canonical string used in [`chunks::KEY_SILERO_VERSION`] for
/// [`SileroVariant::V6_2_1`].
pub const TAG_V6_2_1: &str = "v6.2.1";

impl SileroVariant {
    /// The canonical release-tag string this variant is stamped with.
    pub fn tag(self) -> &'static str {
        match self {
            Self::V5 => TAG_V5,
            Self::V6_2_1 => TAG_V6_2_1,
        }
    }

    /// Parses a release-tag string into a variant. Only the two canonical
    /// spellings ([`TAG_V5`] / [`TAG_V6_2_1`]) match; every other value is a
    /// fail-closed error (FR-EX-08 — unknown tags may imply a topology change
    /// this build cannot honor, and silent V5 fallback would misread the
    /// weights).
    pub fn from_tag(tag: &str) -> Result<Self> {
        match tag {
            TAG_V5 => Ok(Self::V5),
            TAG_V6_2_1 => Ok(Self::V6_2_1),
            other => Err(VokraError::ModelLoad(format!(
                "unknown `{}` value `{}`: this build recognizes `{}` and `{}` only",
                chunks::KEY_SILERO_VERSION,
                other,
                TAG_V5,
                TAG_V6_2_1,
            ))),
        }
    }

    /// Reads the release tag from a GGUF's `vokra.silero.version` string
    /// metadata, falling back to [`SileroVariant::V5`] when the key is absent
    /// (pre-tagging backward compatibility). A **present** key with an
    /// unrecognized value is a fail-closed [`VokraError::ModelLoad`]; a
    /// present key of the wrong GGUF value type is likewise an error, never
    /// silently ignored.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        match gguf.get(chunks::KEY_SILERO_VERSION) {
            None => Ok(Self::V5),
            Some(value) => match value.as_str() {
                Some(tag) => Self::from_tag(tag),
                None => Err(VokraError::ModelLoad(format!(
                    "`{}` metadata is present but not a string (got type {:?})",
                    chunks::KEY_SILERO_VERSION,
                    value.value_type(),
                ))),
            },
        }
    }
}

/// Stamps a Silero release tag onto a GGUF builder. Idempotent: calling twice
/// with different variants replaces the previous value (writer semantics of
/// [`crate::gguf::GgufBuilder::add_string`]). Written by the converter, read
/// back by [`SileroVariant::from_gguf`].
///
/// The runtime always emits the tag on new conversions (writing `"v5"`
/// explicitly is fine — round-trip parses back to [`SileroVariant::V5`]).
#[cfg(feature = "std")]
pub fn stamp_variant(builder: &mut crate::gguf::GgufBuilder, variant: SileroVariant) {
    builder.add_string(chunks::KEY_SILERO_VERSION, variant.tag());
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::gguf::GgufBuilder;

    fn to_gguf(b: &GgufBuilder) -> GgufFile {
        GgufFile::parse(b.to_bytes().unwrap()).unwrap()
    }

    #[test]
    fn tag_round_trip_covers_every_variant() {
        for &v in &[SileroVariant::V5, SileroVariant::V6_2_1] {
            let parsed = SileroVariant::from_tag(v.tag()).expect("tag parses");
            assert_eq!(parsed, v);
        }
    }

    #[test]
    fn from_gguf_absent_key_defaults_to_v5() {
        // Pre-tagging GGUFs (the committed fixture + every artifact converted
        // before this key existed) must still load, tagged as V5 — the
        // backward-compat contract.
        let b = GgufBuilder::new();
        let gguf = to_gguf(&b);
        assert_eq!(SileroVariant::from_gguf(&gguf).unwrap(), SileroVariant::V5);
    }

    #[test]
    fn from_gguf_reads_v5_tag_explicitly() {
        let mut b = GgufBuilder::new();
        stamp_variant(&mut b, SileroVariant::V5);
        assert_eq!(
            SileroVariant::from_gguf(&to_gguf(&b)).unwrap(),
            SileroVariant::V5
        );
    }

    #[test]
    fn from_gguf_reads_v6_2_1_tag() {
        let mut b = GgufBuilder::new();
        stamp_variant(&mut b, SileroVariant::V6_2_1);
        assert_eq!(
            SileroVariant::from_gguf(&to_gguf(&b)).unwrap(),
            SileroVariant::V6_2_1
        );
    }

    /// FR-EX-08: unknown tags fail loudly, never a silent V5 fallback that
    /// would misread the weights if the topology diverged.
    #[test]
    fn from_gguf_rejects_unknown_tag() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_SILERO_VERSION, "v7-experimental");
        let e = SileroVariant::from_gguf(&to_gguf(&b)).unwrap_err();
        assert!(
            matches!(&e, VokraError::ModelLoad(m) if m.contains("v7-experimental")),
            "want ModelLoad naming the offending tag, got {e:?}"
        );
        // Both canonical spellings should be surfaced so the caller sees
        // what the build accepts.
        let msg = format!("{e:?}");
        assert!(msg.contains(TAG_V5) && msg.contains(TAG_V6_2_1));
    }

    /// FR-EX-08: a non-string value type is also a hard error — not a
    /// silent fallback (which would misclassify the artifact).
    #[test]
    fn from_gguf_rejects_wrong_value_type() {
        let mut b = GgufBuilder::new();
        b.add_u32(chunks::KEY_SILERO_VERSION, 6);
        let e = SileroVariant::from_gguf(&to_gguf(&b)).unwrap_err();
        assert!(
            matches!(&e, VokraError::ModelLoad(m) if m.contains("not a string")),
            "want ModelLoad naming the type mismatch, got {e:?}"
        );
    }

    /// Idempotence of `stamp_variant`: writing twice keeps the last-written
    /// tag (writer replaces on duplicate key), which matches the "converter
    /// stamps its own tag last" contract.
    #[test]
    fn stamp_variant_is_idempotent_on_repeat() {
        let mut b = GgufBuilder::new();
        stamp_variant(&mut b, SileroVariant::V5);
        stamp_variant(&mut b, SileroVariant::V6_2_1);
        assert_eq!(
            SileroVariant::from_gguf(&to_gguf(&b)).unwrap(),
            SileroVariant::V6_2_1
        );
    }
}
