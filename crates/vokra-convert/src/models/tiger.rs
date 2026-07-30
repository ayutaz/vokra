//! **TIGER** (JusperLee TIGER family): Time-frequency Interleaved Gain
//! Extraction from a Restructured net — safetensors → GGUF conversion.
//!
//! # Family (2 variants share this converter)
//!
//! - **`TIGER-DnR`** — trained on the Divide-and-Remaster (DnR) benchmark
//!   for cinematic dialog / narration / effects source separation. HF:
//!   `JusperLee/TIGER-DnR`. Task tag: `dnr` (denoise + reverb / SFX-strip
//!   flavour). Category tag: `enhancement` (per implementer spec — the
//!   dialog-only path used downstream is treated as an enhancement stage).
//! - **`TIGER-speech`** — same architecture, trained for speech / speaker
//!   separation. HF: `JusperLee/TIGER-speech`. Task tag: `speech`.
//!   Category tag: `enhancement` (siblings the `TIGER-DnR` category so a
//!   single downstream dispatch handles both).
//!
//! Both variants share the same architecture (Time-Frequency dual-path
//! block + selective frequency splitting + Refinement); only the training
//! data + separation head count differ. One `tiger.rs` converter therefore
//! covers both — the caller passes a [`TigerVariant`] and the emitted
//! GGUF's `vokra.model.name` + `vokra.provenance.upstream_hf` reflect the
//! specific release, while `vokra.model.arch = "tiger_separator"` is
//! shared (silently sharing would misroute the downstream loader if the
//! two families ever diverged in tensor topology — today they do not, but
//! the shared arch tag + variant tag is the honest posture).
//!
//! # Provenance
//!
//! - **License (SPDX)**: `apache-2.0` for both variants (per HF model card
//!   `cardData.license`, primary source `huggingface.co/JusperLee/TIGER-DnR`
//!   and `huggingface.co/JusperLee/TIGER-speech`). Class = `Permissive`.
//! - **Attribution**: none required by license (apache-2.0 permits
//!   redistribution without runtime-side attribution obligation).
//! - **Model category** (`vokra.model.category`): `enhancement` (per
//!   implementer spec — a downstream that speaks the speech-enhancement
//!   API can pick this without inspecting the arch).
//! - **Variant tag** (`vokra.tiger.variant`): `"dnr"` or `"speech"` so a
//!   consumer that needs to pick a specific separation head can inspect
//!   this without parsing free-text `vokra.model.name`.
//!
//! # BF16 pass-through (mirror of qwen3_tts / vibevoice / voxcpm2 / wespeaker)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm — no
//! convert-time widening. BF16 stays GGUF type 30 (`GgmlType::BF16`); the
//! runtime widens BF16 → f32 losslessly at load via the single choke
//! point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is
//! the top 16 bits of an f32 — `bits << 16` is exact). The observability
//! counter [`TigerReport::bf16_passthrough`] records how many BF16
//! tensors landed on this arm so a silent widen / downcast cannot slip
//! in undetected.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VibeVoice /
//! VoxCPM contract). Real-weight parity + a native
//! `TigerSeparator::from_gguf` forward path are deferred to a follow-up
//! wave gated on owner sign-off (`docs/license-audit.md` §3.1) — this
//! converter provides the byte-parallel GGUF surface only. The internal
//! Time-Frequency dual-path body is intentionally NOT re-implemented on
//! this pass: transcribing that topology from the paper + upstream repo
//! is a `loud-partial` sibling wave (the RMVPE / Charsiu / pyannote
//! precedent — real converter + real config + `VokraError::UnsupportedOp`
//! forward until an owner-provided real weight anchors the tensor
//! manifest).
//!
//! # No ONNX (permanent)
//!
//! The TIGER family is distributed as safetensors + a PyTorch pipeline;
//! this converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively in a future `crates/vokra-models/src/tiger/`
//! module (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for TIGER-family GGUFs. Shared across both
/// [`TigerVariant`]s — the topology is the same across `dnr` +
/// `speech`; only the training data + separation head count differ.
pub const ARCH: &str = "tiger_separator";

/// `vokra.model.category` for every TIGER GGUF. Per the implementer
/// spec, both variants tag as `enhancement` — a downstream that speaks
/// the speech-enhancement API can pick either without inspecting arch
/// or variant.
pub const CATEGORY: &str = "enhancement";

/// Default upstream weight license (SPDX) — apache-2.0 for both
/// variants per each HF model-card `cardData.license`.
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (mirror of the sibling converters'
// duplication rule).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
/// `vokra.tiger.variant`: `"dnr"` or `"speech"`. Consumers pick a
/// separation head without parsing free-text `vokra.model.name`.
pub const KEY_TIGER_VARIANT: &str = "vokra.tiger.variant";

/// Which TIGER release the caller is converting. Selects the model
/// name / upstream HF slug / variant tag written into the GGUF.
///
/// Both variants share the [`ARCH`] tag `tiger_separator` — the
/// topology is identical, only the trained-in separation head + data
/// differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TigerVariant {
    /// `JusperLee/TIGER-DnR`: dialog / narration / SFX cinematic
    /// source separation. `vokra.tiger.variant = "dnr"`.
    Dnr,
    /// `JusperLee/TIGER-speech`: speaker separation on speech
    /// mixtures. `vokra.tiger.variant = "speech"`.
    Speech,
}

impl TigerVariant {
    /// The `vokra.model.name` string for this release.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Dnr => "tiger-dnr",
            Self::Speech => "tiger-speech",
        }
    }

    /// The `vokra.provenance.upstream_hf` slug (`org/name`) for this
    /// release — the primary redistribution source the model-card
    /// generator anchors on.
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Dnr => "JusperLee/TIGER-DnR",
            Self::Speech => "JusperLee/TIGER-speech",
        }
    }

    /// The `vokra.tiger.variant` tag written under [`KEY_TIGER_VARIANT`].
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Dnr => "dnr",
            Self::Speech => "speech",
        }
    }

    /// One-line free-text description used for the `vokra.provenance.source`
    /// stamp (`stamp_provenance`'s `source` argument).
    pub const fn source_description(self) -> &'static str {
        match self {
            Self::Dnr => {
                "JusperLee/TIGER-DnR (Time-Frequency Interleaved Gain Extraction, DnR benchmark, apache-2.0)"
            }
            Self::Speech => {
                "JusperLee/TIGER-speech (Time-Frequency Interleaved Gain Extraction, speech separation, apache-2.0)"
            }
        }
    }
}

/// Outcome of a TIGER conversion. Mirrors the sibling converters'
/// counter shape (`ecapa_tdnn::EcapaTdnnReport`,
/// `wespeaker::WespeakerReport`, `neucodec::NeucodecReport`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TigerReport {
    /// Total tensors surfaced by the safetensors reader.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16 at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`), so a non-zero
    /// here would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Additive observability counter — a silent
    /// widen / downcast cannot slip in undetected without this counter
    /// also drifting.
    pub bf16_passthrough: usize,
    /// Which TIGER variant was written.
    pub variant: Option<TigerVariant>,
}

/// File-based TIGER converter.
///
/// Reads `input` (upstream `JusperLee/TIGER-DnR` or
/// `JusperLee/TIGER-speech` `model.safetensors`), writes a Vokra GGUF
/// to `output` carrying every F32 / F16 / BF16 tensor verbatim under
/// its upstream safetensors name + the `vokra.model.*`
/// (arch / name / category) + `vokra.provenance.*` metadata chunks +
/// the `vokra.tiger.variant` tag.
///
/// `variant` selects which TIGER release the input came from — the
/// GGUF's `vokra.model.name`, `vokra.provenance.upstream_hf`, and
/// `vokra.tiger.variant` stamps reflect it.
///
/// `license` optionally overrides the default `apache-2.0` provenance
/// stamp (the same override pattern as `wespeaker::convert_wespeaker_file`
/// and `ecapa_tdnn::convert_ecapa_tdnn_file`). `None` keeps the
/// built-in `apache-2.0` stamp.
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure; [`ConvertError::Parse`]
/// on a malformed safetensors input; [`ConvertError::Gguf`] if the GGUF
/// serialization fails.
pub fn convert_tiger_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    variant: TigerVariant,
) -> Result<TigerReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    // Category / upstream-HF / variant stamps — not covered by
    // `stamp_provenance` (which handles the SPDX + class + model_id +
    // source group only), so written directly. Consumers pick a decode
    // path by category and trace the artifact back to its serving
    // location by upstream_hf; the variant tag lets a caller pick a
    // specific separation head without parsing free-text
    // `vokra.model.name`.
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, variant.upstream_hf());
    b.add_string(KEY_TIGER_VARIANT, variant.tag());

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = apache-2.0 (upstream HF model card).
    // `license` overrides for callers who obtained the weight under a
    // different SPDX (see `convert_file_licensed` in `lib.rs`).
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(variant.name()),
        Some(variant.source_description()),
    );

    let mut report = TigerReport {
        variant: Some(variant),
        ..TigerReport::default()
    };
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30), same posture as
    // qwen3_tts / vibevoice / voxcpm2 / wespeaker; runtime widens
    // BF16 → f32 exactly at load via
    // `vokra-core::gguf::quant::decode_bf16` (`bits << 16` is exact).
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )
                .map_err(|e| ConvertError::Gguf(e.to_string()))?;
                report.written += 1;
                if t.dtype == GgmlType::BF16 {
                    report.bf16_passthrough += 1;
                }
            }
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 2;
        assert_eq!(bf16_bytes.len(), expected, "shape × 2 BF16");
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"BF16","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            bf16_bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(bf16_bytes);
        out
    }

    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-tiger-{kind}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, bytes).expect("write temp file");
        p
    }

    /// BF16 round-trips byte-identical for the `Dnr` variant with the
    /// `vokra.model.name` / `vokra.tiger.variant` stamps reflecting
    /// the DnR release. This is the primary end-to-end pin the sibling
    /// converters use (`ecapa_tdnn::tests::bf16_tensor_passes_through_verbatim`,
    /// `neucodec::tests::bf16_tensor_passes_through_verbatim`).
    #[test]
    fn bf16_dnr_round_trips_verbatim() {
        // Non-zero bit patterns so a silent widen / downcast at
        // convert time cannot round-trip trivially.
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12);
        let input_bytes = safetensors_one_bf16("separator.block0.weight", &[2, 3], &bf16);
        let input = write_temp("dnr-in", &input_bytes);
        let output = write_temp("dnr-out", &[]);

        let report = convert_tiger_file(&input, &output, None, TigerVariant::Dnr)
            .expect("convert TIGER-DnR");
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1, "BF16 must reach the pass-through arm");
        assert_eq!(report.bf16_passthrough, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.variant, Some(TigerVariant::Dnr));

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("separator.block0.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16, "BF16 stays BF16 (GGUF type 30)");
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical"
        );

        // Provenance / variant / category / arch stamps landed.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("tiger-dnr")
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );
        assert_eq!(
            file.get(KEY_TIGER_VARIANT).and_then(|v| v.as_str()),
            Some("dnr")
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("JusperLee/TIGER-DnR")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// `Speech` variant re-uses the same converter body but the name /
    /// variant / upstream stamps differ. This guards the variant switch
    /// (a silent same-name emission would misroute a downstream loader
    /// that dispatches on `vokra.model.name`).
    #[test]
    fn speech_variant_emits_distinct_stamps() {
        let values: [f32; 2] = [1.0, -1.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("separator.blockA.weight", &[1, 2], &bf16);
        let input = write_temp("speech-in", &input_bytes);
        let output = write_temp("speech-out", &[]);

        convert_tiger_file(&input, &output, None, TigerVariant::Speech)
            .expect("convert TIGER-speech");

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("tiger-speech"),
            "Speech must emit its own model.name, not fall back to Dnr"
        );
        assert_eq!(
            file.get(KEY_TIGER_VARIANT).and_then(|v| v.as_str()),
            Some("speech")
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("JusperLee/TIGER-speech")
        );
        // Category + arch are shared with Dnr — same downstream dispatch
        // path handles both.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// The optional `license` override lands on the artifact per the
    /// `wespeaker` / `ecapa_tdnn` precedent — the SPDX + weight class +
    /// neutral source restatement all follow.
    #[test]
    fn license_override_reaches_the_artifact() {
        let values: [f32; 2] = [0.5, -0.5];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("separator.weight", &[1, 2], &bf16);
        let input = write_temp("license-in", &input_bytes);
        let output = write_temp("license-out", &[]);

        // Pretend the caller re-obtained this checkpoint under MIT
        // (say, an internal re-release).
        convert_tiger_file(&input, &output, Some("mit"), TigerVariant::Dnr)
            .expect("convert with override");

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
            "override must land on `vokra.provenance.license`"
        );
        // Class is re-derived — MIT resolves to Permissive so the
        // weight_license class does not shift.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// F32 + F16 pass through in the same conversion (mixed-dtype loops
    /// don't collapse to one arm), and BF16 counter stays at Default 0
    /// when no BF16 tensor is present (additive-field regression guard).
    #[test]
    fn f32_and_f16_pass_through() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_words: [u16; 4] = [0x3C00, 0xC000, 0xB800, 0x4200];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let f32_len = f32_bytes.len();
        let total = f32_len + f16_bytes.len();
        let header = format!(
            r#"{{"a.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{f32_len}]}},"b.weight":{{"dtype":"F16","shape":[2,2],"data_offsets":[{f32_len},{total}]}}}}"#
        );
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);
        input_bytes.extend_from_slice(&f16_bytes);
        let input = write_temp("mixed-in", &input_bytes);
        let output = write_temp("mixed-out", &[]);

        let report =
            convert_tiger_file(&input, &output, None, TigerVariant::Speech).expect("mixed convert");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2);
        assert_eq!(report.bf16_passthrough, 0, "no BF16 in the input");
        assert_eq!(report.skipped_non_float, 0);

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
