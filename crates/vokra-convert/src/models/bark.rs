//! **Bark** (`suno/bark` + `suno/bark-small`, MIT): safetensors → GGUF
//! conversion (implementer C wave, 2026-07-30).
//!
//! Input: an upstream Bark release — the upstream ships
//! `pytorch_model.bin` (torch pickle); callers must offline-flatten to
//! safetensors first (mirror of the CSM / DAC / DFN3 prepare-script
//! pattern). Output: a GGUF carrying every float tensor plus the
//! `vokra.provenance.*` / `vokra.model.*` metadata chunks the native
//! Bark loader reads.
//!
//! # Architecture (primary source, 2026-07-30 CC fetch)
//!
//! Bark is a **hierarchical AR LM over discrete audio codes** (three
//! transformer stages + EnCodec vocoder), the `suno-ai/bark` /
//! `huggingface/transformers` design:
//!
//! 1. **Semantic** stage — text → semantic tokens
//!    - `input_vocab_size = 129600` (BERT tokenizer over Bark voice
//!      prompts + raw UTF-8)
//!    - `output_vocab_size = 10048` (semantic code alphabet)
//! 2. **Coarse acoustics** stage — semantic tokens → 2 coarse EnCodec
//!    codebooks
//!    - `input_vocab_size = output_vocab_size = 12096`
//! 3. **Fine acoustics** stage — coarse codes → 6 fine EnCodec codebooks
//!    - `input_vocab_size = output_vocab_size = 1056`
//!    - `n_codes_total = 8`, `n_codes_given = 1`
//!
//! **Variant axis** — `suno/bark-small` shrinks each stage to
//! `hidden_size = 768`, `num_heads = 12`, `num_layers = 12`;
//! `suno/bark` (full) uses `hidden_size = 1024`, `num_heads = 16`, and
//! `num_layers = 24`. Both use `block_size = 1024` and the same
//! per-stage vocab axes. These values are pinned to the immutable upstream
//! revisions documented by [`BarkVariant::upstream_revision`] and are also
//! independently visible in the released embedding/attention tensor shapes.
//!
//! Every hparam below is transcribed from the immutable upstream
//! `config.json` for the selected variant (not from mutable `main`).
//!
//! # Embedded vocoder / codec
//!
//! The released `suno/bark` and `suno/bark-small` checkpoints contain the
//! complete EnCodec 24 kHz module under `codec_model.*`; it is not a separate
//! runtime dependency. The converter preserves those tensors together with
//! the three language-model stages. `vokra.bark.codec.upstream_hf` records the
//! architecture origin (`facebook/encodec_24khz`), while provenance/license
//! metadata applies to the combined checkpoint supplied to this converter.
//! The official Suno repositories are tagged MIT and are already recorded in
//! `docs/license-audit.md`; a caller converting a differently licensed mirror
//! must pass its actual license explicitly.
//!
//! # Canonical dtype
//!
//! The two audited Suno releases are entirely F32. Canonical conversion
//! requires that exact dtype and the complete sorted name/shape manifest.
//! F16, BF16, partial, and same-count unrelated checkpoints fail explicitly;
//! the converter never writes an artifact the mmap-only Bark binder would
//! later have to widen or silently reinterpret.
//!
//! # Real-weight parity
//!
//! Real-weight parity vs the upstream `suno-ai/bark` / `transformers`
//! `BarkModel` pipeline remains a VAST + remote Apple-Silicon gate. No
//! numerical result is claimed until those runs complete.
//!
//! # No ONNX (permanent)
//!
//! Bark ships as a torch pickle (`pytorch_model.bin`) via the
//! `transformers` `BarkModel` release; the converter **never** touches
//! ONNX (FR-LD-05); the pipeline is re-implemented natively in
//! `crates/vokra-models/src/bark/` (whisper.cpp 型 self re-implementation).

use std::collections::BTreeMap;
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Bark GGUFs.
pub(crate) const ARCH: &str = "bark";
/// Model category tag — `tts`.
pub(crate) const CATEGORY: &str = "tts";

/// The Bark release variants. Both share the same three-stage hierarchical LM
/// topology, but width, head count, and layer count differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarkVariant {
    /// `suno/bark-small` (MIT). 768 hidden, 12 heads, 12 layers per stage.
    Small,
    /// `suno/bark` (MIT). 1024 hidden, 16 heads, 24 layers per stage.
    Full,
}

impl BarkVariant {
    /// Canonical `vokra.model.name` for this variant.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Small => "bark-small",
            Self::Full => "bark",
        }
    }

    /// Short `vokra.bark.variant` tag written on the GGUF.
    pub const fn variant_tag(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Full => "full",
        }
    }

    /// Upstream HF repo slug (recorded under
    /// `vokra.provenance.upstream_hf`).
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Small => "suno/bark-small",
            Self::Full => "suno/bark",
        }
    }

    /// Immutable upstream Hugging Face revision whose configuration and
    /// checkpoint contract this converter implements.
    pub const fn upstream_revision(self) -> &'static str {
        match self {
            Self::Small => "1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd",
            Self::Full => "70a8a7d34168586dc5d028fa9666aceade177992",
        }
    }

    /// Transformer hidden width for every language-model stage.
    pub const fn hidden_size(self) -> u32 {
        match self {
            Self::Small => 768,
            Self::Full => 1_024,
        }
    }

    /// Self-attention head count for every language-model stage.
    pub const fn num_heads(self) -> u32 {
        match self {
            Self::Small => 12,
            Self::Full => 16,
        }
    }

    /// Exact tensor count in the combined three-stage + codec checkpoint.
    pub const fn tensor_count(self) -> usize {
        match self {
            Self::Small => 518,
            Self::Full => 758,
        }
    }

    /// SHA-256 of the complete sorted `(tensor name, dimensions)` manifest.
    pub const fn tensor_manifest_sha256(self) -> &'static str {
        match self {
            Self::Small => "25adef111ab1318346c4f54003bdfa7dc3305bc1b20fdcbd3a9cdfbe1e4ff127",
            Self::Full => "c32d8b203779ea68235c0304152781315a8a18694938c4872bfe476ea0da6424",
        }
    }

    /// Per-stage `num_layers`. Small = 12 (primary source config), full
    /// = 24 (upstream `BarkModel` `transformers` default + `suno-ai/bark`
    /// README).
    pub const fn num_layers(self) -> u32 {
        match self {
            Self::Small => 12,
            Self::Full => 24,
        }
    }
}

// ---- Hparams shared across variants -----------------------------------

// All 3 stages in both variants share this context length.
pub(crate) const BLOCK_SIZE: u32 = 1_024;

// Per-stage vocab axes (identical across small + full):
pub(crate) const SEMANTIC_INPUT_VOCAB_SIZE: u32 = 129_600;
pub(crate) const SEMANTIC_OUTPUT_VOCAB_SIZE: u32 = 10_048;
pub(crate) const COARSE_INPUT_VOCAB_SIZE: u32 = 12_096;
pub(crate) const COARSE_OUTPUT_VOCAB_SIZE: u32 = 12_096;
pub(crate) const FINE_INPUT_VOCAB_SIZE: u32 = 1_056;
pub(crate) const FINE_OUTPUT_VOCAB_SIZE: u32 = 1_056;
pub(crate) const FINE_N_CODES_TOTAL: u32 = 8;
pub(crate) const FINE_N_CODES_GIVEN: u32 = 1;

// Embedded EnCodec 24 kHz architecture/provenance ref.
pub(crate) const CODEC_UPSTREAM_HF: &str = "facebook/encodec_24khz";
pub(crate) const CODEC_SAMPLE_RATE: u32 = 24_000;

// ---- Additive metadata keys ---------------------------------------------

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_VARIANT: &str = "vokra.bark.variant";
const KEY_TENSOR_MANIFEST_SHA256: &str = "vokra.bark.tensor_manifest_sha256";

const KEY_HIDDEN_SIZE: &str = "vokra.bark.hidden_size";
const KEY_NUM_HEADS: &str = "vokra.bark.num_heads";
const KEY_BLOCK_SIZE: &str = "vokra.bark.block_size";
const KEY_NUM_LAYERS_PER_STAGE: &str = "vokra.bark.num_layers_per_stage";

const KEY_SEMANTIC_INPUT_VOCAB_SIZE: &str = "vokra.bark.semantic.input_vocab_size";
const KEY_SEMANTIC_OUTPUT_VOCAB_SIZE: &str = "vokra.bark.semantic.output_vocab_size";
const KEY_COARSE_INPUT_VOCAB_SIZE: &str = "vokra.bark.coarse.input_vocab_size";
const KEY_COARSE_OUTPUT_VOCAB_SIZE: &str = "vokra.bark.coarse.output_vocab_size";
const KEY_FINE_INPUT_VOCAB_SIZE: &str = "vokra.bark.fine.input_vocab_size";
const KEY_FINE_OUTPUT_VOCAB_SIZE: &str = "vokra.bark.fine.output_vocab_size";
const KEY_FINE_N_CODES_TOTAL: &str = "vokra.bark.fine.n_codes_total";
const KEY_FINE_N_CODES_GIVEN: &str = "vokra.bark.fine.n_codes_given";

const KEY_CODEC_UPSTREAM_HF: &str = "vokra.bark.codec.upstream_hf";
const KEY_CODEC_SAMPLE_RATE: &str = "vokra.bark.codec.sample_rate";

/// Outcome of a Bark conversion. Mirrors the shared BF16 pass-through
/// report shape.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct BarkReport {
    /// Total tensors observed in the safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter).
    pub skipped_non_float: usize,
    /// BF16 tensors on the pass-through arm.
    pub bf16_passthrough: usize,
}

/// File-based Bark converter (`vokra-cli convert --model bark-small` /
/// `--model bark`).
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O; [`ConvertError::Parse`] for malformed
/// safetensors; [`ConvertError::Gguf`] for GGUF writer failure.
pub fn convert_bark_file(
    input: &Path,
    output: &Path,
    variant: BarkVariant,
    license: Option<&str>,
) -> Result<BarkReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;
    validate_canonical_checkpoint(&st, variant)?;
    write_bark_gguf(&st, output, variant, license)
}

fn write_bark_gguf(
    st: &SafetensorsFile,
    output: &Path,
    variant: BarkVariant,
    license: Option<&str>,
) -> Result<BarkReport, ConvertError> {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_UPSTREAM_HF, variant.upstream_hf());
    b.add_string(KEY_UPSTREAM_REVISION, variant.upstream_revision());
    b.add_string(KEY_VARIANT, variant.variant_tag());
    b.add_string(KEY_TENSOR_MANIFEST_SHA256, variant.tensor_manifest_sha256());

    // Variant-specific per-stage architecture axes. Full Bark is wider than
    // Bark Small; using the Small values here produced an invalid Full
    // metadata contract before this was made variant-aware.
    b.add_u32(KEY_HIDDEN_SIZE, variant.hidden_size());
    b.add_u32(KEY_NUM_HEADS, variant.num_heads());
    b.add_u32(KEY_BLOCK_SIZE, BLOCK_SIZE);
    b.add_u32(KEY_NUM_LAYERS_PER_STAGE, variant.num_layers());

    // Per-stage vocab axes.
    b.add_u32(KEY_SEMANTIC_INPUT_VOCAB_SIZE, SEMANTIC_INPUT_VOCAB_SIZE);
    b.add_u32(KEY_SEMANTIC_OUTPUT_VOCAB_SIZE, SEMANTIC_OUTPUT_VOCAB_SIZE);
    b.add_u32(KEY_COARSE_INPUT_VOCAB_SIZE, COARSE_INPUT_VOCAB_SIZE);
    b.add_u32(KEY_COARSE_OUTPUT_VOCAB_SIZE, COARSE_OUTPUT_VOCAB_SIZE);
    b.add_u32(KEY_FINE_INPUT_VOCAB_SIZE, FINE_INPUT_VOCAB_SIZE);
    b.add_u32(KEY_FINE_OUTPUT_VOCAB_SIZE, FINE_OUTPUT_VOCAB_SIZE);
    b.add_u32(KEY_FINE_N_CODES_TOTAL, FINE_N_CODES_TOTAL);
    b.add_u32(KEY_FINE_N_CODES_GIVEN, FINE_N_CODES_GIVEN);

    // Embedded EnCodec vocoder architecture/provenance ref.
    b.add_string(KEY_CODEC_UPSTREAM_HF, CODEC_UPSTREAM_HF);
    b.add_u32(KEY_CODEC_SAMPLE_RATE, CODEC_SAMPLE_RATE);

    // Default license = mit (Bark README: "Bark is now licensed under
    // the MIT License…", 2023-05-01 change from CC-BY-NC 4.0 — see the
    // `docs/license-audit.md` §CC-verified entry). `license` overrides
    // for callers who obtained the weight under a different SPDX.
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => ("mit".to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(variant.name()),
        Some(variant.upstream_hf()),
    );

    let mut report = BarkReport::default();
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

fn validate_canonical_checkpoint(
    file: &SafetensorsFile,
    variant: BarkVariant,
) -> Result<(), ConvertError> {
    if file.tensors().len() != variant.tensor_count() {
        return Err(ConvertError::Parse(format!(
            "Bark {} checkpoint has {} tensors; expected exactly {} from {}@{}",
            variant.variant_tag(),
            file.tensors().len(),
            variant.tensor_count(),
            variant.upstream_hf(),
            variant.upstream_revision(),
        )));
    }

    let mut manifest = BTreeMap::new();
    for tensor in file.tensors() {
        if tensor.dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "Bark {} tensor `{}` is {:?}; the canonical {}@{} checkpoint is F32 and the zero-copy runtime does not silently widen other dtypes",
                variant.variant_tag(),
                tensor.name,
                tensor.dtype,
                variant.upstream_hf(),
                variant.upstream_revision(),
            )));
        }
        if manifest
            .insert(tensor.name.clone(), tensor.shape.clone())
            .is_some()
        {
            return Err(ConvertError::Parse(format!(
                "Bark {} checkpoint repeats tensor `{}`",
                variant.variant_tag(),
                tensor.name
            )));
        }
    }
    let actual = crate::models::canary_1b_flash::hex(
        &crate::models::canary_1b_flash::manifest_sha256(&manifest),
    );
    if actual != variant.tensor_manifest_sha256() {
        return Err(ConvertError::Parse(format!(
            "Bark {} complete tensor name/shape manifest SHA-256 is {actual}; expected {} for {}@{}",
            variant.variant_tag(),
            variant.tensor_manifest_sha256(),
            variant.upstream_hf(),
            variant.upstream_revision(),
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-bark-{}-{}-{}.bin",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        p
    }

    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        assert_eq!(bf16_bytes.len(), elems as usize * 2);
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

    fn synth_bf16() -> (Vec<u8>, Vec<u8>) {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        (
            safetensors_one_bf16(
                "semantic.transformer.h.0.attn.c_attn.weight",
                &[2, 3],
                &bf16,
            ),
            bf16,
        )
    }

    fn write_fixture(
        input: &Path,
        output: &Path,
        variant: BarkVariant,
    ) -> Result<BarkReport, ConvertError> {
        let file = SafetensorsFile::parse(std::fs::read(input)?)?;
        write_bark_gguf(&file, output, variant, None)
    }

    #[test]
    fn small_variant_stamps_12_layers() {
        let (input_bytes, bf16_payload) = synth_bf16();
        let input = scratch_path("small-in");
        let output = scratch_path("small-out");
        std::fs::write(&input, &input_bytes).unwrap();

        let report = write_fixture(&input, &output, BarkVariant::Small).unwrap();
        assert_eq!(report.read, 1);
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);

        let file = GgufFile::parse(std::fs::read(&output).unwrap()).unwrap();
        let info = file
            .tensor_info("semantic.transformer.h.0.attn.c_attn.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), bf16_payload.as_slice());

        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(BarkVariant::Small.name())
        );
        assert_eq!(
            file.get(KEY_VARIANT).and_then(|v| v.as_str()),
            Some("small")
        );
        assert_eq!(
            file.get(KEY_UPSTREAM_REVISION).and_then(|v| v.as_str()),
            Some(BarkVariant::Small.upstream_revision())
        );
        assert_eq!(
            file.get(KEY_TENSOR_MANIFEST_SHA256)
                .and_then(|v| v.as_str()),
            Some(BarkVariant::Small.tensor_manifest_sha256())
        );
        assert_eq!(
            file.get(KEY_NUM_LAYERS_PER_STAGE).and_then(|v| v.as_u64()),
            Some(12),
            "`suno/bark-small` config.json pins num_layers = 12 per stage"
        );

        // Shared architecture axes.
        for (k, expect) in [
            (KEY_HIDDEN_SIZE, 768u64),
            (KEY_NUM_HEADS, 12),
            (KEY_BLOCK_SIZE, 1_024),
            (KEY_SEMANTIC_INPUT_VOCAB_SIZE, 129_600),
            (KEY_SEMANTIC_OUTPUT_VOCAB_SIZE, 10_048),
            (KEY_COARSE_INPUT_VOCAB_SIZE, 12_096),
            (KEY_COARSE_OUTPUT_VOCAB_SIZE, 12_096),
            (KEY_FINE_INPUT_VOCAB_SIZE, 1_056),
            (KEY_FINE_OUTPUT_VOCAB_SIZE, 1_056),
            (KEY_FINE_N_CODES_TOTAL, 8),
            (KEY_FINE_N_CODES_GIVEN, 1),
            (KEY_CODEC_SAMPLE_RATE, 24_000),
        ] {
            assert_eq!(
                file.get(k).and_then(|v| v.as_u64()),
                Some(expect),
                "{k} pin"
            );
        }

        // Codec upstream ref recorded.
        assert_eq!(
            file.get(KEY_CODEC_UPSTREAM_HF).and_then(|v| v.as_str()),
            Some(CODEC_UPSTREAM_HF)
        );

        // Default license = mit (README + HF card).
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn full_variant_stamps_24_layers_and_distinct_provenance() {
        let (input_bytes, _) = synth_bf16();
        let input = scratch_path("full-in");
        let output = scratch_path("full-out");
        std::fs::write(&input, &input_bytes).unwrap();

        write_fixture(&input, &output, BarkVariant::Full).unwrap();
        let file = GgufFile::parse(std::fs::read(&output).unwrap()).unwrap();

        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(BarkVariant::Full.name())
        );
        assert_eq!(
            file.get(KEY_UPSTREAM_HF).and_then(|v| v.as_str()),
            Some(BarkVariant::Full.upstream_hf())
        );
        assert_eq!(file.get(KEY_VARIANT).and_then(|v| v.as_str()), Some("full"));
        assert_eq!(
            file.get(KEY_NUM_LAYERS_PER_STAGE).and_then(|v| v.as_u64()),
            Some(24),
            "full `suno/bark` uses num_layers = 24 per stage"
        );
        assert_eq!(
            file.get(KEY_HIDDEN_SIZE).and_then(|v| v.as_u64()),
            Some(1_024),
            "full `suno/bark` uses hidden_size = 1024"
        );
        assert_eq!(
            file.get(KEY_NUM_HEADS).and_then(|v| v.as_u64()),
            Some(16),
            "full `suno/bark` uses num_heads = 16"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn canonical_conversion_rejects_a_partial_checkpoint() {
        let (input_bytes, _) = synth_bf16();
        let input = scratch_path("partial-in");
        let output = scratch_path("partial-out");
        std::fs::write(&input, &input_bytes).unwrap();

        let error = convert_bark_file(&input, &output, BarkVariant::Small, None).unwrap_err();
        assert!(
            matches!(error, ConvertError::Parse(ref message) if message.contains("expected exactly 518")),
            "got {error:?}"
        );
        assert!(!output.exists());

        std::fs::remove_file(&input).ok();
    }
}
