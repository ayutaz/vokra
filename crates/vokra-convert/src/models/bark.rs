//! **Bark** (`suno/bark` + `suno/bark-small`, MIT): safetensors → GGUF
//! conversion (implementer C wave, 2026-07-30).
//!
//! Input: an upstream Bark release — the upstream ships
//! `pytorch_model.bin` (torch pickle); callers must offline-flatten to
//! safetensors first (mirror of the CSM / DAC / DFN3 prepare-script
//! pattern). Output: a GGUF carrying every float tensor plus the
//! `vokra.provenance.*` / `vokra.model.*` metadata chunks a future
//! native Bark loader will read.
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
//! **Variant axis** — `suno/bark-small` shares the topology but shrinks
//! each stage to `num_layers = 12` (fetched 2026-07-30 primary source);
//! `suno/bark` (full) uses `num_layers = 24` on each stage. Both share
//! `hidden_size = 768`, `num_heads = 12`, `block_size = 1024`, and
//! per-stage vocab sizes.
//!
//! Every hparam below is transcribed verbatim from `huggingface.co/
//! suno/bark-small/raw/main/config.json` (`bark-small` — the variant
//! this converter defaults to per implementer C task spec, fetched
//! 2026-07-30 — CLAUDE.md「ハルシネーション厳禁」). The full-size
//! `suno/bark` variant only differs in `num_layers` (24 vs 12).
//!
//! # Vocoder / codec dependency (external)
//!
//! Bark's terminal step is EnCodec 24 kHz (`facebook/encodec_24khz`);
//! the runtime consumes it via a separate GGUF (Vokra's Mimi / DAC /
//! codec fleet — this converter does NOT ship the EnCodec vocoder).
//! **⚠️  EnCodec weight license = CC-BY-NC 4.0** (research-only) — the
//! M2-13 gate refuses to load an EnCodec GGUF in commercial mode. Bark
//! itself is MIT (per README: "Bark is now licensed under the MIT
//! License, meaning it's now available for commercial use!"), but a
//! commercial-mode caller who wires an EnCodec GGUF still trips the
//! codec-side gate (fail-closed by design — the whole voice-cloning
//! posture in the README lives under `docs/legal-compliance.md` §9).
//!
//! # BF16 pass-through
//!
//! F32 / F16 / BF16 tensors pass through **verbatim** under their
//! upstream safetensors names. BF16 stays GGUF type 30 — no convert-
//! time widening; runtime widens BF16 → f32 losslessly via
//! `decode_bf16`.
//!
//! # Real-weight parity
//!
//! Real-weight parity vs the upstream `suno-ai/bark` / `transformers`
//! `BarkModel` pipeline is deferred to owner (`docs/license-audit.md`
//! §3.1 sign-off — the row lives in the queue with fail-closed default
//! pending owner decision on "research purposes only" model-card
//! advisory).
//!
//! # No ONNX (permanent)
//!
//! Bark ships as a torch pickle (`pytorch_model.bin`) via the
//! `transformers` `BarkModel` release; the converter **never** touches
//! ONNX (FR-LD-05); the pipeline is re-implemented natively in a
//! future `crates/vokra-models/src/bark/` module (whisper.cpp 型 self
//! re-implementation, CLAUDE.md 設計判断 4).

#![allow(dead_code)]

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Bark GGUFs.
pub(crate) const ARCH: &str = "bark";
/// Model category tag — `tts`.
pub(crate) const CATEGORY: &str = "tts";

/// The Bark release variants. Both share the same topology (3-stage AR
/// hierarchical LM); they differ only in per-stage `num_layers` (12 for
/// `-small`, 24 for the full `bark`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarkVariant {
    /// `suno/bark-small` (MIT). `num_layers=12` per stage; ~700 MB
    /// checkpoint (float32). The variant implementer C's task ships;
    /// full `suno/bark` is a follow-up.
    Small,
    /// `suno/bark` (MIT). `num_layers=24` per stage. Added as a
    /// forward-compatible axis; a caller who has offline-flattened the
    /// full checkpoint can pass this variant.
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

// ---- Hparams shared across variants (from `suno/bark-small`
//      config.json, fetched 2026-07-30) --------------------------------

// All 3 stages share these:
pub(crate) const HIDDEN_SIZE: u32 = 768;
pub(crate) const NUM_HEADS: u32 = 12;
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

// External EnCodec 24 kHz vocoder ref.
pub(crate) const CODEC_UPSTREAM_HF: &str = "facebook/encodec_24khz";
pub(crate) const CODEC_SAMPLE_RATE: u32 = 24_000;

// ---- Additive metadata keys ---------------------------------------------

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_VARIANT: &str = "vokra.bark.variant";

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

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_UPSTREAM_HF, variant.upstream_hf());
    b.add_string(KEY_VARIANT, variant.variant_tag());

    // Shared per-stage architecture axes.
    b.add_u32(KEY_HIDDEN_SIZE, HIDDEN_SIZE);
    b.add_u32(KEY_NUM_HEADS, NUM_HEADS);
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

    // External EnCodec vocoder ref (research-only weight — see docstring).
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

    #[test]
    fn small_variant_stamps_12_layers() {
        let (input_bytes, bf16_payload) = synth_bf16();
        let input = scratch_path("small-in");
        let output = scratch_path("small-out");
        std::fs::write(&input, &input_bytes).unwrap();

        let report = convert_bark_file(&input, &output, BarkVariant::Small, None).unwrap();
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

        convert_bark_file(&input, &output, BarkVariant::Full, None).unwrap();
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

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
