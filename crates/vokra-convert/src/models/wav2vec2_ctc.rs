//! **wav2vec 2.0 CTC** family (`facebook/wav2vec2-base-960h`,
//! `facebook/wav2vec2-large-xlsr-53`, `jonatasgrosman/wav2vec2-large-xlsr-53-{japanese,chinese-zh-cn}`,
//! apache-2.0): safetensors → GGUF conversion (SoTA plan Phase 5 ASR
//! fleet, 2026-07-30).
//!
//! Input: an upstream `Wav2Vec2ForCTC` / `Wav2Vec2ForPreTraining` HF
//! safetensors checkpoint (primary source
//! `huggingface.co/{facebook,jonatasgrosman}/wav2vec2-*/raw/main/config.json`).
//! Output: a GGUF carrying every float tensor verbatim under its
//! upstream safetensors name, plus the `vokra.wav2vec2_ctc.*` hparam
//! chunk group and `vokra.provenance.*` / `vokra.model.*` metadata
//! chunks a future native wav2vec2 loader will read.
//!
//! # HF / licence / category
//!
//! All four canonical variants ship under `apache-2.0` per their HF
//! model cards (CC-verified via HF API `cardData.license` on
//! 2026-07-30):
//!
//! - `facebook/wav2vec2-base-960h` — 95M params, base topology
//!   (12 layers × d=768 × 12h × ffn=3072, `feat_extract_norm=group`,
//!   `do_stable_layer_norm=false`), English LibriSpeech 960h,
//!   `Wav2Vec2ForCTC` with `vocab_size=32` char CTC head.
//! - `facebook/wav2vec2-large-xlsr-53` — 300M params, large topology
//!   (24 layers × d=1024 × 16h × ffn=4096, `feat_extract_norm=layer`,
//!   `do_stable_layer_norm=true`), multilingual XLSR-53 pretrained,
//!   `Wav2Vec2ForPreTraining` (no CTC head — reused as encoder base
//!   for the fine-tuned siblings below).
//! - `jonatasgrosman/wav2vec2-large-xlsr-53-japanese` — same large
//!   topology as XLSR-53 base, `Wav2Vec2ForCTC` head with
//!   `vocab_size=2341` (Japanese kana + kanji).
//! - `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` — same
//!   large topology, `Wav2Vec2ForCTC` head with `vocab_size=3503`
//!   (Simplified Chinese).
//!
//! Model category: `asr` (recorded under `vokra.model.category`).
//!
//! # Architecture summary (primary source: HF `config.json`)
//!
//! All wav2vec 2.0 variants share the same feature-extractor
//! topology (7-layer Conv1D at 320× total downsampling):
//! - `conv_dim = [512, 512, 512, 512, 512, 512, 512]`
//! - `conv_kernel = [10, 3, 3, 3, 3, 2, 2]`
//! - `conv_stride = [5, 2, 2, 2, 2, 2, 2]`
//! - `num_feat_extract_layers = 7`
//!
//! The transformer encoder + CTC head axes vary per variant:
//!
//! - **base-960h**: `hidden_size=768`, `num_hidden_layers=12`,
//!   `num_attention_heads=12`, `intermediate_size=3072`,
//!   `vocab_size=32`, `feat_extract_norm="group"`,
//!   `do_stable_layer_norm=false`, `layer_norm_eps=1e-5`,
//!   `hidden_act="gelu"`, `num_conv_pos_embeddings=128`,
//!   `num_conv_pos_embedding_groups=16`.
//! - **large-xlsr-53 base** (Wav2Vec2ForPreTraining, no CTC head):
//!   `hidden_size=1024`, `num_hidden_layers=24`,
//!   `num_attention_heads=16`, `intermediate_size=4096`,
//!   `vocab_size=32` (from pretraining), `feat_extract_norm="layer"`,
//!   `do_stable_layer_norm=true`.
//! - **large-xlsr-53-japanese** (Wav2Vec2ForCTC): same large
//!   topology, `vocab_size=2341`.
//! - **large-xlsr-53-chinese-zh-cn** (Wav2Vec2ForCTC): same large
//!   topology, `vocab_size=3503`.
//!
//! The variant chosen by the caller pins the axes; the actual
//! `vocab_size` written to the GGUF is transcribed from the primary
//! source `config.json` per variant.
//!
//! # BF16 pass-through (mirror of `qwen3_tts` / `vibevoice` / `voxcpm2`)
//!
//! BF16 tensors are emitted verbatim as GGUF type 30
//! (`GgmlType::BF16`) — the same posture as the sibling converters.
//! No convert-time widening; runtime widens BF16 → f32 losslessly via
//! the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). Every F32 / F16
//! tensor passes through under its upstream name.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM /
//! VibeVoice / Neucodec / Wespeaker contract). Real-weight binding
//! is a follow-up wave gated on the upstream tensor-name manifest
//! fetch; this converter passes every F32 / F16 / BF16 tensor through
//! unchanged so a future `Wav2Vec2CtcWeights::from_gguf` can walk the
//! same names.
//!
//! # Real-weight parity
//!
//! Real-weight parity against the upstream HF pipeline is deferred to
//! owner (`docs/license-audit.md` §3.1 sign-off) — this converter
//! provides the byte-parallel GGUF surface only.
//!
//! # No ONNX (permanent)
//!
//! wav2vec 2.0 is distributed as safetensors + a Python pipeline;
//! this converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively in a future
//! `crates/vokra-models/src/wav2vec2_ctc/` module (whisper.cpp 型 self
//! re-implementation, CLAUDE.md 設計判断 4).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for wav2vec 2.0 CTC GGUFs.
pub(crate) const ARCH: &str = "wav2vec2_ctc";

/// Model-category tag written under `vokra.model.category`.
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
pub(crate) const MODEL_CATEGORY: &str = "asr";

/// Upstream-HF slug key (`vokra.provenance.upstream_hf`).
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

// -- `vokra.wav2vec2_ctc.*` transformer-encoder hparam keys -------------
pub(crate) const KEY_HIDDEN_SIZE: &str = "vokra.wav2vec2_ctc.hidden_size";
pub(crate) const KEY_N_LAYER: &str = "vokra.wav2vec2_ctc.n_layer";
pub(crate) const KEY_N_HEAD: &str = "vokra.wav2vec2_ctc.n_head";
pub(crate) const KEY_INTERMEDIATE_SIZE: &str = "vokra.wav2vec2_ctc.intermediate_size";
pub(crate) const KEY_VOCAB_SIZE: &str = "vokra.wav2vec2_ctc.vocab_size";
pub(crate) const KEY_LAYER_NORM_EPS: &str = "vokra.wav2vec2_ctc.layer_norm_eps";
pub(crate) const KEY_FEAT_EXTRACT_NORM: &str = "vokra.wav2vec2_ctc.feat_extract_norm";
pub(crate) const KEY_DO_STABLE_LAYER_NORM: &str = "vokra.wav2vec2_ctc.do_stable_layer_norm";
pub(crate) const KEY_HIDDEN_ACT: &str = "vokra.wav2vec2_ctc.hidden_act";
pub(crate) const KEY_NUM_CONV_POS_EMBEDDINGS: &str = "vokra.wav2vec2_ctc.num_conv_pos_embeddings";
pub(crate) const KEY_NUM_CONV_POS_EMBEDDING_GROUPS: &str =
    "vokra.wav2vec2_ctc.num_conv_pos_embedding_groups";
pub(crate) const KEY_HAS_CTC_HEAD: &str = "vokra.wav2vec2_ctc.has_ctc_head";

// -- `vokra.wav2vec2_ctc.*` feature-extractor conv1d topology (shared) -
pub(crate) const KEY_NUM_FEAT_EXTRACT_LAYERS: &str = "vokra.wav2vec2_ctc.num_feat_extract_layers";
pub(crate) const KEY_CONV_DIM: &str = "vokra.wav2vec2_ctc.conv_dim";
pub(crate) const KEY_CONV_KERNEL: &str = "vokra.wav2vec2_ctc.conv_kernel";
pub(crate) const KEY_CONV_STRIDE: &str = "vokra.wav2vec2_ctc.conv_stride";

/// Which wav2vec 2.0 checkpoint the converter is bound to. Base and
/// large topologies differ; language fine-tunes on top of large-XLSR-53
/// differ only in `vocab_size`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Variant {
    /// `facebook/wav2vec2-base-960h`. Base topology: 12 × d=768 × 12h ×
    /// ffn=3072, `feat_extract_norm="group"`, English LibriSpeech 960h
    /// CTC head with `vocab_size=32` char tokenizer.
    Base960h,
    /// `facebook/wav2vec2-large-xlsr-53`. Large topology (24 × d=1024 ×
    /// 16h × ffn=4096) trained with `Wav2Vec2ForPreTraining` — the
    /// pretrained base for the multilingual fine-tunes below. No CTC
    /// head; `vocab_size=32` is the pretraining sentinel.
    LargeXlsr53Base,
    /// `jonatasgrosman/wav2vec2-large-xlsr-53-japanese`. Large
    /// topology + CTC head with `vocab_size=2341` (Japanese kana +
    /// kanji).
    LargeXlsr53Japanese,
    /// `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`. Large
    /// topology + CTC head with `vocab_size=3503` (Simplified
    /// Chinese).
    LargeXlsr53ChineseZhCn,
}

/// Per-variant axes transcribed verbatim from the primary-source
/// `config.json`.
#[derive(Debug, Clone, Copy)]
struct VariantAxes {
    name: &'static str,
    upstream_hf: &'static str,
    hidden_size: u32,
    n_layer: u32,
    n_head: u32,
    intermediate_size: u32,
    vocab_size: u32,
    layer_norm_eps: f32,
    /// `"group"` on the base topology, `"layer"` on large.
    feat_extract_norm: &'static str,
    /// `false` on base, `true` on large.
    do_stable_layer_norm: bool,
    /// Always `"gelu"` on the released wav2vec 2.0 family — kept per-
    /// variant so a future release that flips it cannot silently
    /// misroute.
    hidden_act: &'static str,
    /// 128 on both base and large.
    num_conv_pos_embeddings: u32,
    /// 16 on both base and large.
    num_conv_pos_embedding_groups: u32,
    /// `true` when the checkpoint carries a `lm_head.weight` CTC head
    /// (`Wav2Vec2ForCTC`), `false` for the plain
    /// `Wav2Vec2ForPreTraining` XLSR base.
    has_ctc_head: bool,
}

/// Feature-extractor conv1d topology — every wav2vec 2.0 checkpoint
/// in the covered family shares the same 7-layer Conv1D chain
/// (`conv_dim=[512×7]`, `conv_kernel=[10,3,3,3,3,2,2]`,
/// `conv_stride=[5,2,2,2,2,2,2]`; primary source: all four `config.json`
/// files fetched 2026-07-30).
const CONV_DIM: [u32; 7] = [512, 512, 512, 512, 512, 512, 512];
const CONV_KERNEL: [u32; 7] = [10, 3, 3, 3, 3, 2, 2];
const CONV_STRIDE: [u32; 7] = [5, 2, 2, 2, 2, 2, 2];

impl Variant {
    fn axes(self) -> VariantAxes {
        match self {
            // Primary source: huggingface.co/facebook/wav2vec2-base-960h/raw/main/config.json
            // Fetched 2026-07-30 (CLAUDE.md「ハルシネーション厳禁」).
            Self::Base960h => VariantAxes {
                name: "wav2vec2-base-960h",
                upstream_hf: "facebook/wav2vec2-base-960h",
                hidden_size: 768,
                n_layer: 12,
                n_head: 12,
                intermediate_size: 3072,
                vocab_size: 32,
                layer_norm_eps: 1e-5,
                feat_extract_norm: "group",
                do_stable_layer_norm: false,
                hidden_act: "gelu",
                num_conv_pos_embeddings: 128,
                num_conv_pos_embedding_groups: 16,
                has_ctc_head: true,
            },
            // Primary source: huggingface.co/facebook/wav2vec2-large-xlsr-53/raw/main/config.json
            // Fetched 2026-07-30 (CLAUDE.md「ハルシネーション厳禁」).
            Self::LargeXlsr53Base => VariantAxes {
                name: "wav2vec2-large-xlsr-53",
                upstream_hf: "facebook/wav2vec2-large-xlsr-53",
                hidden_size: 1024,
                n_layer: 24,
                n_head: 16,
                intermediate_size: 4096,
                vocab_size: 32,
                layer_norm_eps: 1e-5,
                feat_extract_norm: "layer",
                do_stable_layer_norm: true,
                hidden_act: "gelu",
                num_conv_pos_embeddings: 128,
                num_conv_pos_embedding_groups: 16,
                // Wav2Vec2ForPreTraining — no CTC head; the base is used
                // as an encoder foundation for the fine-tuned siblings
                // below.
                has_ctc_head: false,
            },
            // Primary source: huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-japanese/raw/main/config.json
            // Fetched 2026-07-30 (CLAUDE.md「ハルシネーション厳禁」).
            Self::LargeXlsr53Japanese => VariantAxes {
                name: "wav2vec2-large-xlsr-53-japanese",
                upstream_hf: "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
                hidden_size: 1024,
                n_layer: 24,
                n_head: 16,
                intermediate_size: 4096,
                vocab_size: 2341,
                layer_norm_eps: 1e-5,
                feat_extract_norm: "layer",
                do_stable_layer_norm: true,
                hidden_act: "gelu",
                num_conv_pos_embeddings: 128,
                num_conv_pos_embedding_groups: 16,
                has_ctc_head: true,
            },
            // Primary source: huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn/raw/main/config.json
            // Fetched 2026-07-30 (CLAUDE.md「ハルシネーション厳禁」).
            Self::LargeXlsr53ChineseZhCn => VariantAxes {
                name: "wav2vec2-large-xlsr-53-chinese-zh-cn",
                upstream_hf: "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
                hidden_size: 1024,
                n_layer: 24,
                n_head: 16,
                intermediate_size: 4096,
                vocab_size: 3503,
                layer_norm_eps: 1e-5,
                feat_extract_norm: "layer",
                do_stable_layer_norm: true,
                hidden_act: "gelu",
                num_conv_pos_embeddings: 128,
                num_conv_pos_embedding_groups: 16,
                has_ctc_head: true,
            },
        }
    }
}

/// Outcome of a wav2vec 2.0 CTC conversion.
#[derive(Debug, Default)]
pub struct Wav2Vec2CtcReport {
    /// Total tensors observed in the input safetensors header.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`)).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]).
    pub bf16_passthrough: usize,
}

/// Variant-taking file-based wav2vec 2.0 CTC converter — the CLI
/// dispatch arm picks the [`Variant`] from the `--model` string
/// (`wav2vec2-base-960h` / `wav2vec2-large-xlsr-53` /
/// `wav2vec2-large-xlsr-53-japanese` /
/// `wav2vec2-large-xlsr-53-chinese-zh-cn`).
///
/// Reads `input` (an upstream `wav2vec2-*` `model.safetensors`),
/// writes a Vokra GGUF to `output`. `license` overrides the default
/// `apache-2.0` provenance stamp; pass `None` to keep the built-in
/// stamp.
///
/// # Errors
///
/// [`ConvertError::Io`] for I/O failures reading `input` or writing
/// `output`; [`ConvertError::Parse`] for malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_wav2vec2_ctc_file_with_variant(
    input: &Path,
    output: &Path,
    variant: Variant,
    license: Option<&str>,
) -> Result<Wav2Vec2CtcReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;
    let axes = variant.axes();

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, axes.name);
    b.add_string(KEY_MODEL_CATEGORY, MODEL_CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, axes.upstream_hf);

    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => ("apache-2.0".to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(axes.name),
        Some(&format!(
            "{} (wav2vec 2.0 waveform-in encoder + optional CTC head, apache-2.0)",
            axes.upstream_hf
        )),
    );

    write_hparams(&mut b, &axes);

    let mut report = Wav2Vec2CtcReport::default();
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

/// Default file-based converter — base-960h (the smallest / most
/// widely-used release). The CLI's other slugs dispatch to
/// [`convert_wav2vec2_ctc_file_with_variant`] with the matching
/// [`Variant`].
pub fn convert_wav2vec2_ctc_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<Wav2Vec2CtcReport, ConvertError> {
    convert_wav2vec2_ctc_file_with_variant(input, output, Variant::Base960h, license)
}

fn write_hparams(b: &mut GgufBuilder, axes: &VariantAxes) {
    b.add_u32(KEY_HIDDEN_SIZE, axes.hidden_size);
    b.add_u32(KEY_N_LAYER, axes.n_layer);
    b.add_u32(KEY_N_HEAD, axes.n_head);
    b.add_u32(KEY_INTERMEDIATE_SIZE, axes.intermediate_size);
    b.add_u32(KEY_VOCAB_SIZE, axes.vocab_size);
    b.add_f32(KEY_LAYER_NORM_EPS, axes.layer_norm_eps);
    b.add_string(KEY_FEAT_EXTRACT_NORM, axes.feat_extract_norm);
    b.add_bool(KEY_DO_STABLE_LAYER_NORM, axes.do_stable_layer_norm);
    b.add_string(KEY_HIDDEN_ACT, axes.hidden_act);
    b.add_u32(KEY_NUM_CONV_POS_EMBEDDINGS, axes.num_conv_pos_embeddings);
    b.add_u32(
        KEY_NUM_CONV_POS_EMBEDDING_GROUPS,
        axes.num_conv_pos_embedding_groups,
    );
    b.add_bool(KEY_HAS_CTC_HEAD, axes.has_ctc_head);

    // Feature-extractor conv1d topology (shared across all variants).
    b.add_u32(KEY_NUM_FEAT_EXTRACT_LAYERS, CONV_DIM.len() as u32);
    write_u32_array(b, KEY_CONV_DIM, &CONV_DIM);
    write_u32_array(b, KEY_CONV_KERNEL, &CONV_KERNEL);
    write_u32_array(b, KEY_CONV_STRIDE, &CONV_STRIDE);
}

fn write_u32_array(b: &mut GgufBuilder, key: &str, values: &[u32]) {
    b.add_metadata(
        key,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: values.iter().map(|v| GgufMetadataValue::U32(*v)).collect(),
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 2;
        assert_eq!(
            bf16_bytes.len(),
            expected,
            "test fixture: payload len must match shape × 2 BF16"
        );
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

    fn safetensors_f32_then_f16(
        f32_name: &str,
        f32_shape: &[u64],
        f32_bytes: &[u8],
        f16_name: &str,
        f16_shape: &[u64],
        f16_bytes: &[u8],
    ) -> Vec<u8> {
        let f32_elems: u64 = f32_shape.iter().product();
        assert_eq!(f32_bytes.len(), f32_elems as usize * 4);
        let f16_elems: u64 = f16_shape.iter().product();
        assert_eq!(f16_bytes.len(), f16_elems as usize * 2);
        let f32_shape_str = f32_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let f16_shape_str = f16_shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let f32_len = f32_bytes.len();
        let total = f32_len + f16_bytes.len();
        let header = format!(
            r#"{{"{f32_name}":{{"dtype":"F32","shape":[{f32_shape_str}],"data_offsets":[0,{f32_len}]}},"{f16_name}":{{"dtype":"F16","shape":[{f16_shape_str}],"data_offsets":[{f32_len},{total}]}}}}"#
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(f32_bytes);
        out.extend_from_slice(f16_bytes);
        out
    }

    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-wav2vec2-ctc-{kind}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, bytes).expect("write temp file");
        p
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12);

        let input_bytes = safetensors_one_bf16(
            "wav2vec2.encoder.layers.0.attention.q_proj.weight",
            &[2, 3],
            &bf16,
        );
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_wav2vec2_ctc_file(&input_path, &output_path, None)
            .expect("convert_wav2vec2_ctc_file must accept a well-formed BF16 checkpoint");
        assert_eq!(report.read, 1);
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror qwen3_tts / vibevoice / voxcpm2)"
        );
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("wav2vec2.encoder.layers.0.attention.q_proj.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn f32_and_f16_tensors_pass_through_and_stamps_land() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_words: [u16; 6] = [0x3C00, 0xC000, 0xB800, 0x4200, 0x3100, 0x5140];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();

        let input_bytes = safetensors_f32_then_f16(
            "wav2vec2.feature_projection.projection.weight",
            &[1, 2],
            &f32_bytes,
            "wav2vec2.encoder.layers.0.attention.q_proj.weight",
            &[2, 3],
            &f16_bytes,
        );
        let input_path = write_temp("mixed-in", &input_bytes);
        let output_path = write_temp("mixed-out", &[]);

        let report = convert_wav2vec2_ctc_file_with_variant(
            &input_path,
            &output_path,
            Variant::Base960h,
            None,
        )
        .expect("mixed F32/F16 must convert");

        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2);
        assert_eq!(report.bf16_passthrough, 0);
        assert_eq!(report.skipped_non_float, 0);

        let out_bytes = std::fs::read(&output_path).expect("read");
        let file = GgufFile::parse(out_bytes).expect("parse");

        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("wav2vec2-base-960h")
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
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("facebook/wav2vec2-base-960h")
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(MODEL_CATEGORY)
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn hparam_chunk_pins_base_960h_axes() {
        let bytes = safetensors_one_bf16("dummy.weight", &[1, 2], &[0u8; 4]);
        let input_path = write_temp("base-hparam-in", &bytes);
        let output_path = write_temp("base-hparam-out", &[]);

        let _ = convert_wav2vec2_ctc_file_with_variant(
            &input_path,
            &output_path,
            Variant::Base960h,
            None,
        )
        .expect("base 960h conversion must succeed");

        let out = std::fs::read(&output_path).expect("read");
        let file = GgufFile::parse(out).expect("parse");

        // Every axis transcribed from
        // huggingface.co/facebook/wav2vec2-base-960h/raw/main/config.json.
        assert_eq!(
            file.get(KEY_HIDDEN_SIZE).and_then(|v| v.as_u64()),
            Some(768)
        );
        assert_eq!(file.get(KEY_N_LAYER).and_then(|v| v.as_u64()), Some(12));
        assert_eq!(file.get(KEY_N_HEAD).and_then(|v| v.as_u64()), Some(12));
        assert_eq!(
            file.get(KEY_INTERMEDIATE_SIZE).and_then(|v| v.as_u64()),
            Some(3072)
        );
        assert_eq!(file.get(KEY_VOCAB_SIZE).and_then(|v| v.as_u64()), Some(32));
        assert_eq!(
            file.get(KEY_FEAT_EXTRACT_NORM).and_then(|v| v.as_str()),
            Some("group")
        );
        assert_eq!(
            file.get(KEY_DO_STABLE_LAYER_NORM).and_then(|v| v.as_bool()),
            Some(false)
        );
        assert_eq!(
            file.get(KEY_HIDDEN_ACT).and_then(|v| v.as_str()),
            Some("gelu")
        );
        assert_eq!(
            file.get(KEY_NUM_CONV_POS_EMBEDDINGS)
                .and_then(|v| v.as_u64()),
            Some(128)
        );
        assert_eq!(
            file.get(KEY_NUM_CONV_POS_EMBEDDING_GROUPS)
                .and_then(|v| v.as_u64()),
            Some(16)
        );
        assert_eq!(
            file.get(KEY_HAS_CTC_HEAD).and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            file.get(KEY_NUM_FEAT_EXTRACT_LAYERS)
                .and_then(|v| v.as_u64()),
            Some(7)
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    #[test]
    fn hparam_chunk_pins_large_xlsr_53_variants() {
        let bytes = safetensors_one_bf16("dummy.weight", &[1, 2], &[0u8; 4]);

        // Base (Wav2Vec2ForPreTraining, no CTC head).
        let in_base = write_temp("xlsr53-base-in", &bytes);
        let out_base = write_temp("xlsr53-base-out", &[]);
        convert_wav2vec2_ctc_file_with_variant(&in_base, &out_base, Variant::LargeXlsr53Base, None)
            .expect("large-xlsr-53 base conversion must succeed");
        let file = GgufFile::parse(std::fs::read(&out_base).unwrap()).unwrap();
        assert_eq!(
            file.get(KEY_HIDDEN_SIZE).and_then(|v| v.as_u64()),
            Some(1024)
        );
        assert_eq!(file.get(KEY_N_LAYER).and_then(|v| v.as_u64()), Some(24));
        assert_eq!(file.get(KEY_N_HEAD).and_then(|v| v.as_u64()), Some(16));
        assert_eq!(
            file.get(KEY_INTERMEDIATE_SIZE).and_then(|v| v.as_u64()),
            Some(4096)
        );
        assert_eq!(file.get(KEY_VOCAB_SIZE).and_then(|v| v.as_u64()), Some(32));
        assert_eq!(
            file.get(KEY_FEAT_EXTRACT_NORM).and_then(|v| v.as_str()),
            Some("layer"),
            "large topology uses feat_extract_norm=layer (base uses group)"
        );
        assert_eq!(
            file.get(KEY_DO_STABLE_LAYER_NORM).and_then(|v| v.as_bool()),
            Some(true),
            "large topology sets do_stable_layer_norm=true (base=false)"
        );
        assert_eq!(
            file.get(KEY_HAS_CTC_HEAD).and_then(|v| v.as_bool()),
            Some(false),
            "Wav2Vec2ForPreTraining has no CTC head"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("facebook/wav2vec2-large-xlsr-53")
        );

        // Japanese (Wav2Vec2ForCTC, vocab=2341).
        let in_ja = write_temp("xlsr53-ja-in", &bytes);
        let out_ja = write_temp("xlsr53-ja-out", &[]);
        convert_wav2vec2_ctc_file_with_variant(&in_ja, &out_ja, Variant::LargeXlsr53Japanese, None)
            .expect("large-xlsr-53-japanese conversion must succeed");
        let file = GgufFile::parse(std::fs::read(&out_ja).unwrap()).unwrap();
        assert_eq!(
            file.get(KEY_VOCAB_SIZE).and_then(|v| v.as_u64()),
            Some(2341),
            "Japanese fine-tune has vocab_size=2341"
        );
        assert_eq!(
            file.get(KEY_HAS_CTC_HEAD).and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("jonatasgrosman/wav2vec2-large-xlsr-53-japanese")
        );

        // Chinese (Wav2Vec2ForCTC, vocab=3503).
        let in_zh = write_temp("xlsr53-zh-in", &bytes);
        let out_zh = write_temp("xlsr53-zh-out", &[]);
        convert_wav2vec2_ctc_file_with_variant(
            &in_zh,
            &out_zh,
            Variant::LargeXlsr53ChineseZhCn,
            None,
        )
        .expect("large-xlsr-53-chinese-zh-cn conversion must succeed");
        let file = GgufFile::parse(std::fs::read(&out_zh).unwrap()).unwrap();
        assert_eq!(
            file.get(KEY_VOCAB_SIZE).and_then(|v| v.as_u64()),
            Some(3503),
            "Chinese fine-tune has vocab_size=3503"
        );
        assert_eq!(
            file.get(KEY_HAS_CTC_HEAD).and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
        );

        for p in [&in_base, &out_base, &in_ja, &out_ja, &in_zh, &out_zh] {
            std::fs::remove_file(p).ok();
        }
    }

    #[test]
    fn license_override_replaces_stamp() {
        let bytes = safetensors_one_bf16("dummy.weight", &[1, 2], &[0u8; 4]);
        let input_path = write_temp("lic-in", &bytes);
        let output_path = write_temp("lic-out", &[]);

        let _ = convert_wav2vec2_ctc_file(&input_path, &output_path, Some("MIT"))
            .expect("license override must succeed");

        let out = std::fs::read(&output_path).expect("read");
        let file = GgufFile::parse(out).expect("parse");

        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("MIT")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }
}
