//! **WavLM Base+ SV** (`microsoft/wavlm-base-plus-sv`, CC-BY-SA-3.0):
//! safetensors → GGUF conversion (Wave 7, speaker-fleet extension,
//! 2026-08-14).
//!
//! WavLM = **Large-Scale Self-Supervised Pre-Training for Full Stack
//! Speech Processing** — a self-supervised speech encoder (HuBERT
//! lineage) augmented with **gated relative position bias +
//! convolutional position-bias fusion** that neither wav2vec2 nor
//! HuBERT expose. The `-sv` release is the Base+ WavLM fine-tuned on
//! **VoxCeleb1** with an **XVector head + Additive Margin Softmax**
//! for speaker verification (512-d embedding, EER ~0.84% on
//! VoxCeleb1). Paper: arXiv:2110.13900 (Chen et al. 2022 "WavLM:
//! Large-Scale Self-Supervised Pre-Training for Full Stack Speech
//! Processing"). Upstream code: `github.com/microsoft/UniSpeech`.
//!
//! # Vokra scope — speaker fleet extension over redimnet / titanet / ecapa_tdnn / …
//!
//! Complements the sibling speaker converters:
//! - `campplus` (`iic/speech_campplus`) — CAM++ 192-d, complete forward.
//! - `wespeaker` (`Wespeaker/wespeaker-voxceleb-resnet34-LM`) —
//!   ResNet-34 backbone loud-partial.
//! - `ecapa_tdnn` (`speechbrain/spkrec-ecapa-voxceleb`) — ECAPA-TDNN
//!   loud-partial.
//! - `titanet` (`nvidia/speakerverification_en_titanet_large`) —
//!   depth-wise separable Conv1D loud-partial.
//! - `speaker_3d` (`iic/speech_eres2net_sv_zh-cn_16k-common`) —
//!   ERes2Net loud-partial.
//! - `redimnet` (`Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM`) — 2D
//!   dim-reduction + 1D conv+att + ASTP loud-partial (Wave 4 land).
//!
//! WavLM-SV brings a **Transformer + gated relative position bias +
//! XVector** hybrid backbone that no sibling covers — the whole
//! speaker fleet ships as loud-partial (from_gguf real, encode =
//! UnsupportedOp) today, and this converter is one more strand for
//! the future WavLM Python source transcription wave. Distinct arch
//! tag `wavlm_sv` (never `campplus`, `wespeaker`, `ecapa_tdnn`,
//! `titanet`, `speaker_3d`, or `redimnet` — silently sharing an
//! arch would misroute runtime dispatch, FR-EX-08). Category
//! `speaker` (mirror of the sibling speaker fleet — the converter
//! fleet groups speaker-embedding / verification networks under one
//! category so downstream consumers pick a load path without
//! inspecting the arch).
//!
//! # License posture — CC-BY-SA-3.0 (Copyleft, HF card LICENSE link
//! primary source)
//!
//! HF `microsoft/wavlm-base-plus-sv` cardData links its license to
//! `github.com/microsoft/UniSpeech/blob/main/LICENSE` = **CC-BY-SA-3.0**
//! (Attribution-ShareAlike 3.0 Unported, SPDX = `cc-by-sa-3.0`).
//! **Copyleft** per `LicenseClass::from_license_str` (`has_sa` arm
//! fires before the plain `cc-by` arm — the ordering-pin invariant
//! recorded in `docs/license-audit.md` dictates this). `-sv` is NOT
//! MIT and NOT MSR-LA — the scout task's tentative "MSR-LA — check"
//! hypothesis was resolved to CC-BY-SA-3.0 via the LICENSE URL walk.
//!
//! `scripts/publish/fetch_license.sh` already supports `cc-by-sa-3.0`
//! (lines 47 + 139), so there is no publisher-side gap. Downstream
//! redistribution must preserve the SA license — publishing as
//! Apache-2.0 would be misrepresentation.
//!
//! §3.1 sign-off column is **BLANK** in `docs/license-audit.md` per
//! the fail-closed default (memory `[[feedback-license-signoff-primary-source]]`).
//! **Copyleft** propagates: an owner MUST sign a copyleft weight row
//! because the share-alike obligation propagates to downstream
//! consumers and is a legal decision (not a CC judgement).
//!
//! # Scale — local convert OK (~377 MB `pytorch_model.bin`)
//!
//! Well below the M1 iMac 16 GB local-convert threshold (memory
//! `[[feedback-large-models-on-vast-ai]]`: <2 GB safe). No vast.ai
//! handoff needed. The upstream release ships `pytorch_model.bin`
//! (torch pickle), bridged offline to safetensors through the sibling
//! `tools/parity/nemo_pt_to_safetensors.py` flow (uv-managed Python
//! 3.12 sidecar per memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`); this converter accepts safetensors
//! only (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).
//!
//! # No ONNX / no pickle (permanent)
//!
//! WavLM ships as PyTorch pickle upstream (`pytorch_model.bin`); this
//! converter **never** touches ONNX or pickle (FR-LD-05 /
//! NFR-DS-02). Runtime tree carries neither `torch` nor `onnxruntime`.
//!
//! # BF16 pass-through
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm. BF16 is
//! emitted as GGUF type 30 ([`GgmlType::BF16`]); the runtime widens
//! BF16 → f32 losslessly at load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (the same
//! choke point every sibling BF16 pass-through converter binds
//! against — never fabricated fp32 conversions elsewhere in the
//! tree).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for WavLM-SV GGUFs. Distinct from every sibling
/// speaker converter arch — never `campplus` (CAM++), never
/// `wespeaker` (ResNet-34), never `ecapa_tdnn` (TDNN stack), never
/// `titanet` (depth-wise separable Conv1D), never `speaker_3d`
/// (ERes2Net), never `redimnet` (2D dim-reduction + 1D conv+att).
/// Silently sharing an arch would misroute runtime dispatch
/// (FR-EX-08).
pub const ARCH: &str = "wavlm_sv";

/// `vokra.model.name` — canonical `microsoft/wavlm-base-plus-sv`
/// release, lowercase (HF org uses lowercase-only, arch-tag /
/// slug space is lowercase-only per the whole speaker fleet
/// convention).
pub const NAME: &str = "wavlm-base-plus-sv";

/// `vokra.model.category` — speaker (mirror of the sibling speaker
/// fleet — campplus / wespeaker / ecapa_tdnn / titanet / speaker_3d
/// / redimnet all stamp this same category so downstream consumers
/// can dispatch to a shared speaker-embedding load path).
pub const CATEGORY: &str = "speaker";

/// Upstream HF repository slug (`org/name`), recorded under
/// `vokra.provenance.upstream_hf` so a downstream can trace the
/// artifact back to its serving location without parsing the
/// free-text `vokra.provenance.source`. The slug preserves upstream
/// casing (`microsoft/wavlm-base-plus-sv` — HF slug is lowercase).
pub const UPSTREAM_HF: &str = "microsoft/wavlm-base-plus-sv";

/// Default SPDX. Upstream HF `microsoft/wavlm-base-plus-sv` card
/// links its LICENSE to `github.com/microsoft/UniSpeech/blob/main/LICENSE`
/// = **CC-BY-SA-3.0** (scout-time WebFetch, 2026-08-14). Overridable
/// through the `license` argument. Note: `LicenseClass::from_license_str`
/// resolves this to `Copyleft` via the `has_sa` arm (the ordering
/// pin dictates share-alike is tested before plain `cc-by`).
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-sa-3.0";

// ---------------------------------------------------------------------------
// WavLM Base+ SV hyperparameters — transcribed verbatim from the
// upstream `config.json` of `microsoft/wavlm-base-plus-sv` (scout-time
// WebFetch, 2026-08-14). Stamped on the GGUF so the runtime binder
// can validate topology + surface xvector_output_dim without
// re-inspecting tensor shapes.
//
// Sources:
// - huggingface.co/microsoft/wavlm-base-plus-sv/blob/main/config.json
// - github.com/microsoft/UniSpeech/tree/main/WavLM
// - arXiv:2110.13900 (Chen et al. 2022 "WavLM: Large-Scale
//   Self-Supervised Pre-Training for Full Stack Speech Processing")
// ---------------------------------------------------------------------------

/// Transformer hidden size — 768.
pub const HIDDEN_SIZE: u32 = 768;
/// Transformer layer count — 12.
pub const NUM_HIDDEN_LAYERS: u32 = 12;
/// Transformer attention head count — 12.
pub const NUM_ATTENTION_HEADS: u32 = 12;
/// Transformer FFN intermediate size — 3072.
pub const INTERMEDIATE_SIZE: u32 = 3072;
/// Number of 1D conv feature-extractor layers — 7.
pub const NUM_FEAT_EXTRACT_LAYERS: u32 = 7;
/// X-Vector head output dim (speaker embedding dim) — **512**.
/// NOTE: distinct from the sibling 192-d speaker fleet
/// (CAM++/ReDimNet); WavLM-SV uses 512-d embeddings.
pub const XVECTOR_OUTPUT_DIM: u32 = 512;
/// CTC vocab size present in config (unused for SV, but stamped for
/// audit) — 80.
pub const NUM_CTC_CLASSES: u32 = 80;
/// Convolutional positional-embedding kernel size — 128.
pub const NUM_CONV_POS_EMBEDDINGS: u32 = 128;
/// Convolutional positional-embedding group count — 16.
pub const NUM_CONV_POS_EMBEDDING_GROUPS: u32 = 16;
/// Audio sample rate — 16 kHz mono.
pub const SAMPLE_RATE: u32 = 16000;
/// LayerNorm epsilon written as a scaled u32 (× 1_000_000_000). The
/// primary-source value is `1e-5 = 0.00001`. Encoded as an integer
/// so the GGUF reader can round-trip it without floating-point
/// serialization ambiguity (some sibling converters do the same
/// dance — the runtime binder converts back to f32 on load).
pub const LAYER_NORM_EPS_SCALED_1E9: u32 = 10_000; // 1e-5 * 1e9 = 1e4
/// Pre-emphasis flag / feat_extract_norm flag — 1 = "group" (the
/// value the upstream `config.json` carries for Base+). Runtime
/// binder maps 1 = group / 0 = layer. WavLM defaults are group at
/// base scale, layer at large scale.
pub const FEAT_EXTRACT_NORM_GROUP: u32 = 1;
/// Hidden dropout scaled by 1000 for u32 encoding — 0.1 * 1000 = 100.
pub const HIDDEN_DROPOUT_SCALED_1E3: u32 = 100;

// Convolutional feature-extractor axis arrays. Same length = 7.
// Stamped as 7 scalar u32s each so a reader can reconstruct order
// deterministically (no array parsing needed).
pub const CONV_DIM: [u32; 7] = [512, 512, 512, 512, 512, 512, 512];
pub const CONV_STRIDE: [u32; 7] = [5, 2, 2, 2, 2, 2, 2];
pub const CONV_KERNEL: [u32; 7] = [10, 3, 3, 3, 3, 2, 2];

// XVector head TDNN axis arrays. Same length = 5. Stamped as 5
// scalar u32s each so a reader can reconstruct order deterministically.
pub const TDNN_DIM: [u32; 5] = [512, 512, 512, 512, 1500];
pub const TDNN_KERNEL: [u32; 5] = [5, 3, 3, 1, 1];
pub const TDNN_DILATION: [u32; 5] = [1, 2, 3, 1, 1];

// ---------------------------------------------------------------------------
// GGUF chunk keys — mirror of
// `crates/vokra-models/src/wavlm/mod.rs` `GGUF_KEY_*` (see runtime
// binder module doc for the cross-crate duplication rationale —
// `vokra-models` must not gain a dep edge onto `vokra-convert`).
// ---------------------------------------------------------------------------

/// `vokra.model.category` — auxiliary category stamp.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
/// `vokra.provenance.upstream_hf` — auxiliary provenance stamp.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

// Scalar topology chunks (`vokra.wavlm.*`).
const KEY_WAVLM_HIDDEN_SIZE: &str = "vokra.wavlm.hidden_size";
const KEY_WAVLM_NUM_HIDDEN_LAYERS: &str = "vokra.wavlm.num_hidden_layers";
const KEY_WAVLM_NUM_ATTENTION_HEADS: &str = "vokra.wavlm.num_attention_heads";
const KEY_WAVLM_INTERMEDIATE_SIZE: &str = "vokra.wavlm.intermediate_size";
const KEY_WAVLM_NUM_FEAT_EXTRACT_LAYERS: &str = "vokra.wavlm.num_feat_extract_layers";
const KEY_WAVLM_XVECTOR_OUTPUT_DIM: &str = "vokra.wavlm.xvector_output_dim";
const KEY_WAVLM_NUM_CTC_CLASSES: &str = "vokra.wavlm.num_ctc_classes";
const KEY_WAVLM_NUM_CONV_POS_EMBEDDINGS: &str = "vokra.wavlm.num_conv_pos_embeddings";
const KEY_WAVLM_NUM_CONV_POS_EMBEDDING_GROUPS: &str = "vokra.wavlm.num_conv_pos_embedding_groups";
const KEY_WAVLM_SAMPLE_RATE: &str = "vokra.wavlm.sample_rate";
const KEY_WAVLM_LAYER_NORM_EPS_SCALED_1E9: &str = "vokra.wavlm.layer_norm_eps_scaled_1e9";
const KEY_WAVLM_FEAT_EXTRACT_NORM_GROUP: &str = "vokra.wavlm.feat_extract_norm_group";
const KEY_WAVLM_HIDDEN_DROPOUT_SCALED_1E3: &str = "vokra.wavlm.hidden_dropout_scaled_1e3";

// Axis-array chunks (`vokra.wavlm.conv_*_{0..6}`) — 7 scalars each.
const KEY_WAVLM_CONV_DIM_PREFIX: &str = "vokra.wavlm.conv_dim";
const KEY_WAVLM_CONV_STRIDE_PREFIX: &str = "vokra.wavlm.conv_stride";
const KEY_WAVLM_CONV_KERNEL_PREFIX: &str = "vokra.wavlm.conv_kernel";

// XVector TDNN axis-array chunks (`vokra.wavlm.tdnn_*_{0..4}`) — 5 scalars each.
const KEY_WAVLM_TDNN_DIM_PREFIX: &str = "vokra.wavlm.tdnn_dim";
const KEY_WAVLM_TDNN_KERNEL_PREFIX: &str = "vokra.wavlm.tdnn_kernel";
const KEY_WAVLM_TDNN_DILATION_PREFIX: &str = "vokra.wavlm.tdnn_dilation";

const UPSTREAM_SOURCE: &str = "microsoft/wavlm-base-plus-sv \
     (WavLM Base+ speaker verification, HuBERT-lineage SSL encoder + gated relative position \
     bias + convolutional position-bias fusion + XVector head with Additive Margin Softmax, \
     VoxCeleb1 fine-tuned, 16 kHz mono → 512-d embedding, ~377 MB pytorch_model.bin, \
     arXiv:2110.13900, cc-by-sa-3.0 via github.com/microsoft/UniSpeech LICENSE)";

/// Outcome of a WavLM-SV conversion. Mirrors the counter shape of
/// [`crate::models::redimnet::RedimnetReport`] — the invariant
/// `read == written + skipped_non_float` is auditable at the report
/// level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WavlmSvReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16 at parse time, so any
    /// tensor reaching this counter would signal a reader change
    /// upstream; kept for symmetry with the sibling wespeaker /
    /// ecapa_tdnn / titanet / speaker_3d / redimnet reports).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16
    /// (subset counter). Emits GGUF type 30 verbatim; the runtime
    /// widens BF16 → f32 losslessly via the single choke point
    /// `vokra_core::gguf::quant::decode_bf16`.
    pub bf16_passthrough: usize,
}

/// Converts a `microsoft/wavlm-base-plus-sv` safetensors checkpoint
/// at `input` into a Vokra-native GGUF at `output`, returning a
/// [`WavlmSvReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` / `vokra.provenance.*` /
/// `vokra.wavlm.*` chunks are stamped for the runtime compliance
/// gate (FR-CP-03) and the runtime binder topology validation.
///
/// # License override
///
/// `license` overrides the default `cc-by-sa-3.0` SPDX string
/// stamped on `vokra.provenance.license` (whisper / kokoro-family
/// override pattern — see `convert_file_licensed` in `lib.rs`).
/// `None` keeps the built-in `cc-by-sa-3.0` stamp. **Note the
/// `Copyleft` class**: a `None` override binds `LicenseClass::Copyleft`
/// (share-alike obligations propagate); a `Some("apache-2.0")`
/// override would be a misrepresentation unless the owner
/// re-licensed the artifact separately.
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_wavlm_sv_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<WavlmSvReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    // WavLM Base+ SV topology axes (transcribed verbatim from the
    // upstream config.json + arXiv:2110.13900 §3-4 + the UniSpeech
    // repo's WavLM directory). The runtime binder is a strict
    // loader: every axis is required (FR-EX-08 — no primary-source
    // constant fallback since a partial stamp would fabricate axes
    // without primary-source backing).
    b.add_u32(KEY_WAVLM_HIDDEN_SIZE, HIDDEN_SIZE);
    b.add_u32(KEY_WAVLM_NUM_HIDDEN_LAYERS, NUM_HIDDEN_LAYERS);
    b.add_u32(KEY_WAVLM_NUM_ATTENTION_HEADS, NUM_ATTENTION_HEADS);
    b.add_u32(KEY_WAVLM_INTERMEDIATE_SIZE, INTERMEDIATE_SIZE);
    b.add_u32(KEY_WAVLM_NUM_FEAT_EXTRACT_LAYERS, NUM_FEAT_EXTRACT_LAYERS);
    b.add_u32(KEY_WAVLM_XVECTOR_OUTPUT_DIM, XVECTOR_OUTPUT_DIM);
    b.add_u32(KEY_WAVLM_NUM_CTC_CLASSES, NUM_CTC_CLASSES);
    b.add_u32(KEY_WAVLM_NUM_CONV_POS_EMBEDDINGS, NUM_CONV_POS_EMBEDDINGS);
    b.add_u32(
        KEY_WAVLM_NUM_CONV_POS_EMBEDDING_GROUPS,
        NUM_CONV_POS_EMBEDDING_GROUPS,
    );
    b.add_u32(KEY_WAVLM_SAMPLE_RATE, SAMPLE_RATE);
    b.add_u32(
        KEY_WAVLM_LAYER_NORM_EPS_SCALED_1E9,
        LAYER_NORM_EPS_SCALED_1E9,
    );
    b.add_u32(KEY_WAVLM_FEAT_EXTRACT_NORM_GROUP, FEAT_EXTRACT_NORM_GROUP);
    b.add_u32(
        KEY_WAVLM_HIDDEN_DROPOUT_SCALED_1E3,
        HIDDEN_DROPOUT_SCALED_1E3,
    );

    // Axis arrays: 7 conv-* scalars + 5 tdnn-* scalars. Indexed keys
    // (`vokra.wavlm.conv_dim_0` .. `vokra.wavlm.conv_dim_6`) so the
    // reader reconstructs order deterministically without needing
    // array parsing.
    for (i, &v) in CONV_DIM.iter().enumerate() {
        b.add_u32(&format!("{KEY_WAVLM_CONV_DIM_PREFIX}_{i}"), v);
    }
    for (i, &v) in CONV_STRIDE.iter().enumerate() {
        b.add_u32(&format!("{KEY_WAVLM_CONV_STRIDE_PREFIX}_{i}"), v);
    }
    for (i, &v) in CONV_KERNEL.iter().enumerate() {
        b.add_u32(&format!("{KEY_WAVLM_CONV_KERNEL_PREFIX}_{i}"), v);
    }
    for (i, &v) in TDNN_DIM.iter().enumerate() {
        b.add_u32(&format!("{KEY_WAVLM_TDNN_DIM_PREFIX}_{i}"), v);
    }
    for (i, &v) in TDNN_KERNEL.iter().enumerate() {
        b.add_u32(&format!("{KEY_WAVLM_TDNN_KERNEL_PREFIX}_{i}"), v);
    }
    for (i, &v) in TDNN_DILATION.iter().enumerate() {
        b.add_u32(&format!("{KEY_WAVLM_TDNN_DILATION_PREFIX}_{i}"), v);
    }

    // Self-describing redistribution: the artifact carries its own
    // licence. Default = cc-by-sa-3.0 (HF card LICENSE link primary
    // source 2026-08-14, resolves to `LicenseClass::Copyleft`).
    // `license` overrides for callers who obtained the weight under
    // a different SPDX.
    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Copyleft),
    };
    vokra_core::stamp_provenance(&mut b, class, &spdx, Some(NAME), Some(UPSTREAM_SOURCE));

    let mut report = WavlmSvReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30), same posture as wespeaker /
    // ecapa_tdnn / titanet / speaker_3d / redimnet; runtime widens
    // BF16 → f32 exactly at load via
    // `vokra_core::gguf::quant::decode_bf16` (`bits << 16` is exact).
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
    use vokra_core::gguf::{GgmlType, GgufFile};

    /// Builds a single-BF16-tensor safetensors buffer with a
    /// caller-supplied raw payload. Mirror of the redimnet test
    /// harness — same JSON header shape.
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

    /// Builds a two-tensor safetensors buffer (F32 first, then F16)
    /// with caller-supplied payloads. Mirror of the redimnet test
    /// harness.
    fn safetensors_f32_then_f16(
        f32_name: &str,
        f32_shape: &[u64],
        f32_bytes: &[u8],
        f16_name: &str,
        f16_shape: &[u64],
        f16_bytes: &[u8],
    ) -> Vec<u8> {
        let f32_elems: u64 = f32_shape.iter().product();
        assert_eq!(
            f32_bytes.len(),
            f32_elems as usize * 4,
            "F32 payload len must match shape × 4"
        );
        let f16_elems: u64 = f16_shape.iter().product();
        assert_eq!(
            f16_bytes.len(),
            f16_elems as usize * 2,
            "F16 payload len must match shape × 2"
        );
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

    /// Writes `bytes` to a fresh temp file and returns its path.
    /// Nanosecond suffix keeps parallel `cargo test` runs from
    /// colliding on the same PID.
    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-wavlm-sv-{kind}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, bytes).expect("write temp file");
        p
    }

    // -----------------------------------------------------------------------
    // 1. BF16 round-trip (byte-identical, counter surfaces, provenance
    //    stamps landed as Copyleft per default cc-by-sa-3.0)
    // -----------------------------------------------------------------------

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Non-zero BF16 bit patterns so a subsequent byte-identity assert
        // catches any silent widen / downcast attempt (zeroed payloads
        // would round-trip trivially through F32 / F16 widen too).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");

        // Mirror a plausible upstream WavLM tensor name. The
        // `wavlm.feature_extractor.conv_layers.0.conv.weight` prefix
        // matches HuBERT/wav2vec2 lineage naming.
        let input_bytes = safetensors_one_bf16(
            "wavlm.feature_extractor.conv_layers.0.conv.weight",
            &[2, 3],
            &bf16,
        );
        let input_path = write_temp("bf16-in", &input_bytes);
        let output_path = write_temp("bf16-out", &[]);

        let report = convert_wavlm_sv_file(&input_path, &output_path, None)
            .expect("convert_wavlm_sv_file must accept a well-formed BF16 checkpoint");
        assert_eq!(report.read, 1, "one tensor observed");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm (mirror redimnet)"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 tensor must increment the observability counter"
        );

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");
        let info = file
            .tensor_info("wavlm.feature_extractor.conv_layers.0.conv.weight")
            .expect("BF16 tensor present in output");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — BF16 stays BF16 (GGUF type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info).len(),
            12,
            "2 rows × 3 cols × 2 B BF16 verbatim"
        );
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );

        // Provenance stamps landed on the arch / name / category /
        // upstream-hf axes.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );

        // Default license = cc-by-sa-3.0 → LicenseClass::Copyleft
        // (share-alike arm in `from_license_str` fires before plain
        // `cc-by`). The whole point of the row is to catch anyone who
        // regresses that ordering pin.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Copyleft.as_str()),
            "cc-by-sa-3.0 must classify as Copyleft (share-alike arm before plain cc-by)"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    // -----------------------------------------------------------------------
    // 2. Mixed F32 / F16 round-trip with metadata assertions
    // -----------------------------------------------------------------------

    #[test]
    fn f32_and_f16_tensors_pass_through_with_full_metadata() {
        // Non-zero payloads so a silent-widen regression can't hide
        // behind trivial round-trips.
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        // F16 exact-representable values via manual half bit-fiddling
        // (no external crate). 1.0 = 0x3C00, -2.0 = 0xC000,
        // -0.5 = 0xB800, 3.0 = 0x4200, 0.15625 = 0x3100, 42.0 = 0x5140.
        // Six values for a [2,3] tensor = 12 bytes.
        let f16_words: [u16; 6] = [0x3C00, 0xC000, 0xB800, 0x4200, 0x3100, 0x5140];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(f16_bytes.len(), 12, "6 elements × 2 bytes F16 payload");

        let input_bytes = safetensors_f32_then_f16(
            "wavlm.encoder.layers.0.attention.q_proj.weight",
            &[1, 2],
            &f32_bytes,
            "objective.projection.weight",
            &[2, 3],
            &f16_bytes,
        );
        let input_path = write_temp("mixed-in", &input_bytes);
        let output_path = write_temp("mixed-out", &[]);

        let report = convert_wavlm_sv_file(&input_path, &output_path, None)
            .expect("convert_wavlm_sv_file must accept a mixed F32/F16 checkpoint");

        assert_eq!(report.read, 2, "two tensors observed");
        assert_eq!(
            report.written, 2,
            "both F32 and F16 tensors must pass through"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32/F16 must NOT increment the BF16 counter"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "no tensor may reach the skipped arm"
        );

        // Round-trip carries both tensors with their dtypes preserved
        // AND the arch / provenance / category / topology stamps land.
        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");

        let f32_info = file
            .tensor_info("wavlm.encoder.layers.0.attention.q_proj.weight")
            .expect("F32 tensor present");
        assert_eq!(f32_info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(file.tensor_bytes(f32_info), f32_bytes.as_slice());

        let f16_info = file
            .tensor_info("objective.projection.weight")
            .expect("F16 tensor present");
        assert_eq!(f16_info.dtype, GgmlType::F16, "F16 stays F16");
        assert_eq!(file.tensor_bytes(f16_info), f16_bytes.as_slice());

        // Task-spec pins: `KEY_MODEL_ARCH` / `KEY_MODEL_CATEGORY` /
        // `KEY_PROVENANCE_UPSTREAM_HF` all land + license class is
        // Copyleft for the default cc-by-sa-3.0.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Copyleft.as_str())
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    // -----------------------------------------------------------------------
    // 3. Full topology chunk group is stamped and readable — every
    //    scalar axis + every axis-array (conv_dim/stride/kernel × 7,
    //    tdnn_dim/kernel/dilation × 5) round-trips.
    // -----------------------------------------------------------------------

    #[test]
    fn topology_chunks_round_trip() {
        // A single F32 tensor is enough to trigger the write path;
        // all scalar + array topology axes must round-trip through the GGUF.
        let f32_bytes: Vec<u8> = [0.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = format!(
            r#"{{"dummy":{{"dtype":"F32","shape":[1],"data_offsets":[0,{}]}}}}"#,
            f32_bytes.len()
        );
        let mut input = Vec::new();
        input.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input.extend_from_slice(header.as_bytes());
        input.extend_from_slice(&f32_bytes);
        let input_path = write_temp("topology-in", &input);
        let output_path = write_temp("topology-out", &[]);

        convert_wavlm_sv_file(&input_path, &output_path, None).expect("convert");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");

        // Scalar topology axes — every stamped axis must round-trip; a
        // rename would land here in the same commit or fail this test.
        assert_eq!(
            file.get(KEY_WAVLM_HIDDEN_SIZE).and_then(|v| v.as_u64()),
            Some(u64::from(HIDDEN_SIZE))
        );
        assert_eq!(
            file.get(KEY_WAVLM_NUM_HIDDEN_LAYERS)
                .and_then(|v| v.as_u64()),
            Some(u64::from(NUM_HIDDEN_LAYERS))
        );
        assert_eq!(
            file.get(KEY_WAVLM_NUM_ATTENTION_HEADS)
                .and_then(|v| v.as_u64()),
            Some(u64::from(NUM_ATTENTION_HEADS))
        );
        assert_eq!(
            file.get(KEY_WAVLM_INTERMEDIATE_SIZE)
                .and_then(|v| v.as_u64()),
            Some(u64::from(INTERMEDIATE_SIZE))
        );
        assert_eq!(
            file.get(KEY_WAVLM_NUM_FEAT_EXTRACT_LAYERS)
                .and_then(|v| v.as_u64()),
            Some(u64::from(NUM_FEAT_EXTRACT_LAYERS))
        );
        assert_eq!(
            file.get(KEY_WAVLM_XVECTOR_OUTPUT_DIM)
                .and_then(|v| v.as_u64()),
            Some(u64::from(XVECTOR_OUTPUT_DIM))
        );
        assert_eq!(
            file.get(KEY_WAVLM_NUM_CTC_CLASSES).and_then(|v| v.as_u64()),
            Some(u64::from(NUM_CTC_CLASSES))
        );
        assert_eq!(
            file.get(KEY_WAVLM_NUM_CONV_POS_EMBEDDINGS)
                .and_then(|v| v.as_u64()),
            Some(u64::from(NUM_CONV_POS_EMBEDDINGS))
        );
        assert_eq!(
            file.get(KEY_WAVLM_NUM_CONV_POS_EMBEDDING_GROUPS)
                .and_then(|v| v.as_u64()),
            Some(u64::from(NUM_CONV_POS_EMBEDDING_GROUPS))
        );
        assert_eq!(
            file.get(KEY_WAVLM_SAMPLE_RATE).and_then(|v| v.as_u64()),
            Some(u64::from(SAMPLE_RATE))
        );
        assert_eq!(
            file.get(KEY_WAVLM_LAYER_NORM_EPS_SCALED_1E9)
                .and_then(|v| v.as_u64()),
            Some(u64::from(LAYER_NORM_EPS_SCALED_1E9))
        );
        assert_eq!(
            file.get(KEY_WAVLM_FEAT_EXTRACT_NORM_GROUP)
                .and_then(|v| v.as_u64()),
            Some(u64::from(FEAT_EXTRACT_NORM_GROUP))
        );
        assert_eq!(
            file.get(KEY_WAVLM_HIDDEN_DROPOUT_SCALED_1E3)
                .and_then(|v| v.as_u64()),
            Some(u64::from(HIDDEN_DROPOUT_SCALED_1E3))
        );

        // Array axes — 7 conv-* per array + 5 tdnn-* per array must
        // round-trip in exactly the primary-source order.
        for (i, &expected) in CONV_DIM.iter().enumerate() {
            let k = format!("{KEY_WAVLM_CONV_DIM_PREFIX}_{i}");
            assert_eq!(
                file.get(&k).and_then(|v| v.as_u64()),
                Some(u64::from(expected)),
                "conv_dim[{i}] mismatch"
            );
        }
        for (i, &expected) in CONV_STRIDE.iter().enumerate() {
            let k = format!("{KEY_WAVLM_CONV_STRIDE_PREFIX}_{i}");
            assert_eq!(
                file.get(&k).and_then(|v| v.as_u64()),
                Some(u64::from(expected)),
                "conv_stride[{i}] mismatch"
            );
        }
        for (i, &expected) in CONV_KERNEL.iter().enumerate() {
            let k = format!("{KEY_WAVLM_CONV_KERNEL_PREFIX}_{i}");
            assert_eq!(
                file.get(&k).and_then(|v| v.as_u64()),
                Some(u64::from(expected)),
                "conv_kernel[{i}] mismatch"
            );
        }
        for (i, &expected) in TDNN_DIM.iter().enumerate() {
            let k = format!("{KEY_WAVLM_TDNN_DIM_PREFIX}_{i}");
            assert_eq!(
                file.get(&k).and_then(|v| v.as_u64()),
                Some(u64::from(expected)),
                "tdnn_dim[{i}] mismatch"
            );
        }
        for (i, &expected) in TDNN_KERNEL.iter().enumerate() {
            let k = format!("{KEY_WAVLM_TDNN_KERNEL_PREFIX}_{i}");
            assert_eq!(
                file.get(&k).and_then(|v| v.as_u64()),
                Some(u64::from(expected)),
                "tdnn_kernel[{i}] mismatch"
            );
        }
        for (i, &expected) in TDNN_DILATION.iter().enumerate() {
            let k = format!("{KEY_WAVLM_TDNN_DILATION_PREFIX}_{i}");
            assert_eq!(
                file.get(&k).and_then(|v| v.as_u64()),
                Some(u64::from(expected)),
                "tdnn_dilation[{i}] mismatch"
            );
        }

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    // -----------------------------------------------------------------------
    // 4. License override (Apache-2.0) still classifies as Permissive,
    //    proving the override wiring does NOT get pinned to Copyleft
    //    only because the default is Copyleft. Documents the semver
    //    override contract at test level.
    // -----------------------------------------------------------------------

    #[test]
    fn license_override_reclassifies_via_from_license_str() {
        // Even for an override the class is derived from the SPDX via
        // `LicenseClass::from_license_str` — the override does NOT
        // silently inherit the default Copyleft class.
        let f32_bytes: Vec<u8> = [0.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = format!(
            r#"{{"dummy":{{"dtype":"F32","shape":[1],"data_offsets":[0,{}]}}}}"#,
            f32_bytes.len()
        );
        let mut input = Vec::new();
        input.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input.extend_from_slice(header.as_bytes());
        input.extend_from_slice(&f32_bytes);
        let input_path = write_temp("license-in", &input);
        let output_path = write_temp("license-out", &[]);

        convert_wavlm_sv_file(&input_path, &output_path, Some("apache-2.0")).expect("convert");

        let out_bytes = std::fs::read(&output_path).expect("read output GGUF");
        let file = GgufFile::parse(out_bytes).expect("parse output GGUF");

        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0"),
            "override SPDX must land verbatim"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "override apache-2.0 must reclassify to Permissive (not pinned to Copyleft \
             just because the default is)"
        );

        std::fs::remove_file(&input_path).ok();
        std::fs::remove_file(&output_path).ok();
    }

    // -----------------------------------------------------------------------
    // 5. Arch-tag distinctness pin — WavLM-SV MUST NOT collide with
    //    any sibling speaker-fleet arch.
    // -----------------------------------------------------------------------

    #[test]
    fn arch_tag_distinct_from_sibling_speaker_arches() {
        // Pin the arch string so a rename would land here in the same
        // commit or fail this test.
        assert_eq!(ARCH, "wavlm_sv");
        assert_ne!(
            ARCH, "campplus",
            "wavlm_sv (Transformer + XVector) and campplus (CAM++ D-TDNN) are \
             different topologies — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "wespeaker",
            "wavlm_sv and wespeaker (ResNet-34 backbone) are different \
             topologies — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "ecapa_tdnn",
            "wavlm_sv and ecapa_tdnn (TDNN stack backbone) are different \
             topologies — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "titanet",
            "wavlm_sv and titanet (depth-wise separable Conv1D backbone) \
             are different topologies — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "speaker_3d",
            "wavlm_sv and speaker_3d (ERes2Net backbone) are different \
             topologies — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "redimnet",
            "wavlm_sv (Transformer + XVector) and redimnet (2D dim-reduction + \
             1D conv+att + ASTP) are different topologies — sharing arch would \
             mis-route (FR-EX-08)"
        );
    }
}
