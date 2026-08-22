//! **plain VITS JA**: safetensors checkpoint → GGUF conversion
//! (SoTA plan Phase 5 JA-TTS-2, 2026-07-24).
//!
//! Input: an ESPnet-family plain VITS (Kim et al. 2021 VITS +
//! HiFi-GAN generator) safetensors checkpoint — typically an ESPnet
//! JSUT / JVS / COEIROINK export, or any downstream re-training on a
//! permissive corpus. Output: a GGUF carrying every float tensor plus
//! the `vokra.vits_ja.*` / `vokra.provenance.*` / `vokra.model.*`
//! metadata chunks that the native plain VITS JA implementation
//! (`crates/vokra-models/src/vits_ja/`) reads.
//!
//! # ⚠️  Weight redistribution default is **`RedistributionForbidden`**
//!
//! The publicly distributed ESPnet-JSUT / ESPnet-JVS / COEIROINK JA
//! VITS checkpoints ride on **corpus terms that forbid re-distribution
//! of the trained weight**:
//!
//! - **JSUT** (`sites.google.com/site/shinnosuketakamichi/publication/jsut`)
//!   — *"Re-distribution is not permitted"*.
//! - **JVS** (`sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus`)
//!   — same re-distribution ban.
//! - **COEIROINK** — per-character licence terms that a converter
//!   cannot machine-check.
//!
//! The provenance stamp therefore defaults to
//! [`LicenseClass::RedistributionForbidden`]. A user who trained their
//! own VITS on a permissive corpus overrides at the outer
//! `vokra-convert --license <spdx>` boundary (see the `--license` flag
//! in `crates/vokra-convert/src/lib.rs`), which rewrites the
//! provenance chunk to the correct SPDX id.
//!
//! Architecture rides Apache 2.0 (ESPnet's `espnet2/gan_tts/vits/`) and
//! MIT (`jaywalnut310/vits` reference) and is *always* independently
//! implementable (whisper.cpp 型 self re-implementation, CLAUDE.md
//! 設計判断 4).
//!
//! # What is transcribed vs. shape-driven
//!
//! - **Transcribed constants** — every hparam of the `vokra.vits_ja.*`
//!   chunk group is transcribed **verbatim** from the primary sources
//!   `egs2/jsut/tts1/conf/tuning/train_vits.yaml` +
//!   `egs2/jvs/tts1/conf/tuning/finetune_vits.yaml` +
//!   `espnet2/gan_tts/vits/{vits,generator}.py` (fetched 2026-07-24
//!   — CLAUDE.md「ハルシネーション厳禁」).
//! - **Sample rate** — the JSUT / JVS default is **22050 Hz**
//!   (`train_vits.yaml.tts_conf.sampling_rate: 22050`); the full-band
//!   variant `train_full_band_vits.yaml` re-shapes the decoder + FFT
//!   and emits **44100 Hz**. This converter defaults to the 22 kHz
//!   axis; a caller who trained on the full-band recipe re-stamps via
//!   the `restamp` subcommand or a follow-up `--config` side-car.
//! - **Speaker count / vocabulary** — `spks` / `vocab_size` are
//!   **not** encoded in the shared training YAML for the ESPnet
//!   defaults; the JVS variant sets `spks = 100`. This converter's
//!   defaults follow the JSUT single-speaker recipe. A JVS or
//!   downstream multi-speaker variant would override at bind time.
//! - **No side-car config today** — every field of the JSUT
//!   `train_vits.yaml` is fixed for the 22 kHz single-speaker recipe
//!   and byte-parallel to the transcribed constants below. A future
//!   `--config` axis (for the JVS variant, the full-band variant, or
//!   a downstream re-training) is a follow-up.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the normalized upstream `VITSGenerator` names
//! verbatim. Operator-held ESPnet `.pth` files must first pass through
//! `tools/parity/vits_ja_prepare_checkpoint.py`, which strips known wrapper
//! prefixes and validates the canonical 885-tensor manifest. The runtime's
//! `VitsJaCheckpoint` binds that complete manifest; arbitrary training-state
//! dictionaries are rejected rather than partly consumed.
//!
//! # BF16 posture
//!
//! ESPnet VITS checkpoints are typically served in F32 (the default
//! `save_pretrained` posture) or F16 (mixed-precision train + widen at
//! save). BF16 is also accepted through the pass-through arm
//! (2026-07-25, mirror of qwen3-tts / vibevoice / voxcpm2 / moshi /
//! voxtral): BF16 bytes emit as GGUF type 30 verbatim and the runtime
//! widens on load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 = top
//! 16 bits of an f32 — `bits << 16` is exact, no precision loss).
//!
//! # No ONNX (permanent)
//!
//! ESPnet distributes VITS checkpoints as PyTorch `.pth` (`espnet2/gan_tts`);
//! this converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively in `crates/vokra-models/src/vits_ja/`
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// Emit a `u32` array under `key`. Follows the CSM / VibeVoice /
/// distil-whisper pattern (`add_metadata(GgufMetadataValue::Array(...))`)
/// — the builder does not carry a typed `add_*_array` shortcut.
fn add_u32_array(b: &mut GgufBuilder, key: &str, values: &[u32]) {
    b.add_metadata(
        key,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: values.iter().map(|&v| GgufMetadataValue::U32(v)).collect(),
        }),
    );
}

/// `vokra.model.arch` for plain VITS JA GGUFs — kept in sync with the
/// runtime constant `vokra-models::vits_ja::EXPECTED_ARCH`.
/// Intentionally **distinct** from piper-plus (MB-iSTFT-VITS2) because
/// plain VITS decodes through a HiFi-GAN generator directly while
/// piper-plus decodes through a sub-band iSTFT + PQMF post-net;
/// silently sharing an arch tag would misroute the runtime dispatch.
pub(crate) const ARCH: &str = "vits-ja";

/// `vokra.model.name` value written for the canonical ESPnet JA VITS
/// (JSUT 22 kHz single-speaker) GGUF.
pub(crate) const NAME: &str = "espnet-jsut-vits-22khz";

// --- vokra.vits_ja.* metadata keys --------------------------------------
// The runtime side lives in `crates/vokra-models/src/vits_ja/mod.rs`
// — the two crates share only `vokra-core`, so the cross-crate constant
// duplication rule the CSM / CosyVoice2 / Kokoro / Chatterbox / Qwen3-TTS
// / VoxCPM / VibeVoice / Irodori family converters use applies.

// Top-level
const KEY_MODEL_FAMILY: &str = "vokra.vits_ja.model_family";
const KEY_SAMPLE_RATE_HZ: &str = "vokra.vits_ja.sample_rate_hz";
const KEY_VOCAB_SIZE: &str = "vokra.vits_ja.vocab_size";
const KEY_N_MELS: &str = "vokra.vits_ja.n_mels";
const KEY_AUX_CHANNELS: &str = "vokra.vits_ja.aux_channels";
const KEY_HIDDEN_CHANNELS: &str = "vokra.vits_ja.hidden_channels";
const KEY_SEGMENT_SIZE: &str = "vokra.vits_ja.segment_size";
const KEY_SPKS: &str = "vokra.vits_ja.spks";

// Text encoder — train_vits.yaml.tts_conf.generator_params.*
const KEY_TEXT_N_LAYER: &str = "vokra.vits_ja.text.n_layer";
const KEY_TEXT_N_HEAD: &str = "vokra.vits_ja.text.n_head";
const KEY_TEXT_FFN_EXPAND: &str = "vokra.vits_ja.text.ffn_expand";
const KEY_TEXT_POSITIONWISE_CONV_KERNEL: &str = "vokra.vits_ja.text.positionwise_conv_kernel";
const KEY_TEXT_DROPOUT_RATE: &str = "vokra.vits_ja.text.dropout_rate";
const KEY_TEXT_POSITIONAL_DROPOUT_RATE: &str = "vokra.vits_ja.text.positional_dropout_rate";
const KEY_TEXT_ATTENTION_DROPOUT_RATE: &str = "vokra.vits_ja.text.attention_dropout_rate";
const KEY_TEXT_USE_MACARON_STYLE: &str = "vokra.vits_ja.text.use_macaron_style";
const KEY_TEXT_USE_CONFORMER_CONV: &str = "vokra.vits_ja.text.use_conformer_conv";

// Flow (residual affine coupling) — train_vits.yaml.tts_conf.generator_params.*
const KEY_FLOW_N_FLOW: &str = "vokra.vits_ja.flow.n_flow";
const KEY_FLOW_KERNEL_SIZE: &str = "vokra.vits_ja.flow.kernel_size";
const KEY_FLOW_BASE_DILATION: &str = "vokra.vits_ja.flow.base_dilation";
const KEY_FLOW_N_LAYER: &str = "vokra.vits_ja.flow.n_layer";
const KEY_FLOW_DROPOUT_RATE: &str = "vokra.vits_ja.flow.dropout_rate";
const KEY_FLOW_USE_ONLY_MEAN: &str = "vokra.vits_ja.flow.use_only_mean";

// Stochastic duration predictor — train_vits.yaml.tts_conf.generator_params.*
const KEY_SDP_KERNEL_SIZE: &str = "vokra.vits_ja.sdp.kernel_size";
const KEY_SDP_DROPOUT_RATE: &str = "vokra.vits_ja.sdp.dropout_rate";
const KEY_SDP_N_FLOW: &str = "vokra.vits_ja.sdp.n_flow";
const KEY_SDP_DDS_CONV_LAYERS: &str = "vokra.vits_ja.sdp.dds_conv_layers";

// HiFi-GAN decoder — train_vits.yaml.tts_conf.generator_params.*
const KEY_DECODER_KERNEL_SIZE: &str = "vokra.vits_ja.decoder.kernel_size";
const KEY_DECODER_INITIAL_CHANNEL: &str = "vokra.vits_ja.decoder.initial_channel";
const KEY_DECODER_UPSAMPLE_SCALES: &str = "vokra.vits_ja.decoder.upsample_scales";
const KEY_DECODER_UPSAMPLE_KERNEL_SIZES: &str = "vokra.vits_ja.decoder.upsample_kernel_sizes";
const KEY_DECODER_RESBLOCK_KERNEL_SIZES: &str = "vokra.vits_ja.decoder.resblock_kernel_sizes";
// resblock_dilations is a 2D structure; we flatten it into pairs
// (branch-index → dilations) via a `[n_branch * n_layer_per_branch]`
// array + a companion width chunk. For the ESPnet JA default all
// branches share the same layer count (3), so a single `stride` chunk
// suffices.
const KEY_DECODER_RESBLOCK_DILATIONS_FLAT: &str =
    "vokra.vits_ja.decoder.resblock_dilations_flat_u32";
const KEY_DECODER_RESBLOCK_DILATIONS_STRIDE: &str =
    "vokra.vits_ja.decoder.resblock_dilations_stride";
const KEY_DECODER_USE_WEIGHT_NORM: &str = "vokra.vits_ja.decoder.use_weight_norm";

// --- Transcribed constants ------------------------------------------------
// Primary sources: `egs2/jsut/tts1/conf/tuning/train_vits.yaml` +
// `egs2/jvs/tts1/conf/tuning/finetune_vits.yaml` +
// `espnet2/gan_tts/vits/vits.py::AVAILABLE_GENERATERS.vits_generator`
// (fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」).

/// Model family marker.
const MODEL_FAMILY: &str = "vits-ja";

/// PCM sample rate the JSUT / JVS 22 kHz recipe emits.
const SAMPLE_RATE_HZ: u32 = 22_050;

/// pyopenjtalk-derived JA phoneme vocabulary size (canonical JSUT
/// recipe default; a downstream re-training would override).
const VOCAB_SIZE: u32 = 43;

/// Mel bin count fed to the HiFi-GAN decoder (matches ESPnet's
/// `mel_loss_params.n_mels = 80` acoustic target — see
/// `train_vits.yaml.tts_conf.mel_loss_params`).
const N_MELS: u32 = 80;

/// Posterior-encoder input width `aux_channels = n_fft / 2 + 1`
/// (`n_fft = 1024` on the JA 22 kHz recipe).
const AUX_CHANNELS: u32 = 513;

/// Residual-stream width (`hidden_channels = 192`).
const HIDDEN_CHANNELS: u32 = 192;

/// Training-time random-segment length in posterior-encoder frames.
const SEGMENT_SIZE: u32 = 32;

/// Speaker count — **0** sentinel = single-speaker (JSUT default);
/// the JVS variant sets 100.
const SPKS: u32 = 0;

// Text encoder — train_vits.yaml.tts_conf.generator_params.*
const TEXT_N_LAYER: u32 = 6;
const TEXT_N_HEAD: u32 = 2;
const TEXT_FFN_EXPAND: u32 = 4;
const TEXT_POSITIONWISE_CONV_KERNEL: u32 = 3;
const TEXT_DROPOUT_RATE: f32 = 0.1;
const TEXT_POSITIONAL_DROPOUT_RATE: f32 = 0.0;
const TEXT_ATTENTION_DROPOUT_RATE: f32 = 0.1;
const TEXT_USE_MACARON_STYLE: bool = true;
const TEXT_USE_CONFORMER_CONV: bool = false;

// Flow (residual affine coupling).
const FLOW_N_FLOW: u32 = 4;
const FLOW_KERNEL_SIZE: u32 = 5;
const FLOW_BASE_DILATION: u32 = 1;
const FLOW_N_LAYER: u32 = 4;
const FLOW_DROPOUT_RATE: f32 = 0.0;
const FLOW_USE_ONLY_MEAN: bool = true;

// Stochastic duration predictor.
const SDP_KERNEL_SIZE: u32 = 3;
const SDP_DROPOUT_RATE: f32 = 0.5;
const SDP_N_FLOW: u32 = 4;
const SDP_DDS_CONV_LAYERS: u32 = 3;

// HiFi-GAN decoder (22 kHz JA recipe).
const DECODER_KERNEL_SIZE: u32 = 7;
const DECODER_INITIAL_CHANNEL: u32 = 512;
const DECODER_UPSAMPLE_SCALES: [u32; 4] = [8, 8, 2, 2];
const DECODER_UPSAMPLE_KERNEL_SIZES: [u32; 4] = [16, 16, 4, 4];
const DECODER_RESBLOCK_KERNEL_SIZES: [u32; 3] = [3, 7, 11];
const DECODER_RESBLOCK_DILATIONS_FLAT: [u32; 9] = [1, 3, 5, 1, 3, 5, 1, 3, 5];
const DECODER_RESBLOCK_DILATIONS_STRIDE: u32 = 3;
const DECODER_USE_WEIGHT_NORM: bool = true;

/// Outcome of a plain VITS JA conversion.
#[derive(Debug, Default)]
pub(crate) struct VitsJaReport {
    /// Float tensors written verbatim (F32 / F16 / BF16 — all three go
    /// through the same byte-copy path since the BF16 pass-through land
    /// 2026-07-25, mirror of `qwen3-tts` / `vibevoice` / `voxcpm2` /
    /// `moshi` / `voxtral`).
    pub(crate) written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time; anything
    /// that reaches this arm is a quantized dtype the runtime is not
    /// expected to consume).
    pub(crate) skipped_non_float: usize,
    /// Of the tensors in `written`, how many were BF16 (subset counter).
    /// Emits GGUF type 30 verbatim; runtime widens BF16 → f32 losslessly
    /// via the single choke point `crates/vokra-core/src/gguf/quant/mod.rs
    /// decode_bf16` (BF16 = top 16 bits of an f32 — `bits << 16` is exact).
    pub(crate) bf16_passthrough: usize,
    /// Operator-facing diagnostics (never fail the conversion — the
    /// runtime is the authoritative gate, FR-EX-08).
    pub(crate) notes: Vec<String>,
}

/// Converts an ESPnet-family plain VITS JA safetensors buffer into a
/// populated GGUF builder.
///
/// Every F32 / F16 tensor passes through under its upstream name; the
/// `vokra.vits_ja.*` chunk group is written from the transcribed
/// constants above (JSUT 22 kHz single-speaker recipe defaults); the
/// provenance stamps mark the weight as
/// [`LicenseClass::RedistributionForbidden`] by default — the JSUT /
/// JVS corpus terms explicitly forbid re-distribution of the trained
/// weight, so a converter cannot silently claim otherwise. A user who
/// trained on a permissive corpus overrides via
/// `vokra-convert --license <spdx>` at the outer boundary
/// (`crates/vokra-convert/src/lib.rs`).
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, VitsJaReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    write_hparams(&mut b);
    // Self-describing redistribution stamp: the JSUT / JVS / COEIROINK
    // corpora explicitly forbid trained-weight redistribution; a
    // converter cannot silently label the artifact permissive. A user
    // who trained on a permissive corpus overrides at the
    // `convert_file --license <spdx>` boundary.
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::RedistributionForbidden,
        "corpus-restricted",
        Some(NAME),
        Some(
            "ESPnet-family plain VITS JA — architecture Apache-2.0 (ESPnet) + MIT \
             (jaywalnut310/vits); trained weight typically bound by JSUT / JVS / COEIROINK \
             corpus terms that forbid re-distribution. Override with --license <spdx> at \
             conversion time if trained on a permissive corpus.",
        ),
    );

    let mut report = VitsJaReport::default();
    for t in st.tensors() {
        match t.dtype {
            // BF16 pass-through added 2026-07-25 (mirror of qwen3-tts +
            // vibevoice + voxcpm2 + moshi + voxtral): downstream ESPnet
            // VITS re-trainings that ship BF16 now hit this arm. Emit as
            // GGUF type 30 verbatim; runtime widens on load via
            // `decode_bf16` (exact, `bits << 16`).
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )?;
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
    if report.written == 0 {
        report.notes.push(
            "no float tensors passed through — this GGUF is metadata-only and the runtime will \
             refuse to bind any weights (FR-EX-08). ESPnet-family plain VITS JA checkpoints ship \
             F32 (default) or F16 (mixed-precision); the BF16 pass-through path is now wired \
             (2026-07-25), so this state is only reachable when the release contains no F32 / \
             F16 / BF16 float tensors at all."
                .into(),
        );
    }
    // Always emit the redistribution note so an operator reading the
    // conversion output cannot miss the default stamp's meaning.
    report.notes.push(
        "provenance defaults to `RedistributionForbidden` — JSUT / JVS / COEIROINK corpus terms \
         forbid trained-weight redistribution. Override with `vokra-convert --license <spdx>` if \
         you trained this VITS on a permissive corpus."
            .into(),
    );
    Ok((b, report))
}

/// Writes the `vokra.vits_ja.*` chunk group from the transcribed
/// constants above (primary sources: `train_vits.yaml` +
/// `vits.py::AVAILABLE_GENERATERS.vits_generator`).
fn write_hparams(b: &mut GgufBuilder) {
    // Top-level.
    b.add_string(KEY_MODEL_FAMILY, MODEL_FAMILY);
    b.add_u32(KEY_SAMPLE_RATE_HZ, SAMPLE_RATE_HZ);
    b.add_u32(KEY_VOCAB_SIZE, VOCAB_SIZE);
    b.add_u32(KEY_N_MELS, N_MELS);
    b.add_u32(KEY_AUX_CHANNELS, AUX_CHANNELS);
    b.add_u32(KEY_HIDDEN_CHANNELS, HIDDEN_CHANNELS);
    b.add_u32(KEY_SEGMENT_SIZE, SEGMENT_SIZE);
    b.add_u32(KEY_SPKS, SPKS);

    // Text encoder.
    b.add_u32(KEY_TEXT_N_LAYER, TEXT_N_LAYER);
    b.add_u32(KEY_TEXT_N_HEAD, TEXT_N_HEAD);
    b.add_u32(KEY_TEXT_FFN_EXPAND, TEXT_FFN_EXPAND);
    b.add_u32(
        KEY_TEXT_POSITIONWISE_CONV_KERNEL,
        TEXT_POSITIONWISE_CONV_KERNEL,
    );
    b.add_f32(KEY_TEXT_DROPOUT_RATE, TEXT_DROPOUT_RATE);
    b.add_f32(
        KEY_TEXT_POSITIONAL_DROPOUT_RATE,
        TEXT_POSITIONAL_DROPOUT_RATE,
    );
    b.add_f32(KEY_TEXT_ATTENTION_DROPOUT_RATE, TEXT_ATTENTION_DROPOUT_RATE);
    b.add_bool(KEY_TEXT_USE_MACARON_STYLE, TEXT_USE_MACARON_STYLE);
    b.add_bool(KEY_TEXT_USE_CONFORMER_CONV, TEXT_USE_CONFORMER_CONV);

    // Flow.
    b.add_u32(KEY_FLOW_N_FLOW, FLOW_N_FLOW);
    b.add_u32(KEY_FLOW_KERNEL_SIZE, FLOW_KERNEL_SIZE);
    b.add_u32(KEY_FLOW_BASE_DILATION, FLOW_BASE_DILATION);
    b.add_u32(KEY_FLOW_N_LAYER, FLOW_N_LAYER);
    b.add_f32(KEY_FLOW_DROPOUT_RATE, FLOW_DROPOUT_RATE);
    b.add_bool(KEY_FLOW_USE_ONLY_MEAN, FLOW_USE_ONLY_MEAN);

    // Stochastic duration predictor.
    b.add_u32(KEY_SDP_KERNEL_SIZE, SDP_KERNEL_SIZE);
    b.add_f32(KEY_SDP_DROPOUT_RATE, SDP_DROPOUT_RATE);
    b.add_u32(KEY_SDP_N_FLOW, SDP_N_FLOW);
    b.add_u32(KEY_SDP_DDS_CONV_LAYERS, SDP_DDS_CONV_LAYERS);

    // HiFi-GAN decoder.
    b.add_u32(KEY_DECODER_KERNEL_SIZE, DECODER_KERNEL_SIZE);
    b.add_u32(KEY_DECODER_INITIAL_CHANNEL, DECODER_INITIAL_CHANNEL);
    add_u32_array(b, KEY_DECODER_UPSAMPLE_SCALES, &DECODER_UPSAMPLE_SCALES);
    add_u32_array(
        b,
        KEY_DECODER_UPSAMPLE_KERNEL_SIZES,
        &DECODER_UPSAMPLE_KERNEL_SIZES,
    );
    add_u32_array(
        b,
        KEY_DECODER_RESBLOCK_KERNEL_SIZES,
        &DECODER_RESBLOCK_KERNEL_SIZES,
    );
    add_u32_array(
        b,
        KEY_DECODER_RESBLOCK_DILATIONS_FLAT,
        &DECODER_RESBLOCK_DILATIONS_FLAT,
    );
    b.add_u32(
        KEY_DECODER_RESBLOCK_DILATIONS_STRIDE,
        DECODER_RESBLOCK_DILATIONS_STRIDE,
    );
    b.add_bool(KEY_DECODER_USE_WEIGHT_NORM, DECODER_USE_WEIGHT_NORM);
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    fn minimal_safetensors_one_f32() -> Vec<u8> {
        // Single f32 tensor so the pass-through arm fires once and the
        // report counts a non-zero write. The tensor name mirrors the
        // canonical normalized ESPnet VITS generator name.
        let header =
            r#"{"text_encoder.emb.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 24]);
        out
    }

    fn minimal_safetensors_no_tensors() -> Vec<u8> {
        let header = r#"{}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out
    }

    fn minimal_safetensors_one_f16() -> Vec<u8> {
        let header =
            r#"{"text_encoder.emb.weight":{"dtype":"F16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    fn minimal_safetensors_one_bf16() -> Vec<u8> {
        let header =
            r#"{"text_encoder.emb.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    fn get_u32(file: &GgufFile, key: &str) -> u32 {
        match file.get(key) {
            Some(GgufMetadataValue::U32(v)) => *v,
            other => panic!("{key}: unexpected {other:?}"),
        }
    }

    fn get_f32(file: &GgufFile, key: &str) -> f32 {
        match file.get(key) {
            Some(GgufMetadataValue::F32(v)) => *v,
            other => panic!("{key}: unexpected {other:?}"),
        }
    }

    fn get_bool(file: &GgufFile, key: &str) -> bool {
        match file.get(key) {
            Some(GgufMetadataValue::Bool(v)) => *v,
            other => panic!("{key}: unexpected {other:?}"),
        }
    }

    fn get_string(file: &GgufFile, key: &str) -> String {
        file.get(key)
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| panic!("{key}: missing"))
            .to_owned()
    }

    fn get_u32_array(file: &GgufFile, key: &str) -> Vec<u32> {
        let arr = file
            .get(key)
            .and_then(|v| v.as_array())
            .unwrap_or_else(|| panic!("{key}: not an array"));
        arr.values
            .iter()
            .map(|v| match v {
                GgufMetadataValue::U32(x) => *x,
                other => panic!("{key}: array elem not u32 ({other:?})"),
            })
            .collect()
    }

    // ---- Arch tag distinctness ------------------------------------------

    #[test]
    fn arch_string_matches_runtime_constant() {
        // The two crates only share `vokra-core`, so this constant is
        // the sole handshake with `vokra-models::vits_ja::EXPECTED_ARCH`.
        assert_eq!(ARCH, "vits-ja");
    }

    #[test]
    fn arch_is_distinct_from_piper_plus_and_siblings() {
        // plain VITS decodes through a HiFi-GAN generator directly;
        // piper-plus decodes through a sub-band iSTFT + PQMF post-net.
        // Silently sharing an arch tag would misroute the runtime dispatch.
        assert_ne!(ARCH, "piper-plus-mb-istft-vits2");
        // Distinct from every neighbouring TTS arch tag.
        assert_ne!(ARCH, "irodori-tts");
        assert_ne!(ARCH, "vibevoice");
        assert_ne!(ARCH, "voxcpm2");
        assert_ne!(ARCH, "cosyvoice2");
        assert_ne!(ARCH, "cosyvoice3");
        assert_ne!(ARCH, "qwen3_tts");
        assert_ne!(ARCH, "chatterbox");
        assert_ne!(ARCH, "chatterbox_turbo");
        assert_ne!(ARCH, "chatterbox_nano");
        assert_ne!(ARCH, "dia");
        assert_ne!(ARCH, "zonos");
        assert_ne!(ARCH, "csm");
    }

    #[test]
    fn name_string_matches_default_recipe_id() {
        assert_eq!(NAME, "espnet-jsut-vits-22khz");
    }

    /// Every transcribed constant must equal the primary-source value.
    /// Changing any of these silently mis-shapes the text encoder / SDP
    /// / flow / HiFi-GAN decoder.
    #[test]
    fn transcribed_constants_match_primary_source() {
        // Top-level.
        assert_eq!(MODEL_FAMILY, "vits-ja");
        assert_eq!(SAMPLE_RATE_HZ, 22_050);
        assert_eq!(N_MELS, 80);
        assert_eq!(AUX_CHANNELS, 513); // n_fft=1024 → 513
        assert_eq!(HIDDEN_CHANNELS, 192);
        assert_eq!(SEGMENT_SIZE, 32);

        // Text encoder — train_vits.yaml.
        assert_eq!(TEXT_N_LAYER, 6);
        assert_eq!(TEXT_N_HEAD, 2);
        assert_eq!(TEXT_FFN_EXPAND, 4);
        assert_eq!(TEXT_POSITIONWISE_CONV_KERNEL, 3);
        assert!((TEXT_DROPOUT_RATE - 0.1).abs() < 1e-6);
        assert!((TEXT_POSITIONAL_DROPOUT_RATE - 0.0).abs() < 1e-9);
        assert!((TEXT_ATTENTION_DROPOUT_RATE - 0.1).abs() < 1e-6);
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(TEXT_USE_MACARON_STYLE);
            assert!(!TEXT_USE_CONFORMER_CONV);
        }

        // Flow.
        assert_eq!(FLOW_N_FLOW, 4);
        assert_eq!(FLOW_KERNEL_SIZE, 5);
        assert_eq!(FLOW_BASE_DILATION, 1);
        assert_eq!(FLOW_N_LAYER, 4);
        assert!((FLOW_DROPOUT_RATE - 0.0).abs() < 1e-9);
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(FLOW_USE_ONLY_MEAN);
        }

        // Stochastic duration predictor.
        assert_eq!(SDP_KERNEL_SIZE, 3);
        assert!((SDP_DROPOUT_RATE - 0.5).abs() < 1e-6);
        assert_eq!(SDP_N_FLOW, 4);
        assert_eq!(SDP_DDS_CONV_LAYERS, 3);

        // HiFi-GAN decoder — train_vits.yaml.
        assert_eq!(DECODER_KERNEL_SIZE, 7);
        assert_eq!(DECODER_INITIAL_CHANNEL, 512);
        assert_eq!(DECODER_UPSAMPLE_SCALES, [8, 8, 2, 2]);
        assert_eq!(DECODER_UPSAMPLE_KERNEL_SIZES, [16, 16, 4, 4]);
        assert_eq!(DECODER_RESBLOCK_KERNEL_SIZES, [3, 7, 11]);
        assert_eq!(DECODER_RESBLOCK_DILATIONS_FLAT, [1, 3, 5, 1, 3, 5, 1, 3, 5]);
        assert_eq!(DECODER_RESBLOCK_DILATIONS_STRIDE, 3);
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(DECODER_USE_WEIGHT_NORM);
        }

        // Compile-time algebra: head split divides evenly, RoPE-even
        // head_dim, upsample product = hop_length for the 22 kHz recipe.
        const _: () = {
            assert!(HIDDEN_CHANNELS % TEXT_N_HEAD == 0);
            assert!((HIDDEN_CHANNELS / TEXT_N_HEAD) % 2 == 0);
            // 8 * 8 * 2 * 2 = 256 == hop_length for 22 kHz JA recipe.
            assert!(
                DECODER_UPSAMPLE_SCALES[0]
                    * DECODER_UPSAMPLE_SCALES[1]
                    * DECODER_UPSAMPLE_SCALES[2]
                    * DECODER_UPSAMPLE_SCALES[3]
                    == 256
            );
            // resblock kernel count matches the flattened dilation stride.
            assert!(
                DECODER_RESBLOCK_KERNEL_SIZES.len() as u32
                    == (DECODER_RESBLOCK_DILATIONS_FLAT.len() as u32
                        / DECODER_RESBLOCK_DILATIONS_STRIDE)
            );
            // n_mels + aux_channels + sample_rate positive.
            assert!(N_MELS > 0);
            assert!(AUX_CHANNELS > 0);
            assert!(SAMPLE_RATE_HZ > 0);
        };
    }

    #[test]
    fn round_trip_carries_arch_chunks_and_provenance() {
        let (builder, report) = convert(minimal_safetensors_one_f32()).expect("convert");
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        // The redistribution note must always fire.
        assert!(
            report
                .notes
                .iter()
                .any(|n| n.contains("RedistributionForbidden")),
            "report must always carry the redistribution note: {:?}",
            report.notes,
        );

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH),
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME),
        );
        assert_eq!(get_string(&file, KEY_MODEL_FAMILY), MODEL_FAMILY);
        assert_eq!(get_u32(&file, KEY_SAMPLE_RATE_HZ), SAMPLE_RATE_HZ);
        assert_eq!(get_u32(&file, KEY_VOCAB_SIZE), VOCAB_SIZE);
        assert_eq!(get_u32(&file, KEY_N_MELS), N_MELS);
        assert_eq!(get_u32(&file, KEY_AUX_CHANNELS), AUX_CHANNELS);
        assert_eq!(get_u32(&file, KEY_HIDDEN_CHANNELS), HIDDEN_CHANNELS);
        assert_eq!(get_u32(&file, KEY_SEGMENT_SIZE), SEGMENT_SIZE);

        // Every transcribed U32 hparam round-trips verbatim.
        for (key, want) in [
            (KEY_TEXT_N_LAYER, TEXT_N_LAYER),
            (KEY_TEXT_N_HEAD, TEXT_N_HEAD),
            (KEY_TEXT_FFN_EXPAND, TEXT_FFN_EXPAND),
            (
                KEY_TEXT_POSITIONWISE_CONV_KERNEL,
                TEXT_POSITIONWISE_CONV_KERNEL,
            ),
            (KEY_FLOW_N_FLOW, FLOW_N_FLOW),
            (KEY_FLOW_KERNEL_SIZE, FLOW_KERNEL_SIZE),
            (KEY_FLOW_BASE_DILATION, FLOW_BASE_DILATION),
            (KEY_FLOW_N_LAYER, FLOW_N_LAYER),
            (KEY_SDP_KERNEL_SIZE, SDP_KERNEL_SIZE),
            (KEY_SDP_N_FLOW, SDP_N_FLOW),
            (KEY_SDP_DDS_CONV_LAYERS, SDP_DDS_CONV_LAYERS),
            (KEY_DECODER_KERNEL_SIZE, DECODER_KERNEL_SIZE),
            (KEY_DECODER_INITIAL_CHANNEL, DECODER_INITIAL_CHANNEL),
            (
                KEY_DECODER_RESBLOCK_DILATIONS_STRIDE,
                DECODER_RESBLOCK_DILATIONS_STRIDE,
            ),
        ] {
            assert_eq!(get_u32(&file, key), want, "{key}");
        }

        // F32 constants round-trip.
        assert!((get_f32(&file, KEY_TEXT_DROPOUT_RATE) - TEXT_DROPOUT_RATE).abs() < 1e-6);
        assert!(
            (get_f32(&file, KEY_TEXT_POSITIONAL_DROPOUT_RATE) - TEXT_POSITIONAL_DROPOUT_RATE).abs()
                < 1e-9
        );
        assert!(
            (get_f32(&file, KEY_TEXT_ATTENTION_DROPOUT_RATE) - TEXT_ATTENTION_DROPOUT_RATE).abs()
                < 1e-6
        );
        assert!((get_f32(&file, KEY_FLOW_DROPOUT_RATE) - FLOW_DROPOUT_RATE).abs() < 1e-9);
        assert!((get_f32(&file, KEY_SDP_DROPOUT_RATE) - SDP_DROPOUT_RATE).abs() < 1e-6);

        // Bool constants round-trip.
        assert_eq!(
            get_bool(&file, KEY_TEXT_USE_MACARON_STYLE),
            TEXT_USE_MACARON_STYLE
        );
        assert_eq!(
            get_bool(&file, KEY_TEXT_USE_CONFORMER_CONV),
            TEXT_USE_CONFORMER_CONV
        );
        assert_eq!(get_bool(&file, KEY_FLOW_USE_ONLY_MEAN), FLOW_USE_ONLY_MEAN);
        assert_eq!(
            get_bool(&file, KEY_DECODER_USE_WEIGHT_NORM),
            DECODER_USE_WEIGHT_NORM
        );

        // Decoder array chunks round-trip.
        assert_eq!(
            get_u32_array(&file, KEY_DECODER_UPSAMPLE_SCALES),
            DECODER_UPSAMPLE_SCALES.to_vec(),
        );
        assert_eq!(
            get_u32_array(&file, KEY_DECODER_UPSAMPLE_KERNEL_SIZES),
            DECODER_UPSAMPLE_KERNEL_SIZES.to_vec(),
        );
        assert_eq!(
            get_u32_array(&file, KEY_DECODER_RESBLOCK_KERNEL_SIZES),
            DECODER_RESBLOCK_KERNEL_SIZES.to_vec(),
        );
        assert_eq!(
            get_u32_array(&file, KEY_DECODER_RESBLOCK_DILATIONS_FLAT),
            DECODER_RESBLOCK_DILATIONS_FLAT.to_vec(),
        );

        // Provenance is stamped `RedistributionForbidden` by default —
        // NEVER Permissive, even for a stock ESPnet checkpoint.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some("redistribution-forbidden"),
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("corpus-restricted"),
        );
    }

    #[test]
    fn f16_tensor_passes_through() {
        // Pins the F16 leg of the `GgmlType::F32 | GgmlType::F16` union arm.
        let (_builder, report) = convert(minimal_safetensors_one_f16()).expect("convert");
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
    }

    /// Pins the BF16 leg of the `GgmlType::F32 | GgmlType::F16 |
    /// GgmlType::BF16` union: BF16 must reach the pass-through arm,
    /// emit as GGUF type 30 verbatim, and increment `bf16_passthrough`.
    /// Mirror of vibevoice / voxcpm2 / qwen3-tts /
    /// `bf16_tensor_passes_through_verbatim` and moshi's `assert_eq!(
    /// info.dtype, GgmlType::BF16, "no convert-time widening")`.
    ///
    /// Rewritten 2026-07-25 from the earlier "counted as skipped" pin —
    /// the earlier pin encoded the pre-BF16-fix scaffold posture.
    /// Removing the pin outright would let a latent silent-widen slip in
    /// undetected; rewriting to the passes-through invariant keeps the
    /// regression guard.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (builder, report) = convert(minimal_safetensors_one_bf16()).expect("convert");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm and increment `written`"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 subset counter must record the pass-through"
        );
        // The tensor survives the round trip under its upstream name and
        // preserves its BF16 dtype (no convert-time widening — runtime
        // widens on load via `decode_bf16`).
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file
            .tensor_info("text_encoder.emb.weight")
            .expect("BF16 tensor must be present after pass-through");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — GGUF dtype must remain BF16"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info).len(),
            12,
            "BF16 payload = 6 elements × 2 bytes = 12 bytes"
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
    }

    #[test]
    fn empty_safetensors_emits_loud_note() {
        // No tensors → the "no float tensors" loud note fires; the
        // hparam chunk group still round-trips (so a caller inspecting
        // the metadata-only GGUF sees the release axes).
        let (builder, report) = convert(minimal_safetensors_no_tensors()).expect("convert");
        assert_eq!(report.written, 0);
        assert_eq!(report.skipped_non_float, 0);
        assert!(
            report.notes.iter().any(|n| n.contains("no float tensors")),
            "must fire the no-float-tensors note: {:?}",
            report.notes
        );
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        assert_eq!(
            get_u32(&file, KEY_DECODER_INITIAL_CHANNEL),
            DECODER_INITIAL_CHANNEL
        );
    }

    /// The M2-13 compliance registry must resolve every canonical
    /// vits-ja id to `RedistributionForbidden`. Cross-check that the
    /// converter's default stamp agrees with the registry.
    #[test]
    fn provenance_default_class_matches_registry_default() {
        use vokra_core::compliance::registry_lookup;
        assert_eq!(
            registry_lookup(ARCH),
            Some(LicenseClass::RedistributionForbidden),
            "registry must map `{ARCH}` to RedistributionForbidden"
        );
    }
}
