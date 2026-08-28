//! **ECAPA-TDNN** (SpeechBrain speaker encoder): safetensors checkpoint →
//! GGUF conversion (SoTA plan, 2026-07-25).
//!
//! Input: the upstream `speechbrain/spkrec-ecapa-voxceleb` release — an
//! ECAPA-TDNN 192-dim speaker embedding model trained on VoxCeleb 1+2.
//! Output: a GGUF carrying every float tensor plus `vokra.model.*` and
//! `vokra.provenance.*` metadata identifying the model as a `speaker`
//! category weight with an `apache-2.0` licence.
//!
//! # Provenance
//!
//! - **HF path**: `speechbrain/spkrec-ecapa-voxceleb`.
//! - **License (SPDX)**: `apache-2.0` — end-to-end (SpeechBrain code +
//!   trained weight; see `docs/license-audit.md §3.1` sign-off queue).
//! - **Category**: `speaker` (speaker encoder / embedding extractor —
//!   fbank-80 → 192-d embedding, alternate realisation of the same
//!   functional surface as `campplus.rs`). Category tag is written under
//!   the raw `vokra.model.category` key so the model-card tooling can
//!   classify without reaching into per-converter constants.
//!
//! # BF16 pass-through (mirror of qwen3_tts / vibevoice / voxcpm2)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm — no
//! convert-time widening. BF16 stays GGUF type 30 (`GgmlType::BF16`); the
//! runtime widens BF16 → f32 losslessly at load via the single choke
//! point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is
//! the top 16 bits of an f32 — `bits << 16` is exact). The observability
//! counter [`EcapaTdnnReport::bf16_passthrough`] records how many BF16
//! tensors landed on this arm so a silent widen / downcast cannot slip
//! in undetected.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VibeVoice /
//! VoxCPM contract), after validating the complete official 200-tensor
//! manifest. Real-weight parity is pinned by
//! `crates/vokra-models/tests/parity_ecapa_tdnn_real.rs`.
//!
//! # No ONNX (permanent)
//!
//! SpeechBrain ships PyTorch checkpoints (safetensors); this converter
//! **never** touches ONNX (FR-LD-05).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for ECAPA-TDNN GGUFs — intentionally **distinct**
/// from `campplus` because ECAPA-TDNN and CAM++ share a functional
/// surface (fbank-80 → 192-d embedding) but NOT their tensor topology
/// (ECAPA-TDNN uses SE-Res2Blocks + attentive stat pooling; CAM++ uses
/// D-TDNN with context-aware masking). Silently sharing an arch tag
/// would mis-route runtime dispatch.
pub const ARCH: &str = "ecapa_tdnn";

/// `vokra.model.name` value written for the canonical
/// `speechbrain/spkrec-ecapa-voxceleb` GGUF.
pub const NAME: &str = "spkrec-ecapa-voxceleb";

/// `vokra.model.category` value written for every ECAPA-TDNN GGUF.
pub const CATEGORY: &str = "speaker";

/// `vokra.provenance.upstream_hf` value — the primary redistribution
/// source used by the model-card generator.
pub const UPSTREAM_HF: &str = "speechbrain/spkrec-ecapa-voxceleb";

/// Pinned upstream revision used by the independent parity oracle.
pub const UPSTREAM_REVISION: &str = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286";

/// Default upstream weight licence (SPDX).
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants (the cross-crate constant duplication rule
// the sibling converters use applies).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_PROVENANCE_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_SAMPLE_RATE: &str = "vokra.ecapa.sample_rate";
const KEY_N_MELS: &str = "vokra.ecapa.n_mels";
const KEY_N_FFT: &str = "vokra.ecapa.n_fft";
const KEY_WIN_LENGTH: &str = "vokra.ecapa.win_length";
const KEY_HOP_LENGTH: &str = "vokra.ecapa.hop_length";
const KEY_EMBED_DIM: &str = "vokra.ecapa.embed_dim";
const KEY_TDNN_CHANNELS: &str = "vokra.ecapa.tdnn_channels";
const KEY_MFA_CHANNELS: &str = "vokra.ecapa.mfa_channels";
const KEY_ATTENTION_CHANNELS: &str = "vokra.ecapa.attention_channels";
const KEY_RES2NET_SCALE: &str = "vokra.ecapa.res2net_scale";
const KEY_BN_EPS: &str = "vokra.ecapa.bn_eps";
const KEY_STATS_EPS: &str = "vokra.ecapa.stats_eps";
const KEY_FRONTEND: &str = "vokra.ecapa.frontend";
const KEY_PADDING: &str = "vokra.ecapa.padding";
const KEY_LAYOUT: &str = "vokra.ecapa.artifact_layout";

const INPUT_DIM: u64 = 80;
const TDNN_CHANNELS: u64 = 1_024;
const RES2NET_SCALE: usize = 8;
const RES2NET_CHANNELS: u64 = TDNN_CHANNELS / RES2NET_SCALE as u64;
const MFA_CHANNELS: u64 = 3_072;
const ATTENTION_CHANNELS: u64 = 128;
const STATS_CHANNELS: u64 = MFA_CHANNELS * 2;
const EMBED_DIM: u64 = 192;
const TENSOR_COUNT: usize = 200;

/// Outcome of an ECAPA-TDNN conversion.
///
/// Mirrors the sibling converters' counter shape
/// (`super::qwen3_tts::Qwen3TtsReport`, `super::vibevoice::VibeVoiceReport`,
/// `super::voxcpm2::VoxCpm2Report`) adapted to the file-oriented
/// `convert_ecapa_tdnn_file` surface (adds `read` tracking every tensor
/// the safetensors reader surfaced so the invariant
/// `read == written + skipped_non_float` is auditable).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EcapaTdnnReport {
    /// Total tensors surfaced by the safetensors reader (before any
    /// dispatch to the pass-through / skipped arm).
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16 at parse time, so a non-zero
    /// here would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm (subset of
    /// [`Self::written`]). Additive observability counter — a latent
    /// silent widen / downcast cannot slip in undetected without this
    /// counter also drifting.
    pub bf16_passthrough: usize,
}

/// Converts a `speechbrain/spkrec-ecapa-voxceleb` safetensors checkpoint
/// at `input` into a Vokra-native GGUF at `output`, returning an
/// [`EcapaTdnnReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream name;
/// the `vokra.model.*` (arch / name / category) and `vokra.provenance.*`
/// (weight_license / license / model_id / source / upstream_hf) chunks
/// are stamped for the runtime compliance gate (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string; the [`LicenseClass`] is re-derived via
/// [`LicenseClass::from_license_str`]). The default is
/// `DEFAULT_LICENSE_SPDX` (`"apache-2.0"`, `Permissive`) — the upstream
/// HF release ships apache-2.0 end-to-end.
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure; [`ConvertError::Parse`]
/// on a malformed safetensors input.
pub fn convert_ecapa_tdnn_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<EcapaTdnnReport, ConvertError> {
    // Load the whole checkpoint into memory: the ECAPA-TDNN release is
    // ~83 MiB (192-d embedding backbone) — comfortably below the
    // smaller than the streaming-mandated Moshi 14 GiB tier, so the
    // simple `std::fs::read` posture the sibling non-streaming
    // converters (qwen3_tts / vibevoice / voxcpm2) use applies.
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;
    validate_manifest(&st)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // Default provenance stamp — Permissive apache-2.0 end-to-end
    // (upstream `speechbrain/spkrec-ecapa-voxceleb` model card + repo
    // LICENSE). The optional `license` argument overrides below.
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        DEFAULT_LICENSE_SPDX,
        Some(NAME),
        Some("speechbrain/spkrec-ecapa-voxceleb (apache-2.0 end-to-end)"),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    b.add_string(KEY_PROVENANCE_UPSTREAM_REVISION, UPSTREAM_REVISION);
    b.add_u32(KEY_SAMPLE_RATE, 16_000);
    b.add_u32(KEY_N_MELS, INPUT_DIM as u32);
    b.add_u32(KEY_N_FFT, 400);
    b.add_u32(KEY_WIN_LENGTH, 400);
    b.add_u32(KEY_HOP_LENGTH, 160);
    b.add_u32(KEY_EMBED_DIM, EMBED_DIM as u32);
    b.add_u32(KEY_TDNN_CHANNELS, TDNN_CHANNELS as u32);
    b.add_u32(KEY_MFA_CHANNELS, MFA_CHANNELS as u32);
    b.add_u32(KEY_ATTENTION_CHANNELS, ATTENTION_CHANNELS as u32);
    b.add_u32(KEY_RES2NET_SCALE, RES2NET_SCALE as u32);
    b.add_f32(KEY_BN_EPS, 1.0e-5);
    b.add_f32(KEY_STATS_EPS, 1.0e-12);
    b.add_string(KEY_FRONTEND, "speechbrain-fbank-v1");
    b.add_string(KEY_PADDING, "reflect-same");
    b.add_string(KEY_LAYOUT, "speechbrain-ecapa-200-v1");

    let mut report = EcapaTdnnReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (mirror of qwen3_tts / vibevoice / voxcpm2 / moshi); the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
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

    // Optional weight-license override — mirrors the outer
    // `convert_file_licensed` (lib.rs) branch so both a Vokra-CLI caller
    // and a direct `convert_ecapa_tdnn_file` caller land the same
    // provenance surface for the same SPDX string. Restates the source
    // neutrally so it does not contradict the stamped default's
    // parenthetical.
    if let Some(lic) = license {
        let class = LicenseClass::from_license_str(lic);
        b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, class.as_str());
        b.add_string(chunks::KEY_PROVENANCE_LICENSE, lic);
        b.add_string(
            chunks::KEY_PROVENANCE_SOURCE,
            &format!("{UPSTREAM_HF} (licence {lic} per source)"),
        );
    }

    // Serialize and land the emitted GGUF at `output`. `to_bytes()`
    // stamps `vokra.schema.version` + `vokra.schema.producer` on its
    // own via the writer's built-in schema stamper — no per-converter
    // duplication needed.
    let out_bytes = b.to_bytes()?;
    std::fs::write(output, &out_bytes)?;
    Ok(report)
}

fn validate_manifest(st: &SafetensorsFile) -> Result<(), ConvertError> {
    if st.tensors().len() != TENSOR_COUNT {
        return Err(ConvertError::Parse(format!(
            "ecapa_tdnn: unsupported tensor manifest: count={}, expected exactly {TENSOR_COUNT}",
            st.tensors().len()
        )));
    }
    let expected = expected_manifest();
    debug_assert_eq!(expected.len(), TENSOR_COUNT);
    for (name, shape) in expected {
        check_shape(st, &name, &shape)?;
    }
    for tensor in st.tensors() {
        if !matches!(tensor.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
            return Err(ConvertError::Parse(format!(
                "ecapa_tdnn: tensor `{}` uses unsupported dtype {:?}; every manifest tensor must be F32, F16, or BF16",
                tensor.name, tensor.dtype
            )));
        }
    }
    Ok(())
}

fn expected_manifest() -> Vec<(String, Vec<u64>)> {
    let mut expected = Vec::with_capacity(TENSOR_COUNT);
    push_tdnn(&mut expected, "blocks.0", INPUT_DIM, TDNN_CHANNELS, 5);
    for block in 1..=3 {
        let prefix = format!("blocks.{block}");
        push_tdnn(
            &mut expected,
            &format!("{prefix}.tdnn1"),
            TDNN_CHANNELS,
            TDNN_CHANNELS,
            1,
        );
        for inner in 0..RES2NET_SCALE - 1 {
            push_tdnn(
                &mut expected,
                &format!("{prefix}.res2net_block.blocks.{inner}"),
                RES2NET_CHANNELS,
                RES2NET_CHANNELS,
                3,
            );
        }
        push_tdnn(
            &mut expected,
            &format!("{prefix}.tdnn2"),
            TDNN_CHANNELS,
            TDNN_CHANNELS,
            1,
        );
        push_conv(
            &mut expected,
            &format!("{prefix}.se_block.conv1.conv"),
            TDNN_CHANNELS,
            ATTENTION_CHANNELS,
            1,
        );
        push_conv(
            &mut expected,
            &format!("{prefix}.se_block.conv2.conv"),
            ATTENTION_CHANNELS,
            TDNN_CHANNELS,
            1,
        );
    }
    push_tdnn(&mut expected, "mfa", MFA_CHANNELS, MFA_CHANNELS, 1);
    push_tdnn(
        &mut expected,
        "asp.tdnn",
        MFA_CHANNELS * 3,
        ATTENTION_CHANNELS,
        1,
    );
    push_conv(
        &mut expected,
        "asp.conv.conv",
        ATTENTION_CHANNELS,
        MFA_CHANNELS,
        1,
    );
    push_norm(&mut expected, "asp_bn.norm", STATS_CHANNELS);
    push_conv(&mut expected, "fc.conv", STATS_CHANNELS, EMBED_DIM, 1);
    expected
}

fn push_tdnn(
    expected: &mut Vec<(String, Vec<u64>)>,
    prefix: &str,
    input_channels: u64,
    output_channels: u64,
    kernel: u64,
) {
    push_conv(
        expected,
        &format!("{prefix}.conv.conv"),
        input_channels,
        output_channels,
        kernel,
    );
    push_norm(expected, &format!("{prefix}.norm.norm"), output_channels);
}

fn push_conv(
    expected: &mut Vec<(String, Vec<u64>)>,
    prefix: &str,
    input_channels: u64,
    output_channels: u64,
    kernel: u64,
) {
    expected.push((
        format!("{prefix}.weight"),
        vec![output_channels, input_channels, kernel],
    ));
    expected.push((format!("{prefix}.bias"), vec![output_channels]));
}

fn push_norm(expected: &mut Vec<(String, Vec<u64>)>, prefix: &str, channels: u64) {
    for suffix in ["weight", "bias", "running_mean", "running_var"] {
        expected.push((format!("{prefix}.{suffix}"), vec![channels]));
    }
}

fn check_shape(st: &SafetensorsFile, name: &str, expected: &[u64]) -> Result<(), ConvertError> {
    let tensor = st.tensor_info(name).ok_or_else(|| {
        ConvertError::Parse(format!("ecapa_tdnn: required tensor `{name}` is missing"))
    })?;
    if tensor.shape != expected {
        return Err(ConvertError::Parse(format!(
            "ecapa_tdnn: tensor `{name}` has shape {:?}, expected {expected:?}",
            tensor.shape
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_is_exactly_the_canonical_200_tensor_checkpoint() {
        let manifest = expected_manifest();
        assert_eq!(manifest.len(), TENSOR_COUNT);
        let mut names = manifest.iter().map(|(name, _)| name).collect::<Vec<_>>();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), TENSOR_COUNT);
        assert!(manifest.iter().any(|(name, shape)| {
            name == "asp.tdnn.conv.conv.weight" && shape == &[128, 9_216, 1]
        }));
    }

    #[test]
    fn partial_checkpoint_is_rejected_before_any_output_is_written() {
        let payload = [0u8; 12];
        let header =
            r#"{"blocks.0.conv.conv.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&payload);
        let parsed = SafetensorsFile::parse(bytes).unwrap();
        let error = validate_manifest(&parsed).unwrap_err();
        assert!(error.to_string().contains("expected exactly 200"));
    }
}
