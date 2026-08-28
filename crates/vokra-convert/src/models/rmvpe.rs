//! **RMVPE** (Robust Model for Vocal Pitch Estimation): safetensors →
//! GGUF conversion (F0 pitch-extractor tier, 2026-07-30).
//!
//! Input: an offline `.pt` → safetensors flattening (via
//! `tools/parity/nemo_pt_to_safetensors.py`) of the upstream
//! fixed `yxlllc/RMVPE` release. Output: a GGUF carrying the exact 623
//! inference tensors plus optional BatchNorm counters and the
//! `vokra.rmvpe.*` metadata read by the native runtime. The unused
//! `unet.tf.*` module constructed by `DeepUnet0` is validated and omitted.
//!
//! # Model class
//!
//! RMVPE (Wei et al. 2023) is a CNN + GRU U-Net polyphonic vocal pitch
//! extractor:
//!
//! ```text
//! PCM (16 kHz mono)
//!   -> HTK magnitude-mel (n_mels=128, hop=160, win=n_fft=1024)
//!   -> U-Net encoder (5 down blocks: residual Conv2d + BN + ReLU, then
//!      AvgPool2d)
//!   -> four 512-channel intermediate residual layers
//!   -> U-Net decoder (5 up blocks: ConvTranspose2d + skip-concat + residual
//!      Conv2d + BN + ReLU)
//!   -> 3-channel Conv2d + bidirectional GRU(hidden=256)
//!   -> 360-pitch-class Linear → Sigmoid
//!      over a 20-cents/class grid anchored at exactly 31.7 Hz)
//! ```
//!
//! This is the pitch front-end **required by RVC v2** and is commonly
//! reused by other singing-voice / voice-conversion (GPT-SoVITS,
//! Retrieval-based VC) pipelines. It shares the "per-hop F0 track"
//! output contract with the CREPE / FCPE / PyIN / Harvest siblings in
//! `vokra-models::f0`.
//!
//! # License
//!
//! `Dream-High/RMVPE` is Apache-2.0, but it is not a GitHub fork relationship.
//! The exact `yxlllc/RMVPE` repository and checkpoint carry no license grant
//! (GitHub API rechecked 2026-08-26). Code and weight therefore default to
//! [`LicenseClass::Unknown`] and stay fail-closed. A caller may pass an SPDX
//! override only after independently establishing terms for the exact source
//! and checkpoint.
//!
//! # BF16 posture
//!
//! Every runnable F32 / F16 / BF16 tensor passes through **verbatim** as the
//! matching GGUF type (BF16 emits type 30 = `GgmlType::BF16`, no
//! convert-time widening — the runtime widens BF16 → f32 losslessly at
//! load via the single choke point `crates/vokra-core/src/gguf/quant/
//! mod.rs decode_bf16`). Mirror of the emotion2vec / qwen3_tts /
//! vibevoice / voxcpm2 / moshi / voxtral posture that keeps the CI
//! cache footprint at the smallest tensor payload while preserving the
//! exact upstream bit pattern.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the fixed upstream flattened `state_dict` keys. The
//! converter validates all 623 inference names/shapes before writing bytes;
//! partial, extra, or fork-named checkpoints fail loudly. `unet.tf.*` is the
//! sole explicit omission because fixed `DeepUnet0.forward` does not read it.
//!
//! # No ONNX (permanent)
//!
//! RMVPE upstream is distributed as a torch `.pt` pickle; this
//! converter **never** touches ONNX (FR-LD-05). The `.pt` → safetensors
//! bridge lives in `tools/parity/nemo_pt_to_safetensors.py` (an offline
//! side-car tool, not part of the runtime).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for RMVPE GGUFs. Distinct from every sibling arch
/// tag because RMVPE is the first `category = "f0"` binder in the
/// converter tree — silently sharing an arch tag would misroute the
/// runtime dispatch (an ASR / TTS backbone would try to interpret the
/// 360-class pitch head).
pub const ARCH: &str = "rmvpe";

/// `vokra.model.name` value written for the canonical RMVPE GGUF.
pub const NAME: &str = "rmvpe";

/// `vokra.model.category` value — the first `"f0"` in the converter
/// tree. Consumed by the model-card generator + zoo manifest tier gate
/// so an F0 extractor is not accidentally advertised as an ASR / TTS
/// release.
pub const CATEGORY: &str = "f0";

/// `vokra.provenance.upstream_hf` value — the source repository the
/// weights come from. Recorded so a downstream consumer can re-fetch /
/// re-verify without a separate manifest lookup. RMVPE is distributed
/// via GitHub (no HF Hub mirror at time of writing); this is the
/// GitHub coordinate rather than an `<org>/<repo>` HF path.
pub const UPSTREAM_HF: &str = "yxlllc/RMVPE";
/// Exact source revision whose E2E0 topology the converter validates.
pub const UPSTREAM_REVISION: &str = "0aabafba18289ca938a73af0b0297686abf4922d";

/// Fail-closed weight license marker. `Dream-High/RMVPE` code is
/// Apache-2.0, but the checkpoint-publishing `yxlllc/RMVPE` repository
/// has no license declaration. Override via [`convert_rmvpe_file`] only
/// after verifying terms for the exact checkpoint.
pub const DEFAULT_LICENSE: &str = "unknown";

/// Ad-hoc metadata key for the model category. Kept as a converter-side
/// constant (not a `chunks::KEY_*` alias) until a sibling `category`
/// consumer lands in `vokra-core`. Same key emotion2vec uses (they
/// share the same `vokra.model.category` chunk namespace).
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

// GGUF metadata keys for the RMVPE hparam chunk group. Kept in sync
// with the runtime consumer `vokra_models::f0::rmvpe::{GGUF_KEY_HOP,
// GGUF_KEY_FMIN, GGUF_KEY_FMAX, GGUF_KEY_N_MELS, GGUF_KEY_N_FFT,
// GGUF_KEY_WIN_LENGTH, GGUF_KEY_SAMPLE_RATE, GGUF_KEY_N_CLASS,
// GGUF_KEY_CENTS_PER_CLASS, GGUF_KEY_BASE_HZ,
// GGUF_KEY_UPSTREAM_REVISION}`.
pub(crate) const KEY_HOP: &str = "vokra.rmvpe.hop";
pub(crate) const KEY_FMIN: &str = "vokra.rmvpe.fmin";
pub(crate) const KEY_FMAX: &str = "vokra.rmvpe.fmax";
pub(crate) const KEY_N_MELS: &str = "vokra.rmvpe.n_mels";
pub(crate) const KEY_N_FFT: &str = "vokra.rmvpe.n_fft";
pub(crate) const KEY_WIN_LENGTH: &str = "vokra.rmvpe.win_length";
pub(crate) const KEY_SAMPLE_RATE: &str = "vokra.rmvpe.sample_rate";
pub(crate) const KEY_N_CLASS: &str = "vokra.rmvpe.n_class";
pub(crate) const KEY_CENTS_PER_CLASS: &str = "vokra.rmvpe.cents_per_class";
pub(crate) const KEY_BASE_HZ: &str = "vokra.rmvpe.base_hz";
pub(crate) const KEY_UPSTREAM_REVISION: &str = "vokra.rmvpe.upstream_revision";

// Canonical hparam values transcribed from yxlllc/RMVPE commit
// 0aabafba18289ca938a73af0b0297686abf4922d. Kept here as
// converter-side compile-time constants so a GGUF that never had a
// `vokra.rmvpe.*` chunk written (e.g. an emergency hand-crafted
// checkpoint) still round-trips through the runtime binder's default
// fallback.
pub const DEFAULT_HOP: u32 = 160;
pub const DEFAULT_FMIN: f32 = 30.0;
pub const DEFAULT_FMAX: f32 = 1000.0;
pub const DEFAULT_N_MELS: u32 = 128;
pub const DEFAULT_N_FFT: u32 = 1024;
pub const DEFAULT_WIN_LENGTH: u32 = 1024;
pub const DEFAULT_SAMPLE_RATE: u32 = 16000;
pub const DEFAULT_N_CLASS: u32 = 360;
/// Upstream RMVPE pitch-class grid spacing (20 cents / class = 12
/// classes per semitone). The 360-class head therefore spans
/// `360 * 20 = 7200` cents ≈ 6 octaves starting at `base_hz`.
pub const DEFAULT_CENTS_PER_CLASS: f32 = 20.0;
/// Log-Hz grid anchor. `src/constants.py` uses cents offset
/// `1997.3794084376191`, so class zero is exactly
/// `10 * 2^(1997.3794084376191 / 1200) = 31.7 Hz`.
pub const DEFAULT_BASE_HZ: f32 = 31.7;

const REQUIRED_INFERENCE_TENSORS: usize = 623;
const OPTIONAL_COUNTERS: usize = 118;
const INFERENCE_INERT_TF_TENSORS: usize = 50;
const INFERENCE_INERT_TF_COUNTERS: usize = 10;

#[derive(Debug)]
struct TensorContract {
    required: BTreeMap<String, Vec<usize>>,
    counters: BTreeSet<String>,
    inference_inert: BTreeMap<String, Vec<usize>>,
    inference_inert_counters: BTreeSet<String>,
}

fn add_tensor(map: &mut BTreeMap<String, Vec<usize>>, name: String, shape: &[usize]) {
    assert!(
        map.insert(name, shape.to_vec()).is_none(),
        "RMVPE contract tensor names must be unique"
    );
}

fn add_bn(
    tensors: &mut BTreeMap<String, Vec<usize>>,
    counters: &mut BTreeSet<String>,
    prefix: &str,
    channels: usize,
) {
    for suffix in ["weight", "bias", "running_mean", "running_var"] {
        add_tensor(tensors, format!("{prefix}.{suffix}"), &[channels]);
    }
    counters.insert(format!("{prefix}.num_batches_tracked"));
}

fn add_conv_block(
    tensors: &mut BTreeMap<String, Vec<usize>>,
    counters: &mut BTreeSet<String>,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
) {
    add_tensor(
        tensors,
        format!("{prefix}.conv.0.weight"),
        &[out_channels, in_channels, 3, 3],
    );
    add_bn(tensors, counters, &format!("{prefix}.conv.1"), out_channels);
    add_tensor(
        tensors,
        format!("{prefix}.conv.3.weight"),
        &[out_channels, out_channels, 3, 3],
    );
    add_bn(tensors, counters, &format!("{prefix}.conv.4"), out_channels);
    if in_channels != out_channels {
        add_tensor(
            tensors,
            format!("{prefix}.shortcut.weight"),
            &[out_channels, in_channels, 1, 1],
        );
        add_tensor(tensors, format!("{prefix}.shortcut.bias"), &[out_channels]);
    }
}

fn tensor_contract() -> TensorContract {
    let mut required = BTreeMap::new();
    let mut counters = BTreeSet::new();
    add_bn(&mut required, &mut counters, "unet.encoder.bn", 1);

    let mut in_channels = 1usize;
    let mut out_channels = 16usize;
    for layer in 0..5 {
        for block in 0..4 {
            let block_in = if block == 0 {
                in_channels
            } else {
                out_channels
            };
            add_conv_block(
                &mut required,
                &mut counters,
                &format!("unet.encoder.layers.{layer}.conv.{block}"),
                block_in,
                out_channels,
            );
        }
        in_channels = out_channels;
        out_channels *= 2;
    }

    in_channels = 256;
    out_channels = 512;
    for layer in 0..4 {
        for block in 0..4 {
            let block_in = if block == 0 {
                in_channels
            } else {
                out_channels
            };
            add_conv_block(
                &mut required,
                &mut counters,
                &format!("unet.intermediate.layers.{layer}.conv.{block}"),
                block_in,
                out_channels,
            );
        }
        in_channels = out_channels;
    }

    in_channels = 512;
    for layer in 0..5 {
        out_channels = in_channels / 2;
        add_tensor(
            &mut required,
            format!("unet.decoder.layers.{layer}.conv1.0.weight"),
            &[in_channels, out_channels, 3, 3],
        );
        add_bn(
            &mut required,
            &mut counters,
            &format!("unet.decoder.layers.{layer}.conv1.1"),
            out_channels,
        );
        for block in 0..4 {
            let block_in = if block == 0 {
                out_channels * 2
            } else {
                out_channels
            };
            add_conv_block(
                &mut required,
                &mut counters,
                &format!("unet.decoder.layers.{layer}.conv2.{block}"),
                block_in,
                out_channels,
            );
        }
        in_channels = out_channels;
    }

    add_tensor(&mut required, "cnn.weight".into(), &[3, 16, 3, 3]);
    add_tensor(&mut required, "cnn.bias".into(), &[3]);
    for suffix in ["l0", "l0_reverse"] {
        add_tensor(
            &mut required,
            format!("fc.0.gru.weight_ih_{suffix}"),
            &[768, 384],
        );
        add_tensor(
            &mut required,
            format!("fc.0.gru.weight_hh_{suffix}"),
            &[768, 256],
        );
        add_tensor(&mut required, format!("fc.0.gru.bias_ih_{suffix}"), &[768]);
        add_tensor(&mut required, format!("fc.0.gru.bias_hh_{suffix}"), &[768]);
    }
    add_tensor(&mut required, "fc.1.weight".into(), &[360, 512]);
    add_tensor(&mut required, "fc.1.bias".into(), &[360]);

    // DeepUnet0 constructs a TimbreFilter but never calls it in forward. The
    // released `.pt` can therefore contain these state_dict entries even
    // though the public runnable GGUF correctly omits them.
    let mut inference_inert = BTreeMap::new();
    let mut inference_inert_counters = BTreeSet::new();
    for (layer, channels) in [16usize, 32, 64, 128, 256].into_iter().enumerate() {
        add_conv_block(
            &mut inference_inert,
            &mut inference_inert_counters,
            &format!("unet.tf.layers.{layer}"),
            channels,
            channels,
        );
    }

    assert_eq!(required.len(), REQUIRED_INFERENCE_TENSORS);
    assert_eq!(counters.len(), OPTIONAL_COUNTERS);
    assert_eq!(inference_inert.len(), INFERENCE_INERT_TF_TENSORS);
    assert_eq!(inference_inert_counters.len(), INFERENCE_INERT_TF_COUNTERS);
    TensorContract {
        required,
        counters,
        inference_inert,
        inference_inert_counters,
    }
}

fn contract_error(message: impl Into<String>) -> ConvertError {
    ConvertError::Parse(format!("rmvpe: {} (FR-EX-08)", message.into()))
}

fn validate_tensor_contract(st: &SafetensorsFile) -> Result<BTreeSet<String>, ConvertError> {
    let contract = tensor_contract();
    let actual: BTreeMap<&str, _> = st
        .tensors()
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect();
    for (name, expected_shape) in &contract.required {
        let tensor = actual
            .get(name.as_str())
            .ok_or_else(|| contract_error(format!("required tensor `{name}` is missing")))?;
        let actual_shape: Vec<usize> = tensor.shape.iter().map(|&dim| dim as usize).collect();
        if actual_shape != *expected_shape {
            return Err(contract_error(format!(
                "tensor `{name}` shape {:?} != fixed upstream shape {expected_shape:?}",
                actual_shape
            )));
        }
        if !matches!(tensor.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
            return Err(contract_error(format!(
                "tensor `{name}` dtype {:?} is not F32/F16/BF16",
                tensor.dtype
            )));
        }
    }

    let has_tf = actual.keys().any(|name| name.starts_with("unet.tf."));
    if has_tf {
        for (name, expected_shape) in &contract.inference_inert {
            let tensor = actual.get(name.as_str()).ok_or_else(|| {
                contract_error(format!(
                    "partial inference-inert TimbreFilter: `{name}` is missing"
                ))
            })?;
            let actual_shape: Vec<usize> = tensor.shape.iter().map(|&dim| dim as usize).collect();
            if actual_shape != *expected_shape {
                return Err(contract_error(format!(
                    "inference-inert tensor `{name}` shape {:?} != {expected_shape:?}",
                    actual_shape
                )));
            }
            if !matches!(tensor.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
                return Err(contract_error(format!(
                    "inference-inert tensor `{name}` dtype {:?} is not F32/F16/BF16",
                    tensor.dtype
                )));
            }
        }
    }

    let mut skipped = BTreeSet::new();
    for tensor in st.tensors() {
        let name = tensor.name.as_str();
        if contract.required.contains_key(name) {
            continue;
        }
        if contract.counters.contains(name) {
            if !(tensor.shape.is_empty() || tensor.shape == [1]) {
                return Err(contract_error(format!(
                    "BatchNorm counter `{name}` shape {:?} must be scalar or [1]",
                    tensor.shape
                )));
            }
            if !matches!(tensor.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
                return Err(contract_error(format!(
                    "BatchNorm counter `{name}` dtype {:?} is not F32/F16/BF16",
                    tensor.dtype
                )));
            }
            continue;
        }
        if contract.inference_inert.contains_key(name) {
            if !has_tf {
                unreachable!("an unet.tf tensor makes has_tf true");
            }
            skipped.insert(tensor.name.clone());
            continue;
        }
        if contract.inference_inert_counters.contains(name) {
            if !(tensor.shape.is_empty() || tensor.shape == [1]) {
                return Err(contract_error(format!(
                    "inference-inert BatchNorm counter `{name}` shape {:?} must be \
                     scalar or [1]",
                    tensor.shape
                )));
            }
            skipped.insert(tensor.name.clone());
            continue;
        }
        return Err(contract_error(format!(
            "unsupported tensor `{name}` is outside the fixed E2E0 manifest"
        )));
    }
    Ok(skipped)
}

/// Outcome of an RMVPE conversion.
///
/// All counters are additive and default to zero. Conversion succeeds only
/// after the complete fixed manifest has passed validation.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RmvpeReport {
    /// Total tensors seen in the upstream safetensors header (the sum of
    /// `written + skipped_non_float + skipped_inference_inert`). Pins the budget so a
    /// truncated header cannot silently drop tensors without the caller
    /// noticing.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16 all go through
    /// the same byte-copy path since the BF16 pass-through landed
    /// 2026-07-25).
    pub written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time; anything
    /// that reaches this arm is a quantized dtype the runtime is not
    /// expected to consume).
    pub skipped_non_float: usize,
    /// Float tensors intentionally omitted because `DeepUnet0` constructs the
    /// `unet.tf.*` TimbreFilter but never calls it in its forward.
    pub skipped_inference_inert: usize,
    /// Of the tensors in `written`, how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; runtime widens BF16 →
    /// f32 losslessly via the single choke point
    /// `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 =
    /// top 16 bits of an f32 — `bits << 16` is exact).
    pub bf16_passthrough: usize,
}

/// Reads a safetensors checkpoint at `input` and writes an RMVPE GGUF
/// to `output`.
///
/// Every runnable F32 / F16 / BF16 tensor is emitted verbatim under its
/// upstream name; the `vokra.provenance.*` + `vokra.model.*` + `vokra.rmvpe.*`
/// chunk groups pin the upstream repo, weight license, model category
/// and RMVPE hparams so the runtime binder can bring the graph up
/// without a side-car config lookup.
///
/// `license` overrides the fail-closed `DEFAULT_LICENSE` (`"unknown"`)
/// after the caller has verified terms for the exact checkpoint.
pub fn convert_rmvpe_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<RmvpeReport, ConvertError> {
    // Whole-file read: an RMVPE checkpoint is below the repository's 2 GB
    // remote-work threshold — no need for the
    // streaming path the Moshi / Voxtral GB-scale converters run.
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;
    let skipped_inference_inert = validate_tensor_contract(&st)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // RMVPE hparam chunk group — every value is transcribed from the fixed
    // upstream source commit named in the module documentation.
    b.add_u32(KEY_HOP, DEFAULT_HOP);
    b.add_f32(KEY_FMIN, DEFAULT_FMIN);
    b.add_f32(KEY_FMAX, DEFAULT_FMAX);
    b.add_u32(KEY_N_MELS, DEFAULT_N_MELS);
    b.add_u32(KEY_N_FFT, DEFAULT_N_FFT);
    b.add_u32(KEY_WIN_LENGTH, DEFAULT_WIN_LENGTH);
    b.add_u32(KEY_SAMPLE_RATE, DEFAULT_SAMPLE_RATE);
    b.add_u32(KEY_N_CLASS, DEFAULT_N_CLASS);
    b.add_f32(KEY_CENTS_PER_CLASS, DEFAULT_CENTS_PER_CLASS);
    b.add_f32(KEY_BASE_HZ, DEFAULT_BASE_HZ);
    b.add_string(KEY_UPSTREAM_REVISION, UPSTREAM_REVISION);

    // Fail closed: the exact source/checkpoint repository declares no code or
    // weight terms. Dream-High's separate Apache-2.0 grant does not transfer.
    // A caller-provided SPDX is classified only after that caller has
    // independently verified the exact checkpoint's licence.
    let effective_license = license.unwrap_or(DEFAULT_LICENSE);
    let license_class = LicenseClass::from_license_str(effective_license);
    vokra_core::stamp_provenance(
        &mut b,
        license_class,
        effective_license,
        Some(NAME),
        Some(UPSTREAM_HF),
    );

    let mut report = RmvpeReport::default();
    // Float tensors pass through **verbatim** — no convert-time
    // widening. BF16 stays GGUF `BF16` (type 30) per the accepted ADR
    // (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough); the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. Mirrors
    // `emotion2vec::convert_emotion2vec_file` / `qwen3_tts::convert`.
    for t in st.tensors() {
        report.read += 1;
        if skipped_inference_inert.contains(&t.name) {
            report.skipped_inference_inert += 1;
            continue;
        }
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

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Parse(e.to_string()))?;
    std::fs::write(output, out_bytes).map_err(ConvertError::Io)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-process, per-test scratch path in the system temp dir
    /// (emotion2vec pattern — no external `tempfile` dep, preserving
    /// zero-dep NFR-DS-02). The nanosecond suffix separates the tests
    /// in this module so a parallel `cargo test` cannot clobber files
    /// across them.
    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-rmvpe-{}-{}-{}.bin",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        p
    }

    /// Builds a synthetic safetensors buffer with a single BF16 tensor
    /// (mirror of `emotion2vec::tests::synthetic_bf16_safetensors`) so
    /// a byte-identity assert catches any silent widen / downcast
    /// attempt — the raw zeroed payload would round-trip trivially
    /// through F32 / F16 widen and defeat the pin.
    fn synthetic_bf16_safetensors() -> (Vec<u8>, Vec<u8>) {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");
        let header = r#"{"unet.encoder.layer0.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&bf16);
        (buf, bf16)
    }

    /// A float tensor with an RMVPE-looking prefix is not enough. Conversion
    /// must fail before writing a GGUF unless the complete fixed manifest is
    /// present.
    #[test]
    fn partial_bf16_checkpoint_is_rejected() {
        let (input_bytes, _) = synthetic_bf16_safetensors();
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let error = convert_rmvpe_file(&input, &output, None).expect_err("must reject partial");
        assert!(
            error.to_string().contains("required tensor") && error.to_string().contains("FR-EX-08"),
            "strict error must identify the missing fixed contract: {error}"
        );
        assert!(
            !output.exists(),
            "a rejected conversion must not write output"
        );

        std::fs::remove_file(&input).ok();
    }

    /// Metadata-only output would never load as RMVPE, so an empty checkpoint
    /// is rejected before any output is created.
    #[test]
    fn empty_input_is_rejected() {
        // Minimal safetensors: 8-byte header size (=2), empty JSON `{}`,
        // and no tensor data.
        let header = b"{}";
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header);

        let input = scratch_path("empty-in");
        let output = scratch_path("empty-out");
        std::fs::write(&input, &buf).expect("write empty safetensors");

        let error = convert_rmvpe_file(&input, &output, None).expect_err("must reject empty");
        assert!(error.to_string().contains("required tensor"));
        assert!(!output.exists());

        std::fs::remove_file(&input).ok();
    }

    /// The generated converter manifest is independently count-pinned so a
    /// loop-bound edit cannot silently drift from the audited public header.
    #[test]
    fn exact_contract_counts_and_head_shapes_are_pinned() {
        let contract = tensor_contract();
        assert_eq!(contract.required.len(), 623);
        assert_eq!(contract.counters.len(), 118);
        assert_eq!(contract.inference_inert.len(), 50);
        assert_eq!(contract.inference_inert_counters.len(), 10);
        assert_eq!(
            contract.required.get("fc.0.gru.weight_ih_l0"),
            Some(&vec![768, 384])
        );
        assert_eq!(contract.required.get("fc.1.weight"), Some(&vec![360, 512]));
    }
}
