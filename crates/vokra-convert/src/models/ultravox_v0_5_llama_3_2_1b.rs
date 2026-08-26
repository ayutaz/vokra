#![allow(clippy::doc_lazy_continuation)]
//! **Ultravox v0.5 (Llama-3.2-1B)** (`fixie-ai/ultravox-v0_5-llama-3_2-1b`,
//! **MIT**): safetensors → GGUF conversion (Wave residual, 2026-08-02).
//!
//! Ultravox family entry — fixie-ai's audio-text-to-text multimodal model
//! combining a separately acquired **Llama-3.2-1B** language backbone with a
//! **Whisper encoder + projection adapter** front-end.  The fixie-ai
//! safetensors — and therefore the 1,366,275,264-byte public Vokra GGUF —
//! contain only the 491 BF16 audio-tower/projector tensors.  Llama weights are
//! not bundled. Real audio is fed through the Whisper encoder, projected into
//! the separately licensed Llama token embedding space, then decoded by that
//! companion backbone.
//!
//! **Distinct from siblings [`crate::ModelKind::Voxtral`]** (Mistral text
//! decoder + Whisper encoder) **and [`crate::ModelKind::Qwen2Audio`]**
//! (Qwen2-7B decoder + Whisper encoder). All three are "Whisper encoder +
//! text LM decoder" audio-LLMs, but the decoder backbone (Llama vs Mistral
//! vs Qwen2) fixes tensor layout + tokenizer + rope base; silently sharing
//! a converter or arch tag across the three would misroute runtime dispatch
//! at the LM decoder loader (FR-EX-08 forbids silent shape misroute). This
//! converter therefore emits a distinct arch tag `ultravox` (shared with any
//! future Ultravox v0.6+ / Llama-3.2-3B siblings — the family topology is
//! the same, only the decoder scale differs, mirroring the MusicGen family
//! shared-arch-tag pattern).
//!
//! Category `audio-llm` is shared with sibling Qwen2-Audio-7B / Voxtral /
//! Kimi-Audio / Step-Audio2-Mini / Baichuan-Audio siblings.  The strict native
//! binder now executes all 32 Whisper layers plus the exact stack-8 SwiGLU
//! projector on CPU or Metal.  [`convert_ultravox_llama_companion_file`]
//! separately validates and streams the exact 146-tensor gated Meta base, but
//! a complete text-generation route remains deliberately partial until that
//! companion, tokenizer and chat/audio-placeholder contract are runtime-bound.
//! The public GGUF is never treated as a standalone LM.
//!
//! # License posture — MIT (**Permissive**)
//!
//! Sibling to the first-party Whisper / piper-plus / Silero / CAM++ /
//! Moonshine Permissive posture. Upstream `fixie-ai/ultravox-v0_5-llama-3_2-1b`
//! HF card declares `license: mit` per the SoTA scope-expansion 2026-07-30
//! canary sweep. Default license `mit` +
//! [`vokra_core::LicenseClass::Permissive`]; override via
//! [`crate::convert_file_licensed`] `license` when the caller legitimately
//! holds a different SPDX id (the Whisper / kokoro / xcodec2 override
//! pattern). §3.1 sign-off remains owner (fail-closed default per memory
//! `[[feedback-license-signoff-primary-source]]`).
//!
//! # Scale — 1.37 GB public artifact
//!
//! The audited public GGUF is exactly 1,366,275,264 bytes. This converter reads
//! both the source and generated GGUF into owned buffers, so model conversion
//! and real-weight validation belong on the configured remote workflow when
//! the maintainer requests that the Mac stay idle; the mmap runtime binder is
//! the bounded-memory path.
//!
//! # BF16 pass-through skeleton
//!
//! Mirror of sibling `musicgen_small.rs` / `moonshine_base.rs` /
//! `hubert_large_ls960.rs` / `openwakeword.rs` / `demucs_htdemucs.rs`
//! skeleton. Every F32 / F16 / BF16 tensor passes through verbatim; non-
//! float tensors are skipped (no quantisation applied at the converter
//! boundary — quantisation is a separate pass).  The output is the MIT audio
//! component, not the separately distributed Llama companion.
//!
//! # Separate Llama companion
//!
//! The companion conversion is intentionally stricter than the historical
//! audio converter: it requires the exact official Llama-3.2-1B-Instruct
//! config, the tied-embedding 146-tensor BF16 manifest, and an explicit
//! immutable source revision.  Its output carries arch
//! `ultravox_llama_companion` and `ConditionalCommercial`, preserving Meta's
//! 700-million-MAU threshold.  No function in this module downloads or
//! publishes the gated checkpoint, and the streaming path is reserved for
//! VAST under the repository's >=2 GB policy.

use std::collections::BTreeMap;
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, GgufStreamWriter, GgufTensorDecl, chunks};
use vokra_core::json::{self, JsonValue};

use crate::ConvertError;
use crate::safetensors::{SafetensorsFile, SafetensorsFileReader};

pub const ARCH: &str = "ultravox";
pub const NAME: &str = "ultravox-v0-5-llama-3-2-1b";
pub const CATEGORY: &str = "audio-llm";
pub const UPSTREAM_HF: &str = "fixie-ai/ultravox-v0_5-llama-3_2-1b";
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

const UPSTREAM_SOURCE: &str = "fixie-ai/ultravox-v0_5-llama-3_2-1b (Ultravox v0.5 Whisper encoder + projection adapter only; Llama-3.2-1B companion not bundled, mit)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Architecture tag of the separately acquired Llama text companion.
///
/// It is deliberately not `ultravox`: the public MIT artifact and the Meta
/// Community License artifact must remain independently policy-checkable.
pub const COMPANION_ARCH: &str = "ultravox_llama_companion";
/// Model identity stamped into a user-converted companion.
pub const COMPANION_NAME: &str = "meta-llama-3.2-1b-instruct-ultravox-companion";
/// Exact gated upstream repository required by Ultravox v0.5.
pub const COMPANION_UPSTREAM_HF: &str = "meta-llama/Llama-3.2-1B-Instruct";
/// Exact Hugging Face `license` identifier carried by the gated Meta release.
pub const COMPANION_LICENSE: &str = "llama3.2";
/// Shape/name manifest of the official tied-embedding 146-tensor checkpoint.
pub const COMPANION_MANIFEST_SHA256: &str =
    "7832a30cf077054292c8728a5e04621bfb431369566db282b3ccf1692a4e3712";

const COMPANION_CATEGORY: &str = "text-llm-companion";
const KEY_PROVENANCE_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_COMPANION_SOURCE_REVISION: &str = "vokra.ultravox.companion.source_revision";
const KEY_COMPANION_CONFIG_SHA256: &str = "vokra.ultravox.companion.config_sha256";
const KEY_COMPANION_MANIFEST_SHA256: &str = "vokra.ultravox.companion.tensor_manifest_sha256";
const KEY_COMPANION_HIDDEN_SIZE: &str = "vokra.ultravox.companion.hidden_size";
const KEY_COMPANION_N_LAYER: &str = "vokra.ultravox.companion.n_layer";
const KEY_COMPANION_N_HEAD: &str = "vokra.ultravox.companion.n_head";
const KEY_COMPANION_N_KV_HEAD: &str = "vokra.ultravox.companion.n_kv_head";
const KEY_COMPANION_HEAD_DIM: &str = "vokra.ultravox.companion.head_dim";
const KEY_COMPANION_FFN_DIM: &str = "vokra.ultravox.companion.ffn_dim";
const KEY_COMPANION_VOCAB_SIZE: &str = "vokra.ultravox.companion.vocab_size";
const KEY_COMPANION_MAX_POSITIONS: &str = "vokra.ultravox.companion.max_positions";
const KEY_COMPANION_RMS_NORM_EPS: &str = "vokra.ultravox.companion.rms_norm_eps";
const KEY_COMPANION_ROPE_THETA: &str = "vokra.ultravox.companion.rope_theta";
const KEY_COMPANION_ROPE_FACTOR: &str = "vokra.ultravox.companion.rope.factor";
const KEY_COMPANION_ROPE_LOW_FACTOR: &str = "vokra.ultravox.companion.rope.low_freq_factor";
const KEY_COMPANION_ROPE_HIGH_FACTOR: &str = "vokra.ultravox.companion.rope.high_freq_factor";
const KEY_COMPANION_ROPE_ORIGINAL_MAX: &str =
    "vokra.ultravox.companion.rope.original_max_positions";
const KEY_COMPANION_TIED_EMBEDDINGS: &str = "vokra.ultravox.companion.tied_embeddings";
const KEY_COMPANION_ATTENTION_BIAS: &str = "vokra.ultravox.companion.attention_bias";
const KEY_COMPANION_MLP_BIAS: &str = "vokra.ultravox.companion.mlp_bias";

const COMPANION_HIDDEN_SIZE: u32 = 2_048;
const COMPANION_N_LAYER: u32 = 16;
const COMPANION_N_HEAD: u32 = 32;
const COMPANION_N_KV_HEAD: u32 = 8;
const COMPANION_HEAD_DIM: u32 = 64;
const COMPANION_FFN_DIM: u32 = 8_192;
const COMPANION_VOCAB_SIZE: u32 = 128_256;
const COMPANION_MAX_POSITIONS: u32 = 131_072;
const COMPANION_RMS_NORM_EPS: f32 = 1e-5;
const COMPANION_ROPE_THETA: f32 = 500_000.0;
const COMPANION_ROPE_FACTOR: f32 = 32.0;
const COMPANION_ROPE_LOW_FACTOR: f32 = 1.0;
const COMPANION_ROPE_HIGH_FACTOR: f32 = 4.0;
const COMPANION_ROPE_ORIGINAL_MAX: u32 = 8_192;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct UltravoxV05Llama321bReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

/// Report from the bounded-memory Llama companion conversion.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct UltravoxLlamaCompanionReport {
    /// Exact BF16 tensors observed and copied.
    pub written: usize,
    /// Number of metadata entries stamped before GGUF serialization.
    pub metadata_count: usize,
}

pub fn convert_ultravox_v0_5_llama_3_2_1b_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<UltravoxV05Llama321bReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(&mut b, class, &spdx, Some(NAME), Some(UPSTREAM_SOURCE));
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let mut report = UltravoxV05Llama321bReport::default();
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

/// Converts the separately licensed Meta Llama-3.2-1B-Instruct checkpoint
/// into an mmap-friendly Ultravox companion GGUF.
///
/// `input` must be the official single-file BF16 `model.safetensors` and
/// `config` the matching upstream `config.json`.  The source revision is an
/// explicit immutable 40-hex commit because raw Hugging Face config files do
/// not reliably embed the snapshot revision.  Tensor data is streamed one
/// tensor at a time; the whole >2 GB checkpoint is never materialized.
///
/// This function does not download or publish anything.  The input is gated
/// and remains a user-acquired artifact under the Llama 3.2 Community License.
pub fn convert_ultravox_llama_companion_file(
    input: &Path,
    config: &Path,
    source_revision: &str,
    output: &Path,
) -> Result<UltravoxLlamaCompanionReport, ConvertError> {
    let revision = validate_source_revision(source_revision)?;
    let config_bytes = std::fs::read(config)?;
    validate_companion_config(&config_bytes)?;

    let mut reader = SafetensorsFileReader::open(input)?;
    let expected = companion_tensor_contract();
    validate_companion_manifest(&reader, &expected)?;

    let metadata = companion_metadata(&revision, &config_bytes);
    let metadata_count = metadata.metadata_count();
    let declarations = expected
        .iter()
        .map(|(name, dimensions)| GgufTensorDecl {
            name: name.clone(),
            dtype: GgmlType::BF16,
            dimensions: dimensions.clone(),
        })
        .collect::<Vec<_>>();

    let output_file = std::fs::File::create(output)?;
    let mut writer = GgufStreamWriter::begin(
        std::io::BufWriter::new(output_file),
        &metadata,
        &declarations,
    )?;
    // Keep the metadata builder visibly tensor-free: streamed declarations
    // are the sole tensor source and the writer rejects accidental mixing.
    debug_assert_eq!(metadata.tensor_count(), 0);
    let mut payload = Vec::new();
    for declaration in &declarations {
        reader.read_tensor_into(&declaration.name, &mut payload)?;
        writer.write_tensor(&declaration.name, &payload)?;
    }
    drop(payload);
    let output_file = writer
        .finish()?
        .into_inner()
        .map_err(|error| ConvertError::Io(error.into_error()))?;
    output_file.sync_all()?;
    Ok(UltravoxLlamaCompanionReport {
        written: declarations.len(),
        metadata_count,
    })
}

fn companion_metadata(source_revision: &str, config_bytes: &[u8]) -> GgufBuilder {
    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, COMPANION_ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, COMPANION_NAME);
    builder.add_string(KEY_MODEL_CATEGORY, COMPANION_CATEGORY);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_HF, COMPANION_UPSTREAM_HF);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_REVISION, source_revision);
    builder.add_string(KEY_COMPANION_SOURCE_REVISION, source_revision);
    builder.add_string(
        KEY_COMPANION_CONFIG_SHA256,
        &crate::models::canary_1b_flash::hex(&crate::models::canary_1b_flash::sha256(config_bytes)),
    );
    builder.add_string(KEY_COMPANION_MANIFEST_SHA256, COMPANION_MANIFEST_SHA256);
    builder.add_u32(KEY_COMPANION_HIDDEN_SIZE, COMPANION_HIDDEN_SIZE);
    builder.add_u32(KEY_COMPANION_N_LAYER, COMPANION_N_LAYER);
    builder.add_u32(KEY_COMPANION_N_HEAD, COMPANION_N_HEAD);
    builder.add_u32(KEY_COMPANION_N_KV_HEAD, COMPANION_N_KV_HEAD);
    builder.add_u32(KEY_COMPANION_HEAD_DIM, COMPANION_HEAD_DIM);
    builder.add_u32(KEY_COMPANION_FFN_DIM, COMPANION_FFN_DIM);
    builder.add_u32(KEY_COMPANION_VOCAB_SIZE, COMPANION_VOCAB_SIZE);
    builder.add_u32(KEY_COMPANION_MAX_POSITIONS, COMPANION_MAX_POSITIONS);
    builder.add_f32(KEY_COMPANION_RMS_NORM_EPS, COMPANION_RMS_NORM_EPS);
    builder.add_f32(KEY_COMPANION_ROPE_THETA, COMPANION_ROPE_THETA);
    builder.add_f32(KEY_COMPANION_ROPE_FACTOR, COMPANION_ROPE_FACTOR);
    builder.add_f32(KEY_COMPANION_ROPE_LOW_FACTOR, COMPANION_ROPE_LOW_FACTOR);
    builder.add_f32(KEY_COMPANION_ROPE_HIGH_FACTOR, COMPANION_ROPE_HIGH_FACTOR);
    builder.add_u32(KEY_COMPANION_ROPE_ORIGINAL_MAX, COMPANION_ROPE_ORIGINAL_MAX);
    builder.add_bool(KEY_COMPANION_TIED_EMBEDDINGS, true);
    builder.add_bool(KEY_COMPANION_ATTENTION_BIAS, false);
    builder.add_bool(KEY_COMPANION_MLP_BIAS, false);
    vokra_core::stamp_provenance(
        &mut builder,
        LicenseClass::ConditionalCommercial,
        COMPANION_LICENSE,
        Some(COMPANION_NAME),
        Some(&format!(
            "{COMPANION_UPSTREAM_HF}@{source_revision}; user-acquired gated companion; not bundled with the MIT Ultravox audio artifact"
        )),
    );
    builder
}

fn companion_tensor_contract() -> BTreeMap<String, Vec<u64>> {
    let hidden = u64::from(COMPANION_HIDDEN_SIZE);
    let ffn = u64::from(COMPANION_FFN_DIM);
    let q_width = u64::from(COMPANION_N_HEAD) * u64::from(COMPANION_HEAD_DIM);
    let kv_width = u64::from(COMPANION_N_KV_HEAD) * u64::from(COMPANION_HEAD_DIM);
    let vocab = u64::from(COMPANION_VOCAB_SIZE);
    let mut tensors = BTreeMap::new();
    tensors.insert("model.embed_tokens.weight".to_owned(), vec![vocab, hidden]);
    tensors.insert("model.norm.weight".to_owned(), vec![hidden]);
    for layer in 0..COMPANION_N_LAYER {
        let prefix = format!("model.layers.{layer}");
        tensors.insert(format!("{prefix}.input_layernorm.weight"), vec![hidden]);
        tensors.insert(
            format!("{prefix}.self_attn.q_proj.weight"),
            vec![q_width, hidden],
        );
        tensors.insert(
            format!("{prefix}.self_attn.k_proj.weight"),
            vec![kv_width, hidden],
        );
        tensors.insert(
            format!("{prefix}.self_attn.v_proj.weight"),
            vec![kv_width, hidden],
        );
        tensors.insert(
            format!("{prefix}.self_attn.o_proj.weight"),
            vec![hidden, q_width],
        );
        tensors.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![hidden],
        );
        tensors.insert(format!("{prefix}.mlp.gate_proj.weight"), vec![ffn, hidden]);
        tensors.insert(format!("{prefix}.mlp.up_proj.weight"), vec![ffn, hidden]);
        tensors.insert(format!("{prefix}.mlp.down_proj.weight"), vec![hidden, ffn]);
    }
    debug_assert_eq!(tensors.len(), 146);
    tensors
}

fn validate_companion_manifest(
    reader: &SafetensorsFileReader,
    expected: &BTreeMap<String, Vec<u64>>,
) -> Result<(), ConvertError> {
    if reader.tensors().len() != expected.len() {
        return Err(ConvertError::Parse(format!(
            "ultravox Llama companion has {} tensors, expected exactly {}",
            reader.tensors().len(),
            expected.len()
        )));
    }
    for tensor in reader.tensors() {
        let expected_shape = expected.get(&tensor.name).ok_or_else(|| {
            ConvertError::Parse(format!(
                "ultravox Llama companion contains unexpected tensor {:?}",
                tensor.name
            ))
        })?;
        if tensor.dtype != GgmlType::BF16 {
            return Err(ConvertError::Parse(format!(
                "ultravox Llama companion tensor {:?} is {:?}, expected canonical BF16",
                tensor.name, tensor.dtype
            )));
        }
        if &tensor.shape != expected_shape {
            return Err(ConvertError::Parse(format!(
                "ultravox Llama companion tensor {:?} has shape {:?}, expected {:?}",
                tensor.name, tensor.shape, expected_shape
            )));
        }
    }
    if let Some(name) = expected
        .keys()
        .find(|name| reader.tensor_info(name).is_none())
    {
        return Err(ConvertError::Parse(format!(
            "ultravox Llama companion is missing tensor {name:?}"
        )));
    }
    let digest = crate::models::canary_1b_flash::hex(
        &crate::models::canary_1b_flash::manifest_sha256(expected),
    );
    if digest != COMPANION_MANIFEST_SHA256 {
        return Err(ConvertError::Parse(format!(
            "internal Ultravox companion manifest digest {digest} != pinned {COMPANION_MANIFEST_SHA256}"
        )));
    }
    Ok(())
}

fn validate_source_revision(source_revision: &str) -> Result<String, ConvertError> {
    let revision = source_revision.trim();
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ConvertError::Usage(format!(
            "ultravox Llama companion --revision must be an immutable 40-hex commit, got {source_revision:?}"
        )));
    }
    Ok(revision.to_ascii_lowercase())
}

fn validate_companion_config(bytes: &[u8]) -> Result<(), ConvertError> {
    let root = json::parse(bytes).map_err(|error| {
        ConvertError::Parse(format!("ultravox Llama companion config.json: {error}"))
    })?;
    require_json_str(&root, "model_type", "llama")?;
    require_json_str(&root, "hidden_act", "silu")?;
    require_json_u64(&root, "hidden_size", u64::from(COMPANION_HIDDEN_SIZE))?;
    require_json_u64(&root, "intermediate_size", u64::from(COMPANION_FFN_DIM))?;
    require_json_u64(&root, "num_hidden_layers", u64::from(COMPANION_N_LAYER))?;
    require_json_u64(&root, "num_attention_heads", u64::from(COMPANION_N_HEAD))?;
    require_json_u64(&root, "num_key_value_heads", u64::from(COMPANION_N_KV_HEAD))?;
    require_json_u64(&root, "head_dim", u64::from(COMPANION_HEAD_DIM))?;
    require_json_u64(&root, "vocab_size", u64::from(COMPANION_VOCAB_SIZE))?;
    require_json_u64(
        &root,
        "max_position_embeddings",
        u64::from(COMPANION_MAX_POSITIONS),
    )?;
    require_json_f64(&root, "rms_norm_eps", f64::from(COMPANION_RMS_NORM_EPS))?;
    require_json_f64(&root, "rope_theta", f64::from(COMPANION_ROPE_THETA))?;
    require_json_bool(&root, "tie_word_embeddings", true)?;
    require_json_bool(&root, "attention_bias", false)?;
    require_json_bool(&root, "mlp_bias", false)?;

    let rope = root.get("rope_scaling").ok_or_else(|| {
        ConvertError::Parse("ultravox Llama companion config: missing `rope_scaling`".to_owned())
    })?;
    require_json_str(rope, "rope_type", "llama3")?;
    require_json_f64(rope, "factor", f64::from(COMPANION_ROPE_FACTOR))?;
    require_json_f64(
        rope,
        "low_freq_factor",
        f64::from(COMPANION_ROPE_LOW_FACTOR),
    )?;
    require_json_f64(
        rope,
        "high_freq_factor",
        f64::from(COMPANION_ROPE_HIGH_FACTOR),
    )?;
    require_json_u64(
        rope,
        "original_max_position_embeddings",
        u64::from(COMPANION_ROPE_ORIGINAL_MAX),
    )?;
    Ok(())
}

fn require_json_str(root: &JsonValue, key: &str, expected: &str) -> Result<(), ConvertError> {
    let actual = root.get(key).and_then(JsonValue::as_str);
    if actual != Some(expected) {
        return Err(ConvertError::Parse(format!(
            "ultravox Llama companion config `{key}` is {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn require_json_u64(root: &JsonValue, key: &str, expected: u64) -> Result<(), ConvertError> {
    let actual = root.get(key).and_then(JsonValue::as_u64);
    if actual != Some(expected) {
        return Err(ConvertError::Parse(format!(
            "ultravox Llama companion config `{key}` is {actual:?}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_json_bool(root: &JsonValue, key: &str, expected: bool) -> Result<(), ConvertError> {
    let actual = match root.get(key) {
        Some(JsonValue::Bool(value)) => Some(*value),
        _ => None,
    };
    if actual != Some(expected) {
        return Err(ConvertError::Parse(format!(
            "ultravox Llama companion config `{key}` is {actual:?}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_json_f64(root: &JsonValue, key: &str, expected: f64) -> Result<(), ConvertError> {
    let actual = match root.get(key) {
        Some(JsonValue::Int(value)) => Some(*value as f64),
        Some(JsonValue::Float(value)) => Some(*value),
        _ => None,
    };
    if !actual.is_some_and(|value| {
        let tolerance = expected.abs().max(1.0) * 1e-7;
        (value - expected).abs() <= tolerance
    }) {
        return Err(ConvertError::Parse(format!(
            "ultravox Llama companion config `{key}` is {actual:?}, expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::GgufFile;

    fn tmp_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-ultravox-v0-5-llama-3-2-1b-{tag}-{}-{n}",
            std::process::id()
        ));
        p
    }

    fn safetensors_one(name: &str, dtype: &str, shape: &[u64], payload: &[u8]) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"{dtype}","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn official_companion_config() -> Vec<u8> {
        br#"{
          "model_type":"llama",
          "hidden_act":"silu",
          "hidden_size":2048,
          "intermediate_size":8192,
          "num_hidden_layers":16,
          "num_attention_heads":32,
          "num_key_value_heads":8,
          "head_dim":64,
          "vocab_size":128256,
          "max_position_embeddings":131072,
          "rms_norm_eps":0.00001,
          "rope_theta":500000,
          "tie_word_embeddings":true,
          "attention_bias":false,
          "mlp_bias":false,
          "rope_scaling":{
            "rope_type":"llama3",
            "factor":32,
            "low_freq_factor":1,
            "high_freq_factor":4,
            "original_max_position_embeddings":8192
          }
        }"#
        .to_vec()
    }

    #[test]
    fn ultravox_v0_5_llama_3_2_1b_f32_tensor_passes_through_and_default_license_is_permissive() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // Exact public projector namespace (the real release shape is
        // [2048, 2048]; this tiny payload only pins identity pass-through).
        let st = safetensors_one(
            "multi_modal_projector.linear_2.weight",
            "F32",
            &[1, 2],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();
        let r = convert_ultravox_v0_5_llama_3_2_1b_file(&inp, &outp, None).unwrap();
        assert_eq!(r.read, 1);
        assert_eq!(r.written, 1);
        assert_eq!(r.bf16_passthrough, 0);

        let g = GgufFile::open(&outp).unwrap();
        let read_str = |key: &str| -> String {
            g.get(key)
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| panic!("{key}: missing"))
                .to_owned()
        };
        assert_eq!(read_str(chunks::KEY_MODEL_ARCH), ARCH);
        assert_eq!(read_str(chunks::KEY_MODEL_NAME), NAME);
        assert_eq!(read_str(KEY_MODEL_CATEGORY), CATEGORY);
        assert_eq!(read_str(KEY_PROVENANCE_UPSTREAM_HF), UPSTREAM_HF);
        // Permissive default (mit) — sibling to Moonshine / Whisper /
        // piper-plus / Silero / CAM++ first-party posture.
        assert_eq!(read_str("vokra.provenance.license"), DEFAULT_LICENSE_SPDX);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn ultravox_v0_5_llama_3_2_1b_bf16_tensor_passes_through_verbatim() {
        let inp = tmp_path("bf16-in");
        let outp = tmp_path("bf16-out");
        // BF16 payload — matches the SoTA plan skeleton contract: runtime
        // widens BF16 → F32 exactly at load, so the converter must not
        // touch the bytes.
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        // Exact public Whisper encoder namespace.  No `language_model.*`
        // tensor exists in this artifact; the Llama companion is separate.
        let st = safetensors_one(
            "audio_tower.layers.0.self_attn.q_proj.weight",
            "BF16",
            &[1, 2],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();
        let r = convert_ultravox_v0_5_llama_3_2_1b_file(&inp, &outp, None).unwrap();
        assert_eq!(r.bf16_passthrough, 1);
        assert_eq!(r.written, 1);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn ultravox_v0_5_llama_3_2_1b_license_override_swaps_stamp() {
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();
        convert_ultravox_v0_5_llama_3_2_1b_file(&inp, &outp, Some("apache-2.0")).unwrap();
        let g = GgufFile::open(&outp).unwrap();
        assert_eq!(
            g.get("vokra.provenance.license").and_then(|v| v.as_str()),
            Some("apache-2.0")
        );
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn companion_contract_is_exact_and_digest_pinned() {
        let contract = companion_tensor_contract();
        assert_eq!(contract.len(), 146);
        assert_eq!(
            contract.get("model.embed_tokens.weight"),
            Some(&vec![128_256, 2_048])
        );
        assert_eq!(
            contract.get("model.layers.15.self_attn.k_proj.weight"),
            Some(&vec![512, 2_048])
        );
        assert_eq!(
            contract.get("model.layers.15.mlp.down_proj.weight"),
            Some(&vec![2_048, 8_192])
        );
        assert!(!contract.contains_key("lm_head.weight"));
        let digest = crate::models::canary_1b_flash::hex(
            &crate::models::canary_1b_flash::manifest_sha256(&contract),
        );
        assert_eq!(digest, COMPANION_MANIFEST_SHA256);
    }

    #[test]
    fn companion_config_and_revision_fail_closed() {
        let config = official_companion_config();
        validate_companion_config(&config).expect("official config");
        assert_eq!(
            validate_source_revision("0123456789ABCDEF0123456789ABCDEF01234567").expect("revision"),
            "0123456789abcdef0123456789abcdef01234567"
        );
        assert!(validate_source_revision("main").is_err());

        let drifted = String::from_utf8(config).expect("utf8").replacen(
            "\"num_key_value_heads\":8",
            "\"num_key_value_heads\":4",
            1,
        );
        let error = validate_companion_config(drifted.as_bytes()).expect_err("KV drift");
        assert!(error.to_string().contains("num_key_value_heads"));
    }

    #[test]
    fn companion_metadata_separates_mit_audio_and_meta_license() {
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let config = official_companion_config();
        let builder = companion_metadata(revision, &config);
        assert_eq!(builder.tensor_count(), 0);
        let file = GgufFile::parse(builder.to_bytes().expect("metadata GGUF"))
            .expect("parse metadata GGUF");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH)
                .and_then(|value| value.as_str()),
            Some(COMPANION_ARCH)
        );
        assert_eq!(
            file.get("vokra.provenance.weight_license")
                .and_then(|value| value.as_str()),
            Some("conditional-commercial")
        );
        assert_eq!(
            file.get(KEY_COMPANION_SOURCE_REVISION)
                .and_then(|value| value.as_str()),
            Some(revision)
        );
    }
}
