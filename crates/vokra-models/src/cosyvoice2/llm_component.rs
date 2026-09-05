//! Private strict binder for the prepared CosyVoice2 LLM component.
//!
//! The standalone component remains inspection-only. This module binds the
//! exact converter contract and constructs the existing real Qwen2 backbone
//! plus speech wrapper seam, but it is not exported as a production model
//! entry point and makes no parity or publication claim.

use std::collections::{BTreeMap, BTreeSet};

use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, GgufTensorInfo, chunks};
use vokra_core::{Result, VokraError};

use super::llm::{LlmBackbone, LlmBackboneConfig, LlmWeights};
use super::speech_lm::{SpeechLm, SpeechLmTensors};
use crate::strict_checkpoint::sha256_bytes;

const LABEL: &str = "cosyvoice2 LLM component";
const ARCH: &str = "cosyvoice2";
const MODEL_NAME: &str = "cosyvoice2-0.5b";
const CATEGORY: &str = "llm";
const COMPONENT: &str = "llm";
const COMPOSITE_STATUS: &str = "INSPECTION_ONLY";
const UPSTREAM_HF: &str = "FunAudioLLM/CosyVoice2-0.5B";
const UPSTREAM_REVISION: &str = "eec1ae6c79877dbd9379285cf8789c9e0879293d";
const CHECKPOINT_FILE: &str = "llm.pt";
const CHECKPOINT_BYTES: u32 = 2_023_316_821;
const CHECKPOINT_SHA256: &str = "b144ef55b51ce8cfb79a73c90dbba0bdaba4e451c0ebcfab20f769264f84a608";
const CONFIG_FILE: &str = "cosyvoice2.yaml";
const CONFIG_BYTES: u32 = 7_330;
const CONFIG_SHA256: &str = "0af2c0d010c477187c39f3e8fd5f1ae2e4e6f90ad03ba37c10ed6c6a87b05959";
const CONFIG_BLOB: &str = "bc19267bbfd373c9a760b7667a74349ddd487db1";
const QWEN_CONFIG_FILE: &str = "CosyVoice-BlankEN/config.json";
const QWEN_CONFIG_BYTES: u32 = 659;
const QWEN_CONFIG_SHA256: &str = "168aa1bd401abc3bc262ba15ba4e499627a8b4e006e9d050b47c22de20660185";
const QWEN_CONFIG_BLOB: &str = "463b055262b6c66c4629a74a4b300bfe2ed31d3c";
const SOURCE_REPOSITORY: &str = "https://github.com/FunAudioLLM/CosyVoice.git";
const SOURCE_REVISION: &str = "8555549e882236e6541748b1042d95693caa82ba";
const MANIFEST_SHA256: &str = "07cf10ae088c27a7c88e1c08fb231d00b01bba0c13f312a74d2fd4b35403bda2";
const SOURCE_LICENSE_PATH: &str = "LICENSE";
const SOURCE_LICENSE_BYTES: u64 = 11_357;
const SOURCE_LICENSE_SHA256: &str =
    "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4";
const SOURCE_LICENSE_BLOB: &str = "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64";
const SOURCE_LICENSE_SPDX: &str = "Apache-2.0";
const PREPARED_STATUS: &str = "PREPARED_INPUT_DIGEST_RECORDED_NOT_PINNED";
const VOCAB_SIZE: u32 = 151_936;
const HIDDEN_DIM: u32 = 896;
const N_LAYER: u32 = 24;
const FFN_DIM: u32 = 4_864;
const N_HEAD: u32 = 14;
const N_HEAD_KV: u32 = 2;
const N_CTX: u32 = 32_768;
const ROPE_BASE: f32 = 1_000_000.0;
const RMS_NORM_EPS: f32 = 1.0e-6;
const TENSOR_COUNT: usize = 295;
const SPEECH_HEAD_SIZE: usize = 6_564;
const MANIFEST_DIGEST: [u8; 32] = [
    0x07, 0xcf, 0x10, 0xae, 0x08, 0x8c, 0x27, 0xa7, 0xc8, 0x8e, 0x1c, 0x08, 0xfb, 0x23, 0x1d, 0x00,
    0xb0, 0x1b, 0xba, 0x0c, 0x13, 0xf3, 0x12, 0xa7, 0x4d, 0x2f, 0xd4, 0xb3, 0x54, 0x03, 0xbd, 0xa2,
];

const KEY_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_WEIGHT_LICENSE: &str = "vokra.provenance.weight_license";
const KEY_LICENSE: &str = "vokra.provenance.license";
const KEY_MODEL_ID: &str = "vokra.provenance.model_id";
const KEY_COMPONENT: &str = "vokra.cosyvoice2.component";
const KEY_COMPOSITE_STATUS: &str = "vokra.cosyvoice2.composite_status";
const KEY_UPSTREAM_COMPONENT_FILE: &str = "vokra.cosyvoice2.upstream_component.file";
const KEY_UPSTREAM_COMPONENT_BYTES: &str = "vokra.cosyvoice2.upstream_component.bytes";
const KEY_UPSTREAM_COMPONENT_SHA256: &str = "vokra.cosyvoice2.upstream_component.sha256";
const KEY_SOURCE_LICENSE_PATH: &str = "vokra.cosyvoice2.source_license.path";
const KEY_SOURCE_LICENSE_BYTES: &str = "vokra.cosyvoice2.source_license.bytes";
const KEY_SOURCE_LICENSE_SHA256: &str = "vokra.cosyvoice2.source_license.sha256";
const KEY_SOURCE_LICENSE_BLOB: &str = "vokra.cosyvoice2.source_license.git_blob_sha1";
const KEY_SOURCE_LICENSE_SPDX: &str = "vokra.cosyvoice2.source_license.spdx";
const KEY_PREPARED_BYTES: &str = "vokra.cosyvoice2.prepared_input.bytes";
const KEY_PREPARED_SHA256: &str = "vokra.cosyvoice2.prepared_input.sha256";
const KEY_PREPARED_STATUS: &str = "vokra.cosyvoice2.prepared_input.authentication_status";
const KEY_CONFIG_FILE: &str = "vokra.cosyvoice2.config_file";
const KEY_CONFIG_BYTES: &str = "vokra.cosyvoice2.config_bytes";
const KEY_CONFIG_SHA256: &str = "vokra.cosyvoice2.config_sha256";
const KEY_CONFIG_BLOB: &str = "vokra.cosyvoice2.config_git_blob_sha1";
const KEY_QWEN_CONFIG_FILE: &str = "vokra.cosyvoice2.qwen_config_file";
const KEY_QWEN_CONFIG_BYTES: &str = "vokra.cosyvoice2.qwen_config_bytes";
const KEY_QWEN_CONFIG_SHA256: &str = "vokra.cosyvoice2.qwen_config_sha256";
const KEY_QWEN_CONFIG_BLOB: &str = "vokra.cosyvoice2.qwen_config_git_blob_sha1";
const KEY_SOURCE_REPOSITORY: &str = "vokra.cosyvoice2.source_repository";
const KEY_SOURCE_REVISION: &str = "vokra.cosyvoice2.source_revision";
const KEY_TENSOR_MANIFEST: &str = "vokra.cosyvoice2.tensor_manifest_sha256";
const KEY_VOCAB_SIZE: &str = "vokra.cosyvoice2.arch.vocab_size";
const KEY_HIDDEN_DIM: &str = "vokra.cosyvoice2.arch.hidden_dim";
const KEY_N_LAYER: &str = "vokra.cosyvoice2.arch.n_layer";
const KEY_FFN_DIM: &str = "vokra.cosyvoice2.arch.ffn_dim";
const KEY_N_HEAD: &str = "vokra.cosyvoice2.arch.n_head";
const KEY_N_HEAD_KV: &str = "vokra.cosyvoice2.arch.n_head_kv";
const KEY_N_CTX: &str = "vokra.cosyvoice2.arch.n_ctx";
const KEY_ROPE_BASE: &str = "vokra.cosyvoice2.arch.rope_base";
const KEY_RMS_NORM_EPS: &str = "vokra.cosyvoice2.arch.rms_norm_eps";

const SOURCE_ROLES: &[(&str, &str, &str)] = &[
    (
        "cosyvoice/cli/cosyvoice.py",
        "8e44f0f0144378561a00ebc065fdb15a843bc4650e68683bebb6624827731859",
        "cc443bed44c651a47492fc7e2142e3a88fb47627",
    ),
    (
        "cosyvoice/llm/llm.py",
        "6439d57fcf78bcdcad6d31812f3f4b02bd34f513333711ee317d71d1fd14d2de",
        "59ebd48fde1f1b69240391fdac6e2afc1035e123",
    ),
    (
        "cosyvoice/tokenizer/tokenizer.py",
        "94340fc7cdf270c69a3aeb63290c5241044e20714e01fea736f361f9e5a56df2",
        "43fb39a2b543cc7ba4ec95fca9327596c34dcff0",
    ),
];

#[derive(Debug, Clone)]
pub(crate) struct BoundCosyVoice2Llm {
    backbone: LlmBackbone,
    speech: SpeechLmTensors,
}

impl BoundCosyVoice2Llm {
    pub(crate) fn from_gguf(file: &GgufFile) -> Result<Self> {
        validate_metadata(file)?;
        validate_tensor_schema(file)?;
        let llm_config = LlmBackboneConfig {
            vocab_size: VOCAB_SIZE as usize,
            hidden_dim: HIDDEN_DIM as usize,
            n_layer: N_LAYER as usize,
            n_head_q: N_HEAD as usize,
            n_head_kv: N_HEAD_KV as usize,
            ffn_dim: FFN_DIM as usize,
            rope_base: ROPE_BASE,
            rms_norm_eps: RMS_NORM_EPS,
            n_ctx: N_CTX as usize,
        };
        let weights = LlmWeights::from_gguf(file, &llm_config)?;
        let backbone = LlmBackbone::new(llm_config, weights)?;
        let speech = SpeechLmTensors {
            llm_embedding: checked_f32_tensor(
                file,
                "llm_embedding.weight",
                &[2, HIDDEN_DIM as usize],
            )?,
            speech_embedding: checked_f32_tensor(
                file,
                "speech_embedding.weight",
                &[SPEECH_HEAD_SIZE, HIDDEN_DIM as usize],
            )?,
            decoder_weight: checked_f32_tensor(
                file,
                "llm_decoder.weight",
                &[SPEECH_HEAD_SIZE, HIDDEN_DIM as usize],
            )?,
            decoder_bias: checked_f32_tensor(file, "llm_decoder.bias", &[SPEECH_HEAD_SIZE])?,
        };
        let bound = Self { backbone, speech };
        bound.speech_lm()?;
        Ok(bound)
    }

    pub(crate) fn backbone(&self) -> &LlmBackbone {
        &self.backbone
    }

    pub(crate) fn speech_lm(&self) -> Result<SpeechLm<'_>> {
        SpeechLm::new(&self.backbone, self.speech.clone())
    }
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    match file.get(key).and_then(GgufMetadataValue::as_str) {
        Some(value) if value == expected => Ok(()),
        Some(value) => Err(VokraError::ModelLoad(format!(
            "{LABEL}: metadata {key}={value:?}, expected {expected:?}"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "{LABEL}: metadata {key} is missing or not a string"
        ))),
    }
}

fn require_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::U32(value)) if *value == expected => Ok(()),
        Some(value) => Err(VokraError::ModelLoad(format!(
            "{LABEL}: metadata {key}={value:?}, expected UINT32({expected})"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "{LABEL}: metadata {key} is missing or not UINT32"
        ))),
    }
}

fn require_u64(file: &GgufFile, key: &str, expected: u64) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::U64(value)) if *value == expected => Ok(()),
        Some(value) => Err(VokraError::ModelLoad(format!(
            "{LABEL}: metadata {key}={value:?}, expected UINT64({expected})"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "{LABEL}: metadata {key} is missing or not UINT64"
        ))),
    }
}

fn require_f32(file: &GgufFile, key: &str, expected: f32) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::F32(value)) if value.to_bits() == expected.to_bits() => Ok(()),
        Some(value) => Err(VokraError::ModelLoad(format!(
            "{LABEL}: metadata {key}={value:?}, expected FLOAT32({expected})"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "{LABEL}: metadata {key} is missing or not FLOAT32"
        ))),
    }
}

fn require_sha256(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let value = file
        .get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "{LABEL}: metadata {key} is missing or not a string"
            ))
        })?;
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: metadata {key} is not a 64-digit hexadecimal digest"
        )));
    }
    if value != expected {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: metadata {key}={value:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn require_digest_format(file: &GgufFile, key: &str) -> Result<()> {
    let value = file
        .get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "{LABEL}: metadata {key} is missing or not a string"
            ))
        })?;
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: metadata {key} is not a 64-digit hexadecimal digest"
        )));
    }
    Ok(())
}

fn validate_metadata(file: &GgufFile) -> Result<()> {
    for (key, expected) in [
        (chunks::KEY_MODEL_ARCH, ARCH),
        (chunks::KEY_MODEL_NAME, MODEL_NAME),
        (KEY_CATEGORY, CATEGORY),
        (KEY_COMPONENT, COMPONENT),
        (KEY_COMPOSITE_STATUS, COMPOSITE_STATUS),
        (KEY_UPSTREAM_HF, UPSTREAM_HF),
        (KEY_UPSTREAM_REVISION, UPSTREAM_REVISION),
        (KEY_WEIGHT_LICENSE, "permissive"),
        (KEY_LICENSE, "apache-2.0"),
        (KEY_MODEL_ID, MODEL_NAME),
        (KEY_UPSTREAM_COMPONENT_FILE, CHECKPOINT_FILE),
        (KEY_CONFIG_FILE, CONFIG_FILE),
        (KEY_CONFIG_SHA256, CONFIG_SHA256),
        (KEY_CONFIG_BLOB, CONFIG_BLOB),
        (KEY_QWEN_CONFIG_FILE, QWEN_CONFIG_FILE),
        (KEY_QWEN_CONFIG_SHA256, QWEN_CONFIG_SHA256),
        (KEY_QWEN_CONFIG_BLOB, QWEN_CONFIG_BLOB),
        (KEY_SOURCE_REPOSITORY, SOURCE_REPOSITORY),
        (KEY_SOURCE_REVISION, SOURCE_REVISION),
        (KEY_TENSOR_MANIFEST, MANIFEST_SHA256),
        (KEY_SOURCE_LICENSE_PATH, SOURCE_LICENSE_PATH),
        (KEY_SOURCE_LICENSE_SHA256, SOURCE_LICENSE_SHA256),
        (KEY_SOURCE_LICENSE_BLOB, SOURCE_LICENSE_BLOB),
        (KEY_SOURCE_LICENSE_SPDX, SOURCE_LICENSE_SPDX),
        (KEY_PREPARED_STATUS, PREPARED_STATUS),
    ] {
        require_string(file, key, expected)?;
    }
    require_u32(file, KEY_UPSTREAM_COMPONENT_BYTES, CHECKPOINT_BYTES)?;
    require_u32(file, KEY_CONFIG_BYTES, CONFIG_BYTES)?;
    require_u32(file, KEY_QWEN_CONFIG_BYTES, QWEN_CONFIG_BYTES)?;
    require_u32(file, KEY_VOCAB_SIZE, VOCAB_SIZE)?;
    require_u32(file, KEY_HIDDEN_DIM, HIDDEN_DIM)?;
    require_u32(file, KEY_N_LAYER, N_LAYER)?;
    require_u32(file, KEY_FFN_DIM, FFN_DIM)?;
    require_u32(file, KEY_N_HEAD, N_HEAD)?;
    require_u32(file, KEY_N_HEAD_KV, N_HEAD_KV)?;
    require_u32(file, KEY_N_CTX, N_CTX)?;
    require_f32(file, KEY_ROPE_BASE, ROPE_BASE)?;
    require_f32(file, KEY_RMS_NORM_EPS, RMS_NORM_EPS)?;
    require_u64(file, KEY_SOURCE_LICENSE_BYTES, SOURCE_LICENSE_BYTES)?;
    let prepared_bytes = match file.get(KEY_PREPARED_BYTES) {
        Some(GgufMetadataValue::U64(value)) if *value > 0 => *value,
        Some(value) => {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: metadata {KEY_PREPARED_BYTES}={value:?} is not a positive UINT64"
            )));
        }
        None => {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: metadata {KEY_PREPARED_BYTES} is missing or not UINT64"
            )));
        }
    };
    if prepared_bytes == 0 {
        unreachable!("positive check above");
    }
    require_sha256(file, KEY_UPSTREAM_COMPONENT_SHA256, CHECKPOINT_SHA256)?;
    require_digest_format(file, KEY_PREPARED_SHA256)?;
    for (path, sha256, blob) in SOURCE_ROLES {
        require_string(
            file,
            &format!("vokra.cosyvoice2.source.{path}.sha256"),
            sha256,
        )?;
        require_string(
            file,
            &format!("vokra.cosyvoice2.source.{path}.git_blob_sha1"),
            blob,
        )?;
    }
    Ok(())
}

fn expected_tensor_shapes() -> BTreeMap<String, Vec<u64>> {
    let mut expected = BTreeMap::new();
    expected.insert("llm.model.lm_head.weight".to_owned(), vec![151_936, 896]);
    expected.insert(
        "llm.model.model.embed_tokens.weight".to_owned(),
        vec![151_936, 896],
    );
    expected.insert("llm.model.model.norm.weight".to_owned(), vec![896]);
    for layer in 0..N_LAYER as usize {
        let prefix = format!("llm.model.model.layers.{layer}");
        for (suffix, shape) in [
            ("input_layernorm.weight", vec![896]),
            ("mlp.down_proj.weight", vec![896, 4_864]),
            ("mlp.gate_proj.weight", vec![4_864, 896]),
            ("mlp.up_proj.weight", vec![4_864, 896]),
            ("post_attention_layernorm.weight", vec![896]),
            ("self_attn.k_proj.bias", vec![128]),
            ("self_attn.k_proj.weight", vec![128, 896]),
            ("self_attn.o_proj.weight", vec![896, 896]),
            ("self_attn.q_proj.bias", vec![896]),
            ("self_attn.q_proj.weight", vec![896, 896]),
            ("self_attn.v_proj.bias", vec![128]),
            ("self_attn.v_proj.weight", vec![128, 896]),
        ] {
            expected.insert(format!("{prefix}.{suffix}"), shape);
        }
    }
    expected.insert("llm_decoder.bias".to_owned(), vec![6_564]);
    expected.insert("llm_decoder.weight".to_owned(), vec![6_564, 896]);
    expected.insert("llm_embedding.weight".to_owned(), vec![2, 896]);
    expected.insert("speech_embedding.weight".to_owned(), vec![6_564, 896]);
    expected
}

fn canonical_manifest_digest(manifest: &BTreeMap<String, Vec<u64>>) -> [u8; 32] {
    let mut canonical = Vec::new();
    for (name, dimensions) in manifest {
        canonical.extend_from_slice(name.as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(&(dimensions.len() as u64).to_le_bytes());
        for dimension in dimensions {
            canonical.extend_from_slice(&dimension.to_le_bytes());
        }
    }
    sha256_bytes(&canonical)
}

fn validate_tensor_schema(file: &GgufFile) -> Result<()> {
    validate_tensor_infos(file.tensors())
}

fn validate_tensor_infos(infos: &[GgufTensorInfo]) -> Result<()> {
    let expected = expected_tensor_shapes();
    if expected.len() != TENSOR_COUNT || canonical_manifest_digest(&expected) != MANIFEST_DIGEST {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: compiled tensor schema does not match authenticated manifest"
        )));
    }
    if infos.len() != expected.len() {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: tensor count {}, expected {}",
            infos.len(),
            expected.len()
        )));
    }
    let mut seen = BTreeSet::new();
    for info in infos {
        if !seen.insert(info.name.as_str()) {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: duplicate tensor {}",
                info.name
            )));
        }
        let Some(shape) = expected.get(&info.name) else {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: unexpected tensor {}",
                info.name
            )));
        };
        if info.dtype != GgmlType::F32 || info.dimensions != *shape {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: tensor {} dtype {:?}, shape {:?}; expected F32, {:?}",
                info.name, info.dtype, info.dimensions, shape
            )));
        }
    }
    if seen.len() != expected.len() {
        let missing = expected
            .keys()
            .find(|name| !seen.contains(name.as_str()))
            .map_or("unknown", String::as_str);
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: missing tensor {missing}"
        )));
    }
    Ok(())
}

fn checked_f32_tensor(file: &GgufFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| VokraError::ModelLoad(format!("{LABEL}: tensor {name} is missing")))?;
    let expected_shape = shape.iter().map(|&value| value as u64).collect::<Vec<_>>();
    if info.dtype != GgmlType::F32 || info.dimensions != expected_shape {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: tensor {name} dtype {:?}, shape {:?}; expected F32, {:?}",
            info.dtype, info.dimensions, shape
        )));
    }
    let values = file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!("{LABEL}: tensor {name} decode failed: {error}"))
    })?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: tensor {name} contains non-finite values"
        )));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;

    fn synthetic_infos() -> Vec<GgufTensorInfo> {
        expected_tensor_shapes()
            .into_iter()
            .map(|(name, dimensions)| GgufTensorInfo {
                name,
                dimensions,
                dtype: GgmlType::F32,
                offset: 0,
            })
            .collect()
    }

    fn metadata_file(tamper_status: bool) -> GgufFile {
        let mut builder = GgufBuilder::new();
        let prepared_sha = "a".repeat(64);
        for (key, value) in [
            (chunks::KEY_MODEL_ARCH, ARCH),
            (chunks::KEY_MODEL_NAME, MODEL_NAME),
            (KEY_CATEGORY, CATEGORY),
            (KEY_COMPONENT, COMPONENT),
            (KEY_COMPOSITE_STATUS, COMPOSITE_STATUS),
            (KEY_UPSTREAM_HF, UPSTREAM_HF),
            (KEY_UPSTREAM_REVISION, UPSTREAM_REVISION),
            (KEY_WEIGHT_LICENSE, "permissive"),
            (KEY_LICENSE, "apache-2.0"),
            (KEY_MODEL_ID, MODEL_NAME),
            (KEY_UPSTREAM_COMPONENT_FILE, CHECKPOINT_FILE),
            (KEY_UPSTREAM_COMPONENT_SHA256, CHECKPOINT_SHA256),
            (KEY_CONFIG_FILE, CONFIG_FILE),
            (KEY_CONFIG_SHA256, CONFIG_SHA256),
            (KEY_CONFIG_BLOB, CONFIG_BLOB),
            (KEY_QWEN_CONFIG_FILE, QWEN_CONFIG_FILE),
            (KEY_QWEN_CONFIG_SHA256, QWEN_CONFIG_SHA256),
            (KEY_QWEN_CONFIG_BLOB, QWEN_CONFIG_BLOB),
            (KEY_SOURCE_REPOSITORY, SOURCE_REPOSITORY),
            (KEY_SOURCE_REVISION, SOURCE_REVISION),
            (KEY_TENSOR_MANIFEST, MANIFEST_SHA256),
            (KEY_SOURCE_LICENSE_PATH, SOURCE_LICENSE_PATH),
            (KEY_SOURCE_LICENSE_SHA256, SOURCE_LICENSE_SHA256),
            (KEY_SOURCE_LICENSE_BLOB, SOURCE_LICENSE_BLOB),
            (KEY_SOURCE_LICENSE_SPDX, SOURCE_LICENSE_SPDX),
            (
                KEY_PREPARED_STATUS,
                if tamper_status {
                    "NOT_AUTHENTICATED"
                } else {
                    PREPARED_STATUS
                },
            ),
            (KEY_PREPARED_SHA256, prepared_sha.as_str()),
        ] {
            builder.add_string(key, value);
        }
        for (path, sha256, blob) in SOURCE_ROLES {
            builder
                .add_string(&format!("vokra.cosyvoice2.source.{path}.sha256"), sha256)
                .add_string(
                    &format!("vokra.cosyvoice2.source.{path}.git_blob_sha1"),
                    blob,
                );
        }
        builder
            .add_u32(KEY_UPSTREAM_COMPONENT_BYTES, CHECKPOINT_BYTES)
            .add_u32(KEY_CONFIG_BYTES, CONFIG_BYTES)
            .add_u32(KEY_QWEN_CONFIG_BYTES, QWEN_CONFIG_BYTES)
            .add_u32(KEY_VOCAB_SIZE, VOCAB_SIZE)
            .add_u32(KEY_HIDDEN_DIM, HIDDEN_DIM)
            .add_u32(KEY_N_LAYER, N_LAYER)
            .add_u32(KEY_FFN_DIM, FFN_DIM)
            .add_u32(KEY_N_HEAD, N_HEAD)
            .add_u32(KEY_N_HEAD_KV, N_HEAD_KV)
            .add_u32(KEY_N_CTX, N_CTX)
            .add_f32(KEY_ROPE_BASE, ROPE_BASE)
            .add_f32(KEY_RMS_NORM_EPS, RMS_NORM_EPS)
            .add_metadata(
                KEY_SOURCE_LICENSE_BYTES,
                GgufMetadataValue::U64(SOURCE_LICENSE_BYTES),
            )
            .add_metadata(KEY_PREPARED_BYTES, GgufMetadataValue::U64(1));
        GgufFile::parse(builder.to_bytes().expect("metadata GGUF")).expect("parse")
    }

    #[test]
    fn authenticated_axes_derive_exact_295_manifest() {
        let manifest = expected_tensor_shapes();
        assert_eq!(manifest.len(), TENSOR_COUNT);
        assert_eq!(canonical_manifest_digest(&manifest), MANIFEST_DIGEST);
        assert_eq!(
            manifest["llm.model.model.layers.0.self_attn.k_proj.weight"],
            [128, 896]
        );
        assert_eq!(manifest["llm_decoder.weight"], [6_564, 896]);
        assert_eq!(manifest["llm_embedding.weight"], [2, 896]);
    }

    #[test]
    fn manifest_string_matches_digest_array() {
        let rendered = MANIFEST_DIGEST
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(rendered, MANIFEST_SHA256);
    }

    #[test]
    fn wrapper_contract_is_exact_and_finite_checked() {
        assert_eq!(SPEECH_HEAD_SIZE, 6_564);
        let expected = expected_tensor_shapes();
        for (name, shape) in [
            ("llm_embedding.weight", &[2, 896][..]),
            ("speech_embedding.weight", &[6_564, 896][..]),
            ("llm_decoder.weight", &[6_564, 896][..]),
            ("llm_decoder.bias", &[6_564][..]),
        ] {
            assert_eq!(expected[name], shape);
        }
    }

    #[test]
    fn synthetic_schema_accepts_exact_contract_without_payload() {
        validate_tensor_infos(&synthetic_infos()).expect("descriptor-only contract");
    }

    #[test]
    fn synthetic_schema_rejects_extra_missing_shape_and_dtype() {
        let mut extra = synthetic_infos();
        extra[0].name = "llm.unexpected.weight".to_owned();
        assert!(
            validate_tensor_infos(&extra)
                .expect_err("extra tensor")
                .to_string()
                .contains("unexpected")
        );

        let mut missing = synthetic_infos();
        missing.pop();
        assert!(
            validate_tensor_infos(&missing)
                .expect_err("missing tensor")
                .to_string()
                .contains("tensor count")
        );

        let mut wrong_shape = synthetic_infos();
        wrong_shape[0].dimensions = vec![1];
        assert!(
            validate_tensor_infos(&wrong_shape)
                .expect_err("wrong shape")
                .to_string()
                .contains("shape")
        );

        let mut wrong_dtype = synthetic_infos();
        wrong_dtype[0].dtype = GgmlType::F16;
        assert!(
            validate_tensor_infos(&wrong_dtype)
                .expect_err("wrong dtype")
                .to_string()
                .contains("dtype")
        );
    }

    #[test]
    fn metadata_tamper_and_prepared_status_are_fail_closed() {
        validate_metadata(&metadata_file(false)).expect("converter metadata contract");
        let error = validate_metadata(&metadata_file(true)).expect_err("tampered status");
        assert!(error.to_string().contains("authentication_status"));
    }
}
