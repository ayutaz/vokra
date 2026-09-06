//! Strict CosyVoice2 LLM component converter.
//!
//! The full CosyVoice2 release is composite and remains `INSPECTION_ONLY` at
//! the public composite entry point. The side-car-aware entry point accepts
//! only the authenticated, prepared LLM safetensors component and preserves
//! its runtime tensor names verbatim. Flow, HiFT, tokenizer, and speaker
//! components are not synthesized or silently omitted.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, GgufMetadataValue, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "cosyvoice2";
pub const NAME: &str = "cosyvoice2-0.5b";
pub const CATEGORY: &str = "llm";
pub const UPSTREAM_HF: &str = "FunAudioLLM/CosyVoice2-0.5B";
pub const UPSTREAM_REVISION: &str = "eec1ae6c79877dbd9379285cf8789c9e0879293d";
pub const CHECKPOINT_FILE: &str = "llm.pt";
pub const CHECKPOINT_BYTES: u64 = 2_023_316_821;
pub const CHECKPOINT_SHA256: &str =
    "b144ef55b51ce8cfb79a73c90dbba0bdaba4e451c0ebcfab20f769264f84a608";
pub const CONFIG_FILE: &str = "cosyvoice2.yaml";
pub const CONFIG_BYTES: usize = 7_330;
pub const CONFIG_SHA256: &str = "0af2c0d010c477187c39f3e8fd5f1ae2e4e6f90ad03ba37c10ed6c6a87b05959";
pub const CONFIG_GIT_BLOB_SHA1: &str = "bc19267bbfd373c9a760b7667a74349ddd487db1";
pub const QWEN_CONFIG_FILE: &str = "CosyVoice-BlankEN/config.json";
pub const QWEN_CONFIG_BYTES: usize = 659;
pub const QWEN_CONFIG_SHA256: &str =
    "168aa1bd401abc3bc262ba15ba4e499627a8b4e006e9d050b47c22de20660185";
pub const QWEN_CONFIG_GIT_BLOB_SHA1: &str = "463b055262b6c66c4629a74a4b300bfe2ed31d3c";
pub const SOURCE_REPOSITORY: &str = "https://github.com/FunAudioLLM/CosyVoice.git";
pub const SOURCE_REVISION: &str = "8555549e882236e6541748b1042d95693caa82ba";
pub const TENSOR_COUNT: usize = 295;
pub const TENSOR_MANIFEST_SHA256: &str =
    "07cf10ae088c27a7c88e1c08fb231d00b01bba0c13f312a74d2fd4b35403bda2";

const SOURCE_LICENSE_PATH: &str = "LICENSE";
const SOURCE_LICENSE_BYTES: u64 = 11_357;
const SOURCE_LICENSE_SHA256: &str =
    "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4";
const SOURCE_LICENSE_GIT_BLOB_SHA1: &str = "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64";
const SOURCE_LICENSE_SPDX: &str = "Apache-2.0";

const KEY_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct DerivedHparams {
    pub(crate) vocab_size: u32,
    pub(crate) hidden_dim: u32,
    pub(crate) n_layer: u32,
    pub(crate) ffn_dim: u32,
    pub(crate) n_head: u32,
    pub(crate) n_head_kv: u32,
    pub(crate) n_ctx: u32,
    pub(crate) has_attn_bias: bool,
}

#[derive(Debug, Default)]
pub(crate) struct CosyVoice2Report {
    pub(crate) written: usize,
    pub(crate) skipped_non_float: usize,
    #[allow(dead_code)]
    pub(crate) bf16_passthrough: usize,
    pub(crate) derived: Option<DerivedHparams>,
    pub(crate) tokenizer_embedded: bool,
    pub(crate) notes: Vec<String>,
}

/// Summary of a staged standalone LLM component conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ConvertCosyVoice2LlmReport {
    /// Number of authenticated F32 tensors copied verbatim.
    pub written: usize,
    /// Number of GGUF metadata entries stamped by the converter.
    pub metadata_count: usize,
    /// Serialized GGUF size in bytes.
    pub output_bytes: u64,
}

#[allow(dead_code)]
pub(crate) struct TokenizerFiles<'a> {
    pub(crate) vocab_json: &'a [u8],
    pub(crate) merges_txt: &'a [u8],
}

fn inspection_only() -> ConvertError {
    ConvertError::Usage(
        "CosyVoice2 composite conversion is INSPECTION_ONLY: flow, HiFT, speech-tokenizer, and speaker components are not bound; no full TTS output was written"
            .to_owned(),
    )
}

pub(crate) fn convert(_bytes: Vec<u8>) -> Result<(GgufBuilder, CosyVoice2Report), ConvertError> {
    Err(inspection_only())
}

/// Shape-only legacy surface remains fail-closed; a real component requires
/// the exact config sidecar through this config-aware path.
#[allow(dead_code)]
pub(crate) fn convert_with_config(
    _bytes: Vec<u8>,
    _config_json: Option<&[u8]>,
) -> Result<(GgufBuilder, CosyVoice2Report), ConvertError> {
    Err(inspection_only())
}

pub(crate) fn convert_with_config_and_tokenizer(
    _bytes: Vec<u8>,
    _config_json: Option<&[u8]>,
    _tokenizer: Option<TokenizerFiles<'_>>,
) -> Result<(GgufBuilder, CosyVoice2Report), ConvertError> {
    Err(inspection_only())
}

/// Converts the separately authenticated CosyVoice2 LLM component from
/// prepared safetensors and the exact YAML plus Qwen config sidecars, publishing only to an
/// absent output path. The raw PyTorch checkpoint is never accepted here;
/// its identity is retained as provenance from the VAST preparation gate.
#[allow(dead_code)]
pub(crate) fn convert_cosyvoice2_llm_file(
    input: &Path,
    config: &Path,
    qwen_config: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<ConvertCosyVoice2LlmReport, ConvertError> {
    require_explicit_license(license)?;
    require_regular_file(input, "prepared LLM safetensors")?;
    require_regular_file(config, CONFIG_FILE)?;
    require_regular_file(qwen_config, QWEN_CONFIG_FILE)?;
    require_absent_output(output)?;
    let input_bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let config_bytes = std::fs::read(config).map_err(ConvertError::Io)?;
    let qwen_config_bytes = std::fs::read(qwen_config).map_err(ConvertError::Io)?;
    let (builder, report) = convert_component(
        input_bytes,
        Some(&config_bytes),
        Some(&qwen_config_bytes),
        None,
    )?;
    let output_bytes = builder.to_bytes()?;
    write_no_replace(output, &output_bytes).map_err(ConvertError::Io)?;
    Ok(ConvertCosyVoice2LlmReport {
        written: report.written,
        metadata_count: builder.metadata_count(),
        output_bytes: output_bytes.len() as u64,
    })
}

/// Validate the VAST handoff paths before reading any payload. Symlink paths
/// are rejected at this handoff boundary; callers must keep the reviewed
/// regular files in place for the conversion operation.
fn require_regular_file(path: &Path, label: &str) -> Result<(), ConvertError> {
    if !path.is_absolute() {
        return Err(ConvertError::Usage(format!(
            "{ARCH} LLM: {label} path must be absolute"
        )));
    }
    let metadata = std::fs::symlink_metadata(path).map_err(ConvertError::Io)?;
    let file_type = metadata.file_type();
    if !file_type.is_file() || file_type.is_symlink() {
        return Err(ConvertError::Usage(format!(
            "{ARCH} LLM: {label} must be a regular non-symlink file"
        )));
    }
    Ok(())
}

fn require_absent_output(path: &Path) -> Result<(), ConvertError> {
    if !path.is_absolute() {
        return Err(ConvertError::Usage(
            "cosyvoice2 LLM: output path must be absolute".to_owned(),
        ));
    }
    match std::fs::symlink_metadata(path) {
        Ok(_) => Err(ConvertError::Usage(
            "cosyvoice2 LLM: output path must be absent (no replacement)".to_owned(),
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(ConvertError::Io(error)),
    }
}

fn require_explicit_license(license: Option<&str>) -> Result<(), ConvertError> {
    match license {
        Some(value) if value.eq_ignore_ascii_case("apache-2.0") => Ok(()),
        Some(value) => Err(ConvertError::Usage(format!(
            "{ARCH} LLM: explicit Apache-2.0 license attestation required; got `{value}`"
        ))),
        None => Err(ConvertError::Usage(format!(
            "{ARCH} LLM: explicit Apache-2.0 license attestation is required"
        ))),
    }
}

/// Publish bytes to an absent path without replacement. A same-filesystem
/// hard link is the std-only atomic no-replace primitive; the temporary
/// sibling is cleaned up on both success and failure.
fn write_no_replace(output: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let name = output.file_name().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "output has no file name")
    })?;
    let mut temporary = None;
    for attempt in 0..100u32 {
        let candidate = parent.join(format!(
            ".{}.vokra-cosyvoice2-llm-{}-{}",
            name.to_string_lossy(),
            std::process::id(),
            attempt
        ));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                temporary = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    let Some((temporary_path, mut file)) = temporary else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not allocate temporary CosyVoice2 LLM output",
        ));
    };
    let operation = (|| {
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()?;
        drop(file);
        std::fs::hard_link(&temporary_path, output)
    })();
    let cleanup = std::fs::remove_file(&temporary_path);
    match operation {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = cleanup;
            Err(error)
        }
    }
}

fn convert_component(
    bytes: Vec<u8>,
    config: Option<&[u8]>,
    qwen_config: Option<&[u8]>,
    tokenizer: Option<TokenizerFiles<'_>>,
) -> Result<(GgufBuilder, CosyVoice2Report), ConvertError> {
    let config = config.ok_or_else(|| {
        ConvertError::Usage(
            "CosyVoice2 LLM component requires the exact cosyvoice2.yaml sidecar; no output was written"
                .to_owned(),
        )
    })?;
    validate_config(config)?;
    let qwen_config = qwen_config.ok_or_else(|| {
        ConvertError::Usage(
            "CosyVoice2 LLM component requires the exact Qwen config.json sidecar; no output was written"
                .to_owned(),
        )
    })?;
    validate_qwen_config(qwen_config)?;
    let prepared_bytes = bytes.len() as u64;
    let prepared_sha256 =
        crate::models::canary_1b_flash::hex(&crate::models::canary_1b_flash::sha256(&bytes));
    let st = SafetensorsFile::parse(bytes).map_err(ConvertError::from)?;
    validate_manifest(&st)?;
    let embedding = st
        .tensor_info("llm.model.model.embed_tokens.weight")
        .expect("validated manifest includes token embedding");
    let head = st
        .tensor_info("llm.model.lm_head.weight")
        .expect("validated manifest includes lm head");
    if st.tensor_bytes(embedding) != st.tensor_bytes(head) {
        return Err(ConvertError::Parse(
            "cosyvoice2 LLM: lm_head.weight is not byte-identical to embed_tokens.weight; tied-head contract rejected"
                .to_owned(),
        ));
    }

    let mut builder = GgufBuilder::new();
    builder
        .add_string(chunks::KEY_MODEL_ARCH, ARCH)
        .add_string(chunks::KEY_MODEL_NAME, NAME)
        .add_string(KEY_CATEGORY, CATEGORY)
        .add_string(KEY_COMPONENT, "llm")
        .add_string(KEY_COMPOSITE_STATUS, "INSPECTION_ONLY")
        .add_string(KEY_UPSTREAM_HF, UPSTREAM_HF)
        .add_string(KEY_UPSTREAM_REVISION, UPSTREAM_REVISION)
        .add_string(KEY_UPSTREAM_COMPONENT_SHA256, CHECKPOINT_SHA256)
        .add_string(KEY_UPSTREAM_COMPONENT_FILE, CHECKPOINT_FILE)
        .add_u32(KEY_UPSTREAM_COMPONENT_BYTES, CHECKPOINT_BYTES as u32)
        .add_string(KEY_CONFIG_FILE, CONFIG_FILE)
        .add_u32(KEY_CONFIG_BYTES, CONFIG_BYTES as u32)
        .add_string(KEY_CONFIG_SHA256, CONFIG_SHA256)
        .add_string(KEY_CONFIG_BLOB, CONFIG_GIT_BLOB_SHA1)
        .add_string(KEY_QWEN_CONFIG_FILE, QWEN_CONFIG_FILE)
        .add_u32(KEY_QWEN_CONFIG_BYTES, QWEN_CONFIG_BYTES as u32)
        .add_string(KEY_QWEN_CONFIG_SHA256, QWEN_CONFIG_SHA256)
        .add_string(KEY_QWEN_CONFIG_BLOB, QWEN_CONFIG_GIT_BLOB_SHA1)
        .add_string(KEY_SOURCE_REPOSITORY, SOURCE_REPOSITORY)
        .add_string(KEY_SOURCE_REVISION, SOURCE_REVISION)
        .add_string(KEY_TENSOR_MANIFEST, TENSOR_MANIFEST_SHA256)
        .add_string(KEY_SOURCE_LICENSE_PATH, SOURCE_LICENSE_PATH)
        .add_string(KEY_SOURCE_LICENSE_SHA256, SOURCE_LICENSE_SHA256)
        .add_string(KEY_SOURCE_LICENSE_BLOB, SOURCE_LICENSE_GIT_BLOB_SHA1)
        .add_string(KEY_SOURCE_LICENSE_SPDX, SOURCE_LICENSE_SPDX)
        .add_metadata(
            KEY_SOURCE_LICENSE_BYTES,
            GgufMetadataValue::U64(SOURCE_LICENSE_BYTES),
        )
        .add_string(KEY_PREPARED_SHA256, &prepared_sha256)
        .add_string(
            KEY_PREPARED_STATUS,
            "PREPARED_INPUT_DIGEST_RECORDED_NOT_PINNED",
        )
        .add_metadata(KEY_PREPARED_BYTES, GgufMetadataValue::U64(prepared_bytes))
        .add_u32(KEY_VOCAB_SIZE, 151_936)
        .add_u32(KEY_HIDDEN_DIM, 896)
        .add_u32(KEY_N_LAYER, 24)
        .add_u32(KEY_FFN_DIM, 4_864)
        .add_u32(KEY_N_HEAD, 14)
        .add_u32(KEY_N_HEAD_KV, 2)
        .add_u32(KEY_N_CTX, 32_768)
        .add_f32(KEY_ROPE_BASE, 1_000_000.0)
        .add_f32(KEY_RMS_NORM_EPS, 1.0e-6);
    vokra_core::stamp_provenance(
        &mut builder,
        LicenseClass::Permissive,
        "apache-2.0",
        Some(NAME),
        Some(SOURCE_REPOSITORY),
    );
    for (path, sha256, blob) in SOURCE_ROLES {
        builder
            .add_string(&format!("vokra.cosyvoice2.source.{path}.sha256"), sha256)
            .add_string(
                &format!("vokra.cosyvoice2.source.{path}.git_blob_sha1"),
                blob,
            );
    }
    for tensor in st.tensors() {
        builder.add_tensor(
            &tensor.name,
            GgmlType::F32,
            tensor.shape.clone(),
            st.tensor_bytes(tensor).to_vec(),
        )?;
    }
    let mut report = CosyVoice2Report {
        written: TENSOR_COUNT,
        derived: Some(DerivedHparams {
            vocab_size: 151_936,
            hidden_dim: 896,
            n_layer: 24,
            ffn_dim: 4_864,
            n_head: 14,
            n_head_kv: 2,
            n_ctx: 32_768,
            has_attn_bias: true,
        }),
        ..CosyVoice2Report::default()
    };
    if tokenizer.is_some() {
        report.notes.push(
            "tokenizer sidecars are not part of the standalone LLM component; they were not embedded"
                .to_owned(),
        );
    }
    Ok((builder, report))
}

fn validate_config(config: &[u8]) -> Result<(), ConvertError> {
    if config.len() != CONFIG_BYTES {
        return Err(ConvertError::Parse(format!(
            "cosyvoice2 LLM: {CONFIG_FILE} is {} bytes, expected {CONFIG_BYTES}",
            config.len()
        )));
    }
    let sha256 =
        crate::models::canary_1b_flash::hex(&crate::models::canary_1b_flash::sha256(config));
    let mut blob = format!("blob {}\0", config.len()).into_bytes();
    blob.extend_from_slice(config);
    let blob_sha1 = hex_bytes(&sha1(&blob));
    if sha256 != CONFIG_SHA256 || blob_sha1 != CONFIG_GIT_BLOB_SHA1 {
        return Err(ConvertError::Parse(format!(
            "cosyvoice2 LLM: config identity mismatch (sha256={sha256}, blob_sha1={blob_sha1})"
        )));
    }
    Ok(())
}

fn validate_qwen_config(config: &[u8]) -> Result<(), ConvertError> {
    if config.len() != QWEN_CONFIG_BYTES {
        return Err(ConvertError::Parse(format!(
            "cosyvoice2 LLM: {QWEN_CONFIG_FILE} is {} bytes, expected {QWEN_CONFIG_BYTES}",
            config.len()
        )));
    }
    let sha256 =
        crate::models::canary_1b_flash::hex(&crate::models::canary_1b_flash::sha256(config));
    let mut blob = format!("blob {}\0", config.len()).into_bytes();
    blob.extend_from_slice(config);
    let blob_sha1 = hex_bytes(&sha1(&blob));
    if sha256 != QWEN_CONFIG_SHA256 || blob_sha1 != QWEN_CONFIG_GIT_BLOB_SHA1 {
        return Err(ConvertError::Parse(format!(
            "cosyvoice2 LLM: Qwen config identity mismatch (sha256={sha256}, blob_sha1={blob_sha1})"
        )));
    }
    Ok(())
}

fn validate_manifest(st: &SafetensorsFile) -> Result<(), ConvertError> {
    let expected = expected_tensor_shapes();
    let expected_digest = crate::models::canary_1b_flash::hex(
        &crate::models::canary_1b_flash::manifest_sha256(&expected),
    );
    if expected_digest != TENSOR_MANIFEST_SHA256 {
        return Err(ConvertError::Parse(format!(
            "cosyvoice2 LLM: internal manifest drift (sha256={expected_digest})"
        )));
    }
    if st.tensors().len() != expected.len() {
        return Err(ConvertError::Parse(format!(
            "cosyvoice2 LLM: authenticated manifest requires {} tensors, input has {}",
            expected.len(),
            st.tensors().len()
        )));
    }
    let mut seen = BTreeSet::new();
    for tensor in st.tensors() {
        if !seen.insert(tensor.name.as_str()) {
            return Err(ConvertError::Parse(format!(
                "cosyvoice2 LLM: duplicate tensor `{}`",
                tensor.name
            )));
        }
        let Some(shape) = expected.get(tensor.name.as_str()) else {
            return Err(ConvertError::Parse(format!(
                "cosyvoice2 LLM: unexpected tensor `{}`",
                tensor.name
            )));
        };
        if tensor.dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "cosyvoice2 LLM: `{}` dtype {:?} != F32",
                tensor.name, tensor.dtype
            )));
        }
        if tensor.shape != *shape {
            return Err(ConvertError::Parse(format!(
                "cosyvoice2 LLM: `{}` shape {:?} != {:?}",
                tensor.name, tensor.shape, shape
            )));
        }
    }
    if seen.len() != expected.len() {
        let missing: Vec<_> = expected
            .keys()
            .filter(|name| !seen.contains(name.as_str()))
            .collect();
        return Err(ConvertError::Parse(format!(
            "cosyvoice2 LLM: missing authenticated tensors {missing:?}"
        )));
    }
    let actual = st
        .tensors()
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor.shape.clone()))
        .collect::<BTreeMap<_, _>>();
    let actual_digest = crate::models::canary_1b_flash::hex(
        &crate::models::canary_1b_flash::manifest_sha256(&actual),
    );
    if actual_digest != TENSOR_MANIFEST_SHA256 {
        return Err(ConvertError::Parse(format!(
            "cosyvoice2 LLM: input manifest digest {actual_digest} != pinned {TENSOR_MANIFEST_SHA256}"
        )));
    }
    Ok(())
}

fn expected_tensor_shapes() -> BTreeMap<String, Vec<u64>> {
    let mut expected = BTreeMap::new();
    expected.insert("llm.model.lm_head.weight".into(), vec![151_936, 896]);
    expected.insert(
        "llm.model.model.embed_tokens.weight".into(),
        vec![151_936, 896],
    );
    expected.insert("llm.model.model.norm.weight".into(), vec![896]);
    for layer in 0..24 {
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
    expected.insert("llm_decoder.bias".into(), vec![6_564]);
    expected.insert("llm_decoder.weight".into(), vec![6_564, 896]);
    expected.insert("llm_embedding.weight".into(), vec![2, 896]);
    expected.insert("speech_embedding.weight".into(), vec![6_564, 896]);
    expected
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h = [
        0x67452301u32,
        0xefcdab89,
        0x98badcfe,
        0x10325476,
        0xc3d2e1f0,
    ];
    let bit_len = (data.len() as u64) * 8;
    let mut padded = data.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in padded.chunks_exact(64) {
        let mut words = [0u32; 80];
        for index in 0..16 {
            words[index] = u32::from_be_bytes(
                chunk[index * 4..index * 4 + 4]
                    .try_into()
                    .expect("four-byte SHA-1 word"),
            );
        }
        for index in 16..80 {
            words[index] =
                (words[index - 3] ^ words[index - 8] ^ words[index - 14] ^ words[index - 16])
                    .rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (index, word) in words.iter().enumerate() {
            let (f, k) = match index {
                0..=19 => ((b & c) | (!b & d), 0x5a827999),
                20..=39 => (b ^ c ^ d, 0x6ed9eba1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1bbcdc),
                _ => (b ^ c ^ d, 0xca62c1d6),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    let mut output = [0u8; 20];
    for (index, value) in h.into_iter().enumerate() {
        output[index * 4..index * 4 + 4].copy_from_slice(&value.to_be_bytes());
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    #[test]
    fn authenticated_manifest_has_exact_295_tensor_contract() {
        let manifest = expected_tensor_shapes();
        assert_eq!(manifest.len(), TENSOR_COUNT);
        assert_eq!(
            crate::models::canary_1b_flash::hex(&crate::models::canary_1b_flash::manifest_sha256(
                &manifest
            )),
            TENSOR_MANIFEST_SHA256
        );
        assert_eq!(
            manifest["llm.model.model.layers.0.self_attn.k_proj.weight"],
            [128, 896]
        );
        assert_eq!(manifest["llm_decoder.weight"], [6_564, 896]);
    }

    #[test]
    fn wrong_or_partial_config_fails_closed() {
        assert!(validate_config(&[]).is_err());
        let mut wrong = vec![0u8; CONFIG_BYTES];
        wrong[0] = 1;
        assert!(validate_config(&wrong).is_err());
    }

    #[test]
    fn sha1_git_blob_vector_is_authenticated() {
        let mut blob = b"blob 3\0".to_vec();
        blob.extend_from_slice(b"abc");
        assert_eq!(
            hex_bytes(&sha1(&blob)),
            "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f"
        );
    }

    #[test]
    fn full_composite_surface_remains_inspection_only() {
        let error = convert(Vec::new()).expect_err("composite must remain blocked");
        assert!(error.to_string().contains("INSPECTION_ONLY"));
    }

    #[test]
    fn standalone_component_requires_explicit_apache_attestation() {
        assert!(require_explicit_license(None).is_err());
        assert!(require_explicit_license(Some("mit")).is_err());
        require_explicit_license(Some("Apache-2.0")).expect("fixed license is accepted");
    }

    #[test]
    fn standalone_output_is_no_replace() {
        let root = std::env::temp_dir().join(format!(
            "vokra-cosyvoice2-llm-atomic-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir(&root).expect("temporary test directory");

        let existing = root.join("existing.gguf");
        std::fs::write(&existing, b"original").expect("seed existing output");
        let error = write_no_replace(&existing, b"replacement").expect_err("must reject overwrite");
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&existing).unwrap(), b"original");

        let fresh = root.join("fresh.gguf");
        write_no_replace(&fresh, b"complete").expect("publish fresh output");
        assert_eq!(std::fs::read(&fresh).unwrap(), b"complete");
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 2);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn standalone_validation_failure_never_creates_or_replaces_output() {
        let root = std::env::temp_dir().join(format!(
            "vokra-cosyvoice2-llm-validation-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir(&root).expect("temporary test directory");
        let input = root.join("llm.safetensors");
        let config = root.join(CONFIG_FILE);
        let qwen_config = root.join("qwen_config.json");
        let output = root.join("model.gguf");
        std::fs::write(&input, b"not a safetensors checkpoint").expect("seed input");
        std::fs::write(&config, b"wrong config").expect("seed config");
        std::fs::write(&qwen_config, b"wrong qwen config").expect("seed Qwen config");

        std::fs::write(&output, b"original").expect("seed output");
        let error =
            convert_cosyvoice2_llm_file(&input, &config, &qwen_config, &output, Some("apache-2.0"))
                .expect_err("wrong config must fail before publication");
        let _ = error;
        assert_eq!(std::fs::read(&output).unwrap(), b"original");

        std::fs::remove_file(&output).expect("remove test output");
        let _ =
            convert_cosyvoice2_llm_file(&input, &config, &qwen_config, &output, Some("apache-2.0"));
        assert!(!output.exists(), "validation failure must leave no output");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn vast_handoff_paths_fail_closed_for_relative_missing_symlink_and_output() {
        let root =
            std::env::temp_dir().join(format!("vokra-cosyvoice2-llm-paths-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir(&root).expect("temporary test directory");
        let regular = root.join("prepared.safetensors");
        std::fs::write(&regular, b"placeholder").expect("regular input");
        assert!(require_regular_file(Path::new("relative.safetensors"), "input").is_err());
        assert!(require_regular_file(&root.join("missing"), "input").is_err());
        #[cfg(unix)]
        {
            let link = root.join("link.safetensors");
            std::os::unix::fs::symlink(&regular, &link).expect("symlink");
            assert!(require_regular_file(&link, "input").is_err());
        }
        let existing = root.join("existing.gguf");
        std::fs::write(&existing, b"keep").expect("existing output");
        assert!(require_absent_output(&existing).is_err());
        assert!(require_absent_output(Path::new("relative.gguf")).is_err());
        assert!(require_absent_output(&root.join("new.gguf")).is_ok());
        let _ = std::fs::remove_dir_all(root);
    }

    fn required_vast_path(name: &str) -> std::path::PathBuf {
        let value = std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"));
        let path = std::path::PathBuf::from(value);
        require_regular_file(&path, name).unwrap_or_else(|error| panic!("{name}: {error}"));
        path
    }

    fn required_vast_license(name: &str) -> String {
        let license = std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"));
        require_explicit_license(Some(&license)).unwrap_or_else(|error| panic!("{name}: {error}"));
        license
    }

    fn validate_vast_prepared_manifest(path: &Path, component: &str, input: &Path) {
        let bytes = std::fs::read(path).expect("prepared manifest");
        let root = crate::json::parse(&bytes).expect("prepared manifest JSON");
        fn string<'a>(object: &'a vokra_core::json::JsonValue, key: &str) -> Option<&'a str> {
            object
                .get(key)
                .and_then(vokra_core::json::JsonValue::as_str)
        }
        assert_eq!(
            string(&root, "format"),
            Some("vokra-cosyvoice2-component-prepared-safetensors-v1")
        );
        assert_eq!(string(&root, "status"), Some("PREPARED_SAFETENSORS_READY"));
        assert_eq!(string(&root, "component"), Some(component));
        let output_record = root.get("output").expect("prepared output record");
        assert_eq!(string(output_record, "path"), input.to_str());
        assert_eq!(
            output_record
                .get("bytes")
                .and_then(vokra_core::json::JsonValue::as_u64),
            Some(std::fs::metadata(input).unwrap().len())
        );
        let sha = crate::models::canary_1b_flash::hex(&crate::models::canary_1b_flash::sha256(
            &std::fs::read(input).expect("prepared input"),
        ));
        assert_eq!(string(output_record, "sha256"), Some(sha.as_str()));
        let execution = root.get("execution").expect("execution contract");
        assert_eq!(string(execution, "model_execution"), Some("NOT_RUN"));
        assert_eq!(string(execution, "torch_import"), Some("NOT_RUN"));
        assert_eq!(string(execution, "publication"), Some("NO_UPLOAD"));
    }

    /// VAST-only: converts one authenticated prepared component and leaves
    /// bind-only verification to the matching model-crate ignored test.
    #[test]
    #[ignore = "requires the authenticated VAST prepared CosyVoice2 LLM artifact"]
    fn vast_real_prepared_llm_conversion() {
        let input = required_vast_path("VOKRA_COSYVOICE2_LLM_PREPARED");
        let config = required_vast_path("VOKRA_COSYVOICE2_LLM_CONFIG");
        let qwen_config = required_vast_path("VOKRA_COSYVOICE2_LLM_QWEN_CONFIG");
        let manifest = required_vast_path("VOKRA_COSYVOICE2_LLM_PREPARED_MANIFEST");
        validate_vast_prepared_manifest(&manifest, "llm", &input);
        let license = required_vast_license("VOKRA_COSYVOICE2_LLM_LICENSE");
        let output = std::path::PathBuf::from(
            std::env::var("VOKRA_COSYVOICE2_LLM_OUTPUT")
                .expect("VOKRA_COSYVOICE2_LLM_OUTPUT is required"),
        );
        require_absent_output(&output).expect("LLM output must be an absent absolute path");
        let report =
            convert_cosyvoice2_llm_file(&input, &config, &qwen_config, &output, Some(&license))
                .expect("VAST prepared LLM conversion");
        let file = GgufFile::open(&output).expect("converted LLM GGUF");
        assert_eq!(file.tensors().len(), TENSOR_COUNT);
        assert_eq!(report.written, TENSOR_COUNT);
        assert_eq!(
            report.output_bytes,
            std::fs::metadata(&output).unwrap().len()
        );
        assert_eq!(
            file.get(KEY_COMPONENT).and_then(GgufMetadataValue::as_str),
            Some("llm")
        );
        assert_eq!(
            file.get(KEY_COMPOSITE_STATUS)
                .and_then(GgufMetadataValue::as_str),
            Some("INSPECTION_ONLY")
        );
        assert!(
            manifest.is_file(),
            "manifest was validated as a regular file"
        );
    }
}
