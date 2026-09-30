//! Strict native bridge for the official VibeVoice Realtime voice-preset cache.
//!
//! The offline exporter reads Microsoft's fixed Carter `.pt` with the
//! upstream `weights_only=True` safe-global allowlist and writes F32
//! safetensors plus a deterministic manifest.  This module consumes those
//! bytes only; it does not load pickle, reconstruct a `DynamicCache`, execute
//! a model, or make any voice-consent/publication claim.

use std::collections::BTreeSet;
use vokra_core::gguf::GgmlType;
use vokra_core::json::JsonValue;
use vokra_core::safetensors::SafetensorsFile;
use vokra_core::{Result, VokraError};

use crate::strict_checkpoint::sha256_bytes;

use super::language::{
    VibeVoiceRealtimeKvCacheLayer, VibeVoiceRealtimeKvCacheSnapshot, VibeVoiceRealtimeLanguage,
    VibeVoiceRealtimeLanguageKvCachePair,
};

const FORMAT: &str = "vokra-vibevoice-realtime-0.5b-preset-v1";
const SOURCE_REVISION: &str = "94da20d98b2fa7688e9cbfaf7692ddb4954f7600";
const MODEL_REVISION: &str = "6bce5f06044837fe6d2c5d7a71a84f0416bd57e4";
const SOURCE_PATH: &str = "demo/voices/streaming_model/en-Carter_man.pt";
const SOURCE_BYTES: u64 = 4_256_002;
const SOURCE_BLOB_SHA1: &str = "1d795ef667e6641eecb8b22452bb853b089bfdbe";
const SOURCE_PAYLOAD_SHA256: &str =
    "a7bfdf1cd4939c22469bcfc6f427ae9c4467b3df46c2c14303a39c294cfc6897";
const HIDDEN: usize = 896;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const MAX_POSITIONS: usize = 8_192;

/// One of the four independently cached official preset branches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VibeVoiceRealtimePresetBranch {
    /// Positive text language-model output and cache.
    Lm,
    /// Positive TTS language-model output and cache.
    TtsLm,
    /// Negative text language-model output and cache.
    NegLm,
    /// Negative TTS language-model output and cache.
    NegTtsLm,
}

impl VibeVoiceRealtimePresetBranch {
    /// Returns the manifest/safetensors branch key.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Lm => "lm",
            Self::TtsLm => "tts_lm",
            Self::NegLm => "neg_lm",
            Self::NegTtsLm => "neg_tts_lm",
        }
    }

    const fn layer_count(self) -> usize {
        match self {
            Self::Lm | Self::NegLm => 4,
            Self::TtsLm | Self::NegTtsLm => 20,
        }
    }
}

/// An owned native-layout cache layer from one voice preset branch.
#[derive(Debug, Clone, PartialEq)]
pub struct VibeVoiceRealtimePresetCacheLayer {
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl VibeVoiceRealtimePresetCacheLayer {
    /// Native post-RoPE keys in flattened `[position, kv-head, head-dim]` order.
    #[must_use]
    pub fn keys(&self) -> &[f32] {
        &self.keys
    }

    /// Native values in flattened `[position, kv-head, head-dim]` order.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// One complete hidden-state/cache branch from the official preset.
#[derive(Debug, Clone, PartialEq)]
pub struct VibeVoiceRealtimePresetOutput {
    hidden_rows: usize,
    cache_position: usize,
    hidden: Vec<f32>,
    layers: Vec<VibeVoiceRealtimePresetCacheLayer>,
}

impl VibeVoiceRealtimePresetOutput {
    /// All hidden rows in row-major `[hidden_rows, 896]` order.
    #[must_use]
    pub fn hidden(&self) -> &[f32] {
        &self.hidden
    }

    /// Number of hidden rows. This is independent of [`Self::cache_position`].
    #[must_use]
    pub const fn hidden_rows(&self) -> usize {
        self.hidden_rows
    }

    /// Number of positions represented by every imported cache layer.
    #[must_use]
    pub const fn cache_position(&self) -> usize {
        self.cache_position
    }

    /// Native cache layers in their authenticated upstream order.
    #[must_use]
    pub fn layers(&self) -> &[VibeVoiceRealtimePresetCacheLayer] {
        &self.layers
    }
}

/// Four-output, structurally authenticated Realtime voice-preset cache.
#[derive(Debug, Clone, PartialEq)]
pub struct VibeVoiceRealtimePresetCache {
    outputs: [VibeVoiceRealtimePresetOutput; 4],
    manifest_sha256: [u8; 32],
    safetensors_sha256: [u8; 32],
}

impl VibeVoiceRealtimePresetCache {
    /// Parses and authenticates an exported preset cache.
    ///
    /// `expected_manifest_sha256` is supplied by an external reviewed packet;
    /// the value declared by the manifest is never treated as its own proof.
    /// The safetensors payload, exact four branch keys, layer counts, native
    /// shapes, finite values, and per-tensor payload hashes are all checked.
    pub fn from_bytes(
        safetensors_bytes: &[u8],
        manifest_bytes: &[u8],
        expected_manifest_sha256: &str,
    ) -> Result<Self> {
        let expected_manifest = parse_hex_digest(
            expected_manifest_sha256,
            "vibevoice realtime preset expected manifest SHA-256",
        )?;
        let actual_manifest = sha256_bytes(manifest_bytes);
        if actual_manifest != expected_manifest {
            return Err(error(format!(
                "vibevoice realtime preset manifest SHA-256 {} does not match external expected {}",
                hex_digest(&actual_manifest),
                expected_manifest_sha256
            )));
        }
        let manifest = parse_manifest(manifest_bytes)?;
        let actual_safetensors = sha256_bytes(safetensors_bytes);
        if actual_safetensors != manifest.output_sha256 {
            return Err(error(format!(
                "vibevoice realtime preset safetensors SHA-256 {} does not match manifest {}",
                hex_digest(&actual_safetensors),
                hex_digest(&manifest.output_sha256)
            )));
        }
        let file = SafetensorsFile::parse(safetensors_bytes.to_vec()).map_err(|e| {
            error(format!(
                "vibevoice realtime preset safetensors parse failed: {e}"
            ))
        })?;
        if file.tensors().len() != manifest.tensors.len() {
            return Err(error(format!(
                "vibevoice realtime preset tensor count {} does not match manifest {}",
                file.tensors().len(),
                manifest.tensors.len()
            )));
        }
        for info in file.tensors() {
            let Some(spec) = manifest.tensor(info.name.as_str()) else {
                return Err(error(format!(
                    "vibevoice realtime preset unexpected tensor `{}`",
                    info.name
                )));
            };
            if info.dtype != GgmlType::F32 || spec.dtype != "F32" {
                return Err(error(format!(
                    "vibevoice realtime preset tensor `{}` must be F32",
                    info.name
                )));
            }
            let expected_shape: Vec<u64> = spec.shape.iter().map(|&v| v as u64).collect();
            if info.shape != expected_shape {
                return Err(error(format!(
                    "vibevoice realtime preset tensor `{}` shape {:?} does not match manifest {:?}",
                    info.name, info.shape, expected_shape
                )));
            }
            let payload_sha = sha256_bytes(file.tensor_bytes(info));
            if payload_sha != spec.sha256 {
                return Err(error(format!(
                    "vibevoice realtime preset tensor `{}` payload SHA-256 {} does not match manifest {}",
                    info.name,
                    hex_digest(&payload_sha),
                    hex_digest(&spec.sha256)
                )));
            }
            if file
                .tensor_f32(info.name.as_str())
                .map_err(|e| {
                    error(format!(
                        "vibevoice realtime preset tensor `{}` decode failed: {e}",
                        info.name
                    ))
                })?
                .iter()
                .any(|v| !v.is_finite())
            {
                return Err(error(format!(
                    "vibevoice realtime preset tensor `{}` contains non-finite values",
                    info.name
                )));
            }
        }

        let outputs = [
            parse_output(&file, &manifest, VibeVoiceRealtimePresetBranch::Lm)?,
            parse_output(&file, &manifest, VibeVoiceRealtimePresetBranch::TtsLm)?,
            parse_output(&file, &manifest, VibeVoiceRealtimePresetBranch::NegLm)?,
            parse_output(&file, &manifest, VibeVoiceRealtimePresetBranch::NegTtsLm)?,
        ];
        Ok(Self {
            outputs,
            manifest_sha256: actual_manifest,
            safetensors_sha256: actual_safetensors,
        })
    }

    /// Returns one of the four complete branch outputs.
    #[must_use]
    pub fn output(&self, branch: VibeVoiceRealtimePresetBranch) -> &VibeVoiceRealtimePresetOutput {
        &self.outputs[branch_index(branch)]
    }

    /// Returns the externally authenticated manifest digest.
    #[must_use]
    pub const fn manifest_sha256(&self) -> [u8; 32] {
        self.manifest_sha256
    }

    /// Returns the checked safetensors payload digest.
    #[must_use]
    pub const fn safetensors_sha256(&self) -> [u8; 32] {
        self.safetensors_sha256
    }

    /// Imports the selected pair into a language branch through the existing
    /// transactional #179 cache API.
    pub(crate) fn import_branch_into(
        &self,
        language: &mut VibeVoiceRealtimeLanguage,
        text_branch: VibeVoiceRealtimePresetBranch,
        tts_branch: VibeVoiceRealtimePresetBranch,
    ) -> Result<()> {
        let text = self.output(text_branch);
        let tts = self.output(tts_branch);
        let text_layers: Vec<_> = text
            .layers
            .iter()
            .map(|layer| VibeVoiceRealtimeKvCacheLayer {
                keys: &layer.keys,
                values: &layer.values,
            })
            .collect();
        let tts_layers: Vec<_> = tts
            .layers
            .iter()
            .map(|layer| VibeVoiceRealtimeKvCacheLayer {
                keys: &layer.keys,
                values: &layer.values,
            })
            .collect();
        language.import_kv_cache_pair(VibeVoiceRealtimeLanguageKvCachePair {
            text: VibeVoiceRealtimeKvCacheSnapshot {
                position: text.cache_position,
                layers: &text_layers,
            },
            tts: VibeVoiceRealtimeKvCacheSnapshot {
                position: tts.cache_position,
                layers: &tts_layers,
            },
        })
    }
}

fn parse_output(
    file: &SafetensorsFile,
    manifest: &Manifest,
    branch: VibeVoiceRealtimePresetBranch,
) -> Result<VibeVoiceRealtimePresetOutput> {
    let binding = manifest.binding(branch.as_str())?;
    if binding.layer_count != branch.layer_count()
        || binding.source_layout != "[batch,kv_head,position,head_dim]"
        || binding.native_layout != "[position,kv_head,head_dim]"
    {
        return Err(error(format!(
            "vibevoice realtime preset `{}` binding does not match the fixed cache contract",
            branch.as_str()
        )));
    }
    let name = format!("{}.hidden", branch.as_str());
    let hidden_info = file
        .tensor_info(&name)
        .ok_or_else(|| error(format!("vibevoice realtime preset missing `{name}`")))?;
    let shape = &hidden_info.shape;
    if shape.len() != 3 || shape[0] != 1 || shape[2] != HIDDEN as u64 {
        return Err(error(format!(
            "vibevoice realtime preset `{name}` shape {:?} is not [1,hidden_rows,896]",
            shape
        )));
    }
    let hidden_rows =
        usize::try_from(shape[1]).map_err(|_| error("hidden row count overflows usize"))?;
    if hidden_rows == 0 || hidden_rows > MAX_POSITIONS {
        return Err(error(format!(
            "vibevoice realtime preset `{name}` has invalid hidden rows {hidden_rows}"
        )));
    }
    if hidden_rows != binding.hidden_rows {
        return Err(error(format!(
            "vibevoice realtime preset `{}` hidden rows {} do not match binding {}",
            branch.as_str(),
            hidden_rows,
            binding.hidden_rows
        )));
    }
    let hidden = file.tensor_f32(&name).map_err(|e| error(e.to_string()))?;
    let mut layers = Vec::with_capacity(branch.layer_count());
    let mut cache_position = None;
    for layer_index in 0..branch.layer_count() {
        let key_name = format!("{}.cache.{layer_index}.key", branch.as_str());
        let value_name = format!("{}.cache.{layer_index}.value", branch.as_str());
        let key = parse_cache_tensor(file, manifest, &key_name)?;
        let value = parse_cache_tensor(file, manifest, &value_name)?;
        if key.0 != value.0 {
            return Err(error(format!(
                "vibevoice realtime preset `{branch:?}` layer {layer_index} key/value positions differ"
            )));
        }
        if let Some(expected) = cache_position {
            if expected != key.0 {
                return Err(error(format!(
                    "vibevoice realtime preset `{branch:?}` cache layers have independent positions"
                )));
            }
        } else {
            cache_position = Some(key.0);
        }
        layers.push(VibeVoiceRealtimePresetCacheLayer {
            keys: key.1,
            values: value.1,
        });
    }
    let cache_position = cache_position.unwrap_or(0);
    if cache_position != binding.cache_position {
        return Err(error(format!(
            "vibevoice realtime preset `{}` cache position {} does not match binding {}",
            branch.as_str(),
            cache_position,
            binding.cache_position
        )));
    }
    Ok(VibeVoiceRealtimePresetOutput {
        hidden_rows,
        cache_position,
        hidden,
        layers,
    })
}

fn parse_cache_tensor(
    file: &SafetensorsFile,
    manifest: &Manifest,
    name: &str,
) -> Result<(usize, Vec<f32>)> {
    let info = file
        .tensor_info(name)
        .ok_or_else(|| error(format!("vibevoice realtime preset missing `{name}`")))?;
    let shape = &info.shape;
    if shape.len() != 3 || shape[1] != KV_HEADS as u64 || shape[2] != HEAD_DIM as u64 {
        return Err(error(format!(
            "vibevoice realtime preset `{name}` shape {:?} is not native [position,2,64]",
            shape
        )));
    }
    let position =
        usize::try_from(shape[0]).map_err(|_| error("cache position overflows usize"))?;
    if position == 0 || position > MAX_POSITIONS {
        return Err(error(format!(
            "vibevoice realtime preset `{name}` position exceeds {MAX_POSITIONS}"
        )));
    }
    let spec = manifest.tensor(name).ok_or_else(|| {
        error(format!(
            "vibevoice realtime preset manifest missing `{name}`"
        ))
    })?;
    let values = file.tensor_f32(name).map_err(|e| error(e.to_string()))?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(error(format!(
            "vibevoice realtime preset `{name}` contains non-finite values"
        )));
    }
    if values.len() != position * KV_HEADS * HEAD_DIM
        || spec.shape != shape.iter().map(|&v| v as usize).collect::<Vec<_>>()
    {
        return Err(error(format!(
            "vibevoice realtime preset `{name}` native element count mismatch"
        )));
    }
    Ok((position, values))
}

struct Manifest {
    output_sha256: [u8; 32],
    tensors: Vec<(String, TensorSpec)>,
    bindings: Vec<(String, Binding)>,
}

struct TensorSpec {
    dtype: String,
    shape: Vec<usize>,
    sha256: [u8; 32],
}

struct Binding {
    hidden_rows: usize,
    cache_position: usize,
    layer_count: usize,
    source_layout: String,
    native_layout: String,
}

impl Manifest {
    fn tensor(&self, name: &str) -> Option<&TensorSpec> {
        self.tensors
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, spec)| spec)
    }

    fn binding(&self, branch: &str) -> Result<&Binding> {
        self.bindings
            .iter()
            .find(|(key, _)| key == branch)
            .map(|(_, binding)| binding)
            .ok_or_else(|| error(format!("manifest binding for `{branch}` is missing")))
    }
}

fn parse_manifest(bytes: &[u8]) -> Result<Manifest> {
    let root = vokra_core::json::parse(bytes).map_err(|e| error(e.to_string()))?;
    reject_duplicate_keys(&root)?;
    require_str(&root, "format", FORMAT)?;
    require_str(&root, "classification", "INSPECTION_ONLY")?;
    require_str(&root, "publication", "NO_UPLOAD")?;
    require_str(&root, "voice_consent", "UNPROVEN")?;
    require_str(&root, "execution", "NO_MODEL_FORWARD_NO_AUDIO")?;
    let source = field(&root, "source")?;
    require_str(source, "repository", "microsoft/VibeVoice")?;
    require_str(
        source,
        "origin",
        "https://github.com/microsoft/VibeVoice.git",
    )?;
    require_str(source, "revision", SOURCE_REVISION)?;
    require_str(source, "relative_path", SOURCE_PATH)?;
    require_u64(source, "bytes", SOURCE_BYTES)?;
    require_str(source, "git_blob_sha1", SOURCE_BLOB_SHA1)?;
    require_str(source, "payload_sha256", SOURCE_PAYLOAD_SHA256)?;
    require_str(source, "model_revision", MODEL_REVISION)?;
    let preset = field(&root, "preset")?;
    require_str(preset, "id", "en-Carter_man")?;
    require_str(preset, "rights", "UNPROVEN")?;
    let lock = field(&root, "reference_lock")?;
    require_str(
        lock,
        "relative_path",
        "tools/parity/vibevoice_realtime_0_5b_reference/uv.lock",
    )?;
    require_hex_field(lock, "sha256", "reference lock SHA-256")?;
    let output = field(&root, "output")?;
    require_str(output, "relative_path", "cache.safetensors")?;
    let output_sha256 = require_hex_field(output, "sha256", "safetensors output SHA-256")?;
    let branches = field(&root, "branches")?
        .as_array()
        .ok_or_else(|| error("branches must be an array"))?;
    let expected_branches = ["lm", "tts_lm", "neg_lm", "neg_tts_lm"];
    if branches.len() != expected_branches.len()
        || branches
            .iter()
            .zip(expected_branches)
            .any(|(v, expected)| v.as_str() != Some(expected))
    {
        return Err(error(
            "manifest branches do not contain exactly the four official outputs",
        ));
    }
    let layout = field(&root, "tensor_layout_contract")?;
    require_str(layout, "hidden", "[1,hidden_rows,896] -> [hidden_rows,896]")?;
    require_str(layout, "framework_cache", "[1,2,position,64]")?;
    require_str(layout, "native_cache", "[position,2,64]")?;
    require_u64(layout, "max_positions", MAX_POSITIONS as u64)?;
    let bindings = parse_bindings(field(&root, "branch_bindings")?)?;
    let tensors = field(&root, "tensors")?
        .as_object()
        .ok_or_else(|| error("tensors must be an object"))?;
    let mut parsed = Vec::with_capacity(tensors.len());
    for (name, value) in tensors {
        let dtype = value
            .get("dtype")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| error(format!("tensor `{name}` dtype missing")))?
            .to_owned();
        let shape = value
            .get("shape")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| error(format!("tensor `{name}` shape missing")))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|d| usize::try_from(d).ok())
                    .ok_or_else(|| error(format!("tensor `{name}` shape is invalid")))
            })
            .collect::<Result<Vec<_>>>()?;
        let sha256 = require_hex_field(value, "sha256", &format!("tensor `{name}` SHA-256"))?;
        parsed.push((
            name.clone(),
            TensorSpec {
                dtype,
                shape,
                sha256,
            },
        ));
    }
    Ok(Manifest {
        output_sha256,
        tensors: parsed,
        bindings,
    })
}

fn parse_bindings(value: &JsonValue) -> Result<Vec<(String, Binding)>> {
    let entries = value
        .as_object()
        .ok_or_else(|| error("branch_bindings must be an object"))?;
    if entries.len() != 4 {
        return Err(error("branch_bindings must contain exactly four outputs"));
    }
    let mut bindings = Vec::with_capacity(entries.len());
    for (branch, value) in entries {
        if !matches!(branch.as_str(), "lm" | "tts_lm" | "neg_lm" | "neg_tts_lm") {
            return Err(error(format!("unexpected branch binding `{branch}`")));
        }
        let hidden_rows = require_bounded_usize(value, "hidden_rows", "hidden rows")?;
        let cache_position = require_bounded_usize(value, "cache_position", "cache position")?;
        let layer_count = require_bounded_usize(value, "layer_count", "layer count")?;
        if hidden_rows == 0
            || hidden_rows > MAX_POSITIONS
            || cache_position == 0
            || cache_position > MAX_POSITIONS
        {
            return Err(error(format!(
                "branch binding `{branch}` has an invalid position"
            )));
        }
        let source_layout = value
            .get("source_layout")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                error(format!(
                    "branch binding `{branch}` source layout is missing"
                ))
            })?
            .to_owned();
        let native_layout = value
            .get("native_layout")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                error(format!(
                    "branch binding `{branch}` native layout is missing"
                ))
            })?
            .to_owned();
        let source_dtypes = value.get("source_dtypes").ok_or_else(|| {
            error(format!(
                "branch binding `{branch}` source dtypes are missing"
            ))
        })?;
        let hidden_dtype = source_dtypes
            .get("hidden")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| error(format!("branch binding `{branch}` hidden dtype is missing")))?;
        if !matches!(hidden_dtype, "BFLOAT16" | "FLOAT16" | "FLOAT32") {
            return Err(error(format!(
                "branch binding `{branch}` has an invalid hidden dtype"
            )));
        }
        let cache_dtypes = source_dtypes
            .get("cache")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| {
                error(format!(
                    "branch binding `{branch}` cache dtypes are missing"
                ))
            })?;
        if cache_dtypes.is_empty()
            || cache_dtypes
                .iter()
                .any(|dtype| !matches!(dtype.as_str(), Some("BFLOAT16" | "FLOAT16" | "FLOAT32")))
        {
            return Err(error(format!(
                "branch binding `{branch}` has invalid cache dtypes"
            )));
        }
        bindings.push((
            branch.clone(),
            Binding {
                hidden_rows,
                cache_position,
                layer_count,
                source_layout,
                native_layout,
            },
        ));
    }
    Ok(bindings)
}

fn require_bounded_usize(value: &JsonValue, key: &str, label: &str) -> Result<usize> {
    let value = value
        .get(key)
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| error(format!("{label} is missing or not an integer")))?;
    usize::try_from(value).map_err(|_| error(format!("{label} overflows usize")))
}

fn reject_duplicate_keys(value: &JsonValue) -> Result<()> {
    match value {
        JsonValue::Object(entries) => {
            let mut keys = BTreeSet::new();
            for (key, value) in entries {
                if !keys.insert(key) {
                    return Err(error(format!("duplicate JSON object key `{key}`")));
                }
                reject_duplicate_keys(value)?;
            }
        }
        JsonValue::Array(values) => {
            for value in values {
                reject_duplicate_keys(value)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn parse_hex_digest(value: &str, label: &str) -> Result<[u8; 32]> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(error(format!("{label} must be 64 hexadecimal characters")));
    }
    let mut out = [0u8; 32];
    for (index, slot) in out.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| error(format!("{label} is not valid hexadecimal")))?;
    }
    Ok(out)
}

fn require_hex_field(value: &JsonValue, key: &str, label: &str) -> Result<[u8; 32]> {
    let text = value
        .get(key)
        .and_then(JsonValue::as_str)
        .ok_or_else(|| error(format!("{label} is missing")))?;
    parse_hex_digest(text, label)
}

fn field<'a>(value: &'a JsonValue, key: &str) -> Result<&'a JsonValue> {
    value
        .get(key)
        .ok_or_else(|| error(format!("manifest field `{key}` is missing")))
}

fn require_str(value: &JsonValue, key: &str, expected: &str) -> Result<()> {
    let actual = field(value, key)?
        .as_str()
        .ok_or_else(|| error(format!("manifest field `{key}` is not a string")))?;
    if actual != expected {
        return Err(error(format!(
            "manifest field `{key}` is {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn require_u64(value: &JsonValue, key: &str, expected: u64) -> Result<()> {
    let actual = field(value, key)?.as_u64().ok_or_else(|| {
        error(format!(
            "manifest field `{key}` is not a non-negative integer"
        ))
    })?;
    if actual != expected {
        return Err(error(format!(
            "manifest field `{key}` is {actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn branch_index(branch: VibeVoiceRealtimePresetBranch) -> usize {
    match branch {
        VibeVoiceRealtimePresetBranch::Lm => 0,
        VibeVoiceRealtimePresetBranch::TtsLm => 1,
        VibeVoiceRealtimePresetBranch::NegLm => 2,
        VibeVoiceRealtimePresetBranch::NegTtsLm => 3,
    }
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for &byte in bytes {
        out.push(char::from(DIGITS[(byte >> 4) as usize]));
        out.push(char::from(DIGITS[(byte & 0x0f) as usize]));
    }
    out
}

fn error(message: impl Into<String>) -> VokraError {
    VokraError::ModelLoad(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[derive(Debug, Clone, Copy)]
    enum Mutation {
        Good,
        WrongKey,
        MissingLayer,
        WrongShape,
        WrongHash,
        NonFinite,
        DuplicateJsonKey,
        WrongBinding,
    }

    #[test]
    fn branch_contract_keeps_positive_negative_and_layer_counts_distinct() {
        assert_eq!(VibeVoiceRealtimePresetBranch::Lm.layer_count(), 4);
        assert_eq!(VibeVoiceRealtimePresetBranch::NegLm.layer_count(), 4);
        assert_eq!(VibeVoiceRealtimePresetBranch::TtsLm.layer_count(), 20);
        assert_eq!(VibeVoiceRealtimePresetBranch::NegTtsLm.layer_count(), 20);
        assert_ne!(
            VibeVoiceRealtimePresetBranch::Lm.as_str(),
            VibeVoiceRealtimePresetBranch::NegLm.as_str()
        );
    }

    #[test]
    fn parser_accepts_four_outputs_and_preserves_independent_lengths() {
        let (safetensors, manifest, expected) = synthetic_bundle(Mutation::Good);
        let preset = VibeVoiceRealtimePresetCache::from_bytes(&safetensors, &manifest, &expected)
            .expect("synthetic preset should pass the structural bridge");
        for branch in [
            VibeVoiceRealtimePresetBranch::Lm,
            VibeVoiceRealtimePresetBranch::TtsLm,
            VibeVoiceRealtimePresetBranch::NegLm,
            VibeVoiceRealtimePresetBranch::NegTtsLm,
        ] {
            let output = preset.output(branch);
            let expected_hidden_rows = match branch {
                VibeVoiceRealtimePresetBranch::Lm => 2,
                VibeVoiceRealtimePresetBranch::TtsLm => 3,
                VibeVoiceRealtimePresetBranch::NegLm => 4,
                VibeVoiceRealtimePresetBranch::NegTtsLm => 5,
            };
            let expected_cache_position = expected_hidden_rows - 1;
            assert_eq!(output.hidden_rows(), expected_hidden_rows);
            assert_eq!(output.cache_position(), expected_cache_position);
            assert_eq!(output.hidden().len(), expected_hidden_rows * HIDDEN);
            assert_eq!(output.hidden()[0], expected_hidden_rows as f32);
            assert_eq!(output.layers().len(), branch.layer_count());
        }
    }

    #[test]
    fn parser_rejects_wrong_keys_layers_shapes_hashes_and_nonfinite_values() {
        for mutation in [
            Mutation::WrongKey,
            Mutation::MissingLayer,
            Mutation::WrongShape,
            Mutation::WrongHash,
            Mutation::NonFinite,
            Mutation::DuplicateJsonKey,
            Mutation::WrongBinding,
        ] {
            let (safetensors, manifest, expected) = synthetic_bundle(mutation);
            assert!(
                VibeVoiceRealtimePresetCache::from_bytes(&safetensors, &manifest, &expected)
                    .is_err(),
                "mutation should be rejected: {mutation:?}"
            );
        }
    }

    #[test]
    fn parser_requires_external_manifest_digest() {
        let (safetensors, manifest, _) = synthetic_bundle(Mutation::Good);
        assert!(
            VibeVoiceRealtimePresetCache::from_bytes(&safetensors, &manifest, &"0".repeat(64))
                .is_err()
        );
    }

    #[test]
    #[ignore = "requires the disposable VAST exported cache; never loads model weights"]
    fn vast_exported_cache_roundtrip_is_parser_only() {
        let safetensors_path = std::env::var_os("VOKRA_REAL_PRESET_SAFETENSORS")
            .expect("set VOKRA_REAL_PRESET_SAFETENSORS on VAST");
        let manifest_path = std::env::var_os("VOKRA_REAL_PRESET_MANIFEST")
            .expect("set VOKRA_REAL_PRESET_MANIFEST on VAST");
        let expected = std::env::var("VOKRA_REAL_PRESET_MANIFEST_SHA256")
            .expect("set VOKRA_REAL_PRESET_MANIFEST_SHA256 on VAST");
        let safetensors = std::fs::read(safetensors_path).expect("read exported safetensors");
        let manifest = std::fs::read(manifest_path).expect("read exported manifest");
        let preset = VibeVoiceRealtimePresetCache::from_bytes(&safetensors, &manifest, &expected)
            .expect("VAST exported cache must pass the native parser");
        assert_eq!(
            preset
                .output(VibeVoiceRealtimePresetBranch::Lm)
                .layers()
                .len(),
            4
        );
        assert_eq!(
            preset
                .output(VibeVoiceRealtimePresetBranch::TtsLm)
                .layers()
                .len(),
            20
        );
        assert_eq!(
            preset
                .output(VibeVoiceRealtimePresetBranch::NegLm)
                .layers()
                .len(),
            4
        );
        assert_eq!(
            preset
                .output(VibeVoiceRealtimePresetBranch::NegTtsLm)
                .layers()
                .len(),
            20
        );
    }

    fn synthetic_bundle(mutation: Mutation) -> (Vec<u8>, Vec<u8>, String) {
        let mut tensors: BTreeMap<String, (Vec<u64>, Vec<u8>)> = BTreeMap::new();
        let branches = [
            ("lm", 2usize, 1usize),
            ("tts_lm", 3, 2),
            ("neg_lm", 4, 3),
            ("neg_tts_lm", 5, 4),
        ];
        for (branch, hidden_rows, cache_position) in branches {
            let hidden_shape = if matches!(mutation, Mutation::WrongShape) && branch == "lm" {
                vec![1, 1, 895]
            } else {
                vec![1, hidden_rows as u64, HIDDEN as u64]
            };
            let hidden_len = hidden_shape.iter().product::<u64>() as usize;
            let mut hidden = vec![0u8; hidden_len * 4];
            hidden[..4].copy_from_slice(&(hidden_rows as f32).to_le_bytes());
            if matches!(mutation, Mutation::NonFinite) && branch == "lm" {
                hidden[..4].copy_from_slice(&f32::NAN.to_le_bytes());
            }
            tensors.insert(format!("{branch}.hidden"), (hidden_shape, hidden));
            let layer_count = if branch == "lm" || branch == "neg_lm" {
                4
            } else {
                20
            };
            for layer in 0..layer_count {
                if matches!(mutation, Mutation::MissingLayer)
                    && branch == "neg_tts_lm"
                    && layer == layer_count - 1
                {
                    continue;
                }
                for kind in ["key", "value"] {
                    tensors.insert(
                        format!("{branch}.cache.{layer}.{kind}"),
                        (
                            vec![cache_position as u64, KV_HEADS as u64, HEAD_DIM as u64],
                            vec![0u8; cache_position * KV_HEADS * HEAD_DIM * 4],
                        ),
                    );
                }
            }
        }
        if matches!(mutation, Mutation::WrongKey) {
            let value = tensors.remove("lm.hidden").expect("hidden fixture");
            tensors.insert("wrong.hidden".to_owned(), value);
        }
        let safetensors = make_safetensors(&tensors);
        let output_sha = hex_digest(&sha256_bytes(&safetensors));
        let mut tensor_json = String::new();
        for (index, (name, (shape, bytes))) in tensors.iter().enumerate() {
            if index != 0 {
                tensor_json.push(',');
            }
            let shape_json = shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let mut digest = hex_digest(&sha256_bytes(&payload_for(name, &safetensors, &tensors)));
            if matches!(mutation, Mutation::WrongHash) && name == "lm.hidden" {
                digest = "0".repeat(64);
            }
            let _ = bytes;
            tensor_json.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{shape_json}],\"sha256\":\"{digest}\"}}"
            ));
        }
        let binding_json = "\"lm\":{\"cache_position\":1,\"hidden_rows\":2,\"layer_count\":4,\"native_layout\":\"[position,kv_head,head_dim]\",\"source_dtypes\":{\"cache\":[\"BFLOAT16\"],\"hidden\":\"BFLOAT16\"},\"source_layout\":\"[batch,kv_head,position,head_dim]\"},\"neg_lm\":{\"cache_position\":3,\"hidden_rows\":4,\"layer_count\":4,\"native_layout\":\"[position,kv_head,head_dim]\",\"source_dtypes\":{\"cache\":[\"BFLOAT16\"],\"hidden\":\"BFLOAT16\"},\"source_layout\":\"[batch,kv_head,position,head_dim]\"},\"neg_tts_lm\":{\"cache_position\":4,\"hidden_rows\":5,\"layer_count\":20,\"native_layout\":\"[position,kv_head,head_dim]\",\"source_dtypes\":{\"cache\":[\"BFLOAT16\"],\"hidden\":\"BFLOAT16\"},\"source_layout\":\"[batch,kv_head,position,head_dim]\"},\"tts_lm\":{\"cache_position\":2,\"hidden_rows\":3,\"layer_count\":20,\"native_layout\":\"[position,kv_head,head_dim]\",\"source_dtypes\":{\"cache\":[\"BFLOAT16\"],\"hidden\":\"BFLOAT16\"},\"source_layout\":\"[batch,kv_head,position,head_dim]\"}";
        let mut manifest = format!(
            "{{\"branch_bindings\":{{{binding_json}}},\"branches\":[\"lm\",\"tts_lm\",\"neg_lm\",\"neg_tts_lm\"],\"classification\":\"INSPECTION_ONLY\",\"execution\":\"NO_MODEL_FORWARD_NO_AUDIO\",\"format\":\"{FORMAT}\",\"output\":{{\"relative_path\":\"cache.safetensors\",\"sha256\":\"{output_sha}\"}},\"preset\":{{\"id\":\"en-Carter_man\",\"rights\":\"UNPROVEN\"}},\"publication\":\"NO_UPLOAD\",\"reference_lock\":{{\"relative_path\":\"tools/parity/vibevoice_realtime_0_5b_reference/uv.lock\",\"sha256\":\"{}\"}},\"source\":{{\"bytes\":{SOURCE_BYTES},\"git_blob_sha1\":\"{SOURCE_BLOB_SHA1}\",\"model_revision\":\"{MODEL_REVISION}\",\"origin\":\"https://github.com/microsoft/VibeVoice.git\",\"payload_sha256\":\"{}\",\"relative_path\":\"{SOURCE_PATH}\",\"repository\":\"microsoft/VibeVoice\",\"revision\":\"{SOURCE_REVISION}\"}},\"tensor_layout_contract\":{{\"framework_cache\":\"[1,2,position,64]\",\"hidden\":\"[1,hidden_rows,896] -> [hidden_rows,896]\",\"max_positions\":{MAX_POSITIONS},\"native_cache\":\"[position,2,64]\"}},\"tensors\":{{{tensor_json}}},\"voice_consent\":\"UNPROVEN\"}}\n",
            "0".repeat(64),
            SOURCE_PAYLOAD_SHA256
        )
        .into_bytes();
        match mutation {
            Mutation::DuplicateJsonKey => {
                let end = manifest.len() - 2;
                manifest.splice(
                    end..end,
                    b",\"format\":\"duplicate-root-key\"".iter().copied(),
                );
            }
            Mutation::WrongBinding => {
                replace_once(&mut manifest, b"\"hidden_rows\":2", b"\"hidden_rows\":99");
            }
            _ => {}
        }
        let expected = hex_digest(&sha256_bytes(&manifest));
        (safetensors, manifest, expected)
    }

    fn replace_once(bytes: &mut Vec<u8>, needle: &[u8], replacement: &[u8]) {
        let start = bytes
            .windows(needle.len())
            .position(|window| window == needle)
            .expect("synthetic mutation needle");
        bytes.splice(start..start + needle.len(), replacement.iter().copied());
    }

    fn make_safetensors(tensors: &BTreeMap<String, (Vec<u64>, Vec<u8>)>) -> Vec<u8> {
        let mut offset = 0usize;
        let mut entries = String::from("{");
        let mut payload = Vec::new();
        for (index, (name, (shape, bytes))) in tensors.iter().enumerate() {
            if index != 0 {
                entries.push(',');
            }
            let shape_json = shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            entries.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{shape_json}],\"data_offsets\":[{offset},{}]}}",
                offset + bytes.len()
            ));
            offset += bytes.len();
            payload.extend_from_slice(bytes);
        }
        entries.push('}');
        let mut out = (entries.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(entries.as_bytes());
        out.extend_from_slice(&payload);
        out
    }

    fn payload_for(
        name: &str,
        safetensors: &[u8],
        tensors: &BTreeMap<String, (Vec<u64>, Vec<u8>)>,
    ) -> Vec<u8> {
        // Fixture tensors are sorted by name, matching make_safetensors.
        let mut offset = 0usize;
        let header_len = u64::from_le_bytes(safetensors[..8].try_into().unwrap()) as usize;
        for (candidate, (_, bytes)) in tensors {
            if candidate == name {
                let start = 8 + header_len + offset;
                return safetensors[start..start + bytes.len()].to_vec();
            }
            offset += bytes.len();
        }
        panic!("missing fixture tensor {name}");
    }
}
