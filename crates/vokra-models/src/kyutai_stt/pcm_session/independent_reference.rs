//! Bounded, test-only consumer for the official Kyutai PCM packet.
//!
//! This module is deliberately not a loader API.  It authenticates the
//! owner-supplied component packet, then exercises the existing
//! `KyutaiSttPcmEngine::from_paths` and borrowed observer seam in an ignored
//! real-weight test.  The reviewed closure allowlist is empty, so a real run
//! remains fail-closed before any component is opened.  The unit tests cover
//! only packet/schema and schedule safety; they are not PCM parity evidence.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use vokra_core::json::{self, JsonValue};
use vokra_core::{BackendKind, Result, VokraError};

use super::{
    KyutaiSttPcmArtifactDigest, KyutaiSttPcmArtifactDigests, KyutaiSttPcmEngine, PcmObserver,
};
use crate::strict_checkpoint::sha256_bytes;

const PACKET_SCHEMA: &str = "vokra-kyutai-stt-pcm-components-v1";
const PACKET_MAX_BYTES: u64 = 8 * 1024 * 1024;
// The packet contains model inputs, not only reference outputs.  The pinned
// decoder source is 5,234,275,128 bytes; reject only above an explicit finite
// upper bound and keep hashing mmap-backed so this does not allocate it.
const COMPONENT_MAX_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const COMPONENT_AGGREGATE_MAX_BYTES: u64 = 12 * 1024 * 1024 * 1024;
const PCM_MAX_BYTES: u64 = 64 * 1024 * 1024;
const REFERENCE_MANIFEST_MAX_BYTES: u64 = 16 * 1024 * 1024;
const REFERENCE_ARTIFACT_MAX_BYTES: u64 = 512 * 1024 * 1024;
const REFERENCE_TOTAL_ARTIFACT_MAX_BYTES: u64 = (5 * 1024 * 1024 * 1024) / 4;
const REFERENCE_TREE_ENTRY_MAX: usize = 4_096;
const MAX_FRAMES: usize = 4_096;
const MAX_CALLS: usize = 8_192;
const EXPECTED_SAMPLE_RATE: u64 = 24_000;
const EXPECTED_CHANNELS: u64 = 1;
const EXPECTED_DTYPE: &str = "float32-le";
const REVIEWED_DEPENDENCY_CLOSURES: &[&str] = &[];

const PACKET_ENV: &str = "VOKRA_KYUTAI_STT_PCM_COMPONENTS_PACKET";
const PACKET_SHA_ENV: &str = "VOKRA_KYUTAI_STT_PCM_COMPONENTS_PACKET_SHA256";
const GGUF_ENV: &str = "VOKRA_KYUTAI_STT_STREAMING_GGUF";
const GGUF_SHA_ENV: &str = "VOKRA_KYUTAI_STT_STREAMING_GGUF_SHA256";
const REFERENCE_ENV: &str = "VOKRA_KYUTAI_STT_STREAMING_REFERENCE";
const REFERENCE_SHA_ENV: &str = "VOKRA_KYUTAI_STT_STREAMING_REFERENCE_MANIFEST_SHA256";
const APPROVAL_ENV: &str = "VOKRA_KYUTAI_STT_STREAMING_APPROVAL";
const APPROVAL_SHA_ENV: &str = "VOKRA_KYUTAI_STT_STREAMING_APPROVAL_SHA256";
const CLOSURE_SHA_ENV: &str = "VOKRA_KYUTAI_STT_STREAMING_DEPENDENCY_CLOSURE_SHA256";
const EXPECTED_HEAD_ENV: &str = "VOKRA_KYUTAI_STT_STREAMING_EXPECTED_HEAD";
const MODEL_BYTES: u64 = 5_234_275_128;
const MODEL_SHA256: &str = "2471add7da1fdb2d5dc4561e88a9069376333d992760d55d29d1db46c52849b2";
const MODEL_TENSOR_MANIFEST_SHA256: &str =
    "e62488c9d16953010c758ec17f4c70e8ee30d348adfab3811eb5dfecb435d5df";
const MIMI_NAME: &str = "mimi-pytorch-e351c8d8@125.safetensors";
const MIMI_BYTES: u64 = 384_644_900;
const MIMI_SHA256: &str = "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50";
const TOKENIZER_NAME: &str = "tokenizer_en_audio_4000.model";
const TOKENIZER_BYTES: u64 = 59_339;
const TOKENIZER_SHA256: &str = "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6";
const CONFIG_NAME: &str = "config.json";
const CONFIG_BYTES: u64 = 1_257;
const CONFIG_SHA256: &str = "b79ea52a30329887a2d0ce2dd5473a63fc5083e441e7986f64f01050c06239c9";
const PCM_REFERENCE_SCHEMA: &str = "vokra-kyutai-stt-independent-pcm-reference-v1";
const MOSHI_REPOSITORY: &str = "https://github.com/kyutai-labs/moshi.git";
const MOSHI_REVISION: &str = "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362";
const DSM_REPOSITORY: &str = "https://github.com/kyutai-labs/delayed-streams-modeling.git";
const DSM_REVISION: &str = "4c4f65e147df056adf3346290d64c7b9649b18c9";
const SOURCE_PACKET_MANIFEST_SHA256: &str =
    "85223a7ac8b947eaeafa7b2f337a1ac60dea84a75d44ee0c607df72ad25d6342";
const DSM_TREE_SHA: &str = "1ab73718d99c5bb6ff94c1bc84a783a4d2e3a7e0";
const MOSHI_TREE_SHA: &str = "2a6d6afe53d70bac490117651dfc478cf87a940e";
const SOURCE_CONTRACT_SCHEMA: &str = "vokra-kyutai-stt-streaming-source-contract-v1";
const SOURCE_CONTRACT_STATUS: &str = "AUTHENTICATED_SOURCE_CONTRACT";
const SOURCE_CONTRACT_RUNTIME_STATUS: &str = "BLOCKED_NOT_EXECUTED";
const SOURCE_CONTRACT_DSM_ROLES: &[&str] = &[
    "configs/config-stt-en-hf.toml",
    "scripts/stt_from_file_pytorch.py",
];
const SOURCE_CONTRACT_MOSHI_ROLES: &[&str] = &[
    "moshi/moshi/models/lm.py",
    "moshi/moshi/models/lm_utils.py",
    "moshi/moshi/models/loaders.py",
    "moshi/moshi/utils/sampling.py",
    "moshi/moshi/modules/transformer.py",
];
const SOURCE_MOSHI_LM_SHA256: &str =
    "38991e83d7e3aa0ff1483b27a0b59e1ff43642ad6a32dc8dc4272ad2b6c0dc5f";
const SOURCE_MOSHI_TRANSFORMER_SHA256: &str =
    "f5a73d752a5bde1eda2b0b14bebd13fd81db81017d361782a07164580a687622";

#[derive(Debug, Clone, PartialEq, Eq)]
struct PcmBinding {
    path: PathBuf,
    bytes: u64,
    sha256: String,
    sample_rate: u64,
    channels: u64,
    dtype: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ComponentBinding {
    role: String,
    path: PathBuf,
    bytes: u64,
    sha256: String,
    source_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ComponentPacket {
    expected_head: String,
    reference_manifest_sha256: String,
    approval_sha256: String,
    dependency_closure_sha256: String,
    pcm: PcmBinding,
    components: Vec<ComponentBinding>,
}

#[derive(Debug, Clone)]
struct ReferenceEvidence {
    root: PathBuf,
    manifest: JsonValue,
    codes: Vec<[u32; 32]>,
    logits: Vec<Vec<f32>>,
    tokens: Vec<u32>,
    kv_events: BTreeMap<(String, usize, usize), KvEvent>,
}

#[derive(Debug, Clone)]
struct KvEvent {
    positions: Vec<i64>,
    keys: String,
    values: String,
}

#[derive(Debug, Clone, Copy)]
struct ExistingBindings<'a> {
    expected_head: &'a str,
    decoder_sha256: &'a str,
    reference_manifest_sha256: &'a str,
    approval_sha256: &'a str,
    dependency_closure_sha256: &'a str,
}

fn invalid(message: impl Into<String>) -> VokraError {
    VokraError::InvalidArgument(message.into())
}

fn required_string<'a>(value: &'a JsonValue, key: &str, label: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(JsonValue::as_str)
        .ok_or_else(|| invalid(format!("{label}.{key} must be a string")))
}

fn required_u64(value: &JsonValue, key: &str, label: &str) -> Result<u64> {
    value
        .get(key)
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| invalid(format!("{label}.{key} must be a non-negative integer")))
}

fn exact_object(value: &JsonValue, expected: &[&str], label: &str) -> Result<()> {
    let entries = value
        .as_object()
        .ok_or_else(|| invalid(format!("{label} must be an object")))?;
    if entries.len() != expected.len() {
        return Err(invalid(format!("{label} has unexpected key count")));
    }
    let mut seen = BTreeSet::new();
    for (key, _) in entries {
        if !seen.insert(key.as_str()) || !expected.contains(&key.as_str()) {
            return Err(invalid(format!(
                "{label} has duplicate or unknown key {key:?}"
            )));
        }
    }
    Ok(())
}

fn reject_duplicate_keys(value: &JsonValue, label: &str) -> Result<()> {
    match value {
        JsonValue::Object(entries) => {
            let mut seen = BTreeSet::new();
            for (key, child) in entries {
                if !seen.insert(key.as_str()) {
                    return Err(invalid(format!("{label} contains duplicate key {key:?}")));
                }
                reject_duplicate_keys(child, label)?;
            }
        }
        JsonValue::Array(values) => {
            for value in values {
                reject_duplicate_keys(value, label)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn lower_hex(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(format!("{label} must be lowercase SHA-256 hex")));
    }
    Ok(())
}

fn lower_git_blob_sha1(value: &str, label: &str) -> Result<()> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(format!("{label} must be lowercase Git SHA-1 hex")));
    }
    Ok(())
}

fn parse_pcm(value: &JsonValue) -> Result<PcmBinding> {
    exact_object(
        value,
        &[
            "path",
            "bytes",
            "sha256",
            "sample_rate",
            "channels",
            "dtype",
        ],
        "pcm",
    )?;
    let path = PathBuf::from(required_string(value, "path", "pcm")?);
    let bytes = required_u64(value, "bytes", "pcm")?;
    let sha256 = required_string(value, "sha256", "pcm")?.to_owned();
    let sample_rate = required_u64(value, "sample_rate", "pcm")?;
    let channels = required_u64(value, "channels", "pcm")?;
    let dtype = required_string(value, "dtype", "pcm")?.to_owned();
    lower_hex(&sha256, "pcm.sha256")?;
    if !path.is_absolute() {
        return Err(invalid("pcm.path must be absolute"));
    }
    if bytes == 0 || bytes > PCM_MAX_BYTES || bytes % 4 != 0 {
        return Err(invalid(
            "pcm byte count is outside the bounded float32 contract",
        ));
    }
    if sample_rate != EXPECTED_SAMPLE_RATE
        || channels != EXPECTED_CHANNELS
        || dtype != EXPECTED_DTYPE
    {
        return Err(invalid(
            "pcm geometry or dtype is not the authenticated contract",
        ));
    }
    Ok(PcmBinding {
        path,
        bytes,
        sha256,
        sample_rate,
        channels,
        dtype,
    })
}

fn parse_component(value: &JsonValue, index: usize) -> Result<ComponentBinding> {
    let label = format!("components[{index}]");
    exact_object(
        value,
        &["role", "path", "bytes", "sha256", "source_sha256"],
        &label,
    )?;
    let role = required_string(value, "role", &label)?.to_owned();
    let path = PathBuf::from(required_string(value, "path", &label)?);
    let bytes = required_u64(value, "bytes", &label)?;
    let sha256 = required_string(value, "sha256", &label)?.to_owned();
    let source_sha256 = required_string(value, "source_sha256", &label)?.to_owned();
    lower_hex(&sha256, &format!("{label}.sha256"))?;
    lower_hex(&source_sha256, &format!("{label}.source_sha256"))?;
    if bytes == 0 || bytes > COMPONENT_MAX_BYTES {
        return Err(invalid(format!("{label}.bytes exceeds its bound")));
    }
    Ok(ComponentBinding {
        role,
        path,
        bytes,
        sha256,
        source_sha256,
    })
}

fn parse_packet(bytes: &[u8]) -> Result<ComponentPacket> {
    if bytes.is_empty() || bytes.len() as u64 > PACKET_MAX_BYTES {
        return Err(invalid("component packet is empty or exceeds its bound"));
    }
    let value =
        json::parse(bytes).map_err(|error| invalid(format!("component packet JSON: {error}")))?;
    reject_duplicate_keys(&value, "component packet")?;
    exact_object(
        &value,
        &[
            "format",
            "expected_head",
            "reference_manifest_sha256",
            "approval_sha256",
            "dependency_closure_sha256",
            "pcm",
            "components",
        ],
        "component packet",
    )?;
    let format = required_string(&value, "format", "component packet")?;
    if format != PACKET_SCHEMA {
        return Err(invalid(
            "component packet schema is not the pinned PCM schema",
        ));
    }
    let expected_head = required_string(&value, "expected_head", "component packet")?.to_owned();
    if expected_head.len() != 40
        || !expected_head
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(invalid(
            "component packet expected_head must be lowercase git SHA-1",
        ));
    }
    let reference_manifest_sha256 =
        required_string(&value, "reference_manifest_sha256", "component packet")?.to_owned();
    let approval_sha256 =
        required_string(&value, "approval_sha256", "component packet")?.to_owned();
    let dependency_closure_sha256 =
        required_string(&value, "dependency_closure_sha256", "component packet")?.to_owned();
    lower_hex(&reference_manifest_sha256, "reference_manifest_sha256")?;
    lower_hex(&approval_sha256, "approval_sha256")?;
    lower_hex(&dependency_closure_sha256, "dependency_closure_sha256")?;
    let pcm = parse_pcm(
        value
            .get("pcm")
            .ok_or_else(|| invalid("component packet.pcm missing"))?,
    )?;
    let values = value
        .get("components")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| invalid("component packet.components must be an array"))?;
    if values.len() != 4 {
        return Err(invalid(
            "component packet must contain exactly four components",
        ));
    }
    let components: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(index, value)| parse_component(value, index))
        .collect::<Result<_>>()?;
    let required_roles = ["decoder", "tokenizer", "mimi_gguf", "raw_mimi"];
    let roles: BTreeSet<_> = components.iter().map(|item| item.role.as_str()).collect();
    if roles.len() != required_roles.len()
        || required_roles.iter().any(|role| !roles.contains(role))
    {
        return Err(invalid(
            "component packet roles must be unique decoder/tokenizer/mimi_gguf/raw_mimi",
        ));
    }
    let aggregate = components
        .iter()
        .try_fold(0u64, |total, item| total.checked_add(item.bytes))
        .ok_or_else(|| invalid("component aggregate byte count overflow"))?;
    if aggregate > COMPONENT_AGGREGATE_MAX_BYTES {
        return Err(invalid("component aggregate byte count exceeds its bound"));
    }
    Ok(ComponentPacket {
        expected_head,
        reference_manifest_sha256,
        approval_sha256,
        dependency_closure_sha256,
        pcm,
        components,
    })
}

fn validate_gate(packet: &ComponentPacket, expected: ExistingBindings<'_>) -> Result<()> {
    if packet.expected_head != expected.expected_head
        || packet.reference_manifest_sha256 != expected.reference_manifest_sha256
        || packet.approval_sha256 != expected.approval_sha256
        || packet.dependency_closure_sha256 != expected.dependency_closure_sha256
    {
        return Err(invalid(
            "component packet is not bound to the existing head/reference/approval/closure",
        ));
    }
    if !REVIEWED_DEPENDENCY_CLOSURES
        .iter()
        .any(|digest| *digest == packet.dependency_closure_sha256)
    {
        return Err(invalid(
            "BLOCKED_DEPENDENCY_CLOSURE: no reviewed PCM dependency closure is accepted",
        ));
    }
    let decoder = packet
        .components
        .iter()
        .find(|item| item.role == "decoder")
        .ok_or_else(|| invalid("component packet has no decoder role"))?;
    if decoder.sha256 != expected.decoder_sha256 {
        return Err(invalid(
            "component decoder digest does not match the existing main GGUF binding",
        ));
    }
    for component in &packet.components {
        let expected_source = match component.role.as_str() {
            "decoder" => MODEL_SHA256,
            "tokenizer" => TOKENIZER_SHA256,
            "mimi_gguf" | "raw_mimi" => MIMI_SHA256,
            _ => return Err(invalid("unknown component role")),
        };
        if component.source_sha256 != expected_source {
            return Err(invalid(format!(
                "{} source identity is not the pinned upstream identity",
                component.role
            )));
        }
    }
    Ok(())
}

fn validate_disjoint_paths(packet: &ComponentPacket, packet_path: &Path) -> Result<()> {
    let packet_canonical = fs::canonicalize(packet_path)
        .map_err(|error| invalid(format!("component packet canonicalization failed: {error}")))?;
    let packet_metadata = fs::metadata(packet_path)
        .map_err(|error| invalid(format!("component packet metadata failed: {error}")))?;
    let packet_identity = file_identity(&packet_metadata);
    let mut seen = BTreeSet::new();
    let mut seen_ids = BTreeSet::new();
    let mut paths = vec![(&packet.pcm.path, "canonical PCM")];
    paths.extend(
        packet
            .components
            .iter()
            .map(|component| (&component.path, component.role.as_str())),
    );
    for (path, label) in paths {
        reject_symlink_path(path, label)?;
        let canonical = fs::canonicalize(path)
            .map_err(|error| invalid(format!("{label} canonicalization failed: {error}")))?;
        let metadata = fs::metadata(path).map_err(|error| invalid(format!("{label}: {error}")))?;
        if canonical == packet_canonical
            || file_identity(&metadata) == packet_identity
            || !seen.insert(canonical)
            || !seen_ids.insert(file_identity(&metadata))
        {
            return Err(invalid(format!(
                "{label} path is not disjoint from another packet input"
            )));
        }
    }
    Ok(())
}

fn validate_external_disjoint(
    packet: &ComponentPacket,
    packet_path: &Path,
    reference_root: &Path,
    approval_path: &Path,
) -> Result<()> {
    let mut paths = vec![
        fs::canonicalize(packet_path).map_err(|error| {
            invalid(format!("component packet canonicalization failed: {error}"))
        })?,
        fs::canonicalize(reference_root)
            .map_err(|error| invalid(format!("reference canonicalization failed: {error}")))?,
        fs::canonicalize(approval_path)
            .map_err(|error| invalid(format!("approval canonicalization failed: {error}")))?,
    ];
    paths.extend(
        std::iter::once(&packet.pcm.path)
            .chain(packet.components.iter().map(|component| &component.path))
            .map(|path| {
                fs::canonicalize(path)
                    .map_err(|error| invalid(format!("input canonicalization failed: {error}")))
            })
            .collect::<Result<Vec<_>>>()?,
    );
    for left in 0..paths.len() {
        for right in (left + 1)..paths.len() {
            if paths[left].starts_with(&paths[right]) || paths[right].starts_with(&paths[left]) {
                return Err(invalid(
                    "PCM packet, reference, approval, and component paths overlap",
                ));
            }
        }
    }
    Ok(())
}

fn reject_symlink_path(path: &Path, label: &str) -> Result<()> {
    if !path.is_absolute() {
        return Err(invalid(format!("{label} must be an absolute path")));
    }
    if path.components().any(|component| {
        matches!(
            component,
            std::path::Component::CurDir | std::path::Component::ParentDir
        )
    }) {
        return Err(invalid(format!("{label} contains a dot path component")));
    }
    let mut prefix = PathBuf::new();
    for component in path.components() {
        prefix.push(component.as_os_str());
        let metadata = fs::symlink_metadata(&prefix)
            .map_err(|error| invalid(format!("{label} path component is inaccessible: {error}")))?;
        if metadata.file_type().is_symlink() {
            return Err(invalid(format!("{label} path contains a symlink")));
        }
    }
    let metadata =
        fs::metadata(path).map_err(|error| invalid(format!("{label} is inaccessible: {error}")))?;
    if !metadata.is_file() {
        return Err(invalid(format!("{label} is not a regular file")));
    }
    Ok(())
}

#[cfg(unix)]
fn file_identity(metadata: &std::fs::Metadata) -> (u64, u64) {
    use std::os::unix::fs::MetadataExt;
    (metadata.dev(), metadata.ino())
}

#[cfg(not(unix))]
fn file_identity(metadata: &std::fs::Metadata) -> (u64, u64) {
    (
        metadata.len(),
        metadata.modified().ok().map_or(0, |value| {
            value.elapsed().ok().map_or(0, |age| age.as_nanos() as u64)
        }),
    )
}

fn read_bounded_file(path: &Path, max_bytes: u64, label: &str) -> Result<(Vec<u8>, (u64, u64))> {
    reject_symlink_path(path, label)?;
    let before = fs::metadata(path).map_err(|error| invalid(format!("{label}: {error}")))?;
    if before.len() > max_bytes {
        return Err(invalid(format!("{label} exceeds its bounded byte budget")));
    }
    let identity = file_identity(&before);
    let mut file = File::open(path).map_err(|error| invalid(format!("{label}: {error}")))?;
    let mut body = Vec::with_capacity(before.len() as usize);
    let mut limited = Read::by_ref(&mut file).take(max_bytes.saturating_add(1));
    limited
        .read_to_end(&mut body)
        .map_err(|error| invalid(format!("{label}: {error}")))?;
    if body.len() as u64 > max_bytes {
        return Err(invalid(format!(
            "{label} grew beyond its bounded byte budget"
        )));
    }
    // Re-check the pathname after the read.  Metadata on the already-open
    // descriptor alone does not detect replacing the pathname with a symlink
    // during a bounded read.
    reject_symlink_path(path, label)?;
    let after_file = file
        .metadata()
        .map_err(|error| invalid(format!("{label}: {error}")))?;
    let after_path = fs::metadata(path).map_err(|error| invalid(format!("{label}: {error}")))?;
    if after_file.len() != before.len()
        || after_path.len() != before.len()
        || file_identity(&after_file) != identity
        || file_identity(&after_path) != identity
        || body.len() as u64 != before.len()
    {
        return Err(invalid(format!("{label} changed while being read")));
    }
    Ok((body, identity))
}

fn expected_kv_positions(step: usize) -> BTreeSet<i64> {
    let start = (step + 1).saturating_sub(375) as i64;
    (start..=step as i64).collect()
}

fn expected_checkpoint_coordinates() -> BTreeSet<(String, usize, usize)> {
    [0usize, 1, 374, 375, 376]
        .into_iter()
        .chain(std::iter::once(0))
        .enumerate()
        .flat_map(|(index, step)| {
            let phase = if index == 5 { "after_reset" } else { "warmup" };
            (0..48).map(move |layer| (phase.to_owned(), step, layer))
        })
        .collect()
}

fn validate_kv_positions(step: usize, positions: &[i64]) -> Result<()> {
    if positions.len() != 375 {
        return Err(invalid("KV snapshot positions are not context length 375"));
    }
    let mut valid_positions = BTreeSet::new();
    for position in positions {
        if *position < -1 || *position > step as i64 {
            return Err(invalid("KV snapshot position is outside the causal range"));
        }
        if *position >= 0 && !valid_positions.insert(*position) {
            return Err(invalid("KV snapshot contains duplicate valid positions"));
        }
    }
    if valid_positions != expected_kv_positions(step) {
        return Err(invalid(
            "KV snapshot absolute positions do not match causal eviction",
        ));
    }
    Ok(())
}

fn expected_native_positions(step: usize) -> Vec<usize> {
    let start = (step + 1).saturating_sub(375);
    (start..=step).collect()
}

fn verify_file_binding(path: &Path, bytes: u64, digest: &str, label: &str) -> Result<()> {
    reject_symlink_path(path, label)?;
    let metadata = fs::metadata(path).map_err(|error| invalid(format!("{label}: {error}")))?;
    if metadata.len() != bytes {
        return Err(invalid(format!(
            "{label} byte count changed while authenticating"
        )));
    }
    let mapping =
        vokra_mmap::Mmap::open(path).map_err(|error| invalid(format!("{label}: {error}")))?;
    if mapping.bytes().len() as u64 != bytes {
        return Err(invalid(format!("{label} changed while reading")));
    }
    let actual = hex_digest(&sha256_bytes(mapping.bytes()));
    if actual != digest {
        return Err(invalid(format!("{label} SHA-256 mismatch")));
    }
    reject_symlink_path(path, label)?;
    let after = fs::metadata(path).map_err(|error| invalid(format!("{label}: {error}")))?;
    if after.len() != metadata.len() || file_identity(&after) != file_identity(&metadata) {
        return Err(invalid(format!(
            "{label} changed while being authenticated"
        )));
    }
    Ok(())
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(64);
    for byte in bytes {
        text.push(char::from(HEX[(byte >> 4) as usize]));
        text.push(char::from(HEX[(byte & 0xf) as usize]));
    }
    text
}

fn json_string<'a>(value: &'a JsonValue, key: &str, label: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(JsonValue::as_str)
        .ok_or_else(|| invalid(format!("{label}.{key} must be a string")))
}

fn json_bytes(value: &JsonValue, key: &str, label: &str) -> Result<u64> {
    value
        .get(key)
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| invalid(format!("{label}.{key} must be a non-negative integer")))
}

fn json_f64(value: &JsonValue, key: &str, label: &str) -> Result<f64> {
    value
        .get(key)
        .and_then(JsonValue::as_f64)
        .ok_or_else(|| invalid(format!("{label}.{key} must be a number")))
}

fn safe_artifact_name(name: &str) -> Result<()> {
    let path = Path::new(name);
    if name.is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(invalid(format!("unsafe reference artifact path {name:?}")));
    }
    Ok(())
}

fn collect_reference_files(
    root: &Path,
    current: &Path,
    output: &mut BTreeSet<String>,
    entries_seen: &mut usize,
    depth: usize,
) -> Result<()> {
    if depth > 32 || *entries_seen > REFERENCE_TREE_ENTRY_MAX {
        return Err(invalid("reference artifact tree exceeds its bound"));
    }
    let entries = fs::read_dir(current)
        .map_err(|error| invalid(format!("reference artifact directory: {error}")))?;
    for entry in entries {
        *entries_seen = (*entries_seen)
            .checked_add(1)
            .ok_or_else(|| invalid("reference artifact entry count overflow"))?;
        if *entries_seen > REFERENCE_TREE_ENTRY_MAX {
            return Err(invalid("reference artifact tree exceeds its entry bound"));
        }
        let entry = entry.map_err(|error| invalid(format!("reference artifact entry: {error}")))?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| invalid(format!("reference artifact metadata: {error}")))?;
        if metadata.file_type().is_symlink() {
            return Err(invalid("reference artifact tree contains a symlink"));
        }
        if metadata.is_dir() {
            collect_reference_files(root, &path, output, entries_seen, depth + 1)?;
        } else if metadata.is_file() {
            let name = path
                .strip_prefix(root)
                .map_err(|_| invalid("reference artifact path escaped root"))?
                .to_string_lossy()
                .replace('\\', "/");
            safe_artifact_name(&name)?;
            output.insert(name);
            if output.len() > 4_096 {
                return Err(invalid("reference artifact tree exceeds its bound"));
            }
        } else {
            return Err(invalid(
                "reference artifact tree contains a non-regular entry",
            ));
        }
    }
    Ok(())
}

fn artifact_bytes(
    root: &Path,
    artifacts: &JsonValue,
    name: &str,
    max_bytes: u64,
) -> Result<Vec<u8>> {
    safe_artifact_name(name)?;
    let row = artifacts
        .get(name)
        .ok_or_else(|| invalid(format!("reference artifact {name} is not declared")))?;
    exact_object(row, &["bytes", "sha256"], &format!("artifacts.{name}"))?;
    let expected_bytes = json_bytes(row, "bytes", &format!("artifacts.{name}"))?;
    let expected_sha = json_string(row, "sha256", &format!("artifacts.{name}"))?;
    lower_hex(expected_sha, &format!("artifacts.{name}.sha256"))?;
    if expected_bytes > max_bytes {
        return Err(invalid(format!(
            "reference artifact {name} exceeds its bound"
        )));
    }
    let path = root.join(name);
    let (body, _) = read_bounded_file(&path, max_bytes, &format!("reference artifact {name}"))?;
    if body.len() as u64 != expected_bytes || hex_digest(&sha256_bytes(&body)) != expected_sha {
        return Err(invalid(format!(
            "reference artifact {name} changed or has a bad SHA-256"
        )));
    }
    Ok(body)
}

fn flatten_signed_integers(value: &JsonValue, output: &mut Vec<i64>) -> Result<()> {
    match value {
        JsonValue::Int(number) => output.push(*number),
        JsonValue::Array(values) => {
            for value in values {
                flatten_signed_integers(value, output)?;
            }
        }
        _ => return Err(invalid("KV positions must be an integer array")),
    }
    Ok(())
}

fn integer_array(value: &JsonValue, label: &str) -> Result<Vec<i64>> {
    let mut output = Vec::new();
    flatten_signed_integers(value, &mut output)?;
    if output.is_empty() {
        return Err(invalid(format!("{label} must not be empty")));
    }
    Ok(output)
}

fn parse_kv_events(
    manifest: &JsonValue,
    artifacts: &JsonValue,
) -> Result<BTreeMap<(String, usize, usize), KvEvent>> {
    let ring_cache = manifest
        .get("ring_cache")
        .ok_or_else(|| invalid("streaming manifest ring_cache missing"))?;
    if json_string(ring_cache, "checkpoint_file_pattern", "ring_cache")?
        != "kv/{phase}-step-{step:04d}-layer-{layer:02d}-{keys|values}.bin"
    {
        return Err(invalid(
            "ring-cache checkpoint filename pattern is not authenticated",
        ));
    }
    if integer_array(
        ring_cache
            .get("checkpoints")
            .ok_or_else(|| invalid("ring-cache checkpoint list missing"))?,
        "ring_cache.checkpoints",
    )?
    .as_slice()
        != [0, 1, 374, 375, 376]
    {
        return Err(invalid(
            "ring-cache checkpoint list is not the bounded contract",
        ));
    }
    let events = ring_cache
        .get("events")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| invalid("streaming manifest KV events missing"))?;
    if events.len() != (377 + 1) * 48 {
        return Err(invalid(
            "reference KV event coverage is not 18144 phase/layer events",
        ));
    }
    let mut parsed = BTreeMap::new();
    let mut seen_all = BTreeSet::new();
    for event in events {
        let phase = json_string(event, "phase", "KV event")?.to_owned();
        let step = json_bytes(event, "step", "KV event")? as usize;
        let layer = json_bytes(event, "layer", "KV event")? as usize;
        if !((phase == "warmup" && step < 377) || (phase == "after_reset" && step == 0))
            || layer >= 48
        {
            return Err(invalid(
                "KV event phase/layer is outside the authenticated contract",
            ));
        }
        if !seen_all.insert((phase.clone(), step, layer)) {
            return Err(invalid("duplicate KV phase/step/layer event"));
        }
        if json_bytes(event, "lm_call_ordinal", "KV event")? != 0
            || integer_array(
                event
                    .get("delta_shape")
                    .ok_or_else(|| invalid("KV event delta_shape missing"))?,
                "KV event delta_shape",
            )?
            .as_slice()
                != [1, 32, 1, 64]
            || json_string(event, "delta_dtype", "KV event")? != "torch.bfloat16"
            || json_bytes(event, "capacity", "KV event")? != 375
        {
            return Err(invalid(
                "KV event delta metadata is not the producer contract",
            ));
        }
        let expected_delta_keys = format!("kv/{phase}-layer-{layer:02}-delta-keys.bin");
        let expected_delta_values = format!("kv/{phase}-layer-{layer:02}-delta-values.bin");
        if json_string(event, "delta_keys", "KV event")? != expected_delta_keys
            || json_string(event, "delta_values", "KV event")? != expected_delta_values
            || json_bytes(event, "delta_keys_bytes", "KV event")? != 32 * 64 * 2
            || json_bytes(event, "delta_values_bytes", "KV event")? != 32 * 64 * 2
            || json_bytes(event, "delta_keys_offset", "KV event")? != step as u64 * 32 * 64 * 2
            || json_bytes(event, "delta_values_offset", "KV event")? != step as u64 * 32 * 64 * 2
        {
            return Err(invalid(
                "KV delta artifact association is not authenticated",
            ));
        }
        let end_offset = integer_array(
            event
                .get("end_offset")
                .ok_or_else(|| invalid("KV event end_offset missing"))?,
            "KV event end_offset",
        )?;
        if end_offset.as_slice() != [step as i64 + 1] {
            return Err(invalid("KV event end_offset is not causal"));
        }
        for name in [&expected_delta_keys, &expected_delta_values] {
            let row = artifacts
                .get(name.as_str())
                .ok_or_else(|| invalid(format!("KV delta artifact {name} is not declared")))?;
            let expected_bytes = if phase == "warmup" {
                377 * 32 * 64 * 2
            } else {
                32 * 64 * 2
            };
            if json_bytes(row, "bytes", &format!("artifacts.{name}"))? != expected_bytes {
                return Err(invalid(format!("KV delta artifact {name} has wrong size")));
            }
        }
        let Some(keys) = event.get("keys").and_then(JsonValue::as_str) else {
            if event.get("values").is_some() {
                return Err(invalid("KV event has values without keys artifact"));
            }
            continue;
        };
        let values = event
            .get("values")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| invalid("KV snapshot is missing values artifact"))?;
        if integer_array(
            event
                .get("shape")
                .ok_or_else(|| invalid("KV snapshot shape missing"))?,
            "KV snapshot shape",
        )?
        .as_slice()
            != [1, 32, 375, 64]
            || json_string(event, "dtype", "KV snapshot")? != "torch.bfloat16"
        {
            return Err(invalid("KV snapshot metadata is not the producer contract"));
        }
        let mut positions = Vec::new();
        flatten_signed_integers(
            event
                .get("positions")
                .ok_or_else(|| invalid("KV snapshot is missing positions"))?,
            &mut positions,
        )?;
        validate_kv_positions(step, &positions)?;
        for name in [keys, values] {
            safe_artifact_name(name)?;
            let row = artifacts
                .get(name)
                .ok_or_else(|| invalid(format!("KV artifact {name} is not declared")))?;
            if json_bytes(row, "bytes", &format!("artifacts.{name}"))? != 32 * 375 * 64 * 2 {
                return Err(invalid(format!(
                    "KV snapshot artifact {name} has wrong size"
                )));
            }
        }
        let expected_keys = format!("kv/{phase}-step-{step:04}-layer-{layer:02}-keys.bin");
        let expected_values = format!("kv/{phase}-step-{step:04}-layer-{layer:02}-values.bin");
        if keys != expected_keys || values != expected_values {
            return Err(invalid(
                "KV snapshot artifact name is not the producer contract",
            ));
        }
        if parsed
            .insert(
                (phase, step, layer),
                KvEvent {
                    positions,
                    keys: keys.to_owned(),
                    values: values.to_owned(),
                },
            )
            .is_some()
        {
            return Err(invalid("duplicate KV snapshot event"));
        }
    }
    let expected = expected_checkpoint_coordinates();
    if parsed.keys().cloned().collect::<BTreeSet<_>>() != expected {
        return Err(invalid(
            "reference KV checkpoint coverage is not the exact five warmup plus reset set",
        ));
    }
    Ok(parsed)
}

fn authenticate_approval(path: &Path, packet: &ComponentPacket, checkout: &Path) -> Result<()> {
    let (body, _) = read_bounded_file(path, 64 * 1024, "composite approval")?;
    if hex_digest(&sha256_bytes(&body)) != packet.approval_sha256 {
        return Err(invalid("composite approval SHA-256 mismatch"));
    }
    let value =
        json::parse(&body).map_err(|error| invalid(format!("composite approval JSON: {error}")))?;
    reject_duplicate_keys(&value, "composite approval")?;
    exact_object(
        &value,
        &[
            "schema",
            "scope",
            "decision",
            "execution",
            "head",
            "checkout",
        ],
        "composite approval",
    )?;
    if json_string(&value, "schema", "composite approval")?
        != "vokra-kyutai-stt-pytorch-pcm-oracle-approval-v1"
        || json_string(&value, "scope", "composite approval")?
            != "KYUTAI_STT_PYTORCH_PCM_ORACLE_CAPTURE"
        || json_string(&value, "decision", "composite approval")? != "APPROVED"
        || json_string(&value, "execution", "composite approval")? != "VAST_ONLY"
        || json_string(&value, "head", "composite approval")? != packet.expected_head
    {
        return Err(invalid("composite approval facts do not match the packet"));
    }
    let approved_checkout = json_string(&value, "checkout", "composite approval")?;
    let approved_path = Path::new(approved_checkout);
    if !approved_path.is_absolute()
        || approved_path.components().any(|component| {
            matches!(
                component,
                std::path::Component::CurDir | std::path::Component::ParentDir
            )
        })
    {
        return Err(invalid(
            "composite approval checkout must be an absolute clean path",
        ));
    }
    let approved_checkout = fs::canonicalize(approved_checkout)
        .map_err(|error| invalid(format!("composite approval checkout: {error}")))?;
    let actual_checkout =
        fs::canonicalize(checkout).map_err(|error| invalid(format!("actual checkout: {error}")))?;
    if approved_checkout != actual_checkout {
        return Err(invalid(
            "composite approval checkout does not match actual checkout",
        ));
    }
    Ok(())
}

fn authenticate_reference(
    reference_root: &Path,
    packet: &ComponentPacket,
    expected_source_head: &str,
) -> Result<ReferenceEvidence> {
    let root_meta = fs::symlink_metadata(reference_root)
        .map_err(|error| invalid(format!("streaming reference: {error}")))?;
    if !root_meta.is_dir() || root_meta.file_type().is_symlink() {
        return Err(invalid(
            "streaming reference must be a non-symlink directory",
        ));
    }
    let manifest_path = reference_root.join("manifest.json");
    let (manifest_body, _) = read_bounded_file(
        &manifest_path,
        REFERENCE_MANIFEST_MAX_BYTES,
        "streaming reference manifest",
    )?;
    if hex_digest(&sha256_bytes(&manifest_body)) != packet.reference_manifest_sha256 {
        return Err(invalid("streaming reference manifest SHA-256 mismatch"));
    }
    let manifest = json::parse(&manifest_body)
        .map_err(|error| invalid(format!("streaming reference manifest JSON: {error}")))?;
    reject_duplicate_keys(&manifest, "streaming reference manifest")?;
    exact_object(
        &manifest,
        &[
            "format",
            "status",
            "scope",
            "claim_boundary",
            "expected_head",
            "dependency_closure_sha256",
            "approval_sha256",
            "source",
            "model",
            "input",
            "mimi",
            "lm",
            "execution",
            "coverage",
            "joint_steps",
            "text",
            "ring_cache",
            "artifacts",
            "artifact_budget_bytes",
        ],
        "streaming manifest",
    )?;
    let format = json_string(&manifest, "format", "streaming manifest")?;
    if format != PCM_REFERENCE_SCHEMA
        || json_string(&manifest, "status", "streaming manifest")? != "REFERENCE_READY"
        || !json_string(&manifest, "claim_boundary", "streaming manifest")?
            .contains("no native parity")
        || json_string(&manifest, "expected_head", "streaming manifest")? != expected_source_head
        || json_string(&manifest, "dependency_closure_sha256", "streaming manifest")?
            != packet.dependency_closure_sha256
        || json_string(&manifest, "approval_sha256", "streaming manifest")?
            != packet.approval_sha256
    {
        return Err(invalid(
            "streaming manifest is not bound to the packet/source boundary",
        ));
    }
    let model = manifest
        .get("model")
        .ok_or_else(|| invalid("streaming manifest model missing"))?;
    exact_object(
        model,
        &["status", "files", "tensor_manifest_sha256"],
        "streaming model",
    )?;
    if json_string(model, "status", "streaming model")? != "AUTHENTICATED_FOUR_FILE_COMPOSITE_INPUT"
    {
        return Err(invalid("streaming model status is not authenticated"));
    }
    if json_string(model, "tensor_manifest_sha256", "streaming model")?
        != MODEL_TENSOR_MANIFEST_SHA256
    {
        return Err(invalid("streaming model tensor manifest pin mismatch"));
    }
    let files = model
        .get("files")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| invalid("streaming model files missing"))?;
    let expected_model_names =
        BTreeSet::from(["model.safetensors", MIMI_NAME, TOKENIZER_NAME, CONFIG_NAME]);
    if files.len() != expected_model_names.len()
        || files
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<BTreeSet<_>>()
            != expected_model_names
    {
        return Err(invalid(
            "streaming model files must be exactly the four pinned inputs",
        ));
    }
    let model_row = files
        .iter()
        .find(|(name, _)| name == "model.safetensors")
        .map(|(_, value)| value)
        .ok_or_else(|| invalid("streaming model main file pin missing"))?;
    exact_object(model_row, &["bytes", "sha256"], "streaming model main file")?;
    if json_bytes(model_row, "bytes", "streaming model main file")? != MODEL_BYTES
        || json_string(model_row, "sha256", "streaming model main file")? != MODEL_SHA256
    {
        return Err(invalid("streaming model main file pin mismatch"));
    }
    for (name, bytes, digest) in [
        (MIMI_NAME, MIMI_BYTES, MIMI_SHA256),
        (TOKENIZER_NAME, TOKENIZER_BYTES, TOKENIZER_SHA256),
        (CONFIG_NAME, CONFIG_BYTES, CONFIG_SHA256),
    ] {
        let row = files
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, value)| value)
            .ok_or_else(|| invalid(format!("streaming model file pin missing: {name}")))?;
        exact_object(
            row,
            &["bytes", "sha256"],
            &format!("streaming model file {name}"),
        )?;
        if json_bytes(row, "bytes", &format!("streaming model file {name}"))? != bytes
            || json_string(row, "sha256", &format!("streaming model file {name}"))? != digest
        {
            return Err(invalid(format!(
                "streaming model file pin mismatch: {name}"
            )));
        }
    }
    let sources = manifest
        .get("source")
        .ok_or_else(|| invalid("streaming manifest source missing"))?;
    exact_object(
        sources,
        &["pcm", "lm", "dsm", "moshi", "contract"],
        "source",
    )?;
    validate_pcm_lm_source_roles(sources)?;
    let pcm_source = sources
        .get("pcm")
        .ok_or_else(|| invalid("streaming source pcm missing"))?;
    exact_object(
        pcm_source,
        &[
            "path",
            "manifest_sha256",
            "dsm_tree_sha",
            "moshi_tree_sha",
            "status",
        ],
        "source.pcm",
    )?;
    if json_string(pcm_source, "path", "source.pcm")?.is_empty()
        || json_string(pcm_source, "manifest_sha256", "source.pcm")?
            != SOURCE_PACKET_MANIFEST_SHA256
        || json_string(pcm_source, "dsm_tree_sha", "source.pcm")? != DSM_TREE_SHA
        || json_string(pcm_source, "moshi_tree_sha", "source.pcm")? != MOSHI_TREE_SHA
        || json_string(pcm_source, "status", "source.pcm")? != "SOURCE_ONLY_PREPARATION"
    {
        return Err(invalid("streaming PCM source identity mismatch"));
    }
    let lm_source = sources
        .get("lm")
        .ok_or_else(|| invalid("streaming source lm missing"))?;
    exact_object(
        lm_source,
        &[
            "path",
            "manifest_sha256",
            "moshi_lm_sha256",
            "moshi_transformer_sha256",
            "status",
        ],
        "source.lm",
    )?;
    if json_string(lm_source, "path", "source.lm")?.is_empty()
        || json_string(lm_source, "manifest_sha256", "source.lm")?
            != json_string(pcm_source, "manifest_sha256", "source.pcm")?
        || json_string(lm_source, "moshi_lm_sha256", "source.lm")? != SOURCE_MOSHI_LM_SHA256
        || json_string(lm_source, "moshi_transformer_sha256", "source.lm")?
            != SOURCE_MOSHI_TRANSFORMER_SHA256
        || json_string(lm_source, "status", "source.lm")? != "SOURCE_ONLY_PREPARATION"
    {
        return Err(invalid("streaming LM source identity mismatch"));
    }
    lower_hex(
        json_string(lm_source, "manifest_sha256", "source.lm")?,
        "source.lm.manifest_sha256",
    )?;
    for (name, repository, revision) in [
        ("dsm", DSM_REPOSITORY, DSM_REVISION),
        ("moshi", MOSHI_REPOSITORY, MOSHI_REVISION),
    ] {
        let source = sources
            .get(name)
            .ok_or_else(|| invalid(format!("streaming source {name} missing")))?;
        exact_object(
            source,
            &["repository", "revision", "origin"],
            &format!("sources.{name}"),
        )?;
        if json_string(source, "repository", &format!("sources.{name}"))? != repository
            || json_string(source, "revision", &format!("sources.{name}"))? != revision
            || json_string(source, "origin", &format!("sources.{name}"))?.is_empty()
        {
            return Err(invalid(format!("streaming source {name} pin mismatch")));
        }
    }
    let source_contract = sources
        .get("contract")
        .ok_or_else(|| invalid("streaming source contract missing"))?;
    validate_source_contract(source_contract)?;
    let input = manifest
        .get("input")
        .ok_or_else(|| invalid("streaming manifest input missing"))?;
    let raw_pcm = input
        .get("raw_pcm")
        .ok_or_else(|| invalid("streaming manifest raw PCM identity missing"))?;
    if json_bytes(raw_pcm, "bytes", "streaming input.raw_pcm")? != packet.pcm.bytes
        || json_string(raw_pcm, "sha256", "streaming input.raw_pcm")? != packet.pcm.sha256
        || json_bytes(raw_pcm, "sample_rate", "streaming input.raw_pcm")? != EXPECTED_SAMPLE_RATE
        || json_bytes(raw_pcm, "channels", "streaming input.raw_pcm")? != EXPECTED_CHANNELS
        || json_string(raw_pcm, "dtype", "streaming input.raw_pcm")? != EXPECTED_DTYPE
    {
        return Err(invalid(
            "reference raw PCM identity does not match the transfer packet",
        ));
    }
    let padding = input
        .get("padding")
        .ok_or_else(|| invalid("streaming input padding missing"))?;
    if packet.pcm.bytes % 4 != 0
        || json_bytes(padding, "sample_rate", "padding")? != EXPECTED_SAMPLE_RATE
        || json_bytes(padding, "frame_size", "padding")? != 1_920
        || json_f64(padding, "prefix_seconds", "padding")? != 1.0
        || json_f64(padding, "suffix_seconds", "padding")? != 3.0
        || json_bytes(padding, "prefix_samples", "padding")? != EXPECTED_SAMPLE_RATE
        || json_bytes(padding, "suffix_samples", "padding")? != 3 * EXPECTED_SAMPLE_RATE
        || json_bytes(padding, "unrounded_samples", "padding")?
            != packet.pcm.bytes / 4 + EXPECTED_SAMPLE_RATE + 3 * EXPECTED_SAMPLE_RATE
        || json_bytes(padding, "padded_samples", "padding")?
            != ((packet.pcm.bytes / 4 + 4 * EXPECTED_SAMPLE_RATE + 1_919) / 1_920) * 1_920
        || json_bytes(padding, "frames", "padding")? != 377
        || json_string(padding, "padding", "padding")?
            != "official evaluator prefix/suffix then ceil to frame_size"
    {
        return Err(invalid("reference official PCM padding schedule mismatch"));
    }
    let mimi = manifest
        .get("mimi")
        .ok_or_else(|| invalid("streaming Mimi geometry missing"))?;
    if json_bytes(mimi, "sample_rate", "mimi")? != EXPECTED_SAMPLE_RATE
        || json_bytes(mimi, "frame_size", "mimi")? != 1_920
        || json_bytes(mimi, "channels", "mimi")? != EXPECTED_CHANNELS
        || json_bytes(mimi, "num_codebooks", "mimi")? != 32
    {
        return Err(invalid(
            "reference Mimi geometry is not the native contract",
        ));
    }
    let lm = manifest
        .get("lm")
        .ok_or_else(|| invalid("streaming LM geometry missing"))?;
    if json_bytes(lm, "n_q", "lm")? != 32
        || json_bytes(lm, "dep_q", "lm")? != 0
        || json_bytes(lm, "text_card", "lm")? != 4_000
        || json_bytes(lm, "dim", "lm")? != 2_048
        || json_bytes(lm, "context", "lm")? != 375
    {
        return Err(invalid("reference LM geometry is not the native contract"));
    }
    let execution = manifest
        .get("execution")
        .ok_or_else(|| invalid("streaming execution facts missing"))?;
    if json_string(execution, "implementation", "execution")?
        != "official CheckpointInfo.get_mimi -> MimiModel.encode -> LMGen.step"
        || json_bytes(execution, "num_threads", "execution")? != 1
        || json_bytes(execution, "num_interop_threads", "execution")? != 1
        || !matches!(
            execution.get("deterministic_algorithms"),
            Some(JsonValue::Bool(true))
        )
        || json_string(execution, "publication", "execution")? != "NO_UPLOAD"
    {
        return Err(invalid("official execution facts are not locked"));
    }
    let environment = execution
        .get("environment")
        .ok_or_else(|| invalid("official execution environment missing"))?;
    exact_object(
        environment,
        &[
            "platform",
            "machine",
            "processor",
            "cpu_count",
            "toolchain",
            "torch_cpu_capability",
            "dtype",
        ],
        "execution.environment",
    )?;
    if json_string(environment, "platform", "execution.environment")?.is_empty()
        || json_string(environment, "machine", "execution.environment")?.is_empty()
        || json_string(environment, "toolchain", "execution.environment")?.is_empty()
        || json_string(environment, "torch_cpu_capability", "execution.environment")?.is_empty()
        || json_string(environment, "dtype", "execution.environment")? != "torch.bfloat16"
    {
        return Err(invalid(
            "official execution environment facts are incomplete",
        ));
    }
    let artifacts = manifest
        .get("artifacts")
        .ok_or_else(|| invalid("streaming reference artifacts missing"))?;
    let artifact_rows = artifacts
        .as_object()
        .ok_or_else(|| invalid("streaming reference artifacts must be an object"))?;
    let total_declared = artifact_rows.iter().try_fold(0u64, |total, (name, row)| {
        safe_artifact_name(name)?;
        let bytes = json_bytes(row, "bytes", &format!("artifacts.{name}"))?;
        total
            .checked_add(bytes)
            .ok_or_else(|| invalid("reference artifact total overflow"))
    })?;
    if total_declared > REFERENCE_TOTAL_ARTIFACT_MAX_BYTES {
        return Err(invalid("reference artifact total exceeds its bound"));
    }
    if json_bytes(&manifest, "artifact_budget_bytes", "streaming manifest")?
        != REFERENCE_TOTAL_ARTIFACT_MAX_BYTES
    {
        return Err(invalid("reference artifact budget pin mismatch"));
    }
    let mut actual_files = BTreeSet::new();
    let mut entries_seen = 0;
    collect_reference_files(
        reference_root,
        reference_root,
        &mut actual_files,
        &mut entries_seen,
        0,
    )?;
    actual_files.remove("manifest.json");
    let declared_files = artifact_rows
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<BTreeSet<_>>();
    if actual_files != declared_files {
        return Err(invalid(
            "reference artifact tree does not exactly match manifest declarations",
        ));
    }
    for name in &declared_files {
        let _ = artifact_bytes(
            reference_root,
            artifacts,
            name,
            REFERENCE_ARTIFACT_MAX_BYTES,
        )?;
    }
    let raw_pcm_artifact = artifact_bytes(
        reference_root,
        artifacts,
        "input/raw_pcm.f32le",
        REFERENCE_ARTIFACT_MAX_BYTES,
    )?;
    if raw_pcm_artifact.len() as u64 != packet.pcm.bytes
        || hex_digest(&sha256_bytes(&raw_pcm_artifact)) != packet.pcm.sha256
    {
        return Err(invalid(
            "reference raw PCM artifact does not match packet bytes",
        ));
    }
    let codes_body = artifact_bytes(
        reference_root,
        artifacts,
        "mimi/codes.i64",
        REFERENCE_ARTIFACT_MAX_BYTES,
    )?;
    if codes_body.len() % (32 * 8) != 0 {
        return Err(invalid("reference Mimi code artifact is not frame-aligned"));
    }
    let codes = codes_body
        .chunks_exact(32 * 8)
        .map(|frame| {
            let mut row = [0u32; 32];
            for (index, bytes) in frame.chunks_exact(8).enumerate() {
                let value = u64::from_le_bytes(bytes.try_into().expect("eight-byte code"));
                if value >= 2_048 {
                    return Err(invalid("reference Mimi code is outside cardinality"));
                }
                row[index] = value as u32;
            }
            Ok(row)
        })
        .collect::<Result<Vec<_>>>()?;
    let logits_body = artifact_bytes(
        reference_root,
        artifacts,
        "lm/text_logits.f32",
        REFERENCE_ARTIFACT_MAX_BYTES,
    )?;
    if logits_body.len() % (4_000 * 4) != 0 {
        return Err(invalid("reference logits artifact is not row-aligned"));
    }
    let logits = logits_body
        .chunks_exact(4_000 * 4)
        .map(|row| {
            row.chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte logit")))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if !logits.iter().flatten().all(|value| value.is_finite()) {
        return Err(invalid("reference logits contain non-finite values"));
    }
    let tokens_body = artifact_bytes(
        reference_root,
        artifacts,
        "lm/text_tokens.i64",
        REFERENCE_ARTIFACT_MAX_BYTES,
    )?;
    if tokens_body.len() % 8 != 0 {
        return Err(invalid("reference token artifact is not row-aligned"));
    }
    let tokens = tokens_body
        .chunks_exact(8)
        .map(|bytes| {
            let token = u64::from_le_bytes(bytes.try_into().expect("eight-byte token"));
            if token >= 4_000 {
                Err(invalid("reference token outside text card"))
            } else {
                Ok(token as u32)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let step_outputs_body = artifact_bytes(
        reference_root,
        artifacts,
        "lm/step_outputs.i64",
        REFERENCE_ARTIFACT_MAX_BYTES,
    )?;
    if step_outputs_body.len() != tokens.len() * 8 {
        return Err(invalid(
            "reference LM step-output artifact row count mismatch",
        ));
    }
    let step_outputs = step_outputs_body
        .chunks_exact(8)
        .map(|bytes| u64::from_le_bytes(bytes.try_into().expect("eight-byte step output")))
        .collect::<Vec<_>>();
    if codes.len() != 378 || logits.len() != 378 || tokens.len() != 378 || step_outputs.len() != 378
    {
        return Err(invalid(
            "reference PCM evidence must contain exactly 378 ordered rows",
        ));
    }
    let coverage = manifest
        .get("coverage")
        .ok_or_else(|| invalid("streaming coverage missing"))?;
    if json_bytes(coverage, "warmup_frames", "coverage")? != 377
        || json_bytes(coverage, "post_reset_frames", "coverage")? != 1
        || json_bytes(coverage, "mimi_encode_calls", "coverage")? != codes.len() as u64
        || json_bytes(coverage, "lm_step_calls", "coverage")? != 378
        || json_bytes(coverage, "text_events", "coverage")? != tokens.len() as u64
        || json_bytes(coverage, "logit_events", "coverage")? != logits.len() as u64
    {
        return Err(invalid(
            "reference coverage does not prove warmup/reset scheduling",
        ));
    }
    let joint_steps = manifest
        .get("joint_steps")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| invalid("reference joint_steps missing"))?;
    if joint_steps.len() != codes.len() || joint_steps.len() != tokens.len() {
        return Err(invalid(
            "reference joint PCM/code/LM event coverage mismatch",
        ));
    }
    let mut joint_coordinates = BTreeSet::new();
    for (index, event) in joint_steps.iter().enumerate() {
        let phase = json_string(event, "phase", "joint step")?;
        let step = json_bytes(event, "step", "joint step")? as usize;
        let ordinal = json_bytes(event, "lm_call_ordinal", "joint step")? as usize;
        let (expected_phase, expected_step) = if index < 377 {
            ("warmup", index)
        } else {
            ("after_reset", 0)
        };
        if !((phase == "warmup" && step < 377 && ordinal == 0)
            || (phase == "after_reset" && step == 0 && ordinal == 0))
            || phase != expected_phase
            || step != expected_step
            || !joint_coordinates.insert((phase.to_owned(), step, ordinal))
        {
            return Err(invalid("reference joint step coordinate/ordinal mismatch"));
        }
        let codes_meta = event
            .get("codes")
            .ok_or_else(|| invalid("joint step code metadata missing"))?;
        if integer_array(
            codes_meta
                .get("shape")
                .ok_or_else(|| invalid("joint step code shape missing"))?,
            "joint step code shape",
        )?
        .as_slice()
            != [1, 32, 1]
            || json_string(codes_meta, "dtype", "joint step codes")? != "torch.int64"
            || json_bytes(codes_meta, "bytes", "joint step codes")? != 32 * 8
            || json_bytes(codes_meta, "offset", "joint step codes")? != index as u64 * 32 * 8
            || json_bytes(event, "pcm_offset_samples", "joint step")?
                != if index < 377 { index as u64 * 1_920 } else { 0 }
            || json_bytes(event, "pcm_samples", "joint step")? != 1_920
        {
            return Err(invalid("joint step code metadata mismatch"));
        }
        let output = event
            .get("lm_output")
            .ok_or_else(|| invalid("joint step LM output metadata missing"))?;
        if !matches!(output.get("returned"), Some(JsonValue::Bool(true)))
            || integer_array(
                output
                    .get("shape")
                    .ok_or_else(|| invalid("joint step LM output shape missing"))?,
                "joint step LM output shape",
            )?
            .as_slice()
                != [1, 1, 1]
            || json_bytes(output, "bytes", "joint step LM output")? != 8
            || json_string(output, "dtype", "joint step LM output")? != "torch.int64"
            || json_bytes(output, "offset", "joint step LM output")? != index as u64 * 8
            || integer_array(
                output
                    .get("values")
                    .ok_or_else(|| invalid("joint step LM output values missing"))?,
                "joint step LM output values",
            )?
            .as_slice()
                != [step_outputs[index] as i64]
        {
            return Err(invalid("joint step LM output metadata mismatch"));
        }
    }
    if joint_coordinates.len() != 378 {
        return Err(invalid("reference joint step coverage is incomplete"));
    }
    let text = manifest
        .get("text")
        .ok_or_else(|| invalid("reference text evidence missing"))?;
    for (field, expected_len) in [("logits", logits.len()), ("tokens", tokens.len())] {
        let rows = text
            .get(field)
            .and_then(JsonValue::as_array)
            .ok_or_else(|| invalid(format!("reference text.{field} evidence missing")))?;
        if rows.len() != expected_len {
            return Err(invalid(format!("reference text.{field} coverage mismatch")));
        }
        let mut coordinates = BTreeSet::new();
        for (index, row) in rows.iter().enumerate() {
            let phase = json_string(row, "phase", &format!("text.{field}"))?;
            let step = json_bytes(row, "step", &format!("text.{field}"))? as usize;
            let ordinal = json_bytes(row, "lm_call_ordinal", &format!("text.{field}"))? as usize;
            let (expected_phase, expected_step) = if index < 377 {
                ("warmup", index)
            } else {
                ("after_reset", 0)
            };
            if !((phase == "warmup" && step < 377 && ordinal == 0)
                || (phase == "after_reset" && step == 0 && ordinal == 0))
                || phase != expected_phase
                || step != expected_step
                || !coordinates.insert((phase.to_owned(), step, ordinal))
            {
                return Err(invalid(format!(
                    "reference text.{field} coordinate mismatch"
                )));
            }
            if field == "logits" {
                let source_shape = integer_array(
                    row.get("source_shape")
                        .ok_or_else(|| invalid("text logits source shape missing"))?,
                    "text logits source shape",
                )?;
                let comparison_shape = integer_array(
                    row.get("comparison_shape")
                        .ok_or_else(|| invalid("text logits comparison shape missing"))?,
                    "text logits comparison shape",
                )?;
                if source_shape.as_slice() != [1, 1, 1, 4_000]
                    || comparison_shape.as_slice() != [1, 4_000]
                    || json_string(row, "dtype", "text logits")? != "torch.float32"
                    || json_bytes(row, "offset", "text logits")? != index as u64 * 4_000 * 4
                    || json_bytes(row, "bytes", "text logits")? != 4_000 * 4
                    || json_string(row, "source_dtype", "text logits")? != "torch.bfloat16"
                    || json_string(row, "conversion", "text logits")?
                        != "official logits converted to contiguous torch.float32 for comparison export"
                {
                    return Err(invalid("text logits metadata mismatch"));
                }
            }
            if field == "tokens" {
                let shape = integer_array(
                    row.get("shape")
                        .ok_or_else(|| invalid("text token shape missing"))?,
                    "text token shape",
                )?;
                let values = integer_array(
                    row.get("values")
                        .ok_or_else(|| invalid("text token values missing"))?,
                    "text token values",
                )?;
                if shape.as_slice() != [1]
                    || json_string(row, "dtype", "text token")? != "torch.int64"
                    || json_bytes(row, "offset", "text token")? != index as u64 * 8
                    || json_bytes(row, "bytes", "text token")? != 8
                    || values.as_slice() != [tokens[index] as i64]
                {
                    return Err(invalid("text token metadata mismatch"));
                }
            }
        }
        if coordinates.len() != expected_len {
            return Err(invalid(format!(
                "reference text.{field} coordinate coverage is incomplete"
            )));
        }
    }
    let ring_cache = manifest
        .get("ring_cache")
        .ok_or_else(|| invalid("streaming ring_cache missing"))?;
    if json_bytes(ring_cache, "lm_calls", "ring_cache")? != codes.len() as u64
        || json_bytes(ring_cache, "calls_per_pcm_frame", "ring_cache")? != 1
        || ring_cache
            .get("reset_mask")
            .and_then(JsonValue::as_array)
            .map(|mask| mask.len() != 1 || !matches!(&mask[0], JsonValue::Bool(true)))
            .unwrap_or(true)
    {
        return Err(invalid(
            "reference ring-cache reset/call schedule is not authenticated",
        ));
    }
    let snapshots = ring_cache
        .get("state_snapshots")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| invalid("reference ring-cache state snapshots missing"))?;
    if snapshots.len() != 96 {
        return Err(invalid(
            "reference ring-cache state snapshots are not 96 layer snapshots",
        ));
    }
    let mut snapshot_coordinates = BTreeSet::new();
    for snapshot in snapshots {
        let phase = json_string(snapshot, "phase", "state snapshot")?;
        let step = json_bytes(snapshot, "step", "state snapshot")? as usize;
        let layer = json_bytes(snapshot, "layer", "state snapshot")? as usize;
        if !matches!(phase, "initial" | "after_reset")
            || step != 0
            || layer >= 48
            || json_bytes(snapshot, "capacity", "state snapshot")? != 375
            || !snapshot_coordinates.insert((phase.to_owned(), step, layer))
        {
            return Err(invalid("reference ring-cache snapshot coordinate mismatch"));
        }
        let end_offset = snapshot
            .get("end_offset")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| invalid("state snapshot end_offset missing"))?;
        if end_offset.len() != 1 || !matches!(&end_offset[0], JsonValue::Int(0)) {
            return Err(invalid("reference initial/reset KV end_offset is not zero"));
        }
    }
    for phase in ["initial", "after_reset"] {
        if !snapshots.iter().any(|snapshot| {
            json_string(snapshot, "phase", "state snapshot")
                .map(|value| value == phase)
                .unwrap_or(false)
                && json_bytes(snapshot, "step", "state snapshot")
                    .map(|value| value == 0)
                    .unwrap_or(false)
        }) {
            return Err(invalid(format!(
                "reference ring-cache {phase} snapshot missing"
            )));
        }
    }
    let kv_events = parse_kv_events(&manifest, artifacts)?;
    let declared_kv_bytes = artifact_rows
        .iter()
        .filter(|(name, _)| name.starts_with("kv/"))
        .try_fold(0u64, |total, (_, row)| {
            total
                .checked_add(json_bytes(row, "bytes", "KV artifact")?)
                .ok_or_else(|| invalid("KV artifact byte total overflow"))
        })?;
    if declared_kv_bytes != json_bytes(ring_cache, "kv_artifact_bytes", "ring_cache")? {
        return Err(invalid("ring-cache KV artifact byte total is not bound"));
    }
    let mut referenced = BTreeSet::from([
        "input/raw_pcm.f32le".to_owned(),
        "mimi/codes.i64".to_owned(),
        "lm/text_logits.f32".to_owned(),
        "lm/text_tokens.i64".to_owned(),
        "lm/step_outputs.i64".to_owned(),
    ]);
    for event in kv_events.values() {
        referenced.insert(event.keys.clone());
        referenced.insert(event.values.clone());
    }
    for phase in ["warmup", "after_reset"] {
        for layer in 0..48 {
            referenced.insert(format!("kv/{phase}-layer-{layer:02}-delta-keys.bin"));
            referenced.insert(format!("kv/{phase}-layer-{layer:02}-delta-values.bin"));
        }
    }
    if declared_files != referenced {
        return Err(invalid(
            "reference artifact tree contains an unreferenced or missing artifact",
        ));
    }
    Ok(ReferenceEvidence {
        root: reference_root.to_owned(),
        manifest,
        codes,
        logits,
        tokens,
        kv_events,
    })
}

fn load_packet_from_env(expected_head: &str) -> Result<(ComponentPacket, ReferenceEvidence)> {
    let packet_path = PathBuf::from(
        std::env::var(PACKET_ENV).map_err(|_| invalid(format!("missing {PACKET_ENV}")))?,
    );
    let packet_sha =
        std::env::var(PACKET_SHA_ENV).map_err(|_| invalid(format!("missing {PACKET_SHA_ENV}")))?;
    let _reference_path =
        std::env::var(REFERENCE_ENV).map_err(|_| invalid(format!("missing {REFERENCE_ENV}")))?;
    let _approval_path =
        std::env::var(APPROVAL_ENV).map_err(|_| invalid(format!("missing {APPROVAL_ENV}")))?;
    let decoder_sha256 =
        std::env::var(GGUF_SHA_ENV).map_err(|_| invalid(format!("missing {GGUF_SHA_ENV}")))?;
    let reference_manifest_sha256 = std::env::var(REFERENCE_SHA_ENV)
        .map_err(|_| invalid(format!("missing {REFERENCE_SHA_ENV}")))?;
    let approval_sha256 = std::env::var(APPROVAL_SHA_ENV)
        .map_err(|_| invalid(format!("missing {APPROVAL_SHA_ENV}")))?;
    let dependency_closure_sha256 = std::env::var(CLOSURE_SHA_ENV)
        .map_err(|_| invalid(format!("missing {CLOSURE_SHA_ENV}")))?;
    let expected = ExistingBindings {
        expected_head,
        decoder_sha256: &decoder_sha256,
        reference_manifest_sha256: &reference_manifest_sha256,
        approval_sha256: &approval_sha256,
        dependency_closure_sha256: &dependency_closure_sha256,
    };
    lower_hex(&packet_sha, PACKET_SHA_ENV)?;
    let (body, _) = read_bounded_file(&packet_path, PACKET_MAX_BYTES, "component packet")?;
    if hex_digest(&sha256_bytes(&body)) != packet_sha {
        return Err(invalid("component packet SHA-256 mismatch"));
    }
    let packet = parse_packet(&body)?;
    validate_gate(&packet, expected)?;
    let reference_path = PathBuf::from(_reference_path);
    let approval_path = PathBuf::from(_approval_path);
    let reference = authenticate_reference(&reference_path, &packet, expected_head)?;
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| invalid("cannot derive checkout root"))?;
    authenticate_approval(&approval_path, &packet, checkout)?;
    validate_disjoint_paths(&packet, &packet_path)?;
    validate_external_disjoint(&packet, &packet_path, &reference_path, &approval_path)?;
    let main_decoder =
        PathBuf::from(std::env::var(GGUF_ENV).map_err(|_| invalid(format!("missing {GGUF_ENV}")))?);
    let decoder = packet
        .components
        .iter()
        .find(|item| item.role == "decoder")
        .unwrap();
    if main_decoder != decoder.path {
        return Err(invalid(
            "component decoder path does not match the main GGUF transfer binding",
        ));
    }
    Ok((packet, reference))
}

fn verify_host_and_head(expected_head: &str) -> Result<()> {
    if std::env::consts::OS != "linux" || std::env::consts::ARCH != "x86_64" {
        return Err(invalid("authenticated PCM consumer requires Linux x86_64"));
    }
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| invalid("cannot derive checkout root"))?;
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(checkout)
        .output()
        .map_err(|error| invalid(format!("git HEAD check failed: {error}")))?;
    if !output.status.success() || String::from_utf8_lossy(&output.stdout).trim() != expected_head {
        return Err(invalid(
            "checkout HEAD does not match authenticated expected_head",
        ));
    }
    let status = std::process::Command::new("git")
        .args(["status", "--porcelain=v1"])
        .current_dir(checkout)
        .output()
        .map_err(|error| invalid(format!("git clean check failed: {error}")))?;
    if !status.status.success() || !status.stdout.is_empty() {
        return Err(invalid(
            "checkout is not clean for authenticated PCM execution",
        ));
    }
    Ok(())
}

#[derive(Debug)]
struct BoundedPcmObserver<'a> {
    reference: &'a ReferenceEvidence,
    after_reset: bool,
    frames: usize,
    calls: usize,
    current_codes: Option<[u32; 32]>,
    call_ordinals: Vec<(usize, usize)>,
    kv_views: usize,
    common_code_frames: usize,
    unmatched_frames: usize,
    code_mismatches: usize,
    unmatched_calls: usize,
    logit_matches: usize,
    logit_mismatches: usize,
    token_matches: usize,
    token_mismatches: usize,
    kv_comparisons: usize,
    kv_mismatches: usize,
    logit_max_abs: f32,
    logit_abs_sum: f64,
    logit_values: usize,
    logit_worst: Option<(&'static str, usize, usize)>,
    kv_max_abs: f32,
    kv_abs_sum: f64,
    kv_values: usize,
    kv_worst: Option<(
        &'static str,
        usize,
        usize,
        usize,
        usize,
        usize,
        &'static str,
    )>,
}

fn aligned_reference_frame(
    after_reset: bool,
    native_frame: usize,
    ordinal: usize,
    native_position: usize,
) -> Option<usize> {
    if ordinal != 0 {
        return None;
    }
    if after_reset {
        (native_frame == 0 && native_position == 0).then_some(377)
    } else {
        (native_frame < 377 && native_position == native_frame).then_some(native_frame)
    }
}

fn code_reference_frame(after_reset: bool, native_frame: usize) -> Option<usize> {
    if after_reset {
        (native_frame == 0).then_some(377)
    } else {
        (native_frame < 377).then_some(native_frame)
    }
}

fn validate_pcm_lm_source_roles(sources: &JsonValue) -> Result<()> {
    exact_object(
        sources,
        &["pcm", "lm", "dsm", "moshi", "contract"],
        "source",
    )?;
    let pcm = sources
        .get("pcm")
        .ok_or_else(|| invalid("source.pcm missing"))?;
    exact_object(
        pcm,
        &[
            "path",
            "manifest_sha256",
            "dsm_tree_sha",
            "moshi_tree_sha",
            "status",
        ],
        "source.pcm",
    )?;
    if json_string(pcm, "manifest_sha256", "source.pcm")? != SOURCE_PACKET_MANIFEST_SHA256
        || json_string(pcm, "dsm_tree_sha", "source.pcm")? != DSM_TREE_SHA
        || json_string(pcm, "moshi_tree_sha", "source.pcm")? != MOSHI_TREE_SHA
        || json_string(pcm, "status", "source.pcm")? != "SOURCE_ONLY_PREPARATION"
    {
        return Err(invalid("source.pcm identity mismatch"));
    }
    let lm = sources
        .get("lm")
        .ok_or_else(|| invalid("source.lm missing"))?;
    exact_object(
        lm,
        &[
            "path",
            "manifest_sha256",
            "moshi_lm_sha256",
            "moshi_transformer_sha256",
            "status",
        ],
        "source.lm",
    )?;
    if json_string(lm, "moshi_lm_sha256", "source.lm")? != SOURCE_MOSHI_LM_SHA256
        || json_string(lm, "moshi_transformer_sha256", "source.lm")?
            != SOURCE_MOSHI_TRANSFORMER_SHA256
        || json_string(lm, "status", "source.lm")? != "SOURCE_ONLY_PREPARATION"
    {
        return Err(invalid("source.lm identity mismatch"));
    }
    Ok(())
}

fn validate_source_contract(value: &JsonValue) -> Result<()> {
    exact_object(
        value,
        &[
            "schema",
            "status",
            "runtime_status",
            "source_identity",
            "source_roles",
            "contracts",
            "blockers",
            "claim_boundary",
        ],
        "source.contract",
    )?;
    if json_string(value, "schema", "source.contract")? != SOURCE_CONTRACT_SCHEMA
        || json_string(value, "status", "source.contract")? != SOURCE_CONTRACT_STATUS
        || json_string(value, "runtime_status", "source.contract")?
            != SOURCE_CONTRACT_RUNTIME_STATUS
        || !json_string(value, "claim_boundary", "source.contract")?.contains("no model execution")
    {
        return Err(invalid("source contract status or claim boundary mismatch"));
    }
    let identity = value
        .get("source_identity")
        .ok_or_else(|| invalid("source.contract source_identity missing"))?;
    exact_object(
        identity,
        &["dsm", "moshi"],
        "source.contract.source_identity",
    )?;
    for (name, repository, revision) in [
        ("dsm", DSM_REPOSITORY, DSM_REVISION),
        ("moshi", MOSHI_REPOSITORY, MOSHI_REVISION),
    ] {
        let row = identity
            .get(name)
            .ok_or_else(|| invalid(format!("source identity {name} missing")))?;
        exact_object(
            row,
            &["repository", "revision", "roles"],
            &format!("source_identity.{name}"),
        )?;
        if json_string(row, "repository", name)? != repository
            || json_string(row, "revision", name)? != revision
        {
            return Err(invalid(format!("source identity {name} mismatch")));
        }
        let expected = if name == "dsm" {
            SOURCE_CONTRACT_DSM_ROLES
        } else {
            SOURCE_CONTRACT_MOSHI_ROLES
        };
        let role_rows = row
            .get("roles")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| invalid(format!("source identity {name} roles missing")))?;
        if role_rows.len() != expected.len() {
            return Err(invalid(format!(
                "source identity {name} role count mismatch"
            )));
        }
        for (role, expected_path) in role_rows.iter().zip(expected) {
            exact_object(
                role,
                &["path", "bytes", "sha256", "git_blob_sha1"],
                &format!("source_identity.{name}.role"),
            )?;
            if json_string(role, "path", name)? != *expected_path
                || json_bytes(role, "bytes", name)? == 0
            {
                return Err(invalid(format!("source identity {name} role mismatch")));
            }
            lower_hex(json_string(role, "sha256", name)?, name)?;
            lower_git_blob_sha1(json_string(role, "git_blob_sha1", name)?, name)?;
        }
    }
    let roles = value
        .get("source_roles")
        .ok_or_else(|| invalid("source.contract source_roles missing"))?;
    exact_object(roles, &["dsm", "moshi"], "source.contract.source_roles")?;
    for (name, expected) in [
        ("dsm", SOURCE_CONTRACT_DSM_ROLES),
        ("moshi", SOURCE_CONTRACT_MOSHI_ROLES),
    ] {
        let values = roles
            .get(name)
            .and_then(JsonValue::as_array)
            .ok_or_else(|| invalid(format!("source role list {name} missing")))?;
        if values.len() != expected.len()
            || values
                .iter()
                .zip(expected)
                .any(|(value, expected)| value.as_str() != Some(*expected))
        {
            return Err(invalid(format!("source role list {name} mismatch")));
        }
    }
    let contracts = value
        .get("contracts")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| invalid("source.contract contracts missing"))?;
    if contracts.len() != 3 {
        return Err(invalid("source.contract contract coverage mismatch"));
    }
    for contract in contracts {
        let name = json_string(contract, "name", "source contract row")?;
        if name == "temperature_zero_selection" {
            exact_object(
                contract,
                &["name", "status", "roles", "expressions", "tie_resolution"],
                "source.contract contract row",
            )?;
            if json_string(contract, "tie_resolution", "source contract row")?
                != "BLOCKED_EXTERNAL_TORCH_SEMANTICS"
            {
                return Err(invalid("source contract tie resolution mismatch"));
            }
        } else {
            exact_object(
                contract,
                &["name", "status", "roles", "expressions"],
                "source.contract contract row",
            )?;
        }
        if json_string(contract, "status", "source contract row")?
            != "SOURCE_EXPRESSION_AUTHENTICATED"
            || contract
                .get("roles")
                .and_then(JsonValue::as_array)
                .map_or(true, |values| values.is_empty())
            || contract
                .get("expressions")
                .and_then(JsonValue::as_array)
                .map_or(true, |values| values.is_empty())
        {
            return Err(invalid("source contract row is incomplete"));
        }
    }
    let blockers = value
        .get("blockers")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| invalid("source.contract blockers missing"))?;
    if blockers.len() != 3 || blockers.iter().any(|value| value.as_str().is_none()) {
        return Err(invalid("source contract blockers are incomplete"));
    }
    Ok(())
}

impl<'a> BoundedPcmObserver<'a> {
    fn new(reference: &'a ReferenceEvidence, after_reset: bool) -> Self {
        Self {
            reference,
            after_reset,
            frames: 0,
            calls: 0,
            current_codes: None,
            call_ordinals: Vec::new(),
            kv_views: 0,
            common_code_frames: 0,
            unmatched_frames: 0,
            code_mismatches: 0,
            unmatched_calls: 0,
            logit_matches: 0,
            logit_mismatches: 0,
            token_matches: 0,
            token_mismatches: 0,
            kv_comparisons: 0,
            kv_mismatches: 0,
            logit_max_abs: 0.0,
            logit_abs_sum: 0.0,
            logit_values: 0,
            logit_worst: None,
            kv_max_abs: 0.0,
            kv_abs_sum: 0.0,
            kv_values: 0,
            kv_worst: None,
        }
    }
}

fn greedy_index(values: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

impl PcmObserver for BoundedPcmObserver<'_> {
    fn on_frame(&mut self, frame: &[f32], codes: &[u32]) -> Result<()> {
        if self.frames >= MAX_FRAMES || frame.len() != 1_920 || codes.len() != 32 {
            return Err(invalid("native PCM observer exceeded frame/shape bounds"));
        }
        if !frame.iter().all(|value| value.is_finite()) {
            return Err(invalid("native PCM observer saw non-finite frame data"));
        }
        // The official evaluator prepends exactly one second of zero PCM;
        // the first frame in both warmup and post-reset must therefore be the
        // same padded silence frame.  This is a direct schedule check, not a
        // shifted/fitted comparison against a later frame.
        if self.frames == 0 && !frame.iter().all(|value| *value == 0.0) {
            return Err(invalid(
                "native first frame is not the authenticated padded silence frame",
            ));
        }
        let mut row = [0u32; 32];
        row.copy_from_slice(codes);
        self.current_codes = Some(row);
        let expected = code_reference_frame(self.after_reset, self.frames)
            .and_then(|index| self.reference.codes.get(index));
        if let Some(expected) = expected {
            self.common_code_frames += 1;
            if expected != &row {
                self.code_mismatches += 1;
            }
        } else {
            self.unmatched_frames += 1;
        }
        self.frames += 1;
        Ok(())
    }

    fn on_lm_call(
        &mut self,
        _previous_text_token: Option<u32>,
        codes: &[u32],
        step: &super::super::KyutaiSttStreamingLmStep,
        lm: &super::super::KyutaiSttStreamingLm<'_>,
    ) -> Result<()> {
        if self.calls >= MAX_CALLS
            || self.current_codes.as_ref().map(|row| row.as_slice()) != Some(codes)
        {
            return Err(invalid(
                "native LM observer saw an unbound or over-budget code row",
            ));
        }
        if step.logits().frames() != 1 || step.logits().vocab() != 4_000 {
            return Err(invalid("native LM observer saw unexpected logits geometry"));
        }
        let frame = self
            .frames
            .checked_sub(1)
            .ok_or_else(|| invalid("LM call preceded Mimi frame"))?;
        let ordinal = self
            .call_ordinals
            .iter()
            .rev()
            .find(|(known_frame, _)| *known_frame == frame)
            .map_or(0, |(_, ordinal)| ordinal + 1);
        let reference_frame =
            aligned_reference_frame(self.after_reset, frame, ordinal, step.position());
        let causally_aligned = reference_frame.is_some();
        if !causally_aligned {
            self.unmatched_calls += 1;
        } else {
            let reference_frame = reference_frame.expect("causally aligned frame");
            let reference_logits = self.reference.logits.get(reference_frame);
            let reference_token = self.reference.tokens.get(reference_frame);
            if let Some(reference_logits) = reference_logits {
                for (index, (native, expected)) in step
                    .logits()
                    .as_slice()
                    .iter()
                    .zip(reference_logits)
                    .enumerate()
                {
                    if !native.is_finite() || !expected.is_finite() {
                        return Err(invalid("non-finite aligned logit"));
                    }
                    let difference = (native - expected).abs();
                    self.logit_values += 1;
                    self.logit_abs_sum += f64::from(difference);
                    if difference > self.logit_max_abs {
                        self.logit_max_abs = difference;
                        self.logit_worst = Some((
                            if self.after_reset {
                                "after_reset"
                            } else {
                                "warmup"
                            },
                            frame,
                            index,
                        ));
                    }
                }
                if step.logits().as_slice() == reference_logits.as_slice() {
                    self.logit_matches += 1;
                } else {
                    self.logit_mismatches += 1;
                }
            }
            if let Some(reference_token) = reference_token {
                if greedy_index(step.logits().as_slice()) as u32 == *reference_token {
                    self.token_matches += 1;
                } else {
                    self.token_mismatches += 1;
                }
            }
        }
        for layer in 0..48 {
            let (positions, keys, values) = lm
                .layer_cache_view(layer)
                .ok_or_else(|| invalid(format!("native LM observer missing KV layer {layer}")))?;
            if keys.len() != positions.len() * 2_048 || values.len() != positions.len() * 2_048 {
                return Err(invalid(
                    "native LM observer saw invalid borrowed KV geometry",
                ));
            }
            if positions != expected_native_positions(step.position()).as_slice() {
                return Err(invalid(
                    "native LM observer saw a non-causal or unordered KV position window",
                ));
            }
            if ordinal == 0 && causally_aligned {
                let phase = if self.after_reset {
                    "after_reset"
                } else {
                    "warmup"
                };
                if let Some(event) = self
                    .reference
                    .kv_events
                    .get(&(phase.to_owned(), frame, layer))
                {
                    let valid = event
                        .positions
                        .iter()
                        .enumerate()
                        .filter_map(|(physical, position)| {
                            (*position >= 0).then_some((physical, *position))
                        })
                        .collect::<Vec<_>>();
                    let mut valid = valid;
                    valid.sort_by_key(|(_, position)| *position);
                    let expected_positions = valid
                        .iter()
                        .map(|(_, position)| *position as usize)
                        .collect::<Vec<_>>();
                    if positions != expected_positions.as_slice() {
                        return Err(invalid(
                            "native/official causal KV position sets are not aligned",
                        ));
                    }
                    let key_body = artifact_bytes(
                        &self.reference.root,
                        self.reference.manifest.get("artifacts").expect("artifacts"),
                        &event.keys,
                        REFERENCE_ARTIFACT_MAX_BYTES,
                    )?;
                    let value_body = artifact_bytes(
                        &self.reference.root,
                        self.reference.manifest.get("artifacts").expect("artifacts"),
                        &event.values,
                        REFERENCE_ARTIFACT_MAX_BYTES,
                    )?;
                    if key_body.len() != 32 * 375 * 64 * 2 || value_body.len() != 32 * 375 * 64 * 2
                    {
                        return Err(invalid("reference KV snapshot byte geometry mismatch"));
                    }
                    for (row, (physical, _)) in valid.iter().enumerate() {
                        for head in 0..32 {
                            for dimension in 0..64 {
                                let official_index = (head * 375 + *physical) * 64 + dimension;
                                let native_index = row * 2_048 + head * 64 + dimension;
                                let key_offset = official_index * 2;
                                let value_offset = official_index * 2;
                                let official_key = bf16_to_f32(u16::from_le_bytes(
                                    key_body[key_offset..key_offset + 2]
                                        .try_into()
                                        .expect("BF16 key"),
                                ));
                                let official_value = bf16_to_f32(u16::from_le_bytes(
                                    value_body[value_offset..value_offset + 2]
                                        .try_into()
                                        .expect("BF16 value"),
                                ));
                                if !official_key.is_finite()
                                    || !official_value.is_finite()
                                    || !keys[native_index].is_finite()
                                    || !values[native_index].is_finite()
                                {
                                    return Err(invalid("non-finite KV comparison value"));
                                }
                                self.kv_comparisons += 2;
                                let key_difference = (official_key - keys[native_index]).abs();
                                let value_difference =
                                    (official_value - values[native_index]).abs();
                                self.kv_values += 2;
                                self.kv_abs_sum +=
                                    f64::from(key_difference) + f64::from(value_difference);
                                if key_difference > self.kv_max_abs {
                                    self.kv_max_abs = key_difference;
                                    self.kv_worst = Some((
                                        if self.after_reset {
                                            "after_reset"
                                        } else {
                                            "warmup"
                                        },
                                        frame,
                                        layer,
                                        row,
                                        head,
                                        dimension,
                                        "key",
                                    ));
                                }
                                if value_difference > self.kv_max_abs {
                                    self.kv_max_abs = value_difference;
                                    self.kv_worst = Some((
                                        if self.after_reset {
                                            "after_reset"
                                        } else {
                                            "warmup"
                                        },
                                        frame,
                                        layer,
                                        row,
                                        head,
                                        dimension,
                                        "value",
                                    ));
                                }
                                if official_key.to_bits() != keys[native_index].to_bits() {
                                    self.kv_mismatches += 1;
                                }
                                if official_value.to_bits() != values[native_index].to_bits() {
                                    self.kv_mismatches += 1;
                                }
                            }
                        }
                    }
                }
            }
            self.kv_views += 1;
        }
        self.call_ordinals.push((frame, ordinal));
        self.calls += 1;
        Ok(())
    }
}

#[derive(Debug)]
struct PcmDiagnostic {
    status: &'static str,
    frames: usize,
    calls: usize,
    common_code_frames: usize,
    unmatched_frames: usize,
    code_mismatches: usize,
    unmatched_calls: usize,
    logit_matches: usize,
    logit_mismatches: usize,
    token_matches: usize,
    token_mismatches: usize,
    kv_comparisons: usize,
    kv_mismatches: usize,
    logit_max_abs: f32,
    logit_abs_sum: f64,
    logit_values: usize,
    logit_worst: Option<(&'static str, usize, usize)>,
    kv_max_abs: f32,
    kv_abs_sum: f64,
    kv_values: usize,
    kv_worst: Option<(
        &'static str,
        usize,
        usize,
        usize,
        usize,
        usize,
        &'static str,
    )>,
}

fn authenticate_and_consume_pcm(
    packet: &ComponentPacket,
    reference: &ReferenceEvidence,
) -> Result<PcmDiagnostic> {
    let decoder = packet
        .components
        .iter()
        .find(|item| item.role == "decoder")
        .unwrap();
    let tokenizer = packet
        .components
        .iter()
        .find(|item| item.role == "tokenizer")
        .unwrap();
    let mimi_gguf = packet
        .components
        .iter()
        .find(|item| item.role == "mimi_gguf")
        .unwrap();
    let raw_mimi = packet
        .components
        .iter()
        .find(|item| item.role == "raw_mimi")
        .unwrap();
    verify_file_binding(
        &packet.pcm.path,
        packet.pcm.bytes,
        &packet.pcm.sha256,
        "canonical PCM",
    )?;
    for component in [decoder, tokenizer, mimi_gguf, raw_mimi] {
        verify_file_binding(
            &component.path,
            component.bytes,
            &component.sha256,
            &component.role,
        )?;
    }
    if raw_mimi.bytes != MIMI_BYTES
        || raw_mimi.sha256 != MIMI_SHA256
        || raw_mimi.source_sha256 != super::super::KYUTAI_STT_MIMI_SHA256
    {
        return Err(invalid(
            "raw Mimi source identity is not the pinned upstream identity",
        ));
    }
    let digests = KyutaiSttPcmArtifactDigests {
        decoder: KyutaiSttPcmArtifactDigest::new(&decoder.sha256, decoder.bytes)?,
        tokenizer: KyutaiSttPcmArtifactDigest::new(&tokenizer.sha256, tokenizer.bytes)?,
        mimi_gguf: KyutaiSttPcmArtifactDigest::new(&mimi_gguf.sha256, mimi_gguf.bytes)?,
    };
    let (body, _) = read_bounded_file(&packet.pcm.path, PCM_MAX_BYTES, "canonical PCM")?;
    if body.len() as u64 != packet.pcm.bytes
        || hex_digest(&sha256_bytes(&body)) != packet.pcm.sha256
    {
        return Err(invalid("canonical PCM changed after authentication"));
    }
    let samples: Vec<f32> = body
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte PCM chunk")))
        .collect();
    if !samples.iter().all(|sample| sample.is_finite()) {
        return Err(invalid("canonical PCM contains non-finite samples"));
    }
    let engine = KyutaiSttPcmEngine::from_paths(
        &decoder.path,
        &tokenizer.path,
        &mimi_gguf.path,
        &raw_mimi.path,
        &digests,
        BackendKind::Cpu,
    )?;
    let max_frames = engine.contract().padded_frame_count(samples.len())?;
    let mut session = engine.session(max_frames)?;
    verify_empty_native_kv(&session)?;
    let mut observer = BoundedPcmObserver::new(reference, false);
    session.push_pcm_with_observer(&samples, &mut observer)?;
    session.finish_with_observer(&mut observer)?;
    let warmup = (observer.frames, observer.calls);
    session.reset();
    verify_empty_native_kv(&session)?;
    let mut after_reset = BoundedPcmObserver::new(reference, true);
    session.push_pcm_with_observer(&samples[..samples.len().min(1_920)], &mut after_reset)?;
    Ok(PcmDiagnostic {
        status: "NOT_PARITY_PASS",
        frames: warmup.0 + after_reset.frames,
        calls: warmup.1 + after_reset.calls,
        common_code_frames: observer.common_code_frames + after_reset.common_code_frames,
        unmatched_frames: observer.unmatched_frames + after_reset.unmatched_frames,
        code_mismatches: observer.code_mismatches + after_reset.code_mismatches,
        unmatched_calls: observer.unmatched_calls + after_reset.unmatched_calls,
        logit_matches: observer.logit_matches + after_reset.logit_matches,
        logit_mismatches: observer.logit_mismatches + after_reset.logit_mismatches,
        token_matches: observer.token_matches + after_reset.token_matches,
        token_mismatches: observer.token_mismatches + after_reset.token_mismatches,
        kv_comparisons: observer.kv_comparisons + after_reset.kv_comparisons,
        kv_mismatches: observer.kv_mismatches + after_reset.kv_mismatches,
        logit_max_abs: observer.logit_max_abs.max(after_reset.logit_max_abs),
        logit_abs_sum: observer.logit_abs_sum + after_reset.logit_abs_sum,
        logit_values: observer.logit_values + after_reset.logit_values,
        kv_max_abs: observer.kv_max_abs.max(after_reset.kv_max_abs),
        kv_abs_sum: observer.kv_abs_sum + after_reset.kv_abs_sum,
        kv_values: observer.kv_values + after_reset.kv_values,
        logit_worst: if observer.logit_max_abs >= after_reset.logit_max_abs {
            observer.logit_worst
        } else {
            after_reset.logit_worst
        },
        kv_worst: if observer.kv_max_abs >= after_reset.kv_max_abs {
            observer.kv_worst
        } else {
            after_reset.kv_worst
        },
    })
}

fn verify_empty_native_kv(session: &super::KyutaiSttPcmSession<'_>) -> Result<()> {
    for layer in 0..48 {
        let (positions, keys, values) = session
            .lm
            .layer_cache_view(layer)
            .ok_or_else(|| invalid(format!("native initial/reset KV layer {layer} missing")))?;
        if !positions.is_empty() || !keys.is_empty() || !values.is_empty() {
            return Err(invalid("native initial/reset KV cache is not empty"));
        }
    }
    Ok(())
}

#[test]
fn component_packet_parser_is_strict_and_bounded() {
    let digest = "a".repeat(64);
    let components = ["decoder", "tokenizer", "mimi_gguf", "raw_mimi"]
        .into_iter()
        .map(|role| {
            format!(
                "{{\"role\":\"{role}\",\"path\":\"/tmp/{role}\",\"bytes\":1,\"sha256\":\"{digest}\",\"source_sha256\":\"{digest}\"}}"
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let packet = format!(
        "{{\"format\":\"{PACKET_SCHEMA}\",\"expected_head\":\"{}\",\"reference_manifest_sha256\":\"{digest}\",\"approval_sha256\":\"{digest}\",\"dependency_closure_sha256\":\"{digest}\",\"pcm\":{{\"path\":\"/tmp/input.f32le\",\"bytes\":4,\"sha256\":\"{digest}\",\"sample_rate\":24000,\"channels\":1,\"dtype\":\"float32-le\"}},\"components\":[{components}]}}",
        "b".repeat(40),
    );
    let parsed = parse_packet(packet.as_bytes()).unwrap();
    assert_eq!(parsed.components.len(), 4);
    assert!(parse_packet(packet.replace("mimi_gguf", "decoder").as_bytes()).is_err());
    assert!(parse_packet(packet.replace("float32-le", "float64-le").as_bytes()).is_err());
}

#[test]
fn producer_source_roles_require_pcm_tree_and_lm_file_identities() {
    let source = format!(
        r#"{{
            "pcm":{{"path":"/tmp/pcm-source","manifest_sha256":"{packet}","dsm_tree_sha":"{dsm}","moshi_tree_sha":"{moshi}","status":"SOURCE_ONLY_PREPARATION"}},
            "lm":{{"path":"/tmp/lm-source","manifest_sha256":"{packet}","moshi_lm_sha256":"{lm}","moshi_transformer_sha256":"{transformer}","status":"SOURCE_ONLY_PREPARATION"}},
            "dsm":{{"repository":"{dsm_repo}","revision":"{dsm_rev}","origin":"receipt"}},
            "moshi":{{"repository":"{moshi_repo}","revision":"{moshi_rev}","origin":"receipt"}},
            "contract":{{"path":"/tmp/contract","manifest_sha256":"{packet}","moshi_lm_sha256":"{lm}","moshi_transformer_sha256":"{transformer}","status":"SOURCE_ONLY_PREPARATION"}}
        }}"#,
        packet = SOURCE_PACKET_MANIFEST_SHA256,
        dsm = DSM_TREE_SHA,
        moshi = MOSHI_TREE_SHA,
        lm = SOURCE_MOSHI_LM_SHA256,
        transformer = SOURCE_MOSHI_TRANSFORMER_SHA256,
        dsm_repo = DSM_REPOSITORY,
        dsm_rev = DSM_REVISION,
        moshi_repo = MOSHI_REPOSITORY,
        moshi_rev = MOSHI_REVISION,
    );
    let value = json::parse(source.as_bytes()).unwrap();
    validate_pcm_lm_source_roles(&value).unwrap();
    let bad = source.replace("dsm_tree_sha", "moshi_lm_sha256");
    assert!(
        json::parse(bad.as_bytes())
            .ok()
            .and_then(|value| validate_pcm_lm_source_roles(&value).ok())
            .is_none()
    );
    let role = |path: &str| {
        format!(
            "{{\"path\":\"{path}\",\"bytes\":1,\"sha256\":\"{}\",\"git_blob_sha1\":\"{}\"}}",
            "a".repeat(64),
            "b".repeat(40)
        )
    };
    let contract = format!(
        r#"{{"schema":"{schema}","status":"AUTHENTICATED_SOURCE_CONTRACT","runtime_status":"BLOCKED_NOT_EXECUTED","source_identity":{{"dsm":{{"repository":"{dsm_repo}","revision":"{dsm_rev}","roles":[{dsm1},{dsm2}]}},"moshi":{{"repository":"{moshi_repo}","revision":"{moshi_rev}","roles":[{m1},{m2},{m3},{m4},{m5}]}}}},"source_roles":{{"dsm":["configs/config-stt-en-hf.toml","scripts/stt_from_file_pytorch.py"],"moshi":["moshi/moshi/models/lm.py","moshi/moshi/models/lm_utils.py","moshi/moshi/models/loaders.py","moshi/moshi/utils/sampling.py","moshi/moshi/modules/transformer.py"]}},"contracts":[{{"name":"a","status":"SOURCE_EXPRESSION_AUTHENTICATED","roles":["r"],"expressions":["e"]}},{{"name":"temperature_zero_selection","status":"SOURCE_EXPRESSION_AUTHENTICATED","roles":["r"],"expressions":["e"],"tie_resolution":"BLOCKED_EXTERNAL_TORCH_SEMANTICS"}},{{"name":"c","status":"SOURCE_EXPRESSION_AUTHENTICATED","roles":["r"],"expressions":["e"]}}],"blockers":["a","b","c"],"claim_boundary":"source expressions only; no model execution, numerical parity, PCM transcription, or framework tie-rule claim"}}"#,
        schema = SOURCE_CONTRACT_SCHEMA,
        dsm_repo = DSM_REPOSITORY,
        dsm_rev = DSM_REVISION,
        moshi_repo = MOSHI_REPOSITORY,
        moshi_rev = MOSHI_REVISION,
        dsm1 = role("configs/config-stt-en-hf.toml"),
        dsm2 = role("scripts/stt_from_file_pytorch.py"),
        m1 = role("moshi/moshi/models/lm.py"),
        m2 = role("moshi/moshi/models/lm_utils.py"),
        m3 = role("moshi/moshi/models/loaders.py"),
        m4 = role("moshi/moshi/utils/sampling.py"),
        m5 = role("moshi/moshi/modules/transformer.py"),
    );
    validate_source_contract(&json::parse(contract.as_bytes()).unwrap()).unwrap();
    let bad_contract = contract.replace("BLOCKED_NOT_EXECUTED", "EXECUTED");
    assert!(validate_source_contract(&json::parse(bad_contract.as_bytes()).unwrap()).is_err());
}

#[test]
fn primary_source_contract_fixture_preserves_helper_shape_and_missing_fact() {
    let fixture = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tools/parity/kyutai_stt_streaming_reference/fixtures/source_contract_primary_packet.json"
    ));
    let value = json::parse(fixture.as_bytes()).expect("source-contract fixture must be JSON");
    assert_eq!(
        value.get("schema").and_then(JsonValue::as_str),
        Some(SOURCE_CONTRACT_SCHEMA)
    );
    assert_eq!(
        value.get("status").and_then(JsonValue::as_str),
        Some(SOURCE_CONTRACT_STATUS)
    );
    assert_eq!(
        value.get("runtime_status").and_then(JsonValue::as_str),
        Some(SOURCE_CONTRACT_RUNTIME_STATUS)
    );
    assert_eq!(
        value
            .get("contracts")
            .and_then(JsonValue::as_array)
            .map(Vec::len),
        Some(3)
    );
    assert_eq!(
        value
            .get("source_identity")
            .and_then(|identity| identity.get("dsm"))
            .and_then(|dsm| dsm.get("roles"))
            .and_then(JsonValue::as_array)
            .and_then(|roles| roles.first())
            .and_then(|role| role.get("sha256"))
            .and_then(JsonValue::as_str),
        Some("81f77d642689e1acb276089f62064dab2e71a5532fcac2d3c12563cf4946552c")
    );
    validate_source_contract(&value).expect("reviewed source metadata must validate");
    let missing_config_sha = fixture.replace(
        "81f77d642689e1acb276089f62064dab2e71a5532fcac2d3c12563cf4946552c",
        "null",
    );
    assert!(
        validate_source_contract(&json::parse(missing_config_sha.as_bytes()).unwrap()).is_err(),
        "missing config hash must remain fail-closed"
    );
}

#[test]
fn closure_gate_rejects_before_component_access() {
    let packet = ComponentPacket {
        expected_head: "b".repeat(40),
        reference_manifest_sha256: "a".repeat(64),
        approval_sha256: "a".repeat(64),
        dependency_closure_sha256: "a".repeat(64),
        pcm: PcmBinding {
            path: PathBuf::from("/definitely/missing/pcm"),
            bytes: 4,
            sha256: "a".repeat(64),
            sample_rate: 24_000,
            channels: 1,
            dtype: EXPECTED_DTYPE.to_owned(),
        },
        components: Vec::new(),
    };
    let error = validate_gate(
        &packet,
        ExistingBindings {
            expected_head: &packet.expected_head,
            decoder_sha256: &"a".repeat(64),
            reference_manifest_sha256: &packet.reference_manifest_sha256,
            approval_sha256: &packet.approval_sha256,
            dependency_closure_sha256: &packet.dependency_closure_sha256,
        },
    )
    .unwrap_err();
    assert!(format!("{error:?}").contains("BLOCKED_DEPENDENCY_CLOSURE"));
}

#[test]
fn packet_rejects_uppercase_hash_and_duplicate_keys() {
    assert!(
        parse_packet(br#"{"format":"vokra-kyutai-stt-pcm-components-v1","format":"x"}"#).is_err()
    );
    let mut digest = "a".repeat(64);
    digest.replace_range(0..1, "A");
    assert!(lower_hex(&digest, "test").is_err());
}

#[test]
fn git_blob_sha1_is_exactly_lowercase_40_hex() {
    assert!(lower_git_blob_sha1(&"b".repeat(40), "blob").is_ok());
    assert!(lower_git_blob_sha1(&"a".repeat(64), "blob").is_err());
    assert!(lower_git_blob_sha1(&"B".repeat(40), "blob").is_err());
}

#[test]
fn packet_rejects_relative_pcm_and_aggregate_budget() {
    let digest = "a".repeat(64);
    let components = ["decoder", "tokenizer", "mimi_gguf", "raw_mimi"]
        .into_iter()
        .map(|role| {
            format!(
                "{{\"role\":\"{role}\",\"path\":\"/tmp/{role}\",\"bytes\":4000000000,\"sha256\":\"{digest}\",\"source_sha256\":\"{digest}\"}}"
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let packet = format!(
        "{{\"format\":\"{PACKET_SCHEMA}\",\"expected_head\":\"{}\",\"reference_manifest_sha256\":\"{digest}\",\"approval_sha256\":\"{digest}\",\"dependency_closure_sha256\":\"{digest}\",\"pcm\":{{\"path\":\"relative.f32le\",\"bytes\":4,\"sha256\":\"{digest}\",\"sample_rate\":24000,\"channels\":1,\"dtype\":\"float32-le\"}},\"components\":[{components}]}}",
        "b".repeat(40),
    );
    assert!(parse_packet(packet.as_bytes()).is_err());
    let valid_pcm_packet = packet.replace("relative.f32le", "/tmp/input.f32le");
    assert!(parse_packet(valid_pcm_packet.as_bytes()).is_err());
}

#[test]
fn file_binding_detects_hash_and_byte_changes() {
    let path = std::env::temp_dir().join(format!(
        "vokra-kyutai-pcm-consumer-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    assert!(!path.exists());
    let body = b"bounded-pcm-component";
    std::fs::write(&path, body).unwrap();
    let digest = hex_digest(&sha256_bytes(body));
    verify_file_binding(&path, body.len() as u64, &digest, "synthetic").unwrap();
    assert!(verify_file_binding(&path, body.len() as u64 + 1, &digest, "synthetic").is_err());
    assert!(verify_file_binding(&path, body.len() as u64, &"0".repeat(64), "synthetic").is_err());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn native_frame_cap_is_derived_from_pcm_geometry() {
    let contract = super::super::KyutaiSttStreamingContract::from_config(
        &super::super::KyutaiSttConfig::stt_2_6b_en(),
    )
    .unwrap();
    let one_frame = contract.padded_frame_count(1_920).unwrap();
    assert!(one_frame > 0 && one_frame < MAX_FRAMES);
    assert_ne!(one_frame, 1_000);
}

#[test]
fn kv_checkpoint_positions_cover_eviction_without_accepting_future_rows() {
    assert_eq!(expected_kv_positions(0), (0..=0).collect());
    assert_eq!(expected_kv_positions(374), (0..=374).collect());
    assert_eq!(expected_kv_positions(375), (1..=375).collect());
    assert_eq!(expected_kv_positions(376), (2..=376).collect());
    assert!(!expected_kv_positions(375).contains(&0));
    let checkpoints = expected_checkpoint_coordinates();
    assert_eq!(checkpoints.len(), 288);
    assert!(checkpoints.contains(&("warmup".to_owned(), 375, 47)));
    assert!(checkpoints.contains(&("warmup".to_owned(), 376, 0)));
    assert!(checkpoints.contains(&("after_reset".to_owned(), 0, 47)));
    let mut valid = vec![-1; 375];
    valid.copy_from_slice(&(1..=375).collect::<Vec<_>>());
    validate_kv_positions(375, &valid).unwrap();
    valid[0] = 0;
    assert!(validate_kv_positions(375, &valid).is_err());
    valid.copy_from_slice(&(2..=376).collect::<Vec<_>>());
    validate_kv_positions(376, &valid).unwrap();
    valid[374] = 377;
    assert!(validate_kv_positions(376, &valid).is_err());
}

#[test]
fn causal_alignment_does_not_shift_first_double_or_reset_tail() {
    assert_eq!(aligned_reference_frame(false, 0, 0, 0), Some(0));
    assert_eq!(aligned_reference_frame(false, 0, 1, 1), None);
    assert_eq!(aligned_reference_frame(false, 0, 0, 1), None);
    assert_eq!(aligned_reference_frame(true, 0, 0, 0), Some(377));
    assert_eq!(aligned_reference_frame(true, 1, 0, 1), None);
    assert_eq!(aligned_reference_frame(true, 0, 1, 0), None);
    assert_eq!(code_reference_frame(false, 376), Some(376));
    assert_eq!(code_reference_frame(false, 377), None);
    assert_eq!(code_reference_frame(true, 0), Some(377));
    assert_eq!(code_reference_frame(true, 1), None);
}

#[test]
fn producer_kv_parser_covers_all_events_and_rejects_missing_checkpoint() {
    fn fixture(omit_first_snapshot: bool) -> (JsonValue, JsonValue) {
        let digest = "a".repeat(64);
        let mut events = Vec::new();
        let mut artifacts = BTreeMap::new();
        for (phase, steps) in [("warmup", 377usize), ("after_reset", 1usize)] {
            for layer in 0..48 {
                let delta_bytes = if phase == "warmup" {
                    377 * 32 * 64 * 2
                } else {
                    32 * 64 * 2
                };
                for name in [
                    format!("kv/{phase}-layer-{layer:02}-delta-keys.bin"),
                    format!("kv/{phase}-layer-{layer:02}-delta-values.bin"),
                ] {
                    artifacts.insert(name, delta_bytes);
                }
            }
            for step in 0..steps {
                let is_checkpoint =
                    phase == "after_reset" || matches!(step, 0 | 1 | 374 | 375 | 376);
                for layer in 0..48 {
                    let delta_keys = format!("kv/{phase}-layer-{layer:02}-delta-keys.bin");
                    let delta_values = format!("kv/{phase}-layer-{layer:02}-delta-values.bin");
                    let mut event = format!(
                        "{{\"phase\":\"{phase}\",\"step\":{step},\"layer\":{layer},\"lm_call_ordinal\":0,\"delta_shape\":[1,32,1,64],\"delta_dtype\":\"torch.bfloat16\",\"capacity\":375,\"delta_keys\":\"{delta_keys}\",\"delta_keys_offset\":{},\"delta_keys_bytes\":4096,\"delta_values\":\"{delta_values}\",\"delta_values_offset\":{},\"delta_values_bytes\":4096,\"end_offset\":[{}]",
                        step * 4096,
                        step * 4096,
                        step + 1,
                    );
                    if is_checkpoint
                        && !(omit_first_snapshot && phase == "warmup" && step == 0 && layer == 0)
                    {
                        let expected_start = (step + 1).saturating_sub(375);
                        let valid_count = step + 1 - expected_start;
                        let positions: Vec<String> = (0..375)
                            .map(|index| {
                                if index < 375 - valid_count {
                                    "-1".to_owned()
                                } else {
                                    (expected_start + index - (375 - valid_count)).to_string()
                                }
                            })
                            .collect();
                        let keys = format!("kv/{phase}-step-{step:04}-layer-{layer:02}-keys.bin");
                        let values =
                            format!("kv/{phase}-step-{step:04}-layer-{layer:02}-values.bin");
                        artifacts.insert(keys.clone(), 32 * 375 * 64 * 2);
                        artifacts.insert(values.clone(), 32 * 375 * 64 * 2);
                        event.push_str(&format!(
                            ",\"shape\":[1,32,375,64],\"dtype\":\"torch.bfloat16\",\"positions\":[{}],\"keys\":\"{keys}\",\"values\":\"{values}\"",
                            positions.join(",")
                        ));
                    }
                    event.push('}');
                    events.push(event);
                }
            }
        }
        let artifact_json = artifacts
            .into_iter()
            .map(|(name, bytes)| {
                format!("\"{name}\":{{\"bytes\":{bytes},\"sha256\":\"{digest}\"}}")
            })
            .collect::<Vec<_>>()
            .join(",");
        (
            json::parse(
                format!(
                    "{{\"ring_cache\":{{\"checkpoint_file_pattern\":\"kv/{{phase}}-step-{{step:04d}}-layer-{{layer:02d}}-{{keys|values}}.bin\",\"checkpoints\":[0,1,374,375,376],\"events\":[{}]}}}}",
                    events.join(",")
                )
                .as_bytes(),
            )
            .unwrap(),
            json::parse(format!("{{{artifact_json}}}").as_bytes()).unwrap(),
        )
    }

    let (manifest, artifacts) = fixture(false);
    assert_eq!(parse_kv_events(&manifest, &artifacts).unwrap().len(), 288);
    let (manifest, artifacts) = fixture(true);
    assert!(parse_kv_events(&manifest, &artifacts).is_err());
}

#[test]
fn reference_tree_counts_empty_directories_and_enforces_entry_bound() {
    let root = std::env::temp_dir().join(format!(
        "vokra-kyutai-reference-tree-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    for index in 0..=REFERENCE_TREE_ENTRY_MAX {
        std::fs::create_dir(root.join(format!("empty-{index:04}"))).unwrap();
    }
    let mut files = BTreeSet::new();
    let mut entries = 0;
    assert!(collect_reference_files(&root, &root, &mut files, &mut entries, 0).is_err());
    for index in (0..=REFERENCE_TREE_ENTRY_MAX).rev() {
        std::fs::remove_dir(root.join(format!("empty-{index:04}"))).unwrap();
    }
    std::fs::remove_dir(root).unwrap();
}

#[test]
#[ignore = "requires authenticated four-component packet, PCM, and approved real weights on VAST"]
fn authenticated_pcm_consumer_is_external_only() {
    let expected_head =
        std::env::var(EXPECTED_HEAD_ENV).expect("real PCM test requires expected head");
    verify_host_and_head(&expected_head).expect("authenticated host and HEAD");
    let (packet, reference) =
        load_packet_from_env(&expected_head).expect("authenticated component packet");
    let diagnostic =
        authenticate_and_consume_pcm(&packet, &reference).expect("native PCM consumer");
    assert_eq!(diagnostic.status, "NOT_PARITY_PASS");
    assert!(diagnostic.frames > 0 && diagnostic.calls > 0);
    eprintln!(
        "KYUTAI_STT_PCM status={} frames={} calls={} common_codes={} unmatched_frames={} code_mismatches={} unmatched_calls={} logits={}/{} tokens={}/{} kv={} kv_mismatches={} logit_max={} logit_mean={} logit_worst={:?} kv_max={} kv_mean={} kv_worst={:?}",
        diagnostic.status,
        diagnostic.frames,
        diagnostic.calls,
        diagnostic.common_code_frames,
        diagnostic.unmatched_frames,
        diagnostic.code_mismatches,
        diagnostic.unmatched_calls,
        diagnostic.logit_matches,
        diagnostic.logit_mismatches,
        diagnostic.token_matches,
        diagnostic.token_mismatches,
        diagnostic.kv_comparisons,
        diagnostic.kv_mismatches,
        diagnostic.logit_max_abs,
        if diagnostic.logit_values == 0 {
            0.0
        } else {
            diagnostic.logit_abs_sum / diagnostic.logit_values as f64
        },
        diagnostic.logit_worst,
        diagnostic.kv_max_abs,
        if diagnostic.kv_values == 0 {
            0.0
        } else {
            diagnostic.kv_abs_sum / diagnostic.kv_values as f64
        },
        diagnostic.kv_worst,
    );
    panic!("NOT_PARITY_PASS: no approved numerical bound exists for native PCM comparison");
}
