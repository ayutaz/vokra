//! VAST-only independent Kyutai STT streaming-LM consumer.
//!
//! This test consumes the packet emitted by the pinned official Moshi
//! `LMGen`/`RingKVCache.complete` hook.  It authenticates every input and
//! artifact before opening the GGUF, compares official BF16 cache bytes after
//! physical-ring-to-chronological reordering with the native borrowed cache
//! view, and compares greedy text ids.  No fixture or missing input is a
//! successful result.  Numerical collection is deliberately reported as
//! `NOT_PARITY_PASS` until a reviewed bound is approved.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::json::JsonValue;
use vokra_models::kyutai_stt::{KyutaiSttAsr, KyutaiSttWeights};

const GGUF_ENV: &str = "VOKRA_KYUTAI_STT_STREAMING_GGUF";
const GGUF_SHA_ENV: &str = "VOKRA_KYUTAI_STT_STREAMING_GGUF_SHA256";
const REFERENCE_ENV: &str = "VOKRA_KYUTAI_STT_STREAMING_REFERENCE";
const REFERENCE_SHA_ENV: &str = "VOKRA_KYUTAI_STT_STREAMING_REFERENCE_MANIFEST_SHA256";
const APPROVAL_ENV: &str = "VOKRA_KYUTAI_STT_STREAMING_APPROVAL";
const APPROVAL_SHA_ENV: &str = "VOKRA_KYUTAI_STT_STREAMING_APPROVAL_SHA256";
const DEPENDENCY_CLOSURE_SHA_ENV: &str = "VOKRA_KYUTAI_STT_STREAMING_DEPENDENCY_CLOSURE_SHA256";
const COMPOSITE_APPROVAL_SCHEMA: &str = "vokra-kyutai-stt-pytorch-pcm-oracle-approval-v1";
const COMPOSITE_APPROVAL_SCOPE: &str = "KYUTAI_STT_PYTORCH_PCM_ORACLE_CAPTURE";
const EXECUTION_APPROVAL_MAX_BYTES: u64 = 64 * 1024;
// Synchronized with contract.py: no dependency closure is reviewed yet.
const REVIEWED_DEPENDENCY_CLOSURES: &[&str] = &[];
const HF_REVISION: &str = "a07aec56d22be5589cd0bc8709c75b6cf3e3039d";
const MOSHI_REPOSITORY: &str = "https://github.com/kyutai-labs/moshi.git";
const MOSHI_REVISION: &str = "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362";
const DSM_REPOSITORY: &str = "https://github.com/kyutai-labs/delayed-streams-modeling.git";
const DSM_REVISION: &str = "4c4f65e147df056adf3346290d64c7b9649b18c9";
const SOURCE_PACKET_MANIFEST_SHA256: &str =
    "85223a7ac8b947eaeafa7b2f337a1ac60dea84a75d44ee0c607df72ad25d6342";
const STREAMING_SCHEMA: &str = "vokra-kyutai-stt-independent-streaming-reference-v1";
const STREAMING_SCOPE: &str = "official Moshi LMGen main-transformer KV cache only; no Mimi PCM, depformer, tokenizer, transcription, publication, or Apple parity claim";
const MODEL_BYTES: u64 = 5_234_275_128;
const MODEL_SHA256: &str = "2471add7da1fdb2d5dc4561e88a9069376333d992760d55d29d1db46c52849b2";
const MODEL_TENSOR_MANIFEST_SHA256: &str =
    "e62488c9d16953010c758ec17f4c70e8ee30d348adfab3811eb5dfecb435d5df";
const CONFIG_SHA256: &str = "b79ea52a30329887a2d0ce2dd5473a63fc5083e441e7986f64f01050c06239c9";
const MIMI_NAME: &str = "mimi-pytorch-e351c8d8@125.safetensors";
const MIMI_BYTES: u64 = 384_644_900;
const MIMI_SHA256: &str = "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50";
const TOKENIZER_NAME: &str = "tokenizer_en_audio_4000.model";
const TOKENIZER_BYTES: u64 = 59_339;
const TOKENIZER_SHA256: &str = "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6";
const CONFIG_BYTES: u64 = 1_257;
const CONFIG_NAME: &str = "config.json";
const N_Q: usize = 32;
const TEXT_CARD: usize = 4_000;
const AUDIO_CARD: usize = 2_048;
const N_LAYERS: usize = 48;
const D_MODEL: usize = 2_048;
const N_HEADS: usize = 32;
const HEAD_DIM: usize = 64;
const CONTEXT: usize = 375;
const FRAMES: usize = CONTEXT + 2;
const CHECKPOINT_STEPS: [usize; 5] = [0, 1, 374, 375, 376];
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ARTIFACT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TOTAL_ARTIFACT_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_ARTIFACT_ENTRIES: usize = 4_096;
const MAX_ARTIFACT_DEPTH: usize = 32;
// No reviewed numerical bound exists for BF16 official weights vs native f32.
const FIXED_ATOL: Option<f32> = None;

fn exact_keys(value: &JsonValue, expected: &[&str], label: &str) {
    let object = value
        .as_object()
        .unwrap_or_else(|| panic!("{label} is not an object"));
    assert_eq!(object.len(), expected.len(), "{label} key count");
    for key in expected {
        assert!(
            object.iter().any(|(actual, _)| actual == key),
            "{label} key {key}"
        );
    }
}

fn string<'a>(value: &'a JsonValue, key: &str) -> &'a str {
    value
        .get(key)
        .and_then(JsonValue::as_str)
        .unwrap_or_else(|| panic!("missing string {key}"))
}

fn integer(value: &JsonValue, key: &str) -> u64 {
    value
        .get(key)
        .and_then(JsonValue::as_u64)
        .unwrap_or_else(|| panic!("missing integer {key}"))
}

fn signed(value: &JsonValue, label: &str) -> i64 {
    match value {
        JsonValue::Int(value) => *value,
        _ => panic!("{label} must be an integer"),
    }
}

fn object<'a>(value: &'a JsonValue, key: &str) -> &'a JsonValue {
    value
        .get(key)
        .and_then(|v| v.as_object().map(|_| v))
        .unwrap_or_else(|| panic!("missing object {key}"))
}

fn lower_hex(value: &str, digits: usize, label: &str) {
    assert_eq!(value.len(), digits, "{label} digest length");
    assert!(
        value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "{label} digest casing"
    );
}

fn reject_duplicate_keys(bytes: &[u8], label: &str) {
    let value = vokra_core::json::parse(bytes)
        .unwrap_or_else(|error| panic!("{label} malformed JSON: {error}"));
    fn visit(value: &JsonValue, label: &str) {
        match value {
            JsonValue::Object(entries) => {
                let mut keys = HashSet::new();
                for (key, child) in entries {
                    assert!(keys.insert(key), "{label} duplicate decoded key: {key}");
                    visit(child, label);
                }
            }
            JsonValue::Array(items) => {
                for item in items {
                    visit(item, label);
                }
            }
            JsonValue::Null
            | JsonValue::Bool(_)
            | JsonValue::Int(_)
            | JsonValue::Float(_)
            | JsonValue::Str(_) => {}
        }
    }
    visit(&value, label);
}

fn reject_raw_path(value: &str, label: &str) {
    assert!(value.starts_with('/'), "{label} must be absolute");
    for part in value.split('/') {
        assert!(
            part != "." && part != "..",
            "{label} contains dot component"
        );
    }
}

fn required_path(name: &str) -> PathBuf {
    let value = std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"));
    reject_raw_path(&value, name);
    PathBuf::from(value)
}

fn required_file(path: &Path, label: &str) {
    assert!(path.is_absolute(), "{label} must be absolute");
    let metadata = fs::symlink_metadata(path).unwrap_or_else(|error| panic!("{label}: {error}"));
    assert!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "{label} must be a regular file"
    );
    for component in path.components() {
        assert!(!matches!(
            component,
            Component::CurDir | Component::ParentDir
        ));
    }
    for ancestor in path.ancestors() {
        let metadata = fs::symlink_metadata(ancestor).expect("path ancestry metadata");
        assert!(
            !metadata.file_type().is_symlink(),
            "{label} has symlink ancestry"
        );
    }
}

fn required_directory(path: &Path, label: &str) {
    let metadata = fs::symlink_metadata(path).unwrap_or_else(|error| panic!("{label}: {error}"));
    assert!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "{label} must be a real directory"
    );
    for ancestor in path.ancestors() {
        let metadata = fs::symlink_metadata(ancestor).expect("directory ancestry metadata");
        assert!(
            !metadata.file_type().is_symlink(),
            "{label} has symlink ancestry"
        );
    }
}

fn require_disjoint(paths: &[(&Path, &str)]) {
    let canonical = paths
        .iter()
        .map(|(path, label)| {
            (
                path.canonicalize()
                    .unwrap_or_else(|_| panic!("{label} canonical path")),
                *label,
            )
        })
        .collect::<Vec<_>>();
    for left in 0..canonical.len() {
        for right in (left + 1)..canonical.len() {
            assert!(
                !canonical[left].0.starts_with(&canonical[right].0)
                    && !canonical[right].0.starts_with(&canonical[left].0),
                "{} and {} overlap",
                canonical[left].1,
                canonical[right].1
            );
        }
    }
}

fn file_digest(path: &Path) -> String {
    for program in ["shasum", "sha256sum"] {
        let output = if program == "shasum" {
            Command::new(program)
                .arg("-a")
                .arg("256")
                .arg(path)
                .output()
        } else {
            Command::new(program).arg(path).output()
        };
        if let Ok(output) = output {
            if output.status.success() {
                if let Some(value) = String::from_utf8_lossy(&output.stdout)
                    .split_whitespace()
                    .next()
                {
                    if value.len() == 64
                        && value
                            .bytes()
                            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
                    {
                        return value.to_owned();
                    }
                }
            }
        }
    }
    panic!("a streaming shasum/sha256sum implementation is required")
}

fn bytes_digest_with_programs(body: &[u8], programs: &[&str]) -> Option<String> {
    for program in programs {
        let mut command = Command::new(*program);
        if *program == "shasum" {
            command.args(["-a", "256"]);
        }
        let Ok(mut child) = command.stdin(Stdio::piped()).stdout(Stdio::piped()).spawn() else {
            continue;
        };
        let Some(mut stdin) = child.stdin.take() else {
            let _ = child.kill();
            let _ = child.wait();
            continue;
        };
        if stdin.write_all(body).is_err() {
            drop(stdin);
            let _ = child.kill();
            let _ = child.wait();
            continue;
        }
        drop(stdin);
        let Ok(output) = child.wait_with_output() else {
            continue;
        };
        if output.status.success() {
            if let Some(value) = String::from_utf8_lossy(&output.stdout)
                .split_whitespace()
                .next()
            {
                if value.len() == 64
                    && value
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
                {
                    return Some(value.to_owned());
                }
            }
        }
    }
    None
}

fn bytes_digest(body: &[u8]) -> String {
    bytes_digest_with_programs(body, &["shasum", "sha256sum"])
        .unwrap_or_else(|| panic!("a streaming shasum/sha256sum implementation is required"))
}

#[cfg(unix)]
fn file_identity(metadata: &fs::Metadata) -> Result<(u64, u64), String> {
    use std::os::unix::fs::MetadataExt;
    Ok((metadata.dev(), metadata.ino()))
}

#[cfg(not(unix))]
fn file_identity(metadata: &fs::Metadata) -> Result<(u64, u64), String> {
    let modified = metadata
        .modified()
        .map_err(|error| format!("file modification time unavailable: {error}"))?;
    let nanos = modified
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| format!("file modification time predates UNIX epoch: {error}"))?
        .as_nanos();
    let nanos = u64::try_from(nanos)
        .map_err(|_| "file modification time does not fit in the identity bound".to_owned())?;
    // Windows has no stable std-only inode API in the supported MSRV.  The
    // epoch timestamp is stable for the read and failures remain fail-closed.
    Ok((metadata.len(), nanos))
}

fn read_bounded_file(path: &Path, label: &str, max_bytes: u64) -> Vec<u8> {
    required_file(path, label);
    let before = fs::metadata(path).unwrap_or_else(|error| panic!("{label}: {error}"));
    assert!(before.is_file(), "{label} must be a regular file");
    assert!(before.len() <= max_bytes, "{label} exceeds byte bound");
    let identity = file_identity(&before).unwrap_or_else(|error| panic!("{label}: {error}"));
    let mut file = fs::File::open(path).unwrap_or_else(|error| panic!("{label}: {error}"));
    let opened = file
        .metadata()
        .unwrap_or_else(|error| panic!("{label}: {error}"));
    assert_eq!(opened.len(), before.len(), "{label} changed while opening");
    assert_eq!(
        file_identity(&opened).unwrap_or_else(|error| panic!("{label}: {error}")),
        identity,
        "{label} replaced while opening"
    );
    let mut body = Vec::with_capacity(before.len() as usize);
    let mut limited = std::io::Read::by_ref(&mut file).take(max_bytes.saturating_add(1));
    limited
        .read_to_end(&mut body)
        .unwrap_or_else(|error| panic!("{label}: {error}"));
    assert!(
        body.len() as u64 <= max_bytes,
        "{label} grew beyond byte bound"
    );
    let after_file = file
        .metadata()
        .unwrap_or_else(|error| panic!("{label}: {error}"));
    let after_path = fs::metadata(path).unwrap_or_else(|error| panic!("{label}: {error}"));
    required_file(path, label);
    assert_eq!(
        after_file.len(),
        before.len(),
        "{label} changed while reading"
    );
    assert_eq!(
        after_path.len(),
        before.len(),
        "{label} changed while reading"
    );
    assert_eq!(
        file_identity(&after_file).unwrap_or_else(|error| panic!("{label}: {error}")),
        identity,
        "{label} replaced while reading"
    );
    assert_eq!(
        file_identity(&after_path).unwrap_or_else(|error| panic!("{label}: {error}")),
        identity,
        "{label} replaced while reading"
    );
    assert_eq!(
        body.len() as u64,
        before.len(),
        "{label} changed while reading"
    );
    body
}

fn parse_json(path: &Path, label: &str, max_bytes: u64) -> (Vec<u8>, JsonValue) {
    let bytes = read_bounded_file(path, label, max_bytes);
    reject_duplicate_keys(&bytes, label);
    let value = vokra_core::json::parse(&bytes).unwrap_or_else(|error| panic!("{label}: {error}"));
    (bytes, value)
}

fn require_execution_readiness(approval: &Path, manifest: &JsonValue, checkout: &Path) {
    let expected_head = string(manifest, "expected_head");
    let expected_approval_sha = string(manifest, "approval_sha256");
    let approval_sha = std::env::var(APPROVAL_SHA_ENV).expect("composite approval SHA");
    lower_hex(&approval_sha, 64, "composite approval SHA");
    assert_eq!(
        approval_sha, expected_approval_sha,
        "approval SHA does not match manifest"
    );
    required_file(approval, "composite approval");
    require_disjoint(&[(approval, "composite approval"), (checkout, "checkout")]);
    let (approval_bytes, document) =
        parse_json(approval, "composite approval", EXECUTION_APPROVAL_MAX_BYTES);
    assert_eq!(
        bytes_digest(&approval_bytes),
        approval_sha,
        "composite approval SHA mismatch"
    );
    assert_eq!(
        approval_bytes.len() as u64,
        fs::metadata(approval).expect("approval metadata").len(),
        "composite approval changed while reading"
    );
    exact_keys(
        &document,
        &[
            "schema",
            "scope",
            "decision",
            "execution",
            "head",
            "checkout",
        ],
        "composite approval",
    );
    assert_eq!(string(&document, "schema"), COMPOSITE_APPROVAL_SCHEMA);
    assert_eq!(string(&document, "scope"), COMPOSITE_APPROVAL_SCOPE);
    assert_eq!(string(&document, "decision"), "APPROVED");
    assert_eq!(string(&document, "execution"), "VAST_ONLY");
    assert_eq!(string(&document, "head"), expected_head);
    assert_eq!(
        string(&document, "checkout"),
        checkout.to_string_lossy().as_ref()
    );
    let closure = std::env::var(DEPENDENCY_CLOSURE_SHA_ENV).expect("dependency closure SHA");
    lower_hex(&closure, 64, "dependency closure SHA");
    assert!(
        reviewed_dependency_closure(&closure),
        "{}: no reviewed dependency closure",
        "BLOCKED_DEPENDENCY_CLOSURE"
    );
}

fn reviewed_dependency_closure(value: &str) -> bool {
    REVIEWED_DEPENDENCY_CLOSURES.contains(&value)
}

#[derive(Clone)]
struct ArtifactRow {
    bytes: u64,
    sha256: String,
}

struct AuthenticatedReference {
    root: PathBuf,
    manifest: JsonValue,
    artifacts: BTreeMap<String, ArtifactRow>,
}

fn safe_artifact_name(name: &str) {
    let path = Path::new(name);
    assert!(
        !path.is_absolute() && !name.is_empty(),
        "artifact path must be relative"
    );
    for component in path.components() {
        assert!(
            matches!(component, Component::Normal(_)),
            "unsafe artifact path {name}"
        );
    }
}

fn collect_artifact_files(root: &Path, current: &Path, output: &mut BTreeSet<String>) {
    let mut entries = 0;
    collect_artifact_files_inner(root, current, output, 0, &mut entries);
}

fn collect_artifact_files_inner(
    root: &Path,
    current: &Path,
    output: &mut BTreeSet<String>,
    depth: usize,
    entries: &mut usize,
) {
    assert!(
        depth <= MAX_ARTIFACT_DEPTH,
        "reference directory depth exceeds bound"
    );
    let directory_entries = fs::read_dir(current).expect("reference directory entries");
    for entry in directory_entries {
        *entries = entries
            .checked_add(1)
            .expect("reference entry count overflow");
        assert!(
            *entries <= MAX_ARTIFACT_ENTRIES,
            "reference entry count exceeds bound"
        );
        let entry = entry.expect("reference entry");
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path).expect("reference entry metadata");
        assert!(
            !metadata.file_type().is_symlink(),
            "reference symlink is forbidden"
        );
        if metadata.is_dir() {
            collect_artifact_files_inner(root, &path, output, depth + 1, entries);
        } else {
            assert!(metadata.is_file(), "reference contains a non-regular entry");
            output.insert(
                path.strip_prefix(root)
                    .expect("artifact relative path")
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
}

fn authenticate_reference(reference: &Path, manifest: JsonValue) -> AuthenticatedReference {
    required_directory(reference, "streaming reference");
    exact_keys(
        &manifest,
        &[
            "format",
            "status",
            "scope",
            "claim_boundary",
            "expected_head",
            "dependency_closure_sha256",
            "approval_sha256",
            "approval_decision",
            "approval_scope",
            "source_packet",
            "model",
            "sources",
            "input",
            "execution",
            "ring_cache",
            "scheduler",
            "text",
            "artifacts",
        ],
        "streaming manifest",
    );
    assert_eq!(string(&manifest, "format"), STREAMING_SCHEMA);
    assert_eq!(string(&manifest, "status"), "REFERENCE_READY");
    assert_eq!(string(&manifest, "scope"), STREAMING_SCOPE);
    assert!(string(&manifest, "claim_boundary").contains("not PCM/ASR parity"));
    lower_hex(
        string(&manifest, "expected_head"),
        40,
        "expected source head",
    );
    let closure_sha = string(&manifest, "dependency_closure_sha256");
    lower_hex(closure_sha, 64, "dependency closure SHA");
    let expected_closure_sha =
        std::env::var(DEPENDENCY_CLOSURE_SHA_ENV).expect("dependency closure SHA");
    assert_eq!(
        closure_sha, expected_closure_sha,
        "manifest dependency closure SHA mismatch"
    );
    lower_hex(string(&manifest, "approval_sha256"), 64, "approval SHA");
    assert_eq!(string(&manifest, "approval_decision"), "APPROVED");
    assert_eq!(
        string(&manifest, "approval_scope"),
        COMPOSITE_APPROVAL_SCOPE
    );

    let model = object(&manifest, "model");
    exact_keys(
        model,
        &["status", "files", "tensor_manifest_sha256"],
        "model record",
    );
    assert_eq!(
        string(model, "status"),
        "AUTHENTICATED_FOUR_FILE_COMPOSITE_INPUT"
    );
    assert_eq!(
        string(model, "tensor_manifest_sha256"),
        MODEL_TENSOR_MANIFEST_SHA256
    );
    let model_files = object(model, "files");
    exact_keys(
        model_files,
        &[MODEL_NAME, MIMI_NAME, TOKENIZER_NAME, CONFIG_NAME],
        "model files",
    );
    for (name, bytes, sha) in [
        (MODEL_NAME, MODEL_BYTES, MODEL_SHA256),
        (MIMI_NAME, MIMI_BYTES, MIMI_SHA256),
        (TOKENIZER_NAME, TOKENIZER_BYTES, TOKENIZER_SHA256),
        (CONFIG_NAME, CONFIG_BYTES, CONFIG_SHA256),
    ] {
        let row = object(model_files, name);
        exact_keys(row, &["bytes", "sha256"], "model file record");
        assert_eq!(integer(row, "bytes"), bytes);
        assert_eq!(string(row, "sha256"), sha);
    }

    let source_packet = object(&manifest, "source_packet");
    exact_keys(
        source_packet,
        &[
            "path",
            "manifest_sha256",
            "moshi_lm_sha256",
            "moshi_transformer_sha256",
            "status",
        ],
        "source packet",
    );
    assert_eq!(string(source_packet, "status"), "SOURCE_ONLY_PREPARATION");
    assert_eq!(
        string(source_packet, "manifest_sha256"),
        SOURCE_PACKET_MANIFEST_SHA256
    );
    assert_eq!(
        string(source_packet, "moshi_lm_sha256"),
        "38991e83d7e3aa0ff1483b27a0b59e1ff43642ad6a32dc8dc4272ad2b6c0dc5f"
    );
    assert_eq!(
        string(source_packet, "moshi_transformer_sha256"),
        "f5a73d752a5bde1eda2b0b14bebd13fd81db81017d361782a07164580a687622"
    );
    assert!(!string(source_packet, "path").is_empty());

    let sources = object(&manifest, "sources");
    exact_keys(sources, &["dsm", "moshi", "contract"], "sources");
    for (key, repository, revision) in [
        ("dsm", DSM_REPOSITORY, DSM_REVISION),
        ("moshi", MOSHI_REPOSITORY, MOSHI_REVISION),
    ] {
        let source = object(sources, key);
        exact_keys(
            source,
            &["repository", "revision", "origin"],
            "source identity",
        );
        assert_eq!(string(source, "repository"), repository);
        assert_eq!(string(source, "revision"), revision);
        assert!(!string(source, "origin").is_empty());
    }
    let source_contract = object(sources, "contract");
    exact_keys(
        source_contract,
        &[
            "path",
            "manifest_sha256",
            "moshi_lm_sha256",
            "moshi_transformer_sha256",
            "status",
        ],
        "source contract",
    );
    assert_eq!(
        string(source_contract, "manifest_sha256"),
        SOURCE_PACKET_MANIFEST_SHA256
    );
    assert_eq!(
        string(source_contract, "moshi_lm_sha256"),
        "38991e83d7e3aa0ff1483b27a0b59e1ff43642ad6a32dc8dc4272ad2b6c0dc5f"
    );
    assert_eq!(
        string(source_contract, "moshi_transformer_sha256"),
        "f5a73d752a5bde1eda2b0b14bebd13fd81db81017d361782a07164580a687622"
    );
    assert!(!string(source_contract, "path").is_empty());
    assert_eq!(string(source_contract, "status"), "SOURCE_ONLY_PREPARATION");

    let scheduler = object(&manifest, "scheduler");
    exact_keys(scheduler, &["scheduler"], "scheduler");
    let scheduler_rows = scheduler
        .get("scheduler")
        .and_then(JsonValue::as_array)
        .expect("scheduler rows");
    assert_eq!(
        scheduler_rows.len(),
        CHECKPOINT_STEPS.len() + 1,
        "scheduler checkpoint coverage"
    );
    let mut scheduler_coordinates = BTreeSet::new();
    for row in scheduler_rows {
        exact_keys(
            row,
            &[
                "phase",
                "step",
                "offsets",
                "cache_shape",
                "cache_dtype",
                "cache",
            ],
            "scheduler row",
        );
        let phase = string(row, "phase");
        let step = integer(row, "step") as usize;
        assert!(
            phase == "warmup" || phase == "after_reset",
            "scheduler phase"
        );
        assert!(
            phase == "after_reset" && step == 0
                || phase == "warmup" && CHECKPOINT_STEPS.contains(&step),
            "scheduler coordinate"
        );
        assert!(
            scheduler_coordinates.insert((phase.to_owned(), step)),
            "duplicate scheduler coordinate"
        );
        assert_eq!(
            string(row, "cache_dtype"),
            "torch.bfloat16",
            "scheduler cache dtype"
        );
        assert!(
            row.get("cache_shape")
                .and_then(JsonValue::as_array)
                .is_some(),
            "scheduler cache shape"
        );
        assert!(
            row.get("offsets").and_then(JsonValue::as_array).is_some(),
            "scheduler offsets"
        );
        assert!(
            row.get("cache").and_then(JsonValue::as_array).is_some(),
            "scheduler cache"
        );
    }

    let input = object(&manifest, "input");
    exact_keys(
        input,
        &["audio_codes", "context", "frames", "kind", "model_revision"],
        "input",
    );
    assert_eq!(
        string(input, "kind"),
        "deterministic_mimi_code_boundary_packet"
    );
    assert_eq!(string(input, "model_revision"), HF_REVISION);
    assert_eq!(integer(input, "context"), CONTEXT as u64);
    assert_eq!(integer(input, "frames"), FRAMES as u64);
    let frames = input
        .get("audio_codes")
        .and_then(JsonValue::as_array)
        .expect("audio codes");
    assert_eq!(frames.len(), FRAMES);
    for frame in frames {
        let codes = frame.as_array().expect("audio frame");
        assert_eq!(codes.len(), N_Q);
        for code in codes {
            assert!(
                integer_value(code) < AUDIO_CARD as u64,
                "audio code out of range"
            );
        }
    }

    let execution = object(&manifest, "execution");
    exact_keys(
        execution,
        &[
            "implementation",
            "device",
            "weights_dtype",
            "kv_dtype_recorded_before_conversion",
            "num_threads",
            "num_interop_threads",
            "deterministic_algorithms",
            "publication",
            "environment",
        ],
        "execution",
    );
    assert_eq!(
        string(execution, "implementation"),
        "official Moshi LMGen.step with RingKVCache.complete hook"
    );
    assert_eq!(string(execution, "device"), "cpu");
    assert_eq!(string(execution, "weights_dtype"), "torch.bfloat16");
    assert_eq!(
        execution.get("kv_dtype_recorded_before_conversion"),
        Some(&JsonValue::Bool(true))
    );
    assert_eq!(integer(execution, "num_threads"), 1);
    assert_eq!(integer(execution, "num_interop_threads"), 1);
    assert_eq!(
        execution.get("deterministic_algorithms"),
        Some(&JsonValue::Bool(true))
    );
    assert_eq!(string(execution, "publication"), "NO_UPLOAD");
    assert!(object(execution, "environment").as_object().is_some());

    let artifacts = object(&manifest, "artifacts");
    let mut rows = BTreeMap::new();
    let mut total = 0u64;
    for (name, row) in artifacts.as_object().expect("artifact map") {
        safe_artifact_name(name);
        exact_keys(row, &["bytes", "sha256"], "artifact row");
        let bytes = integer(row, "bytes");
        assert!(
            bytes <= MAX_ARTIFACT_BYTES,
            "artifact exceeds per-file bound: {name}"
        );
        total = total
            .checked_add(bytes)
            .expect("artifact byte total overflow");
        assert!(
            total <= MAX_TOTAL_ARTIFACT_BYTES,
            "artifact total exceeds bounded ceiling"
        );
        let sha = string(row, "sha256").to_owned();
        lower_hex(&sha, 64, "artifact SHA");
        assert!(
            rows.insert(name.clone(), ArtifactRow { bytes, sha256: sha })
                .is_none(),
            "duplicate artifact name"
        );
    }
    assert!(!rows.is_empty(), "reference artifact map is empty");

    let mut actual = BTreeSet::new();
    collect_artifact_files(reference, reference, &mut actual);
    let mut expected = rows.keys().cloned().collect::<BTreeSet<_>>();
    expected.insert("manifest.json".to_owned());
    assert_eq!(
        actual, expected,
        "reference artifact coverage differs from manifest"
    );
    for (name, row) in &rows {
        let path = reference.join(name);
        required_file(&path, "reference artifact");
        assert_eq!(
            fs::metadata(&path).expect("artifact metadata").len(),
            row.bytes,
            "artifact size {name}"
        );
        let body = read_bounded_file(&path, &format!("artifact {name}"), row.bytes);
        assert_eq!(bytes_digest(&body), row.sha256, "artifact digest {name}");
    }

    AuthenticatedReference {
        root: reference.to_owned(),
        manifest,
        artifacts: rows,
    }
}

fn integer_value(value: &JsonValue) -> u64 {
    value
        .as_u64()
        .unwrap_or_else(|| panic!("expected non-negative integer"))
}

const MODEL_NAME: &str = "model.safetensors";

fn artifact_bytes(reference: &AuthenticatedReference, name: &str) -> Vec<u8> {
    let row = reference
        .artifacts
        .get(name)
        .unwrap_or_else(|| panic!("artifact not declared: {name}"));
    let path = reference.root.join(name);
    let bytes = read_bounded_file(&path, &format!("artifact {name}"), row.bytes);
    assert_eq!(
        bytes.len() as u64,
        row.bytes,
        "artifact {name} size changed"
    );
    assert_eq!(
        bytes_digest(&bytes),
        row.sha256,
        "artifact {name} digest changed"
    );
    bytes
}

fn array_usize(value: &JsonValue, label: &str) -> Vec<usize> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("{label} must be an array"))
        .iter()
        .map(|item| integer_value(item) as usize)
        .collect()
}

fn shape(value: &JsonValue, expected: &[usize], label: &str) {
    assert_eq!(array_usize(value, label), expected, "{label} shape");
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn bf16_at(bytes: &[u8], index: usize) -> f32 {
    let start = index.checked_mul(2).expect("BF16 offset overflow");
    assert!(start + 2 <= bytes.len(), "BF16 artifact index out of range");
    bf16_to_f32(u16::from_le_bytes([bytes[start], bytes[start + 1]]))
}

fn f32_at(bytes: &[u8], index: usize) -> f32 {
    let start = index.checked_mul(4).expect("F32 offset overflow");
    assert!(start + 4 <= bytes.len(), "F32 artifact index out of range");
    f32::from_le_bytes(bytes[start..start + 4].try_into().expect("F32 bytes"))
}

fn checkpoint_positions(value: &JsonValue) -> Vec<i64> {
    let outer = value.as_array().expect("positions outer array");
    assert_eq!(outer.len(), 1, "positions batch");
    outer[0]
        .as_array()
        .expect("positions row")
        .iter()
        .map(|item| signed(item, "position"))
        .collect()
}

fn event_key(phase: &str, step: usize, layer: usize) -> String {
    format!("{phase}:{step}:{layer}")
}

fn checkpoint_event<'a>(
    events: &'a [JsonValue],
    phase: &str,
    step: usize,
    layer: usize,
) -> &'a JsonValue {
    events
        .iter()
        .find(|event| {
            string(event, "phase") == phase
                && integer(event, "step") == step as u64
                && integer(event, "layer") == layer as u64
        })
        .unwrap_or_else(|| panic!("missing event {}", event_key(phase, step, layer)))
}

fn validate_events(manifest: &JsonValue, reference: &AuthenticatedReference) -> BTreeSet<String> {
    let ring = object(manifest, "ring_cache");
    exact_keys(
        ring,
        &["events", "state_snapshots", "reset_semantics", "comparison"],
        "ring cache",
    );
    assert!(string(ring, "reset_semantics").contains("stale physical"));
    assert!(string(ring, "comparison").contains("positions=-1"));
    let events = ring
        .get("events")
        .and_then(JsonValue::as_array)
        .expect("ring events");
    assert_eq!(
        events.len(),
        (FRAMES + 1) * N_LAYERS,
        "event coverage count"
    );
    let mut seen = HashSet::new();
    let mut referenced = BTreeSet::new();
    for event in events {
        let phase = string(event, "phase");
        let step = integer(event, "step") as usize;
        let layer = integer(event, "layer") as usize;
        assert!(phase == "warmup" || phase == "after_reset", "unknown phase");
        assert!(
            layer < N_LAYERS
                && (phase == "warmup" && step < FRAMES || phase == "after_reset" && step == 0),
            "event coordinates out of range"
        );
        assert!(
            seen.insert(event_key(phase, step, layer)),
            "duplicate event coordinates"
        );
        let checkpoint = event.get("keys").is_some();
        if checkpoint {
            exact_keys(
                event,
                &[
                    "phase",
                    "step",
                    "layer",
                    "delta_shape",
                    "delta_dtype",
                    "delta_keys",
                    "delta_keys_offset",
                    "delta_keys_bytes",
                    "delta_values",
                    "delta_values_offset",
                    "delta_values_bytes",
                    "end_offset",
                    "capacity",
                    "shape",
                    "dtype",
                    "positions",
                    "keys",
                    "values",
                    "valid_positions",
                ],
                "checkpoint event",
            );
        } else {
            exact_keys(
                event,
                &[
                    "phase",
                    "step",
                    "layer",
                    "delta_shape",
                    "delta_dtype",
                    "delta_keys",
                    "delta_keys_offset",
                    "delta_keys_bytes",
                    "delta_values",
                    "delta_values_offset",
                    "delta_values_bytes",
                    "end_offset",
                    "capacity",
                ],
                "delta event",
            );
        }
        let delta_shape = event.get("delta_shape").expect("delta shape");
        shape(delta_shape, &[1, N_Q, 1, HEAD_DIM], "delta shape");
        assert_eq!(string(event, "delta_dtype"), "torch.bfloat16");
        for key in ["delta_keys", "delta_values"] {
            let name = string(event, key);
            safe_artifact_name(name);
            referenced.insert(name.to_owned());
            let bytes = integer(event, &format!("{key}_bytes"));
            assert_eq!(bytes, (N_Q * HEAD_DIM * 2) as u64, "delta bytes");
            assert!(
                integer(event, &format!("{key}_offset")) + bytes
                    <= reference.artifacts.get(name).expect("delta artifact").bytes,
                "delta range"
            );
        }
        let end_offset = event
            .get("end_offset")
            .and_then(JsonValue::as_array)
            .expect("event end offset");
        assert!(!end_offset.is_empty(), "event end offset");
        assert_eq!(integer(event, "capacity"), CONTEXT as u64, "event capacity");
        if checkpoint {
            assert!(
                event.get("values").is_some() && event.get("positions").is_some(),
                "incomplete checkpoint event"
            );
            shape(
                event.get("shape").expect("checkpoint shape"),
                &[1, N_Q, CONTEXT, HEAD_DIM],
                "checkpoint shape",
            );
            assert_eq!(string(event, "dtype"), "torch.bfloat16");
            let positions = checkpoint_positions(event.get("positions").unwrap());
            assert_eq!(positions.len(), CONTEXT);
            let mut valid = positions
                .iter()
                .copied()
                .filter(|value| *value >= 0)
                .collect::<Vec<_>>();
            valid.sort_unstable();
            let manifest_valid = event
                .get("valid_positions")
                .map(|v| v.as_array().expect("valid positions"))
                .expect("valid positions")
                .iter()
                .map(|v| signed(v, "valid position"))
                .collect::<Vec<_>>();
            assert_eq!(valid, manifest_valid, "valid positions order");
            for key in ["keys", "values"] {
                let name = string(event, key);
                safe_artifact_name(name);
                referenced.insert(name.to_owned());
                assert_eq!(
                    reference
                        .artifacts
                        .get(name)
                        .expect("checkpoint artifact")
                        .bytes,
                    (N_Q * CONTEXT * HEAD_DIM * 2) as u64
                );
            }
        }
    }
    assert_eq!(seen.len(), events.len());
    for step in 0..FRAMES {
        for layer in 0..N_LAYERS {
            assert!(seen.contains(&event_key("warmup", step, layer)));
        }
    }
    for layer in 0..N_LAYERS {
        assert!(seen.contains(&event_key("after_reset", 0, layer)));
    }
    let snapshots = ring
        .get("state_snapshots")
        .and_then(JsonValue::as_array)
        .expect("state snapshots");
    assert_eq!(snapshots.len(), 2 * N_LAYERS, "state snapshot coverage");
    let mut snapshot_coordinates = BTreeSet::new();
    for snapshot in snapshots {
        exact_keys(
            snapshot,
            &["phase", "step", "layer", "end_offset", "capacity"],
            "state snapshot",
        );
        let phase = string(snapshot, "phase");
        let step = integer(snapshot, "step") as usize;
        let layer = integer(snapshot, "layer") as usize;
        assert!(
            (phase == "initial" || phase == "after_reset") && step == 0 && layer < N_LAYERS,
            "state snapshot coordinate"
        );
        assert!(
            snapshot_coordinates.insert((phase.to_owned(), step, layer)),
            "duplicate state snapshot"
        );
        assert_eq!(integer(snapshot, "capacity"), CONTEXT as u64);
        let offsets = snapshot
            .get("end_offset")
            .and_then(JsonValue::as_array)
            .expect("snapshot end offset");
        assert!(
            !offsets.is_empty() && offsets.iter().all(|value| integer_value(value) == 0),
            "snapshot reset offset"
        );
    }
    referenced
}

fn parse_audio_codes(manifest: &JsonValue) -> Vec<Vec<u32>> {
    object(manifest, "input")
        .get("audio_codes")
        .and_then(JsonValue::as_array)
        .expect("audio codes")
        .iter()
        .map(|frame| {
            frame
                .as_array()
                .expect("audio frame")
                .iter()
                .map(|code| integer_value(code) as u32)
                .collect()
        })
        .collect()
}

fn expected_text_tokens(reference: &AuthenticatedReference) -> BTreeMap<(String, usize), u32> {
    let text = object(&reference.manifest, "text");
    exact_keys(text, &["tokens", "logits"], "text evidence");
    let tokens = text
        .get("tokens")
        .and_then(JsonValue::as_array)
        .expect("text tokens");
    let mut result = BTreeMap::new();
    for event in tokens {
        exact_keys(
            event,
            &["phase", "step", "shape", "dtype", "values"],
            "text token event",
        );
        let phase = string(event, "phase").to_owned();
        let step = integer(event, "step") as usize;
        assert!(
            phase == "warmup" && step < FRAMES || phase == "after_reset" && step == 0,
            "text token coordinate"
        );
        let token_shape = array_usize(
            event.get("shape").expect("text token shape"),
            "text token shape",
        );
        assert!(
            !token_shape.is_empty() && token_shape.iter().copied().product::<usize>() == 1,
            "text token shape"
        );
        assert_eq!(string(event, "dtype"), "torch.int64");
        let values = event
            .get("values")
            .and_then(JsonValue::as_array)
            .expect("text values");
        assert_eq!(values.len(), 1, "one greedy text token per frame");
        let token = integer_value(&values[0]) as u32;
        assert!(token < TEXT_CARD as u32, "text token range");
        assert!(
            result.insert((phase, step), token).is_none(),
            "duplicate text token event"
        );
    }
    assert_eq!(result.len(), FRAMES + 1, "text token event coverage");
    for step in 0..FRAMES {
        assert!(
            result.contains_key(&("warmup".to_owned(), step)),
            "missing warmup text token"
        );
    }
    assert!(
        result.contains_key(&("after_reset".to_owned(), 0)),
        "missing post-reset text token"
    );
    let logits = text
        .get("logits")
        .and_then(JsonValue::as_array)
        .expect("text logits");
    assert_eq!(
        logits.len(),
        CHECKPOINT_STEPS.len() + 1,
        "text logits coverage"
    );
    let mut logits_coordinates = BTreeSet::new();
    for row in logits {
        exact_keys(
            row,
            &[
                "phase",
                "step",
                "shape",
                "source_shape",
                "dtype",
                "source_dtype",
                "conversion",
                "artifact",
                "bytes",
            ],
            "text logits row",
        );
        let phase = string(row, "phase");
        let step = integer(row, "step") as usize;
        assert!(
            phase == "warmup" && CHECKPOINT_STEPS.contains(&step)
                || phase == "after_reset" && step == 0,
            "text logits coordinate"
        );
        assert!(
            logits_coordinates.insert((phase.to_owned(), step)),
            "duplicate text logits row"
        );
        let logit_shape = array_usize(row.get("shape").expect("logit shape"), "logit shape");
        assert_eq!(logit_shape, vec![1, TEXT_CARD], "logit comparison shape");
        let source_shape = array_usize(
            row.get("source_shape").expect("logit source shape"),
            "logit source shape",
        );
        assert_eq!(source_shape, vec![1, 1, 1, TEXT_CARD], "logit source shape");
        assert_eq!(string(row, "dtype"), "torch.float32");
        assert!(
            !string(row, "source_dtype").is_empty(),
            "logit source dtype"
        );
        assert_eq!(
            string(row, "conversion"),
            "official logits converted to contiguous torch.float32 for comparison export"
        );
        safe_artifact_name(string(row, "artifact"));
        let artifact = reference
            .artifacts
            .get(string(row, "artifact"))
            .expect("logit artifact");
        assert_eq!(
            integer(row, "bytes"),
            artifact.bytes,
            "logit artifact bytes"
        );
    }
    result
}

fn argmax(values: &[f32]) -> usize {
    assert!(!values.is_empty(), "empty logits");
    assert!(
        values.iter().all(|value| value.is_finite()),
        "non-finite logits"
    );
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

fn compare_checkpoint(
    event: &JsonValue,
    reference: &AuthenticatedReference,
    stream: &vokra_models::kyutai_stt::KyutaiSttStreamingLm<'_>,
    metrics: &mut Metrics,
) {
    let layer = integer(event, "layer") as usize;
    let positions = checkpoint_positions(event.get("positions").expect("checkpoint positions"));
    let mut valid = positions
        .iter()
        .enumerate()
        .filter_map(|(physical, position)| {
            (*position >= 0).then_some((physical, *position as usize))
        })
        .collect::<Vec<_>>();
    valid.sort_by_key(|(_, position)| *position);
    let (native_positions, native_keys, native_values) =
        stream.layer_cache_view(layer).expect("native layer view");
    assert_eq!(
        native_positions,
        &valid
            .iter()
            .map(|(_, position)| *position)
            .collect::<Vec<_>>()
    );
    let key_bytes = artifact_bytes(reference, string(event, "keys"));
    let value_bytes = artifact_bytes(reference, string(event, "values"));
    for (row, (physical, position)) in valid.iter().enumerate() {
        for head in 0..N_HEADS {
            for dimension in 0..HEAD_DIM {
                let raw_index = (head * CONTEXT + physical) * HEAD_DIM + dimension;
                let official_key = bf16_at(&key_bytes, raw_index);
                let official_value = bf16_at(&value_bytes, raw_index);
                let native_index = row * D_MODEL + head * HEAD_DIM + dimension;
                let native_key = native_keys[native_index];
                let native_value = native_values[native_index];
                assert!(
                    official_key.is_finite()
                        && official_value.is_finite()
                        && native_key.is_finite()
                        && native_value.is_finite(),
                    "non-finite KV evidence"
                );
                for (kind, official, native) in [
                    ("key", official_key, native_key),
                    ("value", official_value, native_value),
                ] {
                    let difference = (official - native).abs();
                    metrics.kv_abs_sum += f64::from(difference);
                    metrics.kv_values += 1;
                    if difference > metrics.max_abs {
                        metrics.max_abs = difference;
                        metrics.kv_worst = Some(format!(
                            "{}:{}:{}:{}:{}:{}:{}",
                            string(event, "phase"),
                            integer(event, "step"),
                            layer,
                            position,
                            head,
                            dimension,
                            kind
                        ));
                    }
                }
            }
        }
    }
}

fn compare_logits(
    reference: &AuthenticatedReference,
    phase: &str,
    step: usize,
    native: &[f32],
    metrics: &mut Metrics,
) {
    assert_eq!(native.len(), TEXT_CARD, "native logits width");
    let rows = object(&reference.manifest, "text")
        .get("logits")
        .and_then(JsonValue::as_array)
        .expect("text logits");
    let row = rows
        .iter()
        .find(|row| string(row, "phase") == phase && integer(row, "step") == step as u64)
        .unwrap_or_else(|| panic!("missing logits artifact {phase}:{step}"));
    let bytes = artifact_bytes(reference, string(row, "artifact"));
    assert_eq!(bytes.len(), TEXT_CARD * 4, "logits artifact width");
    for (index, actual) in native.iter().copied().enumerate() {
        let expected = f32_at(&bytes, index);
        assert!(
            actual.is_finite() && expected.is_finite(),
            "non-finite logits evidence"
        );
        let difference = (actual - expected).abs();
        metrics.logit_max_abs = metrics.logit_max_abs.max(difference);
        metrics.logit_abs_sum += f64::from(difference);
        metrics.logit_values += 1;
    }
}

#[derive(Default)]
struct Metrics {
    max_abs: f32,
    kv_abs_sum: f64,
    kv_values: u64,
    kv_worst: Option<String>,
    logit_max_abs: f32,
    logit_abs_sum: f64,
    logit_values: u64,
    argmax_matches: usize,
    argmax_total: usize,
}

fn authenticate_and_run() {
    assert_eq!(
        std::env::consts::OS,
        "linux",
        "streaming reference requires Linux"
    );
    assert_eq!(
        std::env::consts::ARCH,
        "x86_64",
        "streaming reference requires x86_64"
    );
    let gguf = required_path(GGUF_ENV);
    let reference = required_path(REFERENCE_ENV);
    required_directory(&reference, "streaming reference");
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("checkout root");
    require_disjoint(&[(&reference, "streaming reference"), (checkout, "checkout")]);
    let manifest_path = reference.join("manifest.json");
    let manifest_sha = std::env::var(REFERENCE_SHA_ENV).expect("reference manifest SHA");
    lower_hex(&manifest_sha, 64, "reference manifest SHA");
    let (manifest_bytes, manifest) =
        parse_json(&manifest_path, "streaming manifest", MAX_MANIFEST_BYTES);
    assert_eq!(
        bytes_digest(&manifest_bytes),
        manifest_sha,
        "reference manifest SHA mismatch"
    );
    let approval = required_path(APPROVAL_ENV);
    require_execution_readiness(&approval, &manifest, checkout);
    required_file(&gguf, "streaming GGUF");
    require_disjoint(&[
        (&gguf, "streaming GGUF"),
        (&reference, "streaming reference"),
        (checkout, "checkout"),
    ]);
    let gguf_sha = std::env::var(GGUF_SHA_ENV).expect("streaming GGUF SHA");
    lower_hex(&gguf_sha, 64, "streaming GGUF SHA");
    assert_eq!(file_digest(&gguf), gguf_sha, "streaming GGUF SHA mismatch");
    let reference = authenticate_reference(&reference, manifest);
    let event_artifacts = validate_events(&reference.manifest, &reference);
    let expected_artifacts = reference.artifacts.keys().cloned().collect::<BTreeSet<_>>();
    let mut referenced = event_artifacts;
    referenced.insert("input.json".to_owned());
    referenced.insert("text_tokens.i64".to_owned());
    if let Some(logits) = object(&reference.manifest, "text")
        .get("logits")
        .and_then(JsonValue::as_array)
    {
        for row in logits {
            referenced.insert(string(row, "artifact").to_owned());
        }
    }
    assert_eq!(
        referenced, expected_artifacts,
        "artifact coverage is incomplete or has unreferenced files"
    );

    let input_bytes = artifact_bytes(&reference, "input.json");
    reject_duplicate_keys(&input_bytes, "input artifact");
    let input = vokra_core::json::parse(&input_bytes).expect("input artifact JSON");
    assert_eq!(
        input,
        *object(&reference.manifest, "input"),
        "input manifest/artifact mismatch"
    );
    let audio_codes = parse_audio_codes(&reference.manifest);
    let expected_tokens = expected_text_tokens(&reference);
    let token_bytes = artifact_bytes(&reference, "text_tokens.i64");
    assert_eq!(token_bytes.len() % 8, 0, "text token artifact alignment");
    assert_eq!(
        token_bytes.len(),
        (FRAMES + 1) * 8,
        "text token artifact coverage"
    );
    let expected_sequence = (0..FRAMES)
        .map(|step| {
            *expected_tokens
                .get(&("warmup".to_owned(), step))
                .expect("warmup token sequence")
        })
        .chain(std::iter::once(
            *expected_tokens
                .get(&("after_reset".to_owned(), 0))
                .expect("post-reset token sequence"),
        ))
        .collect::<Vec<_>>();
    for (index, expected) in expected_sequence.iter().enumerate() {
        let start = index * 8;
        let token = u64::from_le_bytes(
            token_bytes[start..start + 8]
                .try_into()
                .expect("token bytes"),
        );
        assert_eq!(
            token, *expected as u64,
            "text token artifact mismatch at {index}"
        );
    }

    let file = GgufFile::open(&gguf).expect("open authenticated streaming GGUF");
    assert_eq!(
        file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE),
        Some(&GgufMetadataValue::String(
            "attribution-required".to_owned()
        ))
    );
    assert_eq!(
        file.get(chunks::KEY_PROVENANCE_LICENSE),
        Some(&GgufMetadataValue::String("cc-by-4.0".to_owned()))
    );
    let weights =
        KyutaiSttWeights::from_component_gguf(&file).expect("bind authenticated Kyutai weights");
    assert!(
        !weights.is_synthesized,
        "streaming consumer must not use synthesized weights"
    );
    let asr = KyutaiSttAsr::new(
        vokra_models::kyutai_stt::KyutaiSttConfig::stt_2_6b_en(),
        weights,
    )
    .expect("Kyutai ASR engine");
    let mut stream = asr
        .streaming_lm(BackendKind::Cpu)
        .expect("CPU streaming LM");
    for layer in 0..N_LAYERS {
        assert_eq!(stream.layer_cache_view(layer).unwrap().0, &[]);
    }
    let mut metrics = Metrics::default();
    for (step, audio_frame) in audio_codes.iter().enumerate() {
        let previous = (step > 0).then(|| {
            *expected_tokens
                .get(&("warmup".to_owned(), step - 1))
                .expect("warmup previous text token")
        });
        let output = stream
            .step_frame(previous, audio_frame)
            .expect("native streaming step");
        let actual = output.logits().as_slice();
        assert!(
            actual.iter().all(|value| value.is_finite()),
            "native logits non-finite"
        );
        if CHECKPOINT_STEPS.contains(&step) {
            compare_logits(&reference, "warmup", step, actual, &mut metrics);
        }
        let expected = *expected_tokens
            .get(&("warmup".to_owned(), step))
            .expect("warmup text token");
        metrics.argmax_total += 1;
        if argmax(actual) == expected as usize {
            metrics.argmax_matches += 1;
        }
        if CHECKPOINT_STEPS.contains(&step) {
            for layer in 0..N_LAYERS {
                compare_checkpoint(
                    checkpoint_event(
                        object(&reference.manifest, "ring_cache")
                            .get("events")
                            .and_then(JsonValue::as_array)
                            .unwrap(),
                        "warmup",
                        step,
                        layer,
                    ),
                    &reference,
                    &stream,
                    &mut metrics,
                );
            }
        }
    }
    stream.reset();
    for layer in 0..N_LAYERS {
        assert_eq!(stream.layer_cache_view(layer).unwrap().0, &[]);
    }
    let output = stream
        .step_frame(None, &audio_codes[0])
        .expect("native post-reset step");
    compare_logits(
        &reference,
        "after_reset",
        0,
        output.logits().as_slice(),
        &mut metrics,
    );
    let expected = *expected_tokens
        .get(&("after_reset".to_owned(), 0))
        .expect("post-reset text token");
    metrics.argmax_total += 1;
    if argmax(output.logits().as_slice()) == expected as usize {
        metrics.argmax_matches += 1;
    }
    for layer in 0..N_LAYERS {
        compare_checkpoint(
            checkpoint_event(
                object(&reference.manifest, "ring_cache")
                    .get("events")
                    .and_then(JsonValue::as_array)
                    .unwrap(),
                "after_reset",
                0,
                layer,
            ),
            &reference,
            &stream,
            &mut metrics,
        );
    }
    println!(
        "KYUTAI_STT_STREAMING verdict=NOT_PARITY_PASS frames={} layers={} kv_values={} kv_max_abs={} kv_mean_abs={} kv_worst={} logits={} logit_max_abs={} logit_mean_abs={} argmax={}/{} fixed_atol={:?}",
        FRAMES + 1,
        N_LAYERS,
        metrics.kv_values,
        metrics.max_abs,
        metrics.kv_abs_sum / metrics.kv_values as f64,
        metrics.kv_worst.as_deref().unwrap_or("none"),
        metrics.logit_values,
        metrics.logit_max_abs,
        metrics.logit_abs_sum / metrics.logit_values as f64,
        metrics.argmax_matches,
        metrics.argmax_total,
        FIXED_ATOL
    );
    panic!(
        "Kyutai streaming measurements are NOT_PARITY_PASS until a reviewed numerical bound is approved"
    );
}

#[test]
#[ignore = "VAST-only: requires authenticated Kyutai GGUF and official streaming reference packet"]
fn parity_kyutai_stt_streaming_real() {
    authenticate_and_run();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_layout_uses_physical_ring_index() {
        let bytes = [0x80, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40];
        assert_eq!(bf16_at(&bytes, 0), 1.0);
        assert_eq!(bf16_at(&bytes, 1), 2.0);
        assert_eq!(bf16_at(&bytes, 2), 3.0);
        assert_eq!(bf16_at(&bytes, 3), 4.0);
    }

    #[test]
    fn unsafe_artifact_paths_are_rejected() {
        for path in ["/absolute", "../escape", "nested/../escape", "./dot"] {
            let result = std::panic::catch_unwind(|| safe_artifact_name(path));
            assert!(result.is_err(), "unsafe path accepted: {path}");
        }
        safe_artifact_name("kv/warmup-step-0000-layer-00-keys.bin");
    }

    #[test]
    fn duplicate_json_keys_are_rejected() {
        let result =
            std::panic::catch_unwind(|| reject_duplicate_keys(br#"{"a":1,"a":2}"#, "fixture"));
        assert!(result.is_err());
        let escaped = std::panic::catch_unwind(|| {
            reject_duplicate_keys(br#"{"a":{"nested":1},"\u0061":{"nested":2}}"#, "escaped")
        });
        assert!(escaped.is_err());
        let nested = std::panic::catch_unwind(|| {
            reject_duplicate_keys(br#"{"rows":[{"k":1,"k":2}]}"#, "nested")
        });
        assert!(nested.is_err());
        reject_duplicate_keys(br#"{"rows":[{"k":1},{"k":2}]}"#, "array-of-objects");
    }

    #[test]
    fn valid_positions_are_absolute_and_sorted_before_native_compare() {
        let value = JsonValue::Array(vec![JsonValue::Array(vec![
            JsonValue::Int(-1),
            JsonValue::Int(7),
            JsonValue::Int(6),
        ])]);
        let positions = checkpoint_positions(&value);
        let mut valid = positions
            .iter()
            .enumerate()
            .filter_map(|(index, position)| (*position >= 0).then_some((index, *position as usize)))
            .collect::<Vec<_>>();
        valid.sort_by_key(|(_, position)| *position);
        assert_eq!(valid, vec![(2, 6), (1, 7)]);
    }

    #[test]
    fn unreviewed_dependency_closure_blocks_weight_gate() {
        assert!(REVIEWED_DEPENDENCY_CLOSURES.is_empty());
        assert!(!reviewed_dependency_closure(
            "0000000000000000000000000000000000000000000000000000000000000000"
        ));
    }

    #[test]
    fn argmax_matches_torch_first_max_and_rejects_nonfinite() {
        assert_eq!(argmax(&[4.0, 9.0, 9.0, 1.0]), 1);
        let result = std::panic::catch_unwind(|| argmax(&[1.0, f32::NAN]));
        assert!(result.is_err());
    }

    #[test]
    fn bounded_reader_rejects_growth_bound_before_json_parse() {
        let temp_root = std::env::temp_dir().canonicalize().unwrap();
        let path = temp_root.join(format!(
            "vokra-kyutai-bounded-json-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, b"123").unwrap();
        let metadata = std::fs::metadata(&path).unwrap();
        let identity = file_identity(&metadata).unwrap();
        assert_eq!(identity, file_identity(&metadata).unwrap());
        assert_eq!(read_bounded_file(&path, "bounded fixture", 3), b"123");
        let result = std::panic::catch_unwind(|| read_bounded_file(&path, "bounded fixture", 2));
        assert!(result.is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn artifact_binding_hashes_the_returned_bounded_bytes() {
        let temp_root = std::env::temp_dir().canonicalize().unwrap();
        let root = temp_root.join(format!(
            "vokra-kyutai-artifact-hash-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("artifact.bin"), b"actual").unwrap();
        let mut artifacts = BTreeMap::new();
        artifacts.insert(
            "artifact.bin".to_owned(),
            ArtifactRow {
                bytes: 6,
                sha256: bytes_digest(b"different"),
            },
        );
        let reference = AuthenticatedReference {
            root: root.clone(),
            manifest: JsonValue::Null,
            artifacts,
        };
        let result = std::panic::catch_unwind(|| artifact_bytes(&reference, "artifact.bin"));
        assert!(result.is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bytes_digest_falls_back_when_first_candidate_is_unavailable() {
        let digest = bytes_digest_with_programs(
            b"abc",
            &["vokra-command-that-does-not-exist", "sha256sum", "shasum"],
        )
        .expect("approved digest candidate");
        assert_eq!(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn artifact_walk_rejects_entry_and_depth_explosion() {
        let temp_root = std::env::temp_dir().canonicalize().unwrap();
        let root = temp_root.join(format!(
            "vokra-kyutai-artifact-walk-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let mut deep = root.clone();
        for _ in 0..=MAX_ARTIFACT_DEPTH {
            deep.push("d");
            std::fs::create_dir(&deep).unwrap();
        }
        std::fs::write(deep.join("artifact"), b"x").unwrap();
        let result =
            std::panic::catch_unwind(|| collect_artifact_files(&root, &root, &mut BTreeSet::new()));
        assert!(result.is_err());
        std::fs::remove_dir_all(&root).unwrap();

        let temp_root = std::env::temp_dir().canonicalize().unwrap();
        let root = temp_root.join(format!(
            "vokra-kyutai-artifact-entries-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        for index in 0..=MAX_ARTIFACT_ENTRIES {
            std::fs::write(root.join(format!("{index}")), b"x").unwrap();
        }
        let result =
            std::panic::catch_unwind(|| collect_artifact_files(&root, &root, &mut BTreeSet::new()));
        assert!(result.is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
