//! Strict owner-scope and packet binding for the VibeVoice Realtime CLI.
//!
//! This authenticates input records only. It does not itself authorize a
//! model/audio run, prove a license, or establish numerical parity.

use std::collections::BTreeSet;
use std::path::Path;

use vokra_core::json::{JsonValue, parse as parse_json};

use super::sha256;
use super::vibevoice_realtime_noise::{
    ensure_regular_file, hex_digest, parse_sha256, read_regular_file, reject_duplicate_keys,
};

const MAX_SCOPE_BYTES: u64 = 256 * 1024;
const MAX_REFERENCE_BYTES: u64 = 16 * 1024 * 1024;
const SCOPE_SCHEMA: &str = "vokra-vibevoice-realtime-streaming-execution-v1";
const FORMAT: &str = "vokra-vibevoice-realtime-streaming-reference-v1";
const OPEN_STATUS: &str = "REFERENCE_RUN_OPEN_NOT_RUST_PARITY";
const NO_UPLOAD: &str = "NO_UPLOAD";
const SOURCE_REPOSITORY: &str = "microsoft/VibeVoice";
const SOURCE_REVISION: &str = "94da20d98b2fa7688e9cbfaf7692ddb4954f7600";
const SOURCE_ORIGIN: &str = "https://github.com/microsoft/VibeVoice.git";
const CHECKPOINT_REVISION: &str = "6bce5f06044837fe6d2c5d7a71a84f0416bd57e4";
const CHECKPOINT_BYTES: usize = 2_035_332_888;
const CHECKPOINT_SHA256: &str = "7758b150b8139deb48ac1ff6f181f745c8fedd5511232fd974b3eb217d83b514";
const CONFIG_BYTES: usize = 2_117;
const CONFIG_SHA256: &str = "caee2691e790b04054bbe14a753b40149fa7c0c16fadb58d9adf5412343dcf57";
const PRESET_RELATIVE_PATH: &str = "demo/voices/streaming_model/en-Carter_man.pt";
const PRESET_BYTES: usize = 4_256_002;
const PRESET_GIT_BLOB_SHA1: &str = "1d795ef667e6641eecb8b22452bb853b089bfdbe";
const PRESET_PAYLOAD_SHA256: &str =
    "a7bfdf1cd4939c22469bcfc6f427ae9c4467b3df46c2c14303a39c294cfc6897";
const TOKENIZER_REPOSITORY: &str = "Qwen/Qwen2.5-0.5B";
const TOKENIZER_REVISION: &str = "060db6499f32faf8b98477b0a26969ef7d8b9987";
const TOKENIZER_FILES: &[(&str, usize, &str)] = &[
    (
        "vocab.json",
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
    (
        "merges.txt",
        1_671_839,
        "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    ),
    (
        "tokenizer_config.json",
        7_228,
        "c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9",
    ),
    (
        "tokenizer.json",
        7_031_645,
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    ),
];
const COMPATIBILITY_ROUTE: &str =
    "TRUSTED_RUN_REFERENCE_COMPATIBILITY_AND_OFFICIAL_DYNAMICCACHE_DDP_CACHE_DATA";

/// External bindings required before accepting an owner scope.
pub(crate) struct VibeVoiceRealtimeOwnerInputs<'a> {
    pub(crate) owner_scope: &'a Path,
    pub(crate) canonical_payload: &'a Path,
    pub(crate) expected_owner_scope_sha256: &'a str,
    pub(crate) expected_canonical_sha256: &'a str,
    pub(crate) reference_json: &'a Path,
    pub(crate) expected_reference_sha256: &'a str,
    pub(crate) expected_vokra_head: &'a str,
    pub(crate) expected_vokra_tree_sha1: &'a str,
    pub(crate) measured_text_sha256: &'a str,
    pub(crate) expected_reference_script_sha256: &'a str,
    pub(crate) expected_uv_lock_sha256: &'a str,
    pub(crate) expected_trusted_runner_sha256: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct VibeVoiceRealtimeExecution {
    pub(crate) cfg_scale: f32,
    pub(crate) max_new_tokens: usize,
    pub(crate) ddpm_steps: usize,
    pub(crate) benchmark_warmups: usize,
    pub(crate) benchmark_repeats: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct VibeVoiceRealtimeOwnerContract {
    pub(crate) execution: VibeVoiceRealtimeExecution,
}

struct ScopeValidation<'a> {
    expected_canonical: &'a [u8; 32],
    measured_text: &'a [u8; 32],
    expected_head: &'a str,
    expected_tree: &'a str,
    expected_script: &'a [u8; 32],
    expected_lock: &'a [u8; 32],
    expected_runner: &'a [u8; 32],
}

struct ReferenceValidation<'a> {
    scope: &'a JsonValue,
    expected_scope_file: &'a [u8; 32],
    expected_canonical: &'a [u8; 32],
    expected_text: &'a [u8; 32],
    expected_head: &'a str,
    expected_tree: &'a str,
    expected_script: &'a [u8; 32],
    expected_lock: &'a [u8; 32],
    expected_runner: &'a [u8; 32],
}

impl VibeVoiceRealtimeOwnerContract {
    pub(crate) fn load(inputs: VibeVoiceRealtimeOwnerInputs<'_>) -> Result<Self, String> {
        let scope_file = digest_arg(
            inputs.expected_owner_scope_sha256,
            "owner scope file SHA-256",
        )?;
        let canonical_sha =
            digest_arg(inputs.expected_canonical_sha256, "owner canonical SHA-256")?;
        let reference_sha = digest_arg(inputs.expected_reference_sha256, "reference.json SHA-256")?;
        let head = git_arg(inputs.expected_vokra_head, "expected Vokra HEAD")?;
        let tree = git_arg(inputs.expected_vokra_tree_sha1, "expected Vokra tree SHA-1")?;
        let text_sha = digest_arg(inputs.measured_text_sha256, "measured text SHA-256")?;
        let script_sha = digest_arg(
            inputs.expected_reference_script_sha256,
            "reference script SHA-256",
        )?;
        let lock_sha = digest_arg(inputs.expected_uv_lock_sha256, "reference uv.lock SHA-256")?;
        let runner_sha = digest_arg(
            inputs.expected_trusted_runner_sha256,
            "trusted runner SHA-256",
        )?;

        let scope_bytes = read_checked(
            inputs.owner_scope,
            "owner scope",
            MAX_SCOPE_BYTES,
            &scope_file,
        )?;
        let canonical_bytes = read_checked(
            inputs.canonical_payload,
            "owner canonical payload",
            MAX_SCOPE_BYTES,
            &canonical_sha,
        )?;
        let reference_bytes = read_checked(
            inputs.reference_json,
            "reference.json",
            MAX_REFERENCE_BYTES,
            &reference_sha,
        )?;
        let scope = parse_checked(&scope_bytes, "owner scope")?;
        let canonical = parse_checked(&canonical_bytes, "owner canonical payload")?;
        let reference = parse_checked(&reference_bytes, "reference.json")?;
        reject_duplicate_keys(&scope, "owner_scope")?;
        reject_duplicate_keys(&canonical, "owner_canonical_payload")?;
        reject_duplicate_keys(&reference, "reference")?;

        validate_scope(
            &scope,
            &canonical,
            ScopeValidation {
                expected_canonical: &canonical_sha,
                measured_text: &text_sha,
                expected_head: &head,
                expected_tree: &tree,
                expected_script: &script_sha,
                expected_lock: &lock_sha,
                expected_runner: &runner_sha,
            },
        )?;
        let execution = parse_execution(&scope)?;
        validate_reference(
            &reference,
            ReferenceValidation {
                scope: &scope,
                expected_scope_file: &scope_file,
                expected_canonical: &canonical_sha,
                expected_text: &text_sha,
                expected_head: &head,
                expected_tree: &tree,
                expected_script: &script_sha,
                expected_lock: &lock_sha,
                expected_runner: &runner_sha,
            },
        )?;
        Ok(Self { execution })
    }
}

fn read_checked(
    path: &Path,
    label: &str,
    max_bytes: u64,
    expected: &[u8; 32],
) -> Result<Vec<u8>, String> {
    ensure_regular_file(path, label)?;
    let bytes = read_regular_file(path, label, max_bytes)?;
    let actual = sha256(&bytes);
    if &actual != expected {
        return Err(format!(
            "{label} SHA-256 mismatch: expected {}, got {}",
            hex_digest(expected),
            hex_digest(&actual)
        ));
    }
    Ok(bytes)
}

fn parse_checked(bytes: &[u8], label: &str) -> Result<JsonValue, String> {
    parse_json(bytes).map_err(|error| format!("{label}: {error}"))
}

fn digest_arg(value: &str, label: &str) -> Result<[u8; 32], String> {
    parse_sha256(value, label)
}

fn git_arg(value: &str, label: &str) -> Result<String, String> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!(
            "{label} must be 40 lowercase hexadecimal characters"
        ));
    }
    Ok(value.to_owned())
}

fn validate_scope(
    scope: &JsonValue,
    canonical: &JsonValue,
    context: ScopeValidation<'_>,
) -> Result<(), String> {
    exact_keys(
        scope,
        &[
            "schema",
            "owner",
            "approved_at_utc",
            "decision",
            "publication",
            "source",
            "checkpoint",
            "config",
            "preset",
            "tokenizer",
            "voice",
            "execution",
            "dependencies",
            "input",
            "runtime",
            "scope_sha256",
        ],
        "owner scope",
    )?;
    exact_keys(
        canonical,
        &[
            "schema",
            "owner",
            "approved_at_utc",
            "decision",
            "publication",
            "source",
            "checkpoint",
            "config",
            "preset",
            "tokenizer",
            "voice",
            "execution",
            "dependencies",
            "input",
            "runtime",
        ],
        "owner canonical payload",
    )?;
    if parse_sha256(
        string_field(scope, "scope_sha256", "owner scope")?,
        "owner scope scope_sha256",
    )? != *context.expected_canonical
        || !equal_without_scope_hash(scope, canonical)
    {
        return Err("owner scope does not match externally authenticated canonical payload".into());
    }
    if string_field(scope, "schema", "owner scope")? != SCOPE_SCHEMA
        || string_field(scope, "publication", "owner scope")? != NO_UPLOAD
    {
        return Err("owner scope schema/publication mismatch".into());
    }
    nonempty(string_field(scope, "owner", "owner scope")?, "owner")?;
    validate_timestamp(string_field(scope, "approved_at_utc", "owner scope")?)?;
    if !matches!(
        string_field(scope, "decision", "owner scope")?,
        "APPROVE_COMMERCIAL_EXECUTION" | "APPROVE_RESEARCH_ONLY_EXECUTION"
    ) {
        return Err("owner scope decision is not an execution approval".into());
    }
    validate_source(scope)?;
    validate_checkpoint(scope)?;
    validate_config(scope)?;
    validate_preset(scope)?;
    validate_tokenizer(scope)?;
    validate_voice(scope)?;
    validate_dependencies(scope, context.expected_lock)?;
    validate_input(scope, context.measured_text)?;
    validate_runtime(
        scope,
        context.expected_head,
        context.expected_tree,
        context.expected_script,
        context.expected_lock,
        context.expected_runner,
    )?;
    Ok(())
}

fn validate_reference(
    reference: &JsonValue,
    context: ReferenceValidation<'_>,
) -> Result<(), String> {
    if string_field(reference, "format", "reference")? != FORMAT
        || string_field(reference, "status", "reference")? != OPEN_STATUS
        || string_field(reference, "publication", "reference")? != NO_UPLOAD
    {
        return Err("reference packet format/status/publication mismatch".into());
    }
    let source = object_field(reference, "source", "reference")?;
    if string_field(source, "repository", "reference.source")? != SOURCE_REPOSITORY
        || string_field(source, "revision", "reference.source")? != SOURCE_REVISION
        || string_field(source, "origin", "reference.source")? != SOURCE_ORIGIN
    {
        return Err("reference source identity mismatch".into());
    }
    let checkpoint = object_field(reference, "checkpoint", "reference")?;
    if usize_field(checkpoint, "bytes", "reference.checkpoint")? != CHECKPOINT_BYTES
        || string_field(checkpoint, "sha256", "reference.checkpoint")? != CHECKPOINT_SHA256
    {
        return Err("reference checkpoint identity mismatch".into());
    }
    let config = object_field(reference, "config", "reference")?;
    if usize_field(config, "bytes", "reference.config")? != CONFIG_BYTES
        || string_field(config, "sha256", "reference.config")? != CONFIG_SHA256
    {
        return Err("reference config identity mismatch".into());
    }
    let preset = object_field(reference, "preset", "reference")?;
    if string_field(preset, "relative_path", "reference.preset")? != PRESET_RELATIVE_PATH
        || usize_field(preset, "bytes", "reference.preset")? != PRESET_BYTES
        || string_field(preset, "git_blob_sha1", "reference.preset")? != PRESET_GIT_BLOB_SHA1
        || string_field(preset, "payload_sha256", "reference.preset")? != PRESET_PAYLOAD_SHA256
    {
        return Err("reference Carter preset identity mismatch".into());
    }
    let tokenizer = object_field(reference, "tokenizer", "reference")?;
    exact_keys(
        tokenizer,
        &["repository", "revision", "files"],
        "reference.tokenizer",
    )?;
    if string_field(tokenizer, "repository", "reference.tokenizer")? != TOKENIZER_REPOSITORY
        || string_field(tokenizer, "revision", "reference.tokenizer")? != TOKENIZER_REVISION
    {
        return Err("reference tokenizer identity mismatch".into());
    }
    let files = object_field(tokenizer, "files", "reference.tokenizer")?;
    let expected_files: BTreeSet<&str> = TOKENIZER_FILES.iter().map(|(name, _, _)| *name).collect();
    let actual_files: BTreeSet<&str> = files
        .as_object()
        .ok_or("reference.tokenizer.files must be an object")?
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    if expected_files != actual_files {
        return Err("reference tokenizer must contain exactly four files".into());
    }
    for (name, bytes, expected_sha) in TOKENIZER_FILES {
        let identity = files
            .as_object()
            .and_then(|entries| entries.iter().find(|(key, _)| key == name))
            .map(|(_, value)| value)
            .ok_or_else(|| format!("reference tokenizer missing {name}"))?;
        exact_keys(
            identity,
            &["path", "bytes", "sha256"],
            &format!("reference.tokenizer.files.{name}"),
        )?;
        let path = string_field(identity, "path", "reference tokenizer file")?;
        if path.is_empty()
            || !Path::new(path).is_absolute()
            || usize_field(identity, "bytes", "reference tokenizer file")? != *bytes
            || string_field(identity, "sha256", "reference tokenizer file")? != *expected_sha
        {
            return Err(format!("reference tokenizer {name} identity mismatch"));
        }
        parse_sha256(expected_sha, &format!("reference tokenizer {name}"))?;
    }
    if string_field(reference, "owner_scope_sha256", "reference")?
        != hex_digest(context.expected_canonical)
        || string_field(reference, "owner_scope_file_sha256", "reference")?
            != hex_digest(context.expected_scope_file)
    {
        return Err("reference owner-scope SHA binding mismatch".into());
    }
    let input = object_field(reference, "input", "reference")?;
    if string_field(input, "text_sha256", "reference.input")? != hex_digest(context.expected_text)
        || usize_field(input, "sample_rate", "reference.input")? != 24_000
    {
        return Err("reference text/sample-rate binding mismatch".into());
    }
    let runtime = object_field(reference, "runtime", "reference")?;
    if string_field(runtime, "vokra_head", "reference.runtime")? != context.expected_head
        || string_field(runtime, "vokra_tree_sha1", "reference.runtime")? != context.expected_tree
    {
        return Err("reference Vokra HEAD/tree binding mismatch".into());
    }
    for (name, expected) in [
        ("reference_script", context.expected_script),
        ("uv_lock", context.expected_lock),
        ("trusted_runner", context.expected_runner),
    ] {
        let identity = object_field(runtime, name, "reference.runtime")?;
        validate_reference_identity(identity, &format!("reference.runtime.{name}"), expected)?;
    }
    if string_field(context.scope, "scope_sha256", "owner scope")?
        != hex_digest(context.expected_canonical)
    {
        return Err("owner scope/reference canonical SHA mismatch".into());
    }
    Ok(())
}

fn validate_reference_identity(
    value: &JsonValue,
    label: &str,
    expected_sha: &[u8; 32],
) -> Result<(), String> {
    exact_keys(value, &["path", "bytes", "sha256"], label)?;
    let path = string_field(value, "path", label)?;
    if path.is_empty() || !Path::new(path).is_absolute() {
        return Err(format!("{label}.path must be a non-empty absolute path"));
    }
    if usize_field(value, "bytes", label)? == 0 {
        return Err(format!("{label}.bytes must be non-zero"));
    }
    if string_field(value, "sha256", label)? != hex_digest(expected_sha) {
        return Err(format!("{label}.sha256 mismatch"));
    }
    Ok(())
}

fn validate_source(scope: &JsonValue) -> Result<(), String> {
    let value = object_field(scope, "source", "owner scope")?;
    exact_keys(value, &["repository", "revision"], "owner scope.source")?;
    if string_field(value, "repository", "owner scope.source")? != SOURCE_REPOSITORY
        || string_field(value, "revision", "owner scope.source")? != SOURCE_REVISION
    {
        return Err("owner scope source identity mismatch".into());
    }
    Ok(())
}

fn validate_checkpoint(scope: &JsonValue) -> Result<(), String> {
    let value = object_field(scope, "checkpoint", "owner scope")?;
    exact_keys(
        value,
        &["revision", "sha256", "bytes"],
        "owner scope.checkpoint",
    )?;
    if string_field(value, "revision", "owner scope.checkpoint")? != CHECKPOINT_REVISION
        || string_field(value, "sha256", "owner scope.checkpoint")? != CHECKPOINT_SHA256
        || usize_field(value, "bytes", "owner scope.checkpoint")? != CHECKPOINT_BYTES
    {
        return Err("owner scope checkpoint identity mismatch".into());
    }
    Ok(())
}

fn validate_config(scope: &JsonValue) -> Result<(), String> {
    let value = object_field(scope, "config", "owner scope")?;
    exact_keys(value, &["sha256", "bytes"], "owner scope.config")?;
    if string_field(value, "sha256", "owner scope.config")? != CONFIG_SHA256
        || usize_field(value, "bytes", "owner scope.config")? != CONFIG_BYTES
    {
        return Err("owner scope config identity mismatch".into());
    }
    Ok(())
}

fn validate_preset(scope: &JsonValue) -> Result<(), String> {
    let value = object_field(scope, "preset", "owner scope")?;
    exact_keys(
        value,
        &["relative_path", "bytes", "git_blob_sha1", "payload_sha256"],
        "owner scope.preset",
    )?;
    if string_field(value, "relative_path", "owner scope.preset")? != PRESET_RELATIVE_PATH
        || usize_field(value, "bytes", "owner scope.preset")? != PRESET_BYTES
        || string_field(value, "git_blob_sha1", "owner scope.preset")? != PRESET_GIT_BLOB_SHA1
        || string_field(value, "payload_sha256", "owner scope.preset")? != PRESET_PAYLOAD_SHA256
    {
        return Err("owner scope Carter preset identity mismatch".into());
    }
    Ok(())
}

fn validate_tokenizer(scope: &JsonValue) -> Result<(), String> {
    let value = object_field(scope, "tokenizer", "owner scope")?;
    exact_keys(
        value,
        &["repository", "revision", "files"],
        "owner scope.tokenizer",
    )?;
    if string_field(value, "repository", "owner scope.tokenizer")? != TOKENIZER_REPOSITORY
        || string_field(value, "revision", "owner scope.tokenizer")? != TOKENIZER_REVISION
    {
        return Err("owner scope tokenizer identity mismatch".into());
    }
    let files = object_field(value, "files", "owner scope.tokenizer")?;
    let expected: BTreeSet<&str> = TOKENIZER_FILES.iter().map(|(name, _, _)| *name).collect();
    let actual: BTreeSet<&str> = files
        .as_object()
        .ok_or("owner scope tokenizer.files must be an object")?
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    if expected != actual {
        return Err("owner scope tokenizer must contain exactly four files".into());
    }
    for (name, bytes, expected_sha) in TOKENIZER_FILES {
        let value = files
            .as_object()
            .and_then(|entries| entries.iter().find(|(key, _)| key == name))
            .map(|(_, value)| value)
            .ok_or_else(|| format!("owner scope tokenizer missing {name}"))?;
        if *bytes == 0 || value.as_str() != Some(*expected_sha) {
            return Err(format!("owner scope tokenizer {name} identity mismatch"));
        }
        parse_sha256(expected_sha, &format!("owner scope tokenizer {name}"))?;
    }
    Ok(())
}

fn validate_voice(scope: &JsonValue) -> Result<(), String> {
    let value = object_field(scope, "voice", "owner scope")?;
    exact_keys(
        value,
        &[
            "consent",
            "consent_evidence_reference",
            "consent_evidence_sha256",
            "disclaimer",
            "watermark",
        ],
        "owner scope.voice",
    )?;
    if string_field(value, "consent", "owner scope.voice")? != "PROVED"
        || string_field(value, "disclaimer", "owner scope.voice")? != "PRESERVE_UPSTREAM"
        || string_field(value, "watermark", "owner scope.voice")? != "DEFERRED_NO_EMBEDDED_CLAIM"
    {
        return Err("owner scope voice consent/disclaimer/watermark mismatch".into());
    }
    nonempty(
        string_field(value, "consent_evidence_reference", "owner scope.voice")?,
        "voice.consent_evidence_reference",
    )?;
    parse_sha256(
        string_field(value, "consent_evidence_sha256", "owner scope.voice")?,
        "owner scope voice consent evidence",
    )?;
    Ok(())
}

fn validate_dependencies(scope: &JsonValue, expected_lock: &[u8; 32]) -> Result<(), String> {
    let value = object_field(scope, "dependencies", "owner scope")?;
    exact_keys(
        value,
        &[
            "uv_lock_sha256",
            "license_audit",
            "security_disposition",
            "use",
            "evidence_reference",
            "evidence_sha256",
        ],
        "owner scope.dependencies",
    )?;
    if string_field(value, "uv_lock_sha256", "owner scope.dependencies")?
        != hex_digest(expected_lock)
        || string_field(value, "license_audit", "owner scope.dependencies")?
            != "APPROVED_FOR_REFERENCE_EXECUTION"
        || string_field(value, "security_disposition", "owner scope.dependencies")?
            != "APPROVED_FOR_REFERENCE_EXECUTION"
        || string_field(value, "use", "owner scope.dependencies")?
            != "REFERENCE_ONLY_NO_RUNTIME_REUSE"
    {
        return Err("owner scope dependency disposition mismatch".into());
    }
    nonempty(
        string_field(value, "evidence_reference", "owner scope.dependencies")?,
        "dependencies.evidence_reference",
    )?;
    parse_sha256(
        string_field(value, "evidence_sha256", "owner scope.dependencies")?,
        "owner scope dependency evidence",
    )?;
    Ok(())
}

fn validate_input(scope: &JsonValue, expected_text: &[u8; 32]) -> Result<(), String> {
    let value = object_field(scope, "input", "owner scope")?;
    exact_keys(value, &["text_sha256", "sample_rate"], "owner scope.input")?;
    if parse_sha256(
        string_field(value, "text_sha256", "owner scope.input")?,
        "owner scope text SHA-256",
    )? != *expected_text
        || usize_field(value, "sample_rate", "owner scope.input")? != 24_000
    {
        return Err("owner scope input binding mismatch".into());
    }
    Ok(())
}

fn validate_runtime(
    scope: &JsonValue,
    expected_head: &str,
    expected_tree: &str,
    expected_script: &[u8; 32],
    expected_lock: &[u8; 32],
    expected_runner: &[u8; 32],
) -> Result<(), String> {
    let value = object_field(scope, "runtime", "owner scope")?;
    exact_keys(
        value,
        &[
            "reference_script_sha256",
            "uv_lock_sha256",
            "trusted_runner_sha256",
            "compatibility_route",
            "seed",
            "noise_policy",
            "device_selection_policy",
            "vokra_head",
            "vokra_tree_sha1",
        ],
        "owner scope.runtime",
    )?;
    let script = string_field(value, "reference_script_sha256", "owner scope.runtime")?;
    let runner = string_field(value, "trusted_runner_sha256", "owner scope.runtime")?;
    parse_sha256(script, "owner scope reference script")?;
    parse_sha256(runner, "owner scope trusted runner")?;
    if string_field(value, "reference_script_sha256", "owner scope.runtime")?
        != hex_digest(expected_script)
        || string_field(value, "uv_lock_sha256", "owner scope.runtime")?
            != hex_digest(expected_lock)
        || string_field(value, "compatibility_route", "owner scope.runtime")? != COMPATIBILITY_ROUTE
        || usize_field(value, "seed", "owner scope.runtime")? != 1234
        || string_field(value, "noise_policy", "owner scope.runtime")? != "CONTROLLED_CPU_TAPE"
        || string_field(value, "device_selection_policy", "owner scope.runtime")?
            != "CUDA_IF_FULL_TRACE_GUARD_AND_MEDIAN_FASTER"
        || string_field(value, "trusted_runner_sha256", "owner scope.runtime")?
            != hex_digest(expected_runner)
        || string_field(value, "vokra_head", "owner scope.runtime")? != expected_head
        || string_field(value, "vokra_tree_sha1", "owner scope.runtime")? != expected_tree
    {
        return Err("owner scope runtime binding mismatch".into());
    }
    Ok(())
}

fn parse_execution(scope: &JsonValue) -> Result<VibeVoiceRealtimeExecution, String> {
    let value = object_field(scope, "execution", "owner scope")?;
    exact_keys(
        value,
        &[
            "model_forward",
            "audio_generation",
            "max_new_tokens",
            "ddpm_steps",
            "cfg_scale",
            "benchmark_warmups",
            "benchmark_repeats",
        ],
        "owner scope.execution",
    )?;
    if string_field(value, "model_forward", "owner scope.execution")? != "APPROVED"
        || string_field(value, "audio_generation", "owner scope.execution")? != "APPROVED"
    {
        return Err("owner scope execution is not approved".into());
    }
    let max_new_tokens = usize_field(value, "max_new_tokens", "owner scope.execution")?;
    let ddpm_steps = usize_field(value, "ddpm_steps", "owner scope.execution")?;
    let benchmark_warmups = usize_field(value, "benchmark_warmups", "owner scope.execution")?;
    let benchmark_repeats = usize_field(value, "benchmark_repeats", "owner scope.execution")?;
    if !(1..=64).contains(&max_new_tokens) || ddpm_steps != 20 {
        return Err("owner scope execution budget mismatch".into());
    }
    if benchmark_warmups != 1 || benchmark_repeats != 3 {
        return Err("owner scope benchmark budget mismatch".into());
    }
    let cfg_scale = match field(value, "cfg_scale", "owner scope.execution")? {
        JsonValue::Int(value) if *value >= 0 => *value as f32,
        JsonValue::Float(value) if value.is_finite() && *value >= 0.0 => *value as f32,
        _ => return Err("owner scope cfg_scale must be finite and non-negative".into()),
    };
    if !cfg_scale.is_finite() {
        return Err("owner scope cfg_scale overflows f32".into());
    }
    Ok(VibeVoiceRealtimeExecution {
        cfg_scale,
        max_new_tokens,
        ddpm_steps,
        benchmark_warmups,
        benchmark_repeats,
    })
}

fn exact_keys(value: &JsonValue, expected: &[&str], label: &str) -> Result<(), String> {
    let entries = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    let expected: BTreeSet<&str> = expected.iter().copied().collect();
    let actual: BTreeSet<&str> = entries.iter().map(|(key, _)| key.as_str()).collect();
    if expected != actual {
        return Err(format!("{label} has unexpected or missing fields"));
    }
    Ok(())
}

fn object_field<'a>(
    value: &'a JsonValue,
    name: &str,
    label: &str,
) -> Result<&'a JsonValue, String> {
    let value = field(value, name, label)?;
    if value.as_object().is_none() {
        return Err(format!("{label}.{name} must be an object"));
    }
    Ok(value)
}

fn field<'a>(value: &'a JsonValue, name: &str, label: &str) -> Result<&'a JsonValue, String> {
    value
        .as_object()
        .and_then(|entries| entries.iter().find(|(key, _)| key == name))
        .map(|(_, value)| value)
        .ok_or_else(|| format!("{label} is missing {name}"))
}

fn string_field<'a>(value: &'a JsonValue, name: &str, label: &str) -> Result<&'a str, String> {
    field(value, name, label)?
        .as_str()
        .ok_or_else(|| format!("{label}.{name} must be a string"))
}

fn usize_field(value: &JsonValue, name: &str, label: &str) -> Result<usize, String> {
    field(value, name, label)?
        .as_u64()
        .ok_or_else(|| format!("{label}.{name} must be a non-negative integer"))
        .and_then(|value| {
            usize::try_from(value).map_err(|_| format!("{label}.{name} overflows usize"))
        })
}

fn nonempty(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty()
        || matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "todo" | "pending" | "tbd" | "unknown"
        )
    {
        return Err(format!("{label} is empty or a placeholder"));
    }
    Ok(())
}

fn validate_timestamp(value: &str) -> Result<(), String> {
    let bytes = value.as_bytes();
    let fraction_end = if bytes.len() > 20 && bytes.get(19) == Some(&b'.') {
        bytes
            .len()
            .checked_sub(1)
            .filter(|end| (1..=6).contains(&(*end - 20)))
    } else {
        Some(19)
    }
    .ok_or("owner scope approved_at_utc has invalid fractional seconds")?;
    if bytes.len() != fraction_end + 1
        || bytes[fraction_end] != b'Z'
        || bytes.get(4) != Some(&b'-')
        || bytes.get(7) != Some(&b'-')
        || bytes.get(10) != Some(&b'T')
        || bytes.get(13) != Some(&b':')
        || bytes.get(16) != Some(&b':')
        || !bytes[..fraction_end]
            .iter()
            .enumerate()
            .all(|(index, byte)| {
                matches!(index, 4 | 7 | 10 | 13 | 16 | 19) || byte.is_ascii_digit()
            })
    {
        return Err("owner scope approved_at_utc must be RFC3339 UTC".into());
    }
    let number = |start: usize, end: usize| value[start..end].parse::<u32>().unwrap_or(u32::MAX);
    let year = number(0, 4);
    let month = number(5, 7);
    let day = number(8, 10);
    let hour = number(11, 13);
    let minute = number(14, 16);
    let second = number(17, 19);
    let leap = year % 400 == 0 || (year % 4 == 0 && year % 100 != 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => 0,
    };
    if year == 0
        || year == u32::MAX
        || days == 0
        || day == 0
        || day > days
        || hour > 23
        || minute > 59
        || second > 59
    {
        return Err("owner scope approved_at_utc is not a valid UTC timestamp".into());
    }
    Ok(())
}

fn equal_without_scope_hash(scope: &JsonValue, canonical: &JsonValue) -> bool {
    let scope = match scope {
        JsonValue::Object(entries) => JsonValue::Object(
            entries
                .iter()
                .filter(|(key, _)| key != "scope_sha256")
                .cloned()
                .collect(),
        ),
        _ => return false,
    };
    json_equal(&scope, canonical)
}

fn json_equal(left: &JsonValue, right: &JsonValue) -> bool {
    match (left, right) {
        (JsonValue::Null, JsonValue::Null)
        | (JsonValue::Bool(_), JsonValue::Bool(_))
        | (JsonValue::Int(_), JsonValue::Int(_))
        | (JsonValue::Str(_), JsonValue::Str(_)) => left == right,
        (JsonValue::Float(left), JsonValue::Float(right)) => left.to_bits() == right.to_bits(),
        (JsonValue::Array(left), JsonValue::Array(right)) => {
            left.len() == right.len() && left.iter().zip(right).all(|(l, r)| json_equal(l, r))
        }
        (JsonValue::Object(left), JsonValue::Object(right)) => {
            left.len() == right.len()
                && left.iter().all(|(key, value)| {
                    right
                        .iter()
                        .find(|(other, _)| other == key)
                        .is_some_and(|(_, other)| json_equal(value, other))
                })
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn json_string(value: &str) -> String {
        let mut escaped = String::with_capacity(value.len() + 2);
        escaped.push('"');
        for character in value.chars() {
            match character {
                '"' => escaped.push_str("\\\""),
                '\\' => escaped.push_str("\\\\"),
                '\u{08}' => escaped.push_str("\\b"),
                '\u{0c}' => escaped.push_str("\\f"),
                '\n' => escaped.push_str("\\n"),
                '\r' => escaped.push_str("\\r"),
                '\t' => escaped.push_str("\\t"),
                character if character.is_control() => {
                    escaped.push_str(&format!("\\u{:04x}", character as u32));
                }
                character => escaped.push(character),
            }
        }
        escaped.push('"');
        escaped
    }

    fn absolute_path_json(root: &Path, leaf: &str) -> String {
        let path = root.join(leaf);
        assert!(path.is_absolute(), "fixture path must be platform-absolute");
        json_string(&path.to_string_lossy())
    }

    #[test]
    fn fixture_json_string_roundtrips_native_escape_forms() {
        let mut value = String::from("quote\" slash\\ ");
        value.extend(
            (0..=0x1f)
                .chain(std::iter::once(0x7f))
                .map(|code| char::from_u32(code).expect("JSON control code point")),
        );
        value.push('Δ');
        let encoded = json_string(&value);
        assert_eq!(
            parse_json(encoded.as_bytes()).unwrap().as_str(),
            Some(value.as_str())
        );

        let verbatim_path = r"\\?\C:\fixtures\owner.json";
        let encoded = json_string(verbatim_path);
        assert_eq!(
            parse_json(encoded.as_bytes()).unwrap().as_str(),
            Some(verbatim_path)
        );
    }

    #[test]
    fn order_independent_json_preserves_float_kind_and_signed_zero() {
        let left = parse_json(br#"{"a":1,"b":-0.0,"nested":{"x":"\u00e9"}}"#).unwrap();
        let right = parse_json("{\"nested\":{\"x\":\"é\"},\"b\":-0.0,\"a\":1}".as_bytes()).unwrap();
        assert!(json_equal(&left, &right));
        assert!(!json_equal(
            &parse_json(br#"{"a":1}"#).unwrap(),
            &parse_json(br#"{"a":1.0}"#).unwrap()
        ));
        assert!(!json_equal(
            &parse_json(br#"{"a":0.0}"#).unwrap(),
            &parse_json(br#"{"a":-0.0}"#).unwrap()
        ));
    }

    #[test]
    fn duplicate_and_unknown_scope_fields_fail_closed() {
        let duplicate = parse_checked(br#"{"a":1,"a":2}"#, "fixture")
            .and_then(|value| reject_duplicate_keys(&value, "fixture"));
        assert!(duplicate.is_err());
        let nested_duplicate = parse_checked(br#"{"nested":{"a":1,"a":2}}"#, "fixture")
            .and_then(|value| reject_duplicate_keys(&value, "fixture"));
        assert!(nested_duplicate.is_err());
        let value = parse_json(br#"{"schema":"x","extra":true}"#).unwrap();
        assert!(exact_keys(&value, &["schema"], "fixture").is_err());
    }

    #[test]
    fn hash_and_timestamp_validation_are_strict() {
        assert!(digest_arg("A", "digest").is_err());
        assert!(validate_timestamp("2026-09-30T00:00:00Z").is_ok());
        assert!(validate_timestamp("2026-02-29T00:00:00Z").is_err());
        assert!(validate_timestamp("2024-02-29T00:00:00.123456Z").is_ok());
    }

    #[test]
    fn invalid_external_hashes_fail_before_any_file_io() {
        let zero_sha = "0".repeat(64);
        let head = "a".repeat(40);
        let tree = "b".repeat(40);
        let result = VibeVoiceRealtimeOwnerContract::load(VibeVoiceRealtimeOwnerInputs {
            owner_scope: Path::new("/path/that/does/not/exist/scope.json"),
            canonical_payload: Path::new("/path/that/does/not/exist/canonical.json"),
            expected_owner_scope_sha256: "not-a-sha",
            expected_canonical_sha256: &zero_sha,
            reference_json: Path::new("/path/that/does/not/exist/reference.json"),
            expected_reference_sha256: &zero_sha,
            expected_vokra_head: &head,
            expected_vokra_tree_sha1: &tree,
            measured_text_sha256: &zero_sha,
            expected_reference_script_sha256: &zero_sha,
            expected_uv_lock_sha256: &zero_sha,
            expected_trusted_runner_sha256: &zero_sha,
        });
        assert!(
            result
                .expect_err("invalid SHA must be rejected before opening paths")
                .contains("64 lowercase hexadecimal")
        );
    }

    #[test]
    fn execution_schema_rejects_unknown_fields_and_unsafe_budgets() {
        let fixture = |cfg_scale: &str,
                       max_new_tokens: usize,
                       ddpm_steps: usize,
                       warmups: usize,
                       repeats: usize,
                       extra: &str| {
            parse_json(
                format!(
                    "{{\"execution\":{{\"audio_generation\":\"APPROVED\",\"benchmark_repeats\":{repeats},\"benchmark_warmups\":{warmups},\"cfg_scale\":{cfg_scale},\"ddpm_steps\":{ddpm_steps},\"max_new_tokens\":{max_new_tokens},\"model_forward\":\"APPROVED\"{extra}}}}}"
                )
                .as_bytes(),
            )
            .unwrap()
        };
        assert!(parse_execution(&fixture("3.0", 1, 20, 1, 3, "")).is_ok());
        assert!(parse_execution(&fixture("65", 1, 20, 1, 3, "")).is_ok());
        assert!(parse_execution(&fixture("3.0", 1, 20, 1, 3, ",\"unknown\":true")).is_err());
        assert!(parse_execution(&fixture("3.0", 0, 20, 1, 3, "")).is_err());
        assert!(parse_execution(&fixture("3.0", 65, 20, 1, 3, "")).is_err());
        assert!(parse_execution(&fixture("3.0", 1, 19, 1, 3, "")).is_err());
        assert!(parse_execution(&fixture("3.0", 1, 21, 1, 3, "")).is_err());
        assert!(parse_execution(&fixture("3.0", 1, 20, 0, 3, "")).is_err());
        assert!(parse_execution(&fixture("3.0", 1, 20, 1, 2, "")).is_err());
        assert!(parse_execution(&fixture("1e39", 1, 20, 1, 3, "")).is_err());
        assert!(parse_execution(&fixture("-1.0", 1, 20, 1, 3, "")).is_err());
        let unapproved = parse_json(
            br#"{"execution":{"audio_generation":"APPROVED","benchmark_repeats":3,"benchmark_warmups":1,"cfg_scale":3.0,"ddpm_steps":20,"max_new_tokens":1,"model_forward":"PENDING"}}"#,
        )
        .unwrap();
        assert!(parse_execution(&unapproved).is_err());
    }

    #[test]
    fn owner_tokenizer_requires_the_complete_four_file_identity_set() {
        let valid = parse_json(
            br#"{"tokenizer":{"repository":"Qwen/Qwen2.5-0.5B","revision":"060db6499f32faf8b98477b0a26969ef7d8b9987","files":{"merges.txt":"599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3","tokenizer.json":"c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539","tokenizer_config.json":"c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9","vocab.json":"ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"}}}"#,
        )
        .unwrap();
        assert!(validate_tokenizer(&valid).is_ok());
        let missing = parse_json(
            br#"{"tokenizer":{"repository":"Qwen/Qwen2.5-0.5B","revision":"060db6499f32faf8b98477b0a26969ef7d8b9987","files":{"vocab.json":"ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"}}}"#,
        )
        .unwrap();
        assert!(validate_tokenizer(&missing).is_err());
    }

    #[test]
    fn dependency_and_voice_dispositions_fail_closed() {
        let lock = [2_u8; 32];
        let dependencies = parse_json(
            format!(
                "{{\"dependencies\":{{\"evidence_reference\":\"synthetic\",\"evidence_sha256\":\"{}\",\"license_audit\":\"APPROVED_FOR_REFERENCE_EXECUTION\",\"security_disposition\":\"APPROVED_FOR_REFERENCE_EXECUTION\",\"use\":\"REFERENCE_ONLY_NO_RUNTIME_REUSE\",\"uv_lock_sha256\":\"{}\"}}}}",
                hex_digest(&lock),
                hex_digest(&lock)
            )
            .as_bytes(),
        )
        .unwrap();
        assert!(validate_dependencies(&dependencies, &lock).is_ok());
        let denied = parse_json(
            br#"{"dependencies":{"evidence_reference":"synthetic","evidence_sha256":"0202020202020202020202020202020202020202020202020202020202020202","license_audit":"PENDING","security_disposition":"APPROVED_FOR_REFERENCE_EXECUTION","use":"REFERENCE_ONLY_NO_RUNTIME_REUSE","uv_lock_sha256":"0202020202020202020202020202020202020202020202020202020202020202"}}"#,
        )
        .unwrap();
        assert!(validate_dependencies(&denied, &lock).is_err());

        let voice = parse_json(
            br#"{"voice":{"consent":"PROVED","consent_evidence_reference":"synthetic","consent_evidence_sha256":"0202020202020202020202020202020202020202020202020202020202020202","disclaimer":"PRESERVE_UPSTREAM","watermark":"DEFERRED_NO_EMBEDDED_CLAIM"}}"#,
        )
        .unwrap();
        assert!(validate_voice(&voice).is_ok());
        let withheld = parse_json(
            br#"{"voice":{"consent":"PENDING","consent_evidence_reference":"synthetic","consent_evidence_sha256":"0202020202020202020202020202020202020202020202020202020202020202","disclaimer":"PRESERVE_UPSTREAM","watermark":"DEFERRED_NO_EMBEDDED_CLAIM"}}"#,
        )
        .unwrap();
        assert!(validate_voice(&withheld).is_err());
    }

    #[test]
    fn reference_identity_requires_absolute_path_bytes_and_external_sha() {
        let expected = [7_u8; 32];
        let temp_root = fs::canonicalize(std::env::temp_dir()).unwrap();
        let native_path = absolute_path_json(&temp_root, "synthetic/reference.py");
        let valid = parse_json(
            format!(
                "{{\"bytes\":12,\"path\":{native_path},\"sha256\":\"{}\"}}",
                hex_digest(&expected)
            )
            .as_bytes(),
        )
        .unwrap();
        assert!(validate_reference_identity(&valid, "identity", &expected).is_ok());
        let relative = parse_json(
            format!(
                "{{\"bytes\":12,\"path\":\"reference.py\",\"sha256\":\"{}\"}}",
                hex_digest(&expected)
            )
            .as_bytes(),
        )
        .unwrap();
        assert!(validate_reference_identity(&relative, "identity", &expected).is_err());
        let zero = parse_json(
            format!(
                "{{\"bytes\":0,\"path\":{native_path},\"sha256\":\"{}\"}}",
                hex_digest(&expected)
            )
            .as_bytes(),
        )
        .unwrap();
        assert!(validate_reference_identity(&zero, "identity", &expected).is_err());
        #[cfg(windows)]
        for root_relative in [r"\synthetic\reference.py", r"C:synthetic\reference.py"] {
            let root_relative = parse_json(
                format!(
                    "{{\"bytes\":12,\"path\":{},\"sha256\":\"{}\"}}",
                    json_string(root_relative),
                    hex_digest(&expected)
                )
                .as_bytes(),
            )
            .unwrap();
            assert!(
                validate_reference_identity(&root_relative, "identity", &expected).is_err(),
                "root-relative Windows path must not satisfy an absolute identity"
            );
        }
    }

    #[test]
    fn bounded_file_guards_reject_oversize_nonregular_and_symlink_inputs() {
        let base = std::fs::canonicalize(std::env::temp_dir()).unwrap();
        let stem = format!("vokra-owner-guards-{}", std::process::id());
        let root = (0..32)
            .map(|index| base.join(format!("{stem}-{index}")))
            .find(|path| match fs::create_dir(path) {
                Ok(()) => true,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => false,
                Err(error) => panic!("create fixture root: {error}"),
            })
            .expect("unique owner guard fixture root");
        let oversized = root.join("oversized.json");
        fs::write(&oversized, vec![b'x'; MAX_SCOPE_BYTES as usize + 1]).unwrap();
        assert_eq!(
            read_checked(&oversized, "oversized scope", MAX_SCOPE_BYTES, &[0; 32]).unwrap_err(),
            format!("oversized scope exceeds the {MAX_SCOPE_BYTES}-byte resource limit")
        );
        let directory = root.join("directory");
        fs::create_dir(&directory).unwrap();
        assert!(ensure_regular_file(&directory, "directory").is_err());
        #[cfg(unix)]
        {
            let link = root.join("link.json");
            std::os::unix::fs::symlink(&oversized, &link).unwrap();
            assert!(ensure_regular_file(&link, "symlink").is_err());
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn authenticates_complete_synthetic_scope_packet_and_rejects_rebound_mismatch() {
        let base = std::fs::canonicalize(std::env::temp_dir()).unwrap();
        let stem = format!("vokra-owner-contract-{}", std::process::id());
        let root = (0..32)
            .map(|index| base.join(format!("{stem}-{index}")))
            .find(|path| match fs::create_dir(path) {
                Ok(()) => true,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => false,
                Err(error) => panic!("create fixture root: {error}"),
            })
            .expect("unique owner fixture root");
        let head = "a".repeat(40);
        let tree = "b".repeat(40);
        let script = "1".repeat(64);
        let lock = "2".repeat(64);
        let runner = "3".repeat(64);
        let text = hex_digest(&sha256(b"hello"));
        let reference_script_path =
            absolute_path_json(root.as_path(), "run_streaming_reference.py");
        let trusted_runner_path = absolute_path_json(root.as_path(), "trusted_runner.py");
        let uv_lock_path = absolute_path_json(root.as_path(), "uv.lock");
        let merges_path = absolute_path_json(root.as_path(), "merges.txt");
        let tokenizer_json_path = absolute_path_json(root.as_path(), "tokenizer.json");
        let tokenizer_config_path = absolute_path_json(root.as_path(), "tokenizer_config.json");
        let vocab_path = absolute_path_json(root.as_path(), "vocab.json");
        let canonical = format!(
            "{{\"approved_at_utc\":\"2026-09-30T00:00:00Z\",\"checkpoint\":{{\"bytes\":{CHECKPOINT_BYTES},\"revision\":\"{CHECKPOINT_REVISION}\",\"sha256\":\"{CHECKPOINT_SHA256}\"}},\"config\":{{\"bytes\":{CONFIG_BYTES},\"sha256\":\"{CONFIG_SHA256}\"}},\"decision\":\"APPROVE_RESEARCH_ONLY_EXECUTION\",\"dependencies\":{{\"evidence_reference\":\"synthetic\",\"evidence_sha256\":\"{runner}\",\"license_audit\":\"APPROVED_FOR_REFERENCE_EXECUTION\",\"security_disposition\":\"APPROVED_FOR_REFERENCE_EXECUTION\",\"use\":\"REFERENCE_ONLY_NO_RUNTIME_REUSE\",\"uv_lock_sha256\":\"{lock}\"}},\"execution\":{{\"audio_generation\":\"APPROVED\",\"benchmark_repeats\":3,\"benchmark_warmups\":1,\"cfg_scale\":3.0,\"ddpm_steps\":20,\"max_new_tokens\":1,\"model_forward\":\"APPROVED\"}},\"input\":{{\"sample_rate\":24000,\"text_sha256\":\"{text}\"}},\"owner\":\"synthetic-model-free-test-owner\",\"preset\":{{\"bytes\":{PRESET_BYTES},\"git_blob_sha1\":\"{PRESET_GIT_BLOB_SHA1}\",\"payload_sha256\":\"{PRESET_PAYLOAD_SHA256}\",\"relative_path\":\"{PRESET_RELATIVE_PATH}\"}},\"publication\":\"NO_UPLOAD\",\"runtime\":{{\"compatibility_route\":\"{COMPATIBILITY_ROUTE}\",\"device_selection_policy\":\"CUDA_IF_FULL_TRACE_GUARD_AND_MEDIAN_FASTER\",\"noise_policy\":\"CONTROLLED_CPU_TAPE\",\"reference_script_sha256\":\"{script}\",\"seed\":1234,\"trusted_runner_sha256\":\"{runner}\",\"uv_lock_sha256\":\"{lock}\",\"vokra_head\":\"{head}\",\"vokra_tree_sha1\":\"{tree}\"}},\"schema\":\"{SCOPE_SCHEMA}\",\"source\":{{\"repository\":\"{SOURCE_REPOSITORY}\",\"revision\":\"{SOURCE_REVISION}\"}},\"tokenizer\":{{\"files\":{{\"merges.txt\":\"599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3\",\"tokenizer.json\":\"c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539\",\"tokenizer_config.json\":\"c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9\",\"vocab.json\":\"ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910\"}},\"repository\":\"{TOKENIZER_REPOSITORY}\",\"revision\":\"{TOKENIZER_REVISION}\"}},\"voice\":{{\"consent\":\"PROVED\",\"consent_evidence_reference\":\"synthetic\",\"consent_evidence_sha256\":\"{script}\",\"disclaimer\":\"PRESERVE_UPSTREAM\",\"watermark\":\"DEFERRED_NO_EMBEDDED_CLAIM\"}}}}"
        );
        let canonical_sha = hex_digest(&sha256(canonical.as_bytes()));
        let scope = format!(
            "{{\"scope_sha256\":\"{canonical_sha}\",{}}}",
            &canonical[1..canonical.len() - 1]
        );
        let scope_sha = hex_digest(&sha256(scope.as_bytes()));
        let reference = format!(
            "{{\"checkpoint\":{{\"bytes\":{CHECKPOINT_BYTES},\"sha256\":\"{CHECKPOINT_SHA256}\"}},\"config\":{{\"bytes\":{CONFIG_BYTES},\"sha256\":\"{CONFIG_SHA256}\"}},\"format\":\"{FORMAT}\",\"input\":{{\"sample_rate\":24000,\"text_sha256\":\"{text}\"}},\"owner_scope_file_sha256\":\"{scope_sha}\",\"owner_scope_sha256\":\"{canonical_sha}\",\"preset\":{{\"bytes\":{PRESET_BYTES},\"git_blob_sha1\":\"{PRESET_GIT_BLOB_SHA1}\",\"payload_sha256\":\"{PRESET_PAYLOAD_SHA256}\",\"relative_path\":\"{PRESET_RELATIVE_PATH}\"}},\"publication\":\"NO_UPLOAD\",\"runtime\":{{\"reference_script\":{{\"bytes\":1,\"path\":{reference_script_path},\"sha256\":\"{script}\"}},\"trusted_runner\":{{\"bytes\":1,\"path\":{trusted_runner_path},\"sha256\":\"{runner}\"}},\"uv_lock\":{{\"bytes\":1,\"path\":{uv_lock_path},\"sha256\":\"{lock}\"}},\"vokra_head\":\"{head}\",\"vokra_tree_sha1\":\"{tree}\"}},\"source\":{{\"origin\":\"{SOURCE_ORIGIN}\",\"repository\":\"{SOURCE_REPOSITORY}\",\"revision\":\"{SOURCE_REVISION}\"}},\"status\":\"{OPEN_STATUS}\",\"tokenizer\":{{\"repository\":\"{TOKENIZER_REPOSITORY}\",\"revision\":\"{TOKENIZER_REVISION}\"}}}}"
        );
        let reference = reference.replace(
            &format!(
                "\"tokenizer\":{{\"repository\":\"{TOKENIZER_REPOSITORY}\",\"revision\":\"{TOKENIZER_REVISION}\"}}"
            ),
            &format!(
                "\"tokenizer\":{{\"files\":{{\"merges.txt\":{{\"bytes\":1671839,\"path\":{merges_path},\"sha256\":\"599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3\"}},\"tokenizer.json\":{{\"bytes\":7031645,\"path\":{tokenizer_json_path},\"sha256\":\"c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539\"}},\"tokenizer_config.json\":{{\"bytes\":7228,\"path\":{tokenizer_config_path},\"sha256\":\"c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9\"}},\"vocab.json\":{{\"bytes\":2776833,\"path\":{vocab_path},\"sha256\":\"ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910\"}}}},\"repository\":\"{TOKENIZER_REPOSITORY}\",\"revision\":\"{TOKENIZER_REVISION}\"}}"
            ),
        );
        let scope_path = root.join("scope.json");
        let canonical_path = root.join("canonical.json");
        let reference_path = root.join("reference.json");
        fs::write(&scope_path, &scope).unwrap();
        fs::write(&canonical_path, &canonical).unwrap();
        fs::write(&reference_path, &reference).unwrap();
        let reference_sha = hex_digest(&sha256(reference.as_bytes()));
        let load = |expected_scope_sha: &str, expected_reference_sha: &str| {
            VibeVoiceRealtimeOwnerContract::load(VibeVoiceRealtimeOwnerInputs {
                owner_scope: &scope_path,
                canonical_payload: &canonical_path,
                expected_owner_scope_sha256: expected_scope_sha,
                expected_canonical_sha256: &canonical_sha,
                reference_json: &reference_path,
                expected_reference_sha256: expected_reference_sha,
                expected_vokra_head: &head,
                expected_vokra_tree_sha1: &tree,
                measured_text_sha256: &text,
                expected_reference_script_sha256: &script,
                expected_uv_lock_sha256: &lock,
                expected_trusted_runner_sha256: &runner,
            })
        };
        assert_eq!(
            load(&scope_sha, &reference_sha)
                .unwrap()
                .execution
                .max_new_tokens,
            1
        );

        let malformed_search = format!("\"bytes\":7031645,\"path\":{tokenizer_json_path}");
        let malformed_replacement = format!("\"bytes\":7031644,\"path\":{tokenizer_json_path}");
        assert!(reference.contains(&malformed_search));
        let malformed_reference = reference.replace(&malformed_search, &malformed_replacement);
        assert_ne!(malformed_reference, reference);
        let malformed_reference_sha = hex_digest(&sha256(malformed_reference.as_bytes()));
        fs::write(&reference_path, malformed_reference).unwrap();
        assert!(load(&scope_sha, &malformed_reference_sha).is_err());
        fs::write(&reference_path, &reference).unwrap();
        let relative_search = format!("\"path\":{uv_lock_path}");
        let relative_replacement = format!("\"path\":{}", json_string("synthetic/uv.lock"));
        assert!(reference.contains(&relative_search));
        let relative_identity_reference =
            reference.replace(&relative_search, &relative_replacement);
        assert_ne!(relative_identity_reference, reference);
        let relative_identity_sha = hex_digest(&sha256(relative_identity_reference.as_bytes()));
        fs::write(&reference_path, relative_identity_reference).unwrap();
        assert!(load(&scope_sha, &relative_identity_sha).is_err());
        fs::write(&reference_path, &reference).unwrap();

        let rebound = scope.replace("\"cfg_scale\":3.0", "\"cfg_scale\":3");
        let rebound_scope_sha = hex_digest(&sha256(rebound.as_bytes()));
        let rebound_reference = reference.replace(
            &format!("\"owner_scope_file_sha256\":\"{scope_sha}\""),
            &format!("\"owner_scope_file_sha256\":\"{rebound_scope_sha}\""),
        );
        let rebound_reference_sha = hex_digest(&sha256(rebound_reference.as_bytes()));
        fs::write(&scope_path, rebound).unwrap();
        fs::write(&reference_path, rebound_reference).unwrap();
        let error = load(&rebound_scope_sha, &rebound_reference_sha).unwrap_err();
        assert!(
            error.contains("canonical payload"),
            "number-kind mismatch must reach AST validation, got {error}"
        );
        fs::remove_dir_all(root).unwrap();
    }
}
