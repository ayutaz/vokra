//! VAST-only consumer for the official VibeVoice Realtime streaming packet.
//!
//! The packet is produced by Microsoft's pinned implementation and contains
//! observational stage/cache traces. This test authenticates packet identity
//! and trace files before loading the native GGUF, preset, tokenizer, and text
//! inputs, then feeds the recorded diffusion noise tape to the native
//! composition. Full-waveform tolerances remain OPEN until independently
//! registered evidence supplies a bound.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use vokra_core::backend::BackendKind;
use vokra_core::json::{JsonValue, parse as parse_json};
use vokra_models::vibevoice_streaming::runtime::{
    VibeVoiceRealtimeDiagnosticBranch, VibeVoiceRealtimeDiagnosticCall,
    VibeVoiceRealtimeDiagnosticEvent, VibeVoiceRealtimeDiagnosticObserver,
};
use vokra_models::vibevoice_streaming::tokenizer::VibeVoiceRealtimeTokenizer;
use vokra_models::vibevoice_streaming::{
    VibeVoiceRealtimeGenerationStopReason, VibeVoiceRealtimePresetBranch,
    VibeVoiceRealtimePresetCache, VibeVoiceRealtimeRuntime, VibeVoiceRealtimeSynthesisConfig,
    VibeVoiceRealtimeSynthesisStep,
};

const FORMAT: &str = "vokra-vibevoice-realtime-streaming-reference-v1";
const OPEN_STATUS: &str = "REFERENCE_RUN_OPEN_NOT_RUST_PARITY";
const SOURCE_REVISION: &str = "94da20d98b2fa7688e9cbfaf7692ddb4954f7600";
const CHECKPOINT_SHA256: &str = "7758b150b8139deb48ac1ff6f181f745c8fedd5511232fd974b3eb217d83b514";
const PRESET_PAYLOAD_SHA256: &str =
    "a7bfdf1cd4939c22469bcfc6f427ae9c4467b3df46c2c14303a39c294cfc6897";
const INFERENCE_STEPS: usize = 20;
const LATENT_WIDTH: usize = 64;
const AUDIO_CHUNK_SAMPLES: usize = 3_200;

fn required_path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} must point to an authenticated VAST artifact"))
}

fn required_string(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required for the authenticated run"))
}

fn sha256_file(path: &Path) -> String {
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .unwrap_or_else(|error| panic!("sha256sum {}: {error}", path.display()));
    assert!(
        output.status.success(),
        "sha256sum {} failed: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    let hash = String::from_utf8(output.stdout)
        .unwrap_or_else(|error| panic!("sha256sum output is not UTF-8: {error}"))
        .split_whitespace()
        .next()
        .unwrap_or_else(|| panic!("sha256sum returned no digest for {}", path.display()))
        .to_owned();
    assert!(
        hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "sha256sum returned a malformed digest for {}",
        path.display()
    );
    hash
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut child = Command::new("sha256sum")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap_or_else(|error| panic!("sha256sum stdin: {error}"));
    child
        .stdin
        .take()
        .expect("sha256sum stdin pipe")
        .write_all(bytes)
        .expect("write SHA-256 payload");
    let output = child.wait_with_output().expect("wait for sha256sum stdin");
    assert!(
        output.status.success(),
        "sha256sum stdin failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let hash = String::from_utf8(output.stdout)
        .expect("sha256sum stdin output is not UTF-8")
        .split_whitespace()
        .next()
        .expect("sha256sum stdin returned no digest")
        .to_owned();
    assert!(
        hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "sha256sum stdin returned a malformed digest"
    );
    hash
}

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

fn object<'a>(value: &'a JsonValue, label: &str) -> &'a [(String, JsonValue)] {
    value
        .as_object()
        .unwrap_or_else(|| panic!("{label} must be a JSON object"))
}

fn field<'a>(value: &'a JsonValue, name: &str) -> &'a JsonValue {
    value
        .get(name)
        .unwrap_or_else(|| panic!("missing JSON field {name:?}"))
}

fn string_field(value: &JsonValue, name: &str) -> String {
    field(value, name)
        .as_str()
        .unwrap_or_else(|| panic!("JSON field {name:?} must be a string"))
        .to_owned()
}

fn usize_field(value: &JsonValue, name: &str) -> usize {
    let number = field(value, name)
        .as_u64()
        .unwrap_or_else(|| panic!("JSON field {name:?} must be a non-negative integer"));
    usize::try_from(number).unwrap_or_else(|_| panic!("JSON field {name:?} overflows usize"))
}

fn f64_field(value: &JsonValue, name: &str) -> f64 {
    match field(value, name) {
        JsonValue::Float(number) => *number,
        JsonValue::Int(number) => *number as f64,
        _ => panic!("JSON field {name:?} must be a finite number"),
    }
}

fn bool_field(value: &JsonValue, name: &str) -> bool {
    match field(value, name) {
        JsonValue::Bool(value) => *value,
        _ => panic!("JSON field {name:?} must be a boolean"),
    }
}

fn array_field<'a>(value: &'a JsonValue, name: &str) -> &'a [JsonValue] {
    field(value, name)
        .as_array()
        .unwrap_or_else(|| panic!("JSON field {name:?} must be an array"))
}

fn reject_duplicate_keys(value: &JsonValue, path: &str) {
    match value {
        JsonValue::Object(entries) => {
            let mut keys = BTreeSet::new();
            for (key, child) in entries {
                assert!(
                    keys.insert(key),
                    "reference packet contains duplicate JSON key {path}.{key}"
                );
                reject_duplicate_keys(child, &format!("{path}.{key}"));
            }
        }
        JsonValue::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                reject_duplicate_keys(child, &format!("{path}[{index}]"));
            }
        }
        _ => {}
    }
}

#[derive(Debug)]
struct NpyF32 {
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn read_npy_f32(path: &Path) -> NpyF32 {
    let bytes = read(path);
    assert!(
        bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY",
        "{} is not NPY",
        path.display()
    );
    let version = (bytes[6], bytes[7]);
    let (header_start, header_len) = match version {
        (1, 0) => (10usize, u16::from_le_bytes([bytes[8], bytes[9]]) as usize),
        (2, 0) | (3, 0) => {
            assert!(
                bytes.len() >= 12,
                "{} has a truncated NPY header",
                path.display()
            );
            (
                12usize,
                u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            )
        }
        other => panic!("{} uses unsupported NPY version {other:?}", path.display()),
    };
    let header_end = header_start
        .checked_add(header_len)
        .unwrap_or_else(|| panic!("{} NPY header overflows", path.display()));
    assert!(
        header_end <= bytes.len(),
        "{} has a truncated NPY header",
        path.display()
    );
    let header = std::str::from_utf8(&bytes[header_start..header_end])
        .unwrap_or_else(|error| panic!("{} NPY header is not UTF-8: {error}", path.display()));
    assert!(
        header.contains("'descr': '<f4'") || header.contains("\"descr\": \"<f4\""),
        "{} is not little-endian f32",
        path.display()
    );
    assert!(
        header.contains("'fortran_order': False") || header.contains("\"fortran_order\": false"),
        "{} is not C-contiguous",
        path.display()
    );
    let marker = header
        .find("'shape': (")
        .or_else(|| header.find("\"shape\": ("))
        .unwrap_or_else(|| panic!("{} has no NPY shape", path.display()));
    let shape_start = marker + header[marker..].find('(').unwrap() + 1;
    let shape_end = shape_start
        + header[shape_start..]
            .find(')')
            .unwrap_or_else(|| panic!("{} has an invalid NPY shape", path.display()));
    let shape: Vec<usize> = header[shape_start..shape_end]
        .split(',')
        .filter_map(|item| {
            let item = item.trim();
            (!item.is_empty()).then(|| {
                item.parse::<usize>().unwrap_or_else(|error| {
                    panic!("{} has invalid NPY shape: {error}", path.display())
                })
            })
        })
        .collect();
    assert!(
        !shape.is_empty(),
        "{} has an empty NPY shape",
        path.display()
    );
    let elements = shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .unwrap_or_else(|| panic!("{} NPY shape overflows", path.display()));
    assert_eq!(
        elements.checked_mul(4),
        Some(bytes.len() - header_end),
        "{} NPY shape/data mismatch",
        path.display()
    );
    let values: Vec<f32> = bytes[header_end..]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    assert!(
        values.iter().all(|value| value.is_finite()),
        "{} contains non-finite f32 values",
        path.display()
    );
    NpyF32 { shape, values }
}

#[derive(Debug, Clone)]
struct TensorRecord {
    file: String,
    shape: Vec<usize>,
    sha256: String,
}

#[derive(Debug, Clone)]
struct TraceEvent {
    stage: String,
    ordinal: usize,
    tensor: Option<TensorRecord>,
    tensor_values: Option<Vec<f32>>,
    cache_layers: Option<usize>,
    cache_lengths: Option<Vec<usize>>,
}

#[derive(Debug, Clone)]
enum NativeDiagnosticEvent {
    Hidden {
        branch: VibeVoiceRealtimeDiagnosticBranch,
        call: VibeVoiceRealtimeDiagnosticCall,
        text_window_index: Option<usize>,
        text_token_index: Option<usize>,
        text_window_tokens: usize,
        speech_step: Option<usize>,
        values: Vec<f32>,
    },
    CachePosition {
        branch: VibeVoiceRealtimeDiagnosticBranch,
        call: VibeVoiceRealtimeDiagnosticCall,
        text_window_index: Option<usize>,
        text_token_index: Option<usize>,
        text_window_tokens: usize,
        speech_step: Option<usize>,
        position: usize,
        layers: usize,
    },
    DiffusionPrediction {
        speech_step: usize,
        diffusion_step: usize,
        timestep: usize,
        conditional: Vec<f32>,
        unconditional: Vec<f32>,
    },
    SampledLatent {
        speech_step: usize,
        values: Vec<f32>,
    },
    DecoderInput {
        speech_step: usize,
        scaled: Vec<f32>,
        unscaled: Vec<f32>,
    },
    DecoderChunk {
        speech_step: usize,
        pcm: Vec<f32>,
    },
    Connector {
        speech_step: usize,
        input: Vec<f32>,
        output: Vec<f32>,
    },
    Eos {
        branch: VibeVoiceRealtimeDiagnosticBranch,
        call: VibeVoiceRealtimeDiagnosticCall,
        text_window_index: Option<usize>,
        text_token_index: Option<usize>,
        text_window_tokens: usize,
        speech_step: Option<usize>,
        classifier_pass: usize,
        value: f32,
    },
}

#[derive(Default)]
struct NativeDiagnosticTrace {
    events: Vec<NativeDiagnosticEvent>,
}

impl VibeVoiceRealtimeDiagnosticObserver for NativeDiagnosticTrace {
    fn observe(&mut self, event: VibeVoiceRealtimeDiagnosticEvent<'_>) -> vokra_core::Result<()> {
        self.events.push(match event {
            VibeVoiceRealtimeDiagnosticEvent::Hidden {
                branch,
                call,
                text_window_index,
                text_token_index,
                text_window_tokens,
                speech_step,
                values,
            } => NativeDiagnosticEvent::Hidden {
                branch,
                call,
                text_window_index,
                text_token_index,
                text_window_tokens,
                speech_step,
                values: values.to_vec(),
            },
            VibeVoiceRealtimeDiagnosticEvent::CachePosition {
                branch,
                call,
                text_window_index,
                text_token_index,
                text_window_tokens,
                speech_step,
                position,
                layers,
            } => NativeDiagnosticEvent::CachePosition {
                branch,
                call,
                text_window_index,
                text_token_index,
                text_window_tokens,
                speech_step,
                position,
                layers,
            },
            VibeVoiceRealtimeDiagnosticEvent::DiffusionPrediction {
                speech_step,
                diffusion_step,
                timestep,
                conditional,
                unconditional,
            } => NativeDiagnosticEvent::DiffusionPrediction {
                speech_step,
                diffusion_step,
                timestep,
                conditional: conditional.to_vec(),
                unconditional: unconditional.to_vec(),
            },
            VibeVoiceRealtimeDiagnosticEvent::SampledLatent {
                speech_step,
                values,
            } => NativeDiagnosticEvent::SampledLatent {
                speech_step,
                values: values.to_vec(),
            },
            VibeVoiceRealtimeDiagnosticEvent::DecoderInput {
                speech_step,
                scaled,
                unscaled,
            } => NativeDiagnosticEvent::DecoderInput {
                speech_step,
                scaled: scaled.to_vec(),
                unscaled: unscaled.to_vec(),
            },
            VibeVoiceRealtimeDiagnosticEvent::DecoderChunk { speech_step, pcm } => {
                NativeDiagnosticEvent::DecoderChunk {
                    speech_step,
                    pcm: pcm.to_vec(),
                }
            }
            VibeVoiceRealtimeDiagnosticEvent::Connector {
                speech_step,
                input,
                output,
            } => NativeDiagnosticEvent::Connector {
                speech_step,
                input: input.to_vec(),
                output: output.to_vec(),
            },
            VibeVoiceRealtimeDiagnosticEvent::Eos {
                branch,
                call,
                text_window_index,
                text_token_index,
                text_window_tokens,
                speech_step,
                classifier_pass,
                value,
            } => NativeDiagnosticEvent::Eos {
                branch,
                call,
                text_window_index,
                text_token_index,
                text_window_tokens,
                speech_step,
                classifier_pass,
                value,
            },
        });
        Ok(())
    }
}

fn tensor_record(value: &JsonValue, label: &str) -> TensorRecord {
    let file = string_field(value, "file");
    assert!(!file.is_empty() && !file.contains('/') && !file.contains('\\'));
    let shape = array_field(value, "shape")
        .iter()
        .map(|item| {
            usize::try_from(item.as_u64().expect("tensor shape must be integer"))
                .expect("tensor shape overflows usize")
        })
        .collect();
    assert_eq!(string_field(value, "dtype"), "float32", "{label} dtype");
    let sha256 = string_field(value, "sha256");
    assert_eq!(parse_hex32(&sha256, label), sha256.to_ascii_lowercase());
    assert!(bool_field(value, "finite"), "{label} finite marker");
    TensorRecord {
        file,
        shape,
        sha256,
    }
}

fn trace_events(reference_dir: &Path, cpu: &JsonValue) -> Vec<TraceEvent> {
    let mut events = Vec::new();
    let mut ordinals = BTreeMap::<String, usize>::new();
    for (index, value) in array_field(cpu, "traces").iter().enumerate() {
        let stage = string_field(value, "stage");
        let ordinal = usize_field(value, "ordinal");
        let expected = ordinals.entry(stage.clone()).or_default();
        let expected_ordinal = *expected;
        *expected += 1;
        assert_eq!(
            ordinal, expected_ordinal,
            "trace ordinal gap at {stage} index {index}"
        );
        let has_tensor = object(value, "trace record")
            .iter()
            .any(|(key, _)| key == "tensor");
        let has_cache = object(value, "trace record")
            .iter()
            .any(|(key, _)| key == "cache_lengths");
        assert_ne!(
            has_tensor, has_cache,
            "trace record must be tensor or cache: {stage}"
        );
        let (tensor, tensor_values) = if has_tensor {
            let record = tensor_record(field(value, "tensor"), &stage);
            let path = reference_dir.join("cpu").join(&record.file);
            assert_eq!(sha256_file(&path), record.sha256, "trace hash {stage}");
            let npy = read_npy_f32(&path);
            assert_eq!(npy.shape, record.shape, "trace shape {stage}");
            (Some(record), Some(npy.values))
        } else {
            (None, None)
        };
        let (cache_layers, cache_lengths) = if has_cache {
            let lengths = array_field(value, "cache_lengths")
                .iter()
                .map(|item| {
                    usize::try_from(item.as_u64().expect("cache length must be integer"))
                        .expect("cache length overflows usize")
                })
                .collect::<Vec<_>>();
            (Some(usize_field(value, "cache_layers")), Some(lengths))
        } else {
            (None, None)
        };
        events.push(TraceEvent {
            stage,
            ordinal,
            tensor,
            tensor_values,
            cache_layers,
            cache_lengths,
        });
    }
    events
}

fn count_stage(events: &[TraceEvent], stage: &str) -> usize {
    events.iter().filter(|event| event.stage == stage).count()
}

fn event_for<'a>(events: &'a [TraceEvent], stage: &str, ordinal: usize) -> &'a TraceEvent {
    events
        .iter()
        .find(|event| event.stage == stage && event.ordinal == ordinal)
        .unwrap_or_else(|| panic!("reference trace missing {stage}/{ordinal}"))
}

fn parse_hex32(value: &str, label: &str) -> String {
    assert!(
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{label} must be a 64-character hexadecimal SHA-256"
    );
    value.to_ascii_lowercase()
}

fn parse_git_hex40(value: &str, label: &str) -> String {
    assert!(
        value.len() == 40
            && value
                .bytes()
                .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f')),
        "{label} must be a 40-character hexadecimal Git identity"
    );
    value.to_owned()
}

/// Bind the packet's runtime identity to the authenticated owner scope.
///
/// The reference producer records both values in `runtime`; checking only one
/// field would leave packet/owner-scope identity disagreement undetected. This
/// helper is deliberately model-free so it can be exercised without loading
/// any native artifact.
fn validate_runtime_identity(packet: &JsonValue, scope: &JsonValue, expected_head: &str) {
    let expected_head = parse_git_hex40(expected_head, "expected Vokra HEAD");
    let packet_runtime = field(packet, "runtime");
    let scope_runtime = field(scope, "runtime");
    let packet_head = parse_git_hex40(
        &string_field(packet_runtime, "vokra_head"),
        "reference packet runtime.vokra_head",
    );
    let packet_tree = parse_git_hex40(
        &string_field(packet_runtime, "vokra_tree_sha1"),
        "reference packet runtime.vokra_tree_sha1",
    );
    let scope_head = parse_git_hex40(
        &string_field(scope_runtime, "vokra_head"),
        "owner scope runtime.vokra_head",
    );
    let scope_tree = parse_git_hex40(
        &string_field(scope_runtime, "vokra_tree_sha1"),
        "owner scope runtime.vokra_tree_sha1",
    );
    assert_eq!(
        packet_head, expected_head,
        "reference packet runtime HEAD differs from externally expected HEAD"
    );
    assert_eq!(
        scope_head, expected_head,
        "owner scope runtime HEAD differs from externally expected HEAD"
    );
    assert_eq!(
        packet_tree, scope_tree,
        "reference packet and owner scope runtime tree identities differ"
    );
}

fn validate_packet_identity<'a>(
    reference_dir: &Path,
    packet: &'a JsonValue,
    input_text: &Path,
) -> &'a JsonValue {
    assert_eq!(string_field(packet, "format"), FORMAT);
    assert_eq!(string_field(packet, "status"), OPEN_STATUS);
    assert_eq!(string_field(packet, "publication"), "NO_UPLOAD");
    let source = field(packet, "source");
    assert_eq!(string_field(source, "repository"), "microsoft/VibeVoice");
    assert_eq!(string_field(source, "revision"), SOURCE_REVISION);
    assert_eq!(
        string_field(source, "origin"),
        "https://github.com/microsoft/VibeVoice.git"
    );
    let checkpoint = field(packet, "checkpoint");
    assert_eq!(usize_field(checkpoint, "bytes"), 2_035_332_888);
    assert_eq!(string_field(checkpoint, "sha256"), CHECKPOINT_SHA256);
    let preset = field(packet, "preset");
    assert_eq!(
        string_field(preset, "payload_sha256"),
        PRESET_PAYLOAD_SHA256
    );
    let runtime = field(packet, "runtime");
    assert_eq!(
        string_field(runtime, "vokra_head"),
        required_string("VOKRA_VIBEVOICE_REALTIME_EXPECTED_VOKRA_HEAD")
    );
    let input_sha = string_field(field(packet, "input"), "text_sha256");
    assert_eq!(
        sha256_file(input_text),
        input_sha,
        "authenticated input text hash"
    );
    let cpu = field(field(packet, "packets"), "cpu");
    let packet_sha = parse_hex32(
        &required_string("VOKRA_VIBEVOICE_REALTIME_REFERENCE_SHA256"),
        "reference packet SHA-256",
    );
    assert_eq!(
        sha256_file(&reference_dir.join("reference.json")),
        packet_sha
    );
    cpu
}

fn execution_contract(
    packet: &JsonValue,
    input_text: &Path,
    speech_count: usize,
) -> (f32, usize, usize) {
    let scope_path = required_path("VOKRA_VIBEVOICE_REALTIME_OWNER_SCOPE");
    let expected_file_sha = parse_hex32(
        &required_string("VOKRA_VIBEVOICE_REALTIME_OWNER_SCOPE_SHA256"),
        "owner scope file SHA-256",
    );
    assert_eq!(
        sha256_file(&scope_path),
        expected_file_sha,
        "authenticated owner scope file"
    );
    let scope_bytes = read(&scope_path);
    let scope = parse_json(&scope_bytes).expect("owner scope must be valid JSON");
    reject_duplicate_keys(&scope, "owner_scope");
    assert_eq!(
        string_field(&scope, "schema"),
        "vokra-vibevoice-realtime-streaming-execution-v1"
    );
    let expected_scope_sha = parse_hex32(
        &required_string("VOKRA_VIBEVOICE_REALTIME_OWNER_SCOPE_CANONICAL_SHA256"),
        "owner scope canonical SHA-256",
    );
    assert_eq!(
        string_field(&scope, "scope_sha256"),
        expected_scope_sha,
        "owner scope contents must match the externally reviewed canonical digest"
    );
    assert_eq!(
        string_field(packet, "owner_scope_sha256"),
        expected_scope_sha,
        "reference packet owner scope binding"
    );
    assert_eq!(
        string_field(packet, "owner_scope_file_sha256"),
        expected_file_sha,
        "reference packet owner scope file binding"
    );
    validate_runtime_identity(
        packet,
        &scope,
        &required_string("VOKRA_VIBEVOICE_REALTIME_EXPECTED_VOKRA_HEAD"),
    );
    assert_eq!(
        string_field(&scope, "publication"),
        "NO_UPLOAD",
        "owner scope publication"
    );
    assert!(
        matches!(
            string_field(&scope, "decision").as_str(),
            "APPROVE_COMMERCIAL_EXECUTION" | "APPROVE_RESEARCH_ONLY_EXECUTION"
        ),
        "owner scope execution decision"
    );
    assert_eq!(
        string_field(field(&scope, "source"), "repository"),
        "microsoft/VibeVoice"
    );
    assert_eq!(
        string_field(field(&scope, "source"), "revision"),
        SOURCE_REVISION
    );
    assert_eq!(
        string_field(field(&scope, "checkpoint"), "sha256"),
        string_field(field(packet, "checkpoint"), "sha256"),
        "owner scope/reference checkpoint binding"
    );
    assert_eq!(
        string_field(field(&scope, "preset"), "payload_sha256"),
        string_field(field(packet, "preset"), "payload_sha256"),
        "owner scope/reference preset binding"
    );
    assert_eq!(
        string_field(field(&scope, "tokenizer"), "repository"),
        string_field(field(packet, "tokenizer"), "repository"),
        "owner scope/reference tokenizer binding"
    );
    assert_eq!(
        string_field(field(&scope, "tokenizer"), "revision"),
        string_field(field(packet, "tokenizer"), "revision"),
        "owner scope/reference tokenizer revision binding"
    );
    let input = field(&scope, "input");
    assert_eq!(
        string_field(input, "text_sha256"),
        string_field(field(packet, "input"), "text_sha256"),
        "owner scope/reference text binding"
    );
    assert_eq!(
        sha256_file(input_text),
        string_field(input, "text_sha256"),
        "owner scope/input text hash"
    );
    assert_eq!(usize_field(input, "sample_rate"), 24_000);
    let execution = field(&scope, "execution");
    assert_eq!(string_field(execution, "model_forward"), "APPROVED");
    assert_eq!(string_field(execution, "audio_generation"), "APPROVED");
    assert_eq!(
        usize_field(execution, "ddpm_steps"),
        INFERENCE_STEPS,
        "native/reference diffusion step count"
    );
    let guidance_scale = f64_field(execution, "cfg_scale") as f32;
    assert!(
        guidance_scale.is_finite() && guidance_scale >= 0.0,
        "owner scope cfg_scale must be finite and non-negative"
    );
    let max_new_tokens = usize_field(execution, "max_new_tokens");
    assert!(
        (1..=64).contains(&max_new_tokens),
        "owner scope max_new_tokens must be in 1..=64"
    );
    // This is a native caller budget, not an official VibeVoice setting. It
    // must be supplied separately and may only bound the authenticated tape;
    // EOS is allowed to stop earlier and must never be used to infer it.
    let max_speech_steps = required_string("VOKRA_VIBEVOICE_REALTIME_MAX_SPEECH_STEPS")
        .parse::<usize>()
        .expect("native max speech steps must be usize");
    assert!(
        max_speech_steps >= speech_count,
        "native speech budget must cover the authenticated reference tape"
    );
    (guidance_scale, max_new_tokens, max_speech_steps)
}

fn assert_required_stage_contract(
    events: &[TraceEvent],
    speech_count: usize,
    requested_text_windows: usize,
    max_new_tokens: usize,
) -> bool {
    for stage in [
        "lm.positive.prefill.hidden",
        "lm.positive.prefill.cache",
        "lm.negative.prefill.hidden",
        "lm.negative.prefill.cache",
        "tts.positive.prefill.hidden",
        "tts.positive.prefill.cache",
        "tts.negative.prefill.hidden",
        "tts.negative.prefill.cache",
    ] {
        assert_eq!(count_stage(events, stage), 1, "prefill stage {stage}");
    }
    let observed_text_windows = count_stage(events, "lm.positive.hidden");
    assert!(
        observed_text_windows > 0 && observed_text_windows <= requested_text_windows,
        "unsupported official packet: observed text windows must be a non-empty prefix of the authenticated input plan"
    );
    let positive_tts_count = count_stage(events, "tts.positive.hidden");
    assert!(
        positive_tts_count >= observed_text_windows,
        "unsupported official packet: positive TTS hidden rows must include each text window"
    );
    let tts_update_count = positive_tts_count - observed_text_windows;
    assert!(
        tts_update_count == speech_count || tts_update_count + 1 == speech_count,
        "unsupported official packet: TTS update count must equal sampled speech count or be one terminal max-length chunk short"
    );
    let terminal_max_length_chunk = tts_update_count + 1 == speech_count;
    if terminal_max_length_chunk {
        assert!(
            max_new_tokens > 0,
            "unsupported official packet: terminal max-length chunk requires a positive max_new_tokens"
        );
    }
    assert_eq!(
        count_stage(events, "tts.eos"),
        observed_text_windows + 3 * tts_update_count,
        "official EOS classifier calls must include text windows and positive/negative/stop speech calls"
    );
    for (stage, expected) in [
        ("lm.positive.hidden", observed_text_windows),
        ("lm.positive.cache", observed_text_windows),
        (
            "tts.positive.hidden",
            observed_text_windows + tts_update_count,
        ),
        (
            "tts.positive.cache",
            observed_text_windows + tts_update_count,
        ),
        ("tts.negative.hidden", tts_update_count),
        ("tts.negative.cache", tts_update_count),
        ("speech.sampled_latent", speech_count),
        ("acoustic.decode_input_unscaled", speech_count),
        ("acoustic.decoder_chunk", speech_count),
        ("acoustic.connector.input", speech_count),
        ("acoustic.connector", speech_count),
        ("tts.eos", observed_text_windows + 3 * tts_update_count),
        ("diffusion.prediction", speech_count * INFERENCE_STEPS),
    ] {
        assert_eq!(count_stage(events, stage), expected, "stage count {stage}");
    }
    terminal_max_length_chunk
}

fn cache_position(event: &TraceEvent, stage: &str, expected_layers: usize) -> usize {
    assert_eq!(event.stage, stage);
    assert_eq!(
        event.cache_layers,
        Some(expected_layers),
        "cache layer count {stage}"
    );
    let lengths = event.cache_lengths.as_ref().expect("cache lengths");
    assert_eq!(lengths.len(), expected_layers, "cache layer vector {stage}");
    let first = lengths[0];
    assert!(
        lengths.iter().all(|length| *length == first),
        "cache lengths disagree at {stage}"
    );
    first
}

fn hidden_rows(event: &TraceEvent, label: &str) -> usize {
    let tensor = event
        .tensor
        .as_ref()
        .unwrap_or_else(|| panic!("{label} tensor"));
    assert_eq!(tensor.shape.len(), 3, "{label} rank");
    assert_eq!(tensor.shape[0], 1, "{label} batch");
    assert_eq!(tensor.shape[2], 896, "{label} width");
    tensor.shape[1]
}

fn positive_tts_speech_positions(events: &[TraceEvent], initial: usize) -> Vec<usize> {
    let mut position = initial;
    let mut expect_text = false;
    let mut speech_positions = Vec::new();
    for event in events {
        if event.stage == "lm.positive.cache" {
            expect_text = true;
            continue;
        }
        if event.stage != "tts.positive.cache" {
            continue;
        }
        let hidden = events
            .iter()
            .find(|candidate| {
                candidate.stage == "tts.positive.hidden" && candidate.ordinal == event.ordinal
            })
            .unwrap_or_else(|| panic!("missing positive TTS hidden ordinal {}", event.ordinal));
        let rows = hidden_rows(hidden, "positive TTS");
        position += rows;
        assert_eq!(
            cache_position(event, "tts.positive.cache", 20),
            position,
            "positive TTS cache position"
        );
        if expect_text {
            expect_text = false;
        } else {
            assert_eq!(rows, 1, "speech TTS output must contain one row");
            speech_positions.push(position);
        }
    }
    speech_positions
}

fn negative_tts_cache_positions(events: &[TraceEvent], initial: usize, count: usize) {
    for ordinal in 0..count {
        let cache = events
            .iter()
            .find(|event| event.stage == "tts.negative.cache" && event.ordinal == ordinal)
            .unwrap();
        let hidden = events
            .iter()
            .find(|event| event.stage == "tts.negative.hidden" && event.ordinal == ordinal)
            .unwrap();
        assert_eq!(hidden_rows(hidden, "negative TTS"), 1);
        assert_eq!(
            cache_position(cache, "tts.negative.cache", 20),
            initial + ordinal + 1
        );
    }
}

#[derive(Debug)]
enum NativeDiagnosticRecord {
    Tensor {
        stage: &'static str,
        values: Vec<f32>,
    },
    Cache {
        stage: &'static str,
        position: usize,
        layers: usize,
    },
}

fn diagnostic_hidden(
    events: &[NativeDiagnosticEvent],
    index: &mut usize,
    branch: VibeVoiceRealtimeDiagnosticBranch,
    call: VibeVoiceRealtimeDiagnosticCall,
    text_window_index: Option<usize>,
    text_token_index: Option<usize>,
    text_window_tokens: usize,
    speech_step: Option<usize>,
) -> Vec<f32> {
    let event = events
        .get(*index)
        .unwrap_or_else(|| panic!("native diagnostic ended before hidden event"));
    let values = match event {
        NativeDiagnosticEvent::Hidden {
            branch: actual_branch,
            call: actual_call,
            text_window_index: actual_window,
            text_token_index: actual_token,
            text_window_tokens: actual_tokens,
            speech_step: actual_speech,
            values,
        } => {
            assert_eq!(*actual_branch, branch, "native hidden branch ordering");
            assert_eq!(*actual_call, call, "native hidden call ordering");
            assert_eq!(
                *actual_window, text_window_index,
                "native hidden window metadata"
            );
            assert_eq!(
                *actual_token, text_token_index,
                "native hidden token metadata"
            );
            assert_eq!(
                *actual_tokens, text_window_tokens,
                "native hidden window size"
            );
            assert_eq!(*actual_speech, speech_step, "native hidden speech metadata");
            assert!(values.iter().all(|value| value.is_finite()));
            assert_eq!(values.len() % 896, 0, "native hidden width");
            values.clone()
        }
        other => panic!("expected native hidden event, got {other:?}"),
    };
    *index += 1;
    values
}

fn diagnostic_cache(
    events: &[NativeDiagnosticEvent],
    index: &mut usize,
    branch: VibeVoiceRealtimeDiagnosticBranch,
    call: VibeVoiceRealtimeDiagnosticCall,
    text_window_index: Option<usize>,
    text_token_index: Option<usize>,
    text_window_tokens: usize,
    speech_step: Option<usize>,
) -> (usize, usize) {
    let event = events
        .get(*index)
        .unwrap_or_else(|| panic!("native diagnostic ended before cache event"));
    let result = match event {
        NativeDiagnosticEvent::CachePosition {
            branch: actual_branch,
            call: actual_call,
            text_window_index: actual_window,
            text_token_index: actual_token,
            text_window_tokens: actual_tokens,
            speech_step: actual_speech,
            position,
            layers,
        } => {
            assert_eq!(*actual_branch, branch, "native cache branch ordering");
            assert_eq!(*actual_call, call, "native cache call ordering");
            assert_eq!(
                *actual_window, text_window_index,
                "native cache window metadata"
            );
            assert_eq!(
                *actual_token, text_token_index,
                "native cache token metadata"
            );
            assert_eq!(
                *actual_tokens, text_window_tokens,
                "native cache window size"
            );
            assert_eq!(*actual_speech, speech_step, "native cache speech metadata");
            assert!(*layers > 0, "native cache must expose layers");
            (*position, *layers)
        }
        other => panic!("expected native cache event, got {other:?}"),
    };
    *index += 1;
    result
}

fn diagnostic_eos(
    events: &[NativeDiagnosticEvent],
    index: &mut usize,
    branch: VibeVoiceRealtimeDiagnosticBranch,
    call: VibeVoiceRealtimeDiagnosticCall,
    text_window_index: Option<usize>,
    text_token_index: Option<usize>,
    text_window_tokens: usize,
    speech_step: Option<usize>,
    classifier_pass: usize,
) -> f32 {
    let event = events
        .get(*index)
        .unwrap_or_else(|| panic!("native diagnostic ended before EOS event"));
    let value = match event {
        NativeDiagnosticEvent::Eos {
            branch: actual_branch,
            call: actual_call,
            text_window_index: actual_window,
            text_token_index: actual_token,
            text_window_tokens: actual_tokens,
            speech_step: actual_speech,
            classifier_pass: actual_pass,
            value,
        } => {
            assert_eq!(*actual_branch, branch, "native EOS branch ordering");
            assert_eq!(*actual_call, call, "native EOS call ordering");
            assert_eq!(
                *actual_window, text_window_index,
                "native EOS window metadata"
            );
            assert_eq!(*actual_token, text_token_index, "native EOS token metadata");
            assert_eq!(*actual_tokens, text_window_tokens, "native EOS window size");
            assert_eq!(*actual_speech, speech_step, "native EOS speech metadata");
            assert_eq!(*actual_pass, classifier_pass, "native EOS classifier pass");
            assert!(value.is_finite(), "native EOS must be finite");
            *value
        }
        other => panic!("expected native EOS event, got {other:?}"),
    };
    *index += 1;
    value
}

fn native_diagnostic_records(events: &[NativeDiagnosticEvent]) -> Vec<NativeDiagnosticRecord> {
    let mut index = 0;
    let mut records = Vec::new();
    for (branch, hidden_stage, cache_stage) in [
        (
            VibeVoiceRealtimeDiagnosticBranch::PositiveLm,
            "lm.positive.prefill.hidden",
            "lm.positive.prefill.cache",
        ),
        (
            VibeVoiceRealtimeDiagnosticBranch::NegativeLm,
            "lm.negative.prefill.hidden",
            "lm.negative.prefill.cache",
        ),
        (
            VibeVoiceRealtimeDiagnosticBranch::PositiveTts,
            "tts.positive.prefill.hidden",
            "tts.positive.prefill.cache",
        ),
        (
            VibeVoiceRealtimeDiagnosticBranch::NegativeTts,
            "tts.negative.prefill.hidden",
            "tts.negative.prefill.cache",
        ),
    ] {
        records.push(NativeDiagnosticRecord::Tensor {
            stage: hidden_stage,
            values: diagnostic_hidden(
                events,
                &mut index,
                branch,
                VibeVoiceRealtimeDiagnosticCall::Prefill,
                None,
                None,
                0,
                None,
            ),
        });
        let (position, layers) = diagnostic_cache(
            events,
            &mut index,
            branch,
            VibeVoiceRealtimeDiagnosticCall::Prefill,
            None,
            None,
            0,
            None,
        );
        records.push(NativeDiagnosticRecord::Cache {
            stage: cache_stage,
            position,
            layers,
        });
    }

    let mut expected_window = 0;
    let mut speech_step = 0;
    loop {
        if matches!(
            events.get(index),
            Some(NativeDiagnosticEvent::Hidden {
                branch: VibeVoiceRealtimeDiagnosticBranch::PositiveLm,
                call: VibeVoiceRealtimeDiagnosticCall::Text,
                ..
            })
        ) {
            let (window, window_tokens) = match &events[index] {
                NativeDiagnosticEvent::Hidden {
                    text_window_index: Some(window),
                    text_token_index: Some(0),
                    text_window_tokens,
                    ..
                } => (*window, *text_window_tokens),
                event => panic!("invalid first text-window event {event:?}"),
            };
            assert_eq!(window, expected_window, "native text window ordering");
            assert!(window_tokens > 0, "native text window must not be empty");
            let mut lm_values = Vec::new();
            let mut tts_values = Vec::new();
            let mut lm_position = None;
            let mut tts_position = None;
            let mut lm_layers = None;
            let mut tts_layers = None;
            let mut text_eos_values = Vec::new();
            for token in 0..window_tokens {
                lm_values.extend(diagnostic_hidden(
                    events,
                    &mut index,
                    VibeVoiceRealtimeDiagnosticBranch::PositiveLm,
                    VibeVoiceRealtimeDiagnosticCall::Text,
                    Some(window),
                    Some(token),
                    window_tokens,
                    None,
                ));
                let (position, layers) = diagnostic_cache(
                    events,
                    &mut index,
                    VibeVoiceRealtimeDiagnosticBranch::PositiveLm,
                    VibeVoiceRealtimeDiagnosticCall::Text,
                    Some(window),
                    Some(token),
                    window_tokens,
                    None,
                );
                if let Some(previous) = lm_position {
                    assert_eq!(position, previous + 1, "native text cache progression");
                }
                lm_position = Some(position);
                lm_layers = Some(layers);
                text_eos_values.push(diagnostic_eos(
                    events,
                    &mut index,
                    VibeVoiceRealtimeDiagnosticBranch::PositiveTts,
                    VibeVoiceRealtimeDiagnosticCall::Text,
                    Some(window),
                    Some(token),
                    window_tokens,
                    None,
                    0,
                ));
                tts_values.extend(diagnostic_hidden(
                    events,
                    &mut index,
                    VibeVoiceRealtimeDiagnosticBranch::PositiveTts,
                    VibeVoiceRealtimeDiagnosticCall::Text,
                    Some(window),
                    Some(token),
                    window_tokens,
                    None,
                ));
                let (position, layers) = diagnostic_cache(
                    events,
                    &mut index,
                    VibeVoiceRealtimeDiagnosticBranch::PositiveTts,
                    VibeVoiceRealtimeDiagnosticCall::Text,
                    Some(window),
                    Some(token),
                    window_tokens,
                    None,
                );
                if let Some(previous) = tts_position {
                    assert_eq!(position, previous + 1, "native TTS text cache progression");
                }
                tts_position = Some(position);
                tts_layers = Some(layers);
            }
            records.push(NativeDiagnosticRecord::Tensor {
                stage: "lm.positive.hidden",
                values: lm_values,
            });
            records.push(NativeDiagnosticRecord::Cache {
                stage: "lm.positive.cache",
                position: lm_position.unwrap(),
                layers: lm_layers.unwrap(),
            });
            // Native token steps run the EOS head once per token; the official
            // window call records only its final-row classifier output. Retain and
            // validate every native call above, then map the final call explicitly
            // to the authenticated official window record.
            records.push(NativeDiagnosticRecord::Tensor {
                stage: "tts.eos",
                values: vec![*text_eos_values.last().unwrap()],
            });
            records.push(NativeDiagnosticRecord::Tensor {
                stage: "tts.positive.hidden",
                values: tts_values,
            });
            records.push(NativeDiagnosticRecord::Cache {
                stage: "tts.positive.cache",
                position: tts_position.unwrap(),
                layers: tts_layers.unwrap(),
            });
            expected_window += 1;
        }
        if !matches!(
            events.get(index),
            Some(NativeDiagnosticEvent::DiffusionPrediction { .. })
        ) {
            break;
        }
        let mut previous_timestep = None;
        for diffusion_step in 0..INFERENCE_STEPS {
            let event = events
                .get(index)
                .unwrap_or_else(|| panic!("native diffusion trace ended early"));
            let (actual_speech, actual_step, timestep, conditional, unconditional) = match event {
                NativeDiagnosticEvent::DiffusionPrediction {
                    speech_step,
                    diffusion_step,
                    timestep,
                    conditional,
                    unconditional,
                } => (
                    *speech_step,
                    *diffusion_step,
                    *timestep,
                    conditional,
                    unconditional,
                ),
                other => panic!("expected diffusion event, got {other:?}"),
            };
            assert_eq!(actual_speech, speech_step);
            assert_eq!(actual_step, diffusion_step);
            assert!(conditional.iter().all(|value| value.is_finite()));
            assert!(unconditional.iter().all(|value| value.is_finite()));
            assert_eq!(
                conditional.len(),
                LATENT_WIDTH,
                "native conditional prediction width"
            );
            assert_eq!(
                unconditional.len(),
                LATENT_WIDTH,
                "native unconditional prediction width"
            );
            if let Some(previous) = previous_timestep {
                assert!(
                    timestep < previous,
                    "native diffusion timesteps must descend"
                );
            }
            previous_timestep = Some(timestep);
            let mut values = conditional.clone();
            values.extend(unconditional);
            records.push(NativeDiagnosticRecord::Tensor {
                stage: "diffusion.prediction",
                values,
            });
            index += 1;
        }
        let event = events
            .get(index)
            .unwrap_or_else(|| panic!("native diffusion trace has no sampled latent"));
        match event {
            NativeDiagnosticEvent::SampledLatent {
                speech_step: actual_speech,
                values,
            } => {
                assert_eq!(*actual_speech, speech_step);
                assert_eq!(values.len(), LATENT_WIDTH, "native sampled latent width");
                records.push(NativeDiagnosticRecord::Tensor {
                    stage: "speech.sampled_latent",
                    values: values.clone(),
                });
            }
            other => panic!("expected sampled latent event, got {other:?}"),
        }
        index += 1;
        let event = events
            .get(index)
            .unwrap_or_else(|| panic!("native trace has no decoder input"));
        match event {
            NativeDiagnosticEvent::DecoderInput {
                speech_step: actual_speech,
                scaled,
                unscaled,
            } => {
                assert_eq!(*actual_speech, speech_step);
                assert_eq!(scaled.len(), LATENT_WIDTH, "native scaled latent width");
                assert_eq!(unscaled.len(), LATENT_WIDTH, "native unscaled latent width");
                records.push(NativeDiagnosticRecord::Tensor {
                    stage: "acoustic.decode_input_unscaled",
                    values: unscaled.clone(),
                });
            }
            other => panic!("expected decoder input event, got {other:?}"),
        }
        index += 1;
        let event = events
            .get(index)
            .unwrap_or_else(|| panic!("native trace has no decoder chunk"));
        match event {
            NativeDiagnosticEvent::DecoderChunk {
                speech_step: actual_speech,
                pcm,
            } => {
                assert_eq!(*actual_speech, speech_step);
                assert_eq!(pcm.len(), AUDIO_CHUNK_SAMPLES, "native decoder chunk size");
                records.push(NativeDiagnosticRecord::Tensor {
                    stage: "acoustic.decoder_chunk",
                    values: pcm.clone(),
                });
            }
            other => panic!("expected decoder chunk event, got {other:?}"),
        }
        index += 1;
        let event = events
            .get(index)
            .unwrap_or_else(|| panic!("native trace has no connector event"));
        match event {
            NativeDiagnosticEvent::Connector {
                speech_step: actual_speech,
                input,
                output,
            } => {
                assert_eq!(*actual_speech, speech_step);
                assert_eq!(input.len(), LATENT_WIDTH, "native connector input width");
                assert_eq!(output.len(), 896, "native connector output width");
                records.push(NativeDiagnosticRecord::Tensor {
                    stage: "acoustic.connector.input",
                    values: input.clone(),
                });
                records.push(NativeDiagnosticRecord::Tensor {
                    stage: "acoustic.connector",
                    values: output.clone(),
                });
            }
            other => panic!("expected connector event, got {other:?}"),
        }
        index += 1;
        if matches!(
            events.get(index),
            Some(NativeDiagnosticEvent::Eos {
                branch: VibeVoiceRealtimeDiagnosticBranch::PositiveTts,
                call: VibeVoiceRealtimeDiagnosticCall::Speech,
                ..
            })
        ) {
            let positive_eos = diagnostic_eos(
                events,
                &mut index,
                VibeVoiceRealtimeDiagnosticBranch::PositiveTts,
                VibeVoiceRealtimeDiagnosticCall::Speech,
                None,
                None,
                0,
                Some(speech_step),
                0,
            );
            let positive = diagnostic_hidden(
                events,
                &mut index,
                VibeVoiceRealtimeDiagnosticBranch::PositiveTts,
                VibeVoiceRealtimeDiagnosticCall::Speech,
                None,
                None,
                0,
                Some(speech_step),
            );
            let (positive_position, positive_layers) = diagnostic_cache(
                events,
                &mut index,
                VibeVoiceRealtimeDiagnosticBranch::PositiveTts,
                VibeVoiceRealtimeDiagnosticCall::Speech,
                None,
                None,
                0,
                Some(speech_step),
            );
            let negative_eos = diagnostic_eos(
                events,
                &mut index,
                VibeVoiceRealtimeDiagnosticBranch::NegativeTts,
                VibeVoiceRealtimeDiagnosticCall::Speech,
                None,
                None,
                0,
                Some(speech_step),
                0,
            );
            let negative = diagnostic_hidden(
                events,
                &mut index,
                VibeVoiceRealtimeDiagnosticBranch::NegativeTts,
                VibeVoiceRealtimeDiagnosticCall::Speech,
                None,
                None,
                0,
                Some(speech_step),
            );
            let (negative_position, negative_layers) = diagnostic_cache(
                events,
                &mut index,
                VibeVoiceRealtimeDiagnosticBranch::NegativeTts,
                VibeVoiceRealtimeDiagnosticCall::Speech,
                None,
                None,
                0,
                Some(speech_step),
            );
            records.push(NativeDiagnosticRecord::Tensor {
                stage: "tts.eos",
                values: vec![positive_eos],
            });
            records.push(NativeDiagnosticRecord::Tensor {
                stage: "tts.positive.hidden",
                values: positive,
            });
            records.push(NativeDiagnosticRecord::Cache {
                stage: "tts.positive.cache",
                position: positive_position,
                layers: positive_layers,
            });
            records.push(NativeDiagnosticRecord::Tensor {
                stage: "tts.eos",
                values: vec![negative_eos],
            });
            records.push(NativeDiagnosticRecord::Tensor {
                stage: "tts.negative.hidden",
                values: negative,
            });
            records.push(NativeDiagnosticRecord::Cache {
                stage: "tts.negative.cache",
                position: negative_position,
                layers: negative_layers,
            });
            let stop_eos = diagnostic_eos(
                events,
                &mut index,
                VibeVoiceRealtimeDiagnosticBranch::PositiveTts,
                VibeVoiceRealtimeDiagnosticCall::Speech,
                None,
                None,
                0,
                Some(speech_step),
                1,
            );
            records.push(NativeDiagnosticRecord::Tensor {
                stage: "tts.eos",
                values: vec![stop_eos],
            });
        }
        speech_step += 1;
    }
    assert_eq!(
        index,
        events.len(),
        "native diagnostic has unconsumed events"
    );
    records
}

fn diagnostic_metric_bin(expected: &[f32], native: &[f32]) -> (usize, f32, f64, usize, f32, f32) {
    assert_eq!(expected.len(), native.len(), "diagnostic metric lengths");
    let mut max_error = 0.0_f32;
    let mut squared = 0.0_f64;
    let mut max_index = 0;
    let mut max_reference = 0.0;
    let mut max_native = 0.0;
    for (index, (&reference, &actual)) in expected.iter().zip(native).enumerate() {
        let error = (reference - actual).abs();
        squared += f64::from(error) * f64::from(error);
        if error > max_error {
            max_error = error;
            max_index = index;
            max_reference = reference;
            max_native = actual;
        }
    }
    (
        expected.len(),
        max_error,
        squared,
        max_index,
        max_reference,
        max_native,
    )
}

fn compare_diagnostic_trace(reference: &[TraceEvent], native: &[NativeDiagnosticEvent]) {
    let records = native_diagnostic_records(native);
    assert_eq!(records.len(), reference.len(), "trace record count/order");
    let mut max_abs = 0.0_f32;
    let mut squared = 0.0_f64;
    let mut count = 0usize;
    let mut stage_metrics =
        BTreeMap::<String, (usize, f32, f64, usize, f32, f32, Vec<usize>)>::new();
    for (index, (expected, actual)) in reference.iter().zip(records.iter()).enumerate() {
        match actual {
            NativeDiagnosticRecord::Tensor { stage, values } => {
                assert_eq!(expected.stage, *stage, "trace stage ordering at {index}");
                let tensor = expected.tensor.as_ref().expect("expected tensor record");
                let expected_values = expected.tensor_values.as_ref().unwrap();
                let elements = tensor.shape.iter().product::<usize>();
                assert_eq!(elements, expected_values.len(), "reference shape product");
                assert_eq!(
                    tensor.shape,
                    native_stage_shape(stage, values.len()),
                    "trace tensor rank/layout {stage}"
                );
                assert_eq!(
                    expected_values.len(),
                    values.len(),
                    "trace tensor shape {stage}"
                );
                assert!(values.iter().all(|value| value.is_finite()));
                let (
                    stage_count,
                    stage_max,
                    stage_squared,
                    stage_max_index,
                    stage_reference,
                    stage_native,
                ) = diagnostic_metric_bin(expected_values, values);
                max_abs = max_abs.max(stage_max);
                squared += stage_squared;
                count += stage_count;
                stage_metrics.insert(
                    format!("{stage}/{}", expected.ordinal),
                    (
                        stage_count,
                        stage_max,
                        stage_squared,
                        stage_max_index,
                        stage_reference,
                        stage_native,
                        tensor.shape.clone(),
                    ),
                );
            }
            NativeDiagnosticRecord::Cache {
                stage,
                position,
                layers,
            } => {
                assert_eq!(expected.stage, *stage, "trace stage ordering at {index}");
                assert!(expected.tensor.is_none(), "cache stage must not be tensor");
                assert_eq!(
                    expected.cache_layers,
                    Some(*layers),
                    "cache layer count {stage}"
                );
                let lengths = expected.cache_lengths.as_ref().expect("cache lengths");
                assert_eq!(lengths.len(), *layers, "cache layer vector {stage}");
                assert!(lengths.iter().all(|length| length == position));
            }
        }
    }
    assert!(count > 0, "diagnostic trace must contain measured values");
    println!(
        "VIBEVOICE_REALTIME_STAGE_VALUES_OPEN elements={} max_abs={max_abs:.9e} rmse={:.9e} status=MEASURED_NOT_GATED",
        count,
        (squared / count as f64).sqrt()
    );
    for (
        stage,
        (
            stage_count,
            stage_max,
            stage_squared,
            stage_max_index,
            stage_reference,
            stage_native,
            stage_shape,
        ),
    ) in stage_metrics
    {
        println!(
            "VIBEVOICE_REALTIME_STAGE_VALUE stage={stage} shape={stage_shape:?} elements={stage_count} max_abs={stage_max:.9e} max_index={stage_max_index} reference={stage_reference:.9e} native={stage_native:.9e} rmse={:.9e} status=MEASURED_NOT_GATED",
            (stage_squared / stage_count as f64).sqrt()
        );
    }
}

fn native_stage_shape(stage: &str, elements: usize) -> Vec<usize> {
    // The authenticated source/config contract is batch=1, decoder channels=1,
    // VAE width=64, hidden width=896.  The native observer retains the
    // sampler's rank-2 output and rank-3 decoder/connector arrays; it does not
    // squeeze same-numel tensors.
    match stage {
        "lm.positive.prefill.hidden"
        | "lm.negative.prefill.hidden"
        | "tts.positive.prefill.hidden"
        | "tts.negative.prefill.hidden"
        | "lm.positive.hidden"
        | "tts.positive.hidden"
        | "tts.negative.hidden" => {
            assert_eq!(elements % 896, 0, "native hidden width {stage}");
            vec![1, elements / 896, 896]
        }
        "diffusion.prediction" => {
            assert_eq!(elements, 2 * LATENT_WIDTH, "native diffusion shape");
            vec![2, LATENT_WIDTH]
        }
        "speech.sampled_latent" => {
            assert_eq!(elements, LATENT_WIDTH, "native latent shape {stage}");
            vec![1, LATENT_WIDTH]
        }
        "acoustic.decode_input_unscaled" | "acoustic.connector.input" => {
            assert_eq!(elements, LATENT_WIDTH, "native latent shape {stage}");
            vec![1, 1, LATENT_WIDTH]
        }
        "acoustic.decoder_chunk" => {
            assert_eq!(elements, AUDIO_CHUNK_SAMPLES, "native PCM shape");
            vec![1, 1, AUDIO_CHUNK_SAMPLES]
        }
        "acoustic.connector" => {
            assert_eq!(elements, 896, "native connector shape");
            vec![1, 1, 896]
        }
        "tts.eos" => {
            assert_eq!(elements, 1, "native EOS shape");
            vec![1, 1]
        }
        other => panic!("unsupported native diagnostic tensor stage {other}"),
    }
}

#[test]
#[ignore = "requires authenticated VAST Realtime GGUF, Carter cache, Qwen sidecars, input text, and official packet"]
fn vibevoice_realtime_native_matches_official_streaming_structure_and_pcm_diagnostic() {
    assert_eq!(std::env::var("VOKRA_PUBLISH_ON_VAST").as_deref(), Ok("1"));
    assert_eq!(std::env::consts::OS, "linux");
    assert_eq!(std::env::consts::ARCH, "x86_64");

    let gguf_path = required_path("VOKRA_VIBEVOICE_REALTIME_GGUF");
    let gguf_sha = parse_hex32(
        &required_string("VOKRA_VIBEVOICE_REALTIME_GGUF_SHA256"),
        "GGUF SHA-256",
    );
    assert_eq!(sha256_file(&gguf_path), gguf_sha, "authenticated GGUF hash");
    let reference_dir = required_path("VOKRA_VIBEVOICE_REALTIME_REFERENCE_DIR");
    let packet_path = reference_dir.join("reference.json");
    let packet_bytes = read(&packet_path);
    let packet = parse_json(&packet_bytes).expect("reference.json must be valid JSON");
    reject_duplicate_keys(&packet, "reference");
    let input_text = required_path("VOKRA_VIBEVOICE_REALTIME_INPUT_TEXT_FILE");
    let cpu = validate_packet_identity(&reference_dir, &packet, &input_text);
    assert_eq!(string_field(cpu, "device"), "cpu");

    let events = trace_events(&reference_dir, cpu);
    let text = String::from_utf8(read(&input_text)).expect("input text must be UTF-8");
    let tokenizer_root = required_path("VOKRA_VIBEVOICE_TOKENIZER_DIR");
    let tokenizer = VibeVoiceRealtimeTokenizer::from_files(&tokenizer_root)
        .expect("authenticated Qwen sidecars must bind");
    let text_ids = tokenizer
        .streaming_text_ids(&text)
        .expect("authenticated input text must tokenize");
    let text_windows = text_ids.len().div_ceil(5);
    assert!(text_windows > 0);

    let speech_count = count_stage(&events, "speech.sampled_latent");
    assert!(
        speech_count > 0,
        "official packet must contain speech stages"
    );
    let (guidance_scale, max_new_tokens, max_speech_steps) =
        execution_contract(&packet, &input_text, speech_count);
    let terminal_max_length_chunk =
        assert_required_stage_contract(&events, speech_count, text_windows, max_new_tokens);

    let preset_root = required_path("VOKRA_VIBEVOICE_REALTIME_PRESET_DIR");
    let safetensors_path = preset_root.join("cache.safetensors");
    let manifest_path = preset_root.join("manifest.json");
    let expected_manifest = parse_hex32(
        &required_string("VOKRA_VIBEVOICE_REALTIME_PRESET_MANIFEST_SHA256"),
        "preset manifest SHA-256",
    );
    let expected_safetensors = parse_hex32(
        &required_string("VOKRA_VIBEVOICE_REALTIME_PRESET_SAFETENSORS_SHA256"),
        "preset safetensors SHA-256",
    );
    let safetensors = read(&safetensors_path);
    let manifest = read(&manifest_path);
    let preset =
        VibeVoiceRealtimePresetCache::from_bytes(&safetensors, &manifest, &expected_manifest)
            .expect("authenticated four-output preset cache");
    let actual_manifest = sha256_file(&manifest_path);
    let actual_safetensors = sha256_file(&safetensors_path);
    assert_eq!(actual_manifest, expected_manifest);
    assert_eq!(actual_safetensors, expected_safetensors);
    assert_eq!(actual_manifest, hex_digest(&preset.manifest_sha256()));
    assert_eq!(actual_safetensors, hex_digest(&preset.safetensors_sha256()));
    for (stage, branch, layers) in [
        (
            "lm.positive.prefill.cache",
            VibeVoiceRealtimePresetBranch::Lm,
            4,
        ),
        (
            "tts.positive.prefill.cache",
            VibeVoiceRealtimePresetBranch::TtsLm,
            20,
        ),
        (
            "lm.negative.prefill.cache",
            VibeVoiceRealtimePresetBranch::NegLm,
            4,
        ),
        (
            "tts.negative.prefill.cache",
            VibeVoiceRealtimePresetBranch::NegTtsLm,
            20,
        ),
    ] {
        let initial = event_for(&events, stage, 0);
        assert_eq!(
            cache_position(initial, stage, layers),
            preset.output(branch).cache_position(),
            "prefill cache position {stage}"
        );
    }

    let noise_records = array_field(cpu, "diffusion_initial_noise");
    assert_eq!(
        usize_field(cpu, "noise_draws"),
        noise_records.len(),
        "official noise draw count"
    );
    assert!(
        bool_field(cpu, "noise_matching"),
        "official noise tape must report matching replay draws"
    );
    let noise_hashes = array_field(cpu, "noise_hashes");
    assert_eq!(
        noise_hashes.len(),
        noise_records.len(),
        "official noise hash count"
    );
    assert_eq!(noise_records.len(), speech_count, "noise tape/stage count");
    let mut noise = Vec::with_capacity(speech_count);
    for (index, record) in noise_records.iter().enumerate() {
        let tensor = tensor_record(record, "diffusion noise");
        let payload_sha = parse_hex32(
            noise_hashes[index]
                .as_str()
                .unwrap_or_else(|| panic!("noise hash {index} must be a string")),
            "official noise payload hash",
        );
        let path = reference_dir.join("cpu").join(&tensor.file);
        assert_eq!(sha256_file(&path), tensor.sha256, "noise hash {index}");
        let values = read_npy_f32(&path);
        assert_eq!(
            values.shape, tensor.shape,
            "official noise manifest shape {index}"
        );
        assert_eq!(
            values.shape,
            [2, LATENT_WIDTH],
            "official noise shape {index}"
        );
        let mut payload = Vec::with_capacity(values.values.len() * 4);
        for value in &values.values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        assert_eq!(
            sha256_bytes(&payload),
            payload_sha,
            "official noise raw payload hash {index}"
        );
        noise.push(values.values[..LATENT_WIDTH].to_vec());
    }

    let pcm_record = tensor_record(field(cpu, "pcm"), "official PCM");
    let pcm_path = reference_dir.join("cpu").join(&pcm_record.file);
    assert_eq!(
        sha256_file(&pcm_path),
        pcm_record.sha256,
        "official PCM hash"
    );
    let official_pcm = read_npy_f32(&pcm_path);
    assert_eq!(
        official_pcm.shape.len(),
        2,
        "official PCM must be [batch,time]"
    );
    assert_eq!(official_pcm.shape[0], 1, "official PCM batch shape");
    assert_eq!(
        official_pcm.shape[1] % AUDIO_CHUNK_SAMPLES,
        0,
        "PCM chunk alignment"
    );
    assert_eq!(
        official_pcm.shape[1] / AUDIO_CHUNK_SAMPLES,
        speech_count - eos_drained_count(&events, terminal_max_length_chunk),
        "official PCM/audio-stage count"
    );

    let gguf = vokra_core::gguf::GgufFile::open(&gguf_path).expect("open authenticated GGUF");
    let mut runtime = VibeVoiceRealtimeRuntime::from_gguf(&gguf, BackendKind::Cpu)
        .expect("bind authenticated Realtime native composition");
    let config = VibeVoiceRealtimeSynthesisConfig {
        max_new_tokens,
        max_speech_steps,
        guidance_scale,
    };
    let mut native_trace = NativeDiagnosticTrace::default();
    let mut session = runtime
        .start_session_with_observer(&preset, &tokenizer, &text, config, &mut native_trace)
        .expect("start native session from authenticated same input");
    let mut native_pcm = Vec::new();
    let mut native_positions = Vec::new();
    let mut native_terminal = Vec::new();
    let mut native_draining = 0usize;
    for (index, initial_noise) in noise.iter().enumerate() {
        match session
            .step(initial_noise, false)
            .unwrap_or_else(|error| panic!("native step {index} failed: {error}"))
        {
            VibeVoiceRealtimeSynthesisStep::Audio(chunk) => {
                assert_eq!(chunk.pcm.len(), AUDIO_CHUNK_SAMPLES);
                assert!(chunk.pcm.iter().all(|value| value.is_finite()));
                native_pcm.extend_from_slice(&chunk.pcm);
                native_positions.push(session.generated_positions());
                native_terminal.push(chunk.terminal_after);
            }
            VibeVoiceRealtimeSynthesisStep::Draining { .. } => {
                native_draining += 1;
                // Draining has no position field, so read the authoritative
                // session counter after step_inner has completed. The
                // runtime increments it before returning this variant.
                native_positions.push(session.generated_positions());
                native_terminal.push(Some(VibeVoiceRealtimeGenerationStopReason::EndOfSpeech));
            }
            VibeVoiceRealtimeSynthesisStep::Finished { reason } => {
                panic!("native finished at recorded speech step {index}: {reason:?}");
            }
        }
    }
    let final_step = session
        .step(&[], false)
        .expect("native terminal observation");
    let native_reason = match final_step {
        VibeVoiceRealtimeSynthesisStep::Finished { reason } => reason,
        other => panic!("native consumed the reference tape but did not finish: {other:?}"),
    };
    drop(session);
    compare_diagnostic_trace(&events, &native_trace.events);
    let reference_eos = eos_steps(&events, terminal_max_length_chunk);
    if let Some(eos_step) = reference_eos.first().copied() {
        assert_eq!(
            native_reason,
            VibeVoiceRealtimeGenerationStopReason::EndOfSpeech
        );
        assert_eq!(
            native_terminal[eos_step],
            Some(VibeVoiceRealtimeGenerationStopReason::EndOfSpeech)
        );
    } else {
        assert_ne!(
            native_reason,
            VibeVoiceRealtimeGenerationStopReason::EndOfSpeech
        );
    }

    let positive_positions = positive_tts_speech_positions(
        &events,
        preset
            .output(VibeVoiceRealtimePresetBranch::TtsLm)
            .cache_position(),
    );
    negative_tts_cache_positions(
        &events,
        preset
            .output(VibeVoiceRealtimePresetBranch::NegTtsLm)
            .cache_position(),
        count_stage(&events, "tts.negative.hidden"),
    );
    let relative_positive_positions: Vec<usize> = positive_positions
        .iter()
        .map(|position| {
            position
                .checked_sub(
                    preset
                        .output(VibeVoiceRealtimePresetBranch::TtsLm)
                        .cache_position(),
                )
                .expect("reference speech cache position precedes preset cache")
        })
        .collect();
    let native_speech_positions = if terminal_max_length_chunk {
        assert_eq!(
            native_positions.len(),
            relative_positive_positions.len() + 1,
            "terminal max-length packet must have one uncached final audio chunk"
        );
        assert_eq!(
            native_positions.last().copied(),
            Some(max_new_tokens),
            "terminal max-length chunk logical position"
        );
        &native_positions[..native_positions.len() - 1]
    } else {
        &native_positions[..]
    };
    assert_eq!(
        relative_positive_positions.as_slice(),
        native_speech_positions,
        "native/reference relative speech cache positions"
    );
    assert_eq!(
        native_draining,
        eos_drained_count(&events, terminal_max_length_chunk),
        "native/reference EOS drain count"
    );
    assert_eq!(
        native_pcm.len(),
        official_pcm.values.len(),
        "native/reference PCM length"
    );
    let (max_abs, rmse) = pcm_error(&native_pcm, &official_pcm.values);
    println!(
        "VIBEVOICE_REALTIME_CPU_PCM_OPEN samples={} max_abs={max_abs:.9e} rmse={rmse:.9e} status=MEASURED_NOT_GATED",
        native_pcm.len()
    );
    println!(
        "VIBEVOICE_REALTIME_CPU_STAGE_CACHE_EOS_PASS speech_steps={} eos_steps={} status=STRUCTURAL_PASS",
        speech_count,
        reference_eos.len()
    );
}

fn eos_steps(events: &[TraceEvent], terminal_max_length_chunk: bool) -> Vec<usize> {
    // A speech call has three ordered classifier observations: positive
    // forward, negative forward, then the positive stop-classifier fan-out.
    // Text-window EOS records are the only EOS calls outside that state
    // machine and must immediately follow the LM cache record.
    let mut speech_step = None;
    let mut phase = 0_u8;
    let mut result = Vec::new();
    for (index, event) in events.iter().enumerate() {
        match (phase, event.stage.as_str()) {
            (0, "speech.sampled_latent") => {
                speech_step = Some(speech_step.map_or(0, |step| step + 1));
                phase = 1;
            }
            (1, "acoustic.connector") => phase = 2,
            (1, "tts.eos") => panic!("speech EOS appeared before acoustic connector"),
            (1, "speech.sampled_latent") => panic!("speech step restarted before EOS drain"),
            (1, _) => {}
            (2, "tts.eos") => phase = 3,
            (3, "tts.positive.hidden") => phase = 4,
            (4, "tts.positive.cache") => phase = 5,
            (5, "tts.eos") => phase = 6,
            (6, "tts.negative.hidden") => phase = 7,
            (7, "tts.negative.cache") => phase = 8,
            (8, "tts.eos") => {
                let values = event.tensor_values.as_ref().expect("EOS tensor");
                assert_eq!(values.len(), 1, "EOS tensor must be scalar");
                if values[0] > 0.0 {
                    result.push(speech_step.expect("speech EOS without speech step"));
                }
                phase = 0;
            }
            (0, "tts.eos") => {
                assert_eq!(
                    events
                        .get(index.wrapping_sub(1))
                        .map(|event| event.stage.as_str()),
                    Some("lm.positive.cache"),
                    "text EOS must follow its positive LM cache"
                );
                let values = event.tensor_values.as_ref().expect("EOS tensor");
                assert_eq!(values.len(), 1, "EOS tensor must be scalar");
            }
            (0, _) => {}
            _ => panic!(
                "malformed ordered EOS tape at stage {} in phase {phase}",
                event.stage
            ),
        }
    }
    assert!(
        phase == 0 || (terminal_max_length_chunk && phase == 2),
        "EOS tape ended before the stop classifier call"
    );
    result
}

fn eos_drained_count(events: &[TraceEvent], terminal_max_length_chunk: bool) -> usize {
    let Some(first) = eos_steps(events, terminal_max_length_chunk)
        .first()
        .copied()
    else {
        return 0;
    };
    count_stage(events, "speech.sampled_latent").saturating_sub(first + 1)
}

fn pcm_error(native: &[f32], official: &[f32]) -> (f32, f32) {
    assert_eq!(native.len(), official.len());
    let mut max_abs = 0.0_f32;
    let mut squared = 0.0_f64;
    for (&left, &right) in native.iter().zip(official) {
        let difference = (left - right).abs();
        max_abs = max_abs.max(difference);
        squared += f64::from(difference) * f64::from(difference);
    }
    (max_abs, (squared / native.len() as f64).sqrt() as f32)
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod diagnostic_parser_tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    fn prefill(events: &mut Vec<NativeDiagnosticEvent>) {
        for (branch, layers) in [
            (VibeVoiceRealtimeDiagnosticBranch::PositiveLm, 4),
            (VibeVoiceRealtimeDiagnosticBranch::NegativeLm, 4),
            (VibeVoiceRealtimeDiagnosticBranch::PositiveTts, 20),
            (VibeVoiceRealtimeDiagnosticBranch::NegativeTts, 20),
        ] {
            events.push(NativeDiagnosticEvent::Hidden {
                branch,
                call: VibeVoiceRealtimeDiagnosticCall::Prefill,
                text_window_index: None,
                text_token_index: None,
                text_window_tokens: 0,
                speech_step: None,
                values: vec![0.0; 896],
            });
            events.push(NativeDiagnosticEvent::CachePosition {
                branch,
                call: VibeVoiceRealtimeDiagnosticCall::Prefill,
                text_window_index: None,
                text_token_index: None,
                text_window_tokens: 0,
                speech_step: None,
                position: 1,
                layers,
            });
        }
    }

    fn text_window(events: &mut Vec<NativeDiagnosticEvent>, window: usize, tokens: usize) {
        for token in 0..tokens {
            for (branch, position) in [(VibeVoiceRealtimeDiagnosticBranch::PositiveLm, token + 1)] {
                events.push(NativeDiagnosticEvent::Hidden {
                    branch,
                    call: VibeVoiceRealtimeDiagnosticCall::Text,
                    text_window_index: Some(window),
                    text_token_index: Some(token),
                    text_window_tokens: tokens,
                    speech_step: None,
                    values: vec![token as f32; 896],
                });
                events.push(NativeDiagnosticEvent::CachePosition {
                    branch,
                    call: VibeVoiceRealtimeDiagnosticCall::Text,
                    text_window_index: Some(window),
                    text_token_index: Some(token),
                    text_window_tokens: tokens,
                    speech_step: None,
                    position,
                    layers: if branch == VibeVoiceRealtimeDiagnosticBranch::PositiveLm {
                        4
                    } else {
                        20
                    },
                });
            }
            events.push(NativeDiagnosticEvent::Eos {
                branch: VibeVoiceRealtimeDiagnosticBranch::PositiveTts,
                call: VibeVoiceRealtimeDiagnosticCall::Text,
                text_window_index: Some(window),
                text_token_index: Some(token),
                text_window_tokens: tokens,
                speech_step: None,
                classifier_pass: 0,
                value: -1.0,
            });
            events.push(NativeDiagnosticEvent::Hidden {
                branch: VibeVoiceRealtimeDiagnosticBranch::PositiveTts,
                call: VibeVoiceRealtimeDiagnosticCall::Text,
                text_window_index: Some(window),
                text_token_index: Some(token),
                text_window_tokens: tokens,
                speech_step: None,
                values: vec![token as f32; 896],
            });
            events.push(NativeDiagnosticEvent::CachePosition {
                branch: VibeVoiceRealtimeDiagnosticBranch::PositiveTts,
                call: VibeVoiceRealtimeDiagnosticCall::Text,
                text_window_index: Some(window),
                text_token_index: Some(token),
                text_window_tokens: tokens,
                speech_step: None,
                position: token + 1,
                layers: 20,
            });
        }
    }

    fn speech(events: &mut Vec<NativeDiagnosticEvent>, speech_step: usize, update_tts: bool) {
        for diffusion_step in 0..INFERENCE_STEPS {
            events.push(NativeDiagnosticEvent::DiffusionPrediction {
                speech_step,
                diffusion_step,
                timestep: 999 - diffusion_step * 50,
                conditional: vec![0.0; LATENT_WIDTH],
                unconditional: vec![0.0; LATENT_WIDTH],
            });
        }
        events.push(NativeDiagnosticEvent::SampledLatent {
            speech_step,
            values: vec![0.0; LATENT_WIDTH],
        });
        events.push(NativeDiagnosticEvent::DecoderInput {
            speech_step,
            scaled: vec![0.0; LATENT_WIDTH],
            unscaled: vec![0.0; LATENT_WIDTH],
        });
        events.push(NativeDiagnosticEvent::DecoderChunk {
            speech_step,
            pcm: vec![0.0; AUDIO_CHUNK_SAMPLES],
        });
        events.push(NativeDiagnosticEvent::Connector {
            speech_step,
            input: vec![0.0; LATENT_WIDTH],
            output: vec![0.0; 896],
        });
        if update_tts {
            events.push(NativeDiagnosticEvent::Eos {
                branch: VibeVoiceRealtimeDiagnosticBranch::PositiveTts,
                call: VibeVoiceRealtimeDiagnosticCall::Speech,
                text_window_index: None,
                text_token_index: None,
                text_window_tokens: 0,
                speech_step: Some(speech_step),
                classifier_pass: 0,
                value: -1.0,
            });
            for branch in [
                VibeVoiceRealtimeDiagnosticBranch::PositiveTts,
                VibeVoiceRealtimeDiagnosticBranch::NegativeTts,
            ] {
                events.push(NativeDiagnosticEvent::Hidden {
                    branch,
                    call: VibeVoiceRealtimeDiagnosticCall::Speech,
                    text_window_index: None,
                    text_token_index: None,
                    text_window_tokens: 0,
                    speech_step: Some(speech_step),
                    values: vec![0.0; 896],
                });
                events.push(NativeDiagnosticEvent::CachePosition {
                    branch,
                    call: VibeVoiceRealtimeDiagnosticCall::Speech,
                    text_window_index: None,
                    text_token_index: None,
                    text_window_tokens: 0,
                    speech_step: Some(speech_step),
                    position: speech_step + 1,
                    layers: 20,
                });
                if branch == VibeVoiceRealtimeDiagnosticBranch::PositiveTts {
                    events.push(NativeDiagnosticEvent::Eos {
                        branch: VibeVoiceRealtimeDiagnosticBranch::NegativeTts,
                        call: VibeVoiceRealtimeDiagnosticCall::Speech,
                        text_window_index: None,
                        text_token_index: None,
                        text_window_tokens: 0,
                        speech_step: Some(speech_step),
                        classifier_pass: 0,
                        value: -1.0,
                    });
                }
            }
            events.push(NativeDiagnosticEvent::Eos {
                branch: VibeVoiceRealtimeDiagnosticBranch::PositiveTts,
                call: VibeVoiceRealtimeDiagnosticCall::Speech,
                text_window_index: None,
                text_token_index: None,
                text_window_tokens: 0,
                speech_step: Some(speech_step),
                classifier_pass: 1,
                value: -1.0,
            });
        }
    }

    fn interleaved_fixture() -> Vec<NativeDiagnosticEvent> {
        let mut events = Vec::new();
        prefill(&mut events);
        text_window(&mut events, 0, 6);
        for step in 0..7 {
            speech(&mut events, step, true);
        }
        text_window(&mut events, 1, 1);
        for step in 7..15 {
            speech(&mut events, step, step < 14);
        }
        events
    }

    fn expected_source_stage_names() -> Vec<&'static str> {
        let mut expected = vec![
            "lm.positive.prefill.hidden",
            "lm.positive.prefill.cache",
            "lm.negative.prefill.hidden",
            "lm.negative.prefill.cache",
            "tts.positive.prefill.hidden",
            "tts.positive.prefill.cache",
            "tts.negative.prefill.hidden",
            "tts.negative.prefill.cache",
        ];
        for _ in [6, 1] {
            expected.extend([
                "lm.positive.hidden",
                "lm.positive.cache",
                "tts.eos",
                "tts.positive.hidden",
                "tts.positive.cache",
            ]);
        }
        for step in 0..15 {
            expected.extend(std::iter::repeat("diffusion.prediction").take(INFERENCE_STEPS));
            expected.extend([
                "speech.sampled_latent",
                "acoustic.decode_input_unscaled",
                "acoustic.decoder_chunk",
                "acoustic.connector.input",
                "acoustic.connector",
            ]);
            if step < 14 {
                expected.extend([
                    "tts.eos",
                    "tts.positive.hidden",
                    "tts.positive.cache",
                    "tts.eos",
                    "tts.negative.hidden",
                    "tts.negative.cache",
                    "tts.eos",
                ]);
            }
        }
        expected
    }

    fn independent_shape_reference() -> Vec<TraceEvent> {
        let mut ordinals = BTreeMap::<String, usize>::new();
        expected_source_stage_names()
            .into_iter()
            .map(|stage| {
                let ordinal = ordinals.entry(stage.to_owned()).or_default();
                let result = if stage.ends_with(".cache") {
                    let layers = if stage.starts_with("lm.") { 4 } else { 20 };
                    TraceEvent {
                        stage: stage.to_owned(),
                        ordinal: *ordinal,
                        tensor: None,
                        tensor_values: None,
                        cache_layers: Some(layers),
                        cache_lengths: Some(vec![1; layers]),
                    }
                } else {
                    let elements = match stage {
                        "diffusion.prediction" => 2 * LATENT_WIDTH,
                        "speech.sampled_latent"
                        | "acoustic.decode_input_unscaled"
                        | "acoustic.connector.input" => LATENT_WIDTH,
                        "acoustic.decoder_chunk" => AUDIO_CHUNK_SAMPLES,
                        "acoustic.connector"
                        | "lm.positive.prefill.hidden"
                        | "lm.negative.prefill.hidden"
                        | "tts.positive.prefill.hidden"
                        | "tts.negative.prefill.hidden"
                        | "lm.positive.hidden"
                        | "tts.positive.hidden"
                        | "tts.negative.hidden" => 896,
                        "tts.eos" => 1,
                        other => panic!("unmapped independent stage {other}"),
                    };
                    TraceEvent {
                        stage: stage.to_owned(),
                        ordinal: *ordinal,
                        tensor: Some(TensorRecord {
                            file: format!("{stage}-{ordinal}.npy"),
                            shape: native_stage_shape(stage, elements),
                            sha256: "0".repeat(64),
                        }),
                        tensor_values: Some(vec![0.0; elements]),
                        cache_layers: None,
                        cache_lengths: None,
                    }
                };
                *ordinal += 1;
                result
            })
            .collect()
    }

    #[test]
    fn parser_accepts_interleaved_windows_eos_drain_and_terminal_chunk() {
        let native = interleaved_fixture();
        let records = native_diagnostic_records(&native);
        assert!(records.len() > 14 * INFERENCE_STEPS);
        let actual: Vec<&str> = records
            .iter()
            .map(|record| match record {
                NativeDiagnosticRecord::Tensor { stage, .. }
                | NativeDiagnosticRecord::Cache { stage, .. } => *stage,
            })
            .collect();
        let expected = expected_source_stage_names();
        assert_eq!(actual, expected, "source-ordered official diagnostic tape");
    }

    #[test]
    fn parser_rejects_missing_reordered_and_local_reset_events() {
        let native = interleaved_fixture();
        for mutate in [0usize, 1usize] {
            let mut malformed = native.clone();
            if mutate == 0 {
                malformed.remove(16);
            } else {
                malformed.swap(8, 9);
            }
            assert!(
                catch_unwind(AssertUnwindSafe(|| native_diagnostic_records(&malformed))).is_err()
            );
        }
        let mut local_reset = native;
        for event in &mut local_reset {
            if let NativeDiagnosticEvent::DiffusionPrediction { speech_step, .. } = event {
                if *speech_step == 7 {
                    *speech_step = 0;
                    break;
                }
            }
        }
        assert!(
            catch_unwind(AssertUnwindSafe(|| native_diagnostic_records(&local_reset))).is_err()
        );

        let mut missing_stop = interleaved_fixture();
        let stop_index = missing_stop
            .iter()
            .position(|event| {
                matches!(
                    event,
                    NativeDiagnosticEvent::Eos {
                        call: VibeVoiceRealtimeDiagnosticCall::Speech,
                        classifier_pass: 1,
                        ..
                    }
                )
            })
            .unwrap();
        missing_stop.remove(stop_index);
        assert!(
            catch_unwind(AssertUnwindSafe(|| native_diagnostic_records(
                &missing_stop
            )))
            .is_err()
        );

        let mut reordered_stop = interleaved_fixture();
        let stop_index = reordered_stop
            .iter()
            .position(|event| {
                matches!(
                    event,
                    NativeDiagnosticEvent::Eos {
                        call: VibeVoiceRealtimeDiagnosticCall::Speech,
                        classifier_pass: 1,
                        ..
                    }
                )
            })
            .unwrap();
        reordered_stop.swap(stop_index - 1, stop_index);
        assert!(
            catch_unwind(AssertUnwindSafe(|| native_diagnostic_records(
                &reordered_stop
            )))
            .is_err()
        );

        let mut extra_stop = interleaved_fixture();
        let stop_index = extra_stop
            .iter()
            .position(|event| {
                matches!(
                    event,
                    NativeDiagnosticEvent::Eos {
                        call: VibeVoiceRealtimeDiagnosticCall::Speech,
                        classifier_pass: 1,
                        ..
                    }
                )
            })
            .unwrap();
        let duplicate = extra_stop[stop_index].clone();
        extra_stop.insert(stop_index, duplicate);
        assert!(catch_unwind(AssertUnwindSafe(|| native_diagnostic_records(&extra_stop))).is_err());
    }

    #[test]
    fn parser_rejects_token_aggregation_and_same_numel_shape_mismatch() {
        let native = interleaved_fixture();
        let mut token_mismatch = native.clone();
        for event in &mut token_mismatch {
            if let NativeDiagnosticEvent::Hidden {
                call: VibeVoiceRealtimeDiagnosticCall::Text,
                text_token_index: Some(token),
                ..
            } = event
            {
                if *token == 1 {
                    *token = 0;
                    break;
                }
            }
        }
        assert!(
            catch_unwind(AssertUnwindSafe(|| native_diagnostic_records(
                &token_mismatch
            )))
            .is_err()
        );

        let mut reference = independent_shape_reference();
        let first_tensor = reference
            .iter_mut()
            .find(|event| event.tensor.is_some())
            .unwrap();
        let elements = first_tensor.tensor_values.as_ref().unwrap().len();
        first_tensor.tensor.as_mut().unwrap().shape = vec![elements];
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                compare_diagnostic_trace(&reference, &native)
            }))
            .is_err()
        );
    }

    fn speech_eos_tape(positive: f32, negative: f32, stop: f32) -> Vec<TraceEvent> {
        let tensor = |stage: &str, values: Vec<f32>| TraceEvent {
            stage: stage.to_owned(),
            ordinal: 0,
            tensor: None,
            tensor_values: Some(values),
            cache_layers: None,
            cache_lengths: None,
        };
        vec![
            tensor("speech.sampled_latent", vec![0.0]),
            tensor("acoustic.decoder_chunk", vec![0.0]),
            tensor("acoustic.connector", vec![0.0]),
            tensor("tts.eos", vec![positive]),
            tensor("tts.positive.hidden", vec![0.0]),
            tensor("tts.positive.cache", vec![]),
            tensor("tts.eos", vec![negative]),
            tensor("tts.negative.hidden", vec![0.0]),
            tensor("tts.negative.cache", vec![]),
            tensor("tts.eos", vec![stop]),
        ]
    }

    #[test]
    fn eos_stop_selection_uses_third_ordered_classifier_call() {
        assert_eq!(eos_steps(&speech_eos_tape(-1.0, 7.0, 2.0), false), vec![0]);
        assert!(eos_steps(&speech_eos_tape(2.0, -7.0, -2.0), false).is_empty());
    }

    #[test]
    fn eos_order_rejects_missing_reordered_and_extra_calls() {
        let tape = speech_eos_tape(-1.0, 7.0, 2.0);
        let mut missing = tape.clone();
        missing.pop();
        assert!(catch_unwind(AssertUnwindSafe(|| eos_steps(&missing, false))).is_err());

        let mut reordered = tape.clone();
        reordered.swap(3, 4);
        assert!(catch_unwind(AssertUnwindSafe(|| eos_steps(&reordered, false))).is_err());

        let mut extra = tape.clone();
        let duplicate = extra[9].clone();
        extra.insert(9, duplicate);
        assert!(catch_unwind(AssertUnwindSafe(|| eos_steps(&extra, false))).is_err());

        let mut terminal = speech_eos_tape(-1.0, 7.0, 2.0);
        terminal.truncate(3);
        assert!(eos_steps(&terminal, true).is_empty());
        assert!(catch_unwind(AssertUnwindSafe(|| eos_steps(&terminal, false))).is_err());
    }

    #[test]
    fn diagnostic_metric_reports_nonzero_worst_bin_and_values() {
        let metric = diagnostic_metric_bin(&[0.0, 4.0, 2.0], &[0.0, 1.0, 2.0]);
        assert_eq!(metric.0, 3);
        assert_eq!(metric.1, 3.0);
        assert_eq!(metric.3, 1);
        assert_eq!(metric.4, 4.0);
        assert_eq!(metric.5, 1.0);
    }

    #[test]
    fn source_shapes_retain_sampler_and_decoder_layouts() {
        assert_eq!(
            native_stage_shape("speech.sampled_latent", LATENT_WIDTH),
            vec![1, LATENT_WIDTH]
        );
        assert_eq!(
            native_stage_shape("acoustic.decode_input_unscaled", LATENT_WIDTH),
            vec![1, 1, LATENT_WIDTH]
        );
        assert_eq!(
            native_stage_shape("acoustic.connector.input", LATENT_WIDTH),
            vec![1, 1, LATENT_WIDTH]
        );
        assert_eq!(
            native_stage_shape("acoustic.connector", 896),
            vec![1, 1, 896]
        );
        assert_eq!(
            native_stage_shape("acoustic.decoder_chunk", AUDIO_CHUNK_SAMPLES),
            vec![1, 1, AUDIO_CHUNK_SAMPLES]
        );
        assert_eq!(native_stage_shape("tts.eos", 1), vec![1, 1]);
    }
}

#[cfg(test)]
mod runtime_identity_tests {
    use super::{JsonValue, parse_json, validate_runtime_identity};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    const HEAD: &str = "0123456789abcdef0123456789abcdef01234567";
    const TREE: &str = "89abcdef0123456789abcdef0123456789abcdef";
    const OTHER_TREE: &str = "abcdef0123456789abcdef0123456789abcdef01";

    fn identities(
        packet_head: &str,
        packet_tree: &str,
        scope_head: &str,
        scope_tree: &str,
    ) -> (JsonValue, JsonValue) {
        let packet = parse_json(format!(
            r#"{{"runtime":{{"vokra_head":"{packet_head}","vokra_tree_sha1":"{packet_tree}"}}}}"#
        ).as_bytes())
        .expect("packet fixture JSON");
        let scope = parse_json(
            format!(
                r#"{{"runtime":{{"vokra_head":"{scope_head}","vokra_tree_sha1":"{scope_tree}"}}}}"#
            )
            .as_bytes(),
        )
        .expect("scope fixture JSON");
        (packet, scope)
    }

    fn rejects(packet: JsonValue, scope: JsonValue, expected_head: &str) {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                validate_runtime_identity(&packet, &scope, expected_head)
            }))
            .is_err()
        );
    }

    #[test]
    fn matching_packet_and_owner_runtime_identity_passes() {
        let (packet, scope) = identities(HEAD, TREE, HEAD, TREE);
        validate_runtime_identity(&packet, &scope, HEAD);
    }

    #[test]
    fn changed_packet_tree_fails_even_when_head_is_unchanged() {
        let (packet, scope) = identities(HEAD, OTHER_TREE, HEAD, TREE);
        rejects(packet, scope, HEAD);
    }

    #[test]
    fn changed_owner_tree_fails() {
        let (packet, scope) = identities(HEAD, TREE, HEAD, OTHER_TREE);
        rejects(packet, scope, HEAD);
    }

    #[test]
    fn missing_runtime_fields_fail_closed() {
        let packet =
            parse_json(br#"{"runtime":{"vokra_head":"0123456789abcdef0123456789abcdef01234567"}}"#)
                .expect("packet fixture JSON");
        let scope = parse_json(
            format!(r#"{{"runtime":{{"vokra_head":"{HEAD}","vokra_tree_sha1":"{TREE}"}}}}"#)
                .as_bytes(),
        )
        .expect("scope fixture JSON");
        rejects(packet, scope, HEAD);

        let packet = parse_json(
            format!(r#"{{"runtime":{{"vokra_head":"{HEAD}","vokra_tree_sha1":"{TREE}"}}}}"#)
                .as_bytes(),
        )
        .expect("packet fixture JSON");
        let scope = parse_json(format!(r#"{{"runtime":{{"vokra_head":"{HEAD}"}}}}"#).as_bytes())
            .expect("scope fixture JSON");
        rejects(packet, scope, HEAD);
    }

    #[test]
    fn malformed_uppercase_and_truncated_identities_fail() {
        let (uppercase, scope) = identities(&HEAD.to_ascii_uppercase(), TREE, HEAD, TREE);
        rejects(uppercase, scope.clone(), HEAD);
        let (uppercase_tree, scope) = identities(HEAD, &TREE.to_ascii_uppercase(), HEAD, TREE);
        rejects(uppercase_tree, scope, HEAD);
        let non_hex = "0123456789abcdef0123456789abcdef0123456g";
        let (non_hex_tree, scope) = identities(HEAD, non_hex, HEAD, TREE);
        rejects(non_hex_tree, scope, HEAD);
        let truncated = &TREE[..39];
        let (packet, scope) = identities(HEAD, truncated, HEAD, TREE);
        rejects(packet, scope, HEAD);
    }

    #[test]
    fn head_mismatch_fails_for_packet_owner_or_external_expectation() {
        let other_head = "fedcba9876543210fedcba9876543210fedcba98";
        let (packet, scope) = identities(other_head, TREE, HEAD, TREE);
        rejects(packet, scope, HEAD);
        let (packet, scope) = identities(HEAD, TREE, other_head, TREE);
        rejects(packet, scope, HEAD);
        let (packet, scope) = identities(HEAD, TREE, HEAD, TREE);
        rejects(packet, scope, other_head);
    }
}
