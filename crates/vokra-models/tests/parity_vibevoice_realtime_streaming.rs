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
use vokra_models::vibevoice_streaming::{
    VibeVoiceRealtimeGenerationStopReason, VibeVoiceRealtimePresetBranch,
    VibeVoiceRealtimePresetCache, VibeVoiceRealtimeRuntime, VibeVoiceRealtimeSynthesisConfig,
    VibeVoiceRealtimeSynthesisStep, VibeVoiceRealtimeTokenizer,
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
        (1, 0) => (10, u16::from_le_bytes([bytes[8], bytes[9]]) as usize),
        (2, 0) | (3, 0) => {
            assert!(
                bytes.len() >= 12,
                "{} has a truncated NPY header",
                path.display()
            );
            (
                12,
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

#[derive(Debug)]
struct TraceEvent {
    stage: String,
    ordinal: usize,
    tensor: Option<TensorRecord>,
    tensor_values: Option<Vec<f32>>,
    cache_layers: Option<usize>,
    cache_lengths: Option<Vec<usize>>,
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
    let tts_update_count = count_stage(events, "tts.eos");
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
    let observed_text_windows = count_stage(events, "lm.positive.hidden");
    assert!(
        observed_text_windows > 0 && observed_text_windows <= requested_text_windows,
        "unsupported official packet: observed text windows must be a non-empty prefix of the authenticated input plan"
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
        ("tts.eos", tts_update_count),
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
        speech_count - eos_drained_count(&events),
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
    let mut session = runtime
        .start_session(&preset, &tokenizer, &text, config)
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
    let reference_eos = eos_steps(&events);
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
        count_stage(&events, "tts.eos"),
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
        eos_drained_count(&events),
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

fn eos_steps(events: &[TraceEvent]) -> Vec<usize> {
    events
        .iter()
        .filter(|event| event.stage == "tts.eos")
        .enumerate()
        .filter_map(|(index, event)| {
            let values = event.tensor_values.as_ref().expect("EOS tensor");
            assert_eq!(values.len(), 1, "EOS tensor must be scalar");
            (values[0] > 0.0).then_some(index)
        })
        .collect()
}

fn eos_drained_count(events: &[TraceEvent]) -> usize {
    let Some(first) = eos_steps(events).first().copied() else {
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
