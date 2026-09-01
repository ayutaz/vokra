//! microWakeWord host parity harness — env-gated (M5-03b Phase 4).
//!
//! Sibling of `crates/vokra-models/tests/parity_rmvpe.rs` (F0 tier) and
//! `parity_openwakeword.rs` (KWS tier for the RPi/Linux openWakeWord line):
//! every test that needs a real microWakeWord artefact is gated on an
//! environment variable and skips cleanly when unset — never a fabricated
//! pass. Once opted in, every failure is hard (a missing / malformed /
//! wrong-shaped fixture is a loud panic — FR-EX-08).
//!
//! # Env vars (owner-set to bind the fixture-gated paths)
//!
//! * `VOKRA_KWS_REAL_GGUF` — path to a Vokra microWakeWord GGUF, as
//!   emitted by
//!   `tools/parity/microwakeword/prepare_checkpoint.py --output <gguf>`.
//!   Naming rationale: Vokra runtime consumes GGUF, not `.tflite`
//!   (FR-LD-05 sidecar isolation — no TFLite / Python / FlatBuffer in
//!   the runtime). The task text names `VOKRA_KWS_REAL_TFLITE` for the
//!   upstream artefact — here we use `_REAL_GGUF` for what the runtime
//!   actually opens, matching the `VOKRA_RMVPE_REAL_GGUF` /
//!   `VOKRA_OPENWAKEWORD_REAL_GGUF` sibling precedent.
//! * `VOKRA_KWS_REAL_FIXTURES` — path to the directory of reference
//!   dumps emitted by `tools/parity/microwakeword/dump_reference.py`
//!   (`input_pcm.bin` + `features_ref.bin` + per-invocation input/output
//!   files + `manifest.json`). Owner triggers the dumper on the same source
//!   `.tflite` the GGUF was converted from.
//!
//! # Fixture recipe (owner-side)
//!
//! ```text
//! # 0. On VAST, complete the isolated dependency/native-license audit.
//! #    The current result is BLOCKED_UNREVIEWED_TRANSITIVE; do not proceed
//! #    until inspect.py reports fixture_generation_permitted=true.
//! cd tools/parity/microwakeword-reference
//! uv run --no-project --offline --python 3.12 python inspect.py
//!
//! # 1. Use the fixed candidate paths materialized by the VAST validation
//! #    worker (regular files outside the checkout). Do not use a direct
//! #    ad-hoc URL conversion recipe: production conversion remains blocked
//! #    and candidate output is not an authenticated production bind.
//! #    The worker's `--candidate` invocation consumes its fixed TFLite and
//! #    raw-inventory paths and emits a separately named candidate GGUF.
//! #    Keep those exact worker-reported paths for the environment below.
//!
//! # 2. After the audit is PASS, run the reference dumper (owner walkthrough
//! #    — the DL from step 1
//! #    lives in the tmpdir; use --input to keep it):
//! # (the regular .tflite path is owner-provided; no symlinks are accepted)
//! cd ../microwakeword-reference
//! uv run python ../microwakeword/dump_reference.py \
//!     --tflite-path ~/.cache/vokra-eval/weights/microwakeword/hey_jarvis.tflite \
//!     --output-dir  ~/.cache/vokra-eval/fixtures/microwakeword \
//!     --dependency-evidence /absolute/path/to/dependency-evidence.json
//!
//! # 3. Point the parity harness at both artefacts:
//! export VOKRA_KWS_REAL_GGUF=~/.cache/vokra-eval/weights/microwakeword/hey_jarvis.gguf
//! export VOKRA_KWS_REAL_FIXTURES=~/.cache/vokra-eval/fixtures/microwakeword
//! CARGO_BUILD_JOBS=1 cargo test -p vokra-kws-micro --test parity_microwakeword -- --nocapture
//! ```
//!
//! # Test layers (mirroring `parity_rmvpe.rs`)
//!
//! ## Path A — full GGUF smoke (`VOKRA_KWS_REAL_GGUF`)
//!
//! Loads the real Vokra GGUF via [`crate::model::Model::from_bytes`],
//! asserts the `vokra.kws.*` metadata contract holds (arch, threshold,
//! sample rate, mel width) and the tensor count is above the
//! "synthesized 1-tensor smoke" floor. This validates that the loader
//! survives a real MC-MobileNet checkpoint without silent binds.
//!
//! ## Path B — log-mel feature extractor parity (`VOKRA_KWS_REAL_FIXTURES`)
//!
//! Reads the dumped `input_pcm.bin` (raw `i16` little-endian,
//! [`features::WINDOW_SAMPLES`] samples @ 16 kHz), runs Vokra's
//! [`features::FeatureExtractor::compute_frame_f32`], and compares
//! per-band `|Δ|` against the dumped `features_ref.bin`
//! (numpy transcription of the standard log-mel algorithm) at
//! `atol = 5e-2` (registered f32/numpy architectural bound).
//!
//! The numpy reference is a transcription of the same algorithm the
//! Rust code implements (Hann window, radix-2 FFT, mel filterbank,
//! log10 with floor), so parity validates transcription faithfulness.
//! It does not validate against the training-time TensorFlow
//! `tf.signal` mel front-end used to train microWakeWord — that
//! would require `tensorflow`, out of the sidecar's two-dependency footprint.
//! No TensorFlow `tf.signal` comparison is performed by this fixture. The
//! registered `5e-2` boundary applies only to the independent numpy
//! transcription versus Rust f32 comparison; training-time TensorFlow
//! front-end parity remains unverified.
//!
//! ## Path C — authenticated streaming contract (both env vars set)
//!
//! This path integrity-checks every independent `ai_edge_litert` invocation:
//! exact `[1,3,40]` int8 input bytes, exact `[1,1]` uint8 output bytes,
//! affine dequantisation, model identity, and fresh-interpreter replay. It
//! then reaches the explicit production binder boundary. The current
//! `Model::bind_authenticated_chain` deliberately returns
//! `AUTHENTICATED_TOPOLOGY_REQUIRED`; fixture verification therefore ends in
//! a hard failure, never a parity PASS. A future stateful authenticated binder
//! must implement the typed per-step/reset seam below. Artifact SHA
//! recomputation remains an outer VAST evidence-gate responsibility.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use vokra_core::json::{self, JsonValue};
use vokra_kws_micro::features::{self, FeatureExtractor};
use vokra_kws_micro::model::Model;

/// Env var the owner sets to point Path A + Path C at a real Vokra
/// microWakeWord GGUF. Absent = skip cleanly.
const GGUF_ENV: &str = "VOKRA_KWS_REAL_GGUF";

/// Env var the owner sets to point Path B + Path C at the directory of
/// reference dumps emitted by `dump_reference.py`. Absent = skip cleanly.
const FIXTURES_ENV: &str = "VOKRA_KWS_REAL_FIXTURES";

const AUTHENTICATED_TFLITE_SHA256: &str =
    "21a7976add39ee24ec96c63d96b7aaa18e24d1d9824b963e451da8feb4b78b77";
const AUTHENTICATED_TFLITE_BYTES: i64 = 52_272;
const INVOCATION_COUNT: usize = 4;
const INPUT_BYTES: usize = 3 * 40;
const OUTPUT_BYTES: usize = 1;
const OUTPUT_SCALE: f32 = 1.0 / 256.0;
const DEPENDENCY_EVIDENCE_SCHEMA: &str = "microwakeword-reference-dependency-evidence-v1";
const DEPENDENCY_EVIDENCE_STATUS: &str = "EVIDENCE_COLLECTED_OWNER_REVIEW_REQUIRED";
const REFERENCE_PROJECT_SHA256: &str =
    "2b114885d54470c8397528b37572e3632202ca0b9d65ac349ec7e7da4e331f03";
const REFERENCE_LOCK_SHA256: &str =
    "da75839f6195c27c32a15f097a40450c18b317ad78e9036ec2a1618472b85555";
const REFERENCE_DISTRIBUTIONS: &[(&str, &str)] = &[
    ("ai-edge-litert", "2.2.0"),
    ("backports-strenum", "1.3.1"),
    ("flatbuffers", "25.12.19"),
    ("ml-dtypes", "0.6.0"),
    ("numpy", "2.5.2"),
    ("protobuf", "7.36.1"),
    ("tqdm", "4.70.0"),
    ("typing-extensions", "4.16.0"),
];

fn object<'a>(value: &'a JsonValue, label: &str) -> &'a [(String, JsonValue)] {
    value
        .as_object()
        .unwrap_or_else(|| panic!("{label} must be a JSON object"))
}

fn field<'a>(value: &'a JsonValue, key: &str, label: &str) -> &'a JsonValue {
    let entries = object(value, label);
    let matches: Vec<&JsonValue> = entries
        .iter()
        .filter(|(name, _)| name == key)
        .map(|(_, item)| item)
        .collect();
    assert_eq!(matches.len(), 1, "{label}.{key} must occur exactly once");
    matches[0]
}

fn exact_keys(value: &JsonValue, expected: &[&str], label: &str) {
    let entries = object(value, label);
    assert_eq!(entries.len(), expected.len(), "{label} key count drift");
    for key in expected {
        assert_eq!(
            entries.iter().filter(|(name, _)| name == key).count(),
            1,
            "{label}.{key} missing/duplicate"
        );
    }
}

fn reject_duplicate_keys(value: &JsonValue, label: &str) {
    match value {
        JsonValue::Object(entries) => {
            for (index, (key, child)) in entries.iter().enumerate() {
                assert_eq!(
                    entries[index + 1..]
                        .iter()
                        .filter(|(name, _)| name == key)
                        .count(),
                    0,
                    "duplicate JSON key {label}.{key}"
                );
                if matches!(child, JsonValue::Object(_) | JsonValue::Array(_)) {
                    reject_duplicate_keys(child, label);
                }
            }
        }
        JsonValue::Array(items) => {
            for item in items {
                if matches!(item, JsonValue::Object(_) | JsonValue::Array(_)) {
                    reject_duplicate_keys(item, label);
                }
            }
        }
        _ => {}
    }
}

fn string_field<'a>(value: &'a JsonValue, key: &str, label: &str) -> &'a str {
    field(value, key, label)
        .as_str()
        .unwrap_or_else(|| panic!("{label}.{key} must be a string"))
}

fn sha256_text(value: &str, label: &str) {
    assert_eq!(value.len(), 64, "{label} must be 64 hex characters");
    assert!(
        value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        "{label} must be lowercase hex"
    );
}

fn int_field(value: &JsonValue, key: &str, label: &str) -> i64 {
    match field(value, key, label) {
        JsonValue::Int(number) => *number,
        other => panic!("{label}.{key} must be an integer, got {other:?}"),
    }
}

fn bool_field(value: &JsonValue, key: &str, label: &str) -> bool {
    match field(value, key, label) {
        JsonValue::Bool(flag) => *flag,
        other => panic!("{label}.{key} must be a boolean, got {other:?}"),
    }
}

fn shape_field(value: &JsonValue, key: &str, label: &str) -> Vec<usize> {
    match field(value, key, label) {
        JsonValue::Array(items) => items
            .iter()
            .map(|item| match item {
                JsonValue::Int(number) if *number >= 0 => *number as usize,
                other => {
                    panic!("{label}.{key} contains a non-negative integer violation: {other:?}")
                }
            })
            .collect(),
        other => panic!("{label}.{key} must be an array, got {other:?}"),
    }
}

fn float_field(value: &JsonValue, key: &str, label: &str) -> f64 {
    match field(value, key, label) {
        JsonValue::Float(number) if number.is_finite() => *number,
        JsonValue::Int(number) => *number as f64,
        other => panic!("{label}.{key} must be a finite number, got {other:?}"),
    }
}

fn verify_distribution_versions(value: &JsonValue, label: &str) {
    exact_keys(
        value,
        &[
            "ai-edge-litert",
            "backports-strenum",
            "flatbuffers",
            "ml-dtypes",
            "numpy",
            "protobuf",
            "tqdm",
            "typing-extensions",
        ],
        label,
    );
    for &(name, expected) in REFERENCE_DISTRIBUTIONS {
        assert_eq!(
            string_field(value, name, label),
            expected,
            "{label}.{name} version drift"
        );
    }
}

fn verify_dependency_evidence(value: &JsonValue) {
    exact_keys(
        value,
        &[
            "schema",
            "status",
            "publication_permitted",
            "fixture_generation_permitted",
            "owner_review_required",
            "failures",
            "path",
            "sha256",
            "project_sha256",
            "uv_lock_sha256",
            "platform",
            "installed_distributions",
        ],
        "manifest.dependency_evidence",
    );
    assert_eq!(
        string_field(value, "schema", "manifest.dependency_evidence"),
        DEPENDENCY_EVIDENCE_SCHEMA
    );
    assert_eq!(
        string_field(value, "status", "manifest.dependency_evidence"),
        DEPENDENCY_EVIDENCE_STATUS
    );
    assert!(!bool_field(
        value,
        "publication_permitted",
        "manifest.dependency_evidence"
    ));
    assert!(!bool_field(
        value,
        "fixture_generation_permitted",
        "manifest.dependency_evidence"
    ));
    assert!(bool_field(
        value,
        "owner_review_required",
        "manifest.dependency_evidence"
    ));
    assert!(matches!(
        field(value, "failures", "manifest.dependency_evidence"),
        JsonValue::Array(items) if items.is_empty()
    ));
    let path = string_field(value, "path", "manifest.dependency_evidence");
    assert!(!path.is_empty() && !path.contains('/') && !path.contains('\\'));
    sha256_text(
        string_field(value, "sha256", "manifest.dependency_evidence"),
        "manifest.dependency_evidence.sha256",
    );
    assert_eq!(
        string_field(value, "project_sha256", "manifest.dependency_evidence"),
        REFERENCE_PROJECT_SHA256
    );
    assert_eq!(
        string_field(value, "uv_lock_sha256", "manifest.dependency_evidence"),
        REFERENCE_LOCK_SHA256
    );
    let platform = field(value, "platform", "manifest.dependency_evidence");
    exact_keys(
        platform,
        &["system", "machine", "python"],
        "manifest.dependency_evidence.platform",
    );
    assert_eq!(
        string_field(platform, "system", "manifest.dependency_evidence.platform"),
        "Linux"
    );
    assert_eq!(
        string_field(platform, "machine", "manifest.dependency_evidence.platform"),
        "x86_64"
    );
    assert_eq!(
        string_field(platform, "python", "manifest.dependency_evidence.platform"),
        "3.12"
    );
    verify_distribution_versions(
        field(
            value,
            "installed_distributions",
            "manifest.dependency_evidence",
        ),
        "manifest.dependency_evidence.installed_distributions",
    );
}

fn verify_reference_manifest(root: &JsonValue) {
    reject_duplicate_keys(root, "manifest");
    exact_keys(
        root,
        &[
            "schema",
            "status",
            "generator",
            "generator_version",
            "oracle",
            "source_tflite",
            "source_tflite_sha256",
            "source_tflite_bytes",
            "authenticated_model_sha256",
            "constants",
            "pcm_synthesis",
            "authenticated_io",
            "persistent_sequence",
            "artefacts",
            "tflite_topology",
            "frontend_parity_boundary",
            "reference_environment",
            "dependency_evidence",
        ],
        "manifest",
    );
    assert_eq!(
        string_field(root, "schema", "manifest"),
        "microwakeword-reference-v2"
    );
    assert_eq!(
        string_field(root, "status", "manifest"),
        "REFERENCE_COMPLETE"
    );
    assert_eq!(
        string_field(root, "oracle", "manifest"),
        "ai_edge_litert.Interpreter running the pinned upstream TFLite; never a Vokra mirror"
    );
    for key in [
        "generator",
        "generator_version",
        "source_tflite",
        "frontend_parity_boundary",
    ] {
        let _ = string_field(root, key, "manifest");
    }
    let source_digest = string_field(root, "source_tflite_sha256", "manifest");
    let auth_digest = string_field(root, "authenticated_model_sha256", "manifest");
    sha256_text(source_digest, "manifest.source_tflite_sha256");
    sha256_text(auth_digest, "manifest.authenticated_model_sha256");
    assert_eq!(source_digest, AUTHENTICATED_TFLITE_SHA256);
    assert_eq!(auth_digest, AUTHENTICATED_TFLITE_SHA256);
    assert_eq!(
        int_field(root, "source_tflite_bytes", "manifest"),
        AUTHENTICATED_TFLITE_BYTES
    );

    let constants = field(root, "constants", "manifest");
    exact_keys(
        constants,
        &[
            "sample_rate",
            "hop_ms",
            "window_ms",
            "n_mels",
            "hop_samples",
            "window_samples",
            "n_fft",
            "n_bins",
            "log_mel_epsilon",
        ],
        "constants",
    );
    for key in [
        "sample_rate",
        "hop_ms",
        "window_ms",
        "n_mels",
        "hop_samples",
        "window_samples",
        "n_fft",
        "n_bins",
    ] {
        assert!(int_field(constants, key, "constants") > 0);
    }
    assert!(float_field(constants, "log_mel_epsilon", "constants") > 0.0);
    let pcm = field(root, "pcm_synthesis", "manifest");
    exact_keys(
        pcm,
        &[
            "seed",
            "sine_hz",
            "sine_amplitude",
            "noise_stddev",
            "distinct_frame_schedule",
        ],
        "pcm_synthesis",
    );
    assert!(int_field(pcm, "seed", "pcm_synthesis") >= 0);
    for key in ["sine_hz", "sine_amplitude", "noise_stddev"] {
        assert!(float_field(pcm, key, "pcm_synthesis") > 0.0);
    }
    let _ = string_field(pcm, "distinct_frame_schedule", "pcm_synthesis");

    let io = field(root, "authenticated_io", "manifest");
    exact_keys(io, &["input", "output"], "authenticated_io");
    let input = field(io, "input", "authenticated_io");
    exact_keys(
        input,
        &["shape", "dtype", "scale", "zero_point"],
        "authenticated_io.input",
    );
    assert_eq!(
        shape_field(input, "shape", "authenticated_io.input"),
        vec![1, 3, 40]
    );
    assert_eq!(
        string_field(input, "dtype", "authenticated_io.input"),
        "int8"
    );
    assert_eq!(
        float_field(input, "scale", "authenticated_io.input"),
        0.10196078568696976
    );
    assert_eq!(
        int_field(input, "zero_point", "authenticated_io.input"),
        -128
    );
    let output = field(io, "output", "authenticated_io");
    exact_keys(
        output,
        &["shape", "dtype", "scale", "zero_point"],
        "authenticated_io.output",
    );
    assert_eq!(
        shape_field(output, "shape", "authenticated_io.output"),
        vec![1, 1]
    );
    assert_eq!(
        string_field(output, "dtype", "authenticated_io.output"),
        "uint8"
    );
    assert_eq!(
        float_field(output, "scale", "authenticated_io.output"),
        0.00390625
    );
    assert_eq!(
        int_field(output, "zero_point", "authenticated_io.output"),
        0
    );

    let sequence = field(root, "persistent_sequence", "manifest");
    exact_keys(
        sequence,
        &[
            "invocation_count",
            "frames_per_invocation",
            "distinct_frames",
            "single_persistent_interpreter",
            "fresh_interpreter_reset_replay",
        ],
        "persistent_sequence",
    );
    assert_eq!(
        int_field(sequence, "invocation_count", "persistent_sequence"),
        INVOCATION_COUNT as i64
    );
    assert_eq!(
        int_field(sequence, "frames_per_invocation", "persistent_sequence"),
        3
    );
    assert!(bool_field(
        sequence,
        "distinct_frames",
        "persistent_sequence"
    ));
    assert!(bool_field(
        sequence,
        "single_persistent_interpreter",
        "persistent_sequence"
    ));
    let replay = field(
        sequence,
        "fresh_interpreter_reset_replay",
        "persistent_sequence",
    );
    exact_keys(
        replay,
        &["status", "invocation_count", "raw_outputs_match"],
        "fresh_interpreter_reset_replay",
    );
    assert_eq!(
        string_field(replay, "status", "fresh_interpreter_reset_replay"),
        "PASS"
    );
    assert_eq!(
        int_field(replay, "invocation_count", "fresh_interpreter_reset_replay"),
        INVOCATION_COUNT as i64
    );
    assert!(bool_field(
        replay,
        "raw_outputs_match",
        "fresh_interpreter_reset_replay"
    ));

    let topology = field(root, "tflite_topology", "manifest");
    exact_keys(
        topology,
        &[
            "input_name",
            "input_shape",
            "input_dtype",
            "input_scale",
            "input_zero_point",
            "output_name",
            "output_shape",
            "output_dtype",
            "output_scale",
            "output_zero_point",
        ],
        "tflite_topology",
    );
    let _ = string_field(topology, "input_name", "tflite_topology");
    assert_eq!(
        shape_field(topology, "input_shape", "tflite_topology"),
        vec![1, 3, 40]
    );
    assert_eq!(
        string_field(topology, "input_dtype", "tflite_topology"),
        "int8"
    );
    assert_eq!(
        float_field(topology, "input_scale", "tflite_topology"),
        0.10196078568696976
    );
    assert_eq!(
        int_field(topology, "input_zero_point", "tflite_topology"),
        -128
    );
    let _ = string_field(topology, "output_name", "tflite_topology");
    assert_eq!(
        shape_field(topology, "output_shape", "tflite_topology"),
        vec![1, 1]
    );
    assert_eq!(
        string_field(topology, "output_dtype", "tflite_topology"),
        "uint8"
    );
    assert_eq!(
        float_field(topology, "output_scale", "tflite_topology"),
        0.00390625
    );
    assert_eq!(
        int_field(topology, "output_zero_point", "tflite_topology"),
        0
    );

    let expected: Vec<(String, Vec<usize>, &str, usize)> = {
        let mut rows = vec![
            ("input_pcm".to_string(), vec![512], "int16", 1024),
            ("features_ref".to_string(), vec![40], "float32", 160),
        ];
        for index in 0..INVOCATION_COUNT {
            rows.extend([
                (
                    format!("features_invocation_{index:02}"),
                    vec![3, 40],
                    "float32",
                    480,
                ),
                (
                    format!("input_invocation_{index:02}"),
                    vec![1, 3, 40],
                    "int8",
                    120,
                ),
                (
                    format!("output_invocation_{index:02}"),
                    vec![1, 1],
                    "uint8",
                    1,
                ),
                (
                    format!("output_invocation_{index:02}_f32"),
                    vec![1, 1],
                    "float32",
                    4,
                ),
            ]);
        }
        rows.push(("output_ref".to_string(), vec![4, 1], "float32", 16));
        rows
    };
    let artefacts = field(root, "artefacts", "manifest")
        .as_array()
        .expect("artefacts must be an array");
    assert_eq!(artefacts.len(), expected.len());
    for (name, shape, dtype, bytes) in expected {
        let matching: Vec<&JsonValue> = artefacts
            .iter()
            .filter(|item| string_field(item, "name", "artefact") == name)
            .collect();
        assert_eq!(matching.len(), 1, "artefact {name} missing/duplicated");
        let item = matching[0];
        exact_keys(
            item,
            &[
                "name",
                "path",
                "shape",
                "dtype",
                "byte_order",
                "bytes",
                "sha256",
                "role",
            ],
            &format!("artefact.{name}"),
        );
        assert_eq!(
            string_field(item, "path", &format!("artefact.{name}")),
            format!("{name}.bin")
        );
        assert_eq!(
            shape_field(item, "shape", &format!("artefact.{name}")),
            shape
        );
        assert_eq!(
            string_field(item, "dtype", &format!("artefact.{name}")),
            dtype
        );
        assert_eq!(
            string_field(item, "byte_order", &format!("artefact.{name}")),
            "little-endian"
        );
        let _ = string_field(item, "role", &format!("artefact.{name}"));
        assert_eq!(
            int_field(item, "bytes", &format!("artefact.{name}")),
            bytes as i64
        );
        sha256_text(
            string_field(item, "sha256", &format!("artefact.{name}")),
            &format!("artefact.{name}.sha256"),
        );
    }
    let reference_environment = field(root, "reference_environment", "manifest");
    exact_keys(
        reference_environment,
        &["python", "system", "machine", "installed_distributions"],
        "manifest.reference_environment",
    );
    assert_eq!(
        string_field(
            reference_environment,
            "python",
            "manifest.reference_environment"
        ),
        "3.12"
    );
    assert_eq!(
        string_field(
            reference_environment,
            "system",
            "manifest.reference_environment"
        ),
        "Linux"
    );
    assert_eq!(
        string_field(
            reference_environment,
            "machine",
            "manifest.reference_environment"
        ),
        "x86_64"
    );
    verify_distribution_versions(
        field(
            reference_environment,
            "installed_distributions",
            "manifest.reference_environment",
        ),
        "manifest.reference_environment.installed_distributions",
    );
    verify_dependency_evidence(field(root, "dependency_evidence", "manifest"));
    // The outer VAST evidence gate is responsible for recomputing these
    // hashes against the files and the pinned source bytes. This local
    // contract checks their shape/type/presence without downloading data.
}

/// Future production integration contract. It intentionally has no
/// implementation until the authenticated, stateful binder is exported;
/// stateless `ChainConfig` cannot satisfy it.
#[allow(dead_code)]
trait AuthenticatedStreamingBinder {
    fn step(&mut self, quantized_input: &[i8]) -> Result<Vec<u8>, String>;
    fn reset(&mut self) -> Result<(), String>;
}

#[allow(dead_code)]
fn compare_authenticated_sequence<B: AuthenticatedStreamingBinder>(
    mut binder: B,
    inputs: &[Vec<i8>],
    expected_outputs: &[u8],
) {
    assert_eq!(inputs.len(), expected_outputs.len());
    for (input, &expected) in inputs.iter().zip(expected_outputs.iter()) {
        let output = binder
            .step(input)
            .expect("authenticated streaming step failed");
        assert_eq!(output, vec![expected], "streaming output drift");
    }
    binder
        .reset()
        .expect("authenticated streaming reset failed");
    for (input, &expected) in inputs.iter().zip(expected_outputs.iter()) {
        let output = binder.step(input).expect("streaming replay failed");
        assert_eq!(output, vec![expected], "streaming reset replay drift");
    }
}

/// Per-band `|Δ|` gate for the log-mel feature extractor parity
/// (Path B). `5e-2` is the honest architectural bound:
///
/// * ``np.fft.rfft`` (the numpy reference) computes internally in
///   float64 and casts to float32 at output.
/// * Vokra's Rust FFT is float32 throughout its log₂(N_FFT) = 9
///   butterfly stages (target-architecture-realistic for Cortex-M55).
///
/// Empirically the two agree to `< 1e-4` at low bands (0–15) but drift
/// to `~3e-2` at high bands (~30) where the f32 rounding accumulates.
/// This is a real precision gap between the higher-precision numpy
/// reference and the f32 target-realistic Rust code — not a Rust bug.
/// A regression that actually broke the FFT / filterbank / log10
/// chain would produce deltas well above `5e-2` (e.g. an FFT twiddle
/// sign flip is `~1.0`; a mel-filterbank off-by-one is `~0.1+` at
/// affected bands; a log10 floor bug is `~ln(1e-10)` at silent
/// frames). The 5e-2 atol leaves ~1.7× margin above the observed
/// baseline delta while still catching every regression class above
/// that scale. Same "honest atol from architectural bound, not from
/// CI-green wishing" posture as the Kokoro `PROSODY_F0_ATOL` calibration
/// (see `parity_kokoro.rs`).
const FEATURES_ATOL: f32 = 5e-2;

// ---------------------------------------------------------------------------
// FIXTURE-FREE tests (always run — no env vars required)
// ---------------------------------------------------------------------------

/// FIXTURE-FREE: pin the extractor's public constants against the
/// primary source (microWakeWord upstream trains at 16 kHz / 40 mels /
/// 10 ms hop / 32 ms window, verified in the sidecar's
/// `DEFAULT_SAMPLE_RATE` / `DEFAULT_HOP_MS` / `DEFAULT_WINDOW_MS` /
/// `DEFAULT_N_MELS` constants). A silent drift here would mis-align the
/// mel front-end against every real checkpoint.
#[test]
fn kws_features_constants_match_primary_source() {
    assert_eq!(
        features::SAMPLE_RATE,
        16_000,
        "microWakeWord is trained at 16 kHz PCM in"
    );
    assert_eq!(
        features::HOP_MS,
        10,
        "microWakeWord's canonical streaming hop is 10 ms"
    );
    assert_eq!(
        features::WINDOW_MS,
        32,
        "microWakeWord's canonical STFT window is 32 ms"
    );
    assert_eq!(
        features::N_MELS,
        40,
        "microWakeWord's canonical mel band count is 40"
    );
    // Derived constants must be internally consistent.
    assert_eq!(features::HOP_SAMPLES, 160, "SAMPLE_RATE * HOP_MS / 1000");
    assert_eq!(
        features::WINDOW_SAMPLES,
        512,
        "SAMPLE_RATE * WINDOW_MS / 1000"
    );
    assert_eq!(features::N_FFT, 512, "next pow-of-two >= WINDOW_SAMPLES");
    assert_eq!(features::N_BINS, 257, "N_FFT / 2 + 1");
}

// ---------------------------------------------------------------------------
// GATED tests (skip cleanly when env vars unset)
// ---------------------------------------------------------------------------

/// GATED (Path A): opens a real Vokra microWakeWord GGUF and validates
/// the metadata contract + tensor manifest lower bound.
///
/// Skips cleanly when [`GGUF_ENV`] is unset. Once set, all failures are
/// hard: a missing / malformed / wrong-arch fixture fails loudly.
#[test]
fn parity_microwakeword_gguf_smoke() {
    let Some(gguf_path) = env::var(GGUF_ENV).ok() else {
        eprintln!(
            "{GGUF_ENV} unset — skipping microWakeWord GGUF parity smoke; \
             this is a clean skip (never a fabricated pass). See the module \
             docs for the fixture recipe."
        );
        return;
    };

    let path = Path::new(&gguf_path);
    let bytes = fs::read(path).unwrap_or_else(|e| {
        panic!(
            "microWakeWord GGUF at {gguf_path} failed to read: {e:?} \
             (opted-in ⇒ any error is a hard failure — FR-EX-08)"
        )
    });
    let m = Model::from_bytes(&bytes).unwrap_or_else(|e| {
        panic!(
            "microWakeWord GGUF at {gguf_path} failed to parse: {e:?} \
             (opted-in ⇒ any error is a hard failure — FR-EX-08)"
        )
    });

    // Metadata contract: primary-source defaults must round-trip through
    // the `vokra.kws.*` chunk group. A real hey_jarvis GGUF must carry
    // 16 kHz + 40 mels (a differently-configured artefact is either
    // misconfigured or a non-canonical fork — either way, a hard failure
    // rather than a silent bind that would produce nonsense scores).
    assert_eq!(
        m.header.sample_rate,
        features::SAMPLE_RATE,
        "microWakeWord trained at 16 kHz; got {}",
        m.header.sample_rate
    );
    assert_eq!(
        m.header.n_mels as usize,
        features::N_MELS,
        "microWakeWord canonical mel width = {}; got {}",
        features::N_MELS,
        m.header.n_mels
    );
    assert!(
        (0.0..=1.0).contains(&m.header.threshold) && m.header.threshold.is_finite(),
        "threshold {} must be a finite probability in [0, 1]",
        m.header.threshold
    );
    assert!(
        !m.header.model.is_empty(),
        "model name must be non-empty (real GGUF carries e.g. 'hey_jarvis')"
    );
    assert!(
        !m.header.tflite_sha256.is_empty(),
        "tflite_sha256 must be non-empty for provenance audit"
    );

    // Tensor manifest lower bound: a real microWakeWord MC-MobileNet
    // checkpoint carries dozens of weight tensors across conv / dwconv /
    // dense layers. A one-tensor GGUF (as the from_gguf smoke fixture
    // uses) would be accepted by from_gguf but is not a real checkpoint.
    // Conservative floor of 10 tensors — the exact upstream count depends
    // on the model variant and is not primary-source-transcribable
    // without a dedicated dumper.
    assert!(
        m.tensor_count() >= 10,
        "real microWakeWord checkpoint must carry >= 10 tensors; got {} — \
         refusing a synthesized-shape fixture (FR-EX-08)",
        m.tensor_count()
    );

    eprintln!(
        "microWakeWord GGUF loaded from {gguf_path}: model={:?}, \
         sr={}, n_mels={}, threshold={}, {} tensors bound",
        m.header.model,
        m.header.sample_rate,
        m.header.n_mels,
        m.header.threshold,
        m.tensor_count(),
    );
}

/// GATED (Path B): reads the dumper's `input_pcm.bin` + `features_ref.bin`
/// and validates Vokra's [`FeatureExtractor::compute_frame_f32`] output
/// per-band `|Δ|` against the numpy reference at [`FEATURES_ATOL`].
///
/// Skips cleanly when [`FIXTURES_ENV`] is unset.
#[test]
fn parity_microwakeword_feature_extractor_matches_reference() {
    let Some(fixtures_dir) = env::var(FIXTURES_ENV).ok() else {
        eprintln!(
            "{FIXTURES_ENV} unset — skipping microWakeWord feature-extractor \
             parity; this is a clean skip (never a fabricated pass). See the \
             module docs for the fixture recipe."
        );
        return;
    };
    let dir = PathBuf::from(&fixtures_dir);

    // Read PCM input (raw i16 little-endian, WINDOW_SAMPLES samples).
    let pcm_path = dir.join("input_pcm.bin");
    let pcm_bytes = fs::read(&pcm_path).unwrap_or_else(|e| {
        panic!(
            "Path-B: failed to read {}: {e:?} — is the dumper output complete?",
            pcm_path.display()
        )
    });
    assert_eq!(
        pcm_bytes.len(),
        features::WINDOW_SAMPLES * 2,
        "Path-B: input_pcm.bin len {} != WINDOW_SAMPLES ({}) * 2 (i16 LE)",
        pcm_bytes.len(),
        features::WINDOW_SAMPLES,
    );
    let pcm: Vec<i16> = pcm_bytes
        // Exact byte length is asserted above; `chunks` keeps this fixture
        // reader compatible with the workspace's Rust 1.85 MSRV.
        .chunks(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    // Read reference features (raw f32 little-endian, N_MELS floats).
    let ref_path = dir.join("features_ref.bin");
    let ref_bytes = fs::read(&ref_path).unwrap_or_else(|e| {
        panic!(
            "Path-B: failed to read {}: {e:?} — is the dumper output complete?",
            ref_path.display()
        )
    });
    assert_eq!(
        ref_bytes.len(),
        features::N_MELS * 4,
        "Path-B: features_ref.bin len {} != N_MELS ({}) * 4 (f32 LE)",
        ref_bytes.len(),
        features::N_MELS,
    );
    let features_ref: Vec<f32> = ref_bytes
        // Exact byte length is asserted above; see the PCM reader above.
        .chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Run Vokra's feature extractor on the same PCM.
    let extractor = FeatureExtractor::new();
    let features_vokra = extractor.compute_frame_f32(&pcm);
    assert_eq!(
        features_vokra.len(),
        features::N_MELS,
        "extractor produced {} features, expected N_MELS = {}",
        features_vokra.len(),
        features::N_MELS,
    );

    // Per-band |Δ| gate. FEATURES_ATOL = 5e-2 is the registered architectural
    // bound for numpy's higher-precision FFT versus Vokra's f32 FFT path;
    // see the constant's rationale above.
    let mut max_delta = 0.0f32;
    let mut worst_band = 0usize;
    for (i, (&v, &r)) in features_vokra.iter().zip(features_ref.iter()).enumerate() {
        assert!(v.is_finite(), "band {i}: Vokra feature {v} is not finite");
        assert!(
            r.is_finite(),
            "band {i}: reference feature {r} is not finite"
        );
        let d = (v - r).abs();
        if d > max_delta {
            max_delta = d;
            worst_band = i;
        }
    }
    eprintln!(
        "Path-B log-mel parity: max |Δ| = {max_delta:.6} at band {worst_band} \
         (gate = {FEATURES_ATOL:.6}, N_MELS = {} bands)",
        features::N_MELS,
    );
    assert!(
        max_delta <= FEATURES_ATOL,
        "Path-B log-mel per-band parity failed: max |Δ| = {max_delta} at band \
         {worst_band} exceeds atol = {FEATURES_ATOL}. Vokra features[..8] = {:?}; \
         reference[..8] = {:?}. Investigate the FFT / mel-filterbank / log10 \
         chain for a regression.",
        &features_vokra[..8.min(features_vokra.len())],
        &features_ref[..8.min(features_ref.len())],
    );
}

/// GATED (Path C): authenticated streaming fixture contract.
#[test]
fn parity_microwakeword_end_to_end_output() {
    let (Some(gguf_path), Some(fixtures_dir)) =
        (env::var(GGUF_ENV).ok(), env::var(FIXTURES_ENV).ok())
    else {
        eprintln!("Path-C: {GGUF_ENV} and/or {FIXTURES_ENV} unset — skipping cleanly.");
        return;
    };
    let gguf_bytes =
        fs::read(&gguf_path).unwrap_or_else(|e| panic!("Path-C GGUF read failed: {e:?}"));
    let model = Model::from_bytes(&gguf_bytes)
        .unwrap_or_else(|e| panic!("Path-C GGUF parse failed: {e:?}"));
    assert_eq!(model.header.tflite_sha256, AUTHENTICATED_TFLITE_SHA256);
    let dir = Path::new(&fixtures_dir);
    let manifest_bytes = fs::read(dir.join("manifest.json"))
        .unwrap_or_else(|e| panic!("Path-C manifest read failed: {e:?}"));
    let manifest = json::parse(&manifest_bytes)
        .unwrap_or_else(|e| panic!("Path-C manifest JSON parse failed: {e:?}"));
    verify_reference_manifest(&manifest);

    let mut inputs: Vec<Vec<i8>> = Vec::with_capacity(INVOCATION_COUNT);
    let mut outputs: Vec<u8> = Vec::with_capacity(INVOCATION_COUNT);
    for index in 0..INVOCATION_COUNT {
        let input = fs::read(dir.join(format!("input_invocation_{index:02}.bin")))
            .unwrap_or_else(|e| panic!("Path-C input {index} read failed: {e:?}"));
        assert_eq!(
            input.len(),
            INPUT_BYTES,
            "input invocation {index} byte count"
        );
        inputs.push(input.into_iter().map(|x| x as i8).collect());
        let output = fs::read(dir.join(format!("output_invocation_{index:02}.bin")))
            .unwrap_or_else(|e| panic!("Path-C output {index} read failed: {e:?}"));
        assert_eq!(
            output.len(),
            OUTPUT_BYTES,
            "output invocation {index} byte count"
        );
        let dequant = fs::read(dir.join(format!("output_invocation_{index:02}_f32.bin")))
            .unwrap_or_else(|e| panic!("Path-C dequant output {index} read failed: {e:?}"));
        assert_eq!(dequant.len(), 4, "dequant output {index} byte count");
        let expected_f32 = output[0] as f32 * OUTPUT_SCALE;
        assert_eq!(
            f32::from_le_bytes([dequant[0], dequant[1], dequant[2], dequant[3]]),
            expected_f32
        );
        outputs.push(output[0]);
    }
    assert!(
        inputs.windows(2).any(|pair| pair[0] != pair[1]),
        "sequence must use distinct frames"
    );
    eprintln!(
        "Path-C fixture contract verified for {INVOCATION_COUNT} persistent invocations; outputs={outputs:?}"
    );

    match model.bind_authenticated_chain() {
        Err(error)
            if error
                .to_string()
                .contains("AUTHENTICATED_TOPOLOGY_REQUIRED") =>
        {
            panic!(
                "Path-C UNRESOLVED: authenticated streaming binder is not exported; fixture verification cannot be a parity PASS: {error:?}"
            );
        }
        Err(error) => panic!("Path-C authenticated binder failed unexpectedly: {error:?}"),
        Ok(_) => panic!("Path-C requires a stateful streaming binder, not stateless ChainConfig"),
    }
}
