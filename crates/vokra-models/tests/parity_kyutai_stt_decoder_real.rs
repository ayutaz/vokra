//! Authenticated Kyutai STT `dep_q=0` decoder-component parity.
//!
//! This is deliberately not an ASR test: Mimi, SentencePiece, streaming
//! delay/state, sampling, and transcription are outside this contract.
//! Reference values are produced only by the pinned official Moshi/DSM code
//! in `tools/parity/kyutai_stt_decoder_dump_reference.py`.

use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::json::JsonValue;
use vokra_models::kyutai_stt::{KyutaiSttAsr, KyutaiSttWeights};

const GGUF_ENV: &str = "VOKRA_KYUTAI_STT_DECODER_GGUF";
const GGUF_SHA_ENV: &str = "VOKRA_KYUTAI_STT_DECODER_GGUF_SHA256";
const REFERENCE_ENV: &str = "VOKRA_KYUTAI_STT_DECODER_REFERENCE";
const REFERENCE_SHA_ENV: &str = "VOKRA_KYUTAI_STT_DECODER_REFERENCE_MANIFEST_SHA256";
const HF_REPOSITORY: &str = "kyutai/stt-2.6b-en";
const HF_REVISION: &str = "a07aec56d22be5589cd0bc8709c75b6cf3e3039d";
const MODEL_BYTES: u64 = 5_234_275_128;
const MODEL_SHA256: &str = "2471add7da1fdb2d5dc4561e88a9069376333d992760d55d29d1db46c52849b2";
const CONFIG_SHA256: &str = "b79ea52a30329887a2d0ce2dd5473a63fc5083e441e7986f64f01050c06239c9";
const DSM_REVISION: &str = "4c4f65e147df056adf3346290d64c7b9649b18c9";
const MOSHI_REVISION: &str = "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362";
const FRAMES: usize = 4;
const N_Q: usize = 32;
const TEXT_CARD: usize = 4000;
const MODEL_TENSOR_MANIFEST_SHA256: &str =
    "e62488c9d16953010c758ec17f4c70e8ee30d348adfab3811eb5dfecb435d5df";
// Deliberately unset until a VAST measurement is reviewed and committed.
// Operator-provided tolerances are forbidden: a real test must not claim PASS
// against an arbitrary environment value.
const FIXED_ATOL: Option<f32> = None;

const DSM_REPOSITORY: &str = "https://github.com/kyutai-labs/delayed-streams-modeling.git";
const MOSHI_REPOSITORY: &str = "https://github.com/kyutai-labs/moshi.git";
const DSM_ROLES: &[&str] = &[
    "configs/config-stt-en-hf.toml",
    "scripts/stt_from_file_pytorch.py",
];
const MOSHI_ROLES: &[&str] = &[
    "moshi/moshi/models/lm.py",
    "moshi/moshi/models/lm_utils.py",
    "moshi/moshi/models/loaders.py",
];
const TEXT_TOKENS: [u32; FRAMES] = [3, 17, 23, 29];

fn expected_tensor_manifest() -> Vec<(String, &'static str, Vec<u64>)> {
    let mut rows = vec![("text_emb.weight".to_owned(), "BF16", vec![4001, 2048])];
    rows.extend(
        (0..N_Q).map(|channel| (format!("emb.{channel}.weight"), "BF16", vec![2049, 2048])),
    );
    for layer in 0..48 {
        let prefix = format!("transformer.layers.{layer}");
        rows.extend([
            (
                format!("{prefix}.self_attn.in_proj_weight"),
                "BF16",
                vec![6144, 2048],
            ),
            (
                format!("{prefix}.self_attn.out_proj.weight"),
                "BF16",
                vec![2048, 2048],
            ),
            (
                format!("{prefix}.gating.linear_in.weight"),
                "BF16",
                vec![11264, 2048],
            ),
            (
                format!("{prefix}.gating.linear_out.weight"),
                "BF16",
                vec![2048, 5632],
            ),
            (format!("{prefix}.norm1.alpha"), "BF16", vec![2048]),
            (format!("{prefix}.norm2.alpha"), "BF16", vec![2048]),
        ]);
    }
    rows.extend([
        ("out_norm.alpha".to_owned(), "BF16", vec![2048]),
        ("text_linear.weight".to_owned(), "BF16", vec![4000, 2048]),
    ]);
    assert_eq!(rows.len(), 323);
    rows
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
    assert!(
        path.is_file() && !path.is_symlink(),
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
    assert!(
        path.is_absolute() && path.is_dir() && !path.is_symlink(),
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

fn string(value: &JsonValue, key: &str) -> &str {
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

fn validate_source(
    value: &JsonValue,
    repository: &str,
    revision: &str,
    roles: &[&str],
    label: &str,
) {
    exact_keys(value, &["repository", "revision", "roles"], label);
    assert_eq!(string(value, "repository"), repository);
    assert_eq!(string(value, "revision"), revision);
    let rows = value
        .get("roles")
        .and_then(JsonValue::as_array)
        .expect("source roles");
    assert_eq!(rows.len(), roles.len(), "{label} role count");
    for (row, expected_path) in rows.iter().zip(roles) {
        exact_keys(
            row,
            &["path", "bytes", "sha256", "git_blob_sha1"],
            "source role",
        );
        assert_eq!(string(row, "path"), *expected_path);
        assert!(integer(row, "bytes") > 0);
        lower_hex(string(row, "sha256"), 64, "source role sha256");
        lower_hex(string(row, "git_blob_sha1"), 40, "source role git blob");
    }
}

fn tensor_manifest_digest(rows: &[(String, &str, Vec<u64>)]) -> String {
    let canonical = rows
        .iter()
        .map(|(name, dtype, shape)| {
            format!(
                "{name}\0{dtype}\0{}\n",
                shape
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect::<String>();
    digest(canonical.as_bytes())
}

fn reject_duplicate_keys(bytes: &[u8], label: &str) {
    let mut stack = Vec::<HashSet<String>>::new();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => {
                stack.push(HashSet::new());
                i += 1;
            }
            b'}' => {
                assert!(stack.pop().is_some(), "{label} malformed object");
                i += 1;
            }
            b'"' => {
                let start = i + 1;
                i = start;
                while i < bytes.len() {
                    match bytes[i] {
                        b'\\' => i += 2,
                        b'"' => break,
                        _ => i += 1,
                    }
                }
                assert!(i < bytes.len(), "{label} unterminated string");
                let key = String::from_utf8_lossy(&bytes[start..i]).into_owned();
                i += 1;
                while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                if i < bytes.len() && bytes[i] == b':' {
                    if let Some(keys) = stack.last_mut() {
                        assert!(keys.insert(key), "{label} duplicate key");
                    }
                }
            }
            _ => i += 1,
        }
    }
    assert!(stack.is_empty(), "{label} unclosed object");
}

fn digest(bytes: &[u8]) -> String {
    let mut h = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0x98b70dc6, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut padded = bytes.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&(bytes.len() as u64 * 8).to_be_bytes());
    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let a = w[i - 15];
            let b = w[i - 2];
            w[i] = w[i - 16]
                .wrapping_add(a.rotate_right(7) ^ a.rotate_right(18) ^ (a >> 3))
                .wrapping_add(w[i - 7])
                .wrapping_add(b.rotate_right(17) ^ b.rotate_right(19) ^ (b >> 10));
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut x) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let t1 = x
                .wrapping_add(e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25))
                .wrapping_add((e & f) ^ ((!e) & g))
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let t2 = (a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22))
                .wrapping_add((a & b) ^ (a & c) ^ (b & c));
            (x, g, f, e, d, c, b, a) = (g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2));
        }
        for (dst, add) in h.iter_mut().zip([a, b, c, d, e, f, g, x]) {
            *dst = dst.wrapping_add(add);
        }
    }
    h.iter().map(|v| format!("{v:08x}")).collect()
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
                            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
                    {
                        return value.to_owned();
                    }
                }
            }
        }
    }
    panic!("a streaming shasum/sha256sum implementation is required")
}

fn read_f32(path: &Path, expected: usize) -> Vec<f32> {
    let bytes = fs::read(path).expect("reference f32 artifact");
    assert_eq!(bytes.len(), expected * 4, "f32 artifact byte length");
    let values = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert!(
        values.iter().all(|value| value.is_finite()),
        "non-finite f32 artifact"
    );
    assert!(
        values.iter().any(|value| *value != 0.0),
        "vacuous zero f32 artifact"
    );
    values
}

fn authenticate_reference(
    reference: &Path,
    manifest_bytes: &[u8],
) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>) {
    required_directory(reference, "reference");
    let expected = ["hidden.f32", "input.json", "logits.f32", "manifest.json"];
    let mut names = fs::read_dir(reference)
        .expect("reference entries")
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, expected);
    let manifest_path = reference.join("manifest.json");
    let input_path = reference.join("input.json");
    let hidden_path = reference.join("hidden.f32");
    let logits_path = reference.join("logits.f32");
    for (path, label) in [
        (&manifest_path, "manifest"),
        (&input_path, "input"),
        (&hidden_path, "hidden"),
        (&logits_path, "logits"),
    ] {
        required_file(path, label);
    }
    reject_duplicate_keys(manifest_bytes, "reference manifest");
    let manifest = vokra_core::json::parse(manifest_bytes).expect("manifest JSON");
    exact_keys(
        &manifest,
        &[
            "format",
            "status",
            "component",
            "scope",
            "model",
            "sources",
            "config",
            "packet",
            "execution",
            "artifacts",
        ],
        "manifest",
    );
    assert_eq!(
        string(&manifest, "format"),
        "vokra-kyutai-stt-decoder-reference-v1"
    );
    assert_eq!(string(&manifest, "status"), "REFERENCE_READY");
    assert_eq!(string(&manifest, "component"), "decoder");
    assert_eq!(
        string(&manifest, "scope"),
        "dep_q=0 text decoder only; no Mimi/tokenizer/streaming/transcription"
    );
    let model = object(&manifest, "model");
    exact_keys(
        model,
        &[
            "repository",
            "revision",
            "path",
            "bytes",
            "sha256",
            "config_sha256",
            "mimi",
            "tokenizer",
            "tensor_manifest_sha256",
            "tensor_manifest",
        ],
        "model",
    );
    assert_eq!(string(model, "repository"), HF_REPOSITORY);
    assert_eq!(string(model, "revision"), HF_REVISION);
    assert_eq!(string(model, "path"), "model.safetensors");
    assert_eq!(integer(model, "bytes"), MODEL_BYTES);
    assert_eq!(string(model, "sha256"), MODEL_SHA256);
    assert_eq!(string(model, "config_sha256"), CONFIG_SHA256);
    let mimi = object(model, "mimi");
    exact_keys(mimi, &["path", "bytes", "sha256"], "mimi");
    assert_eq!(
        string(mimi, "path"),
        "mimi-pytorch-e351c8d8@125.safetensors"
    );
    assert_eq!(integer(mimi, "bytes"), 384_644_900);
    assert_eq!(
        string(mimi, "sha256"),
        "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50"
    );
    let tokenizer = object(model, "tokenizer");
    exact_keys(tokenizer, &["path", "bytes", "sha256"], "tokenizer");
    assert_eq!(string(tokenizer, "path"), "tokenizer_en_audio_4000.model");
    assert_eq!(integer(tokenizer, "bytes"), 59_339);
    assert_eq!(
        string(tokenizer, "sha256"),
        "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6"
    );
    let expected_tensors = expected_tensor_manifest();
    assert_eq!(
        string(model, "tensor_manifest_sha256"),
        MODEL_TENSOR_MANIFEST_SHA256
    );
    let tensor_rows = model
        .get("tensor_manifest")
        .and_then(JsonValue::as_array)
        .expect("tensor manifest");
    assert_eq!(tensor_rows.len(), expected_tensors.len());
    for (row, (name, dtype, shape)) in tensor_rows.iter().zip(&expected_tensors) {
        exact_keys(row, &["name", "dtype", "shape"], "tensor manifest row");
        assert_eq!(string(row, "name"), name);
        assert_eq!(string(row, "dtype"), *dtype);
        let actual_shape = row
            .get("shape")
            .and_then(JsonValue::as_array)
            .expect("tensor shape")
            .iter()
            .map(|v| v.as_u64().expect("tensor dimension"))
            .collect::<Vec<_>>();
        assert_eq!(&actual_shape, shape);
    }
    assert_eq!(
        tensor_manifest_digest(&expected_tensors),
        MODEL_TENSOR_MANIFEST_SHA256
    );
    let sources = object(&manifest, "sources");
    exact_keys(sources, &["dsm", "moshi"], "sources");
    validate_source(
        sources.get("dsm").unwrap(),
        DSM_REPOSITORY,
        DSM_REVISION,
        DSM_ROLES,
        "dsm source",
    );
    validate_source(
        sources.get("moshi").unwrap(),
        MOSHI_REPOSITORY,
        MOSHI_REVISION,
        MOSHI_ROLES,
        "moshi source",
    );
    let config = object(&manifest, "config");
    exact_keys(
        config,
        &[
            "n_q",
            "dep_q",
            "d_model",
            "text_card",
            "audio_card",
            "tensor_count",
        ],
        "config",
    );
    assert_eq!(integer(config, "n_q"), N_Q as u64);
    assert_eq!(integer(config, "dep_q"), 0);
    assert_eq!(integer(config, "d_model"), 2048);
    assert_eq!(integer(config, "text_card"), TEXT_CARD as u64);
    assert_eq!(integer(config, "audio_card"), 2048);
    assert_eq!(integer(config, "tensor_count"), 323);
    let packet = object(&manifest, "packet");
    exact_keys(packet, &["text_tokens", "mimi_codes"], "packet");
    let text = packet
        .get("text_tokens")
        .and_then(JsonValue::as_array)
        .expect("text packet")
        .iter()
        .map(|v| v.as_u64().expect("text token") as u32)
        .collect::<Vec<_>>();
    let codes = packet
        .get("mimi_codes")
        .and_then(JsonValue::as_array)
        .expect("audio packet")
        .iter()
        .flat_map(|row| {
            row.as_array()
                .expect("audio row")
                .iter()
                .map(|v| v.as_u64().expect("audio code") as u32)
        })
        .collect::<Vec<_>>();
    assert_eq!(text.len(), FRAMES);
    assert_eq!(codes.len(), FRAMES * N_Q);
    assert!(codes.iter().all(|v| *v < 2048));
    let execution = object(&manifest, "execution");
    exact_keys(
        execution,
        &[
            "implementation",
            "dtype",
            "device",
            "python_version",
            "torch_version",
            "num_threads",
            "num_interop_threads",
            "deterministic_algorithms",
            "publication",
        ],
        "execution",
    );
    assert_eq!(
        string(execution, "implementation"),
        "official Moshi LMModel.forward_text"
    );
    assert_eq!(string(execution, "dtype"), "F32");
    assert_eq!(string(execution, "device"), "cpu");
    assert_eq!(string(execution, "python_version"), "3.12");
    assert_eq!(string(execution, "torch_version"), "2.13.0");
    assert_eq!(integer(execution, "num_threads"), 1);
    assert_eq!(integer(execution, "num_interop_threads"), 1);
    assert!(matches!(
        execution.get("deterministic_algorithms"),
        Some(JsonValue::Bool(true))
    ));
    assert_eq!(string(execution, "publication"), "NO_UPLOAD");
    let artifacts = object(&manifest, "artifacts");
    exact_keys(
        artifacts,
        &["input.json", "hidden.f32", "logits.f32"],
        "artifacts",
    );
    for (name, path, dtype, role) in [
        ("input.json", &input_path, "json", "input"),
        ("hidden.f32", &hidden_path, "f32-le", "diagnostics-only"),
        ("logits.f32", &logits_path, "f32-le", "parity"),
    ] {
        let row = object(artifacts, name);
        exact_keys(row, &["bytes", "sha256", "dtype", "role"], "artifact");
        assert_eq!(integer(row, "bytes"), fs::metadata(path).unwrap().len());
        assert_eq!(string(row, "sha256"), file_digest(path));
        assert_eq!(string(row, "dtype"), dtype);
        assert_eq!(string(row, "role"), role);
    }
    let input_bytes = fs::read(&input_path).expect("input JSON bytes");
    reject_duplicate_keys(&input_bytes, "input JSON");
    let input = vokra_core::json::parse(&input_bytes).expect("input JSON");
    exact_keys(&input, &["text_tokens", "mimi_codes"], "input");
    let input_text = input
        .get("text_tokens")
        .and_then(JsonValue::as_array)
        .expect("input text tokens");
    assert_eq!(input_text.len(), FRAMES);
    for (value, expected) in input_text.iter().zip(TEXT_TOKENS) {
        assert_eq!(value.as_u64(), Some(expected as u64));
    }
    let input_codes = input
        .get("mimi_codes")
        .and_then(JsonValue::as_array)
        .expect("input mimi codes");
    assert_eq!(input_codes.len(), FRAMES);
    for (row, expected_row) in input_codes.iter().zip(codes.chunks_exact(N_Q)) {
        let values = row.as_array().expect("input mimi row");
        assert_eq!(values.len(), N_Q);
        for (value, expected) in values.iter().zip(expected_row) {
            assert_eq!(value.as_u64(), Some(*expected as u64));
        }
    }
    let hidden = read_f32(&hidden_path, FRAMES * 2048);
    let logits = read_f32(&logits_path, FRAMES * TEXT_CARD);
    (text, codes, hidden, logits)
}

fn compare(actual: &[f32], expected: &[f32], atol: Option<f32>) -> (f32, usize) {
    assert_eq!(actual.len(), expected.len());
    let mut max = 0.0;
    let mut argmax = 0;
    for (a, e) in actual.iter().zip(expected) {
        let d = (*a - *e).abs();
        assert!(d.is_finite());
        max = max.max(d);
    }
    for row in 0..FRAMES {
        let ar = &actual[row * TEXT_CARD..(row + 1) * TEXT_CARD];
        let er = &expected[row * TEXT_CARD..(row + 1) * TEXT_CARD];
        let ai = ar
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let ei = er
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        if ai == ei {
            argmax += 1;
        }
    }
    if let Some(atol) = atol {
        assert!(max <= atol, "decoder max delta {max} > {atol}");
    }
    (max, argmax)
}

struct AuthenticatedCase {
    gguf: PathBuf,
    text: Vec<u32>,
    codes: Vec<u32>,
    reference_logits: Vec<f32>,
}

fn authenticate_case(apple: bool) -> AuthenticatedCase {
    if apple {
        assert!(cfg!(target_os = "macos") && cfg!(target_arch = "aarch64"));
        assert_eq!(
            std::env::var("VOKRA_REMOTE_APPLE_SILICON").as_deref(),
            Ok("1")
        );
    }
    let gguf = required_path(GGUF_ENV);
    required_file(&gguf, "decoder GGUF");
    let gguf_sha = std::env::var(GGUF_SHA_ENV).expect("GGUF SHA");
    lower_hex(&gguf_sha, 64, "GGUF");
    assert_eq!(file_digest(&gguf), gguf_sha);
    let reference = required_path(REFERENCE_ENV);
    require_disjoint(&[
        (&gguf, "decoder GGUF"),
        (&reference, "reference"),
        (
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(Path::parent)
                .expect("checkout root"),
            "checkout",
        ),
    ]);
    let reference_sha = std::env::var(REFERENCE_SHA_ENV).expect("reference SHA");
    lower_hex(&reference_sha, 64, "reference manifest");
    let manifest_path = reference.join("manifest.json");
    required_file(&manifest_path, "reference manifest");
    let manifest_bytes = fs::read(&manifest_path).expect("reference manifest bytes");
    assert_eq!(digest(&manifest_bytes), reference_sha);
    let (text, codes, _hidden, reference_logits) =
        authenticate_reference(&reference, &manifest_bytes);
    AuthenticatedCase {
        gguf,
        text,
        codes,
        reference_logits,
    }
}

fn forward_case(case: &AuthenticatedCase, backend: BackendKind) -> Vec<f32> {
    let file = GgufFile::open(&case.gguf).expect("open decoder GGUF");
    assert_eq!(
        file.get(vokra_core::gguf::chunks::KEY_PROVENANCE_WEIGHT_LICENSE),
        Some(&GgufMetadataValue::String(
            "attribution-required".to_owned()
        ))
    );
    assert_eq!(
        file.get(vokra_core::gguf::chunks::KEY_PROVENANCE_LICENSE),
        Some(&GgufMetadataValue::String("cc-by-4.0".to_owned()))
    );
    let weights = KyutaiSttWeights::from_component_gguf(&file)
        .expect("bind exact 323-tensor decoder artifact");
    let asr = KyutaiSttAsr::new(
        vokra_models::kyutai_stt::KyutaiSttConfig::stt_2_6b_en(),
        weights,
    )
    .expect("decoder engine");
    let output = asr
        .forward_text_logits(backend, &case.text, &case.codes)
        .expect("decoder logits");
    assert_eq!(output.frames(), FRAMES);
    assert_eq!(output.vocab(), TEXT_CARD);
    assert!(output.as_slice().iter().all(|value| value.is_finite()));
    assert!(output.as_slice().iter().any(|value| *value != 0.0));
    output.as_slice().to_vec()
}

fn run(backend: BackendKind, apple: bool) {
    let measurement_only =
        std::env::var("VOKRA_KYUTAI_STT_DECODER_MEASUREMENT_ONLY").as_deref() == Ok("1");
    let atol = match FIXED_ATOL {
        Some(value) => Some(value),
        None if measurement_only => None,
        None => panic!(
            "Kyutai STT decoder parity is unavailable until the first VAST measurement is reviewed"
        ),
    };
    let case = authenticate_case(apple);
    let output = forward_case(&case, backend);
    let (max, argmax) = compare(&output, &case.reference_logits, atol);
    match atol {
        Some(atol) => eprintln!(
            "KYUTAI_STT_DECODER_PARITY backend={backend:?} max_abs={max:.9e} argmax={argmax}/{FRAMES} atol={atol:.9e} verdict=PASS"
        ),
        None => eprintln!(
            "KYUTAI_STT_DECODER_MEASUREMENT backend={backend:?} max_abs={max:.9e} argmax={argmax}/{FRAMES} verdict=MEASUREMENT_ONLY"
        ),
    }
}

#[test]
#[ignore = "requires VAST-authenticated 323-tensor decoder GGUF, official Moshi reference, and reviewed fixed bound"]
fn parity_kyutai_stt_decoder_real_cpu() {
    run(BackendKind::Cpu, false);
}

#[test]
#[ignore = "requires Apple Silicon, VAST-authenticated 323-tensor decoder GGUF, official reference, and reviewed fixed bound"]
fn parity_kyutai_stt_decoder_real_apple_cpu_metal() {
    assert!(cfg!(target_os = "macos") && cfg!(target_arch = "aarch64"));
    assert!(
        FIXED_ATOL.is_some(),
        "Apple parity unavailable until fixed bound review"
    );
    let case = authenticate_case(true);
    let cpu = forward_case(&case, BackendKind::Cpu);
    let metal = forward_case(&case, BackendKind::Metal);
    let (cpu_max, cpu_argmax) = compare(&cpu, &case.reference_logits, FIXED_ATOL);
    let (metal_max, metal_argmax) = compare(&metal, &case.reference_logits, FIXED_ATOL);
    let (_, cross_argmax) = compare(&metal, &cpu, FIXED_ATOL);
    eprintln!(
        "KYUTAI_STT_DECODER_APPLE cpu_max_abs={cpu_max:.9e} metal_max_abs={metal_max:.9e} cpu_argmax={cpu_argmax}/{FRAMES} metal_argmax={metal_argmax}/{FRAMES} metal_cpu_argmax={cross_argmax}/{FRAMES} verdict=PASS"
    );
}

#[test]
fn decoder_reference_contract_constants_are_fixed() {
    assert_eq!(
        digest(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        digest(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    reject_raw_path("/tmp//decoder", "path");
    assert!(std::panic::catch_unwind(|| reject_raw_path("/tmp/./decoder", "path")).is_err());
    assert!(
        std::panic::catch_unwind(|| require_disjoint(&[
            (Path::new("/tmp"), "a"),
            (Path::new("/tmp"), "b")
        ]))
        .is_err()
    );
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("checkout root");
    let nested = checkout.join("crates");
    assert!(
        std::panic::catch_unwind(|| require_disjoint(&[
            (checkout, "root"),
            (nested.as_path(), "nested")
        ]))
        .is_err()
    );
}
