//! VAST-only CosyVoice2 LLM parity against the official Transformers fixture.
//!
//! This integration test is ignored by default because opening the standalone
//! GGUF reads its multi-gigabyte payload. When opted in, every input is
//! authenticated: the reference manifest, fixed token IDs, artifact digests,
//! and the exact Qwen2 axes are all required before comparing logits.

use std::path::{Path, PathBuf};
use std::process::Command;

use vokra_core::gguf::{GgufFile, chunks};
use vokra_models::cosyvoice2::llm::{LlmBackbone, LlmBackboneConfig, parity};

const GGUF_ENV: &str = "VOKRA_COSYVOICE2_LLM_COMPONENT_GGUF";
const REFERENCE_ENV: &str = "VOKRA_COSYVOICE2_LLM_REFERENCE";
const MODEL_REPOSITORY: &str = "FunAudioLLM/CosyVoice2-0.5B";
const MODEL_REVISION: &str = "eec1ae6c79877dbd9379285cf8789c9e0879293d";
const MODEL_PATH: &str = "llm.pt";
const MODEL_BYTES: u64 = 2_023_316_821;
const MODEL_SHA256: &str = "b144ef55b51ce8cfb79a73c90dbba0bdaba4e451c0ebcfab20f769264f84a608";
const MANIFEST_SHA256: &str = "07cf10ae088c27a7c88e1c08fb231d00b01bba0c13f312a74d2fd4b35403bda2";
const SOURCE_REPOSITORY: &str = "https://github.com/FunAudioLLM/CosyVoice.git";
const TENSOR_COUNT: usize = 295;
const ATOL: f32 = 3e-4;
const TOKEN_IDS: &[u32] = &[
    151_643, 785, 3_974, 13_876, 38_835, 34_208, 916, 279, 15_678, 5_562, 13,
];

fn required_file(path: &Path, label: &str) {
    assert!(path.is_absolute(), "{label} must be an absolute path");
    for ancestor in path.ancestors().skip(1) {
        if let Ok(metadata) = std::fs::symlink_metadata(ancestor) {
            assert!(
                !metadata.file_type().is_symlink(),
                "{label} has a symlinked ancestor: {ancestor:?}"
            );
        }
    }
    let metadata = std::fs::symlink_metadata(path)
        .unwrap_or_else(|error| panic!("{label} is unreadable: {error}"));
    assert!(
        metadata.file_type().is_file(),
        "{label} must be a regular file"
    );
    assert!(
        !metadata.file_type().is_symlink(),
        "{label} must not be a symlink"
    );
}

fn required_path(name: &str) -> PathBuf {
    let value = std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"));
    PathBuf::from(value)
}

fn json_string<'a>(value: &'a vokra_core::json::JsonValue, key: &str) -> &'a str {
    value
        .get(key)
        .and_then(vokra_core::json::JsonValue::as_str)
        .unwrap_or_else(|| panic!("manifest key {key} is missing or not a string"))
}

fn json_u64(value: &vokra_core::json::JsonValue, key: &str) -> u64 {
    value
        .get(key)
        .and_then(vokra_core::json::JsonValue::as_u64)
        .unwrap_or_else(|| panic!("manifest key {key} is missing or not UINT64"))
}

fn json_bool(value: &vokra_core::json::JsonValue, key: &str) -> bool {
    match value.get(key) {
        Some(vokra_core::json::JsonValue::Bool(value)) => *value,
        _ => panic!("manifest key {key} is missing or not BOOL"),
    }
}

fn json_object<'a>(
    value: &'a vokra_core::json::JsonValue,
    key: &str,
) -> &'a vokra_core::json::JsonValue {
    value
        .get(key)
        .and_then(|entry| entry.as_object().map(|_| entry))
        .unwrap_or_else(|| panic!("manifest key {key} is missing or not an object"))
}

fn sha256_file(path: &Path) -> String {
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .unwrap_or_else(|error| panic!("sha256sum {path:?}: {error}"));
    assert!(output.status.success(), "sha256sum failed for {path:?}");
    String::from_utf8(output.stdout)
        .expect("sha256sum output is UTF-8")
        .split_whitespace()
        .next()
        .expect("sha256sum output has a digest")
        .to_owned()
}

fn authenticate_reference(reference: &Path) -> (Vec<u32>, Vec<f32>, (usize, usize)) {
    assert!(
        reference.is_absolute(),
        "reference directory must be absolute"
    );
    assert!(
        !reference.is_symlink() && reference.is_dir(),
        "reference directory must be real"
    );
    let manifest_path = reference.join("manifest.json");
    let token_path = reference.join("token-ids.json");
    let logits_path = reference.join("true_hf_logits.npy");
    let diagnostics_path = reference.join("diagnostics.json");
    required_file(&manifest_path, "reference manifest");
    required_file(&token_path, "token IDs");
    required_file(&logits_path, "reference logits");
    required_file(&diagnostics_path, "reference diagnostics");

    let manifest =
        vokra_core::json::parse(&std::fs::read(&manifest_path).expect("reference manifest bytes"))
            .expect("reference manifest JSON");
    assert_eq!(
        json_string(&manifest, "format"),
        "vokra-cosyvoice2-llm-reference-v1"
    );
    assert_eq!(json_string(&manifest, "status"), "REFERENCE_READY");
    assert_eq!(json_string(&manifest, "component"), "llm");
    let model = json_object(&manifest, "model");
    assert_eq!(json_string(model, "repository"), MODEL_REPOSITORY);
    assert_eq!(json_string(model, "revision"), MODEL_REVISION);
    assert_eq!(json_string(model, "path"), MODEL_PATH);
    assert_eq!(
        model.get("bytes").and_then(|value| value.as_u64()),
        Some(MODEL_BYTES)
    );
    assert_eq!(json_string(model, "sha256"), MODEL_SHA256);
    let execution = json_object(&manifest, "execution");
    assert_eq!(
        json_string(execution, "implementation"),
        "transformers.Qwen2ForCausalLM"
    );
    assert_eq!(json_string(execution, "attention"), "eager");
    assert_eq!(json_string(execution, "dtype"), "F32");
    assert_eq!(json_string(execution, "model_execution"), "RUN");
    assert_eq!(json_u64(execution, "threads"), 1);
    assert!(json_bool(execution, "deterministic_algorithms"));
    assert_eq!(json_string(execution, "publication"), "NO_UPLOAD");
    let state = json_object(&manifest, "state");
    assert_eq!(json_u64(state, "tensor_count"), TENSOR_COUNT as u64);
    assert_eq!(json_string(state, "manifest_sha256"), MANIFEST_SHA256);
    let artifacts = json_object(&manifest, "artifacts");
    for (name, artifact_path) in [
        ("true_hf_logits.npy", &logits_path),
        ("token-ids.json", &token_path),
        ("diagnostics.json", &diagnostics_path),
    ] {
        let artifact = json_object(artifacts, name);
        assert_eq!(
            artifact.get("bytes").and_then(|value| value.as_u64()),
            Some(std::fs::metadata(artifact_path).unwrap().len())
        );
        assert_eq!(json_string(artifact, "sha256"), sha256_file(artifact_path));
    }

    let token_root = vokra_core::json::parse(&std::fs::read(&token_path).expect("token IDs bytes"))
        .expect("token IDs JSON");
    let ids = token_root
        .get("ids")
        .and_then(vokra_core::json::JsonValue::as_array)
        .expect("token IDs array");
    let ids = ids
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .expect("u32 token ID")
        })
        .collect::<Vec<_>>();
    assert_eq!(ids, TOKEN_IDS, "reference token IDs drifted");

    read_npy_f32_2d(&logits_path)
        .map(|(shape, values)| (ids, values, shape))
        .unwrap_or_else(|error| panic!("reference logits: {error}"))
}

fn read_npy_f32_2d(path: &Path) -> Result<((usize, usize), Vec<f32>), String> {
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    if bytes.len() < 10 || bytes.get(..6).map(|value| value == b"\x93NUMPY") != Some(true) {
        return Err("bad NumPy magic".to_owned());
    }
    let (header_start, header_len) = match bytes[6] {
        1 => (10usize, u16::from_le_bytes([bytes[8], bytes[9]]) as usize),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err("truncated NumPy header".to_owned());
            }
            (
                12usize,
                u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            )
        }
        version => return Err(format!("unsupported NumPy version {version}")),
    };
    let header_end = header_start
        .checked_add(header_len)
        .ok_or("header overflow")?;
    let header = std::str::from_utf8(
        bytes
            .get(header_start..header_end)
            .ok_or("truncated header")?,
    )
    .map_err(|error| error.to_string())?;
    if !header.contains("'descr': '<f4'") || !header.contains("'fortran_order': False") {
        return Err("reference must be little-endian C-order F32".to_owned());
    }
    let shape = header.split("'shape':").nth(1).ok_or("shape is missing")?;
    let shape = shape
        .split_once('(')
        .and_then(|(_, rest)| rest.split_once(')'))
        .ok_or("malformed shape")?
        .0
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if shape.len() != 2 {
        return Err(format!("expected rank-2 logits, got {shape:?}"));
    }
    let elements = shape[0].checked_mul(shape[1]).ok_or("shape overflow")?;
    let payload = bytes.get(header_end..).ok_or("missing payload")?;
    if payload.len() != elements.checked_mul(4).ok_or("payload overflow")? {
        return Err("NumPy payload size mismatch".to_owned());
    }
    let values = payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err("reference logits contain non-finite values".to_owned());
    }
    Ok(((shape[0], shape[1]), values))
}

#[test]
#[ignore = "requires VAST CosyVoice2 LLM GGUF and official reference artifacts"]
fn parity_cosyvoice2_llm_component_real_transformers() {
    let gguf_path = required_path(GGUF_ENV);
    required_file(&gguf_path, "CosyVoice2 LLM GGUF");
    let reference = required_path(REFERENCE_ENV);
    let (token_ids, reference_logits, (rows, vocab)) = authenticate_reference(&reference);
    assert_eq!(rows, token_ids.len());

    let file = GgufFile::open(&gguf_path).expect("open CosyVoice2 LLM GGUF");
    assert_eq!(file.tensors().len(), TENSOR_COUNT);
    assert_eq!(
        file.get(chunks::KEY_MODEL_ARCH)
            .and_then(|value| value.as_str()),
        Some("cosyvoice2")
    );
    for (key, expected) in [
        ("vokra.model.name", "cosyvoice2-0.5b"),
        ("vokra.model.category", "llm"),
        ("vokra.cosyvoice2.component", "llm"),
        ("vokra.cosyvoice2.composite_status", "INSPECTION_ONLY"),
        ("vokra.provenance.weight_license", "permissive"),
        ("vokra.provenance.license", "apache-2.0"),
        ("vokra.provenance.model_id", "cosyvoice2-0.5b"),
        ("vokra.provenance.upstream_hf", MODEL_REPOSITORY),
        ("vokra.provenance.upstream_revision", MODEL_REVISION),
        ("vokra.provenance.source", SOURCE_REPOSITORY),
        ("vokra.cosyvoice2.upstream_component.file", MODEL_PATH),
        ("vokra.cosyvoice2.upstream_component.sha256", MODEL_SHA256),
        ("vokra.cosyvoice2.tensor_manifest_sha256", MANIFEST_SHA256),
        (
            "vokra.cosyvoice2.prepared_input.authentication_status",
            "PREPARED_INPUT_DIGEST_RECORDED_NOT_PINNED",
        ),
    ] {
        assert_eq!(
            file.get(key).and_then(|value| value.as_str()),
            Some(expected),
            "GGUF metadata {key}"
        );
    }
    assert_eq!(
        file.get("vokra.cosyvoice2.upstream_component.bytes"),
        Some(&vokra_core::gguf::GgufMetadataValue::U32(
            MODEL_BYTES as u32
        ))
    );
    let prepared_sha = file
        .get("vokra.cosyvoice2.prepared_input.sha256")
        .and_then(|value| value.as_str())
        .expect("prepared input SHA-256 metadata");
    assert_eq!(prepared_sha.len(), 64);
    assert!(
        prepared_sha
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    );
    assert!(json_u64_from_gguf(&file, "vokra.cosyvoice2.prepared_input.bytes") > 0);
    let config = LlmBackboneConfig {
        vocab_size: 151_936,
        hidden_dim: 896,
        n_layer: 24,
        n_head_q: 14,
        n_head_kv: 2,
        ffn_dim: 4_864,
        rope_base: 1_000_000.0,
        rms_norm_eps: 1.0e-6,
        n_ctx: 32_768,
    };
    assert_eq!(config.vocab_size, vocab);
    let backbone = LlmBackbone::from_gguf(&file, &config).expect("bind real GGUF backbone");
    assert!(!backbone.weights().is_synthesized);
    let report = parity::assert_vs_hf_reference(&backbone, &token_ids, &reference_logits, ATOL)
        .expect("official Transformers parity");
    assert_eq!(report.t, rows);
    assert_eq!(report.vocab, vocab);
    assert_eq!(report.argmax_matches, rows);
    eprintln!(
        "CosyVoice2 LLM parity PASS: rows={} vocab={} max_abs={:.6e} mean_abs={:.6e} argmax={}/{} atol={ATOL:.1e}",
        report.t,
        report.vocab,
        report.max_abs_delta,
        report.mean_abs_delta,
        report.argmax_matches,
        report.t,
    );
}

fn json_u64_from_gguf(file: &GgufFile, key: &str) -> u64 {
    match file.get(key) {
        Some(vokra_core::gguf::GgufMetadataValue::U64(value)) => *value,
        Some(value) => panic!("GGUF metadata {key} is not UINT64: {value:?}"),
        None => panic!("GGUF metadata {key} is missing"),
    }
}
