//! VAST-only real-weight OmniASR-CTC-1B parity.
//!
//! This test is ignored by default because it consumes the pinned 1B GGUF and
//! an independent official fairseq2 reference packet.  When enabled, both
//! paths must be configured; partial configuration, malformed/duplicate JSON,
//! missing files, digest drift, non-finite values, encoder/logit shape drift,
//! or token mismatch fail closed.  No synthetic fixture is accepted.

use std::path::{Path, PathBuf};

use vokra_core::VokraError;
use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::json::{self, JsonValue};
use vokra_models::omniasr_ctc::{OMNIASR_CTC_SAMPLE_RATE, OmniasrCtcAsr};

const ATOL: f32 = 0.01;
const MODEL_ID: &str = "facebook/omniASR-CTC-1B";
const HF_REVISION: &str = "8c22e3ffdaa4aab6431b128b84b991a7d9c2515c";
const CHECKPOINT_SHA256: &str = "e8564fa59dab7caedbcdb54ab7fb9bd6c96989f4d19add2ad81ddd969716952c";
const PREPARED_SHA256: &str = "cda8d7dd7cad2a0361b6946c42342b85ef7b0a8d672b99631dc75b4c3123dbc5";
const OMNI_REPOSITORY: &str = "https://github.com/facebookresearch/omnilingual-asr";
const OMNI_REVISION: &str = "a7fb36017a46eee8953f76bd628c174d51aefeef";
const FAIRSEQ2_REPOSITORY: &str = "https://github.com/facebookresearch/fairseq2";
const FAIRSEQ2_REVISION: &str = "8ae890e1b4d3e36307d0ba5fb695f0fc4815ecca";
const OMNI_SOURCE_FILES: &[(&str, &str)] = &[
    (
        "src/omnilingual_asr/cards/models/rc_models.yaml",
        "7c9a28b2a111f2e088a5b2be161dd68686a810cd7462241209c2c5e8a81a2913",
    ),
    (
        "src/omnilingual_asr/datasets/utils/audio.py",
        "e4a36129233325f95ab342939ad294fe37ac4eadaff6366524d60dc7ab8ea69e",
    ),
    (
        "src/omnilingual_asr/models/wav2vec2_asr/config.py",
        "94ee297b4ebb122967631d2739b329e3b0d8432e9bf4a63306e085834e382ff1",
    ),
    (
        "src/omnilingual_asr/models/wav2vec2_ssl/config.py",
        "550c6840b9b594226959948b4a48eb0e696171e9c5ac4fc070a9ea2c3d346414",
    ),
];
const FAIRSEQ2_SOURCE_FILES: &[(&str, &str)] = &[
    (
        "src/fairseq2/models/wav2vec2/config.py",
        "e75143abfa8e208f2291258949c1af7875087514113c0c370fa915b56905bd22",
    ),
    (
        "src/fairseq2/models/wav2vec2/factory.py",
        "de7bbbd70cf06eb99fb363ecd641b13825c50c66fb1694d1f3a866e722523b5a",
    ),
    (
        "src/fairseq2/models/wav2vec2/feature_extractor.py",
        "37ccd7f2209f0cab58cdd9766f71dc5425a1a42399fc9fa4ebef094694427ec9",
    ),
    (
        "src/fairseq2/models/wav2vec2/position_encoder.py",
        "630941cb76bd77fe383e027be872004f2bbc7666c5f4e4619ef7cd16795280f6",
    ),
    (
        "src/fairseq2/models/wav2vec2/frontend.py",
        "80b43735da89510df292fd7c97b0ff32fdbc52431802f6c01b6ebd8b45ed73cc",
    ),
    (
        "src/fairseq2/models/wav2vec2/asr/config.py",
        "4e199ebe027239d23b6351251b997877ed1e67b0bf854930c3b7e9afbc6f1f3c",
    ),
    (
        "src/fairseq2/models/wav2vec2/asr/factory.py",
        "60a59c2f63ac14707565016e034bb729c3ee91973d076b41189bd13173119c16",
    ),
    (
        "src/fairseq2/models/wav2vec2/asr/model.py",
        "42a9bc0f9d11eb88a1848827468b692c972107b7fe3068fcbffa7844a25a1f38",
    ),
    (
        "src/fairseq2/models/transformer/encoder.py",
        "b828efb95036e32865e32f79da9178c1a3dff204c5448194fed52a6c07ba7352",
    ),
    (
        "src/fairseq2/models/transformer/encoder_layer.py",
        "389e6a49c54680a30ff09c3fc1d23c37fd1465f6772b42e25b9da59d8411acfd",
    ),
    (
        "src/fairseq2/models/transformer/ffn.py",
        "c2e60872d4c1500bdc4767d032ab7dc7b0e9d4881ef3c4fe6cfa6b4ca7d321cd",
    ),
    (
        "src/fairseq2/models/transformer/multihead_attention.py",
        "35b54f73e71b052160a0ca0baca998af5543a711ed297ec85d9a5ea7d32f552c",
    ),
    (
        "src/fairseq2/models/transformer/norm_order.py",
        "1c1d4a191707291e8123423b2ac999a2c2f7e71690c238f7b0ce1cf0dc8080c0",
    ),
    (
        "src/fairseq2/models/transformer/sdpa/base.py",
        "2e004badcbf3be84cbc0e74a395c873f0d4febd038b130769e9a29d1f7c1c549",
    ),
    (
        "src/fairseq2/models/transformer/sdpa/default.py",
        "0bb33d8f2fbf7063bc3402ad9e5a5a4c94ea2b08a04282a64300ed8de2451e8b",
    ),
    (
        "src/fairseq2/models/transformer/sdpa/torch.py",
        "b1aa6d3ac26d225a2d7e18bf023615f9aa8538de3e903d1db6f57fe30af1fb61",
    ),
    (
        "src/fairseq2/nn/normalization.py",
        "f8f019e06d7d39040ef394cc292e58ed88a704c215fcac3afe7d9cfc028de158",
    ),
    (
        "src/fairseq2/nn/projection.py",
        "14d625c9ad142e2e148e23ef5479ec29538a48f2a1e8534704d02162d096e052",
    ),
];

fn paths() -> (PathBuf, PathBuf) {
    let gguf = std::env::var_os("VOKRA_OMNIASR_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("VOKRA_OMNIASR_GGUF is required for real OmniASR parity"));
    let reference = std::env::var_os("VOKRA_OMNIASR_REFERENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!("VOKRA_OMNIASR_REFERENCE_DIR is required for real OmniASR parity")
        });
    (gguf, reference)
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "{} is not f32-aligned", path.display());
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert!(!values.is_empty(), "{} is empty", path.display());
    assert!(
        values.iter().all(|v| v.is_finite()),
        "{} is non-finite",
        path.display()
    );
    values
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "{} is not u32-aligned", path.display());
    let values: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert!(!values.is_empty(), "{} is empty", path.display());
    assert!(
        values.iter().all(|v| *v < 9812),
        "token id exceeds Omni vocab"
    );
    values
}

fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
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
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() + 8) % 64 != 0 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, bytes) in chunk.chunks_exact(4).take(16).enumerate() {
            w[i] = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut i) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for t in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = i
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[t])
                .wrapping_add(w[t]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            (i, g, f, e, d, c, b, a) = (
                g,
                f,
                e,
                d.wrapping_add(temp1),
                c,
                b,
                a,
                temp1.wrapping_add(temp2),
            );
        }
        for (dst, add) in h.iter_mut().zip([a, b, c, d, e, f, g, i]) {
            *dst = dst.wrapping_add(add);
        }
    }
    h.iter().map(|v| format!("{v:08x}")).collect()
}

fn no_duplicate_objects(value: &JsonValue) {
    match value {
        JsonValue::Object(entries) => {
            let mut seen = std::collections::BTreeSet::new();
            for (key, child) in entries {
                assert!(seen.insert(key), "duplicate JSON key `{key}`");
                no_duplicate_objects(child);
            }
        }
        JsonValue::Array(items) => items.iter().for_each(no_duplicate_objects),
        _ => {}
    }
}

fn field<'a>(value: &'a JsonValue, key: &str) -> &'a JsonValue {
    value
        .get(key)
        .unwrap_or_else(|| panic!("manifest missing `{key}`"))
}

fn string_field(value: &JsonValue, key: &str) -> String {
    field(value, key)
        .as_str()
        .unwrap_or_else(|| panic!("manifest `{key}` is not a string"))
        .to_owned()
}

fn usize_field(value: &JsonValue, key: &str) -> usize {
    field(value, key)
        .as_u64()
        .unwrap_or_else(|| panic!("manifest `{key}` is not a non-negative integer")) as usize
}

fn f64_field(value: &JsonValue, key: &str) -> f64 {
    match field(value, key) {
        JsonValue::Float(number) => *number,
        JsonValue::Int(number) => *number as f64,
        _ => panic!("manifest `{key}` is not a number"),
    }
}

fn exact_object_keys(value: &JsonValue, expected: &[&str], label: &str) {
    let object = value
        .as_object()
        .unwrap_or_else(|| panic!("manifest `{label}` is not an object"));
    let actual: std::collections::BTreeSet<&str> =
        object.iter().map(|(key, _)| key.as_str()).collect();
    let wanted: std::collections::BTreeSet<&str> = expected.iter().copied().collect();
    assert_eq!(actual, wanted, "manifest `{label}` keys differ");
}

fn exact_source_files(value: &JsonValue, expected: &[(&str, &str)], label: &str) {
    exact_object_keys(value, &["repository", "revision", "files"], label);
    let rows = field(value, "files")
        .as_array()
        .unwrap_or_else(|| panic!("manifest `{label}.files` is not an array"));
    assert_eq!(rows.len(), expected.len(), "manifest `{label}.files` count");
    let mut seen = std::collections::BTreeSet::new();
    for row in rows {
        exact_object_keys(row, &["path", "sha256", "bytes"], "source file row");
        let path = string_field(row, "path");
        let digest = string_field(row, "sha256");
        assert_eq!(digest.len(), 64, "source digest length: {path}");
        assert!(
            digest.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "source digest: {path}"
        );
        assert!(usize_field(row, "bytes") > 0, "source bytes: {path}");
        assert!(seen.insert(path.clone()), "duplicate source path: {path}");
        let (_, expected_digest) = expected
            .iter()
            .find(|(expected_path, _)| *expected_path == path)
            .unwrap_or_else(|| panic!("unexpected source path: {path}"));
        assert_eq!(digest, *expected_digest, "source digest: {path}");
    }
}

fn artifact_shape(row: &JsonValue, name: &str, bytes: usize) -> Vec<usize> {
    let shape = field(row, "shape")
        .as_array()
        .unwrap_or_else(|| panic!("artifact shape missing: {name}"));
    assert!(!shape.is_empty(), "artifact shape empty: {name}");
    let mut dimensions = Vec::with_capacity(shape.len());
    let mut elements = 1usize;
    for dim in shape {
        let dim = usize_field_value(dim, "artifact shape dimension");
        assert!(dim > 0, "artifact shape has zero dimension: {name}");
        dimensions.push(dim);
        elements = elements
            .checked_mul(dim)
            .unwrap_or_else(|| panic!("artifact shape overflows: {name}"));
    }
    assert_eq!(
        elements.checked_mul(4),
        Some(bytes),
        "artifact shape/bytes: {name}"
    );
    match name {
        "pcm.f32le" => assert_eq!(shape.len(), 1, "PCM rank"),
        "tokens.u32le" => assert_eq!(shape.len(), 1, "token rank"),
        "frontend.f32le" | "encoder.f32le" | "ctc_logits.f32le" => {
            assert_eq!(shape.len(), 3, "model artifact rank: {name}")
        }
        _ => panic!("unexpected artifact: {name}"),
    }
    dimensions
}

fn usize_field_value(value: &JsonValue, label: &str) -> usize {
    value
        .as_u64()
        .unwrap_or_else(|| panic!("{label} is not a non-negative integer")) as usize
}

fn verify_packet(reference: &Path) -> JsonValue {
    let manifest_path = reference.join("manifest.json");
    let bytes = std::fs::read(&manifest_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", manifest_path.display()));
    let root = json::parse(&bytes).expect("strict reference JSON parses");
    no_duplicate_objects(&root);
    exact_object_keys(
        &root,
        &[
            "schema",
            "status",
            "model",
            "source",
            "input",
            "artifacts",
            "comparison",
        ],
        "root",
    );
    assert_eq!(string_field(&root, "schema"), "omniasr-ctc-reference-v1");
    assert_eq!(string_field(&root, "status"), "REFERENCE_COMPLETE");
    let model = field(&root, "model");
    exact_object_keys(
        model,
        &[
            "id",
            "hf_revision",
            "checkpoint_sha256",
            "prepared_sha256",
            "checkpoint_bytes",
            "tokenizer_sha256",
            "tokenizer_bytes",
            "dtype",
            "tensor_count",
        ],
        "model",
    );
    assert_eq!(string_field(model, "id"), MODEL_ID);
    assert_eq!(string_field(model, "hf_revision"), HF_REVISION);
    assert_eq!(string_field(model, "checkpoint_sha256"), CHECKPOINT_SHA256);
    assert_eq!(string_field(model, "prepared_sha256"), PREPARED_SHA256);
    assert_eq!(string_field(model, "dtype"), "float32");
    assert_eq!(usize_field(model, "tensor_count"), 807);
    assert!(usize_field(model, "checkpoint_bytes") > 0);
    assert!(usize_field(model, "tokenizer_bytes") > 0);
    for key in ["checkpoint_sha256", "tokenizer_sha256", "prepared_sha256"] {
        let digest = string_field(model, key);
        assert_eq!(digest.len(), 64, "model digest length: {key}");
        assert!(
            digest.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "model digest: {key}"
        );
    }
    let source = field(&root, "source");
    exact_object_keys(source, &["omnilingual_asr", "fairseq2"], "source");
    exact_source_files(
        field(source, "omnilingual_asr"),
        OMNI_SOURCE_FILES,
        "omnilingual_asr",
    );
    exact_source_files(field(source, "fairseq2"), FAIRSEQ2_SOURCE_FILES, "fairseq2");
    assert_eq!(
        string_field(field(source, "omnilingual_asr"), "revision"),
        OMNI_REVISION
    );
    assert_eq!(
        string_field(field(source, "fairseq2"), "revision"),
        FAIRSEQ2_REVISION
    );
    assert_eq!(
        string_field(field(source, "omnilingual_asr"), "repository"),
        OMNI_REPOSITORY
    );
    assert_eq!(
        string_field(field(source, "fairseq2"), "repository"),
        FAIRSEQ2_REPOSITORY
    );
    let input = field(&root, "input");
    exact_object_keys(
        input,
        &[
            "sample_rate",
            "channels",
            "samples",
            "pcm_sha256",
            "dtype",
            "normalization",
        ],
        "input",
    );
    assert_eq!(
        usize_field(input, "sample_rate"),
        OMNIASR_CTC_SAMPLE_RATE as usize
    );
    assert_eq!(usize_field(input, "channels"), 1);
    assert_eq!(usize_field(input, "samples"), 16_000);
    assert_eq!(string_field(input, "dtype"), "float32-le");
    let pcm_digest = string_field(input, "pcm_sha256");
    assert_eq!(pcm_digest.len(), 64);
    assert!(pcm_digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
    assert_eq!(
        string_field(input, "normalization"),
        "torch_layer_norm_waveform_eps_1e-5"
    );
    let artifacts = field(&root, "artifacts")
        .as_array()
        .expect("artifact array");
    assert_eq!(artifacts.len(), 5);
    let mut names = std::collections::BTreeSet::new();
    for row in artifacts {
        exact_object_keys(
            row,
            &["path", "sha256", "bytes", "dtype", "shape"],
            "artifact row",
        );
        let name = string_field(row, "path");
        assert!(
            names.insert(name.clone()),
            "duplicate artifact row `{name}`"
        );
        let path = reference.join(&name);
        assert!(
            !path.is_symlink() && path.is_file(),
            "artifact missing/symlink: {name}"
        );
        let file_bytes = std::fs::read(&path).expect("artifact bytes");
        assert_eq!(
            file_bytes.len(),
            usize_field(row, "bytes"),
            "artifact size: {name}"
        );
        assert_eq!(
            sha256_hex(&file_bytes),
            string_field(row, "sha256"),
            "artifact digest: {name}"
        );
        let dtype = string_field(row, "dtype");
        assert_eq!(
            dtype,
            if name == "tokens.u32le" {
                "uint32-le"
            } else {
                "float32-le"
            }
        );
        let shape = artifact_shape(row, &name, file_bytes.len());
        match name.as_str() {
            "pcm.f32le" => assert_eq!(shape.as_slice(), &[16_000]),
            "frontend.f32le" | "encoder.f32le" => {
                assert_eq!(shape[0], 1, "batch dimension: {name}");
                assert_eq!(shape[2], 1280, "model dimension: {name}");
            }
            "ctc_logits.f32le" => {
                assert_eq!(shape[0], 1, "logit batch dimension");
                assert_eq!(shape[2], 9812, "logit vocabulary dimension");
            }
            "tokens.u32le" => {}
            _ => unreachable!(),
        }
        let digest = string_field(row, "sha256");
        assert_eq!(digest.len(), 64, "artifact digest length: {name}");
        assert!(
            digest.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "artifact digest: {name}"
        );
        if name == "pcm.f32le" {
            assert_eq!(digest, pcm_digest, "PCM input/artifact digest");
        }
    }
    assert_eq!(
        names,
        [
            "ctc_logits.f32le",
            "encoder.f32le",
            "frontend.f32le",
            "pcm.f32le",
            "tokens.u32le"
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    );
    let comparison = field(&root, "comparison");
    exact_object_keys(
        comparison,
        &[
            "frontend_atol",
            "encoder_atol",
            "logits_atol",
            "tokens",
            "status",
        ],
        "comparison",
    );
    assert_eq!(f64_field(comparison, "frontend_atol"), 0.01);
    assert_eq!(f64_field(comparison, "encoder_atol"), 0.01);
    assert_eq!(f64_field(comparison, "logits_atol"), 0.01);
    assert_eq!(string_field(comparison, "tokens"), "exact");
    assert_eq!(string_field(comparison, "status"), "NOT_RUN_RUST");
    root
}

fn selected_backend() -> BackendKind {
    match std::env::var("VOKRA_OMNIASR_BACKEND").as_deref() {
        Ok("metal") => BackendKind::Metal,
        Ok("cpu") | Err(_) => BackendKind::Cpu,
        Ok(other) => panic!("VOKRA_OMNIASR_BACKEND must be cpu or metal, got {other:?}"),
    }
}

fn require_forward<T>(result: vokra_core::Result<T>, label: &str) -> T {
    match result {
        Ok(value) => value,
        Err(VokraError::UnsupportedOp(message)) => {
            // This marker is the only condition an Apple worker may classify
            // as a backend capability block.  Numeric, schema, and compile
            // failures never emit it.
            println!("OMNIASR_UNSUPPORTED_OP");
            panic!("{label}: explicit UnsupportedOp: {message}");
        }
        Err(error) => panic!("{label}: {error}"),
    }
}

#[test]
#[ignore = "real 1B GGUF/reference packet runs on VAST or an authenticated Apple worker only"]
fn real_omniasr_ctc_encoder_logits_and_tokens_match_official() {
    let (gguf_path, reference) = paths();
    let root = verify_packet(&reference);
    let pcm = read_f32(&reference.join("pcm.f32le"));
    let expected_frontend = read_f32(&reference.join("frontend.f32le"));
    let expected_encoder = read_f32(&reference.join("encoder.f32le"));
    let expected_logits = read_f32(&reference.join("ctc_logits.f32le"));
    let expected_tokens = read_u32(&reference.join("tokens.u32le"));
    assert_eq!(pcm.len(), 16_000);
    assert!(pcm.iter().any(|v| *v != 0.0), "PCM must be non-zero");
    assert!(string_field(field(&root, "comparison"), "status") == "NOT_RUN_RUST");

    let file = GgufFile::open(&gguf_path).expect("open authenticated OmniASR GGUF");
    let model = OmniasrCtcAsr::from_gguf(&file)
        .expect("strict OmniASR manifest/provenance bind")
        .with_backend(selected_backend());
    let (frontend, encoder, frames) = require_forward(
        model.diagnostic_trace(&pcm),
        "native frontend/encoder forward",
    );
    assert_eq!(
        frontend.len(),
        expected_frontend.len(),
        "frontend extent differs"
    );
    assert_eq!(
        encoder.len(),
        expected_encoder.len(),
        "encoder extent differs"
    );
    let (logits, logit_frames) =
        require_forward(model.ctc_logits(&pcm), "native CTC logits forward");
    assert_eq!(logit_frames, frames);
    assert_eq!(logits.len(), expected_logits.len(), "logit extent differs");
    let tokens = require_forward(model.transcribe_tokens(&pcm), "native greedy CTC decode");
    assert_eq!(
        tokens, expected_tokens,
        "greedy token ids differ from official reference"
    );
    let encoder_max = encoder
        .iter()
        .zip(&expected_encoder)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let frontend_max = frontend
        .iter()
        .zip(&expected_frontend)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let logits_max = logits
        .iter()
        .zip(&expected_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        encoder_max <= ATOL,
        "encoder max_abs {encoder_max} exceeds {ATOL}"
    );
    assert!(
        frontend_max <= ATOL,
        "frontend max_abs {frontend_max} exceeds {ATOL}"
    );
    assert!(
        logits_max <= ATOL,
        "logits max_abs {logits_max} exceeds {ATOL}"
    );
    eprintln!(
        "OmniASR {:?}: frames={frames}, frontend_max_abs={frontend_max:.9e}, encoder_max_abs={encoder_max:.9e}, logits_max_abs={logits_max:.9e}, tokens=exact",
        selected_backend()
    );
    println!("OMNIASR_REAL_PARITY_PASS");
}
