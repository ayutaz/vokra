//! Independent real-weight DNSMOS P.808 + P.835 numerical parity.
//!
//! Generate the reference JSONL with the exact official Microsoft
//! `dnsmos_local.py` through `tools/parity/dnsmos_score_reference.py`, then
//! convert the audited 38 tensors through `dnsmos_prepare_checkpoint.py` and
//! the strict Vokra converter. Model execution and this `vokra-models` test
//! belong on VAST; the real-weight legs skip unless explicitly configured.
//!
//! ```text
//! uv run --project tools/parity python tools/parity/dnsmos_score_reference.py \
//!   --source-dir ~/DNS-Challenge \
//!   --p808 ~/DNS-Challenge/DNSMOS/model_v8.onnx \
//!   --p835 ~/DNS-Challenge/DNSMOS/sig_bak_ovr.onnx \
//!   --input-wav ~/clean.wav --input-wav ~/noisy.wav \
//!   --output-jsonl ~/dnsmos-reference.jsonl
//!
//! export VOKRA_DNSMOS_REAL_GGUF=~/dnsmos.gguf
//! export VOKRA_DNSMOS_REAL_WAVS=~/clean.wav:~/noisy.wav
//! export VOKRA_DNSMOS_REFERENCE_JSONL=~/dnsmos-reference.jsonl
//! export VOKRA_DNSMOS_MOS_ATOL=<recorded-independent-VAST-bound>
//! cargo test -p vokra-models --test parity_dnsmos -- --nocapture
//! ```
//!
//! The tolerance is deliberately required from the environment. It is not
//! invented before the first recorded independent VAST comparison.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[cfg(all(feature = "metal", target_os = "macos"))]
use vokra_core::BackendKind;
use vokra_core::engines::MosScore;
use vokra_core::json::JsonValue;
use vokra_models::dnsmos_p808_p835::{
    Dnsmos, DnsmosSubmodel, EXPECTED_SAMPLE_RATE, INPUT_LENGTH_SAMPLES,
};
use vokra_models::silero_vad::wav::read_wav_f32;

const GGUF_ENV: &str = "VOKRA_DNSMOS_REAL_GGUF";
const WAVS_ENV: &str = "VOKRA_DNSMOS_REAL_WAVS";
const REFERENCE_JSONL_ENV: &str = "VOKRA_DNSMOS_REFERENCE_JSONL";
const MOS_ATOL_ENV: &str = "VOKRA_DNSMOS_MOS_ATOL";
#[cfg(all(feature = "metal", target_os = "macos"))]
const METAL_ATOL_ENV: &str = "VOKRA_DNSMOS_METAL_ATOL";

const SOURCE_REVISION: &str = "591184a9fcb2cbdec02520fed81a32bbbf9d73ff";
const SOURCE_SHA256: &str = "1ab566afe006daab32ac7073296a5d0ef99f8b82f91c7266f3ccf26113d7a28b";
const P808_ONNX_SHA256: &str = "9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91";
const P835_ONNX_SHA256: &str = "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd";
const ONNXRUNTIME_VERSION: &str = "1.29.0";
const NUMPY_VERSION: &str = "2.3.5";
const LIBROSA_VERSION: &str = "0.11.0";
const SOUNDFILE_VERSION: &str = "0.14.0";

#[derive(Debug, Clone, Copy)]
struct ReferenceScore {
    p808: f32,
    sig: f32,
    bak: f32,
    ovrl: f32,
}

fn required_bound(name: &str) -> f32 {
    let raw = std::env::var(name).unwrap_or_else(|_| {
        panic!(
            "{name} is required when {GGUF_ENV} is set; calibrate it from the recorded independent VAST comparison"
        )
    });
    let value = raw
        .parse::<f32>()
        .unwrap_or_else(|error| panic!("{name}={raw:?}: {error}"));
    assert!(
        value.is_finite() && value >= 0.0,
        "{name} must be finite and non-negative"
    );
    value
}

fn number(value: &JsonValue, context: &str) -> f32 {
    let value = match value {
        JsonValue::Int(value) => *value as f64,
        JsonValue::Float(value) => *value,
        other => panic!("{context}: expected a JSON number, got {other:?}"),
    };
    assert!(value.is_finite(), "{context}: value must be finite");
    value as f32
}

fn required_string<'a>(root: &'a JsonValue, key: &str, context: &str) -> &'a str {
    root.get(key)
        .and_then(JsonValue::as_str)
        .unwrap_or_else(|| panic!("{context}: missing/non-string `{key}`"))
}

fn parse_references(path: &Path) -> BTreeMap<String, ReferenceScore> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read reference JSONL {}: {error}", path.display()));
    let mut references = BTreeMap::new();
    for (line_index, line) in text.lines().enumerate() {
        assert!(
            !line.trim().is_empty(),
            "{}:{}: blank JSONL record",
            path.display(),
            line_index + 1
        );
        let context = format!("{}:{}", path.display(), line_index + 1);
        let root = vokra_core::json::parse(line.as_bytes())
            .unwrap_or_else(|error| panic!("{context}: {error}"));
        assert_eq!(
            required_string(&root, "source_revision", &context),
            SOURCE_REVISION,
            "{context}: source revision"
        );
        assert_eq!(
            required_string(&root, "source_sha256", &context),
            SOURCE_SHA256,
            "{context}: official source hash"
        );
        assert_eq!(
            required_string(&root, "p808_onnx_sha256", &context),
            P808_ONNX_SHA256,
            "{context}: P.808 ONNX hash"
        );
        assert_eq!(
            required_string(&root, "p835_onnx_sha256", &context),
            P835_ONNX_SHA256,
            "{context}: P.835 ONNX hash"
        );
        for (key, expected) in [
            ("onnxruntime_version", ONNXRUNTIME_VERSION),
            ("numpy_version", NUMPY_VERSION),
            ("librosa_version", LIBROSA_VERSION),
            ("soundfile_version", SOUNDFILE_VERSION),
        ] {
            assert_eq!(
                required_string(&root, key, &context),
                expected,
                "{context}: reference dependency `{key}`"
            );
        }
        let wav = required_string(&root, "wav", &context).to_owned();
        let score = ReferenceScore {
            p808: number(root.get("p808").expect("reference p808"), "p808"),
            sig: number(root.get("sig").expect("reference sig"), "sig"),
            bak: number(root.get("bak").expect("reference bak"), "bak"),
            ovrl: number(root.get("ovrl").expect("reference ovrl"), "ovrl"),
        };
        assert!(
            references.insert(wav.clone(), score).is_none(),
            "{context}: duplicate WAV basename `{wav}`"
        );
    }
    assert!(!references.is_empty(), "reference JSONL is empty");
    references
}

fn real_fixture() -> Option<(PathBuf, Vec<PathBuf>, BTreeMap<String, ReferenceScore>)> {
    let Some(gguf) = std::env::var_os(GGUF_ENV).map(PathBuf::from) else {
        eprintln!(
            "[parity_dnsmos] SKIP: set {GGUF_ENV}, {WAVS_ENV}, {REFERENCE_JSONL_ENV}, and {MOS_ATOL_ENV} after generating the official VAST reference"
        );
        return None;
    };
    let wavs = std::env::var_os(WAVS_ENV)
        .map(|value| std::env::split_paths(&value).collect::<Vec<_>>())
        .filter(|paths| !paths.is_empty())
        .unwrap_or_else(|| panic!("{WAVS_ENV} is required when {GGUF_ENV} is set"));
    let reference_path = std::env::var_os(REFERENCE_JSONL_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{REFERENCE_JSONL_ENV} is required when {GGUF_ENV} is set"));
    Some((gguf, wavs, parse_references(&reference_path)))
}

fn score_fields(score: &MosScore) -> ReferenceScore {
    ReferenceScore {
        p808: score.p808.expect("strict DNSMOS emits P.808"),
        sig: score.sig.expect("strict DNSMOS emits SIG"),
        bak: score.bak.expect("strict DNSMOS emits BAK"),
        ovrl: score.ovrl.expect("strict DNSMOS emits OVRL"),
    }
}

fn compare(label: &str, actual: ReferenceScore, expected: ReferenceScore, bound: f32) {
    for (field, actual, expected) in [
        ("p808", actual.p808, expected.p808),
        ("sig", actual.sig, expected.sig),
        ("bak", actual.bak, expected.bak),
        ("ovrl", actual.ovrl, expected.ovrl),
    ] {
        let delta = (actual - expected).abs();
        eprintln!(
            "DNSMOS {label} {field}: actual={actual:.9e}, reference={expected:.9e}, abs={delta:.9e}"
        );
        assert!(
            delta <= bound,
            "DNSMOS {label} {field}: abs={delta:.9e}, bound={bound:.9e}"
        );
    }
}

fn load_pcm(path: &Path) -> Vec<f32> {
    let wav = read_wav_f32(path)
        .unwrap_or_else(|error| panic!("read DNSMOS WAV {}: {error}", path.display()));
    assert_eq!(
        wav.sample_rate,
        EXPECTED_SAMPLE_RATE,
        "{}: DNSMOS parity WAV must be 16 kHz",
        path.display()
    );
    assert!(!wav.samples.is_empty(), "{}: empty WAV", path.display());
    wav.samples
}

#[test]
fn dnsmos_primary_source_constants_pin() {
    assert_eq!(EXPECTED_SAMPLE_RATE, 16_000);
    assert_eq!(INPUT_LENGTH_SAMPLES, 144_160);
    assert_eq!(DnsmosSubmodel::P808.short(), "p808");
    assert_eq!(DnsmosSubmodel::P835.short(), "p835");
    assert_eq!(DnsmosSubmodel::P808.tensor_prefix(), "p808.");
    assert_eq!(DnsmosSubmodel::P835.tensor_prefix(), "p835.");
}

#[test]
fn parity_dnsmos_gguf_smoke() {
    let Some((gguf, _, _)) = real_fixture() else {
        return;
    };
    let model = Dnsmos::from_path(&gguf)
        .unwrap_or_else(|error| panic!("strict DNSMOS bind {}: {error}", gguf.display()));
    assert_eq!(model.sample_rate(), EXPECTED_SAMPLE_RATE);
    assert_eq!(model.tensor_count(), 38);
    assert_eq!(model.config().bundle, ["p808", "p835"]);
}

#[test]
fn cpu_matches_official_dnsmos() {
    let Some((gguf, wavs, references)) = real_fixture() else {
        return;
    };
    let bound = required_bound(MOS_ATOL_ENV);
    let model = Dnsmos::from_path(&gguf)
        .unwrap_or_else(|error| panic!("strict DNSMOS bind {}: {error}", gguf.display()));
    assert_eq!(references.len(), wavs.len(), "reference/WAV count");
    for wav in wavs {
        let basename = wav
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_else(|| panic!("non-UTF-8 WAV basename: {}", wav.display()));
        let reference = *references
            .get(basename)
            .unwrap_or_else(|| panic!("reference JSONL has no `{basename}`"));
        let actual = model
            .score_all(&load_pcm(&wav))
            .unwrap_or_else(|error| panic!("CPU DNSMOS {}: {error}", wav.display()));
        compare(
            &format!("CPU vs official ({basename})"),
            score_fields(&actual),
            reference,
            bound,
        );
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_matches_cpu_dnsmos() {
    let Some((gguf, wavs, _)) = real_fixture() else {
        return;
    };
    let bound = required_bound(METAL_ATOL_ENV);
    let file = vokra_core::gguf::GgufFile::open(&gguf)
        .unwrap_or_else(|error| panic!("open DNSMOS GGUF {}: {error}", gguf.display()));
    let cpu = Dnsmos::from_gguf(&file).expect("strict CPU DNSMOS bind");
    let metal = Dnsmos::from_gguf_with_backend(&file, BackendKind::Metal)
        .expect("strict Metal DNSMOS bind");
    for wav in wavs {
        let pcm = load_pcm(&wav);
        let cpu_score = cpu
            .score_all(&pcm)
            .unwrap_or_else(|error| panic!("CPU DNSMOS {}: {error}", wav.display()));
        let metal_score = metal
            .score_all(&pcm)
            .unwrap_or_else(|error| panic!("Metal DNSMOS {}: {error}", wav.display()));
        compare(
            &format!("Metal vs CPU ({})", wav.display()),
            score_fields(&metal_score),
            score_fields(&cpu_score),
            bound,
        );
    }
}
