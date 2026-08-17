//! openWakeWord op-wiring converter integration test (2026-08-15
//! converter/binder handshake repair).
//!
//! # The drift this test exists to prevent
//!
//! `vokra-convert`'s `openwakeword_op` converter and the runtime binder
//! at `crates/vokra-models/src/kws/openwakeword/mod.rs` were written for
//! each other, and until 2026-08-15 they could not handshake:
//! `OpenwakewordConfig::from_gguf` reads seven `vokra.openwakeword.*`
//! metadata keys and treats every one as required, and the converter
//! stamped none of them. Every GGUF the converter produced therefore
//! failed at the binder's first load, and the owner recipe documented in
//! `crates/vokra-models/tests/parity_openwakeword.rs` dead-ended there.
//!
//! Nothing in the suite could see it, and the reason is worth recording
//! because it generalises: the binder's own unit tests hand-build their
//! GGUF with `GgufBuilder` instead of running the converter, and the
//! parity harness that *would* have run the real pipeline is env-gated
//! and skips when the owner fixture is absent. Two halves, each tested
//! against a mock of the other. Tensor names matched all along — only
//! the metadata group was missing, which is exactly the kind of gap a
//! name-shaped test cannot see.
//!
//! So this test asserts the **binder's requirement list**, not the
//! converter's behaviour: every key `OpenwakewordConfig::from_gguf`
//! reads must be present, with the type it reads it as. Its sibling
//! `crates/vokra-models/tests/openwakeword_convert_bind.rs` closes the
//! loop from the other side by feeding this converter's output straight
//! into `OpenwakewordSession::from_gguf`.
//!
//! Mirror of the `higgs_audio_v3_tts_4b_roundtrip` / `magpietts_v2602` /
//! `frcrn` roundtrip pattern.

use std::path::PathBuf;

use vokra_convert::{
    ConvertError, ModelKind, convert_file, convert_openwakeword_op_file,
    convert_openwakeword_op_file_with_config,
};
use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufFile, GgufValueType, chunks};

/// A unique temp path for this test process (mirror of
/// `roundtrip.rs::tmp_path` — no external `tempfile` dep, preserving
/// zero-dep NFR-DS-02).
fn tmp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "vokra-convert-openwakeword-op-it-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default(),
    ));
    p
}

/// EVERY `vokra.openwakeword.*` key `OpenwakewordConfig::from_gguf`
/// reads, transcribed from that function body. Six are read through
/// `get_u32`; the seventh is read through `read_string_array`.
///
/// Keeping the list here rather than importing it is deliberate: this
/// test is the fence between two crates that must not depend on each
/// other, so it restates the contract in the words of the reader. If the
/// binder adds an eighth key, this list is where the reminder lands.
const REQUIRED_U32_KEYS: [&str; 6] = [
    "vokra.openwakeword.n_wakewords",
    "vokra.openwakeword.embedding_dim",
    "vokra.openwakeword.window_frames",
    "vokra.openwakeword.mel_bins",
    "vokra.openwakeword.sample_rate",
    "vokra.openwakeword.hop_samples",
];

/// The seventh required key, read as `Array<String>`.
const REQUIRED_STRING_ARRAY_KEY: &str = "vokra.openwakeword.wakeword_names";

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// Two wake-words sharing a 3-wide embedding, plus a shared embedding
/// tensor that rides along under the `openwakeword.embedding.*` prefix.
///
/// - wake-word 0: `hidden_dim = 2`, all four tensors F32.
/// - wake-word 1: `hidden_dim = 1`, `linear1.weight` BF16 and
///   `linear2.weight` F16 so the dtype pass-through matrix is covered on
///   real classifier tensor names (not on placeholder names the binder
///   would never look for).
///
/// Layout matches what `tools/parity/openwakeword_prepare_checkpoint.py`
/// emits and what the runtime binder reads back.
fn synthetic_openwakeword_safetensors() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    // --- wake-word 0 (F32 throughout) ---
    let w0_l1w = f32_bytes(&[0.1, 0.2, -0.1, 0.05, -0.05, 0.1]); // [2, 3]
    let w0_l1b = f32_bytes(&[0.01, -0.02]); // [2]
    let w0_l2w = f32_bytes(&[0.5, -0.3]); // [1, 2]
    let w0_l2b = f32_bytes(&[0.02]); // [1]
    // --- wake-word 1 (BF16 + F16 mixed) ---
    let w1_l1w_vals: [f32; 3] = [1.0, -2.5, 0.15625];
    let w1_l1w = bf16_bytes(&w1_l1w_vals); // [1, 3] BF16 = 6 bytes
    let w1_l1b = f32_bytes(&[0.03]); // [1]
    let w1_l2w_patterns: [u16; 1] = [0x3C00]; // 1.0 in f16
    let w1_l2w: Vec<u8> = w1_l2w_patterns
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect(); // [1, 1]
    let w1_l2b = f32_bytes(&[-0.04]); // [1]
    // --- shared embedding extractor weight (pass-through only) ---
    let embed = f32_bytes(&[0.7, 0.8, 0.9]); // [3]

    let mut off = 0usize;
    let mut bump = |n: usize| {
        let start = off;
        off += n;
        (start, off)
    };
    let (a0, a1) = bump(w0_l1w.len());
    let (b0, b1) = bump(w0_l1b.len());
    let (c0, c1) = bump(w0_l2w.len());
    let (d0, d1) = bump(w0_l2b.len());
    let (e0, e1) = bump(w1_l1w.len());
    let (f0, f1) = bump(w1_l1b.len());
    let (g0, g1) = bump(w1_l2w.len());
    let (h0, h1) = bump(w1_l2b.len());
    let (i0, i1) = bump(embed.len());

    let header = format!(
        r#"{{"openwakeword.classifier.0.linear1.weight":{{"dtype":"F32","shape":[2,3],"data_offsets":[{a0},{a1}]}},"openwakeword.classifier.0.linear1.bias":{{"dtype":"F32","shape":[2],"data_offsets":[{b0},{b1}]}},"openwakeword.classifier.0.linear2.weight":{{"dtype":"F32","shape":[1,2],"data_offsets":[{c0},{c1}]}},"openwakeword.classifier.0.linear2.bias":{{"dtype":"F32","shape":[1],"data_offsets":[{d0},{d1}]}},"openwakeword.classifier.1.linear1.weight":{{"dtype":"BF16","shape":[1,3],"data_offsets":[{e0},{e1}]}},"openwakeword.classifier.1.linear1.bias":{{"dtype":"F32","shape":[1],"data_offsets":[{f0},{f1}]}},"openwakeword.classifier.1.linear2.weight":{{"dtype":"F16","shape":[1,1],"data_offsets":[{g0},{g1}]}},"openwakeword.classifier.1.linear2.bias":{{"dtype":"F32","shape":[1],"data_offsets":[{h0},{h1}]}},"openwakeword.embedding.dense.weight":{{"dtype":"F32","shape":[3],"data_offsets":[{i0},{i1}]}}}}"#
    );

    let mut buf = Vec::new();
    buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    for chunk in [
        &w0_l1w, &w0_l1b, &w0_l2w, &w0_l2b, &w1_l1w, &w1_l1b, &w1_l2w, &w1_l2b, &embed,
    ] {
        buf.extend_from_slice(chunk);
    }
    (buf, w1_l1w, embed)
}

/// The minimal side-car: only `wakeword_names` is required.
const MINIMAL_CONFIG: &str = r#"{"wakeword_names":["alexa","hey_jarvis"]}"#;

/// THE regression fence. Converts a synthetic checkpoint through the
/// public entry point and asserts that every key the runtime binder
/// requires is present, with the type that binder reads it as.
///
/// Before 2026-08-15 this test would have failed on the very first key.
#[test]
fn every_metadata_key_the_binder_requires_is_stamped() {
    let (input_bytes, _, _) = synthetic_openwakeword_safetensors();
    let input = tmp_path("keys-in");
    let config = tmp_path("keys-cfg");
    let output = tmp_path("keys-out");
    std::fs::write(&input, &input_bytes).expect("write input");
    std::fs::write(&config, MINIMAL_CONFIG).expect("write config");

    let summary = convert_openwakeword_op_file_with_config(&input, &config, &output, None)
        .expect("convert with side-car");
    assert_eq!(summary.model, ModelKind::OpenwakewordOp);
    assert_eq!(summary.tensor_count, 9, "8 classifier + 1 embedding tensor");
    assert!(summary.output_bytes > 0);

    let file = GgufFile::open(&output).expect("load output gguf");

    // The six `get_u32` keys. `OpenwakewordConfig::from_gguf` reads each
    // through `gguf.get(k).and_then(|v| v.as_u64())`, so a key stamped
    // as a string or float would parse as absent — assert the read the
    // binder actually performs, not merely `.is_some()`.
    for key in REQUIRED_U32_KEYS {
        let v = file
            .get(key)
            .unwrap_or_else(|| panic!("required metadata key `{key}` is missing"));
        assert!(
            v.as_u64().is_some(),
            "`{key}` must read back through as_u64() the way the binder reads it"
        );
    }

    // The seventh key, read through `read_string_array`, which enforces
    // `element_type == String` and refuses any non-String element.
    let names = file
        .get(REQUIRED_STRING_ARRAY_KEY)
        .unwrap_or_else(|| panic!("required key `{REQUIRED_STRING_ARRAY_KEY}` is missing"))
        .as_array()
        .expect("wakeword_names must be an array");
    assert_eq!(
        names.element_type,
        GgufValueType::String,
        "read_string_array refuses any element_type but String"
    );
    let decoded: Vec<&str> = names.values.iter().filter_map(|v| v.as_str()).collect();
    assert_eq!(decoded, vec!["alexa", "hey_jarvis"]);

    // The binder additionally cross-checks `wakeword_names.len() ==
    // n_wakewords` in `OpenwakewordConfig::validate`, and refuses a
    // 0-sentinel on every hparam. Pin both here.
    let n_wakewords = file
        .get("vokra.openwakeword.n_wakewords")
        .and_then(|v| v.as_u64())
        .expect("n_wakewords");
    assert_eq!(n_wakewords, 2, "two classifier groups in the fixture");
    assert_eq!(
        names.values.len() as u64,
        n_wakewords,
        "validate() refuses a wakeword_names/n_wakewords mismatch"
    );
    for key in REQUIRED_U32_KEYS {
        let v = file.get(key).and_then(|v| v.as_u64()).expect(key);
        assert!(
            v > 0,
            "`{key}` must be > 0 — validate() refuses 0-sentinels"
        );
    }

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&output);
}

/// `embedding_dim` is derived from the classifier weights rather than
/// mirrored, so it tracks the checkpoint. The fixture's 3-wide embedding
/// is deliberately unlike the reference release's 96 — a mirrored
/// constant would be caught here.
#[test]
fn derived_axes_track_the_checkpoint_not_a_constant() {
    let (input_bytes, _, _) = synthetic_openwakeword_safetensors();
    let input = tmp_path("derive-in");
    let config = tmp_path("derive-cfg");
    let output = tmp_path("derive-out");
    std::fs::write(&input, &input_bytes).expect("write input");
    std::fs::write(&config, MINIMAL_CONFIG).expect("write config");

    convert_openwakeword_op_file_with_config(&input, &config, &output, None).expect("convert");
    let file = GgufFile::open(&output).expect("load");

    assert_eq!(
        file.get("vokra.openwakeword.embedding_dim")
            .and_then(|v| v.as_u64()),
        Some(3),
        "embedding_dim comes from dim 1 of classifier.0.linear1.weight"
    );
    assert_eq!(
        file.get("vokra.openwakeword.n_wakewords")
            .and_then(|v| v.as_u64()),
        Some(2),
        "n_wakewords is the length of the contiguous classifier run"
    );

    // The four mirrored front-end axes land at their documented values
    // when the side-car omits them.
    assert_eq!(
        file.get("vokra.openwakeword.window_frames")
            .and_then(|v| v.as_u64()),
        Some(76)
    );
    assert_eq!(
        file.get("vokra.openwakeword.mel_bins")
            .and_then(|v| v.as_u64()),
        Some(32)
    );
    assert_eq!(
        file.get("vokra.openwakeword.sample_rate")
            .and_then(|v| v.as_u64()),
        Some(16_000)
    );
    assert_eq!(
        file.get("vokra.openwakeword.hop_samples")
            .and_then(|v| v.as_u64()),
        Some(160)
    );

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&output);
}

/// The side-car overrides every mirrored front-end axis, so a
/// self-trained checkpoint with a different front-end is expressible
/// without editing the converter.
#[test]
fn side_car_overrides_every_mirrored_front_end_axis() {
    let (input_bytes, _, _) = synthetic_openwakeword_safetensors();
    let input = tmp_path("override-in");
    let config = tmp_path("override-cfg");
    let output = tmp_path("override-out");
    std::fs::write(&input, &input_bytes).expect("write input");
    std::fs::write(
        &config,
        r#"{"wakeword_names":["alexa","hey_jarvis"],"window_frames":40,"mel_bins":16,"sample_rate":8000,"hop_samples":80}"#,
    )
    .expect("write config");

    convert_openwakeword_op_file_with_config(&input, &config, &output, None).expect("convert");
    let file = GgufFile::open(&output).expect("load");
    for (key, expected) in [
        ("vokra.openwakeword.window_frames", 40u64),
        ("vokra.openwakeword.mel_bins", 16),
        ("vokra.openwakeword.sample_rate", 8_000),
        ("vokra.openwakeword.hop_samples", 80),
    ] {
        assert_eq!(
            file.get(key).and_then(|v| v.as_u64()),
            Some(expected),
            "`{key}` must honour the side-car override"
        );
    }

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&output);
}

/// The dtype pass-through matrix survives on real classifier tensor
/// names, and the model / provenance stamps still land alongside the new
/// chunk group.
///
/// This is also the "wrong-arm dispatch" fence: if a future refactor
/// wired `ModelKind::OpenwakewordOp` to a sibling `convert_*_file`, the
/// arch string would change and this assertion would fire.
#[test]
fn dtypes_pass_through_verbatim_and_stamps_land() {
    let (input_bytes, w1_l1w_payload, embed_payload) = synthetic_openwakeword_safetensors();
    let input = tmp_path("dtype-in");
    let config = tmp_path("dtype-cfg");
    let output = tmp_path("dtype-out");
    std::fs::write(&input, &input_bytes).expect("write input");
    std::fs::write(&config, MINIMAL_CONFIG).expect("write config");

    convert_openwakeword_op_file_with_config(&input, &config, &output, None).expect("convert");
    let file = GgufFile::open(&output).expect("load");
    assert_eq!(file.tensors().len(), 9);

    // BF16 stays BF16 — a silent widen to F32 would change the on-disk
    // dtype tag AND double the payload, so this double-locks it.
    let bf16_info = file
        .tensor_info("openwakeword.classifier.1.linear1.weight")
        .expect("BF16 classifier weight present");
    assert_eq!(bf16_info.dtype, GgmlType::BF16);
    assert_eq!(bf16_info.dimensions, vec![1, 3]);
    assert_eq!(file.tensor_bytes(bf16_info), w1_l1w_payload.as_slice());

    // F16 stays F16.
    let f16_info = file
        .tensor_info("openwakeword.classifier.1.linear2.weight")
        .expect("F16 classifier weight present");
    assert_eq!(f16_info.dtype, GgmlType::F16);
    assert_eq!(f16_info.dimensions, vec![1, 1]);

    // The shared embedding extractor weights ride along untouched for
    // the follow-up wave that lights the extractor up.
    let embed_info = file
        .tensor_info("openwakeword.embedding.dense.weight")
        .expect("embedding tensor rides along");
    assert_eq!(embed_info.dtype, GgmlType::F32);
    assert_eq!(file.tensor_bytes(embed_info), embed_payload.as_slice());

    // Dim order is the binder's contract: `linear1.weight` must read
    // back as [hidden_dim, embedding_dim], `linear2.weight` as
    // [1, hidden_dim]. A transposed emit would pass an element-count
    // check but silently misclassify, which is why the binder asserts
    // dims and why this test does too.
    let l1 = file
        .tensor_info("openwakeword.classifier.0.linear1.weight")
        .expect("wake-word 0 linear1");
    assert_eq!(l1.dimensions, vec![2, 3], "[hidden_dim, embedding_dim]");
    let l2 = file
        .tensor_info("openwakeword.classifier.0.linear2.weight")
        .expect("wake-word 0 linear2");
    assert_eq!(l2.dimensions, vec![1, 2], "[1, hidden_dim]");

    assert_eq!(
        file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
        Some("openwakeword_op"),
        "arch stamp distinct from the sibling base `openwakeword` converter"
    );
    assert_eq!(
        file.get("vokra.model.category").and_then(|v| v.as_str()),
        Some("vad-kws")
    );
    assert_eq!(
        file.get("vokra.provenance.upstream_hf")
            .and_then(|v| v.as_str()),
        Some("dscripka/openWakeWord")
    );
    assert_eq!(
        file.get(chunks::KEY_PROVENANCE_LICENSE)
            .and_then(|v| v.as_str()),
        Some("apache-2.0")
    );
    assert_eq!(
        file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str()),
        Some(LicenseClass::Permissive.as_str())
    );
    assert!(file.get(chunks::KEY_SCHEMA_VERSION).is_some());
    assert!(file.get(chunks::KEY_SCHEMA_PRODUCER).is_some());

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&output);
}

/// The `--license cc-by-nc-sa-4.0` override a caller redistributing the
/// upstream official weights must pass flips the fail-closed publish
/// gate to NonCommercialShareAlike.
#[test]
fn license_override_flips_the_publish_disposition() {
    let (input_bytes, _, _) = synthetic_openwakeword_safetensors();
    let input = tmp_path("lic-in");
    let config = tmp_path("lic-cfg");
    let output = tmp_path("lic-out");
    std::fs::write(&input, &input_bytes).expect("write input");
    std::fs::write(&config, MINIMAL_CONFIG).expect("write config");

    convert_openwakeword_op_file_with_config(&input, &config, &output, Some("cc-by-nc-sa-4.0"))
        .expect("convert with override");

    let file = GgufFile::open(&output).expect("load");
    assert_eq!(
        file.get(chunks::KEY_PROVENANCE_LICENSE)
            .and_then(|v| v.as_str()),
        Some("cc-by-nc-sa-4.0")
    );
    assert_eq!(
        file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str()),
        Some(LicenseClass::NonCommercialShareAlike.as_str())
    );

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&output);
}

/// Both config-less surfaces refuse, and both route the caller to the
/// side-car path instead of emitting a GGUF that cannot load.
///
/// This is the pair that used to "succeed": `convert_file` produced an
/// artifact and reported a tensor count, and the failure only surfaced
/// later in someone else's crate.
#[test]
fn config_less_paths_refuse_rather_than_emit_an_unloadable_gguf() {
    let (input_bytes, _, _) = synthetic_openwakeword_safetensors();
    let input = tmp_path("refuse-in");
    let output = tmp_path("refuse-out");
    std::fs::write(&input, &input_bytes).expect("write input");

    // 1. The `--model openwakeword-op` dispatch arm.
    let Err(e) = convert_file(ModelKind::OpenwakewordOp, &input, &output) else {
        panic!("convert_file must refuse openwakeword-op without a --config side-car");
    };
    assert!(
        matches!(e, ConvertError::Usage(_)),
        "the refusal is a usage error, not a parse failure: {e:?}"
    );
    let msg = e.to_string();
    assert!(msg.contains("--config"), "must name the flag: {msg}");
    assert!(
        msg.contains("wakeword_names"),
        "must name the missing axis: {msg}"
    );

    // 2. The direct entry point.
    let Err(e) = convert_openwakeword_op_file(&input, &output, None) else {
        panic!("the plain entry point must refuse");
    };
    let msg = e.to_string();
    assert!(
        msg.contains("convert_openwakeword_op_file_with_config"),
        "must route the caller to the working entry point: {msg}"
    );

    // Neither refusal may leave an artifact behind that a later step
    // could mistake for a successful conversion.
    assert!(
        !output.exists(),
        "a refused conversion must not write an output GGUF"
    );

    let _ = std::fs::remove_file(&input);
}

/// A side-car whose name count disagrees with the checkpoint's group
/// count is refused at convert time. The binder would refuse it too, but
/// naming the mismatch here means the operator learns it while holding
/// the inputs rather than at first load.
#[test]
fn name_count_mismatch_is_refused_at_convert_time() {
    let (input_bytes, _, _) = synthetic_openwakeword_safetensors();
    let input = tmp_path("count-in");
    let config = tmp_path("count-cfg");
    let output = tmp_path("count-out");
    std::fs::write(&input, &input_bytes).expect("write input");
    // One name, two classifier groups.
    std::fs::write(&config, r#"{"wakeword_names":["alexa"]}"#).expect("write config");

    let Err(e) = convert_openwakeword_op_file_with_config(&input, &config, &output, None) else {
        panic!("a name/group count mismatch must be refused");
    };
    let msg = e.to_string();
    assert!(msg.contains("1 wake-word name"), "{msg}");
    assert!(msg.contains("2 classifier group"), "{msg}");

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&config);
    let _ = std::fs::remove_file(&output);
}
