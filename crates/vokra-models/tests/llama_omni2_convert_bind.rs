//! LLaMA-Omni2 converter → binder handshake (2026-08-15 repair).
//!
//! # Why this test exists
//!
//! `vokra-convert`'s `llama_omni2` converter and this crate's
//! `llama_omni2` binder were written for each other and could not
//! handshake. The binder declares eleven `vokra.llama_omni2.*` keys; the
//! converter stamped exactly one of them, the variant tag. The other ten
//! read back through `read_u32_or_zero` / `read_f32_or`, so every one
//! decayed to its `0` placeholder and `validate_for_forward` refused the
//! load with `InvalidArgument("backbone ill-formed (n_layer=0,
//! d_model=0, n_head=0)")`.
//!
//! So **every GGUF `vokra-cli convert --model llama-omni2` produced
//! failed to load in the binder written for it.** The sibling
//! `kyutai_stt`, which the binder's own docs name as its precedent, does
//! stamp its full group — the precedent was real and simply was not
//! carried over.
//!
//! The gap survived because both halves were tested against a mock of
//! the other: this crate's unit tests hand-build their GGUF with
//! `GgufBuilder` rather than running the converter, and the converter's
//! tests asserted only the five strings it did stamp. Nothing anywhere
//! ran the real converter into the real binder.
//!
//! This test is that missing half. It is fixture-free — the synthetic
//! checkpoint is built inline — so it runs in CI with no
//! owner-provisioned weights, and neither crate can drift without
//! something going red. Mirror of
//! `crates/vokra-models/tests/openwakeword_convert_bind.rs`, which
//! repaired the identical defect one round earlier.

use std::path::PathBuf;

use vokra_convert::{LlamaOmni2Variant as ConvertVariant, convert_llama_omni2_file_with_config};
use vokra_core::VokraError;
use vokra_models::llama_omni2::{LlamaOmni2, LlamaOmni2Variant};

/// A unique temp path for this test process (no external `tempfile`
/// dep — zero-dep NFR-DS-02).
fn tmp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "vokra-llama-omni2-convert-bind-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default(),
    ));
    p
}

/// Assembles a safetensors buffer from `(name, shape)` entries, all F32
/// and zero-filled: this test is about metadata and shape derivation,
/// not payload numerics.
fn safetensors_from(entries: &[(&str, &[u64])]) -> Vec<u8> {
    let mut fields = Vec::new();
    let mut payload = Vec::new();
    for (name, shape) in entries {
        let count: u64 = shape.iter().product();
        let start = payload.len();
        payload.extend_from_slice(&vec![0u8; count as usize * 4]);
        let dims: Vec<String> = shape.iter().map(u64::to_string).collect();
        fields.push(format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{start},{}]}}",
            dims.join(","),
            payload.len()
        ));
    }
    let header = format!("{{{}}}", fields.join(","));
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&payload);
    out
}

/// A structurally complete miniature backbone under the default
/// HuggingFace Qwen2 tensor spelling:
///
/// - `n_layer = 3` (a contiguous run 0..2)
/// - `d_model = 8` (dim 1 of the embedding, corroborated by the gate
///   projections)
/// - `vocab = 16` (dim 0 of the embedding)
/// - `intermediate_size = 12` (dim 0 of the gate projection)
///
/// Deliberately NOT the binder's `tiny_for_tests()` shape: if the
/// converter ever fell back to a constant instead of deriving, matching
/// numbers would hide it.
fn synthetic_checkpoint() -> Vec<u8> {
    let mut entries: Vec<(String, Vec<u64>)> =
        vec![("model.embed_tokens.weight".to_owned(), vec![16, 8])];
    for i in 0..3 {
        entries.push((
            format!("model.layers.{i}.mlp.gate_proj.weight"),
            vec![12, 8],
        ));
        // A sibling tensor per layer, so the pass-through count is not
        // trivially equal to the layer count.
        entries.push((
            format!("model.layers.{i}.self_attn.q_proj.weight"),
            vec![8, 8],
        ));
    }
    entries.push(("model.norm.weight".to_owned(), vec![8]));
    let borrowed: Vec<(&str, &[u64])> = entries
        .iter()
        .map(|(n, s)| (n.as_str(), s.as_slice()))
        .collect();
    safetensors_from(&borrowed)
}

/// The side-car matching [`synthetic_checkpoint`]: `n_head = 4` against
/// the derived `d_model = 8` gives `head_dim = 2`, which is even.
const CONFIG_JSON: &str = r#"{
  "n_head": 4,
  "rope_max_period": 1000000.0,
  "rms_norm_eps": 1e-6,
  "sample_rate": 16000,
  "speech_encoder_dim": 24,
  "speech_decoder_dim": 20
}"#;

/// Converts a synthetic checkpoint with the real converter and binds the
/// result with the real binder.
fn convert_and_bind(
    tag: &str,
    config_json: &str,
    variant: ConvertVariant,
) -> (LlamaOmni2, PathBuf, PathBuf, PathBuf) {
    let input = tmp_path(&format!("{tag}-in"));
    let config = tmp_path(&format!("{tag}-cfg"));
    let output = tmp_path(&format!("{tag}-out"));
    std::fs::write(&input, synthetic_checkpoint()).expect("write input");
    std::fs::write(&config, config_json).expect("write config");

    convert_llama_omni2_file_with_config(&input, &config, &output, variant, None)
        .unwrap_or_else(|e| panic!("converter failed: {e}"));

    let engine = LlamaOmni2::from_path(&output).unwrap_or_else(|e| {
        panic!(
            "THE HANDSHAKE IS BROKEN: a GGUF produced by \
             `convert_llama_omni2_file_with_config` failed to load in \
             `LlamaOmni2::from_path`: {e:?}"
        )
    });
    (engine, input, config, output)
}

fn cleanup(paths: [&PathBuf; 3]) {
    for p in paths {
        let _ = std::fs::remove_file(p);
    }
}

/// THE fence. A GGUF from the converter must load in the binder written
/// for it, and every hparam must survive the trip with the value the
/// checkpoint or the side-car implied.
///
/// Before 2026-08-15 this test would have panicked inside
/// `convert_and_bind` with `InvalidArgument("backbone ill-formed
/// (n_layer=0, d_model=0, n_head=0)")`.
#[test]
fn converter_output_binds_in_the_runtime_engine() {
    let (engine, input, config, output) =
        convert_and_bind("bind", CONFIG_JSON, ConvertVariant::_7B);

    let cfg = engine.config();
    // Derived from the tensors by the converter.
    assert_eq!(
        cfg.backbone.n_layer, 3,
        "n_layer comes from the contiguous `model.layers.{{i}}` run"
    );
    assert_eq!(
        cfg.backbone.d_model, 8,
        "d_model comes from dim 1 of the token embedding"
    );
    assert_eq!(
        cfg.backbone.vocab, 16,
        "vocab comes from dim 0 of the token embedding"
    );
    assert_eq!(
        cfg.backbone.intermediate_size, 12,
        "intermediate_size comes from dim 0 of the SwiGLU gate projection"
    );
    // Taken from the side-car — the axes no tensor shape carries, which
    // are never invented.
    assert_eq!(cfg.backbone.n_head, 4);
    assert_eq!(cfg.speech_encoder_dim, 24);
    assert_eq!(cfg.speech_decoder_dim, 20);
    assert_eq!(cfg.sample_rate, 16_000);
    assert!((cfg.backbone.rope_max_period - 1_000_000.0).abs() < 1e-3);
    assert!((cfg.backbone.rms_norm_eps - 1e-6).abs() < 1e-12);
    // The variant tag round-trips across the crate boundary.
    assert_eq!(cfg.variant, LlamaOmni2Variant::_7B);

    // The derived shape is internally consistent, which is what the
    // binder's own gate demands.
    assert_eq!(cfg.backbone.head_dim(), 2);
    assert!(cfg.backbone.is_well_formed());
    cfg.validate_for_forward()
        .expect("a bound config must satisfy the forward gate");

    cleanup([&input, &config, &output]);
}

/// Every variant tag survives the crate boundary, so `--model
/// llama-omni2-32b` cannot bind as a 7B.
#[test]
fn every_variant_tag_round_trips_through_a_real_conversion() {
    for (convert_variant, runtime_variant) in [
        (ConvertVariant::_7B, LlamaOmni2Variant::_7B),
        (
            ConvertVariant::_3BBilingual,
            LlamaOmni2Variant::_3BBilingual,
        ),
        (ConvertVariant::_1_5B, LlamaOmni2Variant::_1_5B),
        (ConvertVariant::_32B, LlamaOmni2Variant::_32B),
    ] {
        let (engine, input, config, output) =
            convert_and_bind(convert_variant.tag(), CONFIG_JSON, convert_variant);
        assert_eq!(
            engine.config().variant,
            runtime_variant,
            "variant `{}` must bind as itself",
            convert_variant.tag()
        );
        cleanup([&input, &config, &output]);
    }
}

/// The load now works; the FORWARD is still a loud-partial, and this
/// test pins that boundary so the repair is not mistaken for more than
/// it is.
///
/// `converse` must return `UnsupportedOp` naming the missing primitives
/// and the primary source, never a fabricated PCM buffer a caller could
/// read as synthesised speech (FR-EX-08).
#[test]
fn forward_remains_a_loud_partial_after_a_successful_bind() {
    let (engine, input, config, output) =
        convert_and_bind("partial", CONFIG_JSON, ConvertVariant::_7B);

    let pcm = vec![0.0f32; 1_600]; // 100 ms at 16 kHz
    let Err(e) = engine.converse(&pcm) else {
        panic!(
            "converse returned Ok — the streaming S2S forward is a loud-partial and must \
             not produce audio (FR-EX-08). If the forward has genuinely landed, update \
             this test and the engine.rs BoundReason row together."
        );
    };
    let msg = match e {
        VokraError::UnsupportedOp(m) => m,
        other => panic!("expected UnsupportedOp from the loud-partial gate, got {other:?}"),
    };
    assert!(
        msg.contains("https://huggingface.co/ICTNLP/LLaMA-Omni2-7B"),
        "the loud-partial must name its primary source: {msg}"
    );
    // The message quotes the bound shape, which is only meaningful now
    // that the shape is real rather than all-zero.
    assert!(
        msg.contains("n_layer=3") && msg.contains("d_model=8"),
        "the loud-partial must quote the shape it actually bound: {msg}"
    );

    cleanup([&input, &config, &output]);
}

/// The weights bound are synthesized, and the engine says so.
///
/// This is what keeps a caller from reading a successful load as "real
/// LLaMA-Omni2 weights are in memory": the GGUF's tensors are carried
/// verbatim, but the binder does not yet map them onto its weight store
/// and builds a deterministic fixture instead.
#[test]
fn a_bound_engine_reports_its_weights_as_synthesized() {
    let (engine, input, config, output) =
        convert_and_bind("synth", CONFIG_JSON, ConvertVariant::_7B);
    assert!(
        engine.is_synthesized(),
        "the binder builds a synthesized weight store until the real tensor-name manifest \
         lands; a caller must be able to tell that apart from real weights"
    );
    cleanup([&input, &config, &output]);
}

/// A side-car whose `n_head` does not fit the derived `d_model` is
/// refused at CONVERT time, not at load time.
///
/// Catching it in the converter is what keeps the operator from
/// discovering a bad transcription only after a multi-hour vast.ai
/// conversion of a 64 GB checkpoint.
#[test]
fn a_bad_n_head_is_refused_before_a_gguf_is_written() {
    let input = tmp_path("badhead-in");
    let config = tmp_path("badhead-cfg");
    let output = tmp_path("badhead-out");
    std::fs::write(&input, synthetic_checkpoint()).expect("write input");
    // d_model is 8; 3 does not divide it.
    std::fs::write(
        &config,
        r#"{"n_head":3,"rope_max_period":1e6,"rms_norm_eps":1e-6,"sample_rate":16000,
            "speech_encoder_dim":24,"speech_decoder_dim":20}"#,
    )
    .expect("write config");

    let Err(e) =
        convert_llama_omni2_file_with_config(&input, &config, &output, ConvertVariant::_7B, None)
    else {
        panic!("an n_head that does not divide the derived d_model must be refused");
    };
    let msg = e.to_string();
    assert!(
        msg.contains("n_head") && msg.contains("does not divide"),
        "the refusal must name the offending field: {msg}"
    );
    assert!(
        !output.exists(),
        "a refused conversion must not leave a GGUF behind"
    );

    cleanup([&input, &config, &output]);
}

/// A missing required side-car field is refused, and the message names
/// the field rather than substituting a plausible number.
#[test]
fn a_missing_required_axis_is_refused_and_named() {
    let input = tmp_path("missing-in");
    let config = tmp_path("missing-cfg");
    let output = tmp_path("missing-out");
    std::fs::write(&input, synthetic_checkpoint()).expect("write input");
    // `speech_decoder_dim` omitted.
    std::fs::write(
        &config,
        r#"{"n_head":4,"rope_max_period":1e6,"rms_norm_eps":1e-6,"sample_rate":16000,
            "speech_encoder_dim":24}"#,
    )
    .expect("write config");

    let Err(e) =
        convert_llama_omni2_file_with_config(&input, &config, &output, ConvertVariant::_7B, None)
    else {
        panic!("a side-car missing `speech_decoder_dim` must be refused");
    };
    let msg = e.to_string();
    assert!(
        msg.contains("speech_decoder_dim"),
        "the refusal must name the missing field: {msg}"
    );

    cleanup([&input, &config, &output]);
}
