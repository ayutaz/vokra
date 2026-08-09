//! `SbV2Model::from_gguf` (Task 24) loader tests.
//!
//! Two of these run unconditionally without any GGUF on disk: a
//! compile-only signature pin, and a negative-path proof that a
//! well-formed-but-empty `main` GGUF fails loudly (FR-EX-08) instead of
//! panicking. The third is real-fixture gated (`#[ignore]`) — it exercises
//! the loader against the repo-root
//! `tests/fixtures/sbv2/{sbv2-v2-multilingual-base,deberta-v2-large-japanese-char-wwm,deberta-v3-large}.gguf`
//! trio (matching `reference_dump.manifest.json`'s `checkpoint` block and
//! the committed `.sha256` sidecars), which land with Task 25 (converter)
//! and Task 28 (real fixture); until then this test only proves the call
//! site compiles.

use std::path::{Path, PathBuf};

use vokra_core::VokraError;
use vokra_core::gguf::{GgufBuilder, GgufFile};
use vokra_models::sbv2::SbV2Model;

/// Repo-root-relative real-fixture directory for SBV2 loader smoke tests
/// (`tests/fixtures/sbv2/`, sibling of the existing `tests/fixtures/audio/`
/// Whisper/Voxtral convention). `CARGO_MANIFEST_DIR` is
/// `<repo>/crates/vokra-models` — `cargo test` sets a test binary's working
/// directory to the crate root, not the invocation directory, so every
/// repo-root fixture path in this workspace's parity/loader tests is built
/// this way (`parity_sbv2_real.rs`, `parity_whisper.rs`, `parity_kokoro.rs`,
/// `parity_voxtral.rs`, `parity_csm.rs`, `parity_moshi.rs`) rather than as a
/// bare relative literal.
fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("sbv2")
}

/// Compile-only sanity: `from_gguf`'s type signature is stable. Taking the
/// function as a value only type-checks if the signature matches exactly
/// (three `&GgufFile` parameters, `Result<SbV2Model>` return) — this test
/// never calls the function.
#[test]
fn from_gguf_signature_compiles() {
    let _f: fn(&GgufFile, &GgufFile, &GgufFile) -> vokra_core::Result<SbV2Model> =
        SbV2Model::from_gguf;
}

/// A `main` GGUF with no `vokra.sbv2.*` metadata at all must fail loudly
/// with `VokraError::ModelLoad` naming the very first metadata key
/// `from_gguf` reads (`vokra.sbv2.d_model`), never panic — proves the
/// FR-EX-08 loud-failure path actually fires (not just type-checks) on the
/// very first read, before any tensor is touched. `bert_ja`/`bert_en` are
/// never consulted on this path (the `d_model` read fails before either
/// BERT file's `from_gguf` is called), so the same empty file is reused for
/// all three arguments.
#[test]
fn from_gguf_on_empty_main_file_fails_loudly_naming_first_missing_key() {
    let empty = GgufFile::parse(
        GgufBuilder::new()
            .to_bytes()
            .expect("build empty gguf bytes"),
    )
    .expect("parse empty gguf");

    // `SbV2Model` (the `Ok` payload) has no `Debug` impl, so `Result::expect_err`
    // (which would need to format it) is not usable here — match directly.
    match SbV2Model::from_gguf(&empty, &empty, &empty) {
        Ok(_) => panic!("an empty main GGUF must fail to load, not succeed"),
        Err(VokraError::ModelLoad(msg)) => {
            assert!(
                msg.contains("vokra.sbv2.d_model"),
                "error message should name the first missing metadata key, got: {msg}"
            );
        }
        Err(other) => panic!("expected VokraError::ModelLoad, got {other:?}"),
    }
}

/// Builds a `main` GGUF with every required `vokra.sbv2.*` **scalar dim** key
/// present but no tensors or decoder-array metadata. Used by WP-13 unit
/// tests that assert loud-fail on a specific missing scalar hparam key
/// (mirrors the "all-except-one" fixture pattern
/// `vokra-bert`'s deberta_v2 loader tests use). Every value is a legal
/// minimum: `n_text_layers`/`n_flow_layers`/`n_sdp_layers` = 0 skips every
/// per-layer loop (each stack's `0` is exercised, per `from_gguf`'s doc),
/// `d_z` = 2 satisfies the "non-zero and even" `SbV2Flow::from_layers`
/// contract with the minimum legal value, and all other dims are non-zero.
/// Values are internally consistent but deliberately not representative of
/// any real checkpoint — these fixtures never progress past the top-level
/// metadata read stage.
fn scalar_dims_only_main() -> GgufBuilder {
    let mut b = GgufBuilder::new();
    b.add_u32("vokra.sbv2.d_model", 8);
    b.add_u32("vokra.sbv2.d_bert", 8);
    b.add_u32("vokra.sbv2.d_speaker", 8);
    b.add_u32("vokra.sbv2.n_speakers", 1);
    b.add_u32("vokra.sbv2.d_style", 8);
    b.add_u32("vokra.sbv2.d_z", 2);
    b.add_u32("vokra.sbv2.n_vocab", 4);
    b.add_u32("vokra.sbv2.n_tones", 2);
    b.add_u32("vokra.sbv2.d_ff", 8);
    b.add_u32("vokra.sbv2.n_text_layers", 0);
    b.add_u32("vokra.sbv2.n_flow_layers", 0);
    b.add_u32("vokra.sbv2.n_sdp_layers", 0);
    b.add_u32("vokra.sbv2.sample_rate", 22050);
    b
}

/// WP-13: a `main` GGUF that populates every required `vokra.sbv2.*` scalar
/// dim key but omits `vokra.sbv2.decoder.leaky_relu_slope` must fail loudly
/// with `VokraError::ModelLoad` naming the missing key, never silently
/// default to the universal jik876/hifi-gan `LRELU_SLOPE = 0.1` — FR-EX-08
/// forbids silent-wrong defaults for hparams the model architecture can
/// legitimately vary.
///
/// Rationale for the "required" classification (WP-13 audit): while `0.1`
/// is the universal jik876/hifi-gan value every sibling Vokra decoder uses
/// (`vits_ja::VITS_JA_LEAKY_RELU_SLOPE`, piper-plus's `LRELU_SLOPE`), a
/// Style-Bert-VITS2 checkpoint's decoder *could* train with a different
/// value. Silently defaulting when the GGUF omits the key would produce
/// audio that is subtly wrong (leaky-ReLU negative-slope drift) without any
/// observable signal to the caller — exactly the class of failure
/// FR-EX-08 forbids. The Vokra converter always emits this key
/// (`vokra_convert::models::sbv2::write_hparams`), so no Vokra-produced
/// GGUF is affected — a third-party GGUF that omits it now surfaces a
/// clear, named-key error rather than degrading silently.
#[test]
fn from_gguf_missing_leaky_relu_slope_fails_loudly() {
    let main = GgufFile::parse(
        scalar_dims_only_main()
            .to_bytes()
            .expect("build main gguf bytes"),
    )
    .expect("parse main gguf");
    let empty = GgufFile::parse(
        GgufBuilder::new()
            .to_bytes()
            .expect("build empty gguf bytes"),
    )
    .expect("parse empty gguf");

    // `SbV2Model` has no `Debug` impl, so `Result::expect_err` is not usable.
    match SbV2Model::from_gguf(&main, &empty, &empty) {
        Ok(_) => panic!(
            "expected loud-fail on missing vokra.sbv2.decoder.leaky_relu_slope; loader \
             silently accepted the omission (FR-EX-08 violation)"
        ),
        Err(VokraError::ModelLoad(msg)) => {
            assert!(
                msg.contains("vokra.sbv2.decoder.leaky_relu_slope"),
                "expected error to name the missing key \
                 `vokra.sbv2.decoder.leaky_relu_slope`, got: {msg}"
            );
        }
        Err(other) => panic!("expected VokraError::ModelLoad, got {other:?}"),
    }
}

/// Real-fixture gated: requires the repo-root
/// `tests/fixtures/sbv2/{sbv2-v2-multilingual-base,deberta-v2-large-japanese-char-wwm,deberta-v3-large}.gguf`
/// trio (matching `reference_dump.manifest.json`'s `checkpoint` block and
/// the committed `.sha256` sidecars), produced by Task 25's converter from
/// real Style-Bert-VITS2 v2 safetensors checkpoints and landed by Task 28.
/// Ignored by default; run with `--include-ignored` once the fixtures are
/// populated.
#[test]
#[ignore = "Task 28 real fixture"]
fn from_gguf_loads_real_sbv2_weights() {
    // Fixture filenames match `reference_dump.manifest.json` (`checkpoint`
    // block) and the committed `.sha256` sidecars — not the older
    // `{main,bert_ja,bert_en}.gguf` shorthand. Paths resolve via
    // `fixtures_dir()` (repo-root) rather than a bare relative literal so
    // the resolution is invocation-cwd-independent, matching every other
    // parity/loader test in this workspace.
    let dir = fixtures_dir();
    let main_path = dir.join("sbv2-v2-multilingual-base.gguf");
    let bert_ja_path = dir.join("deberta-v2-large-japanese-char-wwm.gguf");
    let bert_en_path = dir.join("deberta-v3-large.gguf");

    let main =
        GgufFile::open(&main_path).unwrap_or_else(|e| panic!("{}: {e}", main_path.display()));
    let bert_ja =
        GgufFile::open(&bert_ja_path).unwrap_or_else(|e| panic!("{}: {e}", bert_ja_path.display()));
    let bert_en =
        GgufFile::open(&bert_en_path).unwrap_or_else(|e| panic!("{}: {e}", bert_en_path.display()));

    // Sanity: the loader walks a real checkpoint's metadata/tensor shape
    // end to end without erroring. Per-tensor numeric parity against the
    // Python reference (d_model, n_speakers, ... and every weight value)
    // is Task 27 (synthetic) / Task 28's own dedicated parity test's job,
    // not this loader smoke test's.
    SbV2Model::from_gguf(&main, &bert_ja, &bert_en)
        .unwrap_or_else(|e| panic!("SbV2Model::from_gguf: {e}"));
}
