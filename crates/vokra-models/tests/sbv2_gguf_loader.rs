//! `SbV2Model::from_gguf` (Task 24) loader tests.
//!
//! Two of these run unconditionally without any GGUF on disk: a
//! compile-only signature pin, and a negative-path proof that a
//! well-formed-but-empty `main` GGUF fails loudly (FR-EX-08) instead of
//! panicking. The third is real-fixture gated (`#[ignore]`) — it exercises
//! the loader against `tests/fixtures/sbv2/{main,bert_ja,bert_en}.gguf`,
//! which land with Task 25 (converter) + Task 28 (real fixture); until
//! then this test only proves the call site compiles.

use vokra_core::VokraError;
use vokra_core::gguf::{GgufBuilder, GgufFile};
use vokra_models::sbv2::SbV2Model;

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

/// Real-fixture gated: requires
/// `tests/fixtures/sbv2/{main.gguf,bert_ja.gguf,bert_en.gguf}`, produced by
/// Task 25's converter from real Style-Bert-VITS2 v2 safetensors
/// checkpoints and landed by Task 28. Ignored by default; run with
/// `--include-ignored` once the fixtures are populated.
#[test]
#[ignore = "Task 28 real fixture"]
fn from_gguf_loads_real_sbv2_weights() {
    let main = GgufFile::open("tests/fixtures/sbv2/main.gguf").expect("main.gguf");
    let bert_ja = GgufFile::open("tests/fixtures/sbv2/bert_ja.gguf").expect("bert_ja.gguf");
    let bert_en = GgufFile::open("tests/fixtures/sbv2/bert_en.gguf").expect("bert_en.gguf");

    // Sanity: the loader walks a real checkpoint's metadata/tensor shape
    // end to end without erroring. Per-tensor numeric parity against the
    // Python reference (d_model, n_speakers, ... and every weight value)
    // is Task 27 (synthetic) / Task 28's own dedicated parity test's job,
    // not this loader smoke test's.
    SbV2Model::from_gguf(&main, &bert_ja, &bert_en).expect("loads");
}
