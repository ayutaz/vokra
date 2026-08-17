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
use vokra_models::sbv2::{EXPECTED_ARCH as SBV2_ARCH, SbV2Model};

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

/// WP-19: `from_gguf_with_zh_bert`'s 4-file signature is stable. Same
/// compile-only pin as [`from_gguf_signature_compiles`] above, extended
/// with the fourth `bert_zh: &GgufFile` argument. The two signatures must
/// stay side-by-side so a caller with only JA/EN GGUFs keeps compiling
/// against `from_gguf`, while a caller with a ZH GGUF opts in via
/// `from_gguf_with_zh_bert` — the 3-file path must remain backward
/// compatible (this is a load-time API contract the WP-19 land pins).
#[test]
fn from_gguf_with_zh_bert_signature_compiles() {
    let _f: fn(&GgufFile, &GgufFile, &GgufFile, &GgufFile) -> vokra_core::Result<SbV2Model> =
        SbV2Model::from_gguf_with_zh_bert;
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
    // `main` carries the arch stamp and nothing else: the loader gates
    // `vokra.model.arch` before any metadata read (FR-EX-08), so an
    // unstamped `main` would stop at the arch gate and this test would
    // pass for the wrong reason. `bert_ja` / `bert_en` stay empty — they
    // are never consulted on this path.
    let mut mb = GgufBuilder::new();
    mb.add_string(vokra_core::gguf::chunks::KEY_MODEL_ARCH, SBV2_ARCH);
    let arch_only_main =
        GgufFile::parse(mb.to_bytes().expect("build main gguf bytes")).expect("parse main gguf");

    // `SbV2Model` (the `Ok` payload) has no `Debug` impl, so `Result::expect_err`
    // (which would need to format it) is not usable here — match directly.
    match SbV2Model::from_gguf(&arch_only_main, &empty, &empty) {
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

/// FR-EX-08: a `main` GGUF with no `vokra.model.arch` at all is not a
/// Vokra-native Style-Bert-VITS2 artifact and must be refused by name,
/// before any `vokra.sbv2.*` read. Without this gate the loader would
/// bind whatever `sbv2.*`-shaped tensor names happened to overlap.
#[test]
fn from_gguf_rejects_main_without_arch_stamp() {
    let empty = GgufFile::parse(
        GgufBuilder::new()
            .to_bytes()
            .expect("build empty gguf bytes"),
    )
    .expect("parse empty gguf");

    match SbV2Model::from_gguf(&empty, &empty, &empty) {
        Ok(_) => panic!("an unstamped main GGUF must not load"),
        Err(VokraError::ModelLoad(msg)) => {
            assert!(
                msg.contains("vokra.model.arch"),
                "error must name the missing key, got: {msg}"
            );
            assert!(
                msg.contains(SBV2_ARCH),
                "error must name the expected arch, got: {msg}"
            );
        }
        Err(other) => panic!("expected VokraError::ModelLoad, got {other:?}"),
    }
}

/// FR-EX-08: a foreign-arch `main` must be refused with BOTH the expected
/// and the actual tag named. `deberta_v2` is the live failure mode here —
/// it is the arch of this loader's own `bert_ja` argument, so an
/// argument-order slip lands exactly this GGUF in the `main` slot.
#[test]
fn from_gguf_rejects_foreign_main_arch_naming_expected_and_actual() {
    let mut b = GgufBuilder::new();
    b.add_string(vokra_core::gguf::chunks::KEY_MODEL_ARCH, "deberta_v2");
    let main =
        GgufFile::parse(b.to_bytes().expect("build main gguf bytes")).expect("parse main gguf");
    let empty = GgufFile::parse(
        GgufBuilder::new()
            .to_bytes()
            .expect("build empty gguf bytes"),
    )
    .expect("parse empty gguf");

    match SbV2Model::from_gguf(&main, &empty, &empty) {
        Ok(_) => panic!("a foreign-arch main GGUF must not load"),
        Err(VokraError::ModelLoad(msg)) => {
            assert!(
                msg.contains("deberta_v2"),
                "error must name the actual arch, got: {msg}"
            );
            assert!(
                msg.contains(SBV2_ARCH),
                "error must name the expected arch, got: {msg}"
            );
        }
        Err(other) => panic!("expected VokraError::ModelLoad, got {other:?}"),
    }
}

/// The same arch gate guards the 4-file WP-19 entry point — the three
/// public loaders share `from_gguf_inner`, so this pins that they cannot
/// drift apart.
#[test]
fn from_gguf_with_zh_bert_rejects_foreign_main_arch() {
    let mut b = GgufBuilder::new();
    b.add_string(
        vokra_core::gguf::chunks::KEY_MODEL_ARCH,
        "piper-plus-mb-istft-vits2",
    );
    let main =
        GgufFile::parse(b.to_bytes().expect("build main gguf bytes")).expect("parse main gguf");
    let empty = GgufFile::parse(
        GgufBuilder::new()
            .to_bytes()
            .expect("build empty gguf bytes"),
    )
    .expect("parse empty gguf");

    match SbV2Model::from_gguf_with_zh_bert(&main, &empty, &empty, &empty) {
        Ok(_) => panic!("a foreign-arch main GGUF must not load"),
        Err(VokraError::ModelLoad(msg)) => {
            assert!(
                msg.contains("piper-plus-mb-istft-vits2") && msg.contains(SBV2_ARCH),
                "error must name both the actual and expected arch, got: {msg}"
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
    // The loader gates `vokra.model.arch` before any metadata read
    // (FR-EX-08), so every `main` fixture that expects to reach a
    // *metadata* assertion must carry the stamp.
    b.add_string(vokra_core::gguf::chunks::KEY_MODEL_ARCH, SBV2_ARCH);
    b.add_u32("vokra.sbv2.d_model", 8);
    b.add_u32("vokra.sbv2.d_bert", 8);
    b.add_u32("vokra.sbv2.d_speaker", 8);
    b.add_u32("vokra.sbv2.n_speakers", 1);
    b.add_u32("vokra.sbv2.d_style", 8);
    b.add_u32("vokra.sbv2.d_z", 2);
    b.add_u32("vokra.sbv2.n_vocab", 4);
    b.add_u32("vokra.sbv2.n_tones", 2);
    b.add_u32("vokra.sbv2.d_ff", 8);
    b.add_u32("vokra.sbv2.n_heads", 1);
    b.add_u32("vokra.sbv2.window_size", 4);
    b.add_u32("vokra.sbv2.kernel_ffn", 3);
    b.add_u32("vokra.sbv2.n_text_layers", 0);
    b.add_u32("vokra.sbv2.n_flow_layers", 0);
    b.add_u32("vokra.sbv2.n_sdp_layers", 0);
    b.add_u32("vokra.sbv2.sample_rate", 22050);
    b
}

/// WP-19: the 4-file variant fires `bert_zh`'s own metadata read the same
/// way the 3-file variant fires `bert_ja` / `bert_en`. Feeding an empty ZH
/// GGUF here must surface a `VokraError::ModelLoad` naming a
/// `vokra.bert_base.*` key (the very first key `BertBaseEncoder::from_gguf`
/// reads is `vokra.bert_base.n_layers`) — proving `bert_zh` is actually
/// consulted, not silently dropped on the floor.
///
/// The 3-file `main` fixture (`scalar_dims_only_main`) never touches the
/// BERT files (fails earlier on `vokra.sbv2.decoder.leaky_relu_slope`), so
/// this test uses a `main` fixture that does have every dim+decoder scalar
/// wired — it's the WP-13 fixture plus the `leaky_relu_slope` key, so the
/// loader proceeds far enough that the ZH BERT loader runs. `main`'s
/// tensor tables are still empty (no `sbv2.text_encoder.*` weights), so
/// the ZH BERT read is the FIRST thing that fails on the WP-19 path — that
/// is exactly what this test wants to observe (any earlier failure would
/// leave the WP-19 gate uncovered).
#[test]
fn from_gguf_with_zh_bert_on_empty_zh_file_touches_zh_loader() {
    // `scalar_dims_only_main` + `leaky_relu_slope` (so `from_gguf` reaches
    // the BERT load stage before failing).
    let mut mb = scalar_dims_only_main();
    mb.add_f32("vokra.sbv2.decoder.leaky_relu_slope", 0.1);
    let main =
        GgufFile::parse(mb.to_bytes().expect("build main gguf bytes")).expect("parse main gguf");
    let empty = GgufFile::parse(
        GgufBuilder::new()
            .to_bytes()
            .expect("build empty gguf bytes"),
    )
    .expect("parse empty gguf");

    // With every ZH-side arg set to the empty file, the load must fail
    // loudly. This proves `bert_zh` is threaded into `from_gguf_inner`'s
    // BERT stage — a silent `bert_zh` drop would produce the same error
    // the 3-file `from_gguf` produces (which fails earlier on the empty
    // `bert_ja` file's `vokra.deberta_v2.n_layers` read), and would leave
    // this test unable to distinguish the two paths.
    match SbV2Model::from_gguf_with_zh_bert(&main, &empty, &empty, &empty) {
        Ok(_) => panic!("an empty ZH BERT GGUF must fail to load in the WP-19 4-file variant"),
        Err(VokraError::ModelLoad(_msg)) => {
            // The exact key named depends on the BERT loader order (JA is
            // consulted first in the current `from_gguf_inner`); the
            // load must fail, and the failure must be a ModelLoad — which
            // proves the JA→EN→ZH BERT stage was reached. A silent
            // `bert_zh` drop would still ModelLoad on JA/EN but a
            // subsequent `synthesize` on a ZH request would then also
            // fail because the ZH tokenizer never populated — this test
            // catches only the load-time signal, sibling
            // `synthesize_zh_without_wired_zh_bert_fails_loudly` in
            // `sbv2_model_synthetic.rs` catches the synthesize-time
            // signal.
        }
        Err(other) => panic!("expected VokraError::ModelLoad, got {other:?}"),
    }
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
/// (`write_hparams` in `crates/vokra-convert/src/models/sbv2.rs`), so no Vokra-produced
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

/// Blocker 2b TDD-hardening (2026-08-10) — builds a `main` GGUF with
/// every required scalar dim key present AND `n_flow_layers = 1` (nonzero,
/// so `SbV2Model::from_gguf` takes the flow-hparam-read branch), plus the
/// four `vokra.sbv2.flow.*` keys except the one named in `omit`. Every
/// non-omitted value is a legal minimum: `n_encoder_layers = 6` +
/// `kernel_ffn = 5` + `gin_channels = 8` (non-zero) + `mean_only = true`
/// mirror the base checkpoint's shape at hparam scale.
///
/// The `omit` argument is one of `"vokra.sbv2.flow.n_encoder_layers"`,
/// `"vokra.sbv2.flow.kernel_ffn"`, `"vokra.sbv2.flow.gin_channels"`, or
/// `"vokra.sbv2.flow.mean_only"` — each of the four flow-hparam keys the
/// loader reads at lines 2341-2352 (see `crates/vokra-models/src/sbv2/mod.rs`).
/// Any other value (or `None`) is a helper misuse for these tests.
///
/// The loader stops on the FIRST missing flow-hparam key it reads
/// (read order = `n_encoder_layers → kernel_ffn → gin_channels →
/// mean_only`), so omitting one key while including all three others
/// exposes exactly that key's error message. This is the metadata-key
/// spelling-contract pin between the converter's `KEY_FLOW_*` constants
/// (`crates/vokra-convert/src/models/sbv2.rs` lines 378-381) and the
/// loader's `require_u32` / `.and_then(|v| v.as_bool())` reads: any typo
/// on either side that survives converter tests surfaces here as a
/// wrongly-named error message.
fn n_flow_layers_1_main_omit_flow_key(omit: &str) -> GgufBuilder {
    let mut b = scalar_dims_only_main();
    // Nonzero n_flow_layers triggers the flow-hparam read block.
    b.add_u32("vokra.sbv2.n_flow_layers", 1);
    // Add the four flow-hparam keys, skipping the one named in `omit`.
    for (key, val) in [
        ("vokra.sbv2.flow.n_encoder_layers", 6u32),
        ("vokra.sbv2.flow.kernel_ffn", 5),
        ("vokra.sbv2.flow.gin_channels", 8),
    ] {
        if key != omit {
            b.add_u32(key, val);
        }
    }
    if omit != "vokra.sbv2.flow.mean_only" {
        b.add_bool("vokra.sbv2.flow.mean_only", true);
    }
    b
}

/// Blocker 2b TDD-hardening (2026-08-10) — a `main` GGUF with every
/// non-flow scalar hparam present + `n_flow_layers = 1` but MISSING
/// `vokra.sbv2.flow.n_encoder_layers` must fail loudly with
/// `VokraError::ModelLoad` naming the exact missing key. This pins the
/// FIRST of the four flow-hparam keys `SbV2Model::from_gguf` reads
/// (`crates/vokra-models/src/sbv2/mod.rs` line 2341, via `require_u32`);
/// a converter typo like `flow.n_encoder_layer` (missing `s`) or a loader
/// typo like `flow.num_encoder_layers` would fail this test with a
/// wrongly-named key — the two spellings must match byte-for-byte.
///
/// `bert_ja` / `bert_en` are never consulted on this path (the flow-hparam
/// read fails before either BERT file's `from_gguf` is called), so the same
/// empty file is reused for all three arguments (matches the
/// `from_gguf_on_empty_main_file_fails_loudly_naming_first_missing_key`
/// pattern above).
#[test]
fn from_gguf_positive_n_flow_layers_missing_flow_n_encoder_layers_fails_loudly() {
    let main = GgufFile::parse(
        n_flow_layers_1_main_omit_flow_key("vokra.sbv2.flow.n_encoder_layers")
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

    match SbV2Model::from_gguf(&main, &empty, &empty) {
        Ok(_) => panic!(
            "expected loud-fail on missing vokra.sbv2.flow.n_encoder_layers; loader \
             silently accepted the omission (FR-EX-08 violation)"
        ),
        Err(VokraError::ModelLoad(msg)) => {
            assert!(
                msg.contains("vokra.sbv2.flow.n_encoder_layers"),
                "expected error to name the missing key \
                 `vokra.sbv2.flow.n_encoder_layers`, got: {msg}"
            );
        }
        Err(other) => panic!("expected VokraError::ModelLoad, got {other:?}"),
    }
}

/// Blocker 2b TDD-hardening (2026-08-10) — sibling of
/// `from_gguf_positive_n_flow_layers_missing_flow_n_encoder_layers_fails_loudly`
/// for the SECOND flow-hparam key (`vokra.sbv2.flow.kernel_ffn`). Distinct
/// from the top-level `vokra.sbv2.kernel_ffn` (which is the text encoder's
/// FFN kernel width, = 3 on the base ckpt) — the flow's inner encoder
/// stack ships a different value (= 5 on the base ckpt), see
/// `crates/vokra-models/src/sbv2/flow.rs`'s module doc for why the two
/// live under separate metadata keys.
#[test]
fn from_gguf_positive_n_flow_layers_missing_flow_kernel_ffn_fails_loudly() {
    let main = GgufFile::parse(
        n_flow_layers_1_main_omit_flow_key("vokra.sbv2.flow.kernel_ffn")
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

    match SbV2Model::from_gguf(&main, &empty, &empty) {
        Ok(_) => panic!(
            "expected loud-fail on missing vokra.sbv2.flow.kernel_ffn; loader \
             silently accepted the omission (FR-EX-08 violation)"
        ),
        Err(VokraError::ModelLoad(msg)) => {
            assert!(
                msg.contains("vokra.sbv2.flow.kernel_ffn"),
                "expected error to name the missing key \
                 `vokra.sbv2.flow.kernel_ffn`, got: {msg}"
            );
        }
        Err(other) => panic!("expected VokraError::ModelLoad, got {other:?}"),
    }
}

/// Blocker 2b TDD-hardening (2026-08-10) — sibling of
/// `from_gguf_positive_n_flow_layers_missing_flow_n_encoder_layers_fails_loudly`
/// for the THIRD flow-hparam key (`vokra.sbv2.flow.gin_channels`). This is
/// the `g` (per-utterance conditioning vector) input dimension threaded
/// through every coupling layer's `spk_emb_linear` projection — a silent
/// zero-default would leave every coupling accepting an empty `g` and
/// producing arithmetic-wrong output with no observable signal (FR-EX-08).
#[test]
fn from_gguf_positive_n_flow_layers_missing_flow_gin_channels_fails_loudly() {
    let main = GgufFile::parse(
        n_flow_layers_1_main_omit_flow_key("vokra.sbv2.flow.gin_channels")
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

    match SbV2Model::from_gguf(&main, &empty, &empty) {
        Ok(_) => panic!(
            "expected loud-fail on missing vokra.sbv2.flow.gin_channels; loader \
             silently accepted the omission (FR-EX-08 violation)"
        ),
        Err(VokraError::ModelLoad(msg)) => {
            assert!(
                msg.contains("vokra.sbv2.flow.gin_channels"),
                "expected error to name the missing key \
                 `vokra.sbv2.flow.gin_channels`, got: {msg}"
            );
        }
        Err(other) => panic!("expected VokraError::ModelLoad, got {other:?}"),
    }
}

/// Blocker 2b TDD-hardening (2026-08-10) — sibling of
/// `from_gguf_positive_n_flow_layers_missing_flow_n_encoder_layers_fails_loudly`
/// for the FOURTH and last flow-hparam key (`vokra.sbv2.flow.mean_only`),
/// with two differences from its three siblings: (a) this key is a `bool`
/// not a `u32` (the loader uses `.and_then(|v| v.as_bool()).ok_or_else(...)`
/// at lines 2344-2352, NOT `require_u32`), and (b) the error message
/// hand-formats a distinct suffix `(bool)` to make the type mismatch
/// diagnostic clearer than the generic `require_u32` message. This test
/// pins BOTH the key name AND the `(bool)` suffix.
///
/// A silent default here (say, `false`) would cause the coupling to
/// consume an extra `half_d_z` channels from `post`'s output that the
/// converted GGUF's `post.weight` shape doesn't provide — either loud-fail
/// downstream on shape mismatch (best case) or produce silently-wrong
/// output for a fine-tune ckpt that happens to have matching shape by
/// coincidence (worst case). FR-EX-08 requires the load-time error.
#[test]
fn from_gguf_positive_n_flow_layers_missing_flow_mean_only_fails_loudly() {
    let main = GgufFile::parse(
        n_flow_layers_1_main_omit_flow_key("vokra.sbv2.flow.mean_only")
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

    match SbV2Model::from_gguf(&main, &empty, &empty) {
        Ok(_) => panic!(
            "expected loud-fail on missing vokra.sbv2.flow.mean_only; loader \
             silently accepted the omission (FR-EX-08 violation)"
        ),
        Err(VokraError::ModelLoad(msg)) => {
            assert!(
                msg.contains("vokra.sbv2.flow.mean_only"),
                "expected error to name the missing key \
                 `vokra.sbv2.flow.mean_only`, got: {msg}"
            );
            // The `mean_only` read uses a bespoke error string with a
            // `(bool)` suffix (loader line 2349) — pinning this makes a
            // silent refactor that drops the type-hint diagnostic caught
            // here rather than by a future reader trying to understand a
            // stale error message.
            assert!(
                msg.contains("(bool)"),
                "expected error to include the `(bool)` type-hint suffix, got: {msg}"
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

/// Wave 1 T11 verification (Blocker 5 sealed): loads all 3 real fixture
/// GGUFs via `SbV2Model::from_gguf` and verifies both BERT tokenizer
/// schemes are accessible and dispatch correctly. JA BERT uses
/// `wordpiece_char` (character-based tokenization for DeBERTa v2), EN BERT
/// uses `sentencepiece_bpe` (subword-piece tokenization for DeBERTa v3).
/// Tests that both tokenizers load successfully from the real fixtures and
/// that the tokenizer kind metadata is stamped correctly — proves the scheme
/// dispatch and load path work end-to-end.
#[test]
#[ignore = "Task 28 real fixture"]
fn sbv2_model_from_gguf_dispatches_both_bert_tokenizers() {
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

    // Load the model from real GGUFs — this exercises the SbV2Model::from_gguf
    // loader path that wires the SbertTokenizer for both JA and EN with the
    // correct tokenizer schemes based on the `vokra.bert.tokenizer.kind`
    // metadata in each BERT file.
    let _model = SbV2Model::from_gguf(&main, &bert_ja, &bert_en)
        .expect("SbV2Model::from_gguf should load successfully with real fixtures");

    // Verify tokenizer schemes by reading the metadata directly from the
    // BERT GGUF files (where `vokra.bert.tokenizer.kind` is stamped by the
    // converter during Task 10). This confirms both tokenizer kinds are
    // present and correctly indicate the tokenization scheme.
    let ja_kind = bert_ja
        .get("vokra.bert.tokenizer.kind")
        .and_then(|v| v.as_str())
        .expect("bert_ja should have vokra.bert.tokenizer.kind metadata");
    let en_kind = bert_en
        .get("vokra.bert.tokenizer.kind")
        .and_then(|v| v.as_str())
        .expect("bert_en should have vokra.bert.tokenizer.kind metadata");

    // JA (DeBERTa v2, char-based): bert-charsplit scheme.
    // EN (DeBERTa v3, subword-piece): sentencepiece-unigram scheme.
    // These values are stamped by the Task 10 converter and reflect the
    // tokenization algorithm each BERT variant uses.
    assert_eq!(
        ja_kind, "bert-charsplit",
        "JA BERT tokenizer should use bert-charsplit scheme (char-based for DeBERTa v2)"
    );
    assert_eq!(
        en_kind, "sentencepiece-unigram",
        "EN BERT tokenizer should use sentencepiece-unigram scheme (SentencePiece unigram for DeBERTa v3)"
    );
}

/// Blocker 2c defensive check (2026-08-10) — a `main` GGUF that carries
/// an anomalous `sbv2.sdp.flows.<even>.<w>` tensor must fail loudly with
/// `VokraError::ModelLoad` naming the offending tensor. Upstream VITS-SDP
/// architecture puts `Flip` (parameter-free) modules at even flow slots,
/// so no such production tensor should ever survive the converter's
/// `rewrite_sdp_tensor_name` — the converter maps even-index survivors
/// verbatim to `sbv2.sdp.flows.<even>.<w>` (the loud-detect path) and
/// the loader (`crates/vokra-models/src/sbv2/mod.rs`, lines near the top
/// of `from_gguf_inner`) rejects them here.
///
/// This test needs only the anomalous tensor + a valid GGUF header —
/// the loader's format-anomaly check runs BEFORE any metadata read, so
/// no scalar hparams are required. That makes the test cheap: it does
/// not need to reproduce a fully-loadable SBV2 v2 config, only the
/// anomaly itself.
///
/// If real checkpoints ever legitimately ship `sbv2.sdp.flows.*`
/// tensors (an SDP variant that stores parameters at even slots), this
/// check must be relaxed and this test updated with the rationale.
/// Until then it acts as a canary for converter regressions in
/// `crates/vokra-convert/src/models/sbv2.rs::rewrite_sdp_tensor_name`
/// (Blocker 2c) or checkpoint format corruption.
#[test]
fn from_gguf_rejects_anomalous_sdp_flows_even_index_tensor() {
    use vokra_core::gguf::GgmlType;
    let mut b = GgufBuilder::new();
    // Arch stamp — the loader gates it ahead of the format-anomaly walk
    // (FR-EX-08), so this fixture must carry it to reach the walk at all.
    b.add_string(vokra_core::gguf::chunks::KEY_MODEL_ARCH, SBV2_ARCH);
    // A 4-byte F32 tensor is enough to trip the check — the loader's
    // format-anomaly walk runs on tensor NAMES, not shapes.
    b.add_tensor(
        "sbv2.sdp.flows.2.pre.weight",
        GgmlType::F32,
        vec![1u64],
        vec![0u8; 4],
    )
    .expect("add anomalous flow tensor");
    let main =
        GgufFile::parse(b.to_bytes().expect("build main gguf bytes")).expect("parse main gguf");
    let empty = GgufFile::parse(
        GgufBuilder::new()
            .to_bytes()
            .expect("build empty gguf bytes"),
    )
    .expect("parse empty gguf");

    match SbV2Model::from_gguf(&main, &empty, &empty) {
        Ok(_) => panic!(
            "expected loud-fail on anomalous `sbv2.sdp.flows.2.pre.weight` tensor; loader \
             silently accepted the anomaly (FR-EX-08 violation — would be dropped without \
             the format-anomaly check in from_gguf_inner)"
        ),
        Err(VokraError::ModelLoad(msg)) => {
            assert!(
                msg.contains("sbv2.sdp.flows.2.pre.weight"),
                "expected error to name the offending tensor \
                 `sbv2.sdp.flows.2.pre.weight`, got: {msg}"
            );
            assert!(
                msg.contains("Flip"),
                "expected error to explain the Flip-at-even-slots invariant, got: {msg}"
            );
            // FR-EX-08 attribution should be present so a future maintainer
            // reads the pattern (no silent-wrong) and does not disable the
            // check without recording a rationale.
            assert!(
                msg.contains("FR-EX-08"),
                "expected error to cite FR-EX-08, got: {msg}"
            );
        }
        Err(other) => panic!("expected VokraError::ModelLoad, got {other:?}"),
    }
}
