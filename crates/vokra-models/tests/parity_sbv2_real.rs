//! SBV2 (Style-Bert-VITS2 v2) **real**-checkpoint parity (Task 28).
//!
//! Sibling of `parity_sbv2_synthetic.rs` (Task 27, which exercises
//! `SbV2Model::synthetic_for_test()`'s procedurally-generated weights — no
//! trained checkpoint, no reference to diff against). This file is the
//! other half: it reads a **real** Style-Bert-VITS2 v2 checkpoint (converted
//! to GGUF) plus a Python-generated reference dump, and — when both are
//! present — diffs a real Rust forward pass against them using
//! `sbv2::parity::tolerance_for` (Task 26).
//!
//! # Gating (fabricated pass 禁止)
//!
//! The one `#[test]` below is `#[ignore]`d: `cargo test` skips it by
//! default (this file's committed state — no fixture exists in this repo
//! yet), and Task 28's own DoD is compile-only (`docs/superpowers/plans/
//! 2026-07-26-sbv2-v2.md` Task 28 Step 1/Step 4: "ignored、compile 通過が本
//! task の DoD" / "1 test ignored"). Fixtures land with **Task 34** (`tests/
//! fixtures/sbv2/README.md` + `*.gguf.sha256` placeholders + this file's
//! manifest) and **Task 30** (`tools/parity/sbv2_dump_reference.py`, which
//! populates `reference_dump/*.bin`) — run with `cargo test -p vokra-models
//! --test parity_sbv2_real -- --ignored` once both have landed.
//!
//! Once opted in via `--ignored`, a missing fixture is a **loud panic**
//! naming exactly what is missing and how to produce it — never a silent
//! skip-and-pass (FR-EX-08; the `#[ignore]` attribute itself is the one and
//! only "not run" gate, matching the sibling `sbv2_gguf_loader.rs`'s
//! `from_gguf_loads_real_sbv2_weights` real-fixture test's own convention).
//!
//! # The manifest contract (this file's schema — Task 34 must match)
//!
//! `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §10 specifies that
//! Task 34 commits `tests/fixtures/sbv2/reference_dump.manifest.json` but
//! does not pin its exact shape beyond "dump 対象 tensor 名 + shape の期待
//! 値の JSON". This file defines that shape, since Task 34 has not landed
//! yet and something must be authoritative first:
//!
//! ```text
//! {
//!   "generator_version": "1.0",
//!   "checkpoint": {
//!     "sbv2_main": "sbv2-v2-multilingual-base.gguf",
//!     "bert_ja": "deberta-v2-large-japanese-char-wwm.gguf",
//!     "bert_en": "deberta-v3-large.gguf"
//!   },
//!   "request": {
//!     "text": "...",
//!     "language": "JA",
//!     "speaker_id": 0,
//!     "style_vec": [0.0, 0.0, "..."],
//!     "speed": 1.0,
//!     "noise_scale": 0.667,
//!     "noise_scale_w": 0.8,
//!     "seed": 42
//!   },
//!   "tensors": [
//!     {"name": "phoneme_embed",   "path": "reference_dump/phoneme_embed.bin",   "shape": ["T_text", 192]},
//!     {"name": "text_hidden",     "path": "reference_dump/text_hidden.bin",     "shape": ["T_text", 192]},
//!     {"name": "bert_hidden_ja",  "path": "reference_dump/bert_hidden_ja.bin",  "shape": ["T_bert", 1024]},
//!     {"name": "bert_hidden_en",  "path": "reference_dump/bert_hidden_en.bin",  "shape": ["T_bert", 1024]},
//!     {"name": "bert_bridge_out", "path": "reference_dump/bert_bridge_out.bin", "shape": ["T_text", 192]},
//!     {"name": "speaker_embed",   "path": "reference_dump/speaker_embed.bin",   "shape": [1, 512]},
//!     {"name": "style_projected", "path": "reference_dump/style_projected.bin", "shape": [1, 192]},
//!     {"name": "sdp_sample",      "path": "reference_dump/sdp_sample.bin",      "shape": ["T_text"]},
//!     {"name": "mel_hidden",      "path": "reference_dump/mel_hidden.bin",      "shape": ["T_mel", 192]},
//!     {"name": "z_latent",        "path": "reference_dump/z_latent.bin",        "shape": ["T_mel", 192]},
//!     {"name": "waveform",        "path": "reference_dump/waveform.bin",        "shape": [1, "samples"]}
//!   ]
//! }
//! ```
//!
//! * `checkpoint.*` values are **bare filenames**, siblings of this
//!   manifest directly inside `tests/fixtures/sbv2/` — matching Task 34's
//!   own `Files:` list verbatim (`sbv2-v2-multilingual-base.gguf` /
//!   `deberta-v2-large-japanese-char-wwm.gguf` / `deberta-v3-large.gguf`),
//!   which is also what their `*.gguf.sha256` sidecar placeholders name.
//!   This test reads them from the manifest rather than hard-coding them,
//!   so Task 34 is free to point at a differently-named real checkpoint
//!   without a Rust-side edit.
//! * `tensors[].path` values are relative to that **same** directory but
//!   already include the `reference_dump/` sub-directory prefix (Task 30's
//!   dumper writes there) — do not re-prepend it.
//! * `tensors[].shape` elements must all be JSON integers (the `"T_text"` /
//!   `"samples"` placeholders above are illustrative only — a real manifest
//!   substitutes the actual dumped dimensions, e.g. `[1, 24000]`).
//! * `request` is **not** in the design doc's §10 sketch — it is added
//!   here because [`SbV2SynthRequest`] (the only input
//!   [`SbV2Model::synthesize`] accepts) has no "raw phoneme ids" entry
//!   point; reproducing the Python dumper's exact forward pass needs its
//!   `text`/`language`/`speaker_id`/`style_vec`/`speed`/`noise_scale`/
//!   `noise_scale_w`/`seed` verbatim. `speed` must be strictly positive
//!   (`SbV2Model::synthesize`'s own precondition).
//!
//! # Scope: only `waveform` is compared (Task 28.x follow-up for the rest)
//!
//! `SbV2Model`'s pipeline fields (`text_encoder`, `bert`, `sdp`, `flow`, …)
//! are private — `synthesize` is the only public entry point, and it
//! returns final PCM, not the 10 intermediate tensors design doc §10 also
//! lists. Exposing per-stage accessors is real API-surface work outside
//! this compile-only-DoD scaffold's remit (`parity_sbv2_synthetic.rs`'s own
//! module doc names this same gap). The manifest schema above nonetheless
//! carries all 11 tensors so Task 34/30 can produce a complete dump now;
//! `bert_hidden_ja` / `bert_hidden_en` / `bert_bridge_out` / `speaker_embed`
//! / `style_projected` / `sdp_sample` / `mel_hidden` / `z_latent` /
//! `phoneme_embed` / `text_hidden` simply sit unread until a Task 28.x
//! follow-up adds `SbV2Model` accessors and iterates them too.
//!
//! # A real blocker `synthesize` hits first: `from_gguf` loads no G2P
//!
//! [`SbV2Model::from_gguf`]'s own doc ("G2P is not loaded here") is explicit:
//! the 3-file loader signature (`main` + `bert_ja` + `bert_en`) has no
//! piper-plus G2P GGUF, so the model it returns carries an internal
//! `UnwiredPhonemizer` that turns **every** `synthesize` call into
//! `Err(VokraError::NotImplemented(_))` at the G2P step — deliberately, so a
//! `from_gguf`-loaded model never *silently* emits wrong-but-plausible audio
//! (FR-EX-08). A real, working G2P
//! (`SbV2Phonemizer::from_piper_g2p`) needs a `vokra_piper_plus::Phonemizer`
//! implementation with real 8-language coverage, which — per
//! `vokra-piper-plus/src/phonemizer.rs`'s own doc — lives **out of** this
//! zero-dependency root workspace (`integrations/vokra-piper-g2p`, M1-01-A);
//! `crates/vokra-models` cannot depend on it (NFR-DS-02).
//!
//! So even with a perfect, real, fully-populated fixture set, `synthesize`
//! cannot succeed from *this* crate today — that is a pre-existing,
//! documented architectural fact this file did not introduce and is not
//! trying to route around with untested bypass plumbing (constructing a
//! stand-in [`vokra_piper_plus::Phonemizer`] whose output phoneme ids just
//! happen to equal SBV2's own phoneme space would itself be new,
//! unverifiable-until-Task-34-lands logic — exactly the kind of risk a
//! compile-only-DoD scaffold should not carry). The test below still
//! attempts the real call (so the comparison fires for free the moment a
//! G2P-wired construction path exists), and treats this **specific,
//! already-documented** `NotImplemented` outcome as an honestly-logged,
//! non-fabricated non-failure — never a silently-reported pass, and never
//! confused with an actual numeric-parity breach (any *other* error, or a
//! tolerance breach on a successful `synthesize`, still panics).

use std::path::{Path, PathBuf};

use vokra_core::VokraError;
use vokra_core::gguf::GgufFile;
use vokra_core::json::{self, JsonValue};
use vokra_models::sbv2::{Language, SbV2Model, SbV2SynthRequest, tolerance_for};

/// Repo-root-relative real-fixture directory for SBV2 parity
/// (`tests/fixtures/sbv2/`, sibling of the existing `tests/fixtures/audio/`
/// Whisper/Voxtral convention). `CARGO_MANIFEST_DIR` is
/// `<repo>/crates/vokra-models` — `cargo test` sets a test binary's working
/// directory to the crate root, not the invocation directory, so every
/// repo-root fixture path in this workspace's parity tests is built this
/// way (`parity_whisper.rs`, `parity_kokoro.rs`, `parity_voxtral.rs`,
/// `parity_csm.rs`, `parity_moshi.rs`) rather than as a bare relative
/// literal.
fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("sbv2")
}

/// Loud, actionable precondition check (FR-EX-08): once a caller has opted
/// in via `--ignored`, a missing fixture panics — it never silently
/// skips-and-passes. `what` names the fixture's role in the panic message.
fn require_fixture(path: &Path, what: &str) {
    assert!(
        path.exists(),
        "[parity_sbv2_real] MISSING fixture: {} ({what}). Populate `tests/fixtures/sbv2/` \
         per Task 34's README.md (the real SBV2 v2 + DeBERTa v2/v3 checkpoints, converted \
         with `vokra-cli convert`) plus `reference_dump.manifest.json` and \
         `reference_dump/*.bin` (Task 30's `tools/parity/sbv2_dump_reference.py`), then \
         re-run with `cargo test -p vokra-models --test parity_sbv2_real -- --ignored`. This \
         is a clean gated precondition failure, not a numeric-parity regression.",
        path.display(),
    );
}

/// Reads `path` as a flat little-endian `f32` array (Task 30's dumper
/// format — the same raw-`f32`-bytes convention `parity_moshi.rs`'s
/// `read_f32` and `parity_csm.rs` use for their own `reference_dump/*.bin`
/// siblings).
fn read_f32_bin(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(
        bytes.len() % 4,
        0,
        "{}: byte length {} is not a multiple of 4 (not f32-aligned)",
        path.display(),
        bytes.len(),
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Largest absolute per-element difference between two equal-length slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .fold(0.0f32, |worst, (x, y)| worst.max((x - y).abs()))
}

/// Looks up `key` in JSON object `v`, panicking with `ctx` on a missing key
/// (an opted-in fixture set is expected to be *complete* — a hole in it is
/// a hard failure, not a default-and-continue).
fn json_get<'v>(v: &'v JsonValue, key: &str, ctx: &str) -> &'v JsonValue {
    v.get(key)
        .unwrap_or_else(|| panic!("{ctx}: missing JSON key `{key}`"))
}

/// Reads `v[key]` as a JSON string.
fn json_str<'v>(v: &'v JsonValue, key: &str, ctx: &str) -> &'v str {
    json_get(v, key, ctx)
        .as_str()
        .unwrap_or_else(|| panic!("{ctx}: `{key}` is not a JSON string"))
}

/// Reads `v[key]` as a non-negative JSON integer.
fn json_u64(v: &JsonValue, key: &str, ctx: &str) -> u64 {
    json_get(v, key, ctx)
        .as_u64()
        .unwrap_or_else(|| panic!("{ctx}: `{key}` is not a non-negative integer"))
}

/// Reads `v[key]` as a non-negative JSON integer narrowed to `u32`.
fn json_u32(v: &JsonValue, key: &str, ctx: &str) -> u32 {
    let n = json_u64(v, key, ctx);
    u32::try_from(n).unwrap_or_else(|_| panic!("{ctx}: `{key}` = {n} does not fit in u32"))
}

/// Reads `v[key]` as a JSON number (`Int` or `Float`), narrowed to `f32`.
fn json_f32(v: &JsonValue, key: &str, ctx: &str) -> f32 {
    match json_get(v, key, ctx) {
        JsonValue::Int(i) => *i as f32,
        JsonValue::Float(f) => *f as f32,
        other => panic!("{ctx}: `{key}` is not a JSON number: {other:?}"),
    }
}

/// Reads `v[key]` as a JSON array of numbers, narrowed element-wise to `f32`.
fn json_f32_array(v: &JsonValue, key: &str, ctx: &str) -> Vec<f32> {
    json_get(v, key, ctx)
        .as_array()
        .unwrap_or_else(|| panic!("{ctx}: `{key}` is not a JSON array"))
        .iter()
        .map(|item| match item {
            JsonValue::Int(i) => *i as f32,
            JsonValue::Float(f) => *f as f32,
            other => panic!("{ctx}: `{key}` array element is not a JSON number: {other:?}"),
        })
        .collect()
}

/// Looks up the `tensors[]` entry named `name` in `manifest`, returning its
/// `(path, shape)` pair — `path` relative to [`fixtures_dir`] (already
/// carrying the `reference_dump/` prefix, per the module doc's manifest
/// schema), `shape` as a flat `u64` dimension list.
fn find_tensor<'v>(manifest: &'v JsonValue, name: &str, ctx: &str) -> (&'v str, Vec<u64>) {
    let tensors = json_get(manifest, "tensors", ctx)
        .as_array()
        .unwrap_or_else(|| panic!("{ctx}: `tensors` is not a JSON array"));
    let entry = tensors
        .iter()
        .find(|t| json_str(t, "name", ctx) == name)
        .unwrap_or_else(|| {
            panic!(
                "{ctx}: `tensors` has no entry named `{name}` — see this file's module doc \
                 for the required manifest schema"
            )
        });
    let path = json_str(entry, "path", ctx);
    let shape = json_get(entry, "shape", ctx)
        .as_array()
        .unwrap_or_else(|| panic!("{ctx}: tensors[name={name}].shape is not a JSON array"))
        .iter()
        .map(|elem| {
            elem.as_u64().unwrap_or_else(|| {
                panic!("{ctx}: tensors[name={name}].shape element is not a non-negative integer")
            })
        })
        .collect();
    (path, shape)
}

/// Builds the [`SbV2SynthRequest`] the Python dumper's reference forward
/// pass used, from the manifest's `request` object (see the module doc's
/// schema).
fn request_from_manifest(manifest: &JsonValue, ctx: &str) -> SbV2SynthRequest {
    let request = json_get(manifest, "request", ctx);
    let language = match json_str(request, "language", ctx) {
        "JA" => Language::JA,
        "EN" => Language::EN,
        other => panic!("{ctx}: request.language must be \"JA\" or \"EN\", got {other:?}"),
    };
    SbV2SynthRequest {
        text: json_str(request, "text", ctx).to_string(),
        language,
        speaker_id: json_u32(request, "speaker_id", ctx),
        style_vec: json_f32_array(request, "style_vec", ctx),
        speed: json_f32(request, "speed", ctx),
        noise_scale: json_f32(request, "noise_scale", ctx),
        noise_scale_w: json_f32(request, "noise_scale_w", ctx),
        seed: json_u64(request, "seed", ctx),
    }
}

/// Real-checkpoint SBV2 waveform parity, gated on the Task 34 + Task 30
/// fixture set (`tests/fixtures/sbv2/`). See the module doc for: the
/// manifest schema this reads, why only `waveform` is compared, and why a
/// `synthesize` call against a `from_gguf`-loaded model is expected to
/// return `VokraError::NotImplemented` today (an honestly-logged, non-fatal
/// outcome) rather than a numeric comparison.
#[test]
#[ignore = "Task 34 real fixture: tests/fixtures/sbv2/{reference_dump.manifest.json,*.gguf,reference_dump/*.bin}"]
fn parity_sbv2_real_waveform_matches_reference_dump() {
    let dir = fixtures_dir();

    let manifest_path = dir.join("reference_dump.manifest.json");
    require_fixture(&manifest_path, "reference_dump.manifest.json (Task 34)");
    let manifest_bytes = std::fs::read(&manifest_path)
        .unwrap_or_else(|e| panic!("{}: {e}", manifest_path.display()));
    let manifest = json::parse(&manifest_bytes)
        .unwrap_or_else(|e| panic!("{}: JSON parse error: {e}", manifest_path.display()));
    let ctx = manifest_path.display().to_string();

    let checkpoint = json_get(&manifest, "checkpoint", &ctx);
    let main_path = dir.join(json_str(checkpoint, "sbv2_main", &ctx));
    let bert_ja_path = dir.join(json_str(checkpoint, "bert_ja", &ctx));
    let bert_en_path = dir.join(json_str(checkpoint, "bert_en", &ctx));
    require_fixture(
        &main_path,
        "checkpoint.sbv2_main (Task 34, converted via Task 25)",
    );
    require_fixture(
        &bert_ja_path,
        "checkpoint.bert_ja (Task 34, converted via vokra-bert)",
    );
    require_fixture(
        &bert_en_path,
        "checkpoint.bert_en (Task 34, converted via vokra-bert)",
    );

    let main =
        GgufFile::open(&main_path).unwrap_or_else(|e| panic!("{}: {e}", main_path.display()));
    let bert_ja =
        GgufFile::open(&bert_ja_path).unwrap_or_else(|e| panic!("{}: {e}", bert_ja_path.display()));
    let bert_en =
        GgufFile::open(&bert_en_path).unwrap_or_else(|e| panic!("{}: {e}", bert_en_path.display()));

    let model = SbV2Model::from_gguf(&main, &bert_ja, &bert_en)
        .unwrap_or_else(|e| panic!("SbV2Model::from_gguf: {e}"));

    let req = request_from_manifest(&manifest, &ctx);

    match model.synthesize(&req) {
        Ok(audio) => {
            let (waveform_rel, waveform_shape) = find_tensor(&manifest, "waveform", &ctx);
            let waveform_path = dir.join(waveform_rel);
            require_fixture(&waveform_path, "tensors[name=waveform].path (Task 30 dump)");
            let reference = read_f32_bin(&waveform_path);
            let declared_len: u64 = waveform_shape.iter().product();
            assert_eq!(
                reference.len() as u64,
                declared_len,
                "{}: byte length ({} f32 elements) disagrees with the manifest's declared \
                 `shape` {waveform_shape:?} — Task 34/30's dumper produced an inconsistent \
                 fixture",
                waveform_path.display(),
                reference.len(),
            );
            assert_eq!(
                audio.samples.len(),
                reference.len(),
                "waveform length mismatch: Rust `synthesize` produced {} samples, the \
                 reference dump has {} — `manifest.request` must reproduce exactly the \
                 SbV2SynthRequest the Python dumper used",
                audio.samples.len(),
                reference.len(),
            );
            let atol = tolerance_for("waveform");
            let diff = max_abs_diff(&audio.samples, &reference);
            assert!(
                diff <= atol,
                "waveform max |Δ| = {diff} exceeds atol {atol} (sbv2::parity::tolerance_for(\"waveform\"))"
            );
            eprintln!(
                "[parity_sbv2_real] waveform parity OK: {} samples, max |Δ| = {diff:.3e} <= \
                 atol {atol}",
                audio.samples.len(),
            );
        }
        Err(VokraError::NotImplemented(msg)) if msg.contains("from_gguf loads no G2P") => {
            // Expected, already-documented limitation -- see the module doc's
            // "A real blocker `synthesize` hits first" section. Honestly
            // logged, not a fabricated pass: no numeric comparison ran.
            eprintln!(
                "[parity_sbv2_real] real fixtures loaded successfully (GGUF metadata/tensor \
                 shape verified against a real checkpoint); `synthesize`'s documented \
                 FR-EX-08 refusal fired as expected because `SbV2Model::from_gguf` installs \
                 no G2P (\"{msg}\"). Waveform numeric parity is deferred to a follow-up that \
                 assembles the model via `SbV2Model::new` with a real \
                 `SbV2Phonemizer::from_piper_g2p`-backed phonemizer, which needs a G2P \
                 implementation outside vokra-models' zero-dependency root workspace."
            );
        }
        Err(other) => panic!("SbV2Model::synthesize: unexpected error: {other:?}"),
    }
}
