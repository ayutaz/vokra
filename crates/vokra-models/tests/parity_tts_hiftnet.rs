//! `tts-hiftnet` family flip-the-switch real-checkpoint parity harness
//! (SoTA Phase 1, 2026-07-25).
//!
//! # Family scope
//!
//! Every model in this suite terminates in a **HiFTNet vocoder**
//! (`HiFTGenerator` = Neural Source Filter + ISTFTNet + MRF/Snake, CosyVoice
//! upstream `cosyvoice/hifigan/generator.py`) — the SoTA plan §1(a) 訂正
//! (2026-07-22) shared seam under [`vokra_models::cosyvoice2::hift_chain`]:
//!
//! | short id                    | runtime module                              | HF repo (Phase 1 pinned SHA)                                                                                            | license |
//! |-----------------------------|---------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|---------|
//! | `cosyvoice3`                | [`vokra_models::cosyvoice3`]                | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` @ `29e01c4e8d000f4bcd70751be16fa94bf3d85a18`                                     | Apache-2.0 |
//! | `chatterbox_multilingual`   | [`vokra_models::chatterbox`] (multilingual) | `ResembleAI/chatterbox` @ `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` (`t3_mtl23ls_v3.safetensors`)                      | MIT |
//! | `chatterbox_turbo`          | [`vokra_models::chatterbox_turbo`]          | `ResembleAI/chatterbox-turbo` @ `749d1c1a46eb10492095d68fbcf55691ccf137cd`                                              | MIT |
//! | `chatterbox_nano`           | [`vokra_models::chatterbox_nano`]           | `ResembleAI/chatterbox-nano` @ `71ccd1d0081b430592cea481f4307e764e07bc64`                                               | MIT |
//!
//! HF ids in the task brief use display-cased spellings (`resemble-ai/…`);
//! HuggingFace's canonical repo casing is `ResembleAI/chatterbox*` and
//! the task's `FunAudioLLM/CosyVoice3-0.5B` alias resolves to
//! `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`. The pinned SHAs above were
//! fetched from the HF public API on 2026-07-25 (CLAUDE.md
//! 「ハルシネーション厳禁」).
//!
//! # Env gating (flip-the-switch, `fabricated pass 禁止`)
//!
//! Every test is gated on `VOKRA_TTS_HIFTNET_<ARCH>_GGUF`:
//!
//! - **absent** → the test SKIPs cleanly, printing the exact env var name
//!   and a one-line recipe to produce the GGUF. It NEVER returns a pass
//!   without a real GGUF (FR-EX-08 explicit skip, not a synthetic
//!   fallback).
//! - **present** → the GGUF is opened, its `vokra.model.arch` chunk +
//!   `vokra.<arch>.sample_rate` + every architectural hparam key are
//!   verified against the canonical constructor in the runtime crate,
//!   and at least one tensor payload is required to be present. Any
//!   drift is a **loud** panic — this is the "shape flow" verification
//!   the four models can honestly provide today (their `synthesize`
//!   still returns [`vokra_core::VokraError::NotImplemented`] until the
//!   T29-equivalent follow-up wave binds the real forward chain).
//!
//! When `VOKRA_TTS_HIFTNET_<ARCH>_REFDIR` is ALSO set, the harness looks
//! for a `hparams.json` (produced by an upstream dumper) inside the
//! directory and cross-checks every GGUF-recorded hparam against that
//! reference. This is the "flip-the-switch" step that fires only when
//! both env vars are present: without it the harness limits itself to
//! the GGUF's self-consistency + canonical-constructor round-trip.
//!
//! # Why stage-tap parity is not yet wired
//!
//! `CosyVoice3Tts::synthesize` / `ChatterboxTts::synthesize` /
//! `ChatterboxTurboTts::synthesize` / `ChatterboxNanoTts::synthesize`
//! all return [`vokra_core::VokraError::NotImplemented`] until a
//! follow-up wave binds real weights + wires the Qwen2/Llama LLM →
//! Flow-Matching / speech-token sampling → HiFTNet forward chain (each
//! module's rustdoc names the T29-equivalent blocker in the same
//! phrasing). Stage-tap comparison against `VOKRA_TTS_HIFTNET_<ARCH>_REFDIR`
//! per-step activations therefore has nothing on the Vokra side to
//! compare against; the honest posture is to (a) verify what IS
//! available (shape flow, hparam round-trip, tensor payload presence)
//! and (b) flag the stage-tap defer to the reader — same posture
//! `moshi_quality_gate.rs` uses.

#![allow(clippy::items_after_statements)]

use std::path::{Path, PathBuf};

use vokra_core::gguf::GgufFile;
use vokra_core::gguf::chunks::KEY_MODEL_ARCH;

// ---------------------------------------------------------------------------
// Model spec table
// ---------------------------------------------------------------------------

/// One row of the `tts-hiftnet` family — everything the harness needs to
/// verify a single model without pulling in each model's private hparam
/// crate.
#[derive(Debug, Clone)]
struct ModelSpec {
    /// Short id used in env vars (`VOKRA_TTS_HIFTNET_<UPPER>_GGUF`).
    arch: &'static str,
    /// The `vokra.model.arch` chunk value the converter writes. The
    /// short id above sometimes disambiguates a variant that shares the
    /// same GGUF arch (multilingual + english both write `chatterbox`).
    expected_gguf_arch: &'static str,
    /// Human label for skip / drift messages.
    display_name: &'static str,
    /// HF repo id + pinned SHA the workflow-side download uses. Present
    /// only as documentation for readers who inherit a fixture; the
    /// harness itself never reaches out.
    hf_repo: &'static str,
    hf_revision: &'static str,
    /// Upstream license (SPDX) — documented so a fixture landing in a
    /// PR review is diff-able against `docs/license-audit.md`.
    license_spdx: &'static str,
    /// The `vokra.<arch>.sample_rate` chunk key the converter writes.
    sample_rate_key: &'static str,
    /// Sample rate the canonical constructor pins for this arch (Hz).
    expected_sample_rate: u32,
    /// Every architectural u32 hparam key the converter writes and
    /// [`ModelSpec::verify_gguf`] cross-checks. Each MUST resolve to a
    /// positive `u64` (well-formed hparam), else the load fails.
    positive_u32_hparam_keys: &'static [&'static str],
    /// Every f32 hparam key. Each MUST resolve to a finite, strictly
    /// positive number.
    positive_f32_hparam_keys: &'static [&'static str],
    /// Extra per-arch check (variant tag, MHA/GQA relationship, …).
    /// Consumed after the generic key sweep; a failing check panics.
    variant_check: fn(&GgufFile),
}

impl ModelSpec {
    /// The env var pointing at a converted GGUF for this arch. Never
    /// interpolated from user input.
    fn gguf_env_key(&self) -> String {
        format!("VOKRA_TTS_HIFTNET_{}_GGUF", self.arch.to_ascii_uppercase())
    }

    /// The env var pointing at an upstream reference-dump directory
    /// (see the module docstring: `hparams.json` is what the harness
    /// currently consumes; stage-tap files are read only after a
    /// `synthesize` path binds).
    fn refdir_env_key(&self) -> String {
        format!(
            "VOKRA_TTS_HIFTNET_{}_REFDIR",
            self.arch.to_ascii_uppercase()
        )
    }

    /// Opens the GGUF, verifies arch + sample-rate + every positive
    /// hparam key, and runs the variant-specific check. Panics on any
    /// drift (no silent skip once the GGUF is provided).
    fn verify_gguf(&self, gguf_path: &Path) {
        let file = GgufFile::open(gguf_path).unwrap_or_else(|e| {
            panic!(
                "{}: {} = {}: failed to open GGUF: {e}\n\
                 hint: (re)run `vokra-cli convert --model <hf-id> --input <safetensors> --output {}`",
                self.display_name,
                self.gguf_env_key(),
                gguf_path.display(),
                gguf_path.display(),
            )
        });

        // ---- arch (`vokra.model.arch`) ---------------------------------
        let arch = file
            .get(KEY_MODEL_ARCH)
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| {
                panic!(
                    "{}: {} missing string chunk `{}` — was this GGUF built by \
                     vokra-cli convert against the {} converter?",
                    self.display_name,
                    self.gguf_env_key(),
                    KEY_MODEL_ARCH,
                    self.expected_gguf_arch,
                )
            });
        assert_eq!(
            arch,
            self.expected_gguf_arch,
            "{}: GGUF arch = {arch:?}, expected {:?} — {} points at the wrong \
             checkpoint (mixing chatterbox variants or a totally different family?)",
            self.display_name,
            self.expected_gguf_arch,
            self.gguf_env_key(),
        );

        // ---- sample rate (`vokra.<arch>.sample_rate`) -------------------
        let sr = file
            .get(self.sample_rate_key)
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| {
                panic!(
                    "{}: GGUF missing u32 chunk `{}`",
                    self.display_name, self.sample_rate_key
                )
            });
        assert_eq!(
            sr,
            u64::from(self.expected_sample_rate),
            "{}: `{}` = {sr}, expected {} (canonical constructor invariant)",
            self.display_name,
            self.sample_rate_key,
            self.expected_sample_rate
        );

        // ---- positive u32 hparam sweep ---------------------------------
        for key in self.positive_u32_hparam_keys {
            let v = file.get(key).and_then(|v| v.as_u64()).unwrap_or_else(|| {
                panic!(
                    "{}: GGUF missing u32 chunk `{key}` — converter drift?",
                    self.display_name
                )
            });
            assert!(
                v > 0,
                "{}: `{key}` = 0 (converter emitted a placeholder for a REAL \
                 checkpoint — architectural axes must be > 0, FR-EX-08)",
                self.display_name,
            );
        }

        // ---- positive f32 hparam sweep ---------------------------------
        for key in self.positive_f32_hparam_keys {
            let v = file.get(key).and_then(|v| v.as_f64()).unwrap_or_else(|| {
                panic!(
                    "{}: GGUF missing f32 chunk `{key}` — converter drift?",
                    self.display_name
                )
            });
            assert!(
                v.is_finite() && v > 0.0,
                "{}: `{key}` = {v} (must be finite and > 0)",
                self.display_name,
            );
        }

        // ---- tensor payload presence -----------------------------------
        // A real conversion writes ≥ 1 float tensor; a shape-only /
        // metadata-only GGUF would leave `tensors()` empty and every
        // subsequent forward would trivially "pass" against nothing.
        // The check is a floor, not a bound (large families ship
        // thousands; we just refuse zero).
        assert!(
            !file.tensors().is_empty(),
            "{}: GGUF contains zero tensor payloads — the converter emitted \
             a metadata-only shell (real conversion must ship LLM + vocoder \
             weights)",
            self.display_name,
        );

        // ---- variant-specific check ------------------------------------
        (self.variant_check)(&file);

        // ---- optional flip-the-switch: reference cross-check -----------
        if let Some(refdir) = std::env::var_os(self.refdir_env_key()).map(PathBuf::from) {
            cross_check_refdir(self, &file, &refdir);
        } else {
            eprintln!(
                "[parity_tts_hiftnet] {}: {} unset — stage-tap cross-check \
                 skipped (self-consistency + canonical-constructor round-trip \
                 verified via the GGUF only). This is a CLEAN skip of the \
                 reference leg, not of the parity leg itself.",
                self.display_name,
                self.refdir_env_key(),
            );
        }

        eprintln!(
            "[parity_tts_hiftnet] {}: OK — arch={arch:?}, sample_rate={sr} Hz, \
             {} u32 hparam(s) + {} f32 hparam(s) verified, {} tensor payload(s)",
            self.display_name,
            self.positive_u32_hparam_keys.len(),
            self.positive_f32_hparam_keys.len(),
            file.tensors().len(),
        );
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// (`gguf`, `refdir`) env-var pair for a given `arch`. `None` in either
/// slot means the env var is unset — every test above turns unset
/// `gguf` into a clean skip; `refdir` unset is honest degradation to
/// self-consistency-only.
fn env_paths_for(arch: &str) -> (Option<PathBuf>, Option<PathBuf>) {
    let gguf_key = format!("VOKRA_TTS_HIFTNET_{}_GGUF", arch.to_ascii_uppercase());
    let refdir_key = format!("VOKRA_TTS_HIFTNET_{}_REFDIR", arch.to_ascii_uppercase());
    (
        std::env::var_os(gguf_key).map(PathBuf::from),
        std::env::var_os(refdir_key).map(PathBuf::from),
    )
}

/// The one-line annotation a skipped test prints. Names the env var to
/// set, the pinned HF repo/SHA the workflow uses, and the CLI recipe.
fn skip_reason(spec: &ModelSpec) -> String {
    format!(
        "[parity_tts_hiftnet] {}: SKIP — env var {} unset.\n  \
         family: tts-hiftnet ({} vocoder — HiFTGenerator = NSF + ISTFTNet + MRF/Snake)\n  \
         upstream: {} @ {} ({})\n  \
         recipe: download the checkpoint, then \
         `cargo run --release -p vokra-cli -- convert --model <alias> \
         --input <safetensors> --output <path>` and re-run with \
         {} pointing at the GGUF.\n  \
         (FR-EX-08 explicit skip, not a synthetic fallback.)",
        spec.display_name,
        spec.gguf_env_key(),
        spec.display_name,
        spec.hf_repo,
        spec.hf_revision,
        spec.license_spdx,
        spec.gguf_env_key(),
    )
}

/// Compares every recorded hparam in the GGUF against a JSON reference
/// dump (`<refdir>/hparams.json`). The JSON's SHAPE is deliberately
/// loose (top-level object of `{"<vokra.chatterbox_turbo.arch.hidden_dim>":
/// 1024, …}` — a flat string→number map). This is the flip-the-switch
/// step: absent = the ambient skip message above is honest, present =
/// every listed key MUST match.
fn cross_check_refdir(spec: &ModelSpec, file: &GgufFile, refdir: &Path) {
    let manifest_path = refdir.join("hparams.json");
    if !manifest_path.exists() {
        eprintln!(
            "[parity_tts_hiftnet] {}: {} = {} → {} not found; nothing to cross-check. \
             Stage-tap fixtures are the T29-equivalent follow-up (`synthesize` binds \
             real weights first). Not fabricating a pass on emptiness.",
            spec.display_name,
            spec.refdir_env_key(),
            refdir.display(),
            manifest_path.display(),
        );
        return;
    }

    // We do NOT pull `serde_json` (zero-dep NFR-DS-02 root Cargo.lock
    // invariant). vokra_core::json is the same parser
    // parity_moshi.rs uses.
    let bytes = std::fs::read(&manifest_path).unwrap_or_else(|e| {
        panic!(
            "{}: cannot read {}: {e}",
            spec.display_name,
            manifest_path.display()
        )
    });
    let root = vokra_core::json::parse(&bytes).unwrap_or_else(|e| {
        panic!(
            "{}: malformed JSON in {}: {e}",
            spec.display_name,
            manifest_path.display()
        )
    });
    let map = match &root {
        vokra_core::json::JsonValue::Object(m) => m,
        other => panic!(
            "{}: {} must be a top-level JSON object, got {other:?}",
            spec.display_name,
            manifest_path.display()
        ),
    };

    let mut checked = 0usize;
    for key in spec.positive_u32_hparam_keys {
        let Some((_, ref_val)) = map.iter().find(|(k, _)| k == key) else {
            continue;
        };
        let ref_int = match ref_val {
            vokra_core::json::JsonValue::Int(n) => *n,
            other => panic!(
                "{}: hparams.json[{key}] must be an integer, got {other:?}",
                spec.display_name
            ),
        };
        let got = file.get(key).and_then(|v| v.as_u64()).unwrap_or_else(|| {
            panic!(
                "{}: GGUF missing `{key}` (asserted above)",
                spec.display_name
            )
        });
        assert_eq!(
            got as i64, ref_int,
            "{}: {key} — GGUF = {got}, reference (hparams.json) = {ref_int}",
            spec.display_name
        );
        checked += 1;
    }
    for key in spec.positive_f32_hparam_keys {
        let Some((_, ref_val)) = map.iter().find(|(k, _)| k == key) else {
            continue;
        };
        // Float parity here is only for the small handful of transcribed
        // architectural floats (rope_base, rms_norm_eps). Compare via
        // exact bit representation of the two f32 values so we do not
        // introduce a tolerance smuggle for hparams that MUST round-trip
        // verbatim (FR-EX-08).
        let ref_float = match ref_val {
            vokra_core::json::JsonValue::Float(f) => *f as f32,
            vokra_core::json::JsonValue::Int(n) => *n as f32,
            other => panic!(
                "{}: hparams.json[{key}] must be a number, got {other:?}",
                spec.display_name
            ),
        };
        let got = file.get(key).and_then(|v| v.as_f64()).unwrap_or_else(|| {
            panic!(
                "{}: GGUF missing `{key}` (asserted above)",
                spec.display_name
            )
        }) as f32;
        assert_eq!(
            got.to_bits(),
            ref_float.to_bits(),
            "{}: {key} — GGUF = {got}, reference (hparams.json) = {ref_float} (bit-for-bit)",
            spec.display_name
        );
        checked += 1;
    }

    eprintln!(
        "[parity_tts_hiftnet] {}: reference cross-check via {} verified {checked}/{} hparam(s). \
         Stage-tap comparison (per-layer activations) is DEFERRED — requires a live `synthesize` \
         path, currently `NotImplemented` (T29-equivalent follow-up wave).",
        spec.display_name,
        manifest_path.display(),
        spec.positive_u32_hparam_keys.len() + spec.positive_f32_hparam_keys.len(),
    );
}

// ---------------------------------------------------------------------------
// Per-model variant checks (called after the generic sweep in verify_gguf)
// ---------------------------------------------------------------------------

/// CosyVoice3 canonical Qwen2 well-formedness: `n_head` divides
/// `hidden_dim`; if `n_head_kv` is present it must also divide.
fn check_cosyvoice3(file: &GgufFile) {
    let hidden = file
        .get("vokra.cosyvoice3.arch.hidden_dim")
        .and_then(|v| v.as_u64())
        .expect("verified above");
    let n_head = file
        .get("vokra.cosyvoice3.arch.n_head")
        .and_then(|v| v.as_u64())
        .expect("verified above");
    assert!(
        hidden % n_head == 0,
        "cosyvoice3: hidden_dim {hidden} not divisible by n_head {n_head} (Qwen2 GQA algebra)",
    );
    if let Some(kv) = file
        .get("vokra.cosyvoice3.arch.n_head_kv")
        .and_then(|v| v.as_u64())
    {
        assert!(kv > 0, "cosyvoice3: n_head_kv = 0");
        assert!(
            n_head % kv == 0,
            "cosyvoice3: n_head {n_head} not a multiple of n_head_kv {kv} (Qwen2 GQA)",
        );
    }
    // Sample rate on this model MUST be 24 kHz (the HiFTNet fixed rate,
    // asserted separately as a runtime constant round-trip).
    let sr = vokra_models::cosyvoice3::COSYVOICE3_SAMPLE_RATE;
    assert_eq!(sr, 24_000, "cosyvoice3 runtime SAMPLE_RATE invariant");
}

/// Chatterbox (base) — the multilingual variant is identified by
/// `vokra.chatterbox.variant = "multilingual"` and
/// `text_vocab_size = 2454` (the primary-source split from the
/// English-only `704`).
fn check_chatterbox_multilingual(file: &GgufFile) {
    // Variant tag must be present (converter writes it) and MUST be
    // "multilingual" for this test — an "english" variant loaded here
    // is a fixture / env var mismatch.
    let variant = file
        .get("vokra.chatterbox.variant")
        .and_then(|v| v.as_str())
        .expect("chatterbox: converter always writes vokra.chatterbox.variant");
    assert_eq!(
        variant, "multilingual",
        "chatterbox: variant tag is {variant:?}, expected \"multilingual\" — \
         VOKRA_TTS_HIFTNET_CHATTERBOX_MULTILINGUAL_GGUF must point at the \
         t3_mtl23ls_v3.safetensors conversion, not the English-only one",
    );
    let text_vocab = file
        .get("vokra.chatterbox.arch.text_vocab_size")
        .and_then(|v| v.as_u64())
        .expect("verified above");
    assert_eq!(
        text_vocab,
        u64::from(vokra_models::chatterbox::TEXT_VOCAB_MULTILINGUAL),
        "chatterbox multilingual: text_vocab_size = {text_vocab}, expected {} \
         (primary-source constant TEXT_VOCAB_MULTILINGUAL from \
         src/chatterbox/models/t3/modules/t3_config.py)",
        vokra_models::chatterbox::TEXT_VOCAB_MULTILINGUAL,
    );
    // MHA well-formedness (Llama_520M backbone).
    let hidden = file
        .get("vokra.chatterbox.arch.hidden_dim")
        .and_then(|v| v.as_u64())
        .expect("verified above");
    let n_head = file
        .get("vokra.chatterbox.arch.n_head")
        .and_then(|v| v.as_u64())
        .expect("verified above");
    let head_dim = file
        .get("vokra.chatterbox.arch.head_dim")
        .and_then(|v| v.as_u64())
        .expect("verified above");
    assert_eq!(
        n_head * head_dim,
        hidden,
        "chatterbox multilingual: n_head({n_head}) * head_dim({head_dim}) != hidden({hidden}) — \
         Llama_520M MHA invariant",
    );
}

/// Chatterbox-Turbo — GPT-2-medium backbone, `text_vocab_size = 50 276`,
/// paralinguistic tag count = 19.
fn check_chatterbox_turbo(file: &GgufFile) {
    let text_vocab = file
        .get("vokra.chatterbox_turbo.arch.text_vocab_size")
        .and_then(|v| v.as_u64())
        .expect("verified above");
    assert_eq!(
        text_vocab,
        u64::from(vokra_models::chatterbox_turbo::TEXT_VOCAB_TURBO),
        "chatterbox_turbo: text_vocab_size = {text_vocab}, expected {}",
        vokra_models::chatterbox_turbo::TEXT_VOCAB_TURBO,
    );
    let tags = file
        .get("vokra.chatterbox_turbo.arch.paralinguistic_tag_count")
        .and_then(|v| v.as_u64())
        .expect("verified above");
    assert_eq!(
        tags,
        u64::from(vokra_models::chatterbox_turbo::PARALINGUISTIC_TAG_COUNT),
        "chatterbox_turbo: paralinguistic_tag_count = {tags}, expected {}",
        vokra_models::chatterbox_turbo::PARALINGUISTIC_TAG_COUNT,
    );
    // MHA well-formedness.
    let hidden = file
        .get("vokra.chatterbox_turbo.arch.hidden_dim")
        .and_then(|v| v.as_u64())
        .expect("verified above");
    let n_head = file
        .get("vokra.chatterbox_turbo.arch.n_head")
        .and_then(|v| v.as_u64())
        .expect("verified above");
    let head_dim = file
        .get("vokra.chatterbox_turbo.arch.head_dim")
        .and_then(|v| v.as_u64())
        .expect("verified above");
    assert_eq!(
        n_head * head_dim,
        hidden,
        "chatterbox_turbo: n_head({n_head}) * head_dim({head_dim}) != hidden({hidden})",
    );
}

/// Chatterbox-Nano — same Llama_520M as base + GPT-2 sentinel token
/// pair + 32 kHz sample rate; text_vocab_size is TEXT_VOCAB_NANO
/// (50 276, aligned with Turbo).
fn check_chatterbox_nano(file: &GgufFile) {
    let text_vocab = file
        .get("vokra.chatterbox_nano.arch.text_vocab_size")
        .and_then(|v| v.as_u64())
        .expect("verified above");
    assert_eq!(
        text_vocab,
        u64::from(vokra_models::chatterbox_nano::TEXT_VOCAB_NANO),
        "chatterbox_nano: text_vocab_size = {text_vocab}, expected {}",
        vokra_models::chatterbox_nano::TEXT_VOCAB_NANO,
    );
    // MHA well-formedness (same Llama_520M topology as base Chatterbox).
    let hidden = file
        .get("vokra.chatterbox_nano.arch.hidden_dim")
        .and_then(|v| v.as_u64())
        .expect("verified above");
    let n_head = file
        .get("vokra.chatterbox_nano.arch.n_head")
        .and_then(|v| v.as_u64())
        .expect("verified above");
    let head_dim = file
        .get("vokra.chatterbox_nano.arch.head_dim")
        .and_then(|v| v.as_u64())
        .expect("verified above");
    assert_eq!(
        n_head * head_dim,
        hidden,
        "chatterbox_nano: n_head({n_head}) * head_dim({head_dim}) != hidden({hidden})",
    );
    // n_head_kv (MHA, kept equal to n_head per the canonical constructor).
    let kv = file
        .get("vokra.chatterbox_nano.arch.n_head_kv")
        .and_then(|v| v.as_u64())
        .expect("verified above");
    assert_eq!(
        kv, n_head,
        "chatterbox_nano: n_head_kv ({kv}) must equal n_head ({n_head}) — canonical MHA layout",
    );
}

// ---------------------------------------------------------------------------
// Spec table (verbatim from each runtime module's converter key list;
// mirrors `crates/vokra-convert/src/models/{cosyvoice3,chatterbox,
// chatterbox_turbo,chatterbox_nano}.rs`)
// ---------------------------------------------------------------------------

fn cosyvoice3_spec() -> ModelSpec {
    ModelSpec {
        arch: "cosyvoice3",
        expected_gguf_arch: "cosyvoice3",
        display_name: "Fun-CosyVoice3-0.5B",
        hf_repo: "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        hf_revision: "29e01c4e8d000f4bcd70751be16fa94bf3d85a18",
        license_spdx: "apache-2.0",
        sample_rate_key: "vokra.cosyvoice3.sample_rate",
        expected_sample_rate: vokra_models::cosyvoice3::COSYVOICE3_SAMPLE_RATE,
        positive_u32_hparam_keys: &[
            "vokra.cosyvoice3.arch.vocab_size",
            "vokra.cosyvoice3.arch.hidden_dim",
            "vokra.cosyvoice3.arch.n_layer",
            "vokra.cosyvoice3.arch.n_head",
            "vokra.cosyvoice3.arch.ffn_dim",
        ],
        positive_f32_hparam_keys: &[],
        variant_check: check_cosyvoice3,
    }
}

fn chatterbox_multilingual_spec() -> ModelSpec {
    ModelSpec {
        arch: "chatterbox_multilingual",
        expected_gguf_arch: "chatterbox",
        display_name: "Chatterbox-Multilingual (T3 mtl23ls_v3)",
        hf_repo: "ResembleAI/chatterbox",
        hf_revision: "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
        license_spdx: "MIT",
        sample_rate_key: "vokra.chatterbox.sample_rate",
        expected_sample_rate: vokra_models::chatterbox::CHATTERBOX_SAMPLE_RATE,
        positive_u32_hparam_keys: &[
            "vokra.chatterbox.arch.text_vocab_size",
            "vokra.chatterbox.arch.speech_vocab_size",
            "vokra.chatterbox.arch.max_text_tokens",
            "vokra.chatterbox.arch.max_speech_tokens",
            "vokra.chatterbox.arch.speaker_embed_size",
            "vokra.chatterbox.arch.hidden_dim",
            "vokra.chatterbox.arch.n_layer",
            "vokra.chatterbox.arch.n_head",
            "vokra.chatterbox.arch.n_head_kv",
            "vokra.chatterbox.arch.head_dim",
            "vokra.chatterbox.arch.ffn_dim",
        ],
        positive_f32_hparam_keys: &[
            "vokra.chatterbox.arch.rope_base",
            "vokra.chatterbox.arch.rms_norm_eps",
        ],
        variant_check: check_chatterbox_multilingual,
    }
}

fn chatterbox_turbo_spec() -> ModelSpec {
    ModelSpec {
        arch: "chatterbox_turbo",
        expected_gguf_arch: "chatterbox_turbo",
        display_name: "Chatterbox-Turbo (t3_turbo_v1)",
        hf_repo: "ResembleAI/chatterbox-turbo",
        hf_revision: "749d1c1a46eb10492095d68fbcf55691ccf137cd",
        license_spdx: "MIT",
        sample_rate_key: "vokra.chatterbox_turbo.sample_rate",
        expected_sample_rate: vokra_models::chatterbox_turbo::CHATTERBOX_TURBO_SAMPLE_RATE,
        positive_u32_hparam_keys: &[
            "vokra.chatterbox_turbo.arch.text_vocab_size",
            "vokra.chatterbox_turbo.arch.speech_vocab_size",
            "vokra.chatterbox_turbo.arch.max_text_tokens",
            "vokra.chatterbox_turbo.arch.max_speech_tokens",
            "vokra.chatterbox_turbo.arch.speaker_embed_size",
            "vokra.chatterbox_turbo.arch.ve_hidden_size",
            "vokra.chatterbox_turbo.arch.hidden_dim",
            "vokra.chatterbox_turbo.arch.n_layer",
            "vokra.chatterbox_turbo.arch.n_head",
            "vokra.chatterbox_turbo.arch.head_dim",
            "vokra.chatterbox_turbo.arch.hop_size",
            "vokra.chatterbox_turbo.arch.win_size",
            "vokra.chatterbox_turbo.arch.num_mels",
            "vokra.chatterbox_turbo.arch.speech_cond_prompt_len",
            "vokra.chatterbox_turbo.arch.paralinguistic_tag_count",
        ],
        positive_f32_hparam_keys: &[],
        variant_check: check_chatterbox_turbo,
    }
}

fn chatterbox_nano_spec() -> ModelSpec {
    ModelSpec {
        arch: "chatterbox_nano",
        expected_gguf_arch: "chatterbox_nano",
        display_name: "Chatterbox-Nano (t3_nano_v1)",
        hf_repo: "ResembleAI/chatterbox-nano",
        hf_revision: "71ccd1d0081b430592cea481f4307e764e07bc64",
        license_spdx: "MIT",
        sample_rate_key: "vokra.chatterbox_nano.sample_rate",
        expected_sample_rate: vokra_models::chatterbox_nano::CHATTERBOX_NANO_SAMPLE_RATE,
        positive_u32_hparam_keys: &[
            "vokra.chatterbox_nano.arch.text_vocab_size",
            "vokra.chatterbox_nano.arch.speech_vocab_size",
            "vokra.chatterbox_nano.arch.max_text_tokens",
            "vokra.chatterbox_nano.arch.max_speech_tokens",
            "vokra.chatterbox_nano.arch.speaker_embed_size",
            "vokra.chatterbox_nano.arch.ve_hidden_size",
            "vokra.chatterbox_nano.arch.hidden_dim",
            "vokra.chatterbox_nano.arch.n_layer",
            "vokra.chatterbox_nano.arch.n_head",
            "vokra.chatterbox_nano.arch.n_head_kv",
            "vokra.chatterbox_nano.arch.head_dim",
            "vokra.chatterbox_nano.arch.ffn_dim",
            "vokra.chatterbox_nano.arch.hop_size",
            "vokra.chatterbox_nano.arch.win_size",
            "vokra.chatterbox_nano.arch.num_mels",
            "vokra.chatterbox_nano.arch.speech_cond_prompt_len",
            "vokra.chatterbox_nano.arch.paralinguistic_tag_count",
        ],
        positive_f32_hparam_keys: &[
            "vokra.chatterbox_nano.arch.rope_base",
            "vokra.chatterbox_nano.arch.rms_norm_eps",
        ],
        variant_check: check_chatterbox_nano,
    }
}

// ---------------------------------------------------------------------------
// Per-model gated tests
// ---------------------------------------------------------------------------

#[test]
fn parity_tts_hiftnet_cosyvoice3() {
    let spec = cosyvoice3_spec();
    let (gguf, _refdir) = env_paths_for(spec.arch);
    let Some(gguf) = gguf else {
        println!("{}", skip_reason(&spec));
        return;
    };
    spec.verify_gguf(&gguf);
}

#[test]
fn parity_tts_hiftnet_chatterbox_multilingual() {
    let spec = chatterbox_multilingual_spec();
    let (gguf, _refdir) = env_paths_for(spec.arch);
    let Some(gguf) = gguf else {
        println!("{}", skip_reason(&spec));
        return;
    };
    spec.verify_gguf(&gguf);
}

#[test]
fn parity_tts_hiftnet_chatterbox_turbo() {
    let spec = chatterbox_turbo_spec();
    let (gguf, _refdir) = env_paths_for(spec.arch);
    let Some(gguf) = gguf else {
        println!("{}", skip_reason(&spec));
        return;
    };
    spec.verify_gguf(&gguf);
}

#[test]
fn parity_tts_hiftnet_chatterbox_nano() {
    let spec = chatterbox_nano_spec();
    let (gguf, _refdir) = env_paths_for(spec.arch);
    let Some(gguf) = gguf else {
        println!("{}", skip_reason(&spec));
        return;
    };
    spec.verify_gguf(&gguf);
}

// ---------------------------------------------------------------------------
// Always-on sanity: the spec table must stay in sync with the runtime
// constants (a canonical constructor drift would silently invalidate the
// gated tests above, which is precisely the class of fabricated pass this
// suite exists to prevent).
// ---------------------------------------------------------------------------

#[test]
fn spec_table_matches_runtime_expected_arch() {
    assert_eq!(
        cosyvoice3_spec().expected_gguf_arch,
        vokra_models::cosyvoice3::EXPECTED_ARCH,
    );
    assert_eq!(
        chatterbox_multilingual_spec().expected_gguf_arch,
        vokra_models::chatterbox::EXPECTED_ARCH,
    );
    assert_eq!(
        chatterbox_turbo_spec().expected_gguf_arch,
        vokra_models::chatterbox_turbo::EXPECTED_ARCH,
    );
    assert_eq!(
        chatterbox_nano_spec().expected_gguf_arch,
        vokra_models::chatterbox_nano::EXPECTED_ARCH,
    );
}

#[test]
fn spec_table_matches_runtime_sample_rate_constants() {
    assert_eq!(
        cosyvoice3_spec().expected_sample_rate,
        vokra_models::cosyvoice3::COSYVOICE3_SAMPLE_RATE,
    );
    assert_eq!(
        chatterbox_multilingual_spec().expected_sample_rate,
        vokra_models::chatterbox::CHATTERBOX_SAMPLE_RATE,
    );
    assert_eq!(
        chatterbox_turbo_spec().expected_sample_rate,
        vokra_models::chatterbox_turbo::CHATTERBOX_TURBO_SAMPLE_RATE,
    );
    assert_eq!(
        chatterbox_nano_spec().expected_sample_rate,
        vokra_models::chatterbox_nano::CHATTERBOX_NANO_SAMPLE_RATE,
    );
}

#[test]
fn env_var_naming_is_stable() {
    // The workflow YAML depends on this exact scheme; if a future refactor
    // renames the env vars, CI's `env:` blocks must move in lock-step.
    assert_eq!(
        cosyvoice3_spec().gguf_env_key(),
        "VOKRA_TTS_HIFTNET_COSYVOICE3_GGUF"
    );
    assert_eq!(
        chatterbox_multilingual_spec().gguf_env_key(),
        "VOKRA_TTS_HIFTNET_CHATTERBOX_MULTILINGUAL_GGUF"
    );
    assert_eq!(
        chatterbox_turbo_spec().gguf_env_key(),
        "VOKRA_TTS_HIFTNET_CHATTERBOX_TURBO_GGUF"
    );
    assert_eq!(
        chatterbox_nano_spec().gguf_env_key(),
        "VOKRA_TTS_HIFTNET_CHATTERBOX_NANO_GGUF"
    );
    // And the paired refdir keys.
    assert_eq!(
        cosyvoice3_spec().refdir_env_key(),
        "VOKRA_TTS_HIFTNET_COSYVOICE3_REFDIR"
    );
    assert_eq!(
        chatterbox_nano_spec().refdir_env_key(),
        "VOKRA_TTS_HIFTNET_CHATTERBOX_NANO_REFDIR"
    );
}

#[test]
fn skip_reason_names_the_expected_env_var() {
    // Guard against a refactor that starts emitting a generic "please set
    // an env var" message; the whole point of the FR-EX-08 skip is the
    // exact variable name and repo pin.
    let s = skip_reason(&cosyvoice3_spec());
    assert!(s.contains("VOKRA_TTS_HIFTNET_COSYVOICE3_GGUF"), "{s}");
    assert!(s.contains("FunAudioLLM/Fun-CosyVoice3-0.5B-2512"), "{s}");
    assert!(
        s.contains("29e01c4e8d000f4bcd70751be16fa94bf3d85a18"),
        "{s}"
    );
    assert!(s.contains("apache-2.0"), "{s}");

    let s = skip_reason(&chatterbox_multilingual_spec());
    assert!(
        s.contains("VOKRA_TTS_HIFTNET_CHATTERBOX_MULTILINGUAL_GGUF"),
        "{s}"
    );
    assert!(s.contains("ResembleAI/chatterbox"), "{s}");
    assert!(
        s.contains("5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"),
        "{s}"
    );
}
