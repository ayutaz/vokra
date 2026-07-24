//! **Qwen3-TTS-0.6B** family — real-checkpoint parity CI harness (flip-the-
//! switch). Sibling of `parity_kokoro.rs` / `parity_cosyvoice2.rs` /
//! `parity_moshi.rs` on env-gated shape / metadata / (when the reference
//! dir is present) upstream `config.json` cross-check.
//!
//! # Family and env-var contract
//!
//! Every model in the `tts-qwen3` family gets ONE `#[test]` here. Each test:
//!
//! * reads [`env_paths_for(arch)`](env_paths_for) — the tuple
//!   `(VOKRA_<ARCH>_GGUF, VOKRA_<ARCH>_REFDIR)` — where `<ARCH>` is the
//!   converter's `vokra.model.arch` string, upper-cased with `-` → `_`
//!   (`qwen3_tts` → `VOKRA_QWEN3_TTS_GGUF`);
//! * SKIPS cleanly with a printed reason via [`skip_reason`] when the GGUF
//!   env var is unset (never a fabricated pass — FR-EX-08). The Rust test
//!   framework treats a non-panicking `#[test]` as a pass, so the printed
//!   reason (with `[parity_tts_qwen3] SKIP:` prefix) is the honest audit
//!   trail; the CI workflow's summary parser prints it and the artifact
//!   uploads capture it;
//! * when the GGUF env var IS set: loads the GGUF, verifies
//!     * `vokra.model.arch == "qwen3_tts"` (see
//!       [`vokra_models::qwen3_tts::EXPECTED_ARCH`] — the sole cross-crate
//!       handshake),
//!     * every `vokra.qwen3_tts.*` hparam matches the primary-source
//!       [`Qwen3TtsConfig::qwen3_tts_0_6b_base`] (transcribed **verbatim**
//!       from `huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/config.json`,
//!       fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」),
//!     * at least one talker weight tensor is present so the GGUF is not
//!       metadata-only (the converter surfaces a loud `notes` entry when
//!       `report.written == 0`, and this leg refuses to report a pass on
//!       such a metadata-only artifact);
//! * when BOTH the GGUF **and** the reference-dir env vars are set: reads
//!   the upstream `config.json` at
//!   `<VOKRA_QWEN3_TTS_REFDIR>/config.json`, walks the `talker.*` and
//!   `code_predictor.*` blocks, and cross-checks every architectural axis
//!   against the corresponding `vokra.qwen3_tts.*` metadata chunk in the
//!   GGUF. This is the **flip-the-switch** leg: it fires as soon as the
//!   owner drops the upstream `config.json` next to the GGUF, and (unlike
//!   a synthesized-fixture check) it binds the primary source against the
//!   converter's transcription — a hidden drift between the two would
//!   fail loudly.
//!
//! # Why no `synthesize` / `transcribe` byte-level parity yet
//!
//! `Qwen3TtsTts::synthesize` returns `VokraError::NotImplemented` today
//! (see the module docstring on
//! `vokra_models::qwen3_tts::Qwen3TtsTts::synthesize`): the full
//! talker → code-predictor → `qwen3_tts_codec` → neural-decoder → PCM
//! forward is a follow-up wave. This harness therefore validates what IS
//! available — GGUF binding, arch tag, primary-source-transcribed hparam
//! chunks, talker weight-tensor shape sanity — plus the flip-the-switch
//! reference-dir cross-check that closes the transcription loop against
//! the upstream `config.json`. Byte-level output parity gets added here
//! by name (`parity_qwen3_tts_synthesize_matches_upstream_pcm`) in the
//! follow-up wave that lands the forward.
//!
//! # Zero-dep
//!
//! The whole harness lives in the standard library plus `vokra-core`
//! (`gguf::GgufFile` for the reader, `json::parse` for the reference
//! `config.json` walk) and `vokra-models::qwen3_tts` (the runtime config
//! that carries the primary-source constants). No new third-party crate
//! is pulled in — NFR-DS-02 root-Cargo.lock invariant preserved.

#![allow(clippy::items_after_statements)]

use std::path::{Path, PathBuf};

use vokra_core::gguf::chunks::KEY_MODEL_ARCH;
use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::json::{self, JsonValue};
use vokra_models::qwen3_tts::{
    EXPECTED_ARCH, QWEN3_TTS_NUM_CODE_GROUPS, QWEN3_TTS_SAMPLE_RATE, QWEN3_TTS_SPEAKER_EMBED_DIM,
    Qwen3TtsConfig,
};

// ---------------------------------------------------------------------------
// Env-var contract helpers (family-shaped so a future member drops in with
// only a new `#[test]`)
// ---------------------------------------------------------------------------

/// Returns `(gguf, refdir)` env-var payloads for `arch`.
///
/// * `gguf`   — value of `VOKRA_<ARCH>_GGUF`. Set = run the gated leg.
/// * `refdir` — value of `VOKRA_<ARCH>_REFDIR`. Set alongside the GGUF =
///   run the flip-the-switch cross-check against
///   `<refdir>/config.json`.
///
/// `<ARCH>` is the arch string upper-cased with `-` → `_`. The Qwen3-TTS
/// arch tag is `qwen3_tts`, so the env vars are
/// `VOKRA_QWEN3_TTS_GGUF` / `VOKRA_QWEN3_TTS_REFDIR`.
fn env_paths_for(arch: &str) -> (Option<PathBuf>, Option<PathBuf>) {
    let arch_env = arch.replace('-', "_").to_ascii_uppercase();
    let gguf = std::env::var_os(format!("VOKRA_{arch_env}_GGUF")).map(PathBuf::from);
    let refdir = std::env::var_os(format!("VOKRA_{arch_env}_REFDIR")).map(PathBuf::from);
    (gguf, refdir)
}

/// Renders the clean-skip message the test prints when `VOKRA_<ARCH>_GGUF`
/// is unset.
///
/// Kept as a distinct helper so a follow-up wave (a) can grep every
/// harness's skip text for uniformity, and (b) tell the summary parser
/// (workflow YAML) exactly what to look for.
///
/// Names BOTH env vars (`_GGUF` and `_REFDIR`) so an operator reading a
/// baseline stderr (both unset) learns about the flip-the-switch leg
/// without having to grep the source (sibling `parity_tts_japanese.rs`
/// enforces the same "self-documenting skip" invariant — see
/// `skip_reason_contains_reproduction_recipe_and_both_env_vars`).
fn skip_reason(arch: &str, hf_repo: &str) -> String {
    let arch_env = arch.replace('-', "_").to_ascii_uppercase();
    format!(
        "[parity_tts_qwen3] SKIP: VOKRA_{arch_env}_GGUF unset. Convert the \
         upstream `{hf_repo}` checkpoint with `vokra-cli convert --model \
         qwen3-tts --input <model.safetensors> --output <out.gguf>` and \
         re-run with `VOKRA_{arch_env}_GGUF=<out.gguf>`. To also enable the \
         flip-the-switch cross-check against the upstream `config.json`, \
         drop that file into a directory and additionally set \
         `VOKRA_{arch_env}_REFDIR=<dir>`. This is a clean gated skip, never \
         a fabricated pass (FR-EX-08)."
    )
}

// ---------------------------------------------------------------------------
// Loud metadata accessors — every miss is a hard-fail once the GGUF env var
// is set (opted-in ⇒ incomplete setup is a failure, never a silent pass)
// ---------------------------------------------------------------------------

fn get_u32(file: &GgufFile, key: &str, gguf_path: &Path) -> u32 {
    match file.get(key) {
        Some(GgufMetadataValue::U32(v)) => *v,
        Some(other) => panic!(
            "{}: `{key}` present but not U32 (got {:?}) — the converter contract is `add_u32`",
            gguf_path.display(),
            other.value_type(),
        ),
        None => panic!(
            "{}: missing `{key}` — the converter is expected to write this chunk. Re-convert \
             with a current `vokra-cli convert --model qwen3-tts` release",
            gguf_path.display(),
        ),
    }
}

fn get_f32(file: &GgufFile, key: &str, gguf_path: &Path) -> f32 {
    match file.get(key) {
        Some(GgufMetadataValue::F32(v)) => *v,
        Some(other) => panic!(
            "{}: `{key}` present but not F32 (got {:?})",
            gguf_path.display(),
            other.value_type(),
        ),
        None => panic!("{}: missing `{key}`", gguf_path.display()),
    }
}

fn get_str<'f>(file: &'f GgufFile, key: &str, gguf_path: &Path) -> &'f str {
    match file.get(key) {
        Some(GgufMetadataValue::String(s)) => s,
        Some(other) => panic!(
            "{}: `{key}` present but not String (got {:?})",
            gguf_path.display(),
            other.value_type(),
        ),
        None => panic!("{}: missing `{key}`", gguf_path.display()),
    }
}

/// True if any tensor name matches `pred`.
fn any_tensor(file: &GgufFile, pred: impl Fn(&str) -> bool) -> bool {
    file.tensors().iter().any(|t| pred(&t.name))
}

// ---------------------------------------------------------------------------
// GGUF chunk assertions — every field is cross-checked against the primary
// source `Qwen3TtsConfig::qwen3_tts_0_6b_base()` (transcribed verbatim from
// `huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/config.json`, fetched
// 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」)
// ---------------------------------------------------------------------------

/// Reads every `vokra.qwen3_tts.*` chunk from `file` and asserts each
/// matches the primary-source [`Qwen3TtsConfig::qwen3_tts_0_6b_base`].
///
/// A hidden drift between the converter's transcribed constants and the
/// runtime's transcribed constants would surface as a mismatch here (the
/// two crates share only `vokra-core`, so this comparison is the only
/// cross-crate handshake for the hparam chunk group).
fn assert_metadata_matches_primary_source(file: &GgufFile, gguf_path: &Path) {
    let expected = Qwen3TtsConfig::qwen3_tts_0_6b_base();

    // ---- Top-level ----
    let arch = get_str(file, KEY_MODEL_ARCH, gguf_path);
    assert_eq!(
        arch,
        EXPECTED_ARCH,
        "{}: `{KEY_MODEL_ARCH}` = {arch:?} != {EXPECTED_ARCH:?} — the runtime dispatch keys \
         off this exact string; a silent mismatch would mis-route to CosyVoice2 / Chatterbox",
        gguf_path.display(),
    );

    let sample_rate = get_u32(file, "vokra.qwen3_tts.sample_rate", gguf_path);
    assert_eq!(
        sample_rate, QWEN3_TTS_SAMPLE_RATE,
        "sample_rate mismatch: GGUF={sample_rate} runtime primary-source={QWEN3_TTS_SAMPLE_RATE}"
    );
    assert_eq!(
        sample_rate, expected.sample_rate,
        "sample_rate mismatch vs Qwen3TtsConfig::qwen3_tts_0_6b_base()"
    );

    let speaker_embed_dim = get_u32(file, "vokra.qwen3_tts.speaker_embed_dim", gguf_path);
    assert_eq!(
        speaker_embed_dim, QWEN3_TTS_SPEAKER_EMBED_DIM,
        "speaker_embed_dim mismatch: GGUF={speaker_embed_dim} runtime={QWEN3_TTS_SPEAKER_EMBED_DIM}"
    );
    assert_eq!(speaker_embed_dim, expected.speaker_embed_dim);

    // ---- Talker (main AR LM) ----
    let t = &expected.talker;
    assert_eq!(
        get_u32(file, "vokra.qwen3_tts.talker.hidden_dim", gguf_path),
        t.hidden_dim,
        "talker.hidden_dim drift"
    );
    assert_eq!(
        get_u32(file, "vokra.qwen3_tts.talker.n_layer", gguf_path),
        t.n_layer,
        "talker.n_layer drift"
    );
    assert_eq!(
        get_u32(file, "vokra.qwen3_tts.talker.n_head", gguf_path),
        t.n_head,
        "talker.n_head drift"
    );
    assert_eq!(
        get_u32(file, "vokra.qwen3_tts.talker.n_head_kv", gguf_path),
        t.n_head_kv,
        "talker.n_head_kv drift (Qwen3 GQA: n_head_kv must divide n_head)"
    );
    assert_eq!(
        get_u32(file, "vokra.qwen3_tts.talker.head_dim", gguf_path),
        t.head_dim,
        "talker.head_dim drift (RoPE requires even head_dim)"
    );
    assert_eq!(
        get_u32(file, "vokra.qwen3_tts.talker.ffn_dim", gguf_path),
        t.ffn_dim,
        "talker.ffn_dim drift"
    );
    assert_eq!(
        get_u32(file, "vokra.qwen3_tts.talker.vocab_size", gguf_path),
        t.vocab_size,
        "talker.vocab_size drift"
    );
    assert_eq!(
        get_u32(file, "vokra.qwen3_tts.talker.text_vocab_size", gguf_path),
        t.text_vocab_size,
        "talker.text_vocab_size drift (Qwen3 shared text vocabulary)"
    );
    assert_eq!(
        get_u32(
            file,
            "vokra.qwen3_tts.talker.max_position_embeddings",
            gguf_path
        ),
        t.max_position_embeddings,
        "talker.max_position_embeddings drift"
    );
    let rope_base = get_f32(file, "vokra.qwen3_tts.talker.rope_base", gguf_path);
    assert!(
        (rope_base - t.rope_base).abs() < 1e-3,
        "talker.rope_base drift: GGUF={rope_base} runtime={} (Qwen3 widens to 1_000_000)",
        t.rope_base,
    );
    let rms_norm_eps = get_f32(file, "vokra.qwen3_tts.talker.rms_norm_eps", gguf_path);
    assert!(
        (rms_norm_eps - t.rms_norm_eps).abs() < 1e-12,
        "talker.rms_norm_eps drift: GGUF={rms_norm_eps} runtime={}",
        t.rms_norm_eps,
    );
    assert_eq!(
        get_u32(
            file,
            "vokra.qwen3_tts.talker.position_id_per_seconds",
            gguf_path
        ),
        t.position_id_per_seconds,
        "talker.position_id_per_seconds drift"
    );
    assert_eq!(
        get_u32(file, "vokra.qwen3_tts.talker.num_code_groups", gguf_path),
        t.num_code_groups,
        "talker.num_code_groups drift (must equal the codec's num_quantizers)"
    );
    assert_eq!(
        t.num_code_groups, QWEN3_TTS_NUM_CODE_GROUPS,
        "runtime primary-source drift: Qwen3TtsConfig != QWEN3_TTS_NUM_CODE_GROUPS"
    );
    assert_eq!(
        get_u32(file, "vokra.qwen3_tts.talker.text_hidden_size", gguf_path),
        t.text_hidden_size,
        "talker.text_hidden_size drift"
    );

    // ---- Code predictor (per-step 16-codebook parallel head) ----
    let cp = &expected.code_predictor;
    assert_eq!(
        get_u32(file, "vokra.qwen3_tts.code_predictor.hidden_dim", gguf_path),
        cp.hidden_dim,
        "code_predictor.hidden_dim drift"
    );
    assert_eq!(
        get_u32(file, "vokra.qwen3_tts.code_predictor.n_layer", gguf_path),
        cp.n_layer,
        "code_predictor.n_layer drift (5 for the 0.6B release)"
    );
    assert_eq!(
        get_u32(file, "vokra.qwen3_tts.code_predictor.n_head", gguf_path),
        cp.n_head,
        "code_predictor.n_head drift"
    );
    assert_eq!(
        get_u32(file, "vokra.qwen3_tts.code_predictor.n_head_kv", gguf_path),
        cp.n_head_kv,
        "code_predictor.n_head_kv drift"
    );
    assert_eq!(
        get_u32(file, "vokra.qwen3_tts.code_predictor.head_dim", gguf_path),
        cp.head_dim,
        "code_predictor.head_dim drift"
    );
    assert_eq!(
        get_u32(file, "vokra.qwen3_tts.code_predictor.ffn_dim", gguf_path),
        cp.ffn_dim,
        "code_predictor.ffn_dim drift"
    );
    assert_eq!(
        get_u32(file, "vokra.qwen3_tts.code_predictor.vocab_size", gguf_path),
        cp.vocab_size,
        "code_predictor.vocab_size drift (2048 acoustic; talker keeps a wider semantic vocab)"
    );
    let cp_rope = get_f32(file, "vokra.qwen3_tts.code_predictor.rope_base", gguf_path);
    assert!(
        (cp_rope - cp.rope_base).abs() < 1e-3,
        "code_predictor.rope_base drift"
    );
    let cp_eps = get_f32(
        file,
        "vokra.qwen3_tts.code_predictor.rms_norm_eps",
        gguf_path,
    );
    assert!(
        (cp_eps - cp.rms_norm_eps).abs() < 1e-12,
        "code_predictor.rms_norm_eps drift"
    );
    assert_eq!(
        get_u32(
            file,
            "vokra.qwen3_tts.code_predictor.num_code_groups",
            gguf_path
        ),
        cp.num_code_groups,
        "code_predictor.num_code_groups drift"
    );

    // ---- Codec handshake ----
    // The talker slots N codebook rows per step, the code predictor emits
    // N rows per step, and the codec expects N per-quantizer streams. A
    // silent mismatch here would drop or duplicate codebook rows.
    assert_eq!(
        t.num_code_groups, cp.num_code_groups,
        "talker.num_code_groups != code_predictor.num_code_groups — codec handshake broken"
    );

    // ---- Model family marker ----
    let family = get_str(file, "vokra.qwen3_tts.model_family", gguf_path);
    assert_eq!(
        family, "qwen3",
        "model_family drift: {family:?} != \"qwen3\" (Qwen3-flavour — same op set as Qwen2 \
         but wider head split + rope base 1_000_000)"
    );

    // ---- Config validates against the canonical codec ----
    expected
        .validate_for_forward()
        .expect("primary-source config must validate against the canonical qwen3_tts_12hz codec");

    eprintln!(
        "[parity_tts_qwen3] {} `vokra.qwen3_tts.*` chunk group matches \
         Qwen3TtsConfig::qwen3_tts_0_6b_base() ({} tensors present)",
        gguf_path.display(),
        file.tensors().len(),
    );
}

/// Loud "the GGUF has at least one talker weight tensor" check.
///
/// A metadata-only GGUF (the converter's `report.written == 0` path) is a
/// legitimate conversion outcome (BF16 pass-through arm not yet wired for
/// the release build), but a parity leg claiming a pass on it would be
/// fabricated — no weights means no forward, so no reference could ever
/// have run. Fail loudly (FR-EX-08).
fn assert_has_talker_weights(file: &GgufFile, gguf_path: &Path) {
    let has_talker = any_tensor(file, |name| name.starts_with("talker."));
    assert!(
        has_talker,
        "{}: no `talker.*` weight tensors present — the GGUF is metadata-only \
         (converter `report.written == 0` path; upstream is BF16 and the streaming BF16 \
         pass-through has not landed yet). Refusing to report a pass on a weightless \
         artifact (FR-EX-08). Re-convert after widening to F32 offline, or wait for the \
         Moshi / Kyutai-STT streaming BF16 arm.",
        gguf_path.display(),
    );
    eprintln!(
        "[parity_tts_qwen3] {}: talker weight tensors present ({} total tensors)",
        gguf_path.display(),
        file.tensors().len(),
    );
}

// ---------------------------------------------------------------------------
// Reference-dir cross-check (flip-the-switch: fires as soon as the owner
// drops the upstream config.json next to the converted GGUF)
// ---------------------------------------------------------------------------

/// Looks up a JSON object field by key; panics with context on miss (the
/// reference dir is opted-in fixture set, so incomplete = failure).
fn json_get<'v>(obj: &'v JsonValue, key: &str, ctx: &str) -> &'v JsonValue {
    obj.get(key)
        .unwrap_or_else(|| panic!("{ctx}: missing JSON key `{key}`"))
}

/// Reads a JSON integer field as `u32` (rejecting out-of-range values).
fn json_u32(obj: &JsonValue, key: &str, ctx: &str) -> u32 {
    let v = json_get(obj, key, ctx);
    let n = v
        .as_u64()
        .unwrap_or_else(|| panic!("{ctx}: `{key}` not a non-negative integer: {v:?}"));
    u32::try_from(n).unwrap_or_else(|_| panic!("{ctx}: `{key}` = {n} does not fit in u32"))
}

/// Reads a JSON numeric field (int or float) as `f32`.
fn json_f32(obj: &JsonValue, key: &str, ctx: &str) -> f32 {
    let v = json_get(obj, key, ctx);
    match v {
        JsonValue::Int(i) => *i as f32,
        JsonValue::Float(f) => *f as f32,
        other => panic!("{ctx}: `{key}` not a JSON number: {other:?}"),
    }
}

/// Cross-checks the upstream `config.json` (dropped by the owner into
/// `refdir`) against every `vokra.qwen3_tts.*` metadata chunk in `file`.
///
/// This is the flip-the-switch step: absent the refdir, the harness only
/// checks the GGUF vs the runtime's transcribed constants (a two-crate
/// handshake); present, the check binds both against the primary source.
///
/// Panics on any missing / type-wrong field (opted-in = incomplete
/// setup is a hard failure, per the `parity_cosyvoice2.rs` precedent).
fn assert_gguf_matches_reference_config(file: &GgufFile, refdir: &Path, gguf_path: &Path) {
    let cfg_path = refdir.join("config.json");
    let bytes = std::fs::read(&cfg_path).unwrap_or_else(|e| {
        panic!(
            "{}: unreadable — drop the upstream `config.json` from \
             huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base here: {e}",
            cfg_path.display(),
        )
    });
    let root = json::parse(&bytes)
        .unwrap_or_else(|e| panic!("{}: JSON parse error: {e}", cfg_path.display()));

    let ctx_top = format!("{}", cfg_path.display());

    // --- Top-level ---
    // Upstream ships `model_type = "qwen3_tts"` and
    // `architectures = ["Qwen3TTSForConditionalGeneration"]`.
    let model_type = json_get(&root, "model_type", &ctx_top)
        .as_str()
        .unwrap_or_else(|| panic!("{ctx_top}: `model_type` not a string"));
    assert_eq!(
        model_type, "qwen3_tts",
        "{ctx_top}: `model_type` = {model_type:?} != \"qwen3_tts\" — either the reference \
         dir is not Qwen3-TTS-12Hz-0.6B-Base or upstream renamed the config type"
    );

    // The `talker.*` and `code_predictor.*` sub-blocks carry the transcribed
    // hparams. Miss = the reference is a different variant / release shape
    // → hard-fail (opted-in setup completeness).
    let talker = json_get(&root, "talker", &ctx_top);
    let cp = json_get(&root, "code_predictor", &ctx_top);
    let ctx_talker = format!("{ctx_top}#talker");
    let ctx_cp = format!("{ctx_top}#code_predictor");

    // --- Talker cross-checks (GGUF vs upstream config.json) ---
    // Plain fns (not closures) so both accumulators can coexist without
    // tripping the borrow checker on `&mut mismatches`.
    fn check_u32(mismatches: &mut Vec<String>, name: &str, gguf_v: u32, cfg_v: u32) {
        if gguf_v != cfg_v {
            mismatches.push(format!("{name}: GGUF={gguf_v} config.json={cfg_v}"));
        }
    }
    fn check_f32(mismatches: &mut Vec<String>, name: &str, gguf_v: f32, cfg_v: f32, tol: f32) {
        if (gguf_v - cfg_v).abs() > tol {
            mismatches.push(format!(
                "{name}: GGUF={gguf_v} config.json={cfg_v} (|Δ|>{tol})"
            ));
        }
    }
    let gguf = |key: &str| get_u32(file, key, gguf_path);
    let gguf_f = |key: &str| get_f32(file, key, gguf_path);
    let mut mismatches: Vec<String> = Vec::new();

    // Talker
    check_u32(
        &mut mismatches,
        "talker.hidden_size / hidden_dim",
        gguf("vokra.qwen3_tts.talker.hidden_dim"),
        json_u32(talker, "hidden_size", &ctx_talker),
    );
    check_u32(
        &mut mismatches,
        "talker.num_hidden_layers / n_layer",
        gguf("vokra.qwen3_tts.talker.n_layer"),
        json_u32(talker, "num_hidden_layers", &ctx_talker),
    );
    check_u32(
        &mut mismatches,
        "talker.num_attention_heads / n_head",
        gguf("vokra.qwen3_tts.talker.n_head"),
        json_u32(talker, "num_attention_heads", &ctx_talker),
    );
    check_u32(
        &mut mismatches,
        "talker.num_key_value_heads / n_head_kv",
        gguf("vokra.qwen3_tts.talker.n_head_kv"),
        json_u32(talker, "num_key_value_heads", &ctx_talker),
    );
    check_u32(
        &mut mismatches,
        "talker.head_dim",
        gguf("vokra.qwen3_tts.talker.head_dim"),
        json_u32(talker, "head_dim", &ctx_talker),
    );
    check_u32(
        &mut mismatches,
        "talker.intermediate_size / ffn_dim",
        gguf("vokra.qwen3_tts.talker.ffn_dim"),
        json_u32(talker, "intermediate_size", &ctx_talker),
    );
    check_u32(
        &mut mismatches,
        "talker.vocab_size",
        gguf("vokra.qwen3_tts.talker.vocab_size"),
        json_u32(talker, "vocab_size", &ctx_talker),
    );
    check_u32(
        &mut mismatches,
        "talker.text_vocab_size",
        gguf("vokra.qwen3_tts.talker.text_vocab_size"),
        json_u32(talker, "text_vocab_size", &ctx_talker),
    );
    check_u32(
        &mut mismatches,
        "talker.max_position_embeddings",
        gguf("vokra.qwen3_tts.talker.max_position_embeddings"),
        json_u32(talker, "max_position_embeddings", &ctx_talker),
    );
    check_f32(
        &mut mismatches,
        "talker.rope_theta / rope_base",
        gguf_f("vokra.qwen3_tts.talker.rope_base"),
        json_f32(talker, "rope_theta", &ctx_talker),
        1e-3,
    );
    check_f32(
        &mut mismatches,
        "talker.rms_norm_eps",
        gguf_f("vokra.qwen3_tts.talker.rms_norm_eps"),
        json_f32(talker, "rms_norm_eps", &ctx_talker),
        1e-12,
    );
    check_u32(
        &mut mismatches,
        "talker.position_id_per_seconds",
        gguf("vokra.qwen3_tts.talker.position_id_per_seconds"),
        json_u32(talker, "position_id_per_seconds", &ctx_talker),
    );
    check_u32(
        &mut mismatches,
        "talker.num_code_groups",
        gguf("vokra.qwen3_tts.talker.num_code_groups"),
        json_u32(talker, "num_code_groups", &ctx_talker),
    );
    check_u32(
        &mut mismatches,
        "talker.text_hidden_size",
        gguf("vokra.qwen3_tts.talker.text_hidden_size"),
        json_u32(talker, "text_hidden_size", &ctx_talker),
    );

    // Code predictor
    check_u32(
        &mut mismatches,
        "code_predictor.hidden_size / hidden_dim",
        gguf("vokra.qwen3_tts.code_predictor.hidden_dim"),
        json_u32(cp, "hidden_size", &ctx_cp),
    );
    check_u32(
        &mut mismatches,
        "code_predictor.num_hidden_layers / n_layer",
        gguf("vokra.qwen3_tts.code_predictor.n_layer"),
        json_u32(cp, "num_hidden_layers", &ctx_cp),
    );
    check_u32(
        &mut mismatches,
        "code_predictor.num_attention_heads / n_head",
        gguf("vokra.qwen3_tts.code_predictor.n_head"),
        json_u32(cp, "num_attention_heads", &ctx_cp),
    );
    check_u32(
        &mut mismatches,
        "code_predictor.num_key_value_heads / n_head_kv",
        gguf("vokra.qwen3_tts.code_predictor.n_head_kv"),
        json_u32(cp, "num_key_value_heads", &ctx_cp),
    );
    check_u32(
        &mut mismatches,
        "code_predictor.head_dim",
        gguf("vokra.qwen3_tts.code_predictor.head_dim"),
        json_u32(cp, "head_dim", &ctx_cp),
    );
    check_u32(
        &mut mismatches,
        "code_predictor.intermediate_size / ffn_dim",
        gguf("vokra.qwen3_tts.code_predictor.ffn_dim"),
        json_u32(cp, "intermediate_size", &ctx_cp),
    );
    check_u32(
        &mut mismatches,
        "code_predictor.vocab_size",
        gguf("vokra.qwen3_tts.code_predictor.vocab_size"),
        json_u32(cp, "vocab_size", &ctx_cp),
    );
    check_f32(
        &mut mismatches,
        "code_predictor.rope_theta / rope_base",
        gguf_f("vokra.qwen3_tts.code_predictor.rope_base"),
        json_f32(cp, "rope_theta", &ctx_cp),
        1e-3,
    );
    check_f32(
        &mut mismatches,
        "code_predictor.rms_norm_eps",
        gguf_f("vokra.qwen3_tts.code_predictor.rms_norm_eps"),
        json_f32(cp, "rms_norm_eps", &ctx_cp),
        1e-12,
    );
    check_u32(
        &mut mismatches,
        "code_predictor.num_code_groups",
        gguf("vokra.qwen3_tts.code_predictor.num_code_groups"),
        json_u32(cp, "num_code_groups", &ctx_cp),
    );

    assert!(
        mismatches.is_empty(),
        "GGUF-vs-`config.json` drift ({} mismatch{}) — every axis must \
         match the upstream primary source verbatim, else the runtime \
         would mis-shape the Qwen3 backbone / codec handshake:\n  - {}",
        mismatches.len(),
        if mismatches.len() == 1 { "" } else { "es" },
        mismatches.join("\n  - "),
    );

    eprintln!(
        "[parity_tts_qwen3] flip-the-switch OK: GGUF metadata cross-verified \
         against upstream config.json at {} (talker {} axes + code_predictor {} axes \
         all bit-for-bit equal)",
        cfg_path.display(),
        13, // talker cross-checks above
        10, // code predictor cross-checks above
    );
}

// ---------------------------------------------------------------------------
// One #[test] per model in the tts-qwen3 family (currently: qwen3_tts,
// the Qwen/Qwen3-TTS-12Hz-0.6B-Base release)
// ---------------------------------------------------------------------------

/// **qwen3_tts** — Qwen/Qwen3-TTS-12Hz-0.6B-Base (Apache-2.0 end-to-end).
///
/// Flip-the-switch parity:
///
/// | env vars set                                                | leg that fires                                  |
/// | :---------------------------------------------------------- | :---------------------------------------------- |
/// | *(none)*                                                    | clean skip with reason (never a fabricated pass) |
/// | `VOKRA_QWEN3_TTS_GGUF`                                      | load + arch + `vokra.qwen3_tts.*` chunk check + talker-weight sanity |
/// | `VOKRA_QWEN3_TTS_GGUF` + `VOKRA_QWEN3_TTS_REFDIR`           | above + cross-check every axis against `<refdir>/config.json` |
#[test]
fn parity_qwen3_tts_qwen3_tts_0_6b_base() {
    let arch = "qwen3_tts";
    let hf_repo = "Qwen/Qwen3-TTS-12Hz-0.6B-Base";
    let (gguf_env, refdir_env) = env_paths_for(arch);

    let Some(gguf_path) = gguf_env else {
        eprintln!("{}", skip_reason(arch, hf_repo));
        return;
    };

    // Opted-in: unreadable / malformed / arch-mismatched all hard-fail
    // (never a silent skip once the env var is set).
    let bytes = std::fs::read(&gguf_path).unwrap_or_else(|e| {
        panic!(
            "VOKRA_QWEN3_TTS_GGUF = {}: unreadable: {e}",
            gguf_path.display(),
        )
    });
    let file = GgufFile::parse(bytes).unwrap_or_else(|e| {
        panic!(
            "VOKRA_QWEN3_TTS_GGUF = {}: not a parseable GGUF: {e:?}",
            gguf_path.display(),
        )
    });

    assert_metadata_matches_primary_source(&file, &gguf_path);
    assert_has_talker_weights(&file, &gguf_path);

    if let Some(refdir) = refdir_env {
        assert_gguf_matches_reference_config(&file, &refdir, &gguf_path);
    } else {
        eprintln!(
            "[parity_tts_qwen3] refdir leg (flip-the-switch): \
             VOKRA_QWEN3_TTS_REFDIR unset — GGUF-vs-runtime handshake ran, \
             upstream `config.json` cross-check did NOT (drop the upstream \
             config.json into a dir and re-run with VOKRA_QWEN3_TTS_REFDIR=<dir> \
             to enable). This is a clean gated skip, not a fabricated pass."
        );
    }
}

// ---------------------------------------------------------------------------
// Unit-only helper coverage (fires on every `cargo test`, no env / no fs /
// no real GGUF). Sibling precedent: `parity_tts_dac.rs` lines 651-726 and
// `parity_tts_japanese.rs` lines 245-306. SoTA Phase 1 audit (2026-07-25).
//
// Each block below explains its rationale inline so a future auditor sees
// why the test exists without cross-reference. The eleven blocks address:
//
//   (1)  `env_paths_for` on both-unset — pins the "unset → clean skip"
//        contract at the helper level so a future refactor that returned
//        `Some(PathBuf::default())` fails here, not deep in a live parity
//        job (sibling: `parity_tts_dac.rs::env_paths_for_returns_none_when_unset`).
//   (2)  `env_paths_for` arch-form invariance — the `.replace('-', "_")`
//        + `.to_ascii_uppercase()` transformation had zero pin; a silent
//        regression dropping the hyphen swap would emit
//        `VOKRA_QWEN3-TTS_GGUF`, which is un-set-able from a POSIX shell.
//   (3)  `skip_reason` reproduction-recipe tokens — the CI summary parser
//        greps stderr for these tokens; a silent edit that dropped one
//        would leave a bare `skipped` annotation, defeating the harness's
//        docstring promise (`[parity_tts_qwen3] SKIP:` prefix).
//   (4)  `skip_reason` names REFDIR — the AUDIT FINDING: the baseline
//        (both unset) never surfaced the refdir env-var name in stderr,
//        so a first-time operator had to grep the source file to discover
//        the flip-the-switch leg exists. Sibling
//        `parity_tts_japanese.rs::skip_reason_names_both_env_vars_and_the_convert_recipe`
//        enforces the same self-documenting-skip invariant.
//   (5)  `json_get` missing-key panic — the panic branch only fires under
//        a real refdir walk; without a standalone pin, a copy-paste error
//        that swapped `{ctx}` and `{key}` in the panic message would go
//        undetected.
//   (6)  `json_u32` negative-int panic — `as_u64()` returns `None` for
//        negative `Int`; a future refactor to `as_i64` + unchecked cast
//        would silently accept `-1 as u32 == u32::MAX`, corrupting every
//        talker/code_predictor cross-check.
//   (7)  `json_u32` u32-overflow panic — every talker.* / code_predictor.*
//        axis is a `u32`; a config with `vocab_size` accidentally emitted
//        as `u64::MAX` must fail loudly at cross-check, not silently
//        truncate.
//   (8)  `json_f32` dual-int/float acceptance — the upstream `config.json`
//        may emit `rope_theta` as either `1000000` or `1000000.0`; the
//        harness's rope_base cross-check depends on this dual handling.
//   (9)  `json_f32` non-number panic — silent-coercion regression risk:
//        a `String => s.parse().unwrap_or(0.0)` arm would let a corrupted
//        config compare against `0.0` and silently drift.
//   (10) `synthesize` FR-EX-08 refusal — the file's most load-bearing pin.
//        Without this a future patch returning `Ok(vec![0.0; 24_000])`
//        (silent hallucination) would slip past this harness because the
//        one gated `#[test]` only exercises `synthesize` when the owner
//        has provisioned a real GGUF, which is NOT the CI baseline.
//   (11) Primary-source config validates + module-const agreement — the
//        primary source is transcribed from `config.json` (fetched
//        2026-07-24, per CLAUDE.md「ハルシネーション厳禁」). A silent
//        drift in that transcription — e.g. `hidden_dim` off by a factor
//        of 2 — would only surface today when the owner runs with a real
//        GGUF; a pure standalone pin catches it on every `cargo test`.
// ---------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// (1) `env_paths_for` — both-unset baseline
// -----------------------------------------------------------------------------

/// `env_paths_for` must return `(None, None)` when both env vars are
/// unset (the CI baseline). Sibling precedent:
/// `parity_tts_dac.rs::env_paths_for_returns_none_when_unset`.
///
/// The workspace forbids `unsafe` (`-D unsafe-code`) and
/// `std::env::set_var` is unsafe in Rust 2024, so this test cannot
/// exercise the "env set" branches directly without breaking the lint
/// posture. The unset baseline is nonetheless the CI-critical arm — a
/// refactor that accidentally returned `Some(PathBuf::default())` (e.g.
/// `.map(PathBuf::from).or(Some(_))`) would silently start opening `""`
/// as a GGUF path.
#[test]
fn env_paths_for_returns_none_when_both_env_vars_unset() {
    // Arch string namespaced to this helper so no legitimate CI env var
    // of the corresponding `VOKRA_*_GGUF` / `_REFDIR` name can collide.
    let arch = "qwen3_tts_harness_helper_only";
    let (gguf, refdir) = env_paths_for(arch);
    assert!(
        gguf.is_none(),
        "expected VOKRA_QWEN3_TTS_HARNESS_HELPER_ONLY_GGUF unset in the test env, \
         got Some({gguf:?}) — env leakage?"
    );
    assert!(
        refdir.is_none(),
        "expected VOKRA_QWEN3_TTS_HARNESS_HELPER_ONLY_REFDIR unset in the test env, \
         got Some({refdir:?})"
    );
}

// -----------------------------------------------------------------------------
// (2) `env_paths_for` — arch → env-var-name transformation
// -----------------------------------------------------------------------------

/// `env_paths_for` — kebab and snake spellings of the same arch must
/// produce the SAME env-var name (the `.replace('-', "_")` step is what
/// makes `qwen3-tts` and `qwen3_tts` interchangeable at the call site).
///
/// A silent regression that dropped the hyphen swap would emit
/// `VOKRA_QWEN3-TTS_GGUF` — un-set-able from a POSIX shell. This test
/// runs entirely against unset env-vars (both spellings resolve to
/// `None` for reasons independent of the transformation), so the
/// invariant is that they resolve to the SAME payload, then additionally
/// pin the expected env-var name inline.
#[test]
fn env_paths_for_kebab_and_snake_arch_resolve_identically_on_unset() {
    let arch_snake = "qwen3_tts_harness_helper_only";
    let arch_kebab = "qwen3-tts-harness-helper-only";
    let (gguf_snake, refdir_snake) = env_paths_for(arch_snake);
    let (gguf_kebab, refdir_kebab) = env_paths_for(arch_kebab);
    assert_eq!(
        gguf_snake, gguf_kebab,
        "snake vs kebab arch must resolve to the same env-var payload"
    );
    assert_eq!(
        refdir_snake, refdir_kebab,
        "snake vs kebab arch must resolve to the same env-var payload"
    );
    assert!(
        gguf_snake.is_none() && refdir_snake.is_none(),
        "these namespaced arches must remain unset in the test env"
    );

    // Pin the transformation itself by rebuilding the expected env-var
    // name inline (identical shape to `env_paths_for`'s internal
    // `format!("VOKRA_{arch_env}_GGUF", ...)`). A regression dropping
    // the hyphen swap would emit `VOKRA_QWEN3-TTS_GGUF` here.
    let expected_env = format!(
        "VOKRA_{}_GGUF",
        "qwen3-tts".to_ascii_uppercase().replace('-', "_"),
    );
    assert_eq!(
        expected_env, "VOKRA_QWEN3_TTS_GGUF",
        "the arch → env-var-name transformation drifted"
    );
}

// -----------------------------------------------------------------------------
// (3) `skip_reason` — reproduction-recipe tokens
// -----------------------------------------------------------------------------

/// `skip_reason` must contain every token an operator needs to reproduce
/// both legs (GGUF conversion + flip-the-switch refdir cross-check).
/// Sibling precedent:
/// `parity_tts_dac.rs::skip_reason_contains_reproduction_recipe`.
#[test]
fn skip_reason_contains_reproduction_recipe() {
    let msg = skip_reason("qwen3_tts", "Qwen/Qwen3-TTS-12Hz-0.6B-Base");
    for token in &[
        "[parity_tts_qwen3] SKIP:",
        "VOKRA_QWEN3_TTS_GGUF",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "vokra-cli convert",
        "--model qwen3-tts",
        "fabricated pass",
        "FR-EX-08",
    ] {
        assert!(
            msg.contains(token),
            "skip_reason must contain {token:?}; got: {msg}"
        );
    }
}

// -----------------------------------------------------------------------------
// (4) `skip_reason` — names BOTH env vars (audit finding)
// -----------------------------------------------------------------------------

/// `skip_reason` must name BOTH env vars so the baseline stderr
/// (both unset — the CI baseline) surfaces the flip-the-switch refdir
/// leg's env-var name as well as the GGUF env-var name. Sibling
/// precedent: `parity_tts_japanese.rs::skip_reason_names_both_env_vars_and_the_convert_recipe`.
///
/// The original `skip_reason` implementation only named the GGUF env
/// var, leaving a first-time operator to grep the source to discover
/// the flip-the-switch leg exists. This pin was added at the same
/// commit that updated `skip_reason` to be self-documenting.
#[test]
fn skip_reason_names_both_env_vars() {
    let msg = skip_reason("qwen3_tts", "Qwen/Qwen3-TTS-12Hz-0.6B-Base");
    assert!(
        msg.contains("VOKRA_QWEN3_TTS_GGUF"),
        "skip message omits GGUF env var: {msg:?}"
    );
    assert!(
        msg.contains("VOKRA_QWEN3_TTS_REFDIR"),
        "skip message omits REFDIR env var (audit finding — self-documenting-skip \
         invariant): {msg:?}"
    );
}

// -----------------------------------------------------------------------------
// (5) `json_get` — missing-key panic reachability
// -----------------------------------------------------------------------------

/// `json_get`'s missing-key panic branch only fires under a real refdir
/// walk; without a standalone pin, a copy-paste error that swapped
/// `{ctx}` and `{key}` in the panic message would go undetected until
/// an operator provisioned a reference dir.
///
/// The `expected` string matches a substring of
/// `"{ctx}: missing JSON key `{key}`"` — the backticks around `{key}`
/// are non-portable in `#[should_panic(expected=...)]` matchers if the
/// test asserts on the full field, so we anchor on the invariant
/// prefix.
#[test]
#[should_panic(expected = "missing JSON key")]
fn json_get_panics_on_missing_key() {
    let obj = JsonValue::Object(vec![("present".to_owned(), JsonValue::Int(1))]);
    let _ = json_get(&obj, "absent_key", "some/ctx");
}

// -----------------------------------------------------------------------------
// (6) `json_u32` — rejects negative integers
// -----------------------------------------------------------------------------

/// `json_u32` MUST panic on a negative `Int`. `JsonValue::as_u64()`
/// returns `None` for negatives today; a future refactor to `as_i64` +
/// unchecked cast would silently accept `-1 as u32 == u32::MAX`,
/// corrupting every `talker.*` / `code_predictor.*` cross-check.
#[test]
#[should_panic(expected = "not a non-negative integer")]
fn json_u32_panics_on_negative_int() {
    let obj = JsonValue::Object(vec![("k".to_owned(), JsonValue::Int(-1))]);
    let _ = json_u32(&obj, "k", "ctx");
}

// -----------------------------------------------------------------------------
// (7) `json_u32` — rejects values above `u32::MAX`
// -----------------------------------------------------------------------------

/// `json_u32` MUST panic when the value exceeds `u32::MAX`. Every
/// talker / code-predictor axis is a `u32` — a config with a
/// `vocab_size` accidentally emitted as `u64::MAX` (upstream refactor,
/// exporter bug) must fail loudly at cross-check, not silently
/// truncate.
#[test]
#[should_panic(expected = "does not fit in u32")]
fn json_u32_panics_when_value_exceeds_u32_max() {
    let too_big = i64::from(u32::MAX) + 1;
    let obj = JsonValue::Object(vec![("k".to_owned(), JsonValue::Int(too_big))]);
    let _ = json_u32(&obj, "k", "ctx");
}

// -----------------------------------------------------------------------------
// (8) `json_f32` — accepts both `Int` and `Float`
// -----------------------------------------------------------------------------

/// `json_f32` must accept BOTH `JsonValue::Int` and `JsonValue::Float`.
/// The upstream `config.json` may emit `rope_theta` as either
/// `1000000` or `1000000.0` depending on the exporter version; both
/// must be numerically equivalent to the harness's rope_base
/// cross-check.
#[test]
fn json_f32_accepts_both_int_and_float_variants() {
    let obj = JsonValue::Object(vec![
        ("as_int".to_owned(), JsonValue::Int(42)),
        ("as_float".to_owned(), JsonValue::Float(1.0e6)),
    ]);
    let v_int = json_f32(&obj, "as_int", "ctx");
    let v_float = json_f32(&obj, "as_float", "ctx");
    // Int → f32: 42 is exactly representable.
    assert_eq!(
        v_int, 42.0_f32,
        "json_f32(Int(42)) drifted to {v_int} (expected exactly 42.0)"
    );
    // Float → f32: 1_000_000 is exactly representable in f32.
    assert_eq!(
        v_float, 1.0e6_f32,
        "json_f32(Float(1e6)) drifted to {v_float}"
    );
}

// -----------------------------------------------------------------------------
// (9) `json_f32` — rejects non-numeric values
// -----------------------------------------------------------------------------

/// `json_f32` MUST panic on a non-numeric value (Bool, String, Null,
/// Array, Object). A silent-coercion regression — e.g. an added
/// `JsonValue::Str(s) => s.parse().unwrap_or(0.0)` arm — would let a
/// corrupted config compare against `0.0` and silently drift.
#[test]
#[should_panic(expected = "not a JSON number")]
fn json_f32_panics_on_non_number() {
    let obj = JsonValue::Object(vec![("k".to_owned(), JsonValue::Bool(true))]);
    let _ = json_f32(&obj, "k", "ctx");
}

// -----------------------------------------------------------------------------
// (10) `Qwen3TtsTts::synthesize` — FR-EX-08 refusal pin
// -----------------------------------------------------------------------------

/// FR-EX-08 pin — `Qwen3TtsTts::synthesize` MUST refuse loudly on both
/// (a) empty text (`VokraError::InvalidArgument`) and (b) a real call
/// against synthesized weights (`VokraError::NotImplemented` naming
/// the hallucinated-waveform blocker).
///
/// This is the file's most load-bearing FR-EX-08 pin. The file's one
/// gated `#[test]` (`parity_qwen3_tts_qwen3_tts_0_6b_base`) does NOT
/// exercise `synthesize` on the CI baseline (unset env vars → early
/// return with skip message), and the follow-up-wave real-forward
/// test (`parity_qwen3_tts_synthesize_matches_upstream_pcm`, named in
/// the module docstring) has not landed yet. Without this pin, a
/// future patch that accidentally returned `Ok(vec![0.0; N])` (silent
/// hallucination) would slip past this harness on every CI run.
///
/// This assertion goes AWAY the moment real weights bind and the
/// forward path lands (at which point this test would be replaced by
/// a real end-to-end audio-bound sanity check).
///
/// The engine is built via a hand-rolled small-aligned config (mirrors
/// `crates/vokra-models/src/qwen3_tts/mod.rs::small_aligned_config`)
/// because `Qwen3TtsConfig::tiny_for_tests()` uses `num_code_groups=3`,
/// which fails the codec handshake against the canonical 16-quantizer
/// codec inside `Qwen3TtsWeights::synthesized`. Aligning
/// `num_code_groups=16` keeps this test fast (KB-sized synthesized
/// weights) while still exercising the real engine construction path.
#[test]
fn synthesize_fr_ex_08_refusal_pins() {
    use vokra_core::VokraError;
    use vokra_models::qwen3_tts::{
        Qwen3TtsCodePredictorConfig, Qwen3TtsTalkerConfig, Qwen3TtsTts, Qwen3TtsWeights,
    };

    let cfg = Qwen3TtsConfig {
        sample_rate: QWEN3_TTS_SAMPLE_RATE,
        speaker_embed_dim: 8,
        talker: Qwen3TtsTalkerConfig {
            hidden_dim: 16,
            n_layer: 2,
            n_head: 4,
            n_head_kv: 2,
            head_dim: 8,
            ffn_dim: 32,
            vocab_size: 32,
            text_vocab_size: 64,
            max_position_embeddings: 128,
            rope_base: 1_000_000.0,
            rms_norm_eps: 1e-6,
            position_id_per_seconds: 13,
            num_code_groups: 16, // match canonical qwen3_tts_12hz codec
            text_hidden_size: 24,
        },
        code_predictor: Qwen3TtsCodePredictorConfig {
            hidden_dim: 16,
            n_layer: 2,
            n_head: 4,
            n_head_kv: 2,
            head_dim: 8,
            ffn_dim: 32,
            vocab_size: 24,
            rope_base: 1_000_000.0,
            rms_norm_eps: 1e-6,
            num_code_groups: 16,
        },
    };

    let weights =
        Qwen3TtsWeights::synthesized(&cfg, 42).expect("build small-aligned synthesized weights");
    let tts = Qwen3TtsTts::new(cfg, weights).expect("build engine from small-aligned weights");
    assert!(
        tts.is_synthesized(),
        "sanity: the engine must report synthesized weights"
    );

    // (a) Empty text — the InvalidArgument arm must fire, NOT the
    //     NotImplemented arm (and never `Ok(_)`).
    let empty_err = tts
        .synthesize("")
        .expect_err("synthesize(\"\") must refuse loudly");
    match empty_err {
        VokraError::InvalidArgument(msg) => {
            assert!(
                msg.contains("text is empty"),
                "empty-text arm message drifted: got InvalidArgument({msg:?})"
            );
        }
        other => panic!("expected InvalidArgument on empty text, got {other:?}"),
    }

    // (b) Non-empty text against synthesized weights — the
    //     NotImplemented arm naming the hallucinated-waveform blocker
    //     must fire (locks the specific synthesized-weights arm, not
    //     the generic "real weights bound but forward not landed" arm).
    let synth_err = tts
        .synthesize("hello")
        .expect_err("synthesize with synthesized weights must refuse loudly");
    match synth_err {
        VokraError::NotImplemented(msg) => {
            assert!(
                msg.contains("hallucinated waveform"),
                "expected the synthesized-weights arm naming \
                 \"hallucinated waveform\"; got NotImplemented({msg:?})"
            );
        }
        other => panic!("expected NotImplemented on synthesized weights, got {other:?}"),
    }
}

// -----------------------------------------------------------------------------
// (11) Primary-source config validates + module-const agreement
// -----------------------------------------------------------------------------

/// The primary-source config MUST validate against the canonical codec
/// on every `cargo test`. A silent drift in the transcribed constants
/// (e.g. `hidden_dim` off by a factor of 2, or `n_head_kv` no longer
/// dividing `n_head`) would today only surface under a real-GGUF
/// flip-the-switch run; a standalone pin catches it on every CI
/// invocation.
///
/// Additionally binds the three module-level constants imported at the
/// top of this file (`QWEN3_TTS_SAMPLE_RATE`, `QWEN3_TTS_SPEAKER_EMBED_DIM`,
/// `QWEN3_TTS_NUM_CODE_GROUPS`) against `Qwen3TtsConfig::qwen3_tts_0_6b_base`
/// so a transcription drift between the two crates surfaces here
/// rather than deep inside a gated `assert_metadata_matches_primary_source`
/// leg.
#[test]
fn primary_source_config_validates_and_module_constants_agree() {
    let cfg = Qwen3TtsConfig::qwen3_tts_0_6b_base();
    cfg.validate_for_forward()
        .expect("primary-source config must validate against the canonical qwen3_tts_12hz codec");

    // Module-level constant ↔ full-config field agreement (the harness
    // imports and cross-checks against both — a drift between them
    // would break `assert_metadata_matches_primary_source`).
    assert_eq!(
        cfg.sample_rate, QWEN3_TTS_SAMPLE_RATE,
        "module const QWEN3_TTS_SAMPLE_RATE drifted from Qwen3TtsConfig::qwen3_tts_0_6b_base()"
    );
    assert_eq!(
        cfg.speaker_embed_dim, QWEN3_TTS_SPEAKER_EMBED_DIM,
        "module const QWEN3_TTS_SPEAKER_EMBED_DIM drifted"
    );
    assert_eq!(
        cfg.talker.num_code_groups, QWEN3_TTS_NUM_CODE_GROUPS,
        "module const QWEN3_TTS_NUM_CODE_GROUPS drifted from talker.num_code_groups"
    );

    // Codec handshake — `talker.num_code_groups` == `code_predictor.num_code_groups`
    // is what `assert_metadata_matches_primary_source` cross-checks on
    // a real GGUF; standalone-pin the runtime side too so a
    // Qwen3-flavour widening (e.g. dual-stream 16 + 16) that only
    // updated one field would surface here.
    assert_eq!(
        cfg.talker.num_code_groups, cfg.code_predictor.num_code_groups,
        "runtime primary-source drift: talker vs code_predictor num_code_groups"
    );
}
