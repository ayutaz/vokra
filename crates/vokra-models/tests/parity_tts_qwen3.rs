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
fn skip_reason(arch: &str, hf_repo: &str) -> String {
    let arch_env = arch.replace('-', "_").to_ascii_uppercase();
    format!(
        "[parity_tts_qwen3] SKIP: VOKRA_{arch_env}_GGUF unset. Convert the \
         upstream `{hf_repo}` checkpoint with `vokra-cli convert --model \
         qwen3-tts --input <model.safetensors> --output <out.gguf>` and \
         re-run with `VOKRA_{arch_env}_GGUF=<out.gguf>`. This is a clean \
         gated skip, never a fabricated pass (FR-EX-08)."
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
