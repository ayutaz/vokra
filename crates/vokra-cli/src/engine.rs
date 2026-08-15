//! GGUF → native engine dispatch for the `run` / `bench` subcommands (M1-10a).
//!
//! Loads a GGUF on the CPU backend, reads `vokra.model.arch`, builds the
//! matching native engine from `vokra-models` and injects it into the
//! [`Session`]. This mirrors the private `build_session` in
//! `vokra-capi/src/session.rs`; lifting that dispatch into one public
//! `vokra_models::load` helper shared by capi + cli is a deliberate follow-up
//! (it touches vokra-capi/vokra-models, out of scope for this WP), so for now
//! the small match is duplicated here against the same public APIs. ONNX is
//! never loaded (FR-LD-05).

use std::sync::Arc;

use vokra_core::gguf::GgufFile;
use vokra_core::{BackendKind, Session, VokraError};
use vokra_models::csm::{CsmEngine, EchoPath, FixtureByteTokenizer};
use vokra_models::distil_whisper::DistilWhisperAsr;
use vokra_models::kotoba_whisper::KotobaWhisperAsr;
use vokra_models::piper_plus::PiperPlusTts;
use vokra_models::silero_vad::SileroVadV5;
use vokra_models::whisper::WhisperAsr;

/// The task a loaded model performs (selected by its architecture).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) enum ModelTask {
    /// Voice activity detection (Silero VAD v5).
    Vad,
    /// Speech-to-text (Whisper base).
    Asr,
    /// Speech-to-text through Voxtral (M3-10 / P2 cc-10).
    ///
    /// Like [`ModelTask::Speaker`], the dispatch returns a **bare session**
    /// and the `run` arm binds the concrete
    /// [`vokra_models::voxtral::VoxtralAsr`] from `session.gguf()`. This is
    /// deliberate, not an omission: Voxtral's beam surface
    /// (`transcribe_beam_with_config_overrides`, n-best + length-penalty +
    /// no-repeat-ngram) is a concrete-type API, not part of the
    /// [`vokra_core::AsrEngine`] trait, so injecting only the trait object
    /// would force a SECOND multi-GB load to reach it — the shipping mini
    /// decodes to ~12 GB of f32 weights, so two live copies do not fit on a
    /// 16 GB machine. One load, both surfaces.
    AsrVoxtral,
    /// Text-to-speech (piper-plus native TTS).
    Tts,
    /// Text-to-speech through Kokoro-82M from a **phoneme string** (cc-24).
    ///
    /// Separate from [`ModelTask::Tts`] because the two archs take different
    /// input: piper-plus implements `TtsEngine::synthesize`, which accepts
    /// graphemes-or-phonemes and tokenizes internally, whereas Kokoro's
    /// `TtsEngine::synthesize` is a hard [`vokra_core::VokraError::NotImplemented`]
    /// pending a misaki G2P bridge. The reachable Kokoro surface is the
    /// concrete [`vokra_models::kokoro::KokoroTts::synthesize_phonemes`], which
    /// takes phoneme ids and an explicit style vector.
    ///
    /// Like [`ModelTask::Speaker`] / [`ModelTask::AsrVoxtral`], the dispatch
    /// returns a **bare session** and the `run` arm binds the concrete
    /// `KokoroTts` from the model path: the trait object the session facade
    /// stores cannot reach `synthesize_phonemes`, and injecting it as well
    /// would mean loading the ~330 MB of f32 weights twice.
    TtsKokoro,
    /// Speech-to-speech dialog (Sesame CSM-1B = M4-05). The reply text is
    /// caller-supplied (`--text`), optional `--input` WAV = recorded
    /// context audio (explicit AEC bypass — T16).
    S2s,
    /// Full-duplex speech-to-speech (Moshi = M4-06). No `--text` — the
    /// model GENERATES its reply (inner monologue); `--input` WAV drives
    /// the mic side, `--duplex` selects the continuous push/pull demo
    /// with an optional `--echo-sim` synthetic echo path (T26).
    S2sDuplex,
    /// Speaker embedding (CAM++ / M0-08, FR-OP-81). `--input` WAV →
    /// 192-d embedding L2-norm; with `--compare <b.wav>` also the cosine
    /// similarity of the two embeddings (`speaker::verify`). The encoder
    /// is built in the `run` arm from the session's GGUF (the [`Session`]
    /// facade has no speaker engine slot — deliberate: the embedding is a
    /// conditioning input, not a session task).
    Speaker,
    /// Text-to-speech through SBV2 (Style-Bert-VITS2 v2 = Task 38).
    ///
    /// Like [`ModelTask::TtsKokoro`] / [`ModelTask::AsrVoxtral`] /
    /// [`ModelTask::Speaker`], the dispatch returns a **bare session** and
    /// the `run` arm binds the concrete
    /// [`vokra_models::sbv2::SbV2Model`] itself:
    /// `SbV2Model::from_gguf` needs THREE GGUFs (this model's own weights,
    /// plus the `--bert-ja` / `--bert-en` DeBERTa v2 / v3 side-cars), and
    /// the generic dispatch signature this function shares with every
    /// other arch only carries the one `--model` path (plus the Moshi-only
    /// optional `--mimi`, which is a poor fit here since SBV2's two
    /// side-cars are both *required*, not optional). Binding all three in
    /// the `run` arm — reusing `session.gguf()` for the main file, opening
    /// `--bert-ja` / `--bert-en` fresh — keeps this shared function's
    /// signature untouched.
    Sbv2,
    /// Whisper log-mel front-end only (M2-04-T11). Runs
    /// [`vokra_models::whisper::mel::log_mel`] against the input WAV without
    /// touching the encoder / decoder, so bench-side RTF isolates the fused
    /// vs unfused log-mel path (M2-04-T08 toggle) rather than folding Whisper
    /// decode time into the measurement. Selected by `--task mel-frontend`
    /// when the loaded GGUF has `vokra.model.arch = "whisper"`.
    MelFrontend,
    /// CosyVoice2 chunk-aware streaming synthetic bench (M3-09-T24 scaffold).
    ///
    /// Runs the CosyVoice2 chunk pipeline with **injected deterministic
    /// closures** (zero velocity + constant-ones code closure) against the
    /// M3-06 identity Mimi decoder fixture, so the RTF measurement path is
    /// exercised without a real safetensors checkpoint. This is the
    /// canned "cosyvoice2-synthetic" model kind the T24 spec pins as the
    /// scaffold entry point: today it verifies the measurement harness
    /// works; the real-checkpoint RTF < 1.0 hard-assert lands with the
    /// T19 CUDA seam + a self-hosted CUDA runner (mirrors the M2-14
    /// defer to a stable measurement lab).
    ///
    /// The bench-side RTF is measured over a 1 s target-frame budget: the
    /// pipeline generates a chunk-aware audio stream from a fixed
    /// deterministic seed and reports latency / RTF against a 1 s audio
    /// window (24 kHz Mimi native rate). Selected by
    /// `--task cosyvoice2-synthetic` — no `--model` required (analog to
    /// `mel-frontend`).
    ///
    /// # `dead_code` posture (M3-09-T24 landing state)
    ///
    /// The variant is intentionally *never constructed* by the current
    /// engine.rs — the standalone bench in `bench.rs` skips
    /// [`load_session_with_backend`] entirely (arch dispatch is not yet
    /// wired for `cosyvoice2`, T07/T08 follow-on). The variant is kept
    /// because the exhaustive match arms in [`crate::run::main`] and
    /// `crate::bench::execute` rely on it to surface an explicit
    /// unimplemented signal if a future engine.rs change ever *does*
    /// return it (defense in depth against a silent fall back — the
    /// FR-EX-08 posture the whole CLI upholds). The dead-code allow
    /// documents this state so a reviewer does not delete the arm.
    #[allow(dead_code)]
    Cosyvoice2Synthetic,
    // ---- Wave G (2026-08-15) — binders with a REAL forward ----------------
    /// Voice activity detection through FSMN-VAD (FunASR).
    ///
    /// A separate variant from [`ModelTask::Vad`] only so the CPU-only
    /// diagnostic in `run.rs` can name the right engine; both share the
    /// same `run` / `bench` arms because
    /// [`vokra_models::fsmn_vad::FsmnVadV1`] implements the same
    /// [`vokra_core::engines::VadEngine`] trait Silero does and is injected
    /// into the same session slot. The forward is real (verified: its
    /// `forward_features` / stream `push_pcm` run the encoder, no
    /// loud-partial gate).
    VadFsmn,
    /// Speech enhancement through NSNet2 (Microsoft DNS-Challenge baseline).
    ///
    /// Real forward: [`vokra_models::nsnet2::Nsnet2V1::denoise_pcm`] runs
    /// the STFT → fc → GRU ×2 → fc ×3 → mask → iSTFT chain. The dispatch
    /// hands back a **bare session** and the `run` arm binds the concrete
    /// model from `session.gguf()` — the [`Session`] facade has no denoise
    /// engine slot (the same reason [`ModelTask::Speaker`] binds late).
    Denoise,
    /// Speaker segmentation through pyannote `segmentation-3.0` (PyanNet).
    ///
    /// The SincNet + BiLSTM + linear + powerset-classifier forward is real
    /// but sits behind the binder's own env opt-in
    /// (`VOKRA_PYANNET_ENABLE_FORWARD=1`): the BiLSTM stack has not been
    /// byte-compared against PyTorch cuDNN yet. Routing here is precisely
    /// the honest outcome — without the opt-in the user sees the binder's
    /// own [`vokra_core::VokraError::UnsupportedOp`] naming the gate and the
    /// reason, instead of a misleading "unsupported model arch".
    Segment,
    /// F0 (pitch) extraction through RMVPE.
    ///
    /// Real forward: [`vokra_models::f0::rmvpe::RMVPE::extract_real`] runs
    /// the mel front-end + U-Net CNN + BiGRU + 360-class head + Hz decode
    /// and returns a `Result` — it has no silent all-zero skeleton branch.
    /// (The timebase-only accessor is
    /// [`frame_times`](vokra_models::f0::rmvpe::RMVPE::frame_times), which
    /// returns bare timestamps; before 2026-08-15 that body sat behind the
    /// name `extract`, so the obvious name on a loaded model handed back a
    /// fabricated track.)
    ///
    /// Its siblings FCPE and CREPE are still NOT routed here, but as of
    /// 2026-08-15 no longer because they fabricate: both now refuse a
    /// weightless or wrong-rate artifact with a named error. What they lack
    /// is the CLI wiring. See their rows in [`BOUND_ARCHES`].
    F0Rmvpe,
}

/// Optional caller-supplied hint that overrides the default task selection.
///
/// Today only the Whisper arch supports an override: `Some(TaskHint::MelFrontend)`
/// switches from the full ASR pipeline to the log-mel-only front-end. Other
/// architectures still resolve strictly by `vokra.model.arch` — passing a hint
/// that the arch does not support is a hard error (FR-EX-08: no silent
/// fallback).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TaskHint {
    /// Force the log-mel front-end task on a Whisper GGUF.
    MelFrontend,
    /// CosyVoice2 chunk-aware streaming synthetic bench (M3-09-T24 scaffold).
    ///
    /// Bypasses the GGUF load path — mirrors [`TaskHint::MelFrontend`]. The
    /// pipeline uses the M3-06 identity Mimi decoder and deterministic
    /// injected velocity / code closures, so the bench harness does not
    /// need a real safetensors checkpoint to exercise the measurement API.
    /// Selected by `--task cosyvoice2-synthetic`.
    Cosyvoice2Synthetic,
    /// Swap the CSM GGUF's embedded (T29-gated, `encode = NotImplemented`)
    /// tokenizer for the **explicit fixture byte tokenizer** — the M4-05
    /// host-only smoke path (synthesized weights + fixture tokenizer;
    /// linguistically meaningless, numerically end-to-end). Selected by
    /// `vokra-cli run --fixture-tokenizer`; never inferred (FR-EX-08).
    CsmFixtureTokenizer,
}

/// GGUF metadata key holding the model architecture (written by `vokra-convert`).
const KEY_MODEL_ARCH: &str = "vokra.model.arch";

// Architecture strings, matching vokra-convert/src/models/*.rs and vokra-capi.
const ARCH_WHISPER: &str = "whisper";
const ARCH_SILERO_VAD: &str = "silero-vad";
const ARCH_PIPER_PLUS: &str = "piper-plus-mb-istft-vits2";
const ARCH_CSM: &str = "csm";
const ARCH_MOSHI: &str = "moshi";
const ARCH_CAMPPLUS: &str = "campplus";
/// Voxtral (M3-10) — matches `vokra-convert::models::voxtral::ARCH`.
const ARCH_VOXTRAL: &str = "voxtral";
/// Kokoro-82M (M2-07) — matches `vokra_models::kokoro`'s `EXPECTED_ARCH` and
/// what `vokra-convert --model kokoro` writes.
const ARCH_KOKORO: &str = "kokoro-82m-istftnet";
/// SBV2 / Style-Bert-VITS2 v2 (Task 38) — matches
/// `vokra-convert::models::sbv2::ARCH` (`crates/vokra-convert/src/models/sbv2.rs`).
const ARCH_SBV2: &str = "sbv2";
/// MAGNeT Small 10 secs (post-audit CC-gap 2026-08-13 Wave D) — matches
/// [`vokra_models::magnet::ARCH_SMALL`]. Dispatch here is a **scaffold
/// stop-gap**: the runtime forward is loud-partial pending
/// `docs/adr/M5-magnet-masked-ar-op.md` (Status: Proposed) ratification,
/// combined with `magnet_masked_decode` / `span_masking_scheduler` op
/// landing (FR-OP-85 anchor).
const ARCH_MAGNET_SMALL: &str = "magnet_small_10secs";
/// MAGNeT Medium 30 secs (post-audit CC-gap 2026-08-13 Wave D) —
/// matches [`vokra_models::magnet::ARCH_MEDIUM`]. Same scaffold posture
/// as [`ARCH_MAGNET_SMALL`]; the loud reject at load time makes the
/// deferred state visible so no run silently produces empty output.
const ARCH_MAGNET_MEDIUM: &str = "magnet_medium_30secs";
/// MelodyFlow T24 30secs (post-audit CC-gap 2026-08-13 Wave D remaining
/// WF8) — matches [`vokra_models::melodyflow::ARCH`]. Dispatch here is a
/// **scaffold stop-gap**: the runtime forward is loud-partial pending
/// `docs/adr/M5-melodyflow-dit-sampler.md` (Status: Proposed) ratification,
/// combined with `flow_editing_inversion` / `t24_transformer` op landing
/// (FR-OP-86 anchor). The regeneration ODE integrator already exists
/// (reused `vokra_ops::flow_sampler::flow_sample` from M3-05) — only the
/// reverse-ODE editing inversion driver + the DiT block stack need to
/// land before this reject can flip to a bare session dispatch.
const ARCH_MELODYFLOW_T24_30SECS: &str = "melodyflow_t24_30secs";

// ---- Wave G (2026-08-15) — arches whose binder has a REAL forward ---------

/// FSMN-VAD (FunASR `speech_fsmn_vad_zh-cn-16k-common-pytorch`) — mirror of
/// [`vokra_models::fsmn_vad::ARCH`] and of what `vokra-cli convert --model
/// fsmn-vad` writes.
const ARCH_FSMN_VAD: &str = "fsmn-vad";
/// NSNet2 (Microsoft DNS-Challenge baseline denoiser) — mirror of
/// [`vokra_models::nsnet2::ARCH`].
const ARCH_NSNET2: &str = "nsnet2";
/// pyannote `segmentation-3.0` — mirror of
/// [`vokra_models::pyannote::EXPECTED_ARCH`].
const ARCH_PYANNOTE_SEGMENTATION: &str = "pyannote-segmentation";
/// RMVPE pitch extractor — mirror of `vokra-convert`'s `models::rmvpe::ARCH`.
/// The binder itself does not read `vokra.model.arch` (the whole `f0` family
/// keys off `vokra.f0.*` instead), so this dispatch is the only place the
/// string is matched; it is kept verbatim in lock-step with the converter.
const ARCH_RMVPE: &str = "rmvpe";

// ---- Wave I (2026-08-15) — the two distilled Whisper checkpoints ----------
//
// Both were carried in `BOUND_ARCHES` as `LoudPartialForward`, which told
// every user of a real distil-whisper / kotoba-whisper GGUF that the model
// could not run. That was false: each binder's `from_gguf` loads through
// `vokra_models::whisper::WhisperAsr` and its `transcribe` delegates to that
// shared forward, so the runtime has been complete since those loaders
// landed. Only the config-only scaffold constructors (`::new`, unreachable
// from a GGUF) hard-error. They are routed here instead, into the same
// `ModelTask::Asr` the vanilla Whisper arch uses — which is architecturally
// exact, since the sole difference is a shrunk `n_text_layer`.

/// distil-whisper (`distil-whisper/distil-large-v3.5` and family) — mirror of
/// `vokra-convert::models::distil_whisper::ARCH`.
const ARCH_DISTIL_WHISPER: &str = "distil-whisper";
/// kotoba-whisper (`kotoba-tech/kotoba-whisper-v2.0` and family) — mirror of
/// `vokra-convert::models::kotoba_whisper::ARCH`.
const ARCH_KOTOBA_WHISPER: &str = "kotoba-whisper";

/// Opens the GGUF at `path` on the CPU backend, injects the engine matching its
/// `vokra.model.arch` and returns the ready session plus its task.
#[cfg(test)]
pub(crate) fn load_session(path: &str) -> Result<(Session, ModelTask), String> {
    load_session_with_backend(path, BackendKind::Cpu, None)
}

/// As `load_session`, but runs the model's hot ops on `backend` (CPU / Metal /
/// CUDA) and lets the caller override the default arch → task mapping via
/// `hint`. Only the ASR (Whisper) path is backend-parameterised today; VAD/TTS
/// stay on the CPU. A backend that does not cover the model's op set surfaces an
/// explicit error at inference time (no silent CPU fall back, FR-EX-08); a hint
/// that the loaded arch does not support is likewise a hard error.
pub(crate) fn load_session_with_backend(
    path: &str,
    backend: BackendKind,
    hint: Option<TaskHint>,
) -> Result<(Session, ModelTask), String> {
    load_session_with_backend_and_mimi(path, backend, hint, None)
}

/// As [`load_session_with_backend`], plus an optional **standalone Mimi
/// codec side-car** (`--mimi <gguf>`, produced by `vokra-cli convert
/// --model mimi`): honored only by the Moshi arch, where it replaces the
/// synthesized codec bridge with the real kyutai weights
/// ([`vokra_models::moshi::MoshiEngine::with_mimi_gguf`]). Any other arch
/// rejects the flag loudly — never a silent drop (FR-EX-08).
pub(crate) fn load_session_with_backend_and_mimi(
    path: &str,
    backend: BackendKind,
    hint: Option<TaskHint>,
    mimi: Option<&str>,
) -> Result<(Session, ModelTask), String> {
    // M4 cc-06: open through the true-mmap loader — the session's GGUF pages
    // fault in lazily instead of a whole-file owned read (`Session::from_file`
    // buffered the entire model; on the Moshi full-7B GGUF that is ~14.3 GiB
    // held for the whole run NEXT TO the engine's own weights). Same parser,
    // byte-identical decode (vokra-mmap contract). The explicit is-a-file
    // guard mirrors the `SessionBuilder::build` path check it replaces.
    let metadata = std::fs::metadata(path).map_err(|e| e.to_string())?;
    if !metadata.is_file() {
        return Err(format!("model path `{path}` is not a regular file"));
    }
    let gguf = vokra_mmap::open_gguf(path).map_err(|e| e.to_string())?;
    let session = Session::from_gguf(gguf)
        .with_backend(backend)
        .map_err(|e| e.to_string())?;

    // Own the arch string so the immutable borrow of `session` ends before the
    // session is moved into `with_*_engine` below.
    let arch = session
        .gguf()
        .get(KEY_MODEL_ARCH)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("GGUF is missing the `{KEY_MODEL_ARCH}` metadata key"))?
        .to_owned();

    if mimi.is_some() && arch != ARCH_MOSHI {
        return Err(format!(
            "--mimi is only supported on arch `{ARCH_MOSHI}` (got `{arch}`); the \
             flag attaches the standalone Mimi codec side-car to the Moshi duplex \
             engine — dropping it silently would misrepresent the codec quality \
             (FR-EX-08)"
        ));
    }

    match arch.as_str() {
        ARCH_WHISPER => {
            // The mel-frontend task never touches the encoder / decoder — skip
            // the (potentially large-v3-sized) weight load and return a bare
            // session. The bench harness calls `whisper::mel::log_mel` directly
            // against the input WAV.
            if matches!(hint, Some(TaskHint::MelFrontend)) {
                return Ok((session, ModelTask::MelFrontend));
            }
            let asr = WhisperAsr::from_gguf(session.gguf())
                .map_err(|e| e.to_string())?
                .with_backend(backend);
            Ok((session.with_asr_engine(Arc::new(asr)), ModelTask::Asr))
        }
        // distil-whisper / kotoba-whisper (Wave I): the same `ModelTask::Asr`
        // the vanilla Whisper arm returns. Both binders wrap
        // `WhisperAsr::from_gguf` and delegate `transcribe` to it, so the
        // forward, the op set and the backend seam are literally Whisper's —
        // the only architectural difference is a shrunk `n_text_layer`, which
        // the shared config loader reads from the GGUF.
        //
        // Loading through the model-specific binder rather than `WhisperAsr`
        // directly is deliberate: each one enforces the distil invariant
        // (`n_text_layer < n_audio_layer`) and so refuses a vanilla-Whisper
        // GGUF that was mis-tagged, or a distil whose decoder tensors were
        // flattened to the encoder count (FR-EX-08 — a loud mis-label refusal
        // at load time, which is where such a refusal belongs).
        ARCH_DISTIL_WHISPER => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is only supported on arch `{ARCH_WHISPER}` \
                     (got `{ARCH_DISTIL_WHISPER}`)"
                ));
            }
            let asr = DistilWhisperAsr::from_gguf(session.gguf())
                .map_err(|e| e.to_string())?
                .with_backend(backend);
            Ok((session.with_asr_engine(Arc::new(asr)), ModelTask::Asr))
        }
        ARCH_KOTOBA_WHISPER => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is only supported on arch `{ARCH_WHISPER}` \
                     (got `{ARCH_KOTOBA_WHISPER}`)"
                ));
            }
            let asr = KotobaWhisperAsr::from_gguf(session.gguf())
                .map_err(|e| e.to_string())?
                .with_backend(backend);
            Ok((session.with_asr_engine(Arc::new(asr)), ModelTask::Asr))
        }
        ARCH_SILERO_VAD => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is only supported on arch `{ARCH_WHISPER}` \
                     (got `{ARCH_SILERO_VAD}`)"
                ));
            }
            let vad = SileroVadV5::from_gguf(session.gguf()).map_err(|e| e.to_string())?;
            Ok((session.with_vad_engine(Arc::new(vad)), ModelTask::Vad))
        }
        ARCH_PIPER_PLUS => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is only supported on arch `{ARCH_WHISPER}` \
                     (got `{ARCH_PIPER_PLUS}`)"
                ));
            }
            // `PiperPlusTts::from_gguf` consumes a `GgufFile`, but the session
            // only lends one by reference, so re-parse from the path (matches
            // vokra-capi; a shared-GGUF constructor is the same follow-up).
            let tts = PiperPlusTts::from_path(path).map_err(|e| e.to_string())?;
            Ok((session.with_tts_engine(Arc::new(tts)), ModelTask::Tts))
        }
        ARCH_KOKORO => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_KOKORO}`"
                ));
            }
            // Bare session — the `run` arm binds `KokoroTts` from the model
            // path exactly once (see `ModelTask::TtsKokoro` for why the engine
            // is not injected here). A GGUF whose tensors / metadata do not
            // bind fails loudly there (FR-EX-08).
            Ok((session, ModelTask::TtsKokoro))
        }
        ARCH_VOXTRAL => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_VOXTRAL}`"
                ));
            }
            // Bare session — the `run` arm binds `VoxtralAsr` from
            // `session.gguf()` exactly once (see `ModelTask::AsrVoxtral`
            // for why the engine is not injected here). A GGUF whose
            // tensors / hparams do not bind fails loudly there (FR-EX-08).
            Ok((session, ModelTask::AsrVoxtral))
        }
        ARCH_CAMPPLUS => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_CAMPPLUS}`"
                ));
            }
            // CAM++ speaker encoder (M0-08). The encoder binds lazily in the
            // `run` Speaker arm from `session.gguf()` (the Session facade has
            // no speaker engine slot); a GGUF whose tensors do not bind fails
            // loudly there (FR-EX-08). The selected backend is honored: CAM++
            // dispatches GEMM only, so Metal runs the whole forward on GPU
            // and an unavailable backend errors at embed time.
            Ok((session, ModelTask::Speaker))
        }
        ARCH_SBV2 => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_SBV2}`"
                ));
            }
            // Bare session — the `run` arm binds `SbV2Model` from
            // `session.gguf()` plus the `--bert-ja` / `--bert-en` side-car
            // GGUFs (see `ModelTask::Sbv2` for why the engine is not
            // injected here). A GGUF whose tensors/hparams do not bind
            // fails loudly there (FR-EX-08) — today that includes every
            // real conversion, since `convert_sbv2_file`'s tensor-name
            // mapping (Task 30) has not landed yet (see that function's
            // module doc).
            Ok((session, ModelTask::Sbv2))
        }
        ARCH_MOSHI => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_MOSHI}`"
                ));
            }
            // Moshi (M4-06, full-duplex S2S). `from_path` = strict policy +
            // real LM binding + Mimi synthesized bridge. The FR-MD-09
            // attribution banner prints below (AttributionRequired weight);
            // the AEC recipe is wired so the `--duplex --echo-sim` demo
            // runs the canceller (T26 — AEC 有効); the batch `dialog` path
            // keeps the recorded-file bypass (CSM-mirroring T20 posture).
            let mut engine =
                vokra_models::moshi::MoshiEngine::from_path(path).map_err(|e| e.to_string())?;
            // Optional real-Mimi side-car (`--mimi`): the caller asked for
            // the real codec, so a bind failure is a hard error — the
            // engine never silently keeps the synthesized bridge.
            if let Some(mimi_path) = mimi {
                engine = engine
                    .with_mimi_gguf(mimi_path)
                    .map_err(|e| format!("--mimi {mimi_path}: {e}"))?;
                eprintln!("vokra: real Mimi codec bound from {mimi_path}");
            } else if engine.mimi_is_synthesized() {
                eprintln!(
                    "vokra: NOTE Mimi codec ends are the synthesized bridge (PCM has \
                     no real audio semantics) — pass --mimi <mimi.gguf> from \
                     `vokra-cli convert --model mimi` to bind the real codec"
                );
            }
            let sample_rate = engine.mimi_config().sample_rate;
            let hop = engine
                .mimi_config()
                .frame_hop_samples()
                .map_err(|e| e.to_string())?;
            let frame_size = [128usize, 64, 32, 16, 8, 4, 2, 1]
                .into_iter()
                .find(|fs| hop % fs == 0)
                .unwrap_or(1);
            let engine = engine
                .with_aec(
                    &vokra_ops::aec::AecAttrs {
                        sample_rate,
                        frame_size,
                        filter_length: frame_size * 8,
                    },
                    sample_rate as usize, // 1 s of far-end reference
                )
                .map_err(|e| e.to_string())?
                .with_echo_path(vokra_models::csm::EchoPath::BypassRecordedInput);
            let attribution = engine.attribution().cloned();
            let engine = Arc::new(engine);
            let mut session = session
                .with_s2s_engine(engine.clone())
                .with_s2s_duplex_engine(engine);
            if let Some(info) = attribution {
                print_attribution_banner(&info);
                session = session.with_attribution(info);
            }
            Ok((session, ModelTask::S2sDuplex))
        }
        ARCH_CSM => {
            // Sesame CSM-1B (M4-05, S2S). `from_path` = strict compliance
            // policy + synthesized weight bridge until T29. `vokra-cli run`
            // is the recorded-file demo path (T20/T30), so the explicit
            // EchoPath::BypassRecordedInput opt-in applies — interactive
            // mic dialog wires an AEC front through the Rust API instead
            // (csm::aec_front rustdoc; FR-OP-60).
            let engine = CsmEngine::from_path(path).map_err(|e| e.to_string())?;
            let engine = match hint {
                Some(TaskHint::CsmFixtureTokenizer) => {
                    let vocab = engine.config().text_vocab_size;
                    engine
                        .with_tokenizer(Arc::new(
                            FixtureByteTokenizer::new(vocab).map_err(|e| e.to_string())?,
                        ))
                        .map_err(|e| e.to_string())?
                }
                None => engine,
                Some(other) => {
                    return Err(format!(
                        "task hint {other:?} is not supported on arch `{ARCH_CSM}`"
                    ));
                }
            };
            let engine = engine.with_echo_path(EchoPath::BypassRecordedInput);
            Ok((session.with_s2s_engine(Arc::new(engine)), ModelTask::S2s))
        }
        // ---- Wave G (2026-08-15) — binders with a REAL forward ------------
        ARCH_FSMN_VAD => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_FSMN_VAD}`"
                ));
            }
            // FSMN-VAD implements the SAME `VadEngine` trait Silero does, so
            // it injects into the session's VAD slot and reuses the existing
            // `run` / `bench` VAD arms verbatim — no new output shape, no new
            // code path. `from_gguf` verifies `vokra.model.arch` strictly and
            // refuses a foreign GGUF loudly (FR-EX-08).
            let vad =
                vokra_models::fsmn_vad::FsmnVadV1::from_gguf(session.gguf()).map_err(|e| {
                    let msg = e.to_string();
                    format!("arch `{ARCH_FSMN_VAD}`: {msg}")
                })?;
            Ok((session.with_vad_engine(Arc::new(vad)), ModelTask::VadFsmn))
        }
        ARCH_NSNET2 => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_NSNET2}`"
                ));
            }
            // Bare session — the `run` arm binds the concrete `Nsnet2V1` from
            // `session.gguf()` once (the `Session` facade has no denoise
            // engine slot, the `ModelTask::Speaker` precedent). A GGUF whose
            // arch tag / tensors do not bind fails loudly there (FR-EX-08).
            Ok((session, ModelTask::Denoise))
        }
        ARCH_PYANNOTE_SEGMENTATION => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch \
                     `{ARCH_PYANNOTE_SEGMENTATION}`"
                ));
            }
            // Bare session — `PyanNet::open` takes a path, and the `run` arm
            // binds it there. The forward is real but env-gated by the binder
            // itself; the `run` arm deliberately does NOT set that env var,
            // so a user without the opt-in sees pyannote's own explanation of
            // why the BiLSTM stack is still loud-pending.
            Ok((session, ModelTask::Segment))
        }
        ARCH_RMVPE => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_RMVPE}`"
                ));
            }
            // Bare session — `RMVPE::open` takes a path (the `f0` family reads
            // `vokra.f0.*`, not an already-parsed handle), so the `run` arm
            // binds it from `--model`.
            Ok((session, ModelTask::F0Rmvpe))
        }
        ARCH_MAGNET_SMALL | ARCH_MAGNET_MEDIUM => {
            // post-audit CC-gap 2026-08-13 Wave D scaffold stop-gap. The
            // runtime shell exists in `vokra-models::magnet` (config
            // deserialisation + weight catalogue + validation) but the
            // two runtime primitives that drive MAGNeT's non-autoregressive
            // masked-LM decoding loop (`magnet_masked_decode` +
            // `span_masking_scheduler` — the FR-OP-85 anchor) are
            // deferred to a follow-up wave per
            // `docs/adr/M5-magnet-masked-ar-op.md` (Status: **Proposed**).
            // Rejecting at load time — rather than injecting a bare
            // session that then errors on the first `forward` call —
            // makes the deferred state visible upfront (FR-EX-08: no
            // silent stub, no misleading task-available signal).
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{arch}` (MAGNeT \
                     runtime forward is a scaffold; ADR ratification + real weight \
                     testing required before hints are honored — see \
                     docs/adr/M5-magnet-masked-ar-op.md)"
                ));
            }
            Err(format!(
                "arch `{arch}` runtime forward is a SCAFFOLD — the MAGNeT model \
                 shell + weight catalogue exist in vokra-models::magnet but the \
                 `magnet_masked_decode` + `span_masking_scheduler` ops (FR-OP-85 \
                 anchor) are not yet landed in vokra-ops. See \
                 docs/adr/M5-magnet-masked-ar-op.md (Status: Proposed) — ADR \
                 ratification + real weight testing required before this GGUF \
                 can be executed. The GGUF loaded and validated correctly; the \
                 refusal here is the loud FR-EX-08 gate, not a converter bug."
            ))
        }
        ARCH_MELODYFLOW_T24_30SECS => {
            // post-audit CC-gap 2026-08-13 Wave D remaining WF8 scaffold
            // stop-gap. The runtime shell exists in
            // `vokra-models::melodyflow` (config deserialisation + weight
            // catalogue + validation) but the two runtime primitives that
            // drive MelodyFlow's DiT flow-matching editing pipeline
            // (`flow_editing_inversion` + `t24_transformer` — the FR-OP-86
            // anchor) are deferred to a follow-up wave per
            // `docs/adr/M5-melodyflow-dit-sampler.md` (Status:
            // **Proposed**). The regeneration ODE integrator already
            // exists (reused `vokra_ops::flow_sampler::flow_sample` from
            // M3-05 — `Schedule::Linear` + `OdeSolver::Euler` +
            // `CfgMode::DualForward` matches Le Lan et al. 2024 Algorithm
            // 1), so only the reverse-ODE editing inversion driver + the
            // DiT block stack need to land before this reject can flip
            // to a bare session dispatch. Rejecting at load time — rather
            // than injecting a bare session that then errors on the first
            // `forward` call — makes the deferred state visible upfront
            // (FR-EX-08: no silent stub, no misleading task-available
            // signal). Mirror of the sibling MAGNeT scaffold arm above.
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{arch}` (MelodyFlow \
                     runtime forward is a scaffold; ADR ratification + real weight \
                     testing required before hints are honored — see \
                     docs/adr/M5-melodyflow-dit-sampler.md)"
                ));
            }
            Err(format!(
                "arch `{arch}` runtime forward is a SCAFFOLD — the MelodyFlow model \
                 shell + weight catalogue exist in vokra-models::melodyflow but the \
                 `flow_editing_inversion` + `t24_transformer` ops (FR-OP-86 anchor) \
                 are not yet landed in vokra-ops (the M3-05 flow_sampler seam is \
                 already reused for the regeneration ODE integrator — only the \
                 reverse-ODE editing inversion driver + the DiT block stack are \
                 missing). See docs/adr/M5-melodyflow-dit-sampler.md (Status: \
                 Proposed) — ADR ratification + real weight testing required before \
                 this GGUF can be executed. The bundled 48 kHz RVQ codec decode \
                 integration is a separate owner-driven path per that ADR §D-5. The \
                 GGUF loaded and validated correctly; the refusal here is the loud \
                 FR-EX-08 gate, not a converter bug."
            ))
        }
        // Wave G (2026-08-15): before declaring the arch unknown, check the
        // bound-arch registry. `vokra-models` binds ~70 more architectures
        // than this CLI can run; telling their users "unsupported model arch"
        // was actively misleading — the runtime has a config reader, a strict
        // tensor binder and (usually) a named missing primitive for exactly
        // that model. `bound_arch_error` loads the binder where it can and
        // reports what is actually true.
        other => {
            if let Some(bound) = BOUND_ARCHES.iter().find(|b| b.arch == other) {
                if hint.is_some() {
                    return Err(format!(
                        "task hint {hint:?} is not supported on arch `{other}`: that arch \
                         has a vokra-models binder but no `vokra-cli run` task at all, so \
                         there is no task for the hint to select. Re-run without the hint \
                         to see what the binder does offer"
                    ));
                }
                return Err(bound_arch_error(bound, session.gguf()));
            }
            Err(format!(
                "unsupported model arch `{other}` (expected `{ARCH_WHISPER}` / \
                 `{ARCH_DISTIL_WHISPER}` / `{ARCH_KOTOBA_WHISPER}` / \
                 `{ARCH_SILERO_VAD}` / `{ARCH_PIPER_PLUS}` / `{ARCH_CSM}` / \
                 `{ARCH_MOSHI}` / `{ARCH_CAMPPLUS}` / `{ARCH_VOXTRAL}` / \
                 `{ARCH_KOKORO}` / `{ARCH_SBV2}` / `{ARCH_FSMN_VAD}` / \
                 `{ARCH_NSNET2}` / `{ARCH_PYANNOTE_SEGMENTATION}` / \
                 `{ARCH_RMVPE}` / `{ARCH_MAGNET_SMALL}` / `{ARCH_MAGNET_MEDIUM}` / \
                 `{ARCH_MELODYFLOW_T24_30SECS}`, or one of the {} architectures \
                 vokra-models binds without a CLI task yet)",
                BOUND_ARCHES.len()
            ))
        }
    }
}

/// The FR-MD-09 attribution banner (M4-06-T24): printed to stderr on every
/// load of an `AttributionRequired` weight so deployers see the display
/// obligation even in piped runs. There is deliberately no way to fully
/// silence it from the CLI — whether a future `--quiet` may reduce it to
/// one line is flagged to the T29 owner sign-off (license line stays).
fn print_attribution_banner(info: &vokra_core::AttributionInfo) {
    eprintln!("vokra: ATTRIBUTION ({}) {}", info.license, info.text);
}

// ---------------------------------------------------------------------------
// Wave G (2026-08-15) — bound-arch registry
//
// `crates/vokra-models/src/` holds ~88 runtime modules; the dispatch above
// runs 13 of them and loud-rejects 3 more (the MAGNeT / MelodyFlow ADR
// scaffolds, which have their own messages). Everything else fell through to
// the blanket "unsupported model arch", which is false: the runtime binds
// those models, reads their config out of the GGUF the converter wrote,
// verifies their tensor names and shapes, and (for the loud-partial majority)
// can name the exact primitive its forward is still missing plus the primary
// source to transcribe it from.
//
// This registry replaces that lie with the truth. It carries NO claim about
// *which* primitive is missing — restating that here would be a second source
// of truth that drifts silently away from the binder. It carries only facts
// this file can keep honest: the module that binds the arch, the public entry
// point a caller reaches the runtime through, the class of blocker (verified
// by reading each module's entry point), and — where the binder's loader
// accepts an already-parsed GGUF handle — a probe that actually loads it, so a
// malformed artifact surfaces the BINDER's own load error rather than a
// generic message.
// ---------------------------------------------------------------------------

/// Binds a model from an already-parsed GGUF and throws the result away.
///
/// Used only for its error: a successful probe proves the artifact satisfies
/// the binder's arch tag, tensor-name and shape contracts, and a failing one
/// hands the binder's own diagnostic straight to the user.
type BoundProbe = fn(&GgufFile) -> Result<(), VokraError>;

/// Why an architecture that `vokra-models` binds has no `vokra-cli run` task.
///
/// Each value was established by reading the module's entry point, not
/// inferred: see the per-variant note for the evidence class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundReason {
    /// The binder loads and validates, but its runtime entry point is a
    /// loud-partial: it returns `UnsupportedOp` / `NotImplemented`
    /// unconditionally, naming the missing primitive and its primary source.
    /// Verified per module by reading the entry point's body.
    LoudPartialForward,
    /// The runtime works, but what it emits is not a CLI-shaped artifact —
    /// hidden states, per-token logits, codec codes, or an output that needs
    /// a caller-supplied tokenization / phoneme sequence the GGUF does not
    /// carry. Rendering one of those as a `run` result would mean inventing a
    /// presentation the model never defined.
    NoCliShapedOutput,
    /// The forward is real and complete, but it consumes a **pair** of
    /// strictly sample-aligned input streams — an AEC mic signal plus its
    /// far-end reference — and `run` carries exactly one `--input`. Neither
    /// the GGUF nor any current flag supplies the second stream (`--compare`
    /// is the campplus speaker-verify pair and is rejected on every other
    /// task), so there is no honest input to drive the model with.
    NeedsPairedInput,
    /// The forward is real, complete and fallible — it refuses a weightless
    /// or wrong-rate artifact with a named error rather than degrading to a
    /// zero track — and its output would be CLI-shaped. Nothing about the
    /// binder blocks a `run` task; this CLI simply has not wired one for the
    /// arch yet.
    ///
    /// # A `SkeletonFallback` variant used to sit here
    ///
    /// It named the opposite hazard: a forward that runs when the GGUF
    /// carries weights but degrades to an all-zero, frame-count-only track
    /// when it does not, so that printing the track could not be told apart
    /// from a real measurement. Its only two users were the FCPE and CREPE
    /// rows, and on 2026-08-15 both binders stopped doing that — every
    /// failure is now a named error. The variant was removed with its last
    /// user rather than kept as a label no row could honestly carry.
    ///
    /// If a future binder reintroduces that shape, reintroduce the variant
    /// with it; do not file it under this one, which asserts the opposite.
    RealForwardNoCliTask,
    /// The module has no GGUF loader at all yet — its weights come from a
    /// deterministic `synthesized` fixture, so there is nothing for the CLI
    /// to bind from a converted artifact.
    NoGgufLoader,
}

impl BoundReason {
    /// One sentence naming the blocker class, for the `run` diagnostic.
    fn explain(self) -> &'static str {
        match self {
            Self::LoudPartialForward => {
                "its runtime forward is a loud-partial — calling the entry point below \
                 reports the specific missing primitive and the primary source to \
                 transcribe it from (this CLI deliberately does not restate that gap: a \
                 copy here would drift away from the binder)"
            }
            Self::NoCliShapedOutput => {
                "its output is not a CLI-shaped artifact (hidden states / logits / codec \
                 codes, or it needs a caller-supplied tokenization the GGUF does not \
                 carry), so `run` has no honest way to render it"
            }
            Self::NeedsPairedInput => {
                "its forward is real, but it consumes a PAIR of strictly sample-aligned \
                 streams (an AEC mic signal plus its far-end reference) while `run` \
                 carries exactly one `--input` — neither the GGUF nor any current flag \
                 supplies the second one, so there is nothing honest to feed it"
            }
            Self::RealForwardNoCliTask => {
                "its forward is real, complete and fallible — it refuses a weightless or \
                 wrong-rate artifact with a named error rather than degrading to a zero \
                 track — so nothing about the binder blocks execution; this CLI has \
                 simply not wired a `run` task for the arch yet, and the entry point \
                 below is callable from a library context today"
            }
            Self::NoGgufLoader => {
                "the module has no GGUF loader yet (its weights come from a deterministic \
                 `synthesized` fixture), so there is nothing to bind from this artifact"
            }
        }
    }
}

/// One architecture `vokra-models` binds but `vokra-cli run` cannot execute.
#[derive(Clone, Copy)]
struct BoundArch {
    /// The `vokra.model.arch` string the converter stamps.
    arch: &'static str,
    /// The `vokra_models` module that binds it.
    module: &'static str,
    /// The public entry point a library caller reaches the runtime through.
    entry: &'static str,
    /// The class of blocker keeping it out of `run`.
    reason: BoundReason,
    /// Load probe, when the binder's loader takes an already-parsed
    /// `&GgufFile`. `None` for path-taking loaders and for
    /// [`BoundReason::NoGgufLoader`] rows.
    probe: Option<BoundProbe>,
}

/// Every architecture with a `vokra-models` binder and no `run` task.
///
/// Kept in the order the modules were landed rather than alphabetised, so a
/// reviewer can diff a wave's additions against that wave's module list.
///
/// **Adding a binder?** Add a row here in the same commit. A binder with no
/// row falls back to the blanket "unsupported model arch", which is exactly
/// the misreport this registry exists to remove.
const BOUND_ARCHES: &[BoundArch] = &[
    // --- ASR / speech-to-text -------------------------------------------
    BoundArch {
        arch: "canary",
        module: "vokra_models::canary",
        entry: "CanaryAsr::from_gguf → CanaryAsr::transcribe",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::canary::CanaryAsr::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "canary-1b-flash",
        module: "vokra_models::canary_1b_flash",
        entry: "Canary1bFlashAsr::from_gguf → Canary1bFlashAsr::transcribe",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| {
            vokra_models::canary_1b_flash::Canary1bFlashAsr::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "canary-qwen",
        module: "vokra_models::canary_qwen",
        entry: "CanaryQwenAsr::from_gguf → CanaryQwenAsr::transcribe",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| {
            vokra_models::canary_qwen::CanaryQwenAsr::from_gguf(g).map(|_| ())
        }),
    },
    // `distil-whisper` and `kotoba-whisper` used to sit here as
    // `LoudPartialForward`. Both were false — each binder's `transcribe`
    // delegates to `vokra_models::whisper::WhisperAsr`, a real forward — so
    // the rows were removed and the two arches routed to `ModelTask::Asr`
    // instead (see `ARCH_DISTIL_WHISPER` / `ARCH_KOTOBA_WHISPER` above). Do
    // not re-add them: `bound_arch_registry_is_disjoint_from_the_routed_arches`
    // now fails on a row that shadows either arch.
    BoundArch {
        arch: "whisper-medusa-v1",
        module: "vokra_models::whisper_medusa",
        entry: "WhisperMedusa::from_gguf → WhisperMedusa::transcribe_tokens",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| {
            vokra_models::whisper_medusa::WhisperMedusa::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "moonshine",
        module: "vokra_models::moonshine",
        entry: "Moonshine::from_gguf → Moonshine::transcribe",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::moonshine::Moonshine::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "omniasr-ctc",
        module: "vokra_models::omniasr_ctc",
        entry: "OmniasrCtcAsr::from_gguf → OmniasrCtcAsr::transcribe",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| {
            vokra_models::omniasr_ctc::OmniasrCtcAsr::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "parakeet-ctc",
        module: "vokra_models::parakeet_ctc",
        entry: "ParakeetCtcAsr::from_gguf → ParakeetCtcAsr::transcribe",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| {
            vokra_models::parakeet_ctc::ParakeetCtcAsr::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "parakeet-tdt-1_1b",
        module: "vokra_models::parakeet_tdt_1_1b",
        entry: "ParakeetTdt11b::from_gguf → ParakeetTdt11b::transcribe",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| {
            vokra_models::parakeet_tdt_1_1b::ParakeetTdt11b::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "parakeet-tdt",
        module: "vokra_models::parakeet",
        entry: "ParakeetAsr::transcribe",
        reason: BoundReason::NoGgufLoader,
        probe: None,
    },
    BoundArch {
        arch: "sensevoicesmall",
        module: "vokra_models::sensevoicesmall_runtime",
        entry: "SenseVoiceSmall::from_gguf → SenseVoiceSmall::transcribe",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| {
            vokra_models::sensevoicesmall_runtime::SenseVoiceSmall::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "firered_asr_aed_l",
        module: "vokra_models::firered_asr_aed",
        entry: "FireredAsrAed::from_gguf → FireredAsrAed::transcribe_tokens",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| {
            vokra_models::firered_asr_aed::FireredAsrAed::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "sber_gigaam_v3",
        module: "vokra_models::gigaam",
        entry: "Gigaam::from_gguf → Gigaam::transcribe",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::gigaam::Gigaam::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "gigaam_multilingual",
        module: "vokra_models::gigaam",
        entry: "Gigaam::from_gguf → Gigaam::transcribe",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::gigaam::Gigaam::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "kyutai-stt",
        module: "vokra_models::kyutai_stt",
        entry: "KyutaiSttAsr::from_path → KyutaiSttAsr::transcribe",
        reason: BoundReason::LoudPartialForward,
        probe: None,
    },
    BoundArch {
        arch: "mt3",
        module: "vokra_models::mt3",
        entry: "Mt3::from_gguf → Mt3::transcribe",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::mt3::Mt3::from_gguf(g).map(|_| ())),
    },
    // --- VAD / KWS / turn-taking ----------------------------------------
    BoundArch {
        arch: "firered_vad",
        module: "vokra_models::firered_vad",
        entry: "FireredVad::from_gguf → FireredVad::speech_probabilities",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::firered_vad::FireredVad::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "ten_vad",
        module: "vokra_models::ten_vad",
        entry: "TenVad::from_gguf → TenVad::frame_probability",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::ten_vad::TenVad::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "smart_turn",
        module: "vokra_models::smart_turn",
        entry: "SmartTurn::from_gguf → SmartTurn::predict_endpoint",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::smart_turn::SmartTurn::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "openwakeword_op",
        module: "vokra_models::kws::openwakeword",
        entry: "OpenwakewordSession::from_gguf → OpenwakewordSession::push_pcm16k",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| {
            vokra_models::kws::openwakeword::OpenwakewordSession::from_gguf(g).map(|_| ())
        }),
    },
    // --- TTS -------------------------------------------------------------
    BoundArch {
        arch: "styletts2",
        module: "vokra_models::styletts2",
        entry: "StyleTts2Tts::from_gguf → StyleTts2Tts::synthesize",
        reason: BoundReason::LoudPartialForward,
        probe: None,
    },
    BoundArch {
        arch: "cosyvoice2",
        module: "vokra_models::cosyvoice2",
        entry: "CosyVoice2Tts::from_path → CosyVoice2Tts::synthesize_pcm_from_mel",
        reason: BoundReason::LoudPartialForward,
        probe: None,
    },
    BoundArch {
        arch: "cosyvoice3",
        module: "vokra_models::cosyvoice3",
        entry: "CosyVoice3Tts::synthesize",
        reason: BoundReason::NoGgufLoader,
        probe: None,
    },
    BoundArch {
        arch: "chatterbox",
        module: "vokra_models::chatterbox",
        entry: "ChatterboxTts::synthesize",
        reason: BoundReason::NoGgufLoader,
        probe: None,
    },
    BoundArch {
        arch: "chatterbox_nano",
        module: "vokra_models::chatterbox_nano",
        entry: "ChatterboxNanoTts::synthesize",
        reason: BoundReason::NoGgufLoader,
        probe: None,
    },
    BoundArch {
        arch: "chatterbox_turbo",
        module: "vokra_models::chatterbox_turbo",
        entry: "ChatterboxTurboTts::synthesize",
        reason: BoundReason::NoGgufLoader,
        probe: None,
    },
    BoundArch {
        arch: "dia",
        module: "vokra_models::dia",
        entry: "DiaTts::synthesize",
        reason: BoundReason::NoGgufLoader,
        probe: None,
    },
    BoundArch {
        arch: "irodori-tts",
        module: "vokra_models::irodori",
        entry: "IrodoriTts::synthesize",
        reason: BoundReason::NoGgufLoader,
        probe: None,
    },
    BoundArch {
        arch: "qwen3_tts",
        module: "vokra_models::qwen3_tts",
        entry: "Qwen3TtsTts::synthesize",
        reason: BoundReason::NoGgufLoader,
        probe: None,
    },
    BoundArch {
        arch: "vibevoice",
        module: "vokra_models::vibevoice",
        entry: "VibeVoiceTts::synthesize",
        reason: BoundReason::NoGgufLoader,
        probe: None,
    },
    BoundArch {
        arch: "vits-ja",
        module: "vokra_models::vits_ja",
        entry: "VitsJaTts::synthesize",
        reason: BoundReason::NoGgufLoader,
        probe: None,
    },
    BoundArch {
        arch: "voxcpm2",
        module: "vokra_models::voxcpm2",
        entry: "VoxCpm2Tts::synthesize",
        reason: BoundReason::NoGgufLoader,
        probe: None,
    },
    BoundArch {
        arch: "zonos",
        module: "vokra_models::zonos",
        entry: "ZonosTts::synthesize",
        reason: BoundReason::NoGgufLoader,
        probe: None,
    },
    BoundArch {
        arch: "diffsinger",
        module: "vokra_models::diffsinger",
        entry: "DiffSinger::from_gguf → DiffSinger::synthesize_mel",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::diffsinger::DiffSinger::from_gguf(g).map(|_| ())),
    },
    // --- Speech-to-speech -------------------------------------------------
    BoundArch {
        arch: "llama_omni2",
        module: "vokra_models::llama_omni2",
        entry: "LlamaOmni2::from_path → LlamaOmni2::converse",
        reason: BoundReason::LoudPartialForward,
        probe: None,
    },
    BoundArch {
        arch: "voila",
        module: "vokra_models::voila",
        entry: "Voila::from_gguf → Voila::converse",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::voila::Voila::from_gguf(g).map(|_| ())),
    },
    // --- Music / audio generation ----------------------------------------
    BoundArch {
        arch: "musicgen",
        module: "vokra_models::musicgen",
        entry: "MusicGen::from_gguf → MusicGen::generate",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::musicgen::MusicGen::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "audiogen",
        module: "vokra_models::audiogen",
        entry: "AudioGen::from_gguf → AudioGen::generate",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::audiogen::AudioGen::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "audioldm2",
        module: "vokra_models::audioldm2",
        entry: "AudioLdm2::from_gguf → AudioLdm2::generate",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::audioldm2::AudioLdm2::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "jasco_400m_chords_drums",
        module: "vokra_models::jasco",
        entry: "Jasco::from_gguf → Jasco::generate",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::jasco::Jasco::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "beat-this",
        module: "vokra_models::beat_this",
        entry: "BeatThis::from_gguf → BeatThis::analyze",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::beat_this::BeatThis::from_gguf(g).map(|_| ())),
    },
    // --- Source separation / enhancement / super-resolution ---------------
    BoundArch {
        arch: "sepformer",
        module: "vokra_models::sepformer",
        entry: "SepFormer::from_gguf → SepFormer::separate",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::sepformer::SepFormer::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "conv_tasnet",
        module: "vokra_models::conv_tasnet",
        entry: "ConvTasnet::from_gguf → ConvTasnet::separate",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::conv_tasnet::ConvTasnet::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "demucs",
        module: "vokra_models::demucs",
        entry: "Demucs::from_gguf → Demucs::separate",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::demucs::Demucs::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "gtcrn",
        module: "vokra_models::gtcrn",
        entry: "Gtcrn::from_gguf → Gtcrn::denoise",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::gtcrn::Gtcrn::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "facebook_denoiser",
        module: "vokra_models::facebook_denoiser",
        entry: "FbDenoiser::from_gguf → FbDenoiser::denoise",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| {
            vokra_models::facebook_denoiser::FbDenoiser::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "storm",
        module: "vokra_models::storm",
        entry: "Storm::from_gguf → Storm::enhance",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::storm::Storm::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "audiosr",
        module: "vokra_models::audiosr",
        entry: "AudioSr::from_gguf → AudioSr::super_resolve",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::audiosr::AudioSr::from_gguf(g).map(|_| ())),
    },
    // --- Diarization / speaker -------------------------------------------
    BoundArch {
        arch: "sortformer",
        module: "vokra_models::sortformer_diar_4spk_v1",
        entry: "SortformerDiar::from_gguf → SortformerDiar::diarize",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| {
            vokra_models::sortformer_diar_4spk_v1::SortformerDiar::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "speaker_3d",
        module: "vokra_models::speaker_3d_eres2net",
        entry: "Speaker3dEres2Net::from_gguf → Speaker3dEres2Net::encode",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| {
            vokra_models::speaker_3d_eres2net::Speaker3dEres2Net::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "redimnet",
        module: "vokra_models::redimnet",
        entry: "ReDimNet::from_gguf → ReDimNet::encode",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::redimnet::ReDimNet::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "wavlm_sv",
        module: "vokra_models::wavlm",
        entry: "WavLmSv::from_gguf → WavLmSv::encode",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::wavlm::WavLmSv::from_gguf(g).map(|_| ())),
    },
    // --- SSL / representation encoders and classifiers --------------------
    //
    // TWO blockers apply to the encoders below, and the `reason` names the one
    // that fires FIRST: their forwards are loud-partials today, so that is
    // what a caller hits. Even once those land, an SSL encoder emits `[T, D]`
    // hidden states — not a CLI-shaped artifact — so these rows would move to
    // `NoCliShapedOutput` rather than gain a `run` task. The classifier heads
    // (`emotion2vec`, `panns`, `maest::tag`) are the exception: a label +
    // score list IS printable, so those become candidates for a real `run`
    // arm the day their forwards land.
    BoundArch {
        arch: "atst",
        module: "vokra_models::atst",
        entry: "Atst::from_gguf → Atst::encode / Atst::embed",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::atst::Atst::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "eat",
        module: "vokra_models::eat",
        entry: "Eat::from_gguf → Eat::encode / Eat::embed_utterance",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::eat::Eat::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "m2d",
        module: "vokra_models::m2d",
        entry: "M2d::from_gguf → M2d::encode / M2d::embed",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::m2d::M2d::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "maest",
        module: "vokra_models::maest",
        entry: "Maest::from_gguf → Maest::encode / Maest::tag",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::maest::Maest::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "w2v-bert-2",
        module: "vokra_models::w2v_bert2",
        entry: "W2vBert2::from_gguf → W2vBert2::encode",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::w2v_bert2::W2vBert2::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "clap",
        module: "vokra_models::clap",
        entry: "Clap::from_gguf → Clap::encode_audio",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::clap::Clap::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "emotion2vec",
        module: "vokra_models::emotion2vec",
        entry: "Emotion2Vec::from_gguf → Emotion2Vec::classify",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| {
            vokra_models::emotion2vec::Emotion2Vec::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "panns",
        module: "vokra_models::panns",
        entry: "Panns::from_gguf → Panns::classify",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::panns::Panns::from_gguf(g).map(|_| ())),
    },
    // --- Quality metrics --------------------------------------------------
    BoundArch {
        arch: "dnsmos",
        module: "vokra_models::dnsmos_p808_p835",
        entry: "Dnsmos::from_gguf → Dnsmos::score_all",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| {
            vokra_models::dnsmos_p808_p835::Dnsmos::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "nisqa_v2_weight",
        module: "vokra_models::nisqa",
        entry: "Nisqa::from_gguf → Nisqa::score",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::nisqa::Nisqa::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "utmosv2",
        module: "vokra_models::utmosv2",
        entry: "Utmosv2::from_gguf → Utmosv2::predict_mos",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::utmosv2::Utmosv2::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "torchaudio_squim",
        module: "vokra_models::squim",
        entry: "Squim::from_gguf → Squim::estimate_objective / estimate_subjective",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::squim::Squim::from_gguf(g).map(|_| ())),
    },
    // --- Vocoders / codecs -----------------------------------------------
    BoundArch {
        arch: "bigvgan",
        module: "vokra_models::bigvgan",
        entry: "BigVGan::from_gguf → BigVGan::decode",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::bigvgan::BigVGan::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "vocos",
        module: "vokra_models::vocos",
        entry: "Vocos::from_gguf → Vocos::decode",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::vocos::Vocos::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "hifigan_vocoder",
        module: "vokra_models::hifigan",
        entry: "HiFiGan::from_gguf → HiFiGan::decode",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::hifigan::HiFiGan::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "speecht5_hifigan",
        module: "vokra_models::hifigan",
        entry: "HiFiGan::from_gguf → HiFiGan::decode",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::hifigan::HiFiGan::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "snac",
        module: "vokra_models::snac",
        entry: "Snac::from_gguf → Snac::encode / Snac::decode",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::snac::Snac::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "mimi",
        module: "vokra_models::mimi",
        entry: "MimiEncoder::encode_all / MimiNeuralDecoder::decode_all",
        reason: BoundReason::NoCliShapedOutput,
        probe: None,
    },
    // --- Text / alignment side-cars ---------------------------------------
    BoundArch {
        arch: "ct_punc",
        module: "vokra_models::ct_punc",
        entry: "CtPunc::from_gguf → CtPunc::restore",
        reason: BoundReason::NoCliShapedOutput,
        probe: Some(|g: &GgufFile| vokra_models::ct_punc::CtPunc::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "wetextprocessing",
        module: "vokra_models::wetextprocessing",
        entry: "WeTextProcessing::from_gguf → WeTextProcessing::normalize",
        reason: BoundReason::NoCliShapedOutput,
        probe: Some(|g: &GgufFile| {
            vokra_models::wetextprocessing::WeTextProcessing::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "charsiu",
        module: "vokra_models::align::charsiu",
        entry: "Charsiu::from_gguf → Charsiu::align",
        reason: BoundReason::NoCliShapedOutput,
        probe: None,
    },
    // --- F0 siblings that are NOT routed to `ModelTask::F0Rmvpe` ----------
    //
    // Both stopped being skeleton-fallbacks on 2026-08-15, in the same change
    // that gave RMVPE its `extract` / `extract_real` / `frame_times` shape.
    // Each now refuses loudly instead of degrading to a zero track:
    // `ModelLoad` when no weights were bound, `InvalidArgument` when the
    // caller's sample rate is not the one the checkpoint is defined at, and —
    // for FCPE — the STFT front-end's own error propagated verbatim instead
    // of swallowed by an `Err(_) =>` arm. The timebase-only half moved to a
    // `frame_times` accessor that returns bare `f32` seconds, so nothing it
    // returns can be read as a pitch estimate.
    //
    // Both rows stay because no `run` task is wired for either arch — but the
    // reason they carry had to change with the binders, and the entries they
    // name had to stop pointing at a method whose contract no longer holds.
    BoundArch {
        arch: "fcpe",
        module: "vokra_models::f0::fcpe",
        entry: "FCPE::from_gguf → FCPE::extract",
        reason: BoundReason::RealForwardNoCliTask,
        probe: None,
    },
    BoundArch {
        arch: "crepe",
        module: "vokra_models::f0::crepe",
        entry: "CREPE::from_gguf → CREPE::extract",
        reason: BoundReason::RealForwardNoCliTask,
        probe: None,
    },
    // --- Wave H (2026-08-15) — five binders this registry had missed -------
    //
    // Each of these landed a `vokra-models` binder without the row the doc
    // comment above requires ("Adding a binder? Add a row here in the same
    // commit") — three of them in the very commit that wrote that rule. Their
    // users got the blanket "unsupported model arch", the exact misreport this
    // table exists to remove. Grouped as one wave rather than filed into the
    // family sections above, per this table's stated landing-order rule, so
    // the omission and its fix read as one diff.
    //
    // `scripts/check-bound-arch-coverage.sh` is what keeps the next one from
    // going quiet: it walks every `pub const ARCH…: &str` under
    // `crates/vokra-models/src/` and fails unless the arch is either routed by
    // the dispatch or present here. The in-crate test below can only inspect
    // rows that exist, which is why it never noticed these five.
    BoundArch {
        arch: "chattts",
        module: "vokra_models::chattts",
        entry: "ChatTts::from_gguf → ChatTts::synthesize",
        reason: BoundReason::LoudPartialForward,
        // The probe is the un-gated `&GgufFile` binder, which reads only the
        // tensor MANIFEST (`ChatTtsWeights` holds `(name, dims)` pairs — no
        // payload) and drops it. ChatTTS weights are CC-BY-NC-4.0 and the
        // M2-13 research-flag gate lives on the `from_path` /
        // `from_gguf_with_policy` routes a real consumer takes, so probing
        // here neither loads weights for use nor steps around that gate.
        probe: Some(|g: &GgufFile| vokra_models::chattts::ChatTts::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "deepfake_detection",
        module: "vokra_models::deepfake_detection",
        entry: "DeepfakeDetection::from_gguf → DeepfakeDetection::score",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| {
            vokra_models::deepfake_detection::DeepfakeDetection::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "lang_id_ecapa",
        module: "vokra_models::lang_id",
        entry: "LangIdEcapa::from_gguf → LangIdEcapa::identify",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::lang_id::LangIdEcapa::from_gguf(g).map(|_| ())),
    },
    // Both AEC binders take a paired (mic, far-end) input that `run` cannot
    // supply, but `reason` names the blocker that fires FIRST for the library
    // caller the row points at — and the two differ. DTLN-AEC's `process`
    // returns `UnsupportedOp` unconditionally (the generic LSTM primitive is
    // absent from `vokra-ops`), so a caller holding both streams still hits a
    // loud-partial. NKF-AEC's forward is a real native re-implementation with
    // an upstream parity harness, so for it the paired input IS the only
    // blocker. Reporting both as loud-partials would slander a working model.
    BoundArch {
        arch: "dtln_aec",
        module: "vokra_models::aec::dtln_aec",
        entry: "DtlnAec::from_gguf → DtlnAec::process",
        reason: BoundReason::LoudPartialForward,
        probe: Some(|g: &GgufFile| vokra_models::aec::dtln_aec::DtlnAec::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "nkf_aec",
        module: "vokra_models::aec::nkf_aec",
        entry: "NkfAec::from_gguf → AecEngine::open_stream → NkfAecStream::push_paired",
        reason: BoundReason::NeedsPairedInput,
        probe: Some(|g: &GgufFile| vokra_models::aec::nkf_aec::NkfAec::from_gguf(g).map(|_| ())),
    },
];

/// Builds the `run` diagnostic for an arch in [`BOUND_ARCHES`].
///
/// When the row carries a probe this actually **loads the binder** from the
/// session's own GGUF first, so a malformed or foreign artifact reports the
/// binder's own error (strict `vokra.model.arch` mismatch, a missing tensor, a
/// refused weight license) instead of a generic message. A successful probe is
/// reported too: it is the evidence behind the claim that the model is bound.
fn bound_arch_error(bound: &BoundArch, gguf: &GgufFile) -> String {
    let load_line = match bound.probe {
        Some(probe) => match probe(gguf) {
            Ok(()) => "This GGUF LOADED and validated against that binder (arch tag, tensor \
                       names, shapes, weight-license gate all passed)."
                .to_owned(),
            Err(e) => {
                let msg = e.to_string();
                format!("Loading this GGUF through that binder FAILED — the binder reports: {msg}")
            }
        },
        None => "This CLI did not probe-load it: that binder's loader does not take an \
                 already-parsed GGUF handle, so the load has to happen through the library \
                 entry point below."
            .to_owned(),
    };
    let arch = bound.arch;
    let module = bound.module;
    let entry = bound.entry;
    let reason = bound.reason.explain();
    format!(
        "arch `{arch}` is BOUND by this build — `{module}` has a runtime binder for it, \
         not an unknown architecture. {load_line} What it does NOT have is a `vokra-cli \
         run` task, because {reason}. Reach the runtime through the library API — module \
         `{module}`, entry point `{entry}`. (FR-EX-08: refusing here rather than routing \
         the model through a different arch's task or printing a fabricated result.)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Path to the committed both-rate Silero VAD fixture GGUF (M0-05 asset).
    fn silero_fixture() -> String {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/parity/silero_vad/silero-vad-v5.gguf")
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn load_session_detects_silero_vad_and_injects_engine() {
        let (session, task) = load_session(&silero_fixture()).expect("silero session builds");
        assert_eq!(task, ModelTask::Vad);
        // The VAD engine was injected: opening a stream succeeds.
        assert!(session.open_vad_stream().is_ok());
    }

    #[test]
    fn load_session_rejects_missing_file() {
        assert!(load_session("/no/such/vokra-cli-model.gguf").is_err());
    }

    /// A `campplus` arch GGUF dispatches to [`ModelTask::Speaker`] — the
    /// encoder itself binds later in the `run` Speaker arm, so a
    /// metadata-only fixture is enough here (mirrors the unknown-arch test).
    #[test]
    fn load_session_detects_campplus_as_speaker_task() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "campplus");
        let bytes = b.to_bytes().expect("serialize gguf");
        let mut path = std::env::temp_dir();
        path.push(format!(
            "vokra-cli-campplus-arch-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let result = load_session(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        let (_session, task) = result.expect("campplus session builds (bare)");
        assert_eq!(task, ModelTask::Speaker);
    }

    /// Task hints are rejected on the campplus arch (FR-EX-08 — no silent
    /// hint drop).
    #[test]
    fn load_session_rejects_hint_on_campplus() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "campplus");
        let bytes = b.to_bytes().expect("serialize gguf");
        let mut path = std::env::temp_dir();
        path.push(format!(
            "vokra-cli-campplus-hint-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let result = load_session_with_backend(
            path.to_str().unwrap(),
            BackendKind::Cpu,
            Some(TaskHint::MelFrontend),
        );
        let _ = std::fs::remove_file(&path);
        let err = result.expect_err("hint on campplus is rejected");
        assert!(
            err.contains("not supported on arch `campplus`"),
            "got: {err}"
        );
    }

    /// `--mimi` is a Moshi-only flag: every other arch rejects it loudly
    /// (FR-EX-08 — silently dropping the side-car would misrepresent the
    /// codec quality of the run).
    #[test]
    fn load_session_rejects_mimi_sidecar_on_non_moshi_arch() {
        let err = load_session_with_backend_and_mimi(
            &silero_fixture(),
            BackendKind::Cpu,
            None,
            Some("/no/such/mimi.gguf"),
        )
        .expect_err("--mimi on silero-vad is rejected");
        assert!(
            err.contains("--mimi is only supported on arch `moshi`"),
            "got: {err}"
        );
    }

    /// A `voxtral` arch GGUF dispatches to [`ModelTask::AsrVoxtral`] with a
    /// bare session — the concrete engine binds in the `run` arm (P2 cc-10),
    /// so a metadata-only fixture is enough here (campplus precedent).
    #[test]
    fn load_session_detects_voxtral_as_asr_voxtral_task() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "voxtral");
        let bytes = b.to_bytes().expect("serialize gguf");
        let mut path = std::env::temp_dir();
        path.push(format!(
            "vokra-cli-voxtral-arch-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let result = load_session(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        let (_session, task) = result.expect("voxtral session builds (bare)");
        assert_eq!(task, ModelTask::AsrVoxtral);
    }

    /// Task hints are rejected on the voxtral arch (FR-EX-08 — no silent
    /// hint drop).
    #[test]
    fn load_session_rejects_hint_on_voxtral() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "voxtral");
        let bytes = b.to_bytes().expect("serialize gguf");
        let mut path = std::env::temp_dir();
        path.push(format!(
            "vokra-cli-voxtral-hint-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let result = load_session_with_backend(
            path.to_str().unwrap(),
            BackendKind::Cpu,
            Some(TaskHint::MelFrontend),
        );
        let _ = std::fs::remove_file(&path);
        let err = result.expect_err("hint on voxtral is rejected");
        assert!(
            err.contains("not supported on arch `voxtral`"),
            "got: {err}"
        );
    }

    /// An `sbv2` arch GGUF dispatches to [`ModelTask::Sbv2`] with a bare
    /// session — the concrete engine binds in the `run` arm (Task 38,
    /// mirroring the campplus / voxtral precedent), so a metadata-only
    /// fixture is enough here.
    #[test]
    fn load_session_detects_sbv2_as_sbv2_task() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "sbv2");
        let bytes = b.to_bytes().expect("serialize gguf");
        let mut path = std::env::temp_dir();
        path.push(format!("vokra-cli-sbv2-arch-{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let result = load_session(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        let (_session, task) = result.expect("sbv2 session builds (bare)");
        assert_eq!(task, ModelTask::Sbv2);
    }

    /// Task hints are rejected on the sbv2 arch (FR-EX-08 — no silent hint
    /// drop).
    #[test]
    fn load_session_rejects_hint_on_sbv2() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "sbv2");
        let bytes = b.to_bytes().expect("serialize gguf");
        let mut path = std::env::temp_dir();
        path.push(format!("vokra-cli-sbv2-hint-{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let result = load_session_with_backend(
            path.to_str().unwrap(),
            BackendKind::Cpu,
            Some(TaskHint::MelFrontend),
        );
        let _ = std::fs::remove_file(&path);
        let err = result.expect_err("hint on sbv2 is rejected");
        assert!(err.contains("not supported on arch `sbv2`"), "got: {err}");
    }

    /// `--mimi` is a Moshi-only flag: sbv2's own multi-GGUF side-cars are
    /// `--bert-ja` / `--bert-en`, handled entirely in `run.rs`, not through
    /// this function's `mimi` parameter.
    #[test]
    fn load_session_rejects_mimi_sidecar_on_sbv2_arch() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "sbv2");
        let bytes = b.to_bytes().expect("serialize gguf");
        let mut path = std::env::temp_dir();
        path.push(format!(
            "vokra-cli-sbv2-mimi-reject-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let result = load_session_with_backend_and_mimi(
            path.to_str().unwrap(),
            BackendKind::Cpu,
            None,
            Some("/no/such/mimi.gguf"),
        );
        let _ = std::fs::remove_file(&path);
        let err = result.expect_err("--mimi on sbv2 is rejected");
        assert!(
            err.contains("--mimi is only supported on arch `moshi`"),
            "got: {err}"
        );
    }

    /// A `magnet_small_10secs` arch GGUF is loud-rejected at load time
    /// with a scaffold message naming the ADR — the runtime forward is
    /// deferred, so the CLI never pretends a working task exists
    /// (FR-EX-08). Mirror of the RMVPE / DNSMOS loud-partial posture.
    #[test]
    fn load_session_rejects_magnet_small_arch_with_scaffold_message() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", ARCH_MAGNET_SMALL);
        let bytes = b.to_bytes().expect("serialize gguf");
        let mut path = std::env::temp_dir();
        path.push(format!(
            "vokra-cli-magnet-small-arch-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let result = load_session(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        let err = result.expect_err("magnet small arch must be loud-rejected as scaffold");
        assert!(
            err.contains("SCAFFOLD"),
            "message must self-identify as scaffold: {err}"
        );
        assert!(
            err.contains("docs/adr/M5-magnet-masked-ar-op.md"),
            "message must name the ADR to ratify: {err}"
        );
        assert!(
            err.contains("magnet_masked_decode") && err.contains("span_masking_scheduler"),
            "message must name the deferred ops: {err}"
        );
        assert!(err.contains("FR-OP-85"), "message must cite anchor: {err}");
    }

    /// Same loud-reject contract for `magnet_medium_30secs` — the two
    /// variants share the FR-EX-08 gate.
    #[test]
    fn load_session_rejects_magnet_medium_arch_with_scaffold_message() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", ARCH_MAGNET_MEDIUM);
        let bytes = b.to_bytes().expect("serialize gguf");
        let mut path = std::env::temp_dir();
        path.push(format!(
            "vokra-cli-magnet-medium-arch-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let result = load_session(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        let err = result.expect_err("magnet medium arch must be loud-rejected as scaffold");
        assert!(err.contains("SCAFFOLD"), "must self-identify: {err}");
        assert!(
            err.contains(ARCH_MAGNET_MEDIUM),
            "must name the arch: {err}"
        );
    }

    /// A `melodyflow_t24_30secs` arch GGUF is loud-rejected at load
    /// time with a scaffold message naming the ADR — the runtime
    /// forward is deferred, so the CLI never pretends a working task
    /// exists (FR-EX-08). Mirror of the sibling MAGNeT loud-partial
    /// posture.
    #[test]
    fn load_session_rejects_melodyflow_arch_with_scaffold_message() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", ARCH_MELODYFLOW_T24_30SECS);
        let bytes = b.to_bytes().expect("serialize gguf");
        let mut path = std::env::temp_dir();
        path.push(format!(
            "vokra-cli-melodyflow-arch-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let result = load_session(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        let err = result.expect_err("melodyflow arch must be loud-rejected as scaffold");
        assert!(
            err.contains("SCAFFOLD"),
            "message must self-identify as scaffold: {err}"
        );
        assert!(
            err.contains("docs/adr/M5-melodyflow-dit-sampler.md"),
            "message must name the ADR to ratify: {err}"
        );
        assert!(
            err.contains("flow_editing_inversion") && err.contains("t24_transformer"),
            "message must name the deferred ops: {err}"
        );
        assert!(err.contains("FR-OP-86"), "message must cite anchor: {err}");
        assert!(
            err.contains("flow_sampler"),
            "message must name the reused M3-05 seam so an owner knows only the \
             two new ops are missing: {err}"
        );
    }

    /// Task hints are rejected on the melodyflow arch too — same
    /// FR-EX-08 rule as every other arch that returns a bare / rejected
    /// session. Mirror of the sibling magnet arch hint reject.
    #[test]
    fn load_session_rejects_hint_on_melodyflow_arch() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", ARCH_MELODYFLOW_T24_30SECS);
        let bytes = b.to_bytes().expect("serialize gguf");
        let mut path = std::env::temp_dir();
        path.push(format!(
            "vokra-cli-melodyflow-hint-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let result = load_session_with_backend(
            path.to_str().unwrap(),
            BackendKind::Cpu,
            Some(TaskHint::MelFrontend),
        );
        let _ = std::fs::remove_file(&path);
        let err = result.expect_err("hint on melodyflow arch must be rejected");
        assert!(
            err.contains("MelodyFlow runtime forward is a scaffold"),
            "hint reject must cite the same scaffold rationale: {err}"
        );
    }

    /// Task hints are rejected on the magnet arch too — same FR-EX-08
    /// rule as every other arch that returns a bare / rejected session.
    #[test]
    fn load_session_rejects_hint_on_magnet_arch() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", ARCH_MAGNET_SMALL);
        let bytes = b.to_bytes().expect("serialize gguf");
        let mut path = std::env::temp_dir();
        path.push(format!("vokra-cli-magnet-hint-{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let result = load_session_with_backend(
            path.to_str().unwrap(),
            BackendKind::Cpu,
            Some(TaskHint::MelFrontend),
        );
        let _ = std::fs::remove_file(&path);
        let err = result.expect_err("hint on magnet arch must be rejected");
        assert!(
            err.contains("MAGNeT runtime forward is a scaffold"),
            "hint reject must cite the same scaffold rationale: {err}"
        );
    }

    // ---- Wave G (2026-08-15) — newly routed arches ----------------------

    /// Writes an arch-only GGUF to a unique temp path, runs `f` against it and
    /// removes the file. Shared by the Wave G tests below (the older tests
    /// inline the same steps; they are left byte-identical on purpose).
    fn with_arch_only_gguf<T>(arch: &str, tag: &str, f: impl FnOnce(&str) -> T) -> T {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", arch);
        let bytes = b.to_bytes().expect("serialize gguf");
        let mut path = std::env::temp_dir();
        path.push(format!("vokra-cli-{tag}-{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let out = f(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        out
    }

    /// An `fsmn-vad` GGUF is a REAL VAD: the dispatch binds
    /// [`vokra_models::fsmn_vad::FsmnVadV1`] into the session's VAD slot. A
    /// metadata-only fixture cannot bind (no config chunk, no tensors), so
    /// what must be asserted here is that the failure comes from the FSMN
    /// binder — never the old "unsupported model arch".
    #[test]
    fn load_session_routes_fsmn_vad_to_its_own_binder() {
        let err = with_arch_only_gguf("fsmn-vad", "fsmn-arch", |p| {
            let Err(e) = load_session(p) else {
                panic!("metadata-only fsmn-vad cannot bind");
            };
            e
        });
        assert!(
            err.contains("arch `fsmn-vad`"),
            "error must name the arch it routed to: {err}"
        );
        assert!(
            !err.contains("unsupported model arch"),
            "fsmn-vad is bound and routed — it must never read as unknown: {err}"
        );
    }

    /// An `nsnet2` GGUF dispatches to [`ModelTask::Denoise`] with a bare
    /// session — the concrete `Nsnet2V1` binds in the `run` arm (the
    /// campplus / voxtral precedent), so a metadata-only fixture is enough.
    #[test]
    fn load_session_detects_nsnet2_as_denoise_task() {
        let (_session, task) = with_arch_only_gguf("nsnet2", "nsnet2-arch", |p| {
            load_session(p).expect("nsnet2 session builds (bare)")
        });
        assert_eq!(task, ModelTask::Denoise);
    }

    /// A `pyannote-segmentation` GGUF dispatches to [`ModelTask::Segment`].
    #[test]
    fn load_session_detects_pyannote_as_segment_task() {
        let (_session, task) = with_arch_only_gguf("pyannote-segmentation", "pyannote-arch", |p| {
            load_session(p).expect("pyannote session builds (bare)")
        });
        assert_eq!(task, ModelTask::Segment);
    }

    /// An `rmvpe` GGUF dispatches to [`ModelTask::F0Rmvpe`].
    #[test]
    fn load_session_detects_rmvpe_as_f0_task() {
        let (_session, task) = with_arch_only_gguf("rmvpe", "rmvpe-arch", |p| {
            load_session(p).expect("rmvpe session builds (bare)")
        });
        assert_eq!(task, ModelTask::F0Rmvpe);
    }

    /// Task hints stay rejected on the newly routed arches (FR-EX-08 — no
    /// silent hint drop), same rule every other arch follows.
    #[test]
    fn load_session_rejects_hint_on_newly_routed_arches() {
        for (arch, tag) in [
            ("nsnet2", "nsnet2-hint"),
            ("pyannote-segmentation", "pyannote-hint"),
            ("rmvpe", "rmvpe-hint"),
            ("fsmn-vad", "fsmn-hint"),
        ] {
            let err = with_arch_only_gguf(arch, tag, |p| {
                // let-else rather than `.expect_err()`: `Session` is `!Debug`
                // (it owns a `Compute`), so the `Debug` bound `expect_err`
                // needs is not satisfiable here.
                let Err(e) =
                    load_session_with_backend(p, BackendKind::Cpu, Some(TaskHint::MelFrontend))
                else {
                    panic!("hint must be rejected");
                };
                e
            });
            assert!(
                err.contains("task hint") && err.contains(arch),
                "hint reject must name the hint and the arch ({arch}): {err}"
            );
        }
    }

    // ---- Wave G (2026-08-15) — bound-but-not-runnable arches -------------

    /// A bound arch no longer reads as "unknown": the message names the
    /// binding module and says the model IS bound.
    #[test]
    fn load_session_bound_arch_reports_the_binder_not_an_unknown_arch() {
        let err = with_arch_only_gguf("sepformer", "sepformer-arch", |p| {
            let Err(e) = load_session(p) else {
                panic!("sepformer has no run task");
            };
            e
        });
        assert!(
            !err.contains("unsupported model arch"),
            "sepformer has a binder — the old blanket message is the bug being fixed: {err}"
        );
        assert!(err.contains("is BOUND"), "must state it is bound: {err}");
        assert!(
            err.contains("vokra_models::sepformer"),
            "must name the binding module: {err}"
        );
        assert!(
            err.contains("SepFormer::separate"),
            "must name the library entry point to call: {err}"
        );
    }

    /// A row that carries a probe really loads the binder: the message
    /// reports one of the two probe outcomes, never neither.
    #[test]
    fn load_session_bound_arch_probe_reports_a_load_outcome() {
        let err = with_arch_only_gguf("emotion2vec", "emotion2vec-arch", |p| {
            let Err(e) = load_session(p) else {
                panic!("emotion2vec has no run task");
            };
            e
        });
        assert!(
            err.contains("LOADED and validated") || err.contains("FAILED — the binder reports:"),
            "a probed row must report the binder's own load outcome: {err}"
        );
    }

    /// A module with no GGUF loader says exactly that, rather than implying
    /// a forward gap it has not reached yet.
    #[test]
    fn load_session_bound_arch_without_a_gguf_loader_says_so() {
        let err = with_arch_only_gguf("zonos", "zonos-arch", |p| {
            let Err(e) = load_session(p) else {
                panic!("zonos has no run task");
            };
            e
        });
        assert!(
            err.contains("no GGUF loader yet"),
            "must name the real blocker for a synthesized-weights module: {err}"
        );
        assert!(
            err.contains("vokra_models::zonos"),
            "must name the binding module: {err}"
        );
    }

    // ---- Wave H (2026-08-15) — the five rows the registry had missed -----
    //
    // One test per row, not one loop over all five: a loop reports only the
    // first arch that regresses, and these five went missing INDIVIDUALLY.

    /// Asserts `arch` resolves through [`BOUND_ARCHES`] to its own binder —
    /// the message states the model is bound, names `module` and
    /// `entry_fragment`, and never falls back to the blanket "unsupported
    /// model arch". Returns the message so a caller can assert further.
    fn assert_bound_arch(arch: &str, tag: &str, module: &str, entry_fragment: &str) -> String {
        let err = with_arch_only_gguf(arch, tag, |p| {
            // let-else rather than `.expect_err()`: `Session` is `!Debug`.
            let Err(e) = load_session(p) else {
                panic!("`{arch}` has no run task — load_session must refuse it");
            };
            e
        });
        assert!(
            !err.contains("unsupported model arch"),
            "`{arch}` has a vokra-models binder — the blanket message is the bug being \
             fixed here: {err}"
        );
        assert!(
            err.contains("is BOUND"),
            "`{arch}` must state that this build binds it: {err}"
        );
        assert!(
            err.contains(module),
            "`{arch}` must name its binding module `{module}`: {err}"
        );
        assert!(
            err.contains(entry_fragment),
            "`{arch}` must name the library entry point `{entry_fragment}`: {err}"
        );
        err
    }

    /// ChatTTS (`vokra_models::chattts`) — loud-partial `synthesize`.
    #[test]
    fn load_session_binds_chattts_arch() {
        assert_bound_arch(
            "chattts",
            "chattts-arch",
            "vokra_models::chattts",
            "ChatTts::synthesize",
        );
    }

    /// Deepfake detection (`vokra_models::deepfake_detection`) — loud-partial
    /// `score`. A spoof detector misreported as an unknown arch is the worst
    /// of the five: the caller cannot tell "no such model" from "the model is
    /// here but its feature extractor is deferred".
    #[test]
    fn load_session_binds_deepfake_detection_arch() {
        assert_bound_arch(
            "deepfake_detection",
            "deepfake-arch",
            "vokra_models::deepfake_detection",
            "DeepfakeDetection::score",
        );
    }

    /// Spoken-language ID (`vokra_models::lang_id`) — loud-partial `identify`.
    /// Note the arch tag (`lang_id_ecapa`) is not the module name.
    #[test]
    fn load_session_binds_lang_id_ecapa_arch() {
        assert_bound_arch(
            "lang_id_ecapa",
            "lang-id-arch",
            "vokra_models::lang_id",
            "LangIdEcapa::identify",
        );
    }

    /// DTLN-AEC (`vokra_models::aec::dtln_aec`) — loud-partial `process`
    /// (the generic LSTM primitive is absent from `vokra-ops`).
    #[test]
    fn load_session_binds_dtln_aec_arch() {
        let err = assert_bound_arch(
            "dtln_aec",
            "dtln-aec-arch",
            "vokra_models::aec::dtln_aec",
            "DtlnAec::process",
        );
        assert!(
            err.contains("its runtime forward is a loud-partial"),
            "dtln_aec's blocker is its deferred forward, not its input shape: {err}"
        );
    }

    /// NKF-AEC (`vokra_models::aec::nkf_aec`) — the one row of the five whose
    /// forward is REAL (native re-implementation with an upstream parity
    /// harness). Its blocker is the paired (mic, far-end) input `run` cannot
    /// supply, so it must NOT be reported as a loud-partial like its DTLN
    /// sibling — that would slander a working model.
    #[test]
    fn load_session_binds_nkf_aec_arch_as_paired_input_not_loud_partial() {
        let err = assert_bound_arch(
            "nkf_aec",
            "nkf-aec-arch",
            "vokra_models::aec::nkf_aec",
            "NkfAecStream::push_paired",
        );
        assert!(
            err.contains("PAIR of strictly sample-aligned"),
            "nkf_aec must name the paired-input blocker: {err}"
        );
        assert!(
            !err.contains("its runtime forward is a loud-partial"),
            "nkf_aec's forward is real — reporting it as a loud-partial is a lie: {err}"
        );
    }

    // ---- Wave I (2026-08-15) — distil-whisper / kotoba-whisper ----------
    //
    // Both shipped as `BOUND_ARCHES` rows labelled `LoudPartialForward`, so a
    // user holding a real GGUF for either was told the runtime forward was a
    // loud-partial and `run` refused the model. Reading the binders shows the
    // opposite: `DistilWhisperAsr::transcribe` / `KotobaWhisperAsr::transcribe`
    // delegate to `WhisperAsr::transcribe_tokens` whenever the handle came
    // from `from_gguf` — which is the ONLY way the CLI can build one. The
    // scaffold arms that do hard-error are reachable only from the in-process
    // `::new` constructors, which no GGUF can reach.
    //
    // Nothing pinned the falsehood, so nothing blocked the fix; these tests
    // exist so nothing un-fixes it either. Same shape as the fsmn-vad /
    // nsnet2 routing tests above: a metadata-only fixture cannot bind a
    // 1.5 B-parameter Whisper checkpoint, so what is asserted is WHICH loader
    // the dispatch reached — the shared Whisper config reader, never the
    // registry and never the blanket unknown-arch message.

    /// Asserts `arch` is routed to the shared Whisper ASR loader: the failure
    /// on a metadata-only fixture is the Whisper config reader's own missing-key
    /// error, and none of the three "this model cannot run" messages appear.
    /// Returns the message so a caller can assert further.
    fn assert_routed_to_whisper_asr(arch: &str, tag: &str) -> String {
        let err = with_arch_only_gguf(arch, tag, |p| {
            // let-else rather than `.expect_err()`: `Session` is `!Debug`.
            let Err(e) = load_session(p) else {
                panic!("a metadata-only `{arch}` GGUF carries no weights — it cannot bind");
            };
            e
        });
        assert!(
            err.contains("whisper config"),
            "`{arch}` must reach the shared Whisper config loader — that is what proves \
             the dispatch routes it to the ASR task rather than refusing it: {err}"
        );
        assert!(
            err.contains("vokra.whisper.n_mels"),
            "`{arch}` must fail on the first missing Whisper hparam key, i.e. inside the \
             loader and not before it: {err}"
        );
        assert!(
            !err.contains("unsupported model arch"),
            "`{arch}` is routed — it must never read as unknown: {err}"
        );
        assert!(
            !err.contains("is BOUND"),
            "`{arch}` is routed, so it must not fall through to the BOUND_ARCHES \
             registry (a row there would now be unreachable): {err}"
        );
        assert!(
            !err.contains("its runtime forward is a loud-partial"),
            "`{arch}` delegates its forward to WhisperAsr — calling it a loud-partial is \
             the exact lie this routing removes: {err}"
        );
        err
    }

    /// distil-whisper (`vokra_models::distil_whisper`) — a real Whisper
    /// forward behind a shrunk decoder, routed to [`ModelTask::Asr`].
    #[test]
    fn load_session_routes_distil_whisper_to_the_whisper_asr_task() {
        assert_routed_to_whisper_asr("distil-whisper", "distil-whisper-arch");
    }

    /// kotoba-whisper (`vokra_models::kotoba_whisper`) — same delegation, same
    /// task. Landed as its own test rather than a loop over both: the two rows
    /// were wrong INDIVIDUALLY, and a loop reports only the first regression.
    #[test]
    fn load_session_routes_kotoba_whisper_to_the_whisper_asr_task() {
        assert_routed_to_whisper_asr("kotoba-whisper", "kotoba-whisper-arch");
    }

    /// The registry must not carry either arch again. `assert_routed_to_whisper_asr`
    /// checks the message a user sees; this checks the data behind it, so a
    /// re-added row fails here with a direct explanation rather than only as a
    /// downstream symptom.
    #[test]
    fn bound_arch_registry_does_not_slander_the_distilled_whispers() {
        for arch in [ARCH_DISTIL_WHISPER, ARCH_KOTOBA_WHISPER] {
            assert!(
                BOUND_ARCHES.iter().all(|b| b.arch != arch),
                "`{arch}` has a real forward (its binder delegates to WhisperAsr) and is \
                 routed to ModelTask::Asr — a BOUND_ARCHES row for it is both unreachable \
                 and untrue"
            );
        }
    }

    /// The registry is well formed: no duplicate arch strings, and no row
    /// shadowing an arch the dispatch actually runs (a duplicate there would
    /// be unreachable and would rot into a lie).
    ///
    /// This test can only inspect rows that EXIST — which is why it never
    /// noticed the five binders that shipped with no row at all. The
    /// completeness half lives in `scripts/check-bound-arch-coverage.sh`,
    /// which starts from the binders instead: a Rust test would have to walk
    /// `crates/vokra-models/src/` from a crate-relative CWD to do the same.
    #[test]
    fn bound_arch_registry_is_disjoint_from_the_routed_arches() {
        let routed = [
            ARCH_WHISPER,
            ARCH_DISTIL_WHISPER,
            ARCH_KOTOBA_WHISPER,
            ARCH_SILERO_VAD,
            ARCH_PIPER_PLUS,
            ARCH_CSM,
            ARCH_MOSHI,
            ARCH_CAMPPLUS,
            ARCH_VOXTRAL,
            ARCH_KOKORO,
            ARCH_SBV2,
            ARCH_FSMN_VAD,
            ARCH_NSNET2,
            ARCH_PYANNOTE_SEGMENTATION,
            ARCH_RMVPE,
            ARCH_MAGNET_SMALL,
            ARCH_MAGNET_MEDIUM,
            ARCH_MELODYFLOW_T24_30SECS,
        ];
        for (i, row) in BOUND_ARCHES.iter().enumerate() {
            assert!(
                !routed.contains(&row.arch),
                "`{}` is routed by the dispatch — a registry row for it is unreachable",
                row.arch
            );
            assert!(
                BOUND_ARCHES.iter().skip(i + 1).all(|o| o.arch != row.arch),
                "duplicate registry row for arch `{}`",
                row.arch
            );
            assert!(
                row.module.starts_with("vokra_models::"),
                "row `{}` must name a real vokra-models module path, got `{}`",
                row.arch,
                row.module
            );
            assert!(
                !row.entry.is_empty(),
                "row `{}` must name a library entry point",
                row.arch
            );
            if row.reason == BoundReason::NoGgufLoader {
                assert!(
                    row.probe.is_none(),
                    "row `{}` claims no GGUF loader yet carries a probe",
                    row.arch
                );
            }
        }
    }

    #[test]
    fn load_session_rejects_unknown_arch() {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "gpt2");
        let bytes = b.to_bytes().expect("serialize gguf");
        let mut path = std::env::temp_dir();
        path.push(format!("vokra-cli-arch-{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let result = load_session(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        let err = result.expect_err("unknown arch is rejected");
        assert!(err.contains("unsupported model arch"), "got: {err}");
    }
}
