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

use vokra_core::{BackendKind, Session};
use vokra_models::csm::{CsmEngine, EchoPath, FixtureByteTokenizer};
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
    /// [`crate::bench::execute`] rely on it to surface an explicit
    /// unimplemented signal if a future engine.rs change ever *does*
    /// return it (defense in depth against a silent fall back — the
    /// FR-EX-08 posture the whole CLI upholds). The dead-code allow
    /// documents this state so a reviewer does not delete the arm.
    #[allow(dead_code)]
    Cosyvoice2Synthetic,
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

/// Opens the GGUF at `path` on the CPU backend, injects the engine matching its
/// `vokra.model.arch` and returns the ready session plus its task.
#[cfg(test)]
pub(crate) fn load_session(path: &str) -> Result<(Session, ModelTask), String> {
    load_session_with_backend(path, BackendKind::Cpu, None)
}

/// As [`load_session`], but runs the model's hot ops on `backend` (CPU / Metal /
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
        other => Err(format!(
            "unsupported model arch `{other}` (expected `{ARCH_WHISPER}` / \
             `{ARCH_SILERO_VAD}` / `{ARCH_PIPER_PLUS}` / `{ARCH_CSM}` / \
             `{ARCH_MOSHI}` / `{ARCH_CAMPPLUS}` / `{ARCH_VOXTRAL}` / \
             `{ARCH_KOKORO}`)"
        )),
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
