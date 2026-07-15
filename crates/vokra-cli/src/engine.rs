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
    /// Text-to-speech (piper-plus native TTS).
    Tts,
    /// Speech-to-speech dialog (Sesame CSM-1B = M4-05). The reply text is
    /// caller-supplied (`--text`), optional `--input` WAV = recorded
    /// context audio (explicit AEC bypass — T16).
    S2s,
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
    let session = Session::from_file(path)
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
             `{ARCH_SILERO_VAD}` / `{ARCH_PIPER_PLUS}` / `{ARCH_CSM}`)"
        )),
    }
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
