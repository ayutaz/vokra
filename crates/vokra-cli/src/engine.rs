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
use vokra_models::moonshine::Moonshine;
use vokra_models::parakeet::ParakeetAsr;
use vokra_models::parakeet_ctc::ParakeetCtcAsr;
use vokra_models::piper_plus::PiperPlusTts;
use vokra_models::silero_vad::SileroVadV5;
use vokra_models::whisper::WhisperAsr;
use vokra_models::whisper_medusa::WhisperMedusa;

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
    /// Voice activity detection through FireRedTeam Stream-VAD.
    ///
    /// The native Kaldi-fbank + CMVN + causal DFSMN forward implements the
    /// same `VadEngine` contract as Silero and FSMN-VAD.
    VadFirered,
    /// Voice activity detection through native TEN-VAD v1.0.
    ///
    /// The LPCNet-derived frontend and separable-conv/two-LSTM network share
    /// the common `VadEngine` stream contract. Canonical official weights are
    /// local-use only because their upstream deployment license is restricted.
    VadTen,
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
    F0Rmvpe,
    /// F0 extraction through FCPE. The concrete path-taking binder is opened
    /// once by the `run` arm and returns the same [`vokra_models::f0::F0Frame`]
    /// rows as RMVPE.
    F0Fcpe,
    /// F0 extraction through CREPE. Kept distinct from FCPE/RMVPE so rate and
    /// checkpoint diagnostics name the actual front-end.
    F0Crepe,
    /// Charsiu neural forced alignment. The run arm binds the concrete model
    /// from the already parsed GGUF and pairs `--input` audio with the
    /// whitespace-delimited official phone sequence in `--text`.
    AlignCharsiu,
    /// WeTextProcessing inverse/text normalization (`--text` → text). The
    /// concrete binder is built from the session GGUF in the `run` arm; a
    /// build without `vokra-wfst` retains the binder's loud feature error.
    TextNormalize,
    /// NKF-AEC offline paired-WAV route. The concrete model opens a stateful
    /// [`vokra_core::engines::AecStreamHandle`] in the `run` arm because the
    /// generic [`Session`] facade has no AEC engine slot.
    AecNkf,
    /// CT-Transformer punctuation restoration. The generic session has no
    /// punctuation-engine slot, so `run` binds [`vokra_models::ct_punc::CtPunc`]
    /// from `session.gguf()` and consumes the versioned paired token/id TSV.
    CtPunc,
    /// Standalone Mimi codec encode/decode. `run` binds the real encoder,
    /// effective RVQ tables, and neural decoder from the same GGUF and uses
    /// the versioned portable code container between the two modes.
    MimiCodec,
    /// NVIDIA BigVGAN mel-to-waveform vocoder. `run` binds the concrete
    /// model from the session GGUF and consumes channel-major little-endian
    /// f32 mel frames from `--input`.
    VocoderBigVgan,
    /// Microsoft SpeechT5 HiFi-GAN mel-to-waveform vocoder. The concrete
    /// model is bound from the session GGUF in the `run` arm.
    VocoderHifiGan,
    /// Charactr Vocos feature-to-waveform vocoder. The mel variant consumes
    /// 100-channel features; the Encodec variant consumes 128-channel
    /// features plus an explicit bandwidth condition.
    VocoderVocos,
    /// openWakeWord streaming keyword spotting. The generic Session facade
    /// has no KWS slot, so `run` binds the concrete mutable session once from
    /// the already parsed GGUF and feeds the complete 16 kHz clip.
    KwsOpenwakeword,
    /// Pipecat smart-turn v2 semantic endpointing. The whole input utterance
    /// maps to one completion probability; this is deliberately distinct
    /// from the streaming frame-level VAD tasks.
    SmartTurn,
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
/// CrisperWhisper (`nyrahealth/CrisperWhisper`) — a Whisper checkpoint
/// retrained for verbatim, disfluency-preserving transcription.
///
/// Routed to the same arm as [`ARCH_WHISPER`] because it IS a Whisper
/// checkpoint: `vokra_models::whisper::ACCEPTED_ARCHS` lists it, and
/// `WhisperAsr::from_gguf` binds it through the identical path.
///
/// This constant closes a round-trip hole found by the 2026-08-15 audit:
/// `vokra-cli convert --model crisperwhisper` accepted four spellings and
/// the converter stamped this tag, but the dispatch below matched
/// `ARCH_WHISPER` alone — so the CLI refused the GGUF it had just produced,
/// reporting "unsupported model arch" for a model it fully supports. Both
/// arch gates were blind to it: `pub(crate)` on the converter side, and a
/// member of an aggregate `ACCEPTED_ARCHS` slice rather than its own
/// `pub const` on the binder side.
const ARCH_CRISPER_WHISPER: &str = "crisper-whisper";
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
/// FireRedTeam Stream-VAD native DFSMN.
const ARCH_FIRERED_VAD: &str = "firered_vad";
/// TEN-framework TEN-VAD v1.0 native streaming VAD.
const ARCH_TEN_VAD: &str = "ten_vad";
/// openWakeWord native v0.5.1 KWS pipeline.
const ARCH_OPENWAKEWORD_OP: &str = "openwakeword_op";
/// Pipecat smart-turn v2 utterance-level endpoint classifier.
const ARCH_SMART_TURN: &str = "smart_turn";
/// NSNet2 (Microsoft DNS-Challenge baseline denoiser) — mirror of
/// [`vokra_models::nsnet2::ARCH`].
const ARCH_NSNET2: &str = "nsnet2";
/// Xiph RNNoise v0.2 native waveform denoiser.
const ARCH_RNNOISE: &str = "rnnoise";
/// pyannote `segmentation-3.0` — mirror of
/// [`vokra_models::pyannote::EXPECTED_ARCH`].
const ARCH_PYANNOTE_SEGMENTATION: &str = "pyannote-segmentation";
/// RMVPE pitch extractor — mirror of `vokra-convert`'s `models::rmvpe::ARCH`.
/// The binder itself does not read `vokra.model.arch` (the whole `f0` family
/// keys off `vokra.f0.*` instead), so this dispatch is the only place the
/// string is matched; it is kept verbatim in lock-step with the converter.
const ARCH_RMVPE: &str = "rmvpe";
/// FCPE pitch extractor — mirror of `vokra_models::f0::fcpe::ARCH`.
const ARCH_FCPE: &str = "fcpe";
/// CREPE pitch extractor — mirror of `vokra_models::f0::crepe::ARCH`.
const ARCH_CREPE: &str = "crepe";
/// Charsiu English 10 ms forced aligner.
const ARCH_CHARSIU: &str = "charsiu";
/// WeTextProcessing ITN/TN bundle — mirror of
/// `vokra_models::wetextprocessing::ARCH`.
const ARCH_WETEXTPROCESSING: &str = "wetextprocessing";
/// NKF-AEC — mirror of `vokra_models::aec::nkf_aec::ARCH`.
const ARCH_NKF_AEC: &str = "nkf_aec";
/// CT-Punc text post-processor — mirror of
/// [`vokra_models::ct_punc::ARCH`].
const ARCH_CT_PUNC: &str = "ct_punc";
/// Standalone Kyutai Mimi codec — mirror of what
/// `vokra-cli convert --model mimi` writes.
const ARCH_MIMI: &str = "mimi";
/// NVIDIA BigVGAN vocoder — mirror of [`vokra_models::bigvgan::ARCH`].
const ARCH_BIGVGAN: &str = "bigvgan";
/// Microsoft SpeechT5 HiFi-GAN vocoder.
const ARCH_SPEECHT5_HIFIGAN: &str = "speecht5_hifigan";
/// SpeechBrain LibriTTS 22.05 kHz HiFi-GAN vocoder.
const ARCH_HIFIGAN_VOCODER: &str = "hifigan_vocoder";
/// Charactr Fourier-space Vocos vocoder.
const ARCH_VOCOS: &str = "vocos";

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
/// Moonshine Tiny/Base raw-waveform encoder-decoder ASR.
const ARCH_MOONSHINE: &str = "moonshine";
/// NVIDIA Parakeet-TDT-0.6B-v3 FastConformer + TDT ASR.
const ARCH_PARAKEET_TDT: &str = "parakeet-tdt";
/// NVIDIA Parakeet-CTC-1.1B FastConformer + CTC ASR.
const ARCH_PARAKEET_CTC: &str = "parakeet-ctc";
/// aiola Whisper-Medusa-v1 official module-0 ASR forward.
const ARCH_WHISPER_MEDUSA_V1: &str = "whisper-medusa-v1";

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
        // CrisperWhisper shares this arm: same architecture, same loader, only
        // the training objective differs. Splitting it out would duplicate the
        // mel-frontend and beam handling below for no behavioural difference.
        ARCH_WHISPER | ARCH_CRISPER_WHISPER => {
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
        ARCH_MOONSHINE => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is only supported on arch `{ARCH_WHISPER}` \
                     (got `{ARCH_MOONSHINE}`)"
                ));
            }
            let asr = Moonshine::from_gguf(session.gguf())
                .map_err(|error| error.to_string())?
                .with_backend(backend);
            Ok((session.with_asr_engine(Arc::new(asr)), ModelTask::Asr))
        }
        ARCH_PARAKEET_TDT => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is only supported on arch `{ARCH_WHISPER}` \
                     (got `{ARCH_PARAKEET_TDT}`)"
                ));
            }
            if backend != BackendKind::Cpu {
                return Err(format!(
                    "Parakeet-TDT currently implements the exact FastConformer/TDT forward on CPU only; backend {backend:?} is unsupported (no silent CPU fallback)"
                ));
            }
            let asr = ParakeetAsr::from_gguf(session.gguf()).map_err(|error| error.to_string())?;
            if !asr.has_tokenizer() {
                return Err(
                    "Parakeet-TDT GGUF has no embedded official tokenizer.json; reconvert with `vokra-cli convert --model parakeet-tdt --tokenizer tokenizer.json`"
                        .to_owned(),
                );
            }
            Ok((session.with_asr_engine(Arc::new(asr)), ModelTask::Asr))
        }
        ARCH_PARAKEET_CTC => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is only supported on arch `{ARCH_WHISPER}` \
                     (got `{ARCH_PARAKEET_CTC}`)"
                ));
            }
            if backend != BackendKind::Cpu {
                return Err(format!(
                    "Parakeet-CTC currently implements the exact FastConformer/CTC forward on CPU only; backend {backend:?} is unsupported (no silent CPU fallback)"
                ));
            }
            let asr =
                ParakeetCtcAsr::from_gguf(session.gguf()).map_err(|error| error.to_string())?;
            Ok((session.with_asr_engine(Arc::new(asr)), ModelTask::Asr))
        }
        ARCH_WHISPER_MEDUSA_V1 => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is only supported on arch `{ARCH_WHISPER}` \
                     (got `{ARCH_WHISPER_MEDUSA_V1}`)"
                ));
            }
            if backend != BackendKind::Cpu {
                return Err(format!(
                    "Whisper-Medusa module-0 output adaptation is CPU-only; backend \
                     {backend:?} is unsupported (no silent CPU fallback)"
                ));
            }
            let asr =
                WhisperMedusa::from_gguf(session.gguf()).map_err(|error| error.to_string())?;
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
        ARCH_FIRERED_VAD => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_FIRERED_VAD}`"
                ));
            }
            let vad = vokra_models::firered_vad::FireredVad::from_gguf(session.gguf())
                .map_err(|error| format!("arch `{ARCH_FIRERED_VAD}`: {error}"))?;
            Ok((
                session.with_vad_engine(Arc::new(vad)),
                ModelTask::VadFirered,
            ))
        }
        ARCH_TEN_VAD => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_TEN_VAD}`"
                ));
            }
            let vad = vokra_models::ten_vad::TenVad::from_gguf(session.gguf())
                .map_err(|error| format!("arch `{ARCH_TEN_VAD}`: {error}"))?;
            Ok((session.with_vad_engine(Arc::new(vad)), ModelTask::VadTen))
        }
        ARCH_OPENWAKEWORD_OP => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_OPENWAKEWORD_OP}`"
                ));
            }
            Ok((session, ModelTask::KwsOpenwakeword))
        }
        ARCH_SMART_TURN => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_SMART_TURN}`"
                ));
            }
            // Bare session: the run/bench arm binds the 379 MB concrete model
            // exactly once and calls its utterance-level endpoint surface.
            Ok((session, ModelTask::SmartTurn))
        }
        ARCH_NSNET2 | ARCH_RNNOISE => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on denoise arch `{arch}`"
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
        ARCH_FCPE => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_FCPE}`"
                ));
            }
            // Path-taking binder; bind once in the run arm so the same loaded
            // weights produce both the config/rate check and the F0 track.
            Ok((session, ModelTask::F0Fcpe))
        }
        ARCH_CREPE => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_CREPE}`"
                ));
            }
            Ok((session, ModelTask::F0Crepe))
        }
        ARCH_CHARSIU => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_CHARSIU}`"
                ));
            }
            Ok((session, ModelTask::AlignCharsiu))
        }
        ARCH_WETEXTPROCESSING => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_WETEXTPROCESSING}`"
                ));
            }
            // The binder consumes the already parsed GGUF in the run arm and
            // the feature-gated FST pipeline returns its own precise error.
            Ok((session, ModelTask::TextNormalize))
        }
        ARCH_NKF_AEC => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_NKF_AEC}`"
                ));
            }
            Ok((session, ModelTask::AecNkf))
        }
        ARCH_CT_PUNC => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_CT_PUNC}`"
                ));
            }
            // Bare session: the run arm binds the concrete model once so its
            // paired token/id API remains reachable without adding a fake
            // generic text-engine trait.
            Ok((session, ModelTask::CtPunc))
        }
        ARCH_MIMI => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_MIMI}`"
                ));
            }
            // Bare session: standalone encode/decode needs all three concrete
            // codec components and a versioned codes container, none of which
            // belongs in the ASR/TTS/S2S session slots.
            Ok((session, ModelTask::MimiCodec))
        }
        ARCH_BIGVGAN => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_BIGVGAN}`"
                ));
            }
            Ok((session, ModelTask::VocoderBigVgan))
        }
        ARCH_SPEECHT5_HIFIGAN | ARCH_HIFIGAN_VOCODER => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_SPEECHT5_HIFIGAN}`"
                ));
            }
            Ok((session, ModelTask::VocoderHifiGan))
        }
        ARCH_VOCOS => {
            if hint.is_some() {
                return Err(format!(
                    "task hint {hint:?} is not supported on arch `{ARCH_VOCOS}`"
                ));
            }
            Ok((session, ModelTask::VocoderVocos))
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
                 `{ARCH_MOONSHINE}` / `{ARCH_PARAKEET_TDT}` / \
                 `{ARCH_WHISPER_MEDUSA_V1}` / \
                 `{ARCH_SILERO_VAD}` / `{ARCH_PIPER_PLUS}` / `{ARCH_CSM}` / \
                 `{ARCH_MOSHI}` / `{ARCH_CAMPPLUS}` / `{ARCH_VOXTRAL}` / \
                 `{ARCH_KOKORO}` / `{ARCH_SBV2}` / `{ARCH_FSMN_VAD}` / \
                 `{ARCH_FIRERED_VAD}` / \
                 `{ARCH_OPENWAKEWORD_OP}` / \
                 `{ARCH_SMART_TURN}` / \
                 `{ARCH_NSNET2}` / `{ARCH_RNNOISE}` / `{ARCH_PYANNOTE_SEGMENTATION}` / \
                 `{ARCH_RMVPE}` / `{ARCH_FCPE}` / `{ARCH_CREPE}` / \
                 `{ARCH_CHARSIU}` / \
                 `{ARCH_WETEXTPROCESSING}` / `{ARCH_NKF_AEC}` / \
                 `{ARCH_CT_PUNC}` / `{ARCH_MIMI}` / \
                 `{ARCH_MAGNET_SMALL}` / `{ARCH_MAGNET_MEDIUM}` / \
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
// point a caller reaches the runtime through, and — where the binder's loader
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

/// Every current registry row has a GGUF loader and a loud-partial runtime
/// forward. The former `NoGgufLoader` class reached zero in the 2026-08-22 TTS
/// loader wave, so the diagnostic no longer carries a dead blocker variant.
const LOUD_PARTIAL_EXPLANATION: &str = "its runtime forward is a loud-partial — calling the entry point below reports the \
     specific missing primitive and the primary source to transcribe it from (this CLI \
     deliberately does not restate that gap: a copy here would drift away from the binder)";

/// One architecture `vokra-models` binds but `vokra-cli run` cannot execute.
#[derive(Clone, Copy)]
struct BoundArch {
    /// The `vokra.model.arch` string the converter stamps.
    arch: &'static str,
    /// The `vokra_models` module that binds it.
    module: &'static str,
    /// The public entry point a library caller reaches the runtime through.
    entry: &'static str,
    /// Load probe, when the binder's loader takes an already-parsed
    /// `&GgufFile`. `None` for path-taking loaders.
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
        probe: Some(|g: &GgufFile| vokra_models::canary::CanaryAsr::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "canary-1b-flash",
        module: "vokra_models::canary_1b_flash",
        entry: "Canary1bFlashAsr::from_gguf → Canary1bFlashAsr::transcribe",
        probe: Some(|g: &GgufFile| {
            vokra_models::canary_1b_flash::Canary1bFlashAsr::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "canary-qwen",
        module: "vokra_models::canary_qwen",
        entry: "CanaryQwenAsr::from_gguf → CanaryQwenAsr::transcribe",
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
        arch: "omniasr-ctc",
        module: "vokra_models::omniasr_ctc",
        entry: "OmniasrCtcAsr::from_gguf → OmniasrCtcAsr::transcribe",
        probe: Some(|g: &GgufFile| {
            vokra_models::omniasr_ctc::OmniasrCtcAsr::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "parakeet-tdt-1_1b",
        module: "vokra_models::parakeet_tdt_1_1b",
        entry: "ParakeetTdt11b::from_gguf → ParakeetTdt11b::transcribe",
        probe: Some(|g: &GgufFile| {
            vokra_models::parakeet_tdt_1_1b::ParakeetTdt11b::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "sensevoicesmall",
        module: "vokra_models::sensevoicesmall_runtime",
        entry: "SenseVoiceSmall::from_gguf → SenseVoiceSmall::transcribe",
        probe: Some(|g: &GgufFile| {
            vokra_models::sensevoicesmall_runtime::SenseVoiceSmall::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "firered_asr_aed_l",
        module: "vokra_models::firered_asr_aed",
        entry: "FireredAsrAed::from_gguf → FireredAsrAed::transcribe_tokens",
        probe: Some(|g: &GgufFile| {
            vokra_models::firered_asr_aed::FireredAsrAed::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "sber_gigaam_v3",
        module: "vokra_models::gigaam",
        entry: "Gigaam::from_gguf → Gigaam::transcribe",
        probe: Some(|g: &GgufFile| vokra_models::gigaam::Gigaam::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "gigaam_multilingual",
        module: "vokra_models::gigaam",
        entry: "Gigaam::from_gguf → Gigaam::transcribe",
        probe: Some(|g: &GgufFile| vokra_models::gigaam::Gigaam::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "kyutai-stt",
        module: "vokra_models::kyutai_stt",
        entry: "KyutaiSttAsr::from_path → KyutaiSttAsr::transcribe",
        probe: None,
    },
    BoundArch {
        arch: "mt3",
        module: "vokra_models::mt3",
        entry: "Mt3::from_gguf → Mt3::transcribe",
        probe: Some(|g: &GgufFile| vokra_models::mt3::Mt3::from_gguf(g).map(|_| ())),
    },
    // --- TTS -------------------------------------------------------------
    BoundArch {
        arch: "styletts2",
        module: "vokra_models::styletts2",
        entry: "StyleTts2Tts::from_gguf → StyleTts2Tts::synthesize",
        probe: None,
    },
    BoundArch {
        arch: "cosyvoice2",
        module: "vokra_models::cosyvoice2",
        entry: "CosyVoice2Tts::from_path → CosyVoice2Tts::synthesize_pcm_from_mel",
        probe: None,
    },
    BoundArch {
        arch: "cosyvoice3",
        module: "vokra_models::cosyvoice3",
        entry: "CosyVoice3Checkpoint::from_gguf → CosyVoice3Checkpoint::synthesize",
        probe: Some(|g: &GgufFile| {
            vokra_models::cosyvoice3::CosyVoice3Checkpoint::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "chatterbox",
        module: "vokra_models::chatterbox",
        entry: "ChatterboxCheckpoint::from_gguf → ChatterboxCheckpoint::synthesize",
        probe: Some(|g: &GgufFile| {
            vokra_models::chatterbox::ChatterboxCheckpoint::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "chatterbox_nano",
        module: "vokra_models::chatterbox_nano",
        entry: "ChatterboxNanoCheckpoint::from_gguf → ChatterboxNanoCheckpoint::synthesize",
        probe: Some(|g: &GgufFile| {
            vokra_models::chatterbox_nano::ChatterboxNanoCheckpoint::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "chatterbox_turbo",
        module: "vokra_models::chatterbox_turbo",
        entry: "ChatterboxTurboCheckpoint::from_gguf → ChatterboxTurboCheckpoint::synthesize",
        probe: Some(|g: &GgufFile| {
            vokra_models::chatterbox_turbo::ChatterboxTurboCheckpoint::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "dia",
        module: "vokra_models::dia",
        entry: "DiaCheckpoint::from_gguf → DiaCheckpoint::synthesize",
        probe: Some(|g: &GgufFile| vokra_models::dia::DiaCheckpoint::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "irodori-tts",
        module: "vokra_models::irodori",
        entry: "IrodoriCheckpoint::from_gguf → IrodoriCheckpoint::synthesize",
        probe: Some(|g: &GgufFile| {
            vokra_models::irodori::IrodoriCheckpoint::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "qwen3_tts",
        module: "vokra_models::qwen3_tts",
        entry: "Qwen3TtsCheckpoint::from_gguf → Qwen3TtsCheckpoint::synthesize",
        probe: Some(|g: &GgufFile| {
            vokra_models::qwen3_tts::Qwen3TtsCheckpoint::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "vibevoice",
        module: "vokra_models::vibevoice",
        entry: "VibeVoiceCheckpoint::from_gguf → VibeVoiceCheckpoint::synthesize",
        probe: Some(|g: &GgufFile| {
            vokra_models::vibevoice::VibeVoiceCheckpoint::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "vits-ja",
        module: "vokra_models::vits_ja",
        entry: "VitsJaCheckpoint::from_gguf → VitsJaCheckpoint::synthesize",
        probe: Some(|g: &GgufFile| {
            vokra_models::vits_ja::VitsJaCheckpoint::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "voxcpm2",
        module: "vokra_models::voxcpm2",
        entry: "VoxCpm2Checkpoint::from_gguf → VoxCpm2Checkpoint::synthesize",
        probe: Some(|g: &GgufFile| {
            vokra_models::voxcpm2::VoxCpm2Checkpoint::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "zonos",
        module: "vokra_models::zonos",
        entry: "ZonosCheckpoint::from_gguf → ZonosCheckpoint::synthesize",
        probe: Some(|g: &GgufFile| vokra_models::zonos::ZonosCheckpoint::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "diffsinger",
        module: "vokra_models::diffsinger",
        entry: "DiffSinger::from_gguf → DiffSinger::synthesize_mel",
        probe: Some(|g: &GgufFile| vokra_models::diffsinger::DiffSinger::from_gguf(g).map(|_| ())),
    },
    // --- Speech-to-speech -------------------------------------------------
    // The same defect the `openwakeword_op` row above carried, found one
    // round later in this one. `LoudPartialForward` asserts the LOAD
    // works and only the forward is partial; until 2026-08-15 that was
    // false here. The named converter
    // (`crates/vokra-convert/src/models/llama_omni2.rs`) stamped one of
    // the eleven `vokra.llama_omni2.*` keys the binder declares, so the
    // other ten decayed to `0` and `validate_for_forward` refused every
    // artifact with "backbone ill-formed (n_layer=0, d_model=0,
    // n_head=0)". This row described the second failure of a pipeline
    // that never survived the first.
    //
    // `probe: None` is correct rather than an oversight — `from_path`
    // takes a path, not an already-parsed `&GgufFile`, which is exactly
    // the case the field's doc calls out. It also would not have caught
    // this: a probe only runs against a caller-supplied GGUF, so it fails
    // on someone's machine, never in CI. What holds this row honest is
    // `crates/vokra-models/tests/llama_omni2_convert_bind.rs`, which runs
    // the real converter into the real binder with no fixture.
    //
    // The claim is accurate as of the converter repair: the load binds a
    // real shape config (four axes derived from the tensors, six from a
    // now-required `--config` side-car) and `converse` is a genuine
    // loud-partial — `UnsupportedOp` naming the missing Qwen2.5 forward,
    // speech encoder, streaming AR decoder and the primary source. Note
    // the weight store is still `synthesized`, so a successful load is
    // not a claim that real ICTNLP weights are bound.
    BoundArch {
        arch: "llama_omni2",
        module: "vokra_models::llama_omni2",
        entry: "LlamaOmni2::from_path → LlamaOmni2::converse",
        probe: None,
    },
    BoundArch {
        arch: "voila",
        module: "vokra_models::voila",
        entry: "Voila::from_gguf → Voila::converse",
        probe: Some(|g: &GgufFile| vokra_models::voila::Voila::from_gguf(g).map(|_| ())),
    },
    // --- Music / audio generation ----------------------------------------
    BoundArch {
        arch: "musicgen",
        module: "vokra_models::musicgen",
        entry: "MusicGen::from_gguf → MusicGen::generate",
        probe: Some(|g: &GgufFile| vokra_models::musicgen::MusicGen::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "audiogen",
        module: "vokra_models::audiogen",
        entry: "AudioGen::from_gguf → AudioGen::generate",
        probe: Some(|g: &GgufFile| vokra_models::audiogen::AudioGen::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "audioldm2",
        module: "vokra_models::audioldm2",
        entry: "AudioLdm2::from_gguf → AudioLdm2::generate",
        probe: Some(|g: &GgufFile| vokra_models::audioldm2::AudioLdm2::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "jasco_400m_chords_drums",
        module: "vokra_models::jasco",
        entry: "Jasco::from_gguf → Jasco::generate",
        probe: Some(|g: &GgufFile| vokra_models::jasco::Jasco::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "beat-this",
        module: "vokra_models::beat_this",
        entry: "BeatThis::from_gguf → BeatThis::analyze",
        probe: Some(|g: &GgufFile| vokra_models::beat_this::BeatThis::from_gguf(g).map(|_| ())),
    },
    // --- Source separation / enhancement / super-resolution ---------------
    BoundArch {
        arch: "sepformer",
        module: "vokra_models::sepformer",
        entry: "SepFormer::from_gguf → SepFormer::separate",
        probe: Some(|g: &GgufFile| vokra_models::sepformer::SepFormer::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "conv_tasnet",
        module: "vokra_models::conv_tasnet",
        entry: "ConvTasnet::from_gguf → ConvTasnet::separate",
        probe: Some(|g: &GgufFile| vokra_models::conv_tasnet::ConvTasnet::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "demucs",
        module: "vokra_models::demucs",
        entry: "Demucs::from_gguf → Demucs::separate",
        probe: Some(|g: &GgufFile| vokra_models::demucs::Demucs::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "gtcrn",
        module: "vokra_models::gtcrn",
        entry: "Gtcrn::from_gguf → Gtcrn::denoise",
        probe: Some(|g: &GgufFile| vokra_models::gtcrn::Gtcrn::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "facebook_denoiser",
        module: "vokra_models::facebook_denoiser",
        entry: "FbDenoiser::from_gguf → FbDenoiser::denoise",
        probe: Some(|g: &GgufFile| {
            vokra_models::facebook_denoiser::FbDenoiser::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "storm",
        module: "vokra_models::storm",
        entry: "Storm::from_gguf → Storm::enhance",
        probe: Some(|g: &GgufFile| vokra_models::storm::Storm::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "audiosr",
        module: "vokra_models::audiosr",
        entry: "AudioSr::from_gguf → AudioSr::super_resolve",
        probe: Some(|g: &GgufFile| vokra_models::audiosr::AudioSr::from_gguf(g).map(|_| ())),
    },
    // --- Diarization / speaker -------------------------------------------
    BoundArch {
        arch: "sortformer",
        module: "vokra_models::sortformer_diar_4spk_v1",
        entry: "SortformerDiar::from_gguf → SortformerDiar::diarize",
        probe: Some(|g: &GgufFile| {
            vokra_models::sortformer_diar_4spk_v1::SortformerDiar::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "speaker_3d",
        module: "vokra_models::speaker_3d_eres2net",
        entry: "Speaker3dEres2Net::from_gguf → Speaker3dEres2Net::encode",
        probe: Some(|g: &GgufFile| {
            vokra_models::speaker_3d_eres2net::Speaker3dEres2Net::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "redimnet",
        module: "vokra_models::redimnet",
        entry: "ReDimNet::from_gguf → ReDimNet::encode",
        probe: Some(|g: &GgufFile| vokra_models::redimnet::ReDimNet::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "wavlm_sv",
        module: "vokra_models::wavlm",
        entry: "WavLmSv::from_gguf → WavLmSv::encode",
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
        probe: Some(|g: &GgufFile| vokra_models::atst::Atst::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "eat",
        module: "vokra_models::eat",
        entry: "Eat::from_gguf → Eat::encode / Eat::embed_utterance",
        probe: Some(|g: &GgufFile| vokra_models::eat::Eat::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "m2d",
        module: "vokra_models::m2d",
        entry: "M2d::from_gguf → M2d::encode / M2d::embed",
        probe: Some(|g: &GgufFile| vokra_models::m2d::M2d::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "maest",
        module: "vokra_models::maest",
        entry: "Maest::from_gguf → Maest::encode / Maest::tag",
        probe: Some(|g: &GgufFile| vokra_models::maest::Maest::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "w2v-bert-2",
        module: "vokra_models::w2v_bert2",
        entry: "W2vBert2::from_gguf → W2vBert2::encode",
        probe: Some(|g: &GgufFile| vokra_models::w2v_bert2::W2vBert2::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "clap",
        module: "vokra_models::clap",
        entry: "Clap::from_gguf → Clap::encode_audio",
        probe: Some(|g: &GgufFile| vokra_models::clap::Clap::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "emotion2vec",
        module: "vokra_models::emotion2vec",
        entry: "Emotion2Vec::from_gguf → Emotion2Vec::classify",
        probe: Some(|g: &GgufFile| {
            vokra_models::emotion2vec::Emotion2Vec::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "panns",
        module: "vokra_models::panns",
        entry: "Panns::from_gguf → Panns::classify",
        probe: Some(|g: &GgufFile| vokra_models::panns::Panns::from_gguf(g).map(|_| ())),
    },
    // --- Quality metrics --------------------------------------------------
    BoundArch {
        arch: "dnsmos",
        module: "vokra_models::dnsmos_p808_p835",
        entry: "Dnsmos::from_gguf → Dnsmos::score_all",
        probe: Some(|g: &GgufFile| {
            vokra_models::dnsmos_p808_p835::Dnsmos::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "nisqa_v2_weight",
        module: "vokra_models::nisqa",
        entry: "Nisqa::from_gguf → Nisqa::score",
        probe: Some(|g: &GgufFile| vokra_models::nisqa::Nisqa::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "utmosv2",
        module: "vokra_models::utmosv2",
        entry: "Utmosv2::from_gguf → Utmosv2::predict_mos",
        probe: Some(|g: &GgufFile| vokra_models::utmosv2::Utmosv2::from_gguf(g).map(|_| ())),
    },
    BoundArch {
        arch: "torchaudio_squim",
        module: "vokra_models::squim",
        entry: "Squim::from_gguf → Squim::estimate_objective / estimate_subjective",
        probe: Some(|g: &GgufFile| vokra_models::squim::Squim::from_gguf(g).map(|_| ())),
    },
    // --- Vocoders / codecs -----------------------------------------------
    // BigVGAN and Vocos left this registry on 2026-08-21 after strict
    // loaders, real forwards, parity, and explicit feature-file CLI
    // contracts landed.
    BoundArch {
        arch: "snac",
        module: "vokra_models::snac",
        entry: "Snac::from_gguf → Snac::encode / Snac::decode",
        probe: Some(|g: &GgufFile| vokra_models::snac::Snac::from_gguf(g).map(|_| ())),
    },
    // --- Text / alignment side-cars ---------------------------------------
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
        probe: Some(|g: &GgufFile| {
            vokra_models::deepfake_detection::DeepfakeDetection::from_gguf(g).map(|_| ())
        }),
    },
    BoundArch {
        arch: "lang_id_ecapa",
        module: "vokra_models::lang_id",
        entry: "LangIdEcapa::from_gguf → LangIdEcapa::identify",
        probe: Some(|g: &GgufFile| vokra_models::lang_id::LangIdEcapa::from_gguf(g).map(|_| ())),
    },
    // DTLN-AEC still stops at the absent generic LSTM primitive. NKF-AEC is
    // routed above now that `run` has an explicit far-end WAV contract.
    BoundArch {
        arch: "dtln_aec",
        module: "vokra_models::aec::dtln_aec",
        entry: "DtlnAec::from_gguf → DtlnAec::process",
        probe: Some(|g: &GgufFile| vokra_models::aec::dtln_aec::DtlnAec::from_gguf(g).map(|_| ())),
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
        // Deliberately states only that no probe ran, not WHY. The rows
        // reaching this arm no longer share a single cause: some have no
        // loader at all, some have one that takes a filesystem path, and
        // some (vocos, hifigan_vocoder) have a
        // `&GgufFile` loader that refuses on every path. The previous
        // wording asserted the second case for all of them, and also told
        // the reader the load "has to happen through the library entry
        // point below" — false wherever that entry is a hand-build
        // constructor like `HiFiGan::new`, which loads nothing.
        None => "This CLI did not probe-load it: no `&GgufFile` probe is registered for this \
                 row, so the binder's own load behaviour is not exercised here. Reach the \
                 runtime through the library entry point below."
            .to_owned(),
    };
    let arch = bound.arch;
    let module = bound.module;
    let entry = bound.entry;
    let reason = LOUD_PARTIAL_EXPLANATION;
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
        static NEXT_FIXTURE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", arch);
        let bytes = b.to_bytes().expect("serialize gguf");
        let mut path = std::env::temp_dir();
        let fixture_id = NEXT_FIXTURE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        path.push(format!(
            "vokra-cli-{tag}-{}-{fixture_id}.gguf",
            std::process::id()
        ));
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

    /// An `nsnet2` or `rnnoise` GGUF dispatches to [`ModelTask::Denoise`] with
    /// a bare session — the concrete model binds in the `run` arm (the
    /// campplus / voxtral precedent), so a metadata-only fixture is enough.
    #[test]
    fn load_session_detects_nsnet2_as_denoise_task() {
        let (_session, task) = with_arch_only_gguf("nsnet2", "nsnet2-arch", |p| {
            load_session(p).expect("nsnet2 session builds (bare)")
        });
        assert_eq!(task, ModelTask::Denoise);

        let (_session, task) = with_arch_only_gguf("rnnoise", "rnnoise-arch", |p| {
            load_session(p).expect("rnnoise session builds (bare)")
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

    #[test]
    fn load_session_routes_openwakeword_to_kws_task() {
        let (_session, task) =
            with_arch_only_gguf(ARCH_OPENWAKEWORD_OP, "openwakeword-routed", |path| {
                load_session(path).expect("openwakeword returns a bare KWS session")
            });
        assert_eq!(task, ModelTask::KwsOpenwakeword);
        assert!(
            BOUND_ARCHES
                .iter()
                .all(|row| row.arch != ARCH_OPENWAKEWORD_OP),
            "a runnable KWS arch must not remain in the loud-partial registry"
        );
    }

    #[test]
    fn load_session_routes_smart_turn_to_endpoint_task() {
        let (_session, task) = with_arch_only_gguf(ARCH_SMART_TURN, "smart-turn-routed", |path| {
            load_session(path).expect("smart-turn returns a bare endpoint session")
        });
        assert_eq!(task, ModelTask::SmartTurn);
        assert!(
            BOUND_ARCHES.iter().all(|row| row.arch != ARCH_SMART_TURN),
            "a runnable endpoint arch must not remain in the loud-partial registry"
        );
    }

    #[test]
    fn load_session_detects_fcpe_crepe_and_nkf_tasks() {
        for (arch, tag, expected) in [
            (ARCH_FCPE, "fcpe-routed", ModelTask::F0Fcpe),
            (ARCH_CREPE, "crepe-routed", ModelTask::F0Crepe),
            (ARCH_NKF_AEC, "nkf-aec-routed", ModelTask::AecNkf),
            (ARCH_CHARSIU, "charsiu-routed", ModelTask::AlignCharsiu),
        ] {
            let (_session, task) = with_arch_only_gguf(arch, tag, |p| {
                load_session(p).expect("newly routed session builds (bare)")
            });
            assert_eq!(task, expected, "wrong task for `{arch}`");
        }
    }

    /// Task hints stay rejected on the newly routed arches (FR-EX-08 — no
    /// silent hint drop), same rule every other arch follows.
    #[test]
    fn load_session_rejects_hint_on_newly_routed_arches() {
        for (arch, tag) in [
            ("nsnet2", "nsnet2-hint"),
            ("pyannote-segmentation", "pyannote-hint"),
            ("rmvpe", "rmvpe-hint"),
            ("fcpe", "fcpe-hint"),
            ("crepe", "crepe-hint"),
            ("wetextprocessing", "wetext-hint"),
            ("nkf_aec", "nkf-aec-hint"),
            ("fsmn-vad", "fsmn-hint"),
            (ARCH_CHARSIU, "charsiu-hint"),
            (ARCH_WHISPER_MEDUSA_V1, "whisper-medusa-hint"),
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

    /// Zonos moved from the last no-loader slice to a strict manifest probe;
    /// an arch-only GGUF must surface the binder failure and the remaining
    /// forward class rather than the retired synthesized-only diagnosis.
    #[test]
    fn load_session_zonos_uses_strict_probe_and_loud_partial_diagnostic() {
        let err = with_arch_only_gguf("zonos", "zonos-arch", |p| {
            let Err(e) = load_session(p) else {
                panic!("zonos has no run task");
            };
            e
        });
        assert!(
            err.contains("FAILED — the binder reports:") && err.contains("loud-partial"),
            "must report the strict binder outcome and remaining forward class: {err}"
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

    /// CrisperWhisper — the round-trip case.
    ///
    /// `vokra-cli convert --model crisperwhisper` accepts four spellings and
    /// stamps `crisper-whisper`, and `vokra_models::whisper::ACCEPTED_ARCHS`
    /// binds it — but until 2026-08-15 this dispatch matched `ARCH_WHISPER`
    /// alone, so `run` refused the GGUF `convert` had just produced. Neither
    /// arch gate could see it: the converter constant is `pub(crate)`, and on
    /// the binder side it is a member of an aggregate `ACCEPTED_ARCHS` slice
    /// rather than its own `pub const`. A test is the only thing standing
    /// between this and a silent re-break.
    #[test]
    fn load_session_routes_crisper_whisper_to_the_whisper_asr_task() {
        assert_routed_to_whisper_asr("crisper-whisper", "crisper-whisper-arch");
    }

    #[test]
    fn load_session_routes_whisper_medusa_to_its_strict_asr_binder() {
        let error = with_arch_only_gguf(ARCH_WHISPER_MEDUSA_V1, "whisper-medusa-routed", |path| {
            let Err(error) = load_session(path) else {
                panic!("metadata-only Whisper-Medusa cannot bind");
            };
            error
        });
        assert!(
            error.contains("vokra.medusa.revision"),
            "route must reach the strict Medusa metadata reader: {error}"
        );
        assert!(!error.contains("unsupported model arch"), "{error}");
        assert!(!error.contains("is BOUND"), "{error}");
        assert!(
            BOUND_ARCHES
                .iter()
                .all(|row| row.arch != ARCH_WHISPER_MEDUSA_V1)
        );
    }

    /// The registry must not carry either arch again. `assert_routed_to_whisper_asr`
    /// checks the message a user sees; this checks the data behind it, so a
    /// re-added row fails here with a direct explanation rather than only as a
    /// downstream symptom.
    #[test]
    fn bound_arch_registry_excludes_routed_asr_forwards() {
        for arch in [
            ARCH_DISTIL_WHISPER,
            ARCH_KOTOBA_WHISPER,
            ARCH_MOONSHINE,
            ARCH_PARAKEET_TDT,
            ARCH_PARAKEET_CTC,
            ARCH_WHISPER_MEDUSA_V1,
        ] {
            assert!(
                BOUND_ARCHES.iter().all(|b| b.arch != arch),
                "`{arch}` has a real ASR forward and is \
                 routed to ModelTask::Asr — a BOUND_ARCHES row for it is both unreachable \
                 and untrue"
            );
        }
    }

    // ---- Wave 3 (2026-08-21) — Charsiu real artifact route ---------------

    /// Charsiu now has a converter, a strict GGUF binder and a paired
    /// audio/phone CLI contract.  The metadata-only fixture is sufficient to
    /// prove dispatch because the concrete binder intentionally runs in the
    /// `run` arm, after all paired inputs have been validated.
    #[test]
    fn load_session_routes_charsiu_to_alignment_task() {
        let (_session, task) = with_arch_only_gguf(ARCH_CHARSIU, "charsiu-routed", |p| {
            load_session(p).expect("charsiu session builds (bare)")
        });
        assert_eq!(task, ModelTask::AlignCharsiu);
    }

    #[test]
    fn load_session_routes_firered_vad_to_native_vad_loader() {
        let error = with_arch_only_gguf(ARCH_FIRERED_VAD, "firered-vad-routed", |path| {
            let Err(error) = load_session(path) else {
                panic!("metadata-only FireRedVAD GGUF must fail its tensor gate");
            };
            error
        });
        assert!(error.contains("zero tensors"), "{error}");
        assert!(!error.contains("is BOUND"), "{error}");
        assert!(
            BOUND_ARCHES.iter().all(|row| row.arch != ARCH_FIRERED_VAD),
            "FireRedVAD has a real native forward and CLI VAD route"
        );
    }

    #[test]
    fn load_session_routes_ten_vad_to_native_vad_loader() {
        let error = with_arch_only_gguf(ARCH_TEN_VAD, "ten-vad-routed", |path| {
            let Err(error) = load_session(path) else {
                panic!("metadata-only TEN-VAD GGUF must fail its strict metadata gate");
            };
            error
        });
        assert!(error.contains("vokra.ten_vad.revision"), "{error}");
        assert!(!error.contains("is BOUND"), "{error}");
        assert!(
            BOUND_ARCHES.iter().all(|row| row.arch != ARCH_TEN_VAD),
            "TEN-VAD has a real native forward and CLI VAD route"
        );
    }

    #[test]
    fn bound_arch_registry_excludes_routed_charsiu() {
        assert!(
            BOUND_ARCHES.iter().all(|row| row.arch != ARCH_CHARSIU),
            "Charsiu has a real GGUF loader and a ModelTask::AlignCharsiu route"
        );
    }

    #[test]
    fn load_session_routes_bigvgan_to_vocoder_task() {
        let (_session, task) = with_arch_only_gguf(ARCH_BIGVGAN, "bigvgan-routed", |path| {
            load_session(path).expect("BigVGAN session builds (bare)")
        });
        assert_eq!(task, ModelTask::VocoderBigVgan);
        assert!(
            BOUND_ARCHES.iter().all(|row| row.arch != ARCH_BIGVGAN),
            "BigVGAN has a strict real-weight loader, complete forward, and CLI route"
        );
    }

    #[test]
    fn load_session_routes_speecht5_hifigan_to_vocoder_task() {
        let (_session, task) =
            with_arch_only_gguf(ARCH_SPEECHT5_HIFIGAN, "speecht5-hifigan-routed", |path| {
                load_session(path).expect("SpeechT5 HiFi-GAN session builds (bare)")
            });
        assert_eq!(task, ModelTask::VocoderHifiGan);
        assert!(
            BOUND_ARCHES
                .iter()
                .all(|row| row.arch != ARCH_SPEECHT5_HIFIGAN),
            "SpeechT5 HiFi-GAN has a strict real-weight loader, complete forward, and CLI route"
        );
    }

    #[test]
    fn load_session_routes_speechbrain_hifigan_to_vocoder_task() {
        let (_session, task) =
            with_arch_only_gguf(ARCH_HIFIGAN_VOCODER, "hifigan-vocoder-routed", |path| {
                load_session(path).expect("SpeechBrain HiFi-GAN session builds (bare)")
            });
        assert_eq!(task, ModelTask::VocoderHifiGan);
        assert!(
            BOUND_ARCHES
                .iter()
                .all(|row| row.arch != ARCH_HIFIGAN_VOCODER),
            "SpeechBrain HiFi-GAN has a strict real-weight loader, complete forward, and CLI route"
        );
    }

    #[test]
    fn load_session_routes_vocos_to_vocoder_task() {
        let (_session, task) = with_arch_only_gguf(ARCH_VOCOS, "vocos-routed", |path| {
            load_session(path).expect("Vocos session builds (bare)")
        });
        assert_eq!(task, ModelTask::VocoderVocos);
        assert!(
            BOUND_ARCHES.iter().all(|row| row.arch != ARCH_VOCOS),
            "Vocos has strict loaders for both variants, a real forward, and a CLI route"
        );
    }

    // ---- Wave 1 (2026-08-21) — newly routed small runtime surfaces -------

    /// WeTextProcessing now resolves to a real text task. Binding the grammar
    /// and running its feature-gated FST pipeline happens in `run`, so an
    /// arch-only GGUF is sufficient to prove dispatch ownership here.
    #[test]
    fn load_session_routes_wetextprocessing_to_text_normalize() {
        let (_session, task) = with_arch_only_gguf(ARCH_WETEXTPROCESSING, "wetext-routed", |p| {
            load_session(p).expect("wetextprocessing session builds (bare)")
        });
        assert_eq!(task, ModelTask::TextNormalize);
    }

    /// Wave 2: both text surfaces are routed, but retain their distinct input
    /// contracts. CT-Punc's paired token/id signature is what the versioned
    /// TSV adapter feeds; it is not silently tokenized from `--text`.
    #[test]
    fn text_shaped_entry_points_are_routed_with_their_real_signatures() {
        // Text in, text out: no output shape blocks a `run` task here.
        const _: fn(
            &vokra_models::wetextprocessing::WeTextProcessing,
            &str,
        ) -> vokra_core::Result<String> =
            vokra_models::wetextprocessing::WeTextProcessing::normalize;
        // A caller-supplied tokenization AND its ids, paired by the CLI TSV.
        const _: fn(&vokra_models::ct_punc::CtPunc, &[&str], &[u32]) -> vokra_core::Result<String> =
            vokra_models::ct_punc::CtPunc::restore;

        assert!(
            BOUND_ARCHES
                .iter()
                .all(|b| b.arch != ARCH_WETEXTPROCESSING && b.arch != ARCH_CT_PUNC),
            "both routed text arches must be absent from the unreachable registry"
        );

        let (_session, task) = with_arch_only_gguf(ARCH_CT_PUNC, "ct-punc-routed", |p| {
            load_session(p).expect("ct-punc session builds (bare)")
        });
        assert_eq!(task, ModelTask::CtPunc);
    }

    #[test]
    fn load_session_routes_mimi_to_the_standalone_codec_task() {
        let (_session, task) = with_arch_only_gguf(ARCH_MIMI, "mimi-routed", |p| {
            load_session(p).expect("mimi session builds (bare)")
        });
        assert_eq!(task, ModelTask::MimiCodec);
        assert!(
            BOUND_ARCHES.iter().all(|b| b.arch != ARCH_MIMI),
            "the routed standalone codec must not retain a registry row"
        );
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
            ARCH_MOONSHINE,
            ARCH_PARAKEET_TDT,
            ARCH_PARAKEET_CTC,
            ARCH_WHISPER_MEDUSA_V1,
            ARCH_SILERO_VAD,
            ARCH_PIPER_PLUS,
            ARCH_CSM,
            ARCH_MOSHI,
            ARCH_CAMPPLUS,
            ARCH_VOXTRAL,
            ARCH_KOKORO,
            ARCH_SBV2,
            ARCH_FSMN_VAD,
            ARCH_FIRERED_VAD,
            ARCH_SMART_TURN,
            ARCH_NSNET2,
            ARCH_RNNOISE,
            ARCH_PYANNOTE_SEGMENTATION,
            ARCH_RMVPE,
            ARCH_FCPE,
            ARCH_CREPE,
            ARCH_WETEXTPROCESSING,
            ARCH_NKF_AEC,
            ARCH_CT_PUNC,
            ARCH_MIMI,
            ARCH_BIGVGAN,
            ARCH_SPEECHT5_HIFIGAN,
            ARCH_HIFIGAN_VOCODER,
            ARCH_VOCOS,
            ARCH_CHARSIU,
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
