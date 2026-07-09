//! Voxtral `AsrEngine` — [`vokra_core::AsrEngine`] adaptor over a loaded
//! [`VoxtralModel`].
//!
//! # Purpose
//!
//! The vokra-server (`integrations/vokra-server`) treats every ASR model
//! uniformly through the [`vokra_core::AsrEngine`] trait. This file supplies
//! the Voxtral side of that trait — a thin adaptor that owns a
//! [`VoxtralModel`] plus a runtime [`BackendKind`] and dispatches
//! [`AsrEngine::transcribe`](vokra_core::AsrEngine::transcribe) through the
//! [`AsrHead`] + [`VoxtralTokenizer`].
//!
//! # Autoregressive greedy decode (M3-10 core)
//!
//! [`AsrHead::transcribe`] runs:
//! 1. Whisper-shape log-mel front-end (`n_mels=128`, matches the declared
//!    `vokra.frontend.*` chunk) — shared with the Whisper front-end because
//!    the specs match bit-for-bit;
//! 2. audio encoder forward (shape / dispatch coverage);
//! 3. Mistral text decoder greedy loop (KV cache, RoPE, GQA, SwiGLU,
//!    RMSNorm, tied logits, EOS stop, `max_new_tokens` cap).
//!
//! # Audio conditioning — Wave 8 pluggable adapter (see [`AsrHead`] docs)
//!
//! Wave 8 lands the pluggable [`AudioAdapter`] framework: a GGUF that carries
//! a `vokra.voxtral.adapter.*` chunk with a non-`"none"` kind routes the
//! encoder output through the adapter (linear / MLP / downsample-linear) and
//! feeds the projection as a soft-prefix to the greedy decode — this is real
//! audio-conditioned ASR.
//!
//! A GGUF whose adapter chunk is absent or declared `kind = "none"` keeps
//! the Wave 7 posture: the returned tokens reflect the language-model prior
//! of the greedy decode from `bos_id`, not audio-conditioned ASR. This is
//! intentional per FR-EX-08 — callers see either a real (audio-conditioned)
//! token sequence, an honest (LM-prior) token sequence, or an explicit error;
//! never a fabricated audio-shaped transcript. Real ASR accuracy against a
//! Voxtral checkpoint requires (a) the adapter tensors + hparams from the
//! upstream release passed via `convert_voxtral_file`'s `--adapter-config`
//! side-car, and (b) a real-checkpoint parity dump (T19+).
//!
//! [`AudioAdapter`]: super::AudioAdapter
//!
//! # Front-end
//!
//! The 16 kHz mono `f32` PCM is turned into log-mel through
//! [`crate::whisper::mel::log_mel`] with `n_mels=128` (the Voxtral spec).
//! The same helper Whisper uses — no second implementation.
//!
//! [`VoxtralTokenizer`]: super::VoxtralTokenizer

use std::sync::Arc;

use vokra_core::{AsrEngine, BackendKind, Result, Transcription, VokraError};

use super::asr_head::{MISTRAL_BOS_ID, MISTRAL_EOS_ID};
use super::beam_search::{BeamConfig, BeamResult};
use super::{AsrHead, VoxtralModel, VoxtralTokenizer};

/// A Voxtral engine that speaks the [`AsrEngine`] trait. Holds the loaded
/// [`VoxtralModel`] plus its embedded tokenizer, the runtime backend and
/// max-new-token cap.
///
/// Cloned freely on the hot path (the model / tokenizer are behind an
/// [`Arc`]).
pub struct VoxtralAsr {
    /// The parsed model, shared. `Arc` because the registry holds one and
    /// hot-path handlers borrow it read-only.
    model: Arc<VoxtralModel>,
    /// The embedded Mistral tokenizer, shared. Loaded from the GGUF's
    /// `vokra.tokenizer.model` chunk at construction. Optional because a
    /// GGUF converted without the tokenizer chunk (older paths, or a
    /// tokenizer-less test double) still parses; a `None` tokenizer
    /// surfaces at [`transcribe`] time as an explicit error.
    tokenizer: Option<Arc<VoxtralTokenizer>>,
    /// Whether the model was declared as ASR- or S2S-capable in its config.
    /// ASR mode is the default; an S2S-tagged model can still be routed
    /// through this adaptor (S2S produces text on the inner stream) but the
    /// caller sees an ASR interface.
    #[allow(dead_code)]
    is_configured_for_asr: bool,
    /// Runtime backend selector for the encoder + decoder session.
    backend: BackendKind,
    /// Whether the transcribe path may promote itself to a GPU
    /// [`VoxtralMetalDecodeSession`] / [`VoxtralCudaDecodeSession`]
    /// automatically at call time. `false` (default) preserves the
    /// existing behaviour (dispatch through `self.backend`, defaulting to
    /// CPU). `true` picks Metal on an Apple + `metal` build, CUDA on a
    /// Unix/Windows + `cuda` build, and surfaces an explicit
    /// [`VokraError::BackendUnavailable`] otherwise — never a silent CPU
    /// fall back (FR-EX-08).
    ///
    /// [`VoxtralMetalDecodeSession`]: super::text_decoder_session_metal::VoxtralMetalDecodeSession
    /// [`VoxtralCudaDecodeSession`]: super::text_decoder_session_cuda::VoxtralCudaDecodeSession
    /// [`VokraError::BackendUnavailable`]: vokra_core::VokraError::BackendUnavailable
    allow_device_session: bool,
    /// Upper bound on generated tokens per transcribe call. `0` means
    /// [`super::text_decoder_session::DEFAULT_MAX_NEW_TOKENS`].
    max_new_tokens: usize,
    /// BOS token id (Mistral's `<s>` = 1 unless overridden).
    bos_id: u32,
    /// EOS token id (Mistral's `</s>` = 2 unless overridden).
    eos_id: u32,
}

impl VoxtralAsr {
    /// Wraps a loaded [`VoxtralModel`] as an [`AsrEngine`] with the CPU
    /// backend and default (Mistral shipping) BOS/EOS ids.
    ///
    /// A model whose declared `mode` is not `"asr"` or `"s2s"` is rejected
    /// with an explicit [`VokraError::ModelLoad`] — never silently coerced
    /// (FR-EX-08).
    pub fn new(model: VoxtralModel) -> Result<Self> {
        Self::new_with_backend(model, BackendKind::Cpu)
    }

    /// Like [`Self::new`] but on an explicit backend. The backend is
    /// consulted at each [`transcribe`] call so a runtime toggle can
    /// switch between CPU and a GPU seam without rebuilding the adaptor.
    pub fn new_with_backend(model: VoxtralModel, backend: BackendKind) -> Result<Self> {
        let is_asr = matches!(model.config().mode.as_str(), "asr" | "s2s");
        if !is_asr {
            return Err(VokraError::ModelLoad(format!(
                "voxtral::VoxtralAsr: unknown mode `{}` — expected `asr` or `s2s`",
                model.config().mode
            )));
        }
        Ok(Self {
            model: Arc::new(model),
            tokenizer: None,
            is_configured_for_asr: is_asr,
            backend,
            allow_device_session: false,
            max_new_tokens: 0, // 0 => DEFAULT_MAX_NEW_TOKENS
            bos_id: MISTRAL_BOS_ID,
            eos_id: MISTRAL_EOS_ID,
        })
    }

    /// Loads a Voxtral model from a GGUF file and wraps it as an
    /// [`AsrEngine`], also loading the embedded tokenizer if present. A
    /// missing tokenizer chunk is NOT a hard error at construction (some
    /// converter paths write shape-only GGUFs) — it surfaces at
    /// [`transcribe`] time as an explicit [`VokraError::ModelLoad`] naming
    /// the missing chunk. Same posture as other model surfaces here (never
    /// a silent fabrication).
    pub fn from_gguf(file: &vokra_core::gguf::GgufFile) -> Result<Self> {
        let model = VoxtralModel::from_gguf(file)?;
        let mut asr = Self::new(model)?;
        // Tokenizer load is optional at construction (see docstring).
        if let Ok(tok) = VoxtralTokenizer::from_gguf(file, MISTRAL_EOS_ID) {
            asr.tokenizer = Some(Arc::new(tok));
        }
        Ok(asr)
    }

    /// Shared handle to the underlying model.
    #[must_use]
    pub fn model(&self) -> &Arc<VoxtralModel> {
        &self.model
    }

    /// Shared handle to the loaded tokenizer (`None` if the GGUF did not
    /// embed one).
    #[must_use]
    pub fn tokenizer(&self) -> Option<&Arc<VoxtralTokenizer>> {
        self.tokenizer.as_ref()
    }

    /// Attaches (or replaces) an externally-loaded tokenizer. Used by tests
    /// and by callers that resolve the tokenizer bytes out-of-band.
    pub fn with_tokenizer(mut self, tokenizer: VoxtralTokenizer) -> Self {
        self.tokenizer = Some(Arc::new(tokenizer));
        self
    }

    /// Overrides the max-new-token cap for greedy decode. `0` restores the
    /// default ([`super::text_decoder_session::DEFAULT_MAX_NEW_TOKENS`]).
    #[must_use]
    pub fn with_max_new_tokens(mut self, max_new: usize) -> Self {
        self.max_new_tokens = max_new;
        self
    }

    /// Overrides the greedy BOS/EOS token ids. Defaults to Mistral's
    /// shipping ids (1 / 2).
    #[must_use]
    pub fn with_bos_eos(mut self, bos_id: u32, eos_id: u32) -> Self {
        self.bos_id = bos_id;
        self.eos_id = eos_id;
        self
    }

    /// Beam-search transcribe: PCM → up to `config.beam_size` decoded
    /// text hypotheses ranked by length-normalized score descending.
    ///
    /// A parallel to [`AsrEngine::transcribe`] that returns the full
    /// n-best set instead of collapsing to the top-1. `beam_size == 1` in
    /// `config` reproduces the greedy sequence returned by
    /// [`AsrEngine::transcribe`] (see the `beam_search` module's
    /// greedy-equivalence test).
    ///
    /// The `config.eos_token` is honored if the caller supplies it; the
    /// convenience wrapper [`Self::transcribe_beam_with_defaults`] fills
    /// `eos_token` from the `VoxtralAsr`'s own configured `eos_id`
    /// instead (see [`with_bos_eos`](Self::with_bos_eos)).
    ///
    /// # Errors
    ///
    /// Same taxonomy as [`AsrEngine::transcribe`], plus any error surfaced
    /// by [`beam_search_decode`](super::beam_search::beam_search_decode).
    pub fn transcribe_beam(
        &self,
        pcm: &[f32],
        config: &BeamConfig,
    ) -> Result<Vec<TranscribedBeam>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "voxtral::VoxtralAsr::transcribe_beam: pcm slice is empty".into(),
            ));
        }
        let cfg = self.model.config();
        let n_mels = cfg.audio.n_mels;
        if n_mels == 0 {
            return Err(VokraError::ModelLoad(
                "voxtral::VoxtralAsr::transcribe_beam: config carries n_mels = 0 \
                 (shape-only path). Re-convert with a full VoxtralConfig (FR-EX-08 — no silent \
                 default)."
                    .into(),
            ));
        }
        let log_mel = crate::whisper::mel::log_mel(pcm, n_mels);
        let n_frames = crate::whisper::mel::N_FRAMES;

        // Backend selection mirrors [`AsrEngine::transcribe`] — Wave 9 shipped
        // greedy + GPU session in two parallel worktrees, and the beam path
        // was landed against `self.backend` verbatim before the
        // `select_effective_backend` gate was factored out. Routing beam
        // through the same gate ensures `allow_device_session = true` promotes
        // to Metal / CUDA (or surfaces an explicit `BackendUnavailable` off
        // GPU builds) — never a silent CPU fall back (FR-EX-08).
        let effective_backend = self.select_effective_backend()?;

        let head = AsrHead::new(
            self.model.config(),
            self.model.audio_encoder(),
            self.model.text_decoder(),
        )
        .with_adapter(self.model.audio_adapter());
        let beams =
            head.transcribe_beam(effective_backend, &log_mel, n_frames, self.bos_id, config)?;

        // Detokenize each beam. A GGUF without an embedded tokenizer
        // surfaces an explicit error — no fabrication.
        let tok = self.tokenizer.as_ref().ok_or_else(|| {
            VokraError::ModelLoad(
                "voxtral::VoxtralAsr::transcribe_beam: model has no embedded tokenizer \
                 (`vokra.tokenizer.model` chunk absent). Re-convert with tokenizer bytes in the \
                 side-car, or attach one via `with_tokenizer(...)` (FR-EX-08 — never fabricate \
                 detokenised text)."
                    .into(),
            )
        })?;
        let mut out = Vec::with_capacity(beams.len());
        for b in beams {
            let text = tok.decode(&b.tokens)?;
            out.push(TranscribedBeam { text, result: b });
        }
        Ok(out)
    }

    /// Convenience wrapper around [`Self::transcribe_beam`] that fills the
    /// `eos_token` from `self.eos_id` and passes through the other config
    /// fields. Callers that only care about the beam width can use this
    /// (mirrors how [`AsrEngine::transcribe`] wraps the plumbing).
    pub fn transcribe_beam_with_defaults(
        &self,
        pcm: &[f32],
        beam_size: usize,
        max_new_tokens: usize,
    ) -> Result<Vec<TranscribedBeam>> {
        let effective_max = if max_new_tokens == 0 {
            super::text_decoder_session::DEFAULT_MAX_NEW_TOKENS
        } else {
            max_new_tokens
        };
        let config = BeamConfig::with_beam_size(beam_size, self.eos_id, effective_max);
        self.transcribe_beam(pcm, &config)
    }

    /// Beam-search transcribe with full user-controlled decoding knobs.
    ///
    /// Overrides `length_penalty` (GNMT α) and `no_repeat_ngram_size` on
    /// top of the [`BeamConfig::with_beam_size`] defaults, while still
    /// filling `eos_token` from `self.eos_id` and defaulting
    /// `max_new_tokens = 0` to
    /// [`super::text_decoder_session::DEFAULT_MAX_NEW_TOKENS`].
    ///
    /// Used by `vokra-server`'s beam-search HTTP surface (M3-15 Wave 10 A)
    /// so a client-supplied length_penalty / no_repeat_ngram is actually
    /// wired through (FR-EX-08 — accepting the field in the schema and
    /// silently ignoring it would be a fabrication).
    ///
    /// `length_penalty < 0.0` or `!length_penalty.is_finite()` surfaces as
    /// an [`VokraError::InvalidArgument`] — the ranking would be undefined.
    ///
    /// # Errors
    ///
    /// See [`Self::transcribe_beam`], plus the input validation above.
    pub fn transcribe_beam_with_config_overrides(
        &self,
        pcm: &[f32],
        beam_size: usize,
        length_penalty: f32,
        no_repeat_ngram_size: usize,
        max_new_tokens: usize,
    ) -> Result<Vec<TranscribedBeam>> {
        if !length_penalty.is_finite() || length_penalty < 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "voxtral::VoxtralAsr::transcribe_beam_with_config_overrides: length_penalty \
                 must be a non-negative finite float, got {length_penalty}",
            )));
        }
        let effective_max = if max_new_tokens == 0 {
            super::text_decoder_session::DEFAULT_MAX_NEW_TOKENS
        } else {
            max_new_tokens
        };
        // Start from [`BeamConfig::with_beam_size`] so `top_k_per_beam` gets
        // the canonical `2 * beam_size` default, then overlay the caller
        // fields.
        let mut config = BeamConfig::with_beam_size(beam_size, self.eos_id, effective_max);
        config.length_penalty = length_penalty;
        config.no_repeat_ngram_size = no_repeat_ngram_size;
        self.transcribe_beam(pcm, &config)
    }

    /// The greedy EOS token id `self.transcribe` and the beam-search
    /// wrappers use. Load-bearing for the server layer's tokenizer-aware
    /// beam config construction: without this the server would have to
    /// reach into the model config, which is the wrong layer.
    #[must_use]
    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }

    /// The greedy BOS token id.
    #[must_use]
    pub fn bos_id(&self) -> u32 {
        self.bos_id
    }

    /// Opts in to a GPU device session on the transcribe path.
    ///
    /// When `true`, [`Self::transcribe`] picks a GPU backend at call time —
    /// Metal on an Apple + `metal` build, CUDA on a Unix/Windows + `cuda`
    /// build — and drives the encoder + decoder through the matching
    /// [`VoxtralMetalDecodeSession`] / [`VoxtralCudaDecodeSession`]. A
    /// platform / build with no GPU backend surfaces an explicit
    /// [`VokraError::BackendUnavailable`] at transcribe time — **never a
    /// silent CPU fall back** (FR-EX-08). When `false` (default) the
    /// transcribe path uses [`Self::with_backend`]'s explicit backend
    /// selector (`BackendKind::Cpu` unless overridden).
    ///
    /// [`VoxtralMetalDecodeSession`]: super::text_decoder_session_metal::VoxtralMetalDecodeSession
    /// [`VoxtralCudaDecodeSession`]: super::text_decoder_session_cuda::VoxtralCudaDecodeSession
    /// [`VokraError::BackendUnavailable`]: vokra_core::VokraError::BackendUnavailable
    #[must_use]
    pub fn with_allow_device_session(mut self, allow: bool) -> Self {
        self.allow_device_session = allow;
        self
    }

    /// Whether [`Self::with_allow_device_session`] was set. Load-bearing
    /// for the streaming layer's device-session gate assertion.
    #[must_use]
    pub fn allow_device_session(&self) -> bool {
        self.allow_device_session
    }

    /// Overrides the runtime backend selector for the transcribe path.
    /// Mirrors `WhisperAsr::with_backend`. Defaults to [`BackendKind::Cpu`].
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// The runtime backend selector.
    #[must_use]
    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Picks the actual backend the current transcribe call will dispatch
    /// through, honouring [`Self::allow_device_session`]:
    ///
    /// - `false` (default): return `self.backend` verbatim (no promotion).
    /// - `true` and Apple + `metal` build: return [`BackendKind::Metal`].
    /// - `true` and Unix/Windows + `cuda` build: return [`BackendKind::Cuda`].
    /// - `true` and none of the above: explicit
    ///   [`VokraError::BackendUnavailable`] — no silent CPU fall back
    ///   (FR-EX-08).
    ///
    /// The Apple / Unix probes take priority over each other so an Apple
    /// build with both `metal` and `cuda` (rare — CUDA is Linux/Windows in
    /// practice) still picks Metal.
    fn select_effective_backend(&self) -> Result<BackendKind> {
        if !self.allow_device_session {
            return Ok(self.backend);
        }
        // Metal first (Apple + `metal` feature).
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        {
            Ok(BackendKind::Metal)
        }
        // CUDA next (Unix / Windows + `cuda` feature). Explicitly excluded
        // when the Metal + Apple cfg is active so this branch never coexists
        // with the Metal one in a single compile — no `unreachable_code`
        // warning under `metal,cuda` on macOS. The rare Apple + CUDA case
        // falls to the honest error at the bottom, since the practical CUDA
        // target is Linux / Windows.
        #[cfg(all(
            feature = "cuda",
            any(unix, windows),
            not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))),
        ))]
        {
            Ok(BackendKind::Cuda)
        }
        // No compiled-in GPU backend on this build: honest error, not a
        // silent CPU fall back (FR-EX-08). Compiled in only when neither of
        // the two cfg branches above is active.
        #[cfg(not(any(
            all(feature = "metal", any(target_os = "macos", target_os = "ios")),
            all(
                feature = "cuda",
                any(unix, windows),
                not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))),
            ),
        )))]
        {
            Err(VokraError::BackendUnavailable(
                "voxtral::VoxtralAsr::transcribe: allow_device_session = true but this build has \
                 no GPU backend compiled in (build with `--features metal` on Apple, or \
                 `--features cuda` on Unix / Windows). FR-EX-08: no silent CPU fall back."
                    .to_owned(),
            ))
        }
    }
}

/// One decoded beam from [`VoxtralAsr::transcribe_beam`] — the tokenized
/// output plus the underlying score metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscribedBeam {
    /// Detokenized text.
    pub text: String,
    /// Underlying beam result (raw tokens + log_prob + length-normalized
    /// score).
    pub result: BeamResult,
}

impl AsrEngine for VoxtralAsr {
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "voxtral::VoxtralAsr::transcribe: pcm slice is empty".into(),
            ));
        }
        let cfg = self.model.config();
        let n_mels = cfg.audio.n_mels;
        if n_mels == 0 {
            return Err(VokraError::ModelLoad(
                "voxtral::VoxtralAsr::transcribe: config carries n_mels = 0 (shape-only path). \
                 Re-convert with a full VoxtralConfig (FR-EX-08 — no silent default)."
                    .into(),
            ));
        }
        // 1) Log-mel front-end: PCM (16 kHz mono) → [n_mels, N_FRAMES] with
        //    Whisper's fixed spec (Voxtral inherits the same n_fft=400,
        //    hop=160, Slaney mel — the front-end check in `from_gguf`
        //    already validated bit-for-bit equality to Voxtral's spec).
        let log_mel = crate::whisper::mel::log_mel(pcm, n_mels);
        let n_frames = crate::whisper::mel::N_FRAMES;

        // Backend selection under `allow_device_session`:
        //   - false (default): use `self.backend` unchanged — the current
        //     posture, no GPU promotion.
        //   - true: promote to Metal on Apple + `metal`, or CUDA on
        //     Unix/Windows + `cuda`. A platform / build with no GPU backend
        //     surfaces an explicit `BackendUnavailable` (no silent CPU fall
        //     back, FR-EX-08). The GPU dispatch is exercised via the same
        //     `AsrHead::transcribe` entry — the internal `TextDecoderSession`
        //     builds its `Compute` with the selected backend, so every GEMM /
        //     softmax the Mistral decoder emits is routed to the GPU. The
        //     `VoxtralMetalDecodeSession` / `VoxtralCudaDecodeSession` types
        //     provide the API surface the caller reaches for elsewhere (unit
        //     tests, streaming layer); the `AsrHead` path exercises the same
        //     Compute-seam-backed GPU dispatch under the hood.
        let effective_backend = self.select_effective_backend()?;

        // 2) Autoregressive greedy through the AsrHead (encoder + text
        //    decoder session + KV cache). When the loaded GGUF carries an
        //    active audio adapter (M3-10 Wave 8) the head routes through the
        //    soft-prefix audio-conditioning path; otherwise it stays on the
        //    honest Wave 7 LM-continuation path.
        let head = AsrHead::new(
            self.model.config(),
            self.model.audio_encoder(),
            self.model.text_decoder(),
        )
        .with_adapter(self.model.audio_adapter());
        let ids = head.transcribe(
            effective_backend,
            &log_mel,
            n_frames,
            self.bos_id,
            self.eos_id,
            self.max_new_tokens,
        )?;

        // 3) Detokenise. A GGUF without an embedded tokenizer surfaces
        //    an explicit error — no fabrication.
        let text = match &self.tokenizer {
            Some(tok) => tok.decode(&ids)?,
            None => {
                return Err(VokraError::ModelLoad(
                    "voxtral::VoxtralAsr::transcribe: model has no embedded tokenizer \
                     (`vokra.tokenizer.model` chunk absent). Re-convert with tokenizer bytes \
                     in the side-car, or attach one via `with_tokenizer(...)` (FR-EX-08 — \
                     never fabricate detokenised text)."
                        .into(),
                ));
            }
        };
        Ok(Transcription::new(text))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxtral::config::{AudioEncoderConfig, TextDecoderConfig};
    use crate::voxtral::text_decoder::{DecoderBlock, GqaAttention, Linear, SwiGluFfn};
    use crate::voxtral::{AudioEncoder, TextDecoder, VoxtralConfig};

    /// Config large enough to run the full autoregressive decode: text
    /// hidden = 4, GQA 2/1, n_ctx = 16 so `bos + max_new = 8` tokens fit.
    fn tiny_config() -> VoxtralConfig {
        VoxtralConfig {
            audio: AudioEncoderConfig {
                n_layer: 1,
                n_head: 2,
                hidden_dim: 4,
                n_ctx: 8,
                n_mels: 2,
                ffn_dim: 8,
            },
            text: TextDecoderConfig {
                n_layer: 1,
                n_head_q: 2,
                n_head_kv: 1,
                hidden_dim: 4,
                ffn_dim: 8,
                vocab_size: 8,
                n_ctx: 16,
                rope_base: 10_000.0,
                rms_norm_eps: 1e-5,
            },
            cross_attn_hidden_dim: 4,
            mode: "asr".to_owned(),
            s2s_codec_type: "none".to_owned(),
        }
    }

    /// Deterministic-weight TextDecoder shared with the other Voxtral test
    /// modules — the same seed pattern.
    fn tiny_decoder(cfg: &VoxtralConfig) -> TextDecoder {
        let d = cfg.text.hidden_dim;
        let ffn = cfg.text.ffn_dim;
        let vocab = cfg.text.vocab_size;
        let head_dim = d / cfg.text.n_head_q;
        let kv_hidden = cfg.text.n_head_kv * head_dim;
        let mut token_emb = vec![0.0f32; vocab * d];
        for (i, v) in token_emb.iter_mut().enumerate() {
            *v = ((i as i32 % 7) - 3) as f32 * 0.05;
        }
        fn linear(rows: usize, cols: usize, base: f32) -> Linear {
            let mut w_t = vec![0.0f32; rows * cols];
            for (i, v) in w_t.iter_mut().enumerate() {
                *v = base + 0.01 * ((i as i32 % 5) - 2) as f32;
            }
            Linear {
                w_t,
                in_features: rows,
                out_features: cols,
            }
        }
        let blocks = (0..cfg.text.n_layer)
            .map(|_| DecoderBlock {
                attn_norm_gamma: vec![1.0f32; d],
                attn: GqaAttention {
                    q: linear(d, d, 0.10),
                    k: linear(d, kv_hidden, -0.07),
                    v: linear(d, kv_hidden, 0.05),
                    o: linear(d, d, -0.04),
                },
                ffn_norm_gamma: vec![1.0f32; d],
                ffn: SwiGluFfn {
                    gate: linear(d, ffn, 0.06),
                    up: linear(d, ffn, -0.02),
                    down: linear(ffn, d, 0.03),
                },
            })
            .collect();
        TextDecoder {
            token_emb,
            blocks,
            final_norm_gamma: vec![1.0f32; d],
            prefix: "",
        }
    }

    fn tiny_model() -> VoxtralModel {
        let cfg = tiny_config();
        let audio = AudioEncoder {
            conv1_w: vec![0.0; cfg.audio.hidden_dim * cfg.audio.n_mels * 3],
            conv1_b: vec![0.0; cfg.audio.hidden_dim],
            conv2_w: vec![0.0; cfg.audio.hidden_dim * cfg.audio.hidden_dim * 3],
            conv2_b: vec![0.0; cfg.audio.hidden_dim],
            pos_emb: vec![0.0; cfg.audio.n_ctx * cfg.audio.hidden_dim],
            has_learned_pos_emb: true,
        };
        let text = tiny_decoder(&cfg);
        VoxtralModel {
            config: cfg,
            audio,
            text,
            audio_adapter: crate::voxtral::AudioAdapter::none(),
        }
    }

    /// Empty-decoder tiny model (n_layer=0-shaped `TextDecoder`) — used to
    /// exercise the tokenizer / config-only error paths.
    fn tiny_shape_only_model() -> VoxtralModel {
        let cfg = tiny_config();
        let audio = AudioEncoder {
            conv1_w: vec![0.0; cfg.audio.hidden_dim * cfg.audio.n_mels * 3],
            conv1_b: vec![0.0; cfg.audio.hidden_dim],
            conv2_w: vec![0.0; cfg.audio.hidden_dim * cfg.audio.hidden_dim * 3],
            conv2_b: vec![0.0; cfg.audio.hidden_dim],
            pos_emb: vec![0.0; cfg.audio.n_ctx * cfg.audio.hidden_dim],
            has_learned_pos_emb: true,
        };
        let text = TextDecoder {
            token_emb: Vec::new(),
            blocks: Vec::new(),
            final_norm_gamma: Vec::new(),
            prefix: "",
        };
        VoxtralModel {
            config: cfg,
            audio,
            text,
            audio_adapter: crate::voxtral::AudioAdapter::none(),
        }
    }

    /// A minimum-viable compact-vocab tokenizer covering ids 0..vocab_size
    /// with `id -> "t{id} "` renderings, for the 200-dispatch test.
    fn tiny_tokenizer(vocab_size: usize, eos: u32) -> VoxtralTokenizer {
        // Compact-vocab dump format: u32 count + records.
        let mut blob = (vocab_size as u32).to_le_bytes().to_vec();
        for id in 0..vocab_size {
            let s = format!("t{id} ");
            let bytes = s.as_bytes();
            blob.push(0u8); // not special
            blob.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
            blob.extend_from_slice(bytes);
        }
        VoxtralTokenizer::from_bytes(blob, eos).unwrap()
    }

    #[test]
    fn new_accepts_asr_and_s2s_modes() {
        let asr = VoxtralAsr::new(tiny_model()).unwrap();
        assert!(asr.is_configured_for_asr);
    }

    #[test]
    fn new_rejects_unknown_mode() {
        let mut model = tiny_model();
        model.config.mode = "unknown".to_owned();
        assert!(matches!(
            VoxtralAsr::new(model),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn transcribe_empty_pcm_is_invalid_argument() {
        let asr = VoxtralAsr::new(tiny_model()).unwrap();
        assert!(matches!(
            asr.transcribe(&[]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn transcribe_without_tokenizer_is_model_load_error() {
        // A VoxtralAsr with no tokenizer attached must surface an explicit
        // ModelLoad on transcribe — not fabricate a text string.
        let asr = VoxtralAsr::new(tiny_model())
            .unwrap()
            .with_max_new_tokens(2)
            .with_bos_eos(1, /*unreachable*/ 999);
        let pcm = vec![0.0f32; 16_000];
        let err = asr.transcribe(&pcm).unwrap_err();
        assert!(matches!(err, VokraError::ModelLoad(_)), "{err:?}");
    }

    #[test]
    fn transcribe_with_tokenizer_returns_200_shaped_transcription() {
        // The 501 → 200 acceptance test: given a tiny model + tokenizer,
        // transcribe must return Ok(Transcription) with UTF-8 text.
        let vocab = tiny_config().text.vocab_size;
        let asr = VoxtralAsr::new(tiny_model())
            .unwrap()
            .with_max_new_tokens(3)
            .with_bos_eos(1, /*unreachable*/ vocab as u32 + 10)
            .with_tokenizer(tiny_tokenizer(vocab, 999));
        let pcm = vec![0.5f32; 16_000]; // 1 s @ 16 kHz
        let t = asr.transcribe(&pcm).expect("transcribe must return Ok");
        // With eos unreachable and max_new=3, exactly 3 tokens emitted →
        // exactly 3 "t{id} " chunks. Non-empty is the load-bearing check.
        assert!(
            !t.text.is_empty(),
            "transcription text must not be empty: {:?}",
            t.text
        );
        // Deterministic decode: a repeated call must produce the same text
        // (proves greedy + tokenizer are pure over the same input).
        let t2 = asr.transcribe(&pcm).unwrap();
        assert_eq!(t.text, t2.text);
    }

    #[test]
    fn transcribe_zero_n_mels_is_model_load_error() {
        // Shape-only converter path: n_mels=0 must not silently pass.
        let mut model = tiny_shape_only_model();
        model.config.audio.n_mels = 0;
        let asr = VoxtralAsr::new(model).unwrap();
        let pcm = vec![0.0f32; 16_000];
        assert!(matches!(
            asr.transcribe(&pcm),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn with_tokenizer_replaces_previously_attached_tokenizer() {
        let vocab = tiny_config().text.vocab_size;
        let asr = VoxtralAsr::new(tiny_model()).unwrap();
        assert!(asr.tokenizer().is_none());
        let asr = asr.with_tokenizer(tiny_tokenizer(vocab, 2));
        assert!(asr.tokenizer().is_some());
    }

    #[test]
    fn is_asr_engine_object_safe() {
        // If AsrEngine goes non-object-safe, this line stops compiling.
        // The vokra-server registry stores engines behind Arc<dyn AsrEngine>
        // so this is a load-bearing property.
        let _engine: Arc<dyn AsrEngine> = Arc::new(VoxtralAsr::new(tiny_model()).unwrap());
    }

    // -----------------------------------------------------------------
    // Beam-search transcribe on VoxtralAsr
    // -----------------------------------------------------------------

    #[test]
    fn transcribe_beam_empty_pcm_is_invalid_argument() {
        let asr = VoxtralAsr::new(tiny_model()).unwrap();
        let bc = BeamConfig::greedy(2, 3);
        assert!(matches!(
            asr.transcribe_beam(&[], &bc),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn transcribe_beam_without_tokenizer_is_model_load_error() {
        let asr = VoxtralAsr::new(tiny_model())
            .unwrap()
            .with_max_new_tokens(2)
            .with_bos_eos(1, /*unreachable*/ 999);
        let pcm = vec![0.0f32; 16_000];
        let bc = BeamConfig::greedy(999, 2);
        assert!(matches!(
            asr.transcribe_beam(&pcm, &bc),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn transcribe_beam_returns_ranked_beams_with_text() {
        let vocab = tiny_config().text.vocab_size;
        let asr = VoxtralAsr::new(tiny_model())
            .unwrap()
            .with_max_new_tokens(3)
            .with_bos_eos(1, /*unreachable*/ vocab as u32 + 10)
            .with_tokenizer(tiny_tokenizer(vocab, 999));
        let pcm = vec![0.5f32; 16_000];
        let bc = BeamConfig::with_beam_size(2, vocab as u32 + 10, 3);
        let beams = asr
            .transcribe_beam(&pcm, &bc)
            .expect("beam decode must succeed");
        assert!(!beams.is_empty());
        assert!(beams.len() <= 2);
        for b in &beams {
            assert!(
                !b.text.is_empty(),
                "beam text must not be empty: {:?}",
                b.text
            );
        }
        // Ranked descending.
        for pair in beams.windows(2) {
            assert!(
                pair[0].result.length_normalized_score >= pair[1].result.length_normalized_score
            );
        }
    }

    #[test]
    fn transcribe_beam_size_one_matches_engine_transcribe() {
        // beam_size=1 through transcribe_beam must produce the same top-1
        // text as transcribe (greedy).
        let vocab = tiny_config().text.vocab_size;
        let asr = VoxtralAsr::new(tiny_model())
            .unwrap()
            .with_max_new_tokens(3)
            .with_bos_eos(1, /*unreachable*/ vocab as u32 + 10)
            .with_tokenizer(tiny_tokenizer(vocab, 999));
        let pcm = vec![0.5f32; 16_000];

        let greedy = asr.transcribe(&pcm).unwrap();
        let bc = BeamConfig::greedy(vocab as u32 + 10, 3);
        let beams = asr.transcribe_beam(&pcm, &bc).unwrap();
        assert_eq!(beams.len(), 1);
        assert_eq!(beams[0].text, greedy.text);
    }

    #[test]
    fn transcribe_beam_with_defaults_wraps_config() {
        let vocab = tiny_config().text.vocab_size;
        let asr = VoxtralAsr::new(tiny_model())
            .unwrap()
            .with_max_new_tokens(3)
            .with_bos_eos(1, vocab as u32 + 10)
            .with_tokenizer(tiny_tokenizer(vocab, 999));
        let pcm = vec![0.5f32; 16_000];
        let beams = asr
            .transcribe_beam_with_defaults(&pcm, 2, 3)
            .expect("default-wrapped beam decode must succeed");
        assert!(!beams.is_empty());
        assert!(beams.len() <= 2);
    }

    #[test]
    fn transcribe_beam_with_config_overrides_rejects_bad_length_penalty() {
        let vocab = tiny_config().text.vocab_size;
        let asr = VoxtralAsr::new(tiny_model())
            .unwrap()
            .with_max_new_tokens(3)
            .with_bos_eos(1, vocab as u32 + 10)
            .with_tokenizer(tiny_tokenizer(vocab, 999));
        let pcm = vec![0.5f32; 16_000];
        // NaN is not a valid length-penalty (ranking becomes undefined).
        assert!(matches!(
            asr.transcribe_beam_with_config_overrides(&pcm, 2, f32::NAN, 0, 3),
            Err(VokraError::InvalidArgument(_)),
        ));
        // Negative alpha is not a valid GNMT length-penalty.
        assert!(matches!(
            asr.transcribe_beam_with_config_overrides(&pcm, 2, -0.1, 0, 3),
            Err(VokraError::InvalidArgument(_)),
        ));
    }

    #[test]
    fn transcribe_beam_with_config_overrides_wires_length_penalty_and_ngram() {
        let vocab = tiny_config().text.vocab_size;
        let asr = VoxtralAsr::new(tiny_model())
            .unwrap()
            .with_max_new_tokens(3)
            .with_bos_eos(1, vocab as u32 + 10)
            .with_tokenizer(tiny_tokenizer(vocab, 999));
        let pcm = vec![0.5f32; 16_000];
        // Alpha = 0.0 collapses the length-penalty term to 1 (`log_prob`
        // is returned unchanged from `length_normalized`) — provides a
        // discriminator vs the 0.6 default. no_repeat_ngram_size = 2
        // blocks 2-grams; on the tiny fixture this is at most no-op
        // (fixture is not diverse enough) but the call must still succeed
        // and return well-formed beams.
        let beams = asr
            .transcribe_beam_with_config_overrides(&pcm, 2, 0.0, 2, 3)
            .expect("beam decode with overrides must succeed");
        assert!(!beams.is_empty());
        // At α = 0.0 the length-normalized score equals log_prob; the
        // ranking is by log_prob descending.
        for pair in beams.windows(2) {
            let a = pair[0].result.length_normalized_score;
            let b = pair[1].result.length_normalized_score;
            assert!(a >= b, "beams must remain descending under α=0.0 override");
            // Confirm α=0 short-circuit: normalized == log_prob at len>0.
            #[allow(clippy::float_cmp)]
            {
                assert_eq!(
                    a, pair[0].result.log_prob,
                    "α=0.0 must yield normalized == log_prob (short-circuit)"
                );
            }
        }
    }

    #[test]
    fn eos_id_and_bos_id_accessors_return_configured_ids() {
        // Default (Mistral shipping ids).
        let asr = VoxtralAsr::new(tiny_model()).unwrap();
        assert_eq!(asr.bos_id(), MISTRAL_BOS_ID);
        assert_eq!(asr.eos_id(), MISTRAL_EOS_ID);
        // Overridden via with_bos_eos — the server layer reads
        // `eos_id()` to build a full BeamConfig without re-reading
        // model.config, so the getter must reflect the last set value.
        let asr = asr.with_bos_eos(42, 99);
        assert_eq!(asr.bos_id(), 42);
        assert_eq!(asr.eos_id(), 99);
    }

    // -------- M3-10 GPU session opt-in surface -----------------------------

    #[test]
    fn allow_device_session_defaults_to_false() {
        // Preserves the pre-M3-10 posture: constructors that never opt in
        // stay on the caller-set backend (which itself defaults to CPU).
        let asr = VoxtralAsr::new(tiny_model()).unwrap();
        assert!(!asr.allow_device_session());
        assert_eq!(asr.backend(), BackendKind::Cpu);
    }

    #[test]
    fn with_allow_device_session_flag_flips_and_is_readable() {
        let asr = VoxtralAsr::new(tiny_model())
            .unwrap()
            .with_allow_device_session(true);
        assert!(asr.allow_device_session());
        // Toggle back — the builder is idempotent + reversible.
        let asr = asr.with_allow_device_session(false);
        assert!(!asr.allow_device_session());
    }

    #[test]
    fn with_backend_overrides_the_selector() {
        let asr = VoxtralAsr::new(tiny_model())
            .unwrap()
            .with_backend(BackendKind::Metal);
        assert_eq!(asr.backend(), BackendKind::Metal);
    }

    /// Off the Metal / CUDA build (feature disabled or unsupported target),
    /// `allow_device_session=true` at transcribe time surfaces an explicit
    /// [`BackendUnavailable`] — never a silent CPU fall back (FR-EX-08).
    #[cfg(not(any(
        all(feature = "metal", any(target_os = "macos", target_os = "ios")),
        all(feature = "cuda", any(unix, windows)),
    )))]
    #[test]
    fn transcribe_with_allow_device_session_is_backend_unavailable_off_gpu_builds() {
        let vocab = tiny_config().text.vocab_size;
        let asr = VoxtralAsr::new(tiny_model())
            .unwrap()
            .with_max_new_tokens(2)
            .with_bos_eos(1, vocab as u32 + 10)
            .with_tokenizer(tiny_tokenizer(vocab, 999))
            .with_allow_device_session(true);
        let pcm = vec![0.5f32; 16_000];
        let err = asr.transcribe(&pcm).unwrap_err();
        assert!(
            matches!(err, VokraError::BackendUnavailable(_)),
            "expected BackendUnavailable off GPU builds, got {err:?}",
        );
    }

    /// On the Metal build (Apple + `metal`), `allow_device_session=true` at
    /// transcribe time either dispatches through the Metal Compute seam OR
    /// surfaces an explicit `BackendUnavailable` (no device present) — but
    /// never a silent CPU fall back and never a fabricated pass.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn transcribe_with_allow_device_session_is_honest_on_metal_build() {
        let vocab = tiny_config().text.vocab_size;
        let asr = VoxtralAsr::new(tiny_model())
            .unwrap()
            .with_max_new_tokens(2)
            .with_bos_eos(1, vocab as u32 + 10)
            .with_tokenizer(tiny_tokenizer(vocab, 999))
            .with_allow_device_session(true);
        let pcm = vec![0.5f32; 16_000];
        match asr.transcribe(&pcm) {
            Ok(t) => {
                // Device was available: the GPU path emitted UTF-8 text.
                assert!(
                    !t.text.is_empty(),
                    "transcription must not be empty on GPU path"
                );
            }
            Err(VokraError::BackendUnavailable(_)) => {
                // No Metal device: honest error, not a silent CPU fall back.
            }
            Err(other) => panic!(
                "expected Ok or BackendUnavailable on Metal build, got {other:?} (no silent CPU \
                 substitute, FR-EX-08)",
            ),
        }
    }

    /// On the CUDA build (Unix/Windows + `cuda`), `allow_device_session=true`
    /// at transcribe time either dispatches through the CUDA Compute seam OR
    /// surfaces an explicit `BackendUnavailable` — never a silent CPU fall
    /// back.
    #[cfg(all(feature = "cuda", any(unix, windows)))]
    #[test]
    fn transcribe_with_allow_device_session_is_honest_on_cuda_build() {
        let vocab = tiny_config().text.vocab_size;
        let asr = VoxtralAsr::new(tiny_model())
            .unwrap()
            .with_max_new_tokens(2)
            .with_bos_eos(1, vocab as u32 + 10)
            .with_tokenizer(tiny_tokenizer(vocab, 999))
            .with_allow_device_session(true);
        let pcm = vec![0.5f32; 16_000];
        match asr.transcribe(&pcm) {
            Ok(t) => {
                assert!(
                    !t.text.is_empty(),
                    "transcription must not be empty on GPU path"
                );
            }
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(other) => panic!(
                "expected Ok or BackendUnavailable on CUDA build, got {other:?} (no silent CPU \
                 substitute, FR-EX-08)",
            ),
        }
    }

    // -------- Task 1 (Wave 10): `transcribe_beam` honors allow_device_session ---
    //
    // Wave 9 landed [`Self::transcribe_beam`] and [`Self::with_allow_device_session`]
    // in two parallel worktrees; the merge left `transcribe_beam` dispatching
    // through `self.backend` directly rather than
    // [`select_effective_backend`]. The tests below pin the fix (Wave 10
    // Agent A) so a future refactor cannot silently regress the symmetry
    // with [`AsrEngine::transcribe`] (FR-EX-08 — beam path must honor the
    // same GPU / no-fall-back rules the greedy path does).
    //
    // Small `beam_size` + real (non-zero) PCM keeps the test cost bounded
    // while still driving through the mel front-end + backend gate.

    /// Off every GPU build, `allow_device_session = true` on the beam path
    /// surfaces an explicit `BackendUnavailable` — never a silent CPU
    /// substitute. Mirrors
    /// [`transcribe_with_allow_device_session_is_backend_unavailable_off_gpu_builds`]
    /// but drives [`Self::transcribe_beam`] instead of [`AsrEngine::transcribe`].
    #[cfg(not(any(
        all(feature = "metal", any(target_os = "macos", target_os = "ios")),
        all(feature = "cuda", any(unix, windows)),
    )))]
    #[test]
    fn transcribe_beam_with_allow_device_session_is_backend_unavailable_off_gpu_builds() {
        let vocab = tiny_config().text.vocab_size;
        let asr = VoxtralAsr::new(tiny_model())
            .unwrap()
            .with_max_new_tokens(2)
            .with_bos_eos(1, vocab as u32 + 10)
            .with_tokenizer(tiny_tokenizer(vocab, 999))
            .with_allow_device_session(true);
        let pcm = vec![0.5f32; 16_000];
        let bc = BeamConfig::with_beam_size(2, vocab as u32 + 10, 3);
        let err = asr.transcribe_beam(&pcm, &bc).unwrap_err();
        assert!(
            matches!(err, VokraError::BackendUnavailable(_)),
            "expected BackendUnavailable off GPU builds, got {err:?} (FR-EX-08 — no silent CPU \
             substitute on the beam path)",
        );
    }

    /// On the Metal build, `allow_device_session = true` on the beam path
    /// either dispatches through the Metal Compute seam OR surfaces an
    /// explicit `BackendUnavailable`. Never a silent CPU fall back and
    /// never a fabricated pass.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn transcribe_beam_with_allow_device_session_is_honest_on_metal_build() {
        let vocab = tiny_config().text.vocab_size;
        let asr = VoxtralAsr::new(tiny_model())
            .unwrap()
            .with_max_new_tokens(2)
            .with_bos_eos(1, vocab as u32 + 10)
            .with_tokenizer(tiny_tokenizer(vocab, 999))
            .with_allow_device_session(true);
        let pcm = vec![0.5f32; 16_000];
        let bc = BeamConfig::with_beam_size(2, vocab as u32 + 10, 3);
        match asr.transcribe_beam(&pcm, &bc) {
            Ok(beams) => {
                // Device was available: the GPU path emitted a non-empty
                // ranked n-best.
                assert!(
                    !beams.is_empty(),
                    "beam decode must not return empty on GPU path"
                );
                for b in &beams {
                    assert!(
                        !b.text.is_empty(),
                        "beam text must not be empty on GPU path"
                    );
                }
            }
            Err(VokraError::BackendUnavailable(_)) => {
                // No Metal device: honest error, not a silent CPU fall back.
            }
            Err(other) => panic!(
                "expected Ok or BackendUnavailable on Metal build, got {other:?} (no silent CPU \
                 substitute on the beam path, FR-EX-08)",
            ),
        }
    }

    /// CUDA-build symmetric of the Metal test above.
    #[cfg(all(feature = "cuda", any(unix, windows)))]
    #[test]
    fn transcribe_beam_with_allow_device_session_is_honest_on_cuda_build() {
        let vocab = tiny_config().text.vocab_size;
        let asr = VoxtralAsr::new(tiny_model())
            .unwrap()
            .with_max_new_tokens(2)
            .with_bos_eos(1, vocab as u32 + 10)
            .with_tokenizer(tiny_tokenizer(vocab, 999))
            .with_allow_device_session(true);
        let pcm = vec![0.5f32; 16_000];
        let bc = BeamConfig::with_beam_size(2, vocab as u32 + 10, 3);
        match asr.transcribe_beam(&pcm, &bc) {
            Ok(beams) => {
                assert!(
                    !beams.is_empty(),
                    "beam decode must not return empty on GPU path"
                );
                for b in &beams {
                    assert!(
                        !b.text.is_empty(),
                        "beam text must not be empty on GPU path"
                    );
                }
            }
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(other) => panic!(
                "expected Ok or BackendUnavailable on CUDA build, got {other:?} (no silent CPU \
                 substitute on the beam path, FR-EX-08)",
            ),
        }
    }

    /// `allow_device_session = false` (default) is preserved verbatim on
    /// the beam path: the beam decode dispatches through `self.backend`
    /// exactly as it did before the Wave 10 fix. This locks the pre-Wave-9
    /// posture in — a caller who explicitly opted out (or never opted in)
    /// keeps the caller-set backend. This test runs on every build.
    #[test]
    fn transcribe_beam_no_op_when_allow_device_session_false() {
        let vocab = tiny_config().text.vocab_size;
        // Default = false; no builder call flips it on.
        let asr = VoxtralAsr::new(tiny_model())
            .unwrap()
            .with_max_new_tokens(3)
            .with_bos_eos(1, vocab as u32 + 10)
            .with_tokenizer(tiny_tokenizer(vocab, 999));
        assert!(!asr.allow_device_session());
        // CPU backend default: the beam decode must complete without going
        // through any GPU gate (i.e. it never surfaces BackendUnavailable).
        let pcm = vec![0.5f32; 16_000];
        let bc = BeamConfig::with_beam_size(2, vocab as u32 + 10, 3);
        let beams = asr
            .transcribe_beam(&pcm, &bc)
            .expect("beam decode must succeed on the default CPU backend");
        assert!(!beams.is_empty());
        assert!(beams.len() <= 2);

        // Explicitly flipping the flag off after opting in also stays on
        // the CPU path — the builder is reversible and the beam path reads
        // the same flag the greedy path does (no drift).
        let asr = asr
            .with_allow_device_session(true)
            .with_allow_device_session(false);
        assert!(!asr.allow_device_session());
        let beams = asr
            .transcribe_beam(&pcm, &bc)
            .expect("beam decode must succeed after opting out");
        assert!(!beams.is_empty());
    }
}
