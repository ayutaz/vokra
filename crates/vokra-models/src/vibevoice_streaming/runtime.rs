//! Native VibeVoice Realtime streaming composition.
//!
//! This is the first module in the Realtime path that owns the complete
//! source-ordered composition.  It does not replace the language, diffusion,
//! connector, or acoustic implementations; it sequences those concrete
//! components around the authenticated four-output preset cache.
//!
//! The ordering is pinned to Microsoft's Realtime source at
//! `94da20d98b2fa7688e9cbfaf7692ddb4954f7600`,
//! `modeling_vibevoice_streaming_inference.py::generate`:
//!
//! 1. Consume a five-token text window with incremental text-LM and TTS-LM
//!    steps.
//! 2. For each of at most six speech steps, sample a 64-wide latent with
//!    positive/negative CFG, decode that scaled latent, and pass the *same
//!    scaled* latent to the acoustic connector.
//! 3. Append the connector output to both positive and negative TTS caches
//!    with token `1` and `is_text = false`, then classify EOS from the positive
//!    TTS output.
//!
//! The source's prompt cache is imported before this session starts.  The
//! session never calls the prefill methods after import: text is appended with
//! the incremental `step` APIs, and the negative text branch is never advanced
//! after its authenticated import.  A failed operation poisons and resets all
//! mutable generation state; callers cannot accidentally reuse a partial
//! cache.

use vokra_core::backend::BackendKind;
use vokra_core::{Result, VokraError};

use super::acoustic::{REALTIME_ACOUSTIC_CHUNK_SAMPLES, VibeVoiceRealtimeAcousticDecoderStream};
use super::connector::VibeVoiceRealtimeAcousticConnector;
use super::generation::VibeVoiceRealtimeGenerationStopReason;
use super::language::VibeVoiceRealtimeLanguage;
use super::preset::{VibeVoiceRealtimePresetBranch, VibeVoiceRealtimePresetCache};
use super::sampler::{VIBEVOICE_REALTIME_LATENT_WIDTH, sample_vibevoice_realtime_cfg};
use super::state::{TTS_SPEECH_WINDOW_SIZE, VibeVoiceStreamingState, VibeVoiceStreamingTextPlan};
use super::tokenizer::VibeVoiceRealtimeTokenizer;
use super::{HIDDEN, MAX_POSITIONS, VibeVoiceStreamingCheckpoint, VibeVoiceStreamingDiffusionHead};
use crate::compute::{Compute, HotOp};

/// Union of every learned operation used by the complete Realtime composite.
/// Host-side token/cache bookkeeping and the DPM scheduler are intentionally
/// absent: they are control stages, not learned backend operations.
const REALTIME_COMPOSITE_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::RmsNorm,
    HotOp::Silu,
    HotOp::Relu,
    HotOp::Conv1d,
    HotOp::ConvTranspose1d,
    HotOp::GroupedConv1d,
    HotOp::Gelu,
];

const SPEECH_TOKEN_ID: u32 = 1;

/// Explicit bounds and sampling controls for one Realtime synthesis session.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VibeVoiceRealtimeSynthesisConfig {
    /// Maximum number of new logical TTS positions (text plus speech).
    /// This is the caller's finite equivalent of the source generation
    /// `max_new_tokens` budget; it is not inferred from cache positions.
    pub max_new_tokens: usize,
    /// Maximum number of speech steps permitted by the caller.  This separate
    /// control bound prevents an unbounded zero-text continuation.
    pub max_speech_steps: usize,
    /// Classifier-free guidance scale used by the authenticated sampler.
    pub guidance_scale: f32,
}

impl VibeVoiceRealtimeSynthesisConfig {
    fn validate(self, initial_tts_hidden_rows: usize) -> Result<()> {
        if self.max_new_tokens == 0 || self.max_speech_steps == 0 {
            return Err(VokraError::InvalidArgument(
                "vibevoice realtime synthesis budgets must be non-zero".to_owned(),
            ));
        }
        if !self.guidance_scale.is_finite() {
            return Err(VokraError::InvalidArgument(
                "vibevoice realtime guidance scale must be finite".to_owned(),
            ));
        }
        if initial_tts_hidden_rows > MAX_POSITIONS
            || self.max_new_tokens > MAX_POSITIONS - initial_tts_hidden_rows
        {
            return Err(VokraError::InvalidArgument(
                "vibevoice realtime max_new_tokens exceeds the authenticated model position budget"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

/// One successfully decoded streaming audio chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct VibeVoiceRealtimeAudioChunk {
    /// Mono 24 kHz PCM samples from the causal acoustic decoder.
    pub pcm: Vec<f32>,
    /// Source text window that preceded this speech step, or `None` for the
    /// source's zero-text continuation.
    pub text_window_index: Option<usize>,
    /// Zero-based step within the source six-step speech window.
    pub speech_step: usize,
    /// Number of logical positions consumed after this step.
    pub generated_positions: usize,
    /// If set, this chunk observed a terminal condition.  For EOS the source
    /// still drains the remainder of its six-step inner loop; those suppressed
    /// cache steps are reported as [`VibeVoiceRealtimeSynthesisStep::Draining`].
    pub terminal_after: Option<VibeVoiceRealtimeGenerationStopReason>,
}

/// One externally observable result from a synthesis session.
#[derive(Debug, Clone, PartialEq)]
pub enum VibeVoiceRealtimeSynthesisStep {
    /// One decoded audio chunk.  The caller supplies a fresh noise vector on
    /// each call that returns this variant.
    Audio(VibeVoiceRealtimeAudioChunk),
    /// A source speech step executed after positive EOS was observed. The
    /// upstream six-step inner loop drains these cache updates but suppresses
    /// the corresponding audio-chunk append.
    Draining {
        /// Source text window index for the drained step.
        text_window_index: Option<usize>,
        /// Zero-based step within the six-step source window.
        speech_step: usize,
    },
    /// No more model work is permitted for this session.
    Finished {
        /// Exact observed terminal condition.
        reason: VibeVoiceRealtimeGenerationStopReason,
    },
}

/// Native, authenticated Realtime model components.
#[derive(Debug, Clone)]
pub struct VibeVoiceRealtimeRuntime {
    language: VibeVoiceRealtimeLanguage,
    diffusion_head: VibeVoiceStreamingDiffusionHead,
    connector: VibeVoiceRealtimeAcousticConnector,
    acoustic_decoder: super::acoustic::VibeVoiceRealtimeAcousticDecoder,
}

impl VibeVoiceRealtimeRuntime {
    /// Loads the complete native composition from one authenticated GGUF.
    ///
    /// The complete composition supports CPU and the Metal learned-op path.
    /// The backend is checked before any model tensor binding so an uncovered
    /// backend receives an explicit error and cannot trigger an implicit CPU
    /// path.  The sampler's DPM scheduler remains an explicit host-control
    /// stage; it does not move learned tensors off the selected backend.
    /// Dispatch coverage here is not real Metal hardware or numerical-parity
    /// evidence; those remain separate verification gates.
    pub fn from_gguf(file: &vokra_core::gguf::GgufFile, backend: BackendKind) -> Result<Self> {
        require_realtime_backend_before_binding(backend)?;
        // Probe the complete learned-op registry before authentication or any
        // component loads.  In particular this is a loud BackendUnavailable
        // on a non-Apple/feature-off Metal build, not a late error after
        // binding a large language-model tensor set.
        preflight_realtime_backend(backend)?;
        VibeVoiceStreamingCheckpoint::from_gguf(file)?;
        Ok(Self {
            language: VibeVoiceRealtimeLanguage::from_gguf(file, backend)?,
            diffusion_head: VibeVoiceStreamingDiffusionHead::from_gguf_with_backend(file, backend)?,
            connector: VibeVoiceRealtimeAcousticConnector::from_gguf(file, backend)?,
            acoustic_decoder: super::acoustic::VibeVoiceRealtimeAcousticDecoder::from_gguf(
                file, backend,
            )?,
        })
    }

    /// Returns the explicitly selected backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.language.backend()
    }

    /// Starts an authenticated session from fixed tokenizer sidecars and the
    /// official four-output preset cache.
    ///
    /// Input, budget, prompt lengths, and initial conditions are validated
    /// before the positive language cache is replaced.  This keeps malformed
    /// caller input from mutating an already-bound runtime.
    pub fn start_session<'a>(
        &'a mut self,
        preset: &VibeVoiceRealtimePresetCache,
        tokenizer: &VibeVoiceRealtimeTokenizer,
        text: &str,
        config: VibeVoiceRealtimeSynthesisConfig,
    ) -> Result<VibeVoiceRealtimeSynthesisSession<'a>> {
        let positive_tts = preset.output(VibeVoiceRealtimePresetBranch::TtsLm);
        let negative_tts = preset.output(VibeVoiceRealtimePresetBranch::NegTtsLm);
        let positive_condition =
            last_hidden_row(positive_tts.hidden(), positive_tts.hidden_rows())?;
        let negative_condition =
            last_hidden_row(negative_tts.hidden(), negative_tts.hidden_rows())?;
        let prompt = super::state::VibeVoiceStreamingPrompt::from_tokenizer(
            tokenizer,
            preset
                .output(VibeVoiceRealtimePresetBranch::Lm)
                .hidden_rows(),
            positive_tts.hidden_rows(),
        )?;
        let state = VibeVoiceStreamingState::new(prompt);
        let text_ids = tokenizer.streaming_text_ids(text)?;
        let plan = state.plan_text_windows(&text_ids)?;
        config.validate(positive_tts.hidden_rows())?;

        let negative = self.language.import_preset_cache(preset)?;
        let executor = NativeExecutor {
            runtime: self,
            negative,
            acoustic_stream: None,
        };
        Ok(VibeVoiceRealtimeSynthesisSession {
            core: RealtimeCore::new(
                executor,
                plan,
                positive_condition,
                negative_condition,
                config,
            ),
        })
    }
}

/// A borrow-scoped real-weight Realtime session.
pub struct VibeVoiceRealtimeSynthesisSession<'a> {
    core: RealtimeCore<NativeExecutor<'a>>,
}

impl std::fmt::Debug for VibeVoiceRealtimeSynthesisSession<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VibeVoiceRealtimeSynthesisSession")
            .field("finished", &self.core.finished)
            .field("failed", &self.core.failed)
            .field("generated_positions", &self.core.generated_positions)
            .finish_non_exhaustive()
    }
}

impl VibeVoiceRealtimeSynthesisSession<'_> {
    /// Executes one text/speech iteration with caller-owned diffusion noise.
    ///
    /// The slice is consumed synchronously and never retained.  A wrong-size
    /// or non-finite noise vector poisons the session before any model state is
    /// advanced.  `external_stop` is checked before text or audio mutation.
    pub fn step(
        &mut self,
        initial_noise: &[f32],
        external_stop: bool,
    ) -> Result<VibeVoiceRealtimeSynthesisStep> {
        self.core.step(initial_noise, external_stop)
    }

    /// Returns the observed terminal reason, if any.
    ///
    /// EOS is recorded as soon as the positive classifier crosses its
    /// threshold, while the source-faithful six-step cache drain may still be
    /// in progress. The reason is final once `is_finished()` is true.
    #[must_use]
    pub const fn stop_reason(&self) -> Option<VibeVoiceRealtimeGenerationStopReason> {
        self.core.stop_reason
    }

    /// Returns whether the session has reached an explicit terminal state.
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        self.core.finished
    }

    /// Returns the logical text-plus-speech positions consumed so far.
    #[must_use]
    pub const fn generated_positions(&self) -> usize {
        self.core.generated_positions
    }
}

trait RealtimeExecutor {
    fn process_text_token(&mut self, token: u32) -> Result<Vec<f32>>;
    fn sample(
        &mut self,
        positive_condition: &[f32],
        negative_condition: &[f32],
        initial_noise: &[f32],
        guidance_scale: f32,
    ) -> Result<Vec<f32>>;
    fn decode(&mut self, scaled_latent: &[f32]) -> Result<Vec<f32>>;
    fn connector(&mut self, scaled_latent: &[f32]) -> Result<Vec<f32>>;
    fn process_speech(
        &mut self,
        acoustic_embedding: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, f32, f32)>;
    fn reset(&mut self);
}

struct NativeExecutor<'a> {
    runtime: &'a mut VibeVoiceRealtimeRuntime,
    negative: VibeVoiceRealtimeLanguage,
    acoustic_stream: Option<VibeVoiceRealtimeAcousticDecoderStream>,
}

impl RealtimeExecutor for NativeExecutor<'_> {
    fn process_text_token(&mut self, token: u32) -> Result<Vec<f32>> {
        let lm = self.runtime.language.forward_lm_step(token)?;
        let lm_hidden = last_hidden_row(&lm.hidden, 1)?;
        let tts = self
            .runtime
            .language
            .forward_tts_lm_step(token, &lm_hidden, true)?;
        last_hidden_row(&tts.hidden, 1)
    }

    fn sample(
        &mut self,
        positive_condition: &[f32],
        negative_condition: &[f32],
        initial_noise: &[f32],
        guidance_scale: f32,
    ) -> Result<Vec<f32>> {
        sample_vibevoice_realtime_cfg(
            &self.runtime.diffusion_head,
            positive_condition,
            negative_condition,
            initial_noise,
            guidance_scale,
        )
    }

    fn decode(&mut self, scaled_latent: &[f32]) -> Result<Vec<f32>> {
        let stream = self
            .acoustic_stream
            .get_or_insert_with(|| self.runtime.acoustic_decoder.stream());
        stream.decode_scaled_latent(scaled_latent)
    }

    fn connector(&mut self, scaled_latent: &[f32]) -> Result<Vec<f32>> {
        self.runtime.connector.forward(scaled_latent)
    }

    fn process_speech(
        &mut self,
        acoustic_embedding: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, f32, f32)> {
        let positive = self.runtime.language.forward_tts_lm_step(
            SPEECH_TOKEN_ID,
            acoustic_embedding,
            false,
        )?;
        let negative =
            self.negative
                .forward_tts_lm_step(SPEECH_TOKEN_ID, acoustic_embedding, false)?;
        let positive_hidden = last_hidden_row(&positive.hidden, 1)?;
        let negative_hidden = last_hidden_row(&negative.hidden, 1)?;
        Ok((
            positive_hidden,
            negative_hidden,
            positive.eos_logit,
            negative.eos_logit,
        ))
    }

    fn reset(&mut self) {
        self.runtime.language.reset();
        self.negative.reset();
        if let Some(stream) = self.acoustic_stream.as_mut() {
            stream.reset();
        }
    }
}

struct RealtimeCore<E> {
    executor: E,
    plan: VibeVoiceStreamingTextPlan,
    positive_condition: Vec<f32>,
    negative_condition: Vec<f32>,
    config: VibeVoiceRealtimeSynthesisConfig,
    next_window: usize,
    active_window: Option<usize>,
    speech_step: usize,
    generated_positions: usize,
    speech_steps: usize,
    eos_seen: bool,
    finished: bool,
    failed: bool,
    stop_reason: Option<VibeVoiceRealtimeGenerationStopReason>,
}

impl<E: RealtimeExecutor> RealtimeCore<E> {
    fn new(
        executor: E,
        plan: VibeVoiceStreamingTextPlan,
        positive_condition: Vec<f32>,
        negative_condition: Vec<f32>,
        config: VibeVoiceRealtimeSynthesisConfig,
    ) -> Self {
        Self {
            executor,
            plan,
            positive_condition,
            negative_condition,
            config,
            next_window: 0,
            active_window: None,
            speech_step: 0,
            generated_positions: 0,
            speech_steps: 0,
            eos_seen: false,
            finished: false,
            failed: false,
            stop_reason: None,
        }
    }

    fn step(
        &mut self,
        initial_noise: &[f32],
        external_stop: bool,
    ) -> Result<VibeVoiceRealtimeSynthesisStep> {
        let result = self.step_inner(initial_noise, external_stop);
        if result.is_err() {
            self.failed = true;
            self.executor.reset();
        }
        result
    }

    fn step_inner(
        &mut self,
        initial_noise: &[f32],
        external_stop: bool,
    ) -> Result<VibeVoiceRealtimeSynthesisStep> {
        if self.failed {
            return Err(VokraError::InvalidArgument(
                "vibevoice realtime synthesis session is poisoned after an earlier failure"
                    .to_owned(),
            ));
        }
        if self.finished {
            return Ok(VibeVoiceRealtimeSynthesisStep::Finished {
                reason: self
                    .stop_reason
                    .expect("finished Realtime core must have a stop reason"),
            });
        }
        if external_stop {
            return Ok(self.finish(VibeVoiceRealtimeGenerationStopReason::ExternalStop));
        }
        validate_noise(initial_noise)?;
        // A caller-owned speech budget must not allow an otherwise unused text
        // window to mutate the positive language cache.
        if self.speech_steps >= self.config.max_speech_steps {
            return Ok(self.finish(VibeVoiceRealtimeGenerationStopReason::ControlBudgetExhausted));
        }
        self.prepare_next_text_window()?;
        if self.finished {
            return Ok(VibeVoiceRealtimeSynthesisStep::Finished {
                reason: self
                    .stop_reason
                    .expect("finished Realtime core must have a stop reason"),
            });
        }
        let eos_before_step = self.eos_seen;
        // The following order is intentionally source-visible: the connector
        // receives the sampled scaled latent, not the decoder's unscaled copy.
        let scaled_latent = self.executor.sample(
            &self.positive_condition,
            &self.negative_condition,
            initial_noise,
            self.config.guidance_scale,
        )?;
        if scaled_latent.len() != VIBEVOICE_REALTIME_LATENT_WIDTH
            || scaled_latent.iter().any(|value| !value.is_finite())
        {
            return Err(VokraError::ModelLoad(
                "vibevoice realtime sampler returned an invalid 64-wide latent".to_owned(),
            ));
        }
        let pcm = self.executor.decode(&scaled_latent)?;
        if pcm.len() != REALTIME_ACOUSTIC_CHUNK_SAMPLES
            || pcm.iter().any(|value| !value.is_finite())
        {
            return Err(VokraError::ModelLoad(
                "vibevoice realtime acoustic decoder returned invalid PCM".to_owned(),
            ));
        }
        let acoustic_embedding = self.executor.connector(&scaled_latent)?;
        if acoustic_embedding.len() != HIDDEN
            || acoustic_embedding.iter().any(|value| !value.is_finite())
        {
            return Err(VokraError::ModelLoad(
                "vibevoice realtime acoustic connector returned an invalid hidden row".to_owned(),
            ));
        }

        // Microsoft decodes and connects the final chunk before appending the
        // next TTS token and checking `max_length`.  Preserve that terminal
        // audio, but do not advance either TTS cache when the append would be
        // beyond the logical max_new_tokens budget.
        if self.generated_positions >= self.config.max_new_tokens {
            let text_window_index = self.active_window;
            let speech_step = self.speech_step;
            if self.eos_seen {
                self.finish_reason(VibeVoiceRealtimeGenerationStopReason::EndOfSpeech);
                return Ok(VibeVoiceRealtimeSynthesisStep::Finished {
                    reason: VibeVoiceRealtimeGenerationStopReason::EndOfSpeech,
                });
            }
            self.finish_reason(VibeVoiceRealtimeGenerationStopReason::MaxLength);
            return Ok(VibeVoiceRealtimeSynthesisStep::Audio(
                VibeVoiceRealtimeAudioChunk {
                    pcm,
                    text_window_index,
                    speech_step,
                    generated_positions: self.generated_positions,
                    terminal_after: Some(VibeVoiceRealtimeGenerationStopReason::MaxLength),
                },
            ));
        }
        let (positive_hidden, negative_hidden, positive_eos_logit, negative_eos_logit) =
            self.executor.process_speech(&acoustic_embedding)?;
        validate_hidden_row(&positive_hidden)?;
        validate_hidden_row(&negative_hidden)?;
        self.positive_condition = positive_hidden;
        self.negative_condition = negative_hidden;
        if !positive_eos_logit.is_finite() {
            return Err(VokraError::ModelLoad(
                "vibevoice realtime positive EOS logit is non-finite".to_owned(),
            ));
        }
        // Upstream uses only the positive logit for terminal classification,
        // but both branch outputs must remain finite before the cache state is
        // accepted. The negative value is deliberately not a stop signal.
        if !negative_eos_logit.is_finite() {
            return Err(VokraError::ModelLoad(
                "vibevoice realtime negative EOS logit is non-finite".to_owned(),
            ));
        }
        self.generated_positions += 1;
        self.speech_steps += 1;
        let text_window_index = self.active_window;
        let speech_step = self.speech_step;
        let window_complete = speech_step + 1 >= TTS_SPEECH_WINDOW_SIZE;
        self.advance_speech_cursor();

        let observed_eos = sigmoid(positive_eos_logit) > 0.5;
        if observed_eos {
            self.eos_seen = true;
            self.stop_reason = Some(VibeVoiceRealtimeGenerationStopReason::EndOfSpeech);
        }
        let terminal_after = if observed_eos {
            Some(VibeVoiceRealtimeGenerationStopReason::EndOfSpeech)
        } else if self.speech_steps >= self.config.max_speech_steps {
            Some(VibeVoiceRealtimeGenerationStopReason::ControlBudgetExhausted)
        } else {
            None
        };
        let drained = eos_before_step;
        if self.eos_seen && window_complete {
            self.finished = true;
            self.stop_reason = Some(VibeVoiceRealtimeGenerationStopReason::EndOfSpeech);
        } else if !self.eos_seen {
            if let Some(reason) = terminal_after {
                self.stop_reason = Some(reason);
                self.finished = true;
            }
        }
        if drained {
            return Ok(VibeVoiceRealtimeSynthesisStep::Draining {
                text_window_index,
                speech_step,
            });
        }
        Ok(VibeVoiceRealtimeSynthesisStep::Audio(
            VibeVoiceRealtimeAudioChunk {
                pcm,
                text_window_index,
                speech_step,
                generated_positions: self.generated_positions,
                terminal_after,
            },
        ))
    }

    fn prepare_next_text_window(&mut self) -> Result<()> {
        if self.active_window.is_some() || self.next_window >= self.plan.len() {
            return Ok(());
        }
        let window = &self.plan.text_windows()[self.next_window];
        let count = window.text_ids().len();
        let next_positions = self.generated_positions.checked_add(count).ok_or_else(|| {
            VokraError::InvalidArgument(
                "vibevoice realtime generated position counter overflow".to_owned(),
            )
        })?;
        if next_positions > self.config.max_new_tokens {
            self.finish_reason(VibeVoiceRealtimeGenerationStopReason::MaxLength);
            return Ok(());
        }
        // Check the complete window before mutating either cache.  Once the
        // first token is accepted, any operational error poisons the session.
        let mut condition = None;
        for &token in window.text_ids() {
            condition = Some(self.executor.process_text_token(token)?);
        }
        let condition = condition.ok_or_else(|| {
            VokraError::ModelLoad(
                "vibevoice realtime text window was unexpectedly empty".to_owned(),
            )
        })?;
        validate_hidden_row(&condition)?;
        self.positive_condition = condition;
        self.active_window = Some(window.index());
        self.speech_step = 0;
        self.next_window += 1;
        self.generated_positions += count;
        Ok(())
    }

    fn advance_speech_cursor(&mut self) {
        if self.speech_step + 1 >= TTS_SPEECH_WINDOW_SIZE {
            self.speech_step = 0;
            self.active_window = None;
        } else {
            self.speech_step += 1;
        }
    }

    fn finish(
        &mut self,
        reason: VibeVoiceRealtimeGenerationStopReason,
    ) -> VibeVoiceRealtimeSynthesisStep {
        self.finish_reason(reason);
        VibeVoiceRealtimeSynthesisStep::Finished {
            reason: self
                .stop_reason
                .expect("finished Realtime core must have a stop reason"),
        }
    }

    fn finish_reason(&mut self, reason: VibeVoiceRealtimeGenerationStopReason) {
        // Explicit caller stop/control limits override an incomplete EOS
        // drain. This keeps the returned variant and stop_reason() identical.
        self.stop_reason = Some(reason);
        self.finished = true;
    }
}

fn require_realtime_backend_before_binding(backend: BackendKind) -> Result<()> {
    match backend {
        BackendKind::Cpu | BackendKind::Metal => Ok(()),
        _ => Err(VokraError::UnsupportedOp(format!(
            "vibevoice realtime composite backend {backend:?} lacks complete learned-op coverage; no CPU fallback is used"
        ))),
    }
}

fn preflight_realtime_backend(backend: BackendKind) -> Result<()> {
    Compute::for_backend(backend, REALTIME_COMPOSITE_HOT_OPS).map(|_| ())
}

fn validate_noise(noise: &[f32]) -> Result<()> {
    if noise.len() != VIBEVOICE_REALTIME_LATENT_WIDTH {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime caller-owned noise must have width 64".to_owned(),
        ));
    }
    if noise.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime caller-owned noise must be finite".to_owned(),
        ));
    }
    Ok(())
}

fn last_hidden_row(hidden: &[f32], rows: usize) -> Result<Vec<f32>> {
    if rows == 0 || hidden.len() != rows * HIDDEN {
        return Err(VokraError::ModelLoad(
            "vibevoice realtime preset hidden-state shape does not match 896-wide rows".to_owned(),
        ));
    }
    let start = (rows - 1) * HIDDEN;
    let row = hidden[start..].to_vec();
    if row.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::ModelLoad(
            "vibevoice realtime preset hidden state contains non-finite values".to_owned(),
        ));
    }
    Ok(row)
}

fn validate_hidden_row(hidden: &[f32]) -> Result<()> {
    if hidden.len() != HIDDEN || hidden.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::ModelLoad(
            "vibevoice realtime hidden output must be one finite 896-wide row".to_owned(),
        ));
    }
    Ok(())
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    #[derive(Debug, Default)]
    struct TraceExecutor {
        events: Vec<String>,
        latent: Vec<f32>,
        eos: VecDeque<f32>,
        fail_at: Option<&'static str>,
    }

    impl TraceExecutor {
        fn maybe_fail(&self, stage: &'static str) -> Result<()> {
            if self.fail_at == Some(stage) {
                return Err(VokraError::ModelLoad(format!(
                    "synthetic failure at {stage}"
                )));
            }
            Ok(())
        }
    }

    impl RealtimeExecutor for TraceExecutor {
        fn process_text_token(&mut self, token: u32) -> Result<Vec<f32>> {
            self.maybe_fail("text")?;
            self.events.push(format!("text:{token}"));
            Ok(vec![token as f32; HIDDEN])
        }

        fn sample(
            &mut self,
            _positive_condition: &[f32],
            _negative_condition: &[f32],
            initial_noise: &[f32],
            _guidance_scale: f32,
        ) -> Result<Vec<f32>> {
            self.maybe_fail("sample")?;
            self.events.push("sample".into());
            self.latent = initial_noise.to_vec();
            Ok(initial_noise.to_vec())
        }

        fn decode(&mut self, scaled_latent: &[f32]) -> Result<Vec<f32>> {
            self.maybe_fail("decode")?;
            assert_eq!(scaled_latent, self.latent.as_slice());
            self.events.push("decode".into());
            Ok(vec![0.0; REALTIME_ACOUSTIC_CHUNK_SAMPLES])
        }

        fn connector(&mut self, scaled_latent: &[f32]) -> Result<Vec<f32>> {
            self.maybe_fail("connector")?;
            assert_eq!(scaled_latent, self.latent.as_slice());
            self.events.push("connector".into());
            Ok(vec![1.0; HIDDEN])
        }

        fn process_speech(
            &mut self,
            _acoustic_embedding: &[f32],
        ) -> Result<(Vec<f32>, Vec<f32>, f32, f32)> {
            self.maybe_fail("speech")?;
            self.events.push("positive_tts".into());
            self.events.push("negative_tts".into());
            Ok((
                vec![2.0; HIDDEN],
                vec![3.0; HIDDEN],
                self.eos.pop_front().unwrap_or(-100.0),
                -100.0,
            ))
        }

        fn reset(&mut self) {
            self.events.push("reset".into());
        }
    }

    fn plan(tokens: &[u32]) -> VibeVoiceStreamingTextPlan {
        let state = super::super::state::synthetic_generation_test_state();
        state.plan_text_windows(tokens).unwrap()
    }

    fn core(
        tokens: &[u32],
        max_new_tokens: usize,
        max_speech_steps: usize,
        executor: TraceExecutor,
    ) -> RealtimeCore<TraceExecutor> {
        RealtimeCore::new(
            executor,
            plan(tokens),
            vec![0.0; HIDDEN],
            vec![0.0; HIDDEN],
            VibeVoiceRealtimeSynthesisConfig {
                max_new_tokens,
                max_speech_steps,
                guidance_scale: 3.0,
            },
        )
    }

    #[test]
    fn source_order_and_original_scaled_latent_are_preserved() {
        let mut core = core(&[10, 11], 20, 20, TraceExecutor::default());
        let noise = vec![0.25; VIBEVOICE_REALTIME_LATENT_WIDTH];
        let _ = core.step(&noise, false).unwrap();
        let events = &core.executor.events;
        assert_eq!(
            &events[..7],
            &[
                "text:10",
                "text:11",
                "sample",
                "decode",
                "connector",
                "positive_tts",
                "negative_tts",
                // The next call is allowed to enter the next source phase;
                // this assertion only covers the first exact speech step.
            ]
        );
        assert_eq!(core.executor.latent, noise);
        assert_eq!(core.positive_condition[0], 2.0);
        assert_eq!(core.negative_condition[0], 3.0);
    }

    #[test]
    fn negative_text_branch_is_never_advanced_and_zero_text_continues() {
        let mut core = core(&[10], 20, 8, TraceExecutor::default());
        let noise = vec![0.0; VIBEVOICE_REALTIME_LATENT_WIDTH];
        for _ in 0..7 {
            let _ = core.step(&noise, false).unwrap();
        }
        assert_eq!(
            core.executor
                .events
                .iter()
                .filter(|event| event.starts_with("text:"))
                .count(),
            1
        );
        assert!(core.active_window.is_none());
        assert!(!core.finished);
    }

    #[test]
    fn invalid_noise_is_rejected_before_executor_and_poisons_session() {
        let mut core = core(&[10], 20, 20, TraceExecutor::default());
        assert!(core.step(&[0.0; 63], false).is_err());
        assert_eq!(core.executor.events, vec!["reset"]);
        assert!(core.step(&[0.0; 64], false).is_err());
    }

    #[test]
    fn eos_and_max_length_are_checked_after_audio_step() {
        let mut executor = TraceExecutor::default();
        executor.eos.push_back(100.0);
        let mut eos_core = core(&[10], 20, 20, executor);
        let audio = eos_core
            .step(&[0.0; VIBEVOICE_REALTIME_LATENT_WIDTH], false)
            .unwrap();
        assert!(matches!(
            audio,
            VibeVoiceRealtimeSynthesisStep::Audio(VibeVoiceRealtimeAudioChunk {
                terminal_after: Some(VibeVoiceRealtimeGenerationStopReason::EndOfSpeech),
                ..
            })
        ));
        for _ in 0..5 {
            assert!(matches!(
                eos_core
                    .step(&[0.0; VIBEVOICE_REALTIME_LATENT_WIDTH], false)
                    .unwrap(),
                VibeVoiceRealtimeSynthesisStep::Draining { .. }
            ));
        }
        assert!(matches!(
            eos_core
                .step(&[0.0; VIBEVOICE_REALTIME_LATENT_WIDTH], false)
                .unwrap(),
            VibeVoiceRealtimeSynthesisStep::Finished {
                reason: VibeVoiceRealtimeGenerationStopReason::EndOfSpeech
            }
        ));

        let mut max_core = core(&[10], 1, 20, TraceExecutor::default());
        let audio = max_core
            .step(&[0.0; VIBEVOICE_REALTIME_LATENT_WIDTH], false)
            .unwrap();
        // The source decodes the final chunk before checking that appending a
        // further TTS token would exceed max_length.
        assert!(matches!(
            audio,
            VibeVoiceRealtimeSynthesisStep::Audio(VibeVoiceRealtimeAudioChunk {
                terminal_after: Some(VibeVoiceRealtimeGenerationStopReason::MaxLength),
                ..
            })
        ));
        assert_eq!(
            max_core.executor.events.last(),
            Some(&"connector".to_owned())
        );

        let mut equality_core = core(&[10], 2, 20, TraceExecutor::default());
        let first = equality_core
            .step(&[0.0; VIBEVOICE_REALTIME_LATENT_WIDTH], false)
            .unwrap();
        assert!(matches!(
            first,
            VibeVoiceRealtimeSynthesisStep::Audio(VibeVoiceRealtimeAudioChunk {
                terminal_after: None,
                generated_positions: 2,
                ..
            })
        ));
        let second = equality_core
            .step(&[0.0; VIBEVOICE_REALTIME_LATENT_WIDTH], false)
            .unwrap();
        assert!(matches!(
            second,
            VibeVoiceRealtimeSynthesisStep::Audio(VibeVoiceRealtimeAudioChunk {
                terminal_after: Some(VibeVoiceRealtimeGenerationStopReason::MaxLength),
                generated_positions: 2,
                ..
            })
        ));
        assert_eq!(
            equality_core
                .executor
                .events
                .iter()
                .filter(|event| event.as_str() == "positive_tts")
                .count(),
            1
        );
    }

    #[test]
    fn every_audio_step_after_first_eos_is_suppressed() {
        for eos_logits in [[100.0_f32, 100.0, 100.0], [100.0, -100.0, 100.0]] {
            let mut executor = TraceExecutor::default();
            executor.eos.extend(eos_logits);
            let mut core = core(&[10], 20, 20, executor);
            assert!(matches!(
                core.step(&[0.0; VIBEVOICE_REALTIME_LATENT_WIDTH], false)
                    .unwrap(),
                VibeVoiceRealtimeSynthesisStep::Audio(_)
            ));
            for _ in 0..2 {
                assert!(matches!(
                    core.step(&[0.0; VIBEVOICE_REALTIME_LATENT_WIDTH], false)
                        .unwrap(),
                    VibeVoiceRealtimeSynthesisStep::Draining { .. }
                ));
            }
        }
    }

    #[test]
    fn explicit_stop_and_control_budget_override_incomplete_eos_drain() {
        let mut executor = TraceExecutor::default();
        executor.eos.push_back(100.0);
        let mut external = core(&[10], 20, 20, executor);
        let _ = external
            .step(&[0.0; VIBEVOICE_REALTIME_LATENT_WIDTH], false)
            .unwrap();
        let stopped = external
            .step(&[0.0; VIBEVOICE_REALTIME_LATENT_WIDTH], true)
            .unwrap();
        assert!(matches!(
            stopped,
            VibeVoiceRealtimeSynthesisStep::Finished {
                reason: VibeVoiceRealtimeGenerationStopReason::ExternalStop
            }
        ));
        assert_eq!(
            external.stop_reason,
            Some(VibeVoiceRealtimeGenerationStopReason::ExternalStop)
        );

        let mut executor = TraceExecutor::default();
        executor.eos.push_back(100.0);
        let mut budget = core(&[10], 20, 1, executor);
        let _ = budget
            .step(&[0.0; VIBEVOICE_REALTIME_LATENT_WIDTH], false)
            .unwrap();
        let stopped = budget
            .step(&[0.0; VIBEVOICE_REALTIME_LATENT_WIDTH], false)
            .unwrap();
        assert!(matches!(
            stopped,
            VibeVoiceRealtimeSynthesisStep::Finished {
                reason: VibeVoiceRealtimeGenerationStopReason::ControlBudgetExhausted
            }
        ));
        assert_eq!(
            budget.stop_reason,
            Some(VibeVoiceRealtimeGenerationStopReason::ControlBudgetExhausted)
        );
    }

    #[test]
    fn operational_failure_resets_and_forbids_cache_reuse() {
        let executor = TraceExecutor {
            fail_at: Some("connector"),
            ..TraceExecutor::default()
        };
        let mut core = core(&[10], 20, 20, executor);
        assert!(
            core.step(&[0.0; VIBEVOICE_REALTIME_LATENT_WIDTH], false)
                .is_err()
        );
        assert!(core.failed);
        assert_eq!(core.executor.events.last(), Some(&"reset".to_owned()));
        assert!(
            core.step(&[0.0; VIBEVOICE_REALTIME_LATENT_WIDTH], false)
                .is_err()
        );
    }

    #[test]
    fn metal_backend_selection_is_permitted_before_weight_binding() {
        assert!(require_realtime_backend_before_binding(BackendKind::Metal).is_ok());
    }

    #[test]
    fn composite_registry_covers_all_component_learned_ops() {
        let component_registries = [
            crate::vibevoice::QWEN2_HOT_OPS,
            crate::vibevoice::VIBEVOICE_TOKENIZER_HOT_OPS,
            crate::vibevoice::VIBEVOICE_ACOUSTIC_DECODER_HOT_OPS,
            crate::vibevoice_streaming::connector::VIBEVOICE_REALTIME_ACOUSTIC_CONNECTOR_HOT_OPS,
            crate::vibevoice_streaming::diffusion::VIBEVOICE_STREAMING_DIFFUSION_HOT_OPS,
            crate::vibevoice_streaming::language::VIBEVOICE_REALTIME_LANGUAGE_HOT_OPS,
        ];
        for registry in component_registries {
            for op in registry {
                assert!(
                    REALTIME_COMPOSITE_HOT_OPS.contains(op),
                    "composite registry omitted component op {op:?}"
                );
            }
        }
    }

    #[test]
    fn uncovered_backend_is_rejected_before_weight_binding() {
        let error = require_realtime_backend_before_binding(BackendKind::Cuda).unwrap_err();
        assert!(matches!(error, VokraError::UnsupportedOp(_)));
        assert!(error.to_string().contains("no CPU fallback"));
    }

    #[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
    #[test]
    fn metal_feature_off_fails_during_backend_preflight() {
        let error = preflight_realtime_backend(BackendKind::Metal).unwrap_err();
        assert!(matches!(error, VokraError::BackendUnavailable(_)));
    }
}
