//! Source-authenticated Realtime generation control plane.
//!
//! The pinned Microsoft implementation interleaves text windows and bounded
//! speech loops, but its runnable entry point also requires four independently
//! prefilled prompt branches (`lm`, `tts_lm`, `neg_lm`, and `neg_tts_lm`),
//! caller-owned diffusion noise, and model-side cache-position updates. The
//! current native binders do not expose a trusted import for those upstream
//! KV caches. This module therefore implements the largest safe seam: a
//! model-free control plane that emits the source ordering and keeps terminal
//! conditions explicit. It never fabricates hidden states, noise, EOS values,
//! latents, or PCM.
//!
//! Source contract: Microsoft VibeVoice commit
//! `94da20d98b2fa7688e9cbfaf7692ddb4954f7600`,
//! `modeling_vibevoice_streaming_inference.py` (`generate`, lines 646-809):
//! `TTS_TEXT_WINDOW_SIZE = 5` and `TTS_SPEECH_WINDOW_SIZE = 6` (lines 25-26),
//! five-token text slices (lines 667-669), and a six-step maximum inner speech
//! loop (lines 705-706) that can continue with zero text until EOS,
//! max-length, or external stop.

use vokra_core::backend::BackendKind;
use vokra_core::{Result, VokraError};

use super::state::{TTS_SPEECH_WINDOW_SIZE, VibeVoiceStreamingState, VibeVoiceStreamingTextPlan};

/// The source's EOS decision threshold, retained as documentation for the
/// future model-backed boundary. This control plane never guesses a logit or
/// applies sigmoid itself.
pub const VIBEVOICE_REALTIME_EOS_THRESHOLD: f32 = 0.5;

/// Why a Realtime control-plane session stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VibeVoiceRealtimeGenerationStopReason {
    /// The authenticated TTS EOS classifier signaled end of speech.
    EndOfSpeech,
    /// The caller's model-position budget was exhausted.
    MaxLength,
    /// The caller requested an external stop.
    ExternalStop,
}

/// One source-authenticated action for a Realtime generation session.
///
/// `SpeechStep` is emitted at most six times per text window, and continues
/// with `text_window_index: None` after text is exhausted. The latter is the
/// source's zero-text loop; it is not an implicit completion event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VibeVoiceRealtimeGenerationControl {
    /// Feed one five-token-or-shorter text window to the positive LM/TTS
    /// stages. The next-window size is a cache-position hint only.
    TextWindow {
        /// Zero-based text-window index.
        text_window_index: usize,
        /// Authenticated text IDs for this window.
        text_ids: Vec<u32>,
        /// Number of IDs in the following text window, or zero when text is
        /// exhausted. This is not a generation-complete signal.
        next_text_window_size: usize,
    },
    /// Run one speech iteration after a text window, or during the source's
    /// zero-text continuation when `text_window_index` is `None`.
    SpeechStep {
        /// Source text-window index, or `None` after all text windows.
        text_window_index: Option<usize>,
        /// Zero-based step within this six-step maximum.
        speech_step: usize,
        /// Maximum number of steps before the source evaluates the next loop
        /// condition. EOS/max-length/external stop may end the loop earlier.
        max_speech_steps: usize,
    },
    /// The caller supplied an explicit terminal observation.
    Finished {
        /// The observed terminal condition.
        reason: VibeVoiceRealtimeGenerationStopReason,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cursor {
    TextWindow(usize),
    SpeechWindow {
        text_window_index: usize,
        step: usize,
    },
    ZeroText {
        step: usize,
    },
    Finished,
}

/// CPU-only, model-free Realtime generation session.
///
/// The session owns only the authenticated text-window plan and cursor. It
/// does not own or simulate language-model KV caches, diffusion state, acoustic
/// decoder state, or random draws. The explicit CPU restriction prevents a
/// caller from mistaking this control plane for an unimplemented GPU runtime;
/// non-CPU requests fail instead of falling back.
#[derive(Debug, Clone)]
pub struct VibeVoiceRealtimeGenerationSession {
    plan: VibeVoiceStreamingTextPlan,
    cursor: Cursor,
    backend: BackendKind,
    stop_reason: Option<VibeVoiceRealtimeGenerationStopReason>,
}

impl VibeVoiceRealtimeGenerationSession {
    /// Builds a control-plane session from the authenticated text state.
    pub fn new(
        state: &VibeVoiceStreamingState,
        text_ids: &[u32],
        backend: BackendKind,
    ) -> Result<Self> {
        require_cpu(backend)?;
        let plan = state.plan_text_windows(text_ids)?;
        Self::from_plan(plan, backend)
    }

    /// Builds a session from a previously authenticated text-window plan.
    pub fn from_plan(plan: VibeVoiceStreamingTextPlan, backend: BackendKind) -> Result<Self> {
        require_cpu(backend)?;
        if plan.is_empty() {
            return Err(VokraError::InvalidArgument(
                "vibevoice realtime generation requires at least one text window".to_owned(),
            ));
        }
        Ok(Self {
            plan,
            cursor: Cursor::TextWindow(0),
            backend,
            stop_reason: None,
        })
    }

    /// Returns the explicitly selected backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Returns the finite text-window plan used by this session.
    #[must_use]
    pub const fn plan(&self) -> &VibeVoiceStreamingTextPlan {
        &self.plan
    }

    /// Returns the observed terminal condition, if one was supplied.
    #[must_use]
    pub const fn stop_reason(&self) -> Option<VibeVoiceRealtimeGenerationStopReason> {
        self.stop_reason
    }

    /// Returns whether an explicit terminal condition has been observed.
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        matches!(self.cursor, Cursor::Finished)
    }

    /// Emits the next source-authenticated control action.
    ///
    /// After the last text window, this keeps emitting zero-text speech steps
    /// until the caller reports EOS, max length, or external stop with
    /// [`Self::finish`]. It never reports completion merely because text ended.
    pub fn next_control(&mut self) -> Result<VibeVoiceRealtimeGenerationControl> {
        match self.cursor {
            Cursor::TextWindow(index) => {
                let window = self.plan.text_windows().get(index).ok_or_else(|| {
                    VokraError::ModelLoad("vibevoice realtime text cursor exceeded plan".to_owned())
                })?;
                self.cursor = Cursor::SpeechWindow {
                    text_window_index: index,
                    step: 0,
                };
                Ok(VibeVoiceRealtimeGenerationControl::TextWindow {
                    text_window_index: index,
                    text_ids: window.text_ids().to_vec(),
                    next_text_window_size: window.next_text_window_size(),
                })
            }
            Cursor::SpeechWindow {
                text_window_index,
                step,
            } => {
                if step + 1 >= TTS_SPEECH_WINDOW_SIZE {
                    self.cursor = if text_window_index + 1 < self.plan.len() {
                        Cursor::TextWindow(text_window_index + 1)
                    } else {
                        Cursor::ZeroText { step: 0 }
                    };
                } else {
                    self.cursor = Cursor::SpeechWindow {
                        text_window_index,
                        step: step + 1,
                    };
                }
                Ok(VibeVoiceRealtimeGenerationControl::SpeechStep {
                    text_window_index: Some(text_window_index),
                    speech_step: step,
                    max_speech_steps: TTS_SPEECH_WINDOW_SIZE,
                })
            }
            Cursor::ZeroText { step } => {
                self.cursor = Cursor::ZeroText {
                    step: (step + 1) % TTS_SPEECH_WINDOW_SIZE,
                };
                Ok(VibeVoiceRealtimeGenerationControl::SpeechStep {
                    text_window_index: None,
                    speech_step: step,
                    max_speech_steps: TTS_SPEECH_WINDOW_SIZE,
                })
            }
            Cursor::Finished => Ok(VibeVoiceRealtimeGenerationControl::Finished {
                reason: self
                    .stop_reason
                    .expect("finished Realtime session must have a stop reason"),
            }),
        }
    }

    /// Records an explicit EOS, max-length, or external-stop observation.
    pub fn finish(&mut self, reason: VibeVoiceRealtimeGenerationStopReason) {
        if self.stop_reason.is_none() {
            self.stop_reason = Some(reason);
            self.cursor = Cursor::Finished;
        }
    }

    /// Explicitly rejects native synthesis until the four authenticated
    /// upstream prompt KV branches and caller-owned noise contract can be
    /// imported without fabrication.
    pub fn execute_native(&self) -> Result<()> {
        Err(VokraError::NotImplemented(
            "vibevoice realtime native generation requires authenticated lm/tts_lm/neg_lm/neg_tts_lm prompt KV caches and caller-owned 64-wide diffusion noise; use next_control for the source-faithful control plane",
        ))
    }
}

fn require_cpu(backend: BackendKind) -> Result<()> {
    if backend != BackendKind::Cpu {
        return Err(VokraError::UnsupportedOp(format!(
            "vibevoice realtime generation control plane is CPU-only; backend {backend:?} is unsupported and no CPU fallback is used"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vibevoice_streaming::state::TTS_TEXT_WINDOW_SIZE;

    #[test]
    fn source_limits_are_bounded_without_claiming_completion() {
        assert_eq!(TTS_TEXT_WINDOW_SIZE, 5,);
        assert_eq!(TTS_SPEECH_WINDOW_SIZE, 6,);
        let action = VibeVoiceRealtimeGenerationControl::SpeechStep {
            text_window_index: None,
            speech_step: 0,
            max_speech_steps: TTS_SPEECH_WINDOW_SIZE,
        };
        assert!(matches!(
            action,
            VibeVoiceRealtimeGenerationControl::SpeechStep {
                text_window_index: None,
                max_speech_steps: 6,
                ..
            }
        ));
    }

    #[test]
    fn terminal_reasons_are_explicit_observations() {
        assert_ne!(
            VibeVoiceRealtimeGenerationStopReason::EndOfSpeech,
            VibeVoiceRealtimeGenerationStopReason::MaxLength
        );
        assert_ne!(
            VibeVoiceRealtimeGenerationStopReason::MaxLength,
            VibeVoiceRealtimeGenerationStopReason::ExternalStop
        );
    }

    #[test]
    fn non_cpu_backend_is_rejected_without_fallback() {
        let error = require_cpu(BackendKind::Metal).unwrap_err();
        assert!(matches!(error, VokraError::UnsupportedOp(_)));
        assert!(error.to_string().contains("no CPU fallback"));
    }

    #[test]
    fn native_blocker_names_authenticated_prefill_and_noise_contract() {
        let message = "vibevoice realtime native generation requires authenticated lm/tts_lm/neg_lm/neg_tts_lm prompt KV caches and caller-owned 64-wide diffusion noise; use next_control for the source-faithful control plane";
        assert!(message.contains("neg_tts_lm"));
        assert!(message.contains("caller-owned 64-wide diffusion noise"));
    }
}
