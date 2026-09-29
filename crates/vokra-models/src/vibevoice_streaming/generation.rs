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

/// Why a Realtime control-plane session stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VibeVoiceRealtimeGenerationStopReason {
    /// The authenticated TTS EOS classifier signaled end of speech.
    EndOfSpeech,
    /// The caller's model-position budget was exhausted.
    MaxLength,
    /// The finite caller-owned control-plane speech budget was exhausted.
    ///
    /// This is intentionally distinct from [`Self::MaxLength`]: the native
    /// control plane does not infer model positions or authenticate a model
    /// `max_length` from this budget.
    ControlBudgetExhausted,
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
    remaining_speech_steps: usize,
    stop_reason: Option<VibeVoiceRealtimeGenerationStopReason>,
}

impl VibeVoiceRealtimeGenerationSession {
    /// Builds a control-plane session from the authenticated text state.
    pub fn new(
        state: &VibeVoiceStreamingState,
        text_ids: &[u32],
        backend: BackendKind,
        speech_step_budget: usize,
    ) -> Result<Self> {
        require_cpu(backend)?;
        let plan = state.plan_text_windows(text_ids)?;
        Self::from_plan(plan, backend, speech_step_budget)
    }

    /// Builds a session from a previously authenticated text-window plan.
    ///
    /// `speech_step_budget` is an explicit caller-owned control-plane budget.
    /// It is consumed once per emitted speech action and is deliberately not
    /// presented as an inferred replacement for the upstream model's
    /// `max_length` position accounting.
    pub fn from_plan(
        plan: VibeVoiceStreamingTextPlan,
        backend: BackendKind,
        speech_step_budget: usize,
    ) -> Result<Self> {
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
            remaining_speech_steps: speech_step_budget,
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

    /// Returns the number of speech actions still permitted by the caller's
    /// explicit control-plane budget.
    #[must_use]
    pub const fn remaining_speech_steps(&self) -> usize {
        self.remaining_speech_steps
    }

    /// Emits the next source-authenticated control action.
    ///
    /// After the last text window, this keeps emitting zero-text speech steps
    /// until the explicit caller budget is exhausted or the caller reports EOS or
    /// external stop with [`Self::finish`]. It never reports completion merely
    /// because text ended.
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
                if self.remaining_speech_steps == 0 {
                    self.finish(VibeVoiceRealtimeGenerationStopReason::ControlBudgetExhausted);
                    return self.next_control();
                }
                self.remaining_speech_steps -= 1;
                let budget_exhausted = self.remaining_speech_steps == 0;
                if step + 1 >= TTS_SPEECH_WINDOW_SIZE {
                    self.cursor = if budget_exhausted {
                        Cursor::Finished
                    } else if text_window_index + 1 < self.plan.len() {
                        Cursor::TextWindow(text_window_index + 1)
                    } else {
                        Cursor::ZeroText { step: 0 }
                    };
                } else {
                    self.cursor = if budget_exhausted {
                        Cursor::Finished
                    } else {
                        Cursor::SpeechWindow {
                            text_window_index,
                            step: step + 1,
                        }
                    };
                }
                if budget_exhausted {
                    self.stop_reason =
                        Some(VibeVoiceRealtimeGenerationStopReason::ControlBudgetExhausted);
                }
                Ok(VibeVoiceRealtimeGenerationControl::SpeechStep {
                    text_window_index: Some(text_window_index),
                    speech_step: step,
                    max_speech_steps: TTS_SPEECH_WINDOW_SIZE,
                })
            }
            Cursor::ZeroText { step } => {
                if self.remaining_speech_steps == 0 {
                    self.finish(VibeVoiceRealtimeGenerationStopReason::ControlBudgetExhausted);
                    return self.next_control();
                }
                self.remaining_speech_steps -= 1;
                let budget_exhausted = self.remaining_speech_steps == 0;
                self.cursor = Cursor::ZeroText {
                    step: (step + 1) % TTS_SPEECH_WINDOW_SIZE,
                };
                if budget_exhausted {
                    self.cursor = Cursor::Finished;
                    self.stop_reason =
                        Some(VibeVoiceRealtimeGenerationStopReason::ControlBudgetExhausted);
                }
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

    fn session(speech_step_budget: usize) -> VibeVoiceRealtimeGenerationSession {
        let state = super::super::state::synthetic_generation_test_state();
        VibeVoiceRealtimeGenerationSession::new(
            &state,
            &(101..=112).collect::<Vec<_>>(),
            BackendKind::Cpu,
            speech_step_budget,
        )
        .unwrap()
    }

    fn assert_text_window(
        session: &mut VibeVoiceRealtimeGenerationSession,
        index: usize,
        text_ids: &[u32],
        next_text_window_size: usize,
    ) {
        assert_eq!(
            session.next_control().unwrap(),
            VibeVoiceRealtimeGenerationControl::TextWindow {
                text_window_index: index,
                text_ids: text_ids.to_vec(),
                next_text_window_size,
            }
        );
    }

    fn assert_six_speech_steps(
        session: &mut VibeVoiceRealtimeGenerationSession,
        text_window_index: usize,
    ) {
        for speech_step in 0..TTS_SPEECH_WINDOW_SIZE {
            assert_eq!(
                session.next_control().unwrap(),
                VibeVoiceRealtimeGenerationControl::SpeechStep {
                    text_window_index: Some(text_window_index),
                    speech_step,
                    max_speech_steps: TTS_SPEECH_WINDOW_SIZE,
                }
            );
        }
    }

    #[test]
    fn control_plane_emits_three_windows_and_six_steps_each() {
        let mut session = session(24);
        assert_text_window(&mut session, 0, &[101, 102, 103, 104, 105], 5);
        assert_six_speech_steps(&mut session, 0);
        assert_text_window(&mut session, 1, &[106, 107, 108, 109, 110], 2);
        assert_six_speech_steps(&mut session, 1);
        assert_text_window(&mut session, 2, &[111, 112], 0);
        assert_six_speech_steps(&mut session, 2);
        assert_eq!(session.remaining_speech_steps(), 6);
        assert_eq!(
            session.next_control().unwrap(),
            VibeVoiceRealtimeGenerationControl::SpeechStep {
                text_window_index: None,
                speech_step: 0,
                max_speech_steps: TTS_SPEECH_WINDOW_SIZE,
            }
        );
        assert!(!session.is_finished());
    }

    #[test]
    fn zero_text_budget_exhaustion_is_checked_without_off_by_one() {
        let mut session = session(19);
        for index in 0..3 {
            match index {
                0 => assert_text_window(&mut session, 0, &[101, 102, 103, 104, 105], 5),
                1 => assert_text_window(&mut session, 1, &[106, 107, 108, 109, 110], 2),
                2 => assert_text_window(&mut session, 2, &[111, 112], 0),
                _ => unreachable!(),
            }
            assert_six_speech_steps(&mut session, index);
        }
        assert_eq!(session.remaining_speech_steps(), 1);
        assert_eq!(
            session.next_control().unwrap(),
            VibeVoiceRealtimeGenerationControl::SpeechStep {
                text_window_index: None,
                speech_step: 0,
                max_speech_steps: TTS_SPEECH_WINDOW_SIZE,
            }
        );
        assert_eq!(session.remaining_speech_steps(), 0);
        assert!(session.is_finished());
        assert_eq!(TTS_SPEECH_WINDOW_SIZE, 6,);
        assert!(matches!(
            session.next_control().unwrap(),
            VibeVoiceRealtimeGenerationControl::Finished {
                reason: VibeVoiceRealtimeGenerationStopReason::ControlBudgetExhausted
            }
        ));
    }

    #[test]
    fn zero_budget_stops_before_first_speech_action() {
        let mut session = session(0);
        assert_text_window(&mut session, 0, &[101, 102, 103, 104, 105], 5);
        assert_eq!(
            session.next_control().unwrap(),
            VibeVoiceRealtimeGenerationControl::Finished {
                reason: VibeVoiceRealtimeGenerationStopReason::ControlBudgetExhausted
            }
        );
        assert_eq!(session.remaining_speech_steps(), 0);
    }

    #[test]
    fn non_cpu_backend_is_rejected_without_fallback() {
        let state = super::super::state::synthetic_generation_test_state();
        let error = VibeVoiceRealtimeGenerationSession::new(&state, &[101], BackendKind::Metal, 1)
            .unwrap_err();
        assert!(matches!(error, VokraError::UnsupportedOp(_)));
        assert!(error.to_string().contains("no CPU fallback"));
    }

    #[test]
    fn terminal_observations_are_explicit_and_idempotent() {
        let mut eos_session = session(24);
        eos_session.finish(VibeVoiceRealtimeGenerationStopReason::EndOfSpeech);
        eos_session.finish(VibeVoiceRealtimeGenerationStopReason::MaxLength);
        assert_eq!(
            eos_session.next_control().unwrap(),
            VibeVoiceRealtimeGenerationControl::Finished {
                reason: VibeVoiceRealtimeGenerationStopReason::EndOfSpeech,
            }
        );
        assert_eq!(
            eos_session.stop_reason(),
            Some(VibeVoiceRealtimeGenerationStopReason::EndOfSpeech)
        );
        assert!(eos_session.is_finished());

        let mut max_length_session = session(24);
        max_length_session.finish(VibeVoiceRealtimeGenerationStopReason::MaxLength);
        assert_eq!(
            max_length_session.next_control().unwrap(),
            VibeVoiceRealtimeGenerationControl::Finished {
                reason: VibeVoiceRealtimeGenerationStopReason::MaxLength,
            }
        );

        let mut externally_stopped = session(24);
        externally_stopped.finish(VibeVoiceRealtimeGenerationStopReason::ExternalStop);
        assert_eq!(
            externally_stopped.next_control().unwrap(),
            VibeVoiceRealtimeGenerationControl::Finished {
                reason: VibeVoiceRealtimeGenerationStopReason::ExternalStop,
            }
        );
    }
}
