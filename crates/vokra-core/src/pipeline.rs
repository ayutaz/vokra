//! Declarative pipeline builder (M0-02-T13).
//!
//! FR-API-02 defines the declarative pipeline
//! `AudioPipeline::new().vad().asr().llm().tts().build()`. M0 keeps the
//! stage list and performs minimal validation only (an empty pipeline is
//! rejected); execution wiring happens in the M0-05〜M0-07 demos.

use crate::error::{Result, VokraError};

/// One stage of a declarative audio pipeline (FR-API-02).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    /// Voice activity detection (wired by M0-05, Silero VAD).
    Vad,
    /// Speech recognition (wired by M0-06, Whisper base).
    Asr,
    /// Language-model stage (integration point; wiring beyond M0).
    Llm,
    /// Speech synthesis (wired by M0-07, piper-plus native TTS).
    Tts,
}

/// Builder for a declarative audio pipeline (FR-API-02).
///
/// Stage methods consume `self` so the FR-API-02 chain works verbatim:
///
/// ```
/// use vokra_core::{AudioPipeline, PipelineStage};
///
/// let pipeline = AudioPipeline::new().vad().asr().llm().tts().build()?;
/// assert_eq!(
///     pipeline.stages(),
///     &[PipelineStage::Vad, PipelineStage::Asr, PipelineStage::Llm, PipelineStage::Tts],
/// );
/// # Ok::<(), vokra_core::VokraError>(())
/// ```
#[derive(Debug, Default)]
pub struct AudioPipeline {
    stages: Vec<PipelineStage>,
}

impl AudioPipeline {
    /// Creates an empty pipeline builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a voice-activity-detection stage.
    pub fn vad(mut self) -> Self {
        self.stages.push(PipelineStage::Vad);
        self
    }

    /// Appends a speech-recognition stage.
    pub fn asr(mut self) -> Self {
        self.stages.push(PipelineStage::Asr);
        self
    }

    /// Appends a language-model stage.
    pub fn llm(mut self) -> Self {
        self.stages.push(PipelineStage::Llm);
        self
    }

    /// Appends a speech-synthesis stage.
    pub fn tts(mut self) -> Self {
        self.stages.push(PipelineStage::Tts);
        self
    }

    /// Finalizes the pipeline.
    ///
    /// M0 behaviour: keeps the declared stage list and rejects an empty
    /// pipeline with
    /// [`VokraError::InvalidArgument`]; stage execution is wired by the
    /// M0-05〜M0-07 demos.
    pub fn build(self) -> Result<Pipeline> {
        if self.stages.is_empty() {
            return Err(VokraError::InvalidArgument(
                "audio pipeline must declare at least one stage".to_owned(),
            ));
        }
        Ok(Pipeline {
            stages: self.stages,
        })
    }
}

/// A built (declared) pipeline — the result of [`AudioPipeline::build`].
#[derive(Debug, Clone)]
pub struct Pipeline {
    stages: Vec<PipelineStage>,
}

impl Pipeline {
    /// Declared stages, in execution order.
    pub fn stages(&self) -> &[PipelineStage] {
        &self.stages
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fr_api_02_chain_builds_and_keeps_stage_order() {
        // FR-API-02 shape, verbatim:
        let pipeline = AudioPipeline::new()
            .vad()
            .asr()
            .llm()
            .tts()
            .build()
            .expect("valid");
        assert_eq!(
            pipeline.stages(),
            &[
                PipelineStage::Vad,
                PipelineStage::Asr,
                PipelineStage::Llm,
                PipelineStage::Tts
            ],
        );
    }

    #[test]
    fn partial_chain_is_allowed() {
        let pipeline = AudioPipeline::new().vad().asr().build().expect("valid");
        assert_eq!(pipeline.stages(), &[PipelineStage::Vad, PipelineStage::Asr]);
    }

    #[test]
    fn repeated_stages_are_kept_in_order() {
        let pipeline = AudioPipeline::new().tts().tts().build().expect("valid");
        assert_eq!(pipeline.stages(), &[PipelineStage::Tts, PipelineStage::Tts]);
    }

    #[test]
    fn empty_pipeline_is_rejected() {
        let result = AudioPipeline::new().build();
        assert!(matches!(result, Err(VokraError::InvalidArgument(_))));
    }
}
