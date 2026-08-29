//! Internal source-shaped CosyVoice3 route.
//!
//! This module records the exact batch-one contracts of the pinned upstream
//! implementation.  It is deliberately `pub(crate)`: an authenticated
//! composite checkpoint and numerical evidence are still required before a
//! production loader can be exposed.  In particular, this is not a reuse of
//! the CosyVoice2 conformer/flow path.

use vokra_core::{Result, VokraError};

use crate::strict_checkpoint::sha256_bytes;

#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const OFFICIAL_SOURCE_REVISION: &str = "0d990d60740bf174904a5185cce910b847bd3684";
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const OFFICIAL_MODEL_REVISION: &str = "29e01c4e8d000f4bcd70751be16fa94bf3d85a18";
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const OFFICIAL_REFERENCE_FORMAT: &str = "vokra-cosyvoice3-official-reference-v1";
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const SAMPLE_RATE: u32 = 24_000;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const LLM_INPUT_SIZE: usize = 896;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const LLM_OUTPUT_SIZE: usize = 896;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const SPEECH_TOKEN_SIZE: usize = 6_561;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const HEAD_SIZE: usize = SPEECH_TOKEN_SIZE + 200;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const SOS_TOKEN: u32 = SPEECH_TOKEN_SIZE as u32;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const EOS_TOKEN: u32 = SOS_TOKEN + 1;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const TASK_TOKEN: u32 = SOS_TOKEN + 2;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const FILL_TOKEN: u32 = SOS_TOKEN + 3;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const FLOW_INPUT_SIZE: usize = 80;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const FLOW_OUTPUT_SIZE: usize = 80;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const FLOW_SPEAKER_SIZE: usize = 192;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const FLOW_PRE_LOOKAHEAD: usize = 3;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const TOKEN_MEL_RATIO: usize = 2;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const FLOW_NOISE_FRAMES: usize = 15_000;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const FLOW_STEPS: usize = 10;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const FLOW_CFG_RATE: f32 = 0.7;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const FLOW_CHUNK_SIZE: usize = 25;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const DIT_DIM: usize = 1_024;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const DIT_DEPTH: usize = 22;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const DIT_HEADS: usize = 16;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const DIT_HEAD_DIM: usize = 64;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const DIT_FF_MULT: usize = 2;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const HIFT_BASE_CHANNELS: usize = 512;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const HIFT_HARMONICS: usize = 8;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const HIFT_UPSAMPLE_RATES: [usize; 3] = [8, 5, 3];
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const HIFT_UPSAMPLE_KERNELS: [usize; 3] = [16, 11, 7];
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const HIFT_NFFT: usize = 16;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const HIFT_HOP: usize = 4;
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub const HIFT_PRE_LOOK_RIGHT: usize = 4;

#[allow(dead_code)] // staged until the authenticated composite binder is wired
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RasConfig {
    pub top_p: f32,
    pub top_k: usize,
    pub window: usize,
    pub tau: f32,
}

#[allow(dead_code)] // staged until the authenticated composite binder is wired
impl Default for RasConfig {
    fn default() -> Self {
        Self {
            top_p: 0.8,
            top_k: 25,
            window: 10,
            tau: 0.1,
        }
    }
}

#[allow(dead_code)] // staged until the authenticated composite binder is wired
impl RasConfig {
    fn validate(self) -> Result<Self> {
        if !self.top_p.is_finite()
            || !(0.0..=1.0).contains(&self.top_p)
            || self.top_p == 0.0
            || self.top_k == 0
            || self.window == 0
            || !self.tau.is_finite()
            || self.tau < 0.0
        {
            return Err(VokraError::InvalidArgument(
                "cosyvoice3 native: invalid upstream RAS configuration".into(),
            ));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub(crate) enum InputKind {
    Sos,
    PromptText,
    TargetText,
    Task,
    PromptSpeech,
}

/// Flattened input rows in the order used by `Qwen2LM.inference`.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub(crate) struct LlmPrompt {
    rows: Vec<f32>,
    kinds: Vec<InputKind>,
}

#[allow(dead_code)] // staged until the authenticated composite binder is wired
impl LlmPrompt {
    pub(crate) fn build(
        sos: &[f32],
        prompt_text: &[f32],
        target_text: &[f32],
        task: &[f32],
        prompt_speech: &[f32],
    ) -> Result<Self> {
        if sos.len() != LLM_INPUT_SIZE || task.len() != LLM_INPUT_SIZE {
            return Err(VokraError::InvalidArgument(
                "cosyvoice3 native: SOS/task must each contain one 896-wide row".into(),
            ));
        }
        let parts = [
            (InputKind::Sos, sos),
            (InputKind::PromptText, prompt_text),
            (InputKind::TargetText, target_text),
            (InputKind::Task, task),
            (InputKind::PromptSpeech, prompt_speech),
        ];
        let mut rows = Vec::new();
        let mut kinds = Vec::new();
        for (kind, values) in parts {
            if values.len() % LLM_INPUT_SIZE != 0 || values.iter().any(|x| !x.is_finite()) {
                return Err(VokraError::InvalidArgument(format!(
                    "cosyvoice3 native: {kind:?} embeddings are not finite 896-wide rows"
                )));
            }
            rows.extend_from_slice(values);
            kinds.extend(std::iter::repeat_n(kind, values.len() / LLM_INPUT_SIZE));
        }
        if kinds.first() != Some(&InputKind::Sos)
            || !kinds.contains(&InputKind::Task)
            || kinds
                .iter()
                .filter(|kind| **kind == InputKind::Task)
                .count()
                != 1
        {
            return Err(VokraError::InvalidArgument(
                "cosyvoice3 native: malformed Qwen2 prompt sequence".into(),
            ));
        }
        Ok(Self { rows, kinds })
    }

    pub(crate) fn rows(&self) -> usize {
        self.kinds.len()
    }
    pub(crate) fn values(&self) -> &[f32] {
        &self.rows
    }
    pub(crate) fn kinds(&self) -> &[InputKind] {
        &self.kinds
    }
}

/// Full source `rand_noise` packet.  A digest supplied by the caller is not
/// enough: it is checked against the bytes, and the evidence identity must
/// match the fixed source/model pair.
#[derive(Debug, Clone)]
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub(crate) struct FlowNoise {
    values: Vec<f32>,
    sha256: [u8; 32],
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // evidence is consumed when the authenticated binder is enabled
pub(crate) struct NoiseEvidence<'a> {
    pub format: &'a str,
    pub source_revision: &'a str,
    pub model_revision: &'a str,
    pub status: &'a str,
    pub source_tap: &'a str,
    pub sha256: [u8; 32],
}

// Only code in this module can mint this capability.  In particular, a
// caller cannot turn arbitrary bytes plus a self-asserted digest into an
// authenticated flow packet.
#[allow(dead_code)]
struct FixedNoiseCapability(());
#[allow(dead_code)]
const FIXED_NOISE_CAPABILITY: FixedNoiseCapability = FixedNoiseCapability(());

#[allow(dead_code)] // staged until the authenticated composite binder is wired
impl FlowNoise {
    #[allow(dead_code)]
    fn authenticated(
        values: Vec<f32>,
        evidence: NoiseEvidence<'_>,
        _capability: &FixedNoiseCapability,
    ) -> Result<Self> {
        if evidence.format != OFFICIAL_REFERENCE_FORMAT
            || evidence.source_revision != OFFICIAL_SOURCE_REVISION
            || evidence.model_revision != OFFICIAL_MODEL_REVISION
            || evidence.status != "AUTHENTICATED_REFERENCE_EVIDENCE"
            || evidence.source_tap != "CausalConditionalCFM.rand_noise"
        {
            return Err(VokraError::InvalidArgument(
                "cosyvoice3 native: flow noise evidence identity mismatch".into(),
            ));
        }
        let expected = FLOW_INPUT_SIZE * FLOW_NOISE_FRAMES;
        if values.len() != expected
            || values.iter().any(|x| !x.is_finite())
            || values.iter().all(|x| *x == 0.0)
        {
            return Err(VokraError::InvalidArgument(
                "cosyvoice3 native: flow noise must be a non-empty finite [1,80,15000] tensor"
                    .into(),
            ));
        }
        let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        let actual = sha256_bytes(&bytes);
        if actual != evidence.sha256 {
            return Err(VokraError::InvalidArgument(
                "cosyvoice3 native: flow noise SHA-256 does not match authenticated evidence"
                    .into(),
            ));
        }
        Ok(Self {
            values,
            sha256: actual,
        })
    }

    pub(crate) fn values(&self) -> &[f32] {
        &self.values
    }
    pub(crate) fn digest(&self) -> [u8; 32] {
        self.sha256
    }
    pub(crate) fn prefix(&self, frames: usize) -> Result<Vec<f32>> {
        if frames > FLOW_NOISE_FRAMES {
            return Err(VokraError::InvalidArgument(
                "cosyvoice3 native: flow prefix exceeds authenticated noise".into(),
            ));
        }
        // CFM stores [channel, frame] contiguous rows and takes
        // `rand_noise[:, :, :mu.size(2)]`; gather each channel's prefix rather
        // than accidentally taking one flat prefix from channel zero.
        let mut out = Vec::with_capacity(FLOW_INPUT_SIZE * frames);
        for channel in 0..FLOW_INPUT_SIZE {
            let start = channel * FLOW_NOISE_FRAMES;
            out.extend_from_slice(&self.values[start..start + frames]);
        }
        Ok(out)
    }
}

/// Deterministic cosine time grid used by the official ten-step Euler CFM.
/// This is the direct translation of pinned upstream
/// `CausalConditionalCFM.forward`: `torch.linspace(0, 1, n_timesteps + 1)`
/// followed by `1 - torch.cos(t_span * 0.5 * torch.pi)` when the YAML tag is
/// `cosine`.  The estimator update is the source's `x = x + dt * dphi_dt`.
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub(crate) fn cosine_time_grid(steps: usize) -> Result<Vec<f32>> {
    if steps == 0 {
        return Err(VokraError::InvalidArgument(
            "cosyvoice3 native: flow steps must be positive".into(),
        ));
    }
    let pi = std::f32::consts::PI;
    Ok((0..=steps)
        .map(|i| 1.0 - (i as f32 / steps as f32 * 0.5 * pi).cos())
        .collect())
}

/// Source-shaped Euler update.  The estimator is injected by a future native
/// DiT binder; no fallback estimator or CPU substitution is constructed.
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub(crate) fn euler_step(x: &mut [f32], velocity: &[f32], dt: f32) -> Result<()> {
    if x.len() != velocity.len()
        || x.is_empty()
        || !dt.is_finite()
        || x.iter().chain(velocity).any(|v| !v.is_finite())
    {
        return Err(VokraError::InvalidArgument(
            "cosyvoice3 native: malformed Euler state".into(),
        ));
    }
    for (sample, derivative) in x.iter_mut().zip(velocity) {
        *sample += dt * derivative;
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "cosyvoice3 native: Euler state became non-finite".into(),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub(crate) struct SpeechTrace {
    pub yielded_tokens: Vec<u32>,
    pub sampled_tokens: Vec<u32>,
    pub terminated_eos: bool,
}

#[allow(dead_code)] // staged until the authenticated composite binder is wired
impl SpeechTrace {
    pub(crate) fn validate(self) -> Result<Self> {
        if self.yielded_tokens.is_empty()
            || self.sampled_tokens.len() < self.yielded_tokens.len()
            || self
                .sampled_tokens
                .iter()
                .any(|token| *token >= HEAD_SIZE as u32)
            || self.yielded_tokens.contains(&EOS_TOKEN)
        {
            return Err(VokraError::InvalidArgument(
                "cosyvoice3 native: invalid official speech-token trace".into(),
            ));
        }
        if self.terminated_eos && self.sampled_tokens.last() != Some(&EOS_TOKEN) {
            return Err(VokraError::InvalidArgument(
                "cosyvoice3 native: EOS termination lacks the final EOS sample".into(),
            ));
        }
        let yielded_prefix = &self.sampled_tokens[..self.yielded_tokens.len()];
        if yielded_prefix != self.yielded_tokens.as_slice() {
            return Err(VokraError::InvalidArgument(
                "cosyvoice3 native: yielded token order differs from sampled trace".into(),
            ));
        }
        if self.terminated_eos {
            if self.sampled_tokens.len() != self.yielded_tokens.len() + 1 {
                return Err(VokraError::InvalidArgument(
                    "cosyvoice3 native: EOS must be the sole non-yielded terminal sample".into(),
                ));
            }
        } else if self.sampled_tokens.len() != self.yielded_tokens.len() {
            return Err(VokraError::InvalidArgument(
                "cosyvoice3 native: non-EOS control sample was omitted from the yielded trace"
                    .into(),
            ));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub(crate) struct GeneratedMel {
    pub data: Vec<f32>,
    pub frames: usize,
    pub channels: usize,
}

#[allow(dead_code)] // staged until the authenticated composite binder is wired
impl GeneratedMel {
    fn validate(self) -> Result<Self> {
        let expected = FLOW_OUTPUT_SIZE.checked_mul(self.frames).ok_or_else(|| {
            VokraError::InvalidArgument("cosyvoice3 native: mel frame count overflow".into())
        })?;
        if self.frames == 0
            || self.channels != FLOW_OUTPUT_SIZE
            || self.data.len() != expected
            || self.data.iter().any(|x| !x.is_finite())
        {
            return Err(VokraError::InvalidArgument(
                "cosyvoice3 native: generated mel must be finite non-empty [80,frames]".into(),
            ));
        }
        Ok(self)
    }
}

/// Components injected by a future authenticated checkpoint binder.  Keeping
/// this seam internal allows parity workers to exercise source ordering while
/// preventing an incomplete model from becoming a public loader.
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub(crate) trait BatchOneComponents {
    fn generate_speech_tokens(
        &self,
        prompt: &LlmPrompt,
        min_tokens: usize,
        max_tokens: usize,
        sampling: RasConfig,
    ) -> Result<SpeechTrace>;
    fn generate_mel(
        &self,
        speech_tokens: &[u32],
        prompt_mel: &[f32],
        prompt_mel_frames: usize,
        streaming: bool,
        finalize: bool,
        flow_noise: &[f32],
    ) -> Result<GeneratedMel>;
    fn decode_hift(&self, mel: &[f32], frames: usize) -> Result<(u32, Vec<f32>)>;
}

/// Inputs for one official non-streaming batch-one route.
#[derive(Debug, Clone)]
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub(crate) struct BatchOneConditioning {
    pub sos: Vec<f32>,
    pub prompt_text: Vec<f32>,
    pub target_text: Vec<f32>,
    pub task: Vec<f32>,
    pub prompt_speech: Vec<f32>,
    pub prompt_mel: Vec<f32>,
    pub prompt_mel_frames: usize,
    pub min_tokens: usize,
    pub max_tokens: usize,
    pub streaming: bool,
    pub finalize: bool,
}

/// Source-shaped orchestration: Qwen2 AR → CausalMaskedDiffWithDiT →
/// generated-only mel → causal HiFT.  The flow implementation is supplied by
/// the caller; no CosyVoice2 flow, zero tensor, or silent CPU fallback exists.
#[allow(dead_code)] // staged until the authenticated composite binder is wired
pub(crate) struct BatchOneRoute<'a, C: BatchOneComponents> {
    components: &'a C,
    sampling: RasConfig,
}

#[allow(dead_code)] // staged until the authenticated composite binder is wired
impl<'a, C: BatchOneComponents> BatchOneRoute<'a, C> {
    pub(crate) fn new(components: &'a C, sampling: RasConfig) -> Result<Self> {
        Ok(Self {
            components,
            sampling: sampling.validate()?,
        })
    }

    pub(crate) fn run(
        &self,
        input: BatchOneConditioning,
        noise: &FlowNoise,
    ) -> Result<(SpeechTrace, GeneratedMel, Vec<f32>)> {
        if input.min_tokens == 0
            || input.max_tokens < input.min_tokens
            || input.max_tokens > 100_000
        {
            return Err(VokraError::InvalidArgument(
                "cosyvoice3 native: invalid AR token bounds".into(),
            ));
        }
        let prompt = LlmPrompt::build(
            &input.sos,
            &input.prompt_text,
            &input.target_text,
            &input.task,
            &input.prompt_speech,
        )?;
        let mel_len = FLOW_OUTPUT_SIZE
            .checked_mul(input.prompt_mel_frames)
            .ok_or_else(|| {
                VokraError::InvalidArgument("cosyvoice3 native: prompt mel frame overflow".into())
            })?;
        if input.prompt_mel.len() != mel_len || input.prompt_mel.iter().any(|x| !x.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "cosyvoice3 native: malformed prompt mel".into(),
            ));
        }
        let trace = self
            .components
            .generate_speech_tokens(&prompt, input.min_tokens, input.max_tokens, self.sampling)?
            .validate()?;
        let mel = self
            .components
            .generate_mel(
                &trace.yielded_tokens,
                &input.prompt_mel,
                input.prompt_mel_frames,
                input.streaming,
                input.finalize,
                noise
                    .prefix(input.prompt_mel_frames + trace.yielded_tokens.len() * TOKEN_MEL_RATIO)?
                    .as_slice(),
            )?
            .validate()?;
        let expected_frames = trace.yielded_tokens.len() * TOKEN_MEL_RATIO;
        if mel.frames != expected_frames {
            return Err(VokraError::InvalidArgument(format!(
                "cosyvoice3 native: generated mel frames {} != {} yielded tokens * {}",
                mel.frames,
                trace.yielded_tokens.len(),
                TOKEN_MEL_RATIO
            )));
        }
        let (sample_rate, pcm) = self.components.decode_hift(&mel.data, mel.frames)?;
        if sample_rate != SAMPLE_RATE
            || pcm.len() != mel.frames * 480
            || pcm.is_empty()
            || pcm.iter().any(|x| !x.is_finite() || x.abs() > 1.1)
        {
            return Err(VokraError::InvalidArgument(
                "cosyvoice3 native: HiFT PCM has wrong sample rate/length or is empty, non-finite, or out of bounds".into(),
            ));
        }
        Ok((trace, mel, pcm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_architecture_contract_is_distinct_from_cosyvoice2() {
        assert_eq!(SAMPLE_RATE, 24_000);
        assert_eq!(HEAD_SIZE, 6_761);
        assert_eq!(
            (SOS_TOKEN, EOS_TOKEN, TASK_TOKEN, FILL_TOKEN),
            (6561, 6562, 6563, 6564)
        );
        assert_eq!(
            (DIT_DIM, DIT_DEPTH, DIT_HEADS, DIT_HEAD_DIM, DIT_FF_MULT),
            (1024, 22, 16, 64, 2)
        );
        assert_eq!(HIFT_UPSAMPLE_RATES, [8, 5, 3]);
    }

    #[test]
    fn prompt_sequence_preserves_source_order() {
        let row = |value| vec![value; LLM_INPUT_SIZE];
        let prompt =
            LlmPrompt::build(&row(1.0), &row(2.0), &row(3.0), &row(4.0), &row(5.0)).unwrap();
        assert_eq!(prompt.rows(), 5);
        assert_eq!(
            prompt.kinds(),
            &[
                InputKind::Sos,
                InputKind::PromptText,
                InputKind::TargetText,
                InputKind::Task,
                InputKind::PromptSpeech
            ]
        );
        assert_eq!(prompt.values()[..LLM_INPUT_SIZE], row(1.0));
    }

    #[test]
    fn malformed_prompt_and_euler_fail_closed() {
        let row = vec![0.0; LLM_INPUT_SIZE];
        assert!(LlmPrompt::build(&row[..1], &[], &row, &row, &[]).is_err());
        assert!(euler_step(&mut [0.0], &[1.0], f32::NAN).is_err());
        assert!(cosine_time_grid(0).is_err());
    }

    #[test]
    fn ras_rejects_empty_top_p_domain() {
        let mut config = RasConfig::default();
        config.top_p = 0.0;
        assert!(config.validate().is_err());
        config.top_p = f32::NAN;
        assert!(config.validate().is_err());
    }

    #[test]
    fn speech_trace_keeps_non_yielded_control_samples_explicit() {
        let control = SpeechTrace {
            yielded_tokens: vec![7],
            sampled_tokens: vec![7, FILL_TOKEN],
            terminated_eos: false,
        };
        assert!(control.validate().is_err());

        let eos = SpeechTrace {
            yielded_tokens: vec![7],
            sampled_tokens: vec![7, EOS_TOKEN],
            terminated_eos: true,
        };
        assert!(eos.validate().is_ok());
    }

    #[test]
    fn noise_evidence_cannot_self_assert_wrong_digest() {
        let values = vec![1.0; FLOW_INPUT_SIZE * FLOW_NOISE_FRAMES];
        let evidence = NoiseEvidence {
            format: OFFICIAL_REFERENCE_FORMAT,
            source_revision: OFFICIAL_SOURCE_REVISION,
            model_revision: OFFICIAL_MODEL_REVISION,
            status: "AUTHENTICATED_REFERENCE_EVIDENCE",
            source_tap: "CausalConditionalCFM.rand_noise",
            sha256: [0; 32],
        };
        assert!(FlowNoise::authenticated(values, evidence, &FIXED_NOISE_CAPABILITY).is_err());
    }

    #[test]
    fn noise_evidence_cannot_self_assert_wrong_source_tap() {
        let values = vec![1.0; FLOW_INPUT_SIZE * FLOW_NOISE_FRAMES];
        let evidence = NoiseEvidence {
            format: OFFICIAL_REFERENCE_FORMAT,
            source_revision: OFFICIAL_SOURCE_REVISION,
            model_revision: OFFICIAL_MODEL_REVISION,
            status: "AUTHENTICATED_REFERENCE_EVIDENCE",
            source_tap: "caller.supplied_noise",
            sha256: [0; 32],
        };
        assert!(FlowNoise::authenticated(values, evidence, &FIXED_NOISE_CAPABILITY).is_err());
    }
}
