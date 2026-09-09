//! Internal CosyVoice2 Qwen2 speech-token sampling seam.
//!
//! This wrapper owns no checkpoint binding and makes no production-support
//! claim.  It only joins an injected [`LlmBackbone`] with the authenticated
//! source-shaped embedding/head tensors and the native RAS sampler.  The
//! synthetic constructor and tests below are numerical fixtures, not model
//! weights.

use super::llm::{LlmBackbone, LlmBackboneStep};
use super::native::{
    InputKind, LlmCallEvidence, LlmInputSequence, RandomSource, SamplingCallEvidence,
    SamplingConfig, SpeechGeneration, TerminationReason, ras_sampling,
};
use vokra_core::{Result, VokraError};

const SPEECH_VOCAB_SIZE: usize = 6_561;
const CONTROL_ID_COUNT: usize = 3;
const HEAD_SIZE: usize = SPEECH_VOCAB_SIZE + CONTROL_ID_COUNT;
const EOS_ID: u32 = SPEECH_VOCAB_SIZE as u32;
const MAX_EOS_RETRIES: usize = 100;

/// Caller-supplied wrapper tensors in row-major layout.
///
/// `llm_embedding` contains SOS then TASK (`[2, d]`); all other tensors have
/// `HEAD_SIZE` speech/control rows.  These plain vectors are materialized F32
/// rows for a future authenticated binder.
#[derive(Debug, Clone)]
pub(crate) struct SpeechLmTensors {
    pub(crate) llm_embedding: Vec<f32>,
    pub(crate) speech_embedding: Vec<f32>,
    pub(crate) decoder_weight: Vec<f32>,
    pub(crate) decoder_bias: Vec<f32>,
}

/// Internal source-shaped Qwen2LM speech-token wrapper.
///
/// This is an injection seam only.  It does not load GGUF/safetensors or
/// authorize a CosyVoice2 model for production use.
pub(crate) struct SpeechLm<'a> {
    backbone: &'a LlmBackbone,
    tensors: &'a SpeechLmTensors,
    hidden_dim: usize,
}

impl<'a> SpeechLm<'a> {
    /// Binds wrapper tensors after strict shape, finite-value, and dimension
    /// checks.  The backbone remains borrowed and supplies the Compute-backed
    /// untied-head projection.
    pub(crate) fn new(backbone: &'a LlmBackbone, tensors: &'a SpeechLmTensors) -> Result<Self> {
        let hidden_dim = backbone.config().hidden_dim;
        if hidden_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 speech LM: backbone hidden dimension must be non-zero".into(),
            ));
        }
        let expected_llm = 2usize.checked_mul(hidden_dim).ok_or_else(|| {
            VokraError::InvalidArgument(
                "cosyvoice2 speech LM: llm_embedding shape overflows".into(),
            )
        })?;
        let expected_head = HEAD_SIZE.checked_mul(hidden_dim).ok_or_else(|| {
            VokraError::InvalidArgument("cosyvoice2 speech LM: speech head shape overflows".into())
        })?;
        if tensors.llm_embedding.len() != expected_llm
            || tensors.speech_embedding.len() != expected_head
            || tensors.decoder_weight.len() != expected_head
            || tensors.decoder_bias.len() != HEAD_SIZE
        {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 speech LM: wrapper tensor shapes must be llm [2,d], speech [6564,d], decoder.weight [6564,d], bias [6564]".into(),
            ));
        }
        if tensors
            .llm_embedding
            .iter()
            .chain(&tensors.speech_embedding)
            .chain(&tensors.decoder_weight)
            .chain(&tensors.decoder_bias)
            .any(|value| !value.is_finite())
        {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 speech LM: wrapper tensors contain non-finite values".into(),
            ));
        }
        Ok(Self {
            backbone,
            tensors,
            hidden_dim,
        })
    }

    fn initial_input(&self, input: &LlmInputSequence) -> Result<(Vec<f32>, usize)> {
        let rows = input.rows();
        let row_width = input.row_width();
        if rows == 0 || row_width != self.hidden_dim {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 speech LM: input sequence width/rows do not match the backbone".into(),
            ));
        }
        let expected = rows.checked_mul(self.hidden_dim).ok_or_else(|| {
            VokraError::InvalidArgument(
                "cosyvoice2 speech LM: input sequence shape overflows".into(),
            )
        })?;
        if input.embeddings().len() != expected
            || input.embeddings().iter().any(|value| !value.is_finite())
        {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 speech LM: input sequence has invalid embedding rows".into(),
            ));
        }
        let mut sos_count = 0;
        let mut task_count = 0;
        let mut embeddings = Vec::with_capacity(expected);
        for (row, kind) in input.kinds().iter().enumerate() {
            let source = match kind {
                InputKind::Sos => {
                    sos_count += 1;
                    &self.tensors.llm_embedding[..self.hidden_dim]
                }
                InputKind::Task => {
                    task_count += 1;
                    &self.tensors.llm_embedding[self.hidden_dim..2 * self.hidden_dim]
                }
                InputKind::PromptText | InputKind::TargetText | InputKind::PromptSpeech => {
                    &input.embeddings()[row * self.hidden_dim..(row + 1) * self.hidden_dim]
                }
            };
            embeddings.extend_from_slice(source);
        }
        if sos_count != 1 || task_count != 1 {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 speech LM: input sequence must contain exactly one SOS and TASK row"
                    .into(),
            ));
        }
        Ok((embeddings, rows))
    }

    fn speech_row(&self, token: u32) -> Result<Vec<f32>> {
        if token as usize >= SPEECH_VOCAB_SIZE {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 speech LM: control token cannot be used as a speech embedding".into(),
            ));
        }
        let index = token as usize * self.hidden_dim;
        Ok(self.tensors.speech_embedding[index..index + self.hidden_dim].to_vec())
    }

    /// Generates speech/control IDs with the official outer-loop and retry
    /// semantics.  `min_tokens` is the number of outer steps during which EOS
    /// is ignored; `max_tokens` counts outer LLM calls, not retry attempts.
    pub(crate) fn generate(
        &self,
        input: &LlmInputSequence,
        min_tokens: usize,
        max_tokens: usize,
        sampling: SamplingConfig,
        random: &mut dyn RandomSource,
    ) -> Result<SpeechGeneration> {
        if max_tokens == 0 || min_tokens > max_tokens {
            return Err(VokraError::InvalidArgument(
                "cosyvoice2 speech LM: generation bounds must satisfy 0 < max_tokens and min_tokens <= max_tokens".into(),
            ));
        }
        let (mut lm_input, mut input_rows) = self.initial_input(input)?;
        let mut state = LlmBackboneStep::new();
        let mut yielded_tokens = Vec::new();
        let mut sampled_tokens = Vec::new();
        let mut llm_calls = Vec::new();
        let mut sampling_calls = Vec::new();
        let mut termination = TerminationReason::MaxTokens;

        for generation_step in 0..max_tokens {
            let hidden = self
                .backbone
                .step_embeddings(&mut state, &lm_input, input_rows)?;
            llm_calls.push(LlmCallEvidence {
                call_index: generation_step,
                input_rows,
                output_rows: input_rows,
            });
            let hidden_start = (input_rows - 1) * self.hidden_dim;
            let last_hidden = &hidden[hidden_start..hidden_start + self.hidden_dim];
            let logits = self.backbone.project_linear_head(
                last_hidden,
                1,
                HEAD_SIZE,
                &self.tensors.decoder_weight,
                Some(&self.tensors.decoder_bias),
            )?;
            let log_probs = stable_log_softmax(&logits)?;
            let mut accepted = None;
            for attempt_index in 0..=MAX_EOS_RETRIES {
                let selected = ras_sampling(&log_probs, &yielded_tokens, sampling, random)?;
                let ignore_eos = generation_step < min_tokens;
                let ignored_eos = selected == EOS_ID && ignore_eos;
                let stop = selected == EOS_ID && !ignore_eos;
                let yielded = selected < EOS_ID;
                let skipped = selected > EOS_ID;
                sampled_tokens.push(selected);
                sampling_calls.push(SamplingCallEvidence {
                    call_index: sampling_calls.len(),
                    generation_step,
                    attempt_index,
                    selected_token: selected,
                    ignore_eos,
                    decoded_count: yielded_tokens.len(),
                    yielded,
                    skipped,
                    ignored_eos,
                    stop,
                });
                if ignored_eos {
                    continue;
                }
                accepted = Some(selected);
                break;
            }
            let selected = accepted.ok_or_else(|| {
                VokraError::InvalidArgument(
                    "cosyvoice2 speech LM: EOS retry guard exhausted after 100 retries".into(),
                )
            })?;
            if selected == EOS_ID {
                termination = TerminationReason::Eos;
                break;
            }
            if selected < EOS_ID {
                yielded_tokens.push(selected);
                lm_input = self.speech_row(selected)?;
                input_rows = 1;
            }
        }

        SpeechGeneration {
            yielded_tokens,
            termination,
            min_tokens,
            max_tokens,
            configured_vllm_stop_token_ids: vec![EOS_ID, EOS_ID + 1, EOS_ID + 2],
            native_terminal_eos: EOS_ID,
            sampled_tokens,
            llm_calls,
            sampling_calls,
        }
        .validate()
    }
}

fn stable_log_softmax(logits: &[f32]) -> Result<Vec<f32>> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "cosyvoice2 speech LM: logits must be non-empty and finite".into(),
        ));
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = logits
        .iter()
        .map(|value| (*value - maximum).exp())
        .try_fold(0.0f32, |sum, value| {
            let next = sum + value;
            next.is_finite().then_some(next)
        })
        .ok_or_else(|| {
            VokraError::InvalidArgument(
                "cosyvoice2 speech LM: log-softmax mass is non-finite".into(),
            )
        })?;
    if sum <= 0.0 || !sum.is_finite() {
        return Err(VokraError::InvalidArgument(
            "cosyvoice2 speech LM: log-softmax mass is not positive".into(),
        ));
    }
    let log_sum = sum.ln();
    let output: Vec<f32> = logits
        .iter()
        .map(|value| (*value - maximum) - log_sum)
        .collect();
    if output.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "cosyvoice2 speech LM: log-softmax output is non-finite".into(),
        ));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Draws {
        values: Vec<f32>,
        cursor: usize,
    }

    impl Draws {
        fn new(values: &[f32]) -> Self {
            Self {
                values: values.to_vec(),
                cursor: 0,
            }
        }
    }

    impl RandomSource for Draws {
        fn next_f32(&mut self) -> f32 {
            let value = self.values[self.cursor];
            self.cursor += 1;
            value
        }
    }

    fn backbone_with_context(n_ctx: usize) -> LlmBackbone {
        LlmBackbone::synthesized(
            super::super::llm::LlmBackboneConfig {
                vocab_size: 16,
                hidden_dim: 8,
                n_layer: 2,
                n_head_q: 2,
                n_head_kv: 1,
                ffn_dim: 16,
                rope_base: 10_000.0,
                rms_norm_eps: 1e-5,
                n_ctx,
            },
            17,
        )
        .unwrap()
    }

    fn backbone() -> LlmBackbone {
        backbone_with_context(8)
    }

    fn tensors(decoder_bias: &[f32]) -> SpeechLmTensors {
        let hidden = 8;
        SpeechLmTensors {
            llm_embedding: vec![0.0; 2 * hidden],
            speech_embedding: vec![0.0; HEAD_SIZE * hidden],
            decoder_weight: vec![0.0; HEAD_SIZE * hidden],
            decoder_bias: decoder_bias.to_vec(),
        }
    }

    fn input(prompt_rows: usize) -> LlmInputSequence {
        LlmInputSequence::new(
            8,
            &[0.0; 8],
            &vec![0.0; prompt_rows * 8],
            &[0.0; 8],
            &[0.0; 8],
            &[0.0; 8],
        )
        .unwrap()
    }

    #[test]
    fn prompt_prefix_becomes_single_speech_row() {
        let backbone = backbone();
        let wrapper = tensors(&[0.0; HEAD_SIZE]);
        let model = SpeechLm::new(&backbone, &wrapper).unwrap();
        let mut draws = Draws::new(&[0.0, 0.0]);
        let generated = model
            .generate(
                &input(2),
                0,
                2,
                SamplingConfig {
                    top_p: 1.0,
                    top_k: HEAD_SIZE,
                    tau_r: 2.0,
                    ..SamplingConfig::default()
                },
                &mut draws,
            )
            .unwrap();
        assert_eq!(generated.termination, TerminationReason::MaxTokens);
        assert_eq!(generated.yielded_tokens, [0, 0]);
        assert_eq!(generated.llm_calls[0].input_rows, 6);
        assert_eq!(generated.llm_calls[1].input_rows, 1);
        assert_eq!(draws.cursor, 2);
    }

    #[test]
    fn ignored_eos_retries_without_an_extra_llm_call() {
        let backbone = backbone();
        let wrapper = tensors(&[0.0; HEAD_SIZE]);
        let model = SpeechLm::new(&backbone, &wrapper).unwrap();
        let mut draws = Draws::new(&[0.9996, 0.0, 0.9996]);
        let generated = model
            .generate(
                &input(1),
                1,
                2,
                SamplingConfig {
                    top_p: 1.0,
                    top_k: HEAD_SIZE,
                    tau_r: 2.0,
                    ..SamplingConfig::default()
                },
                &mut draws,
            )
            .unwrap();
        assert_eq!(generated.termination, TerminationReason::Eos);
        assert_eq!(generated.yielded_tokens, [0]);
        assert_eq!(generated.llm_calls.len(), 2);
        assert_eq!(generated.sampling_calls.len(), 3);
        assert!(generated.sampling_calls[0].ignored_eos);
        assert_eq!(generated.sampling_calls[0].attempt_index, 0);
        assert_eq!(generated.sampling_calls[1].attempt_index, 1);
        assert_eq!(generated.sampling_calls[1].generation_step, 0);
        assert_eq!(draws.cursor, 3);
    }

    #[test]
    fn control_skip_preserves_single_row_transition() {
        let backbone = backbone();
        let wrapper = tensors(&[0.0; HEAD_SIZE]);
        let model = SpeechLm::new(&backbone, &wrapper).unwrap();
        let mut draws = Draws::new(&[0.0, 0.99975, 0.9996]);
        let generated = model
            .generate(
                &input(1),
                0,
                3,
                SamplingConfig {
                    top_p: 1.0,
                    top_k: HEAD_SIZE,
                    tau_r: 2.0,
                    ..SamplingConfig::default()
                },
                &mut draws,
            )
            .unwrap();
        assert_eq!(generated.termination, TerminationReason::Eos);
        assert_eq!(generated.yielded_tokens, [0]);
        assert_eq!(generated.sampled_tokens, [0, EOS_ID + 1, EOS_ID]);
        assert_eq!(
            generated
                .llm_calls
                .iter()
                .map(|call| call.input_rows)
                .collect::<Vec<_>>(),
            [5, 1, 1]
        );
    }

    #[test]
    fn eos_and_max_termination_are_distinct() {
        let backbone = backbone();
        let wrapper = tensors(&[0.0; HEAD_SIZE]);
        let model = SpeechLm::new(&backbone, &wrapper).unwrap();
        let sampling = SamplingConfig {
            top_p: 1.0,
            top_k: HEAD_SIZE,
            tau_r: 2.0,
            ..SamplingConfig::default()
        };
        let mut eos_draw = Draws::new(&[0.9996]);
        let eos = model
            .generate(&input(0), 0, 3, sampling, &mut eos_draw)
            .unwrap();
        assert_eq!(eos.termination, TerminationReason::Eos);
        assert_eq!(eos.llm_calls.len(), 1);

        let mut max_draw = Draws::new(&[0.0, 0.0]);
        let maxed = model
            .generate(&input(0), 0, 2, sampling, &mut max_draw)
            .unwrap();
        assert_eq!(maxed.termination, TerminationReason::MaxTokens);
        assert_eq!(maxed.llm_calls.len(), 2);
    }

    #[test]
    fn first_control_reuses_full_multirow_prefix_without_reset() {
        let backbone = backbone_with_context(32);
        let wrapper = tensors(&[0.0; HEAD_SIZE]);
        let model = SpeechLm::new(&backbone, &wrapper).unwrap();
        let mut draws = Draws::new(&[0.99975, 0.0, 0.9996]);
        let generated = model
            .generate(
                &input(1),
                0,
                3,
                SamplingConfig {
                    top_p: 1.0,
                    top_k: HEAD_SIZE,
                    tau_r: 2.0,
                    ..SamplingConfig::default()
                },
                &mut draws,
            )
            .unwrap();
        assert_eq!(generated.sampled_tokens, [EOS_ID + 1, 0, EOS_ID]);
        assert_eq!(generated.yielded_tokens, [0]);
        assert_eq!(
            generated
                .llm_calls
                .iter()
                .map(|call| call.input_rows)
                .collect::<Vec<_>>(),
            [5, 5, 1]
        );
        assert_eq!(draws.cursor, 3);
    }

    #[test]
    fn control_preserves_context_capacity_without_reset() {
        let backbone = backbone_with_context(9);
        let wrapper = tensors(&[0.0; HEAD_SIZE]);
        let model = SpeechLm::new(&backbone, &wrapper).unwrap();
        let mut draws = Draws::new(&[0.99975]);
        let error = model
            .generate(
                &input(1),
                0,
                2,
                SamplingConfig {
                    top_p: 1.0,
                    top_k: HEAD_SIZE,
                    tau_r: 2.0,
                    ..SamplingConfig::default()
                },
                &mut draws,
            )
            .unwrap_err();
        assert!(matches!(error, VokraError::InvalidArgument(_)));
        assert_eq!(draws.cursor, 1);
    }

    #[test]
    fn eos_retry_guard_allows_initial_plus_one_hundred_retries() {
        let backbone = backbone();
        let wrapper = tensors(&[0.0; HEAD_SIZE]);
        let model = SpeechLm::new(&backbone, &wrapper).unwrap();
        let mut draws = Draws::new(&[0.9996; 101]);
        let error = model
            .generate(
                &input(0),
                1,
                1,
                SamplingConfig {
                    top_p: 1.0,
                    top_k: HEAD_SIZE,
                    tau_r: 2.0,
                    ..SamplingConfig::default()
                },
                &mut draws,
            )
            .unwrap_err();
        assert!(matches!(error, VokraError::InvalidArgument(_)));
        assert_eq!(draws.cursor, 101);
    }

    #[test]
    fn stable_log_softmax_handles_extreme_finite_logits() {
        let output = stable_log_softmax(&[f32::MAX, f32::MAX]).unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
        assert!((output[0] + std::f32::consts::LN_2).abs() < 1e-6);
        assert!((output[1] + std::f32::consts::LN_2).abs() < 1e-6);
    }

    #[test]
    fn wrapper_rejects_invalid_tensors_and_bounds() {
        let backbone = backbone();
        let mut invalid = tensors(&[0.0; HEAD_SIZE]);
        invalid.decoder_bias.pop();
        assert!(SpeechLm::new(&backbone, &invalid).is_err());

        let mut invalid = tensors(&[0.0; HEAD_SIZE]);
        invalid.speech_embedding[0] = f32::NAN;
        assert!(SpeechLm::new(&backbone, &invalid).is_err());

        let wrapper = tensors(&[0.0; HEAD_SIZE]);
        let model = SpeechLm::new(&backbone, &wrapper).unwrap();
        let mut draws = Draws::new(&[0.0]);
        assert!(
            model
                .generate(&input(0), 0, 0, SamplingConfig::default(), &mut draws)
                .is_err()
        );
        assert!(
            model
                .generate(&input(0), 2, 1, SamplingConfig::default(), &mut draws)
                .is_err()
        );
        assert!(
            model
                .generate(
                    &input(0),
                    0,
                    1,
                    SamplingConfig {
                        top_k: 0,
                        ..SamplingConfig::default()
                    },
                    &mut draws,
                )
                .is_err()
        );
    }
}
