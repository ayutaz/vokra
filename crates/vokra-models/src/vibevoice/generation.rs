//! Single-sample VibeVoice composite generation.
//!
//! This module is deliberately an offline, typed generation boundary.  It
//! owns every mutable cache locally and accepts every random draw from the
//! caller, so an error cannot partially mutate a reusable model handle and no
//! hidden RNG can make a result non-reproducible.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use super::{
    BOS_EOS_TOKEN_ID, Qwen2Runtime, SpeechConnector, VibeVoiceAcousticDecoder,
    VibeVoiceDiffusionHead, VibeVoiceDpmSolverMultistep, VibeVoiceLatentScale,
    VibeVoiceTokenizerEncoder, VibeVoiceTokenizerStream, combine_next_lm_embedding,
};

const SAMPLE_RATE: u32 = 24_000;
const PROMPT_STD: f32 = 0.5 / 0.8;
const LATENT_WIDTH: usize = 64;
const DIFFUSION_STEPS: usize = 20;
const TRAIN_STEPS: usize = 1_000;

/// All deterministic and random inputs for one batch-1 VibeVoice request.
#[derive(Debug, Clone)]
pub struct VibeVoiceGenerationPacket {
    /// Already-tokenized Qwen prompt, including any speech marker positions.
    pub token_ids: Vec<u32>,
    /// Token positions replaced by one prompt acoustic+semantic row each.
    pub speech_replacement_positions: Vec<usize>,
    /// Optional 24 kHz mono prompt PCM. Length must be a multiple of 3,200.
    pub prompt_pcm: Option<Vec<f32>>,
    /// Sample rate for `prompt_pcm`; must be exactly 24,000 when present.
    pub prompt_sample_rate_hz: u32,
    /// Caller-owned Gaussian draws, row-major `[prompt_frames, 64]`.
    pub prompt_latent_draws: Vec<f32>,
    /// One caller-owned 64-wide Gaussian draw per generated diffusion token.
    pub diffusion_initial_draws: Vec<Vec<f32>>,
    /// Classifier-free guidance scale selected by the caller.
    pub guidance_scale: f32,
    /// Maximum number of constrained speech tokens to generate.
    pub max_generated_tokens: usize,
}

/// Result of one deterministic batch-1 composite request.
#[derive(Debug, Clone)]
pub struct VibeVoiceGenerationResult {
    /// Constrained tokens selected after the prompt.
    pub generated_tokens: Vec<u32>,
    /// Concatenated 24 kHz mono PCM decoded from generated acoustic latents.
    pub pcm: Vec<f32>,
    /// Whether the source loop returned because the configured cap was hit
    /// before EOS. Partial output is explicit rather than silently rejected.
    pub reached_max_steps: bool,
}

/// Authenticated VibeVoice composite runtime with one selected backend.
#[derive(Debug)]
pub struct VibeVoiceComposite {
    qwen: Qwen2Runtime,
    diffusion: VibeVoiceDiffusionHead,
    acoustic_connector: SpeechConnector,
    semantic_connector: SpeechConnector,
    acoustic_encoder: VibeVoiceTokenizerEncoder,
    semantic_encoder: VibeVoiceTokenizerEncoder,
    acoustic_decoder: VibeVoiceAcousticDecoder,
    latent_scale: VibeVoiceLatentScale,
    backend: BackendKind,
}

impl VibeVoiceComposite {
    /// Loads every composite section from the same authenticated GGUF.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        let qwen = Qwen2Runtime::from_gguf_with_backend(file, backend)?;
        let diffusion = VibeVoiceDiffusionHead::from_gguf_with_backend(file, backend)?;
        let acoustic_connector = SpeechConnector::acoustic_from_gguf(file, backend)?;
        let semantic_connector = SpeechConnector::semantic_from_gguf(file, backend)?;
        let acoustic_encoder = VibeVoiceTokenizerEncoder::acoustic_from_gguf(file, backend)?;
        let semantic_encoder = VibeVoiceTokenizerEncoder::semantic_from_gguf(file, backend)?;
        let acoustic_decoder = VibeVoiceAcousticDecoder::from_gguf(file, backend)?;
        let latent_scale = VibeVoiceLatentScale::from_gguf(file)?;
        Ok(Self {
            qwen,
            diffusion,
            acoustic_connector,
            semantic_connector,
            acoustic_encoder,
            semantic_encoder,
            acoustic_decoder,
            latent_scale,
            backend,
        })
    }

    /// Returns the one backend selected at construction.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Runs the fixed single-sample composite path.
    pub fn generate(
        &self,
        packet: &VibeVoiceGenerationPacket,
    ) -> Result<VibeVoiceGenerationResult> {
        validate_packet(packet)?;
        let mut acoustic_stream = self.acoustic_encoder.stream();
        let mut semantic_stream = self.semantic_encoder.stream();
        let mut decoder_stream = self.acoustic_decoder.stream();

        let prompt_replacements = self.prompt_replacements(packet, &mut acoustic_stream)?;
        let mut positive = self.qwen.fork_empty_cache();
        // The batch-1 unconditional branch is the official one-token
        // speech-start context, not a copy of the positive prompt. This is
        // intentionally explicit: no implicit text/audio negative prompt is
        // synthesized here.
        let mut negative = self.qwen.fork_empty_cache();
        let positive_hidden = if let Some(replacements) = &prompt_replacements {
            let refs = replacement_refs(replacements);
            positive.prefill_mixed_embeddings(&packet.token_ids, &refs)?
        } else {
            positive.prefill(&packet.token_ids)?
        };
        let negative_hidden = negative.prefill(&[super::SPEECH_START_TOKEN_ID])?;
        let hidden_width = positive.config().hidden_size;
        let last_positive = last_hidden(&positive_hidden, hidden_width)?;
        let last_negative = last_hidden(&negative_hidden, hidden_width)?;
        let mut positive_hidden = last_positive.to_vec();
        let mut negative_hidden = last_negative.to_vec();
        let mut generated_tokens = Vec::new();
        let mut pcm = Vec::new();
        let mut noise_index = 0;
        let mut terminated = false;

        for _ in 0..packet.max_generated_tokens {
            let positive_logits = positive.logits(&positive_hidden)?;
            let token = constrained_greedy_token(&positive_logits)?;
            generated_tokens.push(token);
            let plan = token_plan(token);
            debug_assert_eq!(plan.cache_advances, if plan.terminal { 0 } else { 1 });
            if plan.terminal {
                terminated = true;
                break;
            }
            if plan.diffusion {
                debug_assert_eq!(plan.negative_action, NegativeAction::SpeechEmbedding);
                let noise = packet
                    .diffusion_initial_draws
                    .get(noise_index)
                    .ok_or_else(|| {
                        VokraError::InvalidArgument(
                            "vibevoice generation is missing a diffusion Gaussian draw".to_owned(),
                        )
                    })?;
                let latent = self.sample_diffusion(
                    &positive_hidden,
                    &negative_hidden,
                    noise,
                    packet.guidance_scale,
                )?;
                noise_index += 1;
                let unscaled = self.latent_scale.unscale_generated(&latent)?;
                let pcm_chunk = decoder_stream.decode_chunk(&unscaled, 1)?;
                let semantic_rows = semantic_stream.encode_chunk(&pcm_chunk)?;
                let semantic = last_hidden(&semantic_rows, self.semantic_encoder.output_dim())?;
                let next_embedding = combine_next_lm_embedding(
                    &self.acoustic_connector,
                    &latent,
                    &self.semantic_connector,
                    semantic,
                )?;
                positive_hidden = positive.step_embedding(&next_embedding)?;
                negative_hidden = negative.step_embedding(&next_embedding)?;
                pcm.extend_from_slice(&pcm_chunk);
            } else if plan.refresh_negative {
                debug_assert_eq!(plan.negative_action, NegativeAction::RefreshSpeechStart);
                positive_hidden = positive.step(token)?;
                negative = self.qwen.fork_empty_cache();
                negative_hidden = negative.prefill(&[super::SPEECH_START_TOKEN_ID])?;
            } else if plan.clear_codec {
                debug_assert_eq!(plan.negative_action, NegativeAction::Hold);
                // The source clears codec state at a speech boundary but
                // continues autoregressive generation until EOS.
                positive_hidden = positive.step(token)?;
                acoustic_stream.set_to_zero();
                semantic_stream.set_to_zero();
                decoder_stream.set_to_zero();
            } else {
                debug_assert_eq!(plan.negative_action, NegativeAction::Hold);
                positive_hidden = positive.step(token)?;
            }
        }
        if noise_index != packet.diffusion_initial_draws.len() {
            return Err(VokraError::InvalidArgument(
                "vibevoice generation supplied unused diffusion Gaussian draws".to_owned(),
            ));
        }
        Ok(VibeVoiceGenerationResult {
            generated_tokens,
            pcm,
            reached_max_steps: !terminated,
        })
    }

    #[allow(clippy::type_complexity)] // tuple shape mirrors the source prompt contract
    fn prompt_replacements(
        &self,
        packet: &VibeVoiceGenerationPacket,
        acoustic_stream: &mut VibeVoiceTokenizerStream,
    ) -> Result<Option<Vec<(usize, Vec<f32>)>>> {
        let Some(pcm) = packet.prompt_pcm.as_deref() else {
            if packet.speech_replacement_positions.is_empty()
                && packet.prompt_latent_draws.is_empty()
            {
                return Ok(None);
            }
            return Err(VokraError::InvalidArgument(
                "vibevoice prompt draws/replacements require prompt PCM".to_owned(),
            ));
        };
        // The pinned source's `_process_speech_inputs` replaces prompt rows
        // with the acoustic connector only. Semantic conditioning starts
        // after generated PCM is re-encoded below; do not encode prompt PCM
        // through the semantic connector here.
        let acoustic_rows = acoustic_stream.encode_chunk(pcm)?;
        let frames = acoustic_rows.len() / self.acoustic_encoder.output_dim();
        if frames == 0 || acoustic_rows.len() != frames * LATENT_WIDTH {
            return Err(VokraError::ModelLoad(
                "vibevoice acoustic prompt encoder returned invalid row shape".to_owned(),
            ));
        }
        if packet.speech_replacement_positions.len() != frames
            || packet.prompt_latent_draws.len() != frames * LATENT_WIDTH
        {
            return Err(VokraError::InvalidArgument(
                "vibevoice prompt row/draw/replacement count mismatch".to_owned(),
            ));
        }
        // The fixed source uses the configured Gaussian scale directly; its
        // only stochastic prompt input is randn_like(mean), captured in
        // `prompt_latent_draws`. There is no independent scalar std draw.
        let std = PROMPT_STD;
        let mut replacements = Vec::with_capacity(frames);
        for frame in 0..frames {
            let mean = &acoustic_rows[frame * LATENT_WIDTH..(frame + 1) * LATENT_WIDTH];
            let draws =
                &packet.prompt_latent_draws[frame * LATENT_WIDTH..(frame + 1) * LATENT_WIDTH];
            let sampled: Vec<f32> = mean
                .iter()
                .zip(draws)
                .map(|(&mean, &draw)| mean + std * draw)
                .collect();
            let scaled = self.latent_scale.scale_raw(&sampled)?;
            let embedding = self.acoustic_connector.forward(&scaled)?;
            replacements.push((packet.speech_replacement_positions[frame], embedding));
        }
        Ok(Some(replacements))
    }

    fn sample_diffusion(
        &self,
        positive_condition: &[f32],
        negative_condition: &[f32],
        initial_noise: &[f32],
        guidance_scale: f32,
    ) -> Result<Vec<f32>> {
        if initial_noise.len() != LATENT_WIDTH {
            return Err(VokraError::InvalidArgument(
                "vibevoice diffusion initial draw must have width 64".to_owned(),
            ));
        }
        if !guidance_scale.is_finite() {
            return Err(VokraError::InvalidArgument(
                "vibevoice guidance scale must be finite".to_owned(),
            ));
        }
        let mut scheduler = VibeVoiceDpmSolverMultistep::new(TRAIN_STEPS, DIFFUSION_STEPS)?;
        let mut sample = initial_noise.to_vec();
        let timesteps = scheduler.timesteps().to_vec();
        for timestep in timesteps {
            let positive = self
                .diffusion
                .forward(&sample, positive_condition, timestep as f32)?;
            let negative = self
                .diffusion
                .forward(&sample, negative_condition, timestep as f32)?;
            let guided: Vec<f32> = positive
                .iter()
                .zip(negative)
                .map(|(&pos, neg)| neg + guidance_scale * (pos - neg))
                .collect();
            sample = scheduler.step(&guided, timestep, &sample)?.sample;
        }
        Ok(sample)
    }
}

fn validate_packet(packet: &VibeVoiceGenerationPacket) -> Result<()> {
    if packet.token_ids.is_empty() || packet.max_generated_tokens == 0 {
        return Err(VokraError::InvalidArgument(
            "vibevoice generation requires non-empty tokens and generation limit".to_owned(),
        ));
    }
    if let Some(pcm) = &packet.prompt_pcm {
        if packet.prompt_sample_rate_hz != SAMPLE_RATE || pcm.is_empty() || pcm.len() % 3_200 != 0 {
            return Err(VokraError::InvalidArgument(
                "vibevoice prompt PCM must be non-empty and 3200-sample aligned".to_owned(),
            ));
        }
        if pcm.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "vibevoice prompt PCM contains non-finite values".to_owned(),
            ));
        }
    } else if packet.prompt_sample_rate_hz != 0 {
        return Err(VokraError::InvalidArgument(
            "vibevoice prompt sample rate is set without prompt PCM".to_owned(),
        ));
    }
    if packet
        .prompt_latent_draws
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(
            "vibevoice prompt Gaussian draws must be finite".to_owned(),
        ));
    }
    if !packet.guidance_scale.is_finite() {
        return Err(VokraError::InvalidArgument(
            "vibevoice guidance scale must be finite".to_owned(),
        ));
    }
    if packet
        .diffusion_initial_draws
        .iter()
        .any(|draw| draw.len() != LATENT_WIDTH || draw.iter().any(|value| !value.is_finite()))
    {
        return Err(VokraError::InvalidArgument(
            "vibevoice diffusion draws must be finite width-64 rows".to_owned(),
        ));
    }
    Ok(())
}

fn replacement_refs(replacements: &[(usize, Vec<f32>)]) -> Vec<(usize, &[f32])> {
    replacements
        .iter()
        .map(|(index, embedding)| (*index, embedding.as_slice()))
        .collect()
}

fn last_hidden(rows: &[f32], width: usize) -> Result<&[f32]> {
    if width == 0 || rows.is_empty() || rows.len() % width != 0 {
        return Err(VokraError::ModelLoad(
            "vibevoice hidden rows have invalid shape".to_owned(),
        ));
    }
    Ok(&rows[rows.len() - width..])
}

fn constrained_greedy_token(logits: &[f32]) -> Result<u32> {
    let allowed = [
        BOS_EOS_TOKEN_ID,
        super::SPEECH_END_TOKEN_ID,
        super::SPEECH_DIFFUSION_TOKEN_ID,
        super::SPEECH_START_TOKEN_ID,
    ];
    let mut best = None;
    for token in allowed {
        let value = *logits.get(token as usize).ok_or_else(|| {
            VokraError::ModelLoad("vibevoice constrained token exceeds logits".to_owned())
        })?;
        if value == f32::NEG_INFINITY {
            continue;
        }
        if !value.is_finite() {
            return Err(VokraError::ModelLoad(
                "vibevoice constrained logits contain non-finite value".to_owned(),
            ));
        }
        if best.is_none_or(|(_, current)| value > current) {
            best = Some((token, value));
        }
    }
    best.map(|(token, _)| token)
        .ok_or_else(|| VokraError::ModelLoad("vibevoice constrained token set is empty".to_owned()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TokenPlan {
    /// Conditions are read from the current hidden before cache advancement.
    diffusion: bool,
    /// The unconditional branch is refreshed from one speech-start token.
    refresh_negative: bool,
    /// Codec state is cleared, but autoregressive generation continues.
    clear_codec: bool,
    /// EOS terminates; speech_end does not.
    terminal: bool,
    /// Number of normal branch cache advances for this token.
    cache_advances: u8,
    negative_action: NegativeAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NegativeAction {
    Hold,
    RefreshSpeechStart,
    SpeechEmbedding,
}

fn token_plan(token: u32) -> TokenPlan {
    if token == BOS_EOS_TOKEN_ID {
        TokenPlan {
            diffusion: false,
            refresh_negative: false,
            clear_codec: true,
            terminal: true,
            cache_advances: 0,
            negative_action: NegativeAction::Hold,
        }
    } else if token == super::SPEECH_DIFFUSION_TOKEN_ID {
        TokenPlan {
            diffusion: true,
            refresh_negative: false,
            clear_codec: false,
            terminal: false,
            cache_advances: 1,
            negative_action: NegativeAction::SpeechEmbedding,
        }
    } else if token == super::SPEECH_START_TOKEN_ID {
        TokenPlan {
            diffusion: false,
            refresh_negative: true,
            clear_codec: false,
            terminal: false,
            cache_advances: 1,
            negative_action: NegativeAction::RefreshSpeechStart,
        }
    } else if token == super::SPEECH_END_TOKEN_ID {
        TokenPlan {
            diffusion: false,
            refresh_negative: false,
            clear_codec: true,
            terminal: false,
            cache_advances: 1,
            negative_action: NegativeAction::Hold,
        }
    } else {
        TokenPlan {
            diffusion: false,
            refresh_negative: false,
            clear_codec: false,
            terminal: false,
            cache_advances: 1,
            negative_action: NegativeAction::Hold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_sampling_uses_exact_single_latent_draw_contract() {
        let mean = 2.0_f32;
        let latent_draw = -4.0_f32;
        assert_eq!(mean + PROMPT_STD * latent_draw, -0.5);
    }

    #[test]
    fn constrained_logits_are_deterministic_and_marker_limited() {
        let mut logits = vec![f32::NEG_INFINITY; 151_936];
        logits[BOS_EOS_TOKEN_ID as usize] = 1.0;
        logits[super::super::SPEECH_DIFFUSION_TOKEN_ID as usize] = 2.0;
        logits[42] = 100.0;
        assert_eq!(
            constrained_greedy_token(&logits).unwrap(),
            super::super::SPEECH_DIFFUSION_TOKEN_ID
        );
        assert_eq!(
            constrained_greedy_token(&logits).unwrap(),
            super::super::SPEECH_DIFFUSION_TOKEN_ID
        );
    }

    #[test]
    fn constrained_sequence_tracks_start_then_diffusion_then_end() {
        let mut logits = vec![f32::NEG_INFINITY; 151_936];
        logits[super::super::SPEECH_START_TOKEN_ID as usize] = 3.0;
        logits[super::super::SPEECH_DIFFUSION_TOKEN_ID as usize] = 2.0;
        assert_eq!(
            constrained_greedy_token(&logits).unwrap(),
            super::super::SPEECH_START_TOKEN_ID
        );
        logits[super::super::SPEECH_START_TOKEN_ID as usize] = f32::NEG_INFINITY;
        logits[super::super::SPEECH_DIFFUSION_TOKEN_ID as usize] = 4.0;
        assert_eq!(
            constrained_greedy_token(&logits).unwrap(),
            super::super::SPEECH_DIFFUSION_TOKEN_ID
        );
        logits[super::super::SPEECH_DIFFUSION_TOKEN_ID as usize] = f32::NEG_INFINITY;
        logits[super::super::SPEECH_END_TOKEN_ID as usize] = 5.0;
        assert_eq!(
            constrained_greedy_token(&logits).unwrap(),
            super::super::SPEECH_END_TOKEN_ID
        );
    }

    #[test]
    fn token_plan_proves_condition_before_one_cache_advance_and_negative_refresh() {
        let start = token_plan(super::super::SPEECH_START_TOKEN_ID);
        assert!(start.refresh_negative);
        assert_eq!(start.cache_advances, 1);
        assert!(!start.terminal);

        let diffusion = token_plan(super::super::SPEECH_DIFFUSION_TOKEN_ID);
        assert!(diffusion.diffusion);
        assert_eq!(diffusion.cache_advances, 1);

        let end = token_plan(super::super::SPEECH_END_TOKEN_ID);
        assert!(end.clear_codec);
        assert!(!end.terminal);
        assert_eq!(end.cache_advances, 1);
        assert_eq!(end.negative_action, NegativeAction::Hold);

        let eos = token_plan(BOS_EOS_TOKEN_ID);
        assert!(eos.terminal);
        assert_eq!(eos.cache_advances, 0);
    }

    #[test]
    fn packet_rejects_hidden_rng_or_malformed_draws() {
        let packet = VibeVoiceGenerationPacket {
            token_ids: vec![1],
            speech_replacement_positions: Vec::new(),
            prompt_pcm: None,
            prompt_sample_rate_hz: 0,
            prompt_latent_draws: Vec::new(),
            diffusion_initial_draws: vec![vec![0.0; LATENT_WIDTH - 1]],
            guidance_scale: 1.0,
            max_generated_tokens: 1,
        };
        assert!(validate_packet(&packet).is_err());
    }

    #[test]
    fn packet_requires_explicit_prompt_sample_rate_and_finite_draws() {
        let mut packet = VibeVoiceGenerationPacket {
            token_ids: vec![1],
            speech_replacement_positions: vec![0],
            prompt_pcm: Some(vec![0.0; 3_200]),
            prompt_sample_rate_hz: 16_000,
            prompt_latent_draws: vec![0.0; LATENT_WIDTH],
            diffusion_initial_draws: Vec::new(),
            guidance_scale: 1.0,
            max_generated_tokens: 1,
        };
        assert!(validate_packet(&packet).is_err());
        packet.prompt_sample_rate_hz = SAMPLE_RATE;
        packet.prompt_latent_draws[0] = f32::NAN;
        assert!(validate_packet(&packet).is_err());
    }
}
