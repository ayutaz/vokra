//! Official Parler delayed-code generation over the authenticated shared LM.
//!
//! The nine codebooks use distinct BOS (`1025`) and PAD/EOS (`1024`) tokens.
//! Codebook `q` starts predicting at delayed position `q + 1`; output frame
//! `f` is recovered from position `1 + q + f`. Sampling is deterministic for
//! a fixed seed and consumes one draw per codebook per delayed position before
//! the structural delay mask is applied, matching the Transformers loop.

use vokra_core::{Result, Sampler, SamplerConfig, VokraError};

use super::{
    BOS_TOKEN_ID, CODEBOOK_SIZE, DECODER_VOCAB_SIZE, NUM_CODEBOOKS, PAD_EOS_TOKEN_ID, ParlerModel,
    ParlerVariant,
};

/// Host-side generation controls for Parler's delayed nine-codebook stream.
#[derive(Debug, Clone, PartialEq)]
pub struct ParlerGenerationConfig {
    /// Maximum decoded DAC frames before early EOS filtering.
    pub max_frames: usize,
    /// Softmax temperature; zero selects exact greedy argmax.
    pub temperature: f32,
    /// Top-k cutoff when `top_p` is absent. Official default is 50.
    pub top_k: Option<usize>,
    /// Optional nucleus threshold in `(0, 1]`; takes precedence over top-k.
    pub top_p: Option<f32>,
    /// Number of newly generated delayed positions that suppress EOS.
    pub min_new_tokens: usize,
    /// Deterministic sampler seed.
    pub seed: u64,
}

impl ParlerGenerationConfig {
    /// Official release sampling controls and release-specific length cap.
    #[must_use]
    pub const fn official(variant: ParlerVariant, seed: u64) -> Self {
        Self {
            // generation_config.max_length minus the nine delay rows.
            max_frames: match variant {
                ParlerVariant::MiniV1English => 2_571,
                ParlerVariant::MiniMultilingualV11 => 2_601,
            },
            temperature: 1.0,
            top_k: Some(50),
            top_p: None,
            min_new_tokens: match variant {
                ParlerVariant::MiniV1English => 10,
                ParlerVariant::MiniMultilingualV11 => 0,
            },
            seed,
        }
    }

    /// Deterministic greedy generation for parity and diagnostics.
    #[must_use]
    pub const fn greedy(max_frames: usize) -> Self {
        Self {
            max_frames,
            temperature: 0.0,
            top_k: None,
            top_p: None,
            min_new_tokens: 0,
            seed: 0,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.max_frames == 0 {
            return Err(VokraError::InvalidArgument(
                "parler generation max_frames must be non-zero".to_owned(),
            ));
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "parler generation temperature must be finite and >= 0, got {}",
                self.temperature
            )));
        }
        if self.top_p.is_none() {
            if let Some(top_k) = self.top_k {
                if top_k == 0 || top_k > DECODER_VOCAB_SIZE {
                    return Err(VokraError::InvalidArgument(format!(
                        "parler generation top_k {top_k} must be in 1..={DECODER_VOCAB_SIZE}"
                    )));
                }
            }
        }
        if let Some(top_p) = self.top_p {
            if !top_p.is_finite() || !(0.0 < top_p && top_p <= 1.0) {
                return Err(VokraError::InvalidArgument(format!(
                    "parler generation top_p must be finite and in (0, 1], got {top_p}"
                )));
            }
        }
        Ok(())
    }

    fn sampler_config(&self) -> SamplerConfig {
        SamplerConfig {
            temperature: self.temperature,
            top_k: if self.top_p.is_some() {
                None
            } else {
                self.top_k
            },
            top_p: self.top_p,
            repetition_penalty: None,
            seed: self.seed,
        }
    }
}

/// Valid DAC indices in frame-major `[frames, 9]` order.
///
/// Any frame containing EOS/PAD/reserved rows in at least one codebook is
/// omitted, matching Parler's sequential decode path before DAC invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParlerGeneratedCodes {
    codes: Vec<u32>,
    frames: usize,
}

impl ParlerGeneratedCodes {
    #[must_use]
    pub const fn frames(&self) -> usize {
        self.frames
    }

    #[must_use]
    pub const fn num_codebooks(&self) -> usize {
        NUM_CODEBOOKS
    }

    /// Frame-major `[frames, 9]` indices, all strictly below 1024.
    #[must_use]
    pub fn as_frame_major(&self) -> &[u32] {
        &self.codes
    }

    #[must_use]
    pub fn into_frame_major(self) -> Vec<u32> {
        self.codes
    }

    pub fn frame(&self, frame: usize) -> Result<&[u32]> {
        if frame >= self.frames {
            return Err(VokraError::InvalidArgument(format!(
                "parler generated frame {frame} >= {}",
                self.frames
            )));
        }
        let start = frame * NUM_CODEBOOKS;
        Ok(&self.codes[start..start + NUM_CODEBOOKS])
    }
}

impl ParlerModel {
    /// Generates authenticated DAC codes from explicit description and prompt
    /// token IDs.
    ///
    /// The public GGUFs contain neither tokenizer, so raw strings are never
    /// guessed or passed through a mismatched vocabulary. `description_mask`
    /// applies to FLAN-T5 and the decoder's cross-attention condition. Prompt
    /// IDs use the distinct learned `embed_prompts` table and are causally
    /// prefilled into decoder self-attention.
    pub fn generate_codes(
        &self,
        description_token_ids: &[u32],
        description_mask: Option<&[bool]>,
        prompt_token_ids: &[u32],
        generation: &ParlerGenerationConfig,
    ) -> Result<ParlerGeneratedCodes> {
        generation.validate()?;
        let description = self.encode_description(description_token_ids, description_mask)?;
        let mask_storage = description_mask.map(|mask| {
            mask.iter()
                .map(|&visible| u8::from(visible))
                .collect::<Vec<_>>()
        });
        let condition = self.decoder().prepare_condition(
            &description,
            description_token_ids.len(),
            mask_storage.as_deref(),
        )?;
        let prompt = self.embed_prompt_tokens(prompt_token_ids)?;

        let max_length = generation
            .max_frames
            .checked_add(NUM_CODEBOOKS)
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "parler generation max_frames + codebooks overflows usize".to_owned(),
                )
            })?;
        let decode_steps = max_length - 1;
        let mut state =
            self.decoder()
                .new_state_with_prefix_embeddings(&condition, &prompt, decode_steps)?;
        let sequence_len = NUM_CODEBOOKS.checked_mul(max_length).ok_or_else(|| {
            VokraError::InvalidArgument(
                "parler generation delayed sequence shape overflows usize".to_owned(),
            )
        })?;
        let logits_len = NUM_CODEBOOKS
            .checked_mul(DECODER_VOCAB_SIZE)
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "parler generation logits shape overflows usize".to_owned(),
                )
            })?;
        let mut sequence = vec![PAD_EOS_TOKEN_ID; sequence_len];
        let mut current = [BOS_TOKEN_ID; NUM_CODEBOOKS];
        for codebook in 0..NUM_CODEBOOKS {
            sequence[codebook * max_length] = BOS_TOKEN_ID;
        }
        let mut logits = vec![0.0; logits_len];
        let mut finished = [false; NUM_CODEBOOKS];
        let mut sampler = Sampler::new(generation.sampler_config());

        for target in 1..max_length {
            self.decoder()
                .step_into(&mut state, &current, &mut logits)?;
            if logits.iter().any(|value| !value.is_finite()) {
                return Err(VokraError::InvalidArgument(format!(
                    "parler generation produced a non-finite logit at delayed position {target}"
                )));
            }
            if target <= generation.min_new_tokens {
                for codebook in 0..NUM_CODEBOOKS {
                    logits[codebook * DECODER_VOCAB_SIZE + PAD_EOS_TOKEN_ID as usize] =
                        f32::NEG_INFINITY;
                }
            }

            for (codebook, current_token) in current.iter_mut().enumerate() {
                let row =
                    &mut logits[codebook * DECODER_VOCAB_SIZE..(codebook + 1) * DECODER_VOCAB_SIZE];
                // Transformers samples every flattened codebook row first;
                // the structural delay mask then replaces non-predictive rows.
                let sampled = sampler.sample(row);
                let token = match delay_slot(codebook, target, max_length) {
                    DelaySlot::Bos => BOS_TOKEN_ID,
                    DelaySlot::Pad => PAD_EOS_TOKEN_ID,
                    DelaySlot::Predict if finished[codebook] => PAD_EOS_TOKEN_ID,
                    DelaySlot::Predict => {
                        if sampled == PAD_EOS_TOKEN_ID {
                            finished[codebook] = true;
                        }
                        sampled
                    }
                };
                sequence[codebook * max_length + target] = token;
                *current_token = token;
            }
        }

        extract_valid_frames(&sequence, max_length, generation.max_frames)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DelaySlot {
    Bos,
    Predict,
    Pad,
}

fn delay_slot(codebook: usize, position: usize, max_length: usize) -> DelaySlot {
    if position <= codebook {
        DelaySlot::Bos
    } else if position >= max_length - NUM_CODEBOOKS + 1 + codebook {
        DelaySlot::Pad
    } else {
        DelaySlot::Predict
    }
}

fn extract_valid_frames(
    sequence: &[u32],
    max_length: usize,
    requested_frames: usize,
) -> Result<ParlerGeneratedCodes> {
    if sequence.len() != NUM_CODEBOOKS.saturating_mul(max_length) {
        return Err(VokraError::InvalidArgument(format!(
            "parler delayed sequence len {} != {NUM_CODEBOOKS} * {max_length}",
            sequence.len()
        )));
    }
    let mut codes = Vec::with_capacity(requested_frames.saturating_mul(NUM_CODEBOOKS));
    for frame in 0..requested_frames {
        let mut row = [0u32; NUM_CODEBOOKS];
        let mut valid = true;
        for (codebook, token) in row.iter_mut().enumerate() {
            let position = 1 + codebook + frame;
            let value = sequence[codebook * max_length + position];
            *token = value;
            valid &= (value as usize) < CODEBOOK_SIZE;
        }
        if valid {
            codes.extend_from_slice(&row);
        }
    }
    Ok(ParlerGeneratedCodes {
        frames: codes.len() / NUM_CODEBOOKS,
        codes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_schedule_matches_official_triangles() {
        let max_length = 14;
        assert_eq!(delay_slot(0, 0, max_length), DelaySlot::Bos);
        assert_eq!(delay_slot(0, 1, max_length), DelaySlot::Predict);
        assert_eq!(delay_slot(0, 5, max_length), DelaySlot::Predict);
        assert_eq!(delay_slot(0, 6, max_length), DelaySlot::Pad);

        assert_eq!(delay_slot(8, 8, max_length), DelaySlot::Bos);
        assert_eq!(delay_slot(8, 9, max_length), DelaySlot::Predict);
        assert_eq!(delay_slot(8, 13, max_length), DelaySlot::Predict);
    }

    #[test]
    fn extraction_transposes_delay_positions_and_drops_special_frames() {
        let frames = 3;
        let max_length = frames + NUM_CODEBOOKS;
        let mut sequence = vec![PAD_EOS_TOKEN_ID; NUM_CODEBOOKS * max_length];
        for frame in 0..frames {
            for codebook in 0..NUM_CODEBOOKS {
                sequence[codebook * max_length + 1 + codebook + frame] =
                    (frame * NUM_CODEBOOKS + codebook) as u32;
            }
        }
        sequence[4 * max_length + 1 + 4 + 1] = PAD_EOS_TOKEN_ID;
        let output = extract_valid_frames(&sequence, max_length, frames).unwrap();
        assert_eq!(output.frames(), 2);
        assert_eq!(output.frame(0).unwrap(), &[0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            output.frame(1).unwrap(),
            &[18, 19, 20, 21, 22, 23, 24, 25, 26]
        );
    }

    #[test]
    fn generation_controls_fail_closed() {
        let mut config = ParlerGenerationConfig::greedy(1);
        assert!(config.validate().is_ok());
        config.max_frames = 0;
        assert!(config.validate().is_err());
        config.max_frames = 1;
        config.temperature = f32::NAN;
        assert!(config.validate().is_err());
        config.temperature = 1.0;
        config.top_k = Some(DECODER_VOCAB_SIZE + 1);
        assert!(config.validate().is_err());
        config.top_p = Some(0.0);
        assert!(config.validate().is_err());
    }
}
