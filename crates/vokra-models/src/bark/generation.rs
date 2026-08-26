//! Official Bark semantic → coarse → fine token schedule.
//!
//! This layer accepts already-tokenized Bark text IDs. Tokenization is kept an
//! explicit sidecar boundary because the two public GGUFs contain model
//! tensors only; inventing a tokenizer vocabulary at runtime would silently
//! change text conditioning. With no history prompt, the schedule matches the
//! official default path: 256 text/history slots, alternating two-codebook
//! coarse windows, then six non-causal fine-codebook passes.

use vokra_core::{Result, Sampler, SamplerConfig, VokraError};

use crate::compute::Compute;

use super::transformer::{
    CausalStage, causal_embedding, causal_prefill, causal_token_step, fine_logits,
};
use super::{BarkModel, CODEBOOK_SIZE, CODEBOOKS_USED};

const LABEL: &str = "bark";

const TEXT_ENCODING_OFFSET: u32 = 10_048;
const TEXT_PAD_TOKEN: u32 = 129_595;
const SEMANTIC_INFER_TOKEN: u32 = 129_599;
const SEMANTIC_PAD_TOKEN: u32 = 10_000;
const SEMANTIC_OUTPUT_VOCAB: usize = 10_048;
const SEMANTIC_ALLOWED: usize = 10_001;
const MAX_TEXT_TOKENS: usize = 256;
const SEMANTIC_RATE_HZ: f64 = 49.9;

const COARSE_SEMANTIC_PAD_TOKEN: u32 = 12_048;
const COARSE_INFER_TOKEN: u32 = 12_050;
const COARSE_RATE_HZ: f64 = 75.0;
const COARSE_CODEBOOKS: usize = 2;
const MAX_COARSE_INPUT: usize = 256;
const MAX_COARSE_HISTORY: usize = 630;
const COARSE_WINDOW: usize = 60;

const MAX_FINE_HISTORY: usize = 512;
const MAX_FINE_INPUT: usize = 1_024;
const FINE_INPUT_VOCAB: u32 = 1_056;

/// Host-side generation controls for the official no-history Bark route.
#[derive(Debug, Clone, PartialEq)]
pub struct BarkGenerationConfig {
    /// Maximum generated semantic tokens (official default: 768).
    pub max_semantic_tokens: usize,
    /// Semantic sampler temperature (official default: 0.7; zero = greedy).
    pub semantic_temperature: f32,
    /// Semantic top-k cutoff (release config: 50).
    pub semantic_top_k: Option<usize>,
    /// Coarse sampler temperature (official default: 0.7; zero = greedy).
    pub coarse_temperature: f32,
    /// Coarse per-codebook top-k cutoff (release config: 50).
    pub coarse_top_k: Option<usize>,
    /// Fine sampler temperature (official default: 0.5; zero = greedy).
    pub fine_temperature: f32,
    /// Deterministic host RNG seed.
    pub seed: u64,
}

impl Default for BarkGenerationConfig {
    fn default() -> Self {
        Self {
            max_semantic_tokens: 768,
            semantic_temperature: 0.7,
            semantic_top_k: Some(50),
            coarse_temperature: 0.7,
            coarse_top_k: Some(50),
            fine_temperature: 0.5,
            seed: 0,
        }
    }
}

impl BarkGenerationConfig {
    /// Deterministic greedy generation, useful for parity and regression runs.
    #[must_use]
    pub const fn greedy(max_semantic_tokens: usize) -> Self {
        Self {
            max_semantic_tokens,
            semantic_temperature: 0.0,
            semantic_top_k: None,
            coarse_temperature: 0.0,
            coarse_top_k: None,
            fine_temperature: 0.0,
            seed: 0,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.max_semantic_tokens == 0 || self.max_semantic_tokens > 768 {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: max_semantic_tokens {} must be in 1..=768",
                self.max_semantic_tokens
            )));
        }
        for (name, value) in [
            ("semantic_temperature", self.semantic_temperature),
            ("coarse_temperature", self.coarse_temperature),
            ("fine_temperature", self.fine_temperature),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(VokraError::InvalidArgument(format!(
                    "{LABEL}: {name} must be finite and >= 0, got {value}"
                )));
            }
        }
        for (name, value) in [
            ("semantic_top_k", self.semantic_top_k),
            ("coarse_top_k", self.coarse_top_k),
        ] {
            if value.is_some_and(|top_k| top_k == 0 || top_k > CODEBOOK_SIZE) {
                return Err(VokraError::InvalidArgument(format!(
                    "{LABEL}: {name} must be in 1..={CODEBOOK_SIZE} when present, got {value:?}"
                )));
            }
        }
        Ok(())
    }
}

/// Eight-codebook EnCodec indices in frame-major order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BarkGeneratedCodes {
    codes: Vec<u32>,
    frames: usize,
}

impl BarkGeneratedCodes {
    /// Validates a caller-supplied frame-major eight-codebook packet.
    ///
    /// This is the explicit boundary for decoding previously generated Bark
    /// codes or independent parity fixtures; no codebook-major transpose,
    /// clamping, or missing-channel fill is inferred.
    pub fn from_frame_major(codes: Vec<u32>, frames: usize) -> Result<Self> {
        if frames == 0 {
            return Err(VokraError::InvalidArgument(
                "bark: generated code frames must be > 0".to_owned(),
            ));
        }
        let expected = frames.checked_mul(CODEBOOKS_USED).ok_or_else(|| {
            VokraError::InvalidArgument("bark: generated code shape overflows usize".to_owned())
        })?;
        if codes.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "bark: frame-major codes have {} values, expected {frames}x{CODEBOOKS_USED}={expected}",
                codes.len()
            )));
        }
        if let Some((index, code)) = codes
            .iter()
            .copied()
            .enumerate()
            .find(|(_, code)| *code >= CODEBOOK_SIZE as u32)
        {
            return Err(VokraError::InvalidArgument(format!(
                "bark: frame-major code[{index}]={code} is outside 0..{CODEBOOK_SIZE}"
            )));
        }
        Ok(Self { codes, frames })
    }

    /// Number of 75 Hz codec frames.
    pub const fn frames(&self) -> usize {
        self.frames
    }

    /// Frame-major `[frames, 8]` codebook indices in `0..1024`.
    pub fn as_frame_major(&self) -> &[u32] {
        &self.codes
    }

    /// Consumes the packet and returns frame-major indices.
    pub fn into_frame_major(self) -> Vec<u32> {
        self.codes
    }

    /// One eight-codebook frame.
    pub fn frame(&self, frame: usize) -> Result<&[u32]> {
        if frame >= self.frames {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: frame {frame} >= {}",
                self.frames
            )));
        }
        let start = frame * CODEBOOKS_USED;
        Ok(&self.codes[start..start + CODEBOOKS_USED])
    }
}

impl BarkModel {
    /// Generates all eight EnCodec codebooks from pre-tokenized Bark text.
    ///
    /// The public GGUF does not embed the Bark tokenizer. `text_token_ids`
    /// must therefore come from the pinned `suno/bark` processor vocabulary;
    /// `attention_mask`, when supplied, has one entry per ID. This method does
    /// not substitute byte/character tokenization.
    pub fn generate_codes_from_tokens(
        &self,
        text_token_ids: &[u32],
        attention_mask: Option<&[bool]>,
        generation: &BarkGenerationConfig,
    ) -> Result<BarkGeneratedCodes> {
        generation.validate()?;
        validate_text_tokens(text_token_ids, attention_mask)?;
        let weights = self.mapped()?;
        let compute = Compute::for_backend(self.backend, super::BARK_HOT_OPS)?;
        let semantic = generate_semantic(
            weights,
            &compute,
            &self.config,
            text_token_ids,
            attention_mask,
            generation,
        )?;
        let coarse = generate_coarse(weights, &compute, &self.config, &semantic, generation)?;
        generate_fine(weights, &compute, &self.config, &coarse, generation)
    }
}

fn validate_text_tokens(ids: &[u32], mask: Option<&[bool]>) -> Result<()> {
    if ids.is_empty() || ids.len() > MAX_TEXT_TOKENS {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: text_token_ids length {} must be in 1..={MAX_TEXT_TOKENS}",
            ids.len()
        )));
    }
    if mask.is_some_and(|values| values.len() != ids.len()) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: attention_mask length {:?} != text_token_ids length {}",
            mask.map(<[bool]>::len),
            ids.len()
        )));
    }
    let max_raw = 129_600u32 - TEXT_ENCODING_OFFSET;
    if let Some((index, token)) = ids
        .iter()
        .copied()
        .enumerate()
        .find(|(_, token)| *token >= max_raw)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: text_token_ids[{index}]={token} plus offset {TEXT_ENCODING_OFFSET} exceeds semantic input vocab"
        )));
    }
    Ok(())
}

fn generate_semantic(
    weights: &super::weights::BarkMappedWeights,
    compute: &Compute,
    config: &super::BarkConfig,
    text_ids: &[u32],
    attention_mask: Option<&[bool]>,
    generation: &BarkGenerationConfig,
) -> Result<Vec<u32>> {
    let hidden = config.hidden_size;
    let mut embeddings = Vec::with_capacity((MAX_TEXT_TOKENS + 1) * hidden);
    for position in 0..MAX_TEXT_TOKENS {
        let active =
            position < text_ids.len() && attention_mask.map_or(true, |mask| mask[position]);
        let text_token = if active {
            text_ids[position] + TEXT_ENCODING_OFFSET
        } else {
            TEXT_PAD_TOKEN
        };
        let mut row = causal_embedding(weights, config, CausalStage::Semantic, text_token)?;
        let history = causal_embedding(weights, config, CausalStage::Semantic, SEMANTIC_PAD_TOKEN)?;
        add_assign(&mut row, &history)?;
        embeddings.extend_from_slice(&row);
    }
    embeddings.extend_from_slice(&causal_embedding(
        weights,
        config,
        CausalStage::Semantic,
        SEMANTIC_INFER_TOKEN,
    )?);

    let (mut cache, mut logits) =
        causal_prefill(weights, compute, config, CausalStage::Semantic, &embeddings)?;
    debug_assert_eq!(logits.len(), SEMANTIC_OUTPUT_VOCAB);
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: generation.semantic_temperature,
        top_k: generation.semantic_top_k,
        top_p: None,
        repetition_penalty: None,
        seed: generation.seed,
    });
    let mut output = Vec::new();
    for step in 0..generation.max_semantic_tokens {
        let mut allowed = logits[..SEMANTIC_ALLOWED].to_vec();
        let token = sampler.sample(&mut allowed);
        output.push(token);
        if token == SEMANTIC_PAD_TOKEN {
            break;
        }
        if step + 1 < generation.max_semantic_tokens {
            logits = causal_token_step(weights, compute, config, &mut cache, token)?;
        }
    }
    Ok(output)
}

fn generate_coarse(
    weights: &super::weights::BarkMappedWeights,
    compute: &Compute,
    config: &super::BarkConfig,
    semantic: &[u32],
    generation: &BarkGenerationConfig,
) -> Result<Vec<u32>> {
    if semantic.is_empty() {
        return Err(VokraError::InvalidArgument(
            "bark/coarse: semantic output is empty".to_owned(),
        ));
    }
    let semantic: Vec<u32> = semantic
        .iter()
        .map(|&token| {
            if token == SEMANTIC_PAD_TOKEN {
                COARSE_SEMANTIC_PAD_TOKEN
            } else {
                token
            }
        })
        .collect();
    let semantic_to_coarse = COARSE_RATE_HZ / SEMANTIC_RATE_HZ * COARSE_CODEBOOKS as f64;
    let max_semantic_history = (MAX_COARSE_HISTORY as f64 / semantic_to_coarse).floor() as usize;
    let frames =
        (semantic.len() as f64 * semantic_to_coarse / COARSE_CODEBOOKS as f64).floor() as usize;
    let max_generated = frames * COARSE_CODEBOOKS;
    if max_generated == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "bark/coarse: {} semantic token(s) produce zero codec frames",
            semantic.len()
        )));
    }

    let mut sampler = Sampler::new(SamplerConfig {
        temperature: generation.coarse_temperature,
        top_k: generation.coarse_top_k,
        top_p: None,
        repetition_penalty: None,
        seed: generation.seed ^ 0xC0A4_5E00_D15C_A11E,
    });
    let mut coarse = Vec::with_capacity(max_generated);
    while coarse.len() < max_generated {
        let semantic_idx = (coarse.len() as f64 / semantic_to_coarse).round() as usize;
        let start = semantic_idx.saturating_sub(max_semantic_history);
        let mut prompt: Vec<u32> = semantic[start..]
            .iter()
            .copied()
            .take(MAX_COARSE_INPUT)
            .collect();
        prompt.resize(MAX_COARSE_INPUT, COARSE_SEMANTIC_PAD_TOKEN);
        prompt.push(COARSE_INFER_TOKEN);
        let history_start = coarse.len().saturating_sub(MAX_COARSE_HISTORY);
        prompt.extend_from_slice(&coarse[history_start..]);

        let mut embeddings = Vec::with_capacity(prompt.len() * config.hidden_size);
        for &token in &prompt {
            embeddings.extend_from_slice(&causal_embedding(
                weights,
                config,
                CausalStage::Coarse,
                token,
            )?);
        }
        let (mut cache, mut logits) =
            causal_prefill(weights, compute, config, CausalStage::Coarse, &embeddings)?;
        let window = COARSE_WINDOW.min(max_generated - coarse.len());
        for local in 0..window {
            let codebook = local % COARSE_CODEBOOKS;
            let offset = SEMANTIC_PAD_TOKEN as usize + codebook * CODEBOOK_SIZE;
            let mut allowed = logits[offset..offset + CODEBOOK_SIZE].to_vec();
            let token = offset as u32 + sampler.sample(&mut allowed);
            coarse.push(token);
            if local + 1 < window {
                logits = causal_token_step(weights, compute, config, &mut cache, token)?;
            }
        }
    }
    Ok(coarse)
}

fn generate_fine(
    weights: &super::weights::BarkMappedWeights,
    compute: &Compute,
    config: &super::BarkConfig,
    coarse: &[u32],
    generation: &BarkGenerationConfig,
) -> Result<BarkGeneratedCodes> {
    if coarse.is_empty() || !coarse.len().is_multiple_of(COARSE_CODEBOOKS) {
        return Err(VokraError::InvalidArgument(format!(
            "bark/fine: coarse length {} must be a non-zero multiple of {COARSE_CODEBOOKS}",
            coarse.len()
        )));
    }
    let frames = coarse.len() / COARSE_CODEBOOKS;
    let mut fine = vec![CODEBOOK_SIZE as u32; frames * CODEBOOKS_USED];
    for frame in 0..frames {
        for codebook in 0..COARSE_CODEBOOKS {
            let encoded = coarse[frame * COARSE_CODEBOOKS + codebook];
            let expected_start = SEMANTIC_PAD_TOKEN + (codebook * CODEBOOK_SIZE) as u32;
            let expected_end = expected_start + CODEBOOK_SIZE as u32;
            if !(expected_start..expected_end).contains(&encoded) {
                return Err(VokraError::InvalidArgument(format!(
                    "bark/fine: coarse[{frame},{codebook}]={encoded} is outside alternating codebook range {expected_start}..{expected_end}"
                )));
            }
            fine[frame * CODEBOOKS_USED + codebook] =
                (encoded - SEMANTIC_PAD_TOKEN) % CODEBOOK_SIZE as u32;
        }
    }

    let padded_frames = frames.max(MAX_FINE_INPUT);
    fine.resize(padded_frames * CODEBOOKS_USED, CODEBOOK_SIZE as u32);
    let extra_loops = frames
        .saturating_sub(MAX_FINE_INPUT)
        .div_ceil(MAX_FINE_HISTORY);
    let loops = extra_loops + 1;
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: generation.fine_temperature,
        top_k: None,
        top_p: None,
        repetition_penalty: None,
        seed: generation.seed ^ 0xF1AE_C0DE_5EED_0008,
    });

    for outer in 0..loops {
        let start = (outer * MAX_FINE_HISTORY).min(padded_frames - MAX_FINE_INPUT);
        let fill_start = (outer * MAX_FINE_HISTORY).min(padded_frames - MAX_FINE_HISTORY);
        let relative_fill = fill_start - start;
        let mut window =
            fine[start * CODEBOOKS_USED..(start + MAX_FINE_INPUT) * CODEBOOKS_USED].to_vec();
        for codebook in COARSE_CODEBOOKS..CODEBOOKS_USED {
            let logits = fine_logits(weights, compute, config, &window, MAX_FINE_INPUT, codebook)?;
            for row in relative_fill..MAX_FINE_INPUT {
                let mut allowed = logits[row * FINE_INPUT_VOCAB as usize
                    ..row * FINE_INPUT_VOCAB as usize + CODEBOOK_SIZE]
                    .to_vec();
                window[row * CODEBOOKS_USED + codebook] = sampler.sample(&mut allowed);
            }
        }
        let written = MAX_FINE_INPUT - relative_fill;
        for row in 0..written {
            for codebook in COARSE_CODEBOOKS..CODEBOOKS_USED {
                fine[(fill_start + row) * CODEBOOKS_USED + codebook] =
                    window[(relative_fill + row) * CODEBOOKS_USED + codebook];
            }
        }
    }
    fine.truncate(frames * CODEBOOKS_USED);
    BarkGeneratedCodes::from_frame_major(fine, frames)
}

fn add_assign(left: &mut [f32], right: &[f32]) -> Result<()> {
    if left.len() != right.len() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: embedding sum length mismatch {} != {}",
            left.len(),
            right.len()
        )));
    }
    for (target, &value) in left.iter_mut().zip(right) {
        *target += value;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_pin_official_generation_axes() {
        let config = BarkGenerationConfig::default();
        assert_eq!(config.max_semantic_tokens, 768);
        assert_eq!(config.semantic_temperature, 0.7);
        assert_eq!(config.coarse_temperature, 0.7);
        assert_eq!(config.fine_temperature, 0.5);
        assert_eq!(config.semantic_top_k, Some(50));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn text_boundary_rejects_implicit_truncation_and_bad_offsets() {
        assert!(validate_text_tokens(&[], None).is_err());
        assert!(validate_text_tokens(&vec![0; 257], None).is_err());
        assert!(validate_text_tokens(&[119_552], None).is_err());
        assert!(validate_text_tokens(&[1, 2], Some(&[true])).is_err());
    }

    #[test]
    fn generated_codes_expose_frame_major_rows() {
        let packet = BarkGeneratedCodes::from_frame_major((0..16).collect(), 2).unwrap();
        assert_eq!(packet.frame(1).unwrap(), &[8, 9, 10, 11, 12, 13, 14, 15]);
        assert!(packet.frame(2).is_err());
        assert!(BarkGeneratedCodes::from_frame_major(vec![0; 7], 1).is_err());
        assert!(BarkGeneratedCodes::from_frame_major(vec![0; 8], 0).is_err());
        assert!(
            BarkGeneratedCodes::from_frame_major(vec![CODEBOOK_SIZE as u32; CODEBOOKS_USED], 1)
                .is_err()
        );
    }
}
