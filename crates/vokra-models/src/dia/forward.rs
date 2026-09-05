//! Source-shaped, batch-one Dia forward primitives.
//!
//! This module is deliberately crate-private.  Its real-weight route is
//! available only after the authenticated 343-tensor bind and the complete
//! six-file/VAST contract; the small fixture constructor exists solely for
//! deterministic shape tests.

use std::cmp::Ordering;

use vokra_core::backend::BackendKind;
use vokra_core::{Result, VokraError};

use crate::compute::{Compute, HotOp};

use super::{DiaConfig, DiaEncoderBlockWeights, DiaWeights};

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
const HOT_OPS: &[HotOp] = &[HotOp::Gemm, HotOp::RmsNorm, HotOp::Softmax];

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
#[derive(Debug, Clone, Default)]
struct KvHistory {
    /// Head-major `[heads, time, head_dim]` keys.
    keys: Vec<f32>,
    /// Head-major `[heads, time, head_dim]` values.
    values: Vec<f32>,
    heads: usize,
    head_dim: usize,
    len: usize,
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
impl KvHistory {
    fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.len = 0;
    }

    fn append(
        &mut self,
        keys: &[f32],
        values: &[f32],
        heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        let row = heads
            .checked_mul(head_dim)
            .ok_or_else(|| VokraError::InvalidArgument("dia KV shape overflow".to_owned()))?;
        if keys.len() != row || values.len() != row || row == 0 {
            return Err(VokraError::InvalidArgument(
                "dia KV append shape mismatch".to_owned(),
            ));
        }
        if self.len == 0 {
            self.heads = heads;
            self.head_dim = head_dim;
        } else if self.heads != heads || self.head_dim != head_dim {
            return Err(VokraError::InvalidArgument(
                "dia KV append head shape changed".to_owned(),
            ));
        }
        self.keys.extend_from_slice(keys);
        self.values.extend_from_slice(values);
        self.len += 1;
        Ok(())
    }

    fn truncate(&mut self, len: usize) {
        let width = self.heads * self.head_dim;
        self.keys.truncate(len.saturating_mul(width));
        self.values.truncate(len.saturating_mul(width));
        self.len = len;
        if len == 0 {
            self.heads = 0;
            self.head_dim = 0;
        }
    }
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
#[derive(Debug, Clone, Default)]
struct CrossHistory {
    keys: Vec<f32>,
    values: Vec<f32>,
    heads: usize,
    head_dim: usize,
    len: usize,
    /// Source encoder padding mask (`true` means a real text token).
    valid: Vec<bool>,
}

/// Strict batch-one source route.  It uses the shared `Compute` seam for all
/// learned GEMM, RMSNorm, and attention softmax operations; unsupported GPU
/// coverage is rejected at construction, never replaced by CPU work.
#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
pub(crate) struct DiaBatchOne<'a> {
    cfg: &'a DiaConfig,
    weights: &'a DiaWeights,
    compute: Compute,
    encoder: Vec<f32>,
    cross: Vec<CrossHistory>,
    self_kv: Vec<KvHistory>,
    position: usize,
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
impl<'a> DiaBatchOne<'a> {
    /// Construct the route for an authenticated (non-synthesized) weight set.
    pub(crate) fn from_authenticated(
        cfg: &'a DiaConfig,
        weights: &'a DiaWeights,
        backend: BackendKind,
    ) -> Result<Self> {
        if weights.is_synthesized {
            return Err(VokraError::InvalidArgument(
                "dia native route requires authenticated real weights; synthesized fixtures are test-only"
                    .to_owned(),
            ));
        }
        Self::new_checked(cfg, weights, backend)
    }

    /// Fixture-only route for structural tests.  This never creates a public
    /// production handle and rejects no real checkpoint contract.
    #[cfg(test)]
    pub(crate) fn for_tests(
        cfg: &'a DiaConfig,
        weights: &'a DiaWeights,
        backend: BackendKind,
    ) -> Result<Self> {
        Self::new_checked(cfg, weights, backend)
    }

    fn new_checked(
        cfg: &'a DiaConfig,
        weights: &'a DiaWeights,
        backend: BackendKind,
    ) -> Result<Self> {
        cfg.validate_for_forward()?;
        validate_weights(cfg, weights)?;
        let compute = Compute::for_backend(backend, HOT_OPS)?;
        Ok(Self {
            cfg,
            weights,
            compute,
            encoder: Vec::new(),
            cross: vec![CrossHistory::default(); cfg.decoder.n_layer],
            self_kv: vec![KvHistory::default(); cfg.decoder.n_layer],
            position: 0,
        })
    }

    /// Encode one byte-token sequence and precompute decoder cross-attention
    /// K/V tensors exactly once, as in the official `Dia._prepare_generation`.
    pub(crate) fn prepare(&mut self, text_ids: &[u32]) -> Result<&[f32]> {
        if text_ids.is_empty() || text_ids.len() > self.cfg.text_length {
            return Err(VokraError::InvalidArgument(
                "dia batch-one text length is outside the configured range".to_owned(),
            ));
        }
        for &id in text_ids {
            if id as usize >= self.cfg.src_vocab_size {
                return Err(VokraError::InvalidArgument(
                    "dia text token is outside the source vocabulary".to_owned(),
                ));
            }
        }
        let mut padded = vec![self.cfg.text_pad_value; self.cfg.text_length];
        padded[..text_ids.len()].copy_from_slice(text_ids);
        let mut valid = vec![false; self.cfg.text_length];
        valid[..text_ids.len()].fill(true);
        self.prepare_branch(&padded, &valid)
    }

    /// Prepare one fixed-width source branch. The official implementation
    /// pads every text item to `data.text_length`; the mask is kept separate
    /// because CFG's unconditional branch contains zero ids but shares the
    /// conditional padding mask.
    fn prepare_branch(&mut self, text_ids: &[u32], valid: &[bool]) -> Result<&[f32]> {
        if text_ids.len() != self.cfg.text_length || valid.len() != text_ids.len() {
            return Err(VokraError::InvalidArgument(
                "dia source branch must use configured fixed text length".to_owned(),
            ));
        }
        if !valid.iter().any(|&value| value) {
            return Err(VokraError::InvalidArgument(
                "dia source branch requires at least one valid text position".to_owned(),
            ));
        }
        let d = self.cfg.encoder.n_embd;
        let mut x = Vec::with_capacity(text_ids.len() * d);
        for (&id, &is_valid) in text_ids.iter().zip(valid) {
            if id as usize >= self.cfg.src_vocab_size
                || (!is_valid && id != self.cfg.text_pad_value)
            {
                return Err(VokraError::InvalidArgument(
                    "dia source branch has invalid token/pad value".to_owned(),
                ));
            }
            let start = id as usize * d;
            x.extend_from_slice(&self.weights.text_embedding[start..start + d]);
        }
        for block in &self.weights.encoder_blocks {
            x = encoder_block(&self.compute, self.cfg, block, &x, Some(valid))?;
        }
        let mut normalized = vec![0.0; x.len()];
        self.compute.rms_norm_f32(
            &x,
            &mut normalized,
            self.cfg.text_length,
            d,
            &self.weights.encoder_norm,
            self.cfg.norm_eps,
        )?;
        self.encoder = normalized;
        self.position = 0;
        for cache in &mut self.self_kv {
            cache.clear();
        }
        for (index, block) in self.weights.decoder_blocks.iter().enumerate() {
            let k = project(
                &self.compute,
                &self.encoder,
                self.cfg.encoder.n_embd,
                self.cfg.decoder.cross_query_heads * self.cfg.decoder.cross_head_dim,
                &block.xa_k_proj,
            )?;
            let v = project(
                &self.compute,
                &self.encoder,
                self.cfg.encoder.n_embd,
                self.cfg.decoder.cross_query_heads * self.cfg.decoder.cross_head_dim,
                &block.xa_v_proj,
            )?;
            let k = rope(
                &k,
                self.cfg.decoder.cross_query_heads,
                self.cfg.decoder.cross_head_dim,
                0,
                self.cfg,
            )?;
            self.cross[index] = CrossHistory {
                keys: to_head_major(
                    &k,
                    self.cfg.decoder.cross_query_heads,
                    self.cfg.decoder.cross_head_dim,
                )?,
                values: to_head_major(
                    &v,
                    self.cfg.decoder.cross_query_heads,
                    self.cfg.decoder.cross_head_dim,
                )?,
                heads: self.cfg.decoder.cross_query_heads,
                head_dim: self.cfg.decoder.cross_head_dim,
                len: self.cfg.text_length,
                valid: valid.to_vec(),
            };
        }
        Ok(&self.encoder)
    }

    /// Decode one delayed audio frame and return `[channels, target_vocab]`
    /// logits.  Cache mutation is transactional: every layer is truncated to
    /// its prior length if any later operation fails or emits non-finite data.
    pub(crate) fn step(&mut self, channel_tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        if self.encoder.is_empty() || channel_tokens.len() != self.cfg.channels {
            return Err(VokraError::InvalidArgument(
                "dia decoder step requires prepare() and exactly nine channel tokens".to_owned(),
            ));
        }
        for &token in channel_tokens {
            if token as usize >= self.cfg.tgt_vocab_size {
                return Err(VokraError::InvalidArgument(
                    "dia decoder token is outside the target vocabulary".to_owned(),
                ));
            }
        }
        validate_finite(channel_tokens.iter().map(|&v| v as f32))?;
        let lengths: Vec<usize> = self.self_kv.iter().map(|cache| cache.len).collect();
        let result = self.step_inner(channel_tokens);
        if result.is_err() {
            for (cache, length) in self.self_kv.iter_mut().zip(lengths) {
                cache.truncate(length);
            }
        }
        result
    }

    /// Run a prepared batch-one frame sequence through the persistent
    /// layer-wise cache.  This is the staged full decoder route: each frame is
    /// appended once, rather than rebuilding the prefix for every position.
    pub(crate) fn forward_frames(&mut self, frames: &[Vec<u32>]) -> Result<Vec<Vec<Vec<f32>>>> {
        if frames.is_empty() {
            return Err(VokraError::InvalidArgument(
                "dia frame sequence is empty".to_owned(),
            ));
        }
        frames.iter().map(|frame| self.step(frame)).collect()
    }

    fn self_cache_lengths(&self) -> Vec<usize> {
        self.self_kv.iter().map(|cache| cache.len).collect()
    }

    fn step_inner(&mut self, channel_tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        let d = self.cfg.decoder.n_embd;
        let mut x = vec![0.0; d];
        for (channel, &token) in channel_tokens.iter().enumerate() {
            let table = &self.weights.channel_embeddings[channel];
            let start = token as usize * d;
            for (dst, src) in x.iter_mut().zip(&table[start..start + d]) {
                *dst += *src;
            }
        }
        let q_heads = self.cfg.decoder.gqa_query_heads;
        let kv_heads = self.cfg.decoder.kv_heads;
        let hd = self.cfg.decoder.gqa_head_dim;
        for (layer, block) in self.weights.decoder_blocks.iter().enumerate() {
            let mut norm = vec![0.0; d];
            self.compute
                .rms_norm_f32(&x, &mut norm, 1, d, &block.sa_norm, self.cfg.norm_eps)?;
            let q = rope(
                &project(&self.compute, &norm, d, q_heads * hd, &block.sa_q_proj)?,
                q_heads,
                hd,
                self.position,
                self.cfg,
            )?;
            let k = rope(
                &project(&self.compute, &norm, d, kv_heads * hd, &block.sa_k_proj)?,
                kv_heads,
                hd,
                self.position,
                self.cfg,
            )?;
            let v = project(&self.compute, &norm, d, kv_heads * hd, &block.sa_v_proj)?;
            self.self_kv[layer].append(
                &to_head_major(&k, kv_heads, hd)?,
                &to_head_major(&v, kv_heads, hd)?,
                kv_heads,
                hd,
            )?;
            let sa = attention_cached(
                &self.compute,
                &q,
                &self.self_kv[layer],
                q_heads,
                kv_heads,
                hd,
                None,
            )?;
            add_in_place(
                &mut x,
                &project(&self.compute, &sa, q_heads * hd, d, &block.sa_o_proj)?,
            )?;

            self.compute
                .rms_norm_f32(&x, &mut norm, 1, d, &block.xa_norm, self.cfg.norm_eps)?;
            let q = rope(
                &project(&self.compute, &norm, d, q_heads * hd, &block.xa_q_proj)?,
                q_heads,
                hd,
                self.position,
                self.cfg,
            )?;
            let cross = &self.cross[layer];
            let xa = attention_cross(&self.compute, &q, cross, q_heads, hd)?;
            add_in_place(
                &mut x,
                &project(&self.compute, &xa, q_heads * hd, d, &block.xa_o_proj)?,
            )?;

            self.compute
                .rms_norm_f32(&x, &mut norm, 1, d, &block.ffn_norm, self.cfg.norm_eps)?;
            let gate = project(
                &self.compute,
                &norm,
                d,
                block_gate_width(self.cfg),
                &block.gate_proj,
            )?;
            let up = project(
                &self.compute,
                &norm,
                d,
                block_gate_width(self.cfg),
                &block.up_proj,
            )?;
            let hidden: Vec<f32> = gate
                .into_iter()
                .zip(up)
                .map(|(g, u)| g * sigmoid(g) * u)
                .collect();
            add_in_place(
                &mut x,
                &project(
                    &self.compute,
                    &hidden,
                    block_gate_width(self.cfg),
                    d,
                    &block.down_proj,
                )?,
            )?;
        }
        let mut final_norm = vec![0.0; d];
        self.compute.rms_norm_f32(
            &x,
            &mut final_norm,
            1,
            d,
            &self.weights.decoder_norm,
            self.cfg.norm_eps,
        )?;
        let mut logits = Vec::with_capacity(self.cfg.channels);
        for head in &self.weights.logit_heads {
            logits.push(project(
                &self.compute,
                &final_norm,
                d,
                self.cfg.tgt_vocab_size,
                head,
            )?);
        }
        validate_finite(logits.iter().flatten().copied())?;
        self.position += 1;
        Ok(logits)
    }

    fn rollback_self_cache(&mut self, lengths: &[usize], position: usize) {
        for (cache, &length) in self.self_kv.iter_mut().zip(lengths) {
            cache.truncate(length);
        }
        self.position = position;
    }
}

/// The source generation path always evaluates CFG as a two-item batch:
/// unconditional item at index zero and conditional item at index one. This
/// crate-private wrapper keeps both encoder/cross and decoder KV histories
/// persistent; it is not a public production loader.
#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
pub(crate) struct DiaCfgBatchOne<'a> {
    uncond: DiaBatchOne<'a>,
    cond: DiaBatchOne<'a>,
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
impl<'a> DiaCfgBatchOne<'a> {
    pub(crate) fn from_authenticated(
        cfg: &'a DiaConfig,
        weights: &'a DiaWeights,
        backend: BackendKind,
    ) -> Result<Self> {
        Ok(Self {
            uncond: DiaBatchOne::from_authenticated(cfg, weights, backend)?,
            cond: DiaBatchOne::from_authenticated(cfg, weights, backend)?,
        })
    }

    #[cfg(test)]
    pub(crate) fn for_tests(
        cfg: &'a DiaConfig,
        weights: &'a DiaWeights,
        backend: BackendKind,
    ) -> Result<Self> {
        Ok(Self {
            uncond: DiaBatchOne::for_tests(cfg, weights, backend)?,
            cond: DiaBatchOne::for_tests(cfg, weights, backend)?,
        })
    }

    /// Prepare the official pair. The unconditional embedding input is all
    /// pad/zero ids, while both branches use the conditional non-pad mask.
    pub(crate) fn prepare(&mut self, text_ids: &[u32]) -> Result<()> {
        if text_ids.is_empty() || text_ids.len() > self.cond.cfg.text_length {
            return Err(VokraError::InvalidArgument(
                "dia CFG text length is outside the configured range".to_owned(),
            ));
        }
        let mut padded = vec![self.cond.cfg.text_pad_value; self.cond.cfg.text_length];
        padded[..text_ids.len()].copy_from_slice(text_ids);
        let mut valid = vec![false; self.cond.cfg.text_length];
        valid[..text_ids.len()].fill(true);
        self.uncond.prepare_branch(
            &vec![self.uncond.cfg.text_pad_value; self.uncond.cfg.text_length],
            &valid,
        )?;
        self.cond.prepare_branch(&padded, &valid)?;
        Ok(())
    }

    /// Run one frame through both persistent decoder caches and combine
    /// `cond + cfg_scale * (cond - uncond)` atomically.
    pub(crate) fn step(&mut self, channel_tokens: &[u32], cfg_scale: f32) -> Result<Vec<Vec<f32>>> {
        let uncond_lengths = self.uncond.self_cache_lengths();
        let cond_lengths = self.cond.self_cache_lengths();
        let uncond_position = self.uncond.position;
        let cond_position = self.cond.position;
        let mut logits = self.step_raw(channel_tokens, cfg_scale)?;
        if let Err(error) = constrain_audio_logits(self.cond.cfg, &mut logits) {
            self.uncond
                .rollback_self_cache(&uncond_lengths, uncond_position);
            self.cond.rollback_self_cache(&cond_lengths, cond_position);
            return Err(error);
        }
        Ok(logits)
    }

    /// Decode the valid prefix of an audio prompt before the first AR sample.
    ///
    /// The upstream `_prepare_generation` forwards only the rows before the
    /// final prefill row.  That final row is consumed by the first ordinary
    /// `_decoder_step`, which is important because it applies the post-CFG
    /// audio constraints before sampling.  The prefix is already
    /// delay-applied; unknown sentinels are rejected so they can never be
    /// interpreted as embedding ids.
    pub(crate) fn prefill_audio_prompt(
        &mut self,
        delayed: &[Vec<i32>],
        prefill_steps: usize,
        cfg_scale: f32,
    ) -> Result<()> {
        if prefill_steps == 0 || prefill_steps > delayed.len() {
            return Err(VokraError::InvalidArgument(
                "dia audio prompt prefill extent is invalid".to_owned(),
            ));
        }
        for frame in &delayed[..prefill_steps - 1] {
            let tokens = materialize_prompt_frame(self.cond.cfg, frame)?;
            self.step_raw(&tokens, cfg_scale)?;
        }
        Ok(())
    }

    /// Source-shaped batch-one autoregressive code route. It keeps the paired
    /// KV caches alive, consumes exactly one draw per channel/vocabulary
    /// candidate for stochastic sampling, and returns sanitized DAC codes.
    /// PCM decoding is deliberately outside this route until the authenticated
    /// DAC contract is available.
    ///
    /// Errors after the first cache-mutating operation consume the route's
    /// decoder state.  This mirrors the deliberately crate-private staged
    /// API: callers must discard the route after an error rather than treating
    /// a failed generation as an atomic transaction.
    #[allow(clippy::too_many_arguments)] // Source-shaped route keeps each parity-controlled input separate.
    pub(crate) fn generate_codes(
        &mut self,
        cfg: &DiaConfig,
        text_ids: &[u32],
        prompt: Option<&[Vec<u32>]>,
        max_tokens: usize,
        cfg_scale: f32,
        params: SamplingParams,
        draws: &[f32],
    ) -> Result<Vec<Vec<u32>>> {
        if max_tokens <= *cfg.delay_pattern.iter().max().unwrap_or(&0)
            || max_tokens > cfg.audio_length
            || !cfg_scale.is_finite()
            || self.cond.cfg != cfg
        {
            return Err(VokraError::InvalidArgument(
                "dia generation configuration is invalid".to_owned(),
            ));
        }
        let (mut delayed, prefill_steps) = prepare_audio_prompt(cfg, prompt)?;
        self.prepare(text_ids)?;
        let per_sample = cfg
            .channels
            .checked_mul(cfg.tgt_vocab_size)
            .ok_or_else(|| VokraError::InvalidArgument("dia draw count overflow".to_owned()))?;
        let max_delay = *cfg.delay_pattern.iter().max().unwrap_or(&0);
        let mut draw_offset = 0usize;
        self.prefill_audio_prompt(&delayed, prefill_steps, cfg_scale)?;

        // `dec_step` is the row consumed by the decoder.  The sampled row is
        // always `current_step_idx = dec_step + 1`, including the first row
        // after prefill.  This is the same indexing used by upstream
        // `_generate`; in particular, there is no unconstrained special-case
        // first sample.
        let mut dec_step = prefill_steps - 1;
        let mut eos_detected = false;
        let mut eos_countdown: Option<usize> = None;
        let mut finished_step: Option<usize> = None;
        let mut bos_over = false;
        while dec_step < max_tokens {
            if eos_countdown == Some(0) {
                break;
            }
            let current_step_idx = dec_step + 1;
            while current_step_idx >= delayed.len() {
                delayed.push(vec![DIA_UNKNOWN; cfg.channels]);
            }
            let input = materialize_prompt_frame(cfg, &delayed[dec_step])?;
            let logits = self.step(&input, cfg_scale)?;
            let sample_draws: &[f32] = if params.temperature == 0.0 {
                &[]
            } else {
                take_draws(draws, &mut draw_offset, per_sample)?
            };
            let mut next = sample_tokens_inner(
                &logits,
                params,
                sample_draws,
                Some(cfg.audio_eos_value as usize),
            )?;

            let active = eos_countdown != Some(0);
            let eos_trigger = active
                && !eos_detected
                && (next.first().copied() == Some(cfg.audio_eos_value)
                    || current_step_idx >= max_tokens.saturating_sub(max_delay));
            if eos_trigger {
                eos_detected = true;
                if eos_countdown.is_none() {
                    eos_countdown = Some(max_delay);
                    finished_step = Some(current_step_idx);
                }
            }
            if let Some(remaining) = eos_countdown {
                if remaining > 0 {
                    eos_countdown = Some(apply_generation_drain(cfg, &mut next, remaining)?);
                }
            }

            // Upstream keeps the delayed prompt/BOS values until the
            // generated stream has passed the largest delay, then switches
            // to overwrite semantics.  Before that point this is a masked
            // scatter into unknown slots only.
            if !bos_over && dec_step.saturating_sub(prefill_steps) > max_delay {
                bos_over = true;
            }
            write_generated_frame(&mut delayed, current_step_idx, &next, cfg, bos_over)?;
            dec_step += 1;
        }
        if draw_offset != draws.len() {
            return Err(VokraError::InvalidArgument(
                "Dia sampling draw packet contains unused candidate draws".to_owned(),
            ));
        }
        let final_step = dec_step + 1;
        let finished_step = finished_step.unwrap_or(final_step.saturating_sub(max_delay));
        let generated_length = finished_step.saturating_sub(prefill_steps);
        revert_generated_audio(cfg, &delayed, prefill_steps, generated_length)
    }

    /// Same paired cache operation without audio-token constraints. The
    /// official initial prefill sample applies constraints only on loop
    /// samples, so this is intentionally crate-private and narrowly scoped.
    #[allow(clippy::question_mark)] // Preserve explicit rollback sequencing between cache branches.
    fn step_raw(&mut self, channel_tokens: &[u32], cfg_scale: f32) -> Result<Vec<Vec<f32>>> {
        let uncond_lengths = self.uncond.self_cache_lengths();
        let cond_lengths = self.cond.self_cache_lengths();
        let uncond_position = self.uncond.position;
        let cond_position = self.cond.position;
        let uncond = match self.uncond.step(channel_tokens) {
            Ok(logits) => logits,
            Err(error) => return Err(error),
        };
        let cond = match self.cond.step(channel_tokens) {
            Ok(logits) => logits,
            Err(error) => {
                self.uncond
                    .rollback_self_cache(&uncond_lengths, uncond_position);
                return Err(error);
            }
        };
        match classifier_free_guidance(&cond, &uncond, cfg_scale) {
            Ok(logits) => Ok(logits),
            Err(error) => {
                self.uncond
                    .rollback_self_cache(&uncond_lengths, uncond_position);
                self.cond.rollback_self_cache(&cond_lengths, cond_position);
                Err(error)
            }
        }
    }
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn take_draws<'a>(draws: &'a [f32], offset: &mut usize, count: usize) -> Result<&'a [f32]> {
    let end = offset
        .checked_add(count)
        .ok_or_else(|| VokraError::InvalidArgument("dia draw offset overflow".to_owned()))?;
    if end > draws.len() {
        return Err(VokraError::InvalidArgument(
            "Dia sampling draw packet is shorter than consumed candidates".to_owned(),
        ));
    }
    let result = &draws[*offset..end];
    *offset = end;
    Ok(result)
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn write_generated_frame(
    delayed: &mut [Vec<i32>],
    index: usize,
    sampled: &[u32],
    cfg: &DiaConfig,
    overwrite: bool,
) -> Result<()> {
    let frame = delayed.get_mut(index).ok_or_else(|| {
        VokraError::InvalidArgument("dia generation write exceeds delayed buffer".to_owned())
    })?;
    if sampled.len() != cfg.channels || frame.len() != cfg.channels {
        return Err(VokraError::InvalidArgument(
            "dia generation frame shape mismatch".to_owned(),
        ));
    }
    for (slot, &token) in frame.iter_mut().zip(sampled) {
        if overwrite || *slot == DIA_UNKNOWN {
            *slot = token as i32;
        }
    }
    Ok(())
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn apply_generation_drain(cfg: &DiaConfig, next: &mut [u32], remaining: usize) -> Result<usize> {
    if remaining == 0 || remaining > cfg.delay_pattern.iter().copied().max().unwrap_or(0) {
        return Err(VokraError::InvalidArgument(
            "dia EOS drain countdown is invalid".to_owned(),
        ));
    }
    if next.len() != cfg.channels {
        return Err(VokraError::InvalidArgument(
            "dia EOS drain frame width mismatch".to_owned(),
        ));
    }
    let elapsed = cfg.delay_pattern.iter().copied().max().unwrap_or(0) - remaining;
    for (channel, token) in next.iter_mut().enumerate() {
        if elapsed == cfg.delay_pattern[channel] {
            *token = cfg.audio_eos_value;
        } else if elapsed > cfg.delay_pattern[channel] {
            *token = cfg.audio_pad_value;
        }
    }
    Ok(remaining - 1)
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn materialize_prompt_frame(cfg: &DiaConfig, frame: &[i32]) -> Result<Vec<u32>> {
    if frame.len() != cfg.channels {
        return Err(VokraError::InvalidArgument(
            "dia delayed prompt frame width mismatch".to_owned(),
        ));
    }
    frame
        .iter()
        .map(|&token| {
            if token < 0 || token as usize >= cfg.tgt_vocab_size {
                Err(VokraError::InvalidArgument(
                    "dia delayed prompt contains an unknown embedding slot".to_owned(),
                ))
            } else {
                Ok(token as u32)
            }
        })
        .collect()
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn block_gate_width(cfg: &DiaConfig) -> usize {
    cfg.decoder.n_hidden
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn encoder_block(
    compute: &Compute,
    cfg: &DiaConfig,
    block: &DiaEncoderBlockWeights,
    input: &[f32],
    valid: Option<&[bool]>,
) -> Result<Vec<f32>> {
    let rows = input.len() / cfg.encoder.n_embd;
    let mut norm = vec![0.0; input.len()];
    compute.rms_norm_f32(
        input,
        &mut norm,
        rows,
        cfg.encoder.n_embd,
        &block.norm_1,
        cfg.norm_eps,
    )?;
    let q = rope(
        &project(
            compute,
            &norm,
            cfg.encoder.n_embd,
            cfg.encoder.attn_hidden(),
            &block.q_proj,
        )?,
        cfg.encoder.n_head,
        cfg.encoder.head_dim,
        0,
        cfg,
    )?;
    let k = rope(
        &project(
            compute,
            &norm,
            cfg.encoder.n_embd,
            cfg.encoder.attn_hidden(),
            &block.k_proj,
        )?,
        cfg.encoder.n_head,
        cfg.encoder.head_dim,
        0,
        cfg,
    )?;
    let v = project(
        compute,
        &norm,
        cfg.encoder.n_embd,
        cfg.encoder.attn_hidden(),
        &block.v_proj,
    )?;
    let attn = attention_full(
        compute,
        &q,
        &k,
        &v,
        rows,
        cfg.encoder.n_head,
        cfg.encoder.n_head,
        cfg.encoder.head_dim,
        valid,
    )?;
    let attn = project(
        compute,
        &attn,
        cfg.encoder.attn_hidden(),
        cfg.encoder.n_embd,
        &block.o_proj,
    )?;
    let mut x = input.to_vec();
    add_in_place(&mut x, &attn)?;
    compute.rms_norm_f32(
        &x,
        &mut norm,
        rows,
        cfg.encoder.n_embd,
        &block.norm_2,
        cfg.norm_eps,
    )?;
    let gate = project(
        compute,
        &norm,
        cfg.encoder.n_embd,
        cfg.encoder.n_hidden,
        &block.gate_proj,
    )?;
    let up = project(
        compute,
        &norm,
        cfg.encoder.n_embd,
        cfg.encoder.n_hidden,
        &block.up_proj,
    )?;
    let hidden: Vec<f32> = gate
        .into_iter()
        .zip(up)
        .map(|(g, u)| g * sigmoid(g) * u)
        .collect();
    let ffn = project(
        compute,
        &hidden,
        cfg.encoder.n_hidden,
        cfg.encoder.n_embd,
        &block.down_proj,
    )?;
    add_in_place(&mut x, &ffn)?;
    Ok(x)
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn project(
    compute: &Compute,
    input: &[f32],
    input_dim: usize,
    output_dim: usize,
    weight: &[f32],
) -> Result<Vec<f32>> {
    if input_dim == 0
        || output_dim == 0
        || !input.len().is_multiple_of(input_dim)
        || weight.len() != input_dim * output_dim
    {
        return Err(VokraError::InvalidArgument(
            "dia linear shape mismatch".to_owned(),
        ));
    }
    let rows = input.len() / input_dim;
    let mut output = vec![0.0; rows * output_dim];
    compute.gemm_f32(
        rows,
        output_dim,
        input_dim,
        input,
        weight,
        None,
        &mut output,
    )?;
    validate_finite(output.iter().copied())?;
    Ok(output)
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn rope(
    input: &[f32],
    heads: usize,
    head_dim: usize,
    position: usize,
    cfg: &DiaConfig,
) -> Result<Vec<f32>> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) || input.len() % (heads * head_dim) != 0 {
        return Err(VokraError::InvalidArgument(
            "dia RoPE shape mismatch".to_owned(),
        ));
    }
    let rows = input.len() / (heads * head_dim);
    let mut out = input.to_vec();
    let half = head_dim / 2;
    for row in 0..rows {
        let pos = (position + row) as f32;
        for head in 0..heads {
            let base = (row * heads + head) * head_dim;
            for i in 0..half {
                let fraction = (2 * i) as f32 / head_dim as f32;
                let scale = cfg.rope_min_timescale
                    * (cfg.rope_max_timescale / cfg.rope_min_timescale).powf(fraction);
                let angle = pos / scale;
                let (sin, cos) = angle.sin_cos();
                let first = input[base + i];
                let second = input[base + half + i];
                out[base + i] = first * cos - second * sin;
                out[base + half + i] = second * cos + first * sin;
            }
        }
    }
    Ok(out)
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn to_head_major(input: &[f32], heads: usize, head_dim: usize) -> Result<Vec<f32>> {
    let rows = heads
        .checked_mul(head_dim)
        .and_then(|width| input.len().checked_div(width))
        .ok_or_else(|| VokraError::InvalidArgument("dia head reshape overflow".to_owned()))?;
    if input.len() != rows * heads * head_dim {
        return Err(VokraError::InvalidArgument(
            "dia head reshape mismatch".to_owned(),
        ));
    }
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        for head in 0..heads {
            let src = (row * heads + head) * head_dim;
            let dst = (head * rows + row) * head_dim;
            output[dst..dst + head_dim].copy_from_slice(&input[src..src + head_dim]);
        }
    }
    Ok(output)
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
#[allow(clippy::too_many_arguments)] // Attention arguments mirror the fixed source tensor contract.
fn attention_full(
    compute: &Compute,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    valid: Option<&[bool]>,
) -> Result<Vec<f32>> {
    if let Some(mask) = valid {
        if mask.len() != rows {
            return Err(VokraError::InvalidArgument(
                "dia encoder attention mask shape mismatch".to_owned(),
            ));
        }
    }
    let q_head = to_head_major(q, q_heads, head_dim)?;
    let k_head = to_head_major(k, kv_heads, head_dim)?;
    let v_head = to_head_major(v, kv_heads, head_dim)?;
    let mut out = vec![0.0; rows * q_heads * head_dim];
    let groups = q_heads / kv_heads;
    for q_head_index in 0..q_heads {
        let kv_head_index = q_head_index / groups;
        let q_slice = &q_head[q_head_index * rows * head_dim..(q_head_index + 1) * rows * head_dim];
        let k_slice =
            &k_head[kv_head_index * rows * head_dim..(kv_head_index + 1) * rows * head_dim];
        let v_slice =
            &v_head[kv_head_index * rows * head_dim..(kv_head_index + 1) * rows * head_dim];
        let mut kt = vec![0.0; head_dim * rows];
        transpose(k_slice, &mut kt, rows, head_dim)?;
        let mut scores = vec![0.0; rows * rows];
        compute.gemm_f32(rows, rows, head_dim, q_slice, &kt, None, &mut scores)?;
        if let Some(mask) = valid {
            for query in 0..rows {
                for key in 0..rows {
                    if mask[query] != mask[key] {
                        scores[query * rows + key] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        let mut probs = vec![0.0; scores.len()];
        compute.softmax_f32(&scores, &mut probs, rows, rows)?;
        let mut context = vec![0.0; rows * head_dim];
        compute.gemm_f32(rows, head_dim, rows, &probs, v_slice, None, &mut context)?;
        for row in 0..rows {
            let dst = (row * q_heads + q_head_index) * head_dim;
            out[dst..dst + head_dim]
                .copy_from_slice(&context[row * head_dim..(row + 1) * head_dim]);
        }
    }
    Ok(out)
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn attention_cached(
    compute: &Compute,
    q: &[f32],
    cache: &KvHistory,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    valid: Option<&[bool]>,
) -> Result<Vec<f32>> {
    if let Some(mask) = valid {
        if mask.len() != cache.len || !mask.iter().any(|&value| value) {
            return Err(VokraError::InvalidArgument(
                "dia cross-attention mask shape is invalid".to_owned(),
            ));
        }
    }
    let q_head = to_head_major(q, q_heads, head_dim)?;
    let mut out = vec![0.0; q_heads * head_dim];
    let groups = q_heads / kv_heads;
    for q_head_index in 0..q_heads {
        let kv_head_index = q_head_index / groups;
        let q_slice = &q_head[q_head_index * head_dim..(q_head_index + 1) * head_dim];
        let k_slice = &cache.keys
            [kv_head_index * cache.len * head_dim..(kv_head_index + 1) * cache.len * head_dim];
        let v_slice = &cache.values
            [kv_head_index * cache.len * head_dim..(kv_head_index + 1) * cache.len * head_dim];
        let mut scores = vec![0.0; cache.len];
        let mut kt = vec![0.0; head_dim * cache.len];
        transpose(k_slice, &mut kt, cache.len, head_dim)?;
        compute.gemm_f32(1, cache.len, head_dim, q_slice, &kt, None, &mut scores)?;
        if let Some(mask) = valid {
            for (index, &is_valid) in mask.iter().enumerate() {
                if !is_valid {
                    scores[index] = f32::NEG_INFINITY;
                }
            }
        }
        let mut probs = vec![0.0; cache.len];
        compute.softmax_f32(&scores, &mut probs, 1, cache.len)?;
        let mut context = vec![0.0; head_dim];
        compute.gemm_f32(1, head_dim, cache.len, &probs, v_slice, None, &mut context)?;
        out[q_head_index * head_dim..(q_head_index + 1) * head_dim].copy_from_slice(&context);
    }
    Ok(out)
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn attention_cross(
    compute: &Compute,
    q: &[f32],
    cache: &CrossHistory,
    q_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    attention_cached(
        compute,
        q,
        &KvHistory {
            keys: cache.keys.clone(),
            values: cache.values.clone(),
            heads: cache.heads,
            head_dim: cache.head_dim,
            len: cache.len,
        },
        q_heads,
        cache.heads,
        head_dim,
        Some(&cache.valid),
    )
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn transpose(input: &[f32], output: &mut [f32], rows: usize, cols: usize) -> Result<()> {
    if input.len() != rows * cols || output.len() != rows * cols {
        return Err(VokraError::InvalidArgument(
            "dia transpose shape mismatch".to_owned(),
        ));
    }
    for row in 0..rows {
        for col in 0..cols {
            output[col * rows + row] = input[row * cols + col];
        }
    }
    Ok(())
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn add_in_place(dst: &mut [f32], src: &[f32]) -> Result<()> {
    if dst.len() != src.len() {
        return Err(VokraError::InvalidArgument(
            "dia residual shape mismatch".to_owned(),
        ));
    }
    for (left, right) in dst.iter_mut().zip(src) {
        *left += *right;
    }
    validate_finite(dst.iter().copied())
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn validate_finite(mut values: impl Iterator<Item = f32>) -> Result<()> {
    if values.all(f32::is_finite) {
        Ok(())
    } else {
        Err(VokraError::InvalidArgument(
            "dia route encountered non-finite data".to_owned(),
        ))
    }
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn validate_len(actual: usize, expected: usize, label: &str) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(VokraError::InvalidArgument(format!(
            "dia {label} shape mismatch: {actual} != {expected}"
        )))
    }
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn validate_weights(cfg: &DiaConfig, weights: &DiaWeights) -> Result<()> {
    let enc = &cfg.encoder;
    let dec = &cfg.decoder;
    validate_len(
        weights.text_embedding.len(),
        cfg.src_vocab_size * enc.n_embd,
        "text embedding",
    )?;
    validate_len(
        weights.encoder_blocks.len(),
        enc.n_layer,
        "encoder block count",
    )?;
    validate_len(weights.encoder_norm.len(), enc.n_embd, "encoder norm")?;
    validate_len(
        weights.channel_embeddings.len(),
        cfg.channels,
        "channel embedding count",
    )?;
    validate_len(
        weights.decoder_blocks.len(),
        dec.n_layer,
        "decoder block count",
    )?;
    validate_len(weights.decoder_norm.len(), dec.n_embd, "decoder norm")?;
    validate_len(weights.logit_heads.len(), cfg.channels, "logit head count")?;
    for block in &weights.encoder_blocks {
        validate_len(block.norm_1.len(), enc.n_embd, "encoder norm_1")?;
        validate_len(block.norm_2.len(), enc.n_embd, "encoder norm_2")?;
        validate_len(
            block.q_proj.len(),
            enc.n_embd * enc.attn_hidden(),
            "encoder q",
        )?;
        validate_len(
            block.k_proj.len(),
            enc.n_embd * enc.attn_hidden(),
            "encoder k",
        )?;
        validate_len(
            block.v_proj.len(),
            enc.n_embd * enc.attn_hidden(),
            "encoder v",
        )?;
        validate_len(
            block.o_proj.len(),
            enc.attn_hidden() * enc.n_embd,
            "encoder o",
        )?;
        validate_len(
            block.gate_proj.len(),
            enc.n_embd * enc.n_hidden,
            "encoder gate",
        )?;
        validate_len(block.up_proj.len(), enc.n_embd * enc.n_hidden, "encoder up")?;
        validate_len(
            block.down_proj.len(),
            enc.n_hidden * enc.n_embd,
            "encoder down",
        )?;
    }
    for table in &weights.channel_embeddings {
        validate_len(
            table.len(),
            cfg.tgt_vocab_size * dec.n_embd,
            "channel embedding",
        )?;
    }
    let kv = dec.kv_hidden_dim();
    let cross = dec.cross_query_heads * dec.cross_head_dim;
    for block in &weights.decoder_blocks {
        validate_len(block.sa_norm.len(), dec.n_embd, "decoder sa norm")?;
        validate_len(
            block.sa_q_proj.len(),
            dec.n_embd * dec.n_embd,
            "decoder sa q",
        )?;
        validate_len(block.sa_k_proj.len(), dec.n_embd * kv, "decoder sa k")?;
        validate_len(block.sa_v_proj.len(), dec.n_embd * kv, "decoder sa v")?;
        validate_len(
            block.sa_o_proj.len(),
            dec.n_embd * dec.n_embd,
            "decoder sa o",
        )?;
        validate_len(block.xa_norm.len(), dec.n_embd, "decoder xa norm")?;
        validate_len(block.xa_q_proj.len(), dec.n_embd * cross, "decoder xa q")?;
        validate_len(block.xa_k_proj.len(), enc.n_embd * cross, "decoder xa k")?;
        validate_len(block.xa_v_proj.len(), enc.n_embd * cross, "decoder xa v")?;
        validate_len(block.xa_o_proj.len(), cross * dec.n_embd, "decoder xa o")?;
        validate_len(block.ffn_norm.len(), dec.n_embd, "decoder ffn norm")?;
        validate_len(
            block.gate_proj.len(),
            dec.n_embd * dec.n_hidden,
            "decoder gate",
        )?;
        validate_len(block.up_proj.len(), dec.n_embd * dec.n_hidden, "decoder up")?;
        validate_len(
            block.down_proj.len(),
            dec.n_hidden * dec.n_embd,
            "decoder down",
        )?;
    }
    for head in &weights.logit_heads {
        validate_len(head.len(), dec.n_embd * cfg.tgt_vocab_size, "logit head")?;
    }
    validate_finite(all_weights(weights))
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn all_weights<'a>(weights: &'a DiaWeights) -> impl Iterator<Item = f32> + 'a {
    weights
        .text_embedding
        .iter()
        .copied()
        .chain(weights.encoder_norm.iter().copied())
        .chain(weights.decoder_norm.iter().copied())
        .chain(weights.channel_embeddings.iter().flatten().copied())
        .chain(weights.logit_heads.iter().flatten().copied())
        .chain(
            weights
                .encoder_blocks
                .iter()
                .flat_map(|b| {
                    b.norm_1
                        .iter()
                        .chain(&b.q_proj)
                        .chain(&b.k_proj)
                        .chain(&b.v_proj)
                        .chain(&b.o_proj)
                        .chain(&b.norm_2)
                        .chain(&b.gate_proj)
                        .chain(&b.up_proj)
                        .chain(&b.down_proj)
                })
                .copied(),
        )
        .chain(
            weights
                .decoder_blocks
                .iter()
                .flat_map(|b| {
                    b.sa_norm
                        .iter()
                        .chain(&b.sa_q_proj)
                        .chain(&b.sa_k_proj)
                        .chain(&b.sa_v_proj)
                        .chain(&b.sa_o_proj)
                        .chain(&b.xa_norm)
                        .chain(&b.xa_q_proj)
                        .chain(&b.xa_k_proj)
                        .chain(&b.xa_v_proj)
                        .chain(&b.xa_o_proj)
                        .chain(&b.ffn_norm)
                        .chain(&b.gate_proj)
                        .chain(&b.up_proj)
                        .chain(&b.down_proj)
                })
                .copied(),
        )
}

/// Dia source delay pattern: prepend BOS according to each channel's delay
/// and preserve trailing PAD slots. Input and output are time-major frames.
#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
pub(crate) fn apply_delay_pattern(cfg: &DiaConfig, codes: &[Vec<u32>]) -> Result<Vec<Vec<u32>>> {
    validate_codes(cfg, codes)?;
    let extra = *cfg.delay_pattern.iter().max().unwrap_or(&0);
    let mut delayed = vec![vec![cfg.audio_pad_value; cfg.channels]; codes.len() + extra];
    for (time, frame) in delayed.iter_mut().enumerate() {
        for channel in 0..cfg.channels {
            let source = time as isize - cfg.delay_pattern[channel] as isize;
            frame[channel] = if source < 0 {
                cfg.audio_bos_value
            } else if (source as usize) < codes.len() {
                codes[source as usize][channel]
            } else {
                cfg.audio_pad_value
            };
        }
    }
    Ok(delayed)
}

/// Generation-only sentinel used by the upstream `DecoderOutput` for a slot
/// that has not been sampled yet. It must never reach an embedding lookup.
#[allow(dead_code)] // Used by the staged generation helpers and their future binder.
pub(crate) const DIA_UNKNOWN: i32 = -1;

/// Build the official BOS/prompt/delay layout. Unlike the strict public
/// `apply_delay_pattern`, this representation preserves unknown slots so the
/// generation writer can perform masked-scatter semantics without replacing
/// fixed delay padding.
#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
pub(crate) fn prepare_audio_prompt(
    cfg: &DiaConfig,
    prompt: Option<&[Vec<u32>]>,
) -> Result<(Vec<Vec<i32>>, usize)> {
    let prompt_len = prompt.map_or(0, |frames| frames.len());
    if prompt_len >= cfg.audio_length {
        return Err(VokraError::InvalidArgument(
            "dia audio prompt exceeds configured audio length".to_owned(),
        ));
    }
    if let Some(frames) = prompt {
        validate_codes(cfg, frames)?;
        if frames
            .iter()
            .flatten()
            .any(|&token| token >= cfg.audio_eos_value)
        {
            return Err(VokraError::InvalidArgument(
                "dia audio prompt must contain DAC code ids only".to_owned(),
            ));
        }
    }
    let max_delay = *cfg.delay_pattern.iter().max().unwrap_or(&0);
    // Upstream allocates `max(prompt_len + max_delay, 1)` columns: row zero
    // is BOS and prompt rows occupy 1..=prompt_len.
    let source_len = prompt_len.saturating_add(max_delay).max(prompt_len + 1);
    let mut source = vec![vec![DIA_UNKNOWN; cfg.channels]; source_len];
    source[0].fill(cfg.audio_bos_value as i32);
    if let Some(frames) = prompt {
        for (time, frame) in frames.iter().enumerate() {
            source[time + 1] = frame.iter().map(|&token| token as i32).collect();
        }
    }
    let mut delayed = vec![vec![DIA_UNKNOWN; cfg.channels]; source_len];
    for (time, frame) in delayed.iter_mut().enumerate() {
        for channel in 0..cfg.channels {
            let source_time = time as isize - cfg.delay_pattern[channel] as isize;
            frame[channel] = if source_time < 0 {
                cfg.audio_bos_value as i32
            } else if (source_time as usize) < source.len() {
                source[source_time as usize][channel]
            } else {
                DIA_UNKNOWN
            };
        }
    }
    Ok((delayed, prompt_len + 1))
}

/// Revert only the generated slice of a delayed generation buffer, excluding
/// all prompt/BOS rows, then apply the source's terminal sanitization
/// (`>= audio_eos_value` becomes DAC code zero).
///
/// Upstream computes `generated_length = finished_step - prefill_steps`,
/// copies `generated_length + max_delay` rows beginning at `prefill_steps`,
/// reverts that slice, and finally truncates to `generated_length`. Keeping the
/// slice operation here prevents prompt/BOS rows from leaking into returned
/// codes when an audio prompt is present.
#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
pub(crate) fn revert_generated_audio(
    cfg: &DiaConfig,
    delayed: &[Vec<i32>],
    prefill_steps: usize,
    generated_length: usize,
) -> Result<Vec<Vec<u32>>> {
    let max_delay = *cfg.delay_pattern.iter().max().unwrap_or(&0);
    let end = prefill_steps
        .checked_add(generated_length)
        .and_then(|value| value.checked_add(max_delay))
        .ok_or_else(|| VokraError::InvalidArgument("dia generated slice overflow".to_owned()))?;
    if prefill_steps > delayed.len() || end > delayed.len() {
        return Err(VokraError::InvalidArgument(
            "dia generated delay slice is invalid".to_owned(),
        ));
    }
    if delayed[prefill_steps..end]
        .iter()
        .any(|frame| frame.len() != cfg.channels)
    {
        return Err(VokraError::InvalidArgument(
            "dia generated delayed frame width mismatch".to_owned(),
        ));
    }
    let generated = &delayed[prefill_steps..end];
    let mut output = vec![vec![0; cfg.channels]; generated_length];
    for (time, frame) in output.iter_mut().enumerate() {
        for channel in 0..cfg.channels {
            let source = time + cfg.delay_pattern[channel];
            let token = generated[source][channel];
            frame[channel] = if token < 0 || token as u32 >= cfg.audio_eos_value {
                0
            } else {
                token as u32
            };
        }
    }
    Ok(output)
}

/// Official Dia text boundary: UTF-8 bytes, with `[S1]` and `[S2]` replaced
/// by byte ids 1 and 2, truncated to `data.text_length`.
#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
pub(crate) fn encode_text(text: &str, cfg: &DiaConfig) -> Result<Vec<u32>> {
    // The upstream boundary is byte-first: marker replacement is performed
    // on UTF-8 bytes, and an empty string is a valid (empty) token sequence.
    let replaced = replace_byte_marker(
        &replace_byte_marker(text.as_bytes(), b"[S1]", 1),
        b"[S2]",
        2,
    );
    Ok(replaced
        .iter()
        .take(cfg.text_length)
        .map(|&value| u32::from(value))
        .collect())
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn replace_byte_marker(input: &[u8], marker: &[u8], replacement: u8) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len());
    let mut cursor = 0;
    while cursor < input.len() {
        if input[cursor..].starts_with(marker) {
            output.push(replacement);
            cursor += marker.len();
        } else {
            output.push(input[cursor]);
            cursor += 1;
        }
    }
    output
}

/// Inverse delay mapping.  The public strict helper remains separate; this
/// operation is intended for generation state before terminal sanitization.
#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
pub(crate) fn revert_delay_pattern(cfg: &DiaConfig, delayed: &[Vec<u32>]) -> Result<Vec<Vec<u32>>> {
    if delayed.iter().any(|frame| frame.len() != cfg.channels) {
        return Err(VokraError::InvalidArgument(
            "dia delayed frame width mismatch".to_owned(),
        ));
    }
    let extra = *cfg.delay_pattern.iter().max().unwrap_or(&0);
    let length = delayed.len().saturating_sub(extra);
    let mut result = vec![vec![cfg.audio_pad_value; cfg.channels]; length];
    for (time, frame) in result.iter_mut().enumerate() {
        for channel in 0..cfg.channels {
            let source = time + cfg.delay_pattern[channel];
            if source < delayed.len() {
                frame[channel] = delayed[source][channel];
            }
        }
    }
    Ok(result)
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn validate_codes(cfg: &DiaConfig, codes: &[Vec<u32>]) -> Result<()> {
    if codes.is_empty() || codes.iter().any(|frame| frame.len() != cfg.channels) {
        return Err(VokraError::InvalidArgument(
            "dia code frame shape mismatch".to_owned(),
        ));
    }
    if codes
        .iter()
        .flatten()
        .any(|&token| token as usize >= cfg.tgt_vocab_size)
    {
        return Err(VokraError::InvalidArgument(
            "dia code outside target vocabulary".to_owned(),
        ));
    }
    Ok(())
}

/// Caller-owned sampling controls.  `draws` are the exact independent
/// exponential/uniform draws consumed by the official multinomial path; no
/// hidden RNG is permitted here.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
pub(crate) struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
pub(crate) fn sample_tokens(
    logits: &[Vec<f32>],
    params: SamplingParams,
    draws: &[f32],
    eos: usize,
) -> Result<Vec<u32>> {
    sample_tokens_inner(logits, params, draws, Some(eos))
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn sample_tokens_inner(
    logits: &[Vec<f32>],
    params: SamplingParams,
    draws: &[f32],
    eos: Option<usize>,
) -> Result<Vec<u32>> {
    if !params.temperature.is_finite()
        || params.temperature < 0.0
        || !params.top_p.is_finite()
        || !(0.0..=1.0).contains(&params.top_p)
        || params.top_k == 0
    {
        return Err(VokraError::InvalidArgument(
            "dia sampling parameters are invalid".to_owned(),
        ));
    }
    let vocab = logits.first().map_or(0, Vec::len);
    if vocab == 0 || logits.iter().any(|row| row.len() != vocab) {
        return Err(VokraError::InvalidArgument(
            "dia logits shape is invalid".to_owned(),
        ));
    }
    if params.temperature == 0.0 {
        if !draws.is_empty() {
            return Err(VokraError::InvalidArgument(
                "greedy Dia sampling must not receive draws".to_owned(),
            ));
        }
        return logits
            .iter()
            .map(|row| argmax(row).map(|value| value as u32))
            .collect();
    }
    let expected_draws = logits.len().checked_mul(vocab).ok_or_else(|| {
        VokraError::InvalidArgument("dia sampling draw count overflow".to_owned())
    })?;
    if draws.len() != expected_draws || draws.iter().any(|&draw| !draw.is_finite() || draw <= 0.0) {
        return Err(VokraError::InvalidArgument(
            "Dia sampling requires one positive finite draw per row/vocabulary candidate"
                .to_owned(),
        ));
    }
    let mut result = Vec::with_capacity(logits.len());
    for (row_index, row) in logits.iter().enumerate() {
        let mut values: Vec<f32> = row
            .iter()
            .map(|&value| value / params.temperature)
            .collect();
        if let Some(eos) = eos {
            if eos < vocab && argmax(&values)? != eos {
                values[eos] = f32::NEG_INFINITY;
            }
        }
        if params.top_k > vocab {
            return Err(VokraError::InvalidArgument(
                "Dia top-k exceeds the logits vocabulary".to_owned(),
            ));
        }
        if params.top_k < vocab {
            // `torch.topk` selects exactly k indices.  Threshold masking is
            // not equivalent when the kth value is tied with an excluded
            // value, and its tie order is backend-dependent.  Without a
            // caller-supplied selection packet, fail closed at that boundary
            // instead of claiming tie-exact parity.
            let mut order: Vec<usize> = (0..vocab).collect();
            order.sort_by(|&a, &b| {
                values[b]
                    .partial_cmp(&values[a])
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.cmp(&b))
            });
            let pivot = values[order[params.top_k - 1]];
            if values[order[params.top_k]] == pivot {
                return Err(VokraError::InvalidArgument(
                    "Dia top-k boundary tie requires reference selection indices".to_owned(),
                ));
            }
            for &index in &order[params.top_k..] {
                values[index] = f32::NEG_INFINITY;
            }
        }
        let mut probs = softmax(&values)?;
        if params.top_p < 1.0 {
            let mut order: Vec<usize> = (0..vocab).collect();
            order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(Ordering::Equal));
            let mut cumulative = 0.0;
            for &index in &order {
                let previous = cumulative;
                cumulative += probs[index];
                if previous > params.top_p {
                    probs[index] = 0.0;
                }
            }
            let total: f32 = probs.iter().sum();
            if total.partial_cmp(&0.0) != Some(Ordering::Greater) || !total.is_finite() {
                return Err(VokraError::InvalidArgument(
                    "Dia top-p left no probability mass".to_owned(),
                ));
            }
            for probability in &mut probs {
                *probability /= total;
            }
        }
        let start = row_index * vocab;
        let selected = probs
            .iter()
            .enumerate()
            .map(|(index, &probability)| (index, probability / draws[start + index]))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .map(|(index, _)| index)
            .ok_or_else(|| {
                VokraError::InvalidArgument("Dia sampling produced no candidate".to_owned())
            })?;
        result.push(selected as u32);
    }
    Ok(result)
}

/// Official Dia classifier-free guidance combination:
/// `cond + scale * (cond - uncond)`.  The two branches must have identical
/// channel/vocabulary topology; branch construction and conditioning remain
/// outside this crate-private staged route.
#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
pub(crate) fn classifier_free_guidance(
    conditional: &[Vec<f32>],
    unconditional: &[Vec<f32>],
    scale: f32,
) -> Result<Vec<Vec<f32>>> {
    if !scale.is_finite() || conditional.len() != unconditional.len() || conditional.is_empty() {
        return Err(VokraError::InvalidArgument(
            "dia CFG branches are malformed".to_owned(),
        ));
    }
    let mut output = Vec::with_capacity(conditional.len());
    for (cond, uncond) in conditional.iter().zip(unconditional) {
        if cond.len() != uncond.len() || cond.is_empty() {
            return Err(VokraError::InvalidArgument(
                "dia CFG branch shape mismatch".to_owned(),
            ));
        }
        let row: Vec<f32> = cond
            .iter()
            .zip(uncond)
            .map(|(&c, &u)| c + scale * (c - u))
            .collect();
        validate_finite(row.iter().copied())?;
        output.push(row);
    }
    Ok(output)
}

/// Apply Dia's post-CFG audio constraints before temperature/top-k/top-p.
/// Only channel zero may select EOS; every channel rejects ids above EOS.
#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
pub(crate) fn constrain_audio_logits(cfg: &DiaConfig, logits: &mut [Vec<f32>]) -> Result<()> {
    if logits.len() != cfg.channels
        || logits.iter().any(|row| row.len() != cfg.tgt_vocab_size)
        || cfg.audio_eos_value as usize >= cfg.tgt_vocab_size
    {
        return Err(VokraError::InvalidArgument(
            "dia audio logits shape/configuration is invalid".to_owned(),
        ));
    }
    validate_finite(logits.iter().flatten().copied())?;
    let eos = cfg.audio_eos_value as usize;
    for (channel, row) in logits.iter_mut().enumerate() {
        for value in row.iter_mut().skip(eos + 1) {
            *value = f32::NEG_INFINITY;
        }
        if channel != 0 {
            for value in row.iter_mut().skip(eos) {
                *value = f32::NEG_INFINITY;
            }
        } else {
            row[eos] *= 0.8;
        }
    }
    if logits.iter().flatten().any(|value| value.is_nan()) {
        return Err(VokraError::InvalidArgument(
            "dia constrained audio logits became non-finite".to_owned(),
        ));
    }
    Ok(())
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn softmax(values: &[f32]) -> Result<Vec<f32>> {
    let max = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return Err(VokraError::InvalidArgument(
            "Dia sampling logits are all non-finite".to_owned(),
        ));
    }
    let mut output: Vec<f32> = values
        .iter()
        .map(|&value| {
            if value.is_finite() {
                (value - max).exp()
            } else {
                0.0
            }
        })
        .collect();
    let total: f32 = output.iter().sum();
    if total.partial_cmp(&0.0) != Some(Ordering::Greater) || !total.is_finite() {
        return Err(VokraError::InvalidArgument(
            "Dia sampling softmax is degenerate".to_owned(),
        ));
    }
    for value in &mut output {
        *value /= total;
    }
    Ok(output)
}

#[allow(dead_code)] // staged until the authenticated Dia/DAC binder is wired
fn argmax(values: &[f32]) -> Result<usize> {
    let (&first, rest) = values
        .split_first()
        .ok_or_else(|| VokraError::InvalidArgument("Dia logits are empty".to_owned()))?;
    if first.is_nan() {
        return Err(VokraError::InvalidArgument(
            "Dia logits contain NaN".to_owned(),
        ));
    }
    let mut best_index = 0;
    let mut best_value = first;
    for (index, &value) in rest.iter().enumerate() {
        if value.is_nan() {
            return Err(VokraError::InvalidArgument(
                "Dia logits contain NaN".to_owned(),
            ));
        }
        // `torch.argmax` keeps the first index on an exact tie.
        if value > best_value {
            best_index = index + 1;
            best_value = value;
        }
    }
    Ok(best_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dia::DiaWeights;

    #[test]
    fn source_delay_and_revert_preserve_channel_offsets() {
        let cfg = DiaConfig::tiny_for_tests();
        let codes = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 1, 2]];
        let delayed = apply_delay_pattern(&cfg, &codes).expect("delay");
        assert_eq!(
            delayed[0],
            vec![1, cfg.audio_bos_value, cfg.audio_bos_value]
        );
        assert_eq!(delayed[1][0], 4);
        let restored = revert_delay_pattern(&cfg, &delayed).expect("revert");
        assert_eq!(restored, codes);
    }

    #[test]
    fn generation_prompt_preserves_unknown_slots_and_prefill_extent() {
        let cfg = DiaConfig::tiny_for_tests();
        let prompt = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let (delayed, prefill) = prepare_audio_prompt(&cfg, Some(&prompt)).expect("prompt");
        assert_eq!(prefill, 3);
        assert_eq!(delayed.len(), prompt.len() + 2);
        assert_eq!(delayed[0], vec![cfg.audio_bos_value as i32; 3]);
        assert!(delayed.iter().flatten().any(|&value| value == DIA_UNKNOWN));
        assert!(materialize_prompt_frame(&cfg, &delayed[0]).is_ok());
        assert!(materialize_prompt_frame(&cfg, &[DIA_UNKNOWN; 3]).is_err());
    }

    #[test]
    fn generation_revert_sanitizes_terminal_specials_but_strict_revert_does_not() {
        let cfg = DiaConfig::tiny_for_tests();
        let mut delayed = vec![vec![1, 2, 3]; 8];
        delayed[4][0] = cfg.audio_eos_value as i32;
        delayed[5][1] = cfg.audio_pad_value as i32;
        delayed[6][2] = DIA_UNKNOWN;
        let generated =
            revert_generated_audio(&cfg, &delayed, 0, delayed.len() - 2).expect("revert");
        assert!(
            generated
                .iter()
                .flatten()
                .all(|&value| value < cfg.audio_eos_value)
        );
        let strict = delayed
            .iter()
            .map(|frame| frame.iter().map(|&value| value.max(0) as u32).collect())
            .collect::<Vec<Vec<u32>>>();
        assert!(revert_delay_pattern(&cfg, &strict).is_ok());
    }

    #[test]
    fn eos_drain_preserves_future_channels_and_staggers_eos() {
        let cfg = DiaConfig::tiny_for_tests();
        let mut frame = vec![4, 5, 6];
        let remaining = apply_generation_drain(&cfg, &mut frame, 2).expect("first drain");
        assert_eq!(frame, vec![cfg.audio_eos_value, 5, 6]);
        apply_generation_drain(&cfg, &mut frame, remaining).expect("second drain");
        assert_eq!(frame, vec![cfg.audio_pad_value, cfg.audio_eos_value, 6]);
    }

    #[test]
    fn greedy_generation_consumes_no_draws_and_returns_sanitized_codes() {
        let cfg = DiaConfig::tiny_for_tests();
        let weights = DiaWeights::synthesized(&cfg, 23).expect("fixture");
        let mut pair = DiaCfgBatchOne::for_tests(&cfg, &weights, BackendKind::Cpu).expect("pair");
        let params = SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: cfg.tgt_vocab_size,
        };
        let codes = pair
            .generate_codes(&cfg, &[1], None, 5, 1.0, params, &[])
            .expect("greedy codes");
        assert!(
            codes
                .iter()
                .flatten()
                .all(|&value| value < cfg.audio_eos_value)
        );
    }

    #[test]
    fn source_text_boundary_is_utf8_bytes_with_speaker_markers() {
        let cfg = DiaConfig::tiny_for_tests();
        assert_eq!(
            encode_text("A[S1]é[S2]", &cfg).unwrap(),
            vec![65, 1, 195, 169, 2]
        );
        assert_eq!(encode_text("", &cfg).unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn sampling_consumes_per_candidate_draws_and_keeps_top_p_crossing() {
        let logits = vec![vec![4.0, 3.0, 2.0, 1.0]];
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.7,
            top_k: 4,
        };
        let draws = vec![1.0, 100.0, 1.0, 100.0];
        let result = sample_tokens(&logits, params, &draws, 99).expect("sample");
        assert_eq!(result.len(), 1);
        assert!(result[0] <= 1, "crossing token remains eligible");
        assert!(sample_tokens(&logits, params, &draws[..3], 99).is_err());
    }

    #[test]
    fn top_k_boundary_ties_fail_closed_without_reference_indices() {
        let logits = vec![vec![4.0, 3.0, 2.0, 2.0]];
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 3,
        };
        let draws = vec![1.0; 4];
        let error = sample_tokens(&logits, params, &draws, 99).expect_err("tie must be gated");
        assert!(error.to_string().contains("reference selection indices"));
    }

    #[test]
    fn generated_slice_excludes_prompt_and_bos_rows() {
        let cfg = DiaConfig::tiny_for_tests();
        let mut delayed = vec![vec![0; cfg.channels]; 8];
        // Rows before `prefill_steps` stand in for BOS/prompt content and are
        // intentionally distinct from the generated delayed slice.
        delayed[0] = vec![90; cfg.channels];
        delayed[1] = vec![91; cfg.channels];
        delayed[2] = vec![92; cfg.channels];
        delayed[3] = vec![1, 2, 3];
        delayed[4] = vec![4, 5, 6];
        delayed[5] = vec![7, 1, 2];
        let output = revert_generated_audio(&cfg, &delayed, 3, 1).expect("generated slice");
        assert_eq!(output, vec![vec![1, 5, 2]]);
    }

    #[test]
    fn cfg_combination_is_branch_shape_strict() {
        let conditional = vec![vec![3.0, 1.0]];
        let unconditional = vec![vec![1.0, 1.0]];
        assert_eq!(
            classifier_free_guidance(&conditional, &unconditional, 2.0).unwrap(),
            vec![vec![7.0, 1.0]]
        );
        assert!(classifier_free_guidance(&conditional, &[], 2.0).is_err());
    }

    #[test]
    fn attention_uses_source_scale_one_without_head_dimension_factor() {
        let compute = Compute::cpu();
        let q = vec![2.0, 0.0, 0.0, 0.0];
        let k = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let out = attention_full(&compute, &q, &k, &v, 2, 1, 1, 2, None).expect("attention");
        let p = 2.0_f32.exp() / (2.0_f32.exp() + 1.0);
        assert!((out[0] - p).abs() < 1e-5);
        assert!((out[1] - (1.0 - p)).abs() < 1e-5);
    }

    #[test]
    fn encoder_mask_excludes_padded_keys_and_keeps_pad_group_separate() {
        let compute = Compute::cpu();
        let q = vec![1.0, 0.0, 1.0, 0.0];
        let k = vec![1.0, 0.0, 100.0, 0.0];
        let v = vec![2.0, 0.0, 99.0, 0.0];
        let out = attention_full(&compute, &q, &k, &v, 2, 1, 1, 2, Some(&[true, false]))
            .expect("masked attention");
        assert!((out[0] - 2.0).abs() < 1e-5);
        assert!((out[2] - 99.0).abs() < 1e-5);
    }

    #[test]
    fn fixed_text_padding_and_cfg_pair_are_source_shaped() {
        let cfg = DiaConfig::tiny_for_tests();
        let weights = DiaWeights::synthesized(&cfg, 19).expect("fixture");
        let mut pair = DiaCfgBatchOne::for_tests(&cfg, &weights, BackendKind::Cpu).expect("pair");
        pair.prepare(&[1, 2]).expect("prepare");
        assert_eq!(
            pair.cond.encoder.len(),
            cfg.text_length * cfg.encoder.n_embd
        );
        assert_eq!(pair.uncond.cross[0].valid, pair.cond.cross[0].valid);
        assert_eq!(pair.cond.cross[0].valid.iter().filter(|&&v| v).count(), 2);
        let logits = pair.step(&[cfg.audio_bos_value; 3], 1.5).expect("step");
        assert_eq!(logits.len(), cfg.channels);
    }

    #[test]
    fn post_cfg_audio_constraints_only_allow_eos_on_channel_zero() {
        let cfg = DiaConfig::tiny_for_tests();
        let mut logits = vec![vec![0.0; cfg.tgt_vocab_size]; cfg.channels];
        for row in &mut logits {
            row[cfg.audio_eos_value as usize] = 2.0;
            row[cfg.audio_eos_value as usize + 1] = 3.0;
        }
        constrain_audio_logits(&cfg, &mut logits).expect("constraints");
        assert_eq!(logits[0][cfg.audio_eos_value as usize], 1.6);
        assert!(logits[0][cfg.audio_eos_value as usize + 1].is_infinite());
        assert!(logits[1][cfg.audio_eos_value as usize].is_infinite());
        let first_generated = sample_tokens(
            &logits,
            SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: cfg.tgt_vocab_size,
            },
            &[],
            cfg.audio_eos_value as usize,
        )
        .expect("first constrained sample");
        assert_ne!(first_generated[1], cfg.audio_eos_value);
    }

    #[test]
    fn synthesized_fixture_cannot_enter_authenticated_route() {
        let cfg = DiaConfig::tiny_for_tests();
        let weights = DiaWeights::synthesized(&cfg, 7).expect("fixture");
        assert!(DiaBatchOne::from_authenticated(&cfg, &weights, BackendKind::Cpu).is_err());
        let mut route =
            DiaBatchOne::for_tests(&cfg, &weights, BackendKind::Cpu).expect("test route");
        route.prepare(&[0, 1]).expect("encode");
        let logits = route.step(&[cfg.audio_bos_value; 3]).expect("decode");
        assert_eq!(logits.len(), cfg.channels);
        assert_eq!(logits[0].len(), cfg.tgt_vocab_size);
    }

    #[test]
    fn persistent_cache_sequence_matches_repeated_batch_one_steps() {
        let cfg = DiaConfig::tiny_for_tests();
        let weights = DiaWeights::synthesized(&cfg, 11).expect("fixture");
        let frames = vec![vec![10, 10, 10], vec![1, 2, 3]];
        let mut full = DiaBatchOne::for_tests(&cfg, &weights, BackendKind::Cpu).expect("route");
        full.prepare(&[0, 1]).expect("encode");
        let full_logits = full.forward_frames(&frames).expect("sequence");
        let mut repeated = DiaBatchOne::for_tests(&cfg, &weights, BackendKind::Cpu).expect("route");
        repeated.prepare(&[0, 1]).expect("encode");
        let first = repeated.step(&frames[0]).expect("first");
        let second = repeated.step(&frames[1]).expect("second");
        assert_eq!(full_logits[0], first);
        assert_eq!(full_logits[1], second);
    }

    #[test]
    fn nonfinite_static_weight_is_rejected_before_route() {
        let cfg = DiaConfig::tiny_for_tests();
        let mut weights = DiaWeights::synthesized(&cfg, 13).expect("fixture");
        weights.text_embedding[0] = f32::NAN;
        assert!(DiaBatchOne::for_tests(&cfg, &weights, BackendKind::Cpu).is_err());
    }

    #[test]
    fn invalid_step_does_not_advance_any_layer_cache() {
        let cfg = DiaConfig::tiny_for_tests();
        let weights = DiaWeights::synthesized(&cfg, 17).expect("fixture");
        let mut route = DiaBatchOne::for_tests(&cfg, &weights, BackendKind::Cpu).expect("route");
        route.prepare(&[0, 1]).expect("encode");
        route.step(&[10, 10, 10]).expect("first");
        let before = route.self_cache_lengths();
        assert!(route.step(&[10, 99, 10]).is_err());
        assert_eq!(route.self_cache_lengths(), before);
    }
}
