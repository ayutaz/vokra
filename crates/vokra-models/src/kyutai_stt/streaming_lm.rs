//! Incremental Kyutai STT main-language-model state.
//!
//! This is an original first-party Rust implementation of the published Moshi
//! `LMGen` streaming semantics at the `dep_q == 0` boundary; no upstream source
//! is copied.  It keeps one bounded
//! key/value history per transformer layer, applies RoPE at the absolute
//! frame position, and executes one text/audio frame without recomputing a
//! prefix.  The source contract is Moshi commit
//! `e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`.
//!
//! The first call writes the caller's current audio frame before reading the
//! ring position, but the source's initial-token substitution replaces both
//! text and audio rows at offset zero.  The returned text token is expected to
//! be sampled by the caller and supplied as `Some(token)` on the next call;
//! subsequent calls use that token and the current audio frame directly.

#[cfg(test)]
use super::KyutaiSttTrace;
use super::{
    KYUTAI_STT_HOT_OPS, KyutaiSttAsr, KyutaiSttBlockWeights, KyutaiSttConfig, KyutaiSttTextLogits,
};
use crate::compute::Compute;
use crate::csm::rope::{llama3_inv_freqs, rope_apply_adjacent};
use vokra_core::backend::BackendKind;
use vokra_core::{Result, VokraError};

/// One incremental text-logit result.
#[derive(Debug, Clone, PartialEq)]
pub struct KyutaiSttStreamingLmStep {
    position: usize,
    logits: KyutaiSttTextLogits,
}

impl KyutaiSttStreamingLmStep {
    /// Absolute frame position consumed by this step.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Text logits for this frame (`[1, text_card]`).
    #[must_use]
    pub fn logits(&self) -> &KyutaiSttTextLogits {
        &self.logits
    }
}

#[derive(Debug, Clone)]
struct LayerKv {
    positions: Vec<usize>,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl LayerKv {
    fn new() -> Self {
        Self {
            positions: Vec::new(),
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.positions.clear();
        self.keys.clear();
        self.values.clear();
    }

    fn append(&mut self, position: usize, key: &[f32], value: &[f32], context: usize) {
        let d = key.len();
        if self.positions.len() == context {
            self.positions.remove(0);
            self.keys.drain(..d);
            self.values.drain(..d);
        }
        self.positions.push(position);
        self.keys.extend_from_slice(key);
        self.values.extend_from_slice(value);
    }
}

/// Borrowed incremental main-language-model state for the Kyutai STT engine.
///
/// The state borrows the authenticated engine weights and owns only the
/// bounded per-layer KV history and frame bookkeeping.  A
/// backend is selected at construction and every hot operation is dispatched
/// through [`Compute`]; unsupported backends fail before state mutation.
#[derive(Debug)]
pub struct KyutaiSttStreamingLm<'a> {
    asr: &'a KyutaiSttAsr,
    backend: BackendKind,
    layers: Vec<LayerKv>,
    position: usize,
    poisoned: bool,
    #[cfg(test)]
    fail_after_layer: Option<usize>,
}

impl<'a> KyutaiSttStreamingLm<'a> {
    pub(crate) fn new(asr: &'a KyutaiSttAsr, backend: BackendKind) -> Result<Self> {
        asr.cfg.validate_for_forward()?;
        if asr.cfg.dep_q != 0 {
            return Err(VokraError::InvalidArgument(
                "kyutai-stt streaming LM requires dep_q=0".to_owned(),
            ));
        }
        // Validate coverage at construction so a backend failure cannot occur
        // after a caller has begun mutating a stream.
        let _ = Compute::for_backend(backend, KYUTAI_STT_HOT_OPS)?;
        Ok(Self {
            asr,
            backend,
            layers: (0..asr.cfg.backbone.n_layer)
                .map(|_| LayerKv::new())
                .collect(),
            position: 0,
            poisoned: false,
            #[cfg(test)]
            fail_after_layer: None,
        })
    }

    /// Resets all KV layers and returns the stream to the source offset zero.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
        self.position = 0;
        self.poisoned = false;
        #[cfg(test)]
        {
            self.fail_after_layer = None;
        }
    }

    /// Whether a failed step has poisoned this state until [`Self::reset`].
    #[must_use]
    pub const fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Next absolute frame position.
    #[must_use]
    pub const fn next_position(&self) -> usize {
        self.position
    }

    /// Absolute positions currently retained by one layer, oldest first.
    #[must_use]
    pub fn layer_cache_positions(&self, layer: usize) -> Option<&[usize]> {
        self.layers
            .get(layer)
            .map(|cache| cache.positions.as_slice())
    }

    /// Executes one audio/text frame.
    ///
    /// `previous_text_token` is the token sampled from the preceding output;
    /// it must be `None` only for the first call.  The first call still
    /// validates and writes the current audio frame conceptually, but source
    /// initial substitution makes the actual embedding rows `text_card` and
    /// `audio_card`.  Later calls use the supplied text token and current
    /// audio codes without a one-frame audio shift.
    pub fn step_frame(
        &mut self,
        previous_text_token: Option<u32>,
        audio_codes: &[u32],
    ) -> Result<KyutaiSttStreamingLmStep> {
        #[cfg(test)]
        {
            self.step_frame_inner(previous_text_token, audio_codes, None)
        }
        #[cfg(not(test))]
        {
            self.step_frame_inner(previous_text_token, audio_codes)
        }
    }

    #[cfg(test)]
    pub(crate) fn step_frame_with_trace(
        &mut self,
        previous_text_token: Option<u32>,
        audio_codes: &[u32],
    ) -> Result<(KyutaiSttStreamingLmStep, KyutaiSttTrace)> {
        let mut trace = KyutaiSttTrace::default();
        let step = self.step_frame_inner(previous_text_token, audio_codes, Some(&mut trace))?;
        Ok((step, trace))
    }

    fn step_frame_inner(
        &mut self,
        previous_text_token: Option<u32>,
        audio_codes: &[u32],
        #[cfg(test)] trace: Option<&mut KyutaiSttTrace>,
    ) -> Result<KyutaiSttStreamingLmStep> {
        if self.poisoned {
            return Err(VokraError::InvalidArgument(
                "kyutai-stt streaming LM is poisoned; call reset before reuse".to_owned(),
            ));
        }
        let cfg = &self.asr.cfg;
        if audio_codes.len() != cfg.n_q {
            return self.poison(format!(
                "kyutai-stt streaming LM audio frame len {} != n_q {}",
                audio_codes.len(),
                cfg.n_q
            ));
        }
        for (index, &code) in audio_codes.iter().enumerate() {
            if code as usize >= cfg.audio_card {
                return self.poison(format!(
                    "kyutai-stt streaming LM audio_codes[{index}]={code} outside [0, {})",
                    cfg.audio_card
                ));
            }
        }
        if self.position == 0 {
            if previous_text_token.is_some() {
                return self.poison(
                    "kyutai-stt streaming LM first frame cannot provide a previous text token"
                        .to_owned(),
                );
            }
        } else {
            let Some(token) = previous_text_token else {
                return self.poison(
                    "kyutai-stt streaming LM subsequent frame requires previous text token"
                        .to_owned(),
                );
            };
            if token as usize >= cfg.text_card {
                return self.poison(format!(
                    "kyutai-stt streaming LM previous text token {token} outside [0, {})",
                    cfg.text_card
                ));
            }
        }
        let compute = match Compute::for_backend(self.backend, KYUTAI_STT_HOT_OPS) {
            Ok(compute) => compute,
            Err(error) => return self.poison_error(error),
        };
        #[cfg(test)]
        let result = self.step_inner(&compute, previous_text_token, audio_codes, trace);
        #[cfg(not(test))]
        let result = self.step_inner(&compute, previous_text_token, audio_codes);
        match result {
            Ok(logits) => {
                let position = self.position;
                let Some(next_position) = self.position.checked_add(1) else {
                    return self
                        .poison("kyutai-stt streaming LM position counter overflow".to_owned());
                };
                self.position = next_position;
                Ok(KyutaiSttStreamingLmStep { position, logits })
            }
            Err(error) => self.poison_error(error),
        }
    }

    fn step_inner(
        &mut self,
        compute: &Compute,
        previous_text_token: Option<u32>,
        audio_codes: &[u32],
        #[cfg(test)] mut trace: Option<&mut KyutaiSttTrace>,
    ) -> Result<KyutaiSttTextLogits> {
        let cfg = &self.asr.cfg;
        let d = cfg.backbone.d_model;
        let text_id = previous_text_token.unwrap_or(cfg.text_card as u32) as usize;
        let mut hidden = vec![0.0_f32; d];
        let text_row = &self.asr.weights.text_embedding[text_id * d..(text_id + 1) * d];
        hidden.copy_from_slice(text_row);
        for (channel, &code) in audio_codes.iter().enumerate() {
            let code = if self.position == 0 {
                cfg.audio_card
            } else {
                code as usize
            };
            let table = &self.asr.weights.audio_embeddings[channel];
            for (dst, &value) in hidden.iter_mut().zip(&table[code * d..(code + 1) * d]) {
                *dst += value;
            }
        }
        ensure_finite("streaming input embedding", &hidden)?;

        for (index, block) in self.asr.weights.blocks.iter().enumerate() {
            #[cfg(test)]
            {
                hidden = forward_layer(
                    compute,
                    cfg,
                    block,
                    &mut self.layers[index],
                    hidden,
                    self.position,
                    trace.as_deref_mut(),
                    index,
                )?;
            }
            #[cfg(not(test))]
            {
                hidden = forward_layer(
                    compute,
                    cfg,
                    block,
                    &mut self.layers[index],
                    hidden,
                    self.position,
                )?;
            }
            #[cfg(test)]
            if self.fail_after_layer == Some(index) {
                return Err(VokraError::ModelLoad(
                    "kyutai-stt streaming LM injected partial-step failure".to_owned(),
                ));
            }
        }
        let mut norm = vec![0.0_f32; d];
        compute.rms_norm_f32(
            &hidden,
            &mut norm,
            1,
            d,
            &self.asr.weights.final_norm,
            cfg.rms_norm_eps,
        )?;
        #[cfg(test)]
        if let Some(trace) = trace.as_mut() {
            trace.record(
                self.asr.cfg.backbone.n_layer,
                self.position,
                None,
                "final_norm",
                &norm,
            );
        }
        ensure_finite("streaming final norm", &norm)?;
        let mut logits = vec![0.0_f32; cfg.text_card];
        compute.gemm_f32(
            1,
            cfg.text_card,
            d,
            &norm,
            &self.asr.weights.text_head,
            None,
            &mut logits,
        )?;
        #[cfg(test)]
        if let Some(trace) = trace.as_mut() {
            trace.record(
                self.asr.cfg.backbone.n_layer,
                self.position,
                None,
                "logits",
                &logits,
            );
        }
        KyutaiSttTextLogits::new(1, cfg.text_card, logits)
    }

    fn poison<T>(&mut self, message: String) -> Result<T> {
        self.poison_error(VokraError::InvalidArgument(message))
    }

    fn poison_error<T>(&mut self, error: VokraError) -> Result<T> {
        for layer in &mut self.layers {
            layer.clear();
        }
        self.position = 0;
        self.poisoned = true;
        Err(error)
    }

    #[cfg(test)]
    fn inject_failure_after_layer(&mut self, layer: usize) {
        self.fail_after_layer = Some(layer);
    }
}

fn ensure_finite(label: &str, values: &[f32]) -> Result<()> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(VokraError::ModelLoad(format!(
            "kyutai-stt streaming LM {label} contains non-finite value at {index}"
        )));
    }
    Ok(())
}

#[cfg_attr(test, allow(clippy::too_many_arguments))]
fn forward_layer(
    compute: &Compute,
    cfg: &KyutaiSttConfig,
    block: &KyutaiSttBlockWeights,
    cache: &mut LayerKv,
    hidden: Vec<f32>,
    position: usize,
    #[cfg(test)] mut trace: Option<&mut KyutaiSttTrace>,
    #[cfg(test)] layer_index: usize,
) -> Result<Vec<f32>> {
    let d = cfg.backbone.d_model;
    let heads = cfg.backbone.n_head;
    let head_dim = cfg.backbone.head_dim();
    let ffn = cfg.backbone.ffn_hidden();
    let mut norm = vec![0.0_f32; d];
    compute.rms_norm_f32(&hidden, &mut norm, 1, d, &block.attn_norm, cfg.rms_norm_eps)?;
    #[cfg(test)]
    if let Some(trace) = trace.as_mut() {
        trace.record(layer_index, position, None, "attn_norm", &norm);
    }
    ensure_finite("attention norm", &norm)?;
    let mut qkv = vec![0.0_f32; 3 * d];
    compute.gemm_f32(1, 3 * d, d, &norm, &block.qkv_proj, None, &mut qkv)?;
    #[cfg(test)]
    if let Some(trace) = trace.as_mut() {
        trace.record(layer_index, position, None, "qkv", &qkv);
    }
    ensure_finite("QKV projection", &qkv)?;
    let mut q = qkv[..d].to_vec();
    let k = qkv[d..2 * d].to_vec();
    let v = qkv[2 * d..].to_vec();
    let inv_freqs = llama3_inv_freqs(head_dim, cfg.backbone.rope_max_period, None)?;
    apply_rope_row(&mut q, heads, head_dim, &inv_freqs, position)?;
    let mut rotated_k = k;
    apply_rope_row(&mut rotated_k, heads, head_dim, &inv_freqs, position)?;
    #[cfg(test)]
    if let Some(trace) = trace.as_mut() {
        trace.record(layer_index, position, None, "q_rope", &q);
        trace.record(layer_index, position, None, "k_rope", &rotated_k);
        trace.record(layer_index, position, None, "v", &v);
    }
    cache.append(position, &rotated_k, &v, cfg.backbone.context);
    let length = cache.positions.len();
    debug_assert!(length > 0 && length <= cfg.backbone.context);
    if cache.positions.iter().any(|&cached| cached > position) {
        return Err(VokraError::ModelLoad(
            "kyutai-stt streaming LM cache contains a future position".to_owned(),
        ));
    }
    #[cfg(test)]
    if let Some(trace) = trace.as_mut() {
        for (row, &cached_position) in cache.positions.iter().enumerate() {
            trace.record(
                layer_index,
                cached_position,
                None,
                "k_cache",
                &cache.keys[row * d..(row + 1) * d],
            );
            trace.record(
                layer_index,
                cached_position,
                None,
                "v_cache",
                &cache.values[row * d..(row + 1) * d],
            );
        }
    }
    let mut attention_input = vec![0.0_f32; d];
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    for head in 0..heads {
        let q_head = &q[head * head_dim..(head + 1) * head_dim];
        let mut keys_t = vec![0.0_f32; head_dim * length];
        let mut values = vec![0.0_f32; length * head_dim];
        for row in 0..length {
            for column in 0..head_dim {
                keys_t[column * length + row] = cache.keys[row * d + head * head_dim + column];
                values[row * head_dim + column] = cache.values[row * d + head * head_dim + column];
            }
        }
        let mut scores = vec![0.0_f32; length];
        compute.gemm_f32(1, length, head_dim, q_head, &keys_t, None, &mut scores)?;
        #[cfg(test)]
        if let Some(trace) = trace.as_mut() {
            trace.record(layer_index, position, Some(head), "qk_raw", &scores);
        }
        for score in &mut scores {
            *score *= scale;
        }
        #[cfg(test)]
        if let Some(trace) = trace.as_mut() {
            trace.record(
                layer_index,
                position,
                Some(head),
                "qk_scaled_masked",
                &scores,
            );
        }
        let mut probabilities = vec![0.0_f32; length];
        compute.softmax_f32(&scores, &mut probabilities, 1, length)?;
        #[cfg(test)]
        if let Some(trace) = trace.as_mut() {
            trace.record(layer_index, position, Some(head), "softmax", &probabilities);
        }
        let mut weighted = vec![0.0_f32; head_dim];
        compute.gemm_f32(
            1,
            head_dim,
            length,
            &probabilities,
            &values,
            None,
            &mut weighted,
        )?;
        #[cfg(test)]
        if let Some(trace) = trace.as_mut() {
            trace.record(layer_index, position, Some(head), "weighted_v", &weighted);
        }
        for column in 0..head_dim {
            attention_input[head * head_dim + column] = weighted[column];
        }
    }
    ensure_finite("attention output", &attention_input)?;
    let mut attention_output = vec![0.0_f32; d];
    compute.gemm_f32(
        1,
        d,
        d,
        &attention_input,
        &block.out_proj,
        None,
        &mut attention_output,
    )?;
    #[cfg(test)]
    if let Some(trace) = trace.as_mut() {
        trace.record(
            layer_index,
            position,
            None,
            "attention_projection",
            &attention_output,
        );
    }
    let mut hidden = hidden;
    for (dst, &value) in hidden.iter_mut().zip(&attention_output) {
        *dst += value;
    }
    #[cfg(test)]
    if let Some(trace) = trace.as_mut() {
        trace.record(layer_index, position, None, "attention_residual", &hidden);
    }
    ensure_finite("attention residual", &hidden)?;

    compute.rms_norm_f32(&hidden, &mut norm, 1, d, &block.ffn_norm, cfg.rms_norm_eps)?;
    #[cfg(test)]
    if let Some(trace) = trace.as_mut() {
        trace.record(layer_index, position, None, "ffn_norm", &norm);
    }
    let mut ffn_input = vec![0.0_f32; 2 * ffn];
    compute.gemm_f32(1, 2 * ffn, d, &norm, &block.linear_in, None, &mut ffn_input)?;
    #[cfg(test)]
    if let Some(trace) = trace.as_mut() {
        trace.record(layer_index, position, None, "ffn_linear_in", &ffn_input);
    }
    ensure_finite("FFN projection", &ffn_input)?;
    let gate_input = ffn_input[..ffn].to_vec();
    let up = &ffn_input[ffn..];
    let mut gate = vec![0.0_f32; ffn];
    compute.silu_f32(&gate_input, &mut gate)?;
    ensure_finite("FFN activation", &gate)?;
    for (gate_value, &up_value) in gate.iter_mut().zip(up) {
        *gate_value *= up_value;
    }
    #[cfg(test)]
    if let Some(trace) = trace.as_mut() {
        trace.record(layer_index, position, None, "ffn_activated", &gate);
    }
    let mut ffn_output = vec![0.0_f32; d];
    compute.gemm_f32(1, d, ffn, &gate, &block.linear_out, None, &mut ffn_output)?;
    #[cfg(test)]
    if let Some(trace) = trace.as_mut() {
        trace.record(layer_index, position, None, "ffn_output", &ffn_output);
    }
    for (dst, &value) in hidden.iter_mut().zip(&ffn_output) {
        *dst += value;
    }
    #[cfg(test)]
    if let Some(trace) = trace.as_mut() {
        trace.record(layer_index, position, None, "layer_output", &hidden);
    }
    ensure_finite("FFN residual", &hidden)?;
    Ok(hidden)
}

fn apply_rope_row(
    values: &mut [f32],
    n_head: usize,
    head_dim: usize,
    inv_freqs: &[f32],
    position: usize,
) -> Result<()> {
    let d_model = n_head.checked_mul(head_dim).ok_or_else(|| {
        VokraError::InvalidArgument("kyutai-stt streaming RoPE shape overflows usize".to_owned())
    })?;
    if values.len() != d_model {
        return Err(VokraError::InvalidArgument(
            "kyutai-stt streaming RoPE row shape is inconsistent with d_model".to_owned(),
        ));
    }
    let mut head = vec![0.0_f32; head_dim];
    for index in 0..n_head {
        let range = index * head_dim..(index + 1) * head_dim;
        head.copy_from_slice(&values[range.clone()]);
        rope_apply_adjacent(&mut head, 1, head_dim, inv_freqs, position)?;
        values[range].copy_from_slice(&head);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kyutai_stt::KyutaiSttWeights;

    fn ordered_f32_bits(value: f32) -> u32 {
        let bits = value.to_bits();
        if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits ^ 0x8000_0000
        }
    }

    fn ulp_distance(left: f32, right: f32) -> u64 {
        u64::from(ordered_f32_bits(left)).abs_diff(u64::from(ordered_f32_bits(right)))
    }

    fn diagnostic_cpu_field(prefixes: &[&str]) -> String {
        let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") else {
            return "<unavailable>".to_owned();
        };
        cpuinfo
            .lines()
            .find_map(|line| {
                prefixes
                    .iter()
                    .find(|prefix| line.starts_with(*prefix))
                    .map(|_| line.trim().to_owned())
            })
            .unwrap_or_else(|| "<unavailable>".to_owned())
    }

    fn report_trace_difference(
        label: &str,
        layer: usize,
        frame: usize,
        head: Option<usize>,
        full: Option<&[f32]>,
        step: Option<&[f32]>,
        first_stage: &mut Option<String>,
    ) {
        if first_stage.is_some() {
            return;
        }
        let (Some(full), Some(step)) = (full, step) else {
            *first_stage = Some(format!("{label} (missing trace entry)"));
            eprintln!(
                "  actual_stage_trace stage={label} layer={layer} frame={frame} head={head:?} missing full={} step={}",
                full.is_some(),
                step.is_some(),
            );
            return;
        };
        let Some(index) = (0..full.len().max(step.len()))
            .find(|&index| full.get(index).copied() != step.get(index).copied())
        else {
            return;
        };
        *first_stage = Some(label.to_owned());
        let full_value = full.get(index).copied();
        let step_value = step.get(index).copied();
        eprintln!(
            "  actual_stage_trace stage={label} layer={layer} frame={frame} head={head:?} index={index} full_len={} step_len={} full={full_value:?} step={step_value:?}",
            full.len(),
            step.len(),
        );
        if let (Some(full_value), Some(step_value)) = (full_value, step_value) {
            eprintln!(
                "    full_bits=0x{:08x} step_bits=0x{:08x} abs_diff={:?} ulps={}",
                full_value.to_bits(),
                step_value.to_bits(),
                (full_value - step_value).abs(),
                ulp_distance(full_value, step_value),
            );
        }
        eprintln!("    full_values={full:?}");
        eprintln!("    step_values={step:?}");
    }

    fn select_positions(values: Option<&[f32]>, positions: &[usize]) -> Option<Vec<f32>> {
        let values = values?;
        positions
            .iter()
            .map(|&position| values.get(position).copied())
            .collect()
    }

    fn compare_actual_stage_trace(
        full_trace: &KyutaiSttTrace,
        step_trace: &KyutaiSttTrace,
        config: &KyutaiSttConfig,
        frame: usize,
    ) {
        let context_start = frame
            .saturating_add(1)
            .saturating_sub(config.backbone.context);
        let retained_positions: Vec<usize> = (context_start..=frame).collect();
        let mut first_stage = None;

        for layer in 0..config.backbone.n_layer {
            for stage in ["attn_norm", "qkv", "q_rope", "k_rope", "v"] {
                report_trace_difference(
                    &format!("layer{layer} {stage}"),
                    layer,
                    frame,
                    None,
                    full_trace.find(layer, frame, None, stage),
                    step_trace.find(layer, frame, None, stage),
                    &mut first_stage,
                );
            }
            for &position in &retained_positions {
                report_trace_difference(
                    &format!("layer{layer} k_cache"),
                    layer,
                    position,
                    None,
                    full_trace.find(layer, position, None, "k_rope"),
                    step_trace.find(layer, position, None, "k_cache"),
                    &mut first_stage,
                );
                report_trace_difference(
                    &format!("layer{layer} v_cache"),
                    layer,
                    position,
                    None,
                    full_trace.find(layer, position, None, "v"),
                    step_trace.find(layer, position, None, "v_cache"),
                    &mut first_stage,
                );
            }
            for head in 0..config.backbone.n_head {
                for stage in ["qk_raw", "qk_scaled_masked", "softmax"] {
                    let full = select_positions(
                        full_trace.find(layer, frame, Some(head), stage),
                        &retained_positions,
                    );
                    report_trace_difference(
                        &format!("layer{layer} head{head} {stage}"),
                        layer,
                        frame,
                        Some(head),
                        full.as_deref(),
                        step_trace.find(layer, frame, Some(head), stage),
                        &mut first_stage,
                    );
                }
                report_trace_difference(
                    &format!("layer{layer} head{head} weighted_v"),
                    layer,
                    frame,
                    Some(head),
                    full_trace.find(layer, frame, Some(head), "weighted_v"),
                    step_trace.find(layer, frame, Some(head), "weighted_v"),
                    &mut first_stage,
                );
            }
            for stage in [
                "attention_projection",
                "attention_residual",
                "ffn_norm",
                "ffn_linear_in",
                "ffn_activated",
                "ffn_output",
                "layer_output",
            ] {
                report_trace_difference(
                    &format!("layer{layer} {stage}"),
                    layer,
                    frame,
                    None,
                    full_trace.find(layer, frame, None, stage),
                    step_trace.find(layer, frame, None, stage),
                    &mut first_stage,
                );
            }
        }
        let final_layer = config.backbone.n_layer;
        for stage in ["final_norm", "logits"] {
            report_trace_difference(
                &format!("final {stage}"),
                final_layer,
                frame,
                None,
                full_trace.find(final_layer, frame, None, stage),
                step_trace.find(final_layer, frame, None, stage),
                &mut first_stage,
            );
        }
        eprintln!(
            "  actual_stage_trace frame={frame} retained_positions={retained_positions:?} first_divergence={first_stage:?}"
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn report_logits_mismatch(
        frame: usize,
        actual: &[f32],
        expected: &[f32],
        stream: &KyutaiSttStreamingLm<'_>,
        config: &KyutaiSttConfig,
        frames: usize,
        asr: &KyutaiSttAsr,
        full_text: &[u32],
        full_audio: &[u32],
        stream_audio: &[u32],
    ) {
        let mismatch = (0..actual.len().max(expected.len()))
            .find(|&index| actual.get(index).copied() != expected.get(index).copied());
        let stream_length = frame.saturating_add(1).min(config.backbone.context);
        let (index, left, right) = mismatch
            .map(|index| {
                (
                    index,
                    actual.get(index).copied(),
                    expected.get(index).copied(),
                )
            })
            .unwrap_or((usize::MAX, None, None));
        eprintln!(
            "kyutai-stt strict streaming/full mismatch: frame={frame} index={index} actual_len={} expected_len={}",
            actual.len(),
            expected.len()
        );
        if let (Some(left), Some(right)) = (left, right) {
            eprintln!(
                "  actual={left:?} bits=0x{:08x}; expected={right:?} bits=0x{:08x}; abs_diff={:?}; ulps={}",
                left.to_bits(),
                right.to_bits(),
                (left - right).abs(),
                ulp_distance(left, right),
            );
        }
        eprintln!(
            "  cpu={} active_isa={:?} VOKRA_CPU_ISA={:?} simd-transcendental=not_recorded_by_runtime; Cargo dependency default intent=enabled",
            diagnostic_cpu_field(&["model name", "Model", "Hardware"]),
            vokra_backend_cpu::active_isa(),
            std::env::var("VOKRA_CPU_ISA").ok(),
        );
        eprintln!(
            "  cpu_flags={} backend={:?} context={} head_dim={} text_card={}",
            diagnostic_cpu_field(&["flags", "Features"]),
            stream.backend,
            config.backbone.context,
            config.backbone.head_dim(),
            config.text_card,
        );
        eprintln!(
            "  candidate stage shapes only (not a first-divergence finding or relaxed verdict): fused QKV full=[m={},n={},k={}] step=[m=1,n={},k={}]; QK full=[m={},n={},k={}] step=[m=1,n={},k={}]; softmax full=[rows={},cols={}] step=[rows=1,cols={}]; weighted-V full=[m={},n={},k={}] step=[m=1,n={},k={}]; final-logits full=[m={},n={},k={}] step=[m=1,n={},k={}]",
            frames,
            3 * config.backbone.d_model,
            config.backbone.d_model,
            3 * config.backbone.d_model,
            config.backbone.d_model,
            frames,
            frames,
            config.backbone.head_dim(),
            stream_length,
            config.backbone.head_dim(),
            frames,
            frames,
            stream_length,
            frames,
            config.backbone.head_dim(),
            frames,
            config.backbone.head_dim(),
            stream_length,
            frames,
            config.text_card,
            config.backbone.d_model,
            config.text_card,
            config.backbone.d_model,
        );
        let cache_positions: Vec<String> = (0..config.backbone.n_layer)
            .map(|layer| {
                format!(
                    "layer{layer}={:?}",
                    stream.layer_cache_positions(layer).unwrap_or(&[])
                )
            })
            .collect();
        eprintln!("  cache_positions={}", cache_positions.join(" "));
        eprintln!(
            "  actual stage taps are copied from the full/step production paths; no diagnostic recomputation or relaxed verdict is used"
        );

        let Ok((full_logits, full_trace)) =
            asr.forward_text_logits_with_trace(BackendKind::Cpu, full_text, full_audio)
        else {
            eprintln!("  actual_stage_trace unavailable: full production trace failed");
            return;
        };
        let traced_expected =
            &full_logits.as_slice()[frame * config.text_card..(frame + 1) * config.text_card];
        if traced_expected != expected {
            let index = (0..traced_expected.len().max(expected.len()))
                .find(|&index| traced_expected.get(index) != expected.get(index));
            eprintln!(
                "  actual_stage_trace replay drift: traced full logits differ from the original full output at index={index:?}; refusing stage interpretation"
            );
            return;
        }
        let mut traced_stream = match asr.streaming_lm(BackendKind::Cpu) {
            Ok(stream) => stream,
            Err(error) => {
                eprintln!("  actual_stage_trace unavailable: stream construction failed: {error}");
                return;
            }
        };
        let zero_audio = vec![0_u32; config.n_q];
        let mut step_trace = None;
        for (replay_frame, &full_text_token) in full_text.iter().enumerate().take(frame + 1) {
            let previous = (replay_frame != 0).then_some(full_text_token);
            let audio_start = replay_frame * config.n_q;
            let current_audio = if replay_frame == 0 {
                zero_audio.as_slice()
            } else {
                &stream_audio[audio_start..audio_start + config.n_q]
            };
            let Ok((step, trace)) = traced_stream.step_frame_with_trace(previous, current_audio)
            else {
                eprintln!(
                    "  actual_stage_trace unavailable: replay failed at frame={replay_frame}"
                );
                return;
            };
            if replay_frame == frame {
                if step.logits().as_slice() != actual {
                    let index = (0..step.logits().as_slice().len().max(actual.len()))
                        .find(|&index| step.logits().as_slice().get(index) != actual.get(index));
                    eprintln!(
                        "  actual_stage_trace replay drift: traced step logits differ from the original streaming output at index={index:?}; refusing stage interpretation"
                    );
                    return;
                }
                step_trace = Some(trace);
            }
        }
        if let Some(step_trace) = step_trace {
            compare_actual_stage_trace(&full_trace, &step_trace, config, frame);
        } else {
            eprintln!("  actual_stage_trace unavailable: no target-frame trace");
        }
    }

    fn fixture() -> (KyutaiSttAsr, KyutaiSttConfig) {
        let mut config = KyutaiSttConfig::tiny_for_tests();
        config.backbone.context = 3;
        let weights = KyutaiSttWeights::synthesized(&config, 0x0517_EA11).expect("weights");
        let asr = KyutaiSttAsr::new(config.clone(), weights).expect("asr");
        (asr, config)
    }

    #[test]
    fn step_matches_full_component_before_and_after_context_boundary() {
        eprintln!(
            "kyutai-stt streaming diagnostic environment: active_isa={:?} VOKRA_CPU_ISA={:?}",
            vokra_backend_cpu::active_isa(),
            std::env::var("VOKRA_CPU_ISA").ok(),
        );
        let (asr, config) = fixture();
        let frames = 8;
        let text: Vec<u32> = (0..frames)
            .map(|index| (index % config.text_card) as u32)
            .collect();
        let audio: Vec<u32> = (0..frames * config.n_q)
            .map(|index| ((index / config.n_q + index % config.n_q + 1) % config.audio_card) as u32)
            .collect();
        let mut full_text = Vec::with_capacity(frames);
        let mut full_audio = Vec::with_capacity(frames * config.n_q);
        full_text.push(config.text_card as u32);
        full_audio.extend(std::iter::repeat_n(config.audio_card as u32, config.n_q));
        for frame in 1..frames {
            full_text.push(text[frame - 1]);
            full_audio.extend_from_slice(&audio[frame * config.n_q..(frame + 1) * config.n_q]);
        }
        let full = asr
            .forward_text_logits(BackendKind::Cpu, &full_text, &full_audio)
            .expect("full component");
        let mut stream = asr.streaming_lm(BackendKind::Cpu).expect("stream");
        for frame in 0..frames {
            let previous = if frame == 0 {
                None
            } else {
                Some(text[frame - 1])
            };
            let step = stream
                .step_frame(
                    previous,
                    &audio[frame * config.n_q..(frame + 1) * config.n_q],
                )
                .expect("stream step");
            assert_eq!(step.position(), frame);
            let expected =
                &full.as_slice()[frame * config.text_card..(frame + 1) * config.text_card];
            if step.logits().as_slice() != expected {
                report_logits_mismatch(
                    frame,
                    step.logits().as_slice(),
                    expected,
                    &stream,
                    &config,
                    frames,
                    &asr,
                    &full_text,
                    &full_audio,
                    &audio,
                );
            }
            assert_eq!(step.logits().as_slice(), expected);
            let expected_start = frame
                .saturating_add(1)
                .saturating_sub(config.backbone.context);
            for layer in 0..config.backbone.n_layer {
                let positions = stream.layer_cache_positions(layer).expect("layer");
                assert_eq!(positions.len(), frame + 1 - expected_start);
                assert_eq!(positions.first().copied(), Some(expected_start));
                assert_eq!(positions.last().copied(), Some(frame));
            }
        }
    }

    /// Synthetic self-consistency only (not upstream numerical parity): CPU
    /// full attention must use the same visible QK and weighted-V geometry as
    /// the incremental cache for short windows, pre-window queries, and
    /// context lengths around the native width boundary.
    #[test]
    fn cpu_visible_attention_stream_self_consistency_context_windows() {
        for context in [1usize, 3, 8, 9, 12] {
            let mut config = KyutaiSttConfig::tiny_for_tests();
            config.backbone.context = context;
            let weights = KyutaiSttWeights::synthesized(&config, 0x0517_EA11).expect("weights");
            let asr = KyutaiSttAsr::new(config.clone(), weights).expect("asr");
            let frames = 11;
            let text: Vec<u32> = (0..frames)
                .map(|frame| (frame % config.text_card) as u32)
                .collect();
            let audio: Vec<u32> = (0..frames * config.n_q)
                .map(|index| {
                    ((index / config.n_q + index % config.n_q + 1) % config.audio_card) as u32
                })
                .collect();
            let mut full_text = Vec::with_capacity(frames);
            let mut full_audio = Vec::with_capacity(frames * config.n_q);
            full_text.push(config.text_card as u32);
            full_audio.extend(std::iter::repeat_n(config.audio_card as u32, config.n_q));
            for frame in 1..frames {
                full_text.push(text[frame - 1]);
                full_audio.extend_from_slice(&audio[frame * config.n_q..(frame + 1) * config.n_q]);
            }
            let text_before = full_text.clone();
            let audio_before = full_audio.clone();
            let full = asr
                .forward_text_logits(BackendKind::Cpu, &full_text, &full_audio)
                .expect("full component");
            assert_eq!(full_text, text_before, "full text input mutated");
            assert_eq!(full_audio, audio_before, "full audio input mutated");
            let mut stream = asr.streaming_lm(BackendKind::Cpu).expect("stream");
            for frame in 0..frames {
                let previous = (frame > 0).then(|| text[frame - 1]);
                let step = stream
                    .step_frame(
                        previous,
                        &audio[frame * config.n_q..(frame + 1) * config.n_q],
                    )
                    .expect("stream step");
                let expected =
                    &full.as_slice()[frame * config.text_card..(frame + 1) * config.text_card];
                assert_eq!(
                    step.logits().as_slice(),
                    expected,
                    "context={context}, frame={frame}"
                );
                let window_start = (frame + 1).saturating_sub(context);
                let positions = stream.layer_cache_positions(0).expect("layer");
                assert_eq!(positions, &(window_start..=frame).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn initial_audio_is_substituted_and_current_audio_is_not_shifted() {
        let (asr, config) = fixture();
        let first = asr
            .streaming_lm(BackendKind::Cpu)
            .expect("stream")
            .step_frame(None, &vec![1; config.n_q])
            .expect("first");
        let mut a = asr.streaming_lm(BackendKind::Cpu).expect("stream");
        let mut b = asr.streaming_lm(BackendKind::Cpu).expect("stream");
        let first_a = a.step_frame(None, &vec![0; config.n_q]).expect("first");
        let first_b = b.step_frame(None, &vec![1; config.n_q]).expect("first");
        assert_eq!(first.logits(), first_a.logits());
        assert_eq!(first_a.logits(), first_b.logits());
        let next_a = a.step_frame(Some(2), &vec![0; config.n_q]).expect("next");
        let next_b = b.step_frame(Some(2), &vec![1; config.n_q]).expect("next");
        assert_ne!(
            next_a.logits(),
            next_b.logits(),
            "current audio must not be shifted"
        );
    }

    #[test]
    fn validation_finite_reset_and_partial_failure_poison_state() {
        let (mut asr, config) = fixture();
        let mut stream = asr.streaming_lm(BackendKind::Cpu).expect("stream");
        assert!(stream.step_frame(None, &[]).is_err());
        assert!(stream.is_poisoned());
        assert!(stream.step_frame(None, &vec![0; config.n_q]).is_err());
        stream.reset();
        assert!(!stream.is_poisoned());
        assert!(
            stream
                .step_frame(None, &vec![config.audio_card as u32; config.n_q])
                .is_err()
        );
        stream.reset();
        assert!(stream.step_frame(None, &vec![0; config.n_q]).is_ok());
        assert!(stream.step_frame(None, &vec![0; config.n_q]).is_err());
        stream.reset();
        assert!(stream.step_frame(None, &vec![0; config.n_q]).is_ok());
        assert!(
            stream
                .step_frame(Some(config.text_card as u32), &vec![0; config.n_q])
                .is_err()
        );
        drop(stream);
        let d = config.backbone.d_model;
        asr.weights.text_embedding[config.text_card * d] = f32::NAN;
        let mut finite_guard = asr.streaming_lm(BackendKind::Cpu).expect("stream");
        assert!(finite_guard.step_frame(None, &vec![0; config.n_q]).is_err());
        assert!(finite_guard.is_poisoned());

        let (asr, config) = fixture();
        let mut stream = asr.streaming_lm(BackendKind::Cpu).expect("stream");
        stream.inject_failure_after_layer(0);
        assert!(stream.step_frame(None, &vec![0; config.n_q]).is_err());
        assert!(stream.is_poisoned());
        assert_eq!(stream.next_position(), 0);
        assert!(stream.layer_cache_positions(0).unwrap().is_empty());
        stream.reset();
        assert!(stream.step_frame(None, &vec![0; config.n_q]).is_ok());
        assert!(
            stream
                .step_frame(Some(config.text_card as u32), &vec![0; config.n_q])
                .is_err()
        );
    }
}
