//! Native staged language path for VibeVoice Realtime.
//!
//! The pinned Microsoft source exposes two deliberate forward calls rather
//! than one combined model call: a four-layer text AutoModel (with identity
//! final norm) and a twenty-layer TTS Qwen2 model (with final RMSNorm). This
//! module binds those roles to the existing Qwen2/Compute seam. It stops at
//! hidden states and the EOS logit; diffusion, streaming cache composition,
//! acoustic decode, and waveform parity remain separate gates.

use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{embedding_rows, load_tensor};
use crate::vibevoice::{Qwen2Runtime, Qwen2RuntimeConfig};

use super::HIDDEN;

/// Hot operations used by the EOS classifier in addition to the Qwen2 path.
pub const VIBEVOICE_REALTIME_LANGUAGE_HOT_OPS: &[HotOp] = &[HotOp::Gemm, HotOp::Relu];

/// Result of the staged text LM forward.
#[derive(Debug, Clone, PartialEq)]
pub struct VibeVoiceRealtimeLmOutput {
    /// Row-major hidden states [sequence, hidden].
    pub hidden: Vec<f32>,
}

/// Result of the staged TTS LM forward.
#[derive(Debug, Clone, PartialEq)]
pub struct VibeVoiceRealtimeTtsOutput {
    /// Row-major hidden states [sequence, hidden] after the TTS final norm.
    pub hidden: Vec<f32>,
    /// Raw two-layer classifier output for the last hidden row.
    pub eos_logit: f32,
}

/// An independently cached Realtime TTS language-model branch.
///
/// The official streaming generator keeps positive and negative TTS
/// `past_key_values` separately for classifier-free guidance.  This wrapper
/// makes that ownership explicit: it shares immutable weights with the
/// originating language path, but owns an independent KV cache.  It only
/// returns TTS hidden states and the EOS logit; acoustic sampling and PCM
/// decoding remain outside this language-model boundary.
#[derive(Debug, Clone)]
pub struct VibeVoiceRealtimeTtsBranch {
    backend: BackendKind,
    tts_lm: Qwen2Runtime,
    tts_input_types: Vec<f32>,
    eos_fc1: Dense,
    eos_fc2: Dense,
}

#[derive(Debug, Clone)]
struct Dense {
    /// Column-major [in_features, out_features] for Compute::gemm_f32.
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_features: usize,
    out_features: usize,
}

impl Dense {
    fn from_gguf(
        file: &GgufFile,
        prefix: &str,
        input: usize,
        output: usize,
        label: &str,
    ) -> Result<Self> {
        let raw = load_tensor(file, label, &format!("{prefix}.weight"), &[output, input])?;
        let bias = load_tensor(file, label, &format!("{prefix}.bias"), &[output])?;
        let mut weight = vec![0.0; input * output];
        for row in 0..output {
            for col in 0..input {
                weight[col * output + row] = raw[row * input + col];
            }
        }
        Ok(Self {
            weight,
            bias,
            in_features: input,
            out_features: output,
        })
    }

    fn apply(&self, compute: &Compute, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() || input.len() % self.in_features != 0 {
            return Err(VokraError::InvalidArgument(
                "vibevoice realtime classifier input shape mismatch".to_owned(),
            ));
        }
        let rows = input.len() / self.in_features;
        let mut output = vec![0.0; rows * self.out_features];
        compute.gemm_f32(
            rows,
            self.out_features,
            self.in_features,
            input,
            &self.weight,
            Some(&self.bias),
            &mut output,
        )?;
        finite("vibevoice realtime classifier", &output)?;
        Ok(output)
    }
}

/// Build the exact input matrix consumed by the official TTS LM.
///
/// The base model's embedding lookup is used first, then the last
/// `tail_rows` are replaced by the preceding LM hidden states, and finally
/// the selected text/speech type row is added to every sequence position.
/// Keeping this as a checked helper makes the splice contract testable without
/// loading a checkpoint.
fn prepare_tts_embeddings(
    input_ids: &[u32],
    base_embedding: &[f32],
    vocab_size: usize,
    hidden: usize,
    lm_last_hidden_state: &[f32],
    tts_input_types: &[f32],
    tts_text_masks: &[bool],
) -> Result<Vec<f32>> {
    if input_ids.is_empty() || input_ids.len() > 8_192 || hidden == 0 {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime TTS input sequence is outside limits".to_owned(),
        ));
    }
    if lm_last_hidden_state.is_empty() || lm_last_hidden_state.len() % hidden != 0 {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime TTS splice shape mismatch".to_owned(),
        ));
    }
    let tail_rows = lm_last_hidden_state.len() / hidden;
    if tail_rows > input_ids.len() {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime TTS splice is longer than input sequence".to_owned(),
        ));
    }
    if tts_input_types.len() != 2 * hidden {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime TTS type embedding shape mismatch".to_owned(),
        ));
    }
    if tts_text_masks.is_empty()
        || (tts_text_masks.len() != 1 && tts_text_masks.len() != input_ids.len())
    {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime TTS type-mask shape mismatch".to_owned(),
        ));
    }
    finite("vibevoice realtime TTS base embedding", base_embedding)?;
    finite("vibevoice realtime TTS splice", lm_last_hidden_state)?;
    finite("vibevoice realtime TTS type embedding", tts_input_types)?;
    let mut embeddings = embedding_rows(
        "vibevoice realtime TTS base embedding",
        input_ids,
        base_embedding,
        vocab_size,
        hidden,
    )?;
    let start = (input_ids.len() - tail_rows) * hidden;
    embeddings[start..].copy_from_slice(lm_last_hidden_state);
    for row in 0..input_ids.len() {
        let is_text = if tts_text_masks.len() == 1 {
            tts_text_masks[0]
        } else {
            tts_text_masks[row]
        };
        let type_row = usize::from(is_text);
        let offset = row * hidden;
        for column in 0..hidden {
            embeddings[offset + column] += tts_input_types[type_row * hidden + column];
        }
    }
    finite("vibevoice realtime TTS prepared embedding", &embeddings)?;
    Ok(embeddings)
}

/// Build one cached TTS input row from the official incremental splice.
///
/// `forward_tts_lm` receives one new token embedding when a generation cache
/// is present, replaces that row with the preceding text/acoustic hidden row,
/// and adds the text (`true`) or speech (`false`) type embedding.  Keeping
/// this operation separate from the cache step prevents a caller from
/// accidentally reusing the full-prompt prefill path for an incremental
/// token.
fn prepare_tts_step_embedding(
    input_id: u32,
    base_embedding: &[f32],
    vocab_size: usize,
    hidden: usize,
    lm_last_hidden_state: &[f32],
    tts_input_types: &[f32],
    is_text: bool,
) -> Result<Vec<f32>> {
    if hidden == 0 || lm_last_hidden_state.len() != hidden {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime incremental TTS splice shape mismatch".to_owned(),
        ));
    }
    if tts_input_types.len() != 2 * hidden {
        return Err(VokraError::InvalidArgument(
            "vibevoice realtime incremental TTS type embedding shape mismatch".to_owned(),
        ));
    }
    finite(
        "vibevoice realtime incremental TTS splice",
        lm_last_hidden_state,
    )?;
    finite(
        "vibevoice realtime incremental TTS type embedding",
        tts_input_types,
    )?;
    let mut embedding = embedding_rows(
        "vibevoice realtime incremental TTS base embedding",
        &[input_id],
        base_embedding,
        vocab_size,
        hidden,
    )?;
    embedding.copy_from_slice(lm_last_hidden_state);
    let type_offset = usize::from(is_text) * hidden;
    for (value, type_value) in embedding.iter_mut().zip(&tts_input_types[type_offset..]) {
        *value += type_value;
    }
    finite(
        "vibevoice realtime incremental TTS prepared embedding",
        &embedding,
    )?;
    Ok(embedding)
}

fn apply_eos_classifier(compute: &Compute, fc1: &Dense, fc2: &Dense, input: &[f32]) -> Result<f32> {
    let first = fc1.apply(compute, input)?;
    let mut activated = vec![0.0; first.len()];
    compute.relu_f32(&first, &mut activated)?;
    let second = fc2.apply(compute, &activated)?;
    if second.len() != 1 {
        return Err(VokraError::ModelLoad(
            "vibevoice realtime EOS classifier did not return one logit".to_owned(),
        ));
    }
    Ok(second[0])
}

/// Bound native Realtime text/TTS language stages.
#[derive(Debug, Clone)]
pub struct VibeVoiceRealtimeLanguage {
    backend: BackendKind,
    embedding: Arc<Vec<f32>>,
    text_lm: Qwen2Runtime,
    tts_lm: Qwen2Runtime,
    tts_input_types: Vec<f32>,
    eos_fc1: Dense,
    eos_fc2: Dense,
}

impl VibeVoiceRealtimeLanguage {
    /// Loads both authenticated Qwen2 roles and the official TTS splice heads.
    ///
    /// The TTS stack reuses the base text embedding because that is what
    /// Microsoft's forward_tts_lm calls. Its separate embed_tokens descriptor
    /// is still required by the topology gate but is not redundantly decoded.
    pub fn from_gguf(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        super::VibeVoiceStreamingCheckpoint::from_gguf(file)?;
        let text_lm = Qwen2Runtime::from_gguf_section_with_backend(
            file,
            "model.language_model",
            Qwen2RuntimeConfig::vibevoice_realtime_text(),
            backend,
            false,
        )?;
        let embedding = text_lm.shared_embedding();
        let tts_lm = Qwen2Runtime::from_gguf_section_with_backend_and_embedding(
            file,
            "model.tts_language_model",
            Qwen2RuntimeConfig::vibevoice_realtime_tts(),
            backend,
            true,
            Some(Arc::clone(&embedding)),
        )?;
        let tts_input_types = load_tensor(
            file,
            "vibevoice realtime language",
            "model.tts_input_types.weight",
            &[2, HIDDEN],
        )?;
        let eos_fc1 = Dense::from_gguf(
            file,
            "tts_eos_classifier.fc1",
            HIDDEN,
            HIDDEN,
            "vibevoice realtime EOS classifier",
        )?;
        let eos_fc2 = Dense::from_gguf(
            file,
            "tts_eos_classifier.fc2",
            HIDDEN,
            1,
            "vibevoice realtime EOS classifier",
        )?;
        let _ = Compute::for_backend(backend, VIBEVOICE_REALTIME_LANGUAGE_HOT_OPS)?;
        Ok(Self {
            backend,
            embedding,
            text_lm,
            tts_lm,
            tts_input_types,
            eos_fc1,
            eos_fc2,
        })
    }

    /// Returns the explicitly selected backend; no fallback is performed.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Clears both authenticated language-model caches.
    ///
    /// This is the explicit sequence-boundary operation for callers that
    /// reuse one binder for multiple requests.  It does not alter weights or
    /// the selected backend.
    pub fn reset(&mut self) {
        self.text_lm.reset();
        self.tts_lm.reset();
    }

    /// Runs the four-layer text LM with no final normalization.
    pub fn forward_lm(&mut self, input_ids: &[u32]) -> Result<VibeVoiceRealtimeLmOutput> {
        Ok(VibeVoiceRealtimeLmOutput {
            hidden: self.text_lm.prefill(input_ids)?,
        })
    }

    /// Runs one text token against the existing language-model KV cache.
    ///
    /// This corresponds to the pinned Microsoft `forward_lm` call after
    /// `prepare_inputs_for_generation` has reduced a cached sequence to its
    /// newly appended token.  Unlike [`Self::forward_lm`], it never resets the
    /// cache and therefore must only be used after a valid prompt prefill or
    /// another incremental step.
    pub fn forward_lm_step(&mut self, input_id: u32) -> Result<VibeVoiceRealtimeLmOutput> {
        Ok(VibeVoiceRealtimeLmOutput {
            hidden: self.text_lm.step(input_id)?,
        })
    }

    /// Runs one already-mixed text-LM embedding against the existing cache.
    ///
    /// This is the incremental `inputs_embeds` counterpart of
    /// [`Self::forward_lm_step`].  It is useful for an authenticated caller
    /// that has already performed the text embedding splice; it does not
    /// accept a partial row or silently tokenize arbitrary input.
    pub fn forward_lm_embedding_step(
        &mut self,
        embedding: &[f32],
    ) -> Result<VibeVoiceRealtimeLmOutput> {
        Ok(VibeVoiceRealtimeLmOutput {
            hidden: self.text_lm.step_embedding(embedding)?,
        })
    }

    /// Runs the text LM from already mixed embeddings.
    pub fn forward_lm_embeddings(
        &mut self,
        embeddings: &[f32],
        rows: usize,
    ) -> Result<VibeVoiceRealtimeLmOutput> {
        Ok(VibeVoiceRealtimeLmOutput {
            hidden: self.text_lm.prefill_embeddings(embeddings, rows)?,
        })
    }

    /// Runs the twenty-layer TTS LM and the two-layer ReLU EOS classifier.
    ///
    /// lm_last_hidden_state is row-major [tail_rows, 896] and replaces the
    /// tail of the base embedding sequence. tts_text_masks accepts either one
    /// value (upstream incremental calls) or one value per input row.
    pub fn forward_tts_lm(
        &mut self,
        input_ids: &[u32],
        lm_last_hidden_state: &[f32],
        tts_text_masks: &[bool],
    ) -> Result<VibeVoiceRealtimeTtsOutput> {
        let embeddings = prepare_tts_embeddings(
            input_ids,
            &self.embedding,
            151_936,
            HIDDEN,
            lm_last_hidden_state,
            &self.tts_input_types,
            tts_text_masks,
        )?;
        run_tts_prefill(
            &mut self.tts_lm,
            self.backend,
            &self.eos_fc1,
            &self.eos_fc2,
            &embeddings,
            input_ids.len(),
        )
    }

    /// Runs one incremental TTS LM row against the existing TTS KV cache.
    ///
    /// The hidden row is the preceding text-LM output for a text token or the
    /// acoustic connector output for a speech token.  `is_text` is deliberately
    /// explicit because the official generator uses `true` for streamed text
    /// and `false` for both positive and negative speech branches.
    pub fn forward_tts_lm_step(
        &mut self,
        input_id: u32,
        lm_last_hidden_state: &[f32],
        is_text: bool,
    ) -> Result<VibeVoiceRealtimeTtsOutput> {
        let embedding = prepare_tts_step_embedding(
            input_id,
            &self.embedding,
            151_936,
            HIDDEN,
            lm_last_hidden_state,
            &self.tts_input_types,
            is_text,
        )?;
        run_tts_step(
            &mut self.tts_lm,
            self.backend,
            &self.eos_fc1,
            &self.eos_fc2,
            &embedding,
        )
    }

    /// Forks the current TTS cache into an independent same-prefix branch.
    ///
    /// The returned branch starts with the same prefix state as this language
    /// path, while its subsequent KV appends are independent.  Use this only
    /// when the other CFG branch has the same authenticated prefix.  The
    /// official Realtime negative branch has a distinct prompt, so it must be
    /// built with [`Self::fork_empty_tts_lm_branch`] and
    /// [`VibeVoiceRealtimeTtsBranch::prefill`].
    #[must_use]
    pub fn fork_tts_lm_branch(&self) -> VibeVoiceRealtimeTtsBranch {
        VibeVoiceRealtimeTtsBranch {
            backend: self.backend,
            tts_lm: self.tts_lm.clone(),
            tts_input_types: self.tts_input_types.clone(),
            eos_fc1: self.eos_fc1.clone(),
            eos_fc2: self.eos_fc2.clone(),
        }
    }

    /// Creates an empty TTS branch sharing only immutable authenticated
    /// weights.  Callers must use [`VibeVoiceRealtimeTtsBranch::prefill`] with
    /// that branch's own authenticated prompt before stepping it; no cache
    /// contents are implied by this constructor.
    #[must_use]
    pub fn fork_empty_tts_lm_branch(&self) -> VibeVoiceRealtimeTtsBranch {
        VibeVoiceRealtimeTtsBranch {
            backend: self.backend,
            tts_lm: self.tts_lm.fork_empty_cache(),
            tts_input_types: self.tts_input_types.clone(),
            eos_fc1: self.eos_fc1.clone(),
            eos_fc2: self.eos_fc2.clone(),
        }
    }
}

impl VibeVoiceRealtimeTtsBranch {
    /// Returns the explicitly selected backend; no fallback is performed.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Clears this branch's KV cache and starts a new sequence.
    pub fn reset(&mut self) {
        self.tts_lm.reset();
    }

    /// Prefills this branch with its own authenticated TTS prompt.
    ///
    /// This is the branch counterpart of the pinned Microsoft
    /// `forward_tts_lm` call: token embeddings are built first, the supplied
    /// language/acoustic hidden rows replace the input tail, and the explicit
    /// text/speech type mask is added before the causal cache is populated.
    /// A subsequent [`Self::step`] continues this branch without resetting
    /// that cache.  This method is required for the official negative CFG
    /// prompt, whose prefilled cache is distinct from the positive prompt.
    pub fn prefill(
        &mut self,
        input_ids: &[u32],
        lm_last_hidden_state: &[f32],
        tts_text_masks: &[bool],
    ) -> Result<VibeVoiceRealtimeTtsOutput> {
        let base_embedding = self.tts_lm.shared_embedding();
        let embeddings = prepare_tts_embeddings(
            input_ids,
            base_embedding.as_slice(),
            151_936,
            HIDDEN,
            lm_last_hidden_state,
            &self.tts_input_types,
            tts_text_masks,
        )?;
        run_tts_prefill(
            &mut self.tts_lm,
            self.backend,
            &self.eos_fc1,
            &self.eos_fc2,
            &embeddings,
            input_ids.len(),
        )
    }

    /// Runs one incremental TTS LM row on this independent cache branch.
    pub fn step(
        &mut self,
        input_id: u32,
        lm_last_hidden_state: &[f32],
        is_text: bool,
    ) -> Result<VibeVoiceRealtimeTtsOutput> {
        let base_embedding = self.tts_lm.shared_embedding();
        let embedding = prepare_tts_step_embedding(
            input_id,
            base_embedding.as_slice(),
            151_936,
            HIDDEN,
            lm_last_hidden_state,
            &self.tts_input_types,
            is_text,
        )?;
        run_tts_step(
            &mut self.tts_lm,
            self.backend,
            &self.eos_fc1,
            &self.eos_fc2,
            &embedding,
        )
    }
}

fn run_tts_step(
    tts_lm: &mut Qwen2Runtime,
    backend: BackendKind,
    eos_fc1: &Dense,
    eos_fc2: &Dense,
    embedding: &[f32],
) -> Result<VibeVoiceRealtimeTtsOutput> {
    let hidden = tts_lm.step_embedding(embedding)?;
    let compute = Compute::for_backend(backend, VIBEVOICE_REALTIME_LANGUAGE_HOT_OPS)?;
    let eos_logit = apply_eos_classifier(&compute, eos_fc1, eos_fc2, &hidden)?;
    Ok(VibeVoiceRealtimeTtsOutput { hidden, eos_logit })
}

fn run_tts_prefill(
    tts_lm: &mut Qwen2Runtime,
    backend: BackendKind,
    eos_fc1: &Dense,
    eos_fc2: &Dense,
    embeddings: &[f32],
    rows: usize,
) -> Result<VibeVoiceRealtimeTtsOutput> {
    let hidden = tts_lm.prefill_embeddings(embeddings, rows)?;
    let last = &hidden[(rows - 1) * HIDDEN..];
    let compute = Compute::for_backend(backend, VIBEVOICE_REALTIME_LANGUAGE_HOT_OPS)?;
    let eos_logit = apply_eos_classifier(&compute, eos_fc1, eos_fc2, last)?;
    Ok(VibeVoiceRealtimeTtsOutput { hidden, eos_logit })
}

fn finite(label: &str, values: &[f32]) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::ModelLoad(format!(
            "{label} contains non-finite values"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vibevoice_streaming::connector::VibeVoiceRealtimeAcousticConnector;
    use std::path::{Path, PathBuf};

    #[test]
    fn realtime_axes_are_authenticated() {
        let text = Qwen2RuntimeConfig::vibevoice_realtime_text();
        let tts = Qwen2RuntimeConfig::vibevoice_realtime_tts();
        assert_eq!(text.hidden_size, HIDDEN);
        assert_eq!(text.num_layers, 4);
        assert_eq!(tts.num_layers, 20);
        assert_eq!(text.num_attention_heads, 14);
        assert_eq!(text.num_key_value_heads, 2);
        assert_eq!(text.intermediate_size, 4_864);
        assert_eq!(text.max_position_embeddings, 8_192);
    }

    #[test]
    fn tts_prepare_splices_tail_and_selects_bool_type_rows() {
        let output = prepare_tts_embeddings(
            &[1, 0, 2],
            &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
            3,
            2,
            &[100.0, 101.0, 200.0, 201.0],
            &[1.0, 2.0, 3.0, 4.0],
            &[false, true, false],
        )
        .unwrap();
        assert_eq!(output, [21.0, 23.0, 103.0, 105.0, 201.0, 203.0]);
    }

    #[test]
    fn tts_prepare_accepts_incremental_broadcast_mask() {
        let output = prepare_tts_embeddings(
            &[0, 1],
            &[10.0, 11.0, 20.0, 21.0],
            2,
            2,
            &[30.0, 31.0],
            &[1.0, 2.0, 3.0, 4.0],
            &[true],
        )
        .unwrap();
        assert_eq!(output, [13.0, 15.0, 33.0, 35.0]);
    }

    #[test]
    fn tts_step_replaces_token_embedding_and_selects_text_or_speech_type() {
        let base = [10.0, 11.0, 20.0, 21.0, 30.0, 31.0];
        let types = [1.0, 2.0, 3.0, 4.0];
        let text =
            prepare_tts_step_embedding(1, &base, 3, 2, &[100.0, 200.0], &types, true).unwrap();
        let speech =
            prepare_tts_step_embedding(1, &base, 3, 2, &[100.0, 200.0], &types, false).unwrap();
        assert_eq!(text, [103.0, 204.0]);
        assert_eq!(speech, [101.0, 202.0]);
    }

    #[test]
    fn tts_step_rejects_non_single_hidden_rows_and_non_finite_values() {
        let base = [10.0, 11.0, 20.0, 21.0];
        let types = [1.0, 2.0, 3.0, 4.0];
        assert!(prepare_tts_step_embedding(0, &base, 2, 2, &[1.0], &types, true).is_err());
        assert!(
            prepare_tts_step_embedding(0, &base, 2, 2, &[f32::NAN, 2.0], &types, true,).is_err()
        );
        assert!(
            prepare_tts_step_embedding(0, &base, 2, 2, &[1.0, 2.0], &[1.0, 2.0], true,).is_err()
        );
    }

    #[test]
    fn tts_prepare_rejects_invalid_ids_masks_splice_and_nan() {
        let base = [10.0, 11.0, 20.0, 21.0];
        let types = [1.0, 2.0, 3.0, 4.0];
        assert!(prepare_tts_embeddings(&[2], &base, 2, 2, &[1.0, 2.0], &types, &[true]).is_err());
        assert!(prepare_tts_embeddings(&[0], &base, 2, 2, &[1.0, 2.0], &types, &[]).is_err());
        assert!(prepare_tts_embeddings(&[0, 1], &base, 2, 2, &[1.0], &types, &[true]).is_err());
        assert!(
            prepare_tts_embeddings(&[0], &base, 2, 2, &[1.0, 2.0, 3.0, 4.0], &types, &[true])
                .is_err()
        );
        assert!(
            prepare_tts_embeddings(&[0], &base, 2, 2, &[f32::NAN, 2.0], &types, &[true]).is_err()
        );
    }

    #[test]
    fn eos_classifier_runs_dense_relu_dense_path() {
        let fc1 = Dense {
            weight: vec![1.0, 0.0, 0.0, 1.0],
            bias: vec![-1.0, 1.0],
            in_features: 2,
            out_features: 2,
        };
        let fc2 = Dense {
            weight: vec![2.0, -1.0],
            bias: vec![0.0],
            in_features: 2,
            out_features: 1,
        };
        let compute =
            Compute::for_backend(BackendKind::Cpu, VIBEVOICE_REALTIME_LANGUAGE_HOT_OPS).unwrap();
        // fc1([0, 2]) = [-1, 3], ReLU => [0, 3], fc2 => -3.
        let logit = apply_eos_classifier(&compute, &fc1, &fc2, &[0.0, 2.0]).unwrap();
        assert_eq!(logit, -3.0);
    }

    #[derive(Debug)]
    struct NpyF32 {
        shape: Vec<usize>,
        values: Vec<f32>,
    }

    fn read_npy_f32(path: &Path) -> Result<NpyF32> {
        let bytes = std::fs::read(path).map_err(|error| {
            VokraError::ModelLoad(format!("VAST Realtime reference read {path:?}: {error}"))
        })?;
        if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
            return Err(VokraError::ModelLoad(format!(
                "VAST Realtime reference `{path:?}` is not an NPY file"
            )));
        }
        let version = (bytes[6], bytes[7]);
        let (header_len, header_start) = match version {
            (1, 0) | (2, 0) => {
                let width = if version == (1, 0) { 2 } else { 4 };
                if bytes.len() < 8 + width {
                    return Err(VokraError::ModelLoad(format!(
                        "VAST Realtime reference `{path:?}` has a truncated NPY header"
                    )));
                }
                let length = if width == 2 {
                    u16::from_le_bytes([bytes[8], bytes[9]]) as usize
                } else {
                    u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize
                };
                (length, 8 + width)
            }
            (3, 0) => {
                if bytes.len() < 12 {
                    return Err(VokraError::ModelLoad(format!(
                        "VAST Realtime reference `{path:?}` has a truncated NPY v3 header"
                    )));
                }
                (
                    u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                    12,
                )
            }
            _ => {
                return Err(VokraError::ModelLoad(format!(
                    "VAST Realtime reference `{path:?}` uses unsupported NPY version {version:?}"
                )));
            }
        };
        let header_end = header_start.checked_add(header_len).ok_or_else(|| {
            VokraError::ModelLoad("VAST Realtime reference NPY header length overflow".to_owned())
        })?;
        if header_end > bytes.len() {
            return Err(VokraError::ModelLoad(format!(
                "VAST Realtime reference `{path:?}` has a truncated NPY header"
            )));
        }
        let header = std::str::from_utf8(&bytes[header_start..header_end]).map_err(|error| {
            VokraError::ModelLoad(format!(
                "VAST Realtime reference NPY header is not UTF-8: {error}"
            ))
        })?;
        if !(header.contains("'descr': '<f4'") || header.contains("\"descr\": \"<f4\""))
            || !(header.contains("'fortran_order': False")
                || header.contains("\"fortran_order\": false"))
        {
            return Err(VokraError::ModelLoad(format!(
                "VAST Realtime reference `{path:?}` must be little-endian contiguous f32"
            )));
        }
        let shape_marker = header
            .find("'shape': (")
            .or_else(|| header.find("\"shape\": ("));
        let shape_marker = shape_marker.ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "VAST Realtime reference `{path:?}` has no NPY shape"
            ))
        })?;
        let shape_start = shape_marker
            + header[shape_marker..].find('(').ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "VAST Realtime reference `{path:?}` has an invalid NPY shape"
                ))
            })?
            + 1;
        let shape_end = shape_start
            + header[shape_start..].find(')').ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "VAST Realtime reference `{path:?}` has an invalid NPY shape"
                ))
            })?;
        let shape: Vec<usize> = header[shape_start..shape_end]
            .split(',')
            .map(|item| {
                let item = item.trim();
                if item.is_empty() {
                    Ok(None)
                } else {
                    item.parse::<usize>().map(Some).map_err(|error| {
                        VokraError::ModelLoad(format!(
                            "VAST Realtime reference `{path:?}` has invalid NPY shape: {error}"
                        ))
                    })
                }
            })
            .collect::<Result<Vec<_>>>()
            .map(|items| items.into_iter().flatten().collect())?;
        let element_count = shape.iter().try_fold(1usize, |count, &dimension| {
            count.checked_mul(dimension).ok_or_else(|| {
                VokraError::ModelLoad("VAST Realtime reference NPY shape overflow".to_owned())
            })
        })?;
        if shape.is_empty()
            || element_count
                .checked_mul(4)
                .is_none_or(|size| size != bytes.len() - header_end)
        {
            return Err(VokraError::ModelLoad(format!(
                "VAST Realtime reference `{path:?}` NPY shape/data length mismatch"
            )));
        }
        let values = bytes[header_end..]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Ok(NpyF32 { shape, values })
    }

    fn selected_artifact_sha256(packet: &str, filename: &str) -> Result<String> {
        let path_marker = format!("\"path\": \"{filename}\"");
        let start = packet.rfind(&path_marker).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "VAST Realtime reference packet does not identify selected `{filename}`"
            ))
        })?;
        let tail = &packet[start..];
        let marker = "\"sha256\": \"";
        let hash_start = tail.find(marker).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "VAST Realtime reference packet has no hash for selected `{filename}`"
            ))
        })? + marker.len();
        let hash = tail[hash_start..].split('"').next().unwrap_or_default();
        if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(VokraError::ModelLoad(format!(
                "VAST Realtime reference packet has an invalid hash for `{filename}`"
            )));
        }
        Ok(hash.to_owned())
    }

    fn compare_reference(native: &[f32], reference: &NpyF32, label: &str) -> Result<()> {
        if native.len() != reference.values.len()
            || !reference.values.iter().all(|value| value.is_finite())
        {
            return Err(VokraError::ModelLoad(format!(
                "VAST Realtime {label} reference shape/finiteness mismatch"
            )));
        }
        let mut max_abs = 0.0_f32;
        let mut squared = 0.0_f64;
        for (left, right) in native.iter().zip(&reference.values) {
            if !left.is_finite() {
                return Err(VokraError::ModelLoad(format!(
                    "VAST Realtime {label} native output is non-finite"
                )));
            }
            let difference = (*left - *right).abs();
            max_abs = max_abs.max(difference);
            squared += f64::from(difference) * f64::from(difference);
        }
        let rmse = (squared / native.len() as f64).sqrt();
        eprintln!(
            "VAST Realtime diagnostic {label}: elements={}, max_abs={max_abs:.8e}, rmse={rmse:.8e}",
            native.len()
        );
        Ok(())
    }

    #[test]
    #[ignore = "VAST-only authenticated Realtime GGUF and independent official packet"]
    fn vast_real_weight_staged_language_diagnostic() {
        let gguf_path = std::env::var_os("VOKRA_VIBEVOICE_REALTIME_GGUF")
            .map(PathBuf::from)
            .expect("VOKRA_VIBEVOICE_REALTIME_GGUF must point to the VAST-only GGUF");
        let reference_dir = std::env::var_os("VOKRA_VIBEVOICE_REALTIME_REFERENCE_DIR")
            .map(PathBuf::from)
            .expect("VOKRA_VIBEVOICE_REALTIME_REFERENCE_DIR must point to the official packet");
        let packet_path = reference_dir.join("reference.json");
        let packet = std::fs::read_to_string(&packet_path)
            .expect("VAST official reference packet must be readable");
        for marker in [
            "\"status\": \"REFERENCE_RUN_OPEN_NOT_RUST_PARITY\"",
            "\"execution\": \"official_microsoft_vibevoice_only\"",
            "\"revision\": \"94da20d98b2fa7688e9cbfaf7692ddb4954f7600\"",
            "\"sha256\": \"7758b150b8139deb48ac1ff6f181f745c8fedd5511232fd974b3eb217d83b514\"",
            "\"compute_dtype\": \"float32\"",
            "\"input_ids\": [[1, 2, 3, 4]]",
            "\"tts_text_masks\": [[1, 1, 1, 1]]",
        ] {
            assert!(
                packet.contains(marker),
                "reference identity marker missing: {marker}"
            );
        }
        let filenames = [
            "cpu_lm_last_hidden_state.npy",
            "cpu_tts_last_hidden_state.npy",
            "cpu_eos_logits.npy",
            "cpu_acoustic_connector.npy",
        ];
        let paths: Vec<PathBuf> = filenames
            .iter()
            .map(|name| reference_dir.join(name))
            .collect();
        for (name, path) in filenames.iter().zip(&paths) {
            let expected = selected_artifact_sha256(&packet, name).expect("selected artifact hash");
            let bytes = std::fs::read(path).expect("selected artifact must be readable");
            let actual = hex(&crate::strict_checkpoint::sha256_bytes(&bytes));
            assert_eq!(
                actual, expected,
                "selected artifact hash mismatch for {name}"
            );
        }
        let lm_reference = read_npy_f32(&paths[0]).expect("read official CPU LM NPY");
        let tts_reference = read_npy_f32(&paths[1]).expect("read official CPU TTS NPY");
        let eos_reference = read_npy_f32(&paths[2]).expect("read official CPU EOS NPY");
        let acoustic_reference = read_npy_f32(&paths[3]).expect("read official CPU acoustic NPY");
        assert_eq!(lm_reference.shape, [1, 4, HIDDEN]);
        assert_eq!(tts_reference.shape, [1, 4, HIDDEN]);
        assert_eq!(eos_reference.shape, [1, 1]);
        assert_eq!(acoustic_reference.shape, [1, 1, HIDDEN]);

        let file = GgufFile::open(&gguf_path).expect("open VAST Realtime GGUF");
        let mut language = VibeVoiceRealtimeLanguage::from_gguf(&file, BackendKind::Cpu)
            .expect("authenticated Realtime language binder");
        let lm = language
            .forward_lm(&[1, 2, 3, 4])
            .expect("native text LM forward");
        let tts = language
            .forward_tts_lm(&[1, 2, 3, 4], &lm.hidden, &[true, true, true, true])
            .expect("native TTS LM forward");
        assert_eq!(lm.hidden.len(), 4 * HIDDEN);
        assert_eq!(tts.hidden.len(), 4 * HIDDEN);
        assert!(lm.hidden.iter().all(|value| value.is_finite()));
        assert!(tts.hidden.iter().all(|value| value.is_finite()));
        assert!(tts.eos_logit.is_finite());
        compare_reference(&lm.hidden, &lm_reference, "lm_last_hidden_state")
            .expect("LM diagnostic comparison");
        compare_reference(&tts.hidden, &tts_reference, "tts_last_hidden_state")
            .expect("TTS diagnostic comparison");
        compare_reference(&[tts.eos_logit], &eos_reference, "eos_logits")
            .expect("EOS diagnostic comparison");
        let acoustic = VibeVoiceRealtimeAcousticConnector::from_gguf(&file, BackendKind::Cpu)
            .expect("authenticated Realtime acoustic connector binder");
        let acoustic_input: Vec<f32> = (0..64)
            .map(|index| -1.0 + 2.0 * index as f32 / 63.0)
            .collect();
        let acoustic_output = acoustic
            .forward(&acoustic_input)
            .expect("native acoustic connector forward");
        compare_reference(&acoustic_output, &acoustic_reference, "acoustic_connector")
            .expect("acoustic connector diagnostic comparison");
    }

    fn hex(bytes: &[u8; 32]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}
