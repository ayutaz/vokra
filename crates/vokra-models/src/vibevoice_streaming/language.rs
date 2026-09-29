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

    /// Runs the four-layer text LM with no final normalization.
    pub fn forward_lm(&mut self, input_ids: &[u32]) -> Result<VibeVoiceRealtimeLmOutput> {
        Ok(VibeVoiceRealtimeLmOutput {
            hidden: self.text_lm.prefill(input_ids)?,
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
        let hidden = self
            .tts_lm
            .prefill_embeddings(&embeddings, input_ids.len())?;
        let last = &hidden[(input_ids.len() - 1) * HIDDEN..];
        let compute = Compute::for_backend(self.backend, VIBEVOICE_REALTIME_LANGUAGE_HOT_OPS)?;
        let eos_logit = apply_eos_classifier(&compute, &self.eos_fc1, &self.eos_fc2, last)?;
        Ok(VibeVoiceRealtimeTtsOutput { hidden, eos_logit })
    }
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
}
