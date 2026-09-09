//! Standalone runtime front door for the three public BERT-family sidecars.
//!
//! The numerical implementations live in the dependency-only `vokra-bert`
//! crate because SBV2 consumes them internally.  Public GGUFs for plain BERT,
//! DeBERTa v2 and DeBERTa v3 also need a model-level load/execute surface of
//! their own; otherwise a valid sidecar can only be reached indirectly through
//! an SBV2 checkpoint.  This module supplies that surface without duplicating
//! any learned operation.
//!
//! CPU preserves the established scalar oracle. Metal routes every learned
//! projection, attention reduction, softmax, normalization, GELU and (for
//! DeBERTa v2) encoder Conv1D through the existing [`Compute`] kernels. Layout
//! transposes, embedding lookup, residual addition and relative-position index
//! gathering remain host glue. Other backends are rejected explicitly; no
//! scalar CPU forward is labelled as device execution (FR-EX-08).

use vokra_bert::backend::BertBackendOps;
use vokra_bert::bert_base::BertBaseEncoder;
use vokra_bert::deberta_v2::DebertaV2Encoder;
use vokra_bert::deberta_v3::DebertaV3Encoder;
use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{BackendKind, CompliancePolicy, Result, VokraError, check_weight_license};

use crate::compute::{Compute, HotOp};

/// Plain BERT GGUF architecture tag.
pub const ARCH_BERT_BASE: &str = "bert_base";
/// DeBERTa v2 GGUF architecture tag.
pub const ARCH_DEBERTA_V2: &str = "deberta_v2";
/// DeBERTa v3 GGUF architecture tag.
pub const ARCH_DEBERTA_V3: &str = "deberta_v3";

/// Complete learned-op set for plain BERT and DeBERTa v3.
pub const BERT_TRANSFORMER_HOT_OPS: &[HotOp] =
    &[HotOp::Gemm, HotOp::Softmax, HotOp::LayerNorm, HotOp::Gelu];

/// DeBERTa v2 additionally carries its released encoder-input Conv1D.
pub const DEBERTA_V2_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Gelu,
    HotOp::Conv1d,
];

struct ComputeBertBackend<'a> {
    compute: &'a Compute,
}

impl BertBackendOps for ComputeBertBackend<'_> {
    fn linear_f32(
        &self,
        input: &[f32],
        weight_out_in: &[f32],
        bias: Option<&[f32]>,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let mut weight_in_out = vec![0.0; weight_out_in.len()];
        for output_channel in 0..output_dim {
            for input_channel in 0..input_dim {
                weight_in_out[input_channel * output_dim + output_channel] =
                    weight_out_in[output_channel * input_dim + input_channel];
            }
        }
        self.compute.gemm_f32(
            rows,
            output_dim,
            input_dim,
            input,
            &weight_in_out,
            bias,
            output,
        )
    }

    fn softmax_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        self.compute.softmax_f32(input, output, rows, cols)
    }

    fn layer_norm_f32(
        &self,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<()> {
        self.compute
            .layer_norm_f32(input, output, rows, cols, gamma, beta, eps)
    }

    fn gelu_f32(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        self.compute.gelu_f32(input, output)
    }

    fn conv1d_f32(
        &self,
        input: &[f32],
        input_channels: usize,
        input_len: usize,
        weight: &[f32],
        output_channels: usize,
        kernel: usize,
        bias: Option<&[f32]>,
        stride: usize,
        padding: usize,
        output: &mut [f32],
    ) -> Result<()> {
        self.compute.conv1d_f32(
            input,
            input_channels,
            input_len,
            weight,
            output_channels,
            kernel,
            bias,
            stride,
            padding,
            output,
        )
    }
}

/// SBV2's BERT router uses these helpers to share the audited backend adapter
/// without duplicating or exposing the adapter type itself.
pub(crate) fn deberta_v2_forward_with_backend(
    encoder: &DebertaV2Encoder,
    token_ids: &[u32],
    backend: BackendKind,
) -> Result<Vec<f32>> {
    let compute = Compute::for_backend(backend, DEBERTA_V2_HOT_OPS)?;
    encoder.forward_with_backend(&ComputeBertBackend { compute: &compute }, token_ids)
}

pub(crate) fn deberta_v3_forward_with_backend(
    encoder: &DebertaV3Encoder,
    token_ids: &[u32],
    backend: BackendKind,
) -> Result<Vec<f32>> {
    let compute = Compute::for_backend(backend, BERT_TRANSFORMER_HOT_OPS)?;
    encoder.forward_with_backend(&ComputeBertBackend { compute: &compute }, token_ids)
}

pub(crate) fn bert_base_forward_with_backend(
    encoder: &BertBaseEncoder,
    token_ids: &[u32],
    backend: BackendKind,
) -> Result<Vec<f32>> {
    let compute = Compute::for_backend(backend, BERT_TRANSFORMER_HOT_OPS)?;
    encoder.forward_with_backend(&ComputeBertBackend { compute: &compute }, token_ids, None)
}

/// Runtime discriminator read strictly from `vokra.model.arch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BertRuntimeKind {
    /// Plain post-norm BERT with learned absolute positions.
    BertBase,
    /// DeBERTa v2 with disentangled relative-position attention.
    DebertaV2,
    /// DeBERTa v3 with one shared relative-position table.
    DebertaV3,
}

impl BertRuntimeKind {
    /// Return the exact `vokra.model.arch` spelling for this family.
    #[must_use]
    pub const fn arch(self) -> &'static str {
        match self {
            Self::BertBase => ARCH_BERT_BASE,
            Self::DebertaV2 => ARCH_DEBERTA_V2,
            Self::DebertaV3 => ARCH_DEBERTA_V3,
        }
    }
}

enum Encoder {
    BertBase(BertBaseEncoder),
    DebertaV2(DebertaV2Encoder),
    DebertaV3(DebertaV3Encoder),
}

/// A compliance-gated, strictly identified standalone BERT-family encoder.
pub struct BertRuntime {
    kind: BertRuntimeKind,
    encoder: Encoder,
    d_model: usize,
    vocab_size: usize,
    max_positions: Option<usize>,
}

impl BertRuntime {
    /// Bind one of the three public BERT-family GGUF schemas under the strict
    /// runtime licence policy.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        Self::from_gguf_with_policy(file, &CompliancePolicy::strict())
    }

    /// Policy-selectable twin of [`Self::from_gguf`].
    pub fn from_gguf_with_policy(file: &GgufFile, policy: &CompliancePolicy) -> Result<Self> {
        let arch = file
            .get(chunks::KEY_MODEL_ARCH)
            .and_then(|value| value.as_str())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "standalone BERT runtime: GGUF is missing `{}` (expected `{ARCH_BERT_BASE}`, `{ARCH_DEBERTA_V2}`, or `{ARCH_DEBERTA_V3}`)",
                    chunks::KEY_MODEL_ARCH,
                ))
            })?;
        let kind = match arch {
            ARCH_BERT_BASE => BertRuntimeKind::BertBase,
            ARCH_DEBERTA_V2 => BertRuntimeKind::DebertaV2,
            ARCH_DEBERTA_V3 => BertRuntimeKind::DebertaV3,
            other => {
                return Err(VokraError::ModelLoad(format!(
                    "standalone BERT runtime: GGUF arch is `{other}`, expected `{ARCH_BERT_BASE}`, `{ARCH_DEBERTA_V2}`, or `{ARCH_DEBERTA_V3}`; refusing to bind an incompatible text encoder"
                )));
            }
        };

        // Run after the arch gate so a foreign artifact reports the routing
        // mistake before a licence verdict about an unintended model.
        check_weight_license(file, policy)?;

        let require_usize =
            |key: &str| -> Result<usize> {
                let raw = file.get(key).and_then(|value| value.as_u64()).ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "standalone BERT runtime `{arch}`: missing required GGUF metadata `{key}`"
                ))
            })?;
                usize::try_from(raw).map_err(|_| {
                    VokraError::ModelLoad(format!(
                        "standalone BERT runtime `{arch}`: `{key}` value {raw} does not fit usize"
                    ))
                })
            };

        let (encoder, d_model, vocab_size, max_positions) = match kind {
            BertRuntimeKind::BertBase => {
                let vocab = require_usize("vokra.bert_base.vocab")?;
                let max_pos = require_usize("vokra.bert_base.max_pos")?;
                let encoder = BertBaseEncoder::from_gguf(file)?;
                let d_model = encoder.d_model();
                (Encoder::BertBase(encoder), d_model, vocab, Some(max_pos))
            }
            BertRuntimeKind::DebertaV2 => {
                let vocab = require_usize("vokra.bert.deberta_v2.vocab_size")?;
                let encoder = DebertaV2Encoder::from_gguf(file)?;
                let d_model = encoder.get_d_model();
                (Encoder::DebertaV2(encoder), d_model, vocab, None)
            }
            BertRuntimeKind::DebertaV3 => {
                let vocab = require_usize("vokra.bert.deberta_v3.vocab_size")?;
                let encoder = DebertaV3Encoder::from_gguf(file)?;
                let d_model = encoder.get_d_model();
                (Encoder::DebertaV3(encoder), d_model, vocab, None)
            }
        };

        Ok(Self {
            kind,
            encoder,
            d_model,
            vocab_size,
            max_positions,
        })
    }

    /// Execute the final-hidden-state forward as row-major `[T, D]` f32.
    ///
    /// Input validation happens before the dependency encoder, whose legacy
    /// low-level API uses assertions for invalid ids.  The public model-level
    /// surface therefore returns typed errors rather than panicking.
    pub fn encode(&self, token_ids: &[u32], backend: BackendKind) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(VokraError::InvalidArgument(
                "standalone BERT runtime requires at least one token id".to_owned(),
            ));
        }
        if let Some(max_positions) = self.max_positions
            && token_ids.len() > max_positions
        {
            return Err(VokraError::InvalidArgument(format!(
                "standalone BERT runtime `{}` received {} tokens, exceeding the checkpoint max_position_embeddings {max_positions}",
                self.kind.arch(),
                token_ids.len(),
            )));
        }
        if let Some((index, id)) = token_ids
            .iter()
            .copied()
            .enumerate()
            .find(|(_, id)| *id as usize >= self.vocab_size)
        {
            return Err(VokraError::InvalidArgument(format!(
                "standalone BERT runtime `{}` token_ids[{index}]={id} is outside vocab_size {}",
                self.kind.arch(),
                self.vocab_size,
            )));
        }

        let hidden = match backend {
            BackendKind::Cpu => match &self.encoder {
                Encoder::BertBase(encoder) => encoder.forward(token_ids, None),
                Encoder::DebertaV2(encoder) => encoder.forward(token_ids),
                Encoder::DebertaV3(encoder) => encoder.forward(token_ids),
            },
            BackendKind::Metal => {
                let required = match self.kind {
                    BertRuntimeKind::DebertaV2 => DEBERTA_V2_HOT_OPS,
                    BertRuntimeKind::BertBase | BertRuntimeKind::DebertaV3 => {
                        BERT_TRANSFORMER_HOT_OPS
                    }
                };
                let compute = Compute::for_backend(backend, required)?;
                let backend = ComputeBertBackend { compute: &compute };
                match &self.encoder {
                    Encoder::BertBase(encoder) => {
                        encoder.forward_with_backend(&backend, token_ids, None)?
                    }
                    Encoder::DebertaV2(encoder) => {
                        encoder.forward_with_backend(&backend, token_ids)?
                    }
                    Encoder::DebertaV3(encoder) => {
                        encoder.forward_with_backend(&backend, token_ids)?
                    }
                }
            }
            unsupported => {
                return Err(VokraError::UnsupportedOp(format!(
                    "standalone BERT runtime `{}` supports only CPU and Metal, not {unsupported:?}; no silent CPU fallback was performed (FR-EX-08)",
                    self.kind.arch(),
                )));
            }
        };
        let expected = token_ids
            .len()
            .checked_mul(self.d_model)
            .ok_or_else(|| VokraError::InvalidArgument("BERT output shape overflow".to_owned()))?;
        if hidden.len() != expected {
            return Err(VokraError::ModelLoad(format!(
                "standalone BERT runtime `{}` returned {} floats, expected {} tokens × {} hidden = {expected}",
                self.kind.arch(),
                hidden.len(),
                token_ids.len(),
                self.d_model,
            )));
        }
        Ok(hidden)
    }

    #[must_use]
    /// Return the loaded encoder family.
    pub const fn kind(&self) -> BertRuntimeKind {
        self.kind
    }

    #[must_use]
    /// Return the final hidden-state row width.
    pub const fn d_model(&self) -> usize {
        self.d_model
    }

    #[must_use]
    /// Return the checkpoint vocabulary size used for input validation.
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;

    fn parse(builder: GgufBuilder) -> GgufFile {
        GgufFile::parse(builder.to_bytes().expect("build GGUF")).expect("parse GGUF")
    }

    #[test]
    fn missing_arch_is_rejected_before_tensor_binding() {
        let error = BertRuntime::from_gguf(&parse(GgufBuilder::new()))
            .err()
            .expect("missing arch must fail");
        assert!(error.to_string().contains(chunks::KEY_MODEL_ARCH));
    }

    #[test]
    fn foreign_arch_is_rejected_before_licence_or_tensor_binding() {
        let mut builder = GgufBuilder::new();
        builder.add_string(chunks::KEY_MODEL_ARCH, "whisper");
        let error = BertRuntime::from_gguf(&parse(builder))
            .err()
            .expect("foreign arch must fail");
        let message = error.to_string();
        assert!(message.contains("whisper"));
        assert!(message.contains(ARCH_BERT_BASE));
        assert!(message.contains(ARCH_DEBERTA_V2));
        assert!(message.contains(ARCH_DEBERTA_V3));
    }
}
