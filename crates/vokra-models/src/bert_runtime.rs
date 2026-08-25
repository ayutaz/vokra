//! Standalone runtime front door for the three public BERT-family sidecars.
//!
//! The numerical implementations live in the dependency-only `vokra-bert`
//! crate because SBV2 consumes them internally.  Public GGUFs for plain BERT,
//! DeBERTa v2 and DeBERTa v3 also need a model-level load/execute surface of
//! their own; otherwise a valid sidecar can only be reached indirectly through
//! an SBV2 checkpoint.  This module supplies that surface without duplicating
//! any learned operation.
//!
//! CPU is the only complete backend today.  [`BertRuntime::encode`] takes an
//! explicit [`BackendKind`] and rejects every non-CPU selection with
//! [`VokraError::UnsupportedOp`].  It never labels a scalar CPU forward as a
//! Metal/CUDA/Vulkan execution (FR-EX-08).

use vokra_bert::bert_base::BertBaseEncoder;
use vokra_bert::deberta_v2::DebertaV2Encoder;
use vokra_bert::deberta_v3::DebertaV3Encoder;
use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{BackendKind, CompliancePolicy, Result, VokraError, check_weight_license};

/// Plain BERT GGUF architecture tag.
pub const ARCH_BERT_BASE: &str = "bert_base";
/// DeBERTa v2 GGUF architecture tag.
pub const ARCH_DEBERTA_V2: &str = "deberta_v2";
/// DeBERTa v3 GGUF architecture tag.
pub const ARCH_DEBERTA_V3: &str = "deberta_v3";

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
        if backend != BackendKind::Cpu {
            return Err(VokraError::UnsupportedOp(format!(
                "standalone BERT runtime `{}` has no complete {backend:?} route: its layer-norm, GELU, softmax and transformer GEMMs currently execute in the scalar `vokra-bert` CPU implementation. Re-run with BackendKind::Cpu; no silent CPU fallback was performed (FR-EX-08)",
                self.kind.arch(),
            )));
        }
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

        let hidden = match &self.encoder {
            Encoder::BertBase(encoder) => encoder.forward(token_ids, None),
            Encoder::DebertaV2(encoder) => encoder.forward(token_ids),
            Encoder::DebertaV3(encoder) => encoder.forward(token_ids),
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
        GgufFile::parse(builder.build().expect("build GGUF")).expect("parse GGUF")
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
