//! Strict native HuBERT-Large-LS960 CTC runtime.
//!
//! HuBERT and Wav2Vec2 use different public arch tags and tensor
//! namespaces, but this released fine-tune has the same audited
//! waveform stem, stable/pre-LayerNorm Transformer and CTC topology as
//! the large Wav2Vec2 family. Binding remains HuBERT-specific and exact;
//! only the learned-op implementation is shared.

use std::path::Path;

use vokra_core::Result;
use vokra_core::backend::BackendKind;
use vokra_core::engines::AsrEngine;
use vokra_core::gguf::GgufFile;
use vokra_core::tasks::Transcription;

use crate::compute::HotOp;
use crate::wav2vec2_ctc::{WAV2VEC2_CTC_HOT_OPS, Wav2Vec2Ctc, Wav2Vec2CtcConfig};

/// Public GGUF arch tag.
pub const ARCH: &str = "hubert";

/// Complete learned-op registry for CPU/Metal HuBERT inference.
pub const HUBERT_HOT_OPS: &[HotOp] = WAV2VEC2_CTC_HOT_OPS;

/// Audited `facebook/hubert-large-ls960-ft` encoder plus CTC head.
#[derive(Debug, Clone)]
pub struct HubertCtc {
    inner: Wav2Vec2Ctc,
}

impl HubertCtc {
    /// Opens and strictly binds the public HuBERT GGUF.
    pub fn from_gguf(path: impl AsRef<Path>) -> Result<Self> {
        let file = GgufFile::open(path)?;
        Self::from_file(&file)
    }

    /// Strictly binds an already-open GGUF.
    pub fn from_file(file: &GgufFile) -> Result<Self> {
        Ok(Self {
            inner: Wav2Vec2Ctc::from_hubert_file(file)?,
        })
    }

    /// Selects the backend used by every learned operation.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.inner = self.inner.with_backend(backend);
        self
    }

    /// Resolved fixed HuBERT topology.
    pub fn config(&self) -> &Wav2Vec2CtcConfig {
        self.inner.config()
    }

    /// Selected backend.
    pub fn backend(&self) -> BackendKind {
        self.inner.backend()
    }

    /// Runs the waveform stem and HuBERT encoder.
    pub fn encode_features(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        self.inner.encode_features(pcm)
    }

    /// Runs encoder plus the 32-way CTC head.
    pub fn logits(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        self.inner.logits(pcm)
    }

    /// Runs greedy CTC folding.
    pub fn transcribe_tokens(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        self.inner.transcribe_tokens(pcm)
    }

    /// Runs complete native PCM-to-text inference.
    pub fn transcribe_text(&self, pcm: &[f32]) -> Result<String> {
        self.inner.transcribe_text(pcm)
    }
}

impl AsrEngine for HubertCtc {
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        Ok(Transcription::new(self.transcribe_text(pcm)?))
    }

    fn backend(&self) -> BackendKind {
        self.backend()
    }
}
