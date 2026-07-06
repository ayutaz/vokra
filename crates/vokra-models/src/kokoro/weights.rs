//! GGUF tensor access for the Kokoro-82M native TTS (M2-07-T09/T10).
//!
//! Thin typed layer over [`GgufFile`]: fetch a weight by its clean module name
//! (verbatim mirror of the upstream safetensors names, recovered by the
//! `vokra-convert::models::kokoro` converter, M2-07-T07) as an f32 `Vec`, with
//! an optional shape assertion so a wrong-shape tensor fails loudly at load time
//! rather than corrupting a forward pass (FR-EX-08).
//!
//! The converter widens every weight to F32; an F16 tensor is therefore a
//! converter bug and is rejected — same rule as the piper-plus store
//! ([`crate::piper_plus::weights`]).

use vokra_core::gguf::{GgmlType, GgufFile};
use vokra_core::{Result, VokraError};

/// Owns a Kokoro voice GGUF and lends its tensors as f32 vectors.
pub(crate) struct TensorStore {
    file: GgufFile,
}

impl TensorStore {
    /// Wraps an already-parsed voice GGUF.
    pub(crate) fn new(file: GgufFile) -> Self {
        Self { file }
    }

    /// The underlying GGUF (for metadata reads).
    pub(crate) fn file(&self) -> &GgufFile {
        &self.file
    }

    /// Returns a tensor's dimensions (as stored), or an error if absent.
    pub(crate) fn shape(&self, name: &str) -> Result<Vec<usize>> {
        let info = self.file.tensor_info(name).ok_or_else(|| {
            VokraError::InvalidArgument(format!("kokoro voice GGUF missing tensor `{name}`"))
        })?;
        Ok(info.dimensions.iter().map(|&d| d as usize).collect())
    }

    /// Loads a tensor as an f32 `Vec` in stored (row-major) order.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] if the tensor is absent or not
    /// F32 (the converter widens F16 → F32; an F16 tensor is a converter bug).
    pub(crate) fn tensor(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.file.tensor_info(name).ok_or_else(|| {
            VokraError::InvalidArgument(format!("kokoro voice GGUF missing tensor `{name}`"))
        })?;
        if info.dtype != GgmlType::F32 {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro tensor `{name}` is {:?}, expected F32 (converter should widen)",
                info.dtype
            )));
        }
        let bytes = self.file.tensor_bytes(info);
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Loads a tensor and asserts its shape equals `expected`.
    #[allow(dead_code)] // consumed by the T12–T17 forward path
    pub(crate) fn tensor_shaped(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
        let shape = self.shape(name)?;
        if shape != expected {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro tensor `{name}` shape {shape:?}, expected {expected:?}"
            )));
        }
        self.tensor(name)
    }
}
