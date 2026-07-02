//! GGUF tensor access for the piper-plus native TTS (M0-07-T11).
//!
//! Thin typed layer over [`GgufFile`]: fetch a weight by its clean module name
//! (the converter recovered these, M0-07-T07) as an f32 `Vec`, with an optional
//! shape assertion so a wrong-shape tensor fails loudly at load time rather than
//! corrupting a forward pass. The converter widens every weight to F32, so only
//! F32 tensors are expected; an F16 tensor is a converter bug and is rejected.

use vokra_core::gguf::{GgmlType, GgufFile};
use vokra_core::{Result, VokraError};

/// Owns a voice GGUF and lends its tensors as f32 vectors.
pub struct TensorStore {
    file: GgufFile,
}

impl TensorStore {
    /// Wraps an already-parsed voice GGUF.
    pub fn new(file: GgufFile) -> Self {
        Self { file }
    }

    /// The underlying GGUF (for metadata reads).
    pub fn file(&self) -> &GgufFile {
        &self.file
    }

    /// Returns a tensor's dimensions (as stored), or an error if absent.
    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        let info = self.file.tensor_info(name).ok_or_else(|| {
            VokraError::InvalidArgument(format!("piper voice GGUF missing tensor `{name}`"))
        })?;
        Ok(info.dimensions.iter().map(|&d| d as usize).collect())
    }

    /// Loads a tensor as an f32 `Vec` in stored (row-major) order.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] if the tensor is absent or not
    /// F32 (the converter widens F16 → F32; an F16 tensor is a converter bug).
    pub fn tensor(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.file.tensor_info(name).ok_or_else(|| {
            VokraError::InvalidArgument(format!("piper voice GGUF missing tensor `{name}`"))
        })?;
        if info.dtype != GgmlType::F32 {
            return Err(VokraError::InvalidArgument(format!(
                "tensor `{name}` is {:?}, expected F32 (converter should widen)",
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
    pub fn tensor_shaped(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
        let shape = self.shape(name)?;
        if shape != expected {
            return Err(VokraError::InvalidArgument(format!(
                "tensor `{name}` shape {shape:?}, expected {expected:?}"
            )));
        }
        self.tensor(name)
    }
}
