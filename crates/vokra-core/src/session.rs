//! Session construction and management (M0-02-T11).
//!
//! FR-API-02 defines the construction chain
//! `Session::from_file(path).with_backend(...)`, which this module provides
//! verbatim.
//!
//! # Thread-safety design note (recorded, not yet enforced)
//!
//! `Session` will be `Send + Sync` (immutable) per FR-API-03 — a **v0.1 MVP
//! requirement**. M0 does not add compile-time assertions (that verification
//! belongs to M1, milestones.md M1-08) but the structure already avoids
//! interior mutability (`RefCell` & co. are not used); shared data lives
//! behind an [`Arc`], and the only mutable bookkeeping uses atomics
//! (see [`crate::stream`]).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use crate::backend::BackendKind;
use crate::error::{Result, VokraError};

/// Placeholder handle for the loaded model (M0-02).
///
/// Only the source path is retained. Real model data arrives when the GGUF
/// loader is wired in **M0-03** (FR-LD-01/02). An ONNX loader will *never*
/// be wired (FR-LD-05, permanent constraint).
#[derive(Debug)]
pub(crate) struct ModelHandle {
    pub(crate) path: PathBuf,
}

/// Shared, immutable core of a [`Session`] (also referenced by
/// [`Stream`](crate::Stream) handles).
#[derive(Debug)]
pub(crate) struct SessionInner {
    pub(crate) model: ModelHandle,
    pub(crate) backend: BackendKind,
    /// Monotonic id source for [`Stream`](crate::Stream) handles.
    pub(crate) next_stream_id: AtomicU64,
    /// Number of currently open streams (M0 session/stream bookkeeping).
    pub(crate) active_streams: AtomicU64,
}

/// An inference session: one model bound to one backend (FR-API-02).
///
/// Created via [`Session::from_file`]:
///
/// ```no_run
/// use vokra_core::{BackendKind, Session};
///
/// let session = Session::from_file("voice.gguf").with_backend(BackendKind::Cpu)?;
/// assert_eq!(session.backend_kind(), BackendKind::Cpu);
/// # Ok::<(), vokra_core::VokraError>(())
/// ```
#[derive(Debug)]
pub struct Session {
    pub(crate) inner: Arc<SessionInner>,
}

impl Session {
    /// Starts building a session from a model file (FR-API-02:
    /// `Session::from_file(path).with_backend(...)`).
    ///
    /// M0-02 performs only a file-existence check when the builder
    /// finishes; GGUF parsing (FR-LD-01/02) is wired in M0-03.
    pub fn from_file(path: impl AsRef<Path>) -> SessionBuilder {
        SessionBuilder {
            path: path.as_ref().to_path_buf(),
            backend: None,
        }
    }

    /// Backend this session was built with.
    pub fn backend_kind(&self) -> BackendKind {
        self.inner.backend
    }

    /// Path of the model file this session was created from.
    pub fn model_path(&self) -> &Path {
        &self.inner.model.path
    }
}

/// Builder returned by [`Session::from_file`] (FR-API-02).
#[derive(Debug)]
pub struct SessionBuilder {
    path: PathBuf,
    backend: Option<BackendKind>,
}

impl SessionBuilder {
    /// Selects the backend and finishes building the session (the
    /// FR-API-02 chain `Session::from_file(path).with_backend(...)`).
    pub fn with_backend(mut self, backend: BackendKind) -> Result<Session> {
        self.backend = Some(backend);
        self.build()
    }

    /// Finishes building without an explicit backend selection; the default
    /// backend is [`BackendKind::Cpu`] (the only M0 backend, FR-BE-01).
    pub fn build(self) -> Result<Session> {
        let backend = self.backend.unwrap_or(BackendKind::Cpu);

        // Model loading is intentionally NOT wired in M0-02: only an
        // existence check plus a placeholder `ModelHandle` (GGUF loader =
        // M0-03; ONNX loader = never, FR-LD-05).
        let metadata = std::fs::metadata(&self.path)?;
        if !metadata.is_file() {
            return Err(VokraError::InvalidArgument(format!(
                "model path `{}` is not a regular file",
                self.path.display()
            )));
        }

        Ok(Session {
            inner: Arc::new(SessionInner {
                model: ModelHandle { path: self.path },
                backend,
                next_stream_id: AtomicU64::new(0),
                active_streams: AtomicU64::new(0),
            }),
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Creates a unique dummy model file under the OS temp dir; removed by
    /// [`TempModelFile::drop`].
    pub(crate) struct TempModelFile(pub(crate) PathBuf);

    impl TempModelFile {
        pub(crate) fn new(tag: &str) -> Self {
            let mut path = std::env::temp_dir();
            path.push(format!("vokra-core-test-{tag}-{}", std::process::id()));
            std::fs::write(&path, b"vokra placeholder model").expect("write temp model file");
            Self(path)
        }
    }

    impl Drop for TempModelFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn missing_file_is_an_io_error() {
        let result =
            Session::from_file("/nonexistent/vokra/model.gguf").with_backend(BackendKind::Cpu);
        assert!(matches!(result, Err(VokraError::Io(_))));
    }

    #[test]
    fn directory_is_rejected_as_invalid_argument() {
        let result = Session::from_file(std::env::temp_dir()).build();
        assert!(matches!(result, Err(VokraError::InvalidArgument(_))));
    }

    #[test]
    fn fr_api_02_chain_compiles_and_builds() {
        let file = TempModelFile::new("chain");
        // FR-API-02 shape, verbatim: Session::from_file(path).with_backend(...)
        let session = Session::from_file(&file.0)
            .with_backend(BackendKind::Cpu)
            .expect("session builds");
        assert_eq!(session.backend_kind(), BackendKind::Cpu);
        assert_eq!(session.model_path(), file.0.as_path());
    }

    #[test]
    fn build_defaults_to_cpu_backend() {
        let file = TempModelFile::new("default-backend");
        let session = Session::from_file(&file.0).build().expect("session builds");
        assert_eq!(session.backend_kind(), BackendKind::Cpu);
    }
}
