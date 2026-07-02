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

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use crate::backend::BackendKind;
use crate::engines::{AsrEngine, TtsEngine, VadEngine, VadStreamHandle};
use crate::error::{Result, VokraError};
use crate::gguf::GgufFile;

/// Handle for the loaded model.
///
/// Holds the source path and the parsed GGUF container. The GGUF loader is
/// wired here in **M0-03** (FR-LD-01/02): the weights are lent as zero-copy
/// slices by [`GgufFile`]. An ONNX loader will *never* be wired (FR-LD-05,
/// permanent constraint).
#[derive(Debug)]
pub(crate) struct ModelHandle {
    pub(crate) path: PathBuf,
    pub(crate) gguf: GgufFile,
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
///
/// Native model implementations (`vokra-models`) are attached as trait
/// objects with [`with_asr_engine`](Self::with_asr_engine),
/// [`with_tts_engine`](Self::with_tts_engine) and
/// [`with_vad_engine`](Self::with_vad_engine); the task facades
/// ([`asr`](Self::asr) / [`tts`](Self::tts)) then delegate to them.
pub struct Session {
    pub(crate) inner: Arc<SessionInner>,
    asr: Option<Arc<dyn AsrEngine>>,
    tts: Option<Arc<dyn TtsEngine>>,
    vad: Option<Arc<dyn VadEngine>>,
}

impl fmt::Debug for Session {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Engines are opaque trait objects; report presence, not contents.
        f.debug_struct("Session")
            .field("backend", &self.inner.backend)
            .field("model_path", &self.inner.model.path)
            .field("asr_engine", &self.asr.is_some())
            .field("tts_engine", &self.tts.is_some())
            .field("vad_engine", &self.vad.is_some())
            .finish()
    }
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

    /// The parsed GGUF container backing this session (wired in M0-03).
    ///
    /// Downstream native model implementations (e.g. Whisper base in M0-06)
    /// read tensors from here via [`GgufFile::tensor_data`] and metadata via
    /// [`GgufFile::get`].
    pub fn gguf(&self) -> &GgufFile {
        &self.inner.model.gguf
    }

    /// Attaches an ASR engine (Whisper base = M0-06); consumes and returns the
    /// session so it can be chained after [`with_backend`](SessionBuilder::with_backend).
    #[must_use]
    pub fn with_asr_engine(mut self, engine: Arc<dyn AsrEngine>) -> Self {
        self.asr = Some(engine);
        self
    }

    /// Attaches a TTS engine (piper-plus native TTS = M0-07); see
    /// [`with_asr_engine`](Self::with_asr_engine) (M0-07-T10).
    #[must_use]
    pub fn with_tts_engine(mut self, engine: Arc<dyn TtsEngine>) -> Self {
        self.tts = Some(engine);
        self
    }

    /// Attaches a VAD engine (Silero VAD v5 = M0-05).
    #[must_use]
    pub fn with_vad_engine(mut self, engine: Arc<dyn VadEngine>) -> Self {
        self.vad = Some(engine);
        self
    }

    /// The injected ASR engine, if any (used by the [`Asr`](crate::Asr) facade).
    pub(crate) fn asr_engine(&self) -> Option<&Arc<dyn AsrEngine>> {
        self.asr.as_ref()
    }

    /// The injected TTS engine, if any (used by the [`Tts`](crate::Tts) facade).
    pub(crate) fn tts_engine(&self) -> Option<&Arc<dyn TtsEngine>> {
        self.tts.as_ref()
    }

    /// Opens a streaming VAD handle from the injected VAD engine (M0-05).
    ///
    /// Returns [`VokraError::NotImplemented`] if no VAD engine is attached.
    pub fn open_vad_stream(&self) -> Result<Box<dyn VadStreamHandle + Send>> {
        match &self.vad {
            Some(engine) => Ok(engine.open_stream()),
            None => Err(VokraError::NotImplemented(
                "no VAD engine injected (Silero VAD v5 = M0-05)",
            )),
        }
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

        // Reject non-files early with a clear argument error; a missing path
        // surfaces as `VokraError::Io` from the metadata call.
        let metadata = std::fs::metadata(&self.path)?;
        if !metadata.is_file() {
            return Err(VokraError::InvalidArgument(format!(
                "model path `{}` is not a regular file",
                self.path.display()
            )));
        }

        // M0-03: parse the GGUF container (ONNX loader = never, FR-LD-05).
        // `GgufError` converts into `VokraError::ModelLoad` / `::Io`.
        let gguf = GgufFile::open(&self.path)?;

        Ok(Session {
            inner: Arc::new(SessionInner {
                model: ModelHandle {
                    path: self.path,
                    gguf,
                },
                backend,
                next_stream_id: AtomicU64::new(0),
                active_streams: AtomicU64::new(0),
            }),
            asr: None,
            tts: None,
            vad: None,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Creates a unique **valid minimal GGUF** model file under the OS temp
    /// dir; removed by [`TempModelFile::drop`].
    ///
    /// Now that M0-03 wires real GGUF parsing into [`SessionBuilder::build`],
    /// the fixture must be a parseable GGUF rather than arbitrary bytes.
    pub(crate) struct TempModelFile(pub(crate) PathBuf);

    impl TempModelFile {
        pub(crate) fn new(tag: &str) -> Self {
            let mut path = std::env::temp_dir();
            path.push(format!("vokra-core-test-{tag}-{}", std::process::id()));
            let mut b = crate::gguf::GgufBuilder::new();
            b.add_string("vokra.model.arch", "test");
            b.add_tensor("probe", crate::gguf::GgmlType::F32, vec![1], vec![0u8; 4])
                .expect("valid probe tensor");
            let bytes = b.to_bytes().expect("serialize test gguf");
            std::fs::write(&path, &bytes).expect("write temp model file");
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

    #[test]
    fn gguf_is_loaded_and_accessible() {
        let file = TempModelFile::new("gguf-access");
        let session = Session::from_file(&file.0).build().expect("session builds");
        // Metadata and tensor data are reachable through the loaded container.
        assert_eq!(
            session
                .gguf()
                .get("vokra.model.arch")
                .and_then(|v| v.as_str()),
            Some("test")
        );
        assert_eq!(session.gguf().tensor_data("probe"), Some(&[0u8; 4][..]));
    }

    #[test]
    fn non_gguf_file_is_a_model_load_error() {
        let mut path = std::env::temp_dir();
        path.push(format!("vokra-core-test-nongguf-{}", std::process::id()));
        std::fs::write(&path, b"not a gguf file at all").expect("write junk file");
        let result = Session::from_file(&path).build();
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(VokraError::ModelLoad(_))));
    }
}
