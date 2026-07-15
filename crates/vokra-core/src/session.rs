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
use crate::engines::{AsrEngine, S2sEngine, TtsEngine, VadEngine, VadStreamHandle};
use crate::error::{Result, VokraError};
use crate::gguf::GgufFile;
use crate::kv_quant::KvQuant;

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
    /// KV cache quantization mode (M3-04, FR-QT-05). Default `Fp32` preserves
    /// pre-M3-04 behaviour for every existing consumer of `Session`.
    pub(crate) kv_quant: KvQuant,
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
    s2s: Option<Arc<dyn S2sEngine>>,
}

/// `Session` is [`Clone`] via cheap atomic `Arc` bumps (FR-API-03): the clone
/// shares the same immutable [`SessionInner`] (and its stream counters) and the
/// same engine trait objects, so it is the Rust-level mechanism behind the C ABI
/// atomic ref count (`vokra_session_retain`). A model stays alive until the last
/// clone is dropped.
impl Clone for Session {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            asr: self.asr.clone(),
            tts: self.tts.clone(),
            vad: self.vad.clone(),
            s2s: self.s2s.clone(),
        }
    }
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
            .field("s2s_engine", &self.s2s.is_some())
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
            kv_quant: KvQuant::default(),
        }
    }

    /// Backend this session was built with.
    pub fn backend_kind(&self) -> BackendKind {
        self.inner.backend
    }

    /// KV cache quantization mode this session was built with (M3-04). Default
    /// [`KvQuant::Fp32`] is bit-identical to the pre-M3-04 behaviour.
    pub fn kv_quant(&self) -> KvQuant {
        self.inner.kv_quant
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

    /// Attaches an S2S dialog engine (Sesame CSM-1B = M4-05; Moshi =
    /// M4-06); the [`S2s`](crate::S2s) facade delegates to it.
    #[must_use]
    pub fn with_s2s_engine(mut self, engine: Arc<dyn S2sEngine>) -> Self {
        self.s2s = Some(engine);
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

    /// The injected S2S engine, if any (used by the [`S2s`](crate::S2s)
    /// facade).
    pub(crate) fn s2s_engine(&self) -> Option<&Arc<dyn S2sEngine>> {
        self.s2s.as_ref()
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
///
/// Chain [`Self::with_backend`] to finish the FR-API-02 verbatim call, or
/// interleave [`Self::with_kv_quant`] before finalising to select the M3-04
/// runtime KV cache quantization mode:
///
/// ```no_run
/// use vokra_core::{BackendKind, KvQuant, Session};
///
/// // FP32 KV cache (default, pre-M3-04 behaviour):
/// let _ = Session::from_file("model.gguf")
///     .with_backend(BackendKind::Cpu)?;
///
/// // Q8_0 KV cache:
/// let session = Session::from_file("model.gguf")
///     .with_kv_quant(KvQuant::Q8_0)
///     .with_backend(BackendKind::Cpu)?;
/// assert_eq!(session.kv_quant(), KvQuant::Q8_0);
/// # Ok::<(), vokra_core::VokraError>(())
/// ```
#[derive(Debug)]
pub struct SessionBuilder {
    path: PathBuf,
    backend: Option<BackendKind>,
    kv_quant: KvQuant,
}

impl SessionBuilder {
    /// Selects the backend and finishes building the session (the
    /// FR-API-02 chain `Session::from_file(path).with_backend(...)`).
    pub fn with_backend(mut self, backend: BackendKind) -> Result<Session> {
        self.backend = Some(backend);
        self.build()
    }

    /// Selects the runtime KV cache quantization mode (M3-04, FR-QT-05).
    ///
    /// Chainable *before* [`Self::with_backend`] / [`Self::build`]; the
    /// default is [`KvQuant::Fp32`] which preserves pre-M3-04 behaviour.
    #[must_use]
    pub fn with_kv_quant(mut self, kv_quant: KvQuant) -> Self {
        self.kv_quant = kv_quant;
        self
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
                kv_quant: self.kv_quant,
                next_stream_id: AtomicU64::new(0),
                active_streams: AtomicU64::new(0),
            }),
            asr: None,
            tts: None,
            vad: None,
            s2s: None,
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
    fn build_defaults_to_fp32_kv_quant() {
        // M3-04: builder default must preserve pre-M3-04 behaviour.
        let file = TempModelFile::new("default-kv-quant");
        let session = Session::from_file(&file.0).build().expect("session builds");
        assert_eq!(session.kv_quant(), KvQuant::Fp32);
    }

    #[test]
    fn with_kv_quant_chains_before_backend() {
        // M3-04: `.with_kv_quant(...)` is chainable and the choice survives
        // to the finished session.
        let file = TempModelFile::new("with-kv-quant-q8");
        let session = Session::from_file(&file.0)
            .with_kv_quant(KvQuant::Q8_0)
            .with_backend(BackendKind::Cpu)
            .expect("session builds");
        assert_eq!(session.kv_quant(), KvQuant::Q8_0);
        assert_eq!(session.backend_kind(), BackendKind::Cpu);
    }

    #[test]
    fn with_kv_quant_all_three_modes_round_trip() {
        // Every non-Fp32 mode reaches the finished session unchanged.
        for mode in [KvQuant::Q4_0, KvQuant::Q5_0, KvQuant::Q8_0] {
            let file = TempModelFile::new(&format!("with-kv-quant-{}", mode.tag()));
            let session = Session::from_file(&file.0)
                .with_kv_quant(mode)
                .build()
                .expect("session builds");
            assert_eq!(session.kv_quant(), mode);
        }
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

    #[test]
    fn clone_shares_inner_counters_and_outlives_the_original() {
        // FR-API-03: Session: Clone is a cheap Arc bump; the clone shares the
        // same SessionInner, so the stream counters stay consistent across
        // handles, and dropping the original keeps the model alive for the clone.
        let file = TempModelFile::new("session-clone");
        let session = Session::from_file(&file.0).build().expect("session builds");
        let clone = session.clone();

        // A stream opened on `session` is visible through `clone` (shared inner).
        let stream = session.open_stream().expect("stream opens");
        assert_eq!(clone.active_stream_count(), 1);

        // Dropping the original leaves the clone (and the loaded model) usable.
        drop(session);
        assert_eq!(clone.model_path(), file.0.as_path());
        assert_eq!(clone.active_stream_count(), 1);
        drop(stream);
        assert_eq!(clone.active_stream_count(), 0);
    }

    #[test]
    fn clone_is_moved_across_threads_and_used_independently() {
        // Session: Send + Sync + Clone — move a clone onto a worker thread and
        // open streams from both, over the shared inner counter.
        let file = TempModelFile::new("session-clone-thread");
        let session = Session::from_file(&file.0).build().expect("session builds");
        let worker_clone = session.clone();
        // Return the Stream (it is Send) so both stay alive simultaneously.
        let handle = std::thread::spawn(move || worker_clone.open_stream().expect("worker stream"));
        let main_stream = session.open_stream().expect("main stream");
        let worker_stream = handle.join().expect("worker joins");
        // Two streams from two threads share one monotonic id source ⇒ distinct,
        // and the shared inner counter reflects both.
        assert_ne!(worker_stream.id(), main_stream.id());
        assert_eq!(session.active_stream_count(), 2);
        drop(worker_stream);
        assert_eq!(session.active_stream_count(), 1);
    }

    #[test]
    fn open_vad_stream_without_engine_is_not_implemented() {
        // The None branch: no VAD engine injected -> explicit NotImplemented,
        // mirroring the ASR/TTS facade fallbacks (never a silent no-op).
        let file = TempModelFile::new("vad-none");
        let session = Session::from_file(&file.0).build().expect("session builds");
        assert!(matches!(
            session.open_vad_stream(),
            Err(VokraError::NotImplemented(_))
        ));
    }

    #[test]
    fn open_vad_stream_delegates_to_injected_engine() {
        // A fake VAD engine whose stream yields canned probabilities, so the
        // Some-branch delegation (with_vad_engine -> open_vad_stream -> handle)
        // is verified in-crate without a real Silero model or fixture GGUF.
        struct FakeVadStream;
        impl VadStreamHandle for FakeVadStream {
            fn push_pcm(&mut self, _pcm: &[f32], _sample_rate: u32) -> Result<Vec<f32>> {
                Ok(vec![0.9, 0.1])
            }
            fn reset(&mut self) {}
        }
        struct FakeVad;
        impl VadEngine for FakeVad {
            fn open_stream(&self) -> Box<dyn VadStreamHandle + Send> {
                Box::new(FakeVadStream)
            }
        }

        let file = TempModelFile::new("vad-fake");
        let session = Session::from_file(&file.0)
            .build()
            .expect("session builds")
            .with_vad_engine(Arc::new(FakeVad));

        let mut handle = session.open_vad_stream().expect("vad stream opens");
        // The canned per-frame probabilities from the fake engine flow back
        // through the boxed handle unchanged.
        assert_eq!(
            handle.push_pcm(&[0.0; 512], 16_000).unwrap(),
            vec![0.9, 0.1]
        );
        // reset() is reachable through the trait object (a no-op on the fake).
        handle.reset();
    }
}
