//! Session lifecycle: create from a GGUF, destroy, version (M0-09-T06).
//!
//! [`vokra_session_create_from_file`] loads a GGUF on the CPU backend, detects
//! the model architecture from `vokra.model.arch`, builds the matching native
//! engine (`vokra-models`) and injects it into the [`Session`]. The task
//! facades (`vokra_asr_transcribe` / `vokra_tts_synthesize` / the stream
//! functions) then delegate to that engine; a model/task mismatch surfaces as
//! `VOKRA_ERROR_NOT_IMPLEMENTED` (ADR-0003 §2).

use std::ffi::c_char;
use std::sync::Arc;

use vokra_core::gguf::{AsBytes, GgufFile};
use vokra_core::{BackendKind, CompliancePolicy, Session, VokraError};
use vokra_models::moshi::MoshiEngine;
use vokra_models::piper_plus::PiperPlusTts;
use vokra_models::silero_vad::SileroVadV5;
use vokra_models::whisper::WhisperAsr;

use crate::error::vokra_status_t;
use crate::handle::{self, vokra_session_t};
use crate::{error, ffi_guard};

/// GGUF metadata key holding the model architecture (written by `vokra-convert`).
const KEY_MODEL_ARCH: &str = "vokra.model.arch";

// Architecture strings, matching `crates/vokra-convert/src/models/*.rs`.
const ARCH_WHISPER: &str = "whisper";
const ARCH_SILERO_VAD: &str = "silero-vad";
const ARCH_PIPER_PLUS: &str = "piper-plus-mb-istft-vits2";
const ARCH_MOSHI: &str = "moshi";

/// One model buffer shared between the session's `GgufFile` and the
/// piper-plus engine's second parse (M4-02): a cheap-clone [`AsBytes`] source
/// so `vokra_session_create_from_bytes` never duplicates the (potentially
/// large) model bytes.
#[derive(Clone)]
struct SharedBytes(Arc<Vec<u8>>);

impl AsBytes for SharedBytes {
    fn bytes(&self) -> &[u8] {
        &self.0
    }
}

/// Injects the engine matching the session GGUF's `vokra.model.arch` and
/// returns the ready [`Session`] (ADR-0003 §2). `reparse_for_piper` supplies
/// the by-value `GgufFile` that `PiperPlusTts::from_gguf` consumes, and
/// `load_moshi` supplies the [`MoshiEngine`] built from the raw model bytes
/// (its compliance gate re-reads the whole GGUF image) — each from the file
/// path (`build_session`) or from the shared byte buffer
/// (`build_session_from_bytes`, M4-02).
fn inject_engine(
    session: Session,
    reparse_for_piper: impl FnOnce() -> Result<PiperPlusTts, VokraError>,
    load_moshi: impl FnOnce() -> Result<MoshiEngine, VokraError>,
) -> Result<Session, VokraError> {
    // Own the arch string so the immutable borrow of `session` ends before we
    // move `session` into `with_*_engine` below.
    let arch = session
        .gguf()
        .get(KEY_MODEL_ARCH)
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "GGUF is missing the `{KEY_MODEL_ARCH}` metadata key"
            ))
        })?
        .to_owned();

    match arch.as_str() {
        ARCH_WHISPER => {
            let asr = WhisperAsr::from_gguf(session.gguf())?;
            Ok(session.with_asr_engine(Arc::new(asr)))
        }
        ARCH_SILERO_VAD => {
            let vad = SileroVadV5::from_gguf(session.gguf())?;
            Ok(session.with_vad_engine(Arc::new(vad)))
        }
        ARCH_PIPER_PLUS => {
            // `PiperPlusTts::from_gguf` consumes a `GgufFile`, but the session
            // already owns one (lent by reference only), so re-parse from the
            // source. This double-parses the voice GGUF; a shared-GGUF
            // constructor is a spike-level followup (ADR-0003 §2).
            let tts = reparse_for_piper()?;
            Ok(session.with_tts_engine(Arc::new(tts)))
        }
        ARCH_MOSHI => {
            // Moshi (M4-06, full-duplex S2S — FR-MD-09). The engine wires a
            // default AEC recipe derived from the model's frame hop so
            // `vokra_s2s_duplex_open` runs the canceller out of the box;
            // the batch S2s facade keeps the recorded-input bypass. The
            // attribution surface resolves onto the session for
            // `vokra_model_attribution` (T24).
            let engine = load_moshi()?;
            let sample_rate = engine.mimi_config().sample_rate;
            let hop = engine.mimi_config().frame_hop_samples()?;
            let frame_size = [128usize, 64, 32, 16, 8, 4, 2, 1]
                .into_iter()
                .find(|fs| hop % fs == 0)
                .unwrap_or(1);
            let engine = engine
                .with_aec(
                    &vokra_ops::aec::AecAttrs {
                        sample_rate,
                        frame_size,
                        filter_length: frame_size * 8,
                    },
                    sample_rate as usize,
                )?
                .with_echo_path(vokra_models::csm::EchoPath::BypassRecordedInput);
            let attribution = engine.attribution().cloned();
            let engine = Arc::new(engine);
            let mut session = session
                .with_s2s_engine(engine.clone())
                .with_s2s_duplex_engine(engine);
            if let Some(info) = attribution {
                session = session.with_attribution(info);
            }
            Ok(session)
        }
        other => Err(VokraError::InvalidArgument(format!(
            "unsupported model arch `{other}` (supported: `{ARCH_WHISPER}` / \
             `{ARCH_SILERO_VAD}` / `{ARCH_PIPER_PLUS}` / `{ARCH_MOSHI}`)"
        ))),
    }
}

/// Loads the GGUF at `path`, injects the engine matching its `vokra.model.arch`
/// and returns the ready [`Session`] (ADR-0003 §2).
fn build_session(path: &str) -> Result<Session, VokraError> {
    // CPU backend is the only M0 backend (FR-BE-01); the real kernels are
    // M0-08. A backend selector argument is a future breaking change (note 3).
    //
    // M4 cc-06: the session GGUF opens through the true-mmap loader (lazy
    // page faulting; `GgufError::Io` for a missing path converts to
    // `VokraError::Io` — the same status class the old buffered read gave)
    // instead of a whole-file owned read: the Moshi full-7B GGUF is
    // ~14.3 GiB, which `Session::from_file` used to buffer for the whole
    // session lifetime NEXT TO the engine's own weights. The explicit
    // is-a-file guard mirrors the `SessionBuilder::build` path check.
    let metadata = std::fs::metadata(path)?;
    if !metadata.is_file() {
        return Err(VokraError::InvalidArgument(format!(
            "model path `{path}` is not a regular file"
        )));
    }
    let gguf = vokra_mmap::open_gguf(path)?;
    let session = Session::from_gguf(gguf).with_backend(BackendKind::Cpu)?;
    inject_engine(
        session,
        || PiperPlusTts::from_path(path),
        || MoshiEngine::from_path(path),
    )
}

/// Parses `bytes` as a GGUF, injects the matching engine and returns the ready
/// [`Session`] — the bytes-based twin of [`build_session`] (M4-02).
///
/// The caller-side read replaces every Rust-side filesystem access: on Unity
/// WebGL the std fs syscalls are ABI-skewed under Unity-era Emscripten and
/// StreamingAssets are HTTP-served (ADR M4-02 §2/§3), so C# fetches the bytes
/// (UnityWebRequest / `File.ReadAllBytes`) and hands them over. One shared
/// buffer backs both the session GGUF and the piper-plus re-parse.
fn build_session_from_bytes(bytes: Vec<u8>) -> Result<Session, VokraError> {
    let shared = SharedBytes(Arc::new(bytes));
    let gguf = GgufFile::from_external(Box::new(shared.clone()))?;
    let session = Session::from_gguf(gguf).with_backend(BackendKind::Cpu)?;
    let piper_shared = shared.clone();
    inject_engine(
        session,
        move || {
            let reparsed = GgufFile::from_external(Box::new(piper_shared))?;
            PiperPlusTts::from_gguf(reparsed)
        },
        move || MoshiEngine::from_gguf_with_policy(shared.bytes(), &CompliancePolicy::strict()),
    )
}

/// Loads a model from a GGUF file and creates an inference session on the CPU
/// backend (FR-API-01).
///
/// The architecture is detected from the GGUF's `vokra.model.arch` metadata and
/// the matching native engine is injected: `whisper` → ASR, `silero-vad` → VAD,
/// `piper-plus-mb-istft-vits2` → TTS. ONNX is never loaded (FR-LD-05).
///
/// # Parameters
///
/// - `path_utf8`: NUL-terminated UTF-8 path to the `.gguf` model file.
/// - `out_session`: on `VOKRA_OK`, receives a new session handle to be freed
///   with `vokra_session_destroy`. Untouched on error.
///
/// # Returns
///
/// `VOKRA_OK`, or a non-zero status with the detail available from
/// `vokra_last_error()` (missing file, unparsable GGUF, unknown arch, ...).
///
/// # Safety
///
/// `path_utf8` must be a valid NUL-terminated C string and `out_session` a
/// valid, writable `vokra_session_t*` location.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_session_create_from_file(
    path_utf8: *const c_char,
    out_session: *mut *mut vokra_session_t,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: `path_utf8` is a caller-provided C string (validated / NULL
        // rejected inside `required_str`).
        let path = unsafe { ffi_guard::required_str(path_utf8, "path_utf8")? };
        ffi_guard::require_out_ptr(out_session, "out_session")?;
        let session = build_session(path).map_err(|e| error::fail(&e))?;
        let boxed = handle::into_raw(vokra_session_t { session });
        // SAFETY: `out_session` is non-null (checked) and points at a writable
        // pointer slot per the contract.
        unsafe { *out_session = boxed };
        Ok(())
    })
}

/// Loads a model from an in-memory GGUF buffer and creates an inference
/// session on the CPU backend — the bytes-based twin of
/// `vokra_session_create_from_file` (M4-02, prerelease ABI addition).
///
/// The caller reads / downloads the `.gguf` bytes itself and hands them over;
/// Vokra copies them once and never touches the filesystem. Primary model
/// path on Unity WebGL, where StreamingAssets are HTTP-served (no `fopen`)
/// and Rust-side fs syscalls are ABI-skewed under Unity-era Emscripten
/// (ADR M4-02 §2/§3); valid — and useful — on every other platform too
/// (e.g. Android APK assets without the `persistentDataPath` expansion).
///
/// # Parameters
///
/// - `data`: pointer to the first byte of the GGUF buffer. The buffer is
///   copied before this call returns; the caller may free it immediately
///   afterwards.
/// - `len`: buffer length in bytes. Must be non-zero (an empty buffer is
///   never a valid GGUF and is rejected loudly).
/// - `out_session`: on `VOKRA_OK`, receives a new session handle to be freed
///   with `vokra_session_destroy`. Untouched on error.
///
/// # Returns
///
/// `VOKRA_OK`, or a non-zero status with the detail available from
/// `vokra_last_error()` (NULL/empty buffer, unparsable GGUF, unknown arch, ...).
///
/// # Safety
///
/// `data` must point at `len` valid, initialised bytes for the duration of
/// the call, and `out_session` must be a valid, writable `vokra_session_t*`
/// location.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_session_create_from_bytes(
    data: *const u8,
    len: usize,
    out_session: *mut *mut vokra_session_t,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        if data.is_null() {
            return Err(error::fail(&VokraError::InvalidArgument(
                "data must not be NULL".into(),
            )));
        }
        if len == 0 {
            return Err(error::fail(&VokraError::InvalidArgument(
                "len must be non-zero (an empty buffer is never a valid GGUF)".into(),
            )));
        }
        ffi_guard::require_out_ptr(out_session, "out_session")?;
        // SAFETY: `data` is non-null (checked) and the caller guarantees `len`
        // valid initialised bytes for the duration of the call; the slice is
        // copied into an owned Vec before the call returns.
        let bytes = unsafe { std::slice::from_raw_parts(data, len) }.to_vec();
        let session = build_session_from_bytes(bytes).map_err(|e| error::fail(&e))?;
        let boxed = handle::into_raw(vokra_session_t { session });
        // SAFETY: `out_session` is non-null (checked) and points at a writable
        // pointer slot per the contract.
        unsafe { *out_session = boxed };
        Ok(())
    })
}

/// Retains the session, producing an independent handle that shares the same
/// loaded model via an atomic ref count (FR-API-03).
///
/// This is the C ABI atomic ref count: it clones the inner `Session` (a cheap
/// atomic `Arc` bump), so the model is freed only when the last handle is passed
/// to `vokra_session_destroy`. The new handle is safe to move to another thread
/// (`Session` is `Send + Sync`).
///
/// # Parameters
///
/// - `session`: an existing session handle to retain.
/// - `out_session`: on `VOKRA_OK`, receives a new handle to be freed with
///   `vokra_session_destroy`. Untouched on error.
///
/// # Safety
///
/// `session` must be a valid session handle and `out_session` a writable
/// `vokra_session_t*` location.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_session_retain(
    session: *const vokra_session_t,
    out_session: *mut *mut vokra_session_t,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: `session` validated (NULL rejected) by `required_ref`.
        let handle = unsafe { ffi_guard::required_ref(session, "session")? };
        ffi_guard::require_out_ptr(out_session, "out_session")?;
        let boxed = handle::into_raw(vokra_session_t {
            session: handle.session.clone(),
        });
        // SAFETY: `out_session` is non-null (checked) and writable per contract.
        unsafe { *out_session = boxed };
        Ok(())
    })
}

/// Frees a session handle from `vokra_session_create_from_file` /
/// `vokra_session_retain`. `NULL` is a no-op; using the handle after this call
/// is undefined behaviour. The model is freed when the last handle is destroyed.
///
/// # Safety
///
/// `session` must be `NULL` or a handle from `vokra_session_create_from_file`
/// or `vokra_session_retain` that has not already been destroyed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_session_destroy(session: *mut vokra_session_t) {
    ffi_guard::guard_void(|| {
        // SAFETY: `session` is NULL or a live handle from `into_raw`; `drop_raw`
        // frees it once and treats NULL as a no-op.
        unsafe { handle::drop_raw(session) };
    });
}

/// Returns the Vokra runtime version as a static NUL-terminated UTF-8 string
/// (the `vokra-capi` crate version). The pointer is static — never free it.
#[unsafe(no_mangle)]
pub extern "C" fn vokra_version() -> *const c_char {
    // `concat!` yields a 'static, NUL-terminated &str; its data pointer is
    // valid for the whole program. No allocation, no unsafe.
    concat!(env!("CARGO_PKG_VERSION"), "\0")
        .as_ptr()
        .cast::<c_char>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// The committed both-rate Silero VAD fixture GGUF (M0-05 parity asset).
    fn silero_fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/parity/silero_vad/silero-vad-v5.gguf")
    }

    #[test]
    fn build_session_detects_silero_and_injects_vad() {
        let path = silero_fixture();
        let session = build_session(path.to_str().unwrap()).expect("silero session builds");
        // A VAD engine was injected: opening a stream succeeds.
        assert!(session.open_vad_stream().is_ok());
        // No ASR/TTS engine: the facades report NotImplemented (task mismatch).
        assert!(matches!(
            session.asr().transcribe(&[0.0; 512]),
            Err(VokraError::NotImplemented(_))
        ));
    }

    #[test]
    fn build_session_rejects_missing_arch() {
        // A minimal GGUF with no `vokra.model.arch` key.
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.name", "no-arch");
        let bytes = b.to_bytes().expect("serialize gguf");
        let mut path = std::env::temp_dir();
        path.push(format!("vokra-capi-noarch-{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let result = build_session(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(VokraError::ModelLoad(_))));
    }

    #[test]
    fn build_session_rejects_unsupported_arch() {
        // A GGUF whose `vokra.model.arch` is a valid string but not an M0 model.
        // This is a *different* error class (InvalidArgument) from the missing
        // -arch case (ModelLoad): a present-but-unknown arch is a bad argument.
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "gpt2");
        let bytes = b.to_bytes().expect("serialize gguf");
        let mut path = std::env::temp_dir();
        path.push(format!("vokra-capi-gpt2-{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let result = build_session(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        assert!(matches!(result, Err(VokraError::InvalidArgument(_))));
    }

    #[test]
    fn build_session_from_bytes_detects_silero_and_injects_vad() {
        // M4-02: the bytes-based twin of `build_session` — same arch
        // detection and engine injection, no filesystem access after the
        // caller-side read (Unity WebGL primary path, ADR M4-02 §3).
        let bytes = std::fs::read(silero_fixture()).expect("read silero fixture");
        let session = build_session_from_bytes(bytes).expect("silero session builds from bytes");
        assert!(session.open_vad_stream().is_ok());
        assert!(matches!(
            session.asr().transcribe(&[0.0; 512]),
            Err(VokraError::NotImplemented(_))
        ));
    }

    #[test]
    fn build_session_from_bytes_rejects_missing_arch_and_junk() {
        // Missing `vokra.model.arch` -> ModelLoad (mirror of the file path).
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.name", "no-arch");
        let bytes = b.to_bytes().expect("serialize gguf");
        assert!(matches!(
            build_session_from_bytes(bytes),
            Err(VokraError::ModelLoad(_))
        ));
        // Junk bytes -> ModelLoad from the GGUF parser, never a panic.
        assert!(matches!(
            build_session_from_bytes(b"not a gguf at all".to_vec()),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn build_session_from_bytes_rejects_unsupported_arch() {
        // M4-02: the bytes twin of `build_session_rejects_unsupported_arch`.
        // A present-but-unknown `vokra.model.arch` is a bad argument
        // (InvalidArgument from `inject_engine`'s `other =>` arm), a *different*
        // error class from the missing-arch / junk cases (ModelLoad) covered
        // above — and reached with no filesystem access on this path.
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "gpt2");
        let bytes = b.to_bytes().expect("serialize gguf");
        assert!(matches!(
            build_session_from_bytes(bytes),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn create_from_bytes_loads_silero_end_to_end() {
        // Full extern "C" round trip: bytes in, live handle out, destroy.
        let bytes = std::fs::read(silero_fixture()).expect("read silero fixture");
        let mut session: *mut vokra_session_t = std::ptr::null_mut();
        // SAFETY: `bytes` is a live, initialised buffer of `bytes.len()` bytes
        // and `session` is a writable out-slot.
        let st =
            unsafe { vokra_session_create_from_bytes(bytes.as_ptr(), bytes.len(), &mut session) };
        assert_eq!(st, vokra_status_t::VOKRA_OK);
        assert!(!session.is_null());
        // SAFETY: freshly created handle, destroyed exactly once.
        unsafe { vokra_session_destroy(session) };
    }

    #[test]
    fn create_from_bytes_rejects_null_data_empty_len_and_null_out() {
        // NULL data pointer.
        let mut session: *mut vokra_session_t = std::ptr::null_mut();
        // SAFETY: NULL data is the rejected branch; out slot is writable.
        let st = unsafe { vokra_session_create_from_bytes(std::ptr::null(), 4, &mut session) };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
        assert!(session.is_null(), "out_session untouched on reject path");

        // Zero-length buffer: a 0-byte model is never a valid GGUF; reject
        // loudly instead of parsing an empty slice (FR-EX-08 posture).
        let byte = 0u8;
        // SAFETY: valid pointer with len 0 is the rejected branch.
        let st = unsafe { vokra_session_create_from_bytes(&byte, 0, &mut session) };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
        assert!(session.is_null());

        // NULL out_session.
        let bytes = [0u8; 4];
        // SAFETY: NULL out_session is the rejected branch.
        let st = unsafe {
            vokra_session_create_from_bytes(bytes.as_ptr(), bytes.len(), std::ptr::null_mut())
        };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
    }

    #[test]
    fn create_from_file_rejects_null_path_and_null_out() {
        // NULL path: rejected by `required_str` before anything is built.
        let mut session: *mut vokra_session_t = std::ptr::null_mut();
        // SAFETY: NULL path is the rejected branch; out_session is a writable slot.
        let st = unsafe { vokra_session_create_from_file(std::ptr::null(), &mut session) };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
        assert!(
            session.is_null(),
            "out_session is untouched on the reject path"
        );

        // Valid (UTF-8) path but NULL out_session: rejected by `require_out_ptr`
        // before `build_session`, so the path need not exist.
        let cpath = std::ffi::CString::new("/no/such/vokra/model.gguf").unwrap();
        // SAFETY: valid C path; NULL out_session is the rejected branch.
        let st = unsafe { vokra_session_create_from_file(cpath.as_ptr(), std::ptr::null_mut()) };
        assert_eq!(st, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
    }

    #[test]
    fn destroy_null_is_noop() {
        // Documented: `NULL` is a no-op (must not deref / panic).
        // SAFETY: NULL handle is the explicit no-op branch.
        unsafe { vokra_session_destroy(std::ptr::null_mut()) };
    }

    #[test]
    fn version_is_nul_terminated_and_matches_crate() {
        let ptr = vokra_version();
        assert!(!ptr.is_null());
        // SAFETY: static NUL-terminated string from `vokra_version`.
        let s = unsafe { std::ffi::CStr::from_ptr(ptr) };
        assert_eq!(s.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
    }
}
