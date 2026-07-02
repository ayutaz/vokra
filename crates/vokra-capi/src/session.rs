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

use vokra_core::{BackendKind, Session, VokraError};
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

/// Loads the GGUF at `path`, injects the engine matching its `vokra.model.arch`
/// and returns the ready [`Session`] (ADR-0003 §2).
fn build_session(path: &str) -> Result<Session, VokraError> {
    // CPU backend is the only M0 backend (FR-BE-01); the real kernels are
    // M0-08. A backend selector argument is a future breaking change (note 3).
    let session = Session::from_file(path).with_backend(BackendKind::Cpu)?;

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
            // path. This double-parses the voice GGUF; a shared-GGUF
            // constructor is a spike-level followup (ADR-0003 §2).
            let tts = PiperPlusTts::from_path(path)?;
            Ok(session.with_tts_engine(Arc::new(tts)))
        }
        other => Err(VokraError::InvalidArgument(format!(
            "unsupported model arch `{other}` (M0 supports `{ARCH_WHISPER}` / \
             `{ARCH_SILERO_VAD}` / `{ARCH_PIPER_PLUS}`)"
        ))),
    }
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

/// Frees a session handle from `vokra_session_create_from_file`. `NULL` is a
/// no-op; using the handle after this call is undefined behaviour.
///
/// # Safety
///
/// `session` must be `NULL` or a handle from `vokra_session_create_from_file`
/// that has not already been destroyed.
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
    fn version_is_nul_terminated_and_matches_crate() {
        let ptr = vokra_version();
        assert!(!ptr.is_null());
        // SAFETY: static NUL-terminated string from `vokra_version`.
        let s = unsafe { std::ffi::CStr::from_ptr(ptr) };
        assert_eq!(s.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
    }
}
