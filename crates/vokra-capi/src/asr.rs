//! ASR transcription and transcript ownership (M0-09-T07).
//!
//! [`vokra_asr_transcribe`] delegates to the injected [`AsrEngine`] via
//! `session.asr().transcribe()` (Whisper base = M0-06) and returns a
//! Vokra-owned UTF-8 transcript, freed with [`vokra_string_free`]. Calling it
//! on a non-ASR model returns `VOKRA_ERROR_NOT_IMPLEMENTED` (ADR-0003 §2).

use std::ffi::{CString, c_char};

use vokra_core::VokraError;
use vokra_core::gguf::FrontendSpec;

use crate::error::vokra_status_t;
use crate::handle::vokra_session_t;
use crate::{error, ffi_guard};

/// Transcribes mono `f32` PCM to text using the session's ASR engine (FR-API-01).
///
/// # Parameters
///
/// - `session`: a session created from a Whisper GGUF.
/// - `pcm` / `num_samples`: mono `f32` samples (may be empty). `pcm` may be
///   `NULL` only when `num_samples == 0`.
/// - `sample_rate`: the PCM sample rate in Hz. It must equal the model's front
///   end sample rate — Vokra does not resample in M0 (FR-OP-04 is M1); a
///   mismatch is `VOKRA_ERROR_INVALID_ARGUMENT`.
/// - `out_text_utf8`: on `VOKRA_OK`, receives a NUL-terminated UTF-8 transcript
///   to be freed with `vokra_string_free`. Untouched on error.
///
/// # Safety
///
/// `session` must be a valid session handle, `pcm` valid for `num_samples`
/// reads (or `NULL` when `num_samples == 0`), and `out_text_utf8` a writable
/// `char*` location.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_asr_transcribe(
    session: *const vokra_session_t,
    pcm: *const f32,
    num_samples: usize,
    sample_rate: i32,
    out_text_utf8: *mut *mut c_char,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: `session` is validated (NULL rejected) by `required_ref`.
        let handle = unsafe { ffi_guard::required_ref(session, "session")? };
        // SAFETY: `pcm`/`num_samples` validated by `required_slice`.
        let samples = unsafe { ffi_guard::required_slice(pcm, num_samples, "pcm")? };
        ffi_guard::require_out_ptr(out_text_utf8, "out_text_utf8")?;

        // The model's expected input rate (bit-exact front end, FR-MD-02).
        let spec = FrontendSpec::from_gguf(handle.session.gguf())
            .map_err(|e| error::fail(&VokraError::from(e)))?;
        if sample_rate <= 0 {
            return Err(error::fail_invalid("`sample_rate` must be positive"));
        }
        if sample_rate as u32 != spec.sample_rate {
            return Err(error::fail_invalid(&format!(
                "input sample_rate {} Hz != model {} Hz (Vokra does not resample in M0; \
                 FR-OP-04 is M1 — resample the input offline)",
                sample_rate, spec.sample_rate
            )));
        }

        let transcription = handle
            .session
            .asr()
            .transcribe(samples)
            .map_err(|e| error::fail(&e))?;
        let cstring = CString::new(transcription.text)
            .map_err(|_| error::fail_invalid("transcript contained an interior NUL byte"))?;
        // SAFETY: `out_text_utf8` is non-null (checked); transfer ownership of
        // the CString buffer to C (freed by `vokra_string_free`).
        unsafe { *out_text_utf8 = cstring.into_raw() };
        Ok(())
    })
}

/// Frees a transcript returned by `vokra_asr_transcribe`. `NULL` is a no-op;
/// double-free is undefined behaviour.
///
/// # Safety
///
/// `s` must be `NULL` or a pointer returned by `vokra_asr_transcribe` that has
/// not already been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_string_free(s: *mut c_char) {
    ffi_guard::guard_void(|| {
        if !s.is_null() {
            // SAFETY: `s` is non-null and, per the contract, a pointer from
            // `CString::into_raw`; reclaim and drop it exactly once.
            drop(unsafe { CString::from_raw(s) });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcribe_rejects_null_session() {
        let mut out: *mut c_char = std::ptr::null_mut();
        // SAFETY: NULL session is the rejected branch; pcm len 0 skips deref.
        let status = unsafe {
            vokra_asr_transcribe(std::ptr::null(), std::ptr::null(), 0, 16_000, &mut out)
        };
        assert_eq!(status, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
        assert!(out.is_null());
        // A descriptive message is available for vokra_last_error().
        assert!(!error::vokra_last_error().is_null());
    }

    #[test]
    fn string_free_null_is_noop() {
        // SAFETY: NULL is an explicit no-op.
        unsafe { vokra_string_free(std::ptr::null_mut()) };
    }

    #[test]
    fn string_free_reclaims_once() {
        let c = CString::new("hello").unwrap();
        let raw = c.into_raw();
        // SAFETY: `raw` is a live CString pointer, freed exactly once.
        unsafe { vokra_string_free(raw) };
    }
}
