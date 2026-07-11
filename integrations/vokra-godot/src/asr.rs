//! Vokra ASR API exposed to Godot (T06). Once T05's `classdb_register_extension_class3`
//! call registers `VokraSession` as a Godot Object, the trampolines
//! constructed here dispatch to the Vokra C ABI on behalf of a GDScript call
//! like `session.transcribe(audio, 16000)`.
//!
//! The GDScript surface (proposal, finalised in T05):
//!
//! ```gdscript
//! # Returns a String; throws VokraError on non-OK status.
//! var text: String = session.transcribe(pcm: PackedFloat32Array, sample_rate: int)
//! ```

use core::ffi::{CStr, c_char};
use core::ptr;

use crate::error::{VokraError, check};
use crate::ffi::capi::{VokraStatus, vokra_asr_transcribe, vokra_string_free};
use crate::session::VokraSession;

/// Transcribe mono f32 PCM. Owns the round-trip through
/// `vokra_asr_transcribe` + `vokra_string_free` — the returned `String` is a
/// fresh Rust allocation, safe to hand across the FFI boundary into a Godot
/// `String`.
///
/// # Non-panic contract
///
/// This function never panics on validation failure — it returns a typed
/// [`VokraError`] instead. The panic firewall in
/// `error::catch_panic_as_err` MUST still wrap the eventual Godot
/// method-binding trampoline (T05) for defense-in-depth.
pub fn transcribe(
    session: &VokraSession,
    pcm: &[f32],
    sample_rate: i32,
) -> Result<String, VokraError> {
    validate_sample_rate(sample_rate)?;

    let (pcm_ptr, num_samples) = if pcm.is_empty() {
        (ptr::null::<f32>(), 0usize)
    } else {
        (pcm.as_ptr(), pcm.len())
    };

    let mut out_text: *mut c_char = ptr::null_mut();
    // SAFETY: `session.as_raw()` yields a live non-null session handle
    // (Session::from_file guarantees non-null; !Clone + Drop pin the
    // lifetime to `&session`). `pcm_ptr` is either `null` when
    // `num_samples == 0` (C ABI contract for empty PCM) or a valid pointer
    // to `num_samples` `f32` samples for the entire call. `&mut out_text`
    // is a writable `*mut *mut c_char` slot; the ABI writes it only on
    // VOKRA_OK.
    let status = unsafe {
        vokra_asr_transcribe(
            session.as_raw(),
            pcm_ptr,
            num_samples,
            sample_rate,
            &mut out_text,
        )
    };
    // Read `vokra_last_error()` on the same thread before doing anything
    // else that could touch the C ABI (ADR-00xx §2 thread contract).
    check(status)?;

    if out_text.is_null() {
        return Err(VokraError {
            status: VokraStatus::Other,
            message: String::from("vokra_asr_transcribe returned VOKRA_OK with a NULL string"),
        });
    }

    // SAFETY: The C ABI guarantees `out_text` is a NUL-terminated UTF-8
    // string owned by Vokra when it wrote `VOKRA_OK`. Copy into an owned
    // Rust `String` before releasing the C ABI allocation.
    let text = unsafe { CStr::from_ptr(out_text) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: `out_text` came from the paired `vokra_asr_transcribe` call
    // on this thread and has not been freed yet.
    unsafe { vokra_string_free(out_text) };

    Ok(text)
}

/// Reject zero / negative sample rates before entering the C ABI so the
/// error surfaces as InvalidArgument regardless of what the underlying
/// session's front-end declares. Extracted so the validation is testable
/// without a live `VokraSession` (which requires a real GGUF, absent from
/// this excluded workspace by design).
fn validate_sample_rate(sample_rate: i32) -> Result<(), VokraError> {
    if sample_rate <= 0 {
        Err(VokraError {
            status: VokraStatus::InvalidArgument,
            message: format!("sample_rate must be > 0, got {sample_rate}"),
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_zero() {
        let err = validate_sample_rate(0).expect_err("zero must be rejected");
        assert_eq!(err.status, VokraStatus::InvalidArgument);
        assert!(err.message.contains("0"));
    }

    #[test]
    fn validate_rejects_negative() {
        let err = validate_sample_rate(-1).expect_err("negative must be rejected");
        assert_eq!(err.status, VokraStatus::InvalidArgument);
        assert!(err.message.contains("-1"));
    }

    #[test]
    fn validate_accepts_typical_rates() {
        // Whisper front-end runs at 16 kHz; Silero also supports 8/16 kHz.
        for &sr in &[8000, 16000, 22050, 24000, 44100, 48000] {
            validate_sample_rate(sr).expect("standard rates must be accepted");
        }
    }
}
