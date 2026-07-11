//! Vokra TTS API exposed to Godot (T07). Same dispatch pattern as `asr.rs`,
//! wrapping `vokra_tts_synthesize` + `vokra_audio_free`.
//!
//! GDScript surface (proposal, finalised in T05):
//!
//! ```gdscript
//! # Returns a Dictionary { "pcm": PackedFloat32Array, "sample_rate": int }.
//! # Throws VokraError on non-OK status.
//! var out = session.synthesize(text: String)
//! ```

use core::ffi::c_char;
use core::ptr;
use std::ffi::CString;

use crate::error::{VokraError, check};
use crate::ffi::capi::{VokraStatus, vokra_audio_free, vokra_tts_synthesize};
use crate::session::VokraSession;

/// The output of a TTS call: PCM samples + the sample rate they were
/// generated at.
#[derive(Debug)]
pub struct TtsOutput {
    /// Mono f32 PCM in `[-1, 1]`. Fresh Rust allocation (the C ABI buffer is
    /// already released before this struct is returned).
    pub pcm: Vec<f32>,
    /// Output sample rate in Hz (voice-model-specific).
    pub sample_rate: i32,
}

/// Synthesize speech PCM from UTF-8 text using the session's TTS engine.
pub fn synthesize(session: &VokraSession, text: &str) -> Result<TtsOutput, VokraError> {
    let c_text = CString::new(text).map_err(|_| VokraError {
        status: VokraStatus::InvalidArgument,
        message: String::from("text contains an interior NUL byte"),
    })?;

    let mut out_pcm: *mut f32 = ptr::null_mut();
    let mut out_num: usize = 0;
    let mut out_sr: i32 = 0;

    // SAFETY: `session.as_raw()` is a live handle; `c_text.as_ptr()` is a
    // valid NUL-terminated UTF-8 pointer for the call; the three
    // out-pointers are writable slots. The C ABI writes them only on
    // VOKRA_OK.
    let status = unsafe {
        vokra_tts_synthesize(
            session.as_raw(),
            c_text.as_ptr() as *const c_char,
            &mut out_pcm,
            &mut out_num,
            &mut out_sr,
        )
    };
    check(status)?;

    if out_pcm.is_null() && out_num > 0 {
        // C ABI invariant: `out_pcm` is only NULL when `out_num == 0`.
        // Anything else is a Vokra bug; report defensively.
        return Err(VokraError {
            status: VokraStatus::Other,
            message: String::from(
                "vokra_tts_synthesize returned VOKRA_OK with NULL PCM and non-zero num_samples",
            ),
        });
    }

    // SAFETY: `out_pcm` is either NULL (num_samples == 0, empty slice) or
    // points to `out_num` valid f32 samples. The slice reference lives only
    // until we free the buffer at the end of this function.
    let pcm = if out_pcm.is_null() {
        Vec::new()
    } else {
        let slice = unsafe { core::slice::from_raw_parts(out_pcm, out_num) };
        slice.to_vec()
    };

    // SAFETY: `out_pcm` came from this call's `vokra_tts_synthesize`;
    // pair with the matching `num_samples`.
    unsafe { vokra_audio_free(out_pcm, out_num) };

    Ok(TtsOutput {
        pcm,
        sample_rate: out_sr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesize_rejects_interior_nul() {
        // We can't reach the C ABI without a GGUF, but the CString gate
        // runs BEFORE the ABI. This test drives that gate directly.
        // Rebuild the CString path without a session (bypass by unwrap of
        // the intermediate — this is a unit test on the input validator).
        let text_with_nul = "hello\0world";
        let err = CString::new(text_with_nul).expect_err("interior NUL must fail");
        // The error type just tells us the NUL position; wrapping this in
        // our VokraError happens inside `synthesize`. Assert on the shape.
        assert!(format!("{err}").to_lowercase().contains("nul"));
    }
}
