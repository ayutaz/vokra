//! `extern "C"` declarations of the Vokra C ABI (`include/vokra.h`, ADR-0003).
//!
//! These are the SAME symbols that `crates/vokra-capi` defines as
//! `#[no_mangle] pub extern "C" fn`, exported from `libvokra.dylib` /
//! `libvokra.so` / `vokra.dll` on the Unity path. In this crate we depend on
//! `vokra-capi` as an rlib (see this crate's `Cargo.toml`), so the linker
//! folds those symbols directly into `libvokra_godot.{dll,dylib,so}`. Every
//! `extern "C"` declaration below therefore resolves at link time — no
//! `dlopen("libvokra.so")` runtime lookup is needed.
//!
//! # Why redeclare instead of re-import from `vokra-capi`?
//!
//! `vokra-capi`'s inner modules (`session`, `asr`, `tts`, `stream`, `error`,
//! `handle`, `ffi_guard`) are private (`mod ...`, not `pub mod ...`) — the
//! crate exposes ONLY its `#[no_mangle]` C symbols, not a Rust API. So we
//! call them via `extern "C"` just like any other C consumer (Unity C# with
//! `[DllImport]`, Godot GDExtension with these declarations). This keeps the
//! contract at the header, not at Rust internals (ADR-00xx §1 handle rules).

use core::ffi::{c_char, c_float};

/// Mirror of `vokra_status_t` from `include/vokra.h`. The numeric values are
/// part of the (M0-unstable, IF-01) ABI and MUST NOT drift; the M3-16 ABI
/// changelog gates any addition.
#[repr(i32)]
#[allow(dead_code)] // Full enum reflected for future error-path exhaustiveness.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VokraStatus {
    Ok = 0,
    Io = 1,
    ModelLoad = 2,
    UnsupportedOp = 3,
    BackendUnavailable = 4,
    InvalidArgument = 5,
    GraphValidation = 6,
    NotImplemented = 7,
    Panic = 8,
    Other = 9,
}

impl VokraStatus {
    /// Decode a raw i32 returned from the C ABI. Unknown positive codes map
    /// to `Other` (forward-compat for new `VokraError` variants during v0.9
    /// window — IF-01).
    pub fn from_raw(code: i32) -> Self {
        match code {
            0 => Self::Ok,
            1 => Self::Io,
            2 => Self::ModelLoad,
            3 => Self::UnsupportedOp,
            4 => Self::BackendUnavailable,
            5 => Self::InvalidArgument,
            6 => Self::GraphValidation,
            7 => Self::NotImplemented,
            8 => Self::Panic,
            _ => Self::Other,
        }
    }
}

// Opaque handle types — matches `include/vokra.h` typedefs. These are ZSTs
// used only for `*mut VokraSession` / `*mut VokraStream` type hygiene; they
// are never constructed on the Rust side.
#[repr(C)]
pub struct VokraSession {
    _private: [u8; 0],
}
#[repr(C)]
pub struct VokraStream {
    _private: [u8; 0],
}

// Mirror of `vokra_event_kind_t` from `include/vokra.h`.
#[repr(i32)]
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VokraEventKind {
    Unknown = 0,
    SpeechProb = 1,
    Token = 2,
}

/// Mirror of `vokra_event_t` — a fixed 12-byte POD (kind + a + b).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct VokraEvent {
    pub kind: VokraEventKind,
    pub a: u32,
    pub b: c_float,
}

// ---------------------------------------------------------------------------
// C ABI symbol declarations — signatures MUST match `include/vokra.h`.
// The Rust `extern "C"` block below is `unsafe` because these are raw C
// entry points; every call site adds a `// SAFETY:` comment (workspace lint).
// ---------------------------------------------------------------------------

unsafe extern "C" {
    /// Version string (static NUL-terminated UTF-8). Owned by Vokra.
    pub safe fn vokra_version() -> *const c_char;

    /// Thread-local last error string. May return NULL when no error has
    /// been recorded on this thread.
    pub safe fn vokra_last_error() -> *const c_char;

    /// Loads a GGUF and creates a session (CPU backend on M0/M1).
    /// Return: `VokraStatus` cast to i32.
    pub fn vokra_session_create_from_file(
        path_utf8: *const c_char,
        out_session: *mut *mut VokraSession,
    ) -> i32;

    /// Retain: atomic refcount bump (FR-API-03).
    pub fn vokra_session_retain(
        session: *const VokraSession,
        out_session: *mut *mut VokraSession,
    ) -> i32;

    /// Free a session handle.
    pub fn vokra_session_destroy(session: *mut VokraSession);

    /// Transcribe mono f32 PCM. `out_text_utf8` receives a Vokra-owned
    /// NUL-terminated UTF-8 string, freed with `vokra_string_free`.
    pub fn vokra_asr_transcribe(
        session: *const VokraSession,
        pcm: *const c_float,
        num_samples: usize,
        sample_rate: i32,
        out_text_utf8: *mut *mut c_char,
    ) -> i32;

    /// Free an ASR transcript string.
    pub fn vokra_string_free(s: *mut c_char);

    /// Synthesize speech PCM from UTF-8 text.
    pub fn vokra_tts_synthesize(
        session: *const VokraSession,
        text_utf8: *const c_char,
        out_pcm: *mut *mut c_float,
        out_num_samples: *mut usize,
        out_sample_rate: *mut i32,
    ) -> i32;

    /// Free a TTS PCM buffer.
    pub fn vokra_audio_free(pcm: *mut c_float, num_samples: usize);

    /// Open a VAD stream over a Silero session.
    pub fn vokra_stream_open(
        session: *const VokraSession,
        sample_rate: i32,
        out_stream: *mut *mut VokraStream,
    ) -> i32;

    /// Push mono f32 PCM into a VAD stream.
    pub fn vokra_stream_push_pcm(
        stream: *mut VokraStream,
        pcm: *const c_float,
        num_samples: usize,
    ) -> i32;

    /// Drain speech probabilities (fast path over the SPSC ring).
    pub fn vokra_stream_poll(
        stream: *mut VokraStream,
        out_probs: *mut c_float,
        capacity: usize,
        out_count: *mut usize,
    ) -> i32;

    /// Drain typed events (M1-08).
    pub fn vokra_stream_poll_events(
        stream: *mut VokraStream,
        out_events: *mut VokraEvent,
        capacity: usize,
        out_count: *mut usize,
    ) -> i32;

    /// Barge-in: synchronous flush + hidden-state reset (M3-14 / FR-ST-03).
    pub fn vokra_stream_interrupt(stream: *mut VokraStream) -> i32;

    /// Free a stream handle. NULL is a no-op.
    pub fn vokra_stream_destroy(stream: *mut VokraStream);
}

/// Referenced by `lib.rs::LINKER_KEEPALIVE` — takes the address of every C ABI
/// symbol above so the Rust linker cannot dead-code-strip them out of the
/// cdylib. `#[no_mangle]` alone is NOT sufficient when the symbol is defined
/// in an rlib dependency (vokra-capi) but never called on any live path from
/// the cdylib's own code (as is the case at this initial-stub milestone —
/// class registration doesn't dispatch to these until T05+). Taking the
/// address forces the linker to keep them; `black_box` prevents further
/// optimisation folding.
///
/// This is a build-time trampoline, never executed at runtime. It IS reached
/// by the static reference in `lib.rs` so LTO cannot drop it.
#[inline(never)]
pub fn keepalive_c_abi_symbols() -> usize {
    // Cast every extern-C fn to a *const () pointer (fn-item -> fn-ptr ->
    // opaque ptr), then XOR them together. The intermediate `*const ()`
    // step is required by the `function_casts_as_integer` lint. Any live
    // reference will do; XOR-fold keeps the compiler from realising it can
    // be constant-folded when LTO is aggressive.
    let mut acc: usize = 0;
    acc ^= vokra_version as *const () as usize;
    acc ^= vokra_last_error as *const () as usize;
    acc ^= vokra_session_create_from_file as *const () as usize;
    acc ^= vokra_session_retain as *const () as usize;
    acc ^= vokra_session_destroy as *const () as usize;
    acc ^= vokra_asr_transcribe as *const () as usize;
    acc ^= vokra_string_free as *const () as usize;
    acc ^= vokra_tts_synthesize as *const () as usize;
    acc ^= vokra_audio_free as *const () as usize;
    acc ^= vokra_stream_open as *const () as usize;
    acc ^= vokra_stream_push_pcm as *const () as usize;
    acc ^= vokra_stream_poll as *const () as usize;
    acc ^= vokra_stream_poll_events as *const () as usize;
    acc ^= vokra_stream_interrupt as *const () as usize;
    acc ^= vokra_stream_destroy as *const () as usize;
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_roundtrip_all_variants() {
        for &v in &[
            VokraStatus::Ok,
            VokraStatus::Io,
            VokraStatus::ModelLoad,
            VokraStatus::UnsupportedOp,
            VokraStatus::BackendUnavailable,
            VokraStatus::InvalidArgument,
            VokraStatus::GraphValidation,
            VokraStatus::NotImplemented,
            VokraStatus::Panic,
            VokraStatus::Other,
        ] {
            assert_eq!(VokraStatus::from_raw(v as i32), v);
        }
    }

    #[test]
    fn status_unknown_code_maps_to_other() {
        // Any code outside 0..=8 (`Panic`) must map to `Other`. `Other` (9)
        // itself round-trips to `Other`; anything else is coerced.
        assert_eq!(VokraStatus::from_raw(9), VokraStatus::Other);
        assert_eq!(VokraStatus::from_raw(42), VokraStatus::Other);
        assert_eq!(VokraStatus::from_raw(-1), VokraStatus::Other);
        assert_eq!(VokraStatus::from_raw(i32::MAX), VokraStatus::Other);
    }

    #[test]
    fn event_layout_is_12_bytes_pod() {
        // Locks the ABI-visible size of `vokra_event_t`. `include/vokra.h`
        // ships a fixed 12-byte POD; a silent widening would corrupt every
        // event drained through `vokra_stream_poll_events`.
        assert_eq!(core::mem::size_of::<VokraEvent>(), 12);
    }
}
