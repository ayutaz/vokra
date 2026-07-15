//! Full-duplex S2S C ABI (M4-06-T20) + the FR-MD-09 attribution query
//! (T24) — Moshi's continuous mic→model→speaker session for Unity /
//! Godot deployers (IL2CPP / GDExtension: plain handles, no closures).
//!
//! # Handle model
//!
//! - `vokra_s2s_duplex_t` — one live duplex session (push / pull /
//!   text). **Single-owner-thread** like every stateful Vokra handle.
//! - `vokra_s2s_interrupt_t` — a **separate** barge-in handle
//!   (`vokra_s2s_interrupt_handle`) that is safe to fire from any thread
//!   while another thread pushes/pulls: it owns its own atomic-flag
//!   clone, so there is no aliasing with the session handle (this goes
//!   one step past the stream.rs follow-on note — duplex barge-in is a
//!   core feature, ADR M4-06 §D6).
//!
//! # Config flattening
//!
//! `DuplexSessionConfig` is `#[non_exhaustive]` Rust-side; the open call
//! takes its fields as scalars (C-stable). `aec_disabled_explicitly != 0`
//! is the **only** way to skip the canceller and leaves a loud warning on
//! the session (FR-EX-08 — AEC 無しの Moshi/CSM は自己エコーで即崩壊).
//!
//! # Prerelease ABI
//!
//! Added under the Pre-1.0 policy (docs/abi-changelog.md — freeze fires
//! at M5-13 / v1.0 GA); every addition is recorded there.

use std::os::raw::c_char;

use vokra_core::{DuplexInterruptHandle, DuplexSessionConfig, S2sDuplexHandle, Session};

use crate::error::{self, vokra_status_t};
use crate::ffi_guard;
use crate::handle::{self, vokra_session_t};

/// Opaque duplex-session handle (module docs). Created by
/// `vokra_s2s_duplex_open`, released by `vokra_s2s_duplex_destroy`.
#[allow(non_camel_case_types)]
pub struct vokra_s2s_duplex_t {
    /// The live session (declaration order = drop order: the handle drops
    /// before the retained session below).
    pub(crate) duplex: Box<dyn S2sDuplexHandle + Send>,
    /// A retained clone keeping the model alive independently of
    /// `vokra_session_destroy`.
    pub(crate) _session: Session,
}

/// Opaque cross-thread barge-in handle (module docs). Created by
/// `vokra_s2s_interrupt_handle`, released by
/// `vokra_s2s_interrupt_destroy`; firing it is `vokra_s2s_interrupt`.
#[allow(non_camel_case_types)]
pub struct vokra_s2s_interrupt_t {
    pub(crate) handle: DuplexInterruptHandle,
}

/// Opens a full-duplex S2S session (Moshi = M4-06) on a session whose
/// model injected a duplex engine.
///
/// # Parameters
///
/// - `session`: a live session handle (`vokra_session_create_from_file`
///   with a `moshi` GGUF).
/// - `deterministic`: non-zero → greedy (temperature-0) sampling on both
///   decode channels (reproducible demos / parity).
/// - `seed`: stochastic sampling seed (ignored when `deterministic`).
/// - `aec_disabled_explicitly`: non-zero **explicitly** skips the echo
///   canceller (recorded-file input only — a loud warning is recorded on
///   the session; there is no silent variant). Zero requires the engine's
///   AEC wiring and fails loudly without it (FR-OP-60).
/// - `playback_offset_samples`: echo-reference clock compensation for the
///   real playback latency (owner-tunable; 0 for file-driven use).
/// - `out_duplex`: on `VOKRA_OK`, receives the new handle.
///
/// # Returns
///
/// `VOKRA_OK`, or a non-zero status (no duplex engine on this session,
/// AEC posture violation, ...) with detail from `vokra_last_error()`.
///
/// # Safety
///
/// `session` must be a live session handle; `out_duplex` must be a valid,
/// writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_s2s_duplex_open(
    session: *const vokra_session_t,
    deterministic: i32,
    seed: u64,
    aec_disabled_explicitly: i32,
    playback_offset_samples: u64,
    out_duplex: *mut *mut vokra_s2s_duplex_t,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: caller contract — `session` is a live handle.
        let s = unsafe { ffi_guard::required_ref(session, "session")? };
        ffi_guard::require_out_ptr(out_duplex, "out_duplex")?;
        let mut config = DuplexSessionConfig::new().with_seed(seed);
        if deterministic != 0 {
            config = config.deterministic();
        }
        if aec_disabled_explicitly != 0 {
            config = config.with_aec_disabled_explicitly();
        }
        config = config.with_playback_offset_samples(playback_offset_samples);
        let duplex = s
            .session
            .s2s()
            .duplex_with(&config)
            .map_err(|e| error::fail(&e))?;
        let handle = handle::into_raw(vokra_s2s_duplex_t {
            duplex,
            _session: s.session.clone(),
        });
        // SAFETY: `out_duplex` is non-null (checked above) and writable per
        // the caller contract.
        unsafe { *out_duplex = handle };
        Ok(())
    })
}

/// Mono samples per push/pull frame of `duplex` (size your PCM buffers
/// with this — 1920 for the real 24 kHz / 12.5 Hz model).
///
/// # Safety
///
/// `duplex` must be a live duplex handle; `out_hop` valid and writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_s2s_frame_hop(
    duplex: *const vokra_s2s_duplex_t,
    out_hop: *mut usize,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: caller contract — `duplex` is a live handle.
        let d = unsafe { ffi_guard::required_ref(duplex, "duplex")? };
        ffi_guard::require_out_ptr(out_hop, "out_hop")?;
        // SAFETY: non-null + writable per the caller contract.
        unsafe { *out_hop = d.duplex.frame_hop() };
        Ok(())
    })
}

/// PCM sample rate (Hz) of both duplex directions.
///
/// # Safety
///
/// `duplex` must be a live duplex handle; `out_rate` valid and writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_s2s_sample_rate(
    duplex: *const vokra_s2s_duplex_t,
    out_rate: *mut u32,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: caller contract — `duplex` is a live handle.
        let d = unsafe { ffi_guard::required_ref(duplex, "duplex")? };
        ffi_guard::require_out_ptr(out_rate, "out_rate")?;
        // SAFETY: non-null + writable per the caller contract.
        unsafe { *out_rate = d.duplex.sample_rate() };
        Ok(())
    })
}

/// Feeds one mic frame (exactly `vokra_s2s_frame_hop` samples) through
/// the input front (AEC unless explicitly disabled) and one model step.
///
/// `out_emitted` (optional — may be `NULL`) receives non-zero once the
/// model produced a frame for this push (after its delay warmup); pull it
/// with `vokra_s2s_pull_audio`.
///
/// # Safety
///
/// `duplex` must be a live duplex handle owned by the calling thread;
/// `pcm` must point to `len` readable floats; `out_emitted` must be
/// `NULL` or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_s2s_push_mic(
    duplex: *mut vokra_s2s_duplex_t,
    pcm: *const f32,
    len: usize,
    out_emitted: *mut i32,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: caller contract — `duplex` is a live, thread-owned handle.
        let d = unsafe { ffi_guard::required_mut(duplex, "duplex")? };
        if pcm.is_null() {
            return Err(error::fail_invalid("pcm must not be NULL"));
        }
        // SAFETY: caller contract — `pcm` points to `len` readable floats.
        let frame = unsafe { std::slice::from_raw_parts(pcm, len) };
        let report = d
            .duplex
            .push_mic_frame(frame)
            .map_err(|e| error::fail(&e))?;
        if !out_emitted.is_null() {
            // SAFETY: non-null (checked) + writable per the caller contract.
            unsafe { *out_emitted = i32::from(report.step_emitted) };
        }
        Ok(())
    })
}

/// Pops the next model frame for playback into `out_pcm`
/// (`capacity >= vokra_s2s_frame_hop` floats). `*out_len` is the sample
/// count written — `0` means nothing is pending (delay warmup / caught
/// up / flushed by a barge-in). Pulling is the playback hand-off: the
/// frame is stamped into the echo-reference queue at this moment.
///
/// # Safety
///
/// `duplex` must be a live duplex handle owned by the calling thread;
/// `out_pcm` must point to `capacity` writable floats; `out_len` must be
/// valid and writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_s2s_pull_audio(
    duplex: *mut vokra_s2s_duplex_t,
    out_pcm: *mut f32,
    capacity: usize,
    out_len: *mut usize,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: caller contract — `duplex` is a live, thread-owned handle.
        let d = unsafe { ffi_guard::required_mut(duplex, "duplex")? };
        ffi_guard::require_out_ptr(out_len, "out_len")?;
        let hop = d.duplex.frame_hop();
        match d.duplex.pull_model_frame().map_err(|e| error::fail(&e))? {
            Some(frame) => {
                if out_pcm.is_null() || capacity < frame.len() {
                    return Err(error::fail_invalid(&format!(
                        "out_pcm must hold at least one frame ({hop} floats; got \
                         capacity {capacity})"
                    )));
                }
                // SAFETY: `out_pcm` points to `capacity >= frame.len()`
                // writable floats per the caller contract.
                unsafe {
                    std::ptr::copy_nonoverlapping(frame.as_ptr(), out_pcm, frame.len());
                    *out_len = frame.len();
                }
            }
            None => {
                // SAFETY: non-null (checked) + writable per the contract.
                unsafe { *out_len = 0 };
            }
        }
        Ok(())
    })
}

/// Copies the inner monologue accumulated so far (Moshi's self-generated
/// transcript, display-rule filtered) into `buf` as NUL-terminated UTF-8.
///
/// `*out_needed` always receives the byte length **including** the NUL;
/// when `buf` is `NULL` or `buf_len` is too small, nothing is written and
/// the call still returns `VOKRA_OK` — size with a first call, then
/// fetch (the standard two-call string discipline).
///
/// # Safety
///
/// `duplex` must be a live duplex handle owned by the calling thread;
/// `buf` must be `NULL` or `buf_len` writable bytes; `out_needed` valid
/// and writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_s2s_text(
    duplex: *const vokra_s2s_duplex_t,
    buf: *mut c_char,
    buf_len: usize,
    out_needed: *mut usize,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: caller contract — `duplex` is a live handle.
        let d = unsafe { ffi_guard::required_ref(duplex, "duplex")? };
        ffi_guard::require_out_ptr(out_needed, "out_needed")?;
        let text = d.duplex.monologue_text().map_err(|e| error::fail(&e))?;
        write_utf8_with_needed(&text, buf, buf_len, out_needed)
    })
}

/// Creates a cross-thread barge-in handle for `duplex` (module docs —
/// safe to fire from another thread while this one pushes/pulls).
///
/// # Safety
///
/// `duplex` must be a live duplex handle; `out_interrupt` valid and
/// writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_s2s_interrupt_handle(
    duplex: *const vokra_s2s_duplex_t,
    out_interrupt: *mut *mut vokra_s2s_interrupt_t,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: caller contract — `duplex` is a live handle.
        let d = unsafe { ffi_guard::required_ref(duplex, "duplex")? };
        ffi_guard::require_out_ptr(out_interrupt, "out_interrupt")?;
        let handle = handle::into_raw(vokra_s2s_interrupt_t {
            handle: d.duplex.interrupt_handle(),
        });
        // SAFETY: non-null (checked) + writable per the caller contract.
        unsafe { *out_interrupt = handle };
        Ok(())
    })
}

/// Requests barge-in (M3-14 semantics): the session flushes pending model
/// output and resets its generation state at the next push/pull boundary;
/// mic intake continues. Callable from any thread (the handle owns its
/// own atomic flag — module docs).
///
/// # Safety
///
/// `interrupt` must be a live interrupt handle (not yet destroyed).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_s2s_interrupt(
    interrupt: *const vokra_s2s_interrupt_t,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: caller contract — `interrupt` is a live handle.
        let i = unsafe { ffi_guard::required_ref(interrupt, "interrupt")? };
        i.handle.interrupt();
        Ok(())
    })
}

/// Frees a barge-in handle. `NULL` is a no-op; double-free is undefined
/// behaviour.
///
/// # Safety
///
/// `interrupt` must be `NULL` or a handle from
/// `vokra_s2s_interrupt_handle` not already freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_s2s_interrupt_destroy(interrupt: *mut vokra_s2s_interrupt_t) {
    ffi_guard::guard_void(|| {
        // SAFETY: per the contract `interrupt` is NULL or a live handle
        // from `into_raw`, freed exactly once here.
        unsafe { handle::drop_raw(interrupt) };
    });
}

/// Frees a duplex session. `NULL` is a no-op; double-free is undefined
/// behaviour. Outstanding interrupt handles stay valid (they only own an
/// atomic flag) but firing them after destroy has no observer.
///
/// # Safety
///
/// `duplex` must be `NULL` or a handle from `vokra_s2s_duplex_open` not
/// already freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_s2s_duplex_destroy(duplex: *mut vokra_s2s_duplex_t) {
    ffi_guard::guard_void(|| {
        // SAFETY: per the contract `duplex` is NULL or a live handle from
        // `into_raw`, freed exactly once here.
        unsafe { handle::drop_raw(duplex) };
    });
}

/// Copies the model's attribution text (FR-MD-09 — weights whose license
/// requires display, e.g. Moshi / Mimi CC-BY 4.0) into `buf` as
/// NUL-terminated UTF-8, with the two-call sizing discipline of
/// `vokra_s2s_text`.
///
/// `*out_needed == 0` (and nothing written) means the loaded model's
/// weights carry **no** display obligation (permissive licenses) — an
/// attribution-required weight always yields a non-empty text (the
/// runtime falls back to a registry-derived string when the GGUF chunk is
/// absent; M4-06-T23).
///
/// # Safety
///
/// `session` must be a live session handle; `buf` must be `NULL` or
/// `buf_len` writable bytes; `out_needed` valid and writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vokra_model_attribution(
    session: *const vokra_session_t,
    buf: *mut c_char,
    buf_len: usize,
    out_needed: *mut usize,
) -> vokra_status_t {
    ffi_guard::guard(|| {
        // SAFETY: caller contract — `session` is a live handle.
        let s = unsafe { ffi_guard::required_ref(session, "session")? };
        ffi_guard::require_out_ptr(out_needed, "out_needed")?;
        match s.session.attribution() {
            Some(info) => write_utf8_with_needed(&info.text, buf, buf_len, out_needed),
            None => {
                // SAFETY: non-null (checked) + writable per the contract.
                unsafe { *out_needed = 0 };
                Ok(())
            }
        }
    })
}

/// The two-call UTF-8 string discipline: report the NUL-inclusive length,
/// write only when the caller's buffer fits it.
fn write_utf8_with_needed(
    text: &str,
    buf: *mut c_char,
    buf_len: usize,
    out_needed: *mut usize,
) -> Result<(), vokra_status_t> {
    if text.as_bytes().contains(&0) {
        return Err(error::fail_invalid(
            "text contains an interior NUL byte (cannot cross the C boundary)",
        ));
    }
    let needed = text.len() + 1;
    // SAFETY: `out_needed` was null-checked by the caller of this helper.
    unsafe { *out_needed = needed };
    if !buf.is_null() && buf_len >= needed {
        // SAFETY: `buf` points to `buf_len >= needed` writable bytes per
        // the caller contract; the copy + NUL stay within it.
        unsafe {
            std::ptr::copy_nonoverlapping(text.as_ptr(), buf.cast::<u8>(), text.len());
            *buf.add(text.len()) = 0;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::vokra_session_create_from_file;
    use std::ffi::CString;

    /// Builds a converted synthetic Moshi GGUF (BF16 checkpoint + tiny SPM
    /// blob through the real offline converter) and opens a C session.
    fn moshi_session(tag: &str) -> (*mut vokra_session_t, std::path::PathBuf) {
        let dir =
            std::env::temp_dir().join(format!("vokra-capi-moshi-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ckpt = dir.join("model.safetensors");
        let tok = dir.join("tok.model");
        let gguf = dir.join("moshi.gguf");
        std::fs::write(&ckpt, synthetic_checkpoint()).unwrap();
        std::fs::write(&tok, spm_blob(13)).unwrap();
        vokra_convert::convert_moshi_file(&ckpt, Some(tok.as_path()), &gguf).expect("convert");
        let path = CString::new(gguf.to_str().unwrap()).unwrap();
        let mut session: *mut vokra_session_t = std::ptr::null_mut();
        // SAFETY: valid NUL-terminated path + writable out pointer.
        let status = unsafe { vokra_session_create_from_file(path.as_ptr(), &mut session) };
        assert_eq!(status, vokra_status_t::VOKRA_OK, "moshi session opens");
        (session, dir)
    }

    fn spm_blob(n: usize) -> Vec<u8> {
        fn varint(mut v: u64, out: &mut Vec<u8>) {
            loop {
                let mut b = (v & 0x7f) as u8;
                v >>= 7;
                if v != 0 {
                    b |= 0x80;
                }
                out.push(b);
                if v == 0 {
                    break;
                }
            }
        }
        let mut blob = Vec::new();
        for i in 0..n {
            let piece = format!("\u{2581}p{i}");
            let mut msg = Vec::new();
            msg.push(0x0a);
            varint(piece.len() as u64, &mut msg);
            msg.extend_from_slice(piece.as_bytes());
            msg.push(0x18);
            msg.push(0x01);
            blob.push(0x0a);
            varint(msg.len() as u64, &mut blob);
            blob.extend_from_slice(&msg);
        }
        blob
    }

    fn synthetic_checkpoint() -> Vec<u8> {
        let mut entries: Vec<(String, Vec<u64>)> = Vec::new();
        let (d, text, card) = (16u64, 13u64, 9u64);
        let (h_tm, d_dt, h_dt) = (8u64, 8u64, 6u64);
        entries.push(("text_emb.weight".into(), vec![text + 1, d]));
        entries.push(("text_linear.weight".into(), vec![text, d]));
        entries.push(("out_norm.alpha".into(), vec![1, 1, d]));
        for k in 0..4 {
            entries.push((format!("emb.{k}.weight"), vec![card + 1, d]));
        }
        for i in 0..2 {
            let p = format!("transformer.layers.{i}");
            entries.push((format!("{p}.norm1.alpha"), vec![1, 1, d]));
            entries.push((format!("{p}.norm2.alpha"), vec![1, 1, d]));
            entries.push((format!("{p}.self_attn.in_proj_weight"), vec![3 * d, d]));
            entries.push((format!("{p}.self_attn.out_proj.weight"), vec![d, d]));
            entries.push((format!("{p}.gating.linear_in.weight"), vec![2 * h_tm, d]));
            entries.push((format!("{p}.gating.linear_out.weight"), vec![d, h_tm]));
        }
        for cb in 0..2 {
            entries.push((format!("depformer_in.{cb}.weight"), vec![d_dt, d]));
            entries.push((format!("linears.{cb}.weight"), vec![card, d_dt]));
        }
        entries.push(("depformer_text_emb.weight".into(), vec![text + 1, d_dt]));
        entries.push(("depformer_emb.0.weight".into(), vec![card + 1, d_dt]));
        for i in 0..2 {
            let p = format!("depformer.layers.{i}");
            entries.push((format!("{p}.norm1.alpha"), vec![1, 1, d_dt]));
            entries.push((format!("{p}.norm2.alpha"), vec![1, 1, d_dt]));
            entries.push((
                format!("{p}.self_attn.in_proj_weight"),
                vec![2 * 3 * d_dt, d_dt],
            ));
            entries.push((
                format!("{p}.self_attn.out_proj.weight"),
                vec![2 * d_dt, d_dt],
            ));
            for s in 0..2 {
                entries.push((
                    format!("{p}.gating.{s}.linear_in.weight"),
                    vec![2 * h_dt, d_dt],
                ));
                entries.push((
                    format!("{p}.gating.{s}.linear_out.weight"),
                    vec![d_dt, h_dt],
                ));
            }
        }
        let mut header = String::from("{");
        let mut data: Vec<u8> = Vec::new();
        let mut lcg = 0xB16B_00B5u32;
        for (i, (name, shape)) in entries.iter().enumerate() {
            let n: u64 = shape.iter().product();
            let start = data.len();
            for _ in 0..n {
                lcg = lcg.wrapping_mul(1664525).wrapping_add(1013904223);
                let frac = (lcg >> 16) as u16 & 0x007F;
                let sign = ((lcg >> 8) as u16) & 0x8000;
                data.extend_from_slice(&(sign | 0x3E00 | frac).to_le_bytes());
            }
            let end = data.len();
            if i > 0 {
                header.push(',');
            }
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"BF16\",\"shape\":[{}],\"data_offsets\":[{start},{end}]}}",
                shape
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        header.push('}');
        let mut blob = Vec::new();
        blob.extend_from_slice(&(header.len() as u64).to_le_bytes());
        blob.extend_from_slice(header.as_bytes());
        blob.extend_from_slice(&data);
        blob
    }

    #[test]
    fn duplex_open_push_pull_text_interrupt_lifecycle() {
        let (session, dir) = moshi_session("lifecycle");
        let mut duplex: *mut vokra_s2s_duplex_t = std::ptr::null_mut();
        // Recorded-input opt-out keeps the fixture self-contained (the AEC
        // path is covered Rust-side; the posture flag flows through here).
        // SAFETY: live session, writable out pointer.
        let status = unsafe { vokra_s2s_duplex_open(session, 1, 0, 1, 0, &mut duplex) };
        assert_eq!(status, vokra_status_t::VOKRA_OK);

        let mut hop = 0usize;
        // SAFETY: live handles + writable outs.
        unsafe {
            assert_eq!(
                vokra_s2s_frame_hop(duplex, &mut hop),
                vokra_status_t::VOKRA_OK
            );
        }
        assert_eq!(hop, 1920, "24 kHz / 12.5 Hz");
        let mut rate = 0u32;
        // SAFETY: live handle + writable out.
        unsafe {
            assert_eq!(
                vokra_s2s_sample_rate(duplex, &mut rate),
                vokra_status_t::VOKRA_OK
            );
        }
        assert_eq!(rate, 24_000);

        let mic: Vec<f32> = (0..hop).map(|i| ((i as f32) * 0.01).sin() * 0.2).collect();
        let mut emitted = 0i32;
        let mut out = vec![0.0f32; hop];
        let mut out_len = 0usize;
        let mut total = 0usize;
        for _ in 0..3 {
            // SAFETY: live handle, `mic` has hop readable floats.
            let status =
                unsafe { vokra_s2s_push_mic(duplex, mic.as_ptr(), mic.len(), &mut emitted) };
            assert_eq!(status, vokra_status_t::VOKRA_OK);
            // SAFETY: live handle, `out` has hop writable floats.
            let status =
                unsafe { vokra_s2s_pull_audio(duplex, out.as_mut_ptr(), out.len(), &mut out_len) };
            assert_eq!(status, vokra_status_t::VOKRA_OK);
            total += out_len;
        }
        assert!(total > 0, "model frames were pulled through the C ABI");

        // Two-call string discipline for the monologue.
        let mut needed = 0usize;
        // SAFETY: live handle; NULL buf sizes only.
        unsafe {
            assert_eq!(
                vokra_s2s_text(duplex, std::ptr::null_mut(), 0, &mut needed),
                vokra_status_t::VOKRA_OK
            );
        }
        assert!(needed >= 1, "NUL-inclusive length");
        let mut buf = vec![0i8; needed];
        // SAFETY: live handle; buf has `needed` writable bytes.
        unsafe {
            assert_eq!(
                vokra_s2s_text(duplex, buf.as_mut_ptr(), buf.len(), &mut needed),
                vokra_status_t::VOKRA_OK
            );
        }
        assert_eq!(buf[needed - 1], 0, "NUL-terminated");

        // Cross-thread barge-in through the dedicated handle object.
        let mut interrupt: *mut vokra_s2s_interrupt_t = std::ptr::null_mut();
        // SAFETY: live handle + writable out.
        unsafe {
            assert_eq!(
                vokra_s2s_interrupt_handle(duplex, &mut interrupt),
                vokra_status_t::VOKRA_OK
            );
        }
        let fired = {
            let ptr = interrupt as usize;
            std::thread::spawn(move || {
                // SAFETY: the interrupt handle owns its own atomic flag and
                // is documented cross-thread callable.
                unsafe { vokra_s2s_interrupt(ptr as *const vokra_s2s_interrupt_t) }
            })
            .join()
            .unwrap()
        };
        assert_eq!(fired, vokra_status_t::VOKRA_OK);
        // The next boundary acknowledges: pending output flushes.
        // SAFETY: live handle, mic/out as above.
        unsafe {
            assert_eq!(
                vokra_s2s_pull_audio(duplex, out.as_mut_ptr(), out.len(), &mut out_len),
                vokra_status_t::VOKRA_OK
            );
        }
        assert_eq!(out_len, 0, "barge-in flushed the queue");

        // SAFETY: handles freed exactly once; NULL destroy is a no-op.
        unsafe {
            vokra_s2s_interrupt_destroy(interrupt);
            vokra_s2s_duplex_destroy(duplex);
            vokra_s2s_duplex_destroy(std::ptr::null_mut());
            crate::session::vokra_session_destroy(session);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn attribution_query_two_call_discipline_and_permissive_none() {
        let (session, dir) = moshi_session("attribution");
        let mut needed = 0usize;
        // SAFETY: live session; NULL buf sizes only.
        unsafe {
            assert_eq!(
                vokra_model_attribution(session, std::ptr::null_mut(), 0, &mut needed),
                vokra_status_t::VOKRA_OK
            );
        }
        assert!(needed > 1, "AttributionRequired weight → non-empty text");
        let mut buf = vec![0i8; needed];
        // SAFETY: live session; buf has `needed` writable bytes.
        unsafe {
            assert_eq!(
                vokra_model_attribution(session, buf.as_mut_ptr(), buf.len(), &mut needed),
                vokra_status_t::VOKRA_OK
            );
        }
        let text = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        assert!(text.contains("Kyutai"), "names the author: {text}");
        // SAFETY: freed exactly once.
        unsafe { crate::session::vokra_session_destroy(session) };
        std::fs::remove_dir_all(&dir).ok();

        // A permissive model (the committed Silero fixture) reports 0.
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/parity/silero_vad/silero-vad-v5.gguf");
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let mut session: *mut vokra_session_t = std::ptr::null_mut();
        // SAFETY: valid path + writable out.
        let status = unsafe { vokra_session_create_from_file(cpath.as_ptr(), &mut session) };
        assert_eq!(status, vokra_status_t::VOKRA_OK);
        let mut needed = 42usize;
        // SAFETY: live session + writable out.
        unsafe {
            assert_eq!(
                vokra_model_attribution(session, std::ptr::null_mut(), 0, &mut needed),
                vokra_status_t::VOKRA_OK
            );
        }
        assert_eq!(needed, 0, "permissive weights carry no display obligation");
        // SAFETY: freed exactly once.
        unsafe { crate::session::vokra_session_destroy(session) };
    }

    #[test]
    fn duplex_open_without_engine_is_a_loud_error() {
        // A VAD session has no duplex engine — the open fails with detail.
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/parity/silero_vad/silero-vad-v5.gguf");
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let mut session: *mut vokra_session_t = std::ptr::null_mut();
        // SAFETY: valid path + writable out.
        unsafe {
            assert_eq!(
                vokra_session_create_from_file(cpath.as_ptr(), &mut session),
                vokra_status_t::VOKRA_OK
            );
        }
        let mut duplex: *mut vokra_s2s_duplex_t = std::ptr::null_mut();
        // SAFETY: live session + writable out.
        let status = unsafe { vokra_s2s_duplex_open(session, 0, 0, 0, 0, &mut duplex) };
        assert_ne!(status, vokra_status_t::VOKRA_OK);
        assert!(duplex.is_null(), "out pointer untouched on error");
        // SAFETY: freed exactly once.
        unsafe { crate::session::vokra_session_destroy(session) };
    }
}
