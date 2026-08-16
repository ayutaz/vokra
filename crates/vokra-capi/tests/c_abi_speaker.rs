//! C ABI tests for the speaker surface — `vokra_speaker_embed` /
//! `vokra_speaker_verify` (design
//! `docs/superpowers/specs/2026-08-14-c-abi-backend-speaker-design.md`, §3.3
//! new symbols / §5 error handling / §6 T7-T9).
//!
//! Companion to `c_abi_backend_options.rs`, which covers T1-T6 and explicitly
//! defers this surface to "a separate wave".
//!
//! # Why an integration test when `src/speaker.rs` already has unit tests
//!
//! The unit tests inside `src/speaker.rs` call the functions through Rust
//! module paths. That cannot observe the property every binding depends on:
//! that the symbols are actually *exported with C linkage*. Dropping
//! `#[unsafe(no_mangle)]` would leave those unit tests green while every
//! Unity / Godot / Python / Swift / Kotlin consumer broke at load time. This
//! file declares the symbols the way a C caller does, so it fails to **link**
//! if the export disappears.
//!
//! Two mechanics make that work and are easy to lose in a refactor:
//!
//! - `extern crate vokra;` forces the rlib to be linked; without it the
//!   `#[unsafe(no_mangle)]` symbols are never pulled in and the test fails at
//!   link time with `undefined symbol: vokra_speaker_verify`.
//! - `#![allow(unsafe_code)]` is needed per file: the workspace sets
//!   `unsafe_code = "deny"` and `crates/vokra-capi/src/lib.rs`'s crate-level
//!   allow does not reach a separate integration-test crate.
//!
//! # T7 cannot be taken literally, and why
//!
//! Design §6 T7 asks that `vokra_speaker_embed`'s output match the committed
//! onnxruntime reference embedding (`tests/parity/camplus/embedding.f32`, the
//! 192-d vector matched to 7e-6), reusing that fixture rather than minting a
//! new one. That fixture **cannot be produced through this entry point**:
//! `tests/parity/camplus/gen_reference.py` feeds the network a *seeded Gaussian
//! filterbank* (`manifest.txt`: `seed = 1234`, `input_frames = 200`), not audio.
//! No PCM produces it — a CMN'd log-mel surface is not invertible to an
//! arbitrary Gaussian draw. Asserting otherwise would require inventing a new
//! reference, which the design forbids.
//!
//! So T7 is split along the seam that actually exists, with no new oracle:
//!
//! - [`t7_committed_reference_embedding_is_intact`] and
//!   [`t7_c_abi_verify_consumes_the_onnxruntime_reference`] put the committed
//!   reference through the C ABI unconditionally (no model needed).
//! - [`t7_embed_over_real_campplus_matches_the_reference_chain`] re-pins the
//!   fixture against the real network and checks the PCM entry point, gated on
//!   `VOKRA_CAMPLUS_GGUF` (the 27 MB model is not committed) — the same
//!   convention as `vokra-models`' `speaker::parity`.
//!
//! The two halves the C entry composes are each already pinned to an external
//! oracle elsewhere: PCM→fbank against torchaudio in `vokra-ops`'
//! `tests/kaldi_fbank_parity.rs` (atol 2e-4), and fbank→embedding against
//! onnxruntime in `vokra-models`' `speaker::parity` (atol 0.01).
//!
//! # T9 cannot be model-free, and why
//!
//! `vokra_speaker_embed` validates `out_capacity` against the embedding
//! dimension, which it learns by running the model. There is therefore no
//! capacity path without a speaker model loaded, so the capacity contract is
//! gated ([`t9_two_call_sizing_idiom_over_real_campplus`]). What *is* checked
//! unconditionally is the ordering that makes the gated half meaningful:
//! [`t9_argument_validation_precedes_inference`] proves malformed arguments are
//! rejected before the model is ever consulted, using the committed Silero
//! fixture as a live-but-non-speaker session.
//!
//! Run the gated legs with:
//!
//! ```text
//! VOKRA_CAMPLUS_GGUF=campplus.gguf cargo test -p vokra-capi --test c_abi_speaker
//! ```

#![allow(unsafe_code)]

extern crate vokra;

use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::ptr;

// ---------------------------------------------------------------------------
// C ABI mirror — declared exactly as a C caller sees `include/vokra.h`
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn vokra_last_error() -> *const c_char;
    fn vokra_session_create_from_file(
        path_utf8: *const c_char,
        out_session: *mut *mut c_void,
    ) -> i32;
    fn vokra_session_destroy(session: *mut c_void);

    // --- The speaker surface under test (design §3.3) ---
    fn vokra_speaker_embed(
        session: *const c_void,
        pcm: *const f32,
        num_samples: usize,
        sample_rate: i32,
        out_embedding: *mut f32,
        out_capacity: usize,
        out_written: *mut usize,
    ) -> i32;
    fn vokra_speaker_verify(
        a: *const f32,
        a_len: usize,
        b: *const f32,
        b_len: usize,
        threshold: f32,
        out_similarity: *mut f32,
        out_same_speaker: *mut bool,
    ) -> i32;
}

// `vokra_status_t` values, pinned numerically by
// `crates/vokra-capi/src/error.rs::status_codes_pin_numeric_abi`.
const VOKRA_OK: i32 = 0;
const VOKRA_ERROR_INVALID_ARGUMENT: i32 = 5;
const VOKRA_ERROR_NOT_IMPLEMENTED: i32 = 7;

fn status_name(status: i32) -> &'static str {
    match status {
        VOKRA_OK => "VOKRA_OK",
        1 => "VOKRA_ERROR_IO",
        2 => "VOKRA_ERROR_MODEL_LOAD",
        3 => "VOKRA_ERROR_UNSUPPORTED_OP",
        4 => "VOKRA_ERROR_BACKEND_UNAVAILABLE",
        VOKRA_ERROR_INVALID_ARGUMENT => "VOKRA_ERROR_INVALID_ARGUMENT",
        6 => "VOKRA_ERROR_GRAPH_VALIDATION",
        VOKRA_ERROR_NOT_IMPLEMENTED => "VOKRA_ERROR_NOT_IMPLEMENTED",
        8 => "VOKRA_ERROR_PANIC",
        9 => "VOKRA_ERROR_OTHER",
        _ => "<unknown status>",
    }
}

/// CAM++ embedding dimension (`tests/parity/camplus/manifest.txt`:
/// `embed_dim = 192`). Asserted, never assumed, by the tests below.
const CAMPLUS_EMBED_DIM: usize = 192;

/// Reference frame count (`manifest.txt`: `input_frames = 200`).
const REFERENCE_FRAMES: usize = 200;

/// Filterbank width (`manifest.txt`: `feat_dim = 80`).
const REFERENCE_FEAT_DIM: usize = 80;

/// FP32 parity bound shared with `vokra-models`' `speaker::parity`
/// (`manifest.txt`: `atol = 0.01`).
const ATOL: f32 = 0.01;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The committed CAM++ parity fixtures (M0-08), shared with `vokra-models`'
/// `speaker::parity` — this file adds no reference data of its own.
fn camplus_fixture(name: &str) -> PathBuf {
    repo_root().join("tests/parity/camplus").join(name)
}

fn read_f32_fixture(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("committed fixture {} must be readable: {e}", path.display()));
    assert_eq!(
        bytes.len() % 4,
        0,
        "{} is not a whole number of f32 values",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The onnxruntime reference embedding — the 192-d vector matched to 7e-6
/// (`manifest.txt`: `embedding shape=[1, 192] sha256=f95a94c232b25a03`).
fn reference_embedding() -> Vec<f32> {
    read_f32_fixture(&camplus_fixture("embedding.f32"))
}

/// The committed 2 MB Silero VAD v5 GGUF: a real, loadable model that is **not**
/// a speaker encoder. Used to obtain a live session without the gated CAM++
/// weights.
fn silero_path_cstring() -> CString {
    let path = repo_root().join("tests/parity/silero_vad/silero-vad-v5.gguf");
    assert!(
        path.is_file(),
        "committed Silero fixture missing at {}",
        path.display()
    );
    CString::new(path.to_str().expect("fixture path is UTF-8")).expect("path has no interior NUL")
}

/// Opens a session over the committed Silero fixture.
fn open_silero_session() -> *mut c_void {
    let path = silero_path_cstring();
    let mut session: *mut c_void = ptr::null_mut();
    // SAFETY: valid NUL-terminated path; `session` is a writable out-slot.
    let st = unsafe { vokra_session_create_from_file(path.as_ptr(), &mut session) };
    assert_eq!(
        st,
        VOKRA_OK,
        "loading the committed Silero fixture failed: {} ({:?})",
        status_name(st),
        last_error()
    );
    assert!(!session.is_null());
    session
}

/// The calling thread's last recorded error message, if any.
fn last_error() -> Option<String> {
    // SAFETY: `vokra_last_error` returns either NULL or a NUL-terminated string
    // owned by Vokra and valid until this thread records its next error. The
    // borrow ends before this function returns.
    unsafe {
        let ptr = vokra_last_error();
        if ptr.is_null() {
            None
        } else {
            Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
        }
    }
}

/// Thin wrapper so the tests read like the C two-call idiom.
fn embed(
    session: *const c_void,
    pcm: &[f32],
    sample_rate: i32,
    out: Option<&mut [f32]>,
) -> (i32, usize) {
    let mut written = 0usize;
    let (ptr_out, cap) = match out {
        Some(buf) => (buf.as_mut_ptr(), buf.len()),
        None => (ptr::null_mut(), 0),
    };
    // SAFETY: `session` is a live handle; `pcm` is valid for its own length;
    // `ptr_out` is either NULL with capacity 0 or valid for `cap` writes;
    // `written` is a writable out-slot.
    let st = unsafe {
        vokra_speaker_embed(
            session,
            pcm.as_ptr(),
            pcm.len(),
            sample_rate,
            ptr_out,
            cap,
            &mut written,
        )
    };
    (st, written)
}

/// Cosine similarity through the C ABI, decision slot omitted.
fn similarity(a: &[f32], b: &[f32]) -> (i32, f32) {
    let mut sim = f32::NAN;
    // SAFETY: live slices; `sim` is a writable out-slot; a NULL decision slot
    // is the documented similarity-only mode.
    let st = unsafe {
        vokra_speaker_verify(
            a.as_ptr(),
            a.len(),
            b.as_ptr(),
            b.len(),
            f32::NAN,
            &mut sim,
            ptr::null_mut(),
        )
    };
    (st, sim)
}

/// A deterministic, non-degenerate 1 s mono signal at 16 kHz.
///
/// Two incommensurate sinusoids under a slow envelope: CAM++'s front-end
/// subtracts the cepstral mean, so a constant or silent clip would collapse to a
/// zero-norm embedding and make every comparison below vacuous.
fn probe_pcm(base_hz: f32) -> Vec<f32> {
    (0..16_000)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            let env = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * 1.7 * t).sin();
            env * (0.6 * (2.0 * std::f32::consts::PI * base_hz * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * (base_hz * 6.08) * t).sin())
        })
        .collect()
}

fn campplus_gguf() -> Option<String> {
    std::env::var("VOKRA_CAMPLUS_GGUF").ok()
}

fn open_campplus_session(path: &str) -> *mut c_void {
    let cpath = CString::new(path).expect("path has no interior NUL");
    let mut session: *mut c_void = ptr::null_mut();
    // SAFETY: valid NUL-terminated path; `session` is a writable out-slot.
    let st = unsafe { vokra_session_create_from_file(cpath.as_ptr(), &mut session) };
    assert_eq!(
        st,
        VOKRA_OK,
        "loading the CAM++ GGUF at {path} failed: {} ({:?})",
        status_name(st),
        last_error()
    );
    assert!(!session.is_null());
    session
}

// ---------------------------------------------------------------------------
// Header pinning
// ---------------------------------------------------------------------------

/// The generated header must declare both speaker entry points.
///
/// `scripts/gen-c-abi.sh --check` proves the header matches the source; this
/// proves the *surface a binding author reads* actually contains the symbols
/// this file links against, so the two cannot drift apart silently.
#[test]
fn header_declares_the_speaker_symbols() {
    let header = repo_root().join("include/vokra.h");
    let text = std::fs::read_to_string(&header)
        .unwrap_or_else(|e| panic!("{} must be readable: {e}", header.display()));

    for symbol in ["vokra_speaker_embed", "vokra_speaker_verify"] {
        assert!(
            text.contains(&format!("vokra_status_t {symbol}(")),
            "include/vokra.h must declare `{symbol}` returning vokra_status_t — \
             run scripts/gen-c-abi.sh"
        );
    }
}

// ---------------------------------------------------------------------------
// T7 — the committed onnxruntime reference, through the C ABI
// ---------------------------------------------------------------------------

/// Guards the reference fixture itself: 192 finite, non-degenerate floats.
///
/// Every assertion below leans on this vector, so a truncated or zeroed fixture
/// must fail loudly here rather than silently making the comparisons vacuous.
#[test]
fn t7_committed_reference_embedding_is_intact() {
    let emb = reference_embedding();
    assert_eq!(
        emb.len(),
        CAMPLUS_EMBED_DIM,
        "the committed onnxruntime reference is 192-d (manifest.txt: embed_dim = 192)"
    );
    assert!(
        emb.iter().all(|v| v.is_finite()),
        "the reference embedding must be finite"
    );
    let norm: f64 = emb.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    assert!(
        norm > 0.0,
        "the reference embedding must have a direction (non-zero norm)"
    );
}

/// The C ABI must consume the **real** CAM++ embedding, not just synthetic
/// arrays: the onnxruntime reference vector matched against itself is 1.0.
///
/// This is the model-free half of T7 — it reuses the committed reference
/// exactly as the design requires, and needs no GGUF.
#[test]
fn t7_c_abi_verify_consumes_the_onnxruntime_reference() {
    let emb = reference_embedding();

    let (st, sim) = similarity(&emb, &emb);
    assert_eq!(st, VOKRA_OK, "verify failed: {}", status_name(st));
    assert!(
        (sim - 1.0).abs() < 1e-5,
        "self-similarity of the onnxruntime reference embedding must be 1.0, got {sim}"
    );
}

/// GGUF-gated: the PCM entry point over the real network, plus a re-pin of the
/// committed reference through the same encoder.
///
/// Two distinct claims:
///
/// 1. `input_fbank.f32` → `embedding.f32` still holds on this build (atol 0.01),
///    i.e. the fixture this file reuses is still the network's own output.
/// 2. `vokra_speaker_embed` (PCM in) produces exactly what the Rust
///    `kaldi_fbank` + `SpeakerEncoder::embed` chain produces — bit-exact,
///    because the question here is *wiring*, not numerics.
#[test]
fn t7_embed_over_real_campplus_matches_the_reference_chain() {
    let Some(model) = campplus_gguf() else {
        eprintln!("skipping T7 e2e: set VOKRA_CAMPLUS_GGUF to run");
        return;
    };

    // (1) Re-pin the committed reference against the real network.
    let encoder =
        vokra_models::speaker::SpeakerEncoder::from_path(&model).expect("bind CAM++ encoder");
    let fbank = read_f32_fixture(&camplus_fixture("input_fbank.f32"));
    assert_eq!(
        fbank.len(),
        REFERENCE_FRAMES * REFERENCE_FEAT_DIM,
        "the reference filterbank is [1, 200, 80]"
    );
    let want = reference_embedding();
    let got = encoder
        .embed(&fbank, REFERENCE_FRAMES)
        .expect("CAM++ embed over the reference filterbank");
    let peak = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("camplus reference embedding: max|Δ|={peak:.3e} (atol={ATOL})");
    assert!(
        peak <= ATOL,
        "the committed reference no longer matches the network ({peak} > {ATOL})"
    );

    // (2) The C entry must run the Rust chain and no other.
    let session = open_campplus_session(&model);
    let pcm = probe_pcm(220.0);

    let mut buf = vec![0.0f32; CAMPLUS_EMBED_DIM];
    let (st, written) = embed(session, &pcm, 16_000, Some(&mut buf));
    assert_eq!(
        st,
        VOKRA_OK,
        "embed failed: {} ({:?})",
        status_name(st),
        last_error()
    );
    assert_eq!(written, CAMPLUS_EMBED_DIM);
    assert!(
        buf.iter().any(|&v| v != 0.0),
        "the embedding is all zeros — the oracle below would be vacuous"
    );

    let opts = vokra_ops::KaldiFbankOpts::camplus();
    let (probe_fbank, frames) = vokra_ops::kaldi_fbank(&pcm, &opts).expect("fbank");
    let expected = encoder.embed(&probe_fbank, frames).expect("rust embed");
    assert_eq!(expected.len(), buf.len());
    for (i, (got, want)) in buf.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "dimension {i} differs ({got} vs {want}) — the C entry is not running the Rust chain"
        );
    }

    // A rate the front-end was not trained at is refused, never resampled
    // behind the caller's back (FR-EX-08).
    let (st, _) = embed(session, &pcm, 22_050, Some(&mut buf));
    assert_eq!(
        st, VOKRA_ERROR_INVALID_ARGUMENT,
        "a 22.05 kHz clip must be rejected, not silently resampled"
    );

    // SAFETY: freshly created handle, destroyed exactly once.
    unsafe { vokra_session_destroy(session) };
}

// ---------------------------------------------------------------------------
// T8 — verify(emb, emb) == 1.0
// ---------------------------------------------------------------------------

/// T8 on the real reference, including the decision slot.
#[test]
fn t8_verify_of_the_reference_embedding_with_itself_is_one() {
    let emb = reference_embedding();
    let mut sim = f32::NAN;
    let mut same = false;
    // SAFETY: live slices; both out-slots are writable.
    let st = unsafe {
        vokra_speaker_verify(
            emb.as_ptr(),
            emb.len(),
            emb.as_ptr(),
            emb.len(),
            0.99,
            &mut sim,
            &mut same,
        )
    };
    assert_eq!(st, VOKRA_OK, "verify failed: {}", status_name(st));
    assert!(
        (sim - 1.0).abs() < 1e-5,
        "an embedding must match itself exactly, got {sim}"
    );
    assert!(same, "self-match must be accepted at threshold 0.99");
}

/// The ordering that makes T8 non-vacuous: a decorrelated copy of the same
/// vector must score strictly below the self-match.
///
/// Mirrors `vokra-models`' `speaker::parity::speaker_verify_on_reference_embedding`
/// at the C boundary — same values, permuted dimensions, so a `verify` that
/// merely returned 1.0 unconditionally would fail here.
#[test]
fn t8_verify_decorrelated_reference_copy_scores_strictly_lower() {
    let emb = reference_embedding();
    let mut rolled = emb.clone();
    rolled.rotate_left(37);

    let (st_self, self_sim) = similarity(&emb, &emb);
    assert_eq!(st_self, VOKRA_OK);
    let (st_cross, cross_sim) = similarity(&emb, &rolled);
    assert_eq!(st_cross, VOKRA_OK);

    eprintln!("speaker_verify (C ABI): self={self_sim:.6} cross={cross_sim:.6}");
    assert!(
        cross_sim < self_sim,
        "a decorrelated copy must score below self ({cross_sim} !< {self_sim})"
    );
}

// ---------------------------------------------------------------------------
// T9 — the caller-owned buffer contract
// ---------------------------------------------------------------------------

/// T9: a short `out_capacity` is refused, and `*out_written` still carries the
/// required length so the two-call idiom works.
///
/// Gated: the capacity is validated against the embedding dimension, which the
/// implementation learns by running the model, so there is no capacity path
/// without real weights (see the module docs).
#[test]
fn t9_two_call_sizing_idiom_over_real_campplus() {
    let Some(model) = campplus_gguf() else {
        eprintln!("skipping T9 capacity contract: set VOKRA_CAMPLUS_GGUF to run");
        return;
    };
    let session = open_campplus_session(&model);
    let pcm = probe_pcm(220.0);

    // (1) Sizing call: no buffer at all still reports the dimension.
    let (st, needed) = embed(session, &pcm, 16_000, None);
    assert_eq!(
        st,
        VOKRA_ERROR_INVALID_ARGUMENT,
        "a zero-capacity call reports the size and refuses to write, got {}",
        status_name(st)
    );
    assert_eq!(needed, CAMPLUS_EMBED_DIM, "CAM++ embeddings are 192-d");

    // (2) One float short is still refused, still reports the size, and must
    //     not partially fill the caller's buffer.
    let mut short = vec![0.0f32; needed - 1];
    let (st, written) = embed(session, &pcm, 16_000, Some(&mut short));
    assert_eq!(st, VOKRA_ERROR_INVALID_ARGUMENT);
    assert_eq!(
        written, needed,
        "*out_written must carry the required length on the short-buffer path"
    );
    assert!(
        short.iter().all(|&v| v == 0.0),
        "a refused call must not partially fill the caller's buffer"
    );

    // (3) Exactly the reported size succeeds.
    let mut exact = vec![0.0f32; needed];
    let (st, written) = embed(session, &pcm, 16_000, Some(&mut exact));
    assert_eq!(
        st,
        VOKRA_OK,
        "embed failed: {} ({:?})",
        status_name(st),
        last_error()
    );
    assert_eq!(written, needed);
    assert!(exact.iter().any(|&v| v != 0.0));

    // (4) A larger buffer is fine and writes only the reported prefix.
    let mut roomy = vec![7.0f32; needed + 8];
    let (st, written) = embed(session, &pcm, 16_000, Some(&mut roomy));
    assert_eq!(st, VOKRA_OK);
    assert_eq!(written, needed);
    assert_eq!(
        &roomy[..needed],
        &exact[..],
        "the same clip must produce the same embedding"
    );
    assert!(
        roomy[needed..].iter().all(|&v| v == 7.0),
        "embed must not write past *out_written"
    );

    // SAFETY: freshly created handle, destroyed exactly once.
    unsafe { vokra_session_destroy(session) };
}

/// The unconditional half of T9: malformed arguments are rejected **before**
/// the model is consulted.
///
/// Uses the committed Silero fixture — a live session whose model is not a
/// speaker encoder. A valid call on it reports `NOT_IMPLEMENTED`, so seeing
/// `INVALID_ARGUMENT` for a malformed one proves argument validation runs first
/// rather than after a wasted (or, on a real speaker model, expensive) forward.
#[test]
fn t9_argument_validation_precedes_inference() {
    let session = open_silero_session();
    let pcm = probe_pcm(220.0);
    let mut buf = vec![0.0f32; CAMPLUS_EMBED_DIM];

    // Baseline: well-formed arguments reach the model and report the task
    // mismatch. If this ever stops being NOT_IMPLEMENTED the contrasts below
    // lose their meaning.
    let (st, _) = embed(session, &pcm, 16_000, Some(&mut buf));
    assert_eq!(
        st,
        VOKRA_ERROR_NOT_IMPLEMENTED,
        "a non-speaker model must report the task mismatch, got {}",
        status_name(st)
    );

    // A non-positive rate is an argument fault, not a task mismatch.
    for rate in [0, -1, i32::MIN] {
        let (st, _) = embed(session, &pcm, rate, Some(&mut buf));
        assert_eq!(
            st,
            VOKRA_ERROR_INVALID_ARGUMENT,
            "sample_rate = {rate} must be rejected as an argument fault, got {}",
            status_name(st)
        );
    }

    // NULL pcm with a non-zero length is rejected without dereferencing it.
    let mut written = 0usize;
    // SAFETY: NULL with len > 0 is the rejected branch; no deref happens.
    let st = unsafe {
        vokra_speaker_embed(
            session,
            ptr::null(),
            16,
            16_000,
            buf.as_mut_ptr(),
            buf.len(),
            &mut written,
        )
    };
    assert_eq!(
        st, VOKRA_ERROR_INVALID_ARGUMENT,
        "a NULL pcm pointer with len > 0 must be rejected before inference"
    );

    // A non-zero capacity with a NULL destination is a contradiction.
    // SAFETY: NULL destination with capacity > 0 is the rejected branch.
    let st = unsafe {
        vokra_speaker_embed(
            session,
            pcm.as_ptr(),
            pcm.len(),
            16_000,
            ptr::null_mut(),
            CAMPLUS_EMBED_DIM,
            &mut written,
        )
    };
    assert_eq!(
        st, VOKRA_ERROR_INVALID_ARGUMENT,
        "out_capacity > 0 with a NULL out_embedding must be rejected"
    );

    // SAFETY: freshly created handle, destroyed exactly once.
    unsafe { vokra_session_destroy(session) };
}

// ---------------------------------------------------------------------------
// NULL / defensive arguments (design §5: no panic may cross the boundary)
// ---------------------------------------------------------------------------

/// Every required pointer of `vokra_speaker_embed` rejects NULL without
/// panicking, and leaves the caller's outputs untouched.
#[test]
fn embed_rejects_null_arguments_without_panic() {
    let pcm = probe_pcm(220.0);
    let mut buf = vec![0.0f32; CAMPLUS_EMBED_DIM];

    // NULL session.
    let mut written = 99usize;
    // SAFETY: NULL session is the rejected branch.
    let st = unsafe {
        vokra_speaker_embed(
            ptr::null(),
            pcm.as_ptr(),
            pcm.len(),
            16_000,
            buf.as_mut_ptr(),
            buf.len(),
            &mut written,
        )
    };
    assert_eq!(st, VOKRA_ERROR_INVALID_ARGUMENT);
    assert_eq!(written, 99, "out_written untouched on the reject path");
    assert!(
        buf.iter().all(|&v| v == 0.0),
        "out_embedding untouched on the reject path"
    );

    // NULL out_written, with an otherwise valid live session.
    let session = open_silero_session();
    // SAFETY: NULL out-slot is the rejected branch.
    let st = unsafe {
        vokra_speaker_embed(
            session,
            pcm.as_ptr(),
            pcm.len(),
            16_000,
            buf.as_mut_ptr(),
            buf.len(),
            ptr::null_mut(),
        )
    };
    assert_eq!(st, VOKRA_ERROR_INVALID_ARGUMENT);

    // SAFETY: freshly created handle, destroyed exactly once.
    unsafe { vokra_session_destroy(session) };
}

/// Every required pointer of `vokra_speaker_verify` rejects NULL, and the
/// remaining argument faults (length mismatch, empty, zero-norm) are reported
/// rather than panicking.
#[test]
fn verify_rejects_null_and_malformed_arguments_without_panic() {
    let emb = reference_embedding();
    let mut sim = 42.0f32;

    // NULL out_similarity — the one required output.
    // SAFETY: NULL out-slot is the rejected branch.
    let st = unsafe {
        vokra_speaker_verify(
            emb.as_ptr(),
            emb.len(),
            emb.as_ptr(),
            emb.len(),
            0.5,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    assert_eq!(st, VOKRA_ERROR_INVALID_ARGUMENT);

    // NULL embedding with a non-zero length, on each side in turn.
    // SAFETY: NULL with len > 0 is the rejected branch; no deref happens.
    let st = unsafe {
        vokra_speaker_verify(
            ptr::null(),
            emb.len(),
            emb.as_ptr(),
            emb.len(),
            0.5,
            &mut sim,
            ptr::null_mut(),
        )
    };
    assert_eq!(
        st, VOKRA_ERROR_INVALID_ARGUMENT,
        "NULL `a` must be rejected"
    );
    // SAFETY: as above, for `b`.
    let st = unsafe {
        vokra_speaker_verify(
            emb.as_ptr(),
            emb.len(),
            ptr::null(),
            emb.len(),
            0.5,
            &mut sim,
            ptr::null_mut(),
        )
    };
    assert_eq!(
        st, VOKRA_ERROR_INVALID_ARGUMENT,
        "NULL `b` must be rejected"
    );
    assert_eq!(
        sim.to_bits(),
        42.0f32.to_bits(),
        "out_similarity untouched on the reject path"
    );

    // Mismatched lengths have no cosine.
    let (st, _) = similarity(&emb, &emb[..emb.len() - 1]);
    assert_eq!(
        st, VOKRA_ERROR_INVALID_ARGUMENT,
        "mismatched lengths must be rejected"
    );

    // Two empty embeddings have no direction.
    let (st, _) = similarity(&[], &[]);
    assert_eq!(
        st, VOKRA_ERROR_INVALID_ARGUMENT,
        "empty embeddings must be rejected"
    );

    // A zero vector has no direction.
    let zeros = vec![0.0f32; emb.len()];
    let (st, _) = similarity(&zeros, &emb);
    assert_eq!(
        st, VOKRA_ERROR_INVALID_ARGUMENT,
        "a zero-norm embedding must be rejected"
    );

    // A non-finite threshold is only a fault when a decision was requested.
    let mut same = false;
    // SAFETY: live slices; both out-slots are writable.
    let st = unsafe {
        vokra_speaker_verify(
            emb.as_ptr(),
            emb.len(),
            emb.as_ptr(),
            emb.len(),
            f32::NAN,
            &mut sim,
            &mut same,
        )
    };
    assert_eq!(
        st, VOKRA_ERROR_INVALID_ARGUMENT,
        "a NaN threshold must be rejected when a decision is requested"
    );
}

/// A session that holds no speaker model reports the task mismatch rather than
/// guessing — the same posture `vokra_asr_transcribe` takes on a TTS voice.
#[test]
fn speaker_embed_on_a_non_speaker_model_reports_not_implemented() {
    let session = open_silero_session();
    let pcm = probe_pcm(220.0);
    let mut buf = vec![0.0f32; CAMPLUS_EMBED_DIM];

    let (st, _) = embed(session, &pcm, 16_000, Some(&mut buf));
    assert_eq!(
        st,
        VOKRA_ERROR_NOT_IMPLEMENTED,
        "expected the task-mismatch status, got {} ({:?})",
        status_name(st),
        last_error()
    );
    assert!(
        buf.iter().all(|&v| v == 0.0),
        "a refused call must not write the caller's buffer"
    );

    // SAFETY: freshly created handle, destroyed exactly once.
    unsafe { vokra_session_destroy(session) };
}
