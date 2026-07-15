//! Vokra Web entry crate (M4-01-T06 / T19 / T20).
//!
//! Compiled to `wasm32-unknown-unknown` as a cdylib by
//! `scripts/build-wasm.sh` (two artifacts: simd128 + base — ADR M4-01 §4)
//! and driven by:
//!
//! - the npm package loader (`web/pkg/index.js`) — the production
//!   `vokra_wasm_*` session API (always compiled);
//! - the Node kernel-parity harness (`tools/wasm/run-kernel-parity.mjs`) —
//!   the `vokra_test_*` differential entries (feature `test-entries`, ON by
//!   default, OFF in the shipped npm artifact via `--no-default-features`);
//! - the Node Whisper e2e runner (`tools/wasm/run-whisper-wasm.mjs`) and the
//!   browser parity/demo pages — the session API again.
//!
//! # ABI conventions (hand-written JS side: no wasm-bindgen — ADR M4-01 §5)
//!
//! - All exports are `extern "C"` with scalar params; buffers cross as
//!   (pointer, length) pairs into linear memory.
//! - JS allocates transfer buffers with [`vokra_wasm_alloc`] and fills them
//!   **byte-wise** (`new Uint8Array(memory.buffer, ptr, len).set(...)` — no
//!   alignment assumption anywhere; f32 payloads cross as little-endian
//!   bytes and are decoded on the Rust side). It then either hands
//!   ownership to an entry that documents taking it
//!   ([`vokra_wasm_session_create`]) or frees with [`vokra_wasm_free`].
//! - Allocations live in a registry keyed by pointer address — no
//!   `mem::forget` / `Vec::from_raw_parts` reconstruction, so the
//!   allocator's Layout contract is trivially upheld and double-free /
//!   foreign-pointer bugs surface as explicit errors.
//! - Fallible entries return `0` on success / negative on failure and leave
//!   a message readable via [`vokra_wasm_last_error_len`] /
//!   [`vokra_wasm_last_error_read`] (the C-ABI `vokra_last_error` idiom).
//! - wasm32-unknown-unknown builds abort on panic; every entry validates its
//!   inputs and returns error codes instead (public APIs never panic across
//!   the boundary — the same rule the C ABI follows).
//!
//! # Threading
//!
//! wasm32-unknown-unknown is single-threaded (no threads target-feature —
//! multi-threaded kernels are the M4-01 T26 follow-up), so plain
//! `thread_local!` state is the whole synchronisation story.

// The `vokra_test_*` differential entries take raw (ptr, len) pairs, so this
// crate joins the unsafe-boundary allow list (root Cargo.toml, NFR-RL-07);
// every unsafe block carries a // SAFETY: comment.
#![allow(unsafe_code)]

use std::cell::RefCell;
use std::collections::HashMap;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_models::whisper::WhisperAsr;

thread_local! {
    static SESSIONS: RefCell<HashMap<u32, WhisperAsr>> = RefCell::new(HashMap::new());
    static NEXT_ID: RefCell<u32> = const { RefCell::new(1) };
    static LAST_TEXT: RefCell<String> = const { RefCell::new(String::new()) };
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
    /// Registry of live [`vokra_wasm_alloc`] allocations, keyed by pointer
    /// address. Keeping the owning `Vec` here — instead of `mem::forget` +
    /// `Vec::from_raw_parts` reconstruction — upholds the allocator Layout
    /// contract by construction, and unknown/double-freed pointers surface
    /// as explicit errors instead of UB.
    static ALLOCS: RefCell<HashMap<usize, Vec<u8>>> = RefCell::new(HashMap::new());
}

fn set_error(msg: impl Into<String>) {
    LAST_ERROR.with(|e| *e.borrow_mut() = msg.into());
}

/// Copies `s` into the caller's registered output buffer, returns copied
/// length (0 when the buffer is unknown — the registry rejects foreign
/// pointers).
fn copy_str_out(s: &str, dst: *mut u8, cap: u32) -> u32 {
    ALLOCS.with(|a| {
        let mut allocs = a.borrow_mut();
        let Some(buf) = allocs.get_mut(&(dst as usize)) else {
            return 0;
        };
        let n = s.len().min(cap as usize).min(buf.len());
        buf[..n].copy_from_slice(&s.as_bytes()[..n]);
        n as u32
    })
}

/// Allocates `len` zeroed bytes of linear memory for the JS side to fill
/// byte-wise. Returns null on `len == 0`. Pair with [`vokra_wasm_free`]
/// unless an entry documents taking ownership.
#[unsafe(no_mangle)]
pub extern "C" fn vokra_wasm_alloc(len: u32) -> *mut u8 {
    if len == 0 {
        return core::ptr::null_mut();
    }
    let mut v = vec![0u8; len as usize];
    let ptr = v.as_mut_ptr();
    ALLOCS.with(|a| a.borrow_mut().insert(ptr as usize, v));
    ptr
}

/// Frees a buffer produced by [`vokra_wasm_alloc`]. Null, zero-length,
/// unknown, or already-freed pointers are a no-op (`len` is accepted for
/// C-ABI symmetry but the registry knows the true length).
#[unsafe(no_mangle)]
pub extern "C" fn vokra_wasm_free(ptr: *mut u8, len: u32) {
    let _ = len;
    if ptr.is_null() {
        return;
    }
    ALLOCS.with(|a| a.borrow_mut().remove(&(ptr as usize)));
}

/// Whether THIS artifact was compiled with the `simd128` target feature
/// (compile-time — WASM has no runtime feature detection; the JS loader
/// picks the artifact, this export lets harnesses assert artifact identity).
#[unsafe(no_mangle)]
pub extern "C" fn vokra_wasm_simd128_active() -> u32 {
    u32::from(cfg!(target_feature = "simd128"))
}

/// Byte length of the last error message (0 = none).
#[unsafe(no_mangle)]
pub extern "C" fn vokra_wasm_last_error_len() -> u32 {
    LAST_ERROR.with(|e| e.borrow().len() as u32)
}

/// Copies the last error message (UTF-8) into a registered `dst` buffer;
/// returns the copied length.
#[unsafe(no_mangle)]
pub extern "C" fn vokra_wasm_last_error_read(dst: *mut u8, cap: u32) -> u32 {
    LAST_ERROR.with(|e| copy_str_out(&e.borrow(), dst, cap))
}

/// Byte length of the last transcription text.
#[unsafe(no_mangle)]
pub extern "C" fn vokra_wasm_text_len() -> u32 {
    LAST_TEXT.with(|t| t.borrow().len() as u32)
}

/// Copies the last transcription text (UTF-8) into a registered `dst`
/// buffer; returns the copied length.
#[unsafe(no_mangle)]
pub extern "C" fn vokra_wasm_text_read(dst: *mut u8, cap: u32) -> u32 {
    LAST_TEXT.with(|t| copy_str_out(&t.borrow(), dst, cap))
}

/// Creates an ASR session from an in-memory GGUF (M4-01-T19: fetch →
/// ArrayBuffer → linear memory → this entry; **no mmap on wasm** — the M0
/// in-memory bytes loader is the wasm path, `vokra-mmap` is native-only).
///
/// **Takes ownership** of the `(gguf_ptr, gguf_len)` buffer (which must come
/// from [`vokra_wasm_alloc`]) — JS must NOT free it afterwards, on success
/// OR failure.
///
/// `backend`: 0 = CPU (explicit caller choice — the WASM SIMD128/scalar
/// path), 1 = WebGPU (requires a live adapter; absence is an explicit error
/// surfaced through [`vokra_wasm_last_error_read`] at transcribe time —
/// FR-EX-08, never a silent CPU fall back).
///
/// Returns a non-zero session handle, or 0 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn vokra_wasm_session_create(gguf_ptr: *mut u8, gguf_len: u32, backend: u32) -> u32 {
    if gguf_ptr.is_null() || gguf_len == 0 {
        set_error("session_create: null/empty GGUF buffer");
        return 0;
    }
    // Take ownership out of the registry (zero-copy hand-off to the parser).
    let Some(bytes) = ALLOCS.with(|a| a.borrow_mut().remove(&(gguf_ptr as usize))) else {
        set_error(
            "session_create: buffer was not allocated by vokra_wasm_alloc (or already \
             freed/consumed)",
        );
        return 0;
    };
    if bytes.len() != gguf_len as usize {
        set_error(format!(
            "session_create: length mismatch — buffer holds {} bytes, caller says {}",
            bytes.len(),
            gguf_len
        ));
        return 0;
    }
    let kind = match backend {
        0 => BackendKind::Cpu,
        1 => BackendKind::WebGpu,
        other => {
            set_error(format!(
                "session_create: unknown backend code {other} (0 = cpu, 1 = webgpu); Vokra \
                 never picks a backend silently (FR-EX-08)"
            ));
            return 0;
        }
    };
    let file = match GgufFile::parse(bytes) {
        Ok(f) => f,
        Err(e) => {
            set_error(format!("session_create: GGUF parse failed: {e}"));
            return 0;
        }
    };
    let asr = match WhisperAsr::from_gguf(&file) {
        Ok(a) => a.with_backend(kind),
        Err(e) => {
            set_error(format!("session_create: model load failed: {e}"));
            return 0;
        }
    };
    let id = NEXT_ID.with(|n| {
        let mut n = n.borrow_mut();
        let id = *n;
        *n += 1;
        id
    });
    SESSIONS.with(|s| s.borrow_mut().insert(id, asr));
    id
}

/// Destroys a session handle (unknown handles are a no-op).
#[unsafe(no_mangle)]
pub extern "C" fn vokra_wasm_session_destroy(handle: u32) {
    SESSIONS.with(|s| s.borrow_mut().remove(&handle));
}

/// Transcribes 16 kHz mono PCM passed as **little-endian f32 bytes** in a
/// registered buffer (`n_samples` f32 values = `4 * n_samples` bytes; the
/// byte-wise contract avoids any alignment assumption — crate docs). The
/// buffer stays owned by JS (free it afterwards with [`vokra_wasm_free`]).
///
/// On success returns 0 and stores the text for [`vokra_wasm_text_read`];
/// on failure returns -1 with the message in
/// [`vokra_wasm_last_error_read`]. A WebGPU-backend session without a live
/// adapter fails HERE with the explicit BackendUnavailable text (FR-EX-08 —
/// selecting the CPU instead is the caller's explicit choice, backend code
/// 0).
#[unsafe(no_mangle)]
pub extern "C" fn vokra_wasm_transcribe(handle: u32, pcm_ptr: *const u8, n_samples: u32) -> i32 {
    if pcm_ptr.is_null() || n_samples == 0 {
        set_error("transcribe: null/empty PCM buffer");
        return -1;
    }
    let need = n_samples as usize * 4;
    let pcm: Option<Vec<f32>> = ALLOCS.with(|a| {
        let allocs = a.borrow();
        let buf = allocs.get(&(pcm_ptr as usize))?;
        if buf.len() < need {
            return None;
        }
        Some(
            buf[..need]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    });
    let Some(pcm) = pcm else {
        set_error(format!(
            "transcribe: PCM buffer unknown to the registry or shorter than {need} bytes \
             (allocate with vokra_wasm_alloc and fill byte-wise)"
        ));
        return -1;
    };
    SESSIONS.with(|s| {
        let sessions = s.borrow();
        let Some(asr) = sessions.get(&handle) else {
            set_error(format!("transcribe: unknown session handle {handle}"));
            return -1;
        };
        match asr.transcribe_tokens(&pcm) {
            Ok(ids) => match asr.render_ids(&ids) {
                Ok(text) => {
                    LAST_TEXT.with(|t| *t.borrow_mut() = text);
                    0
                }
                Err(e) => {
                    set_error(format!("transcribe: detokenize failed: {e}"));
                    -1
                }
            },
            Err(e) => {
                set_error(format!("transcribe: {e}"));
                -1
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Kernel-parity test entries (M4-01-T06) — feature `test-entries`, not
// shipped in the npm artifact.
// ---------------------------------------------------------------------------

/// `vokra_test_webgpu_*` entries for the browser per-kernel parity harness
/// (M4-01-T18, `tools/wasm/parity.html`): each drives one WGSL kernel
/// through a cached [`vokra_backend_webgpu::WebGpuContext`]; the JS side
/// diffs the output against the CPU oracle (`vokra_test_*` in the same
/// instance) at atol = 0.01 (NFR-QL-01). wasm32-only: on a host without an
/// adapter every entry returns -1 with the explicit BackendUnavailable text
/// in `vokra_wasm_last_error_read` (FR-EX-08 — the harness shows a SKIPPED
/// verdict with the reason, never a fabricated pass).
#[cfg(all(feature = "test-entries", target_arch = "wasm32"))]
mod webgpu_test_entries {
    use std::cell::RefCell;

    use vokra_backend_webgpu::WebGpuContext;
    use vokra_backend_webgpu::plan::{ActivationKind, ElementwiseOp};

    use super::set_error;

    thread_local! {
        static CTX: RefCell<Option<WebGpuContext>> = const { RefCell::new(None) };
    }

    /// Runs `f` with the cached context (created on first use). Returns -1
    /// with the error text stashed when the adapter is unavailable or the
    /// dispatch fails.
    fn with_ctx(f: impl FnOnce(&WebGpuContext) -> vokra_core::Result<()>) -> i32 {
        CTX.with(|c| {
            let mut slot = c.borrow_mut();
            if slot.is_none() {
                match WebGpuContext::new() {
                    Ok(ctx) => *slot = Some(ctx),
                    Err(e) => {
                        set_error(format!("webgpu context unavailable: {e}"));
                        return -1;
                    }
                }
            }
            match f(slot.as_ref().expect("context cached above")) {
                Ok(()) => 0,
                Err(e) => {
                    set_error(format!("webgpu kernel failed: {e}"));
                    -1
                }
            }
        })
    }

    /// # Safety
    /// (ptr, element-count) harness contract (crate docs).
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn vokra_test_webgpu_copy(n: u32, x: *const f32, out: *mut f32) -> i32 {
        let n = n as usize;
        // SAFETY: harness contract — live length-n buffers.
        let (x, out) = unsafe {
            (
                core::slice::from_raw_parts(x, n),
                core::slice::from_raw_parts_mut(out, n),
            )
        };
        with_ctx(|ctx| ctx.copy_f32(x, out))
    }

    /// op: 0 = add, 1 = mul. # Safety: harness contract.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn vokra_test_webgpu_elementwise(
        op: u32,
        n: u32,
        a: *const f32,
        b: *const f32,
        out: *mut f32,
    ) -> i32 {
        let n = n as usize;
        // SAFETY: harness contract — live length-n buffers.
        let (a, b, out) = unsafe {
            (
                core::slice::from_raw_parts(a, n),
                core::slice::from_raw_parts(b, n),
                core::slice::from_raw_parts_mut(out, n),
            )
        };
        let op = if op == 1 {
            ElementwiseOp::Mul
        } else {
            ElementwiseOp::Add
        };
        with_ctx(|ctx| ctx.elementwise_f32(op, a, b, out))
    }

    /// # Safety
    /// Buffers of m*k / k*n / n (nullable) / m*n elements.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn vokra_test_webgpu_gemm(
        m: u32,
        n: u32,
        k: u32,
        a: *const f32,
        b: *const f32,
        bias: *const f32,
        out: *mut f32,
    ) -> i32 {
        let (m, n, k) = (m as usize, n as usize, k as usize);
        // SAFETY: harness contract.
        let (a, b, bias, out) = unsafe {
            (
                core::slice::from_raw_parts(a, m * k),
                core::slice::from_raw_parts(b, k * n),
                if bias.is_null() {
                    None
                } else {
                    Some(core::slice::from_raw_parts(bias, n))
                },
                core::slice::from_raw_parts_mut(out, m * n),
            )
        };
        with_ctx(|ctx| ctx.gemm_f32(m, n, k, a, b, bias, out))
    }

    /// # Safety
    /// Buffers of m*k / k / m (nullable) / m elements.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn vokra_test_webgpu_gemv(
        m: u32,
        k: u32,
        a: *const f32,
        x: *const f32,
        bias: *const f32,
        out: *mut f32,
    ) -> i32 {
        let (m, k) = (m as usize, k as usize);
        // SAFETY: harness contract.
        let (a, x, bias, out) = unsafe {
            (
                core::slice::from_raw_parts(a, m * k),
                core::slice::from_raw_parts(x, k),
                if bias.is_null() {
                    None
                } else {
                    Some(core::slice::from_raw_parts(bias, m))
                },
                core::slice::from_raw_parts_mut(out, m),
            )
        };
        with_ctx(|ctx| ctx.gemv_f32(m, k, a, x, bias, out))
    }

    /// causal: 0 = plain softmax, 1 = causal with `offset`. # Safety:
    /// rows*cols buffers.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn vokra_test_webgpu_softmax(
        rows: u32,
        cols: u32,
        causal: u32,
        offset: u32,
        x: *const f32,
        out: *mut f32,
    ) -> i32 {
        let (rows, cols) = (rows as usize, cols as usize);
        // SAFETY: harness contract.
        let (x, out) = unsafe {
            (
                core::slice::from_raw_parts(x, rows * cols),
                core::slice::from_raw_parts_mut(out, rows * cols),
            )
        };
        with_ctx(|ctx| {
            if causal == 1 {
                ctx.softmax_causal_f32(x, out, rows, cols, offset as usize)
            } else {
                ctx.softmax_f32(x, out, rows, cols)
            }
        })
    }

    /// # Safety
    /// rows*cols x/out; cols gamma/beta.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn vokra_test_webgpu_layer_norm(
        rows: u32,
        cols: u32,
        eps: f32,
        x: *const f32,
        gamma: *const f32,
        beta: *const f32,
        out: *mut f32,
    ) -> i32 {
        let (rows, cols) = (rows as usize, cols as usize);
        // SAFETY: harness contract.
        let (x, gamma, beta, out) = unsafe {
            (
                core::slice::from_raw_parts(x, rows * cols),
                core::slice::from_raw_parts(gamma, cols),
                core::slice::from_raw_parts(beta, cols),
                core::slice::from_raw_parts_mut(out, rows * cols),
            )
        };
        with_ctx(|ctx| ctx.layer_norm_f32(x, out, rows, cols, gamma, beta, eps))
    }

    /// # Safety
    /// Length-n x/out buffers.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn vokra_test_webgpu_gelu(n: u32, x: *const f32, out: *mut f32) -> i32 {
        let n = n as usize;
        // SAFETY: harness contract.
        let (x, out) = unsafe {
            (
                core::slice::from_raw_parts(x, n),
                core::slice::from_raw_parts_mut(out, n),
            )
        };
        with_ctx(|ctx| ctx.gelu_f32(x, out))
    }

    /// kind: 0 relu / 1 sigmoid / 2 tanh. # Safety: length-n buffers.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn vokra_test_webgpu_activation(
        kind: u32,
        n: u32,
        x: *const f32,
        out: *mut f32,
    ) -> i32 {
        let n = n as usize;
        // SAFETY: harness contract.
        let (x, out) = unsafe {
            (
                core::slice::from_raw_parts(x, n),
                core::slice::from_raw_parts_mut(out, n),
            )
        };
        let kind = match kind {
            0 => ActivationKind::Relu,
            1 => ActivationKind::Sigmoid,
            _ => ActivationKind::Tanh,
        };
        with_ctx(|ctx| ctx.activation_f32(kind, x, out))
    }

    /// # Safety
    /// in_ch*in_len x; out_ch*in_ch*kernel w; out_ch bias (nullable);
    /// out_ch*out_len out.
    #[unsafe(no_mangle)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn vokra_test_webgpu_conv1d(
        in_ch: u32,
        in_len: u32,
        out_ch: u32,
        kernel: u32,
        stride: u32,
        padding: u32,
        x: *const f32,
        w: *const f32,
        bias: *const f32,
        out: *mut f32,
        out_len: u32,
    ) -> i32 {
        let (in_ch, in_len, out_ch, kernel, stride, padding, out_len) = (
            in_ch as usize,
            in_len as usize,
            out_ch as usize,
            kernel as usize,
            stride as usize,
            padding as usize,
            out_len as usize,
        );
        // SAFETY: harness contract.
        let (x, w, bias, out) = unsafe {
            (
                core::slice::from_raw_parts(x, in_ch * in_len),
                core::slice::from_raw_parts(w, out_ch * in_ch * kernel),
                if bias.is_null() {
                    None
                } else {
                    Some(core::slice::from_raw_parts(bias, out_ch))
                },
                core::slice::from_raw_parts_mut(out, out_ch * out_len),
            )
        };
        with_ctx(|ctx| {
            ctx.conv1d_f32(
                x, in_ch, in_len, w, out_ch, kernel, bias, stride, padding, out,
            )
        })
    }
}

/// `vokra_test_*` entries for the Node differential harness. Buffers are
/// raw (ptr, element-count) pairs over `vokra_wasm_alloc` allocations; the
/// harness guarantees lengths (test-only surface — the production session
/// API above goes through the registry instead).
#[cfg(feature = "test-entries")]
mod test_entries {
    use vokra_backend_cpu::IsaPath;
    use vokra_backend_cpu::kernels;

    /// Dispatched-ISA code: 0 scalar / 1 avx2 / 2 neon / 3 rvv / 4
    /// wasm-simd128 (mirrors `IsaPath` for JS-side assertions).
    #[unsafe(no_mangle)]
    pub extern "C" fn vokra_test_active_isa_code() -> u32 {
        match vokra_backend_cpu::active_isa() {
            IsaPath::Scalar => 0,
            IsaPath::Avx2 => 1,
            IsaPath::Neon => 2,
            IsaPath::Rvv => 3,
            IsaPath::WasmSimd128 => 4,
        }
    }

    /// Dispatched GEMM (`bias` may be null; `forced_scalar = 1` forces the
    /// scalar ISA path for in-artifact differentials). Returns 0 ok / -1
    /// error.
    ///
    /// # Safety
    /// `a` / `b` / (`bias`) / `out` must be live linear-memory buffers of
    /// exactly `m*k` / `k*n` / `n` / `m*n` f32 elements (harness contract).
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn vokra_test_gemm(
        m: u32,
        n: u32,
        k: u32,
        a: *const f32,
        b: *const f32,
        bias: *const f32,
        out: *mut f32,
        forced_scalar: u32,
    ) -> i32 {
        let (m, n, k) = (m as usize, n as usize, k as usize);
        // SAFETY: harness contract above — element counts match exactly.
        let (a, b, bias, out) = unsafe {
            (
                core::slice::from_raw_parts(a, m * k),
                core::slice::from_raw_parts(b, k * n),
                if bias.is_null() {
                    None
                } else {
                    Some(core::slice::from_raw_parts(bias, n))
                },
                core::slice::from_raw_parts_mut(out, m * n),
            )
        };
        let r = if forced_scalar == 1 {
            kernels::gemm_f32_on(IsaPath::Scalar, m, n, k, a, b, bias, out)
        } else {
            kernels::gemm_f32(m, n, k, a, b, bias, out)
        };
        match r {
            Ok(()) => 0,
            Err(_) => -1,
        }
    }

    /// Dispatched GEMV (`bias` may be null). Returns 0 ok / -1 error.
    ///
    /// # Safety
    /// `a` / `x` / (`bias`) / `out` must be live buffers of exactly `m*k` /
    /// `k` / `m` / `m` f32 elements (harness contract).
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn vokra_test_gemv(
        m: u32,
        k: u32,
        a: *const f32,
        x: *const f32,
        bias: *const f32,
        out: *mut f32,
        forced_scalar: u32,
    ) -> i32 {
        let (m, k) = (m as usize, k as usize);
        // SAFETY: harness contract above — element counts match exactly.
        let (a, x, bias, out) = unsafe {
            (
                core::slice::from_raw_parts(a, m * k),
                core::slice::from_raw_parts(x, k),
                if bias.is_null() {
                    None
                } else {
                    Some(core::slice::from_raw_parts(bias, m))
                },
                core::slice::from_raw_parts_mut(out, m),
            )
        };
        let r = if forced_scalar == 1 {
            kernels::gemv_f32_on(IsaPath::Scalar, m, k, a, x, bias, out)
        } else {
            kernels::gemv_f32(m, k, a, x, bias, out)
        };
        match r {
            Ok(()) => 0,
            Err(_) => -1,
        }
    }

    /// Dispatched element-wise add. Returns 0 ok / -1 error.
    ///
    /// # Safety
    /// `a` / `b` / `out` must be live length-`n` f32 buffers.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn vokra_test_add(
        n: u32,
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        forced_scalar: u32,
    ) -> i32 {
        let n = n as usize;
        // SAFETY: harness contract above.
        let (a, b, out) = unsafe {
            (
                core::slice::from_raw_parts(a, n),
                core::slice::from_raw_parts(b, n),
                core::slice::from_raw_parts_mut(out, n),
            )
        };
        let r = if forced_scalar == 1 {
            kernels::add_f32_on(IsaPath::Scalar, a, b, out)
        } else {
            kernels::add_f32(a, b, out)
        };
        match r {
            Ok(()) => 0,
            Err(_) => -1,
        }
    }

    /// Dispatched element-wise mul. Returns 0 ok / -1 error.
    ///
    /// # Safety
    /// `a` / `b` / `out` must be live length-`n` f32 buffers.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn vokra_test_mul(
        n: u32,
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        forced_scalar: u32,
    ) -> i32 {
        let n = n as usize;
        // SAFETY: harness contract above.
        let (a, b, out) = unsafe {
            (
                core::slice::from_raw_parts(a, n),
                core::slice::from_raw_parts(b, n),
                core::slice::from_raw_parts_mut(out, n),
            )
        };
        let r = if forced_scalar == 1 {
            kernels::mul_f32_on(IsaPath::Scalar, a, b, out)
        } else {
            kernels::mul_f32(a, b, out)
        };
        match r {
            Ok(()) => 0,
            Err(_) => -1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The registry alloc/free pair round-trips; foreign pointers are
    /// rejected; the string-out helper respects both capacity and the
    /// registered buffer length (native smoke of the boundary helpers — the
    /// wasm-side behaviour is exercised by the Node harness, T06/T19).
    #[test]
    fn alloc_registry_roundtrip_and_copy_out() {
        assert!(vokra_wasm_alloc(0).is_null());
        let p = vokra_wasm_alloc(16);
        assert!(!p.is_null());

        set_error("hello");
        assert_eq!(vokra_wasm_last_error_len(), 5);
        // Copy into the registered buffer, capped at 3 bytes.
        let n = vokra_wasm_last_error_read(p, 3);
        assert_eq!(n, 3);
        // A foreign pointer is rejected (returns 0, no write).
        let foreign = [0u8; 4];
        assert_eq!(
            vokra_wasm_last_error_read(foreign.as_ptr() as *mut u8, 4),
            0
        );
        vokra_wasm_free(p, 16);
        // Double free is a no-op.
        vokra_wasm_free(p, 16);
    }

    /// Bad inputs are error codes with readable messages, never panics
    /// (wasm aborts on panic — the entries must stay Result-shaped).
    #[test]
    fn invalid_inputs_are_error_codes_not_panics() {
        assert_eq!(
            vokra_wasm_session_create(core::ptr::null_mut(), 0, 0),
            0,
            "null GGUF must fail"
        );
        assert!(vokra_wasm_last_error_len() > 0);
        assert_eq!(vokra_wasm_transcribe(999, core::ptr::null(), 0), -1);

        // Unknown backend code is an explicit error (FR-EX-08 posture); the
        // buffer is consumed by the taking-ownership contract either way.
        let p = vokra_wasm_alloc(4);
        assert_eq!(vokra_wasm_session_create(p, 4, 7), 0);
        let out = vokra_wasm_alloc(256);
        let n = vokra_wasm_last_error_read(out, 256);
        let msg = ALLOCS.with(|a| {
            String::from_utf8_lossy(&a.borrow()[&(out as usize)][..n as usize]).into_owned()
        });
        assert!(msg.contains("unknown backend code"), "{msg}");
        vokra_wasm_free(out, 256);

        // A non-registry PCM pointer is an explicit error.
        let s = vokra_wasm_session_destroy(0);
        let _ = s;
        let bogus = [0u8; 8];
        assert_eq!(vokra_wasm_transcribe(1, bogus.as_ptr(), 2), -1);
    }

    /// A garbage GGUF fails with a parse error (and the buffer ownership is
    /// consumed — no double-free possible afterwards).
    #[test]
    fn garbage_gguf_is_parse_error() {
        let p = vokra_wasm_alloc(8);
        // Buffer content is zeroed — not a GGUF magic.
        assert_eq!(vokra_wasm_session_create(p, 8, 0), 0);
        let out = vokra_wasm_alloc(512);
        let n = vokra_wasm_last_error_read(out, 512);
        let msg = ALLOCS.with(|a| {
            String::from_utf8_lossy(&a.borrow()[&(out as usize)][..n as usize]).into_owned()
        });
        assert!(msg.contains("GGUF parse failed"), "{msg}");
        vokra_wasm_free(out, 512);
        // The GGUF buffer was consumed; freeing it again is a no-op.
        vokra_wasm_free(p, 8);
    }
}
