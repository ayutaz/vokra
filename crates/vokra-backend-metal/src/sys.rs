//! Raw Objective-C runtime + Metal / Foundation FFI (Apple targets only).
//!
//! This module is the **only** place that talks to the Objective-C runtime and
//! the Metal framework, and it does so with hand-declared `unsafe extern`
//! blocks — **no `metal` / `objc2` / `objc` / `core-foundation` binding crate**
//! (M2-01 red line; keeps the root `Cargo.lock` free of non-`vokra-*` crates,
//! NFR-DS-02). It is compiled only on `macos` / `ios`
//! (`#[cfg(any(target_os = "macos", target_os = "ios"))]`, applied by the
//! parent module) so Linux / Windows / WASM never see a framework link
//! (NFR-PT-01, all-target cross-build).
//!
//! # `objc_msgSend` calling convention (arm64)
//!
//! `objc_msgSend` is declared variadic in C, but on AArch64 a variadic call and
//! a fixed-arity call do **not** share a register/stack layout, so it must be
//! invoked through a function pointer typed with the *real* signature of the
//! selector being sent. Every send below therefore `transmute`s the address of
//! `objc_msgSend` to the exact `extern "C" fn(Id, Sel, ...) -> Ret` for that
//! call. AArch64 needs no `objc_msgSend_stret` variant here (no struct is
//! *returned* by value; the only by-value struct — [`MtlSize`] — is a >16-byte
//! composite passed *indirectly* per AAPCS64, which rustc's own `extern "C"`
//! lowering matches). Each transmute + call carries a `// SAFETY:` note naming
//! the selector and its true signature.

use core::ffi::{c_char, c_void};

/// Objective-C object pointer (`id`).
pub type Id = *mut c_void;
/// Objective-C selector (`SEL`).
pub type Sel = *const c_void;
/// Objective-C class object (`Class`), itself an `id`.
pub type Class = *mut c_void;

// Objective-C runtime. `std` already links `libobjc` transitively, but we name
// it explicitly so the symbols resolve regardless of link order.
#[link(name = "objc", kind = "dylib")]
unsafe extern "C" {
    /// Looks up a registered class by name (`objc_getClass`). Null if absent.
    pub fn objc_getClass(name: *const c_char) -> Class;
    /// Registers / returns the selector for a method name (`sel_registerName`).
    pub fn sel_registerName(name: *const c_char) -> Sel;
    /// The message dispatcher. Declared arg-less on purpose: never called
    /// directly — its address is `transmute`d to each call's real signature.
    pub fn objc_msgSend();
    /// Pushes a fresh autorelease pool; returns the pool token.
    pub fn objc_autoreleasePoolPush() -> *mut c_void;
    /// Pops (drains) the autorelease pool created by the matching push.
    pub fn objc_autoreleasePoolPop(pool: *mut c_void);
}

// Metal framework: the one free C function that bootstraps everything else.
#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    /// Returns the system default `MTLDevice` (`id`), or null if the host has
    /// no Metal-capable GPU. Follows the Create Rule: the returned object is
    /// owned (+1) by the caller.
    pub fn MTLCreateSystemDefaultDevice() -> Id;
}

// Foundation: linked so `objc_getClass("NSString")` / `NSError` resolve. Metal
// depends on Foundation already, but we request it explicitly. No C symbol is
// needed from it directly, hence the empty block carrying only the link.
#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}

/// Metal's `MTLSize` — a triple of `NSUInteger`. 24 bytes (> 16), so AAPCS64
/// passes it *indirectly* (pointer to a caller copy); rustc's `extern "C"`
/// lowering of this `#[repr(C)]` struct matches, so passing it by value through
/// the transmuted `dispatchThreadgroups:threadsPerThreadgroup:` pointer is ABI-
/// correct.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MtlSize {
    /// Extent along x.
    pub width: usize,
    /// Extent along y.
    pub height: usize,
    /// Extent along z.
    pub depth: usize,
}

/// `MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache` = 0.
///
/// On Apple silicon the CPU and GPU share one physical memory pool, so a
/// `Shared` buffer's [`buffer_contents`] pointer is directly readable/writable
/// by the host with no explicit copy (the low-copy path this slice uses).
pub const STORAGE_MODE_SHARED: usize = 0;

/// `MTLGPUFamily` raw values used by the probe. Apple1..=Apple9 are contiguous
/// (1001..=1009); Metal3 is the feature-set umbrella (5001).
pub mod gpu_family {
    /// First Apple-silicon GPU family (`MTLGPUFamilyApple1`).
    pub const APPLE1: isize = 1001;
    /// Highest Apple GPU family this probe scans for (`MTLGPUFamilyApple9`).
    pub const APPLE9: isize = 1009;
    /// The Metal 3 feature-set family (`MTLGPUFamilyMetal3`).
    pub const METAL3: isize = 5001;
}

/// Interns an Objective-C selector from a NUL-terminated byte string.
///
/// `name` **must** end in `\0` (call sites pass `b"...\0"` literals). Returns a
/// process-lifetime `SEL`.
///
/// # Safety
/// `name` must point to a valid NUL-terminated C string.
#[inline]
pub unsafe fn sel(name: &[u8]) -> Sel {
    debug_assert_eq!(
        name.last(),
        Some(&0),
        "selector literal must be NUL-terminated"
    );
    // SAFETY: caller guarantees `name` is a valid NUL-terminated C string;
    // `sel_registerName` copies it and returns a permanent selector.
    unsafe { sel_registerName(name.as_ptr() as *const c_char) }
}

/// Looks up an Objective-C class by NUL-terminated name (null if not loaded).
///
/// # Safety
/// `name` must point to a valid NUL-terminated C string.
#[inline]
pub unsafe fn class(name: &[u8]) -> Class {
    debug_assert_eq!(
        name.last(),
        Some(&0),
        "class name literal must be NUL-terminated"
    );
    // SAFETY: caller guarantees `name` is a valid NUL-terminated C string.
    unsafe { objc_getClass(name.as_ptr() as *const c_char) }
}

// ---------------------------------------------------------------------------
// Typed `objc_msgSend` senders.
//
// Each helper transmutes the `objc_msgSend` address to the exact signature of
// the selector it sends. Receiver + selector validity is the caller's
// contract (documented per call site in the higher-level modules).
// ---------------------------------------------------------------------------

/// `-(id)sel` — zero-argument send returning an object pointer.
///
/// # Safety
/// `recv` must be a valid `id` (or null) that responds to `sel` with an
/// object-returning, zero-argument method.
#[inline]
pub unsafe fn send_id(recv: Id, sel: Sel) -> Id {
    // SAFETY: `objc_msgSend` for a `-(id)sel` method has signature
    // `extern "C" fn(Id, Sel) -> Id` on arm64; both are thin fn pointers.
    let f: unsafe extern "C" fn(Id, Sel) -> Id =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees `recv` responds to `sel`.
    unsafe { f(recv, sel) }
}

/// `-(void)sel` — zero-argument send with no return value (e.g. `release`,
/// `commit`, `waitUntilCompleted`, `endEncoding`).
///
/// # Safety
/// `recv` must be a valid `id` that responds to the void, zero-argument `sel`.
#[inline]
pub unsafe fn send_void(recv: Id, sel: Sel) {
    // SAFETY: `-(void)sel` is `extern "C" fn(Id, Sel)` on arm64.
    let f: unsafe extern "C" fn(Id, Sel) =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees `recv` responds to `sel`.
    unsafe { f(recv, sel) }
}

/// `-(void *)sel` — zero-argument send returning a raw pointer (e.g. a buffer's
/// `contents`, an NSString's `UTF8String`).
///
/// # Safety
/// `recv` must be a valid `id` responding to the pointer-returning `sel`.
#[inline]
pub unsafe fn send_ptr(recv: Id, sel: Sel) -> *mut c_void {
    // SAFETY: `-(void *)sel` is `extern "C" fn(Id, Sel) -> *mut c_void`.
    let f: unsafe extern "C" fn(Id, Sel) -> *mut c_void =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees `recv` responds to `sel`.
    unsafe { f(recv, sel) }
}

/// `-(BOOL)sel:(NSInteger)arg` — used for `supportsFamily:`.
///
/// # Safety
/// `recv` must respond to the `BOOL`-returning, one-`NSInteger`-argument `sel`.
#[inline]
pub unsafe fn send_bool_isize(recv: Id, sel: Sel, arg: isize) -> bool {
    // SAFETY: `-(BOOL)sel:(NSInteger)` is `extern "C" fn(Id, Sel, isize) -> bool`
    // on arm64 (BOOL is C `_Bool`, one byte, matching Rust `bool`).
    let f: unsafe extern "C" fn(Id, Sel, isize) -> bool =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees `recv` responds to `sel`.
    unsafe { f(recv, sel, arg) }
}

/// `+(id)sel:(const char *)arg` — `+[NSString stringWithUTF8String:]`.
///
/// # Safety
/// `recv` must be a class/object responding to `sel`; `cstr` a valid
/// NUL-terminated C string.
#[inline]
pub unsafe fn send_id_cstr(recv: Id, sel: Sel, cstr: *const c_char) -> Id {
    // SAFETY: signature `extern "C" fn(Id, Sel, *const c_char) -> Id`.
    let f: unsafe extern "C" fn(Id, Sel, *const c_char) -> Id =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees `recv` responds to `sel` and `cstr` is valid.
    unsafe { f(recv, sel, cstr) }
}

/// `-(id)sel:(id)arg` — one object argument, object return (e.g.
/// `newFunctionWithName:`).
///
/// # Safety
/// `recv` must respond to the one-`id`-argument, object-returning `sel`.
#[inline]
pub unsafe fn send_id_id(recv: Id, sel: Sel, arg: Id) -> Id {
    // SAFETY: signature `extern "C" fn(Id, Sel, Id) -> Id`.
    let f: unsafe extern "C" fn(Id, Sel, Id) -> Id =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees `recv` responds to `sel`.
    unsafe { f(recv, sel, arg) }
}

/// `-(void)sel:(id)arg` — one object argument, no return (e.g.
/// `setComputePipelineState:`).
///
/// # Safety
/// `recv` must respond to the one-`id`-argument, void `sel`.
#[inline]
pub unsafe fn send_void_id(recv: Id, sel: Sel, arg: Id) {
    // SAFETY: signature `extern "C" fn(Id, Sel, Id)`.
    let f: unsafe extern "C" fn(Id, Sel, Id) =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees `recv` responds to `sel`.
    unsafe { f(recv, sel, arg) }
}

/// `-(id)newLibraryWithSource:(NSString*)src options:(id)opts error:(NSError**)err`.
///
/// # Safety
/// `recv` a valid `MTLDevice`; `src` a valid NSString; `opts` a valid id or
/// null; `err` a valid `*mut Id` or null.
#[inline]
pub unsafe fn send_new_library(recv: Id, sel: Sel, src: Id, opts: Id, err: *mut Id) -> Id {
    // SAFETY: signature `extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id`.
    let f: unsafe extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees the argument contract above.
    unsafe { f(recv, sel, src, opts, err) }
}

/// `-(id)newComputePipelineStateWithFunction:(id)fn error:(NSError**)err`.
///
/// # Safety
/// `recv` a valid `MTLDevice`; `func` a valid `MTLFunction`; `err` a valid
/// `*mut Id` or null.
#[inline]
pub unsafe fn send_new_pipeline(recv: Id, sel: Sel, func: Id, err: *mut Id) -> Id {
    // SAFETY: signature `extern "C" fn(Id, Sel, Id, *mut Id) -> Id`.
    let f: unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees the argument contract above.
    unsafe { f(recv, sel, func, err) }
}

/// `-(id)newBufferWithBytes:(const void*)ptr length:(NSUInteger)len options:(NSUInteger)opt`.
///
/// # Safety
/// `recv` a valid `MTLDevice`; `bytes` points to at least `len` readable bytes.
#[inline]
pub unsafe fn send_new_buffer_bytes(
    recv: Id,
    sel: Sel,
    bytes: *const c_void,
    len: usize,
    opt: usize,
) -> Id {
    // SAFETY: signature `extern "C" fn(Id, Sel, *const c_void, usize, usize) -> Id`.
    let f: unsafe extern "C" fn(Id, Sel, *const c_void, usize, usize) -> Id =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees the argument contract above.
    unsafe { f(recv, sel, bytes, len, opt) }
}

/// `-(id)newBufferWithLength:(NSUInteger)len options:(NSUInteger)opt`.
///
/// # Safety
/// `recv` must be a valid `MTLDevice`.
#[inline]
pub unsafe fn send_new_buffer_len(recv: Id, sel: Sel, len: usize, opt: usize) -> Id {
    // SAFETY: signature `extern "C" fn(Id, Sel, usize, usize) -> Id`.
    let f: unsafe extern "C" fn(Id, Sel, usize, usize) -> Id =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees `recv` is a valid device.
    unsafe { f(recv, sel, len, opt) }
}

/// `-(void)setBuffer:(id)buf offset:(NSUInteger)off atIndex:(NSUInteger)idx`.
///
/// # Safety
/// `recv` a valid compute encoder; `buf` a valid `MTLBuffer`.
#[inline]
pub unsafe fn send_set_buffer(recv: Id, sel: Sel, buf: Id, off: usize, idx: usize) {
    // SAFETY: signature `extern "C" fn(Id, Sel, Id, usize, usize)`.
    let f: unsafe extern "C" fn(Id, Sel, Id, usize, usize) =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees the argument contract above.
    unsafe { f(recv, sel, buf, off, idx) }
}

/// `-(void)setBytes:(const void*)ptr length:(NSUInteger)len atIndex:(NSUInteger)idx`.
///
/// # Safety
/// `recv` a valid compute encoder; `bytes` points to at least `len` bytes.
#[inline]
pub unsafe fn send_set_bytes(recv: Id, sel: Sel, bytes: *const c_void, len: usize, idx: usize) {
    // SAFETY: signature `extern "C" fn(Id, Sel, *const c_void, usize, usize)`.
    let f: unsafe extern "C" fn(Id, Sel, *const c_void, usize, usize) =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees the argument contract above.
    unsafe { f(recv, sel, bytes, len, idx) }
}

/// `-(void)dispatchThreadgroups:(MTLSize)grid threadsPerThreadgroup:(MTLSize)tg`.
///
/// # Safety
/// `recv` must be a valid compute encoder with a pipeline state set.
#[inline]
pub unsafe fn send_dispatch(recv: Id, sel: Sel, grid: MtlSize, tg: MtlSize) {
    // SAFETY: signature `extern "C" fn(Id, Sel, MtlSize, MtlSize)`. Each 24-byte
    // `MtlSize` is passed indirectly per AAPCS64; rustc's `extern "C"` lowering
    // of `#[repr(C)]` matches Metal's expectation.
    let f: unsafe extern "C" fn(Id, Sel, MtlSize, MtlSize) =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees `recv` is a ready compute encoder.
    unsafe { f(recv, sel, grid, tg) }
}

/// Reads an `NSString` (`id`) as an owned Rust `String` via `-UTF8String`.
///
/// Returns `None` if `nsstring` is null or its UTF-8 pointer is null.
///
/// # Safety
/// `nsstring` must be null or a valid `NSString`.
pub unsafe fn nsstring_to_string(nsstring: Id) -> Option<String> {
    if nsstring.is_null() {
        return None;
    }
    // SAFETY: `UTF8String` is a valid `-(const char*)` selector on NSString.
    let utf8 = unsafe { send_ptr(nsstring, sel(b"UTF8String\0")) } as *const c_char;
    if utf8.is_null() {
        return None;
    }
    // SAFETY: `UTF8String` returns a valid NUL-terminated C string owned by the
    // autoreleased NSString, alive for this call; we copy it into an owned
    // String before returning (no dangling borrow).
    let cstr = unsafe { core::ffi::CStr::from_ptr(utf8) };
    Some(cstr.to_string_lossy().into_owned())
}
