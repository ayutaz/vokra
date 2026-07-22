//! Raw Objective-C runtime + CoreML / Foundation FFI (Apple targets only).
//!
//! This module is the **only** place that talks to the Objective-C runtime and
//! the CoreML framework, with hand-declared `unsafe extern` blocks — **no
//! `objc` / `objc2` / `objc2-core-ml` / `core-foundation` binding crate**
//! (the M2-01 red line, inherited; keeps the root `Cargo.lock` free of
//! non-`vokra-*` crates, NFR-DS-02). It is compiled only on `macos` / `ios`
//! (`#[cfg(any(target_os = "macos", target_os = "ios"))]`, applied by the
//! parent module) so Linux / Windows / WASM never see a framework link
//! (NFR-PT-01, all-target cross-build).
//!
//! # `objc_msgSend` calling convention (arm64)
//!
//! `objc_msgSend` is declared arg-less in C, but on AArch64 a variadic call and
//! a fixed-arity call do **not** share a register/stack layout, so it must be
//! invoked through a function pointer typed with the *real* signature of the
//! selector being sent. Every send below therefore `transmute`s the address of
//! `objc_msgSend` to the exact `extern "C" fn(Id, Sel, …) -> Ret` for that
//! call (the exact discipline `vokra-backend-metal/src/sys.rs` documents). Each
//! transmute + call carries a `// SAFETY:` note naming the selector and its
//! true signature.

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

// CoreML framework: the free C function that enumerates compute devices.
// `MLAllComputeDevices()` is `API_AVAILABLE(macos(14.0), ios(17.0), …)`
// (SDK header `CoreML.framework/Headers/MLAllComputeDevices.h`; exported
// symbol `_MLAllComputeDevices`). It returns an autoreleased
// `NSArray<id<MLComputeDeviceProtocol>> *` of the devices CoreML may schedule
// onto — `MLCPUComputeDevice` / `MLGPUComputeDevice` /
// `MLNeuralEngineComputeDevice`.
#[link(name = "CoreML", kind = "framework")]
unsafe extern "C" {
    /// `NSArray<id<MLComputeDeviceProtocol>> *MLAllComputeDevices(void)`.
    /// Autoreleased (not owned by the caller). Empty/absent ANE simply means
    /// no `MLNeuralEngineComputeDevice` element is present.
    pub fn MLAllComputeDevices() -> Id;
}

// Foundation: linked so `objc_getClass("NSString")` / NSArray resolve. CoreML
// depends on Foundation already, but we request it explicitly. No C symbol is
// needed directly, hence the empty block carrying only the link.
#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}

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
// the selector it sends. Receiver + selector validity is the caller's contract
// (documented per call site in `probe.rs`).
// ---------------------------------------------------------------------------

/// `-(NSUInteger)sel` — zero-argument send returning an `NSUInteger`
/// (e.g. `NSArray.count`). `NSUInteger` is `usize` on arm64/x86-64.
///
/// # Safety
/// `recv` must be a valid `id` responding to the `NSUInteger`, zero-argument
/// `sel`.
#[inline]
pub unsafe fn send_usize(recv: Id, sel: Sel) -> usize {
    // SAFETY: `-(NSUInteger)sel` is `extern "C" fn(Id, Sel) -> usize` on arm64.
    let f: unsafe extern "C" fn(Id, Sel) -> usize =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees `recv` responds to `sel`.
    unsafe { f(recv, sel) }
}

/// `-(NSInteger)sel` — zero-argument send returning an `NSInteger`
/// (e.g. `MLNeuralEngineComputeDevice.totalCoreCount`). `NSInteger` is `isize`.
///
/// # Safety
/// `recv` must be a valid `id` responding to the `NSInteger`, zero-argument
/// `sel`.
#[inline]
pub unsafe fn send_isize(recv: Id, sel: Sel) -> isize {
    // SAFETY: `-(NSInteger)sel` is `extern "C" fn(Id, Sel) -> isize` on arm64.
    let f: unsafe extern "C" fn(Id, Sel) -> isize =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees `recv` responds to `sel`.
    unsafe { f(recv, sel) }
}

/// `-(id)sel:(NSUInteger)idx` — one integer argument, object return
/// (e.g. `NSArray objectAtIndex:`).
///
/// # Safety
/// `recv` must respond to the one-`NSUInteger`-argument, object-returning `sel`.
#[inline]
pub unsafe fn send_id_usize(recv: Id, sel: Sel, idx: usize) -> Id {
    // SAFETY: signature `extern "C" fn(Id, Sel, usize) -> Id` on arm64.
    let f: unsafe extern "C" fn(Id, Sel, usize) -> Id =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees `recv` responds to `sel` for a valid index.
    unsafe { f(recv, sel, idx) }
}

/// `-(BOOL)sel:(Class)cls` — used for `isKindOfClass:`.
///
/// # Safety
/// `recv` must respond to the `BOOL`-returning, one-`Class`-argument `sel`;
/// `cls` a valid `Class` (or null).
#[inline]
pub unsafe fn send_bool_class(recv: Id, sel: Sel, cls: Class) -> bool {
    // SAFETY: `-(BOOL)sel:(Class)` is `extern "C" fn(Id, Sel, Class) -> bool`
    // on arm64 (BOOL is C `_Bool`, one byte, matching Rust `bool`).
    let f: unsafe extern "C" fn(Id, Sel, Class) -> bool =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees `recv` responds to `sel`.
    unsafe { f(recv, sel, cls) }
}
