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

/// `-(id)sel` — zero-argument object return (`alloc`, `init`, `retain`, or a
/// property getter).
///
/// # Safety
/// `recv` must respond to the zero-argument, object-returning `sel`.
#[inline]
pub unsafe fn send_id(recv: Id, sel: Sel) -> Id {
    // SAFETY: the selector has signature `extern "C" fn(Id, Sel) -> Id`.
    let f: unsafe extern "C" fn(Id, Sel) -> Id =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees receiver and selector validity.
    unsafe { f(recv, sel) }
}

/// `-(void)sel` — zero-argument void return (`release`).
///
/// # Safety
/// `recv` must respond to the zero-argument, void-returning `sel`.
#[inline]
pub unsafe fn send_void(recv: Id, sel: Sel) {
    // SAFETY: the selector has signature `extern "C" fn(Id, Sel)`.
    let f: unsafe extern "C" fn(Id, Sel) =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees receiver and selector validity.
    unsafe { f(recv, sel) }
}

/// `-(id)sel:(id)arg` — one-object-argument object return.
///
/// # Safety
/// `recv` must respond to `sel` with the stated signature and `arg` must be a
/// valid object for it.
#[inline]
pub unsafe fn send_id_id(recv: Id, sel: Sel, arg: Id) -> Id {
    // SAFETY: the selector has signature `extern "C" fn(Id, Sel, Id) -> Id`.
    let f: unsafe extern "C" fn(Id, Sel, Id) -> Id =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees receiver, selector and argument validity.
    unsafe { f(recv, sel, arg) }
}

/// `-(id)sel:(id)arg error:(NSError **)error`.
///
/// # Safety
/// `recv` and `arg` must be valid for `sel`; `error` must be writable.
#[inline]
pub unsafe fn send_id_id_error(recv: Id, sel: Sel, arg: Id, error: *mut Id) -> Id {
    // SAFETY: exact Objective-C signature `(id, SEL, id, NSError **) -> id`.
    let f: unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees all arguments match the selector contract.
    unsafe { f(recv, sel, arg, error) }
}

/// `+(id)sel:(id)a configuration:(id)b error:(NSError **)error`.
///
/// # Safety
/// Receiver, selector, object arguments and writable error slot must match the
/// declared Objective-C method.
#[inline]
pub unsafe fn send_id_id_id_error(recv: Id, sel: Sel, a: Id, b: Id, error: *mut Id) -> Id {
    // SAFETY: exact Objective-C signature `(id, SEL, id, id, NSError **) -> id`.
    let f: unsafe extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees all arguments match the selector contract.
    unsafe { f(recv, sel, a, b, error) }
}

/// `-(id)sel:(id)shape dataType:(NSInteger)dtype error:(NSError **)error`.
///
/// # Safety
/// Receiver, selector, shape and writable error slot must match the declared
/// Objective-C method.
#[inline]
pub unsafe fn send_id_id_isize_error(
    recv: Id,
    sel: Sel,
    shape: Id,
    dtype: isize,
    error: *mut Id,
) -> Id {
    // SAFETY: exact Objective-C signature `(id, SEL, id, NSInteger, NSError **) -> id`.
    let f: unsafe extern "C" fn(Id, Sel, Id, isize, *mut Id) -> Id =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees all arguments match the selector contract.
    unsafe { f(recv, sel, shape, dtype, error) }
}

/// `+(id)sel:(const char *)utf8`.
///
/// # Safety
/// `utf8` must point at a valid NUL-terminated UTF-8 string.
#[inline]
pub unsafe fn send_id_cstr(recv: Id, sel: Sel, utf8: *const c_char) -> Id {
    // SAFETY: exact Objective-C signature `(id, SEL, const char *) -> id`.
    let f: unsafe extern "C" fn(Id, Sel, *const c_char) -> Id =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees receiver, selector and C-string validity.
    unsafe { f(recv, sel, utf8) }
}

/// `+(id)sel:(unsigned long long)value`.
///
/// # Safety
/// `recv` must respond to the stated numeric factory selector.
#[inline]
pub unsafe fn send_id_u64(recv: Id, sel: Sel, value: u64) -> Id {
    // SAFETY: exact Objective-C signature `(id, SEL, unsigned long long) -> id`.
    let f: unsafe extern "C" fn(Id, Sel, u64) -> Id =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees receiver and selector validity.
    unsafe { f(recv, sel, value) }
}

/// `+(id)sel:(const id *)objects count:(NSUInteger)count`.
///
/// # Safety
/// `objects` must reference `count` valid object pointers.
#[inline]
pub unsafe fn send_id_ptr_usize(recv: Id, sel: Sel, objects: *const Id, count: usize) -> Id {
    // SAFETY: exact Objective-C signature `(id, SEL, const id *, NSUInteger) -> id`.
    let f: unsafe extern "C" fn(Id, Sel, *const Id, usize) -> Id =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees pointer/count validity.
    unsafe { f(recv, sel, objects, count) }
}

/// `+(id)sel:(const id *)objects forKeys:(const id *)keys count:(NSUInteger)count`.
///
/// # Safety
/// Both pointer arrays must reference `count` valid objects.
#[inline]
pub unsafe fn send_id_ptr_ptr_usize(
    recv: Id,
    sel: Sel,
    objects: *const Id,
    keys: *const Id,
    count: usize,
) -> Id {
    // SAFETY: exact Objective-C signature
    // `(id, SEL, const id *, const id *, NSUInteger) -> id`.
    let f: unsafe extern "C" fn(Id, Sel, *const Id, *const Id, usize) -> Id =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees pointer/count validity.
    unsafe { f(recv, sel, objects, keys, count) }
}

/// `+(id)sel:(id)path isDirectory:(BOOL)is_directory`.
///
/// # Safety
/// Receiver, selector and path object must match `NSURL.fileURLWithPath:`.
#[inline]
pub unsafe fn send_id_id_bool(recv: Id, sel: Sel, path: Id, is_directory: bool) -> Id {
    // SAFETY: exact Objective-C signature `(id, SEL, id, BOOL) -> id`.
    let f: unsafe extern "C" fn(Id, Sel, Id, bool) -> Id =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees receiver, selector and path validity.
    unsafe { f(recv, sel, path, is_directory) }
}

/// `-(void)sel:(NSInteger)value`.
///
/// # Safety
/// Receiver must respond to the stated integer setter selector.
#[inline]
pub unsafe fn send_void_isize(recv: Id, sel: Sel, value: isize) {
    // SAFETY: exact Objective-C signature `(id, SEL, NSInteger) -> void`.
    let f: unsafe extern "C" fn(Id, Sel, isize) =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees receiver and selector validity.
    unsafe { f(recv, sel, value) }
}

/// `-(void *)sel` — pointer-valued property getter (`MLMultiArray.dataPointer`).
///
/// # Safety
/// Receiver must respond to the pointer-returning selector.
#[inline]
pub unsafe fn send_ptr(recv: Id, sel: Sel) -> *mut c_void {
    // SAFETY: exact Objective-C signature `(id, SEL) -> void *`.
    let f: unsafe extern "C" fn(Id, Sel) -> *mut c_void =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees receiver and selector validity.
    unsafe { f(recv, sel) }
}

/// `-(unsigned long long)sel` — numeric property getter on `NSNumber`.
///
/// # Safety
/// Receiver must respond to the unsigned-64 selector.
#[inline]
pub unsafe fn send_u64(recv: Id, sel: Sel) -> u64 {
    // SAFETY: exact Objective-C signature `(id, SEL) -> unsigned long long`.
    let f: unsafe extern "C" fn(Id, Sel) -> u64 =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees receiver and selector validity.
    unsafe { f(recv, sel) }
}

/// `-(const char *)sel` — C-string property getter (`NSString.UTF8String`).
///
/// # Safety
/// Receiver must be a valid string responding to the selector.
#[inline]
pub unsafe fn send_cstr(recv: Id, sel: Sel) -> *const c_char {
    // SAFETY: exact Objective-C signature `(id, SEL) -> const char *`.
    let f: unsafe extern "C" fn(Id, Sel) -> *const c_char =
        unsafe { core::mem::transmute(objc_msgSend as unsafe extern "C" fn()) };
    // SAFETY: caller guarantees receiver and selector validity.
    unsafe { f(recv, sel) }
}
