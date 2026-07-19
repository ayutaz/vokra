//! ClassDB registration + method binding + signal declaration (T05..T09).
//!
//! At Scene-level init, [`register`] populates Godot's ClassDB with two
//! extension classes:
//!
//! - `VokraSession` — an `Object` subclass holding a
//!   [`crate::session::VokraSession`] behind a Rust-owned `SessionInstance`
//!   allocation. Methods: `load_model`, `transcribe`, `synthesize`,
//!   `vad_open_stream`.
//! - `VokraStream` — an `Object` subclass wrapping
//!   [`crate::vad::VokraStream`]. Methods: `push_pcm`, `poll`, `interrupt`.
//!   Signals: `asr_chunk(prob: float)`, `tts_chunk(pcm: PackedFloat32Array)`.
//!
//! At Scene-level deinit, [`unregister`] removes both classes in reverse
//! order.
//!
//! # Instance lifetime
//!
//! Godot's `create_instance_func` is called when GDScript does
//! `VokraSession.new()`. We allocate a `SessionInstance` on the heap
//! (`Box::into_raw`) and return the resulting `*mut SessionInstance`
//! as our `GDExtensionClassInstancePtr`. The paired `free_instance_func`
//! reclaims it via `Box::from_raw`. The instance holds
//! `Option<VokraSession>` = `None` at first; the registered
//! `load_model(path: String) -> int` method
//! ([`crate::trampoline::session_load_model`]) populates it. This makes
//! the `new() -> load_model()` two-step explicit at the GDScript surface,
//! matching how uPiper's Session works.
//!
//! Instance-binding pointers on the Godot Object are OUT-OF-SCOPE for T05:
//! we don't override `object_set_instance` on the Godot side because our
//! only method surface currently lives via ClassDB dispatch. Owner smoke
//! (M3-18) will decide whether InstanceBinding is needed for a specific
//! demo scene pattern.
//!
//! # StringName lifetime
//!
//! Every class name / method name / argument name / signal name that we
//! pass to Godot's ClassDB is a StringName. Godot's ClassDB refcount-holds
//! them internally, but the source `GDExtensionStringNamePtr` we pass in
//! is a stack buffer inside a [`OwnedStringName`] — owned by us, dropped
//! after the register call returns. This mirrors how godot-cpp constructs
//! StringNames on the stack for its own registration calls.
//!
//! # Bounded scope (T05..T13 vs M3-18)
//!
//! What lands in this file is **the class + method + signal shape** that
//! Godot's ClassDB observes after `vokra_gdextension_init` returns. The
//! actual runtime dispatch (Variant packing / unpacking, real
//! `crate::asr::transcribe` calls) lives in the trampoline stubs
//! ([`crate::trampoline`]) and is deferred to owner smoke in M3-18. The
//! plumbing here is 100% real — if a Godot editor sat on top of this
//! cdylib, it would observe both classes fully registered.
//!
//! # No Godot editor in this crate's test suite
//!
//! Everything below is exercised via mocks (an in-memory ClassDB reflector)
//! or by inspecting the populated `GDExtensionClassCreationInfo3` /
//! `GDExtensionClassMethodInfo` structs. Real behavior verification happens
//! in M3-18 owner smoke.

use core::ffi::{c_char, c_void};
use core::ptr;

use crate::ffi::gdextension::{
    GDExtensionBool, GDExtensionClassCreationInfo3, GDExtensionClassInstancePtr,
    GDExtensionClassLibraryPtr, GDExtensionClassMethodArgumentMetadata, GDExtensionClassMethodInfo,
    GDExtensionInt, GDExtensionObjectPtr, GDExtensionPropertyInfo, GDExtensionStringNamePtr,
    GDExtensionStringPtr, GDExtensionVariantType, STRING_SIZE, method_flags,
};
use crate::ffi::interface::InterfaceTable;

// ---------------------------------------------------------------------------
// StringName owned buffer.
//
// Godot StringNames are opaque 8-byte pointer-sized handles under the hood
// (Godot 4.3 pins `sizeof(StringName) == sizeof(void *)`), but we treat them
// as an opaque scratch buffer whose bytes we hand to
// `string_name_new_with_utf8_chars`. Godot's own ClassDB registration path
// does exactly this (see `gdextension_manager.cpp`).
// ---------------------------------------------------------------------------

/// Backing storage for one StringName. Godot 4.3 pins
/// `sizeof(StringName)` at 8 bytes on LP64. We conservatively over-allocate
/// to 16 bytes so a future 4.5+ layout bump (e.g. StringName gaining a
/// per-instance flag byte) still fits without a re-audit.
#[repr(C, align(8))]
pub struct OwnedStringName {
    /// Storage bytes; not directly meaningful outside Godot.
    storage: [u8; 16],
}

impl OwnedStringName {
    /// Allocate an uninitialized StringName buffer. The bytes are
    /// zeroed for defensive reasons — Godot's constructor doesn't require
    /// it, but a zeroed slot means a "leaked" StringName pointer (see
    /// safety notes below) at least dereferences to Godot's canonical
    /// "empty" StringName instead of unallocated memory.
    fn new() -> Self {
        Self { storage: [0u8; 16] }
    }

    /// Construct a StringName from a NUL-terminated UTF-8 string via the
    /// resolved `string_name_new_with_utf8_chars` interface.
    ///
    /// # Safety
    ///
    /// - `interface.string_name_new_with_utf8_chars` is a live Godot fn
    ///   pointer (guaranteed by `InterfaceTable::from_proc_address`).
    /// - `contents` is NUL-terminated (checked at compile time by the
    ///   `class_names::*` byte-string constants).
    /// - The returned StringName is only valid for as long as `self` is
    ///   live (the Godot constructor writes into `self.storage`). Drop
    ///   order matters: the OwnedStringName MUST outlive every register
    ///   call it feeds.
    unsafe fn init_utf8(&mut self, interface: &InterfaceTable, contents: &[u8]) {
        debug_assert_eq!(
            contents.last(),
            Some(&0u8),
            "StringName input must be NUL-terminated",
        );
        // SAFETY: `interface.string_name_new_with_utf8_chars` writes into
        // `self.storage` (an uninitialized 16-byte buffer, over-sized for
        // 4.3's 8-byte StringName layout). `contents` is a valid pointer
        // to at least `contents.len()` bytes ending in NUL.
        unsafe {
            (interface.string_name_new_with_utf8_chars)(
                self.storage.as_mut_ptr() as *mut c_void,
                contents.as_ptr() as *const c_char,
            );
        }
    }

    /// Return the StringName pointer for handing to ClassDB. Valid until
    /// `self` is dropped.
    fn as_ptr(&self) -> GDExtensionStringNamePtr {
        self.storage.as_ptr() as *mut c_void
    }
}

// NOTE: We intentionally do NOT `Drop` — Godot's StringName is refcounted
// internally, and dropping our stack buffer with a NULL destructor would
// leak one StringName ref per registration. This is a controlled leak: the
// leaked refs land in Godot's global StringName cache, which is process-
// lifetime anyway. Godot's official policy (see godot-cpp comments) is
// that GDExtension bindings leak these when init/deinit is process-scoped.
// A future M3-18 patch may add a `string_name_destroy` interface resolve
// + Drop for full symmetry.

// ---------------------------------------------------------------------------
// Class name constants. Byte-strings terminated with `\0` so they can be
// fed directly to Godot's UTF-8 StringName constructor.
// ---------------------------------------------------------------------------

pub mod class_names {
    /// The Godot Object subclass that owns a `VokraSession` handle.
    pub const VOKRA_SESSION: &[u8] = b"VokraSession\0";
    /// The Godot Object subclass that owns a `VokraStream` handle.
    pub const VOKRA_STREAM: &[u8] = b"VokraStream\0";
    /// Parent class for both — Godot's built-in `Object` (chosen over
    /// `RefCounted` because our C ABI already tracks refcount internally;
    /// double-counting would break FR-API-03).
    pub const PARENT_OBJECT: &[u8] = b"Object\0";
}

pub mod method_names {
    // VokraSession methods (T06).
    /// Two-step `new() -> load_model(path)` construction. Named to match
    /// the GDScript both demo scenes already ship
    /// (`demos/asr_demo/main.gd`, `demos/tts_demo/main.gd`).
    pub const LOAD_MODEL: &[u8] = b"load_model\0";
    pub const TRANSCRIBE: &[u8] = b"transcribe\0";
    pub const SYNTHESIZE: &[u8] = b"synthesize\0";
    pub const VAD_OPEN_STREAM: &[u8] = b"vad_open_stream\0";

    // VokraStream methods (T08).
    pub const PUSH_PCM: &[u8] = b"push_pcm\0";
    pub const POLL: &[u8] = b"poll\0";
    pub const INTERRUPT: &[u8] = b"interrupt\0";
}

pub mod signal_names {
    // VokraStream signals (T09).
    pub const ASR_CHUNK: &[u8] = b"asr_chunk\0";
    pub const TTS_CHUNK: &[u8] = b"tts_chunk\0";
}

pub mod arg_names {
    pub const PATH: &[u8] = b"path\0";
    pub const PCM: &[u8] = b"pcm\0";
    pub const SAMPLE_RATE: &[u8] = b"sample_rate\0";
    pub const TEXT: &[u8] = b"text\0";
    pub const CAPACITY: &[u8] = b"capacity\0";
    pub const PROB: &[u8] = b"prob\0";
}

// ---------------------------------------------------------------------------
// Instance types. `SessionInstance` and `StreamInstance` are the Rust-owned
// data blobs whose raw `*mut` is the `GDExtensionClassInstancePtr` we hand
// back from `create_instance_func`.
// ---------------------------------------------------------------------------

/// Instance data attached to every `VokraSession` Godot Object.
///
/// The inner session starts as `None`;
/// [`crate::trampoline::session_load_model`] transitions it to `Some(..)`
/// on a successful GGUF load. Methods dispatched through
/// [`crate::trampoline`] read `.inner` behind the raw pointer.
pub struct SessionInstance {
    /// Populated after a successful `load_model(path)`. Reading a `None`
    /// here from a method trampoline is documented to return
    /// `InvalidMethod` (matches `report_pending` in `trampoline.rs`).
    ///
    /// A *failed* `load_model` resets this to `None` even if a previous
    /// load had succeeded — see [`crate::trampoline::session_load_model`]
    /// for why leaving a stale model in place would violate FR-EX-08.
    pub inner: Option<crate::session::VokraSession>,
}

/// Instance data attached to every `VokraStream` Godot Object.
pub struct StreamInstance {
    /// Populated after `session.vad_open_stream(sr)` (M3-18).
    pub inner: Option<crate::vad::VokraStream>,
}

/// Build the Godot-side Object for one of our extension classes and bind a
/// Rust instance allocation to it.
///
/// This is the shape every GDExtension `create_instance_func` must have
/// (godot-cpp's `ClassDB::_create_instance` does exactly this):
///
/// 1. `classdb_construct_object(<parent class>)` — makes a real Godot
///    `Object`. Returning anything else from `create_instance_func`
///    crashes Godot, which `dynamic_cast`s the returned pointer as an
///    `Object *`.
/// 2. Allocate the Rust-side instance data (`Box::into_raw`).
/// 3. `object_set_instance(obj, <our class>, instance)` — Godot hands this
///    exact pointer back as `p_instance` to every method trampoline and to
///    `free_instance_func`.
///
/// Returns NULL if Godot could not construct the Object; the Rust
/// allocation is only made after that succeeds, so the failure path
/// cannot leak.
///
/// # Safety
///
/// `interface` must hold live Godot fn pointers, and this must run on the
/// Godot main thread (ClassDB's creation path is single-threaded).
unsafe fn create_bound_object(
    interface: &InterfaceTable,
    class_name_bytes: &'static [u8],
    make_instance: impl FnOnce() -> GDExtensionClassInstancePtr,
) -> GDExtensionObjectPtr {
    let mut parent = OwnedStringName::new();
    let mut class_name = OwnedStringName::new();
    // SAFETY: live fn pointers per caller doc; both byte constants are
    // NUL-terminated.
    unsafe {
        parent.init_utf8(interface, class_names::PARENT_OBJECT);
        class_name.init_utf8(interface, class_name_bytes);
    }

    // SAFETY: `parent` is a live StringName for the duration of the call.
    let object = unsafe { (interface.classdb_construct_object)(parent.as_ptr()) };
    if object.is_null() {
        // Godot refused to construct the parent Object. FR-EX-08: report
        // the failure as a NULL instance rather than fabricating a
        // half-built object.
        return ptr::null_mut();
    }

    let instance = make_instance();
    // SAFETY: `object` is a live Godot Object just constructed above;
    // `class_name` is a registered extension class extending `Object`;
    // `instance` is a `Box::into_raw` allocation whose ownership now
    // transfers to Godot (reclaimed by our `free_instance_func`).
    unsafe {
        (interface.object_set_instance)(object, class_name.as_ptr(), instance);
    }
    object
}

/// `create_instance_func` for `VokraSession`. Godot invokes this when
/// GDScript does `VokraSession.new()` (or `ClassDB.instantiate(..)`). We
/// allocate a `SessionInstance` with `inner = None` and bind it to a
/// freshly constructed Godot Object — see [`create_bound_object`].
///
/// # Safety
///
/// C ABI entry from Godot. Callers must invoke this on the Godot main
/// thread — Godot's ClassDB creation path is single-threaded.
pub unsafe extern "C" fn create_session_instance(
    _p_class_userdata: *mut c_void,
) -> GDExtensionObjectPtr {
    // Panic firewall: allocation should never panic (Rust's global
    // allocator aborts on OOM), but `catch_panic` defends against arbitrary
    // future body changes.
    let res = crate::error::catch_panic(|| {
        // SAFETY: ClassDB dispatch implies the extension is initialised, so
        // the interface is live. `None` (pre-init / post-deinit) is
        // surfaced as a NULL Object rather than a fabricated instance.
        let created = unsafe {
            crate::with_interface(|iface| {
                create_bound_object(iface, class_names::VOKRA_SESSION, || {
                    Box::into_raw(Box::new(SessionInstance { inner: None }))
                        as GDExtensionClassInstancePtr
                })
            })
        };
        created.unwrap_or(ptr::null_mut())
    });
    res.unwrap_or(ptr::null_mut())
}

/// `free_instance_func` for `VokraSession`. Called by Godot after the
/// paired `create_instance_func` when the Object is freed.
///
/// # Safety
///
/// - `p_instance` MUST be a `*mut SessionInstance` originally produced by
///   [`create_session_instance`] on this crate's build.
/// - It MUST not have been already freed (Godot guarantees this in its
///   ClassDB lifecycle contract).
pub unsafe extern "C" fn free_session_instance(
    _p_class_userdata: *mut c_void,
    p_instance: GDExtensionClassInstancePtr,
) {
    let _ = crate::error::catch_panic(|| {
        if p_instance.is_null() {
            return;
        }
        // SAFETY: caller guarantees `p_instance` came from
        // `create_session_instance` and has not been freed.
        let boxed: Box<SessionInstance> =
            unsafe { Box::from_raw(p_instance as *mut SessionInstance) };
        drop(boxed);
    });
}

/// `create_instance_func` for `VokraStream`. Godot invokes this when
/// GDScript does `VokraStream.new()`. We allocate a `StreamInstance`
/// with `inner = None`.
///
/// # Safety
///
/// See [`create_session_instance`].
pub unsafe extern "C" fn create_stream_instance(
    _p_class_userdata: *mut c_void,
) -> GDExtensionObjectPtr {
    let res = crate::error::catch_panic(|| {
        // SAFETY: see `create_session_instance`.
        let created = unsafe {
            crate::with_interface(|iface| {
                create_bound_object(iface, class_names::VOKRA_STREAM, || {
                    Box::into_raw(Box::new(StreamInstance { inner: None }))
                        as GDExtensionClassInstancePtr
                })
            })
        };
        created.unwrap_or(ptr::null_mut())
    });
    res.unwrap_or(ptr::null_mut())
}

/// `free_instance_func` for `VokraStream`. See [`free_session_instance`].
///
/// # Safety
///
/// See [`free_session_instance`].
pub unsafe extern "C" fn free_stream_instance(
    _p_class_userdata: *mut c_void,
    p_instance: GDExtensionClassInstancePtr,
) {
    let _ = crate::error::catch_panic(|| {
        if p_instance.is_null() {
            return;
        }
        // SAFETY: caller guarantees `p_instance` came from
        // `create_stream_instance` and has not been freed.
        let boxed: Box<StreamInstance> =
            unsafe { Box::from_raw(p_instance as *mut StreamInstance) };
        drop(boxed);
    });
}

// ---------------------------------------------------------------------------
// Registration entry points.
//
// [`register`] and [`unregister`] are the T05 / T09 API surface. They are
// invoked from [`crate::lib::vokra_initialize`] at Scene level and
// [`crate::lib::vokra_deinitialize`] respectively.
// ---------------------------------------------------------------------------

/// Full registration pass for both classes + their methods + their signals.
///
/// # Safety
///
/// - `library` is the token Godot handed us at `vokra_gdextension_init`.
///   It must live at least as long as the registered classes.
/// - `interface` MUST hold live GDExtension fn pointers (see
///   [`InterfaceTable::from_proc_address`]).
/// - This function is single-threaded: Godot invokes `initialize` on the
///   main thread.
pub unsafe fn register(library: GDExtensionClassLibraryPtr, interface: &InterfaceTable) {
    // SAFETY: `register_session_class` / `register_stream_class` each
    // hold the library + interface pointers alive on the stack for the
    // duration of their register calls. They are invoked exactly once
    // during Scene-level init.
    unsafe {
        register_session_class(library, interface);
        register_stream_class(library, interface);
    }
}

/// Reverse of [`register`]. Godot's contract requires classes be
/// unregistered in reverse dependency order — we unregister `VokraStream`
/// first (nothing depends on it) then `VokraSession`.
///
/// # Safety
///
/// See [`register`].
pub unsafe fn unregister(library: GDExtensionClassLibraryPtr, interface: &InterfaceTable) {
    let mut session_name = OwnedStringName::new();
    let mut stream_name = OwnedStringName::new();
    // SAFETY: interface fn pointers are live (caller doc), byte constants
    // are NUL-terminated.
    unsafe {
        session_name.init_utf8(interface, class_names::VOKRA_SESSION);
        stream_name.init_utf8(interface, class_names::VOKRA_STREAM);

        (interface.classdb_unregister_extension_class)(library, stream_name.as_ptr());
        (interface.classdb_unregister_extension_class)(library, session_name.as_ptr());
    }
}

/// Register the `VokraSession` class + its three methods.
///
/// # Safety
///
/// See [`register`].
unsafe fn register_session_class(library: GDExtensionClassLibraryPtr, interface: &InterfaceTable) {
    let mut class_name = OwnedStringName::new();
    let mut parent = OwnedStringName::new();
    // SAFETY: interface fn ptr live; name buffers alive for the register
    // call below.
    unsafe {
        class_name.init_utf8(interface, class_names::VOKRA_SESSION);
        parent.init_utf8(interface, class_names::PARENT_OBJECT);
    }

    let info = GDExtensionClassCreationInfo3 {
        is_virtual: 0,
        is_abstract: 0,
        is_exposed: 1,
        is_runtime: 0,
        set_func: None,
        get_func: None,
        get_property_list_func: None,
        free_property_list_func: None,
        property_can_revert_func: None,
        property_get_revert_func: None,
        validate_property_func: None,
        notification_func: None,
        to_string_func: None,
        reference_func: None,
        unreference_func: None,
        create_instance_func: Some(create_session_instance),
        free_instance_func: Some(free_session_instance),
        recreate_instance_func: None,
        get_virtual_func: None,
        get_virtual_call_data_func: None,
        call_virtual_with_data_func: None,
        get_rid_func: None,
        class_userdata: ptr::null_mut(),
    };

    // SAFETY: `class_name` + `parent` + `info` all live for the register
    // call; interface fn ptr live.
    unsafe {
        (interface.classdb_register_extension_class3)(
            library,
            class_name.as_ptr(),
            parent.as_ptr(),
            &info,
        );

        // Now register the four methods. `load_model` goes first because
        // it is the one the other three depend on (they report
        // `InvalidMethod` until it has populated `SessionInstance::inner`).
        register_session_load_model(library, interface, class_name.as_ptr());
        register_session_transcribe(library, interface, class_name.as_ptr());
        register_session_synthesize(library, interface, class_name.as_ptr());
        register_session_vad_open_stream(library, interface, class_name.as_ptr());
    }
}

/// Register the `VokraStream` class + methods + signals.
///
/// # Safety
///
/// See [`register`].
unsafe fn register_stream_class(library: GDExtensionClassLibraryPtr, interface: &InterfaceTable) {
    let mut class_name = OwnedStringName::new();
    let mut parent = OwnedStringName::new();
    // SAFETY: same rationale as `register_session_class`.
    unsafe {
        class_name.init_utf8(interface, class_names::VOKRA_STREAM);
        parent.init_utf8(interface, class_names::PARENT_OBJECT);
    }

    let info = GDExtensionClassCreationInfo3 {
        is_virtual: 0,
        is_abstract: 0,
        is_exposed: 1,
        is_runtime: 0,
        set_func: None,
        get_func: None,
        get_property_list_func: None,
        free_property_list_func: None,
        property_can_revert_func: None,
        property_get_revert_func: None,
        validate_property_func: None,
        notification_func: None,
        to_string_func: None,
        reference_func: None,
        unreference_func: None,
        create_instance_func: Some(create_stream_instance),
        free_instance_func: Some(free_stream_instance),
        recreate_instance_func: None,
        get_virtual_func: None,
        get_virtual_call_data_func: None,
        call_virtual_with_data_func: None,
        get_rid_func: None,
        class_userdata: ptr::null_mut(),
    };

    // SAFETY: all pointers alive for register call.
    unsafe {
        (interface.classdb_register_extension_class3)(
            library,
            class_name.as_ptr(),
            parent.as_ptr(),
            &info,
        );

        register_stream_push_pcm(library, interface, class_name.as_ptr());
        register_stream_poll(library, interface, class_name.as_ptr());
        register_stream_interrupt(library, interface, class_name.as_ptr());

        // Signals (T09).
        register_stream_asr_chunk_signal(library, interface, class_name.as_ptr());
        register_stream_tts_chunk_signal(library, interface, class_name.as_ptr());
    }
}

// ---------------------------------------------------------------------------
// Individual method registrations. Each builds a `GDExtensionClassMethodInfo`
// on the stack, populates the argument PropertyInfo array, and calls
// `classdb_register_extension_class_method`.
// ---------------------------------------------------------------------------

const PROPERTY_USAGE_DEFAULT: u32 = 7; // STORAGE(1) | EDITOR(2) | NETWORK(4)

/// Backing storage for one Godot `String`. Godot pins `sizeof(String) == 8`
/// on LP64 (a single `CowData` pointer) — see
/// [`crate::ffi::gdextension::STRING_SIZE`]. Same "opaque scratch buffer we
/// hand to a Godot constructor" discipline as [`OwnedStringName`].
///
/// Like `OwnedStringName` this has no `Drop`, but for a stronger reason
/// than that type's documented controlled leak: the only `String` built
/// here is EMPTY, and an empty Godot `String` holds a null `CowData` — no
/// heap allocation is made, so there is nothing to release. This would stop
/// being true the moment a non-empty String is constructed through this
/// type, which is why the constructor is `new_empty` rather than a general
/// `from_utf8`.
#[repr(C, align(8))]
struct OwnedString {
    storage: [u8; STRING_SIZE],
}

impl OwnedString {
    /// Construct an empty Godot `String`.
    ///
    /// # Safety
    ///
    /// `interface.string_new_with_utf8_chars_and_len` must be a live Godot
    /// fn pointer. The resulting String is valid only while `self` lives.
    unsafe fn new_empty(interface: &InterfaceTable) -> Self {
        let mut this = Self {
            storage: [0u8; STRING_SIZE],
        };
        // SAFETY: `storage` is a writable STRING_SIZE-byte, 8-aligned
        // buffer. A zero length with a NUL-terminated empty source is the
        // documented way to build an empty String. `c""` is the empty C
        // string literal — a lone NUL — so the pointer is valid to read.
        unsafe {
            (interface.string_new_with_utf8_chars_and_len)(
                this.storage.as_mut_ptr() as *mut c_void,
                c"".as_ptr(),
                0,
            );
        }
        this
    }

    fn as_ptr(&self) -> GDExtensionStringPtr {
        self.storage.as_ptr() as *mut c_void
    }
}

/// Owned empty `StringName` + empty `String` for the two
/// [`GDExtensionPropertyInfo`] fields we do not otherwise populate.
///
/// # Why this exists (NULL is not a legal "unused" marker)
///
/// Godot converts every `GDExtensionPropertyInfo` we hand it into its own
/// `PropertyInfo` by **unconditionally dereferencing** both pointers:
///
/// ```cpp
/// PropertyInfo(Variant::Type(p_info.type),
///              *reinterpret_cast<const StringName *>(p_info.name),
///              PropertyHint(p_info.hint),
///              *reinterpret_cast<const String *>(p_info.hint_string),
///              p_info.usage,
///              *reinterpret_cast<const StringName *>(p_info.class_name));
/// ```
///
/// There is no NULL check on `hint_string` / `class_name`. Passing
/// `ptr::null_mut()` — which this file did until the M3-11 T19 headless
/// leg caught it — segfaults Godot inside
/// `GDExtension::_register_extension_class_method`, i.e. the extension
/// killed the host process the first time any method was registered.
/// Reproduced identically on Godot 4.3-stable and 4.7.1-stable.
///
/// godot-cpp never hits this because its `PropertyInfo` wrapper always
/// materialises real (empty) StringName / String values.
///
/// One scratch is shared by every PropertyInfo within a single
/// registration call: Godot copies the values out during registration
/// ("Provided struct can be safely freed once the function returns" —
/// `gdextension_interface.h`), so a shared empty value is sound.
struct EmptyPropertyFields {
    class_name: OwnedStringName,
    hint_string: OwnedString,
}

impl EmptyPropertyFields {
    /// # Safety
    ///
    /// `interface` must hold live Godot fn pointers.
    unsafe fn new(interface: &InterfaceTable) -> Self {
        let mut class_name = OwnedStringName::new();
        // SAFETY: caller guarantees live fn pointers; `b"\0"` is a
        // NUL-terminated empty string.
        unsafe {
            class_name.init_utf8(interface, b"\0");
        }
        Self {
            class_name,
            // SAFETY: same.
            hint_string: unsafe { OwnedString::new_empty(interface) },
        }
    }
}

/// Build a `GDExtensionPropertyInfo` for an argument/return value.
///
/// Every pointer inside the returned struct — `name`, and both fields of
/// `empties` — MUST stay live for the duration of the register call. See
/// [`EmptyPropertyFields`] for why `class_name` / `hint_string` cannot be
/// NULL.
fn make_property_info(
    ty: GDExtensionVariantType,
    name: GDExtensionStringNamePtr,
    empties: &EmptyPropertyFields,
) -> GDExtensionPropertyInfo {
    GDExtensionPropertyInfo {
        r#type: ty,
        name,
        class_name: empties.class_name.as_ptr(),
        hint: 0,
        hint_string: empties.hint_string.as_ptr(),
        usage: PROPERTY_USAGE_DEFAULT,
    }
}

/// `VokraSession::load_model(path: String) -> int`.
///
/// The second half of the `new() -> load_model(path)` two-step documented
/// in this module's "Instance lifetime" section. Returns a
/// [`crate::ffi::capi::VokraStatus`] numeric code rather than raising a
/// CallError, because a missing / unreadable GGUF is an expected runtime
/// condition that GDScript branches on:
///
/// ```gdscript
/// var load_status: int = session.load_model(MODEL_PATH)
/// if load_status != 0:
///     _status.text = "Error: load_model returned status=%d" % load_status
/// ```
///
/// See [`crate::trampoline::session_load_model`] for the error-posture
/// split between in-band status codes and CallErrors.
///
/// # Safety
///
/// See [`register`].
unsafe fn register_session_load_model(
    library: GDExtensionClassLibraryPtr,
    interface: &InterfaceTable,
    class_name: GDExtensionStringNamePtr,
) {
    let mut method_name = OwnedStringName::new();
    let mut arg0 = OwnedStringName::new();
    let mut ret_name = OwnedStringName::new();
    // SAFETY: interface live; byte constants NUL-terminated.
    unsafe {
        method_name.init_utf8(interface, method_names::LOAD_MODEL);
        arg0.init_utf8(interface, arg_names::PATH);
        ret_name.init_utf8(interface, b"status\0");
    }

    // SAFETY: `interface` holds live Godot fn pointers (caller doc).
    let empties = unsafe { EmptyPropertyFields::new(interface) };
    let mut args = [make_property_info(
        GDExtensionVariantType::String,
        arg0.as_ptr(),
        &empties,
    )];
    let mut args_meta = [GDExtensionClassMethodArgumentMetadata::None];
    let mut ret_info = make_property_info(GDExtensionVariantType::Int, ret_name.as_ptr(), &empties);

    let method_info = GDExtensionClassMethodInfo {
        name: method_name.as_ptr(),
        method_userdata: ptr::null_mut(),
        call_func: Some(crate::trampoline::session_load_model),
        ptrcall_func: None,
        method_flags: method_flags::NORMAL,
        has_return_value: 1,
        return_value_info: &mut ret_info,
        return_value_metadata: GDExtensionClassMethodArgumentMetadata::None,
        argument_count: args.len() as u32,
        arguments_info: args.as_mut_ptr(),
        arguments_metadata: args_meta.as_mut_ptr(),
        default_argument_count: 0,
        default_arguments: ptr::null_mut(),
    };

    // SAFETY: all pointers inside method_info are live for this call.
    unsafe {
        (interface.classdb_register_extension_class_method)(library, class_name, &method_info);
    }
}

/// `VokraSession::transcribe(pcm: PackedFloat32Array, sample_rate: int) -> String`.
///
/// # Safety
///
/// See [`register`].
unsafe fn register_session_transcribe(
    library: GDExtensionClassLibraryPtr,
    interface: &InterfaceTable,
    class_name: GDExtensionStringNamePtr,
) {
    let mut method_name = OwnedStringName::new();
    let mut arg0 = OwnedStringName::new();
    let mut arg1 = OwnedStringName::new();
    let mut ret_name = OwnedStringName::new();
    // SAFETY: interface live; byte constants NUL-terminated.
    unsafe {
        method_name.init_utf8(interface, method_names::TRANSCRIBE);
        arg0.init_utf8(interface, arg_names::PCM);
        arg1.init_utf8(interface, arg_names::SAMPLE_RATE);
        ret_name.init_utf8(interface, b"result\0");
    }

    // SAFETY: `interface` holds live Godot fn pointers (caller doc).
    let empties = unsafe { EmptyPropertyFields::new(interface) };
    let mut args = [
        make_property_info(
            GDExtensionVariantType::PackedFloat32Array,
            arg0.as_ptr(),
            &empties,
        ),
        make_property_info(GDExtensionVariantType::Int, arg1.as_ptr(), &empties),
    ];
    let mut args_meta = [
        GDExtensionClassMethodArgumentMetadata::None,
        GDExtensionClassMethodArgumentMetadata::None,
    ];
    let mut ret_info =
        make_property_info(GDExtensionVariantType::String, ret_name.as_ptr(), &empties);

    let method_info = GDExtensionClassMethodInfo {
        name: method_name.as_ptr(),
        method_userdata: ptr::null_mut(),
        call_func: Some(crate::trampoline::session_transcribe),
        ptrcall_func: None,
        method_flags: method_flags::NORMAL,
        has_return_value: 1,
        return_value_info: &mut ret_info,
        return_value_metadata: GDExtensionClassMethodArgumentMetadata::None,
        argument_count: args.len() as u32,
        arguments_info: args.as_mut_ptr(),
        arguments_metadata: args_meta.as_mut_ptr(),
        default_argument_count: 0,
        default_arguments: ptr::null_mut(),
    };

    // SAFETY: all pointers inside method_info are live for this call.
    unsafe {
        (interface.classdb_register_extension_class_method)(library, class_name, &method_info);
    }
}

/// `VokraSession::synthesize(text: String) -> Dictionary`.
///
/// # Safety
///
/// See [`register`].
unsafe fn register_session_synthesize(
    library: GDExtensionClassLibraryPtr,
    interface: &InterfaceTable,
    class_name: GDExtensionStringNamePtr,
) {
    let mut method_name = OwnedStringName::new();
    let mut arg0 = OwnedStringName::new();
    let mut ret_name = OwnedStringName::new();
    // SAFETY: interface live; byte constants NUL-terminated.
    unsafe {
        method_name.init_utf8(interface, method_names::SYNTHESIZE);
        arg0.init_utf8(interface, arg_names::TEXT);
        ret_name.init_utf8(interface, b"result\0");
    }

    // SAFETY: `interface` holds live Godot fn pointers (caller doc).
    let empties = unsafe { EmptyPropertyFields::new(interface) };
    let mut args = [make_property_info(
        GDExtensionVariantType::String,
        arg0.as_ptr(),
        &empties,
    )];
    let mut args_meta = [GDExtensionClassMethodArgumentMetadata::None];
    let mut ret_info = make_property_info(
        GDExtensionVariantType::Dictionary,
        ret_name.as_ptr(),
        &empties,
    );

    let method_info = GDExtensionClassMethodInfo {
        name: method_name.as_ptr(),
        method_userdata: ptr::null_mut(),
        call_func: Some(crate::trampoline::session_synthesize),
        ptrcall_func: None,
        method_flags: method_flags::NORMAL,
        has_return_value: 1,
        return_value_info: &mut ret_info,
        return_value_metadata: GDExtensionClassMethodArgumentMetadata::None,
        argument_count: args.len() as u32,
        arguments_info: args.as_mut_ptr(),
        arguments_metadata: args_meta.as_mut_ptr(),
        default_argument_count: 0,
        default_arguments: ptr::null_mut(),
    };

    // SAFETY: all pointers live for this call.
    unsafe {
        (interface.classdb_register_extension_class_method)(library, class_name, &method_info);
    }
}

/// `VokraSession::vad_open_stream(sample_rate: int) -> Object`.
///
/// Returns Nil at the ClassDB surface for now — the Object return type is
/// declared via `return_value_info` but Godot 4.3 documents that Object
/// returns from extension methods travel through Variant boxing anyway.
/// Owner M3-18 resolves this.
///
/// # Safety
///
/// See [`register`].
unsafe fn register_session_vad_open_stream(
    library: GDExtensionClassLibraryPtr,
    interface: &InterfaceTable,
    class_name: GDExtensionStringNamePtr,
) {
    let mut method_name = OwnedStringName::new();
    let mut arg0 = OwnedStringName::new();
    let mut ret_name = OwnedStringName::new();
    // SAFETY: interface live; byte constants NUL-terminated.
    unsafe {
        method_name.init_utf8(interface, method_names::VAD_OPEN_STREAM);
        arg0.init_utf8(interface, arg_names::SAMPLE_RATE);
        ret_name.init_utf8(interface, b"stream\0");
    }

    // SAFETY: `interface` holds live Godot fn pointers (caller doc).
    let empties = unsafe { EmptyPropertyFields::new(interface) };
    let mut args = [make_property_info(
        GDExtensionVariantType::Int,
        arg0.as_ptr(),
        &empties,
    )];
    let mut args_meta = [GDExtensionClassMethodArgumentMetadata::None];
    // Returns a Nil Variant at ClassDB level; the trampoline is expected
    // to box the Godot Object separately.
    let mut ret_info = make_property_info(GDExtensionVariantType::Nil, ret_name.as_ptr(), &empties);

    let method_info = GDExtensionClassMethodInfo {
        name: method_name.as_ptr(),
        method_userdata: ptr::null_mut(),
        call_func: Some(crate::trampoline::session_vad_open_stream),
        ptrcall_func: None,
        method_flags: method_flags::NORMAL,
        has_return_value: 1,
        return_value_info: &mut ret_info,
        return_value_metadata: GDExtensionClassMethodArgumentMetadata::None,
        argument_count: args.len() as u32,
        arguments_info: args.as_mut_ptr(),
        arguments_metadata: args_meta.as_mut_ptr(),
        default_argument_count: 0,
        default_arguments: ptr::null_mut(),
    };

    // SAFETY: same rationale.
    unsafe {
        (interface.classdb_register_extension_class_method)(library, class_name, &method_info);
    }
}

/// `VokraStream::push_pcm(pcm: PackedFloat32Array) -> void`.
///
/// # Safety
///
/// See [`register`].
unsafe fn register_stream_push_pcm(
    library: GDExtensionClassLibraryPtr,
    interface: &InterfaceTable,
    class_name: GDExtensionStringNamePtr,
) {
    let mut method_name = OwnedStringName::new();
    let mut arg0 = OwnedStringName::new();
    // SAFETY: interface live; byte constants NUL-terminated.
    unsafe {
        method_name.init_utf8(interface, method_names::PUSH_PCM);
        arg0.init_utf8(interface, arg_names::PCM);
    }

    // SAFETY: `interface` holds live Godot fn pointers (caller doc).
    let empties = unsafe { EmptyPropertyFields::new(interface) };
    let mut args = [make_property_info(
        GDExtensionVariantType::PackedFloat32Array,
        arg0.as_ptr(),
        &empties,
    )];
    let mut args_meta = [GDExtensionClassMethodArgumentMetadata::None];

    let method_info = GDExtensionClassMethodInfo {
        name: method_name.as_ptr(),
        method_userdata: ptr::null_mut(),
        call_func: Some(crate::trampoline::stream_push_pcm),
        ptrcall_func: None,
        method_flags: method_flags::NORMAL,
        has_return_value: 0,
        return_value_info: ptr::null_mut(),
        return_value_metadata: GDExtensionClassMethodArgumentMetadata::None,
        argument_count: args.len() as u32,
        arguments_info: args.as_mut_ptr(),
        arguments_metadata: args_meta.as_mut_ptr(),
        default_argument_count: 0,
        default_arguments: ptr::null_mut(),
    };

    // SAFETY: same rationale.
    unsafe {
        (interface.classdb_register_extension_class_method)(library, class_name, &method_info);
    }
}

/// `VokraStream::poll(capacity: int) -> PackedFloat32Array`.
///
/// # Safety
///
/// See [`register`].
unsafe fn register_stream_poll(
    library: GDExtensionClassLibraryPtr,
    interface: &InterfaceTable,
    class_name: GDExtensionStringNamePtr,
) {
    let mut method_name = OwnedStringName::new();
    let mut arg0 = OwnedStringName::new();
    let mut ret_name = OwnedStringName::new();
    // SAFETY: interface live; byte constants NUL-terminated.
    unsafe {
        method_name.init_utf8(interface, method_names::POLL);
        arg0.init_utf8(interface, arg_names::CAPACITY);
        ret_name.init_utf8(interface, b"probs\0");
    }

    // SAFETY: `interface` holds live Godot fn pointers (caller doc).
    let empties = unsafe { EmptyPropertyFields::new(interface) };
    let mut args = [make_property_info(
        GDExtensionVariantType::Int,
        arg0.as_ptr(),
        &empties,
    )];
    let mut args_meta = [GDExtensionClassMethodArgumentMetadata::None];
    let mut ret_info = make_property_info(
        GDExtensionVariantType::PackedFloat32Array,
        ret_name.as_ptr(),
        &empties,
    );

    let method_info = GDExtensionClassMethodInfo {
        name: method_name.as_ptr(),
        method_userdata: ptr::null_mut(),
        call_func: Some(crate::trampoline::stream_poll),
        ptrcall_func: None,
        method_flags: method_flags::NORMAL,
        has_return_value: 1,
        return_value_info: &mut ret_info,
        return_value_metadata: GDExtensionClassMethodArgumentMetadata::None,
        argument_count: args.len() as u32,
        arguments_info: args.as_mut_ptr(),
        arguments_metadata: args_meta.as_mut_ptr(),
        default_argument_count: 0,
        default_arguments: ptr::null_mut(),
    };

    // SAFETY: same rationale.
    unsafe {
        (interface.classdb_register_extension_class_method)(library, class_name, &method_info);
    }
}

/// `VokraStream::interrupt() -> void`.
///
/// # Safety
///
/// See [`register`].
unsafe fn register_stream_interrupt(
    library: GDExtensionClassLibraryPtr,
    interface: &InterfaceTable,
    class_name: GDExtensionStringNamePtr,
) {
    let mut method_name = OwnedStringName::new();
    // SAFETY: interface live; byte constant NUL-terminated.
    unsafe {
        method_name.init_utf8(interface, method_names::INTERRUPT);
    }

    let method_info = GDExtensionClassMethodInfo {
        name: method_name.as_ptr(),
        method_userdata: ptr::null_mut(),
        call_func: Some(crate::trampoline::stream_interrupt),
        ptrcall_func: None,
        method_flags: method_flags::NORMAL,
        has_return_value: 0,
        return_value_info: ptr::null_mut(),
        return_value_metadata: GDExtensionClassMethodArgumentMetadata::None,
        argument_count: 0,
        arguments_info: ptr::null_mut(),
        arguments_metadata: ptr::null_mut(),
        default_argument_count: 0,
        default_arguments: ptr::null_mut(),
    };

    // SAFETY: name lives; interface fn ptr live.
    unsafe {
        (interface.classdb_register_extension_class_method)(library, class_name, &method_info);
    }
}

// ---------------------------------------------------------------------------
// Signal registration (T09).
//
// GDExtension signals are declared through `classdb_register_extension_class_signal`
// with a PropertyInfo array describing each argument. The Godot Object then
// emits them via its built-in `emit_signal` — we don't need to bind
// `object_emit_signal` ourselves at registration time; the M3-18 patch that
// promotes the stream trampolines to real dispatch will call `emit_signal`
// through Godot's classdb method-bind (`Object.emit_signal(String, ...)`),
// avoiding an extra proc-address resolve.
// ---------------------------------------------------------------------------

/// Signal: `VokraStream::asr_chunk(prob: float)`.
///
/// # Safety
///
/// See [`register`].
unsafe fn register_stream_asr_chunk_signal(
    library: GDExtensionClassLibraryPtr,
    interface: &InterfaceTable,
    class_name: GDExtensionStringNamePtr,
) {
    let mut signal_name = OwnedStringName::new();
    let mut arg0 = OwnedStringName::new();
    // SAFETY: interface live; byte constants NUL-terminated.
    unsafe {
        signal_name.init_utf8(interface, signal_names::ASR_CHUNK);
        arg0.init_utf8(interface, arg_names::PROB);
    }

    // SAFETY: `interface` holds live Godot fn pointers (caller doc).
    let empties = unsafe { EmptyPropertyFields::new(interface) };
    let arg_infos = [make_property_info(
        GDExtensionVariantType::Float,
        arg0.as_ptr(),
        &empties,
    )];
    // SAFETY: name + arg_info alive for this call.
    unsafe {
        (interface.classdb_register_extension_class_signal)(
            library,
            class_name,
            signal_name.as_ptr(),
            arg_infos.as_ptr(),
            arg_infos.len() as GDExtensionInt,
        );
    }
}

/// Signal: `VokraStream::tts_chunk(pcm: PackedFloat32Array)`.
///
/// # Safety
///
/// See [`register`].
unsafe fn register_stream_tts_chunk_signal(
    library: GDExtensionClassLibraryPtr,
    interface: &InterfaceTable,
    class_name: GDExtensionStringNamePtr,
) {
    let mut signal_name = OwnedStringName::new();
    let mut arg0 = OwnedStringName::new();
    // SAFETY: interface live; byte constants NUL-terminated.
    unsafe {
        signal_name.init_utf8(interface, signal_names::TTS_CHUNK);
        arg0.init_utf8(interface, arg_names::PCM);
    }

    // SAFETY: `interface` holds live Godot fn pointers (caller doc).
    let empties = unsafe { EmptyPropertyFields::new(interface) };
    let arg_infos = [make_property_info(
        GDExtensionVariantType::PackedFloat32Array,
        arg0.as_ptr(),
        &empties,
    )];
    // SAFETY: name + arg_info alive for this call.
    unsafe {
        (interface.classdb_register_extension_class_signal)(
            library,
            class_name,
            signal_name.as_ptr(),
            arg_infos.as_ptr(),
            arg_infos.len() as GDExtensionInt,
        );
    }
}

// ---------------------------------------------------------------------------
// Signature guard: every `create_instance_func` MUST fit the Godot typedef.
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(dead_code)]
fn create_instance_signature_fits() {
    let _: crate::ffi::gdextension::GDExtensionClassCreateInstance = create_session_instance;
    let _: crate::ffi::gdextension::GDExtensionClassCreateInstance = create_stream_instance;
    let _: crate::ffi::gdextension::GDExtensionClassFreeInstance = free_session_instance;
    let _: crate::ffi::gdextension::GDExtensionClassFreeInstance = free_stream_instance;
}

// Suppress the unused `GDExtensionBool` import warning when tests are off
// (Bool is used through the struct literal only).
#[allow(dead_code)]
const _BOOL_MARKER: GDExtensionBool = 0;

// ---------------------------------------------------------------------------
// Mock-driven smoke tests.
//
// Real Godot editor testing lives at M3-18. Here we drive the register /
// unregister paths through a mock `InterfaceTable` whose functions:
// - Record every call (class name, method name, signal name, arity, flags).
// - Do NOT allocate real Godot StringNames (they leave the storage bytes
//   untouched, which is fine because our own code never inspects them).
// The tests then assert the recorded events match expectations.
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::sync::Mutex;

    // ------------------------------------------------------------------
    // Recording mock.
    //
    // Every mock fn is `extern "C"` and pushes an event into a
    // process-lifetime `Mutex<Vec<Event>>`. Tests read + clear the log.
    // ------------------------------------------------------------------

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) enum Event {
        ClassRegistered {
            class_name: String,
            parent_name: String,
            has_create: bool,
            has_free: bool,
            is_exposed: bool,
        },
        MethodRegistered {
            class_name: String,
            method_name: String,
            arg_count: u32,
            has_return: bool,
            has_call_func: bool,
            method_flags: u32,
            /// Declared return Variant type, read back from
            /// `return_value_info`. `None` when the method declares no
            /// return value. Lets tests assert the GDScript-visible
            /// signature (e.g. `load_model` MUST return Int so the demo's
            /// `var load_status: int = session.load_model(...)` coerces).
            return_type: Option<GDExtensionVariantType>,
        },
        SignalRegistered {
            class_name: String,
            signal_name: String,
            arg_count: i64,
        },
        ClassUnregistered {
            class_name: String,
        },
        /// `classdb_construct_object(<class>)` — the Godot-side Object a
        /// `create_instance_func` must build.
        ObjectConstructed {
            class_name: String,
        },
        /// `object_set_instance(obj, <class>, instance)` — binds our Rust
        /// allocation to that Object.
        InstanceBound {
            class_name: String,
            object_is_null: bool,
            /// Raw instance pointer, so a test can reclaim the leaked
            /// allocation through `free_*_instance`.
            instance: usize,
        },
    }

    thread_local! {
        // Each StringName pointer we get from the mock is really a pointer
        // to a `&'static [u8]` (the byte constant we passed to
        // `init_utf8`). The mock records that pointer verbatim; on the
        // recording side we look it up in this table. This works because
        // we control both ends of the mock plumbing.
        pub(crate) static NAME_TABLE: RefCell<Vec<(usize, &'static [u8])>> = const { RefCell::new(Vec::new()) };
    }

    pub(crate) static EVENTS: Mutex<Vec<Event>> = Mutex::new(Vec::new());

    /// Serialize registry tests that share `EVENTS` + `NAME_TABLE`. Cargo's
    /// default parallelism would let them interleave and race.
    pub(crate) static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn resolve_name(ptr: GDExtensionStringNamePtr) -> String {
        let key = ptr as usize;
        // Reverse-search: `OwnedStringName` buffers are stack-allocated
        // and pointers may be reused across nested calls
        // (`register_session_class` unwinds its stack before
        // `register_stream_class` runs). The LAST push wins — take the
        // most recent binding for this pointer.
        NAME_TABLE.with(|t| {
            for (k, bytes) in t.borrow().iter().rev() {
                if *k == key {
                    // Strip trailing NUL for readability.
                    let s = std::str::from_utf8(&bytes[..bytes.len() - 1]).unwrap_or("<utf8?>");
                    return s.to_string();
                }
            }
            format!("<unknown:{key:#x}>")
        })
    }

    // The mock `string_name_new_with_utf8_chars` looks up the input
    // C-string in a small side table (mirrors the byte constants we
    // declared as `class_names::*` / `method_names::*` etc.), and stores
    // the resolved slice in `NAME_TABLE` keyed by the destination
    // pointer. Then subsequent register calls see that pointer and can
    // reconstruct the intended name.
    unsafe extern "C" fn mock_string_name_new_with_utf8_chars(
        r_dest: crate::ffi::gdextension::GDExtensionUninitializedStringNamePtr,
        p_contents: *const c_char,
    ) {
        // SAFETY: `r_dest` is a writable 16-byte OwnedStringName buffer we
        // allocated; `p_contents` is a NUL-terminated C string.
        let bytes = unsafe { core::ffi::CStr::from_ptr(p_contents) }.to_bytes_with_nul();
        // Compare against known constants. Extend as new names appear.
        let matched: &'static [u8] = match bytes {
            b if b == class_names::VOKRA_SESSION => class_names::VOKRA_SESSION,
            b if b == class_names::VOKRA_STREAM => class_names::VOKRA_STREAM,
            b if b == class_names::PARENT_OBJECT => class_names::PARENT_OBJECT,
            b if b == method_names::LOAD_MODEL => method_names::LOAD_MODEL,
            b if b == method_names::TRANSCRIBE => method_names::TRANSCRIBE,
            b if b == method_names::SYNTHESIZE => method_names::SYNTHESIZE,
            b if b == method_names::VAD_OPEN_STREAM => method_names::VAD_OPEN_STREAM,
            b if b == method_names::PUSH_PCM => method_names::PUSH_PCM,
            b if b == method_names::POLL => method_names::POLL,
            b if b == method_names::INTERRUPT => method_names::INTERRUPT,
            b if b == signal_names::ASR_CHUNK => signal_names::ASR_CHUNK,
            b if b == signal_names::TTS_CHUNK => signal_names::TTS_CHUNK,
            b if b == arg_names::PATH => arg_names::PATH,
            b if b == arg_names::PCM => arg_names::PCM,
            b if b == arg_names::SAMPLE_RATE => arg_names::SAMPLE_RATE,
            b if b == arg_names::TEXT => arg_names::TEXT,
            b if b == arg_names::CAPACITY => arg_names::CAPACITY,
            b if b == arg_names::PROB => arg_names::PROB,
            b if b == b"result\0" => b"result\0",
            b if b == b"status\0" => b"status\0",
            b if b == b"stream\0" => b"stream\0",
            b if b == b"probs\0" => b"probs\0",
            _ => b"<unrecorded>\0",
        };
        let key = r_dest as usize;
        NAME_TABLE.with(|t| t.borrow_mut().push((key, matched)));
    }

    unsafe extern "C" fn mock_string_name_new_with_latin1_chars(
        _r_dest: crate::ffi::gdextension::GDExtensionUninitializedStringNamePtr,
        _p_contents: *const c_char,
        _p_is_static: GDExtensionBool,
    ) {
        // Unused in registry.rs — the byte constants all pass through the
        // utf8 path. Present only so InterfaceTable's typing is
        // satisfied for the mock.
    }

    unsafe extern "C" fn mock_mem_alloc(_p_bytes: usize) -> *mut c_void {
        // We don't route through Godot's allocator in registry.rs; this
        // is a defensive stub. Any invocation is a test-side bug.
        panic!("mock_mem_alloc should not be reached from registry.rs paths");
    }

    unsafe extern "C" fn mock_mem_free(_p_ptr: *mut c_void) {
        panic!("mock_mem_free should not be reached from registry.rs paths");
    }

    unsafe extern "C" fn mock_classdb_register_extension_class3(
        _p_library: GDExtensionClassLibraryPtr,
        p_class_name: crate::ffi::gdextension::GDExtensionConstStringNamePtr,
        p_parent_class_name: crate::ffi::gdextension::GDExtensionConstStringNamePtr,
        p_extension_funcs: *const GDExtensionClassCreationInfo3,
    ) {
        let class_name = resolve_name(p_class_name as *mut _);
        let parent_name = resolve_name(p_parent_class_name as *mut _);
        // SAFETY: caller (registry.rs) provides a valid pointer to a
        // populated `GDExtensionClassCreationInfo3` struct.
        let info = unsafe { &*p_extension_funcs };
        EVENTS.lock().unwrap().push(Event::ClassRegistered {
            class_name,
            parent_name,
            has_create: info.create_instance_func.is_some(),
            has_free: info.free_instance_func.is_some(),
            is_exposed: info.is_exposed != 0,
        });
    }

    unsafe extern "C" fn mock_classdb_register_extension_class_method(
        _p_library: GDExtensionClassLibraryPtr,
        p_class_name: crate::ffi::gdextension::GDExtensionConstStringNamePtr,
        p_method_info: *const GDExtensionClassMethodInfo,
    ) {
        let class_name = resolve_name(p_class_name as *mut _);
        // SAFETY: caller (registry.rs) provides a valid pointer.
        let info = unsafe { &*p_method_info };
        let method_name = resolve_name(info.name);
        // Read back the declared return type. `return_value_info` is only
        // meaningful when `has_return_value != 0`; a void method leaves it
        // NULL (see `register_stream_push_pcm`).
        let return_type = if info.has_return_value != 0 && !info.return_value_info.is_null() {
            // SAFETY: `has_return_value != 0` means the registering fn
            // populated `return_value_info` with a live
            // `GDExtensionPropertyInfo` that outlives this call (it is a
            // stack local in the caller's frame).
            Some(unsafe { (*info.return_value_info).r#type })
        } else {
            None
        };
        EVENTS.lock().unwrap().push(Event::MethodRegistered {
            class_name,
            method_name,
            arg_count: info.argument_count,
            has_return: info.has_return_value != 0,
            has_call_func: info.call_func.is_some(),
            method_flags: info.method_flags,
            return_type,
        });
    }

    unsafe extern "C" fn mock_classdb_register_extension_class_signal(
        _p_library: GDExtensionClassLibraryPtr,
        p_class_name: crate::ffi::gdextension::GDExtensionConstStringNamePtr,
        p_signal_name: crate::ffi::gdextension::GDExtensionConstStringNamePtr,
        _p_argument_info: *const GDExtensionPropertyInfo,
        p_argument_count: GDExtensionInt,
    ) {
        let class_name = resolve_name(p_class_name as *mut _);
        let signal_name = resolve_name(p_signal_name as *mut _);
        EVENTS.lock().unwrap().push(Event::SignalRegistered {
            class_name,
            signal_name,
            arg_count: p_argument_count,
        });
    }

    /// Stand-in for a Godot `Object`. The registry only NULL-checks the
    /// value and forwards it to `object_set_instance`, so any stable
    /// non-null address works.
    static FAKE_GODOT_OBJECT: u8 = 0;

    unsafe extern "C" fn mock_classdb_construct_object(
        p_classname: crate::ffi::gdextension::GDExtensionConstStringNamePtr,
    ) -> GDExtensionObjectPtr {
        EVENTS.lock().unwrap().push(Event::ObjectConstructed {
            class_name: resolve_name(p_classname as *mut _),
        });
        &FAKE_GODOT_OBJECT as *const u8 as GDExtensionObjectPtr
    }

    /// `classdb_construct_object` that refuses to build — drives the
    /// FR-EX-08 "Godot said no" path in [`create_bound_object`].
    unsafe extern "C" fn mock_classdb_construct_object_fails(
        _p_classname: crate::ffi::gdextension::GDExtensionConstStringNamePtr,
    ) -> GDExtensionObjectPtr {
        ptr::null_mut()
    }

    unsafe extern "C" fn mock_object_set_instance(
        p_o: GDExtensionObjectPtr,
        p_classname: crate::ffi::gdextension::GDExtensionConstStringNamePtr,
        p_instance: GDExtensionClassInstancePtr,
    ) {
        EVENTS.lock().unwrap().push(Event::InstanceBound {
            class_name: resolve_name(p_classname as *mut _),
            object_is_null: p_o.is_null(),
            instance: p_instance as usize,
        });
    }

    unsafe extern "C" fn mock_classdb_unregister_extension_class(
        _p_library: GDExtensionClassLibraryPtr,
        p_class_name: crate::ffi::gdextension::GDExtensionConstStringNamePtr,
    ) {
        let class_name = resolve_name(p_class_name as *mut _);
        EVENTS
            .lock()
            .unwrap()
            .push(Event::ClassUnregistered { class_name });
    }

    pub(crate) fn make_mock_interface() -> InterfaceTable {
        // Variant-support fields are unused by registry-side tests (no
        // Variant packing during class registration). Route them through
        // the sig-aware sibling mocks so the resulting table is
        // structurally complete; individual tests may override fields.
        //
        // For the M3-11 T14/M3-18 unpack foundation fields
        // (variant_new_copy, variant_destroy, packed_float32_array_*,
        // string_*, and the additional cached typed constructors) route
        // through simple no-op mocks defined below. Every mock has the
        // documented C-header signature so the InterfaceTable is
        // structurally valid; registry tests never invoke these fields.
        use crate::ffi::interface::tests as iface_tests;
        InterfaceTable {
            classdb_register_extension_class3: mock_classdb_register_extension_class3,
            classdb_register_extension_class_method: mock_classdb_register_extension_class_method,
            classdb_register_extension_class_signal: mock_classdb_register_extension_class_signal,
            classdb_unregister_extension_class: mock_classdb_unregister_extension_class,
            classdb_construct_object: mock_classdb_construct_object,
            object_set_instance: mock_object_set_instance,
            string_name_new_with_utf8_chars: mock_string_name_new_with_utf8_chars,
            string_name_new_with_latin1_chars: mock_string_name_new_with_latin1_chars,
            mem_alloc: mock_mem_alloc,
            mem_free: mock_mem_free,
            variant_get_type: iface_tests::mock_variant_get_type,
            variant_new_nil: iface_tests::mock_variant_new_nil,
            variant_from_int_ctor: iface_tests::mock_variant_from_int,
            variant_to_int_ctor: iface_tests::mock_variant_to_int,
            // M3-11 T14/M3-18 unpack foundation additions. Registry-side
            // tests never invoke these; the mocks below satisfy the fn-ptr
            // shape only.
            variant_new_copy: mock_variant_new_copy,
            variant_destroy: mock_variant_destroy,
            variant_from_string_ctor: iface_tests::mock_variant_from_int,
            variant_to_string_ctor: iface_tests::mock_variant_to_int,
            variant_from_packed_float32_array_ctor: iface_tests::mock_variant_from_int,
            variant_to_packed_float32_array_ctor: iface_tests::mock_variant_to_int,
            variant_from_dictionary_ctor: iface_tests::mock_variant_from_int,
            variant_to_dictionary_ctor: iface_tests::mock_variant_to_int,
            variant_from_object_ctor: iface_tests::mock_variant_from_int,
            variant_to_object_ctor: iface_tests::mock_variant_to_int,
            packed_float32_array_operator_index_const:
                mock_packed_float32_array_operator_index_const,
            packed_float32_array_operator_index: mock_packed_float32_array_operator_index,
            string_new_with_utf8_chars_and_len: mock_string_new_with_utf8_chars_and_len,
            string_to_utf8_chars: mock_string_to_utf8_chars,
            variant_get_ptr_constructor: mock_variant_get_ptr_constructor,
            variant_get_ptr_builtin_method: mock_variant_get_ptr_builtin_method,
            variant_get_ptr_destructor: mock_variant_get_ptr_destructor,
            pfa_default_ctor: mock_pfa_default_ctor,
            pfa_resize_method: mock_pfa_builtin_method,
            pfa_destructor: mock_pfa_destructor,
            pfa_size_method: mock_pfa_builtin_method,
            string_destructor: mock_string_destructor,
            // Dictionary packing pipeline (hook-added; registry tests never
            // invoke, but the struct literal requires every field).
            dict_default_ctor: mock_pfa_default_ctor,
            dict_destructor: mock_pfa_destructor,
            dictionary_operator_index: mock_dictionary_operator_index,
        }
    }

    /// No-op mock matching the `dictionary_operator_index` signature.
    /// Returns a null Variant pointer — registry tests never invoke this
    /// field, so the null return is safe; if a future test needs it, the
    /// caller overrides the field on the returned InterfaceTable.
    unsafe extern "C" fn mock_dictionary_operator_index(
        _p_self: crate::ffi::gdextension::GDExtensionTypePtr,
        _p_key: crate::ffi::gdextension::GDExtensionConstVariantPtr,
    ) -> crate::ffi::gdextension::GDExtensionVariantPtr {
        core::ptr::null_mut()
    }

    // M3-11 T14/M3-18 unpack foundation: registry-side no-op mocks.
    // Signatures match the C header verbatim so `InterfaceTable` builds
    // cleanly; these are never invoked by registry tests. If a future
    // test needs behaviour (e.g. record inputs), it should override
    // the relevant field on the returned table before calling `register`.

    unsafe extern "C" fn mock_variant_new_copy(
        _r_dest: crate::ffi::gdextension::GDExtensionUninitializedVariantPtr,
        _p_src: crate::ffi::gdextension::GDExtensionConstVariantPtr,
    ) {
    }

    unsafe extern "C" fn mock_variant_destroy(
        _p_self: crate::ffi::gdextension::GDExtensionVariantPtr,
    ) {
    }

    unsafe extern "C" fn mock_packed_float32_array_operator_index_const(
        _p_self: crate::ffi::gdextension::GDExtensionConstTypePtr,
        _p_index: crate::ffi::gdextension::GDExtensionInt,
    ) -> *const f32 {
        core::ptr::null()
    }

    unsafe extern "C" fn mock_string_new_with_utf8_chars_and_len(
        _r_dest: crate::ffi::gdextension::GDExtensionUninitializedStringPtr,
        _p_contents: *const core::ffi::c_char,
        _p_size: crate::ffi::gdextension::GDExtensionInt,
    ) {
    }

    unsafe extern "C" fn mock_string_to_utf8_chars(
        _p_self: crate::ffi::gdextension::GDExtensionConstStringPtr,
        _r_text: *mut core::ffi::c_char,
        _p_max_write_length: crate::ffi::gdextension::GDExtensionInt,
    ) -> crate::ffi::gdextension::GDExtensionInt {
        0
    }

    // T14-followup PackedFloat32Array packing/unpacking pipeline mocks.
    // Registry-side tests never invoke these; signatures match the C header
    // verbatim so the struct literal type-checks.

    unsafe extern "C" fn mock_packed_float32_array_operator_index(
        _p_self: crate::ffi::gdextension::GDExtensionTypePtr,
        _p_index: crate::ffi::gdextension::GDExtensionInt,
    ) -> *mut f32 {
        core::ptr::null_mut()
    }

    unsafe extern "C" fn mock_variant_get_ptr_constructor(
        _p_type: crate::ffi::gdextension::GDExtensionVariantType,
        _p_constructor: i32,
    ) -> Option<crate::ffi::gdextension::GDExtensionPtrConstructor> {
        Some(mock_pfa_default_ctor)
    }

    unsafe extern "C" fn mock_variant_get_ptr_builtin_method(
        _p_type: crate::ffi::gdextension::GDExtensionVariantType,
        _p_method: crate::ffi::gdextension::GDExtensionConstStringNamePtr,
        _p_hash: crate::ffi::gdextension::GDExtensionInt,
    ) -> Option<crate::ffi::gdextension::GDExtensionPtrBuiltInMethod> {
        Some(mock_pfa_builtin_method)
    }

    unsafe extern "C" fn mock_variant_get_ptr_destructor(
        _p_type: crate::ffi::gdextension::GDExtensionVariantType,
    ) -> Option<crate::ffi::gdextension::GDExtensionPtrDestructor> {
        Some(mock_pfa_destructor)
    }

    unsafe extern "C" fn mock_pfa_default_ctor(
        _p_base: crate::ffi::gdextension::GDExtensionUninitializedTypePtr,
        _p_args: *const crate::ffi::gdextension::GDExtensionConstTypePtr,
    ) {
    }

    unsafe extern "C" fn mock_pfa_builtin_method(
        _p_base: crate::ffi::gdextension::GDExtensionTypePtr,
        _p_args: *const crate::ffi::gdextension::GDExtensionConstTypePtr,
        _r_return: crate::ffi::gdextension::GDExtensionTypePtr,
        _p_argument_count: i32,
    ) {
    }

    unsafe extern "C" fn mock_pfa_destructor(_p_base: crate::ffi::gdextension::GDExtensionTypePtr) {
    }

    unsafe extern "C" fn mock_string_destructor(
        _p_base: crate::ffi::gdextension::GDExtensionTypePtr,
    ) {
    }

    pub(crate) fn reset_recorder() {
        EVENTS.lock().unwrap().clear();
        NAME_TABLE.with(|t| t.borrow_mut().clear());
    }

    // ------------------------------------------------------------------
    // Actual tests.
    // ------------------------------------------------------------------

    #[test]
    fn register_produces_both_classes_with_expected_methods_and_signals() {
        // Serialize against other tests that touch EVENTS/NAME_TABLE.
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_recorder();

        let interface = make_mock_interface();
        // SAFETY: mock interface's fn ptrs are live; single-threaded test.
        unsafe { register(ptr::null_mut(), &interface) };

        let events = EVENTS.lock().unwrap().clone();
        // Class registration should come first for each class.
        let class_events: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                Event::ClassRegistered { class_name, .. } => Some(class_name.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            class_events,
            vec!["VokraSession".to_string(), "VokraStream".to_string()],
            "expected VokraSession then VokraStream registered",
        );

        // Methods: transcribe, synthesize, vad_open_stream on VokraSession;
        // push_pcm, poll, interrupt on VokraStream.
        let method_events: Vec<(String, String, u32)> = events
            .iter()
            .filter_map(|e| match e {
                Event::MethodRegistered {
                    class_name,
                    method_name,
                    arg_count,
                    ..
                } => Some((class_name.clone(), method_name.clone(), *arg_count)),
                _ => None,
            })
            .collect();
        let expected_methods = [
            ("VokraSession".to_string(), "load_model".to_string(), 1u32),
            ("VokraSession".to_string(), "transcribe".to_string(), 2),
            ("VokraSession".to_string(), "synthesize".to_string(), 1),
            ("VokraSession".to_string(), "vad_open_stream".to_string(), 1),
            ("VokraStream".to_string(), "push_pcm".to_string(), 1),
            ("VokraStream".to_string(), "poll".to_string(), 1),
            ("VokraStream".to_string(), "interrupt".to_string(), 0),
        ];
        for m in &expected_methods {
            assert!(
                method_events.contains(m),
                "expected method registration {m:?} missing from events; got {method_events:#?}",
            );
        }

        // Signals: asr_chunk (1 arg), tts_chunk (1 arg).
        let signal_events: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                Event::SignalRegistered {
                    class_name,
                    signal_name,
                    arg_count,
                } => Some((class_name.clone(), signal_name.clone(), *arg_count)),
                _ => None,
            })
            .collect();
        let expected_signals = [
            ("VokraStream".to_string(), "asr_chunk".to_string(), 1i64),
            ("VokraStream".to_string(), "tts_chunk".to_string(), 1),
        ];
        for s in &expected_signals {
            assert!(
                signal_events.contains(s),
                "expected signal registration {s:?} missing from events",
            );
        }
    }

    /// `load_model(path: String) -> int` is the entry point BOTH demo
    /// scenes call first (`demos/asr_demo/main.gd:66`,
    /// `demos/tts_demo/main.gd:76`), and the GDScript there binds the
    /// result to a statically-typed `int`:
    ///
    /// ```gdscript
    /// var load_status: int = session.load_model(MODEL_PATH)
    /// ```
    ///
    /// A declared return type other than `Int` would make Godot reject
    /// that assignment at runtime, so the type is asserted here rather
    /// than left to owner smoke.
    #[test]
    fn load_model_is_registered_on_session_returning_int() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_recorder();
        let interface = make_mock_interface();
        // SAFETY: mock interface's fn ptrs are live; single-threaded test.
        unsafe { register(ptr::null_mut(), &interface) };

        let events = EVENTS.lock().unwrap().clone();
        let load_model = events
            .iter()
            .find(|e| {
                matches!(
                    e,
                    Event::MethodRegistered { class_name, method_name, .. }
                        if class_name == "VokraSession" && method_name == "load_model"
                )
            })
            .cloned()
            .expect("VokraSession::load_model must be registered — both demos call it first");

        match load_model {
            Event::MethodRegistered {
                arg_count,
                has_return,
                has_call_func,
                return_type,
                ..
            } => {
                assert_eq!(arg_count, 1, "load_model takes exactly one arg (path)");
                assert!(has_return, "load_model returns a status int");
                assert!(
                    has_call_func,
                    "load_model MUST carry a dispatch trampoline — without one \
                     Godot would reject the call and the demos could never load a model",
                );
                assert_eq!(
                    return_type,
                    Some(GDExtensionVariantType::Int),
                    "load_model must declare an Int return so GDScript's \
                     `var load_status: int = session.load_model(path)` coerces",
                );
            }
            other => panic!("expected MethodRegistered, got {other:?}"),
        }
    }

    #[test]
    fn every_registered_method_has_call_func_and_normal_flags() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_recorder();
        let interface = make_mock_interface();
        // SAFETY: mock; single-threaded.
        unsafe { register(ptr::null_mut(), &interface) };

        let events = EVENTS.lock().unwrap().clone();
        for e in events.iter() {
            if let Event::MethodRegistered {
                method_name,
                has_call_func,
                method_flags,
                ..
            } = e
            {
                assert!(
                    *has_call_func,
                    "method {method_name} MUST have a call_func trampoline",
                );
                assert_eq!(
                    *method_flags,
                    super::method_flags::NORMAL,
                    "method {method_name} MUST be registered with NORMAL flags (got {method_flags})",
                );
            }
        }
    }

    #[test]
    fn every_registered_class_has_create_and_free_instance() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_recorder();
        let interface = make_mock_interface();
        // SAFETY: mock; single-threaded.
        unsafe { register(ptr::null_mut(), &interface) };

        let events = EVENTS.lock().unwrap().clone();
        for e in events.iter() {
            if let Event::ClassRegistered {
                class_name,
                has_create,
                has_free,
                is_exposed,
                ..
            } = e
            {
                assert!(
                    *has_create,
                    "class {class_name} MUST have create_instance_func",
                );
                assert!(*has_free, "class {class_name} MUST have free_instance_func",);
                assert!(
                    *is_exposed,
                    "class {class_name} MUST be exposed (is_exposed=1)",
                );
            }
        }
    }

    #[test]
    fn unregister_removes_classes_in_reverse_order() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_recorder();
        let interface = make_mock_interface();
        // SAFETY: mock; single-threaded.
        unsafe { unregister(ptr::null_mut(), &interface) };

        let events = EVENTS.lock().unwrap().clone();
        let unreg_events: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                Event::ClassUnregistered { class_name } => Some(class_name.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            unreg_events,
            vec!["VokraStream".to_string(), "VokraSession".to_string()],
            "unregister must reverse the register order",
        );
    }

    /// Install a mock interface in `EXTENSION_STATE` for the duration of a
    /// test. Mirrors `trampoline::tests::MockStateGuard`, which is private
    /// to that module.
    struct StateGuard;

    impl StateGuard {
        fn install(interface: InterfaceTable) -> Self {
            let mut guard = crate::EXTENSION_STATE.lock().unwrap();
            *guard = Some(crate::ExtensionState {
                library: ptr::null_mut(),
                interface,
            });
            Self
        }
    }

    impl Drop for StateGuard {
        fn drop(&mut self) {
            if let Ok(mut guard) = crate::EXTENSION_STATE.lock() {
                *guard = None;
            }
        }
    }

    /// Regression test for the crash the M3-11 T19 headless leg caught.
    ///
    /// `create_instance_func` MUST return a Godot `Object` built by
    /// `classdb_construct_object`, with the Rust allocation attached via
    /// `object_set_instance`. Returning the bare `Box::into_raw` pointer
    /// — which this file did before T19 — makes Godot `dynamic_cast` a
    /// non-Object and segfault on the first `VokraSession.new()`.
    #[test]
    fn create_session_instance_constructs_object_and_binds_rust_instance() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_recorder();
        let _state = StateGuard::install(make_mock_interface());

        // SAFETY: mock interface installed; single-threaded test.
        let obj = unsafe { create_session_instance(ptr::null_mut()) };
        assert!(!obj.is_null(), "must return a Godot Object, not NULL");

        let events = EVENTS.lock().unwrap().clone();
        // The Object is constructed from the PARENT class name — Godot
        // rejects constructing a not-yet-instantiable extension class.
        assert!(
            events.iter().any(
                |e| matches!(e, Event::ObjectConstructed { class_name } if class_name == "Object")
            ),
            "must construct the parent Object; got {events:#?}",
        );

        let bound = events
            .iter()
            .find_map(|e| match e {
                Event::InstanceBound {
                    class_name,
                    object_is_null,
                    instance,
                } => Some((class_name.clone(), *object_is_null, *instance)),
                _ => None,
            })
            .expect("must bind the Rust instance via object_set_instance");
        assert_eq!(bound.0, "VokraSession", "bind under our extension class");
        assert!(!bound.1, "the bound Object must not be NULL");
        assert_ne!(bound.2, 0, "the bound instance pointer must not be NULL");

        // The bound pointer is the `SessionInstance` Godot will hand back
        // to every trampoline — and it must be reclaimable by our paired
        // `free_instance_func`.
        //
        // SAFETY: `bound.2` is the `Box::into_raw(SessionInstance)` we just
        // produced and have not freed.
        unsafe {
            free_session_instance(ptr::null_mut(), bound.2 as GDExtensionClassInstancePtr);
        }
    }

    #[test]
    fn create_stream_instance_constructs_object_and_binds_rust_instance() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_recorder();
        let _state = StateGuard::install(make_mock_interface());

        // SAFETY: mock interface installed; single-threaded test.
        let obj = unsafe { create_stream_instance(ptr::null_mut()) };
        assert!(!obj.is_null(), "must return a Godot Object, not NULL");

        let events = EVENTS.lock().unwrap().clone();
        let bound = events
            .iter()
            .find_map(|e| match e {
                Event::InstanceBound {
                    class_name,
                    instance,
                    ..
                } => Some((class_name.clone(), *instance)),
                _ => None,
            })
            .expect("must bind the Rust instance via object_set_instance");
        assert_eq!(bound.0, "VokraStream");

        // SAFETY: reclaim the allocation we just produced.
        unsafe {
            free_stream_instance(ptr::null_mut(), bound.1 as GDExtensionClassInstancePtr);
        }
    }

    /// FR-EX-08: when Godot cannot construct the Object, we surface NULL
    /// rather than fabricating an instance — and we must NOT have
    /// allocated (and leaked) the Rust side first.
    #[test]
    fn create_instance_returns_null_when_godot_refuses_to_construct() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_recorder();
        let mut iface = make_mock_interface();
        iface.classdb_construct_object = mock_classdb_construct_object_fails;
        let _state = StateGuard::install(iface);

        // SAFETY: mock interface installed; single-threaded test.
        let obj = unsafe { create_session_instance(ptr::null_mut()) };
        assert!(obj.is_null(), "a refused construction must yield NULL");

        let events = EVENTS.lock().unwrap().clone();
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, Event::InstanceBound { .. })),
            "no instance may be allocated or bound when construction fails",
        );
    }

    /// Pre-init / post-deinit: no resolved interface means we cannot build
    /// an Object at all. NULL is the honest answer (FR-EX-08); the old
    /// code path would have handed Godot a raw Rust pointer.
    #[test]
    fn create_instance_returns_null_when_extension_uninitialised() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        {
            let mut state = crate::EXTENSION_STATE.lock().unwrap();
            *state = None;
        }
        // SAFETY: no interface installed; the fn must bail before any
        // Godot call.
        let obj = unsafe { create_session_instance(ptr::null_mut()) };
        assert!(obj.is_null(), "uninitialised extension must yield NULL");
    }

    #[test]
    fn free_instance_null_is_noop() {
        // SAFETY: NULL branch is the exact case we're testing.
        unsafe {
            free_session_instance(ptr::null_mut(), ptr::null_mut());
            free_stream_instance(ptr::null_mut(), ptr::null_mut());
        }
    }
}
