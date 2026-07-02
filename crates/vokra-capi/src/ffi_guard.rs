//! FFI boundary guard and argument validation (M0-09-T05).
//!
//! Two safety obligations at the C boundary (NFR-RL-07, ADR-0003 §4):
//!
//! 1. **No panic may cross the boundary** — unwinding into C is undefined
//!    behaviour. [`guard`] wraps every `extern "C"` body in
//!    [`std::panic::catch_unwind`] and turns a panic into
//!    [`vokra_status_t::VOKRA_ERROR_PANIC`] plus a thread-local message.
//! 2. **Every raw argument is validated** — NULL pointers and non-UTF-8
//!    strings are normalised to `VOKRA_ERROR_INVALID_ARGUMENT` with an
//!    explanatory `vokra_last_error()` message, never a deref of a bad pointer.

use std::any::Any;
use std::ffi::{CStr, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::error::{fail_invalid, set_last_error, vokra_status_t};

/// Runs the body of an `extern "C"` function behind a panic firewall.
///
/// The body returns `Result<(), vokra_status_t>`: `Ok(())` becomes
/// `VOKRA_OK`, an `Err(status)` is returned as-is (its message was already
/// recorded via [`crate::error`]), and a panic is caught and reported as
/// `VOKRA_ERROR_PANIC`. `AssertUnwindSafe` is sound here because a caught panic
/// leaves no observable half-updated state across the boundary: outputs are
/// only written on the `Ok` path.
pub(crate) fn guard<F>(body: F) -> vokra_status_t
where
    F: FnOnce() -> Result<(), vokra_status_t>,
{
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(Ok(())) => vokra_status_t::VOKRA_OK,
        Ok(Err(status)) => status,
        Err(payload) => {
            set_last_error(&format!(
                "internal panic caught at FFI boundary: {}",
                describe_panic(&payload)
            ));
            vokra_status_t::VOKRA_ERROR_PANIC
        }
    }
}

/// Panic firewall for a `void`-returning function (destroy / free). A panic
/// while dropping a handle or buffer must not unwind into C; it is caught and
/// swallowed (there is no status channel for these functions).
pub(crate) fn guard_void<F: FnOnce()>(body: F) {
    let _ = catch_unwind(AssertUnwindSafe(body));
}

/// Best-effort human-readable text for a caught panic payload.
fn describe_panic(payload: &Box<dyn Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_owned()
    }
}

/// Borrows a required, NUL-terminated UTF-8 string from a C pointer.
///
/// # Safety
///
/// `ptr` must be `NULL` or point at a NUL-terminated C string valid for the
/// duration of the call.
pub(crate) unsafe fn required_str<'a>(
    ptr: *const c_char,
    name: &str,
) -> Result<&'a str, vokra_status_t> {
    if ptr.is_null() {
        return Err(fail_invalid(&format!("argument `{name}` must not be NULL")));
    }
    // SAFETY: `ptr` is non-null and, per the contract, a valid NUL-terminated
    // C string that outlives this call.
    let cstr = unsafe { CStr::from_ptr(ptr) };
    cstr.to_str()
        .map_err(|_| fail_invalid(&format!("argument `{name}` is not valid UTF-8")))
}

/// Borrows a required `&T` from a handle pointer (`NULL` → InvalidArgument).
///
/// # Safety
///
/// `ptr` must be `NULL` or a valid pointer to a live `T` that outlives the
/// borrow (e.g. a handle from `Box::into_raw`).
pub(crate) unsafe fn required_ref<'a, T>(
    ptr: *const T,
    name: &str,
) -> Result<&'a T, vokra_status_t> {
    // SAFETY: `as_ref` null-checks; a non-null `ptr` is a valid `T` per contract.
    match unsafe { ptr.as_ref() } {
        Some(r) => Ok(r),
        None => Err(fail_invalid(&format!("argument `{name}` must not be NULL"))),
    }
}

/// Borrows a required `&mut T` from a handle pointer (`NULL` → InvalidArgument).
///
/// # Safety
///
/// `ptr` must be `NULL` or a valid pointer to a live `T` with no other live
/// reference for the duration of the borrow.
pub(crate) unsafe fn required_mut<'a, T>(
    ptr: *mut T,
    name: &str,
) -> Result<&'a mut T, vokra_status_t> {
    // SAFETY: `as_mut` null-checks; a non-null `ptr` is a uniquely-owned `T`
    // per contract (the caller must not alias the handle).
    match unsafe { ptr.as_mut() } {
        Some(r) => Ok(r),
        None => Err(fail_invalid(&format!("argument `{name}` must not be NULL"))),
    }
}

/// Borrows a required `&[T]` from a `(ptr, len)` pair. A zero length yields an
/// empty slice without dereferencing `ptr` (so `NULL`/`len == 0` is allowed).
///
/// # Safety
///
/// If `len > 0`, `ptr` must be non-null and valid for reads of `len` elements
/// of `T` for the duration of the borrow.
pub(crate) unsafe fn required_slice<'a, T>(
    ptr: *const T,
    len: usize,
    name: &str,
) -> Result<&'a [T], vokra_status_t> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(fail_invalid(&format!(
            "argument `{name}` must not be NULL when its length is non-zero"
        )));
    }
    // SAFETY: `ptr` is non-null and, per the contract, valid for `len` reads of
    // `T` for the duration of the borrow.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// Checks that an out-pointer is non-null (no deref). Used before writing back
/// through `*mut *mut _` / `*mut size_t` out-params.
pub(crate) fn require_out_ptr<T>(ptr: *const T, name: &str) -> Result<(), vokra_status_t> {
    if ptr.is_null() {
        Err(fail_invalid(&format!("argument `{name}` must not be NULL")))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::vokra_last_error;
    use std::ffi::CString;

    #[test]
    fn guard_passes_ok_through() {
        let status = guard(|| Ok(()));
        assert_eq!(status, vokra_status_t::VOKRA_OK);
    }

    #[test]
    fn guard_returns_err_status() {
        let status = guard(|| Err(vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT));
        assert_eq!(status, vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
    }

    #[test]
    fn guard_catches_panic_without_aborting() {
        let status = guard(|| panic!("boom from inside the boundary"));
        assert_eq!(status, vokra_status_t::VOKRA_ERROR_PANIC);
        // The panic message is recorded for vokra_last_error().
        assert!(!vokra_last_error().is_null());
    }

    #[test]
    fn null_string_is_invalid_argument() {
        // SAFETY: NULL is an explicit branch; no deref happens.
        let r = unsafe { required_str(std::ptr::null(), "path") };
        assert_eq!(r.unwrap_err(), vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
    }

    #[test]
    fn valid_utf8_string_borrows() {
        let c = CString::new("hello").unwrap();
        // SAFETY: `c` is a live NUL-terminated C string for the call.
        let s = unsafe { required_str(c.as_ptr(), "text") }.unwrap();
        assert_eq!(s, "hello");
    }

    #[test]
    fn non_utf8_string_is_invalid_argument() {
        // 0xFF is not valid UTF-8; build a NUL-terminated buffer by hand.
        let bytes = [0xFFu8, 0x00u8];
        // SAFETY: buffer is NUL-terminated and outlives the call.
        let r = unsafe { required_str(bytes.as_ptr().cast(), "text") };
        assert_eq!(r.unwrap_err(), vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
    }

    #[test]
    fn zero_length_slice_ignores_null() {
        // SAFETY: len == 0 means `ptr` is never dereferenced.
        let s = unsafe { required_slice::<f32>(std::ptr::null(), 0, "pcm") }.unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn non_empty_slice_requires_non_null() {
        // SAFETY: NULL with len > 0 is the rejected branch; no deref happens.
        let r = unsafe { required_slice::<f32>(std::ptr::null(), 4, "pcm") };
        assert_eq!(r.unwrap_err(), vokra_status_t::VOKRA_ERROR_INVALID_ARGUMENT);
    }
}
