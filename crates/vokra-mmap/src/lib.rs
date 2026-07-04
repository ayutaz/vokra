//! # vokra-mmap
//!
//! True `mmap`-backed GGUF loading for Vokra (FR-LD-01 / NFR-PF-11): map a
//! model file into memory read-only and parse it **without copying** and
//! **without eagerly touching every page**, so cold start is lazy — the kernel
//! faults tensor pages in only as the runtime reads them.
//!
//! # Why this is a separate crate
//!
//! `vokra-core` is `unsafe`-free (workspace lint `unsafe_code = "deny"`), and a
//! real memory map requires `unsafe`. So the mapping lives here, behind a
//! crate-root `#![allow(unsafe_code)]`, and plugs into the core loader through
//! the [`vokra_core::gguf::AsBytes`] trait: [`Mmap`] implements `AsBytes`, and
//! [`open_gguf`] hands the boxed mapping to
//! [`GgufFile::from_external`]. The
//! **exact same** parser and zero-copy accessors then run over the mapped bytes
//! — no parsing logic is duplicated or changed.
//!
//! # Zero external dependencies (NFR-DS-02)
//!
//! No `libc` / `memmap2` crate is used. The platform primitives are declared
//! inline in `unsafe extern` blocks:
//!
//! - **Unix**: POSIX `mmap` / `munmap` (`PROT_READ | MAP_PRIVATE`);
//! - **Windows**: `CreateFileMappingW` / `MapViewOfFile` / `UnmapViewOfFile` /
//!   `CloseHandle` (`PAGE_READONLY` / `FILE_MAP_READ`).
//!
//! `std` already links the C runtime / `kernel32` that export these symbols, so
//! calling them adds nothing to `Cargo.lock`.
//!
//! # Safety model
//!
//! [`Mmap`] owns a **read-only, immutable** mapping for its whole lifetime and
//! keeps the backing [`File`] alive alongside it. Because the
//! bytes are never written through any handle, lending them as `&[u8]`
//! ([`AsBytes::bytes`]) and sharing the handle
//! across threads (`Send + Sync`) are sound. The mapping is released once, in
//! [`Drop`].

#![allow(unsafe_code)]

use std::fs::File;
use std::io;
use std::path::Path;
use std::slice;

use vokra_core::gguf::{AsBytes, GgufError, GgufFile};

/// A read-only memory mapping of a whole file.
///
/// Constructed with [`Mmap::open`]. The mapped bytes are borrowed with
/// [`AsBytes::bytes`] and released in [`Drop`]. The mapping is immutable, so the
/// handle is `Send + Sync` and can back a shared [`GgufFile`].
pub struct Mmap {
    /// The mapped file, kept open for the lifetime of the mapping. It is only
    /// held for that lifetime (and its `Drop` closing the descriptor/handle),
    /// never read through — hence `dead_code` is allowed.
    #[allow(dead_code)]
    file: File,
    /// Base address of the mapping (`len` valid read-only bytes follow).
    ptr: *const u8,
    /// Length of the mapping in bytes (always non-zero once constructed).
    len: usize,
}

impl Mmap {
    /// Maps `path` into memory read-only and returns the mapping.
    ///
    /// Returns an [`io::Error`] if the file cannot be opened or mapped. A
    /// zero-length file is rejected with [`io::ErrorKind::InvalidInput`]
    /// (POSIX `mmap` rejects a zero length, and an empty file is never a valid
    /// GGUF anyway).
    pub fn open(path: impl AsRef<Path>) -> io::Result<Mmap> {
        let path = path.as_ref();
        #[cfg(unix)]
        {
            unix::open(path)
        }
        #[cfg(windows)]
        {
            windows::open(path)
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = path;
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "vokra-mmap requires a Unix or Windows target",
            ))
        }
    }
}

// SAFETY: a `Mmap` owns a read-only, immutable memory mapping; the bytes are
// never mutated through this handle or any other. Moving the handle between
// threads (`Send`) and sharing `&Mmap` across threads (`Sync`) therefore cannot
// introduce a data race — the raw pointer is only ever read, and the backing
// `File` is itself `Send + Sync`.
unsafe impl Send for Mmap {}
// SAFETY: see the `Send` impl — a read-only, never-written mapping is sound to
// share by shared reference across threads.
unsafe impl Sync for Mmap {}

impl AsBytes for Mmap {
    fn bytes(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `open` established `ptr` as the base of `len` contiguous,
        // initialised, read-only bytes from a successful mapping that lives
        // until `self` is dropped. The returned slice borrows `&self`, so it
        // cannot outlive the mapping, and the mapping is never written through
        // any handle, so no `&mut` ever aliases these bytes.
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        if self.ptr.is_null() || self.len == 0 {
            return;
        }
        #[cfg(unix)]
        unix::unmap(self.ptr, self.len);
        #[cfg(windows)]
        windows::unmap(self.ptr, self.len);
    }
}

impl std::fmt::Debug for Mmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Omit the raw pointer / bytes; the length is the only useful field.
        f.debug_struct("Mmap").field("len", &self.len).finish()
    }
}

/// Memory-maps a GGUF file read-only and parses it into a [`GgufFile`].
///
/// This is the true-`mmap` counterpart to
/// [`GgufFile::open`] (which reads the whole
/// file into an owned buffer): the returned [`GgufFile`] borrows its tensor
/// payloads straight out of the mapping, so weights are faulted in lazily and
/// never copied. The mapping is kept alive by the returned [`GgufFile`].
///
/// Returns [`GgufError::Io`] if the file cannot be opened or mapped, or a parse
/// error variant for malformed GGUF content.
pub fn open_gguf(path: impl AsRef<Path>) -> Result<GgufFile, GgufError> {
    let mmap = Mmap::open(path).map_err(GgufError::Io)?;
    GgufFile::from_external(Box::new(mmap))
}

#[cfg(unix)]
mod unix {
    //! POSIX `mmap` / `munmap` binding (no `libc` crate).

    use std::fs::File;
    use std::io;
    use std::os::raw::{c_int, c_void};
    use std::os::unix::io::AsRawFd;
    use std::path::Path;
    use std::ptr;

    use super::Mmap;

    /// Pages may be read (`PROT_READ`). Identical value on Linux and BSD/macOS.
    const PROT_READ: c_int = 1;
    /// Private copy-on-write mapping (`MAP_PRIVATE`); we never write, so this is
    /// purely a read-only view. Identical value on Linux and BSD/macOS.
    const MAP_PRIVATE: c_int = 2;

    // Every Unix target Vokra ships (macOS / iOS / Linux / Android on
    // arm64 / x86_64) is LP64, where `off_t` is 64-bit; the offset is 0 here
    // regardless. `std` links libc, which exports these symbols.
    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: i64,
        ) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> c_int;
    }

    pub(super) fn open(path: &Path) -> io::Result<Mmap> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        // `mmap` rejects a zero length with EINVAL, and a 0-byte file is never a
        // valid GGUF (the header alone is 24 bytes): reject it up front so the
        // non-null invariant below always holds.
        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot memory-map an empty file",
            ));
        }
        let len = usize::try_from(len).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "file is larger than the address space",
            )
        })?;
        let fd = file.as_raw_fd();
        // SAFETY: `fd` is a valid, open, readable descriptor owned by `file`
        // (kept alive in the returned `Mmap`). A null `addr` lets the kernel
        // choose the address; `len` is non-zero; `PROT_READ | MAP_PRIVATE` with
        // offset 0 requests a read-only view of the whole file. Failure is
        // reported as `MAP_FAILED`, which is checked before the pointer is used.
        let ptr = unsafe { mmap(ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, fd, 0) };
        // `MAP_FAILED` is `(void *) -1`, i.e. the all-ones address.
        if ptr.addr() == usize::MAX {
            return Err(io::Error::last_os_error());
        }
        // A successful `mmap` never returns null, but never trust that blindly.
        if ptr.is_null() {
            return Err(io::Error::other("mmap returned a null address"));
        }
        Ok(Mmap {
            file,
            ptr: ptr as *const u8,
            len,
        })
    }

    pub(super) fn unmap(ptr: *const u8, len: usize) {
        // SAFETY: `ptr` / `len` are exactly the base / length of the live
        // mapping returned by `open`'s `mmap`, not previously unmapped;
        // `munmap` over that region is its matching teardown, and `Drop` invokes
        // this exactly once.
        unsafe {
            munmap(ptr as *mut c_void, len);
        }
    }
}

#[cfg(windows)]
#[allow(non_snake_case)]
mod windows {
    //! Win32 file-mapping binding (no `winapi` / `windows-sys` crate).

    use std::fs::File;
    use std::io;
    use std::os::raw::c_void;
    use std::os::windows::io::AsRawHandle;
    use std::path::Path;
    use std::ptr;

    use super::Mmap;

    type Handle = *mut c_void;
    type Dword = u32;
    type Bool = i32;

    /// Read-only page protection for the section object.
    const PAGE_READONLY: Dword = 0x02;
    /// Read access for the mapped view.
    const FILE_MAP_READ: Dword = 0x04;

    // `std` links `kernel32`, which exports these symbols. `extern "system"` is
    // the Win32 calling convention (`stdcall` on x86, plain on x86-64/arm64).
    unsafe extern "system" {
        fn CreateFileMappingW(
            hFile: Handle,
            lpFileMappingAttributes: *mut c_void,
            flProtect: Dword,
            dwMaximumSizeHigh: Dword,
            dwMaximumSizeLow: Dword,
            lpName: *const u16,
        ) -> Handle;
        fn MapViewOfFile(
            hFileMappingObject: Handle,
            dwDesiredAccess: Dword,
            dwFileOffsetHigh: Dword,
            dwFileOffsetLow: Dword,
            dwNumberOfBytesToMap: usize,
        ) -> *mut c_void;
        fn UnmapViewOfFile(lpBaseAddress: *const c_void) -> Bool;
        fn CloseHandle(hObject: Handle) -> Bool;
    }

    pub(super) fn open(path: &Path) -> io::Result<Mmap> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        if len == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot memory-map an empty file",
            ));
        }
        let len = usize::try_from(len).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "file is larger than the address space",
            )
        })?;
        let handle = file.as_raw_handle() as Handle;
        // SAFETY: `handle` is a valid open file handle owned by `file`. Null
        // attributes / null name with a zero maximum size ask Windows to create
        // a read-only section sized to the whole file. The returned handle is
        // checked for null before use.
        let hmap = unsafe {
            CreateFileMappingW(handle, ptr::null_mut(), PAGE_READONLY, 0, 0, ptr::null())
        };
        if hmap.is_null() {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `hmap` is the valid section handle just created; offset 0 and
        // length 0 map the entire file for reading.
        let ptr = unsafe { MapViewOfFile(hmap, FILE_MAP_READ, 0, 0, 0) };
        // The view holds its own reference to the section, so the section handle
        // can be closed immediately; the view stays valid until `UnmapViewOfFile`.
        // SAFETY: `hmap` is a valid handle no longer needed once the view exists.
        unsafe {
            CloseHandle(hmap);
        }
        if ptr.is_null() {
            return Err(io::Error::last_os_error());
        }
        Ok(Mmap {
            file,
            ptr: ptr as *const u8,
            len,
        })
    }

    pub(super) fn unmap(ptr: *const u8, _len: usize) {
        // SAFETY: `ptr` is the base address returned by `MapViewOfFile` in
        // `open`, not previously unmapped; `UnmapViewOfFile` is its matching
        // teardown, and `Drop` invokes this exactly once.
        unsafe {
            UnmapViewOfFile(ptr as *const c_void);
        }
    }
}
