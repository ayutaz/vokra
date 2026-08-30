# vokra (Python binding)

Python binding sources for **Vokra**, implemented as a thin
[`ctypes`](https://docs.python.org/3/library/ctypes.html) layer over the C ABI.
The intended wheel bundles `libvokra.dylib`, `libvokra.so`, or `vokra.dll` and
keeps third-party Python runtime dependencies at zero.

## Status: source implementation current, package unpublished

**Reviewed:** 2026-08-30 against GitHub `main`
`41ce9ffdd4b0959497f55afa5016822f77a8a7b6`, the pre-documentation code
baseline branch `feat/mac-cpu-metal-full-coverage-2026-08-28` at
`c64b7b7237b70c5dc70ffd60394af325016d9a8d`, and the generated C header.

The workspace is `0.2.0` development with no Git tag or published release;
the package metadata remains `0.1.0.dev0` for unpublished source wheels. This
checkout must not be documented as an installed `vokra==0.1.0` release. The source tree exports `Session`,
`Stream`, `Event`, and the typed `VokraError` hierarchy without loading the
native library at import time. `vokra.__abi_version__` is not exposed: the C
header has a runtime version function, not a separately versioned ABI symbol.

The source-side C-ABI drift is closed in this worktree:

- `include/vokra.h` is the canonical `vokra_*` function set;
- `src/vokra/_bindings.py` contains exactly one prototype for every function;
- the generator discovers all four enums, both concrete structs, and all nine
  opaque handles, including the `uint8_t`, `uint64_t`, plain-`bool`, and
  struct-pointer shapes that previously blocked generation;
- the required `license` job runs the uv-only drift check, and the wheel smoke
  loads the matching native library after asserting the public API and exact
  generated table.

This is still not a publication claim. A final branch CI run must prove the
four release wheels against their bundled libraries, and PyPI/TestPyPI upload
requires separate authorization and destination verification.

The release matrix is Linux x86_64 (`manylinux_2_28`), macOS arm64, macOS
x86_64, and Windows x86_64. The two macOS architectures are separate truthful
wheels; this project does not claim a universal2 binary. The source loader also
recognizes Linux aarch64, but that wheel is not in the current release matrix.

## Pre-1.0 compatibility

The Vokra C ABI remains unfrozen until v1.0 GA (IF-01). Once wheels are
released, callers must pin an exact pre-1.0 version; a minor release may change
the ABI without a deprecation window.

## Design invariants

- **Zero third-party runtime dependencies (NFR-DS-02).** `dependencies = []`
  in `pyproject.toml`. NumPy is optional caller-side interop and is not imported
  by the package.
- **Explicit errors, no silent fallback (FR-EX-08).** A non-OK C status maps to
  a `VokraError` subclass. Unsupported or unavailable GPU work must not fall
  back to CPU silently.
- **Apache-2.0.** The binding adds no Rust crate or runtime Python dependency.
- **Locale-independent (NFR-RL-01).** Numeric FFI values use `ctypes` scalars;
  the wrapper does not parse them with locale-sensitive conversion.

## Error contract

The shipping `vokra_status_t` enum has ten values including `VOKRA_OK`; the
nine error values map to nine subclasses, with unknown future values falling
back to `VokraError`:

| `vokra_status_t` | Python exception |
|---|---|
| `VOKRA_OK` | no exception |
| `VOKRA_ERROR_IO` | `VokraIoError` |
| `VOKRA_ERROR_MODEL_LOAD` | `VokraModelLoadError` |
| `VOKRA_ERROR_UNSUPPORTED_OP` | `VokraUnsupportedOpError` |
| `VOKRA_ERROR_BACKEND_UNAVAILABLE` | `VokraBackendUnavailableError` |
| `VOKRA_ERROR_INVALID_ARGUMENT` | `VokraInvalidArgumentError` |
| `VOKRA_ERROR_GRAPH_VALIDATION` | `VokraGraphValidationError` |
| `VOKRA_ERROR_NOT_IMPLEMENTED` | `VokraNotImplementedError` |
| `VOKRA_ERROR_PANIC` | `VokraPanicError` |
| `VOKRA_ERROR_OTHER` | `VokraOtherError` |

`vokra_last_error()` is thread-local and must be read in the same call frame as
the failing native call.

## Development setup

Use uv for all Python work. From the repository root:

```sh
uv sync --python 3.12 --project bindings/python --extra dev
uv run --python 3.12 --project bindings/python --extra dev \
  python -m pytest bindings/python/tests
```

Run the same generated binding check used by CI:

```sh
uv run --no-project --python 3.12 python \
  bindings/python/scripts/gen-py-bindings.py --check --header include/vokra.h
```

## Layout

```text
bindings/python/
├── pyproject.toml
├── README.md
├── scripts/
│   ├── gen-py-bindings.py
│   └── check-py-bindings.sh
├── src/vokra/
│   ├── __init__.py        # lazy public Session/Stream/Event/error exports
│   ├── _native.py         # ctypes.CDLL loader
│   ├── _bindings.py       # generated full C-function ctypes table
│   ├── _handles.py        # opaque handle wrappers
│   ├── session.py         # Session lifecycle + ASR/TTS wrapper
│   ├── stream.py          # Stream lifecycle + push/poll/event wrapper
│   ├── errors.py          # VokraError hierarchy
│   └── _lib/              # native library injected by CI
└── tests/
```

## Building an unpublished development wheel

First build the native C ABI and place the platform library under
`bindings/python/src/vokra/_lib/`:

```sh
cargo build --release -p vokra-capi
cp target/release/libvokra.dylib bindings/python/src/vokra/_lib/  # macOS
# Linux: copy libvokra.so; Windows: copy vokra.dll.
uv build --project bindings/python --wheel --out-dir bindings/python/dist
```

This only creates a development artifact and does not authorize publication.
The build fails if the host library is absent or a requested tag names another
OS/architecture. CI additionally runs auditwheel/delocate/delvewheel, validates
the archive and binary architecture, and clean-installs each wheel on Python
3.9 and 3.12. Passing those gates is still not evidence that a compatible wheel
has been published.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
