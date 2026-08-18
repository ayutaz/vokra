# vokra (Python binding)

Python binding sources for **Vokra**, implemented as a thin
[`ctypes`](https://docs.python.org/3/library/ctypes.html) layer over the C ABI.
The intended wheel bundles `libvokra.dylib`, `libvokra.so`, or `vokra.dll` and
keeps third-party Python runtime dependencies at zero.

## Status: pre-alpha, not a current release surface

**Reviewed:** 2026-08-18 against `main` at `6d64fdf`.

The package metadata is `0.1.0.dev0`; this checkout must not be documented as
an installed `vokra==0.1.0` release. The source tree contains the loader,
session, stream, audio, and error modules, but `src/vokra/__init__.py` currently
exports only `__version__`. Therefore examples such as
`from vokra import Session` and `vokra.__abi_version__` are not part of the
current package surface.

There is also known C-ABI drift:

- `include/vokra.h` currently exports 41 `vokra_*` functions;
- the checked-in `src/vokra/_bindings.py` contains the earlier 14-function
  ASR/TTS/streaming subset;
- the generator's declaration parser recognizes only 39 of the 41 current
  functions, and generation then stops at `vokra_aec_config_t` because that
  struct is not mapped.

Do not publish a wheel or claim full C-ABI coverage until the generator supports
the current structs, `_bindings.py` is regenerated, the intended public names
are exported, and the binding test/release gates pass. The current generated
file may still be useful for internal development against its 14-function
subset, but it is not a 1:1 representation of the current header.

## Pre-1.0 compatibility

The Vokra C ABI remains unfrozen until v1.0 GA (IF-01). Once wheels are
released, callers must pin an exact pre-1.0 version; a minor release may change
the ABI without a deprecation window.

## Design invariants

- **Zero third-party runtime dependencies (NFR-DS-02).** `dependencies = []`
  in `pyproject.toml`. NumPy is optional interop and is imported lazily.
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
uv sync --project bindings/python --extra dev
uv run --project bindings/python --extra dev pytest bindings/python/tests
```

The generated binding check currently fails honestly at
`vokra_aec_config_t`; after adding support for the current C structs, run:

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
│   ├── __init__.py        # currently exports __version__ only
│   ├── _native.py         # ctypes.CDLL loader
│   ├── _bindings.py       # generated; currently the 14-function subset
│   ├── _handles.py        # opaque handle wrappers
│   ├── session.py         # internal Session wrapper
│   ├── stream.py          # internal Stream wrapper
│   ├── audio.py           # WAV I/O + optional NumPy interop
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

This only creates a development artifact. It does not resolve the ABI drift or
authorize publication. Platform wheel targets in `pyproject.toml` and CI are
release goals, not evidence that a compatible wheel is currently published.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
