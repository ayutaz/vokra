# Python bindings tutorial

**English** | [日本語](python.ja.md)

> **Implementation status (reviewed 2026-08-18): pre-alpha.** The repository
> contains internal `ctypes` modules, but the package root currently exports
> only `__version__`. No current release should be documented as supporting
> `from vokra import Session` yet. See the
> [binding README](../../bindings/python/README.md) for the exact drift and
> completion gates.

## What exists today

- Package metadata: `0.1.0.dev0`, Python 3.9–3.12.
- Runtime dependency list: empty; NumPy is optional interop.
- Internal modules: native loader, handle/session/stream wrappers, WAV helpers,
  and a nine-class error hierarchy.
- Generated FFI table: the earlier 14-function ASR/TTS/streaming subset.
- Current C header: 41 functions. The generator parser recognizes 39, then
  generation is blocked on the newer `vokra_aec_config_t` structure mapping.

The planned wheel targets are Linux x86_64/aarch64, macOS universal2, and
Windows x86_64. A target declared in packaging or CI is not proof that a
compatible wheel has been released.

## Development setup

From a source checkout, use uv:

```sh
uv sync --project bindings/python --extra dev
uv run --project bindings/python --extra dev pytest bindings/python/tests
```

Build and stage the platform C library before testing a real `ctypes` load:

```sh
cargo build --release -p vokra-capi
cp target/release/libvokra.dylib bindings/python/src/vokra/_lib/  # macOS
# Linux: copy libvokra.so; Windows: copy vokra.dll.
```

## Intended public API

After the ABI generator and package exports are brought current, the intended
surface is `Session`, `Stream`, WAV helpers, and `VokraError` subclasses. The
following shape is illustrative and deliberately not presented as runnable on
the current checkout:

```python
from vokra import Session, read_wav_mono_f32

pcm, sample_rate = read_wav_mono_f32("speech.wav")
with Session.open("whisper-base.gguf") as session:
    text = session.transcribe(pcm, sample_rate)
```

Before promoting this example to a quick start, all of these conditions must
hold:

1. the generator handles every type in `include/vokra.h` and emits all current
   functions;
2. the generated drift check passes;
3. `src/vokra/__init__.py` exports and tests the documented names;
4. a wheel built with the matching native library passes load, ASR/TTS,
   streaming, error, and platform smoke tests;
5. the exact released version and download location are verified.

## Error and threading contracts

The C enum has ten values including `VOKRA_OK`; its nine error values map to
nine subclasses in `src/vokra/errors.py`. `vokra_last_error()` is thread-local
and must be captured in the same call frame as the failure. Unsupported work
must raise explicitly rather than silently fall back to CPU.

`Stream` is caller-serialized and should not be shared across threads without
a lock. Close streams before their parent session; nested context managers are
the intended ownership pattern.

## Next steps

- Current binding status and source-wheel instructions:
  [bindings/python/README.md](../../bindings/python/README.md)
- Native CLI path that is available independently of the Python package:
  [Getting Started](../getting-started.md)
- HTTP compatibility path for Python clients:
  [`integrations/vokra-server`](../../integrations/vokra-server)
