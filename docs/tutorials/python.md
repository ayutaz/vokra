# Python bindings tutorial

**English** | [日本語](python.ja.md)

> **Implementation status (reviewed 2026-08-22): source-complete, unpublished.**
> The package root exports `Session`, `Stream`, `Event`, and typed errors, and
> its generated table covers all 48 current C functions. No PyPI release has
> been verified or authorized. See the
> [binding README](../../bindings/python/README.md) for the exact gates.

## What exists today

- Package metadata: `0.1.0.dev0`, Python 3.9–3.12.
- Runtime dependency list: empty; NumPy is optional interop.
- Public source API: `Session`, `Stream`, `Event`, and a nine-class error
  hierarchy. Audio-file decoding remains caller-owned.
- Generated FFI table: all 48 functions in the current C header, plus its four
  enums, two concrete structures, and eight opaque handles.
- CI contract: the required `license` job checks generator drift; each wheel
  smoke asserts the public names, table size, and native symbol load.

The release workflow builds four wheels: Linux x86_64 (`manylinux_2_28`),
macOS arm64, macOS x86_64, and Windows x86_64. macOS does not claim universal2,
and the source loader's Linux aarch64 support is not yet a released wheel. A
target declared in packaging or CI is not proof that a compatible wheel has
been released.

## Development setup

From a source checkout, use uv:

```sh
uv run --no-project --python 3.12 --with pytest \
  python -m pytest bindings/python/tests
```

Build and stage the platform C library before testing a real `ctypes` load:

```sh
cargo build --release -p vokra-capi
cp target/release/libvokra.dylib bindings/python/src/vokra/_lib/  # macOS
# Linux: copy libvokra.so; Windows: copy vokra.dll.
```

## Source public API

The source surface is `Session`, `Stream`, `Event`, and `VokraError`
subclasses. WAV loading is not part of the package API; the example therefore
uses a local standard-library helper. It is runnable after the matching native
library has been built and staged as shown above:

```python
import struct
import wave

from vokra import Session


def read_pcm16_wav_mono(path: str) -> tuple[list[float], int]:
    with wave.open(path, "rb") as source:
        wav_format = (
            source.getnchannels(),
            source.getsampwidth(),
            source.getcomptype(),
        )
        if wav_format != (1, 2, "NONE"):
            raise ValueError("expected an uncompressed mono 16-bit PCM WAV")
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    pcm = [sample / 32768.0 for (sample,) in struct.iter_unpack("<h", frames)]
    return pcm, sample_rate


pcm, sample_rate = read_pcm16_wav_mono("speech.wav")
with Session.open("whisper-base.gguf") as session:
    text = session.transcribe(pcm, sample_rate)
```

The source-side first three conditions are implemented. Release promotion
still requires the external evidence in the final two:

1. the generator handles every type in `include/vokra.h` and emits all current
   functions;
2. the generated drift check passes;
3. `src/vokra/__init__.py` exports and tests the documented names;
4. final-head wheels with matching native libraries pass load, ASR/TTS,
   streaming, error, and platform smoke tests;
5. the exact release version and destination are explicitly authorized and
   verified after upload.

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
