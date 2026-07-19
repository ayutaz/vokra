# Python bindings tutorial

**English** | [日本語](python.ja.md)

This tutorial walks through installing and using the **Vokra Python
binding** (`bindings/python`) — a thin `ctypes` wrapper over the C ABI,
with zero third-party Python runtime dependencies.

## 1. Install

```sh
pip install "vokra==0.1.0"
```

Pin the exact version — the Vokra C ABI is not frozen before v1.0 GA
(IF-01), so upgrades may require code changes.

**Distribution**: wheels are published on PyPI via `cibuildwheel`. Supported
platforms:

| OS      | Arch                       | Wheel tag                |
| ------- | -------------------------- | ------------------------ |
| Linux   | x86_64                     | `manylinux_2_28_x86_64`  |
| Linux   | aarch64                    | `manylinux_2_28_aarch64` |
| macOS   | universal2 (x86_64 + arm64)| `macosx_11_0_universal2` |
| Windows | x86_64                     | `win_amd64`              |

Python 3.9 – 3.12 supported. iOS / Android / WASM are out of scope for the
Python binding — use the Swift / Kotlin bindings (planned) or the WASM
target (v1.0-rc+, formerly v1.5+) instead.

## 2. Convert a model

Vokra loads GGUF only. Convert an upstream Whisper checkpoint using the
CLI (see [Getting Started](../getting-started.md) for the full recipe):

```sh
cargo run --release --bin vokra-cli -- convert \
  --model whisper \
  --input whisper-base/model.safetensors \
  --output whisper-base.gguf
```

## 3. ASR — transcribe from Python

```python
from vokra import Session

session = Session.open("whisper-base.gguf")
try:
    # `pcm` is mono float32 in [-1, 1]; any sequence works
    # (list, tuple, array.array('f'), or numpy.ndarray via .tolist()).
    with open("speech.wav", "rb") as f:
        # See bindings/python/src/vokra/audio.py for a bundled WAV reader.
        pcm, sample_rate = read_wav_mono_f32(f)
    text = session.transcribe(pcm, sample_rate)
    print(text)
finally:
    session.close()  # or use `with Session.open(...) as session:`
```

## 4. TTS — synthesize from text

```python
from vokra import Session

with Session.open("en_US-lessac-medium.gguf") as session:
    pcm, sample_rate = session.synthesize("Hello from Vokra.")
    write_wav_mono_f32("hello.wav", pcm, sample_rate)
```

`pcm` is a plain Python `list[float]`; convert to `numpy.ndarray` at the
call site if that fits your pipeline. The binding deliberately does not
import `numpy` — it stays pure `ctypes` at the FFI seam (see the "Zero
third-party dependency" invariant in `bindings/python/README.md`).

## 5. Error contract (no silent fallback)

Every C ABI call is checked in the same Python frame; `vokra_last_error()`
is read immediately on any non-OK status. The 10 status codes map 1:1 to
exception classes:

```python
from vokra import (
    Session,
    VokraError,           # base — catches everything
    VokraInvalidArgument, # e.g. sample-rate mismatch
    VokraModelError,      # malformed / missing GGUF
    VokraUnsupportedBackend,  # backend requested but not linked
    # ...see bindings/python/src/vokra/errors.py for the full list
)

try:
    with Session.open("some-model.gguf") as s:
        s.transcribe(pcm, 16000)
except VokraUnsupportedBackend as e:
    # The GPU backend a model was built for is not available in this build
    # — FR-EX-08 forbids silent fallback, so this is an explicit signal.
    print(f"backend not available: {e}")
except VokraError as e:
    print(f"vokra error: {e}")
```

## 6. Thread safety

- `Session` is `Send + Sync` on the C side and safe to share across
  Python threads (the GIL adds an extra layer of serialization for the
  Python-level attribute access).
- `vokra_last_error()` is thread-local; do not read it from a different
  thread than the failing call.
- The wrapper's `Session.__del__` releases the native handle if the user
  forgot to call `close()` (RAII). Prefer the `with` block.

## 7. Verifying the wheel

The Python wheel bundles a prebuilt `libvokra` in `src/vokra/_lib/`.
Verify the load path with:

```python
import vokra
print(vokra.__version__)             # e.g. "0.1.0"
print(vokra.__abi_version__)         # must match the loaded native lib
```

At import time, an ABI-version mismatch raises `VokraError` immediately —
this is intentional (a stale wheel with an incompatible C ABI is a bug,
not something to silently paper over).

## 8. Debugging

- **`OSError: cannot find libvokra.*`**: the wheel was not built for your
  platform, or `src/vokra/_lib/` is empty. Reinstall with a wheel that
  matches your OS / arch, or build from source (see
  [`bindings/python/README.md`](../../bindings/python/README.md#build-from-source)).
- **`VokraInvalidArgument: sample rate 16000 != model rate 22050`**: the
  model's front-end sample rate does not match the PCM. M0 does not
  resample; convert the audio (via `soxr`, `librosa`, whatever's already
  in your stack) before calling `transcribe`.

## Next steps

- **Server**: run [`integrations/vokra-server`](../../integrations/vokra-server)
  and use `openai-python` / `faster-whisper` clients unmodified — Vokra
  is a drop-in for their HTTP shape.
- **Migration**: if you are coming from `faster-whisper` / `piper-py`,
  see [Migration Guide](../migration-guide.md).
- **License**: Apache-2.0; the wheel bundles native code with the same
  license. See `LICENSE` and `NOTICE` inside the wheel.
