# vokra-godot — Godot 4.x GDExtension binding for Vokra

**State (2026-07-09, Wave 11)**: T02..T04 + T05..T10 + T13 landed. Class
registration (`classdb_register_extension_class3`), method binding
(`classdb_register_extension_class_method` for 6 methods across
`VokraSession` + `VokraStream`), signal declaration
(`classdb_register_extension_class_signal` for `asr_chunk` + `tts_chunk`),
panic firewall at every trampoline, and compile-time layout guards for
`GDExtensionClassCreationInfo3` (160 bytes) / `GDExtensionClassMethodInfo`
(88 bytes) are all wired against the Godot 4.3-stable header. Trampoline
runtime dispatch (Variant packing/unpacking to call real
`crate::asr::transcribe` etc.) is **honest scope-out to M3-18 owner smoke**:
each trampoline exists with correct signature + arity enforcement + panic
firewall + `catch_unwind`, and returns `InvalidMethod` with a documented
"runtime dispatch pending" marker until the real Variant plumbing lands.
T11..T18 (Windows/macOS/Android crossbuild + AssetLib packaging + demo
scenes + release CI) remain follow-up.

## What it is

The Godot 4.x GDExtension surface for the [Vokra](https://github.com/ayutaz/vokra)
speech-first inference runtime. Exposes a `VokraSession` Godot Object
class (once T05 lands) that wraps the Vokra C ABI (`include/vokra.h`,
cbindgen-generated from `crates/vokra-capi`) and dispatches to native
Whisper base ASR, piper-plus TTS, and Silero VAD v5 engines.

Sister binding: [`bindings/unity/com.vokra.unity`](../../bindings/unity/com.vokra.unity)
(the Unity UPM package that landed in M2-11).

## What makes it different

- **No `godot-cpp`, no `gdext-rs`, no bindgen** — hand-written `extern "C"`
  bridge over `gdextension_interface.h`. Matches Vokra's Metal / CUDA raw-FFI
  posture (`docs/adr/0011-godot-gdextension.md` §D1/D3).
- **Excluded from the Vokra root workspace** — mirrors
  `integrations/vokra-piper-g2p/` and `integrations/vokra-server/`. Root
  `Cargo.lock` stays zero-dep (NFR-DS-02).
- **`silent CPU fallback 禁止`** (FR-EX-08) — every backend error
  propagates through `vokra_last_error()` into a Godot-side
  `VokraError` (see `src/error.rs`); the binding never retries on a
  different backend.
- **Panic firewall at every C boundary** — the GDExtension entry point,
  init/deinit callbacks, and future method-binding trampolines wrap
  their Rust body in `error::catch_panic`; the workspace
  `panic = "unwind"` policy (root `Cargo.toml`) makes this functional.

## Platform matrix

Scope of M3-11 (ADR-0011 §D4):

| Target | Cargo triple | Artifact |
|--------|-------------|----------|
| Windows x86_64 | `x86_64-pc-windows-msvc`   | `vokra_godot.dll` |
| macOS (universal) | `aarch64-apple-darwin` + `x86_64-apple-darwin` | `libvokra_godot.dylib` |
| Linux x86_64   | `x86_64-unknown-linux-gnu` | `libvokra_godot.so` |
| Android arm64  | `aarch64-linux-android`     | `libvokra_godot.so` |

iOS and Web (HTML5) are deferred to M4.

## Local dev

```
cd integrations/vokra-godot
cargo build --release            # host cdylib
cargo test                       # 46 unit tests as of Wave 11 (T05..T10 + T13)
```

or use the FR-TL-04 helper:

```
bash scripts/build-godot-gdextension.sh          # host-only cdylib sync
bash scripts/build-godot-gdextension.sh --pack   # + assemble AssetLib zip
```

The zip lands at `dist/godot/vokra-godot-<version>.zip`.
**This is dev iteration ONLY**; the canonical release-signed zip that
consumers install comes from the CD job in `.github/workflows/release.yml`
(NFR-MT-08).

## Layout

```
integrations/vokra-godot/
├── Cargo.toml              # excluded workspace (own Cargo.lock)
├── README.md               # this file
├── vokra.gdextension       # AssetLib config template (ADR-0011 §D9)
└── src/
    ├── lib.rs              # crate root + GDExtension entry point + EXTENSION_STATE
    ├── error.rs            # panic firewall + status → VokraError
    ├── session.rs          # RAII wrapper over vokra_session_t
    ├── asr.rs              # transcribe(...)
    ├── tts.rs              # synthesize(...)
    ├── vad.rs              # VokraStream RAII + push_pcm/poll/...
    ├── registry.rs         # T05 classdb_register_extension_class3 + methods + signals
    ├── trampoline.rs       # T06 method call trampolines (catch_unwind firewalled)
    └── ffi/
        ├── mod.rs
        ├── capi.rs         # extern "C" for the Vokra C ABI
        ├── gdextension.rs  # extern "C" for gdextension_interface.h (4.3-stable) subset
        └── interface.rs    # resolved GDExtension interface table (from p_get_proc_address)
```

## Godot usage (target GDScript surface, finalised in T05)

```gdscript
extends Control

func _ready() -> void:
    var session := VokraSession.new()
    session.load_model("res://models/whisper-base.gguf")

    var pcm: PackedFloat32Array = _load_pcm("res://audio/jfk.wav")
    var text: String = session.transcribe(pcm, 16000)
    $Label.text = text
```

## License

Apache-2.0 (workspace-wide policy; see `LICENSE` at repo root).
Godot Engine itself is MIT; this binding does not link `godot-cpp`.

## Related documents

- [`docs/adr/0011-godot-gdextension.md`](../../docs/adr/0011-godot-gdextension.md) — design record
- [`docs/tickets/m3/M3-11-godot-gdextension.md`](../../docs/tickets/m3/M3-11-godot-gdextension.md) — ticket list
- [`docs/adr/0003-c-abi-design.md`](../../docs/adr/0003-c-abi-design.md) — the Vokra C ABI we wrap
- [`docs/adr/0007-unity-official-plugin.md`](../../docs/adr/0007-unity-official-plugin.md) — sister binding
