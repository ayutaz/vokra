# Godot (GDExtension) tutorial

**English** | [日本語](godot.ja.md)

Vokra ships a Godot 4.x **GDExtension** binding in `integrations/vokra-godot`
<!-- anchor: integrations/vokra-godot -->. It is an isolated workspace over the
Vokra C ABI (raw FFI, no binding crate), so it never perturbs the root
`Cargo.lock` zero-dependency invariant (`NFR-DS-02`). It exposes two classes —
`VokraSession` and `VokraStream` — plus two demo projects.

## 1. Build the GDExtension

`scripts/build-godot-gdextension.sh` <!-- anchor: scripts/build-godot-gdextension.sh -->
cross-builds the native library for one of five targets (macOS Intel /
Apple Silicon, Linux x64, Windows MSVC, Android arm64) selected by
`TARGET_TRIPLE`; an unknown triple exits non-zero rather than guessing
(`FR-EX-08`):

```sh
TARGET_TRIPLE=aarch64-apple-darwin scripts/build-godot-gdextension.sh
```

## 2. Install into a Godot project

Copy the `addons/vokra/` tree into your project (this is the Godot AssetLib
layout: the `.gdextension` descriptor plus the per-platform `bin/` libraries).
Godot loads the extension on project open and registers the Vokra classes.

## 3. The `VokraSession` / `VokraStream` API

`VokraSession` loads a GGUF and runs a task; the trampolines unpack Godot
Variants and call the real runtime:

```
var session := VokraSession.new()
session.load_model("res://models/whisper-base.gguf")

# ASR: PackedFloat32Array (16 kHz mono) + sample rate -> String
var text: String = session.transcribe(pcm, 16000)

# TTS: String -> Dictionary { "pcm": PackedFloat32Array, "sample_rate": int }
var out: Dictionary = session.synthesize("Hello from Vokra.")
```

`VokraStream` provides the streaming primitives — `push_pcm(pcm)`, `poll(n)`
and `interrupt()` (barge-in). `session.vad_open_stream(16000)` returns a live
`VokraStream` Object whose lifetime is owned by Godot:

```gdscript
var stream: VokraStream = session.vad_open_stream(16000)
stream.push_pcm(pcm_chunk)
var probabilities: PackedFloat32Array = stream.poll(64)
stream.interrupt()
stream.free()
```

## 4. Demo projects

Two ready-to-open projects live under `demos/`:
`integrations/vokra-godot/demos/asr_demo` <!-- anchor: integrations/vokra-godot/demos/asr_demo -->
loads 16 kHz mono PCM16 and calls `transcribe`; `demos/tts_demo` calls
`synthesize` and streams into an `AudioStreamGenerator`.

## 5. Explicit errors and NVIDIA non-bundling

Every trampoline routes a backend error to an explicit Godot `CallError`
(`FR-EX-08`); the `vokra_last_error()` string is available on the same thread
for GDScript introspection, and a Rust panic is caught at every boundary before
it can reach Godot (`NFR-RL-07`). The packaged addon is scanned to ensure **no
NVIDIA runtime is bundled** by `scripts/compliance/check-godot-package-no-nvidia.sh`
<!-- anchor: scripts/compliance/check-godot-package-no-nvidia.sh --> (a CUDA
build `dlopen`s the system CUDA at run time; it never ships a `libcudart` /
`libcudnn` / `libcublas` / `libnvrtc`).

## 6. Verification state

The trampoline runtime dispatch is code-complete: `transcribe`, `synthesize`,
`vad_open_stream`, `push_pcm`, `poll` and `interrupt` unpack/pack Variants and call the runtime
(`integrations/vokra-godot/src/trampoline.rs`
<!-- anchor: integrations/vokra-godot/src/trampoline.rs -->). The Linux CI leg
downloads the checksum-pinned official Godot 4.7.1 binary and runs both the
asset-free ClassDB/error harness and a real Silero VAD stream smoke. Opening
the ASR/TTS demo scenes interactively in the editor remains a manual release
check; it is not represented as automated evidence.

## 7. Troubleshooting

| symptom | cause / fix |
|---|---|
| The extension does not load | The `.gdextension` `bin/` paths must match your platform/arch; rebuild for the right `TARGET_TRIPLE`. |
| `unknown target triple` from the build script | Pass one of the five supported triples (`FR-EX-08`, no silent guess). |
| `session.transcribe` returns an error | Read `vokra_last_error()`; a backend/op error surfaces as an explicit `CallError`, never a fake result. |
| `vad_open_stream` reports an error | The loaded model must be a streaming VAD model and the requested sample rate must match it; backend details remain in `vokra_last_error()`. |

## Next steps

- [Adding a backend](../backend-guide.md)
- [Desktop CLI](cli.md) — the `convert` step that produces the GGUF you load
- [Unity + IL2CPP](unity.md) — the other game-engine binding

## Keeping this page current

**Last verified: 2026-08-22 — official Godot 4.7.1 headless, real Silero VAD
GGUF + raw-f32 PCM stream.**

- **Update responsibility**: a PR that changes the GDExtension API, the build
  targets, or the compliance scanner updates this page and its Japanese twin in
  the same PR.
- **Review cadence**: quarterly Go/No-go review (`NFR-MT-05`); the interactive
  editor check remains a manual release item, recorded honestly above.
- **Re-verify the dispatch state** (do not trust the README):

```sh
sed -n '1,45p' integrations/vokra-godot/src/trampoline.rs
```
