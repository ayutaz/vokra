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
and `interrupt()` (barge-in). Note the honest gap: opening a VAD stream from
GDScript (`session.vad_open_stream`, which returns an Object) is **not yet
wired** and reports an explicit error rather than a fake value — that return
path needs more Variant plumbing.

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

## 6. Honest state (owner editor verification)

The trampoline runtime dispatch is code-complete: `transcribe`, `synthesize`,
`push_pcm`, `poll` and `interrupt` unpack/pack Variants and call the runtime
(`integrations/vokra-godot/src/trampoline.rs`
<!-- anchor: integrations/vokra-godot/src/trampoline.rs -->). What is **not**
done here is running it inside a real Godot 4.x editor and driving the demos
end-to-end — that runtime verification is an owner task (M3-11 T19). Treat this
page as the API contract, not a certification that it has been clicked through
in the editor.

## 7. Troubleshooting

| symptom | cause / fix |
|---|---|
| The extension does not load | The `.gdextension` `bin/` paths must match your platform/arch; rebuild for the right `TARGET_TRIPLE`. |
| `unknown target triple` from the build script | Pass one of the five supported triples (`FR-EX-08`, no silent guess). |
| `session.transcribe` returns an error | Read `vokra_last_error()`; a backend/op error surfaces as an explicit `CallError`, never a fake result. |
| `vad_open_stream` reports an error | Expected — the Object return path is not yet wired (§3). |

## Next steps

- [Adding a backend](../backend-guide.md)
- [Desktop CLI](cli.md) — the `convert` step that produces the GGUF you load
- [Unity + IL2CPP](unity.md) — the other game-engine binding

## Keeping this page current

**Last verified: 2026-07-21 — against `integrations/vokra-godot/src/` (the
trampolines dispatch; the README dated 2026-07-10 pre-dates that and is
stale).**

- **Update responsibility**: a PR that changes the GDExtension API, the build
  targets, or the compliance scanner updates this page and its Japanese twin in
  the same PR.
- **Review cadence**: quarterly Go/No-go review (`NFR-MT-05`); the real-editor
  runtime verification is the owner's (M3-11 T19), recorded honestly above.
- **Re-verify the dispatch state** (do not trust the README):

```sh
sed -n '1,45p' integrations/vokra-godot/src/trampoline.rs
```
