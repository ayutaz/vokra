# Vokra Godot demo projects

Godot 4.x demo scaffolds for the `vokra-godot` GDExtension binding
(M3-11-T14 + T15).

## What's here

```
demos/
├── asr_demo/          — Whisper base ASR via VokraSession.transcribe
│   ├── project.godot
│   ├── main.tscn
│   └── main.gd
└── tts_demo/          — piper-plus TTS via VokraSession.synthesize
    ├── project.godot
    ├── main.tscn
    └── main.gd
```

Each project ships as a **scaffold only** — a `project.godot` INI, a
minimal `main.tscn`, and a GDScript entry point that wires the GDExtension
API described in [`../README.md`](../README.md) to the on-screen UI.

## What's NOT here (by design)

- **Native binaries** (`libvokra_godot.so` / `.dylib` / `.dll`) — pulled
  from the CI-produced AssetLib package (M3-11-T12 / T16).
- **Model weights** (`whisper-base.gguf`, `piper-en-amy.gguf`) — fetched
  out-of-band by `fetch-demo-models.sh` (MIT weights only, per
  `docs/adr/0011-godot-gdextension.md` §D5). CC-BY-NC weights are
  never distributed via the demos.
- **Audio fixtures** (`res://audio/jfk.wav`) — copy from the runtime
  parity fixtures under `tests/fixtures/audio/` or supply your own
  16 kHz mono PCM16 WAV.

## Runtime verification

Opening either project in the Godot 4.x Editor and pressing Play is
**owner work** (M3-11-T19). This CC-authored scaffold covers the
files Godot needs to load the project (INI `config_version=5`,
`.tscn` with `format=3`, GDScript `extends Control`); it has not been
opened in the Godot Editor from this session's environment. If the
`.tscn` or `.gd` fails to load in the Editor, capture the error and
file it against M3-11.

## How the AssetLib addon plugs in

After running `scripts/build-godot-gdextension.sh --pack` (or after
the CI matrix job in `.github/workflows/godot-crossbuild.yml`
uploads a per-target artifact), the resulting
`dist/godot/vokra-godot/addons/vokra/` tree is copied verbatim into
either `asr_demo/addons/` or `tts_demo/addons/`. Godot 4.1+ then
auto-discovers `addons/vokra/vokra.gdextension` and dlopens the
platform library from `addons/vokra/bin/<platform>/<arch>/`.

That path is the same one the release-signed AssetLib package (M3-11-T17)
lands at, so a working demo against the local dev build is the direct
smoke test for the release train.
