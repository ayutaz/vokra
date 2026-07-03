# Vokra Unity demo — testing record (M0-10)

WP completion condition (milestones.md M0-10 / SRS §6): *"the 3 demos (VAD / ASR /
TTS) run on macOS / Linux / Windows"*, confirmed on real hardware. Fill one block
per OS. **Do not fabricate results** — leave a block `PENDING` until actually run.

Record per run: date · OS + CPU (Apple Silicon / Intel / x64) · Unity version ·
scripting backend (Mono / IL2CPP) · plugin source (local build / CI artifact) ·
run form (Editor / player / headless) · VAD frames · ASR text · TTS (played / WAV
written) · result (OK/NG + notes). For TTS confirm it used the **real voice GGUF +
real G2P** (not the M0-07-T08 mock) by noting the loaded paths from the log.

---

## macOS (M0-10-T12 — maintainer)

- Status: **PENDING**
- Date:
- OS / CPU:
- Unity version:
- Scripting backend:
- Plugin source:
- Run form: Editor ▢  player ▢
- VAD frames (speech / total):
- ASR text:
- TTS: played ▢  WAV written ▢  · voice GGUF / G2P paths:
- Result: OK ▢  NG ▢  · notes:

## Windows (M0-10-T13 — maintainer)

- Status: **PENDING**
- Date:
- OS / CPU:
- Unity version:
- Scripting backend:
- Plugin source: local `cargo build` ▢  CI Windows artifact ▢
- Run form: Editor ▢  player ▢
- VAD frames (speech / total):
- ASR text:
- TTS: played ▢  WAV written ▢  · voice GGUF / G2P paths:
- Result: OK ▢  NG ▢  · notes:

## Linux (M0-10-T11 — headless, machine-checkable)

- Status: **PENDING** — needs a Unity Linux player build (Linux Build Support +
  Unity license, or a CI Unity job — decision pending, not required for M0). The
  demo code + `build-unity-plugin.sh` (produces `libvokra.so`) + `-batchmode
  -nographics` headless path are ready; this block records the actual run.
- Date:
- Player build source: maintainer Editor + Linux Build Support ▢  CI ▢
- Native plugin (`libvokra.so`): local build ▢  M0-01 CI ubuntu artifact ▢
- Command: `./VokraDemo -batchmode -nographics -vokraModelsDir <dir> -vokraInput <wav> -vokraOutput <wav>`
- VAD frames (speech / total):
- ASR text:
- TTS: WAV written ▢  · voice GGUF / G2P paths (confirm real, not mock):
- Exit code:
- Result: OK ▢  NG ▢  · notes:

---

## Sign-off (M0-10-T14)

- [ ] macOS record complete (VAD→ASR→TTS OK)
- [ ] Windows record complete (VAD→ASR→TTS OK)
- [ ] Linux record complete (headless VAD→ASR→TTS OK)
- [ ] `Packages/manifest.json` has no Sentis / Inference Engine (SRS §5-(10))
- [ ] callbacks are `[MonoPInvokeCallback]` + static + GCHandle userdata (NFR-RL-02)
- [ ] no iOS/Android code (v0.5 = FR-API-04)
- [ ] PR CI green (build/test/fmt/clippy/parity/license)
