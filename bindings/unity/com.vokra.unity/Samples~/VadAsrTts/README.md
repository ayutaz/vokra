# VAD -> ASR -> TTS demo (com.vokra.unity Sample)

> **2026-10-23 current-status pointer:** Use [`docs/current-status.md`](../../../../../docs/current-status.md)
> or the [Japanese counterpart](../../../../../docs/current-status.ja.md) for
> current release and project status. The latest stable source release is
> `v0.3.0`; the published desktop preview is `v0.4.0-rc.1` with 22 assets.
> This sample documentation refresh adds no runtime or model-execution claim.

Minimal end-to-end pipeline showing the three demonstrated verticals of the
v0.1 MVP Vokra C ABI:

1. **VAD** — Silero VAD v5 speech-probability per 512-sample frame.
2. **ASR** — Whisper base transcription of the whole 16 kHz mono input.
3. **TTS** — piper-plus MB-iSTFT-VITS2 synthesis of the (possibly fallback)
   text into mono float32 PCM.

The whole C ABI call sequence runs on a single background thread; results are
handed to the Unity main thread via a `ConcurrentQueue`, and IMGUI (OnGUI) is
used so the scene has no Canvas / EventSystem / Font asset dependencies
(preserved as-is from the M0-10 unity-demo migration).

## Import

From the Package Manager, select **Vokra** → **Samples** → **VAD -> ASR -> TTS
demo** → **Import**. Unity copies the sample under
`Assets/Samples/Vokra/<version>/VAD -> ASR -> TTS demo/`.

## Models

**Models are never committed to git** (`*.gguf` is `.gitignore`d — see
NFR-DS-04). The `v0.3.0` GitHub Release contains no GGUF model assets, so
release presence alone does not make the sample models fetchable. Before
running the fetcher, operators must provide all three independently verified
MIT source URLs:

```
export VOKRA_SILERO_URL=<verified-silero-gguf-url>
export VOKRA_WHISPER_URL=<verified-whisper-gguf-url>
export VOKRA_PIPER_URL=<verified-piper-gguf-url>
bash Samples~/VadAsrTts/scripts/fetch-demo-models.sh
```

The fetcher defaults to GitHub `releases/latest/download/…` asset names for
convenience, but the currently published `v0.3.0` release contains no GGUF
model assets, and a desktop preview is not a verified model source. Set all
three URL overrides above to independently verified MIT sources before
fetching; otherwise the default endpoints may return missing assets.

Verify license and provenance against
[`docs/license-audit.md`](../../../../../docs/license-audit.md) before fetching.

The fetcher expects three artifacts in
`Samples~/VadAsrTts/StreamingAssets/models/`. The URL overrides are
user-supplied: verify each artifact's license, provenance, and checksum
against [`docs/license-audit.md`](../../../../../docs/license-audit.md) before
use. The fetcher does not establish those claims:

| File                  | Model              | Expected license | Expected source          |
| --------------------- | ------------------ | ---------------- | ------------------------ |
| `silero-vad-v5.gguf`  | Silero VAD v5      | MIT              | snakers4/silero-vad      |
| `whisper-base.gguf`    | Whisper base       | MIT              | openai/whisper           |
| `voice.gguf`           | piper-plus voice   | MIT              | ayutaz/piper-plus        |

Override each source with `VOKRA_SILERO_URL`, `VOKRA_WHISPER_URL`, and
`VOKRA_PIPER_URL`, or the whole destination with `VOKRA_MODELS_DIR`.

### CC-BY-NC / research-flag exclusion (M2-13 compliance)

The `com.vokra.unity` official package and this sample's fetch script are
limited to **MIT-licensed** weights. The following are **excluded** from the
fetch script:

- **F5-TTS** — CC-BY-NC 4.0 (non-commercial).
- **Fish-Speech v1.4 / v1.5** — CC-BY-NC-SA 4.0 (non-commercial).
- **EnCodec** — CC-BY-NC 4.0.
- **RVC / GPT-SoVITS** — training-data provenance unresolved; scoped to the
  separate `vokra-voiceclone-experimental` repo (ELVIS Act / NO FAKES Act
  isolation; see [`docs/legal-compliance.md`](../../../../../docs/legal-compliance.md)).

These are surfaced through the `vokra-cli` research flag against a distinct
model index and never through the official Vokra distribution channel.

## StreamingAssets ↔ persistentDataPath (Android)

`PipelineRunner` routes every model path through
`Vokra.VokraAndroidAssets.EnsureLocalCopy(...)` before calling
`VokraSession.CreateFromFile(...)`, so that on Android the file is copied out
of the APK jar (`jar:file://…/base.apk!/assets/models/<file>`) into
`Application.persistentDataPath/vokra/models/<file>` on first access. The
copy is idempotent (skipped when the destination file already exists at the
matching byte count) and satisfies **NFR-RL-04**. On desktop platforms
(macOS/Linux/Windows) and iOS, the helper is a no-op — the returned path is
just `Application.streamingAssetsPath/models/<file>`.

The runner still accepts `-vokraModelsDir <abs>` for headless / scripted
runs; when that override is supplied, the helper is bypassed and the given
absolute path is used verbatim.

On WebGL, Unity uses the main-thread async byte-fetch path documented in
`PipelineRunner.RunWebGlAsync`: model and input bytes are fetched through
`ReadBytesAsync`, VAD yields between chunks, and the one-shot ASR/TTS calls may
block the browser tab for their duration. Other platforms use the single
background-thread path described above.

## Command-line flags

The scene contains a single `DemoUi` MonoBehaviour that reads:

| Flag                | Default                                                   |
| ------------------- | --------------------------------------------------------- |
| `-vokraInput <wav>` | `<StreamingAssets>/test_16k.wav`                          |
| `-vokraModelsDir`   | `<StreamingAssets>/models` (via `EnsureLocalCopy`)        |
| `-vokraOutput`      | (unset — no WAV written)                                  |
| `-vokraText`        | `Hello from Vokra.` (fallback when ASR emits no real text)|

In `-batchmode`, the pipeline runs once, every event is logged, the TTS WAV
is written (if `-vokraOutput` is set), and `Application.Quit(0|1)` returns
the aggregate error state.
