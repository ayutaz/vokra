# VAD -> ASR -> TTS demo (com.vokra.unity Sample)

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
NFR-DS-04). Run the fetcher before opening the scene:

```
bash Samples~/VadAsrTts/scripts/fetch-demo-models.sh
```

This downloads three **MIT-licensed** artifacts into
`Samples~/VadAsrTts/StreamingAssets/models/`:

| File                  | Model              | License | Source                    |
| --------------------- | ------------------ | ------- | ------------------------- |
| `silero-vad-v5.gguf`  | Silero VAD v5      | MIT     | snakers4/silero-vad       |
| `whisper-base.gguf`   | Whisper base       | MIT     | openai/whisper            |
| `voice.gguf`          | piper-plus voice   | MIT     | ayutaz/piper-plus         |

Override any URL via `VOKRA_SILERO_URL`, `VOKRA_WHISPER_URL`,
`VOKRA_PIPER_URL`, or the whole destination via `VOKRA_MODELS_DIR`.

### CC-BY-NC / research-flag exclusion (M2-13 compliance)

The `com.vokra.unity` official package and this sample distribute **only**
MIT-licensed weights. The following are **excluded** from the fetch script:

- **F5-TTS** — CC-BY-NC 4.0 (non-commercial).
- **Fish-Speech v1.4 / v1.5** — CC-BY-NC-SA 4.0 (non-commercial).
- **EnCodec** — CC-BY-NC 4.0.
- **RVC / GPT-SoVITS** — training-data provenance unresolved; scoped to the
  separate `vokra-voiceclone-experimental` repo (ELVIS Act / NO FAKES Act
  isolation, CLAUDE.md design note 8).

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
