# Assets/StreamingAssets/models — demo GGUF models

The pipeline loads three GGUF models from this folder (via the Vokra C ABI). They
are **not committed** (placed by `scripts/fetch-demo-models.sh`):

| File | Task | Source | License (code / weights) |
|------|------|--------|--------------------------|
| `silero-vad-v5.gguf` | VAD (Silero VAD v5, M0-05) | committed fixture `tests/parity/silero_vad/silero-vad-v5.gguf` (~2 MB) | MIT / MIT |
| `whisper-base.gguf` | ASR (Whisper base, M0-06) | convert a Whisper base checkpoint with `vokra-convert` (M0-03) | MIT / MIT |
| `voice.gguf` | TTS (piper-plus native, M0-07) | convert a piper-plus voice with `vokra-convert` (`vokra.*` chunks; **no onnxruntime** — native path) | MIT / MIT |

Only the Silero VAD model is small enough to ship from the committed fixture, so
**VAD works out of the box**. ASR and TTS need the larger checkpoints, which you
convert/place yourself; until then the demo cleanly reports those stages as
"skipped" (matching the M0-09 C-smoke env-gating).

Desktop (macOS/Linux/Windows) reads `StreamingAssets` as real file paths, so the
C ABI receives a plain path. (Android's `StreamingAssets` jar-URL problem —
NFR-RL-04 — and the `persistentDataPath` extraction helper are v0.5 scope,
FR-API-04.)

## G2P (piper-plus, JA/EN)

The TTS G2P is reused from piper-plus (M0-07). Its artifact form (library vs.
dictionary data) is defined by the M0-07-T04 record in
`docs/piper-plus-integration.md` — follow that; place any files under
`Assets/StreamingAssets/piper-plus-g2p/` as instructed there. This is separate
from the `vokra` cdylib placed by `build-unity-plugin.sh`.

See `../../README.md` for the full run procedure and license attribution
(cross-check `docs/license-audit.md` and `NOTICE`).
