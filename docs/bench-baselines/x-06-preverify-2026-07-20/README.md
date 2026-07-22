# X-06 pre-verify evidence — 2026-07-20

Evidence that the never-run nightly workflows will be green on their first
cron fire. **This is pre-verify evidence, NOT a CI-gate baseline file** — do
not wire any gate to these files (see `docs/bench-baselines/README.md`).

## T05 — nightly-asr-wer scorer path (LibriSpeech ASR-WER), green

Ran the full nightly-asr-wer scorer path locally before the first cron:
decode → `vokra-cli run` (release) → `tools/eval/librispeech_wer.py`, over the
8 pinned campaign utterances of dev-clean 1272/128104.

- **Result: PASS — WER 4.3689% <= 6.0000% threshold, delta vs baseline +0.0000%.**
- The WER **reproduces the committed baseline exactly** (`asr-wer-summary.m1.md`).
  Whisper greedy decode is deterministic and flac→WAV was verified bit-identical
  by the workflow, so this WER is **hardware-independent** (an M1 measurement
  equals the ubuntu campaign measurement). Unlike an RTF baseline, a WER is not
  rig-scoped — but this file is still evidence, not a gate.
- Assets used (local, not committed): `~/.cache/vokra-eval/gguf/whisper-base.gguf`,
  `~/.cache/vokra-eval/weights/LibriSpeech/dev-clean/1272/128104/`,
  `~/.cache/vokra-eval/weights/whisper-base/normalizer.json`.
- Conclusion: the scorer path is correct end-to-end; the first `nightly-asr-wer`
  cron (owner dispatch) is expected green.

## T06 — nightly-webgl wasm-harness leg, NOT run locally (honest)

`scripts/build-unity-webgl-lib.sh --verify` needs a pinned emsdk
(`VOKRA_WEBGL_EMSDK_VERSION=3.1.38`). **No emsdk / emcc is installed on this
machine**, so the wasm-harness leg was NOT run locally. What was verified
instead (X-06-T02 + T06):

- The workflow surface oracle `tools/parity/test_nightly_webgl_workflow.py` is
  green (dual-leg structure, license-gated skip ≠ fabricated pass, cron slot,
  advisory posture).
- The build script `scripts/build-unity-webgl-lib.sh` carries the internal
  root-Cargo.lock tripwire the workflow header delegates to (asserted by the
  oracle).

The first real wasm-harness run needs emsdk (CI installs the pin) and is the
owner's `workflow_dispatch` — CC could not run it here. This is an honest gap,
not a skipped verification.
