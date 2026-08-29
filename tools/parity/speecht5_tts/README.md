# SpeechT5 TTS parity closure

The VAST worker uses the pinned `uv.lock` project and the official
`SpeechT5ForTextToSpeech.generate_speech` route from `transformers==5.5.0`.
The lock and project bytes, all canonical dependency rows, fixed TTS and
HiFi-GAN revisions/artifact hashes, and the historical public GGUF identity
are bound by `license_gate_manifest.json`.

`preflight_gate.py` is standard-library-only and runs with
`uv run --no-project --offline` before scratch creation, synchronization,
source/model download, conversion, or Cargo. Production review rows and
operator signer/digest intentionally remain pending/null; it exits 2 until
tracked human review supplies the separate authenticated evidence file. No
license or model-distribution conclusion is implied by this placeholder.

The source dumper imports the exact locked Transformers package and fails
loudly on a version mismatch. CPU and Apple workers require the named Cargo
test to report exactly one passed, zero failed/ignored, and require each
model-level parity sentinel exactly once. No upload path is present.
