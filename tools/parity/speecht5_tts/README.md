# SpeechT5 TTS parity closure

The VAST worker uses the pinned `uv.lock` project and the official
`SpeechT5ForTextToSpeech.generate_speech` route from `transformers==5.5.0`.
The lock and project bytes, all canonical dependency rows, fixed TTS and
HiFi-GAN revisions/artifact hashes, and the historical public GGUF identity
are bound by `license_gate_manifest.json`.

`preflight_gate.py` is standard-library-only and runs with
`uv run --no-project --offline` before scratch creation, synchronization,
source/model download, conversion, or Cargo. It binds the exact Linux lock,
including uv's build-constraint manifest for the isolated NumPy source build.
The compact `dependency_audit_evidence.json` records the full fresh VAST audit
digest without committing the 1.3 MB report. `patchelf` is GPL build-only and
is not installed in or redistributed with the final environment; its operator
approval remains an explicit gate.

`post_sync_audit.py` runs immediately after `uv sync` and before any source or
model acquisition. It independently checks the synchronized package closure,
absence of build-only dependencies, source-built NumPy native libraries and
ELF NEEDED allowlists, and the exact reviewed torch `libgomp` identity. The
five model/source factual review rows are complete; acquisition and validation
remain fail-closed solely until operator approval supplies the separate
authenticated evidence file.

The source dumper imports the exact locked Transformers package and fails
loudly on a version mismatch. CPU and Apple workers require the named Cargo
test to report exactly one passed, zero failed/ignored, and require each
model-level parity sentinel exactly once. No upload path is present.
