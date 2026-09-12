# Whisper extras independent reference

`dump_reference.py` is the only reference generator for the distil-whisper and
kotoba-whisper real-weight parity packet. It imports the pinned
`transformers.WhisperForConditionalGeneration` implementation and never
imports Vokra. The fixed snapshot revisions and input hash are recorded in the
packet `manifest.json`.

Run this only on a disposable VAST worker. The authoritative real-weight
entrypoint requires an explicit, absent work directory:

```bash
bash tools/parity/whisper_extras/run_vast_validation.sh \
  --work-dir /vast/work/whisper-extras-<run-id>
```

It leaves a small `whisper-extras-evidence.tar.gz` containing the Git HEAD,
fixed input SHA, per-variant logs, packet manifests, and packet hashes. The
runner is `NO_UPLOAD` and the worker must be destroyed after retrieval. The
GitHub workflow is model-free and cannot establish real-weight parity.

The VAST worker must convert the Kotoba snapshot with the explicit
`--model kotoba-whisper-v2.2` selector; the generic `kotoba-whisper` alias is
the legacy family route and must not be used for this v2.2 packet.

For the reference dumper itself, use:

```bash
uv sync --frozen --project tools/parity/whisper_extras
uv run --frozen --project tools/parity/whisper_extras python \
  tools/parity/whisper_extras/dump_reference.py \
  --model distil_whisper --checkpoint-dir /vast/hf-snapshot \
  --audio tests/fixtures/audio/jfk-30s.wav --output-dir /vast/reference
```

The output is a verification packet, not a publication artifact. The Kotoba
v2.2 revision is pinned in the manifest. At that exact revision, the official
Kotoba pipeline sets `language="ja"` and `task="transcribe"`; the generation
config supplies the task/no-timestamps ids while the tokenizer supplies the
language id; the dumper resolves the exact token ids from the pinned tokenizer
and generation config. The language choice is not inferred from tensor shape:
it follows the official `kotoba_whisper.py` pipeline at revision
`9d33482a0eb9b57f1ad80708e8ac5538246d8355`, while the numeric ids are checked
against that snapshot's `generation_config.json` and `tokenizer.json`. A GGUF
whose provenance source still
names v2.0 (or lacks the dedicated upstream metadata keys) is rejected
fail-closed by the Rust harness. Select `kotoba-whisper-v2.2` explicitly in
the converter so the artifact carries the authenticated release identity. Do
not run the model generator on the maintainer Mac or upload its output.
