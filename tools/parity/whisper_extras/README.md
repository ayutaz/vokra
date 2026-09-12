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

## Apple Silicon handoff

After the VAST runner passes, transfer the two GGUFs, both reference
directories, the two per-variant CPU logs, and `evidence/git-head.txt` to the
disposable Apple checkout. The Apple worker binds every input by SHA-256 and
requires the head file to contain the exact clean VAST `HEAD`:

```bash
bash scripts/verify/apple-silicon-whisper-extras.sh \
  --distil-gguf /transfer/distil_whisper.gguf \
  --distil-gguf-sha256 <VAST-GGUF-SHA256> \
  --distil-reference /transfer/reference/distil_whisper \
  --distil-reference-manifest-sha256 <MANIFEST-SHA256> \
  --distil-reference-packet-sha256 <CANONICAL-PACKET-SHA256> \
  --distil-cpu-log /transfer/evidence/distil_whisper.log \
  --distil-cpu-log-sha256 <CPU-LOG-SHA256> \
  --kotoba-gguf /transfer/kotoba_whisper.gguf \
  --kotoba-gguf-sha256 <VAST-GGUF-SHA256> \
  --kotoba-reference /transfer/reference/kotoba_whisper \
  --kotoba-reference-manifest-sha256 <MANIFEST-SHA256> \
  --kotoba-reference-packet-sha256 <CANONICAL-PACKET-SHA256> \
  --kotoba-cpu-log /transfer/evidence/kotoba_whisper.log \
  --kotoba-cpu-log-sha256 <CPU-LOG-SHA256> \
  --vast-head-file /transfer/evidence/git-head.txt \
  --vast-head-sha256 <HEAD-FILE-SHA256> \
  --expected-head <CLEAN-HEAD-HEX40> \
  --evidence-dir /transfer/apple-evidence
```

`<CANONICAL-PACKET-SHA256>` is the SHA-256 of the newline-separated sorted
records `sha256  filename` for exactly `encoder.f32le`,
`greedy_tokens.u32le`, `input_pcm.f32le`, `logits_last.f32le`, and
`manifest.json`. The worker runs each named Apple test exactly once with
`--ignored --exact`; the tests compare CPU/reference, Metal/reference, and
Metal/CPU, and require exact greedy token equality. A missing Metal device or
unsupported hot op is an error, never a CPU fallback. The worker records
`publication=NO_UPLOAD` and creates no evidence directory until every input
gate passes.
