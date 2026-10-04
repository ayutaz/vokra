# Kyutai STT independent streaming reference

This directory contains a fail-closed observer for the official Kyutai Moshi
`LMGen` main-transformer streaming KV state.  It does not reimplement the
ring cache and does not claim PCM, transcription, Apple CPU/Metal, or Rust
consumer parity.  `CONTEXT=375` is the authenticated official configuration;
frames 376 and 377 are retained to observe the official eviction boundary.

There are two deliberately separate reference boundaries here.  The existing
`dump.py` starts at an authenticated Mimi-code packet and observes the
official LMGen/RingKV state; it is the decoder-code-boundary producer and does
not prove raw-PCM or full-Mimi behavior.  The separate `pcm_dump.py` is a
source candidate for the official raw-PCM -> Mimi -> LMGen schema.  That PCM
candidate has an empty reviewed dependency closure, remains `NO_UPLOAD`, is
`NOT_NUMERICAL_PASS`, and does not yet constitute a native-consumer result;
the native PCM consumer is still work in progress.

The source packet is source-only evidence.  It authenticates the Moshi commit,
complete tree, Git blob headers, and the two role files, but it is never
imported or treated as an execution approval.  The real path additionally
requires an external six-field `KYUTAI_STT_PYTORCH_PCM_ORACLE_CAPTURE`
approval, its exact SHA, a mandatory authenticated source packet, clean pinned
DSM/Moshi checkouts, all four exact model inputs, and a VAST-only execution
boundary.  The dependency-closure SHA is recorded in the manifest and checked
against the readiness gate.  The reviewed-closure allowlist is currently
empty, so real capture is deliberately blocked before model access; no fake
`APPROVED` receipt or future approval template is shipped here.

The separate `require_pcm_source_packet` contract authenticates a source-only
packet for the four additional PCM roles: DSM
[the pinned official evaluator](https://github.com/kyutai-labs/delayed-streams-modeling/blob/4c4f65e147df056adf3346290d64c7b9649b18c9/scripts/stt_evaluate_on_dataset.py)
and
[the pinned official transport client](https://github.com/kyutai-labs/delayed-streams-modeling/blob/4c4f65e147df056adf3346290d64c7b9649b18c9/scripts/stt_from_file_rust_server.py),
plus Moshi
[the pinned compression source](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi/moshi/models/compression.py)
and
[the pinned streaming source](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi/moshi/modules/streaming.py).
These are the exact pinned source-packet role paths; they are not files that
this checkout is allowed to synthesize.
It is preparation evidence only and does not authorize a PCM/full-Mimi
capture.  The PCM source candidate now exists, but no real PCM capture has
run; the native PCM tests below are structural borrowed-observer tests, not
real-weight or numerical proof.

The two schedules must not be silently conflated.  The official evaluator
boundary represented by `pcm_dump.py` uses one second of left silence, three
seconds of right silence (`delay + 0.5`), ceiling to a complete frame, and one
LM call per encoded frame.  The existing native PCM session uses 3.5 seconds
of right silence (`delay + 1.0`), drains complete hops with a final partial
hop discarded, and makes the first frame's LM call twice before continuing
with one call per frame.  This is an explicit schema/scope difference to be
reported by a future consumer, not a license to shift, drop, or fit calls.

The real CLI, when the separately reviewed closure is non-empty, is:

```bash
uv run --no-project --no-sync --python /path/to/reviewed-kyutai-env/bin/python \
  python tools/parity/kyutai_stt_streaming_reference/dump.py real \
  --model /path/to/exact-four-file-input-directory \
  --dsm-source /path/to/clean-pinned-dsm-checkout \
  --moshi-source /path/to/clean-pinned-moshi-checkout \
  --source-packet /path/to/authenticated-source-packet \
  --out /path/to/new-output-directory \
  --expected-head <40-hex-checkout-head> \
  --approval-evidence /path/to/external-approval.json \
  --approval-sha256 <64-hex-approval-sha> \
  --dependency-closure-sha256 <64-hex-reviewed-closure-sha>
```

The raw-PCM candidate has a separate, blocked invocation shape.  It is shown
for schema review only; it must not be run until the composite approval and a
non-empty reviewed dependency closure exist:

```bash
uv run --no-project --no-sync --python /path/to/reviewed-kyutai-env/bin/python \
  python tools/parity/kyutai_stt_streaming_reference/pcm_dump.py real \
  --model /path/to/exact-four-file-input-directory \
  --dsm-source /path/to/clean-pinned-dsm-checkout \
  --moshi-source /path/to/clean-pinned-moshi-checkout \
  --source-packet /path/to/authenticated-source-packet \
  --pcm-input /path/to/mono-24k-f32le.pcm \
  --pcm-input-sha256 <64-hex-pcm-sha> \
  --out /path/to/new-output-directory \
  --expected-head <40-hex-checkout-head> \
  --approval-evidence /path/to/external-approval.json \
  --approval-sha256 <64-hex-approval-sha> \
  --dependency-closure-sha256 <64-hex-reviewed-closure-sha>
```

`/path/to/reviewed-kyutai-env/bin/python` is a placeholder for a separately
reviewed, isolated Python 3.12 environment; no such approved environment or
non-empty closure exists as of this snapshot.  Do not replace it with the
shared `tools/parity` project or invoke this command before the readiness gate
is approved.

Readiness is fail-closed before upstream imports or model access: Linux
`x86_64`, bounded unique-key approval JSON, exact schema/scope/decision/
execution/head/checkout, mandatory source packet, clean source identities,
and a reviewed dependency closure are all required.  `self-test` and the
stdlib-only tests do not satisfy any of these gates.

The output is designed to record raw CPU storage before any dtype conversion.
BF16 bytes
are not converted through NumPy.  It captures the official 48-layer stream at
the 375-frame eviction boundary: warm-up steps 0, 1, 374, 375, 376 and the
first post-reset step, plus bounded per-step deltas.  The resulting raw KV
the source-derived expected KV budget is 1,033,371,648 bytes (about 985.5 MiB,
0.962 GiB, or 1.03 GB decimal); this is not a measurement from an executed
packet.  Any resulting artifact must remain remote.  Ring `positions` are accepted only from the official `complete()`
return; reset has no positions API, so reset snapshots record only official
`end_offset`/capacity and explicitly do not compare stale physical backing
slots.  Initial and post-reset `end_offset` must be zero.  KV remains raw
BF16; exported checkpoint logits are contiguous float32 with their original
`source_dtype` and conversion recorded in the manifest.

The Rust consumer reads these artifacts through the following required
environment variables: `VOKRA_KYUTAI_STT_STREAMING_GGUF`,
`VOKRA_KYUTAI_STT_STREAMING_GGUF_SHA256`,
`VOKRA_KYUTAI_STT_STREAMING_REFERENCE`,
`VOKRA_KYUTAI_STT_STREAMING_REFERENCE_MANIFEST_SHA256`,
`VOKRA_KYUTAI_STT_STREAMING_APPROVAL`,
`VOKRA_KYUTAI_STT_STREAMING_APPROVAL_SHA256`, and
`VOKRA_KYUTAI_STT_STREAMING_DEPENDENCY_CLOSURE_SHA256`.  It authenticates
paths, hashes, artifacts, shapes, positions, and text evidence before opening
the GGUF.  The numerical bound is intentionally `FIXED_ATOL=None`; the
consumer reports bounded BF16/f32 max, mean, and worst-coordinate diagnostics,
then explicitly fails with `NOT_PARITY_PASS`.  This is not PCM, ASR,
transcription, full-model, Apple CPU/Metal, or publication parity.  Independent
full PCM/ASR execution and approval remain outstanding.

Local verification is limited to stdlib-only tests:

```bash
UV_CACHE_DIR=/private/tmp/vokra-kyutai-uv-cache \
  uv run --no-sync --no-project --python 3.12 python -S \
  -m unittest discover -s tools/parity/kyutai_stt_streaming_reference -p 'test_*.py'
```

The dated
[`2026-10-04 Kyutai handoff`](../../../docs/handoff/kyutai-independent-reference-2026-10-04.md)
records the test inventory and receipt for each candidate snapshot.  This
README intentionally does not freeze a test count while the PCM producer and
native consumer are under review.  The tests use only synthetic
parser/tensor contracts and mock receipts; they are not upstream execution,
real-weight parity, license approval, or publication evidence.

No model download, upstream import, package installation, project sync, heavy
Cargo, SSH, or provider operation is part of this local path.  Real execution
is approved VAST-only and remains `NO_UPLOAD`.
