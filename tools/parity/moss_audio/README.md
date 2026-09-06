# MOSS-Audio independent real-checkpoint parity

This directory stages the VAST-only oracle for the exact released
`OpenMOSS-Team/MOSS-Audio-4B-Instruct` and
`OpenMOSS-Team/MOSS-Audio-8B-Instruct` revisions accepted by Vokra.
It imports `MossAudioModel` and `MossAudioProcessor` directly from official
OpenMOSS commit `5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883`; there is no locally
reimplemented MOSS/Qwen model in the dumper.

The official calls used as reference are:

- `MossAudioProcessor.from_pretrained` and `processor(...)` for the 16 kHz
  log-mel frontend, one-audio ChatML prompt and two-second time markers;
- `model.get_audio_features`, `model.audio_adapter` and the three official
  `deepstack_audio_merger_list` modules for all four projected audio taps;
- `model.generate(do_sample=False)` for the greedy token sequence;
- `processor.decode(..., skip_special_tokens=True)` for generated text.

The environment is locked by this directory's `uv.lock`. The model snapshot
and official source checkout are downloaded at exact 40-hex revisions. Config,
sidecars, source files and generated evidence are all hashed. A missing import,
source path outside the official checkout, revision/shape drift, non-FP32 CPU
reference, or modified sidecar aborts loudly.

The VAST worker runs the dependency-free `preflight_gate.py` against the exact
project/lock bytes before checking host capacity, checkout cleanliness, tokens,
scratch paths, synchronization, or downloads. The checked-in manifest is
intentionally pending review and therefore exits 2; no model or source
acquisition is reachable until dependency, source-license, model-license, and
exact checkpoint-file evidence is authenticated by a later owner review.
The current official-source route is pinned to Transformers 5.5.0, below the
repository's patched 5.10.x floor. Because compatibility with a patched
release has not been proven without loading the real model, production
preflight fails closed with `BLOCKED_UNVERIFIED_API_SMOKE`; an owner-approved
model-free source inspection/API smoke must re-authenticate the patched lock
before any VAST model acquisition is scheduled. The VAST worker now requires
the smoke evidence path and its SHA-256, and validates that evidence against
the exact requested variants, source/metadata identities, approval scope,
Transformers 5.10.4, and `checkpoint_load=NOT_PERFORMED`; a missing or stale
packet remains blocked before dependency sync or model acquisition. This bridge
does not change the active 5.5.0 lock or declare the main route compatible.

No numerical fixture is committed before an actual run. The Rust consumer in
`crates/vokra-models/tests/moss_audio_real.rs` is environment-gated and uses
the repository FP32 ceiling `atol=0.01` for primary and all three DeepStack
audio projections. Prompt and greedy token ids and decoded text must match
exactly. Metal uses the independently validated CPU implementation as oracle,
keeps the same `atol=0.01`, and also requires exact greedy ids.

Run only through the VAST worker after provisioning:

```sh
scripts/publish/vast-ai/run-moss-audio-validation.sh --variant 4b \
  --approval-evidence /path/to/approval.json \
  --api-smoke-evidence /path/to/api-smoke-evidence.json \
  --api-smoke-sha256 <64-hex> --expected-head <40-hex>
scripts/publish/vast-ai/run-moss-audio-validation.sh --variant 8b \
  --approval-evidence /path/to/approval.json \
  --api-smoke-evidence /path/to/api-smoke-evidence.json \
  --api-smoke-sha256 <64-hex> --expected-head <40-hex>
```

The worker uses the committed two-second mono 16 kHz clip at
`tests/parity/utmos/ref-clip.wav`, performs no upload, and leaves only the
small logs/summaries to recover locally. Transfer the large GGUF and complete
reference packets directly from VAST to the disposable Apple host; do not pull
the source snapshot, merged safetensors, GGUF, or reference packet to the
maintainer Mac. Destroy the VAST instance after the small logs are recovered.

After both CPU runs pass, transfer the GGUF/reference pairs directly from
VAST to a disposable Apple Silicon host with at least 64 GB RAM:

```sh
VOKRA_REMOTE_APPLE_SILICON=1 \
scripts/verify/apple-silicon-moss-audio.sh \
  --gguf-4b /remote/stage/moss-audio-4b-instruct.gguf \
  --gguf-4b-sha256 <VAST_GGUF_4B_SHA256> \
  --reference-4b /remote/stage/reference-4b \
  --reference-4b-sha256 <VAST_REFERENCE_4B_PACKET_SHA256> \
  --gguf-8b /remote/stage/moss-audio-8b-instruct.gguf \
  --gguf-8b-sha256 <VAST_GGUF_8B_SHA256> \
  --reference-8b /remote/stage/reference-8b \
  --reference-8b-sha256 <VAST_REFERENCE_8B_PACKET_SHA256> \
  --approval-evidence /remote/stage/approval.json \
  --expected-head <VAST_EXPECTED_HEAD> \
  --evidence-dir /remote/evidence/moss-audio-metal
```

That worker has no network, conversion, publication or deletion path. Pull
only its evidence, then remove staged model data or destroy the remote host.
