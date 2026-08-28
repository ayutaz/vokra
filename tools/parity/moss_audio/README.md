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

No numerical fixture is committed before an actual run. The Rust consumer in
`crates/vokra-models/tests/moss_audio_real.rs` is environment-gated and uses
the repository FP32 ceiling `atol=0.01` for primary and all three DeepStack
audio projections. Prompt and greedy token ids and decoded text must match
exactly. Metal uses the independently validated CPU implementation as oracle,
keeps the same `atol=0.01`, and also requires exact greedy ids.

Run only through the VAST worker after provisioning:

```sh
scripts/publish/vast-ai/run-moss-audio-validation.sh --variant 4b
scripts/publish/vast-ai/run-moss-audio-validation.sh --variant 8b
```

The worker uses the committed two-second mono 16 kHz clip at
`tests/parity/utmos/ref-clip.wav`, performs no upload, and leaves only the
small reference/evidence directory to pull. Do not pull the source snapshot,
merged safetensors, or GGUF to the maintainer Mac. Destroy the VAST instance
after evidence is recovered.

After both CPU runs pass, transfer the GGUF/reference pairs directly from
VAST to a disposable Apple Silicon host with at least 64 GB RAM:

```sh
VOKRA_REMOTE_APPLE_SILICON=1 \
scripts/verify/apple-silicon-moss-audio.sh \
  --gguf-4b /remote/stage/moss-audio-4b-instruct.gguf \
  --reference-4b /remote/stage/reference-4b \
  --gguf-8b /remote/stage/moss-audio-8b-instruct.gguf \
  --reference-8b /remote/stage/reference-8b \
  --evidence-dir /remote/evidence/moss-audio-metal
```

That worker has no network, conversion, publication or deletion path. Pull
only its evidence, then remove staged model data or destroy the remote host.
