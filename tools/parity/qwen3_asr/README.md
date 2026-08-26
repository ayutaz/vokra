# Qwen3-ASR independent real-checkpoint parity

This directory stages the VAST-only oracle for the exact released
`Qwen/Qwen3-ASR-0.6B` and `Qwen/Qwen3-ASR-1.7B` revisions accepted by Vokra.
It imports the official Apache-2.0 `qwen-asr==0.0.6` package; there is no
second, locally reimplemented Qwen model in the dumper.

The official calls used as reference are:

- `Qwen3ASRModel._build_text_prompt` and its official processor for prompt and
  16 kHz log-mel preparation;
- `model.thinker.get_audio_features` for the final projected audio rows;
- `model.generate` for the greedy token sequence;
- the official tokenizer and `parse_asr_output` for raw/final text.

The environment is locked by this directory's `uv.lock`. The model snapshot is
downloaded by exact 40-hex revision, then all source files and generated
artifacts are hashed into the output. Missing imports, revision/shape drift,
sidecar drift, a non-local snapshot, non-FP32 execution, or a non-CPU official
reference aborts loudly.

No numerical fixture is committed before an actual run. The Rust consumer in
`crates/vokra-models/tests/qwen3_asr_real.rs` is environment-gated and uses the
repository FP32 bound `atol=0.01` for projected audio. Greedy token ids,
language, and text must match exactly. Metal uses the same CPU model run as its
oracle and also requires exact greedy ids; it never falls back to CPU.

Run only through the VAST worker after provisioning:

```sh
scripts/publish/vast-ai/run-qwen3-asr-validation.sh --variant 0.6b
scripts/publish/vast-ai/run-qwen3-asr-validation.sh --variant 1.7b
```

The worker uses the committed two-second mono 16 kHz JFK-derived clip at
`tests/parity/utmos/ref-clip.wav`, performs no upload, and leaves only the small
reference/evidence directory to pull before the VAST instance is destroyed.
Do not pull the source snapshot or GGUF back to the maintainer Mac.
