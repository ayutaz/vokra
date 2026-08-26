# NeuTTS Air independent real-checkpoint parity

This directory contains the VAST-only numerical oracle for the exact public
`vokra/neutts-air` GGUF. The oracle loads the exact gated
`neuphonic/neutts-air` snapshot at revision
`3b58b776406b62fdc137e31ea53d728f5c22a4ed` through Hugging Face's official
`Qwen2ForCausalLM`; it does not contain a second Qwen implementation.

Prompt construction executes Neuphonic's released
`NeuTTSAir._apply_chat_template` directly from source commit
`3e9415df12633f8a74ac6f92418c7cd5c8c4bf0e`. The source file is fixed at
9,035 bytes and SHA-256
`e68b87dae6718903337a08eff56afbd58ba261d829624ea5a00a343c8cefb7c1`.
Only the phonemizer call is replaced with identity over already-phonemized test
strings, so this gate isolates the language model without inventing an eSpeak
result. Official tokenizer control IDs and the complete prompt are recorded.

The reference emits the first-position 217,652-way FP32 logit vector and a
short deterministic greedy token sequence. The Rust test compares logits at
the repository FP32 bound `atol=0.01` and greedy IDs exactly. The separately
validated NeuCodec route is composition-smoked when the Distill companion is
provided. No fixture is committed before a real run, and a missing environment
variable is a visible skip rather than a pass.

Run only on a provisioned VAST host after the checkout containing this work is
available there:

```sh
scripts/publish/vast-ai/run-neutts-air-validation.sh
```

The worker has no upload or publish path. It downloads the exact public GGUF,
the exact upstream snapshot and the exact public Distill NeuCodec companion,
then runs the official CPU comparison and workspace/Metal cross-build gates.
Pull only its small evidence/reference directory before destroying the VAST
instance; do not pull model payloads to the maintainer Mac.

After the VAST CPU gate passes, transfer model/reference data directly to a
disposable Apple Silicon host and run:

```sh
VOKRA_REMOTE_APPLE_SILICON=1 \
scripts/verify/apple-silicon-neutts-air.sh \
  --gguf /remote/stage/neutts-air.gguf \
  --companion /remote/stage/distill-neucodec.gguf \
  --reference /remote/stage/reference \
  --evidence-dir /remote/evidence/neutts-air-metal
```

That worker refuses the maintainer-machine class, performs no network or
publication action, and records CPU/Metal logits, greedy IDs and composition
results against the same reference.
