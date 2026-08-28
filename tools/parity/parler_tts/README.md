# Parler-TTS official-reference parity

This directory is a VAST-only independent oracle for the public
`vokra/parler-tts-mini-v1` and `vokra/parler-tts-mini-multilingual` GGUFs. It
imports `ParlerTTSForConditionalGeneration` from the official
`huggingface/parler-tts` repository at commit
`d108732cd57788ec86bc857d99a6cabd66663d68`, with Transformers 4.46.1. It does
not import Vokra or reproduce the Rust graph in Python.

The no-upload worker is:

```sh
bash scripts/publish/vast-ai/run-parler-tts-validation.sh
```

For each release the oracle runs a fixed, explicit description-token sequence
and a separate fixed prompt-token sequence. It records the official FLAN-T5
hidden states, four-frame greedy delayed code packet, and official embedded-DAC
PCM. The Rust gate compares codes exactly, compares the T5 hidden states under
the recorded FP32 ceiling, then decodes the official packet independently and
applies `max_abs <= 0.01` to PCM. These legs separate text-encoder, LM/schedule,
and codec drift.

The worker has no upload or push option. Pull its `logs/` and `reference/`
directories before destroying the VAST instance. Metal execution is a separate
remote Apple Silicon run using the same GGUF and reference; never execute these
real models on the maintainer Mac.
