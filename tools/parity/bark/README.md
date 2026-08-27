# Bark official-reference parity

This directory is a VAST-only independent oracle for the public `vokra/bark`
and `vokra/bark-small` GGUFs. It imports the released `BarkModel` directly from
locked Transformers 5.5.0 and loads the exact immutable Suno checkpoint. It
does not import Vokra or reproduce Vokra's Rust graph in Python.

The no-upload worker is:

```sh
bash scripts/publish/vast-ai/run-bark-validation.sh
```

It generates four greedy semantic tokens from fixed caller-visible text token
IDs, runs the official coarse/fine schedule, and records the final frame-major
eight-codebook packet plus official 24 kHz PCM. The Rust gate first compares
generated codes exactly, then decodes the official packet independently and
applies the standard FP32 `max_abs <= 0.01` ceiling to PCM. This separation
distinguishes LM/schedule drift from codec drift.

The worker has no upload or push option. Pull its `logs/` and `reference/`
directories before destroying the VAST instance. Metal execution is a separate
remote Apple Silicon run using the same GGUF and reference; do not execute real
Bark models on the maintainer Mac.
