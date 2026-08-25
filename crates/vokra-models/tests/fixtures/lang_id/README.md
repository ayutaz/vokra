# SpeechBrain Lang-ID real-reference fixture

This suite covers both complete official releases:

- `speechbrain/lang-id-voxlingua107-ecapa` at
  `0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`;
- `speechbrain/lang-id-commonlanguage_ecapa` at
  `70a742bbc513f693efcf73d6d64a5ed14b3a34a4`.

`tools/parity/speechbrain_lang_id_dump_reference.py` imports the real pinned
`speechbrain==1.0.3` `EncoderClassifier`. It records the official normalized
features, ECAPA embedding, classifier output and ordered label encoder for the
same PCM. It never reads a Vokra GGUF and has no local frontend, ECAPA or
classifier mirror.

The generated variant directory contains:

- `pcm.f32.bin`
- `features.f32.bin`
- `embedding.f32.bin`
- `scores.f32.bin`
- `labels.json`
- `manifest.json`

The binary fixture is not committed yet. Generation and every
`-p vokra-models` consumer run remotely; do not generate or consume it on the
maintainer Mac. The ignored Rust shell test requires both
`VOKRA_LANG_ID_GGUF` and `VOKRA_LANG_ID_REFERENCE_DIR`. Explicitly invoking
the test without either input fails loudly.

No numerical tolerance is registered before the first real measurement. The
test reports max absolute error, worst index/value, mean absolute error,
relative L1 and cosine similarity for frontend, embedding, classifier and
end-to-end scores. It also requires the winning official label to agree. A
numeric parity claim remains forbidden until the VAST CPU and Apple-silicon
Metal measurements have been reviewed and the derived bounds are committed.

The complete generation, conversion and execution recipe is in
`docs/handoff/parity-speechbrain-lang-id-real.md`.
