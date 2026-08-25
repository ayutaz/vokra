# AudioSeal official parity fixture

Generate this fixture only on VAST from Meta's official AudioSeal source at
revision `e63a8a0e5cdf7bb797159c92ba15961557fe9bd2` and the four checkpoint
files at immutable Hugging Face revision
`3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba`.

Run `tools/parity/audioseal_dump_reference.py` once with `--variant base` and
the base generator/detector, then once with `--variant streaming` and the
streaming pair. The script imports `AudioSeal.load_generator` and
`AudioSeal.load_detector` from the pinned official checkout, verifies each
checkpoint SHA-256, and emits input PCM, message bits, generator latent and
conditioned tensors, watermark/embedded PCM, raw detector logits, detection
probabilities and recovered bits.

No numerical fixture or tolerance is committed yet. First generate the two
official-reference directories, run the Rust CPU consumer, and record the
worst element and stage. Register a tolerance only from those measurements;
do not widen one in response to a failure. Apple Metal/CPU parity is a
separate real-device pass after the VAST CPU/reference run.

The streaming-trained checkpoint is evaluated over one complete buffer. This
fixture does not claim parity for AudioSeal's state-carrying streaming context.
