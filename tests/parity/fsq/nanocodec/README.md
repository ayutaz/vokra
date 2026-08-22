# NanoCodec grouped-FSQ real-checkpoint fixture

Reference oracle: NVIDIA-NeMo/Speech
`nemo.collections.tts.modules.audio_codec_modules.GroupFiniteScalarQuantizer`
at commit `4fcff72febec9395fdbd4bfa0747bfda2ecd3cef`, matching the NeMo version
recorded by the released checkpoint and called through the
quantizer restored from NVIDIA's real
`nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps` checkpoint.

This is a real checkpoint-loaded quantizer fixture with deterministic synthetic
indices. Group-FSQ has no learned codebook tensor: the checkpoint supplies the
actual runtime group count and `num_levels` buffers, while fixed indices make
the small committed oracle reproducible and cover both codebook boundaries.

Committed generated files:

- `levels_per_group.u32`: `[9, 8, 8, 7]`, the checkpoint's four scalar axes.
- `codes_time_group.u32`: `[time=16, groups=4]`, row-major.
- `decoded_features.f32`: `[time=16, feature=16]`, NeMo output transposed from
  `[batch, feature, time]`.
- `manifest.txt`: source/checkpoint/version pins plus SHA-256 for every binary.

The Rust parity test uses `atol = 1e-6`; do not widen it in response to a
failure. Inspect the worst group/dimension and the recorded oracle environment
first. Regenerate with the locked command in
`tools/parity/nanocodec/README.md`. The `.nemo` checkpoint is not committed.
