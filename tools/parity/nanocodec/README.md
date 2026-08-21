# NanoCodec Group-FSQ parity environment

This isolated Python 3.12 uv project exists only to regenerate the committed
Group-FSQ oracle fixture for issue #45. It is not part of the Rust runtime and
does not alter the root `Cargo.lock`.

The lock pins `nemo-toolkit[tts]` directly to NVIDIA-NeMo/Speech commit
`4fcff72febec9395fdbd4bfa0747bfda2ecd3cef`, matching the version recorded by
the released checkpoint. It also pins `peft==0.20.0`, which that NeMo commit's
TTS package imports transitively but does not declare in its `tts` extra. The
dumper additionally checks
that the imported `GroupFiniteScalarQuantizer` source is inside a checkout at
that exact commit and refuses any fallback implementation.

Run from the repository root:

```sh
uv sync --python 3.12 --project tools/parity/nanocodec --frozen
uv run --python 3.12 --project tools/parity/nanocodec \
  python tools/parity/fsq_dump.py nanocodec \
  --checkpoint /path/to/nemo-nano-codec-22khz-0.6kbps-12.5fps.nemo \
  --nemo-source-root /path/to/NVIDIA-NeMo-Speech \
  --out tests/parity/fsq/nanocodec \
  --time 16
```

The checkpoint must be the file at HF repo commit
`5c8e22ed763c14d81337fbe6ca74062f3d10f7e5`, SHA-256
`bd5883099d0c74ceda760b6b7a1600b86da4d8a02531c9c282679951dcb08870`.
The checkpoint remains a temporary local/VAST input and is never committed.
