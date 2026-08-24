# MeloTTS English official acoustic-core fixture

This directory is generated only by
`tools/parity/melotts_dump_reference.py`. The oracle is the official
`myshell-ai/MeloTTS` PyTorch source at commit
`209145371cff8fc3bd60d7be902ea69cbdb7965a` and the official
`myshell-ai/MeloTTS-English` checkpoint at revision
`bb4fb7346d566d277ba8c8c7dbfdf6786139b8ef`.

The fixture intentionally begins after raw-text normalization, G2P and BERT
tokenization. It supplies deterministic position-level BERT features and
records official outputs for `enc_p`, deterministic duration prediction,
prior expansion, inverse flow and the speaker-conditioned HiFi-GAN decoder.
It never imports Vokra and never reads a Vokra GGUF.

Regenerate on a disposable VAST instance through the repository Python 3.12
environment:

```sh
git clone https://github.com/myshell-ai/MeloTTS.git /tmp/melotts-upstream
git -C /tmp/melotts-upstream checkout 209145371cff8fc3bd60d7be902ea69cbdb7965a
uv run --project tools/parity python tools/parity/melotts_dump_reference.py \
  --source-root /tmp/melotts-upstream \
  --output crates/vokra-models/tests/fixtures/melotts_english
```

`manifest.json` pins the source/checkpoint revisions, source artifact hashes,
PyTorch version, geometry, and the SHA-256 of every binary fixture. Floating
files are little-endian FP32; ID files are little-endian U32; durations are
little-endian I32.
