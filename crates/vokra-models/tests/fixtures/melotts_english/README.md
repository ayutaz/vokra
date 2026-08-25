# MeloTTS five-release official acoustic-core fixtures

This directory is generated only by
`tools/parity/melotts_dump_reference.py`. The oracle is the official
`myshell-ai/MeloTTS` PyTorch source at commit
`209145371cff8fc3bd60d7be902ea69cbdb7965a` and the official
one of the five official language checkpoints. Each sibling directory's
`manifest.json` pins its exact upstream repository, revision, checkpoint hash,
speaker, geometry, PyTorch version, and generated-file hashes.

The fixture intentionally begins after raw-text normalization, G2P and BERT
tokenization. It supplies deterministic position-level BERT features and
records official outputs for `enc_p`, deterministic duration prediction,
prior expansion, inverse flow and the speaker-conditioned HiFi-GAN decoder.
It never imports Vokra and never reads a Vokra GGUF. The versioned
`features.vmf` is the same deterministic input in the public CLI contract; it
does not claim that a raw-text frontend is embedded in the acoustic GGUF.

The released Vokra artifacts used by the Rust CPU gate are:

| Variant | Public revision | GGUF SHA-256 | Bytes |
|---|---|---|---:|
| English | `41fc375b3677373e2141ba5b80cd072581ee4308` | `1196312e86d8e9ba553f505d8cbc151cf6a53c56d0c91dd1c1989c26e2567ee4` | 207,575,360 |
| Chinese | `2d02213da50af3d5384c2f972681014a2eb05ab5` | `11f87f890e95cf572ad207aae87f6a961b7c9ebe4eee81c69b0a6c2440376a1e` | 207,484,736 |
| Korean | `3737e27dba5f54e98ab3ae816bf610ae6edaeeb2` | `6e27bbc9c55dd5acc756317044be42fac4a85f5315aca38cfb881ac5984f24d9` | 207,575,360 |
| Spanish | `1ee8c1c2df484ea59bd7382f88b292b0da95df3e` | `3a293e474c3d51e271a4bcb7e980f5f3e6866cbf2ba9a7c3780cf36f9c10e184` | 207,575,360 |
| Japanese | `5c61fa7b6f723c039e7d4721f3d5ab77b99d867e` | `f12c079ae4df51e59895ac29a8bb0043ae3c78be3aa1ad22ab84de71d4ff81a8` | 207,575,360 |

Regenerate on a disposable VAST instance through the repository Python 3.12
environment:

```sh
git clone https://github.com/myshell-ai/MeloTTS.git /tmp/melotts-upstream
git -C /tmp/melotts-upstream checkout 209145371cff8fc3bd60d7be902ea69cbdb7965a
uv run --project tools/parity/melotts --frozen \
  python tools/parity/melotts_dump_reference.py \
  --source-root /tmp/melotts-upstream \
  --variant <english|chinese|korean|spanish|japanese> \
  --output crates/vokra-models/tests/fixtures/melotts_<variant>
```

`manifest.json` pins the source/checkpoint revisions, source artifact hashes,
Floating files are little-endian FP32; ID files are little-endian U32;
durations are little-endian I32. The Rust comparison keeps the repository FP32
gate unchanged at `atol = 0.01`.
