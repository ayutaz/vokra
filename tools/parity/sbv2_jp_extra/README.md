# SBV2 JP-Extra source-contract boundary

This directory is a plan and landing point for the production Japanese G2P
contract. It deliberately contains no copied Style-Bert-VITS2 implementation,
phoneme table, tone algorithm, or production runtime wiring.

## Authenticated inputs

The contract generator must execute the official reference in an isolated VAST
worker, using these immutable identities:

- HF model revision: `a731761009f3c96d104487be6ad332bf1bb5a3a5`
- Official Style-Bert-VITS2 tag: `2.0`
- Official source commit: `ef93f388fc1ddf0dc0f598126c1964923f1df94f`
- `text/symbols.py` blob: `846de64584e9ba4b8d96aab36d4efbcefb1a11e7`
- `text/japanese.py` blob: `5c055875626c16bd7d3489d02b4952ec90a3bbf6`
- License blob: `0ad25db4bd1d86c452db3f9602ccdbe172438f52`

The worker must verify every identity before importing the reference. A
different revision or missing blob is a hard failure, not a best-effort
substitution.

## Required generated artifacts

The VAST worker should emit a hash-bound contract sidecar containing, as data:

1. the exact sorted symbol table and row count;
2. language-id mapping and per-language tone offsets;
3. model tensor-derived vocabulary/tone dimensions;
4. the reference source/checkpoint identities above; and
5. deterministic reference outputs for a small Japanese sentence corpus,
   including phones, tones, word boundaries, and normalized input text.

The sidecar and fixtures must be reviewed as independent reference output,
then committed with SHA-256 manifests. The converter may consume the sidecar
only after validating its source hashes, model identity, dimensions, and
internal lengths; malformed or mismatched data must fail closed. It must embed
the validated contract data in GGUF rather than recreate it from constants.
The runtime must read and validate that GGUF data and must not hardcode a
replacement mapping.

## License and execution boundary

The official implementation is AGPL-3.0. It may be executed only as an
independent reference in the VAST parity worker; no AGPL source may be copied
into Apache-licensed runtime, converter, or integration code. No model weights
are acquired or executed locally. Until the VAST-generated contract and
committed fixtures exist, `sbv2-v2-jp-extra-base` Japanese production G2P is
intentionally unavailable and callers must receive an explicit failure.
