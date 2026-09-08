# Irodori TextBlock reference route

This is the narrow, model-free reference environment for the independent
`irodori_tts.model.TextBlock` fixture. It is deliberately not the full
Irodori inference environment.

The oracle is imported from the authenticated upstream checkout
`Aratako/Irodori-TTS` at revision
`8224dafb46d0aba89209a8f905f1cb7e3299d9c1`. The primary source is the pinned
upstream [`irodori_tts/model.py`](https://github.com/Aratako/Irodori-TTS/blob/8224dafb46d0aba89209a8f905f1cb7e3299d9c1/irodori_tts/model.py);
the accompanying [`speaker_inversion.py`](https://github.com/Aratako/Irodori-TTS/blob/8224dafb46d0aba89209a8f905f1cb7e3299d9c1/irodori_tts/speaker_inversion.py)
is also pinned. The only external imports in the exercised
closure are `torch` and `safetensors` (via `speaker_inversion.py`). The package
`__init__.py` and the full upstream resolver are intentionally not imported.

The full upstream inference path remains blocked because its authenticated
lock reaches `dacvae -> descript-audiotools -> librosa -> soxr/soundfile`.
This isolated project must never be used to claim full TTS, codec, PCM, or
Metal parity.

The source license evidence is the exact 1,064-byte `LICENSE` payload at the
pinned commit, SHA-256
`dfd47cc99fd79cced8e0cab04ed81f70b3d4f2f475a75746b341671bd3991c00`, with
disposition `MIT`. The isolated dependency closure is structural evidence
only: its status is `PENDING_DEPENDENCY_LICENSE_AUDIT` and
`PENDING_OWNER_REVIEW`. It is not a license approval, executable reference
environment, or publication authorization.

Run the source-only audit without synchronizing or importing dependencies:

```text
uv run --no-project --offline --python 3.12 python \
  tools/parity/irodori_text_block_source_audit.py --self-test
```

The actual fixture dumper remains VAST-only and is gated by the same source
audit before importing the upstream class.

The VAST dependency worker performs a frozen, CPU-only sync of this project,
records publisher or locked-sdist license bytes and native ELF dependencies,
then validates the complete report schema and exact lock identities before
preserving the evidence. A factual pass still has disposition
`BLOCKED_OWNER_REVIEW`; it never authorizes reference execution or publication.
