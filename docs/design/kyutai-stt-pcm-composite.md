# Kyutai STT PCM composite boundary

This design records the first native PCM-to-text composition seam for the
authenticated Kyutai STT-2.6B-EN components. It is crate-private and does not
enable the existing public `from_gguf`, `from_path`, or `transcribe` routes.
Real-weight CPU parity, backend coverage, independent reference comparison,
and owner/legal review remain open.

## Construction and provenance

`kyutai_stt::pcm_session::KyutaiSttPcmEngine::from_paths` maps the decoder,
dedicated tokenizer, and converted Mimi GGUFs read-only. An external
authenticated transfer packet must provide each GGUF's whole-file SHA-256 and
byte count; the mapped bytes are measured before GGUF parsing and must match
those values. A caller-supplied path, filename, or digest is not by itself a
source or rights decision.

The raw Mimi sidecar is independently checked against the fixed Kyutai
identity already recorded by the STT component (`384,644,900` bytes and
`09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50`). The
converted Mimi GGUF must also contain the measured input fields emitted by the
current converter:

- `vokra.provenance.checkpoint_sha256` = that exact raw Mimi SHA-256;
- `vokra.provenance.checkpoint_bytes` = `384644900`.

This is a strict source-to-converted-artifact identity gate, not an independent
conversion proof. The decoder and tokenizer whole-file packet, converter
lineage, weight rights, and publication decision remain separate authenticated
records. If the conversion metadata or external packet is absent, construction
fails closed.

## Runtime schedule

The engine uses the existing `MimiEncoder`, `KyutaiSttAsr`,
`KyutaiSttStreamingLm`, and `KyutaiSttTokenizer` components. It validates the
fixed 24 kHz / 1,920-sample / 32-codebook contract and preflights the complete
Mimi and decoder backend capability sets before returning a session. No
unsupported backend operation falls back to CPU.

Each session prepends 24,000 zero samples once, buffers only incomplete PCM
frames, and appends 84,000 zero samples exactly once at `finish`. Complete
frames are encoded immediately; any final partial frame is dropped. The first
Mimi row is passed to the LM twice: the first greedy sample seeds the next
call and is retained only in the raw diagnostic stream; the second and later
samples are externally emitted after suppressing text IDs `0` and `3`. EOS ID
`2` is retained in the raw stream and does not stop processing. Argmax is
finite-checked and first-index on ties.

Any invalid/non-finite PCM, shape, token, overflow, backend, encoder, or LM
operation poisons the session. `reset` clears both causal states, PCM carry,
schedule, and outputs together. A successful `finish` is terminal and cannot
be repeated.

## Evidence boundary

The tests in `pcm_session.rs` are model-free control tests for digest parsing,
whole-file mmap authentication failures, measured Mimi provenance metadata,
backend preflight rejection, exact padding/carry chunking, overflow and cap
handling, transactional callback failure, reset, greedy tie behavior, the
first-double-step schedule, and raw-token suppression. The scheduling and
buffering helpers used by those tests are the same private helpers used by the
production session callbacks. They do not execute learned weights, establish
ASR quality, or provide numerical parity. A future independent packet must
compare Mimi rows, LM logits, raw sampled IDs, and decoded text against the
pinned official reference before this seam can be promoted or exposed
publicly.
