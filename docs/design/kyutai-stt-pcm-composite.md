# Kyutai STT PCM composite boundary

> **2026-10-14 source reconciliation:** The schedule below records the
> original candidate, not an authenticated PyTorch PCM reference contract.
> [Fixed-source comparison](../handoff/kyutai-stt-pcm-contract-reconciliation-2026-10-14.md)
> found MLX-style padding mixed with PyTorch-style LM inputs and an unjustified
> first double call. The correction targets the private DSM/PyTorch PCM path;
> the public MLX padding contract remains separate. No real-weight parity or
> public loader enablement follows from this review.

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

## Original runtime schedule (retained)

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

## 2026-10-14 private PyTorch correction

The private PCM implementation now targets the fixed DSM/PyTorch caller,
not the separate MLX preparation convention above. It prepends 13 complete
Mimi silence frames (24,960 samples), right-pads the caller audio itself to
the next 1,920-sample boundary, and appends 32 complete silence frames
(61,440 samples). Empty audio has 45 padding frames. Each encoded row,
including the first, makes one native LM call; its sampled token enters the
raw stream and is emitted unless it is ID 0 or 3. EOS does not stop the
schedule. The public MLX `KyutaiSttStreamingContract` is unchanged.

Focused callback tests cover the exact expected padded PCM sequence,
partition-independent framing, the first single call and feedback, empty
audio, finish capacity failure before callbacks, poison/reset and suppression.
These are source/mechanism checks, not independent framework or real-weight
parity. At the correction's source commit, Rust compilation/test execution
were still pending. The [subsequent exact-head VAST replay](../handoff/execution-readback-2026-10-14.md#exact-source-replay-passed-and-owned-worker-destroyed)
at `4263f416` passed Clippy and workspace tests, including all 20 private PCM
tests; the owned worker/storage were destroyed. Independent full-PCM and
real-weight parity remain absent.

## Evidence boundary

The tests in `pcm_session.rs` are model-free control tests for digest parsing,
whole-file mmap authentication failures, measured Mimi provenance metadata,
backend preflight rejection, exact padding/carry chunking, overflow and cap
handling, transactional callback failure, reset, greedy tie behavior, the
original first-double-step schedule, and raw-token suppression. The current
correction replaces the double-step test with the single-step schedule and
adds private PyTorch padding/cap coverage. The scheduling and
buffering helpers used by those tests are the same private helpers used by the
production session callbacks. They do not execute learned weights, establish
ASR quality, or provide numerical parity. A future independent packet must
compare Mimi rows, LM logits, raw sampled IDs, and decoded text against the
pinned official reference before this seam can be promoted or exposed
publicly.
