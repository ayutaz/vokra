# Streaming codec decoder C API

The `vokra_codec_decoder_*` surface lets an application run its acoustic model
elsewhere and use Vokra only for the stateful codec/vocoder half: one complete
codebook frame goes in, one mono PCM frame comes out.

## Ownership and threads

Each successful `vokra_codec_decoder_open(session)` returns an independent
causal state machine and retains the model session internally. The original
`vokra_session_t` may be destroyed after open without invalidating the decoder.

A decoder handle has one owner thread at a time. It may be moved between
threads while idle, but the same handle must not be used concurrently by
`push_codes`, `pull_pcm`, `reset`, or `destroy`. Applications needing parallel
streams open one handle per stream. Immutable weights are shared; mutable
causal state and scratch are not.

Vokra creates no worker thread for this API. The owner drives all progress by
calling push and pull, which keeps the surface valid for the single-threaded
Unity WebGL build.

## Shape and backpressure

Call `vokra_codec_decoder_n_codebooks` after open and pass that value back as
the `n_codebooks` argument of every `vokra_codec_decoder_push_codes` call. It
is checkpoint data, not a header constant, so models with different codebook
counts use the same ABI.

One push accepts one complete `[n_codebooks]` frame. A successful push reports
one emitted frame; pull it into a buffer of at least
`vokra_codec_decoder_frame_hop` floats before the next push. A second push with
PCM still pending returns `VOKRA_ERROR_INVALID_ARGUMENT` instead of dropping or
overwriting audio. A pull with nothing pending succeeds with `out_len == 0`.

After the first warmup cycle, successful push plus pull performs no heap
allocation. `reset` may rebuild scratch/state and is intentionally outside that
hot-path guarantee.

## Family support

The C layer is codec-family neutral: a model opts in through Vokra's streaming
codec engine trait only when it has a complete, real **causal frame** decoder.
Standalone Mimi is currently connected. DAC and SNAC now have complete offline
token-to-PCM decoders, but their released convolutional graphs are non-causal
whole-sequence models; presenting either one-frame push as causal streaming
would fabricate context/state semantics. They therefore remain explicit
unsupported families on this streaming handle and are exposed through offline
model APIs instead.
