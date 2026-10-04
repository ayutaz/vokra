# FireRed PCM beam composition

Status: explicit diagnostic seam, `SOURCE_IMPLEMENTED_PARITY_PENDING`.

This design records the narrow native composition contract for
FireRedASR-AED-L. It does not enable the ordinary `AsrEngine` transcription
surface, claim numerical parity, approve checkpoint redistribution, or infer
owner/legal sign-off. Real-weight execution remains a VAST-only task after the
independent upstream reference and license/dependency gates are satisfied.

## Production path

`FireredAsrAed::transcribe_pcm_beam_pending` accepts finite mono `f32` PCM and
an authenticated sample rate, then uses only bound model state:

1. Validate the authenticated sample rate and exact `batch_beam_search` policy
   before touching runtime tensor execution. Missing or drifted metadata is a
   loud error.
2. Convert PCM with the existing first-party Kaldi fbank contract: 16 kHz,
   80 bins, 400-sample frames, 160-sample shift, Povey window, DC removal,
   pre-emphasis `0.97`, low frequency `20`, log-power output, and authenticated
   CMVN. No resampling or guessed PCM prefix is inserted.
3. Run the existing `Compute`-dispatched encoder with an all-valid source mask,
   validate `[source_frames, 1280]` memory, then call the existing native
   FireRed `batch_beam_search` implementation. Its per-beam decoder caches are
   independent; `decode_max_len=0` means encoder-output time `Ti`.
4. Select the official n-best result (`beam_size=3`, `nbest=1`, smoothing
   `1.25`, length penalty `0.6`, EOS penalty `1.0`), validate finite score,
   `u32` conversion, vocabulary bounds, and rejection of SOS/EOS/PAD/BLANK in
   content, then render through the authenticated output dictionary.

The result is `FireRedPcmBeamHypothesis`, deliberately distinct from the
ordinary `Transcription` API. The helper used for final hypothesis validation
is the same helper called by the production diagnostic method, so model-free
tests cover the actual post-beam validation path rather than a numerical
mirror.

## Structural markers and failure order

The pinned FireRed decoder uses SOS `3`, EOS `4`, PAD `2`, and BLANK `0`.
Native beam output excludes SOS and strips only a terminal EOS. The composite
result validator rejects all structural markers in content, non-finite
normalized scores, `usize` values that cannot become `u32`, and IDs outside
the authenticated vocabulary. No n-best hypothesis is a hard `ModelLoad`
error. An immediate-EOS hypothesis is different: the pinned source computes
length from non-EOS IDs and slices the result accordingly, so it legitimately
produces empty content and empty rendered text. Non-empty content is rendered
through the authenticated dictionary.

Metadata and backend failure are intentionally fail-closed before expensive
execution: wrong sample rate, missing/drifted official search metadata, and
missing authenticated sidecars fail before PCM/weights are used; backend
preflight in `from_gguf_with_backend` occurs before exact runtime provenance
binding. No selected backend falls back silently to CPU.

## Model-free coverage

The focused tests use synthetic GGUF metadata and a tiny synthetic dictionary;
they do not construct runtime weights or run inference. They cover:

- inspection-only composition refusing to execute without exact runtime tensor
  binding;
- authenticated sample-rate mismatch before runtime binding;
- missing and drifted search policy before sidecar/runtime work;
- valid hypothesis rendering through the test dictionary;
- immediate-EOS empty result preservation, non-finite score, SOS/EOS/PAD/BLANK,
  vocabulary overflow, and `u32` conversion rejection through the production
  validator;
- unsupported Vulkan backend refusal before provenance validation, proving no
  implicit CPU fallback.

These tests establish structural behavior only. They cannot establish fbank,
encoder, decoder, cache, beam-ranking, EOS, or rendered-text numerical parity.
The independent source helper and real-weight VAST comparison remain required,
with source/model/checkpoint/GGUF/sidecar identities bound in the transfer
packet before any promotion decision.

## Primary-source anchor

The immediate-EOS behavior is anchored to the pinned upstream
`fireredasr/models/module/transformer_decoder.py` at source revision
`834635e4cf277ed8ca92049fc375b17c3dc20748`:

`https://raw.githubusercontent.com/FireRedTeam/FireRedASR/834635e4cf277ed8ca92049fc375b17c3dc20748/fireredasr/models/module/transformer_decoder.py`

The source-only byte read was 11,033 bytes with SHA-256
`f0dd5d0ba224ec0be9d2778d3d4ae514ef5ab24c879436aad756353b81f4eedb`.
This was a text fetch for source authentication only; no Python import,
checkpoint, or model execution was performed.

## Open gates

The ordinary transcription API remains loud-partial until a pinned official
reference run and native CPU comparison agree at the frontend, encoder,
decoder-logit, cache, beam-parent/order, token, score, EOS, and dictionary
stages. Metal is a separate backend gate. Existing dependency, training-data
provenance, license, owner-consent, and publication gates remain unchanged.
