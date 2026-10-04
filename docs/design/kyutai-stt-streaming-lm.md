# Kyutai STT incremental language model

`KyutaiSttStreamingLm` is the one-frame, main-language-model component for
the `dep_q = 0` Kyutai STT path. It owns one bounded key/value history per
transformer layer and applies RoPE using the absolute frame position. The
context limit is read from the authenticated Kyutai backbone metadata; it is
not a separate guessed streaming constant.

The implementation follows the published Moshi `LMGen` boundary semantics at
revision `e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`: the first frame uses the
initial text/audio sentinel rows even though the caller supplies the current
audio frame, while later frames use the previously sampled text token and the
current audio codes. Each step performs one transformer row and never
recomputes a full prefix. A failed step clears all layer caches and poisons the
stream until `reset` is called.

Primary sources: [Moshi `LMGen` at the pinned revision](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi/moshi/models/lm.py)
and the [pinned Moshi package license](https://github.com/kyutai-labs/moshi/blob/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi/LICENSE).
The released Kyutai STT configuration carries a 375-frame context bound;
the implementation reads that bound from authenticated metadata rather than
hard-coding it. This change has not validated the real checkpoint: its tests
use synthesized weights and a context of 3, so the real-weight bound remains
pending.

The component is intentionally not a complete STT implementation. It does
not encode PCM with Mimi, sample text, tokenize, or claim upstream numerical
parity, transcription quality, or Apple CPU/Metal completion. Its synthetic
tests cover state and boundary invariants only. Real-weight reference parity,
Mimi composition, and platform completion remain separate gates.
