# Kyutai STT PCM contract reconciliation — 2026-10-14

The read-only review starts at clean source `c4c024c0e0c182777b5a8720fdb07db20e53e81b`.
The crate-private PCM composition already exists; a second PCM/Mimi/LM join
is not the next task. Its input preparation and first-frame scheduling must
instead be reconciled with one explicit independent reference contract.
No checkpoint, tokenizer or learned model was acquired or executed in this
review. Source facts do not establish API compatibility or numerical parity.

## Fixed primary sources

These are the UTF-8 bodies obtained from the official repositories at the
fixed revisions, with byte counts, SHA-256 and Git blob SHA-1 measured from
the retrieved text. The DSM caller identity exactly matches the existing
`tools/parity/kyutai_stt_pytorch_pcm_source_receipt.json`; that receipt is
source-only and remains `execution=NOT_RUN`.

| Source | Bytes | SHA-256 | Git blob SHA-1 |
|---|---:|---|---|
| [DSM PyTorch caller](https://raw.githubusercontent.com/kyutai-labs/delayed-streams-modeling/4c4f65e147df056adf3346290d64c7b9649b18c9/scripts/stt_from_file_pytorch.py) | 8452 | `2ac2d9bff71d3d6a874bed070eb9d4e60736209e3cbaa9697fbe735dc79d2955` | `cf3fb05b0e0c1f265a667276d2886ce2664d79ff` |
| [Moshi PyTorch LM/LMGen](https://raw.githubusercontent.com/kyutai-labs/moshi/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi/moshi/models/lm.py) | 37312 | `38991e83d7e3aa0ff1483b27a0b59e1ff43642ad6a32dc8dc4272ad2b6c0dc5f` | `209b7a59c9c086810a81a7d8b99c5233a0ad87ff` |
| [Moshi PyTorch sampling](https://raw.githubusercontent.com/kyutai-labs/moshi/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi/moshi/utils/sampling.py) | 4699 | `113da9542ecdcc521945a337a63bec28abba7fd61c2eacaec5855fc7d7562126` | `2a35f6cdff1981bb6914b27410c6b152be2a77b1` |
| [Moshi PyTorch transformer/cache](https://raw.githubusercontent.com/kyutai-labs/moshi/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi/moshi/modules/transformer.py) | 38159 | `f5a73d752a5bde1eda2b0b14bebd13fd81db81017d361782a07164580a687622` | `244e73a3bc012256325e5b759cd13588a96d2673` |
| [Moshi MLX file caller](https://raw.githubusercontent.com/kyutai-labs/moshi/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/run_inference.py) | 6137 | `6214618788b16fe21ab2dd8ae55d115f3669a2d775a57cae3fa5af941a94cc3b` | `bf2b6ca0221cc377b418ffd26c2d1317217c3f69` |
| [Moshi MLX generation](https://raw.githubusercontent.com/kyutai-labs/moshi/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi_mlx/moshi_mlx/models/generate.py) | 5689 | `3f995b9f9275dd21c5ecd6fc21fe004fdef2461a2dccccfae264237981c7a54b` | `79908f61cc20b4dc88b2efd43eabdbe4422da249` |

The [PR #230 link-transport follow-up](pr230-link-transport-2026-10-14.md)
changes only these official source URLs, not any measured source identity,
contract, test result or approval.

## Causal finding: distinct official contracts were mixed

The pinned STT-2.6B configuration already recorded by the inspector has
24 kHz PCM, 1920 samples per Mimi frame, a 1.0-second prefix and 2.5-second
audio delay. Applying the two official callers to those axes gives different
schedules:

| Boundary | Fixed DSM/PyTorch caller | Fixed MLX caller | Existing private PCM candidate at review start |
|---|---|---|---|
| Prefix | `ceil(1.0 * 12.5)` = 13 whole silence frames / 24960 samples | 24000 zero samples concatenated with audio | MLX-style 24000 samples |
| Audio tail | Pad the audio itself to a whole frame | Floor the length of the complete padded stream | Drop the final partial concatenated frame |
| Suffix | `ceil(2.5 * 12.5)` = 32 whole silence frames / 61440 samples | `(2.5 + 1.0) * 24000` = 84000 samples | MLX-style 84000 samples |
| LM schedule | One `LMGen.step` per Mimi row, including the first | One generator step per Mimi row | Two native LM calls for the first row, then one |
| Audio selected by LM | Current row, with the initial sentinel substitution at offset zero when all channel delays are zero | `step_idx - 1 - delay`, initially audio padding | PyTorch-style current row / initial sentinel |

The initial double call is not justified by either inspected fixed generator.
PyTorch `LMGen._step` increments its offset before the early-output check;
with the authenticated zero channel delays, the first call returns a sampled
text token. It is not a discarded warm-up call. The DSM caller requests
`temp_text=0.0`, which selects `torch.argmax` in `sample_token`; an EOS token
does not break the PCM loop. IDs 0 and 3 are suppressed for piece emission.
These are source observations, not executed framework tie-rule or tokenizer
equivalence evidence.

The existing public `KyutaiSttStreamingContract` explicitly documents the
MLX padding convention. A private PyTorch PCM correction must not silently
rewrite that public convention or present MLX/PyTorch outputs as interchangeable.
The existing Python decoder dumper is a separate explicit text/Mimi-row
component reference, not an independent full-PCM oracle.

The fixed PyTorch attention constructs its ring cache with `capacity=context`
and masks keys with nonnegative positions and `0 <= query_position -
key_position < context`. Its reset zeros offsets, causing old positions to
be invalid until overwritten. The native single-stream state instead stores
the retained positions chronologically and clears its vectors on reset.
The retained position set can agree while physical key ordering, precision,
reduction order and numerical outputs still differ. No cache numerical
equivalence is inferred; the real upstream reference must cross the recorded
375-frame context boundary, not only compare short synthesized sequences.

## Bounded correction and remaining gates

Use the fixed DSM/PyTorch caller as the explicit reference target for the
crate-private PCM session: whole prefix/suffix frames, right-padding the
caller audio only, one sampled LM step per encoded Mimi row, and first-token
emission subject to the same 0/3 suppression. Keep the public MLX contract
unchanged and make the private preparation convention explicit.

Production helper tests must cover empty and partial audio, arbitrary push
chunking, caps/overflow, finish/reset/poison behavior, first-step feedback and
EOS continuation. Such callback tests protect mechanics only. The next
independent PCM reference must observe actual Mimi codes, LM logits, sampled
tokens and text through the fixed official implementation, including the
context-window boundary. Exact dependency/license/operator approval,
real-weight CPU parity and Apple CPU/Metal/no-fallback remain open.

No row promotion, owner sign-off, upload, final Apple packet or Scaleway run
follows from this source review. The public loaders/transcription entry points
must remain explicitly unsupported until the separate gates close.

## Correction review and verification boundary

The bounded implementation changes only the private `pcm_session.rs` path;
the public MLX contract and the native Mimi/LM implementations are unchanged.
Manager review found an unused parameter and a success test that allocated
only 45 frames for one sample of audio (which requires 46). Those were
corrected before remote execution. The PCM test also now compares captured
frames against the explicit expected prefix/audio-only-tail/suffix sequence,
not merely against another invocation of the same helper.

Formatting, zero-dependency, forbidden-symbol and documentation citation
checks pass. No local Rust compilation or model execution was performed.
The correction is not yet supported by a green exact-head remote Rust run,
an independent full-PCM reference or real-weight numerical evidence.

### Subsequent exact-head Rust result

The earlier pending Rust boundary above is superseded by the
[recovered VAST source replay](execution-readback-2026-10-14.md#exact-source-replay-passed-and-owned-worker-destroyed)
at `4263f41631bd15288f06d3cd5c6b83ea4a8fdf60`: Clippy and the full workspace
tests/gates passed, including 20/20 private PCM tests. Instance `55867942`
and storage were destroyed with explicit individual null readback. This
does not supersede the missing independent full-PCM reference, dependency/
license/operator closure, real-weight CPU parity or Apple evidence.
