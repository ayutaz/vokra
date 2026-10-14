# Kyutai independent-reference routing readback — 2026-10-14

## Accepted baseline and scope

[PR #230](https://github.com/ayutaz/vokra/pull/230) was normally squash-merged
at 06:19:07 UTC. Its reviewed head was
`8dd9b1ff772a665683b7b928d64a94b15c26ef84`; accepted main is
`f8804b7429691bd90520aec801869262ba53573b`. The final head readback contained
86 check runs: 81 success, five skipped, no pending or failed checks. All 16
required context/application pairs succeeded. No protection override was used.
Local main was synchronized and clean before this documentation branch.

Subsequently, Dependabot [PR #231](https://github.com/ayutaz/vokra/pull/231)
was normally accepted at 07:03:27 UTC, producing main
`9ddc0da53edae6ac8ebf25600891d849327be83b`. Its only changed file updates
pyannote's lock from urllib3 2.7.0 to 2.8.0. Official PyPI JSON matches both
artifact URLs, SHA-256 values and sizes; Python `>=3.10` admits this tree's
3.12 pin, and upstream's license remains MIT. Exact head `a6c9d7d` had
72 successful checks and one skipped check, including all 16 fresh required
context/application pairs. This is dependency/source acceptance, not a
pyannote model run or reference-closure approval.

The former local main documentation commit `1a63a8d1` remains recoverable on
`codex/archive-main-pre230-20261014`; it was not discarded or force-reset.
The [exact source replay](execution-readback-2026-10-14.md#exact-source-replay-passed-and-owned-worker-destroyed)
remains evidence for `4263f416`, not a new real-weight run at the merge commit.

This follow-up reads public source/configuration/registry metadata only.
No model, Mimi payload, tokenizer payload, tensor header or model execution
was acquired locally. No new cloud lease or upload was started.

## Same declared model bytes, different runtime contracts

The fixed Hugging Face API trees are:

- [PyTorch distribution](https://huggingface.co/api/models/kyutai/stt-2.6b-en/tree/a07aec56d22be5589cd0bc8709c75b6cf3e3039d?recursive=true&expand=false),
  revision `a07aec56d22be5589cd0bc8709c75b6cf3e3039d`.
- [Candle distribution](https://huggingface.co/api/models/kyutai/stt-2.6b-en-candle/tree/8d380a4d647cd8f67b22e58f50424c8bf04c182b?recursive=true&expand=false),
  revision `8d380a4d647cd8f67b22e58f50424c8bf04c182b`.

Both trees declare the following identical LFS objects and byte sizes:

| File | Bytes | LFS SHA-256 |
|---|---:|---|
| `model.safetensors` | 5,234,275,128 | `2471add7da1fdb2d5dc4561e88a9069376333d992760d55d29d1db46c52849b2` |
| `mimi-pytorch-e351c8d8@125.safetensors` | 384,644,900 | `09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50` |
| `tokenizer_en_audio_4000.model` | 59,339 | `d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6` |

This is content-addressed metadata identity, not a downloaded-byte receipt,
license sign-off, weight-key inspection, runtime compatibility or parity result.
Both repository metadata records declare `cc-by-4.0`; their six-file trees
contain no separate LICENSE file. No new owner decision is inferred.

The configuration bodies differ. PyTorch's 1,257-byte config has Git blob
`cb670551e4f81233fa9f60e025f9101e14b7bc88` and SHA-256
`b79ea52a30329887a2d0ce2dd5473a63fc5083e441e7986f64f01050c06239c9`;
Candle's 1,129-byte config has blob
`4058808f177fd0bcf196d5f1ff5c3e6e317e68b7` and SHA-256
`bbc6a18a4ded8eee5ef8ba4ccfef789dd1d6ef3828f23575b06bf4d166616d99`.
Both declare dimension 2,048, 32 heads, 48 layers, 32 codebooks and context 375.

The official [fixed Rust ASR TOML](https://raw.githubusercontent.com/kyutai-labs/delayed-streams-modeling/4c4f65e147df056adf3346290d64c7b9649b18c9/configs/config-stt-en-hf.toml)
selects this 2.6B architecture dynamically. The Rust 1B convenience factory
is not evidence for this model. Its HF aliases must become immutable local
input identities before execution.

The apparent feed-forward difference is not itself a weight-shape mismatch:
the [fixed Python gating implementation](https://raw.githubusercontent.com/kyutai-labs/moshi/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi/moshi/modules/gating.py)
derives width 5,632 from 8,448, while the
[fixed Rust MLP implementation](https://raw.githubusercontent.com/kyutai-labs/moshi/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/rust/moshi-core/src/transformer.rs)
derives the same width from its special 8,192 branch. No numerical equivalence
is inferred, and the official TOML is not silently rewritten to 8,448.

## Dependency admission: fixed Rust lock is not green

The [fixed upstream lock](https://raw.githubusercontent.com/kyutai-labs/moshi/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/rust/Cargo.lock)
contains 592 package entries and pins Candle core/nn/transformers 0.9.1,
safetensors 0.4.5, memmap2 0.9.5 and zip 1.1.4. This is the whole upstream
workspace lock, not a resolved Linux core-only/default-feature closure.
Optional GPU, other workspace-member, target and dev/build dependencies must
not be presented as the active CPU graph.

At RustSec database commit
`6aed84668c23d7f535077a9fe918bf1fa0f6a5bc`, the
[memmap2 advisory](https://raw.githubusercontent.com/RustSec/advisory-db/6aed84668c23d7f535077a9fe918bf1fa0f6a5bc/crates/memmap2/RUSTSEC-2026-0186.md)
records unsound range methods and patched versions `>=0.9.11`. Locked 0.9.5
is below that threshold. Do not use this unchanged lock for model execution
or suppress the finding as a substitute for a corrected, reviewed scope.

The published [Candle-core 0.9.1 manifest](https://docs.rs/crate/candle-core/0.9.1/source/Cargo.toml)
declares memmap2 as a non-optional dependency with requirement `0.9.3`;
disabling GPU features does not remove it. A precise update to 0.9.11 fits
that declared semver range, unlike overriding the PyTorch Torch ceiling.
This is a candidate resolution, not an executed lock update or compatibility
result. Preserve the original lock, inspect every resulting package change,
and bind the new checksum/feature/target/license/advisory evidence separately.
An isolated core projection must also avoid other workspace members' feature
unification. Whole-lock audit output and Vokra's runtime-only dependency bans
are not substitutes for that reference-specific projection and review.

The [zip advisory](https://raw.githubusercontent.com/RustSec/advisory-db/6aed84668c23d7f535077a9fe918bf1fa0f6a5bc/crates/zip/RUSTSEC-2025-0168.md)
explicitly marks `<1.3.0` unaffected; locked 1.1.4 is not affected by that
particular advisory. Neither observation is a whole-closure security verdict.
The complete fixed database tree has no safetensors advisory directory;
absence from that database is not proof that the package or closure is safe.

## Remaining exact gates

1. Establish the Linux core-only CPU dependency/feature projection, primary
   crate licenses and any native/build payloads. Prepare a separately identified
   patched lock; preserve the upstream original and verify support/compatibility.
   Do not rent a worker merely to repeat the known-blocked unchanged lock.
2. Bind a concrete full-PCM official reference contract. Rust ASR consumes the
   previous Mimi row with initial padding; its global model counter is not reset
   by `State::reset`. Its pre-forward callback alone does not expose the last
   greedy token/logits. Observe actual upstream state, not a hand-written mirror
   or token-position correction. Retain context-375, reset, explicit precision,
   tokenizer and complete PCM scheduling evidence.
3. Resolve exact source/model/dependency/operator review and execution authority.
   Candle is a distinct source/runtime scope, not carryover approval from Draft
   #191 or the private PyTorch helper. The Python caller pins
   [Moshi 0.2.11](https://pypi.org/pypi/moshi/0.2.11/json) while the
   [inspected source](https://raw.githubusercontent.com/kyutai-labs/moshi/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362/moshi/moshi/__init__.py)
   identifies 0.2.13; their implementation/dependency
   differences must not be erased by identical checkpoint metadata.
4. Only then run the no-upload full reference and native real-weight CPU parity
   on disposable VAST. Recover small exact evidence and destroy worker/data.
   Final Apple packet and Scaleway follow the all-row non-Apple exit gates.

The 58 unresolved public rows remain unresolved. No CPU/Metal row promotion,
owner sign-off, new parity bound, public transcription enablement, publication
or final packet follows from this readback.

## Documentation verification

The docs reference, runbook-path citation and documentation-example gates
passed, along with diff whitespace hygiene. The example gate still lists 30
deferred platform/UI blocks, not newly verified hardware/toolchain examples.
No local Cargo compilation, model execution or paid dependency audit was run.

The first PR #233 documentation-link job `115725024350` at `510dc36a`
failed with three transport-status entries: one GitHub CONTRIBUTING link
returned 503 and the existing HF organization link returned 504 in two files.
The mandatory gate correctly failed; no new status acceptance, exclusion or
protection override is introduced. The accepted PR #231 main was normally
merged into the documentation branch, so subsequent CI must cover its new
head, not relabel this failed job as passed.

Both failed public pages returned HTTP 200 in the subsequent bounded
credential-free readback. The normal push's compliance regression passed
8/8, then the maintainer safety hook refused the deep Cargo path triggered
by the already-accepted dependency lock merge; no Cargo compilation began.
Manager verified that all Rust/crate/root Cargo/C-ABI test inputs remain
identical to the recorded green VAST `4263f416` run, and that the diff from
accepted main consists only of the four documentation files. The dependency
lock itself already passed PR #231's exact-head CI. Use the established
recorded-remote-green push procedure for this documentation update, without
changing hooks or repeating the unchanged native run on a paid worker.
