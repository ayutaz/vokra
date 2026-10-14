# PR #230 primary-source link transport — 2026-10-14

The source replay remains bound to `4263f416`; PR #230 initially used
documentation-recording head `8f18a6a0`. No source, model, owner decision or
numerical gate changes in this documentation-only follow-up.

The original documentation-links job `115694375101` failed on two GitHub
HTML blob pages with HTTP 503 (ANTLR setup and VibeVoice guidance). Bounded
credential-free GET requests to those exact URLs returned 200 at 05:24 UTC.
After exact-head/failure revalidation, only that failed job was rerun once.
The new job `115696532722` passed the original two links but failed on three
different GitHub HTML pages with 503: Moshi MLX inference, Demucs README and
MUSDB README. Neither result is relabeled green. The moving error set supports
an external HTML-response problem, not a claim that the source files are absent;
the specific server-side cause is not established.

The fix uses the official raw bodies for the same source paths. ANTLR's tag
link is strengthened to the already authenticated tag commit. The six-row
Kyutai table consistently uses the raw endpoints whose exact bodies were
already measured. No citation is deleted, no checker is excluded, no HTTP
status is accepted as success, and no retry/timeout or workflow is weakened.
New-head required CI must prove that the correction passes.

Fresh official bodies at review time:

| Source | Bytes | SHA-256 | Git blob SHA-1 |
|---|---:|---|---|
| ANTLR setup at `e4c1a74c66bd5290364ea2b36c97cd724b247357` | 527 | `00ce3bb1670e8e4961d9ef2414dfa6ac467b52465382841b432578f91733ec67` | `75d0dc5f55c117e03801da4fe0f8a6552ef3f62a` |
| VibeVoice guidance at `94da20d98b2fa7688e9cbfaf7692ddb4954f7600` | 8095 | `5bdb2d52a0677e32074fa03b25ddd4d758a8856499113e5dd7d42d13acd886fe` | `6ca2deb24edf31fa342bd7c407e66143056f59ed` |
| Demucs README at `e976d93ecc3865e5757426930257e200846a520a` | 18462 | `b2156a62247dae102c6ca3b000022d67f512743bd919523df320780120a329be` | `fe7e77b992b7918eba9491119d313abf5bc3eff1` |
| MUSDB README at the existing `master` alias | 9567 | `83c68379d6e7f0f135c63e06c4e465a303b46f1baddf16a53c8d275b615c8d74` | `880ffe2b42bd3eeebcf10a056eedcc04359c7ded` |

MUSDB `master` remains mutable; this readback is not an immutable source pin
or a new dataset/model license decision. All historical terms, packet
denominators, scope hashes, signer fields and execution/publication holds
remain unchanged. The raw Kyutai bodies retain every size/hash in the
[source reconciliation](kyutai-stt-pcm-contract-reconciliation-2026-10-14.md).

This follow-up does not claim global GitHub recovery or guarantee future
network availability. The normal link checker remains authoritative for the
new HEAD. No VAST/Scaleway instance or local model/Cargo run is needed.
