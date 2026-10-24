# Whisper base provenance repair — 2026-10-23

## Scope and acceptance contract

The user requested steps 1–3 of the next CPU-preview announcement plan:
investigate the Whisper discrepancy, make the bounded correction and regression
tests, then validate and publish the corrected model through the gated workflow.
The publication target is **`vokra/whisper-base` only**. Other model repositories,
X posts, a new GitHub Release, Scaleway, and the follow-up guide/pin/PR campaign
are outside these three steps. This record does not authorize blanket model
publication or a new owner/legal decision.

Baseline: accepted main `92a336318a54f659fa991c129aaa991f46ea6231` (PR #289).
Work branch: `fix/whisper-base-provenance-2026-10-23`.
The pre-existing dirty `.codex/hooks/guard-local-models.sh` is unrelated and must
remain byte-identical; SHA-256
`0e268b8ca047d16019ba4df60ef478d44b615d38e73313adc4c7bd392fe93351`.
The protected CosyVoice manifest must not be read, hashed, materialized, staged,
or changed, including in a remote checkout.

The original [announcement review](announcement-readiness-2026-10-23.md) remains
dated evidence. Its receipts, model pins and warning are not silently rewritten.

Required exit evidence:

1. Read the exact published GGUF's provenance and reconcile it with pinned
   primary upstream sources. Do not infer an original conversion revision that
   the old artifact cannot establish.
2. Make only necessary metadata/source-attribution corrections; retain the
   existing signed MIT disposition for Whisper base in the
   [license audit](../license-audit.md), without creating or changing a signature.
3. Verify all tensor names, types, dimensions and payload bytes against the old
   artifact; preserve all unrelated metadata. Any upstream-reference comparison
   establishes content identity to that fixed reference, not numerical parity
   or historical conversion provenance.
4. Run focused regression tests on the remote worker and the publication
   dry-run with the correct upstream LICENSE/copyright text and generated card.
   Never hand-upload a card or bypass `publish-one.sh`.
5. Publish only after the target, candidate hash and all gates are checked;
   read back the new HF revision, card/license/source files and artifact identity.
6. Recover small evidence only, then destroy this task's instance and saved data
   and independently confirm its absence. Do not acquire or execute a model on
   the maintainer Mac.

## Primary-source observations

OpenAI Whisper main resolved to
`86098128c0b4f24f0e2aa2994de830614b474227` during this investigation.
The [fixed upstream README](https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/README.md#license)
explicitly places both code and weights under MIT, and the
[fixed LICENSE](https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/LICENSE)
retains OpenAI's 2022 copyright notice and MIT terms. The upstream HF card still
displays Apache-2.0; that discrepancy is recorded, not resolved merely by both
licenses being permissive.

Known published input contract, to be freshly verified on the worker:

- repository: `vokra/whisper-base`;
- revision: `dddd93ff6301a82aa3750d398cb2e0a599cd9e83`;
- size: 290,859,360 bytes;
- SHA-256: `7e774425585d6e9ba58ac5337b406522b1408ab6ff6765d2938b092aef4c8e27`.

The exact remote file name, GGUF header, current HF branch head, and fixed
upstream weight revision have not yet been recovered in this repair task.
Do not fill them from inference or claim this input has been repaired.

## Source investigation and bounded correction

The vanilla Whisper converter already stamps MIT, a permissive class and an
OpenAI source description. An explicit license override intentionally rewrites
the license/class/source. Its generic source wording matches the published card;
that match is a hypothesis about the artifact's path, not a substitute for
reading the actual GGUF header.

Luna added a synthetic regression in `crates/vokra-convert/src/lib.rs` and
corrected the inaccurate override comment. There is no global license-policy
change, forced relabeling of fine-tuned models, or generator/upload bypass.
The regression and any repaired artifact remain unaccepted until their remote
evidence is recorded below. Tensor-preserving restamping must use the established
converter path, not an invented tensor rewrite.

## Worker and credential boundary

Use the existing VAST key only through `vastai-safe.sh`. A fresh, verified offer
must be within USD 0.20/hour including 200 GB storage, with at least 64 GB RAM.
The initial worker has a 30-minute bound including cleanup; readiness is bounded
and errors must not cause an automatic second instance. Confirm the exact
created ID and unique label before SSH or destroy.

The first audit is anonymous and transfers no HF credential. Any HF credential
needed for the targeted publication must remain out of arguments, logs, files
and Git, and travel only through the approved encrypted SSH stdin path.
The previous static-audit controller's plain-text destroy response and pricing
object parsing failures must not be repeated. Recover phase evidence before
cleanup, and distinguish a failed controller from independent absence readback.

No paid worker has been allocated for this repair at the time of this entry.
No HF update, local model acquisition/execution, new Release, X post or guide-pin
update has been performed by this task at this point.

## Execution evidence

Pending. A successful source test, license sign-off or dry-run alone is not a
publication receipt or an announcement-readiness verdict.

### Audit attempt 1 — instance creation not established

The audit-only controller selected offer `37484919` and submitted one create
request with label `vokra-whisper-audit-f80e338376ed`. The CLI exited zero but
the controller could not establish a numeric instance ID; its recovery query
found no instance with that label. No SSH transfer or model audit was reached.
The original provider create body was not retained, so its rejection reason
cannot be inferred from this failed attempt.

An independent `show instances-v1 --raw --limit 100` readback then returned
`success: true`, `total_instances: 0`, an empty array and `next_token: null`.
There is no confirmed instance to destroy from this attempt. The failed receipt
is retained locally; it is not a successful audit or publication result.
The controller's create-response diagnostics must be corrected before another
root-controlled attempt. No automatic create retry is authorized by this record.

### Audit attempt 2 — SSH failure, cleanup confirmed

Offer `43339009` created instance `57335330`, label
`vokra-whisper-audit-beefb990c269`, at USD 0.13555555555555554/hour including
the requested storage. The create reply explicitly returned `success: true`
and that numeric ID. The provider reported running and a direct SSH endpoint,
but the first archive-transfer SSH command exited 255. No audit or publication
was reached. The controller did not retain stderr, so the precise SSH failure
category is not established by this attempt.

The controller destroyed the instance and confirmed individual absence. A
separate root `show instance 57335330 --raw` also returned `instances: null`.
Read-only account inspection confirmed the existing local public SSH key is
registered (four account keys, matching key present); no key bodies or private
key material were displayed. Before another paid attempt, the controller must
probe SSH readiness and distinguish transient connection failures from
authentication/host-key failures without exposing credentials.

### Audit attempt 3 — exact artifact inspected, cleanup confirmed

The corrected SSH readiness probe observed one refused connection and then a
successful response before any transfer. Instance `57335831`, label
`vokra-whisper-audit-ed7d877b43aa`, used offer `43339009` at
USD 0.13555555555555554/hour including storage. The remote audit succeeded.

- Exact published file: `whisper-base.gguf`, revision
  `dddd93ff6301a82aa3750d398cb2e0a599cd9e83`, 290,859,360 bytes,
  SHA-256 `7e774425585d6e9ba58ac5337b406522b1408ab6ff6765d2938b092aef4c8e27`.
- The current target head was the same revision. The actual GGUF contains
  245 tensors, `license=apache-2.0`, `weight_license=permissive`,
  `model_id=whisper`, the generic override source wording, and no attribution
  field. The discrepancy is inside the artifact, not merely its card.
- Fixed comparison reference: `openai/whisper-base` revision
  `e37978b90ca9030d5170a5c07aadb050351a65bb`, `model.safetensors`,
  290,403,936 bytes, LFS SHA-256
  `07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b`.
  Its HF card still reports Apache-2.0; it is not represented as MIT evidence.
- The fixed OpenAI GitHub LICENSE was retrieved successfully: 1,063 bytes,
  SHA-256 `b5d65a59060e68c4ff940e1eddfa6f94b2d68fdf58ed7f4dd57721c997e35e9d`,
  retaining OpenAI's 2022 copyright. The same fixed README explicitly declares
  both code and model weights MIT.

The controller completed with exit 0, destroyed the worker and confirmed
absence. An independent root individual readback also returned `instances: null`.
Only the small JSON audit was retained locally; no model bytes or HF secret
were transferred to the maintainer machine. Tensor-content identity to the
fixed reference, restamping, remote Rust tests and publication remain pending.

Local static gates `check-zero-deps.sh` and `check-forbidden-symbols.sh` also
passed. The unrelated dirty guard and approved converter overlay retained
their recorded SHA-256 values.

The normalized, model-free audit JSON for the repair worker has SHA-256
`d0906112fe13a2bcbf523de797870cfdf65e3cf3adeadef12df1db84e13d1ea4`.
The strict receipt validator accepted the actual audit and bound the target
head, exact upstream safetensors identity and OpenAI LICENSE hash above.
The new prepare/publish worker and lifecycle remain subject to root review;
their synthetic tests are not real-weight evidence.

### Repair attempt 1 — bootstrap failure, no validation or publication

Instance `57336523`, label `vokra-whisper-repair-907478013ac6`, was created
from offer `43339009` at USD 0.13555555555555554/hour including storage.
Its six-member input archive SHA-256 was
`f22e5dca461b2c9ad2fd79035f2e8642f1e64f78f886af812616049d6d00c763`;
members were the model-free audit JSON, phase contract, two worker Python
sources, tensor verifier and approved converter overlay. No credential or
model file was sent in that archive.

SSH probing and transfer succeeded. The worker completed the protected-file-
excluded checkout at `92a33631`, then returned a structured prepare failure:
`FileNotFoundError`, missing `rustup`. No Rust test, real-weight comparison,
restamp, dry-run or publication succeeded. No HF secret was sent.
The bootstrap had suppressed installer diagnostics and continued after failure;
that control flow must fail closed before another attempt. The installer was
also named `.rustup-init` instead of the required `rustup-init` basename;
upstream [rustup issue 3634](https://github.com/rust-lang/rustup/issues/3634)
documents the executable-name dispatch behavior. This is the identified
bootstrap defect, not evidence that a corrected remote installation has run.
The official 1.29.0 installer checksum text was independently fetched and
matches the existing repository pin; that pin is not relaxed.

The failure JSON was recovered before cleanup. The lifecycle destroyed the
worker and confirmed `instances: null`; a separate concurrent individual root
readback also found no instance. The failure receipt remains a failure.

### Repair attempt 2 — tensor verification passed, dry-run PATH failure

Instance `57336973`, label `vokra-whisper-repair-8bc57b16fbbb`, used offer
`37484919` at USD 0.12222222222222222/hour including storage. The corrected
bootstrap worked. The approved overlay's new vanilla/override regression and
the existing CrisperWhisper provenance regression both passed (one test each,
zero failures); the standalone converter release build also succeeded.

Both full verifier reports were recovered in the failure receipt before
cleanup:

| Comparison | Result | Tensor count | Unexpected tensor/metadata differences |
|---|---|---:|---|
| Old published GGUF versus conversion of the exact fixed upstream safetensors | PASS | 245 each | none |
| Old published GGUF versus MIT/source restamp | PASS | 245 each | none |

All three tensor aggregates were
`8302b81a7a2a6d96449dabd02774a3958f3d17965cc2dfab2633a46de96c0a8b`.
The reference GGUF SHA-256 was
`2c5fc6c217454459ab7a38d50f8212f9a209195d74f8596d0064c2150c8787b8`;
the repaired candidate was 290,859,488 bytes, SHA-256
`5eea7ff64a6ea80a6145e680e7451dc1fcdc92fe049f8aa96b4c60b8c5b86014`.
Only license, source and schema producer changed. The model ID and absence of
an attribution field were preserved. These are exact content comparisons,
not model inference or independent numerical-parity results.

The catalog gate passed, but `publish-one.sh` dry-run failed before staging
because its child process could not resolve `uv` in PATH. The controller had
used an absolute UV executable without exposing its verified directory to the
repository scripts. This is an environment defect; no publication gate is
waived. The task requires a corrected PATH, a complete green dry-run and a
fresh review before publication. No HF secret or upload was attempted.
The controller destroyed the instance and individually confirmed absence.

Additional read-only publication checks: `signoff_match.py --self-test` passed
(approval, ledger-alias, converter and escaped-pipe cases). The current
`publishability-report.py` completed but classified Whisper as having no
provenance stamp: its literal-source regex cannot recognize the converter's
`WhisperVariant`-selected stamp. That advisory source scan is not recorded as
a green Whisper result or used as publication authorization. The actual
artifact header, successful default-provenance regression, exact target
sign-off check and full `publish-one.sh` gates remain the acceptance evidence.
No parser change, owner signature or publication gate is relaxed here.

### Repair attempt 3 — published and read back (2026-10-23 07:26 UTC)

**Steps 1–3 are complete for `vokra/whisper-base`.** This does not clear the
old CPU-preview guide pin or complete the released-CLI pairing/PR work.

Instance `57337648`, label `vokra-whisper-repair-e273b60a73d6`, used offer
`43339009` at USD 0.13555555555555554/hour including 200 GB storage.
The corrected bootstrap exposes the verified UV executable to child scripts,
checks its exact path/version, and fails before model work if a tool stage
fails. The final controller passed seven offline regressions; the root
reviewed its implementation and independently repeated the relevant checks.

The remote worker ran the exact approved overlay on base `92a33631` with
Rust/Cargo 1.99.1 and UV 0.12.5. Both focused Rust regressions, converter release
build, catalog check, upstream tensor identity, restamp identity, owner sign-off,
card generation, upstream LICENSE retrieval and complete `publish-one.sh`
dry-run passed. The regenerated candidate exactly matched attempt 2's SHA-256
and size. The root inspected the full prepare evidence, generated card,
LICENSE, NOTICE and SOURCE before issuing the candidate-bound publish command.

Only then was the existing HF credential sent over encrypted SSH stdin.
It was not sent in an argument, archive, file or log. The publication process
was restricted to the official HF endpoint and the repository's
`publish-one.sh --push` chain for this one target.

Published immutable revision:
[`bf440f6064ffc882d53d3694729dc4616dc2cfa7`](https://huggingface.co/vokra/whisper-base/tree/bf440f6064ffc882d53d3694729dc4616dc2cfa7).

| Published file | Bytes | SHA-256 |
|---|---:|---|
| `whisper-base.gguf` | 290,859,488 | `5eea7ff64a6ea80a6145e680e7451dc1fcdc92fe049f8aa96b4c60b8c5b86014` |
| `README.md` | 1,977 | `1b9568d400500602c5b85c88c3709d291bc5848ad6a7397049dbd48473d6c577` |
| `LICENSE` | 1,063 | `b5d65a59060e68c4ff940e1eddfa6f94b2d68fdf58ed7f4dd57721c997e35e9d` |
| `NOTICE` | 305 | `d259de6d6887e16bcc76e8014cd2d7f32ba13bf9a3740b1b1305364ebfabacaf` |
| `SOURCE.md` | 879 | `9c32bc288936cb7103acb54317fb6f04eb0d575133f320d91265b7f430636d00` |

The worker read back the new revision's API/LFS identity, actual GGUF bytes,
MIT card and all four text files. Every size/hash matched the reviewed staged
bundle, including OpenAI's original LICENSE text/copyright. The readback
tensor verifier returned PASS for all 245 tensors and aggregate
`8302b81a7a2a6d96449dabd02774a3958f3d17965cc2dfab2633a46de96c0a8b`.
An independent HF connector lookup subsequently reported `license:mit` and
an update date of 2026-10-23.

The lifecycle completed with exit 0 and destroyed instance `57337648` and its
instance storage. Independent root readbacks returned `instances: null` for
that ID, then `success: true`, `total_instances: 0`, an empty array and
`next_token: null` for the complete instance inventory. No separate volume
was created by this task; the instance inventory is not a separate-volume
account audit. All paid attempts stayed within their 30-minute bounds.

The complete model-free success evidence is retained in
[the JSON receipt](evidence/whisper-base-provenance-repair-2026-10-23.json),
SHA-256 `9da9588842bac0a5e5d04a5067e22c432b618f65966e8d4e876fece03ce75839`.
The local guard retained its original recorded hash; the protected manifest
was not read, hashed, materialized or changed. No real model was acquired or
executed on the maintainer Mac, and no new model inference was run remotely.

The license audit's current source-citation cell was corrected to cite the
fixed OpenAI README/LICENSE rather than incorrectly claiming an MIT HF card.
License/class and owner-signature cells are unchanged. Local zero-dependency,
forbidden-symbol, formatting, documentation-reference and diff-hygiene checks
passed. No workspace or `vokra-models` Cargo command was run locally.

The converter regression, source-citation correction and this evidence are
still worktree changes for a later logical commit/PR. The pre-existing dirty
guard is unrelated and must not be included. Guide/pin updates, a new smoke
using the released CLI plus this new revision, PR/CI/merge and X preparation
remain the next plan steps; the old dated RC receipts remain intact.
