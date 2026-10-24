# PR processing and announcement readback — 2026-10-24

Checkpoint: **07:44 UTC**. This is a dated engineering record, not an assertion
that every model, dependency audit or Apple backend has been completed.

## Accepted main

| PR | Accepted commit | Acceptance evidence |
|---|---|---|
| [#298](https://github.com/ayutaz/vokra/pull/298) | `8bc58adbd010fb071295f3e44166745a480efc8c` | All 16 required context/app pairs and all three server-compat jobs succeeded at reviewed head `db4b19d7c55be1a7681540ef75b7412da2352631`; ordinary protected squash merge. |
| [#278](https://github.com/ayutaz/vokra/pull/278) | `6c5b4a86c21fe4a46b5b7c254a69f6c43a05a3f1` | All 16 required context/app pairs, all three server-compat jobs and the genuine current-head source-only producer/consumer succeeded at `7b28175be8c1e608779f160b6d1beccd67d6b157`; ordinary protected squash merge. |

#278's accepted tree is `f25d5a2614fa2f711d627e86b002bf65729ef029`, exactly
the reviewed source tree, with #298's accepted commit as its sole parent.
Local main and origin/main were advanced to `6c5b4a86` after acceptance.
Optional packaging jobs still pending at acceptance are not described as passed.

The [current consumer readback](candle-current-consumer-success-2026-10-24.md)
preserves the authenticated application/envelope/consumer artifact hashes and
raw advisory report. The real downloaded-artifact suite executed all 18 tests
with no skips. This source-only, fixed-database advisory result does not supply
license review, root runtime audit, model execution, weight parity or an upload
approval. Earlier cap failures and cancelled runs remain separately recorded.

## #271: reconciled candidate published; new-head execution pending

[PR #271](https://github.com/ayutaz/vokra/pull/271) is now at
`d404be56e856d9ae140a7195602639c3138f9e57`, incorporating the actual accepted
#278 commit. The final comparison against `6c5b4a86` contains exactly **58
modified files: 56 workflow YAML files and two Candle Python pin constants or
fixtures**. Every modified Git blob matches the reviewed Luna candidate.

The reviewed workflow projection contains all **60** current YAML workflows.
Four workflows using none of the updated actions remain byte-identical. The
previously omitted `release-candidate-published-cli-smoke.yml` has both its
setup-uv and upload-artifact pins updated; its existing uv 0.12.5 input remains.
The [earlier preparation](pr271-action-update-preparation-2026-10-24.md) and its
55-file manifest remain a dated partial checkpoint, not full current coverage.

The final candidate manifest has SHA-256
`cdf619911a1d8d646a7f6c8ad169c927eeaf9c6bdf5dbe96789b0d660778749d`.
The final Git tree is `3ebe4ed9d108541ce977e0cb7ca47d4f3ff165e7`.
Actionlint over all 60 workflows, the real forward/reverse pins checker, its six
self-tests, the old-pin residual scan and the four unchanged-workflow comparisons
passed. The manager independently repeated the real pins checker using the
existing offline parity environment, without installing packages or importing
models. The two Python candidates retain #278's authenticated exact-total byte
cap correction. Their offline regression result remains 37 executed passes and
16 fixture-dependent skips; it is not a genuine artifact-consumption result.

Fresh hosted workflows have been created at `d404be56`, including required CI
run [40075263238](https://github.com/ayutaz/vokra/actions/runs/40075263238) and
genuine source-only Candle integration run
[40075263086](https://github.com/ayutaz/vokra/actions/runs/40075263086).
They are **pending at this checkpoint**. #278's successful integration used the
older action pins and does not prove that the new pins execute successfully.
Merge requires fresh protected checks and real producer/consumer success at
this PR's own final head. No protection or evidence gate is bypassed.

## Independent dependency PRs

- [#297](https://github.com/ayutaz/vokra/pull/297), Pillow 13: all 16 required
  context/app pairs succeeded at `0524d0fbb164ea07a346160b3dc457f2ae0935db`.
  Documentation-links attempt 1 failed on an OpenSLR connection refusal and a
  GitHub HTTP 503. Both exact URLs subsequently returned HTTP 200, and one
  targeted retry passed as job
  [120805155841](https://github.com/ayutaz/vokra/actions/runs/40065380100/job/120805155841).
  No URL, exclusion, accepted status or checker rule changed. Since main has
  advanced, #297 needs a normal base update and fresh checks before acceptance.
- [#300](https://github.com/ayutaz/vokra/pull/300) remains Draft at
  `2fdf25e813cb5a8cdc735aeeab6db944051ae1a7`. It combines the filelock and
  setuptools source-audit inputs. The dedicated hosted source self-tests
  [passed](https://github.com/ayutaz/vokra/actions/runs/40067051660), without
  proving actual installed/native/model compatibility. Old capture and license
  evidence is not restamped to the changed closure. The shared heavy CI run
  was intentionally cancelled while these real-audit gates remain unresolved;
  it is not green. Final acceptance requires updated inputs, real audit evidence
  and full fresh CI. Its new source-test workflow must use the accepted action
  pins when it is reconciled after #271.
- [#299](https://github.com/ayutaz/vokra/pull/299) stays open until #300's
  combined responsibility is actually accepted. Its obsolete standalone shared
  CI run was cancelled; inclusion in a preparation commit is not acceptance.
- [#290](https://github.com/ayutaz/vokra/pull/290) was closed unmerged. Its CUDA
  closure is not substituted for the selected CPU dependency closure.

## Remaining model and owner conditions

| PR | Actual remaining condition |
|---|---|
| [#182](https://github.com/ayutaz/vokra/pull/182) / [#169](https://github.com/ayutaz/vokra/pull/169) | The approved Oct-13 static project/lock inputs already match the candidate. Missing raw capture/native source-build-license binding and independent real-weight CPU evidence remain; do not repeat the static audit merely because main had older inputs. #169 stays open until its responsibility is accepted. Exact candidate safetensors/tokenizers PyPI provenance endpoints returned 404, which does not establish the old installed wheel identity or a license violation. |
| [#191](https://github.com/ayutaz/vokra/pull/191) | The PyTorch upstream version restriction remains unresolved. The official Candle 2.6B source configuration is available and its active FFN dimensions are compatible with the PyTorch graph. The Rust route still needs authenticated checkpoint/tensor mapping and an independent real-audio/state/reset/context receipt; source compatibility and publisher HEAD metadata do not supply those results. |
| [#152](https://github.com/ayutaz/vokra/pull/152) | Existing Research-only weight sign-off is retained. Actual XCodec API/decoder compatibility and exact native source/license closure remain unproven. TorchAudio 2.11's stable ABI supports Torch >=2.11, so Torch 2.13 plus TorchAudio 2.11 is not rejected solely from unequal version numbers. |
| [#198](https://github.com/ayutaz/vokra/pull/198) | Official legal-source HTTP 403 plus version/chapter/applicability and owner/counsel conditions remain. No challenge bypass, new exclusion, accepted-403 rule or unchanged retry is used as a substitute. |

These remain concrete holds, not a claim that all existing approvals are missing
or that completing Scaleway alone would finish every model.

## Announcement and next acceptance order

The [Japanese X draft](cpu-preview-announcement-draft-2026-10-24.md) is for the
already published **v0.4.0-rc.1 Linux CPU preview**: fixed Whisper base ASR and
Silero VAD v5, using the public CLI. It does not announce the current main as a
new release, all-model parity, other-OS real inference or completed Metal.
No new Release or X post has been made in this task.

Continue with #271's own-head CI/integration, then reconcile #297 against the
accepted main and run fresh checks before ordinary merge. Preserve the real
dependency/native/legal holds of the Draft PRs. Record final acceptance without
rewriting earlier failed, cancelled or partial evidence.

No local model download/inference, local workspace/model Cargo execution,
VAST/Scaleway allocation or Hugging Face publication was performed in this
checkpoint. The unrelated dirty local-model guard was preserved. The protected
CosyVoice2 manifest was outside all inputs and changes.
