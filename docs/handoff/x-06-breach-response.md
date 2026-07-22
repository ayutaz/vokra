# X-06 threshold-breach response runbook (NFR-MT-07)

X-06's exit criteria require that a nightly threshold breach **drives a
revert or block** of the causing PR. The workflows provide the *mechanism*
(they go red and upload evidence); this runbook is the *procedure*. Authoring
is CC (milestones.md §10: "revert/block operation = CC"); **execution is the
owner** (repo write / branch protection — red-line 6).

## Which legs fail vs only annotate

Not every leg reddens the build. Know which is which before reacting.

### Fail-on-breach — these drive revert/block

| leg | workflow | breach condition |
|-----|----------|------------------|
| mel-frontend RTF | `ci.yml` bench-regression (**required, PR-blocking**) | > baseline × 1.05 (NFR-PF-13) |
| Whisper WER | `nightly-asr-wer.yml` | WER > per-slice calibrated threshold (NFR-QL-04) |
| Tier 1/2 device RTF | `nightly-tier2-device.yml` | RTF > NFR-PF-09/10 (once a device runner + staged GGUF exist) |
| CUDA large-v3 RTF | `gpu-cuda-rtf.yml` | > baseline × 1.05 or > 0.10 hard ceiling (once the self-hosted runner exists) |
| Silero VAD RTF | `ci.yml` bench-regression | **only if** the owner set `VOKRA_SILERO_GATE_REQUIRED=true` |

### Advisory record-only — these annotate, they do NOT fail

| leg | workflow | why advisory |
|-----|----------|--------------|
| Silero VAD RTF | `ci.yml` bench-regression | default posture (open question #1) — shared-runner variance not yet shown to hold 5% |
| Whisper base RTF companion | `nightly-asr-wer.yml` (X-06-T11) | full-model RTF on a shared runner is noisier than mel (open question #6) |
| piper TTS RTF companion | `nightly-asr-wer.yml` (X-06-T12) | same, and clean-skips until a piper GGUF is staged (open question #7) |

### Clean-skip — no verdict at all (never a pass)

Placeholder baselines, absent self-hosted runners, and absent staged GGUFs
produce a printed skip notice and green. A skip is **not** a breach and **not**
a pass; it means "not measured". Do not react to a skip.

## Procedure on a red fail-on-breach nightly

1. **Confirm it is a real breach, not infra.** Open the run; read the step
   summary. A checksum mismatch (corpus changed), a download failure, or a
   CLI-contract error is infra/data, not a code regression — those hard-fail
   by design so they cannot masquerade as a quality regression. Fix the infra;
   do not revert a PR for it.
2. **Pull the evidence.** Every fail-on-breach leg uploads an artifact even on
   breach (`nightly-asr-wer` → `asr-wer-report`; device → step summary; CUDA →
   `cuda-rtf-runs`). That artifact is the basis of the decision.
3. **Find the PR range.** Identify the last green run of the same workflow and
   the first red one; the causing change is in `git log <last-green-sha>..<red-sha>`.
   For a nightly this is usually one day of merges.
   ```
   gh run list --workflow <name> --branch main --limit 20
   git log --oneline <last-green-sha>..<red-sha>
   ```
4. **Revert or block** (owner):
   - **Revert** the offending PR (`git revert -m 1 <merge-sha>` → PR → merge)
     when the cause is a single identifiable change.
   - **Block** (tighten branch protection / mark the leg required) when the
     regression is systemic or the cause is unclear and further merges would
     compound it.
5. **Re-run** the workflow after the fix and confirm green.
6. **Record** the breach + response in the **X-02 quarterly Go/No-go review**:
   date, workflow, metric, breach value vs threshold, causing PR, action taken
   (revert/block), outcome. This is the audit trail NFR-MT-07 requires.

## advisory → required promotion

An advisory leg becomes a hard gate only when its measurement is shown to be
stable enough that the 5% (or absolute) threshold will not false-fail:

- **Silero VAD** (`VOKRA_SILERO_GATE_REQUIRED=true`): promote after several
  weeks of ubuntu-latest runs where the seeded baseline holds within 5%.
- **whisper/piper nightly RTF**: promote (flip the companion step from
  `::warning::` to `exit 1`) after the shared-runner variance is characterised
  (open question #6). Until then they are records, and the WER breach is the
  actionable gate on that workflow.
- **CUDA / Tier 2 device**: promotion is gated on the self-hosted runner
  standing up first (owner, §11) and then an owner sign-off, exactly as
  `gpu-cuda-rtf.yml` documents.

Promotion is always an owner decision; CC provides the mechanism (a repo
variable or a one-line step change) and the criteria above.
