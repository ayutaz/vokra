# Codex operations handoff (2026-08-18)

## Historical baseline at handoff (2026-08-18; superseded)

- Repository: `ayutaz/vokra`
- Current branch: `main`
- Baseline commit: `6d64fdf04c3ffd9f3d99ccda55685b6b8b3f3174`
  (`fix(claude): enforce uv compatibility (#37)`)
- Local status at handoff start: clean and synchronized with `origin/main`.
- Remote branch set after retirement: `origin/main` only. A later read-only
  `git ls-remote` retry hit transient DNS failure, but the deletion push had
  already succeeded and the fetched remote-tracking view contained only main.
- PR #37 completed with all 56 reported checks green before merge.

This file records the decisions carried from the Claude Code session into
Codex. Reusable policy belongs in `AGENTS.md`, `.agents/skills/`, and hooks;
this file keeps the dated evidence and branch history.

## Current-state supersession notice (2026-08-30)

This handoff is history-only. The baseline branch name was
`feat/mac-cpu-metal-full-coverage-2026-08-28`. The 2026-08-30 documentation
refresh cross-checked the pre-documentation implementation/code baseline at
`c64b7b7237b70c5dc70ffd60394af325016d9a8d`; workspace `0.2.0`, with C ABI
57 functions / 15 typedefs and M5 49 checked / 33 unchecked. The GitHub `main`
reference remains `41ce9ffdd4b0959497f55afa5016822f77a8a7b6`. Current model,
VAST and Scaleway status is authoritative only in
`docs/handoff/mac-cpu-metal-full-coverage-2026-08-28.md` and
`docs/m5-owner-verification-checklist.md`; the dated baseline and all later
history in this file are not current-state claims.

## Mainline history used for the reconciliation

| PR | Main commit | Result relevant to this handoff |
|---:|---|---|
| #24 | `02664f6` | M5 gap wave, coverage audit, KWS/denoise/MOS/AEC binders and CI resolution |
| #27 | `0937ef8` | Voxtral/SBV2 fixes and coverage-audit wiring |
| #28 | `40558f5` | SBV2 Blocker 2b/2c/3/5 verification wave |
| #30 | `93b484b` | C ABI GPU backend selection + speaker embedding |
| #29 | `8e048d8` | Audio-wide coverage, 14 audit rounds, seven gates, documentation truth sweep |
| #31 | `c9b74a8` | Production `actions/cache` dependency update |
| #32 | `31cd78b` | Primary workflow migration from Claude Code to Codex |
| #33 | `883255a` | VAST parity completion, uv workflows, M5 reconciliation, large-model safety |
| #36 | `ce84832` | Four-file SBV2 ZH real parity |
| #35 | `09f3585` | SBV2 parity protobuf lock update |
| #34 | `8320de6` | SBV2 parity torch lock update |
| #37 | `6d64fdf` | Claude compatibility brought to the same uv-only command policy |

The ordering above follows the work relationship; first-parent main places
#36 before the two dependency updates and #37 last. Commit ids are the source
of truth when chronology matters.

## Old remote branch disposition

The retired branch `feat/audit-followup-cc-wave1-2026-08-14` was not merged as
a branch. Its long-lived history had 123 branch commits and overlapped later
squash merges:

- the first 121 commits were represented by the audit work merged in PR #29;
- the Codex migration delta matched work already merged in PR #32;
- a direct merge still produced 17 conflicted files / 80 conflict hunks, so it
  was not a safe way to recover the two genuine compatibility deltas;
- seven Claude compatibility files were selected, reviewed, merged through PR
  #37, and verified by CI instead.

After PR #37 merged, the owner approved deletion of the stale remote branch.
The remote ref was deleted. The local branch remains at `8811f4d` as a
recoverable backup and must not be pushed or merged wholesale. Other local
branches whose upstream is `gone` are local residues, not active remote work.

Direction from here:

1. start new work from current `main`;
2. inspect an old branch by patch/commit content, not by branch age or name;
3. salvage only a unique, still-correct delta into a fresh branch/PR;
4. never merge a long-lived branch whose work has already arrived through
   squash merges.

## Pre-push deletion incident and remediation

The first deletion push supplied no commit diff. The old pre-push classifier
treated that empty diff as a deep path and launched workspace clippy/tests,
including `vokra-models`, on the 16 GB M1 Mac. The process group was terminated
and a process audit confirmed that no matching hook, Cargo, rustc, or push
process remained. The deletion was then completed with
`VOKRA_SKIP_HOOKS=1`.

The hook now parses Git's pre-push stdin. Only one or more well-formed updates
whose local SHA is all zeroes are classified as deletion-only. Such a push
still runs the compliance scanner regression, then skips Cargo. A normal,
mixed, empty, or malformed update cannot take that path. The maintainer Mac
also refuses every other deep pre-push path before Cargo; code is verified on
VAST and pushed with the bypass only after its green result is recorded.
`scripts/test-pre-push-fastpath.sh` pins all of this in 50 classifier and
production-hook integration cases.

## Standing execution policy

- Run every Python entry point through uv. Use a repository project where one
  exists; otherwise use `uv run --no-project --python 3.12 python ...`. Do not
  use bare `python`, `python3`, pip, conda, or requirements-based setup.
- Run conversion, validation, or publication involving model artefacts whose
  aggregate size is at least 2 GB on VAST. Count all shards, not only the
  largest file.
- Run workspace-wide Cargo and every `-p vokra-models` Cargo operation that
  compiles, tests, checks, runs, documents, or audits on VAST, not on the
  maintainer Mac. Cheap fmt/metadata/tree inspection remains local.
- The narrow local large-file exception is provenance-only
  `restamp_provenance`, which does not touch tensor data and has an 8.7 GB / 6.4
  MB peak-footprint precedent.
- Destroy disposable VAST instances after evidence is saved. A retained volume
  must be stopped and named in the relevant handoff.
- Publishing is a separate irreversible action. Use only
  `scripts/publish/publish-one.sh`; dry-run first, and add `--push` only after
  explicit owner authorization for the exact repo/artifact.

## VAST and parity state

PR #36 closed the four-file SBV2 ZH numerical leg on disposable VAST instance
`47977839`:

- all four regenerated GGUF hashes matched committed sidecars;
- the independent upstream reference emitted the expected ZH tensors;
- the Rust consumer passed `1 / 0 / 0` in 1026.70 seconds;
- unchanged bounds held: ZH BERT max `1.907349e-5`, waveform max
  `1.031446e-1`, mel-loss RMS `1.820711e-1`;
- no HF credential or upload was used, and the instance was destroyed.

The larger Voxtral evidence on stopped instance `47955178` remains a separate
owner action. Its corrected 48.5 GB artifact passed conversion, dry-run, and
real parity, but the live HF repo still has stale provenance. Resume only after
explicit upload authorization, publish through `publish-one.sh`, live-verify,
then destroy the retained instance/volume.

No Hugging Face upload was performed in the session summarized here.

## M5 remaining-work interpretation

`docs/m5-owner-verification-checklist.md` currently contains 42 checked and 36
unchecked boxes. The old 94 unchecked count was historical and never meant 94
missing implementations. The 36-box ledger mixes distinct done-conditions:

- NPU/CoreML/QNN real-hardware capture and C ABI GO/NO-GO inputs;
- Cortex-M55/FVP, console SDK/NDA, legal/ADR, and GA/branch-protection actions;
- real-checkpoint reference output for the remaining parity families;
- deliberate implementation follow-ups such as BF16 native compute and full
  vocoder GPU paths;
- voice-conversion destination/publication policy;
- explicitly authorized HF publication/live verification, including corrected
  Voxtral-Small-24B;
- optional GitHub Pages deployment.

Read and close each literal box independently. Also follow the prose-only GA
gates in the checklist's §0 live index; the 36 literal boxes are not an
exhaustive count of M5 work. Do not infer implementation status from the total.

## Credential hygiene

A VAST API credential appeared in prior session output. Its value is not
stored here and must never be copied into tracked files, command arguments, or
logs. Treat any credential exposed in output as compromised and rotate it
before reuse. HF/VAST credentials belong in ephemeral environment variables;
stopping or destroying an instance reduces persistence but does not undo a
secret already printed to a terminal or log.

## Files synchronized by this refresh

- Codex project policy: `AGENTS.md`
- Contributor workflow: `CONTRIBUTING.md`
- Reusable Codex skills: all six under `.agents/skills/`
- Claude compatibility copies: the matching six under `.claude/skills/`
- Codex/Claude memory guards and Git pre-push classifier/tests
- M5 current-state header and `CHANGELOG.md`
- VAST publication runbook and this handoff
