# Readback/documentation branch closeout (2026-10-04)

## Scope and acceptance

The user requested this sequence: integrate current main, review the net diff,
verify the integrated HEAD, push/open and validate the PR, then merge and
update local main with a follow-up handoff. This record tracks that branch
closeout, not completion of the public model catalog or a GA release.

Branch: `codex/kyutai-failed-verification-readback-20260930`.
The pre-integration HEAD was `3f2c5ca6ac3321bbdda73163df3c148c1725fea0`.
Main `97447185361a37af64c1b30fe87e8e2618d96e20` was integrated with merge
commit `403e0e0ebe537ac91c49abd44b1d79beb28082af`. Two documentation conflicts
were resolved; upstream implementation changes were retained. The integrated
tree was clean after the merge commit.

Before adding this handoff, the net diff against that main contained 104
Markdown documents and these three non-document files only:

- `.codex/hooks/guard-local-models.sh`
- `tools/parity/kyutai_stt_pytorch_pcm_source_audit.py`
- `tools/parity/kyutai_stt_pytorch_pcm_source_receipt.json`

Rust source, Cargo configuration, and runtime behavior have no branch-unique
diff against the integrated main. Staged Rust files observed during the merge
were incoming main changes, not new branch-owned implementations.

## Verification already observed

The resolved integration tree passed the following local, model-free checks:

- Kyutai PCM source auditor `--self-test`: five synthetic tests passed.
- Local-model guard `--self-test`; Codex hook checker and its self-test.
- Documentation example checker and self-test: 140 blocks across 26 documents;
  30 deferred blocks were explicitly **not verified**.
- Documentation references and self-test; runbook path citations; community
  documentation; parity-sidecar citations.
- Owner-checklist drift, platform support, ABI changelog, workflow hygiene,
  and staged diff whitespace checks, reported by the integration implementer.
- Normal merge-commit pre-commit checks: formatting, forbidden symbols,
  first-party-only lockfile, 190 fixture EOL pins, and pipefail lint.

These checks are not upstream execution, actual-weight parity, Apple hardware
verification, legal approval, or publication evidence. No local model
download/import/load/forward or workspace-scale Cargo verification was run.
The authenticated source receipt deliberately retains `execution: NOT_RUN`.

Independent review identified two input-contract weaknesses. The implementer
fixed model/reference execution hidden inside command, backtick, and process
substitution before read-only/static/help exceptions, with regression cases
that do not execute their payloads. The source auditor now requires the exact
source-only receipt scope and rejects contradictory execution/publication
claims. Root reviewed both diffs and re-ran the guard and hook contracts plus
the expanded source-audit suite (six synthetic tests), all passing.

The legacy, unused archive-controller exception retains its exact path/hash
check. This is not an atomic hash-and-execute primitive or an OS security
sandbox against a same-user writer; this closeout does not run that controller.
Its hash check must not be represented as a broader TOCTOU guarantee.

## Remaining merge gates

1. Commit the root-accepted contract fixes and this handoff with the normal
   pre-commit gates, then freeze a clean reviewed HEAD.
2. Record corresponding remote verification
   before a code push. The maintainer Mac's deep pre-push Cargo path remains
   prohibited; do not weaken the classifier or substitute an unverified hook
   bypass. Limit this branch's remote contract tests to its actual net scope;
   do not claim workspace or model parity from script self-tests.
3. Open a PR with the final net scope and limitations. Require checks from the
   latest head and the current main, not old checkpoints or another PR.
4. Merge only after required CI and review acceptance. The read-only main
   protection snapshot has strict checks enabled and 16 required contexts;
   re-read the protection when making the merge decision.
5. Verify the PR is merged, fast-forward local main to the resulting remote
   main, verify a clean worktree, and record the merge/hash and follow-up
   boundaries in the PR closeout or a subsequent dated record.

At this record's creation, this branch had no PR. PR checks, final-head remote
verification and final merge are not represented as completed here. Later PR
events/logs supply the closeout evidence rather than rewriting these earlier
observations as historical passes.

## Separate follow-up responsibilities

The [documentation refresh audit](documentation-refresh-2026-10-04.md) records
the reviewed documentation corpus and dated external snapshots. Continue model
work using the [execution plan](mac-cpu-metal-execution-plan-2026-09-07.md),
[remaining-task ledger](mac-pre-scaleway-remaining-tasks-2026-09-05.md), and
[owner disposition packet](mac-cpu-metal-owner-disposition-packet-2026-09-07.md).
Their live metadata code/artifact classification must not be presented as an
Apple completion verdict.

Separate work includes unresolved source/package/license/owner dispositions,
native model routes, independent actual-weight CPU parity on VAST, final
Apple CPU/Metal/reference and no-fallback checks on Scaleway, repository-scoped
publication decisions, security remediation, and release/GA conditions.
None is closed by this branch's documentation update or model-free CI.
PR #152 is a separate candidate, not this branch's PR.

The protected CosyVoice2 LLM owner manifest is excluded from inspection,
changes and commits. Credentials and local `.env` files must not enter Git,
bundles, remote workers, or published evidence. Any newly allocated VAST
worker for this closeout must be destroyed after small-evidence recovery and
its removal verified individually; unrelated workers are outside this scope.
