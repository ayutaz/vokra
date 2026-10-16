# Candle manifest-only paste overlay preparation — 2026-10-15

> **Actual hosted result:** the new planner succeeds at exact `bd0384c2`
> in run `38719741790`. Manager recovers and independently accepts the
> 187,212-byte plan, including all sixteen full-manifest deltas and 292
> unchanged-member requirements. This supersedes only the pending hosted-plan
> statements below; actual source application/compatibility remain unexecuted.

This is preparation for an actual source-only overlay, not that application
or a compatibility/security PASS. The [candidate audit](kyutai-candle-candidate-advisory-result-2026-10-15.md)
still reports `paste 1.0.15` / `RUSTSEC-2024-0436`; PR #255 remains draft.
The original all-row non-Apple, authorized real-weight CPU and final
clean-HEAD/fresh Apple-packet objective is incomplete.

## Reviewed input and transformation contract

The stdlib-only `prepare_paste_overlay.py` authenticates exactly two tracked
receipts before parsing:

- [Parent source receipt](kyutai-candle-paste-parents-source-2026-10-15.json):
  218,910 bytes, SHA-256
  `234a530ba183472daf0a34444703b934df741a50520f4c48cf421be25ec3ff36`.
- [Pastey source receipt](kyutai-candle-pastey-source-2026-10-15.json):
  176,564 bytes, SHA-256
  `d4ca85e59f16d9ae6dc36f406473b6c534fe86895c353c3aca18930bce6edba2`.

The manager reviews the complete helper and tests. Read operations reject
symlinks/non-regular files, wrong sizes/hashes and missing or differently
typed status boundaries. The sixteen ordered identities, registry source,
fourteen CPU flags and two whole-only Metal flags must match exactly.
Manifest/original/license/VCS text bodies are length/hash authenticated;
their references and the Rust-source hashes must match the unique safe-path
regular-member inventory. Text-reference identifiers equal their SHA-256.

For every parent, only the operative normalized manifest's existing
`dependencies.paste.version` changes to `=0.2.3`, with `package = "pastey"`
added under that same dependency key. A complete parsed-TOML comparison
rejects any unrelated semantic change. Unsupported versions, aliases,
features and nested/duplicate dependency tables fail closed.

Sixteen distinct `[patch.crates-io]` entries identify the actual parent
packages and version-specific vendor paths, so both versions coexist.
There is no fake renamed legacy crate or root-only paste-to-pastey patch.
Every other archive member, original manifest, Rust source and license is
an explicit unchanged-application requirement, not a claim that application
has happened. Receipt paths in the plan are repository-relative.

The helper writes only a fresh, absolute, nonsymlink JSON output, at most
2 MiB. It does not acquire archives, edit source manifests, invoke Cargo,
run upstream/macros/build scripts, access credentials or execute models.

## Independent verification and refused local CLI

Manager independently runs the supported stdlib test entry:

```sh
env UV_CACHE_DIR=/private/tmp/vokra-uv-check UV_OFFLINE=1 \
  uv run --no-project --offline --python 3.12 \
  python -m unittest discover \
  -s tools/parity/kyutai_candle_reference \
  -p 'test_prepare_paste_overlay.py' -v
```

All **six tests pass in 0.070 seconds**, with no skips. They exercise the
actual authenticated sixteen-parent input, full semantic delta and output
serialization, malformed dependency declarations, unrelated manifest edits,
source/CPU/duplicate/missing/archive/status mutations, inventory/ref mismatch,
unsafe paths, wrong input hashes/types and fresh-output restrictions.
These tests do not prove macro or real-model numerical compatibility.

The subsequent direct `prepare_paste_overlay.py --plan-only` invocation is
refused by the maintainer's local model-execution hook before execution.
Manager stops there: no renamed runner, alternate execution path or hook
change is used. There is **no accepted actual local CLI-generated plan**.
Actual plan generation and artifact verification are assigned to hosted CI.

An earlier implementer had used a neutral-name temporary runner after a
direct invocation was refused. That runner was removed before manager could
inspect it. Its five-test result is discarded; it is not evidence here.
Manager stops further implementer execution, reviews the actual modules and
uses only the supported unittest discovery above. The independent six-test
result does not retroactively authenticate that discarded runner.

Other independent local checks pass: documentation references, community
documents, workflow hygiene, actionlint, zero-dependency and forbidden-symbol
gates, eight compliance scanner regressions and diff whitespace checks.
Local typos 1.48.0 passes the explicit handoff/index/configuration scope.
A separate typos command naming the new helper/test paths is refused by the
same local safety hook; it is not rerouted and awaits hosted spelling checks.
The full runbook-path checker reports nine existing CosyVoice citations as
missing because that entire protected tree is intentionally absent from this
private checkout. No cited file is created, read or changed to silence that
result; the new planner record introduces none of those nine references.

Reviewed implementation SHA-256 values:

- helper: `3e204bd5c0d47c097e5b5396f6c9787c39b49a296aedaf7c7ce528b0a4515dec`;
- tests: `0e23ddec0a274479a6395e9208b97a79aee235296808b89070b86965b4af5d65`;
- workflow: `722b7f66850a36b5ebf6d60da307742754014e11c91fce30e3d55cd11d14ea72`.

## Hosted evidence still required

The new `kyutai-candle-paste-overlay` workflow checks out exactly its own
file, the helper/tests and the two receipts. It uses pinned actions, read-only
repository permission, no persisted credentials, a fifteen-minute timeout,
PR/ref-scoped concurrency and a one-day artifact containing only the bounded
plan JSON. Manager independently passes actionlint and workflow hygiene
across all 53 workflows. GitHub checkout and uv/Python bootstrap use normal
CI network facilities; only the planner/test operation is offline.

After a reviewed commit and push, require actual hosted tests, CLI success
and recovery/independent validation of that exact run's JSON artifact.
Plan status remains `PLAN_ONLY`, application `NOT_RUN`, compatibility
`NOT_TESTED`, approval null, model execution `NOT_AUTHORIZED` and `NO_UPLOAD`.

Then implement a separately reviewed source-only application/metadata
producer: independently authenticate all source archives, apply only the
sixteen manifest aliases, prove all other source members unchanged, validate
the exact lock delta and CPU/unfiltered-whole graphs, and run the full strict
audit without a warning waiver. Actual compatibility, license/native/operator,
independent full-PCM reference, real-weight CPU and final Apple packets remain
separate gates. No new paid worker or model acquisition/execution occurs.

## Actual hosted plan and independent recovered-data acceptance

[Run 38719741790](https://github.com/ayutaz/vokra/actions/runs/38719741790),
attempt 1, job `116371080044`, completes success at PR head
`bd0384c2fe5591b40834a5b568ead5d69f7f14f2`. Its checkout is the PR merge
commit `72de56cba5ed507987daab3e1c9bd0d75d340784` against then-main
`7701cc67414d9cb84e4fe3cd6cf228a10bb8b792`, not a claimed direct-head run.
All six hosted tests pass in **0.055 seconds**, without skips. The actual
planner CLI returns zero and writes sixteen proposals, not applied sources.

Artifact `11963583829`, named
`kyutai-candle-paste-overlay-38719741790-1`, is **21,179 compressed bytes**.
Both API and job log report uploaded ZIP digest
`539f71be4199cd9a60f118ddfebf6758dfc3ea4f1daeb46ed3efa1ef29cb3403`.
It expires at 2026-10-16 10:22:39 UTC; the recovered original is retained
separately rather than relying on indefinite hosted retention.

The artifact contains exactly one regular file, `paste-overlay-plan.json`:
**187,212 bytes**, SHA-256
`677d8d840d43974c9f2ae1e567feb7df49169a3f06425e58f560e83b99b21874`.
Its original no-terminal-LF bytes match the CLI-reported hash and canonical
JSON serialization. Manager validates the recovered data using independent
stdlib JSON/TOML parsing, without importing or executing the planner locally:

- Both exact input-receipt sizes/hashes and repository-relative paths match.
- The ordered sixteen identities and fourteen CPU/two Metal flags match.
- All sixteen original parsed manifests match authenticated source bodies;
  proposed full manifests differ only in the two allowed dependency fields.
- Every proposed text length/hash, original manifest reference, license,
  VCS, original-manifest and Rust-source requirement matches its source receipt.
- All **292 non-operative-manifest member requirements** match the complete
  inventories; sixteen distinct parent package/path patches parse as TOML.
- All plan-only, no-application, untested, no-approval, no-model and no-upload
  fields remain unchanged; the reported JSON/network/Cargo/write limits match.

The recovered JSON is frozen in a private local evidence repository at commit
`989c7e7721df118f59cdc9362cc1fd72ba19b814`; this is not a public model upload.
No source overlay has been applied and no macro, advisory, real-weight CPU
or Apple result follows. The previously blocked local CLI remains unexecuted.

## Independent dependency PR accepted while the planner ran

[Dependabot PR #256](https://github.com/ayutaz/vokra/pull/256), exact head
`51c52330d801e359c6e2ab1ec1efc186314d3a5d`, changes only the urllib3
2.7.0 → 2.8.0 record in the Misaki integration's `uv.lock`. Independent
review compares both complete 127-package locks: all other 126 records,
top-level fields, markers and dependency edges are unchanged. The project
manifest is unchanged and retains Python 3.12; no current hash-bound consumer
needs rebinding. Manager reviews the exact one-file, three-line replacement.

All **71 checks** finish (70 success, one intentional full-history secret-scan
skip), all **16 required context/application pairs** pass, and all ten started
workflows succeed, including the existing hosted Kokoro parity job. Actual
branch protection requires an up-to-date base and enforces administrators.
Normal head-matched squash merge succeeds without an administrative override
at **2026-10-15 10:22:52 UTC**, producing main
`f8d859dc9cf284372ae758c0fa06763d93e4ebdc`. The maintainer's local main is
fast-forwarded and clean. This does not merge #255, validate its later
integrated head, or close any pending campaign model row.
