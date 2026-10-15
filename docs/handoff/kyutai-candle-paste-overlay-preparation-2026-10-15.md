# Candle manifest-only paste overlay preparation — 2026-10-15

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
