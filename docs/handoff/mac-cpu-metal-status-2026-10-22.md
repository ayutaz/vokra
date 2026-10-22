# Mac CPU / Metal status and next gates (2026-10-22)

Observed through 14:04 UTC. Code baseline:
`e6661039f509a228c46ae5bd33cde66f05eca951` on main, after normal protected
acceptance of [PR #282](https://github.com/ayutaz/vokra/pull/282).
This dated record supersedes earlier pending-main/PR routing, not historical
measurements, owner decisions, model permissions or numerical bounds.
All five execution requirements and all 194 public model rows remain in scope.

## 14:34 UTC continuation: PR #283 repair pushed, acceptance pending

This continuation supersedes only the earlier #283 routing below. Main is
still `e6661039`; the original 14:04 snapshot is retained as dated evidence.

The original #283
[documentation-links job](https://github.com/ayutaz/vokra/actions/runs/39789253663/job/119860568686)
failed with five HTTP 403 bodies across four distinct API targets. The fourth
was an existing fixed Qwen2-Audio tree link, outside the gate's three literal
targets. An independent authenticated official API read confirms its exact
SHA `0d66378d9e4b1a4d6bdba3969aa87ef777f487cd`, 23 entries and
`truncated=false`. HTTP 403 alone does not prove its precise upstream cause;
no certifi regression or missing source-license grant is inferred.

Luna's bounded repair adds only that exact HTTPS target; a fresh authenticated
HTTP 200 and matching tree identity remain mandatory. Root reviewed both
scripts, required actual-error fixture corrections, and independently passed
25 offline tests. Original scan/report/counters, no-redirect credential-safe
transport, shared limits and the separate original-503 Contents/ref/anchor
route remain unchanged. No exclusion or status waiver was added. A separate
root replay accepts the complete native-log-derived report with four offline
response fixtures; this is not a live HTTP recovery result.

Normal source commit `18130828` passed all five pre-commit gates and was
cherry-picked onto the actual bot head as `ace09f0c`. A non-force push
advanced the existing PR branch; the original certifi-only delta is preserved.
The total PR delta is three files, 99 additions/6 deletions. Its title/body
disclose both changes and the validation limits. Nine new exact-head
workflows exist, including
[Security](https://github.com/ayutaz/vokra/actions/runs/39795709947),
[Quality](https://github.com/ayutaz/vokra/actions/runs/39795709949) and
[Core](https://github.com/ayutaz/vokra/actions/runs/39795710007).
The observed rollup has 12 successful, eight running and 47 queued checks;
it is not terminal acceptance. Inspect all new-head workflows/checks and
required context/application pairs before normal protected merge. Do not
claim live recovery-branch coverage if the ordinary link scan has no errors.

## Accepted source and package evidence

At 13:54 UTC, the exact-baseline check inventory has 79 terminal checks:
72 successful, seven event-gated skips, no failures/cancellations/timeouts.
All 12 listed workflows, including the two dependency updater workflows,
are completed successfully. Original Core run
[39786863522](https://github.com/ayutaz/vokra/actions/runs/39786863522) and
Quality run
[39786863568](https://github.com/ayutaz/vokra/actions/runs/39786863568)
both finish successfully; neither was restarted because of queued jobs or
an observation timeout.

The seven push-event skips are dependency review, full-history gitleaks,
semver, Miri, two dispatch-only Miri diagnostics and address sanitizer.
Their existing event conditions are unchanged. In particular, dependency
review is PR-only: do not describe this push run as 16 successful required
PR context/application pairs. Those pairs must be checked independently
at each proposed PR head before normal protected acceptance.

The original
[cargo-audit job 119852272603](https://github.com/ayutaz/vokra/actions/runs/39786863568/job/119852272603)
uses cargo-audit 0.22.2 and 1,349 advisories, covering the root and all seven
isolated integration lockfiles. The actual per-scope output contains no
vulnerability error; only the existing bincode/encoding maintenance warnings
remain in Piper and server. Job success alone is insufficient because the
integration sweep reports findings as warnings; the original output was
reviewed. The root-lock mutation check also passes.

PR #282's
[acceptance receipt](https://github.com/ayutaz/vokra/pull/282#issuecomment-6341904678)
records a separate actual Linux NumPy/Matplotlib/Pillow package-consumer and
native-payload test. It is not full YuE execution, model parity, Apple
hardware evidence, a blanket native-license approval, or permission for
another model environment. Its owned VAST worker and saved data were
destroyed and independently confirmed absent. This is per-instance cleanup
evidence, not a fresh whole-account inventory.

## PR responsibilities still open

There are nine open PRs in this snapshot. New checks may appear as dependent
jobs are created; a temporary count is not a terminal acceptance result.

| PR | Observed head | Next acceptance boundary |
|---|---|---|
| [#283](https://github.com/ayutaz/vokra/pull/283) | `6d410119` | Certifi-only YuE lock delta, 3 additions/3 deletions. Static distribution/source/RECORD/certificate-data review is complete; fresh exact-head CI remains running. Verify the entire final check/workflow inventory and required pairs before merge. |
| [#284](https://github.com/ayutaz/vokra/pull/284) | `8f65cef4` | Separate XCodec2 lock delta, 18 additions/13 deletions. Only file metadata has been reviewed; dependency and consumer review remain. Do not reuse #282's different consumer scope as blanket acceptance. |
| [#278](https://github.com/ayutaz/vokra/pull/278) | `11e33a10`, Draft | Repair the actual extracted-size producer/consumer contract without increasing a security cap or confusing compressed and extracted sizes. Its old server failure is not repaired merely because newer main audits pass. |
| [#271](https://github.com/ayutaz/vokra/pull/271) | `d777dd71` | Behind main; reconcile action updates and held Candle-file responsibilities, then obtain fresh CI. Old green checks are not acceptance of a rebased head. |
| [#198](https://github.com/ayutaz/vokra/pull/198) | `5614a92c`, Draft | Current official primary legal text/version and owner/legal review remain unresolved. Preserve failed source retrieval and link-check evidence; do not accept HTTP 403 or repeatedly rerun unchanged CI as a fix. |
| [#191](https://github.com/ayutaz/vokra/pull/191) | `63cadf3c`, Draft | Reconcile its Kyutai source/runtime/reference responsibility and held reads before claiming acceptance or retirement. |
| [#169](https://github.com/ayutaz/vokra/pull/169), [#182](https://github.com/ayutaz/vokra/pull/182) | `b3e9d5c1`, `a3afc1ec`, Draft | Preserve the distinct VibeVoice dependency/staged-runtime responsibilities and source-read holds; do not merge overlapping old branches wholesale. |
| [#152](https://github.com/ayutaz/vokra/pull/152) | `a9f54e1a`, Draft | Retain XCodec2 pre-import/evidence duties and held-source boundaries. No closure or retirement is inferred from the new dependency PR. |

The #283 static check verifies the actual two
[PyPI release archive](https://pypi.org/project/certifi/2026.10.19/) digests, all 12
wheel RECORD entries, identical packaged/pinned-source API and license bytes,
absence of native archive members, and parsing of all 120 CA certificates
through the Python standard library. No certifi code was imported or installed.
This is not a live HTTPS/installed-consumer test or a new MPL-2.0 approval.
GitHub reports the upstream annotated tag as unverified (`bad_cert`); the
separate PyPI publisher metadata must not be represented as verification of
that tag. Both temporary archive inputs were removed after the small receipts
were saved. Full YuE/model execution and publication remain blocked separately.

## AudioLDM2: dependency API work before a runnable lock

Read-only review of the existing Base/Large inspector, reference entry point
and project confirms an empty dependency list and absent dedicated `uv.lock`.
The current entry points correctly stop before model execution. No dependency
versions were selected, source checkout installed, or model acquired.

The fixed Diffusers source is
`29f15673ed5c14e4843d7c837890910207f72129` (v0.21.0). Its
[utils initializer](https://raw.githubusercontent.com/huggingface/diffusers/29f15673ed5c14e4843d7c837890910207f72129/src/diffusers/utils/__init__.py)
imports dynamic-module utilities, whose
[module-level import](https://raw.githubusercontent.com/huggingface/diffusers/29f15673ed5c14e4843d7c837890910207f72129/src/diffusers/utils/dynamic_modules_utils.py)
requires `cached_download`. The official
[Hub v0.26.0 release](https://github.com/huggingface/huggingface_hub/releases/tag/v0.26.0)
removes that API. The inspected
[Hub 1.27.0 initializer](https://raw.githubusercontent.com/huggingface/huggingface_hub/a7d85da5de06e8b85e94999e7cf985f6e0f9991b/src/huggingface_hub/__init__.py)
at pinned commit `a7d85da5de06e8b85e94999e7cf985f6e0f9991b` does not export
it either.

Official [Transformers 5.10.4 metadata](https://pypi.org/pypi/transformers/5.10.4/json)
requires Hub `>=1.5.0,<2.0`. Thus simply combining the unchanged old
Diffusers source with this modern candidate dependency set does not establish
an import-compatible reference environment. This is a static API finding,
not an executed import test, a proof about every possible version combination,
or permission to patch the independent reference silently.

Do not solve the mismatch by choosing a vulnerable Transformers release:
[GHSA-xrqw-3rrv-vx5w](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w)
affects versions below 5.10.0. A new official source route or explicitly
reviewed compatibility design, exact dependency/native/license graph and
model-free API proof must precede any real-weight reference. No security
ignore, dependency downgrade, source-pin change or approval is made here.

The guessed GitHub tag `v5.10.4` returned 404, while exact PyPI version
metadata exists; this does not prove that the distribution is absent.
The fixed model-card reads did not produce accepted evidence: the browser
fetch failed, and the Base README-only curl request was refused by the local
model guard. No alternative path was used after that refusal. Therefore no
claim is made that the inspector's expected model-card license is correct
or incorrect. Keep historical model/source license decisions unchanged.

### 14:34 UTC candidate official route, not an adopted source change

The current official
[Diffusers v0.41.0 release](https://github.com/huggingface/diffusers/releases/tag/v0.41.0)
resolves to `086bf9578c0f4acbc66e48cd1e7cc26befd9e10f`; GitHub reports
its annotated tag unsigned. Its
[dynamic-module utilities](https://github.com/huggingface/diffusers/blob/086bf9578c0f4acbc66e48cd1e7cc26befd9e10f/src/diffusers/utils/dynamic_modules_utils.py)
no longer import the old `cached_download` API, and the official
[AudioLDM2 tests](https://github.com/huggingface/diffusers/blob/086bf9578c0f4acbc66e48cd1e7cc26befd9e10f/tests/pipelines/audioldm2/test_audioldm2.py)
still name Base and Large. This identifies a candidate independent upstream
route, not proof of compatibility with the existing authenticated bundle.

Actual [Diffusers 0.41.0 metadata](https://pypi.org/pypi/diffusers/0.41.0/json)
requires Hub `>=1.32.0,<2.0` and safetensors `>=0.8.0`; therefore the earlier
Hub 1.27.0 candidate cannot simply be reused. Current official
[Transformers 5.20.2 metadata](https://pypi.org/pypi/transformers/5.20.2/json)
requires Hub `>=1.31.0,<3.0`, giving an overlapping interval, not a proved
closure. Its fixed
[GPT2 source](https://github.com/huggingface/transformers/blob/18329f1065382e54db698e242cf98bd3f3e5a766/src/transformers/models/gpt2/modeling_gpt2.py)
and the new Diffusers tests use a generation-capable GPT2LMHeadModel route.
Exact bundle class/loading/hidden-state semantics, dependencies, native
licenses, security and model-free API behavior still need review. Do not add
an invented LM head or silently patch the oracle.

No model/source pin, lock, loader, owner decision or execution gate changed.
Only official source and package JSON metadata were read; no package archive
was acquired or installed, and no model was instantiated. A narrow local
AudioLDM2 source search was refused by the model guard; no alternate reader,
agent or remote copy was used. The model-card hold also remains unresolved.

## Full remaining order and completion boundary

The retained public metadata classification remains 194 repositories,
136 code/artifact-full and 58 non-full. These are not real-weight CPU or
Apple pass counts. The original 63-row owner packet is a separate frozen
hash-bound scope; it must not be replaced with the live 58-row denominator.

Prior Candle/Vibe source-read holds and the new bounded MOSS Nano review
remain unresolved. MOSS's ordinary and normal escalated read-only search
were both refused; no alternate read was used and no source-build dependency
defect was inferred from the refusal. Its bounded read request is unanswered.

1. Finish PR #278's actual contract repair and reconcile #271 plus all Draft
   duties, with reviewed source, fresh complete CI and normal acceptance.
2. Close each of the 194 rows' exact source, artifact, dependency/license,
   converter/binder/native/CLI, independent-reference and genuine owner
   prerequisites. Preserve the named real BF16 checkpoint, HiFTNet and four
   BigVGAN obligations. Source-read holds are not approvals or waived gates.
3. Run only the authorized exact-scope no-upload VAST real-weight conversion,
   independent pinned reference and unchanged-bound CPU parity. Recover
   small original evidence, then destroy owned workers and saved data.
4. Fix one final reviewed clean HEAD and regenerate authenticated transfer
   packets for every still-required row. Earlier packets and source-only
   successes do not establish equivalence to that final HEAD.
5. Provision Scaleway last, after the non-Apple exit gate. Record Apple
   CPU/reference, Metal/reference, Metal/CPU and explicit no-fallback results;
   fix failures, normally accept the result PR and remove unneeded resources.

The [remaining-task ledger](mac-pre-scaleway-remaining-tasks-2026-09-05.md),
[execution plan](mac-cpu-metal-execution-plan-2026-09-07.md) and
[frozen owner packet](mac-cpu-metal-owner-disposition-packet-2026-09-07.md)
retain their full scope. No model row, owner/license decision, upload,
withdrawal, Apple result or release-completion status is promoted by this
documentation change. No local model acquisition/execution, heavy Cargo,
new paid worker or public-artifact operation was performed for this readback.
