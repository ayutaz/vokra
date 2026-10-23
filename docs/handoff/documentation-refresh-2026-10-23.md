# Documentation refresh and coverage audit — 2026-10-23

## Scope and completion boundary

The goal is to make the entire repository documentation set current without
rewriting historical evidence. Review baseline is local/GitHub main
`151d94a8613d5bd4acb1671a461f59742c37b8fc`, not a newly accepted documentation
commit. Start inventory: **400 tracked Markdown/MDX/RST/AsciiDoc files**.
The six per-scope inventories below plus the 17 root-owned rows in this
record account for that complete tracked set (deduplicated). Five additional
ignored local planning/reference documents are classified in those inventories;
they are not force-added or made publication authority.

“Covered” distinguishes current-guide/source review from historical
path/heading/status classification and static citation checks. It does **not**
mean every dated record was reread line by line or every command executed.
The per-file evidence and limits are recorded explicitly. Unchanged documents
need no date-only churn. Historical measurements, license grants, signed
scopes, source holds and model status are not promoted.

Current entry points are [English status](../current-status.md),
[日本語の現況](../current-status.ja.md) and [the guide index](../README.md).
The old index's complete checkpoint body is preserved in the
[relocated historical index](documentation-index-checkpoints-through-2026-10-22.md);
only relative Markdown destinations were rebased for its new directory.

## Per-scope inventories

| Inventory | Rows | Review boundary |
|---|---:|---|
| [Public guides](documentation-public-guides-coverage-2026-10-23.md) | 47 | 42 tracked and five ignored local guides/plans; English/Japanese and source-surface checks |
| [Bindings/integrations/examples](documentation-integrations-coverage-2026-10-23.md) | 69 | Stable-vs-preview assets, public API/source contracts and limits; no engine/device run |
| [Tooling/workflows](documentation-tooling-coverage-2026-10-23.md) | 62 | Sidecar/runbook/workflow facts, without held-source or model execution |
| [Design/governance/bench records](documentation-design-governance-coverage-2026-10-23.md) | 59 | Active protocol routing and dated evidence classification; no new benchmark/signature |
| [Historical handoffs](documentation-historical-coverage-2026-10-23.md) | 136 | Path/heading/status and static citation coverage; eight active entry pointers refreshed |
| [Agent guidance](documentation-agent-guidance-coverage-2026-10-23.md) | 17 | AGENTS and 16 skill files; eight twins match, no execution/publication authority changed |

These inventories overlap on two paths; their sum is not the unique tracked
file count. The root rows below close the remaining 17 tracked paths.
The M5 owner checklist also received a current pointer without changing any
checkbox. License texts, fixture text, API symbol snapshots and other generated
non-prose surfaces are evidence, not prose to rewrite.

### Root-owned per-file disposition

| File | Disposition | Evidence / limits |
|---|---|---|
| `CHANGELOG.md` | Updated | Cargo version, post-preview merged history and published RC/run evidence; historical stable entries retained. |
| `docs/README.md` | Updated | Current guide index; original checkpoint body archived with relative-link rebasing, not destroyed. |
| `docs/abi-changelog.md` | Updated | Current source line/pointer; existing static ABI gate, no header/anchor/entry rotation. |
| `docs/handoff/mac-cpu-metal-execution-plan-2026-09-07.md` | Pointer updated | Current remaining order; all historical steps/results/permissions unchanged. |
| `docs/handoff/mac-cpu-metal-full-coverage-2026-08-28.md` | Pointer updated | Current metadata classification distinguished from dated 131/63 evidence. |
| `docs/handoff/mac-cpu-metal-owner-disposition-packet-2026-09-07.md` | Pointer updated | Frozen 63-row/hash/signature denominator retained, not replaced by live 58. |
| `docs/handoff/mac-cpu-metal-residual-execution-2026-09-12.md` | Pointer updated | Current metadata pointer; dated 135/59 and exact parity measurements retained. |
| `docs/handoff/mac-cpu-metal-status-2026-10-22.md` | Pointer updated | Fresh main/PR routing supersedes pending queue, not old device/provider observations. |
| `docs/handoff/mac-pre-scaleway-remaining-tasks-2026-09-05.md` | Pointer updated | Fresh complete TSV and source/main order; per-model historical table not re-certified. |
| `docs/handoff/public-catalog-security-completion-2026-09-29.md` | Pointer updated | Fresh all-pages alert counts and still-open model/security responsibilities. |
| `docs/handoff/security-remediation-2026-09-21.md` | Pointer updated | 572 Dependabot/eight Code Scanning snapshot; no alert dismissal or runtime/owner inference. |
| `docs/legal-compliance.md` | Updated | Fresh official-source retrieval/limits, implementation Deferred boundary and unmerged legal PR. |
| `docs/license-audit.md` | Pointer updated | Current routing and security distinction; dated licenses/signatures/protected input unchanged. |
| `docs/release-0.4.0-rc1.ja.md` | Reviewed, unchanged | Published source/tag, 22-asset metadata and EN twin match; no downloads/model execution. |
| `docs/release-0.4.0-rc1.md` | Reviewed, unchanged | Published source/tag, 22-asset metadata and JA twin match; historical receipt not rerun. |
| `third_party/NVIDIA-EULA.md` | Updated | Official current CUDA EULA distribution/critical-use clauses; unsupported private-directory attribution corrected; no-bundle policy retained. |
| `third_party/QUALCOMM-QNN-NOTES.md` | Pointer updated | Read-only raw-loader/probe source; exact SDK/EULA acceptance/device execution still pending. |

### New documentation and evidence

- `docs/current-status.md` and `docs/current-status.ja.md`: paired current routing,
  exact source/release/PR/security/model snapshots and non-claims.
- Six per-scope coverage records linked above: checked inventories and limits.
- `docs/handoff/documentation-index-checkpoints-through-2026-10-22.md`:
  preserved old index, with an explicit historical wrapper.
- This audit record: integrated evidence, coverage and gate results.
- `docs/handoff/model-inventory-2026-10-23.tsv`: fresh metadata-only 194-row
  snapshot; SHA-256
  `9ef3eb07f12a98de7f425dcd75b98f4cd1be45ab1d3907aafeca2626196cdd56`.
  It contains repository/card/GGUF-name metadata, not downloaded tensor data.

Ignored research/tickets/ADRs/milestone/superpowers logs are local historical
working records, not current public guides; generated dependency/venv/vendor
documentation is not project-authored documentation. They are not force-added,
bulk-restamped or used to override current tracked guides. The five ignored
plans/references explicitly reviewed above remain local. No protected manifest
is read, hashed, materialized, staged or changed.

A path-only inventory under `docs/` additionally classified 338 ignored prose
paths: seven top-level, 65 research, 69 ADR, two handoff, seven superpowers and
188 ticket files. This is not a claim of full-body review of those private
records. The public coverage denominator deliberately remains the 400 tracked
documents; generated dependency documentation and local historical logs are
not bulk-edited.

## Fresh external evidence

Read-only APIs, with no mutations:

- GitHub main, pull requests, exact-head checks/workflows, releases/assets and
  branch protection. Exact main has 77 terminal checks (70 success/seven
  event-specific skips), nine successful push workflows and eight open PRs
  (six drafts). Ordinary CI is not a license/model/device verdict.
- Stable `v0.3.0`: 18 assets. Published `v0.4.0-rc.1`: 22 assets, prerelease,
  exact source `d84d311e98d0abb2987b5c4e96157ae8f2068e1a`; no package
  registry publication follows. Stable Unity/Godot/web/Python/XCFramework
  assets exist, but those packages are not in the desktop RC payload.
- Four named non-secret publish flags (crates.io, PyPI, npm, OpenUPM) read
  `false`. Godot AssetLib authorization was not inferred from those flags.
- All-pages alerts: 572 Dependabot (93 critical/247 high/173 medium/59 low;
  556 named patched version/16 without), eight Code Scanning (four CodeQL
  server-test log-injection / four Scorecard). No finding dismissed/closed.
- Public HF API/model cards: 194 repositories, 193 GGUF-bearing, 198 files;
  CPU full136/partial43/no-binder14/non-artifact1; Metal full136/CPU-blocked57/
  non-artifact1. All source/artifact classification only; 58 remain non-full.
- Official legal/store/vendor pages: see the [dated retrieval table](../legal-compliance.md)
  and [CUDA terms correction](../../third_party/NVIDIA-EULA.md). EUR-Lex bot
  verification and California HTTP403 prevent current-law reauthentication;
  cached federal introduction data is not a current enactment verdict.
  Current Apple/Google policy retrieval and HHS role guidance do not establish
  application compliance. No model license class or owner sign-off changed.
- Official Codex customization/skill documentation was read for the narrow
  skill wording cleanup, not as permission to weaken project safety rules.

## Verification record

All commands use Python 3.12 through `uv run` where applicable, offline with a
temporary UV cache and interpreter downloads disabled. The initial default-cache
permission failure in workflow hygiene is setup failure, not a passing gate;
its corrected-cache rerun is recorded below. Failed network/API retrievals
are never counted as empty inventories.

Integrated verification:

| Check | Result / boundary |
|---|---|
| `check_doc_examples.py --self-test` | PASS; intentional negative fixtures are rejected |
| `check_doc_examples.py` | PASS: 144 blocks / 26 docs; 30 Tier C examples deferred, not verified |
| `check-doc-references.sh --self-test` | PASS: 17 cases |
| `check-doc-references.sh` | PASS: 110 IDs, 33 architecture anchors, 12 hierarchy docs / six EN-JA pairs |
| `check-runbook-path-citations.sh` | PASS; absent/local historical citations remain explicitly classified |
| `check-community-docs.sh` | PASS: ten required community files, two forms, three twins |
| `check-parity-sidecar-citations.sh` | PASS; no reference or model execution |
| `check-owner-checklist-drift.sh` | PASS: 28 claim/evidence anchors; advisory, not owner acceptance |
| `check-platform-support.sh` | PASS: 53 anchors; no device/platform verdict promoted |
| `check-abi-changelog.sh` | PASS: 124 prefixes and changed C symbols covered; historical anchor unchanged |
| `check-workflow-hygiene.sh` | PASS with temporary UV cache: 59 workflows / 36 cron entries |
| `check-codex-hooks.sh` | PASS: eight shell files, schema/targets/syntax/self-tests/diff; existing guard repair remains separate |
| `check-zero-deps.sh` | PASS: root lockfile first-party only; no Cargo execution |
| Integrated per-file coverage | 400/400 starting tracked documents, zero missing paths |
| Relative Markdown paths | All 410 tracked/new Markdown files scanned; no missing relative destinations |
| Changed/new Markdown fragments | Non-empty local Markdown heading targets resolve; empty `#` is a valid top-of-page link |
| Preservation checks | Old index body exact after link rebasing; M5 checkbox states unchanged (53/29); all eight skill twins identical |
| `git diff --check` | PASS with protected manifest excluded; new Markdown whitespace checked separately |

Model, workspace/`vokra-models` Cargo, toolchain/device/UI execution, full
external link crawling and legal sign-off are not part of these static checks.
Unsupported broad GPU-speed comparisons were removed from the migration guide;
its replacement cites the exact dated RTX 4090 reports, not a new benchmark.
Skill `quick_validate.py` could not start because its PyYAML dependency is absent;
no package installation or parity-environment sync was attempted. Frontmatter
and UI metadata were not changed, all eight skill twins match, and repository
doc-reference/diff gates still apply.

## Non-claims and remaining project work

The documentation refresh is not full model-catalog completion. PR/source/
dependency/license/owner duties, real BF16/HiFTNet/all four BigVGAN prerequisites,
independent real-weight VAST CPU, final reviewed clean HEAD/fresh packets,
Scaleway-last CPU/reference + Metal/reference + Metal/CPU/no-fallback, result
acceptance and separately authorized publication remain distinct future work.
Historical named Apple/publication successes do not close all 194 rows.

No local model was downloaded/loaded/run; no paid worker was created and no
fresh VAST/Scaleway account inventory is claimed. No commit, push, PR change,
merge, registry/HF/X publication, guard weakening or automatic approval change
was performed. The previous task's dirty `.codex/hooks/guard-local-models.sh`
remains separate and must not be included in a docs-only logical commit.
