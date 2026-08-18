# Vokra documentation map

**Current-state review:** 2026-08-18
**Repository baseline:** `main` at `6d64fdf` (PR #37)

This page explains which documents describe current behavior and which are
dated evidence. A dated benchmark, ADR, plan, or handoff remains true for the
commit and environment it names; its old branch, test count, ABI count, or
“next step” is not automatically a current instruction.

## Current sources

| Need | Authoritative source |
|---|---|
| Repository invariants and agent operation | [`AGENTS.md`](../AGENTS.md) |
| Contributor workflow and local/VAST boundary | [`CONTRIBUTING.md`](../CONTRIBUTING.md) |
| User-facing capabilities and model roster | [`README.md`](../README.md) / [日本語](../README.ja.md) |
| Build and first run | [Getting started](getting-started.md) / [日本語](getting-started.ja.md) |
| Architecture and crate layout | [Architecture](architecture.md) / [日本語](architecture.ja.md) |
| C ABI and CLI surface | [API reference](api-reference.md) / [日本語](api-reference.ja.md) |
| Backend behavior | [Backend guide](backend-guide.md) / [日本語](backend-guide.ja.md) |
| Model and dependency decisions | [Licence audit](license-audit.md) |
| Deployment law and policy | [Legal compliance](legal-compliance.md) |
| Current M5 actions | [M5 owner checklist](m5-owner-verification-checklist.md) |
| 2026-08-18 Git/VAST/session reconciliation | [Codex operations handoff](handoff/codex-operations-2026-08-18.md) |
| 2026-08-18 workflow Python migration | [Workflow Python/uv migration](handoff/workflow-python-uv-migration-2026-08-18.md) |

The product-planning set (`requirements.md`, `system-requirements.md`,
`deliverables.md`, and `milestones.md`) and `CLAUDE.md` are intentionally
`gitignore-local`; they are available in the maintainer workspace but are not
links that a public clone can resolve.

The generated source or checker wins when a human-readable page disagrees
with it: `include/vokra.h` for the C ABI, `vokra-cli convert --help` for
model-kind dispatch, model manifests for implementation coverage, and the
publication scripts for release gates.

## Current repository state

- `main` and `origin/main` were synchronized at `6d64fdf` when this
  review began. The only active fetched remote branch was `origin/main`.
- The retired audit branch was not merged wholesale. PR #29 carried the audit
  work, PR #32 carried the Codex migration, and PR #37 carried the remaining
  Claude compatibility delta.
- The generated C header currently has 41 `vokra_*` functions. References
  to 33 functions in dated M4/M5 reports are historical snapshots.
- M5 is not complete and the C ABI is not frozen. The live checklist has
  42 checked and 36 unchecked actions; unchecked actions are not equivalent
  to missing implementations.
- The official-zoo reality gate currently accepts 20 advertised rows and
  reports no declared implementation gaps.

## Living plans versus implementation evidence

`requirements.md`, `system-requirements.md`, `deliverables.md`, and
`milestones.md` are living product/planning documents. They describe target
scope and acceptance conditions; a target row is not proof that the
implementation or real-hardware verification is complete. Current completion
evidence comes from source, generated surfaces, green checks, parity reports,
and the owner checklists.

The following directories are primarily dated records:

- `handoff/` — branch- or campaign-specific transfer notes;
- `bench-baselines/` and `benchmarks/` — measurements for named hardware,
  commit, flags, and fixtures;
- `adr/` — decisions at the status and date written;
- `_research/` — initial 2026-07 research snapshots;
- `superpowers/` — implementation plans/specifications, not live status.

Do not rewrite measured results to match a newer branch. Add a supersession
note or a newer report when disposition changes.

Some planning sections still cite the former `CLAUDE.md` project chronicle
as the source of a 2026-07 estimate or decision. Those citations mean the
version committed at that date, not the current compatibility entry point.
They are historical provenance. Public-clone instructions come from
`AGENTS.md`, the tracked guides, and this map; maintainers also reconcile
them with the `gitignore-local` living requirements in their workspace.

## Documentation conventions

- Commands intended to be run today use uv for Python. Historical prose may
  mention an old pip incident, but executable recipes must use `uv run`,
  `uv sync`, or `uv add`.
- On the maintainer Mac, aggregate model artefacts of 2 GB or more,
  workspace-wide Cargo, and every compiling/testing/checking
  `-p vokra-models` command run on VAST. Public examples use focused package
  commands where possible.
- English/Japanese twins are maintained for the public entry guides and
  platform tutorials. Audit, legal, benchmark, ADR, and handoff records may
  intentionally have one language only; the repository does not promise a
  translation for every top-level Markdown file.
- Links to absent private tickets are valid only when explicitly labelled
  `gitignore-local` or historical.
- Never copy credentials into documentation. Tokens belong in ephemeral
  environment variables and an exposed credential must be rotated.

## Lightweight validation

These checks do not compile the workspace or `vokra-models`:

```sh
scripts/check-doc-references.sh
scripts/check-runbook-path-citations.sh
scripts/check-community-docs.sh
scripts/publish/check-catalog-reality.sh
scripts/check-workflow-hygiene.sh
git diff --check
```

`check-community-docs.sh` currently reports a deliberate pending state for
four contact-dependent files (`CODE_OF_CONDUCT{,.ja}.md` and
`SECURITY{,.ja}.md`) until X-05-T04 supplies owner contact points. That is an
explicit owner dependency, not a broken-link result.
