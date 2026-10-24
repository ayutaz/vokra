# Documentation structure review — 2026-10-25 JST

## Baseline and investigation

The reviewed source is accepted main
`c3e044929a0231fc4eda29aa5f5a34bcf62ea26e`, before this documentation change.
The structural inventory covered 419 tracked Markdown files, including 260
under `docs/`: 36 root entries, 161 handoffs, 27 benchmark baselines, three
benchmark guides, nine designs, six governance records, two platform-support
documents and sixteen tutorials.

The review compared complete-file hashes, repeated prose, document size and
relative prose links. No files under `docs/` were byte-identical. The eight
identical Codex/legacy-Claude skill pairs belong to separate clients and were
not deleted. Translations, benchmark methodology repeated by generated reports
and dated evidence are not disposable duplicate files.

The main duplication was status and release/PR/catalog snapshots repeated in
README, API and backend guides, plus a backend-creation checklist repeated in
the architecture guide. No model or workspace-scale command was executed.

## Canonical ownership after the refactor

| Information | Canonical location |
|---|---|
| Project introduction and navigation | Root English/Japanese READMEs |
| Source/release/PR/model/security readback | [Current status](../current-status.md) / [日本語](../current-status.ja.md) |
| Exact preview distribution and model pins | [Preview guide](../release-0.4.0-rc1.md) / [日本語](../release-0.4.0-rc1.ja.md) |
| Source build and conversion recipes | [Getting started](../getting-started.md) / [日本語](../getting-started.ja.md) |
| Model-specific CLI options and formats | [CLI tutorial](../tutorials/cli.md) / [日本語](../tutorials/cli.ja.md) |
| Execution model and dependency boundaries | [Architecture](../architecture.md) / [日本語](../architecture.ja.md) |
| Backend implementation checklist | [Backend guide](../backend-guide.md) / [日本語](../backend-guide.ja.md) |
| API source and generated surfaces | [API index](../api-reference.md) / [日本語](../api-reference.ja.md) |
| Original results and old public-guide checkpoints | [Engineering records](README.md) |

Repeated status narratives were removed from the current guides and replaced
with these links. Immutable pre-refactor guide versions are linked from the
engineering-record index. The original evidence files, dated audit reports,
raw receipts, model inventory, license rows, ABI entries, owner checkboxes and
scope hashes remain unchanged. The license, ABI, M5 and platform documents
only replace their leading routing banners; their historical bodies are
preserved byte-for-byte from the stated markers onward.

Source-build instructions now point new evaluators to the verified prebuilt
preview first and do not promise a five-minute three-model validation. CLI
build examples select `vokra-cli` explicitly. Migration examples distinguish
whisper.cpp's ggml files from Vokra GGUF and clarify that native execution still
requires a supported converter and architecture; checkpoint weights alone do
not establish support.

The English README shrank from 245 to 103 lines and the Japanese README from
223 to 98. Old source-guide heading IDs are retained as compatibility anchors,
including the Android tutorial's existing link to the C ABI example.

The source/PR/CI readback retained in current status records accepted main
`c3e0449` with 80 terminal checks (72 success, eight conditional skips) and
seven open Draft PRs. That is source evidence, not a new model/device result.
The 2026-10-23 catalog and alert counts remain explicitly dated; they were not
retrieved again or promoted into current hardware/security acceptance.

## Verification boundary

Verification uses repository documentation gates, relative-link and changed
heading checks, and scoped Git diff review. Python is invoked through offline
`uv run --no-project --python 3.12`; heavy Cargo, model execution, cloud
allocation and publication are outside this documentation change.

| Check | Result |
|---|---|
| Documentation examples | PASS: 138 blocks across 26 documents; 30 Tier C cases remain explicitly unverified |
| Documentation references | PASS: 110 requirement IDs, 33 architecture anchors, English/Japanese and entry-link checks |
| Community documents | PASS: required files, Issue Forms, policy twins and navigation |
| Runbook paths | PASS: existing path-citation gate; ignored/local and absent-but-marked paths keep their stated limits |
| ABI changelog | PASS: changed symbols and 124 converter-stamped prefixes remain covered; no anchor rotation |
| Owner checklist drift | PASS: 28 claim/evidence anchors, no dead flag names |
| Platform matrix | PASS: 53 evidence anchors |
| Repository prose links | PASS across 421 Markdown documents, including the two new indexes/records; 11 fragment links into changed documents resolve |
| Translation structure | PASS: eight changed English/Japanese pairs have matching heading hierarchies |
| Historical preservation | Four ledger bodies remain byte-identical; existing local guard remains unchanged |

The only initial fragment mismatch was the old source-guide C ABI heading;
retaining its published ID resolved it without dropping the Android link or
waiving the check. The new indexes add navigation, not additional measurement
or acceptance evidence.

This is a structural review of the Markdown inventory and selected public
entry paths, not a new primary-source legal audit, remeasurement of benchmarks,
rerun of historical tools or assertion that every external link is live.
