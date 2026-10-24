# Engineering records

These are dated engineering records, not a second current-status feed. Start
with [current status](../current-status.md) / [現況](../current-status.ja.md)
for releases, source, open PRs and outstanding model/device work. Use the
[documentation map](../README.md) for user and contributor guides.

## Find evidence by topic

| Topic | Starting records |
|---|---|
| Published CPU preview | [Packaging and publication](release-0.4.0-rc1-packaging-2026-10-17.md), [CPU results](release-0.4.0-preview-cpu-results-2026-10-17.md) |
| Repaired Whisper and released-CLI pairing | [Provenance repair](whisper-base-provenance-repair-2026-10-23.md), [execution record](cpu-preview-announcement-execution-2026-10-23.md), [raw final receipt](evidence/published-rc1-cli-pairing-final-2026-10-23.json) |
| Preview announcement draft | [Draft and fixed verification scope](cpu-preview-announcement-draft-2026-10-24.md) |
| PR and CI work | [PR readback](pr-processing-readback-2026-10-24.md), [action-update preparation](pr271-action-update-preparation-2026-10-24.md), [current status](../current-status.md#main-ci-and-open-pull-requests) for later acceptance |
| Candle same-run artifacts | [Producer readback](candle-current-producer-readback-2026-10-24.md), [consumer result](candle-current-consumer-success-2026-10-24.md) |
| Mac campaign prerequisites | [Remaining-task ledger](mac-pre-scaleway-remaining-tasks-2026-09-05.md), [execution plan](mac-cpu-metal-execution-plan-2026-09-07.md) |
| Owner decisions and Apple evidence | [Frozen owner packet](mac-cpu-metal-owner-disposition-packet-2026-09-07.md), [dated Apple results](mac-cpu-metal-scaleway-results-2026-09-11.md) |
| Catalog snapshot | [2026-10-23 metadata-only inventory](model-inventory-2026-10-23.tsv); this is not a hardware pass count |
| Dependency and source audits | [VibeVoice static audit](vibevoice-realtime-static-closure-audit-2026-10-13.md), [dated security plan](security-remediation-2026-09-21.md), [license audit](../license-audit.md) |
| Measurements | [Baseline provenance index](../bench-baselines/README.md), separate from source and catalog readiness |
| Documentation review | [2026-10-23 coverage audit](documentation-refresh-2026-10-23.md), [2026-10-25 structural refactor](documentation-refactor-2026-10-25.md) |
| Older index checkpoints | [Archived documentation index](documentation-index-checkpoints-through-2026-10-22.md) |

## Earlier public-guide checkpoints

Release/PR/catalog summaries formerly repeated in user guides are available at
the immutable pre-refactor source `c3e044929a0231fc4eda29aa5f5a34bcf62ea26e`:

- Root README: [English](https://github.com/ayutaz/vokra/blob/c3e044929a0231fc4eda29aa5f5a34bcf62ea26e/README.md) / [日本語](https://github.com/ayutaz/vokra/blob/c3e044929a0231fc4eda29aa5f5a34bcf62ea26e/README.ja.md).
- API index: [English](https://github.com/ayutaz/vokra/blob/c3e044929a0231fc4eda29aa5f5a34bcf62ea26e/docs/api-reference.md) / [日本語](https://github.com/ayutaz/vokra/blob/c3e044929a0231fc4eda29aa5f5a34bcf62ea26e/docs/api-reference.ja.md).
- Backend guide: [English](https://github.com/ayutaz/vokra/blob/c3e044929a0231fc4eda29aa5f5a34bcf62ea26e/docs/backend-guide.md) / [日本語](https://github.com/ayutaz/vokra/blob/c3e044929a0231fc4eda29aa5f5a34bcf62ea26e/docs/backend-guide.ja.md).
- Earlier routing banners in the [license ledger](https://github.com/ayutaz/vokra/blob/c3e044929a0231fc4eda29aa5f5a34bcf62ea26e/docs/license-audit.md), [ABI ledger](https://github.com/ayutaz/vokra/blob/c3e044929a0231fc4eda29aa5f5a34bcf62ea26e/docs/abi-changelog.md), [M5 checklist](https://github.com/ayutaz/vokra/blob/c3e044929a0231fc4eda29aa5f5a34bcf62ea26e/docs/m5-owner-verification-checklist.md) and [platform matrix](https://github.com/ayutaz/vokra/blob/c3e044929a0231fc4eda29aa5f5a34bcf62ea26e/docs/platform-support/v1.0-rc-support-matrix.md).

Their old counts, pending states and hardware claims retain their original
dates and limitations. Removing repeated summaries from current guides does
not replace the original result records, evidence files or signed decisions.

## Reading a record

Check its date, source HEAD, model/fixture identity, environment and result
scope. A successful source check, synthetic test, model conversion or metadata
audit is not independent real-weight parity, Apple acceptance or publication
permission. Old cloud instance IDs and packets must not be assumed retained.

Filenames identify topics and observation dates; the latest filename alone is
not an execution instruction. Historical failures, measurements, checksums,
owner decisions and evidence links remain attached to their original records.
