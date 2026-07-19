# Quarterly review records

Storage conventions for the quarterly Go/No-go review (`NFR-MT-05`). One
review produces two artefacts: a **metrics snapshot** (machine-generated) and
a **review record** (written by the maintainer).

See [../quarterly-review-runbook.md](../quarterly-review-runbook.md) for how a
review is actually conducted, and
[../kill-switch-metrics-runbook.md](../kill-switch-metrics-runbook.md) for how
the metrics are collected and interpreted.

## File naming

| Artefact | Name | Produced by |
|---|---|---|
| Metrics snapshot | `<YYYY>-Q<N>.metrics.json` | `scripts/kill-switch-metrics.sh` |
| Review record | `<YYYY>-Q<N>.md` | The maintainer, from the template |

Example for the third quarter of 2026: `2026-Q3.metrics.json` and
`2026-Q3.md`.

The metrics filename is **not** a free choice — it is the output path the
metrics script's own usage text documents:

```bash
bash scripts/kill-switch-metrics.sh > docs/governance/quarterly-reviews/2026-Q3.metrics.json
```

Keep the two names in step. A record with no snapshot beside it cannot be
audited later, and a snapshot with no record is just numbers.

## What goes in a review record

Start from [../vokra-go-nogo-v0.5.md](../vokra-go-nogo-v0.5.md), which is the
blank template: Kill switch A–L status table, the individual verdict fields,
the continuous-monitoring section, known risks, the overall Go/No-go verdict
and the decision log. Copy it to `<YYYY>-Q<N>.md` and fill it in; do not edit
the template itself when recording a review.

If the verdict is No-go, the exit path is chosen from
[../exit-path-playbook.md](../exit-path-playbook.md) and recorded in the same
file.

## Rules that matter later

- **Record the numbers you actually had.** If a metric could not be collected,
  write why rather than leaving it blank or carrying forward a previous
  value. A review whose inputs cannot be reconstructed cannot support a
  decision to keep going *or* to stop.
- **Snapshots are append-only.** Never edit a landed `*.metrics.json`. If a
  measurement was wrong, add a correction note to the review record; the
  original stays.
- **A metrics run is not a verdict.** The script deliberately reports raw
  counts and leaves several judgements — Kill switch K's competitor
  comparison, and which contributor count the Kill switch D threshold applies
  to — to the maintainer. Do not automate those into the script.
- **Do not schedule this.** The metrics script is intentionally not wired into
  CI, and a scheduled workflow must not be created to run it (maintainer
  decision, 2026-07-04). Reviews are run by hand, quarterly.

## Current state

No review records have been filed yet. The first one is due per the Kill
switch evaluation calendar in the runbook.

**Open item affecting the calendar**: the runbook derives the Kill switch C/D
start date from the `v0.5.0` release tag, but no git tag and no GitHub release
currently exist, and the planning documents give two different calendar
windows for the D verdict. Resolving that — pick a start date, or tag
retroactively — is a maintainer decision and is tracked as X-05-T23. Until it
is settled, record the date you used and why, in the review record itself.
