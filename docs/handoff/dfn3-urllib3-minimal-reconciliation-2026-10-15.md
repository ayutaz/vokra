# DFN3 urllib3 minimal reconciliation — 2026-10-15

This dated record audits Dependabot PR #254 at head
`02c314897ba22596415025dae011bbe74e031014`, based on main
`5ceb432545cc1d9bdf2ff5c5ff289d683276539a`. The bot lock update is retained for
urllib3 2.8.0, while unrelated resolution-edge marker churn is restored to the
accepted base.

## Exact change boundary

The original bot diff touched only `tools/parity/dfn3/uv.lock` with 33 additions
and 33 deletions. Thirty deleted dependency-edge marker lines were unrelated to
urllib3 and are restored byte-for-byte from main. The remaining three changed
records are the urllib3 version, sdist, and wheel records:

- `2.7.0` → `2.8.0`;
- sdist URL, SHA-256, size, and upload time are the exact bot values;
- wheel URL, SHA-256, size, and upload time are the exact bot values.

The project and lock remain constrained to Python 3.11 on Linux x86_64. The
lock has 26 package records and 30 dependency edges; the package inventory,
all non-urllib3 package records, all dependency edges, resolution markers, and
supported markers match the accepted base.

## Binding and verification

The corrected lock SHA-256 is
`d78f54a654681763b4e33e5c69b149e9ebcaf277c0bdeecf4b8df66cc135c844`.
The accepted-base lock SHA-256 is
`b9c9933959e7e0d7ba0e0944e605d2f769924bf17d2e3b1ff62ec72c15f0a643`.
The unreviewed bot-head lock SHA-256 was
`caf6f1f72eda18507a871dd2435191dc147696f251406a5223aadedf018ce356`.

An offline Python 3.12 stdlib `tomllib` comparison passed: 26 package records,
30 dependency edges, Linux x86_64 scope, and the exact two urllib3 2.8.0
artifact records. No current DFN3 consumer binds an evidence packet to a lock
SHA-256; the workflow references the lock path but does not contain a literal
lock digest, so no consumer restamp is required.

Manager independently repeats the complete-record comparison: all 25 other
packages, all root lock fields, package ordering and all 30 dependency edges
match main; the urllib3 record is exactly the bot's. The [official release JSON](https://pypi.org/pypi/urllib3/2.8.0/json)
independently matches both artifact URLs, lengths and SHA-256 values. It
declares Python `>=3.10`, admitting this existing Python 3.11 reference tree,
and MIT metadata. No archive was installed, imported or executed for this check.
This source update is not a new DFN3 numerical/reference-environment result.

No model, checkpoint, tokenizer, upstream import, dependency sync/install,
Cargo, provider, or `.env` operation was performed. The protected
`tools/parity/cosyvoice2_llm_reference/` tree and `tests/parity/silero_vad/`
tree were excluded from the sparse checkout.

## Read-only residual urllib3 2.7.0 inventory

Other dedicated parity locks still containing urllib3 2.7.0 at this head were
observed but not edited:

- `tools/parity/baichuan_audio_instruct/uv.lock`
- `tools/parity/canary_1b_reference/uv.lock`
- `tools/parity/chattts/uv.lock`
- `tools/parity/dac/uv.lock`
- `tools/parity/owsm_v4_medium_1b_reference/uv.lock`
- `tools/parity/vibevoice_1_5b_reference/uv.lock`

This record does not claim those six scopes are safe to batch-update; each needs
its own marker, evidence, and consumer review.
