# DAC/OWSM urllib3 reconciliation — 2026-10-15

This handoff records a bounded, model-free lock reconciliation from
`5ceb432545cc1d9bdf2ff5c5ff289d683276539a`. Only these two lock files are in
scope:

- `tools/parity/dac/uv.lock`
- `tools/parity/owsm_v4_medium_1b_reference/uv.lock`

The protected CosyVoice2 reference tree and the Silero VAD test tree were not
materialized in the private sparse checkout. No model, upstream source,
checkpoint, provider, or credential was accessed.

## Exact lock change

Both lock files retain every package row other than `urllib3`, every dependency
edge, root field, resolution marker, and ordering. The only change is the
official `urllib3==2.8.0` package record:

- sdist: size `458972`, SHA-256
  `63bf2ead4c879426ebf22ef2a781eeb4aa3b4ae798a0435506f8687fd5bb9b63`
- wheel: size `135717`, SHA-256
  `0cf3cae568d36aa9576b28dfb35f11328f1cb974ca7647d9475ebb86c75ac6e3`

The source URLs and upload timestamps are the official PyPI records already
verified for the Dependabot update. In each lock, the sole incoming edge is
from `requests==2.34.2`. PyPI's primary release metadata for that exact
version declares `urllib3<3,>=1.26`, which admits `2.8.0`:
<https://pypi.org/pypi/requests/2.34.2/json>.

| lock | rows (other than urllib3) | dependency edges | HEAD SHA-256 | reconciled SHA-256 |
| --- | ---: | ---: | --- | --- |
| DAC | 108 of 109 | 146 | `4cc22fdf4da625b404b48fa1e858788e336a230c8dbd7cf93d75f9d881f03740` | `6ac987a73e0512365cb90623c57e67bbf553e97d48c0ad9097617cead07c65d1` |
| OWSM | 42 of 43 | 64 | `a7358dac9d5e753dd9c296f8895bec08ee8a98f0eb73dfb9940b3e0a59c2fe52` | `21dc57e49571c7b47e84c91ae84166c35ca418b26635d1f4ac236d69925b1ebb` |

No literal lock-hash consumer was found in the DAC tree. OWSM's audit computes
the lock hash at runtime in `dependency_audit.py`; no historical approval,
installed/native evidence, or audit artifact was restamped.

## Focused verification

Using only offline Python 3.12 through `uv run --no-project`, a TOML
comparison proved that all non-`urllib3` rows and edges match `HEAD`, the root
fields and resolution markers are unchanged, and both exact 2.8.0 records
match the approved source/hash/size values.

The existing OWSM model-free gates also pass:

Manager independently compares the ordered complete package records and root
fields for both locks and reproduces both new hashes. OWSM's 43 records include
two distinct Torch variants; a name-keyed dictionary is not sufficient to
prove that count or preserve both records. The ordered comparison retains
both, and both urllib3 records exactly match the reviewed DFN3 2.8.0 record.
Manager independently repeats the self-test and all eight tests below, then
runs `dependency_license_gate.py --project .`: it returns the expected
pending status with no structural errors, not an execution approval.

```text
uv run --no-project --offline --python 3.12 python dependency_license_gate.py --self-test
owsm_v4_medium_1b dependency/license gate self-test: OK

uv run --no-project --offline --python 3.12 python -m unittest discover -s . -p 'test_*.py'
Ran 8 tests ... OK
```

The normal gate remains fail-closed and returns
`BLOCKED_UNREVIEWED_TRANSITIVE`, with `lock_package_count=43`,
`execution=NO_LOCAL_EXECUTION`, and `publication=NO_UPLOAD`. The existing
`torch-complex`/native/source review blockers are unchanged.

No `uv sync`, install, upstream import, model execution, Cargo command,
provider access, `.env` access, commit, push, or PR mutation was performed.
