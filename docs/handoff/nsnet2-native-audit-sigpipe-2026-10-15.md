# NSNet2 pre-download native-audit correction — 2026-10-15 JST

Source commit `e3b954e2b841fe76ae3f3cfa363de87dfe6aa87e`, based on accepted
main `731592f97251f52f0e80ea81d211d9dde6b52d71`, corrects the model-free
native admission check in the [VAST worker](../../scripts/publish/vast-ai/run-nsnet2-validation.sh).
It does not authorize or execute that worker. Actual dependency, model,
reference, CPU and Apple evidence remain absent for this current run.

## Reproduced production defect

The audit runs after reference-environment setup and before the official ONNX
download. Previously it captured `ldd` output, then piped `printf` into
`grep -q`. With a forbidden library on the first line and a large remaining
output, grep exited early; printf received SIGPIPE and the pipeline returned
141 under pipefail. The conditional therefore missed the forbidden library.
An inspection error other than normal no-match also fell through to PASS.

Luna compares original and corrected production functions using the same
synthetic files and unconditional `set -euo pipefail` child invocation. No real
library, package, model or network is used:

| Synthetic input | Bytes | Original exit | Corrected exit |
|---|---:|---:|---:|
| Small forbidden dependency | 52 | 2 | 2 |
| Large forbidden dependency | 5,700,052 | 0, incorrectly PASS | 2 |
| Large clean output | 5,700,000 | 0 | 0 |
| ldd error | 38 | 2 | 2 |
| grep inspection error | 38 | 0, incorrectly PASS | 2 |

An earlier diagnostic put the call under an outer conditional, suppressing
Bash errexit and making the small forbidden case appear to pass. Rechecking
the actual unconditional production context corrected that diagnosis. The
large-output and inspection-error defects remain independently reproduced.

## Fix and verification

The corrected function uses a pipeline-free here-string, classifies grep
status explicitly (match is rejection, normal no-match is allowed, other
errors reject), and returns on failed audit checks. Existing forbidden
basenames, native policy, source-build requirement, pre-download ordering,
license/owner gates and NO_UPLOAD behavior remain unchanged.

The [focused regression](../../scripts/compliance/test-nsnet2-native-audit-sigpipe.sh)
extracts the real production audit function instead of testing a duplicate.
It runs all five controls with synthetic site-packages and mocked ldd/grep;
CI Quality's checkout-only repo-hygiene job explicitly runs it. Manager
independently passes the five regressions, four worker self-test cases,
workflow hygiene and diff checks. All five normal pre-commit checks pass.

Default ShellCheck passes the new test. The whole worker retains an unrelated
existing SC2016 informational diagnostic for an intentionally literal
self-test marker; warning-severity ShellCheck passes. The workflow-hygiene
self-test has seven identical failures on original main and this candidate;
the normal 48-workflow gate passes. Neither limitation is hidden or described
as new-diff success.

This is a source-only correction to a prerequisite, not a real NumPy/native
audit or CPU parity result. Primary current execution scope, native build
evidence, independent reference and replacement/publication approval remain
separate. No dependency synchronization, model acquisition/load/forward,
Cargo compilation/testing, VAST lease, Scaleway or upload occurred. The
[full plan](next-steps-2026-10-14.md), 58 unresolved public rows and frozen
63-row owner scope are not narrowed or promoted by these tests.
