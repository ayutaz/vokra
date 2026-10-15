# XCodec2 source verification: actual compile failure and repair

Readback: 2026-10-15 JST. This records an actual model-free VAST attempt,
not numerical parity, Apple verification or a completed source gate.

## Exact input and authorization

The tested clean HEAD was `bcfa7888250534a609e28c7d01b4f58f1741d2c1`,
normally integrated with accepted main `8bd30d9a`. Source bundle: 18,926
bytes, SHA-256 `3255423ba27fa90703869b6a02fdbbaff73218eb7467145237e82a946e6135a3`.
The two prerequisites were published XCodec2 `eaf0e5fa` and accepted main.

The initial USD 0.20/hour / zero-bandwidth search returned no eligible offer
and stopped before create. The user subsequently approved USD 0.35/hour
including allocated storage, upload and download each at most USD 0.005/GB,
at most 30 minutes, one model-free verification and owned-resource/data
destruction. The 10-GB cost illustration was not a hard total-spend cap.

The reviewed controller SHA-256 was
`911e1bbd2fc52471a7ddc6b22516a74f4202be1ddf37edebfe64e403824dddaf`;
harness SHA-256 was
`4d8d79a50e10d94f9a636dd3ea28431503e06a41ce302195a3b2d63c6c4ade8c`.
Manager's actual ordinary offline harness exited zero, including five
positive and 16 negative fee cases, existing lifecycle/source-exclusion
controls, syntax and default ShellCheck. Actual source-bundle admission passed.

## Actual execution and unchanged failure

Actual controller session 71999 started at 01:17:25 UTC. It selected offer
44261417 and created owned instance `56019893` with 16 effective CPU cores,
96,397 raw MB RAM and storage CLI value 150. The quoted and claimed hourly
rate was USD 0.26566666666666666; upload/download rates were respectively
0.00390625 / 0.0026041666666666665 USD/GB. These are quotes, not a billing total.

Direct SSH succeeded. Fixed-HEAD/clean-source, filter/sparse model exclusion
and format checks passed before workspace all-target/all-feature Clippy.
Clippy stopped with inner exit 101 and **25 E0308 compile errors**; the
controller ultimately exited 1. The targeted Rust tests and final three
shell gates were not reached. No model was acquired, loaded or executed.

All errors have two common causes in the new test source:

- The SHA-256 round table declared 64 entries but contained 63: standard
  word `0xa2bfe8a1` was missing at zero-based index 40.
- Twenty-four negative-test closures return `Packet`, vectors or a backend
  enum, while their helper accepted only `FnOnce() -> ()`.

## Reviewed repair, not a new runtime result

Luna restored the missing standard word and generalized only the helper's
return type to `FnOnce() -> T`. The `catch_unwind(f).is_err()` assertion and
all call sites are unchanged; normal-return closures still fail the assertion.
No fixture, expected artifact hash, numerical bound, approval field or test
skip was changed. Source SHA-256 after repair is
`50a96c031d931f2a8309fc2d1c53d2f898d16caa5e26a0d45d4442ed75c32dc1`.

Manager reviewed the entire diff and independently derived all 64 words
using exact integer cube roots of the first 64 primes, following
[FIPS 180-4 section 4.2.2](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf).
All values and order match. The old table is exactly the same sequence with
only index 40 omitted. Other than that table, the helper signature is the
only source change. This static proof is not Rust execution.

## Recovered evidence and destruction

Both result and full bounded remote log were recovered before destruction.
The recovered log contains all 25 errors; the local log is the identical
recovered body plus a 117-byte SSH welcome prefix, not two different failures.

| Evidence | SHA-256 |
|---|---|
| Recovered remote log | `94d2bcce1e14ae143eb1bc798dc56a615ce28495b0bd642ab281ebdff93db50e` |
| Recovered result | `5b4df6db54174e346c00fec31d6486e9cf6bf5b4f3a5c44750277521d5eabad3` |
| Individual deletion readback | `817de4eb9b246ba142dd72761e9f1ac6f4aa9f0da57bd831acdfd81f78797057` |
| Controller status | `8d06121190fffdee58ad724bd4ee199c9225db713fa543ee08c9974fd40652aa` |

At 01:20:01 UTC the owned instance and saved data were destroyed; the
individual API readback was `{"instances":null}`. Recovery and cleanup both
exited zero. This is not an account-wide inventory or billing guarantee.

The repaired candidate remains uncompiled and unpushed. Regenerate/review its
source bundle and obtain authorization for an additional paid verification;
do not reuse this consumed one-run permission. Keep PR #152 Draft with its
full dependency/native/license/API/owner scope. Real-weight CPU, all 58
unresolved public rows, frozen 63-row decisions and final Apple packets are
unchanged. No local heavy Cargo, model operation, HF token transfer or upload
occurred. The protected CosyVoice2 owner manifest was not inspected or changed.
