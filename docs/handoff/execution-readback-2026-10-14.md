# Execution readback — 2026-10-14

This record follows the [prioritized plan](next-steps-2026-10-14.md). It records
source preparation and a stopped workspace-verification attempt, not completed
real-weight CPU parity or a final Apple packet. Reviewed source candidate was
`bf1f1eb28361403366c882dda8577d60e0fdb271`; accepted remote main remained
`134eb9c7dbbfdafd2e4c1bdddac2e97253f9eea4` at the preceding readback.

## Current public metadata

A fresh `hf_mac_coverage.py --workers 6 --format tsv` invocation through offline
UV/Python 3.12 fetched only public API metadata and fixed-revision model cards.
Exit was 0: 194 repositories, 193 GGUF-bearing repositories, 198 files;
CPU full 136 / partial 43 / no-runtime-binder 14 / not-artifact 1; Metal full
136 / blocked-by-CPU 57 / not-artifact 1. The captured TSV SHA-256 is
`9ef3eb07f12a98de7f425dcd75b98f4cd1be45ab1d3907aafeca2626196cdd56`.
No model bytes were acquired. These remain code/artifact classifications,
not 136 Apple hardware passes or a new all-row execution-readiness verdict.

Bounded inspection of eight artifact-replacement families identified WeSpeaker
as a useful next dependency-audit candidate. Its current closure has 13 external
distributions plus one virtual project, with 14 dependency and five component
reviews pending. The corrected WeSpeaker repository is already in the full
136-row set; validating it does not silently close the separate legacy
`vokra/wespeaker` row in the 58 unresolved rows. No new owner disposition was
invented, and the protected CosyVoice2 manifest was excluded.

## Reviewed source change

The [Qwen3-TTS lock reconciliation](qwen3-tts-current-lock-reconciliation-2026-10-14.md)
was committed as `bf1f1eb2`. Current lock binding is repaired while all new-scope
approval fields remain null and historical audit/operator evidence is retained.
Manager self-tests, current-lock rejection ordering, documentation references,
runbook citations, community gates, and normal pre-commit checks passed.
The source and documentation commits have not yet been pushed or accepted by PR.

## OpenMP source/native evidence

The official PyTorch v2.13.0 tag resolves to
`cf30153c4c131c8164ee7798e5022d810682e2cb`. Its pinned
[OpenMP acquisition script](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/.ci/wheel/install_libomp.sh)
selects conda-forge `llvm-openmp-21.1.8-h4a912ad_0` for macOS arm64.
The [release metadata](https://api.anaconda.org/release/conda-forge/llvm-openmp/21.1.8)
and actual small software archive were checked without installation or execution.

| Actual retained payload | Bytes | SHA-256 |
|---|---:|---|
| `.conda` archive | 285,974 | `56bcd20a0a44ddd143b6ce605700fdf876bcf5c509adc50bf27e76673407a070` |
| `lib/libomp.dylib` tar member | 843,984 | `363a5a476fda527920fd9919ac34490331e5b667239108e691d5b3ea1aeec531` |
| `info/licenses/openmp/LICENSE.TXT` in the **package** tar | 19,741 | `fdad1758a9e1f9d5a81e18879b3406772115edc92c24bfa36b70c654f325e8e4` |

The archive records Apache-2.0 WITH LLVM-exception and accompanying notices.
Its embedded recipe identifies feedstock commit
`9a0e9859495a5a8c247cae11d866a0aac58d22f9`; five exposed recipe source files
were authenticated and compared byte-for-byte. Manager independently rehashed
the archive and streamed library/license members. This is not a reproduced build
or a correspondence proof for a Torch wheel: PyTorch's
[embedding step](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/tools/embed_libomp_macos.py)
rewrites load paths, and its acquisition script changes the install name and
codesigns the library. No package, model, owner, or publication gate is promoted.

## VAST admission and stopped worker

The new workspace-only scope was bounded to USD 0.20/hour including 150 GiB
storage, with bandwidth charged separately and a 27-minute work deadline plus
recovery/deletion time. This is not a reuse of the prior one-off private
41-wheel static-audit authorization. HF tokens and external model data were not
part of this scope.

- At 01:23 UTC the strict offer postfilter found no qualifying offer. Creation
  was not attempted; this admission incurred no new worker charge.
- A later offer `51326863` qualified at USD 0.195/hour. Owned instance
  `55844038`, label `vokra-workspace-bf1f1eb2-20261014T0130`, was created and
  recovered through exact-label/baseline ownership checks.
- The stock image had not assigned a direct SSH port: `direct_port_start=-1`,
  `actual_status=null`. The original short readiness loop stopped before SSH,
  source transfer, remote bootstrap, Cargo, or model work. Default create output
  also differed from the parser's JSON expectation; label recovery prevented
  loss of ownership tracking.
- The owned instance and its storage were destroyed. Controller individual
  readback and a subsequent independent authenticated lookup both confirmed
  explicit absence (`instances: null`). This short lease is not described as
  free; no invoice total or zero bandwidth charge is inferred.

Corrections add raw create output and bounded cold-boot polling, including
blocked-poll process supervision. Offline mock passes do not establish actual
provider boot, SSH, bootstrap, or Cargo success. A later manager full-test attempt
returned nonzero despite mock PASS/result recovery and null deletion readback;
exit-status diagnosis remains open. Do not rent again or claim remote green
until that cause and focused/full regressions are resolved.

The default SSH public key was initially checked against the wrong API column.
Correct `public_key` comparison and noninteractive private-to-public derivation
confirmed the existing local key. No new key was created or registered, and no
key values or private material were printed. That corrected diagnostic is not
an actual SSH authentication result.

## Remaining gates

The WeSpeaker static collector is a temporary candidate, not acquired-wheel
evidence: its offline report has 13 `NOT_ACQUIRED` rows and one virtual row,
`BLOCKED_UNRESOLVED_EVIDENCE`, `NO_UPLOAD`, and null owner fields. Review fixes
cover streamed large native files, ELF32/64 linkage parsing, exact license-byte
retention, and a shared interrupting deadline. Manager stdlib-only self-test and
offline integration passed; live Linux fetch/native facts remain unmeasured.

Next: resolve workspace-controller exit status, obtain exact-source remote green
and PR acceptance, then perform the separately bounded current dependency/API
work and only execution-approved real-weight CPU work. All 58 unresolved rows
remain accounted for; none is generically withheld merely for being difficult.
Final clean HEAD/Apple packet, Scaleway, and publication are still incomplete.
Local model execution and local workspace Cargo were not performed. The original
local-model safety guard remains unchanged. Free space was approximately 199 GiB
at the filesystem readback; no additional model was deleted by this work.
