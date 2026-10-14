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
exit-status diagnosis was still open at that readback. The later resolution
below is not a remote Cargo result.

The default SSH public key was initially checked against the wrong API column.
Correct `public_key` comparison and noninteractive private-to-public derivation
confirmed the existing local key. No new key was created or registered, and no
key values or private material were printed. That corrected diagnostic is not
an actual SSH authentication result.

### Subsequent controller resolution and no-lease admission

The status receipt isolated the failing leg as
`initial=0 recovery=1 cleanup=0 validation=pass exit_status=0`: the second
offline log-recovery call crossed its one-second deadline despite successfully
producing mock evidence. Successful offline recovery now has five seconds;
failed/deadline fixtures retain their short causal budget, and production
recovery remains bounded to 60 seconds. Both recovery-call statuses are
recorded independently. Remote stages distinguish fmt, clippy, tests and
shell gates. The implementer repeated the full offline suite three times;
manager independent full regression, syntax and ShellCheck passed.

- Controller SHA-256:
  `f59a9000aaf7f1d018cf8d444e31b4e754f168753cf1bdb0caa0143db129e5d1`.
- Causal test SHA-256:
  `14b50a112461c89bed68a9e3d368818539dfbe30d9bce59f424e762e278758b8`.
- Frozen subsequent source HEAD:
  `d4ce1040b793fba1c75d7650caea503fac18fe29`; bundle SHA-256
  `bbe1ab802f554157ebcb3a000ecb0565b38345f751f155d2ddff8e635ef4e6bd`.

At 01:54 UTC the corrected controller ran a fresh admission. The cheapest
otherwise qualifying offer was USD 0.2027778/hour including 150 GiB storage,
above the unchanged USD 0.20/hour ceiling. It stopped before baseline inventory
or creation; no new instance was rented. Provider availability remains external
waiting, not a license to raise the ceiling, run Cargo locally, or claim green.

The [58-row prerequisite inventory](current-58-prerequisite-inventory-2026-10-14.md)
retains every non-full public row from the exact TSV and identifies the next
row-specific prerequisite. `NOT_PROVEN` is not a verdict that every row has
been exhaustively proved unready; the protected CosyVoice2 row remains
`UNKNOWN_EXCLUDED`. Full code/artifact candidates outside that denominator
are recorded separately rather than silently added to the 58.

## Actual workspace run and Linux shell-gate diagnosis

At 02:38:46 UTC, offer `51326867` qualified at USD 0.195/hour including
150 GiB storage (bandwidth separate). The unchanged reviewed controller rented
owned instance `55851188` for exact clean HEAD
`005ac12626f58f1b343203d52452a92fa9a9d2f3`; source bundle SHA-256 was
`6bbdc83e46ae72505e36bc014a4ba88c3c913c9e8493ce463bd1ba77c3e0b763`.
Existing-key direct SSH, bootstrap, checkout, fmt and all-target/all-feature
Clippy succeeded. The recovered complete test log contains 323 successful
test-result lines: 8,218 passed, zero failed and 105 ignored. This is workspace
test evidence, not external-weight CPU parity or Apple hardware evidence.

The overall result remains **FAIL / shell-gates**. Zero-dependency and
forbidden-symbol checks passed; the hook-contract gate then failed because
`check-protected-path-metadata.sh` used `/private/tmp` when `TMPDIR` was unset.
That directory does not exist on the Linux worker. The bounded fix changes
only the fallback to portable `/tmp`, preserving guard rules and expected
return codes. The corrected source requires a new exact-head remote pass;
the failed run is not push/merge approval.

Small result and log recovery succeeded, and cleanup ended at 02:54:23 UTC.
The owned instance and storage were destroyed; individual readback returned
explicit `instances: null`. Recovered log SHA-256 is
`d297ae25ec4b96857781726fe2a14c5e493b9de6e01e459b7b3157d7e2017c26`.
Other pre-existing account instances were outside this run's ownership and
were not changed. No HF token or external model weight was transferred.

### Corrected source: transport-only stopped attempt

The portable-directory fix and preceding receipt are committed at
`9e528d27bc1ca917a42c418be9396329b027ee7d`. Manager independently passed the
full offline hook contract and focused regression with `TMPDIR` unset;
the implementer also passed an explicit existing temporary-directory case.
The original local-model guard retains SHA-256
`0b8f3b1a343e1a8fd10e47f47d09dc65cbf9dc43aabb53bbeb3b68c9611edc62`.

A new bounded admission at 02:58:43 UTC selected offer `54502206` at
USD 0.1833333333/hour including 150 GiB storage, bandwidth separate.
Owned instance `55853246` reached provider `running`, but the readback's
explicit `22/tcp` host port refused all five SSH attempts. No source transfer,
bootstrap or Cargo started. The short five-attempt service-readiness window
is under review; provider `running` is not SSH readiness. No remote verdict
is inferred from local tests or the preceding HEAD's successful Rust stages.
Recovery could not connect; cleanup finished at 02:59:37 UTC with successful
destroy and explicit individual `instances: null`. No owned paid worker
from these two attempts remains at that readback.

### Fresh existing-PR readback

Current exact-head checks retain Draft #152 at 69 success / one skipped and
Draft #182 at 105 success / 13 skipped, both zero pending/failures. These
source-CI results do not close their model/dependency/native/owner gates or
approve a Draft merge. #169 is 68 success / one skipped / one failure;
#191 is 76 success / five skipped / one failure; #198 is 55 success /
one skipped / one failure. Each failure is `documentation-links`.
The #169/#191 job logs identify the official SB942 versions-comparison URL's
HTTP 403, separate from their implementation. A single unauthenticated,
20-second-bounded HEAD request at 03:02 UTC still returns HTTP 403 with
`cf-mitigated: challenge`. No unchanged CI retry, challenge bypass, citation
removal or protection waiver is made. Required-status-protection access via
the current GitHub integration returns 403; its settings are not inferred.

### Frozen SSH correction and admission without creation

The reviewed controller now gives the SSH service a bounded 180-second
readiness window instead of five attempts. Every attempt remains supervised
under both stage and total deadlines. Total work remains 1,620 seconds, the
remote leaf 1,500 seconds, and recovery/deletion retain their existing bounds.
Frozen controller/test SHA-256 values are respectively
`05effb78632260c406e124966078ffd4dfc974627f7d14a1c507c4c80dbcee1f` and
`d171d048cc2f4fab1c75941b6fcea9586689294875bfe5243644b2866a83595c`.
Implementer and manager independently passed syntax, ShellCheck and the full
offline causal suite at those fixed hashes. The delayed-service fixture only
succeeds on the seventh SSH call, beyond the old five-attempt window;
persistent refusal stops within its two-second test budget. Existing blocking
create/SSH child-termination, ownership and null-readback cases remain intact.
An earlier manager run during test-file editing was rejected, not counted as
acceptance; the subsequent frozen-hash run exited zero with unchanged hashes.
Offline mock evidence does not prove actual provider SSH or source acceptance.

A new admission for exact clean source
`5d95472770537ebcae0b152c615d02f5965bde23` stopped before baseline inventory
or creation because no returned offer passed the unchanged USD 0.20/hour
policy. Bundle SHA-256 is
`cbbe9c92fb20363ad75e7be5c8d3d6dad8930baaaf90400ed1e3b7f3575b791e`.
A separate read-only search lowering only the CPU floor to four cores still
returned a lowest eligible quote of USD 0.2283333333/hour including 150 GiB
storage, bandwidth separate. It created nothing and did not authorize a
different execution scope or raise the ceiling. Provider availability is
external waiting; a new exact-source VAST green and PR/CI acceptance remain
open. These admissions created no paid worker or storage.

The local static documentation/example, community, parity-citation,
owner-checklist, platform, ABI-coverage, workflow and diff-hygiene gates passed.
The workflow gate's first local invocation lacked the approved temporary UV
cache setting and failed before inspection; rerunning with that setting passed
all 48 workflow checks without changing a workflow or permission.

## Remaining gates

The [TorchAudio archive inspection](wespeaker-torchaudio211-static-facts-2026-10-14.md)
adds exact small-distribution LICENSE/RECORD facts, not whole-closure approval.
Manager byte-level review rejected the first ELF parser's false no-linkage
result; both libraries actually have PT_DYNAMIC and eight DT_NEEDED entries.
Corrected results and the rejected originals remain separate. This software-
only work adds no model, native-runtime, owner or public-row completion.

The WeSpeaker static collector is a temporary candidate, not acquired-wheel
evidence: its offline report has 13 `NOT_ACQUIRED` rows and one virtual row,
`BLOCKED_UNRESOLVED_EVIDENCE`, `NO_UPLOAD`, and null owner fields. Review fixes
cover streamed large native files, ELF32/64 linkage parsing, exact license-byte
retention, and a shared interrupting deadline. Manager stdlib-only self-test and
offline integration passed; live Linux fetch/native facts remain unmeasured.

Next: obtain exact-source remote green and PR acceptance when admission is
available, while preparing the separately bounded current dependency/API
work and only execution-approved real-weight CPU work. All 58 unresolved rows
remain accounted for; none is generically withheld merely for being difficult.
Final clean HEAD/Apple packet, Scaleway, and publication are still incomplete.
Local model execution and local workspace Cargo were not performed. The original
local-model safety guard remains unchanged. Free space was approximately 199 GiB
at the filesystem readback; no additional model was deleted by this work.
