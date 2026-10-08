# NPU bakeoff protocol — shared rules

Shared rules for the owner-fillable [M5-01 CoreML/ANE
template](m5-01-coreml-bakeoff-template.md) and [M5-02 QNN/Hexagon
template](m5-02-qnn-bakeoff-template.md). The templates retain their
device-, SDK-, placement-, and delegate-specific fields. Copy a template to
a dated sibling; never put real measurements in the template itself.

## 1. Two different measurement paths

These paths are complementary but are **not one experiment**. Never combine
their numbers.

### 1.1 Exact Whisper encoder gate (`npu-bakeoff`)

`vokra-cli npu-bakeoff` is an encoder-only, same-process/same-session gate.
Its command contract, fixed parity/speed defaults, and no-fallback behavior
are in the [CLI tutorial §6](../tutorials/cli.md#6-npu-bakeoff--same-session-delegate-gate-experimental).
Keep its unchanged output, delegate placement evidence, and artifact identity
together in the dated report. The 2026-08-24 [CoreML report](m5-01-coreml-bakeoff-2026-08-24.md)
is the precedent; it is not an all-model ANE/HTP qualification or an
independent upstream/PyTorch parity result. QNN remains explicitly
unsupported until whole-encoder execution exists.

### 1.2 Legacy all-model RTF harness

`tools/parity/npu_rtf_variance.sh` runs one fresh `vokra-cli bench` process
per iteration; `npu_rtf_analyze.py` reports its RTF statistics, CV, and
placement warnings. Its parser accepts `cpu`, `coreml`, and `qnn` labels, but
an accepted label is not proof of an executable whole-model delegate path;
record backend/runtime failures in the dated report. This harness is not
same-session encoder evidence.

## 2. Shared acceptance rules for the legacy RTF lane

| Input | Rule | Evidence |
|---|---|---|
| CPU baseline | M5-14-post CPU (SIMD hot-path optimized, libm route), same host/session as delegate | Harness output and hardware fingerprint |
| Session discipline | Hold model, input, OS/driver, thermal state, and relevant load constant | Harness trailer and fingerprint |
| Variation | `CV <= 0.20` is the stable boundary | Analyzer report for each leg |
| Placement | Target NPU fraction ≥ `0.90`; missing/invalid placement is not a pass | CoreML ANE or QNN HTP evidence and analyzer output |
| Performance | CPU median/p50 ÷ delegate median/p50 against `2.0` | Dated report; analyzer never promotes the threshold |
| Fallback | Unsupported work is an explicit error; `NPU || CPU-fallback` is not accepted | Runtime error or placement evidence |

`CV > 0.20` is a `WARN`, not silent acceptance. For the legacy RTF verdict,
the owner may accept a WARN only with an explicit explanation of why it is not
fatal for that run, while preserving the raw evidence; otherwise rerun or
defer. Placement below `0.90` or an absent probe is `INSUFFICIENT DATA`; a
clean speed result below `2.0` is `FAIL` and feeds the documented C-ABI
`NO-GO`. That NO-GO is recoverable post-GA through an additive MINOR version
bump and is not itself a v1.0 blocker. Numerical parity is a
delegate-specific gate with its own oracle; this generic RTF harness does not
invent one. For QNN, `htp_frac` is preferred and legacy `dsp_frac` is an
accepted analyzer alias only; it does not prove graph execution.

## 3. Preparation and capture

1. Record owner-supplied device, OS, SDK/runtime, thermal/power state, model,
   input fixture, and hashes in the applicable template.
2. Copy the template, preserving its filename and section anchors:

   ```sh
   cp docs/handoff/m5-01-coreml-bakeoff-template.md \
      docs/handoff/m5-01-coreml-bakeoff-YYYY-MM-DD.md
   # or the corresponding m5-02-qnn-bakeoff-* names
   ```

3. Validate the delegate-specific placement probe before timing. Missing
   probe means stop/defer. Capture the matched CPU baseline and delegate leg;
   do not mix legacy all-model RTF with exact encoder output.
4. Run the analyzer on the target/remote environment with Python 3.12 via
   `uv` (not bare `python`, `python3`, pip, or conda):

   ```sh
   uv run --no-project --python 3.12 python -B \
     tools/parity/npu_rtf_analyze.py rtf.jsonl --output rtf.report.md
   ```

These are owner-side hardware measurements. Do not download, load, run,
convert, or benchmark model artifacts on the maintainer Mac.

## 4. Shared RTF metric block and decision inputs

For a dated report, copy this block and fill it from the analyzer. This keeps
the common CPU/delegate table in one place while retaining raw attachments.

| field | CPU baseline | delegate |
|---|---|---|
| N (requested / successful / failed) | TBD | TBD |
| mean RTF | TBD | TBD |
| median RTF | TBD | TBD |
| CV | TBD (≤ `0.20`) | TBD (≤ `0.20`) |
| p95 RTF | TBD | TBD |
| p99 RTF | TBD | TBD |
| Analyzer CV verdict | TBD (`OK` / `WARN`) | TBD (`OK` / `WARN`) |
| JSONL artifact | TBD | TBD |
| Report artifact | TBD | TBD |

Also record artifact/fixture hashes, placement probe/version, valid fraction
count, target-NPU mean/minimum, final `PASS`/`FAIL`/`INSUFFICIENT DATA` and
the reason, plus the M5-13 T19 GO/NO-GO input. For the exact encoder gate,
attach its unchanged command output, placement evidence, and bound artifact
record instead of copying this RTF block into its result.

## 5. Rerun and defer matrix

| Condition | Required action |
|---|---|
| Both legacy legs have `CV > 0.20` | Cool down, keep conditions fixed, rerun with at least 20 iterations; if still noisy, defer unless the owner records the explicit WARN acceptance reason in the dated report |
| Placement missing/invalid/`< 0.90` | Inspect trace/profiler, report operation and shape, record `INSUFFICIENT DATA`; do not calculate the ratio |
| Clean speedup `< 2.0` | Record `FAIL`/`NO-GO`; do not broaden scope to reverse it |
| Delegate/runtime/device unavailable | Record exact prerequisite failure as `INSUFFICIENT DATA`; never substitute CPU |
| Only `dsp_frac` is emitted | Preserve profiler version and analyzer alias; placement rule is unchanged |

## 6. Evidence and publication boundary

Commit the raw CPU/delegate JSONL, analyzer reports, filled dated report, and
delegate-specific placement/parity artifacts under the paths named by the
applicable template. Evidence does not publish a model, freeze a C symbol, or
authorize a public artifact update; those remain owner/legal decisions.

## 7. References

- Owner runbook: [`m5-owner-verification-checklist.md` §1.5](../m5-owner-verification-checklist.md)
- Requirement: [`requirement-ids.md` NFR-PF-12](../requirement-ids.md)
- Exact gate: [`npu_bakeoff.rs`](../../crates/vokra-cli/src/npu_bakeoff.rs)
- Legacy harness/analyzer: [`npu_rtf_variance.sh`](../../tools/parity/npu_rtf_variance.sh), [`npu_rtf_analyze.py`](../../tools/parity/npu_rtf_analyze.py)
- C-ABI decision: [`m5-13.md` §(c) T19](m5-13.md#t19--gono-go-on-the-c-export-candidates-freeze-surface-decision)
