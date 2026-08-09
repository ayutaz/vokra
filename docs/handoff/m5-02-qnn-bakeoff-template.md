# M5-02 QNN/Hexagon bakeoff report — template

**Owner-fillable template**. Copy this to a dated sibling (e.g.
`docs/handoff/m5-02-qnn-bakeoff-YYYY-MM-DD.md`) and populate every
`TBD` field from a real run of `tools/parity/npu_rtf_variance.sh
--backend qnn`. Do NOT edit this template in-place — the template is
the fresh sheet for the *next* bakeoff.

**Position in the plan** — this feeds `docs/m5-owner-verification-checklist.md`
§1.5 (NPU bakeoff runbook) which in turn feeds `docs/handoff/m5-13.md`
§(c) T19 (C-ABI freeze GO/NO-GO for the NPU delegate selector). The 2×
verdict recorded here is the input to that owner decision, not a
release-gate on its own.

**NFR-PF-12 protocol** (2026-08-09 codification): the CPU baseline for
the 2× ratio is **M5-14-post CPU** (SIMD hot-path optimised,
libm-route). An NPU RTF captured without a matched CPU baseline
collected on the same host in the same session **cannot** feed the 2×
verdict. Silent-CPU-fallback (placement < 90 %) **disqualifies** the
run — this is the FR-EX-08 hazard clause, not a soft warning.

---

## 1. Hardware fingerprint

| field | value |
|---|---|
| Date (UTC) | TBD |
| Owner | TBD |
| Device model | TBD (e.g. `Qualcomm QRD 8 Gen 3 devboard` / `Samsung Galaxy S24 Ultra` / `RB3 Gen 2`) |
| SoC | TBD (e.g. `Snapdragon 8 Gen 3 (SM8650), Adreno 750, HTP v75`) |
| Hexagon HTP generation | TBD (e.g. `Hexagon v75 @ 45 TOPS`) |
| Android / Linux version | TBD (e.g. `Android 14 (UP1A.231005.007)` / `Ubuntu 22.04 LTS on Debian devroot`) |
| QNN SDK version | TBD (e.g. `qnn-2.24.0.240626` — `qnn-net-run --version`) |
| Thermal state at start | TBD (`nominal` — Snapdragon reports via `getprop persist.vendor.thermal.status` on Android, or `/sys/class/thermal/thermal_zone*/temp` on Linux) |
| Battery / plugged in / active cooling | TBD (bakeoff must be plugged in, active cooling on if the devboard has a fan — Snapdragon throttles aggressively) |

Notes on device selection (owner records why this rig was chosen):

> TBD (e.g. "8 Gen 3 devboard chosen because HTP v75 is the latest
> shipping in 2025 phones; older Snapdragon numbers are recorded as
> historical baselines")

## 2. Baseline (M5-14-post CPU RTF)

Captured on the **same host in the same session** with `--backend cpu`
so the 2× ratio compares apples to apples (thermal state / big-core
availability / cpufreq policy are held constant).

```bash
./tools/parity/npu_rtf_variance.sh \
    --gguf   /data/local/tmp/whisper-large-v3.gguf \
    --audio  /data/local/tmp/jfk-30s.wav \
    --backend cpu \
    --iters 10 \
    --warmup 1 \
    --label  m5-14-post-cpu-baseline \
    --output /data/local/tmp/rtf-cpu-baseline.jsonl

./tools/parity/npu_rtf_analyze.py /data/local/tmp/rtf-cpu-baseline.jsonl \
    --output /data/local/tmp/rtf-cpu-baseline.report.md
```

On Android devices the invocation is via `adb shell` — the
harness itself is portable bash + python3, so no port required.

| field | value |
|---|---|
| GGUF | TBD (SHA256 recommended) |
| Audio fixture | TBD (e.g. `jfk-30s.wav 16 kHz mono PCM16`) |
| N (iters) | TBD (default 10, extend if CV > 0.20) |
| mean RTF | TBD |
| median RTF | TBD |
| CV | TBD (must be ≤ 0.20 to record the mean without WARN) |
| p95 RTF | TBD |
| p99 RTF | TBD |
| Analyzer CV verdict | TBD (`OK` / `WARN`) |
| JSONL artifact | TBD (path in `docs/bench-baselines/…`) |
| Report artifact | TBD (path in `docs/bench-baselines/…`) |

## 3. QNN/HTP run

Owner must wire up an HTP placement probe before running —
`qnn-net-run --profiling_option=op --profiling_level=basic` is the
reference; parse its `op stats` output into
`{"htp_frac": <0..1>, "cpu_frac": <0..1>}` and expose that as the
probe. Older QNN profiler dumps use `dsp_frac` — the analyzer accepts
either key for back-compat.

```bash
./tools/parity/npu_rtf_variance.sh \
    --gguf   /data/local/tmp/whisper-large-v3.gguf \
    --audio  /data/local/tmp/jfk-30s.wav \
    --backend qnn \
    --iters 10 \
    --warmup 1 \
    --placement-probe /data/local/tmp/htp_placement.sh \
    --label  m5-02-qnn-htp \
    --output /data/local/tmp/rtf-qnn.jsonl

./tools/parity/npu_rtf_analyze.py /data/local/tmp/rtf-qnn.jsonl \
    --output /data/local/tmp/rtf-qnn.report.md
```

| field | value |
|---|---|
| N (iters) | TBD |
| mean RTF | TBD |
| median RTF | TBD |
| CV | TBD |
| p95 RTF | TBD |
| p99 RTF | TBD |
| Analyzer CV verdict | TBD (`OK` / `WARN`) |
| **NPU fraction (mean, HTP)** | TBD (must be ≥ 0.90 to record the mean) |
| NPU fraction (min, HTP) | TBD |
| Placement probe used | TBD (path to the qnn-net-run wrapper) |
| Analyzer placement verdict | TBD (`OK` / `WARN`) |
| JSONL artifact | TBD |
| Report artifact | TBD |

If the placement probe is not yet wired up:

> **STOP**. Per FR-EX-08 an NPU bakeoff without a placement probe is
> not a bakeoff — you are measuring `HTP || CPU-fallback` vs pure CPU,
> which is not the same experiment. Wire the probe up (or record the
> bakeoff as "insufficient tooling, deferred") before proceeding.

## 4. NFR-PF-12 verdict

Only fill this section if:
- (a) both §2 and §3 have `Analyzer CV verdict = OK` (or the owner has
  chosen to accept high-CV numbers with an explicit note explaining why
  a WARN is not fatal for this run), AND
- (b) §3 has `Analyzer placement verdict = OK` (≥ 90 % HTP placement).

If either condition fails, the verdict is **INSUFFICIENT DATA**; record
the reason and re-run.

| field | value |
|---|---|
| CPU baseline median RTF (§2) | TBD |
| QNN median RTF (§3) | TBD |
| Speedup (CPU / QNN) | TBD (compute: `CPU_median / QNN_median`) |
| NFR-PF-12 threshold | 2.0 |
| **Verdict** | TBD (`PASS` / `FAIL` / `INSUFFICIENT DATA`) |
| Reason (if FAIL / INSUFFICIENT) | TBD |
| Feeds M5-13 T19 GO/NO-GO | TBD (`GO` = expose the delegate selector as a frozen C symbol; `NO-GO` = keep Rust-only per handoff m5-13.md §(c) T19) |

## 5. Rerun / defer conditions

| symptom | action |
|---|---|
| `Analyzer CV verdict = WARN` on both §2 and §3 | Re-run with `--iters 20` after a thermal cooldown, active cooling on, other workloads killed. Snapdragon CV is inherently higher than Apple M-series; a `CV = 0.15` on HTP is normal, `> 0.20` calls for more iters. |
| `placement < 0.90` on §3 | The delegate is silently falling back — inspect the `qnn-net-run --profiling_option=op` dump to identify which op(s). Report the op + shape to CC as an M5-02 follow-up ticket. Verdict: `INSUFFICIENT DATA`. |
| `Speedup < 2.0` cleanly | Verdict: `FAIL`. Feeds `NO-GO` for the M5-13 T19 C-ABI symbol call. NO-GO is recoverable post-GA via an additive MINOR bump (handoff m5-13.md §(c) T19), so this is not a v1.0 blocker — just a signal that the delegate is not ready for a frozen selector. |
| HTP unreachable / QNN backend load fails | Bakeoff cannot fire. Record the failure (Android SELinux denials, `qnn-net-run` diagnostics, `libQnnHtp.so` presence, SDK vs firmware compatibility) and hand back to CC. |
| Only `dsp_frac` reported (no `htp_frac`) | Legacy QNN profiler — the analyzer accepts `dsp_frac` as an alias for back-compat, so this is not itself a bakeoff blocker. Note the profiler version so a future upgrade is not confused by the alias fallthrough. |

## 6. Artifacts to commit

After the verdict is recorded:

- [ ] `rtf-cpu-baseline.jsonl` → `docs/bench-baselines/m5-02-qnn-bakeoff-YYYY-MM-DD/`
- [ ] `rtf-qnn.jsonl` → same directory
- [ ] `rtf-cpu-baseline.report.md` → same directory
- [ ] `rtf-qnn.report.md` → same directory
- [ ] filled-out copy of this template → `docs/handoff/m5-02-qnn-bakeoff-YYYY-MM-DD.md`
- [ ] `docs/m5-owner-verification-checklist.md` §1.5 checkbox tick

## 7. Cross-references

- Runbook: `docs/m5-owner-verification-checklist.md` §1.5
- Sister template: `docs/handoff/m5-01-coreml-bakeoff-template.md`
- Harness: `tools/parity/npu_rtf_variance.sh`
- Analyzer: `tools/parity/npu_rtf_analyze.py`
- Feeds: `docs/handoff/m5-13.md` §(c) T19 (C-ABI freeze GO/NO-GO)
- Priors: `docs/handoff/m5-02.md` (spec + NFR-PF-12 baseline discussion)
- NFR-PF-12 protocol: `docs/system-requirements.md` (gitignored-local) /
  public glossary `docs/requirement-ids.md` NFR-PF-12
