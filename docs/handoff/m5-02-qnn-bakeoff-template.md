# M5-02 QNN/Hexagon bakeoff report — template

**Owner-fillable template.** Copy this to a dated sibling (for example
`docs/handoff/m5-02-qnn-bakeoff-YYYY-MM-DD.md`) and populate every `TBD`
from a real owner-side run. Do not edit this template in place. The shared
rules, including the distinction between the exact encoder gate and the
legacy all-model RTF harness, are in
[`npu-bakeoff-protocol.md`](npu-bakeoff-protocol.md); their outputs must not
be combined as one experiment.

**Position in the plan** — this feeds
`docs/m5-owner-verification-checklist.md` §1.5, which in turn feeds
`docs/handoff/m5-13.md` §(c) T19 (C-ABI freeze GO/NO-GO). The recorded 2×
verdict is input to that owner decision, not a release gate on its own.

The QNN examples in §2 and §3 use the legacy product-RTF path. The exact
`vokra-cli npu-bakeoff --delegate qnn` surface is separate and currently
returns an explicit unsupported error until whole-encoder QNN execution and
its real SDK/runtime contract exist. Do not substitute a CPU result.

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
| Battery / plugged in / active cooling | TBD (bakeoff must be plugged in; active cooling on if available) |

Notes on device selection (owner records why this rig was chosen):

> TBD (e.g. "8 Gen 3 devboard chosen because HTP v75 is the latest
> shipping in 2025 phones; older Snapdragon numbers are recorded as
> historical baselines")

## 2. Baseline (M5-14-post CPU RTF)

For the legacy RTF lane, capture the M5-14-post CPU leg on the same host and
in the same session as §3. Hold thermal state, big-core availability,
cpufreq policy, and background load constant. Use the shared protocol for the
exact encoder gate and do not merge the two lanes.

```bash
uv run --no-project --python 3.12 bash ./tools/parity/npu_rtf_variance.sh \
    --gguf   /data/local/tmp/whisper-large-v3.gguf \
    --audio  /data/local/tmp/jfk-30s.wav \
    --backend cpu \
    --iters 10 \
    --warmup 1 \
    --label  m5-14-post-cpu-baseline \
    --output /data/local/tmp/rtf-cpu-baseline.jsonl

uv run --no-project --python 3.12 python -B ./tools/parity/npu_rtf_analyze.py /data/local/tmp/rtf-cpu-baseline.jsonl \
    --output /data/local/tmp/rtf-cpu-baseline.report.md
```

On Android, the owner invokes the portable harness through `adb shell` on
the target's prepared Python 3.12 environment; this is not a maintainer-Mac
run.

| field | value |
|---|---|
| GGUF | TBD (SHA256 recommended) |
| Audio fixture | TBD (e.g. `jfk-30s.wav 16 kHz mono PCM16`) |

In the dated copy, add the shared CPU/delegate RTF metric block from
[`npu-bakeoff-protocol.md` §4](npu-bakeoff-protocol.md#4-shared-rtf-metric-block-and-decision-inputs)
and link both raw JSONL and analyzer report artifacts.

## 3. QNN/HTP run

Wire and validate an HTP placement probe before timing. The owner supplies a
`qnn-net-run --profiling_option=op --profiling_level=basic` wrapper that emits
`{"htp_frac": <0..1>, "cpu_frac": <0..1>}`. Older profiler dumps may emit
`dsp_frac`; the analyzer accepts that alias for back-compat, but it does not
change the placement rule. A missing or invalid probe is `INSUFFICIENT DATA`.

```bash
uv run --no-project --python 3.12 bash ./tools/parity/npu_rtf_variance.sh \
    --gguf   /data/local/tmp/whisper-large-v3.gguf \
    --audio  /data/local/tmp/jfk-30s.wav \
    --backend qnn \
    --iters 10 \
    --warmup 1 \
    --placement-probe /data/local/tmp/htp_placement.sh \
    --label  m5-02-qnn-htp \
    --output /data/local/tmp/rtf-qnn.jsonl

uv run --no-project --python 3.12 python -B ./tools/parity/npu_rtf_analyze.py /data/local/tmp/rtf-qnn.jsonl \
    --output /data/local/tmp/rtf-qnn.report.md
```

| field | value |
|---|---|
| **NPU fraction (mean, HTP)** | TBD (must be ≥ 0.90) |
| NPU fraction (min, HTP) | TBD |
| Placement probe used | TBD (path to the `qnn-net-run` wrapper) |
| Legacy placement key, if used | TBD (`dsp_frac` only when emitted by the profiler) |
| Analyzer placement verdict | TBD (`OK` / `WARN`) |

## 4. NFR-PF-12 verdict

Apply the shared protocol's acceptance and decision rules. For this QNN
legacy lane, a numeric 2× record requires acceptable CV evidence for §2 and
§3 and `Analyzer placement verdict = OK` (≥ 90% HTP). A missing probe,
placement failure, or unresolved noisy run is **INSUFFICIENT DATA**.

| field | value |
|---|---|
| CPU baseline median RTF (§2) | TBD |
| QNN median RTF (§3) | TBD |
| Speedup (CPU / QNN) | TBD (compute: `CPU_median / QNN_median`) |
| NFR-PF-12 threshold | 2.0 |
| **Verdict** | TBD (`PASS` / `FAIL` / `INSUFFICIENT DATA`) |
| Reason (if FAIL / INSUFFICIENT) | TBD |
| Feeds M5-13 T19 GO/NO-GO | TBD (`GO` = expose the delegate selector as a frozen C symbol; `NO-GO` = keep Rust-only per `m5-13.md` §(c) T19) |

## 5. Rerun / defer conditions

Apply the shared protocol's rerun/defer matrix. QNN-specific follow-up:

| symptom | action |
|---|---|
| `placement < 0.90` on §3 | Inspect the `qnn-net-run --profiling_option=op` dump, report the failing operation and shape, and record `INSUFFICIENT DATA` as an M5-02 follow-up. |
| HTP unreachable / QNN backend load fails | Record Android SELinux denials, `qnn-net-run` diagnostics, `libQnnHtp.so` presence, and SDK/firmware compatibility; the bakeoff is `INSUFFICIENT DATA`. |
| Only `dsp_frac` is reported | Preserve the profiler version and use the analyzer alias; do not treat the alias as evidence that a QNN graph executed. |

## 6. Artifacts to commit

Follow the shared evidence boundary and retain the QNN-specific rows:

- [ ] `rtf-cpu-baseline.jsonl` → `docs/bench-baselines/m5-02-qnn-bakeoff-YYYY-MM-DD/`
- [ ] `rtf-qnn.jsonl` → same directory
- [ ] `rtf-cpu-baseline.report.md` → same directory
- [ ] `rtf-qnn.report.md` → same directory
- [ ] HTP placement/profiler export, including SDK/profiler version → same dated evidence set
- [ ] filled-out copy of this template → `docs/handoff/m5-02-qnn-bakeoff-YYYY-MM-DD.md`
- [ ] `docs/m5-owner-verification-checklist.md` §1.5 checkbox tick

## 7. Cross-references

- Shared protocol: `docs/handoff/npu-bakeoff-protocol.md`
- Runbook: `docs/m5-owner-verification-checklist.md` §1.5
- Sister template: `docs/handoff/m5-01-coreml-bakeoff-template.md`
- Harness: `tools/parity/npu_rtf_variance.sh`
- Analyzer: `tools/parity/npu_rtf_analyze.py`
- Exact encoder gate: `vokra-cli npu-bakeoff` / `crates/vokra-cli/src/npu_bakeoff.rs`
- Feeds: `docs/handoff/m5-13.md` §(c) T19 (C-ABI freeze GO/NO-GO)
- Priors: `docs/handoff/m5-02.md` (spec + NFR-PF-12 baseline discussion)
- NFR-PF-12: public glossary `docs/requirement-ids.md`
