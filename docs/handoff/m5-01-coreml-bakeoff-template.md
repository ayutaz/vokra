# M5-01 CoreML/ANE bakeoff report — template

**Owner-fillable template**. Copy this to a dated sibling (e.g.
`docs/handoff/m5-01-coreml-bakeoff-YYYY-MM-DD.md`) and populate every
`TBD` field from a real run of `tools/parity/npu_rtf_variance.sh
--backend coreml`. Do NOT edit this template in-place with real numbers
— the template is the fresh sheet for the *next* bakeoff.

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
| Owner | TBD (yousan?) |
| Device model | TBD (e.g. `Mac mini M4 Pro / iPhone 16 Pro / iPad Pro M4`) |
| SoC | TBD (e.g. `Apple M4 Pro, 12C/16GPU/16NE, 24 GB unified`) |
| Neural Engine generation | TBD (e.g. `16-core NE @ 38 TOPS`) |
| macOS / iOS version | TBD (e.g. `macOS 15.4 (24E248)` / `iOS 18.4 (22E237)`) |
| Xcode / CoreMLCompiler version | TBD (e.g. `Xcode 16.3 (16E140)`) |
| Thermal state at start | TBD (`nominal` / `fair` / `serious` / `critical` — read from `pmset -g therm` on macOS) |
| Battery / plugged in | TBD (bakeoff must be plugged in — battery power throttles the ANE) |

Notes on device selection (owner records why this rig was chosen):

> TBD (e.g. "M4 Pro chosen because M4 is the latest ANE generation
> shipping in 2025 devices; older M-series ANE numbers are recorded
> separately as historical baselines")

## 2. Baseline (M5-14-post CPU RTF)

Captured on the **same host in the same session** with `--backend cpu`
so the 2× ratio compares apples to apples (thermal state / background
load / macOS version are held constant).

```bash
./tools/parity/npu_rtf_variance.sh \
    --gguf   /path/to/whisper-large-v3.gguf \
    --audio  /path/to/jfk-30s.wav \
    --backend cpu \
    --iters 10 \
    --warmup 1 \
    --label  m5-14-post-cpu-baseline \
    --output rtf-cpu-baseline.jsonl

./tools/parity/npu_rtf_analyze.py rtf-cpu-baseline.jsonl \
    --output rtf-cpu-baseline.report.md
```

| field | value |
|---|---|
| GGUF | TBD (SHA256 recommended, e.g. `whisper-large-v3.gguf, sha256 2ebfc46a…`) |
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

## 3. CoreML/ANE run

Owner must wire up an ANE placement probe before running — Xcode
Instruments `MLModel` trace is the reference; see the Apple developer
docs on `MLModel` Metrics for the JSON export path. The probe should
emit `{"ane_frac": <0..1>, "gpu_frac": <0..1>, "cpu_frac": <0..1>}` to
stdout on each invocation; `npu_rtf_variance.sh` folds that JSON into
the per-iteration line.

```bash
./tools/parity/npu_rtf_variance.sh \
    --gguf   /path/to/whisper-large-v3.gguf \
    --audio  /path/to/jfk-30s.wav \
    --backend coreml \
    --iters 10 \
    --warmup 1 \
    --placement-probe /opt/probes/ane_placement.sh \
    --label  m5-01-coreml-ane \
    --output rtf-coreml.jsonl

./tools/parity/npu_rtf_analyze.py rtf-coreml.jsonl \
    --output rtf-coreml.report.md
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
| **NPU fraction (mean, ANE)** | TBD (must be ≥ 0.90 to record the mean) |
| NPU fraction (min, ANE) | TBD |
| Placement probe used | TBD (path to the shell wrapper around Xcode Instruments) |
| Analyzer placement verdict | TBD (`OK` / `WARN`) |
| JSONL artifact | TBD |
| Report artifact | TBD |

If the placement probe is not yet wired up:

> **STOP**. Per FR-EX-08 an NPU bakeoff without a placement probe is
> not a bakeoff — you are measuring `ANE || CPU-fallback` vs pure CPU,
> which is not the same experiment. Wire the probe up (or record the
> bakeoff as "insufficient tooling, deferred") before proceeding.

## 4. NFR-PF-12 verdict

Only fill this section if:
- (a) both §2 and §3 have `Analyzer CV verdict = OK` (or the owner has
  chosen to accept high-CV numbers with an explicit note explaining why
  a WARN is not fatal for this run), AND
- (b) §3 has `Analyzer placement verdict = OK` (≥ 90 % ANE placement).

If either condition fails, the verdict is **INSUFFICIENT DATA**; record
the reason and re-run.

| field | value |
|---|---|
| CPU baseline median RTF (§2) | TBD |
| CoreML median RTF (§3) | TBD |
| Speedup (CPU / CoreML) | TBD (compute: `CPU_median / CoreML_median`) |
| NFR-PF-12 threshold | 2.0 |
| **Verdict** | TBD (`PASS` / `FAIL` / `INSUFFICIENT DATA`) |
| Reason (if FAIL / INSUFFICIENT) | TBD |
| Feeds M5-13 T19 GO/NO-GO | TBD (`GO` = expose the delegate selector as a frozen C symbol; `NO-GO` = keep Rust-only per handoff m5-13.md §(c) T19) |

## 5. Rerun / defer conditions

| symptom | action |
|---|---|
| `Analyzer CV verdict = WARN` on both §2 and §3 | Re-run with `--iters 20` on a cooled-down box, plugged in, other workloads killed. If still WARN, defer with a note. |
| `placement < 0.90` on §3 | The delegate is silently falling back — inspect the Xcode Instruments MLModel trace to see which op(s). Report the op + shape to CC as an M5-01 follow-up ticket. Verdict: `INSUFFICIENT DATA`. |
| `Speedup < 2.0` cleanly | Verdict: `FAIL`. Feeds `NO-GO` for the M5-13 T19 C-ABI symbol call. NO-GO is recoverable post-GA via an additive MINOR bump (handoff m5-13.md §(c) T19), so this is not a v1.0 blocker — just a signal that the delegate is not ready for a frozen selector. |
| ANE not reachable / CoreML load fails | Bakeoff cannot fire. Record the failure (macOS version, CoreML SDK version, ONNX / mlmodel bundle path if any) and hand back to CC. |

## 6. Artifacts to commit

After the verdict is recorded:

- [ ] `rtf-cpu-baseline.jsonl` → `docs/bench-baselines/m5-01-coreml-bakeoff-YYYY-MM-DD/`
- [ ] `rtf-coreml.jsonl` → same directory
- [ ] `rtf-cpu-baseline.report.md` → same directory
- [ ] `rtf-coreml.report.md` → same directory
- [ ] filled-out copy of this template → `docs/handoff/m5-01-coreml-bakeoff-YYYY-MM-DD.md`
- [ ] `docs/m5-owner-verification-checklist.md` §1.5 checkbox tick

## 7. Cross-references

- Runbook: `docs/m5-owner-verification-checklist.md` §1.5
- Sister template: `docs/handoff/m5-02-qnn-bakeoff-template.md`
- Harness: `tools/parity/npu_rtf_variance.sh`
- Analyzer: `tools/parity/npu_rtf_analyze.py`
- Feeds: `docs/handoff/m5-13.md` §(c) T19 (C-ABI freeze GO/NO-GO)
- NFR-PF-12 protocol: `docs/system-requirements.md` (gitignored-local) /
  public glossary `docs/requirement-ids.md` NFR-PF-12
- Handoff sibling: `docs/handoff/m5-02.md` §"NFR-PF-12 baseline"
