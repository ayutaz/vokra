# M5-01 CoreML/ANE bakeoff report — template

**Owner-fillable template.** Copy this to a dated sibling (for example
`docs/handoff/m5-01-coreml-bakeoff-YYYY-MM-DD.md`) and populate every `TBD`
from a real owner-side run. Do not edit this template in place. The shared
rules, including the distinction between the exact encoder gate and the
legacy all-model RTF harness, are in
[`npu-bakeoff-protocol.md`](npu-bakeoff-protocol.md); their outputs must not
be combined as one experiment.

**Position in the plan** — this feeds
`docs/m5-owner-verification-checklist.md` §1.5, which in turn feeds
`docs/handoff/m5-13.md` §(c) T19 (C-ABI freeze GO/NO-GO). The recorded 2×
verdict is input to that owner decision, not a release gate on its own.

The command examples in §2 and §3 show the legacy product-RTF path using
`tools/parity/npu_rtf_variance.sh`. For the current exact Whisper encoder
gate, use `vokra-cli npu-bakeoff` as specified in the shared protocol and
record its parity/speed output separately from these all-model RTF rows.

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

For the legacy RTF lane, capture the M5-14-post CPU leg on the same host and
in the same session as §3. Hold thermal state, OS, and background load
constant. Use the shared protocol for the exact encoder gate's in-process
CPU leg and do not merge the two lanes.

```bash
uv run --no-project --python 3.12 bash ./tools/parity/npu_rtf_variance.sh \
    --gguf   /path/to/whisper-large-v3.gguf \
    --audio  /path/to/jfk-30s.wav \
    --backend cpu \
    --iters 10 \
    --warmup 1 \
    --label  m5-14-post-cpu-baseline \
    --output rtf-cpu-baseline.jsonl

uv run --no-project --python 3.12 python -B ./tools/parity/npu_rtf_analyze.py rtf-cpu-baseline.jsonl \
    --output rtf-cpu-baseline.report.md
```

| field | value |
|---|---|
| GGUF | TBD (SHA256 recommended, e.g. `whisper-large-v3.gguf, sha256 2ebfc46a…`) |
| Audio fixture | TBD (e.g. `jfk-30s.wav 16 kHz mono PCM16`) |

In the dated copy, add the shared CPU/delegate RTF metric block from
[`npu-bakeoff-protocol.md` §4](npu-bakeoff-protocol.md#4-shared-rtf-metric-block-and-decision-inputs)
and link both raw JSONL and analyzer report artifacts.

## 3. CoreML/ANE run

Wire and validate an ANE placement probe before timing. The owner supplies
the CoreML `MLComputePlan`/Instruments evidence; the legacy harness probe must
emit `{"ane_frac": <0..1>, "gpu_frac": <0..1>, "cpu_frac": <0..1>}` on
each invocation. A missing or invalid probe is `INSUFFICIENT DATA`, not an
inferred 100% ANE result.

```bash
uv run --no-project --python 3.12 bash ./tools/parity/npu_rtf_variance.sh \
    --gguf   /path/to/whisper-large-v3.gguf \
    --audio  /path/to/jfk-30s.wav \
    --backend coreml \
    --iters 10 \
    --warmup 1 \
    --placement-probe /opt/probes/ane_placement.sh \
    --label  m5-01-coreml-ane \
    --output rtf-coreml.jsonl

uv run --no-project --python 3.12 python -B ./tools/parity/npu_rtf_analyze.py rtf-coreml.jsonl \
    --output rtf-coreml.report.md
```

| field | value |
|---|---|
| **NPU fraction (mean, ANE)** | TBD (must be ≥ 0.90) |
| NPU fraction (min, ANE) | TBD |
| Placement probe used | TBD (path to the shell wrapper around Xcode Instruments) |
| Analyzer placement verdict | TBD (`OK` / `WARN`) |

## 4. NFR-PF-12 verdict

Apply the shared protocol's acceptance and decision rules. For this CoreML
legacy lane, a numeric 2× record requires both §2 and §3 to have acceptable CV
evidence and §3 to have `Analyzer placement verdict = OK` (≥ 90% ANE). A
missing probe, placement failure, or unresolved noisy run is
**INSUFFICIENT DATA**.

| field | value |
|---|---|
| CPU baseline median RTF (§2) | TBD |
| CoreML median RTF (§3) | TBD |
| Speedup (CPU / CoreML) | TBD (compute: `CPU_median / CoreML_median`) |
| NFR-PF-12 threshold | 2.0 |
| **Verdict** | TBD (`PASS` / `FAIL` / `INSUFFICIENT DATA`) |
| Reason (if FAIL / INSUFFICIENT) | TBD |
| Feeds M5-13 T19 GO/NO-GO | TBD (`GO` = expose the delegate selector as a frozen C symbol; `NO-GO` = keep Rust-only per `m5-13.md` §(c) T19) |

The exact encoder gate additionally requires its fixed CPU-oracle parity bound
and reports a separate p50 speed verdict; record that output as its own
evidence lane, never as an all-model RTF result.

## 5. Rerun / defer conditions

Apply the shared protocol's rerun/defer matrix. CoreML-specific follow-up:

| symptom | action |
|---|---|
| `placement < 0.90` on §3 | Inspect the Xcode Instruments/MLComputePlan trace, report the failing operation and shape, and record `INSUFFICIENT DATA` as an M5-01 follow-up. |
| ANE not reachable / CoreML load fails | Record the macOS version, CoreML SDK version, model/compiled-tree path, and exact failure; the bakeoff is `INSUFFICIENT DATA`. |

## 6. Artifacts to commit

Follow the shared evidence boundary and retain the CoreML-specific rows:

- [ ] `rtf-cpu-baseline.jsonl` → `docs/bench-baselines/m5-01-coreml-bakeoff-YYYY-MM-DD/`
- [ ] `rtf-coreml.jsonl` → same directory
- [ ] `rtf-cpu-baseline.report.md` → same directory
- [ ] `rtf-coreml.report.md` → same directory
- [ ] CoreML placement export/probe evidence → same dated evidence set
- [ ] filled-out copy of this template → `docs/handoff/m5-01-coreml-bakeoff-YYYY-MM-DD.md`
- [ ] `docs/m5-owner-verification-checklist.md` §1.5 checkbox tick

## 7. Cross-references

- Shared protocol: `docs/handoff/npu-bakeoff-protocol.md`
- Runbook: `docs/m5-owner-verification-checklist.md` §1.5
- Sister template: `docs/handoff/m5-02-qnn-bakeoff-template.md`
- Harness: `tools/parity/npu_rtf_variance.sh`
- Analyzer: `tools/parity/npu_rtf_analyze.py`
- Exact encoder gate: `vokra-cli npu-bakeoff` / `crates/vokra-cli/src/npu_bakeoff.rs`
- Feeds: `docs/handoff/m5-13.md` §(c) T19 (C-ABI freeze GO/NO-GO)
- NFR-PF-12: public glossary `docs/requirement-ids.md`
- Handoff sibling: `docs/handoff/m5-02.md` §"NFR-PF-12 baseline"
