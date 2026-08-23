# M5-01 CoreML/ANE bakeoff report — 2026-08-24

**Recorded verdict: FAIL / C ABI NO-GO.** The complete delegated Whisper
encoder reached 99.6281551% estimated ANE cost placement, but the same-session
median speedup was 1.422828x against the first-party Rust CPU encoder and the
FP16 output failed the fixed `atol = 0.01` CPU-oracle parity gate. The C ABI
therefore remains unchanged.

This dated report is the filled M5-01 sibling of
`m5-01-coreml-bakeoff-template.md`. It uses the newer exact-submodel harness
instead of the older whole-ASR RTF harness: the timed unit is precisely the
`WhisperEncoder` delegate boundary, while model load, audio decode, mel
generation, and the shared CPU decoder are excluded from both legs. CPU and
CoreML are sampled in alternating order in one process and one live model /
delegate session.

## 1. Hardware fingerprint

| field | value |
|---|---|
| Date (UTC) | 2026-08-23 |
| Local date / timezone | 2026-08-24 / Asia/Tokyo |
| Operator | Codex on the maintainer-owned host |
| Device model | iMac (`iMac21,1`) |
| SoC | Apple M1, 8 CPU cores (4 performance + 4 efficiency), 16 GB unified memory |
| Neural Engine | Apple M1 ANE; core count was not emitted by the bakeoff harness |
| OS | macOS 26.3 (25D125), Darwin 25.3.0 |
| Xcode / CoreML toolchain | Xcode 26.6 (17F113), coremltools 9.0 |
| Thermal state at start | unavailable: `pmset -g therm` returned IOKit error `0xe00002bc` |
| Power | AC power |

Device-selection note: this was the available maintainer Mac and provides real
Apple ANE silicon. It is an M1 historical baseline, not a claim about the
latest Apple Neural Engine generation.

## 2. Artifact identity and delegated boundary

| field | value |
|---|---|
| Model architecture | Whisper base |
| Source GGUF SHA-256 | `7e774425585d6e9ba58ac5337b406522b1408ab6ff6765d2938b092aef4c8e27` |
| CoreML FP16 tree SHA-256 | `477dd5393d57eb38139b88e3e48f79b7e92d8498a9b8b73d4862175cc6f4f7a9` |
| Input | `log_mel`, `[1, 80, 3000]` |
| Output | `encoder_hidden`, `[1, 1500, 512]` |
| CoreML precision | FP16 |
| Minimum deployment target | macOS 14 |
| Audio fixture | `tests/fixtures/audio/jfk-30s.wav` |
| Audio SHA-256 | `58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f` |
| Source base commit | `5d93d3daa533427b375a18930b72f006e820fcef` |

The sidecar manifest binds the source GGUF, compiled CoreML tree, feature
names, tensor shapes, precision, deployment target, and converter version.
Runtime loading verifies both hashes before `MLModel` creation and does not
fall back to CPU on a mismatch or CoreML failure.

## 3. Same-session CPU baseline

| field | value |
|---|---:|
| Build | release |
| Warmup | 2 alternating CPU/CoreML pairs |
| Measured iterations | 10 alternating pairs |
| Timed unit | complete Whisper encoder only |
| CPU implementation | first-party Rust CPU |
| Median | 1047.544000 ms |
| Mean | 1070.674571 ms |
| p95 | 1303.389500 ms |
| Standard deviation | 158.748191 ms |
| CV | 0.148269320 — OK (`<= 0.20`) |

## 4. CoreML/ANE run and placement

| field | value |
|---|---:|
| Median | 736.240541 ms |
| Mean | 784.899941 ms |
| p95 | 979.793417 ms |
| Standard deviation | 103.245346 ms |
| CV | 0.131539500 — OK (`<= 0.20`) |
| Placement probe | `tools/coreml/check_placement.sh` using CoreML `MLComputePlan` |
| Total CoreML operations | 388 |
| ANE-preferred operations | 151 |
| CPU-preferred operations | 2 (`ios17.cast`, `ios17.cast`) |
| Zero-cost constants / unknown | 235 |
| ANE operation-count fraction | 0.986928105 |
| **ANE estimated-cost fraction** | **0.996281551 — PASS (`>= 0.90`)** |

The placement denominator is estimated compute cost. Constants are reported by
CoreML without a preferred device but have zero estimated cost, so they do not
inflate or dilute the NPU fraction. Only two casts, totalling 0.3718449% of
estimated cost, prefer CPU.

## 5. Numerical parity

The first-party Rust CPU encoder is the independent oracle. The existing FP32
tolerance remains fixed; it was not relaxed to make the delegate pass.

| field | value |
|---|---:|
| Compared values | 7,680,000 |
| Absolute tolerance | 0.01 |
| Maximum absolute error | 7.176184654 |
| Mean absolute error | 0.009151214 |
| Values over tolerance | 2,081,830 (27.106% of compared values) |
| Max-error audio position / channel | 128 / 145 |
| CPU value at max error | 8.348059654 |
| CoreML value at max error | 1.171875000 |
| **Parity verdict** | **FAIL** |

The maximum is within the speech-bearing encoder region, not only padded
frames. A diagnostic FP32 CoreML model passes parity (`max_abs_error =
0.001298904`, zero values over tolerance) but has 0% ANE estimated-cost
placement. This isolates the incompatibility to the ANE-capable FP16 path; the
FP32 result is not an NPU performance result.

## 6. NFR-PF-12 2x verdict

| field | value |
|---|---:|
| CPU median | 1047.544000 ms |
| CoreML median | 736.240541 ms |
| Median speedup | 1.422828x |
| p95 speedup | 1.330270x |
| Threshold | 2.0x |
| Placement prerequisite | PASS |
| CV prerequisite | PASS for both legs |
| Speed verdict | **FAIL** |
| Combined parity + speed verdict | **FAIL** |

The encoder-only speedup is an upper bound on the hybrid full-ASR speedup. If
`C` is CPU encoder time, `A` is delegated encoder time, and `D >= 0` is the
unchanged CPU decoder / surrounding work, then for `C > A`:

```text
(C + D) / (A + D) <= C / A = 1.422828 < 2.0
```

Consequently, rerunning the whole-ASR RTF harness cannot reverse this clean
2x FAIL while the decoder and surrounding work remain shared. Whole-ASR RTF
may still be useful as a product baseline, but it is not needed to decide this
acceptance threshold.

## 7. C ABI decision

**CoreML delegate selector: NO-GO.** A frozen C selector must not expose a path
that fails the CPU-oracle parity gate and misses the 2x NFR. No CoreML or QNN
value or delegate-selection symbol is added to `include/vokra.h`.

This is recoverable after GA as an additive minor-version API once numerical
parity and performance meet the acceptance contract. The independent QNN audit
recorded `INSUFFICIENT DATA`, so the combined v1.0 NPU selector decision is also
NO-GO; see `m5-02-qnn-bakeoff-2026-08-24.md`.

## 8. Committed evidence

- `docs/bench-baselines/m5-01-coreml-bakeoff-2026-08-24/bakeoff.txt`
- `docs/bench-baselines/m5-01-coreml-bakeoff-2026-08-24/placement.txt`
- `docs/bench-baselines/m5-01-coreml-bakeoff-2026-08-24/fp32-diagnostic.txt`
- `docs/bench-baselines/m5-01-coreml-bakeoff-2026-08-24/README.md`

The local GGUF and compiled model are not repository artifacts; their
content-addressed hashes above are the evidence binding.
