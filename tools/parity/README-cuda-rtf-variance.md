# CUDA large-v3 RTF variance analysis harness

Reference-measurement harness that samples the Whisper large-v3 CUDA RTF `N`
times on a real GPU and reduces the collected JSONL into a markdown report
with **mean / median / stddev / CV / p50 / p95 / p99 / min / max /
histogram**. Complements the single-shot sanity gate in
[`crates/vokra-backend-cuda/tests/whisper_cuda_large_v3_rtf.rs`](../../crates/vokra-backend-cuda/tests/whisper_cuda_large_v3_rtf.rs)
by adding statistical rigor over multiple iterations without pretending to
be an always-on gate.

**Position in the plan** — this is the *variance analysis* rung. The formal
`RTF < 0.10` always-on gate lives at:

- **M2-14** — owner self-hosted CUDA runner (docs: [`docs/m2-owner-verification-checklist.md`](../../docs/m2-owner-verification-checklist.md) §2)
- **M3-01** — 5% regression gate

See [`docs/adr/M2-03-followup-rtf.md`](../../docs/adr/M2-03-followup-rtf.md)
§D6 for the red-line: **this harness never asserts an RTF ceiling and never
promotes any threshold**. All promote/defer decisions are the owner's after
they inspect the report.

## Files

| File | Purpose |
|------|---------|
| [`cuda_rtf_variance.sh`](cuda_rtf_variance.sh) | Bash orchestration — loops `vokra-cli bench` N times, emits JSONL |
| [`cuda_rtf_analyze.py`](cuda_rtf_analyze.py) | Python analyzer (stdlib only) — JSONL → markdown or JSON report |
| [`README-cuda-rtf-variance.md`](README-cuda-rtf-variance.md) | (this file) |
| [`../../docs/m2-cuda-rtf-variance-template.md`](../../docs/m2-cuda-rtf-variance-template.md) | Report template for the owner to fill in after their vast.ai run |

## Zero-dep + NVIDIA EULA red-lines

The harness runs on a stock `bash` + `python3` (stdlib) + `nvidia-smi`
(optional metadata) host. It does **not**:

- `pip install` any package (numpy / pandas / matplotlib all forbidden — see NFR-DS-02)
- `apt install` cuDNN / cuBLAS / cuFFT / cudart
- Bundle or link any NVIDIA runtime library

The `vokra-cli` binary discovers CUDA at runtime via
`dlopen("libcuda.so.1")` + `dlopen("libnvrtc.so.12")` (raw FFI, EULA install
model). Per [`docs/adr/M2-03-followup-rtf.md`](../../docs/adr/M2-03-followup-rtf.md)
§D8 this is the *only* CUDA linkage in the whole tree.

## Prerequisites

- **CUDA host** — any Linux box with an NVIDIA GPU (Ampere or newer,
  `d_head=64` Whisper is FA v2-capable on RTX 30 / 40 / A100 / H100). The
  reference measurement in
  [`docs/bench-baselines/whisper_large_v3_cuda_rtf.json`](../../docs/bench-baselines/whisper_large_v3_cuda_rtf.json)
  is a vast.ai RTX 4090 spot instance (`nvidia/cuda:12.6.2-devel-ubuntu22.04`).
- **NVIDIA driver + CUDA toolkit** — driver `>= 555` and toolkit `>= 12.4`
  (matches the M2-03 follow-up baseline). Newer is fine; older may fail
  NVRTC compile.
- **Rust `1.86+`** — same toolchain the baseline was collected with.
- **Whisper large-v3 GGUF** — converted from the HF safetensors by
  `vokra-cli convert --model whisper --input model.safetensors --output whisper-large-v3.gguf`.
  Verify `sha256sum whisper-large-v3.gguf ==`
  `2ebfc46a95ad3831377ae5f4d9d30e35dd2d87fb0526769a02f78b237d30e761`
  to make sure the harness measures the same GGUF the baseline used.
- **A 30 s mono 16 kHz WAV** — Whisper always processes a 30 s window
  internally; if the input WAV is shorter or longer the RTF denominator
  (`audio_seconds`) is the raw WAV length, so the number is only
  comparable to the baseline when the WAV is exactly 30 s. Any real
  speech clip works; the baseline uses `tests/fixtures/audio/jfk-30s.wav`
  when available or a synthetic tone (see the sanity test's
  `make_sample_pcm()`).

## Usage

### 1. Build `vokra-cli` on the CUDA host

```bash
cargo build --release -p vokra-cli
# ~5 minutes cold on a vast.ai spot
```

### 2. Run the harness

```bash
./tools/parity/cuda_rtf_variance.sh \
    --gguf   /root/whisper-large-v3.gguf \
    --audio  /root/jfk-30s.wav \
    --iters  10 \
    --warmup 1 \
    --fa-v2  off \
    --label  decomposed \
    --output /root/rtf-decomposed.jsonl
```

Options (see `--help`):

| Flag | Default | Notes |
|------|---------|-------|
| `--gguf PATH` | required | Whisper large-v3 GGUF |
| `--audio PATH` | required | 30 s mono 16 kHz WAV |
| `--iters N` | 10 | Timed steady-state iterations (≥ 1) |
| `--warmup M` | 1 | Untimed warmup iterations per invocation |
| `--fa-v2 on\|off` | `on` | `off` sets `VOKRA_CUDA_DISABLE_FA_V2=1` (decomposed path); `on` uses the default FA v2 gated wrapper (t_q ≥ 16) |
| `--backend NAME` | `cuda` | `cuda` / `metal` / `cpu` — anything but `cuda` is for smoke-testing only |
| `--label STR` | `""` | Free-form label written into every JSONL line (e.g. `decomposed` / `gated_fa_v2`) |
| `--output PATH` | (none) | If set, JSONL is written here **in addition** to stdout |
| `--vokra-cli PATH` | auto-discovered | Explicit binary path; defaults to `../../target/release/vokra-cli` relative to the script |

**Runtime budget** — each iteration is a fresh `vokra-cli bench` process
that pays session build + NVRTC JIT + weight upload as warmup, then runs
one timed transcribe. On an RTX 4090 with whisper-large-v3 that is
approximately **~20 s per iteration**, so `--iters 10` lands in
**~4 minutes** wall-clock (~$0.03 of vast.ai spot at $0.4/h).

### 3. Analyze

```bash
./tools/parity/cuda_rtf_analyze.py /root/rtf-decomposed.jsonl \
    --format markdown \
    --output /root/rtf-decomposed.report.md
```

Or stream from stdin:

```bash
./cuda_rtf_variance.sh --gguf ... --audio ... --iters 10 --fa-v2 off | \
    ./cuda_rtf_analyze.py -
```

Options:

| Flag | Default | Notes |
|------|---------|-------|
| positional `input` | `-` (stdin) | Path to the JSONL, or `-` for stdin |
| `--output PATH` | `-` (stdout) | Where to write the report |
| `--format markdown\|json` | `markdown` | `json` is machine-readable (feeds a downstream aggregator) |

### 4. Two-mode run (recommended)

The baseline JSON records two paths — decomposed and gated FA v2. To
reproduce the same A/B on a fresh instance, invoke the harness twice:

```bash
./tools/parity/cuda_rtf_variance.sh --gguf ... --audio ... --iters 10 \
    --fa-v2 off --label decomposed  --output rtf-decomposed.jsonl
./tools/parity/cuda_rtf_variance.sh --gguf ... --audio ... --iters 10 \
    --fa-v2 on  --label gated_fa_v2 --output rtf-fa-v2.jsonl

./tools/parity/cuda_rtf_analyze.py rtf-decomposed.jsonl --output rtf-decomposed.md
./tools/parity/cuda_rtf_analyze.py rtf-fa-v2.jsonl      --output rtf-fa-v2.md
```

The owner then folds both reports into
[`docs/m2-cuda-rtf-variance-template.md`](../../docs/m2-cuda-rtf-variance-template.md)
and commits the filled template alongside the two JSONL files.

## Owner workflow (vast.ai)

The vast.ai instance lifecycle (create / ssh / destroy / API key) is
**deliberately not in the shell script**. Per
[`docs/adr/M2-03-followup-rtf.md`](../../docs/adr/M2-03-followup-rtf.md)
§D7 and the `CLAUDE.md` operational note ("vast.ai … 都度起動 → 計測 →
`vastai destroy`"), the owner drives that lifecycle from their local
machine using their `vastai` CLI + API key.

Recommended shape (adapt to your credentials — never commit the API key):

```bash
# 1. On the local (owner) machine — search + create a spot instance.
OFFER=$(vastai search offers 'gpu_name=RTX_4090 num_gpus=1 rentable=true verified=true' --raw \
        | python3 -c 'import json,sys; print(sorted(json.load(sys.stdin), key=lambda o: o["dph_total"])[0]["id"])')
INSTANCE=$(vastai create instance $OFFER --image nvidia/cuda:12.6.2-devel-ubuntu22.04 \
        --disk 40 --ssh --raw | python3 -c 'import json,sys; print(json.load(sys.stdin)["new_contract"])')

# ALWAYS pair the create with a trap that destroys the instance on exit
# (a hung ssh session must not leak a running RTX 4090).
trap "vastai destroy instance $INSTANCE" EXIT

# 2. ssh in, build, transfer the GGUF + audio.
SSH_TARGET=$(vastai ssh-url $INSTANCE)
ssh $SSH_TARGET '
    apt-get update && apt-get install -y git build-essential curl
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.86.0
    . "$HOME/.cargo/env"
    git clone https://github.com/ayutaz/vokra.git && cd vokra
    cargo build --release -p vokra-cli
'
scp whisper-large-v3.gguf $SSH_TARGET:/root/
scp jfk-30s.wav           $SSH_TARGET:/root/

# 3. Run the harness (two modes).
ssh $SSH_TARGET '
    cd vokra
    ./tools/parity/cuda_rtf_variance.sh --gguf /root/whisper-large-v3.gguf \
        --audio /root/jfk-30s.wav --iters 10 --fa-v2 off \
        --label decomposed --output /root/rtf-decomposed.jsonl
    ./tools/parity/cuda_rtf_variance.sh --gguf /root/whisper-large-v3.gguf \
        --audio /root/jfk-30s.wav --iters 10 --fa-v2 on \
        --label gated_fa_v2 --output /root/rtf-fa-v2.jsonl
'

# 4. Pull the JSONL back and analyze locally.
scp $SSH_TARGET:/root/rtf-decomposed.jsonl .
scp $SSH_TARGET:/root/rtf-fa-v2.jsonl .

./tools/parity/cuda_rtf_analyze.py rtf-decomposed.jsonl \
    --output rtf-decomposed.report.md
./tools/parity/cuda_rtf_analyze.py rtf-fa-v2.jsonl \
    --output rtf-fa-v2.report.md

# 5. Fold both into the template, commit, then let the trap destroy the instance.
# (The trap fires on the `exit` at the end of this block.)
```

**Cost estimate** — RTX 4090 spot at ~$0.4/h × 10 min setup + 4 min per
mode + 2 min analyze/transfer ≈ **~$0.15** per two-mode run.

## Interpreting the report

The report is meant to answer a specific question the owner has after the
M2-03 follow-up single-shot measurement: **is the observed 0.081 – 0.115
RTF range on RTX 4090 hardware variability, or a single-shot outlier?**

The analyzer computes CV = stddev / mean and warns if CV > 0.20. Guidance:

- **CV `< 0.01`** — measurement is stable, the single-shot 0.081–0.115
  range was likely an outlier or a different device state (thermal
  throttling on a warm instance vs a fresh one).
- **CV `0.01 – 0.10`** — moderate variance, expected on shared spot
  instances. Mean / median are still trustworthy; use `--iters 20` for a
  tighter interval.
- **CV `> 0.20`** — high variance. The report surfaces a **WARNING** but
  does NOT exit non-zero. Likely causes: thermal throttling, GPU
  boost-clock jitter, PCIe contention, or a mixed instance state. Do NOT
  promote the formal `<0.10` gate off this run — hand the raw JSONL to
  M2-14.

**Formal `<0.10` gate** — the analyzer never asserts an RTF ceiling and
never returns non-zero on a "too slow" verdict. That decision is the
owner's after inspecting the report, per
[`docs/adr/M2-03-followup-rtf.md`](../../docs/adr/M2-03-followup-rtf.md)
§D6.

## JSONL schema

Each line is a JSON object with one of these three `status` shapes:

**Successful iteration**:

```json
{
  "iter": 3,
  "timestamp": "2026-07-08T10:00:40Z",
  "status": "ok",
  "rtf": 0.1133,
  "latency_ms": 3400.0,
  "fa_v2_mode": "off",
  "backend": "cuda",
  "gguf": "/root/whisper-large-v3.gguf",
  "audio": "/root/jfk-30s.wav",
  "host": "vast-instance",
  "gpu": "NVIDIA GeForce RTX 4090",
  "driver": "580.126.09",
  "label": "decomposed",
  "bench": {"task": "asr", "iters": 1, "warmup": 1, "audio_seconds": 30.0, "rtf": 0.1133, "ttfa_ms": 3400.0, "latency_ms": {"p50": 3400.0, "p95": 3400.0, "p99": 3400.0, "mean": 3400.0, "jitter": 0.0, "min": 3400.0, "max": 3400.0}}
}
```

**Failed iteration** (bench exit != 0 or malformed JSON):

```json
{
  "iter": 4,
  "timestamp": "2026-07-08T10:01:00Z",
  "status": "error",
  "exit_code": 3,
  "error": "error: BackendUnavailable(Cuda) — ...",
  "fa_v2_mode": "on",
  "backend": "cuda",
  "label": "gated_fa_v2"
}
```

**Trailer** (one per run, at the end):

```json
{
  "type": "summary",
  "iters_requested": 10,
  "iters_failed": 0,
  "started_at": "2026-07-08T10:00:00Z",
  "ended_at": "2026-07-08T10:03:15Z",
  "fa_v2_mode": "off",
  "backend": "cuda",
  "label": "decomposed",
  "host": "vast-instance",
  "gpu": "NVIDIA GeForce RTX 4090",
  "driver": "580.126.09",
  "gguf": "/root/whisper-large-v3.gguf",
  "audio": "/root/jfk-30s.wav"
}
```

## FAQ

**Q. Why not one long `--iters 10` bench call?**
> `vokra-cli bench --iters N` aggregates internally and only emits the
> summary JSON (see `crates/vokra-cli/src/report.rs::to_json`). We need
> per-iter samples for the CV / histogram, so the harness loops
> `--iters 1 --warmup 1` N times instead. The extra warmups cost about
> 4 min for `--iters 10` on RTX 4090.

**Q. Why not add per-iter emission to `vokra-cli bench`?**
> Deferred. The harness deliberately uses only the current bench contract
> so it survives future bench refactors. If per-iter emission lands, the
> harness can switch to one call and the analyzer stays unchanged.

**Q. Can this measure Metal / CPU too?**
> Yes — pass `--backend metal` or `--backend cpu`. There is no
> `--fa-v2 on/off` toggle on those backends, so the flag is ignored (the
> env var is CUDA-specific). Metal has its own always-on gate seam
> (`whisper_metal_e2e_matches_cpu`); this harness is not the right tool
> for a formal Metal RTF gate.

**Q. Where does the baseline live?**
> [`docs/bench-baselines/whisper_large_v3_cuda_rtf.json`](../../docs/bench-baselines/whisper_large_v3_cuda_rtf.json)
> — decomposed median 0.1133, gated FA v2 median 0.1323 (both on RTX
> 4090 spot, 2026-07-07).

**Q. What if `nvidia-smi` is not installed?**
> The `gpu` / `driver` fields are populated with
> `"unavailable (no nvidia-smi)"` and the run continues. Only the
> metadata is lost; the RTF measurement is unaffected.

## References

- ADR: [`docs/adr/M2-03-followup-rtf.md`](../../docs/adr/M2-03-followup-rtf.md) (§D6, §D7, §D8)
- Owner checklist: [`docs/m2-owner-verification-checklist.md`](../../docs/m2-owner-verification-checklist.md) §2
- Existing sanity test: [`crates/vokra-backend-cuda/tests/whisper_cuda_large_v3_rtf.rs`](../../crates/vokra-backend-cuda/tests/whisper_cuda_large_v3_rtf.rs)
- Baseline JSON: [`docs/bench-baselines/whisper_large_v3_cuda_rtf.json`](../../docs/bench-baselines/whisper_large_v3_cuda_rtf.json)
- Bench CLI source: [`crates/vokra-cli/src/bench.rs`](../../crates/vokra-cli/src/bench.rs), [`crates/vokra-cli/src/report.rs`](../../crates/vokra-cli/src/report.rs)
