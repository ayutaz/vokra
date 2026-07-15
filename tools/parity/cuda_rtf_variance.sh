#!/usr/bin/env bash
# cuda_rtf_variance.sh — M2-03-followup / M2-14 CUDA large-v3 RTF variance harness.
#
# Runs N successive ``vokra-cli bench`` invocations against a Whisper large-v3
# GGUF on a CUDA host, one JSON line per iteration, so the companion analyzer
# (``cuda_rtf_analyze.py``) can compute mean / median / stddev / CV /
# p50/p95/p99 / min / max / histogram over the collected samples.
#
# **Why one bench invocation per iteration (fresh process each time)** — the
# existing ``vokra-cli bench --iters N`` aggregates in-process and emits only
# the summary JSON (``report.rs::to_json``), so it does not expose per-iter
# latency samples needed for a histogram + CV plot. Looping ``--iters 1
# --warmup 1`` gives us one **steady-state** sample per invocation (the warmup
# absorbs the CUDA session build + NVRTC JIT + weight upload cost), at the
# price of paying that warmup N times. On an RTX 4090 with whisper-large-v3
# that is roughly ~20 s per iteration; for the default N=10 the whole run
# lands in ~4 minutes (~$0.03 of vast.ai spot).
#
# **Position in the plan** — this is the *variance analysis* rung on the same
# family of sanity numbers reported by
# ``crates/vokra-backend-cuda/tests/whisper_cuda_large_v3_rtf.rs``:
# a reference measurement, not the formal < 0.10 always-on gate. That
# always-on gate lives at **M2-14** (owner self-hosted CUDA runner) + **M3-01**
# (5% regression gate) per ``docs/adr/M2-03-followup-rtf.md`` §D6. This script
# only emits samples — it never asserts an RTF ceiling and never promotes any
# threshold. All promotion / demotion decisions are the owner's after they
# inspect the analyzer's output.
#
# **Zero-dep + NVIDIA EULA red-lines** (NFR-DS-02 / ``CLAUDE.md``): this
# harness does NOT ``pip install`` anything (analyzer is stdlib), does NOT
# apt-install cuDNN/cuBLAS/cuFFT, and does NOT bundle cudart. Only ``bash``,
# GNU coreutils, ``python3`` (stdlib), and the pre-built ``vokra-cli`` binary
# are required. CUDA is discovered at runtime via ``libcuda.so.1`` +
# ``libnvrtc.so.12`` dlopen (raw FFI, EULA install model). See
# ``docs/adr/M2-03-followup-rtf.md`` §D8.
#
# **Owner scope** — vast.ai instance lifecycle (create / ssh / destroy),
# API key handling, and any decision on whether the measured CV / p99
# should promote the formal < 0.10 gate. See
# ``tools/parity/README-cuda-rtf-variance.md`` for the full walkthrough.
#
# Usage::
#
#   ./cuda_rtf_variance.sh \
#       --gguf   whisper-large-v3.gguf \
#       --audio  jfk-30s.wav          \
#       --iters  10                   \
#       [--warmup 1]                  \
#       [--fa-mode decomposed|v2|v3]  \
#       [--fa-v2 on|off]              \
#       [--backend cuda]              \
#       [--vokra-cli ./target/release/vokra-cli] \
#       [--output rtf_samples.jsonl]  \
#       [--label  decomposed]
#
# Emits one JSON object per iteration on stdout (and, if ``--output`` is
# given, to that file — one JSON line per iteration, ``jsonlines/ndjson``
# format). ``--output`` overwrites any existing file.
#
# **FA mode (M4-07-T13)** — three attention-path legs on the same binary:
#   decomposed  VOKRA_CUDA_DISABLE_FA_V2=1 + VOKRA_CUDA_DISABLE_FA_V3=1
#   v2          VOKRA_CUDA_DISABLE_FA_V3=1        (gated FA v2, t_q >= 16)
#   v3          VOKRA_CUDA_FA_V3_ENCODER=1        (FA v3 priority when the
#               Hopper probe holds, incl. the encoder opt-in so the v3 gain
#               surface — t_q = 1500 self-attention — is exposed e2e)
# ``--fa-v2 on|off`` is kept as a backward-compatible alias (on = v2,
# off = decomposed); passing both --fa-mode and --fa-v2 is an error rather
# than a silent precedence rule. On pre-M4-07 binaries the extra V3 env vars
# are simply unread, so old measurement recipes reproduce identically.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

ITERS=10
WARMUP=1
FA_MODE=""
FA_V2=""
BACKEND=cuda
GGUF=""
AUDIO=""
LABEL=""
OUTPUT=""
VOKRA_CLI=""

# ---------------------------------------------------------------------------
# CLI parsing (hand-written; no getopt to keep the script portable across
# BSD userland on macOS and GNU userland on Linux vast.ai instances)
# ---------------------------------------------------------------------------

usage() {
    sed -n '2,55p' "$0"
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --gguf)      GGUF="$2";      shift 2 ;;
        --audio)     AUDIO="$2";     shift 2 ;;
        --iters)     ITERS="$2";     shift 2 ;;
        --warmup)    WARMUP="$2";    shift 2 ;;
        --fa-mode)   FA_MODE="$2";   shift 2 ;;
        --fa-v2)     FA_V2="$2";     shift 2 ;;
        --backend)   BACKEND="$2";   shift 2 ;;
        --label)     LABEL="$2";     shift 2 ;;
        --output)    OUTPUT="$2";    shift 2 ;;
        --vokra-cli) VOKRA_CLI="$2"; shift 2 ;;
        -h|--help)   usage 0 ;;
        *) echo "error: unexpected argument '$1'" >&2; usage 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Argument validation — bail loudly on missing inputs (FR-EX-08 spirit: no
# silent fallback to a default GGUF / audio path)
# ---------------------------------------------------------------------------

if [ -z "$GGUF"  ]; then echo "error: --gguf is required"  >&2; exit 2; fi
if [ -z "$AUDIO" ]; then echo "error: --audio is required" >&2; exit 2; fi
if [ ! -f "$GGUF"  ]; then echo "error: gguf not found: $GGUF"   >&2; exit 2; fi
if [ ! -f "$AUDIO" ]; then echo "error: audio not found: $AUDIO" >&2; exit 2; fi

# FA mode resolution (M4-07-T13). --fa-v2 stays as a compat alias; a
# simultaneous explicit --fa-mode + --fa-v2 is rejected (no silent precedence).
if [ -n "$FA_MODE" ] && [ -n "$FA_V2" ]; then
    echo "error: pass either --fa-mode or the legacy --fa-v2 alias, not both" >&2
    exit 2
fi
if [ -n "$FA_V2" ]; then
    case "$FA_V2" in
        on)  FA_MODE=v2 ;;
        off) FA_MODE=decomposed ;;
        *) echo "error: --fa-v2 must be 'on' or 'off' (got '$FA_V2')" >&2; exit 2 ;;
    esac
fi
if [ -z "$FA_MODE" ]; then
    FA_MODE=v2   # historical default: --fa-v2 on (the gated FA v2 wrapper)
fi
case "$FA_MODE" in
    decomposed|v2|v3) ;;
    *) echo "error: --fa-mode must be 'decomposed' | 'v2' | 'v3' (got '$FA_MODE')" >&2; exit 2 ;;
esac
# Legacy JSONL field value (kept alongside the new fa_mode field so the
# pre-M4-07 analyzer / recorded baselines stay comparable): v3 keeps FA v2
# enabled below its own gate, so it maps to "on".
case "$FA_MODE" in
    decomposed) FA_V2_COMPAT=off ;;
    *)          FA_V2_COMPAT=on ;;
esac

case "$BACKEND" in
    cuda|metal|cpu) ;;
    *) echo "error: --backend must be 'cuda' | 'metal' | 'cpu' (got '$BACKEND')" >&2; exit 2 ;;
esac

if ! [[ "$ITERS" =~ ^[0-9]+$ ]] || [ "$ITERS" -lt 1 ]; then
    echo "error: --iters must be a positive integer (got '$ITERS')" >&2
    exit 2
fi
if ! [[ "$WARMUP" =~ ^[0-9]+$ ]]; then
    echo "error: --warmup must be a non-negative integer (got '$WARMUP')" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# vokra-cli binary discovery — prefer explicit --vokra-cli, else fall back to
# ``target/release/vokra-cli`` next to the repo root (script lives at
# ``tools/parity/`` so ``../../target/release/vokra-cli`` is the canonical
# release binary). Bail loudly if none found.
# ---------------------------------------------------------------------------

if [ -z "$VOKRA_CLI" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    CANDIDATE="$SCRIPT_DIR/../../target/release/vokra-cli"
    if [ -x "$CANDIDATE" ]; then
        VOKRA_CLI="$CANDIDATE"
    elif command -v vokra-cli >/dev/null 2>&1; then
        VOKRA_CLI="$(command -v vokra-cli)"
    else
        echo "error: vokra-cli not found — pass --vokra-cli PATH or build with 'cargo build --release -p vokra-cli'" >&2
        exit 2
    fi
fi

if [ ! -x "$VOKRA_CLI" ]; then
    echo "error: vokra-cli is not executable: $VOKRA_CLI" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Host / GPU fingerprint — best-effort, all optional (script must run on
# hosts without ``nvidia-smi`` too, e.g. dry-run on a Mac)
# ---------------------------------------------------------------------------

HOSTNAME_STR="$(hostname 2>/dev/null || echo unknown)"

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/"/\\"/g' || echo unknown)"
    GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
else
    GPU_NAME="unavailable (no nvidia-smi)"
    GPU_DRIVER="unavailable (no nvidia-smi)"
fi

# ---------------------------------------------------------------------------
# JSON string escape — the metadata strings above (hostname / gpu name)
# may contain characters that break naive JSON emission. Escape via python3
# so we do not hand-roll (the whole point of the analyzer being stdlib is
# to keep JSON handling correct).
# ---------------------------------------------------------------------------

json_escape() {
    python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

HOSTNAME_JSON="$(json_escape "$HOSTNAME_STR")"
GPU_NAME_JSON="$(json_escape "$GPU_NAME")"
GPU_DRIVER_JSON="$(json_escape "$GPU_DRIVER")"
GGUF_JSON="$(json_escape "$GGUF")"
AUDIO_JSON="$(json_escape "$AUDIO")"
LABEL_JSON="$(json_escape "$LABEL")"

# ---------------------------------------------------------------------------
# FA mode → env var toggles (all presence-based; see fa_v3.rs / context.rs)
#   VOKRA_CUDA_DISABLE_FA_V2=1  forces the decomposed (2 + 7·n_head) chain
#   VOKRA_CUDA_DISABLE_FA_V3=1  deselects the FA v3 arm (v2 stays gated)
#   VOKRA_CUDA_FA_V3_ENCODER=1  opts the encoder chains into FA v3 when the
#                               Hopper probe holds (decomposed continue + a
#                               stderr note elsewhere — never an error)
# Every mode sets at least one variable, so the ``env`` prefix is uniform.
# ---------------------------------------------------------------------------

case "$FA_MODE" in
    decomposed) FA_ENV=(VOKRA_CUDA_DISABLE_FA_V2=1 VOKRA_CUDA_DISABLE_FA_V3=1) ;;
    v2)         FA_ENV=(VOKRA_CUDA_DISABLE_FA_V3=1) ;;
    v3)         FA_ENV=(VOKRA_CUDA_FA_V3_ENCODER=1) ;;
esac

# ---------------------------------------------------------------------------
# Output file setup — truncate on start, then tee each iteration line into
# it. If ``--output`` is empty we only emit to stdout.
# ---------------------------------------------------------------------------

if [ -n "$OUTPUT" ]; then
    : > "$OUTPUT"   # truncate
fi

emit_line() {
    local line="$1"
    printf '%s\n' "$line"
    if [ -n "$OUTPUT" ]; then
        printf '%s\n' "$line" >> "$OUTPUT"
    fi
}

# ---------------------------------------------------------------------------
# Iteration loop
#
# Each iteration is a fresh ``vokra-cli bench`` process with
# ``--iters 1 --warmup <M>``. The warmup absorbs session build + NVRTC JIT
# + weight upload; the single timed pass is the steady-state sample. The
# full report JSON emitted by bench (see ``report.rs::to_json``) is nested
# into our per-iter envelope under ``"bench"`` so no data is dropped.
#
# We deliberately do NOT abort on a single-iteration failure — if the CUDA
# device flaps on one iteration the analyzer still gets N-1 samples and can
# flag the missing one. A non-zero exit is only produced if *every*
# iteration fails.
# ---------------------------------------------------------------------------

START_TS_RUN="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
FAIL_COUNT=0

for i in $(seq 1 "$ITERS"); do
    ITER_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    # Assemble the bench invocation. ``env`` prefix ensures the FA toggles
    # only apply to this call and do not leak to the enclosing shell.
    BENCH_OUT="$(env "${FA_ENV[@]}" "$VOKRA_CLI" bench \
        --model "$GGUF" \
        --input "$AUDIO" \
        --backend "$BACKEND" \
        --iters 1 \
        --warmup "$WARMUP" \
        --format json 2>&1)" || RC=$?
    RC="${RC:-0}"

    if [ "$RC" -ne 0 ]; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
        # Emit a failure record so the analyzer can count / display it,
        # then continue to the next iteration.
        FAIL_MSG_JSON="$(json_escape "$BENCH_OUT")"
        emit_line "{\"iter\":$i,\"timestamp\":\"$ITER_TS\",\"status\":\"error\",\"exit_code\":$RC,\"error\":$FAIL_MSG_JSON,\"fa_mode\":\"$FA_MODE\",\"fa_v2_mode\":\"$FA_V2_COMPAT\",\"backend\":\"$BACKEND\",\"label\":$LABEL_JSON}"
        unset RC
        continue
    fi
    unset RC

    # ``vokra-cli bench --format json`` prints a single JSON line. If
    # anything else was printed (e.g. a stray log line on a debug build)
    # we defensively pick the *last* line that starts with ``{``.
    BENCH_JSON="$(printf '%s\n' "$BENCH_OUT" | awk '/^\{/{last=$0} END{print last}')"
    if [ -z "$BENCH_JSON" ]; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_MSG_JSON="$(json_escape "no JSON line in bench output: $BENCH_OUT")"
        emit_line "{\"iter\":$i,\"timestamp\":\"$ITER_TS\",\"status\":\"error\",\"error\":$FAIL_MSG_JSON,\"fa_mode\":\"$FA_MODE\",\"fa_v2_mode\":\"$FA_V2_COMPAT\",\"backend\":\"$BACKEND\",\"label\":$LABEL_JSON}"
        continue
    fi

    # Extract rtf + mean latency with python3 so the analyzer's mean / p95
    # numbers agree with the shell's per-iter emission. Any parse failure
    # (a debug build emitted malformed JSON, an exception dumped to stdout,
    # etc.) prints ``null`` for both fields and lets the iteration continue
    # — we do NOT want set -e to abort the whole run on one bad line.
    EXTRACT="$(python3 -c '
import json, sys
try:
    d = json.loads(sys.argv[1])
except Exception:
    print("null null")
    sys.exit(0)
r = d.get("rtf")
r = r if isinstance(r, (int, float)) else "null"
lm = d.get("latency_ms", {}) if isinstance(d, dict) else {}
m = lm.get("mean") if isinstance(lm, dict) else None
m = m if isinstance(m, (int, float)) else "null"
print(f"{r} {m}")
' "$BENCH_JSON")"
    RTF="${EXTRACT% *}"
    WALL_MS="${EXTRACT##* }"

    if [ "$RTF" = "null" ]; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_MSG_JSON="$(json_escape "malformed bench JSON: $BENCH_JSON")"
        emit_line "{\"iter\":$i,\"timestamp\":\"$ITER_TS\",\"status\":\"error\",\"error\":$FAIL_MSG_JSON,\"fa_mode\":\"$FA_MODE\",\"fa_v2_mode\":\"$FA_V2_COMPAT\",\"backend\":\"$BACKEND\",\"label\":$LABEL_JSON}"
        continue
    fi

    # Compose the per-iter envelope. ``bench`` field is the raw report JSON
    # (round-trippable) so nothing is lost.
    emit_line "{\"iter\":$i,\"timestamp\":\"$ITER_TS\",\"status\":\"ok\",\"rtf\":$RTF,\"latency_ms\":$WALL_MS,\"fa_mode\":\"$FA_MODE\",\"fa_v2_mode\":\"$FA_V2_COMPAT\",\"backend\":\"$BACKEND\",\"gguf\":$GGUF_JSON,\"audio\":$AUDIO_JSON,\"host\":$HOSTNAME_JSON,\"gpu\":$GPU_NAME_JSON,\"driver\":$GPU_DRIVER_JSON,\"label\":$LABEL_JSON,\"bench\":$BENCH_JSON}"
done

END_TS_RUN="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---------------------------------------------------------------------------
# Run trailer — a summary metadata line so the analyzer can print the
# fingerprint of the collection without re-scraping every sample. This is
# emitted last so the analyzer's ``jq`` / line-by-line reader can either
# use or ignore it (a line with ``"type":"summary"``).
# ---------------------------------------------------------------------------

emit_line "{\"type\":\"summary\",\"iters_requested\":$ITERS,\"iters_failed\":$FAIL_COUNT,\"started_at\":\"$START_TS_RUN\",\"ended_at\":\"$END_TS_RUN\",\"fa_mode\":\"$FA_MODE\",\"fa_v2_mode\":\"$FA_V2_COMPAT\",\"backend\":\"$BACKEND\",\"label\":$LABEL_JSON,\"host\":$HOSTNAME_JSON,\"gpu\":$GPU_NAME_JSON,\"driver\":$GPU_DRIVER_JSON,\"gguf\":$GGUF_JSON,\"audio\":$AUDIO_JSON}"

if [ "$FAIL_COUNT" -eq "$ITERS" ]; then
    echo "error: all $ITERS iterations failed — see JSONL output above" >&2
    exit 1
fi

exit 0
