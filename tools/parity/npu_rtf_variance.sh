#!/usr/bin/env bash
# npu_rtf_variance.sh — M5-01 / M5-02 NPU delegate RTF variance harness.
#
# Runs N successive ``vokra-cli bench`` invocations against a Whisper /
# piper-plus / Kokoro GGUF on an NPU-capable host, one JSON line per
# iteration, so the companion analyzer (``npu_rtf_analyze.py``) can
# compute mean / median / stddev / CV / p50/p95/p99 / min / max /
# histogram over the collected samples PLUS surface the **placement
# fraction** — the share of hot ops that actually ran on the NPU vs
# fell back to CPU. Silent CPU fallback disqualifies the run per the
# NFR-PF-12 protocol (FR-EX-08); the analyzer WARNs when placement
# drops below 90 %, and the raw fraction is preserved for the owner.
#
# **Position in the plan** — this is the *variance analysis* rung for
# NPU delegates (M5-01 CoreML/ANE + M5-02 QNN/Hexagon). The formal
# NFR-PF-12 acceptance criterion (≥ 2× over the CPU baseline) is an
# owner judgment based on the report this harness produces. This
# script only emits samples; it never asserts an RTF ceiling and never
# promotes any threshold — same red-line as
# ``tools/parity/cuda_rtf_variance.sh`` (see
# ``docs/adr/M2-03-followup-rtf.md`` §D6).
#
# **Baseline reference** — per the 2026-08-09 protocol codification,
# the "CPU baseline" for the 2× gate is **M5-14-post CPU** (SIMD
# hot-path optimised, libm-route). Bakeoff numbers must be paired
# with a M5-14-post CPU sample collected on the same host in the same
# session, else the 2× ratio is meaningless. See
# ``docs/handoff/m5-02.md`` §"NFR-PF-12 baseline".
#
# **Zero-dep + delegate-runtime handling** (NFR-DS-02 / ``CLAUDE.md``):
# this harness does NOT ``pip install`` anything (analyzer is stdlib),
# does NOT apt-install the CoreML / QNN runtimes, and does NOT bundle
# any Apple / Qualcomm SDK. CoreML is discovered via
# ``dlopen("CoreML.framework/CoreML")`` on macOS; QNN via
# ``dlopen("libQnnHtp.so")`` on the Qualcomm SoC. Both are owner-side
# system installs — the harness only calls the ``vokra-cli`` binary and
# expects the delegate runtime to already be resolvable.
#
# **Owner scope** — bakeoff rig lifecycle (macOS/iOS with M-series or
# ANE; Snapdragon devboard with Hexagon), any decision on whether the
# measured CV / placement fraction / RTF should promote the 2× verdict.
# See ``docs/m5-owner-verification-checklist.md`` §1.5 and the
# per-delegate templates ``docs/handoff/m5-01-coreml-bakeoff-template.md``
# / ``docs/handoff/m5-02-qnn-bakeoff-template.md``.
#
# Usage::
#
#   ./npu_rtf_variance.sh \
#       --gguf         whisper-large-v3.gguf \
#       --audio        jfk-30s.wav          \
#       --backend      coreml               \
#       --iters        10                   \
#       [--warmup 1]                        \
#       [--placement-probe /path/to/probe.sh] \
#       [--vokra-cli ./target/release/vokra-cli] \
#       [--output rtf_samples.jsonl]        \
#       [--label ane-m4pro]
#
# Emits one JSON object per iteration on stdout (and, if ``--output`` is
# given, to that file — one JSON line per iteration, ``jsonlines/ndjson``
# format). ``--output`` overwrites any existing file.
#
# **Placement probe** — when ``--placement-probe`` is set, the given
# command is invoked after each iteration and its stdout is folded into
# the iteration line under ``"placement"``. The probe is expected to
# emit a single JSON object like
# ``{"ane_frac": 0.94, "gpu_frac": 0.03, "cpu_frac": 0.03}`` (values in
# [0, 1] summing to ~1.0). Missing / non-JSON output is folded as
# ``"placement": null`` and the analyzer surfaces it as
# ``placement=unknown`` with a WARN. This is deliberate — CC cannot
# ship the CoreML / QNN placement inspector (Xcode Instruments MLModel
# trace / QNN ``qnn-net-run --profiling_option op``) as part of the
# zero-dep tree; the owner wires it up per the templates.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

ITERS=10
WARMUP=1
BACKEND=""
GGUF=""
AUDIO=""
LABEL=""
OUTPUT=""
VOKRA_CLI=""
PLACEMENT_PROBE=""

# ---------------------------------------------------------------------------
# CLI parsing (hand-written; no getopt to keep the script portable across
# BSD userland on macOS and GNU userland on Linux Snapdragon boxes)
# ---------------------------------------------------------------------------

usage() {
    sed -n '2,80p' "$0"
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --gguf)            GGUF="$2";            shift 2 ;;
        --audio)           AUDIO="$2";           shift 2 ;;
        --iters)           ITERS="$2";           shift 2 ;;
        --warmup)          WARMUP="$2";          shift 2 ;;
        --backend)         BACKEND="$2";         shift 2 ;;
        --label)           LABEL="$2";           shift 2 ;;
        --output)          OUTPUT="$2";          shift 2 ;;
        --vokra-cli)       VOKRA_CLI="$2";       shift 2 ;;
        --placement-probe) PLACEMENT_PROBE="$2"; shift 2 ;;
        -h|--help)         usage 0 ;;
        *) echo "error: unexpected argument '$1'" >&2; usage 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Argument validation — bail loudly on missing inputs (FR-EX-08 spirit: no
# silent fallback to a default GGUF / audio path / backend)
# ---------------------------------------------------------------------------

if [ -z "$GGUF"    ]; then echo "error: --gguf is required"    >&2; exit 2; fi
if [ -z "$AUDIO"   ]; then echo "error: --audio is required"   >&2; exit 2; fi
if [ -z "$BACKEND" ]; then echo "error: --backend is required (coreml | qnn | cuda | cpu)" >&2; exit 2; fi
if [ ! -f "$GGUF"  ]; then echo "error: gguf not found: $GGUF"   >&2; exit 2; fi
if [ ! -f "$AUDIO" ]; then echo "error: audio not found: $AUDIO" >&2; exit 2; fi

case "$BACKEND" in
    coreml|qnn|cuda|cpu) ;;
    *) echo "error: --backend must be 'coreml' | 'qnn' | 'cuda' | 'cpu' (got '$BACKEND')" >&2; exit 2 ;;
esac

if ! [[ "$ITERS" =~ ^[0-9]+$ ]] || [ "$ITERS" -lt 1 ]; then
    echo "error: --iters must be a positive integer (got '$ITERS')" >&2
    exit 2
fi
if ! [[ "$WARMUP" =~ ^[0-9]+$ ]]; then
    echo "error: --warmup must be a non-negative integer (got '$WARMUP')" >&2
    exit 2
fi

if [ -n "$PLACEMENT_PROBE" ]; then
    # Validate the placement probe is executable now, not on first iter.
    # A missing probe partway through a 10-iter run wastes owner time.
    if [ ! -x "$PLACEMENT_PROBE" ] && ! command -v "$PLACEMENT_PROBE" >/dev/null 2>&1; then
        echo "error: --placement-probe not executable: $PLACEMENT_PROBE" >&2
        exit 2
    fi
fi

# ---------------------------------------------------------------------------
# vokra-cli binary discovery — prefer explicit --vokra-cli, else fall back
# to ``target/release/vokra-cli`` next to the repo root (script lives at
# ``tools/parity/`` so ``../../target/release/vokra-cli`` is canonical).
# Bail loudly if none found.
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
# Host / delegate fingerprint — best-effort, all optional
#
# For CoreML (macOS) we harvest sw_vers + sysctl. For QNN (Linux on
# Snapdragon) we harvest uname -m + getprop when Android tools are on
# PATH. Failures are per-field, never fatal.
# ---------------------------------------------------------------------------

HOSTNAME_STR="$(hostname 2>/dev/null || echo unknown)"

case "$BACKEND" in
    coreml)
        DEVICE_NAME="$(sw_vers -productName 2>/dev/null || echo macOS-unknown)"
        DEVICE_OS="$(sw_vers -productVersion 2>/dev/null || echo unknown)"
        DEVICE_SOC="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
        ;;
    qnn)
        DEVICE_NAME="$(uname -m 2>/dev/null || echo linux-unknown)"
        DEVICE_OS="$(uname -sr 2>/dev/null || echo unknown)"
        DEVICE_SOC="$(getprop ro.hardware 2>/dev/null || echo unknown)"
        ;;
    cuda)
        if command -v nvidia-smi >/dev/null 2>&1; then
            DEVICE_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
            DEVICE_OS="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
            DEVICE_SOC="cuda"
        else
            DEVICE_NAME="unavailable (no nvidia-smi)"
            DEVICE_OS="unavailable (no nvidia-smi)"
            DEVICE_SOC="cuda"
        fi
        ;;
    cpu)
        DEVICE_NAME="cpu"
        DEVICE_OS="$(uname -sr 2>/dev/null || echo unknown)"
        DEVICE_SOC="cpu-baseline"
        ;;
esac

# ---------------------------------------------------------------------------
# JSON string escape — hostname / device fields may contain characters
# that break naive JSON emission. Escape via python3 so we do not
# hand-roll (the whole point of the analyzer being stdlib is to keep
# JSON handling correct).
# ---------------------------------------------------------------------------

json_escape() {
    python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

HOSTNAME_JSON="$(json_escape "$HOSTNAME_STR")"
DEVICE_NAME_JSON="$(json_escape "$DEVICE_NAME")"
DEVICE_OS_JSON="$(json_escape "$DEVICE_OS")"
DEVICE_SOC_JSON="$(json_escape "$DEVICE_SOC")"
GGUF_JSON="$(json_escape "$GGUF")"
AUDIO_JSON="$(json_escape "$AUDIO")"
LABEL_JSON="$(json_escape "$LABEL")"

# ---------------------------------------------------------------------------
# Output file setup — truncate on start, then tee each iteration line
# into it. If ``--output`` is empty we only emit to stdout.
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

# Read the placement probe stdout as JSON and re-emit it inside our
# envelope. Any parse failure yields ``null`` — the analyzer surfaces it
# as WARN rather than the harness silently synthesising 100 % NPU. This
# is the FR-EX-08 red-line — a missing probe must never look like a
# passing bakeoff.
placement_snapshot() {
    if [ -z "$PLACEMENT_PROBE" ]; then
        printf '%s' 'null'
        return
    fi
    local raw
    raw="$("$PLACEMENT_PROBE" 2>/dev/null || true)"
    if [ -z "$raw" ]; then
        printf '%s' 'null'
        return
    fi
    # Validate it parses as JSON (dict); else null.
    python3 -c '
import json, sys
try:
    obj = json.loads(sys.argv[1])
except Exception:
    print("null"); sys.exit(0)
if not isinstance(obj, dict):
    print("null"); sys.exit(0)
print(json.dumps(obj))
' "$raw"
}

# ---------------------------------------------------------------------------
# Iteration loop
#
# Each iteration is a fresh ``vokra-cli bench`` process with
# ``--iters 1 --warmup <M>``. The warmup absorbs delegate session build
# + weight upload; the single timed pass is the steady-state sample.
# The full report JSON emitted by bench (see ``report.rs::to_json``) is
# nested into our per-iter envelope under ``"bench"`` so no data is
# dropped.
#
# We deliberately do NOT abort on a single-iteration failure — if the
# NPU delegate flaps on one iteration the analyzer still gets N-1
# samples and can flag the missing one. A non-zero exit is only
# produced if *every* iteration fails.
# ---------------------------------------------------------------------------

START_TS_RUN="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
FAIL_COUNT=0

for i in $(seq 1 "$ITERS"); do
    ITER_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    BENCH_OUT="$("$VOKRA_CLI" bench \
        --model "$GGUF" \
        --input "$AUDIO" \
        --backend "$BACKEND" \
        --iters 1 \
        --warmup "$WARMUP" \
        --format json 2>&1)" || RC=$?
    RC="${RC:-0}"

    if [ "$RC" -ne 0 ]; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_MSG_JSON="$(json_escape "$BENCH_OUT")"
        emit_line "{\"iter\":$i,\"timestamp\":\"$ITER_TS\",\"status\":\"error\",\"exit_code\":$RC,\"error\":$FAIL_MSG_JSON,\"backend\":\"$BACKEND\",\"label\":$LABEL_JSON}"
        unset RC
        continue
    fi
    unset RC

    # ``vokra-cli bench --format json`` prints a single JSON line. If
    # anything else was printed (a stray log line on a debug build) we
    # defensively pick the *last* line that starts with ``{``.
    BENCH_JSON="$(printf '%s\n' "$BENCH_OUT" | awk '/^\{/{last=$0} END{print last}')"
    if [ -z "$BENCH_JSON" ]; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_MSG_JSON="$(json_escape "no JSON line in bench output: $BENCH_OUT")"
        emit_line "{\"iter\":$i,\"timestamp\":\"$ITER_TS\",\"status\":\"error\",\"error\":$FAIL_MSG_JSON,\"backend\":\"$BACKEND\",\"label\":$LABEL_JSON}"
        continue
    fi

    EXTRACT="$(python3 -c '
import json, sys
try:
    d = json.loads(sys.argv[1])
except Exception:
    print("null null"); sys.exit(0)
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
        emit_line "{\"iter\":$i,\"timestamp\":\"$ITER_TS\",\"status\":\"error\",\"error\":$FAIL_MSG_JSON,\"backend\":\"$BACKEND\",\"label\":$LABEL_JSON}"
        continue
    fi

    # Placement snapshot AFTER the bench call so the probe observes the
    # process state that just finished (op counters, dispatch trace, etc).
    PLACEMENT_JSON="$(placement_snapshot)"

    emit_line "{\"iter\":$i,\"timestamp\":\"$ITER_TS\",\"status\":\"ok\",\"rtf\":$RTF,\"latency_ms\":$WALL_MS,\"backend\":\"$BACKEND\",\"placement\":$PLACEMENT_JSON,\"gguf\":$GGUF_JSON,\"audio\":$AUDIO_JSON,\"host\":$HOSTNAME_JSON,\"device_name\":$DEVICE_NAME_JSON,\"device_os\":$DEVICE_OS_JSON,\"device_soc\":$DEVICE_SOC_JSON,\"label\":$LABEL_JSON,\"bench\":$BENCH_JSON}"
done

END_TS_RUN="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

emit_line "{\"type\":\"summary\",\"iters_requested\":$ITERS,\"iters_failed\":$FAIL_COUNT,\"started_at\":\"$START_TS_RUN\",\"ended_at\":\"$END_TS_RUN\",\"backend\":\"$BACKEND\",\"label\":$LABEL_JSON,\"host\":$HOSTNAME_JSON,\"device_name\":$DEVICE_NAME_JSON,\"device_os\":$DEVICE_OS_JSON,\"device_soc\":$DEVICE_SOC_JSON,\"gguf\":$GGUF_JSON,\"audio\":$AUDIO_JSON,\"placement_probe\":$(json_escape "$PLACEMENT_PROBE")}"

if [ "$FAIL_COUNT" -eq "$ITERS" ]; then
    echo "error: all $ITERS iterations failed — see JSONL output above" >&2
    exit 1
fi

exit 0
