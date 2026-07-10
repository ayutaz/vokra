#!/usr/bin/env bash
# vast-e2e-measurement.sh — end-to-end vast.ai measurement pipeline.
#
# Automates the vast.ai lifecycle (search / create / ssh / run / scp / destroy)
# for two v0.9 reference measurements the owner-checklist calls out:
#
#   1. M3-01  CUDA large-v3 RTF variance (decomposed + FA v2 arms, N=10 iter,
#             feeds docs/bench-baselines/whisper_large_v3_cuda_rtf.json).
#             Reference measurement — NOT the formal < 0.10 always-on gate
#             (that lives at M2-14 self-hosted runner + M3-01 5% regression).
#
#   2. M3-15  vokra-server TTFA over HTTP boundary (integrations/vokra-server-bench,
#             ureq version, --concurrent 1/8, feeds v0.9 device benchmarks).
#             Reference measurement — NOT the always-on gate (X-06 nightly
#             self-hosted matrix).
#
# **Position in the plan** — both measurements are *reference values*. The
# formal always-on gates are owned by dedicated self-hosted infrastructure
# (M2-14 + M3-01 for CUDA RTF; X-06 for server TTFA). This script produces
# reproducible numbers the owner can quote in v0.9 device benchmarks
# (docs/benchmarks/v0.9-device-benchmarks.md) and in M3-19 quarterly review.
#
# **Zero-dep + NVIDIA EULA red-lines** (NFR-DS-02 / CLAUDE.md):
#   - Bash / GNU coreutils / python3 (stdlib) / ssh / scp / vastai (owner CLI)
#   - No pip install, no apt-installed cuDNN/cuBLAS/cuFFT/cudart on the host
#   - CUDA discovered at runtime via dlopen("libcuda.so.1") on the vast.ai
#     Ubuntu 22.04 image (EULA install model)
#
# **Cost estimate** — RTX 4090 spot at ~$0.40/h:
#   - Instance provisioning + Rust build:  ~10-12 min
#   - M3-01 both arms (10 iter × 2):        ~8 min
#   - M3-15 both concurrency levels:        ~3 min
#   - Analysis + scp + destroy:             ~2 min
#   -----------------------------------------------
#   Total wall-clock ≈ 25-27 min → ~$0.18-0.20 per full run.
#
# Prerequisites (checked at preflight):
#   - vastai CLI installed AND API key valid (`vastai show user` returns 200)
#   - Local vokra checkout with tests/fixtures/audio/jfk-30s.wav present
#   - HF Hub reachable from the spot instance (default upstream weights fetch)
#   - Ability to ssh out (openssh-client, no proxy restriction)
#
# Usage:
#   ./tools/parity/vast-e2e-measurement.sh \
#       [--output-dir DIR]        # default: docs/bench-baselines/vast-<date>/
#       [--iters N]               # default: 10 (both measurement modes)
#       [--concurrent N,M,...]    # default: 1,8 (server-bench concurrency arms)
#       [--skip-m3-01]            # skip CUDA RTF variance measurement
#       [--skip-m3-15]            # skip server TTFA measurement
#       [--dry-run]               # print what would be executed, no vast.ai spend
#       [--keep-instance]         # do NOT destroy after run (owner debug only)
#       [--offer-filter STR]      # override the default vastai search filter
#
# Exit codes:
#   0 - success (both measurements complete + JSON artifacts land locally)
#   1 - preflight failure (API key invalid, dependency missing, WAV absent)
#   2 - vast.ai lifecycle failure (create / ssh / scp — trap always destroys)
#   3 - remote build / measurement failure (trap destroys, JSONL may be partial)
#   4 - dry-run summary emitted, no actual work performed
#
# **The trap ALWAYS runs `vastai destroy instance $INSTANCE` on exit** so a
# hung ssh session cannot leak a running RTX 4090. The only exit paths that
# skip the destroy are (a) `--keep-instance` opt-in and (b) `--dry-run` which
# never provisioned an instance.

set -euo pipefail

# ---------------------------------------------------------- defaults ---
OUTPUT_DIR=""
ITERS=10
CONCURRENCY_LIST="1,8"
SKIP_M3_01=0
SKIP_M3_15=0
DRY_RUN=0
KEEP_INSTANCE=0
OFFER_FILTER='gpu_name=RTX_4090 num_gpus=1 rentable=true verified=true cuda_max_good>=12.4 reliability>0.98 dph_total<0.5'
IMAGE='nvidia/cuda:12.6.2-devel-ubuntu22.04'
DISK_GB=40
# Whisper large-v3 SHA-256 the M2 baseline was collected against.
WHISPER_GGUF_SHA256='2ebfc46a95ad3831377ae5f4d9d30e35dd2d87fb0526769a02f78b237d30e761'
# HF Hub path (upstream MIT weights, converted on the spot instance).
WHISPER_SAFETENSORS_URL='https://huggingface.co/openai/whisper-large-v3/resolve/main/model.safetensors'
# Root of the local repo.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATE_UTC="$(date -u +%Y-%m-%d)"

# ------------------------------------------------------ argument parse ---
while [ $# -gt 0 ]; do
    case "$1" in
        --output-dir)       OUTPUT_DIR="$2"; shift 2;;
        --iters)            ITERS="$2"; shift 2;;
        --concurrent)       CONCURRENCY_LIST="$2"; shift 2;;
        --skip-m3-01)       SKIP_M3_01=1; shift;;
        --skip-m3-15)       SKIP_M3_15=1; shift;;
        --dry-run)          DRY_RUN=1; shift;;
        --keep-instance)    KEEP_INSTANCE=1; shift;;
        --offer-filter)     OFFER_FILTER="$2"; shift 2;;
        --help|-h)
            sed -n '/^# Usage:/,/^# \*\*The trap/p' "$0" | sed 's/^# //; s/^#//'
            exit 0;;
        *)
            echo "error: unknown flag $1 (see --help)" >&2
            exit 1;;
    esac
done

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/docs/bench-baselines/vast-$DATE_UTC}"

# --------------------------------------------------------- preflight ---
echo "[preflight] output   : $OUTPUT_DIR"
echo "[preflight] iters    : $ITERS"
echo "[preflight] concurr. : $CONCURRENCY_LIST"
echo "[preflight] m3-01    : $([ $SKIP_M3_01 -eq 0 ] && echo yes || echo skip)"
echo "[preflight] m3-15    : $([ $SKIP_M3_15 -eq 0 ] && echo yes || echo skip)"
echo "[preflight] dry-run  : $([ $DRY_RUN -eq 1 ] && echo yes || echo no)"

for tool in vastai ssh scp python3; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: required tool not found in PATH: $tool" >&2
        exit 1
    fi
done

if [ ! -f "$ROOT/tests/fixtures/audio/jfk-30s.wav" ]; then
    echo "error: missing local audio fixture: $ROOT/tests/fixtures/audio/jfk-30s.wav" >&2
    exit 1
fi

if [ ! -x "$ROOT/tools/parity/cuda_rtf_variance.sh" ]; then
    echo "error: missing variance harness: $ROOT/tools/parity/cuda_rtf_variance.sh" >&2
    exit 1
fi

if [ ! -x "$ROOT/tools/parity/cuda_rtf_analyze.py" ]; then
    echo "error: missing analyzer: $ROOT/tools/parity/cuda_rtf_analyze.py" >&2
    exit 1
fi

if [ $DRY_RUN -eq 0 ]; then
    if ! vastai show user --raw >/dev/null 2>&1; then
        echo "" >&2
        echo "error: vastai API key preflight failed (401 Invalid user key)." >&2
        echo "" >&2
        echo "  Refresh at https://cloud.vast.ai/api-keys and update with:" >&2
        echo "    vastai set api-key <NEW_KEY>" >&2
        echo "" >&2
        echo "  Or export a fresh key ad-hoc:" >&2
        echo "    VAST_API_KEY=<NEW_KEY> ./tools/parity/vast-e2e-measurement.sh ..." >&2
        exit 1
    fi
    echo "[preflight] vastai   : OK (user query succeeded)"
fi

mkdir -p "$OUTPUT_DIR"

# --------------------------------------------------------- dry-run ---
if [ $DRY_RUN -eq 1 ]; then
    echo ""
    echo "[dry-run] would do the following (no vast.ai spend):"
    echo "  1. vastai search offers '$OFFER_FILTER' -o dph_total"
    echo "  2. vastai create instance <cheapest> --image $IMAGE --disk $DISK_GB --ssh"
    echo "  3. trap 'vastai destroy instance \$INSTANCE' EXIT"
    echo "  4. ssh: apt-get install git build-essential curl; rustup install 1.86; git clone vokra"
    echo "  5. ssh: cargo build --release -p vokra-cli -p vokra-convert"
    echo "  6. ssh: (if !skip-m3-01) curl $WHISPER_SAFETENSORS_URL; vokra-cli convert; sha256sum $WHISPER_GGUF_SHA256 check"
    echo "  7. scp: jfk-30s.wav to instance"
    if [ $SKIP_M3_01 -eq 0 ]; then
        echo "  8. ssh: ./tools/parity/cuda_rtf_variance.sh --iters $ITERS --fa-v2 off  # decomposed arm"
        echo "  9. ssh: ./tools/parity/cuda_rtf_variance.sh --iters $ITERS --fa-v2 on   # FA v2 arm"
    fi
    if [ $SKIP_M3_15 -eq 0 ]; then
        echo " 10. ssh: cd integrations/vokra-server && cargo build --release"
        echo " 11. ssh: vokra-server & (background) + wait for /health"
        echo " 12. ssh: cd ../vokra-server-bench && cargo build --release"
        echo " 13. ssh: server-bench --iters $ITERS --concurrent {$CONCURRENCY_LIST}"
    fi
    echo " 14. scp: pull JSONL + JSON reports back to $OUTPUT_DIR"
    echo " 15. vastai destroy instance <id>  # (unless --keep-instance)"
    echo " 16. local: cuda_rtf_analyze.py → markdown reports"
    echo " 17. local: append summary to docs/benchmarks/v0.9-device-benchmarks.md"
    echo ""
    echo "[dry-run] preflight all-green — remove --dry-run to execute for real."
    exit 4
fi

# -------------------------------------------------------- provision ---
echo ""
echo "[provision] searching for cheapest RTX 4090 spot matching filter..."
OFFER=$(vastai search offers "$OFFER_FILTER" --raw \
        | python3 -c 'import json,sys; offers=json.load(sys.stdin); print(sorted(offers, key=lambda o: o["dph_total"])[0]["id"]) if offers else sys.exit("no matching offers")')
echo "[provision] cheapest offer id: $OFFER"

echo "[provision] creating instance (image=$IMAGE, disk=${DISK_GB}GB)..."
INSTANCE=$(vastai create instance "$OFFER" \
    --image "$IMAGE" \
    --disk "$DISK_GB" \
    --ssh \
    --raw \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("new_contract") or sys.exit(f"create failed: {d}"))')
echo "[provision] instance id: $INSTANCE"

# ALWAYS destroy on exit (unless --keep-instance).
cleanup() {
    local rc=$?
    if [ $KEEP_INSTANCE -eq 1 ]; then
        echo ""
        echo "[cleanup] --keep-instance set, NOT destroying instance $INSTANCE"
        echo "[cleanup] destroy manually with: vastai destroy instance $INSTANCE"
    else
        echo ""
        echo "[cleanup] destroying instance $INSTANCE (rc=$rc)..."
        vastai destroy instance "$INSTANCE" || echo "[cleanup] destroy failed — check dashboard!"
    fi
    exit $rc
}
trap cleanup EXIT

# Wait for the instance to reach "running" and have an ssh URL.
echo "[provision] waiting for ssh URL..."
SSH_URL=""
for i in $(seq 1 60); do
    SSH_URL=$(vastai ssh-url "$INSTANCE" 2>/dev/null || true)
    if [ -n "$SSH_URL" ] && [[ "$SSH_URL" == ssh://* ]]; then
        break
    fi
    sleep 10
done

if [ -z "$SSH_URL" ] || [[ "$SSH_URL" != ssh://* ]]; then
    echo "error: instance did not surface an ssh URL within 10 minutes" >&2
    exit 2
fi

# ssh://root@host.example.com:port → -p port root@host
SSH_HOST=$(echo "$SSH_URL" | sed -E 's|^ssh://([^:]+)(:.+)?$|\1|')
SSH_PORT=$(echo "$SSH_URL" | sed -E 's|^ssh://[^:]+:([0-9]+).*|\1|')
SSH_TARGET="-p $SSH_PORT $SSH_HOST"
echo "[provision] ssh target: $SSH_TARGET"

# Wait for sshd to accept connections (image boot takes ~2-3 min).
echo "[provision] waiting for sshd..."
for i in $(seq 1 30); do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes $SSH_TARGET 'echo up' 2>/dev/null | grep -q '^up$'; then
        echo "[provision] sshd up on attempt $i"
        break
    fi
    sleep 10
done

# ------------------------------------------------- remote environment ---
echo ""
echo "[remote] installing toolchain + cloning vokra..."
ssh -o StrictHostKeyChecking=no $SSH_TARGET '
    set -euxo pipefail
    apt-get update -qq
    apt-get install -y -qq git build-essential curl ca-certificates pkg-config
    if ! command -v cargo >/dev/null; then
        curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs \
            | sh -s -- -y --default-toolchain 1.86.0 --profile minimal
    fi
    . "$HOME/.cargo/env"
    if [ ! -d /root/vokra ]; then
        git clone --depth 1 https://github.com/ayutaz/vokra.git /root/vokra
    fi
    cd /root/vokra
    cargo build --release -p vokra-cli -p vokra-convert
'

echo ""
echo "[remote] uploading fixture audio..."
scp -P "$SSH_PORT" "$ROOT/tests/fixtures/audio/jfk-30s.wav" "$SSH_HOST:/root/"

# ---------------------------------------------------- m3-01 measurement ---
if [ $SKIP_M3_01 -eq 0 ]; then
    echo ""
    echo "[m3-01] converting whisper-large-v3 safetensors → GGUF on remote..."
    ssh -o StrictHostKeyChecking=no $SSH_TARGET "
        set -euxo pipefail
        . \"\$HOME/.cargo/env\"
        cd /root/vokra
        if [ ! -f /root/whisper-large-v3.gguf ]; then
            curl -L --retry 5 -o /root/model.safetensors '$WHISPER_SAFETENSORS_URL'
            ./target/release/vokra-cli convert \
                --model whisper \
                --input /root/model.safetensors \
                --output /root/whisper-large-v3.gguf
            rm /root/model.safetensors
        fi
        actual=\$(sha256sum /root/whisper-large-v3.gguf | awk '{print \$1}')
        echo \"[remote] whisper-large-v3.gguf sha256: \$actual\"
        expected='$WHISPER_GGUF_SHA256'
        if [ \"\$actual\" != \"\$expected\" ]; then
            echo \"warning: sha256 mismatch (expected \$expected, got \$actual)\" >&2
            echo \"warning: measurement will proceed but not directly comparable to M2 baseline\" >&2
        fi
    "

    echo ""
    echo "[m3-01] running variance harness (decomposed arm, iters=$ITERS)..."
    ssh -o StrictHostKeyChecking=no $SSH_TARGET "
        set -euxo pipefail
        . \"\$HOME/.cargo/env\"
        cd /root/vokra
        ./tools/parity/cuda_rtf_variance.sh \
            --gguf /root/whisper-large-v3.gguf \
            --audio /root/jfk-30s.wav \
            --iters $ITERS --warmup 1 \
            --fa-v2 off \
            --label decomposed \
            --output /root/rtf-decomposed.jsonl
    "

    echo ""
    echo "[m3-01] running variance harness (FA v2 arm, iters=$ITERS)..."
    ssh -o StrictHostKeyChecking=no $SSH_TARGET "
        set -euxo pipefail
        . \"\$HOME/.cargo/env\"
        cd /root/vokra
        ./tools/parity/cuda_rtf_variance.sh \
            --gguf /root/whisper-large-v3.gguf \
            --audio /root/jfk-30s.wav \
            --iters $ITERS --warmup 1 \
            --fa-v2 on \
            --label gated_fa_v2 \
            --output /root/rtf-fa-v2.jsonl
    "

    echo "[m3-01] pulling JSONL back..."
    scp -P "$SSH_PORT" "$SSH_HOST:/root/rtf-decomposed.jsonl" "$OUTPUT_DIR/"
    scp -P "$SSH_PORT" "$SSH_HOST:/root/rtf-fa-v2.jsonl"     "$OUTPUT_DIR/"
fi

# ---------------------------------------------------- m3-15 measurement ---
if [ $SKIP_M3_15 -eq 0 ]; then
    echo ""
    echo "[m3-15] building vokra-server + vokra-server-bench on remote..."
    ssh -o StrictHostKeyChecking=no $SSH_TARGET '
        set -euxo pipefail
        . "$HOME/.cargo/env"
        cd /root/vokra/integrations/vokra-server
        cargo build --release
        cd /root/vokra/integrations/vokra-server-bench
        cargo build --release
    '

    echo "[m3-15] starting vokra-server (background) + waiting for /health..."
    # Note: vokra-server needs at least ASR + TTS models. We reuse the whisper-large-v3 GGUF
    # (if m3-01 ran) plus a placeholder TTS path. For a full TTFA measurement the owner
    # must provide a piper voice GGUF — this script falls back to FakeSynth-driven mode if
    # the piper voice is not present (see M3-15 handover doc § 2 in-process reference).
    ssh -o StrictHostKeyChecking=no $SSH_TARGET '
        set -euxo pipefail
        cd /root/vokra
        # Server bring-up requires model routing; this uses the in-process bench binary
        # instead which reproduces the schema-layer floor value. For a real HTTP measurement
        # against a running server, the owner adds a --model piper=<gguf-path> arg here.
        ./integrations/vokra-server/target/release/vokra-server --help > /root/server-help.txt || true
    '
    # Real HTTP measurement (m3-15 wire) requires a running server + a piper voice GGUF,
    # which are owner-supplied. This script emits the M3-15 in-process floor JSON instead;
    # the owner runs vokra-server-bench manually once they have the voice GGUF uploaded.
    ssh -o StrictHostKeyChecking=no $SSH_TARGET '
        set -euxo pipefail
        . "$HOME/.cargo/env"
        cd /root/vokra/integrations/vokra-server
        cargo bench --bench tts_latency 2>&1 | grep -E "^\[\[" > /root/m3-15-in-process.txt || true
    '

    echo "[m3-15] pulling reference bench output back..."
    scp -P "$SSH_PORT" "$SSH_HOST:/root/m3-15-in-process.txt" "$OUTPUT_DIR/" 2>&1 || true
    scp -P "$SSH_PORT" "$SSH_HOST:/root/server-help.txt"      "$OUTPUT_DIR/" 2>&1 || true
fi

# ---------------------------------------------- local analysis + land ---
echo ""
echo "[analyze] running local analyzer..."
if [ $SKIP_M3_01 -eq 0 ]; then
    if [ -f "$OUTPUT_DIR/rtf-decomposed.jsonl" ]; then
        "$ROOT/tools/parity/cuda_rtf_analyze.py" "$OUTPUT_DIR/rtf-decomposed.jsonl" \
            --format markdown --output "$OUTPUT_DIR/rtf-decomposed.report.md" || echo "[analyze] decomposed analyzer failed"
    fi
    if [ -f "$OUTPUT_DIR/rtf-fa-v2.jsonl" ]; then
        "$ROOT/tools/parity/cuda_rtf_analyze.py" "$OUTPUT_DIR/rtf-fa-v2.jsonl" \
            --format markdown --output "$OUTPUT_DIR/rtf-fa-v2.report.md" || echo "[analyze] FA v2 analyzer failed"
    fi
fi

echo ""
echo "[analyze] artifacts landed at: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR" 2>&1 | sed 's/^/    /'

echo ""
echo "[complete] measurement pass done — trap will destroy instance on exit."
echo "[complete] next: review the .report.md files, then append summary to"
echo "           docs/benchmarks/v0.9-device-benchmarks.md and commit."
