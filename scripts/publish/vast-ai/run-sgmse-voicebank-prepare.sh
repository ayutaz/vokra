#!/usr/bin/env bash
# VAST/Linux-only preparation of the fixed SGMSE VoiceBank checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
PREPARER="$VOKRA_ROOT/tools/parity/sgmse_prepare_checkpoint.py"
CONTRACT="$VOKRA_ROOT/tools/parity/fixtures/sgmse_voicebank_typed_contract_v1.json"
INPUT_DIR="/workspace/vokra-sgmse-voicebank-inspection/hf"
OUTPUT_DIR="/workspace/vokra-sgmse-voicebank-prepared"
UV_CACHE_DIR_SGMSE="${SGMSE_UV_CACHE_DIR:-/tmp/vokra-sgmse-uv-cache}"

log() { printf '[sgmse-voicebank-prepare-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF'
usage: run-sgmse-voicebank-prepare.sh [--input-dir <inspection-hf-dir>] [--output-dir <absent-dir>]
       run-sgmse-voicebank-prepare.sh --self-test

VAST/Linux-only preparation of the fixed SpeechBrain SGMSE checkpoint. It
emits one safetensors file and its strict sidecar; no GGUF conversion, upload,
push, or model execution is performed.
EOF
}

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token
  for token in 'VOKRA_PUBLISH_ON_VAST=1' 'Linux' 'x86_64' 'sgmse_prepare_checkpoint.py' \
    '--prepare' '--ckpt' '--output' '--sidecar' '--contract' 'score_model_ema.ckpt' \
    'sgmse_voicebank_typed_contract_v1.json' 'vokra-sgmse-voicebank-prepared-v1' \
    'NO_UPLOAD' 'no GGUF' 'output-dir must be absent' 'git status --porcelain'; do
    if ! grep -Fq -- "$token" "$path"; then
      log "self-test FAIL: missing contract token: $token"
      fail=1
    fi
  done
  for token in 'PREPARED_FORMAT' 'weights_only=True' 'map_location="cpu"' \
    'typed_manifest_sha256' 'REVIEWED_TENSOR_MANIFEST_SHA256'; do
    if ! grep -Fq -- "$token" "$PREPARER"; then
      log "self-test FAIL: preparer is missing safety token: $token"
      fail=1
    fi
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|--push|--upload)([[:space:]]|$)' "$path" >/dev/null; then
    log 'self-test FAIL: publication command found'
    fail=1
  fi
  if ! UV_CACHE_DIR="$UV_CACHE_DIR_SGMSE" uv run --frozen --no-sync --project "$PARITY_PROJECT" \
    --python 3.12 python "$PREPARER" --self-test >/dev/null; then
    log 'self-test FAIL: preparer self-test failed'
    fail=1
  fi
  if "$path" --self-test --output-dir /tmp/not-accepted >/dev/null 2>&1; then
    log 'self-test FAIL: extra argument accepted'
    fail=1
  fi
  if "$path" --unknown-flag >/dev/null 2>&1; then
    log 'self-test FAIL: unknown argument accepted'
    fail=1
  fi
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

self=0
while (($#)); do
  case "$1" in
    --self-test) self=1; shift ;;
    --input-dir) (($# >= 2)) || die '--input-dir requires a path'; INPUT_DIR="$2"; shift 2 ;;
    --output-dir) (($# >= 2)) || die '--output-dir requires a path'; OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  [[ "$INPUT_DIR" == "/workspace/vokra-sgmse-voicebank-inspection/hf" ]] || die '--self-test accepts no other arguments'
  [[ "$OUTPUT_DIR" == "/workspace/vokra-sgmse-voicebank-prepared" ]] || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi

[[ "$(uname -s)" == Linux ]] || die 'SGMSE preparation is VAST/Linux-only'
[[ "$(uname -m)" == x86_64 ]] || die 'VAST host must be x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
[[ -f "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" ]] || die 'locked parity project is missing'
[[ -f "$PREPARER" && -f "$CONTRACT" ]] || die 'SGMSE preparation files are missing'
[[ -d "$INPUT_DIR" && ! -L "$INPUT_DIR" ]] || die 'input inspection directory is missing or symlinked'
[[ -f "$INPUT_DIR/score_model_ema.ckpt" && ! -L "$INPUT_DIR/score_model_ema.ckpt" ]] || die 'fixed checkpoint is missing'
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || die 'output-dir must be absent (no-clobber)'
for tool in git uv; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done

mkdir "$OUTPUT_DIR"
cleanup_empty_output() {
  if [[ ! -e "$OUTPUT_DIR/sgmse_voicebank.safetensors" && ! -e "$OUTPUT_DIR/sgmse_voicebank.manifest.json" ]]; then
    rmdir "$OUTPUT_DIR" 2>/dev/null || true
  fi
}
trap cleanup_empty_output EXIT
export UV_CACHE_DIR="$UV_CACHE_DIR_SGMSE"
UV_CACHE_DIR="$UV_CACHE_DIR_SGMSE" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python \
  "$PREPARER" --prepare \
  --ckpt "$INPUT_DIR/score_model_ema.ckpt" \
  --output "$OUTPUT_DIR/sgmse_voicebank.safetensors" \
  --sidecar "$OUTPUT_DIR/sgmse_voicebank.manifest.json" \
  --contract "$CONTRACT"
[[ -s "$OUTPUT_DIR/sgmse_voicebank.safetensors" ]] || die 'prepared safetensors is missing'
[[ -s "$OUTPUT_DIR/sgmse_voicebank.manifest.json" ]] || die 'prepared sidecar is missing'
log 'preparation complete: safetensors and sidecar only; NO_UPLOAD'
