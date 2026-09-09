#!/usr/bin/env bash
# VAST/Linux-only independent SGMSE score reference. This consumes an existing
# inspection directory, imports only the pinned upstream source, and writes
# deterministic score fixtures to VAST storage. It never emits GGUF or uploads.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
REFERENCE_TOOL="$VOKRA_ROOT/tools/parity/sgmse_dump_reference.py"
VERIFY_TOOL="$VOKRA_ROOT/tools/parity/sgmse_verify_reference.py"
NATIVE_PARITY_TOOL="$VOKRA_ROOT/tools/parity/sgmse_native_score_parity.py"
INSPECTION_DIR="/workspace/vokra-sgmse-voicebank-inspection"
OUTPUT_DIR="/workspace/vokra-sgmse-voicebank-reference"
UV_CACHE_DIR_SGMSE="${SGMSE_UV_CACHE_DIR:-/tmp/vokra-sgmse-uv-cache}"

log() { printf '[sgmse-voicebank-reference-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF'
usage: run-sgmse-voicebank-reference.sh [--inspection-dir <dir>] [--output-dir <absent-dir>]
       [--native-score-dir <dir>]
       run-sgmse-voicebank-reference.sh --self-test

VAST/Linux-only independent reference for the fixed SGMSE VoiceBank model.
The pinned source is imported directly, the checkpoint uses weights_only=True,
and deterministic score tensors are written under output-dir. No GGUF or
publication action is performed. When native-score-dir is supplied, its
score_real.f32 and score_imag.f32 are compared against the completed fixture.
EOF
}

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token
  for token in 'VOKRA_PUBLISH_ON_VAST=1' 'Linux' 'x86_64' 'CARGO_BUILD_JOBS=1' \
    'sgmse_dump_reference.py' 'vokra-sgmse-score-reference-v1' \
    'sgmse_verify_reference.py' 'sgmse_native_score_parity.py' \
    'REFERENCE_MANIFEST_VERIFIED' \
    '5ebd87c6257537c3997c134b279d85cd7bebccce0e6d3fc68f7a36f15096aa51' \
    'REFERENCE_COMPLETE_NO_UPLOAD' 'BLOCKED_INDEPENDENT_REFERENCE_UNAVAILABLE' \
    'SOURCE_ROUTE_VERIFIED_STRICT_LOAD' 'weights_only=True' 'strict=True' \
    'construction_evidence' 'vokra-sgmse-construction-evidence-v1' \
    'ncsnpp_all_modules' 'named_modules' 'canonical_sha256' \
    'score_model_ema.ckpt' 'NO_UPLOAD' 'fixture payload retained' \
    'run.log' 'torch_deterministic_algorithms' 'inspection-manifest' \
    'native-score-dir' \
    'git status --porcelain' 'output-dir must be absent'; do
    if ! grep -Fq -- "$token" "$path"; then
      log "self-test FAIL: missing contract token: $token"
      fail=1
    fi
  done
  if ! grep -Fq -- 'command -v ninja' "$path"; then
    log 'self-test FAIL: missing native ninja preflight'
    fail=1
  fi
  if grep -Fq -- "mkdir -p \"\$OUTPUT_DIR\"" "$path"; then
    log 'self-test FAIL: wrapper pre-created output directory'
    fail=1
  fi
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|--push|--upload)([[:space:]]|$)' "$path" >/dev/null; then
    log 'self-test FAIL: publication command found'
    fail=1
  fi
  if ! UV_CACHE_DIR="$UV_CACHE_DIR_SGMSE" uv run --frozen --no-sync --project "$PARITY_PROJECT" \
    --python 3.12 python "$REFERENCE_TOOL" --self-test >/dev/null; then
    log 'self-test FAIL: reference tool self-test failed'
    fail=1
  fi
  if ! UV_CACHE_DIR="$UV_CACHE_DIR_SGMSE" uv run --frozen --no-sync --project "$PARITY_PROJECT" \
    --python 3.12 python "$NATIVE_PARITY_TOOL" --self-test >/dev/null; then
    log 'self-test FAIL: native score parity tool self-test failed'
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
NATIVE_SCORE_DIR=""
while (($#)); do
  case "$1" in
    --self-test) self=1; shift ;;
    --inspection-dir) (($# >= 2)) || die '--inspection-dir requires a path'; INSPECTION_DIR="$2"; shift 2 ;;
    --output-dir) (($# >= 2)) || die '--output-dir requires a path'; OUTPUT_DIR="$2"; shift 2 ;;
    --native-score-dir) (($# >= 2)) || die '--native-score-dir requires a path'; NATIVE_SCORE_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  [[ "$INSPECTION_DIR" == "/workspace/vokra-sgmse-voicebank-inspection" ]] || die '--self-test accepts no other arguments'
  [[ "$OUTPUT_DIR" == "/workspace/vokra-sgmse-voicebank-reference" ]] || die '--self-test accepts no other arguments'
  [[ -z "$NATIVE_SCORE_DIR" ]] || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi

[[ "$(uname -s)" == Linux ]] || die 'SGMSE reference work is VAST/Linux-only'
[[ "$(uname -m)" == x86_64 ]] || die 'VAST host must be x86_64'
command -v ninja >/dev/null 2>&1 || die 'ninja is missing; run provision.sh first'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
[[ -f "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" ]] || die 'locked parity project is missing'
[[ -f "$REFERENCE_TOOL" ]] || die 'SGMSE reference tool is missing'
[[ -f "$VERIFY_TOOL" ]] || die 'SGMSE reference verifier is missing'
[[ -f "$INSPECTION_DIR/evidence/sgmse_voicebank_manifest.json" ]] || die 'inspection manifest is missing'
[[ -f "$INSPECTION_DIR/hf/score_model_ema.ckpt" ]] || die 'inspected checkpoint is missing'
[[ -f "$INSPECTION_DIR/hf/hyperparams.yaml" ]] || die 'inspected hyperparams are missing'
[[ -d "$INSPECTION_DIR/source/repo/.git" ]] || die 'pinned SGMSE source checkout is missing'
[[ -d "$INSPECTION_DIR/source/speechbrain/.git" ]] || die 'pinned SpeechBrain source checkout is missing'
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || die 'output-dir must be absent (no-clobber)'

export CARGO_BUILD_JOBS=1
export UV_CACHE_DIR="$UV_CACHE_DIR_SGMSE"
set +e
UV_CACHE_DIR="$UV_CACHE_DIR_SGMSE" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python \
  "$REFERENCE_TOOL" \
  --source-dir "$INSPECTION_DIR/source/repo" \
  --speechbrain-source-dir "$INSPECTION_DIR/source/speechbrain" \
  --checkpoint "$INSPECTION_DIR/hf/score_model_ema.ckpt" \
  --hyperparams "$INSPECTION_DIR/hf/hyperparams.yaml" \
  --inspection-manifest "$INSPECTION_DIR/evidence/sgmse_voicebank_manifest.json" \
  --output-dir "$OUTPUT_DIR" \
  --vokra-root "$VOKRA_ROOT"
reference_rc=$?
set -e
if (( reference_rc != 0 )); then
  log "reference blocked; preserve any evidence under $OUTPUT_DIR"
  exit "$reference_rc"
fi
[[ -s "$OUTPUT_DIR/manifest.json" ]] || die 'reference manifest is missing'
set +e
UV_CACHE_DIR="$UV_CACHE_DIR_SGMSE" uv run --frozen --no-sync --project "$PARITY_PROJECT" --python 3.12 python \
  "$VERIFY_TOOL" \
  --manifest "$OUTPUT_DIR/manifest.json" \
  --output-dir "$OUTPUT_DIR" \
  --vokra-root "$VOKRA_ROOT"
verify_rc=$?
set -e
(( verify_rc == 0 )) || die "completion verifier failed: rc=$verify_rc"
if [[ -n "$NATIVE_SCORE_DIR" ]]; then
  set +e
  UV_CACHE_DIR="$UV_CACHE_DIR_SGMSE" uv run --frozen --no-sync --project "$PARITY_PROJECT" --python 3.12 python \
    "$NATIVE_PARITY_TOOL" \
    --reference-dir "$OUTPUT_DIR" \
    --native-dir "$NATIVE_SCORE_DIR"
  parity_rc=$?
  set -e
  (( parity_rc == 0 )) || die "native score parity failed: rc=$parity_rc"
fi
log "reference complete and verified: evidence=$OUTPUT_DIR; fixture payload retained for native parity; no conversion or upload performed"
