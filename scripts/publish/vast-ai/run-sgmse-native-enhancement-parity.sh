#!/usr/bin/env bash
# VAST/Linux-only SGMSE end-to-end CPU enhancement parity.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
TOOL="$PARITY_PROJECT/sgmse_native_enhancement_parity.py"
REFERENCE_DIR="/workspace/vokra-sgmse-enhancement-reference"
NATIVE_OUTPUT_DIR="/workspace/vokra-sgmse-enhancement-native"
INSPECTION_DIR="/workspace/vokra-sgmse-voicebank-inspection"
SOURCE_DIR=""
SPEECHBRAIN_SOURCE_DIR=""
CHECKPOINT=""
HYPERPARAMS=""
INSPECTION_MANIFEST=""
INPUT_WAV=""
GGUF=""
GGUF_SHA256=""
SELF_TEST=0
GENERATE_REFERENCE=0

log() { printf '[sgmse-native-enhancement-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }

usage() {
  cat <<'EOF'
usage: run-sgmse-native-enhancement-parity.sh \
  --gguf <absolute-authenticated-gguf> --gguf-sha256 <64-hex-digest> \
  --reference-dir <absolute-verified-enhancement-packet> \
  --native-output-dir <absolute-absent-tmpfs-dir>
       run-sgmse-native-enhancement-parity.sh --generate-reference \
  [--inspection-dir <dir>] [--reference-dir <absolute-absent-dir>]
       run-sgmse-native-enhancement-parity.sh --self-test

The --generate-reference path consumes the authenticated source/checkpoint/
hyperparameters/inspection inputs from the VoiceBank reference inspection and
the authenticated tests/parity/utmos/ref-clip.wav fixture. It derives the
first 4096 samples (centered n_fft=510/hop=128: 33 frames, reflection-padded
to 64 frames) and binds that crop provenance and hash in the packet before
calling the official upstream
ScoreModel.enhance path with captured prior/corrector/predictor noise. The
native path consumes that exact packet. Neither path downloads a model,
converts GGUF, or publishes.
EOF
}

self_test() {
  local fail=0 token
  for token in \
    'VOKRA_PUBLISH_ON_VAST=1' 'VAST/Linux-only' 'x86_64' \
    'sgmse_native_enhancement_parity.py' 'SGMSEEnhancement.enhance_batch' \
    'ScoreModel.enhance' 'speechbrain/inference/enhancement.py' 'CUDA' \
    'REFERENCE_COMPLETE_NO_UPLOAD' 'CPU_ENHANCEMENT_PARITY_PASS' \
    'prior/corrector/predictor' 'noise_calls.txt' 'NO_UPLOAD' \
    'cargo test --locked --release --test sgmse_native_enhancement -p vokra-models' \
    '-- --ignored --exact --show-output' 'findmnt' 'tmpfs' \
    'reference packet' '--generate-reference' 'score_model_ema.ckpt' \
    'hyperparams.yaml' 'sgmse_voicebank_manifest.json' 'ref-clip.wav' \
    'first 4096 samples' '33 frames' 'reflection-padded' '64 frames' \
    'official upstream' 'git status --porcelain --untracked-files=all' \
    'output must be absent (no-clobber)' 'native output parent must be tmpfs'; do
    grep -Fq -- "$token" "$0" || { log "self-test FAIL: missing token: $token"; fail=1; }
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|--push|--upload)([[:space:]]|$)' "$0" >/dev/null; then
    log 'self-test FAIL: publication command found'
    fail=1
  fi
  if ! UV_CACHE_DIR="$(printenv UV_CACHE_DIR || printf /tmp/vokra-sgmse-uv-cache)" \
    uv run --frozen --no-sync --project "$PARITY_PROJECT" --python 3.12 python \
    "$TOOL" --self-test >/dev/null; then
    log 'self-test FAIL: enhancement tool self-test failed'
    fail=1
  fi
  if "$0" --self-test --gguf /tmp/not-accepted >/dev/null 2>&1; then
    log 'self-test FAIL: extra argument accepted with --self-test'
    fail=1
  fi
  if "$0" --self-test --generate-reference >/dev/null 2>&1; then
    log 'self-test FAIL: generation accepted with --self-test'
    fail=1
  fi
  if "$0" --unknown-flag >/dev/null 2>&1; then
    log 'self-test FAIL: unknown argument accepted'
    fail=1
  fi
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

while (($#)); do
  case "$1" in
    --self-test) (( SELF_TEST == 0 )) || die 'duplicate --self-test'; SELF_TEST=1; shift ;;
    --generate-reference) (( GENERATE_REFERENCE == 0 )) || die 'duplicate --generate-reference'; GENERATE_REFERENCE=1; shift ;;
    --gguf) (($# >= 2)) || die '--gguf requires a path'; GGUF="$2"; shift 2 ;;
    --gguf-sha256) (($# >= 2)) || die '--gguf-sha256 requires a digest'; GGUF_SHA256="$2"; shift 2 ;;
    --reference-dir) (($# >= 2)) || die '--reference-dir requires a path'; REFERENCE_DIR="$2"; shift 2 ;;
    --native-output-dir) (($# >= 2)) || die '--native-output-dir requires a path'; NATIVE_OUTPUT_DIR="$2"; shift 2 ;;
    --inspection-dir) (($# >= 2)) || die '--inspection-dir requires a path'; INSPECTION_DIR="$2"; shift 2 ;;
    --source-dir) (($# >= 2)) || die '--source-dir requires a path'; SOURCE_DIR="$2"; shift 2 ;;
    --speechbrain-source-dir) (($# >= 2)) || die '--speechbrain-source-dir requires a path'; SPEECHBRAIN_SOURCE_DIR="$2"; shift 2 ;;
    --checkpoint) (($# >= 2)) || die '--checkpoint requires a path'; CHECKPOINT="$2"; shift 2 ;;
    --hyperparams) (($# >= 2)) || die '--hyperparams requires a path'; HYPERPARAMS="$2"; shift 2 ;;
    --inspection-manifest) (($# >= 2)) || die '--inspection-manifest requires a path'; INSPECTION_MANIFEST="$2"; shift 2 ;;
    --input-wav) (($# >= 2)) || die '--input-wav requires a path'; INPUT_WAV="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( SELF_TEST )); then
  [[ "$GENERATE_REFERENCE" == 0 && -z "$GGUF$GGUF_SHA256$SOURCE_DIR$SPEECHBRAIN_SOURCE_DIR$CHECKPOINT$HYPERPARAMS$INSPECTION_MANIFEST$INPUT_WAV" && "$REFERENCE_DIR" == "/workspace/vokra-sgmse-enhancement-reference" && "$NATIVE_OUTPUT_DIR" == "/workspace/vokra-sgmse-enhancement-native" ]] || die '--self-test accepts no other arguments'
  self_test
  exit 0
fi

if (( GENERATE_REFERENCE )); then
  [[ -z "$GGUF$GGUF_SHA256" ]] || die '--generate-reference cannot combine with GGUF arguments'
  [[ "$(uname -s)" == Linux ]] || die 'SGMSE enhancement reference is VAST/Linux-only'
  [[ "$(uname -m)" == x86_64 ]] || die 'SGMSE enhancement reference requires x86_64 VAST'
  [[ "$(printenv VOKRA_PUBLISH_ON_VAST || true)" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
  [[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST Vokra checkout must be clean'
  [[ -f "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" && -f "$TOOL" ]] || die 'parity project/tool is missing'
  [[ "$REFERENCE_DIR" == /* && "$VOKRA_ROOT" == /* ]] || die 'reference and Vokra paths must be absolute'
  [[ -z "$SOURCE_DIR" ]] && SOURCE_DIR="$INSPECTION_DIR/source/repo"
  [[ -z "$SPEECHBRAIN_SOURCE_DIR" ]] && SPEECHBRAIN_SOURCE_DIR="$INSPECTION_DIR/source/speechbrain"
  [[ -z "$CHECKPOINT" ]] && CHECKPOINT="$INSPECTION_DIR/hf/score_model_ema.ckpt"
  [[ -z "$HYPERPARAMS" ]] && HYPERPARAMS="$INSPECTION_DIR/hf/hyperparams.yaml"
  [[ -z "$INSPECTION_MANIFEST" ]] && INSPECTION_MANIFEST="$INSPECTION_DIR/evidence/sgmse_voicebank_manifest.json"
  [[ -z "$INPUT_WAV" ]] && INPUT_WAV="$VOKRA_ROOT/tests/parity/utmos/ref-clip.wav"
  for path in "$SOURCE_DIR" "$SPEECHBRAIN_SOURCE_DIR" "$CHECKPOINT" "$HYPERPARAMS" "$INSPECTION_MANIFEST" "$INPUT_WAV"; do
    [[ "$path" == /* ]] || die 'all generation inputs must be absolute'
  done
  [[ -d "$SOURCE_DIR" && ! -L "$SOURCE_DIR" ]] || die 'SGMSE source checkout is missing or symlinked'
  [[ -d "$SPEECHBRAIN_SOURCE_DIR" && ! -L "$SPEECHBRAIN_SOURCE_DIR" ]] || die 'SpeechBrain source checkout is missing or symlinked'
  [[ -f "$CHECKPOINT" && ! -L "$CHECKPOINT" ]] || die 'checkpoint is missing or symlinked'
  [[ -f "$HYPERPARAMS" && ! -L "$HYPERPARAMS" ]] || die 'hyperparams are missing or symlinked'
  [[ -f "$INSPECTION_MANIFEST" && ! -L "$INSPECTION_MANIFEST" ]] || die 'inspection manifest is missing or symlinked'
  [[ -f "$INPUT_WAV" && ! -L "$INPUT_WAV" ]] || die 'fixed input fixture is missing or symlinked'
  [[ ! -e "$REFERENCE_DIR" && ! -L "$REFERENCE_DIR" ]] || die 'reference output must be absent (no-clobber)'
  command -v uv >/dev/null 2>&1 || die 'uv is missing'
  UV_CACHE_DIR="$(printenv UV_CACHE_DIR || printf /tmp/vokra-sgmse-uv-cache)" \
    uv run --frozen --no-sync --project "$PARITY_PROJECT" --python 3.12 python \
    "$TOOL" --generate-reference \
    --source-dir "$SOURCE_DIR" \
    --speechbrain-source-dir "$SPEECHBRAIN_SOURCE_DIR" \
    --checkpoint "$CHECKPOINT" \
    --hyperparams "$HYPERPARAMS" \
    --inspection-manifest "$INSPECTION_MANIFEST" \
    --input-wav "$INPUT_WAV" \
    --output-dir "$REFERENCE_DIR" \
    --vokra-root "$VOKRA_ROOT"
  UV_CACHE_DIR="$(printenv UV_CACHE_DIR || printf /tmp/vokra-sgmse-uv-cache)" \
    uv run --frozen --no-sync --project "$PARITY_PROJECT" --python 3.12 python \
    "$TOOL" --verify-reference --reference-dir "$REFERENCE_DIR" --vokra-root "$VOKRA_ROOT"
  log "official enhancement reference complete and verified: no upload performed"
  exit 0
fi

[[ -z "$SOURCE_DIR$SPEECHBRAIN_SOURCE_DIR$CHECKPOINT$HYPERPARAMS$INSPECTION_MANIFEST$INPUT_WAV" && "$INSPECTION_DIR" == "/workspace/vokra-sgmse-voicebank-inspection" ]] || die 'generation-only arguments require --generate-reference'
[[ -n "$GGUF" && -n "$GGUF_SHA256" ]] || { usage >&2; exit 1; }
[[ "$(uname -s)" == Linux ]] || die 'SGMSE enhancement parity is VAST/Linux-only'
[[ "$(uname -m)" == x86_64 ]] || die 'SGMSE enhancement parity requires x86_64 VAST'
[[ "$(printenv VOKRA_PUBLISH_ON_VAST || true)" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST Vokra checkout must be clean'
[[ -f "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" && -f "$TOOL" ]] || die 'parity project/tool is missing'
[[ "$GGUF" == /* && "$REFERENCE_DIR" == /* && "$NATIVE_OUTPUT_DIR" == /* ]] || die 'all paths must be absolute'
[[ "$GGUF_SHA256" =~ ^[0-9a-f]{64}$ ]] || die '--gguf-sha256 must be lowercase 64-hex'
[[ -f "$GGUF" && ! -L "$GGUF" ]] || die 'GGUF is missing or symlinked'
[[ -d "$REFERENCE_DIR" && ! -L "$REFERENCE_DIR" ]] || die 'reference packet is missing or symlinked'
[[ ! -e "$NATIVE_OUTPUT_DIR" && ! -L "$NATIVE_OUTPUT_DIR" ]] || die 'native output must be absent'
native_parent="$(dirname "$NATIVE_OUTPUT_DIR")"
[[ -d "$native_parent" && ! -L "$native_parent" ]] || die 'native output parent must be real'
[[ "$(findmnt -T "$native_parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die 'native output parent must be tmpfs'
command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is missing'
command -v awk >/dev/null 2>&1 || die 'awk is missing'
command -v cargo >/dev/null 2>&1 || die 'cargo is missing'
command -v uv >/dev/null 2>&1 || die 'uv is missing'

# Independent packet verification is the first data operation.
UV_CACHE_DIR="$(printenv UV_CACHE_DIR || printf /tmp/vokra-sgmse-uv-cache)" \
  uv run --frozen --no-sync --project "$PARITY_PROJECT" --python 3.12 python \
  "$TOOL" --verify-reference --reference-dir "$REFERENCE_DIR" --vokra-root "$VOKRA_ROOT"
actual_gguf_sha256="$(sha256sum "$GGUF" | awk '{print $1}')"
[[ "$actual_gguf_sha256" == "$GGUF_SHA256" ]] || die "GGUF SHA-256 mismatch: got $actual_gguf_sha256"

cargo_build_jobs="$(printenv CARGO_BUILD_JOBS || printf 1)"
export CARGO_BUILD_JOBS="$cargo_build_jobs"
tmpdir="$(printenv TMPDIR || printf /tmp)"
native_log="$(mktemp "$tmpdir/vokra-sgmse-native-enhancement.XXXXXX")"
trap 'rm -f -- "$native_log"' EXIT
VOKRA_SGMSE_GGUF="$GGUF" \
VOKRA_SGMSE_GGUF_SHA256="$GGUF_SHA256" \
VOKRA_SGMSE_REFERENCE_DIR="$REFERENCE_DIR" \
VOKRA_SGMSE_NATIVE_OUTPUT_DIR="$NATIVE_OUTPUT_DIR" \
VOKRA_PUBLISH_ON_VAST=1 \
  cargo test --locked --release --test sgmse_native_enhancement -p vokra-models \
    -- --ignored --exact --show-output 2>&1 | tee "$native_log"
[[ "$(grep -Fxc 'test sgmse_native_enhancement_matches_official_reference ... ok' "$native_log" || true)" == 1 ]] || die 'native enhancement test did not pass exactly once'
[[ -f "$NATIVE_OUTPUT_DIR/enhanced_pcm.f32" ]] || die 'native enhancement output is missing'

UV_CACHE_DIR="$(printenv UV_CACHE_DIR || printf /tmp/vokra-sgmse-uv-cache)" \
  uv run --frozen --no-sync --project "$PARITY_PROJECT" --python 3.12 python \
  "$TOOL" --compare --reference-dir "$REFERENCE_DIR" --native-dir "$NATIVE_OUTPUT_DIR" --vokra-root "$VOKRA_ROOT"
log "CPU enhancement parity complete; no upload performed; destroy the disposable VAST instance after evidence transfer"
