#!/usr/bin/env bash
# VAST/Linux-only SGMSE-VoiceBank authentication wave. It downloads the
# exact HF snapshot and pinned upstream implementation, records safe-load and
# tensor evidence, and never emits a safetensors/GGUF or uploads anything.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
PREPARER="$VOKRA_ROOT/tools/parity/sgmse_prepare_checkpoint.py"
MODEL_REPOSITORY="speechbrain/sgmse-voicebank"
MODEL_REVISION="8f4ff7b65284c49492a43349b8106e094ac0d365"
SOURCE_URL="https://github.com/sp-uhh/sgmse.git"
SOURCE_REVISION="1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e"
SPEECHBRAIN_URL="https://github.com/speechbrain/speechbrain.git"
SPEECHBRAIN_REVISION="31c1e329048c0380dc7f2acbe680c44a036b6286"
SPEECHBRAIN_VERSION="1.0.3"
SPEECHBRAIN_SDIST_SHA256="fcab3c6e90012cecb1eed40ea235733b550137e73da6bfa2340ba191ec714052"
SPEECHBRAIN_WHEEL_SHA256="9859d4c1b1fb3af3b85523c0c89f52e45a04f305622ed55f31aa32dd2fba19e9"
HF_ROOT="https://huggingface.co/$MODEL_REPOSITORY/resolve/$MODEL_REVISION"
EXPECTED_SIZE=262593305
EXPECTED_SHA256="7ca96321aca40cdca90c450d1450a5c7f343935e5b46ee34a1b575f9f774ccc3"
COMPANIONS=(README.md .gitattributes example.wav)
MIN_VAST_MEM_KIB=$((8 * 1024 * 1024))
MIN_FREE_DISK_KIB=$((4 * 1024 * 1024))
SGMSE_UV_CACHE_DIR="${SGMSE_UV_CACHE_DIR:-/tmp/vokra-sgmse-uv-cache}"

log() { printf '[sgmse-voicebank-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF'
usage: run-sgmse-voicebank-inspection.sh [--work-dir <empty-dir>]
       run-sgmse-voicebank-inspection.sh --self-test

VAST/Linux-only inspection of the fixed SpeechBrain HF snapshot and pinned
SGMSE source. Checkpoint loading uses weights_only=True only. The result is
INSPECTION_ONLY; no native forward, parity PASS, conversion, or upload occurs.
EOF
}

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token
  for token in 'VOKRA_PUBLISH_ON_VAST=1' 'Linux' 'x86_64' 'CARGO_BUILD_JOBS=1' \
    'cargo fmt --all -- --check' 'cargo metadata --no-deps --format-version 1' \
    "$MODEL_REPOSITORY" "$MODEL_REVISION" "$SOURCE_REVISION" "$SPEECHBRAIN_URL" \
    "$SPEECHBRAIN_REVISION" "$SPEECHBRAIN_VERSION" "$SPEECHBRAIN_SDIST_SHA256" \
    "$SPEECHBRAIN_WHEEL_SHA256" "$EXPECTED_SHA256" \
    '262593305' 'weights_only=True' 'unsafe pickle fallback' 'INSPECTION_ONLY' \
    'NO_UPLOAD' 'sgmse_prepare_checkpoint.py --self-test' '"blockers": []' \
    'README.md' '.gitattributes' 'example.wav' \
    'sgmse_plus.py' 'speechbrain/inference/enhancement.py' 'class SGMSEEnhancement' 'files_by_role' \
    'importlib.metadata' \
    'verdict=BLOCKED' 'blocker_exit=2' 'git status --porcelain'; do
    if ! grep -Fq -- "$token" "$path"; then
      log "self-test FAIL: missing contract token: $token"
      fail=1
    fi
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|--push|--upload)([[:space:]]|$)' "$path" >/dev/null; then
    log 'self-test FAIL: publication command found'
    fail=1
  fi
  if "$path" --self-test --work-dir /tmp/not-accepted >/dev/null 2>&1; then
    log 'self-test FAIL: extra argument accepted'
    fail=1
  fi
  if "$path" --unknown-flag >/dev/null 2>&1; then
    log 'self-test FAIL: unknown argument accepted'
    fail=1
  fi
  if ! UV_CACHE_DIR="$SGMSE_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" \
    --python 3.12 python "$PREPARER" --self-test >/dev/null; then
    log 'self-test FAIL: preparer self-test failed'
    fail=1
  fi
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

work_dir="/workspace/vokra-sgmse-voicebank-inspection"
self=0
while (($#)); do
  case "$1" in
    --self-test) self=1; shift ;;
    --work-dir) (($# >= 2)) || die '--work-dir requires a path'; work_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  [[ "$work_dir" == "/workspace/vokra-sgmse-voicebank-inspection" ]] || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi

[[ "$(uname -s)" == Linux ]] || die 'SGMSE checkpoint work is VAST/Linux-only'
[[ "$(uname -m)" == x86_64 ]] || die 'VAST host must be x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
[[ -f "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" ]] || die 'locked parity project is missing'
[[ -f "$PREPARER" ]] || die 'SGMSE preparer is missing'
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'VAST memory value is invalid'
(( mem_kib >= MIN_VAST_MEM_KIB )) || die 'VAST memory guard failed'
mkdir -p "$(dirname "$work_dir")"
[[ ! -e "$work_dir" || -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work-dir must be empty'
free_kib="$(df -Pk "$(dirname "$work_dir")" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || die 'VAST disk value is invalid'
(( free_kib >= MIN_FREE_DISK_KIB )) || die 'VAST disk guard failed'
for tool in cargo git uv curl awk find df; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done

mkdir -p "$work_dir/source" "$work_dir/hf" "$work_dir/evidence"
work_dir="$(cd "$work_dir" && pwd)"
export CARGO_BUILD_JOBS=1
export UV_CACHE_DIR="$SGMSE_UV_CACHE_DIR"
locked_speechbrain_version="$(UV_CACHE_DIR="$SGMSE_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python -c 'import importlib.metadata as m; print(m.version("speechbrain"))')"
[[ "$locked_speechbrain_version" == "$SPEECHBRAIN_VERSION" ]] || die "locked SpeechBrain package is $locked_speechbrain_version, expected $SPEECHBRAIN_VERSION"
{
  echo "model_repository=$MODEL_REPOSITORY"
  echo "model_revision=$MODEL_REVISION"
  echo "source_url=$SOURCE_URL"
  echo "source_revision=$SOURCE_REVISION"
  echo "speechbrain_url=$SPEECHBRAIN_URL"
  echo "speechbrain_revision=$SPEECHBRAIN_REVISION"
  echo "speechbrain_version=$SPEECHBRAIN_VERSION"
  echo "expected_checkpoint_size=$EXPECTED_SIZE"
  echo "expected_checkpoint_sha256=$EXPECTED_SHA256"
  echo "locked_speechbrain_version=$locked_speechbrain_version"
  echo 'runtime_status=INSPECTION_ONLY'
  echo 'parity_status=INSPECTION_ONLY'
  echo 'publication=NO_UPLOAD'
  cargo fmt --all -- --check
  cargo metadata --no-deps --format-version 1
} > "$work_dir/evidence/validation.log" 2>&1

git clone --filter=blob:none --no-checkout "$SOURCE_URL" "$work_dir/source/repo" >> "$work_dir/evidence/validation.log" 2>&1
git -C "$work_dir/source/repo" checkout --detach "$SOURCE_REVISION" >> "$work_dir/evidence/validation.log" 2>&1
[[ "$(git -C "$work_dir/source/repo" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || die 'pinned SGMSE source revision mismatch'
git clone --filter=blob:none --no-checkout "$SPEECHBRAIN_URL" "$work_dir/source/speechbrain" >> "$work_dir/evidence/validation.log" 2>&1
git -C "$work_dir/source/speechbrain" checkout --detach "$SPEECHBRAIN_REVISION" >> "$work_dir/evidence/validation.log" 2>&1
[[ "$(git -C "$work_dir/source/speechbrain" rev-parse HEAD)" == "$SPEECHBRAIN_REVISION" ]] || die 'pinned SpeechBrain source revision mismatch'

curl --fail --location --retry 3 --silent --show-error \
  "$HF_ROOT/hyperparams.yaml?download=true" --output "$work_dir/hf/hyperparams.yaml"
curl --fail --location --retry 3 --silent --show-error \
  "$HF_ROOT/$CHECKPOINT_NAME?download=true" --output "$work_dir/hf/$CHECKPOINT_NAME"
for companion in "${COMPANIONS[@]}"; do
  curl --fail --location --retry 3 --silent --show-error \
    "$HF_ROOT/$companion?download=true" --output "$work_dir/hf/$companion"
done

set +e
UV_CACHE_DIR="$SGMSE_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python \
  "$PREPARER" --ckpt "$work_dir/hf/$CHECKPOINT_NAME" \
  --hyperparams "$work_dir/hf/hyperparams.yaml" --companion-dir "$work_dir/hf" \
  --algorithm-source-dir "$work_dir/source/repo" \
  --speechbrain-source-dir "$work_dir/source/speechbrain" \
  --manifest "$work_dir/evidence/sgmse_voicebank_manifest.json" \
  >> "$work_dir/evidence/validation.log" 2>&1
inspect_rc=$?
set -e
[[ "$inspect_rc" == 0 || "$inspect_rc" == 2 ]] || die "unexpected inspector exit code: $inspect_rc"
[[ -s "$work_dir/evidence/sgmse_voicebank_manifest.json" ]] || die 'inspection manifest is missing'
grep -Fq '"runtime_status": "INSPECTION_ONLY"' "$work_dir/evidence/sgmse_voicebank_manifest.json" || die 'inspection status missing'
if ! grep -Fq '"blockers": []' "$work_dir/evidence/sgmse_voicebank_manifest.json"; then
  {
    echo 'runtime_status=INSPECTION_ONLY'
    echo 'parity_status=INSPECTION_ONLY'
    echo 'verdict=BLOCKED'
    echo 'blocker_exit=2'
    echo 'native_blocker=see sgmse_voicebank_manifest.json blockers and safe-load evidence'
  } | tee -a "$work_dir/evidence/validation.log"
  die 'inspection blockers remain; evidence was preserved and worker exits 2'
fi
{
  echo 'runtime_status=INSPECTION_ONLY'
  echo 'parity_status=INSPECTION_ONLY'
  echo 'verdict=INSPECTION_COMPLETE_NO_UPLOAD'
  echo 'native_blocker=native NCSN++/STFT/OUVE route and parity remain unaudited'
} | tee -a "$work_dir/evidence/validation.log"
log "inspection complete: evidence=$work_dir; no conversion or upload performed"
