#!/usr/bin/env bash
# VAST/Linux-only XTTS-v2 checkpoint inspection.  This script never converts,
# executes, uploads, or publishes the Coqui pickle assets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
INSPECTOR="$PARITY_PROJECT/xtts_v2_inspect.py"
MODEL_REPOSITORY="coqui/XTTS-v2"
MODEL_REVISION="6c2b0d75eae4b7047358e3b6bd9325f857d43f77"
SOURCE_URL="https://github.com/coqui-ai/TTS.git"
SOURCE_REVISION="480a6cdf7dab508063c5d2e1b92fb7cd9f4f63c1"
MIN_VAST_MEM_KIB=$((128 * 1024 * 1024))
MIN_FREE_DISK_KIB=$((16 * 1024 * 1024))
MIN_TMPFS_KIB=$((4 * 1024 * 1024))
XTTS_UV_CACHE_DIR="${XTTS_UV_CACHE_DIR:-/tmp/vokra-xtts-uv-cache}"

log() { printf '[xtts-v2-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }

usage() {
  cat <<'EOF'
usage: run-xtts-v2-inspection.sh [--work-dir <empty-dir>]
       run-xtts-v2-inspection.sh --self-test

The model revision is fixed in this worker and cannot be overridden.
The result is always INSPECTION_ONLY. Pickle safe-load failures are blockers;
there is no custom-global allowlist or unsafe fallback.
EOF
}

self_test() {
  local path="${BASH_SOURCE[0]}" inspector="$PARITY_PROJECT/xtts_v2_inspect.py" fail=0 token
  for token in \
    'Linux' 'x86_64' '128' 'tmpfs' 'VOKRA_PUBLISH_ON_VAST=1' '/proc/meminfo' \
    'df -Pk' 'CARGO_BUILD_JOBS=1' 'cargo fmt --all -- --check' \
    'cargo metadata --no-deps --format-version 1' 'coqui/XTTS-v2' \
    'MODEL_REVISION="6c2b0d75eae4b7047358e3b6bd9325f857d43f77"' \
    'SOURCE_URL="https://github.com/coqui-ai/TTS.git"' \
    'SOURCE_REVISION="480a6cdf7dab508063c5d2e1b92fb7cd9f4f63c1"' \
    'xtts_v2_inspect.py' 'get_unsafe_globals_in_checkpoint' 'weights_only=True' \
    'torch.serialization' 'INSPECTION_ONLY' 'NO_UPLOAD' 'model_tree.json' \
    'server/local tree mismatch' 'blocker_exit=2' 'git status --porcelain' \
    'remote get-url origin' 'resolved_origin' 'status=BLOCKED' 'evidence_stage=INSPECTION_ONLY'; do
    if ! grep -Fq -- "$token" "$path"; then
      log "self-test FAIL: missing contract token: $token"
      fail=1
    fi
  done
  if grep -En 'weights_only=False|pickle\.load|torch\.load\([^)]*False' "$inspector" >/dev/null 2>&1; then
    log 'self-test FAIL: unsafe checkpoint loading found'
    fail=1
  fi
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|--push|--upload)([[:space:]]|$)' "$path" >/dev/null; then
    log 'self-test FAIL: publication command found'
    fail=1
  fi
  if grep -Eq '^XTTS_MODEL_REVISION=' "$path" || sed -n '/^MODEL_REVISION=/p' "$path" | grep -Fq "\${"; then
    log 'self-test FAIL: model revision can be operator-overridden'
    fail=1
  fi
  if grep -Eq '^SOURCE_REVISION_PREFIX=' "$path" || sed -n '/^SOURCE_REVISION=/p' "$path" | grep -Fq "\${"; then
    log 'self-test FAIL: source revision can be operator-overridden'
    fail=1
  fi
  if ! grep -Fq "[[ \"\$inspect_rc\" == 2 ]]" "$path"; then
    log 'self-test FAIL: worker does not require inspector exit 2'
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
  if ! UV_CACHE_DIR="$XTTS_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 \
    python "$inspector" --self-test >/dev/null; then
    log 'self-test FAIL: inspector self-test failed'
    fail=1
  fi
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

work_dir="/workspace/vokra-xtts-v2-inspection"
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
  [[ "$work_dir" == "/workspace/vokra-xtts-v2-inspection" ]] || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi

[[ "$(uname -s)" == Linux ]] || die 'XTTS checkpoint work is VAST/Linux-only'
[[ "$(uname -m)" == x86_64 ]] || die 'VAST host must be x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ "$MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]] || die 'worker model revision is invalid'
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
[[ -f "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" ]] || die 'locked parity project is missing'
[[ -f "$INSPECTOR" ]] || die 'XTTS inspector is missing'
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'VAST memory value is invalid'
(( mem_kib >= MIN_VAST_MEM_KIB )) || die '128 GiB memory guard failed'
tmpfs_kib="$(df -Pk /dev/shm | awk 'NR == 2 {print $4}')"
[[ "$tmpfs_kib" =~ ^[0-9]+$ ]] || die 'tmpfs value is invalid'
(( tmpfs_kib >= MIN_TMPFS_KIB )) || die 'tmpfs guard failed'
mkdir -p "$(dirname "$work_dir")"
[[ ! -e "$work_dir" || -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work-dir must be empty'
free_kib="$(df -Pk "$(dirname "$work_dir")" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || die 'VAST disk value is invalid'
(( free_kib >= MIN_FREE_DISK_KIB )) || die 'VAST disk guard failed'
for tool in cargo git uv awk find df; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done

mkdir -p "$work_dir/model" "$work_dir/source" "$work_dir/evidence"
work_dir="$(cd "$work_dir" && pwd)"
export CARGO_BUILD_JOBS=1
export UV_CACHE_DIR="$XTTS_UV_CACHE_DIR"
{
  echo "model_repository=$MODEL_REPOSITORY"
  echo "model_revision=$MODEL_REVISION"
  echo "source_url=$SOURCE_URL"
  echo "source_revision=$SOURCE_REVISION"
  echo 'runtime_status=NOT_IMPLEMENTED_FAIL_CLOSED'
  echo 'cpu_status=UNSUPPORTED'
  echo 'metal_status=BLOCKED_BY_CPU'
  echo 'parity_status=NOT_RUN'
  echo 'publication=NO_UPLOAD'
  cargo fmt --all -- --check
  cargo metadata --no-deps --format-version 1
} > "$work_dir/evidence/validation.log" 2>&1

# Capture the exact server tree and require the API to resolve the supplied
# commit before any large asset is downloaded.
# shellcheck disable=SC2129 # heredoc command output is intentionally one evidence stream
UV_CACHE_DIR="$XTTS_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 \
  python - "$work_dir/model_tree.json" "$MODEL_REPOSITORY" "$MODEL_REVISION" <<'PY' \
  >> "$work_dir/evidence/validation.log" 2>&1
import json
import sys
from pathlib import Path
from huggingface_hub import HfApi

repository, revision = sys.argv[2:4]
api = HfApi()
info = api.model_info(repository, revision=revision)
if info.sha != revision:
    raise RuntimeError(f"{repository} resolved {info.sha!r}, expected {revision!r}")
rows = []
for item in api.list_repo_tree(repository, revision=revision, recursive=True):
    rows.append({"path": getattr(item, "path", None), "type": getattr(item, "type", None), "size": getattr(item, "size", None), "oid": getattr(item, "oid", None)})
Path(sys.argv[1]).write_text(json.dumps({"repository": repository, "revision": revision, "files": rows}, indent=2, sort_keys=True) + "\n")
PY

UV_CACHE_DIR="$XTTS_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 \
  python - "$MODEL_REPOSITORY" "$MODEL_REVISION" "$work_dir/model" <<'PY' \
  >> "$work_dir/evidence/validation.log" 2>&1
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])
PY

git clone --filter=blob:none --no-checkout "$SOURCE_URL" "$work_dir/source/repo" >> "$work_dir/evidence/validation.log" 2>&1
git -C "$work_dir/source/repo" checkout --detach "$SOURCE_REVISION" >> "$work_dir/evidence/validation.log" 2>&1
source_revision="$(git -C "$work_dir/source/repo" rev-parse HEAD)"
[[ "$source_revision" == "$SOURCE_REVISION" ]] || die 'Coqui TTS source revision mismatch'
source_origin="$(git -C "$work_dir/source/repo" remote get-url origin)"
[[ "$source_origin" == "$SOURCE_URL" ]] || die 'Coqui TTS source origin mismatch'
echo "source_resolved_revision=$source_revision" >> "$work_dir/evidence/validation.log"
echo "source_resolved_origin=$source_origin" >> "$work_dir/evidence/validation.log"

set +e
UV_CACHE_DIR="$XTTS_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python \
  "$INSPECTOR" --model-dir "$work_dir/model" --source-dir "$work_dir/source/repo" \
  --evidence-dir "$work_dir/evidence" --model-tree "$work_dir/model_tree.json" \
  --revision "$MODEL_REVISION" >> "$work_dir/evidence/validation.log" 2>&1
inspect_rc=$?
set -e
[[ "$inspect_rc" == 2 ]] || die "unexpected inspector exit code: $inspect_rc"
manifest="$work_dir/evidence/xtts_v2_manifest.json"
[[ -s "$manifest" ]] || die 'inspection manifest is missing'
grep -Fq '"status": "BLOCKED"' "$manifest" || die 'blocked status missing'
grep -Fq '"evidence_stage": "INSPECTION_ONLY"' "$manifest" || die 'inspection stage missing'
grep -Fq "\"resolved_origin\": \"$SOURCE_URL\"" "$manifest" || die 'source origin missing or mismatched'
grep -Fq '"runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED"' "$manifest" || die 'runtime status missing'
grep -Fq '"cpu_status": "UNSUPPORTED"' "$manifest" || die 'CPU status missing'
grep -Fq '"metal_status": "BLOCKED_BY_CPU"' "$manifest" || die 'Metal status missing'
grep -Fq '"parity_status": "NOT_RUN"' "$manifest" || die 'parity status missing'
{
  echo 'runtime_status=NOT_IMPLEMENTED_FAIL_CLOSED'
  echo 'cpu_status=UNSUPPORTED'
  echo 'metal_status=BLOCKED_BY_CPU'
  echo 'parity_status=NOT_RUN'
  echo 'verdict=BLOCKED'
  echo 'blocker_exit=2'
  echo 'native_blocker=pickle/safe-load/license/source contract requires review; evidence preserved'
} | tee -a "$work_dir/evidence/validation.log"
die 'inspection evidence preserved; worker always exits 2 until runtime/parity/license gates are reviewed'
