#!/usr/bin/env bash
# VAST/Linux-only CLAP checkpoint and independent-reference inspection.
# This worker is deliberately non-publishing. The native CLAP binder remains
# disabled until its exact tensor manifest and topology audit are reviewed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
REFERENCE_DUMPER="tools/parity/clap_dump_reference.py"
DEDICATED_PROJECT="$VOKRA_ROOT/tools/parity/clap_htsat_fused_reference"
UPSTREAM_REPO="laion/clap-htsat-fused"
UPSTREAM_REVISION="365dea6ef167def6676140ed93bbc43f84dabb28"
MIN_VAST_MEM_KIB=$((64 * 1024 * 1024))
MIN_FREE_DISK_KIB=$((150 * 1024 * 1024))
CLAP_UV_CACHE_DIR="${CLAP_UV_CACHE_DIR:-/tmp/vokra-clap-uv-cache}"

log() { printf '[clap-htsat-fused-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }
CLAP_SELF_TEST_TMP=""
# shellcheck disable=SC2329 # Invoked by the EXIT trap below.
cleanup_self_test() {
  [[ -n "$CLAP_SELF_TEST_TMP" ]] && rm -rf -- "$CLAP_SELF_TEST_TMP"
}

require_preflight() {
  local project="${1:-$DEDICATED_PROJECT}" approval="${2:-}"
  local gate="$project/license_gate.py" manifest="$project/license_gate_manifest.json"
  [[ -d "$project" && ! -L "$project" ]] || { log 'dedicated CLAP reference project is missing; identity/license gate is unresolved'; return 2; }
  [[ -f "$project/pyproject.toml" && ! -L "$project/pyproject.toml" ]] || { log 'dedicated CLAP pyproject.toml is missing'; return 2; }
  [[ -f "$project/uv.lock" && ! -L "$project/uv.lock" ]] || { log 'dedicated CLAP uv.lock is missing; refuse before acquisition'; return 2; }
  [[ -f "$gate" && ! -L "$gate" && -f "$manifest" && ! -L "$manifest" ]] || { log 'CLAP license gate/manifest is missing; refuse before acquisition'; return 2; }
  [[ -f "$approval" && ! -L "$approval" && -s "$approval" ]] || { log 'approval evidence must be a nonempty regular file'; return 2; }
}

validate_absent_work() {
  local work="$1" component rest current parent candidate item
  local -a suffix=()
  [[ "$work" == /* && "$work" != *$'\n'* && "$work" != *$'\r'* ]] || return 2
  [[ "$work" != */../* && "$work" != */.. && "$work" != *'/./'* && "$work" != *'/.' ]] || return 2
  rest="${work#/}"
  current="/"
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then component="${rest%%/*}"; rest="${rest#*/}"; else component="$rest"; rest=""; fi
    [[ -n "$component" ]] || continue
    current="$current$component"
    [[ ! -L "$current" ]] || return 2
    current="$current/"
  done
  [[ ! -e "$work" && ! -L "$work" ]] || return 2
  parent="$work"
  while [[ ! -e "$parent" ]]; do
    [[ ! -L "$parent" ]] || return 2
    item="${parent##*/}"
    [[ -n "$item" ]] || return 2
    suffix+=("$item")
    [[ "$parent" != / ]] || return 2
    parent="${parent%/*}"
    [[ -n "$parent" ]] || parent=/
  done
  [[ -d "$parent" && ! -L "$parent" ]] || return 2
  candidate="$(cd -P "$parent" && pwd)"
  for (( item = ${#suffix[@]} - 1; item >= 0; item-- )); do candidate="$candidate/${suffix[item]}"; done
  local root_real project_parent project_real
  root_real="$(cd -P "$VOKRA_ROOT" 2>/dev/null && pwd)" || return 2
  [[ "$candidate" != "$root_real" && "$candidate/" != "$root_real/"* && "$root_real/" != "$candidate/"* ]] || return 2
  if [[ -d "$DEDICATED_PROJECT" ]]; then
    project_real="$(cd -P "$DEDICATED_PROJECT" 2>/dev/null && pwd)" || return 2
    [[ "$candidate" != "$project_real" && "$candidate/" != "$project_real/"* && "$project_real/" != "$candidate/"* ]] || return 2
  else
    project_parent="$(cd -P "$(dirname "$DEDICATED_PROJECT")" 2>/dev/null && pwd)" || return 2
    [[ "$candidate" != "$project_parent" && "$candidate/" != "$project_parent/"* ]] || return 2
  fi
}

usage() {
  cat <<'EOF'
usage: run-clap-htsat-fused-validation.sh --approval-evidence <file> [--work-dir <absent-dir>]
       run-clap-htsat-fused-validation.sh --self-test

The normal path is Linux/VAST-only. It resolves the exact Hugging Face
revision with uv-managed Transformers, writes an independent official audio
embedding and complete state-dict name/shape manifest, and records hashes.
The result is INSPECTION_ONLY: no native runtime parity or upload occurs.
EOF
}

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token tmp fake_project approval work rc
  for token in \
    'VOKRA_PUBLISH_ON_VAST=1' 'uname -s' 'uname -m' 'MIN_VAST_MEM_KIB' \
    '/proc/meminfo' 'df -Pk' 'CARGO_BUILD_JOBS=1' 'cargo fmt --all -- --check' \
    'cargo metadata --no-deps --format-version 1' 'snapshot_download' \
    "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$REFERENCE_DUMPER" \
    'transformers_clap_model_source_sha256' 'tensor_manifest' \
    'INSPECTION_ONLY' 'no upload' 'VOKRA_CLAP_REAL_GGUF' 'GGUFReader' \
    'clap_dump_reference.py" --self-test' \
    'git status --porcelain'; do
    if ! grep -Fq -- "$token" "$path"; then
      log "self-test FAIL: missing contract token: $token"
      fail=1
    fi
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*publish-one\.sh|.*upload\.sh)([[:space:]]|$)' "$path" >/dev/null; then
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
  tmp="$(mktemp -d)"
  tmp="$(cd -P "$tmp" && pwd)"
  CLAP_SELF_TEST_TMP="$tmp"
  trap cleanup_self_test EXIT
  fake_project="$tmp/project"
  mkdir -p "$fake_project"
  printf '[project]\nname = "synthetic-clap"\nversion = "0.0.0"\n' >"$fake_project/pyproject.toml"
  if require_preflight "$fake_project" "$tmp/approval.json" >/dev/null 2>&1; then
    log 'self-test FAIL: missing dedicated lock/gate was accepted'
    fail=1
  fi
  approval="$tmp/approval.json"
  work="$tmp/work/nested"
  printf '{}\n' >"$approval"
  validate_absent_work "$work" || { log 'self-test FAIL: safe absent work path rejected'; fail=1; }
  if "$path" --self-test --self-test >/dev/null 2>&1; then
    log 'self-test FAIL: duplicate --self-test accepted'; fail=1
  fi
  set +e
  VOKRA_PUBLISH_ON_VAST=1 CLAP_UV_CACHE_DIR="$tmp/cache" \
    "$path" --approval-evidence "$approval" --work-dir "$work" >/dev/null 2>&1
  rc=$?
  set -e
  if [[ "$rc" != 2 || -e "$work" || -e "$tmp/cache" ]]; then
    log 'self-test FAIL: production-shaped missing-lock probe had effects or wrong status'
    fail=1
  fi
  if ! UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python \
    "$VOKRA_ROOT/$REFERENCE_DUMPER" --self-test >/dev/null; then
    log 'self-test FAIL: independent dumper self-test failed'
    fail=1
  fi
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

work_dir="/workspace/vokra-clap-htsat-fused-validation"
approval_evidence=''
self=0
seen_self=0
seen_work=0
seen_approval=0
while (($#)); do
  case "$1" in
    --self-test) (( seen_self == 0 )) || die 'duplicate --self-test'; seen_self=1; self=1; shift ;;
    --work-dir) (( seen_work == 0 )) || die 'duplicate --work-dir'; (( $# >= 2 )) || die '--work-dir requires a path'; [[ -n "$2" && "$2" != -* ]] || die '--work-dir must be a nonempty path'; seen_work=1; work_dir="$2"; shift 2 ;;
    --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; (( $# >= 2 )) || die '--approval-evidence requires a file'; [[ -n "$2" && "$2" != -* ]] || die '--approval-evidence must be a nonempty file path'; seen_approval=1; approval_evidence="$2"; shift 2 ;;
    -h|--help) [[ $self == 0 && $# == 1 ]] || die '--help cannot be combined with other arguments'; usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  [[ "$seen_work" == 0 && "$seen_approval" == 0 ]] || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi

[[ -n "$approval_evidence" ]] || die '--approval-evidence is required'
require_preflight "$DEDICATED_PROJECT" "$approval_evidence" || die 'CLAP dedicated lock/license/approval gate is unresolved; refuse before host/work/network'

[[ "$(uname -s)" == Linux ]] || die 'inspection is Linux/VAST-only'
[[ "$(uname -m)" == x86_64 ]] || die 'VAST host must be x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
[[ -f "$VOKRA_ROOT/$REFERENCE_DUMPER" ]] || die 'reference dumper is missing'

mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'VAST memory value is invalid'
(( mem_kib >= MIN_VAST_MEM_KIB )) || die 'VAST memory guard failed'
validate_absent_work "$work_dir" || die 'work-dir must be absent, disjoint, and free of symlink ancestors'
free_kib="$(df -Pk "$(dirname "$work_dir")" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || die 'VAST disk value is invalid'
(( free_kib >= MIN_FREE_DISK_KIB )) || die 'VAST disk guard failed'
for tool in cargo git uv sha256sum awk find df; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done

mkdir -p "$work_dir"
work_dir="$(cd "$work_dir" && pwd)"
export CARGO_BUILD_JOBS=1
printf 'repository=%s\nrevision=%s\n' "$UPSTREAM_REPO" "$UPSTREAM_REVISION" > "$work_dir/validation.log"
cargo fmt --all -- --check >> "$work_dir/validation.log" 2>&1
cargo metadata --no-deps --format-version 1 >> "$work_dir/validation.log" 2>&1

snapshot_path_file="$work_dir/snapshot-path.txt"
UV_CACHE_DIR="$CLAP_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python - \
  "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$work_dir/hf-cache" "$snapshot_path_file" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo, revision, cache_dir, output = sys.argv[1:]
path = Path(snapshot_download(
    repo_id=repo,
    revision=revision,
    cache_dir=cache_dir,
    allow_patterns=["*.json", "*.txt", "*.safetensors", "*.safetensors.index.json"],
))
if path.name != revision:
    raise SystemExit(f"snapshot revision drift: {path.name!r} != {revision!r}")
for required in ("config.json", "preprocessor_config.json"):
    if not (path / required).is_file():
        raise SystemExit(f"missing pinned upstream file: {required}")
Path(output).write_text(str(path) + "\n", encoding="utf-8")
PY

snapshot_dir="$(< "$snapshot_path_file")"
UV_CACHE_DIR="$CLAP_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python \
  "$VOKRA_ROOT/$REFERENCE_DUMPER" --model-dir "$snapshot_dir" \
  --output-dir "$work_dir/reference"
find "$snapshot_dir" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > "$work_dir/upstream-file-sha256.txt"
sha256sum "$work_dir/reference"/* > "$work_dir/reference-sha256.txt"
if [[ -n "${VOKRA_CLAP_REAL_GGUF:-}" ]]; then
  [[ -s "$VOKRA_CLAP_REAL_GGUF" ]] || die "VOKRA_CLAP_REAL_GGUF is missing or empty"
  UV_CACHE_DIR="$CLAP_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python - \
    "$VOKRA_CLAP_REAL_GGUF" "$work_dir/supplied-gguf-manifest.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from gguf import GGUFReader

path, output = sys.argv[1:]
reader = GGUFReader(path)
manifest = {
    "file": str(Path(path).resolve()),
    "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    "identity_status": "UNAUTHENTICATED_SUPPLIED_FILE",
    "public_artifact_status": "NOT_ASSERTED",
    "tensor_manifest": {
        tensor.name: {
            "shape": [int(axis) for axis in tensor.shape],
            "dtype": str(tensor.tensor_type),
        }
        for tensor in sorted(reader.tensors, key=lambda item: item.name)
    },
    "parity_status": "INSPECTION_ONLY",
}
Path(output).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  echo "supplied_gguf_sha256=$(sha256sum "$VOKRA_CLAP_REAL_GGUF" | awk '{print $1}')" | tee -a "$work_dir/validation.log"
  echo 'supplied_gguf_identity=UNAUTHENTICATED_SUPPLIED_FILE' | tee -a "$work_dir/validation.log"
else
  echo 'supplied_gguf=NOT_SUPPLIED' | tee -a "$work_dir/validation.log"
fi
{
  echo 'runtime_status=INSPECTION_ONLY'
  echo 'parity_status=INSPECTION_ONLY'
  echo 'verdict=NO_UPLOAD'
  echo "upstream_repo=$UPSTREAM_REPO"
  echo "upstream_revision=$UPSTREAM_REVISION"
} | tee -a "$work_dir/validation.log"
log "inspection complete: evidence remains at $work_dir; no upload or publication was performed"
