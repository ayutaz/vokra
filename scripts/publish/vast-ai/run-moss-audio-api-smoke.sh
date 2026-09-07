#!/usr/bin/env bash
# VAST-only model-free MOSS-Audio official Transformers API smoke.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PROJECT="$VOKRA_ROOT/tools/parity/moss_audio/api_smoke"
SMOKE="$PROJECT/api_smoke.py"
SOURCE_REPO="https://github.com/OpenMOSS/MOSS-Audio.git"
SOURCE_REVISION="5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883"
MIN_VAST_MEM_KIB=12000000

log() { printf '[moss-audio-api-vast] %s\n' "$*" >&2; }
step() { printf '\n[moss-audio-api-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-moss-audio-api-smoke.sh --variant <4b|8b|all> \
  --approval-evidence <file> --expected-head <40-hex> [--work-dir <absent-dir>]
       run-moss-audio-api-smoke.sh --model-free --variant <4b|8b|all> \
  --expected-head <40-hex> [--work-dir <absent-dir>]
       run-moss-audio-api-smoke.sh --closure-only --variant <4b|8b|all> \
  --approval-evidence <file> --expected-head <40-hex>
       run-moss-audio-api-smoke.sh --self-test

This VAST/Linux worker authenticates only the pinned OpenMOSS source files and
non-weight model metadata, then imports the official config/model/processor
classes under Transformers 5.10.4 without loading a checkpoint. It has no
upload or model-weight path. --closure-only performs only local HEAD,
project/lock, and approval closure checks.

The --model-free phase skips owner approval, stages only metadata, imports the
official classes, and records pending source/model/operator approvals. It never
loads a checkpoint, uploads, or claims parity.
EOF
}

require_head() {
  local expected="$1" actual
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head must be lowercase 40-hex'
  [[ -d "$VOKRA_ROOT/.git" ]] || die 'Vokra checkout is missing'
  actual="$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
  [[ "$actual" == "$expected" ]] || die "HEAD mismatch: $actual != $expected"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'Vokra checkout is dirty'
}

require_host() {
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is required'
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'API smoke requires VAST Linux x86_64'
  local memory
  memory="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$memory" =~ ^[0-9]+$ && "$memory" -ge "$MIN_VAST_MEM_KIB" ]] || die 'VAST memory guard failed'
}

require_inputs() {
  for tool in uv git awk find tee sha256sum; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -f "$PROJECT/pyproject.toml" && ! -L "$PROJECT/pyproject.toml" ]] || die 'API smoke pyproject is missing'
  [[ -f "$PROJECT/uv.lock" && ! -L "$PROJECT/uv.lock" ]] || die 'API smoke uv.lock is missing'
  [[ -f "$SMOKE" && ! -L "$SMOKE" ]] || die 'API smoke implementation is missing'
}

require_absent_work_dir() {
  local target="$1" probe canonical protected other
  [[ "$target" == /* ]] || die '--work-dir must be absolute'
  [[ "$target" != *'/../'* && "$target" != ../* && "$target" != '..' ]] || die '--work-dir must not contain parent traversal'
  probe="$target"
  while [[ "$probe" != / ]]; do
    [[ ! -L "$probe" ]] || die "--work-dir has a symlinked ancestor: $probe"
    probe="$(dirname "$probe")"
  done
  [[ ! -e "$target" && ! -L "$target" ]] || die '--work-dir must be absent'
  canonical="$(canonicalize_uncreated "$target")" || die '--work-dir cannot be canonicalized'
  for protected in "$VOKRA_ROOT" "$PROJECT" "$SMOKE" "$PROJECT/uv.lock" "$PROJECT/pyproject.toml"; do
    [[ -e "$protected" || -L "$protected" ]] || continue
    [[ ! -L "$protected" ]] || die "protected path is symlinked: $protected"
    other="$(canonicalize_uncreated "$protected")" || die "protected path cannot be canonicalized: $protected"
    paths_overlap "$canonical" "$other" && die "--work-dir overlaps protected path: $protected"
  done
}

canonicalize_uncreated() {
  local path="$1" suffix='' name parent
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"
    [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="${path%/*}"
    [[ "$parent" == "$path" ]] && parent='/'
    path="$parent"
    [[ ! -L "$path" ]] || return 1
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}

paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }

run_self_test() {
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$SMOKE" --self-test
  local script_path="${BASH_SOURCE[0]}" repo_overlap project_overlap
  for bad in \
    '--self-test --self-test' \
    '--closure-only --self-test' \
    '--variant 4b --variant 8b' \
    '--expected-head 0000000000000000000000000000000000000000 --expected-head 0000000000000000000000000000000000000000'; do
    # shellcheck disable=SC2086
    if bash "$script_path" $bad >/dev/null 2>&1; then die "accepted malformed args: $bad"; fi
  done
  repo_overlap="$VOKRA_ROOT/.moss-audio-api-overlap-$$"
  if require_absent_work_dir "$repo_overlap" >/dev/null 2>&1; then die 'self-test accepted work directory inside Vokra checkout'; fi
  project_overlap="$PROJECT/.moss-audio-api-overlap-$$"
  if require_absent_work_dir "$project_overlap" >/dev/null 2>&1; then die 'self-test accepted work directory inside API smoke project'; fi
  log 'self-test PASS'
}

run_model_free() {
  local selection="$1" expected_head="$2" work_dir="$3"
  local selected source_dir snapshot_root evidence smoke_status evidence_sha
  [[ "$selection" == 4b || "$selection" == 8b || "$selection" == all ]] || die '--variant must be 4b, 8b, or all'
  require_head "$expected_head"
  require_inputs
  require_host
  [[ -n "$work_dir" ]] || work_dir="${VOKRA_SCRATCH:-$HOME/scratchpad}/moss-audio-model-free-api-smoke-${expected_head:0:12}"
  require_absent_work_dir "$work_dir"
  mkdir -p "$work_dir"
  selected="$selection"
  source_dir="$work_dir/official-source"
  snapshot_root="$work_dir/metadata"
  evidence="$work_dir/model-free-api-smoke-evidence.json"
  step 'Install the reviewed frozen model-free API smoke environment'
  UV_NO_CACHE=1 uv sync --project "$PROJECT" --frozen --python 3.12
  step 'Checkout the exact official source without model files'
  git clone --filter=blob:none --no-checkout "$SOURCE_REPO" "$source_dir"
  git -C "$source_dir" checkout --detach "$SOURCE_REVISION"
  mkdir -p "$snapshot_root"
  for variant in ${selected/all/4b 8b}; do
    local repo revision
    if [[ "$variant" == 4b ]]; then repo='OpenMOSS-Team/MOSS-Audio-4B-Instruct'; revision='6907a499dc0e87cc77c8ae0fe23fd0eb5476a02d'; else repo='OpenMOSS-Team/MOSS-Audio-8B-Instruct'; revision='6521a39181b47a18f2d9f4b3acfb5bca7b76b57f'; fi
    step "Acquire $variant metadata only"
    download_metadata "$repo" "$revision" "$snapshot_root/$variant"
  done
  step 'Run official model-free API smoke'
  set +e
  UV_NO_CACHE=1 uv run --no-cache --project "$PROJECT" --frozen --python 3.12 python "$SMOKE" \
    --model-free --vokra-root "$VOKRA_ROOT" --project "$PROJECT" --source-dir "$source_dir" \
    --snapshot-root "$snapshot_root" --expected-head "$expected_head" --variant "$selected" --output "$evidence"
  smoke_status=$?
  set -e
  evidence_sha='UNAVAILABLE'
  if [[ -f "$evidence" && ! -L "$evidence" ]]; then evidence_sha="$(sha256sum "$evidence" | awk '{print $1}')"; fi
  cat > "$work_dir/model-free-summary.txt" <<EOF
format=vokra-moss-audio-model-free-api-summary-v1
status=$([[ "$smoke_status" == 0 ]] && echo PASS_MODEL_FREE || echo BLOCKED_OR_FAILED)
exit_status=$smoke_status
expected_head=$expected_head
evidence_sha256=$evidence_sha
checkpoint_load=NOT_PERFORMED
publication=NO_UPLOAD
approvals=PENDING_OWNER_APPROVAL
EOF
  [[ "$smoke_status" == 0 ]] || { die "model-free API smoke did not pass (exit=$smoke_status); evidence=$evidence; destroy the VAST instance"; return 2; }
  log "PASS_MODEL_FREE: no checkpoint load and no upload; evidence=$evidence (sha256=$evidence_sha); destroy the VAST instance"
}

download_metadata() {
  local repo="$1" revision="$2" output="$3"
  mkdir -p "$output"
  UV_NO_CACHE=1 uv run --no-cache --project "$PROJECT" --frozen --python 3.12 python -c \
    'import os,sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], allow_patterns=["config.json", "tokenizer_config.json", "processor_config.json", "vocab.json", "merges.txt", "chat_template.jinja", "generation_config.json"], token=os.environ.get("HF_TOKEN") or os.environ.get("HF"))' \
    "$repo" "$revision" "$output"
}

main() {
  local selection='' approval='' expected_head='' work_dir='' self_test=0 closure_only=0 model_free=0
  local seen_variant=0 seen_approval=0 seen_head=0 seen_work=0 seen_self=0 seen_closure=0 seen_model_free=0
  while (( $# > 0 )); do
    case "$1" in
      --variant) (( seen_variant == 0 )) || die 'duplicate --variant'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--variant requires a value'; selection="$2"; seen_variant=1; shift 2 ;;
      --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a value'; approval="$2"; seen_approval=1; shift 2 ;;
      --expected-head) (( seen_head == 0 )) || die 'duplicate --expected-head'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--expected-head requires a value'; expected_head="$2"; seen_head=1; shift 2 ;;
      --work-dir) (( seen_work == 0 )) || die 'duplicate --work-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--work-dir requires a value'; work_dir="$2"; seen_work=1; shift 2 ;;
      --self-test) (( seen_self == 0 )) || die 'duplicate --self-test'; self_test=1; seen_self=1; shift ;;
      --closure-only) (( seen_closure == 0 )) || die 'duplicate --closure-only'; closure_only=1; seen_closure=1; shift ;;
      --model-free) (( seen_model_free == 0 )) || die 'duplicate --model-free'; model_free=1; seen_model_free=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) usage; die "unknown argument: $1" ;;
    esac
  done
  if (( self_test == 1 )); then
    [[ -z "$selection$approval$expected_head$work_dir" && "$closure_only" == 0 && "$model_free" == 0 ]] || die '--self-test accepts no other options'
    run_self_test
    return
  fi
  if (( model_free == 1 )); then
    [[ "$seen_approval" == 0 && "$seen_closure" == 0 ]] || die '--model-free does not accept approval evidence or --closure-only'
    [[ "$seen_variant" == 1 && "$seen_head" == 1 ]] || die '--model-free requires --variant and --expected-head'
    run_model_free "$selection" "$expected_head" "$work_dir"
    return
  fi
  [[ "$selection" == 4b || "$selection" == 8b || "$selection" == all ]] || die '--variant must be 4b, 8b, or all'
  [[ -n "$approval" && -f "$approval" && ! -L "$approval" && -s "$approval" ]] || die '--approval-evidence must be a nonempty regular file'
  [[ -n "$expected_head" ]] || die '--expected-head is required'
  require_head "$expected_head"
  require_inputs
  local selected="$selection"
  if [[ "$selection" == all ]]; then selected='4b,8b'; fi
  # This is intentionally stdlib-only: reject an unapproved dependency
  # closure before uv is allowed to resolve/install the smoke environment.
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$SMOKE" \
    --closure-only --vokra-root "$VOKRA_ROOT" --project "$PROJECT" \
    --approval-evidence "$approval" --expected-head "$expected_head" --variant "$selection"
  if (( closure_only == 1 )); then return; fi
  require_host
  [[ -n "$work_dir" ]] || work_dir="${VOKRA_SCRATCH:-$HOME/scratchpad}/moss-audio-api-smoke-${expected_head:0:12}"
  require_absent_work_dir "$work_dir"
  mkdir -p "$work_dir"
  local source_dir="$work_dir/official-source" snapshot_root="$work_dir/metadata" evidence="$work_dir/api-smoke-evidence.json"
  step 'Checkout the exact official source without model files'
  git clone --filter=blob:none --no-checkout "$SOURCE_REPO" "$source_dir"
  git -C "$source_dir" checkout --detach "$SOURCE_REVISION"
  mkdir -p "$snapshot_root"
  for variant in ${selected//,/ }; do
    local repo revision
    if [[ "$variant" == 4b ]]; then repo='OpenMOSS-Team/MOSS-Audio-4B-Instruct'; revision='6907a499dc0e87cc77c8ae0fe23fd0eb5476a02d'; else repo='OpenMOSS-Team/MOSS-Audio-8B-Instruct'; revision='6521a39181b47a18f2d9f4b3acfb5bca7b76b57f'; fi
    step "Acquire $variant metadata only"
    download_metadata "$repo" "$revision" "$snapshot_root/$variant"
  done
  step 'Run the official model-free API smoke'
  set +e
  UV_NO_CACHE=1 uv run --no-cache --project "$PROJECT" --frozen --python 3.12 python "$SMOKE" \
    --vokra-root "$VOKRA_ROOT" --project "$PROJECT" --source-dir "$source_dir" \
    --snapshot-root "$snapshot_root" --approval-evidence "$approval" \
    --expected-head "$expected_head" --variant "$selection" --output "$evidence"
  local smoke_status=$?
  set -e
  local evidence_sha='UNAVAILABLE'
  if [[ -f "$evidence" && ! -L "$evidence" ]]; then
    evidence_sha="$(sha256sum "$evidence" | awk '{print $1}')"
  fi
  local verdict='BLOCKED_OR_FAILED'
  if (( smoke_status == 0 )); then verdict='PASS'; fi
  cat > "$work_dir/api-smoke-summary.txt" <<EOF
format=vokra-moss-audio-api-smoke-summary-v1
verdict=$verdict
exit_status=$smoke_status
expected_head=$expected_head
evidence_sha256=$evidence_sha
reference_transport=VAST_TO_APPLE_DIRECT
local_recovery=SMALL_LOGS_ONLY
checkpoint_load=NOT_PERFORMED
upload=FORBIDDEN
EOF
  if (( smoke_status != 0 )); then
    die "API smoke did not pass (exit=$smoke_status); evidence=$evidence; destroy the VAST instance"
  fi
  log "PASS: no checkpoint load and no upload; evidence/reference packet goes VAST->Apple directly; only small logs are recovered locally; evidence=$evidence; destroy the VAST instance"
}

main "$@"
