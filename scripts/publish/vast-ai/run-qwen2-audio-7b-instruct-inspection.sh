#!/usr/bin/env bash
# VAST-only Qwen2-Audio-7B-Instruct inspection wave.
#
# The five-shard 7B audio/text-to-text checkpoint is inventoried without
# conversion or runtime execution.  Model/source/Transformers identities and
# license evidence remain separate; this worker always stops before parity.

set -euo pipefail

UPSTREAM_REPOSITORY="Qwen/Qwen2-Audio-7B-Instruct"
UPSTREAM_REVISION="0a095220c30b7b31434169c3086508ef3ea5bf0a"
SOURCE_REPOSITORY="https://github.com/QwenLM/Qwen2-Audio.git"
SOURCE_REVISION="595360e82b5839c1507492ec83cae5bda6d5c7d4"
TRANSFORMERS_REPOSITORY="https://github.com/huggingface/transformers"
TRANSFORMERS_TAG="v4.45.0"
TRANSFORMERS_REVISION="2ef31dec1676249d26044a8aa8abe33dbecf0d10"
INSPECTOR="tools/parity/qwen2_audio_7b_instruct_inspect.py"
UV_CMD=(uv run --frozen --project tools/parity --python 3.12 python)
MIN_VAST_MEM_KIB=$((128 * 1024 * 1024))
MIN_FREE_DISK_KIB=$((60 * 1024 * 1024))

usage() {
  cat <<'EOF'
Usage:
  run-qwen2-audio-7b-instruct-inspection.sh [--work-dir <tmpfs-dir>]
  run-qwen2-audio-7b-instruct-inspection.sh --self-test

The real path is VAST-only: Linux x86_64, clean checkout, 128 GiB RAM, and
tmpfs storage are required. It snapshots the exact five-shard Qwen2-Audio
release, official Qwen source, and Transformers v4.45.0 commit. The result is
always fail-closed: no conversion, runtime, parity, upload, or publication.
EOF
}

die() {
  echo "run-qwen2-audio-7b-instruct-inspection: $*" >&2
  exit 2
}

canonicalize_github_remote() {
  local remote="$1" expected_path="$2" value path
  value="$remote"
  [[ "$value" != */ ]] || value="${value%/}"
  case "$value" in
    https://github.com/*) path="${value#https://github.com/}" ;;
    ssh://git@github.com/*) path="${value#ssh://git@github.com/}" ;;
    git@github.com:*) path="${value#git@github.com:}" ;;
    *) return 1 ;;
  esac
  [[ "$path" != */ ]] || return 1
  [[ "$path" == *.git ]] && path="${path%.git}"
  [[ "$path" == "$expected_path" ]] || return 1
  printf 'https://github.com/%s\n' "$path"
}

run_self_test() {
  local script_path="${BASH_SOURCE[0]}" repo_root fail=0 cases=0 required status
  repo_root="$(cd "$(dirname "$script_path")/../../.." && pwd)"
  [[ -f "$repo_root/$INSPECTOR" ]] || die "inspection oracle is missing"
  cases=$((cases + 1))
  for required in \
    "$UPSTREAM_REPOSITORY" "$UPSTREAM_REVISION" "$SOURCE_REPOSITORY" \
    "$SOURCE_REVISION" "$TRANSFORMERS_REPOSITORY" "$TRANSFORMERS_TAG" \
    "$TRANSFORMERS_REVISION" "$INSPECTOR" "SHARD_COUNT" "safe_open" \
    "server-tree" "resolved_revision" "RepoFile" "model_info" "demo/web_demo_audio.py" "audio_frontend" "decoder_configuration" "SOURCE_LICENSE_UNKNOWN_BLOCKER" "MAX_HEADER_BYTES" "evidence_stage" "INSPECTION_ONLY" "weights_only=True" "NOT_IMPLEMENTED_FAIL_CLOSED" \
    "UNSUPPORTED" "BLOCKED_BY_CPU" "NO_UPLOAD" "MIN_VAST_MEM_KIB" \
    "MIN_FREE_DISK_KIB"; do
    if ! grep -Fq -- "$required" "$script_path" && ! grep -Fq -- "$required" "$repo_root/$INSPECTOR"; then
      echo "run-qwen2-audio-7b-instruct-inspection: self-test FAIL: missing contract: $required" >&2
      fail=1
    fi
  done
  local obsolete_tag obsolete_revision
  obsolete_tag="$(printf 'v4.%s' '44.0')"
  obsolete_revision="$(printf '%s%s' '984bc11b0882' 'ff1e5b34ba717ea357e069ceced9')"
  if grep -Fq "$obsolete_tag" "$script_path" || grep -Fq "$obsolete_revision" "$script_path"; then
    echo "run-qwen2-audio-7b-instruct-inspection: self-test FAIL: obsolete Transformers pin found" >&2
    fail=1
  fi
  cases=$((cases + 1))
  for required in \
    'uname -s' 'uname -m' 'VOKRA_PUBLISH_ON_VAST' \
    'git status --porcelain --untracked-files=all' 'findmnt' \
    'cargo fmt --all -- --check' 'cargo build --locked --release -p vokra-cli' \
    'uv run --frozen --project tools/parity --python 3.12' \
    'snapshot_download' 'allow_patterns=["*"]' 'list_repo_tree' \
    'git clone --no-tags --filter=blob:none' 'CARGO_BUILD_JOBS'; do
    if ! grep -Fq -- "$required" "$script_path"; then
      echo "run-qwen2-audio-7b-instruct-inspection: self-test FAIL: missing VAST gate: $required" >&2
      fail=1
    fi
  done
  cases=$((cases + 1))
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh)([[:space:]]|$)' "$script_path" >/dev/null; then
    echo "run-qwen2-audio-7b-instruct-inspection: self-test FAIL: publication command found" >&2
    fail=1
  fi
  if grep -En '(^|[[:space:]])(python|python3|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    echo "run-qwen2-audio-7b-instruct-inspection: self-test FAIL: raw Python/pip invocation found" >&2
    fail=1
  fi
  if grep -En 'vokra-cli[[:space:]]+convert|cargo[[:space:]]+(run|test|check)' "$script_path" >/dev/null; then
    echo "run-qwen2-audio-7b-instruct-inspection: self-test FAIL: conversion/local Cargo command found" >&2
    fail=1
  fi
  cases=$((cases + 1))
  local remote canonical
  for remote in \
    "https://github.com/QwenLM/Qwen2-Audio" \
    "https://github.com/QwenLM/Qwen2-Audio.git" \
    "https://github.com/QwenLM/Qwen2-Audio.git/" \
    "ssh://git@github.com/QwenLM/Qwen2-Audio.git" \
    "git@github.com:QwenLM/Qwen2-Audio.git"; do
    canonical="$(canonicalize_github_remote "$remote" "QwenLM/Qwen2-Audio")" || {
      echo "run-qwen2-audio-7b-instruct-inspection: self-test FAIL: accepted remote rejected: $remote" >&2
      fail=1
      continue
    }
    [[ "$canonical" == "https://github.com/QwenLM/Qwen2-Audio" ]] || {
      echo "run-qwen2-audio-7b-instruct-inspection: self-test FAIL: remote canonicalization drift: $remote" >&2
      fail=1
    }
  done
  for remote in \
    "https://github.com/other/Qwen2-Audio.git" \
    "https://evil.example/QwenLM/Qwen2-Audio.git" \
    "https://github.com/QwenLM/Qwen2-Audio/extra.git" \
    "git://github.com/QwenLM/Qwen2-Audio.git"; do
    if canonicalize_github_remote "$remote" "QwenLM/Qwen2-Audio" >/dev/null 2>&1; then
      echo "run-qwen2-audio-7b-instruct-inspection: self-test FAIL: foreign remote accepted: $remote" >&2
      fail=1
    fi
  done
  cases=$((cases + 1))
  if bash "$script_path" --self-test --work-dir /tmp/qwen2-audio-self-test >/dev/null 2>&1; then
    echo "run-qwen2-audio-7b-instruct-inspection: self-test FAIL: extra argument accepted" >&2
    fail=1
  else
    status=$?
    if [[ "$status" != 2 ]]; then
      echo "run-qwen2-audio-7b-instruct-inspection: self-test FAIL: blocker exited $status, expected 2" >&2
      fail=1
    fi
  fi
  if bash "$script_path" --unknown-flag >/dev/null 2>&1; then
    echo "run-qwen2-audio-7b-instruct-inspection: self-test FAIL: unknown argument accepted" >&2
    fail=1
  else
    status=$?
    if [[ "$status" != 2 ]]; then
      echo "run-qwen2-audio-7b-instruct-inspection: self-test FAIL: unknown blocker exited $status, expected 2" >&2
      fail=1
    fi
  fi
  if (( fail == 0 )); then
    echo "run-qwen2-audio-7b-instruct-inspection.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

work_dir="/dev/shm/vokra-qwen2-audio-7b-inspection"
self_test=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --self-test) self_test=1; shift ;;
    --work-dir)
      [[ $# -ge 2 ]] || die "--work-dir requires a path"
      work_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if [[ $self_test -eq 1 ]]; then
  [[ "$work_dir" == "/dev/shm/vokra-qwen2-audio-7b-inspection" ]] \
    || die "--self-test accepts no other arguments"
  run_self_test
  exit $?
fi

[[ "$(uname -s)" == "Linux" ]] || die "actual inspection is Linux/VAST-only"
[[ "$(uname -m)" == "x86_64" ]] || die "actual inspection requires Linux x86_64"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
[[ -f Cargo.toml && -d crates/vokra-convert ]] || die "not a Vokra checkout"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "worktree is not clean"
work_parent="$(dirname "$work_dir")"
[[ -d "$work_parent" ]] || die "work-dir parent is missing: $work_parent"
[[ "$(findmnt -T "$work_parent" -no FSTYPE 2>/dev/null || true)" == "tmpfs" ]] || die "work-dir parent must be tmpfs/RAM-disk"
if [[ -e "$work_dir" ]]; then
  [[ -d "$work_dir" ]] || die "work-dir exists but is not a directory"
  [[ -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "work-dir must be absent or empty"
fi
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ && mem_kib -ge MIN_VAST_MEM_KIB ]] || die "host RAM is below 128-GiB guard"
free_kib="$(df -Pk "$work_parent" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ && free_kib -ge MIN_FREE_DISK_KIB ]] || die "tmpfs free space is below 60-GiB guard"
for command in cargo rustc rustfmt uv git sha256sum findmnt; do
  command -v "$command" >/dev/null 2>&1 || die "required VAST tool is missing: $command"
done

mkdir -p "$work_dir"
work_dir="$(cd "$work_dir" && pwd)"
log_path="$work_dir/inspection.log"
cache_dir="$work_dir/hf-cache"
snapshot_path_file="$work_dir/hf-snapshot-path.txt"
server_tree="$work_dir/hf-server-tree.json"
source_dir="$work_dir/official-source"
transformers_dir="$work_dir/transformers-source"
evidence_dir="$work_dir/evidence"
mkdir -p "$cache_dir" "$evidence_dir"

run_logged() {
  echo "+ $*" | tee -a "$log_path"
  "$@" 2>&1 | tee -a "$log_path"
}

export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
export RUST_BACKTRACE=1
run_logged cargo fmt --all -- --check
run_logged bash scripts/check-forbidden-symbols.sh
run_logged bash scripts/check-zero-deps.sh
run_logged cargo build --locked --release -p vokra-cli

run_logged "${UV_CMD[@]}" - "$UPSTREAM_REPOSITORY" "$UPSTREAM_REVISION" "$cache_dir" "$snapshot_path_file" "$server_tree" <<'PY'
import json
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, snapshot_download

repo, revision, cache_dir, snapshot_output, tree_output = sys.argv[1:]
api = HfApi()
info = api.model_info(repo_id=repo, revision=revision)
if info.sha != revision:
    raise SystemExit(f"HF revision drift: {info.sha!r} != {revision!r}")
resolved = Path(snapshot_download(repo_id=repo, revision=revision, cache_dir=cache_dir, allow_patterns=["*"], token=os.environ.get("HF_TOKEN") or os.environ.get("HF")))
if resolved.name != revision:
    raise SystemExit(f"snapshot revision drift: {resolved.name!r} != {revision!r}")
records = []
for item in api.list_repo_tree(repo_id=repo, revision=revision, recursive=True):
    # RepoFile has ``path``/``size``; RepoFolder has no size field in the
    # pinned huggingface_hub API.  Do not rely on a non-existent ``type``
    # attribute, or the authenticated tree would be recorded as empty.
    if not isinstance(item, RepoFile):
        continue
    size = getattr(item, "size", None)
    if not isinstance(size, int):
        raise SystemExit(f"server tree has unknown file size: {item.path}")
    lfs = getattr(item, "lfs", None)
    lfs_sha256 = getattr(lfs, "sha256", None)
    records.append({"path": item.path, "size": size, "oid": lfs_sha256 or getattr(item, "blob_id", None) or getattr(item, "xet_hash", None), "lfs_sha256": lfs_sha256})
Path(snapshot_output).write_text(str(resolved) + "\n", encoding="utf-8")
Path(tree_output).write_text(json.dumps({"repository": repo, "revision": revision, "resolved_revision": resolved.name, "files": sorted(records, key=lambda value: value["path"])}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
snapshot_path="$(< "$snapshot_path_file")"

run_logged git clone --no-tags --filter=blob:none "$SOURCE_REPOSITORY" "$source_dir"
run_logged git -C "$source_dir" checkout --detach "$SOURCE_REVISION"
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || die "official source revision mismatch"
canonicalize_github_remote "$(git -C "$source_dir" remote get-url origin)" "QwenLM/Qwen2-Audio" >/dev/null || die "official source remote mismatch"
run_logged git clone --filter=blob:none "$TRANSFORMERS_REPOSITORY" "$transformers_dir"
run_logged git -C "$transformers_dir" checkout --detach "$TRANSFORMERS_REVISION"
[[ "$(git -C "$transformers_dir" rev-parse HEAD)" == "$TRANSFORMERS_REVISION" ]] || die "Transformers revision mismatch"
[[ "$(git -C "$transformers_dir" describe --exact-match --tags HEAD)" == "$TRANSFORMERS_TAG" ]] || die "Transformers tag mismatch"
canonicalize_github_remote "$(git -C "$transformers_dir" remote get-url origin)" "huggingface/transformers" >/dev/null || die "Transformers remote mismatch"

set +e
"${UV_CMD[@]}" "$INSPECTOR" --snapshot "$snapshot_path" --source "$source_dir" --transformers "$transformers_dir" --server-tree "$server_tree" --output "$evidence_dir" 2>&1 | tee -a "$log_path"
inspector_status="${PIPESTATUS[0]}"
set -e
if [[ "$inspector_status" == "2" ]]; then
  grep -Fq '"status": "BLOCKED"' "$evidence_dir/manifest.json" || die "inspector exit 2 without BLOCKED manifest"
  grep -Fq '"evidence_stage": "INSPECTION_ONLY"' "$evidence_dir/manifest.json" || die "inspector manifest missing INSPECTION_ONLY evidence stage"
  echo "Qwen2-Audio inspection BLOCKED; evidence=$evidence_dir" >&2
  exit 2
fi
[[ "$inspector_status" == "0" ]] || die "inspector failed without blocker status"
die "inspection unexpectedly returned no dataset/provenance blocker"
