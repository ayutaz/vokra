#!/usr/bin/env bash
# VAST-only ACE-Step 1.5 composite-bundle inspection wave.
#
# This worker authenticates the immutable HF bundle and official source, then
# records component/tensor/container/dependency evidence.  It never converts,
# stages, uploads, or publishes a GGUF; every successful verdict remains
# INSPECTION_ONLY.

set -euo pipefail

UPSTREAM_REPOSITORY="ACE-Step/Ace-Step1.5"
UPSTREAM_REVISION="19671f406d603126926c1b7e2adc169acbcade22"
SOURCE_REPOSITORY="https://github.com/ace-step/ACE-Step-1.5"
SOURCE_REVISION="7202bc354d7fc31d1c0e5a90b0b49fb610e52362"
INSPECTOR="tools/parity/ace_step_v15_inspect.py"
UV_CMD=(uv run --frozen --project tools/parity --python 3.12 python)
MIN_VAST_MEM_KIB=$((128 * 1024 * 1024))
MIN_FREE_DISK_KIB=$((60 * 1024 * 1024))

usage() {
  cat <<'EOF'
Usage:
  run-ace-step-v15-inspection.sh [--work-dir <tmpfs-dir>]
  run-ace-step-v15-inspection.sh --self-test

The real path is VAST-only: Linux x86_64, clean checkout, 128 GiB RAM, and
tmpfs storage are required.  It downloads the exact ACE-Step 1.5 composite
bundle, inventories it with a safe inspection oracle, and remains
INSPECTION_ONLY.  No conversion, GGUF, runtime, parity, upload, or publish is
performed.
EOF
}

die() {
  echo "run-ace-step-v15-inspection: $*" >&2
  exit 2
}

run_self_test() {
  local script_path="${BASH_SOURCE[0]}" repo_root fail=0 cases=0 required status
  repo_root="$(cd "$(dirname "$script_path")/../../.." && pwd)"
  [[ -f "$repo_root/$INSPECTOR" ]] || die "inspection oracle is missing"
  cases=$((cases + 1))
  for required in \
    "$UPSTREAM_REPOSITORY" "$UPSTREAM_REVISION" "$SOURCE_REPOSITORY" \
    "$SOURCE_REVISION" "$INSPECTOR" "COMPONENTS" "safe_open" \
    "weights_only=True" "uv.lock" "INSPECTION_ONLY" "BLOCKED" \
    "NOT_IMPLEMENTED_FAIL_CLOSED" "UNSUPPORTED" "BLOCKED_BY_CPU" "NOT_RUN" "NO_UPLOAD" \
    "server-tree" "server/local tree mismatch" "UNAUTHENTICATED_BLOCKER" "UNREVIEWED_BLOCKER" \
    "source-inventory.json" "component-inventory.json" "companion-inventory.json" "tensor-inventory.json" \
    ".cache/huggingface" "symlink" "RepoFile" "RepoFolder" "classify_entry" "walk_tree" "path_in_repo" "recursive=False" \
    "MIN_VAST_MEM_KIB" "MIN_FREE_DISK_KIB"; do
    if ! grep -Fq -- "$required" "$script_path" && ! grep -Fq -- "$required" "$repo_root/$INSPECTOR"; then
      echo "run-ace-step-v15-inspection: self-test FAIL: missing contract: $required" >&2
      fail=1
    fi
  done
  cases=$((cases + 1))
  for required in \
    'uname -s' 'uname -m' 'VOKRA_PUBLISH_ON_VAST' \
    'git status --porcelain --untracked-files=all' 'findmnt' \
    'cargo fmt --all -- --check' 'cargo build --locked --release -p vokra-cli' \
    'uv run --frozen --project tools/parity --python 3.12' \
    'snapshot_download' 'local_dir' 'materialized snapshot' 'allow_patterns=["*"]' 'git clone --no-tags --filter=blob:none' \
    'CARGO_BUILD_JOBS=1'; do
    if ! grep -Fq -- "$required" "$script_path"; then
      echo "run-ace-step-v15-inspection: self-test FAIL: missing VAST gate: $required" >&2
      fail=1
    fi
  done
  cases=$((cases + 1))
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh)([[:space:]]|$)' "$script_path" >/dev/null; then
    echo "run-ace-step-v15-inspection: self-test FAIL: publication command found" >&2
    fail=1
  fi
  if grep -En '(^|[[:space:]])(python|python3|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    echo "run-ace-step-v15-inspection: self-test FAIL: raw Python/pip invocation found" >&2
    fail=1
  fi
  if grep -En 'vokra-cli[[:space:]]+convert' "$script_path" >/dev/null || grep -En 'cargo[[:space:]]+(run|test|check)' "$script_path" >/dev/null; then
    echo "run-ace-step-v15-inspection: self-test FAIL: conversion/local Cargo command found" >&2
    fail=1
  fi
  if grep -En 'weights_only=False|pickle\.load' "$repo_root/$INSPECTOR" >/dev/null; then
    echo "run-ace-step-v15-inspection: self-test FAIL: unsafe loader found" >&2
    fail=1
  fi
  cases=$((cases + 1))
  local tree_source
  tree_source="$(mktemp "${TMPDIR:-/tmp}/ace-step-hf-tree-self-test.XXXXXX.py")"
  awk '/^import hashlib$/{capture=1} capture && /^PY$/{exit} capture{print}' "$script_path" > "$tree_source"
  if ! ACE_STEP_HF_TREE_SELF_TEST=1 "${UV_CMD[@]}" "$tree_source" dummy dummy dummy; then
    echo "run-ace-step-v15-inspection: self-test FAIL: Hub tree class/path contract" >&2
    fail=1
  fi
  rm -f -- "$tree_source"
  cases=$((cases + 1))
  if bash "$script_path" --self-test --work-dir /tmp/ace-step-v15-self-test >/dev/null 2>&1; then
    echo "run-ace-step-v15-inspection: self-test FAIL: extra argument accepted" >&2
    fail=1
  else
    status=$?
    if [[ "$status" != 2 ]]; then
      echo "run-ace-step-v15-inspection: self-test FAIL: blocker exited $status, expected 2" >&2
      fail=1
    fi
  fi
  if bash "$script_path" --unknown-flag >/dev/null 2>&1; then
    echo "run-ace-step-v15-inspection: self-test FAIL: unknown argument accepted" >&2
    fail=1
  else
    status=$?
    if [[ "$status" != 2 ]]; then
      echo "run-ace-step-v15-inspection: self-test FAIL: unknown blocker exited $status, expected 2" >&2
      fail=1
    fi
  fi
  if (( fail == 0 )); then
    echo "run-ace-step-v15-inspection.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

work_dir="/dev/shm/vokra-ace-step-v15-inspection"
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
  [[ "$work_dir" == "/dev/shm/vokra-ace-step-v15-inspection" ]] \
    || die "--self-test accepts no other arguments"
  run_self_test
  exit $?
fi

[[ "$(uname -s)" == "Linux" ]] || die "actual inspection is Linux/VAST-only"
[[ "$(uname -m)" == "x86_64" ]] || die "actual inspection requires Linux x86_64"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
  || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
[[ -f Cargo.toml && -d crates/vokra-convert ]] || die "not a Vokra checkout"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] \
  || die "worktree is not clean; transfer a committed git-bundle checkpoint"
work_parent="$(dirname "$work_dir")"
[[ -d "$work_parent" ]] || die "work-dir parent is missing: $work_parent"
[[ "$(findmnt -T "$work_parent" -no FSTYPE 2>/dev/null || true)" == "tmpfs" ]] \
  || die "work-dir parent must be tmpfs/RAM-disk"
if [[ -e "$work_dir" ]]; then
  [[ -d "$work_dir" ]] || die "work-dir exists but is not a directory"
  [[ -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
    || die "work-dir must be absent or empty"
fi
[[ -r /proc/meminfo ]] || die "/proc/meminfo is unavailable"
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ && mem_kib -ge MIN_VAST_MEM_KIB ]] \
  || die "host RAM is below the 128-GiB guard"
free_kib="$(df -Pk "$work_parent" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ && free_kib -ge MIN_FREE_DISK_KIB ]] \
  || die "tmpfs free space is below the 60-GiB guard"
for command in cargo rustc rustfmt uv git sha256sum findmnt; do
  command -v "$command" >/dev/null 2>&1 || die "required VAST tool is missing: $command"
done

mkdir -p "$work_dir"
work_dir="$(cd "$work_dir" && pwd)"
log_path="$work_dir/inspection.log"
cache_dir="$work_dir/hf-cache"
snapshot_path_file="$work_dir/hf-snapshot-path.txt"
server_tree_path="$work_dir/server-tree.json"
source_dir="$work_dir/official-source"
evidence_dir="$work_dir/evidence"
mkdir -p "$cache_dir" "$evidence_dir"

run_logged() {
  echo "+ $*" | tee -a "$log_path"
  "$@" 2>&1 | tee -a "$log_path"
}

export CARGO_BUILD_JOBS=1
export RUST_BACKTRACE=1
run_logged cargo fmt --all -- --check
run_logged bash scripts/check-forbidden-symbols.sh
run_logged bash scripts/check-zero-deps.sh
run_logged cargo build --locked --release -p vokra-cli

run_logged "${UV_CMD[@]}" - "$UPSTREAM_REPOSITORY" "$UPSTREAM_REVISION" "$server_tree_path" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, RepoFolder

repo, revision, output = sys.argv[1:]
def field(item, name, default=None):
    return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)

def classify_entry(item):
    if isinstance(item, RepoFolder):
        if field(item, "type") not in {None, "directory"}:
            raise RuntimeError(f"invalid RepoFolder type: {item!r}")
        return "directory"
    if isinstance(item, RepoFile):
        if field(item, "type") not in {None, "file"}:
            raise RuntimeError(f"invalid RepoFile type: {item!r}")
        return "file"
    if isinstance(item, dict):
        kind = item.get("type")
        if kind in {"directory", "file"}:
            return kind
        if kind is None and any(key in item for key in ("size", "blob_id", "oid")):
            return "file"
    raise RuntimeError(f"unknown HF tree entry type: {type(item).__name__}")

def walk_tree(api, repository, revision):
    pending = [""]
    seen = set()
    while pending:
        path_in_repo = pending.pop()
        if path_in_repo in seen:
            continue
        seen.add(path_in_repo)
        for item in api.list_repo_tree(repo_id=repository, revision=revision, path_in_repo=path_in_repo, recursive=False):
            kind = classify_entry(item)
            path = field(item, "path")
            if not isinstance(path, str) or not path:
                raise RuntimeError(f"HF tree entry has no path: {item!r}")
            if kind == "directory":
                pending.append(path)
            else:
                yield item

if os.environ.get("ACE_STEP_HF_TREE_SELF_TEST") == "1":
    file_entry = RepoFile(path="model.bin", size=1, oid="a" * 40)
    file_entry.type = None
    folder_entry = RepoFolder(path="nested", oid="b" * 40)
    folder_entry.type = None
    dict_entry = {"path": "dict.bin", "type": None, "size": 1, "oid": "c" * 40}

    class FakeApi:
        def __init__(self):
            self.calls = []
            self.entries = {"": [folder_entry, file_entry, dict_entry], "nested": [{"path": "nested/file", "type": "file", "size": 1, "oid": "d" * 40}]}

        def list_repo_tree(self, **kwargs):
            self.calls.append(kwargs)
            return iter(self.entries[kwargs["path_in_repo"]])

    fake_api = FakeApi()
    rows = list(walk_tree(fake_api, repo, revision))
    assert classify_entry(folder_entry) == "directory"
    assert [classify_entry(item) for item in rows] == ["file", "file", "file"]
    assert [field(item, "path") for item in rows] == ["model.bin", "dict.bin", "nested/file"]
    assert fake_api.calls[0]["path_in_repo"] == ""
    assert fake_api.calls[0]["recursive"] is False
    assert fake_api.calls[1]["path_in_repo"] == "nested"
    try:
        classify_entry(object())
    except RuntimeError:
        pass
    else:
        raise AssertionError("unknown HF tree entry was accepted")
    print("ACE-Step Hub tree class/path/pagination self-test: PASS")
    raise SystemExit(0)

api = HfApi()
info = api.model_info(repo, revision=revision)
if info.sha != revision:
    raise SystemExit(f"HF revision drift: {info.sha!r} != {revision!r}")
rows = []

for item in walk_tree(api, repo, revision):
    path = field(item, "path")
    size = field(item, "size")
    blob = field(item, "blob_id") or field(item, "oid")
    if not isinstance(path, str) or not isinstance(size, int) or not isinstance(blob, str):
        raise SystemExit(f"HF file metadata is incomplete: {path!r}")
    row = {"path": path, "type": "file", "size": size, "git_blob_sha1": blob}
    lfs = field(item, "lfs")
    lfs_sha = field(lfs, "sha256") if lfs is not None else None
    lfs_size = field(lfs, "size") if lfs is not None else None
    if isinstance(lfs, dict):
        lfs_sha = lfs.get("sha256")
        lfs_size = lfs.get("size")
    if lfs_sha is not None or lfs_size is not None:
        if not isinstance(lfs_sha, str) or not isinstance(lfs_size, int):
            raise SystemExit(f"HF LFS metadata is incomplete: {path!r}")
        pointer = (b"version https://git-lfs.github.com/spec/v1\n"
                   + f"oid sha256:{lfs_sha}\nsize {lfs_size}\n".encode("ascii"))
        row.update({"lfs_sha256": lfs_sha, "lfs_size": lfs_size,
                    "lfs_pointer_sha1": hashlib.sha1(f"blob {len(pointer)}\0".encode("ascii") + pointer).hexdigest()})
    rows.append(row)
Path(output).write_text(json.dumps({"repository": repo, "requested_revision": revision, "revision": revision, "resolved_revision": info.sha, "files": rows}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY

run_logged "${UV_CMD[@]}" - "$UPSTREAM_REPOSITORY" "$UPSTREAM_REVISION" "$cache_dir" "$snapshot_path_file" <<'PY'
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo, revision, cache_dir, output = sys.argv[1:]
local_dir = Path(output).parent / "hf-snapshot"
if local_dir.exists():
    raise SystemExit(f"materialized snapshot destination already exists: {local_dir}")
resolved = Path(snapshot_download(
    repo_id=repo,
    revision=revision,
    cache_dir=cache_dir,
    local_dir=str(local_dir),
    allow_patterns=["*"],
    token=os.environ.get("HF_TOKEN") or os.environ.get("HF"),
))
if resolved.resolve() != local_dir.resolve():
    raise SystemExit(f"snapshot was not materialized at the fixed local directory: {resolved!r}")
if any(path.is_symlink() for path in resolved.rglob("*")):
    raise SystemExit("materialized snapshot contains a symlink")
Path(output).write_text(str(local_dir.resolve()) + "\n", encoding="utf-8")
PY
snapshot_path="$(< "$snapshot_path_file")"

run_logged git clone --no-tags --filter=blob:none "$SOURCE_REPOSITORY" "$source_dir"
run_logged git -C "$source_dir" checkout --detach "$SOURCE_REVISION"
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$SOURCE_REVISION" ]] \
  || die "official source revision mismatch"
[[ "$(git -C "$source_dir" remote get-url origin | sed 's#/$##; s#\.git$##')" == "$SOURCE_REPOSITORY" ]] \
  || die "official source remote mismatch"

set +e
"${UV_CMD[@]}" "$INSPECTOR" --snapshot "$snapshot_path" --source "$source_dir" --output "$evidence_dir" --server-tree "$server_tree_path" 2>&1 | tee -a "$log_path"
inspector_status="${PIPESTATUS[0]}"
set -e
[[ "$inspector_status" == "2" ]] || die "inspector must remain fail-closed with exit 2"
grep -Fq '"status": "BLOCKED"' "$evidence_dir/manifest.json" \
  || die "inspector exit 2 without BLOCKED evidence manifest"
grep -Fq '"runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED"' "$evidence_dir/manifest.json" \
  || die "runtime fail-closed status missing"
grep -Fq '"publication": "NO_UPLOAD"' "$evidence_dir/manifest.json" \
  || die "publication status missing"
{
  echo "verdict=INSPECTION_ONLY_BLOCKED"
  echo "runtime_status=NOT_IMPLEMENTED_FAIL_CLOSED"
  echo "cpu_status=UNSUPPORTED"
  echo "metal_status=BLOCKED_BY_CPU"
  echo "parity_status=NOT_RUN"
  echo "publication=NO_UPLOAD"
  echo "upstream_revision=$UPSTREAM_REVISION"
  echo "source_revision=$SOURCE_REVISION"
  echo "manifest_sha256=$(sha256sum "$evidence_dir/manifest.json" | awk '{print $1}')"
} | tee "$evidence_dir/summary.txt"
echo "ACE-Step 1.5 inspection blocked (by contract); evidence=$evidence_dir" >&2
exit 2
