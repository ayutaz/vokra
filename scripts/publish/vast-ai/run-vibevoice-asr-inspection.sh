#!/usr/bin/env bash
# VAST-only VibeVoice-ASR inspection wave.
#
# The canonical Microsoft 9B ASR release is eight large safetensors shards.
# This worker authenticates the immutable HF snapshot and official source,
# inventories shards/tensors through a streaming oracle, and stops before any
# converter/runtime/parity claim.  BitNet and VibeASR.cpp are deliberately out
# of scope and are never substituted for this artifact.

set -euo pipefail

UPSTREAM_REPOSITORY="microsoft/VibeVoice-ASR"
UPSTREAM_REVISION="d0c9efdb8d614685062c04425d91e01b6f37d944"
SOURCE_REPOSITORY="https://github.com/microsoft/VibeVoice"
SOURCE_REVISION="94da20d98b2fa7688e9cbfaf7692ddb4954f7600"
TRANSFORMERS_REPOSITORY="https://github.com/huggingface/transformers"
TRANSFORMERS_TAG="v4.51.3"
TRANSFORMERS_REVISION="5f4ecf2d9f867a1255131d2461d75793c0cf1db2"
INSPECTOR="tools/parity/vibevoice_asr_inspect_reference.py"
UV_CMD=(uv run --frozen --project tools/parity --python 3.12 python)
MIN_VAST_MEM_KIB=$((128 * 1024 * 1024))
MIN_FREE_DISK_KIB=$((60 * 1024 * 1024))

usage() {
  cat <<'EOF'
Usage:
  run-vibevoice-asr-inspection.sh [--work-dir <tmpfs-dir>]
  run-vibevoice-asr-inspection.sh --self-test

The real path is VAST-only: Linux x86_64, clean checkout, 128 GiB RAM, and
tmpfs storage are required. It snapshots the exact eight-shard
microsoft/VibeVoice-ASR release and the pinned Microsoft source, then records
streaming shard/index/tensor/companion/source evidence. Verdict is always
INSPECTION_ONLY; no conversion, runtime, CPU, Metal, or parity result exists.
EOF
}

die() {
  echo "run-vibevoice-asr-inspection: $*" >&2
  exit 2
}

run_self_test() {
  local script_path="${BASH_SOURCE[0]}" repo_root fail=0 cases=0 required
  repo_root="$(cd "$(dirname "$script_path")/../../.." && pwd)"
  [[ -f "$repo_root/$INSPECTOR" ]] || die "inspection oracle is missing"
  cases=$((cases + 1))
  for required in \
    "$UPSTREAM_REPOSITORY" "$UPSTREAM_REVISION" "$SOURCE_REPOSITORY" \
    "$SOURCE_REVISION" "$TRANSFORMERS_REPOSITORY" "$TRANSFORMERS_TAG" "$TRANSFORMERS_REVISION" "$INSPECTOR" "safe_open" "SHARD_COUNT" \
    "model.safetensors.index.json" "INSPECTION_ONLY" \
    "BLOCKED" "Exception" "--transformers-source" "resident_scope" "allow_patterns=[\"*\"]" "companion-inventory" "source-inventory" "transformers-inventory" "MIN_VAST_MEM_KIB" \
    "MIN_FREE_DISK_KIB" "tmpfs" "server-tree.json" "local_dir" "requested_revision" "resolved_revision" "recursive_file_only" "RepoFolder" "expand=True" "lfs_pointer_git_blob_sha1" "UNSELECTED_BLOCKER" "NOT_DOWNLOADED" "transport_cache" "snapshot_root_exact_transport_subtree" "NON_IDENTITY_TRANSPORT_METADATA" "connector_topology" "acoustic_connector" "semantic_connector" "symlinks" "120000" "gitlink"; do
    if ! grep -Fq -- "$required" "$script_path" && ! grep -Fq -- "$required" "$repo_root/$INSPECTOR"; then
      echo "run-vibevoice-asr-inspection: self-test FAIL: missing contract: $required" >&2
      fail=1
    fi
  done
  cases=$((cases + 1))
  for required in \
    'uname -s' 'uname -m' 'VOKRA_PUBLISH_ON_VAST' \
    'git status --porcelain --untracked-files=all' 'findmnt' \
    'cargo fmt --all -- --check' 'cargo build --locked --release -p vokra-cli' \
    'uv run --frozen --project tools/parity --python 3.12' \
    'snapshot_download' 'git clone --no-tags --filter=blob:none' 'exit 2'; do
    if ! grep -Fq -- "$required" "$script_path"; then
      echo "run-vibevoice-asr-inspection: self-test FAIL: missing VAST gate: $required" >&2
      fail=1
    fi
  done
  cases=$((cases + 1))
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh)([[:space:]]|$)' "$script_path" >/dev/null; then
    echo "run-vibevoice-asr-inspection: self-test FAIL: publication command found" >&2
    fail=1
  fi
  if grep -En '(^|[[:space:]])(python|python3|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    echo "run-vibevoice-asr-inspection: self-test FAIL: raw Python/pip invocation found" >&2
    fail=1
  fi
  if grep -En 'cargo[[:space:]]+(run|test|check)' "$script_path" >/dev/null; then
    echo "run-vibevoice-asr-inspection: self-test FAIL: local-style Cargo execution found" >&2
    fail=1
  fi
  cases=$((cases + 1))
  if bash "$script_path" --self-test --work-dir /tmp/vibevoice-asr-self-test >/dev/null 2>&1; then
    echo "run-vibevoice-asr-inspection: self-test FAIL: extra argument accepted" >&2
    fail=1
  else
    status=$?
    if [[ "$status" != 2 ]]; then
      echo "run-vibevoice-asr-inspection: self-test FAIL: contract blocker exited $status, expected 2" >&2
      fail=1
    fi
  fi
  if bash "$script_path" --unknown-flag >/dev/null 2>&1; then
    echo "run-vibevoice-asr-inspection: self-test FAIL: unknown argument accepted" >&2
    fail=1
  else
    status=$?
    if [[ "$status" != 2 ]]; then
      echo "run-vibevoice-asr-inspection: self-test FAIL: unknown-argument blocker exited $status, expected 2" >&2
      fail=1
    fi
  fi
  if (( fail == 0 )); then
    echo "run-vibevoice-asr-inspection.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

work_dir="/dev/shm/vokra-vibevoice-asr-inspection"
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
  [[ "$work_dir" == "/dev/shm/vokra-vibevoice-asr-inspection" ]] \
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
local_dir="$work_dir/hf-snapshot"
snapshot_path_file="$work_dir/hf-snapshot-path.txt"
server_tree_path="$work_dir/server-tree.json"
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

run_logged "${UV_CMD[@]}" - "$UPSTREAM_REPOSITORY" "$UPSTREAM_REVISION" "$cache_dir" "$local_dir" "$snapshot_path_file" <<'PY'
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo, revision, cache_dir, local_dir, output = sys.argv[1:]
resolved = Path(snapshot_download(
    repo_id=repo,
    revision=revision,
    cache_dir=cache_dir,
    local_dir=local_dir,
    # Download the complete non-weight tree: extension-based allowlists can
    # silently omit LICENSE, .gitattributes, or extensionless companions.
    allow_patterns=["*"],
    token=os.environ.get("HF_TOKEN") or os.environ.get("HF"),
))
if resolved.resolve() != Path(local_dir).resolve():
    raise SystemExit(f"local_dir materialization mismatch: {resolved} != {local_dir}")
for required in ("model.safetensors.index.json", "config.json"):
    if not (resolved / required).is_file():
        raise SystemExit(f"required HF companion missing: {required}")
Path(output).write_text(str(resolved) + "\n", encoding="utf-8")
PY
snapshot_path="$(< "$snapshot_path_file")"

run_logged "${UV_CMD[@]}" - "$UPSTREAM_REPOSITORY" "$UPSTREAM_REVISION" "$snapshot_path" "$server_tree_path" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, RepoFolder

repo, revision, snapshot, output = sys.argv[1:]
api = HfApi()
info = api.model_info(repo_id=repo, revision=revision)
if info.sha != revision:
    raise SystemExit(f"resolved revision mismatch: {info.sha} != {revision}")
rows = []; seen = set()
for item in api.list_repo_tree(repo_id=repo, revision=revision, recursive=True, expand=True):
    if isinstance(item, RepoFolder):
        continue
    if not isinstance(item, RepoFile):
        raise SystemExit(f"unknown HF tree entry type: {type(item).__name__}")
    path = item.path
    if not isinstance(path, str) or not path or "\x00" in path or "\\" in path or path.startswith("/") or ".." in Path(path).parts or path in seen:
        raise SystemExit(f"unsafe/duplicate HF path: {path!r}")
    seen.add(path)
    size = item.size; blob = getattr(item, "blob_id", None)
    if not isinstance(size, int) or isinstance(size, bool) or size < 0 or not isinstance(blob, str) or not re.fullmatch(r"[0-9a-f]{40}", blob):
        raise SystemExit(f"invalid HF file identity: {path}")
    lfs = getattr(item, "lfs", None)
    lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    lfs_oid = lfs.get("oid") if isinstance(lfs, dict) else getattr(lfs, "oid", None)
    lfs_size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
    if lfs_sha is None:
        if lfs_oid is not None or lfs_size is not None:
            raise SystemExit(f"regular file has LFS metadata: {path}")
        row = {"path": path, "type": "file", "size": size, "git_blob_sha1": blob, "lfs_pointer_git_blob_sha1": None, "lfs_sha256": None}
    else:
        if not isinstance(lfs_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", lfs_sha) or (lfs_oid is not None and (not isinstance(lfs_oid, str) or not re.fullmatch(r"[0-9a-f]{64}", lfs_oid) or lfs_oid != lfs_sha)) or not isinstance(lfs_size, int) or isinstance(lfs_size, bool) or lfs_size < 0 or lfs_size != size:
            raise SystemExit(f"invalid LFS identity/size: {path}")
        pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{lfs_sha}\nsize {size}\n".encode()
        pointer_id = hashlib.sha1(f"blob {len(pointer)}\0".encode() + pointer).hexdigest()
        if pointer_id != blob:
            raise SystemExit(f"canonical LFS pointer mismatch: {path}")
        row = {"path": path, "type": "file", "size": size, "git_blob_sha1": None, "lfs_pointer_git_blob_sha1": blob, "lfs_sha256": lfs_sha}
    rows.append(row)
Path(output).write_text(json.dumps({"repository": repo, "requested_revision": revision, "resolved_revision": info.sha, "walk": "recursive_file_only", "files": sorted(rows, key=lambda row: row["path"])}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY

run_logged git clone --no-tags --filter=blob:none "$SOURCE_REPOSITORY" "$source_dir"
run_logged git -C "$source_dir" checkout --detach "$SOURCE_REVISION"
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$SOURCE_REVISION" ]] \
  || die "official source revision mismatch"
[[ "$(git -C "$source_dir" remote get-url origin | sed 's#/$##; s#\.git$##')" == "$SOURCE_REPOSITORY" ]] \
  || die "official source remote mismatch"
run_logged git clone --filter=blob:none "$TRANSFORMERS_REPOSITORY" "$transformers_dir"
run_logged git -C "$transformers_dir" checkout --detach "$TRANSFORMERS_REVISION"
[[ "$(git -C "$transformers_dir" rev-parse HEAD)" == "$TRANSFORMERS_REVISION" ]] \
  || die "Transformers commit mismatch"
[[ "$(git -C "$transformers_dir" describe --exact-match --tags HEAD)" == "$TRANSFORMERS_TAG" ]] \
  || die "Transformers tag mismatch"
[[ "$(git -C "$transformers_dir" remote get-url origin | sed 's#/$##; s#\.git$##')" == "$TRANSFORMERS_REPOSITORY" ]] \
  || die "Transformers source remote mismatch"

set +e
"${UV_CMD[@]}" "$INSPECTOR" \
  --snapshot "$snapshot_path" --source "$source_dir" \
  --transformers-source "$transformers_dir" --server-tree "$server_tree_path" --output "$evidence_dir" \
  2>&1 | tee -a "$log_path"
inspector_status="${PIPESTATUS[0]}"
set -e
[[ "$inspector_status" == "2" ]] || die "inspector did not return expected exit 2"
run_logged "${UV_CMD[@]}" - "$evidence_dir/manifest.json" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("status") != "BLOCKED" or manifest.get("evidence_stage") != "INSPECTION_ONLY": raise SystemExit("invalid fail-closed manifest")
if manifest.get("inspection_status") != "AUTHENTICATED_EVIDENCE_COMPLETE" or manifest.get("collection_status") != "AUTHENTICATED": raise SystemExit("inspection evidence is incomplete")
if manifest.get("runtime_status") != "NOT_IMPLEMENTED_FAIL_CLOSED" or manifest.get("cpu_status") != "UNSUPPORTED" or manifest.get("metal_status") != "BLOCKED_BY_CPU" or manifest.get("parity_status") != "NOT_RUN" or manifest.get("publication") != "NO_UPLOAD": raise SystemExit("unsafe runtime/publication status")
dependency = manifest.get("external_dependency", {})
if dependency != {"repository": "Qwen/Qwen2.5-7B", "revision": "UNSELECTED_BLOCKER", "selection_status": "BLOCKED", "files": "NOT_DOWNLOADED", "model_weights": "NOT_DOWNLOADED"}: raise SystemExit("Qwen external dependency was not fail-closed")
if manifest.get("tensor_count", 0) <= 0 or manifest.get("upstream", {}).get("shard_count") != 8: raise SystemExit("incomplete shard/tensor evidence")
transport = manifest.get("hf_server_tree", {}).get("transport_cache")
if not isinstance(transport, dict) or transport.get("path") != ".cache/huggingface" or transport.get("scope") != "snapshot_root_exact_transport_subtree" or transport.get("identity_role") != "NON_IDENTITY_TRANSPORT_METADATA": raise SystemExit("transport cache evidence is missing or has the wrong scope")
if transport.get("status") not in {"ABSENT", "EXCLUDED"} or not isinstance(transport.get("present"), bool): raise SystemExit("invalid transport cache evidence")
if transport["status"] == "ABSENT" and transport["present"] is not False: raise SystemExit("absent transport cache was marked present")
if transport["status"] == "EXCLUDED" and transport["present"] is not True: raise SystemExit("excluded transport cache was not marked present")
if transport["status"] == "EXCLUDED" and (not isinstance(transport.get("entry_count"), int) or isinstance(transport.get("entry_count"), bool) or transport["entry_count"] < 0): raise SystemExit("transport cache entry count is invalid")
connector = manifest.get("official_source", {}).get("connector_topology")
if not isinstance(connector, dict) or connector.get("path") != "vibevoice/modular/modeling_vibevoice_asr.py" or connector.get("status") != "AUTHENTICATED" or connector.get("role_blob_authenticated") is not True or connector.get("markers") != {"acoustic": True, "semantic": True}: raise SystemExit("ASR connector topology evidence is incomplete")
symlinks = manifest.get("official_source", {}).get("transformers", {}).get("symlinks")
if not isinstance(symlinks, list) or len(symlinks) != 2 or {item.get("path") for item in symlinks if isinstance(item, dict)} != {"docs/source/en/contributing.md", "docs/source/en/notebooks.md"} or any(not isinstance(item, dict) or item.get("mode") != "120000" or item.get("status") != "AUTHENTICATED" for item in symlinks): raise SystemExit("Transformers symlink evidence is incomplete")
PY

{
  echo "verdict=BLOCKED; evidence_stage=INSPECTION_ONLY"
  echo "runtime_parity=NOT_RUN"
  echo "numerical_parity=NOT_RUN"
  echo "upstream_revision=$UPSTREAM_REVISION"
  echo "source_revision=$SOURCE_REVISION"
  echo "manifest_sha256=$(sha256sum "$evidence_dir/manifest.json" | awk '{print $1}')"
  echo "shard_inventory_sha256=$(sha256sum "$evidence_dir/shard-inventory.json" | awk '{print $1}')"
  echo "tensor_inventory_sha256=$(sha256sum "$evidence_dir/tensor-inventory.json" | awk '{print $1}')"
  echo "companion_inventory_sha256=$(sha256sum "$evidence_dir/companion-inventory.json" | awk '{print $1}')"
  echo "source_inventory_sha256=$(sha256sum "$evidence_dir/source-inventory.json" | awk '{print $1}')"
} | tee "$evidence_dir/summary.txt"
echo "VibeVoice-ASR inspection evidence: $evidence_dir"
