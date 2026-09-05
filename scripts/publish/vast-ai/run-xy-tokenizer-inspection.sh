#!/usr/bin/env bash
# VAST-only XY-Tokenizer inspection wave.
#
# The upstream checkpoint is a >2 GiB PyTorch artifact.  This worker performs
# the only permitted safe-load/preparation on VAST, records the complete tensor
# inventory, and dry-runs the existing converter against that authenticated
# replacement.  There is no native binder or numerical parity verdict.

set -euo pipefail

UPSTREAM_REPOSITORY="OpenMOSS-Team/XY_Tokenizer_TTSD_V0"
UPSTREAM_REVISION="c83433728e698ed0698e88cb5096bc221fb8f8c5"
CHECKPOINT_FILENAME="xy_tokenizer.ckpt"
CHECKPOINT_BYTES=2137328977
CHECKPOINT_SHA256="37c7ac18d0a48f5a1d0687e31af7c0264861232c500206718c98acd8e37d1671"
SOURCE_REPOSITORY="https://github.com/gyt1145028706/XY-Tokenizer"
SOURCE_REVISION="5df5609c5883e555bd39a2d0b1005ca8f1a8f12e"
CONFIG_RELATIVE="config/xy_tokenizer_config.yaml"
CONFIG_SHA256="e7d48677e34f77e5b9fd7dc7a3e0eef7f2d2dd9be9a245d5c1d56489dc748938"
INSPECTOR="tools/parity/xy_tokenizer_inspect_reference.py"
UV_CMD=(uv run --frozen --project tools/parity --python 3.12 python)
MIN_VAST_MEM_KIB=$((128 * 1024 * 1024))
MIN_FREE_DISK_KIB=$((30 * 1024 * 1024))

usage() {
  cat <<'EOF'
Usage:
  run-xy-tokenizer-inspection.sh [--work-dir <tmpfs-dir>]
  run-xy-tokenizer-inspection.sh --self-test

The real path is VAST-only: Linux x86_64, clean checkout, 128 GiB RAM, and
tmpfs work storage are required.  It authenticates the pinned HF checkpoint,
official source/config, and safe-prepares a safetensors replacement.  The
result is always INSPECTION_ONLY; no runtime or numerical parity is claimed.
EOF
}

die() {
  echo "run-xy-tokenizer-inspection: $*" >&2
  exit 1
}

run_self_test() {
  local script_path="${BASH_SOURCE[0]}" repo_root fail=0 cases=0 required
  repo_root="$(cd "$(dirname "$script_path")/../../.." && pwd)"
  [[ -f "$repo_root/$INSPECTOR" ]] || die "inspection oracle is missing"

  cases=$((cases + 1))
  for required in \
    "$UPSTREAM_REPOSITORY" "$UPSTREAM_REVISION" "$CHECKPOINT_FILENAME" \
    "$CHECKPOINT_BYTES" "$CHECKPOINT_SHA256" "$SOURCE_REPOSITORY" \
    "$SOURCE_REVISION" "$CONFIG_RELATIVE" "$CONFIG_SHA256" "$INSPECTOR" \
    "weights_only=True" "save_file" "INSPECTION_ONLY" "torch.load" \
    "HfApi" "RepoFile" "RepoFolder" "server-packet" "collection_status" \
    "AUTHENTICATED_EVIDENCE_COMPLETE" "NO_UPLOAD" \
    "SOURCE_LICENSE_README_DECLARATION_NO_FULL_FILE" "SOURCE_LICENSE_EVIDENCE_UNAVAILABLE" \
    "SOURCE_README_BLOB_SHA1" "SOURCE_README_SHA256" "SOURCE_LICENSE_HEADING" \
    "SOURCE_LICENSE_DECLARATION" \
    "SELECTED_MODEL_FILES" "source_config" "prepared_config" "AUTHENTICATED_OFFICIAL_SOURCE" \
    "all_tracked_regular_files" "inference.py" "xy_tokenizer/model.py" \
    "MIN_VAST_MEM_KIB" "MIN_FREE_DISK_KIB" "tmpfs"; do
    if ! grep -Fq -- "$required" "$script_path" && ! grep -Fq -- "$required" "$repo_root/$INSPECTOR"; then
      echo "run-xy-tokenizer-inspection: self-test FAIL: missing contract: $required" >&2
      fail=1
    fi
  done
  cases=$((cases + 1))
  for required in \
    'uname -s' 'uname -m' 'VOKRA_PUBLISH_ON_VAST' \
    'git status --porcelain --untracked-files=all' 'findmnt' \
    'cargo fmt --all -- --check' 'cargo build --locked --release -p vokra-cli' \
    'uv run --frozen --project tools/parity --python 3.12' \
    'vokra-cli" convert --model xy-tokenizer' 'apache-2.0' 'exit 2'; do
    if ! grep -Fq -- "$required" "$script_path"; then
      echo "run-xy-tokenizer-inspection: self-test FAIL: missing VAST gate: $required" >&2
      fail=1
    fi
  done
  cases=$((cases + 1))
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh)([[:space:]]|$)' "$script_path" >/dev/null; then
    echo "run-xy-tokenizer-inspection: self-test FAIL: publication command found" >&2
    fail=1
  fi
  if grep -En '(^|[[:space:]])(python|python3|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    echo "run-xy-tokenizer-inspection: self-test FAIL: raw Python/pip invocation found" >&2
    fail=1
  fi
  if grep -En 'weights_only=False|pickle\.load' "$repo_root/$INSPECTOR" >/dev/null; then
    echo "run-xy-tokenizer-inspection: self-test FAIL: unsafe checkpoint loader found" >&2
    fail=1
  fi

  cases=$((cases + 1))
  local python_source
  python_source="$(mktemp "${TMPDIR:-/tmp}/xy-tokenizer-tree-self-test.XXXXXX.py")"
  awk '/<<'"'"'PY'"'"'/{capture=1; next} capture && /^PY$/{exit} capture' \
    "$script_path" >"$python_source"
  if ! XY_TOKENIZER_MATERIALIZED_TREE_SELF_TEST=1 "${UV_CMD[@]}" "$python_source"; then
    echo "run-xy-tokenizer-inspection: self-test FAIL: materialized tree regression" >&2
    fail=1
  fi
  rm -f -- "$python_source"

  cases=$((cases + 1))
  if "$script_path" --self-test --work-dir /tmp/xy-tokenizer-self-test >/dev/null 2>&1; then
    echo "run-xy-tokenizer-inspection: self-test FAIL: extra argument accepted" >&2
    fail=1
  fi
  if "$script_path" --unknown-flag >/dev/null 2>&1; then
    echo "run-xy-tokenizer-inspection: self-test FAIL: unknown argument accepted" >&2
    fail=1
  fi
  if (( fail == 0 )); then
    echo "run-xy-tokenizer-inspection.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

work_dir="/dev/shm/vokra-xy-tokenizer-inspection"
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
  [[ "$work_dir" == "/dev/shm/vokra-xy-tokenizer-inspection" ]] \
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
  || die "tmpfs free space is below the 30-GiB guard"
for command in cargo rustc rustfmt uv git sha256sum findmnt; do
  command -v "$command" >/dev/null 2>&1 || die "required VAST tool is missing: $command"
done

mkdir -p "$work_dir"
work_dir="$(cd "$work_dir" && pwd)"
log_path="$work_dir/inspection.log"
cache_dir="$work_dir/hf-cache"
snapshot_path_file="$work_dir/hf-snapshot-path.txt"
server_packet="$work_dir/server-packet.json"
assets_dir="$work_dir/assets"
source_dir="$work_dir/official-source"
prepared_dir="$work_dir/prepared"
evidence_dir="$work_dir/evidence"
mkdir -p "$cache_dir" "$assets_dir" "$prepared_dir/config" "$evidence_dir"

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

run_logged "${UV_CMD[@]}" - "$UPSTREAM_REPOSITORY" "$UPSTREAM_REVISION" "$cache_dir" "$snapshot_path_file" "$server_packet" <<'PY'
import os
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, RepoFolder, snapshot_download

selected = {".gitattributes", "README.md", "xy_tokenizer.ckpt"}


def validate_materialized_tree(root: Path, expected: set[str]) -> None:
    if root.is_symlink() or not root.is_dir():
        raise SystemExit(f"materialized root is not a regular directory: {root}")
    actual = set()
    for candidate in sorted(root.rglob("*")):
        rel = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            raise SystemExit(f"materialized payload is symlinked: {rel}")
        if rel in {".cache", ".cache/huggingface"} or rel.startswith(".cache/huggingface/"):
            continue
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise SystemExit(f"materialized payload is not a regular file: {rel}")
        actual.add(rel)
    if actual != expected:
        raise SystemExit(f"materialized selected tree drift: {sorted(actual)!r}")


if os.environ.get("XY_TOKENIZER_MATERIALIZED_TREE_SELF_TEST") == "1":
    expected = {".gitattributes", "README.md", "xy_tokenizer.ckpt"}
    with tempfile.TemporaryDirectory(prefix="xy-tokenizer-materialized-tree-") as temp_dir:
        root = Path(temp_dir) / "model-materialized"
        (root / "config").mkdir(parents=True)
        for relative in expected:
            (root / relative).write_bytes(b"fixture")
        validate_materialized_tree(root, expected)

        def rejects(label: str, setup) -> None:
            setup()
            try:
                validate_materialized_tree(root, expected)
            except SystemExit:
                return
            raise AssertionError(f"tree self-test accepted {label}")

        rejects("symlink", lambda: (root / "config/link").symlink_to(root / "config/xy_tokenizer_config.yaml"))
        (root / "config/link").unlink()
        rejects("extra file", lambda: (root / "extra").write_bytes(b"extra"))
        (root / "extra").unlink()
        try:
            (root / "fifo").mkfifo()
        except (AttributeError, NotImplementedError):
            pass
        else:
            rejects("FIFO", lambda: None)
            (root / "fifo").unlink()
        outside = Path(temp_dir) / "outside"
        outside.mkdir()
        rejects("path escape", lambda: (root / "escape").symlink_to(outside, target_is_directory=True))
    print("xy-tokenizer materialized tree self-test: PASS")
    raise SystemExit(0)

repo, revision, cache_dir, output, packet_output = sys.argv[1:]
api = HfApi()
info = api.model_info(repo_id=repo, revision=revision)
if info.sha != revision:
    raise SystemExit(f"HF revision drift: {info.sha!r} != {revision!r}")
rows = {}
for item in api.list_repo_tree(repo_id=repo, revision=revision, recursive=True, expand=True):
    if isinstance(item, RepoFolder):
        continue
    if not isinstance(item, RepoFile) or not isinstance(item.path, str) or not item.path or "\\" in item.path or "\x00" in item.path or Path(item.path).is_absolute() or ".." in Path(item.path).parts or item.path in rows:
        raise SystemExit(f"invalid/duplicate HF tree entry: {item!r}")
    blob = getattr(item, "blob_id", None)
    size = getattr(item, "size", None)
    if not isinstance(blob, str) or len(blob) != 40 or any(c not in "0123456789abcdef" for c in blob) or isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise SystemExit(f"incomplete HF server identity: {item.path}")
    lfs = getattr(item, "lfs", None)
    if lfs is None:
        rows[item.path] = {"path": item.path, "type": "file", "size": size, "git_blob_sha1": blob}
        continue
    oid = (lfs.get("sha256") or lfs.get("oid", "")) if isinstance(lfs, dict) else (getattr(lfs, "sha256", None) or getattr(lfs, "oid", ""))
    oid = oid.removeprefix("sha256:")
    lfs_size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
    if len(oid) != 64 or any(c not in "0123456789abcdef" for c in oid) or isinstance(lfs_size, bool) or not isinstance(lfs_size, int) or lfs_size != size:
        raise SystemExit(f"incomplete HF LFS identity: {item.path}")
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {lfs_size}\n".encode()
    pointer_sha = hashlib.sha1(f"blob {len(pointer)}\0".encode() + pointer).hexdigest()
    if pointer_sha != blob:
        raise SystemExit(f"HF LFS pointer identity mismatch: {item.path}")
    rows[item.path] = {"path": item.path, "type": "file", "size": size, "git_blob_sha1": pointer_sha, "lfs_sha256": oid, "lfs_size": lfs_size, "lfs_pointer_sha1": pointer_sha}
if set(rows) != selected:
    raise SystemExit(f"selected HF file set mismatch: {sorted(rows)!r}")
Path(packet_output).write_text(json.dumps({"repository": repo, "requested_revision": revision, "resolved_revision": info.sha, "files": [rows[path] for path in sorted(rows)]}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
materialized = Path(output).with_name("model-materialized")
resolved = Path(snapshot_download(
    repo_id=repo,
    revision=revision,
    cache_dir=cache_dir,
    local_dir=materialized,
    allow_patterns=sorted(selected),
    token=os.environ.get("HF_TOKEN") or os.environ.get("HF"),
))
if resolved.resolve() != materialized.resolve():
    raise SystemExit("HF local_dir escaped work directory")
validate_materialized_tree(resolved, selected)
Path(output).write_text(str(resolved) + "\n", encoding="utf-8")
PY
snapshot_path="$(< "$snapshot_path_file")"
for relative in .gitattributes README.md "$CHECKPOINT_FILENAME"; do
  mkdir -p "$assets_dir/$(dirname "$relative")"
  cp -- "$snapshot_path/$relative" "$assets_dir/$relative"
done
actual_checkpoint_bytes="$(stat -c '%s' "$assets_dir/$CHECKPOINT_FILENAME")"
actual_checkpoint_sha="$(sha256sum "$assets_dir/$CHECKPOINT_FILENAME" | awk '{print $1}')"
[[ "$actual_checkpoint_bytes" == "$CHECKPOINT_BYTES" && "$actual_checkpoint_sha" == "$CHECKPOINT_SHA256" ]] \
  || die "checkpoint identity mismatch"

run_logged git clone --no-tags --filter=blob:none "$SOURCE_REPOSITORY" "$source_dir"
run_logged git -C "$source_dir" checkout --detach "$SOURCE_REVISION"
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$SOURCE_REVISION" ]] \
  || die "official source revision mismatch"
[[ "$(git -C "$source_dir" remote get-url origin | sed 's#/$##; s#\.git$##')" == "$SOURCE_REPOSITORY" ]] \
  || die "official source remote mismatch"
[[ -f "$source_dir/$CONFIG_RELATIVE" ]] || die "official config is missing"
config_sha="$(sha256sum "$source_dir/$CONFIG_RELATIVE" | awk '{print $1}')"
[[ "$config_sha" == "$CONFIG_SHA256" ]] || die "official config identity mismatch"
source_config="$source_dir/$CONFIG_RELATIVE"
prepared_config="$prepared_dir/$CONFIG_RELATIVE"
mkdir -p "$(dirname "$assets_dir/$CONFIG_RELATIVE")" "$(dirname "$prepared_config")"
cp -- "$source_config" "$assets_dir/$CONFIG_RELATIVE"
cp -- "$source_config" "$prepared_config"

set +e
run_logged "${UV_CMD[@]}" "$INSPECTOR" \
  --checkpoint "$assets_dir/$CHECKPOINT_FILENAME" \
  --config "$prepared_config" --source "$source_dir" \
  --server-packet "$server_packet" --prepared "$prepared_dir/model.safetensors" --output "$evidence_dir"
inspect_rc=$?
set -e
[[ "$inspect_rc" == "2" ]] || die "inspector must remain fail-closed with exit 2: $inspect_rc"
cp "$evidence_dir/manifest.json" "$prepared_dir/xy_tokenizer_prepared_manifest.json"
set +e
run_logged "${UV_CMD[@]}" - "$evidence_dir/manifest.json" <<'PY'
import json
import sys
manifest = json.loads(open(sys.argv[1], encoding="utf-8").read())
if manifest.get("status") != "BLOCKED":
    raise SystemExit("inspection manifest is not BLOCKED")
if manifest.get("inspection_status") != "AUTHENTICATED_EVIDENCE_COMPLETE" or manifest.get("collection_status") != "AUTHENTICATED":
    raise SystemExit("inspection evidence is not authenticated")
required = {"evidence_stage": "INSPECTION_ONLY", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD"}
for key, expected in required.items():
    if manifest.get(key) != expected:
        raise SystemExit(f"manifest contract mismatch: {key}={manifest.get(key)!r}")
PY
manifest_rc=$?
set -e
if (( manifest_rc != 0 )); then
  echo "run-xy-tokenizer-inspection: inspection remained fail-closed; no prepared conversion allowed" >&2
  exit 2
fi
[[ -s "$prepared_dir/model.safetensors" ]] || die "prepared safetensors is empty"
set +e
"$repo_root/target/release/vokra-cli" convert \
  --model xy-tokenizer --input "$prepared_dir/model.safetensors" \
  --output "$work_dir/should-not-be-created.gguf" --license apache-2.0 \
  >"$evidence_dir/converter-refusal.log" 2>&1
converter_status=$?
set -e
(( converter_status != 0 )) || die "converter unexpectedly accepted inspection-only XY artifact"
grep -Fq "INSPECTION_ONLY" "$evidence_dir/converter-refusal.log" \
  || die "converter refusal did not identify INSPECTION_ONLY status"
[[ ! -e "$work_dir/should-not-be-created.gguf" ]] \
  || die "inspection-only converter created an output artifact"

{
  echo "verdict=INSPECTION_ONLY"
  echo "runtime_parity=NOT_RUN"
  echo "numerical_parity=NOT_RUN"
  echo "upstream_revision=$UPSTREAM_REVISION"
  echo "checkpoint_bytes=$actual_checkpoint_bytes"
  echo "checkpoint_sha256=$actual_checkpoint_sha"
  echo "source_revision=$SOURCE_REVISION"
  echo "config_sha256=$config_sha"
  echo "prepared_bytes=$(stat -c '%s' "$prepared_dir/model.safetensors")"
  echo "prepared_sha256=$(sha256sum "$prepared_dir/model.safetensors" | awk '{print $1}')"
  echo "prepared_manifest_sha256=$(sha256sum "$prepared_dir/xy_tokenizer_prepared_manifest.json" | awk '{print $1}')"
  echo "converter_verdict=INSPECTION_ONLY_REFUSAL"
  echo "tensor_inventory_sha256=$(sha256sum "$evidence_dir/tensor-inventory.json" | awk '{print $1}')"
  echo "source_inventory_sha256=$(sha256sum "$evidence_dir/source-inventory.json" | awk '{print $1}')"
} | tee "$evidence_dir/summary.txt"
echo "XY-Tokenizer inspection evidence: $evidence_dir"
exit 2
