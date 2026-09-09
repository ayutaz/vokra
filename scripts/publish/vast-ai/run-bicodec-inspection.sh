#!/usr/bin/env bash
# VAST-only BiCodec inspection wave.
#
# This worker authenticates the Spark-TTS BiCodec safetensors/config and the
# pinned official source, inventories tensors through the independent
# safetensors oracle, and performs an inventory-only conversion dry run. It
# deliberately stops at INSPECTION_ONLY; native-reference evidence is produced
# by the separate run-bicodec-native-parity.sh worker.

set -euo pipefail

UPSTREAM_REPO="SparkAudio/Spark-TTS-0.5B"
UPSTREAM_REVISION="642071559bfc6346c2359d19dcb6be3f9dd8a05d"
SOURCE_REPOSITORY="https://github.com/SparkAudio/Spark-TTS"
SOURCE_REVISION="2f1ea9082400547242641f5271b6f941c9f439d1"
CHECKPOINT_RELATIVE="BiCodec/model.safetensors"
CONFIG_RELATIVE="BiCodec/config.yaml"
CHECKPOINT_BYTES=625518756
CHECKPOINT_SHA256="e9940cd48d4446e4340ced82d234bf5618350dd9f5db900ebe47a4fdb03867ec"
CONFIG_BYTES=1164
CONFIG_SHA256="744f4093ae2381a2eb44ea8c4a5268a8d1e581498e9bf0808c034d1b076429be"
INSPECTOR="tools/parity/bicodec_inspect_reference.py"
UV_CMD=(uv run --frozen --project tools/parity --python 3.12 python)
MIN_VAST_MEM_KIB=$((64 * 1024 * 1024))
MIN_FREE_DISK_KIB=$((20 * 1024 * 1024))

usage() {
  cat <<'EOF'
Usage:
  run-bicodec-inspection.sh [--work-dir <tmpfs-dir>]
  run-bicodec-inspection.sh --self-test

The real path is VAST-only and requires a clean checkout, Linux x86_64,
tmpfs work storage, 64 GiB RAM, and VOKRA_PUBLISH_ON_VAST=1.  It downloads
only the pinned Spark-TTS BiCodec snapshot, clones the pinned official
source, inventories all tensors/config topology, and converts to a local GGUF.
The evidence verdict is always INSPECTION_ONLY; no numerical decode is claimed.
EOF
}

die() {
  echo "run-bicodec-inspection: $*" >&2
  exit 1
}

run_self_test() {
  local script_path="${BASH_SOURCE[0]}" fail=0 cases=0 required
  local repo_root
  repo_root="$(cd "$(dirname "$script_path")/../../.." && pwd)"
  [[ -f "$script_path" ]] || die "worker is missing"
  [[ -f "$repo_root/$INSPECTOR" ]] \
    || die "inspection oracle is missing"

  cases=$((cases + 1))
  for required in \
    "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$SOURCE_REPOSITORY" \
    "$SOURCE_REVISION" "$CHECKPOINT_RELATIVE" "$CONFIG_RELATIVE" \
    "$CHECKPOINT_BYTES" "$CHECKPOINT_SHA256" "$CONFIG_BYTES" "$CONFIG_SHA256" \
    "$INSPECTOR" "snapshot_download" "HfApi" "RepoFile" "RepoFolder" "server-packet" \
    "collection_status" "AUTHENTICATED" "FAILED" "inspection_status" "ERROR" "BLOCKED" "INSPECTION_ONLY" "NO_UPLOAD" \
    "CHECKPOINT_SHA256" "CONFIG_SHA256" "sha256sum" "tmpfs" \
    "MIN_VAST_MEM_KIB" "MIN_FREE_DISK_KIB" "CARGO_BUILD_JOBS"; do
    if ! grep -Fq -- "$required" "$script_path" && ! grep -Fq -- "$required" "$repo_root/$INSPECTOR"; then
      echo "run-bicodec-inspection: self-test FAIL: missing contract: $required" >&2
      fail=1
    fi
  done

  cases=$((cases + 1))
  for required in \
    'uname -s' 'uname -m' 'VOKRA_PUBLISH_ON_VAST' \
    'git status --porcelain --untracked-files=all' 'findmnt' \
    'cargo fmt --all -- --check' 'cargo build --locked --release -p vokra-cli' \
    'uv run --frozen --project tools/parity --python 3.12' \
    'vokra-cli" convert --model bicodec' 'cc-by-nc-sa-4.0' \
    'model_info' 'list_repo_tree' 'local_dir' 'RepoFolder' 'selected.issubset(by_path)' 'exit 2'; do
    if ! grep -Fq -- "$required" "$script_path"; then
      echo "run-bicodec-inspection: self-test FAIL: missing VAST gate: $required" >&2
      fail=1
    fi
  done

  cases=$((cases + 1))
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh)([[:space:]]|$)' "$script_path" >/dev/null; then
    echo "run-bicodec-inspection: self-test FAIL: publication command found" >&2
    fail=1
  fi
  if grep -En '(^|[[:space:]])(python|python3|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    echo "run-bicodec-inspection: self-test FAIL: raw Python/pip invocation found" >&2
    fail=1
  fi
  if grep -En 'weights_only=False|torch\.load|pickle\.load' "$repo_root/$INSPECTOR" >/dev/null; then
    echo "run-bicodec-inspection: self-test FAIL: unsafe loader found" >&2
    fail=1
  fi

  cases=$((cases + 1))
  local python_source
  python_source="$(mktemp "${TMPDIR:-/tmp}/bicodec-tree-self-test.XXXXXX.py")"
  awk '/<<'"'"'PY'"'"'/{capture=1; next} capture && /^PY$/{exit} capture' \
    "$script_path" >"$python_source"
  if ! BICODEC_MATERIALIZED_TREE_SELF_TEST=1 "${UV_CMD[@]}" "$python_source"; then
    echo "run-bicodec-inspection: self-test FAIL: materialized tree regression" >&2
    fail=1
  fi
  rm -f -- "$python_source"

  cases=$((cases + 1))
  if "$script_path" --self-test --work-dir /tmp/bicodec-self-test >/dev/null 2>&1; then
    echo "run-bicodec-inspection: self-test FAIL: extra argument accepted" >&2
    fail=1
  fi
  if "$script_path" --unknown-flag >/dev/null 2>&1; then
    echo "run-bicodec-inspection: self-test FAIL: unknown argument accepted" >&2
    fail=1
  fi
  if (( fail == 0 )); then
    echo "run-bicodec-inspection.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

work_dir="/dev/shm/vokra-bicodec-inspection"
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
  [[ "$work_dir" == "/dev/shm/vokra-bicodec-inspection" ]] \
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
  || die "work-dir parent must be tmpfs/RAM-disk: $work_parent"
if [[ -e "$work_dir" ]]; then
  [[ -d "$work_dir" ]] || die "work-dir exists but is not a directory"
  [[ -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
    || die "work-dir must be absent or empty"
fi
[[ -r /proc/meminfo ]] || die "/proc/meminfo is unavailable"
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ && mem_kib -ge MIN_VAST_MEM_KIB ]] \
  || die "host RAM is below the 64-GiB guard"
free_kib="$(df -Pk "$work_parent" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ && free_kib -ge MIN_FREE_DISK_KIB ]] \
  || die "tmpfs free space is below the 20-GiB guard"
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
evidence_dir="$work_dir/evidence"
gguf_path="$work_dir/spark-tts-bicodec.gguf"
mkdir -p "$cache_dir" "$assets_dir/BiCodec" "$evidence_dir"

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

run_logged "${UV_CMD[@]}" - "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$cache_dir" "$snapshot_path_file" "$server_packet" <<'PY'
import os
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, RepoFolder, snapshot_download

selected = {".gitattributes", "README.md", "BiCodec/config.yaml", "BiCodec/model.safetensors"}


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


if os.environ.get("BICODEC_MATERIALIZED_TREE_SELF_TEST") == "1":
    expected = {"BiCodec/config.yaml", "BiCodec/model.safetensors"}
    with tempfile.TemporaryDirectory(prefix="bicodec-materialized-tree-") as temp_dir:
        root = Path(temp_dir) / "model-materialized"
        (root / "BiCodec").mkdir(parents=True)
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

        rejects("symlink", lambda: (root / "BiCodec/link").symlink_to(root / "BiCodec/config.yaml"))
        (root / "BiCodec/link").unlink()
        rejects("extra file", lambda: (root / "BiCodec/extra").write_bytes(b"extra"))
        (root / "BiCodec/extra").unlink()
        try:
            (root / "BiCodec/fifo").mkfifo()
        except (AttributeError, NotImplementedError):
            pass
        else:
            rejects("FIFO", lambda: None)
            (root / "BiCodec/fifo").unlink()
        outside = Path(temp_dir) / "outside"
        outside.mkdir()
        rejects("path escape", lambda: (root / "escape").symlink_to(outside, target_is_directory=True))
    print("bicodec materialized tree self-test: PASS")
    raise SystemExit(0)

repo, revision, cache_dir, output, packet_output = sys.argv[1:]
api = HfApi()
info = api.model_info(repo_id=repo, revision=revision)
if info.sha != revision:
    raise SystemExit(f"resolved revision drift: {info.sha!r} != {revision!r}")
rows = []
tree = list(api.list_repo_tree(repo_id=repo, revision=revision, recursive=True, expand=True))
by_path = {}
for item in tree:
    if isinstance(item, RepoFolder):
        continue
    if not isinstance(item, RepoFile):
        raise SystemExit(f"unexpected recursive server tree entry: {item!r}")
    path = getattr(item, "path", None)
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path or Path(path).is_absolute() or ".." in Path(path).parts or path in by_path:
        raise SystemExit(f"invalid/duplicate server tree file: {path!r}")
    by_path[path] = item
if not selected.issubset(by_path) or len(selected.intersection(by_path)) != len(selected):
    raise SystemExit(f"server tree selected files missing: {sorted(selected - set(by_path))!r}")
for relative in sorted(selected):
    item = by_path.get(relative)
    if item is None or not isinstance(item, RepoFile):
        raise SystemExit(f"canonical server file missing or not a file: {relative}")
    blob = getattr(item, "blob_id", None)
    size = getattr(item, "size", None)
    lfs = getattr(item, "lfs", None)
    if not isinstance(blob, str) or not isinstance(size, int) or size < 0:
        raise SystemExit(f"server identity incomplete: {relative}")
    if lfs is not None:
        oid = (lfs.get("sha256") or lfs.get("oid", "")) if isinstance(lfs, dict) else (getattr(lfs, "sha256", None) or getattr(lfs, "oid", ""))
        oid = oid.removeprefix("sha256:")
        lfs_size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
        if len(oid) != 64 or any(c not in "0123456789abcdef" for c in oid) or not isinstance(lfs_size, int):
            raise SystemExit(f"server LFS identity incomplete: {relative}")
        pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {lfs_size}\n".encode()
        pointer_sha = hashlib.sha1(f"blob {len(pointer)}\0".encode() + pointer).hexdigest()
        if blob != pointer_sha or lfs_size != size:
            raise SystemExit(f"server LFS pointer identity mismatch: {relative}")
        rows.append({"path": relative, "type": "file", "size": size, "git_blob_sha1": pointer_sha, "lfs_sha256": oid, "lfs_size": lfs_size, "lfs_pointer_sha1": pointer_sha})
    else:
        if not isinstance(blob, str) or len(blob) != 40:
            raise SystemExit(f"server Git identity incomplete: {relative}")
        rows.append({"path": relative, "type": "file", "size": size, "git_blob_sha1": blob})
Path(packet_output).write_text(json.dumps({"repository": repo, "requested_revision": revision, "resolved_revision": info.sha, "files": rows}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
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
    raise SystemExit(f"materialized snapshot escaped work directory: {resolved}")
validate_materialized_tree(resolved, selected)
Path(output).write_text(str(resolved) + "\n", encoding="utf-8")
PY
snapshot_path="$(< "$snapshot_path_file")"
for relative in .gitattributes README.md "$CHECKPOINT_RELATIVE" "$CONFIG_RELATIVE"; do
  mkdir -p "$assets_dir/$(dirname "$relative")"
  cp -- "$snapshot_path/$relative" "$assets_dir/$relative"
done
[[ "$(find "$assets_dir" -type f | wc -l | tr -d ' ')" == "4" ]] \
  || die "staged assets include unexpected files"
actual_checkpoint_bytes="$(stat -c '%s' "$assets_dir/$CHECKPOINT_RELATIVE")"
actual_checkpoint_sha="$(sha256sum "$assets_dir/$CHECKPOINT_RELATIVE" | awk '{print $1}')"
actual_config_bytes="$(stat -c '%s' "$assets_dir/$CONFIG_RELATIVE")"
actual_config_sha="$(sha256sum "$assets_dir/$CONFIG_RELATIVE" | awk '{print $1}')"
[[ "$actual_checkpoint_bytes" == "$CHECKPOINT_BYTES" && "$actual_checkpoint_sha" == "$CHECKPOINT_SHA256" ]] \
  || die "checkpoint identity mismatch"
[[ "$actual_config_bytes" == "$CONFIG_BYTES" && "$actual_config_sha" == "$CONFIG_SHA256" ]] \
  || die "config identity mismatch"

run_logged git clone --no-tags --filter=blob:none "$SOURCE_REPOSITORY" "$source_dir"
run_logged git -C "$source_dir" checkout --detach "$SOURCE_REVISION"
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$SOURCE_REVISION" ]] \
  || die "official source revision mismatch"
[[ "$(git -C "$source_dir" remote get-url origin)" == "$SOURCE_REPOSITORY" ]] \
  || die "official source remote mismatch"

set +e
run_logged "${UV_CMD[@]}" "$INSPECTOR" \
  --model-dir "$assets_dir" --source-dir "$source_dir" \
  --server-packet "$server_packet" --output "$evidence_dir"
inspect_rc=$?
set -e
[[ "$inspect_rc" == "2" ]] || die "inspector did not return its fail-closed status 2: $inspect_rc"
grep -Fq '"status": "BLOCKED"' "$evidence_dir/manifest.json" \
  || die "inspection manifest was not BLOCKED"
run_logged "${UV_CMD[@]}" - "$evidence_dir/manifest.json" <<'PY'
import json
import sys
manifest = json.loads(open(sys.argv[1], encoding="utf-8").read())
required = {
    "status": "BLOCKED",
    "evidence_stage": "INSPECTION_ONLY",
    "collection_status": "AUTHENTICATED",
    "inspection_status": "COMPLETE",
    "runtime_status": "NOT_IMPLEMENTED",
    "cpu_status": "UNSUPPORTED",
    "metal_status": "BLOCKED_BY_CPU",
    "numerical_status": "NOT_RUN",
    "publication": "NO_UPLOAD",
}
for key, expected in required.items():
    if manifest.get(key) != expected:
        raise SystemExit(f"inspection manifest contract mismatch: {key}={manifest.get(key)!r}")
if manifest.get("collection_status") in {"FAILED", "ERROR"} or manifest.get("inspection_status") == "ERROR":
    raise SystemExit("inspection error was treated as authenticated collection")
PY
run_logged "$repo_root/target/release/vokra-cli" convert \
  --model bicodec --input "$assets_dir/$CHECKPOINT_RELATIVE" \
  --output "$gguf_path" --license cc-by-nc-sa-4.0
[[ -s "$gguf_path" ]] || die "converter produced no GGUF"

{
  echo "verdict=INSPECTION_ONLY"
  echo "status=BLOCKED"
  echo "collection_status=AUTHENTICATED"
  echo "publication=NO_UPLOAD"
  echo "runtime_parity=NOT_RUN"
  echo "numerical_parity=NOT_RUN"
  echo "source_revision=$SOURCE_REVISION"
  echo "upstream_revision=$UPSTREAM_REVISION"
  echo "checkpoint_sha256=$actual_checkpoint_sha"
  echo "config_sha256=$actual_config_sha"
  echo "gguf_sha256=$(sha256sum "$gguf_path" | awk '{print $1}')"
  echo "tensor_manifest_sha256=$(sha256sum "$evidence_dir/tensor-inventory.json" | awk '{print $1}')"
  echo "config_manifest_sha256=$(sha256sum "$evidence_dir/config.json" | awk '{print $1}')"
  echo "source_manifest_sha256=$(sha256sum "$evidence_dir/source-inventory.json" | awk '{print $1}')"
  echo "inspection_manifest_sha256=$(sha256sum "$evidence_dir/manifest.json" | awk '{print $1}')"
} | tee "$evidence_dir/summary.txt"
echo "BiCodec inspection evidence: $evidence_dir"
exit 2
