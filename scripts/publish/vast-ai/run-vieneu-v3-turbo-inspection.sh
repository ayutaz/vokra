#!/usr/bin/env bash
# VAST/Linux-only evidence collection for the pinned VieNeu-TTS-v3-Turbo
# bundle.  This worker never converts, uploads, publishes, or executes ONNX.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
INSPECTOR="$PARITY_PROJECT/vieneu_v3_turbo_inspect.py"
MODEL_REPOSITORY="pnnbao-ump/VieNeu-TTS-v3-Turbo"
MODEL_REVISION="2da0efab622a1722125991736524f080b751ef5b"
SOURCE_URL="https://github.com/pnnbao97/VieNeu-TTS.git"
SOURCE_TAG_OBJECT="1bc18895b8c6c6f8c927272d36c9b0befc127029"
SOURCE_TAG_NAME="v3.0.0"
SOURCE_PEELED_COMMIT="28392eee571db0da31632882ac7226faa2d09d5d"
SOURCE_REVISION="$SOURCE_TAG_OBJECT"
MOSS_REPOSITORY="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
MOSS_REVISION="6aa02b01e445cc585582cf0ba480bc3ea6c8dd68"
MIN_VAST_MEM_KIB=$((32 * 1024 * 1024))
MIN_FREE_DISK_KIB=$((8 * 1024 * 1024))
VIENEU_UV_CACHE_DIR="${VIENEU_UV_CACHE_DIR:-/tmp/vokra-vieneu-uv-cache}"

log() { printf '[vieneu-v3-turbo-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }

usage() {
  cat <<'EOF'
usage: run-vieneu-v3-turbo-inspection.sh [--work-dir <empty-dir>]
       run-vieneu-v3-turbo-inspection.sh --self-test

Downloads only immutable model/source snapshots and writes an INSPECTION_ONLY
manifest. No ONNX execution, conversion, upload, publication, or parity gate
is performed. Any manifest blocker propagates as exit 2.
EOF
}

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token
  for token in \
    'Linux' 'x86_64' 'VOKRA_PUBLISH_ON_VAST=1' '/proc/meminfo' 'df -Pk' \
    'CARGO_BUILD_JOBS=1' 'cargo fmt --all -- --check' \
    'cargo metadata --no-deps --format-version 1' 'pnnbao-ump/VieNeu-TTS-v3-Turbo' \
    "$MODEL_REVISION" 'pnnbao97/VieNeu-TTS.git' "$SOURCE_REVISION" "$SOURCE_TAG_NAME" "$SOURCE_PEELED_COMMIT" \
    'OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano' "$MOSS_REVISION" \
    'vieneu_v3_turbo_inspect.py' 'safetensors' 'onnx' 'load_external_data=False' \
    'RepoFile' 'RepoFolder' 'resolved_revision' 'git_blob_sha1' 'lfs_sha256' \
    'lfs_pointer_sha1' 'INSPECTION_ONLY' 'NO_UPLOAD' 'blockers' 'exit 2' 'snapshot_download' \
    'model_tree.json' 'moss_tree.json' '--model-tree' '--moss-tree' \
    'HF server/local tree mismatch' 'DECLARED_UNVERIFIED' 'UNKNOWN' \
    'AUTHENTICATED_EVIDENCE_COMPLETE' 'INSPECTION_ERROR' 'FAILED' \
    'collection_status' 'object_pairs_hook' 'git cat-file -t' 'git cat-file -p' \
    'pinned_tag_object' 'pinned_tag_name' 'pinned_peeled_commit' 'resolved_tag_object' \
    'resolved_peeled_commit' 'annotated tag identity' \
    'external data path traversal' 'dependency_license_status' \
    'SOURCE_ROLE_BLOBS_UNREVIEWED_BLOCKER' 'TOPOLOGY_CONTRACT_UNVERIFIED_BLOCKER' \
    'SOURCE_ROLE_BLOBS' 'AUTHENTICATED_APACHE_2' 'license_status' \
    'optional_source_roles' 'UNVERIFIED_TOPOLOGY' \
    'src/vieneu/_v3_turbo_engine/inference_v3_turbo.py' 'tracked_files' \
    'git status --porcelain'; do
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
  if ! UV_CACHE_DIR="$VIENEU_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" \
    --python 3.12 python "$INSPECTOR" --self-test >/dev/null; then
    log 'self-test FAIL: inspector self-test failed'
    fail=1
  fi
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

work_dir="/workspace/vokra-vieneu-v3-turbo-inspection"
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
  [[ "$work_dir" == "/workspace/vokra-vieneu-v3-turbo-inspection" ]] || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi

[[ "$(uname -s)" == Linux ]] || die 'VieNeu model work is VAST/Linux-only'
[[ "$(uname -m)" == x86_64 ]] || die 'VAST host must be x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
[[ -f "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" ]] || die 'locked parity project is missing'
[[ -f "$INSPECTOR" ]] || die 'VieNeu inspector is missing'
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'VAST memory value is invalid'
(( mem_kib >= MIN_VAST_MEM_KIB )) || die 'VAST memory guard failed'
mkdir -p "$(dirname "$work_dir")"
[[ ! -e "$work_dir" || -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work-dir must be empty'
free_kib="$(df -Pk "$(dirname "$work_dir")" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || die 'VAST disk value is invalid'
(( free_kib >= MIN_FREE_DISK_KIB )) || die 'VAST disk guard failed'
for tool in cargo git uv awk find df; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done

mkdir -p "$work_dir/model" "$work_dir/source" "$work_dir/moss" "$work_dir/evidence"
work_dir="$(cd "$work_dir" && pwd)"
export CARGO_BUILD_JOBS=1
export UV_CACHE_DIR="$VIENEU_UV_CACHE_DIR"
{
  echo "model_repository=$MODEL_REPOSITORY"
  echo "model_revision=$MODEL_REVISION"
  echo "source_url=$SOURCE_URL"
  echo "source_tag_object=$SOURCE_TAG_OBJECT"
  echo "source_tag_name=$SOURCE_TAG_NAME"
  echo "source_peeled_commit=$SOURCE_PEELED_COMMIT"
  echo "moss_repository=$MOSS_REPOSITORY"
  echo "moss_revision=$MOSS_REVISION"
  echo 'runtime_status=NOT_IMPLEMENTED_FAIL_CLOSED'
  echo 'parity_status=NOT_RUN'
  echo 'publication=NO_UPLOAD'
  cargo fmt --all -- --check
  cargo metadata --no-deps --format-version 1
} > "$work_dir/evidence/validation.log" 2>&1

# Resolve and retain complete server-side file trees before downloading
# snapshots. The packet is produced only from HfApi model_info/list_repo_tree;
# local path/size observations can never authenticate a mutable revision.
# shellcheck disable=SC2129 # heredoc command output is intentionally one evidence stream
{
UV_CACHE_DIR="$VIENEU_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 \
  python - "$work_dir/model_tree.json" "$work_dir/moss_tree.json" "$MODEL_REPOSITORY" "$MODEL_REVISION" "$MOSS_REPOSITORY" "$MOSS_REVISION" <<'PY'
import json
import hashlib
import sys
from pathlib import Path

from huggingface_hub import HfApi, RepoFile, RepoFolder


def safe_path(path):
    candidate = Path(path)
    if not isinstance(path, str) or not path or candidate.is_absolute() or "\\" in path or "\x00" in path or ".." in candidate.parts:
        raise RuntimeError(f"unsafe/invalid server path: {path!r}")
    return path


def field(item, name):
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def capture(api, repository, revision, output):
    info = api.model_info(repository, revision=revision)
    if info.sha != revision:
        raise RuntimeError(f"{repository} resolved {info.sha!r}, expected {revision!r}")
    rows = {}
    for item in api.list_repo_tree(repository, revision=revision, recursive=True, expand=True):
        if isinstance(item, RepoFolder) or field(item, "type") == "directory":
            continue
        if not isinstance(item, RepoFile) and field(item, "type") != "file":
            raise RuntimeError(f"unknown HF tree entry: {item!r}")
        path = safe_path(field(item, "path"))
        if path in rows:
            raise RuntimeError(f"duplicate HF tree path: {path}")
        size = field(item, "size")
        blob = field(item, "oid") or field(item, "blob_id")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0 or not isinstance(blob, str) or len(blob) != 40 or any(c not in "0123456789abcdef" for c in blob):
            raise RuntimeError(f"incomplete HF identity: {path}")
        lfs = field(item, "lfs")
        if lfs is None:
            rows[path] = {"path": path, "type": "file", "size": size, "git_blob_sha1": blob}
            continue
        oid = field(lfs, "sha256") or field(lfs, "oid")
        if isinstance(oid, str):
            oid = oid.removeprefix("sha256:")
        lfs_size = field(lfs, "size")
        if not isinstance(oid, str) or len(oid) != 64 or any(c not in "0123456789abcdef" for c in oid) or isinstance(lfs_size, bool) or not isinstance(lfs_size, int) or lfs_size != size:
            raise RuntimeError(f"incomplete HF LFS identity: {path}")
        pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {lfs_size}\n".encode("ascii")
        pointer_sha = hashlib.sha1(f"blob {len(pointer)}\0".encode("ascii") + pointer).hexdigest()
        if blob != pointer_sha:
            raise RuntimeError(f"HF LFS canonical pointer mismatch: {path}")
        rows[path] = {"path": path, "type": "file", "size": size, "git_blob_sha1": blob, "lfs_sha256": oid, "lfs_size": lfs_size, "lfs_pointer_sha1": pointer_sha}
    if not rows:
        raise RuntimeError(f"empty HF tree: {repository}")
    packet = {"repository": repository, "requested_revision": revision, "resolved_revision": info.sha, "files": [rows[path] for path in sorted(rows)]}
    Path(output).write_text(json.dumps(packet, sort_keys=True, indent=2) + "\n", encoding="utf-8")


api = HfApi()
capture(api, sys.argv[3], sys.argv[4], sys.argv[1])
capture(api, sys.argv[5], sys.argv[6], sys.argv[2])
PY
} >> "$work_dir/evidence/validation.log" 2>&1

UV_CACHE_DIR="$VIENEU_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 \
  python - "$MODEL_REPOSITORY" "$MODEL_REVISION" "$work_dir/model" "$MOSS_REPOSITORY" "$MOSS_REVISION" "$work_dir/moss" <<'PY' \
  >> "$work_dir/evidence/validation.log" 2>&1
import sys
from huggingface_hub import snapshot_download

snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], allow_patterns=["*"])
snapshot_download(repo_id=sys.argv[4], revision=sys.argv[5], local_dir=sys.argv[6], allow_patterns=["*"])
PY

git clone --filter=blob:none --no-checkout "$SOURCE_URL" "$work_dir/source/repo" \
  >> "$work_dir/evidence/validation.log" 2>&1
tag_object="$(git -C "$work_dir/source/repo" rev-parse "${SOURCE_TAG_NAME}^{tag}")"
[[ "$tag_object" == "$SOURCE_TAG_OBJECT" ]] || die 'VieNeu annotated tag object mismatch'
tag_type="$(git -C "$work_dir/source/repo" cat-file -t "$tag_object")"
[[ "$tag_type" == tag ]] || die 'VieNeu source object is not an annotated tag'
tag_content="$(git -C "$work_dir/source/repo" cat-file -p "$tag_object")"
grep -Fqx "object $SOURCE_PEELED_COMMIT" <<<"$tag_content" || die 'VieNeu annotated tag target mismatch'
grep -Fqx 'type commit' <<<"$tag_content" || die 'VieNeu annotated tag target type mismatch'
grep -Fqx "tag $SOURCE_TAG_NAME" <<<"$tag_content" || die 'VieNeu annotated tag name mismatch'
peeled_commit="$(git -C "$work_dir/source/repo" rev-parse "${SOURCE_TAG_NAME}^{commit}")"
[[ "$peeled_commit" == "$SOURCE_PEELED_COMMIT" ]] || die 'VieNeu annotated tag peeling mismatch'
git -C "$work_dir/source/repo" checkout --detach "$SOURCE_PEELED_COMMIT" \
  >> "$work_dir/evidence/validation.log" 2>&1
[[ "$(git -C "$work_dir/source/repo" rev-parse HEAD)" == "$SOURCE_PEELED_COMMIT" ]] || die 'VieNeu source HEAD mismatch'

set +e
UV_CACHE_DIR="$VIENEU_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python \
  "$INSPECTOR" --model-dir "$work_dir/model" --source-dir "$work_dir/source/repo" \
  --moss-dir "$work_dir/moss" --evidence-dir "$work_dir/evidence" \
  --model-tree "$work_dir/model_tree.json" --moss-tree "$work_dir/moss_tree.json" \
  >> "$work_dir/evidence/validation.log" 2>&1
inspect_rc=$?
set -e
[[ "$inspect_rc" == 2 ]] || die "inspector must remain fail-closed with exit 2: $inspect_rc"
manifest="$work_dir/evidence/vieneu_v3_turbo_manifest.json"
[[ -s "$manifest" ]] || die 'inspection manifest is missing'
UV_CACHE_DIR="$VIENEU_UV_CACHE_DIR" uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python - "$manifest" <<'PY'
import json
import sys


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle, object_pairs_hook=unique)
required = {
    "status": "BLOCKED",
    "evidence_stage": "INSPECTION_ONLY",
    "inspection_status": "AUTHENTICATED_EVIDENCE_COMPLETE",
    "collection_status": "AUTHENTICATED",
    "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
    "cpu_status": "UNSUPPORTED",
    "metal_status": "BLOCKED_BY_CPU",
    "parity_status": "NOT_RUN",
    "publication": "NO_UPLOAD",
    "exit_code": 2,
}
if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in required.items()):
    raise SystemExit("inspection manifest is not the authenticated normal blocked schema")
if not isinstance(manifest.get("blockers"), list) or not manifest["blockers"]:
    raise SystemExit("inspection manifest lost explicit blockers")
if "error" in manifest or manifest.get("inspection_status") in {"INSPECTION_ERROR", "FAILED", "UNVERIFIED"}:
    raise SystemExit("error/unverified evidence cannot be promoted")
PY
{
  echo 'runtime_status=NOT_IMPLEMENTED_FAIL_CLOSED'
  echo 'parity_status=NOT_RUN'
  echo 'verdict=BLOCKED_INSPECTION_ONLY'
  echo 'blocker_exit=2'
  echo 'native_blocker=native runtime and numerical parity remain unaudited'
} | tee -a "$work_dir/evidence/validation.log"
log "inspection complete: evidence=$work_dir; no conversion or upload performed"
exit 2
