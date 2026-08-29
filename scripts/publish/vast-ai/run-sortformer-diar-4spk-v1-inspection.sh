#!/usr/bin/env bash
# VAST-only fixed-HF Sortformer inspection. No conversion, parity, or upload.
set -euo pipefail
ROOT="${VOKRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
INSPECTOR="$ROOT/tools/parity/sortformer_diar_4spk_v1_inspect.py"
REPO="nvidia/diar_sortformer_4spk-v1"
HF_REVISION="9f17b10df44c0a4c8f3c86fbddc9ee2d6ab9ac08"
SOURCE_URL="https://github.com/NVIDIA/NeMo.git"
SOURCE_REVISION="505acacf6444a67ff9a4020fb03a5e6d59953e05"
MODEL_BYTES=494206256
MODEL_SHA256="e8abcc5f3a82ff23134c98a37f70fef3f159611f394bb191a0ad0a6f4b052974"
WORK="/workspace/vokra-sortformer-diar-4spk-v1-inspection"
MIN_MEM_KIB=$((64 * 1024 * 1024))
MIN_DISK_KIB=$((150 * 1024 * 1024))
die() { echo "sortformer inspection: $*" >&2; exit 2; }
usage() { echo "usage: run-sortformer-diar-4spk-v1-inspection.sh [--work-dir DIR] | --self-test"; }
self_test() {
  local fail=0 token
  for token in "$REPO" "$HF_REVISION" "$SOURCE_URL" "$SOURCE_REVISION" "$MODEL_BYTES" "$MODEL_SHA256" "493434880" "bc74dfd8ca314240abcdc7e2949901eeaa72947a04ce1fab893e373d81f1e689" x86_64 MIN_MEM_KIB MIN_DISK_KIB /proc/meminfo 'df -Pk' VOKRA_PUBLISH_ON_VAST CARGO_BUILD_JOBS snapshot_download HfApi requested_revision lfs_pointer_git_blob_sha1 config.json processor_config.json model.safetensors diar_sortformer_4spk-v1.nemo sortformer_diar_4spk_v1_inspect.py weights_only=True 'status=BLOCKED' 'evidence_stage=INSPECTION_ONLY' INSPECTION_ONLY REFERENCE_SOURCE_SELECTED WEIGHT_BUILD_PROVENANCE_BLOCKED_MUTABLE_MAIN NO_UPLOAD sha256sum --server-tree; do
    grep -Fq -- "$token" "${BASH_SOURCE[0]}" || { echo "missing contract: $token" >&2; fail=1; }
  done
  if grep -En '^[[:space:]]*(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh)' "${BASH_SOURCE[0]}" >/dev/null; then
    echo 'publication command found' >&2
    fail=1
  fi
  UV_CACHE_DIR="${SORTFORMER_UV_CACHE_DIR:-/tmp/vokra-sortformer-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --self-test >/dev/null || fail=1
  if ! UV_CACHE_DIR="${SORTFORMER_UV_CACHE_DIR:-/tmp/vokra-sortformer-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - <<'PY'
from huggingface_hub import RepoFile, RepoFolder

def classify(entry):
    if isinstance(entry, RepoFolder):
        return "directory"
    if isinstance(entry, RepoFile):
        return "file"
    raise RuntimeError(f"unexpected entry: {entry!r}")

file_entry = RepoFile(path=".gitattributes", size=1584, oid="a" * 40)
file_entry.type = None
assert classify(file_entry) == "file"
assert classify(RepoFolder(path="nested", oid="b" * 40)) == "directory"
try:
    classify(object())
except RuntimeError:
    pass
else:
    raise AssertionError("unexpected server-tree entry was accepted")
PY
  then
    echo 'hermetic RepoFile/RepoFolder regression failed' >&2
    fail=1
  fi
  (( fail == 0 )) || return 1
  echo 'run-sortformer-diar-4spk-v1-inspection.sh self-test: OK'
}
self=0
while (($#)); do
  case "$1" in
    --self-test) self=1; shift ;;
    --work-dir) (($# >= 2)) || die '--work-dir requires DIR'; WORK="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  [[ "$WORK" == /workspace/vokra-sortformer-diar-4spk-v1-inspection ]] || die '--self-test accepts no other arguments'
  self_test
  exit 0
fi
[[ "$(uname -s)" == Linux ]] || die 'Linux/VAST required'
[[ "$(uname -m)" == x86_64 ]] || die 'x86_64 VAST host required'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 required'
[[ -f "$ROOT/Cargo.toml" && -d "$ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
[[ -f "$INSPECTOR" ]] || die 'inspector missing'
parent="$(dirname "$WORK")"
mkdir -p "$parent"
[[ ! -e "$WORK" || -z "$(find "$WORK" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work directory must be empty'
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
if ! [[ "$mem_kib" =~ ^[0-9]+$ ]] || (( mem_kib < MIN_MEM_KIB )); then die 'memory guard failed'; fi
free_kib="$(df -Pk "$parent" | awk 'NR == 2 {print $4}')"
if ! [[ "$free_kib" =~ ^[0-9]+$ ]] || (( free_kib < MIN_DISK_KIB )); then die 'disk guard failed'; fi
for tool in cargo git uv sha256sum awk find df; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done
mkdir -p "$WORK/evidence"
WORK="$(cd "$WORK" && pwd)"
export CARGO_BUILD_JOBS=1
export UV_CACHE_DIR="${SORTFORMER_UV_CACHE_DIR:-/tmp/vokra-sortformer-uv-cache}"
{
  echo "status=BLOCKED"
  echo "evidence_stage=INSPECTION_ONLY"
  echo "publication=NO_UPLOAD"
  echo "hf_revision=$HF_REVISION"
  echo "expected_model_bytes=$MODEL_BYTES"
  echo "expected_model_sha256=$MODEL_SHA256"
  cargo fmt --all -- --check
  cargo metadata --locked --no-deps --format-version 1 >/dev/null
} > "$WORK/evidence/validation.log" 2>&1
snapshot="$(uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$REPO" "$HF_REVISION" "$WORK/hf" "$WORK/hf-cache" 2>> "$WORK/evidence/validation.log" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, revision, local_dir, cache_dir = sys.argv[1:]
print(snapshot_download(repo_id=repo, revision=revision, cache_dir=cache_dir, local_dir=local_dir, allow_patterns=["*"]))
PY
)"
printf '%s\n' "snapshot_path=$snapshot" | tee -a "$WORK/evidence/validation.log" >/dev/null
# shellcheck disable=SC2129 # heredoc output is one validation stream
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$REPO" "$HF_REVISION" "$snapshot" "$WORK/server-tree.json" <<'PY' >> "$WORK/evidence/validation.log" 2>&1
import json, re, sys
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, RepoFolder
repo, revision, snapshot, output = sys.argv[1:]
api = HfApi(); info = api.model_info(repo, revision=revision)
if info.sha != revision: raise RuntimeError("resolved revision mismatch")
rows=[]
for entry in api.list_repo_tree(repo, revision=revision, recursive=True):
    if isinstance(entry, RepoFolder): continue
    if not isinstance(entry, RepoFile): raise RuntimeError(f"unexpected HF tree entry: {entry!r}")
    path = getattr(entry, "path", None); blob = getattr(entry, "blob_id", None) or getattr(entry, "oid", None); size = getattr(entry, "size", None)
    lfs = getattr(entry, "lfs", None); lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    local = Path(snapshot) / path if isinstance(path, str) else None
    if local is None or not local.is_file() or local.is_symlink() or not isinstance(size, int) or local.stat().st_size != size or not isinstance(blob, str) or not re.fullmatch(r"[0-9a-f]{40}", blob) or (lfs_sha is not None and not re.fullmatch(r"[0-9a-f]{64}", lfs_sha)):
        raise RuntimeError(f"incomplete or unmaterialized HF row: {path}")
    rows.append({"path": path, "type": "file", "size": size, "git_blob_sha1": blob if lfs_sha is None else None, "lfs_pointer_git_blob_sha1": blob if lfs_sha is not None else None, "lfs_sha256": lfs_sha})
if len({row["path"] for row in rows}) != len(rows): raise RuntimeError("duplicate HF tree path")
Path(output).write_text(json.dumps({"repository": repo, "requested_revision": revision, "resolved_revision": info.sha, "walk": "recursive_file_only", "files": sorted(rows, key=lambda row: row["path"])}, sort_keys=True, indent=2) + "\n")
PY
git clone --filter=blob:none --no-tags "$SOURCE_URL" "$WORK/source/repo" >> "$WORK/evidence/validation.log" 2>&1
git -C "$WORK/source/repo" checkout --detach "$SOURCE_REVISION" >> "$WORK/evidence/validation.log" 2>&1
set +e
uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --snapshot "$snapshot" --evidence "$WORK/evidence" --server-tree "$WORK/server-tree.json" --source "$WORK/source/repo" >> "$WORK/evidence/validation.log" 2>&1
rc=$?
set -e
[[ "$rc" == 2 ]] || die "expected inspection blocker exit=2, got $rc"
[[ -s "$WORK/evidence/manifest.json" ]] || die 'inspection manifest missing before blocker exit'
grep -Fq '"status": "BLOCKED"' "$WORK/evidence/manifest.json" || die 'manifest did not remain BLOCKED'
grep -Fq '"evidence_stage": "INSPECTION_ONLY"' "$WORK/evidence/manifest.json" || die 'evidence stage missing'
echo 'source_status=SEE_MANIFEST_AFTER_AUTHENTICATION' | tee -a "$WORK/evidence/validation.log"
echo 'weight_build_provenance=WEIGHT_BUILD_PROVENANCE_BLOCKED_MUTABLE_MAIN' | tee -a "$WORK/evidence/validation.log"
echo 'verdict=BLOCKED; evidence_stage=INSPECTION_ONLY; publication=NO_UPLOAD' | tee -a "$WORK/evidence/validation.log"
exit 2
