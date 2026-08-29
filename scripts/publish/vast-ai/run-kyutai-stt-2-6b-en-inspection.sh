#!/usr/bin/env bash
# VAST/Linux-only Kyutai STT composite inspection.  This worker never
# converts, loads a runtime model, publishes, or uploads an artifact.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
INSPECTOR="$ROOT/tools/parity/kyutai_stt_2_6b_en_inspect.py"
REPOSITORY="kyutai/stt-2.6b-en"
HF_REVISION="a07aec56d22be5589cd0bc8709c75b6cf3e3039d"
SOURCE_URL="https://github.com/kyutai-labs/delayed-streams-modeling.git"
SOURCE_REVISION="4c4f65e147df056adf3346290d64c7b9649b18c9"
MOSHI_URL="https://github.com/kyutai-labs/moshi.git"
MOSHI_REVISION="e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
TOTAL_BYTES=5618985925
WORK="/dev/shm/vokra-kyutai-stt-2-6b-en-inspection"
MIN_MEM_KIB=$((128 * 1024 * 1024))
MIN_TMPFS_KIB=$((32 * 1024 * 1024))
UV_CACHE_DIR="${KYUTAI_STT_UV_CACHE_DIR:-/tmp/vokra-kyutai-stt-uv-cache}"

log() { printf '[kyutai-stt-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }
usage() { echo 'usage: run-kyutai-stt-2-6b-en-inspection.sh [--work-dir DIR] | --self-test'; }

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token
  for token in \
    'kyutai/stt-2.6b-en' 'a07aec56d22be5589cd0bc8709c75b6cf3e3039d' \
    'delayed-streams-modeling.git' '4c4f65e147df056adf3346290d64c7b9649b18c9' \
    'moshi.git' 'e6a55d2722a65870ef52a6c9f6ecfc0e90f38362' \
    'mimi-pytorch-e351c8d8@125.safetensors' 'model.safetensors' 'tokenizer_spm_4k_en.model' \
    'tokenizer_en_audio_4000.model' 'b79ea52a30329887a2d0ce2dd5473a63fc5083e441e7986f64f01050c06239c9' \
    '6a93b7d998b32cb65f07e8948508004421042f100130c3572de13af5cab9e4f9' \
    'c8f5779f1471f34734aafe1999082ca33862bc5e' 'd25302da6650309c094d0cbf10cfecfb507c31408b820304bda0c3195482f990' \
    '5618985925' 'model_info' 'list_repo_tree' 'path_in_repo' 'git_blob_sha1' 'lfs_sha256' \
    '128' '32' 'CARGO_BUILD_JOBS=1' 'status": "BLOCKED"' \
    'evidence_stage' 'AUTHENTICATED_EVIDENCE_COMPLETE' 'INSPECTION_ERROR' 'NO_UPLOAD'; do
    if ! grep -Fq -- "$token" "$path"; then
      log "self-test FAIL: missing contract token: $token"; fail=1
    fi
  done
  if ! UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$path" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text(encoding="utf-8")
calls = re.findall(r"list_repo_tree\([^\n]*\)", source)
if not calls:
    raise SystemExit("Kyutai STT tree walk call missing")
for call in calls:
    if "path_in_repo=" not in call or re.search(r"(?<![A-Za-z0-9_])path=", call):
        raise SystemExit(f"Kyutai STT tree walk has incompatible path keyword: {call}")
PY
  then
    log 'self-test FAIL: frozen HfApi.list_repo_tree path_in_repo contract regression'
    fail=1
  fi
  if ! UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - <<'PY'
from huggingface_hub import RepoFile, RepoFolder

def classify_entry(entry):
    if isinstance(entry, RepoFolder):
        if getattr(entry, "type", None) not in {None, "directory"}:
            raise RuntimeError("unknown RepoFolder type")
        return "directory"
    if isinstance(entry, RepoFile):
        if getattr(entry, "type", None) not in {None, "file"}:
            raise RuntimeError("unknown RepoFile type")
        return "file"
    raise RuntimeError(f"unknown HF tree entry: {entry!r}")

file_entry = RepoFile(path="README.md", size=1, oid="a" * 40)
file_entry.type = None
assert classify_entry(file_entry) == "file"
folder_entry = RepoFolder(path="nested", oid="b" * 40)
folder_entry.type = None
assert classify_entry(folder_entry) == "directory"
try:
    classify_entry(object())
except RuntimeError:
    pass
else:
    raise AssertionError("unknown HF tree entry was accepted")
print("Kyutai STT RepoFile/RepoFolder self-test: PASS")
PY
  then
    log 'self-test FAIL: RepoFile/RepoFolder class-identity regression'
    fail=1
  fi
  if grep -En '^[[:space:]]*git[[:space:]]+push|^[[:space:]]*(curl|wget)[^#]*(upload|push)' "$path" >/dev/null; then
    log 'self-test FAIL: publication command found'; fail=1
  fi
  if grep -Eq '^(HF_REVISION|SOURCE_REVISION|MOSHI_REVISION)=.*\$\{' "$path"; then
    log 'self-test FAIL: fixed identity is operator-overridable'; fail=1
  fi
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 \
    python "$INSPECTOR" --self-test >/dev/null || fail=1
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

work_dir="$WORK"
self=0
while (($#)); do
  case "$1" in
    --self-test) self=1; shift ;;
    --work-dir) (($# >= 2)) || die '--work-dir requires DIR'; work_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  [[ "$work_dir" == "$WORK" ]] || die '--self-test accepts no other arguments'
  self_test; exit $?
fi

[[ "$(uname -s)" == Linux ]] || die 'inspection requires Linux VAST'
[[ "$(uname -m)" == x86_64 ]] || die 'inspection requires x86_64 VAST'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$ROOT/Cargo.toml" && -d "$ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
[[ -f "$ROOT/tools/parity/pyproject.toml" && -f "$ROOT/tools/parity/uv.lock" ]] || die 'locked parity project missing'
[[ -f "$INSPECTOR" ]] || die 'inspector missing'
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'invalid memory value'
(( mem_kib >= MIN_MEM_KIB )) || die '128 GiB memory guard failed'
mkdir -p "$(dirname "$work_dir")"
[[ ! -e "$work_dir" || -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work directory must be empty'
mount_kind="$(findmnt -T "$(dirname "$work_dir")" -no FSTYPE 2>/dev/null || true)"
[[ "$mount_kind" == tmpfs ]] || die 'work directory parent must be tmpfs'
tmpfs_kib="$(df -Pk "$(dirname "$work_dir")" | awk 'NR == 2 {print $4}')"
[[ "$tmpfs_kib" =~ ^[0-9]+$ ]] || die 'invalid tmpfs free-space value'
(( tmpfs_kib >= MIN_TMPFS_KIB )) || die '32 GiB tmpfs guard failed'
for tool in cargo git uv awk find df findmnt; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done

mkdir -p "$work_dir/model" "$work_dir/source" "$work_dir/evidence"
work_dir="$(cd "$work_dir" && pwd)"
export CARGO_BUILD_JOBS=1
export UV_CACHE_DIR
{
  echo 'status=BLOCKED'
  echo 'evidence_stage=INSPECTION_ONLY'
  echo 'runtime_status=NOT_IMPLEMENTED_FAIL_CLOSED'
  echo 'cpu_status=UNSUPPORTED'
  echo 'metal_status=BLOCKED_BY_CPU'
  echo 'parity_status=NOT_RUN'
  echo 'publication=NO_UPLOAD'
  echo "hf_total_bytes=$TOTAL_BYTES"
  cargo fmt --all -- --check
  cargo metadata --locked --no-deps --format-version 1 >/dev/null
} > "$work_dir/evidence/validation.log" 2>&1

emit_tree() {
  local output="$1"
  # shellcheck disable=SC2129
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 \
    python - "$output" "$REPOSITORY" "$HF_REVISION" <<'PY' >> "$work_dir/evidence/validation.log" 2>&1
import json, sys
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, RepoFolder

output, repository, revision = sys.argv[1:]
api = HfApi()
info = api.model_info(repository, revision=revision)
if info.sha != revision:
    raise RuntimeError(f"resolved revision {info.sha!r} != {revision!r}")
rows = []
pending, visited = [""], set()
while pending:
    path = pending.pop()
    if path in visited:
        continue
    visited.add(path)
    for item in api.list_repo_tree(repository, revision=revision, path_in_repo=path, recursive=False):
        if isinstance(item, RepoFolder):
            if getattr(item, "type", None) not in {None, "directory"}:
                raise RuntimeError(f"invalid RepoFolder type: {item!r}")
            item_type = "directory"
        elif isinstance(item, RepoFile):
            if getattr(item, "type", None) not in {None, "file"}:
                raise RuntimeError(f"invalid RepoFile type: {item!r}")
            item_type = "file"
        else:
            raise RuntimeError(f"unknown HF tree entry type: {type(item).__name__}")
        item_path = getattr(item, "path", None)
        if not isinstance(item_path, str):
            raise RuntimeError(f"invalid HF tree entry path: {item!r}")
        if item_type == "directory":
            pending.append(item_path)
            continue
        lfs = getattr(item, "lfs", None)
        lfs_sha256 = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        git_blob_sha1 = getattr(item, "blob_id", None) or getattr(item, "oid", None)
        size = getattr(item, "size", None)
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise RuntimeError(f"invalid HF file size: {item_path}")
        if not isinstance(git_blob_sha1, str) or len(git_blob_sha1) != 40:
            raise RuntimeError(f"invalid Git blob identity: {item_path}")
        if lfs_sha256 is not None and (not isinstance(lfs_sha256, str) or len(lfs_sha256) != 64):
            raise RuntimeError(f"invalid LFS identity: {item_path}")
        rows.append({"path": item_path, "type": "file", "size": size,
                     "git_blob_sha1": git_blob_sha1, "lfs_sha256": lfs_sha256})
paths = [row["path"] for row in rows]
if len(paths) != len(set(paths)):
    raise RuntimeError("duplicate HF tree path")
Path(output).write_text(json.dumps({"repository": repository, "revision": revision,
    "resolved_revision": info.sha, "files": sorted(rows, key=lambda row: row["path"])},
    indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

emit_tree "$work_dir/server_tree.json"
# shellcheck disable=SC2129
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 \
  python - "$REPOSITORY" "$HF_REVISION" "$work_dir/model" <<'PY' >> "$work_dir/evidence/validation.log" 2>&1
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3],
                        allow_patterns=[".gitattributes", "README.md", "config.json",
                                        "mimi-pytorch-e351c8d8@125.safetensors", "model.safetensors",
                                        "tokenizer_spm_4k_en.model"]))
PY
git clone --filter=blob:none --no-checkout "$SOURCE_URL" "$work_dir/source/delayed-streams-modeling" >> "$work_dir/evidence/validation.log" 2>&1
git -C "$work_dir/source/delayed-streams-modeling" checkout --detach "$SOURCE_REVISION" >> "$work_dir/evidence/validation.log" 2>&1
git clone --filter=blob:none --no-checkout "$MOSHI_URL" "$work_dir/source/moshi" >> "$work_dir/evidence/validation.log" 2>&1
git -C "$work_dir/source/moshi" checkout --detach "$MOSHI_REVISION" >> "$work_dir/evidence/validation.log" 2>&1

set +e
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 \
  python "$INSPECTOR" --snapshot "$work_dir/model" --server-tree "$work_dir/server_tree.json" \
  --source "$work_dir/source/delayed-streams-modeling" --moshi-source "$work_dir/source/moshi" \
  --evidence "$work_dir/evidence" >> "$work_dir/evidence/validation.log" 2>&1
inspect_rc=$?
set -e
[[ "$inspect_rc" == 2 ]] || die "inspector must exit 2, got $inspect_rc"
[[ -s "$work_dir/evidence/manifest.json" ]] || die 'manifest missing'
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$work_dir/evidence/manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
required = {
    "status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY",
    "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED",
    "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD",
}
if any(m.get(k) != v for k, v in required.items()):
    raise SystemExit("manifest fail-closed status mismatch")
if m.get("inspection_status") != "AUTHENTICATED_EVIDENCE_COMPLETE":
    raise SystemExit("inspection evidence incomplete or errored")
if m.get("inspection_status") == "INSPECTION_ERROR":
    raise SystemExit("inspection error was incorrectly accepted")
PY
die 'Kyutai STT inspection evidence preserved; native runtime/parity remain blocked'
