#!/usr/bin/env bash
# VAST-only FireRedASR-AED-L inspection. No conversion or publication.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
# The source contract is authenticated, while exact checkpoint geometry and
# token dictionary binding remain blocked until the VAST checkpoint review.
INSPECTOR="$ROOT/tools/parity/firered_asr_aed_l_inspect.py"
PREPARER="$ROOT/tools/parity/firered_asr_aed_l_prepare_checkpoint.py"
REPOSITORY="FireRedTeam/FireRedASR-AED-L"
REVISION="e57f5960d03cff1071ff7acbb409314d1e70ed3d"
SOURCE_URL="https://github.com/FireRedTeam/FireRedASR.git"
SOURCE_REVISION="834635e4cf277ed8ca92049fc375b17c3dc20748"
WORK="/dev/shm/vokra-firered-asr-aed-l-inspection"
MIN_MEM_KIB=$((128 * 1024 * 1024))
MIN_DISK_KIB=$((32 * 1024 * 1024))
UV_CACHE_DIR="${FIRERED_ASR_UV_CACHE_DIR:-/tmp/vokra-firered-asr-uv-cache}"

log() { printf '[firered-asr-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }
usage() { echo 'usage: run-firered-asr-aed-l-inspection.sh [--work-dir DIR] | --self-test'; }

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token
  for token in \
    'FireRedTeam/FireRedASR-AED-L' 'e57f5960d03cff1071ff7acbb409314d1e70ed3d' \
    'FireRedASR.git' '834635e4cf277ed8ca92049fc375b17c3dc20748' \
    'model.pth.tar' '12380d0b4b6b83b09306292f3ab7e276bc84e2feeec33ce956b1a488cd4867e3' \
    'train_bpe1000.model' '473bbc157cb4eade2059b30a3c877a1c29bd50cadbfbed869ae36eeade7fee07' \
    'model_info' 'list_repo_tree' 'path_in_repo' 'git_blob_sha1' 'lfs_sha256' 'weights_only=True' \
    '128' '32' '/dev/shm' 'findmnt' 'CARGO_BUILD_JOBS=1' 'status": "BLOCKED"' 'INSPECTION_ONLY' 'NO_UPLOAD' \
    'config.yaml' 'BLOCKER_EMPTY_CONFIG' 'git ls-files' 'git status' \
    'source_contract' 'AUTHENTICATED_SOURCE_CONTRACT' 'SOURCE_FACTS_AUTHENTICATED' \
    'checkpoint geometry' 'token dictionary binding'; do
    if ! grep -Fq -- "$token" "$path"; then log "self-test FAIL: missing token $token"; fail=1; fi
  done
  if ! UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$path" <<'PY'
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text(encoding="utf-8")
calls = re.findall(r"list_repo_tree\([^\n]*\)", source)
if not calls:
    raise SystemExit("FireRedASR tree walk call missing")
for call in calls:
    if "path_in_repo=" not in call or re.search(r"(?<![A-Za-z0-9_])path=", call):
        raise SystemExit(f"FireRedASR tree walk has incompatible path keyword: {call}")
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
print("FireRedASR RepoFile/RepoFolder self-test: PASS")
PY
  then
    log 'self-test FAIL: RepoFile/RepoFolder class-identity regression'
    fail=1
  fi
  if grep -En '^[[:space:]]*git[[:space:]]+push|^[[:space:]]*(curl|wget)[^#]*(upload|push)' "$path" >/dev/null; then
    log 'self-test FAIL: publication command found'; fail=1
  fi
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$PREPARER" --self-test >/dev/null || fail=1
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --self-test >/dev/null || fail=1
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
if (( self )); then [[ "$work_dir" == "$WORK" ]] || die '--self-test accepts no other arguments'; self_test; exit $?; fi
[[ "$(uname -s)" == Linux ]] || die 'Linux VAST required'
[[ "$(uname -m)" == x86_64 ]] || die 'x86_64 VAST required'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$ROOT/Cargo.toml" && -d "$ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
[[ -f "$ROOT/tools/parity/pyproject.toml" && -f "$ROOT/tools/parity/uv.lock" ]] || die 'locked parity project missing'
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'invalid memory value'
(( mem_kib >= MIN_MEM_KIB )) || die '128 GiB memory guard failed'
mkdir -p "$(dirname "$work_dir")"
[[ ! -e "$work_dir" || -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work directory must be empty'
free_kib="$(df -Pk "$(dirname "$work_dir")" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || die 'invalid disk value'
(( free_kib >= MIN_DISK_KIB )) || die '32 GiB disk guard failed'
for tool in cargo git uv awk find df findmnt sha256sum; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done
[[ "$(findmnt -T "$(dirname "$work_dir")" -no FSTYPE)" == tmpfs ]] || die 'parent work filesystem must be tmpfs'
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
  cargo fmt --all -- --check
  cargo metadata --locked --no-deps --format-version 1 >/dev/null
} > "$work_dir/evidence/validation.log" 2>&1

# shellcheck disable=SC2129
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$work_dir/server_tree.json" <<'PY' >> "$work_dir/evidence/validation.log" 2>&1
import json, sys
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, RepoFolder
repo, rev = "FireRedTeam/FireRedASR-AED-L", "e57f5960d03cff1071ff7acbb409314d1e70ed3d"
api = HfApi()
info = api.model_info(repo, revision=rev)
if info.sha != rev: raise RuntimeError(f"resolved revision {info.sha!r} != {rev!r}")
rows=[]; pending=[""]; visited=set()
while pending:
    path=pending.pop()
    if path in visited: continue
    visited.add(path)
    for item in api.list_repo_tree(repo, revision=rev, path_in_repo=path, recursive=False):
        if isinstance(item, RepoFolder):
            if getattr(item,"type",None) not in {None,"directory"}: raise RuntimeError(f"invalid RepoFolder type {item!r}")
            item_type="directory"
        elif isinstance(item, RepoFile):
            if getattr(item,"type",None) not in {None,"file"}: raise RuntimeError(f"invalid RepoFile type {item!r}")
            item_type="file"
        else:
            raise RuntimeError(f"unknown HF tree entry type: {type(item).__name__}")
        item_path=getattr(item,"path",None)
        if not isinstance(item_path,str): raise RuntimeError(f"invalid HF entry path {item!r}")
        if item_type=="directory": pending.append(item_path); continue
        lfs=getattr(item,"lfs",None)
        lfs_sha=lfs.get("sha256") if isinstance(lfs,dict) else getattr(lfs,"sha256",None)
        blob=getattr(item,"blob_id",None) or getattr(item,"oid",None)
        size=getattr(item,"size",None)
        if not isinstance(size,int) or isinstance(size,bool) or size<0 or not isinstance(blob,str): raise RuntimeError(f"invalid identity {item_path}")
        rows.append({"path":item_path,"type":"file","size":size,"git_blob_sha1":blob,"lfs_sha256":lfs_sha})
if len({x["path"] for x in rows}) != len(rows): raise RuntimeError("duplicate server path")
Path(sys.argv[1]).write_text(json.dumps({"repository":repo,"revision":rev,"resolved_revision":info.sha,"files":sorted(rows,key=lambda x:x["path"])},indent=2,sort_keys=True)+"\n")
PY
# shellcheck disable=SC2129
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$REPOSITORY" "$REVISION" "$work_dir/model" <<'PY' >> "$work_dir/evidence/validation.log" 2>&1
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3]))
PY
git clone --filter=blob:none --no-checkout "$SOURCE_URL" "$work_dir/source/repo" >> "$work_dir/evidence/validation.log" 2>&1
git -C "$work_dir/source/repo" checkout --detach "$SOURCE_REVISION" >> "$work_dir/evidence/validation.log" 2>&1
set +e
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --snapshot "$work_dir/model" --server-tree "$work_dir/server_tree.json" --source "$work_dir/source/repo" --evidence "$work_dir/evidence" >> "$work_dir/evidence/validation.log" 2>&1
inspect_rc=$?
set -e
[[ "$inspect_rc" == 2 ]] || die "inspector must exit 2, got $inspect_rc"
[[ -s "$work_dir/evidence/manifest.json" ]] || die 'manifest missing'
grep -Fq '"status": "BLOCKED"' "$work_dir/evidence/manifest.json" || die 'blocked status missing'
grep -Fq '"evidence_stage": "INSPECTION_ONLY"' "$work_dir/evidence/manifest.json" || die 'inspection stage missing'
grep -Fq '"publication": "NO_UPLOAD"' "$work_dir/evidence/manifest.json" || die 'publication status missing'
grep -Fq '"inspection_status": "AUTHENTICATED_EVIDENCE_COMPLETE"' "$work_dir/evidence/manifest.json" || die 'inspection did not complete authenticated evidence'
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$work_dir/evidence/manifest.json" <<'PY'
import json, re, sys
manifest = json.loads(open(sys.argv[1], encoding="utf-8").read())
assert manifest["status"] == "BLOCKED"
assert manifest["evidence_stage"] == "INSPECTION_ONLY"
assert manifest["runtime_status"] == "NOT_IMPLEMENTED_FAIL_CLOSED"
assert manifest["publication"] == "NO_UPLOAD"
assert manifest["inspection_status"] == "AUTHENTICATED_EVIDENCE_COMPLETE"
contract = manifest.get("source_contract")
assert isinstance(contract, dict)
assert contract.get("status") == "AUTHENTICATED_SOURCE_CONTRACT"
expected_paths = [
    "fireredasr/models/fireredasr_aed.py",
    "fireredasr/data/asr_feat.py",
    "fireredasr/tokenizer/aed_tokenizer.py",
    "README.md",
]
records = contract.get("records")
assert isinstance(records, list) and len(records) == len(expected_paths)
assert [record.get("path") for record in records] == expected_paths
for record in records:
    assert set(record) == {"path", "sha256", "markers", "status"}
    assert record["status"] == "SOURCE_FACTS_AUTHENTICATED"
    assert isinstance(record["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
    assert isinstance(record["markers"], list) and record["markers"]
assert "INSPECTION_ERROR" not in json.dumps(manifest)
PY
die 'FireRedASR inspection evidence preserved; conversion/runtime/parity remain blocked'
