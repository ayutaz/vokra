#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HF_REPOSITORY="espnet/owsm_v4_medium_1B"; HF_REVISION="e10985c8f1d592e905c24d2ac2b2c53e3feb24dc"
SOURCE_URL="https://github.com/espnet/espnet.git"; SOURCE_REVISION="cccc29023d43a3f504e28df7d1324bb4eb6daedd"
INSPECTOR="$ROOT/tools/parity/owsm_v4_medium_1b_inspect.py"; PREPARER="$ROOT/tools/parity/owsm_v4_medium_1b_prepare_checkpoint.py"; CHECKPOINT_RELATIVE="exp/s2t_train_conv2d8_size1024_e18_d18_mel128_raw_bpe50000/valid.total_count.ave_5best.pth"; MIN_MEM_KIB=$((128*1024*1024)); MIN_DISK_KIB=$((32*1024*1024))
APPROVAL_SCHEMA="owsm-v4-medium-1b-vast-approval-v1"
WRITER_STATUS="MISSING_OWSM_GGUF_WRITER_CONTRACT"; NATIVE_STATUS="NOT_IMPLEMENTED_FAIL_CLOSED"
LICENSE_STATUS="BLOCKED_SOURCE_DEPENDENCY_DATASET_PROVENANCE"; DEPENDENCY_STATUS="UNREVIEWED_BLOCKER"; DATASET_STATUS="UNAUTHENTICATED_BLOCKER"
APPROVAL_SCOPE_JSON='{"checkpoint":"exp/s2t_train_conv2d8_size1024_e18_d18_mel128_raw_bpe50000/valid.total_count.ave_5best.pth","dataset_status":"UNAUTHENTICATED_BLOCKER","dependency_status":"UNREVIEWED_BLOCKER","license_status":"BLOCKED_SOURCE_DEPENDENCY_DATASET_PROVENANCE","model_repository":"espnet/owsm_v4_medium_1B","model_revision":"e10985c8f1d592e905c24d2ac2b2c53e3feb24dc","native_status":"NOT_IMPLEMENTED_FAIL_CLOSED","source_revision":"cccc29023d43a3f504e28df7d1324bb4eb6daedd","source_url":"https://github.com/espnet/espnet.git","writer_status":"MISSING_OWSM_GGUF_WRITER_CONTRACT"}'
die(){ echo "owsm-v4-vast: ERROR: $*" >&2; exit 2; }
sha256_file(){ sha256sum "$1" | awk '{print $1}'; }
require_clean_expected_head(){
  local expected="$1" actual
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || die 'expected HEAD must be exactly 40 lowercase hex characters'
  [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean before OWSM acquisition'
  actual="$(git -C "$ROOT" rev-parse --verify HEAD)"; [[ "$actual" == "$expected" ]] || die "HEAD mismatch: expected $expected, observed $actual"
}
require_regular_approval_path(){
  local path="$1"
  [[ "$path" != "$ROOT" && "$path" != "$ROOT"/* ]] || die 'approval evidence must be outside the checkout'
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$path" <<'PY'
import os, pathlib, stat, sys
p=pathlib.Path(sys.argv[1])
if not p.is_absolute() or any(part in ("", ".", "..") for part in p.parts[1:]): raise SystemExit("unsafe approval path")
cur=pathlib.Path(p.anchor)
for part in p.parts[1:-1]:
    cur /= part
    if cur.is_symlink(): raise SystemExit("approval path has symlinked ancestor")
st=os.lstat(p)
if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode): raise SystemExit("approval must be regular non-symlink")
PY
}
require_approval_binding(){
  local path="$1" expected_sha="$2" actual
  [[ "$expected_sha" =~ ^[0-9a-f]{64}$ ]] || die 'approval SHA-256 must be exactly 64 lowercase hex characters'
  require_regular_approval_path "$path" || die 'approval evidence path is unsafe or not a regular file'
  actual="$(sha256_file "$path")"; [[ "$actual" == "$expected_sha" ]] || die 'approval evidence SHA-256 mismatch'
}
require_blocked_approval(){
  local path="$1" expected_head="$2"
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$path" "$expected_head" "$APPROVAL_SCOPE_JSON" "$APPROVAL_SCHEMA" "$HF_REPOSITORY" "$HF_REVISION" "$SOURCE_URL" "$SOURCE_REVISION" "$CHECKPOINT_RELATIVE" "$WRITER_STATUS" "$NATIVE_STATUS" "$LICENSE_STATUS" "$DEPENDENCY_STATUS" "$DATASET_STATUS" <<'PY'
import hashlib, json, pathlib, re, sys
p, expected_head, scope, schema, model_repo, model_rev, source_url, source_rev, checkpoint, writer, native, license_status, dependency_status, dataset_status = sys.argv[1:]
def pairs(items):
    out={}
    for key,value in items:
        if key in out: raise ValueError(f"duplicate key: {key}")
        out[key]=value
    return out
with open(p, encoding="utf-8") as handle: value=json.load(handle, object_pairs_hook=pairs)
keys={"schema","decision","status","evidence_stage","no_upload","expected_head","source_url","source_revision","model_repository","model_revision","checkpoint","writer_status","native_status","license_status","dependency_status","dataset_status","scope_sha256"}
if set(value)!=keys: raise ValueError("approval key set mismatch")
scope_value=json.loads(scope, object_pairs_hook=pairs)
canonical_scope=json.dumps(scope_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))+"\n"
expected={"schema":schema,"decision":"BLOCKED","status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","no_upload":True,"expected_head":expected_head,"source_url":source_url,"source_revision":source_rev,"model_repository":model_repo,"model_revision":model_rev,"checkpoint":checkpoint,"writer_status":writer,"native_status":native,"license_status":license_status,"dependency_status":dependency_status,"dataset_status":dataset_status,"scope_sha256":hashlib.sha256(canonical_scope.encode()).hexdigest()}
if value != expected: raise ValueError("approval identity or disposition mismatch")
if not re.fullmatch(r"[0-9a-f]{40}", value["expected_head"]): raise ValueError("approval expected_head is not lowercase 40-hex")
print("OWSM_APPROVAL_VALID_BUT_BLOCKED: BLOCKED/INSPECTION_ONLY/NO_UPLOAD")
PY
  then return 0; else return 1; fi
}
require_absent_directory(){
  local path="$1"
  [[ "$path" != "$ROOT" && "$path" != "$ROOT"/* && "$ROOT" != "$path"/* ]] || return 1
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$path" <<'PY'
import os, pathlib, stat, sys
p=pathlib.Path(sys.argv[1])
if not p.is_absolute() or any(part in ("", ".", "..") for part in p.parts[1:]): raise SystemExit("unsafe directory path")
cur=pathlib.Path(p.anchor)
for part in p.parts[1:-1]:
    cur /= part
    if cur.is_symlink() or not cur.is_dir(): raise SystemExit("directory has unsafe/missing ancestor")
if os.path.lexists(p): raise SystemExit("directory target must be absent")
parent=p.parent
if parent.is_symlink() or not parent.is_dir(): raise SystemExit("directory parent must be regular")
PY
}
claim_absent_directory(){ local path="$1"; require_absent_directory "$path" || return 1; mkdir "$path"; }
self_test(){ local path="${BASH_SOURCE[0]}" token fail=0; for token in "$HF_REPOSITORY" "$HF_REVISION" "$SOURCE_URL" "$SOURCE_REVISION" 'allow_pickle=False' 'weights_only=True' 'canonical_payload_sha256' 'source_name_set_sha256' 'next_vast_command' 'MISSING_OWSM_GGUF_WRITER_CONTRACT' 'structural-manifest' 'resolved revision mismatch' 'selected materialized files mismatch' 'inspection manifest' 'payload manifest' 'SOURCE_SEMANTICS_AUTHENTICATED' 'INSPECTION_ONLY' 'INSPECTION_ERROR' 'AUTHENTICATED_EVIDENCE_COMPLETE' 'NO_UPLOAD' 'exit 2' 'CARGO_BUILD_JOBS=1' 'materialized_files' 'findmnt' 'README.md' 'cc-by-4.0' 'espnet/yodas_owsmv4' 'RepoFile' 'RepoFolder' 'classify_entry' 'OWSM_HF_TREE_SELF_TEST' 'expected HEAD' 'approval-evidence' 'BLOCKED_APPROVAL'; do if ! grep -Fq -- "$token" "$path" && ! grep -Fq -- "$token" "$INSPECTOR" && ! grep -Fq -- "$token" "$PREPARER"; then echo "missing contract $token" >&2; fail=1; fi; done; if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push|--upload' "$path" | grep -v 'grep -En' >/dev/null; then fail=1; fi; UV_NO_CACHE=1 uv run --no-sync --frozen --offline --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --self-test || fail=1; UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PREPARER" --self-test || fail=1; if ! OWSM_HF_TREE_SELF_TEST=1 UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - <<'PY'
try:
    from huggingface_hub import RepoFile, RepoFolder
except ModuleNotFoundError:
    class RepoFile:
        def __init__(self, path, size, oid): self.path, self.size, self.oid = path, size, oid
    class RepoFolder:
        def __init__(self, path, oid): self.path, self.oid = path, oid

def classify_entry(entry):
    if isinstance(entry, RepoFolder):
        return "directory"
    if isinstance(entry, RepoFile):
        return "file"
    raise RuntimeError(f"unknown HF tree entry: {entry!r}")

file_entry = RepoFile(path="README.md", size=1, oid="a" * 40)
file_entry.type = None
assert classify_entry(file_entry) == "file"
assert classify_entry(RepoFolder(path="nested", oid="b" * 40)) == "directory"
try:
    classify_entry(object())
except RuntimeError:
    pass
else:
    raise AssertionError("unknown HF tree entry was accepted")
print("OWSM_HF_TREE_SELF_TEST: PASS")
PY
  then echo 'OWSM RepoFile/RepoFolder self-test failed' >&2; fail=1; fi
  local tmp approval valid_sha expected_head
  tmp="$(cd -P "$(mktemp -d)" && pwd)"; approval="$tmp/approval.json"; expected_head="$(git -C "$ROOT" rev-parse HEAD)"
  printf '{}' >"$approval"; if require_blocked_approval "$approval" "$expected_head" 2>/dev/null; then fail=1; fi
  printf '%s\n' "$(cat <<JSON
{"schema":"$APPROVAL_SCHEMA","decision":"BLOCKED","status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","no_upload":true,"expected_head":"$expected_head","source_url":"$SOURCE_URL","source_revision":"$SOURCE_REVISION","model_repository":"$HF_REPOSITORY","model_revision":"$HF_REVISION","checkpoint":"$CHECKPOINT_RELATIVE","writer_status":"$WRITER_STATUS","native_status":"$NATIVE_STATUS","license_status":"$LICENSE_STATUS","dependency_status":"$DEPENDENCY_STATUS","dataset_status":"$DATASET_STATUS","scope_sha256":"$(printf '%s\n' "$APPROVAL_SCOPE_JSON" | sha256sum | awk '{print $1}')"}
JSON
)" >"$approval"; valid_sha="$(sha256_file "$approval")"; if ! require_approval_binding "$approval" "$valid_sha"; then fail=1; fi; if ! require_blocked_approval "$approval" "$expected_head" >/dev/null; then fail=1; fi
  cp "$approval" "$tmp/wrong-head.json"; sed -i.bak 's/"expected_head":"[0-9a-f][0-9a-f]*/"expected_head":"0000000000000000000000000000000000000000/' "$tmp/wrong-head.json"; if require_blocked_approval "$tmp/wrong-head.json" "$expected_head" 2>/dev/null; then fail=1; fi
  cp "$approval" "$tmp/wrong-scope.json"; sed -i.bak 's/"scope_sha256":"[0-9a-f][0-9a-f]*/"scope_sha256":"0000000000000000000000000000000000000000000000000000000000000000/' "$tmp/wrong-scope.json"; if require_blocked_approval "$tmp/wrong-scope.json" "$expected_head" 2>/dev/null; then fail=1; fi
  printf '%s\n' '{"schema":"x","schema":"y"}' >"$tmp/duplicate.json"; if require_blocked_approval "$tmp/duplicate.json" "$expected_head" 2>/dev/null; then fail=1; fi
  if ! grep -Fq 'OWSM_APPROVAL_VALID_BUT_BLOCKED' "$path"; then fail=1; fi
  mkdir "$tmp/parent"; if ! claim_absent_directory "$tmp/parent/claim"; then fail=1; fi; if claim_absent_directory "$tmp/parent/claim" 2>/dev/null; then fail=1; fi
  ln -s "$tmp/parent" "$tmp/link"; if require_absent_directory "$tmp/link/escape" 2>/dev/null; then fail=1; fi
  rm -rf "$tmp"
  ((fail==0)) || return 1; echo 'run-owsm-v4-medium-1b-inspection.sh self-test: OK'; }
if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no other arguments'; self_test; exit $?; fi
approval_path=''; approval_sha=''; expected_head=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --approval-evidence) [[ -z "$approval_path" && $# -ge 2 ]] || die 'duplicate or missing --approval-evidence'; approval_path="$2"; shift 2 ;;
    --approval-sha256) [[ -z "$approval_sha" && $# -ge 2 ]] || die 'duplicate or missing --approval-sha256'; approval_sha="$2"; shift 2 ;;
    --expected-head) [[ -z "$expected_head" && $# -ge 2 ]] || die 'duplicate or missing --expected-head'; expected_head="$2"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "$approval_path" && -n "$approval_sha" && -n "$expected_head" ]] || die 'required: --approval-evidence FILE --approval-sha256 SHA256 --expected-head HEX40'
require_clean_expected_head "$expected_head"
require_approval_binding "$approval_path" "$approval_sha"
require_blocked_approval "$approval_path" "$expected_head" || die 'approval schema/identity/disposition validation failed'
die 'BLOCKED_APPROVAL/INSPECTION_ONLY: OWSM writer/native contract is not authenticated; acquisition and upload remain disabled'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'; [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'; [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
mem_kib="$(awk '$1=="MemTotal:"{print $2;exit}' /proc/meminfo)"; [[ "$mem_kib" =~ ^[0-9]+$ && $mem_kib -ge $MIN_MEM_KIB ]] || die '128 GiB memory guard failed'; [[ "$(findmnt -T /dev/shm -n -o FSTYPE 2>/dev/null)" == tmpfs ]] || die '/dev/shm must be tmpfs'; free_kib="$(df -Pk /dev/shm | awk 'NR==2{print $4}')"; [[ "$free_kib" =~ ^[0-9]+$ && $free_kib -ge $MIN_DISK_KIB ]] || die 'tmpfs disk guard failed'; for command in cargo git uv awk find df findmnt; do command -v "$command" >/dev/null || die "missing tool: $command"; done
 work=/dev/shm/vokra-owsm-v4-medium-1b; require_absent_directory "$work"; claim_absent_directory "$work" || die 'inspection directory claim raced or was redirected'; mkdir "$work/model" "$work/source" "$work/evidence"; export CARGO_BUILD_JOBS=1; export UV_CACHE_DIR="${OWSM_UV_CACHE_DIR:-/tmp/vokra-owsm-uv-cache}"
# shellcheck disable=SC2129 # validation output is one stream
 { cargo fmt --all -- --check; cargo metadata --locked --no-deps --format-version 1 >/dev/null; } >"$work/evidence/validation.log" 2>&1
# shellcheck disable=SC2129 # heredoc output is one validation stream
set +e
 uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$HF_REPOSITORY" "$HF_REVISION" "$work/tree.json" <<'PY' >>"$work/evidence/validation.log" 2>&1
import json,re,sys
from pathlib import Path
from huggingface_hub import HfApi,RepoFile,RepoFolder
repo,rev,out=sys.argv[1:]
def require(condition,label,**details):
 if not condition: raise RuntimeError(json.dumps({"gate":"hf_tree","failure":label,**details},sort_keys=True))
api=HfApi(); info=api.model_info(repo,revision=rev)
require(info.sha==rev,"resolved revision mismatch",expected=rev,observed=info.sha)
rows=[]
for x in api.list_repo_tree(repo,revision=rev,recursive=True):
 if isinstance(x,RepoFolder): continue
 if not isinstance(x,RepoFile): raise RuntimeError(f"unknown HF tree entry: {x!r}")
 path=getattr(x,"path",None); size=getattr(x,"size",None); blob=getattr(x,"blob_id",None) or getattr(x,"oid",None)
 if not isinstance(path,str) or not path or "\\" in path or "\x00" in path or Path(path).is_absolute() or ".." in Path(path).parts: raise RuntimeError(f"unsafe HF tree path: {path!r}")
 if not isinstance(size,int) or isinstance(size,bool) or size < 0: raise RuntimeError(f"invalid HF tree size: {path!r}")
 if not isinstance(blob,str) or not re.fullmatch(r"[0-9a-f]{40}",blob): raise RuntimeError(f"invalid HF tree Git blob: {path!r}")
 lfs=getattr(x,"lfs",None); lfs_sha=getattr(lfs,"sha256",None) if lfs is not None else None
 if isinstance(lfs,dict): lfs_sha=lfs.get("sha256")
 if lfs_sha is not None and (not isinstance(lfs_sha,str) or not re.fullmatch(r"[0-9a-f]{64}",lfs_sha)): raise RuntimeError(f"invalid HF tree LFS SHA256: {path!r}")
 rows.append({"path":path,"type":"file","size":size,"git_blob_sha1":blob,"lfs_sha256":lfs_sha})
for row in rows:
 require(isinstance(row["path"],str) and row["path"] and "\\" not in row["path"] and ".." not in Path(row["path"]).parts and not row["path"].startswith("/"),"unsafe file path",path=repr(row["path"]))
 require(isinstance(row["size"],int) and not isinstance(row["size"],bool) and row["size"] >= 0,"invalid file size",path=row["path"],size=repr(row["size"]))
 require(re.fullmatch(r"[0-9a-f]{40}",str(row["git_blob_sha1"])) is not None,"invalid Git blob SHA1",path=row["path"],observed=repr(row["git_blob_sha1"]))
 require(row["lfs_sha256"] is None or re.fullmatch(r"[0-9a-f]{64}",str(row["lfs_sha256"])) is not None,"invalid LFS SHA256",path=row["path"],observed=repr(row["lfs_sha256"]))
require(len({row["path"] for row in rows})==len(rows),"duplicate file path",count=len(rows),unique_count=len({row["path"] for row in rows}))
selected={"exp/s2t_train_conv2d8_size1024_e18_d18_mel128_raw_bpe50000/valid.total_count.ave_5best.pth","exp/s2t_train_conv2d8_size1024_e18_d18_mel128_raw_bpe50000/config.yaml","exp/s2t_stats_raw_bpe50000/train/feats_stats.npz","data/token_list/bpe_unigram50000/bpe.model","README.md"}
materialized=[row for row in rows if row["path"] in selected]
require({row["path"] for row in materialized}==selected,"selected materialized files mismatch",expected=sorted(selected),observed=sorted(row["path"] for row in materialized))
Path(out).write_text(json.dumps({"repository":repo,"revision":rev,"resolved_revision":info.sha,"walk":"recursive_file_only","files":rows,"materialized_files":materialized,"materialized_scope":"selected_runtime_inputs"},sort_keys=True,indent=2)+"\n")
PY
tree_rc=$?
set -e
[[ "$tree_rc" == 0 ]] || die "HF tree evidence failed (exit $tree_rc; see $work/evidence/validation.log)"
# shellcheck disable=SC2129 # validation output is one stream
 uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$HF_REPOSITORY" "$HF_REVISION" "$work/model" <<'PY' >>"$work/evidence/validation.log" 2>&1
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3],allow_patterns=["exp/s2t_train_conv2d8_size1024_e18_d18_mel128_raw_bpe50000/valid.total_count.ave_5best.pth","exp/s2t_train_conv2d8_size1024_e18_d18_mel128_raw_bpe50000/config.yaml","exp/s2t_stats_raw_bpe50000/train/feats_stats.npz","data/token_list/bpe_unigram50000/bpe.model","README.md"])
PY
git clone --filter=blob:none "$SOURCE_URL" "$work/source/repo" >>"$work/evidence/validation.log" 2>&1; git -C "$work/source/repo" checkout --detach "$SOURCE_REVISION" >>"$work/evidence/validation.log" 2>&1; [[ "$(git -C "$work/source/repo" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || die 'source revision mismatch'; [[ "$(git -C "$work/source/repo" remote get-url origin)" == "$SOURCE_URL" ]] || die 'source origin mismatch'
set +e; uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --snapshot "$work/model" --source "$work/source/repo" --server-tree "$work/tree.json" --output "$work/evidence" >>"$work/evidence/validation.log" 2>&1; rc=$?; set -e; [[ "$rc" == 2 ]] || die 'inspector must exit 2'; uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$work/evidence/manifest.json" <<'PY'
import json,sys
p=json.loads(open(sys.argv[1]).read())
expected={"status":"BLOCKED","inspection_status":"AUTHENTICATED_EVIDENCE_COMPLETE","evidence_stage":"INSPECTION_ONLY","runtime_status":"NOT_IMPLEMENTED_FAIL_CLOSED","publication":"NO_UPLOAD"}
observed={key:p.get(key) for key in expected}
if observed != expected:
 raise RuntimeError(json.dumps({"gate":"inspection manifest","failure":"contract mismatch","expected":expected,"observed":observed},sort_keys=True))
PY
set +e; uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$PREPARER" --checkpoint "$work/model/$CHECKPOINT_RELATIVE" --structural-manifest "$work/evidence/manifest.json" --output "$work/evidence/payload-manifest.json" >>"$work/evidence/validation.log" 2>&1; payload_rc=$?; set -e; [[ "$payload_rc" == 2 ]] || die 'payload preparer must remain blocked by the missing GGUF writer contract'; uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - "$work/evidence/payload-manifest.json" <<'PY'
import json,sys
p=json.loads(open(sys.argv[1],encoding="utf-8").read())
expected={"status":"BLOCKED_WRITER_CONTRACT","completed_evidence":False,"blocked_evidence":True,"writer_status":"MISSING_OWSM_GGUF_WRITER_CONTRACT","publication":"NO_UPLOAD","tensor_count":1172,"tensor_rows":1172}
observed={"status":p.get("status"),"completed_evidence":p.get("completed_evidence"),"blocked_evidence":p.get("blocked_evidence"),"writer_status":(p.get("writer_contract") or {}).get("status"),"publication":(p.get("writer_contract") or {}).get("publication"),"tensor_count":p.get("tensor_count"),"tensor_rows":len(p.get("tensors",[])) if isinstance(p.get("tensors"),list) else None}
if observed != expected:
 raise RuntimeError(json.dumps({"gate":"payload manifest","failure":"contract mismatch","expected":expected,"observed":observed},sort_keys=True))
PY
exit 2
