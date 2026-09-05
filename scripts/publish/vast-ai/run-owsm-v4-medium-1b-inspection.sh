#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HF_REPOSITORY="espnet/owsm_v4_medium_1B"; HF_REVISION="e10985c8f1d592e905c24d2ac2b2c53e3feb24dc"
SOURCE_URL="https://github.com/espnet/espnet.git"; SOURCE_REVISION="cccc29023d43a3f504e28df7d1324bb4eb6daedd"
INSPECTOR="$ROOT/tools/parity/owsm_v4_medium_1b_inspect.py"; PREPARER="$ROOT/tools/parity/owsm_v4_medium_1b_prepare_checkpoint.py"; CHECKPOINT_RELATIVE="exp/s2t_train_conv2d8_size1024_e18_d18_mel128_raw_bpe50000/valid.total_count.ave_5best.pth"; MIN_MEM_KIB=$((128*1024*1024)); MIN_DISK_KIB=$((32*1024*1024))
die(){ echo "owsm-v4-vast: ERROR: $*" >&2; exit 2; }
self_test(){ local path="${BASH_SOURCE[0]}" token fail=0; for token in "$HF_REPOSITORY" "$HF_REVISION" "$SOURCE_URL" "$SOURCE_REVISION" 'allow_pickle=False' 'weights_only=True' 'canonical_payload_sha256' 'MISSING_OWSM_GGUF_WRITER_CONTRACT' 'structural-manifest' 'resolved revision mismatch' 'selected materialized files mismatch' 'inspection manifest' 'payload manifest' 'INSPECTION_ONLY' 'INSPECTION_ERROR' 'AUTHENTICATED_EVIDENCE_COMPLETE' 'NO_UPLOAD' 'exit 2' 'CARGO_BUILD_JOBS=1' 'materialized_files' 'findmnt' 'README.md' 'cc-by-4.0' 'espnet/yodas_owsmv4' 'RepoFile' 'RepoFolder' 'classify_entry' 'OWSM_HF_TREE_SELF_TEST'; do if ! grep -Fq -- "$token" "$path" && ! grep -Fq -- "$token" "$INSPECTOR" && ! grep -Fq -- "$token" "$PREPARER"; then echo "missing contract $token" >&2; fail=1; fi; done; if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push|--upload' "$path" | grep -v 'grep -En' >/dev/null; then fail=1; fi; UV_CACHE_DIR="${OWSM_UV_CACHE_DIR:-/tmp/vokra-owsm-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$INSPECTOR" --self-test || fail=1; UV_CACHE_DIR="${OWSM_UV_CACHE_DIR:-/tmp/vokra-owsm-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$PREPARER" --self-test || fail=1; if ! OWSM_HF_TREE_SELF_TEST=1 UV_CACHE_DIR="${OWSM_UV_CACHE_DIR:-/tmp/vokra-owsm-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - <<'PY'
from huggingface_hub import RepoFile, RepoFolder

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
  then echo 'OWSM RepoFile/RepoFolder self-test failed' >&2; fail=1; fi; ((fail==0)) || return 1; echo 'run-owsm-v4-medium-1b-inspection.sh self-test: OK'; }
if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || die '--self-test accepts no other arguments'; self_test; exit $?; fi
[[ $# == 0 ]] || die 'arguments are not accepted; revisions are fixed'; [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'; [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'; [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
mem_kib="$(awk '$1=="MemTotal:"{print $2;exit}' /proc/meminfo)"; [[ "$mem_kib" =~ ^[0-9]+$ && $mem_kib -ge $MIN_MEM_KIB ]] || die '128 GiB memory guard failed'; [[ "$(findmnt -T /dev/shm -n -o FSTYPE 2>/dev/null)" == tmpfs ]] || die '/dev/shm must be tmpfs'; free_kib="$(df -Pk /dev/shm | awk 'NR==2{print $4}')"; [[ "$free_kib" =~ ^[0-9]+$ && $free_kib -ge $MIN_DISK_KIB ]] || die 'tmpfs disk guard failed'; for command in cargo git uv awk find df findmnt; do command -v "$command" >/dev/null || die "missing tool: $command"; done
 work=/dev/shm/vokra-owsm-v4-medium-1b; if [[ -e "$work" ]]; then [[ -z "$(find "$work" -mindepth 1 -print -quit)" ]] || die 'inspection directory must be empty'; else mkdir -p "$work"; fi; mkdir -p "$work/model" "$work/source" "$work/evidence"; export CARGO_BUILD_JOBS=1; export UV_CACHE_DIR="${OWSM_UV_CACHE_DIR:-/tmp/vokra-owsm-uv-cache}"
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
